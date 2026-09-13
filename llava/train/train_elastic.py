"""Launch elastic / adaptive-matryoshka training.

This is a thin wrapper over M3's existing `llava.train.train.train()`. It does
exactly one extra thing: it sets the module-level `ELASTIC_CONFIG`, which the
attach hook inside `train()` reads to:
  1. build an ElasticEngine and attach it to the model (`attach_elastic_engine`),
     injecting rank-nested LoRA into the (frozen) vision tower if use_lora;
  2. mark ONLY the elastic modules trainable (resampler, projector, LoRA factors)
     so the CLIP/SigLIP backbone and the LLM stay frozen;
  3. let the elastic forward branch run the L_tok grid each step with the
     LM loss + prefix-KL (coarse->fine) + CORAL (latent stability) terms.

Everything else -- data module, tokenizer, DeepSpeed, checkpointing -- is M3's
unchanged machinery, so all the usual LLaVA training flags still apply.

Elastic-specific args (parsed here, stripped before HfArgumentParser sees argv):
  --tok_levels        INT [INT ...]  visual-token budgets per level, descending
                                     e.g. --tok_levels 256 144 64 16
  --lora_ranks        INT [INT ...]  LoRA rank per tok level (must match len)
                                     e.g. --lora_ranks 8 16 32 64
  --prefix_kl_weight  FLOAT          weight on the prefix-KL distillation term (default 1.0)
  --coral_weight      FLOAT          weight on the CORAL token-alignment term (default 0.1)
  --use_kd            BOOL           enable prefix-KL self-distillation (default True)
                                     e.g. --use_kd False  to disable for CE-only training
  --use_coral         BOOL           enable CORAL alignment loss (default True)
                                     e.g. --use_coral False  to disable
  --n_sample_students INT            number of students to sample per step (default 0 = full grid)
                                     1 → teacher + 1 random student (~50% compute saving)
                                     2 → teacher + 2 random students (~25% compute saving)
                                     0 or ≥ n_levels-1 → full grid (no sampling)

All other flags are standard LLaVA training args.

Usage example:
    deepspeed llava/train/train_elastic.py \\
        --tok_levels 256 144 64 16 \\
        --lora_ranks 8 16 32 64 \\
        --model_name_or_path lmsys/vicuna-7b-v1.5 \\
        --version plain \\
        --data_path ./playground/data/LLaVA-Pretrain/blip_laion_cc_sbu_558k.json \\
        --image_folder ./playground/data/LLaVA-Pretrain/images \\
        --vision_tower openai/clip-vit-large-patch14-336 \\
        --mm_projector_type mlp2x_gelu \\
        --freeze_backbone True \\
        --output_dir ./checkpoints/llava-elastic-pretrain \\
        --bf16 True --num_train_epochs 1 \\
        --per_device_train_batch_size 32 --gradient_accumulation_steps 1 \\
        --learning_rate 1e-3 --tf32 True --model_max_length 2048
"""

import argparse
import json
import os
import sys

import llava.train.train as m3train
from llava.model.elastic import ElasticConfig, NestedQueryResampler


def _get_argv_value(key: str) -> str:
    """Scan sys.argv for --key VALUE and return VALUE, or '?' if absent."""
    try:
        idx = sys.argv.index(key)
        return sys.argv[idx + 1] if idx + 1 < len(sys.argv) else "?"
    except ValueError:
        return "?"


def _write_run_manifest(elastic_args, tok_levels, lora_ranks,
                        nest_version, lora_type) -> None:
    """Write run_manifest.json into output_dir: everything needed to reproduce
    this run, in one file, at launch time.

    elastic_config.json already records the elastic architecture, but not the
    things that live in HF TrainingArguments or in git -- optimizer, LR,
    schedule, epochs, batch size, dataset, seed, commit. Reconstructing those
    later means trusting a launcher script that has since been edited, which is
    exactly how v8's provenance ended up recoverable only from its checkpoint
    (see docs/EXPERIMENT_JOURNAL.md 16o). Written once at launch, never
    updated, so it records what the run STARTED with.
    """
    import subprocess, datetime, socket
    out_dir = _get_argv_value("--output_dir")
    if out_dir in (None, "?"):
        return

    def _git(*args, default="unknown"):
        try:
            return subprocess.check_output(["git", *args], cwd=os.path.dirname(
                os.path.abspath(__file__)), stderr=subprocess.DEVNULL).decode().strip()
        except Exception:
            return default

    # A dirty tree means the commit hash alone does not identify the code that
    # ran, so say so rather than recording a hash that is quietly incomplete.
    dirty = _git("status", "--porcelain", default="")
    manifest = {
        "nest_version":     nest_version,
        "lora_type":        lora_type,
        "lora_ranks":       list(lora_ranks),
        "tok_levels":       list(tok_levels),
        "backbone":         _get_argv_value("--model_name_or_path"),
        "vision_tower":     _get_argv_value("--vision_tower"),
        "conv_version":     _get_argv_value("--version"),
        "kd": {
            "kd_type":              "logits (prefix-KL); no hidden/attention/response KD exists here",
            "requested_teacher":    elastic_args.kd_teacher or elastic_args.teacher,
            "kd_student_key":       elastic_args.kd_student_key,
            "use_prefix_kl":        elastic_args.use_kd,
            "prefix_kl_weight":     elastic_args.prefix_kl_weight,
            "teacher":              elastic_args.teacher,
            "teacher_model_path":   (elastic_args.teacher_model_path
                                     if elastic_args.teacher != "self" else None),
            "kl_teacher_tok_level": 0,
            "kl_teacher_budget":    tok_levels[0],
            "distillation_target":  "next-token logits over labelled text positions, "
                                    "right-aligned (prefix_kl_loss)",
            "temperature":          "n/a -- prefix_kl_loss uses plain log_softmax KL, "
                                    "no temperature parameter exists in this codebase",
            "mechanism":            ("SELF-distillation: teacher is THIS MODEL at "
                                     "tok_levels[0]; teacher and student share weights "
                                     "exactly. Measured KL ~0.006."
                                     if elastic_args.teacher == "self" else
                                     "EXTERNAL teacher: frozen separate checkpoint, "
                                     "independent weights."),
        },
        "coral":            {"enabled": elastic_args.use_coral,
                             "weight": elastic_args.coral_weight},
        "decorrelation":    {"enabled": elastic_args.use_token_decorrelation,
                             "weight": elastic_args.decorr_weight},
        "vision_lora":      {"enabled": elastic_args.vision_lora_enable,
                             "specialize_tok": elastic_args.vision_lora_specialize_tok,
                             "ranks": list(lora_ranks)},
        "resampler":        {"arch": elastic_args.resampler_arch,
                             "anchor_mode": elastic_args.anchor_mode,
                             "anchor_ratio": elastic_args.anchor_ratio,
                             "query_selection": elastic_args.query_selection},
        "n_sample_students": elastic_args.n_sample_students,
        "use_nested_dropout": elastic_args.use_nested_dropout,
        "optimizer":        "adamw_torch (HF default)",
        "learning_rate":    _get_argv_value("--learning_rate"),
        "scheduler":        _get_argv_value("--lr_scheduler_type"),
        "warmup_ratio":     _get_argv_value("--warmup_ratio"),
        "weight_decay":     _get_argv_value("--weight_decay"),
        "epochs":           _get_argv_value("--num_train_epochs"),
        "per_device_batch": _get_argv_value("--per_device_train_batch_size"),
        "grad_accum":       _get_argv_value("--gradient_accumulation_steps"),
        "seed":             _get_argv_value("--seed"),   # "?" => HF default 42
        "dataset":          _get_argv_value("--data_path"),
        "image_folder":     _get_argv_value("--image_folder"),
        "image_aspect_ratio": _get_argv_value("--image_aspect_ratio"),
        "model_max_length": _get_argv_value("--model_max_length"),
        "deepspeed":        _get_argv_value("--deepspeed"),
        "pretrain_elastic_path": _get_argv_value("--pretrain_elastic_path"),
        "checkpoint":       out_dir,
        "git_commit":       _git("rev-parse", "HEAD"),
        "git_branch":       _git("rev-parse", "--abbrev-ref", "HEAD"),
        "git_dirty":        bool(dirty),
        "git_dirty_files":  dirty.splitlines() if dirty else [],
        "slurm_job_id":     os.environ.get("SLURM_JOB_ID"),
        "node":             socket.gethostname(),
        "launched_utc":     datetime.datetime.utcnow().isoformat() + "Z",
    }
    try:
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "run_manifest.json"), "w") as f:
            json.dump(manifest, f, indent=2)
        print(f"[elastic] wrote run_manifest.json -> {out_dir}", flush=True)
    except OSError as e:
        print(f"[elastic] could not write run_manifest.json: {e}", flush=True)


def _print_config_banner(elastic_args, tok_levels, lora_ranks) -> None:
    llm = _get_argv_value("--model_name_or_path")
    ve = _get_argv_value("--vision_tower")
    teacher_tok = tok_levels[0]
    n = elastic_args.n_sample_students
    n_students_total = len(tok_levels) - 1
    if 0 < n < n_students_total:
        mode = f"sample_student({n}) — teacher + {n} random student{'s' if n > 1 else ''} per step"
    else:
        mode = "all-levels — full grid every step"
    kd_str = (
        f"enabled  (prefix_kl_weight={elastic_args.prefix_kl_weight})"
        if elastic_args.use_kd
        else "DISABLED"
    )
    coral_str = (
        f"enabled  (coral_weight={elastic_args.coral_weight})"
        if elastic_args.use_coral
        else "DISABLED"
    )
    decorr_str = (
        f"enabled  (decorr_weight={elastic_args.decorr_weight}, query tokens only)"
        if elastic_args.use_token_decorrelation
        else "DISABLED"
    )
    llm_lora_enabled = _get_argv_value("--lora_enable").lower() not in ("false", "0", "no", "?")
    if llm_lora_enabled:
        llm_lora_str = f"enabled  (rank={_get_argv_value('--lora_r')}, alpha={_get_argv_value('--lora_alpha')})"
    else:
        llm_lora_str = "disabled (full fine-tune)"
    if elastic_args.vision_lora_enable:
        vt_lora_str = (
            f"enabled  (lora_type={elastic_args.lora_type or 'explicit ranks'}, "
            f"ranks={lora_ranks}, "
            f"{'specialized per tok_level' if elastic_args.vision_lora_specialize_tok else 'one shared adapter'})"
        )
    else:
        vt_lora_str = "disabled (frozen, unspecialized vision tower)"
    qsel = elastic_args.query_selection
    qsel_str = qsel if qsel == "prefix" else f"{qsel}  (NON-DEFAULT, untested)"
    sep = "=" * 64
    print(
        f"\n{sep}\n"
        f"[FlexLLaVA] Elastic Training — Job Configuration\n"
        f"  Vision Encoder : {ve}\n"
        f"  LLM            : {llm}\n"
        f"  Token budgets  : {tok_levels}  (teacher = tok{teacher_tok})\n"
        f"  NEST version   : {elastic_args.nest_version or '(unset)'}   lora_type: {elastic_args.lora_type or '(explicit ranks)'}\n"
        f"  LoRA ranks     : {lora_ranks}\n"
        f"  Training mode  : {mode}\n"
        f"  LLM LoRA       : {llm_lora_str}\n"
        f"  Vision LoRA    : {vt_lora_str}\n"
        f"  Query select   : {qsel_str}\n"
        f"  KD loss        : {kd_str}\n"
        f"  CORAL loss     : {coral_str}\n"
        f"  Decorr loss    : {decorr_str}\n"
        f"{sep}\n",
        flush=True,
    )


# ---- defaults (used when args are not passed on the CLI) -----------------
_DEFAULT_TOK_LEVELS      = [256, 144, 64, 16]
_DEFAULT_LORA_RANKS      = [8, 16, 32, 64]
_DEFAULT_KL_WEIGHT       = 1.0
_DEFAULT_CORAL_WEIGHT    = 0.01


def _parse_anchor_routing(spec):
    """'256:64,144:36' -> {256: 64, 144: 36}. None/empty -> None (auto-derive)."""
    if not spec:
        return None
    routing = {}
    for pair in spec.split(","):
        pair = pair.strip()
        if not pair:
            continue
        budget, _, n_anchor = pair.partition(":")
        if not n_anchor:
            raise ValueError(f"--anchor_routing entry {pair!r} is not 'BUDGET:NANCHOR'")
        routing[int(budget)] = int(n_anchor)
    return routing or None


def _parse_elastic_args():
    """Pre-parse only the elastic-specific flags, leaving everything else in
    sys.argv for HfArgumentParser inside m3train.train()."""
    p = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    p.add_argument("--tok_levels", type=int, nargs="+", default=_DEFAULT_TOK_LEVELS,
                   help="Visual-token budgets per level, descending (e.g. 256 144 64 16).")
    p.add_argument("--lora_ranks", type=int, nargs="+", default=None,
                   help="LoRA rank per tok level — must have same length as tok_levels. "
                        "Usually left unset: --lora_type derives it. Setting both is an "
                        "error unless they agree, so a run can never silently disagree "
                        "with the lora_type recorded in its own checkpoint.")
    p.add_argument("--lora_type", choices=("v8", "asc"), default=None,
                   help="Which level->rank assignment to use, over the SAME "
                        "NestedLoRALinear (there is no second LoRA implementation). "
                        "'v8'  = rank ascends as budget DESCENDS (256->8 ... 16->64), "
                        "the v4-v8 convention: small budgets get more adapter. "
                        "'asc' = rank ascends WITH budget (256->64 ... 16->8), the v14 "
                        "hypothesis: the teacher level gets the most adapter. "
                        "Ranks are taken from the geometric ladder 8/16/32/64 for a "
                        "4-level grid and generalised as powers of two otherwise.")
    p.add_argument("--kd_teacher", default=None, metavar="KEY",
                   help="Which KD teacher to use when --use_kd is on and an EXTERNAL "
                        "teacher is wanted. 'auto' consults the compatibility registry "
                        "(llava/model/elastic/kd_teachers.py) and picks the preferred "
                        "family-matched teacher for --kd_student_key, failing LOUDLY if "
                        "none is runnable. A registry key ('llava7b', 'mobilevlm2_1.7b', "
                        "...) forces that teacher and still refuses if the audit says it "
                        "is incompatible. Omit for self-distillation (teacher='self').")
    p.add_argument("--kd_student_key", default=None, metavar="KEY",
                   help="Which backbone is the student (tinyllama|mobilellama|smollm2|"
                        "qwen0.5b|qwen1.5b). Required by --kd_teacher resolution; the "
                        "launchers pass their own LLM_KEY.")
    p.add_argument("--kd_type", choices=("auto", "logits"), default="auto",
                   help="KD mechanism. Only 'logits' (prefix-KL over the vocabulary, "
                        "masked to assistant-response positions) is implemented -- there "
                        "is no hidden-state, attention, or response-level KD loss in this "
                        "repo, and CORAL is self-sourced even with an external teacher. "
                        "'auto' resolves to 'logits'. The choice list is deliberately "
                        "short: adding a name here without a loss behind it would let a "
                        "run claim a KD type it did not perform.")
    p.add_argument("--nest_version", choices=("v8", "v14"), default=None,
                   help="Named recipe preset. Currently sets ONLY the default "
                        "--lora_type (v8->'v8', v14->'asc'), because as of "
                        "2026-09-13 that is the sole difference between the two "
                        "recipes — every other component is identical (see "
                        "docs/EXPERIMENT_JOURNAL.md 16o). An explicit --lora_type "
                        "overrides it, which is what makes the 2x2 version x lora "
                        "matrix expressible even though 2 of its 4 cells are "
                        "duplicates today. Recorded in the checkpoint either way.")
    p.add_argument("--prefix_kl_weight", type=float, default=_DEFAULT_KL_WEIGHT,
                   help="Weight on the prefix-KL self-distillation loss term.")
    p.add_argument("--coral_weight", type=float, default=_DEFAULT_CORAL_WEIGHT,
                   help="Weight on the CORAL token-alignment loss term.")
    p.add_argument("--use_kd", type=lambda x: x.lower() not in ("false", "0", "no"),
                   default=True, metavar="BOOL",
                   help="Enable prefix-KL distillation (default True; pass False to disable).")
    p.add_argument("--use_coral", type=lambda x: x.lower() not in ("false", "0", "no"),
                   default=True, metavar="BOOL",
                   help="Enable CORAL alignment loss (default True; pass False to disable).")
    p.add_argument("--projector_out_norm", type=lambda x: x.lower() not in ("false", "0", "no"),
                   default=False, metavar="BOOL",
                   help="Add a LayerNorm on the projector output, with its gain "
                        "calibrated to the LLM's token-embedding std at attach time "
                        "(default False). Without it the projector emits tokens ~36x "
                        "larger than the embeddings, which is survivable while the "
                        "backbone is frozen but is a divergence risk in a Stage-2 "
                        "full finetune.")
    p.add_argument("--teacher", choices=("self", "llava"), default="self",
                   help="KD target. 'self' (default) distills from the SAME model at "
                        "tok_levels[0] -- no second model, and measured KL ~0.006 "
                        "because teacher and student are identical weights. 'llava' "
                        "uses a frozen external LLaVA-1.5-7B, so every level becomes "
                        "a student and the KL carries real signal; costs ~14 GB/GPU.")
    p.add_argument("--teacher_model_path", default="liuhaotian/llava-v1.5-7b",
                   help="Checkpoint for --teacher llava. Must share the student's "
                        "tokenizer/vocab (Llama-32000) or the KL is meaningless.")
    p.add_argument("--use_pos_embed", type=lambda x: x.lower() not in ("false", "0", "no"),
                   default=False, metavar="BOOL",
                   help="Positional encodings in the resampler (default False). With "
                        "them off the 256 resampler outputs collapse to an effective "
                        "rank of ~12-21 out of 256 (job 26568), which is why 16 tokens "
                        "carry as much signal as 256.")
    p.add_argument("--pos_embed_type", choices=("learned", "sincos2d"), default="learned",
                   help="'learned' = trainable encodings; 'sincos2d' = frozen 2-D "
                        "sine-cosine grid shared by queries and patches, the "
                        "MQT-LLaVA design. Only used when --use_pos_embed True.")
    p.add_argument("--use_nested_dropout", type=lambda x: x.lower() not in ("false", "0", "no"),
                   default=True, metavar="BOOL",
                   help="Randomly truncate each non-teacher level to randint(1, n_tok) "
                        "queries per step (default True). This is what induces the "
                        "Matryoshka ordering, but it means a level trains at ~half its "
                        "nominal budget on average and at its NOMINAL length only 1/n "
                        "of the time -- the 16-token level runs at a single visual "
                        "token in ~6% of steps. Pass False to train each level at "
                        "exactly its tok_levels entry.")
    p.add_argument("--n_sample_students", type=int, default=0, metavar="INT",
                   help="Students sampled per step (0=full grid, 1=Option A, etc.).")
    p.add_argument("--vision_lora_enable", type=lambda x: x.lower() not in ("false", "0", "no"),
                   default=True, metavar="BOOL",
                   help="Inject rank-nested LoRA into the vision tower (default True; "
                        "pass False to run with a fully frozen, unspecialized vision "
                        "tower -- token-budget reduction still happens, just without "
                        "per-level encoder adapters).")
    p.add_argument("--vision_lora_specialize_tok", type=lambda x: x.lower() not in ("false", "0", "no"),
                   default=True, metavar="BOOL",
                   help="Tie the vision-tower LoRA adapter level to each tok_level "
                        "(default True). If False but --vision_lora_enable is True, "
                        "a single shared adapter (max rank) is used for all levels "
                        "instead of one adapter per level. Ignored if "
                        "--vision_lora_enable is False.")
    p.add_argument("--query_selection",
                   choices=NestedQueryResampler.QUERY_SELECTION_MODES, default="prefix",
                   help="How the resampler picks its n_tok output tokens (default "
                        "'prefix': the original queries[:n_tok] slice, content-"
                        "agnostic, used by every run through v5). 'magnitude' / "
                        "'attn_energy' / 'learned' instead run the full query bank "
                        "and keep the n_tok tokens an importance criterion ranks "
                        "highest -- see NestedQueryResampler's docstring "
                        "(llava/model/elastic/resampler.py) before using a "
                        "non-default value.")
    p.add_argument("--resampler_arch", choices=("query", "pool_anchored"), default="query",
                   help="'query' (default) = the original learned query bank alone. "
                        "'pool_anchored' = PARCEL-style: part of each budget is a "
                        "deterministic average-pooled spatial grid, the rest are "
                        "queries made pool-aware by self-attention before they "
                        "cross-attend to the patches. See docs/EXPERIMENT_JOURNAL.md.")
    p.add_argument("--anchor_mode", choices=("ratio", "fixed"), default="ratio",
                   help="'ratio' (default): anchor count is always --anchor_ratio of "
                        "whatever budget is active, computed for any budget. 'fixed': "
                        "--anchor_routing IS the routing table, a step function over "
                        "the budgets you declare (PARCEL's own literal design) -- "
                        "requires --anchor_routing.")
    p.add_argument("--anchor_ratio", type=float, default=0.25,
                   help="Fraction of the budget spent on anchors in --anchor_mode "
                        "ratio (default 0.25). Ignored in 'fixed' mode.")
    p.add_argument("--anchor_routing", type=str, default=None, metavar="B:NP,...",
                   help="Anchor counts per budget, e.g. '256:64,144:36,64:16,16:4'. "
                        "REQUIRED in --anchor_mode fixed (it is the routing table); "
                        "optional in 'ratio' mode (a per-budget override). Each NP "
                        "must be a perfect square reachable by integer pooling of the "
                        "patch grid, and monotone in budget.")
    p.add_argument("--use_token_decorrelation", type=lambda x: x.lower() not in ("false", "0", "no"),
                   default=False, metavar="BOOL",
                   help="Penalize off-diagonal cosine similarity among the QUERY "
                        "tokens each step (default False). For resampler_arch="
                        "pool_anchored this excludes the deterministic anchor "
                        "prefix (already ~98%% effective rank, see "
                        "EXPERIMENT_JOURNAL.md §11) and applies only to the "
                        "learned queries, which §11 measured as MORE collapsed "
                        "than the plain-query baseline (13.9%% vs 50.8%% rank). "
                        "For resampler_arch=query every token is a query, so it "
                        "applies to all of them.")
    p.add_argument("--decorr_weight", type=float, default=0.01,
                   help="Weight on the token-decorrelation loss term (default 0.01, "
                        "matching coral_weight's scale). Ignored if "
                        "--use_token_decorrelation is False.")
    elastic_args, remaining = p.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining  # hide elastic flags from HfArgumentParser
    return elastic_args


def main():
    elastic_args = _parse_elastic_args()

    tok_levels   = elastic_args.tok_levels

    # --- resolve lora_type / nest_version -> lora_ranks ---------------------
    # Order: explicit --lora_ranks wins, then --lora_type, then the preset
    # implied by --nest_version, then the historical default. Whatever is
    # resolved gets recorded on the ElasticConfig so the checkpoint says which
    # recipe made it instead of that living only in a directory name.
    lora_type    = elastic_args.lora_type
    nest_version = elastic_args.nest_version
    if lora_type is None and nest_version is not None:
        lora_type = {"v8": "v8", "v14": "asc"}[nest_version]

    derived = None
    if lora_type is not None:
        n = len(tok_levels)
        # Geometric ladder: 8,16,32,64 for n=4; generalises as 8*2^i so any
        # grid length works and max(ranks) stays the Stage-1 buffer width.
        ladder = [8 * (2 ** i) for i in range(n)]
        # "v8": rank ascends as budget descends -> ladder applied in order,
        # since tok_levels is descending. "asc": rank ascends WITH budget ->
        # ladder reversed.
        derived = ladder if lora_type == "v8" else ladder[::-1]

    if elastic_args.lora_ranks is not None:
        lora_ranks = elastic_args.lora_ranks
        if derived is not None and list(lora_ranks) != derived:
            raise ValueError(
                f"--lora_ranks {list(lora_ranks)} contradicts --lora_type "
                f"{lora_type!r} (which implies {derived}). Pass one or the other, "
                f"or make them agree -- otherwise the checkpoint would record a "
                f"lora_type that does not describe its own weights.")
    elif derived is not None:
        lora_ranks = derived
    else:
        lora_ranks = _DEFAULT_LORA_RANKS

    if len(lora_ranks) != len(tok_levels):
        raise ValueError(
            f"--lora_ranks length ({len(lora_ranks)}) must equal "
            f"--tok_levels length ({len(tok_levels)})"
        )

    if os.environ.get("LOCAL_RANK", "0") == "0":
        _print_config_banner(elastic_args, tok_levels, lora_ranks)
        _write_run_manifest(elastic_args, tok_levels, lora_ranks,
                            nest_version, lora_type)

    m3train.ELASTIC_CONFIG = ElasticConfig(
        token_reduction="nested_query",
        tok_levels=tok_levels,
        num_query_tokens=tok_levels[0],   # full query bank = largest level
        use_lora=elastic_args.vision_lora_enable,
        lora_specialize_tok=elastic_args.vision_lora_specialize_tok,
        lora_ranks=lora_ranks,
        lora_alpha=1.0,
        nest_version=nest_version,
        lora_type=lora_type,
        kd_student_key=elastic_args.kd_student_key,
        use_prefix_kl=elastic_args.use_kd,     prefix_kl_weight=elastic_args.prefix_kl_weight,
        use_coral_align=elastic_args.use_coral, coral_weight=elastic_args.coral_weight,
        # --kd_teacher, when given, becomes cfg.teacher and is resolved against
        # the registry inside attach_kd_teacher (which has the student model in
        # hand for the vocab check). Falls back to the historical --teacher.
        teacher=(elastic_args.kd_teacher or elastic_args.teacher),
        teacher_model_path=elastic_args.teacher_model_path,
        use_pos_embed=elastic_args.use_pos_embed,
        pos_embed_type=elastic_args.pos_embed_type,
        query_selection=elastic_args.query_selection,
        resampler_arch=elastic_args.resampler_arch,
        anchor_mode=elastic_args.anchor_mode,
        anchor_ratio=elastic_args.anchor_ratio,
        anchor_routing=_parse_anchor_routing(elastic_args.anchor_routing),
        use_nested_dropout=elastic_args.use_nested_dropout,
        projector_out_norm=elastic_args.projector_out_norm,
        use_token_decorrelation=elastic_args.use_token_decorrelation,
        decorr_weight=elastic_args.decorr_weight,
        kl_teacher_tok_level=0,                 # largest tok level is teacher
        n_sample_students=elastic_args.n_sample_students,
        log_adapter_every=50,
    )

    m3train.train(attn_implementation="flash_attention_2")


if __name__ == "__main__":
    main()
