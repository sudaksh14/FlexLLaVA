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
import os
import sys

import llava.train.train as m3train
from llava.model.elastic import ElasticConfig


def _get_argv_value(key: str) -> str:
    """Scan sys.argv for --key VALUE and return VALUE, or '?' if absent."""
    try:
        idx = sys.argv.index(key)
        return sys.argv[idx + 1] if idx + 1 < len(sys.argv) else "?"
    except ValueError:
        return "?"


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
    llm_lora_enabled = _get_argv_value("--lora_enable").lower() not in ("false", "0", "no", "?")
    if llm_lora_enabled:
        llm_lora_str = f"enabled  (rank={_get_argv_value('--lora_r')}, alpha={_get_argv_value('--lora_alpha')})"
    else:
        llm_lora_str = "disabled (full fine-tune)"
    if elastic_args.vision_lora_enable:
        vt_lora_str = (
            f"enabled  (ranks={lora_ranks}, "
            f"{'specialized per tok_level' if elastic_args.vision_lora_specialize_tok else 'one shared adapter'})"
        )
    else:
        vt_lora_str = "disabled (frozen, unspecialized vision tower)"
    sep = "=" * 64
    print(
        f"\n{sep}\n"
        f"[FlexLLaVA] Elastic Training — Job Configuration\n"
        f"  Vision Encoder : {ve}\n"
        f"  LLM            : {llm}\n"
        f"  Token budgets  : {tok_levels}  (teacher = tok{teacher_tok})\n"
        f"  LoRA ranks     : {lora_ranks}\n"
        f"  Training mode  : {mode}\n"
        f"  LLM LoRA       : {llm_lora_str}\n"
        f"  Vision LoRA    : {vt_lora_str}\n"
        f"  KD loss        : {kd_str}\n"
        f"  CORAL loss     : {coral_str}\n"
        f"{sep}\n",
        flush=True,
    )


# ---- defaults (used when args are not passed on the CLI) -----------------
_DEFAULT_TOK_LEVELS      = [256, 144, 64, 16]
_DEFAULT_LORA_RANKS      = [8, 16, 32, 64]
_DEFAULT_KL_WEIGHT       = 1.0
_DEFAULT_CORAL_WEIGHT    = 0.01


def _parse_elastic_args():
    """Pre-parse only the elastic-specific flags, leaving everything else in
    sys.argv for HfArgumentParser inside m3train.train()."""
    p = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    p.add_argument("--tok_levels", type=int, nargs="+", default=_DEFAULT_TOK_LEVELS,
                   help="Visual-token budgets per level, descending (e.g. 256 144 64 16).")
    p.add_argument("--lora_ranks", type=int, nargs="+", default=_DEFAULT_LORA_RANKS,
                   help="LoRA rank per tok level — must have same length as tok_levels.")
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
    elastic_args, remaining = p.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining  # hide elastic flags from HfArgumentParser
    return elastic_args


def main():
    elastic_args = _parse_elastic_args()

    tok_levels   = elastic_args.tok_levels
    lora_ranks   = elastic_args.lora_ranks

    if len(lora_ranks) != len(tok_levels):
        raise ValueError(
            f"--lora_ranks length ({len(lora_ranks)}) must equal "
            f"--tok_levels length ({len(tok_levels)})"
        )

    if os.environ.get("LOCAL_RANK", "0") == "0":
        _print_config_banner(elastic_args, tok_levels, lora_ranks)

    m3train.ELASTIC_CONFIG = ElasticConfig(
        token_reduction="nested_query",
        tok_levels=tok_levels,
        num_query_tokens=tok_levels[0],   # full query bank = largest level
        use_lora=elastic_args.vision_lora_enable,
        lora_specialize_tok=elastic_args.vision_lora_specialize_tok,
        lora_ranks=lora_ranks,
        lora_alpha=1.0,
        use_prefix_kl=elastic_args.use_kd,     prefix_kl_weight=elastic_args.prefix_kl_weight,
        use_coral_align=elastic_args.use_coral, coral_weight=elastic_args.coral_weight,
        use_pos_embed=elastic_args.use_pos_embed,
        pos_embed_type=elastic_args.pos_embed_type,
        use_nested_dropout=elastic_args.use_nested_dropout,
        projector_out_norm=elastic_args.projector_out_norm,
        kl_teacher_tok_level=0,                 # largest tok level is teacher
        n_sample_students=elastic_args.n_sample_students,
        log_adapter_every=50,
    )

    m3train.train(attn_implementation="flash_attention_2")


if __name__ == "__main__":
    main()
