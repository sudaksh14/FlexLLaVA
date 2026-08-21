"""Pre-run verification gate for the Otter-style pipeline.

WHY THIS EXISTS
---------------
Four separate training/eval cycles on this project were burned by data-side
defects that were all detectable in seconds, on CPU, before the job started:

  * pad_token == eos_token deleted every EOS from the labels, so TinyLlama
    answered correctly and never stopped, scoring 0.00% on GQA
    (`ensure_distinct_pad_token`, memory `project-pad-eos-collision-bug`);
  * Phi-3.5's template terminates turns with '<|end|>' while its tokenizer's
    eos_token was '<|endoftext|>' -- 0 EOS in the entire training sequence
    (`align_eos_with_template`);
  * BPE-tokenizer backbones (SmolLM2, Qwen2.5) mis-derived round boundaries, so
    only 3.1% of short-answer samples had EOS supervised and the supervised
    span was role scaffolding rather than the answer (`_auto_prefix_len`, then
    the prefix-tokenisation rewrite in preprocess_mpt);
  * an elastic Stage-1 checkpoint whose keys did not load, warm-starting
    nothing.

Otter gates its runs the same way -- `verify_yaml` shells out to
`pytest -m prerun` and aborts if it fails (pipeline/train/train_utils.py:155).
Otter's own checks are shallow (paths exist, JSON parses; its parquet column
assertion is commented out), so this goes further: it runs the REAL
preprocessing on REAL samples and inspects the labels that would be trained on.

Everything here is CPU-only and loads no model weights, so it costs seconds and
can run at the top of the sbatch script before the GPUs are touched.

    python -m llava.data_otter.verify \\
        --mixture_config configs/otter/mix665k_baseline.yaml \\
        --model_name_or_path TinyLlama/TinyLlama-1.1B-Chat-v1.0 \\
        --version v1 --model_max_length 2048

Exit code 0 = safe to train, 1 = do not start the job.
"""

from __future__ import annotations

import argparse
import glob
import os
import random
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# Thresholds. Deliberately loose: this gate exists to catch catastrophic,
# whole-run defects (0% EOS, everything masked), not to police normal variation.
MIN_EOS_SUPERVISED_FRAC = 0.90     # of non-truncated samples
MIN_ANSWER_COVERAGE_FRAC = 0.90    # supervised span actually contains the answer
MAX_FULLY_MASKED_FRAC = 0.02       # preprocess IGNOREs a whole sample on mismatch
MAX_TRUNCATED_FRAC = 0.05
MAX_MISSING_IMAGE_FRAC = 0.01


@dataclass
class Report:
    checks: List[tuple] = field(default_factory=list)   # (level, name, message)

    def ok(self, name, msg=""):
        self.checks.append(("PASS", name, msg))

    def warn(self, name, msg):
        self.checks.append(("WARN", name, msg))

    def fail(self, name, msg):
        self.checks.append(("FAIL", name, msg))

    @property
    def failed(self) -> bool:
        return any(level == "FAIL" for level, _, _ in self.checks)

    def render(self) -> str:
        width = max((len(n) for _, n, _ in self.checks), default=10)
        lines = ["", "=" * 72, "[otter-verify] pre-run verification", "=" * 72]
        for level, name, msg in self.checks:
            lines.append(f"  [{level}] {name:<{width}}  {msg}")
        lines.append("=" * 72)
        n_fail = sum(1 for l, _, _ in self.checks if l == "FAIL")
        n_warn = sum(1 for l, _, _ in self.checks if l == "WARN")
        verdict = "REFUSING TO TRAIN" if n_fail else "OK to train"
        lines.append(f"[otter-verify] {len(self.checks)} checks, {n_fail} failed, "
                     f"{n_warn} warnings -- {verdict}")
        lines.append("=" * 72)
        return "\n".join(lines)


# ----------------------------------------------------------------------------
# Tokenizer / template setup -- mirrors llava/train/train.py's train() exactly
# ----------------------------------------------------------------------------
def prepare_tokenizer_and_template(model_name_or_path: str,
                                   version: str,
                                   model_max_length: int,
                                   cache_dir: Optional[str],
                                   report: Report):
    """Reproduce train()'s tokenizer setup, with no model loaded."""
    import transformers
    from llava import conversation as conversation_lib
    from llava.train.train import align_eos_with_template, ensure_distinct_pad_token

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_name_or_path,
        cache_dir=cache_dir,
        model_max_length=model_max_length,
        padding_side="right",
        use_fast=("mpt" in model_name_or_path),   # train() uses use_fast=False otherwise
    )

    if version in conversation_lib.conv_templates:
        conv = conversation_lib.conv_templates[version]
        report.ok("template", f"version={version!r} -> sep_style={conv.sep_style.name}")
    else:
        conv = conversation_lib.conv_templates["vicuna_v1"]
        report.warn("template", f"version={version!r} is not a known template; train() would "
                                f"silently fall back to vicuna_v1. Is that intended?")
    conversation_lib.default_conversation = conv

    # Same order as train(): resolve template -> align eos -> separate pad.
    # smart_tokenizer_and_embedding_resize already tolerates model=None (it
    # returns after adding the token), so both helpers are safe to call here.
    align_eos_with_template(tokenizer, conv, model=None)
    ensure_distinct_pad_token(tokenizer, model=None)

    # ---- the two collision bugs, checked explicitly ------------------------
    if tokenizer.pad_token_id == tokenizer.eos_token_id:
        report.fail("pad!=eos",
                    f"pad_token_id == eos_token_id == {tokenizer.eos_token_id}. Every EOS would "
                    f"be stripped from the labels by the attention-mask compaction in "
                    f"prepare_inputs_labels_for_multimodal; the model would never learn to stop.")
    else:
        report.ok("pad!=eos", f"pad={tokenizer.pad_token!r}({tokenizer.pad_token_id}) "
                              f"eos={tokenizer.eos_token!r}({tokenizer.eos_token_id})")

    term = conv.sep2 if conv.sep_style.name == "TWO" else conv.sep
    if term:
        term_id = tokenizer.convert_tokens_to_ids(term.strip())
        unk = getattr(tokenizer, "unk_token_id", None)
        if term_id is None or term_id == unk:
            report.warn("eos-align", f"template terminator {term.strip()!r} is not a single "
                                     f"registered token; EOS supervision is checked below")
        elif term_id != tokenizer.eos_token_id:
            report.fail("eos-align",
                        f"template terminates turns with {term.strip()!r} (id {term_id}) but "
                        f"eos_token_id is {tokenizer.eos_token_id}. generate() would wait for a "
                        f"token the model is never trained to emit.")
        else:
            report.ok("eos-align", f"terminator {term.strip()!r} == eos (id {term_id})")
    return tokenizer, conv


# ----------------------------------------------------------------------------
# Label-level checks: run the REAL preprocessing and inspect what gets supervised
# ----------------------------------------------------------------------------
def check_supervision(records: List[Dict[str, Any]],
                      source_names: List[str],
                      tokenizer,
                      conv,
                      data_args,
                      n_samples: int,
                      seed: int,
                      report: Report,
                      log=print) -> None:
    """Tokenise real samples and assert the labels are trainable.

    No images are opened: preprocess(..., has_image=True) only routes text
    through tokenizer_image_token, so label masking can be verified with zero
    filesystem cost.
    """
    import copy
    import torch
    from llava.constants import IGNORE_INDEX
    from llava.train.train import preprocess, preprocess_multimodal

    is_plain = conv.sep_style.name == "PLAIN"
    rng = random.Random(seed)
    idx = list(range(len(records)))
    rng.shuffle(idx)
    idx = idx[:n_samples]

    n_eos = n_nonempty = n_truncated = n_fully_masked = 0
    n_answer_covered = n_answer_checked = 0
    n_ok = 0
    per_source_supervised: Dict[str, List[int]] = {}
    failures: List[str] = []

    for i in idx:
        rec = records[i]
        source = source_names[i]
        has_image = bool(rec.get("image"))
        sources = [copy.deepcopy(rec["conversations"])]
        if has_image:
            sources = preprocess_multimodal(sources, data_args)
        try:
            out = preprocess(sources, tokenizer, has_image=has_image)
        except Exception as e:                      # noqa: BLE001 - report, don't crash the gate
            failures.append(f"sample {i} ({source}): preprocess raised {type(e).__name__}: {e}")
            continue

        input_ids = out["input_ids"][0]
        labels = out["labels"][0]
        n_ok += 1

        supervised = labels[labels != IGNORE_INDEX]
        per_source_supervised.setdefault(source, []).append(int(supervised.numel()))

        if supervised.numel() == 0:
            n_fully_masked += 1
            continue
        n_nonempty += 1

        truncated = int(input_ids.shape[0]) >= tokenizer.model_max_length
        if truncated:
            n_truncated += 1

        if not is_plain and not truncated:
            if (supervised == tokenizer.eos_token_id).any():
                n_eos += 1

        # Does the supervised span actually contain the ANSWER, or just role
        # scaffolding? This is the check that would have caught the SmolLM2
        # regression, where 1 token per sample was supervised and it was the
        # wrong one.
        gpt_turns = [t["value"] for t in rec["conversations"] if t.get("from") == "gpt"]
        if gpt_turns and not truncated:
            answer = gpt_turns[0].strip()
            probe = "".join(answer.split())[:12]
            if len(probe) >= 4:
                n_answer_checked += 1
                decoded = tokenizer.decode(supervised, skip_special_tokens=True)
                if probe in "".join(decoded.split()):
                    n_answer_covered += 1

    if n_ok == 0:
        report.fail("supervision", "preprocess() failed on every sampled record; see failures above")
        for f in failures[:5]:
            log(f"[otter-verify]   {f}")
        return

    if failures:
        report.warn("preprocess-errors",
                    f"{len(failures)}/{len(idx)} samples raised; first: {failures[0]}")

    frac_masked = n_fully_masked / n_ok
    if frac_masked > MAX_FULLY_MASKED_FRAC:
        report.fail("no-full-mask",
                    f"{n_fully_masked}/{n_ok} ({frac_masked:.1%}) samples came back FULLY masked. "
                    f"preprocess() blanks a sample's labels when its round accounting disagrees "
                    f"with the tokenised length -- those samples contribute no gradient at all.")
    else:
        report.ok("no-full-mask", f"{n_fully_masked}/{n_ok} ({frac_masked:.1%}) fully masked")

    n_checked_eos = n_nonempty - n_truncated
    if is_plain:
        report.ok("eos-supervised", "skipped: PLAIN template (Stage 1 pretrain) has no turn EOS")
    elif n_checked_eos <= 0:
        report.warn("eos-supervised", "no non-truncated samples to check")
    else:
        frac = n_eos / n_checked_eos
        if frac < MIN_EOS_SUPERVISED_FRAC:
            report.fail("eos-supervised",
                        f"only {n_eos}/{n_checked_eos} ({frac:.1%}) non-truncated samples have "
                        f"eos_token_id ({tokenizer.eos_token_id}) in their labels. The model "
                        f"would not learn to stop -- this is the TinyLlama 0.00%-GQA failure.")
        else:
            report.ok("eos-supervised", f"{n_eos}/{n_checked_eos} ({frac:.1%}) samples supervise EOS")

    if n_answer_checked:
        frac = n_answer_covered / n_answer_checked
        if frac < MIN_ANSWER_COVERAGE_FRAC:
            report.fail("answer-supervised",
                        f"the supervised span contains the assistant's answer in only "
                        f"{n_answer_covered}/{n_answer_checked} ({frac:.1%}) samples. The mask "
                        f"window is misaligned -- role scaffolding is being trained on instead "
                        f"of the answer.")
        else:
            report.ok("answer-supervised",
                      f"{n_answer_covered}/{n_answer_checked} ({frac:.1%}) supervise the answer text")

    frac_trunc = n_truncated / n_ok
    if frac_trunc > MAX_TRUNCATED_FRAC:
        report.warn("truncation",
                    f"{n_truncated}/{n_ok} ({frac_trunc:.1%}) samples hit model_max_length="
                    f"{tokenizer.model_max_length}. Truncation can cut the answer and its EOS; "
                    f"if packing is enabled, lower packing.max_turns / max_chars.")
    else:
        report.ok("truncation", f"{n_truncated}/{n_ok} ({frac_trunc:.1%}) at max length")

    detail = ", ".join(
        f"{s}={sum(v) / len(v):.0f}" for s, v in sorted(per_source_supervised.items()) if v)
    report.ok("supervised-tokens", f"mean per sample by source: {detail}")


# ----------------------------------------------------------------------------
# Other checks
# ----------------------------------------------------------------------------
def check_images(records, source_names, image_folders, n_samples, seed, report, log=print):
    rng = random.Random(seed)
    idx = [i for i in range(len(records)) if records[i].get("image")]
    if not idx:
        report.warn("images", "no image samples in the mixture")
        return
    rng.shuffle(idx)
    idx = idx[:n_samples]
    missing = []
    for i in idx:
        folder = image_folders.get(source_names[i]) or ""
        path = os.path.join(folder, records[i]["image"])
        if not os.path.exists(path):
            missing.append(path)
    frac = len(missing) / len(idx)
    if frac > MAX_MISSING_IMAGE_FRAC:
        report.fail("images", f"{len(missing)}/{len(idx)} ({frac:.1%}) sampled images do not "
                              f"exist. First: {missing[0]}")
    elif missing:
        report.warn("images", f"{len(missing)}/{len(idx)} sampled images missing (first: "
                              f"{missing[0]}). Build a manifest so they are dropped "
                              f"deterministically rather than randomly substituted.")
    else:
        report.ok("images", f"all {len(idx)} sampled images resolve")


def check_packing(records, report):
    """Every packed record must carry exactly one <image> for its one image."""
    from .packing import validate_packed_record

    packed = [r for r in records if r.get("n_packed", 1) > 1]
    if not packed:
        report.ok("packing", "no packed records in this mixture")
        return
    problems = []
    for rec in packed:
        p = validate_packed_record(rec)
        if p:
            problems.append(f"{rec.get('id')}: {p}")
    if problems:
        report.fail("packing", f"{len(problems)}/{len(packed)} packed records are malformed. "
                               f"First: {problems[0]}")
    else:
        sizes = [r["n_packed"] for r in packed]
        report.ok("packing", f"{len(packed)} packs valid, mean {sum(sizes) / len(sizes):.1f} "
                             f"records/pack, max {max(sizes)}")


def check_elastic_checkpoint(path: Optional[str], report: Report):
    """Confirm a Stage-1 checkpoint actually contains loadable elastic keys."""
    if not path:
        report.ok("elastic-ckpt", "no --pretrain_elastic_path given (Stage 1 run?)")
        return
    if not os.path.isdir(path):
        report.fail("elastic-ckpt", f"path does not exist: {path}")
        return

    # Same discovery order as _load_elastic_pretrain_weights.
    bins = sorted(glob.glob(os.path.join(path, "pytorch_model*.bin")))
    safes = sorted(glob.glob(os.path.join(path, "model*.safetensors")))
    if not bins and not safes:
        report.fail("elastic-ckpt", f"no pytorch_model*.bin or model*.safetensors in {path}; "
                                    f"train() would skip the warm-start and silently train "
                                    f"the elastic modules from scratch.")
        return

    keys: List[str] = []
    try:
        if safes:
            from safetensors import safe_open
            for f in safes:
                with safe_open(f, framework="pt") as fh:
                    keys.extend(fh.keys())
        else:
            import torch
            for f in bins:
                try:
                    sd = torch.load(f, map_location="cpu", mmap=True, weights_only=True)
                except (TypeError, RuntimeError):
                    # mmap/weights_only need torch>=2.1; fall back to a full read.
                    sd = torch.load(f, map_location="cpu")
                keys.extend(sd.keys())
                del sd
    except Exception as e:                          # noqa: BLE001
        report.warn("elastic-ckpt", f"could not read weights in {path}: {type(e).__name__}: {e}")
        return

    tags = ("elastic_resampler.", "elastic_projector.", ".lora_A", ".lora_B")
    hits = [k for k in keys if any(t in k for t in tags)]
    if not hits:
        report.fail("elastic-ckpt",
                    f"{path} contains {len(keys)} tensors but NONE match "
                    f"{tags}. The warm-start would load nothing.")
    else:
        n_res = sum(1 for k in hits if "elastic_resampler." in k)
        n_proj = sum(1 for k in hits if "elastic_projector." in k)
        n_lora = sum(1 for k in hits if ".lora_A" in k or ".lora_B" in k)
        report.ok("elastic-ckpt", f"{len(hits)} elastic keys (resampler={n_res}, "
                                  f"projector={n_proj}, lora={n_lora})")


def check_tokenizer_files(path: Optional[str], report: Report):
    """A finished checkpoint needs tokenizer files or eval cannot load it.

    Older elastic-finetune-* runs saved the tokenizer only into the last
    checkpoint-N/ subdir, so evaluating the top-level dir failed until the files
    were copied up by hand.
    """
    if not path or not os.path.isdir(path):
        return
    wanted = ("tokenizer_config.json",)
    if not any(os.path.exists(os.path.join(path, w)) for w in wanted):
        report.warn("ckpt-tokenizer",
                    f"{path} has no tokenizer_config.json. If eval loads this directory it will "
                    f"fail; check the checkpoint-N/ subdirs.")


# ----------------------------------------------------------------------------
def run_verification(args, log=print) -> Report:
    from types import SimpleNamespace

    from .mixture import load_mixture_config, build_mixture

    report = Report()

    # ---- 1. config ---------------------------------------------------------
    try:
        cfg = load_mixture_config(args.mixture_config)
        report.ok("mixture-config", f"{args.mixture_config} parsed, "
                                    f"{len(cfg.sources)} sources declared")
    except ValueError as e:
        report.fail("mixture-config", str(e))
        return report

    # ---- 2. mixture materialises ------------------------------------------
    try:
        records, source_names, image_folders = build_mixture(cfg, log=log)
        report.ok("mixture-build", f"{len(records)} samples materialised")
    except (ValueError, OSError) as e:
        report.fail("mixture-build", f"{type(e).__name__}: {e}")
        return report

    if not records:
        report.fail("mixture-build", "mixture is empty")
        return report

    # ---- 3. packing sanity -------------------------------------------------
    check_packing(records, report)

    # ---- 4. images ---------------------------------------------------------
    check_images(records, source_names, image_folders,
                 n_samples=args.n_image_samples, seed=cfg.seed, report=report, log=log)

    # ---- 5. tokenizer + template ------------------------------------------
    try:
        tokenizer, conv = prepare_tokenizer_and_template(
            args.model_name_or_path, args.version, args.model_max_length,
            args.cache_dir, report)
    except Exception as e:                          # noqa: BLE001
        report.fail("tokenizer", f"could not load {args.model_name_or_path}: "
                                 f"{type(e).__name__}: {e}")
        return report

    # ---- 6. label supervision on real samples ------------------------------
    data_args = SimpleNamespace(
        is_multimodal=True,
        mm_use_im_start_end=args.mm_use_im_start_end,
        image_aspect_ratio="pad",
    )
    check_supervision(records, source_names, tokenizer, conv, data_args,
                      n_samples=args.n_samples, seed=cfg.seed, report=report, log=log)

    # ---- 7. warm-start checkpoint -----------------------------------------
    check_elastic_checkpoint(args.pretrain_elastic_path, report)
    check_tokenizer_files(args.pretrain_elastic_path, report)

    return report


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Pre-run verification gate for the Otter-style FlexLLaVA data pipeline.")
    p.add_argument("--mixture_config", required=True)
    p.add_argument("--model_name_or_path", required=True)
    p.add_argument("--version", required=True, help="conversation template key, e.g. v1 / chatml / phi3")
    p.add_argument("--model_max_length", type=int, default=2048)
    p.add_argument("--cache_dir", default=None)
    p.add_argument("--pretrain_elastic_path", default=None)
    p.add_argument("--mm_use_im_start_end", action="store_true", default=False)
    p.add_argument("--n_samples", type=int, default=256,
                   help="how many records to run the real preprocessing on")
    p.add_argument("--n_image_samples", type=int, default=512,
                   help="how many image paths to stat")
    p.add_argument("--warn_only", action="store_true",
                   help="report failures but exit 0 (for inspecting a known-bad config)")
    return p


def main(argv=None) -> int:
    args = build_arg_parser().parse_args(argv)
    report = run_verification(args)
    print(report.render(), flush=True)
    if report.failed and not args.warn_only:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
