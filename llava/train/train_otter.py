"""Launch elastic training on the Otter-style data pipeline.

This is a SECOND launcher, parallel to `llava/train/train_elastic.py`.  The
existing one is untouched and keeps driving the current runs; this one adds:

  #2  a YAML-declared, per-source training mixture      (llava/data_otter/mixture.py)
  #3  multi-turn packing of QA pairs sharing an image   (llava/data_otter/packing.py)
  #5  a pre-run verification gate                       (llava/data_otter/verify.py)
  #6  deterministic handling of unreadable images       (llava/data_otter/manifest.py)
  #7  per-source x per-tok-level loss telemetry         (llava/data_otter/telemetry.py)
  #8  token-accurate lengths for the batch sampler      (llava/data_otter/lengths.py)

HOW IT AVOIDS FORKING train.py
------------------------------
`llava.train.train.train()` looks up `make_supervised_data_module` and
`LLaVATrainer` as MODULE GLOBALS at call time, so rebinding those two names on
the module swaps in the new data module and trainer without editing a line of
it.  Everything else -- tokenizer setup, EOS alignment, the pad guard, the
per-backbone conversation handling, the elastic attach hook, DeepSpeed,
checkpointing -- runs exactly as it does today, from the same source.

The elastic flags are parsed by train_elastic.py's own parser, imported rather
than copied, so the two launchers cannot drift on what `--tok_levels` means.

Usage (see scripts/v1_5/finetune_otter_slm.sh):

    deepspeed llava/train/train_otter.py \\
        --mixture_config configs/otter/mix665k_baseline.yaml \\
        --otter_cache_dir /var/scratch/skalra/flexllava/cache/otter \\
        --tok_levels 256 144 64 16 --lora_ranks 8 16 32 64 \\
        ... all the usual LLaVA/elastic flags ...
"""

import argparse
import os
import sys

import llava.train.train as m3train
from llava.data_otter import dataset as otter_dataset
from llava.data_otter.dataset import OtterDataConfig
from llava.model.elastic import ElasticConfig
from llava.train.otter_trainer import OtterTrainer
# Imported, not duplicated: one definition of the elastic flags for both launchers.
from llava.train.train_elastic import _parse_elastic_args, _print_config_banner


def _parse_otter_args():
    """Pre-parse the pipeline-specific flags, leaving the rest of sys.argv alone."""
    p = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    p.add_argument("--mixture_config", required=True,
                   help="YAML describing the training mixture (see configs/otter/).")
    p.add_argument("--otter_cache_dir", default=None,
                   help="Where token-length and missing-image caches live. Without it every "
                        "run recomputes them (or falls back to the word-count heuristic).")
    p.add_argument("--otter_build_caches",
                   type=lambda x: x.lower() not in ("false", "0", "no"),
                   default=False, metavar="BOOL",
                   help="Build missing caches inline at startup instead of falling back. "
                        "Costs several minutes on 665k samples; prefer building them once "
                        "with `python -m llava.data_otter.prepare`.")
    p.add_argument("--otter_use_manifest",
                   type=lambda x: x.lower() not in ("false", "0", "no"),
                   default=True, metavar="BOOL",
                   help="Drop records whose image is known missing (default True).")
    p.add_argument("--otter_log_every", type=int, default=25,
                   help="Optimizer steps between telemetry flushes (default 25).")
    p.add_argument("--otter_source_grouped_batches",
                   type=lambda x: x.lower() not in ("false", "0", "no"),
                   default=True, metavar="BOOL",
                   help="Make every micro-batch single-source so per-source loss can be "
                        "attributed (default True). Gradient-equivalent at the optimizer step, "
                        "which averages 32 accumulated micro-batches. Pass False for the stock "
                        "batch composition, at the cost of per-source telemetry on "
                        "ocr_vqa/textvqa/vg.")
    p.add_argument("--otter_loss_weighting",
                   type=lambda x: x.lower() not in ("false", "0", "no"),
                   default=False, metavar="BOOL",
                   help="Apply per-source loss_weight from the mixture config. Off by default: "
                        "it scales gradients like a learning-rate change and is only exact for "
                        "source-homogeneous micro-batches.")
    otter_args, remaining = p.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining
    return otter_args


def main():
    # Order matters only in that each parser must strip its own flags before
    # HfArgumentParser sees argv; both use parse_known_args, so either order works.
    otter_args = _parse_otter_args()
    elastic_args = _parse_elastic_args()

    tok_levels = elastic_args.tok_levels
    lora_ranks = elastic_args.lora_ranks
    if len(lora_ranks) != len(tok_levels):
        raise ValueError(
            f"--lora_ranks length ({len(lora_ranks)}) must equal "
            f"--tok_levels length ({len(tok_levels)})")

    is_main = os.environ.get("LOCAL_RANK", "0") == "0"
    if is_main:
        _print_config_banner(elastic_args, tok_levels, lora_ranks)
        sep = "=" * 64
        print(f"{sep}\n[FlexLLaVA] Otter-style data pipeline\n"
              f"  Mixture config : {otter_args.mixture_config}\n"
              f"  Cache dir      : {otter_args.otter_cache_dir or '<none>'}\n"
              f"  Build caches   : {otter_args.otter_build_caches}\n"
              f"  Image manifest : {otter_args.otter_use_manifest}\n"
              f"  Telemetry every: {otter_args.otter_log_every} steps\n"
              f"  Loss weighting : {otter_args.otter_loss_weighting}\n"
              f"  Source batches : {otter_args.otter_source_grouped_batches}\n{sep}\n", flush=True)

    m3train.ELASTIC_CONFIG = ElasticConfig(
        token_reduction="nested_query",
        tok_levels=tok_levels,
        num_query_tokens=tok_levels[0],
        use_lora=elastic_args.vision_lora_enable,
        lora_specialize_tok=elastic_args.vision_lora_specialize_tok,
        lora_ranks=lora_ranks,
        lora_alpha=1.0,
        use_prefix_kl=elastic_args.use_kd, prefix_kl_weight=elastic_args.prefix_kl_weight,
        use_coral_align=elastic_args.use_coral, coral_weight=elastic_args.coral_weight,
        teacher=elastic_args.teacher,
        teacher_model_path=elastic_args.teacher_model_path,
        use_pos_embed=elastic_args.use_pos_embed,
        pos_embed_type=elastic_args.pos_embed_type,
        use_nested_dropout=elastic_args.use_nested_dropout,
        projector_out_norm=elastic_args.projector_out_norm,
        kl_teacher_tok_level=0,
        n_sample_students=elastic_args.n_sample_students,
        log_adapter_every=50,
    )

    otter_dataset.OTTER_DATA_CONFIG = OtterDataConfig(
        mixture_config=otter_args.mixture_config,
        cache_dir=otter_args.otter_cache_dir,
        build_caches=otter_args.otter_build_caches,
        use_manifest=otter_args.otter_use_manifest,
        # The teacher's visual prefix is what dominates a sample's length.
        visual_tokens=tok_levels[0],
        log_every=otter_args.otter_log_every,
    )

    OtterTrainer.otter_log_every = otter_args.otter_log_every
    OtterTrainer.otter_loss_weighting = otter_args.otter_loss_weighting
    OtterTrainer.otter_source_grouped_batches = otter_args.otter_source_grouped_batches

    # The swap. train() resolves both names from its module globals when it runs.
    m3train.make_supervised_data_module = otter_dataset.make_otter_data_module
    m3train.LLaVATrainer = OtterTrainer

    m3train.train(attn_implementation="flash_attention_2")


if __name__ == "__main__":
    main()
