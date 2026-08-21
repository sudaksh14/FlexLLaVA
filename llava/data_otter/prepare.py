"""Build the offline caches the Otter pipeline uses at training time.

Both caches are optional -- without them a run still trains, just with more
padding (no length cache) and with runtime image fallbacks instead of a clean
drop (no manifest).  Both are slow enough that you do not want them on the
critical path of a 24h job, and neither needs a GPU.

    python -m llava.data_otter.prepare \\
        --mixture_config configs/otter/mix665k_baseline.yaml \\
        --model_name_or_path TinyLlama/TinyLlama-1.1B-Chat-v1.0 \\
        --cache_dir /var/scratch/skalra/flexllava/cache/otter \\
        --visual_tokens 256 --build lengths,manifest

Run it as a batch job (scripts/otter/prepare_otter_cache.sh), never on the
login node.
"""

from __future__ import annotations

import argparse
import sys

from . import lengths as _lengths
from . import manifest as _manifest
from .mixture import build_mixture, load_mixture_config


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mixture_config", required=True)
    p.add_argument("--cache_dir", required=True)
    p.add_argument("--build", default="lengths,manifest",
                   help="comma-separated: lengths, manifest")
    p.add_argument("--model_name_or_path", default=None,
                   help="required for --build lengths (tokenizer only, no weights loaded)")
    p.add_argument("--model_max_length", type=int, default=2048)
    p.add_argument("--visual_tokens", type=int, default=256,
                   help="tok_levels[0]; the teacher's visual prefix length")
    p.add_argument("--hf_cache_dir", default=None)
    args = p.parse_args(argv)

    wanted = {w.strip() for w in args.build.split(",") if w.strip()}
    unknown = wanted - {"lengths", "manifest"}
    if unknown:
        print(f"[otter-prepare] unknown --build target(s): {sorted(unknown)}", file=sys.stderr)
        return 2

    cfg = load_mixture_config(args.mixture_config)
    records, source_names, image_folders = build_mixture(cfg, log=print)

    if "manifest" in wanted:
        missing = _manifest.scan_missing(records, source_names, image_folders, log=print)
        _manifest.save_missing(args.cache_dir, cfg.signature(), missing, log=print)

    if "lengths" in wanted:
        if not args.model_name_or_path:
            print("[otter-prepare] --build lengths needs --model_name_or_path", file=sys.stderr)
            return 2
        import transformers

        # The length cache is keyed on the tokenizer, so it must be built with
        # the same one training will use (use_fast=False, matching train()).
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            args.model_name_or_path,
            cache_dir=args.hf_cache_dir,
            model_max_length=args.model_max_length,
            padding_side="right",
            use_fast=("mpt" in args.model_name_or_path),
        )
        # Build against the SAME record list training will see, i.e. after the
        # manifest drop, or the cache length will not match and be discarded.
        known_missing = _manifest.load_missing(args.cache_dir, cfg.signature(), log=print)
        if known_missing:
            records, source_names = _manifest.drop_missing(
                records, source_names, known_missing, log=print)
        _lengths.load_or_build(
            records, tokenizer,
            visual_tokens=args.visual_tokens,
            mixture_signature=cfg.signature(),
            cache_dir=args.cache_dir,
            build_if_missing=True,
            log=print,
        )

    print("[otter-prepare] done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
