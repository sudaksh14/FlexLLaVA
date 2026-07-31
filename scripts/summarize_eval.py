#!/usr/bin/env python3
"""Summarise a FlexLLaVA elastic eval: accuracy and cost, per token level.

eval_lmms_level.sh runs each token level as a separate SLURM array task, so
nothing prints a combined view when they finish. This does.

Usage:
    python3 scripts/summarize_eval.py elastic-finetune-tinyllama
    python3 scripts/summarize_eval.py /path/to/checkpoint      # basename used
    python3 scripts/summarize_eval.py <tag> --log-root /other/eval_logs
"""

import argparse
import glob
import json
import os

LABELS = ["256tok", "144tok", "64tok", "16tok"]

# Preferred display order. Benchmarks are discovered from the results files
# rather than fixed here, so changing TASKS in eval_lmms_level.sh (dropping
# vqav2_val, adding mmbench_en_dev later) needs no edit to this script;
# anything not listed still prints, just after the known ones.
BENCHMARK_ORDER = ["mme", "pope", "scienceqa_img", "textvqa_val", "gqa",
                   "vqav2_val", "mmbench_en_dev"]

# AdaLLaVA paper, latency=0.85, LLaVA-v1.5-7B (Table 2). Reference only --
# that is a 7B Vicuna backbone, so it is an upper bound for our SLM runs, not
# an apples-to-apples target.
ADA_REFERENCE = {
    "mme": "1487.2/324.6",
    "pope": "85.9",
    "scienceqa_img": "70.4",
    "textvqa_val": "58.1",
    "gqa": "62.0",
    "vqav2_val": "76.6",
}

EFF_SUFFIX = ",efficiency"


def load_results(log_root, tag, label):
    """Newest results JSON for one token level, or None."""
    pattern = os.path.join(log_root, tag, label, "**", "*results*.json")
    files = sorted(glob.glob(pattern, recursive=True), key=os.path.getmtime)
    if not files:
        # lmms-eval names vary by version; fall back to any JSON with results
        files = sorted((f for f in glob.glob(os.path.join(log_root, tag, label, "**", "*.json"), recursive=True)
                        if "samples" not in os.path.basename(f)),
                       key=os.path.getmtime)
    for path in reversed(files):
        try:
            with open(path) as fh:
                data = json.load(fh)
            if "results" in data:
                return data
        except (json.JSONDecodeError, OSError):
            continue
    return None


def accuracy_of(data, bench):
    """Primary accuracy metric for a benchmark, ignoring efficiency keys."""
    if data is None:
        return None
    for task_key, metrics in data.get("results", {}).items():
        if task_key.lower() != bench.lower():
            continue
        # MME reports two sub-scores; show both.
        if bench == "mme":
            per = metrics.get("mme_perception_score,none")
            cog = metrics.get("mme_cognition_score,none")
            if per is not None:
                return f"{per:.1f}/{cog:.1f}" if cog is not None else f"{per:.1f}"
        for key, val in metrics.items():
            if not isinstance(val, (int, float)):
                continue
            if "stderr" in key or key.endswith(EFF_SUFFIX):
                continue
            return f"{val * 100:.1f}" if 0.0 <= val <= 1.0 else f"{val:.1f}"
    return None


def efficiency_of(data, bench):
    if data is None:
        return None
    return (data.get("efficiency") or {}).get(bench)


def fmt(value, width=9):
    return f"{'--' if value is None else value:>{width}}"


def discover_benchmarks(loaded):
    """Benchmarks actually present, in preferred order then alphabetical."""
    found = set()
    for data in loaded.values():
        if data:
            found.update(data.get("results", {}).keys())
    ordered = [b for b in BENCHMARK_ORDER if b in found]
    ordered += sorted(found - set(ordered))
    return ordered


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tag", help="checkpoint dir name (or full path)")
    ap.add_argument("--log-root", default="/var/scratch/skalra/flexllava/eval_logs")
    args = ap.parse_args()

    tag = os.path.basename(os.path.normpath(args.tag))
    loaded = {label: load_results(args.log_root, tag, label) for label in LABELS}

    if not any(loaded.values()):
        raise SystemExit(f"No results found under {os.path.join(args.log_root, tag)}")

    benchmarks = discover_benchmarks(loaded)

    print()
    print(f"  FlexLLaVA eval summary — {tag}")
    print(f"  {os.path.join(args.log_root, tag)}")

    # ---- accuracy -------------------------------------------------------
    print()
    print("  ACCURACY")
    header = f"  {'benchmark':<16}" + "".join(f"{l:>10}" for l in LABELS) + f"{'AdaLLaVA-7B*':>15}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for bench in benchmarks:
        row = f"  {bench:<16}"
        for label in LABELS:
            row += fmt(accuracy_of(loaded[label], bench), 10)
        row += f"{ADA_REFERENCE.get(bench, '--'):>15}"
        print(row)
    print("  * AdaLLaVA @0.85 latency on a 7B Vicuna backbone — reference scale,")
    print("    not a like-for-like target for a 1.1B SLM.")

    # ---- efficiency -----------------------------------------------------
    sample_eff = None
    for label in LABELS:
        for bench in benchmarks:
            sample_eff = sample_eff or efficiency_of(loaded[label], bench)
    if sample_eff is None:
        print()
        print("  EFFICIENCY: not recorded (re-run with the efficiency-enabled wrapper)")
        return

    hardware = sample_eff.get("hardware", "?")
    print()
    print(f"  EFFICIENCY — analytic roofline on {hardware}")
    print("  (LLM only; frozen vision tower is a constant offset across levels)")
    print()
    for metric, label_txt, scale, unit in [
        ("prefill_flops", "prefill", 1e12, "TFLOPs"),
        ("prefill_time", "prefill", 1e-3, "ms"),
        ("peak_memory_consumption", "peak mem", 1e9, "GB"),
    ]:
        header = f"  {label_txt + ' (' + unit + ')':<16}" + "".join(f"{l:>10}" for l in LABELS)
        print(header)
        print("  " + "-" * (len(header) - 2))
        for bench in benchmarks:
            row = f"  {bench:<16}"
            for label in LABELS:
                eff = efficiency_of(loaded[label], bench)
                if eff is None or metric not in eff:
                    row += fmt(None, 10)
                else:
                    row += f"{eff[metric] / scale:>10.3f}"
            print(row)
        print()

    # ---- deployment feasibility -----------------------------------------
    print(f"  FITS ON {hardware.upper()}")
    header = f"  {'benchmark':<16}" + "".join(f"{l:>10}" for l in LABELS)
    print(header)
    print("  " + "-" * (len(header) - 2))
    for bench in benchmarks:
        row = f"  {bench:<16}"
        for label in LABELS:
            eff = efficiency_of(loaded[label], bench)
            if eff is None:
                row += fmt(None, 10)
            else:
                fits = eff.get("fits_on_target")
                row += f"{'yes' if fits else 'NO' if fits is False else '?':>10}"
        print(row)
    print()
    print("  Roofline projection: a lower bound assuming perfect overlap, no")
    print("  kernel-launch or thermal overhead, and dense (non-sparse) throughput.")
    print("  Measure on-device before quoting a deployment latency.")
    print()


if __name__ == "__main__":
    main()
