#!/usr/bin/env python3
"""All TinyLlama runs, per visual-token level: dense, FastV, M3/MQT baselines, every elastic recipe.  CPU only.
sbatch scripts/summarize_tinyllama.sh   (merges every results.json under <tag>/<n>tok/)"""
import glob
import json
import os

EV = "/var/scratch/skalra/flexllava/eval_logs"
METRICS = [("mme", "mme_percetion_score,none", "MME-P", 1), ("pope", "pope_accuracy,none", "POPE", 100),
           ("scienceqa_img", "exact_match,none", "SciQA", 100), ("textvqa_val", "exact_match,none", "TextVQA", 100),
           ("gqa", "exact_match,none", "GQA", 100)]
TAGS = ["baseline-tinyllama-576tok-rerun", "baseline-tinyllama-finetune-fastv-k2", "baseline-tinyllama-finetune-fastv-k4",
        "baseline-tinyllama-finetune-fastv-k8", "baseline-tinyllama-ladder-finetune", "baseline-tinyllama-4tok-finetune",
        "mqt-finetune-tinyllama-ladder", "mqt-finetune-tinyllama-4tok", "elastic-finetune-tinyllama-parcel-faithful",
        "elastic-finetune-tinyllama-v8-parcel", "elastic-finetune-tinyllama-v11-parcel-nolora",
        "elastic-finetune-tinyllama-v10-parcel-lora16", "elastic-finetune-tinyllama-v14-parcel-asclora",
        "elastic-finetune-tinyllama-v9-parcel-decorr", "elastic-finetune-tinyllama-v12-parcel-kd7b",
        "elastic-finetune-tinyllama-final-parcel-576ctrl", "elastic-finetune-tinyllama-v13-parcel-longladder",
        "elastic-finetune-tinyllama-v4", "elastic-finetune-tinyllama-v5", "elastic-finetune-tinyllama-v4-query-nopos",
        "elastic-finetune-tinyllama-v4-query-sincos"]


def load(root):
    merged = {}
    for f in sorted(glob.glob(f"{root}/**/*results.json", recursive=True)):
        merged.update(json.load(open(f))["results"])
    return merged


print(f"{'run':48s} {'level':>7s} " + " ".join(f"{m[2]:>7s}" for m in METRICS))
for tag in TAGS:
    root = f"{EV}/{tag}"
    if not os.path.isdir(root):
        print(f"{tag:48s}   (no eval logs)")
        continue
    levels = sorted([d for d in os.listdir(root) if d.endswith("tok") or "tok" in d], key=lambda d: -int("".join(c for c in d if c.isdigit()) or 0))
    for lv in levels:
        res = load(f"{root}/{lv}")
        cells = []
        for task, key, _, scale in METRICS:
            v = res.get(task, {}).get(key)
            cells.append("    n/a" if v is None else f"{v * scale:7.2f}" if scale != 1 else f"{v:7.1f}")
        print(f"{tag:48s} {lv:>7s} " + " ".join(cells))
