"""Crop pilot analysis: accuracy by answer-box size bin, with paired CIs.

Reads lmms-eval per-sample logs written by eval_croppilot_level.sh:
  <ROOT>/<mode>/<model>/<label>/<run>/textvqa_val_gtbox_<mode>.json
joins them with the GT-box sidecar (size bins), and prints, per token budget:
  accuracy (x100) overall and per bin for full / crop / globalcrop, and the
  paired differences crop-full and globalcrop-full with 95% CIs from a
  bootstrap that resamples *images* (questions on one image are correlated).

CPU only; run through sbatch (analyze_croppilot.sh).
"""
import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np

ROOT = "/var/scratch/skalra/flexllava/eval_logs/croppilot"
MODEL = sys.argv[1] if len(sys.argv) > 1 else "elastic-finetune-tinyllama-v8-parcel"
SIDECAR = "/var/scratch/skalra/flexllava/croppilot/textvqa_val_gtbox.json"
LABELS = ["256tok", "144tok", "64tok", "16tok"]
MODES = ["full", "crop", "globalcrop"]
BINS = ["small", "medium", "large"]
N_BOOT, SEED = 10000, 0

side = json.load(open(SIDECAR))


def load(mode, label):
    """-> {qid: (score, image_id)} or None if the run is missing."""
    files = glob.glob(f"{ROOT}/{mode}/{MODEL}/{label}/*/textvqa_val_gtbox_{mode}.json")
    if not files:
        return None
    d = json.load(open(sorted(files)[-1]))
    out = {}
    for e in d["logs"]:
        v = e["exact_match"]
        v = v["exact_match"] if isinstance(v, dict) else v
        out[str(e["doc"]["question_id"])] = (float(v), e["doc"]["image_id"])
    return out


def boot_diff(a, b, qids, rng):
    """Paired mean difference b-a over qids with an image-cluster bootstrap."""
    by_img = defaultdict(list)
    for q in qids:
        by_img[a[q][1]].append(b[q][0] - a[q][0])
    sums = np.array([sum(v) for v in by_img.values()])
    cnts = np.array([len(v) for v in by_img.values()])
    n_img = len(sums)
    idx = rng.integers(0, n_img, size=(N_BOOT, n_img))
    means = sums[idx].sum(1) / cnts[idx].sum(1)
    return 100 * sums.sum() / cnts.sum(), 100 * np.percentile(means, 2.5), 100 * np.percentile(means, 97.5)


rng = np.random.default_rng(SEED)
for label in LABELS:
    runs = {m: load(m, label) for m in MODES}
    print(f"\n===== {MODEL}  {label} =====")
    have = [m for m in MODES if runs[m]]
    if not have:
        print("  no runs found")
        continue
    common = set.intersection(*[set(runs[m]) for m in have])
    print(f"  questions common to {have}: {len(common)}")
    hdr = f"  {'mode':<11}" + "".join(f"{b:>10}" for b in ["all"] + BINS)
    print(hdr)
    for m in have:
        row = []
        for b in ["all"] + BINS:
            qs = [q for q in common if b == "all" or side[q]["bin"] == b]
            row.append(100 * np.mean([runs[m][q][0] for q in qs]) if qs else float("nan"))
        print(f"  {m:<11}" + "".join(f"{v:>10.2f}" for v in row))
    print("  n         " + "".join(f"{sum(1 for q in common if b == 'all' or side[q]['bin'] == b):>10}" for b in ["all"] + BINS))
    if runs["full"]:
        for m in ["crop", "globalcrop"]:
            if not runs[m]:
                continue
            print(f"  paired {m} - full  (diff [95% CI], image-cluster bootstrap)")
            for b in ["all"] + BINS:
                qs = [q for q in common if b == "all" or side[q]["bin"] == b]
                d, lo, hi = boot_diff(runs["full"], runs[m], qs, rng)
                print(f"    {b:<8}{d:>8.2f}  [{lo:>6.2f}, {hi:>6.2f}]")

# Control: our 'full' mode must reproduce the earlier whole-val TextVQA numbers of
# this checkpoint (eval_lmms_level.sh logs) on the same questions, else the
# pilot's pipeline differs from the one that produced the paper's numbers.
print("\n===== control: 'full' vs earlier eval_lmms_level.sh run of the same checkpoint =====")
for label in LABELS:
    cur = load("full", label)
    old_files = glob.glob(f"/var/scratch/skalra/flexllava/eval_logs/{MODEL}/{label}/*/textvqa_val.json") + \
        glob.glob(f"/var/scratch/skalra/flexllava/eval_logs/{MODEL}/{label}/*/*textvqa_val*.json")
    old_files = [f for f in old_files if "samples" in os.path.basename(f) or os.path.basename(f).startswith("textvqa_val")]
    if not cur or not old_files:
        print(f"  {label}: missing ({'current' if not cur else 'earlier'})")
        continue
    d = json.load(open(sorted(old_files)[-1]))
    if "logs" not in d:
        print(f"  {label}: earlier file has no per-sample logs ({os.path.basename(sorted(old_files)[-1])})")
        continue
    old = {}
    for e in d["logs"]:
        v = e.get("exact_match")
        v = v["exact_match"] if isinstance(v, dict) else v
        old[str(e["doc"]["question_id"])] = float(v)
    qs = [q for q in cur if q in old]
    diff = [abs(cur[q][0] - old[q]) for q in qs]
    print(f"  {label}: n={len(qs)}  now={100*np.mean([cur[q][0] for q in qs]):.2f}  "
          f"earlier={100*np.mean([old[q] for q in qs]):.2f}  "
          f"questions with different score: {sum(1 for x in diff if x > 1e-9)}")
