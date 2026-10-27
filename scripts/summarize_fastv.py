#!/usr/bin/env python3
"""I-1 summary: dense 576 (old + re-trained), FastV at matched kept-token counts, and final-parcel (Nest-VLM) per token level.
CPU only.  sbatch scripts/summarize_fastv.sh"""
import glob
import json

EV = "/var/scratch/skalra/flexllava/eval_logs"
# (task, results key, display name, scale)
METRICS = [("mme", "mme_percetion_score,none", "MME-P", 1), ("pope", "pope_accuracy,none", "POPE", 100),
           ("scienceqa_img", "exact_match,none", "SciQA", 100), ("textvqa_val", "exact_match,none", "TextVQA", 100),
           ("gqa", "exact_match,none", "GQA", 100)]


def load(root):
    files = sorted(glob.glob(f"{root}/**/*results.json", recursive=True))
    if not files:
        return None
    merged = {}
    for f in files:                      # one results.json per run; later runs may add other benchmarks (vqav2, vizwiz)
        merged.update(json.load(open(f))["results"])
    return merged


def row(res):
    out = []
    for task, key, _, scale in METRICS:
        v = (res or {}).get(task, {}).get(key)
        out.append("   n/a" if v is None else f"{v * scale:6.2f}" if scale != 1 else f"{v:7.1f}")
    return " ".join(out)


print(f"{'config':44s} " + " ".join(f"{m[2]:>6s}" for m in METRICS))
r = load(f"{EV}/baseline-tinyllama-576tok")
print(f"{'dense 576 (old, deleted ckpt)':44s} {row(r)}")
r = load(f"{EV}/baseline-tinyllama-576tok-rerun")
print(f"{'dense 576 (re-trained)':44s} {row(r)}")
for n in (256, 144, 64, 16):
    r = load(f"{EV}/baseline-tinyllama-576tok-finetune-fastv-k2/{n}tok")
    print(f"{f'FastV K=2, keep {n}/576 (re-trained dense)':44s} {row(r)}")
for n in (256, 144, 64, 16):
    r = load(f"{EV}/elastic-finetune-tinyllama-v8-parcel/{n}tok")
    print(f"{f'final-parcel (Nest-VLM) @ {n} tokens':44s} {row(r)}")
# also dump the raw keys of one results file so a wrong key name is visible
r = load(f"{EV}/baseline-tinyllama-576tok-finetune-fastv-k2/64tok")
print("\nkeys:", {t: [k for k in v if "stderr" not in k and k != "alias"] for t, v in (r or {}).items()})
