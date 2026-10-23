"""I-3 stage D: downstream accuracy of self-guided crops, controls, and an early-exit cascade.  CPU only.

Compares, on the evaluation split (all 4,326 GT-box questions minus the 500
calibration questions used to choose the saliency map in stage B):
  full image | oracle GT crop | self-guided crops (16- and 256-token guide)
  | size-matched controls (centre, random, GT-centred window)
at 256 / 64 / 16 visual tokens, as crop-only and as image+crop, by GT box size
bin, with paired image-cluster bootstrap CIs.  Then an early-exit cascade:
answer with the 16-token guide pass when it is confident, otherwise crop and
answer again at 64 tokens.

The latency column is a MODEL built from the Nest-VLM Jetson logs (TinyLlama,
15 W; vision tower 184-187 ms at every budget, LM prefill 74.2 ms at 67 tokens,
75.1 at 115): it is not a measurement of this pipeline.
"""
import glob
import json
import sys
from collections import defaultdict

import numpy as np

EV = "/var/scratch/skalra/flexllava/eval_logs"
CP = "/var/scratch/skalra/flexllava/croppilot"
MODEL = "elastic-finetune-tinyllama-v8-parcel"
LEVELS = {"256tok": 256, "64tok": 64, "16tok": 16}
N_BOOT, SEED = 10000, 0
side = json.load(open(f"{CP}/textvqa_val_gtbox.json"))
calib = set(json.load(open(f"{CP}/selfroi_calib_qids.json")))
rng = np.random.default_rng(SEED)

# Jetson-log latency model, TinyLlama 15 W (ms): vision tower + resampler/projector, LM prefill.
T_ENC = {"256tok": 184.1 + 8.1, "64tok": 184.8 + 6.3, "16tok": 186.5 + 6.3}
T_LM = {"256tok": 188.6, "64tok": 75.1, "16tok": 74.2}


def load(src, mode, label):
    root = f"{EV}/croppilot/{mode}" if src in ("oracle", "full") else f"{EV}/croppilot_{src}/{mode}"
    files = glob.glob(f"{root}/{MODEL}/{label}/*/textvqa_val_gtbox_{mode}.json")
    if not files:
        return None
    d = json.load(open(sorted(files)[-1]))
    out = {}
    for e in d["logs"]:
        v = e["exact_match"]
        v = v["exact_match"] if isinstance(v, dict) else v
        out[str(e["doc"]["question_id"])] = (float(v), e["doc"]["image_id"])
    return out


def pair_ci(a, b, qs):
    by = defaultdict(list)
    for q in qs:
        by[a[q][1]].append(b[q][0] - a[q][0])
    s = np.array([sum(v) for v in by.values()]); c = np.array([len(v) for v in by.values()])
    idx = rng.integers(0, len(s), size=(N_BOOT, len(s)))
    m = s[idx].sum(1) / c[idx].sum(1)
    return 100 * s.sum() / c.sum(), 100 * np.percentile(m, 2.5), 100 * np.percentile(m, 97.5)


ROWS = [("full", "full", "full image"),
        ("oracle", "crop", "oracle GT crop"), ("g16", "crop", "self-ROI (16-tok guide)"),
        ("g256", "crop", "self-ROI (256-tok guide)"), ("gtctr", "crop", "GT-centred window"),
        ("centre", "crop", "centre window"), ("random", "crop", "random window"),
        ("oracle", "globalcrop", "oracle GT + image"), ("g16", "globalcrop", "self-ROI + image"),
        ("centre", "globalcrop", "centre window + image")]
BINS = ["all", "small", "medium", "large"]

for label in LEVELS:
    R = {(s, m): load(s, m, label) for s, m, _ in ROWS}
    have = [(s, m, n) for s, m, n in ROWS if R[(s, m)]]
    if not have:
        continue
    common = set.intersection(*[set(R[(s, m)]) for s, m, _ in have]) - calib
    print(f"\n===== {label}: evaluation split, {len(common)} questions =====")
    print(f"  {'source / input':<28}" + "".join(f"{b:>9}" for b in BINS))
    for s, m, name in have:
        row = []
        for b in BINS:
            qs = [q for q in common if b == "all" or side[q]["bin"] == b]
            row.append(100 * np.mean([R[(s, m)][q][0] for q in qs]))
        print(f"  {name:<28}" + "".join(f"{v:>9.2f}" for v in row))
    full = R[("full", "full")]
    cmp = [("g16", "crop", "full", "full"), ("g16", "crop", "centre", "crop"), ("g16", "crop", "random", "crop"),
           ("g16", "crop", "gtctr", "crop"), ("g256", "crop", "g16", "crop"), ("oracle", "crop", "g16", "crop"),
           ("g16", "globalcrop", "g16", "crop")]
    print("  paired differences (A - B), 95% CI:")
    for a_s, a_m, b_s, b_m in cmp:
        A, B = R.get((a_s, a_m)), R.get((b_s, b_m))
        if not A or not B:
            continue
        cells = []
        for b in BINS:
            qs = [q for q in common if b == "all" or side[q]["bin"] == b]
            d, lo, hi = pair_ci(B, A, qs)
            cells.append(f"{b} {d:+.1f}[{lo:+.1f},{hi:+.1f}]")
        print(f"    {a_s}/{a_m} - {b_s}/{b_m}:  " + "  ".join(cells))

# ---- early-exit cascade --------------------------------------------------
print("\n===== cascade: 16-token guide pass answers when confident, else crop (self-ROI, 64 tokens) and re-answer =====")
gate = np.load(f"{CP}/selfroi_g16_gate.npz")
gq = [str(q) for q in gate["qids"]]
keep = np.array([q not in calib for q in gq])
crop64 = load("g16", "crop", "64tok")
full256, full16 = load("full", "full", "256tok"), load("full", "full", "16tok")
if crop64:
    qs = [q for q, k in zip(gq, keep) if k and q in crop64]
    pos = {q: i for i, q in enumerate(gq)}
    p1 = np.array([gate["p1_acc"][pos[q]] for q in qs])
    esc_acc = np.array([crop64[q][0] for q in qs])
    print(f"  questions {len(qs)}; guide-pass accuracy {100 * p1.mean():.2f} "
          f"(lmms full@16 on the same questions: {100 * np.mean([full16[q][0] for q in qs]):.2f})")
    print(f"  always escalate (crop@64): {100 * esc_acc.mean():.2f}   full@256 reference: "
          f"{100 * np.mean([full256[q][0] for q in qs]):.2f} at {T_ENC['256tok'] + T_LM['256tok']:.0f} ms (model)")
    t1 = T_ENC["16tok"] + T_LM["16tok"]
    t2 = T_ENC["64tok"] + T_LM["64tok"]
    for cname, conf in (("mean token prob", gate["p1_mean"]), ("first token prob", gate["p1_first"])):
        c = np.array([conf[pos[q]] for q in qs])
        order = np.argsort(c)           # least confident first -> escalated first
        print(f"  gate on {cname}:")
        print(f"    {'escalate':>9}{'accuracy':>10}{'model TTFT ms':>15}")
        for frac in (0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0):
            k = int(round(frac * len(qs)))
            esc = np.zeros(len(qs), bool); esc[order[:k]] = True
            acc = np.where(esc, esc_acc, p1).mean()
            print(f"    {100 * frac:>8.0f}%{100 * acc:>10.2f}{t1 + frac * t2:>15.0f}")
    # references: random escalation and the best possible gate
    for frac in (0.4, 0.6):
        k = int(round(frac * len(qs)))
        r = []
        for _ in range(200):
            esc = np.zeros(len(qs), bool); esc[rng.permutation(len(qs))[:k]] = True
            r.append(np.where(esc, esc_acc, p1).mean())
        print(f"  random gate at {100 * frac:.0f}% escalation: {100 * np.mean(r):.2f}")
    print(f"  oracle gate (escalate only when the guide answer is wrong): "
          f"{100 * np.maximum(p1, esc_acc).mean():.2f} at {100 * (p1 < 1).mean():.0f}% escalation (upper bound)")
else:
    print("  g16 crop@64 results missing")
