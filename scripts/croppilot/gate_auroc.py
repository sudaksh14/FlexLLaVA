"""I-4 check: is the guide pass's confidence a usable early-exit gate?  CPU only.

AUROC (image-cluster bootstrap 95% CI) of two confidence scores from the 16- and
256-token guide passes, on the 3,826 evaluation questions (calibration split excluded), for
  (a) "the guide answer is correct" (TextVQA score >= 0.5, and == 1), the I-4 refutation test (AUROC < 0.7), and
  (b) "escalating to the self-ROI crop at 64 tokens helps" (crop score > guide score), the quantity a gate has to find.
Then one operating point chosen on the calibration split (escalate the least-confident half there) and applied to the evaluation split.
"""
import numpy as np
from sklearn.metrics import roc_auc_score

import sys
sys.path.insert(0, "/home/skalra/FlexLLaVA/scripts/croppilot")
CP = "/var/scratch/skalra/flexllava/croppilot"
EV = "/var/scratch/skalra/flexllava/eval_logs"
import glob, json
from collections import defaultdict

MODEL = "elastic-finetune-tinyllama-v8-parcel"
rng = np.random.default_rng(0)


def load(src, mode, label):
    root = f"{EV}/croppilot/{mode}" if src in ("oracle", "full") else f"{EV}/croppilot_{src}/{mode}"
    files = glob.glob(f"{root}/{MODEL}/{label}/*/textvqa_val_gtbox_{mode}.json")
    d = json.load(open(sorted(files)[-1]))
    out = {}
    for e in d["logs"]:
        v = e["exact_match"]
        v = v["exact_match"] if isinstance(v, dict) else v
        out[str(e["doc"]["question_id"])] = (float(v), e["doc"]["image_id"])
    return out


def auroc_ci(score, label, img, n=2000):
    a = roc_auc_score(label, score)
    ids = {i: k for k, i in enumerate(sorted(set(img)))}
    groups = defaultdict(list)
    for j, i in enumerate(img):
        groups[ids[i]].append(j)
    glist = [np.array(groups[k]) for k in range(len(ids))]
    boots = []
    for _ in range(n):
        pick = rng.integers(0, len(glist), len(glist))
        idx = np.concatenate([glist[p] for p in pick])
        if label[idx].min() == label[idx].max():
            continue
        boots.append(roc_auc_score(label[idx], score[idx]))
    return a, np.percentile(boots, 2.5), np.percentile(boots, 97.5)


crop64 = load("g16", "crop", "64tok")
for guide in ("g16", "g256"):
    g = np.load(f"{CP}/selfroi_{guide}_gate.npz")
    qids = [str(q) for q in g["qids"]]
    ev = np.array([(not c) and q in crop64 for q, c in zip(qids, g["is_calib"])])
    cal = g["is_calib"]
    img = [crop64[q][1] for q, e in zip(qids, ev) if e]
    esc = np.array([crop64[q][0] if q in crop64 else 0.0 for q in qids])
    print(f"\n===== {guide} guide pass: {ev.sum()} evaluation questions, guide accuracy {100 * g['p1_acc'][ev].mean():.2f} =====")
    for name in ("p1_first", "p1_mean"):
        s = g[name][ev]
        for tname, lab in (("guide correct (>=0.5)", g["p1_acc"][ev] >= 0.5),
                           ("guide fully correct (==1)", g["p1_acc"][ev] >= 0.999),
                           ("crop@64 beats guide answer", (esc[ev] > g["p1_acc"][ev]))):
            a, lo, hi = auroc_ci(s, lab.astype(int), img)
            print(f"  {name:9s} -> {tname:28s} AUROC {a:.3f} [{lo:.3f}, {hi:.3f}]   positives {100 * lab.mean():.0f}%")
    if guide == "g16":
        for name in ("p1_first", "p1_mean"):
            thr = np.quantile(g[name][cal], 0.5)          # escalate the least-confident half of the calibration split
            e = g[name][ev] < thr
            acc = np.where(e, esc[ev], g["p1_acc"][ev]).mean()
            print(f"  operating point from calibration ({name}, threshold {thr:.3f}): "
                  f"escalation rate on evaluation {100 * e.mean():.0f}%, accuracy {100 * acc:.2f}")
