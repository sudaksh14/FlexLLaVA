"""Build the question_id -> GT answer box sidecar for the crop pilot.

Joins the TextVQA val split cached for lmms-eval (lmms-lab/textvqa, 5000 q's,
no boxes) with ViCrop's released ground-truth boxes (jrzhang/TextVQA_GT_bbox,
4370 q's; bbox = [x, y, w, h] in image pixels).  Size bins follow ViCrop Sec. 3:
S = box_area / image_area; small < 0.005 <= medium < 0.05 <= large.

CPU only.  Run through sbatch (see build_sidecar.sh).
"""
import io
import json
import os
import sys
from collections import Counter

os.environ.setdefault("HF_HOME", "/var/scratch/skalra/.cache/huggingface")
CACHE = "/var/scratch/skalra/.cache/huggingface/datasets"
OUT = sys.argv[1] if len(sys.argv) > 1 else "/var/scratch/skalra/flexllava/croppilot/textvqa_val_gtbox.json"

import datasets
from PIL import Image

val = datasets.load_dataset("lmms-lab/textvqa", split="validation", cache_dir=CACHE)
gt = datasets.load_dataset("jrzhang/TextVQA_GT_bbox", split="train", cache_dir=CACHE)
gt = gt.cast_column("image", datasets.Image(decode=False))  # read sizes without decoding pixels

val_q = val["question_id"]
val_w, val_h = val["image_width"], val["image_height"]
val_text = val["question"]
pos = {int(q): i for i, q in enumerate(val_q)}

side, stats = {}, Counter()
for i in range(len(gt)):
    r = gt[i]
    qid = int(r["dataset_id"])
    if qid not in pos:
        stats["not_in_val"] += 1
        continue
    if qid in side:
        stats["duplicate_qid"] += 1
        continue
    j = pos[qid]
    W, H = val_w[j], val_h[j]
    gw, gh = Image.open(io.BytesIO(r["image"]["bytes"])).size
    if (gw, gh) != (W, H):
        stats["size_mismatch"] += 1
        continue
    if r["question"].strip().lower() != val_text[j].strip().lower():
        stats["question_text_mismatch"] += 1  # counted, not dropped: ids + sizes agree
    x, y, w, h = [float(v) for v in r["bbox"]]
    x0, y0 = max(0.0, x), max(0.0, y)
    x1, y1 = min(float(W), x + w), min(float(H), y + h)
    if x1 - x0 < 1 or y1 - y0 < 1:
        stats["degenerate_box"] += 1
        continue
    frac = (x1 - x0) * (y1 - y0) / (W * H)
    b = "small" if frac < 0.005 else ("medium" if frac < 0.05 else "large")
    side[str(qid)] = {"bbox_xywh": [x0, y0, x1 - x0, y1 - y0], "img_w": W, "img_h": H,
                      "frac": frac, "bin": b}
    stats["kept"] += 1

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w") as f:
    json.dump(side, f)
print("val questions:", len(val), " gt rows:", len(gt))
print("stats:", dict(stats))
print("bins:", dict(Counter(v["bin"] for v in side.values())), "(ViCrop reports 773 / 2411 / 1186 small/medium/large)")
print("wrote", OUT)
