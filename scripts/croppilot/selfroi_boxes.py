"""I-3 stage B: saliency maps -> crop boxes, with held-out calibration.  CPU only.

Reads the stage-A shards, and for each guide budget (16 / 256 visual tokens):
  * builds candidate saliency maps per question: raw per layer, relative per
    layer (raw / generic-prompt saliency, ViCrop rel-att), all-layer raw sum
    (SGL-style) and all-layer relative;
  * turns each into a crop window with ViCrop's rule (square windows of
    336*{1,1.2,...,2} px, best position per size by integral image, then the
    size whose window sum differs most from its neighbours');
  * scores windows against the ViCrop GT answer box: coverage of the GT box,
    IoU, GT-centre-inside;
  * picks the map variant on a seeded 500-question *calibration* split and
    reports on the remaining questions only.
Writes sidecar json files (same schema as the oracle sidecar) for the chosen
variants plus size-matched controls, so stage C can reuse the pilot's tasks.
"""
import glob
import json
import os
import sys

import numpy as np

ROOT = "/var/scratch/skalra/flexllava/croppilot"
SIDECAR = f"{ROOT}/textvqa_val_gtbox.json"
IN_DIR = sys.argv[1] if len(sys.argv) > 1 else f"{ROOT}/selfroi"
OUT_DIR = sys.argv[2] if len(sys.argv) > 2 else ROOT
ASPECT = os.environ.get("IMAGE_ASPECT", "pad")  # pad (LLaVA-1.5 default) | resize
CELL = 8        # px per coarse cell for the window search
GRID = 24
MULTS = [1.0, 1.2, 1.4, 1.6, 1.8, 2.0]   # ViCrop: window = multiple of the 336 px model input
MODEL_RES = 336
N_CALIB, SEED = 500, 0

side = json.load(open(SIDECAR))


def load_guide(n_tok):
    files = sorted(glob.glob(f"{IN_DIR}/guide{n_tok}_shard*of*.npz"))
    if not files:
        return None
    qs, raw, gen, acc, first, mean = [], [], [], [], [], []
    for f in files:
        d = np.load(f)
        qs += d["qids"].tolist()
        raw.append(d["raw"].astype(np.float32))
        gen.append(d["generic"].astype(np.float32))
        acc.append(d["p1_acc"]); first.append(d["p1_first_prob"]); mean.append(d["p1_mean_prob"])
    return dict(qids=qs, raw=np.concatenate(raw), generic=np.concatenate(gen), p1_acc=np.concatenate(acc),
                p1_first=np.concatenate(first), p1_mean=np.concatenate(mean))


def to_image_map(grid, W, H):
    """24x24 saliency (over the padded square) -> coarse map over the image, CELL px per cell."""
    if ASPECT == "pad":
        S = max(W, H)
        ox, oy = (S - W) / 2.0, (S - H) / 2.0
        sx = sy = S / GRID
    else:
        ox = oy = 0.0
        sx, sy = W / GRID, H / GRID
    wc, hc = max(1, int(round(W / CELL))), max(1, int(round(H / CELL)))
    xs = (np.arange(wc) + 0.5) * (W / wc) + ox
    ys = (np.arange(hc) + 0.5) * (H / hc) + oy
    gx = np.clip(xs / sx - 0.5, 0, GRID - 1)
    gy = np.clip(ys / sy - 0.5, 0, GRID - 1)
    x0, y0 = np.floor(gx).astype(int), np.floor(gy).astype(int)
    x1, y1 = np.minimum(x0 + 1, GRID - 1), np.minimum(y0 + 1, GRID - 1)
    fx, fy = (gx - x0)[None, :], (gy - y0)[:, None]
    m = (grid[y0][:, x0] * (1 - fx) * (1 - fy) + grid[y0][:, x1] * fx * (1 - fy)
         + grid[y1][:, x0] * (1 - fx) * fy + grid[y1][:, x1] * fx * fy)
    return m  # (hc, wc)


def window_sizes(W, H):
    out = []
    for m in MULTS:
        w = int(min(round(MODEL_RES * m), W, H))
        if w not in out:
            out.append(w)
    return out


def select_window(m, W, H):
    """ViCrop box selection on a coarse map. Returns (x, y, w, h) in image px."""
    hc, wc = m.shape
    cw, ch = W / wc, H / hc
    ii = np.pad(m, ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    best = None
    for side_px in window_sizes(W, H):
        ww = max(1, min(wc, int(round(side_px / cw))))
        wh = max(1, min(hc, int(round(side_px / ch))))
        sums = ii[wh:, ww:] - ii[:-wh, ww:] - ii[wh:, :-ww] + ii[:-wh, :-ww]   # (hc-wh+1, wc-ww+1)
        yy, xx = np.unravel_index(np.argmax(sums), sums.shape)
        nb = [sums[y, x] for y, x in ((yy - 1, xx), (yy + 1, xx), (yy, xx - 1), (yy, xx + 1))
              if 0 <= y < sums.shape[0] and 0 <= x < sums.shape[1]]
        score = sums[yy, xx] - (np.mean(nb) if nb else 0.0)
        if best is None or score > best[0]:
            best = (score, xx * cw, yy * ch, ww * cw, wh * ch)
    _, x, y, w, h = best
    return float(x), float(y), float(w), float(h)


def metrics(win, gt):
    x, y, w, h = win
    gx, gy, gw, gh = gt
    ix = max(0.0, min(x + w, gx + gw) - max(x, gx))
    iy = max(0.0, min(y + h, gy + gh) - max(y, gy))
    inter = ix * iy
    cov = inter / (gw * gh)
    iou = inter / (w * h + gw * gh - inter)
    cin = float(x <= gx + gw / 2 <= x + w and y <= gy + gh / 2 <= y + h)
    return cov, iou, cin


def centred(cx, cy, w, h, W, H):
    x = min(max(cx - w / 2, 0.0), W - w)
    y = min(max(cy - h / 2, 0.0), H - h)
    return [float(x), float(y), float(w), float(h)]


def variants(L):
    v = [("raw_all", None), ("rel_all", None)]
    v += [(f"raw_L{l}", l) for l in range(L)] + [(f"rel_L{l}", l) for l in range(L)]
    return v


def grid_for(name, lay, raw_q, gen_q):
    eps = 1e-8
    if name == "raw_all":
        s = raw_q.sum(0)
    elif name == "rel_all":
        s = raw_q.sum(0) / (gen_q.sum(0) + eps)
    elif name.startswith("raw_L"):
        s = raw_q[lay]
    else:
        s = raw_q[lay] / (gen_q[lay] + eps)
    return s.reshape(GRID, GRID)


rng_c = np.random.default_rng(SEED)
all_q = sorted(side.keys(), key=int)
calib = set(rng_c.permutation(all_q)[:N_CALIB].tolist())
os.makedirs(OUT_DIR, exist_ok=True)
json.dump(sorted(calib, key=int), open(f"{OUT_DIR}/selfroi_calib_qids.json", "w"))
print(f"calibration questions: {len(calib)}  (reported numbers use the other {len(all_q) - len(calib)})")

chosen_windows = {}
for n_tok in (16, 256):
    G = load_guide(n_tok)
    if G is None:
        print(f"no guide{n_tok} shards in {IN_DIR}")
        continue
    N, L, _ = G["raw"].shape
    print(f"\n=== guide {n_tok} tokens: {N} questions, {L} layers ===")
    vlist = variants(L)
    cov = {v[0]: np.zeros(N) for v in vlist}
    iou = {v[0]: np.zeros(N) for v in vlist}
    cin = {v[0]: np.zeros(N) for v in vlist}
    wins = {v[0]: [None] * N for v in vlist}
    for i, q in enumerate(G["qids"]):
        s = side[str(q)]
        W, H = s["img_w"], s["img_h"]
        for name, lay in vlist:
            win = select_window(to_image_map(grid_for(name, lay, G["raw"][i], G["generic"][i]), W, H), W, H)
            wins[name][i] = win
            cov[name][i], iou[name][i], cin[name][i] = metrics(win, s["bbox_xywh"])
    is_cal = np.array([str(q) in calib for q in G["qids"]])
    names = [v[0] for v in vlist]
    cal_cov = {n: cov[n][is_cal].mean() for n in names}
    best = max(names, key=lambda n: cal_cov[n])
    print(f"{'variant':<10}{'calib cov':>10}{'eval cov':>10}{'eval IoU':>10}{'ctr-in':>8}")
    for n in sorted(names, key=lambda n: -cal_cov[n])[:8] + ["raw_all", "rel_all"]:
        print(f"{n:<10}{100 * cal_cov[n]:>10.1f}{100 * cov[n][~is_cal].mean():>10.1f}"
              f"{100 * iou[n][~is_cal].mean():>10.1f}{100 * cin[n][~is_cal].mean():>8.1f}")
    print(f"selected on calibration: {best}")
    # per-bin view of the selected variant (eval split)
    bins = np.array([side[str(q)]["bin"] for q in G["qids"]])
    for b in ("small", "medium", "large"):
        mk = (~is_cal) & (bins == b)
        print(f"   {b:<7} n={mk.sum():<5} coverage {100 * cov[best][mk].mean():5.1f}  IoU {100 * iou[best][mk].mean():5.1f}"
              f"  centre-in {100 * cin[best][mk].mean():5.1f}")
    chosen_windows[n_tok] = (G["qids"], wins[best])
    tag = f"g{n_tok}"
    sc = {}
    for q, win in zip(G["qids"], wins[best]):
        s = side[str(q)]
        sc[str(q)] = {"bbox_xywh": [float(v) for v in win], "img_w": s["img_w"], "img_h": s["img_h"],
                      "frac": s["frac"], "bin": s["bin"]}
    json.dump(sc, open(f"{OUT_DIR}/textvqa_val_selfroi_{tag}.json", "w"))
    json.dump({"selected": best, "calib_coverage": float(cal_cov[best])}, open(f"{OUT_DIR}/selfroi_{tag}_selected.json", "w"))
    np.savez(f"{OUT_DIR}/selfroi_{tag}_gate.npz", qids=np.array(G["qids"]), p1_acc=G["p1_acc"],
             p1_first=G["p1_first"], p1_mean=G["p1_mean"], is_calib=is_cal)

# size-matched controls, taken from the 16-token guide's windows
if 16 in chosen_windows:
    qids, wins = chosen_windows[16]
    rng = np.random.default_rng(SEED + 1)
    out = {"centre": {}, "random": {}, "gtctr": {}}
    cov_c, cov_r, cov_g = [], [], []
    for q, win in zip(qids, wins):
        s = side[str(q)]
        W, H = s["img_w"], s["img_h"]
        w, h = win[2], win[3]
        gx, gy, gw, gh = s["bbox_xywh"]
        boxes = {"centre": centred(W / 2, H / 2, w, h, W, H),
                 "random": centred(rng.uniform(w / 2, max(w / 2, W - w / 2)), rng.uniform(h / 2, max(h / 2, H - h / 2)), w, h, W, H),
                 "gtctr": centred(gx + gw / 2, gy + gh / 2, w, h, W, H)}
        for k, b in boxes.items():
            out[k][str(q)] = {"bbox_xywh": b, "img_w": W, "img_h": H, "frac": s["frac"], "bin": s["bin"]}
        cov_c.append(metrics(boxes["centre"], s["bbox_xywh"])[0])
        cov_r.append(metrics(boxes["random"], s["bbox_xywh"])[0])
        cov_g.append(metrics(boxes["gtctr"], s["bbox_xywh"])[0])
    for k in out:
        json.dump(out[k], open(f"{OUT_DIR}/textvqa_val_selfroi_{k}.json", "w"))
    print(f"\ncontrols (same window sizes as g16): GT coverage  centre {100 * np.mean(cov_c):.1f}  random {100 * np.mean(cov_r):.1f}  "
          f"GT-centred {100 * np.mean(cov_g):.1f}")
print("wrote sidecars to", OUT_DIR)
