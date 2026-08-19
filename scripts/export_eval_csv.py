"""Flatten the VALID lmms-eval runs under eval_logs/ into one CSV.

Only the current, trustworthy runs are emitted -- the superseded ones (unmerged
LoRA evaluated with a random projector, the pad==eos TinyLlama, the chatml-masked
runs) are listed in SUPERSEDED below purely so it is obvious they were excluded
on purpose rather than missed.

Efficiency: FLOPs are hardware-independent and come straight from the eval.
prefill_time and the fit flag are NOT -- the evals only ran the cost model for
jetson_orin_nano_8gb. Both targets are read from results/efficiency_targets.json
(debug/compute_efficiency_targets.py), which recomputes Jetson as a check: it
reproduces every stored value to <0.1%, so the A10 column comes from the same
validated code path.

  python3 scripts/export_eval_csv.py [out.csv]
"""
import csv, glob, json, os, sys

LOG_ROOT = "/var/scratch/skalra/flexllava/eval_logs"
EFF_JSON = "results/efficiency_targets.json"
OUT = sys.argv[1] if len(sys.argv) > 1 else "results/flexllava_eval_summary.csv"

MODELS = {
    "llava-elastic-finetune-v3-merged": ("FlexLLaVA-7B", "Vicuna-7B-v1.5", "elastic (nested resampler)"),
    "elastic-finetune-tinyllama-v4":    ("FlexTinyLLaVA-1.1B", "TinyLlama-1.1B-Chat-v1.0", "elastic (nested resampler)"),
    "llava-v1.5-7b-baseline":           ("LLaVA-1.5-7B (reference)", "Vicuna-7B-v1.5", "mlp2x_gelu (no compression)"),
}
SUPERSEDED = ["llava-elastic-finetune", "llava-elastic-finetune-v3",
              "elastic-finetune-tinyllama", "elastic-finetune-tinyllama-v3",
              "elastic-finetune-smollm2-v4", "llava-elastic-pretrain"]

LEVEL_TOKENS = {"256tok": 256, "144tok": 144, "64tok": 64, "16tok": 16, "576tok-native": 576}
METRICS = [("mme/mme_percetion_score", "mme_perception", 1),
           ("mme/mme_cognition_score", "mme_cognition", 1),
           ("pope/pope_accuracy", "pope_acc", 100),
           ("pope/pope_f1_score", "pope_f1", 100),
           ("scienceqa_img/exact_match", "sciqa_img", 100),
           ("textvqa_val/exact_match", "textvqa_val", 100),
           ("gqa/exact_match", "gqa", 100)]

eff = json.load(open(EFF_JSON)) if os.path.exists(EFF_JSON) else {}
rows = []
for tag, (name, backbone, arch) in MODELS.items():
    base = os.path.join(LOG_ROOT, tag)
    if not os.path.isdir(base):
        continue
    for level in sorted(os.listdir(base)):
        found = sorted(glob.glob(os.path.join(base, level, "*", "results.json")))
        if not found:
            continue
        res = json.load(open(found[-1])).get("results", {})
        flat = {f"{t}/{k.split(',')[0]}": v for t, d in res.items() if isinstance(d, dict)
                for k, v in d.items() if isinstance(v, (int, float))}
        row = {"model": name, "backbone": backbone, "architecture": arch,
               "tok_level": level, "n_visual_tokens": LEVEL_TOKENS.get(level, ""),
               "checkpoint": tag, "run_dir": os.path.basename(os.path.dirname(found[-1]))}
        for key, col, sc in METRICS:
            v = flat.get(key)
            row[col] = round(v * sc, 2) if v is not None else ""
        v = flat.get("gqa/prefill_flops")
        row["prefill_tflops"] = round(v / 1e12, 3) if v else ""
        e = eff.get(f"{tag}/{level}", {})
        for hw, pfx in (("jetson_orin_nano_8gb", "jetson"), ("nvidia_A10", "a10")):
            d = e.get(hw, {})
            row[f"{pfx}_prefill_ms"] = round(d["prefill_ms"], 2) if d else ""
            row[f"{pfx}_peak_mem_gb"] = round(d["mem_gb"], 3) if d else ""
            row[f"{pfx}_fits"] = ("yes" if d["fits"] else "no") if d else ""
        rows.append(row)

rows.append({"model": "TinyLLaVA (published)", "backbone": "TinyLlama-1.1B-Chat-v1.0",
             "architecture": "mlp2x_gelu (no compression)", "tok_level": "576tok-native",
             "n_visual_tokens": 576, "mme_perception": 1284.6, "pope_acc": 85.5,
             "sciqa_img": 59.9, "textvqa_val": 46.3, "gqa": 58.0,
             "checkpoint": "TinyLLaVA_Factory/README.md:152", "run_dir": ""})

cols = ["model", "backbone", "architecture", "tok_level", "n_visual_tokens",
        "mme_perception", "mme_cognition", "pope_acc", "pope_f1", "sciqa_img",
        "textvqa_val", "gqa", "prefill_tflops",
        "jetson_prefill_ms", "jetson_peak_mem_gb", "jetson_fits",
        "a10_prefill_ms", "a10_peak_mem_gb", "a10_fits", "checkpoint", "run_dir"]
order = {"256tok": 0, "144tok": 1, "64tok": 2, "16tok": 3, "576tok-native": 4}
mo = {"FlexLLaVA-7B": 0, "FlexTinyLLaVA-1.1B": 1,
      "LLaVA-1.5-7B (reference)": 2, "TinyLLaVA (published)": 3}
rows.sort(key=lambda r: (mo.get(r["model"], 9), order.get(r["tok_level"], 9)))

os.makedirs(os.path.dirname(OUT) or ".", exist_ok=True)
with open(OUT, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
    w.writeheader(); w.writerows(rows)
print(f"wrote {len(rows)} rows -> {OUT}")
print("excluded as superseded:", ", ".join(SUPERSEDED))
