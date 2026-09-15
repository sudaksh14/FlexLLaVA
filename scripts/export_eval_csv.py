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
import csv, glob, json, os, re, sys

LOG_ROOT = "/var/scratch/skalra/flexllava/eval_logs"
EFF_JSON = "results/efficiency_targets.json"
OUT = sys.argv[1] if len(sys.argv) > 1 else "results/flexllava_eval_summary.csv"

MODELS = {
    "llava-elastic-finetune-v3-merged": ("FlexLLaVA-7B", "Vicuna-7B-v1.5", "elastic (nested resampler)"),
    "elastic-finetune-tinyllama-v4":    ("FlexTinyLLaVA-1.1B", "TinyLlama-1.1B-Chat-v1.0", "elastic (nested resampler)"),
    "llava-v1.5-7b-baseline":           ("LLaVA-1.5-7B (reference)", "Vicuna-7B-v1.5", "mlp2x_gelu (no compression)"),
    "baseline-tinyllama-576tok":        ("TinyLLaVA control (ours)", "TinyLlama-1.1B-Chat-v1.0", "mlp2x_gelu (no compression)"),
    "elastic-finetune-smollm2-v4":      ("FlexLLaVA-SmolLM2-1.7B", "SmolLM2-1.7B-Instruct", "elastic (nested resampler)"),
    # v5 = v4 + rank-nested vision LoRA. Kept because it is the matched control
    # for v8 (both vision-LoRA-on); on its own it is a REGRESSION against v4.
    "elastic-finetune-tinyllama-v5":    ("FlexTinyLLaVA-1.1B (v5, vision-LoRA)", "TinyLlama-1.1B-Chat-v1.0", "elastic (nested resampler)"),
    "elastic-finetune-smollm2-v5":      ("FlexLLaVA-SmolLM2-1.7B (v5, vision-LoRA)", "SmolLM2-1.7B-Instruct", "elastic (nested resampler)"),
    # v6 = extended 576..16 ladder. Only its 576/512/448/384 levels exist so far,
    # and those directory names were corrected on 2026-09-09 (see that run's
    # README_RELABEL.txt) -- levels 0-3 had been written under the wrong labels.
    "elastic-finetune-tinyllama-v6-tokrange": ("FlexTinyLLaVA-1.1B (v6, 576-16 ladder)", "TinyLlama-1.1B-Chat-v1.0", "elastic (nested resampler)"),
    # v8 = PARCEL pool-anchored resampler. Current best elastic model.
    "elastic-finetune-tinyllama-v8-parcel": ("FlexTinyLLaVA-1.1B (v8-parcel, BEST)", "TinyLlama-1.1B-Chat-v1.0", "elastic (PARCEL pool-anchored)"),
    # v9 = PARCEL + decorrelation, vision LoRA OFF entirely. Loses to v8 on
    # 4/5 core metrics -- decorrelation does not clear the bar to become
    # default (section 16k), though confounded with the LoRA removal; v11
    # (PARCEL, decorr off, LoRA off) is the clean read once it reports.
    "elastic-finetune-tinyllama-v9-parcel-decorr": ("FlexTinyLLaVA-1.1B (v9, PARCEL+decorr, no LoRA)", "TinyLlama-1.1B-Chat-v1.0", "elastic (PARCEL pool-anchored)"),
    # v10 = PARCEL + a single SHARED rank-16 vision LoRA (not rank-nested),
    # decorrelation off. Clean, unconfounded isolation of LoRA FORM against
    # v8: also loses most of v8's elasticity/accuracy, with no decorr
    # confound -- evidence that v8's full nested-per-level LoRA (not just
    # "any vision LoRA") is doing real work under PARCEL (section 16k).
    "elastic-finetune-tinyllama-v10-parcel-lora16": ("FlexTinyLLaVA-1.1B (v10, PARCEL+shared-r16-LoRA)", "TinyLlama-1.1B-Chat-v1.0", "elastic (PARCEL pool-anchored)"),
    # v11 = PARCEL, no vision LoRA, no decorr, self-teacher. Minimal PARCEL run;
    # the control every other v9-v14 arm isolates against. Evaluated on hipster.
    "elastic-finetune-tinyllama-v11-parcel-nolora": ("FlexTinyLLaVA-1.1B (v11, PARCEL, no LoRA)", "TinyLlama-1.1B-Chat-v1.0", "elastic (PARCEL pool-anchored)"),
    # v13 = v11's recipe (no LoRA) on v6's 8-level 576-16 ladder. Rescues most
    # of v6's failure but still trails v8 -- section 16p.
    "elastic-finetune-tinyllama-v13-parcel-longladder": ("FlexTinyLLaVA-1.1B (v13, PARCEL long-ladder, no LoRA)", "TinyLlama-1.1B-Chat-v1.0", "elastic (PARCEL pool-anchored)"),
}
SUPERSEDED = ["llava-elastic-finetune", "llava-elastic-finetune-v3",
              "elastic-finetune-tinyllama", "elastic-finetune-tinyllama-v3",
              "llava-elastic-pretrain"]

# Derived from the directory name rather than enumerated: eval_lmms_level.sh
# names each level after the checkpoint's own tok_levels entry, so an 8-level
# ladder produces 576tok/512tok/448tok/384tok/... which a hardcoded 4-level map
# would silently emit as a blank n_visual_tokens.
def level_tokens(level):
    m = re.match(r"(\d+)tok", level)
    return int(m.group(1)) if m else ""
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
               "tok_level": level, "n_visual_tokens": level_tokens(level),
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
mo = {"FlexLLaVA-7B": 0, "FlexTinyLLaVA-1.1B": 1, "FlexLLaVA-SmolLM2-1.7B": 2,
      "TinyLLaVA control (ours)": 3,
      "LLaVA-1.5-7B (reference)": 4, "TinyLLaVA (published)": 5}
rows.sort(key=lambda r: (mo.get(r["model"], 9), order.get(r["tok_level"], 9)))

os.makedirs(os.path.dirname(OUT) or ".", exist_ok=True)
with open(OUT, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
    w.writeheader(); w.writerows(rows)
print(f"wrote {len(rows)} rows -> {OUT}")
print("excluded as superseded:", ", ".join(SUPERSEDED))
