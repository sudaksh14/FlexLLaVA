#!/bin/bash
#SBATCH --job-name=eval_lmms
#SBATCH -t 08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:A40:1
#SBATCH --cpus-per-task=8
#SBATCH --output=./jobs/eval_lmms_%A.out
#SBATCH --export=ALL,WANDB_API_KEY=dfcd2574507b9ebe69ca13ab6f6925d864e82ee0

# FlexLLaVA evaluation via lmms-eval (same framework as AdaLLaVA).
# Evaluates all 4 token levels on 6 benchmarks; datasets auto-download from HF.
#
# Benchmarks (matching AdaLLaVA paper protocol):
#   mme · pope · mmbench_en_dev · scienceqa_img · textvqa_val · gqa
#
# Data downloads to HF_HOME/datasets (~15 GB total, cached across runs).
# No manual data preparation required.
#
# Submit: sbatch eval_lmms_job.sh [model_path]

MODEL_PATH=${1:-/var/scratch/skalra/flexllava/checkpoints/llava-elastic-pretrain}
LOG_ROOT=/var/scratch/skalra/flexllava/eval_logs
TASKS="mme,pope,mmbench_en_dev,scienceqa_img,textvqa_val,gqa"

module load cuda12.6/toolkit/12.6

eval "$(conda shell.bash hook)"
conda activate matryoshka-mm

export HF_HOME=/var/scratch/skalra/.cache/huggingface
export HF_DATASETS_CACHE=/var/scratch/skalra/.cache/huggingface/datasets

cd /home/skalra/FlexLLaVA

echo "Job started: $(date)"
echo "Node: $(hostname)"
nvidia-smi | head -12
echo "Model: $MODEL_PATH"

# Install lmms-eval into the env if not already done
pip show lmms-eval >/dev/null 2>&1 || pip install -e lmms-eval -q

TOK_LABELS=("256tok" "144tok" "64tok" "16tok")

for LEVEL in 0 1 2 3; do
    LABEL=${TOK_LABELS[$LEVEL]}
    OUTDIR="${LOG_ROOT}/${LABEL}"
    mkdir -p "$OUTDIR"

    echo ""
    echo "══════════════════════════════════════════════════"
    echo "  Evaluating tok_level=${LEVEL}  (${LABEL})"
    echo "══════════════════════════════════════════════════"

    accelerate launch --num_processes=1 \
        -m lmms_eval \
        --model       llava_elastic \
        --model_args  "pretrained=${MODEL_PATH},tok_level=${LEVEL},device_map=cuda:0" \
        --tasks       "$TASKS" \
        --batch_size  1 \
        --log_samples \
        --log_samples_suffix "elastic_${LABEL}" \
        --output_path "$OUTDIR"

    echo "  Done ${LABEL}: $(date)"
done

# ── Print comparison table ────────────────────────────────────────────────────
SUMMARY="./lmms-eval/eval_lmms_summary_${SLURM_JOB_ID}.txt"

python3 - <<'PYEOF' | tee "$SUMMARY"
import json, glob, os

log_root = "/var/scratch/skalra/flexllava/eval_logs"
labels   = ["256tok", "144tok", "64tok", "16tok"]
benchmarks = ["mme", "pope", "mmbench_en_dev", "scienceqa_img", "textvqa_val", "gqa"]

# AdaLLaVA paper numbers (latency=0.85, LLaVA-v1.5-7b, Table 2)
ada = {
    "mme":           "1487.2 / 324.6",
    "pope":          "85.9",
    "mmbench_en_dev":"68.0",
    "scienceqa_img": "70.4",
    "textvqa_val":   "58.1",
    "gqa":           "62.0",
}

def find_result(label, bench):
    pattern = f"{log_root}/{label}/**/*{bench}*.json"
    files = glob.glob(pattern, recursive=True)
    if not files:
        return "N/A"
    data = json.load(open(files[-1]))
    results = data.get("results", {})
    # Pull the first numeric metric value
    for task_key, metrics in results.items():
        if bench.split("_")[0] in task_key.lower():
            for k, v in metrics.items():
                if isinstance(v, (int, float)) and "stderr" not in k:
                    return f"{v:.1f}"
    return "N/A"

print()
print("╔══════════════════╦══════════╦══════════╦══════════╦══════════╦══════════════════╗")
print("║ Benchmark        ║  256tok  ║  144tok  ║   64tok  ║   16tok  ║ AdaLLaVA (0.85) ║")
print("╠══════════════════╬══════════╬══════════╬══════════╬══════════╬══════════════════╣")
for b in benchmarks:
    vals = [find_result(lbl, b) for lbl in labels]
    ada_val = ada.get(b, "—")
    row = f"║ {b:<16} ║ {vals[0]:>8} ║ {vals[1]:>8} ║ {vals[2]:>8} ║ {vals[3]:>8} ║ {ada_val:<16} ║"
    print(row)
print("╚══════════════════╩══════════╩══════════╩══════════╩══════════╩══════════════════╝")
print()
print("Stage-1 (pretrain-only): token ordering 256>144>64>16 validates Matryoshka training.")
print("For full AdaLLaVA comparison run Stage 2 (finetune_elastic.sh) then re-eval.")
print("AdaLLaVA paper: https://arxiv.org/abs/2503.10905")
PYEOF

echo ""
echo "Summary: $SUMMARY"
echo "Job complete: $(date)"
