#!/bin/bash
#SBATCH --job-name=eval_lmms
#SBATCH -t 100:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --output=./jobs/eval_lmms_%A_tok%a.out
#SBATCH --array=0-3

# Evaluates one tok_level per SLURM array task.
# %a in the output filename correctly maps to 0-3.
#
# Submit all 4 levels:  sbatch eval_lmms_level.sh [model_path]
# Submit one level:     sbatch --array=2 eval_lmms_level.sh [model_path]

MODEL_PATH=${1:-/var/scratch/skalra/flexllava/checkpoints/llava-elastic-pretrain}
LEVEL=$SLURM_ARRAY_TASK_ID

TOK_LABELS=("256tok" "144tok" "64tok" "16tok")
LABEL=${TOK_LABELS[$LEVEL]}
LOG_ROOT=/var/scratch/skalra/flexllava/eval_logs
OUTDIR="${LOG_ROOT}/${LABEL}"
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
echo "Model:     $MODEL_PATH"
echo "tok_level: $LEVEL ($LABEL)"

# Set batch size based on GPU VRAM: A40=46GB → 8, A10=24GB → 4, fallback → 1
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
if echo "$GPU_NAME" | grep -q "A40"; then
    BATCH_SIZE=8
elif echo "$GPU_NAME" | grep -q "A10"; then
    BATCH_SIZE=4
else
    BATCH_SIZE=1
fi
echo "GPU: $GPU_NAME  →  batch_size=${BATCH_SIZE}"

pip show lmms-eval >/dev/null 2>&1 || pip install -e lmms-eval -q

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
    --batch_size  $BATCH_SIZE \
    --log_samples \
    --log_samples_suffix "elastic_${LABEL}" \
    --output_path "$OUTDIR"

echo ""
echo "Done ${LABEL}: $(date)"
