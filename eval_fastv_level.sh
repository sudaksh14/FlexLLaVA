#!/bin/bash
#SBATCH --job-name=eval_fastv
#SBATCH -t 24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --exclude=node205
#SBATCH --output=./jobs/eval_fastv_%A_tok%a.out
#SBATCH --array=0-3

# I-1: FastV (training-free visual-token pruning) on a DENSE 576-token checkpoint, one token budget per array task.
# Budgets match the Nest-VLM ladder (256/144/64/16). FastV keeps all 576 tokens in the first K layers and keeps
# round(n) of them from layer K on, i.e. ratio R = 1 - n/576 (llava/model/fastv.py, mask-based, accuracy-faithful).
#
#   sbatch eval_fastv_level.sh [dense_model_path]       # K=2 by default; K=4 sbatch --export=ALL,FASTV_K=4 ...
#   sbatch --array=2 eval_fastv_level.sh                # one budget
# Reference for the same backbone/data: dense baseline = baseline-tinyllama-576tok (eval_lmms_baseline_llava.sh).

MODEL_PATH=${1:-/var/scratch/skalra/flexllava/checkpoints/baseline-tinyllama-576tok}
FASTV_K=${FASTV_K:-2}
BUDGETS=(256 144 64 16)
NTOK=${BUDGETS[$SLURM_ARRAY_TASK_ID]}
[ -n "$NTOK" ] || { echo "ERROR: array index ${SLURM_ARRAY_TASK_ID} outside 0-3" >&2; exit 1; }
FASTV_RATIO=$(python3 -c "print(1 - ${NTOK}/576)")
MODEL_TAG="$(basename "$MODEL_PATH")-fastv-k${FASTV_K}"
LABEL="${NTOK}tok"
LOG_ROOT=/var/scratch/skalra/flexllava/eval_logs
OUTDIR="${LOG_ROOT}/${MODEL_TAG}/${LABEL}"
TASKS="${TASKS:-mme,pope,scienceqa_img,textvqa_val,gqa}"
LIMIT_ARG=""
[ -n "$EVAL_LIMIT" ] && LIMIT_ARG="--limit $EVAL_LIMIT" && echo "WARNING: --limit $EVAL_LIMIT: subsample, not comparable"

module load cuda12.6/toolkit/12.6
eval "$(conda shell.bash hook)"
conda activate matryoshka-mm
export HF_HOME=/var/scratch/skalra/.cache/huggingface
export HF_DATASETS_CACHE=/var/scratch/skalra/.cache/huggingface/datasets
cd /home/skalra/FlexLLaVA

GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
if echo "$GPU_NAME" | grep -q "A40"; then BATCH_SIZE=${BATCH_SIZE:-8}
elif echo "$GPU_NAME" | grep -q "A10"; then BATCH_SIZE=${BATCH_SIZE:-4}
else BATCH_SIZE=${BATCH_SIZE:-1}; fi

# Same conv-template detection as eval_lmms_level.sh / eval_lmms_baseline_llava.sh.
BASE_LLM=$(python3 -c "
import json
try:
    print(json.load(open('${MODEL_PATH}/config.json')).get('_name_or_path', '').lower())
except Exception:
    print('')
")
case "$BASE_LLM" in
    *phi-3*|*phi3*)                       CONV_TEMPLATE="phi3" ;;
    *tinyllama*|*mobilellama*)            CONV_TEMPLATE="vicuna_v1" ;;
    *qwen*|*stablelm*|*smollm*)           CONV_TEMPLATE="chatml" ;;
    *phi*)                                CONV_TEMPLATE="phi" ;;
    *)                                    CONV_TEMPLATE="vicuna_v1" ;;
esac

echo "Job started: $(date)  node: $(hostname)  GPU: $GPU_NAME  batch=${BATCH_SIZE}"
echo "Model: $MODEL_PATH  FastV K=${FASTV_K} ratio=${FASTV_RATIO} (keeps ${NTOK}/576 visual tokens from layer K)  conv=${CONV_TEMPLATE}"
pip show lmms-eval >/dev/null 2>&1 || pip install -e lmms-eval -q
mkdir -p "$OUTDIR"

accelerate launch --num_processes=1 \
    -m lmms_eval \
    --model       llava \
    --model_args  "pretrained=${MODEL_PATH},conv_template=${CONV_TEMPLATE},attn_implementation=eager,fastv_k=${FASTV_K},fastv_ratio=${FASTV_RATIO}" \
    --tasks       "$TASKS" \
    --batch_size  $BATCH_SIZE \
    $LIMIT_ARG \
    --log_samples \
    --log_samples_suffix "fastv_${LABEL}" \
    --output_path "$OUTDIR"

echo "Done ${LABEL}: $(date)   results: $OUTDIR"
