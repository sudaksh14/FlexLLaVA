#!/bin/bash
#SBATCH --job-name=eval_m3_ladder_hipster
#SBATCH -t 24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
# Overridden at submit time by submit_eval_m3_ladder_hipster.sh.
#SBATCH --partition=capacity
#SBATCH --gres=gpu:l4:1
#SBATCH --cpus-per-task=16
#SBATCH --array=0-3
#SBATCH --output=./jobs/eval_m3_ladder_hipster_%A_%a.out
# Hipster port of eval_lmms_ladder_m3.sh: budget-matched M3 evaluation, one array
# task per budget in {256,144,64,16}, on the checkpoint trained with
# MATRYOSHKA_SCALE=256,144,64,16 (M3's own joint scale loop). Same six tasks as the
# DAS-6 script (Table III columns + vqav2_val).
#
# The checkpoint is a PURE M3 (no elastic_config.json), so the generic `llava`
# wrapper is used, not `llava_elastic`. That wrapper reads
# config.matryoshka_vis_token_scale, which for this checkpoint is the 4-element
# list, and llava_arch.matryoshka_vis_token_process (rightly) rejects a list at
# inference -- so the budget is passed explicitly via model_args.
#
#   sbatch [--dependency=afterok:<train job>] eval_lmms_ladder_m3_hipster.sh [checkpoint_dir]
BUDGETS=(256 144 64 16)
N=${BUDGETS[$SLURM_ARRAY_TASK_ID]}
MODEL_PATH="${1:-/scratch/skalra/flexllava_saves/checkpoints/baseline-tinyllama-ladder-finetune}"
MODEL_TAG="$(basename "$MODEL_PATH")"
TASKS="${TASKS:-mme,pope,scienceqa_img,textvqa_val,gqa,vqav2_val}"
OUTDIR=${OUTROOT:-/home/skalra/flexllava_saves/eval_logs}/${MODEL_TAG}/${N}tok

module load cuda/12.9.1 2>&1 || echo "WARNING: module load cuda/12.9.1 failed -- continuing"
eval "$(conda shell.bash hook)"
conda activate matryoshka-mm
export HF_HOME=/home/skalra/flexllava_saves/cache/huggingface
export HF_DATASETS_CACHE=/home/skalra/flexllava_saves/cache/huggingface/datasets
cd /home/skalra/FlexLLaVA

if [ ! -f "${MODEL_PATH}/config.json" ]; then
    echo "ERROR: ${MODEL_PATH}/config.json not found." >&2
    exit 1
fi

echo "Job started: $(date)  node=$(hostname)  budget=${N}  model=${MODEL_PATH}"
nvidia-smi | head -12
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
# rtx_6000_ada (48GB) -> 8, l4 (24GB) -> 4, else 1 (same table as eval_lmms_level_hipster.sh)
if echo "$GPU_NAME" | grep -qi "6000 Ada"; then BATCH_SIZE="${BATCH_SIZE:-8}"
elif echo "$GPU_NAME" | grep -qi "L4"; then BATCH_SIZE="${BATCH_SIZE:-4}"
else BATCH_SIZE="${BATCH_SIZE:-1}"; fi
echo "GPU: $GPU_NAME -> batch_size=${BATCH_SIZE}  tasks=${TASKS}"

# 4 array tasks may land on different nodes sharing one NFS conda env; serialise a
# possible first-time `pip install -e` with an atomic mkdir lock (see
# eval_lmms_level_hipster.sh for the 2026-09-13 failure this guards against).
LOCKDIR=/home/skalra/flexllava_saves/.lmms_eval_install.lock
LOCK_ACQUIRED=0
for i in $(seq 1 300); do
    if mkdir "$LOCKDIR" 2>/dev/null; then
        LOCK_ACQUIRED=1
        pip show lmms-eval >/dev/null 2>&1 || pip install -e lmms-eval -q
        rmdir "$LOCKDIR" 2>/dev/null
        break
    fi
    sleep 1
done
if [ "$LOCK_ACQUIRED" -ne 1 ]; then
    echo "ERROR: could not acquire $LOCKDIR after 300s." >&2
    exit 1
fi

mkdir -p "$OUTDIR"

accelerate launch --num_processes=1 -m lmms_eval \
    --model llava \
    --model_args "pretrained=${MODEL_PATH},conv_template=vicuna_v1,matryoshka_vis_token_scale=${N}" \
    --tasks "$TASKS" --batch_size $BATCH_SIZE \
    ${LIMIT:+--limit ${LIMIT}} --log_samples --log_samples_suffix "m3_ladder_${N}tok" --output_path "$OUTDIR"
echo "Done: $(date)"
