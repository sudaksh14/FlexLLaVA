#!/bin/bash
#SBATCH --job-name=eval_m3_ladder
#SBATCH -t 24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --array=0-3
#SBATCH --output=./jobs/eval_m3_ladder_%A_%a.out
# Budget-matched M3 evaluation: one array task per budget in {256,144,64,16}, on the
# checkpoint trained with MATRYOSHKA_SCALE=256,144,64,16 (M3's own joint scale loop).
# Same six columns as Table III. The generic `llava` wrapper reads
# config.matryoshka_vis_token_scale, which for this checkpoint is the 4-element list, and
# llava_arch.matryoshka_vis_token_process (rightly) rejects a list at inference, so the
# budget is passed explicitly: lmms-eval turns `matryoshka_vis_token_scale=256` into an int.
#
#   sbatch [--dependency=afterok:<train job>] eval_lmms_ladder_m3.sh [checkpoint_dir]
BUDGETS=(256 144 64 16)
N=${BUDGETS[$SLURM_ARRAY_TASK_ID]}
MODEL_PATH="${1:-/var/scratch/skalra/flexllava/checkpoints/baseline-tinyllama-ladder-finetune}"
MODEL_TAG="$(basename "$MODEL_PATH")"
TASKS="${TASKS:-mme,pope,scienceqa_img,textvqa_val,gqa,vqav2_val}"
OUTDIR=${OUTROOT:-/var/scratch/skalra/flexllava/eval_logs}/${MODEL_TAG}/${N}tok

module load cuda12.6/toolkit/12.6
eval "$(conda shell.bash hook)"
conda activate matryoshka-mm
export HF_HOME=/var/scratch/skalra/.cache/huggingface
export HF_DATASETS_CACHE=/var/scratch/skalra/.cache/huggingface/datasets
cd /home/skalra/FlexLLaVA

echo "Job started: $(date)  node=$(hostname)  budget=${N}  model=${MODEL_PATH}"
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
if echo "$GPU_NAME" | grep -q "A40"; then BATCH_SIZE="${BATCH_SIZE:-4}"
elif echo "$GPU_NAME" | grep -q "A10"; then BATCH_SIZE="${BATCH_SIZE:-2}"
else BATCH_SIZE="${BATCH_SIZE:-1}"; fi
echo "GPU: $GPU_NAME -> batch_size=${BATCH_SIZE}  tasks=${TASKS}"
pip show lmms-eval >/dev/null 2>&1 || pip install -e /home/skalra/FlexLLaVA/lmms-eval -q
mkdir -p "$OUTDIR"

accelerate launch --num_processes=1 -m lmms_eval \
    --model llava \
    --model_args "pretrained=${MODEL_PATH},conv_template=vicuna_v1,matryoshka_vis_token_scale=${N}" \
    --tasks "$TASKS" --batch_size $BATCH_SIZE \
    ${LIMIT:+--limit ${LIMIT}} --log_samples --log_samples_suffix "m3_ladder_${N}tok" --output_path "$OUTDIR"
echo "Done: $(date)"
