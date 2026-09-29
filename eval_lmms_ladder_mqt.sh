#!/bin/bash
#SBATCH --job-name=eval_mqt_ladder
#SBATCH -t 24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --array=0-3
#SBATCH --output=./jobs/eval_mqt_ladder_%A_%a.out
# Budget-matched MQT-LLaVA evaluation: one array task per budget in {256,144,64,16}, on the
# checkpoint trained with num_visual_tokens=256,144,64,16 (random budget per step). The budget
# is passed explicitly (num_visual_tokens=N); do not rely on the checkpoint's config.json.
# Requires MQT's own `llava` first on sys.path: PYTHONPATH set before python starts AND cwd
# = MQT-LLaVA (python -m / -c put cwd at sys.path[0], which beats PYTHONPATH; see
# eval_lmms_baseline_mqt.sh).
#
#   sbatch [--dependency=afterok:<train job>] eval_lmms_ladder_mqt.sh [checkpoint_dir]
BUDGETS=(256 144 64 16)
N=${BUDGETS[$SLURM_ARRAY_TASK_ID]}
MODEL_PATH="${1:-/var/scratch/skalra/flexllava/checkpoints/mqt-finetune-tinyllama-ladder}"
MODEL_TAG="$(basename "$MODEL_PATH")"
TASKS="${TASKS:-mme,pope,scienceqa_img,textvqa_val,gqa,vqav2_val}"
OUTDIR=${OUTROOT:-/var/scratch/skalra/flexllava/eval_logs}/${MODEL_TAG}/${N}tok

module load cuda12.6/toolkit/12.6
eval "$(conda shell.bash hook)"
conda activate matryoshka-mm
export HF_HOME=/var/scratch/skalra/.cache/huggingface
export HF_DATASETS_CACHE=/var/scratch/skalra/.cache/huggingface/datasets
export PYTHONPATH=/home/skalra/FlexLLaVA/MQT-LLaVA${PYTHONPATH:+:${PYTHONPATH}}
cd /home/skalra/FlexLLaVA/MQT-LLaVA

echo "Job started: $(date)  node=$(hostname)  budget=${N}  model=${MODEL_PATH}"
_resolved=$(python3 -c "import llava, os; print(os.path.dirname(llava.__file__))")
if [ "$_resolved" != "/home/skalra/FlexLLaVA/MQT-LLaVA/llava" ]; then
    echo "ERROR: 'llava' resolves to ${_resolved}, not MQT's." >&2; exit 1
fi
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
if echo "$GPU_NAME" | grep -q "A40"; then BATCH_SIZE="${BATCH_SIZE:-4}"
elif echo "$GPU_NAME" | grep -q "A10"; then BATCH_SIZE="${BATCH_SIZE:-2}"
else BATCH_SIZE="${BATCH_SIZE:-1}"; fi
echo "GPU: $GPU_NAME -> batch_size=${BATCH_SIZE}  tasks=${TASKS}"
pip show lmms-eval >/dev/null 2>&1 || pip install -e /home/skalra/FlexLLaVA/lmms-eval -q
mkdir -p "$OUTDIR"

python3 -m lmms_eval \
    --model llava_mqt \
    --model_args "pretrained=${MODEL_PATH},conv_template=vicuna_v1,num_visual_tokens=${N}" \
    --tasks "$TASKS" --batch_size $BATCH_SIZE \
    ${LIMIT:+--limit ${LIMIT}} --log_samples --log_samples_suffix "mqt_ladder_${N}tok" --output_path "$OUTDIR"
echo "Done: $(date)"
