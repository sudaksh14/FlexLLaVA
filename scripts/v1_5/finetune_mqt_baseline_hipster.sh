#!/bin/bash
# Stage 2 — MQT-LLaVA baseline at a FIXED query count (default 4), HIPSTER.
#
#   NUM_VISUAL_TOKENS=4 bash scripts/v1_5/finetune_mqt_baseline_hipster.sh tinyllama
#
# MQT's own train loop (see pretrain_mqt_baseline_hipster.sh header, including
# the PYTHONPATH note -- the same guard is repeated below because this script
# is invoked as a separate process).
#
# num_visual_tokens goes straight to MQT's get_matry_n():
#   'second_stage' -> random.choice(range(2,258,2)) per step   (MQT's real recipe)
#   an integer     -> that many queries, every step
# We pass 4, because the ask is a fixed-4-token baseline number, not MQT's full
# elastic curve. SCOPE: that is a deliberate narrowing -- report as "MQT's
# architecture and train loop at a fixed 4-token budget", NOT MQT as published.
#
# MQT has NO mm_projector (diag job 27419), so Stage 1 writes only
# query_abstractor.bin and no --pretrain_mm_mlp_adapter is passed.
#
# Hyperparameters mirror OUR Stage 2: mix665k, LR 2e-5 cosine, warmup 0.03,
# wd 0, 1 epoch, effective batch 128, bf16, max_len 2048, ViT frozen, full LLM
# finetune.
set -uo pipefail
LLM_KEY=${1:-tinyllama}
LOCAL_SSD=${LOCAL_SSD:?LOCAL_SSD must be set by the caller (run_job_hipster_mqt_baseline.sh) to a node-local staged data directory}
SAVE_ROOT=/home/skalra/flexllava_saves
CHECKPOINT_ROOT=/scratch/skalra/flexllava_saves/checkpoints
REPO_ROOT=/home/skalra/FlexLLaVA

case "$LLM_KEY" in
  tinyllama) MODEL_PATH="TinyLlama/TinyLlama-1.1B-Chat-v1.0"; CONV_VERSION="v1" ;;
  *) echo "MQT baseline is set up for tinyllama only; got '$LLM_KEY'" >&2; exit 1 ;;
esac

TAG="${BASELINE_RUN_TAG:+-${BASELINE_RUN_TAG}}"
PRETRAIN_TAG="${BASELINE_PRETRAIN_TAG:+-${BASELINE_PRETRAIN_TAG}}"
: "${PRETRAIN_TAG:=$TAG}"
PRETRAIN_DIR="${CHECKPOINT_ROOT}/mqt-pretrain-${LLM_KEY}${PRETRAIN_TAG}"
OUTPUT_DIR="${CHECKPOINT_ROOT}/mqt-finetune-${LLM_KEY}${TAG}"
LOG_DIR="${SAVE_ROOT}/logs/mqt-finetune-${LLM_KEY}${TAG}"
RUN_NAME="mqt-finetune-${LLM_KEY}${TAG}-hipster"
NUM_VISUAL_TOKENS="${NUM_VISUAL_TOKENS:-4}"

NUM_GPUS="${NUM_GPUS:-4}"
PER_DEVICE="${PER_DEVICE:-2}"
GRAD_ACCUM="${GRAD_ACCUM:-$(( 128 / (PER_DEVICE * NUM_GPUS) ))}"
MASTER_PORT="${MASTER_PORT:-$((29500 + RANDOM % 1000))}"
MAX_STEPS_ARG=""; [ -n "${MAX_STEPS:-}" ] && MAX_STEPS_ARG="--max_steps ${MAX_STEPS}"

if [ ! -f "${PRETRAIN_DIR}/query_abstractor.bin" ]; then
    echo "ERROR: ${PRETRAIN_DIR}/query_abstractor.bin not found -- Stage 1 must finish first." >&2
    ls "${PRETRAIN_DIR}" 2>/dev/null >&2 || true
    exit 1
fi

echo "[mqt-baseline] Stage 2  LLM=${MODEL_PATH}  conv=${CONV_VERSION}  num_visual_tokens=${NUM_VISUAL_TOKENS}"
echo "[mqt-baseline] gpus=${NUM_GPUS} per_device=${PER_DEVICE} accum=${GRAD_ACCUM} -> eff batch $((PER_DEVICE*GRAD_ACCUM*NUM_GPUS))"
echo "[mqt-baseline] warm start <- ${PRETRAIN_DIR}"
echo "[mqt-baseline] out        -> ${OUTPUT_DIR}"
mkdir -p "$LOG_DIR"

cd "${REPO_ROOT}/MQT-LLaVA"
# See pretrain_mqt_baseline_hipster.sh for why this is mandatory: our editable
# install shadows MQT's `llava` unless PYTHONPATH puts MQT first, and the
# failure mode is training OUR model under MQT's script.
export PYTHONPATH=${REPO_ROOT}/MQT-LLaVA${PYTHONPATH:+:${PYTHONPATH}}
_resolved=$(python3 -c "import llava, os; print(os.path.dirname(llava.__file__))")
if [ "$_resolved" != "${REPO_ROOT}/MQT-LLaVA/llava" ]; then
    echo "ERROR: 'llava' resolves to ${_resolved}, not MQT's." >&2
    echo "       Refusing to run: this would train OUR model under MQT's script." >&2
    exit 1
fi
echo "[mqt-baseline] llava package -> ${_resolved} (MQT's, verified)"

deepspeed --num_gpus ${NUM_GPUS} --master_port ${MASTER_PORT} llava/train/train.py \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path "${MODEL_PATH}" \
    --version "${CONV_VERSION}" \
    --data_path ${LOCAL_SSD}/LLaVA-Finetune/llava_v1_5_mix665k.json \
    --image_folder ${LOCAL_SSD}/LLaVA-Finetune \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --mm_query_abstractor_type matry_query \
    --pretrain_mm_query_abstractor "${PRETRAIN_DIR}/query_abstractor.bin" \
    --num_visual_tokens "${NUM_VISUAL_TOKENS}" \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir "${OUTPUT_DIR}" \
    --num_train_epochs 1 \
    ${MAX_STEPS_ARG} \
    --per_device_train_batch_size ${PER_DEVICE} \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps ${GRAD_ACCUM} \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 500 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb \
    --run_name "${RUN_NAME}" \
    --logging_dir "${LOG_DIR}"
