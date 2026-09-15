#!/bin/bash
# Stage 2 — M3 baseline at a fixed token budget, HIPSTER variant.
#
#   MATRYOSHKA_SCALE=4 bash scripts/v1_5/finetune_baseline_slm_hipster.sh tinyllama
#
# MATRYOSHKA_SCALE is what makes this the M3 baseline rather than the plain
# 576-token control. This repo is an M3 fork with M3 intact underneath, so
# passing a scale list here IS M3's own training loop: with no elastic_engine
# attached, LlavaElasticMixin.forward takes its
#   "# ---- Pure M3 (no elastic engine, explicit scale list)"
# branch, looping over config.matryoshka_vis_token_scale and averaging CE
# across scales, while llava_arch.matryoshka_vis_token_process does M3's
# avg-pool (pool = stride = sqrt(576/scale)). A single-element list trains one
# granularity with one forward per step:
#   MATRYOSHKA_SCALE=4  -> 12x12 avg-pool -> exactly 4 visual tokens
# Unset leaves all 576 tokens (the non-elastic control).
#
# SCOPE: a fixed single scale narrows M3, which normally trains a LIST of
# scales jointly. Report this as "M3's architecture and training loop at a
# fixed 4-token budget", NOT as M3 as published.
#
# Hyperparameters are identical to DAS-6's finetune_baseline_slm.sh; only data
# path ($LOCAL_SSD), checkpoint root (/scratch), NUM_GPUS default (4) and the
# -hipster run-name suffix differ.
set -uo pipefail
LLM_KEY=${1:-tinyllama}
LOCAL_SSD=${LOCAL_SSD:?LOCAL_SSD must be set by the caller (run_job_hipster_m3_baseline.sh) to a node-local staged data directory}
SAVE_ROOT=/home/skalra/flexllava_saves
CHECKPOINT_ROOT=/scratch/skalra/flexllava_saves/checkpoints

case "$LLM_KEY" in
  tinyllama)   MODEL_PATH="TinyLlama/TinyLlama-1.1B-Chat-v1.0";  CONV_VERSION="v1" ;;
  smollm2)     MODEL_PATH="HuggingFaceTB/SmolLM2-1.7B-Instruct"; CONV_VERSION="chatml" ;;
  mobilellama) MODEL_PATH="mtgv/MobileLLaMA-1.4B-Chat";          CONV_VERSION="v1" ;;
  *) echo "Unknown LLM_KEY='$LLM_KEY'" >&2; exit 1 ;;
esac

TAG="${BASELINE_RUN_TAG:+-${BASELINE_RUN_TAG}}"
PRETRAIN_TAG="${BASELINE_PRETRAIN_TAG:+-${BASELINE_PRETRAIN_TAG}}"
: "${PRETRAIN_TAG:=$TAG}"
PRETRAIN_DIR="${CHECKPOINT_ROOT}/baseline-${LLM_KEY}${PRETRAIN_TAG}-pretrain"
OUTPUT_DIR="${CHECKPOINT_ROOT}/baseline-${LLM_KEY}${TAG}-finetune"
LOG_DIR="${SAVE_ROOT}/logs/baseline-${LLM_KEY}${TAG}-finetune"
RUN_NAME="baseline-${LLM_KEY}${TAG}-finetune-hipster"

NUM_GPUS="${NUM_GPUS:-4}"
PER_DEVICE="${PER_DEVICE:-2}"
# Effective batch 128, invariant to NUM_GPUS -- same target as DAS-6.
GRAD_ACCUM="${GRAD_ACCUM:-$(( 128 / (PER_DEVICE * NUM_GPUS) ))}"
MASTER_PORT="${MASTER_PORT:-$((29500 + RANDOM % 1000))}"
MAX_STEPS_ARG=""; [ -n "${MAX_STEPS:-}" ] && MAX_STEPS_ARG="--max_steps ${MAX_STEPS}"

if [ ! -f "${PRETRAIN_DIR}/mm_projector.bin" ]; then
    echo "ERROR: ${PRETRAIN_DIR}/mm_projector.bin not found -- Stage 1 must finish first." >&2
    ls "${PRETRAIN_DIR}" 2>/dev/null >&2 || true
    exit 1
fi

echo "[m3-baseline] Stage 2  LLM=${MODEL_PATH}  conv=${CONV_VERSION}  ${MATRYOSHKA_SCALE:+M3 scale(s)=${MATRYOSHKA_SCALE} (avg-pooled)}${MATRYOSHKA_SCALE:-576 tokens (no compression)}"
echo "[m3-baseline] gpus=${NUM_GPUS} per_device=${PER_DEVICE} accum=${GRAD_ACCUM} -> eff batch $((PER_DEVICE*GRAD_ACCUM*NUM_GPUS))"
echo "[m3-baseline] warm start <- ${PRETRAIN_DIR}"
echo "[m3-baseline] out        -> ${OUTPUT_DIR}"
mkdir -p "$LOG_DIR"

deepspeed --num_gpus ${NUM_GPUS} --master_port ${MASTER_PORT} llava/train/train_mem.py \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path "${MODEL_PATH}" \
    --version "${CONV_VERSION}" \
    ${MATRYOSHKA_SCALE:+--matryoshka_vis_token_scale ${MATRYOSHKA_SCALE}} \
    --data_path ${LOCAL_SSD}/LLaVA-Finetune/llava_v1_5_mix665k.json \
    --image_folder ${LOCAL_SSD}/LLaVA-Finetune \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --pretrain_mm_mlp_adapter "${PRETRAIN_DIR}/mm_projector.bin" \
    --mm_projector_type mlp2x_gelu \
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
