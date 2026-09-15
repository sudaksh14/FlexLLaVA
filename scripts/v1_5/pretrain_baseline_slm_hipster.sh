#!/bin/bash
# Stage 1 — M3 baseline (plain projector), HIPSTER variant.
#
# Differences from the DAS-6 scripts/v1_5/pretrain_baseline_slm.sh, and ONLY
# these: data comes from the node-local staged copy ($LOCAL_SSD, set by
# run_job_hipster_m3_baseline.sh), checkpoints go to /scratch and logs to
# /home (per the split adopted 2026-09-11), NUM_GPUS defaults to 4, and the
# run name is suffixed -hipster. Model, data, objective and every
# hyperparameter are identical.
#
# This is ordinary LLaVA Stage-1 pretraining: the projector is trained on all
# 576 CLIP tokens with the LLM and ViT frozen. That is also M3's Stage 1 --
# M3 pools AFTER the projector, so there is nothing matryoshka about Stage 1
# and no scale is passed here. The 4-token setting appears in Stage 2 only.
#
# Not meant to be invoked directly.
set -uo pipefail
LLM_KEY=${1:-tinyllama}
LOCAL_SSD=${LOCAL_SSD:?LOCAL_SSD must be set by the caller (run_job_hipster_m3_baseline.sh) to a node-local staged data directory}
SAVE_ROOT=/home/skalra/flexllava_saves
CHECKPOINT_ROOT=/scratch/skalra/flexllava_saves/checkpoints

case "$LLM_KEY" in
  tinyllama)   MODEL_PATH="TinyLlama/TinyLlama-1.1B-Chat-v1.0" ;;
  smollm2)     MODEL_PATH="HuggingFaceTB/SmolLM2-1.7B-Instruct" ;;
  mobilellama) MODEL_PATH="mtgv/MobileLLaMA-1.4B-Chat" ;;
  *) echo "Unknown LLM_KEY='$LLM_KEY'" >&2; exit 1 ;;
esac

TAG="${BASELINE_RUN_TAG:+-${BASELINE_RUN_TAG}}"
OUTPUT_DIR="${CHECKPOINT_ROOT}/baseline-${LLM_KEY}${TAG}-pretrain"
LOG_DIR="${SAVE_ROOT}/logs/baseline-${LLM_KEY}${TAG}-pretrain"
RUN_NAME="baseline-${LLM_KEY}-576tok${TAG}-pretrain-hipster"

NUM_GPUS="${NUM_GPUS:-4}"
PER_DEVICE="${PER_DEVICE:-8}"
# Effective batch 256, invariant to NUM_GPUS -- same target as DAS-6.
GRAD_ACCUM="${GRAD_ACCUM:-$(( 256 / (PER_DEVICE * NUM_GPUS) ))}"
MASTER_PORT="${MASTER_PORT:-$((29500 + RANDOM % 1000))}"
MAX_STEPS_ARG=""; [ -n "${MAX_STEPS:-}" ] && MAX_STEPS_ARG="--max_steps ${MAX_STEPS}"

echo "[m3-baseline] Stage 1  LLM=${MODEL_PATH}  (576 tokens, no compression)"
echo "[m3-baseline] gpus=${NUM_GPUS} per_device=${PER_DEVICE} accum=${GRAD_ACCUM} -> eff batch $((PER_DEVICE*GRAD_ACCUM*NUM_GPUS))"
echo "[m3-baseline] data -> ${LOCAL_SSD}/LLaVA-Pretrain"
echo "[m3-baseline] out  -> ${OUTPUT_DIR}"
mkdir -p "$LOG_DIR"

deepspeed --num_gpus ${NUM_GPUS} --master_port ${MASTER_PORT} llava/train/train_mem.py \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path "${MODEL_PATH}" \
    --version plain \
    --data_path ${LOCAL_SSD}/LLaVA-Pretrain/blip_laion_cc_sbu_558k.json \
    --image_folder ${LOCAL_SSD}/LLaVA-Pretrain \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --tune_mm_mlp_adapter True \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio square \
    --bf16 True \
    --output_dir "${OUTPUT_DIR}" \
    --num_train_epochs 1 \
    ${MAX_STEPS_ARG} \
    --per_device_train_batch_size ${PER_DEVICE} \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps ${GRAD_ACCUM} \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 500 \
    --save_total_limit 1 \
    --learning_rate 1e-3 \
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
