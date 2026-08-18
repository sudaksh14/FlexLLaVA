#!/bin/bash
# Stage 2 — NON-ELASTIC control, TinyLLaVA-equivalent. See the header of
# pretrain_baseline_slm.sh for what this control is for.
#
#   bash scripts/v1_5/finetune_baseline_slm.sh [LLM_KEY]
#
# Mirrors TinyLLaVA's scripts/train/finetune.sh: effective batch 128 (they use
# 4 GPUs x 8 x 4; we use 2 x 2 x 32), LR 2e-5, 1 epoch, full LLM finetune with
# the vision tower FROZEN, model_max_length 2048.
#
# Two flags from our scripts/v1_5/finetune.sh are deliberately absent:
#   --matryoshka_vis_token_scale : leaving it None keeps all 576 tokens. Setting
#                                  it would reintroduce the very compression
#                                  this control exists to measure against.
#   --unfreeze_mm_vision_tower   : TinyLLaVA keeps the ViT frozen in Stage 2.

LLM_KEY=${1:-"tinyllama"}

case "$LLM_KEY" in
  tinyllama)
    MODEL_PATH="TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    # v1 == TinyLLaVA's "llama" template (vicuna-style USER/ASSISTANT + </s>).
    CONV_VERSION="v1"
    ;;
  smollm2)
    MODEL_PATH="HuggingFaceTB/SmolLM2-1.7B-Instruct"
    CONV_VERSION="chatml"
    ;;
  phi2)
    MODEL_PATH="microsoft/phi-2"
    CONV_VERSION="phi"
    ;;
  *)
    echo "Unknown LLM_KEY='$LLM_KEY' for the baseline control (tinyllama|smollm2|phi2)"
    exit 1
    ;;
esac

TAG="${BASELINE_RUN_TAG:+-${BASELINE_RUN_TAG}}"
PRETRAIN_DIR="/var/scratch/skalra/flexllava/checkpoints/baseline-${LLM_KEY}${TAG}-pretrain"
OUTPUT_DIR="/var/scratch/skalra/flexllava/checkpoints/baseline-${LLM_KEY}${TAG}-finetune"
LOG_DIR="/var/scratch/skalra/flexllava/logs/baseline-${LLM_KEY}${TAG}-finetune"
RUN_NAME="baseline-${LLM_KEY}-576tok${TAG}-finetune"

# Effective batch = per_device * accum * gpus = 2 * 32 * 2 = 128, matching
# TinyLLaVA. Same self-checking form as Stage 1: derive accum from PER_DEVICE
# and print what the product actually is, rather than asserting it in a comment.
NUM_GPUS="${NUM_GPUS:-2}"
PER_DEVICE="${PER_DEVICE:-2}"
GRAD_ACCUM="${GRAD_ACCUM:-$(( 128 / (PER_DEVICE * NUM_GPUS) ))}"
EFF_BATCH=$(( PER_DEVICE * GRAD_ACCUM * NUM_GPUS ))
echo "[baseline] num_gpus=${NUM_GPUS} per_device=${PER_DEVICE} grad_accum=${GRAD_ACCUM} -> effective batch ${EFF_BATCH} (target 128)"
echo "[baseline] Finetune LLM=${MODEL_PATH}  conv=${CONV_VERSION}  576 tokens"
echo "[baseline] Warm start ← ${PRETRAIN_DIR}/mm_projector.bin"
echo "[baseline] Output    → ${OUTPUT_DIR}"

if [ ! -f "${PRETRAIN_DIR}/mm_projector.bin" ]; then
    echo "[baseline] ERROR: ${PRETRAIN_DIR}/mm_projector.bin not found."
    echo "[baseline] Stage 1 must finish first (tune_mm_mlp_adapter writes it)."
    exit 1
fi

deepspeed --num_gpus ${NUM_GPUS} llava/train/train_mem.py \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path "${MODEL_PATH}" \
    --cache_dir /var/scratch/skalra/.cache/huggingface/hub \
    --version "${CONV_VERSION}" \
    --data_path /var/scratch/skalra/flexllava/data/LLaVA-Finetune/llava_v1_5_mix665k.json \
    --image_folder /var/scratch/skalra/flexllava/data/LLaVA-Finetune \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --pretrain_mm_mlp_adapter "${PRETRAIN_DIR}/mm_projector.bin" \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio square \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir "${OUTPUT_DIR}" \
    --num_train_epochs 1 \
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
    --dataloader_num_workers 8 \
    --lazy_preprocess True \
    --report_to wandb \
    --run_name "${RUN_NAME}" \
    --logging_dir "${LOG_DIR}"
