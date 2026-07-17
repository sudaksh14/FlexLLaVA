#!/bin/bash
# Stage 2 — Elastic Visual Instruction Tuning for Small LLM Backbones
#
# Usage:
#   bash scripts/v1_5/finetune_elastic_slm.sh <LLM_KEY>
#
# LLM_KEY choices: tinyllama | mobilellama | smollm2 | qwen0.5b | qwen1.5b | qwen3b | phi2 | stablelm
#
# --pretrain_elastic_path points to the Stage 1 output so elastic_resampler,
# elastic_projector, and LoRA weights are warm-started.
#
# In Stage 2 the LLM is UNFROZEN (freeze_backbone not set) alongside the
# elastic modules.  ZeRO-3 is required to shard the full LLM gradients.

LLM_KEY=${1:-"qwen0.5b"}

case "$LLM_KEY" in
  qwen0.5b)
    MODEL_PATH="Qwen/Qwen2.5-0.5B-Instruct"
    CONV_VERSION="chatml"
    ;;
  qwen1.5b)
    MODEL_PATH="Qwen/Qwen2.5-1.5B-Instruct"
    CONV_VERSION="chatml"
    ;;
  qwen3b)
    MODEL_PATH="Qwen/Qwen2.5-3B-Instruct"
    CONV_VERSION="chatml"
    ;;
  tinyllama)
    MODEL_PATH="TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    # See scripts/v1_5/pretrain_elastic_slm.sh for why this is v1, not chatml.
    CONV_VERSION="v1"
    ;;
  mobilellama)
    MODEL_PATH="mtgv/MobileLLaMA-1.4B-Chat"
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
  stablelm)
    MODEL_PATH="stabilityai/stablelm-2-zephyr-1_6b"
    CONV_VERSION="chatml"
    ;;
  *)
    echo "Unknown LLM_KEY='$LLM_KEY'. Choose one of: tinyllama mobilellama smollm2 qwen0.5b qwen1.5b qwen3b phi2 stablelm"
    exit 1
    ;;
esac

PRETRAIN_CKPT="/var/scratch/skalra/flexllava/checkpoints/elastic-pretrain-${LLM_KEY}"
OUTPUT_DIR="/var/scratch/skalra/flexllava/checkpoints/elastic-finetune-${LLM_KEY}"
LOG_DIR="/var/scratch/skalra/flexllava/logs/elastic-finetune-${LLM_KEY}"
RUN_NAME="elastic-finetune-${LLM_KEY}-tok256-144-64-16"

echo "[FlexLLaVA] Finetune  LLM=${MODEL_PATH}  conv=${CONV_VERSION}"
echo "[FlexLLaVA] Pretrain checkpoint → ${PRETRAIN_CKPT}"
echo "[FlexLLaVA] Output             → ${OUTPUT_DIR}"

deepspeed --num_gpus 2 llava/train/train_elastic.py \
    --tok_levels 256 144 64 16 \
    --lora_ranks 8 16 32 64 \
    --prefix_kl_weight 0.1 \
    --coral_weight 0.1 \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path "${MODEL_PATH}" \
    --pretrain_elastic_path "${PRETRAIN_CKPT}" \
    --cache_dir /var/scratch/skalra/.cache/huggingface/hub \
    --version "${CONV_VERSION}" \
    --data_path /var/scratch/skalra/flexllava/data/LLaVA-Finetune/llava_v1_5_mix665k.json \
    --image_folder /var/scratch/skalra/flexllava/data/LLaVA-Finetune \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir "${OUTPUT_DIR}" \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 32 \
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
    --model_max_length 1024 \
    --gradient_checkpointing True \
    --gradient_checkpointing_kwargs '{"use_reentrant": false}' \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb \
    --run_name "${RUN_NAME}" \
    --logging_dir "${LOG_DIR}"
