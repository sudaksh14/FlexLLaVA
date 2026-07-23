#!/bin/bash
# Stage 2 — Elastic Visual Instruction Tuning for SLMs, FULL fine-tune via PURE
# PyTorch DDP (no DeepSpeed).
#
# Why this variant exists:
#   Full fine-tuning the SLM under DeepSpeed ZeRO-2 deadlocks deterministically at
#   step 136 -- the embed_tokens gradient all-reduce desyncs across ranks on a
#   data-dependent batch (see run_26205/26209/26213). This is a known DeepSpeed
#   ZeRO-2 class of bug (DeepSpeed issue #7044; LLaVA issue #1759), and the
#   community workaround is to drop DeepSpeed and use plain PyTorch DDP, whose
#   gradient reduction is a single deterministic collective rather than ZeRO's
#   overlapped per-bucket scheme.
#
# So this launches with torchrun (HF Trainer -> DDP) and passes NO --deepspeed.
# A 1.1B SLM full fine-tune fits on a single A40 (46 GB) without ZeRO sharding:
#   weights ~2.2G + grads ~2.2G + Adam states ~13G + vision/elastic + activations.
#
# Usage:  bash scripts/v1_5/finetune_elastic_slm_ddp.sh <LLM_KEY>
# NOTE: this is EXPERIMENTAL. If DDP errors on unused params, toggle
#       --ddp_find_unused_parameters; if it OOMs, lower per_device_train_batch_size
#       and raise gradient_accumulation_steps to keep the global batch size fixed.

LLM_KEY=${1:-"tinyllama"}
# $2 toggles DDP unused-param detection. Default True (safe for the elastic
# model's conditional branches). Pass False for the faster path / if True throws
# a "parameter marked ready twice" error under gradient checkpointing.
FIND_UNUSED=${2:-True}

case "$LLM_KEY" in
  qwen0.5b)    MODEL_PATH="Qwen/Qwen2.5-0.5B-Instruct";        CONV_VERSION="chatml" ;;
  qwen1.5b)    MODEL_PATH="Qwen/Qwen2.5-1.5B-Instruct";        CONV_VERSION="chatml" ;;
  qwen3b)      MODEL_PATH="Qwen/Qwen2.5-3B-Instruct";          CONV_VERSION="chatml" ;;
  tinyllama)   MODEL_PATH="TinyLlama/TinyLlama-1.1B-Chat-v1.0"; CONV_VERSION="v1" ;;
  mobilellama) MODEL_PATH="mtgv/MobileLLaMA-1.4B-Chat";        CONV_VERSION="v1" ;;
  smollm2)     MODEL_PATH="HuggingFaceTB/SmolLM2-1.7B-Instruct"; CONV_VERSION="chatml" ;;
  phi2)        MODEL_PATH="microsoft/phi-2";                   CONV_VERSION="phi" ;;
  stablelm)    MODEL_PATH="stabilityai/stablelm-2-zephyr-1_6b"; CONV_VERSION="chatml" ;;
  *)
    echo "Unknown LLM_KEY='$LLM_KEY'. Choose one of: tinyllama mobilellama smollm2 qwen0.5b qwen1.5b qwen3b phi2 stablelm"
    exit 1 ;;
esac

PRETRAIN_CKPT="/var/scratch/skalra/flexllava/checkpoints/elastic-pretrain-${LLM_KEY}"
OUTPUT_DIR="/var/scratch/skalra/flexllava/checkpoints/elastic-finetune-${LLM_KEY}-ddp"
LOG_DIR="/var/scratch/skalra/flexllava/logs/elastic-finetune-${LLM_KEY}-ddp"
RUN_NAME="elastic-finetune-${LLM_KEY}-ddp-tok256-144-64-16"

echo "[FlexLLaVA] Finetune (FULL, pure DDP)  LLM=${MODEL_PATH}  conv=${CONV_VERSION}"
echo "[FlexLLaVA] Pretrain checkpoint → ${PRETRAIN_CKPT}"
echo "[FlexLLaVA] Output             → ${OUTPUT_DIR}"

# Pure PyTorch DDP: torchrun instead of deepspeed, and NO --deepspeed flag.
torchrun --standalone --nnodes=1 --nproc_per_node=2 llava/train/train_elastic.py \
    --tok_levels 256 144 64 16 \
    --lora_ranks 8 16 32 64 \
    --prefix_kl_weight 0.1 \
    --coral_weight 0.1 \
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
    --model_max_length 2048 \
    --ddp_timeout 1800 \
    --ddp_find_unused_parameters ${FIND_UNUSED} \
    --gradient_checkpointing True \
    --gradient_checkpointing_kwargs '{"use_reentrant": false}' \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb \
    --run_name "${RUN_NAME}" \
    --logging_dir "${LOG_DIR}"
