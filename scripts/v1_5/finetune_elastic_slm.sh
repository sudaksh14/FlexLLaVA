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
# elastic modules.
#
# ── 2026-07-31: full finetune, and why the hyperparameters changed ──────────
# The previous settings (LR 2e-4, lora_enable True, lora_alpha 256 / r 128 ->
# LoRA scale 2.0) diverged: run_26226 trained cleanly to loss ~1.14 then blew
# up (16 -> 456 -> ... -> 21589) about a quarter of an epoch in, and because
# save_total_limit=1 kept only post-divergence checkpoints, every later run
# that resumed inherited the wreckage (run_26233 started at loss ~28000).
#
# Compare the 7B script (scripts/v1_5/finetune_elastic.sh): LR 2e-5, alpha
# 128 / r 128 -> scale 1.0. That is a ~20x smaller effective update, on a 6x
# LARGER model. The 7B run was never "more stable because it is bigger" -- it
# was simply configured far more conservatively. Nothing was clipping either
# run: scripts/zero{2,3}.json were missing the "gradient_clipping" key, so
# max_grad_norm never reached DeepSpeed (now fixed).
#
# Now: full finetune (--lora_enable False) at LR 2e-5. At 1.1B params LoRA
# buys little, it removes the alpha/r scale as a failure mode, and it matches
# the protocol of AdaLLaVA, LLaVA-1.5 stage 2, TinyLLaVA and MobileVLM.
# --lora_r / --lora_alpha below are inert while --lora_enable is False; they
# are left at the 7B-matching scale 1.0 in case LoRA is switched back on.
#
# NOTE: --lora_enable controls PEFT LoRA on the *LLM* only. The rank-nested
# LoRA on the vision tower -- the actual elastic mechanism -- is controlled by
# --vision_lora_enable (default True) and stays ON.

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

# Optional run tag -- must match the ELASTIC_RUN_TAG used for Stage 1, since it
# selects both the warm-start checkpoint and this run's output dir.
TAG="${ELASTIC_RUN_TAG:+-${ELASTIC_RUN_TAG}}"

PRETRAIN_CKPT="/var/scratch/skalra/flexllava/checkpoints/elastic-pretrain-${LLM_KEY}${TAG}"
OUTPUT_DIR="/var/scratch/skalra/flexllava/checkpoints/elastic-finetune-${LLM_KEY}${TAG}"
LOG_DIR="/var/scratch/skalra/flexllava/logs/elastic-finetune-${LLM_KEY}${TAG}"
RUN_NAME="elastic-finetune-${LLM_KEY}-tok256-144-64-16${TAG}"

echo "[FlexLLaVA] Finetune  LLM=${MODEL_PATH}  conv=${CONV_VERSION}"
echo "[FlexLLaVA] Pretrain checkpoint → ${PRETRAIN_CKPT}"
echo "[FlexLLaVA] Output             → ${OUTPUT_DIR}"

deepspeed --num_gpus 2 llava/train/train_elastic.py \
    --tok_levels 256 144 64 16 \
    --lora_ranks 8 16 32 64 \
    --prefix_kl_weight 0.1 \
    --coral_weight 0.1 \
    --use_coral False \
    --use_pos_embed True \
    --pos_embed_type learned \
    --use_nested_dropout False \
    --n_sample_students 1 \
    --vision_lora_enable False \
    --lora_enable False \
    --lora_r 128 \
    --lora_alpha 128 \
    --mm_projector_lr 2e-5 \
    --mm_vision_tower_lr 2e-5 \
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
    --save_total_limit 5 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --ddp_timeout 900 \
    --gradient_checkpointing True \
    --gradient_checkpointing_kwargs '{"use_reentrant": false}' \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb \
    --run_name "${RUN_NAME}" \
    --logging_dir "${LOG_DIR}"
