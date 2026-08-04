#!/bin/bash
# Stage 1 — Elastic Feature Alignment for Small LLM Backbones
#
# Usage:
#   bash scripts/v1_5/pretrain_elastic_slm.sh <LLM_KEY>
#
# LLM_KEY choices:
#   --- 3 SLMs from the reference repos ---
#   tinyllama      → TinyLlama/TinyLlama-1.1B-Chat-v1.0      (TinyLLaVA repo, conv: v1)
#   mobilellama    → mtgv/MobileLLaMA-1.4B-Chat               (MobileVLM repo, conv: v1)
#   smollm2        → HuggingFaceTB/SmolLM2-1.7B-Instruct     (SmolLM repo,    conv: chatml)
#   --- additional Qwen2.5 / Phi-2 / StableLM variants ---
#   qwen0.5b       → Qwen/Qwen2.5-0.5B-Instruct   (conv: chatml)
#   qwen1.5b       → Qwen/Qwen2.5-1.5B-Instruct   (conv: chatml)
#   qwen3b         → Qwen/Qwen2.5-3B-Instruct      (conv: chatml)
#   phi2           → microsoft/phi-2               (conv: phi)
#   stablelm       → stabilityai/stablelm-2-zephyr-1_6b  (conv: chatml)
#
# Stage 1 freezes the LLM and ViT; only elastic_resampler, elastic_projector,
# lora_A, lora_B are trained.  ZeRO-2 is sufficient since frozen params need
# no gradient sharding.

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
    # v1 (vicuna-style USER/ASSISTANT + "</s>"), not chatml: TinyLlama's tokenizer
    # never registered <|im_start|>/<|im_end|> as real special tokens, so they
    # fragment into 7 context-dependent BPE pieces each -- this silently masked
    # ~86% of finetune data's labels (see preprocess_mpt's round-length
    # accounting, which assumes a fixed token cost per separator occurrence).
    # "</s>" IS a real special token here, so preprocess_v1 has no such issue.
    # Matches the reference TinyLLaVA_Factory's own llama_template.py for
    # this exact model (vendored at ./TinyLLaVA_Factory in this repo).
    CONV_VERSION="v1"
    ;;
  mobilellama)
    MODEL_PATH="mtgv/MobileLLaMA-1.4B-Chat"
    CONV_VERSION="v1"   # MobileLLaMA uses the standard Vicuna/LLaVA-v1 template
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

# Optional run tag, e.g.  ELASTIC_RUN_TAG=outnorm sbatch run_job_pretrain_slm.sh
# Suffixes the checkpoint/log dirs so a variant run does not overwrite the
# untagged baseline. Stage 2 reads the same variable to find its warm-start.
TAG="${ELASTIC_RUN_TAG:+-${ELASTIC_RUN_TAG}}"

OUTPUT_DIR="/var/scratch/skalra/flexllava/checkpoints/elastic-pretrain-${LLM_KEY}${TAG}"
LOG_DIR="/var/scratch/skalra/flexllava/logs/elastic-pretrain-${LLM_KEY}${TAG}"
RUN_NAME="elastic-pretrain-${LLM_KEY}-tok256-144-64-16${TAG}"

echo "[FlexLLaVA] Pretrain  LLM=${MODEL_PATH}  conv=${CONV_VERSION}"
echo "[FlexLLaVA] Output → ${OUTPUT_DIR}"

deepspeed --num_gpus 2 llava/train/train_elastic.py \
    --tok_levels 256 \
    --lora_ranks 8 \
    --prefix_kl_weight 0.1 \
    --vision_lora_enable False \
    --coral_weight 0.01 \
    --use_pos_embed True \
    --pos_embed_type learned \
    --use_nested_dropout False \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path "${MODEL_PATH}" \
    --cache_dir /var/scratch/skalra/.cache/huggingface/hub \
    --version "${CONV_VERSION}" \
    --data_path /var/scratch/skalra/flexllava/data/LLaVA-Pretrain/blip_laion_cc_sbu_558k.json \
    --image_folder /var/scratch/skalra/flexllava/data/LLaVA-Pretrain \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio square \
    --freeze_backbone True \
    --bf16 True \
    --output_dir "${OUTPUT_DIR}" \
    --num_train_epochs 1 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 8 \
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
    --model_max_length 1024 \
    --gradient_checkpointing True \
    --dataloader_num_workers 16 \
    --lazy_preprocess True \
    --report_to wandb \
    --run_name "${RUN_NAME}" \
    --logging_dir "${LOG_DIR}"
