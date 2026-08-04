#!/bin/bash
# Stage 1 — Elastic Feature Alignment
#
# Trains the elastic projector, nested-query resampler, and nested LoRA adapters
# on the 558K LAION/CC/SBU pretrain subset while keeping the LLM and ViT base
# weights fully frozen. Only elastic_projector, elastic_resampler, lora_A, and
# lora_B are in the optimizer (see train.py elastic attach hook).
#
# Token levels and LoRA config are set in llava/train/train_elastic.py (CONFIG).
# Edit that file to change tok_levels, lora_ranks, or loss weights before running.
#
# Expected data layout (download with huggingface_hub liuhaotian/LLaVA-Pretrain):
#   ./playground/data/LLaVA-Pretrain/blip_laion_cc_sbu_558k.json
#   ./playground/data/LLaVA-Pretrain/images/
#
# Global batch size = per_device_train_batch_size * gradient_accumulation_steps * num_gpus
# 2x A40 (46 GB each) + flash attention: 32 * 4 * 2 = 256 (matches LLaVA-1.5 pretrain recipe).
#
# ZeRO-2 is used (not ZeRO-3) because only the small elastic modules are
# trainable; ZeRO-2 avoids the ZeRO-3 communication overhead when the
# frozen LLM parameters don't need gradient sharding.

# Optional run tag, e.g.  ELASTIC_RUN_TAG=outnorm sbatch run_job_pretrain.sh
# Suffixes the checkpoint/log dirs so a variant run does not overwrite the
# untagged baseline. Stage 2 reads the same variable to find its warm-start.
TAG="${ELASTIC_RUN_TAG:+-${ELASTIC_RUN_TAG}}"
CKPT_ROOT=/var/scratch/skalra/flexllava/checkpoints
LOG_ROOT=/var/scratch/skalra/flexllava/logs

echo "[FlexLLaVA] Stage 1 output → ${CKPT_ROOT}/llava-elastic-pretrain${TAG}"

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
    --model_name_or_path lmsys/vicuna-7b-v1.5 \
    --cache_dir /var/scratch/skalra/.cache/huggingface/hub \
    --version plain \
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
    --output_dir "${CKPT_ROOT}/llava-elastic-pretrain${TAG}" \
    --num_train_epochs 1 \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 4 \
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
    --dataloader_num_workers 16 \
    --lazy_preprocess True \
    --report_to wandb \
    --run_name "elastic-pretrain-tok256-144-64-16${TAG}" \
    --logging_dir "${LOG_ROOT}/elastic-pretrain${TAG}"
