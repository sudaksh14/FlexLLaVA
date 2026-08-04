#!/bin/bash
# Stage 2 — Elastic Visual Instruction Tuning
#
# Fine-tunes with the LLM unfrozen alongside the elastic projector, resampler,
# and nested LoRA adapters, using the LLaVA-v1.5 665K instruction mix.
#
# Starting model: liuhaotian/llava-v1.5-7b provides the LLM + base projector.
# --pretrain_elastic_path points to the Stage 1 output dir so elastic_resampler,
# elastic_projector, and LoRA A/B are warm-started from pretrain weights.
# (attach_elastic_engine always creates fresh modules; warm-start loads them
# back after attachment via _load_elastic_pretrain_weights in train.py.)
#
# Token levels and LoRA config are set in llava/train/train_elastic.py (CONFIG).
# Edit that file to change tok_levels, lora_ranks, or loss weights before running.
#
# Expected data layout:
#   ./playground/data/llava_v1_5_mix665k.json
#   ./playground/data/coco/train2017/
#   ./playground/data/gqa/images/
#   ./playground/data/ocr_vqa/images/
#   ./playground/data/textvqa/train_images/
#   ./playground/data/vg/VG_100K/ and VG_100K_2/
#
# Global batch size = per_device_train_batch_size * gradient_accumulation_steps * num_gpus
# 2x A40 (46 GB each): 2 * 32 * 2 = 128 (matches LLaVA-1.5 finetune recipe).
# LLM LoRA (rank 128): only ~416M LoRA params need optimizer states instead of 7B,
# reducing GPU memory from ~43 GB to ~16 GB per GPU — fits on A40 with ZeRO-2.

# Optional run tag -- must match the ELASTIC_RUN_TAG used for Stage 1, since it
# selects both the warm-start checkpoint and this run's output dir.
TAG="${ELASTIC_RUN_TAG:+-${ELASTIC_RUN_TAG}}"
CKPT_ROOT=/var/scratch/skalra/flexllava/checkpoints
LOG_ROOT=/var/scratch/skalra/flexllava/logs

echo "[FlexLLaVA] Warm-start from → ${CKPT_ROOT}/llava-elastic-pretrain${TAG}"
echo "[FlexLLaVA] Stage 2 output  → ${CKPT_ROOT}/llava-elastic-finetune${TAG}"

deepspeed --num_gpus 2 llava/train/train_elastic.py \
    --tok_levels 256 144 64 16 \
    --lora_ranks 8 16 32 64 \
    --prefix_kl_weight 0.1 \
    --coral_weight 0.1 \
    --use_coral False \
    --use_pos_embed True \
    --pos_embed_type learned \
    --use_nested_dropout False \
    --vision_lora_enable False \
    --n_sample_students 1 \
    --lora_enable True \
    --lora_r 128 \
    --lora_alpha 128 \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path liuhaotian/llava-v1.5-7b \
    --pretrain_elastic_path "${CKPT_ROOT}/llava-elastic-pretrain${TAG}" \
    --cache_dir /var/scratch/skalra/.cache/huggingface/hub \
    --version v1 \
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
    --output_dir "${CKPT_ROOT}/llava-elastic-finetune${TAG}" \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 32 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 100 \
    --save_total_limit 5 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --gradient_checkpointing_kwargs '{"use_reentrant": false}' \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb \
    --run_name "elastic-finetune-tok256-144-64-16${TAG}" \
    --logging_dir "${LOG_ROOT}/elastic-finetune${TAG}"
