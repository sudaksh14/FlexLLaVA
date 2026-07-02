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
# 2x A40 (46 GB each): 4 * 16 * 2 = 128 (matches LLaVA-1.5 finetune recipe).
# 8-bit Adam (bitsandbytes) reduces optimizer states from 84 GB to ~21 GB,
# so 21 GB / 2 GPUs = 10.5 GB/GPU fits alongside params + grads in 46 GB A40.

deepspeed --num_gpus 2 llava/train/train_elastic.py \
    --tok_levels 256 144 64 16 \
    --lora_ranks 8 16 32 64 \
    --prefix_kl_weight 1.0 \
    --coral_weight 0.1 \
    --n_sample_students 1 \
    --deepspeed ./scripts/zero3_elastic.json \
    --model_name_or_path liuhaotian/llava-v1.5-7b \
    --pretrain_elastic_path /var/scratch/skalra/flexllava/checkpoints/llava-elastic-pretrain \
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
    --output_dir /var/scratch/skalra/flexllava/checkpoints/llava-elastic-finetune \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 64 \
    --optim adamw_bnb_8bit \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50000 \
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
    --run_name "elastic-finetune-tok256-144-64-16" \
    --logging_dir /var/scratch/skalra/flexllava/logs/elastic-finetune
