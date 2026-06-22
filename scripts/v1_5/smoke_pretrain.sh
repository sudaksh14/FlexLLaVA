#!/bin/bash
# Smoke-test: identical to pretrain_elastic.sh but runs only MAX_STEPS steps
# with dummy data so the full pipeline can be verified quickly.
#
# Data path and step count are injected by the caller (smoke_test.sh).
# All model paths, elastic config, ZeRO, and flash-attention settings are
# identical to the real pretrain so this validates the actual training stack.

SMOKE_DATA_DIR=${SMOKE_DATA_DIR:-/var/scratch/skalra/flexllava/data/smoke_test}
SMOKE_MAX_STEPS=${SMOKE_MAX_STEPS:-5}

deepspeed --num_gpus 2 llava/train/train_elastic.py \
    --tok_levels 256 144 64 16 \
    --lora_ranks 8 16 32 64 \
    --prefix_kl_weight 1.0 \
    --coral_weight 0.01 \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path lmsys/vicuna-7b-v1.5 \
    --cache_dir /var/scratch/skalra/.cache/huggingface/hub \
    --version plain \
    --data_path "${SMOKE_DATA_DIR}/smoke_data.json" \
    --image_folder "${SMOKE_DATA_DIR}/images" \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio square \
    --freeze_backbone True \
    --bf16 True \
    --output_dir /var/scratch/skalra/flexllava/checkpoints/smoke-pretrain \
    --max_steps "${SMOKE_MAX_STEPS}" \
    --per_device_train_batch_size 32 \
    --gradient_accumulation_steps 4 \
    --save_strategy "no" \
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
    --report_to none
