#!/bin/bash
# Stage 1 — NON-ELASTIC control, TinyLLaVA-equivalent.
#
#   bash scripts/v1_5/pretrain_baseline_slm.sh [LLM_KEY]
#
# Purpose: establish whether OUR pipeline can reproduce TinyLLaVA's published
# baseline for the identical backbone + vision tower. Their row in
# TinyLLaVA_Factory/README.md:152 (clip-vit-large-patch14-336 + TinyLlama-1.1B,
# recipe "base") reports GQA 58.0 / SQA-I 59.9 / TextVQA 46.3 / POPE 85.5 /
# MME 1284.6, versus our elastic run's 52.3 / 51.3 / 23.1 / 81.6 / 1113.5.
#
# The control removes the elastic machinery entirely -- train.py, not
# train_elastic.py, and mm_projector_type mlp2x_gelu with no resampler, so all
# 576 CLIP patches reach the LLM 1:1 (matryoshka_vis_token_scale defaults to
# None, which skips the pooling path). If this reproduces ~58 GQA the pipeline
# is sound and the whole gap is token compression; if it lands near 52 the
# recipe is wrong independently of elasticity and no resampler work will fix it.
#
# Hyperparameters deliberately mirror TinyLLaVA's scripts/train/pretrain.sh:
# effective batch 256 (they use 4 GPUs x 32 x 2; we use 2 x 8 x 16), LR 1e-3,
# 1 epoch, cosine, warmup 0.03, model_max_length 2048, frozen LLM + frozen ViT.
# image_aspect_ratio is 'square' in BOTH stages to match them -- note our
# elastic Stage 2 uses 'pad' (the LLaVA-1.5 default), so that is a second, small
# difference between this control and our elastic runs.

LLM_KEY=${1:-"tinyllama"}

case "$LLM_KEY" in
  tinyllama)
    MODEL_PATH="TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    ;;
  smollm2)
    MODEL_PATH="HuggingFaceTB/SmolLM2-1.7B-Instruct"
    ;;
  phi2)
    MODEL_PATH="microsoft/phi-2"
    ;;
  *)
    echo "Unknown LLM_KEY='$LLM_KEY' for the baseline control (tinyllama|smollm2|phi2)"
    exit 1
    ;;
esac

TAG="${BASELINE_RUN_TAG:+-${BASELINE_RUN_TAG}}"
OUTPUT_DIR="/var/scratch/skalra/flexllava/checkpoints/baseline-${LLM_KEY}${TAG}-pretrain"
LOG_DIR="/var/scratch/skalra/flexllava/logs/baseline-${LLM_KEY}${TAG}-pretrain"
RUN_NAME="baseline-${LLM_KEY}-576tok${TAG}-pretrain"

# Effective batch = per_device * accum * gpus = 8 * 16 * 2 = 256, matching
# TinyLLaVA. per_device is 8 rather than their 32 because this runs on 24GB
# A10s and the sequence carries all 576 visual tokens uncompressed.
# accum = 256 / (PER_DEVICE * NUM_GPUS); keep PER_DEVICE in the arithmetic so
# the two cannot drift apart -- a bare "128 / NUM_GPUS" here silently produced
# an effective batch of 1024 while the banner still claimed 256 (run 26697).
NUM_GPUS="${NUM_GPUS:-2}"
PER_DEVICE="${PER_DEVICE:-8}"
GRAD_ACCUM="${GRAD_ACCUM:-$(( 256 / (PER_DEVICE * NUM_GPUS) ))}"
EFF_BATCH=$(( PER_DEVICE * GRAD_ACCUM * NUM_GPUS ))
echo "[baseline] num_gpus=${NUM_GPUS} per_device=${PER_DEVICE} grad_accum=${GRAD_ACCUM} -> effective batch ${EFF_BATCH} (target 256)"
echo "[baseline] Pretrain LLM=${MODEL_PATH}  576 tokens, mlp2x_gelu, NO resampler"
echo "[baseline] Output → ${OUTPUT_DIR}"

deepspeed --num_gpus ${NUM_GPUS} llava/train/train_mem.py \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path "${MODEL_PATH}" \
    --cache_dir /var/scratch/skalra/.cache/huggingface/hub \
    --version plain \
    --data_path /var/scratch/skalra/flexllava/data/LLaVA-Pretrain/blip_laion_cc_sbu_558k.json \
    --image_folder /var/scratch/skalra/flexllava/data/LLaVA-Pretrain \
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
    --dataloader_num_workers 8 \
    --lazy_preprocess True \
    --report_to wandb \
    --run_name "${RUN_NAME}" \
    --logging_dir "${LOG_DIR}"
