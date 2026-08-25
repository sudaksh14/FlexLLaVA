#!/bin/bash
# Stage 1 — Elastic Feature Alignment on the OTTER-STYLE DATA PIPELINE.
#
#   bash scripts/v1_5/pretrain_otter_slm.sh <LLM_KEY> [MIXTURE_CONFIG]
#
# Parallel to scripts/v1_5/pretrain_elastic_slm.sh, which is unchanged. Every
# hyperparameter below is identical to that script -- single tok_level 256,
# frozen LLM and ViT, LR 1e-3, batch 16, model_max_length 1024, square aspect
# ratio. The only differences are the data pipeline and the pre-run gate.
#
# Stage 1 freezes the LLM and ViT; only elastic_resampler / elastic_projector
# (and LoRA, when enabled) train. ZeRO-2 is sufficient since frozen params need
# no gradient sharding.
#
# NOTE on --tok_levels 256: Stage 1 trains a SINGLE level, so there is one
# forward per step and no elastic grid. Per-level gap telemetry is therefore
# inert here; it is Stage 2 that exercises it.

set -euo pipefail

LLM_KEY=${1:-"tinyllama"}
MIXTURE_CONFIG=${2:-${OTTER_MIXTURE_CONFIG:-configs/otter/pretrain558k.yaml}}

case "$LLM_KEY" in
  qwen0.5b)    MODEL_PATH="Qwen/Qwen2.5-0.5B-Instruct";          CONV_VERSION="chatml" ;;
  qwen1.5b)    MODEL_PATH="Qwen/Qwen2.5-1.5B-Instruct";          CONV_VERSION="chatml" ;;
  qwen3b)      MODEL_PATH="Qwen/Qwen2.5-3B-Instruct";            CONV_VERSION="chatml" ;;
  # v1 not chatml for TinyLlama: its tokenizer never registered
  # <|im_start|>/<|im_end|>, so they fragment into 7 BPE pieces each and the
  # round accounting silently masks ~86% of labels. See pretrain_elastic_slm.sh.
  tinyllama)   MODEL_PATH="TinyLlama/TinyLlama-1.1B-Chat-v1.0";  CONV_VERSION="v1" ;;
  mobilellama) MODEL_PATH="mtgv/MobileLLaMA-1.4B-Chat";          CONV_VERSION="v1" ;;
  smollm2)     MODEL_PATH="HuggingFaceTB/SmolLM2-1.7B-Instruct"; CONV_VERSION="chatml" ;;
  phi2)        MODEL_PATH="microsoft/phi-2";                     CONV_VERSION="phi" ;;
  phi3.5|phi3) MODEL_PATH="microsoft/Phi-3.5-mini-instruct";     CONV_VERSION="phi3" ;;
  stablelm)    MODEL_PATH="stabilityai/stablelm-2-zephyr-1_6b";  CONV_VERSION="chatml" ;;
  *)
    echo "Unknown LLM_KEY='$LLM_KEY'. Choose: tinyllama mobilellama smollm2 qwen0.5b qwen1.5b qwen3b phi2 phi3.5 stablelm"
    exit 1 ;;
esac

TAG="${ELASTIC_RUN_TAG:+-${ELASTIC_RUN_TAG}}"
OUTPUT_DIR="/var/scratch/skalra/flexllava/checkpoints/otter-pretrain-${LLM_KEY}${TAG}"
LOG_DIR="/var/scratch/skalra/flexllava/logs/otter-pretrain-${LLM_KEY}${TAG}"
RUN_NAME="otter-pretrain-${LLM_KEY}-tok256${TAG}"
OTTER_CACHE_DIR="${OTTER_CACHE_DIR:-/var/scratch/skalra/flexllava/cache/otter}"

NUM_GPUS="${NUM_GPUS:-2}"
# Derive accumulation so the effective batch (per_device * accum * gpus) is
# invariant to NUM_GPUS and MATCHES pretrain_elastic_slm.sh: 16 * 8 * 2 = 256,
# which is LLaVA's standard Stage-1 batch.
#
# This was 2*2/NUM_GPUS in the first version of this script -- accum 2, i.e. an
# effective batch of 64, a 4x smaller batch at the same LR 1e-3. That produced
# otter-pretrain-tinyllama-otter1, whose Stage-1 loss ran ~0.8 HIGHER than the
# v4 checkpoint throughout (3.07 vs 2.26), and every Stage-2 and eval number
# warm-started from it inherited the deficit. Do not "simplify" this again.
GRAD_ACCUM="${GRAD_ACCUM:-$(( 8 * 2 / NUM_GPUS ))}"
BATCH="${OTTER_BATCH:-16}"

echo "[FlexLLaVA/otter] Pretrain  LLM=${MODEL_PATH}  conv=${CONV_VERSION}"
echo "[FlexLLaVA/otter] mixture=${MIXTURE_CONFIG}"
echo "[FlexLLaVA/otter] num_gpus=${NUM_GPUS} batch=${BATCH} grad_accum=${GRAD_ACCUM}"
echo "[FlexLLaVA/otter] Output → ${OUTPUT_DIR}"

# ---- PRE-RUN VERIFICATION GATE (#5) ---------------------------------------
# Stage 1 is a ~10h job; every defect this catches is detectable in seconds on
# CPU. set -e makes it a gate rather than a suggestion.
if [ "${OTTER_SKIP_VERIFY:-0}" != "1" ]; then
  echo "[FlexLLaVA/otter] running pre-run verification gate..."
  python -m llava.data_otter.verify \
      --mixture_config "${MIXTURE_CONFIG}" \
      --model_name_or_path "${MODEL_PATH}" \
      --version "${CONV_VERSION}" \
      --model_max_length 1024 \
      --cache_dir /var/scratch/skalra/.cache/huggingface/hub \
      --n_samples "${OTTER_VERIFY_SAMPLES:-256}"
  echo "[FlexLLaVA/otter] verification passed; starting training."
else
  echo "[FlexLLaVA/otter] WARNING: OTTER_SKIP_VERIFY=1, skipping the pre-run gate."
fi

deepspeed --num_gpus ${NUM_GPUS} llava/train/train_otter.py \
    --mixture_config "${MIXTURE_CONFIG}" \
    --otter_cache_dir "${OTTER_CACHE_DIR}" \
    --otter_build_caches "${OTTER_BUILD_CACHES:-False}" \
    --otter_log_every "${OTTER_LOG_EVERY:-25}" \
    --otter_source_grouped_batches "${OTTER_SOURCE_BATCHES:-True}" \
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
    --per_device_train_batch_size ${BATCH} \
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
    --model_max_length 1024 \
    --gradient_checkpointing True \
    --dataloader_num_workers "${OTTER_WORKERS:-8}" \
    --lazy_preprocess True \
    --report_to wandb \
    --run_name "${RUN_NAME}" \
    --logging_dir "${LOG_DIR}"
