#!/bin/bash
# Stage 2 — Elastic Visual Instruction Tuning on the OTTER-STYLE DATA PIPELINE.
#
#   bash scripts/v1_5/finetune_otter_slm.sh <LLM_KEY> [MIXTURE_CONFIG]
#
# This is a PARALLEL recipe to scripts/v1_5/finetune_elastic_slm.sh, which is
# unchanged and still drives the current runs. Same model, same optimizer
# settings, same elastic flags -- the ONLY differences are the data pipeline
# (llava/data_otter/) and the trainer's telemetry (llava/train/otter_trainer.py),
# so a run of this against configs/otter/mix665k_baseline.yaml is a controlled
# A/B of the pipeline itself.
#
# What it adds over the elastic recipe:
#   * a hard pre-run verification gate  (#5) -- the job aborts before touching a
#     GPU if EOS supervision, pad!=eos, label masking, images or the warm-start
#     checkpoint are wrong. Four burned train/eval cycles on this project were
#     all preventable here.
#   * per-source x per-tok-level loss telemetry (#7)
#   * a YAML-declared mixture (#2) and QA packing (#3)
#   * deterministic missing-image handling (#6) and token-accurate lengths (#8)
#
# Hyperparameters are deliberately IDENTICAL to finetune_elastic_slm.sh (full
# finetune, LR 2e-5, cosine, warmup 0.03). See that file's header for why LoRA
# is off and why the LR is what it is -- the 2e-4 / alpha-256 configuration
# diverged in run_26226.

set -euo pipefail

LLM_KEY=${1:-"tinyllama"}
MIXTURE_CONFIG=${2:-${OTTER_MIXTURE_CONFIG:-configs/otter/mix665k_baseline.yaml}}

case "$LLM_KEY" in
  qwen0.5b)    MODEL_PATH="Qwen/Qwen2.5-0.5B-Instruct";        CONV_VERSION="chatml" ;;
  qwen1.5b)    MODEL_PATH="Qwen/Qwen2.5-1.5B-Instruct";        CONV_VERSION="chatml" ;;
  qwen3b)      MODEL_PATH="Qwen/Qwen2.5-3B-Instruct";          CONV_VERSION="chatml" ;;
  tinyllama)   MODEL_PATH="TinyLlama/TinyLlama-1.1B-Chat-v1.0"; CONV_VERSION="v1" ;;
  mobilellama) MODEL_PATH="mtgv/MobileLLaMA-1.4B-Chat";        CONV_VERSION="v1" ;;
  smollm2)     MODEL_PATH="HuggingFaceTB/SmolLM2-1.7B-Instruct"; CONV_VERSION="chatml" ;;
  phi2)        MODEL_PATH="microsoft/phi-2";                   CONV_VERSION="phi" ;;
  phi3.5|phi3) MODEL_PATH="microsoft/Phi-3.5-mini-instruct";   CONV_VERSION="phi3" ;;
  stablelm)    MODEL_PATH="stabilityai/stablelm-2-zephyr-1_6b"; CONV_VERSION="chatml" ;;
  *)
    echo "Unknown LLM_KEY='$LLM_KEY'. Choose one of: tinyllama mobilellama smollm2 qwen0.5b qwen1.5b qwen3b phi2 phi3.5 stablelm"
    exit 1 ;;
esac

# Run tag selects both the warm-start checkpoint and the output dir, exactly as
# in the elastic recipe. The "-otter" suffix keeps these runs from ever
# overwriting an elastic-pipeline checkpoint of the same tag.
TAG="${ELASTIC_RUN_TAG:+-${ELASTIC_RUN_TAG}}"
PRETRAIN_TAG="${ELASTIC_PRETRAIN_TAG:+-${ELASTIC_PRETRAIN_TAG}}"
: "${PRETRAIN_TAG:=$TAG}"

# Stage-1 checkpoint to warm-start from. pretrain_otter_slm.sh writes
# otter-pretrain-*, while the elastic recipe writes elastic-pretrain-*; prefer
# the otter one, fall back to an existing elastic Stage-1 checkpoint so a
# Stage-2 otter run can reuse one without re-deriving it. Override explicitly
# with OTTER_PRETRAIN_DIR.
CKPT_ROOT=/var/scratch/skalra/flexllava/checkpoints
if [ -n "${OTTER_PRETRAIN_DIR:-}" ]; then
    PRETRAIN_CKPT="${OTTER_PRETRAIN_DIR}"
elif [ -d "${CKPT_ROOT}/otter-pretrain-${LLM_KEY}${PRETRAIN_TAG}" ]; then
    PRETRAIN_CKPT="${CKPT_ROOT}/otter-pretrain-${LLM_KEY}${PRETRAIN_TAG}"
elif [ -d "${CKPT_ROOT}/elastic-pretrain-${LLM_KEY}${PRETRAIN_TAG}" ]; then
    PRETRAIN_CKPT="${CKPT_ROOT}/elastic-pretrain-${LLM_KEY}${PRETRAIN_TAG}"
    echo "[FlexLLaVA/otter] no otter-pretrain-${LLM_KEY}${PRETRAIN_TAG}; warm-starting from the elastic Stage-1 checkpoint instead"
else
    echo "[FlexLLaVA/otter] ERROR: no Stage-1 checkpoint found for ${LLM_KEY}${PRETRAIN_TAG}."
    echo "  looked for: ${CKPT_ROOT}/otter-pretrain-${LLM_KEY}${PRETRAIN_TAG}"
    echo "              ${CKPT_ROOT}/elastic-pretrain-${LLM_KEY}${PRETRAIN_TAG}"
    echo "  run Stage 1 first, or set OTTER_PRETRAIN_DIR / ELASTIC_PRETRAIN_TAG."
    exit 1
fi
OUTPUT_DIR="/var/scratch/skalra/flexllava/checkpoints/otter-finetune-${LLM_KEY}${TAG}"
LOG_DIR="/var/scratch/skalra/flexllava/logs/otter-finetune-${LLM_KEY}${TAG}"
RUN_NAME="otter-finetune-${LLM_KEY}-tok256-144-64-16${TAG}"
OTTER_CACHE_DIR="${OTTER_CACHE_DIR:-/var/scratch/skalra/flexllava/cache/otter}"

NUM_GPUS="${NUM_GPUS:-2}"
GRAD_ACCUM="${GRAD_ACCUM:-$(( 32 * 2 / NUM_GPUS ))}"

echo "[FlexLLaVA/otter] LLM=${MODEL_PATH}  conv=${CONV_VERSION}"
echo "[FlexLLaVA/otter] mixture=${MIXTURE_CONFIG}"
echo "[FlexLLaVA/otter] num_gpus=${NUM_GPUS}  grad_accum=${GRAD_ACCUM}  (effective batch unchanged)"
echo "[FlexLLaVA/otter] pretrain ckpt -> ${PRETRAIN_CKPT}"
echo "[FlexLLaVA/otter] output       -> ${OUTPUT_DIR}"
echo "[FlexLLaVA/otter] cache dir    -> ${OTTER_CACHE_DIR}"

# ---------------------------------------------------------------------------
# PRE-RUN VERIFICATION GATE (#5)
#
# CPU-only, no model weights, seconds to run. `set -e` plus this exit code is
# what makes it a gate rather than a suggestion: a job that would have trained
# for 24h on unsupervised EOS or fully-masked labels never starts.
# Set OTTER_SKIP_VERIFY=1 to bypass (and own the consequences).
# ---------------------------------------------------------------------------
if [ "${OTTER_SKIP_VERIFY:-0}" != "1" ]; then
  echo "[FlexLLaVA/otter] running pre-run verification gate..."
  python -m llava.data_otter.verify \
      --mixture_config "${MIXTURE_CONFIG}" \
      --model_name_or_path "${MODEL_PATH}" \
      --version "${CONV_VERSION}" \
      --model_max_length 2048 \
      --cache_dir /var/scratch/skalra/.cache/huggingface/hub \
      --pretrain_elastic_path "${PRETRAIN_CKPT}" \
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
    --gradient_accumulation_steps ${GRAD_ACCUM} \
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
    --ddp_timeout 900 \
    --gradient_checkpointing True \
    --gradient_checkpointing_kwargs '{"use_reentrant": false}' \
    --dataloader_num_workers "${OTTER_WORKERS:-4}" \
    --dataloader_persistent_workers True \
    --dataloader_prefetch_factor 4 \
    --lazy_preprocess True \
    --report_to wandb \
    --run_name "${RUN_NAME}" \
    --logging_dir "${LOG_DIR}"
