#!/bin/bash
# Stage 1 — Elastic Feature Alignment for Small LLM Backbones (hipster cluster)
#
# Line-for-line the same recipe as scripts/v1_5/pretrain_elastic_slm.sh (DAS-6,
# do not diverge on any --flag value or default without a reason recorded on
# DAS-6's copy first). The ONLY things that differ here are cluster mechanics:
# data comes from a node-local staged copy instead of /var/scratch, and
# checkpoints/cache/logs land under hipster's shared-home save tree instead of
# DAS-6's /var/scratch. This file intentionally does not carry the DAS-6
# decision history in comments -- read scripts/v1_5/pretrain_elastic_slm.sh
# (same repo, same home, always in sync) for the "why" behind any flag.
#
# Usage:
#   LOCAL_SSD=/local_scratch/skalra/flexllava_data_$SLURM_JOB_ID \
#       bash scripts/v1_5/pretrain_elastic_slm_hipster.sh <LLM_KEY>
#
# Not meant to be invoked directly -- run_job_hipster.sh sets LOCAL_SSD (after
# staging data onto it) and calls this.

LLM_KEY=${1:-"tinyllama"}
LOCAL_SSD=${LOCAL_SSD:?LOCAL_SSD must be set by the caller (run_job_hipster.sh) to a node-local staged data directory}
SAVE_ROOT=/home/skalra/flexllava_saves
# Checkpoints only -- moved to the cluster-wide /scratch pool 2026-09-11
# (per-user /home quota is 200GB, tight; /scratch is a shared 15T pool with
# no personal reservation, and no purge policy could be confirmed either
# way -- not a guaranteed-durable location, just a bigger one). logs/cache/
# wandb deliberately stay under SAVE_ROOT (/home) -- only checkpoints were
# asked to move.
CHECKPOINT_ROOT=/scratch/skalra/flexllava_saves/checkpoints

case "$LLM_KEY" in
  qwen0.5b)    MODEL_PATH="Qwen/Qwen2.5-0.5B-Instruct";         CONV_VERSION="chatml" ;;
  qwen1.5b)    MODEL_PATH="Qwen/Qwen2.5-1.5B-Instruct";         CONV_VERSION="chatml" ;;
  qwen3b)      MODEL_PATH="Qwen/Qwen2.5-3B-Instruct";           CONV_VERSION="chatml" ;;
  tinyllama)   MODEL_PATH="TinyLlama/TinyLlama-1.1B-Chat-v1.0"; CONV_VERSION="v1" ;;
  mobilellama) MODEL_PATH="mtgv/MobileLLaMA-1.4B-Chat";         CONV_VERSION="v1" ;;
  smollm2)     MODEL_PATH="HuggingFaceTB/SmolLM2-1.7B-Instruct"; CONV_VERSION="chatml" ;;
  phi2)        MODEL_PATH="microsoft/phi-2";                    CONV_VERSION="phi" ;;
  phi3.5|phi3) MODEL_PATH="microsoft/Phi-3.5-mini-instruct";    CONV_VERSION="phi3" ;;
  stablelm)    MODEL_PATH="stabilityai/stablelm-2-zephyr-1_6b"; CONV_VERSION="chatml" ;;
  *)
    echo "Unknown LLM_KEY='$LLM_KEY'. Choose one of: tinyllama mobilellama smollm2 qwen0.5b qwen1.5b qwen3b phi2 phi3.5 stablelm" >&2
    exit 1 ;;
esac

TAG="${ELASTIC_RUN_TAG:+-${ELASTIC_RUN_TAG}}"

OUTPUT_DIR="${CHECKPOINT_ROOT}/elastic-pretrain-${LLM_KEY}${TAG}"
LOG_DIR="${SAVE_ROOT}/logs/elastic-pretrain-${LLM_KEY}${TAG}"
RUN_NAME="elastic-pretrain-${LLM_KEY}-tok256-144-64-16${TAG}-hipster"

NUM_GPUS="${NUM_GPUS:-2}"
GRAD_ACCUM="${GRAD_ACCUM:-$(( 8 * 2 / NUM_GPUS ))}"
echo "[FlexLLaVA] num_gpus=${NUM_GPUS}  grad_accum=${GRAD_ACCUM}  (effective batch unchanged)"
echo "[FlexLLaVA] Pretrain  LLM=${MODEL_PATH}  conv=${CONV_VERSION}"
echo "[FlexLLaVA] Output → ${OUTPUT_DIR}"

STAGE1_TOK_LEVEL="${STAGE1_TOK_LEVEL:-256}"
STAGE1_LORA_RANK="${STAGE1_LORA_RANK:-64}"
echo "[FlexLLaVA] tok_level=${STAGE1_TOK_LEVEL}  lora_rank=${STAGE1_LORA_RANK}  (must match finetune's largest tok_level / max lora_rank)"
echo "[FlexLLaVA] vision_lora_enable=${VISION_LORA_ENABLE:-False}  specialize_tok=${VISION_LORA_SPECIALIZE_TOK:-True}"

# Unique --master_port per job: hipster runs are NOT --exclusive (shared
# etiquette, unlike DAS-6), so two of our own concurrent jobs can land on the
# same physical node -- deepspeed's default port 29500 then collides, and
# whichever job loses the race crashes with "Address already in use" (found
# 2026-09-14: two concurrent Stage-1 matrix jobs on hipster-cn008).
MASTER_PORT=$(( 20000 + SLURM_JOB_ID % 10000 ))
echo "[FlexLLaVA] master_port=${MASTER_PORT}  (derived from SLURM_JOB_ID=${SLURM_JOB_ID}, avoids collision with co-located jobs)"
deepspeed --num_gpus ${NUM_GPUS} --master_port ${MASTER_PORT} llava/train/train_elastic.py \
    --tok_levels ${STAGE1_TOK_LEVEL} \
    --lora_ranks ${STAGE1_LORA_RANK} \
    ${NEST_VERSION:+--nest_version ${NEST_VERSION}} \
    --resampler_arch "${RESAMPLER_ARCH:-query}" \
    --anchor_routing "${ANCHOR_ROUTING:-}" \
    --anchor_mode "${ANCHOR_MODE:-ratio}" \
    --anchor_ratio "${ANCHOR_RATIO:-0.25}" \
    --use_token_decorrelation "${USE_TOKEN_DECORRELATION:-False}" \
    --decorr_weight "${DECORR_WEIGHT:-0.01}" \
    --prefix_kl_weight 0.1 \
    --vision_lora_enable "${VISION_LORA_ENABLE:-False}" \
    --vision_lora_specialize_tok "${VISION_LORA_SPECIALIZE_TOK:-True}" \
    --coral_weight 0.01 \
    --use_pos_embed True \
    --pos_embed_type learned \
    --use_nested_dropout False \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path "${MODEL_PATH}" \
    --cache_dir ${SAVE_ROOT}/cache/huggingface/hub \
    --version "${CONV_VERSION}" \
    --data_path ${LOCAL_SSD}/LLaVA-Pretrain/blip_laion_cc_sbu_558k.json \
    --image_folder ${LOCAL_SSD}/LLaVA-Pretrain \
    --vision_tower "${VISION_TOWER:-openai/clip-vit-large-patch14-336}" \
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
    --dataloader_num_workers 16 \
    --lazy_preprocess True \
    --report_to wandb \
    --run_name "${RUN_NAME}" \
    --logging_dir "${LOG_DIR}"
