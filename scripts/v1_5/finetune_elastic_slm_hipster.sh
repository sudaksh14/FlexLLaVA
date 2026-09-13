#!/bin/bash
# Stage 2 — Elastic Visual Instruction Tuning for SLMs (hipster cluster)
#
# Line-for-line the same recipe as scripts/v1_5/finetune_elastic_slm.sh
# (DAS-6). Do not diverge on any --flag value or default without recording
# the reason on DAS-6's copy first -- read that file for the "why" behind
# every flag; this one only carries the cluster-mechanics differences.
#
# NOT the ZeRO-3 full-FT variant (finetune_elastic_slm_zero3_hipster.sh) --
# that is a separate, untested idea for scaling to qwen3b and is unrelated to
# this recipe. This script matches DAS-6: --lora_enable False on the LLM
# (full finetune, already the norm for 1-3B backbones here), ZeRO-2, 2 GPUs.
#
# Usage:
#   LOCAL_SSD=/local_scratch/skalra/flexllava_data_$SLURM_JOB_ID \
#       bash scripts/v1_5/finetune_elastic_slm_hipster.sh <LLM_KEY>
#
# Not meant to be invoked directly -- run_job_hipster.sh sets LOCAL_SSD (after
# staging data onto it) and calls this after Stage 1.

LLM_KEY=${1:-"tinyllama"}
LOCAL_SSD=${LOCAL_SSD:?LOCAL_SSD must be set by the caller (run_job_hipster.sh) to a node-local staged data directory}
SAVE_ROOT=/home/skalra/flexllava_saves
# Checkpoints only -- moved to /scratch 2026-09-11; see pretrain_elastic_
# slm_hipster.sh for why. Must match that script's CHECKPOINT_ROOT or Stage
# 2 warm-starts from the wrong place (silently starts fresh, or errors).
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
PRETRAIN_TAG="${ELASTIC_PRETRAIN_TAG:+-${ELASTIC_PRETRAIN_TAG}}"
: "${PRETRAIN_TAG:=$TAG}"

PRETRAIN_CKPT="${CHECKPOINT_ROOT}/elastic-pretrain-${LLM_KEY}${PRETRAIN_TAG}"
OUTPUT_DIR="${CHECKPOINT_ROOT}/elastic-finetune-${LLM_KEY}${TAG}"
LOG_DIR="${SAVE_ROOT}/logs/elastic-finetune-${LLM_KEY}${TAG}"
RUN_NAME="elastic-finetune-${LLM_KEY}-tok256-144-64-16${TAG}-hipster"

NUM_GPUS="${NUM_GPUS:-2}"
GRAD_ACCUM="${GRAD_ACCUM:-$(( 32 * 2 / NUM_GPUS ))}"
echo "[FlexLLaVA] num_gpus=${NUM_GPUS}  grad_accum=${GRAD_ACCUM}  (effective batch unchanged)"
echo "[FlexLLaVA] Finetune  LLM=${MODEL_PATH}  conv=${CONV_VERSION}"
echo "[FlexLLaVA] Pretrain checkpoint → ${PRETRAIN_CKPT}"
echo "[FlexLLaVA] Output             → ${OUTPUT_DIR}"
# LoRA ladder: set LORA_TYPE (v8|asc) OR LORA_RANKS, not both -- train_elastic.py
# errors if they contradict, so a checkpoint can never record a lora_type that
# does not describe its own weights. If NEITHER is set, fall back to the
# historical explicit ladder so every pre-existing caller behaves identically.
# Ported verbatim from DAS-6's scripts/v1_5/finetune_elastic_slm.sh (commit
# 8d9369e) -- see docs/EXPERIMENT_JOURNAL.md section 17d/18f/18g.
if [ -z "${LORA_TYPE:-}" ] && [ -z "${NEST_VERSION:-}" ] && [ -z "${LORA_RANKS:-}" ]; then
    LORA_RANKS="8 16 32 64"
fi
echo "[FlexLLaVA] nest_version=${NEST_VERSION:-<unset>}  lora_type=${LORA_TYPE:-<unset>}  lora_ranks=${LORA_RANKS:-<derived from lora_type>}"
echo "[FlexLLaVA] tok_levels=${TOK_LEVELS:-256 144 64 16}  lora_ranks=${LORA_RANKS:-<derived>}"
echo "[FlexLLaVA] vision_lora_enable=${VISION_LORA_ENABLE:-False}  specialize_tok=${VISION_LORA_SPECIALIZE_TOK:-True}"
echo "[FlexLLaVA] use_kd=${USE_KD:-True} (prefix-KL SELF-distillation across budgets; teacher and student share weights unless TEACHER=llava or KD_TEACHER is set)"
echo "[FlexLLaVA] teacher=${TEACHER:-self}  kd_teacher=${KD_TEACHER:-<unset>}  kd_student_key=${KD_STUDENT_KEY:-<unset>}  prefix_kl_weight=${PREFIX_KL_WEIGHT:-0.1}"

# Unique --master_port per job -- see pretrain_elastic_slm_hipster.sh for why.
MASTER_PORT=$(( 20000 + SLURM_JOB_ID % 10000 ))
echo "[FlexLLaVA] master_port=${MASTER_PORT}  (derived from SLURM_JOB_ID=${SLURM_JOB_ID}, avoids collision with co-located jobs)"
deepspeed --num_gpus ${NUM_GPUS} --master_port ${MASTER_PORT} llava/train/train_elastic.py \
    --tok_levels ${TOK_LEVELS:-256 144 64 16} \
    ${LORA_RANKS:+--lora_ranks ${LORA_RANKS}} \
    ${LORA_TYPE:+--lora_type ${LORA_TYPE}} \
    ${NEST_VERSION:+--nest_version ${NEST_VERSION}} \
    --resampler_arch "${RESAMPLER_ARCH:-query}" \
    --anchor_routing "${ANCHOR_ROUTING:-}" \
    --anchor_mode "${ANCHOR_MODE:-ratio}" \
    --anchor_ratio "${ANCHOR_RATIO:-0.25}" \
    --use_token_decorrelation "${USE_TOKEN_DECORRELATION:-False}" \
    --decorr_weight "${DECORR_WEIGHT:-0.01}" \
    --use_kd "${USE_KD:-True}" \
    --teacher "${TEACHER:-self}" \
    ${KD_TEACHER:+--kd_teacher ${KD_TEACHER}} \
    ${KD_STUDENT_KEY:+--kd_student_key ${KD_STUDENT_KEY}} \
    ${KD_TYPE:+--kd_type ${KD_TYPE}} \
    --teacher_model_path "${TEACHER_MODEL_PATH:-liuhaotian/llava-v1.5-7b}" \
    --prefix_kl_weight "${PREFIX_KL_WEIGHT:-0.1}" \
    --coral_weight 0.1 \
    --use_coral False \
    --use_pos_embed True \
    --pos_embed_type learned \
    --use_nested_dropout False \
    --n_sample_students 1 \
    --vision_lora_enable "${VISION_LORA_ENABLE:-False}" \
    --vision_lora_specialize_tok "${VISION_LORA_SPECIALIZE_TOK:-True}" \
    --lora_enable False \
    --lora_r 128 \
    --lora_alpha 128 \
    --mm_projector_lr 2e-5 \
    --mm_vision_tower_lr 2e-5 \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path "${MODEL_PATH}" \
    --pretrain_elastic_path "${PRETRAIN_CKPT}" \
    --cache_dir ${SAVE_ROOT}/cache/huggingface/hub \
    --version "${CONV_VERSION}" \
    --data_path ${LOCAL_SSD}/LLaVA-Finetune/llava_v1_5_mix665k.json \
    --image_folder ${LOCAL_SSD}/LLaVA-Finetune \
    --vision_tower "${VISION_TOWER:-openai/clip-vit-large-patch14-336}" \
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
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb \
    --run_name "${RUN_NAME}" \
    --logging_dir "${LOG_DIR}"
