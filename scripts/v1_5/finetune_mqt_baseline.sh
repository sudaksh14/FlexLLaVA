#!/bin/bash
# Stage 2 — MQT-LLaVA baseline at a FIXED small query count (default 4).
#
#   NUM_VISUAL_TOKENS=4 bash scripts/v1_5/finetune_mqt_baseline.sh [LLM_KEY]
#
# MQT's own train loop (see pretrain_mqt_baseline.sh header). num_visual_tokens
# is passed straight to get_matry_n():
#   'second_stage' -> random.choice(range(2,258,2)) per step  (MQT's real recipe)
#   an integer     -> that many queries, every step
# We pass the integer 4, because the ask is a fixed-4-token baseline number,
# not MQT's full elastic curve. That is a deliberate narrowing of MQT's recipe
# and must be described that way: it is "MQT's architecture and train loop at a
# fixed budget", NOT "MQT as published".
#
# NOTE (verified, diag job 27419): MQT has no mm_projector -- the Resampler
# projects to the LLM width itself. Stage 1 therefore writes only
# query_abstractor.bin, and --mm_projector_type / --pretrain_mm_mlp_adapter are
# not passed.
#
# Hyperparameters mirror OUR Stage 2 exactly (finetune_elastic_slm.sh):
# mix665k, LR 2e-5 cosine, warmup 0.03, wd 0, 1 epoch, eff batch 128,
# bf16, max_len 2048, vision tower frozen, full LLM finetune.
set -uo pipefail
LLM_KEY=${1:-tinyllama}
case "$LLM_KEY" in
  tinyllama) MODEL_PATH="TinyLlama/TinyLlama-1.1B-Chat-v1.0"; CONV_VERSION="v1" ;;
  *) echo "MQT baseline is set up for tinyllama only; got '$LLM_KEY'" >&2; exit 1 ;;
esac

TAG="${BASELINE_RUN_TAG:+-${BASELINE_RUN_TAG}}"
PRETRAIN_TAG="${BASELINE_PRETRAIN_TAG:+-${BASELINE_PRETRAIN_TAG}}"
: "${PRETRAIN_TAG:=$TAG}"
PRETRAIN_DIR="/var/scratch/skalra/flexllava/checkpoints/mqt-pretrain-${LLM_KEY}${PRETRAIN_TAG}"
OUTPUT_DIR="/var/scratch/skalra/flexllava/checkpoints/mqt-finetune-${LLM_KEY}${TAG}"
NUM_VISUAL_TOKENS="${NUM_VISUAL_TOKENS:-4}"
NUM_GPUS="${NUM_GPUS:-2}"
PER_DEVICE="${PER_DEVICE:-2}"
GRAD_ACCUM="${GRAD_ACCUM:-$(( 128 / (PER_DEVICE * NUM_GPUS) ))}"
MAX_STEPS_ARG=""; [ -n "${MAX_STEPS:-}" ] && MAX_STEPS_ARG="--max_steps ${MAX_STEPS}"

if [ ! -f "${PRETRAIN_DIR}/query_abstractor.bin" ]; then
    echo "ERROR: ${PRETRAIN_DIR}/query_abstractor.bin not found -- run Stage 1 first." >&2
    ls "${PRETRAIN_DIR}" 2>/dev/null >&2 || true
    exit 1
fi

echo "[mqt] Stage 2  LLM=${MODEL_PATH}  conv=${CONV_VERSION}  num_visual_tokens=${NUM_VISUAL_TOKENS}"
echo "[mqt] gpus=${NUM_GPUS} per_device=${PER_DEVICE} accum=${GRAD_ACCUM} -> eff batch $((PER_DEVICE*GRAD_ACCUM*NUM_GPUS))"
echo "[mqt] warm start <- ${PRETRAIN_DIR}"
echo "[mqt] out        -> ${OUTPUT_DIR}"

cd /home/skalra/FlexLLaVA/MQT-LLaVA

# ---------------------------------------------------------------------------
# CRITICAL: make MQT's vendored `llava` win over our pip-installed one.
#
# FlexLLaVA is installed editable, and setuptools registers a MetaPathFinder
# mapping 'llava' -> /home/skalra/FlexLLaVA/llava. `cd MQT-LLaVA` is NOT enough
# to beat it: when deepspeed runs `llava/train/train.py`, sys.path[0] is the
# SCRIPT's directory (.../MQT-LLaVA/llava/train), not the cwd, so nothing puts
# MQT's package on sys.path and every `import llava` resolves to OURS. That is
# not a subtle difference -- MQT's train.py then drives our model, and dies with
# "'LlavaLlamaModel' object has no attribute 'query_abstractor'" (jobs 27418,
# 27420). Silent-wrong-model is the worse failure mode: our model has an
# mm_projector, so a differently-shaped run could have trained to completion and
# produced numbers that were not MQT at all.
#
# The editable finder is APPENDED to sys.meta_path, so PathFinder (sys.path,
# which PYTHONPATH feeds) is consulted first and wins.
export PYTHONPATH=/home/skalra/FlexLLaVA/MQT-LLaVA${PYTHONPATH:+:${PYTHONPATH}}
_resolved=$(python3 -c "import llava, os; print(os.path.dirname(llava.__file__))")
if [ "$_resolved" != "/home/skalra/FlexLLaVA/MQT-LLaVA/llava" ]; then
    echo "ERROR: 'llava' resolves to ${_resolved}, not MQT's." >&2
    echo "       Refusing to run: this would train OUR model under MQT's script." >&2
    exit 1
fi
echo "[mqt] llava package -> ${_resolved} (MQT's, verified)"
# ---------------------------------------------------------------------------
deepspeed --num_gpus ${NUM_GPUS} llava/train/train.py \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path "${MODEL_PATH}" \
    --version "${CONV_VERSION}" \
    --data_path /var/scratch/skalra/flexllava/data/LLaVA-Finetune/llava_v1_5_mix665k.json \
    --image_folder /var/scratch/skalra/flexllava/data/LLaVA-Finetune \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --mm_query_abstractor_type matry_query \
    --pretrain_mm_query_abstractor "${PRETRAIN_DIR}/query_abstractor.bin" \
    --num_visual_tokens "${NUM_VISUAL_TOKENS}" \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir "${OUTPUT_DIR}" \
    --num_train_epochs 1 \
    ${MAX_STEPS_ARG} \
    --per_device_train_batch_size ${PER_DEVICE} \
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
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to none
