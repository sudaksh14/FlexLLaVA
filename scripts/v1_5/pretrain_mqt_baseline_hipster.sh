#!/bin/bash
# Stage 1 — MQT-LLaVA baseline, HIPSTER variant, using MQT's OWN train loop.
#
# Runs MQT-LLaVA/llava/train/train.py. MQT's mechanism is a 2D perceiver
# Resampler ("query_abstractor", mm_query_abstractor_type=matry_query): 256
# learnable queries plus a frozen 2D sincos positional embedding,
# cross-attending to the CLIP patches. num_visual_tokens keeps a prefix of
# them. Per MQT's own recipe Stage 1 uses the full bank:
#   num_visual_tokens=first_stage -> get_matry_n() returns 256
#
# MQT has NO mm_projector (verified on DAS-6, diag job 27419):
# build_vision_projector does not exist anywhere in its tree, and
# initialize_vision_modules builds only query_abstractor -- the Resampler's own
# `proj` (kv_dim -> embed_dim) does that job. So --mm_projector_type,
# --tune_mm_mlp_adapter and --pretrain_mm_mlp_adapter are INVALID here and are
# deliberately not passed; passing them crashes on a missing attribute.
#
# Data and hyperparameters are OURS on purpose, so the baseline is comparable
# to our runs: blip_laion_cc_sbu_558k, LR 1e-3, 1 epoch, effective batch 256,
# bf16, CLIP-L/336 frozen, LLM frozen.
set -uo pipefail
LLM_KEY=${1:-tinyllama}
LOCAL_SSD=${LOCAL_SSD:?LOCAL_SSD must be set by the caller (run_job_hipster_mqt_baseline.sh) to a node-local staged data directory}
SAVE_ROOT=/home/skalra/flexllava_saves
CHECKPOINT_ROOT=/scratch/skalra/flexllava_saves/checkpoints
REPO_ROOT=/home/skalra/FlexLLaVA

case "$LLM_KEY" in
  tinyllama) MODEL_PATH="TinyLlama/TinyLlama-1.1B-Chat-v1.0" ;;
  *) echo "MQT baseline is set up for tinyllama only; got '$LLM_KEY'" >&2; exit 1 ;;
esac

TAG="${BASELINE_RUN_TAG:+-${BASELINE_RUN_TAG}}"
OUTPUT_DIR="${CHECKPOINT_ROOT}/mqt-pretrain-${LLM_KEY}${TAG}"
LOG_DIR="${SAVE_ROOT}/logs/mqt-pretrain-${LLM_KEY}${TAG}"
RUN_NAME="mqt-pretrain-${LLM_KEY}${TAG}-hipster"

NUM_GPUS="${NUM_GPUS:-4}"
PER_DEVICE="${PER_DEVICE:-8}"
GRAD_ACCUM="${GRAD_ACCUM:-$(( 256 / (PER_DEVICE * NUM_GPUS) ))}"
MASTER_PORT="${MASTER_PORT:-$((29500 + RANDOM % 1000))}"
MAX_STEPS_ARG=""; [ -n "${MAX_STEPS:-}" ] && MAX_STEPS_ARG="--max_steps ${MAX_STEPS}"

echo "[mqt-baseline] Stage 1  LLM=${MODEL_PATH}  num_visual_tokens=first_stage (256 queries)"
echo "[mqt-baseline] gpus=${NUM_GPUS} per_device=${PER_DEVICE} accum=${GRAD_ACCUM} -> eff batch $((PER_DEVICE*GRAD_ACCUM*NUM_GPUS))"
echo "[mqt-baseline] data -> ${LOCAL_SSD}/LLaVA-Pretrain"
echo "[mqt-baseline] out  -> ${OUTPUT_DIR}"
mkdir -p "$LOG_DIR"

cd "${REPO_ROOT}/MQT-LLaVA"
# ---------------------------------------------------------------------------
# CRITICAL: make MQT's vendored `llava` win over our pip-installed one.
#
# FlexLLaVA is installed editable, and setuptools registers a MetaPathFinder
# mapping 'llava' -> <repo>/llava. `cd MQT-LLaVA` is NOT enough to beat it:
# when deepspeed runs `llava/train/train.py`, sys.path[0] is the SCRIPT's
# directory (.../MQT-LLaVA/llava/train), not the cwd, so nothing puts MQT's
# package on sys.path and every `import llava` resolves to OURS. Caught on
# DAS-6 (jobs 27418/27420): MQT's train.py drove OUR model and died on a
# missing query_abstractor. Silent-wrong-model is the worse failure mode --
# our model HAS an mm_projector, so a different flag set could have trained to
# completion and produced numbers that were not MQT at all.
#
# The editable finder is APPENDED to sys.meta_path, so PathFinder (which reads
# PYTHONPATH) is consulted first and wins.
export PYTHONPATH=${REPO_ROOT}/MQT-LLaVA${PYTHONPATH:+:${PYTHONPATH}}
_resolved=$(python3 -c "import llava, os; print(os.path.dirname(llava.__file__))")
if [ "$_resolved" != "${REPO_ROOT}/MQT-LLaVA/llava" ]; then
    echo "ERROR: 'llava' resolves to ${_resolved}, not MQT's." >&2
    echo "       Refusing to run: this would train OUR model under MQT's script." >&2
    exit 1
fi
echo "[mqt-baseline] llava package -> ${_resolved} (MQT's, verified)"
# ---------------------------------------------------------------------------

deepspeed --num_gpus ${NUM_GPUS} --master_port ${MASTER_PORT} llava/train/train.py \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path "${MODEL_PATH}" \
    --version plain \
    --data_path ${LOCAL_SSD}/LLaVA-Pretrain/blip_laion_cc_sbu_558k.json \
    --image_folder ${LOCAL_SSD}/LLaVA-Pretrain \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --mm_query_abstractor_type matry_query \
    --tune_mm_query_abstractor True \
    --num_visual_tokens first_stage \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir "${OUTPUT_DIR}" \
    --num_train_epochs 1 \
    ${MAX_STEPS_ARG} \
    --per_device_train_batch_size ${PER_DEVICE} \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps ${GRAD_ACCUM} \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 2000 \
    --save_total_limit 1 \
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
    --report_to wandb \
    --run_name "${RUN_NAME}" \
    --logging_dir "${LOG_DIR}"
