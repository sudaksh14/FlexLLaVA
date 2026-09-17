#!/bin/bash
# Stage 1 — MQT-LLaVA baseline, using MQT's OWN train loop and code.
#
#   bash scripts/v1_5/pretrain_mqt_baseline.sh [LLM_KEY]
#
# Runs MQT-LLaVA/llava/train/train.py from inside MQT-LLaVA/ so that its
# vendored `llava` package shadows ours -- this is MQT's pipeline, not ours
# with an MQT-shaped part bolted on. Verified to import cleanly in the
# matryoshka-mm env despite MQT pinning transformers==4.36.2 (we run 4.44.2):
# job 27415.
#
# NOTE (verified, diag job 27419): MQT has NO mm_projector. Its
# initialize_vision_modules builds ONLY query_abstractor, and
# build_vision_projector does not exist anywhere in MQT's tree -- the Resampler's
# own `proj` (kv_dim -> embed_dim) does that job. So --mm_projector_type,
# --tune_mm_mlp_adapter and --pretrain_mm_mlp_adapter are all invalid here and
# are deliberately NOT passed; passing them crashes on a missing attribute.
#
# MQT's mechanism: a 2D perceiver Resampler ("query_abstractor",
# mm_query_abstractor_type=matry_query) with 256 learnable queries + frozen 2D
# sincos pos-emb, cross-attending to the CLIP patches. num_visual_tokens picks
# how many of those queries are kept, matryoshka-style.
# Per MQT's own recipe, Stage 1 uses the full query bank:
#   num_visual_tokens=first_stage  -> get_matry_n() returns 256
#
# Data and hyperparameters are OURS, deliberately, so the baseline is
# comparable to our runs: blip_laion_cc_sbu_558k, LR 1e-3, 1 epoch, effective
# batch 256, bf16, CLIP-L/336 frozen, LLM frozen.
set -uo pipefail
LLM_KEY=${1:-tinyllama}
case "$LLM_KEY" in
  tinyllama) MODEL_PATH="TinyLlama/TinyLlama-1.1B-Chat-v1.0" ;;
  *) echo "MQT baseline is set up for tinyllama only; got '$LLM_KEY'" >&2; exit 1 ;;
esac

TAG="${BASELINE_RUN_TAG:+-${BASELINE_RUN_TAG}}"
OUTPUT_DIR="/var/scratch/skalra/flexllava/checkpoints/mqt-pretrain-${LLM_KEY}${TAG}"
NUM_GPUS="${NUM_GPUS:-2}"
PER_DEVICE="${PER_DEVICE:-8}"
# Effective batch 256, matching pretrain_baseline_slm.sh, invariant to NUM_GPUS.
GRAD_ACCUM="${GRAD_ACCUM:-$(( 256 / (PER_DEVICE * NUM_GPUS) ))}"
MAX_STEPS_ARG=""; [ -n "${MAX_STEPS:-}" ] && MAX_STEPS_ARG="--max_steps ${MAX_STEPS}"

echo "[mqt] Stage 1  LLM=${MODEL_PATH}  num_visual_tokens=first_stage (256 queries)"
echo "[mqt] gpus=${NUM_GPUS} per_device=${PER_DEVICE} accum=${GRAD_ACCUM} -> eff batch $((PER_DEVICE*GRAD_ACCUM*NUM_GPUS))"
echo "[mqt] out -> ${OUTPUT_DIR}"

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
    --version plain \
    --data_path /var/scratch/skalra/flexllava/data/LLaVA-Pretrain/blip_laion_cc_sbu_558k.json \
    --image_folder /var/scratch/skalra/flexllava/data/LLaVA-Pretrain \
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
    --report_to none
