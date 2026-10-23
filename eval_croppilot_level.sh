#!/bin/bash
#SBATCH --job-name=croppilot
#SBATCH -t 06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --exclude=node205
#SBATCH --output=/var/scratch/skalra/flexllava/croppilot/logs/croppilot_%x_%A_tok%a.out
#SBATCH --array=0-3

# Crop pilot (brainstorm.md rank 1): TextVQA val on the ViCrop GT-box subset,
# one tok_level per array task, same setup as eval_lmms_level.sh.
#
#   MODE=full|crop|globalcrop  sbatch eval_croppilot_level.sh [model_path]
#   MODE=crop EVAL_LIMIT=24 sbatch --array=0 eval_croppilot_level.sh      # smoke test
#
# GPU jobs run on node206-208 only (node205 excluded).  Results land in
# /var/scratch/skalra/flexllava/eval_logs/croppilot/<MODE>/<model>/<label>.

MODEL_PATH=${1:-/var/scratch/skalra/flexllava/checkpoints/elastic-finetune-tinyllama-v8-parcel}
MODE=${MODE:?set MODE=full|crop|globalcrop}
LEVEL=$SLURM_ARRAY_TASK_ID
TASKS="textvqa_val_gtbox_${MODE}"

module load cuda12.6/toolkit/12.6

eval "$(conda shell.bash hook)"
conda activate matryoshka-mm

if [ ! -f "${MODEL_PATH}/elastic_config.json" ]; then
    echo "ERROR: ${MODEL_PATH}/elastic_config.json not found." >&2
    exit 1
fi
mapfile -t TOK_LEVELS_JSON < <(python3 -c "
import json
cfg = json.load(open('${MODEL_PATH}/elastic_config.json'))
for t in cfg['tok_levels']:
    print(t)
")
TOK_LABELS=()
for t in "${TOK_LEVELS_JSON[@]}"; do TOK_LABELS+=("${t}tok"); done
LABEL=${TOK_LABELS[$LEVEL]}
if [ -z "$LABEL" ]; then
    echo "ERROR: no tok_levels entry at index $LEVEL (checkpoint has ${#TOK_LABELS[@]} levels)." >&2
    exit 1
fi

MODEL_TAG=$(basename "$MODEL_PATH")
# Box source.  Default = the ViCrop ground-truth boxes (oracle).  For another
# source set SRC_TAG=<name> (boxes read from textvqa_val_selfroi_<name>.json);
# its results go to eval_logs/croppilot_<name>/ so oracle logs are never mixed.
SRC_TAG=${SRC_TAG:-}
if [ -n "$SRC_TAG" ]; then
    export CROPPILOT_SIDECAR=/var/scratch/skalra/flexllava/croppilot/textvqa_val_selfroi_${SRC_TAG}.json
    [ -f "$CROPPILOT_SIDECAR" ] || { echo "ERROR: $CROPPILOT_SIDECAR not found" >&2; exit 1; }
    OUTDIR="/var/scratch/skalra/flexllava/eval_logs/croppilot_${SRC_TAG}/${MODE}/${MODEL_TAG}/${LABEL}"
else
    export CROPPILOT_SIDECAR=/var/scratch/skalra/flexllava/croppilot/textvqa_val_gtbox.json
    OUTDIR="/var/scratch/skalra/flexllava/eval_logs/croppilot/${MODE}/${MODEL_TAG}/${LABEL}"
fi

LIMIT_ARG=""
if [ -n "$EVAL_LIMIT" ]; then
    LIMIT_ARG="--limit $EVAL_LIMIT"
    echo "WARNING: --limit $EVAL_LIMIT is set; smoke test only, scores are NOT results."
    OUTDIR="${OUTDIR}_limit${EVAL_LIMIT}"
fi

EFF_HARDWARE="${EFF_HARDWARE:-jetson_orin_nano_8gb}"

export HF_HOME=/var/scratch/skalra/.cache/huggingface
export HF_DATASETS_CACHE=/var/scratch/skalra/.cache/huggingface/datasets

cd /home/skalra/FlexLLaVA

echo "Job started: $(date)"
echo "Node: $(hostname)"
nvidia-smi | head -12
echo "Model:     $MODEL_PATH"
echo "Mode:      $MODE   tasks: $TASKS   box source: ${SRC_TAG:-oracle} ($CROPPILOT_SIDECAR)"
echo "tok_level: $LEVEL ($LABEL)"
echo "Outdir:    $OUTDIR"

GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
if echo "$GPU_NAME" | grep -q "A40"; then
    BATCH_SIZE=8
elif echo "$GPU_NAME" | grep -q "A10"; then
    BATCH_SIZE=4
else
    BATCH_SIZE=1
fi
echo "GPU: $GPU_NAME  →  batch_size=${BATCH_SIZE}"

# Same template auto-detection as eval_lmms_level.sh (tinyllama -> vicuna_v1).
BASE_LLM=$(python3 -c "
import json
try:
    print(json.load(open('${MODEL_PATH}/config.json')).get('_name_or_path', '').lower())
except Exception:
    print('')
")
case "$BASE_LLM" in
    *phi-3*|*phi3*)                       CONV_TEMPLATE="phi3" ;;
    *tinyllama*|*mobilellama*)            CONV_TEMPLATE="vicuna_v1" ;;
    *qwen*|*stablelm*|*smollm*)           CONV_TEMPLATE="chatml" ;;
    *phi*)                                CONV_TEMPLATE="phi" ;;
    *)                                    CONV_TEMPLATE="vicuna_v1" ;;
esac
echo "Base LLM: $BASE_LLM  →  conv_template=${CONV_TEMPLATE}"

pip show lmms-eval >/dev/null 2>&1 || pip install -e lmms-eval -q

mkdir -p "$OUTDIR"

accelerate launch --num_processes=1 \
    -m lmms_eval \
    --model       llava_elastic \
    --model_args  "pretrained=${MODEL_PATH},tok_level=${LEVEL},device_map=cuda:0,conv_template=${CONV_TEMPLATE},eff_hardware=${EFF_HARDWARE}" \
    --tasks       "$TASKS" \
    --batch_size  $BATCH_SIZE \
    $LIMIT_ARG \
    --log_samples \
    --log_samples_suffix "croppilot_${MODE}_${LABEL}" \
    --output_path "$OUTDIR"

echo "Done ${MODE} ${LABEL}: $(date)"
