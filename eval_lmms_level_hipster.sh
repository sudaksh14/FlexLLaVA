#!/bin/bash
#SBATCH --job-name=eval_lmms_hipster
#SBATCH -t 24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
# Overridden at submit time by submit_elastic_run_hipster.sh for the chosen
# partition (performance=rtx_6000_ada, capacity=l4).
#SBATCH --partition=capacity
#SBATCH --gres=gpu:l4:1
#SBATCH --cpus-per-task=16
#SBATCH --output=./jobs/eval_lmms_hipster_%A_tok%a.out
#SBATCH --array=0-3
#
# Port of eval_lmms_level.sh for hipster. Same tok_levels-from-checkpoint
# logic, same task set, same conv-template auto-detect -- only paths, module
# name, and the GPU-name -> batch-size table differ.

MODEL_PATH=${1:?usage: sbatch eval_lmms_level_hipster.sh <model_path>}
LEVEL=$SLURM_ARRAY_TASK_ID

module load cuda/12.9.1 2>&1 || echo "WARNING: module load cuda/12.9.1 failed -- continuing"

eval "$(conda shell.bash hook)"
conda activate matryoshka-mm

if [ ! -f "${MODEL_PATH}/elastic_config.json" ]; then
    echo "ERROR: ${MODEL_PATH}/elastic_config.json not found -- cannot determine tok_levels." >&2
    exit 1
fi
mapfile -t TOK_LEVELS_JSON < <(python3 -c "
import json
cfg = json.load(open('${MODEL_PATH}/elastic_config.json'))
for t in cfg['tok_levels']:
    print(t)
")
if [ ${#TOK_LEVELS_JSON[@]} -eq 0 ]; then
    echo "ERROR: failed to parse tok_levels out of ${MODEL_PATH}/elastic_config.json" >&2
    exit 1
fi
TOK_LABELS=()
for t in "${TOK_LEVELS_JSON[@]}"; do TOK_LABELS+=("${t}tok"); done
LABEL=${TOK_LABELS[$LEVEL]}
if [ -z "$LABEL" ]; then
    echo "ERROR: no tok_levels entry at index $LEVEL (checkpoint has ${#TOK_LABELS[@]} levels)." \
         "Check --array matches the checkpoint's actual grid size." >&2
    exit 1
fi

MODEL_TAG=$(basename "$MODEL_PATH")
LOG_ROOT=/home/skalra/flexllava_saves/eval_logs
OUTDIR="${LOG_ROOT}/${MODEL_TAG}/${LABEL}"

# mmbench_en_dev excluded (needs OPENAI_API_KEY or falls back to random
# option assignment) -- same reasoning as DAS-6's eval_lmms_level.sh.
TASKS="${TASKS:-mme,pope,scienceqa_img,textvqa_val,gqa}"

LIMIT_ARG=""
if [ -n "$EVAL_LIMIT" ]; then
    LIMIT_ARG="--limit $EVAL_LIMIT"
    echo "WARNING: --limit $EVAL_LIMIT is set; scores are NOT comparable to published numbers."
fi

EFF_HARDWARE="${EFF_HARDWARE:-jetson_orin_nano_8gb}"

export HF_HOME=/home/skalra/flexllava_saves/cache/huggingface
export HF_DATASETS_CACHE=/home/skalra/flexllava_saves/cache/huggingface/datasets

cd /home/skalra/FlexLLaVA

echo "Job started: $(date)"
echo "Node: $(hostname)"
nvidia-smi | head -12
echo "Model:     $MODEL_PATH"
echo "tok_level: $LEVEL ($LABEL)"

# Set batch size based on GPU VRAM: rtx_6000_ada=48GB -> 8, l4=24GB -> 4,
# fallback -> 1. Mirrors DAS-6's A40/A10 table at the same VRAM tiers.
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
if echo "$GPU_NAME" | grep -qi "6000 Ada"; then
    BATCH_SIZE=8
elif echo "$GPU_NAME" | grep -qi "L4"; then
    BATCH_SIZE=4
else
    BATCH_SIZE=1
fi
echo "GPU: $GPU_NAME  →  batch_size=${BATCH_SIZE}"

# Conv-template auto-detect -- MUST mirror the CONV_VERSION case in
# scripts/v1_5/finetune_elastic_slm_hipster.sh.
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

# mkdir-based lock, not a bare `pip show || pip install`: this script runs
# as 4 simultaneous array tasks, POSSIBLY ON DIFFERENT COMPUTE NODES, sharing
# one NFS-mounted conda env. If lmms-eval isn't installed yet, they race the
# same `pip install -e` concurrently, corrupting each other's pip cache (seen
# 2026-09-13: 3 of 4 tasks failed with a mangled `origin.json` / missing
# distlib file and silently produced no results, since this script has no
# `set -e` -- SLURM still reported them COMPLETED).
#
# `mkdir` is used instead of `flock` because it's atomic on essentially any
# filesystem including NFS -- flock's own semantics over NFS are not reliably
# consistent across NFS versions/servers, and the lock must be visible across
# nodes (a /tmp-local lock would not serialize tasks landing on different
# nodes). Whichever task creates the directory first holds the lock; the
# others poll until it's gone.
LOCKDIR=/home/skalra/flexllava_saves/.lmms_eval_install.lock
LOCK_ACQUIRED=0
for i in $(seq 1 300); do
    if mkdir "$LOCKDIR" 2>/dev/null; then
        LOCK_ACQUIRED=1
        trap 'rmdir "$LOCKDIR" 2>/dev/null' EXIT
        pip show lmms-eval >/dev/null 2>&1 || pip install -e lmms-eval -q
        rmdir "$LOCKDIR" 2>/dev/null
        trap - EXIT
        break
    fi
    # Stale-lock recovery: a task that died mid-install (OOM-killed, node
    # fault, etc.) without reaching `rmdir` would otherwise wedge this lock
    # for every future eval job forever. 90s is generously above how long
    # even a fresh `pip install -e` of an already-local repo takes.
    if [ "$i" -eq 90 ] && [ -d "$LOCKDIR" ]; then
        LOCK_AGE=$(( $(date +%s) - $(stat -c %Y "$LOCKDIR" 2>/dev/null || echo 0) ))
        if [ "$LOCK_AGE" -gt 90 ]; then
            echo "WARNING: $LOCKDIR is >90s old -- assuming a stale lock from a" \
                 "dead process and removing it." >&2
            rmdir "$LOCKDIR" 2>/dev/null
        fi
    fi
    sleep 1
done
if [ "$LOCK_ACQUIRED" -ne 1 ]; then
    echo "ERROR: could not acquire $LOCKDIR after 300s -- refusing to proceed" \
         "without confirming lmms-eval is installed (this is what silently" \
         "produced zero results on 2026-09-13)." >&2
    exit 1
fi

mkdir -p "$OUTDIR"

echo ""
echo "Evaluating tok_level=${LEVEL}  (${LABEL})"
echo "Tasks: $TASKS"

accelerate launch --num_processes=1 \
    -m lmms_eval \
    --model       llava_elastic \
    --model_args  "pretrained=${MODEL_PATH},tok_level=${LEVEL},device_map=cuda:0,conv_template=${CONV_TEMPLATE},eff_hardware=${EFF_HARDWARE}" \
    --tasks       "$TASKS" \
    --batch_size  $BATCH_SIZE \
    $LIMIT_ARG \
    --log_samples \
    --log_samples_suffix "elastic_${LABEL}" \
    --output_path "$OUTDIR"

echo ""
echo "Done ${LABEL}: $(date)"
