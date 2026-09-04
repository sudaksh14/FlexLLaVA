#!/bin/bash
#SBATCH --job-name=eval_lmms
#SBATCH -t 24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --output=./jobs/eval_lmms_%A_tok%a.out
#SBATCH --array=0-3

# Evaluates one tok_level per SLURM array task.
# %a in the output filename correctly maps to 0-3.
#
# Submit all 4 levels:  sbatch eval_lmms_level.sh [model_path]
# Submit one level:     sbatch --array=2 eval_lmms_level.sh [model_path]
# Checkpoint with a different grid size (e.g. an 8-level experiment):
#   sbatch --array=0-7 eval_lmms_level.sh [model_path]
# Labels are read from the checkpoint's own elastic_config.json, so any grid
# size works as long as --array matches its length.

MODEL_PATH=${1:-/var/scratch/skalra/flexllava/checkpoints/llava-elastic-pretrain}
LEVEL=$SLURM_ARRAY_TASK_ID

# Read tok_levels from the checkpoint's own elastic_config.json rather than
# hardcoding 4 -- a checkpoint trained with a different grid (e.g. the
# extended 8-level range experiment) needs a matching label list, and
# --array must be overridden to match too: sbatch --array=0-7 eval_lmms_level.sh <ckpt>
if [ -f "${MODEL_PATH}/elastic_config.json" ] && command -v jq >/dev/null; then
    mapfile -t TOK_LEVELS_JSON < <(jq -r '.tok_levels[]' "${MODEL_PATH}/elastic_config.json")
    TOK_LABELS=()
    for t in "${TOK_LEVELS_JSON[@]}"; do TOK_LABELS+=("${t}tok"); done
else
    echo "WARNING: ${MODEL_PATH}/elastic_config.json not found or jq missing;" \
         "falling back to the default 4-level label list."
    TOK_LABELS=("256tok" "144tok" "64tok" "16tok")
fi
LABEL=${TOK_LABELS[$LEVEL]}
if [ -z "$LABEL" ]; then
    echo "ERROR: no tok_levels entry at index $LEVEL (checkpoint has ${#TOK_LABELS[@]} levels)." \
         "Check --array matches the checkpoint's actual grid size." >&2
    exit 1
fi
# Namespace by model: without this, two different checkpoints evaluated at
# the same tok_level land in the same directory, and the summary script's
# glob (which only filters by tok_level + benchmark) silently mixes results
# from whichever model ran most recently.
MODEL_TAG=$(basename "$MODEL_PATH")
LOG_ROOT=/var/scratch/skalra/flexllava/eval_logs
OUTDIR="${LOG_ROOT}/${MODEL_TAG}/${LABEL}"
# mmbench_en_dev excluded. Its only real metric is gpt_eval_score, scored in
# two stages (lmms_eval/tasks/mmbench/mmbench_evals.py):
#   1. can_infer() heuristically maps the model's free-form answer onto one of
#      the A/B/C/D options -- this needs no API key and handles clean answers.
#   2. anything it cannot parse is sent to an OpenAI judge. With no key,
#      get_chat_response() returns "Failed to obtain answer via API", the 3
#      retries burn, and extract_answer_from_item() falls through to
#      `rd.randint(...)` -- a RANDOM option.
# So without a key MMBench is not "broken", it is silently *partly random*:
# verbose or hedging answers get a coin flip. That biases smaller/chattier
# backbones exactly where we care, so the number is not trustworthy.
# Re-enable by adding mmbench_en_dev to TASKS once OPENAI_API_KEY is set.
#
# vqav2_val included to match AdaLLaVA's benchmark set (they use vqav2_test +
# EvalAI submission; _val scores locally so we get a number immediately).
# WARNING: VQAv2 val is ~214k questions -- on its own roughly 7x the other
# five tasks combined. Set EVAL_LIMIT to subsample (see LIMIT_ARG below), or
# override TASKS to drop vqav2_val for a quick turnaround, e.g.

# TASKS="${TASKS:-mme,pope,scienceqa_img,textvqa_val,gqa,vqav2_val}"
TASKS="${TASKS:-mme,pope,scienceqa_img,textvqa_val,gqa}"


# Optional per-task sample cap, e.g. EVAL_LIMIT=2000 sbatch eval_lmms_level.sh ...
# Applies to EVERY task, so a capped run is NOT comparable to published
# numbers -- use it for iteration speed, not for reporting.
LIMIT_ARG=""
if [ -n "$EVAL_LIMIT" ]; then
    LIMIT_ARG="--limit $EVAL_LIMIT"
    echo "WARNING: --limit $EVAL_LIMIT is set; scores are on a subsample and"
    echo "         are NOT comparable to published benchmark numbers."
fi

# Analytic efficiency metrics (FLOPs / prefill time / memory) are computed
# alongside accuracy, using the same roofline cost model as AdaLLaVA.
# See llava/eval/efficiency/NOTICE.md for what they do and don't capture.
EFF_HARDWARE="${EFF_HARDWARE:-jetson_orin_nano_8gb}"

module load cuda12.6/toolkit/12.6

eval "$(conda shell.bash hook)"
conda activate matryoshka-mm

export HF_HOME=/var/scratch/skalra/.cache/huggingface
export HF_DATASETS_CACHE=/var/scratch/skalra/.cache/huggingface/datasets

cd /home/skalra/FlexLLaVA

echo "Job started: $(date)"
echo "Node: $(hostname)"
nvidia-smi | head -12
echo "Model:     $MODEL_PATH"
echo "tok_level: $LEVEL ($LABEL)"

# Set batch size based on GPU VRAM: A40=46GB → 8, A10=24GB → 4, fallback → 1
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
if echo "$GPU_NAME" | grep -q "A40"; then
    BATCH_SIZE=8
elif echo "$GPU_NAME" | grep -q "A10"; then
    BATCH_SIZE=4
else
    BATCH_SIZE=1
fi
echo "GPU: $GPU_NAME  →  batch_size=${BATCH_SIZE}"

# Auto-detect the conversation template from the checkpoint's base LLM.
# lmms-eval's Llava/LlavaElastic wrapper defaults to vicuna_v1, which is only
# correct for the original llava-v1.5-7b checkpoints. Using the wrong template
# at eval time doesn't error -- it silently feeds the model prompts formatted
# differently than what it was finetuned on.
#
# This MUST mirror the CONV_VERSION case in scripts/v1_5/finetune_elastic_slm.sh:
#   tinyllama, mobilellama          -> v1   (== vicuna_v1)
#   qwen*, stablelm, smollm2        -> chatml
#   phi2                            -> phi
#   phi3.5                          -> phi3  (MUST precede the *phi* case)
# TinyLlama is v1, NOT chatml: its tokenizer never registered
# <|im_start|>/<|im_end|> as special tokens, so they fragment into context-
# dependent BPE pieces and silently mask ~86% of labels. This block used to map
# tinyllama -> chatml, contradicting the training scripts.
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

echo ""
echo "══════════════════════════════════════════════════"
echo "  Evaluating tok_level=${LEVEL}  (${LABEL})"
echo "══════════════════════════════════════════════════"

echo "Efficiency model: hardware=${EFF_HARDWARE} (analytic roofline)"
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
echo ""
echo "Combined accuracy + efficiency table (once all 4 array tasks finish):"
echo "  python3 scripts/summarize_eval.py ${MODEL_TAG}"
