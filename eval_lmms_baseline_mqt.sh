#!/bin/bash
#SBATCH --job-name=eval_mqt
#SBATCH -t 48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:A10:1
#SBATCH --cpus-per-task=8
#SBATCH --output=./jobs/eval_mqt_%A.out

# Baseline: MQT-LLaVA (TinyLlama-1.1B) at its trained 4-token budget, via the
# dedicated lmms_eval/models/llava_mqt.py wrapper (see that file's own
# docstring for why it is a fork of llava.py rather than a reuse of it --
# short version: the generic wrapper passes the wrong token-count kwarg to
# MQT-LLaVA's generate() and silently swallows the resulting failure as an
# empty prediction rather than raising).
#
# Requires MQT-LLaVA's own `llava` package on PYTHONPATH ahead of
# FlexLLaVA's editable-installed one -- set below, before python3 starts (an
# in-script guard is too late; see debug/measure_mqt_rank.py's docstring for
# the full explanation of why `cd MQT-LLaVA` alone does not work).
#
# Submit: sbatch eval_lmms_baseline_mqt.sh [checkpoint_dir] [num_visual_tokens]

MODEL_PATH="${1:-/var/scratch/skalra/flexllava/checkpoints/mqt-finetune-tinyllama-4tok}"
NUM_VISUAL_TOKENS="${2:-4}"
MODEL_TAG="$(basename "$MODEL_PATH")"
LABEL="${NUM_VISUAL_TOKENS}tok"

# Same five tasks as every other row in Table III, plus VQAv2 as a separate
# TASKS= override (matching the two-job pattern already used for the other
# backbones -- VQAv2's 214k questions run several hours longer than the rest
# combined).
TASKS="${TASKS:-mme,pope,scienceqa_img,textvqa_val,gqa}"
BATCH_SIZE="${BATCH_SIZE:-2}"

LOG_ROOT=/var/scratch/skalra/flexllava/eval_logs
OUTDIR="${LOG_ROOT}/${MODEL_TAG}/${LABEL}"

module load cuda12.1/toolkit/12.1
eval "$(conda shell.bash hook)"
conda activate matryoshka-mm

export HF_HOME=/var/scratch/skalra/.cache/huggingface
export HF_DATASETS_CACHE=/var/scratch/skalra/.cache/huggingface/datasets
export PYTHONPATH=/home/skalra/FlexLLaVA/MQT-LLaVA${PYTHONPATH:+:${PYTHONPATH}}

# cd into MQT-LLaVA itself, NOT the FlexLLaVA repo root: python3 -m / -c both
# implicitly prepend cwd as sys.path[0]='', which wins over PYTHONPATH. Since
# FlexLLaVA's own repo root also contains a llava/ package, cd'ing there
# silently re-shadows MQT's package even with PYTHONPATH set correctly --
# caught by the resolution guard below during the first smoke-test attempt
# (job 27521), which resolved to FlexLLaVA's own llava instead of MQT's.
cd /home/skalra/FlexLLaVA/MQT-LLaVA

echo "Job started: $(date)"
echo "Node: $(hostname)"
nvidia-smi | head -12

_resolved=$(python3 -c "import llava, os; print(os.path.dirname(llava.__file__))")
if [ "$_resolved" != "/home/skalra/FlexLLaVA/MQT-LLaVA/llava" ]; then
    echo "ERROR: 'llava' resolves to ${_resolved}, not MQT's." >&2
    exit 1
fi
echo "[mqt] llava package -> ${_resolved} (MQT's, verified)"

echo "Model:  $MODEL_PATH"
echo "Tag:    $MODEL_TAG   num_visual_tokens=$NUM_VISUAL_TOKENS"
echo "Tasks:  $TASKS"
echo "Batch:  $BATCH_SIZE"

pip show lmms-eval >/dev/null 2>&1 || pip install -e /home/skalra/FlexLLaVA/lmms-eval -q
mkdir -p "$OUTDIR"

echo ""
echo "══════════════════════════════════════════════════"
echo "  BASELINE  ${MODEL_TAG}  (MQT-LLaVA, ${NUM_VISUAL_TOKENS}-token budget)"
echo "══════════════════════════════════════════════════"

# Not using accelerate launch here: MQT-LLaVA's own vendored code (unlike
# our fork) was not written/tested against accelerate's multi-process
# wrapping, and this job requests exactly one GPU, so a plain python3 -m
# invocation is both simpler and avoids exercising an untested path.
python3 -m lmms_eval \
    --model       llava_mqt \
    --model_args  "pretrained=${MODEL_PATH},conv_template=vicuna_v1,num_visual_tokens=${NUM_VISUAL_TOKENS}" \
    --tasks       "$TASKS" \
    --batch_size  $BATCH_SIZE \
    --log_samples \
    --log_samples_suffix "baseline_${LABEL}" \
    --output_path "$OUTDIR"

echo ""
echo "Done baseline: $(date)"

# No analytic-efficiency block here (unlike eval_lmms_baseline_llava.sh):
# llava.eval.efficiency is part of FlexLLaVA's OWN llava package, which this
# job's PYTHONPATH deliberately shadows in favour of MQT's -- and Table III
# (what these results are for) does not carry efficiency columns anyway.
