#!/bin/bash
#SBATCH --job-name=eval_lmms
#SBATCH -t 2-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:A40:1
#SBATCH --cpus-per-task=8
#SBATCH --output=./jobs/eval_lmms_%A.out

# ┌──────────────────────────────────────────────────────────────────────────┐
# │ DEPRECATED / UNMAINTAINED -- use eval_lmms_level.sh instead.              │
# │                                                                          │
# │ This runs all 4 token levels serially inside ONE job; eval_lmms_level.sh  │
# │ runs them as a 4-task SLURM array, i.e. 4 GPUs in parallel. It has never  │
# │ been submitted (no jobs/eval_lmms_<jobid>.out has ever existed) and is    │
# │ kept only as a fallback for when array jobs are inconvenient.            │
# │                                                                          │
# │ Because it duplicates TASKS / EFF_HARDWARE / conv-template detection, it  │
# │ WILL drift from eval_lmms_level.sh -- it already has. Diff the two before │
# │ trusting any number this produces.                                       │
# └──────────────────────────────────────────────────────────────────────────┘
#
# FlexLLaVA evaluation via lmms-eval (same framework as AdaLLaVA).
# Evaluates all 4 token levels on 6 benchmarks; datasets auto-download from HF.
#
# Benchmarks (matching AdaLLaVA paper protocol, minus mmbench_en_dev --
# excluded, see TASKS below):
#   mme · pope · scienceqa_img · textvqa_val · gqa
#
# Data downloads to HF_HOME/datasets (~15 GB total, cached across runs).
# No manual data preparation required.
#
# Submit: sbatch eval_lmms_job.sh [model_path]

MODEL_PATH=${1:-/var/scratch/skalra/flexllava/checkpoints/llava-elastic-pretrain}
# Namespace by model: without this, two different checkpoints evaluated at
# the same tok_level land in the same directory, and the summary table's
# glob (which only filters by tok_level + benchmark) silently mixes results
# from whichever model ran most recently.
export MODEL_TAG=$(basename "$MODEL_PATH")
LOG_ROOT=/var/scratch/skalra/flexllava/eval_logs
# mmbench_en_dev excluded -- see eval_lmms_level.sh for the full explanation
# (without OPENAI_API_KEY, unparseable answers get a random option, so the
# score is partly random rather than simply missing).
#
# vqav2_val matches AdaLLaVA's benchmark set. NOTE it is ~214k questions,
# roughly 7x the other five combined; override TASKS or set EVAL_LIMIT to
# trade completeness for turnaround.
TASKS="${TASKS:-mme,pope,scienceqa_img,textvqa_val,gqa,vqav2_val}"
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
echo "Model: $MODEL_PATH"

# Auto-detect the conversation template from the checkpoint's base LLM (see
# eval_lmms_level.sh for why: the vicuna_v1 default is wrong for SLM
# backbones trained with --version chatml/phi, and the mismatch is silent).
BASE_LLM=$(python3 -c "
import json
try:
    print(json.load(open('${MODEL_PATH}/config.json')).get('_name_or_path', '').lower())
except Exception:
    print('')
")
case "$BASE_LLM" in
    *tinyllama*|*qwen*|*stablelm*) CONV_TEMPLATE="chatml" ;;
    *phi*)                         CONV_TEMPLATE="phi" ;;
    *)                              CONV_TEMPLATE="vicuna_v1" ;;
esac
echo "Base LLM: $BASE_LLM  →  conv_template=${CONV_TEMPLATE}"

# Install lmms-eval into the env if not already done
pip show lmms-eval >/dev/null 2>&1 || pip install -e lmms-eval -q

TOK_LABELS=("256tok" "144tok" "64tok" "16tok")

for LEVEL in 0 1 2 3; do
    LABEL=${TOK_LABELS[$LEVEL]}
    OUTDIR="${LOG_ROOT}/${MODEL_TAG}/${LABEL}"
    mkdir -p "$OUTDIR"

    echo ""
    echo "══════════════════════════════════════════════════"
    echo "  Evaluating tok_level=${LEVEL}  (${LABEL})"
    echo "══════════════════════════════════════════════════"

    accelerate launch --num_processes=1 \
        -m lmms_eval \
        --model       llava_elastic \
        --model_args  "pretrained=${MODEL_PATH},tok_level=${LEVEL},device_map=cuda:0,conv_template=${CONV_TEMPLATE},eff_hardware=${EFF_HARDWARE}" \
        --tasks       "$TASKS" \
        --batch_size  1 \
        --log_samples \
        --log_samples_suffix "elastic_${LABEL}" \
        --output_path "$OUTDIR"

    echo "  Done ${LABEL}: $(date)"
done

# ── Print comparison table ────────────────────────────────────────────────────
SUMMARY="./lmms-eval/eval_lmms_summary_${SLURM_JOB_ID}.txt"

python3 - <<'PYEOF' | tee "$SUMMARY"
import json, glob, os

log_root = os.path.join("/var/scratch/skalra/flexllava/eval_logs", os.environ["MODEL_TAG"])
labels   = ["256tok", "144tok", "64tok", "16tok"]
benchmarks = ["mme", "pope", "scienceqa_img", "textvqa_val", "gqa"]

# AdaLLaVA paper numbers (latency=0.85, LLaVA-v1.5-7b, Table 2)
ada = {
    "mme":           "1487.2 / 324.6",
    "pope":          "85.9",
    "scienceqa_img": "70.4",
    "textvqa_val":   "58.1",
    "gqa":           "62.0",
}

def find_result(label, bench):
    pattern = f"{log_root}/{label}/**/*{bench}*.json"
    files = glob.glob(pattern, recursive=True)
    if not files:
        return "N/A"
    data = json.load(open(files[-1]))
    results = data.get("results", {})
    # Pull the first numeric metric value. Skip ",efficiency" keys: the
    # wrapper now folds FLOPs/time/memory into each task's result block, and
    # those would otherwise be printed as if they were an accuracy score.
    for task_key, metrics in results.items():
        if bench.split("_")[0] in task_key.lower():
            for k, v in metrics.items():
                if isinstance(v, (int, float)) and "stderr" not in k \
                        and not k.endswith(",efficiency"):
                    return f"{v:.1f}"
    return "N/A"

print()
print("╔══════════════════╦══════════╦══════════╦══════════╦══════════╦══════════════════╗")
print("║ Benchmark        ║  256tok  ║  144tok  ║   64tok  ║   16tok  ║ AdaLLaVA (0.85) ║")
print("╠══════════════════╬══════════╬══════════╬══════════╬══════════╬══════════════════╣")
for b in benchmarks:
    vals = [find_result(lbl, b) for lbl in labels]
    ada_val = ada.get(b, "—")
    row = f"║ {b:<16} ║ {vals[0]:>8} ║ {vals[1]:>8} ║ {vals[2]:>8} ║ {vals[3]:>8} ║ {ada_val:<16} ║"
    print(row)
print("╚══════════════════╩══════════╩══════════╩══════════╩══════════╩══════════════════╝")
print()
print("Stage-1 (pretrain-only): token ordering 256>144>64>16 validates Matryoshka training.")
print("For full AdaLLaVA comparison run Stage 2 (finetune_elastic.sh) then re-eval.")
print("AdaLLaVA paper: https://arxiv.org/abs/2503.10905")
PYEOF

echo ""
echo "Summary: $SUMMARY"
echo "Job complete: $(date)"
