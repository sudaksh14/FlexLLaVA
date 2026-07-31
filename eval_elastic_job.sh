#!/bin/bash
#SBATCH --job-name=eval_elastic
#SBATCH -t 12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:A40:1
#SBATCH --cpus-per-task=8
#SBATCH --output=./jobs/eval_%A.out
#SBATCH --export=ALL

# ┌──────────────────────────────────────────────────────────────────────────┐
# │ DEPRECATED / UNMAINTAINED -- use eval_lmms_level.sh instead.              │
# │                                                                          │
# │ This is the PRE-lmms-eval pipeline: it drives llava/eval/model_vqa_*.py   │
# │ against manually-downloaded benchmark data under playground/data/eval/    │
# │ and scrapes scores out of text files. It has never been submitted (no     │
# │ jobs/eval_<jobid>.out has ever existed), and it cannot run as-is: only    │
# │ playground/data/eval/answers/ exists -- the MME / pope / mmbench /        │
# │ scienceqa / textvqa / gqa data dirs it expects were never downloaded.     │
# │                                                                          │
# │ The lmms-eval path supersedes it entirely: datasets auto-download from    │
# │ HF, scoring is the official toolkit (same one AdaLLaVA uses), and it is   │
# │ the only path that carries the elastic engine + efficiency metrics.       │
# │ Kept for reference on the old per-benchmark answer-file layout only.      │
# └──────────────────────────────────────────────────────────────────────────┘
#
# Evaluate FlexLLaVA elastic pretrain checkpoint at all 4 token levels.
# Benchmarks: MME · POPE · MMBench · ScienceQA · TextVQA · GQA
# Results are aggregated into a comparison table at the end.
#
# Before running:
#   1. Place eval data under ./playground/data/eval/ (see eval_elastic.sh header)
#   2. Optionally point MODEL_PATH to a Stage-2 fine-tuned checkpoint for full
#      AdaLLaVA comparison (Stage 1 pretrain shows relative token-level ordering
#      but absolute numbers are lower without instruction tuning).
#
# Submit: sbatch eval_elastic_job.sh

MODEL_PATH=/var/scratch/skalra/flexllava/checkpoints/llava-elastic-pretrain
EVAL_DATA=./playground/data

module load cuda12.6/toolkit/12.6

eval "$(conda shell.bash hook)"
conda activate matryoshka-mm

export HF_HOME=/var/scratch/skalra/.cache/huggingface

echo "Job started: $(date)"
echo "Node: $(hostname)"
nvidia-smi | head -15

TOK_LABELS=("256tok" "144tok" "64tok" "16tok")

for LEVEL in 0 1 2 3; do
    LABEL=${TOK_LABELS[$LEVEL]}
    echo ""
    echo "############################################################"
    echo "  Evaluating token level ${LEVEL} (${LABEL})"
    echo "############################################################"
    bash scripts/v1_5/eval/eval_elastic.sh "$LEVEL" "$MODEL_PATH" "$EVAL_DATA"
done

# ── Aggregate results into a single comparison table ────────────────────────
ANSWERS_BASE="${EVAL_DATA}/eval/answers"
SUMMARY="./jobs/eval_summary_${SLURM_JOB_ID}.txt"

echo "" | tee "$SUMMARY"
echo "╔══════════════════════════════════════════════════════════════════╗" | tee -a "$SUMMARY"
echo "║         FlexLLaVA Elastic Eval  –  All Token Levels             ║" | tee -a "$SUMMARY"
echo "╠══════════╦════════╦════════╦════════╦════════╦══════════════════╣" | tee -a "$SUMMARY"
echo "║ Benchmark║ 256tok ║ 144tok ║  64tok ║  16tok ║ AdaLLaVA (0.85) ║" | tee -a "$SUMMARY"
echo "╠══════════╬════════╬════════╬════════╬════════╬══════════════════╣" | tee -a "$SUMMARY"

extract_mme() {
    local dir="$1"
    grep -h "perception\|cognition" "${dir}/mme_result.txt" 2>/dev/null \
        | awk '{sum+=$NF} END {printf "%.1f", sum}' || echo "N/A"
}
extract_pope() {
    local dir="$1"
    grep -i "accuracy" "${dir}/pope_result.txt" 2>/dev/null \
        | grep -oP '\d+\.\d+' | tail -1 || echo "N/A"
}
extract_textvqa() {
    local dir="$1"
    grep -i "accuracy" "${dir}/textvqa_result.txt" 2>/dev/null \
        | grep -oP '\d+\.\d+' | tail -1 || echo "N/A"
}
extract_sqa() {
    local dir="$1"
    python3 -c "
import json, sys
try:
    r = json.load(open('${dir}/scienceqa_result.json'))
    print(f\"{r.get('acc', r.get('accuracy', 'N/A')):.1f}\")
except: print('N/A')
" 2>/dev/null || echo "N/A"
}
extract_gqa() {
    local dir="$1"
    grep -i "accuracy" "${dir}/gqa_result.txt" 2>/dev/null \
        | grep -oP '\d+\.\d+' | tail -1 || echo "N/A"
}

# Reference numbers from AdaLLaVA paper (latency=0.85, LLaVA-v1.5-7B backbone)
ADA_MME="1487.2"
ADA_POPE="85.9"
ADA_TVQ="58.1"
ADA_SQA="70.4"
ADA_GQA="62.0"

print_row() {
    local bench=$1 fn=$2
    vals=()
    for LEVEL in 0 1 2 3; do
        N_TOK_ARR=(256 144 64 16)
        DIR="${ANSWERS_BASE}/elastic-pretrain-tok${N_TOK_ARR[$LEVEL]}"
        vals+=("$($fn $DIR)")
    done
    printf "║ %-8s ║ %6s ║ %6s ║ %6s ║ %6s ║ %-16s ║\n" \
        "$bench" "${vals[0]}" "${vals[1]}" "${vals[2]}" "${vals[3]}" \
        "$(eval echo \$ADA_${bench^^} 2>/dev/null || echo 'see paper')" \
        | tee -a "$SUMMARY"
}

print_row "MME"     extract_mme
print_row "POPE"    extract_pope
print_row "TextVQA" extract_textvqa
print_row "SQA"     extract_sqa
print_row "GQA"     extract_gqa

echo "╚══════════╩════════╩════════╩════════╩════════╩══════════════════╝" | tee -a "$SUMMARY"
echo "" | tee -a "$SUMMARY"
echo "Note: These are STAGE-1 (pretrain-only) results. Absolute numbers are" | tee -a "$SUMMARY"
echo "lower than AdaLLaVA (which is fully fine-tuned). Key metric: ordering" | tee -a "$SUMMARY"
echo "256tok > 144tok > 64tok > 16tok confirms Matryoshka training worked." | tee -a "$SUMMARY"
echo "Run Stage 2 (finetune_elastic.sh) for a fair AdaLLaVA comparison." | tee -a "$SUMMARY"
echo "" | tee -a "$SUMMARY"
echo "AdaLLaVA reference: https://arxiv.org/abs/2503.10905  (latency=0.85 on V100)" | tee -a "$SUMMARY"

echo ""
echo "Summary written to: $SUMMARY"
echo "Job complete: $(date)"
