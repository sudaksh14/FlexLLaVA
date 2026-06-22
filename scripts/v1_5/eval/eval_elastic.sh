#!/bin/bash
# Evaluate a FlexLLaVA elastic checkpoint on all AdaLLaVA benchmarks at one
# token level.
#
# Usage:
#   bash scripts/v1_5/eval/eval_elastic.sh <tok_level> [model_path] [eval_data_dir]
#
# tok_level: index into tok_levels list
#   0 → 256 tokens  (teacher / full quality)
#   1 → 144 tokens
#   2 →  64 tokens
#   3 →  16 tokens  (most compressed)
#
# Benchmarks evaluated (matching AdaLLaVA paper):
#   MME · POPE · MMBench-en · ScienceQA · TextVQA · GQA
#
# Expected data layout under $EVAL_DATA (create playground symlink or override):
#   eval/MME/            eval/pope/         eval/mmbench/
#   eval/scienceqa/      eval/textvqa/      eval/gqa/
#
# The elastic checkpoint must contain elastic_config.json (written automatically
# by train.py; or see scripts to write it manually).

set -e

TOK_LEVEL=${1:-0}
MODEL_PATH=${2:-/var/scratch/skalra/flexllava/checkpoints/llava-elastic-pretrain}
EVAL_DATA=${3:-./playground/data}

# Derived names
TOK_COUNTS=(256 144 64 16)
N_TOK=${TOK_COUNTS[$TOK_LEVEL]}
CKPT="elastic-pretrain-tok${N_TOK}"
ANSWERS="${EVAL_DATA}/eval/answers/${CKPT}"
mkdir -p "$ANSWERS"

echo "=========================================="
echo "  FlexLLaVA Elastic Eval"
echo "  tok_level=${TOK_LEVEL}  n_tokens=${N_TOK}"
echo "  model: ${MODEL_PATH}"
echo "  answers: ${ANSWERS}"
echo "=========================================="

# ── helper: run elastic vqa loader ──────────────────────────────────────────
run_elastic() {
    local qfile=$1 imgdir=$2 outfile=$3
    python -m llava.eval.model_vqa_elastic \
        --model-path  "$MODEL_PATH" \
        --question-file "$qfile" \
        --image-folder  "$imgdir" \
        --answers-file  "$outfile" \
        --tok-level     "$TOK_LEVEL" \
        --conv-mode     vicuna_v1
}

# ── 1. MME ────────────────────────────────────────────────────────────────────
echo -e "\n[1/6] MME"
MME_DIR="${EVAL_DATA}/eval/MME"
run_elastic \
    "${MME_DIR}/llava_mme.jsonl" \
    "${MME_DIR}/MME_Benchmark_release_version" \
    "${ANSWERS}/mme.jsonl"

cd "$MME_DIR"
python convert_answer_to_mme.py --experiment "${CKPT}"
cd eval_tool
python calculation.py --results_dir "answers/${CKPT}" | tee "${OLDPWD}/${ANSWERS}/mme_result.txt"
cd "$OLDPWD"

# ── 2. POPE ───────────────────────────────────────────────────────────────────
echo -e "\n[2/6] POPE"
POPE_DIR="${EVAL_DATA}/eval/pope"
run_elastic \
    "${POPE_DIR}/llava_pope_test.jsonl" \
    "${POPE_DIR}/val2014" \
    "${ANSWERS}/pope.jsonl"

python llava/eval/eval_pope.py \
    --annotation-dir "${POPE_DIR}/coco" \
    --question-file  "${POPE_DIR}/llava_pope_test.jsonl" \
    --result-file    "${ANSWERS}/pope.jsonl" \
    | tee "${ANSWERS}/pope_result.txt"

# ── 3. MMBench-en ─────────────────────────────────────────────────────────────
echo -e "\n[3/6] MMBench-en"
MMB_DIR="${EVAL_DATA}/eval/mmbench"
MMB_SPLIT="mmbench_dev_20230712"

python -m llava.eval.model_vqa_mmbench \
    --model-path    "$MODEL_PATH" \
    --question-file "${MMB_DIR}/${MMB_SPLIT}.tsv" \
    --answers-file  "${ANSWERS}/mmbench.jsonl" \
    --tok-level     "$TOK_LEVEL" \
    --single-pred-prompt \
    --temperature   0 \
    --conv-mode     vicuna_v1

mkdir -p "${MMB_DIR}/answers_upload/${MMB_SPLIT}"
python scripts/convert_mmbench_for_submission.py \
    --annotation-file "${MMB_DIR}/${MMB_SPLIT}.tsv" \
    --result-dir      "${ANSWERS}" \
    --upload-dir      "${MMB_DIR}/answers_upload/${MMB_SPLIT}" \
    --experiment      "${CKPT}" \
    | tee "${ANSWERS}/mmbench_result.txt"

# ── 4. ScienceQA ──────────────────────────────────────────────────────────────
echo -e "\n[4/6] ScienceQA"
SQA_DIR="${EVAL_DATA}/eval/scienceqa"

python -m llava.eval.model_vqa_science \
    --model-path    "$MODEL_PATH" \
    --question-file "${SQA_DIR}/llava_test_CQM-A.json" \
    --image-folder  "${SQA_DIR}/images/test" \
    --answers-file  "${ANSWERS}/scienceqa.jsonl" \
    --tok-level     "$TOK_LEVEL" \
    --single-pred-prompt \
    --temperature   0 \
    --conv-mode     vicuna_v1


python llava/eval/eval_science_qa.py \
    --base-dir      "$SQA_DIR" \
    --result-file   "${ANSWERS}/scienceqa.jsonl" \
    --output-file   "${ANSWERS}/scienceqa_output.jsonl" \
    --output-result "${ANSWERS}/scienceqa_result.json" \
    && python3 -c "import json; r=json.load(open('${ANSWERS}/scienceqa_result.json')); print('ScienceQA accuracy:', r)" \
    | tee "${ANSWERS}/scienceqa_result.txt"

# ── 5. TextVQA ────────────────────────────────────────────────────────────────
echo -e "\n[5/6] TextVQA"
TVQ_DIR="${EVAL_DATA}/eval/textvqa"
run_elastic \
    "${TVQ_DIR}/llava_textvqa_val_v051_ocr.jsonl" \
    "${TVQ_DIR}/train_images" \
    "${ANSWERS}/textvqa.jsonl"

python -m llava.eval.eval_textvqa \
    --annotation-file "${TVQ_DIR}/TextVQA_0.5.1_val.json" \
    --result-file     "${ANSWERS}/textvqa.jsonl" \
    | tee "${ANSWERS}/textvqa_result.txt"

# ── 6. GQA ────────────────────────────────────────────────────────────────────
echo -e "\n[6/6] GQA"
GQA_DIR="${EVAL_DATA}/eval/gqa"
CHUNKS=1

run_elastic \
    "${GQA_DIR}/llava_gqa_testdev_balanced.jsonl" \
    "${GQA_DIR}/images" \
    "${ANSWERS}/gqa_0.jsonl"

python scripts/convert_gqa_for_submission.py \
    --src "${ANSWERS}/gqa_0.jsonl" \
    --dst "${ANSWERS}/gqa_submit.json"

python -m llava.eval.eval_gqa \
    --predictions "${ANSWERS}/gqa_submit.json" \
    --questions   "${GQA_DIR}/testdev_balanced_questions.json" \
    | tee "${ANSWERS}/gqa_result.txt"

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "=========================================="
echo "  RESULTS  tok_level=${TOK_LEVEL}  n_tokens=${N_TOK}"
echo "=========================================="
for f in mme_result pope_result mmbench_result scienceqa_result textvqa_result gqa_result; do
    echo "--- ${f} ---"
    cat "${ANSWERS}/${f}.txt" 2>/dev/null || echo "(not found)"
done
