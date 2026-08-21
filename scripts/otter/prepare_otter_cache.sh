#!/bin/bash
#SBATCH --job-name=otter_prep
#SBATCH -t 04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --output=./jobs/otterprep_%A.out
#SBATCH --export=ALL

# Build the offline caches the Otter pipeline uses (CPU only, no GPU):
#   * missing-image manifest -- stat every referenced image once, so unreadable
#     samples are DROPPED deterministically instead of randomly substituted at
#     train time (which silently reweights the mixture);
#   * token-length cache -- true tokenizer lengths for the length-grouped
#     sampler, replacing the word-count + hard-coded-128 estimate.
#
#   sbatch scripts/otter/prepare_otter_cache.sh <LLM_KEY> [MIXTURE_CONFIG]
#
# Both caches are keyed on the mixture signature, the tokenizer, visual_tokens
# AND model_max_length, so rerun this whenever any of those changes. Training
# works without them -- it just pads more and falls back at runtime -- so this
# is an optimisation job, not a prerequisite.
#
# IMPORTANT: model_max_length must match the STAGE you are building for, or the
# key will not match and training silently falls back to the word-count
# heuristic ("lengths from heuristic" in the log):
#     Stage 1 (pretrain_otter_slm.sh) uses 1024  -> OTTER_MAX_LEN=1024
#     Stage 2 (finetune_otter_slm.sh) uses 2048  -> OTTER_MAX_LEN=2048 (default)

set -euo pipefail

eval "$(conda shell.bash hook)"
conda activate matryoshka-mm

export HF_HOME=/var/scratch/skalra/.cache/huggingface

LLM_KEY=${1:-tinyllama}
MIXTURE_CONFIG=${2:-configs/otter/mix665k_baseline.yaml}

case "$LLM_KEY" in
  qwen0.5b)    MODEL_PATH="Qwen/Qwen2.5-0.5B-Instruct" ;;
  qwen1.5b)    MODEL_PATH="Qwen/Qwen2.5-1.5B-Instruct" ;;
  qwen3b)      MODEL_PATH="Qwen/Qwen2.5-3B-Instruct" ;;
  tinyllama)   MODEL_PATH="TinyLlama/TinyLlama-1.1B-Chat-v1.0" ;;
  mobilellama) MODEL_PATH="mtgv/MobileLLaMA-1.4B-Chat" ;;
  smollm2)     MODEL_PATH="HuggingFaceTB/SmolLM2-1.7B-Instruct" ;;
  phi2)        MODEL_PATH="microsoft/phi-2" ;;
  phi3.5|phi3) MODEL_PATH="microsoft/Phi-3.5-mini-instruct" ;;
  stablelm)    MODEL_PATH="stabilityai/stablelm-2-zephyr-1_6b" ;;
  *) echo "Unknown LLM_KEY='$LLM_KEY'"; exit 1 ;;
esac

cd /home/skalra/FlexLLaVA

echo "[otter-prep] mixture=${MIXTURE_CONFIG} model=${MODEL_PATH}"
python -m llava.data_otter.prepare \
    --mixture_config "${MIXTURE_CONFIG}" \
    --cache_dir "${OTTER_CACHE_DIR:-/var/scratch/skalra/flexllava/cache/otter}" \
    --model_name_or_path "${MODEL_PATH}" \
    --hf_cache_dir /var/scratch/skalra/.cache/huggingface/hub \
    --model_max_length "${OTTER_MAX_LEN:-2048}" \
    --visual_tokens "${OTTER_VISUAL_TOKENS:-256}" \
    --build lengths,manifest

echo "[otter-prep] done"
