#!/bin/bash
#SBATCH --job-name=otter_FlexLLaVA_SLM
# Same 200h as run_job_finetune_slm.sh: one Stage-2 epoch is ~5200 steps and
# run_26226 hit the old 150h limit at 92%.
#SBATCH -t 200:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:A40:2
#SBATCH --cpus-per-task=32
#SBATCH --output=./jobs/otter_%A.out
#SBATCH --export=ALL,WANDB_API_KEY=dfcd2574507b9ebe69ca13ab6f6925d864e82ee0
# Training takes every GPU on the node, so claim all 64 cores too -- without
# --exclusive, cons_tres confines the job to --cpus-per-task and the rest idle.
# NOTE: do NOT raise --dataloader_num_workers without doing the arithmetic.
# Measured on node205 mid-run (2026-08-19, /proc/PID/smaps_rollup): each
# dataloader worker costs ~4.0GB of PSS, and it is almost all PRIVATE -- the
# fork copy-on-write does not hold, because Python refcounting touches the
# 665k-record list and privatises those pages. At the proven 4 workers/rank the
# job sits at 43GB PSS on a 125GB node with 32GB free. 12 workers/rank would be
# ~107GB and would thrash or OOM, hours into a 200h run.
#SBATCH --exclusive

# Stage 2 on the Otter-style data pipeline. The elastic recipe
# (run_job_finetune_slm.sh) is untouched and unaffected.
#
#   ELASTIC_RUN_TAG=otter1 sbatch run_job_otter_slm.sh tinyllama
#   ELASTIC_RUN_TAG=otter2 OTTER_MIXTURE_CONFIG=configs/otter/mix665k_ocr_heavy.yaml \
#       sbatch run_job_otter_slm.sh tinyllama
#
# ELASTIC_RUN_TAG picks the output dir (otter-finetune-<key>-<tag>) and, unless
# ELASTIC_PRETRAIN_TAG is set, the Stage-1 checkpoint to warm-start from.
# Reusing a tag OVERWRITES that finetune checkpoint.
#
# Build the offline caches first (once per mixture+backbone):
#   sbatch scripts/otter/prepare_otter_cache.sh tinyllama configs/otter/mix665k_baseline.yaml

module load cuda12.1/toolkit/12.1

eval "$(conda shell.bash hook)"
conda activate matryoshka-mm

nvidia-smi

echo "Job Started"
echo | date
echo "Node name: $(hostname)"
echo -n memory=; ulimit -m
echo -n nproc=; nproc

export NCCL_SOCKET_TIMEOUT=3600
export NCCL_DEBUG=WARN
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# See run_job_slm.sh: overlap_comm:false + this flag are what stopped the
# rank-desync deadlock at step 136 in run_26187/26205/26209.
export NCCL_ASYNC_ERROR_HANDLING=1

export WANDB_PROJECT="FlexLLaVA"
export WANDB_DIR=/var/scratch/skalra/flexllava/wandb
export HF_HOME=/var/scratch/skalra/.cache/huggingface

SLM_KEY=${1:-tinyllama}
MIXTURE_CONFIG=${2:-${OTTER_MIXTURE_CONFIG:-configs/otter/mix665k_baseline.yaml}}
echo "[FlexLLaVA/otter] SLM_KEY=${SLM_KEY}  MIXTURE=${MIXTURE_CONFIG}  ELASTIC_RUN_TAG=${ELASTIC_RUN_TAG:-<none>}"

./scripts/v1_5/finetune_otter_slm.sh "$SLM_KEY" "$MIXTURE_CONFIG"

echo "Job Complete"
echo | date
