#!/bin/bash
#SBATCH --job-name=finetune_only_hipster
#SBATCH -t 150:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
# Overridden at submit time by submit_finetune_only.sh.
#SBATCH --partition=capacity
#SBATCH --gres=gpu:l4:4
#SBATCH --cpus-per-task=16
#SBATCH --output=./jobs/run_hipster_%A.out
#SBATCH --export=ALL
# W&B credentials are NOT in this file. They live in ~/.netrc (mode 600,
# machine api.wandb.ai), which wandb reads natively on the compute node --
# $HOME is shared with the login node, so nothing needs exporting. Set it up
# once with `wandb login`, or write the three-line netrc stanza by hand.
# Previously this line carried the key inline, which put it in every commit
# and on GitHub; see docs/EXPERIMENT_JOURNAL.md section 20.
#
# Stage 2 ONLY -- no pretrain_elastic_slm_hipster.sh call. Two valid uses:
#   (a) RESUME an existing Stage-2 run: train.py's own
#       `if glob("checkpoint-*"): resume_from_checkpoint=True` (llava/train/
#       train.py) reloads the full model/optimizer/scheduler/rng state, which
#       supersedes any --pretrain_elastic_path warm-start -- Stage 1's own
#       checkpoint is irrelevant here and does not need to exist.
#   (b) FRESH Stage 2, warm-started from an already-complete Stage-1
#       checkpoint via --pretrain_elastic_path (finetune_elastic_slm_hipster.
#       sh sets this from CHECKPOINT_ROOT/elastic-pretrain-<key>-<tag>
#       automatically) -- skips redoing Stage 1 when it already finished
#       elsewhere (e.g. a checkpoint transferred in from another cluster).
# Data source: hipster's own /home/skalra/llava_data archives (same fast
# path as run_job_hipster.sh) -- LLaVA-Finetune only, since Stage 2 never
# reads LLaVA-Pretrain.
#
# NO --exclusive, same shared-cluster etiquette as run_job_hipster.sh.

set -e

echo "Job Started"; date
echo "Node name: $(hostname)"
nvidia-smi

module load cuda/12.9.1 2>&1 || echo "WARNING: module load cuda/12.9.1 failed -- continuing"

eval "$(conda shell.bash hook)"
conda activate matryoshka-mm

export NCCL_SOCKET_TIMEOUT=3600
export NCCL_DEBUG=WARN
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_ASYNC_ERROR_HANDLING=1

export WANDB_PROJECT="FlexLLaVA"
export WANDB_DIR=/home/skalra/flexllava_saves/wandb
export HF_HOME=/home/skalra/flexllava_saves/cache/huggingface

SLM_KEY=${1:-tinyllama}
echo "[FlexLLaVA] SLM_KEY=${SLM_KEY}  ELASTIC_RUN_TAG=${ELASTIC_RUN_TAG:-<none>}  (finetune-only, hipster data source)"

STAGING=/home/skalra/llava_data
LOCAL_SSD=/local_scratch/skalra/flexllava_data_${SLURM_JOB_ID}
mkdir -p "$LOCAL_SSD/LLaVA-Finetune"

REQUIRED_SPACE_GB=75
AVAILABLE_SPACE_GB=$(df --output=avail -BG "$LOCAL_SSD" | tail -1 | tr -dc '0-9')
echo "Local SSD available: ${AVAILABLE_SPACE_GB}G  (need ~${REQUIRED_SPACE_GB}G for LLaVA-Finetune)"
if [ "$AVAILABLE_SPACE_GB" -lt "$REQUIRED_SPACE_GB" ]; then
    echo "ERROR: not enough local disk on $(hostname):$LOCAL_SSD" >&2
    exit 1
fi

trap "echo 'Cleaning up local SSD...'; rm -rf $LOCAL_SSD" EXIT

echo "Staging LLaVA-Finetune only: $STAGING -> $LOCAL_SSD ..."; date
N_PARALLEL="${STAGE_PARALLEL:-16}"
ARCHIVE_DIR="$STAGING/archives/LLaVA-Finetune"

if [ -d "$ARCHIVE_DIR" ] && [ -n "$(command find "$ARCHIVE_DIR" -name '*.tar' -print -quit 2>/dev/null)" ]; then
    echo "Using pre-built archives in $ARCHIVE_DIR"
    find "$ARCHIVE_DIR" -name '*.tar' -print0 \
        | xargs -0 -P "$N_PARALLEL" -I{} tar -xf {} -C "$LOCAL_SSD"
    # Leftover raw files (annotation JSON, anything archival hasn't reached).
    # `cd` first, relative dir name -- see run_job_hipster.sh's 2026-09-10
    # note on why an absolute path here silently lands files in the wrong
    # place under `cp --parents`.
    ( cd "$STAGING" && command find LLaVA-Finetune -type f -print0 2>/dev/null \
        | xargs -0 -r -P "$N_PARALLEL" -n 500 cp --parents -t "$LOCAL_SSD" 2>/dev/null || true )
else
    ( cd "$STAGING" && command find LLaVA-Finetune -type f -print0 \
        | xargs -0 -P "$N_PARALLEL" -n 500 cp --parents -t "$LOCAL_SSD" )
fi

du -sh "$LOCAL_SSD"/LLaVA-Finetune/*
echo "Staging done."; date
export LOCAL_SSD

./scripts/v1_5/finetune_elastic_slm_hipster.sh "$SLM_KEY"

echo "Job Complete"; date
