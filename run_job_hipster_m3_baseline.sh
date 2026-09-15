#!/bin/bash
# M3 baseline (avg-pool matryoshka) at a fixed token budget -- HIPSTER.
#
# Derived from run_job_hipster.sh: same staging, same CUDA pin, same env. Only
# the two stage scripts it calls differ.
#
#   MATRYOSHKA_SCALE=4 BASELINE_RUN_TAG=4tok sbatch run_job_hipster_m3_baseline.sh tinyllama
#
# This repo is an M3 fork with M3 intact underneath, so Stage 2 with a scale
# list IS M3's own training loop -- see finetune_baseline_slm_hipster.sh.
#SBATCH --job-name=m3_baseline_hipster
#SBATCH -t 150:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
# Defaults below are placeholders -- submit_elastic_run_hipster.sh always
# overrides --partition/--gres/--ntasks-per-node/--cpus-per-task at submit
# time for the chosen partition (performance=rtx_6000_ada, capacity=l4) and
# NUM_GPUS (4 for a real run). Do not rely on these.
#SBATCH --partition=capacity
#SBATCH --gres=gpu:l4:4
#SBATCH --cpus-per-task=16
#SBATCH --output=./jobs/run_hipster_m3_%A.out
#SBATCH --export=ALL,WANDB_API_KEY=dfcd2574507b9ebe69ca13ab6f6925d864e82ee0
#
# NO --exclusive: hipster is a large shared multi-tenant cluster (dozens of
# other users' jobs queued at submit time), unlike DAS-6's small dedicated
# pool where run_job_slm.sh's --exclusive claims a whole node deliberately.
# Here we request exactly the GPUs/CPUs needed and let SLURM co-schedule
# other users' jobs on the rest of the node.

set -e

echo "Job Started"; date
echo "Node name: $(hostname)"
echo -n memory=; ulimit -m
echo -n nproc=; nproc
nvidia-smi

# ---- CUDA module -----------------------------------------------------------
# UNVERIFIED PIN, not a like-for-like match: DAS-6 loads cuda12.1/toolkit/12.1
# (finetune) or cuda12.6/toolkit/12.6 (eval) -- neither module name exists on
# hipster (`module avail cuda` here only offers 11.8.0 / 12.9.1 / 12.9.2 /
# 13.3.0, a completely different naming scheme). 12.9.1 is the closest
# same-major-version option below the newest release. The training path does
# NOT depend on this for the optimizer (zero2.json has no custom "optimizer"
# block, so DeepSpeed does not JIT-compile FusedAdam/CPUAdam), and flash-attn
# 2.5.8 is already installed as a prebuilt wheel (not compiled here) -- so the
# blast radius of a mismatch is smaller than it looks, but it has NOT been
# proven correct end-to-end on this cluster's actual GPUs before this run.
module load cuda/12.9.1 2>&1 || echo "WARNING: module load cuda/12.9.1 failed -- continuing, torch's own bundled cu121 runtime may still work"

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
echo "[FlexLLaVA] SLM_KEY=${SLM_KEY}  BASELINE_RUN_TAG=${BASELINE_RUN_TAG:-<none>}"
echo "[m3-baseline] MATRYOSHKA_SCALE=${MATRYOSHKA_SCALE:-<unset -> 576 tokens>}"

# ---- Stage data onto this node's local SSD ---------------------------------
# Namespaced by $SLURM_JOB_ID: two of our OWN jobs (e.g. tinyllama + smollm2
# recipes submitted close together) can land on the same physical node since
# nothing here is --exclusive, and a shared fixed path would let one job's
# `trap ... rm -rf` cleanup delete the other's in-flight data mid-run.
STAGING=/home/skalra/llava_data
LOCAL_SSD=/local_scratch/skalra/flexllava_data_${SLURM_JOB_ID}
mkdir -p "$LOCAL_SSD"

REQUIRED_SPACE_GB=110
AVAILABLE_SPACE_GB=$(df --output=avail -BG "$LOCAL_SSD" | tail -1 | tr -dc '0-9')
echo "Local SSD available: ${AVAILABLE_SPACE_GB}G  (need ~${REQUIRED_SPACE_GB}G)"
if [ "$AVAILABLE_SPACE_GB" -lt "$REQUIRED_SPACE_GB" ]; then
    echo "ERROR: not enough local disk on $(hostname):$LOCAL_SSD" >&2
    exit 1
fi

trap "echo 'Cleaning up local SSD...'; rm -rf $LOCAL_SSD" EXIT

echo "Staging data $STAGING -> $LOCAL_SSD ..."; date
N_PARALLEL="${STAGE_PARALLEL:-16}"
ARCHIVE_DIR="$STAGING/archives"

if [ -d "$ARCHIVE_DIR" ] && [ -n "$(command find "$ARCHIVE_DIR" -name '*.tar' -print -quit 2>/dev/null)" ]; then
    # Fast path: one-time tar archival (jobs/archive_llava_data.sh) has run,
    # so $STAGING/{LLaVA-Pretrain,LLaVA-Finetune} raw trees may be partially
    # or fully replaced by $ARCHIVE_DIR/*.tar. Each tar is a single large
    # sequential NFS read (fast) + local extraction (fast, no network) --
    # no per-file NFS round trips at all, versus the ~1180 files/s serial
    # rate measured on this cluster (2026-09-09) for the raw many-small-
    # files tree. Extraction is parallelized across tars, same as the
    # fallback path below is parallelized across files.
    echo "Using pre-built archives in $ARCHIVE_DIR"
    mkdir -p "$LOCAL_SSD/LLaVA-Pretrain" "$LOCAL_SSD/LLaVA-Finetune"
    find "$ARCHIVE_DIR" -name '*.tar' -print0 \
        | xargs -0 -P "$N_PARALLEL" -I{} tar -xf {} -C "$LOCAL_SSD"
    # Any files archival hasn't reached yet (still raw) -- copy those too, so
    # this script is correct regardless of how far archival has progressed
    # at the moment a training job happens to start. This ALSO covers the
    # two annotation JSONs (blip_laion_cc_sbu_558k.json,
    # llava_v1_5_mix665k.json), which archival deliberately never tars.
    # `cd` first and use RELATIVE dir names: `cp --parents` reproduces
    # whatever path form it's given, so an absolute `$STAGING/$d` path here
    # would land files at "$LOCAL_SSD/home/skalra/llava_data/..." instead of
    # "$LOCAL_SSD/LLaVA-Pretrain/..." where the training scripts expect them
    # -- caught 2026-09-10 in a bounded local test before it could ship.
    ( cd "$STAGING" && command find LLaVA-Pretrain LLaVA-Finetune -type f -print0 2>/dev/null \
        | xargs -0 -r -P "$N_PARALLEL" -n 500 cp --parents -t "$LOCAL_SSD" 2>/dev/null || true )
else
    # Fallback: raw tree, no archives yet. Parallel at FILE granularity, not
    # directory granularity -- LLaVA-Pretrain has 665 shard dirs (fine either
    # way) but LLaVA-Finetune is only 5-6 leaf dirs each holding 100k+ FLAT
    # files (coco/train2017, gqa/images, ...), so fanning out over top-level
    # dirs there caps parallelism at ~6-way regardless of -P. `cp --parents`
    # preserves each file's path relative to $STAGING under $LOCAL_SSD;
    # -n 500 batches files per cp invocation so process-spawn overhead
    # doesn't dominate at file-level granularity. Measured 2026-09-09: a
    # serial cp -r ran at ~68MB/s / ~1180 files/s; this file-level -P 16
    # approach ran a small flat-directory sample ~3-4x faster.
    cd "$STAGING"
    command find LLaVA-Pretrain LLaVA-Finetune -type f -print0 \
        | xargs -0 -P "$N_PARALLEL" -n 500 cp --parents -t "$LOCAL_SSD"
    cd /home/skalra/FlexLLaVA
fi

du -sh "$LOCAL_SSD"/*
echo "Staging done."; date
export LOCAL_SSD

# --- Stage 1 (plain projector, 576 tok) then Stage 2 (M3 @ MATRYOSHKA_SCALE) ---
# MATRYOSHKA_SCALE is exported by the submit driver; Stage 1 deliberately does
# not take one (M3 pools AFTER the projector, so its Stage 1 is ordinary LLaVA).
./scripts/v1_5/pretrain_baseline_slm_hipster.sh "$SLM_KEY"
./scripts/v1_5/finetune_baseline_slm_hipster.sh "$SLM_KEY"

echo "Job Complete"; date
