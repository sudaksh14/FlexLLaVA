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
#SBATCH --export=ALL
# W&B credentials are NOT in this file. They live in ~/.netrc (mode 600,
# machine api.wandb.ai), which wandb reads natively on the compute node --
# $HOME is shared with the login node, so nothing needs exporting. Set it up
# once with `wandb login`, or write the three-line netrc stanza by hand.
# Previously this line carried the key inline, which put it in every commit
# and on GitHub; see docs/EXPERIMENT_JOURNAL.md section 20.
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
DAS6_HOST=fs2.das6.science.uva.nl
DAS6_DATA=/var/scratch/skalra/flexllava/data
LOCAL_SSD=/local_scratch/skalra/flexllava_data_${SLURM_JOB_ID}
mkdir -p "$LOCAL_SSD/LLaVA-Pretrain" "$LOCAL_SSD/LLaVA-Finetune"

echo "Checking SSH reachability from this compute node to $DAS6_HOST ..."
if ! timeout 15 ssh -o BatchMode=yes -o ConnectTimeout=10 "$DAS6_HOST" "echo ok" 2>&1; then
    echo "ERROR: cannot reach $DAS6_HOST via SSH from $(hostname)." \
         "Compute nodes may have restricted outbound network access even" \
         "when the login node can reach it." >&2
    exit 1
fi
echo "SSH OK."

REQUIRED_SPACE_GB=110
AVAILABLE_SPACE_GB=$(df --output=avail -BG "$LOCAL_SSD" | tail -1 | tr -dc '0-9')
echo "Local SSD available: ${AVAILABLE_SPACE_GB}G  (need ~${REQUIRED_SPACE_GB}G)"
if [ "$AVAILABLE_SPACE_GB" -lt "$REQUIRED_SPACE_GB" ]; then
    echo "ERROR: not enough local disk on $(hostname):$LOCAL_SSD" >&2
    exit 1
fi

trap "echo 'Cleaning up local SSD...'; rm -rf $LOCAL_SSD" EXIT

echo "Streaming LLaVA-Pretrain + LLaVA-Finetune from $DAS6_HOST -> $LOCAL_SSD ..."; date
# hipster's own /home/skalra/llava_data archive was deleted 2026-09-19 to
# reclaim disk (home filesystem was at 96% capacity, 191G/200G used) -- DAS-6
# is now the only data source for every hipster job, streamed straight to
# /local_scratch per job. Same mechanism run_job_pretrain_only_hipster.sh /
# run_job_hipster_finetune_only_das6.sh already used for the elastic/matrix
# jobs; this just combines both trees since this launcher runs Stage 1 and
# Stage 2 back to back in one job. Namespaced by $SLURM_JOB_ID for the same
# reason as the old local-archive path: two of our own jobs can land on the
# same node since nothing here is --exclusive.
N_PARALLEL="${STAGE_PARALLEL:-8}"

echo "Listing LLaVA-Pretrain shards on $DAS6_HOST ..."
mapfile -t SHARDS < <(timeout 30 ssh -o BatchMode=yes "$DAS6_HOST" \
    "find '$DAS6_DATA/LLaVA-Pretrain' -mindepth 1 -maxdepth 1 -type d -printf '%f\n'" | sort)
echo "${#SHARDS[@]} LLaVA-Pretrain shard dirs found."

PIDS=()
for i in $(seq 0 $(( N_PARALLEL - 1 ))); do
    CHUNK=()
    for ((j=i; j<${#SHARDS[@]}; j+=N_PARALLEL)); do CHUNK+=("${SHARDS[$j]}"); done
    [ ${#CHUNK[@]} -eq 0 ] && continue
    ssh -o BatchMode=yes "$DAS6_HOST" "tar -cf - -C $DAS6_DATA/LLaVA-Pretrain ${CHUNK[*]}" \
        | tar -xf - -C "$LOCAL_SSD/LLaVA-Pretrain" &
    PIDS+=($!)
done
scp -o BatchMode=yes -q "$DAS6_HOST:$DAS6_DATA/LLaVA-Pretrain/blip_laion_cc_sbu_558k.json" \
    "$LOCAL_SSD/LLaVA-Pretrain/blip_laion_cc_sbu_558k.json" &
PIDS+=($!)

# One tar-over-ssh pipe per top-level LLaVA-Finetune dir, run in parallel --
# same principle as the pretrain shard chunking above: a single sequential
# stream per dataset avoids per-file SSH/protocol overhead. No compression
# (tar cf, not czf): payload is JPEGs, already compressed.
for d in coco gqa ocr_vqa textvqa vg; do
    ssh -o BatchMode=yes "$DAS6_HOST" "tar -cf - -C $DAS6_DATA/LLaVA-Finetune $d" \
        | tar -xf - -C "$LOCAL_SSD/LLaVA-Finetune" &
    PIDS+=($!)
done
scp -o BatchMode=yes -q "$DAS6_HOST:$DAS6_DATA/LLaVA-Finetune/llava_v1_5_mix665k.json" \
    "$LOCAL_SSD/LLaVA-Finetune/llava_v1_5_mix665k.json" &
PIDS+=($!)

FAIL=0
for pid in "${PIDS[@]}"; do
    wait "$pid" || FAIL=1
done
if [ "$FAIL" -ne 0 ]; then
    echo "ERROR: one or more DAS-6 transfer streams failed -- see output above." >&2
    exit 1
fi

echo "Transfer done."; date
du -sh "$LOCAL_SSD"/*
export LOCAL_SSD

# --- Stage 1 (plain projector, 576 tok) then Stage 2 (M3 @ MATRYOSHKA_SCALE) ---
# MATRYOSHKA_SCALE is exported by the submit driver; Stage 1 deliberately does
# not take one (M3 pools AFTER the projector, so its Stage 1 is ordinary LLaVA).
./scripts/v1_5/pretrain_baseline_slm_hipster.sh "$SLM_KEY"
./scripts/v1_5/finetune_baseline_slm_hipster.sh "$SLM_KEY"

echo "Job Complete"; date
