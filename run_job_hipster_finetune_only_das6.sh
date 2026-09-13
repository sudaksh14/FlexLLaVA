#!/bin/bash
#SBATCH --job-name=finetune_only_das6data
#SBATCH -t 150:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
# Overridden at submit time (see below) -- partition/gres depend on which
# recipe/backbone is being resumed.
#SBATCH --partition=capacity
#SBATCH --gres=gpu:l4:4
#SBATCH --cpus-per-task=16
#SBATCH --output=./jobs/run_hipster_das6data_%A.out
#SBATCH --export=ALL,WANDB_API_KEY=dfcd2574507b9ebe69ca13ab6f6925d864e82ee0
#
# Stage 2 ONLY -- no pretrain_elastic_slm_hipster.sh call. For resuming a
# Stage 2 run from an existing checkpoint-N in its output_dir: train.py's
# own `if glob("checkpoint-*"): resume_from_checkpoint=True` (llava/train/
# train.py) reloads the FULL model/optimizer/scheduler/rng state from that
# checkpoint, which supersedes whatever --pretrain_elastic_path would have
# warm-started -- so Stage 1's own checkpoint is irrelevant to a Stage-2
# resume and does not need to exist. finetune_elastic_slm_hipster.sh treats
# a missing/empty --pretrain_elastic_path as a graceful no-op (confirmed in
# llava/train/train.py's _load_elastic_pretrain_weights), not an error.
#
# Data source: DAS-6 directly, not hipster's own /home/skalra/llava_data
# archives -- tar streamed over an existing SSH connection straight into
# this node's /local_scratch, never touching disk on either end as an
# intermediate archive. Only LLaVA-Finetune is pulled (Stage 2 doesn't need
# LLaVA-Pretrain at all).
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
echo "[FlexLLaVA] SLM_KEY=${SLM_KEY}  ELASTIC_RUN_TAG=${ELASTIC_RUN_TAG:-<none>}  (finetune-only, DAS-6 data source)"

DAS6_HOST=fs2.das6.science.uva.nl
DAS6_DATA=/var/scratch/skalra/flexllava/data/LLaVA-Finetune
LOCAL_SSD=/local_scratch/skalra/flexllava_data_${SLURM_JOB_ID}
mkdir -p "$LOCAL_SSD/LLaVA-Finetune"

echo "Checking SSH reachability from this compute node to $DAS6_HOST ..."
if ! timeout 15 ssh -o BatchMode=yes -o ConnectTimeout=10 "$DAS6_HOST" "echo ok" 2>&1; then
    echo "ERROR: cannot reach $DAS6_HOST via SSH from $(hostname)." \
         "Compute nodes may have restricted outbound network access even" \
         "when the login node can reach it -- this was NOT verified before" \
         "submitting this job. Falling back to hipster's own llava_data" \
         "archives requires a different launcher (run_job_hipster.sh)." >&2
    exit 1
fi
echo "SSH OK."

REQUIRED_SPACE_GB=75
AVAILABLE_SPACE_GB=$(df --output=avail -BG "$LOCAL_SSD" | tail -1 | tr -dc '0-9')
echo "Local SSD available: ${AVAILABLE_SPACE_GB}G  (need ~${REQUIRED_SPACE_GB}G for LLaVA-Finetune)"
if [ "$AVAILABLE_SPACE_GB" -lt "$REQUIRED_SPACE_GB" ]; then
    echo "ERROR: not enough local disk on $(hostname):$LOCAL_SSD" >&2
    exit 1
fi

trap "echo 'Cleaning up local SSD...'; rm -rf $LOCAL_SSD" EXIT

echo "Streaming LLaVA-Finetune from $DAS6_HOST:$DAS6_DATA -> $LOCAL_SSD/LLaVA-Finetune ..."; date
# One tar-over-ssh pipe per top-level dataset dir, run in parallel -- same
# principle as the hipster-local archival/staging work: a single sequential
# stream per dataset avoids the many-small-files SSH/protocol overhead that
# would come from transferring 608k files individually (scp/rsync per-file).
# No compression (tar cf, not czf): payload is JPEGs, already compressed.
PIDS=()
for d in coco gqa ocr_vqa textvqa vg; do
    ssh -o BatchMode=yes "$DAS6_HOST" "tar -cf - -C $DAS6_DATA $d" \
        | tar -xf - -C "$LOCAL_SSD/LLaVA-Finetune" &
    PIDS+=($!)
done
# Annotation JSON, separately (small, not part of the many-files problem).
scp -o BatchMode=yes -q "$DAS6_HOST:$DAS6_DATA/llava_v1_5_mix665k.json" \
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
du -sh "$LOCAL_SSD"/LLaVA-Finetune/*
echo "File count: $(command find "$LOCAL_SSD/LLaVA-Finetune" -type f | wc -l)  (DAS-6 source has 608076 images + 1 json = 608077)"
export LOCAL_SSD

./scripts/v1_5/finetune_elastic_slm_hipster.sh "$SLM_KEY"

echo "Job Complete"; date
