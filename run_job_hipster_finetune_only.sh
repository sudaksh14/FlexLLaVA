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
# Data source: DAS-6 over SSH (hipster's own /home/skalra/llava_data archive
# was deleted 2026-09-19 to reclaim disk) -- LLaVA-Finetune only, since
# Stage 2 never reads LLaVA-Pretrain. Same streaming mechanism as
# run_job_hipster_finetune_only_das6.sh.
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

DAS6_HOST=fs2.das6.science.uva.nl
DAS6_DATA=/var/scratch/skalra/flexllava/data/LLaVA-Finetune
LOCAL_SSD=/local_scratch/skalra/flexllava_data_${SLURM_JOB_ID}
mkdir -p "$LOCAL_SSD/LLaVA-Finetune"

echo "Checking SSH reachability from this compute node to $DAS6_HOST ..."
if ! timeout 15 ssh -o BatchMode=yes -o ConnectTimeout=10 "$DAS6_HOST" "echo ok" 2>&1; then
    echo "ERROR: cannot reach $DAS6_HOST via SSH from $(hostname)." >&2
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
# hipster's own /home/skalra/llava_data archive was deleted 2026-09-19 to
# reclaim disk -- DAS-6 is now the only data source, same mechanism
# run_job_hipster_finetune_only_das6.sh already used. One tar-over-ssh pipe
# per top-level dir, run in parallel; no compression (payload is JPEGs,
# already compressed).
PIDS=()
for d in coco gqa ocr_vqa textvqa vg; do
    ssh -o BatchMode=yes "$DAS6_HOST" "tar -cf - -C $DAS6_DATA $d" \
        | tar -xf - -C "$LOCAL_SSD/LLaVA-Finetune" &
    PIDS+=($!)
done
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
export LOCAL_SSD
./scripts/v1_5/finetune_elastic_slm_hipster.sh "$SLM_KEY"

echo "Job Complete"; date
