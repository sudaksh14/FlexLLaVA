#!/bin/bash
#SBATCH --job-name=m3_ladder_finetune_hipster
#SBATCH -t 150:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
# Defaults are placeholders -- submit_m3_ladder_hipster.sh overrides
# --partition/--gres/--ntasks-per-node/--cpus-per-task at submit time.
#SBATCH --partition=performance
#SBATCH --gres=gpu:rtx_6000_ada:4
#SBATCH --cpus-per-task=32
#SBATCH --output=./jobs/run_hipster_m3_ladder_%A.out
#SBATCH --export=ALL
# W&B credentials live in ~/.netrc (see docs/EXPERIMENT_JOURNAL.md section 20).
#
# Stage 2 ONLY of the budget-matched M3 baseline (scale list 256,144,64,16, M3's
# own native multi-scale loop), warm-started from the 4tok Stage-1 projector.
# Stage 1 trains the plain projector on all 576 tokens independent of Stage 2's
# budget, so the same checkpoint serves every ladder level.
#
# Data and warm-start checkpoint both come from DAS-6 over SSH:
#   - LLaVA-Finetune tar-streamed into this node's /local_scratch
#   - baseline-<llm>-<PRETRAIN_TAG>-pretrain/mm_projector.bin copied into
#     /scratch (skipped if already present)
# MATRYOSHKA_SCALE must arrive via the environment (submit script exports it and
# uses --export=ALL). Never `--export=ALL,VAR=256,144,64,16`: sbatch splits on
# the commas and the job silently gets VAR=256.

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
: "${MATRYOSHKA_SCALE:?MATRYOSHKA_SCALE must be set (e.g. 256,144,64,16)}"
echo "[m3-ladder] SLM_KEY=${SLM_KEY} MATRYOSHKA_SCALE=${MATRYOSHKA_SCALE} BASELINE_RUN_TAG=${BASELINE_RUN_TAG:-<none>} BASELINE_PRETRAIN_TAG=${BASELINE_PRETRAIN_TAG:-<none>}"

DAS6_HOST=fs2.das6.science.uva.nl
DAS6_DATA=/var/scratch/skalra/flexllava/data/LLaVA-Finetune
DAS6_CKPT_ROOT=/var/scratch/skalra/flexllava/checkpoints
CKPT_ROOT=/scratch/skalra/flexllava_saves/checkpoints
PRETRAIN_NAME="baseline-${SLM_KEY}${BASELINE_PRETRAIN_TAG:+-${BASELINE_PRETRAIN_TAG}}-pretrain"
LOCAL_SSD=/local_scratch/skalra/flexllava_data_${SLURM_JOB_ID}
mkdir -p "$LOCAL_SSD/LLaVA-Finetune"

echo "Checking SSH reachability from this compute node to $DAS6_HOST ..."
if ! timeout 15 ssh -o BatchMode=yes -o ConnectTimeout=10 "$DAS6_HOST" "echo ok" 2>&1; then
    echo "ERROR: cannot reach $DAS6_HOST via SSH from $(hostname)." >&2
    exit 1
fi
echo "SSH OK."

# ---- warm-start checkpoint ---------------------------------------------------
PRETRAIN_DIR="$CKPT_ROOT/$PRETRAIN_NAME"
mkdir -p "$PRETRAIN_DIR"
if [ ! -f "$PRETRAIN_DIR/mm_projector.bin" ]; then
    echo "Copying $PRETRAIN_NAME from $DAS6_HOST ..."
    scp -o BatchMode=yes -q "$DAS6_HOST:$DAS6_CKPT_ROOT/$PRETRAIN_NAME/mm_projector.bin" \
        "$DAS6_HOST:$DAS6_CKPT_ROOT/$PRETRAIN_NAME/config.json" "$PRETRAIN_DIR/"
fi
ls -la "$PRETRAIN_DIR/mm_projector.bin"

# ---- data --------------------------------------------------------------------
REQUIRED_SPACE_GB=75
AVAILABLE_SPACE_GB=$(df --output=avail -BG "$LOCAL_SSD" | tail -1 | tr -dc '0-9')
echo "Local SSD available: ${AVAILABLE_SPACE_GB}G  (need ~${REQUIRED_SPACE_GB}G for LLaVA-Finetune)"
if [ "$AVAILABLE_SPACE_GB" -lt "$REQUIRED_SPACE_GB" ]; then
    echo "ERROR: not enough local disk on $(hostname):$LOCAL_SSD" >&2
    exit 1
fi

trap "echo 'Cleaning up local SSD...'; rm -rf $LOCAL_SSD" EXIT

echo "Streaming LLaVA-Finetune from $DAS6_HOST:$DAS6_DATA -> $LOCAL_SSD/LLaVA-Finetune ..."; date
# One tar-over-ssh pipe per top-level dir (5) + 1 scp = 6 concurrent SSH
# sessions, the level already proven safe against DAS-6 sshd throttling.
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

./scripts/v1_5/finetune_baseline_slm_hipster.sh "$SLM_KEY"

echo "Job Complete"; date
