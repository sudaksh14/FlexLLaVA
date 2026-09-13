#!/bin/bash
#SBATCH --job-name=pretrain_only_das6data
#SBATCH -t 24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
# Overridden at submit time -- partition/gres depend on the backbone/matrix.
#SBATCH --partition=capacity
#SBATCH --gres=gpu:l4:2
#SBATCH --cpus-per-task=16
#SBATCH --output=./jobs/run_hipster_das6data_%A.out
#SBATCH --export=ALL,WANDB_API_KEY=dfcd2574507b9ebe69ca13ab6f6925d864e82ee0
#
# Stage 1 ONLY -- no finetune_elastic_slm_hipster.sh call. Mirrors DAS-6's
# run_job_pretrain_slm.sh, adapted for hipster: hipster paths (via
# pretrain_elastic_slm_hipster.sh's own CHECKPOINT_ROOT/SAVE_ROOT), hipster
# module name, hipster conda activation, DAS-6 data streaming instead of a
# local archive/copy.
#
# Data source: DAS-6, not hipster's own /home/skalra/llava_data archives --
# tar streamed over SSH straight into this node's /local_scratch. Only
# LLaVA-Pretrain is needed (Stage 1 never touches LLaVA-Finetune).
# LLaVA-Pretrain has 661 shard dirs (unlike Finetune's 5-6 large leaf dirs),
# so it is chunked into N_PARALLEL groups of shards, one ssh|tar pipe per
# group, rather than one pipe per shard (which would open 661 concurrent SSH
# connections) or one pipe for the whole tree (no parallelism at all).
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
echo "[FlexLLaVA] SLM_KEY=${SLM_KEY}  ELASTIC_RUN_TAG=${ELASTIC_RUN_TAG:-<none>}  (pretrain-only, DAS-6 data source)"

DAS6_HOST=fs2.das6.science.uva.nl
DAS6_DATA=/var/scratch/skalra/flexllava/data/LLaVA-Pretrain
LOCAL_SSD=/local_scratch/skalra/flexllava_data_${SLURM_JOB_ID}
mkdir -p "$LOCAL_SSD/LLaVA-Pretrain"

echo "Checking SSH reachability from this compute node to $DAS6_HOST ..."
if ! timeout 15 ssh -o BatchMode=yes -o ConnectTimeout=10 "$DAS6_HOST" "echo ok" 2>&1; then
    echo "ERROR: cannot reach $DAS6_HOST via SSH from $(hostname)." >&2
    exit 1
fi
echo "SSH OK."

REQUIRED_SPACE_GB=35
AVAILABLE_SPACE_GB=$(df --output=avail -BG "$LOCAL_SSD" | tail -1 | tr -dc '0-9')
echo "Local SSD available: ${AVAILABLE_SPACE_GB}G  (need ~${REQUIRED_SPACE_GB}G for LLaVA-Pretrain)"
if [ "$AVAILABLE_SPACE_GB" -lt "$REQUIRED_SPACE_GB" ]; then
    echo "ERROR: not enough local disk on $(hostname):$LOCAL_SSD" >&2
    exit 1
fi

trap "echo 'Cleaning up local SSD...'; rm -rf $LOCAL_SSD" EXIT

echo "Listing LLaVA-Pretrain shards on $DAS6_HOST ..."; date
mapfile -t SHARDS < <(timeout 30 ssh -o BatchMode=yes "$DAS6_HOST" \
    "find '$DAS6_DATA' -mindepth 1 -maxdepth 1 -type d -printf '%f\n'" | sort)
echo "${#SHARDS[@]} shard dirs found."

N_PARALLEL="${STAGE_PARALLEL:-8}"
echo "Streaming LLaVA-Pretrain from $DAS6_HOST:$DAS6_DATA -> $LOCAL_SSD/LLaVA-Pretrain (chunked into $N_PARALLEL groups) ..."; date
PIDS=()
for i in $(seq 0 $(( N_PARALLEL - 1 ))); do
    CHUNK=()
    for ((j=i; j<${#SHARDS[@]}; j+=N_PARALLEL)); do CHUNK+=("${SHARDS[$j]}"); done
    [ ${#CHUNK[@]} -eq 0 ] && continue
    ssh -o BatchMode=yes "$DAS6_HOST" "tar -cf - -C $DAS6_DATA ${CHUNK[*]}" \
        | tar -xf - -C "$LOCAL_SSD/LLaVA-Pretrain" &
    PIDS+=($!)
done
scp -o BatchMode=yes -q "$DAS6_HOST:$DAS6_DATA/blip_laion_cc_sbu_558k.json" \
    "$LOCAL_SSD/LLaVA-Pretrain/blip_laion_cc_sbu_558k.json" &
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
du -sh "$LOCAL_SSD/LLaVA-Pretrain"
echo "File count: $(command find "$LOCAL_SSD/LLaVA-Pretrain" -type f | wc -l)"
export LOCAL_SSD

./scripts/v1_5/pretrain_elastic_slm_hipster.sh "$SLM_KEY"

echo "Job Complete"; date
