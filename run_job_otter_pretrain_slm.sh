#!/bin/bash
#SBATCH --job-name=otter_pre_SLM
#SBATCH -t 48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
# A10:2 is sufficient for Stage 1 and leaves the A40s for Stage-2 work.
# Measured on 1x A10 with the exact Stage-1 config (jobs/a10probe_26742.out):
# peak 8,622 MiB of 23,028 MiB, i.e. 63% headroom. Stage 1 is light because the
# LLM and ViT are both FROZEN and --tok_levels 256 is a single level, so there
# is one forward per step and no elastic grid. On 2 GPUs it is lower still,
# since ZeRO-2 shards the optimizer states.
#SBATCH --gres=gpu:A10:2
#SBATCH --cpus-per-task=32
#SBATCH --output=./jobs/otterpre_%A.out
#SBATCH --export=ALL
# W&B credentials are NOT in this file. They live in ~/.netrc (mode 600,
# machine api.wandb.ai), which wandb reads natively on the compute node --
# $HOME is shared with the login node, so nothing needs exporting. Set it up
# once with `wandb login`, or write the three-line netrc stanza by hand.
# Previously this line carried the key inline, which put it in every commit
# and on GitHub; see docs/EXPERIMENT_JOURNAL.md section 20.
#SBATCH --exclusive

# Stage 1 (feature alignment) on the Otter-style data pipeline.
#
#   ELASTIC_RUN_TAG=otter1 sbatch run_job_otter_pretrain_slm.sh tinyllama
#
# Build the caches first (or chain with --dependency):
#   sbatch scripts/otter/prepare_otter_cache.sh tinyllama configs/otter/pretrain558k.yaml
#
# Output: checkpoints/otter-pretrain-<key>-<tag>, which Stage 2 warm-starts
# from via ELASTIC_PRETRAIN_TAG.

module load cuda12.1/toolkit/12.1
eval "$(conda shell.bash hook)"
conda activate matryoshka-mm

nvidia-smi
echo "Job Started"; date
echo "Node name: $(hostname)"; echo -n nproc=; nproc

export NCCL_SOCKET_TIMEOUT=3600
export NCCL_DEBUG=WARN
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_ASYNC_ERROR_HANDLING=1

export WANDB_PROJECT="FlexLLaVA"
export WANDB_DIR=/var/scratch/skalra/flexllava/wandb
export HF_HOME=/var/scratch/skalra/.cache/huggingface

SLM_KEY=${1:-tinyllama}
MIXTURE_CONFIG=${2:-${OTTER_MIXTURE_CONFIG:-configs/otter/pretrain558k.yaml}}
echo "[FlexLLaVA/otter] SLM_KEY=${SLM_KEY} MIXTURE=${MIXTURE_CONFIG} TAG=${ELASTIC_RUN_TAG:-<none>}"

./scripts/v1_5/pretrain_otter_slm.sh "$SLM_KEY" "$MIXTURE_CONFIG"

echo "Job Complete"; date
