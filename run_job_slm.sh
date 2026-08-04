#!/bin/bash
#SBATCH --job-name=finetune_FlexLLaVA_SLM
# 200h, not 150h: one epoch of Stage 2 is 5197 steps at ~103s/it on 2x A10 =
# ~149h, which run_26226 proved is too close to the old 150h limit -- it was
# CANCELLED DUE TO TIME LIMIT at 149:58:38, having reached only 92%
# (4778/5197). Both defq and fatq have an infinite partition limit, so the
# 150h was self-imposed with no headroom. Checkpoints every 500 steps mean a
# timeout is recoverable, but only if the checkpoint is healthy -- last time
# it was not (see the diverged-checkpoint note in scripts/v1_5/finetune_elastic_slm.sh).
#SBATCH -t 200:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:A10:2
#SBATCH --cpus-per-task=32
#SBATCH --output=./jobs/run_%A.out
#SBATCH --export=ALL,WANDB_API_KEY=dfcd2574507b9ebe69ca13ab6f6925d864e82ee0

module load cuda12.1/toolkit/12.1

eval "$(conda shell.bash hook)"
conda activate matryoshka-mm
# conda activate /var/scratch/skalra/prune_llm

nvidia-smi

echo "Job Started"
echo | date
echo "Node name: $(hostname)"
echo -n memory=; ulimit -m
echo -n nproc=; nproc

export NCCL_SOCKET_TIMEOUT=3600
export NCCL_DEBUG=WARN
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# run_26187/26205/26209 all deadlocked identically at step 136 (SeqNum=877060):
# the two ranks are stuck on DIFFERENT-sized collectives (65M vs 2048) = a rank
# desync in the overlapped gradient reduction, not a straggler. Fixes:
#   - overlap_comm:false in zero2.json serializes reduction (removes the desync).
#   - NCCL_ASYNC_ERROR_HANDLING=1 (torch 2.1.2 name; not the TORCH_ prefix) makes
#     the watchdog tear the job down promptly on timeout instead of hanging.
#   - --ddp_timeout in the training script fires the watchdog at 15 min instead
#     of the 30 min default, so a repeat desync fails faster.
# NOTE: DEEPSPEED_TIMEOUT was tried and had no effect -- the process-group
# timeout comes from HF's ddp_timeout via Accelerate, not that env var.
export NCCL_ASYNC_ERROR_HANDLING=1

export WANDB_PROJECT="FlexLLaVA"
export WANDB_DIR=/var/scratch/skalra/flexllava/wandb
export HF_HOME=/var/scratch/skalra/.cache/huggingface

# --- SLM pretrain variants (Stage 1) ---
# Uncomment exactly one line and submit: sbatch run_job.sh
./scripts/v1_5/pretrain_elastic_slm.sh tinyllama
# ./scripts/v1_5/pretrain_elastic_slm.sh mobilellama
# ./scripts/v1_5/pretrain_elastic_slm.sh smollm2
# ./scripts/v1_5/pretrain_elastic_slm.sh qwen0.5b
# ./scripts/v1_5/pretrain_elastic_slm.sh qwen1.5b
# ./scripts/v1_5/pretrain_elastic_slm.sh qwen3b
# ./scripts/v1_5/pretrain_elastic_slm.sh phi2
# ./scripts/v1_5/pretrain_elastic_slm.sh stablelm

# --- SLM finetune variants (Stage 2, run after matching pretrain above) ---
./scripts/v1_5/finetune_elastic_slm.sh tinyllama
# ./scripts/v1_5/finetune_elastic_slm.sh mobilellama
# ./scripts/v1_5/finetune_elastic_slm.sh smollm2
# ./scripts/v1_5/finetune_elastic_slm.sh qwen0.5b
# ./scripts/v1_5/finetune_elastic_slm.sh qwen1.5b
# ./scripts/v1_5/finetune_elastic_slm.sh qwen3b
# ./scripts/v1_5/finetune_elastic_slm.sh phi2
# ./scripts/v1_5/finetune_elastic_slm.sh stablelm

echo "Job Complete"
echo | date