#!/bin/bash
#SBATCH --job-name=finetune_FlexLLaVA_SLM
#SBATCH -t 150:00:00
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

export WANDB_PROJECT="FlexLLaVA"
export WANDB_DIR=/var/scratch/skalra/flexllava/wandb
export HF_HOME=/var/scratch/skalra/.cache/huggingface

# --- SLM pretrain variants (Stage 1) ---
# Uncomment exactly one line and submit: sbatch run_job.sh
# ./scripts/v1_5/pretrain_elastic_slm.sh tinyllama
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