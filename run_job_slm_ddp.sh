#!/bin/bash
#SBATCH --job-name=finetune_SLM_ddp
#SBATCH -t 150:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:A40:2
#SBATCH --cpus-per-task=32
#SBATCH --output=./jobs/run_%A.out
#SBATCH --export=ALL,WANDB_API_KEY=dfcd2574507b9ebe69ca13ab6f6925d864e82ee0

# Full fine-tune of the SLM on 2x A40 via PURE PyTorch DDP (no DeepSpeed), to
# sidestep the ZeRO-2 embed-reduction deadlock. torchrun (inside the finetune
# script) forks the 2 ranks itself, so this is a single SLURM task.

module load cuda12.1/toolkit/12.1

eval "$(conda shell.bash hook)"
conda activate matryoshka-mm

nvidia-smi

echo "Job Started"
echo | date
echo "Node name: $(hostname)"
echo -n memory=; ulimit -m
echo -n nproc=; nproc

export NCCL_SOCKET_TIMEOUT=3600
export NCCL_DEBUG=WARN
export NCCL_ASYNC_ERROR_HANDLING=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export WANDB_PROJECT="FlexLLaVA"
export WANDB_DIR=/var/scratch/skalra/flexllava/wandb
export HF_HOME=/var/scratch/skalra/.cache/huggingface

./scripts/v1_5/finetune_elastic_slm_ddp.sh tinyllama

echo "Job Complete"
echo | date
