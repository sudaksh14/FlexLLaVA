#!/bin/bash
#SBATCH --job-name=pretrain_FlexLLaVA
#SBATCH -t 150:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:A40:2
#SBATCH --cpus-per-task=32
#SBATCH --output=./jobs/run_%A.out
#SBATCH --export=ALL
# W&B credentials are NOT in this file. They live in ~/.netrc (mode 600,
# machine api.wandb.ai), which wandb reads natively on the compute node --
# $HOME is shared with the login node, so nothing needs exporting. Set it up
# once with `wandb login`, or write the three-line netrc stanza by hand.
# Previously this line carried the key inline, which put it in every commit
# and on GitHub; see docs/EXPERIMENT_JOURNAL.md section 20.
# Training jobs take every GPU on their node, so no other GPU job can use it.
# --exclusive therefore also claims all 64 cores: without it, cons_tres/CR_CORE
# confines the job to --cpus-per-task cores (task/affinity) and the rest idle
# for nothing. It also makes the allocation robust to a smaller
# --cpus-per-task passed at submit time.
#SBATCH --exclusive

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
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export WANDB_PROJECT="FlexLLaVA"
export WANDB_DIR=/var/scratch/skalra/flexllava/wandb
export HF_HOME=/var/scratch/skalra/.cache/huggingface

./scripts/v1_5/pretrain_elastic.sh

echo "Job Complete"
echo | date
