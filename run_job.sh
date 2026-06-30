#!/bin/bash
#SBATCH --job-name=finetune_FlexLLaVA
#SBATCH -t 150:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:A40:2
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

# ./scripts/v1_5/pretrain_elastic.sh
./scripts/v1_5/finetune_elastic.sh

# pytest llava/model/elastic/tests/test_elastic.py -v
# python -m llava.model.elastic.tests.smoke_elastic
# python -m llava.model.elastic.tests.test_llava_elastic_e2e
# pytest llava/model/elastic/tests/test_llava_elastic_e2e.py -v

echo "Job Complete"
echo | date