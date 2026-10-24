#!/bin/bash
#SBATCH --job-name=fastv_selftest
#SBATCH -t 00:20:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --exclude=node205
#SBATCH --output=/var/scratch/skalra/flexllava/croppilot/logs/fastv_selftest_%j.out
module load cuda12.6/toolkit/12.6
eval "$(conda shell.bash hook)"; conda activate matryoshka-mm
export HF_HOME=/var/scratch/skalra/.cache/huggingface
cd /home/skalra/FlexLLaVA
python3 scripts/fastv_selftest.py "${1:-/var/scratch/skalra/flexllava/checkpoints/baseline-tinyllama-ladder-finetune}"
