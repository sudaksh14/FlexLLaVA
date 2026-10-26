#!/bin/bash
#SBATCH --job-name=summ_tiny
#SBATCH -p fatq
#SBATCH --nodelist=node201
#SBATCH -t 00:10:00
#SBATCH --cpus-per-task=2
#SBATCH --output=/var/scratch/skalra/flexllava/croppilot/logs/summarize_tinyllama_%j.out
eval "$(conda shell.bash hook)"; conda activate matryoshka-mm
python3 /home/skalra/FlexLLaVA/scripts/summarize_tinyllama.py
