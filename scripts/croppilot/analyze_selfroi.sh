#!/bin/bash
#SBATCH --job-name=selfroi_analyze
#SBATCH -p fatq
#SBATCH --nodelist=node201
#SBATCH -t 00:40:00
#SBATCH --cpus-per-task=4
#SBATCH --output=/var/scratch/skalra/flexllava/croppilot/logs/selfroi_analyze_%j.out
# I-3 stage D (CPU only: node201-203).
eval "$(conda shell.bash hook)"; conda activate matryoshka-mm
python3 /home/skalra/FlexLLaVA/scripts/croppilot/analyze_selfroi.py
