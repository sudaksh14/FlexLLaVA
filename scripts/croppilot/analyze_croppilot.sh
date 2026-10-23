#!/bin/bash
#SBATCH --job-name=croppilot_analyze
#SBATCH -p fatq
#SBATCH --nodelist=node201
#SBATCH -t 00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --output=/var/scratch/skalra/flexllava/croppilot/logs/analyze_%j.out
# CPU only: node201-203.
eval "$(conda shell.bash hook)"; conda activate matryoshka-mm
python3 /home/skalra/FlexLLaVA/scripts/croppilot/analyze_croppilot.py "$@"
