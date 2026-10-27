#!/bin/bash
#SBATCH --job-name=selfroi_boxes
#SBATCH -p fatq
#SBATCH --nodelist=node201
#SBATCH -t 01:00:00
#SBATCH --cpus-per-task=8
#SBATCH --output=/var/scratch/skalra/flexllava/croppilot/logs/selfroi_boxes_%j.out
# I-3 stage B (CPU only: node201-203).  Args: [guide_dir] [out_dir]
eval "$(conda shell.bash hook)"; conda activate matryoshka-mm
python3 /home/skalra/FlexLLaVA/scripts/croppilot/selfroi_boxes.py "$@"
