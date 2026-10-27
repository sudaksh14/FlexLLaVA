#!/bin/bash
#SBATCH --job-name=croppilot_sidecar
#SBATCH -p fatq
#SBATCH --nodelist=node201
#SBATCH -t 00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --output=/var/scratch/skalra/flexllava/croppilot/logs/sidecar_%j.out
# CPU only: node201-203.
eval "$(conda shell.bash hook)"; conda activate matryoshka-mm
hostname
python3 /home/skalra/FlexLLaVA/scripts/croppilot/build_textvqa_gtbox_sidecar.py
