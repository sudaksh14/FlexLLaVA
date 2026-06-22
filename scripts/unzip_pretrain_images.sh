#!/bin/bash
#SBATCH --job-name=unzip_images
#SBATCH -t 02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --output=./jobs/unzip_%A.out
#SBATCH --export=ALL

DATA_DIR=/var/scratch/skalra/flexllava/data/LLaVA-Pretrain
ZIP_FILE=${DATA_DIR}/images.zip

echo "Unzip started"; date

# images.zip extracts numbered subdirs (00000-00664) directly into DATA_DIR
unzip -q "${ZIP_FILE}" -d "${DATA_DIR}/"

echo "Unzip complete"; date
echo "Image count: $(find "${DATA_DIR}" -mindepth 2 -maxdepth 2 -type f \( -name "*.jpg" -o -name "*.png" -o -name "*.jpeg" \) | wc -l)"
du -sh "${DATA_DIR}"
