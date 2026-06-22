#!/bin/bash
#SBATCH --job-name=download_ocrvqa
#SBATCH -t 08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --output=./jobs/download_ocrvqa_%A.out
#SBATCH --export=ALL

eval "$(conda shell.bash hook)"
conda activate matryoshka-mm

cd /home/skalra/FlexLLaVA

echo "OCR-VQA download started: $(date)"
echo "Node: $(hostname)"

python loadDataset.py \
    --dataset_json dataset.json \
    --out_dir /var/scratch/skalra/flexllava/data/LLaVA-Finetune/ocr_vqa/images

echo "OCR-VQA download complete: $(date)"
