#!/bin/bash
#SBATCH --job-name=download_data
#SBATCH -t 08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --output=./jobs/download_data_%A.out
#SBATCH --export=ALL

eval "$(conda shell.bash hook)"
conda activate matryoshka-mm

export HF_HOME=/var/scratch/skalra/.cache/huggingface

echo "Download started"
date

/var/scratch/skalra/envs/matryoshka-mm/bin/python3.10 - << 'EOF'
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="liuhaotian/LLaVA-Pretrain",
    repo_type="dataset",
    local_dir="/var/scratch/skalra/flexllava/data/LLaVA-Pretrain",
)
print("Done.")
EOF

echo "Download complete"
date
