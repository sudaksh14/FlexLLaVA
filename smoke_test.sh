#!/bin/bash
#SBATCH --job-name=smoke_test
#SBATCH -t 00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:A40:2
#SBATCH --cpus-per-task=32
#SBATCH --nodelist=node205
#SBATCH --output=./jobs/smoke_%A.out

module load cuda12.6/toolkit/12.6

eval "$(conda shell.bash hook)"
conda activate matryoshka-mm

nvidia-smi
echo "Job Started"; date
echo "Node: $(hostname)"

export HF_HOME=/var/scratch/skalra/.cache/huggingface
export SMOKE_DATA_DIR=/var/scratch/skalra/flexllava/data/smoke_test
export SMOKE_MAX_STEPS=5

# ---------- generate dummy data ----------
/var/scratch/skalra/envs/matryoshka-mm/bin/python3.10 - <<'PYEOF'
import json, os
from pathlib import Path
from PIL import Image
import numpy as np

data_dir = Path(os.environ["SMOKE_DATA_DIR"])
img_dir  = data_dir / "images"
img_dir.mkdir(parents=True, exist_ok=True)

# 20 unique images reused across 2000 JSON entries
# (5 steps × 32 per_device × 4 grad_accum × 2 GPUs = 1280 samples needed)
N_IMAGES = 20
N_SAMPLES = 2000

img_files = []
for i in range(N_IMAGES):
    fname = f"smoke_{i:04d}.jpg"
    color = tuple(np.random.randint(30, 230, 3).tolist())
    Image.new("RGB", (336, 336), color).save(img_dir / fname, quality=85)
    img_files.append(fname)

samples = []
for i in range(N_SAMPLES):
    fname = img_files[i % N_IMAGES]
    samples.append({
        "id": f"smoke_{i:04d}",
        "image": fname,
        "conversations": [
            {"from": "human",
             "value": f"<image>\nDescribe this image briefly."},
            {"from": "gpt",
             "value": f"This is a solid-color test image number {i % N_IMAGES}."}
        ]
    })

with open(data_dir / "smoke_data.json", "w") as f:
    json.dump(samples, f)

print(f"Generated {len(samples)} samples ({N_IMAGES} unique images) in {data_dir}")
PYEOF

echo "Dummy data ready"

# ---------- run smoke training ----------
./scripts/v1_5/smoke_pretrain.sh

echo "Smoke test complete"; date
