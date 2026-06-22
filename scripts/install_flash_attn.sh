#!/bin/bash
#SBATCH --job-name=install_flash_attn
#SBATCH -t 01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:A40:1
#SBATCH --cpus-per-task=8
#SBATCH --nodelist=node205
#SBATCH --output=./jobs/install_flash_%A.out

module load cuda12.6/toolkit/12.6

eval "$(conda shell.bash hook)"
conda activate matryoshka-mm

echo "Pinning setuptools to 69 (torch 2.1 uses pkg_resources.packaging removed in setuptools >=70)..."
pip install "setuptools==69.5.1"

echo "Installing flash-attn from source on $(hostname) (takes ~20 min)..."
# --no-binary forces source compilation against the system GLIBC (2.28 on RHEL 8)
# MAX_JOBS limits parallel compile workers to avoid OOM during build
# 2.8.x requires PyTorch >=2.4 (std::optional change); 2.3.6 targets PyTorch 2.1.x
MAX_JOBS=4 pip install "flash-attn==2.3.6" --no-build-isolation --no-binary flash-attn --no-cache-dir

echo "Verifying..."
python -c "import flash_attn; print('flash_attn', flash_attn.__version__, 'OK')"
