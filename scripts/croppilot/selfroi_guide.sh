#!/bin/bash
#SBATCH --job-name=selfroi_guide
#SBATCH -t 04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --exclude=node205
#SBATCH --output=/var/scratch/skalra/flexllava/croppilot/logs/selfroi_guide_%x_%A_%a.out
#SBATCH --array=0-2

# I-3 stage A.  GUIDE=3 (16 tok) or GUIDE=0 (256 tok); one shard per array task.
#   GUIDE=3 sbatch selfroi_guide.sh                      # full
#   GUIDE=3 LIMIT=12 sbatch --array=0 selfroi_guide.sh   # smoke test
# GPU jobs run on node206-208 only (node205 excluded).
GUIDE=${GUIDE:?set GUIDE=3 or 0}
LIMIT_ARG=""; [ -n "$LIMIT" ] && LIMIT_ARG="--limit $LIMIT"
OUT=/var/scratch/skalra/flexllava/croppilot/selfroi
[ -n "$LIMIT" ] && OUT=${OUT}_smoke

module load cuda12.6/toolkit/12.6
eval "$(conda shell.bash hook)"
conda activate matryoshka-mm
export HF_HOME=/var/scratch/skalra/.cache/huggingface
export HF_DATASETS_CACHE=/var/scratch/skalra/.cache/huggingface/datasets
cd /home/skalra/FlexLLaVA
echo "Node: $(hostname)  $(date)"; nvidia-smi | head -12
pip show lmms-eval >/dev/null 2>&1 || pip install -e lmms-eval -q
python3 scripts/croppilot/selfroi_guide.py --level "$GUIDE" --shard "$SLURM_ARRAY_TASK_ID" --nshards 3 --out "$OUT" $LIMIT_ARG
echo "Done $(date)"
