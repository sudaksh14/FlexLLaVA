#!/bin/bash
# Submits all 4 tok_levels as a SLURM array job (one GPU each, runs in parallel).
#
# Usage: bash eval_lmms_all.sh [model_path]

MODEL_PATH=${1:-/var/scratch/skalra/flexllava/checkpoints/llava-elastic-pretrain}

JID=$(sbatch --parsable eval_lmms_level.sh "$MODEL_PATH")
echo "Submitted array job ${JID} (tasks 0-3) for: $MODEL_PATH"
echo ""
echo "Monitor:"
echo "  squeue -u $USER"
echo "  tail -f jobs/eval_lmms_${JID}_tok0.out   # 256tok"
echo "  tail -f jobs/eval_lmms_${JID}_tok1.out   # 144tok"
echo "  tail -f jobs/eval_lmms_${JID}_tok2.out   # 64tok"
echo "  tail -f jobs/eval_lmms_${JID}_tok3.out   # 16tok"
