#!/bin/bash
# Submits eval_lmms_level.sh (all 4 tok_levels, one SLURM array job each) for
# every SLM checkpoint found under CKPT_ROOT -- i.e. every checkpoint dir
# that is NOT a 7B (llava-*) one. Covers both pretrain and finetune stage
# checkpoints for whichever SLM backbones have been trained so far
# (tinyllama, qwen0.5b, phi2, ... -- see checkpoints/elastic-*).
#
# Each model's results land under their own eval_logs/<checkpoint_dir_name>/
# directory (see eval_lmms_level.sh), so results never mix across models --
# unlike before this was fixed, when all models shared eval_logs/<tok_level>/.
#
# Usage: bash eval_lmms_slm_all.sh [checkpoint_root]                 # all SLM checkpoints
#        bash eval_lmms_slm_all.sh [path/to/one/checkpoint]           # just that one

CKPT_ROOT=${1:-/var/scratch/skalra/flexllava/checkpoints}

if [ -f "$CKPT_ROOT/elastic_config.json" ]; then
    # A single checkpoint dir was given directly, not a root to scan.
    JID=$(sbatch --parsable eval_lmms_level.sh "$CKPT_ROOT")
    echo "Submitted array job ${JID} (tasks 0-3) for: $CKPT_ROOT"
    echo ""
    echo "Monitor:  squeue -u $USER"
    echo "Results land under: /var/scratch/skalra/flexllava/eval_logs/$(basename "$CKPT_ROOT")/<tok_level>/"
    exit 0
fi

shopt -s nullglob
FOUND=0
for ckpt in "$CKPT_ROOT"/elastic-*; do
    [ -d "$ckpt" ] || continue
    [ -f "$ckpt/elastic_config.json" ] || continue
    FOUND=1
    JID=$(sbatch --parsable eval_lmms_level.sh "$ckpt")
    echo "Submitted array job ${JID} (tasks 0-3) for: $ckpt"
done

if [ "$FOUND" -eq 0 ]; then
    echo "No SLM checkpoints found under $CKPT_ROOT (looked for elastic-*/elastic_config.json)."
    exit 1
fi

echo ""
echo "Monitor:  squeue -u $USER"
echo "Results land under: /var/scratch/skalra/flexllava/eval_logs/<checkpoint_dir_name>/<tok_level>/"
