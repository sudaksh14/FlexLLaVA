#!/bin/bash
# Submit the standard 4-tok-level eval for every OTTER-pipeline checkpoint.
#
#   bash scripts/otter/eval_otter_all.sh [checkpoint_root]
#
# WHY THIS EXISTS
# ---------------
# eval_lmms_slm_all.sh sweeps `"$CKPT_ROOT"/elastic-*`. Otter-pipeline runs
# write to `otter-finetune-*` (deliberately, so they can never overwrite an
# elastic-pipeline checkpoint of the same run tag), so the bulk sweep SILENTLY
# SKIPS them -- you would run the sweep, see it submit jobs for the elastic
# checkpoints, and never notice the otter ones were not evaluated.
#
# This does not modify eval_lmms_slm_all.sh. It globs otter-* and hands each
# directory to that script's existing single-checkpoint entry point, which
# needs only an elastic_config.json -- and train_otter.py writes one, because
# it runs the unchanged train() and its ElasticConfigSaver callback.
#
# Equivalent one-off for a single run:
#   bash eval_lmms_slm_all.sh /var/scratch/skalra/flexllava/checkpoints/otter-finetune-tinyllama-otter1

CKPT_ROOT=${1:-/var/scratch/skalra/flexllava/checkpoints}

shopt -s nullglob
FOUND=0
for ckpt in "$CKPT_ROOT"/otter-*; do
    [ -d "$ckpt" ] || continue
    if [ ! -f "$ckpt/elastic_config.json" ]; then
        echo "SKIP $ckpt (no elastic_config.json -- run still in progress, or it died before the first save)"
        continue
    fi
    # The eval side loads a tokenizer from the checkpoint dir. Older runs left
    # the tokenizer only in the last checkpoint-N/ subdir; warn rather than let
    # eval fail an hour into the queue.
    if [ ! -f "$ckpt/tokenizer_config.json" ]; then
        echo "WARN $ckpt has no tokenizer_config.json; if eval fails, copy the files up from its last checkpoint-N/ subdir"
    fi
    FOUND=1
    bash eval_lmms_slm_all.sh "$ckpt"
done

if [ "$FOUND" -eq 0 ]; then
    echo "No otter-pipeline checkpoints found under $CKPT_ROOT (looked for otter-*/elastic_config.json)."
    exit 1
fi
