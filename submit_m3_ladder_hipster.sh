#!/bin/bash
# Submit the budget-matched M3 ladder finetune (Stage 2 only) on hipster.
#
#   bash submit_m3_ladder_hipster.sh [performance|capacity]
#   DRY_RUN=1 bash submit_m3_ladder_hipster.sh
#
# Hipster port of launch_ladder_baselines.sh's M3 half: M3's own native
# multi-scale loop over 256,144,64,16 (one forward per scale per step, CE
# averaged), reusing the 4tok Stage-1 projector from DAS-6. The ladder is
# exported in THIS shell and passed with --export=ALL; never put it in
# `--export=ALL,VAR=256,144,64,16` (sbatch splits on the commas and the job
# silently receives VAR=256).
set -uo pipefail
cd "$(dirname "$0")"

PARTITION="${1:-performance}"
case "$PARTITION" in
  performance) GPU_TYPE="rtx_6000_ada"; CPUS_PER_GPU=32 ;;
  capacity)    GPU_TYPE="l4";           CPUS_PER_GPU=16 ;;
  *) echo "ERROR: PARTITION must be 'performance' or 'capacity', got '$PARTITION'" >&2; exit 1 ;;
esac

SLM_KEY="${SLM_KEY:-tinyllama}"
export NUM_GPUS="${NUM_GPUS:-4}"
export MATRYOSHKA_SCALE="${MATRYOSHKA_SCALE:-256,144,64,16}"
export BASELINE_RUN_TAG="${BASELINE_RUN_TAG:-ladder}"
export BASELINE_PRETRAIN_TAG="${BASELINE_PRETRAIN_TAG:-4tok}"
GRES="gpu:${GPU_TYPE}:${NUM_GPUS}"
CKPT="/scratch/skalra/flexllava_saves/checkpoints/baseline-${SLM_KEY}-${BASELINE_RUN_TAG}-finetune"

echo "── M3 ladder finetune (${SLM_KEY}) on hipster:${PARTITION} ──────────────────"
printf '  %-24s %s\n' "MATRYOSHKA_SCALE"       "$MATRYOSHKA_SCALE"
printf '  %-24s %s\n' "run tag"                "$BASELINE_RUN_TAG"
printf '  %-24s %s\n' "warm-start (pretrain)"  "baseline-${SLM_KEY}-${BASELINE_PRETRAIN_TAG}-pretrain  (from DAS-6)"
printf '  %-24s %s\n' "partition"              "$PARTITION"
printf '  %-24s %s\n' "gres/cpus"              "$GRES / $CPUS_PER_GPU per task"
printf '  %-24s %s\n' "checkpoint"             "$CKPT"

if [ -n "${DRY_RUN:-}" ]; then echo; echo "DRY_RUN set; nothing submitted."; exit 0; fi

JOB_ID=$(sbatch --parsable --export=ALL --partition="$PARTITION" --gres="$GRES" \
              --ntasks-per-node="$NUM_GPUS" --cpus-per-task="$CPUS_PER_GPU" \
              run_job_hipster_m3_ladder_finetune.sh "$SLM_KEY")
echo "  train job   : $JOB_ID  (Stage 2 only)"

LOG=jobs/SUBMITTED_RUNS_HIPSTER.tsv
mkdir -p jobs
[ -s "$LOG" ] || printf 'submitted_utc\trecipe\tslm_key\tpartition\ttrain_job\teval_job\tcheckpoint\n' > "$LOG"
printf '%s\tbaseline-m3-%s\t%s\t%s\t%s\t-\t%s\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$BASELINE_RUN_TAG" "$SLM_KEY" "$PARTITION" "$JOB_ID" "$CKPT" >> "$LOG"
echo "  recorded in : $LOG"
