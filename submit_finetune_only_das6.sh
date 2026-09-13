#!/bin/bash
# Submit a Stage-2-only run, resuming from an existing checkpoint-N, with
# training data streamed directly from DAS-6 instead of hipster's own
# /home/skalra/llava_data archives. See run_job_hipster_finetune_only_das6.sh
# for the staging mechanism and why Stage 1 is skippable for a Stage-2 resume.
#
#   bash submit_finetune_only_das6.sh v11-parcel-nolora tinyllama
#   DRY_RUN=1 bash submit_finetune_only_das6.sh v11-parcel-nolora tinyllama
#
# Recipe env vars below are copied verbatim from submit_elastic_run_hipster.sh
# -- do NOT let these drift from that file's case block. Getting this wrong
# reconstructs the model with a different architecture (e.g. resampler_arch)
# than the one the target checkpoint's weights were saved under, which fails
# to load correctly, possibly silently.
set -euo pipefail
cd "$(dirname "$0")"

RUN=${1:-}
SLM_KEY=${2:-tinyllama}
PARTITION_ARG=${3:-}

case "$RUN" in
  v11-parcel-nolora)
    export ELASTIC_RUN_TAG=v11-parcel-nolora
    export TOK_LEVELS="256 144 64 16"
    export LORA_RANKS="8 16 32 64"
    export RESAMPLER_ARCH=pool_anchored
    export ANCHOR_MODE=ratio
    export ANCHOR_RATIO=0.25
    export USE_TOKEN_DECORRELATION=False
    export VISION_LORA_ENABLE=False
    export VISION_TOWER="openai/clip-vit-large-patch14-336"
    export TEACHER=self
    DEFAULT_PARTITION="capacity"
    ;;
  *)
    echo "Usage: bash submit_finetune_only_das6.sh {v11-parcel-nolora} {tinyllama|smollm2} [performance|capacity]" >&2
    echo "Add other recipes here as needed -- copy the matching case from submit_elastic_run_hipster.sh." >&2
    exit 1 ;;
esac

export NUM_GPUS=4
PARTITION="${PARTITION_ARG:-$DEFAULT_PARTITION}"
case "$PARTITION" in
  performance) GPU_TYPE="rtx_6000_ada"; CPUS_PER_GPU=32 ;;
  capacity)    GPU_TYPE="l4";           CPUS_PER_GPU=16 ;;
  *) echo "ERROR: PARTITION must be 'performance' or 'capacity', got '$PARTITION'" >&2; exit 1 ;;
esac
TRAIN_GRES="gpu:${GPU_TYPE}:${NUM_GPUS}"
TRAIN_CPUS=$(( CPUS_PER_GPU * NUM_GPUS ))
EVAL_GRES="gpu:${GPU_TYPE}:1"

CKPT="/scratch/skalra/flexllava_saves/checkpoints/elastic-finetune-${SLM_KEY}-${ELASTIC_RUN_TAG}"
if [ -z "$(command find "$CKPT" -maxdepth 1 -name 'checkpoint-*' -print -quit 2>/dev/null)" ]; then
    echo "ERROR: no checkpoint-* found under $CKPT -- this launcher is for RESUMING" >&2
    echo "       an existing Stage-2 run only. Use submit_elastic_run_hipster.sh for a fresh run." >&2
    exit 1
fi

TOK_LEVELS_COUNT=$(wc -w <<< "$TOK_LEVELS")
ARRAY="0-$(( TOK_LEVELS_COUNT - 1 ))"

echo "── ${ELASTIC_RUN_TAG} (${SLM_KEY}) finetune-only, DAS-6 data, hipster:${PARTITION} ──"
for v in TOK_LEVELS LORA_RANKS RESAMPLER_ARCH ANCHOR_MODE ANCHOR_RATIO \
         USE_TOKEN_DECORRELATION VISION_LORA_ENABLE TEACHER VISION_TOWER NUM_GPUS; do
    printf '  %-24s %s\n' "$v" "${!v}"
done
printf '  %-24s %s\n' "partition" "$PARTITION"
printf '  %-24s %s\n' "train gres/cpus" "$TRAIN_GRES / $TRAIN_CPUS"
printf '  %-24s %s\n' "resuming from" "$CKPT"

if [ -n "${DRY_RUN:-}" ]; then echo "DRY_RUN set; nothing submitted."; exit 0; fi

TRAIN_ID=$(sbatch --parsable --partition="$PARTITION" --gres="$TRAIN_GRES" \
                  --ntasks-per-node="$NUM_GPUS" --cpus-per-task="$CPUS_PER_GPU" \
                  run_job_hipster_finetune_only_das6.sh "$SLM_KEY")
echo "  train job   : $TRAIN_ID  (Stage 2 only, resumes from checkpoint in place)"
EVAL_ID=$(sbatch --parsable --partition="$PARTITION" --gres="$EVAL_GRES" \
                  --cpus-per-task="$CPUS_PER_GPU" --dependency=afterok:"$TRAIN_ID" \
                  --array="$ARRAY" eval_lmms_level_hipster.sh "$CKPT")
echo "  eval job    : $EVAL_ID  (afterok:$TRAIN_ID)"

LOG=jobs/SUBMITTED_RUNS_HIPSTER.tsv
mkdir -p jobs
[ -s "$LOG" ] || printf 'submitted_utc\trecipe\tslm_key\tpartition\ttrain_job\teval_job\tcheckpoint\n' > "$LOG"
printf '%s\t%s-finetuneonly-das6\t%s\t%s\t%s\t%s\t%s\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$ELASTIC_RUN_TAG" "$SLM_KEY" "$PARTITION" \
    "$TRAIN_ID" "$EVAL_ID" "$CKPT" >> "$LOG"
echo "  recorded in : $LOG"
