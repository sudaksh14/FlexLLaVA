#!/bin/bash
# Submit a Stage-2-only run using hipster's own /home/skalra/llava_data
# archives (not DAS-6 -- see submit_finetune_only_das6.sh for that variant).
# Two cases, both handled automatically by train.py itself:
#   - a checkpoint-N already exists under the Stage-2 output dir -> resumes
#     from it (Stage-1 checkpoint irrelevant, need not exist)
#   - none exists -> fresh Stage 2, warm-started from the Stage-1 checkpoint
#     at elastic-pretrain-<key>-<tag> if one exists there (a no-op, not an
#     error, if it doesn't -- see llava/train/train.py's
#     _load_elastic_pretrain_weights)
#
#   bash submit_finetune_only.sh v11-parcel-nolora smollm2
#   DRY_RUN=1 bash submit_finetune_only.sh v11-parcel-nolora smollm2
#
# Recipe env vars copied verbatim from submit_elastic_run_hipster.sh -- do
# NOT let these drift from that file's case block. A mismatch (e.g. wrong
# resampler_arch) reconstructs the model differently from whatever the
# target checkpoint's weights were saved under.
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
    echo "Usage: bash submit_finetune_only.sh {v11-parcel-nolora} {tinyllama|smollm2} [performance|capacity]" >&2
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
CPUS_PER_TASK="$CPUS_PER_GPU"
EVAL_GRES="gpu:${GPU_TYPE}:1"

PRETRAIN_CKPT="/scratch/skalra/flexllava_saves/checkpoints/elastic-pretrain-${SLM_KEY}-${ELASTIC_RUN_TAG}"
FINETUNE_CKPT="/scratch/skalra/flexllava_saves/checkpoints/elastic-finetune-${SLM_KEY}-${ELASTIC_RUN_TAG}"

if [ -n "$(command find "$FINETUNE_CKPT" -maxdepth 1 -name 'checkpoint-*' -print -quit 2>/dev/null)" ]; then
    MODE="resume (checkpoint already present under $FINETUNE_CKPT)"
elif [ -f "$PRETRAIN_CKPT/model.safetensors" ] || [ -n "$(command find "$PRETRAIN_CKPT" -maxdepth 1 -name 'pytorch_model*.bin' -print -quit 2>/dev/null)" ]; then
    MODE="fresh Stage 2, warm-started from $PRETRAIN_CKPT"
else
    echo "ERROR: no Stage-2 checkpoint to resume AND no Stage-1 checkpoint to warm-start from." >&2
    echo "       Checked: $FINETUNE_CKPT (checkpoint-*), $PRETRAIN_CKPT (model.safetensors)." >&2
    echo "       Use submit_elastic_run_hipster.sh for a genuinely fresh run (Stage 1 + Stage 2)." >&2
    exit 1
fi

TOK_LEVELS_COUNT=$(wc -w <<< "$TOK_LEVELS")
ARRAY="0-$(( TOK_LEVELS_COUNT - 1 ))"

echo "── ${ELASTIC_RUN_TAG} (${SLM_KEY}) finetune-only, hipster data, hipster:${PARTITION} ──"
for v in TOK_LEVELS LORA_RANKS RESAMPLER_ARCH ANCHOR_MODE ANCHOR_RATIO \
         USE_TOKEN_DECORRELATION VISION_LORA_ENABLE TEACHER VISION_TOWER NUM_GPUS; do
    printf '  %-24s %s\n' "$v" "${!v}"
done
printf '  %-24s %s\n' "partition" "$PARTITION"
printf '  %-24s %s\n' "train gres/cpus" "$TRAIN_GRES / $((CPUS_PER_TASK * NUM_GPUS))"
printf '  %-24s %s\n' "mode" "$MODE"

if [ -n "${DRY_RUN:-}" ]; then echo "DRY_RUN set; nothing submitted."; exit 0; fi

TRAIN_ID=$(sbatch --parsable --partition="$PARTITION" --gres="$TRAIN_GRES" \
                  --ntasks-per-node="$NUM_GPUS" --cpus-per-task="$CPUS_PER_TASK" \
                  run_job_hipster_finetune_only.sh "$SLM_KEY")
echo "  train job   : $TRAIN_ID  ($MODE)"
EVAL_ID=$(sbatch --parsable --partition="$PARTITION" --gres="$EVAL_GRES" \
                  --cpus-per-task="$CPUS_PER_TASK" --dependency=afterok:"$TRAIN_ID" \
                  --array="$ARRAY" eval_lmms_level_hipster.sh "$FINETUNE_CKPT")
echo "  eval job    : $EVAL_ID  (afterok:$TRAIN_ID)"

LOG=jobs/SUBMITTED_RUNS_HIPSTER.tsv
mkdir -p jobs
[ -s "$LOG" ] || printf 'submitted_utc\trecipe\tslm_key\tpartition\ttrain_job\teval_job\tcheckpoint\n' > "$LOG"
printf '%s\t%s-finetuneonly\t%s\t%s\t%s\t%s\t%s\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$ELASTIC_RUN_TAG" "$SLM_KEY" "$PARTITION" \
    "$TRAIN_ID" "$EVAL_ID" "$FINETUNE_CKPT" >> "$LOG"
echo "  recorded in : $LOG"
