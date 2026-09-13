#!/bin/bash
# Submit one named elastic experiment on the hipster cluster (Stage 1 + Stage
# 2 + its eval array). Hipster port of submit_elastic_run.sh (DAS-6) --
# see that file for the "why" behind every recipe value below; the case
# bodies here are copied verbatim from it so the two clusters run the exact
# same experiments. Only cluster mechanics differ: partition/GRES selection,
# save paths, and which sbatch scripts get called.
#
#   bash submit_elastic_run_hipster.sh v11-parcel-nolora tinyllama
#   bash submit_elastic_run_hipster.sh v11-parcel-nolora smollm2 performance
#   bash submit_elastic_run_hipster.sh v12-parcel-kd7b   tinyllama
#   DRY_RUN=1 bash submit_elastic_run_hipster.sh v11-parcel-nolora tinyllama
#
# Partitions (checked 2026-09-09):
#   performance -- 5 nodes, 8x RTX 6000 Ada (48GB) each, 256 CPU/node.
#                  Needs 32 CPU/GPU (SLURM-enforced). Mostly full at check
#                  time (4/5 nodes fully allocated, 1 free GPU on the 5th) --
#                  a 4-GPU training job here will likely queue a while.
#   capacity    -- 8 nodes, 8x L4 (24GB) each, 128 CPU/node (default
#                  partition). Needs 16 CPU/GPU. More headroom at check time
#                  (two nodes had 4-5 free GPUs each, enough for a 4-GPU job
#                  without waiting on a second node -- --gres cannot span
#                  nodes anyway, and capacity's own MaxNodes=2/job doesn't
#                  matter here since we only ever request one node).
# Neither is a dedicated pool like DAS-6 -- this is a large shared cluster
# with dozens of other users' jobs queued at any time, so nothing here uses
# --exclusive (see run_job_hipster.sh). Training jobs request 4 GPUs (see
# NUM_GPUS below); eval stays at 1 GPU (single-process, doesn't parallelize).
# SLURM's own --test-only estimate for a fresh 2-GPU job was ~7-9 days out
# under load at check time; a 4-GPU ask is a strictly harder bin-pack, so
# expect that estimate to be a floor, not a ceiling, especially on
# performance. Treat these as real (if often conservative/backfill-
# pessimistic) scheduler signals, not guesses.
set -euo pipefail
cd "$(dirname "$0")"

RUN=${1:-}
SLM_KEY=${2:-tinyllama}
PARTITION_ARG=${3:-}

# ---- shared across every recipe below (identical to DAS-6) ----------------
export TOK_LEVELS="256 144 64 16"
export STAGE1_TOK_LEVEL=256
export RESAMPLER_ARCH=pool_anchored
export ANCHOR_MODE=ratio
export ANCHOR_RATIO=0.25
export VISION_TOWER="openai/clip-vit-large-patch14-336"
export TEACHER=self

# 4 GPUs per training job on hipster (DAS-6 uses 2). GRAD_ACCUM in both
# pretrain_elastic_slm_hipster.sh and finetune_elastic_slm_hipster.sh is
# derived from NUM_GPUS (effective_batch = per_device * accum * num_gpus), so
# this halves accum automatically and keeps the effective batch identical to
# DAS-6 -- not a recipe change, same mechanism DAS-6 already uses for its own
# NUM_GPUS=1 single-GPU-node fallback. Eval is untouched: eval_lmms_level_
# hipster.sh runs one tok_level per array task as a single accelerate process
# (--num_processes=1) -- a 2nd/3rd/4th GPU would sit idle, not speed anything
# up -- so it stays at 1 GPU below.
export NUM_GPUS=4

# Recipe default partition; v12 forces performance below regardless (VRAM).
DEFAULT_PARTITION="capacity"

case "$RUN" in
  v11-parcel-nolora)
    # PARCEL alone: decorrelation OFF, vision LoRA OFF. See
    # docs/EXPERIMENT_JOURNAL.md sections 16d/16e/16g on DAS-6 (same file,
    # same home) for why this run matters -- not reproduced here per the
    # "don't touch docs" instruction for this cluster.
    export ELASTIC_RUN_TAG=v11-parcel-nolora
    export USE_TOKEN_DECORRELATION=False
    export VISION_LORA_ENABLE=False
    export LORA_RANKS="8 16 32 64"
    export STAGE1_LORA_RANK=64
    ;;
  v12-parcel-kd7b)
    # v11 + a frozen external LLaVA-1.5-7B KD teacher.
    case "$SLM_KEY" in
      tinyllama|mobilellama) ;;
      *)
        echo "ERROR: v12-parcel-kd7b needs a Llama-32000-vocab backbone for the" >&2
        echo "       frozen LLaVA-1.5-7B teacher; SLM_KEY='$SLM_KEY' is not one." >&2
        echo "       Valid: tinyllama, mobilellama. attach_kd_teacher would raise" >&2
        echo "       on the vocab_size mismatch after Stage 1 had already run." >&2
        echo "       (smollm2 is NOT eligible -- this is not a hipster-specific" >&2
        echo "       restriction, it is the same vocab guard DAS-6's recipe uses.)" >&2
        exit 1 ;;
    esac
    export ELASTIC_RUN_TAG=v12-parcel-kd7b
    export USE_TOKEN_DECORRELATION=False
    export VISION_LORA_ENABLE=False
    export LORA_RANKS="8 16 32 64"
    export STAGE1_LORA_RANK=64
    export TEACHER=llava
    export TEACHER_MODEL_PATH=liuhaotian/llava-v1.5-7b
    export PREFIX_KL_WEIGHT=0.1
    # Frozen 7B costs ~14GB/GPU on top of the student and is not ZeRO-sharded
    # (plain attribute on ElasticEngine, not a submodule). L4's 24GB does not
    # have headroom for that on top of a training run; RTX 6000 Ada's 48GB
    # does (same logic as DAS-6 requiring A40-class, not A10).
    DEFAULT_PARTITION="performance"
    if [ -n "$PARTITION_ARG" ] && [ "$PARTITION_ARG" != "performance" ]; then
        echo "ERROR: v12-parcel-kd7b needs the 'performance' partition (RTX 6000" >&2
        echo "       Ada, 48GB) for the frozen 7B teacher's ~14GB/GPU unsharded" >&2
        echo "       overhead -- 'capacity' (L4, 24GB) does not fit it alongside" >&2
        echo "       training. Requested partition '$PARTITION_ARG' refused." >&2
        exit 1
    fi
    ;;
  *)
    echo "Usage: bash submit_elastic_run_hipster.sh {v11-parcel-nolora|v12-parcel-kd7b} {tinyllama|smollm2} [performance|capacity]" >&2
    exit 1
    ;;
esac

PARTITION="${PARTITION_ARG:-$DEFAULT_PARTITION}"
case "$PARTITION" in
  performance) GPU_TYPE="rtx_6000_ada"; CPUS_PER_GPU=32 ;;
  capacity)    GPU_TYPE="l4";           CPUS_PER_GPU=16 ;;
  *)
    echo "ERROR: PARTITION must be 'performance' or 'capacity', got '$PARTITION'" >&2
    exit 1 ;;
esac
TRAIN_GRES="gpu:${GPU_TYPE}:${NUM_GPUS}"
TRAIN_CPUS=$(( CPUS_PER_GPU * NUM_GPUS ))
EVAL_GRES="gpu:${GPU_TYPE}:1"
EVAL_CPUS=$CPUS_PER_GPU

# Checkpoints moved to /scratch 2026-09-11 -- must match the CHECKPOINT_ROOT
# in scripts/v1_5/{pretrain,finetune}_elastic_slm_hipster.sh.
CKPT="/scratch/skalra/flexllava_saves/checkpoints/elastic-finetune-${SLM_KEY}-${ELASTIC_RUN_TAG}"
N_LEVELS=$(wc -w <<< "$TOK_LEVELS")
ARRAY="0-$(( N_LEVELS - 1 ))"

echo "── ${ELASTIC_RUN_TAG} (${SLM_KEY}) on hipster:${PARTITION} ──────────────────"
for v in TOK_LEVELS LORA_RANKS STAGE1_TOK_LEVEL STAGE1_LORA_RANK RESAMPLER_ARCH \
         ANCHOR_MODE ANCHOR_RATIO USE_TOKEN_DECORRELATION DECORR_WEIGHT \
         VISION_LORA_ENABLE VISION_LORA_SPECIALIZE_TOK TEACHER TEACHER_MODEL_PATH \
         PREFIX_KL_WEIGHT VISION_TOWER; do
    printf '  %-26s %s\n' "$v" "${!v:-<launcher default>}"
done
printf '  %-26s %s\n' "partition" "$PARTITION"
printf '  %-26s %s\n' "train gres/cpus" "$TRAIN_GRES / $TRAIN_CPUS"
printf '  %-26s %s\n' "eval gres/cpus" "$EVAL_GRES / $EVAL_CPUS"
printf '  %-26s %s\n' "eval --array" "$ARRAY"
printf '  %-26s %s\n' "checkpoint" "$CKPT"

if [ -n "${DRY_RUN:-}" ]; then echo "DRY_RUN set; nothing submitted."; exit 0; fi

TRAIN_ID=$(sbatch --parsable --partition="$PARTITION" --gres="$TRAIN_GRES" \
                  --ntasks-per-node="$NUM_GPUS" --cpus-per-task="$CPUS_PER_GPU" \
                  run_job_hipster.sh "$SLM_KEY")
echo "  train job   : $TRAIN_ID  (Stage 1 + Stage 2)"
EVAL_ID=$(sbatch --parsable --partition="$PARTITION" --gres="$EVAL_GRES" \
                  --cpus-per-task="$EVAL_CPUS" --dependency=afterok:"$TRAIN_ID" \
                  --array="$ARRAY" eval_lmms_level_hipster.sh "$CKPT")
echo "  eval job    : $EVAL_ID  (afterok:$TRAIN_ID)"
echo "  checkpoint  : $CKPT"

# jobs/ is gitignored -- this ledger is hipster-local bookkeeping and never
# reaches git/GitHub, deliberately separate from docs/SUBMITTED_RUNS.tsv
# (DAS-6's tracked ledger, not touched from this cluster).
LOG=jobs/SUBMITTED_RUNS_HIPSTER.tsv
mkdir -p jobs
[ -s "$LOG" ] || printf 'submitted_utc\trecipe\tslm_key\tpartition\ttrain_job\teval_job\tcheckpoint\n' > "$LOG"
printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$ELASTIC_RUN_TAG" "$SLM_KEY" "$PARTITION" \
    "$TRAIN_ID" "$EVAL_ID" "$CKPT" >> "$LOG"
echo "  recorded in : $LOG"
