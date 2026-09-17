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
  v8-siglip-parcel)
    # True v8/final-parcel (see submit_elastic_run.sh's final-parcel case),
    # verbatim, with SigLIP swapped in for CLIP. google/siglip-base-patch16-384
    # per docs/EXPERIMENT_JOURNAL.md section 10b: 384/16=24 gives the identical
    # 576-patch grid as CLIP-L/14-336, so the token ladder and LoRA ranks
    # transfer with zero changes. USE_KD deliberately left unset here (defaults
    # True in train_elastic.py) to match v8's own default -- prefix-KL
    # self-distillation with TEACHER=self (set globally above), not off.
    # Smoke-tested on DAS-6 for 3 steps (TinyLlama only, job 27326) but never
    # run to completion on any cluster; SmolLM2+SigLIP has no test history at
    # all before this run.
    export ELASTIC_RUN_TAG=v8-siglip-parcel
    export LORA_RANKS="8 16 32 64"
    export STAGE1_LORA_RANK=64
    export VISION_LORA_ENABLE=True
    export VISION_LORA_SPECIALIZE_TOK=True
    export USE_TOKEN_DECORRELATION=False
    export VISION_TOWER="google/siglip-base-patch16-384"
    ;;
  v8-siglip-so400m-parcel)
    # True v8/final-parcel (see submit_elastic_run.sh's final-parcel case),
    # with SigLIP so400m-patch14-384 in place of CLIP -- the larger of the two
    # SigLIP checkpoints (428M vision params, 1.41x CLIP-L's 303M; the sibling
    # v8-siglip-parcel above uses siglip-base-patch16-384, 0.31x CLIP-L, a
    # SMALLER tower, not a scaled variant of the same model).
    #
    # so400m's 384/14=27.43 grid rounds to 27x27=729 patches. 729=3^6, so its
    # only integer-pooling anchor counts are {1,9,81,729} vs CLIP's 576-patch
    # {1,4,9,16,36,64,144,576} -- anchor_mode=ratio's automatic 25% target
    # therefore snaps DOWN-only and gets stuck at 9 anchors (3.5%) at the
    # 256-token level specifically (round(256*0.25)=64 is unreachable, and the
    # snap-down search can never find 81, which sits ABOVE that target but is
    # the closest reachable value overall). The other three levels (144/64/16)
    # were already at their best reachable value under plain ratio mode.
    # anchor_mode=fixed with the table below is the corrected version -- table
    # derived two independent ways (closest to the 25% target; largest
    # reachable count that stays a minority of the budget) that agree exactly,
    # and validated against n_anchors_for() directly, not by hand (DAS-6 job
    # 27464). Smoke-tested end to end on DAS-6 (job 27465, TinyLlama, both
    # stages, `anchor_mode`/`anchor_routing` confirmed round-tripped correctly
    # into the saved checkpoint's own elastic_config.json) before this recipe
    # was written.
    export ELASTIC_RUN_TAG=v8-siglip-so400m-parcel
    export LORA_RANKS="8 16 32 64"
    export STAGE1_LORA_RANK=64
    export VISION_LORA_ENABLE=True
    export VISION_LORA_SPECIALIZE_TOK=True
    export USE_TOKEN_DECORRELATION=False
    export VISION_TOWER="google/siglip-so400m-patch14-384"
    export ANCHOR_MODE=fixed
    export ANCHOR_ROUTING="256:81,144:9,64:9,16:1"
    # so400m is 428M params, 1.41x CLIP-L. No partition restriction for
    # SmolLM2 here -- capacity's 4x L4 per job is what actually applies, and
    # was never tested against this combination. What WAS tested, on DAS-6:
    # SmolLM2+so400m OOM'd in Stage 2 backward with ZeRO-2 sharding across only
    # 2x A10 (job 27461, 22.28/22.30GB used, failing on a 22MB marginal
    # allocation -- essentially at the wire, not wildly over budget), and
    # passed on a single UNSHARDED A40 (job 27463, 46GB). Neither config used
    # 4-way sharding. capacity's 4 GPUs is double the sharding of the config
    # that failed, and ZeRO-2 shards optimizer state + gradients (though not
    # weights or per-GPU activations, which don't shrink with more ranks) --
    # plausible this clears the thin margin above, not confirmed at 4-GPU
    # scale. Removed the refusal on request (more GPUs available on hipster's
    # capacity than were tested); if it OOMs again, that is new information
    # (4-way still insufficient), not a repeat of the known 2-way failure.
    ;;
  v8-siglip-so400m-adaptive-parcel)
    # Same as v8-siglip-so400m-parcel (true v8/final-parcel + SigLIP
    # so400m-patch14-384), but anchors via the NEW anchor_mode=adaptive fork
    # (llava/model/elastic/resampler.py's pool_anchors_adaptive, merged from
    # main) instead of anchor_mode=fixed's snapped table.
    #
    # v8-siglip-so400m-parcel's fixed table (256:81,144:9,64:9,16:1) exists
    # ONLY because plain avg-pooling anchor_mode=fixed/ratio requires an
    # anchor count that evenly divides so400m's 27x27=729-patch grid via
    # integer stride -- so400m's only such counts are {1,9,81,729} (729=3^6),
    # nowhere near a clean 25% at every budget (81/256=31.6%, not 25%).
    # anchor_mode=adaptive removes that constraint entirely: pool_anchors_
    # adaptive() uses F.adaptive_avg_pool2d, which accepts ANY anchor count in
    # [1, P], so the exact 25% target is directly reachable at every level --
    # no snapping, no approximation:
    #   budget   25% target   adaptive anchors (exact)
    #   256      64           64
    #   144      36           36
    #   64       16           16
    #   16       4            4
    # This is the same anchor_routing table CLIP's 576-patch grid gets under
    # plain ratio mode (64/36/16/4 IS v8's original fixed table, see decision
    # 35/submit_elastic_run.sh's final-parcel case) -- adaptive mode lets
    # so400m match that exact split despite its grid not factoring the same
    # way CLIP's does.
    export ELASTIC_RUN_TAG=v8-siglip-so400m-adaptive-parcel
    export LORA_RANKS="8 16 32 64"
    export STAGE1_LORA_RANK=64
    export VISION_LORA_ENABLE=True
    export VISION_LORA_SPECIALIZE_TOK=True
    export USE_TOKEN_DECORRELATION=False
    export VISION_TOWER="google/siglip-so400m-patch14-384"
    export ANCHOR_MODE=adaptive
    export ANCHOR_ROUTING="256:64,144:36,64:16,16:4"
    # Same untested-at-4-GPU OOM caveat as v8-siglip-so400m-parcel above
    # applies to SmolLM2 with this recipe too (same backbone+tower memory
    # footprint, anchor_mode doesn't change parameter count) -- not restricted
    # here either, for the same reason.
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
    echo "Usage: bash submit_elastic_run_hipster.sh {v11-parcel-nolora|v12-parcel-kd7b|v8-siglip-parcel|v8-siglip-so400m-parcel|v8-siglip-so400m-adaptive-parcel} {tinyllama|smollm2|mobilellama} [performance|capacity]" >&2
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
         ANCHOR_MODE ANCHOR_RATIO ANCHOR_ROUTING USE_TOKEN_DECORRELATION DECORR_WEIGHT \
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
