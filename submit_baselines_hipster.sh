#!/bin/bash
# Submit the M3 / MQT 4-token baselines on hipster, separately.
#
#   bash submit_baselines_hipster.sh m3   [performance|capacity]
#   bash submit_baselines_hipster.sh mqt  [performance|capacity]
#   DRY_RUN=1 bash submit_baselines_hipster.sh m3
#
# Mirrors submit_elastic_run_hipster.sh: same partition/GRES selection, same
# no---exclusive policy, same hipster-local ledger. One experiment per
# invocation, deliberately -- they are independent baselines and staging them
# separately keeps a failure in one from taking the other down.
#
# ---------------------------------------------------------------------------
# WHAT THESE BASELINES ARE
#   m3   this repo IS an M3 fork with M3 intact underneath. Stage 2 with
#        MATRYOSHKA_SCALE set takes LlavaElasticMixin.forward's
#        "Pure M3 (explicit scale list)" branch and avg-pools
#        (pool = stride = sqrt(576/scale)); scale 4 -> 12x12 -> 4 tokens.
#   mqt  MQT-LLaVA's own train loop and query_abstractor resampler, run from
#        MQT-LLaVA/ with a PYTHONPATH guard.
#
# SCOPE, for the paper: both run at a FIXED 4-token budget, which narrows each
# method (M3 normally trains a scale LIST jointly; MQT normally samples a
# random query count per step). Report as "M3/MQT architecture + train loop at
# a fixed 4-token budget", NOT as M3/MQT headline numbers.
#
# VALIDATION STATUS carried over from DAS-6 (docs/EXPERIMENT_JOURNAL.md 19f):
#   M3  Stage 1 + Stage 2 both PASSED 20 real steps (job 27418).
#   MQT Stage 1 PASSED 20 real steps (27421); forward+backward @4 tokens
#       verified under transformers 4.44.2 (27424). MQT Stage 2 has NEVER
#       completed a full training step -- it OOM'd on a single 24GB A10 during
#       DeepSpeed optimizer-state init, before step 1. On 48GB RTX 6000 Ada
#       with 4-way ZeRO-2 sharding that is not expected to recur, but the first
#       real MQT Stage 2 anywhere will be this run. Watch the first ~200 steps.
# ---------------------------------------------------------------------------
set -uo pipefail
cd "$(dirname "$0")"

EXP=${1:-}
PARTITION_ARG=${2:-}
SLM_KEY="${SLM_KEY:-tinyllama}"

# 4 GPUs for training. per_device * accum * num_gpus is held constant by the
# stage scripts, so NUM_GPUS changes wall-clock, not the effective batch.
export NUM_GPUS="${NUM_GPUS:-4}"
export BASELINE_RUN_TAG="${BASELINE_RUN_TAG:-4tok}"

case "$EXP" in
  m3)
    JOB_SCRIPT=run_job_hipster_m3_baseline.sh
    export MATRYOSHKA_SCALE="${MATRYOSHKA_SCALE:-4}"
    CKPT_NAME="baseline-${SLM_KEY}-${BASELINE_RUN_TAG}-finetune"
    DESC="M3 avg-pool @ scale ${MATRYOSHKA_SCALE}"
    ;;
  mqt)
    JOB_SCRIPT=run_job_hipster_mqt_baseline.sh
    export NUM_VISUAL_TOKENS="${NUM_VISUAL_TOKENS:-4}"
    CKPT_NAME="mqt-finetune-${SLM_KEY}-${BASELINE_RUN_TAG}"
    DESC="MQT query_abstractor @ ${NUM_VISUAL_TOKENS} queries"
    ;;
  *)
    echo "Usage: bash submit_baselines_hipster.sh {m3|mqt} [performance|capacity]" >&2
    exit 1 ;;
esac

# Default to the RTX 6000 Ada partition: Stage 2 is a full LLM finetune and the
# only time it has been attempted on a 24GB card it OOM'd in optimizer-state
# init. 48GB removes that risk outright.
PARTITION="${PARTITION_ARG:-performance}"
case "$PARTITION" in
  performance) GPU_TYPE="rtx_6000_ada"; CPUS_PER_GPU=32 ;;
  capacity)    GPU_TYPE="l4";           CPUS_PER_GPU=16 ;;
  *) echo "ERROR: PARTITION must be 'performance' or 'capacity', got '$PARTITION'" >&2; exit 1 ;;
esac
TRAIN_GRES="gpu:${GPU_TYPE}:${NUM_GPUS}"
TRAIN_CPUS=$CPUS_PER_GPU

CKPT="/scratch/skalra/flexllava_saves/checkpoints/${CKPT_NAME}"

echo "── baseline ${EXP} (${SLM_KEY}) on hipster:${PARTITION} ──────────────────"
printf '  %-24s %s\n' "experiment"      "$DESC"
printf '  %-24s %s\n' "job script"      "$JOB_SCRIPT"
printf '  %-24s %s\n' "tag"             "$BASELINE_RUN_TAG"
printf '  %-24s %s\n' "partition"       "$PARTITION"
printf '  %-24s %s\n' "train gres/cpus" "$TRAIN_GRES / $TRAIN_CPUS per task"
printf '  %-24s %s\n' "NUM_GPUS"        "$NUM_GPUS"
printf '  %-24s %s\n' "checkpoint"      "$CKPT"
[ -n "${MAX_STEPS:-}" ] && printf '  %-24s %s\n' "MAX_STEPS" "$MAX_STEPS (TRUNCATED -- timing probe only)"

echo
echo "  NOTE: eval is NOT wired for these checkpoints -- eval_lmms_level_hipster.sh"
echo "        reads elastic_config.json, which a non-elastic baseline does not have."
echo "        No eval job is submitted; evaluate separately."

if [ -n "${DRY_RUN:-}" ]; then echo; echo "DRY_RUN set; nothing submitted."; exit 0; fi

TRAIN_ID=$(sbatch --parsable --partition="$PARTITION" --gres="$TRAIN_GRES" \
                  --ntasks-per-node="$NUM_GPUS" --cpus-per-task="$CPUS_PER_GPU" \
                  "$JOB_SCRIPT" "$SLM_KEY")
echo "  train job   : $TRAIN_ID  (Stage 1 + Stage 2)"
echo "  checkpoint  : $CKPT"

# jobs/ is gitignored -- hipster-local bookkeeping, never reaches git, kept
# separate from docs/SUBMITTED_RUNS.tsv (DAS-6's tracked ledger).
LOG=jobs/SUBMITTED_RUNS_HIPSTER.tsv
mkdir -p jobs
[ -s "$LOG" ] || printf 'submitted_utc\trecipe\tslm_key\tpartition\ttrain_job\teval_job\tcheckpoint\n' > "$LOG"
printf '%s\tbaseline-%s-%s\t%s\t%s\t%s\t-\t%s\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$EXP" "$BASELINE_RUN_TAG" "$SLM_KEY" \
    "$PARTITION" "$TRAIN_ID" "$CKPT" >> "$LOG"
echo "  recorded in : $LOG"
