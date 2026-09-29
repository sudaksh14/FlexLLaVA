#!/bin/bash
# Queue the follow-up experiments behind the budget-matched M3 / MQT-LLaVA work.
#
#   GATE_MQT=<job spec> GATE_M3=<job spec> bash queue_followup_pipeline.sh
#
# e.g. GATE_MQT=afterok:<mqt eval array> GATE_M3=afterok:<m3 eval array>. Two lanes, one per
# 2-GPU node, each starting once the item-1 half that frees its node has finished:
#   lane A (A40:2 -> node205): rank vs LM scale on the Qwen2.5 family (query-only resampler)
#   lane B (any 2 GPUs -> node208 first): positional-embedding controls, faithful PARCEL,
#                                         two seed replicates of final-parcel
# Within a lane each run waits for the previous one to END (afterany: a crashed run must not
# strand the rest). Every run gets its eval array from submit_elastic_run.sh (afterok on its
# training); query-only and PARCEL runs also get a rank measurement (afterok on training).
set -euo pipefail
cd "$(dirname "$0")"
: "${GATE_MQT:?}" "${GATE_M3:?}"

queue() {   # <dependency spec> <recipe> [ENV=VAL ...]  -> echoes the training job id
    local dep="$1" recipe="$2"; shift 2
    local out
    out=$(env "$@" DEPENDENCY="$dep" bash submit_elastic_run.sh "$recipe")
    echo "$out" >&2
    local train ckpt
    train=$(sed -n 's/.*train job *: *\([0-9]*\).*/\1/p' <<<"$out")
    ckpt=$(sed -n 's/.*checkpoint *: *\(.*\)$/\1/p' <<<"$out")
    if [ -n "${RANK:-}" ]; then
        local tag; tag="$(basename "$ckpt")"
        sbatch --parsable --dependency=afterok:"$train" jobs/run_rank_measure.sh "$ckpt" "$tag" >&2
    fi
    echo "$train"
}

echo "== lane A: rank estimate at a single budget (n=256), learned query bank, MobileLLaMA and SmolLM2"
# (An earlier version queued v4-query on qwen1.5b/0.5b/3b here; replaced 2026-09-29 by request.)
A1=$(RANK=1 queue "$GATE_MQT"       v4-query-256 SLM_KEY=mobilellama)
A2=$(RANK=1 queue "afterany:$A1"    v4-query-256 SLM_KEY=smollm2)

echo "== lane B: TinyLlama positional-embedding controls, faithful PARCEL, seed replicates"
B1=$(RANK=1 queue "$GATE_M3"        v4-query-sincos SLM_KEY=tinyllama)
B2=$(RANK=1 queue "afterany:$B1"    v4-query-nopos SLM_KEY=tinyllama)
B3=$(RANK=1 queue "afterany:$B2"    parcel-faithful SLM_KEY=tinyllama)
B4=$(RANK=  queue "afterany:$B3"    final-parcel SLM_KEY=tinyllama SEED=1)   # single seed replicate
echo "queued: lane A $A1 $A2 | lane B $B1 $B2 $B3 $B4"
