#!/bin/bash
# Controlled v8-vs-v14 experiment matrix for the Hipster cluster.
#
#   bash run_matrix_hipster.sh            # print the matrix, submit nothing
#   SUBMIT=1 bash run_matrix_hipster.sh   # actually submit
#   SUBMIT=1 ONLY=smollm2 bash run_matrix_hipster.sh
#
# Requires (override for hipster's scheduler):
#   GRES        GPU spec for a 2-GPU training job          (default gpu:2)
#   KD_GRES     GPU spec for external-teacher KD runs      (default gpu:A40:2)
#
# ---------------------------------------------------------------------------
# WHY THIS IS 12 RUNS AND NOT 24
# ---------------------------------------------------------------------------
# The requested matrix was 2 versions x 3 backbones x 2 LoRA x 2 KD = 24. An
# audit of the two recipes (docs/EXPERIMENT_JOURNAL.md 16o) found that v8 and
# v14 differ in EXACTLY ONE thing: the lora_ranks ordering. Every other
# component -- resampler, anchor split, token budgets, decorrelation, CORAL,
# KD, optimizer, LR, schedule, epochs, batch, data, augmentation, frozen
# parameters, Stage-1 warm start, eval protocol -- is identical.
#
# So VERSION and LORA are the same axis, and the 2x2 has only 2 distinct
# configurations:
#     nest_version=v8  + lora_type=v8   == v8      <- real
#     nest_version=v14 + lora_type=asc  == v14     <- real
#     nest_version=v8  + lora_type=asc  == v14     <- duplicate
#     nest_version=v14 + lora_type=v8   == v8      <- duplicate
# Submitting all four would spend ~190 GPU-hours per backbone re-deriving
# checkpoints that differ only in a metadata string. The matrix below therefore
# varies lora_type (the thing that actually changes the model) and records
# nest_version alongside it. Set WITH_DUPLICATES=1 to submit the redundant
# cells anyway as a null-control -- they should land within eval noise of their
# twins, which is a real (if expensive) check on run-to-run variance.
#
# ---------------------------------------------------------------------------
# WHAT "KD" MEANS HERE -- READ BEFORE INTERPRETING ANY KD RESULT
# ---------------------------------------------------------------------------
# This codebase has TWO distinct things both called KD; they are not
# interchangeable and only one of them is available on all three backbones.
#
#  (a) --use_kd True|False  ->  cfg.use_prefix_kl
#      SELF-distillation across token budgets. The "teacher" is THIS MODEL at
#      tok_levels[0] (256), its logits detached; the students are the smaller
#      budgets. Teacher and student SHARE WEIGHTS EXACTLY -- it is one network
#      compared against itself on a longer visual prefix. Measured KL ~0.006,
#      i.e. ~0.01% of the loss. This is what the KD axis below toggles, and it
#      works on every backbone.
#      It is NOT an independent-teacher comparison. Do not report it as one.
#
#  (b) --teacher llava  ->  a frozen external LLaVA-1.5-7B, independent weights,
#      real KL signal. This is genuine KD. It requires a matching tokenizer:
#      attach_kd_teacher raises if teacher and student vocab_size differ.
#          TinyLlama    32000  OK
#          MobileLLaMA  32000  OK
#          SmolLM2      49152  IMPOSSIBLE
#          Qwen2.5     151936  IMPOSSIBLE
#      So external-teacher KD CANNOT be run on SmolLM2 or Qwen with the teacher
#      this repo has. It is added below for MobileLLaMA only, as a clearly
#      separate arm -- not as a cell of the main matrix, because a matrix whose
#      "KD" column means self-distillation for two backbones and external-teacher
#      KD for a third would be uninterpretable.
#      Prior evidence: v12 (external KD, TinyLlama) lost to its no-KD control on
#      every metric at every budget (16l).
# ---------------------------------------------------------------------------
set -uo pipefail
cd "$(dirname "$0")"

GRES="${GRES:-gpu:2}"
KD_GRES="${KD_GRES:-gpu:A40:2}"      # external teacher adds ~14 GB/GPU, unsharded
SUBMIT="${SUBMIT:-}"
ONLY="${ONLY:-}"
WITH_DUPLICATES="${WITH_DUPLICATES:-}"
LOG=docs/SUBMITTED_RUNS.tsv

# Qwen: the repo defines qwen0.5b, qwen1.5b and qwen3b. qwen0.5b is BOTH the
# launcher's own default (finetune_elastic_slm.sh) and the first Qwen row of the
# planned matrix in EXPERIMENT_JOURNAL 15e. Using that rather than silently
# picking a different size; override with QWEN_KEY if the paper wants another.
QWEN_KEY="${QWEN_KEY:-qwen0.5b}"
BACKBONES="${BACKBONES:-smollm2 mobilellama $QWEN_KEY}"

echo "=================================================================="
echo " v8-vs-v14 controlled matrix"
echo "   backbones : $BACKBONES"
echo "   lora_type : v8 (rank ascends as budget descends) | asc (rank ascends with budget)"
echo "   KD        : off | on  == prefix-KL SELF-distillation (shared weights)"
echo "   gres      : $GRES      (external-teacher arm: $KD_GRES)"
echo "=================================================================="

submit_one() {
    local key="$1" lora="$2" kd="$3" teacher="$4" tag="$5" gres="$6"
    # teacher="self" -> self-distillation (no external model). Anything else is a
    # kd_teachers registry key, resolved and compatibility-checked at attach time;
    # resolution REFUSES rather than substituting, so an incompatible pair fails
    # at job start instead of silently training against the wrong teacher.
    local nest; [ "$lora" = "v8" ] && nest=v8 || nest=v14
    local ck=/var/scratch/skalra/flexllava/checkpoints/elastic-finetune-${key}-${tag}
    printf '  %-12s lora=%-4s kd=%-3s teacher=%-5s -> %s\n' "$key" "$lora" "$kd" "$teacher" "$tag"
    [ -z "$SUBMIT" ] && return 0

    # Stage 1 is shared across lora_type AND kd for a given backbone: it trains a
    # single tok_level (256) at a single rank (64), so the ladder ordering cannot
    # apply, and with one level there are no students, which makes the prefix-KL
    # term structurally inert. Deriving it once per backbone instead of once per
    # cell saves 9 x ~10h across the matrix. ELASTIC_PRETRAIN_TAG points every
    # Stage 2 at that shared checkpoint.
    local pre_tag="matrixbase"
    local pre=/var/scratch/skalra/flexllava/checkpoints/elastic-pretrain-${key}-${pre_tag}
    local dep=""
    if [ ! -f "${pre}/elastic_config.json" ]; then
        if [ -z "${STAGE1_JOB[$key]:-}" ]; then
            STAGE1_JOB[$key]=$(ELASTIC_RUN_TAG=$pre_tag NEST_VERSION=$nest \
                STAGE1_TOK_LEVEL=256 STAGE1_LORA_RANK=64 \
                RESAMPLER_ARCH=pool_anchored ANCHOR_MODE=ratio ANCHOR_RATIO=0.25 \
                VISION_LORA_ENABLE=True VISION_LORA_SPECIALIZE_TOK=True \
                sbatch --parsable --gres="$gres" run_job_pretrain_slm.sh "$key")
            echo "      stage1 (shared)  : ${STAGE1_JOB[$key]}"
        fi
        dep="--dependency=afterok:${STAGE1_JOB[$key]}"
    fi

    local jid
    jid=$(ELASTIC_RUN_TAG=$tag ELASTIC_PRETRAIN_TAG=$pre_tag \
        NEST_VERSION=$nest LORA_TYPE=$lora \
        TOK_LEVELS="256 144 64 16" \
        RESAMPLER_ARCH=pool_anchored ANCHOR_MODE=ratio ANCHOR_RATIO=0.25 \
        VISION_LORA_ENABLE=True VISION_LORA_SPECIALIZE_TOK=True \
        USE_TOKEN_DECORRELATION=False \
        USE_KD=$kd KD_STUDENT_KEY=$key \
        $( [ "$teacher" = self ] && echo "TEACHER=self" || echo "KD_TEACHER=$teacher" ) \
        sbatch --parsable $dep --gres="$gres" run_job_finetune_slm.sh "$key")
    local ejid
    ejid=$(sbatch --parsable --dependency=afterok:"$jid" --array=0-3 \
        eval_lmms_level.sh "$ck")
    echo "      train $jid  eval $ejid"
    printf '%s\t%s\t%s\t%s\t%s\t256 144 64 16\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$tag" "$key" "$jid" "$ejid" \
        "lora_type=$lora" "pool_anchored" "kd=$kd" "teacher=$teacher" \
        "nest_version=$nest" "$ck" >> "$LOG"
}

declare -A STAGE1_JOB
echo; echo "--- core matrix: 3 backbones x 2 lora_type x 2 KD = 12 runs ---"
for key in $BACKBONES; do
    [ -n "$ONLY" ] && [ "$key" != "$ONLY" ] && continue
    for lora in v8 asc; do
        for kd in True False; do
            k=off; [ "$kd" = "True" ] && k=on
            submit_one "$key" "$lora" "$kd" self "m-${lora}lora-kd${k}" "$GRES"
        done
    done
done

if [ -n "$WITH_DUPLICATES" ]; then
    echo; echo "--- null-control duplicates (same weights, different metadata) ---"
    for key in $BACKBONES; do
        [ -n "$ONLY" ] && [ "$key" != "$ONLY" ] && continue
        submit_one "$key" asc True self "m-dup-v8label-asclora" "$GRES"
        submit_one "$key" v8  True self "m-dup-v14label-v8lora" "$GRES"
    done
fi

echo; echo "--- external-teacher KD arm: resolved per backbone by the registry ---"
echo "    Teacher compatibility (llava/model/elastic/kd_teachers.py, audited jobs"
echo "    27410/27411/27413; report: results/kd_compatibility_report.json):"
echo "      tinyllama    -> llava7b        DIRECT      (token IDs byte-identical)"
echo "      mobilellama  -> llava7b        DIRECT      (token IDs byte-identical)"
echo "      smollm2      -> none           VOCAB_MAPPING_REQUIRED / INCOMPATIBLE"
echo "      qwen*        -> none           VOCAB_MAPPING_REQUIRED, no family VLM present"
echo "    SmolLM2 and Qwen therefore get NO external-teacher arm: the only runnable"
echo "    teacher has a 32000-token vocabulary against their 49152 / 151936, and"
echo "    prefix_kl_loss reduces over the vocab axis elementwise. MobileVLM_V2 (the"
echo "    family match for MobileLLaMA) fails to load: 'Unknown projector type:"
echo "    ldpnetv2'. SmolVLM fails on architecture AND tokenizer AND vocab."
for lora in v8 asc; do
    if [ -z "$ONLY" ] || [ "$ONLY" = "mobilellama" ]; then
        submit_one mobilellama "$lora" True llava7b "m-${lora}lora-kd7b" "$KD_GRES"
    fi
done

echo
echo "Total: 12 core (self-distillation KD axis) + 2 external-teacher = 14 runs."
echo "Cost:  each is Stage 2 ~85h on 2 GPUs, plus 3 shared Stage-1 runs ~10h each."
echo "       ~1200 GPU-pair-hours. Stage every backbone separately unless hipster"
echo "       can run many concurrently."
[ -z "$SUBMIT" ] && echo && echo "DRY RUN -- nothing submitted. Re-run with SUBMIT=1."
