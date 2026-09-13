#!/bin/bash
# Hipster port of run_matrix_hipster.sh (DAS-6, commit 8d9369e). That script
# is unusable on hipster as-is: it hardcodes /var/scratch (DAS-6-only) and
# calls the DAS-6-native launchers (run_job_pretrain_slm.sh, run_job_
# finetune_slm.sh, eval_lmms_level.sh -- all still reference /var/scratch
# internally, confirmed 2026-09-14). This script reuses the SAME submit_one
# logic (identical env var names, identical tag construction, identical
# recipe values) but targets:
#   - hipster paths (/scratch/skalra/flexllava_saves/checkpoints/...)
#   - hipster launchers (run_job_pretrain_only_hipster.sh, run_job_hipster_
#     finetune_only_das6.sh, eval_lmms_level_hipster.sh)
#   - hipster's typed GPU/partition scheme, not DAS-6's untyped gpu:2
#   - DAS-6 data streaming (per the 2026-09-14 decision to use it for all
#     experiments going forward), which is already built into those two
#     launchers -- this script does not touch data staging directly.
#
# Does NOT change: recipe values, KD semantics, tag strings, Stage-1-sharing
# logic, or anything llava/train/train_elastic.py consumes. Same axis, same
# meaning, same checkpoint-naming convention as the DAS-6 original -- only
# cluster mechanics differ. See docs/EXPERIMENT_JOURNAL.md sections 16o/17/18.
#
#   bash submit_matrix_hipster.sh                    # print, submit nothing
#   SUBMIT=1 bash submit_matrix_hipster.sh            # actually submit
#
# UNLIKE the DAS-6 original, this script has NO default matrix -- ONLY is
# unset by default. The DAS-6 script's ONLY=<backbone> selects a whole
# backbone (still 4 cells); it has no way to select individual (backbone,
# lora, kd, teacher) cells, which is what was asked for here (4 specific
# cells across 2 backbones, not both backbones' full cell sets). Added below:
#
#   CELLS="smollm2:m-asclora-kdoff smollm2:m-v8lora-kdon
#          mobilellama:m-v8lora-kdoff mobilellama:m-v8lora-kd7b"
#
# is the exact, hardcoded default -- the four experiments requested
# 2026-09-14 (SmolLM2 V14, SmolLM2 V8+self-distill, MobileLLaMA V8,
# MobileLLaMA V8+7B KD). Override CELLS= to select a different subset; there
# is deliberately no "run everything" default, unlike the DAS-6 script.
set -uo pipefail
cd "$(dirname "$0")"

SUBMIT="${SUBMIT:-}"
CELLS="${CELLS:-smollm2:m-asclora-kdoff smollm2:m-v8lora-kdon mobilellama:m-v8lora-kdoff mobilellama:m-v8lora-kd7b}"
LOG=jobs/SUBMITTED_RUNS_HIPSTER.tsv

# Partition/GRES: hipster requires typed GPU names, unlike DAS-6's untyped
# gpu:2 (which relies on a queue where only 2-GPU nodes exist). capacity=l4
# (24GB, fine for self-distillation/no-KD cells) matches GRES's DAS-6 role;
# performance=rtx_6000_ada (48GB) is the KD_GRES role -- the external 7B
# teacher needs ~14GB/GPU unsharded on top of training, which does not fit
# an L4's 24GB (same reasoning DAS-6 used for its own A40-class KD_GRES).
GRES_PARTITION="${GRES_PARTITION:-capacity}"
KD_GRES_PARTITION="${KD_GRES_PARTITION:-performance}"
case "$GRES_PARTITION" in
  capacity)    GRES_TYPE=l4 ;;
  performance) GRES_TYPE=rtx_6000_ada ;;
  *) echo "ERROR: GRES_PARTITION must be capacity|performance" >&2; exit 1 ;;
esac
case "$KD_GRES_PARTITION" in
  capacity)    KD_GRES_TYPE=l4 ;;
  performance) KD_GRES_TYPE=rtx_6000_ada ;;
  *) echo "ERROR: KD_GRES_PARTITION must be capacity|performance" >&2; exit 1 ;;
esac
NUM_GPUS=2
CPUS_PER_GPU_CAPACITY=16
CPUS_PER_GPU_PERFORMANCE=32

echo "=================================================================="
echo " v8-vs-v14 matrix, hipster (exact-cell subset)"
echo "   cells     : $CELLS"
echo "   data      : DAS-6 streaming (run_job_pretrain_only_hipster.sh /"
echo "               run_job_hipster_finetune_only_das6.sh)"
echo "=================================================================="

declare -A STAGE1_JOB

cpus_for() { [ "$1" = capacity ] && echo "$CPUS_PER_GPU_CAPACITY" || echo "$CPUS_PER_GPU_PERFORMANCE"; }

submit_one() {
    local key="$1" lora="$2" kd="$3" teacher="$4" tag="$5" partition="$6" gres_type="$7"
    local nest; [ "$lora" = "v8" ] && nest=v8 || nest=v14
    local ck=/scratch/skalra/flexllava_saves/checkpoints/elastic-finetune-${key}-${tag}
    local gres="gpu:${gres_type}:${NUM_GPUS}"
    local cpus_per_gpu; cpus_per_gpu=$(cpus_for "$partition")
    printf '  %-12s lora=%-4s kd=%-3s teacher=%-8s partition=%-11s -> %s\n' \
        "$key" "$lora" "$kd" "$teacher" "$partition" "$tag"
    [ -z "$SUBMIT" ] && return 0

    # Stage 1 shared per backbone (§17c) -- identical caching logic to the
    # DAS-6 script, hipster paths and launcher substituted.
    local pre_tag="matrixbase"
    local pre=/scratch/skalra/flexllava_saves/checkpoints/elastic-pretrain-${key}-${pre_tag}
    local dep=""
    if [ ! -f "${pre}/elastic_config.json" ]; then
        # Point at an already-submitted Stage-1 job instead of deriving a new
        # one -- e.g. STAGE1_JOB_OVERRIDE_smollm2=357361 -- so a resubmission
        # (after fixing an unrelated bug in the Stage-2 call, say) does not
        # spawn a duplicate Stage 1 for a backbone whose shared checkpoint is
        # already being produced by a job submitted moments earlier.
        local override_var="STAGE1_JOB_OVERRIDE_${key}"
        if [ -n "${!override_var:-}" ] && [ -z "${STAGE1_JOB[$key]:-}" ]; then
            STAGE1_JOB[$key]="${!override_var}"
            echo "      stage1 (shared)  : ${STAGE1_JOB[$key]}  (override, not resubmitted)"
        fi
        if [ -z "${STAGE1_JOB[$key]:-}" ]; then
            # NEST_VERSION deliberately NOT passed here: this Stage-1 checkpoint
            # is SHARED across both lora_type=v8 and lora_type=asc downstream
            # cells (§17c), so tagging it with whichever cell happened to
            # trigger its derivation first would misrecord a shared artifact as
            # version-specific. It is also load-bearing, not cosmetic: for a
            # 1-level tok_levels list, train_elastic.py's ladder formula
            # (8*2^i) derives rank [8] for v8 AND for asc (reversing a 1-
            # element list is a no-op) -- passing --nest_version here while
            # also passing the required --lora_ranks 64 (Stage-1 buffer must
            # match Stage 2's max rank, not Stage 1's own single level) trips
            # train_elastic.py's new lora_ranks/lora_type contradiction check
            # and fails immediately. Found 2026-09-14 (jobs 357361/357364);
            # the same bug exists in DAS-6's run_matrix_hipster.sh, unexercised
            # there because nothing in the matrix has actually been submitted
            # yet (journal table 18d, all rows PLANNED).
            STAGE1_JOB[$key]=$(ELASTIC_RUN_TAG=$pre_tag \
                STAGE1_TOK_LEVEL=256 STAGE1_LORA_RANK=64 \
                RESAMPLER_ARCH=pool_anchored ANCHOR_MODE=ratio ANCHOR_RATIO=0.25 \
                VISION_LORA_ENABLE=True VISION_LORA_SPECIALIZE_TOK=True \
                sbatch --parsable --partition="$GRES_PARTITION" --gres="gpu:${GRES_TYPE}:${NUM_GPUS}" \
                --ntasks-per-node="$NUM_GPUS" --cpus-per-task="$(cpus_for "$GRES_PARTITION")" \
                run_job_pretrain_only_hipster.sh "$key")
            echo "      stage1 (shared)  : ${STAGE1_JOB[$key]}"
        fi
        dep="--dependency=afterok:${STAGE1_JOB[$key]}"
    fi

    # BUG FOUND 2026-09-14, fixed here: the DAS-6 original (run_matrix_
    # hipster.sh line 130) builds this as
    #   $( [ "$teacher" = self ] && echo "TEACHER=self" || echo "KD_TEACHER=$teacher" ) sbatch ...
    # -- but bash only recognizes a literal `VAR=value cmd` prefix at PARSE
    # TIME; the *output* of a command substitution is just an argument word,
    # not a variable assignment. This makes bash try to execute a command
    # literally named "TEACHER=self", which fails ("command not found"), so
    # `sbatch` is never actually invoked and $jid captures an empty string.
    # Confirmed 2026-09-14 (all 4 real submission attempts failed this way);
    # per journal table 18d, run_matrix_hipster.sh had only ever been
    # dry-run before, which never reaches this line (submit_one returns
    # early when SUBMIT is unset) -- so this bug exists on DAS-6 too and has
    # never been caught there. Fixed here with explicit export instead.
    local jid
    if [ "$teacher" = self ]; then
        export TEACHER=self; unset KD_TEACHER
    else
        export KD_TEACHER="$teacher"; unset TEACHER
    fi
    jid=$(ELASTIC_RUN_TAG=$tag ELASTIC_PRETRAIN_TAG=$pre_tag \
        NEST_VERSION=$nest LORA_TYPE=$lora \
        TOK_LEVELS="256 144 64 16" \
        RESAMPLER_ARCH=pool_anchored ANCHOR_MODE=ratio ANCHOR_RATIO=0.25 \
        VISION_LORA_ENABLE=True VISION_LORA_SPECIALIZE_TOK=True \
        USE_TOKEN_DECORRELATION=False \
        USE_KD=$kd KD_STUDENT_KEY=$key \
        sbatch --parsable $dep --partition="$partition" --gres="$gres" \
        --ntasks-per-node="$NUM_GPUS" --cpus-per-task="$cpus_per_gpu" \
        run_job_hipster_finetune_only_das6.sh "$key")
    if [ -z "$jid" ]; then
        echo "ERROR: sbatch did not return a job id for $key/$tag -- not submitting its eval." >&2
        return 1
    fi
    local ejid
    ejid=$(sbatch --parsable --partition="$partition" --gres="gpu:${gres_type}:1" \
        --cpus-per-task="$cpus_per_gpu" --dependency=afterok:"$jid" --array=0-3 \
        eval_lmms_level_hipster.sh "$ck")
    echo "      train $jid  eval $ejid"
    mkdir -p jobs
    [ -s "$LOG" ] || printf 'submitted_utc\trecipe\tslm_key\tpartition\ttrain_job\teval_job\tcheckpoint\n' > "$LOG"
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$tag" "$key" "$partition" "$jid" "$ejid" "$ck" >> "$LOG"
}

for cell in $CELLS; do
    key="${cell%%:*}"
    tag="${cell#*:}"
    case "$tag" in
        m-v8lora-kdoff)   lora=v8;  kd=False; teacher=self;    partition="$GRES_PARTITION";    gres_type="$GRES_TYPE" ;;
        m-v8lora-kdon)    lora=v8;  kd=True;  teacher=self;    partition="$GRES_PARTITION";    gres_type="$GRES_TYPE" ;;
        m-asclora-kdoff)  lora=asc; kd=False; teacher=self;    partition="$GRES_PARTITION";    gres_type="$GRES_TYPE" ;;
        m-asclora-kdon)   lora=asc; kd=True;  teacher=self;    partition="$GRES_PARTITION";    gres_type="$GRES_TYPE" ;;
        m-v8lora-kd7b)    lora=v8;  kd=True;  teacher=llava7b; partition="$KD_GRES_PARTITION";  gres_type="$KD_GRES_TYPE" ;;
        m-asclora-kd7b)   lora=asc; kd=True;  teacher=llava7b; partition="$KD_GRES_PARTITION";  gres_type="$KD_GRES_TYPE" ;;
        *) echo "ERROR: unknown tag '$tag' in CELLS -- must be one of m-{v8,asc}lora-kd{off,on,7b}" >&2; exit 1 ;;
    esac
    submit_one "$key" "$lora" "$kd" "$teacher" "$tag" "$partition" "$gres_type"
done

echo
[ -z "$SUBMIT" ] && echo "DRY RUN -- nothing submitted. Re-run with SUBMIT=1." \
    || echo "Submitted. Recorded in $LOG."
