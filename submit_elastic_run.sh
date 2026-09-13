#!/bin/bash
# Submit one named elastic experiment (Stage 1 + Stage 2 + its eval array).
#
#   bash submit_elastic_run.sh v9-parcel-decorr
#   bash submit_elastic_run.sh v10-parcel-lora16
#   DRY_RUN=1 bash submit_elastic_run.sh v10-parcel-lora16    # print, don't submit
#
# WHY THIS FILE EXISTS
# --------------------
# Runs used to be launched as bare `ELASTIC_RUN_TAG=... FOO=... sbatch
# run_job_slm.sh tinyllama`, which puts the ENTIRE experiment definition in the
# submitting shell's environment. SLURM propagates it via --export=ALL but does
# not record it anywhere readable: `scontrol show job` shows only the command,
# and once the shell is gone the recipe is unrecoverable. That is exactly what
# happened to v9 (job 27376) -- reconstructing its token ladder needed the eval
# array's --array=0-7 as circumstantial evidence. Every recipe now lives here,
# in git, next to the results it produced.
#
# Each recipe is a full specification: anything NOT set falls through to the
# launcher defaults in scripts/v1_5/{pretrain,finetune}_elastic_slm.sh, so
# recipes state even the values that happen to match a default when that value
# is load-bearing for the experiment.
set -euo pipefail
cd "$(dirname "$0")"

RUN=${1:-}
SLM_KEY=${SLM_KEY:-tinyllama}

# ---- shared across every recipe below -------------------------------------
# The 4-level ladder is v8's, and it is deliberate. v6-tokrange extended it to
# 576 tokens and was a clear regression (TextVQA 14.8 at 576 tokens vs 41.0 for
# the dense 576-token baseline, and flat across 576/512/448/384) -- see
# docs/EXPERIMENT_JOURNAL.md section 16. Runs from v9 on stay on v8's grid so
# they are directly comparable to the current best model.
export TOK_LEVELS="256 144 64 16"
export STAGE1_TOK_LEVEL=256
export RESAMPLER_ARCH=pool_anchored     # PARCEL. The one change that has worked.
export ANCHOR_MODE=ratio
export ANCHOR_RATIO=0.25                # reproduces v8's anchor_routing 64/36/16/4
export VISION_TOWER="openai/clip-vit-large-patch14-336"
export TEACHER=self

# Untyped `gpu:2`, NOT `gpu:A10:2` or `gpu:A40:2`. Only node205 (A40:2) and
# node208 (A10:2) have two GPUs -- node206/207 have one each -- so gpu:2
# already resolves to exactly those two nodes, and lets SLURM place a run on
# whichever frees up first instead of queueing every run behind one node.
# Safe for comparability: nothing in the TRAINING path branches on GPU type.
# per_device_train_batch_size is a fixed constant in both stage scripts (16 /
# 2) and NUM_GPUS defaults to 2 either way, so a run is identical on A40 and
# A10 -- only wall-clock differs. (eval_lmms_level.sh DOES pick its eval batch
# size by GPU type, but that changes throughput, not scores.) Both card types
# have headroom: v8-parcel, which carries a heavier rank-64 nested adapter
# than anything here, completed its full 84.5h run on node208's A10:2.
#
# Recipes may raise this (v12 needs an A40-class card for the frozen 7B
# teacher), and `GRES=... bash submit_elastic_run.sh <recipe>` overrides both
# -- which is how these recipes get run on the hipster cluster, whose card
# names this script cannot know.
DEFAULT_GRES="gpu:2"

case "$RUN" in
  v9-parcel-decorr)
    # PARCEL + token decorrelation, vision LoRA OFF.
    # Isolates the decorrelation term (section 13) against v8 on the same grid.
    export ELASTIC_RUN_TAG=v9-parcel-decorr
    export USE_TOKEN_DECORRELATION=True
    export DECORR_WEIGHT=0.01
    export VISION_LORA_ENABLE=False
    # Inert while vision_lora_enable=False (no adapter is injected at all), but
    # train_elastic.py still requires len(lora_ranks) == len(tok_levels).
    export LORA_RANKS="8 16 32 64"
    export STAGE1_LORA_RANK=64
    ;;
  v10-parcel-lora16)
    # PARCEL + a single SHARED rank-16 vision-LoRA adapter, decorrelation OFF.
    # v8's recipe with the vision LoRA changed from rank-nested-per-level to one
    # fixed rank-16 adapter. specialize_tok=False makes ElasticConfig.
    # lora_level_for_tok always return the LAST rank index, so every tok_level
    # uses lora_ranks[-1] = 16, and NestedLoRALinear allocates its lora_A/lora_B
    # buffer at max(lora_ranks) = 16 -- one adapter, one rank, no nesting.
    # STAGE1_LORA_RANK must equal that 16 or Stage 2's warm-start hits the
    # job-27267 size mismatch on every vision-tower LoRA key.
    export ELASTIC_RUN_TAG=v10-parcel-lora16
    export USE_TOKEN_DECORRELATION=False
    export VISION_LORA_ENABLE=True
    export VISION_LORA_SPECIALIZE_TOK=False
    export LORA_RANKS="16 16 16 16"
    export STAGE1_LORA_RANK=16
    ;;
  v11-parcel-nolora)
    # PARCEL alone: decorrelation OFF, vision LoRA OFF.
    # The control v8 never had, and the reason it matters: v8 is the best model
    # in the project and it has vision LoRA ON, so "vision LoRA hurts" -- which
    # rests entirely on v4-vs-v5, both on the plain `query` resampler -- has
    # never been tested under PARCEL. v9 is the first PARCEL run without it but
    # moves decorrelation at the same time, so it cannot isolate the flag.
    # v11 vs v8 isolates vision LoRA; v11 vs v9 isolates decorrelation.
    # See docs/EXPERIMENT_JOURNAL.md sections 16d (scope limit) and 16g.
    export ELASTIC_RUN_TAG=v11-parcel-nolora
    export USE_TOKEN_DECORRELATION=False
    export VISION_LORA_ENABLE=False
    # Inert while vision_lora_enable=False, but train_elastic.py still requires
    # len(lora_ranks) == len(tok_levels). Same values as v9 so the two runs
    # differ in exactly one flag.
    export LORA_RANKS="8 16 32 64"
    export STAGE1_LORA_RANK=64
    ;;
  v12-parcel-kd7b)
    # v11 + a frozen external LLaVA-1.5-7B KD teacher. HIPSTER-PIPELINE run.
    #
    # Why it exists: `teacher=self` distills the model at tok_levels[0] into its
    # own smaller levels, and measures a KL of ~0.006 -- teacher and student are
    # literally the same weights on informationally equivalent inputs, so there
    # is nothing to distill. `teacher=llava` loads a frozen 7B instead, which
    # makes EVERY level a student (tok_levels[0] included) and gives the KL real
    # signal. That question was raised by v7-kd7b, which was cancelled mid-run by
    # the 2026-09-08 outage (section 15d); v12 supersedes it by asking the same
    # question on top of v11 instead of the superseded v4-era setup.
    #
    # Two hard constraints, both checked or noted below:
    #   1. VOCAB. attach_kd_teacher (llava/model/elastic/engine.py) raises if
    #      teacher and student vocab_size differ, so this only works on
    #      Llama-32000 backbones -- tinyllama and mobilellama. Every other
    #      backbone in the hipster matrix (smollm2, qwen*, phi2) is excluded,
    #      and the guard below fails fast rather than after hours of Stage 1.
    #   2. VRAM. The frozen 7B costs ~14 GB/GPU on top of the student and is a
    #      plain attribute on ElasticEngine, not a submodule, so ZeRO does NOT
    #      shard it. A40-class or better; it does not fit a 23 GB A10 alongside
    #      training. Hence the raised DEFAULT_GRES -- override GRES on hipster.
    case "$SLM_KEY" in
      tinyllama|mobilellama) ;;
      *)
        echo "ERROR: v12-parcel-kd7b needs a Llama-32000-vocab backbone for the" >&2
        echo "       frozen LLaVA-1.5-7B teacher; SLM_KEY='$SLM_KEY' is not one." >&2
        echo "       Valid: tinyllama, mobilellama. attach_kd_teacher would raise" >&2
        echo "       on the vocab_size mismatch after Stage 1 had already run." >&2
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
    DEFAULT_GRES="gpu:A40:2"
    echo "  NOTE: hipster-pipeline recipe. Frozen 7B teacher needs ~14 GB/GPU on"
    echo "        top of the student and is not ZeRO-sharded; verify headroom on"
    echo "        the target GPU and override GRES for that cluster's card names."
    ;;
  v13-parcel-longladder)
    # v11's flags (PARCEL, decorrelation off, vision LoRA off) on v6-tokrange's
    # 8-level 576-16 ladder instead of v8/v9/v10/v11's 4-level grid.
    #
    # Why: v6 (resampler_arch=query) showed the extended ladder is flat and, on
    # the 256-16 sub-range it shares with the short grid, LESS elastic than v4's
    # own short-grid result (docs/EXPERIMENT_JOURNAL.md section 16b). v8-parcel
    # is the one thing that has produced real elasticity on the short grid
    # (section 16c). This run asks whether PARCEL fixes the long-ladder failure
    # or whether the failure is orthogonal to the resampler architecture.
    #
    # NOT free: kl_teacher_tok_level defaults to index 0, and by convention
    # index 0 is the LARGEST tok_levels entry -- the teacher forward pass at
    # that level runs on EVERY step, unlike the single randomly sampled student
    # (n_sample_students=1) whose cost is normally ladder-length-independent.
    # Here index 0 is 576, not 256, so every step's teacher pass is ~2x the
    # v8/v9/v10/v11 cost (matches v6's own measured 1.315 vs 0.639 TFLOPs), and
    # the average sampled-student cost rises too (student pool is {512 448 384
    # 256 144 64 16}, mean ~261 tok, vs {144 64 16}, mean ~75 tok, on the short
    # grid). Stage 1 also runs at 576 tok instead of 256. Real added wall-clock
    # on top of the ~85h short-grid runtime -- not measured precisely, only
    # bounded by the ~2x FLOPs ratio; if this run needs to be time-boxed,
    # measure a few steps first rather than assuming a multiplier.
    #
    # LORA_RANKS mirrors v6's own convention exactly (ranks ascend as budget
    # descends, max stays 64) SPECIFICALLY so the four levels this ladder
    # shares with v8/v9/v10/v11 (256/144/64/16) carry the identical ranks
    # those runs use -- keeping that 4-point comparison uncontaminated by a
    # rank confound, on top of the resampler-architecture question this run
    # actually asks.
    export ELASTIC_RUN_TAG=v13-parcel-longladder
    export TOK_LEVELS="576 512 448 384 256 144 64 16"
    export LORA_RANKS="2 4 6 8 8 16 32 64"
    export STAGE1_TOK_LEVEL=576
    export STAGE1_LORA_RANK=64
    export USE_TOKEN_DECORRELATION=False
    export VISION_LORA_ENABLE=False
    # RESOLVED 2026-09-13 (section 16l, decision 33 closed): decorrelation
    # stays OFF. The clean isolation v9 (decorr on) vs hipster v11 (decorr off),
    # PARCEL + no vision LoRA in both, is negative at 256 tok -- POPE -3.12,
    # TextVQA -2.60, GQA -2.07 -- and it flattens the ladder (TextVQA spread
    # 0.44 vs 10.09). Nothing to flip; leave this run as submitted.
    ;;
  final-parcel)
    # THE FINAL PAPER RECIPE (docs/EXPERIMENT_JOURNAL.md section 16l + the
    # "Final best config" block at the end of section 16). This is v8-parcel,
    # verbatim, made backbone-parametrisable: run it with SLM_KEY=<backbone>
    # for every SLM in the paper, INCLUDING tinyllama again -- that re-run is
    # the seed/cluster replicate of v8 the paper does not yet have.
    #
    # Every choice below is the winner of a single-flag isolation on TinyLlama:
    #   PARCEL vs query        v11 vs v4   -> PARCEL creates the budget gradient
    #   nested LoRA vs none    v8  vs v11  -> +6.3 TextVQA / +2.1 GQA @256, lifts
    #                                        the whole curve; costs ~2 SciQA
    #   shared r16 vs none     v10 vs v11  -> shared adapter is WORSE than none
    #                                        at 256 tok: per-level nesting is
    #                                        the point, not adapter capacity
    #   decorr on vs off       v9  vs v11  -> negative (see v13 note above)
    #   7B KD vs self          v12 vs v11  -> negative everywhere, TextVQA -6.4
    #   CORAL                  never tested on; not introduced into the final
    #                          sweep on purpose
    #   anchor fixed vs ratio  identical at ratio 0.25 on this grid (v8's
    #                          64/36/16/4 table IS 0.25); ratio kept because it
    #                          generalises if the grid ever changes
    #
    # NOTE the launcher default VISION_LORA_ENABLE is False (flipped 2026-09-09
    # on the v4-vs-v5 result, which was measured on the plain query resampler).
    # Under PARCEL the sign reverses, so this recipe sets it True explicitly.
    # Do not rely on the launcher default for any final-sweep run.
    export ELASTIC_RUN_TAG=final-parcel
    export TOK_LEVELS="256 144 64 16"
    export LORA_RANKS="8 16 32 64"
    export STAGE1_TOK_LEVEL=256
    export STAGE1_LORA_RANK=64
    export VISION_LORA_ENABLE=True
    export VISION_LORA_SPECIALIZE_TOK=True
    export USE_TOKEN_DECORRELATION=False
    export TEACHER=self
    ;;
  v14-parcel-asclora)
    # v8 with the LoRA-rank ladder REVERSED so rank follows budget:
    #   v8 :  256->r8   144->r16  64->r32  16->r64   (largest budget, smallest rank)
    #   v14:  256->r64  144->r32  64->r16  16->r8    (rank grows with budget)
    # Everything else identical to v8 / final-parcel.
    #
    # Why: the vision-LoRA arm comparison (EXPERIMENT_JOURNAL 16m: none -> shared
    # r16 -> nested) shows per-level nesting is what makes the adapter help at
    # every budget. v8's pairing was a convention ("small budgets need more
    # adaptation to compensate"), never ablated. The competing hypothesis is
    # that the 256-token level -- the teacher, and the level carrying the most
    # information -- should get the most adapter capacity, with the 16-token
    # level's rank-8 barely perturbing CLIP. v14 tests that directly.
    #
    # Second thing this flips, inherent to the reversal, not a separate flag:
    # Stage 1 trains a rank-64 adapter at 256 tok. In v8, Stage 2's 256 level
    # then uses only the first 8 columns of that warm start; in v14 it uses all
    # 64, so the teacher level's warm start is finally consistent end to end.
    #
    # Requires the 2026-09-13 NestedLoRALinear change (max_rank = max(ranks),
    # no ascending assert) -- validated by jobs/test_lora_rank_order.sh.
    export ELASTIC_RUN_TAG=v14-parcel-asclora
    export TOK_LEVELS="256 144 64 16"
    export LORA_RANKS="64 32 16 8"
    export STAGE1_TOK_LEVEL=256
    export STAGE1_LORA_RANK=64
    export VISION_LORA_ENABLE=True
    export VISION_LORA_SPECIALIZE_TOK=True
    export USE_TOKEN_DECORRELATION=False
    export TEACHER=self
    ;;
  *)
    echo "Usage: bash submit_elastic_run.sh {v9-parcel-decorr|v10-parcel-lora16|v11-parcel-nolora|v12-parcel-kd7b|v13-parcel-longladder|v14-parcel-asclora|final-parcel}" >&2
    exit 1
    ;;
esac

GRES="${GRES:-$DEFAULT_GRES}"
CKPT="/var/scratch/skalra/flexllava/checkpoints/elastic-finetune-${SLM_KEY}-${ELASTIC_RUN_TAG}"
# --array must match the ladder length or eval_lmms_level.sh fails loudly on the
# out-of-range index (it no longer silently falls back to a 4-level default).
N_LEVELS=$(wc -w <<< "$TOK_LEVELS")
ARRAY="0-$(( N_LEVELS - 1 ))"

echo "── ${ELASTIC_RUN_TAG} (${SLM_KEY}) ──────────────────────────────"
for v in TOK_LEVELS LORA_RANKS STAGE1_TOK_LEVEL STAGE1_LORA_RANK RESAMPLER_ARCH \
         ANCHOR_MODE ANCHOR_RATIO USE_TOKEN_DECORRELATION DECORR_WEIGHT \
         VISION_LORA_ENABLE VISION_LORA_SPECIALIZE_TOK TEACHER TEACHER_MODEL_PATH \
         PREFIX_KL_WEIGHT VISION_TOWER; do
    printf '  %-26s %s\n' "$v" "${!v:-<launcher default>}"
done
printf '  %-26s %s\n' "gres" "$GRES"
printf '  %-26s %s\n' "eval --array" "$ARRAY"

if [ -n "${DRY_RUN:-}" ]; then echo "DRY_RUN set; nothing submitted."; exit 0; fi

TRAIN_ID=$(sbatch --parsable --gres="$GRES" run_job_slm.sh "$SLM_KEY")
echo "  train job   : $TRAIN_ID  (Stage 1 + Stage 2)"
EVAL_ID=$(sbatch --parsable --dependency=afterok:"$TRAIN_ID" --array="$ARRAY" \
                 eval_lmms_level.sh "$CKPT")
echo "  eval job    : $EVAL_ID  (afterok:$TRAIN_ID)"
echo "  checkpoint  : $CKPT"

# Job-id -> recipe map. `scontrol show job` records only "run_job_slm.sh
# <key>", which is identical for every recipe here, so without this line there
# is nothing on disk that says which job was which -- the same gap that made
# v9's configuration unrecoverable. Append-only; job ids are never reused.
# docs/, not jobs/: jobs/ is gitignored (it holds the .out logs), and the whole
# point of this file is that it survives in the repo alongside the recipes.
LOG=docs/SUBMITTED_RUNS.tsv
[ -s "$LOG" ] || printf 'submitted_utc\trecipe\tslm_key\ttrain_job\teval_job\ttok_levels\tlora_ranks\tresampler\tdecorr\tvision_lora\tspecialize_tok\tcheckpoint\n' > "$LOG"
printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$ELASTIC_RUN_TAG" "$SLM_KEY" "$TRAIN_ID" "$EVAL_ID" \
    "$TOK_LEVELS" "$LORA_RANKS" "$RESAMPLER_ARCH" "$USE_TOKEN_DECORRELATION" \
    "$VISION_LORA_ENABLE" "${VISION_LORA_SPECIALIZE_TOK:-True}" "$CKPT" >> "$LOG"
echo "  recorded in : $LOG"
