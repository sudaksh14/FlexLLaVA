#!/bin/bash
# M3 and MQT-LLaVA baselines at 4 visual tokens, TinyLlama-1.1B.
#
#   bash run_baselines_4tok.sh              # print plan, submit nothing
#   SUBMIT=1 bash run_baselines_4tok.sh     # submit
#   SUBMIT=1 ONLY=m3  bash run_baselines_4tok.sh
#   SUBMIT=1 MAX_STEPS=500 bash run_baselines_4tok.sh    # truncate for a smoke run
#
# Overrides: GRES (default gpu:2), NUM_GPUS (default 2), NODELIST (default
# unset -- SLURM picks whichever GRES-matching node frees up first).
#
# ---------------------------------------------------------------------------
# WHAT EACH BASELINE IS
# ---------------------------------------------------------------------------
# M3  -- this repo IS an M3 fork with M3 intact underneath (README "Relationship
#        to M3"). With no elastic_engine attached, LlavaElasticMixin.forward
#        takes the "Pure M3 (explicit scale list)" branch and
#        llava_arch.matryoshka_vis_token_process does M3's avg-pool. So this is
#        M3's own train loop, not a reimplementation. scale=4 -> 12x12 pool over
#        the 24x24 CLIP grid -> exactly 4 tokens (verified, job 27415).
#
# MQT -- MQT-LLaVA/llava/train/train.py, run from inside MQT-LLaVA/ so its
#        vendored `llava` shadows ours. Its "query_abstractor" is a 2D perceiver
#        resampler with 256 learnable queries; num_visual_tokens keeps a prefix.
#        Imports cleanly in this env despite pinning transformers==4.36.2 vs our
#        4.44.2 (job 27415).
#
# Held identical to our runs so the numbers are comparable:
#   vision encoder  CLIP-L/336, frozen
#   backbone        TinyLlama-1.1B-Chat-v1.0, conv v1, full LLM finetune
#   Stage 1 data    blip_laion_cc_sbu_558k     LR 1e-3, 1 epoch, eff batch 256
#   Stage 2 data    llava_v1_5_mix665k         LR 2e-5 cosine, warmup 0.03,
#                   wd 0, 1 epoch, eff batch 128, bf16, max_len 2048, pad
#   seed            HF default 42 (unset in every launcher, ours included)
#
# WHAT IS DELIBERATELY NOT MATCHED, and why it is not a confound:
#   Each method keeps ITS OWN Stage-1 convention, because that is part of the
#   pipeline being baselined -- M3 pretrains the plain projector on all 576
#   tokens (it pools AFTER the projector, so Stage 1 is ordinary LLaVA), MQT
#   pretrains its query bank at num_visual_tokens=first_stage (=256). Forcing a
#   common Stage 1 would mean neither baseline was the published method.
#
# HONEST SCOPE: both are run at a FIXED 4-token budget, which narrows each
# method -- M3 normally trains a list of scales jointly and MQT normally samples
# a random query count per step ('second_stage'). These are "M3/MQT
# architecture + train loop at a fixed 4-token budget", NOT the published
# elastic models. Do not report them as M3/MQT headline numbers.
set -uo pipefail
cd "$(dirname "$0")"

GRES="${GRES:-gpu:2}"
NODELIST="${NODELIST:-}"
SUBMIT="${SUBMIT:-}"
ONLY="${ONLY:-}"
TAG="${BASELINE_RUN_TAG:-4tok}"
LOG=docs/SUBMITTED_RUNS.tsv
PASS_ENV="NUM_GPUS=${NUM_GPUS:-2}${MAX_STEPS:+,MAX_STEPS=$MAX_STEPS}"

echo "=================================================================="
echo " 4-token baselines: M3 and MQT-LLaVA, TinyLlama-1.1B"
echo "   tag=${TAG}  gres=${GRES}  ${MAX_STEPS:+MAX_STEPS=$MAX_STEPS (TRUNCATED -- smoke only)}"
echo "=================================================================="

sub() {  # name stage1_cmd stage2_cmd ckpt evalarray
    local name="$1" s1="$2" s2="$3" ck="$4"
    echo
    echo "--- $name"
    echo "    stage1: $s1"
    echo "    stage2: $s2"
    echo "    ckpt  : $ck"
    [ -z "$SUBMIT" ] && return 0
    local j1 j2 je
    j1=$(sbatch --parsable --gres="$GRES" ${NODELIST:+--nodelist="$NODELIST"} \
         --export=ALL,${PASS_ENV},BASELINE_RUN_TAG=${TAG} \
         --wrap="module load cuda12.1/toolkit/12.1; eval \"\$(conda shell.bash hook)\"; conda activate matryoshka-mm; export HF_HOME=/var/scratch/skalra/.cache/huggingface; cd /home/skalra/FlexLLaVA; $s1" \
         --job-name="${name}_s1" --output="./jobs/${name}_s1_%A.out" \
         --nodes=1 --ntasks-per-node=2 --cpus-per-task=32 --exclusive -t 100:00:00)
    j2=$(sbatch --parsable --dependency=afterok:$j1 --gres="$GRES" ${NODELIST:+--nodelist="$NODELIST"} \
         --export=ALL,${PASS_ENV},BASELINE_RUN_TAG=${TAG} \
         --wrap="module load cuda12.1/toolkit/12.1; eval \"\$(conda shell.bash hook)\"; conda activate matryoshka-mm; export HF_HOME=/var/scratch/skalra/.cache/huggingface; cd /home/skalra/FlexLLaVA; $s2" \
         --job-name="${name}_s2" --output="./jobs/${name}_s2_%A.out" \
         --nodes=1 --ntasks-per-node=2 --cpus-per-task=32 --exclusive -t 150:00:00)
    echo "    stage1 job $j1  ->  stage2 job $j2"
    printf '%s\t%s\ttinyllama\t%s\t%s\t4 tokens\tbaseline\t%s\t-\t-\t-\t%s\n' \
        "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$name-$TAG" "$j1" "$j2" "$name" "$ck" >> "$LOG"
}

if [ -z "$ONLY" ] || [ "$ONLY" = "m3" ]; then
  sub "m3-4tok" \
      "bash scripts/v1_5/pretrain_baseline_slm.sh tinyllama" \
      "MATRYOSHKA_SCALE=4 bash scripts/v1_5/finetune_baseline_slm.sh tinyllama" \
      "/var/scratch/skalra/flexllava/checkpoints/baseline-tinyllama-${TAG}-finetune"
fi
if [ -z "$ONLY" ] || [ "$ONLY" = "mqt" ]; then
  sub "mqt-4tok" \
      "bash scripts/v1_5/pretrain_mqt_baseline.sh tinyllama" \
      "NUM_VISUAL_TOKENS=4 bash scripts/v1_5/finetune_mqt_baseline.sh tinyllama" \
      "/var/scratch/skalra/flexllava/checkpoints/mqt-finetune-tinyllama-${TAG}"
fi

echo
echo "COST: both Stage 2s run ONE forward per step at 4 visual tokens, vs our"
echo "      elastic runs' two forwards at 256 + ~75 tokens -- so Stage 2 should be"
echo "      several times cheaper than our ~85h. Stage 1 is the expensive half"
echo "      here (558k samples at 576 tokens for M3 / 256 queries for MQT) and is"
echo "      NOT reduced by the 4-token setting. Estimate, not a measurement:"
echo "      run MAX_STEPS=200 first and read it/s off the log before committing."
echo
echo "EVAL: eval_lmms_level.sh expects an elastic_config.json and will NOT work on"
echo "      these checkpoints. Evaluate with the baseline path"
echo "      (eval_lmms_baseline_llava.sh) or add a 4-token eval wrapper."
[ -z "$SUBMIT" ] && echo && echo "DRY RUN -- nothing submitted. Re-run with SUBMIT=1."
