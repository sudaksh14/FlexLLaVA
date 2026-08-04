#!/bin/bash
#SBATCH --job-name=eval_baseline
#SBATCH -t 48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:A10:1
#SBATCH --nodelist=node207
#SBATCH --cpus-per-task=8
#SBATCH --output=./jobs/eval_baseline_%A.out

# Baseline: stock LLaVA-1.5-7B at its native (maximum) visual-token budget.
#
# Why this needs its own script instead of eval_lmms_level.sh
# ----------------------------------------------------------
# The original model is NOT elastic. eval_lmms_level.sh uses
# --model llava_elastic, whose _attach_elastic() hard-requires
# elastic_config.json and raises FileNotFoundError without it. So this uses
# lmms-eval's plain `llava` wrapper.
#
# What "max level" means here
# ---------------------------
# Stock LLaVA-1.5 has no token levels: CLIP-L/336 yields a 24x24 patch grid,
# so it always feeds the LLM all 576 visual tokens. Our fork's llava wrapper
# passes matryoshka_vis_token_scale from the model config, and stock
# llava-v1.5-7b's config.json has no such key -> None -> the matryoshka
# pooling path is skipped entirely and the full 576 tokens are used. That IS
# the maximum level, and it is the model's native behaviour.
#
# Note this is a HIGHER token budget than our elastic max (tok_levels[0]=256),
# so it is an upper-bound reference rather than a token-matched comparison.
#
# Submit: sbatch eval_lmms_baseline_llava.sh

MODEL_PATH="${1:-liuhaotian/llava-v1.5-7b}"     # resolved from the HF cache
MODEL_TAG="llava-v1.5-7b-baseline"
LABEL="576tok-native"
LOG_ROOT=/var/scratch/skalra/flexllava/eval_logs
OUTDIR="${LOG_ROOT}/${MODEL_TAG}/${LABEL}"

# Same five tasks as the elastic evals so the numbers line up column-for-column.
# mmbench_en_dev still excluded (needs OPENAI_API_KEY; without it unparseable
# answers get a random option -- see eval_lmms_level.sh for the full rationale).
TASKS="${TASKS:-mme,pope,scienceqa_img,textvqa_val,gqa}"

# 7B in fp16 is ~14GB of weights on a 24GB A10, so this runs a smaller batch
# than the 1.1B elastic evals (which use 4). Batch size affects speed only,
# never the scores.
BATCH_SIZE="${BATCH_SIZE:-2}"

module load cuda12.6/toolkit/12.6
eval "$(conda shell.bash hook)"
conda activate matryoshka-mm

export HF_HOME=/var/scratch/skalra/.cache/huggingface
export HF_DATASETS_CACHE=/var/scratch/skalra/.cache/huggingface/datasets

cd /home/skalra/FlexLLaVA

echo "Job started: $(date)"
echo "Node: $(hostname)"
nvidia-smi | head -12
echo "Model:  $MODEL_PATH   (stock LLaVA-1.5-7B, native 576 visual tokens)"
echo "Tasks:  $TASKS"
echo "Batch:  $BATCH_SIZE"

pip show lmms-eval >/dev/null 2>&1 || pip install -e lmms-eval -q
mkdir -p "$OUTDIR"

echo ""
echo "══════════════════════════════════════════════════"
echo "  BASELINE  stock LLaVA-1.5-7B  (max / native token budget)"
echo "══════════════════════════════════════════════════"

accelerate launch --num_processes=1 \
    -m lmms_eval \
    --model       llava \
    --model_args  "pretrained=${MODEL_PATH},conv_template=vicuna_v1" \
    --tasks       "$TASKS" \
    --batch_size  $BATCH_SIZE \
    --log_samples \
    --log_samples_suffix "baseline_${LABEL}" \
    --output_path "$OUTDIR"

echo ""
echo "Done baseline: $(date)"

# ── Analytic efficiency for the baseline ────────────────────────────────────
# The plain `llava` wrapper has no efficiency instrumentation (that lives in
# LlavaElastic), but the cost model is deterministic given the config and the
# sequence length, so it can be computed here for free -- no GPU needed. This
# gives the baseline a FLOPs/latency/memory row comparable to the elastic runs.
echo ""
echo "=== analytic efficiency (stock LLaVA-1.5-7B, 576 visual tokens) ==="
python3 - << 'PYEOF'
from transformers import AutoConfig
import llava.model  # registers the llava_* configs
from llava.eval.efficiency import ElasticAnalyzer, vision_tower_gflops

cfg = AutoConfig.from_pretrained("liuhaotian/llava-v1.5-7b")
TEXT_LEN, GEN, NTOK = 64, 32, 576
for hw in ["jetson_orin_nano_8gb", "nvidia_A10"]:
    an = ElasticAnalyzer(cfg, hw)
    r = an.analyze_generate_task(TEXT_LEN + NTOK, GEN)
    fits = an.fits_in_memory(r["memory_consumption"])
    print(f"  {hw:24} prefill {r['prefill_flops']/1e12:8.3f} TFLOPs | "
          f"{r['prefill_time']*1e3:8.2f} ms | peak {r['memory_consumption']/1e9:6.2f} GB | "
          f"fits={fits}")
print(f"  vision tower constant: {vision_tower_gflops():.0f} GFLOPs/image")
print("  NOTE: 576 tokens vs our elastic max of 256 -- upper-bound reference,")
print("        not a token-matched comparison.")
PYEOF

echo ""
echo "Results: $OUTDIR"
echo "Job complete: $(date)"
