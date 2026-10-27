#!/bin/bash
#SBATCH --job-name=smoke_parcel_exact
#SBATCH -t 02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:A40:1
#SBATCH --nodelist=node206
#SBATCH --cpus-per-task=16
#SBATCH --output=/home/skalra/FlexLLaVA/jobs/smoke_parcel_exact_%j.out
# Gate for the two PARCEL-pipeline runs (parcel-exact-nolora, parcel-exact-lora): 8 real steps of Stage 1
# and Stage 2 each (real LLaVA data, one GPU, the recipes' exact flags via RUN_INLINE), then a 16-sample
# POPE generation smoke at 16 and 256 tokens from the LoRA checkpoint. Fails the job on a non-finite loss,
# a missing banner/manifest entry, a sampled budget outside {16,18,...,256}, or a generation error.
# Submit the real runs with --dependency=afterok:<this job>.
set -uo pipefail
module load cuda12.6/toolkit/12.6
eval "$(conda shell.bash hook)"
conda activate matryoshka-mm
cd /home/skalra/FlexLLaVA
export HF_HOME=/var/scratch/skalra/.cache/huggingface WANDB_MODE=disabled PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NUM_GPUS=1 MAX_STEPS=8 TAG_SUFFIX=-smoke RUN_INLINE=1
CK=/var/scratch/skalra/flexllava/checkpoints
fail=0
for RECIPE in parcel-exact-nolora parcel-exact-lora; do
    LOG=/home/skalra/FlexLLaVA/jobs/smoke_parcel_exact_${RECIPE}_$SLURM_JOB_ID.log
    echo "=== $RECIPE ==="
    SLM_KEY=tinyllama bash submit_elastic_run.sh "$RECIPE" 2>&1 | tee "$LOG" | tail -5
    rc=${PIPESTATUS[0]}
    echo "$RECIPE exit=$rc"
    [ "$rc" -ne 0 ] && fail=1
    nonfinite=$(grep -ciE "\bnan\b|non-finite|\binf\b" "$LOG")
    echo "$RECIPE non-finite lines: $nonfinite"
    [ "$nonfinite" -gt 0 ] && fail=1
    grep -q "Resampler      : parcel_exact" "$LOG" || { echo "FAIL: banner missing parcel_exact"; fail=1; }
    grep -q "B ~ Uniform{16, 18, ..., 256}" "$LOG" || { echo "FAIL: banner missing uniform budget sampling"; fail=1; }
    grep -q "KD loss        : DISABLED" "$LOG" || { echo "FAIL: KD not disabled"; fail=1; }
    # one training loss per step, finite, for both stages
    python3 - "$LOG" <<'PYEOF' || fail=1
import re, sys
txt = open(sys.argv[1], errors="ignore").read()
losses = [float(m) for m in re.findall(r"'loss': ([0-9.]+)", txt)]
print(f"  logged losses: n={len(losses)} first={losses[:1]} last={losses[-1:]}")
if len(losses) < 12 or any(l != l or l > 1e3 for l in losses):
    print("  FAIL: too few or non-finite losses"); sys.exit(1)
PYEOF
    python3 - "$CK/elastic-finetune-tinyllama-${RECIPE}-smoke" <<'PYEOF' || fail=1
import json, sys
d = sys.argv[1]
m = json.load(open(f"{d}/run_manifest.json"))
c = json.load(open(f"{d}/elastic_config.json"))
print("  manifest:", m["resampler"]["arch"], m["budget_sampling"], "lr", m["learning_rate"], "kd", m["kd"]["use_prefix_kl"])
assert m["resampler"]["arch"] == "parcel_exact" and m["budget_sampling"]["mode"] == "uniform_range"
assert m["kd"]["use_prefix_kl"] is False and m["coral"]["enabled"] is False
assert float(m["learning_rate"]) == 1e-5, m["learning_rate"]
assert c["resampler_arch"] == "parcel_exact" and c["anchor_routing"] in ({"16": 16, "64": 64}, {16: 16, 64: 64})
assert c["single_budget_per_step"] is True and c["budget_sampling"] == "uniform_range"
PYEOF
done

echo "=== eval smoke (parcel-exact-lora, 16 samples of POPE at 16 and 256 tokens) ==="
EVALROOT=/var/scratch/skalra/flexllava/eval_logs/elastic-finetune-tinyllama-parcel-exact-lora-smoke
rm -rf "$EVALROOT"
for IDX in 0 3; do
    SLURM_ARRAY_TASK_ID=$IDX EVAL_LIMIT=16 TASKS=pope bash eval_lmms_level.sh \
        "$CK/elastic-finetune-tinyllama-parcel-exact-lora-smoke" 2>&1 | tee "/home/skalra/FlexLLaVA/jobs/smoke_parcel_exact_eval${IDX}_$SLURM_JOB_ID.log" | tail -3
    n=$(grep -c "in generating" "/home/skalra/FlexLLaVA/jobs/smoke_parcel_exact_eval${IDX}_$SLURM_JOB_ID.log")
    echo "eval idx $IDX generation errors: $n"
    [ "$n" -gt 0 ] && fail=1
    ls "$EVALROOT"/*tok/*results.json >/dev/null 2>&1 || ls "$EVALROOT"/*tok/*/*results.json >/dev/null 2>&1 || { echo "FAIL: no results for idx $IDX"; fail=1; }
done

if [ "$fail" -eq 0 ]; then echo "SMOKE OK"; else echo "SMOKE FAILED"; exit 1; fi
