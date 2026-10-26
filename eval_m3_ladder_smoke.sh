#!/bin/bash
#SBATCH --job-name=m3_ladder_smoke
#SBATCH -t 01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --exclude=node201,node202,node203,node204,node205
#SBATCH --output=./jobs/m3_ladder_smoke_%j.out
# Gate for the full M3 ladder eval (eval_lmms_ladder_m3.sh): runs 40 samples of POPE and TextVQA at
# 144 and 16 tokens through the SAME script, then fails the job if any generation raised (lmms-eval
# swallows generate() exceptions and scores them as empty answers, which is how the 2026-09-29 smoke
# runs ended up at POPE 0.0) or if POPE / TextVQA are at chance. Submit the full array with
# --dependency=afterok:<this job> so a broken eval never burns the 4 x ~6 h of GPU time.
#
#   sbatch eval_m3_ladder_smoke.sh [checkpoint_dir]
CKPT="${1:-/var/scratch/skalra/flexllava/checkpoints/baseline-tinyllama-ladder-finetune}"
ROOT=/var/scratch/skalra/flexllava/eval_logs/_smoke_m3_ladder
cd /home/skalra/FlexLLaVA
rm -rf "$ROOT"
for IDX in 1 3; do
    SLURM_ARRAY_TASK_ID=$IDX LIMIT=40 TASKS=pope,textvqa_val OUTROOT="$ROOT" BATCH_SIZE=4 \
        bash eval_lmms_ladder_m3.sh "$CKPT" 2>&1 | tee "$ROOT.$IDX.log"
done

eval "$(conda shell.bash hook)"
conda activate matryoshka-mm
fail=0
for IDX in 1 3; do
    n=$(grep -c "in generating" "$ROOT.$IDX.log")
    echo "array idx $IDX: generation errors = $n"
    [ "$n" -gt 0 ] && fail=1
done
python3 - <<'EOF' || fail=1
import glob, json, sys
root = "/var/scratch/skalra/flexllava/eval_logs/_smoke_m3_ladder"
bad = False
for tok in ("144tok", "16tok"):
    files = glob.glob(f"{root}/*/{tok}/*/results.json")
    if not files:
        print(tok, "NO RESULTS"); bad = True; continue
    r = json.load(open(files[0]))["results"]
    pope, tvqa = r["pope"]["pope_accuracy,none"], r["textvqa_val"]["exact_match,none"]
    print(f"{tok}: pope={pope:.3f} textvqa={tvqa:.3f}")
    if pope < 0.55 or tvqa < 0.02:
        bad = True
sys.exit(1 if bad else 0)
EOF
[ "$fail" -eq 0 ] && echo "SMOKE OK" || { echo "SMOKE FAILED"; exit 1; }
