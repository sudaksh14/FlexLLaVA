#!/bin/bash
# Launch the budget-matched M3 and MQT-LLaVA ladder finetunes (TinyLlama, 256/144/64/16),
# reusing the existing 4tok Stage-1 checkpoints.
#
# The ladder is exported in THIS shell and passed with --export=ALL. Do NOT put it in
# `--export=ALL,VAR=256,144,64,16`: sbatch splits that on the commas and the job silently
# receives VAR=256 (jobs 27537/27538 ran as single-budget 256-token runs because of it).
set -euo pipefail
cd "$(dirname "$0")"
LADDER="256,144,64,16"
export NUM_GPUS=2 BASELINE_RUN_TAG=ladder BASELINE_PRETRAIN_TAG=4tok
WRAP='module load cuda12.1/toolkit/12.1; eval "$(conda shell.bash hook)"; conda activate matryoshka-mm; export HF_HOME=/var/scratch/skalra/.cache/huggingface; cd /home/skalra/FlexLLaVA;'

M3=$(MATRYOSHKA_SCALE="$LADDER" sbatch --parsable --export=ALL --gres=gpu:2 \
    --wrap="$WRAP bash scripts/v1_5/finetune_baseline_slm.sh tinyllama" \
    --job-name=m3-ladder-finetune --output=./jobs/m3-ladder-finetune_%A.out \
    --nodes=1 --ntasks-per-node=2 --cpus-per-task=32 --exclusive -t 200:00:00)
MQT=$(NUM_VISUAL_TOKENS="$LADDER" sbatch --parsable --export=ALL --gres=gpu:2 \
    --wrap="$WRAP bash scripts/v1_5/finetune_mqt_baseline.sh tinyllama" \
    --job-name=mqt-ladder-finetune --output=./jobs/mqt-ladder-finetune_%A.out \
    --nodes=1 --ntasks-per-node=2 --cpus-per-task=32 --exclusive -t 200:00:00)
echo "M3=$M3 MQT=$MQT"
