#!/bin/bash
#SBATCH --job-name=resave_7b
#SBATCH -t 04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:A40:2
#SBATCH --cpus-per-task=32
#SBATCH --output=./jobs/resave_7b_%A.out
#SBATCH --export=ALL,WANDB_API_KEY=dfcd2574507b9ebe69ca13ab6f6925d864e82ee0

# Re-save the 7B Stage-2 checkpoint with the FIXED save path, then strip the
# stale PEFT artefacts.
#
# Why this is needed
# -----------------
# Job 26232 started 2026-07-27 21:23; llava/train/train.py was fixed on
# 2026-07-31. Python loads a module once at process start, so 26232 is running
# the OLD save code in memory and its final checkpoint will have all three of
# the bugs we just fixed for TinyLlama:
#   1. no tokenizer files at the top level  -> eval dies with
#      "OSError: Can't load tokenizer for ..."
#   2. the vision tower's rank-nested LoRA (lora_A/lora_B, i.e. the elastic
#      mechanism itself) is silently dropped -- PEFT's save_pretrained only
#      keeps ITS OWN registered adapters, and get_peft_state_non_lora_*
#      excludes anything with "lora_" in the name
#   3. adapter_config.json is written, which makes transformers'
#      from_pretrained resolve the base model from base_model_name_or_path on
#      the HF Hub instead of reading this directory
#
# Nothing is actually lost: checkpoint-*/global_step*/mp_rank_00_model_states.pt
# is the full DeepSpeed model state and does contain the vision LoRA. Resuming
# with the fixed code rehydrates it and re-saves correctly. Because the run has
# already completed its epoch, trainer.train(resume_from_checkpoint=True) does
# ZERO additional optimizer steps and falls straight through to the save --
# this is the same manoeuvre that repaired the TinyLlama checkpoint (run 26474,
# train_runtime 3.99s).
#
# Must run on 2x A40: the DeepSpeed optimizer state is sharded across the same
# number of ranks it was written with.
#
# Submit chained to the training job:
#   sbatch --dependency=afterany:26232 resave_7b_finetune.sh

CKPT=/var/scratch/skalra/flexllava/checkpoints/llava-elastic-finetune

module load cuda12.1/toolkit/12.1
eval "$(conda shell.bash hook)"
conda activate matryoshka-mm

export NCCL_SOCKET_TIMEOUT=3600
export NCCL_DEBUG=WARN
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_ASYNC_ERROR_HANDLING=1
export WANDB_MODE=disabled
export HF_HOME=/var/scratch/skalra/.cache/huggingface

cd /home/skalra/FlexLLaVA

echo "Re-save started: $(date)"
echo "Node: $(hostname)"
echo "Checkpoint: $CKPT"
echo ""
echo "=== BEFORE ==="
ls -la "$CKPT"

if ! ls -d "$CKPT"/checkpoint-* >/dev/null 2>&1; then
    echo "ERROR: no checkpoint-* under $CKPT -- nothing to resume from."
    echo "Training may not have reached its first save_steps boundary."
    exit 1
fi

# Resumes, runs 0 steps (epoch already complete), re-saves via the fixed path.
./scripts/v1_5/finetune_elastic.sh

echo ""
echo "=== AFTER re-save ==="
ls -la "$CKPT"

# Strip the stale PEFT artefacts written by the ORIGINAL run's old save code.
# The fixed path writes a self-contained pytorch_model.bin and deliberately
# does NOT write these; leaving them behind re-triggers bug (3) above.
for f in adapter_config.json adapter_model.safetensors non_lora_trainables.bin; do
    if [ -f "$CKPT/$f" ]; then
        rm -f "$CKPT/$f" && echo "removed stale $f"
    fi
done

echo ""
echo "=== FINAL (should have pytorch_model.bin + tokenizer.*, no adapter_*) ==="
ls -la "$CKPT"
echo "Re-save finished: $(date)"
