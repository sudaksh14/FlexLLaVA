#!/bin/bash
#SBATCH --job-name=baseline_dense_s2
#SBATCH -t 48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:A10:2
#SBATCH --cpus-per-task=32
#SBATCH --output=./jobs/run_%A.out
#SBATCH --export=ALL
#SBATCH --exclusive

# Re-train ONLY Stage 2 of the dense 576-token TinyLlama control (the checkpoint formerly named baseline-tinyllama-finetune (renamed baseline-tinyllama-576tok-finetune 2026-10-27) was
# removed in the 2026-09 cleanup; its eval numbers survive in eval_logs/baseline-tinyllama-576tok).
# Stage 1 trains the plain projector on all 576 tokens independent of Stage 2's budget (see
# finetune_baseline_slm.sh), so baseline-tinyllama-4tok-pretrain/mm_projector.bin is reused as the warm start.
# Same recipe/hyperparameters as run_job_baseline_slm.sh Stage 2 (original run: 69,983 s on A10:2).
#   sbatch run_job_baseline_finetune.sh            -> /var/scratch/skalra/flexllava/checkpoints/baseline-tinyllama-576tok-finetune

module load cuda12.1/toolkit/12.1
eval "$(conda shell.bash hook)"
conda activate matryoshka-mm
nvidia-smi; echo "Job Started"; date; echo "Node name: $(hostname)"
export NCCL_SOCKET_TIMEOUT=3600 NCCL_DEBUG=WARN PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True NCCL_ASYNC_ERROR_HANDLING=1
export WANDB_PROJECT="FlexLLaVA" WANDB_DIR=/var/scratch/skalra/flexllava/wandb HF_HOME=/var/scratch/skalra/.cache/huggingface
set -e
BASELINE_RUN_TAG=576tok BASELINE_PRETRAIN_TAG=4tok ./scripts/v1_5/finetune_baseline_slm.sh tinyllama
