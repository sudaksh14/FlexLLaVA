#!/bin/bash
#SBATCH --job-name=ft_FlexLLaVA_SLM
# See run_job_slm.sh for why 200h: one epoch of Stage 2 is 5197 steps, and
# run_26226 hit the old self-imposed 150h limit at 92% (4778/5197).
#SBATCH -t 200:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
# A40:2 (node205), not A10:2 (node208): a 1.7B full finetune under ZeRO-2 needs
# ~17GB/GPU before activations, which is uncomfortably close to the A10's 24GB.
# Phi-3.5 already OOM'd at 41.5/44.67GB on A40s (run_26642), so the smaller card
# is not worth the risk for a 24h+ job.
#SBATCH --gres=gpu:A40:2
#SBATCH --cpus-per-task=32
#SBATCH --output=./jobs/run_%A.out
#SBATCH --export=ALL,WANDB_API_KEY=dfcd2574507b9ebe69ca13ab6f6925d864e82ee0
# Training jobs take every GPU on their node, so no other GPU job can use it.
# --exclusive therefore also claims all 64 cores: without it, cons_tres/CR_CORE
# confines the job to --cpus-per-task cores (task/affinity) and the rest idle
# for nothing. It also makes the allocation robust to a smaller
# --cpus-per-task passed at submit time.
#SBATCH --exclusive

# Stage 2 ONLY. run_job_slm.sh runs pretrain+finetune back to back, which wastes
# ~10h re-deriving a Stage-1 checkpoint that already exists and is unaffected by
# whatever made Stage 2 need re-running.
#
#   ELASTIC_RUN_TAG=v4 sbatch run_job_finetune_slm.sh smollm2
#
# ELASTIC_RUN_TAG picks the output dir (elastic-finetune-<key>-<tag>) and, unless
# ELASTIC_PRETRAIN_TAG is set separately, the Stage-1 checkpoint to warm-start
# from. Reusing a tag OVERWRITES that finetune checkpoint.

module load cuda12.1/toolkit/12.1

eval "$(conda shell.bash hook)"
conda activate matryoshka-mm

nvidia-smi

echo "Job Started"
echo | date
echo "Node name: $(hostname)"
echo -n memory=; ulimit -m
echo -n nproc=; nproc

export NCCL_SOCKET_TIMEOUT=3600
export NCCL_DEBUG=WARN
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# See run_job_slm.sh: overlap_comm:false + this flag are what stopped the
# rank-desync deadlock at step 136 in run_26187/26205/26209.
export NCCL_ASYNC_ERROR_HANDLING=1

export WANDB_PROJECT="FlexLLaVA"
export WANDB_DIR=/var/scratch/skalra/flexllava/wandb
export HF_HOME=/var/scratch/skalra/.cache/huggingface

SLM_KEY=${1:-tinyllama}
echo "[FlexLLaVA] SLM_KEY=${SLM_KEY}  ELASTIC_RUN_TAG=${ELASTIC_RUN_TAG:-<none>}"

./scripts/v1_5/finetune_elastic_slm.sh "$SLM_KEY"

echo "Job Complete"
echo | date
