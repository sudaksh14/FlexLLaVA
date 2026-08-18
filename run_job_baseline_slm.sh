#!/bin/bash
#SBATCH --job-name=baseline_TinyLLaVA
#SBATCH -t 200:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
# A10:2 -> node208. node205's A40s are held by the SmolLM2 Stage 2 run, and this
# control is small enough for 24GB cards: Stage 1 trains only the projector, and
# Stage 2 is a 1.1B full finetune under ZeRO-2 (~11GB of states before
# activations) at per_device_train_batch_size 2.
#SBATCH --gres=gpu:A10:2
#SBATCH --cpus-per-task=32
#SBATCH --output=./jobs/run_%A.out
#SBATCH --export=ALL,WANDB_API_KEY=dfcd2574507b9ebe69ca13ab6f6925d864e82ee0
# Takes every GPU on its node, so claim every core too -- see the note in
# run_job_slm.sh.
#SBATCH --exclusive

# NON-ELASTIC control: can our pipeline reproduce TinyLLaVA's published numbers
# for the same backbone? Runs both stages back to back (unlike the elastic SLM
# runs there is no existing Stage 1 to reuse -- this projector is a different
# shape, 576 tokens through mlp2x_gelu rather than 256 resampler queries).
#
#   sbatch run_job_baseline_slm.sh tinyllama
#
# Reference (TinyLLaVA_Factory/README.md:152, clip-L-336 + TinyLlama-1.1B):
#   GQA 58.0 | SQA-I 59.9 | TextVQA 46.3 | POPE 85.5 | MME 1284.6
# Our elastic run at its 256-token maximum:
#   GQA 52.3 | SQA-I 51.3 | TextVQA 23.1 | POPE 81.6 | MME 1113.5

module load cuda12.1/toolkit/12.1

eval "$(conda shell.bash hook)"
conda activate matryoshka-mm

nvidia-smi
echo "Job Started"; date
echo "Node name: $(hostname)"
echo -n nproc=; nproc

export NCCL_SOCKET_TIMEOUT=3600
export NCCL_DEBUG=WARN
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_ASYNC_ERROR_HANDLING=1

export WANDB_PROJECT="FlexLLaVA"
export WANDB_DIR=/var/scratch/skalra/flexllava/wandb
export HF_HOME=/var/scratch/skalra/.cache/huggingface

SLM_KEY=${1:-tinyllama}
echo "[baseline] SLM_KEY=${SLM_KEY}  BASELINE_RUN_TAG=${BASELINE_RUN_TAG:-<none>}"

set -e
./scripts/v1_5/pretrain_baseline_slm.sh "$SLM_KEY"
./scripts/v1_5/finetune_baseline_slm.sh "$SLM_KEY"

echo "Job Complete"; date
