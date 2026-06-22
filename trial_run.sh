#!/bin/bash
#SBATCH --job-name=trial_run
#SBATCH -t 01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:A40:1
#SBATCH --cpus-per-task=8
#SBATCH --output=./jobs/trial_run_%A.out
#SBATCH --export=ALL

module load cuda12.6/toolkit/12.6

eval "$(conda shell.bash hook)"
conda activate /var/scratch/skalra/prune_llm

nvidia-smi

# ── Fill in your model path ───────────────────────────────────────────────────
MODEL_PATH="liuhaotian/llava-v1.5-7b"   # or a local path on the cluster
VISION_TOWER="openai/clip-vit-large-patch14-336"
# ─────────────────────────────────────────────────────────────────────────────

DUMMY_DATA="./playground/data/dummy/dummy_data.json"
DUMMY_IMAGES="./playground/data/dummy/images"

# Generate dummy data if it doesn't exist yet
if [ ! -f "$DUMMY_DATA" ]; then
    echo "Generating dummy data..."
    python scripts/create_dummy_data.py
fi

echo "=== Trial run started ==="
date

python -m llava/train/train_mem \
    --model_name_or_path "$MODEL_PATH" \
    --version v1 \
    --data_path "$DUMMY_DATA" \
    --image_folder "$DUMMY_IMAGES" \
    --vision_tower "$VISION_TOWER" \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --freeze_backbone True \
    --bf16 True \
    --output_dir ./checkpoints/trial_run_${SLURM_JOB_ID} \
    --max_steps 5 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "no" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 512 \
    --gradient_checkpointing False \
    --dataloader_num_workers 2 \
    --lazy_preprocess True \
    --report_to none

echo "=== Trial run complete ==="
date
