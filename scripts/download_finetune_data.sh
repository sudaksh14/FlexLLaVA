#!/bin/bash
#SBATCH --job-name=download_finetune
#SBATCH -t 12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --output=./jobs/download_finetune_%A.out
#SBATCH --export=ALL

# Downloads LLaVA visual instruction tuning image datasets to LLaVA-Finetune/.
#
# Sources (from LLaVA README / M3 README):
#   COCO train2017          ~18 GB  images.cocodataset.org
#   GQA images              ~20 GB  downloads.cs.stanford.edu
#   TextVQA train_val       ~ 8 GB  dl.fbaipublicfiles.com
#   VisualGenome part1+2    ~15 GB  cs.stanford.edu
#   OCR-VQA                        Google Drive — see manual step below
#
# llava_v1_5_mix665k.json is already downloaded; this script handles images only.
#
# Submit: sbatch scripts/download_finetune_data.sh
# Monitor: tail -f jobs/download_finetune_<JOBID>.out

DATA=/var/scratch/skalra/flexllava/data/LLaVA-Finetune
TMP=/var/scratch/skalra/flexllava/data/tmp_downloads
mkdir -p "$TMP"

echo "=== LLaVA Finetune image download started: $(date) ==="
echo "Node: $(hostname)"

# ── Helper ────────────────────────────────────────────────────────────────────
download_and_unzip() {
    local url="$1"
    local dest_dir="$2"
    local zip_name
    zip_name=$(basename "$url")
    local zip_path="$TMP/$zip_name"

    # Check for actual files (not just empty subdirs) before skipping
    if find "$dest_dir" -maxdepth 3 -type f 2>/dev/null | grep -q .; then
        echo "[skip] $dest_dir already has files"
        return
    fi

    echo "[download] $zip_name → $zip_path"
    wget -q --show-progress -c "$url" -O "$zip_path"

    echo "[unzip] $zip_name → $dest_dir"
    unzip -q "$zip_path" -d "$dest_dir"

    rm -f "$zip_path"
    echo "[done] $zip_name"
}

# ── COCO train2017 (~18 GB) ───────────────────────────────────────────────────
echo ""
echo "── COCO train2017 ──"
download_and_unzip \
    "http://images.cocodataset.org/zips/train2017.zip" \
    "$DATA/coco"

# ── GQA images (~20 GB) ──────────────────────────────────────────────────────
echo ""
echo "── GQA images ──"
download_and_unzip \
    "https://downloads.cs.stanford.edu/nlp/data/gqa/images.zip" \
    "$DATA/gqa"

# ── TextVQA train_val_images (~8 GB) ─────────────────────────────────────────
echo ""
echo "── TextVQA ──"
download_and_unzip \
    "https://dl.fbaipublicfiles.com/textvqa/images/train_val_images.zip" \
    "$DATA/textvqa"

# TextVQA zip extracts to train_val_images/; rename to train_images if needed
if [ -d "$DATA/textvqa/train_val_images" ] && [ ! -d "$DATA/textvqa/train_images" ]; then
    mv "$DATA/textvqa/train_val_images" "$DATA/textvqa/train_images"
fi

# ── VisualGenome part1 + part2 (~15 GB total) ─────────────────────────────────
echo ""
echo "── VisualGenome part1 ──"
download_and_unzip \
    "https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip" \
    "$DATA/vg"

echo ""
echo "── VisualGenome part2 ──"
download_and_unzip \
    "https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip" \
    "$DATA/vg"

# images2.zip extracts to VG_100K_2/; already the right name
# images.zip  extracts to VG_100K/;  already the right name

# ── OCR-VQA ───────────────────────────────────────────────────────────────────
echo ""
echo "── OCR-VQA ──"
OCR_OUT="$DATA/ocr_vqa/images"
OCR_JSON="/home/skalra/FlexLLaVA/dataset.json"

if find "$OCR_OUT" -maxdepth 1 -name "*.jpg" 2>/dev/null | grep -q .; then
    echo "[skip] $OCR_OUT already has .jpg files"
elif [ ! -f "$OCR_JSON" ]; then
    echo "[skip] dataset.json not found at $OCR_JSON"
    echo "  → Download dataset.json from:"
    echo "    https://drive.google.com/drive/folders/1_GYPY5UkUy7HIcR0zq3ZCFgeZN7BAfm_"
    echo "  → Place it at $OCR_JSON, then re-run this job."
else
    echo "[download] OCR-VQA images → $OCR_OUT"
    python /home/skalra/FlexLLaVA/loadDataset.py \
        --dataset_json "$OCR_JSON" \
        --out_dir "$OCR_OUT"
fi
echo ""

# ── Summary ───────────────────────────────────────────────────────────────────
echo "=== Download complete: $(date) ==="
echo ""
echo "Directory sizes:"
du -sh "$DATA"/*/  2>/dev/null
echo ""
echo "Expected final structure:"
echo "  $DATA/"
echo "  ├── llava_v1_5_mix665k.json  (983 MB — already downloaded)"
echo "  ├── coco/train2017/           COCO 2017 train images"
echo "  ├── gqa/images/               GQA images"
echo "  ├── ocr_vqa/images/           OCR-VQA images (manual step above)"
echo "  ├── textvqa/train_images/     TextVQA train+val images"
echo "  └── vg/VG_100K/              VisualGenome part1"
echo "      vg/VG_100K_2/            VisualGenome part2"
