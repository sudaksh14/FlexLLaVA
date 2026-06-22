"""OCR-VQA image downloader — fixed for LLaVA instruction tuning.

Changes from the original Google Drive version:
  - Removed pdb.set_trace() (would hang any non-interactive run)
  - Saves all images as .jpg (LLaVA requirement — original saves original ext)
  - Output directory configurable via --out_dir (default: ./images)
  - Skips already-downloaded files so the script is safe to re-run
  - os.makedirs with exist_ok=True instead of os.mkdir

Usage:
  python loadDataset.py --out_dir /var/scratch/skalra/flexllava/data/LLaVA-Finetune/ocr_vqa/images

Requires dataset.json in the current directory (download from the same
Google Drive folder: https://drive.google.com/drive/folders/1_GYPY5UkUy7HIcR0zq3ZCFgeZN7BAfm_)
"""

import argparse
import json
import os
import urllib.request as ureq
from io import BytesIO

from PIL import Image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", default="./images",
                        help="Directory to save downloaded .jpg images")
    parser.add_argument("--dataset_json", default="dataset.json",
                        help="Path to dataset.json from the OCR-VQA Google Drive")
    args = parser.parse_args()

    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    with open(args.dataset_json, "r") as fp:
        data = json.load(fp)

    total = len(data)
    skipped = 0
    downloaded = 0
    failed = 0

    for i, k in enumerate(data.keys()):
        out_path = os.path.join(out_dir, f"{k}.jpg")

        if os.path.exists(out_path):
            skipped += 1
            continue

        url = data[k]["imageURL"]
        try:
            raw = ureq.urlopen(url, timeout=30).read()
            img = Image.open(BytesIO(raw)).convert("RGB")
            img.save(out_path, "JPEG")
            downloaded += 1
        except Exception as e:
            print(f"[WARN] failed {k}: {e}")
            failed += 1

        if (i + 1) % 500 == 0 or (i + 1) == total:
            print(f"  {i+1}/{total}  downloaded={downloaded}  skipped={skipped}  failed={failed}")

    print(f"\nDone. downloaded={downloaded}  skipped={skipped}  failed={failed}")
    print(f"Images saved to: {out_dir}")


if __name__ == "__main__":
    main()
