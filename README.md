# 🌋 FlexLLaVA

Elastic, adaptive-visual-token-budget vision-language models across small LLM (SLM)
backbones — TinyLlama, Phi-2, SmolLM2, Qwen2.5, StableLM, MobileLLaMA. Built on top
of **M3 (Matryoshka Multimodal Models)** and LLaVA-1.5; the original M3 codebase is
kept intact underneath (see [Relationship to M3](#relationship-to-m3)) and this repo
adds a second, parallel training pipeline on top of it that makes the visual-token
count a first-class, trainable elasticity axis for small backbones specifically.

For how the pipeline actually works internally — component-by-component, with file
references — see **[docs/ELASTIC_PIPELINE.md](docs/ELASTIC_PIPELINE.md)**. This
README covers install, data, and how to run experiments.

[![Code License](https://img.shields.io/badge/Code%20License-Apache_2.0-green.svg)](https://github.com/tatsu-lab/stanford_alpaca/blob/main/LICENSE)
**Usage and License Notices**: This project utilizes certain datasets and checkpoints
that are subject to their respective original licenses. Users must comply with all
terms and conditions of these original licenses, including but not limited to the
[OpenAI Terms of Use](https://openai.com/policies/terms-of-use) for the dataset and
the specific licenses for base language models for checkpoints trained using the
dataset. This project does not impose any additional constraints beyond those
stipulated in the original licenses.

## Contents
- [Install](#install)
- [Supported backbones](#supported-backbones)
- [Data preparation](#data-preparation)
- [Running experiments](#running-experiments)
- [Evaluation](#evaluation)
- [Elastic pipeline internals](#elastic-pipeline-internals)
- [Experiment log](#experiment-log)
- [Relationship to M3](#relationship-to-m3)
- [Citation](#citation)

## Install

If you are not using Linux, do *NOT* proceed — see the original M3 project's
instructions for [macOS](docs/macOS.md) and [Windows](docs/Windows.md); they were
not written for this fork and are unverified here.

1. Clone this repository
```bash
git clone <this repo>
cd FlexLLaVA
```

2. Install the package
```Shell
conda create -n matryoshka-mm python=3.10 -y
conda activate matryoshka-mm
pip install --upgrade pip  # enable PEP 660 support
pip install -e .
```

3. Install additional packages for training
```Shell
pip install -e ".[train]"
pip install flash-attn --no-build-isolation
```

All elastic-SLM launcher scripts (`scripts/v1_5/*_elastic_slm.sh`) assume the
`matryoshka-mm` conda env by name — `eval "$(conda shell.bash hook)"; conda activate
matryoshka-mm` is the first thing every sbatch job does.

**Never run training or evaluation Python directly on a login node** — every script
in this repo is meant to be submitted via `sbatch`. See `run_job_slm.sh`,
`run_job_pretrain_slm.sh`, and the `jobs/*.sh` wrappers for the pattern: a thin
sbatch header, `conda activate`, then the actual `*_elastic_slm.sh` call.

## Supported backbones

`LLM_KEY` (used by every `*_elastic_slm.sh` script and `eval_lmms_slm_all.sh`) and
the resulting LLM hidden size (also the width every visual token is projected to —
see [docs/ELASTIC_PIPELINE.md §4](docs/ELASTIC_PIPELINE.md#4-token-reduction-nestedqueryresamplerforward-resamplerpy)):

| `LLM_KEY` | HF model | conv template | `hidden_size` |
|---|---|---|---:|
| `tinyllama` | `TinyLlama/TinyLlama-1.1B-Chat-v1.0` | `v1` | 2048 |
| `mobilellama` | `mtgv/MobileLLaMA-1.4B-Chat` | `v1` | 2048 |
| `smollm2` | `HuggingFaceTB/SmolLM2-1.7B-Instruct` | `chatml` | 2048 |
| `qwen0.5b` | `Qwen/Qwen2.5-0.5B-Instruct` | `chatml` | 896 |
| `qwen1.5b` | `Qwen/Qwen2.5-1.5B-Instruct` | `chatml` | 1536 |
| `qwen3b` | `Qwen/Qwen2.5-3B-Instruct` | `chatml` | 2048 |
| `phi2` | `microsoft/phi-2` | `phi` | 2560 |
| `phi3.5` | `microsoft/Phi-3.5-mini-instruct` | `phi3` | 3072 |
| `stablelm` | `stabilityai/stablelm-2-zephyr-1_6b` | `chatml` | 2048 |

Vision tower for all backbones: `openai/clip-vit-large-patch14-336` (576 patches,
`hidden_size=1024`), connector `mlp2x_gelu` in the base path / the elastic resampler
+ projector when the elastic engine is attached.

## Data preparation

Same annotation files and image sources as upstream LLaVA-1.5 / M3 — this still
applies unchanged:

- Pretrain (Stage 1) annotations: `blip_laion_cc_sbu_558k.json` — the standard LLaVA
  pretrain annotation file (`liuhaotian/LLaVA-Pretrain` on HuggingFace); link omitted
  here rather than risk pointing at a stale/wrong path.
- Finetune (Stage 2) annotations: [llava_v1_5_mix665k.json](https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K/blob/main/llava_v1_5_mix665k.json)
- COCO: [train2017](http://images.cocodataset.org/zips/train2017.zip)
- GQA: [images](https://downloads.cs.stanford.edu/nlp/data/gqa/images.zip)
- OCR-VQA: [download script](https://drive.google.com/drive/folders/1_GYPY5UkUy7HIcR0zq3ZCFgeZN7BAfm_?usp=sharing) — save all files as `.jpg`
- TextVQA: [train_val_images](https://dl.fbaipublicfiles.com/textvqa/images/train_val_images.zip)
- VisualGenome: [part1](https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip), [part2](https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip)

The elastic-SLM launcher scripts point at these under `/var/scratch/skalra/flexllava/data/`
(`LLaVA-Pretrain/`, `LLaVA-Finetune/`) rather than `./playground/data` — update
`--data_path`/`--image_folder` in the scripts if your data lives elsewhere.

## Running experiments

Two stages, mirroring LLaVA/M3: **Stage 1** (feature alignment — LLM and ViT frozen,
only the elastic resampler/projector/LoRA train) then **Stage 2** (full visual
instruction tuning — LLM unfrozen). Stage 1 has a single `tok_level` (256, the
teacher), so it always runs that one forward per step. Stage 2 uses
`--n_sample_students 1` — teacher + one random student level sampled per step, not
the full 4-level grid every step (`n_sample_students=0` would be full-grid; this
trades some throughput for a lower per-step compute cost). Both stages log
per-level CE/KL/CORAL losses for whichever levels ran that step.

### Stage 1 — pretrain
```bash
bash scripts/v1_5/pretrain_elastic_slm.sh <LLM_KEY>
```
Single tok level (`--tok_levels 256`, `--lora_ranks 64`, `--freeze_backbone True`),
`image_aspect_ratio square`, `model_max_length 1024`, LR 1e-3, batch 16 ×
`GRAD_ACCUM` × `NUM_GPUS` (derived to keep the effective batch invariant to GPU
count — see the script's own comments before changing `NUM_GPUS`).

### Stage 2 — finetune
```bash
bash scripts/v1_5/finetune_elastic_slm.sh <LLM_KEY>
```
Full `--tok_levels 256 144 64 16` / `--lora_ranks 8 16 32 64` grid, LLM unfrozen,
full finetune (`--lora_enable False` — LoRA on the *LLM* itself was found to
destabilize at this scale, see the script's header comment), LR 2e-5, batch
2 × `GRAD_ACCUM` × `NUM_GPUS`. Reads its warm-start from Stage 1's output directory —
**the two stages' `--lora_ranks` must end at the same max value**
(see [docs/ELASTIC_PIPELINE.md §10](docs/ELASTIC_PIPELINE.md#10-bugs-found-and-fixed-in-this-pipeline-chronological-most-recent-first)
for what happens if they don't).

### Tagging a run
```bash
ELASTIC_RUN_TAG=v5 bash scripts/v1_5/pretrain_elastic_slm.sh tinyllama
ELASTIC_RUN_TAG=v5 bash scripts/v1_5/finetune_elastic_slm.sh tinyllama
```
Suffixes checkpoint/log/run-name dirs (`elastic-pretrain-tinyllama-v5`,
`elastic-finetune-tinyllama-v5`, ...) so a variant run never collides with or resumes
an existing one. Stage 2 reads the same tag to find its Stage-1 warm-start
(`ELASTIC_PRETRAIN_TAG` overrides this independently if you want Stage 2 to warm-start
from a *different* tagged Stage-1 run).

### Submitting as SLURM jobs
```bash
export ELASTIC_RUN_TAG=v5
sbatch run_job_slm.sh tinyllama          # Stage 1 + Stage 2, sequentially, one job
```
`run_job_slm.sh` defaults to A40:2 and 200h (Stage 2's full grid, 1 epoch on
665k samples, takes close to that on 2 GPUs — see the script's comment on why 150h
wasn't enough). To run on A10 nodes instead (needed when A40s are busy — verified
fine for both stages up to ~1.7B backbones):
```bash
sbatch --gres=gpu:A10:2 --nodelist=node208 run_job_slm.sh smollm2
```
`run_job_pretrain_slm.sh` is a Stage-1-only A10:2 wrapper (hardcoded to `tinyllama` —
edit the script or use `run_job_slm.sh` with `--gres` overrides for other backbones).

Chain evaluation after a training job automatically:
```bash
JID=$(sbatch --parsable run_job_slm.sh tinyllama)
sbatch --dependency=afterok:$JID eval_lmms_level.sh /var/scratch/skalra/flexllava/checkpoints/elastic-finetune-tinyllama-v5
```
**Note**: the launcher scripts have no `set -e`, so a crashed training stage still
exits the sbatch job with status 0 — an `afterok`-chained eval job will fire
regardless. Always confirm the checkpoint directory exists before trusting a chained
eval result.

## Evaluation

`eval_lmms_level.sh <checkpoint_dir>` submits one SLURM array job (4 tasks, one per
`tok_level`) via [lmms-eval](lmms-eval). `eval_lmms_slm_all.sh [checkpoint_root]`
submits it for every `elastic-*` checkpoint found under a root (or a single
checkpoint if pointed at one directly). Benchmarks: GQA, TextVQA, POPE,
ScienceQA-image, MME.

Once all 4 levels finish, get a combined accuracy + analytic-roofline-efficiency
table:
```bash
python3 scripts/summarize_eval.py elastic-finetune-tinyllama-v5
```
(wrap in a trivial sbatch job per the no-direct-python rule above — see
`jobs/cmp_otter2_phi2_eval.sh` for the pattern.)

## Elastic pipeline internals

Component-by-component breakdown — the resampler, nested LoRA, positional
embeddings, nested dropout, the (currently unused) projector-width nesting, the new
content-adaptive query-selection modes, and the full experiment/bug history — lives
in **[docs/ELASTIC_PIPELINE.md](docs/ELASTIC_PIPELINE.md)**. Read it before changing
anything under `llava/model/elastic/`.

## Experiment log

Summary only — see [docs/ELASTIC_PIPELINE.md §9](docs/ELASTIC_PIPELINE.md#9-experiment-history)
for the full table and every measured number.

- **v4**: reference baseline. Vision-tower LoRA and nested dropout both off. Every
  backbone tested (TinyLlama, Phi-2, SmolLM2) shows ~0 accuracy delta between 256
  and 16 visual tokens.
- **otter1/otter2**: an Otter-inspired parallel data pipeline (mixture reweighting,
  QA packing, per-source telemetry — `llava/data_otter/`) was built, verified, and
  run to completion. otter2 (the valid run) lost to v4 on held-out accuracy with no
  unexplained config difference. **Retired** — not used for further runs; the code
  is left in place, untouched, as a working reference.
- **v5** (current): vision-tower LoRA turned on, to test whether per-token-level
  encoder specialization is what the token-budget axis was missing. In progress.

## Relationship to M3

This repo is a fork of [Matryoshka Multimodal Models (M3)](https://github.com/mu-cai/matryoshka-mm)
— Cai, Yang, Gao, Lee, ICLR 2025 — itself built on [LLaVA](https://llava-vl.github.io/).
The original M3 training/inference path (7B Vicuna backbone, avg-pool token
reduction, `scripts/v1_5/{pretrain,finetune}.sh`, the Gradio/CLI serving stack under
`llava/serve/`) is untouched and still functional; see M3's own docs for it:
[Model Zoo](docs/MODEL_ZOO.md), [Data](docs/Data.md), [Evaluation](docs/Evaluation.md),
[LoRA](docs/LoRA.md), [Finetune_Custom_Data](docs/Finetune_Custom_Data.md). This
fork's additions (`llava/model/elastic/`, `*_elastic_slm.sh`, `llava/data_otter/`,
everything referenced above) are a separate, parallel pipeline layered on top via
module-level rebinding at launch time (`train_elastic.py`), not a modification of
M3's own code — `llava/train/train.py`, `llava_trainer.py`, and `conversation.py`
are unmodified by any of it.

## Citation

If you use this work, please cite M3, which this repo builds on:
```bibtex
@article{cai2024matryoshka,
  title={Matryoshka Multimodal Models},
  author={Cai, Mu and Yang, Jianwei and Gao, Jianfeng and Lee, Yong Jae},
  journal={Proceedings of the International Conference on Learning Representation},
  year={2025}
}
```

## Acknowledgement

- [Vicuna](https://github.com/lm-sys/FastChat) / [LLaVA](https://llava-vl.github.io/) / [M3](https://github.com/mu-cai/matryoshka-mm) — the codebase this fork is built on.
