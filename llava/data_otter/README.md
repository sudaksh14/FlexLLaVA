# Otter-style data pipeline

A **parallel** data pipeline and train script for FlexLLaVA Stage 2. The
existing recipe is untouched: `llava/train/train.py`,
`llava/train/train_elastic.py`, `llava/train/llava_trainer.py` and
`scripts/v1_5/finetune_elastic_slm.sh` are unmodified and keep driving the
current runs.

## Why it forks nothing

`train()` resolves `make_supervised_data_module` and `LLaVATrainer` as **module
globals at call time**, so `train_otter.py` simply rebinds those two names:

```python
m3train.make_supervised_data_module = otter_dataset.make_otter_data_module
m3train.LLaVATrainer = OtterTrainer
m3train.train(attn_implementation="flash_attention_2")
```

Everything else — tokenizer setup, `align_eos_with_template`,
`ensure_distinct_pad_token`, the `preprocess_v1` / `preprocess_mpt` round
accounting, the elastic attach hook, DeepSpeed, checkpointing — runs from the
original source. Copying that preprocessing would have put every hard-won
backbone-specific fix at risk of drift.

## Files

| file | purpose |
|---|---|
| `mixture.py` | YAML-declared per-source mixture, partitioning, Otter-style resampling |
| `packing.py` | merge QA pairs that share an image into one multi-turn record |
| `dataset.py` | the Dataset + collator; tags every sample with its source |
| `lengths.py` | token-accurate lengths for the length-grouped sampler (cached) |
| `manifest.py` | missing-image prescan, so bad samples are dropped not swapped |
| `telemetry.py` | per-source x per-tok-level accumulators |
| `sampler.py` | source-grouped batches, so per-source loss is observable at all |
| `verify.py` | the pre-run gate |
| `prepare.py` | CLI to build the two caches offline |
| `../train/otter_trainer.py` | `LLaVATrainer` + telemetry |
| `../train/train_otter.py` | launcher |

## Running

```bash
# 1. (optional, once per mixture+backbone) build the offline caches
sbatch scripts/otter/prepare_otter_cache.sh tinyllama configs/otter/mix665k_baseline.yaml

# 2. train  (the launcher runs the verification gate first and aborts on failure)
ELASTIC_RUN_TAG=otter1 sbatch run_job_otter_slm.sh tinyllama
ELASTIC_RUN_TAG=otter2 OTTER_MIXTURE_CONFIG=configs/otter/mix665k_ocr_heavy.yaml \
    sbatch run_job_otter_slm.sh tinyllama

# 3. eval  (the standard sweep globs elastic-*, so it SKIPS otter-* -- use this)
bash scripts/otter/eval_otter_all.sh
```

Gate on its own (CPU, seconds, exit 1 = do not train):

```bash
python -m llava.data_otter.verify \
    --mixture_config configs/otter/mix665k_baseline.yaml \
    --model_name_or_path TinyLlama/TinyLlama-1.1B-Chat-v1.0 --version v1
```

Tests: `sbatch jobs/test_otter_pipeline.sh` (imports, 17 unit tests, the gate on
real data, plus a negative control that must FAIL) and
`sbatch jobs/smoke_otter_train.sh` (8 real training steps on 1 GPU).

## Reading the telemetry

Per-tok-level loss (`loss/ce_tok256`, ...) already existed. What is new is
per-**source** attribution:

| metric | meaning |
|---|---|
| `otter/gap/<source>` | `ce(smallest budget) - ce(largest)`. **The headline number.** |
| `otter/gap_n/<source>` | observations behind that gap — it is noise at small n |
| `otter/hom_frac` | fraction of micro-batches that were source-homogeneous |
| `otter/ce/<source>/tok<N>` | per-source, per-level CE |
| `otter/supervised_frac` | supervised tokens / TEXT seq len — what packing should move. The denominator excludes the expanded visual tokens (an image is one placeholder at collate time), so compare it across configs rather than reading it as an absolute |
| `otter/share/<source>` | realised sampling share |
| `otter/forward_time`, `otter/between_forwards_time` | timing (the latter includes backward) |

**The decision `otter/gap` exists to make:** if the gap is ~0 on *every*
source, including ocr_vqa and textvqa, then the mixture is not what is
suppressing the token-budget tradeoff — the resampler is — and reweighting the
data is wasted compute. Run `mix665k_baseline.yaml` first and read this before
running `mix665k_ocr_heavy.yaml`.

CE is a token-mean over the batch, so per-source numbers are only collected
from source-homogeneous micro-batches; `hom_frac` says how often that held.

### Why batches are grouped by source

Under the stock `group_by_modality_length` sampler, simulated on the real 665k
mixture (`jobs/otterhom_26736.out`), the sources that matter are almost never
alone in a micro-batch — so the telemetry above would have been silent exactly
where the answer lives:

| source | stock hom rate | `gap_n`/window | source-grouped | `gap_n`/window |
|---|---|---|---|---|
| coco | 50.5% | 74 | 100% | 146 |
| gqa | 13.4% | 4 | 100% | 29 |
| ocr_vqa | 6.9% | 2 | 100% | 32 |
| textvqa | 0.8% | **0** | 100% | 9 |
| vg | 0.0% | **0** | 100% | 35 |

`sampler.py` therefore groups by (source, length) instead of (modality,
length). This is gradient-equivalent at the optimizer step, which averages 32
accumulated micro-batches and still sees a full mixture; only which samples
share a *forward pass* changes. Disable with `--otter_source_grouped_batches
False`.

It also turned out to be a throughput win: measured padding waste drops from
**23.6% to 0.81%** of token positions, because same-source samples have far
more similar lengths than length-matched cross-source ones.

## Measured facts worth not rediscovering

- **mix665k is already packed** for gqa / ocr_vqa / vg / textvqa: each has
  exactly 1 record per image, and the first three already carry 10 / 5 / 10 QA
  pairs per record. Only coco has headroom (4.11 records per image). Packing
  any of the others is a silent no-op. See `jobs/mixstat_26732.out`.
- **Each dataloader worker costs ~4 GB of PRIVATE memory** (fork COW does not
  hold — refcounting privatises the record list). 4 workers/rank is the proven
  setting; 12 would be ~107 GB on a 125 GB node. Do not raise it without
  redoing the arithmetic.

## Not done

- **Parquet / webdataset image storage.** Otter stores base64 images in parquet
  and loads them once into RAM, which removes per-sample filesystem hits. Doing
  that here means an offline conversion of ~600k files and a different read
  path; the manifest prescan is the cheap half of that fix, not the whole one.
- **Per-source `loss_weight`** is implemented but **off by default**
  (`--otter_loss_weighting`). It scales the batch loss by the mean of the
  batch's per-sample weights, which is exact only for a homogeneous batch, and
  it moves gradients the way a learning-rate change would.
