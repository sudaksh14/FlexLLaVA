# Elastic / Adaptive Matryoshka VLM

A drop-in extension of M3 with **one** elasticity axis you actually pay for —
the visual-token count `L_tok` — and a switch between the original pooling
reducer and a new pooling-free nested-query reducer. LoRA is available as an
optional capacity/specialization tool, **not** a compute axis.

## Why there is no separate encoder-compute axis

Nested LoRA on a frozen full-width ViT does not save vision FLOPs (the whole
backbone still runs) and does not change the token count. So an "encoder level"
was a non-axis and was removed. The only knob that changes cost is `L_tok`,
which sets the LLM prefill / KV length — the dominant edge cost.

## Single axis: `L_tok`

| `token_reduction` | how `P` patches -> `n_k` tokens | notes |
|-------------------|---------------------------------|-------|
| `pooling`         | avg-pool the patch grid (M3)    | original baseline |
| `nested_query`    | first `n_k` of a learned query bank | pooling-free, ordered prefix |

## LoRA: optional specialization, off by default

```python
# baseline (recommended start): frozen encoder, no LoRA
ElasticConfig(token_reduction="nested_query", tok_levels=[256,144,64,16])

# pure M3
ElasticConfig(token_reduction="pooling", tok_levels=[576,144,36,9,1])

# LoRA tied to the token budget: each granularity gets tuned features
ElasticConfig(token_reduction="nested_query", tok_levels=[256,144,64,16],
              use_lora=True, lora_specialize_tok=True, lora_ranks=[8,16,32,64])
```

When `lora_specialize_tok` is on, the adapter level is set per `L_tok` *before*
the encoder forward, so each token budget sees features tuned for it (coarse ->
global/semantic, fine -> detail). With `use_lora=True, specialize=False` it is a
single shared task adapter.

## Losses (nested-query method)

- **prefix_kl** — full-token output is the teacher for truncated outputs
  (coarse-to-fine). Wired in the training loop.
- **coral_align** — match mean+covariance of projected tokens across levels →
  latent-space stability as `L_tok` changes. (available via `extra_losses`;
  needs projected tokens plumbed through — see below.)
- **nested_dropout** — random truncation each step induces the query ordering
  (applied in `engine.reduce_tokens`, not a loss term).
- optional: **recon**, **token_decorrelation**.

## Turning it on

```python
from llava.model.elastic import ElasticConfig, attach_elastic_engine
attach_elastic_engine(model, ElasticConfig(token_reduction="nested_query",
                                           tok_levels=[256,144,64,16]))
# omit the call entirely for pure M3
```

## Wiring (already applied to the repo)

- `llava_arch.matryoshka_vis_token_process` → delegates to `engine.reduce_tokens`
  when an engine is attached; original avg-pool otherwise.
- `llava_llama.forward` (training) → loops `engine.grid()` (1-D over `L_tok`),
  sets the LoRA level per level, averages LM loss, adds prefix-KL.

## Frontier & tests

```bash
cd matryoshka-mm
python -m pytest llava/model/elastic/tests/test_elastic.py -q
```
`flops.grid_cost(cfg)` → GFLOPs per token level for the accuracy-per-FLOP plot.
