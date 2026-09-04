# FlexLLaVA Elastic Pipeline — Component Reference

Internal reference for the adaptive-token-budget ("elastic" / Matryoshka) VLM pipeline
built on top of M3 in this repo. Covers what each component does, where it lives in
code, and the experiment history behind the current defaults. For "how do I launch a
run," see the [main README](../README.md#running-experiments) — this doc is about
*how the pipeline works*, not how to invoke it.

## 1. The one axis that costs anything: `L_tok`

Everything here optimizes one thing: **the number of visual tokens fed to the LLM**
(`L_tok`, `ElasticConfig.tok_levels`, e.g. `[256, 144, 64, 16]`). That number directly
sets LLM prefill FLOPs and KV-cache size — it is the only axis in this design that is
actually a compute/latency lever. LoRA rank, positional embeddings, and the query
bank are *capacity/specialization* knobs, not compute knobs: they don't change how
many tokens reach the LLM, they change how good those tokens are for a given budget.
See `llava/model/elastic/config.py`'s module docstring for the original design note
on why there is no separate "encoder compute level" axis.

## 2. Component map

| Component | File | What it does |
|---|---|---|
| `ElasticConfig` | `llava/model/elastic/config.py` | All elasticity knobs in one dataclass: `tok_levels`, `use_lora`/`lora_ranks`/`lora_specialize_tok`, `use_pos_embed`/`pos_embed_type`, `query_selection`, `use_nested_dropout`, `use_prefix_kl`/`use_coral_align` + their weights, KD teacher settings. |
| `ElasticEngine` | `llava/model/elastic/engine.py` | Owns the resampler + projector + (if enabled) vision-tower LoRA wrappers. `reduce_tokens()` is the mode branch: pooling vs nested-query, and where nested dropout is applied. `attach_elastic_engine()` is the one-time setup hook called from the train script. |
| `NestedQueryResampler` | `llava/model/elastic/resampler.py` | Cross-attention token reducer: `P` patches → `n_tok` output tokens. This is where the *token count* actually shrinks. Also owns the four `query_selection` modes (§7) and the positional embeddings (§6). |
| `NestedProjector` | `llava/model/elastic/resampler.py` | `vision_dim → llm_dim` MLP. Has a working Matryoshka-linear-output-width mechanism (`widths=`) but it is **not used** — `engine.py` constructs it with `widths=None`. See §5. |
| `NestedLoRALinear` / `inject_nested_lora` | `llava/model/elastic/nested_lora.py` | Rank-nested LoRA: one `(lora_A, lora_B)` pair per adapted Linear, sized at `max(lora_ranks)`; `set_level()` slices a prefix of that fixed buffer at runtime. Applied to the frozen CLIP/SigLIP tower's `q/k/v/out_proj, fc1, fc2`. |
| `ElasticVisionTower` family | `llava/model/elastic/elastic_vision_tower.py` | An alternate, self-contained encoder-agnostic (CLIP/SigLIP) tower with LoRA injected at construction. **Not the path actually used** by `train_elastic.py` — see §4 note. |
| `CLIPVisionTower._encode` | `llava/model/multimodal_encoder/clip_encoder.py:47-92` | The tower actually used. `forward(images, l_enc=None)` calls `set_level(l_enc)` (injected by `attach_elastic_engine`'s fallback) before the (optionally gradient-checkpointed) CLIP forward. |
| `matryoshka_vis_token_process` / `prepare_inputs_labels_for_multimodal` | `llava/model/llava_arch.py` | Glue: resolves the LoRA level for this `l_tok`, calls the vision tower, then delegates token reduction to `engine.reduce_tokens` (or, with no engine attached, the original M3 avg-pool). |
| `LlavaElasticMixin.forward` | `llava/model/language_model/llava_elastic_mixin.py` | The training-time grid loop: iterates `l_tok` over the active levels, accumulates CE + prefix-KL + CORAL losses, populates `self._loss_components` (the per-level `loss/ce_tok{N}` etc. that `LLaVATrainer`/`OtterTrainer` log). |
| `LLaVATrainer` | `llava/train/llava_trainer.py` | HF `Trainer` subclass: length-grouped sampling, per-module LR groups, resolves and logs `_loss_components`, NaN/Inf loss guard. |
| `train_elastic.py` | `llava/train/train_elastic.py` | The launcher. Parses elastic-only flags, builds `ElasticConfig`, sets `m3train.ELASTIC_CONFIG`, calls the unmodified `llava/train/train.py::train()`. Prints the config banner every run. |

## 3. Full call chain, one training step

```
LlavaElasticMixin.forward()                                    llava_elastic_mixin.py:216
  for l_tok in active_levels:                                              :272
    self.forward_single_matryoshka(..., matryoshka_vis_token_scale=l_tok)  :282
      → self.prepare_inputs_labels_for_multimodal(...)          llava_arch.py:170
          l_enc = cfg.lora_level_for_tok(l_tok)   # which LoRA rank prefix to use   :184-189
          image_features = self.encode_images(images, l_enc=l_enc)                  :195
            → vision_tower(images, l_enc=l_enc)   # CLIPVisionTower.forward, clip_encoder.py:67
                → _encode(): self.set_level(l_enc)   # slices lora_A[:, :r], lora_B[:r, :]
                → full 576-patch CLIP forward, UNCHANGED shape regardless of l_tok
          image_features = self.matryoshka_vis_token_process(image_features, l_tok) :198
            → engine.reduce_tokens(image_features, l_tok)        engine.py:114
                n_tok = cfg.tok_levels[l_tok]
                if use_nested_dropout: n_tok = randint(1, n_tok)   # engine.py:122-134, §6
                reduced = self.resampler(image_features, n_tok=n_tok)   # §4 -- ACTUAL token-count reduction
                out = self.projector(reduced)      # vision_dim -> llm_dim, full width always, §5
      → self.model(inputs_embeds=...) → lm_head → CE loss        llava_elastic_mixin.py:62-86
```

**Two integration paths exist for vision-tower LoRA** — don't confuse them:
`ElasticVisionTower`/`build_elastic_vision_tower` (`elastic_vision_tower.py`) is a
complete, self-contained alternate tower with LoRA injected at construction and a
`forward(images)` signature that does *not* take `l_enc`. It is not what
`train_elastic.py` actually builds. What's actually used is the plain
`CLIPVisionTower` (`multimodal_encoder/clip_encoder.py`), which `attach_elastic_engine`
(`engine.py:227-239`) monkey-patches at runtime — injecting nested LoRA into it and
giving it a `set_level` method — only if it doesn't already have one. If you're
tracing vision-tower behavior and land in `elastic_vision_tower.py`, you're in the
wrong file for any run that has actually happened.

## 4. Token reduction: `NestedQueryResampler.forward` (`resampler.py`)

The **only** place the token count shrinks. Query-bank prefix slicing, not a
post-hoc truncation of a longer sequence:

```python
q = self.queries[:n_tok]              # only n_tok of the num_queries query slots
q = q.unsqueeze(0).expand(N, -1, -1)
for layer in self.layers:
    ... cross_attn(q, kv, kv) ...     # q cross-attends to ALL patches
return self.out_ln(q)                  # shape (N, n_tok, dim) -- n_tok IS the count
```
Because there are only `n_tok` queries, the cross-attention output has `n_tok` rows
directly — nothing is computed then discarded. `kv` is the full 576-patch CLIP
output every time; only the query side shrinks.

**Downstream shape is constant per backbone, only sequence length varies**: the
projector's output width (`llm_dim`) is fixed regardless of `l_tok` (§5), so what
actually reaches the LLM is `(batch, n_tok, llm_dim)` with `n_tok ∈ {256,144,64,16}`
and `llm_dim` = the backbone's `hidden_size` (2048 for TinyLlama/SmolLM2/StableLM/
MobileLLaMA/Qwen2.5-3B, 2560 for Phi-2, 3072 for Phi-3.5-mini, 896/1536 for
Qwen2.5-0.5B/1.5B).

## 5. The projector's Matryoshka width — implemented, not used

`NestedProjector.forward` (`resampler.py:360-374`) has a working nested-output-width
mode: for width `w < llm_dim`, it uses only `fc2.weight[:w, :]` and zero-pads the
rest, with gradient confined to the active prefix (not a post-hoc mask — true nested
subspace training, same idea as OpenAI-style Matryoshka embeddings). **`engine.py:88`
constructs it with `widths=None`**, so this path never executes in any run to date —
every visual token gets the same full-width projection regardless of `l_tok`.

**Do not revisit this as a way to save FLOPs** — it was proposed and rejected in
this exact form: the LLM's attention/FFN are dense matmuls over the fixed
`hidden_size`; a zero in a channel is still multiplied and summed like any other
value. Zero-padding only reduces *information*, not *compute*. The only lever that
reduces LLM FLOPs is shrinking the *token count* (§4), not the per-token width. If
this is revisited, it's as an accuracy/capacity experiment, not an efficiency one —
and it should be tested in isolation, not stacked on an unresolved v5.

## 6. Nested dropout — `ElasticEngine.reduce_tokens` (`engine.py:122-134`)

```python
n_tok = self.cfg.tok_levels[l_tok]
if (self.cfg.use_nested_dropout and self.resampler.training
        and l_tok != self.cfg.kl_teacher_tok_level):
    _rng = random.Random(self._fwd_step * 1315423911 + l_tok)
    n_tok = _rng.randint(1, n_tok)          # only reachable if use_nested_dropout=True
reduced = self.resampler(image_features, n_tok=n_tok)
```
Lives entirely in the engine — the resampler has no dropout logic of its own, it
just receives whatever `n_tok` it's given. Purpose (per `resampler.py`'s module
docstring): force every prefix length to be independently sufficient, which is
supposed to be what induces the queries' Matryoshka importance ordering. Never
applied to the teacher level (`l_tok == kl_teacher_tok_level`, always full nominal
width). Seeded from `self._fwd_step` (not the global RNG) so both DDP ranks draw the
same truncation and run identical collectives — this determinism fix is what
resolved a real NCCL desync (jobs 26187/26205/26209).

**Default is `False`** (`ElasticConfig.use_nested_dropout`, and both launcher
scripts pass it explicitly). This has been off in every run so far — v3, v4, v5,
otter1, otter2 — despite the docstring describing it as the mechanism that makes
short prefixes meaningful. It is an untested, independent candidate explanation for
the flat token-budget result, in the same family as §5's vision LoRA finding and
§8's projector-nesting gap.

## 7. Positional embeddings — `use_pos_embed` / `pos_embed_type`

**On in every run so far** (`--use_pos_embed True --pos_embed_type learned`, both
launcher scripts). Purpose, per measurement (job 26568, recorded in
`resampler.py:84-90`): with positional embeddings **off**, the 256 resampler output
tokens collapse to an effective rank of ~12 (TinyLlama) / ~21 (7B) out of 256 — mean
pairwise cosine +0.91/+0.72 — even though the query *parameters* stay near-orthogonal.
Spatially anonymous queries have no reason to specialize and all converge on the same
global summary, which is exactly why a 16-token prefix would lose nothing relative to
256 — a low-rank representation truncates just as well at any length.

Two additive terms (`resampler.py:216-221`, the default `"prefix"` branch):
```python
kv = image_features + self.patch_pos_embed[:, :P, :]      # anchors each PATCH spatially
q  = self.queries[:n_tok] + self.query_pos_embed[:n_tok]   # anchors each QUERY's ordering
```
`"learned"` (what's running): trainable `nn.Parameter`s, small random init.
`"sincos2d"` (never used in any run): a frozen 2-D sine-cosine grid shared by
queries and patches (the MQT-LLaVA design) — anchors query `(i,j)` to patch `(i,j)`
by construction, so queries can't collapse even in principle, rather than relying on
gradient descent to keep them apart.

**Open question, not yet re-measured**: turning this on is known to have fixed the
rank-12 collapse, but nobody has re-measured *what* rank the queries land at with it
on, or whether the first 16 are meaningfully different from the first 256. This is
arguably a bigger piece of the token-budget puzzle than the LoRA ablation and is a
cheap check against any v5 (or later) checkpoint.

## 8. Query selection modes — `query_selection` (added 2026-08-31)

`NestedQueryResampler.forward` (`resampler.py`), gated by
`ElasticConfig.query_selection` / CLI `--query_selection`:

| mode | behavior | new params | preserves FLOP reduction? |
|---|---|---|---|
| `"prefix"` (**default**, only mode any real run has used) | `queries[:n_tok]`, content-agnostic, fixed position | none | yes (identical to original) |
| `"magnitude"` | run the full query bank once, keep the `n_tok` outputs with largest L2 norm | none | yes |
| `"attn_energy"` | run the full bank, keep the `n_tok` tokens that pulled the most cross-attention mass from patches on the last layer | none | yes |
| `"learned"` | a linear head scores all `num_queries` outputs; keep top `n_tok`, rescaled by `sigmoid(score)` (the only gradient path into the head, since hard top-k blocks gradient into the ranking itself — an approximation, not a rigorous differentiable top-k) | `importance_head: Linear(dim, 1)` | yes |

All three non-default modes ruled out §5's failure mode by construction: they select
*which* `n_tok` tokens reach the projector/LLM (real sequence-length reduction), they
never widen or pad a fixed-width tensor. Nested-by-construction: since the ranking
comes from one full-bank forward, `top-16 ⊂ top-64 ⊂ top-256` automatically.

Not wired into any launcher script — opt in via `--query_selection magnitude` (etc.)
on `train_elastic.py`. Not yet run against real data; validated only against
synthetic tensors for shape/gradient/gate-inversion sanity (`jobs/test_query_selection.sh`).
Candidate next step if positional embeddings + vision LoRA don't close the elasticity
gap: static positional slicing may itself be why the budget axis has been inert,
regardless of encoder capacity.

## 9. Experiment history

| tag | vision LoRA | nested dropout | data pipeline | verdict |
|---|---|---|---|---|
| v3 / v4 | off (deliberate ablation baseline) | off | original llava mixture | reference baseline; flat token-budget accuracy on every backbone tested (TinyLlama, Phi-2, SmolLM2) |
| otter1 | off | off | Otter-inspired mixture pipeline (`llava/data_otter/`) | **invalid** — Stage-1 grad-accum bug ran effective batch 64 instead of 256, 4x more steps, final loss 3.07 vs v4's 2.26 |
| otter2 | off | off | same Otter pipeline, grad-accum fixed | valid run; Stage 1 close to v4 (2.34 vs 2.26); Stage 2 lost to v4 on 3/5 benchmarks (pope, textvqa, gqa) with no unexplained config diff — leading suspect is `--otter_source_grouped_batches True` changing optimization dynamics. **Otter pipeline retired** — code stays (parallel, untouched), not used for further runs. |
| v5 | **on** | off | original llava mixture | first vision-LoRA test. First attempt (job 27267) crashed at Stage-2 warm-start — see §10. Fixed and relaunched (job 27275 tinyllama, 27277 smollm2); in progress. |

Full narrative, decision log, and the forward plan live in
[EXPERIMENT_JOURNAL.md](EXPERIMENT_JOURNAL.md); this table is the quick summary. The otter pipeline itself
(`llava/data_otter/`, `train_otter.py`, `otter_trainer.py`, `configs/otter/`) is
still present and functional — it was a legitimate, verified build, just not the
direction being pursued now.

## 10. Bugs found and fixed in this pipeline (chronological, most recent first)

- **Stage-1/Stage-2 LoRA rank mismatch (2026-08-31)**: `NestedLoRALinear` allocates
  `lora_A`/`lora_B` at `max(lora_ranks)` once, at construction — `set_level` only
  slices a prefix, it never resizes. Stage 1 has one `tok_level`, so `--lora_ranks`
  must have exactly one entry, but that entry sets the buffer width for the whole
  adapter. `pretrain_elastic_slm.sh` used to pass `--lora_ranks 8`; Stage 2 wants
  buffer width 64 (`--lora_ranks 8 16 32 64`). Loading an 8-wide checkpoint into a
  64-wide model hard-crashes with a `size mismatch` on every vision-tower LoRA key.
  Fixed: Stage 1 now uses `--lora_ranks 64` (the value barely matters — Stage 1 only
  ever exercises index 0 — what matters is the buffer *width* matching Stage 2's).
  **If you ever change finetune's max rank, change pretrain's `--lora_ranks` to
  match, or this recurs.** Note also: `run_job_slm.sh` has no `set -e`, so a crashed
  training stage still prints "Job Complete" and exits 0 — a `--dependency=afterok`
  eval job will fire against a checkpoint that was never written. Always check the
  checkpoint directory actually exists after any change to the elastic config shape.
- **`vision_lora_enable=False` (v4)**: not a bug — a deliberate ablation baseline.
  See §9/v4.
- **`pad_token_id == eos_token_id`**: deleted every EOS from supervised labels on
  affected backbones (TinyLlama), so the model answered correctly but never learned
  to stop. Fixed by `ensure_distinct_pad_token` in `llava/train/train.py` (no-op for
  Vicuna-style tokenizers that already have a distinct pad token).
  See `llava/train/train.py`'s tokenizer setup sequence.
- **Diverged TinyLlama checkpoint / no gradient clipping**: an early full-finetune
  run (LR 2e-4, LoRA scale 2.0) diverged mid-epoch; root cause was
  `scripts/zero{2,3}.json` missing the `gradient_clipping` key, so `max_grad_norm`
  never reached DeepSpeed. Fixed in `scripts/zero2.json`; current hyperparameters
  (LR 2e-5, full finetune) documented with the incident in
  `scripts/v1_5/finetune_elastic_slm.sh`'s header comment.
- **LoRA finetune tokenizer save bug**: fixed in `train.py`; older
  `elastic-finetune-*` checkpoints predating the fix need tokenizer files copied up
  from their last `checkpoint-N/` subdirectory before eval.

## 11. Reading this alongside the code

Every file referenced above lives under `llava/model/elastic/`,
`llava/model/language_model/llava_elastic_mixin.py`, `llava/model/llava_arch.py`, or
`llava/train/`. None of `llava/train/train.py`, `llava_trainer.py`, or
`conversation.py` are modified by any of this — the elastic pipeline attaches to
them via `attach_elastic_engine` and module-level rebinding
(`train_elastic.py`/`train_otter.py` overwrite `m3train.make_supervised_data_module`
and `m3train.LLaVATrainer` before calling the unmodified `train()`), so the
original M3 codebase underneath stays intact and traceable on its own terms.
