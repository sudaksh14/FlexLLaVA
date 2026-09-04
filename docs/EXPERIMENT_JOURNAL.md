# FlexLLaVA Experiment Journal

Running log of experiments, decisions, and their outcomes. Newest session first.
Companion to [ELASTIC_PIPELINE.md](ELASTIC_PIPELINE.md) (how the code works) and the
[README](../README.md) (how to run it). This file is the *why* and the *what happened*.

---

# Session: 2026-08-19 → 2026-09-04 — "Otter to PARCEL"

**The question driving the whole session**: the elastic token-budget axis produces no
accuracy tradeoff. 16 visual tokens scores the same as 256 on every backbone tested.
Why, and what fixes it?

**Where it ended**: three candidate explanations tested and largely eliminated (data
mixture, vision-tower adapter capacity, token-count range), one strong new candidate
identified from the literature (the resampler architecture itself lacks a spatial
anchor), with a concrete plan to test it.

## 1. Decision log

Every decision below is the user's; the "result" column is what actually happened.

| # | Decision | Result |
|---|---|---|
| 1 | Build an Otter-inspired data pipeline as a **strictly parallel** addition — verification gate first, then telemetry, then mixture/packing, then throughput. Do not touch the working pipeline. | Built `llava/data_otter/` + `train_otter.py` + `otter_trainer.py`. Zero modified tracked files throughout — verified repeatedly with `git status`. Pipeline worked as designed. |
| 2 | Run TinyLlama pretrain + eval on the new pipeline; check A10 fit. | Stage 1 fit A10 comfortably (8,622 MiB of 23,028 MiB peak). Run completed → **otter1**. |
| 3 | Re-run corrected as **otter2** on A10s with `--otter_source_grouped_batches` ON, then chain FT + eval. | Completed. Stage 1 loss 2.3402 (vs v4's 2.26 — close). Stage 2 lost to v4 on eval. |
| 4 | Retire the Otter pipeline — "leave out otter data, it is useless for our use case." | Code kept in place, untouched, unused. All later runs use the original LLaVA mixture. |
| 5 | Correct my framing: v4's `vision_lora_enable=False` was **a deliberate ablation, not a bug**. Update memory. | Memory + script comments rewritten. v4/v5 now framed as a controlled A/B on one flag. |
| 6 | Turn vision-encoder nested LoRA **on** for all future runs, tag them **v5**, use original LLaVA data. | v5 launched for TinyLlama and SmolLM2. TinyLlama v5 finished and evaluated (§3). |
| 7 | Run SmolLM2 v5 on A10:2 (node208). | **OOM** on the first Stage-2 optimizer step. Moved to 1×A40 (node206); still running at time of writing. |
| 8 | Add all discussed top-k query-selection options as if/else branches in the resampler, **keeping current behavior as default**, without disturbing the running v5 pipeline. | Four modes implemented and unit-tested (`prefix` default, `magnitude`, `attn_energy`, `learned`). Running jobs unaffected. |
| 9 | Make `use_nested_dropout` **False by default**; write a repo-wiki doc; rewrite the M3 README as FlexLLaVA with our experiments and script instructions. | `ELASTIC_PIPELINE.md` + rewritten `README.md` shipped. |
| 10 | Set up a new experiment with an extended token range (16 → 576) and matching LoRA ranks. | **v6-tokrange** launched: `576 512 448 384 256 144 64 16` / ranks `2 4 6 8 8 16 32 64`. Stage 1 in progress. |
| 11 | Delete unwanted checkpoints, keeping only baselines and comparison points. | **~248 GB freed.** All deleted runs' eval numbers survive in `eval_logs/` and this journal. |
| 12 | Read the three adjacent papers and report what's adaptable. | PARCEL is the actionable one (§6, §7). |
| 13 | Create an experiment with a 7B LLaVA teacher; use node205. | **v7-kd7b** queued on node205 behind v6-tokrange (§5b). The smoke test caught a latent bug: the frozen teacher was never moved to GPU (§4). |
| 14 | Execute the v8 PARCEL plan; queue it on node208. | **v8-parcel** launched (job 27303, eval 27304). Unit tests + end-to-end smoke both pass. Two deliberate deviations from the plan below — see §8-note. |

## 2. The Otter arc (decisions 1–4)

**What was built**: a YAML-declared per-source mixture with resampling, multi-turn QA
packing, a pre-run verification gate, deterministic missing-image handling,
token-accurate length caching, and per-source × per-tok-level loss telemetry — all in
`llava/data_otter/`, swapped in at launch time by rebinding two module globals, so
`llava/train/train.py` was never edited.

**Measurements that changed the plan mid-flight**:

- *mix665k is already packed.* Records per distinct image: coco **4.11**, but gqa,
  ocr_vqa, vg, textvqa all exactly **1.00**. gqa/ocr_vqa/vg are already packed
  upstream at 10/5/10 QA pairs per record. The original packing plan (pack the short-QA
  sources) would have been a silent no-op; corrected to coco-only.
- *Per-source telemetry was near-useless under the stock sampler.* Simulated on real
  data: single-source micro-batch rate was coco 50.4%, gqa 13.2%, ocr_vqa 6.6%,
  textvqa 0.7%, vg **0.0%** — i.e. the sources the whole mixture hypothesis depends on
  were never measurable. Built `SourceGroupedLengthSampler` → 100% homogeneity, and
  padding waste dropped from 23.6% to 0.81% as a side effect.
- *Dataloader workers cost ~4 GB PRIVATE PSS each* (fork COW doesn't help — Python
  refcounting privatizes the record-list pages). Worker default corrected 12 → 4
  before it could OOM a node.

**Outcome**: otter1 was invalidated by a Stage-1 grad-accum bug of mine (effective
batch 64 instead of 256 → 8,721 steps instead of 2,180, final loss 3.07 vs v4's 2.26).
otter2 fixed it and ran clean, but lost:

| benchmark | otter2 | v4 |
|---|---|---|
| pope | 73.0 | 81.6 |
| textvqa | 15.4 | 23.1 |
| gqa | 47.8 | 52.3 |
| scienceqa | 46.4 | 51.3 |
| mme | 219.3 | 213.2 |

Stage-2 loss trailed v4 at every matched 400-step block (+0.106 early, narrowing to
+0.049 by step 2400) with no unexplained config difference — leading suspect is
`--otter_source_grouped_batches True` changing optimization dynamics. **Verdict:
retired.** The per-source telemetry did produce its headline number, though:
`otter/gap` (ce@16tok − ce@256tok) was ≈0 on *every* source including ocr_vqa and
textvqa, with real observation counts behind it. That killed the data-mixture theory
and pointed at the model, not the data.

## 3. The vision-LoRA ablation (decisions 5–7)

`--vision_lora_enable False` had been set in both launcher scripts for every run to
date — a deliberate ablation baseline (v4), not a misconfiguration. v5 turns it on.

**TinyLlama v5 vs v4, full eval:**

| benchmark | v5: 256/144/64/16 | v4: 256/144/64/16 |
|---|---|---|
| pope | 79.4 / 79.7 / 79.1 / 78.9 | 81.6 / 81.9 / 81.9 / 80.7 |
| textvqa | 18.6 / 18.4 / 18.2 / 17.7 | 23.1 / 23.0 / 22.6 / 19.9 |
| gqa | 50.9 / 50.8 / 50.7 / 50.3 | 52.3 / 52.0 / 52.0 / 51.2 |
| scienceqa | 49.4 / 49.3 / 48.9 / **49.7** | 51.3 / 51.4 / 51.5 / **44.1** |
| mme | 221 / 216 / 224 / **229** | 213 / 219 / 230 / **189** |

**Reading**: v5 is *consistently worse* than v4 on pope/textvqa/gqa at every level, but
removes v4's cliff at 16 tokens on scienceqa (44.1 → 49.7) and mme (189 → 229). Net:
**flatter, not better** — the opposite of what the hypothesis predicted. The adapters
*are* specializing (per-level LoRA norms `[0.0480, 0.0315, 0.0208, 0.0137]`, consecutive
distances `[0.0306, 0.0202, 0.0134]`, stable over hundreds of steps), so the mechanism
works — it just doesn't buy an accuracy/budget tradeoff.

**Phi-2 side finding**: `elastic-finetune-phi2-v4` badly underperforms the published
[TinyLLaVA-Phi-2](../TinyLLaVA_Factory/README.md) reference on the same backbone and
same-scale data — gqa 39.5 vs 59.4–62.1, textvqa 7.8 vs 53.4–60.3, pope 43–50 (chance)
vs 86.8–87.9. Verified not an answer-parsing artifact (`pope_yes_ratio` 0.50, textvqa
predictions are clean short answers). So phi-2 here is *undertrained on vision-language
alignment*, not capacity-limited — an open thread worth a Stage-1 log comparison.

## 4. Bugs found and fixed this session

All mine, all found before or shortly after they cost real compute:

| bug | cost | fix |
|---|---|---|
| Stage-1 grad accum 2 instead of 8 (otter pipeline) | invalidated otter1 (~30h) | derive `$(( 8 * 2 / NUM_GPUS ))` |
| `prepare_otter_cache.sh` hardcoded `model_max_length 2048` | Stage-1 length cache silently missed → heuristic fallback | parameterized `OTTER_MAX_LEN` |
| Dataloader workers defaulted to 12 | would have OOM'd a 125 GB node | default 4, measured and documented |
| `SourceGroupedLengthSampler` non-deterministic | caught by own unit test | seed a local `torch.Generator` from `(seed, epoch)` |
| **Stage-1/Stage-2 LoRA rank buffer mismatch** | crashed v5 Stage 2 at warm-start, 0 steps run | Stage 1 `--lora_ranks 64` must equal Stage 2's max |
| `run_job_slm.sh` has no `set -e` | crashed stage still exits 0 → `afterok` eval fires on a nonexistent checkpoint | documented; always verify the checkpoint dir exists |
| `eval_lmms_level.sh` hardcoded 4-entry `TOK_LABELS` | would silently produce empty labels for an 8-level checkpoint | read `tok_levels` from the checkpoint's `elastic_config.json` |
| **External KD teacher never moved to GPU** — `attach_kd_teacher` calls `from_pretrained` (lands on CPU) and, because the teacher is deliberately not a submodule so ZeRO-2 won't shard it, nothing else moves it either | every `--teacher llava` run would die at step 1 with `Expected all tensors to be on the same device`; latent since the path had never been run | best-effort `.to(device)` at attach + authoritative check-and-move in the mixin before the first teacher forward (the student is usually still on CPU at attach time) |

## 5. What's running now (2026-09-04)

| job | what | state |
|---|---|---|
| 27282 | SmolLM2 v5 Stage 2, 1×A40 node206 | 37% (1923/5197), ~100h left at 110 s/it |
| 27291 | TinyLlama **v6-tokrange** Stage 1+2, A40:2 node205 | Stage 1 14% (316/2180), ce@576tok 2.92 |
| 27299 | TinyLlama **v7-kd7b** Stage 2 only, A40:2 node205 | queued behind 27291 (§5b) |
| 27283 | SmolLM2 v5 eval (4 levels) | queued on 27282 |
| 27292 | v6-tokrange eval (**8 levels**, `--array=0-7`) | queued on 27291 |
| 27300 | v7-kd7b eval (4 levels) | queued on 27299 |

## 5b. Loss-term audit, and the 7B-teacher experiment it motivated (decision 13)

Auditing what the three elastic loss terms actually contribute:

```
loss = Σ_{l ∈ active}   CE_l / n_active
     + Σ_{l ∈ students} prefix_kl_weight · KL_l    / n_active
     + Σ_{l ∈ students} coral_weight     · CORAL_l / n_active
```
`n_active` = 2 in Stage 2 (`--n_sample_students 1`), 1 in Stage 1. `students` = active
levels except index 0 — *unless* an external teacher is loaded, which makes every level
a student.

| term | Stage 1 | Stage 2 | measured contribution |
|---|---|---|---|
| CE | weight 1.0 | weight 1.0 | **~99.99% of the loss** |
| prefix-KL | 0.1, **unreachable** (single level == teacher level) | 0.1 → effective 0.05 on one level | `loss/kl` 7.7e-5 (tinyllama) / 2.0e-4 (smollm2) = **0.006–0.014%** |
| CORAL | 0.01, **unreachable** (same reason) | `--use_coral False`, weight 0.1 inert | **exactly 0, in every run v3→v6 and both otter runs** |

So CORAL has never contributed anything, and prefix-KL is numerically invisible.
**Nothing currently ties the levels to one another** — each budget is trained on
independent CE with shared weights.

The near-zero KL is not a tuning failure, it is a *measurement*. With `teacher="self"`,
prefix-KL asks "does truncating 256 tokens to 16 change the output distribution?" and
answers **0.004**. That is the eval-level flatness, observed at the loss level, and it
independently corroborates PARCEL's diagnosis (§7): if the queries all encode the same
global summary, a prefix loses nothing, so there is nothing to distill.

**Decision 13**: run the external-teacher variant — `--teacher llava`, a frozen
LLaVA-1.5-7B — so the KL carries real signal for the first time, on node205.

- **Tag `v7-kd7b`**, **Stage 2 only**, warm-started from `elastic-pretrain-tinyllama-v5`
  (`ELASTIC_PRETRAIN_TAG=v5`). Identical Stage 1, identical Stage-2 config to v5 — the
  *only* difference is the teacher. Saves ~5h and keeps it a one-variable A/B.
- Standard 4-level grid (`256 144 64 16` / ranks `8 16 32 64`), not v6's extended ladder,
  for the same reason.
- **TinyLlama only**: `attach_kd_teacher` hard-fails on a vocab mismatch, and only
  Llama-32000 backbones (tinyllama, mobilellama) match Vicuna-7B. smollm2/qwen/phi
  cannot use this teacher.
- **A40 only**: the frozen 7B is a plain attribute on `ElasticEngine`, not a registered
  submodule, so ZeRO-2 does *not* shard it — ~13.5 GB replicated on every rank. Est.
  ~28–30 GB/GPU against the A40's 46 GB (comfortable), vs the A10's 23 GB (impossible).
- `prefix_kl_weight` left at 0.1 for the first run — changing teacher *and* weight
  together would confound. Watch the first few hundred steps: with a genuinely
  different teacher the KL should jump from ~0.004 to order 1, i.e. from 0.01% of the
  loss to a meaningful fraction. If it dominates, lower the weight; if it stays tiny,
  that is itself a strong result (the student's distribution already matches a 7B's).
- Teacher flags are now env-overridable in `finetune_elastic_slm.sh`
  (`TEACHER`, `TEACHER_MODEL_PATH`, `PREFIX_KL_WEIGHT`); default stays `self`.

## 6. Literature review — what's prior art, what isn't

**Nested/rank-truncatable LoRA is not novel.** The mechanism in
`llava/model/elastic/nested_lora.py` (shared `A`,`B`; rank-`r` adapter is the literal
prefix `A[:, :r] @ B[:r, :]`) is well-established:

- [DyLoRA](https://arxiv.org/abs/2210.07558) (EACL 2023) — closest and earliest: one
  LoRA block usable across a *range* of ranks via rank-slice sampling, explicitly
  described as "truncation inspired by nested dropout." Essentially our design.
- [NoRA](https://arxiv.org/html/2408.10280v1) (2024) — nested LoRA via dual-layer SVD.
- [MatryoshkaLoRA](https://arxiv.org/abs/2605.07850) (2026) — same name, same idea,
  **plus** a fixed diagonal matrix `P` between the adapters to make sub-rank scaling
  consistent. Their stated critique of naive truncation ("lack of consistent gradient
  signals across the full hierarchy of ranks") applies directly to our `alpha/r`-only
  implementation.
- [ElaLoRA](https://arxiv.org/pdf/2504.00254) (2025) — elastic/learnable rank allocation.

So: ours is *structurally* Matryoshka-style but is the **vanilla variant**, missing
MatryoshkaLoRA's diagonal-scaling correction. Claiming nested LoRA as a novel
contribution would not survive review. The narrower angle — coupling LoRA rank to a
*visual-token budget* — is unverified either way and would need the three papers below
read closely before any novelty claim.

**The three adjacent papers, read:**

| paper | relevance | verdict |
|---|---|---|
| [PARCEL](https://arxiv.org/html/2605.30126v1) | **high** | Directly diagnoses our failure mode and gives a fix. See §7. |
| [Dynamic Rank Adaptation for VLMs](https://arxiv.org/pdf/2507.05668) | low | Rank is *layer*-dependent, not budget-dependent; few-shot CLIP classification domain (CoOp/Co-CoOp baselines). Doesn't transfer. |
| [Selective LoRA for Visual Tokens (Image-LoRA)](https://arxiv.org/abs/2512.19219) | low/orthogonal | LLM-side LoRA restricted to visual-token positions and probe-selected heads. No elastic component. Doesn't fit our full-finetune Stage 2. |

## 7. PARCEL's diagnosis, and why it matters here

PARCEL splits elastic compression methods into two families and shows both are broken
in complementary ways:

- **Spatial-only (M3-style avg-pool)**: "spatial decimation lowers the representable
  Nyquist range" → spectral aliasing, fine detail blurs. Hurts resolution-sensitive
  tasks (ChartQA).
- **Query-only (MQT-style learned query bank)** — *this is us* — "forces the queries to
  encode both the low-frequency layout and fine-grained semantic details without an
  underlying spatial anchor." Hurts spatial grounding (RefCOCO: MQT 79.0% vs PARCEL
  80.5% retention at 16 tokens).

That second description is our `NestedQueryResampler` exactly. It also explains the
earlier job-26568 finding (queries collapse to effective rank ~12 of 256 without
positional embeddings): positional embeddings are a *soft* nudge toward spatial
specialization; PARCEL's argument is that queries need a *hard* spatial anchor that
they can't drift from, and that they should be told what the anchor already covers so
they spend themselves on the complement.

**Their fix — Pool-Conditioned Query Resampling (PCQR):**

1. **Pooled spatial anchors.** Budget-aware average pooling of the ViT patch grid
   (`2×2`/`4×4`) → `N_p` grid-aligned anchor tokens carrying low-frequency layout,
   deterministic, not learned.
2. **Query ↔ Pool self-attention.** Queries and anchors are concatenated and passed
   through a self-attention block, so queries become "pool-aware" — they know which
   spatial regions the anchors already cover.
3. **Semantic-explorer cross-attention.** The pool-aware queries then cross-attend to
   the *raw* ViT features: `Q_SE = CrossAttn(Q=Q_PA, K=X_v, V=X_v)`, recovering the
   high-frequency detail pooling threw away.
4. **Budget-aware routing.** `16 ≤ B < 64` → `4×4` anchors (`N_p=16`) + `B−16` queries;
   `64 ≤ B ≤ 256` → `8×8` anchors (`N_p=64`) + `B−64` queries.

**Their ablations, which is what makes this worth copying:**

- Ordering matters: sequential PCQR 95.6% > dual cross-attention 95.4% > ViT-only
  cross-attention 95.2% (at 256 tokens). "For the division of labor to work, the
  queries must be pool-aware."
- Routing matters: fixed `4×4` at all budgets → 90.2% at 256 (bad); fixed `2×2` → 95.6%
  at 256 but can't serve budgets below 64. Dynamic routing → 95.6 / 95.3 / 88.3 at
  256 / 64 / 16.
- Gains are architectural, not parametric: MQT *with the same added self-attention*
  reaches 93.3, M3 likewise 92.2, PARCEL 95.6.
- **Elasticity is real for them**: image retention 95.1 → 94.7 → 86.8 across
  256 → 64 → 16. Ours is flat to within ~1 point across the same range. That contrast
  is the single strongest piece of evidence we have about where the problem lives.

## 8. Concrete plan — adapting PARCEL to FlexLLaVA (proposed "v8-parcel")

Nothing below is implemented yet. It is scoped to be additive and opt-in, so v4/v5/v6
stay bit-reproducible.

### Design decisions specific to our stack

Our vision tower is CLIP-L/14-336 → a **24×24 = 576** patch grid, so the clean anchor
grids available by integer pooling are:

| pool kernel `k` | anchor grid | `N_p` |
|---|---|---|
| 24 | 1×1 | 1 |
| 12 | 2×2 | 4 |
| 8 | 3×3 | 9 |
| 6 | 4×4 | **16** |
| 4 | 6×6 | 36 |
| 3 | 8×8 | **64** |
| 2 | 12×12 | **144** |
| 1 | 24×24 | 576 |

Proposed routing over our v6 budget ladder (`576 512 448 384 256 144 64 16`):

| budget `B` | `N_p` (anchors) | `N_q` (queries) | note |
|---|---|---|---|
| 16 | 16 (4×4) | 0 | pure anchors, as in PARCEL's low-budget regime |
| 64 | 64 (8×8) | 0 | pure anchors |
| 144 | 64 (8×8) | 80 | |
| 256 | 64 (8×8) | 192 | matches PARCEL's high regime |
| 384 | 144 (12×12) | 240 | our extension above PARCEL's range |
| 448 | 144 | 304 | |
| 512 | 144 | 368 | |
| 576 | 576 (24×24) | 0 | identity — no compression, reference point |

`N_p` is monotone in `B`, so the anchor branch scales with budget exactly as PARCEL's
ablation says it must.

### Implementation steps

| step | file | change |
|---|---|---|
| 1 | `llava/model/elastic/config.py` | Add `resampler_arch: str = "query"` (`"query"` = today's behavior, `"pool_anchored"` = PCQR) and `anchor_routing: Optional[Dict[int,int]] = None` (budget → `N_p`, defaulting to the table above). Serialized into `elastic_config.json` automatically, so eval picks it up with no extra plumbing. |
| 2 | `llava/model/elastic/resampler.py` | Add `_pool_anchors(image_features, n_p)` — reshape `(N,576,C)` → `(N,C,24,24)`, `avg_pool2d` with the kernel from the table, flatten back. Logic already exists in `ElasticEngine._avg_pool`; factor it out rather than duplicating. |
| 3 | `llava/model/elastic/resampler.py` | Add a `pool_self_attn` block (`nn.MultiheadAttention` + LN + FFN, mirroring the existing layer construction) run over `cat([anchors, queries])`; take the query slice back out as `Q_PA`. |
| 4 | `llava/model/elastic/resampler.py` | In `forward`, branch on `resampler_arch`. `"pool_anchored"`: anchors → pool-self-attn → existing cross-attention stack with `Q_PA` as queries → return `cat([anchors, Q_SE])` of length exactly `n_tok`. `"query"` path untouched, line for line. |
| 5 | `llava/model/elastic/engine.py` | `reduce_tokens` passes the budget through unchanged; the split happens inside the resampler so nested dropout still applies to `N_q` only (anchors are deterministic and must not be dropped). |
| 6 | `llava/train/train_elastic.py` | `--resampler_arch` flag, default `"query"`; banner line. |
| 7 | `unit_tests_otter/` or a new `jobs/test_parcel_resampler.sh` | Shape tests at every budget in the table; assert output length == `n_tok` exactly; assert `"query"` mode is bit-identical to current behavior; assert anchors are a deterministic function of the input (same input → same anchors, no RNG). |
| 8 | `jobs/smoke_v8_parcel.sh` | 3-step Stage 1 + 3-step Stage 2 warm-start smoke, same pattern as `smoke_v6_tokrange.sh` (which caught nothing but cost 20 minutes and would have caught a lot). |

### Run plan

- **Stage 1 must be re-run** — the pool-self-attention block is new parameters, so v5's
  Stage-1 checkpoint cannot warm-start it. Budget ~5 h (TinyLlama, A40:2).
- Stage 2 ~40 h on A40:2. Tag `v8-parcel`. Chain an 8-level eval (`--array=0-7`).
- Keep everything else identical to v6-tokrange (same token ladder, same LoRA ranks,
  same LR/batch) so the *only* difference vs v6 is the resampler architecture.

### Decision criterion — set before the run

v4, v5, otter2 all show ≤ ~1 point spread between the 256-token and 16-token levels on
gqa/pope/textvqa (excluding v4's scienceqa/mme cliffs, which are a different failure).
**PARCEL succeeds here if the 576/256 → 16 spread exceeds ~3 points on textvqa and gqa
specifically** — the two detail-hungry benchmarks. Anything less and the flatness is
not coming from the resampler either, and the next suspect is the training objective
(prefix-KL measured at ~0.006, i.e. contributing essentially nothing, and CORAL is off).

### §8-note — as actually built (2026-09-04, job 27303 / eval 27304)

Implemented and running on node208 (A10:2). Two deliberate deviations from the plan
above:

1. **4-level ladder (`256 144 64 16`), not v6's extended one.** v6 won't finish for
   ~2 days, so using its ladder would mean changing two things with no comparison
   point available. Against v5 — finished and evaluated — v8 is a clean one-variable
   A/B on the resampler architecture alone.
2. **Anchor routing `{256:64, 144:36, 64:16, 16:4}`** — a flat ~25% anchor share —
   rather than PARCEL's literal thresholds. Their table (`N_p`=16 below B=64, 64 up to
   B=256) would give `N_q = 0` at *both* our 16 and 64 levels, degenerating half the
   ladder into plain M3 average pooling. This split keeps queries at every budget while
   preserving the property their ablation actually isolates: anchor resolution monotone
   in budget (`4 → 16 → 36 → 64`).

Everything else matches v5: same Stage-1 recipe, same LoRA ranks, same LR/batch,
`teacher=self`, CORAL off. Stage 1 must be re-run (the `pool_self_attn` block is new
parameters), so this is a full Stage 1 + Stage 2, ~5h + ~150h on A10:2.

Validation before launch: `jobs/test_parcel_resampler.sh` (pooling matches an explicit
block mean; `"query"` arch bit-identical and allocates no extra params; exact output
length at every budget; deterministic; gradients reach both branches; arbitrary
nested-dropout budgets fall back to a valid non-degenerate split; monotonicity;
config/routing validation) and `jobs/smoke_v8_parcel.sh` (3-step Stage 1 + 3-step
Stage-2 warm-start on real data) — both pass.

### Sequencing note

v6-tokrange is already answering a cheaper question (does the budget range simply not
extend high enough?) and finishes in ~2 days. Its result changes what v8 should be:
if v6 shows a real gap opening up at 384–576, the resampler is fine and the ladder was
just too short; if v6 is flat too — the likely outcome given everything else — then
PARCEL's diagnosis is the best remaining explanation and v8 should start immediately.
