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
| 15 | Build a paper motivation section: (a) peak-memory comparison, LLaVA-7B vs FlexLLaVA vs original TinyLLaVA/MobileVLM/SmolVLM, on node207; (b) vision-vs-LLM params/FLOPs/latency split for FlexLLaVA-TinyLlama. | Both measured (§9). Found and fixed a real inference-path gap along the way: bare `forward()` calls silently drop `matryoshka_vis_token_scale` outside training/`generate()`. MobileVLM/SmolVLM initially left unmeasured — package-version conflicts in the shared training env, not fixed in place to avoid risking the 4 jobs running in it. |
| 16 | Extend the vision-vs-LLM split (decision 15b) to LLaVA-7B, TinyLLaVA-Phi2, MobileVLM, SmolVLM too, not just FlexLLaVA. | All 5 measured (§9c), via a new isolated `flexllava-refs` conda env for MobileVLM/SmolVLM — zero changes to the shared `matryoshka-mm` env the 4 training jobs depend on. One number (TinyLLaVA-Phi2's LLM FLOPs) flagged as likely unreliable rather than reported at face value — see §9c. |
| 17 | Formalize two anchor-routing modes (fixed per-budget table vs always-ratio), default to ratio at 25%, use it now. | Implemented, unit-tested (§10a). `anchor_mode`/`anchor_ratio` on `ElasticConfig`, CLI flags, launcher env vars — default unchanged behavior unless set. |
| 18 | Queue SigLIP + TinyLlama + PARCEL on the full budget ladder, on node205 or node208 if it fits. | **v9-siglip-parcel** queued on node208 (job 27324, eval 27325), behind v8-parcel (~4h out at queue time). Found and fixed two real SigLIP bugs along the way — vision LoRA had simply never been exercised on SigLIP before (§10b). |
| 19 | Measure the visual-token rank story with real numbers (no pos-embed → pos-embed → PARCEL) for a paper section; also check whether raising vision-LoRA rank would help, and whether the current rank ladder has any grounding in the literature. | Measured (§11): pos-embed 130/256 (50.8%) vs the cited ~12/256 (4.7%); PARCEL splits into anchors 62.7/64 (98.0%, works as designed) vs queries 26.7/192 (13.9%, **worse** than the plain baseline, not better) — a real, non-obvious finding, written up with the caveat intact rather than only the flattering anchor number. LoRA-rank literature research done separately (not in the journal) — r=64 is defensible only as "above where sweeps go flat," not a validated optimum; one paper (LangVision-LoRA-NAS) suggests r=16 may be closer to a real sweet spot. Also found and fixed a real bug along the way: `anchor_routing`'s dict keys silently became strings on JSON reload (`ElasticConfig.__post_init__`), which would have broken v8-parcel's queued eval. |
| 20 | Check whether the rank measurement is dataset-dependent (COCO vs a more general/detail-hungry set); brainstorm (research only, no implementation) using rank info to condition anchor/query budget allocation on the input. | Measured on TextVQA too (§11a): content-dependence is real and consistently signed across all 8 conditions checked — COCO alone would have been a real gap, not a defensible simplification. Anchors are content-insensitive (already at their ceiling); queries remain content-responsive even under PARCEL's suppressed baseline. Four mechanisms brainstormed (§12), with the existing-but-unused `decorrelation_loss` (mechanism C) argued as the right first move given what §11/§11a actually show, ahead of building any new router. |
| 21 | Implement mechanism C (§12): wire up `use_token_decorrelation`/`decorr_weight`, off by default, applied to the QUERY tokens only. Turn it on for a new run using **CLIP**, not SigLIP, as the vision tower — a fair comparison against v4/v8-parcel (both CLIP), isolating the decorrelation effect from the SigLIP swap. | Implemented (§13). `engine.extra_losses` was dead code — the real per-level loop lives in `llava_elastic_mixin.py` and never called it, so `use_token_decorrelation` had literally never run despite existing on `ElasticConfig` since it was added. Wired a real per-level decorrelation term into that loop, added `ElasticEngine.query_tokens_for_decorr` (slices off the anchor prefix for `pool_anchored`, no-op for plain `query`), added `--use_token_decorrelation`/`--decorr_weight` CLI flags (unit-tested, job 27337, `ALL_TESTS_PASSED`). Cancelled the not-yet-started v9-siglip-parcel jobs (27324/27325 — still `PENDING(Resources)`, nothing lost) and requeued as **v9-parcel-decorr** (job 27338, eval 27339): CLIP vision tower (the launcher's own default — SigLIP was only ever an explicit override), `resampler_arch=pool_anchored`, `use_token_decorrelation=True`, same full ladder as v6/v8. |
| 22 | Queue a final rank re-measurement (COCO + TextVQA) for v8-parcel once it actually finishes training, so §11/§11a's paper numbers are taken against the FINAL checkpoint, not the 96%-trained checkpoint-5000 snapshot. Separately, run a literature check on §12's novelty claim (input-conditional anchor/query allocation). | Rank re-measurement queued as job 27340, `--dependency=afterok:27303` (v8-parcel's training job) — fires automatically on completion, auto-detects whichever checkpoint `save_total_limit=1` leaves behind rather than hardcoding a step number. Literature check done (§14): found the actual paper "PARCEL" is based on, confirmed it is NOT content-adaptive either, found one close routing precedent (AVG-LLaVA) that narrows but does not eliminate the claimed gap. |
| 23 | Research-only survey of adaptive input token budget allocation across the early-exit / MoE / cascade / speculative-decoding / KV-cache literature, including §12 mechanism D, with the KV-cache-under-escalation problem considered for every action point; write it up as a standalone doc with the actionable items loopable into this codebase. | [ADAPTIVE_INPUT_TOKEN_BUDGET_ALLOCATION.md](ADAPTIVE_INPUT_TOKEN_BUDGET_ALLOCATION.md). Key outcomes: (a) adaptive *total* budget per image is well-trodden (a dozen works, several already using effective rank as the router signal) — mechanism D is not a novelty claim; (b) no work found reusing KV across a budget escalation inside a VLM's LLM — the nearest are WaveCLIP (encoder-side causal cross-level attention), CacheBlend/VLCache (partial recompute), and LayerSkip/SpecVLM (make the cheap pass a draft); (c) for *this* stack the KV question is mostly moot: the budget-independent vision tower is 36% of FLOPs, so a 16-token pass costs ≈0.5× a 256-token pass, the cascade breaks even at ~50% escalation rate, and KV reuse can shave at most ~20% off the second prefill — deciding *before* the LLM (vision-side router) or making the cheap pass a speculative draft are the mitigations that actually change the economics; (d) `pool_anchored` as built is not nested across budgets (self-attention over the joint set + budget-dependent anchor grid), plain `query` is, and `lora_specialize_tok` breaks feature-level nesting for both — all config/architecture facts that gate which items are loopable. Eleven actionable items, ordered; the first three are zero-training analyses on existing eval logs (M3-style oracle gap, router-signal correlation, escalation cost model) that also settle the precondition nothing else survives without: whether the ladder has an accuracy gradient to trade on at all. |

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
| **Bare `forward()` silently drops `matryoshka_vis_token_scale` outside training/`generate()`** (`llava_elastic_mixin.py`'s "Standard eval / non-matryoshka forward" branch, reached whenever `self.training` is False and you call the model directly) — it calls `prepare_inputs_labels_for_multimodal` with no scale at all, so token reduction never runs and raw 1024-dim CLIP features get concatenated straight into 2048-dim LLM embeddings | not a training-path bug — `.generate()` (what the eval harness actually uses) and the training grid loop both handle it correctly. Bit two ad-hoc memory/latency probe scripts written this session that called `model(...)` directly instead | probes fixed to go through `.generate()` or to call `prepare_inputs_labels_for_multimodal` directly, matching what the supported paths already do. Documented here as a footgun for future direct-`forward()` scripts against elastic checkpoints, not something changed in the mixin itself. |

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

## 9. Paper motivation section: deployment memory + vision/LLM cost split (decision 15)

Two questions for a paper's motivation section, measured empirically on node207 (A10,
1 GPU) rather than by analytic roofline — scripts: `debug/measure_peak_memory.py`,
`debug/measure_vision_llm_split.py`. Each model in the memory comparison loads in its
own subprocess so a crash or version conflict in one third-party repo can't take out
the others. Raw data: `docs/peak_memory_comparison.csv`, `docs/vision_vs_llm_split.csv`.

### 9a. Why small backbones matter for on-device deployment

Peak memory of one forward pass (real image, short prompt), via `generate()` /
`prepare_inputs_labels_for_multimodal` directly — **not** a bare `forward()` call, see
the bugs table (§4) for why that distinction mattered here.

| model | family | params (B) | peak mem (GB) | tokens |
|---|---|---:|---:|---|
| LLaVA-1.5 | baseline | 7.06 | **14.87** | 576 (no elastic engine) |
| TinyLLaVA-Phi-2-SigLIP | original | 3.22 | **13.02** | 576 (SigLIP-so400m) |
| SmolVLM-Instruct | original | — | not measured | env gap, see below |
| MobileVLM_V2-1.7B | original | — | not measured | env gap, see below |
| FlexLLaVA-SmolLM2 (v4) | ours | 2.05 | 4.16 / 4.27 | 16 / 256 |
| FlexLLaVA-TinyLlama (v4/v6) | ours | 1.44–1.47 | **2.94 / 2.97 / 3.10** | 16 / 256 / 576 |

**Headline**: FlexLLaVA-TinyLlama is ~5x lighter than LLaVA-7B (2.97 GB vs 14.87 GB) —
comfortably inside an 8 GB Jetson Orin Nano with room to spare, where the 7B alone
consumes nearly the whole board.

**Sharper point, from TinyLLaVA-Phi-2**: a "small" VLM label doesn't guarantee a small
footprint — TinyLLaVA-Phi-2 measures 13.0 GB, almost as heavy as the 7B, because Phi-2
(2.7B) + SigLIP-so400m is still a substantial stack. FlexLLaVA-TinyLlama is ~4.3x
lighter than TinyLLaVA-Phi-2 specifically because 1.1B is a smaller commitment than
2.7B — **backbone size is what determines footprint, not the "VLM" label.**

**Token-count sensitivity** (the question that prompted extending this table to 576
tokens): 16→256 tokens costs ~0.9% more memory; 256→576 costs ~4.5% (≈5.4% total across
a 36x token-count range). Real, but small next to the backbone-size effect above — for
both TinyLlama and SmolLM2. This is itself a finding worth stating directly: **memory
is dominated by backbone size, not visual-token count**, which is *why* the token-budget
axis's real payoff has to be latency/FLOPs (§9b), not memory.

**Not measured**: MobileVLM_V2-1.7B (`ModuleNotFoundError: No module named
'timm.layers'`) and SmolVLM-Instruct (unrecognized processor class) both fail on
package-version mismatches in the shared `matryoshka-mm` training env. Deliberately not
fixed in place — four training jobs (27282/27291/27303/27299) were running in that same
env at measurement time, and a version bump to satisfy one comparison-table cell risked
destabilizing multi-day runs. Left as a documented gap in the CSV, not silently dropped;
fix in an isolated venv if these numbers are needed for the paper.

### 9b. Vision vs LLM: params, FLOPs, latency — FlexLLaVA-TinyLlama @256 tokens

FLOPs reuse the same `ElasticAnalyzer` / `vision_tower_gflops()` cost model the eval
harness already reports numbers from (`llava/eval/efficiency/`). Latency is real,
CUDA-synchronized wall-clock (median of 20, 3-call warmup discarded) on node207 — the
codebase's own `NOTICE.md` is explicit that the analytic roofline is not meant to be
read as wall-clock, so this deliberately isn't that.

| component | params (M) | GFLOPs | latency (ms) |
|---|---:|---:|---:|
| vision tower (CLIP-L/14-336) | 303.5 | 162.0 | 13.51 |
| resampler + projector | 32.6 | — | (in full timing) |
| LLM (TinyLlama, 256 tok) | 1106.3 | 508.4 | 18.23 |
| **full forward** | 1442.5 | 670.4 | 32.93 |

**Vision share: 21.5% params, 24.2% FLOPs, 42.6% latency.**

**Honest reading, not the one the question implied**: the LLM dominates, but not
overwhelmingly — roughly 3/4 of params and FLOPs, not 90%+. Vision's latency share
(42.6%) is notably larger than its FLOPs share (24.2%), meaning CLIP's forward is less
latency-efficient per FLOP than TinyLlama's at this scale (lower arithmetic intensity,
less kernel-level optimization, or both — not investigated further here). So "we only
optimize the LLM because vision is negligible" is not a defensible claim from these
numbers; vision is a meaningful fraction of the cost.

**The actually-correct justification, and a stronger one**: the vision tower's cost is
a *fixed constant regardless of token budget* — CLIP always runs its full 576-patch
forward no matter what `tok_level` is requested; nested LoRA changes *what* it computes,
not *how much*. The LLM's cost, by contrast, scales directly with visual-token count.
**Elasticity can only buy anything on the side where cost actually varies with the
budget — the LLM, by construction — not because the vision tower is small.** This row's
numbers would be identical at every tok_level; only the LLM row would change if
re-measured at 16 tokens. That's the sentence for the paper, not a magnitude argument.

### 9c. The same split across all 5 models (decision 16)

§9b's single-model numbers used the Llama-specific analytic roofline for LLM FLOPs,
which does not generalize to Phi-2 (TinyLLaVA's LLM has a parallel attention+MLP
block, not Llama's serial one). Redone with `debug/measure_vision_llm_split_multi.py`:
FLOPs via `torch.utils.flop_counter.FlopCounterMode` (real traced ops, architecture-
agnostic), vision measured directly, LLM+connector = full − vision for both FLOPs and
latency (params use the same subtraction). This supersedes §9b's FlexLLaVA-TinyLlama
row — same checkpoint, consistent method now, different numbers from the mix of
methods used before. LLaVA-7B and TinyLLaVA-Phi2 ran in `matryoshka-mm`; MobileVLM_V2
and SmolVLM needed a fresh, fully isolated `flexllava-refs` conda env — the shared
training env's timm/transformers were too old for either, and four training jobs were
running in it at the time, so it was never touched. Two dependency snags along the way,
both fixed: a missing `requests` import, and an unpinned `pip install` first resolving
to `torch 2.14.0+cu130` / `transformers 5.16.1` (a version whose API had moved past
`AutoModelForVision2Seq`) — pinned to `transformers==4.49.0` instead.

| model | vision params (M) | LLM params (M) | vision GFLOPs | LLM GFLOPs | vision lat (ms) | LLM lat (ms) | vision share: params / FLOPs / latency |
|---|---:|---:|---:|---:|---:|---:|---|
| LLaVA-1.5-7B | 303.5 | 6759.4 | 381.9 | 8543.3 | 48.6 | 175.0 | 4.3% / 4.3% / 21.7% |
| FlexLLaVA-TinyLlama (v4, 256 tok) | 303.5 | 1139.0 | 381.9 | 676.5 | 15.4 | 36.1 | 21.0% / 36.1% / 30.0% |
| MobileVLM_V2-1.7B | 303.5 | 1370.6 | 381.9 | 422.5 | 18.2 | 41.9 | 18.1% / 47.5% / 30.3% |
| TinyLLaVA-Phi-2-SigLIP | 428.2 | 2789.2 | 670.3 | **93.3** ⚠ | 72.5 | 37.1 | 13.3% / **87.8%** ⚠ / 66.1% |
| SmolVLM-Instruct | 413.0 | 1833.3 | 5998.0 | 3029.6 | 267.6 | 75.4 | 18.4% / 66.4% / 78.0% |

**⚠ TinyLLaVA-Phi2's LLM-FLOPs figure is flagged, not trusted.** SigLIP-so400m/384
with `connector_type=mlp2x_gelu` (no downsampling) produces **729 visual tokens —
more than CLIP's 576**, so if anything Phi-2 should see *higher* per-forward LLM FLOPs
than TinyLlama/Vicuna in the other rows, not 93.3 GFLOPs (an order of magnitude below
FlexLLaVA-TinyLlama's 676.5 GFLOPs on a *shorter*, 256-token sequence). Likely cause:
`FlopCounterMode` undercounting Phi-2's attention op if `transformers` routes it
through a fused/SDPA kernel that isn't traced the same way FlexLLaVA/LLaVA-7B's eager
attention is — not verified further. Its params and latency numbers came from direct
measurement (not subtraction) and are not suspect the same way; only the FLOPs split
for this one row should be treated as unreliable.

**Cross-model reading**:
- **Same vision tower, shrinking share as the LLM grows.** LLaVA-7B and
  FlexLLaVA-TinyLlama share byte-identical CLIP-L/14-336 numbers (303.5M params, 381.9
  GFLOPs) — confirms the measurement method is consistent — yet vision's *share*
  falls from 21.0%/36.1%/30.0% (params/FLOPs/latency) at 1.1B down to 4.3%/4.3%/21.7%
  at 7B. **The "vision isn't negligible" caveat from §9b is specifically an SLM-scale
  phenomenon.** At 7B, ignoring vision costs is far more defensible than at 1.1B,
  where it's already ~30–36% of the cost. This sharpens, not weakens, the case for why
  FlexLLaVA's SLM focus needs the elastic-token argument (§9b's actual justification —
  cost varies with budget on the LLM side only) rather than a "vision is small" claim.
- **MobileVLM_V2 is the cleanest reference point for what "SLM-scale, single-crop,
  simple connector" should look like** — its vision tower is the exact same CLIP-L/336
  as ours, its LLM (MobileLLaMA-1.4B) is close in scale to TinyLlama, and its FLOPs
  split (47.5% vision) is architecturally plausible (unlike TinyLLaVA-Phi2's ⚠ above).
  FlexLLaVA-TinyLlama's lower vision-FLOPs-share (36.1%) despite a *smaller* LLM
  (1.1B vs 1.4B) is consistent with running at only 256 tokens rather than MobileVLM's
  576 — exactly the elasticity axis this project is about.
- **SmolVLM is not directly comparable to the others** — its own default dynamic image
  tiling multiplies vision cost by tile count, which is why every one of its numbers
  (5998 GFLOPs, 267ms vision latency) is 5–15x larger than any single-crop model here.
  Its 78% vision-latency share is real for *its own* default settings, not a fair
  "SigLIP vs Llama-family" comparison against the other four rows.

## 10. v9-siglip-parcel: PARCEL with SigLIP, full ladder, formalized anchor modes

### 10a. Two anchor modes, "ratio" made the default (decision 17)

v8-parcel's anchor split (`{256:64, 144:36, 64:16, 16:4}`) was a hand-picked table
that happened to be exactly 25% at every entry. Formalized into two modes on
`NestedQueryResampler` / `ElasticConfig`, both requiring `resampler_arch="pool_anchored"`:

- **`anchor_mode="ratio"`** (new default): anchor count is always `anchor_ratio`
  (default **0.25**) of whatever budget is active, computed fresh for ANY budget —
  declared in `anchor_routing` or not — by snapping down to the nearest anchor count
  reachable by integer pooling. `anchor_routing` becomes an optional per-budget
  override rather than a requirement. This reproduces v8-parcel's exact table with
  **no table at all**, and extends cleanly to budgets it never covered (the full
  576..16 ladder).
- **`anchor_mode="fixed"`**: `anchor_routing` IS the routing table (raises at
  construction if empty) — a literal step function over the budgets you declare,
  PARCEL's own original design (constant within a band, not scaling continuously).
  An undeclared budget takes the value from the largest declared budget ≤ it, or the
  smallest declared entry's value if below all of them.

Both share the same non-degenerate-budget safety clamp (never spend the whole budget
on anchors, always leave ≥1 query token). Note this clamp is not just cosmetic: with
`anchor_mode="fixed"` and a routing table whose smallest entry equals the smallest
budget (e.g. `{64:16, ...}` used at budget 16), the raw lookup would consume 100% of
the budget — the clamp shrinks 16→9 in that case, which surprised a first version of
the unit test that hadn't accounted for it.

`--anchor_mode` / `--anchor_ratio` are new `train_elastic.py` flags and
`ANCHOR_MODE` / `ANCHOR_RATIO` env vars in both launcher scripts, defaulting to
`ratio` / `0.25` — nothing changes for existing runs unless set explicitly.
Validated in `jobs/test_anchor_mode_and_siglip.sh`.

**One real property surfaced by the unit test, not a bug**: on a 24×24 (576-patch)
grid only 8 anchor counts are reachable by integer pooling (1, 4, 9, 16, 36, 64, 144,
576), so `ratio` mode is itself a step function, not smooth — `round(budget × 0.25)`
for budget ∈ {384, 448, 512} all snap down to the same 64 anchors, and only 576
itself reaches the next rung (144). Worth knowing before reading too much into small
differences between adjacent high-end budgets in v9's per-level telemetry.

### 10b. SigLIP support — two real bugs found and fixed, not novel to this experiment

Building the SigLIP+PARCEL run surfaced two bugs in `SigLIPVisionTower`
(`llava/model/multimodal_encoder/siglip_encoder.py`) that had nothing to do with
PARCEL specifically — they'd have broken **any** attempt to run vision LoRA on
SigLIP, elastic or not, because that combination had simply never been exercised
before (every LoRA run to date used CLIP):

1. **`forward()` had no `l_enc` parameter at all**, unlike `CLIPVisionTower`'s. Since
   `encode_images` always calls the tower with `l_enc=...` once vision LoRA is
   attached, this fails loudly (`TypeError`) the instant training starts — safe, but
   a hard blocker. Fixed by mirroring CLIP's `_encode`/`forward` split exactly,
   including the gradient-checkpointing-safe re-application of `set_level` inside the
   (possibly checkpointed) function body.
2. **Blanket `@torch.no_grad()`** — the *exact* bug CLIP's own code comment already
   documents fixing (see §3/decision 6's era): with LoRA injected, `lora_A`/`lora_B`
   need a real autograd graph to receive gradients, and `no_grad()` silently blocks
   that. SigLIP still had it, apparently never ported when CLIP was fixed.
3. **(Found via the smoke test, not the unit test)** `inject_nested_lora`'s fallback
   in `attach_elastic_engine` (`engine.py`) matches by attribute name (`out_proj`)
   anywhere in the tower. `SiglipVisionModel` — unlike `CLIPVisionModel` — has an
   internal attention-pooling head (`vision_model.head`, a
   `MultiheadAttentionPoolingHead` wrapping a plain `nn.MultiheadAttention`) that
   *always* runs during forward regardless of whether the caller reads
   `pooler_output` (`SigLIPVisionTower` never does). Injecting into the whole tower
   also swaps that head's `out_proj`, and PyTorch's fused attention implementation
   reads `.weight`/`.bias` directly off it rather than calling it — a
   `NestedLoRALinear` wrapper has neither, so every forward crashed with
   `'NestedLoRALinear' object has no attribute 'weight'`. Fixed by scoping the
   injection to `vision_model.encoder` (the part actually used) when that nesting is
   present; CLIP has no such head, so it takes the unchanged path.

Also picked `google/siglip-base-patch16-384` over TinyLLaVA's
`so400m-patch14-384`: `384/14≈27.43` gives a 729-patch grid whose divisors
(1, 3, 9, 27) make anchor counts lumpy (only 1/9/81/729 reachable — no clean ~25% at
any of our budgets, see §9c's TinyLLaVA row for the same grid). `patch16-384` gives
`384/16=24` exactly — the identical 576-patch, highly-composite grid as CLIP, so the
existing token ladder, LoRA ranks, and anchor math all transfer with zero changes,
and the model is smaller (768-dim vs 1152-dim, faster to iterate).

`--vision_tower` is now `VISION_TOWER`-env-overridable in both launcher scripts
(previously hardcoded to CLIP inline) — needed to point at SigLIP without forking
the scripts.

### 10c. The run

**v9-siglip-parcel** (job 27324, eval 27325 with `--array=0-7`): SigLIP-base-patch16-384
+ TinyLlama, `resampler_arch=pool_anchored`, `anchor_mode=ratio` (default, 0.25),
same full ladder and LoRA ranks as v6/v8 (`576 512 448 384 256 144 64 16` /
`2 4 6 8 8 16 32 64`). Queued on node208 behind v8-parcel (95% done at queue time, ~4h
left) rather than waiting for a free node — SLURM's own GPU accounting handles the
sequencing. Stage 1 must be re-run (new vision tower entirely, nothing to warm-start
from). Validated first: `jobs/test_anchor_mode_and_siglip.sh` (anchor-mode math,
SigLIP `l_enc`/gradient flow, pooling-head regression guard — all pass) and
`jobs/smoke_v9_siglip_parcel.sh` (Stage 1 + Stage-2 warm-start on real data, node207).

**Smoke result**: passed, but not on the first try. The first attempt (job 27323) ran
Stage 1 cleanly (3/3 steps) then hit a **CUDA OOM in Stage 2 — before any real
training step ran**, inside DeepSpeed's own ZeRO optimizer-state initialization
(`torch.cuda.OutOfMemoryError` at `_multi_tensor_adamw`/`_foreach_sqrt`). Not a
code bug: this is the identical failure already documented for SmolLM2 on a single
A10 (§2/decision 7) — ZeRO-2 cannot shard optimizer state across 1 GPU, and this
config's ~1.47B params (TinyLlama + SigLIP + the new `pool_self_attn` block) doesn't
fit unsharded in 22GB. Deliberately smoke-tested on node207's single A10 for fast
turnaround while node208 (2 GPUs) was occupied by v8-parcel; the real run
(`ELASTIC_RUN_TAG=v9-siglip-parcel`) targets 2 GPUs, where sharding resolves this
without needing offload. Rather than accept "Stage 1 worked" as sufficient — the OOM
happened before Stage 2 ever exercised the actual new code (SigLIP `l_enc`
threading, the pool-anchored branch, the full ladder) — added
`scripts/zero2_offload_smoke_only.json` (CPU-offloaded optimizer state, ZeRO stage
unchanged at 2, so it validates forward/backward shapes identically to the real
config; explicitly not used by any real training script) and re-ran (job 27326).
Both stages then completed cleanly: Stage 1 loss 9.35→8.74→7.82, Stage 2 loss
4.93→6.04→3.61 with stable grad norms (89.4→72.6→66.8, no NaN/Inf) across all 3
steps — real confirmation that SigLIP's `l_enc` path, the scoped LoRA injection, and
PARCEL's anchor/query split all execute correctly together across the full 8-level
budget ladder before committing two A10s to a multi-day run.

## 11. Analyzing visual-token rank degradation under nested-query resampling

Methodology: `debug/measure_token_rank.py`. For a checkpoint, run the vision tower +
resampler + projector (bypassing the LLM) on N=24 real COCO images at the checkpoint's
largest trained tok_level (256, matching what job 26568 originally measured), take the
`(n_tok, llm_dim)` projected-token matrix for one image, center it, and compute: mean
pairwise cosine similarity (job 26568's own metric); numerical rank via SVD at two
thresholds (singular value > 1% / > 5% of the largest — "how many directions actually
carry signal"); and the number of singular values needed to explain 99% of variance
(scale-invariant). Reported as mean ± std across the 24 images. Job 26568's own script
no longer exists to diff against, so its number is cited, not reproduced with this
exact methodology — treat the *direction and magnitude* of the later comparisons as
the reliable part, not bit-for-bit comparability with the historical figure.

| stage | mechanism | rank @1% | rank @5% | rank (99% var) | mean cosine |
|---|---|---:|---:|---:|---:|
| **no positional embeddings** (job 26568, TinyLlama, *cited — checkpoint no longer exists*) | pure queries | **~12 / 256** | — | — | **0.91** |
| **positional embeddings on** (v4, measured this session, n=24 imgs) | pure queries | 130.0 ± 13.8 / 256 (50.8%) | 37.2 ± 5.4 (14.5%) | 57.2 ± 5.5 (22.3%) | 0.672 ± 0.028 |
| **PARCEL spatial anchors** (v8-parcel, checkpoint-5000, 96% trained, n=24 imgs) — **full set** | 64 anchors + 192 queries | 44.0 ± 4.7 / 256 (17.2%) | 7.8 ± 1.0 (3.0%) | 10.4 ± 1.9 (4.1%) | 0.521 ± 0.011 |
| PARCEL — **anchors only** | 64 deterministic pooled | 62.7 ± 0.9 / 64 (**98.0%**) | 40.5 ± 5.5 (63.3%) | 38.8 ± 3.3 (60.5%) | 0.515 ± 0.049 |
| PARCEL — **queries only** | 192 pool-conditioned learned | 26.7 ± 2.0 / 192 (**13.9%**) | 5.1 ± 0.4 (2.7%) | 5.5 ± 0.8 (2.8%) | 0.784 ± 0.012 |

**Reading, in order:**

1. **Positional embeddings closed most of the gap, not all of it.** Job 26568's ~12/256
   (4.7%) rank at no positional embeddings, jumping to 130/256 (50.8%) with them on, is
   a real and large effect — but the stricter thresholds (37/256 at 5%, only 22.3% of
   variance needed for 57 components) show the representation is still meaningfully
   redundant, not anywhere near full rank. This matches the eval-level finding that
   v5 (vision LoRA on top of this) still shows only a small 256-vs-16-token accuracy
   spread — the queries were never as diverse as 256 independent tokens would suggest.

2. **PARCEL's anchors are essentially full rank (98.0% at the 1% threshold) — the
   mechanism works exactly as designed.** They're literal averages of different
   spatial regions of the image, so distinctness is close to guaranteed by
   construction, not something that needed to be learned. This part of PARCEL's
   division-of-labour argument is directly validated by this measurement, independent
   of whatever the eventual eval numbers say.

3. **PARCEL's queries are* more* collapsed than the plain (non-PARCEL) baseline's, not
   less** — 13.9% rank vs 50.8%, and mean cosine *up* (0.784 vs 0.672, i.e. more
   redundant). This is the finding worth being careful with in the paper: it directly
   contradicts a naive "PARCEL fixes the collapse problem" narrative. A plausible
   mechanism (not confirmed, worth checking before publishing as an explanation): the
   pool-conditioning self-attention step has every query attend to the *same* shared
   64-anchor context, which could homogenize their updates rather than differentiate
   them — the opposite of what positional embeddings do for the plain-query baseline
   (nudge each query toward a *different* role). Not measured here: whether this
   homogenization is present from early training or grows over the run, or whether it
   would look different with `anchor_mode="fixed"` instead of `ratio`.

4. **The full-set numbers (44/256, mean cosine 0.521) are a blend that obscures both
   effects above** — lower rank than v4 looks like a regression until you see it's
   actually "excellent anchors + worse queries" averaging out to a lower number, not
   uniform collapse. Don't quote the full-set row alone without the breakdown; it
   reads as evidence against PARCEL when the real story is more specific than that.

**What this predicts for the pending eval** (job 27303 still finishing, eval 27304
queued): if v8-parcel's accuracy does show the budget/accuracy tradeoff v4/v5/otter2
lacked, this data says to attribute it to the anchors carrying real, non-collapsed
spatial information into the LLM — essentially M3's original pooling mechanism,
smuggled back in as half the budget — not to any improvement in the learned query
mechanism, which by this measure is doing its job worse than before. That would be a
more precise and more defensible claim for the paper than "PARCEL improves token
diversity."

**Caveats**: single checkpoint per condition (no seed variance across independent
training runs); v8-parcel measured at 96% trained (checkpoint-5000 of 5197 steps, the
current checkpoint at measurement time — `save_total_limit=1` rotates old ones away,
so this is whatever was latest, not a chosen point, and the number should be re-taken
against the final checkpoint once training completes for paper-final figures). v4 and
PARCEL *were* measured on the identical 24 COCO images (both runs used the script's
default `--seed 0`, so the same `random.sample` draw) — the comparison is apples to
apples on that axis at least.

### 11a. Does rank depend on the dataset? (decision 20) — yes, checked, not assumed

COCO alone risked understating the picture: it's mostly everyday photos, exactly the
"coarse, answerable from a global summary" content the mixture-analysis work earlier
this session (§2) identified as the majority of the training mix. Re-ran both v4 and
v8-parcel on 24 TextVQA images (same script, `--image_source textvqa`, same seed) —
dense signage/text, the "detail-hungry" contrast case that `otter/gap` telemetry (§2)
already flagged as the one place a real budget/accuracy tradeoff would have to show up
if it existed anywhere.

| condition | dataset | rank@1% | rank@5% | rank(99%var) | mean cos |
|---|---|---:|---:|---:|---:|
| v4 — full (pure queries) | COCO | 130.0 ± 13.8 | 37.2 ± 5.4 | 57.2 ± 5.5 | 0.672 ± 0.028 |
| v4 — full | **TextVQA** | **140.7 ± 13.7** (+8.2%) | 39.8 ± 5.9 | 64.8 ± 6.9 (+13.3%) | **0.592 ± 0.059** (−11.9%) |
| PARCEL — full | COCO | 44.0 ± 4.7 | 7.8 ± 1.0 | 10.4 ± 1.9 | 0.521 ± 0.011 |
| PARCEL — full | TextVQA | 46.1 ± 5.6 (+4.8%) | 8.0 ± 1.4 | 11.0 ± 2.1 (+5.8%) | 0.510 ± 0.013 (−2.1%) |
| PARCEL — anchors only | COCO | 62.7 ± 0.9 | 40.5 ± 5.5 | 38.8 ± 3.3 | 0.515 ± 0.049 |
| PARCEL — anchors only | TextVQA | 62.0 ± 2.4 (≈0%) | 41.3 ± 9.2 | 38.6 ± 6.1 (≈0%) | 0.490 ± 0.047 |
| PARCEL — queries only | COCO | 26.7 ± 2.0 | 5.1 ± 0.4 | 5.5 ± 0.8 | 0.784 ± 0.012 |
| PARCEL — queries only | TextVQA | 29.25 ± 2.4 (+9.6%) | 6.1 ± 0.9 | 6.6 ± 1.0 (+21.3%) | 0.773 ± 0.013 |

**Reading:**

1. **Content-dependence is real, and consistently signed** — every row shows TextVQA
   rank ≥ COCO rank and TextVQA mean-cosine ≤ COCO mean-cosine, no exceptions across
   8 conditions. COCO-only would have been a real methodological gap for the paper,
   not just a defensible simplification.
2. **The anchors are essentially insensitive to content** (62.7→62.0, 38.8→38.6,
   flat within noise) — expected, since they're deterministic spatial averages
   already near their 98% ceiling regardless of domain. There is no headroom left in
   the anchor half for image content to modulate.
3. **PARCEL's full-set content-sensitivity is smaller than the plain baseline's**
   (+4.8%/−2.1% vs v4's +8.2%/−11.9%) — but the **query subset alone** still shows a
   swing proportionally close to v4's own (+9.6% rank, +21.3% on the variance
   metric). The queries have not lost their ability to respond to content; PARCEL's
   architecture is suppressing their *absolute* level (§11's finding) while that
   *responsiveness* survives underneath it.

This directly informs §12's brainstorm: point 2 says routing effort toward the anchor
half is wasted (no headroom to condition on); point 3 says the query half is where
input-conditional capacity should go, and that fixing why it's suppressed should come
before building a router to work around the suppression.

## 12. Brainstorm: input-conditional anchor/query budget allocation (decision 20)

Research only, per the user's explicit instruction — nothing below is implemented.
The question: `n_anchors_for()` is currently a pure function of the *requested
budget*, identical for every image. Could the anchor/query split — or the total
budget itself — also condition on the *input*, using §11/§11a's rank measurements as
the signal?

**A. Cheap, non-learned router.** The frozen vision tower's raw patch embeddings are
already computed before pooling; a statistic like patch-feature variance or mean
pairwise patch dissimilarity is close to free (no new params, no extra forward pass)
and is a plausible proxy for "how spatially complex is this image." Use it to
interpolate `anchor_ratio` per image within a fixed total budget. §11a is the
experiment that would tell you which direction to move the knob — and per point 2
above, probably means moving budget INTO the query side for complex images, not the
anchor side, since anchors have no headroom to use it.

**B. Learned router.** Same idea as A, but the statistic-to-split mapping is a small
gating head trained jointly, with a load-balancing auxiliary loss on realized FLOPs if
average cost needs to stay bounded across a batch (standard MoE-style routing). More
powerful, more moving parts, harder to debug than A.

**C. Fix the mechanism before routing around it — highest priority per §11a.** The
codebase already has an unused, directly-relevant piece:
`ElasticConfig.use_token_decorrelation` / `losses.decorrelation_loss` ("penalize
redundancy among retained query tokens"), off by default, never turned on in any run
this session. §11's finding (PARCEL's queries collapse worse than baseline) plus
§11a's finding (they're still content-responsive, just suppressed) together argue for
trying this — on the query branch specifically — before B: a router only helps if the
queries it allocates more budget to can actually use that budget to diversify, and
right now that capability is demonstrably being left on the table.

**D. Genuine adaptive total-budget elasticity (more ambitious).** Rather than only
adapt the anchor/query split at a fixed total, let simple images use a smaller
`tok_level` altogether and complex images use a larger one — real per-image adaptive
compute, not a user-selected fixed level. Not a large architectural leap (the ladder
already supports any `tok_level` on demand), but the decision policy is the hard part,
and a low-budget pass's KV-cache is likely not reusable if escalation is triggered
(the visual tokens differ in count and content), so escalation eating into the savings
unless it's rare is a real risk, not just an implementation detail. Same family as
SmolVLM's own content-adaptive image tiling and the broader adaptive-computation
literature (early-exit, confidence-gated inference) — not compared against here.

**Novelty**: NOT verified against the literature. PARCEL's own routing is
budget-conditional only, not content-conditional, and I'm not aware of a paper doing
input-conditional anchor/query allocation specifically — but that's an absence-of-
search, not a checked claim, and should get the same literature pass the nested-LoRA
question got (§6) before anything above is asserted as novel in a paper.

## 13. Implementing mechanism C: query-branch decorrelation (decision 21)

**It was dead code.** `ElasticConfig.use_token_decorrelation` / `decorr_weight` and
`ElasticEngine.extra_losses` (which read them) have existed since early in the elastic
engine's history, but `extra_losses` is never called anywhere — the real per-level
training loop that actually accumulates CE/KL/CORAL is inlined directly in
`llava_llama.forward` (`llava/model/language_model/llava_elastic_mixin.py`), and it
never called `extra_losses` either. So `use_token_decorrelation=True` would have been a
silent no-op in every run this session, not merely "off by default" — worth flagging
since the config field's existence could otherwise be mistaken for "tried, didn't help."

**What changed:**
- `ElasticEngine.query_tokens_for_decorr(tokens, n_tok)` (`llava/model/elastic/engine.py`):
  for `resampler_arch="pool_anchored"` slices off the leading anchor block (via the
  same `n_anchors_for` the resampler itself uses, so the split always matches what that
  forward pass actually produced — important because nested dropout can truncate a
  non-teacher level below its nominal budget); for `resampler_arch="query"` it's a
  no-op (every token is already a query). Directly implements §12 mechanism C's
  "on the query branch specifically."
- A real decorrelation term added to the per-level loop in `llava_elastic_mixin.py`,
  gated on `cfg.use_token_decorrelation`, applied at **every** active level (teacher
  included — collapse is a per-level property of the resampler output, not something
  specific to distillation, unlike KL/CORAL which are inherently teacher/student
  comparisons). Logged per-level as `loss/decorr_tok{N}` plus an aggregate
  `loss/decorr`, health-checked like every other loss term.
- `--use_token_decorrelation BOOL` (default `False`) and `--decorr_weight FLOAT`
  (default `0.01`, matching `coral_weight`'s scale) added as CLI flags in
  `train_elastic.py`, wired into the `ElasticConfig(...)` construction and the printed
  config banner; `USE_TOKEN_DECORRELATION`/`DECORR_WEIGHT` env-var overrides added to
  both `pretrain_elastic_slm.sh` and `finetune_elastic_slm.sh` (applied in both stages,
  not just finetune, so the fix is present for all of warm-starting Stage 2, not
  introduced only after the resampler is already partially trained collapsed).
- Validated end-to-end before queuing anything: `jobs/test_decorr_loss.sh` (job 27337,
  `ALL_TESTS_PASSED`) checks the anchor/query slice is correct in both resampler_arch
  modes, that `decorrelation_loss` actually backprops a gradient, that a collinear
  (collapsed) token set hits the loss's theoretical max (1.0) versus a random
  full-rank set (0.242) — confirming the term responds in the right direction — and
  that the new CLI flags parse with the correct default/override values.

**The run.** v9 was still queued behind SigLIP (job 27324/27325, both still
`PENDING(Resources)` — never started, no node ever allocated) when this landed. Since
comparing decorrelation's effect against v4/v8-parcel (both CLIP) is a much cleaner
read than compounding it with the SigLIP swap, cancelled 27324/27325 and requeued as
**v9-parcel-decorr** (job 27338, eval 27339, `--array=0-7` matching the 8-level ladder):
`--vision_tower` left unset so the launcher's own default (`openai/clip-vit-large-
patch14-336`) applies — SigLIP was only ever an explicit override in the old
submission, never the script default — `resampler_arch=pool_anchored`,
`anchor_mode=ratio` (0.25, the decision-17 default), `use_token_decorrelation=True`,
`decorr_weight=0.01`, same full ladder and LoRA ranks as v6/v8/the old v9
(`576 512 448 384 256 144 64 16` / `2 4 6 8 8 16 32 64`). Queued behind whichever of
node205/206/208 frees up first — all three were 96-97%+ through their current runs at
queue time.

**What this run should tell us, tied back to §11/§11a**: if mechanism C is doing what
§12 argued, `debug/measure_token_rank.py`'s query-only rank on this checkpoint should
land meaningfully above v8-parcel's 13.9%/26.7 — plausibly approaching or exceeding
v4's plain-query 50.8%/130.0, since decorrelation pressure is now applied exactly
where §11 found the collapse concentrated. Re-run the same measurement (COCO + TextVQA,
full/anchor/query split) once a checkpoint exists, before drawing the paper table
conclusion — this is a prediction from the brainstorm, not yet an observed result.

## 14. Literature check on §12's novelty claim (decision 22)

§12 flagged its own claim ("I'm not aware of a paper doing input-conditional
anchor/query allocation") as an absence-of-search, not a checked one, and asked for
the same literature pass the nested-LoRA question got (§6). Ran that pass. Caveat up
front, same as §6: this is one search agent's pass, not a systematic review — treat
the "genuine gap" read as a working hypothesis for the paper's positioning, not a
settled claim, and independently re-verify every arXiv ID below before citing.

**Found the actual base paper.** "PARCEL: Pool-Anchored Resampling with Conditioned
Elastic Queries for Efficient Vision-Language Understanding" (arXiv:2605.30126, MPI-
Informatik/Google/TUM authors) — fetched directly, not just search snippets. Confirms
our own characterization exactly: anchor/query split is a **deterministic function of
budget B alone** (4×4 pooled grid below B=64, 8×8 at B≥64, queries fill the
remainder), identical across every image at a fixed B. Nothing in the paper does
content-conditional routing, per-image budgets, or MoE-style gating. Good news for
positioning: we now have the real paper to cite and contrast against directly, instead
of arguing from our own reimplementation's behavior.

**Adaptive TOTAL token count per image is a real, populated sub-area** — this axis is
NOT open: LLaVA-PruMerge/PruMerge+ (arXiv:2403.15388, CLS-attention-outlier-driven
count), HiRED (arXiv:2408.10945, AAAI 2025, attention-guided per-partition budget under
a global cap), AVG-LLaVA (arXiv:2410.02745, learned router over discrete pooling
granularities, conditioned on image **and instruction**), DOVE (arXiv:2506.03643,
tokenizer length correlates with image complexity), Adaptive-VoCo (arXiv:2512.18496,
patch-entropy/attention-dispersion-driven compression rate). By contrast PyramidDrop
(arXiv:2410.17247) and the fixed-ratio pruning line (FastV, TokenPacker, VisionZip)
apply the same schedule to every image regardless of content — not adaptive on this
axis at all. This matters for how §12-mechanism-D ("genuine adaptive total-budget
elasticity") gets framed in the paper: it should be positioned as re-deriving a known
result in our own architecture, not as a novel contribution on its own.

**The specific gap §12 cares about — content-conditioning the INTERNAL COMPOSITION of
a fixed budget (anchor vs. query ratio within a hybrid pooled+learned resampler) —
looks narrower but still real.** None of the adaptive-count papers above have a
two-mechanism split to condition in the first place (they have one pooling/pruning
mechanism whose total output size varies). The nearest actual precedent is **AVG-LLaVA**:
a learned router picks among several discrete pooling *granularities* per image, which
is routing, but over one mechanism, not a coarse/fine ratio within a hybrid — and it
needs the text instruction alongside the image, whereas §12's mechanisms A-C are
image-only. One flagged-as-unverified partial precedent: **PruMerge+** merges
attention-selected tokens with a spatially-uniform grid complement, which structurally
resembles an anchor+salient-token hybrid — but whether *that specific ratio* is itself
content-adaptive (vs. just the total count) wasn't confirmed from search alone and
needs a closer read before being cited either way. No paper was found doing MoE-style
routing with a load-balancing loss specifically over token TYPE (pooled-anchor vs.
cross-attended-query) — §12 mechanism B's closest literature analog — but this was a
breadth-first pass, moderate rather than high confidence on that absence specifically.

**Novelty verdict**: defensible as a paper claim, but narrower than "input-conditional
budget allocation" in general (that part is not novel) — the honest framing is
"content-conditioning the *composition* of a fixed budget inside a pool-anchored
hybrid resampler specifically," positioned explicitly against AVG-LLaVA (nearest
routing analog, single-mechanism + needs instruction) and against the adaptive-total-
budget literature (orthogonal axis: those vary B, mechanisms A-C here fix B and vary
its composition). A handful of very-recent 2026 arXiv-only hits (AsymVLM, OccamToken,
COAST, E-AdaPrune) came from search snippets only, not fetched directly — treat as
"worth checking before submission," not yet confirmed either way.
