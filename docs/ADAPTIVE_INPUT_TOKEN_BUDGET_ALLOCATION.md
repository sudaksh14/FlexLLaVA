# Adaptive Input Token Budget Allocation — literature survey and actionable items

Status: **research only, nothing here is implemented** (2026-09-08). Companion to
[EXPERIMENT_JOURNAL.md](EXPERIMENT_JOURNAL.md) §11–§14 (the rank measurements and the
brainstorm this expands) and [ELASTIC_PIPELINE.md](ELASTIC_PIPELINE.md) (how the code
works). Written for two readers: someone deciding *which* of these to try next, and
someone writing the related-work paragraph.

Confidence conventions used throughout: **[fetched]** = arXiv abstract/paper page read
directly; **[snippet]** = characterized from search results only, re-verify before
citing; **[known]** = standard, widely-cited work not re-verified here. Every arXiv ID
should be checked against the actual paper before it goes into a submission.

---

## 0. The question, and why it has a hidden precondition

Today the token budget is a *user-selected* `tok_level` (fixed per eval run, via
`config.matryoshka_vis_token_scale` in `lmms-eval/lmms_eval/models/llava_elastic.py`),
and inside a budget the anchor/query split is a pure function of the budget
(`NestedQueryResampler.n_anchors_for`). The question this document surveys: can the
budget — total tokens, and/or the anchor/query composition — be decided **per input**,
automatically, and what does that cost once the KV cache is accounted for?

**Precondition that is not yet met.** Every adaptive-budget policy trades tokens for
accuracy. Through v5, this codebase's ladder produced **no** accuracy gradient across
budgets (16 tokens ≈ 256 tokens on every backbone — journal §2/§3; memory note
"token budget has no effect"). A router or cascade has nothing to trade on a flat
ladder: the optimal policy is trivially "always pick the cheapest level," and no
mechanism below can look good or bad. v8-parcel (job 27303) and v9-parcel-decorr (job
27338) are the runs that must first show a real tradeoff. **Everything below is
sequenced on that.** Until then the only items worth doing are the zero-training
measurements in §5 (A1, A2), which also tell you whether a tradeoff exists.

---

## 1. Facts about our stack that constrain what is loopable

Verified against the code, because they decide which literature mechanisms transfer:

1. **Sequence layout.** Visual tokens replace the single `IMAGE_TOKEN_INDEX` in the
   prompt (`llava_arch.prepare_inputs_labels_for_multimodal`): `[system prompt] [B
   visual tokens] [question] [answer…]`. Under causal attention, KV entries of a token
   depend only on tokens *before* it. So after any change to the visual block, the
   question's KV is stale; the system prompt's KV is not.
2. **Plain `query` resampler (prefix selection) is budget-invariant per token.**
   `NestedQueryResampler.forward` runs cross-attention (queries → patches) + per-token
   FFN/LayerNorm only — queries never attend to each other. Output token *i* at budget
   B is bit-identical to output token *i* at budget B' > B, given the same patch
   features. The first-B outputs at a high budget ARE the low-budget outputs: nested
   by construction, so escalation could be an *append*.
3. **…but `lora_specialize_tok=True` breaks that across budgets.** The vision tower is
   re-run with a different nested-LoRA rank per `tok_level`
   (`llava_arch.py:189`, `cfg.lora_level_for_tok`), so the patch features — and hence
   every resampler output — differ across budgets. Feature-level nesting only holds
   between budgets that share a LoRA level, or with `lora_specialize_tok=False`
   (shared adapter). This is a config choice, not an architectural one.
4. **`pool_anchored` (PARCEL) is not nested at all.** `_forward_pool_anchored` runs
   full self-attention over `[anchors; queries]` (every output depends on the whole
   set) and the anchor grid itself changes with budget (`n_anchors_for`: 64 anchors at
   B=256, 16 at B=64…). Nothing in the low-budget visual block survives escalation.
5. **The vision tower is budget-independent and large.**
   [vision_vs_llm_split.csv](vision_vs_llm_split.csv): FlexLLaVA-TinyLlama @256 tokens
   = 381.9 GFLOPs vision + 676.5 GFLOPs LLM (vision = 36% of FLOPs, 30% of latency).
   The elastic axis only shrinks the LLM's 64%. With ~50 text tokens, a 16-token pass
   costs ≈ 382 + 676·(66/306) ≈ 530 GFLOPs ≈ **0.50×** the 256-token pass — not 1/16.
   Consequences: (a) a low-then-escalate cascade breaks even at an escalation rate of
   only ≈ 1 − 0.50 = **50%**; (b) caching the encoder output (patch features) across an
   escalation is worth as much as any KV trick, and is only possible under fact 3.
6. **Per-level eval outputs already exist** (`eval_logs/`, one lmms-eval run per
   `tok_level`), so per-sample "which is the cheapest level that answers correctly"
   analysis (M3's *oracle*, §2.1) needs no new training or inference.
7. **Existing but unused hooks.** `ElasticConfig.n_sample_students`, the per-forward
   deterministic RNG seed (`engine._fwd_step`), `debug/measure_token_rank.py`'s
   effective-rank code (a ready-made "image complexity" statistic), and the roofline
   analyzer in `llava/eval/efficiency/` (a place to put an escalation cost model).

---

## 2. Literature by family

### 2.1 Adaptive *total* visual-token count, decided on the vision side (before the LLM)

The most populated family; the decision costs nothing LLM-side and there is no KV
issue because the budget is fixed before prefill. This is where §12's mechanisms A/B
sit, and it is **not** a novel axis — position against it, don't claim it.

| Work | What adapts, on what signal | Notes |
|---|---|---|
| Matryoshka Multimodal Models, M3 — arXiv:2405.17430 [fetched-adjacent, snippet] | Nested token scales {1,9,36,144,576}; scale is user/task-chosen. Defines the **oracle**: per-sample, the fewest-token scale that still answers correctly; reports a large oracle-vs-fixed gap and names "token length predictors" as open. | Our ladder is the same object. The oracle analysis is item **A1** below — zero training. |
| AVG-LLaVA — arXiv:2410.02745, ACL-Findings 2025 [snippet] | Learned **router** (Transformer layer + MLP + voter) picks one of several pooling granularities per *image + instruction*; trained with an RL-style objective (RGLF). 85.3% fewer tokens, 2.53× faster on AI2D. | Nearest precedent to a learned budget router. Single mechanism (pooling granularity), needs the instruction; image-only reportedly worse. |
| LLaVA-PruMerge / PruMerge+ — arXiv:2403.15388 [snippet] | Token count from IQR outlier detection on CLS attention: more tokens for information-dense images; PruMerge+ adds a uniform spatial complement. | Whether the salient/uniform *ratio* is itself adaptive is unverified — read before citing as an anchor/query precedent. |
| HiRED — arXiv:2408.10945, AAAI 2025 [snippet] | Per-partition budget allocation under a global cap, from CLS attention. | Content-adaptive *distribution* of a fixed total across regions — closest in spirit to "adapt composition at fixed B." |
| Adaptive-VoCo — arXiv:2512.18496 [snippet] | Lightweight predictor on vision-encoder statistics (patch-token entropy, attention-map variance) selects the compression rate. | Cheap statistics-based router = §12 mechanism A. |
| E-AdaPrune — arXiv:2603.05950 [snippet] | Budget from the **singular-value spectrum** of the visual features (more tokens for information-dense scenes). | Directly the effective-rank idea in journal §11/§11a, used as a router signal. |
| AgilePruner — arXiv:2603.01236 [snippet] | Effective-rank (erank) of the image: high-erank images keep more, diverse tokens. | Same signal again; note for positioning that erank-as-router is now prior art. |
| OccamToken (2605.29657), AsymVLM (2605.29535), TrimTokenator-LC (2512.22748) [snippet] | Training-free / per-sample adaptive pruning, some query-aware. | Very recent, snippet-only. |
| DOVE — arXiv:2506.03643 [snippet] | Tokenizer whose output length correlates with image complexity. | Adaptive count at the *tokenizer*. |
| ElasticTok — arXiv:2410.08368, ICLR 2025 [snippet] | Variable tokens per frame; trained by **masking a random number of tail tokens** so any prefix is a valid encoding; at inference, pick the shortest prefix meeting a reconstruction threshold. | Tail-masking is nested dropout under another name (`use_nested_dropout`). Their inference rule (shortest prefix under a quality threshold) is a budget policy we could copy. |
| AdaTape — arXiv:2301.13195, ICML 2023 [snippet] | **Elastic input sequence**: appends a variable number of "tape tokens" per input via an ACT-style halting rule, adaptivity in the *input length* rather than depth. | Conceptually the closest ancestor of "per-image visual token count"; cite in motivation. |
| A-ViT (2112.07658, CVPR 2022), DynamicViT [known] | Per-token halting inside the ViT (ACT reformulated); token count varies per image. | Adaptive compute *inside the encoder* — orthogonal to our axis (we keep the encoder dense), but the halting/ponder cost formulation is reusable for a router loss. |
| Resolution selectors: CARES (2510.19496), ResAdapt (2603.28610), Mixture-of-Resolution/LLaVA-HR (2403.03003), SmolVLM dynamic tiling [snippet/known] | Choose input resolution / tile count per image (and sometimes per query). | Same family as §12 mechanism D, at the resolution knob instead of the token knob. |

### 2.2 Adaptive depth: early exit and Mixture-of-Depths

Adaptivity along the *layer* axis. Not our axis, but it is where the KV-cache problem
has already been solved once, and the solutions transfer.

| Work | Mechanism | KV-cache handling |
|---|---|---|
| CALM — arXiv:2207.07061, NeurIPS 2022 [known] | Per-token early exit on a calibrated confidence threshold. | Exited tokens have no KV at the skipped layers; CALM **copies the exit-layer hidden state upward** so later tokens can still attend. Establishes that "stale/approximate KV for a few tokens" is tolerable. |
| LayerSkip — 2024 [known] | Early-exit training + **self-speculative decoding**: early layers draft, remaining layers verify. | Draft and verify **share one KV cache** (the draft's KV for the early layers is exactly the verify pass's KV for those layers) — zero waste. This is the template for item A7. |
| MuE — arXiv:2211.11152, CVPR 2023 [snippet] | Multiple exits in a unified VL encoder-decoder; exit when consecutive-layer hidden-state similarity saturates; can skip different layers per modality. | Similarity-saturation is a *free* confidence signal that needs no head — candidate escalation signal for A6. |
| Mixture-of-Depths — arXiv:2404.02258 [known] | Per-layer top-k router: capacity cap fixes compute; routed-out tokens take the residual path. Causal routing needs an auxiliary predictor at inference. | Routed-out tokens contribute no KV at that layer by design. The **capacity-cap** formulation (fix average compute exactly, let the router choose *which*) is the cleanest way to bound a budget router's mean cost — item A3. |
| p-MoD — arXiv:2412.04449, ICCV 2025; γ-MoD — arXiv:2410.13859, ICLR 2025 [snippet] | MoD applied to *visual* tokens inside an MLLM; p-MoD decays the retention ratio with depth (55.6% TFLOPs, 53.7% KV); γ-MoD picks which layers to convert via attention-map rank (ARank). | Adaptive per-token compute in the LLM for visual tokens. Orthogonal to input-budget adaptivity; combinable. γ-MoD's *rank-of-attention* diagnostic is a cousin of our §11 rank measurement. |
| Mixture-of-Recursions — arXiv:2507.10524 [snippet] | Per-token recursion depth. | Same axis as MoD. |

### 2.3 Capacity routing over *nested* sub-models (MoE with Matryoshka experts)

| Work | Mechanism | Relevance |
|---|---|---|
| Mixture of Nested Experts, MoNE — arXiv:2407.19985, NeurIPS 2024 [fetched-adjacent] | Experts are **nested slices of one model** on an increasing cost/accuracy curve; a router assigns each visual token to a nested expert under a compute budget, "redundant tokens are processed through cheaper nested experts"; ~2× inference FLOP reduction at equal accuracy. | Our nested LoRA ranks (`lora_ranks`, `NestedLoRALinear`) *are* nested experts on the vision tower. MoNE says: route per input (or per token) to a rank, rather than tying rank to the user's budget. Item A9. |
| Switch Transformer load-balancing loss; expert-choice routing [known] | Auxiliary loss keeps expert utilization balanced; expert-choice fixes capacity per expert instead of per token. | The standard tools for keeping a learned budget router's *average* cost bounded (mechanism B in journal §12). |

### 2.4 Cascades and confidence-gated escalation (decide *after* a cheap LLM pass)

This is journal §12's mechanism D as practiced. Every work here **re-runs** the
expensive stage from scratch on escalation; none found reuses the cheap pass's KV.

| Work | Mechanism | KV / cost handling |
|---|---|---|
| GATEKEEPER — arXiv:2502.19335 [snippet] | Model cascades for LLMs/VLMs; **confidence tuning** trains the small model to be under-confident exactly when wrong, so deferral is reliable. | Escalation is a separate full inference. Calibration is the lever for keeping escalation rare — fact 5's 50% break-even makes calibration matter more for us than for a 7B→70B cascade. |
| CascadeVLM (fine-grained classification) — arXiv:2405.11301 [snippet] | Entropy threshold on the small model's output gates escalation. | Full re-run. |
| Zoom pipelines: AVA-VLM (2607.05859), Zoom-consistency (2604.15376), V\*/ZoomEye-style [snippet/known] | Low-res global pass, then crop-and-zoom when uncertain; "zoom consistency" (distance between step-2 prediction and crop center) as a free confidence signal. | Second pass is a *different image* → nothing reusable by construction. |
| WaveCLIP — arXiv:2509.21153 [fetched] | Wavelet coarse-to-fine tokens with **causal cross-level attention**; at inference, KV from coarse levels is reused and only the new fine-level tokens are computed; **confidence-gated early exit** across levels. Vision encoder only (CLIP-style zero-shot). | The one work found that designs the representation so escalation *is* an append. The same idea applied to a resampler = item A4. |

### 2.5 Speculative decoding with a reduced-visual-token draft

Turns the cheap pass from "wasted if we escalate" into "a draft that accelerates the
expensive pass." The most direct literature answer to "escalation eats the savings."

| Work | Mechanism |
|---|---|
| SpecVLM — arXiv:2509.11815 [snippet] | EAGLE-2-style draft for VLMs + an **elastic visual compressor** (pruning / pooling / learned resampler / conv) that adaptively picks how much to compress the draft's visual input; online-logit distillation for the draft. 1.5–2.3× end-to-end. |
| SpecVLM (video) — arXiv:2508.16201; FastVLM self-speculative — arXiv:2510.22641 [snippet] | **Self-speculative**: the draft *is* the target model with aggressively pruned visual tokens (verifier-guided pruning); reduces the draft's KV and compute. |
| HiViS — arXiv:2509.23928, CVPR 2026 [snippet]; ViSpec (2509.15235); MASSV (2505.10526) [snippet] | Drafter sees no visual tokens at all (HiViS) or a vision-aware drafter; various draft-training recipes. |

A draft = "same model at `tok_level=3` (16 tokens)", target = "same model at
`tok_level=0`" is exactly the FastVLM/SpecVLM-video self-speculative setup, with our
nested ladder as the compressor. Not found in this exact form (nested Matryoshka
budgets as the draft/target pair) — flagged as a possible contribution, not verified.

### 2.6 KV-cache reuse and partial recomputation

| Work | What it reuses | Applicability to budget escalation |
|---|---|---|
| vLLM multimodal prefix caching (PRs #8348, #11187) + encoder cache; LMCache [snippet] | Hash of the image → cached **encoder output** and cached **KV blocks** for the image-token prefix, across requests on the same image. | Directly useful for the *same-budget, many-questions* case (eval, multi-turn). For escalation, only if the low-budget visual block is a prefix of the high-budget one (facts 2–4). |
| VLCache — arXiv:2512.12977 [fetched] | Recurring multimodal inputs: reuse KV + encoder cache, **recompute 2–5% of tokens** chosen by a layer-aware policy; formalizes cumulative reuse error; 1.2–16× TTFT. | Same-input reuse; the *selective recompute* machinery is what an escalation would need for the question tokens. Whether it handles a changed visual-token count is not stated in the abstract. |
| CacheBlend — arXiv:2405.16444 [fetched] | **Non-prefix** reuse: a chunk's cached KV is reused even though the preceding context differs, by recomputing KV for a small subset of tokens with the largest deviation; 2.2–3.3× TTFT, quality ≈ full prefill. | The exact tool for "reuse the question's KV after the visual block changed" (item M3). Benefit scales with the length of the reused text — small for a one-line VQA question, real for multi-turn / long-instruction contexts. |
| Visual-KV pruning: VisCache (EMNLP 2026), SparseVILA (2510.17777), ERASE (2605.09982), and the FastV/PyramidDrop line [snippet/known] | Compute *all* visual tokens, then drop their KV (or skip them in later layers) to save decode memory/time. | The **opposite direction** from ours: they pay full prefill and save decode; our elastic axis saves prefill. Complementary, and the right comparison point when someone asks "why not just prune the KV." |
| CALM / LayerSkip (§2.2) | Depth-axis KV: hidden-state copying; draft/verify share one cache. | Template for making the cheap pass's KV *part of* the expensive pass rather than a discard. |

---

## 3. The KV-cache problem, stated precisely for this stack

Escalate from budget B_lo to B_hi on the same image and question. What is reusable?

| Component | Plain `query`, shared LoRA (`lora_specialize_tok=False`) | Plain `query`, specialized LoRA (current default) | `pool_anchored` (v8/v9) |
|---|---|---|---|
| Vision-tower patch features (36% of FLOPs) | **reusable** (cache them) | recompute (different rank) | recompute if specialized, else reusable |
| Low-budget visual tokens' KV (B_lo entries) | **reusable** (fact 2: identical tokens, same positions, causal) | stale (different features) | stale (fact 4) |
| New visual tokens (B_hi − B_lo) | compute | compute | compute (all B_hi) |
| System-prompt KV | reusable | reusable | reusable |
| Question KV | stale — recompute (or CacheBlend-style partial) | stale | stale |
| Generated tokens | restart | restart | restart |

Arithmetic with fact 5 (T ≈ 50 text tokens, B_lo = 16, B_hi = 256): the reusable
fraction of an escalated prefill is at best (system + 16)/(system + 256 + question) ≈
15–20% in the best column, ≈ 0 in the worst; the encoder reuse (36% of total FLOPs)
matters more than any of it. **Conclusion:** KV reuse cannot make escalation cheap;
it shaves at most a fifth off the second pass. The cost model that actually governs a
cascade is

    E[cost] ≈ C(B_lo) + p_esc · (C(B_hi) − reuse)      with C(16)/C(256) ≈ 0.50 here,

so the cascade pays off only while **p_esc ≲ 50%** (less if the encoder must re-run).
The policy's calibration (GATEKEEPER) and the *tradeoff steepness of the ladder*
(precondition in §0) decide the outcome, not the cache mechanism.

### 3.1 Does any work address escalation-time KV reuse? Partially, and none in a VLM LLM.

- **By construction (representation designed to be appendable):** WaveCLIP (encoder
  side, causal cross-level attention); ElasticTok's tail-masking (any prefix is valid).
  Nothing found doing this in the *LLM* for a nested-token VLM.
- **By partial recomputation of a changed context:** CacheBlend, VLCache — general
  serving-side tools, not budget-escalation-specific.
- **By making the cheap pass part of the expensive pass:** LayerSkip (depth), SpecVLM /
  FastVLM self-speculative (visual tokens). This is the strongest answer: the low-budget
  pass is never wasted because it drafts for the high-budget verify.
- **By deciding before the LLM:** the whole of §2.1 — sidesteps the problem entirely.

### 3.2 How it could be overcome here — mitigations, each with its price

| # | Mitigation | Requires | Buys | Price / caveat |
|---|---|---|---|---|
| M1 | **Nested-append escalation**: reuse system-prompt + B_lo visual KV, prefill only the new B_hi − B_lo tokens + question. | Plain `query` resampler; same LoRA level on the escalation pair (or shared adapter); `generate()` plumbing to append into an existing cache with continued position ids. | ≤ 20% of the second prefill; plus encoder reuse (36%) if LoRA level is shared. | Not available for `pool_anchored` as built (fact 4). Position ids of the question shift by B_hi − B_lo — fine, since it is recomputed anyway. |
| M2 | **Nested-consistent PARCEL**: fix the anchor grid across the escalation pair (e.g. always 64 anchors for B ≥ 64, cf. PARCEL's own 8×8 for B ≥ 64) and make `pool_self_attn` **causal in query order** (anchors + query *i* never attend to queries > *i*). Then low-budget outputs are a prefix of high-budget outputs and M1 applies. | Resampler change + retrain; anchors lose "see all queries" (docstring says that was free — it is no longer free). | Makes PARCEL appendable, à la WaveCLIP's causal cross-level attention. | Unknown accuracy cost of causal self-attention; pure-anchor budgets (16 = 4×4 grid) still can't be a prefix of a 64-anchor block — escalation from 16 stays a full recompute; only B ≥ 64 pairs become appendable. |
| M3 | **Selective recompute of the question KV** (CacheBlend/VLCache style) instead of full recompute after the visual block changes. | Per-token KV-deviation heuristic; custom attention forward. | Proportional to question/context length — negligible for one-line VQA, real for long system prompts or multi-turn. | Engineering-heavy for the single-turn eval setting; only worth it if a long-context deployment is the target. |
| M4 | **Decide before the LLM** (vision-side router: journal §12 A/B). | Router on patch statistics or a small learned head. | Zero waste, zero KV issue; the only mitigation that removes the problem rather than shrinking it. | Router sees no question (AVG-LLaVA reports image-only routing is weaker); its cost is paid on every image. |
| M5 | **Decide after k LLM layers** on the low-budget pass (early-exit-style probe: MuE's layer-similarity saturation, or a tiny exit head). | Exit signal; abort-and-restart logic. | Bounds waste to k/L of the cheap LLM pass (encoder cost still paid). Under M1 conditions the first k layers' KV for the visual prefix is reusable. | The signal must be about *the image budget*, not the next token — CALM-style token confidence isn't obviously the right probe; needs a calibration study. |
| M6 | **Self-speculative cross-budget decoding**: `tok_level=3` drafts, `tok_level=0` verifies. | Draft/verify loop in `generate()`; the two passes are the same weights so no extra model. | The cheap pass is never wasted; savings move from prefill to decode (acceptance length). | Verify pass still pays full B_hi prefill; benefit is on generation length, i.e. small for short VQA answers, large for captioning/long answers. Acceptance rate is unmeasured — and would be *high* on a flat ladder, which is exactly the situation where escalation is pointless anyway. |
| M7 | **Reserved position block for visual tokens** (text positions fixed regardless of B; M-RoPE-like). | Position-id surgery + retrain. | Necessary but not sufficient for reusing anything after the visual block: text K/V *values* still depend on the visual content (attention), so M7 only helps together with M3. | Alone it buys nothing; listed so nobody reaches for it first. |
| M8 | **Question-first layout** (`[system][question][visual][answer]`). | Retrain with the new layout. | Question KV becomes reusable across budgets; visual tokens become question-aware (like instruction-aware selection, §2.1). | Speculative — no precedent verified here for LLaVA-style models; interacts with everything trained so far. |

Practical reading: **M4 first** (removes the problem), **M6** as the one mechanism that
makes a cascade cost-safe regardless of p_esc, **M2 only if** escalation from
`pool_anchored` turns out to be needed; M1/M3/M5/M7/M8 are engineering knobs whose
payoff is bounded by the §3 arithmetic.

---

## 4. Novelty / positioning summary

- **Not novel:** adaptive *total* visual-token count per image (§2.1, a dozen works);
  effective-rank / entropy as the routing statistic (E-AdaPrune, AgilePruner,
  Adaptive-VoCo); learned granularity routers (AVG-LLaVA); cascades with confidence
  gating (GATEKEEPER, CascadeVLM); speculative decoding with compressed-visual drafts
  (SpecVLM, FastVLM); coarse-to-fine appendable representations (WaveCLIP, ElasticTok).
- **Narrow gap that still looks open** (journal §14): content-conditioning the
  *composition* (anchor vs. query) of a *fixed* budget inside a pool-anchored hybrid
  resampler; nearest analog AVG-LLaVA (single mechanism, needs the instruction).
- **Possibly open, unverified:** a Matryoshka-nested budget ladder used as the
  draft/target pair for self-speculative decoding (§2.5); a causal-in-query-order
  pool-anchored resampler making budget escalation an append in the LLM (M2). Neither
  was found in this pass; neither has been searched exhaustively.

---

## 5. Actionable items, mapped to the codebase

Ordered by (evidence gained per unit of work) with the §0 precondition in mind. "Where"
names the files that would change; nothing has been changed.

| # | Item | Where | Training? | KV / cost implication | Positions against | Effort |
|---|---|---|---|---|---|---|
| **A1** | **Oracle-gap analysis** (M3's oracle) on existing per-level eval logs: per sample, the cheapest `tok_level` that is correct; report oracle accuracy vs. tokens against each fixed level. Tells you (a) whether a tradeoff exists at all (§0), (b) the ceiling any router can reach, (c) the *escalation rate* a perfect cascade would need. | new `debug/oracle_gap.py` over `eval_logs/`; per-run `lmms-eval` outputs | none | none (analysis) | M3 §"oracle" | small |
| **A2** | **Router-signal correlation study**: compute cheap per-image statistics (patch-feature effective rank via `effective_ranks()` in `debug/measure_token_rank.py`; patch entropy / attention dispersion à la Adaptive-VoCo; CLS-attention IQR à la PruMerge) and correlate with A1's per-sample oracle level. If nothing correlates, mechanism A is dead before it is built. | `debug/measure_token_rank.py` (reuse), new script | none | none | E-AdaPrune, AgilePruner, Adaptive-VoCo | small |
| **A3** | **Cheap statistics router** (journal §12 A): map A2's best statistic → `tok_level` (and/or `anchor_ratio`) per image at inference; evaluate accuracy vs. *expected* tokens. Add a **capacity-cap variant** (MoD-style: fix the batch-mean budget, let the router choose which images get more). | `lmms-eval/.../llava_elastic.py` (per-sample `matryoshka_vis_token_scale`), `NestedQueryResampler.n_anchors_for` (per-image `anchor_ratio` override), `ElasticEngine.reduce_tokens` | none for total-budget routing (the ladder is already trained at every level); composition routing on `pool_anchored` needs the resampler trained on varied ratios | none — decided before the LLM (M4) | AVG-LLaVA (learned, instruction-conditioned) vs. ours (image-only, statistic-based) | small–medium |
| **A4** | **Learned budget/composition router** (journal §12 B) with a Switch-style load-balancing loss or a MoD capacity cap on realized tokens; straight-through/Gumbel over the discrete ladder, or REINFORCE like AVG-LLaVA's RGLF. | new head in `resampler.py` or `engine.py`; loss term in `llava_elastic_mixin.py` next to `decorr`; `ElasticConfig` fields; `train_elastic.py` flags | yes (Stage 2, warm-start) | none (M4) | AVG-LLaVA, MoNE, MoD | medium |
| **A5** | **Escalation cost model** in the efficiency tooling: extend the roofline analyzer with `C(B)` including the budget-independent encoder, and an `E[cost](p_esc, reuse)` function; used to score A3/A4/A6 on cost, not just tokens. | `llava/eval/efficiency/analyzer.py`, `roofline.py` | none | encodes §3 | — | small |
| **A6** | **Confidence-gated cascade harness** (journal §12 D as an *eval-time* experiment): run `tok_level=3`, compute an answer confidence (first-answer-token margin/entropy; or MuE-style layer-similarity saturation), escalate to `tok_level=0` above a threshold; sweep the threshold; report accuracy vs. expected cost (A5) and the realized p_esc against the 50% break-even. Optionally GATEKEEPER-style confidence tuning later. | `lmms-eval/.../llava_elastic.py` generate path; `debug/` | none for the harness; calibration training optional | full recompute on escalation (baseline for M1–M3); this measures whether reuse is even worth building | GATEKEEPER, CascadeVLM | medium |
| **A7** | **Self-speculative cross-budget decoding** (M6): draft with `tok_level=3`, verify with `tok_level=0`, same weights; measure acceptance length on captioning/long-answer tasks where decode dominates. | `generate()` path in `llava_elastic_mixin.py` / a wrapper; needs two KV caches | none | the cheap pass is never wasted; verify still pays full prefill | SpecVLM, FastVLM self-spec, LayerSkip | medium–large |
| **A8** | **Nested-consistent PARCEL** (M2): fixed anchor grid for B ≥ 64 + causal-in-query-order `pool_self_attn`; unit-test that output[:B_lo] at B_hi equals the B_lo output. Then M1 becomes available for `pool_anchored`. | `resampler.py::_forward_pool_anchored`, `n_anchors_for`; `ElasticConfig` flag; tests in `llava/model/elastic/tests/` | yes (retrain) | makes escalation an append (≤ 20% prefill saving + encoder reuse under shared LoRA) | WaveCLIP, ElasticTok | medium; only if A6 shows escalation is needed |
| **A9** | **Per-image LoRA-rank routing** (MoNE-style): decouple the vision-tower rank from the user budget and pick it from A2's statistic (or A4's router); nested ranks make this free of extra parameters. Also the prerequisite for encoder-feature reuse across escalation (fact 3): shared adapter, or router chooses a level shared by both budgets. | `ElasticConfig.lora_specialize_tok`, `lora_level_for_tok`, `llava_arch.py:189` | v4-vs-v5 already exists as the "does specialization help at all" ablation — reuse it | encoder reuse (36% of FLOPs) on escalation | MoNE | small (config) → medium (router) |
| **A10** | **Composition router on a fixed budget** — the §14 gap: per-image `anchor_ratio` from A2's statistic, with A3's capacity cap so mean cost is unchanged. Requires queries that actually diversify when given budget — i.e. v9's decorrelation result first (journal §13). | `n_anchors_for` (per-image override path already exists via `anchor_routing` semantics), `reduce_tokens` | resampler must be trained across ratios (`anchor_ratio` sampled per step — one-line change in `reduce_tokens` using the existing deterministic `_fwd_step` RNG) | none (M4) | HiRED (per-region allocation), AVG-LLaVA | medium |
| **A11** | **Same-image, many-questions caching** for eval throughput: cache encoder output + visual-block KV per (image, `tok_level`) across questions, vLLM-prefix-caching style. Unrelated to adaptivity, but it is where most *real* KV reuse lives in a benchmark setting. | eval wrapper | none | reuse of the whole visual prefix across questions | vLLM #8348/#11187, VLCache | small–medium |

**Recommended order:** A1 → A2 → A5 (all zero-training, ~days) — they answer "is there
a tradeoff," "is there a cheap signal," and "what does escalation cost here." Then, only
if A1 shows a gap: A3 (+A9 config variant) as the first real mechanism, A6 to measure
p_esc, and A10 once v9 shows the queries can use extra budget. A4/A7/A8 are the
larger builds and should each be justified by an A1/A6 number, not by the literature.

---

## 6. Citation list (for the paper; verify each before use)

- PARCEL — arXiv:2605.30126 — budget-conditional anchor/query split, not content-conditional [fetched, journal §14].
- Matryoshka Multimodal Models (M3) — arXiv:2405.17430 — nested scales; oracle gap; token-length predictor named as open [snippet].
- AVG-LLaVA — arXiv:2410.02745 — learned granularity router, image+instruction, RL-style training [snippet].
- LLaVA-PruMerge — arXiv:2403.15388 — adaptive count via CLS-attention outliers [snippet].
- HiRED — arXiv:2408.10945 — per-region allocation under a global cap [snippet].
- Adaptive-VoCo — arXiv:2512.18496 — statistics-based rate predictor [snippet].
- E-AdaPrune — arXiv:2603.05950 — singular-spectrum-driven budget [snippet].
- AgilePruner — arXiv:2603.01236 — effective-rank-driven adaptive count [snippet].
- DOVE — arXiv:2506.03643; ElasticTok — arXiv:2410.08368; AdaTape — arXiv:2301.13195 — adaptive-length representations [snippet].
- A-ViT — arXiv:2112.07658 — per-token halting in ViTs [known].
- CALM — arXiv:2207.07061; LayerSkip; MuE — arXiv:2211.11152; DEED — arXiv:2311.08623 — early exit and its KV handling [known/snippet].
- Mixture-of-Depths — arXiv:2404.02258; p-MoD — arXiv:2412.04449; γ-MoD — arXiv:2410.13859 [known/snippet].
- Mixture of Nested Experts — arXiv:2407.19985 — nested-expert routing under a compute budget [fetched-adjacent].
- GATEKEEPER — arXiv:2502.19335; CascadeVLM — arXiv:2405.11301 — confidence-gated cascades [snippet].
- WaveCLIP — arXiv:2509.21153 — causal cross-level attention, cached progressive inference, confidence-gated exits (encoder side) [fetched].
- SpecVLM — arXiv:2509.11815; SpecVLM-video — arXiv:2508.16201; FastVLM self-speculative — arXiv:2510.22641; HiViS — arXiv:2509.23928 [snippet].
- VLCache — arXiv:2512.12977; CacheBlend — arXiv:2405.16444 — partial-recompute KV reuse [fetched].
- vLLM multimodal prefix caching — PRs #8348, #11187 [snippet].
