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
| 22 | Queue a final rank re-measurement (COCO + TextVQA) for v8-parcel once it actually finishes training, so §11/§11a's paper numbers are taken against the FINAL checkpoint, not the 96%-trained checkpoint-5000 snapshot. Separately, run a literature check on §12's novelty claim (input-conditional anchor/query allocation). | v8-parcel finished training 2026-09-08 17:22 (5197/5197 steps, clean exit). Job 27340 ran automatically and measured the final checkpoint-5197 on both COCO and TextVQA — §11/§11a updated with the real final numbers, statistically indistinguishable from the checkpoint-5000 snapshot (every value moves by less than its own std), so all three prior readings stand unchanged; these are now the paper-final figures. Literature check done (§14): found the actual paper "PARCEL" is based on, confirmed it is NOT content-adaptive either, found one close routing precedent (AVG-LLaVA) that narrows but does not eliminate the claimed gap. |
| 23 | Research-only survey of adaptive input token budget allocation across the early-exit / MoE / cascade / speculative-decoding / KV-cache literature, including §12 mechanism D, with the KV-cache-under-escalation problem considered for every action point; write it up as a standalone doc with the actionable items loopable into this codebase. | [ADAPTIVE_INPUT_TOKEN_BUDGET_ALLOCATION.md](ADAPTIVE_INPUT_TOKEN_BUDGET_ALLOCATION.md). Key outcomes: (a) adaptive *total* budget per image is well-trodden (a dozen works, several already using effective rank as the router signal) — mechanism D is not a novelty claim; (b) no work found reusing KV across a budget escalation inside a VLM's LLM — the nearest are WaveCLIP (encoder-side causal cross-level attention), CacheBlend/VLCache (partial recompute), and LayerSkip/SpecVLM (make the cheap pass a draft); (c) for *this* stack the KV question is mostly moot: the budget-independent vision tower is 36% of FLOPs, so a 16-token pass costs ≈0.5× a 256-token pass, the cascade breaks even at ~50% escalation rate, and KV reuse can shave at most ~20% off the second prefill — deciding *before* the LLM (vision-side router) or making the cheap pass a speculative draft are the mitigations that actually change the economics; (d) `pool_anchored` as built is not nested across budgets (self-attention over the joint set + budget-dependent anchor grid), plain `query` is, and `lora_specialize_tok` breaks feature-level nesting for both — all config/architecture facts that gate which items are loopable. Eleven actionable items, ordered; the first three are zero-training analyses on existing eval logs (M3-style oracle gap, router-signal correlation, escalation cost model) that also settle the precondition nothing else survives without: whether the ladder has an accuracy gradient to trade on at all. |
| 24 | Cluster-wide reboot (all 8 nodes down) interrupted v7-kd7b mid-training and truncated v6-tokrange's eval to 4/8 levels. Cancel v7-kd7b and its eval outright; requeue the missing v6-tokrange levels with the script bug fixed; resize v9-parcel-decorr to fit on A10 (node208) instead of A40 (node205) so node205 stays free; write up the backbone/PARCEL/decorr ablation matrix that doesn't fit locally as a pending-work list for the other ("hipster") cluster. | Done (§15). Found and fixed a real bug in `eval_lmms_level.sh` along the way: `jq`'s availability is inconsistent across compute nodes, so 4 of 8 v6-tokrange eval array tasks silently fell back to a wrong 4-level default and errored out even though `elastic_config.json` was present and correct the whole time — replaced with a `python3`-based parse (env-guaranteed) and a loud failure instead of a silent wrong-grid fallback. v7-kd7b (27299) + its eval (27300) cancelled; v6-tokrange's missing levels (576/512/448/384-token) requeued as job 27375 `--array=4-7`. v9-parcel-decorr requeued as job 27376/27377 with `--gres=gpu:A10:2` (was A40:2) — justified by v8-parcel, the identical architecture minus the decorrelation term, having already completed its full 84.5h run cleanly on node208's A10:2. |
| 25 | Analyse v6's available eval levels and v8 against the earlier runs and say whether any introduced change actually helped; record the outcome; make **v8 the best working model**; drop rank-nested vision LoRA from all future runs including v9; queue **v10** with vision LoRA fixed at 16 for both stages; re-scope the hipster matrix to the v8 setup with vision LoRA off, stating in that section that v8 is the best-performing config and that vision LoRA is off; rename v6's eval folders correctly and verify the pending evals land in the right ones. | Done (§16). **Yes, one change helped: PARCEL (v8) — the only run with a real budget–accuracy gradient, and the new best model (§16c).** The other two introduced changes lost: rank-nested vision LoRA is a regression on both backbones (§16d, default now False), and the extended 576→16 ladder is a large regression (§16b, retired). Found along the way that §15b's root cause was recorded backwards: *every* task of job 27292 hit the `jq` fallback, so v6's four surviving results were not 256/144/64/16 at all but 576/512/448/384 written under the wrong labels — verified against the measured prefill FLOPs and relabelled (§16a). v9 cancelled and re-scoped onto v8's grid with LoRA off; v10 added; v11 deferred by decision; recipes moved out of submitting-shell env vars into `submit_elastic_run.sh` (+ a `docs/SUBMITTED_RUNS.tsv` job-id→recipe log) in git, after v9's ladder had to be recovered from circumstantial evidence. v9 requeued as 27378/27379 and v10 as 27380/27381, both `--array=0-3`; all 8 nodes were still `down` at submit time, so neither has started. |
| 29 | Restrict the v6 eval job to node206/207 so it does not interfere with the training runs. | Done via `scontrol update jobid=27375 ReqNodeList=node206,node207` — applied to all four pending array tasks at once (§16a). Those are the two single-GPU nodes; the eval only asks for `gpu:1`, but without the pin it could take node205 or node208 and block an `--exclusive` training job for a whole benchmark sweep. Scores are unaffected — the only GPU-type branch in the eval path is its batch size (A40 → 8, A10 → 4), which is throughput. The three queued evals (27383/27385/27387) are **not** pinned: they are dependency-held behind their own training jobs, so by the time they release, the node those jobs occupied is free anyway. |
| 28 | Put v11 into the hipster pipeline too, and add a new **v12** there: v11's setup plus the 7B-teacher KD. Update the journal. | Done (§16f). The hipster ports are now stated as **v11 verbatim with a different `SLM_KEY`**, not a separate design — `ELASTIC_RUN_TAG` stays `v11-parcel-nolora` while the checkpoint path interpolates the backbone, so each lands in its own directory and the local TinyLlama v11 (27386) is the shared reference point. **v12-parcel-kd7b** added to `submit_elastic_run.sh`: `TEACHER=llava` on top of v11, superseding v7-kd7b (§15d), which asked the same question from the superseded v4-era setup. Two constraints made enforceable rather than documentary: the recipe **exits 1 up front** if `SLM_KEY` is not a Llama-32000 backbone (`attach_kd_teacher` would otherwise raise only after Stage 1 had run, and this excludes smollm2/qwen*/phi2 — three of the five matrix rows, leaving `mobilellama` as the only eligible hipster backbone), and it raises its own `DEFAULT_GRES` to A40-class for the frozen 7B's unsharded ~14 GB/GPU. `GRES` is now overridable per submission so these recipes can run on hipster's card names. Neither is queued locally: v12 is A40-only and would be a fourth exclusive job against two eligible nodes, which is the contention §15d moved this work off-cluster to avoid. |
| 27 | Queue v11 as well; stop pinning jobs to a GPU type — request 2 GPUs so runs can land on either node205 (A40:2) or node208 (A10:2) instead of stacking on one node. | Done. **v11-parcel-nolora** (PARCEL, decorrelation off, vision LoRA off) queued as 27386/27387 — the control v8 never had, and what makes v9 and v10 attributable (§16e). All three runs resubmitted with untyped `--gres=gpu:2`: node205 and node208 are the only 2-GPU nodes in `defq` (206/207 have one each), so `gpu:2` resolves to exactly those two. Checked before switching that this cannot silently change results — nothing in the training path branches on GPU type (`per_device_train_batch_size` is a constant in both stages, `NUM_GPUS` defaults to 2), so a run is identical on A40 and A10. Final ids: v9 27382/27383, v10 27384/27385, v11 27386/27387, all `--array=0-3`, all still pending on the outage. Also fixed a bug in `submit_elastic_run.sh`'s own log: the header guard tested `-f` rather than `-s`, so a truncated log would never regain its header. |
| 26 | Ask whether "vision LoRA always hurts" is safe to claim in the paper. Make v9's configuration the default for all future experiments **if** v9 improves on v8. | **Pushed back on the first: no, it is not safe (§16d "Scope limit").** Audited `use_lora` across every checkpoint — **v8, the best model in the project, has vision LoRA ON**, as does v6. The only controlled A/B on the flag is v4 vs v5, both on the plain `query` resampler; under PARCEL, vision-LoRA-off has never been run. The literature is mixed rather than unanimous (Prismatic VLMs supports it but for *full fine-tuning*; Qwen-VL/InternVL/Idefics2 adapt the vision encoder deliberately), and no §6/§14-style search pass has been run on this question. Proposed a scoped claim about the *nested per-level* mechanism instead of the component. The v9 conditional is recorded as a standing decision (§16g) — not actionable yet, since v9 has not started, and flagged as needing v11 first because v9 differs from v8 by **two** flags, not one. |

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
| **PARCEL spatial anchors** (v8-parcel, **checkpoint-5197, FINAL**, n=24 imgs) — **full set** | 64 anchors + 192 queries | 43.8 ± 4.5 / 256 (17.1%) | 7.7 ± 1.1 (3.0%) | 10.3 ± 1.8 (4.0%) | 0.521 ± 0.011 |
| PARCEL — **anchors only** | 64 deterministic pooled | 62.6 ± 1.1 / 64 (**97.9%**) | 40.3 ± 5.6 (63.0%) | 38.7 ± 3.3 (60.4%) | 0.516 ± 0.049 |
| PARCEL — **queries only** | 192 pool-conditioned learned | 26.8 ± 1.8 / 192 (**13.9%**) | 5.1 ± 0.6 (2.7%) | 5.5 ± 0.6 (2.9%) | 0.784 ± 0.013 |

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
training runs). v4 and PARCEL *were* measured on the identical 24 COCO images (both
runs used the script's default `--seed 0`, so the same `random.sample` draw) — the
comparison is apples to apples on that axis at least.

**Update (decision 22, job 27340): re-measured against the FINAL checkpoint
(checkpoint-5197 of 5197, training completed 2026-09-08) once v8-parcel actually
finished.** The table above now shows the final numbers directly. They are
statistically indistinguishable from the checkpoint-5000 (96%-trained) numbers this
section originally reported — full-set rank@1% 44.0→43.8, anchors 62.7→62.6, queries
26.7→26.8, every other column moves by less than its own std — so the mid-training
snapshot was already representative and none of the three readings above change. The
"should be re-taken against the final checkpoint" caveat is resolved; these are the
paper-final figures for v8-parcel.

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
| PARCEL — full (**FINAL, checkpoint-5197**) | COCO | 43.8 ± 4.5 | 7.7 ± 1.1 | 10.3 ± 1.8 | 0.521 ± 0.011 |
| PARCEL — full | TextVQA | 45.9 ± 5.9 (+4.7%) | 8.0 ± 1.4 | 11.2 ± 2.1 (+9.0%) | 0.510 ± 0.013 (−2.1%) |
| PARCEL — anchors only | COCO | 62.6 ± 1.1 | 40.3 ± 5.6 | 38.7 ± 3.3 | 0.516 ± 0.049 |
| PARCEL — anchors only | TextVQA | 62.0 ± 2.4 (≈0%) | 41.3 ± 9.2 | 38.75 ± 6.1 (≈0%) | 0.490 ± 0.046 |
| PARCEL — queries only | COCO | 26.8 ± 1.8 | 5.1 ± 0.6 | 5.5 ± 0.6 | 0.784 ± 0.013 |
| PARCEL — queries only | TextVQA | 29.3 ± 2.6 (+9.3%) | 6.2 ± 0.9 | 6.8 ± 1.0 (+22.6%) | 0.773 ± 0.013 |

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
(`576 512 448 384 256 144 64 16` / `2 4 6 8 8 16 32 64`).

*(Corrected 2026-09-09: "same full ladder and LoRA ranks as v6/v8" is wrong about v8.
`elastic-finetune-tinyllama-v8-parcel/elastic_config.json` is the **4-level**
`256 144 64 16` / `8 16 32 64` grid — the same one v4 and v5 use. Only v6 and v9 were
ever on the 8-level ladder. v9 has since been moved onto v8's 4-level grid; see
§16e.)*

Queued behind whichever of
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

## 15. Cluster outage cleanup, and pending experiments for the other cluster (decision 24)

### 15a. What happened

2026-09-08, sometime after v8-parcel's rank remeasurement (job 27340) landed: all 8
local nodes (node201-208) went `down` simultaneously — SLURM reasons "Node
unexpectedly rebooted" / "Not responding". Infrastructure event, not a job or code
issue; nothing runnable locally until nodes come back. Damage assessment:

- **v8-parcel (27303)**: unaffected — finished cleanly *before* the outage (§11 update
  above).
- **smollm2-v5 finetune (27282) + eval (27283)**: unaffected — both completed before
  the outage, all 4 eval levels present.
- **v6-tokrange finetune**: unaffected, completed. Its **eval (27292, `--array=0-7`)
  did not** — only levels 0-3 finished, and per the correction in §15b those are the
  **576/512/448/384**-token levels, not the 256/144/64/16 ones their directory names
  claimed. Levels 4-7 — the real 256/144/64/16 — failed immediately with a wrong-grid
  error, unrelated to the outage itself: a **pre-existing bug**, just noticed while
  triaging. v6-tokrange has still never been evaluated at 256 tokens or below.
- **v7-kd7b (27299)**: caught mid-training (step ~1046/5197, epoch 0.2) when its node
  rebooted. Last checkpoint saved was `checkpoint-1000`, so ~46 steps since the last
  save were lost — recoverable, but per decision 24 this run is being **cancelled
  outright here**, not resumed (see §15c for why, and where it might resume instead).

### 15b. Bug fixed: `eval_lmms_level.sh`'s `jq` dependency was node-inconsistent

The script read `tok_levels` from the checkpoint's `elastic_config.json` via `jq`,
falling back to a hardcoded 4-level list `(256, 144, 64, 16)` if the file or `jq`
looked missing. `elastic_config.json` was present and correct (all 8 levels) the
entire time. Root cause: `jq`'s availability is not a per-job
constant — it depends on which physical node the SLURM array task lands on, and at
least one node in the pool doesn't have it wired onto the PATH the job script sees
that early (before `module load`/`conda activate`, which the check ran ahead of).

**CORRECTION (2026-09-09).** The first version of this section said "array tasks 0-3
read it fine, tasks 4-7 hit the fallback." That is backwards, and the difference
matters. **Every** array task of job 27292 hit the fallback —
`jobs/eval_lmms_27292_tok0.out` carries the same "falling back to the default 4-level
label list" warning that tok4-7 do. Tasks 4-7 then errored, loudly, because index
≥4 does not exist in a 4-entry list. Tasks 0-3 did something worse: they indexed
*successfully* into the wrong list and produced four complete, correct-looking result
sets under **wrong labels**. They evaluated `tok_level` 0/1/2/3 of an 8-level ladder
— i.e. **576/512/448/384 tokens** — and wrote them into directories named
`256tok/144tok/64tok/16tok`. The loud half of this bug cost four eval tasks; the
silent half nearly cost a wrong paper table, since nothing downstream re-derives the
token count from anything but the directory name. The relabelling is §16a.

Fixed by moving `module load`/`conda activate` to the top of the script and reading
`tok_levels` with `python3 -c "import json; ..."` instead of `jq` — the same
interpreter the script already depends on unconditionally a few lines later (base-LLM
detection) — and by making a missing/unparseable config a **loud failure** rather than
a silent wrong-grid fallback, so this class of bug fails fast next time instead of
quietly producing 4 fewer results. v6-tokrange's missing levels requeued as job 27375
`--array=4-7` with the fixed script.

### 15c. v9-parcel-decorr resized to fit A10, kept local

v9-parcel-decorr's original submission inherited `run_job_slm.sh`'s hardcoded
`--gres=gpu:A40:2` default, i.e. node205-only. v8-parcel — architecturally identical
(`resampler_arch=pool_anchored`, same full ladder, same LoRA ranks) minus the
decorrelation term, which adds no new parameters and a single scalar loss — completed
its entire 84.5-hour run cleanly on node208's **A10:2** with no OOM. That's direct
evidence A10 has headroom for this exact config, so v9-parcel-decorr was resubmitted
with `--gres=gpu:A10:2` (job 27376/27377) instead, freeing node205's A40s for other
work rather than contending for the scarcer resource unnecessarily.

### 15d. v7-kd7b: cancelled here, candidate for the other ("hipster") cluster

v7-kd7b's whole point (§5b, decision 13) is a frozen **external LLaVA-1.5-7B** KD
teacher — the self-distillation default measured a KL of ~0.006 (teacher and student
are literally the same weights on informationally-equivalent inputs, nothing to
distill), so `teacher=llava` is the only way this codebase's KD term could carry real
signal. That question is still open and still worth answering, but continuing to
contend for the same 4 local A10/A40 nodes already saturated by
smollm2-v5/v8-parcel/v6-tokrange/v9-parcel-decorr — on top of today's reminder that
the local cluster's uptime is not guaranteed — makes it a better fit for the other
("hipster") cluster than for a fifth slot in the local queue. **Resume point**:
`elastic-finetune-tinyllama-v7-kd7b/checkpoint-1000` is intact (5 checkpoints of
progress, `save_total_limit=1` so only the latest survives) — the exact same
`run_job_finetune_slm.sh tinyllama` invocation with `TEACHER=llava` will resume from
it rather than restart. Needs an A40-class GPU (or better): the frozen 7B teacher costs
~14 GB/GPU on top of the student, and `attach_kd_teacher` is a plain attribute (not a
submodule), so ZeRO does **not** shard it — verify VRAM headroom on whatever hipster
GPU is targeted before assuming node205's A40 numbers transfer directly.

### 15e. Proposed backbone × PARCEL ablation matrix for hipster

> **RECIPE SUPERSEDED 2026-09-09 — read this before using the table below.**
> The rows (which backbones, and why) still stand. The **recipe they inherit does
> not**. Every row runs the **v11 recipe** — the v8-parcel configuration, which is
> the best-performing setup in the project (§16c), with vision LoRA OFF (§16d) — *not*
> the v9 recipe this section originally specified. A second run per backbone,
> **v12**, adds the frozen 7B KD teacher on top of that; it is Llama-vocab only.
> See §16f.
>
> ```
> TOK_LEVELS="256 144 64 16"     LORA_RANKS="8 16 32 64"
> STAGE1_TOK_LEVEL=256           STAGE1_LORA_RANK=64
> RESAMPLER_ARCH=pool_anchored   ANCHOR_MODE=ratio   ANCHOR_RATIO=0.25
> VISION_LORA_ENABLE=False       USE_TOKEN_DECORRELATION=False
> VISION_TOWER=openai/clip-vit-large-patch14-336      TEACHER=self
> ```
>
> Three things changed from what is written below, each for a measured reason:
> **(1)** the ladder is v8's 4-level `256 144 64 16`, not the 8-level `576…16` — that
> ladder is a large regression (§16b) and is retired; **(2)** vision LoRA is **off** —
> rank-nested vision LoRA lost on both backbones it was tried on (§16d), so it is off
> by default now and these ports inherit that; **(3)** decorrelation is **off** —
> v9 has not reported yet, and an unvalidated loss term in every backbone port would
> confound the backbone comparison. Details and the revised sequencing: **§16f**.

Brainstormed, not yet decided or run anywhere — flagging as a candidate work list per
the user's ask, prioritized by what each run would actually tell us. The original text
below reused v9's then-current recipe (`resampler_arch=pool_anchored`,
`anchor_mode=ratio` @0.25, `--use_token_decorrelation True --decorr_weight 0.01`, CLIP
vision tower, the full `576 512 448 384 256 144 64 16` ladder / `2 4 6 8 8 16 32 64`
LoRA ranks) so every run would be a direct extension of v9; per the banner above it is
now a direct extension of **v11** instead. Kept as written, with the one row whose
rationale the change invalidates marked inline.

**v12 eligibility** (the KD arm, §16f): `mobilellama` is the **only** row here that can
run it — every other backbone fails the Llama-32000 vocab requirement the frozen
LLaVA-1.5-7B teacher imposes, and `tinyllama`, which also qualifies, is not in this
table because it runs locally.

| Backbone | `LLM_KEY` | Conv template | Existing local baseline? | Why this one |
|---|---|---|---|---|
| MobileLLaMA-1.4B-Chat | `mobilellama` | `v1` | **none** — never run through the elastic pipeline at all, only used as an external reference point (§9c's peak-memory/FLOPs comparison against the *original* MobileVLM). | Directly comparable against that same §9c reference architecture if it's ever run at all; also the smallest-vocab Llama-family option after TinyLlama, so `teacher=llava` KD stays available if v7-kd7b's question is revisited on the same backbone family. |
| Qwen2.5-0.5B-Instruct | `qwen0.5b` | `chatml` | **none** — untested even at the plain v4/v5 baseline level in this codebase. | Smallest LLM in the whole family: ~~tests whether decorrelation's benefit (or the underlying collapse it's fixing) scales with LLM capacity~~ — **decorrelation is off in the revised recipe**, so this row's question is now whether **PARCEL's** benefit scales with LLM capacity. The logic carries over unchanged: a tiny LLM has the least room to compensate for redundant visual tokens, so this is the sharpest test of whether the resampler fix matters downstream and not just on the rank metric. **Caveat**: needs its own v4-equivalent baseline run first (or alongside) — there is no existing Qwen accuracy number in this codebase to compare against, plain rank numbers alone won't show the accuracy story. |
| Qwen2.5-1.5B-Instruct | `qwen1.5b` | `chatml` | **none** | Mid-size point on the same curve as above; also a non-Llama-vocab backbone, so exercises the `teacher=self` KD path (the only option — `teacher=llava` requires Llama-32000 vocab) under ~~PARCEL+decorr~~ **PARCEL**, which hasn't been combined before. |
| Phi-2 | `phi2` | `phi` | **v4 exists** (`elastic-finetune-phi2-v4`, evaluated) — a real accuracy baseline to compare against, unlike the two Qwen rows. | Completes a third full backbone comparison (alongside TinyLlama and, once run, SmolLM2) with an actual pre-PARCEL accuracy number already in hand; also the backbone flagged in §9c as having an unreliable FLOPs split measurement (`FlopCounterMode` likely undercounting Phi-2's fused attention) — a second look at that gap wouldn't hurt if this run happens anyway. |
| SmolLM2-1.7B-Instruct | `smollm2` | `chatml` | **v4 AND v5 exist**, both evaluated. | Cheapest of these five to justify, and still the one to run first — but ~~reuses an already-proven-working local recipe (v5 already turned vision LoRA on for this backbone)~~ **is wrong as of §16d**: SmolLM2 v5 is a *regression* against its own v4 (SciQA −9.9, MME −38.8, TextVQA −3.4), so v5 is the wrong thing to build on. Corrected rationale: with vision LoRA off, **SmolLM2's v4 is already the matched baseline** for a PARCEL port, making `smollm2` v4 → v8-setup a clean single-change comparison and the cheapest real result available on that cluster. |

**Sequencing suggestion** (not a commitment): SmolLM2 first (cheapest to de-risk, reuses
existing recipe + baseline), then Phi-2 (real accuracy baseline already exists), then
the Qwen pair (need their own v4 baseline alongside, more setup cost), then
MobileLLaMA (least existing scaffolding). v7-kd7b (§15d) is a separate question — same
cluster, but orthogonal to this matrix (KD-signal strength, not PARCEL/decorr) — don't
conflate its results with this table's.

## 16. v6 and v8 results, and the vision-LoRA verdict (decision 25)

### 16a. v6-tokrange's eval directories were relabelled

Per the correction in §15b, job 27292's four surviving array tasks evaluated
`tok_level` 0-3 of v6-tokrange's 8-level ladder — **576/512/448/384 tokens** — under
directory names taken from the 4-level `jq` fallback. Renamed on 2026-09-09 to what
was actually measured:

| directory (was) | directory (now) | `tok_level` | real budget |
|---|---|---|---|
| `256tok/` | `576tok/` | 0 | 576 |
| `144tok/` | `512tok/` | 1 | 512 |
| `64tok/`  | `448tok/` | 2 | 448 |
| `16tok/`  | `384tok/` | 3 | 384 |

The inner `*_elastic_<label>_*` run directories were renamed to match; **file contents
were never touched and were always correct** — only the labels were wrong. A
`README_RELABEL.txt` next to them records this in place, for anyone who finds the
directory without the journal.

Confirmed independently of the log warning, by the measured prefill FLOPs, which are
computed from the actual forward pass and not from any label: the run now under
`576tok/` reports **1.315 TFLOPs**, against **0.639** for a genuine 256-token TinyLlama
run (`elastic-finetune-tinyllama-v4/256tok`). A linear fit through v4's four levels
predicts 1.283 TFLOPs at 576 tokens — the relabelled values land within ~2.5% of that
fit at all four points, and nowhere near the 4-level labels they carried.

Two follow-ups so this cannot recur silently:

- `scripts/export_eval_csv.py`'s `LEVEL_TOKENS` was a hardcoded 4-entry dict, so it
  would have emitted a **blank** `n_visual_tokens` for every one of the renamed
  directories. Replaced with a `level_tokens()` that parses the digits out of the
  directory name, so any ladder works.
- The pending eval arrays were checked against their **submitted** batch scripts
  (`scontrol write batch_script`), not the working tree — SLURM snapshots the script
  at submit time, and the `jq` fix is still uncommitted. Both **27375** (v6-tokrange,
  `--array=4-7`) and the v9 eval carry the fixed `python3` parse and the loud-failure
  path, so their output directories will be named correctly.

**27375 pinned to the single-GPU nodes (2026-09-09).** The eval array requests
`gpu:1`, but nothing stopped it landing on node205 or node208 — the only two-GPU
nodes, and the only ones v9/v10/v11 can run on. A one-GPU eval taking one of those
would block an `--exclusive` training job for the length of a full benchmark sweep.
Restricted in place with `scontrol update jobid=27375 ReqNodeList=node206,node207`
(A40:1 and A10:1 respectively — the two single-GPU nodes), which applies to all four
pending array tasks at once. No effect on the numbers: `eval_lmms_level.sh` picks its
eval batch size by card (A40 → 8, A10 → 4), which changes throughput, not scores.

### 16b. v6-tokrange: extending the ladder to 576 tokens failed

Only the top four levels exist; 256/144/64/16 are still queued as job 27375.

| tokens | MME-P | POPE-acc | POPE-F1 | SciQA | TextVQA | GQA |
|---|---|---|---|---|---|---|
| 576 | 1042.73 | 80.46 | 79.05 | 47.50 | 14.81 | 50.66 |
| 512 | 1043.50 | 79.78 | 78.04 | 47.35 | 14.51 | 50.33 |
| 448 | 1035.22 | 79.98 | 78.40 | 47.10 | 14.35 | 50.29 |
| 384 | 1038.28 | 80.09 | 78.59 | 47.10 | 14.63 | 50.22 |
| *dense 576 baseline* | *1248.32* | *84.89* | *83.28* | *57.36* | *41.00* | *58.30* |

At **576 tokens — the same count the uncompressed baseline uses**, so the resampler is
not compressing anything — v6 gives up 26.2 points of TextVQA, 9.9 of SciQA, 7.6 of
GQA and 206 of MME against that baseline. It is also worse than **v4 at 16 tokens** on
TextVQA (14.81 vs 19.90) and MME (1042.7 vs 1116.4), i.e. worse than the same
architecture on the short ladder using 36× fewer visual tokens.

And the four levels are flat to within noise — TextVQA spread 0.46, GQA 0.44, SciQA
0.40 across a 1.5× change in budget. This is the [token-budget-has-no-effect] pattern
again — the same "16 tokens ≈ 256 tokens" result that made the prefix-KL ~0 — now
observed at the *top* of the ladder, where a gap should be easiest to produce.

The most likely mechanism is the LoRA ranks the extended ladder forced. Keeping the
ranks ascending with budget descending put ranks `2 4 6 8` on the four high-budget
levels, versus `8 16 32 64` for the same architecture on the short ladder. Stated as a
hypothesis, not a measured cause — separating "576 tokens is bad" from "rank 2 is bad"
would need a run at 576 with a large rank, which nothing on the roadmap currently does.

**Consequence:** the 8-level 576→16 ladder is retired. v9 was on it and has been moved
to v8's 4-level grid (§16e).

### 16c. v8-parcel: the first change that improved anything, and the new best model

**`elastic-finetune-tinyllama-v8-parcel` is the best model in the project** —
`resampler_arch=pool_anchored` (PARCEL), `anchor_routing` 64/36/16/4 (the 25% ratio
from decision 17), on the standard `256 144 64 16` grid with ranks `8 16 32 64`.

The correct control is **v5**, not v4: v8 and v5 share the grid, the LoRA ranks and
`use_lora: true`, and differ only in `resampler_arch`. v4 had vision LoRA off, so
v8-vs-v4 confounds PARCEL with the vision-LoRA flag and *understates* PARCEL, because
that flag is itself a regression (§16d).

| tokens | vs **v5** (matched control) | vs **v4** |
|---|---|---|
| 256 | MME **+134.7**, POPE-acc **+4.95**, TextVQA **+9.91**, GQA **+5.16**, SciQA −1.24 | MME +56.8, POPE +2.76, TextVQA +5.32, GQA +3.72, SciQA −3.17 |
| 144 | MME +95.4, POPE +4.50, TextVQA +7.93, GQA +4.52, SciQA −1.24 | MME +35.8, POPE +2.29, TextVQA +3.29, GQA +3.31, SciQA −3.27 |
| 64 | MME +126.0, POPE +3.91, TextVQA +5.55, GQA +3.86, SciQA −1.43 | MME +70.5, POPE +1.14, TextVQA +1.11, GQA +2.52, SciQA −4.06 |
| 16 | MME +83.7, POPE +1.60, TextVQA +1.97, GQA +2.26, SciQA −3.37 | MME +7.5, POPE −0.21, TextVQA −0.19, GQA +1.43, SciQA +2.24 |

Two things matter more than the raw deltas.

**It closes most of the gap to the dense 576-token baseline at 44% of the tokens.**
v8 at 256 tokens reaches POPE-acc 84.33 against the baseline's 84.89 (−0.56,
effectively matched) and GQA 56.07 against 58.30 (−2.23). v4 at the same budget was
−3.32 and −5.95. TextVQA remains far off (28.47 vs 41.00) but moved 5.3 points, and
SciQA is the one place v8 is clearly worse.

**It is the first run in this project with a real budget–accuracy gradient.** Spread
from 256 down to 16 tokens:

| run | MME-P | POPE-acc | POPE-F1 | SciQA | TextVQA | GQA |
|---|---|---|---|---|---|---|
| v4 | −2.94 | 0.87 | 1.65 | 7.19 | 3.25 | 1.17 |
| v5 | −4.68 | 0.49 | 0.85 | −0.35 | 0.82 | 0.56 |
| **v8** | **46.34** | **3.84** | **4.44** | 1.78 | **8.76** | **3.46** |

v4 and v5 were flat — 16 tokens scored the same as 256, sometimes better, which is what
made the whole elasticity axis meaningless and produced the ~0 prefix-KL. v8 is
monotone and materially sloped on MME, POPE, TextVQA and GQA. That is the property the
architecture was supposed to have, and PARCEL is the first thing that produced it.

Caveat kept in view: single seed, one backbone.

### 16d. Rank-nested vision LoRA is a regression; default is now off

Decision 6 turned rank-nested vision LoRA on for all runs (v5) on the hypothesis that a
vision tower which cannot adapt per level was why v4 was flat. **That A/B has now
resolved against it, on both backbones it was run on.**

| run pair | 256-token deltas |
|---|---|
| TinyLlama v5 − v4 | MME −77.9, POPE-acc −2.19, POPE-F1 −2.26, SciQA −1.93, TextVQA −4.59, GQA −1.44 |
| SmolLM2 v5 − v4 | MME −38.8, POPE-acc −0.82, POPE-F1 −0.65, SciQA −9.91, TextVQA −3.39, GQA −0.72 |

It lost on every benchmark on both backbones, and it did not deliver the elasticity it
was adopted for either — v5's 256-vs-16 spread is *flatter* than v4's on every metric
(table in §16c). The hypothesis was reasonable and the ablation was run properly; the
answer is just no.

Changed accordingly, in both `scripts/v1_5/pretrain_elastic_slm.sh` and
`scripts/v1_5/finetune_elastic_slm.sh`:

- `--vision_lora_enable` now defaults to **False** (was hardcoded `True` since v5) and
  reads `${VISION_LORA_ENABLE:-False}`.
- `--vision_lora_specialize_tok` is now passed explicitly and reads
  `${VISION_LORA_SPECIALIZE_TOK:-True}` — previously it was never passed at all, so it
  silently took `train_elastic.py`'s `True` default. Making it explicit is what allows
  a **shared, non-nested** adapter to be requested (v10).
- Both scripts echo the resolved values into the job log, because Stage 1 and Stage 2
  must agree: they determine the adapter buffer width, and a mismatch reproduces the
  job-27267 warm-start size mismatch.
- The v5 rationale comments are kept and annotated with the outcome, rather than
  deleted. A future reader should be able to see that the flag was an ablation with a
  reason, and what the reason turned out to be worth.

SciQA is the one metric where the story is not clean: v8 loses ~3 points to v4 at most
levels, and v4 is the only vision-LoRA-off run in that comparison, so the SciQA
regression tracks the LoRA flag rather than PARCEL. v9 (PARCEL, LoRA off, same grid)
is the run that separates them.

**Scope limit — do NOT write "vision LoRA always hurts" in the paper.** The evidence
does not support a universal claim, and our own best model contradicts one. Every
checkpoint audited 2026-09-09:

| run | resampler | vision LoRA | outcome |
|---|---|---|---|
| v4 (tinyllama, smollm2, phi2) | `query` | **off** | baseline |
| v5 (tinyllama, smollm2) | `query` | on | lost to v4 on both |
| v6-tokrange | `query` | on | lost badly, but confounded by the 576 ladder (§16b) |
| **v8-parcel** | `pool_anchored` | **on** | **best model in the project** |

The only controlled A/B on the flag is **v4 vs v5, both on the plain `query`
resampler**. Under PARCEL, vision-LoRA-off has never been run at all — v8 has it *on*
and wins. v9 is the first PARCEL run without it, and it changes two flags at once
(LoRA off **and** decorrelation on), so it does not isolate the flag either. **v11**
(PARCEL, decorrelation off, vision LoRA off) is the only design that would, which is
the argument for un-deferring it (§16e).

The deltas we do have are real rather than noise — TinyLlama's GQA −1.44 is ~3× the
0.44pp stderr, and TextVQA −4.59 / MME −77.9 are far outside it. What is thin is
**generality**: two backbones, one seed each, one resampler variant. So the defensible
paper claim is scoped to the mechanism, not the component:

> Rank-nested, per-level vision LoRA did not produce the per-level specialization it
> was introduced for: on both backbones tested it degraded accuracy relative to a
> frozen tower and *flattened* rather than widened the budget–accuracy spread.

with the two-backbone/one-seed limitation stated. Note also that the nearest literature
support (Prismatic VLMs, Karamcheti et al. ICML 2024, which found tuning the vision
backbone degrades performance) is about **full fine-tuning**, not LoRA and not
per-level nested LoRA — while Qwen-VL, InternVL and Idefics2 all adapt the vision
encoder deliberately, several reporting gains exactly on the OCR/high-resolution axis
where our TextVQA gap lives. The literature is mixed; it is not a unanimous result to
lean on. A proper search pass in the style of §6/§14 has **not** been run on this
question yet.

### 16g. Standing decision: v9-as-default is conditional and not yet actionable

**Decision (2026-09-09): if v9 beats v8, v9's configuration becomes the default for all
future experiments.** Recorded here because v9 had not started when it was made — all
8 nodes were still `down` — so nothing could be applied yet.

Two conditions on executing it, both to be checked when v9's eval lands:

1. **v9 differs from v8 by two flags**, not one: decorrelation on *and* vision LoRA
   off. "Adopt v9's setting" therefore adopts both, without evidence for which
   produced the gain — and per the scope limit above, vision-LoRA-off is *not* the
   obvious attribution, since v8 wins with it on. **v11 is now queued (27386/27387)
   precisely so this is answerable** — v11 vs v9 isolates decorrelation, v11 vs v8
   isolates vision LoRA — so read all three before promoting any flag to a default.
2. "Any improvement" needs a threshold. GQA's stderr is 0.44pp and POPE/SciQA are
   comparable, so a sub-1-point win on a single benchmark is not a result. Treat a
   consistent gain across benchmarks and across levels — the shape of the v8-vs-v5
   comparison in §16c — as the bar, not a single cell moving.

### 16e. v9 re-scoped, v10 queued, and recipes are now recorded in git

**v9-parcel-decorr** (jobs 27376/27377) was cancelled before it started — it never
got a node, the cluster having been down since §15a — and requeued (finally as **27382/27383**, after a
first pass that pinned the wrong GPU type — see the GPU note below).
Two changes:

- **Ladder**: moved from the 8-level `576…16` to **v8's `256 144 64 16`**. §16b
  retired that ladder, and v9's whole purpose is to isolate the decorrelation term
  against v8 — which it cannot do from a different grid.
- **Vision LoRA off**, per §16d.

Its eval array is `--array=0-3` now, matching the 4-level grid.

So v9 = PARCEL + decorrelation, vision LoRA off, v8's grid. Against v8 it isolates
decorrelation *plus* the removal of nested LoRA; against v4 it is the clean
"does decorrelation help" read on a comparable architecture.

**v10-parcel-lora16** is new: v8's recipe with vision LoRA as a **single shared rank-16
adapter** instead of rank-nested-per-level, decorrelation off. Mechanically that is
`VISION_LORA_ENABLE=True VISION_LORA_SPECIALIZE_TOK=False LORA_RANKS="16 16 16 16"` —
`ElasticConfig.lora_level_for_tok` returns `len(lora_ranks)-1` when `specialize_tok` is
False, so every level uses `lora_ranks[-1] = 16`, and `NestedLoRALinear` allocates its
`lora_A`/`lora_B` at `max(lora_ranks) = 16`. One adapter, one rank, no nesting.
`STAGE1_LORA_RANK` must be 16 to match that buffer width. Queued as **27384/27385**.
Rank 16 is also where §11's
LoRA-rank literature check (LangVision-LoRA-NAS) put the plausible sweet spot, versus
the 64 we have been using on the assumption that "above where sweeps go flat" is safe.

**v11-parcel-nolora** — PARCEL alone: decorrelation off, vision LoRA off. Deferred
when first proposed, then **un-deferred the same day** once the §16d audit showed that
v8, the best model in the project, has vision LoRA *on*, and that the only controlled
A/B on that flag (v4 vs v5) was run on the plain `query` resampler. v11 is the control
v8 never had, and it is what makes the other two runs readable:

| comparison | isolates |
|---|---|
| **v11 vs v8** | vision LoRA, under PARCEL (decorrelation off in both) |
| **v11 vs v9** | decorrelation (vision LoRA off in both) |
| **v10 vs v11** | vision LoRA again, but shared rank-16 instead of rank-nested |

Without v11, v9-vs-v8 moves two flags at once and attributes nothing — which is the
condition §16g attaches to making v9 the default.

**`submit_elastic_run.sh` is new, and exists because of a real failure of record.**
Runs were being launched as bare `ELASTIC_RUN_TAG=… FOO=… sbatch run_job_slm.sh
tinyllama`, which puts the entire experiment definition in the submitting shell's
environment. SLURM propagates it via `--export=ALL` but records it nowhere readable:
`scontrol show job` shows only the command line. Recovering v9's token ladder needed
its eval array's `--array=0-7` as *circumstantial evidence*, because the shell that
defined it was long gone. Every recipe now lives in that script, in git, and
`DRY_RUN=1` prints the fully resolved configuration before anything is submitted.

It also appends every submission to **`docs/SUBMITTED_RUNS.tsv`** — timestamp, recipe,
both job ids, the ladder, ranks, resampler, decorrelation and vision-LoRA flags, and
the checkpoint path. This closes the other half of the same gap: `scontrol show job`
reports `run_job_slm.sh tinyllama` for v9 and v10 *identically*, so even with the
recipes in git there was nothing on disk mapping a job id back to the recipe that
produced it. Append-only, and job ids are never reused.

**GPU request: untyped `--gres=gpu:2`, not a pinned card type.** The first submission
inherited `gpu:A10:2` from §15c, which pins every run to node208 and makes three
`--exclusive` jobs queue single-file behind one node. Only node205 (`A40:2`) and
node208 (`A10:2`) have two GPUs at all — node206 and node207 have one each — so
`gpu:2` already resolves to exactly those two and lets SLURM place a run on whichever
frees first. Verified this does not break comparability: nothing in the **training**
path branches on GPU type. `per_device_train_batch_size` is a fixed constant in both
stage scripts (16 in Stage 1, 2 in Stage 2) and `NUM_GPUS` defaults to 2 either way,
so a run is identical on A40 and A10 and only wall-clock differs. (`eval_lmms_level.sh`
*does* pick its eval batch size by GPU type, but that changes throughput, not scores.)
Both card types have headroom — v8-parcel carries a heavier rank-64 nested adapter than
anything queued here and completed its full 84.5h run on node208's A10:2.

All three runs are queued behind the outage — every node was still `down` at submit
time, so none has started and no checkpoint directory exists yet. Three
`--exclusive` jobs against two eligible nodes means one always waits; that is expected,
not a misconfiguration.

| run | train | eval | decorr | vision LoRA |
|---|---|---|---|---|
| v9-parcel-decorr | 27382 | 27383 | **on** | off |
| v10-parcel-lora16 | 27384 | 27385 | off | **shared r16** |
| v11-parcel-nolora | 27386 | 27387 | off | off |

`docs/SUBMITTED_RUNS.tsv` carries the same mapping with the full flag set.

### 16f. Revision to §15e's hipster matrix

**Every planned hipster run uses the v11 recipe — the v8-parcel configuration, the
best-performing setup in this project (§16c), with vision LoRA OFF (§16d).** A second
run per backbone, **v12**, layers the frozen 7B KD teacher on top of it.

§15e proposed extending five backbones with "v9's exact recipe." Superseded on two
counts: v9's recipe at the time meant the 8-level ladder (§16b) and, from decision 6,
vision LoRA on (§16d). The matrix is unchanged in **which** backbones and **why** —
only the recipe they inherit changes. §15e now carries a banner to that effect so its
table cannot be read with the retired recipe. In full:

```
TOK_LEVELS="256 144 64 16"     LORA_RANKS="8 16 32 64"
STAGE1_TOK_LEVEL=256           STAGE1_LORA_RANK=64
RESAMPLER_ARCH=pool_anchored   ANCHOR_MODE=ratio   ANCHOR_RATIO=0.25
VISION_LORA_ENABLE=False       USE_TOKEN_DECORRELATION=False
VISION_TOWER=openai/clip-vit-large-patch14-336      TEACHER=self
```

That block **is** `v11-parcel-nolora`, verbatim — the hipster ports are not a separate
design, they are v11 with a different backbone:

```
SLM_KEY=<backbone> GRES=<that cluster's 2-GPU spec> \
    bash submit_elastic_run.sh v11-parcel-nolora
```

`ELASTIC_RUN_TAG` stays `v11-parcel-nolora` while the checkpoint path interpolates
`SLM_KEY`, so each backbone lands in its own
`elastic-finetune-<backbone>-v11-parcel-nolora` with no collision, and the local
TinyLlama v11 (27386) is the reference point every port is measured against.
`GRES` is overridable precisely because this script cannot know hipster's card names.

Decorrelation is **off** for these: v9 has not reported yet, and putting an unvalidated
loss term into every backbone port would confound the backbone comparison with it.

#### v12-parcel-kd7b — v11 plus the frozen 7B teacher

**New, hipster-only.** `TEACHER=llava` on top of the v11 recipe, everything else
identical. It supersedes **v7-kd7b** (§15d), which asked the same question from the
superseded v4-era setup and was cancelled mid-run by the outage — v12 asks it from the
best base we have instead, so a positive result is directly comparable to v11 rather
than to a configuration nothing else uses any more.

The question is still the one §15d framed: `teacher=self` distills the model at
`tok_levels[0]` into its own smaller levels and measures a KL of **~0.006** — teacher
and student are literally the same weights on informationally equivalent inputs, so
there is nothing to distill, and the KD term contributes ~0.01% of the loss.
`teacher=llava` makes **every** level a student, `tok_levels[0]` included, which is the
only way the KL in this codebase can carry real signal.

Two hard constraints, both now enforced or flagged in `submit_elastic_run.sh`:

- **Vocab.** `attach_kd_teacher` (`llava/model/elastic/engine.py:30`) raises when
  teacher and student `vocab_size` differ, so v12 runs **only on Llama-32000
  backbones**: `tinyllama` and `mobilellama`. `smollm2`, both Qwen rows and `phi2` are
  excluded — three quarters of the §15e matrix. The recipe checks `SLM_KEY` up front
  and exits 1 with that explanation, rather than crashing after Stage 1 has already
  run.
- **VRAM.** The frozen 7B costs ~14 GB/GPU on top of the student, and because it is a
  plain attribute on `ElasticEngine` rather than a submodule, **ZeRO does not shard
  it**. A40-class or better; it does not fit a 23 GB A10 alongside training. The recipe
  raises its own `DEFAULT_GRES` to `gpu:A40:2` and prints a reminder to verify headroom
  on whatever hipster card is targeted before assuming node205's numbers transfer.

Neither v11-on-other-backbones nor v12 is queued locally — v12 in particular would be a
fourth `--exclusive` job against two eligible nodes, and it is A40-only, which is
exactly the contention §15d moved this work off the local cluster to avoid.

Sequencing for the **v11** ports from §15e still stands (SmolLM2 → Phi-2 → the Qwen
pair → MobileLLaMA), and
so does its caveat that the two Qwen rows need their own baseline before their numbers
mean anything. One addition: with vision LoRA off, **SmolLM2's existing v4 is already
the matched baseline** for its PARCEL run — v4 is vision-LoRA-off on the same grid, so
SmolLM2 v4 → v8-style is a clean one-change comparison and the cheapest real result
available on that cluster.

**v12 sequencing** is separate and short, since only two backbones are eligible:
`tinyllama` first — it has v8, v9, v10 and v11 to compare against, so a KD result is
immediately interpretable — then `mobilellama` only if the TinyLlama result justifies
it. Note `mobilellama` has no elastic baseline of any kind (§15e), so it would need its
v11 port run before its v12 number means anything.

v7-kd7b's cancelled `checkpoint-1000` (§15d) is **not** a resume point for v12: it was
trained under the v4-era setup, not v11. Treat v12 as a fresh run and that checkpoint
as deletable once v12 is under way.
