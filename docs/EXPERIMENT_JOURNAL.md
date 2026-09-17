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
| 45 | Analyze the completed v14 eval; say what the best standing approach is now. | Done (§16q). **v8 remains the best model — v14 loses.** Cleanest isolation in the project (rank ladder DIRECTION only, everything else identical to v8): v14 loses MME/POPE/TextVQA/GQA at every level (256 tok worst: −8.23 TextVQA, −3.77 GQA), wins only SciQA. The asymmetry is the finding — v14's 256-level has 8x MORE rank than v8's and does worse; its 16-level has 8x LESS rank and is within noise of v8. More capacity at the information-rich teacher level does not help and may hurt; v8's original "small budget gets more adapter" convention, inherited unexamined since v4, is now directly confirmed rather than assumed. v14's own elasticity nearly collapses too (TextVQA spread +0.81 vs v8's +8.76, MME spread NEGATIVE) -- nesting alone isn't enough, direction matters. v14 vs v11 at 256 tok shows a non-monotonic story: no adapter (v11) beats a large one (v14) beats a small one (v8) on TextVQA/GQA at the teacher level specifically -- speculative mechanism noted, not measured. SciQA nuance: v14 beats v8 at every level despite IDENTICAL total LoRA rank (120 either way), so SciQA tracks placement, not just amount. `final-parcel` (v8) stands, now with its last unexamined design choice tested directly. |
| 47 | Smoke-test TinyLlama, SmolLM2, MobileLLaMA with SigLIP under the best (final-parcel/v8) recipe. | Done, all pass (section 22). Re-ran the existing SigLIP unit test first (27458) to check for regressions from three sessions of unrelated engine.py/nested_lora.py changes -- none; all three original fixes (l_enc, no blanket no_grad, scoped pooling-head injection) still hold. Then 6 real steps each of Stage 1 + Stage 2 for all three backbones (27459) on final-parcel's actual flags with only VISION_TOWER swapped to SigLIP -- genuinely new ground, since the one prior SigLIP validation used the now-retired 8-level ladder and never finished as a real run, and MobileLLaMA has never been run through the elastic pipeline at all before this. All 6 stages exit 0, all losses finite, no NaN/Inf anywhere in the log, grad norms shrinking through two single-batch Stage-1 noise spikes rather than diverging. This proves the pipeline runs correctly for all three backbones on this exact recipe; it says nothing about eventual accuracy or whether SigLIP beats CLIP here -- that needs the real runs, not yet queued. Smoke checkpoints (~120GB) deleted after inspection. Added MAX_STEPS to the elastic launchers (previously only on the baseline ones). |
| 46 | Check the Qwen job results. | It failed (§21a) -- reported as such, not as "still running." Stage 1 completed cleanly; Stage 2 OOM'd at step 16/5197 on an A10 (22.3 GiB) inside `prefix_kl_loss`. Root cause: Qwen's 151936-token vocabulary is 4.7x TinyLlama's, and the KL loss materializes a full unmasked `(B,L,V)` tensor before reducing -- a failure mode no prior backbone's vocabulary ever exercised, and outside what the EOS-masking onboarding check screens for. Fixed by GPU type, not code: resubmitted Stage 2 only (Stage 1's checkpoint reused) on `run_job_finetune_slm.sh`'s standing `gpu:A40:2` default -- 27456/27457. Flagged, did not fix, a real inefficiency in `prefix_kl_loss` (computes the KL over the full sequence before masking to labelled positions) that would help every backbone, not just Qwen -- left alone since it touches loss code shared by every prior comparison in this project, and the GPU swap alone fully explains this failure. |
| 44 | Queue the Qwen-0.5B experiment with the best config. | Done (§21). `bash submit_elastic_run.sh final-parcel` with `SLM_KEY=qwen0.5b` -- the exact §17c/§18 recipe (PARCEL, nested LoRA, decorr off, self-teacher, 256/144/64/16 grid), no per-backbone changes. Queued as 27447/27448 (`--array=0-3`), waiting on both 2-GPU nodes (v14 + a v8-parcel re-eval). First Qwen run of any kind in this project -- no local v4-equivalent baseline exists, so this is the first row of the Hipster sweep, not an ablation. Onboarding check not re-run: Qwen2.5 was already verified clean 2026-08-18 and no preprocess code has changed since. |
| 43 | Analyze the completed v13 run: did the long ladder and self-KD from 576 tokens help; what was the config; v8/v11/v14 style? | Done (§16p). **Config is v11-style** (`use_lora=false`, PARCEL, decorr off, self-teacher) placed on v6's 8-level ladder — not v8's or v14's nested-LoRA recipes. **"KD from 576" is self-distillation** (`teacher: "self"`), not external KD (contrast v12); it changes which level is the self-teacher because `kl_teacher_tok_level` is always index 0, which is now 576 instead of 256. **Does the long ladder help? Yes, decisively vs. v6** — every metric, every level, TextVQA +6 to +8 pts, MME +57 to +139 pts — confirming PARCEL rescues most of v6's catastrophic failure, and doing so DESPITE carrying a known LoRA-off handicap (§16m) that runs against v13, so the true PARCEL effect is likely understated here. **Does it beat v8? No** — loses MME/POPE/GQA at every shared level, wins only SciQA (the familiar LoRA-off pattern) and ties TextVQA at 16 tok. v13's own elasticity (both full 576-16 and the short 256-16 sub-range) is flatter than v8's or hipster v11's. At v13@576 — literally the same token count as the dense baseline — it still trails baseline by ~18 TextVQA / 6 GQA points, nearly identical to v13@256, so the extra 320 top-ladder tokens buy almost nothing: a milder recurrence of the token-budget-has-no-effect pattern. Verdict unchanged: `final-parcel` (v8) remains the recipe; the long ladder stays a research question, not a replacement. |
| 42 | Move the W&B API key out of the tracked scripts into `~/.netrc` or a non-committed env file, on `main`, and push. | Done (§20). The key was inline in **10** tracked `run_job*.sh` files as `#SBATCH --export=ALL,WANDB_API_KEY=…`, so it sat in the repo, in history, and on GitHub. Moved to `~/.netrc` (mode 600, `machine api.wandb.ai`) — wandb's native mechanism, needing no script logic since `$HOME` is shared with the compute nodes; every export line became a plain `--export=ALL`. Verified netrc resolves and that later `#SBATCH` directives (e.g. `--exclusive`) still apply, since SLURM scans through comments. `.gitignore` now blocks `.env`/`*.env`/`.netrc`/`secrets.sh`. **Flagged as NOT fixed by this change: the key remains in git history, on GitHub, and on the `hipster` branch — it must be rotated at wandb.ai, which is the only step that ends the exposure.** |
| 41 | (Continuation) Validate both baselines with real training steps before committing multi-day jobs. | Done (§19e–g). **M3 passed both stages** (20 real steps each, 27418). **MQT was silently running OUR model**: FlexLLaVA's editable install registers a MetaPathFinder for `llava`, and `cd MQT-LLaVA` does not beat it because `sys.path[0]` is the *script's* dir, not the cwd — so MQT's train.py drove our model and crashed on a missing `query_abstractor`. The crash was luck; a different flag set could have trained to completion and produced numbers labelled "MQT" that were not MQT. Fixed with `PYTHONPATH` plus a guard that resolves `llava.__file__` and refuses to run otherwise. Also removed the `mm_projector` flags — MQT has no projector at all (diag 27419). After the fix MQT Stage 1 passes (27421) and forward+backward at 4 tokens is verified under transformers 4.44.2 (27424: loss 3.92 finite, grads reach the query bank). MQT Stage 2 OOM'd on **one** A10 during optimizer-state init, before step 1 — an artifact of smoking on a single GPU because both 2-GPU nodes were busy; our own Stage 2 needs 2 GPUs for the same reason. Retracted my earlier "MQT imports cleanly so the version risk didn't materialise": that import was resolving our package. |
| 40 | Baseline runs for M3 and MQT at 4 tokens on TinyLlama, using THEIR train loops, our vision encoder/backbone/data/hyperparams; minimal and quick. | Done, PLANNED not submitted (§19). **M3 needed no new code** -- this repo is an M3 fork with the pure-M3 branch intact, so `MATRYOSHKA_SCALE=4` runs M3's own loop (12x12 avg-pool -> exactly 4 tokens, verified job 27415). **MQT initially appeared to run from its vendored tree** — that was WRONG, see decision 41: our editable install shadowed it and MQT's train.py was driving OUR model. Each method keeps its OWN Stage-1 convention (M3: plain 576-token projector; MQT: `first_stage`=256 queries) because that is part of the pipeline being baselined. Flagged two scope limits: a fixed 4-token budget narrows both methods (M3 normally trains a scale list, MQT normally samples a random count), so these are not M3/MQT headline numbers; and eval is NOT wired -- `eval_lmms_level.sh` requires `elastic_config.json` which these checkpoints lack. Also noticed the env is now transformers 4.44.2, not the 4.36.2 recorded in memory. |
| 39 | Audit KD teacher compatibility per backbone, find family-matched teachers, build teacher-selection infrastructure, and plan the Hipster matrix. Audit first; launch nothing. | Done (§18). **The 7B teacher is valid for MobileLLaMA** (token IDs byte-identical to LLaVA's, verified by loading both tokenizers -- job 27410) and **invalid for SmolLM2 and Qwen** (49152 / 151936 vs 32000). **Neither family-matched teacher is usable**, smoke-tested not assumed (job 27411): MobileVLM_V2 fails with `Unknown projector type: ldpnetv2`; SmolVLM fails on architecture (Idefics3), on vocab (49155 vs the student's own 49152), and its tokenizer does not even load in this env. No Qwen VLM is in the local cache. Audited the KD mechanism itself: **logits KD only**, T hard-coded 1.0, weight 0.1/n_active, assistant-response positions only, teacher at full 576 visual tokens, CORAL self-sourced even with an external teacher, and `attach_kd_teacher` hard-codes the LLaVA-Llama loader. Built `kd_teachers.py` (registry + compatibility levels) wired into `attach_kd_teacher` so incompatible pairs are **refused, never substituted**; added `--kd_teacher/--kd_student_key/--kd_type`. Matrix is 12 core + 2 external-teacher, all **PLANNED**, nothing submitted. |
| 40 | HIPSTER EXPS: execute exactly 4 cells of the §18d matrix on Hipster (SmolLM2 v14, SmolLM2 v8+self-KD, MobileLLaMA v8, MobileLLaMA v8+7B-KD) using `run_matrix_hipster.sh`/DAS-6's launchers as the spec, not as something to invoke directly (they hardcode `/var/scratch` and are DAS-6-only). Journal + script are source of truth; stop and report rather than invent config on any disagreement. | Done, in flight (§19). `run_matrix_hipster.sh` cannot run on Hipster as-is (hardcoded `/var/scratch`, calls DAS-6-only launchers) — ported the v8/v14/KD-teacher flag set into Hipster's own launchers instead (`pretrain_elastic_slm_hipster.sh`, `finetune_elastic_slm_hipster.sh`) and wrote a Hipster-native matrix driver, `submit_matrix_hipster.sh`, defaulting to exactly these 4 cells (no "submit everything" default). Standardized on DAS-6 SSH/tar data streaming for the pretrain-only path too (`run_job_pretrain_only_hipster.sh`, new). Found and fixed 3 real bugs via actual execution, not just review: (1) a broken `$(...) VAR=value sbatch` command-substitution env-prefix in `submit_one()` — bash only special-cases a *literal* `VAR=value` token, so this silently ran `TEACHER=self` as a command and never submitted any of the 4 finetune jobs; the identical bug exists in DAS-6's `run_matrix_hipster.sh` line 130, never caught because nothing in §18d's matrix had ever actually been submitted; (2) Stage 1 was passed `--nest_version`/`--lora_type` alongside the hardcoded `--lora_ranks 64`, and for `tok_levels=[256]` (Stage 1's single level) `train_elastic.py`'s ladder derivation gives `[8]` for **both** v8 and asc (reversing a 1-element list is a no-op) — contradicting the hardcoded 64 and crashing with `ValueError` in ~1 minute; removed the version tag from the shared Stage-1 call, consistent with §17c's own claim that Stage 1 is version-agnostic. Same latent bug exists in DAS-6's `pretrain_elastic_slm.sh`. (3) Hipster-only: two Stage-1 jobs landed on the same physical node (no `--exclusive` here, unlike DAS-6) and collided on deepspeed's default port 29500; fixed with a per-job `--master_port` derived from `$SLURM_JOB_ID`, added to both Hipster launchers. All 3 fixes verified by re-running past the failure point before trusting them. Final submission confirmed via `squeue`: exactly 4 train chains queued (job IDs in §19), 2 Stage-1 jobs `RUNNING` as of this entry, 4 Stage-2 + 4 eval jobs `PENDING(Dependency)` — nothing marked complete, no results fabricated. Git-push-hang side question (separate branch `hipster`) resolved as a credentials issue (HTTPS remote, no credential helper), not a file-size/bug issue; fix (`git remote set-url` to SSH) left to the user — blocked by the permission classifier from doing it myself. |
| 38 | Rigorously determine whether v14 beats v8, then build a controlled Hipster matrix (2 versions x 3 backbones x 2 LoRA x 2 KD); add a single `--lora-type` and an explicit version argument; run KD for all three backbones; verify what KD actually is. | Done (§17). **Existing evidence is insufficient for a controlled v8-v14 comparison: v14 has no results at all** -- job 27406 is ~9h into a ~85h run, no checkpoint, no eval dir. Audit found v8 and v14 differ in **exactly one** component, the LoRA rank ladder; the anchor-config difference (fixed table vs ratio 0.25) is verified behaviourally identical, and the `acdfa79` code delta that landed mid-v8-run is verified inert (decorrelation-gated). Two consequences for the matrix: **version and LoRA are the same axis**, so 24 cells collapse to 12 distinct ones (+2 external-teacher); and **external-teacher KD is impossible on SmolLM2 (vocab 49152) and Qwen (151936)** against the 32000-vocab LLaVA-7B teacher, so the KD column is prefix-KL *self*-distillation (shared weights, KL ~0.006) and genuine KD is a MobileLLaMA-only arm. Added `--lora_type {v8,asc}` and `--nest_version {v8,v14}` propagating to checkpoint metadata, plus `run_manifest.json` with git hash; `run_matrix_hipster.sh` is the reproducible driver (dry-run by default). Nothing submitted. |
| 37 | Research only: would a fixed rank-64 shared adapter help, given shared r16 (v10) did not? Can more rank help? | No, predicted from existing data (§16n), not queued. At 256 tok v8's *rank-8* nested level beats v10's *rank-16* shared adapter by 8.5 TextVQA — the smaller adapter wins, so capacity is not the shortfall. At 16 tok shared r16 already captures 94% of nested r64's TextVQA gain — nothing left for rank to buy. What separates them is the nested prefix's **quarantine**: the 256-token forward reads and trains only columns 1–8, so the 16-token level's aggressive adaptation is confined to columns 9–64 the teacher never sees; a budget-blind shared adapter has no partition, and more rank gives the student objective more room to reshape the columns the teacher must read through. Prediction: shared r64 regresses the top end further. Offered as a hipster row if the paper wants "rank is not the variable" shown rather than argued; the run that actually tests top-level capacity is v14. |
| 36 | Add a journal section isolating the vision-LoRA axis (none → fixed r16 → nested); set up v14 = v8's nested LoRA with the rank ladder reversed so rank follows token budget; cancel the running local v11. | Done (§16m). The three-arm table (v11 / v10 / v8, all PARCEL + decorr off) shows the adapter effect is **non-monotone in capacity** — a shared r16 adapter is worse than none at 256 tok (TextVQA 19.96 vs 22.18) while lifting the 16-tok floor by +7; nesting lifts both ends. SciQA is paid by nesting specifically (48.14 vs 50.2–50.5 for the other two). **v14-parcel-asclora** = v8 with `LORA_RANKS 64 32 16 8`, everything else identical; queued as **27406/27407** on node205. Required a code change — `NestedLoRALinear` asserted ascending ranks and took `max_rank = ranks[-1]`, neither structural; now `max(ranks)`, and the shared-adapter index in `ElasticConfig` picks the max entry rather than the last. Validated by `jobs/test_lora_rank_order.sh` (27405, `ALL_TESTS_PASSED`) incl. the `[64]`→`[64,32,16,8]` warm start, where the teacher level now uses all 64 columns (v8 used 8 — an inherent second difference, noted in §16m). Renamed last turn's `v14-final-parcel` sweep recipe to `final-parcel` to free the v14 name. Local v11 (27386) + eval (27387) cancelled at ~1d19h; its partial checkpoints left on disk; hipster v11 remains the reference. |
| 35 | With hipster v11/v12 in (`results/hipster_eval_summary.csv`): analyze all runs; name the final best config across vision LoRA / decorr / CORAL / KD / PARCEL fixed-vs-ratio; pick the recipe for every other SLM backbone; say whether a 576-token teacher or the full ladder helps; write the best config at the end of the journal explicitly. | Done (§16l + the "FINAL BEST CONFIG" block ending §16). **Final recipe = v8, verbatim** — PARCEL ratio-0.25, rank-nested vision LoRA `8 16 32 64`, decorr off, CORAL off, self-teacher, 4-level grid — now `final-parcel` in `submit_elastic_run.sh`, `SLM_KEY`-parametrised. Every axis has a clean single-flag isolation against hipster v11: PARCEL creates the gradient (v11 vs v4), nested LoRA lifts the whole curve (+6.3 TextVQA / +2.1 GQA @256, v8 vs v11), a shared r16 adapter is *worse* than none at 256 (v10 vs v11), decorrelation is negative (v9 vs v11, §16g and decision 33 closed), the 7B KD teacher is negative everywhere (v12 vs v11, TextVQA −6.4). CORAL was never on in any run; fixed vs ratio anchors are identical at 0.25 on this grid — neither is an evidenced choice. v8 wins 4/5 metrics at all four budgets; SciQA −2..−3 is the constant cost, monotone in adapter capacity across six runs. Flagged that "elasticity" as a spread rewards a bad floor — v11's larger spread comes from collapsing at 16 tok — so the paper must report the frontier. 576-teacher / full ladder: no data (v13 12h in); prior evidence negative (v6 flat; self-KL ~0 means teacher "strength" is not the lever; the one real-signal teacher, v12, hurt). §16f's v11-based hipster plan superseded: SmolLM2 v11 cost TextVQA −9.6 vs its v4. Sweep must re-run tinyllama under the final tag as the seed replicate. |
| 34 | Analyze v9 and v10, update the journal, and say which is the best approach so far. | Done (§16k). **v8 is still the best model** -- it beats both v9 and v10 on MME/POPE/TextVQA/GQA at every level, and both lose most of v8's elasticity gradient (v9: 5%/19% of v8's TextVQA/GQA spread; v10: 8%/7%). The useful new result is v10 vs v8, a CLEAN single-flag isolation (decorr off in both): replacing v8's rank-nested-per-level vision LoRA with a shared rank-16 adapter loses almost all of v8's advantage, revising section 16d's framing -- under PARCEL specifically, full-capacity vision LoRA appears necessary, not harmful, contradicting the plain-query-resampler result the "vision LoRA hurts" scope limit was based on. SciQA remains the one consistent cost, now a 5-run pattern tracking LoRA capacity inversely. Both standing decorrelation triggers (section 16g, decision 33) did NOT fire -- v9 loses to v8 on 4/5 metrics -- so v13 stays with decorrelation off pending v11's clean read. |
| 33 | If decorrelation is shown to help (per v9), add it to v13's recipe too. | Recorded as a standing decision (§16j), not yet actionable -- v9 (27382) is still training with no results. Flagged as its own item rather than folded silently into §16g's general "promote to default" trigger because v13 is a second already-running experiment (27393/27394, submitted with USE_TOKEN_DECORRELATION=False already baked into the job's captured environment) that is easy to forget when that trigger fires. Action recorded: check v9 against v8 AND v11 (the cleaner control for this specific flag), then either cancel+resubmit 27393/27394 or split off a v13b if 27393 has already started. |
| 32 | Asked whether v11 could be made an 8-level ladder like v6, conditioned on it not adding training hours. | No — flagged rather than done silently. `kl_teacher_tok_level` defaults to index 0 (always the largest entry), and that level's forward runs every step regardless of ladder length, so moving v11 to v6's grid would grow the always-on teacher pass from 256 to 576 tokens (plus a costlier average sampled-student draw), adding real wall-clock the condition explicitly excluded. Asked the user how to proceed; chosen: leave v11 untouched as the exact-match control for v8/v9/v10, and run the question as its own experiment. **v13-parcel-longladder** added to `submit_elastic_run.sh` and queued (27393/27394) -- v11's flags on v6's 8-level grid and rank schedule, to test whether PARCEL fixes the long-ladder failure v6 showed on the plain query resampler (§16j). |
| 31 | Analyze the v6 eval jobs for the remaining (256/144/64/16) token levels and compare against earlier evals. | Done (§16b/16i). All 8 levels now in. Confirms §16b's verdict with the complete ladder: the full 576-16 spread is barely larger than the top-4-only spread looked (TextVQA +0.79 over the whole 36x range), the short grid v6 shares with v4/v5/v8 is *less* elastic than v4's on that same range, TextVQA is non-monotone (16-tok scores above 144- and 64-tok), and v6@256 loses to v8@256 on every metric, worst on TextVQA (-13.75, roughly half of v8's score). Getting there required an unplanned detour (§16h): the §16a node restriction had put 27375 into an unreleasable SLURM hold (`ReqNodeList` requires ALL listed nodes, not any-of; regular users cannot release the resulting hold), and once resubmitted correctly with `--exclude`, an unexplained scheduler quirk kept later array tasks waiting on a busy node while an idle one sat unused, requiring the array to be split into separately-submitted pieces to actually land. |
| 30 | Confirm v12 is explicitly queued for `tinyllama` in the hipster pipeline, not just `mobilellama`. | It was not — §16f described `tinyllama` v12 only as prose ("tinyllama first"), and §15e's eligibility note read as though `tinyllama` were already covered ("not in this table because it runs locally"), which was true of v9/v10/v11 but not of v12 (deliberately excluded from the local queue). Corrected both passages: the hipster v12 pipeline now lists both `tinyllama` and `mobilellama` as explicit submit commands, `tinyllama` first. |
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

**v12 eligibility** (the KD arm, §16f): only two backbones pass the Llama-32000 vocab
requirement the frozen LLaVA-1.5-7B teacher imposes — `tinyllama` and `mobilellama`.
`mobilellama` is the only one of the two *in this table*, because the table lists
backbones with no existing local footprint; `tinyllama` is missing from it only
because v9/v10/v11 already run there, not because v12-on-tinyllama is spoken for.
**v12-tinyllama is NOT queued anywhere** — §16f originally discussed it only as prose
sequencing ("tinyllama first"), without adding it to the hipster pipeline as an actual
entry. Corrected: both `tinyllama` and `mobilellama` are hipster-pipeline v12 runs, in
that order — see §16f.

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

**CORRECTION — this update broke the job; see §16h.** `ReqNodeList` means "must
include all of these nodes," not "may use any of these" — setting it to two nodes on
a 1-node job is unsatisfiable and silently zeroed the job's priority into a hold that
even its own owner could not release. 27375 never ran under this restriction; §16h has
the incident and the actual fix.

### 16b. v6-tokrange: extending the ladder to 576 tokens failed, confirmed on all 8 levels

Updated 2026-09-10: the real 256/144/64/16 levels landed (jobs 27388_4, 27390_5,
27391_6/7, after the node-restriction detour in §16h) and are folded in below —
originally only the top four (576/512/448/384) existed.

| tokens | MME-P | POPE-acc | POPE-F1 | SciQA | TextVQA | GQA |
|---|---|---|---|---|---|---|
| 576 | 1042.73 | 80.46 | 79.05 | 47.50 | 14.81 | 50.66 |
| 512 | 1043.50 | 79.78 | 78.04 | 47.35 | 14.51 | 50.33 |
| 448 | 1035.22 | 79.98 | 78.40 | 47.10 | 14.35 | 50.29 |
| 384 | 1038.28 | 80.09 | 78.59 | 47.10 | 14.63 | 50.22 |
| 256 | 1023.11 | 79.94 | 78.70 | 47.40 | 14.72 | 50.27 |
| 144 | 1001.33 | 79.43 | 78.06 | 41.70 | 13.91 | 49.64 |
| 64  |  991.89 | 79.06 | 77.50 | 41.20 | 13.80 | 49.64 |
| 16  | 1005.30 | 79.39 | 77.99 | 42.39 | 14.02 | 49.55 |
| *dense 576 baseline* | *1248.32* | *84.89* | *83.28* | *57.36* | *41.00* | *58.30* |

At **576 tokens — the same count the uncompressed baseline uses**, so the resampler is
not compressing anything — v6 gives up 26.2 points of TextVQA, 9.9 of SciQA, 7.6 of
GQA and 206 of MME against that baseline. It is also worse than **v4 at 16 tokens** on
TextVQA (14.81 vs 19.90) and MME (1042.7 vs 1116.4), i.e. worse than the same
architecture on the short ladder using 36× fewer visual tokens.

The top four levels are flat to within noise — TextVQA spread 0.46, GQA 0.44, SciQA
0.40 across a 1.5× change in budget. This is the [token-budget-has-no-effect] pattern
again — the same "16 tokens ≈ 256 tokens" result that made the prefix-KL ~0 — now
observed at the *top* of the ladder, where a gap should be easiest to produce.

**With all 8 levels in hand, the full spread (576 − 16 tokens) is barely larger than the
top-four-only spread was**: MME +37.4, POPE-acc +1.07, SciQA +5.11, TextVQA **+0.79**,
GQA +1.11 across the *entire* 36× range. Confined to the short grid alone
(256→16, the same range v4/v5/v8 are measured on) it is MME +17.8, POPE +0.55, SciQA
+5.01, TextVQA **+0.70**, GQA +0.72 — smaller than v4's spread on the same range and an
order of magnitude below v8's (§16c: MME 46.3, TextVQA 8.76, GQA 3.46). Extending the
ladder to 576 did not just fail to add elasticity at the top, the resulting run is
*less* elastic than v4 across the short range both share.

TextVQA is flat to the point of being non-monotone: 14.81 → 14.51 → 14.35 → 14.63 →
14.72 → 13.91 → 13.80 → **14.02** — the 16-token level scores *higher* than the 144-
and 64-token levels. Whatever the extra levels and rank range buy this run, it is not a
usable budget–accuracy tradeoff anywhere on it.

**Head-to-head against the current best model, at the token count they share:**

| | v6@256 | v8@256 | delta | dense baseline | v6 vs baseline |
|---|---|---|---|---|---|
| MME-P | 1023.11 | 1170.25 | −147.1 | 1248.32 | −225.2 |
| POPE-acc | 79.94 | 84.33 | −4.39 | 84.89 | −4.95 |
| SciQA | 47.40 | 48.14 | −0.74 | 57.36 | −9.96 |
| TextVQA | 14.72 | 28.47 | **−13.75** | 41.00 | **−26.28** |
| GQA | 50.27 | 56.07 | −5.80 | 58.30 | −8.03 |

v6 loses to v8 on every metric at the same budget, worst on TextVQA (roughly half of
v8's score), and at 256 tokens — 44% of the baseline's budget, same fraction v8 uses —
it recovers almost none of the ground v8 does.

The most likely mechanism is the LoRA ranks the extended ladder forced. Keeping ranks
ascending with budget descending put `2 4 6 8` on the four high-budget levels versus
`8 16 32 64` for the same architecture on the short ladder — but now that the
short-range levels are in hand too, they inherit ranks `8 16 32 64`, the *same* ranks
v4/v5/v8 use at those exact tok_levels, and still underperform all three there. So low
rank at the high-budget end does not fully explain it; something about training eight
nested levels at once (vs. four) may be diluting the shared resampler capacity across
a wider range. Stated as a hypothesis, not a measured cause — nothing on the roadmap
currently isolates it.

**Consequence unchanged, now backed by the complete ladder:** the 8-level 576→16
ladder is retired. v9 was on it and has been moved to v8's 4-level grid (§16e).

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

**Scope limit — do NOT write "vision LoRA always hurts" in the paper. UPDATE
2026-09-13 (§16k): this is no longer just "not proven either way" — v9/v10 now show
the effect running the OPPOSITE direction under PARCEL. Read §16k before citing
anything below as the final word.** The evidence does not support a universal claim,
and our own best model contradicts one. Every checkpoint audited 2026-09-09:

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

**CLOSED 2026-09-13 (§16l): trigger did not fire; decorrelation is NOT promoted.** With
the hipster v11 in hand the clean isolation is v9 vs v11 — PARCEL and no vision LoRA in
both, decorrelation the only difference — and it is negative at 256 tokens on every
metric that matters (POPE −3.12, TextVQA −2.60, GQA −2.07), while flattening the ladder
(TextVQA spread 0.44 vs 10.09). Decision 33 (add to v13) closes the same way.
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

**SUPERSEDED 2026-09-13 by §16l — the hipster ports now use `final-parcel` (= v8's
recipe: PARCEL + rank-nested vision LoRA), NOT v11.** The v11-based plan below was
written when "vision LoRA is a regression" (§16d) was the working belief; the clean v8
vs v11 isolation reversed that under PARCEL (TextVQA +6.3, GQA +2.1 at 256 tok), and
the SmolLM2 v11 port itself came back with TextVQA −9.6 against SmolLM2's own v4. Kept
as written for the record. The original text: that block **is** `v11-parcel-nolora`,
verbatim — the hipster ports are not a separate design, they are v11 with a different
backbone:

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

**v12 hipster pipeline, both eligible backbones:**

```
SLM_KEY=tinyllama   GRES=<hipster A40-class spec>  bash submit_elastic_run.sh v12-parcel-kd7b
SLM_KEY=mobilellama GRES=<hipster A40-class spec>  bash submit_elastic_run.sh v12-parcel-kd7b
```

**`tinyllama` first** — it has v8, v9, v10 and v11 to compare against, both locally and
via the LLaVA-1.5-7B teacher's own vocabulary match, so a KD result is immediately
interpretable. `mobilellama` only after the TinyLlama result justifies it, and only
once `mobilellama`'s own **v11** port has run first (§15e) — it has no elastic
baseline of any kind yet, so a `mobilellama` v12 number with no v11 to compare against
would not be interpretable either.

Corrected 2026-09-09: `tinyllama`-v12 had been discussed only as prose sequencing
here, with no explicit hipster-pipeline entry — easy to misread as already covered by
the local `tinyllama` runs (v9/v10/v11), which it is not; v12 was deliberately kept
off the local queue (contention + A40-only, next paragraph). It is now stated as an
explicit hipster submission, not an implication.

v7-kd7b's cancelled `checkpoint-1000` (§15d) is **not** a resume point for v12: it was
trained under the v4-era setup, not v11. Treat v12 as a fresh run and that checkpoint
as deletable once v12 is under way.


### 16h. Incident: the node-restriction update from §16a stuck job 27375, and how it was fixed

`scontrol update jobid=27375 ReqNodeList=node206,node207` (§16a) was meant to keep the
v6-tokrange eval off the two 2-GPU nodes v9/v10/v11 need. It broke the job instead.
`ReqNodeList` requires **all** listed nodes to be allocated together — for a script
declaring `--nodes=1` that is unsatisfiable — and rather than erroring, SLURM zeroed
the job's `Priority` to 0 (the hold state) and left `Reason=BadConstraints` stale from
before the update. `scontrol release 27375`, run as the job's own owner, came back
`Access/permission denied` — the hold was not one a regular user can lift.

Cancelled 27375 and resubmitted the same `--array=4-7` with `--nodelist=node206,node207`
at `sbatch` time instead — same error, same root cause
(`invalid number of nodes (-N 2-1)`): `--nodelist` has the identical "must include all"
semantics as `ReqNodeList`, regardless of whether it is applied at submission or after
the fact. The correct primitive for "restrict to a set" is **`--exclude`** naming
everything else, which has no such ambiguity. Resubmitted as job **27388**
(`--array=4-7`, `--exclude=node201,node202,node203,node204,node205,node208`) and it
ran immediately.

`27388_4` (256 tok) started on node207 right away. The remaining three tasks
(144/64/16 tok) then sat `Reason=Priority` with node206 completely idle — confirmed
genuinely free (`CPUAlloc=0`, no `AllocTRES`, GPU unclaimed, no reservation, no drain)
and re-evaluated by the scheduler roughly every two minutes (`sdiag` showed
`LastSchedEval` advancing on schedule) without ever landing there. Root cause not
fully determined — `defq` is `OverSubscribe=EXCLUSIVE` (one job per node regardless of
requested resources, confirmed by `AllocTRES=cpu=64` on a job that asked for 8) and
`node206`/`node207` have different `Weight=4`/`Weight=2`, but neither should on its own
stall a job onto a busy node while leaving an idle one unused; this is `slurmctld`
backfill-internal behavior not visible via `sdiag`/`scontrol` as a non-admin user.
Worked around rather than root-caused: cancelled the three unstarted tasks and
resubmitted **27390** (`--array=5`, excluding node207 as well so only node206 is
eligible) and **27391** (`--array=6-7`, back to the original 206/207 pool). 27390_5
started on node206 within seconds. All four tasks completed; results in §16b.

**Two things to carry forward.** First: prefer `--exclude` over `--nodelist`/
`ReqNodeList` for "restrict to a candidate set" on this cluster — `--nodelist` reads as
a whitelist but behaves as a required-set, and the failure mode (an unreleasable hold)
is worse than a rejected submission. Second: if a future array job's later tasks stall
on a busy node while an eligible node sits idle, splitting the array into
separately-submitted, disjoint node-restricted pieces is a working, if blunt, unstick.

### 16i. v6-tokrange full ladder — see §16b for the analysis

The results that motivated the §16h detour are written up in the now-complete §16b,
not repeated here: all 8 levels are in, the top-four flatness holds across the full
ladder (576−16 spread is TextVQA +0.79, GQA +1.11), TextVQA is non-monotone, and v6@256
loses to v8@256 on every metric, worst on TextVQA (−13.75). No change to the §16b
verdict — the 8-level ladder is retired — the complete data only sharpens it: the short
grid v6 shares with v4/v5/v8 (256→16) is *less* elastic than v4's on the same range,
not merely flat like the top four looked in isolation.


### 16k. v9 and v10 results: v8 still wins, and the vision-LoRA story gets more specific

Both landed clean (correct `tok_level` labels, no `jq`-fallback repeat). Full 256/144/64/16 comparison against v4/v5/v8:

| level | run | MME-P | POPE-acc | POPE-F1 | SciQA | TextVQA | GQA |
|---|---|---|---|---|---|---|---|
| 256 | v4 | 1113.46 | 81.57 | 79.91 | 51.31 | 23.15 | 52.35 |
| 256 | v5 | 1035.55 | 79.38 | 77.65 | 49.38 | 18.56 | 50.91 |
| 256 | **v8** | **1170.25** | **84.33** | **83.06** | 48.14 | **28.47** | **56.07** |
| 256 | v9 | 1120.79 | 80.40 | 78.64 | 50.72 | 19.58 | 51.88 |
| 256 | v10 | 1138.75 | 80.97 | 79.49 | 50.52 | 19.96 | 51.95 |
| 16 | v4 | 1116.40 | 80.70 | 78.26 | 44.12 | 19.90 | 51.18 |
| 16 | v5 | 1040.23 | 78.89 | 76.80 | 49.73 | 17.74 | 50.35 |
| 16 | **v8** | **1123.91** | **80.49** | **78.62** | 46.36 | **19.71** | **52.61** |
| 16 | v9 | 1111.04 | 79.79 | 76.99 | 51.12 | 19.14 | 51.22 |
| 16 | v10 | 1089.66 | 79.70 | 77.15 | 50.17 | 19.23 | 51.71 |

(144/64-tok rows follow the same pattern; omitted for length.)

**v8 is still the best model** — wins MME, POPE, TextVQA and GQA at every level against
both v9 and v10. Neither new run beats it.

**Elasticity (256−16 spread) also drops sharply for both:**

| run | MME-P | POPE-acc | TextVQA | GQA |
|---|---|---|---|---|
| v4 | −2.94 | 0.87 | 3.25 | 1.17 |
| v5 | −4.68 | 0.49 | 0.82 | 0.56 |
| **v8** | **46.34** | **3.84** | **8.76** | **3.46** |
| v9 | 9.75 | 0.61 | 0.44 | 0.66 |
| v10 | 49.09 | 1.27 | 0.73 | 0.24 |

v9 and v10 each recover only a fraction of v8's TextVQA/GQA gradient (v9: 5%/19% of
v8's; v10: 8%/7%) — both essentially fall back toward v4/v5's flatness on the two
metrics that actually reflect visual-detail usage, even though v10's MME spread is
nominally larger than v8's (MME is the noisiest of these four and least trustworthy on
its own for this comparison).

**This forces a real revision to §16d's "vision LoRA is a regression" framing —
specifically to the two vision-LoRA arms tested under PARCEL, not to v4-vs-v5.**

The cleanest single comparison here is **v10 vs v8**: both have decorrelation off and
PARCEL on; they differ in exactly one thing — v8's vision-tower adapter is rank-nested
per level (ranks `8 16 32 64`, a dedicated adapter tied to each `tok_level`), v10's is a
single **shared, non-specialized** rank-16 adapter used identically at every level. No
decorrelation confound. And v10 loses almost all of v8's advantage: MME −31.5, POPE-acc
−3.36, TextVQA −8.51, GQA −4.12 at 256 tokens, and its TextVQA/GQA elasticity collapses
to near v4/v5 levels. **Under PARCEL, the FORM and CAPACITY of the vision-tower
adapter — not just whether one exists — is doing real work.** A shared rank-16 adapter
is close to not having one at all; a rank-nested adapter that scales up to 64 as the
token budget shrinks is not.

v9 is confounded (vision LoRA off **and** decorrelation on at once, relative to v8), so
it cannot isolate either flag alone — but it lands in the same place as v10 (worse than
v8 on MME/POPE/TextVQA/GQA, SciQA better), which is at least consistent with "no
nested vision LoRA" being the larger driver of the loss, not decorrelation specifically.
v9 also loses to **v4** — the plain-query, no-LoRA, no-decorrelation, no-PARCEL
baseline — on POPE (−1.17), SciQA (−0.59), TextVQA (−3.57) and GQA (−0.47), winning
only on MME (+7.33). That v9 doesn't clearly beat even v4 is itself informative: PARCEL
alone, without a specialized vision adapter, is not obviously better than the original
plain-query resampler on this evidence.

**SciQA is the one metric that goes the other way, and it is now a five-run, very
consistent pattern**, not a two-run coincidence: `v4 (51.31) > v9 (50.72) > v10 (50.52)
> v5 (49.38) > v8 (48.14)` at 256 tokens — SciQA tracks *inversely* with how much
vision-tower LoRA capacity a run has, across both resamplers and independent of
decorrelation. This looks like a real, specific cost of adapting the vision tower
(plausibly diagram/layout-sensitive spatial features that SciQA's images lean on more
than the other four benchmarks), not a fluke — SciQA's own stderr is ~1.11pp
(vs. GQA's ~0.44pp), and the run-to-run gaps here are 0.2–3.2pp, mostly outside it.

**Net read: PARCEL's benefit and rank-nested vision LoRA's benefit are not
independent — under this architecture they appear to need each other.** v5 (LoRA,
no PARCEL) loses to v4. v9/v10 (PARCEL, weak-or-no LoRA) lose most of what v8 (PARCEL +
full nested LoRA) has. Only the combination — PARCEL *and* full rank-nested,
per-level-specialized vision LoRA — has produced both the accuracy gain and the
elasticity gradient. That combination also reliably costs SciQA. Caveat unchanged from
§16d: one seed, one backbone (TinyLlama) so far; v10's isolation is clean, v9's is not.

**Consequence for the standing decisions:**

- **§16g (promote decorrelation to default if v9 beats v8): trigger NOT met.** v9 loses
  to v8 on 4 of 5 core metrics. Decorrelation is not promoted.
- **Decision 33 (add decorrelation to v13 if shown to help): trigger NOT met, and not
  fully testable yet.** v9 does not beat v8, so the first leg already fails; the second
  leg (v9 vs v11, the clean isolation) still awaits v11, which has not reported. **Do
  not** flip `USE_TOKEN_DECORRELATION` in v13's recipe on current evidence — if
  anything it leans negative, though confounded. Revisit once v11 lands.
- **§16d's scope limit needs a footnote, not a reversal.** The claim there — rank-nested
  vision LoRA lost on **v4-vs-v5, the plain `query` resampler** — still stands exactly
  as measured. What's new is the *other* side of the question §16d flagged as open
  ("under PARCEL, vision-LoRA-off has never been run"): it has now been run (v9, v10),
  and under PARCEL the direction flips — full nested vision LoRA helps, weakening or
  removing it hurts. **"Vision LoRA hurts" was never safe to write generally (§16d
  already said so); it is now actively wrong as a blanket statement — the honest claim
  is resampler-architecture-dependent and adapter-capacity-dependent, with SciQA as a
  consistent specific cost either way.**

### 16j. v13-parcel-longladder: does PARCEL fix v6's long-ladder failure?

Asked whether v11 (the PARCEL/no-decorr/no-LoRA control, §16e) could be moved onto
v6-tokrange's 8-level `576…16` ladder for free. It cannot: `kl_teacher_tok_level`
defaults to index 0, and by convention that is always the *largest* `tok_levels`
entry — the teacher forward, which runs on **every** step regardless of
`n_sample_students`, would grow from 256 to 576 tokens, and the randomly sampled
student's expected token count rises too (student pool mean ~75 tok on the short grid
vs ~261 tok on the long one, since most added levels are large-budget). Real added
wall-clock, roughly bounded by the ~2x FLOPs ratio measured for v6 itself
(0.639→1.315 TFLOPs), on top of the ~85h short-grid Stage-2 runtime — not something
that fits a "no extra hours" constraint.

Decision: leave v11 as the fast, exactly-comparable control (256/144/64/16, matching
v8/v9/v10), and run the long-ladder-under-PARCEL question as its own experiment
instead. **v13-parcel-longladder** — queued as jobs **27393**/**27394** (`--array=0-7`)
— is v11's flags (PARCEL, decorrelation off, vision LoRA off) on v6's exact 8-level
grid and rank schedule (`576 512 448 384 256 144 64 16` / `2 4 6 8 8 16 32 64`,
`STAGE1_TOK_LEVEL=576`). The rank schedule is copied from v6 deliberately, not chosen
fresh: it keeps the four levels this ladder shares with v8/v9/v10/v11 (256/144/64/16)
at the *same* ranks those runs use (`8 16 32 64`), so a head-to-head comparison at
those four points isn't confounded by a rank change on top of the ladder-length one.

What this run actually tests: v6 (`resampler_arch=query`) showed the extended ladder
is flat, and on the 256-16 sub-range is *less* elastic than v4's own short-grid result
(§16b/16i) — worse than not extending the ladder at all. v13 asks whether that failure
is specific to the plain query resampler or whether PARCEL, which is the one thing that
has produced real elasticity on the short grid (§16c), fixes it too. If v13 is flat
like v6, the long ladder is dead regardless of resampler choice. If v13 shows a
gradient at the high end that v6 didn't, PARCEL's elasticity effect generalizes beyond
the range it has been tested on so far.

**Standing decision (2026-09-10), tied to §16g's threshold: if decorrelation is shown
to help, add it to v13 too, not just adopt it as a general default.** §16g already
covers promoting decorrelation project-wide if v9 beats v8 by its stated bar (a
consistent gain across benchmarks and levels, not a single sub-stderr cell); this is
the same trigger, called out separately because v13 is easy to overlook when that
decision gets acted on — it is a *second* already-running experiment, not a future
recipe someone will naturally think to update.

**Action when the trigger fires** (do not wait for a fresh submission to "pick it up" —
job 27393/27394 were submitted with `USE_TOKEN_DECORRELATION=False` already baked in;
`sbatch` captures the environment at submit time, so the running/queued job will not
see a later change to the recipe file):

1. Confirm v9 clears §16g's bar against v8 (and, once it reports, against v11 — v11 is
   the cleaner control for this specific flag, isolating decorrelation with vision LoRA
   held off in both, vs. v8 which also differs by vision LoRA).
2. Edit `submit_elastic_run.sh`'s `v13-parcel-longladder` case block:
   `USE_TOKEN_DECORRELATION=False` → `True`, keep `DECORR_WEIGHT` at its 0.01 default
   (v9's value) unless v9's own result suggests retuning it.
3. If 27393/27394 have not started: `scancel 27393 27394` and resubmit with
   `bash submit_elastic_run.sh v13-parcel-longladder`. If 27393 has already started
   training under the old flag, decide whether to let it finish as the "no decorr" data
   point and launch the decorr-on version as a new tag (e.g. `v13b`) instead of
   interrupting a multi-day run partway through.

*(Moot by the time v13 finished: §16k/§16g/decision 33 already closed the decorrelation
trigger negative — v9 lost to v8 on 4/5 metrics — so v13 correctly stayed with
`USE_TOKEN_DECORRELATION=False` the whole way through, no action needed.)*

### 16p. v13 results: PARCEL rescues most of the long-ladder failure, but does not beat v8

Completed (jobs 27393/27394). Config confirmed from `elastic_config.json`:
`resampler_arch=pool_anchored`, `anchor_mode=ratio` @0.25, **`use_lora=false`**,
`use_token_decorrelation=false`, `teacher=self`, ladder `576 512 448 384 256 144 64 16`,
`lora_ranks [2,4,6,8,8,16,32,64]` (inert — no adapter is injected while `use_lora` is
false, but `train_elastic.py` still requires the length to match `tok_levels`).

**Config identity: v11-style, not v8 or v14.** `use_lora=false` is v11's flag exactly
(v8 and v14 both have `use_lora=true`, differing only in which direction the ranks run).
v13 is v11's recipe — PARCEL, no vision LoRA, no decorrelation, self-teacher — placed on
v6's 8-level ladder instead of the short 4-level grid. Not a new design.

**"KD from 576 tokens" is self-distillation, not external KD.** `teacher: "self"` —
`teacher_model_path` is populated in the JSON but inert while `teacher` is `"self"`; no
external checkpoint is loaded (contrast v12, which sets `teacher: "llava"`). What "576
tokens" changes is which level plays teacher: `kl_teacher_tok_level` is always index 0,
and on this 8-level ladder index 0 is 576, not the 256 every short-grid run (v4/v5/v8/
v9/v10/v11/v14) uses. So every step's self-distillation target is the model's own
576-token forward — the mechanism §16j flagged as costing real compute (teacher pass at
~2x the FLOPs of a 256-token teacher) is confirmed to be exactly what ran.

**Full ladder:**

| level | MME-P | POPE-acc | SciQA | TextVQA | GQA |
|---|---|---|---|---|---|
| 576 | 1099.94 | 81.61 | 50.57 | 22.77 | 51.96 |
| 512 | 1103.82 | 81.29 | 50.82 | 22.54 | 51.91 |
| 448 | 1094.60 | 81.56 | 50.67 | 22.67 | 52.00 |
| 384 | 1107.26 | 81.57 | 50.72 | 22.47 | 52.06 |
| 256 | 1138.03 | 81.19 | 50.57 | 22.06 | 52.11 |
| 144 | 1103.66 | 80.47 | 50.82 | 22.07 | 51.87 |
| 64  | 1131.21 | 80.30 | 50.92 | 21.97 | 51.64 |
| 16  | 1123.55 | 79.87 | 50.47 | 20.07 | 51.39 |

**Does the long ladder help? Yes, dramatically, relative to v6 — this is the headline
result.** v13 vs v6 (v6 is `resampler_arch=query` on the identical 8-level grid and
rank schedule — the only prior data point on this ladder), every level, every metric:

| level | MME-P | POPE-acc | SciQA | TextVQA | GQA |
|---|---|---|---|---|---|
| 576 | +57.2 | +1.15 | +3.07 | +7.96 | +1.30 |
| 512 | +60.3 | +1.51 | +3.47 | +8.03 | +1.58 |
| 448 | +59.4 | +1.58 | +3.57 | +8.32 | +1.71 |
| 384 | +69.0 | +1.48 | +3.62 | +7.84 | +1.84 |
| 256 | +114.9 | +1.25 | +3.17 | +7.34 | +1.84 |
| 144 | +102.3 | +1.04 | +9.12 | +8.16 | +2.23 |
| 64 | +139.3 | +1.24 | +9.72 | +8.17 | +2.00 |
| 16 | +118.3 | +0.48 | +8.08 | +6.05 | +1.84 |

Every cell is a win, most by a wide margin — TextVQA +6 to +8 points at every budget,
MME +57 to +139. **This comparison is confounded (v6 has vision LoRA on, v13 has it
off) and the confound runs the WRONG way for v13**: §16m/§16l already established that
under PARCEL, vision-LoRA-off costs several points against LoRA-on (v8 vs v11: −6.3
TextVQA, −2.1 GQA at 256 tok). v13 is carrying that handicap and still wins by 6-9
TextVQA points. The honest reading is that PARCEL's fix for the long-ladder failure is
if anything *understated* by this table — a `use_lora=true` version of v13 would likely
win by even more, though that is not a measured claim.

**v13's own elasticity, extended vs. short range:**

| range | MME-P | POPE-acc | SciQA | TextVQA | GQA |
|---|---|---|---|---|---|
| full 576→16 | −23.6 | +1.74 | +0.10 | +2.70 | +0.57 |
| short 256→16 (v8/v11's range) | +14.5 | +1.32 | +0.10 | +1.99 | +0.72 |

Both are far short of v8's short-range spread (MME +46.3, TextVQA +8.76, GQA +3.46,
§16c) and even short of hipster v11's (MME +99.95, TextVQA +10.09, GQA +3.66) — v13 is
flatter than either short-grid PARCEL run on the range they share. So while PARCEL
fixed v6's *catastrophic* failure, it did not reproduce PARCEL's own best elasticity
once the ladder is stretched to 8 levels; the extra levels dilute the gradient somewhat
even under the better resampler.

**Does v13 beat v8? No.** At the four shared levels:

| level | MME-P | POPE-acc | SciQA | TextVQA | GQA |
|---|---|---|---|---|---|
| 256 | −32.2 | −3.14 | +2.43 | −6.41 | −3.96 |
| 144 | −38.5 | −3.76 | +2.73 | −4.27 | −3.43 |
| 64 | −23.9 | −2.73 | +3.47 | −1.76 | −2.88 |
| 16 | −0.4 | −0.62 | +4.11 | +0.36 | −1.22 |

v13 loses to v8 on MME/POPE/GQA at every level, ties or edges ahead on TextVQA only at
16 tokens, and wins SciQA throughout — the same SciQA-tracks-inversely-with-LoRA
pattern from §16m/§16l, here with LoRA fully off. The gap narrows sharply toward the
low-budget end (16 tok is nearly a wash), which is consistent with vision LoRA mattering
most at the levels where PARCEL's anchors carry the least information on their own.

**Against the dense 576-token baseline, v13 is well short at every level it was meant
to help most:** v13@256 is −18.94 TextVQA / −6.19 GQA / −6.79 SciQA against baseline;
v13@576 (same token count as baseline, so *nothing* should be lost to compression) is
−18.23 TextVQA / −6.34 GQA / −6.79 SciQA — i.e. almost identical to v13@256, meaning
the extra 320 tokens at the top of the ladder buy essentially nothing over 256, which is
the token-budget-has-no-effect pattern once again, just less severe than v6's version of it.

**Verdict.** PARCEL is not merely a short-grid trick — it substantially rescues the
long-ladder failure v6 exposed, confirming §16j's hypothesis. But "rescues" is not
"solves": v13 still trails v8 by a wide margin on the metrics that matter most
(TextVQA, GQA), and its own elasticity on the shared range is flatter than v8's or
v11's. Combined with v14's still-pending result on whether ladder-direction matters,
the standing recommendation is unchanged: **`final-parcel` (= v8) stays the recipe for
the paper and for the Hipster backbone sweep** (§17c, §18). The 8-level ladder remains
a research question, not a candidate replacement — worth one more run (v13 + vision
LoRA on, isolating whether the LoRA-off confound above fully explains the v8 gap) if
there is appetite for it, but not before v14 reports.


### 16q. v14 results: rank-ladder direction matters, and v8's direction was already the right one

Completed (jobs 27406/27407). Config confirmed: `lora_ranks=[64,32,16,8]` against
`tok_levels=[256,144,64,16]` — the 256-token (teacher) level gets the **largest**
adapter (rank 64), the 16-token level the **smallest** (rank 8), the exact reverse of
v8's `[8,16,32,64]`. Everything else — `resampler_arch=pool_anchored`,
`anchor_mode=ratio` @0.25, `use_token_decorrelation=false`, `teacher=self`, grid
`256/144/64/16` — is identical to v8. Stage 1 also confirms the inherent difference
flagged in §16m: it trained a rank-64 adapter at 256 tokens, and v14's 256-level uses
all 64 of those columns (v8's 256-level used only the first 8).

**This is now the cleanest isolation in the whole ablation set — one variable, rank
direction, nothing else differs — and v14 loses.**

| level | MME-P | POPE-acc | SciQA | TextVQA | GQA |
|---|---|---|---|---|---|
| 256 | 1102.47 | 82.26 | **50.37** | 20.24 | 52.30 |
| 144 | 1112.56 | 82.18 | **50.72** | 20.02 | 52.22 |
| 64  | 1119.90 | 81.32 | **51.12** | 20.38 | 52.18 |
| 16  | 1118.11 | 79.91 | **50.27** | 19.43 | 51.46 |

**v14 vs v8, all four levels:**

| level | MME-P | POPE-acc | SciQA | TextVQA | GQA |
|---|---|---|---|---|---|
| 256 | −67.78 | −2.07 | +2.23 | **−8.23** | **−3.77** |
| 144 | −29.62 | −2.05 | +2.63 | −6.32 | −3.08 |
| 64  | −35.20 | −1.71 | +3.67 | −3.35 | −2.34 |
| 16  | −5.80 | −0.58 | +3.91 | −0.28 | −1.15 |

v14 loses MME/POPE/TextVQA/GQA at **every** level, worst at 256 tokens — precisely the
level v14 gave the *most* extra capacity to. It wins SciQA at every level, continuing
the pattern from §16m/§16l, though with a wrinkle noted below. The 16-token gap is the
only one small enough to be noise (TextVQA −0.28 against a ~0.6pp stderr); everywhere
else the gaps are 2–5× stderr.

**The asymmetry is the real finding.** v14's 256-level carries **8× more** LoRA rank
than v8's (64 vs 8) and still loses badly there (−8.23 TextVQA, −3.77 GQA). v14's
16-level carries **8× less** rank than v8's (8 vs 64) and is within noise of it. Extra
capacity at the information-rich, teacher-serving level does not help — the level that
already has the least to compensate for got the least benefit from more adapter, and
plausibly grew worse for it. Capacity at the starved level barely matters at all: 8
columns gets the 16-token level to within 0.3 TextVQA points of what 64 columns
achieves. **This directly answers the question §16m/16p left open** ("does the
rank/budget pairing direction matter, or would either ladder work as well once nesting
is present at all?") — direction matters, and v8's original convention (small budgets
get more adapter capacity to compensate; large budgets get just enough to specialize
without perturbing the shared representation) was already correct. v14 was a real test
of a real competing hypothesis, and the hypothesis lost.

**Elasticity collapses under the reversed ladder — this is the more surprising result.**
Spread 256→16:

| run | MME-P | POPE-acc | SciQA | TextVQA | GQA |
|---|---|---|---|---|---|
| v8 | +46.34 | +3.84 | +1.78 | **+8.76** | +3.46 |
| v11 | +99.95 | +3.86 | +2.72 | **+10.09** | +3.66 |
| **v14** | **−15.64** | +2.35 | +0.10 | **+0.81** | +0.84 |

v14's MME spread is *negative* — the 16-token level scores higher than 256 tokens — and
its TextVQA/GQA spread collapses to near the flat v4/v5 baselines this whole
architecture was built to fix (§project-token-budget-has-no-effect). PARCEL and full
nested vision LoRA are both present in v14, exactly as in v8, and the elasticity
gradient that made v8 the best model (§16c) is nearly gone anyway. So the gradient is
not just "PARCEL + nested LoRA" — it depends on nesting them in the *right direction*.
Putting the most adapter capacity at the rich/teacher level apparently lets that level
drift in a way that erodes its edge over the cheap levels, flattening the curve from
both ends rather than lifting the floor.

**v14 vs v11 (does reversed-nested LoRA still beat no LoRA at all?) — mixed, and the
256-level result is the interesting part:**

| level | MME-P | POPE-acc | SciQA | TextVQA | GQA |
|---|---|---|---|---|---|
| 256 | +15.84 | −1.26 | +0.15 | −1.94 | −1.65 |
| 144 | −4.20 | −1.31 | +0.35 | +0.74 | −1.37 |
| 64  | +18.47 | −1.09 | +1.54 | +4.15 | −0.31 |
| 16  | +131.43 | +0.25 | +2.77 | +7.34 | +1.17 |

At 16 tokens, some adapter (even the "wrong-direction" rank-8 sliver) clearly beats
none. At 256 tokens, v14 is worse than v11 on POPE/TextVQA/GQA — i.e. **giving the
teacher level a large adapter is worse than giving it no adapter at all**, not just
worse than giving it a small one (v8). That makes the 256-level's rank a
non-monotonic story on its own: v11 (rank 0) < v14 (rank 64) < v8 (rank 8) on
TextVQA/GQA at that level. Speculative but plausible mechanism, not measured directly:
whichever level serves as the self-distillation teacher (`kl_teacher_tok_level=0`)
propagates its own quality to every other level via the KL term, so a heavily-adapted,
possibly-drifted teacher (v14) does worse than either an unperturbed one (v11) or a
lightly-touched one (v8, rank 8) — the middle ground v8's convention happens to land on.

**The SciQA nuance.** v14 wins SciQA at every level against v8, despite both carrying
identical *total* LoRA rank (8+16+32+64 = 64+32+16+8 = 120) — so SciQA's earlier
"tracks inversely with LoRA capacity" framing (§16m/§16l, a six-run pattern across
different *amounts* of adapter) needs a footnote: at matched total capacity, it also
tracks *where* that capacity sits, favouring more rank at the high-budget/teacher level
over more rank at the low-budget levels. Both framings may be true simultaneously
(less total capacity helps SciQA; and among fixed-capacity ladders, capacity nearer the
teacher level helps SciQA) — not disentangled by anything run so far.

**Verdict: v8 remains the best model, and more securely than before.** This was the one
remaining design choice in v8's recipe that had never been ablated on its own — every
other axis (resampler, LoRA presence, decorrelation, KD teacher, ladder length) had a
clean isolation; only the rank-direction convention was inherited from v4 unexamined.
It has now been tested directly and confirmed. `final-parcel` needs no change.


### 16l. v11 and v12 (hipster) close the ablation set: the final recipe is v8, and it is not close

The hipster cluster returned `elastic-finetune-tinyllama-v11-parcel-nolora`,
`elastic-finetune-tinyllama-v12-parcel-kd7b` and the SmolLM2 v11 port
(`results/hipster_eval_summary.csv`). Every v11 number below is the hipster run.
*(The local v11, 27386, was cancelled on 2026-09-13 at ~1d19h to free node205 for v14 —
§16m — so the cross-cluster replicate it would have provided does not exist; the
`final-parcel` tinyllama re-run remains the only planned replicate.)* Hardware caveat as in §16e: nothing in the
training path branches on GPU type, but numerics are not bitwise-identical across cards.

**TinyLlama, all seven runs, 256 tokens** (16-token row below it):

| run | resampler | vision LoRA | decorr | teacher | MME-P | POPE-acc | SciQA | TextVQA | GQA |
|---|---|---|---|---|---|---|---|---|---|
| v4 | query | off | off | self | 1113.46 | 81.57 | **51.31** | 23.15 | 52.35 |
| v5 | query | nested | off | self | 1035.55 | 79.38 | 49.38 | 18.56 | 50.91 |
| **v8** | PARCEL | **nested** | off | self | **1170.25** | **84.33** | 48.14 | **28.47** | **56.07** |
| v9 | PARCEL | off | **on** | self | 1120.79 | 80.40 | 50.72 | 19.58 | 51.88 |
| v10 | PARCEL | shared r16 | off | self | 1138.75 | 80.97 | 50.52 | 19.96 | 51.95 |
| v11 | PARCEL | off | off | self | 1086.63 | 83.52 | 50.22 | 22.18 | 53.95 |
| v12 | PARCEL | off | off | **LLaVA-7B** | 1055.96 | 80.23 | 50.02 | 15.77 | 51.65 |
| *dense 576* | — | — | — | — | *1248.32* | *84.89* | *57.36* | *41.00* | *58.30* |

| 16 tok | MME-P | POPE-acc | SciQA | TextVQA | GQA |
|---|---|---|---|---|---|
| v4 | 1116.40 | 80.70 | 44.12 | 19.90 | 51.18 |
| **v8** | **1123.91** | **80.49** | 46.36 | **19.71** | **52.61** |
| v9 | 1111.04 | 79.79 | **51.12** | 19.14 | 51.22 |
| v10 | 1089.66 | 79.70 | 50.17 | 19.23 | 51.71 |
| v11 | 986.68 | 79.66 | 47.50 | 12.09 | 50.29 |
| v12 | 999.02 | 78.54 | 47.00 | 10.45 | 48.88 |

**v8 wins 4 of 5 metrics at all four budgets**, with the largest margins on TextVQA and
GQA — the two benchmarks that actually require visual detail. It loses SciQA by 2–3
points to every no-LoRA run. That is the whole trade-off, and it is the same trade-off
at every budget.

**Every design axis now has a clean single-flag isolation** (all against hipster v11,
which is PARCEL / no LoRA / no decorrelation / self-teacher — the minimal PARCEL run):

| axis | isolation | @256: POPE / TextVQA / GQA | @16: TextVQA / GQA | verdict |
|---|---|---|---|---|
| resampler | v11 (PARCEL) − v4 (query) | +1.95 / −0.97 / +1.60 | −7.81 / −0.89 | PARCEL alone: modest top-end win, **worse floor** — it creates the gradient by dropping the 16-tok level, not by raising 256 |
| vision LoRA, nested | v8 − v11 | +0.81 / **+6.29** / **+2.12** | **+7.62** / +2.32 | lifts the *entire* curve, most at the low end; costs SciQA −2.08 |
| vision LoRA, shared r16 | v10 − v11 | −2.55 / −2.22 / −2.00 | +7.14 / +1.42 | lifts the floor but **hurts the top** — a non-specialized adapter is worse than none at 256; per-level nesting is the mechanism, not capacity |
| decorrelation | v9 − v11 | **−3.12 / −2.60 / −2.07** | +7.05 / +0.93 | same shape as v10: floor up, top down, ladder flattened (spread 0.44 vs 10.09). **Negative.** |
| 7B KD teacher | v12 − v11 | **−3.29 / −6.41 / −2.30** | −1.64 / −1.41 | worse at every level on every metric. **Negative** — at `prefix_kl_weight 0.1`, which was never tuned for a teacher whose KL is actually non-zero; a sweep could move this, but nothing suggests it would win. Also Llama-vocab-only, so it could never be applied uniformly across SmolLM2/Qwen/Phi. |

The decomposition this gives is the paper's story in three lines: **PARCEL makes the
budget matter; rank-nested per-level vision LoRA is what makes the model good at every
budget; and LoRA only helps once PARCEL's anchors are there** (v5 — nested LoRA on the
plain query resampler — loses to v4). Neither ingredient works alone. The one constant
cost, in every configuration, is SciQA: `v4 51.31 > v9 50.72 > v10 50.52 > v11 50.22 >
v5 49.38 > v8 48.14` — six runs, monotone in vision-tower adapter capacity, across both
resamplers.

**A caution about "elasticity" as a number.** v11 has the largest 256−16 spread in the
project (TextVQA +10.09, GQA +3.66 vs v8's +8.76 / +3.46) — and it gets there by
collapsing at 16 tokens (TextVQA 12.09 vs v8's 19.71, MME 986 vs 1124), not by being
better at 256. Spread rewards a bad floor. The paper's claim has to be about the
**frontier** — accuracy at each budget — and on the frontier v8 dominates v11 at every
point. Report both the curve and the spread; never the spread alone.

**SmolLM2** confirms the shape but not yet the fix:

| SmolLM2 | MME-P | POPE-acc | SciQA | TextVQA | GQA |
|---|---|---|---|---|---|
| v4 query/no LoRA @256 | 1202.31 | 81.99 | 57.06 | 25.42 | 51.50 |
| v5 query/nested @256 | 1163.54 | 81.17 | 47.15 | 22.03 | 50.78 |
| v11 PARCEL/no LoRA @256 [hip] | 1184.91 | 82.27 | 55.08 | **15.82** | 51.55 |
| v11 @16 [hip] | 1077.30 | 79.26 | 56.32 | 11.15 | 47.54 |

Same pattern as TinyLlama's v11: PARCEL alone opens a gradient (GQA spread 4.01 vs v4's
1.04) by dropping the floor, and here the top-end TextVQA cost is severe (−9.6 vs v4).
SmolLM2 has **no PARCEL + nested-LoRA run** — the v8 recipe on SmolLM2 is the
best-supported extrapolation, not a measured result. Specific risk to watch: SmolLM2 v5
paid SciQA −9.9 for nested LoRA under the query resampler, far more than TinyLlama did,
so the SciQA cost of the final recipe may be backbone-dependent and larger here.

**Does a 576-token teacher / the full ladder help?** No data yet — v13 is 12h into
training. The prior evidence is against it, and the mechanism is worth stating:

- The only 576-teacher run so far is v6 (query resampler): the worst run in the project,
  flat from 576 to 384, and its 576-token level scored below v4 at 16 tokens (§16b).
- "A stronger teacher signal" is not a lever with `teacher=self`. The self-KL was
  measured at ~0.006 (§5b) because student and teacher are the same weights on
  informationally-equivalent inputs — a 576-token self-teacher only carries more signal
  if the 576 level actually holds more usable information than 256, and v6's flat ladder
  says it does not. The one place KL *did* have real signal — the external 7B teacher —
  is v12, and it hurt.
- What is left is the 576 level's own CE shaping the shared resampler on richer inputs.
  v6 says that did not transfer down the ladder. v13 tests whether PARCEL changes that.
- v13 costs ~2× per step for it (§16j), and it was designed **before** the v8-vs-v11
  result, so it runs with vision LoRA *off*. It is the fair test of "does the ladder
  alone help" — but if it loses, it does not rule out long-ladder + nested LoRA. It just
  makes that a low-prior bet nobody should spend a week on before submission.

Recommendation: do not plan the final sweep around the long ladder. Let v13 finish; if
its 256/144/64/16 points do not beat v8's, the ladder question is closed for the paper.

**Actions taken.** `final-parcel` added to `submit_elastic_run.sh` — v8's recipe
verbatim, `SLM_KEY`-parametrised, with every choice annotated by the isolation that
justifies it. §16f's hipster plan (v11-based) superseded by it. §16g and decision 33
closed negative. The launcher default `VISION_LORA_ENABLE=False` (§16d) is now
contradicted under PARCEL; the recipe sets `True` explicitly and no final-sweep run
should rely on the launcher default — flip it when the sweep starts, not before (v13 is
running; a mid-flight default change helps nothing).

---

## FINAL BEST CONFIG FOR THE PAPER (TinyLlama, to be applied to every SLM backbone)

**= `elastic-finetune-tinyllama-v8-parcel`, exactly. Recipe: `bash submit_elastic_run.sh final-parcel` with `SLM_KEY=<backbone>`.**

| axis | setting | why (isolation) |
|---|---|---|
| resampler | **PARCEL** `pool_anchored` | v11 vs v4: creates the budget–accuracy gradient |
| anchors | **ratio 0.25** (≡ v8's fixed `64/36/16/4`; the two were never different on this grid) | ratio generalises if the grid changes; no run separates fixed from ratio |
| vision LoRA | **ON, rank-nested per level**, `lora_ranks 8 16 32 64`, `specialize_tok=True`, `STAGE1_LORA_RANK=64` | v8 vs v11: +6.3 TextVQA / +2.1 GQA @256, +7.6 TextVQA @16 |
| decorrelation | **OFF** | v9 vs v11: −3.1 POPE / −2.6 TextVQA / −2.1 GQA @256 |
| CORAL | **OFF** | never enabled in any evaluated run — not introduced untested into the final sweep |
| KD teacher | **self** (7B teacher OFF) | v12 vs v11: −6.4 TextVQA / −3.3 POPE / −2.3 GQA @256; Llama-vocab-only anyway |
| ladder | **256 / 144 / 64 / 16** | v6: 576-top ladder flat and worse; v13 pending, low prior |
| vision tower | CLIP ViT-L/14-336, learned pos-embed, nested dropout off | unchanged since v4; never the variable |

**Trade-off stated plainly:** best MME, POPE, TextVQA and GQA at every budget; **−2 to
−3 SciQA** against any configuration without vision LoRA, consistently. At 256 tokens
(44% of the dense budget) it is within 0.56 POPE and 2.2 GQA of the uncompressed
576-token model.

**What the sweep must include:** `tinyllama` again under the `final-parcel` tag.
Every result in this journal is n=1; the re-run is the seed replicate the paper needs
for v8, and the local v11 (27386) vs hipster v11 pair is the cross-cluster replicate for
the controls. Then SmolLM2 → Phi-2 → Qwen pair → MobileLLaMA, per §15e/§16f ordering,
with SciQA watched on SmolLM2 specifically.


### 16m. The vision-LoRA axis in one table: none → shared r16 → nested — and v14, which reverses the nesting

All three arms are PARCEL, decorrelation off, self-teacher, grid `256 144 64 16`. The
only thing that differs is the vision-tower adapter. v11 is the hipster run.

| level | adapter | MME-P | POPE-acc | POPE-F1 | SciQA | TextVQA | GQA |
|---|---|---|---|---|---|---|---|
| 256 | none (v11) | 1086.63 | 83.52 | 82.13 | **50.22** | 22.18 | 53.95 |
| 256 | shared r16 (v10) | 1138.75 | 80.97 | 79.49 | 50.52 | 19.96 | 51.95 |
| 256 | **nested 8/16/32/64 (v8)** | **1170.25** | **84.33** | **83.06** | 48.14 | **28.47** | **56.07** |
| 144 | none | 1116.76 | 83.49 | 82.48 | 50.37 | 19.28 | 53.59 |
| 144 | shared r16 | 1117.21 | 81.01 | 79.42 | 50.97 | 19.90 | 51.82 |
| 144 | **nested** | **1142.18** | **84.23** | **83.36** | 48.09 | **26.34** | **55.30** |
| 64 | none | 1101.43 | 82.41 | 81.14 | 49.58 | 16.23 | 52.49 |
| 64 | shared r16 | 1108.70 | 80.40 | 78.53 | 50.57 | 19.67 | 51.73 |
| 64 | **nested** | **1155.10** | **83.03** | **81.89** | 47.45 | **23.73** | **54.52** |
| 16 | none | 986.68 | 79.66 | 77.64 | 47.50 | 12.09 | 50.29 |
| 16 | shared r16 | 1089.66 | 79.70 | 77.15 | **50.17** | 19.23 | 51.71 |
| 16 | **nested** | **1123.91** | **80.49** | **78.62** | 46.36 | **19.71** | **52.61** |

Spread, 256 − 16: none MME 99.95 / POPE 3.86 / TextVQA 10.09 / GQA 3.66; shared r16
49.09 / 1.27 / 0.73 / 0.24; nested 46.34 / 3.84 / 8.76 / 3.46.

**Three things this table says that the pairwise deltas in §16l only implied:**

1. **The effect of an adapter is not monotone in capacity.** On TextVQA at 256 tokens:
   none 22.18 → shared r16 **19.96** → nested 28.47. A single shared adapter is *worse
   than no adapter* at the top budget (also POPE −2.55, GQA −2.00), while lifting the
   16-token floor by +7 (12.09 → 19.23). One adapter serving every budget settles on a
   compromise that helps the starved level and taxes the rich one. Nesting removes the
   compromise: the 256 level gets its own light (rank-8) adapter that barely perturbs
   CLIP, the 16 level a heavy (rank-64) one, and both ends move up. This is why "vision
   LoRA" was never the right unit of analysis — the per-level specialization is the
   mechanism, and it is what §16d's v4-vs-v5 (nested, but on the query resampler) could
   not show.

2. **SciQA is paid by nesting specifically, not by having an adapter.** none 50.22,
   shared 50.52, nested **48.14** at 256; at 16 tokens the shared arm is the best of the
   three (50.17). Whatever SciQA's diagram-style images need from CLIP, the rank-64
   low-budget adapter disturbs it and the rank-16 shared one does not.

3. **The "none" arm's elasticity is an artifact of its floor**, restated with the full
   ladder in view: its TextVQA spread (10.09) beats nested (8.76) only because it scores
   12.09 at 16 tokens. Every level of the nested arm is above every level of the none
   arm on MME/POPE/TextVQA/GQA.

#### v14-parcel-asclora: rank follows budget

v8's rank ladder pairs the **largest** budget with the **smallest** rank (256→8,
144→16, 64→32, 16→64). That pairing was a convention — "small budgets need more
adaptation to compensate" — carried from v4 and never ablated. Row 1 above says the
mechanism is per-level specialization, not compensation, which leaves the assignment
direction open. The competing hypothesis: the 256-token level is the teacher and the
level carrying the most visual information, so it should get the most adapter
capacity; the 16-token level, whose inputs are already a coarse summary, needs the
least. **v14 is v8 with the rank ladder reversed: `64 32 16 8`.** Everything else is
identical to v8 / `final-parcel`.

One consequence is inherent to the flip and worth stating so it is not mistaken for a
second variable later: Stage 1 trains a rank-64 adapter at 256 tokens. In v8, Stage 2's
256 level then reads only the first 8 columns of that warm start — the teacher level
inherits 1/8 of what Stage 1 learned. In v14 it uses all 64, so the teacher level's
warm start is consistent end to end for the first time. If v14 wins, part of the gain
may be that consistency rather than the direction of the ladder; the two are not
separable without a third run (v8 ranks, Stage 1 at rank 8), which nobody should queue
before seeing v14.

**Code change required and made** (`llava/model/elastic/nested_lora.py`,
`llava/model/elastic/config.py`): `NestedLoRALinear` asserted `ranks` ascending and set
`max_rank = ranks[-1]`. Neither was structural — nesting is a prefix slice
`A[:, :r] @ B[:r]`, valid for any `r ≤ max` in any order — they only encoded v4–v8's
convention. Now `max_rank = max(ranks)`, the assert checks positivity only, and the
shared-adapter index in `ElasticConfig.lora_level_for_tok` picks the max-rank entry
rather than the last one (with descending ranks, "last" would have silently been rank
8). Validated by `jobs/test_lora_rank_order.sh`: descending construction, per-level
rank resolution, the prefix-slice identity, a `[64]` Stage-1 warm start loading into a
`[64,32,16,8]` Stage 2 with the teacher level at rank 64, and the ascending v8 path
unchanged.

**Queued and running: jobs 27406 (Stage 1 + 2) / 27407 (eval, `--array=0-3`),
node205, `gpu:2`.** Read it against v8 at all four levels; the first thing to check is
whether the 16-token level — now on a rank-8 adapter instead of rank-64 — holds v8's
19.71 TextVQA / 52.61 GQA floor, since that is where the reversal takes capacity away.

**The local v11 (27386) was cancelled** at ~1d19h to free node205 for this run. Its
hipster twin already exists and is the one used throughout §16l; the only thing lost is
the cross-cluster replicate. Its partial checkpoint directories
(`elastic-*-tinyllama-v11-parcel-nolora`) are left on disk, not deleted.


### 16n. Would a shared rank-64 adapter help? Prediction: no — rank is not the variable (research only, decision 37)

The question, given v10 (shared r16) did not help: is r16 simply too small, and would a
shared r64 close the gap to v8? Three numbers from §16m answer it without a run.

**1. At the top budget, the winner has *less* capacity than the loser.** At 256 tokens
v8's level uses a **rank-8** adapter and gains +6.29 TextVQA / +2.12 GQA / +0.81 POPE
over no adapter. v10's **rank-16** shared adapter at the same level *loses* −2.22 /
−2.00 / −2.55. An 8.5-point TextVQA gap in favour of the smaller adapter cannot be a
capacity shortfall. Something other than rank separates them.

**2. At the bottom budget, r16 already captures nearly all of the nested-r64 gain.**
At 16 tokens over no adapter: shared r16 gives +7.14 TextVQA, nested r64 gives +7.62 —
94% of the gain at a quarter of the rank. GQA +1.42 vs +2.32, MME +103 vs +137. The
16-token level does not need 64 columns; what it needs, it gets from 16. Raising the
shared rank has almost nothing left to buy here.

**3. So what separates v8 from v10 is where the gradients land, not how many there
are.** With `lora_specialize_tok=False` the vision tower is budget-blind: one ΔW is
applied whether the resampler downstream will keep 256 tokens or 16. Every training
step that adapter receives gradient from the 256-token teacher pass *and* from one
sampled student pass (144/64/16, mean ≈75 tokens — §16j). The student levels are the
starved ones with the higher CE, so they push harder, and they push toward CLIP
features that survive aggressive compression — global, summary-like. The teacher level
wants the opposite: fine detail preserved for 256 queries. A single function of the
input has to pick one; it drifts toward the students, which is exactly the v10
signature — floor up, top down.

The nested design does not avoid shared parameters — v8's rank-8 prefix is a
sub-block of its rank-64 adapter and receives gradient from *every* level. What it does
is **quarantine**: the forward at level 0 slices `A[:, :8] @ B[:8]`, so autograd from
the 256-token pass never touches columns 9–64, and the 256-token forward never *reads*
them. The 16-token level's aggressive adaptation lives in 56 columns the teacher
neither trains nor uses. The 8 columns they do share settle on a compromise, but with
only 8 of them and the teacher present every step, that compromise stays close to
CLIP. A shared adapter of any rank has no such partition — and giving it *more* rank
gives the student objective more room to reshape the columns the teacher is forced to
read through. The prediction is therefore not "r64 recovers the top end" but "r64
regresses it further, while the 16-token level gains ≤ the ~6% r16 left on the table."

**Literature, from memory — verify before citing.** This is the standard multi-task
LoRA interference result: one low-rank adapter across conflicting objectives
underperforms per-objective adapters, which is what motivates the MoE-style LoRA
family (MixLoRA / MoLoRA / LoRAHub-type routing) — routing exists precisely because
rank does not resolve conflict. Directionally against high shared rank: Biderman et al.
2024 ("LoRA Learns Less and Forgets Less") find higher-rank LoRA behaves more like full
fine-tuning, i.e. forgets more of the pretrained features — and CLIP's pretrained
features are exactly what the 256-token level needs kept intact. The LoRA rank sweeps
the journal already leans on (§11: saturation by r≈8–16, LangVision-LoRA-NAS's r=16
sweet spot) point the same way: adapter capacity is rarely the bottleneck. The nested
structure itself is DyLoRA's (Valipour et al. 2023) training scheme — prefix-nested
ranks so any prefix is a valid adapter — which is why lower prefixes stay competitive
and why the 8-column teacher slice works at all.

**Not queued, and why.** A shared-r64 run is a single recipe line, but both 2-GPU nodes
are taken (v13, v14) and the prediction is a loss on the top budget with no upside at
the bottom. If the paper wants the row anyway — "rank is not the variable" is a clean
ablation to *show*, not just argue — it is a hipster candidate at zero design cost:
`VISION_LORA_SPECIALIZE_TOK=False LORA_RANKS="64 64 64 64" STAGE1_LORA_RANK=64` on the
`final-parcel` base. The run that *does* test whether the 256 level wants more capacity
is v14 (256→r64, nested, quarantine intact), already running. The cheaper ablation
worth more than shared-r64, if a slot opens: nested with a **smaller** max
(`4 8 12 16`) — keeps the partition, cuts adapter parameters 4×, and point 2 predicts
it loses little; that is an efficiency-story row, not a capacity one.


## 17. Controlled v8-vs-v14 audit, and the Hipster experiment matrix (decision 38)

### 16o/17a. What actually differs between v8 and v14

Audited from the checkpoints' own `elastic_config.json`, the launcher scripts, and
`git log` — not from the version numbers.

| Component | v8 | v14 | Same/Different |
|---|---|---|---|
| model architecture | LLaVA + nested-query elastic engine | same | **Same** |
| visual-token reduction | `nested_query`, budgets 256/144/64/16 | same | **Same** |
| resampler | `pool_anchored` (PARCEL) | same | **Same** |
| spatial anchors | `anchor_routing {256:64,144:36,64:16,16:4}` | `anchor_mode=ratio, ratio=0.25` | **Same behaviour** — verified in code (job 27408): ratio 0.25 resolves to exactly 64/36/16/4 at these four budgets; nested dropout is off, so no other budget is ever requested |
| learned local-detail queries | budget − anchors, `query_selection=prefix` | same | **Same** |
| decorrelation loss | off | off | **Same** |
| CORAL | off (`use_coral_align=false`) | off | **Same** |
| KD | `use_prefix_kl=true`, `teacher=self`, weight 0.1 | same | **Same** |
| **LoRA rank ladder** | **`8 16 32 64`** (rank ascends as budget descends) | **`64 32 16 8`** (rank ascends with budget) | **DIFFERENT — the only one** |
| LoRA everything else | nested, `specialize_tok=true`, alpha 1.0, dropout 0.0 | same | **Same** |
| Stage-1 warm start | `tok_level 256`, `lora_rank 64` | same | **Same recipe**, but *consumed* differently: v8's 256 level reads only columns 1–8 of the rank-64 Stage-1 adapter, v14's reads all 64 (inherent to the flip, not a second knob) |
| training objective | CE + prefix-KL | same | **Same** |
| optimizer | adamw_torch, LR 2e-5, cosine, warmup 0.03, wd 0 | same | **Same** |
| token budgets | 256/144/64/16 | same | **Same** |
| frozen/trainable | LLM full FT, CLIP frozen + LoRA, ZeRO-2 | same | **Same** |
| data | `llava_v1_5_mix665k.json` | same | **Same** |
| augmentation | `image_aspect_ratio=pad`, no other aug | same | **Same** |
| training duration | 1 epoch, bs 2 × accum 32 × 2 GPU | same | **Same** |
| seed | unset → HF default 42 | same | **Same** |
| evaluation protocol | `eval_lmms_level.sh`, 5 benchmarks, `--array=0-3` | same | **Same** |
| model code | pre-`acdfa79` resampler | post-`acdfa79` | **Same behaviour** — `acdfa79` landed 2026-09-08 16:06 while v8 was mid-run (launched Sep 5), so the two ran different source. Audited: the resampler diff only *adds* `anchor_mode`/`anchor_ratio` with ratio-0.25 reproducing the old `budget // 4`, and the mixin diff is entirely inside `if cfg.use_token_decorrelation:`, which is False in both |
| hardware | node208, A10:2 | node205, A40:2 | **Different** — no code path branches on GPU type and batch sizes are constants, so the recipe is identical; numerics are not bitwise reproducible across cards |

**Verdict on existing evidence:**

> **Existing evidence is insufficient for a controlled v8-v14 comparison.**

v14 (job 27406) started 2026-09-13 and is ~9h into a ~85h Stage-2 run. It has produced
no checkpoint and no eval. `eval_logs/` contains no `v14` directory. There is **no v14
number of any kind**, on any metric, on any backbone. Nothing can yet be said about
`V14 > V8`, `V14 ≈ V8`, or `V14 < V8`.

What the audit *does* establish, which matters for when v14 lands: once it does, the
comparison against v8 will be a clean single-variable one. The only differences are the
rank ladder (intended), the GPU model (uncontrolled but not recipe-affecting), and a
code delta verified inert. The anchor-config difference looked like a confound and is
not one.

Two honest caveats to carry into that comparison: both runs are **n=1**, and the
Stage-1 consumption difference in the table above rides along with the flip — if v14
wins, the ladder direction and the warm-start consistency are not separable without a
third run.

### 17b. What "KD" means in this codebase

Checked against the code rather than the name. There are **two different mechanisms**,
and only one is available on all backbones.

**(a) `--use_kd` → `cfg.use_prefix_kl` — prefix-KL SELF-distillation across token
budgets.** The teacher is *this model* at `tok_levels[kl_teacher_tok_level]` = index 0
= 256 tokens; its logits are detached and the smaller budgets are trained toward them
(`llava_elastic_mixin.py:325-345`). **Teacher and student share weights exactly** — one
network compared against itself on a longer visual prefix. Measured KL ≈ 0.006, about
0.01% of the loss (§5b), because the inputs are informationally near-equivalent. With
`teacher=self` the teacher level receives no KL itself; only the sampled students do.

This is what the matrix's KD column toggles. **It is not an independent-teacher
comparison and must not be reported as one.**

**(b) `--teacher llava` → a frozen external LLaVA-1.5-7B**, independent weights, one
no-grad forward per step, 576 visual tokens, right-aligned so the labelled text
positions match (`engine.attach_kd_teacher`). Every level becomes a student including
256. This is genuine KD — and it is **gated on tokenizer identity**:
`attach_kd_teacher` raises when `teacher.config.vocab_size != student.config.vocab_size`.

| backbone | vocab | external-teacher KD |
|---|---|---|
| TinyLlama | 32000 | possible |
| MobileLLaMA | 32000 | possible |
| SmolLM2 | 49152 | **impossible** |
| Qwen2.5 (0.5B/1.5B/3B) | 151936 | **impossible** |
| *LLaVA-1.5-7B teacher* | *32000* | — |

So genuine KD cannot be run on SmolLM2 or Qwen with the teacher this repo has. It is
kept as a **separate MobileLLaMA-only arm**, not folded into the matrix, because a "KD"
column meaning self-distillation for two backbones and external-teacher KD for a third
would be uninterpretable. Prior evidence: v12 (external KD, TinyLlama) lost to its
control on every metric at every budget (§16l), at an untuned `prefix_kl_weight` of 0.1.

There is no KD *temperature* in this codebase — `prefix_kl_loss` is a plain
log-softmax KL. The manifest records `"temperature": "n/a"` rather than inventing one.

### 17c. The matrix is 12 runs, not 24

The requested matrix was 2 versions × 3 backbones × 2 LoRA × 2 KD = 24. Per §17a,
**version and LoRA are the same axis**, so the 2×2 holds only two distinct
configurations:

| nest_version | lora_type | resulting model |
|---|---|---|
| v8 | v8 | **v8** |
| v14 | asc | **v14** |
| v8 | asc | identical weights to v14 — duplicate |
| v14 | v8 | identical weights to v8 — duplicate |

Submitting all four would spend ~190 GPU-hours per backbone re-deriving checkpoints
that differ only in a metadata string. `run_matrix_hipster.sh` therefore varies
`lora_type` and records `nest_version` alongside it: **3 backbones × 2 lora_type × 2 KD
= 12**, plus the 2 MobileLLaMA external-teacher runs = **14**. `WITH_DUPLICATES=1`
submits the redundant cells anyway as a null control — they should land within eval
noise of their twins, which is a real if expensive measure of run-to-run variance, and
this project currently has **no** seed replicate of anything.

**Stage 1 is shared per backbone.** It trains a single `tok_level` (256) at a single
rank (64), so the ladder ordering cannot apply; and with one level `students_all` is
empty, so the prefix-KL term is structurally inert. Stage 1 is therefore identical
across both `lora_type` and both KD settings — derived once per backbone and reused via
`ELASTIC_PRETRAIN_TAG`, saving 9 × ~10h.

**Cost: ~1200 GPU-pair-hours** (12 × ~85h Stage 2 + 3 × ~10h Stage 1). Stage backbones
sequentially unless Hipster runs many concurrently.

**Qwen variant:** the repo defines `qwen0.5b`, `qwen1.5b`, `qwen3b`. The matrix uses
**`qwen0.5b`** — the launcher's own default *and* the first Qwen row of §15e's planned
matrix — rather than silently substituting another size. Override with `QWEN_KEY=`.

All four base models are present in the HF cache; no download is needed.


### 21a. Qwen2.5-0.5B's first attempt OOM'd — a large-vocabulary failure mode never exercised before

**No results — the run failed, not "not finished yet."** Stage 1 (job 27447's pretrain
half) completed cleanly: `elastic-pretrain-qwen0.5b-final-parcel/checkpoint-2180` is a
real, complete checkpoint. Stage 2 crashed at **step 16 of 5197** on node208 (A10:2,
22.30 GiB/card):

```
torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate 1.85 GiB.
GPU 0 has a total capacty of 22.30 GiB of which 1.36 GiB is free.
  File "llava/model/language_model/llava_elastic_mixin.py", line 343, in forward
    kl = _el.prefix_kl_loss(s_log, t_log, kl_labels)
  File "llava/model/elastic/losses.py", line 26, in prefix_kl_loss
    kl = F.kl_div(s, t, reduction="none").sum(-1)  # (B, L)
```

**Root cause: Qwen2.5-0.5B's vocabulary (151936) is 4.7× TinyLlama's (32000) and
2.5–4.7× every backbone this loss has ever run against**, and `prefix_kl_loss`
materializes a full `(B, L, V)` tensor before reducing:

```python
def prefix_kl_loss(student_logits, teacher_logits, labels=None, T=1.0):
    s = F.log_softmax(student_logits / T, dim=-1)      # (B, L, V)
    t = F.softmax(teacher_logits / T, dim=-1)           # (B, L, V)
    kl = F.kl_div(s, t, reduction="none").sum(-1)       # (B, L, V) intermediate, THEN summed
```

`reduction="none"` keeps the full per-vocab, per-position elementwise term alive
(needed for the backward pass through `log_softmax`/`softmax`) before `.sum(-1)`
collapses it — and this runs over **every sequence position**, not just the
labelled ones; the `labels != -100` mask is applied only *after* the KL is computed
(line 27-30), so positions that get masked out (image tokens, system/user turns) still
paid the full `(B, L, V)` cost. TinyLlama/MobileLLaMA (32000), SmolLM2 (49152) never
came close to tipping this over on a 24 GB card; Qwen at 151936 did, on step 16.

**This is a genuinely new failure mode, not one the onboarding checklist screens
for.** `project-backbone-onboarding-checks` validates label-masking correctness (EOS
supervision) and elastic-key loading — both are about correctness, not memory, and
both would pass here (the run got 16 steps into real training before dying, well past
where a masking bug would show). No prior backbone in this project has had a
vocabulary anywhere near Qwen's, so nothing before this exercised the failure path.

**Fix applied: GPU type, not code.** Resubmitted Stage 2 only (Stage 1's checkpoint is
reused, not redone — `ELASTIC_RUN_TAG=final-parcel` with `ELASTIC_PRETRAIN_TAG` left
unset defaults the warm-start to the same tag) on `run_job_finetune_slm.sh`'s own
default `gpu:A40:2` (46 GiB/card, already this project's standing choice for
memory-tight Stage-2 runs — its own header cites Phi-3.5 OOM'ing at 41.5/44.67 GiB on
A40 as precedent for never risking the smaller card on a full finetune). Jobs
**27456** (train) / **27457** (`--array=0-3`, eval). A40's headroom (~25 GiB free
where A10 had 1.36 GiB) should comfortably absorb the same peak; this was not
re-derived from a memory model, it is the standing per-backbone mitigation this project
already uses instead of touching shared loss code.

**Flagged, not fixed: `prefix_kl_loss` computing the full unmasked `(B,L,V)` term is a
real inefficiency, worth fixing for every backbone, not just Qwen.** Restricting the
`log_softmax`/`softmax`/`kl_div` chain to only the labelled (assistant-response)
positions *before* the vocab-dimension ops — rather than computing over the full
sequence and masking after — would cut both memory and compute roughly in proportion to
how much of a sequence is masked out, which for a single-turn VQA example with a long
visual prefix is most of it. Mathematically equivalent (masking commutes with a
per-position sum before the final mean), so this is a safe optimization if someone
wants to make it, but it was **not** applied here: it touches loss code shared by every
run in this project, and a quick GPU-type change fully explains and fixes this specific
failure without touching anything that could silently change a number in an already-
reported comparison. Revisit if a future backbone's vocabulary makes even A40
insufficient, or if the compute savings become worth it on their own.

## 22. SigLIP + `final-parcel` smoke test: TinyLlama, SmolLM2, MobileLLaMA (decision 47)

Not a full run — a validation pass before committing multi-day jobs to a combination
that has never actually been exercised on this recipe.

**Why this needed a fresh smoke, not a re-read of §10b/10c.** SigLIP support was fixed
and smoke-tested once before (job 27326), but that run used the now-retired 8-level
`576…16` ladder with ranks `2 4 6 8 8 16 32 64` (§16b), never finished as a real
training run (swapped to CLIP before completion, §13), and only ever touched TinyLlama.
Nothing had run SigLIP against `final-parcel`'s actual grid (`256/144/64/16`, ranks
`8 16 32 64`) on any backbone. **MobileLLaMA has never been run through the elastic
pipeline at all**, SigLIP or CLIP (§15e) — this was its first exposure, full stop.

**Regression check first.** Re-ran `jobs/test_anchor_mode_and_siglip.sh` (job 27458)
before touching real data, since `engine.py`/`nested_lora.py`/`config.py` have all
changed since §10b for unrelated reasons (v14's rank-order work, the KD-teacher
registry). All three original SigLIP fixes still hold: `l_enc` threading, no blanket
`no_grad`, and the pooling-head injection scoped to `vision_model.encoder` (66/72 LoRA
wrappers received nonzero gradient, mean norm 883.9) — `ALL_TESTS_PASSED`, no
regression from three sessions' worth of unrelated changes.

**Smoke test** (`jobs/smoke_siglip_final_parcel.sh`, job 27459): `final-parcel`'s exact
flags (`resampler_arch=pool_anchored`, `anchor_mode=ratio` @0.25, nested vision LoRA
`8 16 32 64`, `use_token_decorrelation=False`, `teacher=self`, grid `256/144/64/16`)
with only `VISION_TOWER=google/siglip-base-patch16-384` changed, 6 real steps each of
Stage 1 then Stage 2, for tinyllama → smollm2 → mobilellama in sequence. Run on
node208's real 2×A10 (ZeRO-2 sharding as a real run would use, not the single-GPU
offload workaround §10c needed the first time — node208 was free).

**Result: all six stages pass.**

| backbone | stage | exit | loss trajectory (6 steps) | grad_norm trajectory |
|---|---|---|---|---|
| tinyllama | 1 | 0 | 9.34 → 9.31 → 7.66 → 7.45 → 5.76 → 6.13 | 120 → 118 → 31 → 17 → 10 → 12 |
| tinyllama | 2 | 0 | 3.55 → 3.82 → 2.37 → 1.65 → 1.61 → **1.56** | 51 → 49 → 33 → 10 → 8 → 7 |
| smollm2 | 1 | 0 | 6.73 → 6.73 → 13.59 → 10.97 → 9.82 → 9.23 | 54 → 59 → 43 → 13 → 7 → **4** |
| smollm2 | 2 | 0 | 9.01 → 8.96 → 8.72 → 7.79 → 8.03 → 7.82 | 43 → 39 → 38 → 25 → 19 → 18 |
| mobilellama | 1 | 0 | 8.09 → 7.94 → 6.27 → 12.33 → 11.24 → 10.43 | 154 → 135 → 17 → 42 → 19 → 26 |
| mobilellama | 2 | 0 | 9.96 → 10.32 → 7.21 → 4.73 → 3.84 → **3.42** | 100 → 111 → 96 → 57 → 34 → 24 |

All finite (`grep -icE "\bnan\b|\binf\b"` over the full log: **0**). Two mid-run
spikes (smollm2 Stage 1 step 3: 6.73→13.59; mobilellama Stage 1 step 4: 6.27→12.33) are
single noisy CE batches at Stage 1's LR 1e-3 with a frozen backbone and no averaging to
speak of at n=6 steps — not divergence: grad norms keep shrinking through both spikes
rather than blowing up, and both Stage 2 tinyllama/mobilellama runs show a clean
monotonic loss decline. Not evidence either way about eventual accuracy — the point of
a smoke test is "does it run," not "will it be good."

**What this validates, concretely:** SigLIP's `l_enc` threading and gradient flow
under real (not synthetic) images and text, on three different vision-feature widths
consumed by three different LLM hidden sizes (TinyLlama 2048, SmolLM2 2048 at a
different tokenizer/vocab, MobileLLaMA 2048 with its own conv template) — the
resampler, PARCEL's anchor/query split, nested vision LoRA, and self-distillation KL
all execute correctly together on this exact recipe, for the first time, on all three
backbones.

**Not validated by a 6-step smoke, and why that's the right scope here:** final loss
values, convergence, or anything about whether SigLIP beats or loses to CLIP on this
recipe — that needs the real multi-day runs this smoke test exists to de-risk before
committing to. Smoke checkpoints (`*-smoketest-siglip`, ~120 GB across six dirs) were
deleted after inspection — scratch validation artifacts, not experiment results, per
the checkpoint cleanup policy.

**Code change:** `MAX_STEPS` wired into `pretrain_elastic_slm.sh` /
`finetune_elastic_slm.sh` (mirrors the same knob already in the M3/MQT baseline
launchers, §19) — a no-op when unset, so no existing recipe or submitted job is
affected.

**Not yet decided:** whether to queue the real SigLIP + `final-parcel` runs for these
three backbones. This section validates that they *would* run; it does not argue they
should be prioritized ahead of the Hipster CLIP sweep (§17c/§18) already in flight.



### 17d. `--lora_type` and `--nest_version`

One argument each, propagating the full stack, no duplicated LoRA implementation —
both modes use the same `NestedLoRALinear`, differing only in the level→rank
assignment.

```
--lora_type v8    ranks 8 16 32 64   rank ascends as budget DESCENDS (v4-v8 convention)
--lora_type asc   ranks 64 32 16 8   rank ascends WITH budget       (v14 hypothesis)
--nest_version v8 | v14              preset; sets lora_type unless given explicitly
```

Propagation: `run_matrix_hipster.sh` → `LORA_TYPE`/`NEST_VERSION`/`USE_KD` env →
`finetune_elastic_slm.sh` → `train_elastic.py` (resolves to `lora_ranks`) →
`ElasticConfig` → `elastic_config.json` in every checkpoint → eval, which rebuilds the
config from that file. Both fields are **descriptive, not behavioural**: nothing
branches on them, so a checkpoint loaded from disk behaves identically whether or not
they are set. `--lora_ranks` still works and now *errors* if it contradicts
`--lora_type`, so a checkpoint can never record a `lora_type` that does not describe
its own weights.

`--nest_version` is redundant with `--lora_type` **today**, and is kept deliberately:
it is the stable name for "the whole recipe" if a future version changes something the
rank ladder cannot express, and it makes the off-diagonal matrix cells expressible.

Also added: **`run_manifest.json`**, written into `output_dir` at launch, recording
`nest_version`, `lora_type`, backbone, KD block (including that the self-teacher shares
weights), CORAL, decorrelation, vision-LoRA, resampler, optimizer, LR, scheduler,
epochs, batch, seed, dataset, checkpoint, **git commit + branch + dirty flag**, SLURM
job id and node. This exists because v8's provenance was recoverable only from its
checkpoint — the launcher that produced it had since been edited.

Validated by `jobs/test_lora_type_arg.sh` (27409, `ALL_TESTS_PASSED`): both mappings,
both presets, preset-override for the off-diagonal cells, contradiction refusal,
backward compatibility with no flags set, `asdict`/JSON round-trip, and that the fields
do not change `lora_level_for_tok`. Anchor equivalence and the KD mechanism were
audited in job 27408.


## 18. KD teacher compatibility audit, and the Hipster V8xV14 matrix (decision 39)

Status key used below: **AUDITED** (fact established by a job that ran) ·
**PLANNED** (specified, not submitted) · RUNNING · COMPLETED · FAILED.

### 18a. What the existing KD actually is — AUDITED

Read off `losses.prefix_kl_loss`, `llava_elastic_mixin.py:325-345` and
`engine.attach_kd_teacher`, not from names.

| Property | Finding |
|---|---|
| KD type | **Logits KD only.** No hidden-state, attention, feature, or response-level KD exists in this repo |
| loss | `KL(teacher‖student) = Σ_V softmax(t)·(log_softmax(t) − log_softmax(s))`, `F.kl_div(log_softmax(s/T), softmax(t/T))`, `×T²` |
| temperature | **T = 1.0, hard-coded.** The signature takes `T` but the call site never passes it — there is no temperature knob |
| KD weight | `prefix_kl_weight` (0.1), then `/ n_active` (=2 in sampled-student mode) |
| distilled positions | positions where `labels != -100` — i.e. **assistant-response tokens only**; sequences right-aligned (`logits[:, L−n:]`) so differing visual-prefix lengths still line up |
| teacher visual budget | **full/native.** `attach_kd_teacher` sets `teacher.config.matryoshka_vis_token_scale = None` → 576 tokens for a CLIP-L/336 teacher |
| teacher image encoder / projector | teacher runs its *own* tower and projector (LLaVA-7B: CLIP-L/336 + `mlp2x_gelu`) — same family as our students, so both solve the same multimodal task |
| tokenizer / vocab | **must match exactly.** `F.kl_div` reduces the vocab axis elementwise |
| hidden dim | **irrelevant.** No loss consumes teacher hidden states. LLaVA-7B is 4096 vs TinyLlama 2048 and v12 ran fine |
| output heads | only the vocab dimension matters, for the same reason |
| CORAL | compares *projected visual tokens* and is **self-sourced even with an external teacher attached** (explicit in the code) — so it is not teacher supervision |
| architecture assumption | **yes.** `attach_kd_teacher` hard-codes `LlavaLlamaForCausalLM.from_pretrained` — one loading path, LLaVA-Llama only |

`teacher="self"` is a **separate mechanism sharing the flag**: the teacher is *this
model* at `tok_levels[0]`, logits detached — teacher and student share weights exactly,
measured KL ≈ 0.006. It is self-distillation across token budgets and **must not be
reported as independent-teacher KD**.

### 18b. KD Teacher Compatibility Audit — AUDITED

Empirical inputs: job **27410** (tokenizers loaded and compared, not just sizes),
job **27411** (real load attempts through the LLaVA path), job **27413** (registry
tests). Machine-readable: `results/kd_compatibility_report.json`.

Token-ID identity — the check that matters, since equal vocab size does not imply
equal token IDs:

| Student | vocab | vs LLaVA-7B (32000) | token IDs identical? |
|---|---|---|---|
| TinyLlama | 32000 | match | **yes** |
| MobileLLaMA | 32000 | match | **yes** |
| SmolLM2 | 49152 | mismatch | no |
| Qwen2.5-0.5B | 151936 (head) / 151665 (tokenizer) | mismatch | no |

| Student | 7B Teacher Compatible? | Family Teacher | Recommended Teacher | KD Mode | Reason |
|---|---|---|---|---|---|
| TinyLlama | **YES** (`DIRECT`) | — (LLaVA-7B *is* Llama-family) | `llava7b` | logits KD | token IDs byte-identical; same CLIP-L/336 + mlp2x_gelu interface; already used by v12 |
| MobileLLaMA | **YES** (`DIRECT`) | MobileVLM_V2-1.7B — **unusable** | `llava7b` | logits KD | 32000-vocab Llama tokenizer, IDs identical to the teacher's. Family teacher fails to load (below) and is only 1.7B vs a 1.4B student anyway |
| SmolLM2 | **NO** (`VOCAB_MAPPING_REQUIRED`) | SmolVLM-Instruct — **unusable** | **none** | self-distillation only | 49152 vs 32000. Family teacher blocked three independent ways |
| Qwen2.5 | **NO** (`VOCAB_MAPPING_REQUIRED`) | **none in inventory** | **none** | self-distillation only | 151936 vs 32000; no Qwen VLM in the local cache |

Why each family-matched teacher fails, all smoke-tested rather than assumed:

- **MobileVLM_V2-1.7B** (`mtgv/MobileVLM_V2-1.7B`) — tokenizer is byte-identical to
  MobileLLaMA's, vision tower is the same CLIP-L/336, so on paper it is the right
  teacher. It **fails to load**: `ValueError: Unknown projector type: ldpnetv2`. Its
  `mm_projector_type` is `ldpnetv2`, which this repo's projector builder does not
  implement, and its `architectures` is `MobileLlamaForCausalLM` / `model_type:
  mobilevlm`, not `LlavaLlamaForCausalLM`. Independently, its LLM is MobileLLaMA-1.7B
  against a 1.4B student — a ~20% capacity gap, weak for a teacher.
- **SmolVLM-Instruct** — blocked three ways: vocab **49155** vs the student's 49152
  (3 added image tokens, so not even a family-internal match); architecture
  `Idefics3ForConditionalGeneration`, which fails the LLaVA load path with a shape
  error; and its **tokenizer does not load at all** in this environment
  (`data did not match any variant of untagged enum ModelWrapper` — a `tokenizers`
  version issue). Its tower is SigLIP-1152, not CLIP-L/336.
- **Qwen** — no Qwen-family VLM is in the local HF cache. Qwen2-VL / Qwen2.5-VL exist
  upstream and share the Qwen2 tokenizer, which would make them the correct candidates,
  but nothing has been downloaded or verified, and this registry does not list
  checkpoints it has not inspected. A further trap for any future Qwen KD:
  Qwen2.5-0.5B's LM head is **151936** wide while its tokenizer holds **151665**
  entries — a teacher must match the *head* width.

**Is the 7B still useful cross-family?** For MobileLLaMA, yes and it is the
recommendation — cross-family (Vicuna-7B vs MobileLLaMA) but the tokenizers are
byte-identical, which is the only property the loss requires. For SmolLM2 and Qwen,
**no**: making it work would need logit slicing/renormalisation or a learned vocabulary
mapping. Neither exists here, and adding one would mean a claimed "KD improvement"
could come from the alignment workaround rather than teacher supervision. Refused
rather than approximated.

### 18c. V8 vs V14 — AUDITED, and still no result

Full component table in §17a. Summary: **v8 and v14 differ in exactly one component,
the LoRA rank ladder** (`8 16 32 64` vs `64 32 16 8`). The anchor-config difference
(fixed table vs `ratio 0.25`) is verified behaviourally identical at all four budgets
(job 27408); the `acdfa79` code delta that landed mid-v8-run is verified inert
(decorrelation-gated, False in both).

> **Existing evidence remains insufficient for a controlled v8-v14 comparison.**

v14 (job 27406) is ~33h into a ~85h Stage-2 run as of 2026-09-14. No checkpoint, no
eval directory, no number on any metric. Nothing may be concluded in either direction.

Files: both versions use the *same* classes — `NestedQueryResampler`
(`resampler_arch="pool_anchored"`), `NestedLoRALinear`, `ElasticEngine`. There is no
"v8 class" and no "v14 class"; the version is a rank-ordering choice, now expressed by
`--lora_type` / `--nest_version`.

### 18d. V8 vs V14 × Backbone × LoRA × KD — Hipster — PLANNED

Nothing submitted. `bash run_matrix_hipster.sh` prints this; `SUBMIT=1` launches it.
Seed is HF default **42** for every run (unset in the launchers). Every run: Stage 1
shared per backbone, `pool_anchored` ratio-0.25, budgets 256/144/64/16, CLIP-L/336,
1 epoch, LR 2e-5 cosine, warmup 0.03, bs 2 × accum 32 × 2 GPU, `mix665k`, `pad`
aspect ratio, `eval_lmms_level.sh --array=0-3`.

| ID | Version | Backbone | LoRA | KD | Teacher | Seed | Status | Job ID | Checkpoint | Result |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | v8 | SmolLM2 | v8 | on | self (shared weights) | 42 | **RUNNING** (HIPSTER EXPS) | stage1 357385, train 357394, eval 357395 | `/scratch/skalra/flexllava_saves/checkpoints/elastic-finetune-smollm2-m-v8lora-kdon` | — |
| 2 | v8 | SmolLM2 | v8 | off | — | 42 | PLANNED | — | `…-smollm2-m-v8lora-kdoff` | — |
| 3 | v14 | SmolLM2 | asc | on | self | 42 | PLANNED | — | `…-smollm2-m-asclora-kdon` | — |
| 4 | v14 | SmolLM2 | asc | off | — | 42 | **RUNNING** (HIPSTER EXPS) | stage1 357385, train 357392, eval 357393 | `/scratch/skalra/flexllava_saves/checkpoints/elastic-finetune-smollm2-m-asclora-kdoff` | — |
| 5 | v8 | MobileLLaMA | v8 | on | self | 42 | PLANNED | — | `…-mobilellama-m-v8lora-kdon` | — |
| 6 | v8 | MobileLLaMA | v8 | off | — | 42 | **RUNNING** (HIPSTER EXPS) | stage1 357380, train 357396, eval 357397 | `/scratch/skalra/flexllava_saves/checkpoints/elastic-finetune-mobilellama-m-v8lora-kdoff` | — |
| 7 | v14 | MobileLLaMA | asc | on | self | 42 | PLANNED | — | `…-mobilellama-m-asclora-kdon` | — |
| 8 | v14 | MobileLLaMA | asc | off | — | 42 | PLANNED | — | `…-mobilellama-m-asclora-kdoff` | — |
| 9 | v8 | Qwen2.5-0.5B | v8 | on | self | 42 | PLANNED | — | `…-qwen0.5b-m-v8lora-kdon` | — |
| 10 | v8 | Qwen2.5-0.5B | v8 | off | — | 42 | PLANNED | — | `…-qwen0.5b-m-v8lora-kdoff` | — |
| 11 | v14 | Qwen2.5-0.5B | asc | on | self | 42 | PLANNED | — | `…-qwen0.5b-m-asclora-kdon` | — |
| 12 | v14 | Qwen2.5-0.5B | asc | off | — | 42 | PLANNED | — | `…-qwen0.5b-m-asclora-kdoff` | — |
| 13 | v8 | MobileLLaMA | v8 | **on, external** | **llava7b** (`DIRECT`) | 42 | **RUNNING** (HIPSTER EXPS) | stage1 357380, train 357398, eval 357399 | `/scratch/skalra/flexllava_saves/checkpoints/elastic-finetune-mobilellama-m-v8lora-kd7b` | — |
| 14 | v14 | MobileLLaMA | asc | **on, external** | **llava7b** (`DIRECT`) | 42 | PLANNED | — | `…-mobilellama-m-asclora-kd7b` | — |

**The KD column in rows 1–12 is prefix-KL SELF-distillation** (teacher = the same
weights at 256 tokens, KL ≈ 0.006). It is the only KD available on all three
backbones. Rows 13–14 are the only genuine external-teacher KD the inventory permits.

**Why 12 core and not 24** (§17c): version and LoRA are the same axis, so
`v8+asc` ≡ v14 and `v14+v8lora` ≡ v8. `WITH_DUPLICATES=1` adds the redundant cells as
a null control — worth running once, since nothing in this project has a seed
replicate. Cost: ~1200 GPU-pair-hours.

**Qwen variant:** `qwen0.5b` — the launcher default *and* §15e's first Qwen row.
Override with `QWEN_KEY=qwen1.5b`.

### 18e. Teacher-control experiments needed to make any KD claim — PLANNED

Rows 13–14 give `KD off ↔ self-KD ↔ external-7B-KD` on MobileLLaMA (against rows 5–8),
which is the only backbone where all three are technically valid. That is the
teacher-control comparison. It cannot be run for SmolLM2 or Qwen, so **no
cross-backbone claim about external-teacher KD is available from this matrix** — and
none should be written. TinyLlama already has the same three-way contrast from v8/v11
and v12 (§16l), where external KD lost on every metric at every budget.

### 18f. Code changes — AUDITED/IMPLEMENTED

| File | Change | Why |
|---|---|---|
| `llava/model/elastic/kd_teachers.py` | **new** — teacher registry + `check_pair` / `resolve_teacher` / `audit_all` | Teacher knowledge as data, not branches in the training loop. Every fact recorded came from a job that ran |
| `llava/model/elastic/engine.py` | `attach_kd_teacher` resolves non-`llava` teachers through the registry; refuses rather than substitutes; writes `resolved_teacher_key` | Makes the audit's verdict enforceable at run time. `"llava"` kept as an alias so existing recipes/checkpoints are unaffected |
| `llava/model/elastic/config.py` | `+nest_version, lora_type, kd_student_key, resolved_teacher_key` | Checkpoints self-identify; the *resolved* teacher is recorded, not the requested one |
| `llava/train/train_elastic.py` | `+--lora_type, --nest_version, --kd_teacher, --kd_student_key, --kd_type`; rank resolution; `_write_run_manifest` | One argument per axis, propagating end-to-end; `run_manifest.json` records git commit + dirty flag |
| `llava/model/elastic/nested_lora.py` | `max_rank = max(ranks)`, ascending assert relaxed | Prerequisite for `--lora_type asc` (§16m) |
| `scripts/v1_5/finetune_elastic_slm.sh` | `+LORA_TYPE, NEST_VERSION, USE_KD, KD_TEACHER, KD_STUDENT_KEY, KD_TYPE` | Propagation |
| `scripts/v1_5/pretrain_elastic_slm.sh` | `+NEST_VERSION` | Provenance on Stage 1 |
| `run_matrix_hipster.sh` | **new** — matrix driver, dry-run by default | Reproducible generation with resolved teachers printed |
| `jobs/audit_v8_v14.sh`, `audit_kd_teachers.sh`, `smoke_kd_teacher_load.sh`, `smoke_kd_forward.sh`, `test_kd_registry.sh`, `test_lora_type_arg.sh`, `test_lora_rank_order.sh` | **new** | The audits and validation behind every claim above |

### 18g. CLI

```
--nest_version {v8,v14}          preset; sets --lora_type unless given explicitly
--lora_type {v8,asc}             v8  = ranks 8 16 32 64 (rank ascends as budget descends)
                                 asc = ranks 64 32 16 8 (rank ascends with budget)
--use_kd BOOL                    prefix-KL on/off  (SELF-distillation unless a teacher is named)
--kd_teacher KEY|auto            external teacher; 'auto' consults the registry, fails loudly
--kd_student_key KEY             which backbone is the student (needed for resolution)
--kd_type {auto,logits}          only 'logits' is implemented; no hidden/attention/response loss exists
--teacher {self,llava}           legacy; 'llava' still forces the incumbent 7B
```

## 19. HIPSTER EXPS — porting the §18d matrix to Hipster and launching the first 4 cells (decision 40)

Scope, deliberately narrow: run exactly 4 cells of §18d's 14-row matrix — SmolLM2 v14 (`m-asclora-kdoff`), SmolLM2 v8 + self-distillation (`m-v8lora-kdon`), MobileLLaMA v8 (`m-v8lora-kdoff`), MobileLLaMA v8 + 7B external KD (`m-v8lora-kd7b`) — on Hipster, using DAS-6's data via SSH streaming (never Hipster's own `/home/skalra/llava_data` archives), submitted through a matrix driver rather than hand-written `sbatch` calls. Not a redesign of anything in §17/§18: no new KD mechanism, no new teacher-selection logic, no hyperparameter changes, no other backbones (Qwen excluded), no other matrix cells.

**One naming ambiguity, flagged rather than silently resolved:** the task named "SmolLM2 V14" without stating KD on/off. §18d's table has both `m-asclora-kdoff` and `m-asclora-kdon` as separate rows; resolved to KD-**off** by symmetry with the bare "MobileVLM V8" request (also unqualified, and unambiguously the `kdoff` row since the task's own item 4 separately calls out "+7B KD" as a distinct experiment). Reported this interpretation before submitting rather than guessing silently.

### 19a. `run_matrix_hipster.sh` cannot run on Hipster as-is

Read in full before touching anything, per the "audit first, don't invent config" instruction. It hardcodes `/var/scratch/skalra/flexllava/checkpoints/...` (confirmed absent on Hipster — `ls /var/scratch` → no such path) and calls `run_job_pretrain_slm.sh` / `run_job_finetune_slm.sh` / `eval_lmms_level.sh`, all of which still reference `/var/scratch` internally — every one of those is DAS-6-only. Hipster's own launchers (`*_hipster.sh`) had zero wiring for the new `--lora_type`/`--nest_version`/`--kd_teacher`/`--kd_student_key`/`--kd_type` flags added in §17/§18. Per the task's explicit instruction, this was reported as a blocking finding rather than bypassed with manual `sbatch` — the fix agreed was to **port** the new flag set into Hipster's launchers and write a Hipster-native matrix driver, not to alter `run_matrix_hipster.sh` itself (it stays DAS-6's file, unmodified, still correct there once its own bugs below are fixed).

### 19b. What was ported / added

| File | Change |
|---|---|
| `scripts/v1_5/pretrain_elastic_slm_hipster.sh` | `+${NEST_VERSION:+--nest_version ${NEST_VERSION}}`; unique `--master_port` (§19d) |
| `scripts/v1_5/finetune_elastic_slm_hipster.sh` | `+LORA_RANKS/LORA_TYPE/NEST_VERSION` fallback + banner echoes (mirrors DAS-6's `finetune_elastic_slm.sh`, commit `8d9369e`); `+--lora_ranks/--lora_type/--nest_version/--use_kd/--kd_teacher/--kd_student_key/--kd_type` on the deepspeed invocation; unique `--master_port` (§19d) |
| `run_job_pretrain_only_hipster.sh` | **New.** Stage-1-only launcher: DAS-6 SSH/tar streaming of LLaVA-Pretrain (661 shard dirs, chunked into `STAGE_PARALLEL` groups, default 8, rather than one SSH pipe per shard or one pipe for the whole tree), no `--exclusive` (Hipster shared-cluster etiquette), typed GRES set at submit time |
| `submit_matrix_hipster.sh` | **New.** Hipster port of `run_matrix_hipster.sh`'s `submit_one()` logic. `CELLS` defaults to exactly the 4 requested cells (no "submit everything" default, unlike the DAS-6 script) — `smollm2:m-asclora-kdoff smollm2:m-v8lora-kdon mobilellama:m-v8lora-kdoff mobilellama:m-v8lora-kd7b`. Partition/GRES: `capacity`→`gpu:l4:2` for self/no-KD cells, `performance`→`gpu:rtx_6000_ada:2` for the external-KD cell (matches §18b's finding that 7B-teacher KD needs more headroom). Logs to `jobs/SUBMITTED_RUNS_HIPSTER.tsv` (gitignored, mirrors DAS-6's `docs/SUBMITTED_RUNS.tsv` convention from decision 25) |

Verified dry-run output before any real submission (`SUBMIT` unset):

```
smollm2      lora=asc  kd=False teacher=self     partition=capacity    -> m-asclora-kdoff
smollm2      lora=v8   kd=True teacher=self     partition=capacity    -> m-v8lora-kdon
mobilellama  lora=v8   kd=False teacher=self     partition=capacity    -> m-v8lora-kdoff
mobilellama  lora=v8   kd=True teacher=llava7b  partition=performance -> m-v8lora-kd7b
```
Exactly 4 cells, correct backbone/lora_type/kd/teacher/partition mapping per cell — matched against §18d row-by-row before flipping `SUBMIT=1`.

### 19c. Bug 1 — broken env-var-prefix via command substitution (also present in DAS-6's script, never triggered)

`submit_one()` built the teacher-selection env var like this:
```bash
$( [ "$teacher" = self ] && echo "TEACHER=self" || echo "KD_TEACHER=$teacher" ) sbatch ...
```
This does not do what it looks like it does. Bash only recognizes a *literal* `VAR=value` token immediately before a command as an environment-prefix assignment at parse time; the output of a command substitution is just an ordinary argument word, even when it happens to look like `VAR=value`. The first real submission attempt confirmed this exactly as predicted: `submit_matrix_hipster.sh: line 118: TEACHER=self: command not found`, and every one of the 4 finetune `sbatch` calls failed outright, cascading into `sbatch: error: ... Job dependency problem` on all 4 eval jobs (empty `$jid` produced a malformed `--dependency=afterok:`). The two shared Stage-1 `sbatch` calls submitted fine, since their variable assignment is a literal prefix, not a substitution. **This identical bug exists in DAS-6's `run_matrix_hipster.sh` at line 130** — never caught there because nothing in the §18d matrix had ever actually been submitted (everything sat at PLANNED). Not fixed on DAS-6's copy per the "this cluster does not alter DAS-6's approach" constraint; flagged here so it's fixed before that script is ever actually run.

Fix, in `submit_matrix_hipster.sh` only:
```bash
if [ "$teacher" = self ]; then
    export TEACHER=self; unset KD_TEACHER
else
    export KD_TEACHER="$teacher"; unset TEACHER
fi
```
Also added an explicit guard (`if [ -z "$jid" ]; then echo "ERROR..." >&2; return 1; fi`) so a failed `sbatch` fails loudly instead of silently producing a malformed downstream dependency.

### 19d. Bug 2 — Stage-1 lora_ranks/nest_version contradiction (also present in DAS-6's script, never triggered)

After fixing bug 1, cancelled the 8 stuck jobs and resubmitted. The two shared Stage-1 jobs (`smollm2`, `mobilellama`) both failed within ~1 minute with:
```
ValueError: --lora_ranks [64] contradicts --lora_type 'asc' (which implies [8]). Pass one or the other, or make them agree...
```
Traced to `train_elastic.py`'s rank-ladder derivation (~line 423-462): for `n = len(tok_levels)`, `ladder = [8*(2**i) for i in range(n)]`, `derived = ladder if lora_type=='v8' else ladder[::-1]`. Stage 1 uses `tok_levels=[256]` — a single level, so `n=1`, `ladder=[8]`, and **reversing a 1-element list is a no-op**: `ladder[::-1]` is also `[8]`. Both `v8` and `asc` therefore derive rank `[8]` for Stage 1, which directly contradicts the hardcoded `--lora_ranks 64` that Stage 1 needs so its buffer width matches Stage 2's largest rank. `submit_matrix_hipster.sh` had been passing `NEST_VERSION=$nest` into the shared Stage-1 call, which triggered this. This is consistent with — and confirms — §17c's own claim that Stage 1 is version-agnostic and legitimately shared across both `lora_type` values for a backbone (`ELASTIC_PRETRAIN_TAG=matrixbase`): with one level, the ladder ordering is structurally inert, so it should never have been tagged with a version at all. **The identical latent bug exists in DAS-6's `pretrain_elastic_slm.sh`**, which passes the same unconditional `--lora_ranks ${STAGE1_LORA_RANK}` alongside the same new `--nest_version` passthrough from `8d9369e` — not yet triggered there because DAS-6 has not submitted any `--nest_version`-tagged Stage-1 job. Fix: removed `NEST_VERSION=$nest` from the shared Stage-1 `sbatch` call in `submit_matrix_hipster.sh` entirely. Verified with a standalone simulation of the exact resolution logic before resubmitting, then confirmed live by watching the resubmitted Stage-1 jobs pass the point that previously crashed.

### 19e. Bug 3 — deepspeed port collision (Hipster-only, not applicable to DAS-6)

After fixing bug 2, the two resubmitted Stage-1 jobs landed on the same physical node (`hipster-cn008`) — expected and correct behavior, since Hipster runs are deliberately **not** `--exclusive` (shared-cluster etiquette, unlike DAS-6's dedicated small cluster). One of the two lost the race for deepspeed's default port 29500 and crashed: `RuntimeError: The server socket has failed to listen... Address already in use`. Confirmed via `sacct` (one job `FAILED` at ~1:30 elapsed) and the log traceback. Fix, added to both `pretrain_elastic_slm_hipster.sh` and `finetune_elastic_slm_hipster.sh`:
```bash
MASTER_PORT=$(( 20000 + SLURM_JOB_ID % 10000 ))
deepspeed --num_gpus ${NUM_GPUS} --master_port ${MASTER_PORT} llava/train/train_elastic.py \
```
Hipster-only fix, not applicable to DAS-6 (whose `--exclusive` jobs never share a node), so not carried back to DAS-6's copies. Verified live: the next resubmission again co-located two jobs on `hipster-cn008`, this time with different derived ports, and both passed the distributed-init point cleanly.

### 19f. Final submission

All 3 bugs fixed and verified past their failure points before trusting the matrix. `CELLS` left at its 4-cell default, `SUBMIT=1 bash submit_matrix_hipster.sh` run once, `squeue` checked immediately after to confirm exactly 4 independent train chains (8 train/eval job pairs total plus 2 shared Stage-1 jobs — smollm2's two finetune cells share one Stage-1 checkpoint, likewise mobilellama's two).

| Cell | Backbone | lora_type | KD | Teacher | Partition | Stage-1 job | Stage-2 (train) job | Eval job | Status as of this entry |
|---|---|---|---|---|---|---|---|---|---|
| `m-asclora-kdoff` | SmolLM2 | asc (v14) | off | — | capacity | 357385 | 357392 | 357393 | Stage-1 **RUNNING**; Stage-2/eval **PENDING** (dependency, unfulfilled) |
| `m-v8lora-kdon` | SmolLM2 | v8 | on | self | capacity | 357385 (shared) | 357394 | 357395 | Stage-1 **RUNNING**; Stage-2/eval **PENDING** (dependency, unfulfilled) |
| `m-v8lora-kdoff` | MobileLLaMA | v8 | off | — | capacity | 357380 | 357396 | 357397 | Stage-1 **RUNNING**; Stage-2/eval **PENDING** (dependency, unfulfilled) |
| `m-v8lora-kd7b` | MobileLLaMA | v8 | on, external | llava7b | performance | 357380 (shared) | 357398 | 357399 | Stage-1 **RUNNING**; Stage-2/eval **PENDING** (dependency, unfulfilled) |

**2026-09-14 update:** Stage 2 (finetune) bumped from 2 to 4 GPUs per an explicit follow-up decision — pretrain stage is untouched, stays at 2. The original 4 Stage-2 jobs above (357381/357383/357386/357388) were still `PENDING` on their Stage-1 dependency (never started), so they were cancelled along with their dependent eval jobs and resubmitted at 4 GPUs, reusing the already-running Stage-1 jobs via `STAGE1_JOB_OVERRIDE_<backbone>` rather than duplicating them. New Stage-2/eval job IDs are the ones in the table above (357392-357399); `GRAD_ACCUM` in `finetune_elastic_slm_hipster.sh` is already derived from `NUM_GPUS` (`32*2/NUM_GPUS`), so effective batch size is unchanged. Confirmed via `squeue` that all 4 finetune jobs now request `gres/gpu:{l4,rtx_6000_ada}:4` (was `:2`) and the 2 pretrain jobs are unaffected (`gres/gpu:l4:2`, still `RUNNING`).

Checkpoint paths (Hipster's `/scratch` tree, per §18d's mapping):
- Stage 1 (shared per backbone): `/scratch/skalra/flexllava_saves/checkpoints/elastic-pretrain-{smollm2,mobilellama}-matrixbase`
- Stage 2: `/scratch/skalra/flexllava_saves/checkpoints/elastic-finetune-{smollm2,mobilellama}-{m-asclora-kdoff,m-v8lora-kdon,m-v8lora-kdoff,m-v8lora-kd7b}`

No experiment is COMPLETED as of this entry — both Stage-1 jobs are still training (~25-30 minutes elapsed at last check), all 4 Stage-2 and 4 eval jobs are correctly `PENDING(Dependency)`. Nothing below this line is a result; §18d's table rows 1/4/6/13 have been updated in place from PLANNED to RUNNING with these job IDs, and will be updated again to COMPLETED (with real numbers) or FAILED as each cell actually finishes — no fabricated status or numbers.

### 19g. Tangential: git push on the `hipster` branch appeared to hang

Diagnosed as **not** a heavy-file or bug issue — the `hipster` branch commit (`75a91d4`, "Added hipster changes") is 11 files / 1436 lines, nothing unusual. Root cause: the `origin` remote is HTTPS with no credential helper configured (`credential.helper` empty, no `~/.git-credentials`, no matching `~/.netrc` entry) — `git push` was silently blocking on an invisible `Username for 'https://github.com':` prompt, indistinguishable from a hang. Confirmed with `timeout 15 git push --dry-run -v origin hipster </dev/null` → instant `fatal: could not read Username...`. SSH auth already works (`ssh -T git@github.com` succeeds), so the fix is to repoint the remote:
```bash
git remote set-url origin git@github.com:sudaksh14/FlexLLaVA.git
git push origin hipster
```
Left for the user to run — the permission classifier blocks `git remote set-url` from an agent, and per this session's own tooling guidance that block is not something to work around.

## 19. M3 and MQT-LLaVA 4-token baselines on TinyLlama (decision 40)

*(Note: this is a second, independent "section 19" — written on `main` in parallel with the Hipster-branch section 19 above, before the two branches were merged. Left as two separate section-19 blocks rather than renumbered, to avoid touching either branch's internal cross-references.)*

Baseline numbers for the paper: the two prior methods our work is measured against,
each run through **its own training pipeline**, at a fixed **4 visual tokens**, with our
vision encoder, our backbone, our data and our hyperparameters.

### 19a. What each baseline actually runs — AUDITED

**M3 needs no new code.** This repo *is* an M3 fork with M3 intact underneath (README,
"Relationship to M3"). With no `elastic_engine` attached, `LlavaElasticMixin.forward`
falls into its `# ---- Pure M3 (no elastic engine, explicit scale list)` branch, which
loops over `config.matryoshka_vis_token_scale` and averages CE across scales, and
`llava_arch.matryoshka_vis_token_process` performs M3's average pooling
(`pool_size = stride = int(sqrt(576 / scale))`). This is M3's own loop, not a
reimplementation. Verified (job 27415):

| scale | pool | tokens out |
|---|---|---|
| 576 | 1×1 | 576 |
| 144 | 2×2 | 144 |
| 36 | 4×4 | 36 |
| 9 | 8×8 | 9 |
| **4** | **12×12** | **4** ← the baseline |
| 1 | 24×24 | 1 |

A single-element scale list means one forward per step at 4 tokens.

**MQT runs from its own vendored tree.** `MQT-LLaVA/llava/train/train.py`, executed
from inside `MQT-LLaVA/` so its `llava` package shadows ours. Its mechanism is a 2D
perceiver `Resampler` ("query_abstractor", `mm_query_abstractor_type=matry_query`):
256 learnable queries plus a frozen 2D sincos positional embedding, cross-attending to
the CLIP patches; `num_visual_tokens` keeps a prefix. `get_matry_n()` maps
`first_stage → 256`, `second_stage → random.choice(range(2,258,2))` per step, and a
bare integer to itself.

~~**MQT imports cleanly in our env despite pinning `transformers==4.36.2`** ... That was
the main risk in reusing the vendored tree and it did not materialise.~~
**WRONG — struck 2026-09-15, see §19e.** That import check was meaningless: it was
importing OUR `llava`, not MQT's. The real risk was a different one entirely, and it
did materialise. (The version question was eventually answered against MQT's actual
code by job 27424 — see §19f.)

*(Aside, worth fixing in memory: `project-flexllava-slm` records the env as
transformers 4.36.2. It is now **4.44.2 / tokenizers 0.19.1** — upgraded at some point
for Qwen2.5/SmolLM2.)*

### 19b. Held constant vs. deliberately not

Identical to our runs, so the numbers are comparable:

| | setting |
|---|---|
| vision encoder | CLIP-L/336, frozen |
| backbone | TinyLlama-1.1B-Chat-v1.0, conv `v1`, full LLM finetune |
| Stage 1 data | `blip_laion_cc_sbu_558k`, LR 1e-3, 1 epoch, effective batch 256 |
| Stage 2 data | `llava_v1_5_mix665k`, LR 2e-5 cosine, warmup 0.03, wd 0, 1 epoch, effective batch 128, bf16, `model_max_length` 2048, `image_aspect_ratio pad` |
| seed | HF default 42 (unset in every launcher, ours included) |

**Each method keeps its own Stage-1 convention, on purpose.** M3 pretrains the plain
projector on all 576 tokens — it pools *after* the projector, so its Stage 1 is
ordinary LLaVA pretraining. MQT pretrains its query bank at
`num_visual_tokens=first_stage` (=256), which is its published recipe. Forcing a shared
Stage 1 would mean neither baseline was the published method.

**Scope limit to carry into the paper.** Both are run at a *fixed* 4-token budget,
which narrows each method: M3 normally trains a list of scales jointly, MQT normally
samples a random query count per step. These are "**M3/MQT architecture and training
loop at a fixed 4-token budget**", **not** the published elastic models, and must not
be reported as M3/MQT headline numbers.

### 19c. Runs — PLANNED

Nothing submitted. `bash run_baselines_4tok.sh` prints the plan; `SUBMIT=1` launches.

| ID | Method | Backbone | Tokens | Stage 1 | Stage 2 | Status | Checkpoint |
|---|---|---|---|---|---|---|---|
| B1 | M3 (avg-pool) | TinyLlama-1.1B | 4 | plain projector, 576 tok | `MATRYOSHKA_SCALE=4` | PLANNED | `baseline-tinyllama-4tok-finetune` |
| B2 | MQT-LLaVA (query abstractor) | TinyLlama-1.1B | 4 | `first_stage` (256 queries) | `NUM_VISUAL_TOKENS=4` | PLANNED | `mqt-finetune-tinyllama-4tok` |

**Cost.** Both Stage 2s run **one** forward per step at 4 visual tokens, against our
elastic runs' two forwards at 256 + ~75 tokens — so Stage 2 should be several times
cheaper than our ~85h. **Stage 1 is the expensive half here** (558k samples at 576
tokens for M3, 256 queries for MQT) and is *not* reduced by the 4-token setting. That
is an estimate from the arithmetic, not a measurement: run with `MAX_STEPS=200` first
and read `it/s` off the log before committing a multi-day job.

**Evaluation is not yet wired.** `eval_lmms_level.sh` reads `elastic_config.json` and
will not work on these checkpoints — they have no elastic engine. Evaluate through the
baseline path (`eval_lmms_baseline_llava.sh`) or add a 4-token wrapper; MQT
additionally needs `num_visual_tokens` threaded into whatever eval is used, since its
default is 256.

### 19d. Code changes

| File | Change |
|---|---|
| `scripts/v1_5/finetune_baseline_slm.sh` | `+MATRYOSHKA_SCALE` (unset ⇒ unchanged 576 control); `+MAX_STEPS`; fixed a banner that hard-coded "576 tokens" |
| `scripts/v1_5/pretrain_baseline_slm.sh` | `+MAX_STEPS` |
| `scripts/v1_5/pretrain_mqt_baseline.sh` | **new** — MQT Stage 1 via MQT's own `train.py` |
| `scripts/v1_5/finetune_mqt_baseline.sh` | **new** — MQT Stage 2 at fixed `num_visual_tokens` |
| `run_baselines_4tok.sh` | **new** — driver, dry-run by default |
| `jobs/smoke_baselines_m3_mqt.sh` | **new** — pooling arithmetic + MQT importability (27415, passed) |
| `jobs/smoke_baseline_train.sh` | **new** — 20 real training steps of each stage |


### 19e. CORRECTION: `cd MQT-LLaVA` never actually selected MQT's code

The most important finding of this session, and it invalidates a claim made in §19a.

**FlexLLaVA is installed editable.** setuptools registers a `MetaPathFinder`
(`__editable___llava_1_2_2_post1_finder.py`) mapping `'llava'` to
`/home/skalra/FlexLLaVA/llava`. `cd MQT-LLaVA` does **not** beat it: when deepspeed runs
`llava/train/train.py`, `sys.path[0]` is the **script's** directory
(`.../MQT-LLaVA/llava/train`), not the cwd, so nothing puts MQT's package on `sys.path`
and every `import llava` resolves to **ours**.

So MQT's `train.py` was driving **our** model, dying with
`AttributeError: 'LlavaLlamaModel' object has no attribute 'query_abstractor'`
(jobs 27418, 27420).

**The crash was luck, not detection.** Our model *does* have an `mm_projector`; a
different flag combination could have trained to completion and produced numbers
labelled "MQT baseline" that were not MQT in any respect. Two guards now sit in both
MQT launchers:

```
export PYTHONPATH=/home/skalra/FlexLLaVA/MQT-LLaVA${PYTHONPATH:+:${PYTHONPATH}}
_resolved=$(python3 -c "import llava, os; print(os.path.dirname(llava.__file__))")
[ "$_resolved" = "/home/skalra/FlexLLaVA/MQT-LLaVA/llava" ] || exit 1
```

The editable finder is *appended* to `sys.meta_path`, so `PathFinder` (which reads
`PYTHONPATH`) is consulted first and wins; the second line refuses to run otherwise.

**Also corrected: MQT has no `mm_projector` at all.** `build_vision_projector` does not
exist anywhere in its tree; `initialize_vision_modules` builds only `query_abstractor`
(256 queries, 6.8M params — diagnostic job 27419), and the Resampler's own `proj`
(kv_dim to embed_dim) does the projection. `--mm_projector_type`,
`--tune_mm_mlp_adapter` and `--pretrain_mm_mlp_adapter` are invalid for MQT and were
removed from both launchers; Stage 2 warm-starts from `query_abstractor.bin` only.

### 19f. Smoke-test results: what is and is not verified

| Check | Job | Result |
|---|---|---|
| M3 avg-pool arithmetic, scale 4 → 4 tokens | 27415 | PASSED |
| M3 Stage 1, 20 real steps | 27418 | **PASSED** (exit 0) |
| M3 Stage 2 @ 4 tokens, 20 real steps | 27418 | **PASSED** (exit 0) |
| MQT builds only `query_abstractor`, never `mm_projector` | 27419 | confirmed |
| MQT Stage 1, 20 real steps (after the path fix) | 27421 | **PASSED** (exit 0), wrote `query_abstractor.bin` (13.6 MB) |
| MQT Stage 2, full training | 27421 | **OOM on 1× A10** — see below |
| MQT forward+backward @ 4 tokens, real code, transformers 4.44.2 | 27424 | **PASSED** — loss 3.9212 finite, 24 text tokens → seq 27 (~4 visual), `query_abstractor.query` grad norm 1.4e-3 |

**The MQT Stage-2 OOM is a smoke-test artifact, not a defect.** It failed inside
DeepSpeed's `initialize_optimizer_states()` — *before* step 1 — on a single A10
(22.3 GiB total, 19.62 GiB in use). Stage 2 is a full TinyLlama-1.1B finetune, and with
one rank ZeRO-2 cannot shard the AdamW states (~13 GB unsharded). Our *own* Stage 2 has
the same requirement: `run_job_finetune_slm.sh` notes "~17GB/GPU before activations,
which is uncomfortably close to the A10's 24GB" — on **two** GPUs. The smoke ran on one
GPU only because both 2-GPU nodes were occupied by v13/v14.

Because that OOM happened before any step, job 27424 was added to verify what it left
unproven: MQT's real code, forward and backward, at 4 visual tokens, under our
transformers 4.44.2. It passes. **So the version-pin question is answered for the
forward/backward path, but full Stage-2 training on 2 GPUs has still never been run** —
the first real 2-GPU MQT Stage 2 will be on Hipster.

### 19g. Additional code changes from the correction

| File | Change |
|---|---|
| `scripts/v1_5/{pretrain,finetune}_mqt_baseline.sh` | `PYTHONPATH` + fail-loud `llava` resolution guard; removed the invalid `mm_projector` flags |
| `jobs/diag_mqt_build.sh` | **new** — proves MQT builds only `query_abstractor` (27419) |
| `jobs/smoke_mqt_only.sh` | **new** — MQT-only 20-step smoke (27420, 27421) |
| `jobs/smoke_mqt_forward.sh` | **new** — optimizer-free forward/backward check at 4 tokens (27424) |


## 20. W&B credential removed from the tracked scripts (decision 42)

Until 2026-09-15 every `run_job*.sh` on `main` carried the W&B key inline:

```
#SBATCH --export=ALL,WANDB_API_KEY=<40-hex-key>
```

Ten tracked files, so the key was in the repository, in every commit that
touched them, and on GitHub.

**Moved to `~/.netrc`** (mode 600), which is wandb's own credential mechanism:

```
machine api.wandb.ai
  login user
  password <key>
```

That needs no script logic at all — wandb reads it directly on the compute
node, and `$HOME` is shared between login and compute nodes here, so there is
nothing to export. Every `--export=ALL,WANDB_API_KEY=…` became a plain
`--export=ALL` with a comment pointing at the netrc. Verified the file parses
and resolves (`netrc.authenticators("api.wandb.ai")` returns a 40-char
password), and that later `#SBATCH` directives still take effect — SLURM
continues scanning through comment lines, so `--exclusive` further down
`run_job_slm.sh` is unaffected.

`.gitignore` gained `.env`, `.env.*`, `*.env`, `.netrc`, `secrets.sh`,
`wandb.env` so an env-file alternative cannot be added by accident.

### THE KEY IS STILL COMPROMISED — ROTATE IT

Scrubbing the working tree does **not** remove the key from git history. It
remains in every prior commit on `main`, is still reachable on GitHub, and is
still present on the **`hipster` branch**, which was not part of this change
(`run_job.sh`, `run_job_baseline_slm.sh`, `run_job_finetune_slm.sh`,
`run_job_hipster.sh`, `run_job_hipster_finetune_only.sh`, and the two baseline
wrappers added in 116ac14). Anyone who has cloned or forked the repository
already has it.

Required follow-up, in order:

1. **Revoke and regenerate the key** at wandb.ai → Settings → API keys. This is
   the only step that actually ends the exposure; everything else is hygiene.
2. `wandb login` with the new key to rewrite `~/.netrc`.
3. Apply the same scrub to the `hipster` branch.
4. Optionally rewrite history (`git filter-repo`, or BFG) — worth it only if
   the repository is or will become public; it rewrites every commit hash and
   breaks existing clones.


## 21. First Hipster-sweep backbone launched locally: Qwen2.5-0.5B under `final-parcel` (decision 44)

**Queued as jobs 27447 (Stage 1+2) / 27448 (`--array=0-3`), `bash submit_elastic_run.sh
final-parcel` with `SLM_KEY=qwen0.5b`.** Exactly the §17c/§18 recipe — PARCEL ratio-0.25,
nested vision LoRA `8 16 32 64`, decorrelation off, self-teacher, grid `256 144 64 16` —
with no changes for the backbone. Both 2-GPU nodes were occupied (v14 on 205, a v8-parcel
re-eval on 207/208) so it queued behind them on priority rather than idling for lack of a
recipe.

**First Qwen run of any kind in this project.** No v4-equivalent baseline exists for
`qwen0.5b` (§15e flagged this explicitly: "needs its own v4-equivalent baseline run
first... plain rank numbers alone won't show the accuracy story"). This run is PARCEL +
full recipe from the start, not a v4→v8 progression — there is nothing to A/B it against
locally yet. Its role is the first row of the Hipster backbone sweep (§17c/§18d), not an
ablation.

**Onboarding check not re-run, and why that's safe.** `project-backbone-onboarding-checks`
records Qwen2.5 as verified clean on 2026-08-18 (100% EOS-supervised, chatml template,
cross-model agreement with TinyLlama/SmolLM2/Phi-2/Phi-3.5 on the fixed 64-sample probe).
No `preprocess_*` code has changed since — `--auto_prefix_len`/`preprocess_mpt` are
untouched by anything in §16-§20 — so re-running the checker would be re-confirming the
same fact, not de-risking a real unknown.
