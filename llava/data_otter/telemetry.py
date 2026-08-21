"""Per-source / per-level training telemetry.

WHAT IS ALREADY THERE
---------------------
Per-TOK-LEVEL losses are already logged by the shipped pipeline: the elastic
forward fills `self._loss_components` with `loss/ce_tok256`, `loss/kl_tok16`,
... (llava/model/language_model/llava_elastic_mixin.py:339) and
LLaVATrainer.compute_loss forwards them to wandb once per optimizer step.
This module does not duplicate that.

WHAT IS MISSING, AND WHY IT MATTERS
-----------------------------------
Nothing attributes loss to the DATA SOURCE it came from.  That is precisely the
measurement needed to test the leading hypothesis about the token-budget
result: 16 visual tokens scores the same as 256, and the suspicion is that the
mixture is dominated by data (coco/vg/gqa) answerable from a coarse global
summary, with only ~15% (ocr_vqa/textvqa) that actually needs to read fine
detail.

If that hypothesis is right, then

    ce_tok16 - ce_tok256

should be ~0 on coco but clearly positive on ocr_vqa/textvqa.  If the gap is
~0 on EVERY source including the OCR ones, the mixture is not the problem --
the resampler is -- and reweighting the data would be wasted compute.  That
one number decides whether improvement #2 is worth running, which is why this
lands before the mixture work rather than after it.

HOW ATTRIBUTION WORKS (AND ITS ONE LIMITATION)
----------------------------------------------
The CE loss is a mean over the batch's supervised tokens, so a mixed-source
micro-batch cannot be decomposed without recomputing per-sample losses (which
would mean keeping per-sample logits for every level -- far too expensive).

Instead: attribute only when a micro-batch happens to be SOURCE-HOMOGENEOUS,
and count how often that holds.  With per_device_train_batch_size=2 and
group_by_modality_length grouping similar samples together, homogeneous batches
are the common case, so the estimates converge quickly.  `otter/hom_frac`
reports the fraction of batches that were usable, so the numbers are never
silently based on a handful of samples.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional

_TOK_RE = re.compile(r"^loss/(ce|kl|coral)_tok(\d+)$")


class AverageMeter:
    """Running mean since the last reset (Otter's, pipeline/train/train_utils.py:83)."""

    __slots__ = ("sum", "count")

    def __init__(self):
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1) -> None:
        self.sum += float(val) * n
        self.count += n

    @property
    def avg(self) -> Optional[float]:
        return self.sum / self.count if self.count else None

    def reset(self) -> None:
        self.sum = 0.0
        self.count = 0


class OtterTelemetry:
    """Accumulates per-source and per-source-per-level statistics between flushes."""

    def __init__(self, log_every: int = 25):
        self.log_every = max(1, int(log_every))
        self._last_flush_step = -1
        self.reset()

    def reset(self) -> None:
        self.src_counts: Dict[str, int] = {}
        self.src_loss: Dict[str, AverageMeter] = {}
        # (source, term, tok) -> meter, e.g. ("ocr_vqa", "ce", 16)
        self.src_level: Dict[tuple, AverageMeter] = {}
        # term/tok meters that ignore the source, so a mixed batch is not wasted
        self.level: Dict[tuple, AverageMeter] = {}
        self.n_batches = 0
        self.n_homogeneous = 0
        self.supervised = AverageMeter()
        self.padded = AverageMeter()
        self.seq_len = AverageMeter()
        self.n_packed = AverageMeter()
        self.image_fallbacks = 0
        self.step_time = AverageMeter()
        self.data_time = AverageMeter()
        self.samples = 0

    # ------------------------------------------------------------------
    def record(self,
               meta: Dict,
               components: Optional[Dict[str, float]],
               total_loss: float,
               step_time: Optional[float] = None,
               data_time: Optional[float] = None) -> None:
        sources: List[str] = meta.get("sources", [])
        bs = meta.get("batch_size", len(sources)) or 1

        self.n_batches += 1
        self.samples += bs
        for s in sources:
            self.src_counts[s] = self.src_counts.get(s, 0) + 1

        self.supervised.update(meta.get("supervised_tokens", 0) / bs, bs)
        self.padded.update(meta.get("padded_tokens", 0) / bs, bs)
        self.seq_len.update(meta.get("seq_len", 0))
        for n in meta.get("n_packed", []):
            self.n_packed.update(n)
        self.image_fallbacks += meta.get("image_fallbacks", 0)
        if step_time is not None:
            self.step_time.update(step_time)
        if data_time is not None:
            self.data_time.update(data_time)

        # Level meters that do not need homogeneity.
        if components:
            for key, value in components.items():
                m = _TOK_RE.match(key)
                if m:
                    term, tok = m.group(1), int(m.group(2))
                    self.level.setdefault((term, tok), AverageMeter()).update(value)

        # Source attribution needs a single-source micro-batch.
        homogeneous = bool(sources) and len(set(sources)) == 1
        if not homogeneous:
            return
        self.n_homogeneous += 1
        source = sources[0]
        self.src_loss.setdefault(source, AverageMeter()).update(total_loss)
        if not components:
            return
        for key, value in components.items():
            m = _TOK_RE.match(key)
            if m:
                term, tok = m.group(1), int(m.group(2))
                self.src_level.setdefault((source, term, tok), AverageMeter()).update(value)

    # ------------------------------------------------------------------
    def should_flush(self, global_step: int) -> bool:
        return (global_step != self._last_flush_step
                and global_step % self.log_every == 0
                and self.n_batches > 0)

    def flush(self, global_step: int) -> Dict[str, float]:
        """Return the wandb payload for this window and start a new one."""
        self._last_flush_step = global_step
        out: Dict[str, float] = {}

        total = sum(self.src_counts.values()) or 1
        for source, count in sorted(self.src_counts.items()):
            out[f"otter/share/{source}"] = count / total
        for source, meter in sorted(self.src_loss.items()):
            if meter.avg is not None:
                out[f"otter/loss/{source}"] = meter.avg

        for (source, term, tok), meter in sorted(self.src_level.items()):
            if meter.avg is not None:
                out[f"otter/{term}/{source}/tok{tok}"] = meter.avg
        for (term, tok), meter in sorted(self.level.items()):
            if meter.avg is not None:
                out[f"otter/{term}/all/tok{tok}"] = meter.avg

        # THE headline metric: how much worse is the smallest budget than the
        # largest, per source? A mixture whose elasticity axis does nothing
        # shows ~0 everywhere; a working one shows a positive gap that is
        # largest on the detail-hungry sources.
        out.update(self._token_budget_gaps())

        if self.n_batches:
            out["otter/hom_frac"] = self.n_homogeneous / self.n_batches
        for name, meter in (("supervised_tokens", self.supervised),
                            ("padded_tokens", self.padded),
                            ("seq_len", self.seq_len),
                            ("packed_per_sample", self.n_packed),
                            # forward_time is the compute_loss call itself: the
                            # whole elastic grid of active levels. between_forwards
                            # is everything else in the loop -- backward, the
                            # optimizer step on accumulation boundaries, AND the
                            # dataloader wait. It is not data wait alone, so read
                            # its trend across configs, not its absolute value.
                            ("forward_time", self.step_time),
                            ("between_forwards_time", self.data_time)):
            if meter.avg is not None:
                out[f"otter/{name}"] = meter.avg
        if self.supervised.avg is not None and self.seq_len.avg:
            # Supervision density. NOTE the denominator is the TEXT sequence:
            # the collator sees input_ids, where an image is a single
            # IMAGE_TOKEN_INDEX placeholder that's only expanded to tok_levels[0]
            # visual tokens inside prepare_inputs_labels_for_multimodal. So this
            # is "fraction of text positions supervised", not "fraction of the
            # model's actual sequence" -- for Stage 1 (seq_len ~73, 256 visual
            # tokens) the true figure is ~4x smaller. Use it to COMPARE configs
            # (it is what packing should move), not as an absolute.
            out["otter/supervised_frac"] = self.supervised.avg / self.seq_len.avg
        if self.step_time.avg:
            out["otter/samples_per_second"] = (self.samples / self.n_batches) / self.step_time.avg
        out["otter/image_fallbacks"] = float(self.image_fallbacks)

        self.reset()
        return out

    # ------------------------------------------------------------------
    def _token_budget_gaps(self) -> Dict[str, float]:
        """ce(smallest budget) - ce(largest budget), overall and per source.

        Also emits `otter/gap_n/<x>`, the number of observations behind the
        thinner of the two means. With --n_sample_students 1 only the teacher
        and ONE random student run per step, so the two levels in a gap are
        measured on different batches; a gap computed from a handful of
        observations is noise, and gap_n is what says so.
        """
        gaps: Dict[str, float] = {}

        def gap_for(prefix: str, ce_by_tok: Dict[int, tuple]) -> None:
            if len(ce_by_tok) < 2:
                return
            lo, hi = min(ce_by_tok), max(ce_by_tok)
            gaps[f"otter/gap/{prefix}"] = ce_by_tok[lo][0] - ce_by_tok[hi][0]
            gaps[f"otter/gap_n/{prefix}"] = float(min(ce_by_tok[lo][1], ce_by_tok[hi][1]))

        by_source: Dict[str, Dict[int, tuple]] = {}
        for (source, term, tok), meter in self.src_level.items():
            if term == "ce" and meter.avg is not None:
                by_source.setdefault(source, {})[tok] = (meter.avg, meter.count)
        for source, ce_by_tok in by_source.items():
            gap_for(source, ce_by_tok)

        overall = {tok: (m.avg, m.count) for (term, tok), m in self.level.items()
                   if term == "ce" and m.avg is not None}
        gap_for("all", overall)
        return gaps
