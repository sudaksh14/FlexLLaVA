"""Trainer for the Otter-style pipeline: LLaVATrainer + per-source telemetry.

Subclasses LLaVATrainer rather than replacing it, so every behaviour the
current runs depend on is inherited unchanged -- the length-grouped samplers,
the elastic/mm_projector learning-rate groups in create_optimizer(), the
non-finite-loss guard, the per-tok-level `_loss_components` logging, and the
tune_mm_mlp_adapter checkpoint handling.

What it adds:

  * pops the collator's `otter_meta` payload before the batch reaches the
    model, and feeds it to OtterTelemetry (per-source loss, per-source x
    per-level CE, supervision density, throughput, image fallbacks);
  * optional per-source loss weighting, off by default.

`llava/train/train.py` and `llava/train/llava_trainer.py` are not modified;
train_otter.py swaps this class in by assignment.
"""

from __future__ import annotations

import time
from typing import Dict, Optional

import torch

from llava.data_otter.sampler import SourceGroupedLengthSampler
from llava.data_otter.telemetry import OtterTelemetry
from llava.train.llava_trainer import LLaVATrainer


class OtterTrainer(LLaVATrainer):

    # Set by train_otter.py before Trainer construction.
    otter_log_every: int = 25
    otter_loss_weighting: bool = False
    otter_source_grouped_batches: bool = True

    def _get_train_sampler(self):
        """Single-source micro-batches, so per-source telemetry has data.

        Falls back to LLaVATrainer's stock sampler when disabled, or when the
        dataset carries no source tags (i.e. it is not an OtterMixtureDataset).
        See llava/data_otter/sampler.py for the measurement that motivated this:
        under the stock sampler ocr_vqa/textvqa/vg land in a single-source
        micro-batch 6.6% / 0.7% / 0.0% of the time, so their per-source loss is
        never observed.
        """
        if not self.otter_source_grouped_batches:
            return super()._get_train_sampler()
        ds = self.train_dataset
        source_names = getattr(ds, "source_names", None)
        lengths = getattr(ds, "lengths", None)
        if source_names is None or lengths is None:
            return super()._get_train_sampler()
        return SourceGroupedLengthSampler(
            self.args.train_batch_size,
            world_size=self.args.world_size * self.args.gradient_accumulation_steps,
            lengths=lengths,
            source_names=source_names,
            generator=None,
            seed=self.args.seed,
        )

    def _telemetry(self) -> OtterTelemetry:
        tel = getattr(self, "_otter_telemetry", None)
        if tel is None:
            tel = OtterTelemetry(log_every=self.otter_log_every)
            self._otter_telemetry = tel
        return tel

    def _source_weights(self) -> Dict[str, float]:
        """source name -> loss_weight, from the mixture config."""
        cached = getattr(self, "_otter_source_weights", None)
        if cached is None:
            cached = {}
            mixture = getattr(self.train_dataset, "mixture", None)
            if mixture is not None:
                cached = {n: s.loss_weight for n, s in mixture.sources.items()}
            self._otter_source_weights = cached
        return cached

    def _read_components(self, model) -> Optional[Dict[str, float]]:
        """Read `_loss_components` off whichever module the elastic forward ran as.

        Mirrors LLaVATrainer.compute_loss's own resolution: plain getattr first,
        then the explicit wrapper walk it falls back to. Reading it a second
        time here is safe -- the forward overwrites the dict every call, so this
        is always the current micro-batch's values.
        """
        components = getattr(model, "_loss_components", None)
        if components is None:
            if not hasattr(self, "_otter_owner_cache"):
                self._otter_owner_cache = self._elastic_owner(model)
            owner = self._otter_owner_cache
            if owner is not None:
                components = vars(owner).get("_loss_components")
        return components

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # The meta payload is not a model input. It must come out before the
        # batch reaches model(**inputs) or the forward gets an unexpected kwarg.
        meta = inputs.pop("otter_meta", None)

        now = time.time()
        last_end = getattr(self, "_otter_last_step_end", None)
        data_time = (now - last_end) if last_end is not None else None

        out = super().compute_loss(model, inputs, return_outputs=return_outputs, **kwargs)
        loss = out[0] if isinstance(out, tuple) else out

        # Optional per-source loss weighting. Applied as the mean of the
        # micro-batch's per-sample weights, because CE is a token-mean over the
        # whole batch and cannot be decomposed per sample without recomputing
        # it. Exact for a homogeneous batch, an approximation otherwise. Off
        # unless a source declares loss_weight != 1.0, and note that it scales
        # the gradient like a learning-rate change would.
        if self.otter_loss_weighting and meta:
            weights = self._source_weights()
            if weights:
                per_sample = [weights.get(s, 1.0) for s in meta.get("sources", [])]
                if per_sample:
                    scale = sum(per_sample) / len(per_sample)
                    if abs(scale - 1.0) > 1e-9:
                        loss = loss * scale
                        out = (loss,) + out[1:] if isinstance(out, tuple) else loss

        if meta is not None:
            step_time = time.time() - now
            components = self._read_components(model)
            try:
                loss_value = float(loss.detach().item())
            except (RuntimeError, ValueError):
                loss_value = float("nan")
            tel = self._telemetry()
            tel.record(meta, components, loss_value,
                       step_time=step_time, data_time=data_time)
            step = self.state.global_step
            if tel.should_flush(step):
                payload = tel.flush(step)
                if payload:
                    self.log(payload)

        self._otter_last_step_end = time.time()
        return out

    def _one_time_dataset_summary(self) -> None:
        """Log the realised mixture once, so a wandb run records what it trained on."""
        if getattr(self, "_otter_summary_logged", False):
            return
        self._otter_summary_logged = True
        ds = self.train_dataset
        source_names = getattr(ds, "source_names", None)
        if source_names is None:
            return
        counts: Dict[str, int] = {}
        for s in source_names:
            counts[s] = counts.get(s, 0) + 1
        total = len(source_names) or 1
        payload = {f"otter/mixture_share/{k}": v / total for k, v in sorted(counts.items())}
        payload["otter/mixture_size"] = float(total)
        payload["otter/lengths_exact"] = 1.0 if getattr(
            ds, "length_provenance", "heuristic") in ("cache", "built") else 0.0
        try:
            self.log(payload)
        except Exception:                            # noqa: BLE001 - never break a run to log
            pass

    def training_step(self, *args, **kwargs):
        self._one_time_dataset_summary()
        return super().training_step(*args, **kwargs)
