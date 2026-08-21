"""Source-grouped batch sampler, so per-source telemetry is actually usable.

THE PROBLEM THIS SOLVES
-----------------------
Per-source loss attribution (telemetry.py) can only happen on micro-batches
that contain a single source, because CE is a token-mean over the batch. Under
the stock `group_by_modality_length` sampler that condition almost never holds
for the sources that matter. Simulated on the real 665k baseline mixture with
the real geometry (per_device=2, 2 GPUs, accum=32) -- jobs/otterhom_26736.out:

    source      homogeneous     observations per 25-step window
    coco             50.4%      74
    text_only       100.0%      16     (mm/lang are already segregated)
    gqa              13.2%       4
    ocr_vqa           6.6%       2
    textvqa           0.7%       0
    vg                0.0%       0

ocr_vqa and textvqa are precisely the detail-hungry sources the token-budget
hypothesis is about, and they get nothing. The instrument would have produced a
confident-looking `otter/gap/all` and silence exactly where the answer lives.

Length-grouping is why: it sorts by sequence length, and the sources overlap
heavily in length, so a length-similar pair is usually a cross-source pair.
coco only does well because it is 55% of the data and often pairs with itself.

THE FIX, AND WHY IT IS SAFE
---------------------------
Group by (source, length) instead of (modality, length): partition by source,
length-group within each source, cut each source into micro-batches, then
shuffle the micro-batches together. Every micro-batch is single-source, so
attribution is ~100%.

This does NOT bias the optimizer. gradient_accumulation_steps is 32, so one
optimizer step averages 32 micro-batches drawn from the shuffled stream -- it
still sees a proper mixture of sources. What changes is only which samples
share a *forward pass*.

It should also reduce padding: same-source samples have far more similar
lengths than length-matched cross-source ones, and length-grouping still runs
inside each source.

Left opt-out via --otter_source_grouped_batches False, which restores the stock
sampler (and with it the near-useless per-source telemetry).
"""

from __future__ import annotations

import random
from typing import Dict, Iterator, List, Optional, Sequence

import torch
from torch.utils.data import Sampler

# Reuse the shipped intra-group length ordering rather than reimplementing it.
from llava.train.llava_trainer import get_length_grouped_indices_auto_single


class SourceGroupedLengthSampler(Sampler):
    """Yields indices such that every consecutive run of `batch_size` is one source.

    accelerate's DataLoaderShard hands each rank contiguous chunks of
    `batch_size` from this order, so emitting a flat list whose every aligned
    `batch_size`-run is single-source is sufficient -- no custom batch sampler
    or collate contract is needed.
    """

    def __init__(self,
                 batch_size: int,
                 world_size: int,
                 lengths: Sequence[int],
                 source_names: Sequence[str],
                 generator=None,
                 seed: int = 0):
        if lengths is None or source_names is None:
            raise ValueError("lengths and source_names are both required")
        if len(lengths) != len(source_names):
            raise ValueError(
                f"lengths ({len(lengths)}) and source_names ({len(source_names)}) "
                f"must be the same length")
        self.batch_size = batch_size
        self.world_size = world_size
        self.lengths = lengths
        self.source_names = source_names
        self.generator = generator
        self.seed = seed
        self._epoch = 0

    def __len__(self) -> int:
        return len(self.lengths)

    def set_epoch(self, epoch: int) -> None:
        self._epoch = epoch

    def __iter__(self) -> Iterator[int]:
        # Own seeded generator rather than the global torch RNG. The stock
        # sampler passes generator=None and so depends on whatever global state
        # set_seed left behind, which makes two iterations in one process differ
        # and makes a mixture ablation harder to reproduce exactly. Deriving it
        # from (seed, epoch) makes the order a pure function of the run config.
        gen = self.generator
        if gen is None:
            gen = torch.Generator()
            gen.manual_seed(int(self.seed) + int(self._epoch))

        by_source: Dict[str, List[int]] = {}
        for i, s in enumerate(self.source_names):
            by_source.setdefault(s, []).append(i)

        batches: List[List[int]] = []
        leftovers: List[int] = []

        for source in sorted(by_source):                 # sorted -> deterministic
            idx = by_source[source]
            # Magnitudes only: the modality sign is meaningless within a source
            # (a source is either all-image or all-text), and the length
            # grouper asserts non-zero.
            mags = [max(abs(self.lengths[i]), 1) for i in idx]
            order = [idx[j] for j in get_length_grouped_indices_auto_single(
                mags, self.batch_size, self.world_size, generator=gen)]
            # Whole batches only; a ragged tail would straddle two sources once
            # the per-source streams are concatenated.
            n_full = len(order) // self.batch_size * self.batch_size
            for k in range(0, n_full, self.batch_size):
                batches.append(order[k:k + self.batch_size])
            leftovers.extend(order[n_full:])

        # Shuffle the single-source batches together, so an accumulation window
        # (32 micro-batches) still averages over a proper mixture of sources.
        rng = random.Random(self.seed + self._epoch)
        rng.shuffle(batches)

        out: List[int] = [i for b in batches for i in b]
        # At most (batch_size - 1) per source, so a handful of samples overall;
        # these tail batches may be mixed and simply will not be attributed.
        out.extend(leftovers)
        return iter(out)


def homogeneity_of(order: Sequence[int],
                   source_names: Sequence[str],
                   batch_size: int) -> float:
    """Fraction of aligned batch_size-chunks that are single-source. For tests."""
    n = hom = 0
    for i in range(0, len(order) - batch_size + 1, batch_size):
        chunk = order[i:i + batch_size]
        n += 1
        if len({source_names[j] for j in chunk}) == 1:
            hom += 1
    return hom / n if n else 0.0
