"""YAML-driven training mixture for the Otter-style FlexLLaVA data pipeline.

WHY THIS EXISTS
---------------
The shipped pipeline trains on one monolithic JSON
(`llava_v1_5_mix665k.json`) with no per-source control.  Changing the mixture
means rebuilding that file, so no mixture ablation has ever been cheap enough
to run.  That matters now for one specific reason: the elasticity axis
produces no measurable tradeoff (16 visual tokens scores the same as 256, see
memory `project-token-budget-has-no-effect`).  The leading data-side
explanation is that the mixture is dominated by data answerable from a coarse
global summary --

    coco 364100 | vg 86417 | ocr_vqa 80000 | gqa 72140
    text-only 40688 | textvqa 21953        (total 665298)

-- so only ~15% of samples (ocr_vqa + textvqa) actually require reading fine
detail, which is the only place a 256-token budget can beat a 16-token one.
This module makes "upweight the detail-hungry slices and re-measure" a
one-line config edit rather than a data-rebuild.

It borrows Otter's design (pipeline/mimicit_utils/mimicit_dataset.py): a YAML
listing named sources, each with a `num_samples` that up/down-samples it, and
sampling weights that follow from the resulting sizes.  It deliberately does
NOT borrow Otter's one-dataloader-per-group + `cycle()` + weighted-choice loop,
because that makes "epoch" ill-defined and would break HF Trainer's
checkpoint/resume accounting.  Resampling into a single indexed dataset keeps
every Trainer semantic intact.

CONFIG FORMAT
-------------
    version: 1

    defaults:
      image_folder: /var/scratch/skalra/flexllava/data/LLaVA-Finetune

    # Split one LLaVA-format JSON into named sources by the first component of
    # each sample's image path.  Samples with no "image" key become `text_only`.
    partition:
      data_path: /.../llava_v1_5_mix665k.json
      by: image_prefix          # or "none" -> everything lands in one source

    sources:
      coco:      {num_samples: -1}                 # -1 / 0 = use all, unchanged
      vg:        {num_samples: -1}
      gqa:       {num_samples: -1}
      ocr_vqa:   {num_samples: 160000}             # 2x upsample
      textvqa:   {num_samples: 43906}              # 2x upsample
      text_only: {num_samples: -1}

      # Sources may also bring their own file, for data not in the partitioned
      # JSON (DocVQA, ChartQA, ...).  `image_folder` overrides the default.
      docvqa:
        data_path: /.../docvqa_llava_format.json
        image_folder: /.../docvqa
        num_samples: 20000

    packing:                                       # see packing.py
      enabled: true
      sources: [gqa, ocr_vqa, textvqa]

    seed: 0

EVERY source named under `sources:` must exist in the partition (or carry its
own `data_path`); an unknown name is an error rather than a silent no-op,
because a typo'd source name would otherwise quietly drop that slice from
training.  Partition sources NOT named under `sources:` are dropped, loudly.
"""

from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

# `text_only` is not a path prefix -- it is the bucket for samples with no
# "image" key at all.  Named here so packing.py and verify.py agree on it.
TEXT_ONLY_SOURCE = "text_only"


@dataclass
class SourceSpec:
    """One named slice of the training mixture."""

    name: str
    num_samples: int = -1          # -1 or 0 -> keep every sample, unchanged
    loss_weight: float = 1.0       # see OtterTrainer; 1.0 everywhere = inert
    data_path: Optional[str] = None      # None -> comes from the partition
    image_folder: Optional[str] = None   # None -> defaults.image_folder
    pack: Optional[bool] = None          # None -> packing.sources decides


@dataclass
class PackingSpec:
    enabled: bool = False
    sources: List[str] = field(default_factory=list)
    max_turns: int = 6             # 3 QA pairs; keeps packed samples well under 2048
    max_chars: int = 6000          # cheap length proxy, no tokenizer needed at build time
    min_pack: int = 2              # never emit a "pack" of one -- that is just the original
    shuffle_within_pack: bool = True
    seed: int = 0


@dataclass
class MixtureConfig:
    sources: Dict[str, SourceSpec]
    partition_data_path: Optional[str] = None
    partition_by: str = "image_prefix"
    default_image_folder: Optional[str] = None
    packing: PackingSpec = field(default_factory=PackingSpec)
    seed: int = 0
    path: Optional[str] = None     # where this config was loaded from

    def signature(self) -> str:
        """Stable string identifying this mixture, for cache keys."""
        parts = [f"partition={self.partition_data_path}:{self.partition_by}", f"seed={self.seed}"]
        for name in sorted(self.sources):
            s = self.sources[name]
            parts.append(f"{name}:n={s.num_samples}:p={s.data_path}:pack={s.pack}")
        p = self.packing
        parts.append(f"packing={p.enabled}:{sorted(p.sources)}:{p.max_turns}:{p.max_chars}:{p.seed}")
        return "|".join(parts)


def load_mixture_config(path: str) -> MixtureConfig:
    """Parse and validate a mixture YAML.  Raises ValueError on any problem."""
    import yaml

    if not os.path.exists(path):
        raise ValueError(f"mixture config does not exist: {path}")
    with open(path, "r") as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict):
        raise ValueError(f"mixture config {path} did not parse to a mapping")

    unknown_top = set(raw) - {"version", "defaults", "partition", "sources", "packing", "seed"}
    if unknown_top:
        raise ValueError(
            f"mixture config {path}: unknown top-level key(s) {sorted(unknown_top)}. "
            f"Expected: version, defaults, partition, sources, packing, seed.")

    defaults = raw.get("defaults") or {}
    partition = raw.get("partition") or {}
    partition_by = partition.get("by", "image_prefix")
    if partition_by not in ("image_prefix", "none"):
        raise ValueError(f"mixture config {path}: partition.by must be 'image_prefix' or 'none', "
                         f"got {partition_by!r}")

    raw_sources = raw.get("sources")
    if not raw_sources:
        raise ValueError(f"mixture config {path}: 'sources' is empty; nothing would be trained on")

    sources: Dict[str, SourceSpec] = {}
    for name, spec in raw_sources.items():
        spec = spec or {}
        if not isinstance(spec, dict):
            raise ValueError(f"mixture config {path}: source {name!r} must be a mapping, "
                             f"got {type(spec).__name__}")
        unknown = set(spec) - {"num_samples", "loss_weight", "data_path", "image_folder", "pack"}
        if unknown:
            raise ValueError(f"mixture config {path}: source {name!r} has unknown key(s) "
                             f"{sorted(unknown)}")
        num_samples = spec.get("num_samples", -1)
        if not isinstance(num_samples, int):
            raise ValueError(f"mixture config {path}: source {name!r} num_samples must be an int, "
                             f"got {type(num_samples).__name__}")
        sources[name] = SourceSpec(
            name=name,
            num_samples=num_samples,
            loss_weight=float(spec.get("loss_weight", 1.0)),
            data_path=spec.get("data_path"),
            image_folder=spec.get("image_folder"),
            pack=spec.get("pack"),
        )

    raw_packing = raw.get("packing") or {}
    unknown_pack = set(raw_packing) - {"enabled", "sources", "max_turns", "max_chars",
                                       "min_pack", "shuffle_within_pack", "seed"}
    if unknown_pack:
        raise ValueError(f"mixture config {path}: packing has unknown key(s) {sorted(unknown_pack)}")
    packing = PackingSpec(
        enabled=bool(raw_packing.get("enabled", False)),
        sources=list(raw_packing.get("sources", [])),
        max_turns=int(raw_packing.get("max_turns", 6)),
        max_chars=int(raw_packing.get("max_chars", 6000)),
        min_pack=int(raw_packing.get("min_pack", 2)),
        shuffle_within_pack=bool(raw_packing.get("shuffle_within_pack", True)),
        seed=int(raw_packing.get("seed", 0)),
    )
    for name in packing.sources:
        if name not in sources:
            raise ValueError(f"mixture config {path}: packing.sources names {name!r}, which is "
                             f"not a declared source ({sorted(sources)})")
    if packing.max_turns < 2 or packing.max_turns % 2 != 0:
        raise ValueError(f"mixture config {path}: packing.max_turns must be a positive even "
                         f"number (human/gpt pairs), got {packing.max_turns}")

    cfg = MixtureConfig(
        sources=sources,
        partition_data_path=partition.get("data_path"),
        partition_by=partition_by,
        default_image_folder=defaults.get("image_folder"),
        packing=packing,
        seed=int(raw.get("seed", 0)),
        path=path,
    )

    # A source with no data_path can only be filled from a partition.
    needs_partition = [n for n, s in cfg.sources.items() if s.data_path is None]
    if needs_partition and not cfg.partition_data_path:
        raise ValueError(
            f"mixture config {path}: source(s) {sorted(needs_partition)} have no data_path and "
            f"there is no partition.data_path to draw them from")
    return cfg


def resample_indices(indices: List[int], n: int, seed: int) -> List[int]:
    """Otter's resample_data, on indices.

    n <= 0 (or n == len) keeps the slice untouched.  n > len upsamples by whole
    repetitions plus a seeded random remainder -- deliberately NOT n random
    draws, so a 2x upsample really is every sample exactly twice rather than a
    lumpy bootstrap.  n < len takes a seeded random subset.
    """
    if n is None or n <= 0 or n == len(indices):
        return list(indices)
    rng = random.Random(seed)
    if n > len(indices):
        repeat, remainder = divmod(n, len(indices))
        out = list(indices) * repeat
        out += rng.sample(indices, k=remainder) if remainder <= len(indices) \
            else rng.choices(indices, k=remainder)
        return out
    return rng.sample(indices, n)


def _image_prefix(record: Dict[str, Any]) -> str:
    """Source name for one record: first path component of its image, else text_only."""
    image = record.get("image")
    if not image:
        return TEXT_ONLY_SOURCE
    # Normalise separators so a Windows-style path or a leading "./" cannot
    # invent a distinct source name for the same slice.
    norm = str(image).replace("\\", "/").lstrip("./")
    head = norm.split("/", 1)[0]
    return head or TEXT_ONLY_SOURCE


def partition_records(records: List[Dict[str, Any]], by: str) -> Dict[str, List[int]]:
    """Bucket record indices into named sources."""
    buckets: Dict[str, List[int]] = {}
    if by == "none":
        buckets["all"] = list(range(len(records)))
        return buckets
    for i, rec in enumerate(records):
        buckets.setdefault(_image_prefix(rec), []).append(i)
    return buckets


def build_mixture(cfg: MixtureConfig,
                  log=print) -> Tuple[List[Dict[str, Any]], List[str], Dict[str, str]]:
    """Materialise the mixture.

    Returns (records, source_names, image_folders) where records[i] is a LLaVA
    conversation dict, source_names[i] names its slice, and image_folders maps
    source name -> the folder its images resolve against.

    Records are shared by reference when a sample is upsampled: the dataset
    never mutates them, and copying 665k dicts to no purpose would cost real
    memory on a 2-GPU node.
    """
    # Imported here, not at module scope: packing.py imports PackingSpec from
    # this module, so a top-level import would be circular.
    from .packing import pack_records

    records: List[Dict[str, Any]] = []
    source_names: List[str] = []
    image_folders: Dict[str, str] = {}

    # ---- partitioned sources ------------------------------------------------
    partition_buckets: Dict[str, List[int]] = {}
    partition_records_list: List[Dict[str, Any]] = []
    if cfg.partition_data_path:
        if not os.path.exists(cfg.partition_data_path):
            raise ValueError(f"partition.data_path does not exist: {cfg.partition_data_path}")
        with open(cfg.partition_data_path, "r") as f:
            partition_records_list = json.load(f)
        partition_buckets = partition_records(partition_records_list, cfg.partition_by)
        declared = set(cfg.sources)
        present = set(partition_buckets)
        # Loud about both directions: a dropped slice and a typo'd name are the
        # two ways a mixture silently stops being what the config says it is.
        for missing in sorted(declared - present):
            if cfg.sources[missing].data_path is None:
                raise ValueError(
                    f"source {missing!r} is declared with no data_path but does not appear in "
                    f"{cfg.partition_data_path} (present: {sorted(present)})")
        for dropped in sorted(present - declared):
            log(f"[otter-mix] NOTE: partition slice {dropped!r} "
                f"({len(partition_buckets[dropped])} samples) is not declared in sources -> dropped")

    rows = []
    for name in sorted(cfg.sources):
        spec = cfg.sources[name]
        if spec.data_path is not None:
            if not os.path.exists(spec.data_path):
                raise ValueError(f"source {name!r}: data_path does not exist: {spec.data_path}")
            with open(spec.data_path, "r") as f:
                own = json.load(f)
            base = own
        else:
            base = [partition_records_list[i] for i in partition_buckets[name]]
        n_available = len(base)

        # Packing runs BEFORE resampling.  The other order would group an
        # upsampled source's duplicate copies of a record back into one pack and
        # silently undo the upsample.
        pack_this = spec.pack if spec.pack is not None else (
            cfg.packing.enabled and name in cfg.packing.sources)
        if pack_this and name != TEXT_ONLY_SOURCE:
            base = pack_records(base, cfg.packing, source_name=name, log=log)

        kept = resample_indices(list(range(len(base))), spec.num_samples, seed=cfg.seed)
        for i in kept:
            records.append(base[i])
            source_names.append(name)

        folder = spec.image_folder or cfg.default_image_folder
        if folder is None and name != TEXT_ONLY_SOURCE:
            raise ValueError(f"source {name!r}: no image_folder and no defaults.image_folder")
        image_folders[name] = folder
        rows.append((name, n_available, len(base), len(kept), spec.loss_weight))

    total = len(records)
    log("[otter-mix] mixture built from " + (cfg.path or "<inline>"))
    log(f"[otter-mix] {'source':<12} {'available':>10} {'packed':>10} {'used':>10} "
        f"{'share':>7} {'loss_w':>7}")
    for name, avail, packed, used, w in rows:
        share = 100.0 * used / total if total else 0.0
        log(f"[otter-mix] {name:<12} {avail:>10} {packed:>10} {used:>10} "
            f"{share:>6.1f}% {w:>7.2f}")
    log(f"[otter-mix] {'TOTAL':<12} {'':>10} {'':>10} {total:>10}")
    return records, source_names, image_folders
