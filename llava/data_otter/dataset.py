"""Otter-style mixture dataset + collator for FlexLLaVA.

This is a parallel data pipeline.  It does NOT modify or replace
`llava/train/train.py`'s LazySupervisedDataset -- that one stays exactly as it
is and keeps serving the running jobs.  What this adds on top:

  * a YAML-declared, per-source mixture with resampling      (mixture.py, #2)
  * multi-turn packing of QA pairs that share an image       (packing.py, #3)
  * a source tag on every sample, so loss can be attributed  (telemetry, #7)
  * deterministic handling of unreadable images              (manifest.py, #6)
  * token-accurate lengths for the length-grouped sampler    (lengths.py, #8)

The per-sample text/image processing itself is IMPORTED from
`llava.train.train`, never reimplemented.  preprocess_v1 / preprocess_mpt carry
a lot of hard-won, backbone-specific round-accounting (the `_auto_prefix_len`
work for BPE tokenizers, the prefix-tokenisation rewrite for Phi-3.5/SmolLM2);
forking that code would guarantee the two copies drift and would put every one
of those fixes at risk.
"""

from __future__ import annotations

import copy
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import torch
from PIL import Image
from torch.utils.data import Dataset

from llava.constants import IGNORE_INDEX

from . import lengths as _lengths
from . import manifest as _manifest
from .mixture import MixtureConfig, TEXT_ONLY_SOURCE, build_mixture, load_mixture_config

# Set by llava/train/train_otter.py before it calls train().  Holds everything
# the data module needs that HfArgumentParser does not carry.
OTTER_DATA_CONFIG: Optional["OtterDataConfig"] = None


@dataclass
class OtterDataConfig:
    mixture_config: str
    cache_dir: Optional[str] = None
    build_caches: bool = False        # build length/manifest caches inline if missing
    use_manifest: bool = True
    visual_tokens: int = 256          # tok_levels[0]; the teacher's visual prefix
    log_every: int = 25               # optimizer steps between telemetry flushes


def _expand2square(pil_img, background_color):
    """Pad to a square, matching LazySupervisedDataset's image_aspect_ratio='pad'."""
    width, height = pil_img.size
    if width == height:
        return pil_img
    if width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    result = Image.new(pil_img.mode, (height, height), background_color)
    result.paste(pil_img, ((height - width) // 2, 0))
    return result


class OtterMixtureDataset(Dataset):
    """Supervised fine-tuning dataset over a YAML-declared mixture."""

    def __init__(self, tokenizer, data_args, cfg: OtterDataConfig, log=print):
        super().__init__()
        self.tokenizer = tokenizer
        self.data_args = data_args
        self.cfg = cfg
        self.log = log

        mix: MixtureConfig = load_mixture_config(cfg.mixture_config)
        self.mixture = mix
        records, source_names, image_folders = build_mixture(mix, log=log)

        # Fall back to the CLI --image_folder for any source that declared none.
        default_folder = getattr(data_args, "image_folder", None)
        self.image_folders = {
            k: (v or default_folder) for k, v in image_folders.items()
        }

        if cfg.use_manifest:
            missing = _manifest.load_missing(cfg.cache_dir, mix.signature(), log=log)
            if missing is None and cfg.build_caches:
                found = _manifest.scan_missing(records, source_names, self.image_folders, log=log)
                if cfg.cache_dir:
                    _manifest.save_missing(cfg.cache_dir, mix.signature(), found, log=log)
                missing = set(found)
            if missing:
                records, source_names = _manifest.drop_missing(
                    records, source_names, missing, log=log)
            elif missing is None:
                log("[otter-data] NOTE: no image manifest; unreadable images will be handled "
                    "at runtime by deterministic next-index fallback. Build one with "
                    "`python -m llava.data_otter.prepare --build manifest ...`")

        self.records = records
        self.source_names = source_names
        self.sources = sorted(set(source_names))

        lens, provenance = _lengths.load_or_build(
            records, tokenizer,
            visual_tokens=cfg.visual_tokens,
            mixture_signature=mix.signature(),
            cache_dir=cfg.cache_dir,
            build_if_missing=cfg.build_caches,
            log=log,
        )
        self._token_lengths = lens
        self.length_provenance = provenance
        log(f"[otter-data] {len(self.records)} samples across {len(self.sources)} sources; "
            f"lengths from {provenance}")

    # ---- length-grouped sampler interface (consumed by LLaVATrainer) --------
    @property
    def lengths(self) -> List[int]:
        return self._token_lengths

    @property
    def modality_lengths(self) -> List[int]:
        """Positive for image samples, negative for text-only.

        Sign is the modality flag LengthGroupedSampler splits on; magnitude is
        the true token length rather than a word count.
        """
        out = []
        for rec, n in zip(self.records, self._token_lengths):
            out.append(n if rec.get("image") else -max(n, 1))
        return out

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, i: int) -> Dict[str, Any]:
        # Deterministic fallback, unlike the random-resample in
        # LazySupervisedDataset: walking to the next index gives every run the
        # same substitution for the same failure, so a mixture ablation stays
        # reproducible. The manifest prescan should make this path rare.
        n = len(self.records)
        for attempt in range(10):
            idx = (i + attempt) % n
            try:
                item = self._get_item(idx)
                item["image_fallback"] = 1 if attempt else 0
                return item
            except (FileNotFoundError, OSError) as e:
                if attempt == 0:
                    self.log(f"[otter-data] unreadable sample {idx} "
                             f"({self.records[idx].get('image')}): {e}; advancing to next index")
        raise RuntimeError(f"10 consecutive unreadable samples starting at {i}")

    def _get_item(self, i: int) -> Dict[str, Any]:
        # Imported lazily so `import llava.data_otter.dataset` stays cheap for
        # the verification gate, which does not need the whole training module.
        from llava.train.train import preprocess, preprocess_multimodal

        record = self.records[i]
        source = self.source_names[i]
        sources = [record["conversations"]]
        has_image = bool(record.get("image"))
        image = None

        if has_image:
            processor = self.data_args.image_processor
            folder = self.image_folders.get(source) or ""
            image = Image.open(os.path.join(folder, record["image"])).convert("RGB")
            if self.data_args.image_aspect_ratio == "pad":
                image = _expand2square(image, tuple(int(x * 255) for x in processor.image_mean))
            image = processor.preprocess(image, return_tensors="pt")["pixel_values"][0]
            sources = preprocess_multimodal(copy.deepcopy(sources), self.data_args)
        else:
            sources = copy.deepcopy(sources)

        data_dict = preprocess(sources, self.tokenizer, has_image=has_image)
        out = dict(input_ids=data_dict["input_ids"][0], labels=data_dict["labels"][0])

        if image is not None:
            out["image"] = image
        elif getattr(self.data_args, "is_multimodal", False):
            crop = self.data_args.image_processor.crop_size
            out["image"] = torch.zeros(3, crop["height"], crop["width"])

        out["source"] = source
        out["n_packed"] = int(record.get("n_packed", 1))
        return out


@dataclass
class DataCollatorForOtterMixture:
    """Collate, and carry the per-sample bookkeeping the telemetry needs.

    Identical tensor handling to DataCollatorForSupervisedDataset; the extra
    keys ("sources", "n_packed", ...) are non-tensor and are popped by
    OtterTrainer.compute_loss before the batch reaches the model.
    """

    tokenizer: Any

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, Any]:
        input_ids, labels = tuple([instance[key] for instance in instances]
                                  for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX)
        input_ids = input_ids[:, :self.tokenizer.model_max_length]
        labels = labels[:, :self.tokenizer.model_max_length]

        batch: Dict[str, Any] = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

        if "image" in instances[0]:
            images = [instance["image"] for instance in instances]
            if all(x is not None and x.shape == images[0].shape for x in images):
                batch["images"] = torch.stack(images)
            else:
                batch["images"] = images

        # ---- telemetry payload (popped before the model sees the batch) ----
        n_supervised = int((labels != IGNORE_INDEX).sum().item())
        batch["otter_meta"] = {
            "sources": [inst["source"] for inst in instances],
            "n_packed": [inst.get("n_packed", 1) for inst in instances],
            "image_fallbacks": sum(inst.get("image_fallback", 0) for inst in instances),
            "supervised_tokens": n_supervised,
            "padded_tokens": int((input_ids == self.tokenizer.pad_token_id).sum().item()),
            # Text sequence length: an image is still a single
            # IMAGE_TOKEN_INDEX placeholder here, expanded to tok_levels[0]
            # visual tokens later inside the model.
            "seq_len": int(input_ids.shape[1]),
            "batch_size": len(instances),
        }
        return batch


def make_otter_data_module(tokenizer, data_args) -> Dict[str, Any]:
    """Drop-in replacement for train.make_supervised_data_module.

    train_otter.py assigns this over `llava.train.train.make_supervised_data_module`
    so the unmodified train() picks it up; the signature and return keys must
    therefore match exactly.
    """
    from llava.train.train import rank0_print

    if OTTER_DATA_CONFIG is None:
        raise RuntimeError(
            "OTTER_DATA_CONFIG was never set. Launch through llava/train/train_otter.py, "
            "not by calling train() directly.")
    train_dataset = OtterMixtureDataset(
        tokenizer=tokenizer, data_args=data_args, cfg=OTTER_DATA_CONFIG, log=rank0_print)
    return dict(train_dataset=train_dataset,
                eval_dataset=None,
                data_collator=DataCollatorForOtterMixture(tokenizer=tokenizer))
