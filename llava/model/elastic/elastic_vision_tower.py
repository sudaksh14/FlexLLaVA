"""Encoder-agnostic elastic vision tower (CLIP + SigLIP).

Wraps a frozen HF vision encoder and (optionally) injects rank-nested LoRA into
its attention/MLP Linear layers, exposing ``set_level(L_enc)``. The only
encoder-specific differences are isolated here:
  * CLIP   : has a CLS token -> patch features are hidden_states[:, 1:]
  * SigLIP : no CLS token     -> all tokens are patch features

This means the rest of the pipeline (resampler, projector, loss) is identical
for both encoders, so they can be benchmarked on the same (L_enc, L_tok) grid.
"""

import torch
import torch.nn as nn

from .nested_lora import inject_nested_lora

# Linear sub-layer names to adapt, by encoder family.
CLIP_LORA_TARGETS = {"q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"}
SIGLIP_LORA_TARGETS = {"q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"}


class ElasticVisionTower(nn.Module):
    family = None  # "clip" | "siglip"

    def __init__(self, vision_tower_name, args, elastic_cfg, delay_load=False):
        super().__init__()
        self.vision_tower_name = vision_tower_name
        self.args = args
        self.cfg = elastic_cfg
        self.select_layer = getattr(args, "mm_vision_select_layer", -2)
        self.select_feature = getattr(args, "mm_vision_select_feature", "patch")
        self.is_loaded = False
        self.lora_wrappers = []
        if not delay_load:
            self.load_model()

    # -- subclass hooks --------------------------------------------------
    def _load_backbone(self):
        raise NotImplementedError

    def _lora_targets(self):
        raise NotImplementedError

    def _strip_cls(self, features):
        """Return patch tokens only."""
        raise NotImplementedError

    # -- shared ----------------------------------------------------------
    def load_model(self, device_map=None):
        if self.is_loaded:
            return
        self._load_backbone(device_map)
        self.vision_tower.requires_grad_(False)
        if self.cfg.has_vision_elasticity:
            self.lora_wrappers = inject_nested_lora(
                self.vision_tower,
                self._lora_targets(),
                self.cfg.lora_ranks,
                alpha=self.cfg.lora_alpha,
                dropout=self.cfg.lora_dropout,
            )
        self.is_loaded = True

    def set_level(self, l_enc: int) -> None:
        for w in self.lora_wrappers:
            w.set_level(l_enc)

    def feature_select(self, outputs):
        feats = outputs.hidden_states[self.select_layer]
        if self.select_feature == "patch":
            feats = self._strip_cls(feats)
        return feats

    def forward(self, images):
        # base frozen; LoRA params (if any) are trainable, so no global no_grad.
        outs = self.vision_tower(
            images.to(device=self.device, dtype=self.dtype),
            output_hidden_states=True,
        )
        return self.feature_select(outs).to(images.dtype)

    @property
    def dtype(self):
        return self.vision_tower.dtype

    @property
    def device(self):
        return self.vision_tower.device

    @property
    def config(self):
        return self.vision_tower.config

    @property
    def hidden_size(self):
        return self.config.hidden_size


class ElasticCLIPTower(ElasticVisionTower):
    family = "clip"

    def _load_backbone(self, device_map=None):
        from transformers import CLIPVisionModel, CLIPImageProcessor
        self.image_processor = CLIPImageProcessor.from_pretrained(self.vision_tower_name)
        self.vision_tower = CLIPVisionModel.from_pretrained(self.vision_tower_name, device_map=device_map)

    def _lora_targets(self):
        return CLIP_LORA_TARGETS

    def _strip_cls(self, features):
        return features[:, 1:]  # drop CLS


class ElasticSigLIPTower(ElasticVisionTower):
    family = "siglip"

    def _load_backbone(self, device_map=None):
        from transformers import SiglipVisionModel, SiglipImageProcessor
        self.image_processor = SiglipImageProcessor.from_pretrained(self.vision_tower_name)
        self.vision_tower = SiglipVisionModel.from_pretrained(self.vision_tower_name, device_map=device_map)

    def _lora_targets(self):
        return SIGLIP_LORA_TARGETS

    def _strip_cls(self, features):
        return features  # SigLIP has no CLS token


def build_elastic_vision_tower(vision_tower_name, args, elastic_cfg, **kwargs):
    name = vision_tower_name.lower()
    if "siglip" in name:
        return ElasticSigLIPTower(vision_tower_name, args, elastic_cfg, **kwargs)
    if "clip" in name:
        return ElasticCLIPTower(vision_tower_name, args, elastic_cfg, **kwargs)
    raise ValueError(f"Unknown encoder family for {vision_tower_name}")
