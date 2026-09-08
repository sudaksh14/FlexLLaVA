import torch
import torch.nn as nn
import torch.utils.checkpoint as _checkpoint

from transformers import SiglipVisionModel, SiglipImageProcessor, SiglipVisionConfig


class SigLIPVisionTower(nn.Module):
    """Vision tower backed by a SigLIP encoder (e.g. google/siglip-so400m-patch14-384).

    Unlike CLIP, SigLIP has no CLS token — every position in the hidden states
    is a spatial patch token.  feature_select therefore returns all positions
    regardless of the mm_vision_select_feature setting.

    Typical patch counts:
        google/siglip-so400m-patch14-384  →  27×27 = 729 patches, hidden=1152
        google/siglip-so400m-patch14-224  →  16×16 = 256 patches, hidden=1152
        google/siglip-base-patch16-224    →  14×14 = 196 patches, hidden=768
    """

    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__()
        self.is_loaded = False
        self.vision_tower_name = vision_tower
        self.select_layer = args.mm_vision_select_layer
        self.select_feature = getattr(args, 'mm_vision_select_feature', 'patch')

        if not delay_load:
            self.load_model()
        elif getattr(args, 'unfreeze_mm_vision_tower', False):
            print(f'Checkpoint contains `vision_tower` weights: `unfreeze_mm_vision_tower`: True.')
            self.load_model()
        else:
            self.cfg_only = SiglipVisionConfig.from_pretrained(self.vision_tower_name)

    def load_model(self, device_map=None):
        if self.is_loaded:
            print('{} is already loaded, `load_model` called again, skipping.'.format(self.vision_tower_name))
            return
        self.image_processor = SiglipImageProcessor.from_pretrained(self.vision_tower_name)
        self.vision_tower = SiglipVisionModel.from_pretrained(self.vision_tower_name, device_map=device_map)
        self.vision_tower.requires_grad_(False)
        self.is_loaded = True

    def feature_select(self, image_forward_outs):
        # SigLIP has no CLS token — return all patch positions from selected layer.
        return image_forward_outs.hidden_states[self.select_layer]

    def _encode(self, images, l_enc):
        # See CLIPVisionTower._encode for why l_enc is (re-)applied here rather
        # than by the caller mutating shared LoRA-level state beforehand: HF's
        # per-layer gradient checkpointing recomputes activations during
        # backward, and setting the level inside this function body (which
        # always runs top-to-bottom before delegating into self.vision_tower)
        # is what keeps every recompute -- including nested per-layer ones --
        # seeing the correct level. Mirrored here verbatim; this tower had no
        # l_enc handling at all until nested vision LoRA was first exercised
        # against it (2026-09-08), so it never reached this failure mode
        # before -- SigLIP had simply never been used with vision LoRA on.
        if l_enc is not None and hasattr(self, "set_level"):
            self.set_level(l_enc)
        out = self.vision_tower(images, output_hidden_states=True)
        return self.feature_select(out)

    def forward(self, images, l_enc=None):
        # No blanket @torch.no_grad(): matches the CLIP tower's fix (see its
        # own forward()) for the same reason -- the base SigLIP backbone is
        # frozen (requires_grad_(False) in load_model), so no_grad costs
        # nothing when no LoRA is injected, but when nested LoRA IS injected
        # its lora_A/lora_B need a real autograd graph to receive gradients.
        # This tower had @torch.no_grad() unconditionally until now, which
        # would have silently zeroed every vision-LoRA gradient.
        use_checkpoint = (l_enc is not None and hasattr(self, "set_level")
                          and self.training and torch.is_grad_enabled())
        if type(images) is list:
            image_features = []
            for image in images:
                img = image.to(device=self.device, dtype=self.dtype).unsqueeze(0)
                if use_checkpoint:
                    feat = _checkpoint.checkpoint(self._encode, img, l_enc, use_reentrant=False)
                else:
                    feat = self._encode(img, l_enc)
                image_features.append(feat.to(image.dtype))
        else:
            img = images.to(device=self.device, dtype=self.dtype)
            if use_checkpoint:
                image_features = _checkpoint.checkpoint(self._encode, img, l_enc, use_reentrant=False)
            else:
                image_features = self._encode(img, l_enc)
            image_features = image_features.to(images.dtype)
        return image_features

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.vision_tower.dtype

    @property
    def device(self):
        return self.vision_tower.device

    @property
    def config(self):
        return self.vision_tower.config if self.is_loaded else self.cfg_only

    @property
    def hidden_size(self):
        return self.config.hidden_size

    @property
    def num_patches_per_side(self):
        return self.config.image_size // self.config.patch_size

    @property
    def num_patches(self):
        return self.num_patches_per_side ** 2
