import torch
import torch.nn as nn

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

    @torch.no_grad()
    def forward(self, images):
        if type(images) is list:
            image_features = []
            for image in images:
                out = self.vision_tower(
                    image.to(device=self.device, dtype=self.dtype).unsqueeze(0),
                    output_hidden_states=True,
                )
                image_features.append(self.feature_select(out).to(image.dtype))
        else:
            out = self.vision_tower(
                images.to(device=self.device, dtype=self.dtype),
                output_hidden_states=True,
            )
            image_features = self.feature_select(out).to(images.dtype)
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
