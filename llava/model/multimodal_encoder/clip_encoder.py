import torch
import torch.nn as nn
import torch.utils.checkpoint as _checkpoint

from transformers import CLIPVisionModel, CLIPImageProcessor, CLIPVisionConfig


class CLIPVisionTower(nn.Module):
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
            self.cfg_only = CLIPVisionConfig.from_pretrained(self.vision_tower_name)

    def load_model(self, device_map=None):
        if self.is_loaded:
            print('{} is already loaded, `load_model` called again, skipping.'.format(self.vision_tower_name))
            return

        self.image_processor = CLIPImageProcessor.from_pretrained(self.vision_tower_name)
        self.vision_tower = CLIPVisionModel.from_pretrained(self.vision_tower_name, device_map=device_map)
        self.vision_tower.requires_grad_(False)

        self.is_loaded = True

    def feature_select(self, image_forward_outs):
        image_features = image_forward_outs.hidden_states[self.select_layer]
        if self.select_feature == 'patch':
            image_features = image_features[:, 1:]
        elif self.select_feature == 'cls_patch':
            image_features = image_features
        else:
            raise ValueError(f'Unexpected select feature: {self.select_feature}')
        return image_features

    def _encode(self, images, l_enc):
        # `l_enc` is (re-)applied here, as the first statement of this
        # function, instead of relying on the caller having mutated shared
        # LoRA-level state before calling forward(). HF's internal
        # per-CLIPEncoderLayer gradient checkpointing (enabled transitively
        # by model.gradient_checkpointing_enable() on the outer LLM, which
        # does reach this nested CLIPVisionModel) recomputes activations
        # during backward; if the level were only set externally beforehand,
        # a later-processed level's mutation would already have overwritten
        # it by the time an earlier level's checkpoint gets recomputed,
        # silently training with the wrong LoRA rank. Setting it here, inside
        # the (optionally outer-checkpointed) function body, means every
        # recompute -- including HF's own nested per-layer recompute inside
        # self.vision_tower -- sees the correct level, since this function
        # always runs top-to-bottom before delegating into the inner model.
        if l_enc is not None and hasattr(self, "set_level"):
            self.set_level(l_enc)
        image_forward_outs = self.vision_tower(images, output_hidden_states=True)
        return self.feature_select(image_forward_outs)

    def forward(self, images, l_enc=None):
        # No blanket @torch.no_grad() here: the base CLIP backbone is frozen
        # (requires_grad_(False) in load_model), so when no LoRA is injected
        # autograd builds no graph through this module regardless (nothing in
        # the subgraph requires grad) -- same effective behavior as no_grad,
        # at zero extra cost. But when nested LoRA *is* injected (elastic
        # engine attached with use_lora=True), its lora_A/lora_B need a real
        # graph to receive gradients; no_grad here was silently preventing
        # that in every run so far.
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
        if self.is_loaded:
            return self.vision_tower.config
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        return self.config.hidden_size

    @property
    def num_patches_per_side(self):
        return self.config.image_size // self.config.patch_size

    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2



class CLIPVisionTowerS2(CLIPVisionTower):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__(vision_tower, args, delay_load)

        self.s2_scales = getattr(args, 's2_scales', '336,672,1008')
        self.s2_scales = list(map(int, self.s2_scales.split(',')))
        self.s2_scales.sort()
        self.s2_split_size = self.s2_scales[0]
        self.s2_image_size = self.s2_scales[-1]

        try:
            from s2wrapper import forward as multiscale_forward
        except ImportError:
            raise ImportError('Package s2wrapper not found! Please install by running: \npip install git+https://github.com/bfshi/scaling_on_scales.git')
        self.multiscale_forward = multiscale_forward

        # change resize/crop size in preprocessing to the largest image size in s2_scale
        if not delay_load or getattr(args, 'unfreeze_mm_vision_tower', False):
            self.image_processor.size['shortest_edge'] = self.s2_image_size
            self.image_processor.crop_size['height'] = self.image_processor.crop_size['width'] = self.s2_image_size

    def load_model(self, device_map=None):
        if self.is_loaded:
            print('{} is already loaded, `load_model` called again, skipping.'.format(self.vision_tower_name))
            return

        self.image_processor = CLIPImageProcessor.from_pretrained(self.vision_tower_name)
        self.vision_tower = CLIPVisionModel.from_pretrained(self.vision_tower_name, device_map=device_map)
        self.vision_tower.requires_grad_(False)

        self.image_processor.size['shortest_edge'] = self.s2_image_size
        self.image_processor.crop_size['height'] = self.image_processor.crop_size['width'] = self.s2_image_size

        self.is_loaded = True

    @torch.no_grad()
    def forward_feature(self, images):
        image_forward_outs = self.vision_tower(images.to(device=self.device, dtype=self.dtype), output_hidden_states=True)
        image_features = self.feature_select(image_forward_outs).to(images.dtype)
        return image_features

    @torch.no_grad()
    def forward(self, images, l_enc=None):
        # S2 multi-scale mode is not wired up for nested-LoRA specialization
        # in this codebase (no training script here passes s2_scales); accept
        # and ignore l_enc purely so callers can pass it unconditionally.
        if type(images) is list:
            image_features = []
            for image in images:
                image_feature = self.multiscale_forward(self.forward_feature, image.unsqueeze(0), img_sizes=self.s2_scales, max_split_size=self.s2_split_size)
                image_features.append(image_feature)
        else:
            image_features = self.multiscale_forward(self.forward_feature, images, img_sizes=self.s2_scales, max_split_size=self.s2_split_size)

        return image_features

    @property
    def hidden_size(self):
        return self.config.hidden_size * len(self.s2_scales)
