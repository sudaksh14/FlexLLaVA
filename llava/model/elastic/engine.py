"""Glue between the elastic modules and M3's LLaVA forward.

`ElasticEngine` owns the mode branch so the rest of the model code is mode-
agnostic. There is a single elasticity axis: L_tok (number of visual tokens).
LoRA, when enabled, is a capacity/specialization tool tied (optionally) to the
token budget -- it is NOT a separate compute axis.

Integration points in the original repo:

1) llava_arch.matryoshka_vis_token_process
   -> when an engine is attached, delegate to engine.reduce_tokens(feats, l_tok)
      (handles pooling OR nested-query + LoRA specialization + projector).

2) llava_llama.forward (training branch)
   -> iterate engine.grid() (1-D over tok levels), accumulate LM loss + the
      coarse-to-fine prefix-KL term. Pure M3 (no engine) is unchanged.
"""

import random
import torch
import torch.nn.functional as F

from .resampler import NestedQueryResampler, NestedProjector
from . import losses


class ElasticEngine:
    def __init__(self, cfg, vision_tower, vision_dim, llm_dim):
        self.cfg = cfg
        self.vision_tower = vision_tower
        self.resampler = None
        if cfg.is_nested_query:
            self.resampler = NestedQueryResampler(vision_dim, cfg.num_query_tokens)
        # Projector stays full-width (no nesting): width is not an elasticity axis.
        self.projector = NestedProjector(vision_dim, llm_dim, widths=None)
        self.last_tokens = None   # latest projected visual tokens (for coral)

    # -- grid (1-D over token levels) -----------------------------------
    def grid(self):
        for lt in self.cfg.tok_grid():
            yield lt

    def _set_lora_level(self, l_tok):
        """Specialization only: set the encoder adapter for this token budget."""
        if not self.cfg.use_lora:
            return
        lvl = self.cfg.lora_level_for_tok(l_tok)
        if lvl >= 0 and hasattr(self.vision_tower, "set_level"):
            self.vision_tower.set_level(lvl)

    # -- token reduction (the mode branch) ------------------------------
    def reduce_tokens(self, image_features, l_tok):
        """image_features: (N, P, C) -> reduced + projected tokens.
        Note: image_features must be (re)computed by the caller AFTER the LoRA
        level is set, if specialization is on; see encode hook in llava_arch."""
        if self.cfg.is_pooling:
            reduced = self._avg_pool(image_features, self.cfg.tok_levels[l_tok])
        else:
            n_tok = self.cfg.tok_levels[l_tok]
            # nested dropout induces the query ordering -- but never truncate the
            # teacher level, which must stay the full bank for prefix-KL/CORAL.
            if (self.cfg.use_nested_dropout and self.resampler.training
                    and l_tok != self.cfg.kl_teacher_tok_level):
                n_tok = random.randint(1, n_tok)
            reduced = self.resampler(image_features, n_tok=n_tok)
        out = self.projector(reduced)
        self.last_tokens = out    # (N, n_k, llm_dim); grad kept for student level
        return out

    @staticmethod
    def _avg_pool(image_features, scale):
        import numpy as np
        N, HW, C = image_features.shape
        H = W = int(HW ** 0.5)
        x = image_features.view(N, H, W, C).permute(0, 3, 1, 2)
        k = int(np.sqrt(HW / scale))
        x = F.avg_pool2d(x, kernel_size=k, stride=k)
        return x.permute(0, 2, 3, 1).reshape(N, -1, C)

    # -- extra matryoshka losses (nested_query only) --------------------
    def extra_losses(self, student_logits, teacher_logits, labels,
                     student_tokens=None, teacher_tokens=None):
        if not self.cfg.is_nested_query:
            return torch.zeros((), device=student_logits.device)
        total = torch.zeros((), device=student_logits.device)
        if self.cfg.use_coral_align and teacher_tokens is not None:
            total = total + self.cfg.coral_weight * losses.coral_loss(
                student_tokens, teacher_tokens.detach())
        if self.cfg.use_token_decorrelation and student_tokens is not None:
            total = total + self.cfg.decorr_weight * losses.decorrelation_loss(student_tokens)
        return total


def attach_elastic_engine(model, cfg):
    """Attach an ElasticEngine to a loaded LlavaLlamaForCausalLM so the guarded
    branches activate. Omit the call entirely for pure M3.

    If cfg.use_lora, nested-LoRA must already be injected into the vision tower
    (e.g. via an ElasticVisionTower); otherwise the engine just modulates token
    count. Registers resampler/projector as trainable submodules."""
    vt = model.get_vision_tower()
    # If LoRA is requested but the (plain M3) tower can't switch levels, inject
    # rank-nested LoRA into its HF backbone and give it a set_level() method.
    if cfg.use_lora and not hasattr(vt, "set_level"):
        from .nested_lora import inject_nested_lora
        from .elastic_vision_tower import CLIP_LORA_TARGETS
        inner = getattr(vt, "vision_tower", vt)
        wrappers = inject_nested_lora(inner, CLIP_LORA_TARGETS, cfg.lora_ranks,
                                      cfg.lora_alpha, cfg.lora_dropout)
        vt._lora_wrappers = wrappers
        def _set_level(lvl, _w=wrappers):
            for w in _w:
                w.set_level(lvl)
        vt.set_level = _set_level
    engine = ElasticEngine(cfg, vt, vt.hidden_size, model.config.hidden_size)
    if engine.resampler is not None:
        model.add_module("elastic_resampler", engine.resampler)
    model.add_module("elastic_projector", engine.projector)
    model.elastic_engine = engine
    return engine
