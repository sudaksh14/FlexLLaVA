"""Rank-nested LoRA: the FlexViT slicing idea applied to LoRA factors.

A standard LoRA adds ``B @ A`` (rank r) to a frozen Linear. Here the rank is
the *elasticity level*: we keep the maximum rank's factors and, at level k,
use only the first ``r_k`` columns of A and rows of B::

    delta_k(x) = (alpha / r_k) * (x @ A[:, :r_k]) @ B[:r_k, :]

Because the factors are shared and nested, lower levels are sub-models of
higher levels -- exactly the matryoshka property, but on the *adapter* so the
pretrained CLIP/SigLIP weights stay frozen and the latent space is preserved.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class NestedLoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, ranks, alpha: float = 1.0, dropout: float = 0.0):
        super().__init__()
        assert list(ranks) == sorted(ranks), "ranks must be ascending (nested)"
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)

        self.ranks = list(ranks)
        self.max_rank = self.ranks[-1]
        self.alpha = alpha
        self.level = len(self.ranks) - 1  # default: full rank

        in_f, out_f = base.in_features, base.out_features
        # A: (in, max_rank), B: (max_rank, out). Slicing the rank dim nests them.
        self.lora_A = nn.Parameter(torch.zeros(in_f, self.max_rank))
        self.lora_B = nn.Parameter(torch.zeros(self.max_rank, out_f))
        nn.init.kaiming_uniform_(self.lora_A, a=5 ** 0.5)
        nn.init.zeros_(self.lora_B)  # start as identity -> preserves base output
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def set_level(self, level: int) -> None:
        assert 0 <= level < len(self.ranks)
        self.level = level

    @property
    def current_rank(self) -> int:
        return self.ranks[self.level]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        r = self.current_rank
        scale = self.alpha / r
        delta = (self.dropout(x) @ self.lora_A[:, :r]) @ self.lora_B[:r, :]
        return out + scale * delta


def inject_nested_lora(module: nn.Module, target_names, ranks, alpha=1.0, dropout=0.0):
    """Recursively replace child Linear layers whose attribute name is in
    ``target_names`` with NestedLoRALinear. Returns the list of wrappers so a
    parent can broadcast ``set_level`` to all of them."""
    wrappers = []
    for name, child in module.named_children():
        if isinstance(child, nn.Linear) and name in target_names:
            wrapped = NestedLoRALinear(child, ranks, alpha, dropout)
            setattr(module, name, wrapped)
            wrappers.append(wrapped)
        else:
            wrappers.extend(inject_nested_lora(child, target_names, ranks, alpha, dropout))
    return wrappers
