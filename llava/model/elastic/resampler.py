"""Nested query resampler + nested projector (pooling-free token reduction).

NestedQueryResampler: a learned bank of ``num_queries`` query tokens cross-
attends to the encoder patch features. The output is one token per query, so
the *token count* is controlled by how many queries we use. Crucially the
queries are MATRYOSHKA: using the first ``n_k`` queries is a valid coarse
representation, trained via nested dropout + prefix-sufficiency loss. This
replaces M3's avg-pool with a learned, ordered reduction.

NestedProjector: maps resampler/encoder features into the LLM embedding space,
with FlexViT-style nested output width tied to the encoder level L_enc.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class NestedQueryResampler(nn.Module):
    def __init__(self, dim, num_queries, llm_dim=None, n_heads=8, depth=2):
        super().__init__()
        self.dim = dim
        self.num_queries = num_queries
        self.queries = nn.Parameter(torch.randn(num_queries, dim) * 0.02)
        self.layers = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "cross_attn": nn.MultiheadAttention(dim, n_heads, batch_first=True),
                        "ln_q": nn.LayerNorm(dim),
                        "ln_kv": nn.LayerNorm(dim),
                        "ln_ff": nn.LayerNorm(dim),
                        "ff": nn.Sequential(
                            nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim)
                        ),
                    }
                )
                for _ in range(depth)
            ]
        )
        self.out_ln = nn.LayerNorm(dim)

    def forward(self, image_features: torch.Tensor, n_tok: int = None) -> torch.Tensor:
        """image_features: (N, P, dim). Returns (N, n_tok, dim)."""
        N = image_features.shape[0]
        n_tok = n_tok or self.num_queries
        assert n_tok <= self.num_queries
        q = self.queries[:n_tok].unsqueeze(0).expand(N, -1, -1).contiguous()
        kv = image_features
        for layer in self.layers:
            qn = layer["ln_q"](q)
            kvn = layer["ln_kv"](kv)
            attn_out, _ = layer["cross_attn"](qn, kvn, kvn)
            q = q + attn_out
            q = q + layer["ff"](layer["ln_ff"](q))
        return self.out_ln(q)


class NestedProjector(nn.Module):
    """vision_dim -> llm_dim MLP whose hidden/output width can be sliced per
    L_enc level (FlexViT nested Linear). If ``widths`` is None it behaves like a
    standard full-width LLaVA MLP projector."""

    def __init__(self, vision_dim, llm_dim, widths=None, hidden_mult=1):
        super().__init__()
        self.llm_dim = llm_dim
        self.widths = widths  # nested list of llm_dim sub-widths, or None
        hidden = llm_dim * hidden_mult
        self.fc1 = nn.Linear(vision_dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, llm_dim)
        self.level = (len(widths) - 1) if widths else 0

    def set_level(self, level: int) -> None:
        if self.widths is not None:
            assert 0 <= level < len(self.widths)
            self.level = level

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.fc1(x))
        x = self.fc2(x)
        if self.widths is not None:
            w = self.widths[self.level]
            # Slice active sub-width, zero-pad the rest so LLM dim stays fixed.
            if w < self.llm_dim:
                mask = torch.zeros_like(x)
                mask[..., :w] = 1.0
                x = x * mask
        return x
