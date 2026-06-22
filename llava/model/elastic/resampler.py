"""Nested query resampler + nested projector (pooling-free token reduction).

NestedQueryResampler: a learned bank of ``num_queries`` query tokens cross-
attends to the encoder patch features. The output is one token per query, so
the *token count* is controlled by how many queries we use. Crucially the
queries are MATRYOSHKA: using the first ``n_k`` queries is a valid coarse
representation, trained via nested dropout + prefix-sufficiency loss. This
replaces M3's avg-pool with a learned, ordered reduction.

Two mechanisms enforce Matryoshka query ordering:
  1. ``query_pos_embed``: learned per-query positional encodings that nudge
     different positions to specialize on different content; importance order
     emerges from gradient flow when the prefix-KL + CE losses are computed
     at every truncation level.
  2. Engine-level nested dropout (engine.py): randomly truncates to n_tok
     queries each step so each prefix must be independently sufficient.

``patch_pos_embed``: learned spatial positional encodings added to the ViT
patch KV inputs so the cross-attention can exploit the 2-D grid structure.
CLIP already bakes in its own positional encoding, but re-adding a small
learned one here lets the resampler fine-tune spatial routing to task.

NestedProjector: maps resampler features into the LLM embedding space.
When ``widths`` is not None, fc2 is treated as a Matryoshka linear: the
active output is the FIRST w rows of fc2's weight matrix (not a post-hoc
mask). This gives proper nested gradient flow — only fc2.weight[:w, :] is
trained for width w — mirroring the Matryoshka Representation Learning
formulation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class NestedQueryResampler(nn.Module):
    def __init__(self, dim, num_queries, num_patches=576, n_heads=8, depth=2,
                 use_pos_embed: bool = False):
        """
        Args:
            dim:           hidden dim (must match ViT output dim, e.g. 1024 for CLIP-L)
            num_queries:   maximum number of query tokens (= tok_levels[0])
            num_patches:   maximum ViT patch tokens; sizes patch_pos_embed when
                           use_pos_embed=True.  Sliced to actual P at runtime.
            n_heads:       attention heads (dim must be divisible)
            depth:         number of cross-attention + FFN layers
            use_pos_embed: add learned spatial (patch) and ordering (query)
                           positional encodings.  Marginally helpful for spatial
                           fine-tuning tasks; not needed for caption-style pretrain.
                           Default False to keep the baseline lean.
        """
        super().__init__()
        self.dim = dim
        self.num_queries = num_queries
        self.use_pos_embed = use_pos_embed

        # Content queries — one learned vector per possible query slot.
        self.queries = nn.Parameter(torch.randn(num_queries, dim) * 0.02)

        if use_pos_embed:
            # Per-query positional encoding nudges different positions to
            # specialise on different importance levels, giving the optimizer
            # an explicit symmetry-breaking signal for Matryoshka ordering.
            self.query_pos_embed = nn.Parameter(torch.zeros(num_queries, dim))
            nn.init.trunc_normal_(self.query_pos_embed, std=0.02)

            # Per-patch spatial encoding added to KV.  CLIP already encodes
            # position internally; this small learned residual lets the
            # cross-attention re-tune spatial routing to the downstream task.
            self.patch_pos_embed = nn.Parameter(torch.zeros(1, num_patches, dim))
            nn.init.trunc_normal_(self.patch_pos_embed, std=0.02)

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
        """
        Args:
            image_features: (N, P, dim)  — raw ViT patch features
            n_tok:          number of output query tokens; defaults to num_queries
        Returns:
            (N, n_tok, dim)
        """
        N, P, _ = image_features.shape
        n_tok = n_tok or self.num_queries
        assert n_tok <= self.num_queries

        if self.use_pos_embed:
            kv = image_features + self.patch_pos_embed[:, :P, :]
            q = (self.queries[:n_tok] + self.query_pos_embed[:n_tok])
        else:
            kv = image_features
            q = self.queries[:n_tok]
        q = q.unsqueeze(0).expand(N, -1, -1).contiguous()

        for layer in self.layers:
            qn = layer["ln_q"](q)
            kvn = layer["ln_kv"](kv)
            attn_out, _ = layer["cross_attn"](qn, kvn, kvn)
            q = q + attn_out
            q = q + layer["ff"](layer["ln_ff"](q))
        return self.out_ln(q)


class NestedProjector(nn.Module):
    """vision_dim → llm_dim MLP with optional Matryoshka nested output width.

    When ``widths`` is None this is an ordinary 2-layer LLaVA MLP projector.

    When ``widths`` is a list of increasing output widths, fc2 is treated as a
    Matryoshka linear layer: for width w we compute

        out_active = x @ fc2.weight[:w, :].T + fc2.bias[:w]

    and zero-pad to llm_dim.  Critically this means gradient flows ONLY through
    fc2.weight[:w, :] for this level — the correct nested-subspace behaviour.
    Post-hoc masking (x * mask) would pass gradient through ALL of fc2 and
    break the hierarchical subspace property.
    """

    def __init__(self, vision_dim, llm_dim, widths=None, hidden_mult=1):
        super().__init__()
        self.llm_dim = llm_dim
        self.widths = widths
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
        if self.widths is not None:
            w = self.widths[self.level]
            if w < self.llm_dim:
                # Matryoshka linear: use the first w rows of fc2's weight matrix.
                # Gradient is confined to fc2.weight[:w, :] — proper nested training.
                bias = self.fc2.bias[:w] if self.fc2.bias is not None else None
                out_w = F.linear(x, self.fc2.weight[:w, :], bias)
                # Zero-pad inactive dimensions so LLM input dim stays fixed.
                pad = torch.zeros(*x.shape[:-1], self.llm_dim - w,
                                  dtype=x.dtype, device=x.device)
                x = torch.cat([out_w, pad], dim=-1)
            else:
                x = self.fc2(x)
        else:
            x = self.fc2(x)
        return x
