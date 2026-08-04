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


def sincos_2d_pos_embed(dim: int, grid: int) -> torch.Tensor:
    """Fixed 2-D sine-cosine positional embeddings, (grid*grid, dim).

    The MAE / MQT-LLaVA formulation: half the channels encode the row index and
    half the column index, each with the usual geometric frequency ladder.
    """
    if dim % 4 != 0:
        raise ValueError(f"sincos_2d needs dim divisible by 4, got {dim}")
    omega = 1.0 / 10000 ** (torch.arange(dim // 4, dtype=torch.float32) / (dim // 4))
    coords = torch.arange(grid, dtype=torch.float32)
    out = []
    for axis in torch.meshgrid(coords, coords, indexing="ij"):
        ang = axis.reshape(-1)[:, None] * omega[None, :]      # (grid*grid, dim/4)
        out += [torch.sin(ang), torch.cos(ang)]
    return torch.cat(out, dim=1)                              # (grid*grid, dim)


def _interp_pos_embed(pos: torch.Tensor, tgt_len: int) -> torch.Tensor:
    """Bicubic-resample a square positional grid to another square grid.

    Keeps queries and patches in ONE coordinate frame: query (i,j) and patch
    (i,j) then refer to the same place in the image, which is the whole point of
    anchoring queries spatially. Regenerating sincos at the patch grid size
    instead would put the two on different index ranges and break the alignment.
    """
    src = int(round(pos.shape[0] ** 0.5))
    tgt = int(round(tgt_len ** 0.5))
    if src == tgt:
        return pos
    d = pos.shape[1]
    grid = pos.reshape(1, src, src, d).permute(0, 3, 1, 2)
    grid = F.interpolate(grid, size=(tgt, tgt), mode="bicubic", align_corners=False)
    return grid.permute(0, 2, 3, 1).reshape(tgt * tgt, d)


class NestedQueryResampler(nn.Module):
    def __init__(self, dim, num_queries, num_patches=576, n_heads=8, depth=2,
                 use_pos_embed: bool = False, pos_embed_type: str = "learned"):
        """
        Args:
            dim:           hidden dim (must match ViT output dim, e.g. 1024 for CLIP-L)
            num_queries:   maximum number of query tokens (= tok_levels[0])
            num_patches:   maximum ViT patch tokens; sizes patch_pos_embed when
                           use_pos_embed=True.  Sliced to actual P at runtime.
            n_heads:       attention heads (dim must be divisible)
            depth:         number of cross-attention + FFN layers
            use_pos_embed: add spatial (patch) and ordering (query) positional
                           encodings.  Measured 2026-08-04 (job 26568): with this
                           OFF the 256 output tokens collapse to an effective rank
                           of ~12 (TinyLlama) / ~21 (7B) out of 256, mean pairwise
                           cosine +0.91 / +0.72, even though the learned query
                           PARAMETERS stay near-orthogonal (rank ~235).  Spatially
                           anonymous queries all converge on the same global
                           summary, which is why a 16-token prefix loses nothing.
            pos_embed_type: "learned"  -- trainable encodings, small random init.
                            "sincos2d" -- FROZEN 2-D sine-cosine grid shared between
                            queries and patches (the MQT-LLaVA design).  Each query
                            is anchored to a grid position it cannot drift from, so
                            queries cannot collapse onto one another.  Registered as
                            buffers, so they never enter the optimizer.
        """
        super().__init__()
        self.dim = dim
        self.num_queries = num_queries
        self.use_pos_embed = use_pos_embed
        self.pos_embed_type = pos_embed_type

        # Content queries — one learned vector per possible query slot.
        self.queries = nn.Parameter(torch.randn(num_queries, dim) * 0.02)

        if use_pos_embed:
            if pos_embed_type == "sincos2d":
                grid = int(round(num_queries ** 0.5))
                if grid * grid != num_queries:
                    raise ValueError(
                        "pos_embed_type='sincos2d' needs a square query count, "
                        f"got num_queries={num_queries}")
                q_pos = sincos_2d_pos_embed(dim, grid)                 # (Q, dim)
                # Buffers, not Parameters: frozen by construction, still saved in
                # the state dict so warm-start and eval reconstruct them.
                self.register_buffer("query_pos_embed", q_pos)
                self.register_buffer("patch_pos_embed",
                                     _interp_pos_embed(q_pos, num_patches).unsqueeze(0))
            elif pos_embed_type == "learned":
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
            else:
                raise ValueError(f"unknown pos_embed_type={pos_embed_type!r}; "
                                 "expected 'learned' or 'sincos2d'")

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

    def __init__(self, vision_dim, llm_dim, widths=None, hidden_mult=1,
                 out_norm=False):
        super().__init__()
        self.llm_dim = llm_dim
        self.widths = widths
        hidden = llm_dim * hidden_mult
        self.fc1 = nn.Linear(vision_dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, llm_dim)
        self.level = (len(widths) - 1) if widths else 0
        # Optional output normalization. fc2 is unconstrained, so its output std
        # is whatever pretraining happened to land on (~0.54 measured), while the
        # LLM's own token embeddings sit at ~0.015. Feeding a 36x-larger signal
        # into the residual stream is survivable with a frozen backbone (Stage 1)
        # but destabilizes a full finetune (Stage 2). See calibrate_out_norm.
        self.out_norm = nn.LayerNorm(llm_dim) if out_norm else None

    def set_level(self, level: int) -> None:
        if self.widths is not None:
            assert 0 <= level < len(self.widths)
            self.level = level

    @torch.no_grad()
    def calibrate_out_norm(self, target_std: float) -> bool:
        """Set the output LayerNorm gain so projected tokens enter the LLM at the
        same scale as its own token embeddings.

        LayerNorm's default gain of 1.0 would produce unit-std tokens -- ~67x
        LARGER than the embeddings, i.e. worse than no norm at all -- so the gain
        MUST be calibrated. It stays learnable and can grow if the model wants it.

        No-op (returns False) if the gain has already moved away from its 1.0
        init, which means trained values were loaded from a checkpoint and must
        not be clobbered.
        """
        if self.out_norm is None:
            return False
        w = self.out_norm.weight
        if not torch.allclose(w, torch.ones_like(w)):
            return False
        w.fill_(target_std)
        self.out_norm.bias.zero_()
        return True

    def _apply_out_norm(self, x: torch.Tensor, w: int) -> torch.Tensor:
        """Normalize over the ACTIVE width only. Normalizing over the full
        llm_dim would mix the zero-padded tail into the statistics and destroy
        the nested-subspace property."""
        if self.out_norm is None:
            return x
        if w == self.llm_dim:
            return self.out_norm(x)
        return F.layer_norm(x, (w,), self.out_norm.weight[:w],
                            self.out_norm.bias[:w], self.out_norm.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.fc1(x))
        if self.widths is not None:
            w = self.widths[self.level]
            if w < self.llm_dim:
                # Matryoshka linear: use the first w rows of fc2's weight matrix.
                # Gradient is confined to fc2.weight[:w, :] — proper nested training.
                bias = self.fc2.bias[:w] if self.fc2.bias is not None else None
                out_w = self._apply_out_norm(F.linear(x, self.fc2.weight[:w, :], bias), w)
                # Zero-pad inactive dimensions so LLM input dim stays fixed.
                pad = torch.zeros(*x.shape[:-1], self.llm_dim - w,
                                  dtype=x.dtype, device=x.device)
                x = torch.cat([out_w, pad], dim=-1)
            else:
                x = self._apply_out_norm(self.fc2(x), self.llm_dim)
        else:
            x = self._apply_out_norm(self.fc2(x), self.llm_dim)
        return x
