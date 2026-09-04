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


#: Anchor counts reachable by integer average-pooling of a square patch grid.
#: For CLIP-L/14-336 (24x24 = 576 patches) these are the squares of 24's divisors.
def valid_anchor_counts(num_patches: int) -> list:
    side = int(round(num_patches ** 0.5))
    if side * side != num_patches:
        return []
    return sorted((side // k) ** 2 for k in range(1, side + 1) if side % k == 0)


def pool_anchors(image_features: torch.Tensor, n_anchor: int) -> torch.Tensor:
    """Average-pool a square patch grid down to `n_anchor` grid-aligned tokens.

    (N, P, C) -> (N, n_anchor, C). These are PARCEL's "spatial pool tokens":
    deterministic, grid-aligned, carrying the low-frequency layout that learned
    queries demonstrably fail to preserve on their own. Not learned, no
    parameters -- the whole point is that this half of the budget cannot drift.
    """
    N, P, C = image_features.shape
    side = int(round(P ** 0.5))
    tgt = int(round(n_anchor ** 0.5))
    if side * side != P or tgt * tgt != n_anchor:
        raise ValueError(f"pool_anchors needs square grids, got P={P} n_anchor={n_anchor}")
    if side % tgt != 0:
        raise ValueError(
            f"anchor grid {tgt}x{tgt} does not evenly divide the {side}x{side} patch "
            f"grid; valid anchor counts are {valid_anchor_counts(P)}")
    if tgt == side:
        return image_features
    k = side // tgt
    x = image_features.view(N, side, side, C).permute(0, 3, 1, 2)   # (N, C, side, side)
    x = F.avg_pool2d(x, kernel_size=k, stride=k)                    # (N, C, tgt, tgt)
    return x.permute(0, 2, 3, 1).reshape(N, tgt * tgt, C)


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
    #: Valid `query_selection` modes. "prefix" is the only one used by any run
    #: through v5 -- it is the exact original behaviour (queries[:n_tok]) and
    #: is checked first in forward() so that path is untouched line-for-line.
    QUERY_SELECTION_MODES = ("prefix", "magnitude", "attn_energy", "learned")

    def __init__(self, dim, num_queries, num_patches=576, n_heads=8, depth=2,
                 use_pos_embed: bool = False, pos_embed_type: str = "learned",
                 query_selection: str = "prefix",
                 resampler_arch: str = "query", anchor_routing=None):
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
            query_selection: how the n_tok output tokens are chosen out of the
                            num_queries-slot bank. DEFAULT "prefix" is the
                            original, only-ever-run behaviour: a fixed,
                            content-agnostic slice queries[:n_tok] (see forward()
                            below -- that branch is byte-identical to the
                            pre-2026-08-31 implementation). The other three are
                            untested alternatives added to probe whether static
                            positional slicing -- rather than actual visual
                            content -- is why the token-budget axis has produced
                            ~0 accuracy delta in every run so far:
                              "magnitude"   -- run the FULL query bank, keep the
                                  n_tok output tokens with the largest L2 norm.
                                  Cheapest, no new params; norm is a weak proxy
                                  for "useful to the LLM."
                              "attn_energy" -- run the FULL query bank, keep the
                                  n_tok tokens that pulled the most total
                                  cross-attention mass from the patches on the
                                  LAST layer. No new params; a self-contained
                                  stand-in for CLIP's own CLS-attention saliency
                                  (which would need new plumbing out of the
                                  vision tower -- this does not).
                              "learned"     -- a small linear head scores each
                                  of the num_queries outputs; keep the top n_tok.
                                  Hard top-k blocks gradient into the score
                                  itself (argsort/topk is not differentiable),
                                  so the selected outputs are rescaled by
                                  sigmoid(score) to give the head SOME training
                                  signal -- an approximation, not a rigorous
                                  differentiable top-k (no Gumbel/straight-through
                                  estimator here).
                            All three non-default modes preserve the one property
                            that actually matters for FLOP reduction -- only
                            n_tok tokens ever reach the projector/LLM -- unlike
                            zero-padding the projector's output width, which
                            keeps every channel dense and saves nothing.
        """
        super().__init__()
        if query_selection not in self.QUERY_SELECTION_MODES:
            raise ValueError(f"query_selection={query_selection!r}; expected one "
                             f"of {self.QUERY_SELECTION_MODES}")
        self.dim = dim
        self.num_queries = num_queries
        self.use_pos_embed = use_pos_embed
        self.pos_embed_type = pos_embed_type
        self.query_selection = query_selection
        if query_selection == "learned":
            self.importance_head = nn.Linear(dim, 1)

        # ---- PARCEL-style pooled-anchor branch (resampler_arch="pool_anchored")
        self.resampler_arch = resampler_arch
        self.num_patches = num_patches
        self.anchor_routing = dict(anchor_routing) if anchor_routing else None
        if resampler_arch not in ("query", "pool_anchored"):
            raise ValueError(f"resampler_arch={resampler_arch!r}; expected "
                             f"'query' or 'pool_anchored'")
        if resampler_arch == "pool_anchored":
            valid = valid_anchor_counts(num_patches)
            if self.anchor_routing:
                for b, npch in sorted(self.anchor_routing.items()):
                    if npch not in valid:
                        raise ValueError(
                            f"anchor_routing[{b}]={npch} is not reachable by integer "
                            f"pooling of a {num_patches}-patch grid; valid: {valid}")
                    if npch > b:
                        raise ValueError(
                            f"anchor_routing[{b}]={npch} exceeds the budget {b}")
            # Query <-> Pool self-attention: the step that makes queries "pool-aware"
            # so they spend themselves on what pooling DISCARDS rather than
            # re-encoding layout the anchors already carry. PARCEL's ablation:
            # running this before the ViT cross-attention beats going straight to
            # cross-attention (95.6 vs 95.2 retention @256 tokens).
            self.pool_self_attn = nn.ModuleDict({
                "attn": nn.MultiheadAttention(dim, n_heads, batch_first=True),
                "ln_in": nn.LayerNorm(dim),
                "ln_ff": nn.LayerNorm(dim),
                "ff": nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(),
                                    nn.Linear(dim * 4, dim)),
            })

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

    def n_anchors_for(self, budget: int, num_patches: int) -> int:
        """How many of `budget` tokens are pooled spatial anchors.

        Uses anchor_routing when the exact budget is declared. Otherwise falls
        back to ~25% of the budget snapped DOWN to the nearest reachable grid --
        needed because nested dropout hands us arbitrary budgets that no routing
        table can enumerate. Always leaves at least one query token so the
        pooled branch never degenerates into plain M3 average pooling.
        """
        valid = valid_anchor_counts(num_patches)
        if self.anchor_routing and budget in self.anchor_routing:
            n_p = self.anchor_routing[budget]
        else:
            target = max(1, budget // 4)
            n_p = max((v for v in valid if v <= target), default=1)
        # Never consume the whole budget: keep >=1 query so the "division of
        # labour" this architecture exists for actually happens.
        while n_p >= budget and n_p > 1:
            n_p = max((v for v in valid if v < n_p), default=1)
        return min(n_p, max(budget - 1, 1))

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

        if self.resampler_arch == "pool_anchored":
            return self._forward_pool_anchored(image_features, n_tok)

        assert n_tok <= self.num_queries

        if self.query_selection == "prefix":
            # ---- DEFAULT / the only path any run through v5 uses ----------
            # Unchanged from the original implementation: a fixed slice of the
            # query bank, one cross-attention pass, exactly n_tok tokens out.
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

        # ---- content-adaptive modes: run the FULL query bank, then keep the
        # n_tok outputs an importance criterion ranks highest. Nested-by-
        # construction as long as the ranking itself does not depend on n_tok
        # (it doesn't -- the score is computed once, from the full-bank pass).
        if self.use_pos_embed:
            kv = image_features + self.patch_pos_embed[:, :P, :]
            q = self.queries + self.query_pos_embed
        else:
            kv = image_features
            q = self.queries
        q = q.unsqueeze(0).expand(N, -1, -1).contiguous()

        need_weights = (self.query_selection == "attn_energy")
        attn_weights = None
        for layer in self.layers:
            qn = layer["ln_q"](q)
            kvn = layer["ln_kv"](kv)
            if need_weights:
                # average_attn_weights=True averages over heads -> (N, Q, P).
                # Overwritten each layer on purpose: only the LAST layer's
                # weights reflect where the (already-updated) queries actually
                # ended up attending, which is the more relevant saliency signal.
                attn_out, attn_weights = layer["cross_attn"](
                    qn, kvn, kvn, need_weights=True, average_attn_weights=True)
            else:
                attn_out, _ = layer["cross_attn"](qn, kvn, kvn)
            q = q + attn_out
            q = q + layer["ff"](layer["ln_ff"](q))
        full_out = self.out_ln(q)                     # (N, num_queries, dim)

        if n_tok == self.num_queries:
            return full_out

        if self.query_selection == "magnitude":
            score = full_out.norm(dim=-1)              # (N, Q) -- L2 norm per token
        elif self.query_selection == "attn_energy":
            score = attn_weights.sum(dim=-1)            # (N, Q) -- total attn mass pulled from patches
        elif self.query_selection == "learned":
            score = self.importance_head(full_out).squeeze(-1)   # (N, Q)
        else:
            raise ValueError(f"unreachable query_selection={self.query_selection!r}")

        idx = score.topk(n_tok, dim=1).indices
        idx, _ = idx.sort(dim=1)                        # keep bank order for interpretability
        gathered = torch.gather(full_out, 1, idx.unsqueeze(-1).expand(-1, -1, full_out.shape[-1]))

        if self.query_selection == "learned":
            # Hard top-k is not differentiable w.r.t. WHICH indices were
            # chosen, so this is the only gradient path into importance_head:
            # scale the kept tokens by their own score. Approximate, not a
            # rigorous differentiable top-k -- see the constructor docstring.
            gate = torch.gather(score, 1, idx)
            gathered = gathered * torch.sigmoid(gate).unsqueeze(-1)

        return gathered

    # ------------------------------------------------------------------
    def _forward_pool_anchored(self, image_features: torch.Tensor,
                               n_tok: int) -> torch.Tensor:
        """PARCEL-style Pool-Conditioned Query Resampling.

        budget B  ->  N_p pooled spatial anchors  +  N_q learned queries

          1. anchors = avg_pool(patches)                 low-frequency layout,
                                                          deterministic, no params
          2. [anchors; queries] -> self-attention        queries become pool-aware
          3. Q_PA -> cross-attend(raw patches)           queries fetch the detail
                                                          pooling threw away
          4. output = [anchors'; Q_SE]                   exactly B tokens

        The anchors emitted are the post-self-attention ones: the block updates
        both streams, and letting the anchors see the queries costs nothing while
        keeping them contextualised. Their *spatial* content is still pinned by
        construction, which is the property queries alone could not hold.
        """
        N, P, _ = image_features.shape
        n_p = self.n_anchors_for(n_tok, P)
        n_q = n_tok - n_p
        if n_q > self.num_queries:
            raise ValueError(
                f"budget {n_tok} needs {n_q} query tokens but the bank holds "
                f"{self.num_queries}; raise num_query_tokens (= tok_levels[0])")

        anchors = pool_anchors(image_features, n_p)          # (N, n_p, C)

        if n_q <= 0:                                          # pure-anchor budget
            return self.out_ln(anchors)

        if self.use_pos_embed:
            kv = image_features + self.patch_pos_embed[:, :P, :]
            q = self.queries[:n_q] + self.query_pos_embed[:n_q]
        else:
            kv = image_features
            q = self.queries[:n_q]
        q = q.unsqueeze(0).expand(N, -1, -1).contiguous()

        # ---- 2. Query <-> Pool self-attention -------------------------------
        joint = torch.cat([anchors, q], dim=1)                # (N, n_p + n_q, C)
        jn = self.pool_self_attn["ln_in"](joint)
        attn_out, _ = self.pool_self_attn["attn"](jn, jn, jn)
        joint = joint + attn_out
        joint = joint + self.pool_self_attn["ff"](self.pool_self_attn["ln_ff"](joint))
        anchors_out, q = joint[:, :n_p], joint[:, n_p:]

        # ---- 3. Semantic-explorer cross-attention (existing stack) ----------
        for layer in self.layers:
            qn = layer["ln_q"](q)
            kvn = layer["ln_kv"](kv)
            attn_out, _ = layer["cross_attn"](qn, kvn, kvn)
            q = q + attn_out
            q = q + layer["ff"](layer["ln_ff"](q))

        out = torch.cat([anchors_out, q], dim=1)              # (N, n_tok, C)
        return self.out_ln(out)


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
