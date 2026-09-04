"""Configuration for the Elastic / Adaptive Matryoshka VLM.

Design note (why there is no standalone L_enc compute axis):
    Nested LoRA on a *frozen* full-width ViT does NOT save vision FLOPs -- the
    whole backbone still runs -- and it does not change the token count either.
    So a separate "encoder compute level" was a non-axis. We removed it.

There is ONE elasticity axis the user actually pays for:

    L_tok  -- number of visual tokens fed to the LLM (LLM prefill / KV cost)

Token reduction method is selectable so the original M3 (pooling) path and the
new pooling-free nested-query method live in one codebase:

    token_reduction = "pooling"       -> original M3 (avg-pool patch grid)
    token_reduction = "nested_query"  -> new method (nested query resampler)

LoRA is OFF by default. When enabled it is a *capacity/specialization* tool, not
a compute knob:

    use_lora=True, lora_specialize_tok=False -> one shared adapter (task adapt)
    use_lora=True, lora_specialize_tok=True  -> adapter tied to L_tok, so each
        token budget gets encoder features tuned for that granularity (coarse
        budgets favor global/semantic content; fine budgets favor detail).
"""

from dataclasses import dataclass, field
from typing import List, Literal, Optional


@dataclass
class ElasticConfig:
    # ---- mode switch ---------------------------------------------------
    token_reduction: Literal["pooling", "nested_query"] = "pooling"

    # ---- L_tok axis (the only compute axis) ----------------------------
    # "pooling": square token counts (M3 scales), e.g. [576, 144, 36, 9, 1].
    # "nested_query": query-bank prefix lengths, e.g. [256, 144, 64, 16].
    tok_levels: List[int] = field(default_factory=lambda: [576, 144, 36, 9, 1])
    num_query_tokens: int = 256          # full query bank size (nested_query)
    train_tok_levels: Optional[List[int]] = None  # indices into tok_levels

    # ---- LoRA (capacity / specialization, NOT a compute axis) ----------
    use_lora: bool = True
    lora_specialize_tok: bool = True     # tie adapter level to L_tok
    # If specializing: one rank per tok level (len == len(tok_levels)).
    # If not: a single shared rank -> use lora_ranks[-1].
    lora_ranks: List[int] = field(default_factory=lambda: [8, 16, 32, 64])
    lora_alpha: float = 1.0
    lora_dropout: float = 0.0

    # ---- projector output scale ----------------------------------------
    # The projector's fc2 output is unconstrained; measured std ~0.54 against an
    # LLM token-embedding std of ~0.015 (36x). With a frozen backbone that only
    # skews the visual positions; with the backbone UNFROZEN (Stage-2 full
    # finetune) it is a live divergence risk -- run 26531 blew up at epoch 0.46
    # with gradient clipping correctly enabled. Enabling this adds a LayerNorm on
    # the projector output whose gain is calibrated to the embedding std at
    # attach time. Default False so existing checkpoints load unchanged.
    projector_out_norm: bool = False

    # ---- loss configuration (nested_query method) ----------------------
    # Positional encodings in the resampler. Measured 2026-08-04 (job 26568):
    # with use_pos_embed=False the 256 resampler outputs collapse to an effective
    # rank of ~12 (TinyLlama) / ~21 (7B), mean pairwise cosine +0.91 / +0.72,
    # while the query PARAMETERS stay near-orthogonal (rank ~235). Spatially
    # anonymous queries all converge on one global summary -- which is why a
    # 16-token prefix loses nothing versus 256.
    use_pos_embed: bool = False
    # "learned"  -> trainable encodings
    # "sincos2d" -> frozen 2-D sine-cosine grid shared by queries and patches,
    #               the MQT-LLaVA design (buffers, never in the optimizer)
    pos_embed_type: str = "learned"
    # Resampler architecture.
    #   "query"        -- the original: a pure learned query bank cross-attending
    #                     to patches. Every run through v7 used this.
    #   "pool_anchored" -- PARCEL-style (arXiv 2605.30126) division of labour:
    #                     part of the budget is a deterministic average-pooled
    #                     spatial grid carrying low-frequency layout, the rest is
    #                     learned queries that are first made "pool-aware" via
    #                     self-attention with the anchors, then cross-attend to
    #                     the raw patches for the complementary detail. Motivated
    #                     by PARCEL's finding that query-only compression
    #                     "forces the queries to encode both the low-frequency
    #                     layout and fine-grained semantic details without an
    #                     underlying spatial anchor" -- our exact failure mode.
    resampler_arch: str = "query"
    # budget -> number of pooled anchor tokens, for resampler_arch="pool_anchored".
    # Must be a perfect square that divides the patch grid evenly (for a 24x24
    # CLIP-L/14-336 grid: 1, 4, 9, 16, 36, 64, 144, 576). None -> derive ~25% of
    # the budget, snapped down to the nearest valid grid. Anchor count MUST be
    # monotone in budget: PARCEL's ablation shows a fixed anchor resolution
    # across budgets costs ~5 points at the top of the range.
    anchor_routing: Optional[dict] = None
    # How the resampler picks its n_tok output tokens out of the query bank.
    # DEFAULT "prefix": the original, only-ever-run behaviour (queries[:n_tok],
    # content-agnostic). See NestedQueryResampler's docstring (resampler.py)
    # for "magnitude" / "attn_energy" / "learned" -- untested alternatives that
    # rank the full query bank by an importance criterion instead of slicing by
    # fixed position. Not wired into any launcher script; opt in explicitly.
    query_selection: str = "prefix"
    # Where the KD target comes from.
    #   "self"  -- the SAME model at tok_levels[kl_teacher_tok_level] (256).
    #              Self-distillation across token budgets, no second model.
    #              Measured 2026-08: KL ~0.006 (0.1% of the loss at weight 0.1),
    #              because teacher and student are identical weights seeing
    #              informationally equivalent inputs -- nothing to distill.
    #   "llava" -- a frozen external LLaVA-1.5-7B. A genuinely stronger teacher,
    #              so the KL carries real signal. Every level becomes a student,
    #              including the largest. Costs ~14 GB/GPU for the frozen 7B.
    teacher: str = "self"
    teacher_model_path: str = "liuhaotian/llava-v1.5-7b"

    use_prefix_kl: bool = True           # coarse-to-fine self-distillation
    prefix_kl_weight: float = 1.0
    # Random truncation -> intended to induce prefix ordering (see
    # NestedQueryResampler's docstring, resampler.py). Default False because
    # every run through v5 has passed --use_nested_dropout False in both
    # launcher scripts; this default now matches what has actually been run,
    # it does not reflect a judgment that the mechanism doesn't help -- it is
    # untested with vision_lora_enable also on and remains a candidate
    # explanation for the flat token-budget result. Opt in explicitly.
    use_nested_dropout: bool = False
    use_coral_align: bool = True         # latent-stability alignment
    coral_weight: float = 0.1
    use_recon: bool = False
    recon_weight: float = 0.1
    use_token_decorrelation: bool = False
    decorr_weight: float = 0.01
    kl_teacher_tok_level: int = 0        # index into tok_levels (highest = full)
    n_sample_students: int = 0           # 0 = full grid; k>0 = teacher + k random students per step
    log_adapter_every: int = 0           # >0: log LoRA adapter divergence every N steps

    # ---- derived -------------------------------------------------------
    def tok_grid(self) -> List[int]:
        n = len(self.tok_levels)
        return self.train_tok_levels if self.train_tok_levels is not None else list(range(n))

    def lora_level_for_tok(self, l_tok: int) -> int:
        """Map a tok level to a LoRA rank index."""
        if not self.use_lora:
            return -1
        if self.lora_specialize_tok:
            return min(l_tok, len(self.lora_ranks) - 1)
        return len(self.lora_ranks) - 1   # shared adapter

    @property
    def is_pooling(self) -> bool:
        return self.token_reduction == "pooling"

    @property
    def is_nested_query(self) -> bool:
        return self.token_reduction == "nested_query"
