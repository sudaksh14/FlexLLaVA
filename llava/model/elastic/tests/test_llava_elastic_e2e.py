"""Full end-to-end smoke test for the llava_llama elastic training loop.

Mirrors the *exact* math in LlavaLlamaForCausalLM.forward (elastic branch) with
toy tensors — no HuggingFace weights, no CLIP, no LLaMA. Validates:

  1. Image-token splicing (IMAGE_TOKEN_INDEX stitching, IGNORE_INDEX masking)
  2. CE loss per tok level, prefix-KL, CORAL aggregation
  3. Backward pass: grads reach LoRA A/B, resampler queries, projector fc1/fc2
  4. Frozen base weights accumulate NO grad
  5. LoRA specialization actually routes to distinct rank sub-matrices per level
  6. Both token_reduction modes: "pooling" and "nested_query"
  7. KL / CORAL terms are skipped for the teacher level (l_tok == kl_teacher_tok_level)
  8. Optimizer step does not NaN any parameter

Run:
    cd matryoshka-mm
    python -m llava.model.elastic.tests.test_llava_elastic_e2e
or:
    pytest llava/model/elastic/tests/test_llava_elastic_e2e.py -v
"""

import math
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from llava.model.elastic.config import ElasticConfig
from llava.model.elastic.nested_lora import NestedLoRALinear, inject_nested_lora
from llava.model.elastic.engine import ElasticEngine
from llava.model.elastic import losses as _el

# ── constants matching llava/constants.py ──────────────────────────────────────
IMAGE_TOKEN_INDEX = -200
IGNORE_INDEX = -100


# ══════════════════════════════════════════════════════════════════════════════
# Toy modules — zero HF dependencies
# ══════════════════════════════════════════════════════════════════════════════

class TinyFrozenViT(nn.Module):
    """Single frozen Linear + nested LoRA, emits (N, P, C) patch features.

    Mirrors the CLIP vision tower: the base weights are frozen, nested LoRA is
    injected, and set_level() routes the adapter rank.
    """
    def __init__(self, C: int, ranks, P: int = 64):
        super().__init__()
        base = nn.Linear(C, C, bias=False)
        for p in base.parameters():
            p.requires_grad_(False)                     # freeze base
        self.proj = NestedLoRALinear(base, ranks)
        self.P = P                                      # #patches per image
        self.hidden_size = C                            # for engine

    def set_level(self, lvl: int):
        self.proj.set_level(lvl)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (N, P, C) -> (N, P, C)"""
        return self.proj(x)


class TinyCausalLayer(nn.Module):
    """Single causal self-attention + FFN block (no KV cache needed here)."""
    def __init__(self, d: int, n_heads: int = 2):
        super().__init__()
        self.attn = nn.MultiheadAttention(d, n_heads, batch_first=True)
        self.ff   = nn.Sequential(nn.Linear(d, d * 2), nn.GELU(), nn.Linear(d * 2, d))
        self.ln1  = nn.LayerNorm(d)
        self.ln2  = nn.LayerNorm(d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        L = x.shape[1]
        mask = torch.triu(torch.ones(L, L, device=x.device, dtype=torch.bool), diagonal=1)
        h, _ = self.attn(x, x, x, attn_mask=mask, need_weights=False)
        x = self.ln1(x + h)
        return self.ln2(x + self.ff(x))


class TinyCausalLM(nn.Module):
    """2-layer causal transformer + lm_head (vocab).

    Mirrors the Llama backbone + lm_head used in forward_single_matryoshka.
    All weights are trainable (not frozen) so grad-flow checks are meaningful.
    """
    def __init__(self, d: int, vocab: int, embed_dim: int = None):
        super().__init__()
        embed_dim = embed_dim or d
        self.embed = nn.Embedding(vocab, embed_dim)
        self.proj_in = nn.Linear(embed_dim, d) if embed_dim != d else nn.Identity()
        self.layers = nn.Sequential(TinyCausalLayer(d), TinyCausalLayer(d))
        self.lm_head = nn.Linear(d, vocab, bias=False)

    def embed_tokens(self, ids: torch.Tensor) -> torch.Tensor:
        """Match LlavaMetaForCausalLM.get_model().embed_tokens()"""
        emb = self.embed(ids)
        return self.proj_in(emb) if not isinstance(self.proj_in, nn.Identity) else emb

    def forward(self, inputs_embeds: torch.Tensor,
                attention_mask: torch.Tensor = None,
                labels: torch.Tensor = None):
        """inputs_embeds: (N, L, d).  Returns (loss, logits)."""
        h = self.layers(inputs_embeds)
        logits = self.lm_head(h).float()    # (N, L, V) — matches llava_llama
        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1].contiguous().view(-1, logits.shape[-1])
            shift_labels = labels[:, 1:].contiguous().view(-1).to(shift_logits.device)
            loss = F.cross_entropy(shift_logits, shift_labels, ignore_index=IGNORE_INDEX)
        return loss, logits


# ══════════════════════════════════════════════════════════════════════════════
# Image-token splicing (mirrors prepare_inputs_labels_for_multimodal)
# ══════════════════════════════════════════════════════════════════════════════

def toy_splice_images(
    input_ids: torch.Tensor,       # (N, L_text)  — contains IMAGE_TOKEN_INDEX
    labels: torch.Tensor,          # (N, L_text)
    image_features: torch.Tensor,  # (N, n_k, d)  — projected visual tokens
    lm: TinyCausalLM,
    d: int,
) -> tuple:
    """Replicate the sequence-construction logic from
    prepare_inputs_labels_for_multimodal (flat merge, single image per sample).

    Returns:
        inputs_embeds : (N, L_text - 1 + n_k, d)   — right-padded
        new_labels    : (N, L_text - 1 + n_k)       — IGNORE_INDEX at image positions
        attn_mask     : (N, L_text - 1 + n_k) bool
    """
    N, n_k = image_features.shape[:2]
    all_embeds, all_labels = [], []
    for i in range(N):
        ids_i   = input_ids[i]
        labs_i  = labels[i]
        img_pos = (ids_i == IMAGE_TOKEN_INDEX).nonzero(as_tuple=True)[0]
        assert img_pos.numel() == 1, "expect exactly one image token per sample"
        p = img_pos.item()

        # text before / after image token
        before_ids  = ids_i[:p]
        after_ids   = ids_i[p + 1:]
        before_labs = labs_i[:p]
        after_labs  = labs_i[p + 1:]

        before_emb = lm.embed_tokens(before_ids)    # (p, d)
        after_emb  = lm.embed_tokens(after_ids)     # (L-p-1, d)

        # image features replace the single IMAGE_TOKEN_INDEX slot
        img_emb  = image_features[i]                            # (n_k, d)
        img_labs = torch.full((n_k,), IGNORE_INDEX,
                              dtype=labs_i.dtype, device=labs_i.device)

        seq_emb  = torch.cat([before_emb, img_emb, after_emb],  dim=0)
        seq_labs = torch.cat([before_labs, img_labs, after_labs], dim=0)
        all_embeds.append(seq_emb)
        all_labels.append(seq_labs)

    # right-pad to equal length
    max_len = max(e.shape[0] for e in all_embeds)
    inputs_embeds = torch.zeros(N, max_len, d, device=image_features.device)
    new_labels    = torch.full((N, max_len), IGNORE_INDEX,
                               dtype=labels.dtype, device=labels.device)
    attn_mask     = torch.zeros(N, max_len, dtype=torch.bool, device=image_features.device)
    for i, (emb, lab) in enumerate(zip(all_embeds, all_labels)):
        L = emb.shape[0]
        inputs_embeds[i, :L] = emb
        new_labels[i, :L]    = lab
        attn_mask[i, :L]     = True
    return inputs_embeds, new_labels, attn_mask


# ══════════════════════════════════════════════════════════════════════════════
# Core loop — mirrors llava_llama.forward (elastic branch)
# ══════════════════════════════════════════════════════════════════════════════

def run_elastic_loop(token_reduction: str, use_nested_dropout: bool = True,
                     use_coral: bool = True, use_kl: bool = True,
                     seed: int = 0):
    """End-to-end elastic loop with all optional auxiliary losses."""
    torch.manual_seed(seed)

    # ── hyper-params ──────────────────────────────────────────────────────────
    C    = 32     # vision dim (CLIP hidden)
    d    = 48     # LLM dim (Llama hidden)
    V    = 200    # toy vocab
    N    = 2      # batch size
    P    = 64     # patches per image  (must be square: 8x8)
    Ltxt = 10     # text tokens per sample (including one IMAGE_TOKEN_INDEX)

    lora_ranks   = [4, 8, 16]
    if token_reduction == "pooling":
        # must produce perfect squares after sqrt(P / scale)
        tok_levels = [64, 16, 4]
    else:
        tok_levels = [32, 16, 8]

    cfg = ElasticConfig(
        token_reduction=token_reduction,
        tok_levels=tok_levels,
        num_query_tokens=tok_levels[0],   # full bank = largest level
        use_lora=True,
        lora_specialize_tok=True,
        lora_ranks=lora_ranks,
        lora_alpha=1.0,
        use_prefix_kl=use_kl,
        prefix_kl_weight=1.0,
        use_coral_align=use_coral,
        coral_weight=0.1,
        use_nested_dropout=use_nested_dropout,
        kl_teacher_tok_level=0,           # largest tok level is teacher
    )

    # ── build components ──────────────────────────────────────────────────────
    vit    = TinyFrozenViT(C, lora_ranks, P=P)
    lm     = TinyCausalLM(d, V)
    engine = ElasticEngine(cfg, vit, vision_dim=C, llm_dim=d)
    if engine.resampler is not None:
        engine.resampler.train()
    engine.projector.train()
    lm.train()
    vit.train()                           # LoRA wrappers need train mode

    # ── toy batch ─────────────────────────────────────────────────────────────
    # input_ids: random text with one IMAGE_TOKEN_INDEX at position 3
    input_ids = torch.randint(0, V, (N, Ltxt))
    input_ids[:, 3] = IMAGE_TOKEN_INDEX  # image placeholder

    # labels: supervise text positions only; image positions will be IGNORE_INDEX
    labels_raw = input_ids.clone()
    labels_raw[labels_raw == IMAGE_TOKEN_INDEX] = IGNORE_INDEX

    # raw patch features (what the ViT would emit before projection)
    images = torch.randn(N, P, C)

    # ── elastic training loop (exact replica of llava_llama.forward) ──────────
    loss            = torch.zeros(())
    teacher_logits  = None
    teacher_tokens  = None
    grid            = list(engine.grid())
    seen_ranks      = []
    kl_computed     = []
    coral_computed  = []

    for l_tok in grid:
        # 1. set LoRA level (specialisation)
        engine._set_lora_level(l_tok)
        seen_ranks.append(vit.proj.current_rank)

        # 2. encode: ViT (frozen base + active LoRA) -> raw patch features
        feats = vit(images)                              # (N, P, C)

        # 3. token reduction + project to LLM dim
        vis_tokens = engine.reduce_tokens(feats, l_tok) # (N, n_k, d)

        # 4. splice image tokens into text sequence
        inputs_embeds, new_labels, attn_mask = toy_splice_images(
            input_ids, labels_raw, vis_tokens, lm, d)

        # 5. LM forward — mirrors forward_single_matryoshka's model() + lm_head
        loss_item, logits = lm(inputs_embeds, attn_mask, new_labels)

        # accumulate CE loss (averaged across grid)
        loss = loss + loss_item / len(grid)

        cur_tokens = engine.last_tokens              # (N, n_k, d)
        if l_tok == cfg.kl_teacher_tok_level:
            teacher_logits = logits.detach()
            teacher_tokens = cur_tokens.detach()
        else:
            # prefix-KL: align this level to teacher over the shorter prefix
            if cfg.use_prefix_kl and teacher_logits is not None:
                n = min(teacher_logits.shape[1], logits.shape[1])
                kl = _el.prefix_kl_loss(
                    logits[:, :n],
                    teacher_logits[:, :n],
                    new_labels[:, :n],
                )
                loss = loss + cfg.prefix_kl_weight * kl / len(grid)
                kl_computed.append(l_tok)

            # CORAL: align second-order token statistics to teacher
            if cfg.use_coral_align and teacher_tokens is not None and cur_tokens is not None:
                coral = _el.coral_loss(cur_tokens, teacher_tokens)
                loss = loss + cfg.coral_weight * coral / len(grid)
                coral_computed.append(l_tok)

    # ── sanity: loss finite ───────────────────────────────────────────────────
    assert torch.isfinite(loss), f"[{token_reduction}] loss is not finite: {loss}"

    # ── backward ──────────────────────────────────────────────────────────────
    loss.backward()

    # ── gradient checks ───────────────────────────────────────────────────────
    # a) LoRA adapters got grads
    assert vit.proj.lora_A.grad is not None, "lora_A has no grad"
    assert vit.proj.lora_B.grad is not None, "lora_B has no grad"
    assert torch.any(vit.proj.lora_A.grad != 0), "lora_A grad is all-zero"

    # b) frozen base weights: NO grad at all (requires_grad=False -> grad stays None)
    assert vit.proj.base.weight.grad is None, \
        "frozen base weight accumulated a gradient — freezing is broken!"

    # c) projector (engine.projector) trained
    assert engine.projector.fc1.weight.grad is not None, "projector fc1 no grad"
    assert engine.projector.fc2.weight.grad is not None, "projector fc2 no grad"

    # d) resampler queries (nested_query mode only)
    if engine.resampler is not None:
        assert engine.resampler.queries.grad is not None, "resampler queries no grad"
        assert torch.any(engine.resampler.queries.grad != 0), \
            "resampler queries grad is all-zero"

    # e) LM backbone trained
    assert lm.layers[0].attn.in_proj_weight.grad is not None, "LM attn no grad"
    assert lm.lm_head.weight.grad is not None, "lm_head no grad"

    # ── specialization: each tok level used a different LoRA rank ─────────────
    expected_ranks = lora_ranks[:len(grid)]
    assert seen_ranks == expected_ranks, \
        f"rank routing wrong: got {seen_ranks}, want {expected_ranks}"

    # ── KL / CORAL computed for exactly the non-teacher levels ────────────────
    non_teacher = [l for l in grid if l != cfg.kl_teacher_tok_level]
    if use_kl:
        assert sorted(kl_computed) == sorted(non_teacher), \
            f"prefix-KL not applied to all non-teacher levels: {kl_computed}"
    if use_coral:
        assert sorted(coral_computed) == sorted(non_teacher), \
            f"CORAL not applied to all non-teacher levels: {coral_computed}"

    # ── optimizer step: no NaN in params ─────────────────────────────────────
    opt = torch.optim.SGD(
        list(vit.parameters()) +
        list(engine.projector.parameters()) +
        (list(engine.resampler.parameters()) if engine.resampler else []) +
        list(lm.parameters()),
        lr=1e-3,
    )
    opt.step()
    for name, p in [
        ("lora_A", vit.proj.lora_A),
        ("lora_B", vit.proj.lora_B),
        ("proj.fc1", engine.projector.fc1.weight),
        ("lm_head", lm.lm_head.weight),
    ]:
        assert torch.all(torch.isfinite(p)), f"NaN/Inf in {name} after opt step"

    n_kl    = len(kl_computed)
    n_coral = len(coral_computed)
    print(f"  [{token_reduction}] loss={loss.item():.4f}  ranks={seen_ranks}"
          f"  kl_terms={n_kl}  coral_terms={n_coral}  OK")
    return loss.item()


# ══════════════════════════════════════════════════════════════════════════════
# Additional targeted unit tests
# ══════════════════════════════════════════════════════════════════════════════

def test_image_splicing():
    """Verify toy_splice_images stitches seq correctly and masks image labels."""
    torch.manual_seed(1)
    C, d, V, N = 8, 8, 50, 2
    lm = TinyCausalLM(d, V, embed_dim=d)
    image_features = torch.randn(N, 4, d)
    input_ids = torch.randint(1, V, (N, 6))
    input_ids[:, 2] = IMAGE_TOKEN_INDEX
    labels = input_ids.clone()

    embeds, new_labels, mask = toy_splice_images(input_ids, labels, image_features, lm, d)

    # shape: 6 - 1 (dropped img tok) + 4 (img feats) = 9
    assert embeds.shape == (N, 9, d), f"bad shape {embeds.shape}"
    assert new_labels.shape == (N, 9)
    # image positions 2..5 must be IGNORE_INDEX
    assert (new_labels[:, 2:6] == IGNORE_INDEX).all(), "image labels not masked"
    # text positions after the image should not all be IGNORE_INDEX
    assert not (new_labels[:, 6:] == IGNORE_INDEX).all(), "text after image is masked"
    print("  [splice] image-token splice and label masking: OK")


def test_lora_rank_routing():
    """Each engine._set_lora_level(l) must activate a distinct rank sub-matrix."""
    torch.manual_seed(2)
    C, ranks = 16, [2, 4, 8]
    vit = TinyFrozenViT(C, ranks)
    cfg = ElasticConfig(
        token_reduction="pooling", tok_levels=[64, 16, 4],
        use_lora=True, lora_specialize_tok=True, lora_ranks=ranks,
    )
    engine = ElasticEngine(cfg, vit, vision_dim=C, llm_dim=16)
    seen = []
    for l_tok in engine.grid():
        engine._set_lora_level(l_tok)
        seen.append(vit.proj.current_rank)
    assert seen == ranks, f"rank routing: {seen}"
    print("  [lora]   rank routing across levels: OK")


def test_frozen_base_no_grad():
    """NestedLoRALinear must not accumulate grad on the frozen base weight."""
    torch.manual_seed(3)
    C, ranks = 16, [2, 4, 8]
    base = nn.Linear(C, C, bias=False)
    for p in base.parameters():
        p.requires_grad_(False)
    lora = NestedLoRALinear(base, ranks)
    x = torch.randn(4, C)
    loss = lora(x).sum()
    loss.backward()
    assert base.weight.grad is None, "frozen base got grad"
    assert lora.lora_A.grad is not None
    print("  [freeze] frozen base weight got no grad, LoRA adapter did: OK")


def test_coral_loss_shape_invariance():
    """coral_loss must accept different token counts (student vs teacher)."""
    torch.manual_seed(4)
    d = 32
    teacher = torch.randn(2, 16, d)
    student = torch.randn(2, 8, d)
    c = _el.coral_loss(student, teacher)
    assert c.shape == (), f"coral scalar shape: {c.shape}"
    assert torch.isfinite(c)
    print("  [coral]  CORAL loss shape-invariance (8 vs 16 tokens): OK")


def test_prefix_kl_masked():
    """prefix_kl_loss with IGNORE_INDEX labels should not NaN."""
    torch.manual_seed(5)
    B, L, V = 2, 10, 50
    s_logits = torch.randn(B, L, V)
    t_logits = torch.randn(B, L, V)
    labels = torch.randint(0, V, (B, L))
    labels[:, 4:6] = IGNORE_INDEX       # some ignored positions
    kl = _el.prefix_kl_loss(s_logits, t_logits, labels)
    assert torch.isfinite(kl), "prefix_kl NaN with IGNORE_INDEX labels"
    print("  [kl]     prefix_kl_loss with IGNORE_INDEX mask: OK")


def test_no_auxiliary_losses():
    """Loop should work when KL and CORAL are both disabled."""
    loss = run_elastic_loop("pooling", use_nested_dropout=False,
                            use_coral=False, use_kl=False, seed=6)
    assert math.isfinite(loss)


def test_pooling_mode():
    loss = run_elastic_loop("pooling", seed=7)
    assert math.isfinite(loss)


def test_nested_query_mode():
    loss = run_elastic_loop("nested_query", seed=8)
    assert math.isfinite(loss)


def test_determinism():
    """Same seed -> same loss (no hidden randomness leaking across tests)."""
    l1 = run_elastic_loop("nested_query", seed=42)
    l2 = run_elastic_loop("nested_query", seed=42)
    assert abs(l1 - l2) < 1e-6, f"non-deterministic: {l1} vs {l2}"


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=== Unit tests ===")
    test_image_splicing()
    test_lora_rank_routing()
    test_frozen_base_no_grad()
    test_coral_loss_shape_invariance()
    test_prefix_kl_masked()

    print("\n=== Elastic loop (pooling) — no aux losses ===")
    test_no_auxiliary_losses()

    print("\n=== Elastic loop (pooling) — full losses ===")
    test_pooling_mode()

    print("\n=== Elastic loop (nested_query) — full losses ===")
    test_nested_query_mode()

    print("\n=== Determinism check ===")
    test_determinism()

    print("\nAll smoke tests passed.")
