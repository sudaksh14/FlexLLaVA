"""End-to-end smoke test of the elastic training loop WITHOUT HF weights.

It rebuilds, with toy tensors, exactly the pipeline the real model runs:
    frozen ViT (+nested LoRA) -> resampler -> projector -> tiny LM -> loss
and reproduces the llava_llama.forward elastic branch math (LM CE + prefix-KL
+ CORAL across the L_tok grid). It then checks:
  * loss is finite and backward() works,
  * grads flow to LoRA / resampler / projector,
  * the frozen ViT base weights get NO grad,
  * LoRA specialization actually selects different ranks per token level.

Run:  cd matryoshka-mm && python -m llava.model.elastic.tests.smoke_elastic
"""

import torch
import torch.nn as nn

from llava.model.elastic.config import ElasticConfig
from llava.model.elastic.nested_lora import NestedLoRALinear
from llava.model.elastic.engine import ElasticEngine
from llava.model.elastic import losses as _el


class TinyViT(nn.Module):
    """Frozen linear 'encoder' with a nested-LoRA adapter; emits (N, P, C)."""
    def __init__(self, C, ranks):
        super().__init__()
        base = nn.Linear(C, C)
        self.proj = NestedLoRALinear(base, ranks)
        self.P = 64

    def set_level(self, lvl):
        self.proj.set_level(lvl)

    def forward(self, x):                  # x: (N, P, C)
        return self.proj(x)


class TinyLM(nn.Module):
    """Maps visual tokens to logits over a toy vocab (stand-in for the LLM)."""
    def __init__(self, d, vocab):
        super().__init__()
        self.head = nn.Linear(d, vocab)

    def forward(self, tokens):             # (N, n_k, d) -> (N, n_k, vocab)
        return self.head(tokens).float()


def run(token_reduction):
    torch.manual_seed(0)
    C, d, vocab, N = 32, 48, 100, 2
    cfg = ElasticConfig(
        token_reduction=token_reduction,
        tok_levels=[64, 16, 4] if token_reduction == "pooling" else [64, 32, 8],
        num_query_tokens=64,
        use_lora=True, lora_specialize_tok=True, lora_ranks=[4, 8, 16],
        use_prefix_kl=True, use_coral_align=True, use_nested_dropout=True,
        kl_teacher_tok_level=0,
    )
    vit = TinyViT(C, cfg.lora_ranks)
    for p in vit.proj.base.parameters():
        assert not p.requires_grad           # base frozen
    lm = TinyLM(d, vocab)
    engine = ElasticEngine(cfg, vit, vision_dim=C, llm_dim=d)
    engine.resampler and engine.resampler.train()
    engine.projector.train()

    images = torch.randn(N, vit.P, C)
    labels = torch.randint(0, vocab, (N, max(cfg.tok_levels)))

    # ---- replicate llava_llama elastic branch ----
    loss = 0.0
    teacher_logits = teacher_tokens = None
    grid = list(engine.grid())
    seen_ranks = []
    for l_tok in grid:
        engine._set_lora_level(l_tok)
        seen_ranks.append(vit.proj.current_rank)
        feats = vit(images)                  # re-encode with this level's adapter
        vis_tokens = engine.reduce_tokens(feats, l_tok)   # (N, n_k, d)
        logits = lm(vis_tokens)
        n_k = logits.shape[1]
        ce = nn.functional.cross_entropy(
            logits.reshape(-1, vocab), labels[:, :n_k].reshape(-1))
        loss = loss + ce / len(grid)
        cur = engine.last_tokens
        if l_tok == cfg.kl_teacher_tok_level:
            teacher_logits = logits.detach()
            teacher_tokens = cur.detach()
        else:
            n = min(teacher_logits.shape[1], logits.shape[1])
            loss = loss + cfg.prefix_kl_weight * _el.prefix_kl_loss(
                logits[:, :n], teacher_logits[:, :n], labels[:, :n]) / len(grid)
            loss = loss + cfg.coral_weight * _el.coral_loss(cur, teacher_tokens) / len(grid)

    assert torch.isfinite(loss), "loss not finite"
    loss.backward()

    # gradient checks
    assert vit.proj.lora_A.grad is not None and vit.proj.lora_B.grad is not None
    assert vit.proj.base.weight.grad is None, "frozen base got a grad!"
    assert engine.projector.fc1.weight.grad is not None
    if engine.resampler is not None:
        assert engine.resampler.queries.grad is not None
    # specialization picked distinct ranks per level
    assert seen_ranks == cfg.lora_ranks[:len(grid)], seen_ranks
    print(f"[{token_reduction}] loss={loss.item():.4f}  ranks/level={seen_ranks}  OK")


if __name__ == "__main__":
    run("nested_query")
    run("pooling")
    print("smoke test passed")
