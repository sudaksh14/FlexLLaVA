"""Unit tests for the elastic modules. Run in an env with torch:

    cd matryoshka-mm && python -m pytest llava/model/elastic/tests/test_elastic.py -q

These avoid downloading HF backbones.
"""

import torch
import torch.nn as nn

from llava.model.elastic.config import ElasticConfig
from llava.model.elastic.nested_lora import NestedLoRALinear, inject_nested_lora
from llava.model.elastic.resampler import NestedQueryResampler, NestedProjector
from llava.model.elastic import losses
from llava.model.elastic.flops import grid_cost


def test_nested_lora_levels_and_frozen_base():
    base = nn.Linear(32, 48)
    lora = NestedLoRALinear(base, ranks=[4, 8, 16], alpha=1.0)
    x = torch.randn(2, 5, 32)
    assert torch.allclose(lora(x), base(x), atol=1e-6)  # B init zero -> identity
    assert not base.weight.requires_grad
    assert lora.lora_A.requires_grad and lora.lora_B.requires_grad
    for lvl, r in enumerate([4, 8, 16]):
        lora.set_level(lvl)
        assert lora.current_rank == r
        assert lora(x).shape == (2, 5, 48)


def test_inject_lora_broadcast():
    block = nn.Module()
    block.q_proj = nn.Linear(16, 16)
    block.fc1 = nn.Linear(16, 16)
    block.other = nn.Linear(16, 16)
    wraps = inject_nested_lora(block, {"q_proj", "fc1"}, [2, 4])
    assert len(wraps) == 2
    assert isinstance(block.q_proj, NestedLoRALinear)
    assert not isinstance(block.other, NestedLoRALinear)


def test_nested_query_resampler_prefix():
    res = NestedQueryResampler(dim=64, num_queries=128, n_heads=8, depth=2)
    feats = torch.randn(3, 200, 64)
    assert res(feats, n_tok=128).shape == (3, 128, 64)
    assert res(feats, n_tok=16).shape == (3, 16, 64)


def test_projector_fullwidth():
    proj = NestedProjector(vision_dim=64, llm_dim=32, widths=None)
    x = torch.randn(2, 10, 64)
    assert proj(x).shape == (2, 10, 32)


def test_losses_run():
    s = torch.randn(2, 6, 100)
    t = torch.randn(2, 6, 100)
    labels = torch.randint(0, 100, (2, 6))
    assert losses.prefix_kl_loss(s, t, labels).ndim == 0
    a = torch.randn(2, 16, 32)
    b = torch.randn(2, 64, 32)
    assert losses.coral_loss(a, b).ndim == 0
    assert losses.decorrelation_loss(a).ndim == 0


def test_config_lora_specialization_mapping():
    # LoRA on by default (required by the approach)
    cfg = ElasticConfig()
    assert cfg.use_lora is True and cfg.lora_specialize_tok is True
    # explicitly off
    cfg = ElasticConfig(use_lora=False)
    assert cfg.lora_level_for_tok(2) == -1
    # shared adapter
    cfg = ElasticConfig(use_lora=True, lora_specialize_tok=False, lora_ranks=[8, 16, 32])
    assert cfg.lora_level_for_tok(0) == 2  # always max (shared)
    # specialized per tok budget
    cfg = ElasticConfig(use_lora=True, lora_specialize_tok=True,
                        tok_levels=[256, 144, 64, 16], lora_ranks=[8, 16, 32, 64])
    assert cfg.lora_level_for_tok(0) == 0
    assert cfg.lora_level_for_tok(3) == 3


def test_flops_grid_monotonic_in_tokens():
    cfg = ElasticConfig(token_reduction="nested_query", tok_levels=[256, 64, 16])
    cost = grid_cost(cfg)
    assert cost[0] > cost[1] > cost[2]  # more tokens -> more LLM flops
