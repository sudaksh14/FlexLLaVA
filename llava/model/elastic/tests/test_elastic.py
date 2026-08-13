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


class _FakeTok:
    """Minimal tokenizer stand-in: only what the pad guard touches."""

    def __init__(self, pad, pad_id, eos_id=2, unk="<unk>", unk_id=0):
        self.pad_token, self._pad_id = pad, pad_id
        self.eos_token_id = eos_id
        self.unk_token, self.unk_token_id = unk, unk_id

    @property
    def pad_token_id(self):
        return self._pad_id if self.pad_token is not None else None

    def __setattr__(self, k, v):
        if k == "pad_token" and getattr(self, "unk_token", None) == v and v is not None:
            object.__setattr__(self, "_pad_id", self.unk_token_id)
        object.__setattr__(self, k, v)


def test_pad_guard_replaces_pad_when_it_equals_eos():
    from llava.train.train import ensure_distinct_pad_token

    # TinyLlama-1.1B-Chat ships pad == eos == '</s>' (id 2): the case that
    # silently deleted every EOS from the training labels.
    tok = _FakeTok(pad="</s>", pad_id=2, eos_id=2)
    ensure_distinct_pad_token(tok)
    assert tok.pad_token_id != tok.eos_token_id
    assert tok.pad_token_id == 0          # fell back to unk


def test_pad_guard_is_noop_for_llama_style_tokenizers():
    from llava.train.train import ensure_distinct_pad_token

    # Vicuna/Llama: pad already distinct from eos -> must not be touched,
    # which is what keeps the 7B recipe byte-identical.
    tok = _FakeTok(pad="<unk>", pad_id=0, eos_id=2)
    ensure_distinct_pad_token(tok)
    assert tok.pad_token == "<unk>" and tok.pad_token_id == 0


def test_pad_guard_handles_missing_pad_token():
    from llava.train.train import ensure_distinct_pad_token

    tok = _FakeTok(pad=None, pad_id=None, eos_id=2)
    ensure_distinct_pad_token(tok)
    assert tok.pad_token_id == 0 and tok.pad_token_id != tok.eos_token_id


def test_teacher_defaults_to_self_and_attaches_nothing():
    from llava.model.elastic.engine import attach_kd_teacher

    class _Eng:
        pass
    eng = _Eng()
    assert attach_kd_teacher(eng, ElasticConfig(), student=None) is None
    assert eng.kd_teacher is None          # self-distillation: no second model


def test_teacher_rejects_unknown_mode():
    import pytest
    from llava.model.elastic.engine import attach_kd_teacher

    class _Eng:
        pass
    with pytest.raises(ValueError):
        attach_kd_teacher(_Eng(), ElasticConfig(teacher="gpt4"), student=None)


def test_teacher_config_roundtrips_through_json():
    import dataclasses, json as _json
    cfg = ElasticConfig(teacher="llava", teacher_model_path="liuhaotian/llava-v1.5-7b")
    raw = _json.loads(_json.dumps(dataclasses.asdict(cfg)))
    valid = {f.name for f in dataclasses.fields(ElasticConfig)}
    back = ElasticConfig(**{k: v for k, v in raw.items() if k in valid})
    # eval rebuilds the config from elastic_config.json, so these must survive
    assert back.teacher == "llava"
    assert back.teacher_model_path == "liuhaotian/llava-v1.5-7b"


def test_merge_lora_state_dict_folds_deltas_and_restores_plain_keys():
    from llava.train.train import _merge_lora_state_dict

    torch.manual_seed(0)
    base = torch.randn(8, 6)
    A = torch.randn(2, 6)       # (r, in)
    B = torch.randn(8, 2)       # (out, r)
    sd = {
        "model.layers.0.self_attn.q_proj.base_layer.weight": base.clone(),
        "model.layers.0.self_attn.q_proj.lora_A.default.weight": A.clone(),
        "model.layers.0.self_attn.q_proj.lora_B.default.weight": B.clone(),
        "elastic_projector.fc1.weight": torch.randn(3, 3),   # must pass through
    }
    out = _merge_lora_state_dict(sd, scale=2.0)

    # the plain key from_pretrained looks for now exists, and no PEFT keys remain
    assert "model.layers.0.self_attn.q_proj.weight" in out
    assert not any(".base_layer." in k or ".lora_" in k for k in out)
    assert torch.allclose(out["model.layers.0.self_attn.q_proj.weight"],
                          base + 2.0 * (B @ A), atol=1e-5)
    assert "elastic_projector.fc1.weight" in out


def test_merge_lora_state_dict_is_noop_without_lora():
    from llava.train.train import _merge_lora_state_dict

    sd = {"model.embed_tokens.weight": torch.randn(4, 4),
          "elastic_resampler.queries": torch.randn(8, 4)}
    out = _merge_lora_state_dict(dict(sd), scale=1.0)
    assert set(out) == set(sd)     # non-LoRA (full finetune / pretrain) path untouched


def test_elastic_config_saver_writes_into_checkpoint_dir(tmp_path):
    import json as _json
    import types
    from llava.train.train import ElasticConfigSaver

    cfg = ElasticConfig(tok_levels=[256, 144, 64, 16], num_query_tokens=256,
                        use_pos_embed=True, pos_embed_type="sincos2d")
    ckpt = tmp_path / "checkpoint-500"
    ckpt.mkdir()
    args = types.SimpleNamespace(local_rank=0, output_dir=str(tmp_path))
    ElasticConfigSaver(cfg).on_save(args, types.SimpleNamespace(global_step=500), None)

    got = _json.load(open(ckpt / "elastic_config.json"))
    # the fields eval needs to rebuild the modules
    assert got["tok_levels"] == [256, 144, 64, 16]
    assert got["num_query_tokens"] == 256
    assert got["use_pos_embed"] is True
    assert got["pos_embed_type"] == "sincos2d"


def test_elastic_config_saver_is_rank0_only_and_tolerates_missing_dir(tmp_path):
    import types
    from llava.train.train import ElasticConfigSaver

    saver = ElasticConfigSaver(ElasticConfig())
    ckpt = tmp_path / "checkpoint-10"
    ckpt.mkdir()
    # non-zero rank must not write (all ranks fire on_save; only rank 0 owns the dir)
    saver.on_save(types.SimpleNamespace(local_rank=1, output_dir=str(tmp_path)),
                  types.SimpleNamespace(global_step=10), None)
    assert not (ckpt / "elastic_config.json").exists()
    # a step whose directory does not exist must be a no-op, not a crash
    saver.on_save(types.SimpleNamespace(local_rank=0, output_dir=str(tmp_path)),
                  types.SimpleNamespace(global_step=999), None)


def test_sincos_pos_embed_is_deterministic_and_distinct():
    from llava.model.elastic.resampler import sincos_2d_pos_embed
    a = sincos_2d_pos_embed(64, 4)
    b = sincos_2d_pos_embed(64, 4)
    assert a.shape == (16, 64)
    assert torch.equal(a, b)                       # fixed, not random
    # distinct positions must get distinct encodings -- that is the whole point
    n = torch.nn.functional.normalize(a, dim=-1)
    off = ~torch.eye(16, dtype=torch.bool)
    assert (n @ n.T)[off].max() < 0.999


def test_resampler_sincos_pos_embed_is_frozen_buffer():
    from llava.model.elastic.resampler import NestedQueryResampler
    r = NestedQueryResampler(64, 16, num_patches=36, n_heads=4, depth=1,
                             use_pos_embed=True, pos_embed_type="sincos2d")
    names = {n for n, _ in r.named_parameters()}
    assert "query_pos_embed" not in names and "patch_pos_embed" not in names
    bufs = dict(r.named_buffers())
    assert bufs["query_pos_embed"].shape == (16, 64)
    assert bufs["patch_pos_embed"].shape == (1, 36, 64)   # interpolated 4x4 -> 6x6
    # still round-trips through a state dict, so warm-start/eval rebuild it
    assert "query_pos_embed" in r.state_dict()


def test_resampler_learned_pos_embed_is_trainable():
    from llava.model.elastic.resampler import NestedQueryResampler
    r = NestedQueryResampler(64, 16, num_patches=36, n_heads=4, depth=1,
                             use_pos_embed=True, pos_embed_type="learned")
    names = dict(r.named_parameters())
    assert names["query_pos_embed"].requires_grad
    assert names["patch_pos_embed"].requires_grad


def test_resampler_shapes_unchanged_with_pos_embed():
    from llava.model.elastic.resampler import NestedQueryResampler
    x = torch.randn(2, 36, 64)
    for kind in ("learned", "sincos2d"):
        r = NestedQueryResampler(64, 16, num_patches=36, n_heads=4, depth=1,
                                 use_pos_embed=True, pos_embed_type=kind).eval()
        for n_tok in (16, 8, 1):
            assert r(x, n_tok=n_tok).shape == (2, n_tok, 64)


def test_resampler_rejects_bad_pos_embed_config():
    from llava.model.elastic.resampler import NestedQueryResampler
    import pytest
    with pytest.raises(ValueError):        # non-square query count
        NestedQueryResampler(64, 15, num_patches=36, use_pos_embed=True,
                             pos_embed_type="sincos2d")
    with pytest.raises(ValueError):        # unknown type
        NestedQueryResampler(64, 16, num_patches=36, use_pos_embed=True,
                             pos_embed_type="rotary")


def test_projector_out_norm_off_by_default():
    proj = NestedProjector(16, 32)
    assert proj.out_norm is None
    assert not any(k.startswith("out_norm") for k in proj.state_dict())
    # calibration is a no-op, so old checkpoints keep their exact behaviour
    assert proj.calibrate_out_norm(0.015) is False


def test_projector_out_norm_calibrates_to_embedding_scale():
    torch.manual_seed(0)
    proj = NestedProjector(16, 64, out_norm=True)
    target = 0.0149                                  # TinyLlama embedding std
    assert proj.calibrate_out_norm(target) is True
    assert torch.allclose(proj.out_norm.weight, torch.full((64,), target))
    out = proj(torch.randn(4, 8, 16))
    # LayerNorm gives unit std, the gain rescales it to the embedding scale
    assert abs(out.std().item() - target) < 0.2 * target
    # and the uncalibrated projector is the ~36x-too-large case we are fixing
    raw = NestedProjector(16, 64)(torch.randn(4, 8, 16))
    assert raw.std().item() > 10 * target


def test_projector_out_norm_does_not_clobber_trained_gain():
    proj = NestedProjector(16, 32, out_norm=True)
    with torch.no_grad():
        proj.out_norm.weight.fill_(0.5)              # stand-in for loaded weights
    assert proj.calibrate_out_norm(0.0149) is False
    assert torch.allclose(proj.out_norm.weight, torch.full((32,), 0.5))


def test_projector_out_norm_preserves_nesting():
    torch.manual_seed(0)
    proj = NestedProjector(16, 64, widths=[16, 32, 64], out_norm=True)
    proj.calibrate_out_norm(0.0149)
    for lvl, w in enumerate([16, 32, 64]):
        proj.set_level(lvl)
        out = proj(torch.randn(2, 5, 16))
        assert out.shape == (2, 5, 64)
        # inactive tail must stay exactly zero -- normalizing over the full
        # llm_dim instead of the active slice would break this
        assert torch.all(out[..., w:] == 0)
        assert abs(out[..., :w].std().item() - 0.0149) < 0.2 * 0.0149


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
