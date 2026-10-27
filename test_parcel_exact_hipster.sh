#!/bin/bash
#SBATCH --job-name=test_parcel_exact_hipster
#SBATCH -t 00:30:00
#SBATCH --partition=capacity
#SBATCH --gres=gpu:l4:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --output=./jobs/test_parcel_exact_hipster_%A.out
#SBATCH --export=ALL
# Unit tests for the PARCEL pipeline (resampler_arch=parcel_exact, budget_sampling=uniform_range).
# CPU only (hipster has no CPU partition, so it takes one L4 slot). Prints ALL_TESTS_PASSED on success and exits non-zero on the first failure.
set -uo pipefail
eval "$(conda shell.bash hook)"
conda activate matryoshka-mm
cd "${SLURM_SUBMIT_DIR:?submit from the repo root}"
python3 - <<'PYEOF'
import sys
import torch

from llava.model.elastic.resampler import NestedQueryResampler, pool_anchors
from llava.model.elastic.config import ElasticConfig
from llava.model.elastic.engine import ElasticEngine
from llava.train.train_elastic import _parse_elastic_args, _parse_anchor_routing

torch.manual_seed(0)
ROUTING = {16: 16, 64: 64}
C, P = 64, 576
x = torch.randn(2, P, C)


def make(arch="parcel_exact", **kw):
    args = dict(num_patches=P, n_heads=4, depth=1, resampler_arch=arch,
                anchor_mode="fixed", anchor_routing=ROUTING)
    args.update(kw)
    return NestedQueryResampler(C, 256, **args)


# ---- 1. routing is PARCEL's literal step function, pure-anchor budgets allowed ----
r = make()
for b in range(16, 64):
    assert r.n_anchors_for(b, P) == 16, (b, r.n_anchors_for(b, P))
for b in range(64, 257):
    assert r.n_anchors_for(b, P) == 64, (b, r.n_anchors_for(b, P))
print("OK: 16<=B<64 -> 16 anchors, 64<=B<=256 -> 64 anchors (Sec. 3.3)")

# the existing Nest-VLM arch cannot produce a pure-anchor budget (documented here so it
# is not mistaken for PARCEL's routing again): fixed routing at B=16 / 64 snaps DOWN
old = make("pool_anchored")
assert old.n_anchors_for(16, P) == 9 and old.n_anchors_for(64, P) == 36, \
    (old.n_anchors_for(16, P), old.n_anchors_for(64, P))
print("OK: pool_anchored fixed {16:16,64:64} really runs 9+7 / 36+28 (never pure anchors)")

# ---- 2. pure-anchor budgets are exactly M3 pooling, no parameters touched ----
r.eval()
with torch.no_grad():
    for b, n_p in ((16, 16), (64, 64)):
        out = r(x, n_tok=b)
        assert out.shape == (2, b, C), out.shape
        assert torch.equal(out, pool_anchors(x, n_p)), f"B={b} must be raw pooled anchors"
print("OK: B=16 and B=64 return raw avg-pooled anchors (== M3 pooling)")

# ---- 3. anchors pass through UNCHANGED when queries exist; shapes at odd budgets ----
with torch.no_grad():
    for b in (18, 40, 62, 66, 100, 144, 200, 256):
        out = r(x, n_tok=b)
        n_p = r.n_anchors_for(b, P)
        assert out.shape == (2, b, C), (b, out.shape)
        assert torch.equal(out[:, :n_p], pool_anchors(x, n_p)), f"B={b}: anchors were modified"
print("OK: anchors are concatenated raw (bit-identical) at every sampled budget")

# ---- 4. queries really are pool-aware and really cross-attend ----
r.train()
q_out = r(x, n_tok=144)[:, 64:]
assert q_out.shape == (2, 80, C)
blk = make()
blk.load_state_dict(r.state_dict())
with torch.no_grad():
    blk.pool_self_attn["attn"].out_proj.weight.zero_()
    blk.pool_self_attn["attn"].out_proj.bias.zero_()
    blk.pool_self_attn["ff"][-1].weight.zero_()
    blk.pool_self_attn["ff"][-1].bias.zero_()
    blk.eval(); r.eval()
    assert not torch.allclose(blk(x, n_tok=144)[:, 64:], r(x, n_tok=144)[:, 64:], atol=1e-5), \
        "queries do not depend on the Query<->Pool self-attention block"
print("OK: queries depend on the Query<->Pool self-attention block")

r.train()
out = r(x, n_tok=144)
out[:, 64:].sum().backward()
for name in ("queries", "pool_self_attn.attn.in_proj_weight", "layers.0.cross_attn.in_proj_weight"):
    g = dict(r.named_parameters())[name].grad
    assert g is not None and g.abs().sum() > 0, f"no gradient into {name}"
print("OK: gradients reach the queries, the self-attention block and the cross-attention block")

# ---- 5. structure: exactly one cross-attention block; construction guards ----
assert len(r.layers) == 1
for kw, msg in (({"depth": 2}, "depth"), ({"anchor_mode": "ratio"}, "fixed"),
                ({"anchor_routing": None}, "fixed")):
    try:
        make(**kw)
        raise SystemExit(f"FAIL: parcel_exact accepted {kw}")
    except ValueError as e:
        assert msg in str(e), (kw, str(e))
print("OK: depth 1 enforced; non-fixed routing rejected")

# ---- 6. the query-only LayerNorm is a flag and never touches anchors ----
nonorm = make(parcel_query_out_norm=False)
nonorm.load_state_dict(r.state_dict())
r.eval(); nonorm.eval()
with torch.no_grad():
    a, b_ = r(x, n_tok=144), nonorm(x, n_tok=144)
assert torch.equal(a[:, :64], b_[:, :64]) and not torch.allclose(a[:, 64:], b_[:, 64:])
print("OK: parcel_query_out_norm only changes the query part")

# ---- 7. uniform budget sampling ----
cfg = ElasticConfig(token_reduction="nested_query", tok_levels=[256, 144, 64, 16], num_query_tokens=256,
                    use_lora=False, lora_ranks=[8, 16, 32, 64], use_prefix_kl=False, use_coral_align=False,
                    resampler_arch="parcel_exact", anchor_mode="fixed", anchor_routing=ROUTING,
                    use_pos_embed=False, single_budget_per_step=True, budget_sampling="uniform_range")


class _VT:   # the engine only reads num_patches from the tower here
    num_patches = P


eng = ElasticEngine(cfg, _VT(), C, 48)
assert len(eng.resampler.layers) == 1, "engine must build the depth-1 connector for parcel_exact"
draws = []
for step in range(1, 30001):
    eng._fwd_step = step
    draws.append(eng.uniform_budget())
assert set(draws) == set(range(16, 257, 2)), "not uniform over {16,18,...,256}"
mean = sum(draws) / len(draws)
assert abs(mean - 136) < 1.5, mean
counts = torch.bincount(torch.tensor(draws))[16::2].float()
assert counts.min() > 0.6 * counts.mean() and counts.max() < 1.4 * counts.mean(), "draws are not flat"
eng._fwd_step = 7
assert eng.uniform_budget() == eng.uniform_budget(), "must be deterministic per step (DDP ranks + recompute)"
print(f"OK: B ~ Uniform{{16,18,...,256}}  (mean {mean:.1f}, expected 136, 121 distinct values)")

for b, lvl in ((256, 0), (200, 0), (145, 0), (144, 1), (100, 1), (65, 1), (64, 2), (17, 2), (16, 3)):
    assert eng.level_for_budget(b) == lvl, (b, eng.level_for_budget(b), lvl)
print("OK: LoRA level for an arbitrary B = smallest ladder level >= B")

# ---- 8. engine.reduce_tokens: arbitrary B in training, the ladder at evaluation ----
feats = torch.randn(2, P, C)
eng.resampler.train()
for step in (1, 2, 3, 4, 5, 6, 7, 8):
    eng._fwd_step = step
    out = eng.reduce_tokens(feats, 0)
    assert out.shape[1] == eng.uniform_budget() and out.shape[2] == 48, (out.shape, eng.uniform_budget())
eng.resampler.eval()
for l_tok, n in enumerate((256, 144, 64, 16)):
    assert eng.reduce_tokens(feats, l_tok).shape[1] == n
print("OK: training runs the resampler at the sampled B; evaluation uses the 256/144/64/16 ladder")

cfg2 = ElasticConfig(token_reduction="nested_query", tok_levels=[256, 144, 64, 16], num_query_tokens=256,
                     use_lora=False, resampler_arch="pool_anchored", anchor_mode="ratio")
eng2 = ElasticEngine(cfg2, _VT(), C, 48)
assert eng2.uniform_budget() is None and len(eng2.resampler.layers) == 2
print("OK: the Nest-VLM pipeline (pool_anchored, ladder sampling, depth 2) is untouched")

# ---- 9. CLI wiring and validation ----
sys.argv = ["x", "--tok_levels", "256", "144", "64", "16", "--resampler_arch", "parcel_exact",
            "--anchor_mode", "fixed", "--anchor_routing", "16:16,64:64", "--single_budget_per_step", "True",
            "--budget_sampling", "uniform_range", "--use_kd", "False", "--use_coral", "False"]
a = _parse_elastic_args()
assert a.budget_sampling == "uniform_range" and (a.budget_min, a.budget_max, a.budget_step) == (16, 256, 2)
assert a.resampler_arch == "parcel_exact" and _parse_anchor_routing(a.anchor_routing) == ROUTING
assert a.parcel_query_out_norm is True
print("OK: train_elastic CLI parses the new flags")

# ---- 10. round trip through the saved config (what the eval wrapper does) ----
import dataclasses, json
raw = json.loads(json.dumps(dataclasses.asdict(cfg)))
cfg3 = ElasticConfig(**raw)
assert cfg3.budget_sampling == "uniform_range" and cfg3.anchor_routing == ROUTING
old_raw = {k: v for k, v in raw.items() if k not in ("parcel_query_out_norm", "budget_sampling",
                                                     "budget_min", "budget_max", "budget_step")}
assert ElasticConfig(**old_raw).budget_sampling == "ladder", "old checkpoints must still load"
print("OK: elastic_config.json round-trips; pre-existing checkpoints load with ladder defaults")
print("ALL_TESTS_PASSED")
PYEOF
