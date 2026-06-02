"""Analytic FLOP estimator for the single elasticity axis (L_tok).

Used to plot the accuracy-per-FLOP frontier. The dominant, controllable cost is
LLM prefill, which scales with sequence length = text_len + n_tok. The vision
encoder runs at fixed full width every step (frozen backbone; LoRA, if any, adds
a negligible additive term and is NOT a compute axis), so it is a constant
offset across token levels.
"""


def vision_flops(base_gflops=80.0):
    return base_gflops  # frozen full-width encoder: constant


def llm_prefill_flops(llm_hidden, llm_layers, text_len, n_tok, vocab=32000):
    L = text_len + n_tok
    per_layer_attn = 2 * (4 * L * llm_hidden * llm_hidden) + 2 * (2 * L * L * llm_hidden)
    per_layer_mlp = 2 * (3 * L * llm_hidden * (4 * llm_hidden))
    return llm_layers * (per_layer_attn + per_layer_mlp) / 1e9


def grid_cost(cfg, *, base_gflops=80.0, llm_hidden=4096, llm_layers=32, text_len=64):
    """Returns {l_tok_index: gflops} over the configured token grid."""
    v = vision_flops(base_gflops)
    out = {}
    for lt in cfg.tok_grid():
        n_tok = cfg.tok_levels[lt]
        out[lt] = v + llm_prefill_flops(llm_hidden, llm_layers, text_len, n_tok)
    return out
