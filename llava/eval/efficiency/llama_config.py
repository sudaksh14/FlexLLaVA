"""Per-architecture shape accessors for the cost model (LLaMA family).

Ported from LLM-Viewer's configs/Llama.py (MIT, Copyright (c) 2024 Zhihang
Yuan) -- see NOTICE.md.

Works unmodified for every LLaMA-shaped backbone we train, including
Grouped-Query Attention (TinyLlama has 32 query heads / 4 KV heads), and for
our FlexLLaVA checkpoints whose config.json carries model_type
"llava_llama" but keeps all the standard LLaMA fields.
"""


def get_num_attention_heads(model_params):
    return getattr(model_params, "num_attention_heads")


def get_hidden_size(model_params):
    return getattr(model_params, "hidden_size")


def get_num_key_value_heads(model_params):
    # Older LLaMA-1 style configs have no GQA field: KV heads == query heads.
    return getattr(model_params, "num_key_value_heads",
                   getattr(model_params, "num_attention_heads"))


def get_num_hidden_layers(model_params):
    return getattr(model_params, "num_hidden_layers")


def get_intermediate_size(model_params):
    return getattr(model_params, "intermediate_size")


def get_vocab_size(model_params):
    return getattr(model_params, "vocab_size")


def get_norm_layers(model_params):
    return ["attn_norm", "mlp_norm"]


def get_linear_layers(model_params, tp_size: int = 1):
    hidden_size = get_hidden_size(model_params)
    intermediate_size = get_intermediate_size(model_params)
    key_value_heads = get_num_key_value_heads(model_params)
    attention_heads = get_num_attention_heads(model_params)

    if tp_size > 1:
        assert hidden_size % tp_size == 0
        assert intermediate_size % tp_size == 0
        assert key_value_heads % tp_size == 0

    return {
        "q_proj": [hidden_size, hidden_size // tp_size],
        "k_proj": [hidden_size, hidden_size * key_value_heads // attention_heads // tp_size],
        "v_proj": [hidden_size, hidden_size * key_value_heads // attention_heads // tp_size],
        "out_proj": [hidden_size // tp_size, hidden_size],
        "gate_proj": [hidden_size, intermediate_size // tp_size],
        "up_proj": [hidden_size, intermediate_size // tp_size],
        "down_proj": [intermediate_size // tp_size, hidden_size],
    }


def post_process(model_params, args):
    """lm_head, which sits outside the repeated transformer block."""
    hidden_size = get_hidden_size(model_params)
    vocab_size = get_vocab_size(model_params)
    layers = []
    for stage in ["prefill", "decode"]:
        layers.append({
            "name": "lm_head",
            "stage": stage,
            "OPs": args["batchsize"] * hidden_size * vocab_size * 1,
            "load_weight": hidden_size * vocab_size * args["w_byte"],
            "load_act": hidden_size * args["a_byte"],
            "store_act": vocab_size * args["a_byte"],
        })
    return layers
