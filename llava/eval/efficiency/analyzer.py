"""Analytic FLOPs / latency / memory model for FlexLLaVA elastic checkpoints.

Adapted from AdaLLaVA's ada_analyzer.py, which itself extends LLM-Viewer's
ModelAnalyzer (both MIT) -- see NOTICE.md. Using the same cost model as
AdaLLaVA is deliberate: it makes our numbers directly comparable to the ones
published in their paper.

What differs from AdaLLaVA
--------------------------
AdaLLaVA's elasticity axis is *width*: their scheduler switches attention
heads / MLP neurons off, so `num_heads` varies per layer while the prompt
length stays at 576 visual tokens.

FlexLLaVA's axis is *sequence length*: every layer runs at full width, but the
resampler emits 256 / 144 / 64 / 16 visual tokens, so `prompt_len` is what
changes. The per-head machinery is kept (it costs nothing and it is what a
future 2-D token x layer grid would need), but in our calls num_heads is
simply the model's full head count for every layer.

Scope, and what these numbers do NOT include
--------------------------------------------
This models the *LLM* only -- as AdaLLaVA's does. The CLIP ViT-L/336 vision
tower runs at full width at every token level (it is frozen; nested LoRA adds
a negligible additive term and does not shrink it), so it is a constant offset
that is identical across levels and cancels out when comparing them. The
resampler is likewise small and constant. If you want an absolute end-to-end
cost rather than a cross-level comparison, add the vision tower separately --
`vision_tower_gflops()` below is a rough constant for that.

Everything here is an analytic roofline projection: a lower bound assuming
perfect overlap, no kernel-launch overhead, no thermal throttling. It is
meant for comparing token levels against each other and for sanity-checking
whether a config can fit a target board, not as a predicted wall clock.
"""

import math

from .hardware import hardware_memory, hardware_params
from .roofline import roofline_analyze
from . import llama_config

ALL_DATA_NAMES = [
    "OPs",
    "memory_access",
    "load_weight",
    "load_act",
    "store_act",
    "load_kv_cache",
    "store_kv_cache",
    "inference_time",
]

# Metrics surfaced per sample into the lmms-eval results JSON.
EFFICIENCY_METRICS = [
    "flops",
    "avg_flops",
    "prefill_flops",
    "prefill_time",
    "memory_consumption",
    "prefill_memory_consumption",
]


def vision_tower_gflops(model_name="clip-vit-large-patch14-336"):
    """Constant per-image cost of the frozen vision encoder, GFLOPs.

    CLIP ViT-L/14 at 336px = 577 tokens, 24 layers, width 1024 -> ~162 GFLOPs
    forward. Constant across all token levels, hence excluded from the
    elastic comparison; provided so end-to-end totals can be reported.
    """
    return 162.0


class ElasticAnalyzer:
    """Roofline cost model for one LLM backbone on one target accelerator."""

    def __init__(self, model_config, hardware):
        """
        model_config -- a transformers PretrainedConfig (or anything exposing
                        the LLaMA shape fields); read directly rather than
                        re-loading from the hub, so this works offline and on
                        our llava_llama checkpoints.
        hardware     -- key into hardware.hardware_params
        """
        if hardware not in hardware_params:
            raise KeyError(
                f"unknown hardware {hardware!r}; known: {sorted(hardware_params)}")
        self.model_params = model_config
        self.hardware = hardware
        self.config = llama_config
        self.results = None
        self.w_bit = self.a_bit = self.kv_bit = None
        self.batchsize = self.prompt_len = None
        self.tp_size = 1

    # -- roofline plumbing ------------------------------------------------
    def get_hardware_info(self):
        hw = hardware_params[self.hardware]
        if self.w_bit <= 8 and self.a_bit <= 8 and self.kv_bit <= 8:
            max_OPS = hw["INT8"]
        else:
            max_OPS = hw["FP16"]
        return hw["bandwidth"], max_OPS, hw["onchip_buffer"]

    def _analyze_to_results(self, stage, name, OPs=0, load_weight=0, load_act=0,
                            store_act=0, load_kv_cache=0, store_kv_cache=0):
        bandwidth, max_OPS, _ = self.get_hardware_info()
        memory_access = load_weight + load_act + store_act + load_kv_cache + store_kv_cache
        arithmetic_intensity, performance, bound = roofline_analyze(
            bandwidth, max_OPS, OPs, memory_access)
        self.results[stage][name] = {
            "OPs": OPs,
            "memory_access": memory_access,
            "arithmetic_intensity": arithmetic_intensity,
            "performance": performance,
            "bound": bound,
            "load_weight": load_weight,
            "load_act": load_act,
            "store_act": store_act,
            "load_kv_cache": load_kv_cache,
            "store_kv_cache": store_kv_cache,
            "inference_time": OPs / performance if performance else 0.0,
        }

    # -- one transformer layer -------------------------------------------
    def analyze_one_layer(self, prompt_len, num_heads=None, batchsize=1,
                          w_bit=16, a_bit=16, kv_bit=None,
                          use_flashattention=False, tp_size=1):
        assert prompt_len > 0 and batchsize > 0
        self.results = {"decode": {}, "prefill": {}}
        if kv_bit is None:
            kv_bit = a_bit
        self.w_bit, self.a_bit, self.kv_bit = w_bit, a_bit, kv_bit
        self.batchsize, self.prompt_len, self.tp_size = batchsize, prompt_len, tp_size

        w_byte, a_byte, kv_byte = w_bit / 8, a_bit / 8, kv_bit / 8

        config, model_params = self.config, self.model_params
        num_attention_heads = config.get_num_attention_heads(model_params)
        if num_heads is None:
            num_heads = num_attention_heads
        hidden_size = config.get_hidden_size(model_params)
        num_key_value_heads = config.get_num_key_value_heads(model_params)

        for name, (ic, oc) in config.get_linear_layers(model_params, tp_size).items():
            is_kv_proj = name in ["k_proj", "v_proj"]
            is_normal_proj = not is_kv_proj
            self._analyze_to_results(
                "decode", name,
                OPs=ic * oc * batchsize * 2 * num_heads // num_attention_heads,
                load_weight=ic * oc * w_byte * num_heads // num_attention_heads,
                load_act=ic * batchsize * a_byte * num_heads // num_attention_heads,
                store_act=0 if is_kv_proj else (oc * batchsize * a_byte * num_heads // num_attention_heads),
                load_kv_cache=0,
                store_kv_cache=(0 if is_normal_proj else oc * batchsize * kv_byte),
            )
            self._analyze_to_results(
                "prefill", name,
                OPs=ic * oc * batchsize * prompt_len * 2 * num_heads // num_attention_heads,
                load_weight=ic * oc * w_byte * num_heads // num_attention_heads,
                load_act=ic * batchsize * prompt_len * a_byte * num_heads // num_attention_heads,
                store_act=0 if is_kv_proj else (oc * batchsize * prompt_len * a_byte * num_heads // num_attention_heads),
                load_kv_cache=0,
                store_kv_cache=(0 if is_normal_proj else oc * batchsize * prompt_len * kv_byte),
            )

        head_size = hidden_size // num_attention_heads

        # ---- attention, decode (one new token attending to prompt_len) ----
        qk_matmul_OPs = prompt_len * head_size * num_heads * batchsize * 2
        sv_matmul_OPs = 1 * head_size * prompt_len * num_heads * batchsize * 2
        # softmax: max, subtract, exp, sum, divide -> 5 passes
        softmax_OPs = batchsize * num_heads * prompt_len * 1 * 5
        if use_flashattention:
            _, _, onchip_buffer = self.get_hardware_info()
            block_size_r = min(math.ceil(onchip_buffer / (kv_byte * head_size)), head_size)
            n_blocks_r = math.ceil(1 / block_size_r)
            self._analyze_to_results(
                "decode", "fused_attention",
                OPs=qk_matmul_OPs + sv_matmul_OPs + softmax_OPs,
                load_weight=0,
                load_act=1 * head_size * batchsize * num_heads * a_byte,
                store_act=(1 * prompt_len * batchsize * num_heads * a_byte) * 2,
                load_kv_cache=n_blocks_r * prompt_len * head_size * batchsize * num_key_value_heads * kv_byte * 2,
                store_kv_cache=0,
            )
        else:
            self._analyze_to_results(
                "decode", "qk_matmul",
                OPs=qk_matmul_OPs, load_weight=0,
                load_act=1 * head_size * batchsize * num_heads * a_byte,
                store_act=1 * prompt_len * batchsize * num_heads * a_byte,
                load_kv_cache=prompt_len * head_size * batchsize * num_key_value_heads * kv_byte,
                store_kv_cache=0,
            )
            self._analyze_to_results(
                "decode", "sv_matmul",
                OPs=sv_matmul_OPs, load_weight=0,
                load_act=1 * prompt_len * batchsize * num_heads * a_byte,
                store_act=1 * head_size * batchsize * num_heads * a_byte,
                load_kv_cache=prompt_len * head_size * batchsize * num_key_value_heads * kv_byte,
                store_kv_cache=0,
            )
            self._analyze_to_results(
                "decode", "softmax",
                OPs=softmax_OPs, load_weight=0,
                load_act=batchsize * num_heads * prompt_len * 1 * a_byte,
                store_act=batchsize * num_heads * prompt_len * 1 * a_byte,
            )

        for name in config.get_norm_layers(model_params):
            self._analyze_to_results(
                "decode", name,
                OPs=batchsize * hidden_size * 1 * 7,
                load_act=batchsize * hidden_size * 1 * a_byte,
                store_act=batchsize * hidden_size * 1 * a_byte,
            )
        for name in ["attn_add", "mlp_add"]:
            self._analyze_to_results(
                "decode", name,
                OPs=batchsize * hidden_size * 1,
                load_act=batchsize * hidden_size * 1 * a_byte,
                store_act=batchsize * hidden_size * 1 * a_byte,
            )
        self._analyze_to_results(
            "decode", "mlp_act",
            OPs=batchsize * hidden_size * 1 * 2 * num_heads // num_attention_heads,
            load_act=batchsize * hidden_size * 1 * a_byte * 2 * num_heads // num_attention_heads,
            store_act=batchsize * hidden_size * 1 * a_byte * num_heads // num_attention_heads,
        )

        # ---- attention, prefill (full prompt_len x prompt_len) ------------
        # This is the term our token reduction actually attacks: quadratic in
        # prompt_len, and prompt_len = text_tokens + n_visual_tokens.
        qk_matmul_OPs = prompt_len * prompt_len * head_size * num_heads * batchsize * 2
        sv_matmul_OPs = prompt_len * head_size * prompt_len * num_heads * batchsize * 2
        softmax_OPs = batchsize * num_heads * prompt_len * prompt_len * 5
        if use_flashattention:
            _, _, onchip_buffer = self.get_hardware_info()
            block_size_r = min(math.ceil(onchip_buffer / (kv_byte * head_size)), head_size)
            n_blocks_r = math.ceil(prompt_len / block_size_r)
            self._analyze_to_results(
                "prefill", "fused_attention",
                OPs=qk_matmul_OPs + sv_matmul_OPs + softmax_OPs,
                load_weight=0,
                load_act=prompt_len * head_size * batchsize * num_heads * a_byte,
                store_act=(prompt_len * prompt_len * batchsize * num_heads * a_byte) * 2,
                load_kv_cache=n_blocks_r * prompt_len * head_size * batchsize * num_heads * kv_byte * 2,
                store_kv_cache=0,
            )
        else:
            self._analyze_to_results(
                "prefill", "qk_matmul",
                OPs=qk_matmul_OPs, load_weight=0,
                load_act=prompt_len * head_size * batchsize * num_heads * a_byte,
                store_act=prompt_len * prompt_len * batchsize * num_heads * a_byte,
                load_kv_cache=prompt_len * head_size * batchsize * num_heads * kv_byte,
                store_kv_cache=0,
            )
            self._analyze_to_results(
                "prefill", "sv_matmul",
                OPs=sv_matmul_OPs, load_weight=0,
                load_act=prompt_len * prompt_len * batchsize * num_heads * a_byte,
                store_act=prompt_len * head_size * batchsize * num_heads * a_byte,
                load_kv_cache=prompt_len * head_size * batchsize * num_heads * kv_byte,
                store_kv_cache=0,
            )
            self._analyze_to_results(
                "prefill", "softmax",
                OPs=softmax_OPs, load_weight=0,
                load_act=batchsize * num_heads * prompt_len * prompt_len * a_byte,
                store_act=batchsize * num_heads * prompt_len * prompt_len * a_byte,
            )

        for name in config.get_norm_layers(model_params):
            self._analyze_to_results(
                "prefill", name,
                OPs=batchsize * hidden_size * prompt_len * 7,
                load_act=batchsize * hidden_size * prompt_len * a_byte,
                store_act=batchsize * hidden_size * prompt_len * a_byte,
            )
        for name in ["attn_add", "mlp_add"]:
            self._analyze_to_results(
                "prefill", name,
                OPs=batchsize * hidden_size * prompt_len * 1,
                load_act=batchsize * hidden_size * prompt_len * a_byte,
                store_act=batchsize * hidden_size * prompt_len * a_byte,
            )
        self._analyze_to_results(
            "prefill", "mlp_act",
            OPs=batchsize * hidden_size * prompt_len * 1 * 2 * num_heads // num_attention_heads,
            load_act=batchsize * hidden_size * prompt_len * a_byte * 2 * num_heads // num_attention_heads,
            store_act=batchsize * hidden_size * prompt_len * a_byte * num_heads // num_attention_heads,
        )

        return self.results

    # -- whole model ------------------------------------------------------
    def analyze_all_layers(self, prompt_len, num_heads=None, batchsize=1,
                           w_bit=16, a_bit=16, kv_bit=None,
                           use_flashattention=False, tp_size=1):
        """num_heads: list with one entry per transformer layer. A negative
        entry means the layer is skipped entirely (AdaLLaVA-style layer
        dropping); None means every layer runs at full width, which is the
        FlexLLaVA case."""
        if num_heads is None:
            n_layers = self.config.get_num_hidden_layers(self.model_params)
            full = self.config.get_num_attention_heads(self.model_params)
            num_heads = [full] * n_layers

        results = []
        for curr_num_heads in num_heads:
            if curr_num_heads >= 0:
                results.append(self.analyze_one_layer(
                    prompt_len, curr_num_heads, batchsize, w_bit, a_bit,
                    kv_bit, use_flashattention, tp_size))

        total_results = {"decode": {}, "prefill": {}}
        for data_name in ALL_DATA_NAMES:
            total_results["decode"][data_name] = 0
            total_results["prefill"][data_name] = 0
        for stage in ["decode", "prefill"]:
            for layer_results in results:
                for _, result in layer_results[stage].items():
                    for data_name in ALL_DATA_NAMES:
                        total_results[stage][data_name] += result[data_name]

        # memory footprint: weights + KV cache are persistent, activations
        # are transient and taken from the last layer as a high-water mark.
        weight_kv_footprint = (total_results["prefill"]["load_weight"]
                               + total_results["prefill"]["store_kv_cache"])
        decode_tmp_act = sum(r["store_act"] for r in results[-1]["decode"].values())
        total_results["decode"]["memory_consumption"] = decode_tmp_act + weight_kv_footprint
        total_results["decode"]["memory_consumption_tmp_act"] = decode_tmp_act
        total_results["decode"]["memory_consumption_weight"] = total_results["prefill"]["load_weight"]
        total_results["decode"]["memory_consumption_kv_cache"] = total_results["prefill"]["store_kv_cache"]
        prefill_tmp_act = sum(r["store_act"] for r in results[-1]["prefill"].values())
        total_results["prefill"]["memory_consumption"] = prefill_tmp_act + weight_kv_footprint
        total_results["prefill"]["memory_consumption_tmp_act"] = prefill_tmp_act
        total_results["prefill"]["memory_consumption_weight"] = total_results["prefill"]["load_weight"]
        total_results["prefill"]["memory_consumption_kv_cache"] = total_results["prefill"]["store_kv_cache"]

        # lm_head sits outside the repeated block
        args = {"batchsize": batchsize, "a_byte": a_bit / 8, "w_byte": w_bit / 8}
        for layer_info in self.config.post_process(self.model_params, args):
            self._analyze_to_results(**layer_info)
            for data_name in ALL_DATA_NAMES:
                total_results[layer_info["stage"]][data_name] += \
                    self.results[layer_info["stage"]][layer_info["name"]][data_name]

        return total_results

    # -- one generate() call ----------------------------------------------
    def analyze_generate_task(self, prompt_len, gen_len, num_heads=None,
                              batchsize=1, w_bit=16, a_bit=16, kv_bit=16,
                              use_flashattention=False, tp_size=1):
        """Cost of prefilling `prompt_len` tokens then decoding `gen_len`."""
        gen_len = max(int(gen_len), 1)
        prefill_result = self.analyze_all_layers(
            prompt_len, num_heads, batchsize, w_bit, a_bit, kv_bit,
            use_flashattention=use_flashattention, tp_size=tp_size)
        prefill_time = inference_time = prefill_result["prefill"]["inference_time"]
        prefill_flops = flops = prefill_result["prefill"]["OPs"]
        prefill_memory = memory_consumption = prefill_result["prefill"]["memory_consumption"]

        for i in range(prompt_len, prompt_len + gen_len - 1):
            result = self.analyze_all_layers(
                i, num_heads, batchsize, w_bit, a_bit, kv_bit,
                use_flashattention=use_flashattention, tp_size=tp_size)
            inference_time += result["decode"]["inference_time"]
            flops += result["decode"]["OPs"]
            # DEVIATION from AdaLLaVA: they accumulate (`+=`) memory over
            # decode steps, which scales the reported footprint with gen_len.
            # Memory is a high-water mark, not a running total -- and we need
            # it to be one to answer "does this fit in the Orin's 8GB?".
            memory_consumption = max(memory_consumption,
                                     result["decode"]["memory_consumption"])

        return {
            "flops": flops,
            "avg_flops": flops / gen_len,
            "prefill_flops": prefill_flops,
            "prefill_time": prefill_time,
            "total_time": inference_time,
            "memory_consumption": memory_consumption,
            "prefill_memory_consumption": prefill_memory,
        }

    # -- deployment check --------------------------------------------------
    def fits_in_memory(self, memory_consumption_bytes):
        """True/False/None -- does this footprint fit the target board?"""
        cap = hardware_memory.get(self.hardware)
        if cap is None:
            return None
        return memory_consumption_bytes <= cap
