"""Hardware roofline parameters.

Ported from LLM-Viewer (MIT, Copyright (c) 2024 Zhihang Yuan) --
see NOTICE.md. Only the accelerators we actually report on are kept.

Fields:
    bandwidth      -- peak DRAM bandwidth, bytes/s
    FP16 / INT8    -- peak *dense* throughput, OPs/s
    onchip_buffer  -- shared-memory/L1 budget in bytes, used only to size
                      FlashAttention blocks (irrelevant when
                      use_flashattention=False, which is our default)
"""

hardware_params = {
    # ---- cluster GPUs (what we actually run eval on) --------------------
    "nvidia_A40": {
        "bandwidth": 696e9,
        "FP16": 149.7e12,
        "INT8": 299.3e12,
        "onchip_buffer": 21504e3,
    },
    "nvidia_A100_40G": {
        "bandwidth": 1555e9,
        "FP16": 312e12,
        "INT8": 624e12,
        "onchip_buffer": 27648e3,
    },
    # A10 is what eval_lmms_level.sh usually lands on; not in upstream
    # LLM-Viewer. 24GB GDDR6, GA102, 125 TFLOPS FP16 dense (non-sparse).
    "nvidia_A10": {
        "bandwidth": 600e9,
        "FP16": 125e12,
        "INT8": 250e12,
        "onchip_buffer": 10752e3,
    },
    # V100 kept because AdaLLaVA's published numbers use it, so our table
    # can be put next to theirs on the same hardware model.
    "nvidia_V100": {
        "bandwidth": 900e9,
        "FP16": 112e12,
        "INT8": 62e12,
        "onchip_buffer": 20480e3,
    },
    # ---- deployment target ---------------------------------------------
    # Jetson Orin Nano 8GB, MAXN/15W (15W *is* this board's max mode; the
    # 7W mode clocks lower and is not modelled here).
    #
    #   GPU     1024-core Ampere (8 SM) + 32 tensor cores
    #   Memory  8GB 128-bit LPDDR5 @ 68 GB/s  <-- the binding constraint
    #   CPU     6-core Cortex-A78AE @ 1.5 GHz
    #
    # NVIDIA rates this board at "40 TOPS", which is INT8 *with 2:4
    # sparsity*. We store dense numbers, because that is what the roofline
    # model assumes and what a dense LLaMA forward actually gets:
    #   40 TOPS sparse INT8 -> 20 TOPS dense INT8 -> 10 TFLOPS dense FP16.
    # onchip_buffer: 8 SMs * 164KB combined L1/shared on Ampere.
    #
    # NOTE: this yields an *analytic roofline projection*, i.e. a lower
    # bound on latency assuming perfect overlap and no kernel-launch,
    # memory-copy, or thermal-throttling overhead. Treat it as a
    # comparison across token levels, not as a predicted wall-clock number
    # on the real board -- measure on-device for that.
    "jetson_orin_nano_8gb": {
        "bandwidth": 68e9,
        "FP16": 10e12,
        "INT8": 20e12,
        "onchip_buffer": 1312e3,
    },
}

# Physical DRAM per accelerator, bytes. Used to flag configurations whose
# projected weight+KV+activation footprint cannot fit on the target board.
hardware_memory = {
    "nvidia_A40": 48e9,
    "nvidia_A100_40G": 40e9,
    "nvidia_A10": 24e9,
    "nvidia_V100": 32e9,
    # 8GB is shared between CPU and GPU on Jetson -- the usable share for
    # the model is meaningfully less than 8GB in practice.
    "jetson_orin_nano_8gb": 8e9,
}
