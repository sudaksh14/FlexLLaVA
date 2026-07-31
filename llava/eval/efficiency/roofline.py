"""Roofline performance model.

Ported verbatim (minus dead code) from LLM-Viewer (MIT, Copyright (c) 2024
Zhihang Yuan) -- see NOTICE.md.
"""


def roofline_analyze(bandwidth, max_OPS, OPs, memory_access):
    """Return (arithmetic_intensity, achievable_performance, bound).

    bandwidth      -- bytes/s
    max_OPS        -- peak compute, OPs/s
    OPs            -- operations for this kernel
    memory_access  -- bytes moved for this kernel
    """
    turning_point = max_OPS / bandwidth
    if memory_access == 0:
        # No traffic: compute-bound by definition, avoids div-by-zero on
        # parameter-free ops.
        return float("inf"), max_OPS, "compute"
    arithmetic_intensity = OPs / memory_access
    if arithmetic_intensity < turning_point:
        return arithmetic_intensity, arithmetic_intensity * bandwidth, "memory"
    return arithmetic_intensity, max_OPS, "compute"
