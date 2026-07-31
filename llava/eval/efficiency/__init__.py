"""Analytic efficiency metrics (FLOPs / latency / memory) for FlexLLaVA.

See NOTICE.md for provenance (ported from LLM-Viewer + AdaLLaVA, both MIT)
and for what these numbers do and do not capture.
"""

from .analyzer import (
    EFFICIENCY_METRICS,
    ElasticAnalyzer,
    vision_tower_gflops,
)
from .hardware import hardware_memory, hardware_params

__all__ = [
    "ElasticAnalyzer",
    "EFFICIENCY_METRICS",
    "vision_tower_gflops",
    "hardware_params",
    "hardware_memory",
]
