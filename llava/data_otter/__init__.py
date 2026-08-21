"""Otter-inspired data pipeline for FlexLLaVA.

A parallel pipeline to `llava/train/train.py`'s LazySupervisedDataset, added so
mixture/packing/telemetry work can proceed without touching the recipe that
current jobs depend on. Entry point: `llava/train/train_otter.py`.

Submodules are imported lazily by their consumers -- importing this package
must stay cheap, because the verification gate imports it before any model or
CUDA context exists.
"""

__all__ = [
    "mixture",
    "packing",
    "dataset",
    "lengths",
    "manifest",
    "telemetry",
    "verify",
]
