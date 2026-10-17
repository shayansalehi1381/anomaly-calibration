"""Anomaly scorers behind one small interface (see :mod:`anomcal.scorers.base`).

- ``dummy``: CPU, no torch; pixel-space 1-NN, only for testing the pipeline.
- ``baseline``: global WRN-50 features + kNN distance (no maps).
- ``patchcore``: our lean PatchCore (patch memory bank + greedy coreset), with maps.
- ``dinov2_knn``: DINOv2 patch tokens + cosine kNN (AnomalyDINO-style), with maps.

The torch-based methods are imported lazily, so the CPU-only install never needs torch.
"""

from __future__ import annotations

from importlib import import_module

from anomcal.scorers.base import Scorer, ScorerConfig
from anomcal.scorers.settings import SETTINGS, resolve_settings, settings_dict

_IMPLEMENTATIONS = {
    "dummy": "anomcal.scorers.dummy:DummyScorer",
    "baseline": "anomcal.scorers.baseline:BaselineScorer",
    "patchcore": "anomcal.scorers.patchcore:PatchCoreScorer",
    "dinov2_knn": "anomcal.scorers.dinov2_knn:Dinov2KnnScorer",
}
AVAILABLE_METHODS = tuple(_IMPLEMENTATIONS)
TORCH_METHODS = ("baseline", "patchcore", "dinov2_knn")
assert set(SETTINGS) == set(_IMPLEMENTATIONS)


def build_scorer(name: str, config: ScorerConfig) -> Scorer:
    if name not in _IMPLEMENTATIONS:
        raise KeyError(f"unknown method {name!r}; available: {AVAILABLE_METHODS}")
    module, cls = _IMPLEMENTATIONS[name].split(":")
    try:
        impl = getattr(import_module(module), cls)
    except ImportError as e:
        raise RuntimeError(f"method {name!r} needs torch: pip install -e .[gpu] ({e})") from e
    return impl(config)


__all__ = [
    "AVAILABLE_METHODS",
    "TORCH_METHODS",
    "Scorer",
    "ScorerConfig",
    "build_scorer",
    "resolve_settings",
    "settings_dict",
]
