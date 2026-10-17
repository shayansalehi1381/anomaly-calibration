"""Anomaly scorers behind one small interface (see :mod:`anomcal.scorers.base`).

Phase 1 ships only ``dummy`` (CPU, no torch) so the whole extraction pipeline can be
tested end to end. Phase 2 adds ``baseline``, ``patchcore`` and ``dinov2_knn``; they
are imported lazily so that the CPU-only install never needs torch.
"""

from __future__ import annotations

from anomcal.scorers.base import Scorer, ScorerConfig
from anomcal.scorers.dummy import DummyScorer

_REGISTRY = {"dummy": DummyScorer}
AVAILABLE_METHODS = tuple(_REGISTRY)


def build_scorer(name: str, config: ScorerConfig) -> Scorer:
    if name not in _REGISTRY:
        raise KeyError(f"unknown method {name!r}; available: {AVAILABLE_METHODS}")
    return _REGISTRY[name](config)


__all__ = ["AVAILABLE_METHODS", "Scorer", "ScorerConfig", "build_scorer"]
