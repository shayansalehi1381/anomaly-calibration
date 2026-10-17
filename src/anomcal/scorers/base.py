"""The scorer interface shared by all anomaly detectors.

A scorer is fit once on the memory-bank images of one class and then scores batches
of (possibly corrupted) images. Inputs are always uint8 arrays of shape
(N, S, S, 3) at the canonical resolution S; any model-specific preprocessing
happens inside the scorer. Higher score = more anomalous.

Scorers that localise (``produces_maps = True``) also return per-pixel anomaly maps of
shape (N, S, S), aligned with the canonical image, for pixel AUROC / PRO.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np


@dataclass(frozen=True)
class ScorerConfig:
    seed: int
    device: str = "cpu"
    batch_size: int = 16
    options: dict = field(default_factory=dict)  # method-specific settings (see settings.py)


@runtime_checkable
class Scorer(Protocol):
    name: str
    produces_maps: bool

    def fit(self, images: np.ndarray) -> None:
        """Build the model / memory bank from normal images (N, S, S, 3) uint8."""
        ...

    def score(self, images: np.ndarray) -> np.ndarray:
        """Image-level anomaly scores, shape (N,), float."""
        ...

    def score_with_maps(self, images: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Scores (N,) and anomaly maps (N, S, S) float32; only if produces_maps."""
        ...

    def describe(self) -> dict:
        """JSON-serialisable settings plus runtime info (timings, bank size, peak VRAM)."""
        ...
