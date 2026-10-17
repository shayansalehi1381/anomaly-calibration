"""The scorer interface shared by all anomaly detectors.

A scorer is fit once on the memory-bank images of one class and then scores batches
of (possibly corrupted) images. Inputs are always uint8 arrays of shape
(N, S, S, 3) at the canonical resolution S; any model-specific preprocessing
happens inside the scorer. Higher score = more anomalous.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np


@dataclass(frozen=True)
class ScorerConfig:
    seed: int
    device: str = "cpu"
    batch_size: int = 16


@runtime_checkable
class Scorer(Protocol):
    name: str

    def fit(self, images: np.ndarray) -> None:
        """Build the model / memory bank from normal images (N, S, S, 3) uint8."""
        ...

    def score(self, images: np.ndarray) -> np.ndarray:
        """Image-level anomaly scores, shape (N,), float."""
        ...

    def describe(self) -> dict:
        """JSON-serialisable description of every setting (goes into run metadata)."""
        ...
