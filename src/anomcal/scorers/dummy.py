"""A tiny, dependency-free scorer used to test the pipeline (not a real method).

Each image is reduced to a 16 x 16 grayscale thumbnail (block averages) and scored by
the Euclidean distance to its nearest memory-bank thumbnail: a 1-nearest-neighbour
detector in pixel space. It is cheap, deterministic, needs no GPU and gives sensible
numbers on the fake dataset (anomalies there are bright disks), which is enough to
check the plumbing of `extract_scores.py` on the laptop before the real models run.
"""

from __future__ import annotations

import numpy as np

from anomcal.scorers.base import ScorerConfig
from anomcal.scorers.settings import resolve_settings

THUMB = 16


def thumbnails(images: np.ndarray, size: int = THUMB) -> np.ndarray:
    """(N, S, S, 3) uint8 -> (N, size*size) float64 grayscale block means.

    Blocks are S // size pixels wide; any remainder rows/columns are cropped.
    """
    n, s = images.shape[0], images.shape[1]
    block = s // size
    if block < 1:
        raise ValueError(f"images of size {s} are smaller than the {size}x{size} thumbnail")
    gray = images[:, : block * size, : block * size].astype(np.float64).mean(axis=-1)
    return gray.reshape(n, size, block, size, block).mean(axis=(2, 4)).reshape(n, -1)


class DummyScorer:
    name = "dummy"
    produces_maps = False

    def __init__(self, config: ScorerConfig) -> None:
        self.config = config  # runs on CPU whatever config.device says
        self.settings = resolve_settings("dummy", config.options)
        self._bank: np.ndarray | None = None

    def fit(self, images: np.ndarray) -> None:
        if len(images) == 0:
            raise ValueError("cannot fit on zero images")
        self._bank = thumbnails(images, self.settings.thumbnail)

    def score(self, images: np.ndarray) -> np.ndarray:
        if self._bank is None:
            raise RuntimeError("call fit() before score()")
        feats = thumbnails(images, self.settings.thumbnail)
        # Squared distances via |a|^2 - 2ab + |b|^2: one matrix product instead of a loop.
        # This form subtracts large, nearly equal numbers, so it is done in float64:
        # in float32 an image's distance to itself came out as ~0.7 instead of 0
        # (catastrophic cancellation). The same issue matters for PatchCore's kNN.
        d2 = (
            (feats**2).sum(1, keepdims=True)
            - 2.0 * feats @ self._bank.T
            + (self._bank**2).sum(1)[None, :]
        )
        return np.sqrt(np.maximum(d2.min(axis=1), 0.0)).astype(np.float32)

    def score_with_maps(self, images: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError("the dummy scorer produces no anomaly maps")

    def describe(self) -> dict:
        return {
            "method": self.name,
            "settings": {"thumbnail": self.settings.thumbnail},
            "knn_k": 1,
            "device_used": "cpu",
        }
