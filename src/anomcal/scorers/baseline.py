"""Baseline: global CNN features + k-nearest-neighbour distance (Bergman et al., 2020).

Each image becomes ONE vector: the global average of WRN-50 ``layer4`` (2048-d,
ImageNet weights). The anomaly score is the mean Euclidean distance to the k nearest
memory-bank vectors (k = 1 by default). It has no spatial resolution, so no anomaly
maps: a small scratch barely moves a global average, which is exactly why patch-level
methods (PatchCore, DINOv2-kNN) exist. The baseline shows how much they gain.

Distances are computed in float64 on the CPU. The bank has at most ~900 vectors, so
this is cheap, and it avoids the float32 cancellation problem by construction.
"""

from __future__ import annotations

import time

import numpy as np
import torch

from anomcal.models.backbones import build_cnn
from anomcal.models.torch_utils import to_model_input
from anomcal.scorers.base import ScorerConfig
from anomcal.scorers.torch_base import TorchScorer


class BaselineScorer(TorchScorer):
    name = "baseline"
    produces_maps = False

    def __init__(self, config: ScorerConfig) -> None:
        super().__init__(config)
        if self.settings.knn_k < 1:
            raise ValueError("baseline.knn_k must be >= 1")
        t0 = time.perf_counter()
        self.backbone = build_cnn(self.settings.backbone, self.settings.pretrained).to(self.device)
        self._timed("load_model", t0)
        self.bank: torch.Tensor | None = None

    def _embed(self, batch: np.ndarray) -> torch.Tensor:
        x = to_model_input(batch, self.settings.image_size, self.device)
        with self.autocast():
            pooled = self.backbone(x, need_pooled=True)["pooled"]
        return pooled.float().cpu().double()

    def fit(self, images: np.ndarray) -> None:
        if len(images) == 0:
            raise ValueError("cannot fit on zero images")
        t0 = time.perf_counter()
        self.bank = torch.cat(self.map_batches(images, self._embed))
        self._timed("fit_features", t0)
        self.info["bank_size"] = int(self.bank.shape[0])

    def _score(self, images: np.ndarray, with_maps: bool):
        if self.bank is None:
            raise RuntimeError("call fit() before score()")
        t0 = time.perf_counter()
        feats = torch.cat(self.map_batches(images, self._embed))
        d = torch.cdist(feats, self.bank)  # float64 on CPU: exact enough, tiny bank
        k = min(self.settings.knn_k, self.bank.shape[0])
        scores = d.topk(k, dim=1, largest=False).values.mean(1)
        self._timed("score", t0)
        return scores.numpy().astype(np.float32), None
