"""DINOv2-kNN: self-supervised ViT patch tokens + cosine nearest neighbour.

The same patch-memory idea as PatchCore, with features from DINOv2 ViT-S/14 (Oquab
et al., 2023), in the spirit of AnomalyDINO (Damm et al., WACV 2025):

  1. 224 px input -> 16 x 16 patch tokens (CLS token dropped), 384-d each.
  2. Each token is L2-normalised. For unit vectors, ||a - b||^2 = 2 - 2 cos(a, b), so the
     **cosine distance** 1 - cos = ||a - b||^2 / 2. That lets us reuse the exact
     Euclidean kNN of torch_utils.
  3. The memory bank keeps every memory patch by default (450 images x 256 patches =
     115k x 384 floats = 177 MB), so no coreset is needed.
  4. Image score = mean of the top 1% patch distances (at least one patch). Averaging a
     few top patches is a little more robust to one noisy patch than PatchCore's max.
  5. Maps: patch distances upsampled to 256 px and Gaussian-smoothed, as in PatchCore.

Not implemented (documented): AnomalyDINO's zero-shot background masking and its
higher input resolutions (448/672 px), both optional knobs in that paper.
"""

from __future__ import annotations

import math
import time

import numpy as np
import torch
import torch.nn.functional as F

from anomcal.models.backbones import build_vit
from anomcal.models.torch_utils import (
    greedy_coreset,
    nn_sq_distances,
    patch_scores_to_maps,
    to_model_input,
)
from anomcal.scorers.base import ScorerConfig
from anomcal.scorers.torch_base import TorchScorer


class Dinov2KnnScorer(TorchScorer):
    name = "dinov2_knn"
    produces_maps = True

    def __init__(self, config: ScorerConfig) -> None:
        super().__init__(config)
        s = self.settings
        if not 0 < s.top_fraction <= 1:
            raise ValueError("dinov2_knn.top_fraction must be in (0, 1]")
        if not 0 < s.coreset_ratio <= 1:
            raise ValueError("dinov2_knn.coreset_ratio must be in (0, 1]")
        t0 = time.perf_counter()
        self.backbone = build_vit(s.backbone, s.pretrained, s.image_size).to(self.device)
        self._timed("load_model", t0)
        self.bank: torch.Tensor | None = None
        self.grid: tuple[int, int] | None = None

    def embed(self, batch: np.ndarray) -> torch.Tensor:
        x = to_model_input(batch, self.settings.image_size, self.device)
        with self.autocast():
            tokens, grid = self.backbone(x)
        self.grid = grid
        return F.normalize(tokens.float(), dim=-1)  # unit vectors: cosine via Euclidean

    def fit(self, images: np.ndarray) -> None:
        if len(images) == 0:
            raise ValueError("cannot fit on zero images")
        s = self.settings
        t0 = time.perf_counter()
        all_patches = torch.cat(
            self.map_batches(images, lambda b: self.embed(b).flatten(0, 1).cpu())
        )
        self._timed("fit_features", t0)
        t0 = time.perf_counter()
        if s.coreset_ratio < 1:
            n_select = max(1, int(round(s.coreset_ratio * all_patches.shape[0])))
            idx = greedy_coreset(all_patches, n_select, self.config.seed, s.proj_dim, self.device)
            all_patches = all_patches[idx]
        self.bank = all_patches.to(self.device)
        self._timed("fit_coreset", t0)
        self.info.update(bank_size=int(self.bank.shape[0]), grid=list(self.grid))

    def _score(self, images: np.ndarray, with_maps: bool):
        if self.bank is None:
            raise RuntimeError("call fit() before score()")
        s = self.settings
        t0 = time.perf_counter()
        size = images.shape[1]

        def run(batch):
            feats = self.embed(batch)
            n, p, d = feats.shape
            d2 = nn_sq_distances(feats.reshape(-1, d), self.bank, s.n_candidates, s.chunk_elems)
            dist = (d2 / 2.0).reshape(n, p)  # cosine distance in [0, 2]
            top = max(1, math.ceil(s.top_fraction * p))
            scores = dist.topk(top, dim=1).values.mean(1).cpu().numpy()
            if not with_maps:
                return scores, None
            maps = patch_scores_to_maps(dist, self.grid, size, s.map_sigma)
            return scores, maps.cpu().numpy()

        out = self.map_batches(images, run)
        self._timed("score_maps" if with_maps else "score", t0)
        scores = np.concatenate([o[0] for o in out]).astype(np.float32)
        maps = np.concatenate([o[1] for o in out]).astype(np.float32) if with_maps else None
        return scores, maps
