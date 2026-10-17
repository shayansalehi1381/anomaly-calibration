"""Our lean PatchCore (Roth et al., CVPR 2022). Design choices: docs/MODELS.md.

Fit (per class, on clean memory-bank images):
  1. WRN-50 features from ``layer2`` (28 x 28 at 224 px) and ``layer3`` (14 x 14).
  2. *Local neighbourhood aggregation*: 3 x 3 average pooling (stride 1) on each map,
     so every patch feature also describes its immediate surroundings.
  3. ``layer3`` is upsampled to the 28 x 28 grid and concatenated with ``layer2``
     (512 + 1024 = 1536 channels), then average-pooled along the channel axis to
     ``embed_dim`` = 1024. Result: one 1024-d vector per patch, 784 per image.
  4. *Greedy coreset*: keep ``coreset_ratio`` (10%) of all memory patches, chosen by
     greedy k-center selection on a seeded random projection (torch_utils).

Score (per image):
  5. Each patch's distance to its nearest coreset patch (exact re-ranked kNN, so no
     float32 cancellation). The **image score is the maximum over patches**: one
     anomalous patch is enough to flag the image.
  6. The 28 x 28 patch distances, upsampled to 256 x 256 and Gaussian-smoothed
     (sigma = 4), form the anomaly map.

Simplifications vs. the paper (documented, deliberate): channel reduction by a single
average pooling of the concatenated layers (the paper pools each layer first); no
nearest-neighbour re-weighting of the image score; Gaussian (not sparse) random
projection for the coreset.
"""

from __future__ import annotations

import time

import numpy as np
import torch
import torch.nn.functional as F

from anomcal.models.backbones import build_cnn
from anomcal.models.torch_utils import (
    greedy_coreset,
    nn_sq_distances,
    patch_scores_to_maps,
    to_model_input,
)
from anomcal.scorers.base import ScorerConfig
from anomcal.scorers.torch_base import TorchScorer


class PatchCoreScorer(TorchScorer):
    name = "patchcore"
    produces_maps = True

    def __init__(self, config: ScorerConfig) -> None:
        super().__init__(config)
        s = self.settings
        if not 0 < s.coreset_ratio <= 1:
            raise ValueError("patchcore.coreset_ratio must be in (0, 1]")
        if s.neighbourhood < 1 or s.neighbourhood % 2 == 0:
            raise ValueError("patchcore.neighbourhood must be a positive odd number")
        t0 = time.perf_counter()
        self.backbone = build_cnn(s.backbone, s.pretrained).to(self.device)
        self._timed("load_model", t0)
        self.bank: torch.Tensor | None = None
        self.grid: tuple[int, int] | None = None

    def embed(self, batch: np.ndarray) -> torch.Tensor:
        """uint8 images -> patch features (N, h*w, D) float32 on the device."""
        s = self.settings
        x = to_model_input(batch, s.image_size, self.device)
        with self.autocast():
            f = self.backbone(x)
        p = s.neighbourhood
        l2 = F.avg_pool2d(f["layer2"].float(), p, stride=1, padding=p // 2)
        l3 = F.avg_pool2d(f["layer3"].float(), p, stride=1, padding=p // 2)
        l3 = F.interpolate(l3, size=l2.shape[-2:], mode="bilinear", align_corners=False)
        feats = torch.cat([l2, l3], dim=1)  # (N, C2 + C3, h, w)
        n, c, h, w = feats.shape
        self.grid = (h, w)
        feats = feats.permute(0, 2, 3, 1).reshape(n * h * w, 1, c)
        d = min(s.embed_dim, c)
        feats = F.adaptive_avg_pool1d(feats, d)  # average neighbouring channels down to d
        return feats.reshape(n, h * w, d)

    def fit(self, images: np.ndarray) -> None:
        if len(images) == 0:
            raise ValueError("cannot fit on zero images")
        s = self.settings
        t0 = time.perf_counter()
        # All memory patches live in CPU RAM (up to ~2.9 GB for 905 images x 784 patches x
        # 1024 floats); only the selected coreset goes to the GPU. The buffer is allocated
        # once: collecting per-batch pieces and concatenating would briefly need 2x RAM.
        all_patches: torch.Tensor | None = None
        offset = 0

        def collect(batch: np.ndarray) -> None:
            nonlocal all_patches, offset
            f = self.embed(batch).flatten(0, 1).cpu()
            if all_patches is None:
                per_image = f.shape[0] // len(batch)
                all_patches = torch.empty(len(images) * per_image, f.shape[1])
            all_patches[offset : offset + f.shape[0]] = f
            offset += f.shape[0]

        self.map_batches(images, collect)
        assert all_patches is not None and offset == all_patches.shape[0]
        self._timed("fit_features", t0)

        t0 = time.perf_counter()
        n_select = max(1, int(round(s.coreset_ratio * all_patches.shape[0])))
        idx = greedy_coreset(all_patches, n_select, self.config.seed, s.proj_dim, self.device)
        self.bank = all_patches[idx].to(self.device)
        self._timed("fit_coreset", t0)
        self.info.update(
            n_memory_patches=int(all_patches.shape[0]),
            bank_size=int(self.bank.shape[0]),
            grid=list(self.grid),
        )
        del all_patches  # free the CPU buffer before scoring

    def _patch_distances(self, batch: np.ndarray) -> torch.Tensor:
        feats = self.embed(batch)  # (n, P, D)
        n, p, d = feats.shape
        d2 = nn_sq_distances(
            feats.reshape(-1, d), self.bank, self.settings.n_candidates, self.settings.chunk_elems
        )
        return d2.clamp_min(0).sqrt().reshape(n, p)

    def _score(self, images: np.ndarray, with_maps: bool):
        if self.bank is None:
            raise RuntimeError("call fit() before score()")
        t0 = time.perf_counter()
        size = images.shape[1]

        def run(batch):
            dist = self._patch_distances(batch)
            scores = dist.max(dim=1).values.cpu().numpy()
            if not with_maps:
                return scores, None
            maps = patch_scores_to_maps(dist, self.grid, size, self.settings.map_sigma)
            return scores, maps.cpu().numpy()

        out = self.map_batches(images, run)
        self._timed("score_maps" if with_maps else "score", t0)
        scores = np.concatenate([o[0] for o in out]).astype(np.float32)
        maps = np.concatenate([o[1] for o in out]).astype(np.float32) if with_maps else None
        return scores, maps
