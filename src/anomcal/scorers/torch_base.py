"""Shared machinery for the torch-based scorers (baseline, PatchCore, DINOv2-kNN).

What every torch scorer needs, so each method file only contains its own idea:

- device handling (explicit, never a silent CPU fallback), eval mode, no gradients;
- float16 autocast for the backbone on CUDA (half the activation memory, faster on
  tensor cores); all distances are computed in float32 afterwards;
- sub-batching with **out-of-memory backoff**: if a batch does not fit into the 6 GB
  of VRAM, it is split in half and retried, down to single images;
- timings and peak VRAM, reported by :meth:`describe` and stored in ``run_meta.json``.
"""

from __future__ import annotations

import dataclasses
import time
from collections.abc import Callable
from contextlib import nullcontext

import numpy as np
import torch

from anomcal.models.torch_utils import is_oom, peak_memory_mb, reset_peak_memory, resolve_device
from anomcal.scorers.base import ScorerConfig
from anomcal.scorers.settings import resolve_settings


class TorchScorer:
    name = "torch"
    produces_maps = False

    def __init__(self, config: ScorerConfig) -> None:
        self.config = config
        self.settings = resolve_settings(self.name, config.options)
        self.device = resolve_device(config.device)
        self.batch_size = config.batch_size
        self.timings: dict[str, float] = {}
        self.info: dict = {}
        reset_peak_memory(self.device)

    # ------------------------------------------------------------------ helpers
    def autocast(self):
        if self.device.type == "cuda" and getattr(self.settings, "amp", False):
            return torch.autocast(device_type="cuda", dtype=torch.float16)
        return nullcontext()

    def map_batches(self, images: np.ndarray, fn: Callable[[np.ndarray], object]) -> list:
        """Apply ``fn`` to sub-batches of ``images``, halving the batch size on OOM."""
        results, start = [], 0
        while start < len(images):
            batch = images[start : start + self.batch_size]
            try:
                with torch.inference_mode():
                    results.append(fn(batch))
            except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
                if not is_oom(e) or self.batch_size == 1:
                    raise
                if self.device.type == "cuda":
                    torch.cuda.empty_cache()
                self.batch_size = max(1, self.batch_size // 2)
                self.info["oom_backoffs"] = self.info.get("oom_backoffs", 0) + 1
                continue  # retry the same images with a smaller batch
            start += len(batch)
        return results

    def _timed(self, key: str, t0: float) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        self.timings[key] = self.timings.get(key, 0.0) + time.perf_counter() - t0

    # ------------------------------------------------------------------ interface
    def score(self, images: np.ndarray) -> np.ndarray:
        return self._score(images, with_maps=False)[0]

    def score_with_maps(self, images: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if not self.produces_maps:
            raise NotImplementedError(f"{self.name} produces no anomaly maps")
        scores, maps = self._score(images, with_maps=True)
        return scores, maps

    def _score(self, images: np.ndarray, with_maps: bool) -> tuple[np.ndarray, np.ndarray | None]:
        raise NotImplementedError

    def describe(self) -> dict:
        return {
            "method": self.name,
            "settings": dataclasses.asdict(self.settings),
            "device_used": str(self.device),
            "batch_size_final": self.batch_size,
            "seconds": {k: round(v, 3) for k, v in self.timings.items()},
            "peak_vram_mb": peak_memory_mb(self.device),
            **self.info,
        }
