"""Shared torch utilities for the detectors.

Nearest-neighbour search (:func:`nn_sq_distances`)
------------------------------------------------
PatchCore and DINOv2-kNN score every query patch by its distance to the closest
memory-bank patch. With Q queries and M bank vectors, the fast way to get all
squared distances is the expansion

    ||q - b||^2 = ||q||^2 - 2 q.b + ||b||^2        (one matrix product)

but it subtracts large, nearly equal numbers. In float32 (about 7 significant digits),
the result for a near-duplicate pair is mostly rounding noise: in Phase 1 a memory
image scored 0.71 against *itself* instead of 0. The anomaly score is exactly such a
small distance, so this matters. We therefore do two steps:

1. **Shortlist** (fast, approximate): the expansion in float32, keeping the
   ``n_candidates`` smallest per query (top-k);
2. **Re-rank** (exact): recompute those few distances directly as
   sum((q - b)^2). There is no cancellation, because the difference is taken first.

The re-ranked minimum is exact whenever the true nearest neighbour is in the
shortlist. That holds unless the rounding error exceeds the gap between the 1st and the
``n_candidates``-th neighbour, which is negligible for n_candidates = 8 and is tested
against a float64 brute force.

Greedy coreset (:func:`greedy_coreset`)
---------------------------------------
PatchCore keeps a subset of memory patches chosen by the greedy k-center rule: start
from a random patch, then repeatedly add the patch that is *farthest* from everything
selected so far. The subset then covers the feature space with a small maximum gap,
so dropping 90% of the patches changes nearest-neighbour distances little. Distances
are computed after a seeded Gaussian random projection to ``proj_dim`` dimensions
(Johnson-Lindenstrauss: distances are approximately preserved), which makes each
iteration ~8x cheaper for 1024-d features.
"""

from __future__ import annotations

import math

import numpy as np

try:
    import torch
    import torch.nn.functional as F
except ImportError as e:  # pragma: no cover - exercised only without the [gpu] extra
    raise ImportError(
        "anomcal.models needs torch: pip install -e .[gpu] (see README, laptop section)"
    ) from e

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
DEFAULT_CHUNK_ELEMS = 1 << 26  # 64M floats = 256 MB per distance block
DEFAULT_N_CANDIDATES = 8


def resolve_device(device: str) -> torch.device:
    dev = torch.device(device)
    if dev.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"device {device} requested but CUDA is not available")
    return dev


def to_model_input(images: np.ndarray, size: int, device: torch.device) -> torch.Tensor:
    """uint8 (N, S, S, 3) -> float32 (N, 3, size, size), ImageNet-normalised, on ``device``.

    The resize happens on the device (antialiased bilinear, as in our CPU loader).
    """
    if images.dtype != np.uint8 or images.ndim != 4 or images.shape[-1] != 3:
        raise ValueError(f"expected uint8 (N, S, S, 3), got {images.dtype} {images.shape}")
    x = torch.from_numpy(np.ascontiguousarray(images)).to(device)
    x = x.permute(0, 3, 1, 2).float().div_(255.0)
    if x.shape[-1] != size or x.shape[-2] != size:
        x = F.interpolate(
            x, size=(size, size), mode="bilinear", align_corners=False, antialias=True
        )
    mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)
    return (x - mean) / std


def nn_sq_distances(
    queries: torch.Tensor,
    bank: torch.Tensor,
    n_candidates: int = DEFAULT_N_CANDIDATES,
    chunk_elems: int = DEFAULT_CHUNK_ELEMS,
) -> torch.Tensor:
    """Exact squared Euclidean distance from each query row to its nearest bank row.

    queries: (Q, D), bank: (M, D), same device, float32. Returns (Q,) float32 >= 0.
    Work is split into query chunks so the (chunk, M) distance block stays below
    ``chunk_elems`` floats; that is what keeps a 70k-patch bank inside 6 GB of VRAM.
    """
    if queries.ndim != 2 or bank.ndim != 2 or queries.shape[1] != bank.shape[1]:
        raise ValueError(
            f"shape mismatch: queries {tuple(queries.shape)}, bank {tuple(bank.shape)}"
        )
    if bank.shape[0] == 0:
        raise ValueError("empty memory bank")
    queries, bank = queries.float(), bank.float()
    k = min(n_candidates, bank.shape[0])
    bank_sq = (bank * bank).sum(1)  # ||b||^2, computed once
    # Both the (rows, M) shortlist block and the (rows, k, D) re-rank block stay in budget.
    rows = max(1, min(chunk_elems // bank.shape[0], chunk_elems // (k * bank.shape[1])))
    out = torch.empty(queries.shape[0], device=queries.device, dtype=torch.float32)
    for start in range(0, queries.shape[0], rows):
        q = queries[start : start + rows]
        approx = (q * q).sum(1, keepdim=True) - 2.0 * (q @ bank.T) + bank_sq[None, :]
        idx = approx.topk(k, dim=1, largest=False).indices  # shortlist (Qc, k)
        exact = ((q[:, None, :] - bank[idx]) ** 2).sum(-1)  # re-rank without cancellation
        out[start : start + rows] = exact.min(dim=1).values
    return out


def greedy_coreset(
    features: torch.Tensor,
    n_select: int,
    seed: int,
    proj_dim: int = 128,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Indices (int64, on CPU) of ``n_select`` rows chosen by greedy k-center selection.

    ``features`` may live on the CPU (large banks); projection runs on ``device`` in
    chunks. The random projection and the start point come from a CPU generator seeded
    with ``seed``, so the selection is reproducible.
    """
    n = features.shape[0]
    if n_select >= n:
        return torch.arange(n)
    if n_select < 1:
        raise ValueError("n_select must be >= 1")
    device = device or features.device
    gen = torch.Generator().manual_seed(seed)
    d = features.shape[1]
    if proj_dim and proj_dim < d:
        proj = torch.randn(d, proj_dim, generator=gen) / math.sqrt(proj_dim)
        proj = proj.to(device)
        parts = []
        for start in range(0, n, 65536):
            parts.append(features[start : start + 65536].to(device).float() @ proj)
        points = torch.cat(parts)
    else:
        points = features.to(device).float()
    start_idx = int(torch.randint(n, (1,), generator=gen))
    selected = torch.empty(n_select, dtype=torch.long, device=device)
    selected[0] = start_idx
    # Distances to a new centre c via ||p||^2 - 2 p.c + ||c||^2: one pass over the points
    # (a matrix-vector product) instead of materialising (points - c)^2, which is ~3x
    # less memory traffic per iteration. The float32 cancellation that forced the exact
    # re-rank in nn_sq_distances does not matter here: the greedy step compares LARGE
    # distances (it looks for the farthest point), where the relative error is tiny.
    sq_norms = (points * points).sum(1)

    def sq_dist_to(idx):
        return (sq_norms - 2.0 * (points @ points[idx]) + sq_norms[idx]).clamp_min_(0)

    # min_dist[i] = squared distance from point i to its closest selected point so far.
    min_dist = sq_dist_to(start_idx)
    for i in range(1, n_select):
        nxt = torch.argmax(min_dist)  # the point farthest from the current coreset
        selected[i] = nxt  # no .item(): avoids a GPU->CPU sync on every iteration
        torch.minimum(min_dist, sq_dist_to(nxt), out=min_dist)
    return selected.cpu()


def gaussian_blur(maps: torch.Tensor, sigma: float) -> torch.Tensor:
    """Separable Gaussian smoothing of (N, H, W) maps (PatchCore uses sigma = 4 px)."""
    if sigma <= 0:
        return maps
    radius = int(math.ceil(4 * sigma))
    x = torch.arange(-radius, radius + 1, device=maps.device, dtype=torch.float32)
    kernel = torch.exp(-(x**2) / (2 * sigma**2))
    kernel = kernel / kernel.sum()
    out = maps[:, None].float()
    out = F.conv2d(F.pad(out, (radius, radius, 0, 0), mode="replicate"), kernel.view(1, 1, 1, -1))
    out = F.conv2d(F.pad(out, (0, 0, radius, radius), mode="replicate"), kernel.view(1, 1, -1, 1))
    return out[:, 0]


def patch_scores_to_maps(
    patch_scores: torch.Tensor, grid: tuple[int, int], out_size: int, sigma: float
) -> torch.Tensor:
    """(N, h*w) patch scores -> (N, out_size, out_size) smoothed anomaly maps."""
    n = patch_scores.shape[0]
    m = patch_scores.view(n, 1, *grid).float()
    m = F.interpolate(m, size=(out_size, out_size), mode="bilinear", align_corners=False)[:, 0]
    return gaussian_blur(m, sigma)


def reset_peak_memory(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def peak_memory_mb(device: torch.device) -> float | None:
    """Peak VRAM allocated by tensors since the last reset (None on CPU)."""
    if device.type != "cuda":
        return None
    return torch.cuda.max_memory_allocated(device) / 2**20


def is_oom(exc: BaseException) -> bool:
    return isinstance(exc, torch.cuda.OutOfMemoryError) or "out of memory" in str(exc).lower()
