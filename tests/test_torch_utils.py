"""Torch utilities: exact kNN (the float32 cancellation lesson), coreset, maps."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from anomcal.models.torch_utils import (  # noqa: E402
    gaussian_blur,
    greedy_coreset,
    is_oom,
    nn_sq_distances,
    patch_scores_to_maps,
    to_model_input,
)


def brute_force_min_sq(q, b):
    q64, b64 = q.double(), b.double()
    return ((q64[:, None, :] - b64[None, :, :]) ** 2).sum(-1).min(1).values


def offset_data(n_q=300, n_b=500, d=256, offset=50.0, seed=0):
    """Features with a large common offset: large norms, small differences (like CNN features)."""
    g = torch.Generator().manual_seed(seed)
    bank = offset + torch.randn(n_b, d, generator=g)
    queries = offset + torch.randn(n_q, d, generator=g)
    return queries, bank


def test_naive_float32_expansion_is_inaccurate_for_self_distance():
    """The motivating bug: |q|^2 - 2qb + |b|^2 in float32 is not 0 for q == b."""
    _, bank = offset_data()
    q = bank[:50]
    naive = (q * q).sum(1, keepdim=True) - 2 * q @ bank.T + (bank * bank).sum(1)[None]
    self_d = naive.diagonal()  # each q against itself
    assert self_d.abs().max() > 1e-2  # far from 0: rounding noise of large numbers


def test_reranked_knn_is_exact_for_self_distance():
    _, bank = offset_data()
    d = nn_sq_distances(bank[:50], bank)
    assert torch.all(d == 0)  # direct differences: exactly zero


@pytest.mark.parametrize("chunk", [1 << 26, 5000, 1])
def test_knn_matches_float64_brute_force(chunk):
    q, b = offset_data()
    ours = nn_sq_distances(q, b, n_candidates=8, chunk_elems=chunk)
    ref = brute_force_min_sq(q, b)
    torch.testing.assert_close(ours.double(), ref, rtol=1e-5, atol=1e-4)


def test_knn_small_bank_and_errors():
    q, b = offset_data(n_q=5, n_b=3, d=4)
    torch.testing.assert_close(
        nn_sq_distances(q, b, n_candidates=8).double(), brute_force_min_sq(q, b)
    )
    with pytest.raises(ValueError):
        nn_sq_distances(q, b[:, :3])
    with pytest.raises(ValueError):
        nn_sq_distances(q, b[:0])


def test_greedy_coreset_picks_one_point_per_cluster():
    g = torch.Generator().manual_seed(0)
    centres = torch.tensor([[0.0] * 8, [100.0] * 8, [-100.0] * 8, [100.0] * 4 + [-100.0] * 4])
    pts = torch.cat([c + 0.1 * torch.randn(50, 8, generator=g) for c in centres])
    idx = greedy_coreset(pts, 4, seed=3, proj_dim=0)
    assert sorted((idx // 50).tolist()) == [0, 1, 2, 3]  # k-center covers every cluster


def test_greedy_coreset_deterministic_unique_and_seeded():
    pts = torch.randn(400, 64, generator=torch.Generator().manual_seed(1))
    a = greedy_coreset(pts, 40, seed=0, proj_dim=16)
    assert torch.equal(a, greedy_coreset(pts, 40, seed=0, proj_dim=16))
    assert len(set(a.tolist())) == 40 and a.dtype == torch.long
    assert not torch.equal(a, greedy_coreset(pts, 40, seed=1, proj_dim=16))
    assert torch.equal(greedy_coreset(pts, 500, seed=0), torch.arange(400))
    with pytest.raises(ValueError):
        greedy_coreset(pts, 0, seed=0)


def test_coreset_reduces_max_gap_better_than_random():
    """Greedy k-center: the farthest point from the coreset is closer than for a random subset."""
    pts = torch.randn(2000, 16, generator=torch.Generator().manual_seed(2)) ** 3  # heavy tails
    k = 100
    greedy = pts[greedy_coreset(pts, k, seed=0, proj_dim=0)]
    rand = pts[torch.randperm(2000, generator=torch.Generator().manual_seed(0))[:k]]
    assert nn_sq_distances(pts, greedy).max() < nn_sq_distances(pts, rand).max()


def test_to_model_input_shapes_and_normalisation():
    img = np.empty((2, 64, 64, 3), np.uint8)
    img[...] = (124, 116, 104)  # the ImageNet mean colour, per channel
    x = to_model_input(img, 32, torch.device("cpu"))
    assert x.shape == (2, 3, 32, 32) and x.dtype == torch.float32
    assert x.abs().max() < 0.1  # (124/255 - mean) / std ~ 0
    with pytest.raises(ValueError):
        to_model_input(img.astype(np.float32), 32, torch.device("cpu"))


def test_gaussian_blur_and_maps():
    const = torch.full((2, 20, 20), 3.0)
    torch.testing.assert_close(gaussian_blur(const, 2.0), const)
    spike = torch.zeros(1, 41, 41)
    spike[0, 20, 20] = 1.0
    blurred = gaussian_blur(spike, 2.0)
    assert blurred.argmax().item() == 20 * 41 + 20 and abs(blurred.sum().item() - 1) < 1e-4
    assert torch.equal(gaussian_blur(spike, 0), spike)
    maps = patch_scores_to_maps(torch.rand(3, 16), (4, 4), 32, 1.0)
    assert maps.shape == (3, 32, 32)


def test_is_oom():
    assert is_oom(RuntimeError("CUDA out of memory. Tried to allocate"))
    assert not is_oom(RuntimeError("shape mismatch"))
