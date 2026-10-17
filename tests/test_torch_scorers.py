"""Baseline, PatchCore and DINOv2-kNN on tiny synthetic data with tiny random backbones.

No weights are downloaded: ``tiny_cnn`` / ``tiny_vit`` are seeded random networks with
the same output structure as WRN-50 / DINOv2. What is tested is the *method logic*
(memory bank, coreset, kNN, aggregation, maps, determinism, OOM backoff), not ImageNet
feature quality. Real-weight behaviour is checked by the laptop smoke test.
"""

import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from anomcal.metrics import auroc  # noqa: E402
from anomcal.scorers import ScorerConfig, build_scorer  # noqa: E402

SIZE = 64
TINY = {
    "baseline": {"backbone": "tiny_cnn", "pretrained": False, "image_size": 64},
    "patchcore": {
        "backbone": "tiny_cnn",
        "pretrained": False,
        "image_size": 64,
        "embed_dim": 16,
        "coreset_ratio": 0.5,
        "proj_dim": 8,
    },  # fmt: skip
    "dinov2_knn": {"backbone": "tiny_vit", "pretrained": False, "image_size": 56},
}


def make_images(n, seed, anomalous=False):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:SIZE, 0:SIZE]
    imgs = []
    for _ in range(n):
        base = 120 + 30 * np.sin(xx / 5.0 + rng.uniform(0, 0.3))[..., None] * np.ones(3)
        img = np.clip(base + rng.normal(0, 4, (SIZE, SIZE, 3)), 0, 255)
        if anomalous:
            y, x = rng.integers(8, SIZE - 24, 2)
            img[y : y + 16, x : x + 16] = (255, 30, 30)  # a red defect patch
        imgs.append(img.astype(np.uint8))
    return np.stack(imgs)


NORMAL_TRAIN = make_images(12, 0)
NORMAL_TEST = make_images(8, 1)
ANOMALOUS = make_images(8, 2, anomalous=True)


def scorer(method, seed=0, batch_size=4, **extra):
    opts = {**TINY[method], **extra}
    return build_scorer(
        method, ScorerConfig(seed=seed, device="cpu", batch_size=batch_size, options=opts)
    )


@pytest.mark.parametrize("method", list(TINY))
def test_anomalies_score_higher(method):
    s = scorer(method)
    s.fit(NORMAL_TRAIN)
    normal, anom = s.score(NORMAL_TEST), s.score(ANOMALOUS)
    assert normal.shape == (8,) and normal.dtype == np.float32 and np.isfinite(normal).all()
    labels = np.r_[np.zeros(8), np.ones(8)]
    assert auroc(labels, np.r_[normal, anom]) > 0.9  # trivial toy defect, random features


@pytest.mark.parametrize("method", list(TINY))
def test_deterministic_given_seed(method):
    a, b = scorer(method, seed=5), scorer(method, seed=5)
    a.fit(NORMAL_TRAIN)
    b.fit(NORMAL_TRAIN)
    np.testing.assert_array_equal(a.score(ANOMALOUS), b.score(ANOMALOUS))


@pytest.mark.parametrize("method", ["patchcore", "dinov2_knn"])
def test_memory_images_score_exactly_zero_without_coreset(method):
    """With the full bank, a memory image's patches are in the bank: distance 0, exactly.
    (The float32 expansion alone would give rounding noise here, cf. Phase 1.)"""
    s = scorer(method, coreset_ratio=1.0)
    s.fit(NORMAL_TRAIN)
    assert np.all(s.score(NORMAL_TRAIN[:4]) == 0.0)


@pytest.mark.parametrize("method", ["patchcore", "dinov2_knn"])
def test_maps_shape_and_localisation(method):
    s = scorer(method)
    s.fit(NORMAL_TRAIN)
    scores, maps = s.score_with_maps(ANOMALOUS)
    assert maps.shape == (8, SIZE, SIZE) and maps.dtype == np.float32
    np.testing.assert_allclose(scores, s.score(ANOMALOUS), rtol=1e-6)
    # the map peaks inside (or right next to) the red square
    for img, m in zip(ANOMALOUS, maps, strict=True):
        red = (img[..., 0] == 255) & (img[..., 1] == 30)
        assert m[red].mean() > m[~red].mean()


def test_baseline_has_no_maps():
    s = scorer("baseline")
    s.fit(NORMAL_TRAIN)
    with pytest.raises(NotImplementedError):
        s.score_with_maps(ANOMALOUS)


def test_patchcore_coreset_size_and_describe():
    s = scorer("patchcore", coreset_ratio=0.25)
    s.fit(NORMAL_TRAIN)
    d = s.describe()
    patches = 12 * 8 * 8  # tiny_cnn at 64 px: layer2 grid 8 x 8
    assert d["n_memory_patches"] == patches and d["bank_size"] == round(0.25 * patches)
    assert d["grid"] == [8, 8] and d["settings"]["coreset_ratio"] == 0.25
    assert {"fit_features", "fit_coreset", "load_model"} <= set(d["seconds"])
    assert d["peak_vram_mb"] is None  # CPU
    json.dumps(d)  # goes into run_meta.json


def test_dinov2_top_fraction_and_bank():
    s = scorer("dinov2_knn", top_fraction=1.0)
    s.fit(NORMAL_TRAIN)
    assert s.describe()["bank_size"] == 12 * 16  # 56 px / 14 = 4 x 4 patches


def test_score_before_fit_raises():
    for m in TINY:
        with pytest.raises(RuntimeError):
            scorer(m).score(NORMAL_TEST)


@pytest.mark.parametrize(
    "method, opts",
    [("patchcore", {"coreset_ratio": 0.0}), ("patchcore", {"neighbourhood": 2}),
     ("dinov2_knn", {"top_fraction": 1.5}), ("baseline", {"knn_k": 0}),
     ("dinov2_knn", {"image_size": 50})],
)  # fmt: skip
def test_invalid_settings(method, opts):
    with pytest.raises(ValueError):
        scorer(method, **opts)


def test_out_of_memory_backoff_halves_the_batch(monkeypatch):
    s = scorer("patchcore", batch_size=8)
    s.fit(NORMAL_TRAIN)
    expected = s.score(ANOMALOUS)
    real = s._patch_distances

    def fake_oom(batch):
        if len(batch) > 2:
            raise RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")
        return real(batch)

    monkeypatch.setattr(s, "_patch_distances", fake_oom)
    np.testing.assert_allclose(s.score(ANOMALOUS), expected, rtol=1e-6)
    assert s.batch_size == 2 and s.describe()["oom_backoffs"] == 2


def test_non_oom_errors_are_not_swallowed(monkeypatch):
    s = scorer("patchcore")
    s.fit(NORMAL_TRAIN)
    monkeypatch.setattr(s, "_patch_distances", lambda b: (_ for _ in ()).throw(RuntimeError("bug")))
    with pytest.raises(RuntimeError, match="bug"):
        s.score(ANOMALOUS)


def test_cuda_request_without_cuda_fails_loudly():
    if torch.cuda.is_available():
        pytest.skip("CUDA present")
    with pytest.raises(RuntimeError, match="CUDA"):
        build_scorer("patchcore", ScorerConfig(seed=0, device="cuda", options=TINY["patchcore"]))
