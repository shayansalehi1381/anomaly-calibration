import numpy as np
import pytest

from anomcal.scorers import AVAILABLE_METHODS, Scorer, ScorerConfig, build_scorer
from anomcal.scorers.dummy import thumbnails


def test_registry():
    assert "dummy" in AVAILABLE_METHODS
    s = build_scorer("dummy", ScorerConfig(seed=0))
    assert isinstance(s, Scorer)
    with pytest.raises(KeyError):
        build_scorer("nope", ScorerConfig(seed=0))


def test_thumbnails_block_means():
    imgs = np.zeros((1, 32, 32, 3), np.uint8)
    imgs[0, :16] = 255  # top half white
    t = thumbnails(imgs, size=2).reshape(2, 2)
    np.testing.assert_allclose(t, [[255, 255], [0, 0]])
    with pytest.raises(ValueError):
        thumbnails(np.zeros((1, 8, 8, 3), np.uint8), size=16)


def test_dummy_scores_memory_images_zero_and_outliers_high():
    rng = np.random.default_rng(0)
    normals = rng.integers(100, 120, (10, 32, 32, 3), dtype=np.uint8)
    s = build_scorer("dummy", ScorerConfig(seed=0))
    with pytest.raises(RuntimeError):
        s.score(normals)
    s.fit(normals)
    np.testing.assert_allclose(s.score(normals), 0.0, atol=1e-2)  # nearest neighbour is itself
    outlier = normals[:1].copy()
    outlier[0, 8:24, 8:24] = 255
    assert s.score(outlier)[0] > s.score(rng.integers(100, 120, (1, 32, 32, 3), dtype=np.uint8))[0]
    assert s.describe()["method"] == "dummy"


def test_dummy_fit_requires_images():
    with pytest.raises(ValueError):
        build_scorer("dummy", ScorerConfig(seed=0)).fit(np.zeros((0, 32, 32, 3), np.uint8))
