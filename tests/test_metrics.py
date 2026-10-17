import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from anomcal.metrics import auroc, pixel_auroc, pro_auc, rate_above


@pytest.mark.parametrize("seed", range(5))
def test_auroc_matches_sklearn_with_ties(seed):
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, 2, 300)
    scores = np.round(rng.normal(labels * 0.7, 1.0), 1)  # rounding creates many ties
    assert auroc(labels, scores) == pytest.approx(roc_auc_score(labels, scores), abs=1e-12)


def test_auroc_extremes():
    assert auroc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == 1.0
    assert auroc([0, 0, 1, 1], [0.9, 0.8, 0.2, 0.1]) == 0.0
    assert auroc([0, 1], [0.5, 0.5]) == 0.5  # a tie counts 1/2


@pytest.mark.parametrize(
    "labels, scores",
    [([1, 1], [0.1, 0.2]), ([0, 2], [0.1, 0.2]), ([0, 1], [0.1, np.nan]), ([0, 1], [0.1])],
)
def test_auroc_rejects_bad_input(labels, scores):
    with pytest.raises(ValueError):
        auroc(labels, scores)


def _masks_with_two_regions(n=4, size=32):
    masks = np.zeros((n, size, size), bool)
    masks[0, 4:10, 4:10] = True  # region 1
    masks[0, 20:30, 20:30] = True  # region 2 (separate)
    masks[1, 5:7, 25:27] = True  # region 3 (small)
    return masks


def test_pixel_auroc_matches_sklearn():
    rng = np.random.default_rng(0)
    masks = _masks_with_two_regions()
    maps = rng.normal(masks * 1.0, 1.0).astype(np.float32)
    assert pixel_auroc(maps, masks) == pytest.approx(
        roc_auc_score(masks.ravel(), maps.ravel()), abs=1e-10
    )


def test_pro_perfect_map_is_one():
    masks = _masks_with_two_regions()
    assert pro_auc(masks.astype(np.float32), masks) == pytest.approx(1.0)


def test_pro_random_map_is_about_fpr_limit_over_two():
    rng = np.random.default_rng(1)
    masks = _masks_with_two_regions(n=8, size=64)
    vals = [pro_auc(rng.random(masks.shape), masks) for _ in range(5)]
    assert np.mean(vals) == pytest.approx(0.15, abs=0.03)


def test_pro_weights_regions_equally_unlike_pixel_auroc():
    """Detecting only the big region: pixel AUROC stays high, PRO drops (2 of 3 missed)."""
    masks = _masks_with_two_regions()
    maps = np.zeros(masks.shape, np.float32)
    maps[0, 20:30, 20:30] = 1.0  # only the largest region is found
    pro = pro_auc(maps, masks)
    assert pro == pytest.approx(1 / 3, abs=0.02)
    assert pixel_auroc(maps, masks) > 0.7


def test_pro_diagonal_pixels_are_one_region():
    masks = np.zeros((1, 8, 8), bool)
    masks[0, 2, 2] = masks[0, 3, 3] = True  # touching diagonally -> one region (8-conn.)
    maps = np.zeros((1, 8, 8), np.float32)
    maps[0, 2, 2] = 1.0  # half of that one region detected
    assert pro_auc(maps, masks) == pytest.approx(0.5, abs=0.02)


def test_pro_and_pixel_auroc_input_checks():
    masks = _masks_with_two_regions()
    with pytest.raises(ValueError):
        pro_auc(np.zeros((4, 32, 31)), masks)
    with pytest.raises(ValueError):
        pro_auc(np.zeros(masks.shape), masks.astype(int))
    with pytest.raises(ValueError):
        pro_auc(np.zeros(masks.shape), np.zeros(masks.shape, bool))  # no regions
    with pytest.raises(ValueError):
        pro_auc(np.zeros(masks.shape), masks, fpr_limit=0)


def test_rate_above_is_strict():
    assert rate_above([1, 2, 3, 4], 2) == 0.5  # 3 and 4 only; 2 is not > 2
    with pytest.raises(ValueError):
        rate_above([], 0.0)
