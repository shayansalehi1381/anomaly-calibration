import colorsys

import numpy as np
import pytest

from anomcal.corruptions import (
    CORRUPTION_NAMES,
    CORRUPTIONS,
    MAX_LEVEL,
    MILD_LEVEL,
    apply_corruption,
    apply_corruption_batch,
    conditions,
    levels_for_count,
    severity_table,
)


def textured_image(size: int = 64, seed: int = 0) -> np.ndarray:
    """Smooth gradient + fine texture + a sharp edge, like a product photo."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    base = 90 + 60 * np.sin(xx / 7.0) * np.cos(yy / 5.0)
    checker = 25 * (((xx // 3) + (yy // 3)) % 2)
    edge = 50 * (xx > size / 2)
    rgb = np.stack([base + checker, base + edge, base - checker + edge / 2], axis=-1)
    rgb += rng.normal(0, 3, rgb.shape)
    return np.clip(rgb, 0, 255).astype(np.uint8)


IMG = textured_image()


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return float("inf") if mse == 0 else 10 * np.log10(255.0**2 / mse)


def test_four_families_with_five_levels():
    assert CORRUPTION_NAMES == ("brightness", "gaussian_blur", "gaussian_noise", "jpeg_compression")
    assert all(len(c.params) == MAX_LEVEL for c in CORRUPTIONS.values())


@pytest.mark.parametrize("name", CORRUPTION_NAMES)
def test_level_zero_is_identity_copy(name):
    out = apply_corruption(IMG, name, 0, "img/0")
    np.testing.assert_array_equal(out, IMG)
    assert out is not IMG


@pytest.mark.parametrize("name", CORRUPTION_NAMES)
@pytest.mark.parametrize("level", range(1, MAX_LEVEL + 1))
def test_shape_dtype_preserved_and_input_untouched(name, level):
    before = IMG.copy()
    out = apply_corruption(IMG, name, level, "img/0")
    assert out.shape == IMG.shape and out.dtype == np.uint8
    np.testing.assert_array_equal(IMG, before)  # no in-place modification
    assert not np.array_equal(out, IMG)  # every level actually changes the image


@pytest.mark.parametrize("name", CORRUPTION_NAMES)
def test_deterministic_for_same_image_id(name):
    a = apply_corruption(IMG, name, 3, "candle/normal/0001")
    b = apply_corruption(IMG, name, 3, "candle/normal/0001")
    np.testing.assert_array_equal(a, b)


def test_noise_depends_on_image_id_but_not_on_batching():
    a = apply_corruption(IMG, "gaussian_noise", 2, "id/a")
    b = apply_corruption(IMG, "gaussian_noise", 2, "id/b")
    assert not np.array_equal(a, b)
    batch = np.stack([IMG, IMG])
    out = apply_corruption_batch(batch, ["id/b", "id/a"], "gaussian_noise", 2)
    np.testing.assert_array_equal(out[1], a)  # same noise in any batch position
    np.testing.assert_array_equal(out[0], b)


@pytest.mark.parametrize("name", ["gaussian_blur", "gaussian_noise", "jpeg_compression"])
def test_degradation_increases_with_severity(name):
    """PSNR to the clean image must fall strictly as the level rises."""
    values = [psnr(IMG, apply_corruption(IMG, name, lv, "img/0")) for lv in range(1, 6)]
    assert all(a > b for a, b in zip(values, values[1:], strict=False)), values


def test_brightness_increases_with_severity():
    means = [apply_corruption(IMG, "brightness", lv, "i").mean() for lv in range(0, 6)]
    assert all(a < b for a, b in zip(means, means[1:], strict=False)), means


def test_brightness_equals_hsv_value_shift():
    """Our RGB-scaling shortcut must match the textbook HSV operation (ImageNet-C)."""
    rng = np.random.default_rng(1)
    img = rng.integers(0, 256, (6, 7, 3), dtype=np.uint8)
    img[0, 0] = 0  # include a black pixel
    c = CORRUPTIONS["brightness"].param(2)
    ours = apply_corruption(img, "brightness", 2, "i").astype(int)
    ref = np.empty_like(ours)
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            h, s, v = colorsys.rgb_to_hsv(*(img[y, x] / 255.0))
            r, g, b = colorsys.hsv_to_rgb(h, s, min(v + c, 1.0))
            ref[y, x] = np.rint(np.array([r, g, b]) * 255)
    assert np.abs(ours - ref).max() <= 1  # rounding only


def test_noise_has_expected_std():
    flat = np.full((128, 128, 3), 128, np.uint8)
    out = apply_corruption(flat, "gaussian_noise", 1, "flat").astype(np.float64)
    std = (out - 128).std() / 255.0
    assert abs(std - CORRUPTIONS["gaussian_noise"].param(1)) < 0.005


def test_blur_preserves_constant_image():
    flat = np.full((32, 32, 3), 77, np.uint8)
    np.testing.assert_array_equal(apply_corruption(flat, "gaussian_blur", 5, "f"), flat)


@pytest.mark.parametrize(
    "bad",
    [np.zeros((8, 8), np.uint8), np.zeros((8, 8, 3), np.float32), np.zeros((8, 8, 4), np.uint8)],
)
def test_rejects_wrong_input(bad):
    with pytest.raises(ValueError):
        apply_corruption(bad, "gaussian_blur", 1, "x")


def test_rejects_unknown_name_and_bad_level():
    with pytest.raises(KeyError):
        apply_corruption(IMG, "fog", 1, "x")
    for lv in (-2, 6, 1.5):
        with pytest.raises(ValueError):
            apply_corruption(IMG, "gaussian_blur", lv, "x")


def test_levels_for_count():
    assert levels_for_count(3) == (1, 3, 5)
    assert levels_for_count(5) == (1, 2, 3, 4, 5)
    assert levels_for_count(2) == (1, 5)
    assert levels_for_count(1) == (5,)
    # going from 3 to 5 severities adds exactly levels 2 and 4 (resume computes only those)
    assert set(levels_for_count(5)) - set(levels_for_count(3)) == {2, 4}
    for bad in (0, 6):
        with pytest.raises(ValueError):
            levels_for_count(bad)


def test_conditions_and_severity_table():
    conds = conditions(["gaussian_blur", "jpeg_compression"], [3, 1])
    assert conds == [("none", 0), ("gaussian_blur", 1), ("gaussian_blur", 3),
                     ("jpeg_compression", 1), ("jpeg_compression", 3)]  # fmt: skip
    with pytest.raises(ValueError):
        conditions(["gaussian_blur"], [0])
    table = severity_table(["jpeg_compression"], [1, 5])
    assert table["jpeg_compression"]["levels"] == {"1": 25, "5": 7}
    assert table["jpeg_compression"]["param_name"] == "quality"


# ---- extra mild level (severity -1): milder than ImageNet-C level 1, additive ----


@pytest.mark.parametrize("name", CORRUPTION_NAMES)
def test_mild_level_is_milder_than_level_1_but_not_identity(name):
    mild = apply_corruption(IMG, name, MILD_LEVEL, "img/0").astype(float)
    one = apply_corruption(IMG, name, 1, "img/0").astype(float)
    d_mild, d_one = np.abs(mild - IMG).mean(), np.abs(one - IMG).mean()
    assert 0 < d_mild < d_one
    assert CORRUPTIONS[name].param(MILD_LEVEL) != CORRUPTIONS[name].param(1)


@pytest.mark.parametrize("name", CORRUPTION_NAMES)
def test_mild_level_is_deterministic(name):
    a = apply_corruption(IMG, name, MILD_LEVEL, "img/0")
    np.testing.assert_array_equal(a, apply_corruption(IMG, name, MILD_LEVEL, "img/0"))


def test_mild_level_does_not_change_default_grids_and_is_validated():
    assert levels_for_count(3) == (1, 3, 5) and levels_for_count(5) == (1, 2, 3, 4, 5)
    assert conditions(["gaussian_blur"], [MILD_LEVEL]) == [("none", 0), ("gaussian_blur", -1)]
    assert severity_table(["jpeg_compression"], [MILD_LEVEL])["jpeg_compression"]["levels"] == {
        "-1": 50
    }
    for bad in (-2, 6):
        with pytest.raises(ValueError):
            apply_corruption(IMG, "gaussian_blur", bad, "img/0")


def test_mild_level_in_score_schema_and_cli():
    from anomcal.extract import ExtractConfig, build_parser
    from anomcal.scores_io import make_frame

    df = make_frame(
        image_ids=["c/n/0"], cls="c", labels=[0], roles=["heldout_normal"],
        official_splits=["train"], protocol="calib", method="baseline", seed=0,
        corruption="gaussian_blur", severity=MILD_LEVEL, scores=np.array([1.0]),
    )  # fmt: skip
    from anomcal.scores_io import validate

    validate(df)  # severity -1 with a corruption is valid
    args = build_parser().parse_args(["--severity-levels", "-1", "--protocols", "calib"])
    assert args.severity_levels == [-1]
    cfg = ExtractConfig(data_root=".", device="cpu", severity_levels=args.severity_levels)
    assert cfg.levels == (-1,)
