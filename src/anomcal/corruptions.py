"""Deterministic synthetic distribution shifts (Phase 1; used on the laptop).

Four families, following ImageNet-C (Hendrycks & Dietterich, ICLR 2019). Each has
five severity levels, with the parameter values from the ImageNet-C reference code:

=================  =======================  =========================================
family             what it models           parameter per level 1..5
=================  =======================  =========================================
brightness         illumination change      +c on HSV value: .1 .2 .3 .4 .5
gaussian_blur      defocus / motion          sigma (px): 1 2 3 4 6
gaussian_noise     sensor noise              std on [0,1] scale: .08 .12 .18 .26 .38
jpeg_compression   transmission/storage      JPEG quality: 25 18 15 10 7
=================  =======================  =========================================

Level 0 is "clean" (identity) for every family. The levels actually used are a
config value / CLI flag (default 3 levels: 1, 3, 5), so levels 2 and 4 can be added
later without code changes; the parameters used are written into the run metadata.

ImageNet-C uses 224 px images; we corrupt at our canonical 256 px resolution, so the
same sigma/noise values are slightly milder relative to the image size. All
corruptions operate on uint8 RGB arrays of shape (H, W, 3) and return a new array.

Determinism: blur, brightness and JPEG have no randomness. Gaussian noise draws from
a generator seeded by (image id, family, level) via :func:`anomcal.seeding.item_rng`,
so an image gets the same noise no matter the batch size, the order or a resumed run.
JPEG output depends on the libjpeg build shipped with Pillow, so the Pillow version is
recorded in the run metadata.
"""

from __future__ import annotations

import io
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

from anomcal.seeding import item_rng

MAX_LEVEL = 5
DEFAULT_N_SEVERITIES = 3


def _to_float(img: np.ndarray) -> np.ndarray:
    return img.astype(np.float32) / 255.0


def _to_uint8(x: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(x * 255.0), 0, 255).astype(np.uint8)


def _brightness(img: np.ndarray, c: float, rng: np.random.Generator | None) -> np.ndarray:
    # ImageNet-C: convert to HSV, add c to V, convert back. Scaling R, G and B by the
    # same factor k keeps hue and saturation unchanged and multiplies V = max(R,G,B)
    # by k, so with k = V_new / V this equals the HSV operation without a colour-space
    # round trip. A black pixel (V = 0, saturation 0) becomes grey with value c.
    x = _to_float(img)
    v = x.max(axis=-1, keepdims=True)
    v_new = np.clip(v + c, 0.0, 1.0)
    scale = np.divide(v_new, v, out=np.zeros_like(v), where=v > 0)
    out = np.where(v > 0, x * scale, v_new)
    return _to_uint8(out)


def _gaussian_blur(img: np.ndarray, sigma: float, rng: np.random.Generator | None) -> np.ndarray:
    # sigma 0 along the channel axis: blur each colour channel spatially only.
    # mode="nearest" repeats edge pixels (as skimage's default in ImageNet-C).
    out = gaussian_filter(_to_float(img), sigma=(sigma, sigma, 0), mode="nearest")
    return _to_uint8(out)


def _gaussian_noise(img: np.ndarray, std: float, rng: np.random.Generator | None) -> np.ndarray:
    if rng is None:
        raise ValueError("gaussian_noise needs a seeded generator")
    x = _to_float(img)
    return _to_uint8(x + rng.normal(0.0, std, size=x.shape).astype(np.float32))


def _jpeg(img: np.ndarray, quality: float, rng: np.random.Generator | None) -> np.ndarray:
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="JPEG", quality=int(quality))
    buf.seek(0)
    with Image.open(buf) as im:
        return np.asarray(im.convert("RGB"), dtype=np.uint8).copy()


@dataclass(frozen=True)
class Corruption:
    name: str
    models: str
    param_name: str
    params: tuple[float, ...]  # one value per level 1..MAX_LEVEL
    fn: Callable[[np.ndarray, float, np.random.Generator | None], np.ndarray]
    random: bool = False

    def param(self, level: int) -> float:
        check_level(level)
        if level == 0:
            raise ValueError("level 0 is the clean image and has no parameter")
        return self.params[level - 1]


CORRUPTIONS: dict[str, Corruption] = {
    c.name: c
    for c in (
        Corruption(
            "brightness",
            "illumination change",
            "value_shift",
            (0.1, 0.2, 0.3, 0.4, 0.5),
            _brightness,
        ),
        Corruption("gaussian_blur", "defocus blur", "sigma_px", (1, 2, 3, 4, 6), _gaussian_blur),
        Corruption(
            "gaussian_noise",
            "sensor noise",
            "std",
            (0.08, 0.12, 0.18, 0.26, 0.38),
            _gaussian_noise,
            True,
        ),
        Corruption("jpeg_compression", "compression", "quality", (25, 18, 15, 10, 7), _jpeg),
    )
}
CORRUPTION_NAMES = tuple(CORRUPTIONS)


def check_level(level: int) -> None:
    if not isinstance(level, (int, np.integer)) or not 0 <= level <= MAX_LEVEL:
        raise ValueError(f"severity level must be an int in [0, {MAX_LEVEL}], got {level!r}")


def levels_for_count(n: int) -> tuple[int, ...]:
    """``n`` severity levels spread evenly over 1..5: 1 -> (5,), 2 -> (1, 5),
    3 -> (1, 3, 5), 5 -> (1, 2, 3, 4, 5). Going from 3 to 5 adds exactly levels 2 and 4,
    so a resumed run only computes the new conditions."""
    if not 1 <= n <= MAX_LEVEL:
        raise ValueError(f"number of severities must be in [1, {MAX_LEVEL}], got {n}")
    if n == 1:
        return (MAX_LEVEL,)
    return tuple(sorted({int(round(x)) for x in np.linspace(1, MAX_LEVEL, n)}))


def apply_corruption(img: np.ndarray, name: str, level: int, image_id: str) -> np.ndarray:
    """Return a corrupted copy of ``img`` (uint8, H x W x 3). Level 0 returns a copy."""
    if img.dtype != np.uint8 or img.ndim != 3 or img.shape[2] != 3:
        raise ValueError(f"expected uint8 (H, W, 3), got {img.dtype} {img.shape}")
    if name not in CORRUPTIONS:
        raise KeyError(f"unknown corruption {name!r}; available: {CORRUPTION_NAMES}")
    check_level(level)
    if level == 0:
        return img.copy()
    c = CORRUPTIONS[name]
    rng = item_rng("corruption", image_id, name, int(level)) if c.random else None
    return c.fn(img, c.param(level), rng)


def apply_corruption_batch(
    images: np.ndarray, image_ids: Sequence[str], name: str, level: int
) -> np.ndarray:
    if len(images) != len(image_ids):
        raise ValueError("images and image_ids differ in length")
    if level == 0:
        return images.copy()
    return np.stack(
        [apply_corruption(im, name, level, i) for im, i in zip(images, image_ids, strict=True)]
    )


def conditions(names: Iterable[str], levels: Iterable[int]) -> list[tuple[str, int]]:
    """All scored conditions: ("none", 0) for clean, then every (family, level)."""
    levels = sorted(set(levels))
    for lv in levels:
        check_level(lv)
        if lv == 0:
            raise ValueError("list only shifted levels (>= 1); clean is always included")
    out = [("none", 0)]
    for n in names:
        if n not in CORRUPTIONS:
            raise KeyError(f"unknown corruption {n!r}; available: {CORRUPTION_NAMES}")
        out.extend((n, lv) for lv in levels)
    return out


def severity_table(names: Iterable[str], levels: Iterable[int]) -> dict[str, dict]:
    """Exact parameters used, for the run metadata."""
    return {
        n: {
            "models": CORRUPTIONS[n].models,
            "param_name": CORRUPTIONS[n].param_name,
            "levels": {str(lv): CORRUPTIONS[n].param(lv) for lv in sorted(set(levels))},
        }
        for n in names
    }
