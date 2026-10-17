"""Image and mask loading at a fixed *canonical* resolution.

Pipeline for every image (see docs/SPLIT_PROTOCOL.md, "Image pipeline")::

    JPEG on disk --load + RGB--> resize to canonical S x S --corrupt--> scorer

Corruptions are applied at this one canonical resolution (default 256 x 256) for every
method, so that "blur sigma = 2 px" means the same thing for all detectors. Each
scorer then does its own model-specific preprocessing (crop, resize to 224 or a
multiple of 14, normalisation) starting from the same corrupted uint8 image.

Resizing to a square ignores the aspect ratio (VisA images are not square). This is
the common practice in anomalib-style VisA pipelines; it is documented as a choice.
"""

from __future__ import annotations

from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

DEFAULT_IMAGE_SIZE = 256


def load_image(path: Path, size: int = DEFAULT_IMAGE_SIZE) -> np.ndarray:
    """Load an image as uint8 RGB of shape (size, size, 3).

    Pillow's BILINEAR resize widens its kernel when downsampling (it is area-aware),
    so large VisA images (~1.5k px) are antialiased rather than aliased.
    """
    with Image.open(path) as im:
        im = im.convert("RGB").resize((size, size), Image.Resampling.BILINEAR)
        return np.asarray(im, dtype=np.uint8).copy()


def load_mask(path: Path | None, size: int = DEFAULT_IMAGE_SIZE) -> np.ndarray:
    """Load a VisA mask as bool (size, size). Non-zero pixels are defects.

    VisA masks store a defect-class id per pixel, so we binarise with ``!= 0``.
    NEAREST resampling keeps the mask binary (no interpolated in-between values).
    ``None`` (normal image) gives an all-False mask.
    """
    if path is None:
        return np.zeros((size, size), dtype=bool)
    with Image.open(path) as im:
        im = im.convert("L").resize((size, size), Image.Resampling.NEAREST)
        return np.asarray(im) != 0


def load_images(
    paths: Sequence[Path], size: int = DEFAULT_IMAGE_SIZE, num_workers: int = 0
) -> np.ndarray:
    """Load many images into one (N, size, size, 3) uint8 array, in input order.

    ``num_workers > 0`` uses a thread pool. Threads (not processes) work the same on
    Windows and Linux and need no pickling. JPEG decoding in Pillow releases the GIL,
    so threads do speed it up. The default 0 means plain sequential loading.
    """
    out = np.empty((len(paths), size, size, 3), dtype=np.uint8)
    if num_workers <= 0:
        for i, p in enumerate(paths):
            out[i] = load_image(p, size)
        return out
    with ThreadPoolExecutor(max_workers=num_workers) as pool:
        for i, img in enumerate(pool.map(lambda p: load_image(p, size), paths)):
            out[i] = img
    return out
