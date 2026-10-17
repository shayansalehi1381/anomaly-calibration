"""A tiny synthetic stand-in for VisA, with the same folder layout and split file.

Used by the tests and CI (no real data there) and for a dry run of the extraction
script on any machine::

    python -m anomcal.data.fake_visa outputs/fake_visa

Normal images are smooth class-coloured textures with mild noise. Anomalies are the
same texture plus a bright disk, and the mask marks the disk with a non-zero class
id, as in VisA. Everything is seeded, so the dataset is identical on every run.
"""

from __future__ import annotations

import argparse
import csv
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from PIL import Image

from anomcal.seeding import item_rng

DEFAULT_FAKE_CLASSES = ("candle", "pcb1")


def _texture(rng: np.random.Generator, h: int, w: int, base: np.ndarray) -> np.ndarray:
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    fx, fy, phase = rng.uniform(0.05, 0.15), rng.uniform(0.05, 0.15), rng.uniform(0, 2 * np.pi)
    wave = 25.0 * np.sin(fx * xx + fy * yy + phase)[..., None]
    noise = rng.normal(0.0, 6.0, size=(h, w, 3))
    return np.clip(base[None, None, :] + wave + noise, 0, 255).astype(np.uint8)


def make_fake_visa(
    root: Path,
    classes: Sequence[str] = DEFAULT_FAKE_CLASSES,
    n_train: int = 12,
    n_test_normal: int = 4,
    n_test_anomaly: int = 4,
    size: tuple[int, int] = (96, 64),  # (width, height): non-square on purpose, like VisA
    seed: int = 0,
) -> Path:
    """Write a fake VisA tree under ``root`` and return ``root``."""
    root = Path(root)
    w, h = size
    rows: list[list[str]] = []
    for ci, cls in enumerate(classes):
        base = np.array([60 + 40 * ci, 120, 180 - 40 * ci], dtype=np.float32) % 256
        norm_dir = root / cls / "Data" / "Images" / "Normal"
        anom_dir = root / cls / "Data" / "Images" / "Anomaly"
        mask_dir = root / cls / "Data" / "Masks" / "Anomaly"
        for d in (norm_dir, anom_dir, mask_dir):
            d.mkdir(parents=True, exist_ok=True)

        for i in range(n_train + n_test_normal):
            img = _texture(item_rng("fake", seed, cls, "normal", i), h, w, base)
            name = f"{i:04d}.JPG"
            Image.fromarray(img).save(norm_dir / name, quality=95)
            split = "train" if i < n_train else "test"
            rows.append([cls, split, "normal", f"{cls}/Data/Images/Normal/{name}", ""])

        for i in range(n_test_anomaly):
            rng = item_rng("fake", seed, cls, "anomaly", i)
            img = _texture(rng, h, w, base)
            cy, cx = rng.integers(h // 4, 3 * h // 4), rng.integers(w // 4, 3 * w // 4)
            r = max(3, min(h, w) // 6)
            yy, xx = np.mgrid[0:h, 0:w]
            disk = (yy - cy) ** 2 + (xx - cx) ** 2 <= r**2
            img[disk] = 250
            mask = np.where(disk, 3, 0).astype(np.uint8)  # class-id coded like VisA
            name = f"{i:03d}"
            Image.fromarray(img).save(anom_dir / f"{name}.JPG", quality=95)
            Image.fromarray(mask).save(mask_dir / f"{name}.png")
            rows.append(
                [
                    cls,
                    "test",
                    "anomaly",
                    f"{cls}/Data/Images/Anomaly/{name}.JPG",
                    f"{cls}/Data/Masks/Anomaly/{name}.png",
                ]
            )

    split_dir = root / "split_csv"
    split_dir.mkdir(parents=True, exist_ok=True)
    with open(split_dir / "1cls.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(["object", "split", "label", "image", "mask"])
        writer.writerows(rows)
    return root


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Write a tiny synthetic VisA-like dataset.")
    p.add_argument("root", type=Path)
    p.add_argument("--classes", nargs="+", default=list(DEFAULT_FAKE_CLASSES))
    p.add_argument("--n-train", type=int, default=12)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)
    make_fake_visa(args.root, args.classes, n_train=args.n_train, seed=args.seed)
    print(f"fake VisA written to {args.root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
