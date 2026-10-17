"""Split protocol (documented in docs/SPLIT_PROTOCOL.md).

Two protocols, both per class:

``calib`` (our protocol, used for everything about calibration and shift)
    The official *train* normals are shuffled with a seed and cut in two:
    a fraction goes into the **memory bank** (what the detector is fit on), and the
    rest becomes part of the **held-out normal pool**. The official *test* normals
    join that pool. Anomalies (official test only) are used for evaluation only.
    Calibration/test partitions are drawn later, on CPU, from the held-out pool
    (:func:`partition_heldout`).

``official`` (literature comparison only)
    Memory bank = all official train normals; evaluation = the official test set.
    Only image AUROC on clean images is reported from it, clearly labelled.

Why the held-out pool is so important: a calibration image that the detector has
already *seen* in its memory bank has an artificially low score (its nearest
neighbour is itself, at distance 0). Calibrating on such scores would set the
threshold too low, and the false-alarm guarantee would silently break. So the memory
bank and the held-out pool must be strictly disjoint; :func:`assert_disjoint` checks
this by image id and by resolved file path, and the extraction script calls it before
scoring anything.
"""

from __future__ import annotations

import os
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path

import numpy as np

from anomcal.data.visa import VisaRecord
from anomcal.seeding import item_rng

PROTOCOLS = ("calib", "official")
ROLE_MEMORY = "memory"
ROLE_HELDOUT = "heldout_normal"
ROLE_ANOMALY = "anomaly"
DEFAULT_MEMORY_FRACTION = 0.5


class LeakageError(AssertionError):
    """Two sets that must be disjoint share an image (data leakage)."""


@dataclass(frozen=True)
class ClassSplit:
    cls: str
    protocol: str
    seed: int
    memory_fraction: float | None  # None for the official protocol (all train normals)
    memory: tuple[str, ...]
    heldout_normal: tuple[str, ...]
    anomaly: tuple[str, ...]
    max_images: int | None = field(default=None)

    @property
    def eval_ids(self) -> tuple[str, ...]:
        """Images that get scored (under every corruption): held-out normals + anomalies."""
        return self.heldout_normal + self.anomaly

    def role_of(self) -> dict[str, str]:
        roles = {i: ROLE_MEMORY for i in self.memory}
        roles.update({i: ROLE_HELDOUT for i in self.heldout_normal})
        roles.update({i: ROLE_ANOMALY for i in self.anomaly})
        return roles

    def limited(self, max_images: int | None) -> ClassSplit:
        """Keep at most ``max_images`` per role (for smoke tests). Order is preserved,
        so the subset is a deterministic function of the seed."""
        if max_images is None:
            return self
        if max_images < 1:
            raise ValueError("max_images must be >= 1")
        return ClassSplit(
            cls=self.cls,
            protocol=self.protocol,
            seed=self.seed,
            memory_fraction=self.memory_fraction,
            memory=self.memory[:max_images],
            heldout_normal=self.heldout_normal[:max_images],
            anomaly=self.anomaly[:max_images],
            max_images=max_images,
        )

    def to_dict(self) -> dict:
        return {
            "class": self.cls,
            "protocol": self.protocol,
            "seed": self.seed,
            "memory_fraction": self.memory_fraction,
            "max_images": self.max_images,
            "memory": list(self.memory),
            "heldout_normal": list(self.heldout_normal),
            "anomaly": list(self.anomaly),
        }


def _class_records(records: Iterable[VisaRecord], cls: str) -> list[VisaRecord]:
    # Sort by id so the result never depends on the row order of the CSV file.
    rows = sorted((r for r in records if r.cls == cls), key=lambda r: r.image_id)
    if not rows:
        raise ValueError(f"no records for class {cls!r}")
    return rows


def n_memory(n_train: int, memory_fraction: float) -> int:
    """Number of memory-bank images; at least 1 and leaves at least 1 held out."""
    if not 0.0 < memory_fraction < 1.0:
        raise ValueError(f"memory_fraction must be in (0, 1), got {memory_fraction}")
    if n_train < 2:
        raise ValueError("need at least 2 train normals to split")
    return min(max(round(n_train * memory_fraction), 1), n_train - 1)


def make_calib_split(
    records: Iterable[VisaRecord],
    cls: str,
    seed: int,
    memory_fraction: float = DEFAULT_MEMORY_FRACTION,
) -> ClassSplit:
    rows = _class_records(records, cls)
    train_normals = [r.image_id for r in rows if r.official_split == "train" and r.label == 0]
    test_normals = [r.image_id for r in rows if r.official_split == "test" and r.label == 0]
    anomalies = [r.image_id for r in rows if r.label == 1]

    # One RNG per (purpose, class, seed): the split of class A does not change when
    # class B is added or removed, or when classes are processed in another order.
    rng = item_rng("memory-split", cls, seed)
    order = rng.permutation(len(train_normals))
    shuffled = [train_normals[i] for i in order]
    k = n_memory(len(shuffled), memory_fraction)
    memory, held_train = shuffled[:k], shuffled[k:]

    # Shuffle the held-out pool and anomalies too, so a --max-images subset is random
    # rather than "the first files on disk".
    held = held_train + test_normals
    held = [held[i] for i in item_rng("heldout-order", cls, seed).permutation(len(held))]
    anom = [anomalies[i] for i in item_rng("anomaly-order", cls, seed).permutation(len(anomalies))]
    split = ClassSplit(cls, "calib", seed, memory_fraction, tuple(memory), tuple(held), tuple(anom))
    assert_disjoint(split)
    return split


def make_official_split(records: Iterable[VisaRecord], cls: str, seed: int = 0) -> ClassSplit:
    rows = _class_records(records, cls)
    memory = tuple(r.image_id for r in rows if r.official_split == "train")
    test_normals = tuple(r.image_id for r in rows if r.official_split == "test" and r.label == 0)
    anomalies = tuple(r.image_id for r in rows if r.label == 1)
    split = ClassSplit(cls, "official", seed, None, memory, test_normals, anomalies)
    assert_disjoint(split)
    return split


def make_split(
    records: Sequence[VisaRecord],
    cls: str,
    protocol: str,
    seed: int,
    memory_fraction: float = DEFAULT_MEMORY_FRACTION,
) -> ClassSplit:
    if protocol == "calib":
        return make_calib_split(records, cls, seed, memory_fraction)
    if protocol == "official":
        return make_official_split(records, cls, seed)
    raise ValueError(f"unknown protocol {protocol!r}, expected one of {PROTOCOLS}")


def assert_disjoint(
    split: ClassSplit,
    records: Iterable[VisaRecord] | None = None,
    root: Path | None = None,
) -> None:
    """Raise :class:`LeakageError` if any two roles share an image.

    Checks image ids always. With ``records`` it also checks that the memory bank holds
    only normal *train* images and that no anomaly is labelled normal. With ``records``
    and ``root`` it also compares resolved file paths, which catches the same file
    listed under two different ids.
    """
    groups = {
        ROLE_MEMORY: split.memory,
        ROLE_HELDOUT: split.heldout_normal,
        ROLE_ANOMALY: split.anomaly,
    }
    for name, ids in groups.items():
        if len(set(ids)) != len(ids):
            raise LeakageError(f"{split.cls}: duplicate ids inside {name}")
    names = list(groups)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            common = set(groups[a]) & set(groups[b])
            if common:
                raise LeakageError(
                    f"{split.cls}: {len(common)} images in both {a} and {b}, "
                    f"e.g. {sorted(common)[:3]}"
                )
    if records is None:
        return
    by_id = {r.image_id: r for r in records}
    unknown = [i for ids in groups.values() for i in ids if i not in by_id]
    if unknown:
        raise LeakageError(f"{split.cls}: ids not in the split file, e.g. {unknown[:3]}")
    bad_mem = [i for i in split.memory if by_id[i].label != 0 or by_id[i].official_split != "train"]
    if bad_mem:
        raise LeakageError(f"{split.cls}: memory bank contains non-train/non-normal {bad_mem[:3]}")
    if any(by_id[i].label != 0 for i in split.heldout_normal):
        raise LeakageError(f"{split.cls}: anomaly inside the held-out normal pool")
    if any(by_id[i].label != 1 for i in split.anomaly):
        raise LeakageError(f"{split.cls}: normal image inside the anomaly set")
    if root is None:
        return
    seen: dict[str, str] = {}
    for name, ids in groups.items():
        for i in ids:
            key = os.path.normcase(str(by_id[i].image_path(root).resolve()))
            if key in seen and seen[key] != name:
                raise LeakageError(f"{split.cls}: file {key} is in both {seen[key]} and {name}")
            seen[key] = name


def partition_heldout(
    heldout_ids: Sequence[str],
    cal_fraction: float | Fraction,
    rng: np.random.Generator,
) -> tuple[list[str], list[str]]:
    """Randomly split the held-out normal pool into (calibration, test). Used in Phase 3,
    repeated R times with different generators."""
    if not 0 < cal_fraction < 1:
        raise ValueError("cal_fraction must be in (0, 1)")
    n = len(heldout_ids)
    if n < 2:
        raise ValueError("need at least 2 held-out images")
    n_cal = min(max(int(round(n * float(cal_fraction))), 1), n - 1)
    perm = rng.permutation(n)
    cal = [heldout_ids[i] for i in perm[:n_cal]]
    test = [heldout_ids[i] for i in perm[n_cal:]]
    return cal, test
