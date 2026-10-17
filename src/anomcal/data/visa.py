"""VisA adapter: read the official split file and locate images and masks on disk.

VisA ships as one folder per class::

    <root>/candle/Data/Images/Normal/0000.JPG      normal images
    <root>/candle/Data/Images/Anomaly/000.JPG      anomalous images
    <root>/candle/Data/Masks/Anomaly/000.png       pixel masks (only for anomalies;
                                                   class-id coded, non-zero = defect)
    <root>/split_csv/1cls.csv                      official "1-class" split

The official split file has one row per image (``object,split,label,image,mask``) with
paths relative to ``<root>`` in POSIX form. We never hard-code a path separator: a
relative path is split on ``/`` and re-joined with :mod:`pathlib`, which gives correct
Windows paths on the laptop and POSIX paths in CI.

Command-line use (on the laptop) to verify the dataset and print per-class counts::

    python -m anomcal.data.visa --data-root data/visa
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import sys
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

VISA_CLASSES: tuple[str, ...] = (
    "candle",
    "capsules",
    "cashew",
    "chewinggum",
    "fryum",
    "macaroni1",
    "macaroni2",
    "pcb1",
    "pcb2",
    "pcb3",
    "pcb4",
    "pipe_fryum",
)

# Verified in Phase 0/1 by reading split_csv/1cls.csv from the public
# amazon-science/spot-diff repository (commit 2a692ab, 2022-09-22):
# class -> (train normal, test normal, test anomaly). Totals: 8659 / 962 / 1200.
EXPECTED_1CLS_COUNTS: dict[str, tuple[int, int, int]] = {
    "candle": (900, 100, 100),
    "capsules": (542, 60, 100),
    "cashew": (450, 50, 100),
    "chewinggum": (453, 50, 100),
    "fryum": (450, 50, 100),
    "macaroni1": (900, 100, 100),
    "macaroni2": (900, 100, 100),
    "pcb1": (904, 100, 100),
    "pcb2": (901, 100, 100),
    "pcb3": (905, 101, 100),
    "pcb4": (904, 101, 100),
    "pipe_fryum": (450, 50, 100),
}
# SHA-256 of that file (LF line endings). A different hash is only a warning: the copy
# inside the dataset tar may differ in line endings; the per-class counts are what matter.
EXPECTED_1CLS_SHA256 = "a48557e6033318cb90556f706196bc9d247a776a23ea51aecee5a80dd0332995"

SPLIT_CSV_HEADER = ("object", "split", "label", "image", "mask")
SPLIT_CSV_RELPATH = PurePosixPath("split_csv/1cls.csv")


class VisaFormatError(ValueError):
    """The split file or the folder layout does not look like VisA."""


@dataclass(frozen=True)
class VisaRecord:
    """One image of the official split.

    ``image_id`` is ``"<class>/<normal|anomaly>/<file stem>"``. The label is part of
    the id because VisA reuses file numbers across the Normal and Anomaly folders
    (``Normal/0000.JPG`` and ``Anomaly/000.JPG`` can share digits).
    """

    image_id: str
    cls: str
    official_split: str  # "train" | "test"
    label: int  # 0 = normal, 1 = anomaly
    image_relpath: str  # POSIX, relative to the dataset root
    mask_relpath: str | None  # POSIX, relative; None for normal images

    def image_path(self, root: Path) -> Path:
        return to_local_path(root, self.image_relpath)

    def mask_path(self, root: Path) -> Path | None:
        return None if self.mask_relpath is None else to_local_path(root, self.mask_relpath)


def to_local_path(root: Path, posix_relpath: str) -> Path:
    """Join a POSIX relative path from the CSV onto a local root (Windows-safe)."""
    rel = PurePosixPath(posix_relpath)
    if rel.is_absolute() or ".." in rel.parts:
        raise VisaFormatError(f"refusing path outside the dataset root: {posix_relpath!r}")
    return Path(root).joinpath(*rel.parts)


def make_image_id(cls: str, label: int, image_relpath: str) -> str:
    kind = "anomaly" if label == 1 else "normal"
    return f"{cls}/{kind}/{PurePosixPath(image_relpath).stem}"


def read_split_csv(path: Path) -> list[VisaRecord]:
    """Parse and validate a VisA ``1cls.csv``-style split file."""
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = tuple(next(reader, ()))
        if header != SPLIT_CSV_HEADER:
            raise VisaFormatError(f"unexpected header {header}, expected {SPLIT_CSV_HEADER}")
        records = [_parse_row(row, lineno) for lineno, row in enumerate(reader, start=2)]
    ids = Counter(r.image_id for r in records)
    dupes = [i for i, c in ids.items() if c > 1]
    if dupes:
        raise VisaFormatError(f"duplicate image ids, e.g. {dupes[:3]}")
    return records


def _parse_row(row: Sequence[str], lineno: int) -> VisaRecord:
    if len(row) != 5:
        raise VisaFormatError(f"line {lineno}: expected 5 fields, got {len(row)}")
    cls, split, label_str, image, mask = (x.strip() for x in row)
    if split not in ("train", "test"):
        raise VisaFormatError(f"line {lineno}: unknown split {split!r}")
    if label_str not in ("normal", "anomaly"):
        raise VisaFormatError(f"line {lineno}: unknown label {label_str!r}")
    label = int(label_str == "anomaly")
    if label == 1 and not mask:
        raise VisaFormatError(f"line {lineno}: anomalous image without a mask")
    if label == 0 and mask:
        raise VisaFormatError(f"line {lineno}: normal image with a mask")
    if label == 1 and split == "train":
        raise VisaFormatError(f"line {lineno}: anomaly in the train split (1cls has none)")
    if PurePosixPath(image).parts[:1] != (cls,):
        raise VisaFormatError(f"line {lineno}: image path {image!r} not under class {cls!r}")
    return VisaRecord(
        image_id=make_image_id(cls, label, image),
        cls=cls,
        official_split=split,
        label=label,
        image_relpath=image,
        mask_relpath=mask or None,
    )


def records_from_counts(counts: dict[str, tuple[int, int, int]]) -> list[VisaRecord]:
    """Synthetic records with given per-class counts (no files): lets planning and tests
    use the real VisA split sizes without the dataset, e.g. in the cloud."""
    recs = []
    for cls, (n_tr, n_te, n_an) in counts.items():
        for i in range(n_tr + n_te):
            rel = f"{cls}/Data/Images/Normal/{i:04d}.JPG"
            split = "train" if i < n_tr else "test"
            recs.append(VisaRecord(make_image_id(cls, 0, rel), cls, split, 0, rel, None))
        for i in range(n_an):
            rel = f"{cls}/Data/Images/Anomaly/{i:03d}.JPG"
            mask = f"{cls}/Data/Masks/Anomaly/{i:03d}.png"
            recs.append(VisaRecord(make_image_id(cls, 1, rel), cls, "test", 1, rel, mask))
    return recs


def resolve_visa_root(data_root: Path) -> Path:
    """Accept either ``data/visa`` or ``data/visa/VisA`` (depends on how the tar was extracted).

    The root is the folder that directly contains the class folders.
    """
    data_root = Path(data_root)
    for candidate in (data_root, data_root / "VisA", data_root / "visa"):
        if any((candidate / c).is_dir() for c in VISA_CLASSES):
            return candidate
    raise VisaFormatError(
        f"no VisA class folders (e.g. {data_root / 'candle'}) found under {data_root}. "
        "Extract VisA_20220922.tar there first."
    )


def find_split_csv(root: Path, explicit: Path | None = None) -> Path:
    if explicit is not None:
        if not Path(explicit).is_file():
            raise FileNotFoundError(f"split file not found: {explicit}")
        return Path(explicit)
    path = to_local_path(root, str(SPLIT_CSV_RELPATH))
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} not found. Pass --split-csv, or get the official file from "
            "https://github.com/amazon-science/spot-diff/blob/main/split_csv/1cls.csv"
        )
    return path


def counts_by_class(records: Iterable[VisaRecord]) -> dict[str, tuple[int, int, int]]:
    """class -> (train normal, test normal, test anomaly)."""
    c = Counter((r.cls, r.official_split, r.label) for r in records)
    classes = sorted({k[0] for k in c})
    return {k: (c[(k, "train", 0)], c[(k, "test", 0)], c[(k, "test", 1)]) for k in classes}


def compare_with_expected(
    counts: dict[str, tuple[int, int, int]],
    expected: dict[str, tuple[int, int, int]] = EXPECTED_1CLS_COUNTS,
) -> list[str]:
    """Human-readable mismatches against the verified official counts (empty = match)."""
    problems = []
    for cls in sorted(set(counts) | set(expected)):
        got, want = counts.get(cls), expected.get(cls)
        if got != want:
            problems.append(f"{cls}: got {got}, expected {want}")
    return problems


def missing_files(root: Path, records: Iterable[VisaRecord]) -> list[str]:
    """Relative paths listed in the split but absent on disk."""
    missing = []
    for r in records:
        if not r.image_path(root).is_file():
            missing.append(r.image_relpath)
        mask = r.mask_path(root)
        if mask is not None and not mask.is_file():
            missing.append(r.mask_relpath)
    return missing


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def format_counts_table(counts: dict[str, tuple[int, int, int]]) -> str:
    lines = [f"{'class':<12} {'train_normal':>12} {'test_normal':>11} {'test_anomaly':>12}"]
    totals = [0, 0, 0]
    for cls, row in counts.items():
        lines.append(f"{cls:<12} {row[0]:>12} {row[1]:>11} {row[2]:>12}")
        totals = [t + x for t, x in zip(totals, row, strict=True)]
    lines.append(f"{'TOTAL':<12} {totals[0]:>12} {totals[1]:>11} {totals[2]:>12}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Verify a local VisA copy and print split counts.")
    p.add_argument("--data-root", type=Path, required=True, help="folder with VisA, e.g. data/visa")
    p.add_argument(
        "--split-csv", type=Path, default=None, help="default: <root>/split_csv/1cls.csv"
    )
    p.add_argument("--skip-file-check", action="store_true", help="do not stat every image")
    args = p.parse_args(argv)

    root = resolve_visa_root(args.data_root)
    csv_path = find_split_csv(root, args.split_csv)
    records = read_split_csv(csv_path)
    counts = counts_by_class(records)
    print(f"dataset root : {root}")
    print(f"split file   : {csv_path}")
    digest = sha256_file(csv_path)
    same = "matches" if digest == EXPECTED_1CLS_SHA256 else "DIFFERS from"
    print(f"sha256       : {digest} ({same} the verified spot-diff copy)")
    print(format_counts_table(counts))

    ok = True
    problems = compare_with_expected(counts)
    if problems:
        ok = False
        print("\nCOUNT MISMATCH vs the verified official split:", *problems, sep="\n  ")
    else:
        print("\ncounts match the verified official split")
    if not args.skip_file_check:
        missing = missing_files(root, records)
        if missing:
            ok = False
            print(f"\n{len(missing)} files listed in the split are missing, e.g.:")
            print(*missing[:10], sep="\n  ")
        else:
            print("all images and masks listed in the split exist on disk")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
