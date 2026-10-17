"""Score files: schema, atomic writes, validation and consolidation.

The full schema is documented in docs/score_schema.md. In short, every score file is
a long-format table with one row per (image, condition):

    image_id, class, label, role, official_split, protocol, method, seed,
    corruption, severity, score

The extraction script writes one small *chunk* file per
(protocol, method, seed, class, corruption, severity). A chunk that exists is
complete, because it is written to a temporary file first and then renamed
(``os.replace`` is atomic on both Windows and POSIX for files on the same disk).
This is what makes the run resumable: after an interruption, finished chunks are
skipped and a half-written chunk never looks finished.

At the end, chunks are consolidated into one Parquet file per (protocol, method, seed),
which is the small file that gets committed to ``results/scores/``.
"""

from __future__ import annotations

import os
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd

SCHEMA: dict[str, str] = {
    "image_id": "string",
    "class": "string",
    "label": "int8",
    "role": "string",
    "official_split": "string",
    "protocol": "string",
    "method": "string",
    "seed": "int32",
    "corruption": "string",
    "severity": "int8",
    "score": "float32",
}
COLUMNS = tuple(SCHEMA)
KEY = ("protocol", "method", "seed", "class", "image_id", "corruption", "severity")
VALID_ROLES = {"heldout_normal", "anomaly"}
VALID_PROTOCOLS = {"calib", "official"}


class SchemaError(ValueError):
    pass


def make_frame(
    *,
    image_ids: list[str],
    cls: str,
    labels: list[int],
    roles: list[str],
    official_splits: list[str],
    protocol: str,
    method: str,
    seed: int,
    corruption: str,
    severity: int,
    scores: np.ndarray,
) -> pd.DataFrame:
    n = len(image_ids)
    df = pd.DataFrame(
        {
            "image_id": image_ids,
            "class": [cls] * n,
            "label": labels,
            "role": roles,
            "official_split": official_splits,
            "protocol": [protocol] * n,
            "method": [method] * n,
            "seed": [seed] * n,
            "corruption": [corruption] * n,
            "severity": [severity] * n,
            "score": np.asarray(scores, dtype=np.float32),
        }
    ).astype(SCHEMA)
    validate(df)
    return df


def validate(df: pd.DataFrame) -> None:
    """Raise :class:`SchemaError` unless ``df`` follows the documented schema."""
    if tuple(df.columns) != COLUMNS:
        raise SchemaError(f"columns {tuple(df.columns)} != {COLUMNS}")
    for col, dtype in SCHEMA.items():
        if dtype == "string":
            # pandas 2 vs 3 name string dtypes differently ("string", "str", object);
            # what matters is that the values are strings.
            ok = pd.api.types.is_string_dtype(df[col])
        else:
            ok = str(df[col].dtype) == dtype
        if not ok:
            raise SchemaError(f"column {col!r} has dtype {df[col].dtype}, expected {dtype}")
    if df[list(COLUMNS)].isna().any().any():
        raise SchemaError("missing values")
    if not np.isfinite(df["score"].to_numpy()).all():
        raise SchemaError("non-finite scores")
    if not set(df["label"].unique()) <= {0, 1}:
        raise SchemaError("label must be 0/1")
    if not set(df["role"].unique()) <= VALID_ROLES:
        raise SchemaError(f"unknown role(s) {set(df['role'].unique()) - VALID_ROLES}")
    if not set(df["protocol"].unique()) <= VALID_PROTOCOLS:
        raise SchemaError("unknown protocol")
    bad_role = ((df["role"] == "anomaly") != (df["label"] == 1)).any()
    if bad_role:
        raise SchemaError("role 'anomaly' must coincide with label 1")
    clean = df["corruption"] == "none"
    if ((df["severity"] == 0) != clean).any():
        raise SchemaError("severity 0 must coincide with corruption 'none'")
    if df.duplicated(list(KEY)).any():
        raise SchemaError("duplicate (protocol, method, seed, class, image, condition) rows")


def chunk_path(
    out_dir: Path, protocol: str, method: str, seed: int, cls: str, corruption: str, severity: int
) -> Path:
    return (
        Path(out_dir)
        / "chunks"
        / protocol
        / method
        / f"seed{seed}"
        / cls
        / f"{corruption}_s{severity}.parquet"
    )


def write_atomic(df: pd.DataFrame, path: Path) -> None:
    validate(df)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    df.to_parquet(tmp, index=False)
    os.replace(tmp, path)


def read_scores(paths: Iterable[Path]) -> pd.DataFrame:
    frames = [pd.read_parquet(p) for p in paths]
    if not frames:
        return pd.DataFrame({c: pd.Series(dtype=t) for c, t in SCHEMA.items()})
    df = pd.concat(frames, ignore_index=True).astype(SCHEMA)
    validate(df)
    return df


def consolidate(out_dir: Path, protocol: str, method: str, seed: int) -> Path | None:
    """Merge all chunks of one (protocol, method, seed) into a single sorted Parquet file."""
    root = Path(out_dir) / "chunks" / protocol / method / f"seed{seed}"
    parts = sorted(root.glob("*/*.parquet"))
    if not parts:
        return None
    df = read_scores(parts).sort_values(["class", "corruption", "severity", "image_id"])
    dest = Path(out_dir) / "scores" / f"{protocol}_{method}_seed{seed}.parquet"
    write_atomic(df.reset_index(drop=True), dest)
    return dest
