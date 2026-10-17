import numpy as np
import pandas as pd
import pytest

from anomcal.scores_io import (
    COLUMNS,
    SchemaError,
    chunk_path,
    consolidate,
    make_frame,
    read_scores,
    validate,
    write_atomic,
)


def frame(corruption="none", severity=0, seed=0, cls="candle", n=4) -> pd.DataFrame:
    labels = [0, 0, 1, 1][:n]
    return make_frame(
        image_ids=[f"{cls}/{'anomaly' if lab else 'normal'}/{i}" for i, lab in enumerate(labels)],
        cls=cls,
        labels=labels,
        roles=["anomaly" if lab else "heldout_normal" for lab in labels],
        official_splits=["train", "test", "test", "test"][:n],
        protocol="calib",
        method="dummy",
        seed=seed,
        corruption=corruption,
        severity=severity,
        scores=np.arange(n, dtype=np.float64),
    )


def test_make_frame_follows_schema():
    df = frame()
    assert tuple(df.columns) == COLUMNS
    assert df["score"].dtype == np.float32 and df["severity"].dtype == np.int8


@pytest.mark.parametrize(
    "mutate, msg",
    [
        (lambda d: d.assign(score=d["score"].where(d.index != 0, np.inf)), "non-finite"),
        (lambda d: d.assign(label=d["label"].where(d.index != 0, 1)), "role 'anomaly'"),
        (lambda d: d.assign(severity=np.int8(2)), "severity 0"),
        (lambda d: pd.concat([d, d], ignore_index=True), "duplicate"),
        (lambda d: d.assign(role="memory"), "unknown role"),
        (lambda d: d.drop(columns="seed"), "columns"),
        (lambda d: d.assign(seed=d["seed"].astype("int64")), "dtype"),
    ],
)
def test_validate_rejects_bad_frames(mutate, msg):
    with pytest.raises(SchemaError, match=msg):
        validate(mutate(frame()))


def test_atomic_write_and_read_roundtrip(tmp_path):
    p = chunk_path(tmp_path, "calib", "dummy", 0, "candle", "none", 0)
    write_atomic(frame(), p)
    assert p.exists() and not p.with_name(p.name + ".tmp").exists()
    pd.testing.assert_frame_equal(read_scores([p]), frame())


def test_failed_write_leaves_no_finished_chunk(tmp_path, monkeypatch):
    p = chunk_path(tmp_path, "calib", "dummy", 0, "candle", "none", 0)

    def boom(self, path, **kw):
        path.write_bytes(b"partial")  # simulate a crash mid-write
        raise OSError("disk full")

    monkeypatch.setattr(pd.DataFrame, "to_parquet", boom)
    with pytest.raises(OSError):
        write_atomic(frame(), p)
    assert not p.exists()  # only the .tmp file, which never counts as done


def test_chunk_path_layout(tmp_path):
    p = chunk_path(tmp_path, "calib", "patchcore", 2, "pcb1", "gaussian_blur", 3)
    assert p.relative_to(tmp_path).parts == (
        "chunks", "calib", "patchcore", "seed2", "pcb1", "gaussian_blur_s3.parquet"
    )  # fmt: skip


def test_consolidate(tmp_path):
    for cls in ("candle", "pcb1"):
        for c, s in (("none", 0), ("gaussian_blur", 1)):
            write_atomic(frame(c, s, cls=cls), chunk_path(tmp_path, "calib", "dummy", 0, cls, c, s))
    dest = consolidate(tmp_path, "calib", "dummy", 0)
    df = pd.read_parquet(dest)
    assert len(df) == 16 and dest.name == "calib_dummy_seed0.parquet"
    validate(df.astype({c: str(df[c].dtype) for c in df.columns}))
    assert consolidate(tmp_path, "calib", "dummy", 9) is None


def test_read_scores_empty():
    assert read_scores([]).empty
