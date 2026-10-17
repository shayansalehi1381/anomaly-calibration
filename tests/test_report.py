import numpy as np
import pandas as pd
import pytest

from anomcal.report import clean_auroc, format_markdown, main, sanity_check, summarise
from anomcal.scores_io import make_frame, write_atomic


def frame(method, seed, cls, protocol, sep, corruption="none", severity=0, n=20):
    rng = np.random.default_rng(10 * seed + ("candle", "pcb1").index(cls))
    labels = [0] * n + [1] * n
    scores = np.r_[rng.normal(0, 1, n), rng.normal(sep, 1, n)]
    return make_frame(
        image_ids=[f"{cls}/{'anomaly' if lab else 'normal'}/{i}" for i, lab in enumerate(labels)],
        cls=cls, labels=labels,
        roles=["anomaly" if lab else "heldout_normal" for lab in labels],
        official_splits=["test"] * (2 * n), protocol=protocol, method=method, seed=seed,
        corruption=corruption, severity=severity, scores=scores,
    )  # fmt: skip


@pytest.fixture
def scores():
    parts = []
    for seed in (0, 1, 2):
        for cls in ("candle", "pcb1"):
            parts.append(frame("patchcore", seed, cls, "official", sep=4.0))
            parts.append(frame("dinov2_knn", seed, cls, "official", sep=0.5))
            parts.append(frame("patchcore", seed, cls, "calib", sep=4.0))
            parts.append(frame("patchcore", seed, cls, "calib", 0.0, "gaussian_blur", 1))
    return pd.concat(parts, ignore_index=True)


def test_clean_auroc_ignores_corrupted_rows(scores):
    per = clean_auroc(scores)
    assert len(per) == 3 * 2 * 3  # seeds x classes x (pc-official, dino-official, pc-calib)
    assert per["auroc"].between(0, 100).all()


def test_summary_mean_std_and_class_mean(scores):
    s = summarise(clean_auroc(scores))
    row = s[(s.protocol == "official") & (s.method == "patchcore") & (s["class"] == "MEAN")]
    assert row["seeds"].iloc[0] == 3 and row["mean"].iloc[0] > 95
    md = format_markdown(s, "official")
    assert "| **MEAN** |" in md and "±" in md


def test_sanity_check_flags_large_gaps(scores):
    lines = sanity_check(summarise(clean_auroc(scores)))
    pc = next(x for x in lines if x.startswith("patchcore"))
    dino = next(x for x in lines if x.startswith("dinov2_knn"))
    assert "consistent" in pc or "FLAG" in pc
    assert "FLAG" in dino  # sep 0.5 -> AUROC ~64, far below 95.3


def test_cli(scores, tmp_path, capsys):
    for (proto, method, seed), g in scores.groupby(["protocol", "method", "seed"]):
        write_atomic(g.reset_index(drop=True), tmp_path / f"{proto}_{method}_seed{seed}.parquet")
    assert main([str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "official VisA 1cls split" in out and "Sanity check vs SPARC" in out
    with pytest.raises(FileNotFoundError):
        main([str(tmp_path / "empty")])
