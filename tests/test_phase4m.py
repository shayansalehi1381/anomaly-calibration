"""Phase 4b (small-m recalibration) on synthetic scores with a known answer."""

import numpy as np
import pandas as pd

from anomcal.calibration import tolerance_m_min
from anomcal.phase4 import rank_for
from anomcal.phase4m import MS, main, pool, run_all, run_condition
from anomcal.scores_io import make_frame, write_atomic


def _rows(rows, m, kind, alpha):
    return next(r for r in rows if (r["m"], r["kind"], r["alpha"]) == (m, kind, alpha))


def test_matches_exact_beta_binomial_law_and_alpha_is_met_on_average():
    """Corrupted scores N(d,1): with m=50, alpha=10%: k=46, E[FAR] = (m+1-k)/(m+1) = 5/51."""
    shifted = np.random.default_rng(0).normal(2.0, 1, 4000)
    rows = run_condition(shifted, np.array([9.0]), ("m", 0, "c"), n_partitions=60)
    r = _rows(rows, 50, "conformal", 0.10)
    far = r["sum_far"] / r["n"]
    assert abs(far - 5 / 51) < 0.004 and far <= 0.10
    assert abs(far - r["law_mean"]) < 0.004
    # share of draws with FAR > alpha agrees with the exact law (finite test half included)
    assert abs(r["violations"] / r["n"] - r["law_viol"]) < 0.07
    assert r["sum_tpr"] / r["n"] == 1.0  # the lone anomaly at 9.0 is above every threshold


def test_not_certifiable_is_explicit_and_matches_m_min():
    shifted = np.random.default_rng(1).normal(size=600)
    rows = run_condition(shifted, np.ones(3), ("m", 0, "c"), n_partitions=3)
    for m, kind, alpha in [(10, "conformal", 0.05), (25, "conformal", 0.01),
                           (10, "tolerance", 0.10), (100, "tolerance", 0.01)]:  # fmt: skip
        r = _rows(rows, m, kind, alpha)
        assert r["status"] == "not_certifiable" and r["n"] == 0 and "law_mean" not in r
    assert _rows(rows, 100, "conformal", 0.01)["status"] == "certified"
    # tolerance limit becomes certifiable exactly at m_min(alpha, delta)
    for alpha in (0.05, 0.10):
        m_min = tolerance_m_min(alpha, 0.05)
        assert rank_for("tolerance", m_min, alpha) is not None
        assert rank_for("tolerance", m_min - 1, alpha) is None
    assert (tolerance_m_min(0.10, 0.05), tolerance_m_min(0.05, 0.05)) == (29, 59)
    assert tolerance_m_min(0.01, 0.05) == 299


def test_deterministic_and_nested_draws():
    x, a = np.random.default_rng(2).normal(size=300), np.zeros(4)
    assert run_condition(x, a, ("m", 0, "c"), n_partitions=4) == run_condition(
        x, a, ("m", 0, "c"), n_partitions=4
    )
    assert run_condition(x, a, ("m", 1, "c"), n_partitions=4) != run_condition(
        x, a, ("m", 0, "c"), n_partitions=4
    )


def test_run_all_pool_and_cli(tmp_path):
    frames = []
    for method in ("baseline", "patchcore"):
        for seed in (0, 1):
            for cls in ("c1", "c2"):
                rng = np.random.default_rng(seed + len(cls) + len(method))
                ids = [f"{cls}/n/{i:03d}" for i in range(260)] + [
                    f"{cls}/a/{i:03d}" for i in range(10)
                ]
                labels = [0] * 260 + [1] * 10
                for corr, sev in (("none", 0), ("gaussian_blur", 1), ("gaussian_blur", 3)):
                    frames.append(make_frame(
                        image_ids=ids, cls=cls, labels=labels,
                        roles=["heldout_normal"] * 260 + ["anomaly"] * 10,
                        official_splits=["train"] * 270, protocol="calib", method=method,
                        seed=seed, corruption=corr, severity=sev,
                        scores=np.r_[rng.normal(sev, 1, 260), rng.normal(sev + 2, 1, 10)],
                    ))  # fmt: skip
    df = pd.concat(frames, ignore_index=True)
    raw = run_all(df, n_partitions=2)
    assert set(raw["corruption"]) == {"gaussian_blur"} and set(raw["m"]) == set(MS)
    pooled = pool(raw, ["method", "m", "kind", "alpha"])
    assert (pooled["draws"] > 0).all() and (pooled["far"].between(0, 1)).all()
    scores = tmp_path / "s"
    scores.mkdir()
    for (method, seed), g in df.groupby(["method", "seed"]):
        write_atomic(g, scores / f"calib_{method}_seed{seed}.parquet")
    out = tmp_path / "o"
    assert main(["--scores", str(scores), "--out", str(out), "--partitions", "2", "--boot", "30",
                 "--figures", str(tmp_path / "f")]) == 0  # fmt: skip
    assert (tmp_path / "f" / "phase4_small_m.png").exists()
    assert all(f.stat().st_size < 5 * 2**20 for f in out.iterdir())
