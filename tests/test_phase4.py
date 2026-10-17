"""Phase 4 on synthetic scores with a known answer (no data, no GPU, no network)."""

import numpy as np
import pandas as pd
from scipy import stats

from anomcal.calibration import conformal_threshold, tolerance_threshold
from anomcal.phase4 import class_bootstrap, main, plot_x, rank_for, run_all, run_condition
from anomcal.scores_io import make_frame


def _cell(rows, setting, kind, alpha):
    return next(r for r in rows if (r["setting"], r["kind"], r["alpha"]) == (setting, kind, alpha))


def test_known_shift_no_recal_breaks_and_recal_restores():
    """Normals N(0,1) clean, N(d,1) corrupted: A -> 1 - Phi(z_{1-a} - d), B -> a."""
    rng, d = np.random.default_rng(0), 0.5
    clean, shifted = rng.normal(0, 1, 20000), rng.normal(d, 1, 20000)
    rows = run_condition(clean, shifted, np.array([5.0]), ("m", 0, "c"), n_partitions=20)
    for alpha in (0.05, 0.10):
        theory_a = 1 - stats.norm.cdf(stats.norm.ppf(1 - alpha) - d)
        a = _cell(rows, "A_no_recal", "conformal", alpha)
        b = _cell(rows, "B_recal", "conformal", alpha)
        assert abs(a["far_mean"] - theory_a) < 0.01
        assert a["far_mean"] > 1.5 * alpha  # the shift really breaks the target
        assert abs(b["far_mean"] - alpha) < 0.005
        assert a["tpr_mean"] == 1.0  # the lone anomaly at 5.0 is above every threshold


def test_deterministic_and_partition_independent_of_corruption():
    rng = np.random.default_rng(1)
    clean, shifted, anom = rng.normal(size=300), rng.normal(size=300), rng.normal(size=20)
    a = run_condition(clean, shifted, anom, ("m", 0, "c"), n_partitions=5)
    assert a == run_condition(clean, shifted, anom, ("m", 0, "c"), n_partitions=5)
    # no shift (shifted == clean): setting A and B use the same thresholds
    same = run_condition(clean, clean, anom, ("m", 0, "c"), n_partitions=5)
    assert _cell(same, "A_no_recal", "conformal", 0.05) | {"setting": 0} == (
        _cell(same, "B_recal", "conformal", 0.05) | {"setting": 0}
    )


def test_not_certifiable_is_explicit_and_never_clamped():
    rng = np.random.default_rng(2)
    x = rng.normal(size=400)  # n_cal = 200 < 299 = m_min for alpha = 1%, delta = 5%
    rows = run_condition(x, x + 1, np.ones(5), ("m", 0, "c"), n_partitions=4)
    for setting in ("A_no_recal", "B_recal"):
        tol = _cell(rows, setting, "tolerance", 0.01)
        assert tol["status"] == "not_certifiable"
        assert np.isnan(tol["far_mean"]) and np.isnan(tol["tpr_mean"])
        assert _cell(rows, setting, "conformal", 0.01)["status"] == "certified"
        assert _cell(rows, setting, "tolerance", 0.05)["status"] == "certified"


def test_rank_fast_path_equals_calibration_module():
    cal = np.random.default_rng(3).normal(size=150)
    for alpha in (0.01, 0.05, 0.10):
        c, t = conformal_threshold(cal, alpha), tolerance_threshold(cal, alpha, 0.05)
        assert rank_for("conformal", 150, alpha) == c.rank
        assert rank_for("tolerance", 150, alpha) == t.rank


def test_class_bootstrap_is_deterministic_and_brackets_the_mean():
    vals = np.random.default_rng(4).normal(size=(3, 12))
    m1, lo, hi = class_bootstrap(vals, 500)
    m2, lo2, hi2 = class_bootstrap(vals, 500)
    assert np.array_equal(lo, lo2) and np.array_equal(hi, hi2)
    assert np.allclose(m1, vals.mean(axis=1)) and (lo <= m1).all() and (m1 <= hi).all()


def _synthetic_scores(classes=("a", "b"), methods=("good", "weak"), seeds=(0, 1)):
    frames = []
    for method in methods:
        for seed in seeds:
            for ci, cls in enumerate(classes):
                rng = np.random.default_rng(10 * seed + ci + len(method))
                ids = [f"{cls}/normal/{i:03d}" for i in range(120)]
                ids_a = [f"{cls}/anomaly/{i:03d}" for i in range(15)]
                for corr, sev, shift in (
                    ("none", 0, 0.0),
                    ("gaussian_noise", -1, 0.4),
                    ("gaussian_noise", 1, 1.0),
                ):
                    scores = np.r_[rng.normal(shift, 1, 120), rng.normal(shift + 2, 1, 15)]
                    frames.append(make_frame(
                        image_ids=ids + ids_a, cls=cls, labels=[0] * 120 + [1] * 15,
                        roles=["heldout_normal"] * 120 + ["anomaly"] * 15,
                        official_splits=["train"] * 135, protocol="calib", method=method,
                        seed=seed, corruption=corr, severity=sev, scores=scores,
                    ))  # fmt: skip
    return pd.concat(frames, ignore_index=True)


def test_run_all_and_cli_write_small_outputs(tmp_path):
    from anomcal.scores_io import write_atomic

    df = _synthetic_scores()
    assert len(run_all(df, n_partitions=3)) > 0
    scores = tmp_path / "scores"
    scores.mkdir()
    for (method, seed), g in df.groupby(["method", "seed"]):
        write_atomic(g, scores / f"calib_{method}_seed{seed}.parquet")
    out = tmp_path / "out"
    assert main(["--scores", str(scores), "--out", str(out), "--partitions", "3", "--boot", "50",
                 "--no-figures"]) == 0  # fmt: skip
    hl = pd.read_csv(out / "headline.csv")
    assert {"far", "far_lo", "far_hi", "excess", "worst_class", "level"} <= set(hl.columns)
    # the mild level (severity -1) is analysed and drawn at x = 0.5, between clean and level 1
    mild = hl[hl["severity"] == -1]
    assert len(mild) > 0 and set(mild["level"]) == {0.5}
    stab = pd.read_csv(out / "stability.csv")
    assert set(stab["grid"]) == {"all", "1-3-5"}
    assert (out / "stability.csv").exists() and (out / "auroc_by_condition.csv").exists()
    assert all(f.stat().st_size < 5 * 2**20 for f in out.iterdir())


def test_plot_x_puts_the_mild_level_between_clean_and_level_1():
    assert [plot_x(v) for v in (0, -1, 1, 3, 5)] == [0.0, 0.5, 1.0, 3.0, 5.0]


def test_mild_level_known_answer_shift():
    """A mild shift d=0.3 raises the no-recalibration FAR a little; recalibration restores alpha."""
    rng = np.random.default_rng(5)
    clean, mild = rng.normal(0, 1, 20000), rng.normal(0.3, 1, 20000)
    rows = run_condition(clean, mild, np.array([5.0]), ("m", 0, "c"), n_partitions=20)
    a = _cell(rows, "A_no_recal", "conformal", 0.05)
    expected = 1 - stats.norm.cdf(stats.norm.ppf(0.95) - 0.3)
    assert abs(a["far_mean"] - expected) < 0.01 and 0.05 < a["far_mean"] < 0.5
