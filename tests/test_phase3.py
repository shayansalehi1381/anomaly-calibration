"""Phase 3 evaluation on synthetic score files with known answers."""

import numpy as np
import pandas as pd
import pytest

from anomcal.phase3 import (
    certifiability,
    cross_split_check,
    cross_split_summary,
    law_check,
    main,
    method_summary,
    n_calibration,
    run_all,
    run_experiment,
)
from anomcal.scores_io import make_frame, write_atomic


def synthetic_scores(n_train=150, n_test=50, n_anom=40, seeds=(0, 1), classes=("candle", "pcb1")):
    """Gaussian normal scores (exchangeable by construction), anomalies shifted up."""
    frames = []
    for method, shift in (("good", 3.0), ("weak", 0.3)):
        for seed in seeds:
            for ci, cls in enumerate(classes):
                rng = np.random.default_rng(100 * seed + 10 * ci + len(method))
                ids = [f"{cls}/normal/{i:04d}" for i in range(n_train + n_test)]
                ids += [f"{cls}/anomaly/{i:03d}" for i in range(n_anom)]
                labels = [0] * (n_train + n_test) + [1] * n_anom
                scores = np.r_[rng.normal(0, 1, n_train + n_test), rng.normal(shift, 1, n_anom)]
                frames.append(make_frame(
                    image_ids=ids, cls=cls, labels=labels,
                    roles=["heldout_normal" if lab == 0 else "anomaly" for lab in labels],
                    official_splits=["train"] * n_train + ["test"] * (n_test + n_anom),
                    protocol="calib", method=method, seed=seed, corruption="none",
                    severity=0, scores=scores,
                ))  # fmt: skip
    return pd.concat(frames, ignore_index=True)


@pytest.fixture(scope="module")
def parts():
    return run_all(synthetic_scores(), n_partitions=40)


def test_n_calibration_matches_split_protocol():
    assert n_calibration(275) == 138 and n_calibration(554) == 277 and n_calibration(2) == 1


def test_partitions_are_disjoint_and_deterministic():
    pool, anom = np.random.default_rng(0).normal(size=201), np.zeros(5)
    a = run_experiment(pool, anom, ("m", 0, "c"), n_partitions=3)
    b = run_experiment(pool, anom, ("m", 0, "c"), n_partitions=3)
    assert a == b  # same key -> identical partitions and thresholds
    c = run_experiment(pool, anom, ("m", 1, "c"), n_partitions=3)
    assert [r["threshold"] for r in a] != [r["threshold"] for r in c]
    assert all(r["n_cal"] + r["n_test"] == 201 for r in a)


def test_guarantee_holds_on_exchangeable_gaussian_scores(parts):
    s = method_summary(parts)
    conf = s[s["kind"] == "conformal"]
    # pooled FAR close to the exact theoretical mean (alpha - 1/(n+1) < E[FAR] <= alpha)
    assert np.allclose(conf["far_pooled"], conf["far_theory_mean"], atol=0.01)
    assert (conf["far_theory_mean"] <= conf["alpha"] + 1e-12).all()
    # the good detector catches more anomalies at the same FAR than the weak one
    good = conf[conf["method"] == "good"]["tpr_mean"].to_numpy()
    weak = conf[conf["method"] == "weak"]["tpr_mean"].to_numpy()
    assert (good > weak).all()


def test_not_certifiable_cases_are_explicit(parts):
    cert = certifiability(parts)
    # n_cal = 100 here: conformal certifies 1% (needs >= 99); tolerance needs 299 at 1%
    assert set(cert["n_cal"]) == {100}
    assert (cert["conformal_0.01"] == "certified").all()
    assert (cert["tolerance_0.01"] == "not_certifiable").all()
    assert (cert["tolerance_0.05"] == "certified").all()
    nc = parts[parts["status"] == "not_certifiable"]
    assert nc["threshold"].isna().all() and nc["far"].isna().all()  # no fake threshold


def test_law_check_is_calibrated_under_the_null(parts):
    laws = law_check(parts, n_sim=300)
    assert laws["ks_p"].between(0, 1).all()
    assert (laws["ks_p"] < 0.01).mean() < 0.2  # data drawn from the null
    assert np.allclose(laws["mean_fa_obs"], laws["mean_fa_law"], rtol=0.5, atol=1.0)


def test_cross_split_check(parts):
    cross = cross_split_check(synthetic_scores())
    assert set(cross["n_cal"]) == {150} and set(cross["n_test"]) == {50}
    summ = cross_split_summary(cross)
    assert (summ["wilson_lo"] <= summ["far_pooled"]).all()
    assert (summ["far_pooled"] <= summ["wilson_hi"]).all()


def test_cli_writes_small_outputs(tmp_path):
    scores_dir = tmp_path / "scores"
    df = synthetic_scores(seeds=(0,), classes=("candle",))
    for (method, seed), g in df.groupby(["method", "seed"]):
        write_atomic(g.reset_index(drop=True), scores_dir / f"calib_{method}_seed{seed}.parquet")
    out = tmp_path / "phase3"
    assert main(["--scores", str(scores_dir), "--out", str(out), "--partitions", "5",
                 "--figures", str(tmp_path / "fig")]) == 0  # fmt: skip
    for name in ("summary_by_method", "certifiability", "law_check", "cross_split_summary",
                 "per_class"):  # fmt: skip
        assert (out / f"{name}.csv").stat().st_size < 5 * 1024 * 1024
    assert (tmp_path / "fig" / "phase3_far_vs_alpha.png").exists()


def test_missing_scores_dir(tmp_path):
    with pytest.raises(FileNotFoundError):
        main(["--scores", str(tmp_path), "--out", str(tmp_path / "o"), "--no-figures"])


def test_far_figure_places_points_by_alpha_value(parts, tmp_path, monkeypatch):
    """Regression: the tolerance limit skips alpha = 1%; its points must not shift left."""
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib.axes import Axes

    from anomcal.phase3 import ALPHAS, make_figures

    calls = []
    real = Axes.errorbar

    def spy(self, x, y, *a, **k):
        calls.append((k.get("label", ""), list(np.round(x, 2)), list(y)))
        return real(self, x, y, *a, **k)

    monkeypatch.setattr(Axes, "errorbar", spy)
    make_figures(method_summary(parts), parts, tmp_path)
    tol = [c for c in calls if c[0].startswith("tolerance")][0]
    assert tol[1] == [round(ALPHAS.index(0.05) + 0.09, 2), round(ALPHAS.index(0.10) + 0.09, 2)]
