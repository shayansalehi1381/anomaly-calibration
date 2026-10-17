"""Calibration math on cases with known answers."""

import numpy as np
import pytest
from scipy import stats

from anomcal.calibration import (
    CERTIFIED,
    NOT_CERTIFIABLE,
    conformal_alpha_min,
    conformal_pvalues,
    conformal_rank,
    conformal_threshold,
    false_alarm_count_law,
    far_law,
    tolerance_confidence,
    tolerance_m_min,
    tolerance_rank,
    tolerance_threshold,
    wilson_interval,
)


# ---- Deng's tolerance limit: reproduce the paper's numbers --------------------------
def test_tolerance_reproduces_deng_primary_setting():
    r, conf = tolerance_rank(150, 0.10, 0.05)
    assert r == 142 and round(conf, 4) == 0.9693


@pytest.mark.parametrize("alpha, m_min", [(0.01, 299), (0.05, 59), (0.10, 29)])
def test_tolerance_m_min(alpha, m_min):
    assert tolerance_m_min(alpha, 0.05) == m_min
    assert tolerance_rank(m_min, alpha, 0.05) is not None  # certifiable at m_min...
    assert tolerance_rank(m_min - 1, alpha, 0.05) is None  # ...and not one below


def test_tolerance_rank_is_the_least_conservative_valid_rank():
    r, conf = tolerance_rank(150, 0.10, 0.05)
    assert conf >= 0.95 and tolerance_confidence(150, r - 1, 0.10) < 0.95


# ---- split conformal ----------------------------------------------------------------
@pytest.mark.parametrize(
    "n, alpha, k",
    [(199, 0.05, 190), (99, 0.01, 99), (50, 0.01, None), (500, 0.01, 496), (8, 0.10, None),
     (9, 0.10, 9), (137, 0.01, 137)],
)  # fmt: skip
def test_conformal_rank(n, alpha, k):
    assert conformal_rank(n, alpha) == k


def test_conformal_alpha_min_matches_rank_existence():
    for n in (8, 19, 99, 137, 275):
        a = conformal_alpha_min(n)
        assert conformal_rank(n, a) is not None
        assert conformal_rank(n, a * 0.999) is None
    assert conformal_alpha_min(8) == pytest.approx(1 / 9)


def test_not_certifiable_is_explicit_never_clamped():
    t = conformal_threshold(np.arange(50.0), 0.01)
    assert t.status == NOT_CERTIFIABLE and t.value is None and t.rank is None
    t = tolerance_threshold(np.arange(150.0), 0.01, 0.05)
    assert t.status == NOT_CERTIFIABLE and t.value is None
    ok = conformal_threshold(np.arange(1.0, 200.0), 0.05)
    assert ok.status == CERTIFIED and ok.rank == 190 and ok.value == 190.0


def test_thresholds_sort_their_input():
    rng = np.random.default_rng(0)
    s = rng.normal(size=199)
    assert conformal_threshold(rng.permutation(s), 0.05).value == np.sort(s)[189]


@pytest.mark.parametrize("bad", [0.0, 1.0, -0.1])
def test_bad_alpha(bad):
    with pytest.raises(ValueError):
        conformal_rank(100, bad)
    with pytest.raises(ValueError):
        tolerance_rank(100, bad, 0.05)


# ---- the guarantees, checked by simulation on Gaussian normal scores -----------------
def test_conformal_far_matches_beta_law_on_gaussian_scores():
    """Exchangeable Gaussian scores: the realised FAR follows Beta(n+1-k, k) exactly."""
    rng = np.random.default_rng(1)
    n, alpha, reps = 137, 0.05, 4000
    k = conformal_rank(n, alpha)
    fars = np.empty(reps)
    for i in range(reps):
        t = conformal_threshold(rng.normal(size=n), alpha).value
        fars[i] = stats.norm.sf(t)  # true FAR of this threshold for N(0, 1) normals
    law = far_law(n, k)
    assert alpha - 1 / (n + 1) < law.mean() <= alpha
    assert abs(fars.mean() - law.mean()) < 4 * law.std() / np.sqrt(reps)
    assert stats.kstest(fars, law.cdf).pvalue > 0.01


def test_tolerance_limit_holds_its_confidence_on_gaussian_scores():
    rng = np.random.default_rng(2)
    m, alpha, delta, reps = 150, 0.10, 0.05, 4000
    ok = 0
    for _ in range(reps):
        t = tolerance_threshold(rng.normal(size=m), alpha, delta).value
        ok += stats.norm.sf(t) <= alpha
    assert ok / reps >= 0.9693 - 4 * np.sqrt(0.97 * 0.03 / reps)


def test_count_law_is_exact_for_random_partitions_of_a_fixed_pool():
    """Random cal/test splits of ONE fixed pool: the false-alarm count is Beta-binomial."""
    rng = np.random.default_rng(3)
    pool = rng.gamma(2.0, size=275)  # any continuous distribution
    n, m_test, alpha = 138, 137, 0.05
    k = conformal_rank(n, alpha)
    counts = []
    for _ in range(3000):
        perm = rng.permutation(pool)
        t = conformal_threshold(perm[:n], alpha).value
        counts.append(int((perm[n:] > t).sum()))
    law = false_alarm_count_law(n, k, m_test)
    assert np.mean(counts) == pytest.approx(law.mean(), abs=4 * law.std() / np.sqrt(3000))
    observed = np.bincount(counts, minlength=m_test + 1)[:15]
    expected = law.pmf(np.arange(15)) * 3000
    assert np.abs(observed - expected).max() < 5 * np.sqrt(expected.max())


# ---- Wilson and p-values ------------------------------------------------------------
@pytest.mark.parametrize(
    "x, n, lo, hi",
    [(1, 137, 0.0013, 0.0402), (0, 137, 0.0, 0.0273), (7, 137, 0.0250, 0.1017)],
)
def test_wilson_matches_hand_computed_values(x, n, lo, hi):
    a, b = wilson_interval(x, n)
    assert (round(a, 4), round(b, 4)) == (lo, hi)


def test_wilson_input_checks():
    with pytest.raises(ValueError):
        wilson_interval(5, 3)
    with pytest.raises(ValueError):
        wilson_interval(0, 0)


def test_pvalue_rule_equals_threshold_rule():
    rng = np.random.default_rng(4)
    cal, test = rng.normal(size=199), rng.normal(size=1000)
    t = conformal_threshold(cal, 0.05).value
    p = conformal_pvalues(cal, test)
    np.testing.assert_array_equal(p <= 0.05, test > t)
