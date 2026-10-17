"""False-alarm thresholds from normal calibration scores (Phase 3; written by us).

Two thresholds, both using ONLY normal images (anomalies never enter calibration):

Split conformal (marginal guarantee)
    k = ceil((n+1)(1-alpha)),  t = k-th smallest calibration score,  alarm iff score > t.
    If calibration and test normals are exchangeable:
        alpha - 1/(n+1) < E[FAR] <= alpha        (lower bound needs no ties).
    Certifiable only if k <= n, i.e. alpha >= 1/(n+1). Otherwise status "not_certifiable"
    and NO threshold (we never clamp to the maximum score).

Tolerance limit (PAC guarantee; Deng, arXiv:2608.15090)
    r = smallest rank with Pr{Binomial(m, alpha) >= m - r + 1} >= 1 - delta,  t = S_(r).
    Guarantee over the calibration draw: Pr[ FAR(t) <= alpha ] >= 1 - delta.
    Certifiable only if m >= m_min = ceil(log delta / log(1 - alpha)).

The realised FAR of a threshold at rank k (for continuous scores) follows
Beta(n + 1 - k, k), and the number of false alarms among m_test test normals follows
Beta-binomial(m_test, n + 1 - k, k). See docs/INTERVIEW_NOTES.md for the derivations.

Exact arithmetic: k uses fractions.Fraction, so float rounding of (n+1)(1-alpha) can
never move k across an integer.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction

import numpy as np
from scipy import stats

CERTIFIED = "certified"
NOT_CERTIFIABLE = "not_certifiable"


@dataclass(frozen=True)
class Threshold:
    """A calibrated threshold, or an explicit refusal (status != certified, value None)."""

    kind: str  # "conformal" | "tolerance"
    alpha: float
    n: int  # number of calibration scores
    status: str
    rank: int | None = None  # 1-based order statistic used
    value: float | None = None
    confidence: float | None = None  # tolerance limit: achieved Pr[FAR <= alpha]
    delta: float | None = None

    @property
    def certified(self) -> bool:
        return self.status == CERTIFIED


def _check_alpha(alpha: float) -> Fraction:
    if not 0 < alpha < 1:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    # 0.01 as a float is not exactly 1/100; limit_denominator recovers the intended value.
    return Fraction(alpha).limit_denominator(10**9)


def conformal_rank(n: int, alpha: float) -> int | None:
    """k = ceil((n+1)(1-alpha)) with exact rational arithmetic; None if k > n."""
    if n < 1:
        raise ValueError("need at least one calibration score")
    a = _check_alpha(alpha)
    k = math.ceil((n + 1) * (1 - a))
    return k if k <= n else None


def conformal_alpha_min(n: int) -> float:
    """Smallest alpha split conformal can certify with n calibration scores: 1/(n+1)."""
    return 1.0 / (n + 1)


def tolerance_confidence(m: int, r: int, alpha: float) -> float:
    """Pr[FAR(S_(r)) <= alpha] = Pr{Binomial(m, alpha) >= m - r + 1} (continuous scores)."""
    return float(stats.binom.sf(m - r, m, alpha))


def tolerance_rank(m: int, alpha: float, delta: float) -> tuple[int, float] | None:
    """Smallest rank r whose confidence reaches 1 - delta, with that confidence; else None.

    Confidence grows with r (a higher order statistic is a more conservative threshold),
    so the *smallest* such r is the least conservative valid choice.
    """
    if m < 1:
        raise ValueError("need at least one calibration score")
    _check_alpha(alpha)
    if not 0 < delta < 1:
        raise ValueError(f"delta must be in (0, 1), got {delta}")
    r = np.arange(1, m + 1)
    conf = stats.binom.sf(m - r, m, alpha)
    ok = np.nonzero(conf >= 1 - delta)[0]
    if ok.size == 0:
        return None
    return int(r[ok[0]]), float(conf[ok[0]])


def tolerance_m_min(alpha: float, delta: float) -> int:
    """Smallest m for which ANY rank is certifiable: (1 - alpha)^m <= delta."""
    _check_alpha(alpha)
    m = math.ceil(math.log(delta) / math.log(1 - alpha))
    # guard against float error at the boundary, in both directions
    while m > 1 and (1 - alpha) ** (m - 1) <= delta:
        m -= 1
    while (1 - alpha) ** m > delta:
        m += 1
    return m


def conformal_threshold(cal_scores: np.ndarray, alpha: float) -> Threshold:
    s = np.sort(np.asarray(cal_scores, dtype=np.float64))
    k = conformal_rank(s.size, alpha)
    if k is None:
        return Threshold("conformal", alpha, s.size, NOT_CERTIFIABLE)
    return Threshold("conformal", alpha, s.size, CERTIFIED, rank=k, value=float(s[k - 1]))


def tolerance_threshold(cal_scores: np.ndarray, alpha: float, delta: float) -> Threshold:
    s = np.sort(np.asarray(cal_scores, dtype=np.float64))
    res = tolerance_rank(s.size, alpha, delta)
    if res is None:
        return Threshold("tolerance", alpha, s.size, NOT_CERTIFIABLE, delta=delta)
    r, conf = res
    return Threshold(
        "tolerance", alpha, s.size, CERTIFIED, rank=r, value=float(s[r - 1]),
        confidence=conf, delta=delta,
    )  # fmt: skip


def far_law(n: int, k: int):
    """Distribution of the realised FAR of the rank-k threshold: Beta(n + 1 - k, k)."""
    return stats.beta(n + 1 - k, k)


def false_alarm_count_law(n: int, k: int, m_test: int):
    """Number of false alarms among m_test test normals: Beta-binomial(m_test, n+1-k, k)."""
    return stats.betabinom(m_test, n + 1 - k, k)


def wilson_interval(successes: int, trials: int, confidence: float = 0.95) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion (good near 0, unlike p ± z·se)."""
    if trials < 1 or not 0 <= successes <= trials:
        raise ValueError("need 0 <= successes <= trials and trials >= 1")
    z = stats.norm.ppf(0.5 + confidence / 2)
    p = successes / trials
    denom = 1 + z**2 / trials
    centre = (p + z**2 / (2 * trials)) / denom
    half = z / denom * math.sqrt(p * (1 - p) / trials + z**2 / (4 * trials**2))
    return max(0.0, centre - half), min(1.0, centre + half)


def conformal_pvalues(cal_scores: np.ndarray, test_scores: np.ndarray) -> np.ndarray:
    """p(x) = (1 + #{i: S_i >= s(x)}) / (n + 1); p <= alpha  <=>  s(x) > S_(k)."""
    s = np.sort(np.asarray(cal_scores, dtype=np.float64))
    t = np.asarray(test_scores, dtype=np.float64)
    at_or_above = s.size - np.searchsorted(s, t, side="left")
    return (1 + at_or_above) / (s.size + 1)
