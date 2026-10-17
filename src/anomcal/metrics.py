"""Detection and localisation metrics (pure numpy; written by us and tested against sklearn).

- :func:`auroc`: image- or pixel-level area under the ROC curve, computed from ranks
  (the Mann-Whitney U statistic). AUROC = P(score of a random anomaly > score of a
  random normal), with ties counted as 1/2. That's why we can compute it without
  building the ROC curve at all.
- :func:`pro_auc`: area under the per-region-overlap (PRO) curve up to a false-positive
  rate of 0.3, normalised to [0, 1] (Bergmann et al., CVPR 2020). Unlike pixel AUROC,
  every connected defect region counts equally, so one huge defect cannot dominate.
- :func:`rate_above`: fraction of scores strictly above a threshold; it is the empirical
  false-alarm rate on normals and the detection rate on anomalies (Phase 3 builds on it).
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage
from scipy.stats import rankdata

PRO_FPR_LIMIT = 0.3
# 8-connectivity: diagonal neighbours belong to the same defect region.
_EIGHT_CONNECTED = np.ones((3, 3), dtype=bool)


def auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Area under the ROC curve for binary ``labels`` (1 = anomaly) and real ``scores``.

    Uses average ranks, so tied scores contribute 1/2, exactly like
    ``sklearn.metrics.roc_auc_score``. Raises if only one class is present.
    """
    labels = np.asarray(labels).ravel()
    scores = np.asarray(scores, dtype=np.float64).ravel()
    if labels.shape != scores.shape:
        raise ValueError("labels and scores differ in length")
    if not np.isfinite(scores).all():
        raise ValueError("scores must be finite")
    pos = labels == 1
    n_pos = int(pos.sum())
    n_neg = labels.size - n_pos
    if n_pos == 0 or n_neg == 0:
        raise ValueError("AUROC needs both positive and negative examples")
    if not np.isin(labels, (0, 1)).all():
        raise ValueError("labels must be 0/1")
    ranks = rankdata(scores)  # 1-based, ties get the average rank
    u = ranks[pos].sum() - n_pos * (n_pos + 1) / 2.0  # Mann-Whitney U of the positives
    return float(u / (n_pos * n_neg))


def pixel_auroc(maps: np.ndarray, masks: np.ndarray) -> float:
    """AUROC over all pixels of all images (maps: (N, H, W) float, masks: (N, H, W) bool)."""
    _check_maps(maps, masks)
    return auroc(masks.ravel().astype(np.int8), maps.ravel())


def pro_auc(
    maps: np.ndarray,
    masks: np.ndarray,
    fpr_limit: float = PRO_FPR_LIMIT,
    n_thresholds: int = 200,
) -> float:
    """Normalised area under the PRO curve for FPR in [0, fpr_limit].

    For a threshold t:
      * FPR(t) = fraction of *normal* pixels (mask 0, in any image) with score >= t;
      * PRO(t) = mean over all connected defect regions of the fraction of that
        region's pixels with score >= t.
    We take thresholds at FPR values evenly spaced in [0, fpr_limit] (via quantiles of
    the normal-pixel scores), integrate PRO over FPR with the trapezoid rule and divide
    by ``fpr_limit``. A perfect map gives 1.0. A random map has PRO(FPR) ~ FPR, so its
    normalised area is about fpr_limit / 2 = 0.15, not 0.5 as for AUROC.
    """
    _check_maps(maps, masks)
    if not 0 < fpr_limit <= 1:
        raise ValueError("fpr_limit must be in (0, 1]")
    normal_scores = np.sort(maps[~masks].astype(np.float64))
    if normal_scores.size == 0:
        raise ValueError("PRO needs normal pixels")
    regions = []
    for m, mask in zip(maps, masks, strict=True):
        labelled, n = ndimage.label(mask, structure=_EIGHT_CONNECTED)
        for r in range(1, n + 1):
            regions.append(np.sort(m[labelled == r].astype(np.float64)))
    if not regions:
        raise ValueError("PRO needs at least one defect region")

    # Threshold for each target FPR f: the (1 - f) quantile of normal scores.
    target_fpr = np.linspace(0.0, fpr_limit, n_thresholds)
    n = normal_scores.size
    # In sorted scores s, the fraction >= s[j] is (n - j) / n, so FPR f needs j = n(1 - f).
    idx = np.clip(np.ceil((1.0 - target_fpr) * n).astype(int), 0, n - 1)
    thresholds = np.append(normal_scores[-1] + 1.0, normal_scores[idx[1:]])  # FPR 0 first
    # Achieved FPR and PRO at each threshold (fractions >= t, via binary search).
    fpr = 1.0 - np.searchsorted(normal_scores, thresholds, side="left") / normal_scores.size
    pro = np.mean(
        [1.0 - np.searchsorted(r, thresholds, side="left") / r.size for r in regions], axis=0
    )
    # Ties can make FPR overshoot fpr_limit slightly; clip to the integration range.
    keep = fpr <= fpr_limit + 1e-12
    fpr, pro = fpr[keep], pro[keep]
    order = np.argsort(fpr, kind="stable")
    fpr, pro = fpr[order], pro[order]
    if fpr[-1] < fpr_limit:  # extend the last value to the end of the range
        fpr, pro = np.append(fpr, fpr_limit), np.append(pro, pro[-1])
    area = float(np.sum((fpr[1:] - fpr[:-1]) * (pro[1:] + pro[:-1]) / 2.0))
    return area / fpr_limit


def rate_above(scores: np.ndarray, threshold: float) -> float:
    """Fraction of scores strictly above ``threshold`` (alarm rule: score > t)."""
    scores = np.asarray(scores, dtype=np.float64)
    if scores.size == 0:
        raise ValueError("empty score array")
    return float(np.mean(scores > threshold))


def _check_maps(maps: np.ndarray, masks: np.ndarray) -> None:
    if maps.shape != masks.shape or maps.ndim != 3:
        raise ValueError(f"maps {maps.shape} and masks {masks.shape} must both be (N, H, W)")
    if masks.dtype != bool:
        raise ValueError("masks must be boolean")
    if not np.isfinite(maps).all():
        raise ValueError("maps must be finite")
