"""Phase 4: does the calibrated threshold keep its false-alarm rate (FAR) under corruption?

Same design as Phase 3 (protocol ``calib``; method x seed x class = one experiment;
R = 100 random 50/50 partitions of the held-out normal pool). The SAME partition (same
images in calibration and test) is used for every corruption, so the corruptions are
compared on identical images. For every corruption x severity and each threshold kind
(split conformal; tolerance limit with delta = 5%) at alpha in {1%, 5%, 10%}:

Setting A, "no recalibration"
    threshold from the CLEAN calibration normals, applied to the CORRUPTED test normals.
    Exchangeability between calibration and test normals is violated BY CONSTRUCTION
    here, so the guarantee does not apply; we measure how badly it breaks.
Setting B, "recalibration"
    threshold from the CORRUPTED calibration normals (same corruption, same severity),
    applied to the corrupted test normals. This needs normal images under the new
    condition. Calibration and test are exchangeable again, so the guarantee is back.

Clean images (``none``, severity 0) are included as the reference point, where A == B.

Per (experiment, corruption, severity, setting, kind, alpha) we store the mean and std
of the test FAR over the partitions, the violation rate (share of partitions with
FAR > alpha) and the mean TPR on the corrupted anomalies at that threshold. A setting a
kind cannot certify (e.g. tolerance limit at alpha = 1% with ~140-277 calibration
normals) is stored as ``not_certifiable`` with empty numbers. We never clamp.

Uncertainty: the Phase 3 Wilson intervals treated the 100 splits (which reuse the same
images) as independent, so they were too narrow. Here the headline numbers get a
bootstrap over CLASSES (resample the 12 classes with replacement). See
:func:`class_bootstrap` for exactly what that does and does not cover.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from functools import cache
from pathlib import Path

import numpy as np
import pandas as pd

from anomcal.calibration import conformal_rank, tolerance_rank
from anomcal.phase3 import ALPHAS, DELTA, KINDS, N_PARTITIONS, n_calibration
from anomcal.seeding import item_rng

SETTINGS = ("A_no_recal", "B_recal")
CORRUPTIONS = ("brightness", "gaussian_blur", "gaussian_noise", "jpeg_compression")
N_BOOT = 2000
KEYS = ["method", "seed", "class", "corruption", "severity", "setting", "kind", "alpha"]


@cache
def rank_for(kind: str, n: int, alpha: float) -> int | None:
    """1-based order-statistic rank of the threshold, or None if not certifiable.

    The rank depends only on (kind, n, alpha, delta), never on the scores, so it is
    computed once per calibration size; the threshold is then ``sorted(cal)[rank - 1]``.
    ``tests/test_phase4.py`` checks this equals :func:`anomcal.calibration.conformal_threshold`
    and :func:`~anomcal.calibration.tolerance_threshold`.
    """
    if kind == "conformal":
        return conformal_rank(n, alpha)
    res = tolerance_rank(n, alpha, DELTA)
    return None if res is None else res[0]


def _far_tpr(t: float, test_sorted: np.ndarray, anom_sorted: np.ndarray) -> tuple[float, float]:
    """Alarm iff score > t (same strict rule as Phase 3)."""
    m = test_sorted.size
    far = (m - np.searchsorted(test_sorted, t, side="right")) / m
    if anom_sorted.size == 0:
        return far, float("nan")
    tpr = (anom_sorted.size - np.searchsorted(anom_sorted, t, side="right")) / anom_sorted.size
    return far, tpr


def run_condition(
    clean: np.ndarray,
    shifted: np.ndarray,
    anomalous: np.ndarray,
    key: tuple,
    alphas: Sequence[float] = ALPHAS,
    n_partitions: int = N_PARTITIONS,
) -> list[dict]:
    """One corruption x severity for one experiment.

    ``clean[i]`` and ``shifted[i]`` are the scores of the SAME normal image i before and
    after the corruption (so one permutation splits both). ``anomalous`` are the scores
    of the corrupted anomalous images. The partition depends on ``key`` (method, seed,
    class) only, not on the corruption.
    """
    if clean.size != shifted.size:
        raise ValueError("clean and shifted must score the same normal images")
    n_pool = clean.size
    n_cal = n_calibration(n_pool)
    anom_sorted = np.sort(anomalous)
    acc: dict[tuple, list[tuple[float, float]]] = {}
    for r in range(n_partitions):
        perm = item_rng("phase4-partition", *key, r).permutation(n_pool)
        ci, ti = perm[:n_cal], perm[n_cal:]
        cal_sorted = {"A_no_recal": np.sort(clean[ci]), "B_recal": np.sort(shifted[ci])}
        test_sorted = np.sort(shifted[ti])  # the test images are always the corrupted ones
        for setting, cs in cal_sorted.items():
            for kind in KINDS:
                for alpha in alphas:
                    rank = rank_for(kind, n_cal, alpha)
                    cell = acc.setdefault((setting, kind, alpha), [])
                    if rank is None:
                        continue
                    cell.append(_far_tpr(float(cs[rank - 1]), test_sorted, anom_sorted))
    rows = []
    for (setting, kind, alpha), vals in acc.items():
        row = {
            "setting": setting,
            "kind": kind,
            "alpha": alpha,
            "n_cal": n_cal,
            "n_test": n_pool - n_cal,
            "n_anomalous": anomalous.size,
            "status": "certified" if vals else "not_certifiable",
            "far_mean": np.nan,
            "far_std": np.nan,
            "violation_rate": np.nan,
            "tpr_mean": np.nan,
        }
        if vals:
            far, tpr = np.array(vals).T
            row.update(
                far_mean=far.mean(),
                far_std=far.std(ddof=1),
                violation_rate=float((far > alpha).mean()),
                tpr_mean=np.nanmean(tpr),
            )
        rows.append(row)
    return rows


def load_calib(scores_dir: Path) -> pd.DataFrame:
    files = sorted(Path(scores_dir).glob("calib_*_seed*.parquet"))
    if not files:
        raise FileNotFoundError(f"no calib score files in {scores_dir}")
    cols = [
        "image_id",
        "class",
        "label",
        "role",
        "method",
        "seed",
        "corruption",
        "severity",
        "score",
    ]
    df = pd.concat([pd.read_parquet(f, columns=cols) for f in files], ignore_index=True)
    if df.duplicated(["method", "seed", "image_id", "corruption", "severity"]).any():
        raise ValueError("duplicate (method, seed, image, corruption, severity) rows")
    return df


def run_all(df: pd.DataFrame, n_partitions: int = N_PARTITIONS) -> pd.DataFrame:
    """All experiments x conditions -> one row per (experiment, condition, setting, kind, alpha)."""
    out = []
    for (method, seed, cls), g in df.groupby(["method", "seed", "class"], sort=True):
        key = (method, int(seed), cls)
        pool = g[g["role"] == "heldout_normal"]
        if (pool["label"] != 0).any():
            raise ValueError(f"{method}/{seed}/{cls}: anomaly inside the normal pool")
        anom = g[g["role"] == "anomaly"]
        clean = pool[pool["corruption"] == "none"].sort_values("image_id")
        for (corr, sev), cond in pool.groupby(["corruption", "severity"], sort=True):
            cond = cond.sort_values("image_id")
            if not np.array_equal(cond["image_id"].to_numpy(), clean["image_id"].to_numpy()):
                raise ValueError(f"{key} {corr}/{sev}: normal images differ from the clean set")
            a = anom[(anom["corruption"] == corr) & (anom["severity"] == sev)]
            rows = run_condition(
                clean["score"].to_numpy(np.float64),
                cond["score"].to_numpy(np.float64),
                a["score"].to_numpy(np.float64),
                key,
                n_partitions=n_partitions,
            )
            out += [
                {
                    "method": method,
                    "seed": int(seed),
                    "class": cls,
                    "corruption": corr,
                    "severity": int(sev),
                    **r,
                }
                for r in rows
            ]
    return pd.DataFrame(out)


# ------------------------------------------------------------------ aggregation + bootstrap


def per_class(exp: pd.DataFrame) -> pd.DataFrame:
    """Average the 3 laptop seeds: one row per (method, class, condition, setting, kind, alpha)."""
    keys = [k for k in KEYS if k != "seed"]
    ok = exp[exp["status"] == "certified"].copy()
    ok["excess"] = ok["far_mean"] - ok["alpha"]
    g = ok.groupby(keys)
    out = g.agg(
        far=("far_mean", "mean"),
        excess=("excess", "mean"),
        far_seed_std=("far_mean", "std"),
        violation_rate=("violation_rate", "mean"),
        tpr=("tpr_mean", "mean"),
        seeds=("seed", "nunique"),
    ).reset_index()
    return out


def class_bootstrap(
    values: np.ndarray, n_boot: int = N_BOOT, seed_tag: str = "phase4-boot"
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bootstrap over classes. ``values``: (groups, classes) -> (mean, lo, hi) per group.

    Each replicate draws ``n_classes`` class indices with replacement (the SAME draw for
    all groups, so differences between methods are paired) and averages over them.
    The interval is the 2.5-97.5% percentile range of those means.

    What it covers: which VisA classes happened to be in the study. Because each class
    value is already averaged over the 3 laptop seeds and the 100 partitions, it
    also absorbs most of their variation, but only as part of the class spread.
    What it does NOT cover: other datasets or products, other corruption parameters
    than the 3 severities we chose, sampling of the images inside a class (a class has
    only 138-277 test normals and the 100 partitions reuse them, so within-class
    uncertainty is real and not captured), or the fact that the corruptions are
    synthetic. With 12 classes the interval is itself rough.
    """
    n_classes = values.shape[1]
    idx = item_rng(seed_tag, n_classes).integers(0, n_classes, size=(n_boot, n_classes))
    boot = values[:, idx].mean(axis=2)  # (groups, n_boot)
    lo, hi = np.percentile(boot, [2.5, 97.5], axis=1)
    return values.mean(axis=1), lo, hi


def _class_matrix(pc: pd.DataFrame, value: str, index: list[str]) -> pd.DataFrame:
    m = pc.pivot_table(index=index, columns="class", values=value)
    return m.dropna(axis=0, how="any")  # a group missing a class would bias the bootstrap


def headline(pc: pd.DataFrame, n_boot: int = N_BOOT) -> pd.DataFrame:
    """Per (method, condition, setting, kind, alpha): class-mean FAR/excess/TPR + CI."""
    index = ["method", "corruption", "severity", "setting", "kind", "alpha"]
    far = _class_matrix(pc, "far", index)
    mean, lo, hi = class_bootstrap(far.to_numpy(), n_boot)
    out = far.index.to_frame(index=False)
    out["far"], out["far_lo"], out["far_hi"] = mean, lo, hi
    out["excess"] = out["far"] - out["alpha"]
    out["excess_lo"], out["excess_hi"] = lo - out["alpha"], hi - out["alpha"]
    for col in ("violation_rate", "tpr"):
        m = _class_matrix(pc, col, index)
        out[col] = m.mean(axis=1).reindex(pd.MultiIndex.from_frame(out[index])).to_numpy()
    out["classes"] = far.shape[1]
    # the class with the largest |FAR - alpha| in this cell, so a deviant class is visible
    dev = (far.sub(far.index.get_level_values("alpha").to_numpy(), axis=0)).abs()
    out["worst_class"] = dev.idxmax(axis=1).to_numpy()
    out["worst_class_far"] = [
        far.loc[i, c] for i, c in zip(far.index, out["worst_class"], strict=True)
    ]
    return out


def stability(pc: pd.DataFrame, exp: pd.DataFrame, n_boot: int = N_BOOT) -> pd.DataFrame:
    """Per (setting, kind, alpha, method) over the 12 shifted conditions (corruption != none).

    rms_excess   sqrt(mean over conditions and classes of (FAR - alpha)^2): the main stability
                 number (0 = target kept exactly in every class and condition; penalises
                 over- and under-shooting alike, so read it together with mean_excess).
    class_spread std over classes of the class FAR, averaged over conditions.
    seed_spread  std over the 3 laptop seeds of the per-experiment FAR, averaged.
    rank         1 = smallest rms_excess within (setting, kind, alpha).
    """
    sh = pc[pc["corruption"] != "none"]
    idx = ["setting", "kind", "alpha", "method"]
    e = sh.copy()
    e["sq"] = e["excess"] ** 2
    rows = []
    for key, g in e.groupby(idx):
        piv = g.pivot_table(index=["corruption", "severity"], columns="class", values="sq")
        rows.append((*key, piv))
    out = []
    for (setting, kind, alpha), grp in _by_prefix(rows):
        # (conditions, classes) -> bootstrap over classes, paired across the methods
        sq = {m: piv.to_numpy() for m, piv in grp}
        methods = list(sq)
        stack = np.stack([sq[m] for m in methods])  # (methods, cond, classes)
        n_cls = stack.shape[2]
        idxs = item_rng("phase4-boot-stab", n_cls).integers(0, n_cls, size=(n_boot, n_cls))
        boot = np.sqrt(stack[:, :, idxs].mean(axis=(1, 3)))  # (methods, n_boot)
        point = np.sqrt(stack.mean(axis=(1, 2)))
        lo, hi = np.percentile(boot, [2.5, 97.5], axis=1)
        order = point.argsort().argsort() + 1
        for i, m in enumerate(methods):
            sub = sh[
                (sh["setting"] == setting)
                & (sh["kind"] == kind)
                & (sh["alpha"] == alpha)
                & (sh["method"] == m)
            ]
            ex = exp[
                (exp["status"] == "certified")
                & (exp["setting"] == setting)
                & (exp["kind"] == kind)
                & (exp["alpha"] == alpha)
                & (exp["method"] == m)
                & (exp["corruption"] != "none")
            ]
            cs = sub.pivot_table(index=["corruption", "severity"], columns="class", values="far")
            ss = ex.pivot_table(
                index=["corruption", "severity", "class"], columns="seed", values="far_mean"
            )
            out.append(
                {
                    "setting": setting,
                    "kind": kind,
                    "alpha": alpha,
                    "method": m,
                    "rms_excess": point[i],
                    "rms_lo": lo[i],
                    "rms_hi": hi[i],
                    "rank": int(order[i]),
                    "mean_excess": float(sub["excess"].mean()),
                    "violation_rate": float(sub["violation_rate"].mean()),
                    "class_spread": float(cs.std(axis=1, ddof=1).mean()),
                    "seed_spread": float(ss.std(axis=1, ddof=1).mean()),
                    "tpr": float(sub["tpr"].mean()),
                }
            )
    return pd.DataFrame(out)


def _by_prefix(rows):
    groups: dict[tuple, list] = {}
    for setting, kind, alpha, method, piv in rows:
        groups.setdefault((setting, kind, alpha), []).append((method, piv))
    return sorted(groups.items())


def rank_summary(stab: pd.DataFrame) -> pd.DataFrame:
    """Mean rank over the (kind, alpha) cells per setting; 1 = most stable."""
    return (
        stab.groupby(["setting", "method"])
        .agg(
            mean_rank=("rank", "mean"),
            wins=("rank", lambda r: int((r == 1).sum())),
            cells=("rank", "size"),
            rms_excess=("rms_excess", "mean"),
        )
        .reset_index()
    )


def auroc_by_condition(df: pd.DataFrame) -> pd.DataFrame:
    """Image AUROC (%) of normals vs anomalies WITHIN each condition (both are corrupted).

    Context for the FAR numbers: if the score shift is as large for anomalies as for
    normals, the detector can no longer separate them (AUROC near 50).
    """
    from anomcal.metrics import auroc

    rows = []
    pool = df[df["role"].isin(["heldout_normal", "anomaly"])]
    for (method, seed, cls, corr, sev), g in pool.groupby(
        ["method", "seed", "class", "corruption", "severity"]
    ):
        rows.append(
            {
                "method": method,
                "seed": int(seed),
                "class": cls,
                "corruption": corr,
                "severity": int(sev),
                "auroc": 100 * auroc(g["label"].to_numpy(), g["score"].to_numpy()),
            }
        )
    return pd.DataFrame(rows)


def recal_effect(hl: pd.DataFrame) -> pd.DataFrame:
    """How much of the FAR drift recalibration removes: |excess A| vs |excess B|."""
    a = hl[hl["setting"] == "A_no_recal"].drop(columns=["setting"])
    b = hl[hl["setting"] == "B_recal"].drop(columns=["setting"])
    m = a.merge(b, on=["method", "corruption", "severity", "kind", "alpha"], suffixes=("_A", "_B"))
    m["abs_excess_A"], m["abs_excess_B"] = m["excess_A"].abs(), m["excess_B"].abs()
    return m[
        [
            "method",
            "corruption",
            "severity",
            "kind",
            "alpha",
            "far_A",
            "far_B",
            "excess_A",
            "excess_B",
            "abs_excess_A",
            "abs_excess_B",
            "tpr_A",
            "tpr_B",
        ]
    ]


# ------------------------------------------------------------------------------ figures

METHOD_COLORS = {"baseline": "#8a8a8a", "dinov2_knn": "#2a9d5c", "patchcore": "#2a6fdb"}
KIND_MARKERS = {"conformal": "o", "tolerance": "s"}


def make_figures(hl: pd.DataFrame, fig_dir: Path) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir.mkdir(parents=True, exist_ok=True)
    methods = sorted(hl["method"].unique())
    clean = hl[(hl["corruption"] == "none") & (hl["setting"] == "A_no_recal")]
    corrs = [c for c in CORRUPTIONS if c in set(hl["corruption"])]
    paths = []

    # 1) Setting A: FAR vs severity (rows = alpha, columns = corruption)
    fig, axes = plt.subplots(
        len(ALPHAS),
        len(corrs),
        figsize=(3.1 * len(corrs), 2.6 * len(ALPHAS)),
        sharex=True,
        squeeze=False,
    )
    for i, alpha in enumerate(ALPHAS):
        for j, corr in enumerate(corrs):
            ax = axes[i][j]
            for method in methods:
                for kind, marker in KIND_MARKERS.items():
                    pts = pd.concat(
                        [
                            clean[
                                (clean["method"] == method)
                                & (clean["kind"] == kind)
                                & (clean["alpha"] == alpha)
                            ],
                            hl[
                                (hl["setting"] == "A_no_recal")
                                & (hl["method"] == method)
                                & (hl["kind"] == kind)
                                & (hl["alpha"] == alpha)
                                & (hl["corruption"] == corr)
                            ],
                        ]
                    ).sort_values("severity")
                    if pts.empty:
                        continue
                    ax.plot(
                        pts["severity"],
                        pts["far"],
                        marker=marker,
                        ms=4,
                        lw=1.2,
                        ls="-" if kind == "conformal" else ":",
                        color=METHOD_COLORS[method],
                    )
            ax.axhline(alpha, color="k", lw=0.9, ls="--")
            if not any(((hl["kind"] == "tolerance") & (hl["alpha"] == alpha)).to_numpy()):
                ax.text(
                    0.97,
                    0.05,
                    "tolerance limit:\nnot certifiable",
                    transform=ax.transAxes,
                    ha="right",
                    fontsize=6,
                    color="#555555",
                )
            ax.set_xticks([0, 1, 3, 5])
            ax.grid(alpha=0.3)
            if i == 0:
                ax.set_title(corr, fontsize=9)
            if j == 0:
                ax.set_ylabel(f"FAR (α={alpha:g})", fontsize=8)
            if i == len(ALPHAS) - 1:
                ax.set_xlabel("severity (0 = clean)", fontsize=8)
    handles = [
        plt.Line2D([], [], color=c, lw=2, label=m) for m, c in METHOD_COLORS.items() if m in methods
    ]
    handles += [
        plt.Line2D([], [], color="k", marker=mk, ls="-" if k == "conformal" else ":", ms=4, label=k)
        for k, mk in KIND_MARKERS.items()
    ]
    handles.append(plt.Line2D([], [], color="k", ls="--", lw=0.9, label="target α"))
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), fontsize=8)
    fig.suptitle(
        "Setting A (no recalibration): FAR on corrupted test normals, "
        "clean-calibrated threshold (mean over 12 classes)",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.97))
    p = fig_dir / "phase4_far_vs_severity.png"
    fig.savefig(p, dpi=110)
    plt.close(fig)
    paths.append(p)

    # 2) before/after recalibration at the highest severity, conformal threshold:
    #    top row FAR (recalibration restores it), bottom row TPR (what it costs)
    sev = int(hl["severity"].max())
    fig, axes = plt.subplots(2, len(ALPHAS), figsize=(4.4 * len(ALPHAS), 6.0), squeeze=False)
    for row, (col, label) in enumerate((("far", "FAR"), ("tpr", "detection rate (TPR)"))):
        for ax, alpha in zip(axes[row], ALPHAS, strict=True):
            for mi, method in enumerate(methods):
                for ci, corr in enumerate(corrs):
                    s = hl[
                        (hl["method"] == method)
                        & (hl["corruption"] == corr)
                        & (hl["severity"] == sev)
                        & (hl["kind"] == "conformal")
                        & (hl["alpha"] == alpha)
                    ].set_index("setting")
                    if len(s) < 2:
                        continue
                    x, color = ci + (mi - 1) * 0.22, METHOD_COLORS[method]
                    ya, yb = s.loc["A_no_recal", col], s.loc["B_recal", col]
                    ax.plot([x, x], [ya, yb], color=color, lw=1)
                    ax.plot(x, ya, "o", mfc="white", color=color, ms=6)
                    if col == "far":  # 95% class-bootstrap on the recalibrated FAR
                        err = [[yb - s.loc["B_recal", "far_lo"]], [s.loc["B_recal", "far_hi"] - yb]]
                        ax.errorbar(x, yb, yerr=err, fmt="o", color=color, ms=5, capsize=2, lw=1)
                    else:
                        ax.plot(x, yb, "o", color=color, ms=5)
            if col == "far":
                ax.axhline(alpha, color="k", ls="--", lw=0.9)
                ax.set_title(f"α={alpha:g}", fontsize=9)
            ax.set_xticks(range(len(corrs)), [c.replace("_", "\n") for c in corrs], fontsize=7)
            ax.grid(axis="y", alpha=0.3)
            ax.set_ylim(-0.03, 1.03)
        axes[row][0].set_ylabel(f"{label}, severity {sev} (mean over classes)", fontsize=8)
    handles = [
        plt.Line2D([], [], color=c, marker="o", ls="", label=m)
        for m, c in METHOD_COLORS.items()
        if m in methods
    ]
    handles += [
        plt.Line2D([], [], color="k", marker="o", mfc="white", ls="", label="A: no recal."),
        plt.Line2D([], [], color="k", marker="o", ls="", label="B: recalibrated"),
        plt.Line2D([], [], color="k", ls="--", lw=0.9, label="target α"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), fontsize=7)
    fig.suptitle(
        "Before/after recalibration (split conformal; bars on B = 95% class-bootstrap)", fontsize=10
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.96))
    p = fig_dir / "phase4_recalibration.png"
    fig.savefig(p, dpi=110)
    plt.close(fig)
    paths.append(p)
    return paths


# ------------------------------------------------------------------------------ entry point


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Phase 4: calibration under corruption (CPU).")
    ap.add_argument("--scores", type=Path, default=Path("results/scores"))
    ap.add_argument("--out", type=Path, default=Path("results/phase4"))
    ap.add_argument("--figures", type=Path, default=Path("docs/figures"))
    ap.add_argument("--partitions", type=int, default=N_PARTITIONS)
    ap.add_argument("--boot", type=int, default=N_BOOT)
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args(argv)

    df = load_calib(args.scores)
    exp = run_all(df, args.partitions)
    pc = per_class(exp)
    hl = headline(pc, args.boot)
    stab = stability(pc, exp, args.boot)
    outputs = {
        "experiments.csv": exp,
        "per_class.csv": pc,
        "headline.csv": hl,
        "stability.csv": stab,
        "rank_summary.csv": rank_summary(stab),
        "recalibration_effect.csv": recal_effect(hl),
        "auroc_by_condition.csv": auroc_by_condition(df),
    }
    args.out.mkdir(parents=True, exist_ok=True)
    for name, table in outputs.items():
        table.to_csv(args.out / name, index=False, float_format="%.5g")
        print(f"  -> {args.out / name} ({len(table)} rows)")
    if not args.no_figures:
        for p in make_figures(hl, args.figures):
            print(f"  -> {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
