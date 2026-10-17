"""Phase 4b: recalibration with only m corrupted calibration normals (Setting B').

Same 108 experiments, the same 100 partitions (same key as Phase 4), the 12 corrupted
conditions. In every partition we draw ``DRAWS`` random subsets of m of the n_cal corrupted
calibration normals (nested across m: the m = 10 subset is inside the m = 25 one), compute
the split-conformal and tolerance-limit (delta = 5%) threshold from the m scores only, and
measure the FAR on the corrupted test half and the TPR on the corrupted anomalies.
So every (experiment, condition, m) has R x DRAWS = 100 x 5 = 500 draws; the draws of one
partition share the same test half, so they are correlated (we do not pretend they are
500 independent samples).

Exact law to compare with. Calibration (m scores) and test normals come from the same
condition, hence are exchangeable, and the count of false alarms among n_test test normals
is Beta-binomial(n_test, m + 1 - k, k), where k is the rank of the threshold among the m
scores (see :mod:`anomcal.calibration`). For each cell we store the observed mean FAR, std
of FAR and violation rate (share of draws with FAR > alpha) next to this law's values. A
cell the kind cannot certify (rank None) is ``not_certifiable``, never clamped.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from anomcal.calibration import false_alarm_count_law
from anomcal.phase3 import ALPHAS, KINDS, N_PARTITIONS, n_calibration
from anomcal.phase4 import N_BOOT, class_bootstrap, load_calib, rank_for
from anomcal.seeding import item_rng

MS = (10, 25, 50, 100)
DRAWS = 5  # random m-subsets per partition


def law_stats(n_test: int, m: int, k: int, alpha: float) -> tuple[float, float, float]:
    """(mean FAR, std FAR, P(FAR > alpha)) of the exact Beta-binomial law."""
    law = false_alarm_count_law(m, k, n_test)
    # alarm count c gives FAR = c / n_test > alpha  <=>  c > floor(alpha * n_test)
    return (
        law.mean() / n_test,
        law.std() / n_test,
        float(law.sf(np.floor(alpha * n_test + 1e-9))),
    )


def run_condition(
    shifted: np.ndarray,
    anomalous: np.ndarray,
    key: tuple,
    ms: Sequence[int] = MS,
    alphas: Sequence[float] = ALPHAS,
    n_partitions: int = N_PARTITIONS,
    draws: int = DRAWS,
) -> list[dict]:
    """One corrupted condition of one experiment -> one row per (m, kind, alpha).

    The rows hold sums (n, sum of FAR, sum of FAR^2, violations, sum of TPR) so that
    conditions and seeds can be pooled exactly later.
    """
    n_pool = shifted.size
    n_cal = n_calibration(n_pool)
    n_test = n_pool - n_cal
    anom_sorted = np.sort(anomalous)
    acc: dict[tuple, list[float]] = {}
    for r in range(n_partitions):
        perm = item_rng("phase4-partition", *key, r).permutation(n_pool)  # as in Phase 4
        cal, test = shifted[perm[:n_cal]], np.sort(shifted[perm[n_cal:]])
        order = item_rng("phase4m-draw", *key, r).random((draws, n_cal)).argsort(axis=1)
        for m in ms:
            sub = np.sort(cal[order[:, :m]], axis=1)  # (draws, m), each row sorted
            for kind in KINDS:
                for alpha in alphas:
                    rank = rank_for(kind, m, alpha)
                    cell = acc.setdefault((m, kind, alpha), [0.0, 0.0, 0.0, 0.0, 0.0])
                    if rank is None:
                        continue
                    t = sub[:, rank - 1]
                    far = (n_test - np.searchsorted(test, t, side="right")) / n_test
                    tpr = (anom_sorted.size - np.searchsorted(anom_sorted, t, side="right")) / (
                        anom_sorted.size
                    )
                    cell[0] += draws
                    cell[1] += far.sum()
                    cell[2] += (far**2).sum()
                    cell[3] += (far > alpha).sum()
                    cell[4] += tpr.sum()
    rows = []
    for (m, kind, alpha), (n, s1, s2, viol, st) in acc.items():
        row = {"m": m, "kind": kind, "alpha": alpha, "n_test": n_test, "n": n,
               "sum_far": s1, "sum_far2": s2, "violations": viol, "sum_tpr": st,
               "status": "certified" if n else "not_certifiable"}  # fmt: skip
        k = rank_for(kind, m, alpha)
        if k is not None:
            row["law_mean"], row["law_std"], row["law_viol"] = law_stats(n_test, m, k, alpha)
        rows.append(row)
    return rows


def run_all(df: pd.DataFrame, n_partitions: int = N_PARTITIONS, ms=MS) -> pd.DataFrame:
    out = []
    for (method, seed, cls), g in df.groupby(["method", "seed", "class"], sort=True):
        key = (method, int(seed), cls)
        pool = g[(g["role"] == "heldout_normal") & (g["corruption"] != "none")]
        anom = g[(g["role"] == "anomaly") & (g["corruption"] != "none")]
        for (corr, sev), cond in pool.groupby(["corruption", "severity"], sort=True):
            cond = cond.sort_values("image_id")  # same order as Phase 4, so same partitions
            a = anom[(anom["corruption"] == corr) & (anom["severity"] == sev)]
            for r in run_condition(
                cond["score"].to_numpy(np.float64), a["score"].to_numpy(np.float64),
                key, ms=ms, n_partitions=n_partitions,
            ):  # fmt: skip
                out.append({"method": method, "seed": int(seed), "class": cls,
                            "corruption": corr, "severity": int(sev), **r})  # fmt: skip
    return pd.DataFrame(out)


# ------------------------------------------------------------------------ aggregation

_SUMS = ["n", "sum_far", "sum_far2", "violations", "sum_tpr"]
_CELL = ["m", "kind", "alpha"]


def pool(raw: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    """Pool the sums over everything not in ``by`` (exact for mean, std, violation rate, TPR).

    The law values are averaged weighted by the number of draws (they depend on n_test).
    """
    ok = raw[raw["status"] == "certified"].copy()
    for c in ("law_mean", "law_std", "law_viol"):
        ok[c + "_w"] = ok[c] * ok["n"]
    g = ok.groupby(by)[[*_SUMS, "law_mean_w", "law_std_w", "law_viol_w"]].sum().reset_index()
    out = g[by].copy()
    out["draws"] = g["n"]
    out["far"] = g["sum_far"] / g["n"]
    # pooled std over draws (population form; n is large)
    out["far_std"] = np.sqrt(np.maximum(g["sum_far2"] / g["n"] - out["far"] ** 2, 0))
    out["violation_rate"] = g["violations"] / g["n"]
    out["tpr"] = g["sum_tpr"] / g["n"]
    out["law_far"] = g["law_mean_w"] / g["n"]
    out["law_far_std"] = g["law_std_w"] / g["n"]  # averaged within-cell std (not total variance)
    out["law_violation_rate"] = g["law_viol_w"] / g["n"]
    if "alpha" in by:
        out["excess"] = out["far"] - out["alpha"]
    return out


def certifiability(raw: pd.DataFrame) -> pd.DataFrame:
    s = raw.groupby(_CELL)["status"].agg(lambda x: "/".join(sorted(set(x)))).reset_index()
    return s.pivot(index=["kind", "alpha"], columns="m", values="status").reset_index()


def headline(per_class: pd.DataFrame, n_boot: int = N_BOOT) -> pd.DataFrame:
    """Per (method, m, kind, alpha): class-bootstrap mean + 95% interval for each metric."""
    idx = ["method", "m", "kind", "alpha"]
    out = None
    for col in ("far", "violation_rate", "tpr"):
        mat = per_class.pivot_table(index=idx, columns="class", values=col).dropna()
        mean, lo, hi = class_bootstrap(mat.to_numpy(), n_boot, seed_tag="phase4m-boot")
        part = mat.index.to_frame(index=False)
        part[col], part[col + "_lo"], part[col + "_hi"] = mean, lo, hi
        out = part if out is None else out.merge(part, on=idx)
    laws = per_class.groupby(idx)[["law_far", "law_violation_rate"]].mean().reset_index()
    out = out.merge(laws, on=idx)
    out["excess"] = out["far"] - out["alpha"]
    return out


# ---------------------------------------------------------------------------- figure

METHOD_COLORS = {"baseline": "#8a8a8a", "dinov2_knn": "#2a9d5c", "patchcore": "#2a6fdb"}
KIND_MARKERS = {"conformal": "o", "tolerance": "s"}


def make_figure(hl: pd.DataFrame, fig_dir: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir.mkdir(parents=True, exist_ok=True)
    ms = sorted(hl["m"].unique())
    pos = {m: i for i, m in enumerate(ms)}
    methods = sorted(hl["method"].unique())
    fig, axes = plt.subplots(2, len(ALPHAS), figsize=(4.2 * len(ALPHAS), 6.2), squeeze=False)
    for row, (col, label) in enumerate((("far", "mean FAR"), ("violation_rate", "P(FAR > α)"))):
        for ax, alpha in zip(axes[row], ALPHAS, strict=True):
            for mi, method in enumerate(methods):
                for kind, marker in KIND_MARKERS.items():
                    s = hl[(hl["method"] == method) & (hl["kind"] == kind) & (hl["alpha"] == alpha)]
                    s = s.sort_values("m")
                    x = np.array([pos[m] for m in s["m"]]) + (mi - 1) * 0.07
                    ax.errorbar(x, s[col], yerr=[s[col] - s[col + "_lo"], s[col + "_hi"] - s[col]],
                                marker=marker, ms=4, lw=1, capsize=2, color=METHOD_COLORS[method],
                                ls="-" if kind == "conformal" else ":")  # fmt: skip
            for kind in KIND_MARKERS:  # exact law (same for every method)
                s = hl[(hl["kind"] == kind) & (hl["alpha"] == alpha)]
                s = s.groupby("m")[["law_far", "law_violation_rate"]].mean().reset_index()
                lc = "law_far" if col == "far" else "law_violation_rate"
                ax.plot([pos[m] for m in s["m"]], s[lc], marker="x", ms=7, ls="none", color="k")
            ax.set_xlim(-0.5, len(ms) - 0.5)
            if col == "far":
                ax.axhline(alpha, color="k", ls="--", lw=0.9)
                ax.set_ylim(0, 1.15 * alpha)
            else:
                ax.axhline(0.05, color="#999999", ls="--", lw=0.8)  # delta = 5%
            missing = [m for m in ms if hl[(hl["alpha"] == alpha) & (hl["m"] == m)].empty]
            for m in missing:
                ax.text(pos[m], 0.02, "not\ncertif.", ha="center", fontsize=6, color="#555555",
                        transform=ax.get_xaxis_transform())  # fmt: skip
            ax.set_xticks(range(len(ms)), [str(m) for m in ms])
            ax.grid(alpha=0.3)
            if row == 0:
                ax.set_title(f"α={alpha:g}", fontsize=9)
            if row == 1:
                ax.set_xlabel("m = number of corrupted calibration normals", fontsize=8)
        axes[row][0].set_ylabel(label)
    handles = [plt.Line2D([], [], color=c, lw=2, label=mt) for mt, c in METHOD_COLORS.items()]
    handles += [plt.Line2D([], [], color="k", marker=mk, ls="-" if k == "conformal" else ":",
                           ms=4, label=k) for k, mk in KIND_MARKERS.items()]  # fmt: skip
    handles += [plt.Line2D([], [], color="k", marker="x", ls="", label="exact law")]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), fontsize=7)
    fig.suptitle("Recalibration with m corrupted normals (bars: 95% class-bootstrap; "
                 "dashed: α in top row, δ = 5% in bottom row)", fontsize=9)  # fmt: skip
    fig.tight_layout(rect=(0, 0.05, 1, 0.96))
    p = fig_dir / "phase4_small_m.png"
    fig.savefig(p, dpi=110)
    plt.close(fig)
    return p


# ------------------------------------------------------------------------ entry point


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Phase 4b: small-m recalibration (CPU).")
    ap.add_argument("--scores", type=Path, default=Path("results/scores"))
    ap.add_argument("--out", type=Path, default=Path("results/phase4"))
    ap.add_argument("--figures", type=Path, default=Path("docs/figures"))
    ap.add_argument("--partitions", type=int, default=N_PARTITIONS)
    ap.add_argument("--boot", type=int, default=N_BOOT)
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args(argv)

    raw = run_all(load_calib(args.scores), args.partitions)
    by_exp = pool(raw, ["method", "seed", "class", *_CELL])
    per_class = pool(raw, ["method", "class", *_CELL])
    by_cond = pool(raw, ["method", "corruption", "severity", *_CELL])
    hl = headline(per_class, args.boot)
    outputs = {
        "small_m_experiments.csv": by_exp,
        "small_m_per_class.csv": per_class,
        "small_m_by_condition.csv": by_cond,
        "small_m_headline.csv": hl,
        "small_m_certifiability.csv": certifiability(raw),
    }
    args.out.mkdir(parents=True, exist_ok=True)
    for name, table in outputs.items():
        table.to_csv(args.out / name, index=False, float_format="%.5g")
        print(f"  -> {args.out / name} ({len(table)} rows)")
    if not args.no_figures:
        print(f"  -> {make_figure(hl, args.figures)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
