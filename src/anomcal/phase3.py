"""Phase 3: false-alarm calibration on CLEAN real scores (CPU, from results/scores).

For every (method, seed, class) of the `calib` protocol, i.e. one experiment:

1. Take the held-out normal pool (role ``heldout_normal``; never in the memory bank:
   enforced by the split protocol, and re-checked here against the split manifests'
   logic: memory images are never scored) and the anomalies, clean images only.
2. Draw R = 100 random calibration/test partitions of the pool (50/50, as in
   docs/SPLIT_PROTOCOL.md), each seeded by (method, seed, class, r).
3. For each alpha: compute the split-conformal threshold and the tolerance-limit
   threshold (delta = 0.05) from the calibration half only, or record
   "not_certifiable". Then measure the FAR on the test half and the detection
   rate (TPR) on the anomalies.

Why the realised FAR must follow the Beta-binomial law here: a uniformly random split of
a fixed pool puts the scores in uniformly random order, and the false-alarm count
depends only on that order. So, apart from ties, the count of false alarms among the
m test normals is EXACTLY Beta-binomial(m, n+1-k, k), whatever the detector. This part
checks our implementation (any deviation would be a bug), not the data.

The data-dependent check is the **cross-split check**: calibrate on the held-out *train*
normals and test on the *official test* normals. They were collected separately in VisA,
so this asks whether they are actually exchangeable.

Outputs (small CSVs, under the 5 MiB repo guard) and figures: see :func:`main`.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from anomcal.calibration import (
    conformal_alpha_min,
    conformal_threshold,
    false_alarm_count_law,
    tolerance_m_min,
    tolerance_threshold,
    wilson_interval,
)
from anomcal.seeding import item_rng

ALPHAS = (0.01, 0.05, 0.10)
DELTA = 0.05
N_PARTITIONS = 100
CAL_FRACTION = 0.5
KINDS = ("conformal", "tolerance")


def load_clean_calib(scores_dir: Path) -> pd.DataFrame:
    files = sorted(Path(scores_dir).glob("calib_*_seed*.parquet"))
    if not files:
        raise FileNotFoundError(f"no calib score files in {scores_dir}")
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    df = df[(df["corruption"] == "none") & (df["protocol"] == "calib")]
    if df["image_id"].duplicated().any() and df.duplicated(["method", "seed", "image_id"]).any():
        raise ValueError("duplicate (method, seed, image) rows in clean scores")
    return df


def n_calibration(pool_size: int, fraction: float = CAL_FRACTION) -> int:
    """Same rounding as anomcal.data.splits.partition_heldout."""
    return min(max(int(round(pool_size * fraction)), 1), pool_size - 1)


def _threshold(kind: str, cal: np.ndarray, alpha: float):
    if kind == "conformal":
        return conformal_threshold(cal, alpha)
    return tolerance_threshold(cal, alpha, DELTA)


def run_experiment(
    normal: np.ndarray,
    anomalous: np.ndarray,
    key: tuple,
    alphas: Sequence[float] = ALPHAS,
    n_partitions: int = N_PARTITIONS,
) -> list[dict]:
    """R random partitions of one experiment's normal pool -> one row per (r, alpha, kind)."""
    n_pool = normal.size
    n_cal = n_calibration(n_pool)
    rows = []
    for r in range(n_partitions):
        perm = item_rng("phase3-partition", *key, r).permutation(n_pool)
        cal, test = normal[perm[:n_cal]], normal[perm[n_cal:]]
        for alpha in alphas:
            for kind in KINDS:
                t = _threshold(kind, cal, alpha)
                row = {
                    "partition": r, "alpha": alpha, "kind": kind, "n_cal": n_cal,
                    "n_test": test.size, "status": t.status, "rank": t.rank,
                    "threshold": t.value, "false_alarms": None, "far": None, "tpr": None,
                }  # fmt: skip
                if t.certified:
                    fa = int((test > t.value).sum())
                    row.update(false_alarms=fa, far=fa / test.size)
                    row["tpr"] = float((anomalous > t.value).mean()) if anomalous.size else None
                rows.append(row)
    return rows


def run_all(df: pd.DataFrame, n_partitions: int = N_PARTITIONS) -> pd.DataFrame:
    out = []
    for (method, seed, cls), g in df.groupby(["method", "seed", "class"], sort=True):
        normal = g.loc[g["role"] == "heldout_normal", "score"].to_numpy(np.float64)
        anom = g.loc[g["role"] == "anomaly", "score"].to_numpy(np.float64)
        if (g.loc[g["role"] == "heldout_normal", "label"] != 0).any():
            raise ValueError(f"{method}/{seed}/{cls}: anomaly inside the normal pool")
        for row in run_experiment(
            normal, anom, (method, int(seed), cls), n_partitions=n_partitions
        ):
            out.append({"method": method, "seed": int(seed), "class": cls, **row})
    return pd.DataFrame(out)


# ------------------------------------------------------------------------- summaries


def certifiability(parts: pd.DataFrame) -> pd.DataFrame:
    """Per class: n_cal and whether each (kind, alpha) is certifiable (same for all methods)."""
    rows = []
    for cls, g in parts.groupby("class"):
        n = int(g["n_cal"].iloc[0])
        row = {"class": cls, "n_cal": n, "conformal_alpha_min": conformal_alpha_min(n)}
        for kind in KINDS:
            for alpha in ALPHAS:
                s = g[(g["kind"] == kind) & (g["alpha"] == alpha)]["status"].unique()
                if len(s) != 1:
                    raise AssertionError(f"{cls}: status varies across partitions: {s}")
                row[f"{kind}_{alpha:g}"] = s[0]
        rows.append(row)
    out = pd.DataFrame(rows)
    out["tolerance_m_min_1pct"] = tolerance_m_min(0.01, DELTA)
    out["tolerance_m_min_5pct"] = tolerance_m_min(0.05, DELTA)
    return out


def method_summary(parts: pd.DataFrame) -> pd.DataFrame:
    """Per (method, kind, alpha): pooled FAR + Wilson, spread across partitions, TPR."""
    ok = parts[parts["status"] == "certified"]
    rows = []
    for (method, kind, alpha), g in ok.groupby(["method", "kind", "alpha"]):
        fa, nt = int(g["false_alarms"].sum()), int(g["n_test"].sum())
        lo, hi = wilson_interval(fa, nt)
        # Theory, averaged over the partitions (each has its own n, k, m):
        # E[FAR] and P(empirical FAR > alpha) under the exact Beta-binomial law.
        laws = {
            (n, k, m): false_alarm_count_law(int(n), int(k), int(m))
            for n, k, m in g[["n_cal", "rank", "n_test"]].drop_duplicates().to_numpy()
        }
        keys = list(zip(g["n_cal"], g["rank"], g["n_test"], strict=True))
        theo = float(np.mean([laws[key].mean() / key[2] for key in keys]))
        theo_above = float(np.mean([laws[key].sf(np.floor(alpha * key[2])) for key in keys]))
        rows.append({
            "method": method, "kind": kind, "alpha": alpha,
            "experiments": g.groupby(["seed", "class"]).ngroups,
            "partitions": len(g), "far_pooled": fa / nt, "wilson_lo": lo, "wilson_hi": hi,
            "far_theory_mean": theo, "far_std": g["far"].std(ddof=1),
            "far_p05": g["far"].quantile(0.05), "far_p95": g["far"].quantile(0.95),
            "share_far_above_alpha": float((g["far"] > alpha).mean()),
            "share_far_above_alpha_theory": theo_above,
            "tpr_mean": g["tpr"].mean(),
        })  # fmt: skip
    return pd.DataFrame(rows)


def law_check(parts: pd.DataFrame, n_sim: int = 2000) -> pd.DataFrame:
    """Per experiment and (kind, alpha): observed false-alarm counts vs Beta-binomial.

    Mean and 5/50/95% quantiles against the exact law, and a KS statistic whose p-value
    is obtained by simulation from the same discrete law (the classic KS p-value is only
    valid for continuous distributions). Under the null, about 5% of experiments have
    p < 0.05; the summary reports that share.
    """
    ok = parts[parts["status"] == "certified"]
    rows = []
    for (method, seed, cls, kind, alpha), g in ok.groupby(
        ["method", "seed", "class", "kind", "alpha"]
    ):
        n, k, m = int(g["n_cal"].iloc[0]), int(g["rank"].iloc[0]), int(g["n_test"].iloc[0])
        law = false_alarm_count_law(n, k, m)
        x = np.sort(g["false_alarms"].to_numpy())
        support = np.arange(m + 1)
        cdf = law.cdf(support)

        def ks(sample, support=support, cdf=cdf):
            ecdf = np.searchsorted(np.sort(sample), support, side="right") / sample.size
            return float(np.max(np.abs(ecdf - cdf)))

        d_obs = ks(x)
        rng = item_rng("phase3-ks", method, int(seed), cls, kind, float(alpha))
        sims = law.rvs(size=(n_sim, x.size), random_state=rng)
        p = float((np.array([ks(s) for s in sims]) >= d_obs - 1e-12).mean())
        rows.append({
            "method": method, "seed": int(seed), "class": cls, "kind": kind, "alpha": alpha,
            "n_cal": n, "rank": k, "n_test": m,
            "mean_fa_obs": x.mean(), "mean_fa_law": law.mean(),
            "q05_obs": np.quantile(x, 0.05), "q05_law": law.ppf(0.05),
            "q50_obs": np.quantile(x, 0.5), "q50_law": law.ppf(0.5),
            "q95_obs": np.quantile(x, 0.95), "q95_law": law.ppf(0.95),
            "ks_stat": d_obs, "ks_p": p,
        })  # fmt: skip
    return pd.DataFrame(rows)


def cross_split_check(df: pd.DataFrame, alphas: Sequence[float] = ALPHAS) -> pd.DataFrame:
    """Calibrate on held-out TRAIN normals, test on OFFICIAL TEST normals (no resampling).

    If VisA's train and test normals are exchangeable, the false-alarm count follows
    Beta-binomial(m, n+1-k, k); we report the observed count and its two-sided tail
    probability under that law.
    """
    rows = []
    normal = df[df["role"] == "heldout_normal"]
    for (method, seed, cls), g in normal.groupby(["method", "seed", "class"]):
        cal = g.loc[g["official_split"] == "train", "score"].to_numpy(np.float64)
        test = g.loc[g["official_split"] == "test", "score"].to_numpy(np.float64)
        for alpha in alphas:
            t = conformal_threshold(cal, alpha)
            row = {"method": method, "seed": int(seed), "class": cls, "alpha": alpha,
                   "n_cal": cal.size, "n_test": test.size, "status": t.status}  # fmt: skip
            if t.certified:
                fa = int((test > t.value).sum())
                law = false_alarm_count_law(cal.size, t.rank, test.size)
                row.update(
                    false_alarms=fa,
                    far=fa / test.size,
                    far_expected=law.mean() / test.size,
                    p_upper=float(law.sf(fa - 1)),  # P(count >= observed)
                    p_lower=float(law.cdf(fa)),  # P(count <= observed)
                )
            rows.append(row)
    return pd.DataFrame(rows)


def cross_split_summary(cross: pd.DataFrame) -> pd.DataFrame:
    """Pooled over classes and seeds. Wilson intervals treat test images as independent,
    but the 3 seeds reuse the same official test normals, so they are too narrow."""
    rows = []
    ok = cross[cross["status"] == "certified"]
    for (method, alpha), g in ok.groupby(["method", "alpha"]):
        fa, nt = int(g["false_alarms"].sum()), int(g["n_test"].sum())
        lo, hi = wilson_interval(fa, nt)
        rows.append({
            "method": method, "alpha": alpha, "far_pooled": fa / nt, "wilson_lo": lo,
            "wilson_hi": hi,
            "far_expected": float((g["far_expected"] * g["n_test"]).sum() / nt),
            "share_p_upper_lt_0.05": float((g["p_upper"] < 0.05).mean()),
            "share_p_lower_lt_0.05": float((g["p_lower"] < 0.05).mean()),
        })  # fmt: skip
    return pd.DataFrame(rows)


def auroc_by_class(df: pd.DataFrame) -> pd.DataFrame:
    from anomcal.metrics import auroc

    rows = []
    for (method, seed, cls), g in df.groupby(["method", "seed", "class"]):
        rows.append(
            {
                "method": method,
                "seed": int(seed),
                "class": cls,
                "auroc": 100 * auroc(g["label"].to_numpy(), g["score"].to_numpy()),
            }
        )
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------- figures


def make_figures(summary: pd.DataFrame, parts: pd.DataFrame, fig_dir: Path) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    methods = sorted(summary["method"].unique())
    colors = {"conformal": "#2a6fdb", "tolerance": "#d9822b"}

    # 1) mean FAR vs target alpha, per method and threshold kind
    fig, axes = plt.subplots(1, len(methods), figsize=(4 * len(methods), 3.6), sharey=True)
    for ax, method in zip(np.atleast_1d(axes), methods, strict=True):
        for i, kind in enumerate(KINDS):
            s = summary[(summary["method"] == method) & (summary["kind"] == kind)].sort_values(
                "alpha"
            )
            # x position from the alpha VALUE (a kind may skip an alpha it cannot certify)
            x = np.array([ALPHAS.index(a) for a in s["alpha"]]) + (i - 0.5) * 0.18
            ax.errorbar(
                x, s["far_pooled"],
                yerr=[s["far_pooled"] - s["far_p05"], s["far_p95"] - s["far_pooled"]],
                fmt="o" if kind == "conformal" else "s", color=colors[kind], capsize=3,
                label=f"{kind} (5–95% over partitions)",
            )  # fmt: skip
        for j, a in enumerate(ALPHAS):
            ax.hlines(a, j - 0.35, j + 0.35, colors="#555555", linestyles="dashed", linewidth=1)
            for i, kind in enumerate(KINDS):
                if summary[(summary["method"] == method) & (summary["kind"] == kind)
                           & (summary["alpha"] == a)].empty:  # fmt: skip
                    ax.annotate("not\ncertifiable", (j + (i - 0.5) * 0.18, a), fontsize=6,
                                ha="center", va="bottom", color="#555555",
                                xytext=(0, 6), textcoords="offset points")  # fmt: skip
        ax.set_xticks(range(len(ALPHAS)), [f"α={a:g}" for a in ALPHAS])
        ax.set_title(method)
        ax.grid(axis="y", alpha=0.3)
    np.atleast_1d(axes)[0].set_ylabel("empirical FAR on test normals")
    np.atleast_1d(axes)[0].legend(fontsize=7, loc="upper left")
    fig.suptitle("Phase 3, clean VisA scores: FAR vs target (dashed = α)", fontsize=10)
    fig.tight_layout()
    p = fig_dir / "phase3_far_vs_alpha.png"
    fig.savefig(p, dpi=110)
    plt.close(fig)
    paths.append(p)

    # 2) one experiment's false-alarm counts vs the exact Beta-binomial law
    ex = parts[
        (parts["status"] == "certified") & (parts["kind"] == "conformal") & (parts["alpha"] == 0.05)
    ]
    method, seed, cls = sorted(
        ex[["method", "seed", "class"]].drop_duplicates().itertuples(index=False)
    )[0]
    g = ex[(ex["method"] == method) & (ex["seed"] == seed) & (ex["class"] == cls)]
    n, k, m = int(g["n_cal"].iloc[0]), int(g["rank"].iloc[0]), int(g["n_test"].iloc[0])
    law = false_alarm_count_law(n, k, m)
    support = np.arange(int(law.ppf(0.999)) + 2)
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    counts = np.bincount(g["false_alarms"].astype(int), minlength=support.size)[: support.size]
    ax.bar(
        support,
        counts / counts.sum(),
        color="#2a6fdb",
        alpha=0.6,
        label=f"observed ({len(g)} partitions)",
    )
    ax.plot(support, law.pmf(support), "o-", color="#222222", ms=3, label="Beta-binomial law")
    ax.set_xlabel(f"false alarms among {m} test normals")
    ax.set_ylabel("probability")
    ax.set_title(f"{method}, {cls}, seed {seed}: conformal α=0.05 (n={n}, k={k})", fontsize=9)
    ax.legend(fontsize=8)
    fig.tight_layout()
    p = fig_dir / "phase3_count_law_example.png"
    fig.savefig(p, dpi=110)
    plt.close(fig)
    paths.append(p)
    return paths


# ------------------------------------------------------------------------- entry point


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Phase 3: calibration on clean scores (CPU).")
    ap.add_argument("--scores", type=Path, default=Path("results/scores"))
    ap.add_argument("--out", type=Path, default=Path("results/phase3"))
    ap.add_argument("--figures", type=Path, default=Path("docs/figures"))
    ap.add_argument("--partitions", type=int, default=N_PARTITIONS)
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args(argv)

    df = load_clean_calib(args.scores)
    parts = run_all(df, args.partitions)
    args.out.mkdir(parents=True, exist_ok=True)
    summary = method_summary(parts)
    cert = certifiability(parts)
    laws = law_check(parts)
    cross = cross_split_check(df)
    per_class = (
        parts[parts["status"] == "certified"]
        .groupby(["method", "class", "kind", "alpha"])
        .agg(far_mean=("far", "mean"), far_std=("far", "std"), tpr_mean=("tpr", "mean"))
        .reset_index()
        .merge(auroc_by_class(df).groupby(["method", "class"])["auroc"].mean().reset_index(),
               on=["method", "class"])
    )  # fmt: skip
    outputs = {
        "summary_by_method.csv": summary,
        "certifiability.csv": cert,
        "law_check.csv": laws,
        "cross_split_check.csv": cross,
        "cross_split_summary.csv": cross_split_summary(cross),
        "per_class.csv": per_class,
    }
    for name, table in outputs.items():
        table.to_csv(args.out / name, index=False, float_format="%.6g")
    print(
        f"{len(parts)} partition-rows from "
        f"{parts.groupby(['method', 'seed', 'class']).ngroups} experiments"
    )
    for name in outputs:
        print(f"  -> {args.out / name}")
    if not args.no_figures:
        for p in make_figures(summary, parts, args.figures):
            print(f"  -> {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
