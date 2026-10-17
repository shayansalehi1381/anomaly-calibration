"""AUROC tables from consolidated score files (CPU; used for the Phase 2 sanity check).

Image AUROC is computed per (protocol, method, seed, class) on **clean** images, then
summarised as mean ± std over seeds. The `official` protocol (all train normals in the
memory bank, official test set) is the one comparable with published numbers. The
`calib` protocol uses half the memory bank and a larger normal pool, so it is labelled
separately.

Reference numbers for the sanity check come from SPARC (arXiv:2608.18585, Table 1,
VisA, resize + center crop): PatchCore WRN-50 ~94.3, AnomalyDINO ~95.3 image AUROC
(mean over classes). They use a different protocol and preprocessing, so they are a
sanity check, never a claim; a gap of more than ~5 points is flagged for investigation.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from anomcal.metrics import auroc

SPARC_VISA_REFERENCE = {"patchcore": 94.3, "dinov2_knn": 95.3}  # AnomalyDINO for dinov2_knn
FLAG_GAP_POINTS = 5.0


def clean_auroc(scores: pd.DataFrame) -> pd.DataFrame:
    """One row per (protocol, method, seed, class) with image AUROC (in %) on clean images."""
    clean = scores[scores["corruption"] == "none"]
    rows = []
    for key, g in clean.groupby(["protocol", "method", "seed", "class"], sort=True):
        if g["label"].nunique() < 2:
            continue
        rows.append((*key, 100.0 * auroc(g["label"].to_numpy(), g["score"].to_numpy()), len(g)))
    return pd.DataFrame(rows, columns=["protocol", "method", "seed", "class", "auroc", "n"])


def summarise(per_seed: pd.DataFrame) -> pd.DataFrame:
    """Mean ± std over seeds per (protocol, method, class), plus a class-mean row."""
    per_class = (
        per_seed.groupby(["protocol", "method", "class"])["auroc"]
        .agg(mean="mean", std=lambda x: x.std(ddof=1) if len(x) > 1 else np.nan, seeds="count")
        .reset_index()
    )
    # class mean per seed first, then mean/std of those over seeds
    seed_means = per_seed.groupby(["protocol", "method", "seed"])["auroc"].mean().reset_index()
    overall = (
        seed_means.groupby(["protocol", "method"])["auroc"]
        .agg(mean="mean", std=lambda x: x.std(ddof=1) if len(x) > 1 else np.nan, seeds="count")
        .reset_index()
    )
    overall["class"] = "MEAN"
    return pd.concat([per_class, overall], ignore_index=True)


def sanity_check(summary: pd.DataFrame) -> list[str]:
    """Compare official-protocol class means with the SPARC reference numbers."""
    lines = []
    official = summary[(summary["protocol"] == "official") & (summary["class"] == "MEAN")]
    for _, row in official.iterrows():
        ref = SPARC_VISA_REFERENCE.get(row["method"])
        if ref is None:
            continue
        gap = row["mean"] - ref
        verdict = "FLAG: investigate" if abs(gap) > FLAG_GAP_POINTS else "consistent"
        lines.append(
            f"{row['method']}: ours {row['mean']:.1f} vs SPARC reference {ref:.1f} "
            f"(gap {gap:+.1f} points, {int(row['seeds'])} seed(s)) -> {verdict}"
        )
    return lines


def format_markdown(summary: pd.DataFrame, protocol: str) -> str:
    sub = summary[summary["protocol"] == protocol]
    methods = sorted(sub["method"].unique())
    classes = [c for c in sorted(sub["class"].unique()) if c != "MEAN"] + ["MEAN"]
    lines = ["| class | " + " | ".join(methods) + " |", "|---|" + "---:|" * len(methods)]
    for c in classes:
        cells = []
        for m in methods:
            r = sub[(sub["class"] == c) & (sub["method"] == m)]
            if r.empty:
                cells.append("–")
                continue
            mean, std = r["mean"].iloc[0], r["std"].iloc[0]
            cells.append(f"{mean:.1f}" + (f" ± {std:.1f}" if pd.notna(std) else ""))
        name = f"**{c}**" if c == "MEAN" else c
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def load_scores(scores_dir: Path) -> pd.DataFrame:
    files = sorted(Path(scores_dir).glob("*_seed*.parquet"))
    if not files:
        raise FileNotFoundError(f"no consolidated score files in {scores_dir}")
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Clean image-AUROC tables + SPARC sanity check.")
    p.add_argument("scores_dir", type=Path, help="e.g. outputs/extract/full/scores")
    args = p.parse_args(argv)
    summary = summarise(clean_auroc(load_scores(args.scores_dir)))
    for protocol in sorted(summary["protocol"].unique()):
        label = {
            "official": "official VisA 1cls split (memory = all train normals)",
            "calib": "our calib split (memory = 1/2 train normals, larger normal pool)",
        }.get(protocol, protocol)
        print(f"\nImage AUROC (%), clean images, {label}; mean ± std over seeds\n")
        print(format_markdown(summary, protocol))
    lines = sanity_check(summary)
    if lines:
        print("\nSanity check vs SPARC (arXiv:2608.18585, Table 1; different preprocessing):")
        print(*(f"  {x}" for x in lines), sep="\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
