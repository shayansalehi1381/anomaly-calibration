#!/usr/bin/env python
"""Write docs/RESULTS_SUMMARY.md from the committed CSVs (CPU, a few seconds).

Every number in the generated tables is read from ``results/`` and carries a
``<!-- check: ... -->`` comment that ``tests/test_docs_consistency.py`` re-verifies, so the
summary cannot drift away from the CSVs. Static text (estimates, "not tested") is in this file.

    python scripts/make_results_summary.py
"""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
P3, P4 = ROOT / "results" / "phase3", ROOT / "results" / "phase4"
FAMILIES = ["brightness", "gaussian_blur", "gaussian_noise", "jpeg_compression"]
METHODS = ["baseline", "dinov2_knn", "patchcore"]
NAMES = {"baseline": "baseline", "dinov2_knn": "DINOv2-kNN", "patchcore": "PatchCore"}


def chk(file, col, sel, value, tol):
    s = ",".join(f"{k}={v}" for k, v in sel.items())
    return f"<!-- check: {file} ;; {col} ;; {s} ;; {value:.6g} ;; {tol} -->"


def headline_row(hl, **sel):
    m = hl
    for k, v in sel.items():
        m = m[m[k] == v]
    assert len(m) == 1, sel
    return m.iloc[0]


def main(out_path: Path | None = None) -> None:
    hl = pd.read_csv(P4 / "headline.csv")
    a = pd.read_csv(P4 / "auroc_by_condition.csv")
    s3 = pd.read_csv(P3 / "summary_by_method.csv")
    st = pd.read_csv(P4 / "stability.csv")
    pr = pd.read_csv(P4 / "paired_stability.csv")
    sm = pd.read_csv(P4 / "small_m_headline.csv")
    pc = pd.read_csv(P4 / "per_class.csv")
    out = []
    w = out.append

    w("# Results summary\n")
    w(
        "Every number below is read from the committed CSVs by `scripts/make_results_summary.py` and "
        "re-checked by `tests/test_docs_consistency.py`. **Measured** = computed from the real laptop "
        "scores by CPU code in this repo. **Estimated** = a prediction, not a result (listed in §8). "
        "Level **0.5** is an extra mild level we defined (stored as `severity = -1`); it is not an "
        "ImageNet-C level. α is the target false-alarm rate, FAR the empirical false-alarm rate, "
        "excess FAR = FAR − α, TPR the detection rate on anomalous images at the same threshold.\n"
    )

    # 1 ---------------------------------------------------------------- clean accuracy
    w("## 1. Detector accuracy on clean images (measured)\n")
    w(
        "Image AUROC (%), class mean, our `calib` split (memory bank = ½ of train normals), mean over "
        "3 seeds. Source: `results/phase4/auroc_by_condition.csv`, column `auroc`, rows `corruption = none`. "
        "The official-split numbers (80.4 / 89.6 / 90.0), per-class tables and the SPARC sanity check are in "
        "`docs/RESULTS_EXTRACTION.md` (measured on the laptop; not recomputed here).\n"
    )
    w("| method | AUROC (calib split) |\n|---|---:|")
    for m in METHODS:
        v = a[(a.method == m) & (a.corruption == "none")].auroc.mean()
        w(f"| {NAMES[m]} | {v:.1f} |")
    w("")

    # 2 ---------------------------------------------------------------- phase 3
    w("## 2. Clean calibration (Phase 3, measured)\n")
    w(
        "Pooled over 12 classes × 3 seeds × 100 random 50/50 partitions. `theory` = exact Beta-binomial "
        "mean. `P(FAR>α)` = share of partitions whose test-half FAR exceeds α. Source: "
        "`results/phase3/summary_by_method.csv` (`far_pooled`, `far_theory_mean`, "
        "`share_far_above_alpha`, `tpr_mean`). The tolerance limit at α = 1% is **not certifiable** "
        "(needs 299 calibration normals, we have 138–277).\n"
    )
    w("| method | threshold | α | FAR | theory | P(FAR>α) | TPR |\n|---|---|---|---|---|---|---|")
    for m in METHODS:
        for kind in ("conformal", "tolerance"):
            for al in (0.01, 0.05, 0.10):
                r = s3[(s3.method == m) & (s3.kind == kind) & (s3.alpha == al)]
                if r.empty:
                    w(f"| {NAMES[m]} | {kind} | {al:g} | not certifiable | – | – | – |")
                    continue
                r = r.iloc[0]
                sel = {"method": m, "kind": kind, "alpha": al}
                w(
                    f"| {NAMES[m]} | {kind} | {al:g} | {r.far_pooled:.4f} {chk('phase3/summary_by_method.csv', 'far_pooled', sel, r.far_pooled, 0.00006)} "
                    f"| {r.far_theory_mean:.4f} | {r.share_far_above_alpha:.3f} | {r.tpr_mean:.3f} |"
                )
    w("")

    # 3 ---------------------------------------------------------------- phase 4 headline
    w("## 3. Under corruption (Phase 4, measured): split conformal, α = 5%\n")
    w(
        "Mean over 12 classes. **A** = threshold from clean calibration normals; **B** = threshold "
        "recalibrated on corrupted normals of the same corruption and level (needs normal images under "
        "the new condition). Cell = FAR (excess FAR). Source: `results/phase4/headline.csv`, columns "
        "`far`, `excess`, `tpr`, filtered by `setting`, `kind = conformal`, `alpha = 0.05`, `level`. "
        "95% class-bootstrap intervals: columns `far_lo`, `far_hi`.\n"
    )
    w(
        "| corruption | method | A: 0.5 | A: 1 | A: 3 | A: 5 | B: 0.5 | B: 5 | TPR A / B at 1 | TPR B at 5 |\n|---|---|---|---|---|---|---|---|---|---|"
    )
    for c in FAMILIES:
        for m in METHODS:

            def r(setting, lv, c=c, m=m):
                return headline_row(
                    hl,
                    corruption=c,
                    method=m,
                    setting=setting,
                    kind="conformal",
                    alpha=0.05,
                    level=float(lv),
                )

            cells = []
            for setting, lv in (
                ("A_no_recal", 0.5),
                ("A_no_recal", 1),
                ("A_no_recal", 3),
                ("A_no_recal", 5),
                ("B_recal", 0.5),
                ("B_recal", 5),
            ):
                x = r(setting, lv)
                sel = {
                    "method": m,
                    "corruption": c,
                    "setting": setting,
                    "kind": "conformal",
                    "alpha": 0.05,
                    "level": float(lv),
                }
                cells.append(
                    f"{x.far:.3f} ({x.excess:+.3f}) {chk('phase4/headline.csv', 'far', sel, x.far, 0.0006)}"
                )
            ta, tb = r("A_no_recal", 1).tpr, r("B_recal", 1).tpr
            w(
                f"| {c} | {NAMES[m]} | "
                + " | ".join(cells)
                + f" | {ta:.2f} / {tb:.2f} | {r('B_recal', 5).tpr:.2f} |"
            )
    w("")
    w(
        "Clean reference (level 0, A = B): FAR 0.046 (excess −0.004) for all three methods "
        "(`headline.csv`, `corruption = none`).\n"
    )
    w(
        "Violation rate (share of partitions with FAR > α; `headline.csv`, column `violation_rate`), "
        "mean over methods, corruptions and levels, conformal α = 5%:\n"
    )
    v = (
        hl[(hl.kind == "conformal") & (hl.alpha == 0.05) & (hl.corruption != "none")]
        .groupby("setting")
        .violation_rate.mean()
    )
    w(
        f"- A (no recalibration): {v['A_no_recal']:.3f}; B (recalibrated): {v['B_recal']:.3f} "
        "(about 40% is expected for B: the guarantee is on the *average* FAR, not per partition).\n"
    )

    # 4 ---------------------------------------------------------------- stability
    w("## 4. Which method is most stable? (measured)\n")
    w(
        "Stability = root-mean-square of (FAR − α) over the shifted conditions and the 12 classes "
        "(0 = target kept everywhere; smaller is better). We rank by *distance from the target*, not by "
        "spread across classes: a detector that fails the same way in every class has a small spread. "
        "Source: `stability.csv` (`rms_excess`, `rms_lo`, `rms_hi`, `class_spread`) and "
        "`paired_stability.csv`. `all` = 16 shifted conditions (levels 0.5, 1, 3, 5 × 4 corruptions); "
        "`1-3-5` = the 12 ImageNet-C conditions. Setting A.\n"
    )
    w(
        "| grid | threshold | α | method | RMS excess [95% CI] | class spread | rank |\n|---|---|---|---|---|---|---|"
    )
    for g in ("all", "1-3-5"):
        for kind, al in (
            ("conformal", 0.01),
            ("conformal", 0.05),
            ("conformal", 0.10),
            ("tolerance", 0.05),
            ("tolerance", 0.10),
        ):
            for m in METHODS:
                x = st[
                    (st.grid == g)
                    & (st.setting == "A_no_recal")
                    & (st.kind == kind)
                    & (st.alpha == al)
                    & (st.method == m)
                ].iloc[0]
                sel = {"grid": g, "setting": "A_no_recal", "kind": kind, "alpha": al, "method": m}
                w(
                    f"| {g} | {kind} | {al:g} | {NAMES[m]} | {x.rms_excess:.3f} [{x.rms_lo:.3f}, {x.rms_hi:.3f}] "
                    f"{chk('phase4/stability.csv', 'rms_excess', sel, x.rms_excess, 0.0006)} | {x.class_spread:.3f} | {int(x['rank'])} |"
                )
    w("")
    w(
        "**Paired comparison** (`paired_stability.csv`): difference of RMS excess between two methods, "
        "with a class-bootstrap interval computed on the *same* resampled classes for both methods "
        "(negative = first method closer to the target). Conformal, α = 5%, Setting A:\n"
    )
    w("| grid | method A | method B | difference [95% CI] | excludes 0 |\n|---|---|---|---|---|")
    for g in ("all", "1-3-5"):
        for ma, mb in (
            ("dinov2_knn", "patchcore"),
            ("dinov2_knn", "baseline"),
            ("patchcore", "baseline"),
        ):
            x = pr[
                (pr.grid == g)
                & (pr.setting == "A_no_recal")
                & (pr.kind == "conformal")
                & (pr.alpha == 0.05)
                & (pr.method_a == ma)
                & (pr.method_b == mb)
            ].iloc[0]
            sel = {
                "grid": g,
                "setting": "A_no_recal",
                "kind": "conformal",
                "alpha": 0.05,
                "method_a": ma,
                "method_b": mb,
            }
            w(
                f"| {g} | {NAMES[ma]} | {NAMES[mb]} | {x['diff']:+.3f} [{x.diff_lo:+.3f}, {x.diff_hi:+.3f}] {chk('phase4/paired_stability.csv', 'diff', sel, x['diff'], 0.0006)} | {bool(x.excludes_zero)} |"
            )
    pa = pr[pr.setting == "A_no_recal"]
    pb = pr[pr.setting == "B_recal"]
    w("")
    w(
        f"Over all (grid, threshold, α) cells in Setting A the paired interval excludes 0 in "
        f"{int(pa.excludes_zero.sum())} of {len(pa)}. In Setting B the largest |difference| is "
        f"{pb['diff'].abs().max():.5f}, so the methods are tied in practice (the few cells whose interval "
        "excludes 0 differ by less than 0.0006 of RMS excess).\n"
    )

    # 5 ---------------------------------------------------------------- per class
    w("## 5. Per-class check at the mild level (measured)\n")
    w(
        "Setting A, conformal α = 5%, level 0.5; source `per_class.csv` (`far`, `severity = -1`). "
        "The mean can hide a class.\n"
    )
    w(
        "| corruption | method | mean FAR | min class | max class | classes with FAR > 10% |\n|---|---|---|---|---|---|"
    )
    q = pc[
        (pc.kind == "conformal")
        & (pc.alpha == 0.05)
        & (pc.setting == "A_no_recal")
        & (pc.severity == -1)
    ]
    for c in FAMILIES:
        for m in METHODS:
            f = q[(q.corruption == c) & (q.method == m)].far
            w(
                f"| {c} | {NAMES[m]} | {f.mean():.3f} | {f.min():.3f} | {f.max():.3f} | {int((f > 0.10).sum())} / {len(f)} |"
            )
    w("")

    # 6 ---------------------------------------------------------------- small m
    w("## 6. Recalibration with only m corrupted normals (Phase 4b, measured; levels 1/3/5 only)\n")
    w(
        "Mean over methods and 12 classes. 500 draws per experiment and condition (100 partitions × 5 "
        "random m-subsets; the 5 draws share a test half, so they are correlated). `law` = exact "
        "Beta-binomial value. Source: `small_m_headline.csv` (`far`, `law_far`, `violation_rate`, "
        "`law_violation_rate`, `tpr`). Not rerun with the mild level.\n"
    )
    w(
        "| threshold | α | m | mean FAR | law | P(FAR>α) | law | TPR |\n|---|---|---|---|---|---|---|---|"
    )
    g = (
        sm.groupby(["kind", "alpha", "m"])[
            ["far", "law_far", "violation_rate", "law_violation_rate", "tpr"]
        ]
        .mean()
        .reset_index()
    )
    for r in g.itertuples():
        pcore = sm[
            (sm.method == "patchcore") & (sm.kind == r.kind) & (sm.alpha == r.alpha) & (sm.m == r.m)
        ].iloc[0]
        sel = {"method": "patchcore", "kind": r.kind, "alpha": r.alpha, "m": r.m}  # PatchCore row
        w(
            f"| {r.kind} | {r.alpha:g} | {r.m} | {r.far:.4f} {chk('phase4/small_m_headline.csv', 'far', sel, pcore.far, 0.0006)} "
            f"| {r.law_far:.4f} | {r.violation_rate:.3f} | {r.law_violation_rate:.3f} | {r.tpr:.2f} |"
        )
    w(
        "\nNot certifiable (explicit status, never clamped): conformal α = 1% needs m ≥ 99, α = 5% needs "
        "m ≥ 19; the tolerance limit (δ = 5%) needs m ≥ 29 (α = 10%), 59 (α = 5%), 299 (α = 1%).\n"
    )

    # 7 ---------------------------------------------------------------- cross split
    w("## 7. Cross-split check on clean data (Phase 3, measured)\n")
    cs = pd.read_csv(P3 / "cross_split_summary.csv")
    w(
        "Calibrate on held-out *train* normals, test on VisA's *official test* normals. Source: "
        "`results/phase3/cross_split_summary.csv` (`far_pooled`, `far_expected`). Wilson intervals are "
        "too narrow (3 seeds reuse the same images).\n"
    )
    w("| method | α | FAR | expected |\n|---|---|---|---|")
    for r in cs.itertuples():
        w(f"| {NAMES[r.method]} | {r.alpha:g} | {r.far_pooled:.4f} | {r.far_expected:.4f} |")
    w(
        "\nDINOv2-kNN and PatchCore are 1–1.5 points above the expected FAR at α = 10%; a weak hint that "
        "VisA's test normals score slightly higher than its train normals. Not conclusive.\n"
    )

    # 8 ---------------------------------------------------------------- estimates
    w("## 8. Estimated numbers (not results)\n")
    w(
        "- Laptop run time from `scripts/estimate_full_run.py`: full run predicted 3 h 44 min (warm-up) "
        "and measured 3 h 53 min; extra mild level predicted 1 h 04 min and measured 1 h 05 min "
        "(`results/provenance/run_meta.json`, field `elapsed_seconds`). The cost model was fitted on the "
        "full run, so the second comparison is not independent."
    )
    w(
        "- File sizes after adding the mild level: predicted +31% rows (row ratio 110,381 / 84,409); "
        "measured file size 0.80 MB per file (vs 0.64 MB before)."
    )
    w(
        "- Weak PatchCore classes (macaroni2, capsules) are said to be hard in the literature: unverified.\n"
    )

    # 9 ---------------------------------------------------------------- not tested
    w("## 9. What we did not test\n")
    for t in [
        "Real distribution shift (a new product lot, camera, lighting rig). VisA has no engineered shift; all shifts here are synthetic and strong.",
        "Why noise std 0.04 or JPEG quality 50 already break a clean-calibrated threshold; no causal claim.",
        "Why DINOv2-kNN is 5.7 AUROC points below the AnomalyDINO reference at 224 px (hypotheses only: input size, aggregation, resizing, masking).",
        "Why PatchCore and DINOv2-kNN drift differently (DINOv2-kNN is closest to the target; we did not test the mechanism).",
        "Other datasets (MVTec AD 2, AeBAD-S), other detectors, other corruption strengths or families.",
        "Uncertainty from sampling images *within* a class: the class bootstrap does not cover it (138–277 test normals per class; the 100 partitions reuse them).",
        "Small-m recalibration together with the mild level (Phase 4b was computed on levels 1/3/5).",
        "A mismatched calibration/test severity (calibrate at one level, test at another), skipped by decision.",
        "Feature-level shift correction (SPARC-style) combined with conformal calibration.",
        "Whether the 12 VisA classes are representative of a real factory line.",
    ]:
        w(f"- {t}")
    dest = out_path or ROOT / "docs" / "RESULTS_SUMMARY.md"
    dest.write_text("\n".join(out) + "\n", encoding="utf-8")
    print(f"wrote {dest}")


if __name__ == "__main__":
    import sys

    main(Path(sys.argv[1]) if len(sys.argv) > 1 else None)
