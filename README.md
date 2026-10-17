# Conformal False-Alarm Calibration for Industrial Anomaly Detection under Distribution Shift

[![CI](https://github.com/shayansalehi1381/anomaly-calibration/actions/workflows/ci.yml/badge.svg)](https://github.com/shayansalehi1381/anomaly-calibration/actions/workflows/ci.yml)
![tests](https://img.shields.io/badge/tests-206%20passed%20offline-brightgreen)
![python](https://img.shields.io/badge/python-3.10%2B-blue)
![license](https://img.shields.io/badge/license-MIT-lightgrey)

Industrial anomaly detectors are usually compared by AUROC, but a deployed detector needs
a **threshold**, and the number operators feel is the **false-alarm rate** on good
parts. This project calibrates that threshold with **split conformal prediction** on
normal images only, which gives a finite-sample, distribution-free guarantee
`E[false-alarm rate] ≤ α`. It then tests whether the guarantee **survives
distribution shift** (illumination, blur, noise, JPEG) for three detectors on
**VisA**: a feature-distance baseline, PatchCore and DINOv2-kNN.

> **TL;DR: status: Phases 0–1 of 0–5 built (Phase 1 awaiting the laptop smoke test).
> No experiments have been run yet, so this README contains no results.** Every number that appears here later will
> come from a real run, with mean ± std over seeds. This is a reproduction plus
> extension of existing work, not a novelty claim (see [Prior work](docs/PRIOR_WORK.md)).

---

## Contents
- [Research question](#research-question)
- [Motivation](#motivation)
- [Relation to prior work](#relation-to-prior-work)
- [Method (planned)](#method-planned)
- [Results](#results)
- [Limitations](#limitations)
- [Reproducibility](#reproducibility)
- [Project layout](#project-layout)
- [Roadmap](#roadmap)
- [References](#references)

## Research question

Do conformally calibrated anomaly-score thresholds keep their target false-alarm rate
(α = 1% and 5%) under distribution shift, and which detection method keeps it most
stably?

## Motivation

- **AUROC does not choose a threshold.** Two detectors with the same AUROC can have very
  different false-alarm rates at any given operating point.
- **Picking a threshold "by eye" on validation data has no guarantee.** Split conformal
  calibration does: if calibration and test normals are exchangeable, the expected
  false-alarm rate is at most α, for any detector and any score distribution
  ([math](docs/INTERVIEW_NOTES.md#2-split-conformal-threshold-the-math)).
- **Factories are not exchangeable.** Lighting drifts, cameras defocus, sensors get noisy,
  images get compressed. The guarantee rests on one assumption, and this assumption is
  exactly what breaks under shift. We measure by how much, per method and per class, and
  how many shifted normal images are needed to recalibrate.

## Relation to prior work

**Reproduction plus extension; no novelty claim.** Details: [docs/PRIOR_WORK.md](docs/PRIOR_WORK.md).

- **Deng (arXiv:2608.15090)** calibrates industrial AD thresholds with a one-sided
  order-statistic **tolerance limit**: the smallest rank r with
  Pr{Binomial(m, α) ≥ m − r + 1} ≥ 1 − δ, and τ = S₍ᵣ₎. This gives the PAC guarantee
  Pr_cal[P₀{S > τ} ≤ α] ≥ 1 − δ, valid with ties. Deng evaluates it on ISP-AD and Magnetic
  Tile with PatchCore, DINOv2 patch memory and a PaDiM-style detector, and reports Wilson
  intervals for the realised FPR. The paper states that the guarantee does not extend
  to domain shift and **does not test shift**. **Our PAC threshold is Deng's tolerance
  limit, credited as such**; we reproduce his primary setting (m = 150, α = 0.10,
  δ = 0.05 → r = 142, confidence 0.9693).
- **SPARC (Han, Chu, Nowak, Chin; arXiv:2608.18585)** corrects *features* under shift with
  a gradient-free per-cell subspace projection, built from ≤ 8 verified-normal images of
  the deployment lot. It reports image AUROC and AU-PRO only, with **no threshold
  calibration and no false-alarm guarantee**. It classifies **VisA as shift-free**. Its
  VisA image AUROC (PatchCore WRN-50 about 94.3, AnomalyDINO about 95.3; resize + center
  crop) serves **only as a sanity check** for our detectors.
- **What we add:** we measure how calibrated thresholds, split-conformal (marginal,
  E[FAR] ≤ α) and tolerance-limit (PAC), behave under **controlled synthetic shifts on
  VisA**, and how much **recalibration on a small set of shifted normals** restores the
  target false-alarm rate.

**How small is "small"? (derived by us from standard theory)** With n calibration
normals, split conformal can certify at best **α_min = 1/(n+1)**: n = 8 → 11.1%,
n = 99 → 1%. The tolerance limit needs **m ≥ m_min = ⌈log δ / log(1 − α)⌉** normals:
299 for α = 1%, δ = 5%. Derivations are in
[docs/INTERVIEW_NOTES.md](docs/INTERVIEW_NOTES.md#smallest-certifiable-α-for-a-given-number-of-calibration-normals-derived-by-us).

## Method (planned)

1. **Detectors**, fit on a memory bank of normal training images per class: a global-feature
   kNN baseline, our own lean PatchCore (torch, no faiss) and DINOv2 patch-kNN.
2. **Two thresholds** from held-out normal calibration images, strictly disjoint from the
   memory bank and from the test images: split conformal (marginal) and Deng's tolerance
   limit (PAC).
3. **Clean evaluation** over R = 100 random calibration/test partitions: empirical FAR vs α
   with **Wilson 95% intervals**, its distribution vs the theoretical Beta law, and
   detection rate at the threshold.
4. **Shift evaluation**: 4 corruptions × configurable severities, applied deterministically
   on the laptop. We measure FAR drift and detection rate.
5. **Mitigation**: recalibrate on m shifted normals and measure how much of the guarantee
   comes back.

Details and justification: [ROADMAP.md](ROADMAP.md).

## Results

*None yet.* The results tables (image AUROC on the official split and on our split,
pixel AUROC/PRO, FAR vs α, detection rate, FAR drift under shift) will be added in
Phases 2–5, each from a real run.

## Limitations

Known up front; the list will grow as results come in.

- **VisA contains no engineered shift, and our shifts are synthetic.** SPARC classifies
  VisA as shift-free. Our corruptions (brightness, blur, noise, JPEG) are controlled
  stand-ins for acquisition changes; real shifts (a new product lot, a new camera) may
  behave differently, and we make no claim about natural shift.
- **Shift breaks the assumption by construction.** Split conformal and the tolerance
  limit both rely on calibration and test normals being exchangeable (i.i.d. for the
  tolerance limit). A shifted test set violates this *by design*, so "the guarantee
  fails under shift" is expected. What we measure is *by how much*, for which methods,
  and what recalibration recovers.
- **Few normal test images per class in the official VisA split**: 50–101 per class
  (962 in total; verified from `split_csv/1cls.csv`, see
  [docs/SPLIT_PROTOCOL.md](docs/SPLIT_PROTOCOL.md)). We therefore enlarge the normal pool
  with half of the training normals, which leaves a memory bank half the standard size.
  AUROC is reported on both protocols and labelled clearly.
- **The PAC (tolerance-limit) guarantee at α = 1% is out of reach with our split.**
  n_cal ≈ 138–277 per class (at a 50/50 calibration/test partition), but it needs
  m_min = 299 at δ = 5%. α = 5% (m_min = 59) is reachable, and so is the marginal guarantee
  at α = 1% (n ≥ 99). This is known before running anything, and the results will say so.
- **Corruption strength is approximate.** We use ImageNet-C parameters (defined at 224 px)
  at our 256 px canonical resolution, and resize VisA's non-square images to squares.
- **Small-sample limits are real (derived by us).** Split conformal certifies at best
  α = 1/(n+1), and the tolerance limit needs m ≥ ⌈log δ / log(1 − α)⌉. With 8 shifted
  normals, recalibration can certify at best α ≈ 11.1% (split conformal) or about 31%
  (tolerance limit, δ = 5%), so it **cannot** restore a 1% guarantee.
- **Sanity-check numbers come from a different protocol.** SPARC's VisA AUROC uses resize +
  center crop; we resize the full image to a square at 256 px. Small gaps are expected and
  are not a claim; large gaps will be flagged and investigated.

## Reproducibility

The work is split by hardware. The dataset never leaves the laptop, and only small score
files and figures are committed (CI enforces this with `python -m anomcal.repo_guard`).

### Runs in the cloud / any machine on CPU

```bash
pip install -e ".[plots,dev]"
ruff check . && ruff format --check .
python -m pytest -q              # offline: no dataset, no GPU, no network
python -m anomcal.repo_guard     # no data / weights / large files tracked
```

Calibration, shift analysis, metrics and plots (Phases 3–5) will run from the committed
score files with the same CPU install.

### Runs on the laptop with a GPU (Windows, RTX 3060 6 GB)

Only feature extraction and scoring run here (`scripts/extract_scores.py`). The script is
resumable (rerun the same command after an interruption), prints a runtime estimate
before it starts, and has a `--max-images` smoke-test mode.
**Step-by-step Windows guide for the Phase 1 smoke test: [docs/LAPTOP_SMOKE_TEST.md](docs/LAPTOP_SMOKE_TEST.md).**

```powershell
# 1. CUDA build of torch first (pick the CUDA version matching your driver)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
# 2. this package with the GPU extra
pip install -e ".[gpu,plots,dev]"
# 3. VisA stays local: extract VisA_20220922.tar to data\visa\ (git-ignored)
```

```powershell
# verify the local VisA copy against the official per-class counts
python -m anomcal.data.visa --data-root data\visa
# smoke test (Phase 1: CPU dummy scorer; real methods arrive in Phase 2)
python scripts\extract_scores.py --data-root data\visa --device cpu `
    --methods dummy --classes candle --max-images 20
# print only the runtime estimate of a run
python scripts\extract_scores.py --data-root data\visa --device cpu --methods dummy --estimate-only
```

Output layout and columns: [docs/score_schema.md](docs/score_schema.md). Split protocol and
leakage checks: [docs/SPLIT_PROTOCOL.md](docs/SPLIT_PROTOCOL.md).

Try the full pipeline anywhere, without VisA, on a tiny synthetic stand-in:

```bash
python -m anomcal.data.fake_visa outputs/fake_visa
python scripts/extract_scores.py --data-root outputs/fake_visa --device cpu --seeds 0
```

## Project layout

```
src/anomcal/          library
  data/               VisA adapter, image loading, split protocol, fake VisA
  corruptions.py      deterministic ImageNet-C-style shifts
  scorers/            scorer interface (+ dummy; real methods in Phase 2)
  scores_io.py        score schema, atomic chunk files, consolidation
  extract.py          the laptop pipeline behind scripts/extract_scores.py
scripts/              laptop-side entry points (extract_scores.py)
configs/              example JSON configs for extract_scores.py --config
tests/                offline test suite (synthetic data only)
docs/                 prior work, interview notes, split protocol, score schema
results/              committed small score files and figures (from Phase 2)
data/                 VisA on the laptop only, git-ignored
```

## Roadmap

See [ROADMAP.md](ROADMAP.md): six phases (0–5). Phase 0 (skeleton, CI, prior work) is done; Phase 1 (data, splits, corruptions, extraction script) is in review.

## References

Full annotated list with the role of each paper: [docs/PRIOR_WORK.md](docs/PRIOR_WORK.md).

1. K. Roth et al. *Towards Total Recall in Industrial Anomaly Detection* (PatchCore). CVPR 2022.
2. T. Defard et al. *PaDiM: a Patch Distribution Modeling Framework for Anomaly Detection and Localization.* ICPR Workshops 2021.
3. Y. Zou et al. *SPot-the-Difference Self-Supervised Pre-training for Anomaly Detection and Segmentation* (VisA). ECCV 2022.
4. M. Oquab et al. *DINOv2: Learning Robust Visual Features without Supervision.* TMLR 2024.
5. S. Damm et al. *AnomalyDINO: Boosting Patch-based Few-shot Anomaly Detection with DINOv2.* WACV 2025.
6. L. Bergman, N. Cohen, Y. Hoshen. *Deep Nearest Neighbor Anomaly Detection.* arXiv:2002.10445, 2020.
7. P. Bergmann et al. *Uninformed Students: Student-Teacher Anomaly Detection with Discriminative Latent Embeddings* (PRO metric). CVPR 2020.
8. V. Vovk, A. Gammerman, G. Shafer. *Algorithmic Learning in a Random World.* Springer 2005.
9. V. Vovk. *Conditional Validity of Inductive Conformal Predictors.* ACML 2012.
10. A. N. Angelopoulos, S. Bates. *A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification.* arXiv:2107.07511.
11. R. Laxhammar, G. Falkman. *Inductive conformal anomaly detection for sequential detection of anomalous sub-trajectories.* Ann. Math. Artif. Intell., 2015.
12. S. Bates et al. *Testing for outliers with conformal p-values.* Annals of Statistics, 2023.
13. R. J. Tibshirani et al. *Conformal Prediction Under Covariate Shift.* NeurIPS 2019.
14. R. F. Barber et al. *Conformal prediction beyond exchangeability.* Annals of Statistics, 2023.
15. D. Hendrycks, T. Dietterich. *Benchmarking Neural Network Robustness to Common Corruptions and Perturbations.* ICLR 2019.
16. Deng. *Distribution-free false-alarm calibration and chance-corrected spatial evaluation for industrial anomaly detection.* arXiv:2608.15090, 2026.
17. Han, Chu, Nowak, Chin. *SPARC: Subspace Position-Aware Robust Few-Shot Calibration for Distribution-Shifted Industrial Anomaly Detection.* arXiv:2608.18585, 2026.
18. E. B. Wilson. *Probable inference, the law of succession, and statistical inference.* JASA 22(158), 1927.

## License

Code: MIT. VisA dataset: CC BY 4.0 (not redistributed here).
