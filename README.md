# Conformal False-Alarm Calibration for Industrial Anomaly Detection under Distribution Shift

[![CI](https://github.com/shayansalehi1381/anomaly-calibration/actions/workflows/ci.yml/badge.svg)](https://github.com/shayansalehi1381/anomaly-calibration/actions/workflows/ci.yml)
![tests](https://img.shields.io/badge/tests-34%20passed%20offline-brightgreen)
![python](https://img.shields.io/badge/python-3.10%2B-blue)
![license](https://img.shields.io/badge/license-MIT-lightgrey)

Industrial anomaly detectors are usually compared by AUROC, but a deployed detector needs
a **threshold**, and the number operators feel is the **false-alarm rate** on good
parts. This project calibrates that threshold with **split conformal prediction** on
normal images only, which gives a finite-sample, distribution-free guarantee
`E[false-alarm rate] ≤ α`. It then tests whether the guarantee **survives
distribution shift** (illumination, blur, noise, JPEG) for three detectors on
**VisA**: a feature-distance baseline, PatchCore and DINOv2-kNN.

> **TL;DR: status: Phase 0 of 6 (skeleton only). No experiments have been run yet,
> so this README contains no results.** Every number that appears here later will
> come from a real run, with mean ± std over seeds. This is a reproduction plus
> extension of existing work, not a novelty claim (see [Prior work](docs/PRIOR_WORK.md)).

---

## Contents
- [Research question](#research-question)
- [Motivation](#motivation)
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

## Method (planned)

1. **Detectors**, fit on a memory bank of normal training images per class: a global-feature
   kNN baseline, our own lean PatchCore (torch, no faiss) and DINOv2 patch-kNN.
2. **Split-conformal threshold** from held-out normal calibration images, strictly disjoint
   from the memory bank and from the test images.
3. **Clean evaluation** over R = 100 random calibration/test partitions: empirical FAR vs α,
   its distribution vs the theoretical Beta law, and detection rate at the threshold.
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

- **Synthetic shifts only.** The corruptions are controlled stand-ins for real
  acquisition changes. Real shifts (new product batches, a new camera) may behave
  differently.
- **Few normal test images per class in the official VisA split** (roughly 80; to be
  confirmed in Phase 1). We therefore enlarge the normal pool from the training split,
  which leaves a smaller memory bank than the standard protocol uses. AUROC is reported
  on both splits and labelled clearly.
- **Small-sample limits are real.** A non-trivial conformal threshold at α = 1% needs
  ≥ 99 calibration normals (≥ 299 for a 95%-confidence PAC guarantee). Recalibration with
  fewer shifted normals cannot certify α = 1%.
- **Closest prior work not yet read.** arXiv:2608.15090 and arXiv:2608.18585 could not be
  accessed from the development environment; see [docs/PRIOR_WORK.md](docs/PRIOR_WORK.md).

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

Only feature extraction and scoring run here (`scripts/extract_scores.py`, Phase 1–2).

```powershell
# 1. CUDA build of torch first (pick the CUDA version matching your driver)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
# 2. this package with the GPU extra
pip install -e ".[gpu,plots,dev]"
# 3. VisA stays local: extract VisA_20220922.tar to data\visa\ (git-ignored)
```

Extraction commands and the output schema will be documented in Phase 1
(`docs/score_schema.md`).

## Project layout

```
src/anomcal/          library (seeding, repo guard; data, corruptions, models,
                      metrics, calibration and shift modules added phase by phase)
scripts/              laptop-side entry points (extract_scores.py, Phase 1)
tests/                offline test suite (synthetic data only)
docs/                 prior work, interview notes, split protocol, score schema
results/              committed small score files and figures (from Phase 2)
data/                 VisA on the laptop only, git-ignored
```

## Roadmap

See [ROADMAP.md](ROADMAP.md). Phase 0 (skeleton, CI, prior work) is in review.

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
16. *Distribution-free false-alarm calibration and chance-corrected spatial evaluation for industrial anomaly detection.* arXiv:2608.15090, 2026 (not read; see Prior work).

## License

Code: MIT. VisA dataset: CC BY 4.0 (not redistributed here).
