# CLAUDE.md

Guidance for Claude (and any other contributor) working in this repository.

## Project in one paragraph

Research-style portfolio project: a **reproduction plus extension** of existing
industrial anomaly detection work on the **VisA** dataset (CC BY 4.0; 12 classes,
10,821 images: 9,621 normal, 1,200 anomalous; image- and pixel-level labels).
We benchmark a feature-distance baseline, PatchCore, and DINOv2-kNN, then add
split-conformal false-alarm calibration and a distribution-shift evaluation.
This is **not** a novelty claim; related prior work (e.g. arXiv 2608.15090) must
be cited. The reference for structure and quality is the author's earlier repo
`selective-rag-abstention` (offline test suite, CI, ROADMAP.md, honest README with
TL;DR and explicit limitations).

## Research question

Do conformally calibrated anomaly-score thresholds keep their target false-alarm
rate (alpha = 1% and 5%) under distribution shift (illumination change, blur,
Gaussian noise, JPEG compression), and which anomaly detection method keeps it
most stably?

Measured per method and per class:

- image-level AUROC; pixel-level AUROC or PRO where the method produces maps;
- empirical false-alarm rate (FAR) on held-out normal test images vs. target alpha;
- detection rate (TPR) at the calibrated threshold;
- FAR drift under each corruption and severity;
- how much recalibration on a small shifted normal set restores the guarantee.

## Hardware and workflow constraint (very important)

- The **cloud session has no GPU and no S3 access. Never try to download VisA there.**
- VisA lives only on the author's laptop at `data/visa/` (from `VisA_20220922.tar`).
  It is never committed.
- **Laptop (RTX 3060 Laptop, ~6 GB VRAM, 16 GB RAM, Windows)** runs only the heavy
  steps (feature extraction and anomaly scoring) via `scripts/extract_scores.py`,
  then pushes small score files.
- **Everything else** (calibration, shift analysis, metrics, plots, README) runs on
  **CPU from saved score files** (`.npy` / `.parquet`, small).
- `scripts/extract_scores.py` must be: resumable, with a progress bar, a
  `--max-images` flag for smoke tests, an explicit `--device` flag, memory-conscious
  batching for 6 GB VRAM, and a documented output schema (in `docs/`).
- Corruptions for the shift study are applied **on the laptop inside the extraction
  script**, so scores for every (corruption, severity) are stored. Corruption code is
  deterministic (seeded per image) and unit-tested.
- All code must be testable **offline with synthetic data in CI** (no dataset, no GPU,
  no network, no model downloads in tests).

## Working rules

1. **Phase by phase.** Each phase = code + tests + results or docs + clean commits +
   push. After each phase: stop, summarise, wait for the author's approval.
2. **No invented numbers.** Every reported number comes from a real run. Never
   estimate or fabricate results. If something has not been run, say so explicitly.
3. **No large files in git.** Never commit raw data, model checkpoints, feature
   caches or large files. Keep `.gitignore` correct. Only small score files and
   figures may be committed.
4. **No leakage.** Calibration data must be disjoint from test data and from
   training / memory-bank images. Document the split protocol in `docs/` and test
   for leakage.
5. **Reproducibility.** Fix all random seeds. Use at least 3 seeds or splits wherever
   there is randomness; report mean and standard deviation.
6. **Cite prior work** in the README references: PatchCore, PaDiM, DINOv2, VisA,
   conformal prediction and conformal anomaly detection papers, arXiv 2608.15090
   (only describe its contents if it was actually read).
7. **README.md stays current:** motivation, method, results table, limitations,
   reproducibility split into "runs in cloud on CPU" and "runs on laptop with GPU".
8. **docs/INTERVIEW_NOTES.md stays current:** key decisions, what failed and why,
   and the math behind conformal calibration.
9. **Explain non-trivial concepts** briefly while writing code; the author must
   understand every line, especially calibration and shift evaluation (these are
   written by us, never delegated to a library). anomalib may be used for models
   only if sufficient.
10. **Git hygiene:** small commits with conventional messages (`feat:`, `fix:`,
    `test:`, `docs:`, `chore:`, `ci:`), push after each phase, never rewrite history
    or force-push.
11. **Be honest** about failures and negative results; they go into the README.
12. At the end of the project, ask the author 15 interview-style questions about the
    project and give feedback on the answers.
