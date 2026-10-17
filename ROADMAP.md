# ROADMAP

Phase-by-phase plan. Each phase ends with code + tests + docs/results, a push, a PR
from `claude/festive-cerf-e50k0a` to `main` (merged with a merge commit, not a squash, so
the small commits stay visible), and a stop for the author's approval (see `CLAUDE.md`). Status markers: ☐ not started,
◐ in progress, ☑ done.

Effort estimates are rough planning guesses, **not measurements**. Laptop compute times
are measured for real in the Phase 2 smoke test, and this file is updated with the
measured numbers.

---

## Changes to the original plan, and why

1. **Corruptions move from Phase 4 to Phase 1.** Corrupted images are scored on the
   laptop by `extract_scores.py`. If the corruption code only existed in Phase 4, the
   laptop would have to run the whole pipeline a second time. With corruption code
   written and tested in Phase 1, a **single laptop run** at the end of Phase 2 produces
   every score the rest of the project needs (clean + every corruption × severity, for
   every method and seed). Phase 4 then becomes pure CPU analysis.

2. **"Score everything that is not in the memory bank."** The laptop scores every
   held-out normal image and every anomalous image, not one fixed calibration/test
   partition. The calibration/test partition is then drawn on CPU, many times, with
   fixed seeds. Two reasons:
   - The conformal guarantee is a statement *averaged over random calibration sets*,
     so resampling the partition (e.g. 100 random splits) is how we measure it
     properly, and it costs nothing on CPU.
   - Changing the split protocol later needs no new laptop run.

   Randomness on the laptop side (which training normals go into the memory bank,
   coreset sampling) is covered by **3 seeds** of the laptop run.

3. **Pixel-level metrics are computed on the laptop.** Anomaly maps are too big to
   commit (roughly 2,000 test images × 256×256 floats per method and seed). The
   metric code (pixel AUROC, PRO) lives in the package, is tested on CPU with synthetic
   maps, and the laptop script writes only the resulting small JSON. Pixel metrics are
   reported for clean images only, and only for methods that produce maps (PatchCore,
   DINOv2-kNN; not the global-feature baseline).

4. **A separate final phase for interview prep** (working rule 12), so it does not get
   squeezed into the write-up phase.

---

## Key design decisions to settle early (proposed defaults)

| Topic | Proposed default | Decided in |
|---|---|---|
| Official split | VisA `split_csv/1cls.csv` (train is normal only; test has normal + anomalous). Counts are checked against the file in Phase 1, not assumed. | Phase 1 |
| Our split per class | Official *train* normals → randomly split into **memory bank** (fit) and **held-out normal pool**. Official *test* normals join the held-out pool. Calibration/test partitions are drawn from this pool on CPU. Anomalies are used only for evaluation (AUROC, detection rate). | Phase 1 |
| Why we enlarge the normal pool | The official test split has only roughly 80 normal images per class. At alpha = 1% that means about 0.8 expected false alarms, which is far too noisy to measure an FAR. | Phase 1 |
| Comparability | Image AUROC is also reported on the official test split, so it can be compared with the literature. The cost of a smaller memory bank is documented honestly. | Phase 2 |
| Calibration scope | Per class (one detector per class, as is standard for VisA). | Phase 3 |
| Guarantee type | Primary: marginal, `E[FAR] ≤ α`. Added: the PAC / training-conditional variant `P(FAR ≤ α) ≥ 1 − δ` (Vovk, 2012), which needs n ≥ ln δ / ln(1 − α), i.e. 299 normals for α = 1%, δ = 5%. It is cheap to add and makes the small-sample story explicit. | Phase 3 |
| Conformal threshold | `t = k`-th smallest calibration score with `k = ⌈(n+1)(1−α)⌉`. If `k > n`, the threshold is +∞ (never alarm). Consequence: alpha = 1% needs n ≥ 99 calibration normals. | Phase 3 |
| Corruptions | Brightness/gamma (illumination), Gaussian blur, Gaussian noise, JPEG, following the ImageNet-C convention (Hendrycks & Dietterich, 2019). Applied after resizing to the model input size, seeded per (image id, corruption, severity). **3 severities by default; the number of severities is a config value and a CLI flag**, so the other 2 ImageNet-C levels can be added later without code changes. The exact parameters of every severity are written into the output metadata. | Phase 1 |
| Models | Baseline: ImageNet WideResNet-50 global pooled features, kNN distance to normal set. PatchCore: **our own lean implementation, agreed** (layer2+3 features, local aggregation, greedy coreset in torch with a seeded random projection and a configurable memory-bank size, kNN via chunked `torch.cdist` / matmul distances on GPU with a CPU fallback for tests, **no faiss**). anomalib is at most an optional cross-check, never a core dependency. DINOv2: ViT-S/14 or ViT-B/14 patch tokens + kNN, fp16. | Phase 2 |
| Score file format | One Parquet file per (method, seed), in long format: `image_id, class, label, split_role, corruption, severity, score`. Plus `run_meta.json` (versions, seeds, timings, git hash). Full schema in `docs/score_schema.md`. | Phase 1 |

---

## Phase 0: Skeleton, CI and prior work ◐ (in review)

**Goal:** an empty but professional repository that CI keeps green from day one.

Deliverables
- `pyproject.toml` with optional dependency groups: `[cpu]` (numpy, pandas, pyarrow,
  scipy, scikit-learn, matplotlib), `[gpu]` (torch, torchvision, timm, tqdm) and
  `[dev]` (pytest, ruff, mypy).
- Package layout `src/anomcal/` (data, corruptions, models, scoring, calibration,
  shift, metrics, plots).
- `tests/` with a smoke test, GitHub Actions CI (ruff + pytest on CPU; Linux 3.10/3.12/3.13
  and Windows 3.12; no network or model downloads), a repo guard that fails CI if data,
  weights or large files are tracked, and `.gitignore` hardened (`data/`, `*.tar`,
  `*.pt`, `*.pth`, `*.ckpt`, feature caches, `outputs/` except whitelisted small files).
- `docs/PRIOR_WORK.md`: PatchCore (Roth et al., 2022), PaDiM (Defard et al., 2021),
  VisA/SPD (Zou et al., 2022), DINOv2 (Oquab et al., 2023), conformal prediction
  (Vovk et al.; Angelopoulos & Bates), conformal anomaly detection (Laxhammar &
  Falkman; Bates et al., 2023, "Testing for outliers with conformal p-values"),
  conformal prediction under shift (Tibshirani et al., 2019; Barber et al., 2023),
  ImageNet-C. Also arXiv 2608.15090: I will try to read it. If I cannot access it,
  I will say so and cite it without describing its contents.
- Skeleton for README (TL;DR, status, limitations placeholder) and
  `docs/INTERVIEW_NOTES.md`.
- Mirror the structure of `selective-rag-abstention` (read-only): CI layout incl. a
  Windows job, tests layout, README with TL;DR and honest limitations, ROADMAP style.

Done so far: everything above. arXiv:2608.15090 and arXiv:2608.18585 could **not** be read,
because arxiv.org is blocked by the cloud network policy; both are cited by title only.
Tests: package imports, stable per-item seeds (across processes), repo guard rules,
`.gitignore` coverage. 34 offline tests.
**Effort (rough guess):** ~1 session. **You do locally:** nothing (optionally `pip install -e .[cpu,dev]`
and run `pytest` to check that it works on Windows).

---

## Phase 1: Data adapter, split protocol, corruptions, extraction-script skeleton ☐

**Goal:** everything the laptop run needs, except the models, written and tested
offline.

Deliverables
- `anomcal.data.visa`: parse `split_csv/1cls.csv`, image and mask loading, a
  per-class index with stable `image_id`s.
- `anomcal.data.splits`: seeded split into memory-bank and held-out pool, plus a CPU
  sampler for calibration/test partitions. `docs/SPLIT_PROTOCOL.md` documents it, and
  says which reported numbers use the official split and which use ours. Memory bank
  and held-out pool are **strictly disjoint**, enforced by a unit test that fails on
  any overlap.
- `anomcal.corruptions`: 4 corruptions × N severities (N = 3 by default, set by config
  and the `--severities` flag), deterministic, pure numpy/PIL. Severity parameters
  are stored in `run_meta.json`.
- `scripts/extract_scores.py`: CLI (`--data-root` (required, no default path),
  `--classes`, `--methods`, `--seeds`, `--severities`, `--device`, `--batch-size`,
  `--num-workers` (default 0, which is safe on Windows), `--max-images`, `--out`).
  Resumable after interruption (one finished (class, method, seed, corruption) chunk =
  one file, written atomically and skipped if already present). tqdm progress bar.
  **Prints an estimated runtime before the full run** (from a short timed warm-up).
  Writes the documented schema. `pathlib` everywhere, no hard-coded POSIX paths. A **dummy scorer** stands in for
  the models so the whole pipeline is tested end to end.
- A tiny **fake VisA** generator (same folder and CSV layout, synthetic images and
  masks) for tests and CI.
- `scripts/preprocess_visa.py` (laptop only, optional): caches images resized to the
  model input size, because decoding ~1.5k×1k JPEGs is the likely CPU bottleneck.
  The cache is git-ignored.
- `docs/score_schema.md`.

Tests: split disjointness (memory bank ∩ held-out pool = ∅ and calibration ∩ test = ∅,
on `image_id` and on resolved file path), seed
determinism, corruption determinism and severity monotonicity (for example, PSNR
decreases as severity increases), identity at severity 0, script resumability
(interrupt, then rerun produces identical output), schema validation, and the
`--max-images` behaviour.
**Effort (rough guess):** ~2 sessions. **You do locally:** untar VisA to `data/visa/`, run
`python scripts/extract_scores.py --methods dummy --max-images 20` as a smoke test of
data loading on the real files, and report any errors (no results are pushed yet).

---

## Phase 2: Models + metrics, then the single laptop run ☐

**Goal:** real scores for 3 methods × 3 seeds × (clean + 12 shifted conditions).

Deliverables
- `anomcal.models`: baseline (global features + kNN), PatchCore (torch greedy coreset
  with seeded random projection, configurable memory-bank size, chunked
  `torch.cdist`/matmul kNN on GPU with CPU fallback, anomaly maps), DINOv2-kNN (patch tokens + kNN, anomaly maps). Each has a CPU unit
  test with a tiny random backbone, so CI never downloads weights.
- `anomcal.metrics`: image AUROC, pixel AUROC, PRO (integrated up to FPR 0.3), FAR,
  TPR at threshold. Tested against hand-computed cases and `sklearn`.
- Memory plan for 6 GB VRAM: fp16 backbones, batch size auto-reduced on OOM, coreset
  computed in chunks, kNN via chunked `torch.cdist`.
- Laptop run in three stages: (a) `--max-images 20` smoke test to measure throughput.
  **The rough laptop-time guesses in this file are then replaced with the measured
  numbers.** (b) One full class. (c) All classes.
- Commit `results/scores/*.parquet` (expected to be a few MB in total) plus
  `run_meta.json`.
- `docs/PATCHCORE_DESIGN.md` (and a section per method): backbone, layers, patch
  aggregation, coreset ratio / memory-bank size, image size, and why each was chosen.
- Sanity check **in the README**: our per-class image AUROC on the official split next
  to published PatchCore numbers on VisA where available (cited from primary sources).
  Large discrepancies are flagged honestly, not hidden, and investigated before moving
  on. Optional: an anomalib PatchCore run as a separate cross-check.

**Effort (rough guess):** ~2–3 sessions + laptop compute (several hours in total,
splittable over nights thanks to resumability; measured in stage (a)).
**You do locally:** run stages (a)–(c), push the score files, and report the
throughput and any OOMs.

---

## Phase 3: Split-conformal calibration and empirical FAR (CPU) ☐

**Goal:** answer "does the guarantee hold on clean data?" before asking about shift.

Deliverables
- `anomcal.calibration`: conformal threshold, conformal p-values, and the exact
  finite-sample FAR guarantee `α − 1/(n+1) ≤ E[FAR] ≤ α` (the lower bound holds
  without ties). Hand-written and explained line by line.
- An evaluation over R = 100 random calibration/test partitions per (class, method,
  seed). Reported **per class**, not only on average: mean and std of FAR vs alpha,
  the **histogram of per-calibration-set FAR plotted next to the theoretical
  Beta(n+1−k, k) law** (Beta-binomial for a finite test set), and TPR at the threshold.
- The PAC variant (`P(FAR ≤ α) ≥ 1 − δ`) next to the marginal one. A comparison
  with a naive threshold (the empirical (1−α) quantile without the +1 correction)
  shows why the correction matters.
- Results table and figures in `results/`, README section, and the math in
  `docs/INTERVIEW_NOTES.md`.

Tests (in CI): on a synthetic score distribution with exchangeable scores, the
guarantee must hold. The average FAR lies within [α − 1/(n+1), α] up to a fixed-seed
Monte-Carlo tolerance, and the FAR histogram matches the Beta law (KS test); `k` is
computed with exact rational arithmetic; edge cases (n < 1/α − 1,
ties, alpha = 0/1 rejected); no calibration image appears in the test set.
**Effort (rough guess):** ~1–2 sessions. **You do locally:** nothing.

---

## Phase 4: Distribution-shift evaluation and recalibration (CPU) ☐

**Goal:** the core research question.

Deliverables
- FAR drift: calibrate on clean normals, evaluate on corrupted test normals, per
  (method, class, corruption, severity). Also report TPR on corrupted anomalies,
  so the trade-off between "fewer false alarms" and "detects nothing" is visible.
- A stability summary per method (for example, mean |FAR − α| across shifts and the
  worst case), with seed std.
- Mitigation: recalibrate on m shifted normal images (m ∈ {10, 25, 50, 100}, disjoint
  from the shifted test set), and measure how much of the guarantee is restored.
  Honest note up front: with m < 99, alpha = 1% cannot be certified at all (the
  threshold becomes +∞). That is a result to report, not to hide.
- Optional (only if time permits): a simple score-shift detector (for example, a
  conformal test on a batch of normal scores) to *flag* when recalibration is needed.

Tests: synthetic location-shift scenario where the expected FAR drift is known
analytically; recalibration restores FAR on synthetic shifted data; sets are disjoint.
**Effort (rough guess):** ~2 sessions. **You do locally:** nothing (unless we decide to add
severities or corruptions, which would mean a resumable re-run of extraction only
for the new conditions).

---

## Phase 5: Error analysis, figures, final README ☐

Deliverables
- Per-class error analysis: which classes and corruptions break calibration, and
  whether that correlates with clean AUROC or score variance. Look at the worst
  false alarms (you run a tiny helper locally to render thumbnails, because images
  cannot be committed under our size rules).
- Final figures: FAR vs severity, a per-class heatmap, recalibration curves.
- Final README (TL;DR, results tables with mean ± std, limitations, reproducibility
  split into cloud/CPU and laptop/GPU) and a complete `docs/INTERVIEW_NOTES.md`.
- Optional: PaDiM, if time permits (this would need another laptop run).

**Effort (rough guess):** ~2 sessions. **You do locally:** optionally render thumbnails of failure
cases and review the figures.

---

## Phase 6: Interview check ☐

15 interview-style questions on the project (method, math, design decisions, failures),
and feedback on your answers. **Effort:** 1 session.

---

## Risks and how we handle them

| Risk | Mitigation |
|---|---|
| 6 GB VRAM OOM (PatchCore memory bank, DINOv2-B) | fp16, chunked kNN, coreset ratio adjustable, ViT-S fallback, automatic batch-size backoff |
| Laptop run too slow | Resized-image cache, resumability, `--classes` to split the work over several nights |
| Too few normals for alpha = 1% | Enlarged held-out pool (see design decisions); report per-class n and the resulting resolution of the FAR estimate |
| Windows-specific issues (paths, DataLoader workers, faiss) | `pathlib` everywhere, `num_workers` flag, no faiss |
| Score files grow too large | Long-format Parquet with float32, size checked in CI |
| Results look worse than the literature | Documented honestly; sanity checks in Phase 2 before building on them |
