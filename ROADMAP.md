# ROADMAP

Phase-by-phase plan. Each phase ends with code + tests + docs/results, a push, a PR
from `claude/festive-cerf-e50k0a` to `main` (merged with a merge commit, not a squash, so
the small commits stay visible), and a stop for the author's approval (see `CLAUDE.md`). Status markers: ☐ not started,
◐ in progress, ☑ done.

Effort estimates are rough planning guesses, **not measurements**. Laptop compute times
are replaced with measurements as they come in (see "Laptop compute: measured so far"
under Phase 2).

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

---

## Key design decisions to settle early (proposed defaults)

| Topic | Proposed default | Decided in |
|---|---|---|
| Official split | VisA `split_csv/1cls.csv`. **Verified in Phase 1** from the public spot-diff copy: 8,659 train normals, 962 test normals (50–101 per class), 1,200 anomalies (100 per class). See `docs/SPLIT_PROTOCOL.md`. | Phase 1 ✔ |
| Our split per class | Official *train* normals → randomly split into **memory bank** (fit) and **held-out normal pool**, `memory_fraction = 0.5` by default. Official *test* normals join the held-out pool. Calibration/test partitions are drawn from this pool on CPU. Anomalies are used only for evaluation (AUROC, detection rate). This gives n_cal ≈ 138–277 per class at a 50/50 partition: enough for the marginal guarantee at α = 1% (≥ 99), **not** for the PAC one (≥ 299). | Phase 1 ✔ |
| Why we enlarge the normal pool | The official test split has only 50–101 normal images per class. At alpha = 1% that means 0.5–1 expected false alarms, which is far too noisy to measure an FAR. | Phase 1 ✔ |
| Comparability | Image AUROC is also reported on the official test split, so it can be compared with the literature. The cost of a smaller memory bank is documented honestly. | Phase 2 |
| Calibration scope | Per class (one detector per class, as is standard for VisA). | Phase 3 |
| Guarantee type | Primary: marginal, `E[FAR] ≤ α` (split conformal). Added: the PAC / training-conditional variant `P(FAR ≤ α) ≥ 1 − δ`, implemented as **Deng's tolerance limit** (arXiv:2608.15090; theory: Vovk, 2012), credited explicitly. It needs m ≥ ⌈log δ / log(1 − α)⌉, i.e. 299 normals for α = 1%, δ = 5%. Realised FAR is reported with **Wilson 95% intervals**. | Phase 3 |
| Conformal threshold | `t = k`-th smallest calibration score with `k = ⌈(n+1)(1−α)⌉`. If `k > n`, the threshold is +∞ (never alarm). Consequence: alpha = 1% needs n ≥ 99 calibration normals. | Phase 3 |
| Corruptions | Brightness/gamma (illumination), Gaussian blur, Gaussian noise, JPEG, following the ImageNet-C convention (Hendrycks & Dietterich, 2019). Applied after resizing to one **canonical 256 px** resolution shared by all methods (so σ in pixels means the same for every detector), seeded per (image id, corruption, severity). **3 severities by default; the number of severities is a config value and a CLI flag**, so the other 2 ImageNet-C levels can be added later without code changes. The exact parameters of every severity are written into the output metadata. | Phase 1 |
| Models | Baseline: ImageNet WideResNet-50 global pooled features, kNN distance to normal set. PatchCore: **our own lean implementation, agreed** (layer2+3 features, local aggregation, greedy coreset in torch with a seeded random projection and a configurable memory-bank size, kNN via chunked `torch.cdist` / matmul distances on GPU with a CPU fallback for tests, **no faiss**). anomalib is at most an optional cross-check, never a core dependency. DINOv2: ViT-S/14 or ViT-B/14 patch tokens + kNN, fp16. | Phase 2 |
| Score file format | Long-format Parquet, one consolidated file per (protocol, method, seed): `image_id, class, label, role, official_split, protocol, method, seed, corruption, severity, score`. Plus `run_meta.json` (config, exact severity parameters, versions, git hash, timings). Full schema in `docs/score_schema.md`. | Phase 1 ✔ |

---

## Phase 0: Skeleton, CI and prior work ☑

**Goal:** an empty but professional repository that CI keeps green from day one.

Deliverables
- `pyproject.toml`: core dependencies are the CPU stack (numpy, pandas, pyarrow, scipy,
  scikit-learn, pillow, tqdm); optional groups `[plots]` (matplotlib), `[gpu]` (torch,
  torchvision, timm; laptop only) and `[dev]` (pytest, ruff).
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

Done so far: everything above. arXiv:2608.15090 and arXiv:2608.18585 could **not** be read
from the cloud (arxiv.org is blocked by the network policy). The author has since read
both in full, and the verified facts are in `docs/PRIOR_WORK.md` §1.
Tests: package imports, stable per-item seeds (across processes), repo guard rules,
`.gitignore` coverage. 34 offline tests.
**Effort (rough guess):** ~1 session. **You do locally:** nothing (optionally `pip install -e .[plots,dev]`
and run `pytest` to check that it works on Windows).

---

## Phase 1: Data adapter, split protocol, corruptions, extraction-script skeleton ☑ (laptop smoke test passed)

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
- ~~`scripts/preprocess_visa.py` (resized-image cache)~~: **deferred.** Each class is now
  decoded once per (method, seed) and reused for all 13 conditions, so the cache may not
  be needed. We decide after the smoke test measures the per-image load time on the laptop.
- `docs/score_schema.md`, `docs/SPLIT_PROTOCOL.md`, `docs/LAPTOP_SMOKE_TEST.md`.

Done (in review): everything above. Also added: an `official` protocol in the same
script (memory = all train normals, clean official test set) for literature-comparable
AUROC; a `fingerprint.json` that refuses to resume when result-affecting settings
change; a saved split manifest per class and seed (leakage audit trail);
`--config` JSON files; `--severities N` / `--severity-levels`; `--estimate-only`;
`python -m anomcal.data.visa` to verify a local VisA copy.

Tests: split disjointness (memory bank ∩ held-out pool = ∅ and calibration ∩ test = ∅,
on `image_id` and on resolved file path), seed
determinism, corruption determinism and severity monotonicity (for example, PSNR
decreases as severity increases), identity at severity 0, script resumability
(interrupt, then rerun produces identical output), schema validation, and the
`--max-images` behaviour.
**Effort (rough guess):** ~2 sessions. **You do locally:** follow
`docs/LAPTOP_SMOKE_TEST.md` (clone, venv, untar VisA to `data\visa\`, verify counts, run
the dummy smoke test on `candle` with `--max-images 20`) and send back the console
output and timings (no results are pushed).

---

## Phase 2: Models + metrics, then the single laptop run ☑ (scores in results/scores/)

**Goal:** real scores for 3 methods × 3 seeds × (clean + 12 shifted conditions).

Deliverables
- `anomcal.models`: baseline (global features + kNN), PatchCore (torch greedy coreset
  with seeded random projection, configurable memory-bank size, exact re-ranked kNN on
  GPU with a CPU path for tests, anomaly maps), DINOv2-kNN (patch tokens + cosine kNN,
  anomaly maps). Each is tested on CPU with a tiny seeded random backbone, so CI never
  downloads weights; the real architectures are built with random weights to check shapes.
- `anomcal.metrics`: image AUROC, pixel AUROC, PRO (integrated up to FPR 0.3), FAR,
  TPR at threshold. Tested against hand-computed cases and `sklearn`.
- Numerical care in kNN distances: the |a|² − 2ab + |b|² trick cancels badly in
  float32 (seen in Phase 1: a memory image scored 0.71 against itself instead of 0).
  **Done:** float32 shortlist + exact re-rank; a memory patch scores exactly 0 (tested).
- Memory plan for 6 GB VRAM: fp16 autocast for backbones, batch halved on OOM,
  distance blocks capped at 2²⁶ floats, coreset projection chunked, PatchCore patches
  in a preallocated CPU buffer. Analytic budget in `docs/MODELS.md` (≈ 2 GB VRAM).
- Laptop run in three stages (`docs/LAPTOP_GPU_RUN.md`): (a) CUDA install, warm-up
  estimate and GPU smoke test (`--max-images 20`, class cashew); (b) one full-size class
  (cashew, seed 0) + `scripts/estimate_full_run.py` → choose reductions; (c) all classes.
- Commit `results/scores/*.parquet` (expected to be a few MB in total) plus
  `run_meta.json`.
- `docs/MODELS.md`: backbone, layers, patch aggregation, coreset ratio / memory-bank
  size, image size, and why each was chosen, per method. **Done.**
- Sanity check **in the README**: our image AUROC on the official split, next to
  SPARC's VisA numbers (arXiv:2608.18585, Table 1, resize + center crop: PatchCore
  WRN-50 about 94.3, AnomalyDINO about 95.3), compared with our PatchCore and DINOv2-kNN.
  These are a **sanity check only, not a claim**, because the preprocessing differs.
  Large discrepancies are flagged honestly, not hidden, and investigated before moving
  on. Optional: an anomalib PatchCore run as a separate cross-check.

Done (code, in review): metrics (image/pixel AUROC, PRO), the three scorers, method
options (`--opt`), pixel metrics on the laptop, corruptions shared between methods,
per-job timings and peak VRAM in `run_meta.json`, `scripts/estimate_full_run.py`
(cost model fitted to measured jobs), `scripts/auroc_table.py` (official-split AUROC
next to the SPARC reference numbers), a CI job with CPU torch on Linux and Windows.

### Laptop compute: measured so far

Measured in the Phase 1 smoke test (i7-12650H, NVMe SSD, CPU only), per 256 px image:
load 16.5 ms; corruption at the strongest level: brightness 3.6 ms, blur 4.6 ms,
noise 2.9 ms, JPEG 1.9 ms. The smoke test's own estimate was 10.2 s, and it took 9.98 s.

**CPU part of the full run, derived from those measurements** (arithmetic, not a run):

| item | workload | time |
|---|---|---|
| decode every image once | 10,821 images × 16.5 ms | ≈ 3.0 min |
| corruptions, shared by the 3 methods | 3 seeds × 6,493 eval images × 12 conditions, at the strongest-level cost (upper bound) | ≤ 12.7 min (it would be ≈ 38 min without sharing) |

**Measured (laptop, full run):** 3 h 53 min wall clock for all 1,470 remaining chunks
(764,064 image-conditions); `estimate_full_run.py` had predicted 3 h 44 min from the
one-class measurement (cashew, 3 min 04 s), i.e. about 4% low. Measured peak VRAM on
cashew: baseline 362 MB, PatchCore 1,222 MB, DINOv2-kNN 1,035 MB (upper bounds for the
largest bank: 1,456 / 2,458 / 2,082 MB). No reductions were needed. Results:
`docs/RESULTS_EXTRACTION.md`.

**Before the run (kept for the record): GPU part not measured yet.** Workload per method: 3 × 84,409 calib + 3 × 2,162
official image-conditions ≈ 260k scored images, plus 38,961 memory images to embed.
My FLOP-based guess (an assumption, not a measurement) is ~2–3 h for all three methods
together, dominated by PatchCore's kNN and coreset. Stage (b) replaces this with a
prediction fitted to real timings; this table will then be updated.

Peak VRAM: analytic budget ≈ 2 GB with the defaults (`docs/MODELS.md` §4); the
measured peak per method comes from stage (a)/(b).

**Effort (rough guess):** code done in ~2 sessions; laptop compute see above.
**You do locally:** stages (a)–(c) in `docs/LAPTOP_GPU_RUN.md`, then push the score
files (stage (c) lists what to copy into `results/scores/`).

---

## Phase 3: Split-conformal calibration and empirical FAR (CPU) ☑ (results in results/phase3/, README §Results)

**Goal:** answer "does the guarantee hold on clean data?" before asking about shift.

Deliverables
- `anomcal.calibration`: conformal threshold, conformal p-values, and the exact
  finite-sample FAR guarantee `α − 1/(n+1) ≤ E[FAR] ≤ α` (the lower bound holds
  without ties). Hand-written and explained line by line.
- An evaluation over R = 100 random calibration/test partitions per (class, method,
  seed). Reported **per class**, not only on average: mean and std of FAR vs alpha,
  the **histogram of per-calibration-set FAR plotted next to the theoretical
  Beta(n+1−k, k) law** (Beta-binomial for a finite test set), and TPR at the threshold.
- The PAC variant (`P(FAR ≤ α) ≥ 1 − δ`), i.e. Deng's tolerance limit (smallest rank r
  with Pr{Bin(m, α) ≥ m − r + 1} ≥ 1 − δ), next to the marginal one. A unit test must
  reproduce Deng's primary setting (m = 150, α = 0.10, δ = 0.05 → r = 142, confidence
  0.9693) and m_min = 299 for α = 1%, δ = 5%.
- Wilson 95% intervals for every empirical FAR (tested against hand-computed values,
  e.g. 1/137 → [0.13%, 4.02%]).
- `α_min(n)` helpers: 1/(n+1) for split conformal and 1 − δ^(1/m) for the tolerance limit,
  used to mark settings that cannot be certified. A comparison
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
  Both thresholds are recalibrated (split conformal and Deng's tolerance limit).
  Honest note up front: with m < 99, alpha = 1% cannot be certified by split conformal
  at all (the threshold becomes +∞; α_min = 1/(m+1)), and the tolerance limit needs
  m ≥ 299 at δ = 5%. That is a result to report, not to hide.
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
- `notebooks/results.ipynb`: a lightweight **presentation-only** notebook. It only loads
  the saved score and metric files and shows the final tables and figures: no heavy
  computation, no GPU, runs on CPU in under a minute. It is committed **with its
  outputs**, so it can be read on GitHub without running it. All real logic stays in
  the tested package and scripts; the notebook only calls them. Figures are embedded
  at modest resolution so the file stays under the repo guard's 5 MiB limit. CI
  executes it headless on the committed result files (once they exist) to make sure
  it still runs.
- Optional: PaDiM, if time permits (this would need another laptop run).

**Effort (rough guess):** ~2 sessions. **You do locally:** optionally render thumbnails of failure
cases and review the figures.

---

## Future work (not in scope now)

Listed so they are not forgotten. **Not started, and not part of Phases 0–5.**

- **SPARC-style feature correction + conformal calibration:** apply a per-cell subspace
  correction (arXiv:2608.18585) built from a few deployment-lot normals, then calibrate the
  threshold on corrected scores. Does this restore the false-alarm target better than
  recalibration alone?
- **A natural-shift dataset:** evaluate on MVTec AD 2 or AeBAD-S (shift-prone according to
  SPARC), once access and licence have been verified.

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
