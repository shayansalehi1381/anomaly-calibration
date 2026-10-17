# Split protocol

How every VisA image is used, why, and how leakage is prevented. Code:
`src/anomcal/data/splits.py`; tests: `tests/test_splits.py`, `tests/test_extract.py`.

## 1. The official split (verified)

Source: `split_csv/1cls.csv` from the public
[amazon-science/spot-diff](https://github.com/amazon-science/spot-diff) repository
(commit `2a692ab`, 2022-09-22; SHA-256
`a48557e6033318cb90556f706196bc9d247a776a23ea51aecee5a80dd0332995`), read in the
cloud on 2026-10-17. Only this small CSV was read; no images. Every normal row has an
empty mask, every anomalous row has a mask, and all 10,821 image paths are unique.

| class | train normal | test normal | test anomaly |
|---|---:|---:|---:|
| candle | 900 | 100 | 100 |
| capsules | 542 | 60 | 100 |
| cashew | 450 | 50 | 100 |
| chewinggum | 453 | 50 | 100 |
| fryum | 450 | 50 | 100 |
| macaroni1 | 900 | 100 | 100 |
| macaroni2 | 900 | 100 | 100 |
| pcb1 | 904 | 100 | 100 |
| pcb2 | 901 | 100 | 100 |
| pcb3 | 905 | 101 | 100 |
| pcb4 | 904 | 101 | 100 |
| pipe_fryum | 450 | 50 | 100 |
| **total** | **8,659** | **962** | **1,200** |

These counts are hard-coded in `anomcal.data.visa.EXPECTED_1CLS_COUNTS`. On the laptop,
`python -m anomcal.data.visa --data-root data/visa` re-reads the local CSV, prints this
table, compares it with the verified one, and checks that every listed image and mask
exists on disk.

**Why the official test split is too small for false-alarm rates:** it has only 50–101
normal images per class. At α = 1% that means 0.5–1 expected false alarms per class, so
a single random false alarm moves the measured FAR by 1–2 percentage points.

## 2. Protocol `calib` (all calibration and shift results)

Per class and per seed `s`:

1. Sort the official **train normals** by image id, then shuffle them with a generator
   seeded by `("memory-split", class, s)`.
2. The first `round(memory_fraction × n_train)` images form the **memory bank**
   (default `memory_fraction = 0.5`); the detector is fit on these, on clean images only.
3. The remaining train normals **plus all official test normals** form the
   **held-out normal pool**.
4. All anomalies (official test) form the **anomaly set**, used only for evaluation
   (AUROC, detection rate). Never for fitting or calibration.
5. Every image in the held-out pool and the anomaly set is scored on the laptop under
   every condition (clean + each corruption × severity).
6. On CPU (Phase 3), the held-out pool is split R = 100 times into **calibration** and
   **test** normals (`partition_heldout`, default 50/50), each with its own seed.

Resulting sizes with the defaults. These are derived from the counts above; they are
not experimental results:

| class | memory bank | held-out normals | ≈ n_cal (50%) | ≈ n_test (50%) |
|---|---:|---:|---:|---:|
| candle | 450 | 550 | 275 | 275 |
| capsules | 271 | 331 | 166 | 165 |
| cashew | 225 | 275 | 138 | 137 |
| chewinggum | 226 | 277 | 138 | 139 |
| fryum | 225 | 275 | 138 | 137 |
| macaroni1 | 450 | 550 | 275 | 275 |
| macaroni2 | 450 | 550 | 275 | 275 |
| pcb1 | 452 | 552 | 276 | 276 |
| pcb2 | 450 | 551 | 276 | 275 |
| pcb3 | 452 | 554 | 277 | 277 |
| pcb4 | 452 | 553 | 276 | 277 |
| pipe_fryum | 225 | 275 | 138 | 137 |

(Python's `round` rounds halves to even, e.g. 453 × 0.5 = 226.5 → 226. It is
deterministic, so this does not matter.)

**What these sizes allow (pre-registered, from the math in `INTERVIEW_NOTES.md`):**

- Marginal guarantee at α = 1% needs n_cal ≥ 99, and α = 5% needs ≥ 19: satisfied in
  every class.
- PAC guarantee P(FAR ≤ α) ≥ 95% needs n_cal ≥ 299 at α = 1%: **not satisfied in any
  class** with a 50/50 partition. At α = 5% it needs n_cal ≥ 59: satisfied in every class.
  Phase 3 reports this honestly and may vary the calibration fraction.
- With n_test ≈ 137, one false alarm moves the measured FAR by 0.73 percentage points.
  This is why we average over R = 100 partitions and compare with the Beta law instead
  of trusting a single split.

**Cost of the smaller memory bank:** PatchCore normally uses all train normals. Using
half may lower AUROC somewhat. The `official` protocol below measures that directly.

## 3. Protocol `official` (literature comparison only)

Memory bank = **all** official train normals; evaluation = the official test set
(normals + anomalies), clean images only. Only **image AUROC** (and pixel metrics from
Phase 2) is reported from it, in a table labelled "official VisA 1cls split". No
calibration result ever uses this protocol: its test normals are too few, and there
is no held-out pool.

## 4. Which numbers use which split

| Reported quantity | Protocol | Images |
|---|---|---|
| Image AUROC "for literature comparison" | `official` | official test set, memory = all train normals |
| Image AUROC "on our split" | `calib` | held-out normals + anomalies, memory = ½ train normals |
| Pixel AUROC / PRO | `official` (and `calib` where reported) | anomalous images with masks, clean only |
| FAR, detection rate, everything about calibration and shift | `calib` | calibration/test partitions of the held-out pool |

Every table in the README states its protocol.

## 5. Leakage checks (the guarantee depends on them)

A calibration image that is also in the memory bank has an artificially low score
(its nearest neighbour is itself, at distance ≈ 0). That pulls the threshold down and
silently breaks the false-alarm guarantee. Checks in place:

1. `assert_disjoint(split, records, root)` runs **before any scoring**, for every
   (protocol, class, seed). It raises `LeakageError` (never caught) if:
   - any two of memory / held-out / anomaly share an image id, or a set holds a duplicate;
   - the memory bank contains anything that is not an official *train normal*;
   - the held-out pool contains an anomaly, or the anomaly set a normal image;
   - two ids resolve to the same file on disk.
2. The split ids are saved to `splits/<protocol>/seed<s>/<class>.json` in the output
   folder (an audit trail). On resume, the split must be identical, or the run aborts.
3. Tests: disjointness for all 12 real-sized classes × 3 seeds, 5 fractions × 20 seeds,
   one test per overlap type that **must fail**, and an end-to-end test that no
   memory-bank id ever appears in a score file.
4. Phase 3 adds the same checks for calibration ∩ test.

## 6. Image pipeline

```
JPEG on disk → RGB → resize to 256×256 (bilinear, area-aware) → corruption (or none) → scorer
```

- One **canonical resolution (256 px)** for all methods, so that "blur σ = 2 px" means the
  same thing for every detector. Each scorer then applies its own preprocessing (Phase 2).
- Resizing to a square ignores VisA's aspect ratio. This is common practice and is
  documented as a choice.
- Masks are binarised (`!= 0`, VisA stores defect-class ids) and resized with nearest
  neighbour, so they stay binary.
- Corruptions are deterministic: the noise is seeded per (image id, family, level), so
  resuming, batch size and processing order cannot change any score.

## 7. Settings that must not change within one output folder

`memory_fraction`, `image_size`, `max_images` and the split file's SHA-256 are stored in
`fingerprint.json`. A rerun with different values refuses to resume into the same
folder; use a new `--out`. Methods, seeds, classes and severity levels can be added at
any time. Only the missing chunks are computed.
