# Interview notes

A living document: key decisions, the math, what failed and why. Updated at the end
of every phase. Sections marked *(pending)* will be filled in once the corresponding
phase has real results. Nothing in this file is a result until it says so.

---

## 1. The problem in one minute

An anomaly detector outputs a score `s(x)`, where higher means more anomalous. To deploy
it, you need a **threshold** `t`: alarm if `s(x) > t`. AUROC says nothing about which
`t` to pick or how many false alarms it will produce. In a factory, the false-alarm
rate (FAR, the fraction of good parts flagged) is what the operator feels every day.

**Goal:** pick `t` so that FAR ≤ α (e.g. 1% or 5%) is *guaranteed*, using only normal
images (anomalies are rare and diverse, so we cannot rely on having them), and then check
whether that guarantee survives realistic changes in imaging conditions.

## 2. Split-conformal threshold: the math

**Setup.** Scores of `n` held-out normal calibration images: `S_1, …, S_n`. A new normal
test image has score `S_{n+1}`. The detector (memory bank) was fit on *other* images.

**Assumption: exchangeability.** The joint distribution of `(S_1, …, S_{n+1})` does not
change when the order is permuted. i.i.d. normals satisfy it. Under distribution shift,
it fails. That is the whole point of Phase 4.

**Threshold.** Sort the calibration scores, `S_(1) ≤ … ≤ S_(n)`, and set

```
k = ⌈(n + 1)(1 − α)⌉,      t = S_(k)   (t = +∞ if k > n)
```

**Guarantee.** Under exchangeability, and with no ties, the rank of `S_{n+1}` among all
`n + 1` scores is uniform on `{1, …, n+1}`. The test image raises an alarm iff
`S_{n+1} > S_(k)`, i.e. iff its rank is in `{k+1, …, n+1}`. That happens with probability

```
P(alarm) = (n + 1 − k) / (n + 1)
```

Since `k ≥ (n+1)(1−α)`, this is **≤ α**. Since `k < (n+1)(1−α) + 1`, it is also
**> α − 1/(n+1)**. So `α − 1/(n+1) < E[FAR] ≤ α`. The method is valid, and not much more
conservative than it needs to be.

**Worked examples** (computed, not experimental results):

| n | α | k | E[FAR] |
|---|---|---|---|
| 199 | 0.05 | 190 | 10/200 = 0.050 |
| 99 | 0.01 | 99 | 1/100 = 0.010 |
| 50 | 0.01 | ⌈50.49⌉ = 51 > n | t = +∞, never alarms |
| 500 | 0.01 | 496 | 5/501 ≈ 0.00998 |

**Consequence:** a non-trivial threshold at level α needs `n ≥ 1/α − 1`, i.e. 99 normals
for α = 1% and 19 for α = 5%. This matters for the recalibration study (Phase 4),
where we only have a few shifted normals.

**Why the "+1"?** The naive threshold, the empirical (1 − α) quantile of the `n`
calibration scores, ignores that the test point is the (n+1)-th exchangeable draw. It
under-covers by up to about `1/n`. Phase 3 compares both.

**Conformal p-value (equivalent view).**
`p(x) = (1 + #{i : S_i ≥ s(x)}) / (n + 1)`. Raising an alarm iff `p ≤ α` gives exactly
the same decisions as `s(x) > S_(k)`. Under exchangeability, `P(p ≤ α) ≤ α`.

**The FAR of one calibration draw is random.** The guarantee above is an *average over
calibration sets*. For a fixed calibration set, the true FAR is `1 − F(S_(k))`, where
`F` is the CDF of normal scores. Because `F(S_(k))` is the k-th uniform order statistic,
for continuous scores

```
FAR | calibration set ~ Beta(n + 1 − k, k)      mean (n + 1 − k)/(n + 1)
```

The empirical FAR on `m` test normals is then Beta-binomial. Phase 3 plots the
histogram of FAR over R = 100 random calibration/test partitions next to this
theoretical distribution, which is the honest way to show the guarantee.

**PAC / training-conditional version.** An operator may want "with probability
≥ 1 − δ over the calibration draw, FAR ≤ α", instead of a guarantee on average. Using the
Beta law, choose the smallest `k` with `P(Beta(n+1−k, k) > α) ≤ δ`. With the extreme
choice `k = n` (threshold = maximum calibration score), `P(FAR > α) = (1 − α)^n`. So the
requirement is

```
n ≥ ln δ / ln(1 − α):   α = 1%, δ = 5%  →  n ≥ 299;    α = 5%, δ = 5%  →  n ≥ 59
```

Implementation note: `ceil` is discontinuous at integers. If floating-point rounding
pushes `(n+1)*(1−α)` just above an integer, `k` jumps by one. A quick check of
n < 5000 and common α values found no such case, but Phase 3 still computes `k` with
exact rational arithmetic (`fractions.Fraction`) and tests it, so correctness does not
depend on that luck.

## 3. Why distribution shift breaks it

The proof uses only exchangeability between calibration and test scores. If the test
images are brighter, blurrier, noisier or more compressed than the calibration images,
their normal scores come from a different distribution `F'`. The FAR becomes
`1 − F'(t)`:

- if the corruption **raises** normal scores (typical: the detector sees "unusual"
  texture), FAR goes **above** α: the guarantee is silently broken;
- if it **lowers** them (e.g. blur removes high-frequency texture that the memory bank
  never contained), FAR goes below α, and the **detection rate** usually drops too.

Hence we report FAR *and* detection rate under every shift.

## 4. Key decisions log

| # | Decision | Why | Phase |
|---|---|---|---|
| D1 | Laptop does GPU scoring only; everything else runs on CPU from score files. | No GPU or dataset in the cloud; small, committed score files make every analysis reproducible from git. | 0 |
| D2 | Corruptions are applied on the laptop, inside the extraction script. | One laptop run produces all scores; no second GPU pass is needed for Phase 4. | 0 |
| D3 | Score every image that is not in the memory bank; draw calibration/test partitions on CPU. | The guarantee is an average over calibration draws, so it has to be measured over many draws (R = 100). Changing the split protocol later costs nothing. | 0 |
| D4 | Enlarge the normal test pool beyond the official test split. | The official split has roughly 80 normal test images per class. At α = 1% that means < 1 expected false alarm, which is too noisy. | 0 |
| D5 | Per-item seeds from SHA-256 of the item's identity, never Python's `hash()`. | Corruption noise must not depend on processing order or resumption; `hash()` is salted per process. | 0 |
| D6 | Own lean PatchCore in torch, no faiss. | faiss is painful on Windows; we need full control over the memory bank and over scoring corrupted images. | 0 (agreed) |
| D7 | CI fails if data, weights, archives, images outside figure folders, or files > 5 MiB are tracked. | Working rule 3, enforced mechanically, not by discipline. | 0 |
| D8 | Use the verified official counts (read from the public `1cls.csv`), not the paper's rounded numbers. | Test normals are 50–101 per class, not "about 80"; this decides which guarantees are reachable. | 1 |
| D9 | `memory_fraction = 0.5`: half of the train normals go to the memory bank, the rest + the official test normals form the held-out pool. | Gives n_cal ≈ 138–277 per class (at 50/50): enough for the marginal guarantee at α = 1% (n ≥ 99), not for PAC at α = 1%, δ = 5% (n ≥ 299). Cost: a smaller memory bank, measured with the `official` protocol. | 1 |
| D10 | A separate `official` protocol (all train normals in memory, official test set, clean only). | Literature comparison needs the standard protocol; mixing it with ours would make AUROC numbers incomparable. | 1 |
| D11 | Corrupt at one canonical 256 px resolution for all methods, after resizing. | "σ = 2 px" must mean the same for every detector; each method does its own preprocessing afterwards. | 1 |
| D12 | ImageNet-C parameter tables for all 5 levels; levels used = config (`--severities 3` → 1, 3, 5). | Going from 3 to 5 severities adds exactly levels 2 and 4, so a resumed run computes only the new chunks. | 1 |
| D13 | Brightness implemented as RGB scaling by V_new / V instead of an HSV round trip. | Mathematically the same operation as adding c to V (H and S depend only on channel ratios); a test checks it against `colorsys` to within 1 grey level. | 1 |
| D14 | Resumability through atomic chunk files (`*.tmp` + `os.replace`) and a `fingerprint.json`. | A crash can never leave a half-written file that looks finished; changing a result-affecting setting can never silently mix two experiments in one folder. | 1 |
| D15 | `--device` has no default, and asking for CUDA without CUDA is an error. | A silent CPU fallback would turn a 1-hour GPU run into a multi-day CPU run without anyone noticing. | 1 |

## 5. What failed and why

*(pending: filled in as it happens. Negative results are kept, not deleted.)*

- Phase 0: arXiv:2608.15090 and arXiv:2608.18585 could not be read, because arxiv.org
  was blocked by the cloud environment's network policy. They are cited by title only;
  see `docs/PRIOR_WORK.md`.
- Phase 1, **`.gitignore` bug**: the pattern `data/` (meant for the dataset folder) also
  matched the package folder `src/anomcal/data/`, so the new adapter code was silently
  ignored by git. Found because `git status` did not list the new files. Fix: anchor
  it as `/data/`, plus a regression test. Lesson: an ignore rule is code and deserves
  a test.
- Phase 1, **float32 cancellation in kNN distances**: the dummy scorer computed squared
  distances as |a|² − 2a·b + |b|² in float32, and a memory-bank image scored **0.71**
  against itself instead of 0. Both terms are about 3·10⁶ while their difference is
  tiny, and float32 has only ~7 significant digits. Fix: compute in float64. This
  matters for PatchCore in Phase 2, which uses the same trick on GPU.
- Phase 1, **pandas 3 string dtype**: the schema validator compared dtype names
  (`"string"`), but pandas 3 reports `"str"` for ordinary string columns. Fix: check
  "is a string column" semantically, not by name.

## 6. Results to be able to explain

*(pending: Phase 2 onwards.)*
