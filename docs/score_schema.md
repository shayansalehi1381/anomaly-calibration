# Score file schema and output layout

Produced on the laptop by `scripts/extract_scores.py` (logic in `src/anomcal/extract.py`,
schema in `src/anomcal/scores_io.py`, enforced by `validate()` on every write and read).

## Output folder

```
<out>/                                   default: outputs/extract  (git-ignored)
  full/                                  a full run   (or smoke_max<N>/ for --max-images N)
    fingerprint.json                     settings that must not change on resume
    run_meta.json                        one entry per invocation (see below)
    splits/<protocol>/seed<s>/<class>.json         ids per role: the leakage audit trail
    chunks/<protocol>/<method>/seed<s>/<class>/<corruption>_s<level>.parquet
    scores/<protocol>_<method>_seed<s>.parquet     consolidated: what gets committed
```

- A **chunk** holds the scores of one (protocol, method, seed, class, condition). It is
  written to `*.tmp` and then atomically renamed, so an existing chunk is always
  complete; that is what makes the run resumable.
- The **consolidated** files are rebuilt from the chunks at the end of every complete
  run. Copy them to `results/scores/` and commit them (only these and `run_meta.json`).

## Columns (one row per image × condition)

| column | dtype | meaning |
|---|---|---|
| `image_id` | string | `<class>/<normal\|anomaly>/<file stem>`, e.g. `candle/normal/0836` |
| `class` | string | VisA class, e.g. `pcb1` |
| `label` | int8 | 0 = normal, 1 = anomaly (ground truth) |
| `role` | string | `heldout_normal` or `anomaly` (memory-bank images are never scored) |
| `official_split` | string | `train` or `test` in VisA `1cls.csv` (lets us subset to the official test set) |
| `protocol` | string | `calib` (our protocol) or `official` (literature comparison) |
| `method` | string | `dummy` (Phase 1); `baseline`, `patchcore`, `dinov2_knn` (Phase 2) |
| `seed` | int32 | seed of the memory-bank split and of the method's own randomness |
| `corruption` | string | `none`, `brightness`, `gaussian_blur`, `gaussian_noise`, `jpeg_compression` |
| `severity` | int8 | 0 for `none`, else 1–5 (ImageNet-C level; parameters in `run_meta.json`) |
| `score` | float32 | image-level anomaly score, higher = more anomalous, always finite |

Invariants checked by `validate()`: exact column order and dtypes; no missing values;
`role == "anomaly"` iff `label == 1`; `severity == 0` iff `corruption == "none"`; no
duplicate (protocol, method, seed, class, image_id, corruption, severity).

**Example rows**, copied from a real run on the *synthetic fake dataset* with the
dummy scorer (`python -m anomcal.data.fake_visa`, then
`extract_scores.py --classes candle --protocols calib --seeds 0`). They show the format
only and say nothing about VisA:

```
          image_id  class  label           role official_split protocol method  seed    corruption  severity      score
candle/anomaly/000 candle      1        anomaly           test    calib  dummy     0          none         0 476.857941
candle/normal/0000 candle      0 heldout_normal          train    calib  dummy     0 gaussian_blur         3 298.540100
```

**Expected size of a full run:** per (method, seed), the calib protocol has about
(5,293 held-out normals + 1,200 anomalies) × 13 conditions = 84,409 rows, plus 2,162 rows
from the official protocol. Parquet compresses this to a few MB; the repo guard rejects
any file over 5 MiB, so we will check the real size after the first full run.

## `run_meta.json`

A list with one object per invocation, including interrupted ones:

- `config`: every resolved option (after merging `--config` and CLI flags);
- `fingerprint`: `memory_fraction`, `image_size`, `max_images`, `split_csv_sha256`;
- `levels` and `severity_parameters`: the **exact corruption parameters** of each level used;
- `environment`: Python, numpy, pandas, Pillow, SciPy (and torch / GPU name if present),
  platform, git commit, device;
- `estimate`: the warm-up timings per image (load, each corruption, fit, score) and the
  extrapolated total;
- `status` (`complete`, `interrupted`, `estimate only`, `nothing to do`),
  `chunks_written`, `elapsed_seconds`, and the paths of the consolidated files.

## Pixel-level outputs (Phase 2)

Methods that produce anomaly maps will compute pixel AUROC / PRO on the laptop, for
clean images only. They write a small JSON next to the scores; the maps themselves are
never saved. That schema is added here in Phase 2.
