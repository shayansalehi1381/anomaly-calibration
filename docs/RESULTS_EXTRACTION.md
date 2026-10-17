# Phase 2 results: detector accuracy on clean images

**What this is:** clean-image accuracy of the three detectors on VisA (image AUROC, pixel
AUROC, PRO). **What it is not:** the project's main result. The main question is whether
calibrated thresholds keep their false-alarm rate (Phases 3–4). Detector accuracy only
decides how many anomalies are *caught* at a given false-alarm rate.

All numbers below are **measured**: computed by `scripts/auroc_table.py` and
`pandas` from the committed files in `results/scores/` (laptop GPU run, RTX 3060 Laptop,
`configs/full_run.json`, no reductions). They were recomputed in the cloud from those files
and match the author's laptop output.

## Run facts (measured)

| item | value |
|---|---|
| plan | 12 classes × 2 protocols × 3 seeds × 3 methods, severities 1/3/5 (13 conditions) |
| chunks | 1,512 (42 from the one-class measurement + 1,470 in the full run) |
| image-conditions scored | 764,064 |
| wall-clock time of the full run | **3 h 53 min** (estimate from `estimate_full_run.py`: 3 h 44 min, i.e. about 4% low) |
| one full-size class (cashew, seed 0, stage 2) | 3 min 04 s |
| peak VRAM measured on cashew (baseline / PatchCore / DINOv2-kNN) | 362 / 1,222 / 1,035 MB |
| analytic upper bound for the largest bank | 1,456 / 2,458 / 2,082 MB (6 GB was never at risk) |
| score files | 18 Parquet + `pixel_metrics.csv`, about 6 MB in total, largest 0.61 MB |

## Image AUROC (%), clean images, mean ± std over 3 seeds

### Official VisA 1cls split (memory bank = all train normals): literature-comparable

| class | baseline | dinov2_knn | patchcore |
|---|---:|---:|---:|
| candle | 90.0 ± 0.0 | 96.0 ± 0.0 | 97.2 ± 0.3 |
| capsules | 58.2 ± 0.0 | 88.0 ± 0.0 | 73.4 ± 0.1 |
| cashew | 88.0 ± 0.0 | 94.2 ± 0.0 | 97.2 ± 0.1 |
| chewinggum | 86.8 ± 0.0 | 98.6 ± 0.0 | 99.3 ± 0.3 |
| fryum | 84.2 ± 0.0 | 90.3 ± 0.0 | 92.8 ± 0.2 |
| macaroni1 | 77.9 ± 0.0 | 81.2 ± 0.0 | 85.7 ± 0.4 |
| macaroni2 | 59.3 ± 0.0 | 70.5 ± 0.0 | 55.7 ± 1.1 |
| pcb1 | 79.1 ± 0.0 | 88.7 ± 0.0 | 94.1 ± 0.2 |
| pcb2 | 86.9 ± 0.0 | 87.2 ± 0.0 | 93.3 ± 0.4 |
| pcb3 | 78.6 ± 0.0 | 88.0 ± 0.0 | 92.9 ± 0.3 |
| pcb4 | 97.3 ± 0.0 | 93.3 ± 0.0 | 99.3 ± 0.1 |
| pipe_fryum | 78.1 ± 0.0 | 98.9 ± 0.0 | 98.7 ± 0.1 |
| **mean** | **80.4** | **89.6** | **90.0** |

On the official split, baseline and DINOv2-kNN have std 0.0 because nothing in them is
random (the memory bank is all train normals, and there is no coreset); only PatchCore's
coreset depends on the seed.

### Our calib split (memory bank = ½ of the train normals; larger normal pool)

| class | baseline | dinov2_knn | patchcore |
|---|---:|---:|---:|
| candle | 85.2 ± 0.9 | 95.7 ± 0.6 | 96.2 ± 0.4 |
| capsules | 58.9 ± 0.1 | 85.9 ± 0.5 | 70.2 ± 0.8 |
| cashew | 86.3 ± 0.6 | 93.0 ± 0.7 | 96.0 ± 0.6 |
| chewinggum | 88.7 ± 0.5 | 98.3 ± 0.1 | 99.0 ± 0.2 |
| fryum | 82.7 ± 1.3 | 90.7 ± 0.2 | 92.0 ± 0.4 |
| macaroni1 | 78.6 ± 0.8 | 83.8 ± 1.8 | 85.2 ± 1.3 |
| macaroni2 | 60.0 ± 0.7 | 76.7 ± 1.6 | 59.4 ± 1.5 |
| pcb1 | 79.8 ± 0.3 | 90.4 ± 0.7 | 95.2 ± 0.4 |
| pcb2 | 83.6 ± 0.3 | 85.3 ± 0.8 | 93.0 ± 0.4 |
| pcb3 | 77.7 ± 0.6 | 87.3 ± 0.5 | 91.0 ± 0.3 |
| pcb4 | 96.7 ± 0.7 | 92.5 ± 0.6 | 99.4 ± 0.1 |
| pipe_fryum | 81.8 ± 0.5 | 98.2 ± 0.2 | 98.1 ± 0.3 |
| **mean** | **80.0 ± 0.1** | **89.8 ± 0.3** | **89.6 ± 0.3** |

Halving the memory bank changes the class-mean AUROC by at most 0.4 points (official vs
calib means above). Some classes move by more than that, in both directions; the
evaluation sets also differ between the two protocols.

## Pixel metrics (official split, clean images; `results/scores/pixel_metrics.csv`)

Mean over the 12 classes, then mean ± std over 3 seeds. The baseline has no maps.

| method | pixel AUROC (%) | PRO, FPR ≤ 0.3 (%) |
|---|---:|---:|
| patchcore | 97.7 ± 0.0 | 82.8 ± 0.2 |
| dinov2_knn | 94.8 ± 0.0 | 72.7 ± 0.0 |

## Sanity check against SPARC (arXiv:2608.18585, Table 1; different preprocessing)

| method | ours (official, class mean) | SPARC reference | gap | script verdict (threshold 5 points) |
|---|---:|---:|---:|---|
| PatchCore WRN-50 | 90.0 | ~94.3 | −4.3 | within threshold |
| DINOv2-kNN vs AnomalyDINO | 89.6 | ~95.3 | **−5.7** | **flagged** |

### Known limitation: the DINOv2-kNN gap (not explained)

Our DINOv2-kNN is 5.7 points below the AnomalyDINO number reported by SPARC. **We have
not established why.** Candidate causes, all *unverified hypotheses*:

1. **Input resolution:** we use 224 px (16 × 16 patches). AnomalyDINO's text was not
   accessible from our environment, so we could not confirm which resolution it or SPARC
   used. The Phase 2 docs mentioned "448/672 px" from memory; that detail is **unverified**.
2. **Image-score aggregation:** we use the mean of the top 1% patch distances; with only
   256 patches that is 3 patches. Whether AnomalyDINO uses this exact statistic is also
   **unverified** by us.
3. **Preprocessing:** we squash the full image to a square; SPARC resizes and center-crops.
4. **Masking:** AnomalyDINO describes zero-shot masking for some objects; we use none.

The PatchCore gap (−4.3) has the same preprocessing caveat and is below the 5-point flag,
which is a heuristic, not a test.

**Optional check, not run (no new GPU work without approval):** score one class with
DINOv2 at 448 px (`--opt dinov2_knn.image_size=448` in a new `--out` folder, `--classes
cashew --seeds 0 --protocols official`). Estimated cost: this touches only about 600 images (450 to fit, 150 to score; official
protocol, clean only). The backbone does ~4× the work per image, and kNN ~16× (4× more
query patches against a 4× larger bank). The whole DINOv2 part of the full run
(~260k scored images) was predicted at 49 min, so this is on the order of a few minutes.
Estimate, not a measurement; the bank (~1.4 GB at 448 px) still fits in 6 GB.

### Weak classes: macaroni2 and capsules

- **PatchCore**: macaroni2 55.7–59.4 (near chance) and capsules 70.2–73.4.
- **DINOv2-kNN**: macaroni2 70.5–76.7; capsules 85.9–88.0.
- **Baseline**: both classes near 58–60.

Several papers found by a web search list macaroni2 and capsules among PatchCore's
weakest VisA classes, and one attributes this to many similar object instances appearing
at varying positions in each image. **We did not verify those papers' tables**, so
"known to be hard" is unverified here. What is verified is our own measurement above.

### Why low AUROC does not invalidate the false-alarm study

Every threshold in Phases 3–4 is computed from **normal scores only**, and its guarantee
(E[FAR] ≤ α, or Pr[FAR ≤ α] ≥ 1 − δ) holds for *any* score function, good or bad, as long
as calibration and test normals are exchangeable. A weak detector does not raise the
false-alarm rate. It lowers the **detection rate** at that threshold, which we report per
class and method. Example: on macaroni2, PatchCore at a 5% FAR will catch few anomalies,
and that is a statement about detection power, not about calibration.
