# Results summary

Every number below is read from the committed CSVs by `scripts/make_results_summary.py` and re-checked by `tests/test_docs_consistency.py`. **Measured** = computed from the real laptop scores by CPU code in this repo. **Estimated** = a prediction, not a result (listed in §8). Level **0.5** is an extra mild level we defined (stored as `severity = -1`); it is not an ImageNet-C level. α is the target false-alarm rate, FAR the empirical false-alarm rate, excess FAR = FAR − α, TPR the detection rate on anomalous images at the same threshold.

## 1. Detector accuracy on clean images (measured)

Image AUROC (%), class mean, our `calib` split (memory bank = ½ of train normals), mean over 3 seeds. Source: `results/phase4/auroc_by_condition.csv`, column `auroc`, rows `corruption = none`. The official-split numbers (80.4 / 89.6 / 90.0), per-class tables and the SPARC sanity check are in `docs/RESULTS_EXTRACTION.md` (measured on the laptop; not recomputed here).

| method | AUROC (calib split) |
|---|---:|
| baseline | 80.0 |
| DINOv2-kNN | 89.8 |
| PatchCore | 89.6 |

## 2. Clean calibration (Phase 3, measured)

Pooled over 12 classes × 3 seeds × 100 random 50/50 partitions. `theory` = exact Beta-binomial mean. `P(FAR>α)` = share of partitions whose test-half FAR exceeds α. Source: `results/phase3/summary_by_method.csv` (`far_pooled`, `far_theory_mean`, `share_far_above_alpha`, `tpr_mean`). The tolerance limit at α = 1% is **not certifiable** (needs 299 calibration normals, we have 138–277).

| method | threshold | α | FAR | theory | P(FAR>α) | TPR |
|---|---|---|---|---|---|---|
| baseline | conformal | 0.01 | 0.0073 <!-- check: phase3/summary_by_method.csv ;; far_pooled ;; method=baseline,kind=conformal,alpha=0.01 ;; 0.00732073 ;; 6e-05 --> | 0.0071 | 0.292 | 0.195 |
| baseline | conformal | 0.05 | 0.0469 <!-- check: phase3/summary_by_method.csv ;; far_pooled ;; method=baseline,kind=conformal,alpha=0.05 ;; 0.0468822 ;; 6e-05 --> | 0.0458 | 0.416 | 0.391 |
| baseline | conformal | 0.1 | 0.0975 <!-- check: phase3/summary_by_method.csv ;; far_pooled ;; method=baseline,kind=conformal,alpha=0.1 ;; 0.0974543 ;; 6e-05 --> | 0.0961 | 0.445 | 0.514 |
| baseline | tolerance | 0.01 | not certifiable | – | – | – |
| baseline | tolerance | 0.05 | 0.0275 <!-- check: phase3/summary_by_method.csv ;; far_pooled ;; method=baseline,kind=tolerance,alpha=0.05 ;; 0.027477 ;; 6e-05 --> | 0.0261 | 0.089 | 0.313 |
| baseline | tolerance | 0.1 | 0.0692 <!-- check: phase3/summary_by_method.csv ;; far_pooled ;; method=baseline,kind=tolerance,alpha=0.1 ;; 0.069225 ;; 6e-05 --> | 0.0663 | 0.109 | 0.450 |
| DINOv2-kNN | conformal | 0.01 | 0.0071 <!-- check: phase3/summary_by_method.csv ;; far_pooled ;; method=dinov2_knn,kind=conformal,alpha=0.01 ;; 0.00714682 ;; 6e-05 --> | 0.0071 | 0.293 | 0.376 |
| DINOv2-kNN | conformal | 0.05 | 0.0461 <!-- check: phase3/summary_by_method.csv ;; far_pooled ;; method=dinov2_knn,kind=conformal,alpha=0.05 ;; 0.0461323 ;; 6e-05 --> | 0.0458 | 0.404 | 0.643 |
| DINOv2-kNN | conformal | 0.1 | 0.0965 <!-- check: phase3/summary_by_method.csv ;; far_pooled ;; method=dinov2_knn,kind=conformal,alpha=0.1 ;; 0.0965167 ;; 6e-05 --> | 0.0961 | 0.441 | 0.729 |
| DINOv2-kNN | tolerance | 0.01 | not certifiable | – | – | – |
| DINOv2-kNN | tolerance | 0.05 | 0.0269 <!-- check: phase3/summary_by_method.csv ;; far_pooled ;; method=dinov2_knn,kind=tolerance,alpha=0.05 ;; 0.0269238 ;; 6e-05 --> | 0.0261 | 0.081 | 0.570 |
| DINOv2-kNN | tolerance | 0.1 | 0.0686 <!-- check: phase3/summary_by_method.csv ;; far_pooled ;; method=dinov2_knn,kind=tolerance,alpha=0.1 ;; 0.0686188 ;; 6e-05 --> | 0.0663 | 0.103 | 0.691 |
| PatchCore | conformal | 0.01 | 0.0071 <!-- check: phase3/summary_by_method.csv ;; far_pooled ;; method=patchcore,kind=conformal,alpha=0.01 ;; 0.00714934 ;; 6e-05 --> | 0.0071 | 0.285 | 0.414 |
| PatchCore | conformal | 0.05 | 0.0467 <!-- check: phase3/summary_by_method.csv ;; far_pooled ;; method=patchcore,kind=conformal,alpha=0.05 ;; 0.0467196 ;; 6e-05 --> | 0.0458 | 0.418 | 0.650 |
| PatchCore | conformal | 0.1 | 0.0973 <!-- check: phase3/summary_by_method.csv ;; far_pooled ;; method=patchcore,kind=conformal,alpha=0.1 ;; 0.0973069 ;; 6e-05 --> | 0.0961 | 0.448 | 0.735 |
| PatchCore | tolerance | 0.01 | not certifiable | – | – | – |
| PatchCore | tolerance | 0.05 | 0.0271 <!-- check: phase3/summary_by_method.csv ;; far_pooled ;; method=patchcore,kind=tolerance,alpha=0.05 ;; 0.0270599 ;; 6e-05 --> | 0.0261 | 0.083 | 0.580 |
| PatchCore | tolerance | 0.1 | 0.0690 <!-- check: phase3/summary_by_method.csv ;; far_pooled ;; method=patchcore,kind=tolerance,alpha=0.1 ;; 0.0689565 ;; 6e-05 --> | 0.0663 | 0.101 | 0.692 |

## 3. Under corruption (Phase 4, measured): split conformal, α = 5%

Mean over 12 classes. **A** = threshold from clean calibration normals; **B** = threshold recalibrated on corrupted normals of the same corruption and level (needs normal images under the new condition). Cell = FAR (excess FAR). Source: `results/phase4/headline.csv`, columns `far`, `excess`, `tpr`, filtered by `setting`, `kind = conformal`, `alpha = 0.05`, `level`. 95% class-bootstrap intervals: columns `far_lo`, `far_hi`.

| corruption | method | A: 0.5 | A: 1 | A: 3 | A: 5 | B: 0.5 | B: 5 | TPR A / B at 1 | TPR B at 5 |
|---|---|---|---|---|---|---|---|---|---|
| brightness | baseline | 0.183 (+0.133) <!-- check: phase4/headline.csv ;; far ;; method=baseline,corruption=brightness,setting=A_no_recal,kind=conformal,alpha=0.05,level=0.5 ;; 0.18334 ;; 0.0006 --> | 0.494 (+0.444) <!-- check: phase4/headline.csv ;; far ;; method=baseline,corruption=brightness,setting=A_no_recal,kind=conformal,alpha=0.05,level=1.0 ;; 0.49402 ;; 0.0006 --> | 0.993 (+0.943) <!-- check: phase4/headline.csv ;; far ;; method=baseline,corruption=brightness,setting=A_no_recal,kind=conformal,alpha=0.05,level=3.0 ;; 0.99307 ;; 0.0006 --> | 1.000 (+0.950) <!-- check: phase4/headline.csv ;; far ;; method=baseline,corruption=brightness,setting=A_no_recal,kind=conformal,alpha=0.05,level=5.0 ;; 1 ;; 0.0006 --> | 0.045 (-0.005) <!-- check: phase4/headline.csv ;; far ;; method=baseline,corruption=brightness,setting=B_recal,kind=conformal,alpha=0.05,level=0.5 ;; 0.045187 ;; 0.0006 --> | 0.046 (-0.004) <!-- check: phase4/headline.csv ;; far ;; method=baseline,corruption=brightness,setting=B_recal,kind=conformal,alpha=0.05,level=5.0 ;; 0.045613 ;; 0.0006 --> | 0.72 / 0.31 | 0.14 |
| brightness | DINOv2-kNN | 0.053 (+0.003) <!-- check: phase4/headline.csv ;; far ;; method=dinov2_knn,corruption=brightness,setting=A_no_recal,kind=conformal,alpha=0.05,level=0.5 ;; 0.053183 ;; 0.0006 --> | 0.104 (+0.054) <!-- check: phase4/headline.csv ;; far ;; method=dinov2_knn,corruption=brightness,setting=A_no_recal,kind=conformal,alpha=0.05,level=1.0 ;; 0.10405 ;; 0.0006 --> | 0.422 (+0.372) <!-- check: phase4/headline.csv ;; far ;; method=dinov2_knn,corruption=brightness,setting=A_no_recal,kind=conformal,alpha=0.05,level=3.0 ;; 0.42224 ;; 0.0006 --> | 0.773 (+0.723) <!-- check: phase4/headline.csv ;; far ;; method=dinov2_knn,corruption=brightness,setting=A_no_recal,kind=conformal,alpha=0.05,level=5.0 ;; 0.773 ;; 0.0006 --> | 0.045 (-0.005) <!-- check: phase4/headline.csv ;; far ;; method=dinov2_knn,corruption=brightness,setting=B_recal,kind=conformal,alpha=0.05,level=0.5 ;; 0.04519 ;; 0.0006 --> | 0.046 (-0.004) <!-- check: phase4/headline.csv ;; far ;; method=dinov2_knn,corruption=brightness,setting=B_recal,kind=conformal,alpha=0.05,level=5.0 ;; 0.045538 ;; 0.0006 --> | 0.63 / 0.57 | 0.21 |
| brightness | PatchCore | 0.095 (+0.045) <!-- check: phase4/headline.csv ;; far ;; method=patchcore,corruption=brightness,setting=A_no_recal,kind=conformal,alpha=0.05,level=0.5 ;; 0.094575 ;; 0.0006 --> | 0.279 (+0.229) <!-- check: phase4/headline.csv ;; far ;; method=patchcore,corruption=brightness,setting=A_no_recal,kind=conformal,alpha=0.05,level=1.0 ;; 0.27943 ;; 0.0006 --> | 0.951 (+0.901) <!-- check: phase4/headline.csv ;; far ;; method=patchcore,corruption=brightness,setting=A_no_recal,kind=conformal,alpha=0.05,level=3.0 ;; 0.95068 ;; 0.0006 --> | 1.000 (+0.950) <!-- check: phase4/headline.csv ;; far ;; method=patchcore,corruption=brightness,setting=A_no_recal,kind=conformal,alpha=0.05,level=5.0 ;; 0.99999 ;; 0.0006 --> | 0.046 (-0.004) <!-- check: phase4/headline.csv ;; far ;; method=patchcore,corruption=brightness,setting=B_recal,kind=conformal,alpha=0.05,level=0.5 ;; 0.045821 ;; 0.0006 --> | 0.046 (-0.004) <!-- check: phase4/headline.csv ;; far ;; method=patchcore,corruption=brightness,setting=B_recal,kind=conformal,alpha=0.05,level=5.0 ;; 0.04562 ;; 0.0006 --> | 0.72 / 0.60 | 0.28 |
| gaussian_blur | baseline | 0.142 (+0.092) <!-- check: phase4/headline.csv ;; far ;; method=baseline,corruption=gaussian_blur,setting=A_no_recal,kind=conformal,alpha=0.05,level=0.5 ;; 0.1424 ;; 0.0006 --> | 0.955 (+0.905) <!-- check: phase4/headline.csv ;; far ;; method=baseline,corruption=gaussian_blur,setting=A_no_recal,kind=conformal,alpha=0.05,level=1.0 ;; 0.95545 ;; 0.0006 --> | 1.000 (+0.950) <!-- check: phase4/headline.csv ;; far ;; method=baseline,corruption=gaussian_blur,setting=A_no_recal,kind=conformal,alpha=0.05,level=3.0 ;; 1 ;; 0.0006 --> | 1.000 (+0.950) <!-- check: phase4/headline.csv ;; far ;; method=baseline,corruption=gaussian_blur,setting=A_no_recal,kind=conformal,alpha=0.05,level=5.0 ;; 1 ;; 0.0006 --> | 0.045 (-0.005) <!-- check: phase4/headline.csv ;; far ;; method=baseline,corruption=gaussian_blur,setting=B_recal,kind=conformal,alpha=0.05,level=0.5 ;; 0.045201 ;; 0.0006 --> | 0.046 (-0.004) <!-- check: phase4/headline.csv ;; far ;; method=baseline,corruption=gaussian_blur,setting=B_recal,kind=conformal,alpha=0.05,level=5.0 ;; 0.045532 ;; 0.0006 --> | 0.98 / 0.17 | 0.09 |
| gaussian_blur | DINOv2-kNN | 0.051 (+0.001) <!-- check: phase4/headline.csv ;; far ;; method=dinov2_knn,corruption=gaussian_blur,setting=A_no_recal,kind=conformal,alpha=0.05,level=0.5 ;; 0.051484 ;; 0.0006 --> | 0.160 (+0.110) <!-- check: phase4/headline.csv ;; far ;; method=dinov2_knn,corruption=gaussian_blur,setting=A_no_recal,kind=conformal,alpha=0.05,level=1.0 ;; 0.15984 ;; 0.0006 --> | 0.985 (+0.935) <!-- check: phase4/headline.csv ;; far ;; method=dinov2_knn,corruption=gaussian_blur,setting=A_no_recal,kind=conformal,alpha=0.05,level=3.0 ;; 0.98539 ;; 0.0006 --> | 1.000 (+0.950) <!-- check: phase4/headline.csv ;; far ;; method=dinov2_knn,corruption=gaussian_blur,setting=A_no_recal,kind=conformal,alpha=0.05,level=5.0 ;; 1 ;; 0.0006 --> | 0.045 (-0.005) <!-- check: phase4/headline.csv ;; far ;; method=dinov2_knn,corruption=gaussian_blur,setting=B_recal,kind=conformal,alpha=0.05,level=0.5 ;; 0.045364 ;; 0.0006 --> | 0.045 (-0.005) <!-- check: phase4/headline.csv ;; far ;; method=dinov2_knn,corruption=gaussian_blur,setting=B_recal,kind=conformal,alpha=0.05,level=5.0 ;; 0.045046 ;; 0.0006 --> | 0.70 / 0.57 | 0.12 |
| gaussian_blur | PatchCore | 0.058 (+0.008) <!-- check: phase4/headline.csv ;; far ;; method=patchcore,corruption=gaussian_blur,setting=A_no_recal,kind=conformal,alpha=0.05,level=0.5 ;; 0.058078 ;; 0.0006 --> | 0.831 (+0.781) <!-- check: phase4/headline.csv ;; far ;; method=patchcore,corruption=gaussian_blur,setting=A_no_recal,kind=conformal,alpha=0.05,level=1.0 ;; 0.83098 ;; 0.0006 --> | 1.000 (+0.950) <!-- check: phase4/headline.csv ;; far ;; method=patchcore,corruption=gaussian_blur,setting=A_no_recal,kind=conformal,alpha=0.05,level=3.0 ;; 1 ;; 0.0006 --> | 1.000 (+0.950) <!-- check: phase4/headline.csv ;; far ;; method=patchcore,corruption=gaussian_blur,setting=A_no_recal,kind=conformal,alpha=0.05,level=5.0 ;; 1 ;; 0.0006 --> | 0.046 (-0.004) <!-- check: phase4/headline.csv ;; far ;; method=patchcore,corruption=gaussian_blur,setting=B_recal,kind=conformal,alpha=0.05,level=0.5 ;; 0.04566 ;; 0.0006 --> | 0.046 (-0.004) <!-- check: phase4/headline.csv ;; far ;; method=patchcore,corruption=gaussian_blur,setting=B_recal,kind=conformal,alpha=0.05,level=5.0 ;; 0.045711 ;; 0.0006 --> | 0.98 / 0.61 | 0.15 |
| gaussian_noise | baseline | 0.975 (+0.925) <!-- check: phase4/headline.csv ;; far ;; method=baseline,corruption=gaussian_noise,setting=A_no_recal,kind=conformal,alpha=0.05,level=0.5 ;; 0.97504 ;; 0.0006 --> | 1.000 (+0.950) <!-- check: phase4/headline.csv ;; far ;; method=baseline,corruption=gaussian_noise,setting=A_no_recal,kind=conformal,alpha=0.05,level=1.0 ;; 0.99991 ;; 0.0006 --> | 1.000 (+0.950) <!-- check: phase4/headline.csv ;; far ;; method=baseline,corruption=gaussian_noise,setting=A_no_recal,kind=conformal,alpha=0.05,level=3.0 ;; 1 ;; 0.0006 --> | 1.000 (+0.950) <!-- check: phase4/headline.csv ;; far ;; method=baseline,corruption=gaussian_noise,setting=A_no_recal,kind=conformal,alpha=0.05,level=5.0 ;; 1 ;; 0.0006 --> | 0.046 (-0.004) <!-- check: phase4/headline.csv ;; far ;; method=baseline,corruption=gaussian_noise,setting=B_recal,kind=conformal,alpha=0.05,level=0.5 ;; 0.045918 ;; 0.0006 --> | 0.046 (-0.004) <!-- check: phase4/headline.csv ;; far ;; method=baseline,corruption=gaussian_noise,setting=B_recal,kind=conformal,alpha=0.05,level=5.0 ;; 0.045671 ;; 0.0006 --> | 1.00 / 0.14 | 0.08 |
| gaussian_noise | DINOv2-kNN | 0.582 (+0.532) <!-- check: phase4/headline.csv ;; far ;; method=dinov2_knn,corruption=gaussian_noise,setting=A_no_recal,kind=conformal,alpha=0.05,level=0.5 ;; 0.58212 ;; 0.0006 --> | 0.904 (+0.854) <!-- check: phase4/headline.csv ;; far ;; method=dinov2_knn,corruption=gaussian_noise,setting=A_no_recal,kind=conformal,alpha=0.05,level=1.0 ;; 0.90373 ;; 0.0006 --> | 1.000 (+0.950) <!-- check: phase4/headline.csv ;; far ;; method=dinov2_knn,corruption=gaussian_noise,setting=A_no_recal,kind=conformal,alpha=0.05,level=3.0 ;; 1 ;; 0.0006 --> | 1.000 (+0.950) <!-- check: phase4/headline.csv ;; far ;; method=dinov2_knn,corruption=gaussian_noise,setting=A_no_recal,kind=conformal,alpha=0.05,level=5.0 ;; 1 ;; 0.0006 --> | 0.046 (-0.004) <!-- check: phase4/headline.csv ;; far ;; method=dinov2_knn,corruption=gaussian_noise,setting=B_recal,kind=conformal,alpha=0.05,level=0.5 ;; 0.045659 ;; 0.0006 --> | 0.046 (-0.004) <!-- check: phase4/headline.csv ;; far ;; method=dinov2_knn,corruption=gaussian_noise,setting=B_recal,kind=conformal,alpha=0.05,level=5.0 ;; 0.045836 ;; 0.0006 --> | 0.97 / 0.25 | 0.07 |
| gaussian_noise | PatchCore | 0.847 (+0.797) <!-- check: phase4/headline.csv ;; far ;; method=patchcore,corruption=gaussian_noise,setting=A_no_recal,kind=conformal,alpha=0.05,level=0.5 ;; 0.8471 ;; 0.0006 --> | 1.000 (+0.950) <!-- check: phase4/headline.csv ;; far ;; method=patchcore,corruption=gaussian_noise,setting=A_no_recal,kind=conformal,alpha=0.05,level=1.0 ;; 1 ;; 0.0006 --> | 1.000 (+0.950) <!-- check: phase4/headline.csv ;; far ;; method=patchcore,corruption=gaussian_noise,setting=A_no_recal,kind=conformal,alpha=0.05,level=3.0 ;; 1 ;; 0.0006 --> | 1.000 (+0.950) <!-- check: phase4/headline.csv ;; far ;; method=patchcore,corruption=gaussian_noise,setting=A_no_recal,kind=conformal,alpha=0.05,level=5.0 ;; 1 ;; 0.0006 --> | 0.046 (-0.004) <!-- check: phase4/headline.csv ;; far ;; method=patchcore,corruption=gaussian_noise,setting=B_recal,kind=conformal,alpha=0.05,level=0.5 ;; 0.045722 ;; 0.0006 --> | 0.046 (-0.004) <!-- check: phase4/headline.csv ;; far ;; method=patchcore,corruption=gaussian_noise,setting=B_recal,kind=conformal,alpha=0.05,level=5.0 ;; 0.046075 ;; 0.0006 --> | 1.00 / 0.36 | 0.13 |
| jpeg_compression | baseline | 0.966 (+0.916) <!-- check: phase4/headline.csv ;; far ;; method=baseline,corruption=jpeg_compression,setting=A_no_recal,kind=conformal,alpha=0.05,level=0.5 ;; 0.96562 ;; 0.0006 --> | 1.000 (+0.950) <!-- check: phase4/headline.csv ;; far ;; method=baseline,corruption=jpeg_compression,setting=A_no_recal,kind=conformal,alpha=0.05,level=1.0 ;; 0.99984 ;; 0.0006 --> | 1.000 (+0.950) <!-- check: phase4/headline.csv ;; far ;; method=baseline,corruption=jpeg_compression,setting=A_no_recal,kind=conformal,alpha=0.05,level=3.0 ;; 1 ;; 0.0006 --> | 1.000 (+0.950) <!-- check: phase4/headline.csv ;; far ;; method=baseline,corruption=jpeg_compression,setting=A_no_recal,kind=conformal,alpha=0.05,level=5.0 ;; 1 ;; 0.0006 --> | 0.046 (-0.004) <!-- check: phase4/headline.csv ;; far ;; method=baseline,corruption=jpeg_compression,setting=B_recal,kind=conformal,alpha=0.05,level=0.5 ;; 0.045714 ;; 0.0006 --> | 0.046 (-0.004) <!-- check: phase4/headline.csv ;; far ;; method=baseline,corruption=jpeg_compression,setting=B_recal,kind=conformal,alpha=0.05,level=5.0 ;; 0.045705 ;; 0.0006 --> | 1.00 / 0.18 | 0.08 |
| jpeg_compression | DINOv2-kNN | 0.409 (+0.359) <!-- check: phase4/headline.csv ;; far ;; method=dinov2_knn,corruption=jpeg_compression,setting=A_no_recal,kind=conformal,alpha=0.05,level=0.5 ;; 0.40912 ;; 0.0006 --> | 0.741 (+0.691) <!-- check: phase4/headline.csv ;; far ;; method=dinov2_knn,corruption=jpeg_compression,setting=A_no_recal,kind=conformal,alpha=0.05,level=1.0 ;; 0.74133 ;; 0.0006 --> | 0.967 (+0.917) <!-- check: phase4/headline.csv ;; far ;; method=dinov2_knn,corruption=jpeg_compression,setting=A_no_recal,kind=conformal,alpha=0.05,level=3.0 ;; 0.96725 ;; 0.0006 --> | 1.000 (+0.950) <!-- check: phase4/headline.csv ;; far ;; method=dinov2_knn,corruption=jpeg_compression,setting=A_no_recal,kind=conformal,alpha=0.05,level=5.0 ;; 1 ;; 0.0006 --> | 0.046 (-0.004) <!-- check: phase4/headline.csv ;; far ;; method=dinov2_knn,corruption=jpeg_compression,setting=B_recal,kind=conformal,alpha=0.05,level=0.5 ;; 0.045829 ;; 0.0006 --> | 0.045 (-0.005) <!-- check: phase4/headline.csv ;; far ;; method=dinov2_knn,corruption=jpeg_compression,setting=B_recal,kind=conformal,alpha=0.05,level=5.0 ;; 0.045342 ;; 0.0006 --> | 0.91 / 0.30 | 0.11 |
| jpeg_compression | PatchCore | 0.819 (+0.769) <!-- check: phase4/headline.csv ;; far ;; method=patchcore,corruption=jpeg_compression,setting=A_no_recal,kind=conformal,alpha=0.05,level=0.5 ;; 0.81944 ;; 0.0006 --> | 1.000 (+0.950) <!-- check: phase4/headline.csv ;; far ;; method=patchcore,corruption=jpeg_compression,setting=A_no_recal,kind=conformal,alpha=0.05,level=1.0 ;; 0.99993 ;; 0.0006 --> | 1.000 (+0.950) <!-- check: phase4/headline.csv ;; far ;; method=patchcore,corruption=jpeg_compression,setting=A_no_recal,kind=conformal,alpha=0.05,level=3.0 ;; 1 ;; 0.0006 --> | 1.000 (+0.950) <!-- check: phase4/headline.csv ;; far ;; method=patchcore,corruption=jpeg_compression,setting=A_no_recal,kind=conformal,alpha=0.05,level=5.0 ;; 1 ;; 0.0006 --> | 0.046 (-0.004) <!-- check: phase4/headline.csv ;; far ;; method=patchcore,corruption=jpeg_compression,setting=B_recal,kind=conformal,alpha=0.05,level=0.5 ;; 0.04564 ;; 0.0006 --> | 0.046 (-0.004) <!-- check: phase4/headline.csv ;; far ;; method=patchcore,corruption=jpeg_compression,setting=B_recal,kind=conformal,alpha=0.05,level=5.0 ;; 0.04608 ;; 0.0006 --> | 1.00 / 0.28 | 0.06 |

Clean reference (level 0, A = B): FAR 0.046 (excess −0.004) for all three methods (`headline.csv`, `corruption = none`).

Violation rate (share of partitions with FAR > α; `headline.csv`, column `violation_rate`), mean over methods, corruptions and levels, conformal α = 5%:

- A (no recalibration): 0.940; B (recalibrated): 0.404 (about 40% is expected for B: the guarantee is on the *average* FAR, not per partition).

## 4. Which method is most stable? (measured)

Stability = root-mean-square of (FAR − α) over the shifted conditions and the 12 classes (0 = target kept everywhere; smaller is better). We rank by *distance from the target*, not by spread across classes: a detector that fails the same way in every class has a small spread. Source: `stability.csv` (`rms_excess`, `rms_lo`, `rms_hi`, `class_spread`) and `paired_stability.csv`. `all` = 16 shifted conditions (levels 0.5, 1, 3, 5 × 4 corruptions); `1-3-5` = the 12 ImageNet-C conditions. Setting A.

| grid | threshold | α | method | RMS excess [95% CI] | class spread | rank |
|---|---|---|---|---|---|---|
| all | conformal | 0.01 | baseline | 0.870 [0.847, 0.891] <!-- check: phase4/stability.csv ;; rms_excess ;; grid=all,setting=A_no_recal,kind=conformal,alpha=0.01,method=baseline ;; 0.86976 ;; 0.0006 --> | 0.090 | 3 |
| all | conformal | 0.01 | DINOv2-kNN | 0.642 [0.592, 0.688] <!-- check: phase4/stability.csv ;; rms_excess ;; grid=all,setting=A_no_recal,kind=conformal,alpha=0.01,method=dinov2_knn ;; 0.64225 ;; 0.0006 --> | 0.180 | 1 |
| all | conformal | 0.01 | PatchCore | 0.775 [0.742, 0.808] <!-- check: phase4/stability.csv ;; rms_excess ;; grid=all,setting=A_no_recal,kind=conformal,alpha=0.01,method=patchcore ;; 0.77477 ;; 0.0006 --> | 0.139 | 2 |
| all | conformal | 0.05 | baseline | 0.863 [0.851, 0.877] <!-- check: phase4/stability.csv ;; rms_excess ;; grid=all,setting=A_no_recal,kind=conformal,alpha=0.05,method=baseline ;; 0.86346 ;; 0.0006 --> | 0.058 | 3 |
| all | conformal | 0.05 | DINOv2-kNN | 0.721 [0.681, 0.757] <!-- check: phase4/stability.csv ;; rms_excess ;; grid=all,setting=A_no_recal,kind=conformal,alpha=0.05,method=dinov2_knn ;; 0.72131 ;; 0.0006 --> | 0.155 | 1 |
| all | conformal | 0.05 | PatchCore | 0.833 [0.818, 0.849] <!-- check: phase4/stability.csv ;; rms_excess ;; grid=all,setting=A_no_recal,kind=conformal,alpha=0.05,method=patchcore ;; 0.83318 ;; 0.0006 --> | 0.079 | 2 |
| all | conformal | 0.1 | baseline | 0.826 [0.816, 0.839] <!-- check: phase4/stability.csv ;; rms_excess ;; grid=all,setting=A_no_recal,kind=conformal,alpha=0.1,method=baseline ;; 0.82649 ;; 0.0006 --> | 0.052 | 3 |
| all | conformal | 0.1 | DINOv2-kNN | 0.708 [0.673, 0.739] <!-- check: phase4/stability.csv ;; rms_excess ;; grid=all,setting=A_no_recal,kind=conformal,alpha=0.1,method=dinov2_knn ;; 0.70842 ;; 0.0006 --> | 0.136 | 1 |
| all | conformal | 0.1 | PatchCore | 0.812 [0.804, 0.822] <!-- check: phase4/stability.csv ;; rms_excess ;; grid=all,setting=A_no_recal,kind=conformal,alpha=0.1,method=patchcore ;; 0.8122 ;; 0.0006 --> | 0.050 | 2 |
| all | tolerance | 0.05 | baseline | 0.855 [0.839, 0.870] <!-- check: phase4/stability.csv ;; rms_excess ;; grid=all,setting=A_no_recal,kind=tolerance,alpha=0.05,method=baseline ;; 0.85482 ;; 0.0006 --> | 0.067 | 3 |
| all | tolerance | 0.05 | DINOv2-kNN | 0.695 [0.651, 0.733] <!-- check: phase4/stability.csv ;; rms_excess ;; grid=all,setting=A_no_recal,kind=tolerance,alpha=0.05,method=dinov2_knn ;; 0.69491 ;; 0.0006 --> | 0.162 | 1 |
| all | tolerance | 0.05 | PatchCore | 0.810 [0.790, 0.831] <!-- check: phase4/stability.csv ;; rms_excess ;; grid=all,setting=A_no_recal,kind=tolerance,alpha=0.05,method=patchcore ;; 0.81022 ;; 0.0006 --> | 0.099 | 2 |
| all | tolerance | 0.1 | baseline | 0.821 [0.810, 0.834] <!-- check: phase4/stability.csv ;; rms_excess ;; grid=all,setting=A_no_recal,kind=tolerance,alpha=0.1,method=baseline ;; 0.8211 ;; 0.0006 --> | 0.056 | 3 |
| all | tolerance | 0.1 | DINOv2-kNN | 0.695 [0.659, 0.727] <!-- check: phase4/stability.csv ;; rms_excess ;; grid=all,setting=A_no_recal,kind=tolerance,alpha=0.1,method=dinov2_knn ;; 0.69518 ;; 0.0006 --> | 0.146 | 1 |
| all | tolerance | 0.1 | PatchCore | 0.801 [0.791, 0.813] <!-- check: phase4/stability.csv ;; rms_excess ;; grid=all,setting=A_no_recal,kind=tolerance,alpha=0.1,method=patchcore ;; 0.80086 ;; 0.0006 --> | 0.061 | 2 |
| 1-3-5 | conformal | 0.01 | baseline | 0.936 [0.920, 0.950] <!-- check: phase4/stability.csv ;; rms_excess ;; grid=1-3-5,setting=A_no_recal,kind=conformal,alpha=0.01,method=baseline ;; 0.93588 ;; 0.0006 --> | 0.062 | 3 |
| 1-3-5 | conformal | 0.01 | DINOv2-kNN | 0.729 [0.676, 0.779] <!-- check: phase4/stability.csv ;; rms_excess ;; grid=1-3-5,setting=A_no_recal,kind=conformal,alpha=0.01,method=dinov2_knn ;; 0.72907 ;; 0.0006 --> | 0.201 | 1 |
| 1-3-5 | conformal | 0.01 | PatchCore | 0.863 [0.839, 0.887] <!-- check: phase4/stability.csv ;; rms_excess ;; grid=1-3-5,setting=A_no_recal,kind=conformal,alpha=0.01,method=patchcore ;; 0.86288 ;; 0.0006 --> | 0.115 | 2 |
| 1-3-5 | conformal | 0.05 | baseline | 0.919 [0.911, 0.930] <!-- check: phase4/stability.csv ;; rms_excess ;; grid=1-3-5,setting=A_no_recal,kind=conformal,alpha=0.05,method=baseline ;; 0.9194 ;; 0.0006 --> | 0.038 | 3 |
| 1-3-5 | conformal | 0.05 | DINOv2-kNN | 0.799 [0.766, 0.828] <!-- check: phase4/stability.csv ;; rms_excess ;; grid=1-3-5,setting=A_no_recal,kind=conformal,alpha=0.05,method=dinov2_knn ;; 0.799 ;; 0.0006 --> | 0.143 | 1 |
| 1-3-5 | conformal | 0.05 | PatchCore | 0.902 [0.891, 0.913] <!-- check: phase4/stability.csv ;; rms_excess ;; grid=1-3-5,setting=A_no_recal,kind=conformal,alpha=0.05,method=patchcore ;; 0.90237 ;; 0.0006 --> | 0.058 | 2 |
| 1-3-5 | conformal | 0.1 | baseline | 0.877 [0.869, 0.886] <!-- check: phase4/stability.csv ;; rms_excess ;; grid=1-3-5,setting=A_no_recal,kind=conformal,alpha=0.1,method=baseline ;; 0.87727 ;; 0.0006 --> | 0.031 | 3 |
| 1-3-5 | conformal | 0.1 | DINOv2-kNN | 0.778 [0.751, 0.802] <!-- check: phase4/stability.csv ;; rms_excess ;; grid=1-3-5,setting=A_no_recal,kind=conformal,alpha=0.1,method=dinov2_knn ;; 0.77841 ;; 0.0006 --> | 0.121 | 1 |
| 1-3-5 | conformal | 0.1 | PatchCore | 0.867 [0.860, 0.875] <!-- check: phase4/stability.csv ;; rms_excess ;; grid=1-3-5,setting=A_no_recal,kind=conformal,alpha=0.1,method=patchcore ;; 0.86674 ;; 0.0006 --> | 0.041 | 2 |
| 1-3-5 | tolerance | 0.05 | baseline | 0.913 [0.903, 0.923] <!-- check: phase4/stability.csv ;; rms_excess ;; grid=1-3-5,setting=A_no_recal,kind=tolerance,alpha=0.05,method=baseline ;; 0.91281 ;; 0.0006 --> | 0.045 | 3 |
| 1-3-5 | tolerance | 0.05 | DINOv2-kNN | 0.776 [0.738, 0.811] <!-- check: phase4/stability.csv ;; rms_excess ;; grid=1-3-5,setting=A_no_recal,kind=tolerance,alpha=0.05,method=dinov2_knn ;; 0.77587 ;; 0.0006 --> | 0.156 | 1 |
| 1-3-5 | tolerance | 0.05 | PatchCore | 0.887 [0.873, 0.900] <!-- check: phase4/stability.csv ;; rms_excess ;; grid=1-3-5,setting=A_no_recal,kind=tolerance,alpha=0.05,method=patchcore ;; 0.88718 ;; 0.0006 --> | 0.070 | 2 |
| 1-3-5 | tolerance | 0.1 | baseline | 0.873 [0.865, 0.883] <!-- check: phase4/stability.csv ;; rms_excess ;; grid=1-3-5,setting=A_no_recal,kind=tolerance,alpha=0.1,method=baseline ;; 0.8734 ;; 0.0006 --> | 0.035 | 3 |
| 1-3-5 | tolerance | 0.1 | DINOv2-kNN | 0.767 [0.738, 0.793] <!-- check: phase4/stability.csv ;; rms_excess ;; grid=1-3-5,setting=A_no_recal,kind=tolerance,alpha=0.1,method=dinov2_knn ;; 0.76737 ;; 0.0006 --> | 0.131 | 1 |
| 1-3-5 | tolerance | 0.1 | PatchCore | 0.861 [0.854, 0.869] <!-- check: phase4/stability.csv ;; rms_excess ;; grid=1-3-5,setting=A_no_recal,kind=tolerance,alpha=0.1,method=patchcore ;; 0.86073 ;; 0.0006 --> | 0.049 | 2 |

**Paired comparison** (`paired_stability.csv`): difference of RMS excess between two methods, with a class-bootstrap interval computed on the *same* resampled classes for both methods (negative = first method closer to the target). Conformal, α = 5%, Setting A:

| grid | method A | method B | difference [95% CI] | excludes 0 |
|---|---|---|---|---|
| all | DINOv2-kNN | PatchCore | -0.112 [-0.142, -0.085] <!-- check: phase4/paired_stability.csv ;; diff ;; grid=all,setting=A_no_recal,kind=conformal,alpha=0.05,method_a=dinov2_knn,method_b=patchcore ;; -0.11187 ;; 0.0006 --> | True |
| all | DINOv2-kNN | baseline | -0.142 [-0.179, -0.110] <!-- check: phase4/paired_stability.csv ;; diff ;; grid=all,setting=A_no_recal,kind=conformal,alpha=0.05,method_a=dinov2_knn,method_b=baseline ;; -0.14216 ;; 0.0006 --> | True |
| all | PatchCore | baseline | -0.030 [-0.044, -0.017] <!-- check: phase4/paired_stability.csv ;; diff ;; grid=all,setting=A_no_recal,kind=conformal,alpha=0.05,method_a=patchcore,method_b=baseline ;; -0.030286 ;; 0.0006 --> | True |
| 1-3-5 | DINOv2-kNN | PatchCore | -0.103 [-0.137, -0.076] <!-- check: phase4/paired_stability.csv ;; diff ;; grid=1-3-5,setting=A_no_recal,kind=conformal,alpha=0.05,method_a=dinov2_knn,method_b=patchcore ;; -0.10337 ;; 0.0006 --> | True |
| 1-3-5 | DINOv2-kNN | baseline | -0.120 [-0.151, -0.095] <!-- check: phase4/paired_stability.csv ;; diff ;; grid=1-3-5,setting=A_no_recal,kind=conformal,alpha=0.05,method_a=dinov2_knn,method_b=baseline ;; -0.1204 ;; 0.0006 --> | True |
| 1-3-5 | PatchCore | baseline | -0.017 [-0.031, -0.006] <!-- check: phase4/paired_stability.csv ;; diff ;; grid=1-3-5,setting=A_no_recal,kind=conformal,alpha=0.05,method_a=patchcore,method_b=baseline ;; -0.01703 ;; 0.0006 --> | True |

Over all (grid, threshold, α) cells in Setting A the paired interval excludes 0 in 30 of 30. In Setting B the largest |difference| is 0.00059, so the methods are tied in practice (the few cells whose interval excludes 0 differ by less than 0.0006 of RMS excess).

## 5. Per-class check at the mild level (measured)

Setting A, conformal α = 5%, level 0.5; source `per_class.csv` (`far`, `severity = -1`). The mean can hide a class.

| corruption | method | mean FAR | min class | max class | classes with FAR > 10% |
|---|---|---|---|---|---|
| brightness | baseline | 0.183 | 0.040 | 0.826 | 6 / 12 |
| brightness | DINOv2-kNN | 0.053 | 0.041 | 0.084 | 0 / 12 |
| brightness | PatchCore | 0.095 | 0.049 | 0.340 | 2 / 12 |
| gaussian_blur | baseline | 0.142 | 0.060 | 0.390 | 6 / 12 |
| gaussian_blur | DINOv2-kNN | 0.051 | 0.041 | 0.067 | 0 / 12 |
| gaussian_blur | PatchCore | 0.058 | 0.048 | 0.072 | 0 / 12 |
| gaussian_noise | baseline | 0.975 | 0.749 | 1.000 | 12 / 12 |
| gaussian_noise | DINOv2-kNN | 0.582 | 0.063 | 1.000 | 10 / 12 |
| gaussian_noise | PatchCore | 0.847 | 0.372 | 1.000 | 12 / 12 |
| jpeg_compression | baseline | 0.966 | 0.791 | 1.000 | 12 / 12 |
| jpeg_compression | DINOv2-kNN | 0.409 | 0.055 | 0.917 | 9 / 12 |
| jpeg_compression | PatchCore | 0.819 | 0.304 | 1.000 | 12 / 12 |

## 6. Recalibration with only m corrupted normals (Phase 4b, measured; levels 1/3/5 only)

Mean over methods and 12 classes. 500 draws per experiment and condition (100 partitions × 5 random m-subsets; the 5 draws share a test half, so they are correlated). `law` = exact Beta-binomial value. Source: `small_m_headline.csv` (`far`, `law_far`, `violation_rate`, `law_violation_rate`, `tpr`). Not rerun with the mild level.

| threshold | α | m | mean FAR | law | P(FAR>α) | law | TPR |
|---|---|---|---|---|---|---|---|
| conformal | 0.01 | 100 | 0.0099 <!-- check: phase4/small_m_headline.csv ;; far ;; method=patchcore,kind=conformal,alpha=0.01,m=100 ;; 0.0099321 ;; 0.0006 --> | 0.0099 | 0.373 | 0.373 | 0.13 |
| conformal | 0.05 | 25 | 0.0384 <!-- check: phase4/small_m_headline.csv ;; far ;; method=patchcore,kind=conformal,alpha=0.05,m=25 ;; 0.038454 ;; 0.0006 --> | 0.0385 | 0.292 | 0.292 | 0.20 |
| conformal | 0.05 | 50 | 0.0392 <!-- check: phase4/small_m_headline.csv ;; far ;; method=patchcore,kind=conformal,alpha=0.05,m=50 ;; 0.03932 ;; 0.0006 --> | 0.0392 | 0.305 | 0.305 | 0.21 |
| conformal | 0.05 | 100 | 0.0494 <!-- check: phase4/small_m_headline.csv ;; far ;; method=patchcore,kind=conformal,alpha=0.05,m=100 ;; 0.049629 ;; 0.0006 --> | 0.0495 | 0.456 | 0.456 | 0.23 |
| conformal | 0.1 | 10 | 0.0909 <!-- check: phase4/small_m_headline.csv ;; far ;; method=patchcore,kind=conformal,alpha=0.1,m=10 ;; 0.091152 ;; 0.0006 --> | 0.0909 | 0.352 | 0.352 | 0.27 |
| conformal | 0.1 | 25 | 0.0768 <!-- check: phase4/small_m_headline.csv ;; far ;; method=patchcore,kind=conformal,alpha=0.1,m=25 ;; 0.076877 ;; 0.0006 --> | 0.0769 | 0.282 | 0.282 | 0.27 |
| conformal | 0.1 | 50 | 0.0980 <!-- check: phase4/small_m_headline.csv ;; far ;; method=patchcore,kind=conformal,alpha=0.1,m=50 ;; 0.098078 ;; 0.0006 --> | 0.0980 | 0.439 | 0.439 | 0.30 |
| conformal | 0.1 | 100 | 0.0989 <!-- check: phase4/small_m_headline.csv ;; far ;; method=patchcore,kind=conformal,alpha=0.1,m=100 ;; 0.099041 ;; 0.0006 --> | 0.0990 | 0.459 | 0.461 | 0.30 |
| tolerance | 0.05 | 100 | 0.0198 <!-- check: phase4/small_m_headline.csv ;; far ;; method=patchcore,kind=tolerance,alpha=0.05,m=100 ;; 0.019862 ;; 0.0006 --> | 0.0198 | 0.067 | 0.066 | 0.17 |
| tolerance | 0.1 | 50 | 0.0392 <!-- check: phase4/small_m_headline.csv ;; far ;; method=patchcore,kind=tolerance,alpha=0.1,m=50 ;; 0.03932 ;; 0.0006 --> | 0.0392 | 0.047 | 0.047 | 0.21 |
| tolerance | 0.1 | 100 | 0.0494 <!-- check: phase4/small_m_headline.csv ;; far ;; method=patchcore,kind=tolerance,alpha=0.1,m=100 ;; 0.049629 ;; 0.0006 --> | 0.0495 | 0.047 | 0.048 | 0.23 |

Not certifiable (explicit status, never clamped): conformal α = 1% needs m ≥ 99, α = 5% needs m ≥ 19; the tolerance limit (δ = 5%) needs m ≥ 29 (α = 10%), 59 (α = 5%), 299 (α = 1%).

## 7. Cross-split check on clean data (Phase 3, measured)

Calibrate on held-out *train* normals, test on VisA's *official test* normals. Source: `results/phase3/cross_split_summary.csv` (`far_pooled`, `far_expected`). Wilson intervals are too narrow (3 seeds reuse the same images).

| method | α | FAR | expected |
|---|---|---|---|
| baseline | 0.01 | 0.0097 | 0.0088 |
| baseline | 0.05 | 0.0485 | 0.0486 |
| baseline | 0.1 | 0.0974 | 0.0990 |
| DINOv2-kNN | 0.01 | 0.0069 | 0.0088 |
| DINOv2-kNN | 0.05 | 0.0561 | 0.0486 |
| DINOv2-kNN | 0.1 | 0.1137 | 0.0990 |
| PatchCore | 0.01 | 0.0097 | 0.0088 |
| PatchCore | 0.05 | 0.0502 | 0.0486 |
| PatchCore | 0.1 | 0.1091 | 0.0990 |

DINOv2-kNN and PatchCore are 1–1.5 points above the expected FAR at α = 10%; a weak hint that VisA's test normals score slightly higher than its train normals. Not conclusive.

## 8. Estimated numbers (not results)

- Laptop run time from `scripts/estimate_full_run.py`: full run predicted 3 h 44 min (warm-up) and measured 3 h 53 min; extra mild level predicted 1 h 04 min and measured 1 h 05 min (`results/provenance/run_meta.json`, field `elapsed_seconds`). The cost model was fitted on the full run, so the second comparison is not independent.
- File sizes after adding the mild level: predicted +31% rows (row ratio 110,381 / 84,409); measured file size 0.80 MB per file (vs 0.64 MB before).
- Weak PatchCore classes (macaroni2, capsules) are said to be hard in the literature: unverified.

## 9. What we did not test

- Real distribution shift (a new product lot, camera, lighting rig). VisA has no engineered shift; all shifts here are synthetic and strong.
- Why noise std 0.04 or JPEG quality 50 already break a clean-calibrated threshold; no causal claim.
- Why DINOv2-kNN is 5.7 AUROC points below the AnomalyDINO reference at 224 px (hypotheses only: input size, aggregation, resizing, masking).
- Why PatchCore and DINOv2-kNN drift differently (DINOv2-kNN is closest to the target; we did not test the mechanism).
- Other datasets (MVTec AD 2, AeBAD-S), other detectors, other corruption strengths or families.
- Uncertainty from sampling images *within* a class: the class bootstrap does not cover it (138–277 test normals per class; the 100 partitions reuse them).
- Small-m recalibration together with the mild level (Phase 4b was computed on levels 1/3/5).
- A mismatched calibration/test severity (calibrate at one level, test at another), skipped by decision.
- Feature-level shift correction (SPARC-style) combined with conformal calibration.
- Whether the 12 VisA classes are representative of a real factory line.
