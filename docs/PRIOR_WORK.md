# Prior work

This project is a **reproduction plus extension**, not a novelty claim. This file records
what each piece of prior work contributes, what we reuse from it, and where we differ.
Entries marked **not read** are cited for completeness only. We describe nothing about
their contents beyond the title, because we could not access the text.

## 1. Directly related: calibrated false alarms in industrial anomaly detection

| Reference | Status | Relation to this project |
|---|---|---|
| *Distribution-free false-alarm calibration and chance-corrected spatial evaluation for industrial anomaly detection*, arXiv:2608.15090 (2026). | **Not read.** arxiv.org was unreachable from the development environment (network policy) when this file was written (2026-10-17). Only the title was confirmed, via a web-search result. | Closest prior work by title: distribution-free false-alarm calibration for industrial AD. We treat our calibration part as a reproduction of this *idea* on VisA. Before the final README, the author should read the paper on the laptop and update this row (what it does, how we differ). |
| *SPARC: Subspace Position-Aware Robust Few-Shot Calibration for Distribution-Shifted Industrial Anomaly Detection*, arXiv:2608.18585 (2026). | **Not read.** Found in the same web search; same access problem. | By title, it addresses calibration under distribution shift in industrial AD, which overlaps with our Phase 4. It must be checked before we make any claim about the recalibration study. |

## 2. Anomaly detection methods we benchmark

| Reference | What it gives us |
|---|---|
| Roth, Pemula, Zepeda, Schölkopf, Brox, Gehler. *Towards Total Recall in Industrial Anomaly Detection* (PatchCore). CVPR 2022. arXiv:2106.08265. | Patch features from mid-level layers (layer2 + layer3) of an ImageNet WideResNet-50, local neighbourhood aggregation, a memory bank reduced by greedy (k-center) coreset subsampling with a random projection, and nearest-neighbour patch distance as the anomaly score (with a reweighting step). We re-implement a lean version in torch (no faiss); the design choices go in `docs/` in Phase 2. |
| Defard, Setkov, Loesch, Audigier. *PaDiM: a Patch Distribution Modeling Framework for Anomaly Detection and Localization.* ICPR Workshops 2021. arXiv:2011.08785. | A per-position Gaussian over patch features, scored with the Mahalanobis distance. Optional fourth method, if time permits. |
| Bergman, Cohen, Hoshen. *Deep Nearest Neighbor Anomaly Detection.* arXiv:2002.10445 (2020). | Justifies our baseline: kNN distance on global pretrained features is a strong, simple detector. |
| Cohen, Hoshen. *Sub-Image Anomaly Detection with Deep Pyramid Correspondences* (SPADE). arXiv:2005.02357 (2020). | kNN on pixel-level / patch features for localisation, a precursor of PatchCore. |
| Rippel, Mertens, Merhof. *Modeling the Distribution of Normal Data in Pre-Trained Deep Features for Anomaly Detection.* ICPR 2020. | Distance to a fitted normal distribution of global features (the "mean normal feature" variant of the baseline). |
| Oquab et al. *DINOv2: Learning Robust Visual Features without Supervision.* TMLR 2024. arXiv:2304.07193. | Self-supervised ViT backbone; we use its patch tokens with a kNN score. |
| Damm, Laszkiewicz, Lederer, Fischer. *AnomalyDINO: Boosting Patch-based Few-shot Anomaly Detection with DINOv2.* WACV 2025. arXiv:2405.14529. | Shows that DINOv2 patch features + kNN are a competitive detector; it is the closest prior for our third method. |
| Akcay et al. *Anomalib: A Deep Learning Library for Anomaly Detection.* ICIP 2022. arXiv:2202.08341. | Optional cross-check of our PatchCore only, not a core dependency. |

## 3. Dataset and metrics

| Reference | Use |
|---|---|
| Zou, Jeong, Pemula, Zhang, Dabeer. *SPot-the-Difference Self-Supervised Pre-training for Anomaly Detection and Segmentation.* ECCV 2022. arXiv:2207.14315. | Introduces **VisA** (CC BY 4.0) and its official `1cls` split. |
| Bergmann, Fauser, Sattlegger, Steger. *MVTec AD: A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection.* CVPR 2019. | The standard benchmark that VisA complements; protocol conventions. |
| Bergmann, Fauser, Sattlegger, Steger. *Uninformed Students: Student-Teacher Anomaly Detection with Discriminative Latent Embeddings.* CVPR 2020. | Defines the **PRO** (per-region overlap) metric, which we integrate up to FPR = 0.3. |

## 4. Conformal prediction and conformal anomaly detection

| Reference | What we take |
|---|---|
| Vovk, Gammerman, Shafer. *Algorithmic Learning in a Random World.* Springer 2005 (2nd ed. 2022). | Foundations: exchangeability, conformal p-values, validity. |
| Papadopoulos, Proedrou, Vovk, Gammerman. *Inductive Confidence Machines for Regression.* ECML 2002. | Split (inductive) conformal: calibrate on a held-out set, so the model is trained only once. |
| Lei, G'Sell, Rinaldo, Tibshirani, Wasserman. *Distribution-Free Predictive Inference for Regression.* JASA 2018. | Modern treatment of split conformal and its finite-sample coverage bounds. |
| Vovk. *Conditional Validity of Inductive Conformal Predictors.* ACML 2012. | **Training-conditional (PAC)** validity: P(FAR ≤ α) ≥ 1 − δ over the calibration draw, not just E[FAR] ≤ α. This gives the sample-size requirement n ≥ ln δ / ln(1 − α). |
| Angelopoulos, Bates. *A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification.* arXiv:2107.07511 (2021; Found. Trends ML 2023). | Notation and the Beta distribution of the realised coverage, which we use to check Phase 3 results. |
| Laxhammar, Falkman. *Inductive conformal anomaly detection for sequential detection of anomalous sub-trajectories.* Annals of Mathematics and Artificial Intelligence, 2015. | Conformal anomaly detection: the anomaly score is a nonconformity measure, and the conformal p-value thresholded at α gives a false-alarm rate ≤ α. |
| Bates, Candès, Lei, Romano, Sesia. *Testing for outliers with conformal p-values.* Annals of Statistics 51(1), 2023. arXiv:2104.08279. | Rigorous treatment of conformal p-values for outlier detection, including their marginal vs. calibration-conditional validity. |

## 5. Distribution shift

| Reference | Use |
|---|---|
| Hendrycks, Dietterich. *Benchmarking Neural Network Robustness to Common Corruptions and Perturbations* (ImageNet-C). ICLR 2019. | Corruption families and the severity-level convention we follow (brightness, blur, Gaussian noise, JPEG). |
| Tibshirani, Barber, Candès, Ramdas. *Conformal Prediction Under Covariate Shift.* NeurIPS 2019. | Why the guarantee breaks under shift (exchangeability is lost) and one fix (likelihood-ratio weighting). Discussed; not implemented by default. |
| Barber, Candès, Ramdas, Tibshirani. *Conformal prediction beyond exchangeability.* Annals of Statistics 2023. | Bounds the coverage gap under shift in terms of the total-variation distance between calibration and test data, which motivates measuring the drift empirically. |
| Gibbs, Candès. *Adaptive Conformal Inference Under Distribution Shift.* NeurIPS 2021. | An online alternative to our offline "recalibrate on m shifted normals" mitigation. Discussed in the limitations. |

## What is ours (the extension)

- The same protocol applied across three detectors on all 12 VisA classes, with
  repeated random calibration/test partitions and Beta-distribution checks.
- A controlled, deterministic corruption grid (4 families × configurable severities),
  scored once on a GPU and analysed entirely on CPU.
- Measurement of FAR drift per (method, class, corruption, severity), and of how many
  shifted normals recalibration needs to restore the guarantee, including the cases
  where it cannot (n < 99 at α = 1% for the marginal guarantee).

Whether any of this goes beyond arXiv:2608.15090 and arXiv:2608.18585 cannot be judged
until those papers are read. Until then the README makes **no novelty claim**.
