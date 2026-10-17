# Prior work

This project is a **reproduction plus extension**, not a novelty claim. This file records
what each piece of prior work contributes, what we reuse from it, and where we differ.

## 1. Closest prior work (read in full by the author)

The facts in this section come from the author reading the PDFs (Deng in full; SPARC
pages 1–34) after arxiv.org turned out to be unreachable from the cloud environment.
Nothing else in this repository relies on these papers.

### 1.1 Deng, *Distribution-free false-alarm calibration and chance-corrected spatial evaluation for industrial anomaly detection*, arXiv:2608.15090 (2026), Tongji University

- **Threshold:** a one-sided order-statistic **tolerance limit**. With m calibration
  normals, take the least conservative rank r (i.e. the *smallest* r) such that
  Pr{Binomial(m, α) ≥ m − r + 1} ≥ 1 − δ, and set τ = S₍ᵣ₎.
- **Guarantee:** Pr_cal[ P₀{S > τ} ≤ α ] ≥ 1 − δ. This is training-conditional (PAC),
  holds with confidence 1 − δ over the calibration draw, and stays valid with ties.
- **Primary setting:** m = 150, α = 0.10, δ = 0.05, so r = 142 with achieved confidence
  0.9693. *We reproduced r = 142 and 0.9693 from the formula above (scipy, 2026-10-17).*
- **Sample size:** m_min = ⌈log δ / log(1 − α)⌉, e.g. 299 normals for α = 1%, δ = 5%.
- **Datasets and detectors:** ISP-AD (3 modalities) and Magnetic Tile Defect; exact
  PatchCore (WRN50), DINOv2 patch memory, and a PaDiM-style detector.
- **Findings:** observed FPR of 4.8–6.8% on ISP-AD (conservative relative to the
  10% target). One ViT-B/16 seed on Magnetic Tile reached 12.3%, above the 10% target.
- **Realised FPR** is reported with Wilson intervals.
- **Scope:** the paper states that its guarantee does not extend to batch dependence or
  domain shift, and it **does not test shift**. Its guarantee differs from the
  *marginal* guarantee of split conformal (E[FAR] ≤ α). No code was released.
- **How we use it:** our PAC threshold **is Deng's tolerance limit**, and we credit it
  as such (the underlying theory is Vovk, 2012). We report Wilson intervals as Deng
  does. The part Deng does not do, and we do, is measure how both kinds of threshold
  behave under controlled shift, and how much recalibration restores them.

### 1.2 Han, Chu, Nowak, Chin, *SPARC: Subspace Position-Aware Robust Few-Shot Calibration for Distribution-Shifted Industrial Anomaly Detection*, arXiv:2608.18585 (2026)

- **Method:** a gradient-free, per-cell subspace projection (rank k − 1, from k ≤ 8
  verified-normal images of the *deployment lot*). It sits between a frozen encoder and
  the detector. "Calibration" here means *feature correction*, not threshold calibration.
- **Benchmarks:** MVTec AD 2 and AeBAD-S (shift-prone); **VisA and RAD are classified as
  shift-free** by their definition.
- **Metrics:** image AUROC and AU-PRO only. The paper contains **no threshold calibration
  and no false-alarm-rate guarantee anywhere**.
- **Findings relevant to us:** calibrating with training-pool normals is inert (their
  Table 20); only deployment-lot normals help. Contamination of the calibration set
  with anomalies removes the gain (Table 14).
- **Reference numbers on VisA** (their protocol: resize + center crop; Table 1): image
  AUROC of about **94.3 for PatchCore WRN-50** and about **95.3 for AnomalyDINO**. We use
  these **only as a sanity check** for our PatchCore and DINOv2-kNN under the `official`
  protocol, not as a claim. Our preprocessing differs (square resize to 256 px, no
  crop), so small gaps are expected, and large gaps will be flagged and investigated.
- **How we relate:** SPARC fixes the *features* under shift; we measure what shift does
  to the *threshold* and its false-alarm guarantee. Combining the two is listed as
  future work in `ROADMAP.md`, not in scope.

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
| Vovk. *Conditional Validity of Inductive Conformal Predictors.* ACML 2012. | Theory of **training-conditional (PAC)** validity: P(FAR ≤ α) ≥ 1 − δ over the calibration draw, not just E[FAR] ≤ α. The operational tolerance-limit form we implement is Deng's (§1.1). |
| Angelopoulos, Bates. *A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification.* arXiv:2107.07511 (2021; Found. Trends ML 2023). | Notation and the Beta distribution of the realised coverage, which we use to check Phase 3 results. |
| Laxhammar, Falkman. *Inductive conformal anomaly detection for sequential detection of anomalous sub-trajectories.* Annals of Mathematics and Artificial Intelligence, 2015. | Conformal anomaly detection: the anomaly score is a nonconformity measure, and the conformal p-value thresholded at α gives a false-alarm rate ≤ α. |
| Bates, Candès, Lei, Romano, Sesia. *Testing for outliers with conformal p-values.* Annals of Statistics 51(1), 2023. arXiv:2104.08279. | Rigorous treatment of conformal p-values for outlier detection, including their marginal vs. calibration-conditional validity. |
| Wilson. *Probable inference, the law of succession, and statistical inference.* JASA 22(158), 1927. | The Wilson score interval, which we use for the empirical FAR (a binomial proportion that is often close to 0, where the normal-approximation interval fails). |

## 5. Distribution shift

| Reference | Use |
|---|---|
| Hendrycks, Dietterich. *Benchmarking Neural Network Robustness to Common Corruptions and Perturbations* (ImageNet-C). ICLR 2019. | Corruption families and the severity-level convention we follow (brightness, blur, Gaussian noise, JPEG). |
| Tibshirani, Barber, Candès, Ramdas. *Conformal Prediction Under Covariate Shift.* NeurIPS 2019. | Why the guarantee breaks under shift (exchangeability is lost) and one fix (likelihood-ratio weighting). Discussed; not implemented by default. |
| Barber, Candès, Ramdas, Tibshirani. *Conformal prediction beyond exchangeability.* Annals of Statistics 2023. | Bounds the coverage gap under shift in terms of the total-variation distance between calibration and test data, which motivates measuring the drift empirically. |
| Gibbs, Candès. *Adaptive Conformal Inference Under Distribution Shift.* NeurIPS 2021. | An online alternative to our offline "recalibrate on m shifted normals" mitigation. Discussed in the limitations. |

## What is ours (the extension)

Positioning: **reproduction plus extension, no novelty claim.**

- **Reproduced:** split-conformal (marginal) and tolerance-limit (PAC, Deng 2026)
  false-alarm thresholds, applied to three detectors on all 12 VisA classes, with
  repeated random calibration/test partitions, Beta-law checks and Wilson intervals.
- **Extended:** a controlled, deterministic synthetic shift grid (4 corruption families
  × configurable severities), scored once on a GPU and analysed on CPU. We measure:
  - how far the realised FAR of both thresholds drifts from α, per (method, class,
    corruption, severity);
  - how much recalibration on a small set of shifted normals restores the target,
    including the sizes where it provably cannot (see the derived limits in
    `INTERVIEW_NOTES.md`: 1/(n+1) for split conformal, m_min for the tolerance limit).
- **Not claimed:** anything about natural shift. VisA contains no engineered shift
  (SPARC classifies it as shift-free), and our shifts are synthetic.
