# Models: design choices (Phase 2)

Three detectors share one interface (`src/anomcal/scorers/`): fit on clean memory-bank
images of one class, then score uint8 images at the canonical 256 px resolution.
Every setting below is a default in `src/anomcal/scorers/settings.py`; any of them can be
overridden per run (`--opt patchcore.coreset_ratio=0.05`), and the resolved settings are
stored in `methods.json` (a run refuses to resume if they change).

Shared preprocessing: canonical 256 × 256 uint8 → (on the GPU) antialiased bilinear
resize to the model input size → ImageNet mean/std normalisation. **No center crop**: we
keep the whole image, so maps align with the masks, and defects near the border are not
cut away. SPARC's reference numbers use resize + center crop, which is one reason a small
AUROC gap is expected.

## 1. Baseline: global features + kNN

| choice | value | why |
|---|---|---|
| backbone | torchvision WRN-50-2, `IMAGENET1K_V1` | same backbone as PatchCore, so the comparison isolates *global vs patch* features |
| feature | global average of `layer4` (2048-d) | one vector per image (Bergman et al., 2020) |
| input size | 224 px | standard ImageNet resolution |
| score | mean Euclidean distance to the k = 1 nearest memory image | simplest kNN detector |
| precision | distances in float64 on the CPU | bank ≤ 905 vectors: cheap and exact |
| maps | none | a global vector has no spatial resolution |

## 2. Our lean PatchCore (Roth et al., CVPR 2022)

| choice | value | why |
|---|---|---|
| backbone | WRN-50-2, ImageNet `IMAGENET1K_V1` | as in the paper |
| layers | `layer2` (28 × 28, 512 ch) + `layer3` (14 × 14, 1024 ch) at 224 px | mid-level features: local enough for small defects, semantic enough to ignore pixel noise (paper's choice) |
| local aggregation | 3 × 3 average pooling, stride 1 (p = 3) | each patch feature also describes its neighbourhood (paper's p = 3) |
| merge | `layer3` bilinearly upsampled to 28 × 28, concatenated (1536 ch) | one feature per layer2 position |
| embedding | channel-wise adaptive average pooling 1536 → 1024 | keeps memory and kNN cost bounded. *Simplification*: the paper pools each layer separately first |
| coreset | greedy k-center, **10%** of memory patches, seeded Gaussian random projection to 128-d | the paper reports little AUROC loss at 10% (and at 1%); 10% is our default, 1–5% are listed as reductions |
| kNN | exact nearest neighbour: float32 shortlist (top-8) + exact re-rank | the **float32 cancellation lesson** from Phase 1 (see below) |
| image score | max over the 784 patch distances | one anomalous patch flags the image (paper; *we omit its re-weighting step*) |
| map | 28 × 28 distances → bilinear 256 × 256 → Gaussian σ = 4 px | paper's smoothing |
| amp | float16 autocast for the backbone on CUDA only; distances in float32 | halves activation memory; features cast back to float32 before any distance |

### The float32 lesson, applied to PatchCore

`||q − b||² = ||q||² − 2q·b + ||b||²` is one fast matrix product, but it subtracts large,
nearly equal numbers. In Phase 1, a memory image scored **0.71 against itself** instead of
0 with this trick in float32. PatchCore's anomaly score is exactly such a small nearest
distance, so we use the expansion only to *shortlist* the 8 nearest candidates per patch,
then recompute those 8 distances directly as `sum((q − b)²)` (no cancellation).

Tests: a memory patch scores **exactly 0** against a full bank; the result matches a
float64 brute force on data with a large common offset; and a test shows the naive
expansion failing on the same data.

The greedy coreset *does* use the expansion (one matrix-vector product per iteration,
about 11× faster on CPU than materialising differences). That is safe because it compares
large distances (it looks for the *farthest* point). A test checks that it selects exactly
the same set as the exact version.

## 3. DINOv2-kNN (in the spirit of AnomalyDINO, Damm et al., WACV 2025)

| choice | value | why |
|---|---|---|
| backbone | DINOv2 ViT-S/14 (`vit_small_patch14_dinov2.lvd142m`, timm) | self-supervised features that transfer well to AD; S/14 fits 6 GB easily |
| input size | 224 px → 16 × 16 patch tokens (CLS dropped), 384-d | must be divisible by 14; timm resamples the position embeddings (trained at 518 px) |
| distance | cosine distance on L2-normalised tokens, computed as `||a−b||²/2` | lets us reuse the exact kNN (for unit vectors, 1 − cos = ||a−b||²/2) |
| memory bank | all memory patches (no coreset) | 905 × 256 × 384 floats = 339 MB at most |
| image score | mean of the top 1% patch distances (≥ 1 patch; 3 of 256 at 224 px) | intended to follow AnomalyDINO's aggregation, but **unverified**: we could not read the paper from our environment |
| map | 16 × 16 → 256 × 256, Gaussian σ = 4 | as PatchCore |
| not done | AnomalyDINO's zero-shot masking; higher input resolution (the resolution AnomalyDINO/SPARC used is **unverified** by us) | listed as limitations / possible increases |

## 4. Memory budget for 6 GB VRAM (arithmetic from tensor shapes, not measured)

Largest class in each protocol; float32 unless noted. CUDA's own context (a few hundred
MB) comes on top; the laptop run measures the real peak (`peak_vram_mb` in `run_meta.json`).

| item | PatchCore calib (452 imgs) | PatchCore official (905 imgs) | DINOv2 official (905 imgs) |
|---|---:|---:|---:|
| model weights | 263 MB (WRN-50) | 263 MB | 82 MB (ViT-S) |
| all memory patches (**CPU RAM**) | 1.4 GB | **2.8 GB** | 339 MB |
| coreset projection (GPU) | 173 MB | 346 MB | – |
| memory bank (GPU) | 138 MB | 277 MB | 339 MB |
| one distance block (`chunk_elems` = 2²⁶) | 256 MB (a few copies in flight) | 256 MB | 256 MB |

So the defaults should stay around or below ~2 GB of VRAM. The tight resource is CPU
RAM (16 GB): PatchCore's 2.8 GB patch buffer is allocated once (no concatenation copy)
and freed after the coreset. The shared corrupted-image cache holds at most
654 images × 12 conditions × 196 KB ≈ 1.5 GB (`--no-corruption-cache` disables it).

If the GPU still runs out of memory, every scorer halves its batch and retries (logged as
`oom_backoffs`), down to one image. Further knobs: `--opt patchcore.chunk_elems=16777216`,
smaller `batch_size`, `coreset_ratio`.

## 5. What is tested offline (CI, no downloads)

- Tiny seeded random backbones with the same output structure (`tiny_cnn`, `tiny_vit`):
  anomalies score higher on toy data, results are deterministic per seed, memory images
  score exactly 0, maps peak on the defect, OOM backoff gives identical scores, and
  invalid settings are rejected.
- The real architectures (WRN-50, DINOv2 ViT-S/14 via timm), built with **random**
  weights: output shapes 512 × 28 × 28, 1024 × 14 × 14, 2048, and 256 × 384 tokens.
- End-to-end extraction on fake VisA with all three methods: scores, pixel metrics,
  per-job records, shared corruptions, and the full-run estimator.

**Not testable in the cloud:** the pretrained weights (download.pytorch.org and
huggingface.co are blocked there) and real GPU timings. The laptop smoke test covers both.

## 6. Measured outcome and known limitation (after the full run)

Measured clean image AUROC on the official split (class mean, 3 seeds; full tables in
`docs/RESULTS_EXTRACTION.md`): baseline 80.4, DINOv2-kNN 89.6, PatchCore 90.0. Against
SPARC's VisA reference numbers (different preprocessing), PatchCore is 4.3 points lower
and DINOv2-kNN is 5.7 points lower than AnomalyDINO. **The DINOv2 gap is a known
limitation that we have not explained.** Input resolution (224 px), the top-1%
aggregation, square resizing vs center crop, and the missing masking are hypotheses
only. An optional, cheap one-class check at 448 px is described in
`docs/RESULTS_EXTRACTION.md`. Both patch methods are weak on macaroni2 and capsules
(PatchCore on macaroni2 is near chance).
