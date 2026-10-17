# Laptop GPU run (Phase 2): Windows, PowerShell

Three stages, each one only after the previous one looked fine:

1. **Now:** install CUDA-enabled PyTorch, check the GPU, run the GPU smoke test and the
   quick estimate. Send the output back.
2. **After review:** measure one full-size class (`cashew`, seed 0), then extrapolate the
   full run's time and peak VRAM, and choose reductions if needed.
3. **After approval:** the full run, which resumes from stage 2 (nothing is computed twice).

All commands run from the repository root, with the venv active (`.\.venv\Scripts\Activate.ps1`),
on the latest `main` (`git pull`).

---

## Stage 1a: check the GPU and the driver

You do **not** need to install the "CUDA Toolkit". The PyTorch wheels bring their own
CUDA libraries; Windows only needs a recent **NVIDIA driver**.

```powershell
nvidia-smi
```

In the output, check three things:

- the GPU name, e.g. `NVIDIA GeForce RTX 3060 Laptop GPU`;
- the memory, e.g. `0MiB / 6144MiB` (6144 MiB = 6 GB);
- **`CUDA Version: X.Y`** in the top right. This is the *newest* CUDA your driver supports.

If `nvidia-smi` is not found, or the CUDA version is below 12.6, install the latest
driver for your GPU from <https://www.nvidia.com/Download/index.aspx> (GeForce, RTX 30
Series (Notebooks), RTX 3060 Laptop GPU, Windows 11), reboot, and check again.

Pick the PyTorch build ("index") from that number:

| `CUDA Version` shown by nvidia-smi | index to use |
|---|---|
| 13.0 or higher | `cu130` |
| 12.6 – 12.9 | `cu126` |
| below 12.6 | update the driver first |

## Stage 1b: install PyTorch with CUDA in the existing venv

Pinned versions: **torch 2.14.1, torchvision 0.29.1, timm 1.0.30**. Our cloud tests ran
torch 2.14.1 together with numpy 2.5.4, the same numpy version as your venv. Windows
CUDA builds of torch 2.14 exist for cu126, cu130 and cu132.

```powershell
# replace cu126 by cu130 if nvidia-smi showed CUDA Version >= 13.0 (download ~2.5 GB)
pip install torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cu126
pip install timm==1.0.30
pip install -e ".[gpu,plots,dev]"
```

If pip says *"No matching distribution found"* for the pinned version, list what the
index has and use the newest 2.14.x:

```powershell
pip index versions torch --index-url https://download.pytorch.org/whl/cu126
```

**The 3-line check**: it must print `True`, your GPU name and about 6 GiB:

```powershell
python -c "import torch; print('torch', torch.__version__, '| CUDA available:', torch.cuda.is_available())"
python -c "import torch; p = torch.cuda.get_device_properties(0); print(p.name, '|', round(p.total_memory / 2**30, 2), 'GiB VRAM')"
python -c "import torch; x = torch.randn(4096, 4096, device='cuda'); print('GPU matmul OK:', bool(torch.isfinite(x @ x).all()))"
```

If the first line prints `+cpu` in the version or `False`, a CPU-only torch got installed:
`pip uninstall -y torch torchvision`, then repeat the install command above.

Then the test suite. The model tests now run too (on the CPU, with tiny random networks):

```powershell
python -m pytest -q
```

## Stage 1c: quick estimate and GPU smoke test

The first run downloads the pretrained weights once. They are cached **outside** the
repository: WRN-50 (~263 MB, from download.pytorch.org) in
`%USERPROFILE%\.cache\torch`, and DINOv2 ViT-S/14 (~88 MB, from huggingface.co) in
`%USERPROFILE%\.cache\huggingface`.

```powershell
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"   # silences a harmless Windows warning

# (1) warm-up estimate for the FULL run (scores nothing; ~1-2 min incl. downloads)
python scripts\extract_scores.py --config configs\full_run.json --estimate-only

# (2) GPU smoke test: class cashew, at most 20 images per role, seed 0, all 3 methods,
#     both protocols, all 13 conditions (a few minutes)
Remove-Item -Recurse -Force outputs\extract\smoke_max20 -ErrorAction SilentlyContinue
python scripts\extract_scores.py --config configs\gpu_smoke.json

# (3) quick AUROC on those 20 + 20 images: NOT a result, only a "features look sane" check
python scripts\auroc_table.py outputs\extract\smoke_max20\scores
```

What to expect: (2) prints a plan, a per-method warm-up line (model build, ms/img, warm-up
peak VRAM), a progress bar, and `done: ... chunks`. In (3), PatchCore and DINOv2-kNN should
be clearly above 50 on clean images. A value near 50 would point to a weight-loading or
preprocessing bug, and we would investigate before anything else.

**Send back:** the full console output of (1), (2) and (3), plus the output of
`nvidia-smi` and of the 3-line check.

---

## Stage 2 (after review): one full-size class, then the full-run estimate

```powershell
python scripts\extract_scores.py --config configs\gpu_one_class.json
python scripts\estimate_full_run.py --run-dir outputs\extract\full
```

The first command runs class `cashew` (one of the four smallest classes) with seed 0 at full
size. Its chunks are written into `outputs\extract\full\`, the same folder as the full run,
so they are **reused** later. The second command fits a cost model to the measured jobs and
prints the predicted time per method, plus the measured peak VRAM and a linear upper bound
for the largest memory bank. If the time is too long or the bound comes too close to 6 GB,
we choose reductions together (next section) before starting the full run.

### Possible reductions (you choose; each is one flag)

| knob | flag | effect |
|---|---|---|
| PatchCore coreset 10% → 5% or 1% | `--opt patchcore.coreset_ratio=0.01` | coreset time ~10× less at 1%, kNN ~10× less; the paper reports little AUROC loss |
| fewer severities | `--severities 2` (levels 1, 5) | 9 instead of 13 conditions |
| smaller model input | `--opt patchcore.image_size=196` | 25 × 25 instead of 28 × 28 patches (20% fewer; checked on the real WRN-50) |
| DINOv2 coreset | `--opt dinov2_knn.coreset_ratio=0.25` | smaller bank, faster kNN |
| smaller kNN blocks (if VRAM is tight) | `--opt patchcore.chunk_elems=16777216` | 4× smaller distance blocks |

Changing a method setting needs a new `--out` folder (the run refuses to mix settings).

## Stage 3 (after approval): the full run

```powershell
python scripts\extract_scores.py --config configs\full_run.json
```

- Keep the laptop plugged in and disable sleep for the run:
  `powercfg /change standby-timeout-ac 0` (undo afterwards: `powercfg /change standby-timeout-ac 30`).
- It is resumable: after Ctrl+C, a crash or a reboot, rerun the same command.
- At the end it prints the consolidated files in `outputs\extract\full\scores\`.

What gets committed (small files only; the repo guard rejects anything over 5 MiB):

```powershell
New-Item -ItemType Directory -Force results\scores | Out-Null
Copy-Item outputs\extract\full\scores\*.parquet results\scores\
Copy-Item outputs\extract\full\scores\pixel_metrics.csv results\scores\
Copy-Item outputs\extract\full\run_meta.json, outputs\extract\full\methods.json, outputs\extract\full\fingerprint.json results\scores\
```

## Stage (d): extra mild severity level (only the new conditions)

Why: noise and JPEG already give FAR ≈ 100% at level 1, so levels 1/3/5 show no graded
curve. One extra level per family, **milder than ImageNet-C level 1** and stored as
`severity = -1` (the other levels and all existing scores are untouched):

| family | level -1 | vs level 1 | why |
|---|---|---|---|
| brightness | +0.05 on HSV value | half of +0.1 | same halving as 0.1 → 0.2 |
| gaussian_blur | σ = 0.5 px | half of 1 px | a Gaussian below ~0.5 px is almost a no-op |
| gaussian_noise | std 0.04 | half of 0.08 | the 1 → 2 step is also +0.04 |
| jpeg_compression | quality 50 | 25 | 25…7 are all heavy; 50 is a common "mild" setting |

The JPEG value is the least certain (it may still hurt a lot); we only learn that from the run.

Work to do: 3 methods × 3 seeds × 12 classes × 4 families = **432 new chunks**, `calib`
protocol only. Clean and levels 1/3/5 already exist and are skipped. The memory bank of every
job is rebuilt (that is most of the cost). Config: `configs\extra_mild_level.json`.

```powershell
git pull
# 0. the old run must still be on disk: expect 1512 chunk files (1404 calib + 108 official; arithmetic)
(Get-ChildItem outputs\extract\full\chunks -Recurse -Filter *.parquet).Count

# 1. what will be done (prints the estimate, scores nothing)
python scripts\extract_scores.py --config configs\extra_mild_level.json --estimate-only

# 2. the run (resumable: after Ctrl+C or a crash, rerun the same command)
python scripts\extract_scores.py --config configs\extra_mild_level.json

# 3. check: expect 1944 chunk files and severity -1 in every calib file
(Get-ChildItem outputs\extract\full\chunks -Recurse -Filter *.parquet).Count
python -c "import pandas as pd; d=pd.read_parquet(r'outputs\extract\full\scores\calib_patchcore_seed0.parquet'); print(sorted(d.severity.unique()), len(d))"   # [-1, 0, 1, 3, 5] 110381

# 4. copy the small files back and push on a new branch
git checkout -b results/mild-level
Copy-Item outputs\extract\full\scores\calib_*.parquet results\scores\
Copy-Item outputs\extract\full\run_meta.json results\provenance\
python -m anomcal.repo_guard
git add results; git commit -m "results: extra mild severity level (-1) scores"; git push -u origin results/mild-level
```

Expected: the old 5.8 MB of scores grows by about 31% (rows 84,409 → 110,381 per calib file;
estimate, not measured), each calib file stays below 1 MB, far under the 5 MiB guard.
Prediction for the time: see ROADMAP "Phase 4c". Peak VRAM does not change (same models).

## Troubleshooting

| symptom | fix |
|---|---|
| `torch.cuda.is_available()` is `False` | a CPU wheel got installed, or the driver is too old: see stage 1b / 1a |
| `error: --device cuda requested but CUDA is not available` | same as above |
| `CUDA out of memory` despite backoff | close other GPU programs (browser tabs with video, games); try `--batch-size 8` or `--opt patchcore.chunk_elems=16777216` |
| download fails (proxy, firewall) | rerun later; weights are cached after the first success |
| `HTTP 401/403` from huggingface.co | not expected for DINOv2 (public). Check the network, or `pip install -U huggingface_hub` |
| `... settings changed for outputs\extract\full` | you changed a method option; use a new `--out` folder or remove the old one |
| very slow, GPU utilisation low | check `nvidia-smi` during the run; try `--num-workers 4` (only affects first-time image decoding) |
