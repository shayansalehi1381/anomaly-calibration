# Laptop smoke test (Phase 1): Windows, PowerShell

Goal: check that the real VisA files load correctly on the laptop and that the
extraction pipeline runs end to end. We use the **dummy** scorer (CPU, no torch, no GPU),
so the numbers it produces mean nothing. Nothing from this test is committed except,
optionally, the timing information you send back.

Expected total time: about 15–30 minutes, mostly extracting the tar (a guess, not a
measurement).

## 0. Prerequisites (once)

- **Git for Windows**: <https://git-scm.com/download/win>
- **Python 3.12, 64-bit**: <https://www.python.org/downloads/windows/>. Tick "Add
  python.exe to PATH" in the installer; the `py` launcher is installed by default.
- Windows 10/11 already ships `tar.exe`, which we use to extract VisA.

Check them in a new PowerShell window:

```powershell
git --version
py -3.12 --version
tar --version
```

## 1. Clone the repository

Pick a folder **without spaces and non-ASCII characters** in its path (e.g. `C:\work`):

```powershell
mkdir C:\work -Force
cd C:\work
git clone https://github.com/shayansalehi1381/anomaly-calibration.git
cd anomaly-calibration
git log --oneline -1          # should show the Phase 1 merge commit on main
```

(If you test **before** the Phase 1 PR is merged, also run
`git checkout claude/festive-cerf-e50k0a`.)

## 2. Create a virtual environment and install

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell says *"running scripts is disabled on this system"*, run this once and
then activate again:

```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
.\.venv\Scripts\Activate.ps1
```

Your prompt now starts with `(.venv)`. Install the CPU part. **No torch is needed for
Phase 1**; the GPU install comes in Phase 2.

```powershell
python -m pip install --upgrade pip
pip install -e ".[plots,dev]"
python -m pytest -q
```

Expected: all tests pass (the count is in the Phase 1 PR description).

## 3. Extract VisA into `data\visa\`

`data\` is git-ignored, and the repo guard in CI rejects any file under it, so nothing
from it can be committed by accident. Replace the tar path with where yours is:

```powershell
New-Item -ItemType Directory -Force data\visa | Out-Null
tar -xf "C:\Users\<you>\Downloads\VisA_20220922.tar" -C data\visa
Get-ChildItem data\visa
```

You should see the 12 class folders (`candle`, `capsules`, …, `pipe_fryum`), either
directly in `data\visa` or inside `data\visa\VisA`; both layouts are accepted.

**Split file.** If there is no `split_csv\1cls.csv` next to the class folders, download
the official one (it is small, about 660 KB):

```powershell
$root = if (Test-Path data\visa\VisA\candle) { "data\visa\VisA" } else { "data\visa" }
New-Item -ItemType Directory -Force "$root\split_csv" | Out-Null
Invoke-WebRequest https://raw.githubusercontent.com/amazon-science/spot-diff/main/split_csv/1cls.csv `
    -OutFile "$root\split_csv\1cls.csv"
```

## 4. Verify the dataset and print per-class counts

```powershell
python -m anomcal.data.visa --data-root data\visa
echo "exit code: $LASTEXITCODE"
```

Expected: a 12-row table with totals **8659 / 962 / 1200**, the lines
`counts match the verified official split` and
`all images and masks listed in the split exist on disk`, and `exit code: 0`.
If the SHA-256 line says `DIFFERS`, that alone is fine (it may just be line endings),
as long as the counts match.

## 5. Runtime estimate only (no scoring)

```powershell
python scripts\extract_scores.py --data-root data\visa --device cpu `
    --methods dummy --classes candle --max-images 20 --estimate-only
```

This prints the plan and an `estimated time: …` line, then exits without scoring.

## 6. The smoke test: one class, at most 20 images per role

```powershell
python scripts\extract_scores.py --data-root data\visa --device cpu `
    --methods dummy --classes candle --max-images 20
```

What happens: for 3 seeds × 2 protocols it fits the dummy scorer on ≤ 20 memory-bank
images and scores ≤ 20 held-out normals + ≤ 20 anomalies, under clean + 12 corrupted
conditions (calib) or clean only (official), with a progress bar. Outputs go to
`outputs\extract\smoke_max20\` (git-ignored):

```powershell
Get-ChildItem outputs\extract\smoke_max20\scores
```

You should see 6 files: `calib_dummy_seed{0,1,2}.parquet` and `official_dummy_seed{0,1,2}.parquet`.

## 7. Check resumability (about 1 minute)

1. Run the step 6 command again. It must print `nothing left to do (all chunks exist)`.
2. Delete the output and start again with 2 classes, then press **Ctrl+C** while the
   progress bar is running:

   ```powershell
   Remove-Item -Recurse -Force outputs\extract\smoke_max20
   python scripts\extract_scores.py --data-root data\visa --device cpu `
       --methods dummy --classes candle pcb1 --max-images 20
   ```

   It must print `interrupted: finished chunks are kept; rerun the same command to resume`.
3. Run the same command again. It must say how many chunks were *already done* and
   finish only the rest.

## 8. Full-dataset loading estimate (optional, about 1 minute)

Image decoding is probably the main CPU cost of the real run. This prints an estimate
for all 12 classes without scoring anything:

```powershell
python scripts\extract_scores.py --data-root data\visa --device cpu --methods dummy --estimate-only
```

## 9. What to send back

Copy into the chat:

1. the console output of steps 4, 5 and 6 (and 7 if anything looked wrong);
2. the `"estimate"` block of the last entry in
   `outputs\extract\smoke_max20\run_meta.json` (per-image load, corruption and score
   times), plus the `estimated time` line from step 8;
3. your CPU model, and whether the data is on an SSD or an HDD.

I'll use these to replace the rough laptop-time guesses in `ROADMAP.md` with measured
numbers. Do **not** commit anything under `outputs\` or `data\`.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `error: no VisA class folders ... found` | `--data-root` must point to the folder that contains `candle\` (or a `VisA\` folder containing it). |
| `error: ... 1cls.csv not found` | Do the split-file step in section 3, or pass `--split-csv <path>`. |
| `error: ... was started with {...}` | You changed `--max-images`, `--memory-fraction` or `--image-size` for an existing output folder. Use a new `--out`, or delete the old folder. |
| `error: --device cuda needs torch` | Expected in Phase 1. Use `--device cpu`. |
| Very slow loading | Try `--num-workers 4` (threads, which are safe on Windows), and keep the data on an SSD. |
