"""Score extraction: the only step that runs on the laptop (GPU from Phase 2 on).

Jobs run in the order class -> protocol -> seed -> method. For every job:

1. build the split (memory bank vs. held-out pool) and **assert it is leak-free**;
2. load memory-bank + evaluation images (each image is decoded once per class);
3. fit the scorer on the memory bank (clean images only);
4. for every condition (clean + each corruption x severity) not done yet: take the
   corrupted evaluation images (computed once per class/protocol/seed and shared by all
   methods), score them, and write one chunk file. On clean images, methods with
   anomaly maps also get pixel AUROC / PRO, computed here, because the maps are
   too big to keep.

Everything is resumable at chunk granularity (see :mod:`anomcal.scores_io`). Before the
real work, a short timed warm-up prints an estimated runtime for what is left; per-job
timings and peak VRAM go into ``run_meta.json`` (see :mod:`anomcal.estimate` for an
extrapolation from one full-size class to the whole run).

The command-line entry point is ``scripts/extract_scores.py`` (a thin wrapper around
:func:`main`); run it with ``--help`` for all options.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import platform
import subprocess
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from tqdm import tqdm

import anomcal
from anomcal.corruptions import (
    CORRUPTION_NAMES,
    DEFAULT_N_SEVERITIES,
    MAX_LEVEL,
    apply_corruption_batch,
    conditions,
    levels_for_count,
    severity_table,
)
from anomcal.data.images import DEFAULT_IMAGE_SIZE, load_images, load_mask
from anomcal.data.splits import (
    DEFAULT_MEMORY_FRACTION,
    PROTOCOLS,
    ROLE_ANOMALY,
    ROLE_HELDOUT,
    ClassSplit,
    assert_disjoint,
    make_split,
)
from anomcal.data.visa import (
    VISA_CLASSES,
    VisaRecord,
    find_split_csv,
    read_split_csv,
    resolve_visa_root,
    sha256_file,
)
from anomcal.metrics import pixel_auroc, pro_auc
from anomcal.scorers import AVAILABLE_METHODS, ScorerConfig, build_scorer, settings_dict
from anomcal.scores_io import (
    chunk_path,
    consolidate,
    consolidate_pixel_metrics,
    make_frame,
    pixel_metrics_path,
    write_atomic,
    write_json_atomic,
)
from anomcal.seeding import set_global_seed

# Settings that change the numbers in existing chunk files. If any of them differs from
# what an output folder was started with, resuming into it would silently mix two
# experiments, so we refuse. Methods, seeds, classes and severity levels can be added.
FINGERPRINT_KEYS = ("memory_fraction", "image_size", "max_images", "split_csv_sha256")


@dataclass
class ExtractConfig:
    data_root: Path
    device: str
    out_dir: Path = Path("outputs/extract")
    split_csv: Path | None = None
    classes: list[str] | None = None  # None = every class in the split file
    methods: list[str] = field(default_factory=lambda: ["baseline", "patchcore", "dinov2_knn"])
    method_options: dict = field(default_factory=dict)  # {"patchcore": {"coreset_ratio": 0.1}}
    seeds: list[int] = field(default_factory=lambda: [0, 1, 2])
    protocols: list[str] = field(default_factory=lambda: list(PROTOCOLS))
    corruptions: list[str] = field(default_factory=lambda: list(CORRUPTION_NAMES))
    n_severities: int = DEFAULT_N_SEVERITIES
    severity_levels: list[int] | None = None  # explicit override of n_severities
    memory_fraction: float = DEFAULT_MEMORY_FRACTION
    image_size: int = DEFAULT_IMAGE_SIZE
    batch_size: int = 16
    num_workers: int = 0
    max_images: int | None = None
    warmup_images: int = 8
    estimate_only: bool = False
    no_progress: bool = False
    pixel_metrics_protocols: list[str] = field(default_factory=lambda: ["official"])
    cache_corrupted: bool = True  # share corrupted images between methods (~1.5 GB RAM max)

    def scorer_config(self, method: str, seed: int) -> ScorerConfig:
        opts = dict(self.method_options.get(method, {}))
        return ScorerConfig(seed, self.device, self.batch_size, opts)

    @property
    def levels(self) -> tuple[int, ...]:
        if self.severity_levels:
            return tuple(sorted(set(self.severity_levels)))
        return levels_for_count(self.n_severities)

    @property
    def run_dir(self) -> Path:
        """Smoke-test outputs never mix with full runs."""
        sub = f"smoke_max{self.max_images}" if self.max_images else "full"
        return Path(self.out_dir) / sub

    def to_json(self) -> dict:
        d = dataclasses.asdict(self)
        return {k: (str(v) if isinstance(v, Path) else v) for k, v in d.items()}


@dataclass(frozen=True)
class Job:
    protocol: str
    method: str
    seed: int
    cls: str
    conditions: tuple[tuple[str, int], ...]


# --------------------------------------------------------------------------- planning


def check_device(device: str) -> None:
    """``--device`` is explicit and must exist: never fall back silently to the CPU."""
    if device == "cpu":
        return
    if not device.startswith("cuda"):
        raise ValueError(f"--device must be 'cpu', 'cuda' or 'cuda:N', got {device!r}")
    try:
        import torch
    except ImportError as e:
        raise RuntimeError(f"--device {device} needs torch: pip install -e .[gpu]") from e
    if not torch.cuda.is_available():
        raise RuntimeError(f"--device {device} requested but CUDA is not available")


def plan_jobs(cfg: ExtractConfig, classes: Sequence[str]) -> list[Job]:
    """Order: class -> protocol -> seed -> method.

    Class outermost: each class's images are decoded once. Method innermost: all methods
    of one (class, protocol, seed) score the *same* evaluation images, so the corrupted
    versions are computed once and shared (see ClassData.condition_images).
    """
    shifted = conditions(cfg.corruptions, cfg.levels)
    jobs = []
    for cls in classes:
        for protocol in cfg.protocols:
            conds = tuple(shifted) if protocol == "calib" else (("none", 0),)
            for seed in cfg.seeds:
                for method in cfg.methods:
                    jobs.append(Job(protocol, method, seed, cls, conds))
    return jobs


def pending_conditions(run_dir: Path, job: Job) -> list[tuple[str, int]]:
    return [
        c
        for c in job.conditions
        if not chunk_path(run_dir, job.protocol, job.method, job.seed, job.cls, *c).exists()
    ]


def check_fingerprint(run_dir: Path, fingerprint: dict) -> None:
    path = run_dir / "fingerprint.json"
    if path.exists():
        old = json.loads(path.read_text(encoding="utf-8"))
        if old != fingerprint:
            raise RuntimeError(
                f"{run_dir} was started with {old}, but this run uses {fingerprint}. "
                "Resuming would mix two experiments; use another --out folder."
            )
        return
    run_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(fingerprint, indent=2), encoding="utf-8")


def check_method_settings(run_dir: Path, settings: dict[str, dict]) -> None:
    """Per-method fingerprint: a method's settings may not change within one output folder
    (that would mix e.g. two coreset ratios in one score file). New methods can be added."""
    path = run_dir / "methods.json"
    stored = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    for method, s in settings.items():
        if method in stored and stored[method] != s:
            raise RuntimeError(
                f"{method} settings changed for {run_dir}: was {stored[method]}, now {s}. "
                "Use another --out folder."
            )
        stored[method] = s
    run_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(stored, indent=2, sort_keys=True), encoding="utf-8")


def save_split_manifest(run_dir: Path, split: ClassSplit) -> None:
    """Write the split ids (an audit trail for leakage); on resume, require them unchanged."""
    path = run_dir / "splits" / split.protocol / f"seed{split.seed}" / f"{split.cls}.json"
    data = split.to_dict()
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != data:
            raise RuntimeError(f"split changed since the first run: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=1), encoding="utf-8")


# --------------------------------------------------------------------------- running


class ClassData:
    """Decoded images of ONE class, cached so that every image is read from disk once.

    Jobs run class by class (see :func:`plan_jobs`), so all seeds, protocols and methods
    of a class reuse the same decoded arrays. At 256 px a class needs at most about
    1,100 x 196 KB = 220 MB of RAM; the cache is dropped when the next class starts.

    Corrupted versions of the current evaluation set are cached too (one set of ids at a
    time: at most ~650 images x 13 conditions x 196 KB = 1.6 GB), so the methods of one
    (class, protocol, seed) share them instead of re-corrupting per method.
    """

    def __init__(
        self,
        root: Path,
        cls: str,
        records: Sequence[VisaRecord],
        size: int,
        workers: int,
        cache_corrupted: bool = True,
    ):
        self.root, self.cls, self.size, self.workers = root, cls, size, workers
        self.by_id = {r.image_id: r for r in records if r.cls == cls}
        self.cache_corrupted = cache_corrupted
        self._cache: dict[str, np.ndarray] = {}
        self._masks: dict[str, np.ndarray] = {}
        self._cond_ids: tuple[str, ...] | None = None
        self._cond_cache: dict[tuple[str, int], np.ndarray] = {}
        self.seconds = {"load": 0.0, "corrupt": 0.0}

    def load(self, ids: Sequence[str]) -> np.ndarray:
        missing = [i for i in dict.fromkeys(ids) if i not in self._cache]
        if missing:
            t0 = time.perf_counter()
            paths = [self.by_id[i].image_path(self.root) for i in missing]
            for i, img in zip(missing, load_images(paths, self.size, self.workers), strict=True):
                self._cache[i] = img
            self.seconds["load"] += time.perf_counter() - t0
        return np.stack([self._cache[i] for i in ids])

    def condition_images(self, ids: Sequence[str], corruption: str, level: int) -> np.ndarray:
        ids = tuple(ids)
        if ids != self._cond_ids:  # a new evaluation set: forget the previous one
            self._cond_ids, self._cond_cache = ids, {}
        key = (corruption, level)
        if key in self._cond_cache:
            return self._cond_cache[key]
        clean = self.load(ids)
        t0 = time.perf_counter()
        out = apply_corruption_batch(clean, ids, corruption, level)
        self.seconds["corrupt"] += time.perf_counter() - t0
        if self.cache_corrupted:
            self._cond_cache[key] = out
        return out

    def masks(self, ids: Sequence[str]) -> np.ndarray:
        for i in ids:
            if i not in self._masks:
                self._masks[i] = load_mask(self.by_id[i].mask_path(self.root), self.size)
        return np.stack([self._masks[i] for i in ids])


def score_images(scorer, images: np.ndarray, batch: int, pbar, with_maps: bool):
    """Score in chunks of ``batch`` (the scorer may split further on OOM)."""
    scores, maps = [], []
    for start in range(0, len(images), batch):
        x = images[start : start + batch]
        if with_maps:
            s, m = scorer.score_with_maps(x)
            maps.append(m)
        else:
            s = scorer.score(x)
        scores.append(np.asarray(s, dtype=np.float32))
        if pbar is not None:
            pbar.update(len(x))
    return np.concatenate(scores), (np.concatenate(maps) if with_maps else None)


def run_job(
    cfg: ExtractConfig,
    job: Job,
    records: Sequence[VisaRecord],
    data: ClassData,
    todo: list[tuple[str, int]],
    pbar,
) -> tuple[int, dict]:
    t_job = time.perf_counter()
    before = dict(data.seconds)
    split = make_split(records, job.cls, job.protocol, job.seed, cfg.memory_fraction).limited(
        cfg.max_images
    )
    assert_disjoint(split, records, data.root)  # leakage check on ids, labels and file paths
    save_split_manifest(cfg.run_dir, split)

    set_global_seed(job.seed)
    t0 = time.perf_counter()
    scorer = build_scorer(job.method, cfg.scorer_config(job.method, job.seed))
    t_build = time.perf_counter() - t0
    memory_images = data.load(split.memory)
    t0 = time.perf_counter()
    scorer.fit(memory_images)
    t_fit = time.perf_counter() - t0
    del memory_images

    eval_ids = list(split.eval_ids)
    recs = [data.by_id[i] for i in eval_ids]
    roles = [ROLE_ANOMALY if r.label == 1 else ROLE_HELDOUT for r in recs]
    t_score = t_pixel = 0.0
    pixel = None
    for corruption, level in todo:
        images = data.condition_images(eval_ids, corruption, level)
        want_maps = (
            corruption == "none"
            and scorer.produces_maps
            and job.protocol in cfg.pixel_metrics_protocols
        )
        t0 = time.perf_counter()
        scores, maps = score_images(scorer, images, cfg.batch_size, pbar, want_maps)
        t_score += time.perf_counter() - t0
        if want_maps:
            t0 = time.perf_counter()
            pixel = pixel_metrics(job, data, eval_ids, recs, maps)
            write_json_atomic(
                pixel,
                pixel_metrics_path(cfg.run_dir, job.protocol, job.method, job.seed, job.cls),
            )
            t_pixel = time.perf_counter() - t0
            del maps
        df = make_frame(
            image_ids=eval_ids,
            cls=job.cls,
            labels=[r.label for r in recs],
            roles=roles,
            official_splits=[r.official_split for r in recs],
            protocol=job.protocol,
            method=job.method,
            seed=job.seed,
            corruption=corruption,
            severity=level,
            scores=scores,
        )
        # The chunk is written LAST: if it exists, the pixel metrics of this job exist too.
        write_atomic(
            df,
            chunk_path(cfg.run_dir, job.protocol, job.method, job.seed, job.cls, corruption, level),
        )
    record = {
        "protocol": job.protocol,
        "method": job.method,
        "seed": job.seed,
        "class": job.cls,
        "n_memory": len(split.memory),
        "n_eval": len(eval_ids),
        "conditions": len(todo),
        "seconds": {
            "load": data.seconds["load"] - before["load"],
            "corrupt": data.seconds["corrupt"] - before["corrupt"],
            "build_model": t_build,
            "fit": t_fit,
            "score": t_score,
            "pixel_metrics": t_pixel,
            "total": time.perf_counter() - t_job,
        },
        "scorer": scorer.describe(),
    }
    if pixel is not None:
        record["pixel_metrics"] = {k: pixel[k] for k in ("pixel_auroc", "pro_auc_0.3")}
    del scorer
    _free_device_memory(cfg.device)
    return len(todo), record


def pixel_metrics(job: Job, data: ClassData, ids, recs, maps: np.ndarray) -> dict:
    masks = data.masks(ids)
    out = {
        "protocol": job.protocol,
        "method": job.method,
        "seed": job.seed,
        "class": job.cls,
        "n_images": len(ids),
        "n_anomalous": int(sum(r.label for r in recs)),
        "map_size": int(maps.shape[-1]),
        "pixel_auroc": None,
        "pro_auc_0.3": None,
    }
    if masks.any() and (~masks).any():
        out["pixel_auroc"] = pixel_auroc(maps, masks)
        out["pro_auc_0.3"] = pro_auc(maps, masks)
    return out


def _free_device_memory(device: str) -> None:
    if device.startswith("cuda"):
        import gc

        import torch

        gc.collect()
        torch.cuda.empty_cache()


def eval_size(cfg: ExtractConfig, records: Sequence[VisaRecord], job: Job) -> tuple[int, int]:
    split = make_split(records, job.cls, job.protocol, job.seed, cfg.memory_fraction)
    split = split.limited(cfg.max_images)
    return len(split.memory), len(split.eval_ids)


# --------------------------------------------------------------------------- estimate


def estimate_runtime(
    cfg: ExtractConfig,
    records: Sequence[VisaRecord],
    root: Path,
    work: list[tuple[Job, list[tuple[str, int]]]],
) -> dict:
    """Time a short warm-up and extrapolate to the remaining work (rough by design).

    Measured per image on ``warmup_images`` memory images of the first pending class:
    loading, each corruption family (strongest used level), and per method: fitting
    and scoring (the score is timed on a *second* pass, so one-off CUDA start-up costs
    are excluded). Model construction is timed separately, because it includes a
    one-time weight download.

    Limits, stated in the output: fitting is extrapolated linearly, but PatchCore's
    coreset grows ~quadratically with the number of memory patches, and the warm-up's
    kNN runs against a tiny bank. For an accurate full-run estimate, run one full-size
    class and use ``scripts/estimate_full_run.py`` (see docs/LAPTOP_GPU_RUN.md).
    The warm-up writes nothing and uses no shared random state, so it cannot change any
    result.
    """
    first = work[0][0]
    data = ClassData(root, first.cls, records, cfg.image_size, cfg.num_workers)
    split = make_split(records, first.cls, "calib", first.seed, cfg.memory_fraction)
    ids = list(split.memory[: cfg.warmup_images])
    t0 = time.perf_counter()
    imgs = data.load(ids)
    t_load = (time.perf_counter() - t0) / len(ids)

    t_corrupt = {"none": 0.0}
    top = max(cfg.levels)
    for name in cfg.corruptions:
        t0 = time.perf_counter()
        apply_corruption_batch(imgs, ids, name, top)
        t_corrupt[name] = (time.perf_counter() - t0) / len(ids)

    t_build, t_rebuild, t_fit, t_score, vram = {}, {}, {}, {}, {}
    for method in cfg.methods:
        t0 = time.perf_counter()
        scorer = build_scorer(method, cfg.scorer_config(method, first.seed))
        t_build[method] = time.perf_counter() - t0  # may include a one-time weight download
        del scorer
        t0 = time.perf_counter()  # every job rebuilds its model, from the on-disk cache
        scorer = build_scorer(method, cfg.scorer_config(method, first.seed))
        t_rebuild[method] = time.perf_counter() - t0
        t0 = time.perf_counter()
        scorer.fit(imgs)
        t_fit[method] = (time.perf_counter() - t0) / len(ids)
        scorer.score(imgs)  # first pass: CUDA kernels, cuDNN autotuning
        t0 = time.perf_counter()
        scorer.score(imgs)
        t_score[method] = (time.perf_counter() - t0) / len(ids)
        vram[method] = scorer.describe().get("peak_vram_mb")
        del scorer
        _free_device_memory(cfg.device)

    # Corruptions are computed once per (class, protocol, seed) when cached, else per job.
    total = 0.0
    n_units = 0
    to_load: dict[str, set[str]] = {}
    corrupt_groups: set[tuple] = set()
    for job, todo in work:
        split = make_split(records, job.cls, job.protocol, job.seed, cfg.memory_fraction)
        split = split.limited(cfg.max_images)
        to_load.setdefault(job.cls, set()).update(split.memory, split.eval_ids)
        n_mem, n_eval = len(split.memory), len(split.eval_ids)
        total += t_rebuild[job.method] + n_mem * t_fit[job.method]
        for corruption, level in todo:
            total += n_eval * t_score[job.method]
            group = (job.cls, job.protocol, job.seed, corruption, level)
            if not cfg.cache_corrupted or group not in corrupt_groups:
                corrupt_groups.add(group)
                total += n_eval * t_corrupt[corruption]
            n_units += n_eval
    total += t_load * sum(len(v) for v in to_load.values())
    return {
        "seconds": total,
        "image_condition_units": n_units,
        "per_image_seconds": {
            "load": t_load,
            "corrupt": t_corrupt,
            "fit": t_fit,
            "score": t_score,
        },
        "model_build_seconds": t_build,
        "model_rebuild_seconds": t_rebuild,
        "warmup_peak_vram_mb": vram,
        "warmup_images": len(ids),
    }


def format_duration(seconds: float) -> str:
    seconds = int(round(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h} h {m:02d} min" if h else f"{m} min {s:02d} s"


# --------------------------------------------------------------------------- metadata


def git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(anomcal.__file__).resolve().parent,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def environment_info(device: str) -> dict:
    import pandas as pd
    import PIL
    import scipy

    info = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "anomcal": anomcal.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "pillow": PIL.__version__,
        "scipy": scipy.__version__,
        "git_commit": git_commit(),
        "device": device,
    }
    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        if device.startswith("cuda") and torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(torch.device(device))
    except ImportError:
        info["torch"] = None
    return info


def append_run_meta(run_dir: Path, entry: dict) -> None:
    path = run_dir / "run_meta.json"
    runs = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
    runs.append(entry)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(runs, indent=2, default=str), encoding="utf-8")
    tmp.replace(path)


# --------------------------------------------------------------------------- entry point


def run(cfg: ExtractConfig) -> dict:
    check_device(cfg.device)
    unknown = set(cfg.methods) - set(AVAILABLE_METHODS)
    if unknown:
        raise ValueError(f"unknown method(s) {sorted(unknown)}; available: {AVAILABLE_METHODS}")
    stray = set(cfg.method_options) - set(cfg.methods)
    if stray:
        raise ValueError(f"method_options given for methods not in this run: {sorted(stray)}")
    # Resolve every method's settings now: typos in options fail before any work starts.
    method_settings = {m: settings_dict(m, cfg.method_options.get(m)) for m in cfg.methods}
    if set(cfg.protocols) - set(PROTOCOLS):
        raise ValueError(f"protocols must be among {PROTOCOLS}")
    if cfg.batch_size < 1 or cfg.num_workers < 0 or cfg.warmup_images < 1:
        raise ValueError("batch_size and warmup_images must be >= 1, num_workers >= 0")

    root = resolve_visa_root(cfg.data_root)
    csv_path = find_split_csv(root, cfg.split_csv)
    records = read_split_csv(csv_path)
    present = {r.cls for r in records}
    classes = cfg.classes or [c for c in VISA_CLASSES if c in present] + sorted(
        present - set(VISA_CLASSES)
    )
    missing = set(classes) - present
    if missing:
        raise ValueError(f"classes not in the split file: {sorted(missing)}")

    run_dir = cfg.run_dir
    fingerprint = {
        "memory_fraction": cfg.memory_fraction,
        "image_size": cfg.image_size,
        "max_images": cfg.max_images,
        "split_csv_sha256": sha256_file(csv_path),
    }
    assert tuple(fingerprint) == FINGERPRINT_KEYS
    check_fingerprint(run_dir, fingerprint)
    check_method_settings(run_dir, method_settings)

    jobs = plan_jobs(cfg, classes)
    work = [(j, pending_conditions(run_dir, j)) for j in jobs]
    work = [(j, todo) for j, todo in work if todo]
    done_chunks = sum(len(j.conditions) for j in jobs) - sum(len(t) for _, t in work)
    print(f"output folder : {run_dir}")
    print(
        f"plan          : {len(jobs)} jobs (class x protocol x seed x method), "
        f"{sum(len(j.conditions) for j in jobs)} chunks, {done_chunks} already done"
    )
    print(f"severity lvls : {list(cfg.levels)} of 1..{MAX_LEVEL}")

    started = time.time()
    entry = {
        "started_unix": started,
        "config": cfg.to_json(),
        "fingerprint": fingerprint,
        "method_settings": method_settings,
        "levels": list(cfg.levels),
        "severity_parameters": severity_table(cfg.corruptions, cfg.levels),
        "environment": environment_info(cfg.device),
        "status": "nothing to do",
    }
    if not work:
        print("nothing left to do (all chunks exist)")
        entry["consolidated"] = _consolidate_all(run_dir, jobs)
        append_run_meta(run_dir, entry)
        return entry

    est = estimate_runtime(cfg, records, root, work)
    entry["estimate"] = est
    print(
        f"estimated time: ~{format_duration(est['seconds'])} for "
        f"{est['image_condition_units']} image-conditions "
        f"(rough: from a {est['warmup_images']}-image warm-up; the progress bar refines it)"
    )
    for m in cfg.methods:
        vram = est["warmup_peak_vram_mb"][m]
        vram_txt = f", warm-up peak VRAM {vram:.0f} MB" if vram is not None else ""
        print(
            f"  {m:<11} model build {est['model_build_seconds'][m]:.1f} s (incl. first download), "
            f"fit {1000 * est['per_image_seconds']['fit'][m]:.1f} ms/img, "
            f"score {1000 * est['per_image_seconds']['score'][m]:.1f} ms/img{vram_txt}"
        )
    if any(m in ("patchcore", "dinov2_knn") for m in cfg.methods):
        print(
            "  note: coreset and kNN cost grow with the memory bank; the warm-up bank is tiny. "
            "For a full-run estimate, run one full-size class, then scripts/estimate_full_run.py."
        )
    if cfg.estimate_only:
        entry["status"] = "estimate only"
        append_run_meta(run_dir, entry)
        return entry

    written = 0
    entry["jobs"] = []
    total_units = sum(eval_size(cfg, records, j)[1] * len(t) for j, t in work)
    pbar = tqdm(total=total_units, unit="img", disable=cfg.no_progress, dynamic_ncols=True)
    entry["status"] = "interrupted"
    try:
        data: ClassData | None = None
        for job, todo in work:
            pbar.set_description(f"{job.protocol}/{job.method}/s{job.seed}/{job.cls}")
            if data is None or data.cls != job.cls:  # new class: drop the previous cache
                data = ClassData(
                    root, job.cls, records, cfg.image_size, cfg.num_workers, cfg.cache_corrupted
                )
            n, record = run_job(cfg, job, records, data, todo, pbar)
            written += n
            entry["jobs"].append(record)
        entry["status"] = "complete"
    finally:
        pbar.close()
        entry["chunks_written"] = written
        entry["elapsed_seconds"] = time.time() - started
        if entry["status"] == "complete":
            entry["consolidated"] = _consolidate_all(run_dir, jobs)
        append_run_meta(run_dir, entry)
    print(f"done: {written} chunks in {format_duration(entry['elapsed_seconds'])}")
    for path in entry.get("consolidated", []):
        print(f"  -> {path}")
    return entry


def _consolidate_all(run_dir: Path, jobs: Sequence[Job]) -> list[str]:
    groups = sorted({(j.protocol, j.method, j.seed) for j in jobs})
    out = []
    for g in groups:
        path = consolidate(run_dir, *g)
        if path is not None:
            out.append(str(path))
    pixel = consolidate_pixel_metrics(run_dir)
    if pixel is not None:
        out.append(str(pixel))
    return out


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="extract_scores.py",
        description="Score VisA images (clean + corrupted) for anomaly detection methods.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Defaults are None here so a --config file can supply values; CLI flags win.
    a = p.add_argument
    a("--config", type=Path, help="JSON file with any of the options below (CLI flags override it)")
    a("--data-root", type=Path, help="folder containing VisA (e.g. data/visa); required")
    a("--device", help="explicit device: cpu, cuda or cuda:0; required")
    a("--split-csv", type=Path, help="default: <data-root>/split_csv/1cls.csv")
    a("--out", dest="out_dir", type=Path, help="output folder (default outputs/extract)")
    a("--classes", nargs="+", help="default: all classes in the split file")
    a("--methods", nargs="+", help=f"available: {', '.join(AVAILABLE_METHODS)}")
    a("--seeds", nargs="+", type=int, help="default: 0 1 2")
    a("--protocols", nargs="+", choices=PROTOCOLS, help="default: calib official")
    a("--corruptions", nargs="+", choices=CORRUPTION_NAMES, help="default: all four")
    a(
        "--severities",
        dest="n_severities",
        type=int,
        help=f"number of levels (default {DEFAULT_N_SEVERITIES})",
    )
    a("--severity-levels", nargs="+", type=int, help="explicit levels, overrides --severities")
    a("--memory-fraction", type=float, help=f"default {DEFAULT_MEMORY_FRACTION}")
    a("--image-size", type=int, help=f"canonical resolution (default {DEFAULT_IMAGE_SIZE})")
    a("--batch-size", type=int, help="default 16")
    a("--num-workers", type=int, help="image-loading threads (default 0, safe on Windows)")
    a("--max-images", type=int, help="smoke test: at most N images per role and class")
    a("--warmup-images", type=int, help="images timed for the runtime estimate (default 8)")
    a("--estimate-only", action="store_true", default=None, help="print the estimate and exit")
    a("--no-progress", action="store_true", default=None, help="disable the progress bar")
    a(
        "--opt",
        action="append",
        metavar="METHOD.KEY=VALUE",
        help="method setting, repeatable, e.g. --opt patchcore.coreset_ratio=0.05",
    )
    a("--pixel-metrics-protocols", nargs="*", choices=PROTOCOLS, help="default: official")
    a(
        "--no-corruption-cache",
        dest="cache_corrupted",
        action="store_false",
        default=None,
        help="do not share corrupted images between methods (saves ~1.5 GB RAM)",
    )
    return p


def parse_opts(items: Sequence[str]) -> dict[str, dict]:
    """``["patchcore.coreset_ratio=0.05", "dinov2_knn.amp=false"]`` -> nested dict.

    Values are parsed as JSON (numbers, true/false); anything else stays a string.
    """
    out: dict[str, dict] = {}
    for item in items:
        key, sep, raw = item.partition("=")
        method, dot, name = key.partition(".")
        if not sep or not dot or not method or not name:
            raise SystemExit(f"--opt expects METHOD.KEY=VALUE, got {item!r}")
        try:
            value = json.loads(raw)
        except json.JSONDecodeError:
            value = raw
        out.setdefault(method, {})[name] = value
    return out


def config_from_args(argv: Sequence[str] | None = None) -> ExtractConfig:
    args = vars(build_parser().parse_args(argv))
    config_path = args.pop("config")
    opts = parse_opts(args.pop("opt") or [])
    merged: dict = {}
    if config_path is not None:
        merged.update(json.loads(Path(config_path).read_text(encoding="utf-8")))
    merged.update({k: v for k, v in args.items() if v is not None})
    if opts:  # CLI options are merged into (and override) the config file's options
        combined = {m: dict(o) for m, o in merged.get("method_options", {}).items()}
        for m, o in opts.items():
            combined.setdefault(m, {}).update(o)
        merged["method_options"] = combined
    known = {f.name for f in dataclasses.fields(ExtractConfig)}
    unknown = set(merged) - known
    if unknown:
        raise SystemExit(f"unknown config key(s): {sorted(unknown)}")
    for key in ("data_root", "device"):
        if key not in merged:
            raise SystemExit(f"--{key.replace('_', '-')} is required (no implicit default)")
    for key in ("data_root", "out_dir", "split_csv"):
        if merged.get(key) is not None:
            merged[key] = Path(merged[key])
    return ExtractConfig(**merged)


def main(argv: Sequence[str] | None = None) -> int:
    cfg = config_from_args(argv)
    try:
        run(cfg)
    except KeyboardInterrupt:
        print("\ninterrupted: finished chunks are kept; rerun the same command to resume")
        return 130
    except (ValueError, RuntimeError, FileNotFoundError) as e:
        # Expected user-facing problems (bad flags, missing data, changed settings):
        # a one-line message instead of a traceback.
        print(f"error: {e}", file=sys.stderr)
        return 2
    return 0
