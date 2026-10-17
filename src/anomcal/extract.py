"""Score extraction: the only step that runs on the laptop (GPU from Phase 2 on).

For every (protocol, method, seed, class):

1. build the split (memory bank vs. held-out pool) and **assert it is leak-free**;
2. load memory-bank + evaluation images once, at the canonical resolution;
3. fit the scorer on the memory bank (clean images only);
4. for every condition (clean + each corruption x severity) that is not done yet,
   corrupt the evaluation images batch by batch, score them, and write one chunk file.

Everything is resumable at chunk granularity (see :mod:`anomcal.scores_io`). Before the
real work, a short timed warm-up prints an estimated runtime for what is left.

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
from anomcal.data.images import DEFAULT_IMAGE_SIZE, load_images
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
from anomcal.scorers import AVAILABLE_METHODS, ScorerConfig, build_scorer
from anomcal.scores_io import chunk_path, consolidate, make_frame, write_atomic
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
    methods: list[str] = field(default_factory=lambda: ["dummy"])
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
    """Class is the OUTER loop, so each class's images are decoded only once."""
    shifted = conditions(cfg.corruptions, cfg.levels)
    jobs = []
    for cls in classes:
        for protocol in cfg.protocols:
            conds = tuple(shifted) if protocol == "calib" else (("none", 0),)
            for method in cfg.methods:
                for seed in cfg.seeds:
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
    """

    def __init__(
        self, root: Path, cls: str, records: Sequence[VisaRecord], size: int, workers: int
    ):
        self.root, self.cls, self.size, self.workers = root, cls, size, workers
        self.by_id = {r.image_id: r for r in records if r.cls == cls}
        self._cache: dict[str, np.ndarray] = {}

    def load(self, ids: Sequence[str]) -> np.ndarray:
        missing = [i for i in dict.fromkeys(ids) if i not in self._cache]
        if missing:
            paths = [self.by_id[i].image_path(self.root) for i in missing]
            for i, img in zip(missing, load_images(paths, self.size, self.workers), strict=True):
                self._cache[i] = img
        return np.stack([self._cache[i] for i in ids])


def score_condition(
    scorer, images: np.ndarray, ids: Sequence[str], corruption: str, level: int, batch: int, pbar
) -> np.ndarray:
    out = np.empty(len(ids), dtype=np.float32)
    for start in range(0, len(ids), batch):
        sl = slice(start, start + batch)
        x = apply_corruption_batch(images[sl], ids[sl], corruption, level)
        out[sl] = scorer.score(x)
        if pbar is not None:
            pbar.update(len(x))
    return out


def run_job(
    cfg: ExtractConfig,
    job: Job,
    records: Sequence[VisaRecord],
    data: ClassData,
    todo: list[tuple[str, int]],
    pbar,
) -> int:
    split = make_split(records, job.cls, job.protocol, job.seed, cfg.memory_fraction).limited(
        cfg.max_images
    )
    assert_disjoint(split, records, data.root)  # leakage check on ids, labels and file paths
    save_split_manifest(cfg.run_dir, split)

    set_global_seed(job.seed)
    scorer = build_scorer(job.method, ScorerConfig(job.seed, cfg.device, cfg.batch_size))
    scorer.fit(data.load(split.memory))

    eval_ids = list(split.eval_ids)
    images = data.load(eval_ids)
    recs = [data.by_id[i] for i in eval_ids]
    roles = [ROLE_ANOMALY if r.label == 1 else ROLE_HELDOUT for r in recs]
    for corruption, level in todo:
        scores = score_condition(scorer, images, eval_ids, corruption, level, cfg.batch_size, pbar)
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
        write_atomic(
            df,
            chunk_path(cfg.run_dir, job.protocol, job.method, job.seed, job.cls, corruption, level),
        )
    return len(todo)


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

    Measured per image: loading, each corruption family (at its strongest used
    level) and, per method, fitting and scoring. Fitting is extrapolated linearly,
    which underestimates methods whose fit grows faster than linearly (e.g. a coreset);
    the progress bar's live ETA corrects this as the run proceeds.
    The warm-up writes nothing and uses no shared random state (all randomness is
    seeded per item or per job), so it cannot change any result.
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

    t_fit, t_score = {}, {}
    for method in cfg.methods:
        scorer = build_scorer(method, ScorerConfig(first.seed, cfg.device, cfg.batch_size))
        t0 = time.perf_counter()
        scorer.fit(imgs)
        t_fit[method] = (time.perf_counter() - t0) / len(ids)
        t0 = time.perf_counter()
        scorer.score(imgs)
        t_score[method] = (time.perf_counter() - t0) / len(ids)

    total = 0.0
    n_units = 0
    to_load: dict[str, set[str]] = {}  # images are decoded once per class (ClassData cache)
    for job, todo in work:
        split = make_split(records, job.cls, job.protocol, job.seed, cfg.memory_fraction)
        split = split.limited(cfg.max_images)
        to_load.setdefault(job.cls, set()).update(split.memory, split.eval_ids)
        n_mem, n_eval = len(split.memory), len(split.eval_ids)
        total += n_mem * t_fit[job.method]
        for corruption, _ in todo:
            total += n_eval * (t_corrupt[corruption] + t_score[job.method])
            n_units += n_eval
    total += t_load * sum(len(ids) for ids in to_load.values())
    return {
        "seconds": total,
        "image_condition_units": n_units,
        "per_image_seconds": {
            "load": t_load,
            "corrupt": t_corrupt,
            "fit": t_fit,
            "score": t_score,
        },
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

    jobs = plan_jobs(cfg, classes)
    work = [(j, pending_conditions(run_dir, j)) for j in jobs]
    work = [(j, todo) for j, todo in work if todo]
    done_chunks = sum(len(j.conditions) for j in jobs) - sum(len(t) for _, t in work)
    print(f"output folder : {run_dir}")
    print(
        f"plan          : {len(jobs)} jobs (protocol x method x seed x class), "
        f"{sum(len(j.conditions) for j in jobs)} chunks, {done_chunks} already done"
    )
    print(f"severity lvls : {list(cfg.levels)} of 1..{MAX_LEVEL}")

    started = time.time()
    entry = {
        "started_unix": started,
        "config": cfg.to_json(),
        "fingerprint": fingerprint,
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
    if cfg.estimate_only:
        entry["status"] = "estimate only"
        append_run_meta(run_dir, entry)
        return entry

    written = 0
    total_units = sum(eval_size(cfg, records, j)[1] * len(t) for j, t in work)
    pbar = tqdm(total=total_units, unit="img", disable=cfg.no_progress, dynamic_ncols=True)
    entry["status"] = "interrupted"
    try:
        data: ClassData | None = None
        for job, todo in work:
            pbar.set_description(f"{job.protocol}/{job.method}/s{job.seed}/{job.cls}")
            if data is None or data.cls != job.cls:  # new class: drop the previous cache
                data = ClassData(root, job.cls, records, cfg.image_size, cfg.num_workers)
            written += run_job(cfg, job, records, data, todo, pbar)
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
    return p


def config_from_args(argv: Sequence[str] | None = None) -> ExtractConfig:
    args = vars(build_parser().parse_args(argv))
    config_path = args.pop("config")
    merged: dict = {}
    if config_path is not None:
        merged.update(json.loads(Path(config_path).read_text(encoding="utf-8")))
    merged.update({k: v for k, v in args.items() if v is not None})
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
