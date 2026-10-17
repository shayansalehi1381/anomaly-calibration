"""Extrapolate the full laptop run from per-job measurements of a smaller run.

Workflow: run every method on ONE class at full size (memory bank and evaluation set
not limited), e.g. ``--classes cashew --seeds 0``. Its per-job timings and peak VRAM are
in ``run_meta.json``. This module fits a small cost model per method and predicts the
remaining work of the full plan (all classes, all seeds), skipping chunks that exist.

Cost model per job (per method; all coefficients measured, none assumed):

    build_model                                     (constant per job)
  + a  * n_memory_images                            (backbone features for the bank)
  + c  * n_memory_patches * n_selected_patches      (greedy coreset: every iteration
                                                     scans all patches)
  + (s0 + s1 * bank_size) * n_eval * n_conditions   (backbone + kNN per scored image;
                                                     kNN cost grows with the bank)
  + p  * n_eval                                     (pixel metrics, clean only)

plus, per class, image loading (per image) and corruptions (per image-condition,
computed once per class/protocol/seed). s0 and s1 are fitted from jobs with different
bank sizes (the `calib` and `official` protocols of the same class differ by ~2x); if
only one bank size was measured, s1 = 0.

Peak VRAM: we report the measured peak per method, plus a **linear upper bound** for the
largest planned memory bank (peak x bank_target / bank_measured). Model weights do not
grow, so the true peak is lower than the bound.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from anomcal.corruptions import CORRUPTION_NAMES, conditions, levels_for_count
from anomcal.data.splits import DEFAULT_MEMORY_FRACTION, PROTOCOLS, make_split
from anomcal.data.visa import EXPECTED_1CLS_COUNTS, VISA_CLASSES, records_from_counts
from anomcal.scores_io import chunk_path


def load_jobs(run_dir: Path) -> tuple[list[dict], dict]:
    """All job records of all invocations in ``run_dir/run_meta.json`` + the last config."""
    path = Path(run_dir) / "run_meta.json"
    entries = json.loads(path.read_text(encoding="utf-8"))
    jobs = [j for e in entries for j in e.get("jobs", [])]
    if not jobs:
        raise ValueError(f"{path} has no job records; run at least one class first")
    return jobs, entries[-1]["config"]


def fit_cost_model(jobs: Sequence[dict]) -> dict[str, dict]:
    """Per-method coefficients from measured job records (see module docstring)."""
    by_method: dict[str, list[dict]] = defaultdict(list)
    for j in jobs:
        by_method[j["method"]].append(j)
    models = {}
    for method, js in by_method.items():
        sec = [j["seconds"] for j in js]
        info = [j["scorer"] for j in js]
        n_mem = np.array([j["n_memory"] for j in js], float)
        fit_feat = np.array(
            [
                i.get("seconds", {}).get("fit_features", s["fit"])
                for i, s in zip(info, sec, strict=True)
            ]
        )
        coreset = np.array([i.get("seconds", {}).get("fit_coreset", 0.0) for i in info])
        patches = np.array([i.get("n_memory_patches", 0) for i in info], float)
        bank = np.array([i.get("bank_size", 0) for i in info], float)
        units = np.array([j["n_eval"] * j["conditions"] for j in js], float)
        score = np.array([s["score"] for s in sec])
        pix_jobs = [j for j in js if j["seconds"].get("pixel_metrics", 0) > 0]

        per_patch_img = float(np.sum(patches) / np.sum(n_mem)) if patches.any() else 0.0
        work = patches * bank
        m = {
            "build": float(np.mean([s["build_model"] for s in sec])),
            "feat_per_image": float(np.sum(fit_feat) / np.sum(n_mem)),
            "coreset_per_pair": float(np.sum(coreset) / np.sum(work)) if work.any() else 0.0,
            "patches_per_image": per_patch_img,
            "bank_per_patch": float(np.sum(bank) / np.sum(patches)) if patches.any() else 0.0,
            "bank_per_image": float(np.sum(bank) / np.sum(n_mem)),
            "pixel_per_image": (
                float(
                    sum(j["seconds"]["pixel_metrics"] for j in pix_jobs)
                    / sum(j["n_eval"] for j in pix_jobs)
                )
                if pix_jobs
                else 0.0
            ),
            "peak_vram_mb": max((i.get("peak_vram_mb") or 0.0) for i in info) or None,
            "peak_bank": float(bank[np.argmax([i.get("peak_vram_mb") or 0.0 for i in info])]),
        }
        per_unit = score / units
        if len(set(bank)) >= 2:  # least squares: per-unit score time = s0 + s1 * bank
            s1, s0 = np.polyfit(bank, per_unit, 1)
            s1 = max(float(s1), 0.0)
            s0 = float(np.mean(per_unit - s1 * bank))
        else:
            s0, s1 = float(np.mean(per_unit)), 0.0
        m["score_s0"], m["score_s1"] = max(s0, 0.0), s1
        models[method] = m

    loads = [j["seconds"]["load"] for j in jobs if j["seconds"]["load"] > 0]
    loaded = [j["n_memory"] + j["n_eval"] for j in jobs if j["seconds"]["load"] > 0]
    corr = [(j["seconds"]["corrupt"], j["n_eval"] * j["conditions"]) for j in jobs]
    corr = [(s, u) for s, u in corr if s > 0]
    models["_io"] = {
        # upper bound: the first job of a class loads its images, later ones reuse them
        "load_per_image": float(sum(loads) / sum(loaded)) if loads else 0.0,
        "corrupt_per_unit": float(sum(s for s, _ in corr) / sum(u for _, u in corr))
        if corr
        else 0.0,
    }
    return models


def predict(
    models: dict[str, dict],
    classes: Sequence[str],
    seeds: Sequence[int],
    methods: Sequence[str],
    protocols: Sequence[str],
    calib_conditions: Sequence[tuple[str, int]],
    memory_fraction: float = DEFAULT_MEMORY_FRACTION,
    pixel_protocols: Sequence[str] = ("official",),
    run_dir: Path | None = None,
    counts: dict = EXPECTED_1CLS_COUNTS,
) -> dict:
    """Predicted seconds per method (and I/O), skipping chunks already in ``run_dir``."""
    missing = set(methods) - set(models)
    if missing:
        raise ValueError(f"no measurements for {sorted(missing)}; run them on one class first")
    records = records_from_counts({c: counts[c] for c in classes})
    io = models["_io"]
    out = {m: 0.0 for m in methods}
    max_bank = {m: 0.0 for m in methods}
    io_seconds = 0.0
    corrupt_groups = set()
    for cls in classes:
        class_images = sum(counts[cls])
        class_has_work = False
        for protocol in protocols:
            conds = list(calib_conditions) if protocol == "calib" else [("none", 0)]
            for seed in seeds:
                split = make_split(records, cls, protocol, seed, memory_fraction)
                n_mem, n_eval = len(split.memory), len(split.eval_ids)
                for method in methods:
                    todo = [
                        c
                        for c in conds
                        if run_dir is None
                        or not chunk_path(run_dir, protocol, method, seed, cls, *c).exists()
                    ]
                    if not todo:
                        continue
                    class_has_work = True
                    m = models[method]
                    patches = n_mem * m["patches_per_image"]
                    bank = patches * m["bank_per_patch"] if patches else n_mem * m["bank_per_image"]
                    max_bank[method] = max(max_bank[method], bank)
                    t = m["build"] + m["feat_per_image"] * n_mem
                    t += m["coreset_per_pair"] * patches * bank
                    t += (m["score_s0"] + m["score_s1"] * bank) * n_eval * len(todo)
                    if protocol in pixel_protocols and ("none", 0) in todo:
                        t += m["pixel_per_image"] * n_eval
                    out[method] += t
                    for c in todo:
                        if (cls, protocol, seed, c) not in corrupt_groups:
                            corrupt_groups.add((cls, protocol, seed, c))
                            io_seconds += io["corrupt_per_unit"] * n_eval
        if class_has_work:
            io_seconds += io["load_per_image"] * class_images
    vram = {}
    for method in methods:
        peak, peak_bank = models[method]["peak_vram_mb"], models[method]["peak_bank"]
        bound = (
            peak * max_bank[method] / peak_bank if peak and peak_bank and max_bank[method] else peak
        )
        vram[method] = {"measured_mb": peak, "largest_bank_upper_bound_mb": bound}
    return {
        "seconds_per_method": out,
        "io_seconds": io_seconds,
        "total_seconds": sum(out.values()) + io_seconds,
        "peak_vram": vram,
    }


def _fmt(seconds: float) -> str:
    h, rem = divmod(int(round(seconds)), 3600)
    return f"{h} h {rem // 60:02d} min"


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Extrapolate the full run from measured jobs.")
    p.add_argument("--run-dir", type=Path, required=True, help="e.g. outputs/extract/full")
    p.add_argument("--classes", nargs="+", default=list(VISA_CLASSES))
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--methods", nargs="+", help="default: the methods of the measured run")
    p.add_argument("--protocols", nargs="+", default=list(PROTOCOLS), choices=PROTOCOLS)
    p.add_argument("--severities", type=int, help="default: as in the measured run")
    p.add_argument("--memory-fraction", type=float, help="default: as in the measured run")
    p.add_argument("--vram-gb", type=float, default=6.0, help="GPU memory to compare against")
    args = p.parse_args(argv)

    jobs, cfg = load_jobs(args.run_dir)
    models = fit_cost_model(jobs)
    methods = args.methods or cfg["methods"]
    levels = (
        levels_for_count(args.severities)
        if args.severities
        else tuple(cfg.get("severity_levels") or levels_for_count(cfg.get("n_severities", 3)))
    )
    conds = conditions(cfg.get("corruptions", list(CORRUPTION_NAMES)), levels)
    frac = args.memory_fraction or cfg.get("memory_fraction", DEFAULT_MEMORY_FRACTION)
    res = predict(
        models, args.classes, args.seeds, methods, args.protocols, conds, frac,
        cfg.get("pixel_metrics_protocols", ["official"]), args.run_dir,
    )  # fmt: skip
    measured = sorted({j["class"] for j in jobs})
    print(f"measured on : {', '.join(measured)} ({len(jobs)} jobs)")
    print(f"plan        : {len(args.classes)} classes x seeds {args.seeds} x {args.protocols}, "
          f"severity levels {list(levels)}; already-written chunks are skipped")  # fmt: skip
    header = f"{'method':<12} {'GPU time':>10} {'peak VRAM measured':>19}"
    print(f"{header} {'upper bound (largest bank)':>27}")
    for m in methods:
        v = res["peak_vram"][m]
        meas = f"{v['measured_mb']:.0f} MB" if v["measured_mb"] else "n/a (CPU)"
        bound = v["largest_bank_upper_bound_mb"]
        bound_txt = f"{bound:.0f} MB" if bound else "n/a"
        flag = "  <-- may exceed VRAM" if bound and bound > args.vram_gb * 1024 * 0.9 else ""
        print(f"{m:<12} {_fmt(res['seconds_per_method'][m]):>10} {meas:>19} {bound_txt:>27}{flag}")
    print(f"{'load+corrupt':<12} {_fmt(res['io_seconds']):>10}  (CPU)")
    total = _fmt(res["total_seconds"])
    print(f"{'TOTAL':<12} {total:>10}  (estimate from measurements, not a result)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
