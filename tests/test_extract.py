"""End-to-end tests of the extraction pipeline on the fake dataset (CPU, dummy scorer)."""

import importlib.util
import json

import pandas as pd
import pytest

import anomcal.extract as ex
from anomcal.extract import ExtractConfig, config_from_args, main, run
from anomcal.scores_io import validate

N_CONDITIONS_3 = 1 + 4 * 3  # clean + 4 families x 3 levels


def cfg_for(root, out, **kw) -> ExtractConfig:
    # 64 px keeps the suite fast; the pipeline is resolution-agnostic.
    base = dict(
        data_root=root,
        device="cpu",
        out_dir=out,
        seeds=[0],
        no_progress=True,
        image_size=64,
        methods=["dummy"],
    )
    base.update(kw)
    return ExtractConfig(**base)


def chunks(run_dir):
    return sorted(
        p.relative_to(run_dir).as_posix() for p in (run_dir / "chunks").rglob("*.parquet")
    )


def consolidated(run_dir, name="calib_dummy_seed0.parquet") -> pd.DataFrame:
    return pd.read_parquet(run_dir / "scores" / name)


@pytest.fixture
def full_run(fake_visa_root, tmp_path):
    cfg = cfg_for(fake_visa_root, tmp_path / "out")
    entry = run(cfg)
    return cfg, entry


def test_full_run_writes_every_chunk(full_run):
    cfg, entry = full_run
    assert entry["status"] == "complete"
    names = chunks(cfg.run_dir)
    # 2 classes x 13 calib conditions + 2 classes x 1 official (clean) condition
    assert len(names) == 2 * N_CONDITIONS_3 + 2
    assert sum("chunks/official/" in n for n in names) == 2
    assert all(n.endswith("none_s0.parquet") for n in names if "/official/" in n)
    assert cfg.run_dir.name == "full"


def test_scores_follow_schema_and_cover_eval_sets(full_run, fake_records):
    cfg, _ = full_run
    df = consolidated(cfg.run_dir)
    validate(df.astype({"seed": "int32"}))
    # per class: held-out = 6 train leftovers + 4 test normals; anomalies = 4
    per_cond = df.groupby(["class", "corruption", "severity"]).size()
    assert (per_cond == 14).all() and len(per_cond) == 2 * N_CONDITIONS_3
    official = consolidated(cfg.run_dir, "official_dummy_seed0.parquet")
    assert set(official["official_split"]) == {"test"} and len(official) == 2 * 8


def test_no_memory_image_is_ever_scored(full_run):
    """Leakage check on the actual outputs, using the saved split manifests."""
    cfg, _ = full_run
    scored = consolidated(cfg.run_dir)
    for cls in ("candle", "pcb1"):
        manifest = json.loads(
            (cfg.run_dir / "splits" / "calib" / "seed0" / f"{cls}.json").read_text()
        )
        memory = set(manifest["memory"])
        assert memory and not memory & set(scored.loc[scored["class"] == cls, "image_id"])
        assert set(manifest["heldout_normal"]).isdisjoint(manifest["anomaly"])


def test_run_meta_records_settings(full_run):
    cfg, _ = full_run
    meta = json.loads((cfg.run_dir / "run_meta.json").read_text())[-1]
    assert meta["levels"] == [1, 3, 5]
    assert meta["severity_parameters"]["gaussian_blur"]["levels"] == {"1": 1, "3": 3, "5": 6}
    assert meta["environment"]["device"] == "cpu" and "pillow" in meta["environment"]
    assert meta["fingerprint"]["memory_fraction"] == 0.5
    assert meta["estimate"]["seconds"] >= 0


def test_rerun_is_a_noop(full_run, capsys):
    cfg, _ = full_run
    entry = run(cfg)
    assert entry["status"] == "nothing to do"
    assert "nothing left to do" in capsys.readouterr().out


def test_resume_after_interruption_gives_identical_results(fake_visa_root, tmp_path, monkeypatch):
    reference_cfg = cfg_for(fake_visa_root, tmp_path / "ref", classes=["candle"])
    run(reference_cfg)

    cfg = cfg_for(fake_visa_root, tmp_path / "resumed", classes=["candle"])
    real_write = ex.write_atomic
    calls = {"n": 0}

    def write_then_crash(df, path):
        calls["n"] += 1
        if calls["n"] == 5:
            raise KeyboardInterrupt  # Ctrl+C in the middle of the run
        real_write(df, path)

    monkeypatch.setattr(ex, "write_atomic", write_then_crash)
    with pytest.raises(KeyboardInterrupt):
        run(cfg)
    assert len(chunks(cfg.run_dir)) == 4
    meta = json.loads((cfg.run_dir / "run_meta.json").read_text())
    assert meta[-1]["status"] == "interrupted"

    monkeypatch.setattr(ex, "write_atomic", real_write)
    entry = run(cfg)
    assert entry["chunks_written"] == N_CONDITIONS_3 + 1 - 4
    pd.testing.assert_frame_equal(consolidated(cfg.run_dir), consolidated(reference_cfg.run_dir))


def test_cli_interrupt_returns_130(fake_visa_root, tmp_path, monkeypatch, capsys):
    def crash(*a, **k):
        raise KeyboardInterrupt

    monkeypatch.setattr(ex, "run_job", crash)
    rc = main(
        [
            "--data-root",
            str(fake_visa_root),
            "--device",
            "cpu",
            "--out",
            str(tmp_path),
            "--methods",
            "dummy",
        ]  # fmt: skip
    )
    assert rc == 130 and "resume" in capsys.readouterr().out


def test_stale_tmp_file_is_not_counted_as_done(fake_visa_root, tmp_path):
    cfg = cfg_for(fake_visa_root, tmp_path / "o", classes=["candle"], protocols=["official"])
    stale = cfg.run_dir / "chunks/official/dummy/seed0/candle/none_s0.parquet.tmp"
    stale.parent.mkdir(parents=True)
    stale.write_bytes(b"half-written")
    assert run(cfg)["chunks_written"] == 1


def test_results_are_deterministic_across_runs(fake_visa_root, tmp_path):
    a = cfg_for(fake_visa_root, tmp_path / "a", classes=["pcb1"])
    b = cfg_for(fake_visa_root, tmp_path / "b", classes=["pcb1"])
    run(a)
    run(b)
    pd.testing.assert_frame_equal(consolidated(a.run_dir), consolidated(b.run_dir))


def test_adding_severities_later_computes_only_new_levels(fake_visa_root, tmp_path):
    cfg3 = cfg_for(fake_visa_root, tmp_path / "o", classes=["candle"], protocols=["calib"])
    run(cfg3)
    before = {p: (cfg3.run_dir / p).read_bytes() for p in chunks(cfg3.run_dir)}
    cfg5 = cfg_for(
        fake_visa_root, tmp_path / "o", classes=["candle"], protocols=["calib"], n_severities=5
    )
    entry = run(cfg5)
    assert entry["chunks_written"] == 4 * 2  # levels 2 and 4 for each of the 4 families
    after = chunks(cfg5.run_dir)
    assert {p for p in after if p not in before} == {
        f"chunks/calib/dummy/seed0/candle/{c}_s{lv}.parquet"
        for c in ("brightness", "gaussian_blur", "gaussian_noise", "jpeg_compression")
        for lv in (2, 4)
    }
    assert all((cfg5.run_dir / p).read_bytes() == b for p, b in before.items())  # untouched


def test_changed_settings_refuse_to_resume(fake_visa_root, tmp_path, capsys):
    run(cfg_for(fake_visa_root, tmp_path, classes=["candle"], protocols=["official"]))
    with pytest.raises(RuntimeError, match="mix two experiments"):
        run(cfg_for(fake_visa_root, tmp_path, classes=["candle"], memory_fraction=0.6))
    args = ["--data-root", str(fake_visa_root), "--device", "cpu", "--out", str(tmp_path),
            "--methods", "dummy"]  # fmt: skip
    assert main([*args, "--image-size", "128", "--classes", "candle"]) == 2
    assert "mix two experiments" in capsys.readouterr().err


def test_max_images_smoke_mode(fake_visa_root, tmp_path):
    cfg = cfg_for(fake_visa_root, tmp_path, max_images=3, classes=["candle"])
    run(cfg)
    assert cfg.run_dir.name == "smoke_max3"
    df = consolidated(cfg.run_dir)
    sizes = df.groupby(["corruption", "severity", "role"]).size()
    assert (sizes == 3).all()  # 3 held-out normals + 3 anomalies per condition
    manifest = json.loads((cfg.run_dir / "splits/calib/seed0/candle.json").read_text())
    assert len(manifest["memory"]) == 3 and manifest["max_images"] == 3


def test_estimate_only_writes_no_scores(fake_visa_root, tmp_path, capsys):
    cfg = cfg_for(fake_visa_root, tmp_path, estimate_only=True)
    entry = run(cfg)
    assert entry["status"] == "estimate only" and not (cfg.run_dir / "chunks").exists()
    assert "estimated time" in capsys.readouterr().out
    assert entry["estimate"]["image_condition_units"] == 2 * N_CONDITIONS_3 * 14 + 2 * 8


def test_config_file_and_cli_override(fake_visa_root, tmp_path):
    conf = tmp_path / "c.json"
    conf.write_text(
        json.dumps(
            {
                "data_root": str(fake_visa_root),
                "device": "cpu",
                "seeds": [1, 2],
                "classes": ["candle"],
                "batch_size": 4,
            }
        )  # fmt: skip
    )
    cfg = config_from_args(["--config", str(conf), "--batch-size", "2", "--severities", "5"])
    assert cfg.seeds == [1, 2] and cfg.batch_size == 2 and cfg.levels == (1, 2, 3, 4, 5)
    assert cfg.data_root == fake_visa_root
    cfg = config_from_args(["--config", str(conf), "--severity-levels", "2", "4"])
    assert cfg.levels == (2, 4)


def test_required_and_unknown_options(fake_visa_root, tmp_path):
    with pytest.raises(SystemExit, match="device"):
        config_from_args(["--data-root", str(fake_visa_root)])
    with pytest.raises(SystemExit, match="data-root"):
        config_from_args(["--device", "cpu"])
    conf = tmp_path / "c.json"
    conf.write_text(json.dumps({"data_root": "x", "device": "cpu", "typo_key": 1}))
    with pytest.raises(SystemExit, match="typo_key"):
        config_from_args(["--config", str(conf)])


def test_device_must_be_valid(fake_visa_root, tmp_path):
    with pytest.raises(ValueError, match="--device"):
        run(cfg_for(fake_visa_root, tmp_path, device="gpu0"))


def test_cuda_without_cuda_is_an_error_not_a_silent_cpu_fallback(fake_visa_root, tmp_path):
    if importlib.util.find_spec("torch") is None:
        msg = "needs torch"
    else:
        import torch

        if torch.cuda.is_available():
            pytest.skip("CUDA is available here")
        msg = "CUDA is not available"
    with pytest.raises(RuntimeError, match=msg):
        run(cfg_for(fake_visa_root, tmp_path, device="cuda"))


@pytest.mark.parametrize(
    "kw, msg",
    [
        ({"methods": ["nope"]}, "unknown method"),
        ({"classes": ["bottle"]}, "not in the split file"),
        ({"batch_size": 0}, "batch_size"),
        ({"protocols": ["x"]}, "protocols"),
    ],
)
def test_bad_options_are_rejected(fake_visa_root, tmp_path, kw, msg):
    with pytest.raises(ValueError, match=msg):
        run(cfg_for(fake_visa_root, tmp_path, **kw))


def test_threaded_loading_gives_same_scores(fake_visa_root, tmp_path):
    a = cfg_for(fake_visa_root, tmp_path / "a", classes=["candle"], protocols=["official"])
    b = cfg_for(
        fake_visa_root, tmp_path / "b", classes=["candle"], protocols=["official"], num_workers=2
    )
    run(a)
    run(b)
    name = "official_dummy_seed0.parquet"
    pd.testing.assert_frame_equal(consolidated(a.run_dir, name), consolidated(b.run_dir, name))


def test_script_wrapper_runs(fake_visa_root, tmp_path, repo_root):
    import subprocess
    import sys

    res = subprocess.run(
        [
            sys.executable,
            str(repo_root / "scripts" / "extract_scores.py"),
            "--data-root",
            str(fake_visa_root),
            "--device",
            "cpu",
            "--out",
            str(tmp_path),
            "--classes",
            "candle",
            "--max-images",
            "2",
            "--seeds",
            "0",
            "--no-progress",
            "--image-size",
            "64",
            "--methods",
            "dummy",
        ],
        capture_output=True,
        text=True,
    )
    assert res.returncode == 0, res.stderr
    assert "estimated time" in res.stdout and "done:" in res.stdout


def test_jobs_run_class_by_class(fake_visa_root):
    from anomcal.extract import plan_jobs

    cfg = cfg_for(fake_visa_root, "unused", seeds=[0, 1])
    order = [j.cls for j in plan_jobs(cfg, ["candle", "pcb1"])]
    assert order == sorted(order) and order[0] == "candle"  # all candle jobs, then pcb1


def test_each_image_is_decoded_once_per_run(fake_visa_root, tmp_path, monkeypatch):
    """3 seeds x 2 protocols must not re-read the same JPEGs (the per-class cache)."""
    from collections import Counter

    loads: Counter = Counter()
    real = ex.load_images

    def counting(paths, size, workers):
        loads.update(str(p) for p in paths)
        return real(paths, size, workers)

    monkeypatch.setattr(ex, "load_images", counting)
    cfg = cfg_for(fake_visa_root, tmp_path, seeds=[0, 1, 2], warmup_images=2)
    run(cfg)
    n_images = 2 * (12 + 4 + 4)
    assert len(loads) == n_images  # every image was needed by some job
    # each image once in the run, plus at most the 2 warm-up images a second time
    assert sum(loads.values()) <= n_images + cfg.warmup_images


def test_parse_opts_and_cli_merge(fake_visa_root, tmp_path):
    from anomcal.extract import parse_opts

    assert parse_opts(["patchcore.coreset_ratio=0.05", "patchcore.amp=false", "x.y=abc"]) == {
        "patchcore": {"coreset_ratio": 0.05, "amp": False},
        "x": {"y": "abc"},
    }
    for bad in ("patchcore", "patchcore=1", ".k=1", "m.=1"):
        with pytest.raises(SystemExit):
            parse_opts([bad])
    conf = tmp_path / "c.json"
    conf.write_text(
        json.dumps(
            {
                "data_root": str(fake_visa_root),
                "device": "cpu",
                "method_options": {"patchcore": {"coreset_ratio": 0.2, "embed_dim": 8}},
            }
        )
    )
    cfg = config_from_args(["--config", str(conf), "--opt", "patchcore.coreset_ratio=0.05"])
    assert cfg.method_options == {"patchcore": {"coreset_ratio": 0.05, "embed_dim": 8}}


def test_method_options_are_validated_before_any_work(fake_visa_root, tmp_path):
    with pytest.raises(ValueError, match="unknown dummy option"):
        run(cfg_for(fake_visa_root, tmp_path, method_options={"dummy": {"thumb": 8}}))
    with pytest.raises(ValueError, match="not in this run"):
        run(cfg_for(fake_visa_root, tmp_path, method_options={"patchcore": {"embed_dim": 8}}))
    assert not (tmp_path / "full" / "chunks").exists()


def test_changed_method_settings_refuse_to_resume(fake_visa_root, tmp_path):
    kw = dict(classes=["candle"], protocols=["official"])
    run(cfg_for(fake_visa_root, tmp_path, **kw))
    with pytest.raises(RuntimeError, match="settings changed"):
        run(cfg_for(fake_visa_root, tmp_path, method_options={"dummy": {"thumbnail": 8}}, **kw))


def test_job_order_is_class_protocol_seed_method(fake_visa_root):
    from anomcal.extract import plan_jobs

    cfg = cfg_for(fake_visa_root, "unused", seeds=[0, 1], methods=["dummy", "baseline"])
    jobs = plan_jobs(cfg, ["candle", "pcb1"])
    keys = [(j.cls, j.protocol, j.seed, j.method) for j in jobs]
    assert keys[:4] == [
        ("candle", "calib", 0, "dummy"),
        ("candle", "calib", 0, "baseline"),
        ("candle", "calib", 1, "dummy"),
        ("candle", "calib", 1, "baseline"),
    ]


def test_jobs_are_recorded_with_timings(full_run):
    cfg, entry = full_run
    jobs = entry["jobs"]
    assert len(jobs) == 2 * 2  # 2 classes x 2 protocols (1 seed, 1 method)
    j = jobs[0]
    assert {"load", "corrupt", "build_model", "fit", "score", "total"} <= set(j["seconds"])
    assert j["n_eval"] == 14 and j["scorer"]["method"] == "dummy"
    meta = json.loads((cfg.run_dir / "run_meta.json").read_text())[-1]
    assert len(meta["jobs"]) == 4 and meta["method_settings"]["dummy"] == {"thumbnail": 16}


@pytest.mark.parametrize("name", ["gpu_smoke", "gpu_one_class", "full_run", "smoke_dummy"])
def test_shipped_configs_are_valid(repo_root, name):
    from anomcal.corruptions import levels_for_count
    from anomcal.scorers import AVAILABLE_METHODS, settings_dict

    cfg = config_from_args(["--config", str(repo_root / "configs" / f"{name}.json")])
    assert set(cfg.methods) <= set(AVAILABLE_METHODS)
    for m in cfg.methods:
        settings_dict(m, cfg.method_options.get(m))
    assert cfg.levels == levels_for_count(3)
    if name == "gpu_one_class":  # its chunks must land in the same folder as the full run
        full = config_from_args(["--config", str(repo_root / "configs" / "full_run.json")])
        assert cfg.run_dir == full.run_dir and cfg.max_images is None
