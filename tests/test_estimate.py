"""The full-run estimator recovers known per-unit costs from synthetic job records."""

import json

import pytest

from anomcal.corruptions import conditions, levels_for_count
from anomcal.data.splits import make_split
from anomcal.data.visa import EXPECTED_1CLS_COUNTS, records_from_counts
from anomcal.estimate import fit_cost_model, load_jobs, main, predict

# "True" costs used to fabricate measurements for the TEST ONLY (they are not real timings).
FEAT, PAIR, S0, S1, BUILD, LOAD, CORR = 0.01, 1e-9, 0.004, 1e-7, 2.0, 0.0165, 0.003
PATCHES, RATIO = 784, 0.1
CONDS = conditions(
    ["brightness", "gaussian_blur", "gaussian_noise", "jpeg_compression"], levels_for_count(3)
)
RECORDS = records_from_counts(EXPECTED_1CLS_COUNTS)


def fake_job(cls, protocol, seed=0, first=True):
    split = make_split(RECORDS, cls, protocol, seed)
    n_mem, n_eval = len(split.memory), len(split.eval_ids)
    n_cond = len(CONDS) if protocol == "calib" else 1
    patches = n_mem * PATCHES
    bank = RATIO * patches
    return {
        "protocol": protocol, "method": "patchcore", "seed": seed, "class": cls,
        "n_memory": n_mem, "n_eval": n_eval, "conditions": n_cond,
        "seconds": {
            "load": LOAD * (n_mem + n_eval) if first else 0.0,
            "corrupt": CORR * n_eval * n_cond,
            "build_model": BUILD, "fit": 0.0,
            "score": (S0 + S1 * bank) * n_eval * n_cond, "pixel_metrics": 0.0, "total": 0.0,
        },
        "scorer": {
            "method": "patchcore", "peak_vram_mb": 1000.0 * bank / 35000,
            "n_memory_patches": patches, "bank_size": bank,
            "seconds": {"fit_features": FEAT * n_mem, "fit_coreset": PAIR * patches * bank},
        },
    }  # fmt: skip


def test_cost_model_recovers_coefficients():
    jobs = [fake_job("cashew", "calib"), fake_job("cashew", "official", first=False)]
    m = fit_cost_model(jobs)["patchcore"]
    assert m["feat_per_image"] == pytest.approx(FEAT)
    assert m["coreset_per_pair"] == pytest.approx(PAIR)
    assert m["score_s0"] == pytest.approx(S0, rel=1e-6)
    assert m["score_s1"] == pytest.approx(S1, rel=1e-6)
    assert m["build"] == BUILD and m["patches_per_image"] == PATCHES


def test_prediction_matches_the_generating_costs():
    """Measured on cashew only; the prediction for candle equals the true synthetic cost."""
    jobs = [fake_job("cashew", "calib"), fake_job("cashew", "official", first=False)]
    models = fit_cost_model(jobs)
    res = predict(models, ["candle"], [0], ["patchcore"], ["calib", "official"], CONDS)
    expected = 0.0
    for proto in ("calib", "official"):
        j = fake_job("candle", proto)
        s = j["scorer"]["seconds"]
        expected += BUILD + s["fit_features"] + s["fit_coreset"] + j["seconds"]["score"]
    assert res["seconds_per_method"]["patchcore"] == pytest.approx(expected, rel=1e-6)
    # VRAM bound scales with the largest bank (candle official: 900 images in memory)
    v = res["peak_vram"]["patchcore"]
    assert v["largest_bank_upper_bound_mb"] == pytest.approx(v["measured_mb"] * 900 / 450)


def test_existing_chunks_are_skipped(tmp_path):
    from anomcal.scores_io import chunk_path

    models = fit_cost_model([fake_job("cashew", "calib"), fake_job("cashew", "official", False)])
    full = predict(models, ["cashew"], [0], ["patchcore"], ["official"], CONDS, run_dir=tmp_path)
    p = chunk_path(tmp_path, "official", "patchcore", 0, "cashew", "none", 0)
    p.parent.mkdir(parents=True)
    p.write_bytes(b"x")
    none_left = predict(
        models, ["cashew"], [0], ["patchcore"], ["official"], CONDS, run_dir=tmp_path
    )
    assert full["seconds_per_method"]["patchcore"] > 0
    assert none_left["seconds_per_method"]["patchcore"] == 0 and none_left["io_seconds"] == 0


def test_unmeasured_method_is_an_error():
    models = fit_cost_model([fake_job("cashew", "calib")])
    with pytest.raises(ValueError, match="no measurements"):
        predict(models, ["cashew"], [0], ["dinov2_knn"], ["calib"], CONDS)


def test_cli_end_to_end(tmp_path, capsys):
    config = {"methods": ["patchcore"], "n_severities": 3, "memory_fraction": 0.5}
    jobs = [fake_job("cashew", "calib"), fake_job("cashew", "official", False)]
    meta = [{"config": config, "jobs": jobs}]
    (tmp_path / "run_meta.json").write_text(json.dumps(meta))
    assert main(["--run-dir", str(tmp_path), "--classes", "cashew", "candle"]) == 0
    out = capsys.readouterr().out
    assert "patchcore" in out and "TOTAL" in out and "measured on : cashew" in out
    with pytest.raises(ValueError):
        (tmp_path / "run_meta.json").write_text(json.dumps([{"config": {}}]))
        load_jobs(tmp_path)
