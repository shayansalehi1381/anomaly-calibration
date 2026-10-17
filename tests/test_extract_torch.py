"""End-to-end extraction with the real torch methods on fake VisA (tiny random backbones)."""

import json

import pandas as pd
import pytest

pytest.importorskip("torch")

from anomcal.extract import ExtractConfig, run  # noqa: E402

TINY_OPTS = {
    "baseline": {"backbone": "tiny_cnn", "pretrained": False, "image_size": 64},
    "patchcore": {
        "backbone": "tiny_cnn",
        "pretrained": False,
        "image_size": 64,
        "embed_dim": 16,
        "coreset_ratio": 0.5,
        "proj_dim": 8,
    },  # fmt: skip
    "dinov2_knn": {"backbone": "tiny_vit", "pretrained": False, "image_size": 56},
}


@pytest.fixture(scope="module")
def torch_run(fake_visa_root, tmp_path_factory):
    cfg = ExtractConfig(
        data_root=fake_visa_root,
        device="cpu",
        out_dir=tmp_path_factory.mktemp("out"),
        seeds=[0],
        classes=["candle"],
        methods=list(TINY_OPTS),
        method_options=TINY_OPTS,
        image_size=64,
        corruptions=["gaussian_blur"],
        n_severities=1,
        no_progress=True,
        pixel_metrics_protocols=["official", "calib"],
    )
    return cfg, run(cfg)


def test_all_methods_produce_valid_scores(torch_run):
    cfg, entry = torch_run
    assert entry["status"] == "complete"
    for method in TINY_OPTS:
        df = pd.read_parquet(cfg.run_dir / "scores" / f"calib_{method}_seed0.parquet")
        assert set(df["corruption"]) == {"none", "gaussian_blur"}
        assert df["score"].notna().all() and len(df) == 2 * 14


def test_pixel_metrics_only_for_methods_with_maps(torch_run):
    cfg, _ = torch_run
    pm = pd.read_csv(cfg.run_dir / "scores" / "pixel_metrics.csv")
    assert set(pm["method"]) == {"patchcore", "dinov2_knn"}  # the baseline has no maps
    assert set(pm["protocol"]) == {"official", "calib"}
    assert pm["pixel_auroc"].between(0, 1).all() and pm["pro_auc_0.3"].between(0, 1).all()
    assert (pm["map_size"] == 64).all() and (pm["n_anomalous"] == 4).all()
    one = json.loads(
        (cfg.run_dir / "pixel_metrics/official/patchcore/seed0/candle.json").read_text()
    )
    assert one["n_images"] == 8  # official test: 4 normal + 4 anomalous


def test_job_records_include_scorer_details(torch_run):
    _, entry = torch_run
    pc = [j for j in entry["jobs"] if j["method"] == "patchcore" and j["protocol"] == "calib"][0]
    assert pc["scorer"]["bank_size"] == round(0.5 * 6 * 8 * 8)  # 6 memory imgs x 8x8 patches
    assert "fit_coreset" in pc["scorer"]["seconds"] and "pixel_metrics" in pc
    assert pc["scorer"]["settings"]["embed_dim"] == 16


def test_estimate_reports_every_method(torch_run):
    _, entry = torch_run
    est = entry["estimate"]
    assert set(est["per_image_seconds"]["score"]) == set(TINY_OPTS)
    assert set(est["model_rebuild_seconds"]) == set(TINY_OPTS)


def test_corrupted_images_are_computed_once_for_all_methods(fake_visa_root, tmp_path, monkeypatch):
    """Two methods on the same (class, protocol, seed): each condition is corrupted once."""
    import anomcal.extract as ex

    calls = []
    real = ex.apply_corruption_batch

    def counting(images, ids, name, level):
        calls.append((name, level, len(ids)))
        return real(images, ids, name, level)

    monkeypatch.setattr(ex, "apply_corruption_batch", counting)
    cfg = ExtractConfig(
        data_root=fake_visa_root, device="cpu", out_dir=tmp_path, seeds=[0], classes=["candle"],
        methods=["dummy", "patchcore"], method_options={"patchcore": TINY_OPTS["patchcore"]},
        image_size=64, protocols=["calib"], n_severities=1, no_progress=True,
    )  # fmt: skip
    run(cfg)
    in_run = [c for c in calls if c[2] == 14]  # the warm-up uses fewer images
    assert len(in_run) == 1 + 4  # clean + 4 families x 1 level, not twice
