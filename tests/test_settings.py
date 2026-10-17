"""Method settings resolve without torch (needed for fingerprints on the CPU side)."""

import pytest

from anomcal.scorers import AVAILABLE_METHODS, TORCH_METHODS, resolve_settings, settings_dict


def test_every_method_has_settings():
    for m in AVAILABLE_METHODS:
        assert isinstance(settings_dict(m), dict)
    assert set(TORCH_METHODS) < set(AVAILABLE_METHODS)


def test_defaults_document_the_design():
    pc = resolve_settings("patchcore")
    assert (pc.backbone, pc.image_size, pc.embed_dim, pc.coreset_ratio) == (
        "wide_resnet50_2",
        224,
        1024,
        0.1,
    )
    dino = resolve_settings("dinov2_knn")
    assert dino.backbone == "vit_small_patch14_dinov2.lvd142m" and dino.image_size % 14 == 0


def test_overrides_and_type_coercion():
    s = resolve_settings("patchcore", {"coreset_ratio": 0.05, "image_size": 196.0})
    assert s.coreset_ratio == 0.05 and s.image_size == 196 and isinstance(s.image_size, int)


@pytest.mark.parametrize(
    "opts, msg",
    [
        ({"coreset": 0.1}, "unknown"),
        ({"image_size": 224.5}, "integer"),
        ({"amp": 1}, "true/false"),
        ({"coreset_ratio": "0.1"}, "number"),
    ],
)
def test_bad_options_rejected(opts, msg):
    with pytest.raises(ValueError, match=msg):
        resolve_settings("patchcore", opts)


def test_unknown_method():
    with pytest.raises(KeyError):
        resolve_settings("padim")
