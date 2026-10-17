"""Backbone output structure. Real architectures are built with random weights (no download)."""

import pytest

torch = pytest.importorskip("torch")

from anomcal.models.backbones import build_cnn, build_vit  # noqa: E402


def test_tiny_cnn_structure():
    m = build_cnn("tiny_cnn", pretrained=False)
    out = m(torch.zeros(2, 3, 64, 64), need_pooled=True)
    assert out["layer2"].shape[-2:] == (8, 8) and out["layer3"].shape[-2:] == (4, 4)
    assert out["pooled"].shape == (2, 20)
    assert "pooled" not in m(torch.zeros(1, 3, 64, 64))


def test_tiny_models_are_seeded_and_do_not_touch_global_rng():
    torch.manual_seed(123)
    before = torch.rand(1)
    torch.manual_seed(123)
    a = build_cnn("tiny_cnn", False)
    b = build_cnn("tiny_cnn", False)
    assert torch.rand(1) == before  # global RNG unchanged by model construction
    for pa, pb in zip(a.parameters(), b.parameters(), strict=True):
        assert torch.equal(pa, pb)


def test_wide_resnet50_real_architecture_shapes():
    m = build_cnn("wide_resnet50_2", pretrained=False)
    with torch.no_grad():
        out = m(torch.zeros(1, 3, 224, 224), need_pooled=True)
    assert out["layer2"].shape == (1, 512, 28, 28)
    assert out["layer3"].shape == (1, 1024, 14, 14)
    assert out["pooled"].shape == (1, 2048)


def test_dinov2_real_architecture_shapes():
    pytest.importorskip("timm")
    m = build_vit("vit_small_patch14_dinov2.lvd142m", pretrained=False, image_size=224)
    with torch.no_grad():
        tokens, grid = m(torch.zeros(1, 3, 224, 224))
    assert grid == (16, 16) and tokens.shape == (1, 256, 384)


def test_unknown_backbones_and_bad_vit_size():
    with pytest.raises(ValueError):
        build_cnn("resnet18", False)
    with pytest.raises(ValueError):
        build_vit("tiny_vit", False, image_size=100)
    with pytest.raises(ValueError):
        build_vit("vit_huge", False, image_size=224)
