"""Feature extractors: real pretrained backbones and tiny random stand-ins for tests.

Real backbones (downloaded once on the laptop, cached outside the repository):

- ``wide_resnet50_2``: torchvision WRN-50-2, ImageNet weights ``IMAGENET1K_V1`` (the
  ones PatchCore used). We expose the outputs of ``layer2`` (stride 8), ``layer3``
  (stride 16) and the global-average-pooled ``layer4`` (2048-d, for the baseline).
- ``vit_small_patch14_dinov2.lvd142m``: DINOv2 ViT-S/14 via timm. We expose the patch
  tokens (CLS token dropped) after the final norm, on a (size/14) x (size/14) grid.

Test stand-ins (``tiny_cnn``, ``tiny_vit``): a few random layers with the *same output
structure*. Their weights come from a fixed seed, so tests are deterministic and never
download anything. With ``pretrained=False``, the real architectures are also built
locally with random weights (used in CI to check shapes end to end).
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

import torch
from torch import nn

CNN_BACKBONES = ("wide_resnet50_2", "tiny_cnn")
VIT_BACKBONES = ("vit_small_patch14_dinov2.lvd142m", "vit_base_patch14_dinov2.lvd142m", "tiny_vit")


class CNNFeatures(nn.Module):
    """Returns {"layer2": (N,C2,H/8,W/8), "layer3": (N,C3,H/16,W/16), "pooled": (N,C4)}."""

    def __init__(self, stem: nn.Module, layer1, layer2, layer3, layer4) -> None:
        super().__init__()
        self.stem, self.layer1, self.layer2 = stem, layer1, layer2
        self.layer3, self.layer4 = layer3, layer4

    def forward(self, x: torch.Tensor, need_pooled: bool = False) -> dict[str, torch.Tensor]:
        h = self.layer1(self.stem(x))
        l2 = self.layer2(h)
        l3 = self.layer3(l2)
        out = {"layer2": l2, "layer3": l3}
        if need_pooled:  # layer4 is only needed by the baseline; skip it for PatchCore
            out["pooled"] = self.layer4(l3).mean(dim=(2, 3))
        return out


@contextmanager
def _seeded(seed: int) -> Iterator[None]:
    """Initialise tiny test models from a fixed seed without touching the global RNG."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        yield


def build_cnn(name: str, pretrained: bool) -> CNNFeatures:
    if name == "wide_resnet50_2":
        import torchvision

        weights = torchvision.models.Wide_ResNet50_2_Weights.IMAGENET1K_V1 if pretrained else None
        m = torchvision.models.wide_resnet50_2(weights=weights)
        stem = nn.Sequential(m.conv1, m.bn1, m.relu, m.maxpool)
        return CNNFeatures(stem, m.layer1, m.layer2, m.layer3, m.layer4).eval()
    if name == "tiny_cnn":
        with _seeded(0):

            def block(cin, cout, stride):
                return nn.Sequential(nn.Conv2d(cin, cout, 3, stride, 1), nn.ReLU())

            stem = block(3, 8, 4)  # stride 4, like conv1 + maxpool
            return CNNFeatures(
                stem, block(8, 8, 1), block(8, 12, 2), block(12, 16, 2), block(16, 20, 2)
            ).eval()
    raise ValueError(f"unknown CNN backbone {name!r}; available: {CNN_BACKBONES}")


class TinyViT(nn.Module):
    """Mimics timm's ViT interface: forward_features -> (N, 1 + h*w, C), patch size 14."""

    num_prefix_tokens = 1
    patch_size = 14

    def __init__(self, dim: int = 16) -> None:
        super().__init__()
        self.embed = nn.Conv2d(3, dim, kernel_size=14, stride=14)
        self.mix = nn.Linear(dim, dim)
        self.cls = nn.Parameter(torch.zeros(1, 1, dim))
        self.embed_dim = dim

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        t = self.embed(x).flatten(2).transpose(1, 2)  # (N, h*w, C)
        t = t + torch.tanh(self.mix(t))
        return torch.cat([self.cls.expand(x.shape[0], -1, -1), t], dim=1)


class ViTPatchFeatures(nn.Module):
    """Wraps a ViT; returns patch tokens (N, h*w, C) and the grid (h, w)."""

    def __init__(self, vit: nn.Module, patch_size: int) -> None:
        super().__init__()
        self.vit, self.patch_size = vit, patch_size
        self.n_prefix = int(getattr(vit, "num_prefix_tokens", 1))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int]]:
        tokens = self.vit.forward_features(x)[:, self.n_prefix :]
        grid = (x.shape[-2] // self.patch_size, x.shape[-1] // self.patch_size)
        if tokens.shape[1] != grid[0] * grid[1]:
            raise RuntimeError(f"expected {grid[0] * grid[1]} patch tokens, got {tokens.shape[1]}")
        return tokens, grid


def build_vit(name: str, pretrained: bool, image_size: int) -> ViTPatchFeatures:
    if image_size % 14:
        raise ValueError(f"DINOv2 needs an image size divisible by 14, got {image_size}")
    if name == "tiny_vit":
        with _seeded(0):
            return ViTPatchFeatures(TinyViT(), 14).eval()
    if name in VIT_BACKBONES:
        import timm

        # img_size makes timm resample the pretrained position embeddings (trained at 518 px)
        # to our grid, e.g. 16 x 16 patches for 224 px.
        vit = timm.create_model(name, pretrained=pretrained, img_size=image_size)
        return ViTPatchFeatures(vit, 14).eval()
    raise ValueError(f"unknown ViT backbone {name!r}; available: {VIT_BACKBONES}")
