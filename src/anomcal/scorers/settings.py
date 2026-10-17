"""Settings of every method, importable without torch.

Each method's settings are a frozen dataclass with documented defaults. Any of them
can be overridden per run (``method_options`` in the config file, or
``--opt patchcore.coreset_ratio=0.05`` on the command line). The resolved settings
are part of the output folder's fingerprint, because they change the scores.

The design choices behind the defaults are explained in docs/MODELS.md.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass


@dataclass(frozen=True)
class DummySettings:
    thumbnail: int = 16


@dataclass(frozen=True)
class BaselineSettings:
    backbone: str = "wide_resnet50_2"
    pretrained: bool = True
    image_size: int = 224  # canonical 256 px image resized to this before the backbone
    knn_k: int = 1  # score = mean distance to the k nearest memory images
    amp: bool = True  # float16 autocast for the backbone on CUDA (features cast back to fp32)


@dataclass(frozen=True)
class PatchCoreSettings:
    backbone: str = "wide_resnet50_2"
    pretrained: bool = True
    image_size: int = 224  # 224 px -> 28 x 28 patches from layer2
    neighbourhood: int = 3  # local average pooling (p = 3) of layer2 / layer3 features
    embed_dim: int = 1024  # 512 + 1024 channels average-pooled down to this dimension
    coreset_ratio: float = 0.1  # fraction of memory patches kept by greedy k-center
    proj_dim: int = 128  # random projection used only for coreset selection
    n_candidates: int = 8  # shortlist size for the exact nearest-neighbour re-rank
    chunk_elems: int = 1 << 26  # VRAM budget of one distance block (floats)
    map_sigma: float = 4.0  # Gaussian smoothing of anomaly maps (px at 256 px)
    amp: bool = True


@dataclass(frozen=True)
class Dinov2KnnSettings:
    backbone: str = "vit_small_patch14_dinov2.lvd142m"
    pretrained: bool = True
    image_size: int = 224  # must be divisible by 14 -> 16 x 16 patches
    top_fraction: float = 0.01  # image score = mean of the top 1% patch distances
    coreset_ratio: float = 1.0  # 1.0 = keep every memory patch (no coreset)
    proj_dim: int = 128
    n_candidates: int = 8
    chunk_elems: int = 1 << 26
    map_sigma: float = 4.0
    amp: bool = True


SETTINGS = {
    "dummy": DummySettings,
    "baseline": BaselineSettings,
    "patchcore": PatchCoreSettings,
    "dinov2_knn": Dinov2KnnSettings,
}


def resolve_settings(method: str, options: dict | None = None):
    """Defaults of ``method`` overridden by ``options``; unknown keys or wrong types raise."""
    if method not in SETTINGS:
        raise KeyError(f"unknown method {method!r}; available: {tuple(SETTINGS)}")
    cls = SETTINGS[method]
    options = dict(options or {})
    fields = {f.name: f for f in dataclasses.fields(cls)}
    unknown = set(options) - set(fields)
    if unknown:
        raise ValueError(f"unknown {method} option(s) {sorted(unknown)}; known: {sorted(fields)}")
    for key, value in options.items():
        default = getattr(cls, key) if hasattr(cls, key) else fields[key].default
        if isinstance(default, bool) and not isinstance(value, bool):
            raise ValueError(f"{method}.{key} must be true/false, got {value!r}")
        if isinstance(default, (int, float)) and not isinstance(default, bool):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{method}.{key} must be a number, got {value!r}")
            if isinstance(default, int) and not isinstance(default, bool) and value != int(value):
                raise ValueError(f"{method}.{key} must be an integer, got {value!r}")
            options[key] = type(default)(value)
    return cls(**options)


def settings_dict(method: str, options: dict | None = None) -> dict:
    return dataclasses.asdict(resolve_settings(method, options))
