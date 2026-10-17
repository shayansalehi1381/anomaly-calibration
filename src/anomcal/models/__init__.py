"""Torch building blocks for the real detectors (laptop / GPU side; needs the [gpu] extra).

- ``torch_utils``: preprocessing, exact nearest-neighbour search, greedy coreset,
  Gaussian smoothing of anomaly maps, peak-VRAM bookkeeping.
- ``backbones``: pretrained feature extractors (WRN-50, DINOv2) and tiny random
  stand-ins used by the offline tests (no weight downloads in CI).

Nothing here is imported by the CPU-only part of the package.
"""
