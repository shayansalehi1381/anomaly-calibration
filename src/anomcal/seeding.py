"""Deterministic seeding.

Two kinds of randomness appear in this project:

1. *Global* randomness of an experiment (which normals go into the memory bank,
   coreset sampling, calibration/test partitions). Controlled by one integer seed
   per run, via :func:`set_global_seed` and explicit ``numpy.random.Generator``s.

2. *Per-item* randomness that must not depend on processing order, e.g. the noise
   added to image ``candle/0042`` under ``gaussian_noise`` severity 2. If we drew it
   from one global stream, resuming an interrupted run or changing the batch size
   would change the noise. Instead, :func:`derive_seed` hashes the item's identity
   into its own seed, so each item always gets the same random numbers.

Why not Python's built-in ``hash()``? For strings it is salted per process
(PYTHONHASHSEED), so it gives different values on every run. SHA-256 is stable
across processes, machines and operating systems (cloud Linux vs. laptop Windows).
"""

from __future__ import annotations

import hashlib
import os
import random

import numpy as np

_SEED_BITS = 63  # keep seeds in the non-negative int64 range


def derive_seed(*parts: object) -> int:
    """Map an ordered tuple of identifiers to a stable 63-bit seed.

    >>> derive_seed("candle/0042", "noise", 2) == derive_seed("candle/0042", "noise", 2)
    True

    Parts are joined with an ASCII unit separator, so ("ab", "c") and ("a", "bc")
    give different seeds.
    """
    if not parts:
        raise ValueError("derive_seed needs at least one part")
    key = "\x1f".join(f"{type(p).__name__}:{p}" for p in parts)
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") >> (64 - _SEED_BITS)


def item_rng(*parts: object) -> np.random.Generator:
    """A NumPy generator seeded only by the item's identity (see module docstring)."""
    return np.random.default_rng(derive_seed(*parts))


def set_global_seed(seed: int) -> None:
    """Seed Python, NumPy's legacy global RNG and (if installed) torch.

    New code should prefer explicit ``np.random.default_rng(seed)`` generators;
    this function exists so that third-party code using global RNGs is also fixed.
    """
    if not isinstance(seed, int) or seed < 0:
        raise ValueError(f"seed must be a non-negative int, got {seed!r}")
    random.seed(seed)
    np.random.seed(seed % 2**32)  # legacy API only accepts 32-bit seeds
    os.environ["PYTHONHASHSEED"] = str(seed)  # affects subprocesses only
    try:
        import torch
    except ImportError:  # CPU-only cloud install: torch is optional
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
