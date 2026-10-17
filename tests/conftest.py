"""Shared test configuration. Every test runs offline: no dataset, no GPU, no network.

Markers (see pyproject.toml):

- ``gpu``: needs torch; skipped automatically when torch is not importable.
- ``integration``: needs the real VisA dataset; runs only when the environment
  variable ``ANOMCAL_VISA_ROOT`` points to it (i.e. on the laptop, never in CI).
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def pytest_collection_modifyitems(config, items):
    has_torch = importlib.util.find_spec("torch") is not None
    has_visa = bool(os.environ.get("ANOMCAL_VISA_ROOT"))
    skip_gpu = pytest.mark.skip(reason="needs torch (pip install -e .[gpu])")
    skip_int = pytest.mark.skip(reason="needs real VisA: set ANOMCAL_VISA_ROOT")
    for item in items:
        if "gpu" in item.keywords and not has_torch:
            item.add_marker(skip_gpu)
        if "integration" in item.keywords and not has_visa:
            item.add_marker(skip_int)


@pytest.fixture
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture(scope="session")
def fake_visa_root(tmp_path_factory) -> Path:
    """A tiny synthetic VisA tree (2 classes, 12 train / 4 test normals / 4 anomalies)."""
    from anomcal.data.fake_visa import make_fake_visa

    return make_fake_visa(tmp_path_factory.mktemp("fake_visa"))


@pytest.fixture(scope="session")
def fake_records(fake_visa_root):
    from anomcal.data.visa import read_split_csv

    return read_split_csv(fake_visa_root / "split_csv" / "1cls.csv")
