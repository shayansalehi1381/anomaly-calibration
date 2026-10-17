import shutil
import subprocess

import pytest

from anomcal.repo_guard import MAX_FILE_BYTES, check_files, main, tracked_files

OK_FILES = [
    ("README.md", 1_000),
    ("src/anomcal/seeding.py", 2_000),
    ("results/scores/patchcore_seed0.parquet", 300_000),
    ("docs/figures/far_vs_severity.png", 80_000),
    ("results/figures/heatmap.png", 80_000),
    ("data/.gitkeep", 0),
]


def test_clean_listing_passes():
    assert check_files(OK_FILES) == []


@pytest.mark.parametrize(
    "path",
    [
        "data/visa/candle/Data/Images/Normal/0000.JPG",  # dataset image
        "data/visa/split_csv/1cls.csv",  # anything under data/
        "VisA_20220922.tar",  # archive
        "checkpoints/patchcore.pt",  # weights
        "model.safetensors",
        "cache/features_candle.npz",  # feature cache
        "results/anomaly_map_0001.png",  # image outside a figure dir
        "notebooks/peek.jpeg",
    ],
)
def test_forbidden_files_are_flagged(path):
    assert check_files([(path, 10)]), path


def test_suffix_check_is_case_insensitive():
    assert check_files([("x/IMG.JPG", 10)])
    assert check_files([("x/W.PT", 10)])


def test_large_file_is_flagged_even_if_type_is_allowed():
    v = check_files([("results/scores/huge.parquet", MAX_FILE_BYTES + 1)])
    assert len(v) == 1 and "limit" in v[0].reason


def test_file_at_limit_passes():
    assert check_files([("results/scores/ok.parquet", MAX_FILE_BYTES)]) == []


needs_git = pytest.mark.skipif(shutil.which("git") is None, reason="git not installed")


@needs_git
def test_this_repository_passes_the_guard(repo_root):
    """The actual tracked files of this repo obey the rules (also run as a CI step)."""
    assert check_files(tracked_files(repo_root)) == []
    assert main([str(repo_root)]) == 0


@needs_git
@pytest.mark.parametrize(
    "path",
    [
        "data/visa/candle/Data/Images/Normal/0000.JPG",
        "VisA_20220922.tar",
        "weights/patchcore.pt",
        "cache/features.npz",
        "outputs/run1/scores.parquet",
    ],
)
def test_gitignore_blocks_data_and_weights(repo_root, path):
    res = subprocess.run(["git", "check-ignore", "-q", path], cwd=repo_root)
    assert res.returncode == 0, f"{path} is not ignored by .gitignore"


@needs_git
@pytest.mark.parametrize("path", ["src/anomcal/data/visa.py", "tests/data/test_x.py"])
def test_gitignore_does_not_hide_code_folders_named_data(repo_root, path):
    """Regression: an unanchored `data/` pattern used to ignore src/anomcal/data/."""
    res = subprocess.run(["git", "check-ignore", "-q", path], cwd=repo_root)
    assert res.returncode == 1, f"{path} is wrongly ignored"


@needs_git
def test_gitignore_allows_results(repo_root):
    res = subprocess.run(
        ["git", "check-ignore", "-q", "results/scores/patchcore_seed0.parquet"], cwd=repo_root
    )
    assert res.returncode == 1, "curated results must be committable"
