"""Repository guard: fail CI if raw data, weights or large files are tracked by git.

``.gitignore`` prevents most accidents, but ``git add -f`` or a new file type can
still slip through. This check runs in CI on the list of tracked files.

Usage::

    python -m anomcal.repo_guard            # checks `git ls-files` in the cwd
"""

from __future__ import annotations

import subprocess
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

MAX_FILE_BYTES = 5 * 1024 * 1024  # 5 MiB per file; score files are expected to be far smaller

# Never allowed anywhere: archives, model weights, raw feature caches.
FORBIDDEN_SUFFIXES = frozenset(
    {".tar", ".gz", ".tgz", ".zip", ".pt", ".pth", ".ckpt", ".safetensors", ".onnx", ".npz"}
)
# Images are only allowed as figures (dataset images are .JPG/.png in VisA).
IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"})
FIGURE_DIRS = (PurePosixPath("docs/figures"), PurePosixPath("results/figures"))
# Anything under data/ is the dataset by definition (except placeholder files).
DATA_DIR = PurePosixPath("data")
DATA_PLACEHOLDERS = frozenset({".gitkeep", "README.md"})


@dataclass(frozen=True)
class Violation:
    path: str
    reason: str

    def __str__(self) -> str:
        return f"{self.path}: {self.reason}"


def _is_under(path: PurePosixPath, root: PurePosixPath) -> bool:
    return path.parts[: len(root.parts)] == root.parts


def check_files(
    files: Iterable[tuple[str, int]], max_bytes: int = MAX_FILE_BYTES
) -> list[Violation]:
    """Return every rule violation for ``(posix_path, size_in_bytes)`` pairs."""
    violations: list[Violation] = []
    for raw_path, size in files:
        path = PurePosixPath(raw_path)
        suffix = path.suffix.lower()
        if _is_under(path, DATA_DIR) and path.name not in DATA_PLACEHOLDERS:
            violations.append(Violation(raw_path, "file under data/ (dataset must stay local)"))
        if suffix in FORBIDDEN_SUFFIXES:
            violations.append(Violation(raw_path, f"forbidden file type {suffix}"))
        if suffix in IMAGE_SUFFIXES and not any(_is_under(path, d) for d in FIGURE_DIRS):
            violations.append(Violation(raw_path, "image outside docs/figures or results/figures"))
        if size > max_bytes:
            violations.append(Violation(raw_path, f"{size} bytes > limit of {max_bytes}"))
    return violations


def tracked_files(repo_root: Path) -> list[tuple[str, int]]:
    """``git ls-files`` with sizes (paths are POSIX-style on every OS)."""
    out = subprocess.run(
        ["git", "ls-files", "-z"], cwd=repo_root, check=True, capture_output=True
    ).stdout.decode("utf-8")
    names = [n for n in out.split("\0") if n]
    # A tracked file deleted in the working tree has no size; skip it (git will report it).
    return [(n, (repo_root / n).stat().st_size) for n in names if (repo_root / n).exists()]


def main(argv: list[str] | None = None) -> int:
    root = Path(argv[0]) if argv else Path.cwd()
    violations = check_files(tracked_files(root))
    for v in violations:
        print(f"repo_guard: {v}", file=sys.stderr)
    if violations:
        return 1
    print("repo_guard: OK (no data, weights or large files tracked)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
