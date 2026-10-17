"""Documents must agree with the committed CSVs (offline, CPU, no network)."""

import importlib.util
import json
import re
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
CHECK = re.compile(r"<!-- check: (.+?) ;; (.+?) ;; (.+?) ;; (.+?) ;; (.+?) -->")


def _load_generator():
    spec = importlib.util.spec_from_file_location(
        "make_results_summary", ROOT / "scripts" / "make_results_summary.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _checks(path):
    return CHECK.findall(path.read_text(encoding="utf-8"))


@pytest.mark.parametrize("doc", ["docs/RESULTS_SUMMARY.md", "README.md"])
def test_numbers_in_docs_match_the_csvs(doc):
    checks = _checks(ROOT / doc)
    if doc == "docs/RESULTS_SUMMARY.md":
        assert len(checks) > 100  # the summary is generated with a check on every key number
    for file, col, sel, value, tol in checks:
        df = pd.read_csv(RESULTS / file)
        for item in sel.split(","):
            k, v = item.split("=", 1)
            if pd.api.types.is_numeric_dtype(df[k]):
                df = df[(df[k] - float(v)).abs() < 1e-9]
            else:
                df = df[df[k].astype(str) == v]
        assert len(df) == 1, (doc, file, sel)
        assert abs(float(df.iloc[0][col]) - float(value)) <= float(tol), (doc, file, col, sel)


def test_results_summary_is_up_to_date(tmp_path):
    out = tmp_path / "summary.md"
    _load_generator().main(out)
    assert out.read_text(encoding="utf-8") == (ROOT / "docs" / "RESULTS_SUMMARY.md").read_text(
        encoding="utf-8"
    ), "docs/RESULTS_SUMMARY.md is stale: run python scripts/make_results_summary.py"


def test_notebook_code_cells_run(tmp_path, monkeypatch):
    """The stored notebook's code cells execute on the committed CSVs (CPU, no network)."""
    pytest.importorskip("matplotlib")
    monkeypatch.setenv("MPLBACKEND", "Agg")
    monkeypatch.chdir(ROOT / "notebooks")
    nb = json.loads((ROOT / "notebooks" / "results.ipynb").read_text(encoding="utf-8"))
    ns: dict = {}
    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            exec("".join(cell["source"]), ns)  # noqa: S102
    assert "pr" in ns and len(ns["pr"]) > 0


def test_stored_notebook_is_small():
    assert (ROOT / "notebooks" / "results.ipynb").stat().st_size < 1_500_000


def test_roadmap_marks_all_phases_done():
    text = (ROOT / "ROADMAP.md").read_text(encoding="utf-8")
    for phase in range(6):
        line = next(line for line in text.splitlines() if line.startswith(f"## Phase {phase}:"))
        assert "☑" in line, line
    assert "pending" not in (ROOT / "README.md").read_text(encoding="utf-8").lower()
