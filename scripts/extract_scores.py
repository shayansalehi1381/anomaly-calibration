#!/usr/bin/env python
"""Laptop entry point: score VisA images (clean + corrupted) for each method and seed.

All logic lives in :mod:`anomcal.extract` (tested offline in CI); this file only makes
it runnable as a script. Examples (Windows PowerShell, from the repository root)::

    # dataset check + smoke test on one class, CPU, dummy scorer
    python scripts/extract_scores.py --data-root data/visa --device cpu `
        --methods dummy --classes candle --max-images 20

    # only print the runtime estimate for the full run
    python scripts/extract_scores.py --config configs/full_run.json --estimate-only

Rerunning the same command resumes after an interruption. See docs/score_schema.md.
"""

import sys

from anomcal.extract import main

if __name__ == "__main__":
    sys.exit(main())
