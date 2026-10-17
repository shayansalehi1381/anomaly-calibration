#!/usr/bin/env python
"""Extrapolate the full laptop run (time and peak VRAM) from a measured one-class run.

Logic in :mod:`anomcal.estimate`. Example (PowerShell)::

    python scripts\\estimate_full_run.py --run-dir outputs\\extract\\full
"""

import sys

from anomcal.estimate import main

if __name__ == "__main__":
    sys.exit(main())
