#!/usr/bin/env python
"""Print clean image-AUROC tables and the SPARC sanity check (logic: anomcal.report).

python scripts\\auroc_table.py outputs\\extract\\full\\scores
"""

import sys

from anomcal.report import main

if __name__ == "__main__":
    sys.exit(main())
