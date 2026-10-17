"""anomcal: conformal false-alarm calibration for industrial anomaly detection.

Layout (filled in phase by phase, see ROADMAP.md):

- ``seeding``      deterministic seeds (global and per-item)
- ``repo_guard``   CI check: no raw data, weights or large files in git
- ``data``         VisA adapter and split protocol            (Phase 1)
- ``corruptions``  deterministic synthetic shifts             (Phase 1)
- ``models``       baseline, PatchCore, DINOv2-kNN            (Phase 2, laptop)
- ``metrics``      AUROC, PRO, FAR, detection rate            (Phase 2)
- ``calibration``  split-conformal thresholds and p-values    (Phase 3)
- ``shift``        shift evaluation and recalibration         (Phase 4)
"""

__version__ = "0.0.1"
