"""Generic statistical helpers shared across the tool modules.

Kept dependency-light (numpy only) and free of any tool/IO import so both the
change-detection plane (``src/tools/change.py``) and the climate plane
(``src/tools/climate.py``) can share it without cross-tool coupling (audit R2).
"""

from __future__ import annotations

import numpy as np


def percentile_thresholds(values: list[float], *, level: int) -> dict:
    """P10/P50/P90 of a numeric series + the threshold at the requested ``level``.

    Pure, numpy-linear interpolation. ``level`` is honoured directly (not limited to
    {10, 50, 90}); p10/p50/p90 are always returned as reference context. This is a generic
    percentile routine, agnostic to its input: it is reused both by the observed-NDVI-series
    path (``vegetation_stress``, FIERCE B6) and by the climatological-normals path
    (``climate_percentiles``).
    """
    arr = np.asarray(values, dtype=float)
    return {
        "p10": round(float(np.percentile(arr, 10)), 4),
        "p50": round(float(np.percentile(arr, 50)), 4),
        "p90": round(float(np.percentile(arr, 90)), 4),
        "selected_level": level,
        "selected_threshold": round(float(np.percentile(arr, level)), 4),
    }
