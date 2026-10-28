"""Shared rendering of a normalized gradual item to its display token.

One renderer for both the ``discover_patterns`` parquet output
(``src/tools/patterns.py``) and the exchange-record producer
(``src/patterns/library_adapter.py``) so the two cannot drift (audit R4). Pure: no
so4gp / duckdb / pandas import, unit-testable in the base runtime.
"""

from __future__ import annotations


def render_gradual_item(gi: dict) -> str:
    """Render one normalized gradual item: 'precip+' (contemporaneous) or
    '(precip+) +7.0 days' (lagged).

    Uses ``gi['direction']`` STRICTLY (fail-loud): a normalized item always carries a
    'direction' key (possibly ``None``, which renders as '?'). A *missing* key signals a
    malformed item and raises ``KeyError`` rather than silently rendering '?' into a
    persisted parquet / exchange record — surfacing the producer bug instead of hiding it.
    """
    base = f"{gi['attribute']}{gi['direction'] or '?'}"
    if gi.get("role") == "lagged" and gi.get("lag_raw"):
        return f"({base}) {gi['lag_raw']}"
    return base
