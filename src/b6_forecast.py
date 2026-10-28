"""Shared exact-byte saved forecast validation for CLI and private B6 transport."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from src.b6_explanatory import finite_float


def parse_saved_forecast(
    raw: bytes, zone_id: str, *, filename: str, label: str
) -> tuple[list[dict[str, Any]], dict[str, Any], float]:
    """Return rows, exact-byte identity and effective horizon without file or provider I/O."""
    try:
        value = json.loads(raw)
    except ValueError as exc:
        raise ValueError(f"{label} is not valid JSON") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must hold one forecast_query result object")
    if value.get("error"):
        raise ValueError(f"saved forecast_query error: {value['error']}")
    # A point-mode forecast has no zone_id; it cannot be shown to belong to this zone.
    if value.get("zone_id") != zone_id:
        raise ValueError(f"forecast zone {value.get('zone_id')!r} is not --zone-id {zone_id!r}")
    rows = value.get("forecast")
    if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"{label} has no forecast rows list")
    # Directions are read first-to-last row, so a reordered file would flip every one.
    dates = [row.get("date") for row in rows]
    if not all(isinstance(d, str) for d in dates) or dates != sorted(set(dates)):
        raise ValueError(f"{label} rows need unique dates in ascending order")
    metadata = value.get("metadata") if isinstance(value.get("metadata"), dict) else {}
    # The provider's reported horizon, else the number of rows present.
    horizon = finite_float(metadata.get("horizon_days"))
    if horizon is not None and horizon <= 0:
        raise ValueError(f"{label} reports a non-positive horizon_days")
    # A horizon longer than the rows would admit lags the forecast cannot cover.
    if horizon is not None and horizon > len(rows):
        raise ValueError(f"{label} horizon_days {horizon:g} exceeds its {len(rows)} forecast rows")
    identity = {
        "file": filename,
        "sha256": hashlib.sha256(raw).hexdigest(),
        "zone_id": value.get("zone_id"),
        "source": metadata.get("source"),
        "provider_chain": metadata.get("provider_chain"),
        "issued_at": metadata.get("issued_at"),
        "horizon_days": metadata.get("horizon_days"),
        "rows": len(rows),
    }
    return rows, identity, horizon if horizon is not None else float(len(rows))
