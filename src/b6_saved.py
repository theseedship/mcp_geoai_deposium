"""Private B6 adapter: verify a server-owned package and score saved inputs only.

The catalog pins the delivered original package. Adding a version is a code review,
not a request option. The identical relative path is included in the wheel, so the
resolver works from source checkouts and installed packages without duplicate data.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from types import MappingProxyType
from typing import Any

from src.b6_explanatory import load_verified_library, score_level2
from src.b6_forecast import parse_saved_forecast
from src.b6_saved_contracts import (
    MAX_SAVED_ENVELOPE_BYTES,
    MAX_SAVED_RECORDS,
    B6SavedInputs,
    preflight_saved_inputs,
)
from src.contracts import B6Params
from src.patterns.offline_bundle import _strict_json
from src.utils.code_ref import code_ref


@dataclass(frozen=True)
class CatalogEntry:
    package_dir: Path
    manifest_sha256: str


_CATALOG: Mapping[str, CatalogEntry] = MappingProxyType(
    {
        "2026.10.15-native-pilots.1": CatalogEntry(
            Path(__file__).resolve().parent / "data/pattern-library-2026-10-15/package",
            "bfed1a04a176e9fba288fa822b3a02a3f280591c8fd9c4942bb245854cfa47e2",
        ),
    }
)


def score_saved_inputs(zone_id: str, inputs: B6SavedInputs | dict[str, Any]) -> dict[str, Any]:
    """Bind the complete original model and exact forecast bytes before pure scoring.

    The serialized envelope cap bounds semantic processing, not the HTTP request body.
    No path or catalog override is accepted from the tool or its request.
    """
    preflight_saved_inputs(zone_id, inputs)
    inputs = B6SavedInputs.model_validate(inputs)
    encoded = json.dumps(
        {
            "zone_id": zone_id,
            "mode": "saved_inputs_j1",
            "saved_inputs": inputs.model_dump(mode="json"),
        },
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    ).encode("utf-8")
    if len(encoded) > MAX_SAVED_ENVELOPE_BYTES:
        raise ValueError("saved envelope exceeds 128 KiB")
    entry = _CATALOG.get(inputs.library_version)
    if entry is None or inputs.expected_manifest_sha256 != entry.manifest_sha256:
        raise ValueError("unknown original library version or manifest")
    library, identity = load_verified_library(
        entry.package_dir, entry.manifest_sha256, zone_id, inputs.algorithm
    )
    if (
        identity.library_version != inputs.library_version
        or identity.json_sha256 != inputs.expected_json_sha256
        or identity.parquet_sha256 != inputs.expected_parquet_sha256
        or library.model_dump() != inputs.expected_library.model_dump()
    ):
        raise ValueError("requested original library differs from verified package")
    raw = inputs.forecast_json.encode("utf-8")
    if hashlib.sha256(raw).hexdigest() != inputs.forecast_sha256:
        raise ValueError("saved forecast SHA-256 mismatch")
    # Strict finite JSON also refuses duplicate keys. The shared parser below preserves
    # the CLI's existing zone/order/horizon checks and exact-byte trace identity.
    forecast = _strict_json(raw)
    if not isinstance(forecast, dict):
        raise ValueError("saved forecast must be an object")
    forecast_rows = forecast.get("forecast")
    if not isinstance(forecast_rows, list) or not 1 <= len(forecast_rows) <= MAX_SAVED_RECORDS:
        raise ValueError("saved forecast needs 1..256 rows")
    for row in forecast_rows:
        text = row.get("date") if isinstance(row, dict) else None
        if not isinstance(text, str) or date.fromisoformat(text).isoformat() != text:
            raise ValueError("saved forecast dates must be real YYYY-MM-DD dates")
    metadata = forecast.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError("saved forecast metadata is required")
    horizon = metadata.get("horizon_days")
    if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon != len(forecast_rows):
        raise ValueError("saved forecast horizon must equal the number of scored rows")
    rows, forecast_identity, horizon = parse_saved_forecast(
        raw, zone_id, filename="saved-forecast.json", label="saved forecast"
    )
    forecast_identity["packet_sha256"] = inputs.packet_sha256
    result = score_level2(
        library,
        library_identity=identity,
        forecast_rows=rows,
        forecast_identity=forecast_identity,
        zone_id=zone_id,
        effective_horizon_days=horizon,
        params=B6Params(),
        code_ref=code_ref(),
        computed_at=datetime.now(UTC).isoformat(timespec="seconds"),
    )
    return {
        "zone_id": zone_id,
        "alerts": [],
        "statistics": {
            "n_patterns_read": result.n_patterns_read,
            "n_eligible": result.n_eligible,
            "n_alerts": 0,
        },
        "metadata": {
            "mode": "saved_inputs_j1",
            "packet_sha256": inputs.packet_sha256,
            "forecast_sha256": inputs.forecast_sha256,
            "acquisition": "none",
        },
        "level2_result": result.model_dump(mode="json"),
    }
