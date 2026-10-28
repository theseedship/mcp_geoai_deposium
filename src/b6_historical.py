"""Fixed historical engineering replay; no acquisition or scientific adoption.

The reviewed catalog binds the full native library and the exact retained GFS
forecast. A packet digest is upstream provenance, never tenant authorization.
"""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.b6_explanatory import score_level2
from src.b6_forecast import parse_saved_forecast
from src.b6_saved_contracts import B6HistoricalInputs, preflight_saved_inputs
from src.contracts import B6HistoricalCaseTrace, B6LibraryIdentity, B6Params
from src.patterns.historical_case import (
    BUNDLE_SHA256,
    PARQUET_SHA256,
    ZONES_SHA256,
    _check_historical_forecast_compatibility,
    _read,
    verify_historical_case,
)
from src.patterns.library_schema import PatternLibrary
from src.patterns.offline_bundle import _strict_json
from src.utils.code_ref import code_ref

CASE_ID = "q3-2023-fos-native-recurrence"
CASE_VERSION = "2026.10.27.1"
CASE_MANIFEST_SHA256 = "81df258d3035b578aa658aef854814004dea934ca3273bdb6a238dc4ad93c100"
LIBRARY_JSON_SHA256 = "2d790c561dae5610ae538ea7c7fae1e60059965cd8f3cc48aaac22917c5283eb"
PREPARATION_MANIFEST_SHA256 = "d7b1ae672c6033046f9449908ffa093328c7d8ff4845ad73bc93ef1c8e924e5c"
FORECAST_SHA256 = "e954ee3b33b3f5cefe90c6041ad6fdaef79389f767307b33f0bcd801a986a6fd"
EXTRACTION_MANIFEST_SHA256 = "181adc5231084a819ba2372d8baba3269b8c9104d573cd01efd0d2dba1c6dcd4"
_CASE_ROOT = Path(__file__).resolve().parent / "data/b6-historical-runtime-2026-10-27"
LIMITATIONS = (
    "Contextual historical engineering arithmetic only; scientific/domain adoption remains pending.",
    "Native cross-zone recurrence and stability are not predictive accuracy or independent held-out day-lag validation.",
    "Temperature-only rule15 profile; humidity and evapotranspiration remain null, and every native rule is still assessed.",
    "Retained GFS input uses the containing 0.25 degree grid cell and eight 3-hourly instantaneous samples per UTC day, not an hourly daily mean.",
    "No calibrated confidence, event probability, operational activation or FIERCE reception is established.",
    "The upstream packet digest is provenance only; tenant/space authorization is enforced by the authenticated connector.",
)


def _pinned_file(name: str, expected: str) -> bytes:
    """Read only fixed catalog filenames with bounded regular-file verification."""
    raw = _read(_CASE_ROOT / name)
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError("historical runtime artifact digest mismatch")
    return raw


def verify_pinned_case() -> tuple[dict[str, Any], bytes, dict[str, Any]]:
    """Verify the complete immutable preparation and exact retained forecast inputs."""
    manifest = _strict_json(_pinned_file("case-manifest.json", CASE_MANIFEST_SHA256))
    if manifest != {
        "schema_version": "geoai-b6-historical-runtime-case/v1",
        "case_id": CASE_ID,
        "case_version": CASE_VERSION,
        "preparation_manifest_sha256": PREPARATION_MANIFEST_SHA256,
        "forecast_profile": "rule15-temperature",
        "files": {
            "forecast.json": FORECAST_SHA256,
            "extraction-manifest.json": EXTRACTION_MANIFEST_SHA256,
        },
    }:
        raise ValueError("historical runtime catalog mismatch")
    extraction = _strict_json(_pinned_file("extraction-manifest.json", EXTRACTION_MANIFEST_SHA256))
    raw = _pinned_file("forecast.json", FORECAST_SHA256)
    if extraction["files"]["forecast.json"] != FORECAST_SHA256:
        raise ValueError("historical extraction forecast identity mismatch")
    report = verify_historical_case(_CASE_ROOT / "package", PREPARATION_MANIFEST_SHA256)
    compatibility = _check_historical_forecast_compatibility(
        report,
        raw,
        FORECAST_SHA256,
        forecast_profile="rule15-temperature",
    )
    if compatibility["synthetic"]:
        raise ValueError("fixed real historical case cannot use a synthetic forecast")
    return report, raw, compatibility


def score_historical_case(
    zone_id: str, inputs: B6HistoricalInputs | dict[str, Any]
) -> dict[str, Any]:
    """Run existing pure J1 arithmetic over server-owned inputs without adopting them."""
    preflight_saved_inputs(zone_id, inputs)
    request = B6HistoricalInputs.model_validate(inputs)
    if zone_id != "fos_sur_mer" or request.expected_case_manifest_sha256 != CASE_MANIFEST_SHA256:
        raise ValueError("historical case target or manifest mismatch")
    report, raw, compatibility = verify_pinned_case()
    library_raw = _pinned_file("package/library.json", LIBRARY_JSON_SHA256)
    if len(library_raw) > 256 * 1024 or len(raw) > 256 * 1024:
        raise ValueError("historical source exceeds response byte bound")
    library = PatternLibrary.model_validate(report["library"])
    if PatternLibrary.model_validate(_strict_json(library_raw)) != library:
        raise ValueError("historical source differs from scored native library")
    rows, forecast_identity, horizon = parse_saved_forecast(
        raw, zone_id, filename="forecast.json", label="fixed historical forecast"
    )
    forecast_identity["packet_sha256"] = request.packet_sha256
    identity = B6LibraryIdentity(
        library_version=f"{CASE_ID}/{CASE_VERSION}",
        manifest_sha256=PREPARATION_MANIFEST_SHA256,
        json_sha256=LIBRARY_JSON_SHA256,
        parquet_sha256=PARQUET_SHA256,
        schema_version="1.0.0",
        zone_id="cross_zone",
        algorithm="tgrad",
        area_status="historical_cross_zone_recurrence",
        qualification="scientific_adoption_pending",
    )
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
    historical_trace = B6HistoricalCaseTrace(
        case_id=CASE_ID,
        case_version=CASE_VERSION,
        case_manifest_sha256=CASE_MANIFEST_SHA256,
        preparation_manifest_sha256=PREPARATION_MANIFEST_SHA256,
        forecast_profile="rule15-temperature",
        producer_zone_id="cross_zone",
        target_zone_id="fos_sur_mer",
        forecast_point={"lat": 43.445, "lon": 4.92},
        registry_sha256=ZONES_SHA256,
        source_bundle_sha256=BUNDLE_SHA256,
        native_parquet_sha256=PARQUET_SHA256,
        extraction_manifest_sha256=EXTRACTION_MANIFEST_SHA256,
        forecast_sha256=FORECAST_SHA256,
        model_issued_at=compatibility["model_issued_at"],
        retrieved_at=compatibility["retrieved_at"],
        limitations=list(LIMITATIONS),
    )
    result = result.model_copy(
        update={"trace": result.trace.model_copy(update={"historical_case": historical_trace})}
    )
    return {
        "historical_source": {
            "library_json": library_raw.decode("utf-8"),
            "forecast_json": raw.decode("utf-8"),
        },
        "zone_id": zone_id,
        "alerts": [],
        "statistics": {
            "n_patterns_read": result.n_patterns_read,
            "n_eligible": result.n_eligible,
            "n_alerts": 0,
        },
        "metadata": {
            "mode": "historical_case_j1",
            "packet_sha256": request.packet_sha256,
            "case_manifest_sha256": CASE_MANIFEST_SHA256,
            "forecast_sha256": FORECAST_SHA256,
            "acquisition": "none",
        },
        "level2_result": result.model_dump(mode="json", exclude_none=False),
    }
