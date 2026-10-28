"""Prepare the pinned Q3 Fos case offline; no adoption, scoring or acquisition."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import stat
import tempfile
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from src.b6_saved_contracts import MAX_SAVED_ENVELOPE_BYTES
from src.patterns.offline_bundle import MAX_JSON_BYTES, _read_library, _strict_json

BUNDLE_SHA256 = "e0164cf24138ae5396064b0665ed35389577c2f538cfe588e9e502fa84297c8c"
PARQUET_SHA256 = "ac23cb68dcf1f1b62ce6416b867d1ae5afcd2ecac168c0308b82f0ef9db62e49"
ZONES_SHA256 = "8771cfc5ba55c37e277be0637087567cb1e561b54796e1753cb867ecb66b8467"
REGISTRY_COMMIT = "4ff7441aee3f6970f14706c63937d9069baa39e0"
PARQUET_KEY = "climate/o3/holdout/q3_2023/tgrad_validated.parquet"
MONTHS = ("2023_07", "2023_08", "2023_09")
SOURCE_FILES = {"source/bundle.json", "source/tgrad.parquet", "source/zones.geojson"} | {
    f"source/b4-{month}.json" for month in MONTHS
}
PACKAGE_FILES = SOURCE_FILES | {"manifest.json", "library.json"}
MAX_PACKAGE_BYTES = 8 * MAX_JSON_BYTES
DRIVERS = ("temperature_2m", "humidity", "evapotranspiration")


class _FileIdentity(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    bytes: int = Field(ge=0, le=MAX_JSON_BYTES)


class _Manifest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    schema_version: Literal["geoai-b6-historical-case/v1"]
    descriptor: dict
    files: dict[str, _FileIdentity]


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _json_bytes(value: dict) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def _regular(path: Path) -> None:
    info = path.lstat()
    if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
        raise ValueError("historical case refuses symlinks and reparse points")
    if not (stat.S_ISDIR(info.st_mode) or stat.S_ISREG(info.st_mode)):
        raise ValueError("historical case requires regular files/directories")


def _read(path: Path) -> bytes:
    _regular(path)
    if not path.is_file():
        raise ValueError("historical case source must be a file")
    with path.open("rb") as stream:
        raw = stream.read(MAX_JSON_BYTES + 1)
    if len(raw) > MAX_JSON_BYTES:
        raise ValueError("historical case file exceeds byte limit")
    return raw


def _source(root: Path, name: str) -> bytes:
    # Names are exclusively fixed constants or read from the pinned bundle.
    parts = name.split("/")
    if any(p in ("", ".", "..") for p in parts) or "\\" in name or ":" in name:
        raise ValueError("unsafe historical source path")
    current = root
    _regular(current)
    for part in parts:
        current = current / part
        _regular(current)
    return _read(current)


def _pin(raw: bytes, expected: str) -> None:
    if not re.fullmatch(r"[a-f0-9]{64}", expected) or _sha(raw) != expected:
        raise ValueError("historical case SHA-256 mismatch")


def _descriptor(registry: dict) -> dict:
    feature = next(f for f in registry["features"] if f["properties"]["zone_id"] == "fos_sur_mer")
    return {
        "case_id": "q3-2023-fos-native-recurrence/v1",
        "producer_zone_id": "cross_zone",
        "target_zone_id": "fos_sur_mer",
        "algorithm": "tgrad",
        "source_bundle_sha256": BUNDLE_SHA256,
        "native_parquet_sha256": PARQUET_SHA256,
        "native_record_count": 149,
        "period_start": "2023-07-01",
        "period_end": "2023-09-30",
        "primary_rule_ids": [f"cross_zone/tgrad/{i}" for i in (15, 16, 22)],
        "shorter_lag_alternative_rule_ids": ["cross_zone/tgrad/23"],
        "minimum_horizon_days": 5,
        "geometry": feature["geometry"],
        "bbox": feature["properties"]["bbox"],
        "forecast_point": {"lat": 43.445, "lon": 4.92},
        "registry_sha256": ZONES_SHA256,
        "registry_git_object": f"{REGISTRY_COMMIT}:zones/zones.geojson",
        "evidence_kind": "historical_cross_zone_recurrence",
        "scientific_adoption": "pending",
        "live_activation": False,
        "ready_for_scoring": False,
        "independent_heldout_day_lag_validation_established": False,
        "probabilistic_confidence": None,
    }


def _validate_sources(files: dict[str, bytes]) -> tuple[dict, dict, dict]:
    _pin(files["source/bundle.json"], BUNDLE_SHA256)
    _pin(files["source/tgrad.parquet"], PARQUET_SHA256)
    _pin(files["source/zones.geojson"], ZONES_SHA256)
    bundle = _strict_json(files["source/bundle.json"])
    registry = _strict_json(files["source/zones.geojson"])
    entries = {entry["key"]: entry for entry in bundle["objects"]}
    if entries[PARQUET_KEY]["bytes"] != len(files["source/tgrad.parquet"]):
        raise ValueError("historical Parquet length mismatch")
    provenance = []
    for month in MONTHS:
        key = f"climate/runs/holdout-full-panel-mesh-2023-q3-segmented-proof-{month}-unified/manifest.json"
        raw = files[f"source/b4-{month}.json"]
        entry = entries[key]
        _pin(raw, entry["sha256"])
        if len(raw) != entry["bytes"]:
            raise ValueError("historical B4 manifest length mismatch")
        manifest = _strict_json(raw)
        if manifest["zone_artifact"]["sha256"] != ZONES_SHA256:
            raise ValueError("historical registry differs from B4 provenance")
        provenance.append({"key": key, "sha256": entry["sha256"]})
    # Decode a private snapshot of the exact authenticated bytes.
    with tempfile.TemporaryDirectory(prefix="geoai-historical-read-") as temporary:
        path = Path(temporary) / "native.parquet"
        path.write_bytes(files["source/tgrad.parquet"])
        library = _read_library(path, "tgrad")
    if library.zone_id != "cross_zone" or len(library.patterns) != 149:
        raise ValueError("historical native scope or record count mismatch")
    return (
        library.model_dump(mode="json"),
        _descriptor(registry),
        {
            "bundle_sha256": BUNDLE_SHA256,
            "native_parquet_sha256": PARQUET_SHA256,
            "registry_sha256": ZONES_SHA256,
            "b4_manifests": provenance,
            "caveats": bundle["caveats"],
        },
    )


def verify_historical_case(package_dir: Path, expected_manifest_sha256: str) -> dict:
    """Validate the closed package and compare all records with pinned native bytes."""
    root = Path(package_dir)
    pending = [root]
    actual = set()
    count = total = 0
    while pending:
        current = pending.pop()
        _regular(current)
        with os.scandir(current) as entries:
            for entry in entries:
                count += 1
                if count > len(PACKAGE_FILES) + 1:
                    raise ValueError("historical package exceeds entry limit")
                path = Path(entry.path)
                _regular(path)
                name = path.relative_to(root).as_posix()
                if path.is_dir():
                    if name != "source":
                        raise ValueError("undeclared historical directory")
                    pending.append(path)
                else:
                    total += path.stat().st_size
                    if total > MAX_PACKAGE_BYTES:
                        raise ValueError("historical package exceeds byte limit")
                    actual.add(name)
    if actual != PACKAGE_FILES:
        raise ValueError("missing or undeclared historical package file")
    raw = _read(root / "manifest.json")
    _pin(raw, expected_manifest_sha256)
    manifest = _Manifest.model_validate(_strict_json(raw))
    if set(manifest.files) != PACKAGE_FILES - {"manifest.json"}:
        raise ValueError("historical inventory differs from fixed file set")
    files = {}
    for name, identity in manifest.files.items():
        value = _source(root, name)
        _pin(value, identity.sha256)
        if len(value) != identity.bytes:
            raise ValueError("historical package length mismatch")
        files[name] = value
    library, descriptor, provenance = _validate_sources(files)
    if _json_bytes(_strict_json(files["library.json"])) != _json_bytes(library):
        raise ValueError("derived library differs from full native records")
    if _json_bytes(manifest.descriptor) != _json_bytes(descriptor):
        raise ValueError("historical descriptor differs from fixed case scope")
    return {
        "schema_version": manifest.schema_version,
        "manifest_sha256": _sha(raw),
        "descriptor": descriptor,
        "library": library,
        "provenance": provenance,
    }


def export_historical_case(bundle_dir: Path, zones_geojson: Path, output_dir: Path) -> dict:
    """Create a deterministic authenticated case package without replacing any output."""
    root, output = Path(bundle_dir), Path(output_dir)
    if os.path.lexists(output):
        raise ValueError("historical output already exists")
    files = {
        "source/bundle.json": _source(root, "bundle.json"),
        "source/tgrad.parquet": _source(root, "objects/" + PARQUET_KEY),
        "source/zones.geojson": _read(Path(zones_geojson)),
    }
    for month in MONTHS:
        key = f"climate/runs/holdout-full-panel-mesh-2023-q3-segmented-proof-{month}-unified/manifest.json"
        files[f"source/b4-{month}.json"] = _source(root, "objects/" + key)
    library, descriptor, _ = _validate_sources(files)
    files["library.json"] = _json_bytes(library)
    manifest = {
        "schema_version": "geoai-b6-historical-case/v1",
        "descriptor": descriptor,
        "files": {
            name: {"sha256": _sha(data), "bytes": len(data)} for name, data in sorted(files.items())
        },
    }
    files["manifest.json"] = _json_bytes(manifest)
    digest = _sha(files["manifest.json"])
    with tempfile.TemporaryDirectory(prefix="geoai-historical-export-") as temporary:
        stage = Path(temporary)
        (stage / "source").mkdir()
        for name, data in files.items():
            (stage / name).write_bytes(data)
        verify_historical_case(stage, digest)
        # Exclusive directory creation is the publication guard, including races.
        output.mkdir()
        created = []
        try:
            (output / "source").mkdir()
            for name, data in files.items():
                path = output / name
                with path.open("xb") as stream:
                    created.append(path)
                    stream.write(data)
        except BaseException:
            for path in reversed(created):
                path.unlink(missing_ok=True)
            for path in (output / "source", output):
                try:
                    path.rmdir()
                except OSError:
                    pass
            raise
    return {
        "schema_version": "geoai-b6-historical-case-receipt/v1",
        "manifest_sha256": digest,
        "native_records": len(library["patterns"]),
        "target_zone_id": "fos_sur_mer",
        "ready_for_scoring": False,
        "scientific_adoption": "pending",
        "live_activation": False,
    }


def _finite_number(value) -> bool:
    try:
        return (
            isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
        )
    except OverflowError:
        return False


def preflight_historical_forecast(
    package_dir: Path,
    expected_manifest_sha256: str,
    forecast_raw: bytes,
    expected_forecast_sha256: str,
    *,
    forecast_profile: str = "full-shortlist",
) -> dict:
    """Check compatibility only, without authenticating the caller's origin claims.

    A nonsynthetic replay must be issued by the first UTC day's start so its
    complete first daily row contains no already observed part of that day.
    Synthetic fixtures may be issued during that day and remain nonevidence.
    The temperature profile changes compatibility requirements only; the package
    retains all native rules, and neither profile grants scoring or adoption.
    """
    if forecast_profile not in ("full-shortlist", "rule15-temperature"):
        raise ValueError("unknown historical forecast profile")
    report = verify_historical_case(package_dir, expected_manifest_sha256)
    return _check_historical_forecast_compatibility(
        report, forecast_raw, expected_forecast_sha256, forecast_profile=forecast_profile
    )


def _check_historical_forecast_compatibility(
    report: dict,
    forecast_raw: bytes,
    expected_forecast_sha256: str,
    *,
    forecast_profile: str,
) -> dict:
    """Pure compatibility check for an internally verified historical report.

    The public preflight always verifies its package itself; the fixed runtime
    already verifies that package and uses this private helper on that report.
    This helper reads no files and grants no adoption or scoring readiness.
    """
    if forecast_profile not in ("full-shortlist", "rule15-temperature"):
        raise ValueError("unknown historical forecast profile")
    required_drivers = DRIVERS if forecast_profile == "full-shortlist" else ("temperature_2m",)
    if not isinstance(forecast_raw, bytes) or len(forecast_raw) > MAX_SAVED_ENVELOPE_BYTES:
        raise ValueError("historical forecast exceeds byte limit")
    _pin(forecast_raw, expected_forecast_sha256)
    value = _strict_json(forecast_raw)
    if not isinstance(value, dict) or value.get("error") or value.get("zone_id") != "fos_sur_mer":
        raise ValueError("historical forecast target mismatch or provider error")
    metadata = value.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError("historical forecast metadata required")
    for axis, expected in report["descriptor"]["forecast_point"].items():
        number = metadata.get(axis)
        if not _finite_number(number) or not math.isclose(
            number, expected, abs_tol=1e-9, rel_tol=0
        ):
            raise ValueError("historical forecast geometry/point mismatch")
    if metadata.get("registry_sha256") != ZONES_SHA256:
        raise ValueError("historical forecast registry identity required")
    rows = value.get("forecast")
    if not isinstance(rows, list) or not 5 <= len(rows) <= 92:
        raise ValueError("historical forecast requires 5..92 rows")
    horizon = metadata.get("horizon_days")
    if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon != len(rows):
        raise ValueError("historical forecast horizon must equal rows and cover five days")
    days = []
    for row in rows:
        text = row.get("date") if isinstance(row, dict) else None
        if not isinstance(text, str):
            raise ValueError("historical forecast dates required")
        day = date.fromisoformat(text)
        if day.isoformat() != text or not date(2023, 7, 1) <= day <= date(2023, 9, 30):
            raise ValueError("historical forecast date outside original Q3 domain")
        days.append(day)
        for driver in DRIVERS:
            if driver in row and row[driver] is not None and not _finite_number(row[driver]):
                raise ValueError("historical forecast driver must be finite numeric or null")
    if any(right != left + timedelta(days=1) for left, right in zip(days, days[1:], strict=False)):
        raise ValueError("historical forecast requires consecutive ascending daily rows")
    if any(sum(_finite_number(row.get(driver)) for row in rows) < 2 for driver in required_drivers):
        raise ValueError("historical forecast missing two finite values for each driver")
    synthetic = metadata.get("synthetic", False)
    if not isinstance(synthetic, bool):
        raise ValueError("historical forecast synthetic marker must be boolean")
    issuance = metadata.get("model_issued_at")
    if not isinstance(issuance, str):
        raise ValueError("historical model_issued_at required separately from retrieval issued_at")
    try:
        instant = datetime.fromisoformat(issuance.replace("Z", "+00:00"))
        if instant.tzinfo is None:
            raise ValueError("timezone required")
        # An ISO timestamp within datetime's local bounds can overflow in UTC.
        instant_utc = instant.astimezone(UTC)
    except (ValueError, OverflowError) as exc:
        raise ValueError(
            "historical model issuance must be valid and representable in UTC"
        ) from exc
    if instant_utc.date() > days[0]:
        raise ValueError("historical model issuance must be timezone-aware and precede forecast")
    first_day_start = datetime(days[0].year, days[0].month, days[0].day, tzinfo=UTC)
    if not synthetic and instant_utc > first_day_start:
        raise ValueError("nonsynthetic model issuance must precede the first full UTC forecast day")
    source = metadata.get("source")
    if not isinstance(source, str) or not source.strip():
        raise ValueError("historical forecast source provenance required")
    result = {
        "schema_version": "geoai-b6-historical-forecast-preflight/v1",
        "manifest_sha256": report["manifest_sha256"],
        "forecast_sha256": _sha(forecast_raw),
        "compatible": True,
        "target_zone_id": "fos_sur_mer",
        "producer_zone_id": "cross_zone",
        "horizon_days": horizon,
        "model_issued_at": issuance,
        "forecast_origin_authentication": "not_established_by_metadata",
        "retrieved_at": metadata.get("issued_at"),
        "synthetic": synthetic,
        "evidence_kind": "synthetic_technical_fixture"
        if synthetic
        else "historical_compatibility_only",
        "ready_for_scoring": False,
        "scientific_adoption": "pending",
        "live_activation": False,
    }
    if forecast_profile == "rule15-temperature":
        result.update(
            schema_version="geoai-b6-historical-forecast-preflight/v2",
            profile_id="q3-2023-fos-rule15-temperature/v1",
            required_forecast_variables=list(required_drivers),
            primary_case_rule_ids=["cross_zone/tgrad/15"],
        )
    return result
