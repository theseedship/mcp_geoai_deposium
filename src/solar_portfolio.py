"""Bounded, resumable collection of solar weather data for operator supplied sites.

This module owns only CSV validation, per-query checkpoints and long-form export. It
does not derive PV output or aggregate across sites or scenarios.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from src.solar_contracts import SolarQuery

MANIFEST_SCHEMA = "geoai-solar-portfolio/v2"
ARTIFACT_SCHEMA = "geoai-solar/v1"
ADAPTER_VERSION = "geoai-solar-adapter/v1"
MAX_SITES = 1000
MAX_CSV_BYTES = 2 * 1024 * 1024
ALLOWED_SOURCES = {"forecast", "era5", "satellite"}
ALLOWED_VARIABLES = {
    "shortwave_radiation",
    "diffuse_radiation",
    "direct_normal_irradiance",
    "global_tilted_irradiance",
    "temperature_2m",
    "wind_speed_10m",
}
CSV_FIELDS = {"site_id", "latitude", "longitude", "scenario_id", "tilt", "azimuth"}
RETRYABLE_NAMES = {
    "TimeoutError",
    "ConnectionError",
    "ConnectError",
    "ConnectTimeout",
    "ReadTimeout",
    "WriteTimeout",
    "PoolTimeout",
}
EXPORT_FIELDS = (
    "site_id",
    "scenario_id",
    "tilt",
    "azimuth",
    "time",
    "variable",
    "value",
    "unit",
    "source",
    "product_type",
    "requested_model",
    "actual_model",
    "model_issued_at",
    "retrieved_at",
    "requested_latitude",
    "requested_longitude",
    "returned_latitude",
    "returned_longitude",
    "returned_elevation",
    "temporal_resolution",
    "interval_seconds",
    "native_model_interval_seconds",
    "interval_semantics",
    "variable_semantics",
    "licence",
    "attribution",
)


class SolarPortfolioError(ValueError):
    """A batch, input CSV, manifest, or artifact failed validation."""


class _FetchFailure(Exception):
    def __init__(self, cause: Exception, *, attempts: int) -> None:
        self.cause = cause
        self.attempts = attempts
        self.retryable = (
            bool(getattr(cause, "retryable", False)) or type(cause).__name__ in RETRYABLE_NAMES
        )
        super().__init__(type(cause).__name__)


@dataclass(frozen=True)
class SolarSite:
    site_id: str
    latitude: float
    longitude: float
    scenario_id: str = "default"
    tilt: float | None = None
    azimuth: float | None = None
    line: int = 0

    def canonical(self) -> dict[str, Any]:
        return {
            "site_id": self.site_id,
            "scenario_id": self.scenario_id,
            "latitude": self.latitude,
            "longitude": self.longitude,
            "tilt": self.tilt,
            "azimuth": self.azimuth,
        }


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
        "utf-8"
    )


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _atomic_json(path: Path, value: Any) -> str:
    """Write canonical JSON and return its byte digest after an atomic replace."""
    payload = _canonical_bytes(value) + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
        return _sha256_bytes(payload)
    finally:
        try:
            os.unlink(name)
        except FileNotFoundError:
            pass


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SolarPortfolioError(f"invalid JSON file: {path.name}") from exc


def _finite_number(raw: str | None, *, field: str, line: int, required: bool) -> float | None:
    if raw is None or not raw.strip():
        if required:
            raise SolarPortfolioError(f"line {line}: {field} is required")
        return None
    try:
        number = float(raw)
    except ValueError as exc:
        raise SolarPortfolioError(f"line {line}: {field} must be numeric") from exc
    if not math.isfinite(number):
        raise SolarPortfolioError(f"line {line}: {field} must be finite")
    return number


def _structured_number(raw: Any, *, field: str, line: int, required: bool) -> float | None:
    if raw is None:
        if required:
            raise SolarPortfolioError(f"line {line}: {field} is required")
        return None
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        raise SolarPortfolioError(f"line {line}: {field} must be numeric")
    try:
        number = float(raw)
    except OverflowError:
        raise SolarPortfolioError(f"line {line}: {field} must be finite") from None
    if not math.isfinite(number):
        raise SolarPortfolioError(f"line {line}: {field} must be finite")
    return number


def validate_solar_sites(
    rows: Sequence[Mapping[str, Any] | SolarSite], *, variables: Sequence[str] = ()
) -> list[SolarSite]:
    """Normalize typed sites and validate the same invariants as the CSV adapter."""
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        raise SolarPortfolioError("sites must be a sequence of site objects")
    sites: list[SolarSite] = []
    seen: set[tuple[str, str]] = set()
    gps_by_site: dict[str, tuple[float, float]] = {}
    for index, raw in enumerate(rows, start=1):
        row: Mapping[str, Any]
        if isinstance(raw, SolarSite):
            row = raw.canonical()
            line = raw.line or index
        elif isinstance(raw, Mapping):
            row = raw
            line = index
        else:
            raise SolarPortfolioError(f"line {index}: site must be an object")
        if len(sites) >= MAX_SITES:
            raise SolarPortfolioError(f"line {line}: site limit {MAX_SITES} exceeded")
        if set(row) - CSV_FIELDS:
            raise SolarPortfolioError(f"line {line}: site has unknown fields")
        if not isinstance(row.get("site_id"), str):
            raise SolarPortfolioError(f"line {line}: site_id must be text")
        if not isinstance(row.get("scenario_id", "default"), str):
            raise SolarPortfolioError(f"line {line}: scenario_id must be text")
        site_id = (row.get("site_id") or "").strip()
        scenario_id = (row.get("scenario_id") or "default").strip() or "default"
        if not site_id or len(site_id) > 128:
            raise SolarPortfolioError(f"line {line}: site_id must contain 1 to 128 characters")
        if len(scenario_id) > 128:
            raise SolarPortfolioError(
                f"line {line}: scenario_id must contain at most 128 characters"
            )
        key = (site_id, scenario_id)
        if key in seen:
            raise SolarPortfolioError(
                f"line {line}: duplicate site_id/scenario_id {site_id!r}/{scenario_id!r}"
            )
        latitude = _structured_number(
            row.get("latitude"), field="latitude", line=line, required=True
        )
        longitude = _structured_number(
            row.get("longitude"), field="longitude", line=line, required=True
        )
        assert latitude is not None and longitude is not None
        if not -90 <= latitude <= 90:
            raise SolarPortfolioError(f"line {line}: latitude must be between -90 and 90")
        if not -180 <= longitude <= 180:
            raise SolarPortfolioError(f"line {line}: longitude must be between -180 and 180")
        previous_gps = gps_by_site.get(site_id)
        if previous_gps is not None and previous_gps != (latitude, longitude):
            raise SolarPortfolioError(
                f"line {line}: site_id {site_id!r} changes coordinates across scenarios"
            )
        tilt = _structured_number(row.get("tilt"), field="tilt", line=line, required=False)
        azimuth = _structured_number(row.get("azimuth"), field="azimuth", line=line, required=False)
        if (tilt is None) != (azimuth is None):
            raise SolarPortfolioError(f"line {line}: tilt and azimuth must be supplied together")
        if tilt is not None and not 0 <= tilt <= 90:
            raise SolarPortfolioError(f"line {line}: tilt must be between 0 and 90")
        if azimuth is not None and not -180 <= azimuth <= 180:
            raise SolarPortfolioError(f"line {line}: azimuth must be between -180 and 180")
        if "global_tilted_irradiance" in variables and tilt is None:
            raise SolarPortfolioError(
                f"line {line}: tilt and azimuth are required for global_tilted_irradiance"
            )
        seen.add(key)
        gps_by_site[site_id] = (latitude, longitude)
        sites.append(SolarSite(site_id, latitude, longitude, scenario_id, tilt, azimuth, line))
    if not sites:
        raise SolarPortfolioError("sites have no data rows")
    return sites


def load_solar_sites(path: str | Path, *, variables: Sequence[str] = ()) -> list[SolarSite]:
    """Read and fully validate a bounded site/scenario CSV before any network work."""
    csv_path = Path(path)
    try:
        size = csv_path.stat().st_size
    except OSError as exc:
        raise SolarPortfolioError("cannot read sites CSV") from exc
    if size > MAX_CSV_BYTES:
        raise SolarPortfolioError(f"sites CSV exceeds {MAX_CSV_BYTES} bytes")
    try:
        with csv_path.open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream, strict=True)
            headers = reader.fieldnames
            if not headers:
                raise SolarPortfolioError("sites CSV is missing a header")
            if len(headers) != len(set(headers)):
                raise SolarPortfolioError("sites CSV has duplicate headers")
            unknown = set(headers) - CSV_FIELDS
            missing = {"site_id", "latitude", "longitude"} - set(headers)
            if unknown:
                raise SolarPortfolioError(
                    f"sites CSV has unknown columns: {', '.join(sorted(unknown))}"
                )
            if missing:
                raise SolarPortfolioError(
                    f"sites CSV is missing columns: {', '.join(sorted(missing))}"
                )
            sites: list[SolarSite] = []
            for row in reader:
                line = reader.line_num
                if len(sites) >= MAX_SITES:
                    raise SolarPortfolioError(f"line {line}: site limit {MAX_SITES} exceeded")
                if None in row:
                    raise SolarPortfolioError(f"line {line}: row has more values than headers")
                latitude = _finite_number(
                    row.get("latitude"), field="latitude", line=line, required=True
                )
                longitude = _finite_number(
                    row.get("longitude"), field="longitude", line=line, required=True
                )
                assert latitude is not None and longitude is not None
                sites.append(
                    SolarSite(
                        site_id=row.get("site_id") or "",
                        latitude=latitude,
                        longitude=longitude,
                        scenario_id=row.get("scenario_id") or "default",
                        tilt=_finite_number(
                            row.get("tilt"), field="tilt", line=line, required=False
                        ),
                        azimuth=_finite_number(
                            row.get("azimuth"), field="azimuth", line=line, required=False
                        ),
                        line=line,
                    )
                )
    except UnicodeDecodeError as exc:
        raise SolarPortfolioError(
            f"line {getattr(reader, 'line_num', '?')}: sites CSV is not valid UTF-8"
        ) from exc
    except csv.Error as exc:
        raise SolarPortfolioError(
            f"line {getattr(reader, 'line_num', '?')}: malformed CSV"
        ) from exc
    if not sites:
        raise SolarPortfolioError("sites CSV has no data rows")
    return validate_solar_sites(sites, variables=variables)


def _iso_date(value: str, field: str) -> str:
    try:
        parsed = date.fromisoformat(value)
    except (TypeError, ValueError) as exc:
        raise SolarPortfolioError(f"{field} must be an ISO date YYYY-MM-DD") from exc
    if parsed.isoformat() != value:
        raise SolarPortfolioError(f"{field} must be an ISO date YYYY-MM-DD")
    return value


def _normalise_query(query: SolarQuery) -> dict[str, Any]:
    return query.model_dump(mode="json")


def _query_for(
    site: SolarSite,
    *,
    source: str,
    start_date: str,
    end_date: str,
    variables: Sequence[str],
    model: str | None,
    temporal_resolution: str,
) -> SolarQuery:
    try:
        return SolarQuery(
            source=source,
            latitude=site.latitude,
            longitude=site.longitude,
            start_date=start_date,
            end_date=end_date,
            variables=list(variables),
            model=model,
            tilt=site.tilt if "global_tilted_irradiance" in variables else None,
            azimuth=site.azimuth if "global_tilted_irradiance" in variables else None,
            temporal_resolution=temporal_resolution,
        )
    except Exception as exc:
        raise SolarPortfolioError(f"invalid solar query: {exc}") from None


def _query_id(query_data: Mapping[str, Any], *, endpoint_identity: str) -> str:
    return _sha256_bytes(
        _canonical_bytes(
            {
                "adapter_version": ADAPTER_VERSION,
                "endpoint_identity": endpoint_identity,
                "query": query_data,
            }
        )
    )


def _item_id(site_id: str, scenario_id: str) -> str:
    return _sha256_bytes(_canonical_bytes({"site_id": site_id, "scenario_id": scenario_id}))


def _batch_spec(
    *,
    source: str,
    start_date: str,
    end_date: str,
    variables: Sequence[str],
    model: str | None,
    sites: Sequence[SolarSite],
    endpoint_identity: str,
    temporal_resolution: str,
) -> dict[str, Any]:
    return {
        "schema_version": MANIFEST_SCHEMA,
        "artifact_schema": ARTIFACT_SCHEMA,
        "adapter_version": ADAPTER_VERSION,
        "endpoint_identity": endpoint_identity,
        "source": source,
        "model": model,
        "start_date": start_date,
        "end_date": end_date,
        "variables": list(variables),
        "temporal_resolution": temporal_resolution,
        "sites": [site.canonical() for site in sites],
    }


def _error_record(exc: Exception) -> dict[str, Any]:
    # Do not persist provider messages, which may contain URLs, request headers or tokens.
    cause = exc.cause if isinstance(exc, _FetchFailure) else exc
    name = type(cause).__name__
    retryable = bool(cause.retryable) if hasattr(cause, "retryable") else name in RETRYABLE_NAMES
    return {
        "type": name if re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,63}", name) else "ProviderError",
        "retryable": retryable,
    }


def _parse_timestamp(value: Any) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return None
    return parsed.astimezone(UTC)


def _valid_artifact(
    payload: Any, *, expected_query: Mapping[str, Any], max_age_hours: float | None, now: datetime
) -> bool:
    if not isinstance(payload, dict) or payload.get("schema_version") != ARTIFACT_SCHEMA:
        return False
    recorded_query = payload.get("query")
    if not isinstance(recorded_query, dict):
        return False
    try:
        if _canonical_bytes(recorded_query) != _canonical_bytes(expected_query):
            return False
    except (TypeError, ValueError):
        return False
    retrieved = _parse_timestamp(payload.get("retrieved_at"))
    if retrieved is None or retrieved > now + timedelta(minutes=5):
        return False
    if max_age_hours is not None and now - retrieved > timedelta(hours=max_age_hours):
        return False
    timestamps = payload.get("timestamps")
    values = payload.get("values")
    units = payload.get("units")
    expected_source = expected_query.get("source")
    if payload.get("source") != expected_source:
        return False
    if payload.get("product_type") != {
        "forecast": "forecast",
        "era5": "reanalysis",
        "satellite": "satellite_estimate",
    }.get(expected_source):
        return False
    if payload.get("requested_coordinates") != {
        "latitude": expected_query.get("latitude"),
        "longitude": expected_query.get("longitude"),
    }:
        return False
    returned = payload.get("returned_coordinates")
    if not isinstance(returned, dict) or "elevation" not in returned:
        return False
    for name in ("latitude", "longitude"):
        coordinate = returned.get(name)
        if (
            isinstance(coordinate, bool)
            or not isinstance(coordinate, (int, float))
            or not math.isfinite(coordinate)
        ):
            return False
    elevation = returned.get("elevation")
    if elevation is not None and (
        isinstance(elevation, bool)
        or not isinstance(elevation, (int, float))
        or not math.isfinite(elevation)
    ):
        return False
    if "model_issued_at" not in payload or not (
        payload["model_issued_at"] is None
        or _parse_timestamp(payload["model_issued_at"]) is not None
    ):
        return False
    if "actual_model" not in payload or "native_model_interval_seconds" not in payload:
        return False
    if not isinstance(payload.get("requested_model"), str):
        return False
    requested_model = expected_query.get("model") or {
        "forecast": "ecmwf_aifs025_single",
        "era5": "era5",
        "satellite": "eumetsat_sarah3",
    }.get(expected_source)
    if payload.get("requested_model") != requested_model:
        return False
    if payload.get("actual_model") is not None and not isinstance(payload.get("actual_model"), str):
        return False
    if not isinstance(payload.get("licence"), str) or not payload["licence"].strip():
        return False
    if not isinstance(payload.get("attribution"), str) or not payload["attribution"].strip():
        return False
    if (
        not isinstance(payload.get("interval_semantics"), str)
        or not payload["interval_semantics"].strip()
    ):
        return False
    if (
        not isinstance(timestamps, list)
        or not timestamps
        or not isinstance(values, dict)
        or not isinstance(units, dict)
    ):
        return False
    parsed_times = [_parse_timestamp(value) for value in timestamps]
    if any(value is None for value in parsed_times) or any(
        right <= left for left, right in zip(parsed_times, parsed_times[1:], strict=False)
    ):
        return False
    expected_unit = {
        "shortwave_radiation": "W/m²",
        "diffuse_radiation": "W/m²",
        "direct_normal_irradiance": "W/m²",
        "global_tilted_irradiance": "W/m²",
        "temperature_2m": "°C",
        "wind_speed_10m": "m/s",
    }
    resolution = expected_query.get("temporal_resolution")
    if resolution not in {"hourly", "native"}:
        return False
    expected_interval = 1800 if expected_source == "satellite" and resolution == "native" else 3600
    interval = payload.get("interval_seconds")
    if isinstance(interval, bool) or not isinstance(interval, int) or interval != expected_interval:
        return False
    if any(
        int((right - left).total_seconds()) != interval
        for left, right in zip(parsed_times, parsed_times[1:], strict=False)
    ):
        return False
    number_of_days = (
        date.fromisoformat(expected_query["end_date"])
        - date.fromisoformat(expected_query["start_date"])
    ).days + 1
    expected_count = number_of_days * (24 if expected_interval == 3600 else 48)
    if len(timestamps) != expected_count:
        return False
    expected_start = datetime.combine(
        date.fromisoformat(expected_query["start_date"]), datetime.min.time(), tzinfo=UTC
    )
    expected_end = expected_start + timedelta(seconds=expected_interval * (expected_count - 1))
    if parsed_times[0] != expected_start or parsed_times[-1] != expected_end:
        return False
    if any(
        not isinstance(timestamp, str) or not timestamp.endswith("Z") for timestamp in timestamps
    ):
        return False
    native_interval = payload.get("native_model_interval_seconds")
    if native_interval is not None and (
        isinstance(native_interval, bool)
        or not isinstance(native_interval, int)
        or native_interval <= 0
    ):
        return False
    variable_semantics = payload.get("variable_semantics")
    if not isinstance(variable_semantics, dict) or set(variable_semantics) != set(
        expected_query["variables"]
    ):
        return False
    for variable, semantics in variable_semantics.items():
        expected_semantics = (
            "mean_over_preceding_interval"
            if variable
            in {
                "shortwave_radiation",
                "diffuse_radiation",
                "direct_normal_irradiance",
                "global_tilted_irradiance",
            }
            else "value_at_timestamp"
        )
        if semantics != expected_semantics:
            return False
    for variable in expected_query["variables"]:
        series = values.get(variable)
        if (
            not isinstance(series, list)
            or len(series) != len(timestamps)
            or units.get(variable) != expected_unit.get(variable)
        ):
            return False
        if any(
            value is not None
            and (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
            )
            for value in series
        ):
            return False
    if payload.get("missing_counts") != {
        variable: sum(value is None for value in values[variable])
        for variable in expected_query["variables"]
    }:
        return False
    return True


def _load_manifest(
    path: Path, *, batch_hash: str, batch_spec: Mapping[str, Any], entries: Mapping[str, Any]
) -> dict[str, Any]:
    if not path.exists():
        return {
            "schema_version": MANIFEST_SCHEMA,
            "batch_spec_sha256": batch_hash,
            "batch_spec": batch_spec,
            "items": {
                key: {
                    "site_id": entry["site_id"],
                    "scenario_id": entry["scenario_id"],
                    "query": entry["query"],
                    "query_hash": entry["query_hash"],
                    "status": "pending",
                    "attempts": 0,
                }
                for key, entry in entries.items()
            },
            "counts": {"http_acquisitions": 0, "cache_reuse": 0},
        }
    manifest = _read_json(path)
    if not isinstance(manifest, dict) or manifest.get("schema_version") != MANIFEST_SCHEMA:
        raise SolarPortfolioError("existing manifest has an unsupported schema")
    if manifest.get("batch_spec_sha256") != batch_hash or _canonical_bytes(
        manifest.get("batch_spec")
    ) != _canonical_bytes(batch_spec):
        raise SolarPortfolioError(
            "output directory belongs to a different batch specification; use a new directory"
        )
    old_items = manifest.get("items")
    if not isinstance(old_items, dict) or set(old_items) != set(entries):
        raise SolarPortfolioError("manifest item ledger does not match the batch specification")
    for key, entry in entries.items():
        old = old_items[key]
        if (
            not isinstance(old, dict)
            or old.get("site_id") != entry["site_id"]
            or old.get("scenario_id") != entry["scenario_id"]
            or old.get("query") != entry["query"]
            or old.get("query_hash") != entry["query_hash"]
        ):
            raise SolarPortfolioError(
                "manifest query ledger does not match the batch specification"
            )
        if old.get("status") == "running":
            old.update(status="pending", error=None)
    manifest.setdefault("counts", {"http_acquisitions": 0, "cache_reuse": 0})
    return manifest


def _validate_manifest_items(manifest: Mapping[str, Any]) -> dict[str, Any]:
    batch_spec = manifest.get("batch_spec")
    items = manifest.get("items")
    if not isinstance(batch_spec, dict) or not isinstance(items, dict):
        raise SolarPortfolioError("invalid solar portfolio manifest")
    if manifest.get("batch_spec_sha256") != _sha256_bytes(_canonical_bytes(batch_spec)):
        raise SolarPortfolioError("manifest batch specification checksum is invalid")
    if (
        batch_spec.get("schema_version") != MANIFEST_SCHEMA
        or batch_spec.get("adapter_version") != ADAPTER_VERSION
    ):
        raise SolarPortfolioError("manifest batch specification is unsupported")
    source = batch_spec.get("source")
    endpoint_identity = batch_spec.get("endpoint_identity")
    if source not in ALLOWED_SOURCES or endpoint_identity not in {
        "injected-fetcher",
        f"open-meteo:{source}:public",
        f"open-meteo:{source}:customer",
    }:
        raise SolarPortfolioError("manifest provider identity is invalid")
    sites = batch_spec.get("sites")
    if not isinstance(sites, list) or not sites or len(sites) > MAX_SITES:
        raise SolarPortfolioError("manifest site ledger is invalid")
    expected: dict[str, dict[str, Any]] = {}
    for row in sites:
        if not isinstance(row, dict):
            raise SolarPortfolioError("manifest site ledger is invalid")
        site = SolarSite(
            site_id=row.get("site_id"),
            scenario_id=row.get("scenario_id", "default"),
            latitude=row.get("latitude"),
            longitude=row.get("longitude"),
            tilt=row.get("tilt"),
            azimuth=row.get("azimuth"),
        )
        key = _item_id(site.site_id, site.scenario_id)
        query = _query_for(
            site,
            source=batch_spec["source"],
            start_date=batch_spec["start_date"],
            end_date=batch_spec["end_date"],
            variables=batch_spec["variables"],
            model=batch_spec.get("model"),
            temporal_resolution=batch_spec["temporal_resolution"],
        )
        query_data = _normalise_query(query)
        expected[key] = {
            "site_id": site.site_id,
            "scenario_id": site.scenario_id,
            "query": query_data,
            "query_hash": _query_id(query_data, endpoint_identity=endpoint_identity),
        }
    if set(expected) != set(items):
        raise SolarPortfolioError("manifest item ledger does not match its batch specification")
    for key, wanted in expected.items():
        item = items[key]
        if not isinstance(item, dict) or any(item.get(field) != wanted[field] for field in wanted):
            raise SolarPortfolioError(
                "manifest item identity does not match its batch specification"
            )
        if item.get("status") not in {"pending", "running", "succeeded", "failed"}:
            raise SolarPortfolioError("manifest item status is invalid")
        if (
            isinstance(item.get("attempts"), bool)
            or not isinstance(item.get("attempts"), int)
            or item["attempts"] < 0
        ):
            raise SolarPortfolioError("manifest item attempt count is invalid")
    return items


def _reusable_artifact(
    batch_dir: Path,
    item: dict[str, Any],
    expected_query: Mapping[str, Any],
    *,
    max_age_hours: float | None,
    now: datetime,
) -> Path | None:
    if item.get("status") != "succeeded":
        return None
    digest = item.get("artifact_sha256")
    if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
        return None
    artifact_key = item.get("query_hash")
    if not isinstance(artifact_key, str) or not re.fullmatch(r"[0-9a-f]{64}", artifact_key):
        return None
    artifact_path = batch_dir / f"{artifact_key}.json"
    try:
        content = artifact_path.read_bytes()
    except OSError:
        return None
    if _sha256_bytes(content) != digest:
        return None
    try:
        payload = json.loads(content)
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    return (
        artifact_path
        if _valid_artifact(
            payload, expected_query=expected_query, max_age_hours=max_age_hours, now=now
        )
        else None
    )


def _recoverable_artifact(
    batch_dir: Path,
    query_hash: str,
    expected_query: Mapping[str, Any],
    *,
    expected_sha256: str | None,
    max_age_hours: float | None,
    now: datetime,
) -> tuple[Path, str] | None:
    """Find a fully valid artifact whose manifest success checkpoint was interrupted."""
    if not isinstance(expected_sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", expected_sha256):
        return None
    artifact_path = batch_dir / f"{query_hash}.json"
    try:
        content = artifact_path.read_bytes()
        payload = json.loads(content)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    if _sha256_bytes(content) != expected_sha256:
        return None
    if not _valid_artifact(
        payload, expected_query=expected_query, max_age_hours=max_age_hours, now=now
    ):
        return None
    return artifact_path, _sha256_bytes(content)


def _fetch_with_retry(
    fetcher: Callable[[SolarQuery], dict[str, Any]],
    query: SolarQuery,
    *,
    max_attempts: int,
    sleep: Callable[[float], None],
) -> tuple[dict[str, Any], int]:
    attempts = 0
    while True:
        attempts += 1
        try:
            result = fetcher(query)
            if not isinstance(result, dict):
                raise SolarPortfolioError("solar adapter returned a non-object result")
            return result, attempts
        except Exception as exc:
            if attempts >= max_attempts or not _error_record(exc)["retryable"]:
                raise _FetchFailure(exc, attempts=attempts) from None
            sleep(min(2.0 ** (attempts - 1), 4.0))


def run_solar_portfolio(
    *,
    sites_path: str | Path | None = None,
    sites: Sequence[Mapping[str, Any] | SolarSite] | None = None,
    source: str,
    start_date: str,
    end_date: str,
    variables: Sequence[str],
    output_dir: str | Path,
    model: str | None = None,
    temporal_resolution: str | None = None,
    concurrency: int = 1,
    max_attempts: int = 3,
    max_age_hours: float | None = None,
    fetcher: Callable[[SolarQuery], dict[str, Any]] | None = None,
    clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    """Collect every site/scenario into checksummed artifacts with crash safe resume."""
    if source not in ALLOWED_SOURCES:
        raise SolarPortfolioError("source must be forecast, era5, or satellite")
    if temporal_resolution is None:
        temporal_resolution = "native" if source == "satellite" else "hourly"
    if temporal_resolution not in {"hourly", "native"}:
        raise SolarPortfolioError("temporal_resolution must be hourly or native")
    if temporal_resolution == "native" and source != "satellite":
        raise SolarPortfolioError("native temporal resolution is only available for satellite")
    start_date = _iso_date(start_date, "start_date")
    end_date = _iso_date(end_date, "end_date")
    if start_date > end_date:
        raise SolarPortfolioError("start_date must not be after end_date")
    if (
        not variables
        or len(set(variables)) != len(variables)
        or any(value not in ALLOWED_VARIABLES for value in variables)
    ):
        raise SolarPortfolioError("variables must be unique supported API variable names")
    if not 1 <= concurrency <= 8:
        raise SolarPortfolioError("concurrency must be between 1 and 8")
    if not 1 <= max_attempts <= 3:
        raise SolarPortfolioError("max_attempts must be between 1 and 3")
    if max_age_hours is not None and (not math.isfinite(max_age_hours) or max_age_hours < 0):
        raise SolarPortfolioError("max_age_hours must be a finite non-negative number")
    if source == "forecast" and max_age_hours is None:
        max_age_hours = 6.0

    # Input validation completes before output directory creation, locking, or HTTP calls.
    if (sites_path is None) == (sites is None):
        raise SolarPortfolioError("provide exactly one of sites_path or sites")
    if sites_path is not None:
        sites = load_solar_sites(sites_path, variables=variables)
    else:
        assert sites is not None
        sites = validate_solar_sites(sites, variables=variables)
    if fetcher is None:
        from src.solar import SolarSettings

        endpoint_url, _api_key = SolarSettings().endpoint_for(source)
        endpoint_host = urlsplit(endpoint_url).hostname or ""
        endpoint_class = (
            "customer"
            if endpoint_host.startswith("customer-")
            or endpoint_host == "customer-api.open-meteo.com"
            else "public"
        )
        endpoint_identity = f"open-meteo:{source}:{endpoint_class}"
    else:
        endpoint_identity = "injected-fetcher"
    batch_spec = _batch_spec(
        source=source,
        start_date=start_date,
        end_date=end_date,
        variables=variables,
        model=model,
        sites=sites,
        endpoint_identity=endpoint_identity,
        temporal_resolution=temporal_resolution,
    )
    batch_hash = _sha256_bytes(_canonical_bytes(batch_spec))
    entries: dict[str, dict[str, Any]] = {}
    for site in sites:
        query = _query_for(
            site,
            source=source,
            start_date=start_date,
            end_date=end_date,
            variables=variables,
            model=model,
            temporal_resolution=temporal_resolution,
        )
        query_data = _normalise_query(query)
        item_key = _item_id(site.site_id, site.scenario_id)
        entries[item_key] = {
            "site_id": site.site_id,
            "scenario_id": site.scenario_id,
            "query": query_data,
            "query_hash": _query_id(query_data, endpoint_identity=endpoint_identity),
            "query_object": query,
        }

    try:
        import fcntl
    except ImportError:
        raise SolarPortfolioError(
            "solar collection and export require POSIX file locking"
        ) from None

    batch_dir = Path(output_dir)
    batch_dir.mkdir(parents=True, exist_ok=True)
    lock_path = batch_dir / ".lock"
    with lock_path.open("a+b") as lock_stream:
        try:
            fcntl.flock(lock_stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise SolarPortfolioError("another process is running this solar batch") from exc

        manifest_path = batch_dir / "manifest.json"
        manifest = _load_manifest(
            manifest_path, batch_hash=batch_hash, batch_spec=batch_spec, entries=entries
        )
        manifest["updated_at"] = clock().astimezone(UTC).isoformat()
        _atomic_json(manifest_path, manifest)
        if fetcher is None:
            from src.solar import fetch_solar

            fetcher = fetch_solar

        groups: dict[str, list[tuple[str, dict[str, Any]]]] = {}
        for item_key, entry in entries.items():
            groups.setdefault(entry["query_hash"], []).append((item_key, entry))
        reusable: set[str] = set()
        now = clock().astimezone(UTC)
        for query_hash, group in groups.items():
            source_item = next(
                (
                    manifest["items"][item_key]
                    for item_key, _entry in group
                    if _reusable_artifact(
                        batch_dir,
                        manifest["items"][item_key],
                        _entry["query"],
                        max_age_hours=max_age_hours,
                        now=now,
                    )
                ),
                None,
            )
            if source_item is not None:
                for item_key, _entry in group:
                    item = manifest["items"][item_key]
                    item.update(
                        status="succeeded",
                        query_hash=query_hash,
                        artifact_sha256=source_item["artifact_sha256"],
                        retrieved_at=source_item.get("retrieved_at"),
                        error=None,
                    )
                    reusable.add(item_key)
                manifest["counts"]["cache_reuse"] = int(
                    manifest["counts"].get("cache_reuse", 0)
                ) + len(group)
            else:
                # A fetch can finish and atomically publish its artifact just before
                # the success ledger checkpoint. Recover only interrupted/pending
                # requests; a succeeded entry with a bad checksum must be refetched.
                interrupted = all(
                    manifest["items"][item_key].get("status") in {"pending", "running"}
                    for item_key, _entry in group
                )
                expected_digests = {
                    manifest["items"][item_key].get("artifact_sha256") for item_key, _entry in group
                }
                expected_digest = (
                    next(iter(expected_digests)) if len(expected_digests) == 1 else None
                )
                recovered = (
                    _recoverable_artifact(
                        batch_dir,
                        query_hash,
                        group[0][1]["query"],
                        expected_sha256=expected_digest,
                        max_age_hours=max_age_hours,
                        now=now,
                    )
                    if interrupted
                    else None
                )
                if recovered is not None:
                    _artifact_path, digest = recovered
                    # The durable pre-artifact checkpoint records acquisition cost.
                    # Older ledgers did not have that checkpoint, so count the one
                    # successful response evidenced by this orphan artifact once.
                    if not any(
                        manifest["items"][item_key].get("acquisition_counted") is True
                        for item_key, _entry in group
                    ):
                        manifest["counts"]["http_acquisitions"] = (
                            int(manifest["counts"].get("http_acquisitions", 0)) + 1
                        )
                    for item_key, _entry in group:
                        item = manifest["items"][item_key]
                        item.update(
                            status="succeeded",
                            query_hash=query_hash,
                            artifact_sha256=digest,
                            retrieved_at=_read_json(_artifact_path).get("retrieved_at"),
                            acquisition_counted=True,
                            error=None,
                        )
                        reusable.add(item_key)
                    manifest["counts"]["cache_reuse"] = int(
                        manifest["counts"].get("cache_reuse", 0)
                    ) + len(group)
                    _atomic_json(manifest_path, manifest)
                    continue
                for item_key, _entry in group:
                    manifest["items"][item_key].update(
                        status="pending",
                        query_hash=query_hash,
                        artifact_sha256=None,
                        retrieved_at=None,
                    )
        _atomic_json(manifest_path, manifest)

        def execute(
            query_hash: str, group: list[tuple[str, dict[str, Any]]]
        ) -> tuple[str, dict[str, Any], int]:
            result, attempts = _fetch_with_retry(
                fetcher, group[0][1]["query_object"], max_attempts=max_attempts, sleep=sleep
            )
            return query_hash, result, attempts

        pending = [
            (query_hash, group)
            for query_hash, group in groups.items()
            if not all(item_key in reusable for item_key, _entry in group)
        ]
        failures = 0
        # Submit bounded waves. The durable ledger is updated before the corresponding call.
        for offset in range(0, len(pending), concurrency):
            wave = pending[offset : offset + concurrency]
            futures = {}
            attempts_before: dict[str, int] = {}
            for _query_hash, group in wave:
                for item_key, _entry in group:
                    item = manifest["items"][item_key]
                    attempts_before[item_key] = int(item.get("attempts", 0))
                    item.update(
                        status="running",
                        attempts=attempts_before[item_key] + 1,
                        acquisition_counted=False,
                        error=None,
                    )
            _atomic_json(manifest_path, manifest)
            with ThreadPoolExecutor(max_workers=concurrency) as pool:
                for query_hash, group in wave:
                    futures[pool.submit(execute, query_hash, group)] = (query_hash, group)
                for future in as_completed(futures):
                    query_hash, group = futures[future]
                    attempts = 1
                    try:
                        _, result, attempts = future.result()
                        if not _valid_artifact(
                            result,
                            expected_query=group[0][1]["query"],
                            max_age_hours=None,
                            now=clock().astimezone(UTC),
                        ):
                            raise SolarPortfolioError(
                                "solar adapter returned an invalid or mismatched artifact"
                            )
                        # Persist actual provider attempts before publishing the
                        # artifact. If the process stops after the artifact replace,
                        # resume can salvage it without guessing or double counting.
                        if not any(
                            manifest["items"][item_key].get("acquisition_counted") is True
                            for item_key, _entry in group
                        ):
                            manifest["counts"]["http_acquisitions"] = (
                                int(manifest["counts"].get("http_acquisitions", 0)) + attempts
                            )
                        for item_key, _entry in group:
                            manifest["items"][item_key].update(
                                attempts=attempts_before[item_key] + attempts,
                                acquisition_counted=True,
                                artifact_sha256=_sha256_bytes(_canonical_bytes(result) + b"\n"),
                            )
                        _atomic_json(manifest_path, manifest)
                        artifact_path = batch_dir / f"{query_hash}.json"
                        digest = _atomic_json(artifact_path, result)
                        retrieved_at = result.get("retrieved_at")
                        for item_key, _entry in group:
                            manifest["items"][item_key].update(
                                status="succeeded",
                                attempts=attempts_before[item_key] + attempts,
                                query_hash=query_hash,
                                retrieved_at=retrieved_at,
                                artifact_sha256=digest,
                                acquisition_counted=True,
                                error=None,
                            )
                    except Exception as exc:
                        failures += 1
                        record = _error_record(exc)
                        if isinstance(exc, _FetchFailure):
                            attempts = exc.attempts
                        acquisition_was_recorded = any(
                            manifest["items"][item_key].get("acquisition_counted") is True
                            for item_key, _entry in group
                        )
                        for item_key, _entry in group:
                            manifest["items"][item_key].update(
                                status="failed",
                                attempts=attempts_before[item_key] + attempts,
                                query_hash=query_hash,
                                acquisition_counted=True,
                                error=record,
                            )
                        if not acquisition_was_recorded:
                            manifest["counts"]["http_acquisitions"] = (
                                int(manifest["counts"].get("http_acquisitions", 0)) + attempts
                            )
                    manifest["updated_at"] = clock().astimezone(UTC).isoformat()
                    _atomic_json(manifest_path, manifest)

        summary = {
            "batch_spec_sha256": batch_hash,
            "output_dir": str(batch_dir),
            "total": len(entries),
            "succeeded": sum(
                item.get("status") == "succeeded" for item in manifest["items"].values()
            ),
            "failed": sum(item.get("status") == "failed" for item in manifest["items"].values()),
            "http_acquisitions": manifest["counts"].get("http_acquisitions", 0),
            "cache_reuse": manifest["counts"].get("cache_reuse", 0),
            "failures_this_run": failures,
        }
        manifest["summary"] = summary
        manifest["updated_at"] = clock().astimezone(UTC).isoformat()
        _atomic_json(manifest_path, manifest)
        return summary


def export_solar_portfolio(output_dir: str | Path, destination: str | Path | None = None) -> Path:
    """Export validated successful artifacts to long-form CSV, preserving null and zero."""
    try:
        import fcntl
    except ImportError:
        raise SolarPortfolioError(
            "solar collection and export require POSIX file locking"
        ) from None

    batch_dir = Path(output_dir)
    manifest_path = batch_dir / "manifest.json"
    lock_path = batch_dir / ".lock"
    with lock_path.open("a+b") as lock_stream:
        try:
            fcntl.flock(lock_stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise SolarPortfolioError("another process is running this solar batch") from exc
        manifest = _read_json(manifest_path)
        if not isinstance(manifest, dict) or manifest.get("schema_version") != MANIFEST_SCHEMA:
            raise SolarPortfolioError("invalid solar portfolio manifest")
        items = _validate_manifest_items(manifest)
        destination_path = Path(destination) if destination else batch_dir / "solar-portfolio.csv"
        protected = {manifest_path.resolve(), (batch_dir / ".lock").resolve()}
        protected.update(
            (batch_dir / f"{item['query_hash']}.json").resolve()
            for item in items.values()
            if isinstance(item.get("query_hash"), str)
            and re.fullmatch(r"[0-9a-f]{64}", item["query_hash"])
        )
        if destination_path.resolve() in protected:
            raise SolarPortfolioError("export destination conflicts with portfolio metadata")
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        fd, temp_name = tempfile.mkstemp(
            prefix=f".{destination_path.name}.", suffix=".tmp", dir=destination_path.parent
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=EXPORT_FIELDS, lineterminator="\n")
                writer.writeheader()
                for key in sorted(items):
                    item = items[key]
                    if item.get("status") != "succeeded":
                        continue
                    artifact = _reusable_artifact(
                        batch_dir,
                        item,
                        item.get("query", {}),
                        max_age_hours=None,
                        now=datetime.now(UTC),
                    )
                    if artifact is None:
                        raise SolarPortfolioError(
                            f"artifact for {item.get('site_id', 'unknown')} is missing, stale or corrupt"
                        )
                    payload = _read_json(artifact)
                    query = payload["query"]
                    for variable in query["variables"]:
                        series = payload["values"][variable]
                        unit = payload["units"][variable]
                        for timestamp, value in zip(payload["timestamps"], series, strict=True):
                            writer.writerow(
                                {
                                    "site_id": item["site_id"],
                                    "scenario_id": item["scenario_id"],
                                    "tilt": "" if query["tilt"] is None else query["tilt"],
                                    "azimuth": "" if query["azimuth"] is None else query["azimuth"],
                                    "time": timestamp,
                                    "variable": variable,
                                    "value": "" if value is None else value,
                                    "unit": unit,
                                    "source": query["source"],
                                    "product_type": payload["product_type"],
                                    "requested_model": payload["requested_model"],
                                    "actual_model": ""
                                    if payload["actual_model"] is None
                                    else payload["actual_model"],
                                    "model_issued_at": payload["model_issued_at"] or "",
                                    "retrieved_at": payload["retrieved_at"],
                                    "requested_latitude": payload["requested_coordinates"][
                                        "latitude"
                                    ],
                                    "requested_longitude": payload["requested_coordinates"][
                                        "longitude"
                                    ],
                                    "returned_latitude": payload["returned_coordinates"][
                                        "latitude"
                                    ],
                                    "returned_longitude": payload["returned_coordinates"][
                                        "longitude"
                                    ],
                                    "returned_elevation": ""
                                    if payload["returned_coordinates"]["elevation"] is None
                                    else payload["returned_coordinates"]["elevation"],
                                    "temporal_resolution": query["temporal_resolution"],
                                    "interval_seconds": payload["interval_seconds"],
                                    "native_model_interval_seconds": ""
                                    if payload["native_model_interval_seconds"] is None
                                    else payload["native_model_interval_seconds"],
                                    "interval_semantics": payload["interval_semantics"],
                                    "variable_semantics": payload["variable_semantics"][variable],
                                    "licence": payload["licence"],
                                    "attribution": payload["attribution"],
                                }
                            )
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temp_name, destination_path)
        finally:
            try:
                os.unlink(temp_name)
            except FileNotFoundError:
                pass
        return destination_path


__all__ = [
    "ARTIFACT_SCHEMA",
    "MANIFEST_SCHEMA",
    "SolarPortfolioError",
    "SolarSite",
    "export_solar_portfolio",
    "load_solar_sites",
    "validate_solar_sites",
    "run_solar_portfolio",
]
