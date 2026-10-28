"""Bounded synchronous solar acquisition for the GeoAI solar pilot."""

from __future__ import annotations

import json
import logging
import math
import re
from datetime import UTC, datetime, timedelta
from typing import Any
from urllib.parse import urlsplit

import httpx
from pydantic_settings import BaseSettings, SettingsConfigDict

from src.solar_contracts import SolarQuery

_DEFAULTS = {
    "forecast": "https://api.open-meteo.com/v1/forecast",
    "era5": "https://archive-api.open-meteo.com/v1/archive",
    "satellite": "https://satellite-api.open-meteo.com/v1/archive",
}
_PATHS = {"forecast": "/v1/forecast", "era5": "/v1/archive", "satellite": "/v1/archive"}
_PUBLIC_HOSTS = {
    "forecast": {"api.open-meteo.com"},
    "era5": {"archive-api.open-meteo.com"},
    "satellite": {"satellite-api.open-meteo.com"},
}
_CUSTOMER_HOSTS = {
    "forecast": "customer-api.open-meteo.com",
    "era5": "customer-archive-api.open-meteo.com",
    "satellite": "customer-satellite-api.open-meteo.com",
}
_MODELS = {"forecast": "ecmwf_aifs025_single", "era5": "era5", "satellite": "eumetsat_sarah3"}
_NATIVE_MODEL_INTERVALS = {
    "ecmwf_aifs025_single": 21_600,
    "era5": 3_600,
    "eumetsat_sarah3": 1_800,
}
_IRRADIANCE_VARIABLES = {
    "shortwave_radiation",
    "diffuse_radiation",
    "direct_normal_irradiance",
    "global_tilted_irradiance",
}
_UNITS = {
    "shortwave_radiation": "W/m²",
    "diffuse_radiation": "W/m²",
    "direct_normal_irradiance": "W/m²",
    "global_tilted_irradiance": "W/m²",
    "temperature_2m": "°C",
    "wind_speed_10m": "m/s",
}
_PRODUCTS = {"forecast": "forecast", "era5": "reanalysis", "satellite": "satellite_estimate"}
_LICENSE = "CC BY 4.0"
_MODEL_ATTRIBUTION = {
    ("forecast", "ecmwf_aifs025_single"): "Open-Meteo; ECMWF AIFS forecast",
    ("era5", "era5"): "Open-Meteo; ERA5 from ECMWF/Copernicus Climate Change Service",
    ("satellite", "eumetsat_sarah3"): "Open-Meteo; EUMETSAT SARAH-3",
}
_TIMEOUT_SECONDS = 20.0
_MAX_RESPONSE_BYTES = 16 * 1024 * 1024
_MAX_ROWS = 40_000
_HTTPX_API_KEY_PATTERN = re.compile(r"(?i)([?&]apikey=)[^&\s\"'<>)]*")


class _HttpxApiKeyRedactionFilter(logging.Filter):
    """Redact API-key query parameters from HTTPX's own request log records.

    This stateless filter is scoped to the ``httpx`` logger. It does not alter
    the repository-wide logging configuration, and it safely covers concurrent
    requests without maintaining a process-global set of live credentials.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            message = record.getMessage()
        except Exception:  # pragma: no cover - defensive logging boundary
            return True
        redacted = _HTTPX_API_KEY_PATTERN.sub(r"\1[REDACTED]", message)
        if redacted != message:
            record.msg = redacted
            record.args = ()
        return True


def _install_httpx_api_key_redaction() -> None:
    logger = logging.getLogger("httpx")
    if not any(isinstance(item, _HttpxApiKeyRedactionFilter) for item in logger.filters):
        logger.addFilter(_HttpxApiKeyRedactionFilter())


class SolarSettings(BaseSettings):
    """Independent environment-backed endpoints and keys for solar providers."""

    model_config = SettingsConfigDict(
        env_file=".env.solar",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="forbid",
    )

    SOLAR_FORECAST_API_BASE_URL: str = _DEFAULTS["forecast"]
    SOLAR_FORECAST_API_KEY: str = ""
    SOLAR_ARCHIVE_API_BASE_URL: str = _DEFAULTS["era5"]
    SOLAR_ARCHIVE_API_KEY: str = ""
    SOLAR_SATELLITE_API_BASE_URL: str = _DEFAULTS["satellite"]
    SOLAR_SATELLITE_API_KEY: str = ""

    def endpoint_for(self, source: str) -> tuple[str, str]:
        fields = {
            "forecast": (self.SOLAR_FORECAST_API_BASE_URL, self.SOLAR_FORECAST_API_KEY),
            "era5": (self.SOLAR_ARCHIVE_API_BASE_URL, self.SOLAR_ARCHIVE_API_KEY),
            "satellite": (self.SOLAR_SATELLITE_API_BASE_URL, self.SOLAR_SATELLITE_API_KEY),
        }
        return fields[source]


class SolarFetchError(RuntimeError):
    """Safe provider error that contains no request URL, response body, or credentials."""

    def __init__(self, message: str, *, retryable: bool = False) -> None:
        super().__init__(message)
        self.retryable = retryable


def _validated_endpoint(source: str, raw_url: str, api_key: str) -> tuple[str, bool]:
    parse_failed = False
    try:
        parsed = urlsplit(raw_url.strip())
        host = parsed.hostname.lower() if parsed.hostname else ""
        port = parsed.port
    except (ValueError, AttributeError):
        parse_failed = True
        parsed = None
        host = ""
        port = None
    if parse_failed:
        raise SolarFetchError("Solar API endpoint is invalid")
    assert parsed is not None

    expected_path = _PATHS[source]
    customer_host = host == _CUSTOMER_HOSTS[source]
    public_host = host in _PUBLIC_HOSTS[source]
    valid = (
        parsed.scheme.lower() == "https"
        and port in (None, 443)
        and parsed.username is None
        and parsed.password is None
        and parsed.path.rstrip("/") == expected_path
        and not parsed.query
        and not parsed.fragment
        and (customer_host or public_host)
    )
    if not valid:
        raise SolarFetchError("Solar API endpoint is not an approved HTTPS provider endpoint")
    if api_key and not customer_host:
        raise SolarFetchError("Solar API key requires the approved customer endpoint")
    return raw_url.strip().rstrip("/"), customer_host


def _requested_model(query: SolarQuery) -> str:
    return query.model or _MODELS[query.source]


def _timestamp(value: Any) -> datetime:
    if not isinstance(value, str):
        raise ValueError("timestamp is not text")
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _provider_model(payload: dict[str, Any]) -> str | None:
    raw = payload.get("model", payload.get("models"))
    if isinstance(raw, str) and raw.strip():
        return raw.strip()
    if isinstance(raw, list) and len(raw) == 1 and isinstance(raw[0], str):
        return raw[0]
    return None


def _normalise_payload(payload: Any, query: SolarQuery) -> dict[str, Any]:
    if not isinstance(payload, dict) or payload.get("error"):
        raise ValueError("provider returned an invalid response")

    # Open-Meteo's satellite endpoint returns SARAH-3 native samples in `hourly`
    # at a documented 30-minute cadence; `minutely_15` is not its native product.
    series_key = "hourly"
    series = payload.get(series_key)
    units = payload.get(f"{series_key}_units")
    if not isinstance(series, dict) or not isinstance(units, dict):
        raise ValueError("provider response has no requested time series")
    raw_times = series.get("time")
    if not isinstance(raw_times, list) or not raw_times or len(raw_times) > _MAX_ROWS:
        raise ValueError("provider response has an invalid timestamp series")
    timestamps = [_timestamp(value) for value in raw_times]
    if any(right <= left for left, right in zip(timestamps, timestamps[1:], strict=False)):
        raise ValueError("provider timestamps are not strictly increasing")

    expected_interval = (
        1800 if query.source == "satellite" and query.temporal_resolution == "native" else 3600
    )
    intervals = [
        int((right - left).total_seconds())
        for left, right in zip(timestamps, timestamps[1:], strict=False)
    ]
    if any(interval != expected_interval for interval in intervals):
        raise ValueError("provider timestamps have an irregular interval")
    if len(timestamps) > _MAX_ROWS:
        raise ValueError("provider response exceeds the row limit")

    expected_start = datetime.combine(query.start_date, datetime.min.time(), tzinfo=UTC)
    expected_end = datetime.combine(query.end_date, datetime.min.time(), tzinfo=UTC) + timedelta(
        hours=23,
        minutes=30 if expected_interval == 1800 else 0,
    )
    if timestamps[0] != expected_start or timestamps[-1] != expected_end:
        raise ValueError("provider timestamps do not cover the requested dates")

    values: dict[str, list[float | None]] = {}
    missing_counts: dict[str, int] = {}
    for variable in query.variables:
        if units.get(variable) != _UNITS[variable]:
            raise ValueError("provider returned an unexpected variable unit")
        column = series.get(variable)
        if not isinstance(column, list) or len(column) != len(timestamps):
            raise ValueError("provider returned a missing or ragged variable")
        normalized: list[float | None] = []
        for value in column:
            if value is None:
                normalized.append(None)
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError("provider returned a non-numeric value")
            numeric = float(value)
            if not math.isfinite(numeric):
                raise ValueError("provider returned a non-finite value")
            normalized.append(numeric)
        values[variable] = normalized
        missing_counts[variable] = sum(value is None for value in normalized)

    coordinates: dict[str, float | None] = {}
    for name in ("latitude", "longitude", "elevation"):
        raw = payload.get(name)
        if raw is None and name == "elevation":
            coordinates[name] = None
            continue
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            raise ValueError("provider returned invalid coordinates")
        numeric = float(raw)
        if not math.isfinite(numeric):
            raise ValueError("provider returned invalid coordinates")
        if name == "latitude" and not -90 <= numeric <= 90:
            raise ValueError("provider returned invalid coordinates")
        if name == "longitude" and not -180 <= numeric <= 180:
            raise ValueError("provider returned invalid coordinates")
        coordinates[name] = numeric

    if payload.get("timezone") not in {"UTC", "GMT"} or payload.get("utc_offset_seconds") != 0:
        raise ValueError("provider did not honor the UTC request")

    actual_model = _provider_model(payload)
    if query.source == "forecast":
        attribution = _MODEL_ATTRIBUTION.get((query.source, actual_model or ""), "Open-Meteo")
    else:
        # ERA5 and SARAH-3 are source contracts, not configurable model guesses.
        # Keep their required upstream acknowledgements even if the API omits
        # its optional model field.
        attribution = _MODEL_ATTRIBUTION[(query.source, _MODELS[query.source])]

    return {
        "schema_version": "geoai-solar/v1",
        "query": query.model_dump(mode="json"),
        "product_type": _PRODUCTS[query.source],
        "requested_coordinates": {"latitude": query.latitude, "longitude": query.longitude},
        "returned_coordinates": coordinates,
        "timestamps": [value.isoformat().replace("+00:00", "Z") for value in timestamps],
        "values": values,
        "units": {variable: _UNITS[variable] for variable in query.variables},
        "interval_seconds": expected_interval,
        "native_model_interval_seconds": _NATIVE_MODEL_INTERVALS.get(_requested_model(query)),
        "interval_semantics": "provider-served temporal cadence; no additional aggregation or interpolation",
        "variable_semantics": {
            variable: (
                "mean_over_preceding_interval"
                if variable in _IRRADIANCE_VARIABLES
                else "value_at_timestamp"
            )
            for variable in query.variables
        },
        "model_issued_at": None,
        "requested_model": _requested_model(query),
        "actual_model": actual_model,
        "source": query.source,
        "licence": _LICENSE,
        "attribution": attribution,
        "missing_counts": missing_counts,
    }


def _read_bounded(response: httpx.Response) -> bytes:
    chunks: list[bytes] = []
    size = 0
    for chunk in response.iter_bytes(chunk_size=64 * 1024):
        size += len(chunk)
        if size > _MAX_RESPONSE_BYTES:
            raise ValueError("provider response exceeds the byte limit")
        chunks.append(chunk)
    return b"".join(chunks)


def fetch_solar(
    query: SolarQuery,
    *,
    config: SolarSettings | None = None,
    client: httpx.Client | None = None,
) -> dict[str, Any]:
    """Fetch and validate one solar time series without source fallback or daily conversion."""
    settings = config or SolarSettings()
    raw_url, configured_key = settings.endpoint_for(query.source)
    api_key = configured_key.strip()
    url, customer_endpoint = _validated_endpoint(query.source, raw_url, api_key)
    params: dict[str, str | float] = {
        "latitude": query.latitude,
        "longitude": query.longitude,
        "start_date": query.start_date.isoformat(),
        "end_date": query.end_date.isoformat(),
        "timezone": "UTC",
        "wind_speed_unit": "ms",
        "models": _requested_model(query),
    }
    params["hourly"] = ",".join(query.variables)
    if query.source == "satellite":
        params["temporal_resolution"] = query.temporal_resolution
    if "global_tilted_irradiance" in query.variables:
        assert query.tilt is not None and query.azimuth is not None
        params["tilt"] = query.tilt
        params["azimuth"] = query.azimuth
    if customer_endpoint and api_key:
        params["apikey"] = api_key

    owns_client = client is None
    http = client or httpx.Client(timeout=_TIMEOUT_SECONDS, follow_redirects=False)
    _install_httpx_api_key_redaction()
    request_failure: str | None = None
    try:
        try:
            with http.stream(
                "GET", url, params=params, timeout=_TIMEOUT_SECONDS, follow_redirects=False
            ) as response:
                if response.status_code < 200 or response.status_code >= 300:
                    status = response.status_code
                    raise SolarFetchError(
                        f"Solar API returned HTTP {status}",
                        retryable=status == 429 or 500 <= status <= 599,
                    )
                try:
                    body = _read_bounded(response)
                except ValueError:
                    raise SolarFetchError("Solar API response is invalid") from None
        except SolarFetchError:
            raise
        except httpx.TimeoutException:
            request_failure = "Solar API request timed out"
        except httpx.TransportError:
            request_failure = "Solar API request failed"
        if request_failure is not None:
            raise SolarFetchError(request_failure, retryable=True)
        response_invalid = False
        try:
            payload = json.loads(body)
            result = _normalise_payload(payload, query)
        except (ValueError, TypeError, OverflowError, json.JSONDecodeError):
            response_invalid = True
            result = {}
        if response_invalid:
            raise SolarFetchError("Solar API response is invalid")
        result["retrieved_at"] = datetime.now(UTC).isoformat().replace("+00:00", "Z")
        return result
    finally:
        if owns_client:
            http.close()
