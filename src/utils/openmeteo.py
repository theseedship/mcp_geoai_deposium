"""Open-Meteo **Archive** API client (ERA5 reanalysis) — FIERCE lot B2.

A thin, synchronous httpx client for ``https://archive-api.open-meteo.com/v1/archive``,
mirroring :mod:`src.utils.stac_client`: a module-level singleton + blocking ``fetch_*``
functions that the async tool handler offloads via ``asyncio.to_thread``.

The **six daily variables are LOCKED** to ``terra-cognita §5.1`` (the BEJOF↔INRIA exchange
schema). :data:`PROVIDER_VARS` is the single source of truth: it maps each *normalized* domain
column to its Open-Meteo *provider* variable, its stored unit, and any unit conversion. The
``models=era5`` model is **forced** (never ``best_match``, which can stitch different models across
a multi-decade range and break comparability), and ``wind_speed_unit=ms`` is forced (the API
default is km/h). At ingestion a **provider rename fails loudly** (a missing ``daily.<var>`` key
raises ``KeyError`` rather than silently producing nulls); the exact variable **names and units**
are additionally pinned by a live, skipped-offline contract probe (it asserts ``daily_units``
against :attr:`ProviderVar.source_unit`), so a unit change fails loudly when that probe runs.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, timedelta

import httpx
import structlog

log = structlog.get_logger(__name__)

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
# Open-Meteo **Forecast** endpoint (FIERCE B5). Overridable per-call (provider passes
# settings.FORECAST_API_BASE_URL → customer-api.* for the commercial tier).
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
# Data attribution is independent from the operator's API subscription tier. Runtime
# tool results must not infer whether the configured endpoint is free or commercial.
OPEN_METEO_DATA_LICENSE = "CC-BY 4.0 (Open-Meteo)"

# Transport tuning (the network-free tests monkeypatch fetch_archive_daily, so these
# are only exercised against the live API).
_TIMEOUT_S = 60.0
_MAX_RETRIES = 3
_BACKOFF_BASE_S = 1.5
# Forecast transport — a forecast is ~16 rows, so a SHORT timeout fails fast to the next provider
# (the router's fallback). Same 3× backoff loop as the Archive client.
_FORECAST_TIMEOUT_S = 5.0
_FORECAST_MAX_RETRIES = 3
_FORECAST_BACKOFF_BASE_S = 1.5
# Open-Meteo Archive happily serves multi-decade ranges in one call; we still chunk long
# ranges (guard-rail per the B2 design §3.5) to bound any single request.
_MAX_DAYS_PER_REQUEST = 366


@dataclass(frozen=True)
class ProviderVar:
    """One locked normalized↔provider variable mapping (terra-cognita §5.1)."""

    normalized: str  # the stored, domain-normalized column name
    provider: str  # the Open-Meteo daily variable name
    unit: str  # the unit we STORE (after any conversion)
    # the unit the API returns in `daily_units` (pre-conversion) — pinned to catch drift
    source_unit: str
    note: str = ""


# LOCKED to terra-cognita §5.1 — order is the canonical schema order. No `wind_direction`.
# `source_unit` is the value the Archive API reports in `daily_units` for that provider variable
# (asserted by the live contract probe); `unit` is what we store after any conversion.
PROVIDER_VARS: tuple[ProviderVar, ...] = (
    ProviderVar("temperature_2m", "temperature_2m_mean", "°C", "°C", "API default"),
    ProviderVar("precipitation", "precipitation_sum", "mm", "mm", "API default"),
    ProviderVar("humidity", "relative_humidity_2m_mean", "%", "%"),
    ProviderVar("evapotranspiration", "et0_fao_evapotranspiration", "mm", "mm", "ET0 FAO"),
    ProviderVar("wind_speed", "wind_speed_10m_max", "m/s", "m/s", "forced wind_speed_unit=ms"),
    ProviderVar(
        "solar_radiation",
        "shortwave_radiation_sum",
        "W/m²",
        "MJ/m²",
        "API returns MJ/m²; stored as W/m² daily-mean = MJ/m² × 1e6 / 86400",
    ),
)

# Fast lookups
_BY_NORMALIZED: dict[str, ProviderVar] = {v.normalized: v for v in PROVIDER_VARS}
NORMALIZED_VARS: tuple[str, ...] = tuple(v.normalized for v in PROVIDER_VARS)

# Operational forecast extension only. The research/archive schema remains the six above.
FORECAST_ONLY_PROVIDER_VARS: tuple[ProviderVar, ...] = (
    ProviderVar(
        "temperature_2m_min",
        "temperature_2m_min",
        "°C",
        "°C",
        "Minimum air temperature at 2 m over the full UTC calendar day; not night-specific",
    ),
)
_FORECAST_ONLY_BY_NORMALIZED = {v.normalized: v for v in FORECAST_ONLY_PROVIDER_VARS}
FORECAST_NORMALIZED_VARS = NORMALIZED_VARS + tuple(_FORECAST_ONLY_BY_NORMALIZED)

# Seconds in a day — the divisor for MJ/m² (daily sum) → W/m² (daily mean).
_SECONDS_PER_DAY = 86_400.0
_MJ_TO_J = 1_000_000.0


def provider_var(normalized: str) -> ProviderVar:
    """Resolve a normalized name to its locked :class:`ProviderVar` (raises on unknown)."""
    try:
        return _BY_NORMALIZED[normalized]
    except KeyError:
        raise ValueError(
            f"Unknown climate variable {normalized!r}; "
            f"supported (terra §5.1): {', '.join(NORMALIZED_VARS)}"
        ) from None


def convert_to_stored_unit(normalized: str, raw_value: float | None) -> float | None:
    """Apply the documented unit conversion for a single raw provider value.

    Only ``solar_radiation`` is converted (MJ/m² daily sum → W/m² daily mean); every other
    variable is stored in the API's native unit. ``None`` (a missing daily record) passes through.
    """
    if raw_value is None:
        return None
    if normalized == "solar_radiation":
        return raw_value * _MJ_TO_J / _SECONDS_PER_DAY
    return float(raw_value)


def forecast_provider_var(normalized: str) -> ProviderVar:
    """Resolve a live forecast variable without extending the canonical archive schema."""
    if normalized in _FORECAST_ONLY_BY_NORMALIZED:
        return _FORECAST_ONLY_BY_NORMALIZED[normalized]
    return provider_var(normalized)


def forecast_provider_map(normalized_variables: list[str]) -> dict:
    """Trace forecast values to provider names, stored units and aggregation notes."""
    out = {}
    for variable in normalized_variables:
        pv = forecast_provider_var(variable)
        out[variable] = {"provider": pv.provider, "unit": pv.unit, "note": pv.note}
    return out


def normalize_openmeteo_daily(
    raw: dict,
    normalized_variables: list[str],
    *,
    strict: bool,
    variable_resolver: Callable[[str], ProviderVar] = provider_var,
) -> tuple[list[str], dict[str, list[float | None]]]:
    """Normalize ONE Open-Meteo daily response → (dates, {normalized_var: [stored-unit values]}).

    The shared response→columns mapper for both the **Archive** (B2) and **Forecast** (B5) paths
    (design §4.4 — moved DOWN to this layer so the forecast client doesn't import
    ``climate.py``, which would be a cycle). Applies the documented unit conversions (only
    ``solar_radiation`` MJ/m² → W/m²). The ``strict`` flag is the only behavioural difference:

    - ``strict=True`` (Archive ingestion): a requested provider variable **absent** from
      ``daily.<var>`` raises ``KeyError`` — fail-loud on an upstream rename (the B2 contract the
      live probe pins). Identical to the pre-refactor ``normalize_archive_response`` per-chunk body.
    - ``strict=False`` (forecast): an absent variable is returned as an **all-``None`` column the
      same length as ``dates`` — never raised — so the §4.5 usability check converts it into a clean
      provider outage (→ router fallback) instead of crashing the fetch.
    """
    daily = raw.get("daily") or {}
    dates: list[str] = list(daily.get("time") or [])
    columns: dict[str, list[float | None]] = {}
    for v in normalized_variables:
        pv = variable_resolver(v)
        series = daily.get(pv.provider)
        if series is None:
            if strict:
                raise KeyError(
                    f"Open-Meteo response missing daily.{pv.provider!r} (variable {v!r})"
                )
            # forecast path: absent → all-None (marks a §4.5 outage), never raise
            columns[v] = [None] * len(dates)
            continue
        columns[v] = [convert_to_stored_unit(v, x) for x in series]
    return dates, columns


def normalize_forecast_daily(
    raw: dict, normalized_variables: list[str]
) -> tuple[list[str], dict[str, list[float | None]]]:
    """Normalize a live forecast, requiring explicit Celsius units for optional Tmin.

    Missing series remain null columns for the router's existing outage check. A populated
    optional series without the expected unit is unusable, never silently labeled Celsius.
    """
    daily = raw.get("daily") or {}
    units = raw.get("daily_units") or {}
    for variable in normalized_variables:
        if variable in _FORECAST_ONLY_BY_NORMALIZED:
            pv = forecast_provider_var(variable)
            if daily.get(pv.provider) is not None and units.get(pv.provider) != pv.source_unit:
                raise ValueError(
                    f"Open-Meteo daily.{pv.provider} unit must be {pv.source_unit!r}; "
                    f"got {units.get(pv.provider)!r}"
                )
    return normalize_openmeteo_daily(
        raw, normalized_variables, strict=False, variable_resolver=forecast_provider_var
    )


def split_date_range(
    start_date: str, end_date: str, *, max_days: int | None = None
) -> list[tuple[str, str]]:
    """Split ``[start_date, end_date]`` (inclusive, ISO-8601) into ≤ ``max_days`` sub-ranges.

    Pure + deterministic — the chunking guard-rail (§3.5), unit-tested without network.
    """
    s = date.fromisoformat(start_date)
    e = date.fromisoformat(end_date)
    if e < s:
        raise ValueError(f"end_date {end_date} precedes start_date {start_date}")
    limit = max_days or _MAX_DAYS_PER_REQUEST
    if limit < 1:
        raise ValueError(f"max_days must be >= 1, got {limit}")
    chunks: list[tuple[str, str]] = []
    cur = s
    step = timedelta(days=limit - 1)
    while cur <= e:
        chunk_end = min(cur + step, e)
        chunks.append((cur.isoformat(), chunk_end.isoformat()))
        cur = chunk_end + timedelta(days=1)
    return chunks


_client: httpx.Client | None = None


def get_archive_client() -> httpx.Client:
    """Return a reusable httpx Client for the Open-Meteo Archive API (lazy singleton)."""
    global _client
    if _client is None:
        _client = httpx.Client(timeout=_TIMEOUT_S)
    return _client


def fetch_archive_daily(
    *,
    latitude: float,
    longitude: float,
    start_date: str,
    end_date: str,
    normalized_variables: list[str],
) -> dict:
    """Fetch ERA5 **daily** values for one point over ``[start_date, end_date]`` (inclusive).

    Blocking — call via ``asyncio.to_thread`` from async code. Forces ``models=era5`` and
    ``wind_speed_unit=ms``; requests the provider names mapped from ``normalized_variables``.
    Retries transient transport/5xx errors with exponential backoff. Returns the parsed JSON
    (raw provider keys + ``daily``/``daily_units`` blocks — normalization happens downstream).
    """
    provider_names = [provider_var(n).provider for n in normalized_variables]
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "daily": ",".join(provider_names),
        "models": "era5",  # forced — never best_match (§3.1)
        "wind_speed_unit": "ms",  # forced — API default is km/h (§3.1)
        "timezone": "UTC",
    }
    client = get_archive_client()
    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            resp = client.get(ARCHIVE_URL, params=params)
            resp.raise_for_status()
            return resp.json()
        except (httpx.TransportError, httpx.HTTPStatusError) as exc:
            # Don't retry a 4xx (a bad request won't fix itself); retry transport + 5xx. The final
            # attempt re-raises directly (no sentinel bookkeeping / assert).
            status = getattr(getattr(exc, "response", None), "status_code", None)
            client_error = status is not None and 400 <= status < 500
            if client_error or attempt >= _MAX_RETRIES:
                raise
            delay = _BACKOFF_BASE_S * (2 ** (attempt - 1))
            log.warning("openmeteo_retry", attempt=attempt, delay_s=delay, error=str(exc))
            time.sleep(delay)


# --------------------------------------------------------------------------- #
# Forecast client (FIERCE B5) — short-horizon point forecast for the provider router
# --------------------------------------------------------------------------- #

_forecast_client: httpx.Client | None = None


def get_forecast_client() -> httpx.Client:
    """Return a reusable httpx Client for the Open-Meteo Forecast API (lazy singleton).

    Created with the short forecast timeout; ``fetch_forecast_daily`` still passes a per-request
    ``timeout`` so a provider can tune it without rebuilding the client. The network-free tests
    monkeypatch this to inject a fake client."""
    global _forecast_client
    if _forecast_client is None:
        _forecast_client = httpx.Client(timeout=_FORECAST_TIMEOUT_S)
    return _forecast_client


def fetch_forecast_daily(
    *,
    latitude: float,
    longitude: float,
    normalized_variables: list[str],
    forecast_days: int,
    models: str = "",
    timeout_s: float = _FORECAST_TIMEOUT_S,
    apikey: str = "",
    base_url: str = "",
) -> dict:
    """Fetch a short-term **daily** forecast for one point (FIERCE B5).

    Mirrors :func:`fetch_archive_daily`: maps ``normalized_variables`` → provider names via
    :func:`forecast_provider_var`, forces Celsius, ``wind_speed_unit=ms`` + ``timezone=UTC``
    (consistency with the
    stored ERA5 daily grid), and retries transport/5xx with the same backoff. Differences:

    - ``models`` is added **only when non-empty** — an empty string means *omit the param*
      (Open-Meteo best-match, the reliable physical-NWP provider). ``models=auto`` is **invalid**
      (the API 400s); callers pass either a verified token or ``""``.
    - ``apikey`` is added **only when set** (the commercial ``customer-api.*`` tier).
    - ``base_url`` overrides :data:`FORECAST_URL` (commercial endpoint).

    Returns the parsed JSON (raw provider keys + ``daily``/``daily_units``) — normalization,
    trailing-null trimming, and the §4.5 usability decision happen in the provider layer
    (:mod:`src.utils.forecast`), keeping this function a thin raw-I/O mirror of the Archive client.
    """
    provider_names = [forecast_provider_var(n).provider for n in normalized_variables]
    params: dict = {
        "latitude": latitude,
        "longitude": longitude,
        "daily": ",".join(provider_names),
        "forecast_days": forecast_days,
        "temperature_unit": "celsius",
        "wind_speed_unit": "ms",  # forced — API default is km/h
        "timezone": "UTC",  # forced — consistency with the stored ERA5 daily grid
    }
    if models:  # empty => OMIT (best-match); never send "auto" (API 400)
        params["models"] = models
    if apikey:  # commercial tier only
        params["apikey"] = apikey
    url = base_url or FORECAST_URL
    client = get_forecast_client()
    for attempt in range(1, _FORECAST_MAX_RETRIES + 1):
        try:
            resp = client.get(url, params=params, timeout=timeout_s)
            resp.raise_for_status()
            return resp.json()
        except (httpx.TransportError, httpx.HTTPStatusError) as exc:
            # 4xx won't fix itself (don't retry); retry transport + 5xx. Final attempt re-raises.
            status = getattr(getattr(exc, "response", None), "status_code", None)
            client_error = status is not None and 400 <= status < 500
            if client_error or attempt >= _FORECAST_MAX_RETRIES:
                raise
            delay = _FORECAST_BACKOFF_BASE_S * (2 ** (attempt - 1))
            log.warning("forecast_retry", attempt=attempt, delay_s=delay, error=str(exc))
            time.sleep(delay)
