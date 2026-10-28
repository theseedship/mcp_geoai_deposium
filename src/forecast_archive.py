"""Provider-specific forecast archive acquisition for FIERCE B5a.

This module targets Open-Meteo Previous Runs rather than the live forecast endpoint. It returns
daily rows with a fixed lead time and explicit coverage states; missing historical model data is
never represented as zeros.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from email.utils import parsedate_to_datetime
from typing import Any, NamedTuple

import httpx

from src.config import settings, validate_forecast_archive_auth_endpoint
from src.utils.openmeteo import NORMALIZED_VARS, provider_var

_AGGREGATION = {
    "temperature_2m": "mean",
    "precipitation": "sum",
    "humidity": "mean",
    "evapotranspiration": "sum",
    "wind_speed": "max",
    "solar_radiation": "mean",
}

# The hourly unit Open-Meteo reports in ``hourly_units`` for each normalized variable's Previous
# Runs field, with ``wind_speed_unit=ms`` forced by the request. Taken from the provider's
# documentation, NOT from a live response: to confirm at the coverage preflight (#193). A
# mismatch refuses the whole response, so a wrong entry here fails loudly, never silently.
PREVIOUS_RUNS_HOURLY_UNITS = {
    "temperature_2m": "°C",
    "precipitation": "mm",
    "humidity": "%",
    "evapotranspiration": "mm",
    "wind_speed": "m/s",
    "solar_radiation": "W/m²",
}


@dataclass(frozen=True)
class ArchiveRow:
    zone_id: str
    origin_date: date
    valid_date: date
    horizon_days: int
    variable: str
    value: float
    provider: str
    model: str


@dataclass(frozen=True)
class Coverage:
    zone_id: str
    provider: str
    model: str
    horizon_days: int
    start_date: str
    end_date: str
    expected_days: int
    available_days: int
    status: str
    # Pairs, not days, are the unit of coverage (#193): one incomplete variable must not hide
    # the complete ones of the same day. The defaults keep hand-built coverages valid.
    expected_pairs: int = 0
    available_pairs: int = 0
    missing: tuple[dict[str, Any], ...] = ()


@dataclass(frozen=True)
class RawResponse:
    """The provider's exact response bytes, digested BEFORE anything parses them (#193).

    ``params`` is the request as sent minus the API key: enough to re-issue it, never enough
    to leak the credential into an archive that is published or handed over.
    """

    body: bytes
    sha256: str
    retrieved_at: str
    params: dict[str, Any]


class ArchiveResult(NamedTuple):
    rows: list[ArchiveRow]
    coverage: Coverage
    raw: RawResponse


class PreviousRunsRequestError(RuntimeError):
    """Provider failure that never retains a request URL or credentials."""

    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


def _parse_hour(value: Any) -> datetime:
    if not isinstance(value, str):
        raise ValueError("hourly.time contains a non-string timestamp")
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _utc_hour(value: Any) -> datetime:
    """One stamp as naive UTC on the hourly grid; anything else is a malformed response."""
    parsed = _parse_hour(value)
    if parsed.tzinfo is not None:
        if parsed.utcoffset() != timedelta(0):
            raise ValueError("hourly.time contains a non-UTC timestamp")
        parsed = parsed.replace(tzinfo=None)
    if parsed.minute or parsed.second or parsed.microsecond:
        raise ValueError("hourly.time contains a timestamp off the hourly grid")
    return parsed


def _aggregate(values: list[float], operation: str) -> float:
    if operation == "sum":
        return sum(values)
    if operation == "max":
        return max(values)
    return sum(values) / len(values)


def _previous_field(variable: str, horizon_days: int) -> str:
    provider_name = provider_var(variable).provider
    for suffix in ("_mean", "_sum", "_max"):
        if provider_name.endswith(suffix):
            provider_name = provider_name[: -len(suffix)]
            break
    return f"{provider_name}_previous_day{horizon_days}"


def normalize_previous_runs(
    payload: dict[str, Any],
    *,
    zone_id: str,
    provider: str,
    model: str,
    horizon_days: int,
    variables: list[str],
    expected_start_date: str | None = None,
    expected_end_date: str | None = None,
) -> tuple[list[ArchiveRow], Coverage]:
    """Convert one Previous Runs hourly payload into daily rows, one variable at a time.

    The response FRAME is refused, not trusted (#193): a non-UTC offset, a field unit other
    than :data:`PREVIOUS_RUNS_HOURLY_UNITS`, a duplicate or off-grid stamp, or a stamp outside
    the requested dates. A duplicate is a refusal because 24 entries with a duplicate are not a
    complete day. Inside that frame a gap belongs to one ``(date, variable)``: an incomplete
    variable emits no row and is listed in ``Coverage.missing`` while the other variables of
    the same day are kept. Nothing is filled with zero, and the expected denominator is the
    requested range times the variables, never what happened to come back.
    """
    if payload.get("error"):
        raise ValueError(str(payload.get("reason") or "Previous Runs API error"))
    if payload.get("utc_offset_seconds") != 0 or payload.get("timezone") not in {"GMT", "UTC"}:
        raise ValueError("Previous Runs response must use UTC")
    hourly = payload.get("hourly")
    if not isinstance(hourly, dict) or not isinstance(hourly.get("time"), list):
        raise ValueError("Previous Runs response has no hourly time series")
    units = payload.get("hourly_units")
    if not isinstance(units, dict) or units.get("time") != "iso8601":
        raise ValueError("Previous Runs response must carry iso8601 hourly units")
    times = [_utc_hour(value) for value in hourly["time"]]
    if not times:
        raise ValueError("Previous Runs response contains no timestamps")
    if len(set(times)) != len(times):
        raise ValueError("Previous Runs response contains a duplicate timestamp")
    requested_start = (
        date.fromisoformat(expected_start_date) if expected_start_date else min(times).date()
    )
    requested_end = (
        date.fromisoformat(expected_end_date) if expected_end_date else max(times).date()
    )
    if requested_start > requested_end:
        raise ValueError("Previous Runs start date is after its end date")
    if any(not requested_start <= timestamp.date() <= requested_end for timestamp in times):
        raise ValueError("Previous Runs response contains timestamps outside the requested dates")
    columns: dict[str, list[float | None]] = {}
    for variable in variables:
        provider_name = _previous_field(variable, horizon_days)
        raw = hourly.get(provider_name)
        if not isinstance(raw, list) or len(raw) != len(times):
            raise ValueError(f"missing or ragged Previous Runs variable: {provider_name}")
        expected_unit = PREVIOUS_RUNS_HOURLY_UNITS[variable]
        if units.get(provider_name) != expected_unit:
            raise ValueError(
                f"Previous Runs {provider_name} must use {expected_unit!r}, "
                f"got {units.get(provider_name)!r}"
            )
        # Only a JSON number or null is an hourly value. Anything else is a malformed response,
        # refused as a ValueError so the caller keeps its bytes (``ResponseRefused``) instead of
        # meeting a TypeError from float().
        if any(
            value is not None and (isinstance(value, bool) or not isinstance(value, (int, float)))
            for value in raw
        ):
            raise ValueError(f"Previous Runs {provider_name} contains a non-numeric value")
        columns[variable] = [None if value is None else float(value) for value in raw]

    # Stamps are unique and on the hour, so 24 finite values are exactly the 24 UTC hours.
    grouped: dict[date, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for index, timestamp in enumerate(times):
        for variable, values in columns.items():
            value = values[index]
            if value is not None and math.isfinite(value):
                grouped[timestamp.date()][variable].append(value)

    expected_days = (requested_end - requested_start).days + 1
    rows: list[ArchiveRow] = []
    missing: list[dict[str, Any]] = []
    available_days = 0
    for offset in range(expected_days):
        valid_date = requested_start + timedelta(days=offset)
        daily = grouped.get(valid_date, {})
        complete = True
        for variable in variables:
            values = daily.get(variable, [])
            if len(values) != 24:
                complete = False
                missing.append(
                    {
                        "valid_date": valid_date.isoformat(),
                        "variable": variable,
                        "horizon_days": horizon_days,
                    }
                )
                continue
            rows.append(
                ArchiveRow(
                    zone_id,
                    valid_date - timedelta(days=horizon_days),
                    valid_date,
                    horizon_days,
                    variable,
                    _aggregate(values, _AGGREGATION[variable]),
                    provider,
                    model,
                )
            )
        available_days += complete
    expected_pairs = expected_days * len(variables)
    coverage = Coverage(
        zone_id,
        provider,
        model,
        horizon_days,
        requested_start.isoformat(),
        requested_end.isoformat(),
        expected_days,
        available_days,
        "available" if len(rows) == expected_pairs else "partial_archive",
        expected_pairs,
        len(rows),
        tuple(missing),
    )
    return rows, coverage


def _decode(raw: RawResponse) -> dict[str, Any]:
    payload = json.loads(raw.body)
    if not isinstance(payload, dict):
        raise ValueError("Previous Runs response must be a JSON object")
    return payload


class ResponseRefused(ValueError):
    """A response that ARRIVED but was refused; it keeps its raw bytes for inspection.

    A refusal at the preflight (a unit string guessed wrong, say) is exactly when the bytes
    matter most, so the caller can retain them instead of losing them with the result.
    """

    def __init__(self, message: str, *, raw: RawResponse) -> None:
        super().__init__(message)
        self.raw = raw


def _normalized(
    raw: RawResponse,
    horizon: int,
    normalize: dict[str, Any],
    payload: dict[str, Any] | None = None,
) -> ArchiveResult:
    try:
        rows, coverage = normalize_previous_runs(
            _decode(raw) if payload is None else payload, horizon_days=horizon, **normalize
        )
    except ValueError as exc:
        raise ResponseRefused(str(exc), raw=raw) from exc
    return ArchiveResult(rows, coverage, raw)


class PreviousRunsArchiveClient:
    """Small HTTP client with injectable transport and clock for deterministic tests.

    The transport receives the request parameters and returns the response BYTES; the clock
    stamps ``RawResponse.retrieved_at``. Both exist so a test can replay one fixture twice and
    get byte-identical archives without any network call.
    """

    def __init__(
        self,
        *,
        base_url: str | None = None,
        apikey: str | None = None,
        timeout_s: float = 60.0,
        transport: Callable[[dict[str, Any]], bytes] | None = None,
        now: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        self.base_url = (base_url or settings.FORECAST_ARCHIVE_API_BASE_URL).strip()
        configured_key = settings.FORECAST_ARCHIVE_API_KEY if apikey is None else apikey
        self.apikey = configured_key.strip()
        validate_forecast_archive_auth_endpoint(self.base_url, self.apikey)
        self.timeout_s = timeout_s
        self._transport = transport
        self._now = now

    def fetch_daily(
        self,
        *,
        zone_id: str,
        latitude: float,
        longitude: float,
        start_date: str,
        end_date: str,
        variables: list[str] | None = None,
        provider: str = "openmeteo_previous_runs",
        model: str = "gfs_seamless",
        horizon_days: int = 1,
    ) -> tuple[list[ArchiveRow], Coverage]:
        requested = list(variables or NORMALIZED_VARS)
        unknown = [variable for variable in requested if variable not in NORMALIZED_VARS]
        if unknown or horizon_days < 1 or horizon_days > 7:
            raise ValueError("invalid variables or Previous Runs horizon; expected 1..7 days")
        raw = self._fetch_payload(
            latitude=latitude,
            longitude=longitude,
            start_date=start_date,
            end_date=end_date,
            model=model,
            hourly_fields=[_previous_field(variable, horizon_days) for variable in requested],
        )
        return normalize_previous_runs(
            _decode(raw),
            zone_id=zone_id,
            provider=provider,
            model=model,
            horizon_days=horizon_days,
            variables=requested,
            expected_start_date=start_date,
            expected_end_date=end_date,
        )

    def fetch_daily_multi(
        self,
        *,
        zone_id: str,
        latitude: float,
        longitude: float,
        start_date: str,
        end_date: str,
        horizons: list[int],
        variables: list[str] | None = None,
        provider: str = "openmeteo_previous_runs",
        model: str = "gfs_seamless",
    ) -> dict[int, ArchiveResult | Exception]:
        """One request for every horizon; each result keeps the raw response it came from."""
        requested = list(variables or NORMALIZED_VARS)
        unknown = [variable for variable in requested if variable not in NORMALIZED_VARS]
        if unknown or not horizons or any(horizon < 1 or horizon > 7 for horizon in horizons):
            raise ValueError("invalid variables or Previous Runs horizon; expected 1..7 days")
        unique_horizons = list(dict.fromkeys(horizons))
        request = {
            "latitude": latitude,
            "longitude": longitude,
            "start_date": start_date,
            "end_date": end_date,
            "model": model,
        }
        normalize = {
            "zone_id": zone_id,
            "provider": provider,
            "model": model,
            "variables": requested,
            "expected_start_date": start_date,
            "expected_end_date": end_date,
        }
        try:
            raw = self._fetch_payload(
                **request,
                hourly_fields=[
                    _previous_field(variable, horizon)
                    for horizon in unique_horizons
                    for variable in requested
                ],
            )
            payload = _decode(raw)
        except PreviousRunsRequestError as batch_error:
            if batch_error.status_code == 429:
                raise
            return self._per_horizon(unique_horizons, request, normalize, batch_error)
        except Exception as batch_error:
            return self._per_horizon(unique_horizons, request, normalize, batch_error)
        results: dict[int, ArchiveResult | Exception] = {}
        for horizon in unique_horizons:
            # Only a refused RESPONSE is a per-horizon result; any other exception here is a
            # bug and must surface as itself rather than pass for a provider failure.
            try:
                results[horizon] = _normalized(raw, horizon, normalize, payload)
            except ResponseRefused as exc:
                results[horizon] = exc
        return results

    def _per_horizon(
        self,
        horizons: list[int],
        request: dict[str, Any],
        normalize: dict[str, Any],
        batch_error: Exception,
    ) -> dict[int, ArchiveResult | Exception]:
        # Some providers reject a mixed field list. Preserve independent horizons
        # by retrying each field set separately instead of poisoning the batch.
        results: dict[int, ArchiveResult | Exception] = {}
        for horizon in horizons:
            # A provider or transport failure has no fixed type, so the FETCH is caught broadly;
            # normalization only yields a refused response, and anything else is a bug.
            try:
                raw = self._fetch_payload(
                    **request,
                    hourly_fields=[
                        _previous_field(variable, horizon) for variable in normalize["variables"]
                    ],
                )
            except Exception as exc:
                results[horizon] = exc
                continue
            try:
                results[horizon] = _normalized(raw, horizon, normalize)
            except ResponseRefused as exc:
                results[horizon] = exc
        if any(not isinstance(value, Exception) for value in results.values()):
            return results
        raise batch_error

    def _fetch_payload(
        self,
        *,
        latitude: float,
        longitude: float,
        start_date: str,
        end_date: str,
        model: str,
        hourly_fields: list[str],
    ) -> RawResponse:
        params: dict[str, Any] = {
            "latitude": latitude,
            "longitude": longitude,
            "start_date": start_date,
            "end_date": end_date,
            "models": model,
            "timezone": "UTC",
            "wind_speed_unit": "ms",
            "hourly": ",".join(hourly_fields),
        }
        recorded = dict(params)
        if self.apikey:
            params["apikey"] = self.apikey
        body = self._transport(params) if self._transport else self._get(params)
        if not isinstance(body, bytes):
            raise TypeError("Previous Runs transport must return the response bytes")
        return RawResponse(
            body=body,
            sha256=hashlib.sha256(body).hexdigest(),
            retrieved_at=self._now().astimezone(UTC).isoformat(),
            params=recorded,
        )

    def _get(self, params: dict[str, Any]) -> bytes:
        for attempt in range(3):
            safe_error: PreviousRunsRequestError | None = None
            try:
                response = httpx.get(self.base_url, params=params, timeout=self.timeout_s)
                response.raise_for_status()
                return response.content
            except httpx.HTTPStatusError as exc:
                status_code = exc.response.status_code
                if status_code != 429 or attempt == 2:
                    safe_error = PreviousRunsRequestError(
                        f"Previous Runs API returned HTTP {status_code}",
                        status_code=status_code,
                    )
                else:
                    retry_after = exc.response.headers.get("Retry-After")
                    try:
                        delay = float(retry_after) if retry_after else 2**attempt
                    except ValueError:
                        try:
                            retry_at = parsedate_to_datetime(retry_after)
                            if retry_at.tzinfo is None:
                                retry_at = retry_at.replace(tzinfo=UTC)
                            delay = (retry_at - datetime.now(UTC)).total_seconds()
                        except (TypeError, ValueError, OverflowError):
                            delay = 2**attempt
                    time.sleep(min(max(delay, 1.0), 10.0))
            except httpx.RequestError as exc:
                safe_error = PreviousRunsRequestError(
                    f"Previous Runs API request failed ({type(exc).__name__})"
                )
            if safe_error is not None:
                raise safe_error
        raise RuntimeError("unreachable retry loop")
