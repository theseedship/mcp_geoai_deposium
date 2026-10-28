"""Strict AIFS ensemble precipitation acquisition and normalization.

The Open-Meteo Ensemble API exposes individual AIFS members for only a short
retention window. This module preserves every member used to derive a daily
empirical event probability. It does not claim that the probability is
calibrated against observations.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Sequence
from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime, timedelta
from email.utils import parsedate_to_datetime
from typing import Any
from urllib.parse import urlsplit

import httpx

from src.config import settings, validate_forecast_ensemble_auth_endpoint
from src.forecast_backtest import (
    DEFAULT_PRECIPITATION_EVENT_THRESHOLD_MM,
    PRECIPITATION_EVENT_OPERATOR,
)

MODEL = "ecmwf_aifs025_ensemble"
PROVIDER = "openmeteo_aifs_ensemble"
SCHEMA_VERSION = "forecast-ensemble-archive/v1"
CAPTURE_SCHEMA_VERSION = "forecast-ensemble-capture/v1"
DEFAULT_HORIZONS = (1, 3, 5, 7)
EXPECTED_MEMBER_COUNT = 51
MAX_PROVIDER_COORDINATE_DELTA_DEGREES = 0.25
CAPTURE_WINDOW_START_HOUR_UTC = 19
CAPTURE_WINDOW_END_HOUR_UTC = 21
MEMBER_FIELDS = (
    "precipitation_sum",
    *(f"precipitation_sum_member{index:02d}" for index in range(1, EXPECTED_MEMBER_COUNT)),
)


class EnsembleRequestError(RuntimeError):
    """Provider failure that never retains a request URL or credentials."""

    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class EnsembleDataQualityError(ValueError):
    """Decoded provider content does not meet the scientific snapshot contract."""


@dataclass(frozen=True)
class EnsembleMember:
    member_id: int
    source_field: str
    daily_precipitation_mm: float


@dataclass(frozen=True)
class EnsembleProbabilityRow:
    zone_id: str
    origin_date: str
    valid_date: str
    horizon_days: int
    variable: str
    value: float
    provider: str
    model: str
    probability: float
    probability_event: dict[str, float | str]
    member_count_expected: int
    member_count_used: int
    event_member_count: int
    members: tuple[EnsembleMember, ...]
    daily_aggregation: dict[str, str | int]
    probability_method: str
    calibration_status: str
    origin_semantics: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EnsembleSnapshot:
    zone_id: str
    snapshot_date_utc: str
    request_started_at_utc: str
    retrieved_at_utc: str
    requested_coordinates: dict[str, float]
    provider_coordinates: dict[str, float | None]
    endpoint: str
    request_parameters: dict[str, Any]
    rows: tuple[EnsembleProbabilityRow, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "zone_id": self.zone_id,
            "snapshot_date_utc": self.snapshot_date_utc,
            "request_started_at_utc": self.request_started_at_utc,
            "retrieved_at_utc": self.retrieved_at_utc,
            "provider": PROVIDER,
            "model": MODEL,
            "requested_coordinates": dict(self.requested_coordinates),
            "provider_coordinates": dict(self.provider_coordinates),
            "endpoint": self.endpoint,
            "request_parameters": dict(self.request_parameters),
            "member_contract": {
                "expected": EXPECTED_MEMBER_COUNT,
                "weighting": "equal",
                "source_fields": list(MEMBER_FIELDS),
            },
            "rows": [row.to_dict() for row in self.rows],
        }


def _finite_number(value: Any, field: str, *, nonnegative: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a finite number")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{field} must be a finite number") from exc
    if not math.isfinite(number) or (nonnegative and number < 0):
        suffix = " >= 0" if nonnegative else ""
        raise ValueError(f"{field} must be a finite number{suffix}")
    return number


def _iso_date(value: Any) -> date:
    if not isinstance(value, str):
        raise ValueError("daily.time must contain ISO dates")
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError("daily.time must contain ISO dates") from exc


def _validate_horizons(horizons: Sequence[int]) -> tuple[int, ...]:
    if not horizons or any(
        isinstance(horizon, bool) or not isinstance(horizon, int) or not 1 <= horizon <= 15
        for horizon in horizons
    ):
        raise ValueError("horizons must be unique integers in 1..15")
    unique = tuple(dict.fromkeys(horizons))
    if len(unique) != len(horizons):
        raise ValueError("horizons must be unique integers in 1..15")
    return unique


def _validate_capture_window(timestamp: datetime) -> None:
    if not CAPTURE_WINDOW_START_HOUR_UTC <= timestamp.hour < CAPTURE_WINDOW_END_HOUR_UTC:
        raise ValueError(
            "ensemble acquisition must stay within the fixed UTC capture window "
            f"[{CAPTURE_WINDOW_START_HOUR_UTC:02d}:00, "
            f"{CAPTURE_WINDOW_END_HOUR_UTC:02d}:00)"
        )


def normalize_aifs_ensemble_precipitation(
    payload: dict[str, Any],
    *,
    zone_id: str,
    snapshot_date: date,
    horizons: Sequence[int] = DEFAULT_HORIZONS,
    event_threshold_mm: float = DEFAULT_PRECIPITATION_EVENT_THRESHOLD_MM,
) -> tuple[EnsembleProbabilityRow, ...]:
    """Derive daily probabilities from all 51 provider-returned AIFS members."""
    if not isinstance(payload, dict):
        raise ValueError("Ensemble response must be an object")
    if not isinstance(zone_id, str) or not zone_id:
        raise ValueError("zone_id must be a non-empty string")
    if not isinstance(snapshot_date, date) or isinstance(snapshot_date, datetime):
        raise ValueError("snapshot_date must be a date")
    if payload.get("error"):
        raise ValueError(str(payload.get("reason") or "Ensemble API error"))
    normalized_horizons = _validate_horizons(horizons)
    threshold = _finite_number(event_threshold_mm, "event_threshold_mm", nonnegative=True)
    if payload.get("utc_offset_seconds") != 0 or payload.get("timezone") not in {"GMT", "UTC"}:
        raise ValueError("Ensemble response must use UTC")

    daily = payload.get("daily")
    units = payload.get("daily_units")
    if not isinstance(daily, dict) or not isinstance(units, dict):
        raise ValueError("Ensemble response must contain daily data and units")
    actual_fields = set(daily) - {"time"}
    if actual_fields != set(MEMBER_FIELDS):
        raise ValueError(f"Ensemble response must contain exactly {EXPECTED_MEMBER_COUNT} members")
    if set(units) - {"time"} != set(MEMBER_FIELDS):
        raise ValueError("Ensemble units must describe exactly the member fields")
    if units.get("time") != "iso8601" or any(units.get(field) != "mm" for field in MEMBER_FIELDS):
        raise ValueError("Ensemble precipitation members must use daily millimetres")

    raw_dates = daily.get("time")
    if not isinstance(raw_dates, list) or not raw_dates:
        raise ValueError("Ensemble response has no daily dates")
    dates = [_iso_date(value) for value in raw_dates]
    if len(set(dates)) != len(dates):
        raise ValueError("Ensemble response contains duplicate daily dates")
    values_by_field: dict[str, list[Any]] = {}
    for field in MEMBER_FIELDS:
        values = daily.get(field)
        if not isinstance(values, list) or len(values) != len(dates):
            raise ValueError(f"missing or ragged ensemble member: {field}")
        values_by_field[field] = values

    date_indexes = {value: index for index, value in enumerate(dates)}
    rows: list[EnsembleProbabilityRow] = []
    for horizon in normalized_horizons:
        valid_date = snapshot_date + timedelta(days=horizon)
        if valid_date not in date_indexes:
            raise ValueError(f"Ensemble response is missing horizon {horizon}")
        index = date_indexes[valid_date]
        members = tuple(
            EnsembleMember(
                member_id=member_id,
                source_field=field,
                daily_precipitation_mm=_finite_number(
                    values_by_field[field][index],
                    f"{field}[{index}]",
                    nonnegative=True,
                ),
            )
            for member_id, field in enumerate(MEMBER_FIELDS)
        )
        event_count = sum(member.daily_precipitation_mm > threshold for member in members)
        rows.append(
            EnsembleProbabilityRow(
                zone_id=zone_id,
                origin_date=snapshot_date.isoformat(),
                valid_date=valid_date.isoformat(),
                horizon_days=horizon,
                variable="precipitation",
                value=sum(member.daily_precipitation_mm for member in members)
                / EXPECTED_MEMBER_COUNT,
                provider=PROVIDER,
                model=MODEL,
                probability=event_count / EXPECTED_MEMBER_COUNT,
                probability_event={
                    "operator": PRECIPITATION_EVENT_OPERATOR,
                    "threshold_mm": threshold,
                },
                member_count_expected=EXPECTED_MEMBER_COUNT,
                member_count_used=len(members),
                event_member_count=event_count,
                members=members,
                daily_aggregation={
                    "source": "provider_daily",
                    "operation": "sum",
                    "window_hours": 24,
                    "timezone": "UTC",
                    "unit": "mm",
                },
                probability_method="raw_ensemble_relative_frequency",
                calibration_status="uncalibrated",
                origin_semantics="snapshot_date_utc_not_model_initialization",
            )
        )
    return tuple(rows)


class OpenMeteoEnsembleClient:
    """Small client whose transport and clock are injectable for deterministic tests."""

    def __init__(
        self,
        *,
        base_url: str | None = None,
        apikey: str | None = None,
        timeout_s: float = 30.0,
        transport: Callable[[dict[str, Any]], Any] | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self.base_url = (base_url or settings.FORECAST_ENSEMBLE_API_BASE_URL).strip()
        parsed = urlsplit(self.base_url)
        if (
            parsed.scheme.lower() != "https"
            or not parsed.hostname
            or parsed.port not in (None, 443)
            or parsed.path.rstrip("/") != "/v1/ensemble"
            or parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError("ensemble endpoint must be a credential-free HTTPS /v1/ensemble URL")
        configured_key = settings.FORECAST_ENSEMBLE_API_KEY if apikey is None else apikey
        self.apikey = configured_key.strip()
        validate_forecast_ensemble_auth_endpoint(self.base_url, self.apikey)
        self.timeout_s = timeout_s
        self._transport = transport
        self._clock = clock or (lambda: datetime.now(UTC))

    def fetch_snapshot(
        self,
        *,
        zone_id: str,
        latitude: float,
        longitude: float,
        snapshot_date: date | None = None,
        horizons: Sequence[int] = DEFAULT_HORIZONS,
        event_threshold_mm: float = DEFAULT_PRECIPITATION_EVENT_THRESHOLD_MM,
        on_capture: Callable[[dict[str, Any]], None] | None = None,
    ) -> EnsembleSnapshot:
        """Capture decoded raw JSON before validating provider response content.

        The optional callback receives independent copies of the decoded payload
        and public acquisition metadata, never exact HTTP response bytes.
        Callback failures propagate without data-quality classification.
        """
        normalized_horizons = _validate_horizons(horizons)
        if not isinstance(zone_id, str) or not zone_id:
            raise ValueError("zone_id must be a non-empty string")
        threshold = _finite_number(event_threshold_mm, "event_threshold_mm", nonnegative=True)
        request_started_at = self._clock()
        if (
            not isinstance(request_started_at, datetime)
            or request_started_at.tzinfo is None
            or request_started_at.utcoffset() is None
        ):
            raise ValueError("ensemble client clock must be timezone-aware")
        request_started_at_utc = request_started_at.astimezone(UTC)
        _validate_capture_window(request_started_at_utc)
        requested_snapshot = snapshot_date or request_started_at_utc.date()
        if requested_snapshot != request_started_at_utc.date():
            raise ValueError("snapshot_date must equal the current UTC date")
        requested_latitude = _finite_number(latitude, "latitude")
        requested_longitude = _finite_number(longitude, "longitude")
        if not -90 <= requested_latitude <= 90 or not -180 <= requested_longitude <= 180:
            raise ValueError("invalid WGS84 coordinates")
        start = requested_snapshot + timedelta(days=min(normalized_horizons))
        end = requested_snapshot + timedelta(days=max(normalized_horizons))
        public_params: dict[str, Any] = {
            "latitude": requested_latitude,
            "longitude": requested_longitude,
            "start_date": start.isoformat(),
            "end_date": end.isoformat(),
            "models": MODEL,
            "daily": "precipitation_sum",
            "timezone": "UTC",
            "precipitation_unit": "mm",
        }
        request_params = dict(public_params)
        if self.apikey:
            request_params["apikey"] = self.apikey
        payload = self._fetch_payload(request_params)
        retrieved_at = self._clock()
        if (
            not isinstance(retrieved_at, datetime)
            or retrieved_at.tzinfo is None
            or retrieved_at.utcoffset() is None
        ):
            raise ValueError("ensemble client clock must be timezone-aware")
        retrieved_at_utc = retrieved_at.astimezone(UTC)
        if retrieved_at_utc < request_started_at_utc:
            raise ValueError("ensemble client clock moved backwards during request")
        if retrieved_at_utc.date() != request_started_at_utc.date():
            raise ValueError("ensemble request crossed a UTC date boundary")
        _validate_capture_window(retrieved_at_utc)
        if on_capture is not None:
            on_capture(
                {
                    "schema_version": CAPTURE_SCHEMA_VERSION,
                    "zone_id": zone_id,
                    "snapshot_date_utc": requested_snapshot.isoformat(),
                    "request_started_at_utc": request_started_at_utc.isoformat().replace(
                        "+00:00", "Z"
                    ),
                    "retrieved_at_utc": retrieved_at_utc.isoformat().replace("+00:00", "Z"),
                    "provider": PROVIDER,
                    "model": MODEL,
                    "requested_coordinates": {
                        "latitude": requested_latitude,
                        "longitude": requested_longitude,
                    },
                    "endpoint": self.base_url,
                    "request_parameters": dict(public_params),
                    "payload": deepcopy(payload),
                }
            )
        try:
            if not isinstance(payload, dict):
                raise ValueError("Ensemble API returned a non-object payload")
            rows = normalize_aifs_ensemble_precipitation(
                payload,
                zone_id=zone_id,
                snapshot_date=requested_snapshot,
                horizons=normalized_horizons,
                event_threshold_mm=threshold,
            )
            provider_latitude = _finite_number(payload.get("latitude"), "provider latitude")
            provider_longitude = _finite_number(payload.get("longitude"), "provider longitude")
            provider_elevation = (
                None
                if payload.get("elevation") is None
                else _finite_number(payload.get("elevation"), "provider elevation")
            )
            if not -90 <= provider_latitude <= 90 or not -180 <= provider_longitude <= 180:
                raise ValueError("provider returned invalid WGS84 coordinates")
            latitude_delta = abs(provider_latitude - requested_latitude)
            longitude_delta = abs((provider_longitude - requested_longitude + 180) % 360 - 180)
            if (
                latitude_delta > MAX_PROVIDER_COORDINATE_DELTA_DEGREES
                or longitude_delta > MAX_PROVIDER_COORDINATE_DELTA_DEGREES
            ):
                raise ValueError(
                    "provider resolved coordinates are too far from the requested zone"
                )
            provider_coordinates = {
                "latitude": provider_latitude,
                "longitude": provider_longitude,
                "elevation": provider_elevation,
            }
        except ValueError as exc:
            raise EnsembleDataQualityError(str(exc)) from exc
        return EnsembleSnapshot(
            zone_id=zone_id,
            snapshot_date_utc=requested_snapshot.isoformat(),
            request_started_at_utc=request_started_at_utc.isoformat().replace("+00:00", "Z"),
            retrieved_at_utc=retrieved_at_utc.isoformat().replace("+00:00", "Z"),
            requested_coordinates={
                "latitude": requested_latitude,
                "longitude": requested_longitude,
            },
            provider_coordinates=provider_coordinates,
            endpoint=self.base_url,
            request_parameters=public_params,
            rows=rows,
        )

    def _fetch_payload(self, params: dict[str, Any]) -> Any:
        if self._transport:
            return self._transport(params)
        for attempt in range(3):
            safe_error: EnsembleRequestError | None = None
            try:
                response = httpx.get(self.base_url, params=params, timeout=self.timeout_s)
                response.raise_for_status()
                return response.json()
            except httpx.HTTPStatusError as exc:
                status_code = exc.response.status_code
                retryable_status = status_code == 429 or 500 <= status_code < 600
                if not retryable_status or attempt == 2:
                    safe_error = EnsembleRequestError(
                        f"Ensemble API returned HTTP {status_code}", status_code=status_code
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
                if attempt == 2:
                    safe_error = EnsembleRequestError(
                        f"Ensemble API request failed ({type(exc).__name__})"
                    )
                else:
                    time.sleep(2**attempt)
            if safe_error is not None:
                raise safe_error
        raise RuntimeError("unreachable retry loop")
