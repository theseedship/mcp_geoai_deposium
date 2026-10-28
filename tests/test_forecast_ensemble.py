from __future__ import annotations

import json
from datetime import UTC, date, datetime, timedelta

import httpx
import pytest

from src.forecast_backtest import load_rows
from src.forecast_ensemble import (
    EXPECTED_MEMBER_COUNT,
    MEMBER_FIELDS,
    MODEL,
    PROVIDER,
    EnsembleDataQualityError,
    EnsembleRequestError,
    OpenMeteoEnsembleClient,
    normalize_aifs_ensemble_precipitation,
)

SNAPSHOT_DATE = date(2026, 7, 15)
NOW = datetime(2026, 7, 15, 19, 7, tzinfo=UTC)


def _payload(*, days: int = 8) -> dict:
    dates = [(SNAPSHOT_DATE + timedelta(days=offset)).isoformat() for offset in range(1, days + 1)]
    daily = {"time": dates}
    units = {"time": "iso8601"}
    for member_id, field in enumerate(MEMBER_FIELDS):
        value = 0.1 if member_id < 11 else 1.0
        daily[field] = [value] * days
        units[field] = "mm"
    return {
        "latitude": 43.6,
        "longitude": 3.875,
        "elevation": 42.0,
        "timezone": "GMT",
        "utc_offset_seconds": 0,
        "daily_units": units,
        "daily": daily,
    }


def test_normalizer_uses_all_51_members_and_strict_event_threshold():
    rows = normalize_aifs_ensemble_precipitation(
        _payload(),
        zone_id="montpellier",
        snapshot_date=SNAPSHOT_DATE,
    )

    assert [row.horizon_days for row in rows] == [1, 3, 5, 7]
    row = rows[0]
    assert row.origin_date == "2026-07-15"
    assert row.valid_date == "2026-07-16"
    assert row.variable == "precipitation"
    assert row.value == pytest.approx((11 * 0.1 + 40) / EXPECTED_MEMBER_COUNT)
    assert row.event_member_count == 40
    assert row.probability == pytest.approx(40 / EXPECTED_MEMBER_COUNT)
    assert row.probability_event == {"operator": ">", "threshold_mm": 0.1}
    assert row.member_count_expected == EXPECTED_MEMBER_COUNT
    assert row.member_count_used == EXPECTED_MEMBER_COUNT
    assert [member.member_id for member in row.members] == list(range(EXPECTED_MEMBER_COUNT))
    assert [member.source_field for member in row.members] == list(MEMBER_FIELDS)
    assert row.calibration_status == "uncalibrated"
    assert row.origin_semantics == "snapshot_date_utc_not_model_initialization"


@pytest.mark.parametrize(
    "mutation,match",
    [
        (lambda payload: payload["daily"].pop(MEMBER_FIELDS[-1]), "exactly 51"),
        (
            lambda payload: payload["daily"].__setitem__("precipitation_sum_member51", [1.0] * 8),
            "exactly 51",
        ),
        (
            lambda payload: payload["daily"][MEMBER_FIELDS[1]].pop(),
            "missing or ragged",
        ),
        (
            lambda payload: payload["daily"][MEMBER_FIELDS[1]].__setitem__(0, None),
            "finite number",
        ),
        (
            lambda payload: payload["daily"][MEMBER_FIELDS[1]].__setitem__(0, "1.0"),
            "finite number",
        ),
        (
            lambda payload: payload["daily"][MEMBER_FIELDS[1]].__setitem__(0, float("nan")),
            "finite number",
        ),
        (
            lambda payload: payload["daily"][MEMBER_FIELDS[1]].__setitem__(0, -0.1),
            ">= 0",
        ),
        (
            lambda payload: payload["daily_units"].__setitem__(MEMBER_FIELDS[1], "inch"),
            "millimetres",
        ),
        (lambda payload: payload.__setitem__("utc_offset_seconds", 3600), "use UTC"),
    ],
)
def test_normalizer_fails_closed_on_incomplete_or_invalid_members(mutation, match):
    payload = _payload()
    mutation(payload)

    with pytest.raises(ValueError, match=match):
        normalize_aifs_ensemble_precipitation(
            payload,
            zone_id="montpellier",
            snapshot_date=SNAPSHOT_DATE,
        )


def test_normalizer_rejects_missing_duplicate_dates_and_invalid_horizons():
    payload = _payload(days=6)
    with pytest.raises(ValueError, match="missing horizon 7"):
        normalize_aifs_ensemble_precipitation(
            payload,
            zone_id="montpellier",
            snapshot_date=SNAPSHOT_DATE,
        )

    payload = _payload()
    payload["daily"]["time"][1] = payload["daily"]["time"][0]
    with pytest.raises(ValueError, match="duplicate daily dates"):
        normalize_aifs_ensemble_precipitation(
            payload,
            zone_id="montpellier",
            snapshot_date=SNAPSHOT_DATE,
        )

    with pytest.raises(ValueError, match="unique integers"):
        normalize_aifs_ensemble_precipitation(
            _payload(),
            zone_id="montpellier",
            snapshot_date=SNAPSHOT_DATE,
            horizons=(1, 1),
        )


def test_client_uses_canonical_request_and_excludes_credentials_from_snapshot():
    captured = {}

    def transport(params):
        captured.update(params)
        return _payload()

    client = OpenMeteoEnsembleClient(
        base_url="https://customer-ensemble-api.open-meteo.com/v1/ensemble",
        apikey="ensemble-secret",
        transport=transport,
        clock=lambda: NOW,
    )
    snapshot = client.fetch_snapshot(
        zone_id="montpellier",
        latitude=43.6,
        longitude=3.875,
        snapshot_date=SNAPSHOT_DATE,
    )

    assert captured == {
        "latitude": 43.6,
        "longitude": 3.875,
        "start_date": "2026-07-16",
        "end_date": "2026-07-22",
        "models": MODEL,
        "daily": "precipitation_sum",
        "timezone": "UTC",
        "precipitation_unit": "mm",
        "apikey": "ensemble-secret",
    }
    serialized = snapshot.to_dict()
    assert serialized["provider"] == PROVIDER
    assert serialized["request_started_at_utc"] == "2026-07-15T19:07:00Z"
    assert serialized["retrieved_at_utc"] == "2026-07-15T19:07:00Z"
    assert serialized["provider_coordinates"]["longitude"] == 3.875
    assert "ensemble-secret" not in json.dumps(serialized)
    assert "apikey" not in serialized["request_parameters"]


def test_client_rejects_retrospective_snapshot_and_naive_clock():
    client = OpenMeteoEnsembleClient(transport=lambda _params: _payload(), clock=lambda: NOW)
    with pytest.raises(ValueError, match="current UTC date"):
        client.fetch_snapshot(
            zone_id="montpellier",
            latitude=43.6,
            longitude=3.875,
            snapshot_date=SNAPSHOT_DATE - timedelta(days=1),
        )

    naive = OpenMeteoEnsembleClient(
        transport=lambda _params: _payload(),
        clock=lambda: NOW.replace(tzinfo=None),
    )
    with pytest.raises(ValueError, match="timezone-aware"):
        naive.fetch_snapshot(zone_id="montpellier", latitude=43.6, longitude=3.875)

    clock_values = iter((NOW, NOW.replace(tzinfo=None)))
    naive_completion = OpenMeteoEnsembleClient(
        transport=lambda _params: _payload(),
        clock=lambda: next(clock_values),
    )
    with pytest.raises(ValueError, match="timezone-aware"):
        naive_completion.fetch_snapshot(
            zone_id="montpellier",
            latitude=43.6,
            longitude=3.875,
        )


def test_client_serializes_request_start_and_response_completion():
    events = []
    clock_values = iter((NOW, NOW + timedelta(seconds=4)))

    def clock():
        events.append("clock")
        return next(clock_values)

    def transport(_params):
        events.append("transport")
        return _payload()

    snapshot = OpenMeteoEnsembleClient(transport=transport, clock=clock).fetch_snapshot(
        zone_id="montpellier",
        latitude=43.6,
        longitude=3.875,
    )

    serialized = snapshot.to_dict()
    assert events == ["clock", "transport", "clock"]
    assert snapshot.request_started_at_utc == "2026-07-15T19:07:00Z"
    assert snapshot.retrieved_at_utc == "2026-07-15T19:07:04Z"
    assert serialized["request_started_at_utc"] == snapshot.request_started_at_utc
    assert serialized["retrieved_at_utc"] == snapshot.retrieved_at_utc


def test_client_rejects_request_crossing_utc_midnight():
    clock_values = iter(
        (
            datetime(2026, 7, 15, 20, 59, 59, tzinfo=UTC),
            datetime(2026, 7, 16, 0, 0, tzinfo=UTC),
        )
    )
    client = OpenMeteoEnsembleClient(
        transport=lambda _params: _payload(),
        clock=lambda: next(clock_values),
    )

    with pytest.raises(ValueError, match="crossed a UTC date boundary"):
        client.fetch_snapshot(
            zone_id="montpellier",
            latitude=43.6,
            longitude=3.875,
            snapshot_date=SNAPSHOT_DATE,
        )


@pytest.mark.parametrize(
    "start,finish",
    [
        (datetime(2026, 7, 15, 18, 59, tzinfo=UTC), None),
        (
            datetime(2026, 7, 15, 20, 59, 59, tzinfo=UTC),
            datetime(2026, 7, 15, 21, 0, tzinfo=UTC),
        ),
    ],
)
def test_client_rejects_acquisition_outside_fixed_utc_window(start, finish):
    clock_values = iter((start,) if finish is None else (start, finish))
    client = OpenMeteoEnsembleClient(
        transport=lambda _params: _payload(),
        clock=lambda: next(clock_values),
    )

    with pytest.raises(ValueError, match="fixed UTC capture window"):
        client.fetch_snapshot(
            zone_id="montpellier",
            latitude=43.6,
            longitude=3.875,
            snapshot_date=SNAPSHOT_DATE,
        )


def test_client_rejects_clock_moving_backwards_during_request():
    clock_values = iter((NOW, NOW - timedelta(microseconds=1)))
    client = OpenMeteoEnsembleClient(
        transport=lambda _params: _payload(),
        clock=lambda: next(clock_values),
    )

    with pytest.raises(ValueError, match="clock moved backwards"):
        client.fetch_snapshot(
            zone_id="montpellier",
            latitude=43.6,
            longitude=3.875,
            snapshot_date=SNAPSHOT_DATE,
        )


def test_client_rejects_endpoint_credentials_and_non_object_transport():
    with pytest.raises(ValueError, match="credential-free HTTPS"):
        OpenMeteoEnsembleClient(
            base_url="https://ensemble-api.open-meteo.com/v1/ensemble?token=secret",
            transport=lambda _params: _payload(),
            clock=lambda: NOW,
        )

    client = OpenMeteoEnsembleClient(transport=lambda _params: [], clock=lambda: NOW)
    with pytest.raises(ValueError, match="non-object payload"):
        client.fetch_snapshot(zone_id="montpellier", latitude=43.6, longitude=3.875)


@pytest.mark.parametrize("transient_status", [429, 500, 502, 503, 504, 599])
def test_client_retries_transient_http_status_then_succeeds(monkeypatch, transient_status):
    calls = []
    sleeps = []

    def responder(url, *, params, timeout):
        calls.append((url, params, timeout))
        request = httpx.Request("GET", url, params=params)
        if len(calls) == 1:
            return httpx.Response(transient_status, request=request)
        return httpx.Response(200, json=_payload(), request=request)

    monkeypatch.setattr("src.forecast_ensemble.httpx.get", responder)
    monkeypatch.setattr("src.forecast_ensemble.time.sleep", sleeps.append)
    client = OpenMeteoEnsembleClient(apikey="", clock=lambda: NOW)

    snapshot = client.fetch_snapshot(
        zone_id="montpellier",
        latitude=43.6,
        longitude=3.875,
    )

    assert snapshot.zone_id == "montpellier"
    assert len(calls) == 2
    assert sleeps == [1.0]


def test_client_retries_request_error_then_succeeds(monkeypatch):
    calls = []
    sleeps = []

    def responder(url, *, params, timeout):
        calls.append((url, params, timeout))
        request = httpx.Request("GET", url, params=params)
        if len(calls) == 1:
            raise httpx.ConnectError("temporary outage", request=request)
        return httpx.Response(200, json=_payload(), request=request)

    monkeypatch.setattr("src.forecast_ensemble.httpx.get", responder)
    monkeypatch.setattr("src.forecast_ensemble.time.sleep", sleeps.append)
    client = OpenMeteoEnsembleClient(apikey="", clock=lambda: NOW)

    snapshot = client.fetch_snapshot(
        zone_id="montpellier",
        latitude=43.6,
        longitude=3.875,
    )

    assert snapshot.zone_id == "montpellier"
    assert len(calls) == 2
    assert sleeps == [1]


def test_client_does_not_retry_non_rate_limited_4xx(monkeypatch):
    calls = []

    def responder(url, *, params, timeout):
        calls.append((url, params, timeout))
        request = httpx.Request("GET", url, params=params)
        return httpx.Response(400, request=request)

    monkeypatch.setattr("src.forecast_ensemble.httpx.get", responder)
    client = OpenMeteoEnsembleClient(apikey="", clock=lambda: NOW)

    with pytest.raises(EnsembleRequestError) as caught:
        client.fetch_snapshot(
            zone_id="montpellier",
            latitude=43.6,
            longitude=3.875,
        )

    assert caught.value.status_code == 400
    assert len(calls) == 1


def test_client_exhausts_transient_retry_budget_with_safe_error(monkeypatch):
    calls = []

    def unavailable(url, *, params, timeout):
        calls.append((url, params, timeout))
        request = httpx.Request("GET", url, params=params)
        return httpx.Response(503, request=request)

    monkeypatch.setattr("src.forecast_ensemble.httpx.get", unavailable)
    monkeypatch.setattr("src.forecast_ensemble.time.sleep", lambda _delay: None)
    client = OpenMeteoEnsembleClient(apikey="", clock=lambda: NOW)

    with pytest.raises(EnsembleRequestError) as caught:
        client.fetch_snapshot(
            zone_id="montpellier",
            latitude=43.6,
            longitude=3.875,
        )

    assert caught.value.status_code == 503
    assert str(caught.value) == "Ensemble API returned HTTP 503"
    assert len(calls) == 3


def test_snapshot_rows_are_directly_backtest_compatible(tmp_path):
    client = OpenMeteoEnsembleClient(transport=lambda _params: _payload(), clock=lambda: NOW)
    snapshot = client.fetch_snapshot(
        zone_id="montpellier",
        latitude=43.6,
        longitude=3.875,
        snapshot_date=SNAPSHOT_DATE,
    )
    artifact = tmp_path / "snapshot.json"
    artifact.write_text(json.dumps(snapshot.to_dict()), encoding="utf-8")

    rows = load_rows(artifact, kind="prediction", provider=PROVIDER)

    assert len(rows) == 4
    assert all(row.probability == pytest.approx(40 / EXPECTED_MEMBER_COUNT) for row in rows)
    assert all(row.probability_event is not None for row in rows)


def test_client_captures_negative_member_before_data_quality_rejection():
    payload = _payload()
    payload["daily"][MEMBER_FIELDS[1]][0] = -0.1
    captures = []
    client = OpenMeteoEnsembleClient(transport=lambda _params: payload, clock=lambda: NOW)

    with pytest.raises(EnsembleDataQualityError, match=">= 0") as caught:
        client.fetch_snapshot(
            zone_id="montpellier", latitude=43.6, longitude=3.875, on_capture=captures.append
        )

    assert len(captures) == 1
    assert captures[0]["payload"]["daily"][MEMBER_FIELDS[1]][0] == -0.1
    assert type(caught.value.__cause__) is ValueError


def test_client_capture_metadata_is_public_and_snapshot_format_is_unchanged():
    payload = _payload()
    captures = []
    client = OpenMeteoEnsembleClient(
        base_url="https://customer-ensemble-api.open-meteo.com/v1/ensemble",
        apikey="ensemble-secret",
        transport=lambda _params: payload,
        clock=lambda: NOW,
    )
    snapshot = client.fetch_snapshot(
        zone_id="montpellier", latitude=43.6, longitude=3.875, on_capture=captures.append
    )

    assert captures == [
        {
            "schema_version": "forecast-ensemble-capture/v1",
            "zone_id": "montpellier",
            "snapshot_date_utc": "2026-07-15",
            "request_started_at_utc": "2026-07-15T19:07:00Z",
            "retrieved_at_utc": "2026-07-15T19:07:00Z",
            "provider": PROVIDER,
            "model": MODEL,
            "requested_coordinates": {"latitude": 43.6, "longitude": 3.875},
            "endpoint": client.base_url,
            "request_parameters": snapshot.request_parameters,
            "payload": payload,
        }
    ]
    assert "ensemble-secret" not in json.dumps(captures)
    assert "apikey" not in captures[0]["request_parameters"]
    assert captures[0]["payload"] is not payload
    assert captures[0]["request_parameters"] is not snapshot.request_parameters
    baseline = client.fetch_snapshot(zone_id="montpellier", latitude=43.6, longitude=3.875)
    assert snapshot.to_dict() == baseline.to_dict()


def test_client_capture_callback_cannot_mutate_normalization_or_snapshot_metadata():
    payload = _payload()

    def mutate_capture(capture):
        capture["request_parameters"]["models"] = "other-model"
        capture["requested_coordinates"]["latitude"] = 0
        capture["payload"]["daily"][MEMBER_FIELDS[1]][0] = -0.1

    snapshot = OpenMeteoEnsembleClient(
        transport=lambda _params: payload, clock=lambda: NOW
    ).fetch_snapshot(
        zone_id="montpellier", latitude=43.6, longitude=3.875, on_capture=mutate_capture
    )

    assert payload["daily"][MEMBER_FIELDS[1]][0] == 0.1
    assert snapshot.request_parameters["models"] == MODEL
    assert snapshot.requested_coordinates["latitude"] == 43.6
    assert snapshot.rows[0].event_member_count == 40


@pytest.mark.parametrize("payload", [[], None, "provider failure", 1])
@pytest.mark.parametrize("use_http", [False, True])
def test_client_captures_nonobject_decoded_json_before_quality_rejection(
    monkeypatch, payload, use_http
):
    captures = []

    def responder(url, *, params, timeout):
        return httpx.Response(
            200, content=json.dumps(payload), request=httpx.Request("GET", url, params=params)
        )

    monkeypatch.setattr("src.forecast_ensemble.httpx.get", responder)
    client = OpenMeteoEnsembleClient(
        apikey="", transport=None if use_http else lambda _params: payload, clock=lambda: NOW
    )

    with pytest.raises(EnsembleDataQualityError, match="non-object payload"):
        client.fetch_snapshot(
            zone_id="montpellier", latitude=43.6, longitude=3.875, on_capture=captures.append
        )

    assert len(captures) == 1
    assert captures[0]["payload"] == payload


@pytest.mark.parametrize(
    "field,value", [("latitude", 91), ("longitude", None), ("elevation", "bad")]
)
def test_client_provider_coordinate_refusals_are_data_quality_errors(field, value):
    payload = _payload()
    payload[field] = value
    captures = []
    client = OpenMeteoEnsembleClient(transport=lambda _params: payload, clock=lambda: NOW)

    with pytest.raises(EnsembleDataQualityError, match="provider"):
        client.fetch_snapshot(
            zone_id="montpellier", latitude=43.6, longitude=3.875, on_capture=captures.append
        )

    assert captures[0]["payload"][field] == value


@pytest.mark.parametrize("field", ["latitude", "longitude"])
@pytest.mark.parametrize("delta", [-0.25, 0.25])
def test_client_accepts_provider_coordinates_at_exact_proximity_limit(field, delta):
    requested = {"latitude": 43.5, "longitude": 3.75}
    payload = {**_payload(), **requested}
    payload[field] += delta
    captures = []
    client = OpenMeteoEnsembleClient(transport=lambda _params: payload, clock=lambda: NOW)

    snapshot = client.fetch_snapshot(zone_id="montpellier", **requested, on_capture=captures.append)

    assert snapshot.provider_coordinates[field] == payload[field]
    assert captures[0]["payload"][field] == payload[field]


@pytest.mark.parametrize("field", ["latitude", "longitude"])
@pytest.mark.parametrize("delta", [-0.250001, 0.250001])
def test_client_captures_far_resolved_coordinates_before_data_quality_rejection(field, delta):
    requested = {"latitude": 43.5, "longitude": 3.75}
    payload = {**_payload(), **requested}
    payload[field] += delta
    captures = []
    client = OpenMeteoEnsembleClient(transport=lambda _params: payload, clock=lambda: NOW)

    with pytest.raises(EnsembleDataQualityError, match="resolved coordinates.*too far") as caught:
        client.fetch_snapshot(zone_id="montpellier", **requested, on_capture=captures.append)

    assert len(captures) == 1
    assert captures[0]["payload"][field] == payload[field]
    assert type(caught.value.__cause__) is ValueError


def test_client_provider_longitude_proximity_wraps_at_antimeridian():
    payload = {**_payload(), "longitude": -179.875}
    client = OpenMeteoEnsembleClient(transport=lambda _params: payload, clock=lambda: NOW)

    snapshot = client.fetch_snapshot(zone_id="antimeridian", latitude=43.6, longitude=179.875)

    assert snapshot.provider_coordinates["longitude"] == -179.875


@pytest.mark.parametrize(
    "start,finish,match",
    [
        (NOW.replace(tzinfo=None), NOW, "timezone-aware"),
        (NOW, NOW.replace(tzinfo=None), "timezone-aware"),
        (NOW, NOW - timedelta(seconds=1), "clock moved backwards"),
        (NOW, NOW + timedelta(days=1), "crossed a UTC date boundary"),
        (NOW.replace(hour=18), NOW, "fixed UTC capture window"),
        (NOW, NOW.replace(hour=21), "fixed UTC capture window"),
    ],
)
def test_client_does_not_capture_before_clock_and_window_checks(start, finish, match):
    captures = []
    clock_values = iter((start, finish))
    client = OpenMeteoEnsembleClient(
        transport=lambda _params: _payload(), clock=lambda: next(clock_values)
    )

    with pytest.raises(ValueError, match=match) as caught:
        client.fetch_snapshot(
            zone_id="montpellier", latitude=43.6, longitude=3.875, on_capture=captures.append
        )

    assert not isinstance(caught.value, EnsembleDataQualityError)
    assert captures == []


@pytest.mark.parametrize("error", [ValueError("writer refused"), OSError("disk full")])
def test_client_does_not_wrap_capture_callback_errors(error):
    def fail_capture(_capture):
        raise error

    client = OpenMeteoEnsembleClient(transport=lambda _params: _payload(), clock=lambda: NOW)
    with pytest.raises(type(error)) as caught:
        client.fetch_snapshot(
            zone_id="montpellier", latitude=43.6, longitude=3.875, on_capture=fail_capture
        )

    assert caught.value is error


@pytest.mark.parametrize(
    "kwargs",
    [
        {"zone_id": ""},
        {"event_threshold_mm": -0.1},
        {"horizons": (1, 1)},
        {"latitude": 91},
        {"snapshot_date": SNAPSHOT_DATE - timedelta(days=1)},
    ],
)
def test_client_request_configuration_errors_stay_fatal_before_capture(kwargs):
    captures = []
    requests = []

    def transport(params):
        requests.append(params)
        return _payload()

    options = {"zone_id": "montpellier", "latitude": 43.6, "longitude": 3.875, **kwargs}
    client = OpenMeteoEnsembleClient(transport=transport, clock=lambda: NOW)
    with pytest.raises(ValueError) as caught:
        client.fetch_snapshot(**options, on_capture=captures.append)

    assert not isinstance(caught.value, EnsembleDataQualityError)
    assert requests == []
    assert captures == []


def test_client_transport_value_error_stays_fatal_without_capture():
    captures = []
    error = ValueError("response was not JSON")

    def transport(_params):
        raise error

    client = OpenMeteoEnsembleClient(transport=transport, clock=lambda: NOW)
    with pytest.raises(ValueError) as caught:
        client.fetch_snapshot(
            zone_id="montpellier", latitude=43.6, longitude=3.875, on_capture=captures.append
        )

    assert caught.value is error
    assert captures == []


def test_client_malformed_http_json_stays_fatal_without_capture(monkeypatch):
    captures = []

    def responder(url, *, params, timeout):
        return httpx.Response(
            200, content=b"not JSON", request=httpx.Request("GET", url, params=params)
        )

    monkeypatch.setattr("src.forecast_ensemble.httpx.get", responder)
    client = OpenMeteoEnsembleClient(apikey="", clock=lambda: NOW)
    with pytest.raises(json.JSONDecodeError) as caught:
        client.fetch_snapshot(
            zone_id="montpellier", latitude=43.6, longitude=3.875, on_capture=captures.append
        )

    assert not isinstance(caught.value, EnsembleDataQualityError)
    assert captures == []
