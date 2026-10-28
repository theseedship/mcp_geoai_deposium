import hashlib
import json
from datetime import UTC, date, datetime

import httpx
import pytest

from src.forecast_archive import (
    PREVIOUS_RUNS_HOURLY_UNITS,
    PreviousRunsArchiveClient,
    PreviousRunsRequestError,
    _previous_field,
    normalize_previous_runs,
)
from src.utils.openmeteo import NORMALIZED_VARS

_VALUES = {
    "temperature_2m": 20.0,
    "precipitation": 0.1,
    "humidity": 60.0,
    "evapotranspiration": 0.2,
    "wind_speed": 4.0,
    "solar_radiation": 1.0,
}


def _payload(days=(1,), horizons=(1,)):
    """A synthetic Previous Runs response in the provider's own frame: UTC, hourly, with units."""
    times = [
        datetime(2025, 6, day, hour, tzinfo=UTC).isoformat() for day in days for hour in range(24)
    ]
    hourly = {"time": times}
    units = {"time": "iso8601"}
    for horizon in horizons:
        for variable in NORMALIZED_VARS:
            field = _previous_field(variable, horizon)
            hourly[field] = [_VALUES[variable]] * len(times)
            units[field] = PREVIOUS_RUNS_HOURLY_UNITS[variable]
    return {
        "latitude": 43.6,
        "longitude": 3.5,
        "utc_offset_seconds": 0,
        "timezone": "GMT",
        "hourly_units": units,
        "hourly": hourly,
    }


def _body(payload=None):
    """The exact bytes a transport hands back: the client digests these before parsing."""
    return json.dumps(payload if payload is not None else _payload()).encode("utf-8")


def _normalize(payload, variables=("temperature_2m",), **kwargs):
    return normalize_previous_runs(
        payload,
        zone_id="x",
        provider="gfs",
        model="gfs_seamless",
        horizon_days=1,
        variables=list(variables),
        **kwargs,
    )


def test_normalize_previous_runs_aggregates_complete_daily_rows():
    rows, coverage = normalize_previous_runs(
        _payload(),
        zone_id="montpellier",
        provider="gfs",
        model="gfs_seamless",
        horizon_days=1,
        variables=["temperature_2m", "precipitation", "solar_radiation"],
    )
    assert coverage.status == "available"
    assert coverage.available_days == 1
    assert {row.variable for row in rows} == {"temperature_2m", "precipitation", "solar_radiation"}
    assert all(row.valid_date == date(2025, 6, 1) for row in rows)
    assert all(row.origin_date == date(2025, 5, 31) for row in rows)


def test_client_builds_previous_runs_request_and_preserves_coverage():
    calls = []

    def transport(params):
        calls.append(params)
        return _body()

    rows, coverage = PreviousRunsArchiveClient(transport=transport).fetch_daily(
        zone_id="montpellier",
        latitude=43.6,
        longitude=3.5,
        start_date="2025-06-01",
        end_date="2025-06-01",
        variables=["temperature_2m"],
        model="gfs_seamless",
        horizon_days=1,
    )
    assert calls[0]["models"] == "gfs_seamless"
    assert calls[0]["hourly"] == "temperature_2m_previous_day1"
    assert len(rows) == 1 and coverage.status == "available"


def test_client_uses_archive_settings_and_only_sends_configured_key(monkeypatch):
    calls = []

    def transport(params):
        calls.append(params)
        return _body()

    monkeypatch.setattr(
        "src.config.settings.FORECAST_ARCHIVE_API_BASE_URL",
        "https://customer-previous-runs-api.open-meteo.com/v1/forecast",
    )
    monkeypatch.setattr(
        "src.config.settings.FORECAST_ARCHIVE_API_KEY",
        "archive-secret",
    )
    client = PreviousRunsArchiveClient(transport=transport)
    client.fetch_daily(
        zone_id="montpellier",
        latitude=43.6,
        longitude=3.5,
        start_date="2025-06-01",
        end_date="2025-06-01",
        variables=["temperature_2m"],
    )

    assert client.base_url == ("https://customer-previous-runs-api.open-meteo.com/v1/forecast")
    assert calls[0]["apikey"] == "archive-secret"


def test_client_can_explicitly_suppress_or_override_archive_credentials(monkeypatch):
    calls = []

    def transport(params):
        calls.append(params)
        return _body()

    monkeypatch.setattr(
        "src.config.settings.FORECAST_ARCHIVE_API_KEY",
        "configured-secret",
    )
    client = PreviousRunsArchiveClient(
        base_url="https://override.example.test/v1/forecast",
        apikey="",
        transport=transport,
    )
    client.fetch_daily(
        zone_id="montpellier",
        latitude=43.6,
        longitude=3.5,
        start_date="2025-06-01",
        end_date="2025-06-01",
        variables=["temperature_2m"],
    )

    assert client.base_url == "https://override.example.test/v1/forecast"
    assert "apikey" not in calls[0]


def test_client_treats_whitespace_archive_key_as_unset():
    client = PreviousRunsArchiveClient(
        base_url="https://untrusted.example.test/v1/forecast",
        apikey="   ",
        transport=lambda _params: _body(),
    )

    assert client.apikey == ""


def test_client_http_call_uses_archive_endpoint_and_key(monkeypatch):
    calls = []

    class Response:
        def raise_for_status(self):
            return None

        @property
        def content(self):
            return _body()

    def fake_get(url, *, params, timeout):
        calls.append((url, params, timeout))
        return Response()

    monkeypatch.setattr("src.forecast_archive.httpx.get", fake_get)
    client = PreviousRunsArchiveClient(
        base_url="https://customer-previous-runs-api.open-meteo.com/v1/forecast",
        apikey="archive-secret",
        timeout_s=12.0,
    )
    client.fetch_daily(
        zone_id="montpellier",
        latitude=43.6,
        longitude=3.5,
        start_date="2025-06-01",
        end_date="2025-06-01",
        variables=["temperature_2m"],
    )

    assert calls[0][0] == ("https://customer-previous-runs-api.open-meteo.com/v1/forecast")
    assert calls[0][1]["apikey"] == "archive-secret"
    assert calls[0][2] == 12.0


def test_client_redacts_key_from_http_status_errors(monkeypatch):
    def unauthorized(url, *, params, timeout):
        request = httpx.Request("GET", url, params=params)
        return httpx.Response(401, request=request)

    monkeypatch.setattr("src.forecast_archive.httpx.get", unauthorized)
    client = PreviousRunsArchiveClient(
        base_url="https://customer-previous-runs-api.open-meteo.com/v1/forecast",
        apikey="archive-secret",
    )

    with pytest.raises(PreviousRunsRequestError) as caught:
        client.fetch_daily(
            zone_id="montpellier",
            latitude=43.6,
            longitude=3.5,
            start_date="2025-06-01",
            end_date="2025-06-01",
            variables=["temperature_2m"],
        )

    assert caught.value.status_code == 401
    assert str(caught.value) == "Previous Runs API returned HTTP 401"
    assert "archive-secret" not in str(caught.value)
    assert caught.value.__context__ is None


def test_client_redacts_key_from_transport_errors(monkeypatch):
    def unavailable(url, *, params, timeout):
        request = httpx.Request("GET", url, params=params)
        raise httpx.ConnectError(
            f"failed request {request.url}",
            request=request,
        )

    monkeypatch.setattr("src.forecast_archive.httpx.get", unavailable)
    client = PreviousRunsArchiveClient(
        base_url="https://customer-previous-runs-api.open-meteo.com/v1/forecast",
        apikey="archive-secret",
    )

    with pytest.raises(PreviousRunsRequestError) as caught:
        client.fetch_daily(
            zone_id="montpellier",
            latitude=43.6,
            longitude=3.5,
            start_date="2025-06-01",
            end_date="2025-06-01",
            variables=["temperature_2m"],
        )

    assert caught.value.status_code is None
    assert str(caught.value) == "Previous Runs API request failed (ConnectError)"
    assert "archive-secret" not in str(caught.value)
    assert caught.value.__context__ is None


def test_client_retries_429_without_leaking_key(monkeypatch):
    calls = []

    def rate_limited(url, *, params, timeout):
        calls.append((url, params, timeout))
        request = httpx.Request("GET", url, params=params)
        return httpx.Response(429, headers={"Retry-After": "0"}, request=request)

    monkeypatch.setattr("src.forecast_archive.httpx.get", rate_limited)
    monkeypatch.setattr("src.forecast_archive.time.sleep", lambda _delay: None)
    client = PreviousRunsArchiveClient(
        base_url="https://customer-previous-runs-api.open-meteo.com/v1/forecast",
        apikey="archive-secret",
    )

    with pytest.raises(PreviousRunsRequestError) as caught:
        client.fetch_daily_multi(
            zone_id="montpellier",
            latitude=43.6,
            longitude=3.5,
            start_date="2025-06-01",
            end_date="2025-06-01",
            variables=["temperature_2m"],
            horizons=[1, 2],
        )

    assert len(calls) == 3
    assert caught.value.status_code == 429
    assert "archive-secret" not in str(caught.value)


@pytest.mark.parametrize(
    "base_url",
    [
        "https://previous-runs-api.open-meteo.com/v1/forecast",
        "http://customer-previous-runs-api.open-meteo.com/v1/forecast",
        "https://customer-previous-runs-api.open-meteo.com.evil.test/v1/forecast",
        "https://user@customer-previous-runs-api.open-meteo.com/v1/forecast",
        "https://customer-previous-runs-api.open-meteo.com/v1/other",
        "https://customer-previous-runs-api.open-meteo.com/v1/forecast?target=other",
    ],
)
def test_client_rejects_archive_key_for_unapproved_endpoint(base_url):
    with pytest.raises(ValueError, match="approved Open-Meteo Previous Runs endpoint"):
        PreviousRunsArchiveClient(base_url=base_url, apikey="archive-secret")


def test_client_preserves_provider_field_names_with_compound_suffixes():
    calls = []

    def transport(params):
        calls.append(params)
        return _body()

    PreviousRunsArchiveClient(transport=transport).fetch_daily(
        zone_id="montpellier",
        latitude=43.6,
        longitude=3.5,
        start_date="2025-06-01",
        end_date="2025-06-01",
        variables=["evapotranspiration", "solar_radiation"],
        model="gfs_seamless",
        horizon_days=1,
    )
    assert calls[0]["hourly"] == (
        "et0_fao_evapotranspiration_previous_day1,shortwave_radiation_previous_day1"
    )


def test_client_multi_horizon_uses_one_transport_call():
    calls = []

    def transport(params):
        calls.append(params)
        return _body()

    result = PreviousRunsArchiveClient(transport=transport).fetch_daily_multi(
        zone_id="montpellier",
        latitude=43.6,
        longitude=3.5,
        start_date="2025-06-01",
        end_date="2025-06-01",
        variables=["temperature_2m"],
        model="gfs_seamless",
        horizons=[1],
    )
    assert len(calls) == 1
    assert calls[0]["hourly"] == "temperature_2m_previous_day1"
    assert result[1][1].status == "available"


def test_missing_hourly_variable_fails_closed():
    payload = _payload()
    del payload["hourly"]["temperature_2m_previous_day1"]
    with pytest.raises(ValueError, match="missing or ragged"):
        normalize_previous_runs(
            payload,
            zone_id="x",
            provider="gfs",
            model="gfs_seamless",
            horizon_days=1,
            variables=["temperature_2m"],
        )


def test_partial_day_is_not_available():
    payload = _payload()
    payload["hourly"]["time"] = payload["hourly"]["time"][:12]
    for key in payload["hourly"]:
        if key != "time":
            payload["hourly"][key] = payload["hourly"][key][:12]
    _, coverage = normalize_previous_runs(
        payload,
        zone_id="x",
        provider="gfs",
        model="gfs_seamless",
        horizon_days=1,
        variables=["temperature_2m"],
        expected_start_date="2025-06-01",
        expected_end_date="2025-06-01",
    )
    assert coverage.available_days == 0
    assert coverage.status == "partial_archive"


def test_requested_range_is_used_for_coverage():
    _, coverage = normalize_previous_runs(
        _payload(),
        zone_id="x",
        provider="gfs",
        model="gfs_seamless",
        horizon_days=1,
        variables=["temperature_2m"],
        expected_start_date="2025-05-31",
        expected_end_date="2025-06-02",
    )
    assert coverage.expected_days == 3
    assert coverage.available_days == 1
    assert coverage.status == "partial_archive"


# ---------------------------------------------------------------------------------------
# #193 — the response frame is refused, not trusted: UTC, units, a unique hourly grid and
# the requested dates. Inside that frame a gap belongs to one (date, variable), never a day.
# ---------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("field", "value"),
    [("utc_offset_seconds", 3600), ("utc_offset_seconds", None), ("timezone", "Europe/Paris")],
)
def test_a_response_outside_utc_is_refused(field, value):
    payload = _payload()
    payload[field] = value
    with pytest.raises(ValueError, match="must use UTC"):
        _normalize(payload)


def test_a_unit_other_than_the_expected_one_is_refused():
    payload = _payload()
    payload["hourly_units"]["temperature_2m_previous_day1"] = "°F"
    with pytest.raises(ValueError, match="temperature_2m_previous_day1 must use '°C'"):
        _normalize(payload)


def test_a_missing_unit_is_refused():
    payload = _payload()
    del payload["hourly_units"]["temperature_2m_previous_day1"]
    with pytest.raises(ValueError, match="must use '°C'"):
        _normalize(payload)


def test_twenty_four_entries_with_a_duplicate_are_refused_not_a_complete_day():
    payload = _payload()
    payload["hourly"]["time"][23] = payload["hourly"]["time"][22]
    with pytest.raises(ValueError, match="duplicate timestamp"):
        _normalize(payload, expected_start_date="2025-06-01", expected_end_date="2025-06-01")


def test_a_timestamp_off_the_hourly_grid_is_refused():
    payload = _payload()
    payload["hourly"]["time"][5] = "2025-06-01T05:30:00+00:00"
    with pytest.raises(ValueError, match="hourly grid"):
        _normalize(payload)


def test_a_timestamp_outside_the_requested_dates_is_refused():
    with pytest.raises(ValueError, match="outside the requested dates"):
        _normalize(
            _payload(days=(1, 2)),
            expected_start_date="2025-06-01",
            expected_end_date="2025-06-01",
        )


def test_one_incomplete_variable_keeps_the_other_variables_of_that_day():
    payload = _payload(days=(1, 2))
    payload["hourly"]["relative_humidity_2m_previous_day1"][30] = None
    rows, coverage = _normalize(
        payload,
        variables=NORMALIZED_VARS,
        expected_start_date="2025-06-01",
        expected_end_date="2025-06-02",
    )

    day2 = [row for row in rows if row.valid_date == date(2025, 6, 2)]
    assert len(day2) == 5
    assert "humidity" not in {row.variable for row in day2}
    assert coverage.missing == (
        {"valid_date": "2025-06-02", "variable": "humidity", "horizon_days": 1},
    )
    assert coverage.expected_pairs == 12
    assert coverage.available_pairs == 11
    assert coverage.available_days == 1
    assert coverage.status == "partial_archive"
    assert all(row.value != 0.0 for row in rows)


def test_a_requested_day_with_no_hours_is_listed_missing_and_keeps_the_denominator():
    _, coverage = _normalize(
        _payload(days=(1,)),
        variables=("temperature_2m", "precipitation"),
        expected_start_date="2025-06-01",
        expected_end_date="2025-06-02",
    )
    assert coverage.expected_pairs == 4
    assert coverage.available_pairs == 2
    assert [item["valid_date"] for item in coverage.missing] == ["2025-06-02", "2025-06-02"]


# ---------------------------------------------------------------------------------------
# #193 — the provider's exact bytes are kept beside every parsed result, digested before
# parsing, with the request that produced them and never the API key.
# ---------------------------------------------------------------------------------------

_FIXED_NOW = datetime(2026, 10, 15, 12, 0, tzinfo=UTC)


def _multi(client, horizons=(1,)):
    return client.fetch_daily_multi(
        zone_id="montpellier",
        latitude=43.6,
        longitude=3.5,
        start_date="2025-06-01",
        end_date="2025-06-01",
        variables=["temperature_2m"],
        horizons=list(horizons),
    )


def test_multi_horizon_results_carry_the_exact_raw_bytes_and_their_digest():
    body = _body(_payload(horizons=(1, 3)))
    client = PreviousRunsArchiveClient(transport=lambda _params: body, now=lambda: _FIXED_NOW)

    results = _multi(client, horizons=(1, 3))

    for horizon in (1, 3):
        rows, coverage, raw = results[horizon]
        assert raw.body == body
        assert raw.sha256 == hashlib.sha256(body).hexdigest()
        assert raw.retrieved_at == "2026-10-15T12:00:00+00:00"
        assert coverage.status == "available" and len(rows) == 1
    assert results[1].raw is results[3].raw


def test_raw_params_never_carry_the_api_key(monkeypatch):
    sent = []

    def transport(params):
        sent.append(params)
        return _body()

    client = PreviousRunsArchiveClient(
        base_url="https://customer-previous-runs-api.open-meteo.com/v1/forecast",
        apikey="archive-secret",
        transport=transport,
    )
    raw = _multi(client)[1].raw

    assert sent[0]["apikey"] == "archive-secret"
    assert "apikey" not in raw.params
    assert "archive-secret" not in json.dumps(raw.params)
    assert raw.params["latitude"] == 43.6 and raw.params["models"] == "gfs_seamless"


def test_a_transport_must_return_the_response_bytes():
    client = PreviousRunsArchiveClient(transport=lambda _params: _payload())
    with pytest.raises(TypeError, match="bytes"):
        client.fetch_daily(
            zone_id="montpellier",
            latitude=43.6,
            longitude=3.5,
            start_date="2025-06-01",
            end_date="2025-06-01",
            variables=["temperature_2m"],
        )


@pytest.mark.parametrize("value", [{"v": 1}, [1.0], "1.5", True])
def test_a_non_numeric_hourly_value_is_a_refused_response_with_its_bytes(value):
    from src.forecast_archive import ResponseRefused

    payload = _payload()
    payload["hourly"]["temperature_2m_previous_day1"][3] = value
    body = _body(payload)

    result = _multi(PreviousRunsArchiveClient(transport=lambda _params: body))[1]

    assert isinstance(result, ResponseRefused)
    assert "non-numeric" in str(result)
    assert result.raw.body == body


def test_a_programming_error_in_normalization_is_not_swallowed(monkeypatch):
    """Only a refused RESPONSE becomes a per-horizon result; a bug must surface as itself."""

    def broken(*_args, **_kwargs):
        raise TypeError("a bug, not a malformed response")

    monkeypatch.setattr("src.forecast_archive.normalize_previous_runs", broken)
    client = PreviousRunsArchiveClient(transport=lambda _params: _body())

    with pytest.raises(TypeError, match="a bug"):
        _multi(client)


def test_a_programming_error_in_the_per_horizon_fallback_is_not_swallowed(monkeypatch):
    def broken(*_args, **_kwargs):
        raise TypeError("a bug, not a malformed response")

    def mixed_fields_rejected(params):
        if "," in params["hourly"]:
            raise RuntimeError("provider rejects a mixed field list")
        return _body(_payload(horizons=(1, 3)))

    monkeypatch.setattr("src.forecast_archive.normalize_previous_runs", broken)
    client = PreviousRunsArchiveClient(transport=mixed_fields_rejected)

    with pytest.raises(TypeError, match="a bug"):
        _multi(client, horizons=(1, 3))
