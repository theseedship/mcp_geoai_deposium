from __future__ import annotations

import json
import logging
from datetime import UTC, datetime, timedelta
from urllib.parse import quote

import httpx
import pytest
from pydantic import ValidationError

from src.solar import SolarFetchError, SolarSettings, fetch_solar
from src.solar_contracts import SolarQuery

SOLAR_VARS = [
    "shortwave_radiation",
    "diffuse_radiation",
    "direct_normal_irradiance",
]
UNITS = dict.fromkeys((*SOLAR_VARS, "global_tilted_irradiance"), "W/m²")


def _query(source: str = "forecast", **overrides) -> SolarQuery:
    fields = {
        "source": source,
        "latitude": 43.6,
        "longitude": 3.9,
        "start_date": "2025-06-01",
        "end_date": "2025-06-01",
    }
    fields.update(overrides)
    return SolarQuery(**fields)


def _payload(query: SolarQuery, *, native: bool = False) -> dict:
    step = 30 if native else 60
    count = 48 if native else 24
    series_name = "hourly"
    start = datetime.combine(query.start_date, datetime.min.time(), tzinfo=UTC)
    times = [
        (start + timedelta(minutes=step * i)).isoformat().replace("+00:00", "Z")
        for i in range(count)
    ]
    units = {
        variable: UNITS.get(variable, "°C" if variable == "temperature_2m" else "m/s")
        for variable in query.variables
    }
    return {
        "latitude": 43.61,
        "longitude": 3.91,
        "elevation": 31.0,
        "timezone": "UTC",
        "utc_offset_seconds": 0,
        "models": ["model_returned_by_provider"],
        series_name: {
            "time": times,
            **{variable: [4.25] * count for variable in query.variables},
        },
        f"{series_name}_units": units,
    }


def _client_for(
    query: SolarQuery, payload: dict, seen: list[httpx.Request] | None = None
) -> httpx.Client:
    def handler(request: httpx.Request) -> httpx.Response:
        if seen is not None:
            seen.append(request)
        if payload.get("_allow_nonfinite_json"):
            safe_payload = payload | {"_allow_nonfinite_json": False}
            return httpx.Response(
                200,
                content=json.dumps(safe_payload, allow_nan=True).encode(),
                request=request,
            )
        return httpx.Response(200, json=payload, request=request)

    return httpx.Client(transport=httpx.MockTransport(handler))


def _captured_provider_payload(query: SolarQuery) -> dict:
    """One-day slices from forecast-aifs-fos, archive-era5, and satellite-sarah3 captures."""
    native = query.temporal_resolution == "native"
    step_minutes = 30 if native else 60
    count = 48 if native else 24
    start = datetime.combine(query.start_date, datetime.min.time(), tzinfo=UTC)
    times = [
        (start + timedelta(minutes=step_minutes * index)).strftime("%Y-%m-%dT%H:%M")
        for index in range(count)
    ]
    captured = {
        "forecast": {
            "coordinates": (43.5, 5.0, 4.0),
            "values": {
                "shortwave_radiation": [
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    1,
                    92,
                    235,
                    357,
                    451,
                    510,
                    530,
                    510,
                    449,
                    352,
                    224,
                    76,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                ]
            },
        },
        "era5": {
            "coordinates": (43.5, 3.75, 39.0),
            "values": {
                "shortwave_radiation": [
                    0,
                    0,
                    0,
                    0,
                    0,
                    25,
                    146,
                    293,
                    481,
                    642,
                    767,
                    862,
                    904,
                    876,
                    836,
                    687,
                    496,
                    240,
                    137,
                    32,
                    1,
                    0,
                    0,
                    0,
                ]
            },
        },
        "satellite": {
            "coordinates": (43.6, 3.8499985, 39.0),
            "values": {
                "shortwave_radiation": [
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    5,
                    17,
                    12,
                    19,
                    27,
                    92,
                    169,
                    147,
                    165,
                    100,
                    88,
                    104,
                    115,
                    107,
                    115,
                    184,
                    306,
                    275,
                    115,
                    75,
                    153,
                    190,
                    112,
                    149,
                    84,
                    65,
                    34,
                    46,
                    51,
                    9,
                    2,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                ],
                "diffuse_radiation": [
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    5,
                    17,
                    12,
                    19,
                    27,
                    92,
                    167,
                    145,
                    164,
                    100,
                    88,
                    104,
                    115,
                    107,
                    115,
                    184,
                    305,
                    274,
                    115,
                    75,
                    153,
                    190,
                    112,
                    149,
                    84,
                    65,
                    34,
                    46,
                    49,
                    8,
                    2,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                ],
                "direct_normal_irradiance": [
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    3.8,
                    3.3,
                    1.5,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    1.1,
                    1.1,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    12.2,
                    12.9,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                ],
                "global_tilted_irradiance": [
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    4.7,
                    16.1,
                    11.4,
                    18,
                    25.6,
                    87.1,
                    159.8,
                    139.1,
                    156.2,
                    94.6,
                    83.3,
                    98.4,
                    108.8,
                    101.3,
                    108.8,
                    174.1,
                    289.7,
                    260.4,
                    108.8,
                    71,
                    144.8,
                    179.8,
                    106,
                    141,
                    79.5,
                    61.5,
                    32.2,
                    43.5,
                    46.4,
                    7.6,
                    1.9,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                ],
            },
        },
    }[query.source]
    latitude, longitude, elevation = captured["coordinates"]
    return {
        "latitude": latitude,
        "longitude": longitude,
        "elevation": elevation,
        "timezone": "GMT",
        "timezone_abbreviation": "GMT",
        "utc_offset_seconds": 0,
        "hourly": {"time": times, **captured["values"]},
        "hourly_units": {"time": "iso8601", **UNITS},
    }


@pytest.mark.parametrize(
    ("source", "start", "variables", "resolution"),
    [
        ("forecast", "2026-10-11", ["shortwave_radiation"], "hourly"),
        ("era5", "2025-06-01", ["shortwave_radiation"], "hourly"),
        (
            "satellite",
            "1983-06-01",
            SOLAR_VARS + ["global_tilted_irradiance"],
            "native",
        ),
    ],
)
def test_actual_provider_capture_shapes_are_accepted_offline(source, start, variables, resolution):
    """Actual captures are under data/handoffs/feedgy-solar-20261011."""
    query = _query(
        source,
        start_date=start,
        end_date=start,
        variables=variables,
        temporal_resolution=resolution,
        tilt=30 if "global_tilted_irradiance" in variables else None,
        azimuth=0 if "global_tilted_irradiance" in variables else None,
    )
    payload = _captured_provider_payload(query)
    seen: list[httpx.Request] = []

    result = fetch_solar(query, client=_client_for(query, payload, seen))

    assert result["returned_coordinates"]["latitude"] == payload["latitude"]
    assert result["interval_seconds"] == (1800 if resolution == "native" else 3600)
    assert result["timestamps"][0].startswith(start)
    assert result["actual_model"] is None
    assert result["licence"] == "CC BY 4.0"
    assert seen[0].url.params["hourly"] == ",".join(variables)
    if source == "satellite":
        assert seen[0].url.params["temporal_resolution"] == "native"
    assert "minutely_15" not in seen[0].url.params
    # Forecast attribution needs model evidence; ERA5 and SARAH-3 have fixed
    # source contracts with required upstream acknowledgements.
    assert (
        result["attribution"]
        == {
            "forecast": "Open-Meteo",
            "era5": "Open-Meteo; ERA5 from ECMWF/Copernicus Climate Change Service",
            "satellite": "Open-Meteo; EUMETSAT SARAH-3",
        }[source]
    )
    if source == "satellite":
        assert len(result["timestamps"]) == 48
        assert result["values"]["global_tilted_irradiance"][15] == 159.8
    elif source == "forecast":
        assert result["values"]["shortwave_radiation"][14] == 449
    else:
        assert result["values"]["shortwave_radiation"][14] == 836


def test_forecast_request_keeps_hourly_w_m2_and_provenance_without_key_leak():
    query = _query(variables=["shortwave_radiation", "temperature_2m"])
    payload = _payload(query)
    payload["hourly"]["shortwave_radiation"][1] = None
    payload["hourly_units"]["temperature_2m"] = "°C"
    seen: list[httpx.Request] = []
    client = _client_for(query, payload, seen)

    result = fetch_solar(query, client=client)

    request = seen[0]
    assert request.url.host == "api.open-meteo.com"
    assert request.url.params["hourly"] == "shortwave_radiation,temperature_2m"
    assert request.url.params["timezone"] == "UTC"
    assert request.url.params["wind_speed_unit"] == "ms"
    assert request.url.params["models"] == "ecmwf_aifs025_single"
    assert result["schema_version"] == "geoai-solar/v1"
    assert result["product_type"] == "forecast"
    assert result["values"]["shortwave_radiation"][0] == 4.25
    assert result["values"]["shortwave_radiation"][1] is None
    assert result["missing_counts"]["shortwave_radiation"] == 1
    assert result["units"]["shortwave_radiation"] == "W/m²"
    assert result["interval_seconds"] == 3600
    assert result["native_model_interval_seconds"] == 21_600
    assert result["variable_semantics"] == {
        "shortwave_radiation": "mean_over_preceding_interval",
        "temperature_2m": "value_at_timestamp",
    }
    assert result["actual_model"] == "model_returned_by_provider"
    assert result["model_issued_at"] is None
    assert result["retrieved_at"].endswith("Z")
    assert "apikey" not in json.dumps(result)


@pytest.mark.parametrize(
    ("source", "model", "provider_model", "expected_attribution"),
    [
        ("forecast", None, "ecmwf_aifs025_single", "Open-Meteo; ECMWF AIFS forecast"),
        ("forecast", "custom_model", "custom_model", "Open-Meteo"),
        (
            "era5",
            None,
            None,
            "Open-Meteo; ERA5 from ECMWF/Copernicus Climate Change Service",
        ),
        ("satellite", None, None, "Open-Meteo; EUMETSAT SARAH-3"),
    ],
)
def test_attribution_requires_a_confirmed_provider_model(
    source, model, provider_model, expected_attribution
):
    query = _query(source, variables=["shortwave_radiation"], model=model)
    payload = _payload(query)
    if provider_model is None:
        payload.pop("models")
    else:
        payload["models"] = [provider_model]

    result = fetch_solar(query, client=_client_for(query, payload))

    assert (
        result["requested_model"] == model
        or result["requested_model"]
        == {
            "forecast": "ecmwf_aifs025_single",
            "era5": "era5",
            "satellite": "eumetsat_sarah3",
        }[source]
    )
    assert result["actual_model"] == provider_model
    assert result["attribution"] == expected_attribution


def test_setup_logging_httpx_info_redacts_effective_dotenv_solar_key(tmp_path, monkeypatch, capsys):
    key = "solar+key&with/slashes"
    (tmp_path / ".env.solar").write_text(
        "SOLAR_FORECAST_API_BASE_URL=https://customer-api.open-meteo.com/v1/forecast\n"
        f"SOLAR_FORECAST_API_KEY={key}\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("SOLAR_FORECAST_API_BASE_URL", raising=False)
    monkeypatch.delenv("SOLAR_FORECAST_API_KEY", raising=False)
    config = SolarSettings()
    query = _query(variables=["shortwave_radiation"])
    root_logger = logging.getLogger()
    httpx_logger = logging.getLogger("httpx")
    old_handlers = root_logger.handlers[:]
    old_level = root_logger.level
    old_filters = httpx_logger.filters[:]

    try:
        from src.utils.logging import setup_logging

        setup_logging()
        client = _client_for(query, _payload(query))
        fetch_solar(query, config=config, client=client)
        logging.getLogger("solar-test-unrelated").info("unrelated diagnostic remains visible")

        stderr = capsys.readouterr().err
        assert "apikey=[REDACTED]" in stderr
        assert key not in stderr
        assert quote(key, safe="").lower() not in stderr.lower()
        assert "unrelated diagnostic remains visible" in stderr
    finally:
        root_logger.handlers[:] = old_handlers
        root_logger.setLevel(old_level)
        httpx_logger.filters[:] = old_filters


def test_era5_historical_request_uses_archive_model_and_customer_key_only_on_approved_host():
    query = _query(
        "era5", start_date="2001-06-01", end_date="2001-06-01", variables=["shortwave_radiation"]
    )
    payload = _payload(query)
    seen: list[httpx.Request] = []
    client = _client_for(query, payload, seen)
    config = SolarSettings(
        SOLAR_ARCHIVE_API_BASE_URL="https://customer-archive-api.open-meteo.com/v1/archive",
        SOLAR_ARCHIVE_API_KEY="archive-secret",
    )

    result = fetch_solar(query, config=config, client=client)

    request = seen[0]
    assert request.url.path == "/v1/archive"
    assert request.url.params["models"] == "era5"
    assert request.url.params["apikey"] == "archive-secret"
    assert result["product_type"] == "reanalysis"
    assert result["requested_model"] == "era5"
    assert result["native_model_interval_seconds"] == 3600
    assert "archive-secret" not in json.dumps(result)


def test_satellite_native_uses_half_hour_series_and_rejects_other_variables():
    query = _query("satellite", temporal_resolution="native", tilt=0, azimuth=0)
    payload = _payload(query, native=True)
    seen: list[httpx.Request] = []

    result = fetch_solar(query, client=_client_for(query, payload, seen))

    assert seen[0].url.host == "satellite-api.open-meteo.com"
    assert seen[0].url.params["hourly"] == ",".join(SOLAR_VARS)
    assert seen[0].url.params["temporal_resolution"] == "native"
    assert "minutely_15" not in seen[0].url.params
    assert len(result["timestamps"]) == 48
    assert result["interval_seconds"] == 1800
    assert result["native_model_interval_seconds"] == 1800
    assert result["product_type"] == "satellite_estimate"
    with pytest.raises(ValidationError, match="does not provide temperature or wind"):
        _query("satellite", variables=["temperature_2m"])


def test_satellite_hourly_forwards_explicit_resolution():
    query = _query("satellite", temporal_resolution="hourly", variables=["shortwave_radiation"])
    seen: list[httpx.Request] = []

    fetch_solar(query, client=_client_for(query, _payload(query), seen))

    assert seen[0].url.params["hourly"] == "shortwave_radiation"
    assert seen[0].url.params["temporal_resolution"] == "hourly"


@pytest.mark.parametrize(
    "overrides",
    [
        {"latitude": 91},
        {"longitude": 181},
        {"latitude": float("nan")},
        {"start_date": "2025-01-01", "end_date": "2026-01-01"},
        {"start_date": "2025-06-01", "end_date": "2025-06-17"},
    ],
)
def test_query_rejects_invalid_coordinates_and_source_specific_date_ranges(overrides):
    with pytest.raises(ValidationError):
        _query(**overrides)


def test_query_defaults_to_three_orientation_free_irradiance_variables():
    query = _query()
    assert query.variables == [
        "shortwave_radiation",
        "diffuse_radiation",
        "direct_normal_irradiance",
    ]


@pytest.mark.parametrize("value", [20250601, "20250601", "2025-W22-7", "2025-06-01T00:00:00"])
def test_query_dates_require_iso_calendar_date_strings_or_date_objects(value):
    with pytest.raises(ValidationError, match="YYYY-MM-DD|without a time"):
        _query(start_date=value)


def test_forecast_model_must_be_one_bounded_provider_identifier():
    with pytest.raises(ValidationError, match="provider model identifier"):
        _query(model="ecmwf_aifs025_single,era5")


def test_unknown_explicit_forecast_model_has_no_invented_native_cadence():
    query = _query(variables=["shortwave_radiation"], model="vendor_model_v7")

    result = fetch_solar(query, client=_client_for(query, _payload(query)))

    assert result["requested_model"] == "vendor_model_v7"
    assert result["native_model_interval_seconds"] is None


def test_gti_requires_explicit_tilt_and_azimuth_and_uses_south_zero_convention():
    with pytest.raises(ValidationError, match="requires explicit tilt and azimuth"):
        _query(variables=["global_tilted_irradiance"])
    query = _query(variables=["global_tilted_irradiance"], tilt=30, azimuth=0)
    payload = _payload(query)
    seen: list[httpx.Request] = []

    fetch_solar(query, client=_client_for(query, payload, seen))

    assert seen[0].url.params["tilt"] == "30.0"
    assert seen[0].url.params["azimuth"] == "0.0"


@pytest.mark.parametrize(
    ("field", "bad_value"),
    [
        ("length", None),
        ("unit", None),
        ("nonfinite", None),
        ("numeric_string", None),
        ("irregular", None),
        ("timezone", None),
        ("coordinate_range", None),
    ],
)
def test_malformed_provider_series_are_rejected(field, bad_value):
    query = _query(variables=["shortwave_radiation"])
    payload = _payload(query)
    if field == "length":
        payload["hourly"]["shortwave_radiation"].pop()
    elif field == "unit":
        payload["hourly_units"]["shortwave_radiation"] = "Wh/m²"
    elif field == "nonfinite":
        payload["hourly"]["shortwave_radiation"][0] = float("inf")
        payload["_allow_nonfinite_json"] = True
    elif field == "numeric_string":
        payload["hourly"]["shortwave_radiation"][0] = "4.25"
    elif field == "irregular":
        payload["hourly"]["time"][2] = "2025-06-01T03:00:00Z"
    elif field == "timezone":
        payload["timezone"] = "Europe/Paris"
    elif field == "coordinate_range":
        payload["longitude"] = 181

    with pytest.raises(SolarFetchError, match="response is invalid"):
        fetch_solar(query, client=_client_for(query, payload))


@pytest.mark.parametrize(
    "endpoint",
    [
        "http://customer-forecast-api.open-meteo.com/v1/forecast",
        "https://customer-api.open-meteo.com.evil.test/v1/forecast",
        "https://user:pass@customer-api.open-meteo.com/v1/forecast",
        "https://customer-api.open-meteo.com/v1/forecast?redirect=evil",
        "https://evil.test/v1/forecast",
    ],
)
def test_hostile_or_credential_bearing_endpoints_are_rejected_without_request(endpoint):
    query = _query(variables=["shortwave_radiation"])
    config = SolarSettings(
        SOLAR_FORECAST_API_BASE_URL=endpoint,
        SOLAR_FORECAST_API_KEY="secret-value",
    )
    sent = []
    client = _client_for(query, _payload(query), sent)

    with pytest.raises(SolarFetchError, match="endpoint") as caught:
        fetch_solar(query, config=config, client=client)

    assert not sent
    assert "secret-value" not in str(caught.value)
    assert caught.value.__context__ is None


@pytest.mark.parametrize(("status", "retryable"), [(429, True), (503, True), (401, False)])
def test_http_errors_are_safe_and_only_transient_statuses_are_retryable(status, retryable):
    query = _query(variables=["shortwave_radiation"])

    def handler(request):
        return httpx.Response(status, text="contains-no-safe-diagnostics", request=request)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    with pytest.raises(SolarFetchError) as caught:
        fetch_solar(query, client=client)

    assert caught.value.retryable is retryable
    assert str(status) in str(caught.value)
    assert "contains-no-safe-diagnostics" not in str(caught.value)
    assert caught.value.__context__ is None


def test_transport_failure_is_retryable_and_does_not_retain_request_url():
    query = _query(variables=["shortwave_radiation"])

    def handler(request):
        raise httpx.ConnectError(f"failed request {request.url}", request=request)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    with pytest.raises(SolarFetchError) as caught:
        fetch_solar(query, client=client)

    assert caught.value.retryable
    assert "api.open-meteo.com" not in str(caught.value)
    assert caught.value.__context__ is None
