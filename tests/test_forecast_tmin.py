"""Optional live-forecast Tmin contract; every HTTP response is local and synthetic."""

import asyncio

import httpx
import pytest

from src.contracts import ForecastQueryOutput
from src.tools.climate import normalize_archive_response
from src.tools.forecast import register_forecast_tools
from src.utils.forecast import ForecastProviderUnavailable, OpenMeteoForecastProvider
from src.utils.openmeteo import (
    FORECAST_NORMALIZED_VARS,
    NORMALIZED_VARS,
    PROVIDER_VARS,
    fetch_archive_daily,
    forecast_provider_var,
    normalize_forecast_daily,
    normalize_openmeteo_daily,
    provider_var,
)

TMIN = "temperature_2m_min"
DATES = ["2026-10-07", "2026-10-08", "2026-10-09"]
CANONICAL = (
    "temperature_2m",
    "precipitation",
    "humidity",
    "evapotranspiration",
    "wind_speed",
    "solar_radiation",
)


def _payload(*, value=10.0, minimum=None):
    daily = {"time": DATES}
    daily.update({pv.provider: [value] * 3 for pv in PROVIDER_VARS})
    daily[TMIN] = minimum if minimum is not None else [-1.0, None, 0.0]
    return {"daily": daily, "daily_units": {TMIN: "°C"}}


def _install_transport(monkeypatch, primary, fallback=None):
    calls = []

    def respond(request):
        calls.append(dict(request.url.params))
        payload = primary if request.url.params.get("models") else fallback
        assert payload is not None, "unexpected provider fallback"
        return httpx.Response(200, json=payload)

    client = httpx.Client(transport=httpx.MockTransport(respond))
    monkeypatch.setattr("src.utils.openmeteo.get_forecast_client", lambda: client)
    monkeypatch.setattr("src.config.settings.FORECAST_ENABLE_AI", True)
    monkeypatch.setattr("src.config.settings.FORECAST_AI_MODEL", "ecmwf_aifs025_single")
    monkeypatch.setattr("src.config.settings.FORECAST_PHYSICAL_MODEL", "")
    return calls


def _query(**kwargs):
    class Capture:
        def tool(self, **_kwargs):
            def capture(fn):
                self.fn = fn
                return fn

            return capture

    capture = Capture()
    register_forecast_tools(capture)
    return asyncio.run(capture.fn(latitude=43.6, longitude=3.5, forecast_days=3, **kwargs))


def test_canonical_schema_and_archive_defaults_remain_locked():
    assert NORMALIZED_VARS == CANONICAL
    assert tuple(pv.normalized for pv in PROVIDER_VARS) == CANONICAL
    assert FORECAST_NORMALIZED_VARS == CANONICAL + (TMIN,)
    assert forecast_provider_var(TMIN).source_unit == "°C"
    for variable in CANONICAL:
        assert forecast_provider_var(variable) is provider_var(variable)
    raw = _payload()
    assert normalize_forecast_daily(raw, list(CANONICAL)) == normalize_openmeteo_daily(
        raw, list(CANONICAL), strict=True
    )


def test_archive_cannot_request_forecast_only_tmin():
    with pytest.raises(ValueError, match="Unknown climate variable"):
        provider_var(TMIN)
    with pytest.raises(ValueError, match="Unknown climate variable"):
        normalize_archive_response([_payload()], [TMIN])
    # Rejected before creating a client or issuing an archive HTTP request.
    with pytest.raises(ValueError, match="Unknown climate variable"):
        fetch_archive_daily(
            latitude=43.6,
            longitude=3.5,
            start_date=DATES[0],
            end_date=DATES[-1],
            normalized_variables=[TMIN],
        )


def test_forecast_default_still_requests_only_six(monkeypatch):
    calls = _install_transport(monkeypatch, _payload())
    result = _query()
    assert "error" not in result
    assert result["metadata"]["variables"] == list(CANONICAL)
    assert set(result["forecast"][0]) == {"date", *CANONICAL}
    assert calls[0]["daily"] == ",".join(pv.provider for pv in PROVIDER_VARS)


@pytest.mark.parametrize("variables", [[TMIN], [*CANONICAL, TMIN]])
def test_explicit_tmin_uses_ai_without_forcing_physical(monkeypatch, variables):
    calls = _install_transport(monkeypatch, _payload())
    result = _query(variables=variables)
    assert "error" not in result
    assert len(calls) == 1
    assert calls[0]["models"] == "ecmwf_aifs025_single"
    assert calls[0]["temperature_unit"] == "celsius"
    assert calls[0]["timezone"] == "UTC"
    assert calls[0]["daily"] == ",".join(forecast_provider_var(v).provider for v in variables)
    assert result["metadata"]["source"] == "openmeteo_ai"
    assert result["metadata"]["variables"] == variables
    mapping = result["metadata"]["provider_map"][TMIN]
    assert mapping["provider"] == TMIN and mapping["unit"] == "°C"
    assert "full UTC calendar day" in mapping["note"]
    assert [row[TMIN] for row in result["forecast"]] == [-1.0, None, 0.0]
    assert result["statistics"][TMIN] == {"min": -1.0, "max": 0.0, "mean": -0.5, "n": 2}
    assert ForecastQueryOutput.model_validate(result).forecast == result["forecast"]


@pytest.mark.parametrize("failure", ["missing", "null", "ragged", "fahrenheit", "unit_missing"])
def test_tmin_outage_falls_back_with_whole_forecast_provenance(monkeypatch, failure):
    primary = _payload(value=99.0)
    if failure == "missing":
        del primary["daily"][TMIN]
    elif failure == "null":
        primary["daily"][TMIN] = [None] * 3
    elif failure == "ragged":
        primary["daily"][TMIN] = [-1.0]
    elif failure == "fahrenheit":
        primary["daily_units"][TMIN] = "°F"
    else:
        del primary["daily_units"][TMIN]
    calls = _install_transport(monkeypatch, primary, _payload(value=20.0))
    result = _query(variables=[*CANONICAL, TMIN])
    assert "error" not in result
    assert len(calls) == 2
    assert result["metadata"]["source"] == "openmeteo_physical"
    assert [p["status"] for p in result["metadata"]["provider_chain"]] == ["outage", "ok"]
    # Even the otherwise-valid mean temperature comes from the fallback, not mixed providers.
    assert result["forecast"][0]["temperature_2m"] == 20.0
    assert result["forecast"][0][TMIN] == -1.0


def test_all_down_tmin_is_error_not_no_frost(monkeypatch):
    raw = _payload(minimum=[None] * 3)
    _install_transport(monkeypatch, raw, raw)
    result = _query(variables=[TMIN])
    assert result["error"] == "all forecast providers unavailable"
    assert "forecast" not in result
    assert [p["status"] for p in result["metadata"]["provider_chain"]] == ["outage", "outage"]


@pytest.mark.parametrize("unit", ["°F", None])
def test_provider_rejects_unknown_tmin_units(monkeypatch, unit):
    raw = _payload()
    raw["daily_units"][TMIN] = unit
    _install_transport(monkeypatch, raw)
    provider = OpenMeteoForecastProvider(name="ai", models="ecmwf_aifs025_single")
    with pytest.raises(ForecastProviderUnavailable, match="unit must be"):
        provider.fetch_daily(latitude=43.6, longitude=3.5, variables=[TMIN], forecast_days=3)
