"""Tests for FIERCE B5 forecast fallback (network-free + a live contract probe).

Layers:
1. **Normalizer** — ``normalize_openmeteo_daily`` strict vs lenient (the moved B2 fail-loud +
   the new forecast-path absence-tolerance), plus a regression that Archive still fails loud.
2. **§4.5 usability helpers** — trailing-null trimming + the missing/all-null outage rule (pure).
3. **OpenMeteoForecastProvider** — per-provider horizon cap + outage on a non-usable response.
4. **ForecastRouter** — the DoD star: happy path, outage→fallback (raise + timeout), incomplete
   result → outage, and all-down → the router RAISES with the chain (Codex #294).
5. **fetch_forecast_daily** — request params (models omitted when empty / present when set; apikey
   gated; wind_speed_unit/timezone), 4xx-not-retried / 5xx-retried.
6. **forecast_query handler** — zone & point paths, zone-XOR-point, dedup + unknown var, horizon
   clamp, point-mode write requires output_path, zone-mode WKB round-trip, all-down → {"error"};
   plus the real default-chain test (provider="auto", only HTTP faked — Greptile #303).
7. **nvidia_nim** — unconfigured seam raises (router skips it).
8. **Live probe** (skipped offline) — physical six-var availability + the current AIFS status.
"""

from __future__ import annotations

import asyncio
import json
import os

import httpx
import pytest

from src.utils.duckdb_conn import get_duckdb_connection
from src.utils.forecast import (
    ForecastProviderUnavailable,
    ForecastRouter,
    OpenMeteoForecastProvider,
    _is_usable,
    _trim_trailing_null_days,
    build_router,
)
from src.utils.openmeteo import NORMALIZED_VARS, normalize_openmeteo_daily, provider_var

_RUN_LIVE = os.environ.get("OPENMETEO_LIVE") == "1"
pytestmark = pytest.mark.usefixtures("allow_local_parquet_paths")


# --------------------------------------------------------------------------- #
# Fake httpx layer (monkeypatched into get_forecast_client)
# --------------------------------------------------------------------------- #


class _FakeResponse:
    def __init__(
        self, payload: dict | None = None, status: int = 200, json_exc: Exception | None = None
    ) -> None:
        self._payload = payload or {}
        self.status_code = status
        self._json_exc = json_exc  # set to simulate a non-JSON 200 body (resp.json() raises)

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            req = httpx.Request("GET", "https://forecast.test")
            resp = httpx.Response(self.status_code, request=req)
            raise httpx.HTTPStatusError(f"HTTP {self.status_code}", request=req, response=resp)

    def json(self) -> dict:
        if self._json_exc is not None:
            raise self._json_exc
        return self._payload


class _FakeClient:
    """Records every .get(); a responder(params, attempt) returns a _FakeResponse or an exception
    instance (raised to simulate a transport error before any response)."""

    def __init__(self, responder) -> None:
        self._responder = responder
        self.calls: list[dict] = []

    def get(self, url, params=None, timeout=None):
        self.calls.append({"url": url, "params": params or {}, "timeout": timeout})
        out = self._responder(params or {}, len(self.calls))
        if isinstance(out, BaseException):
            raise out
        return out


def _install_client(monkeypatch, responder) -> _FakeClient:
    client = _FakeClient(responder)
    monkeypatch.setattr("src.utils.openmeteo.get_forecast_client", lambda: client)
    monkeypatch.setattr("src.utils.openmeteo.time.sleep", lambda *_a, **_k: None)  # no real backoff
    return client


def _daily_payload(dates: list[str], *, omit=(), all_null=(), trailing_null: int = 0) -> dict:
    """Build an Open-Meteo daily response over ``dates`` for the six §5.1 provider vars.

    ``omit`` drops a variable entirely; ``all_null`` returns it as all-None; ``trailing_null`` sets
    the last N days of EVERY variable to None (the live 16th-day all-null tail)."""
    n = len(dates)
    daily: dict = {"time": list(dates)}
    for v in NORMALIZED_VARS:
        pname = provider_var(v).provider
        if v in omit:
            continue
        if v in all_null:
            daily[pname] = [None] * n
            continue
        series: list[float | None] = [float(i + 1) for i in range(n)]
        for k in range(trailing_null):
            if n - 1 - k >= 0:
                series[n - 1 - k] = None
        daily[pname] = series
    return {"daily": daily}


_DATES3 = ["2026-06-21", "2026-06-22", "2026-06-23"]


# --------------------------------------------------------------------------- #
# 1. Normalizer (strict vs lenient)
# --------------------------------------------------------------------------- #


class TestNormalizeOpenMeteoDaily:
    def test_lenient_missing_variable_is_absent_not_raised(self):
        # forecast path: a 5-of-6 response → the 6th var is an all-None column (length == dates),
        # never raised, so §4.5 marks it an outage downstream.
        raw = _daily_payload(_DATES3, omit=("humidity",))
        dates, columns = normalize_openmeteo_daily(raw, list(NORMALIZED_VARS), strict=False)
        assert dates == _DATES3
        assert columns["humidity"] == [None, None, None]
        assert columns["temperature_2m"] == [1.0, 2.0, 3.0]

    def test_strict_missing_variable_raises(self):
        # Archive ingestion fail-loud (the moved B2 behaviour) — a missing var is an upstream rename.
        raw = _daily_payload(_DATES3, omit=("humidity",))
        with pytest.raises(KeyError):
            normalize_openmeteo_daily(raw, list(NORMALIZED_VARS), strict=True)

    def test_solar_radiation_converted(self):
        raw = {"daily": {"time": ["2026-06-21"], "shortwave_radiation_sum": [5.0]}}
        _dates, columns = normalize_openmeteo_daily(raw, ["solar_radiation"], strict=False)
        assert columns["solar_radiation"][0] == pytest.approx(57.870, abs=1e-3)  # MJ/m² → W/m²

    def test_archive_normalizer_still_fails_loud_after_refactor(self):
        # regression: climate.normalize_archive_response (now delegating to strict=True) must keep
        # raising KeyError on a renamed/missing var — the B2 contract the live probe relies on.
        from src.tools.climate import normalize_archive_response

        broken = {"daily": {"time": ["2026-06-21"]}}  # no provider series
        with pytest.raises(KeyError):
            normalize_archive_response([broken], ["temperature_2m"])


# --------------------------------------------------------------------------- #
# 2. §4.5 usability helpers (pure)
# --------------------------------------------------------------------------- #


class TestUsabilityHelpers:
    def test_trim_trailing_all_null_days(self):
        dates = list(_DATES3)
        columns = {"temperature_2m": [1.0, 2.0, None], "precipitation": [0.1, 0.2, None]}
        td, tc = _trim_trailing_null_days(dates, columns)
        assert td == ["2026-06-21", "2026-06-22"]
        assert tc["temperature_2m"] == [1.0, 2.0]

    def test_trim_keeps_interior_null_day(self):
        # an interior all-null day is a genuine gap, not a trailing tail → kept
        dates = list(_DATES3)
        columns = {"temperature_2m": [1.0, None, 3.0]}
        td, tc = _trim_trailing_null_days(dates, columns)
        assert td == _DATES3 and tc["temperature_2m"] == [1.0, None, 3.0]

    def test_usable_true_when_every_var_has_a_value(self):
        columns = {v: [1.0, None, 3.0] for v in NORMALIZED_VARS}
        assert _is_usable(_DATES3, columns, list(NORMALIZED_VARS))

    def test_unusable_when_a_var_is_all_null(self):
        columns = {v: [1.0, 2.0, 3.0] for v in NORMALIZED_VARS}
        columns["humidity"] = [None, None, None]
        assert not _is_usable(_DATES3, columns, list(NORMALIZED_VARS))

    def test_unusable_when_column_is_ragged(self):
        columns = {v: [1.0, 2.0, 3.0] for v in NORMALIZED_VARS}
        columns["humidity"] = [1.0, 2.0]
        # Trim must not index into the short column; usability rejects the ragged payload.
        td, tc = _trim_trailing_null_days(_DATES3, columns)
        assert td == _DATES3 and tc is columns
        assert not _is_usable(td, tc, list(NORMALIZED_VARS))

    def test_unusable_when_no_days(self):
        assert not _is_usable([], {v: [] for v in NORMALIZED_VARS}, list(NORMALIZED_VARS))


# --------------------------------------------------------------------------- #
# 3. OpenMeteoForecastProvider
# --------------------------------------------------------------------------- #


class TestOpenMeteoProvider:
    def test_usable_response_returns_normalized(self, monkeypatch):
        _install_client(monkeypatch, lambda _p, _a: _FakeResponse(_daily_payload(_DATES3)))
        provider = OpenMeteoForecastProvider(name="physical", models="", max_days=16)
        result = provider.fetch_daily(
            latitude=43.6, longitude=3.5, variables=list(NORMALIZED_VARS), forecast_days=16
        )
        assert result["dates"] == _DATES3
        assert result["columns"]["temperature_2m"] == [1.0, 2.0, 3.0]

    def test_trailing_null_day_trimmed(self, monkeypatch):
        _install_client(
            monkeypatch, lambda _p, _a: _FakeResponse(_daily_payload(_DATES3, trailing_null=1))
        )
        provider = OpenMeteoForecastProvider(name="physical", max_days=16)
        result = provider.fetch_daily(
            latitude=43.6, longitude=3.5, variables=list(NORMALIZED_VARS), forecast_days=16
        )
        assert result["dates"] == ["2026-06-21", "2026-06-22"]  # trailing all-null trimmed

    def test_missing_variable_is_outage(self, monkeypatch):
        _install_client(
            monkeypatch, lambda _p, _a: _FakeResponse(_daily_payload(_DATES3, omit=("humidity",)))
        )
        provider = OpenMeteoForecastProvider(name="physical", max_days=16)
        with pytest.raises(ForecastProviderUnavailable):
            provider.fetch_daily(
                latitude=43.6, longitude=3.5, variables=list(NORMALIZED_VARS), forecast_days=16
            )

    def test_all_null_response_is_outage(self, monkeypatch):
        # the live AIFS failure mode (2026-06-21): valid schema, all-null values → outage
        _install_client(
            monkeypatch,
            lambda _p, _a: _FakeResponse(_daily_payload(_DATES3, all_null=NORMALIZED_VARS)),
        )
        provider = OpenMeteoForecastProvider(name="ai", models="ecmwf_aifs025", max_days=15)
        with pytest.raises(ForecastProviderUnavailable):
            provider.fetch_daily(
                latitude=43.6, longitude=3.5, variables=list(NORMALIZED_VARS), forecast_days=15
            )

    def test_ragged_response_is_outage(self, monkeypatch):
        payload = _daily_payload(_DATES3)
        payload["daily"][provider_var("humidity").provider] = [1.0, 2.0]
        _install_client(monkeypatch, lambda _p, _a: _FakeResponse(payload))
        provider = OpenMeteoForecastProvider(name="physical", max_days=16)
        with pytest.raises(ForecastProviderUnavailable):
            provider.fetch_daily(
                latitude=43.6, longitude=3.5, variables=list(NORMALIZED_VARS), forecast_days=16
            )

    def test_per_provider_horizon_cap(self, monkeypatch):
        # Codex #128 — a 15-day provider requests min(forecast_days, 15), never the 16-day over-ask
        client = _install_client(monkeypatch, lambda _p, _a: _FakeResponse(_daily_payload(_DATES3)))
        provider = OpenMeteoForecastProvider(name="ai", models="ecmwf_aifs025", max_days=15)
        provider.fetch_daily(
            latitude=43.6, longitude=3.5, variables=list(NORMALIZED_VARS), forecast_days=16
        )
        assert client.calls[0]["params"]["forecast_days"] == 15

    def test_non_json_body_is_outage(self, monkeypatch):
        # a captive-portal/proxy HTML-200 → resp.json() raises JSONDecodeError (a ValueError) →
        # the provider converts it to a clean outage so the router can fall back (not a crash).
        _install_client(
            monkeypatch,
            lambda _p, _a: _FakeResponse(
                json_exc=json.JSONDecodeError("Expecting value", "<html>", 0)
            ),
        )
        provider = OpenMeteoForecastProvider(name="physical", max_days=16)
        with pytest.raises(ForecastProviderUnavailable):
            provider.fetch_daily(
                latitude=43.6, longitude=3.5, variables=list(NORMALIZED_VARS), forecast_days=16
            )

    def test_ragged_series_is_outage(self, monkeypatch):
        # a present-but-short daily.<var> series (shorter than daily.time) → IndexError in the trim →
        # converted to an outage rather than escaping the router.
        ragged = {"daily": {"time": _DATES3, "temperature_2m_mean": [1.0]}}  # 1 value vs 3 days
        _install_client(monkeypatch, lambda _p, _a: _FakeResponse(ragged))
        provider = OpenMeteoForecastProvider(name="physical", max_days=16)
        with pytest.raises(ForecastProviderUnavailable):
            provider.fetch_daily(
                latitude=43.6, longitude=3.5, variables=list(NORMALIZED_VARS), forecast_days=16
            )

    def test_require_models_empty_token_is_outage(self, monkeypatch):
        # the AI slot with an empty token must NOT silently serve the physical best-match (finding #1)
        _install_client(monkeypatch, lambda _p, _a: _FakeResponse(_daily_payload(_DATES3)))
        provider = OpenMeteoForecastProvider(name="openmeteo_ai", models="", require_models=True)
        with pytest.raises(ForecastProviderUnavailable):
            provider.fetch_daily(
                latitude=43.6, longitude=3.5, variables=list(NORMALIZED_VARS), forecast_days=16
            )

    def test_provider_uses_settings_api_key(self, monkeypatch):
        # The runtime providers are built from settings, so the commercial tier key must be
        # propagated beyond the low-level fetch helper.
        monkeypatch.setattr("src.config.settings.FORECAST_API_KEY", "forecast-secret")
        client = _install_client(monkeypatch, lambda _p, _a: _FakeResponse(_daily_payload(_DATES3)))
        provider = OpenMeteoForecastProvider(name="physical", max_days=16)
        provider.fetch_daily(
            latitude=43.6, longitude=3.5, variables=list(NORMALIZED_VARS), forecast_days=3
        )
        assert client.calls[0]["params"]["apikey"] == "forecast-secret"


# --------------------------------------------------------------------------- #
# 4. ForecastRouter — the DoD star
# --------------------------------------------------------------------------- #


class _FakeProvider:
    def __init__(self, name, *, models="", result=None, exc=None) -> None:
        self.name = name
        self.models = models
        self.max_days = 16
        self._result = result
        self._exc = exc
        self.called = False

    def fetch_daily(self, *, latitude, longitude, variables, forecast_days):
        self.called = True
        if self._exc is not None:
            raise self._exc
        return dict(self._result)


def _ok_result():
    return {"dates": _DATES3, "columns": {v: [1.0, 2.0, 3.0] for v in NORMALIZED_VARS}}


class TestForecastRouter:
    def test_happy_path_first_provider_serves_rest_not_called(self):
        p1 = _FakeProvider("p1", result=_ok_result())
        p2 = _FakeProvider("p2", result=_ok_result())
        result, chain = ForecastRouter([p1, p2]).fetch_daily(
            latitude=43.6, longitude=3.5, variables=list(NORMALIZED_VARS), forecast_days=16
        )
        assert result["source"] == "p1"
        assert p2.called is False
        assert [c["status"] for c in chain] == ["ok"]

    @pytest.mark.parametrize(
        "primary_exc",
        [
            ForecastProviderUnavailable("down"),
            httpx.TimeoutException("timeout"),
            RuntimeError("bad provider payload"),
        ],
    )
    def test_outage_falls_back_with_no_error(self, primary_exc):
        primary = _FakeProvider("primary", models="ecmwf_aifs025", exc=primary_exc)
        fallback = _FakeProvider("fallback", result=_ok_result())
        result, chain = ForecastRouter([primary, fallback]).fetch_daily(
            latitude=43.6, longitude=3.5, variables=list(NORMALIZED_VARS), forecast_days=16
        )
        assert result["source"] == "fallback"
        assert chain[0]["provider"] == "primary" and chain[0]["status"] == "outage"
        assert "reason" in chain[0]
        assert chain[1]["status"] == "ok"

    def test_all_providers_down_raises_with_chain(self):
        # Codex #294 — the router RAISES (carrying the chain); the {"error"} translation is the
        # handler's job, so the (result, chain) tuple contract stays unambiguous.
        p1 = _FakeProvider("p1", exc=ForecastProviderUnavailable("down1"))
        p2 = _FakeProvider("p2", exc=httpx.ConnectError("down2"))
        with pytest.raises(ForecastProviderUnavailable) as ei:
            ForecastRouter([p1, p2]).fetch_daily(
                latitude=43.6, longitude=3.5, variables=list(NORMALIZED_VARS), forecast_days=16
            )
        chain = ei.value.provider_chain
        assert [c["provider"] for c in chain] == ["p1", "p2"]
        assert all(c["status"] == "outage" for c in chain)

    def test_router_rejects_nonusable_returned_result(self):
        # Codex P2 — a provider that RETURNS (rather than raises) a non-usable result (here a ragged
        # column, len 1 vs 3 days) must be treated as an outage at the router boundary and skipped,
        # not served (which would IndexError later in the handler).
        bad = _FakeProvider(
            "bad", result={"dates": _DATES3, "columns": {v: [1.0] for v in NORMALIZED_VARS}}
        )
        good = _FakeProvider("good", result=_ok_result())
        result, chain = ForecastRouter([bad, good]).fetch_daily(
            latitude=43.6, longitude=3.5, variables=list(NORMALIZED_VARS), forecast_days=16
        )
        assert result["source"] == "good"
        assert chain[0]["provider"] == "bad" and chain[0]["status"] == "outage"
        assert chain[1]["status"] == "ok"


# --------------------------------------------------------------------------- #
# 5. fetch_forecast_daily — request params + retry policy
# --------------------------------------------------------------------------- #


class TestFetchForecastDaily:
    def _fetch(self, monkeypatch, responder, **kwargs):
        from src.utils.openmeteo import fetch_forecast_daily

        client = _install_client(monkeypatch, responder)
        result = fetch_forecast_daily(
            latitude=43.6,
            longitude=3.5,
            normalized_variables=list(NORMALIZED_VARS),
            forecast_days=16,
            **kwargs,
        )
        return client, result

    def test_models_omitted_when_empty(self, monkeypatch):
        client, _ = self._fetch(
            monkeypatch, lambda _p, _a: _FakeResponse(_daily_payload(_DATES3)), models=""
        )
        params = client.calls[0]["params"]
        assert "models" not in params  # best-match = OMIT the param (never "auto")
        assert params["wind_speed_unit"] == "ms" and params["timezone"] == "UTC"
        assert params["forecast_days"] == 16
        assert params["daily"] == ",".join(provider_var(v).provider for v in NORMALIZED_VARS)

    def test_models_present_when_set(self, monkeypatch):
        client, _ = self._fetch(
            monkeypatch,
            lambda _p, _a: _FakeResponse(_daily_payload(_DATES3)),
            models="ecmwf_aifs025",
        )
        assert client.calls[0]["params"]["models"] == "ecmwf_aifs025"

    def test_apikey_gated(self, monkeypatch):
        client, _ = self._fetch(
            monkeypatch, lambda _p, _a: _FakeResponse(_daily_payload(_DATES3)), apikey=""
        )
        assert "apikey" not in client.calls[0]["params"]
        client2, _ = self._fetch(
            monkeypatch, lambda _p, _a: _FakeResponse(_daily_payload(_DATES3)), apikey="secret"
        )
        assert client2.calls[0]["params"]["apikey"] == "secret"

    def test_base_url_override(self, monkeypatch):
        client, _ = self._fetch(
            monkeypatch,
            lambda _p, _a: _FakeResponse(_daily_payload(_DATES3)),
            base_url="https://customer-api.open-meteo.com/v1/forecast",
        )
        assert client.calls[0]["url"] == "https://customer-api.open-meteo.com/v1/forecast"

    def test_4xx_not_retried(self, monkeypatch):
        # a 4xx is a bad request — it won't fix itself, so exactly one attempt (no retry)
        c = _install_client(monkeypatch, lambda _p, _a: _FakeResponse(status=400))
        from src.utils.openmeteo import fetch_forecast_daily

        with pytest.raises(httpx.HTTPStatusError):
            fetch_forecast_daily(
                latitude=0, longitude=0, normalized_variables=["temperature_2m"], forecast_days=3
            )
        assert len(c.calls) == 1

    def test_5xx_retried_then_succeeds(self, monkeypatch):
        def _responder(_params, attempt):
            if attempt < 3:
                return _FakeResponse(status=500)
            return _FakeResponse(_daily_payload(_DATES3))

        c = _install_client(monkeypatch, _responder)
        from src.utils.openmeteo import fetch_forecast_daily

        result = fetch_forecast_daily(
            latitude=0, longitude=0, normalized_variables=["temperature_2m"], forecast_days=3
        )
        assert "daily" in result and len(c.calls) == 3  # retried twice, served on the 3rd

    def test_transport_error_retried_then_raises(self, monkeypatch):
        c = _install_client(monkeypatch, lambda _p, _a: httpx.ConnectError("boom"))
        from src.utils.openmeteo import fetch_forecast_daily

        with pytest.raises(httpx.ConnectError):
            fetch_forecast_daily(
                latitude=0, longitude=0, normalized_variables=["temperature_2m"], forecast_days=3
            )
        assert len(c.calls) == 3  # exhausted the retry budget


# --------------------------------------------------------------------------- #
# 6. forecast_query handler
# --------------------------------------------------------------------------- #


class _CaptureMCP:
    def __init__(self) -> None:
        self.tools: dict[str, object] = {}

    def tool(self, *_a, **_k):
        def _decorator(fn):
            self.tools[fn.__name__] = fn
            return fn

        return _decorator


def _forecast_tool():
    from src.tools.forecast import register_forecast_tools

    cap = _CaptureMCP()
    register_forecast_tools(cap)
    return cap.tools["forecast_query"]


def _patch_router(monkeypatch, provider):
    monkeypatch.setattr(
        "src.tools.forecast.build_router", lambda _p="auto": ForecastRouter([provider])
    )


class TestForecastQueryHandler:
    def test_zone_mode_returns_forecast(self, monkeypatch):
        _patch_router(monkeypatch, _FakeProvider("openmeteo_physical", result=_ok_result()))
        result = asyncio.run(_forecast_tool()(zone_id="montpellier"))
        assert "error" not in result, result
        assert result["zone_id"] == "montpellier"
        assert len(result["forecast"]) == 3
        assert result["metadata"]["source"] == "openmeteo_physical"
        assert result["metadata"]["horizon_days"] == 3
        assert result["metadata"]["license"] == "CC-BY 4.0 (Open-Meteo)"
        assert "non-commercial" not in result["metadata"]["license"]
        assert "temperature_2m" in result["statistics"]

    def test_point_mode_returns_forecast(self, monkeypatch):
        _patch_router(monkeypatch, _FakeProvider("openmeteo_physical", result=_ok_result()))
        result = asyncio.run(_forecast_tool()(latitude=43.6, longitude=3.5))
        assert "error" not in result, result
        assert result["point"] == {"latitude": 43.6, "longitude": 3.5}
        assert "zone_id" not in result

    def test_zone_xor_point_both_rejected(self):
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            asyncio.run(_forecast_tool()(zone_id="montpellier", latitude=1.0, longitude=2.0))

    def test_zone_xor_point_neither_rejected(self):
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            asyncio.run(_forecast_tool()())

    def test_half_point_rejected(self):
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            asyncio.run(_forecast_tool()(latitude=43.6))  # longitude missing

    def test_unknown_zone_graceful_error(self, monkeypatch):
        _patch_router(monkeypatch, _FakeProvider("p", result=_ok_result()))
        result = asyncio.run(_forecast_tool()(zone_id="atlantis"))
        assert "error" in result and "atlantis" in result["error"]

    def test_unknown_provider_graceful_error(self):
        result = asyncio.run(_forecast_tool()(zone_id="montpellier", provider="bogus"))
        assert "error" in result and "Unknown provider" in result["error"]

    def test_unknown_variable_graceful_error(self):
        result = asyncio.run(_forecast_tool()(zone_id="montpellier", variables=["bogus"]))
        assert "error" in result and "Unknown variables" in result["error"]

    def test_empty_variables_list_rejected(self):
        # an EXPLICIT empty list is malformed — must error, not silently widen to all six
        result = asyncio.run(_forecast_tool()(zone_id="montpellier", variables=[]))
        assert "error" in result and "variables must not be empty" in result["error"]

    def test_write_failure_graceful_error(self, tmp_path, monkeypatch):
        _patch_router(monkeypatch, _FakeProvider("openmeteo_physical", result=_ok_result()))
        # a regex-valid but unwritable destination: a path component that is actually a file
        afile = tmp_path / "afile"
        afile.write_text("not a dir")
        dst = (afile / "out.parquet").as_posix()
        result = asyncio.run(
            _forecast_tool()(zone_id="montpellier", write_output=True, output_path=dst)
        )
        assert "error" in result and "GeoParquet write failed" in result["error"]

    def test_duplicate_variables_deduped(self, monkeypatch):
        _patch_router(monkeypatch, _FakeProvider("p", result=_ok_result()))
        result = asyncio.run(
            _forecast_tool()(zone_id="montpellier", variables=["temperature_2m", "temperature_2m"])
        )
        assert result["metadata"]["variables"] == ["temperature_2m"]

    def test_horizon_clamped(self, monkeypatch):
        provider = _FakeProvider("p", result=_ok_result())
        captured = {}
        orig = provider.fetch_daily

        def _spy(*, latitude, longitude, variables, forecast_days):
            captured["forecast_days"] = forecast_days
            return orig(
                latitude=latitude,
                longitude=longitude,
                variables=variables,
                forecast_days=forecast_days,
            )

        provider.fetch_daily = _spy
        _patch_router(monkeypatch, provider)
        asyncio.run(_forecast_tool()(zone_id="montpellier", forecast_days=999))
        assert captured["forecast_days"] == 16  # clamped to settings.FORECAST_MAX_DAYS

    def test_point_mode_write_requires_output_path(self, monkeypatch):
        _patch_router(monkeypatch, _FakeProvider("p", result=_ok_result()))
        result = asyncio.run(_forecast_tool()(latitude=43.6, longitude=3.5, write_output=True))
        assert "error" in result and "requires an explicit output_path" in result["error"]

    def test_zone_mode_write_round_trip(self, tmp_path, monkeypatch):
        _patch_router(monkeypatch, _FakeProvider("openmeteo_physical", result=_ok_result()))
        dst = (tmp_path / "forecast.parquet").as_posix()
        result = asyncio.run(
            _forecast_tool()(zone_id="montpellier", write_output=True, output_path=dst)
        )
        assert "error" not in result, result
        assert result["output"] == dst
        con = get_duckdb_connection(":memory:")
        n = con.execute(f"SELECT count(*) FROM read_parquet('{dst}')").fetchone()[0]
        cols = {
            d[0] for d in con.execute(f"SELECT * FROM read_parquet('{dst}') LIMIT 0").description
        }
        wkt = con.execute(
            f"SELECT ST_AsText(ST_GeomFromWKB(geometry)) FROM read_parquet('{dst}') LIMIT 1"
        ).fetchone()[0]
        con.close()
        assert n == 3
        assert {"date", "zone_id", "lat", "lon", "geometry", "temperature_2m"} <= cols
        assert wkt == "POINT (3.5 43.6)"

    def test_unsafe_output_path_rejected(self, monkeypatch):
        _patch_router(monkeypatch, _FakeProvider("p", result=_ok_result()))
        result = asyncio.run(
            _forecast_tool()(
                zone_id="montpellier",
                write_output=True,
                output_path="s3://other-bucket/secret.parquet",
            )
        )
        assert "error" in result and "Invalid output_path" in result["error"]

    def test_real_default_chain_ai_outage_physical_serves(self, monkeypatch):
        # Greptile #303 — drive provider="auto" with the REAL provider construction, faking only
        # HTTP: the AI model (models set) returns all-null → outage; the physical (models omitted)
        # serves. Asserts the handler returns the physical forecast + the chain records the AI outage.
        monkeypatch.setattr("src.config.settings.FORECAST_ENABLE_AI", True)

        def _responder(params, _attempt):
            if params.get("models"):  # the AI provider → all-null (the live AIFS failure mode)
                return _FakeResponse(_daily_payload(_DATES3, all_null=NORMALIZED_VARS))
            return _FakeResponse(_daily_payload(_DATES3))  # the physical best-match → real data

        _install_client(monkeypatch, _responder)
        result = asyncio.run(_forecast_tool()(zone_id="montpellier", provider="auto"))
        assert "error" not in result, result
        assert result["metadata"]["source"] == "openmeteo_physical"
        chain = result["metadata"]["provider_chain"]
        assert chain[0]["provider"] == "openmeteo_ai" and chain[0]["status"] == "outage"
        assert chain[1]["provider"] == "openmeteo_physical" and chain[1]["status"] == "ok"

    def test_real_chain_all_down_returns_error(self, monkeypatch):
        # both providers all-null → the router raises → the handler maps it to {"error"} + chain.
        monkeypatch.setattr("src.config.settings.FORECAST_ENABLE_AI", True)
        _install_client(
            monkeypatch,
            lambda _p, _a: _FakeResponse(_daily_payload(_DATES3, all_null=NORMALIZED_VARS)),
        )
        result = asyncio.run(_forecast_tool()(zone_id="montpellier", provider="auto"))
        assert "error" in result and "all forecast providers unavailable" in result["error"]
        assert len(result["metadata"]["provider_chain"]) == 2

    def test_default_chain_ai_off_serves_physical_only(self, monkeypatch):
        # the PRODUCTION default (FORECAST_ENABLE_AI=False) → a length-1 chain, physical only,
        # through the REAL build_router (no monkeypatched router). A regression that ignored the
        # flag (always appending the live-null AI provider) would change the chain length here.
        monkeypatch.setattr("src.config.settings.FORECAST_ENABLE_AI", False)
        _install_client(monkeypatch, lambda _p, _a: _FakeResponse(_daily_payload(_DATES3)))
        result = asyncio.run(_forecast_tool()(zone_id="montpellier", provider="auto"))
        assert "error" not in result, result
        assert result["metadata"]["source"] == "openmeteo_physical"
        chain = result["metadata"]["provider_chain"]
        assert len(chain) == 1 and chain[0]["provider"] == "openmeteo_physical"
        assert "issued_at" in result["metadata"]  # the snapshot timestamp / dated-filename source

    def test_empty_ai_token_auto_falls_back_to_physical(self, monkeypatch):
        # finding #1: AI enabled but FORECAST_AI_MODEL="" must NOT silently serve the physical
        # best-match mislabeled as openmeteo_ai — the AI slot is a clean outage, physical serves.
        monkeypatch.setattr("src.config.settings.FORECAST_ENABLE_AI", True)
        monkeypatch.setattr("src.config.settings.FORECAST_AI_MODEL", "")
        _install_client(monkeypatch, lambda _p, _a: _FakeResponse(_daily_payload(_DATES3)))
        result = asyncio.run(_forecast_tool()(zone_id="montpellier", provider="auto"))
        assert "error" not in result, result
        assert result["metadata"]["source"] == "openmeteo_physical"
        chain = result["metadata"]["provider_chain"]
        assert chain[0]["provider"] == "openmeteo_ai" and chain[0]["status"] == "outage"

    def test_empty_ai_token_ai_pin_errors(self, monkeypatch):
        # the explicit provider="ai" pin with an empty token returns a graceful all-down {"error"},
        # never mislabeled physical data.
        monkeypatch.setattr("src.config.settings.FORECAST_AI_MODEL", "")
        _install_client(monkeypatch, lambda _p, _a: _FakeResponse(_daily_payload(_DATES3)))
        result = asyncio.run(_forecast_tool()(zone_id="montpellier", provider="ai"))
        assert "error" in result and "all forecast providers unavailable" in result["error"]

    def test_malformed_primary_falls_back(self, monkeypatch):
        # finding #2 (integration): a malformed AI response (ragged series) is an outage → the
        # physical provider serves; the chain records the AI outage, no error surfaces.
        monkeypatch.setattr("src.config.settings.FORECAST_ENABLE_AI", True)

        def _responder(params, _attempt):
            if params.get("models"):  # AI → ragged series (IndexError in the trim → outage)
                return _FakeResponse({"daily": {"time": _DATES3, "temperature_2m_mean": [1.0]}})
            return _FakeResponse(_daily_payload(_DATES3))  # physical → good data

        _install_client(monkeypatch, _responder)
        result = asyncio.run(_forecast_tool()(zone_id="montpellier", provider="auto"))
        assert "error" not in result, result
        assert result["metadata"]["source"] == "openmeteo_physical"
        assert result["metadata"]["provider_chain"][0]["status"] == "outage"

    def test_no_destination_graceful_error(self, monkeypatch):
        # finding #3: zone-mode write_output with no output_path and no S3_BUCKET → graceful error
        # (exercises the in-handler dated-URI build returning None → the "No output destination" path)
        monkeypatch.setattr("src.config.settings.S3_BUCKET", "")
        _patch_router(monkeypatch, _FakeProvider("openmeteo_physical", result=_ok_result()))
        result = asyncio.run(_forecast_tool()(zone_id="montpellier", write_output=True))
        assert "error" in result and "No output destination" in result["error"]


# --------------------------------------------------------------------------- #
# 7. NVIDIA NIM gated seam
# --------------------------------------------------------------------------- #


class TestNvidiaNimSeam:
    def test_unconfigured_provider_raises(self, monkeypatch):
        monkeypatch.setattr("src.config.settings.NVIDIA_NIM_FORECAST_URL", "")
        from src.utils.nvidia_nim import NvidiaNimForecastProvider

        with pytest.raises(ForecastProviderUnavailable):
            NvidiaNimForecastProvider().fetch_daily(
                latitude=0, longitude=0, variables=["temperature_2m"], forecast_days=10
            )

    def test_nim_router_skips_to_unavailable(self, monkeypatch):
        # build_router("nim") with the seam unconfigured → the router has only the gated provider,
        # so fetch_daily raises ForecastProviderUnavailable (all — i.e. the one — providers down).
        monkeypatch.setattr("src.config.settings.NVIDIA_NIM_FORECAST_URL", "")
        with pytest.raises(ForecastProviderUnavailable):
            build_router("nim").fetch_daily(
                latitude=0, longitude=0, variables=["temperature_2m"], forecast_days=10
            )


# --------------------------------------------------------------------------- #
# 7b. build_router pins + the dated-URI builder
# --------------------------------------------------------------------------- #


class TestBuildRouterAndUri:
    def test_default_auto_chain_is_aifs_then_physical(self, monkeypatch):
        # pins the SHIPPED decision: AIFS AI-primary ON + verified token → chain [AIFS, physical].
        # Assert the env-INDEPENDENT class-field defaults (Greptile P1: the global `settings` singleton
        # can be flipped by a local .env / CI env var, which would fail this network-free test even
        # though the shipped code default is correct).
        from src.config import Settings, settings

        assert Settings.model_fields["FORECAST_ENABLE_AI"].default is True
        assert Settings.model_fields["FORECAST_AI_MODEL"].default == "ecmwf_aifs025_single"
        # With those shipped values active, the "auto" chain is [AIFS, physical] — set explicitly so a
        # caller's FORECAST_* override can't flip this assertion.
        monkeypatch.setattr(settings, "FORECAST_ENABLE_AI", True)
        monkeypatch.setattr(settings, "FORECAST_AI_MODEL", "ecmwf_aifs025_single")
        names = [p.name for p in build_router("auto").providers]
        assert names == ["openmeteo_ai", "openmeteo_physical"]

    def test_single_provider_pins(self):
        # finding #4 — the documented provider= pins each yield their single named provider; a
        # swapped wiring (physical pin serving the live-null AI model) would fail here.
        assert [p.name for p in build_router("openmeteo").providers] == ["openmeteo_physical"]
        assert [p.name for p in build_router("ai").providers] == ["openmeteo_ai"]

    def test_dated_output_uri(self, monkeypatch):
        # finding #3 — the in-handler dated URI build (§4.8): bucket set → dated path; unset → None
        from src.tools.forecast import _forecast_output_uri

        monkeypatch.setattr("src.config.settings.S3_BUCKET", "deposium")
        assert (
            _forecast_output_uri("montpellier", "2026-06-21")
            == "s3://deposium/climate/forecast/montpellier_2026-06-21.parquet"
        )
        monkeypatch.setattr("src.config.settings.S3_BUCKET", "")
        assert _forecast_output_uri("montpellier", "2026-06-21") is None


# --------------------------------------------------------------------------- #
# 8. Live probe (skipped offline)
# --------------------------------------------------------------------------- #


@pytest.mark.skipif(not _RUN_LIVE, reason="set OPENMETEO_LIVE=1 to hit the live Forecast API")
class TestForecastLiveContract:
    def test_physical_six_variables_available(self):
        # The physical best-match (models omitted) must expose all six §5.1 daily variables + units —
        # the contested humidity (Greptile #57) included. A 4xx/rename/unit drift fails loudly here.
        from src.utils.openmeteo import PROVIDER_VARS, fetch_forecast_daily

        raw = fetch_forecast_daily(
            latitude=43.6,
            longitude=3.5,
            normalized_variables=list(NORMALIZED_VARS),
            forecast_days=3,
            models="",
        )
        daily = raw["daily"]
        units = raw.get("daily_units", {})
        assert daily.get("time")
        for pv in PROVIDER_VARS:
            assert pv.provider in daily, f"Forecast missing {pv.provider!r}"
            assert units.get(pv.provider) == pv.source_unit, (
                f"unit drift on {pv.provider!r}: expected {pv.source_unit!r}, got "
                f"{units.get(pv.provider)!r}"
            )

    def test_aifs_primary_token_usable(self):
        # Prove the SHIPPED AIFS token works. Hard-code `ecmwf_aifs025_single` (Greptile P2: probing
        # settings.FORECAST_AI_MODEL would let an env override silently probe the old null token or a
        # blank token that omits models= and passes against the physical best-match — masking a real
        # regression of the token this PR enables). Also pin that it IS the shipped class default.
        from src.config import Settings
        from src.utils.openmeteo import fetch_forecast_daily

        assert Settings.model_fields["FORECAST_AI_MODEL"].default == "ecmwf_aifs025_single"
        raw = fetch_forecast_daily(
            latitude=43.6,
            longitude=3.5,
            normalized_variables=list(NORMALIZED_VARS),
            forecast_days=3,
            models="ecmwf_aifs025_single",
        )
        dates, columns = normalize_openmeteo_daily(raw, list(NORMALIZED_VARS), strict=False)
        usable = _is_usable(*_trim_trailing_null_days(dates, columns), list(NORMALIZED_VARS))
        assert usable, (
            "AIFS token 'ecmwf_aifs025_single' no longer yields usable data — "
            "the router will fall back to physical, but re-verify the AI primary token"
        )

    def test_old_aifs_token_still_null_for_the_record(self):
        # Documents WHY the token changed: the old ecmwf_aifs025 daily aggregate is still null.
        from src.utils.openmeteo import fetch_forecast_daily

        raw = fetch_forecast_daily(
            latitude=43.6,
            longitude=3.5,
            normalized_variables=list(NORMALIZED_VARS),
            forecast_days=3,
            models="ecmwf_aifs025",
        )
        dates, columns = normalize_openmeteo_daily(raw, list(NORMALIZED_VARS), strict=False)
        usable = _is_usable(*_trim_trailing_null_days(dates, columns), list(NORMALIZED_VARS))
        assert not usable, "ecmwf_aifs025 daily now works too — simplify the token note"
