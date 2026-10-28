from __future__ import annotations

import pandas as pd
import pytest

from src.tools import pattern_alert as tool_module
from src.tools.pattern_alert import register_pattern_alert_tools


class _CaptureMCP:
    def __init__(self) -> None:
        self.tools = {}

    def tool(self, *_args, **_kwargs):
        def deco(fn):
            self.tools[fn.__name__] = fn
            return fn

        return deco


def _tool():
    mcp = _CaptureMCP()
    register_pattern_alert_tools(mcp)
    return mcp.tools["pattern_alert"]


@pytest.fixture
def local_patterns_source(monkeypatch, tmp_path):
    monkeypatch.setattr(tool_module.settings, "ALLOW_LOCAL_PARQUET_SOURCE", True)
    monkeypatch.setattr(tool_module.settings, "LOCAL_PARQUET_SOURCE_ROOT", tmp_path.as_posix())
    return (tmp_path / "patterns.parquet").as_posix()


def _patterns_df(**overrides) -> pd.DataFrame:
    row = {
        "pattern_id": 7,
        "antecedent": "(evapotranspiration+) +7.0 days",
        "consequent": "ndvi_mean+",
        "pattern": "ndvi_mean+, (evapotranspiration+) +7.0 days",
        "variables": '["ndvi_mean", "evapotranspiration"]',
        "support": 0.75,
        "confidence": None,
        "is_temporal": True,
        "lag_days": 7.0,
        "algorithm": "tgrad",
        "zone_id": "cross_zone",
        "zones_validated": '["montpellier", "nairobi"]',
        "period_start": "2023-07-01",
        "period_end": "2023-09-30",
        "n_folds": 5,
        "stability": 0.8,
        "created_at": "2026-06-27T00:00:00Z",
        "extra_metrics": "{}",
    }
    row.update(overrides)
    return pd.DataFrame([row])


def _forecast(rows=None, *, error=None):
    rows = rows or [
        {"date": f"2026-06-{27 + i:02d}", "evapotranspiration": float(i + 1)} for i in range(10)
    ]
    if error:
        return {"error": error, "metadata": {"provider_chain": ["ai", "physical"]}}
    return {
        "zone_id": "montpellier",
        "forecast": rows,
        "metadata": {
            "source": "openmeteo_physical",
            "provider_chain": ["ai", "physical"],
            "horizon_days": len(rows),
        },
    }


async def test_pattern_alert_returns_experimental_score(monkeypatch, local_patterns_source):
    monkeypatch.setattr(tool_module, "load_dataframe", lambda *_a, **_k: _patterns_df())

    async def fake_forecast(**_kwargs):
        return _forecast()

    monkeypatch.setattr(tool_module, "_run_forecast_query", fake_forecast)

    result = await _tool()(
        zone_id="montpellier",
        patterns_source=local_patterns_source,
        allow_local_source=True,
        forecast_horizon_days=10,
    )

    assert "error" not in result, result
    assert result["zone_id"] == "montpellier"
    assert result["metadata"]["confidence_status"] == "experimental_unratified"
    alert = result["alerts"][0]
    assert alert["confidence"] is None
    assert alert["confidence_status"] == "experimental_unratified"
    assert alert["forecast_alignment"] == 1.0
    assert alert["forecast_matches"] == 1
    assert alert["horizon_lag_match"] is True
    assert alert["provisional_score"] == pytest.approx(0.75 * 0.8 * (0.5**0.7))
    assert alert["recommended_action"] == "watch_conditions"


async def test_pattern_alert_rejects_non_o3_library(monkeypatch, local_patterns_source):
    monkeypatch.setattr(
        tool_module,
        "load_dataframe",
        lambda *_a, **_k: _patterns_df(zones_validated="[]", n_folds=None, stability=None),
    )

    async def fake_forecast(**_kwargs):
        return _forecast()

    monkeypatch.setattr(tool_module, "_run_forecast_query", fake_forecast)

    result = await _tool()(
        zone_id="montpellier",
        patterns_source=local_patterns_source,
        allow_local_source=True,
    )

    assert "error" in result
    assert "O3 validated library" in result["error"]


async def test_pattern_alert_lag_outside_horizon_is_not_actionable(
    monkeypatch, local_patterns_source
):
    monkeypatch.setattr(
        tool_module, "load_dataframe", lambda *_a, **_k: _patterns_df(lag_days=30.0)
    )

    async def fake_forecast(**_kwargs):
        return _forecast()

    monkeypatch.setattr(tool_module, "_run_forecast_query", fake_forecast)

    result = await _tool()(
        zone_id="montpellier",
        patterns_source=local_patterns_source,
        allow_local_source=True,
        forecast_horizon_days=10,
    )

    alert = result["alerts"][0]
    assert alert["horizon_lag_match"] is False
    assert alert["provisional_score"] == 0.0
    assert alert["recommended_action"] == "monitor_historical_pattern"


async def test_pattern_alert_uses_effective_forecast_horizon(monkeypatch, local_patterns_source):
    monkeypatch.setattr(tool_module, "load_dataframe", lambda *_a, **_k: _patterns_df(lag_days=7.0))

    async def fake_forecast(**_kwargs):
        return _forecast(
            rows=[
                {"date": "2026-06-27", "evapotranspiration": 1.0},
                {"date": "2026-06-28", "evapotranspiration": 2.0},
            ]
        )

    monkeypatch.setattr(tool_module, "_run_forecast_query", fake_forecast)

    result = await _tool()(
        zone_id="montpellier",
        patterns_source=local_patterns_source,
        allow_local_source=True,
        forecast_horizon_days=99,
    )

    alert = result["alerts"][0]
    assert result["statistics"]["effective_horizon_days"] == 2
    assert alert["horizon_lag_match"] is False
    assert alert["provisional_score"] == 0.0


async def test_pattern_alert_surfaces_forecast_error(monkeypatch, local_patterns_source):
    monkeypatch.setattr(tool_module, "load_dataframe", lambda *_a, **_k: _patterns_df())

    async def fake_forecast(**_kwargs):
        return _forecast(error="all providers down")

    monkeypatch.setattr(tool_module, "_run_forecast_query", fake_forecast)

    result = await _tool()(
        zone_id="montpellier",
        patterns_source=local_patterns_source,
        allow_local_source=True,
    )

    assert result["error"] == "forecast_query failed: all providers down"
    assert result["metadata"]["provider_chain"] == ["ai", "physical"]


async def test_pattern_alert_rejects_local_source_when_global_policy_is_off(monkeypatch, tmp_path):
    monkeypatch.setattr(tool_module.settings, "ALLOW_LOCAL_PARQUET_SOURCE", False)
    monkeypatch.setattr(tool_module.settings, "LOCAL_PARQUET_SOURCE_ROOT", "")
    result = await _tool()(
        zone_id="montpellier",
        patterns_source=(tmp_path / "patterns.parquet").as_posix(),
        allow_local_source=True,
    )

    assert "error" in result
    assert "ALLOW_LOCAL_PARQUET_SOURCE" in result["error"]


async def test_pattern_alert_rejects_local_source_outside_root(monkeypatch, tmp_path):
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    monkeypatch.setattr(tool_module.settings, "ALLOW_LOCAL_PARQUET_SOURCE", True)
    monkeypatch.setattr(tool_module.settings, "LOCAL_PARQUET_SOURCE_ROOT", allowed.as_posix())
    result = await _tool()(
        zone_id="montpellier",
        patterns_source=(tmp_path / "outside.parquet").as_posix(),
        allow_local_source=True,
    )

    assert "error" in result
    assert "LOCAL_PARQUET_SOURCE_ROOT" in result["error"]


async def test_pattern_alert_rejects_cross_bucket_source(monkeypatch):
    monkeypatch.setattr(tool_module.settings, "S3_BUCKET", "deposium")
    result = await _tool()(
        zone_id="montpellier",
        patterns_source="s3://other/climate/o3/holdout/q3_2023/tgrad_validated.parquet",
    )

    assert "error" in result
    assert "configured bucket" in result["error"]
