"""Real fixed-case replay and closed private transport refusal boundaries."""

from __future__ import annotations

import copy
import hashlib
import json
import shutil

import pytest
from pydantic import ValidationError

from src import b6_historical as historical
from src.b6_saved_contracts import B6HistoricalInputs
from src.contracts import (
    B6HistoricalCaseTrace,
    B6HistoricalSource,
    PatternAlertInput,
    PatternAlertOutput,
)
from src.tools import pattern_alert as tool_module
from tests.test_b6_saved_transport import request as saved_request
from tests.test_b6_saved_transport import tool


def request():
    return {
        "schema_version": "geoai-b6-historical-request/v1",
        "case_id": historical.CASE_ID,
        "case_version": historical.CASE_VERSION,
        "expected_case_manifest_sha256": historical.CASE_MANIFEST_SHA256,
        "packet_sha256": "a" * 64,
    }


@pytest.fixture(autouse=True)
def no_acquisition(monkeypatch):
    monkeypatch.setattr(tool_module.settings, "MCP_TRANSPORT", "http")
    monkeypatch.setattr(tool_module.settings, "GEOAI_MESH_AUTH_TOKEN", "x" * 32)

    def forbidden(*_args, **_kwargs):
        pytest.fail("historical replay attempted acquisition or mutable source resolution")

    for name in ("load_dataframe", "_resolve_patterns_source", "_run_forecast_query"):
        monkeypatch.setattr(tool_module, name, forbidden)


async def test_real_pinned_case_preserves_all_native_rules_and_exact_j1_arithmetic():
    result = await tool()(
        zone_id="fos_sur_mer", mode="historical_case_j1", historical_inputs=request()
    )
    assert "error" not in result, result
    PatternAlertOutput.model_validate(result)
    source = result["historical_source"]
    assert (
        hashlib.sha256(source["library_json"].encode()).hexdigest()
        == historical.LIBRARY_JSON_SHA256
    )
    assert (
        hashlib.sha256(source["forecast_json"].encode()).hexdigest() == historical.FORECAST_SHA256
    )
    assert len(json.loads(source["library_json"])["patterns"]) == 149
    assert len(source["library_json"].encode()) <= 256 * 1024
    assert len(source["forecast_json"].encode()) <= 256 * 1024
    score = result["level2_result"]
    assert score["zone_id"] == "fos_sur_mer"
    assert score["confidence"] is None and score["event_probability"] is None
    assert score["n_patterns_read"] == len(score["trace"]["patterns"]) == 149
    assert score["n_eligible"] == 2
    assert [p["rule_id"] for p in score["patterns"]] == [
        "cross_zone/tgrad/39",
        "cross_zone/tgrad/15",
    ]
    for pattern in score["patterns"]:
        factors = {item["name"]: item["value"] for item in pattern["factors"]}
        assert pattern["explanatory_index"] == pytest.approx(
            factors["support"]
            * factors["stability"]
            * factors["forecast_agreement"]
            * factors["lag_decay"]
        )
    trace = score["trace"]
    assert trace["library"]["zone_id"] == "cross_zone"
    assert trace["library"]["qualification"] == "scientific_adoption_pending"
    context = trace["historical_case"]
    assert context["forecast_point"] == {"lat": 43.445, "lon": 4.92}
    assert context["model_issued_at"] == "2023-06-30T00:00:00Z"
    assert context["retrieved_at"] == "2026-10-26T21:50:52Z"
    assert context["scientific_adoption"] == "pending"
    assert context["ready_for_scoring"] is context["live_activation"] is False
    assert context["acquisition"] == "none"
    assert len(context["limitations"]) >= 5
    assert trace["forecast"]["packet_sha256"] == "a" * 64
    assert trace["forecast"]["sha256"] == historical.FORECAST_SHA256
    report, raw, compatibility = historical.verify_pinned_case()
    assert report["descriptor"]["ready_for_scoring"] is compatibility["ready_for_scoring"] is False
    assert report["descriptor"]["scientific_adoption"] == "pending"
    assert hashlib.sha256(raw).hexdigest() == historical.FORECAST_SHA256
    assert all(
        row["humidity"] is row["evapotranspiration"] is None for row in json.loads(raw)["forecast"]
    )


@pytest.mark.parametrize(
    "key,value",
    [
        ("case_id", "other"),
        ("case_version", "2026.10.28.1"),
        ("expected_case_manifest_sha256", "0" * 64),
        ("packet_sha256", "invalid"),
        ("forecast_json", "{}"),
        ("patterns_source", "/secret/path"),
        ("scientific_adoption", "approved"),
        ("geometry", [4.85, 43.4]),
    ],
)
async def test_closed_case_request_refuses_tampering_and_overrides(key, value):
    payload = request()
    payload[key] = value
    result = await tool()(
        zone_id="fos_sur_mer", mode="historical_case_j1", historical_inputs=payload
    )
    assert result == {
        "error": "Historical B6 inputs refused",
        "metadata": {"mode": "historical_case_j1", "code": "B6_HISTORICAL_INPUTS_REFUSED"},
    }


@pytest.mark.parametrize(
    "kwargs",
    [
        {"zone_id": "cross_zone"},
        {"min_support": 0.1},
        {"provider": "gfs"},
        {"patterns_source": "other.parquet"},
        {"allow_local_source": True},
        {"saved_inputs": {}},
        {"historical_inputs": None},
    ],
)
async def test_historical_target_and_legacy_options_refused(kwargs):
    options = {
        "zone_id": "fos_sur_mer",
        "mode": "historical_case_j1",
        "historical_inputs": request(),
    }
    options.update(kwargs)
    assert (await tool()(**options))["metadata"]["code"] == "B6_HISTORICAL_INPUTS_REFUSED"


@pytest.mark.parametrize(
    "name",
    [
        "case-manifest.json",
        "forecast.json",
        "extraction-manifest.json",
        "package/library.json",
        "package/source/tgrad.parquet",
        "package/source/zones.geojson",
    ],
)
async def test_tampered_server_asset_refused_before_scoring(tmp_path, monkeypatch, name):
    root = tmp_path / "case"
    shutil.copytree(historical._CASE_ROOT, root)
    path = root / name
    path.write_bytes(path.read_bytes() + b" ")
    monkeypatch.setattr(historical, "_CASE_ROOT", root)
    monkeypatch.setattr(
        historical, "score_level2", lambda *_a, **_k: pytest.fail("scored bad bytes")
    )
    result = await tool()(
        zone_id="fos_sur_mer", mode="historical_case_j1", historical_inputs=request()
    )
    assert result["metadata"]["code"] == "B6_HISTORICAL_INPUTS_REFUSED"
    assert str(tmp_path) not in json.dumps(result)


async def test_missing_auth_and_oversized_or_deep_inputs_refused_before_scoring(monkeypatch):
    monkeypatch.setattr(
        historical, "score_level2", lambda *_a, **_k: pytest.fail("scored invalid input")
    )
    nested = {}
    for _ in range(20):
        nested = {"x": nested}
    for payload in ({**request(), "x": "x" * 131073}, {**request(), "x": nested}):
        assert "error" in await tool()(
            zone_id="fos_sur_mer", mode="historical_case_j1", historical_inputs=payload
        )
    monkeypatch.setattr(tool_module.settings, "GEOAI_MESH_AUTH_TOKEN", "")
    assert "error" in await tool()(
        zone_id="fos_sur_mer", mode="historical_case_j1", historical_inputs=request()
    )


async def test_existing_saved_v1_remains_unchanged_and_rejects_mixed_case():
    result = await tool()(
        zone_id="bassin_du_lez", mode="saved_inputs_j1", saved_inputs=saved_request()
    )
    assert result["metadata"]["mode"] == "saved_inputs_j1"
    assert "historical_case" not in result["level2_result"]["trace"]
    assert "historical_source" not in result
    assert "historical_source" not in PatternAlertOutput.model_validate(result).model_dump()
    result = await tool()(
        zone_id="bassin_du_lez",
        mode="saved_inputs_j1",
        saved_inputs=saved_request(),
        historical_inputs=request(),
    )
    assert result["metadata"]["code"] == "B6_SAVED_INPUTS_REFUSED"
    with pytest.raises(ValidationError):
        PatternAlertInput(zone_id="fos_sur_mer", historical_inputs=request())
    B6HistoricalInputs.model_validate(request())


def test_typed_historical_context_forbids_false_adoption_or_changed_geometry():
    context = historical.score_historical_case("fos_sur_mer", request())["level2_result"]["trace"][
        "historical_case"
    ]
    for key, value in (
        ("scientific_adoption", "approved"),
        ("ready_for_scoring", True),
        ("live_activation", True),
        ("forecast_point", {"lat": 43.44, "lon": 4.92}),
    ):
        changed = copy.deepcopy(context)
        changed[key] = value
        with pytest.raises(ValidationError):
            B6HistoricalCaseTrace.model_validate(changed)


async def test_fastmcp_serializes_typed_historical_context():
    from fastmcp import Client, FastMCP

    server = FastMCP("historical-fixed-case-test")
    tool_module.register_pattern_alert_tools(server)
    async with Client(server) as client:
        result = await client.call_tool(
            "pattern_alert",
            {
                "zone_id": "fos_sur_mer",
                "mode": "historical_case_j1",
                "historical_inputs": request(),
            },
        )
    score = result.structured_content["level2_result"]
    assert score["trace"]["historical_case"]["scientific_adoption"] == "pending"
    assert score["patterns"][0]["rule_id"] == "cross_zone/tgrad/39"


async def test_http_auth_refuses_missing_and_wrong_bearer_before_historical_scoring(monkeypatch):
    import httpx
    from fastmcp import FastMCP

    from src.auth import build_mesh_auth

    monkeypatch.setattr(
        tool_module,
        "score_historical_case",
        lambda *_a, **_k: pytest.fail("unauthenticated HTTP reached historical scoring"),
    )
    server = FastMCP("historical-auth-test", auth=build_mesh_auth("x" * 32))
    tool_module.register_pattern_alert_tools(server)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=server.http_app()), base_url="http://geoai.test"
    ) as client:
        for headers in ({}, {"Authorization": "Bearer wrong"}):
            response = await client.post(
                "/mcp",
                headers=headers,
                json={
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "tools/call",
                    "params": {
                        "name": "pattern_alert",
                        "arguments": {
                            "zone_id": "fos_sur_mer",
                            "mode": "historical_case_j1",
                            "historical_inputs": request(),
                        },
                    },
                },
            )
            assert response.status_code == 401


def test_historical_source_output_is_closed_and_bounded():
    for source in (
        {"library_json": "{}", "forecast_json": "{}", "path": "/caller"},
        {"library_json": "x" * (256 * 1024 + 1), "forecast_json": "{}"},
        {"library_json": "{}", "forecast_json": "x" * (256 * 1024 + 1)},
    ):
        with pytest.raises(ValidationError):
            B6HistoricalSource.model_validate(source)


def test_runtime_verifies_native_package_exactly_once_and_public_preflight_still_verifies(
    monkeypatch,
):
    from src.patterns import historical_case

    original = historical_case.verify_historical_case
    calls = []

    def counted(*args, **kwargs):
        calls.append(args)
        return original(*args, **kwargs)

    # Count both historical imports: the old nested public preflight made two calls.
    monkeypatch.setattr(historical, "verify_historical_case", counted)
    monkeypatch.setattr(historical_case, "verify_historical_case", counted)
    result = historical.score_historical_case("fos_sur_mer", request())
    assert len(calls) == 1
    raw = result["historical_source"]["forecast_json"].encode()
    public = historical_case.preflight_historical_forecast(
        historical._CASE_ROOT / "package",
        historical.PREPARATION_MANIFEST_SHA256,
        raw,
        historical.FORECAST_SHA256,
        forecast_profile="rule15-temperature",
    )
    assert len(calls) == 2
    context = result["level2_result"]["trace"]["historical_case"]
    assert public["manifest_sha256"] == context["preparation_manifest_sha256"]
    assert public["forecast_sha256"] == context["forecast_sha256"]
    assert public["model_issued_at"] == context["model_issued_at"]
    assert public["ready_for_scoring"] is context["ready_for_scoring"] is False


def test_public_preflight_still_refuses_tampered_package_before_compatibility(tmp_path):
    from src.patterns.historical_case import preflight_historical_forecast

    package = tmp_path / "package"
    shutil.copytree(historical._CASE_ROOT / "package", package)
    library = package / "library.json"
    library.write_bytes(library.read_bytes() + b" ")
    with pytest.raises(ValueError):
        preflight_historical_forecast(
            package,
            historical.PREPARATION_MANIFEST_SHA256,
            (historical._CASE_ROOT / "forecast.json").read_bytes(),
            historical.FORECAST_SHA256,
            forecast_profile="rule15-temperature",
        )
