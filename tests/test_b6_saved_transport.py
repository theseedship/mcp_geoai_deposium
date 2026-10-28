"""Saved B6 mode binds original package bytes without provider or caller-path I/O."""

from __future__ import annotations

import copy
import hashlib
import json
from datetime import date, timedelta
from pathlib import Path

import pytest

from src.tools import pattern_alert as tool_module
from tests.test_b6_explanatory_cli import synthetic_package as exported_package_fixture

synthetic_package = exported_package_fixture

REPO = Path(__file__).resolve().parents[1]
PACKAGE = REPO / "src/data/pattern-library-2026-10-15/package"
MANIFEST_SHA = "bfed1a04a176e9fba288fa822b3a02a3f280591c8fd9c4942bb245854cfa47e2"


class CaptureMCP:
    def tool(self, **_kwargs):
        def register(function):
            self.function = function
            return function

        return register


def tool():
    capture = CaptureMCP()
    tool_module.register_pattern_alert_tools(capture)
    return capture.function


def request(package=PACKAGE, manifest_sha=MANIFEST_SHA):
    manifest = json.loads((package / "manifest.json").read_bytes())
    entry = next(
        entry
        for entry in manifest["libraries"]
        if (entry["zone_id"], entry["algorithm"]) == ("bassin_du_lez", "tgrad")
    )
    forecast = {
        "zone_id": "bassin_du_lez",
        "forecast": [
            {"date": (date(2026, 10, 16) + timedelta(days=i)).isoformat(), "temperature_2m": 20 - i}
            for i in range(10)
        ],
        "metadata": {
            "source": "fixture",
            "provider_chain": [{"provider": "fixture", "models": "fixture", "status": "ok"}],
            "issued_at": "2026-10-16T00:00:00+00:00",
            "horizon_days": 10,
        },
    }
    forecast_json = json.dumps(forecast, sort_keys=True, separators=(",", ":"))
    return {
        "schema_version": "geoai-b6-saved-request/v1",
        "library_version": manifest["library_version"],
        "algorithm": "tgrad",
        "expected_manifest_sha256": manifest_sha,
        "expected_json_sha256": manifest["files"][entry["json_path"]]["sha256"],
        "expected_parquet_sha256": manifest["files"][entry["parquet_path"]]["sha256"],
        "expected_library": json.loads((package / entry["json_path"]).read_bytes()),
        "forecast_json": forecast_json,
        "forecast_sha256": hashlib.sha256(forecast_json.encode()).hexdigest(),
        "packet_sha256": "f" * 64,
    }


@pytest.fixture(autouse=True)
def no_acquisition(monkeypatch):
    monkeypatch.setattr(tool_module.settings, "MCP_TRANSPORT", "http")
    monkeypatch.setattr(tool_module.settings, "GEOAI_MESH_AUTH_TOKEN", "x" * 32)

    def forbidden(*_args, **_kwargs):
        pytest.fail("saved mode attempted provider or caller-path acquisition")

    monkeypatch.setattr(tool_module, "load_dataframe", forbidden)
    monkeypatch.setattr(tool_module, "_resolve_patterns_source", forbidden)
    monkeypatch.setattr(tool_module, "_run_forecast_query", forbidden)


async def test_original_native_package_has_honest_no_conclusion():
    result = await tool()(zone_id="bassin_du_lez", mode="saved_inputs_j1", saved_inputs=request())
    assert "error" not in result, result
    score = result["level2_result"]
    assert score["status"] == "no_conclusion"
    assert score["reason"] == "no_admissible_pattern"
    assert score["trace"]["library"]["qualification"] == "unqualified"
    assert score["confidence"] is None and score["event_probability"] is None
    assert result["alerts"] == []
    assert result["metadata"] == {
        "mode": "saved_inputs_j1",
        "packet_sha256": "f" * 64,
        "forecast_sha256": request()["forecast_sha256"],
        "acquisition": "none",
    }


async def test_verified_synthetic_cross_validated_package_is_explained(
    monkeypatch, synthetic_package, tmp_path
):
    from src import b6_saved

    package, digest = synthetic_package
    payload = request(package, digest)
    monkeypatch.setattr(
        b6_saved, "_CATALOG", {payload["library_version"]: b6_saved.CatalogEntry(package, digest)}
    )
    result = await tool()(zone_id="bassin_du_lez", mode="saved_inputs_j1", saved_inputs=payload)
    assert "error" not in result, result
    score = result["level2_result"]
    assert score["status"] == "explained"
    assert score["patterns"][0]["explanatory_index"] == pytest.approx(0.8 * 0.9 * 0.5**0.7)
    assert score["trace"]["params"] == {
        "min_folds": 5,
        "min_support": 0.5,
        "min_stability": 0.5,
        "halflife_days": 10.0,
        "max_patterns": 10,
    }
    assert score["trace"]["forecast"]["packet_sha256"] == payload["packet_sha256"]
    (tmp_path / "synthetic-wrapper.json").write_text(json.dumps(result), encoding="utf-8")


async def test_tampered_server_owned_package_is_refused_without_path_disclosure(
    monkeypatch, synthetic_package
):
    from src import b6_saved

    package, digest = synthetic_package
    payload = request(package, digest)
    monkeypatch.setattr(
        b6_saved, "_CATALOG", {payload["library_version"]: b6_saved.CatalogEntry(package, digest)}
    )
    (package / "libraries/bassin_du_lez/tgrad.parquet").write_bytes(b"tampered")
    result = await tool()(zone_id="bassin_du_lez", mode="saved_inputs_j1", saved_inputs=payload)
    assert result["error"] == "Saved B6 inputs refused"
    assert str(package) not in json.dumps(result)


@pytest.mark.parametrize(
    ("failure", "category"),
    [
        (FileNotFoundError("/private/package/customer.json"), "package_io_failed"),
        (ValueError("private saved forecast values"), "input_or_package_verification_failed"),
        (RuntimeError("private upstream secret"), "scorer_failed"),
    ],
)
async def test_saved_failure_logs_only_a_safe_category(monkeypatch, failure, category):
    from types import SimpleNamespace

    events = []
    monkeypatch.setattr(
        tool_module,
        "log",
        SimpleNamespace(warning=lambda event, **fields: events.append((event, fields))),
    )

    def broken(*_args):
        raise failure

    monkeypatch.setattr(tool_module, "score_saved_inputs", broken)
    result = await tool()(zone_id="bassin_du_lez", mode="saved_inputs_j1", saved_inputs=request())
    assert result["metadata"]["code"] == "B6_SAVED_INPUTS_REFUSED"
    assert events == [("b6_saved_inputs_refused", {"category": category})]
    assert str(failure) not in json.dumps({"result": result, "events": events})


@pytest.mark.parametrize(
    ("token", "options", "category"),
    [
        ("", {}, "mesh_auth_unavailable"),
        ("x" * 32, {"min_support": 0.1}, "invalid_request"),
    ],
)
async def test_saved_failure_distinguishes_auth_and_invalid_flags(
    monkeypatch, token, options, category
):
    from types import SimpleNamespace

    events = []
    monkeypatch.setattr(tool_module.settings, "GEOAI_MESH_AUTH_TOKEN", token)
    monkeypatch.setattr(
        tool_module,
        "log",
        SimpleNamespace(warning=lambda event, **fields: events.append((event, fields))),
    )
    result = await tool()(
        zone_id="bassin_du_lez", mode="saved_inputs_j1", saved_inputs=request(), **options
    )
    assert result["metadata"]["code"] == "B6_SAVED_INPUTS_REFUSED"
    assert events == [("b6_saved_inputs_refused", {"category": category})]


async def test_saved_mode_uses_existing_mcp_tool_and_typed_output():
    from fastmcp import Client, FastMCP

    server = FastMCP("saved-mode-contract-test")
    tool_module.register_pattern_alert_tools(server)
    async with Client(server) as client:
        tools = await client.list_tools()
        assert [entry.name for entry in tools] == ["pattern_alert"]
        assert tools[0].inputSchema["properties"]["mode"]["default"] == "live"
        saved_schema = next(
            choice
            for choice in tools[0].inputSchema["properties"]["saved_inputs"]["anyOf"]
            if choice.get("type") == "object"
        )
        assert saved_schema["additionalProperties"] is False
        assert saved_schema["properties"]["forecast_sha256"]["pattern"] == "^[a-f0-9]{64}$"
        assert (
            saved_schema["properties"]["expected_library"]["properties"]["patterns"]["maxItems"]
            == 256
        )
        result = await client.call_tool(
            "pattern_alert",
            {
                "zone_id": "bassin_du_lez",
                "mode": "saved_inputs_j1",
                "saved_inputs": request(),
            },
        )
    assert result.structured_content["level2_result"]["status"] == "no_conclusion"


async def test_existing_http_mesh_rejects_missing_and_wrong_bearer_before_scoring(monkeypatch):
    import httpx
    from fastmcp import FastMCP

    from src.auth import build_mesh_auth

    def forbidden(*_args, **_kwargs):
        pytest.fail("unauthenticated HTTP reached scoring")

    monkeypatch.setattr(tool_module, "score_saved_inputs", forbidden)
    server = FastMCP("saved-mode-auth-test", auth=build_mesh_auth("x" * 32))
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
                            "zone_id": "bassin_du_lez",
                            "mode": "saved_inputs_j1",
                            "saved_inputs": request(),
                        },
                    },
                },
            )
            assert response.status_code == 401


@pytest.mark.parametrize(
    "field,value",
    [
        ("library_version", "unknown"),
        ("expected_manifest_sha256", "0" * 64),
        ("expected_json_sha256", "0" * 64),
        ("expected_parquet_sha256", "0" * 64),
        ("forecast_sha256", "0" * 64),
        ("packet_sha256", "invalid"),
        ("schema_version", "unknown"),
        ("package_dir", "/etc/passwd"),
    ],
)
async def test_refusals_are_generic_and_do_not_acquire(field, value):
    payload = request()
    payload[field] = value
    result = await tool()(zone_id="bassin_du_lez", mode="saved_inputs_j1", saved_inputs=payload)
    assert result == {
        "error": "Saved B6 inputs refused",
        "metadata": {"mode": "saved_inputs_j1", "code": "B6_SAVED_INPUTS_REFUSED"},
    }


@pytest.mark.parametrize("mutation", ["cv", "support", "subset", "extra", "zone", "schema"])
async def test_original_library_cannot_be_rewritten(mutation):
    payload = request()
    library = payload["expected_library"]
    if mutation == "cv":
        library["patterns"][0].update(n_folds=5, zones_validated=["bassin_du_lez"], stability=0.9)
    elif mutation == "support":
        library["patterns"][0]["support"] = 0.123
    elif mutation == "subset":
        library["patterns"] = library["patterns"][:1]
    elif mutation == "extra":
        library["invented"] = True
    elif mutation == "zone":
        library["zone_id"] = "other"
    else:
        library["schema_version"] = "9.0.0"
    result = await tool()(zone_id="bassin_du_lez", mode="saved_inputs_j1", saved_inputs=payload)
    assert result["metadata"]["code"] == "B6_SAVED_INPUTS_REFUSED"


@pytest.mark.parametrize(
    "mutation", ["zone", "dates", "order", "duplicate", "horizon", "nan", "error", "rows"]
)
async def test_saved_forecast_refusals(mutation):
    payload = request()
    forecast = json.loads(payload["forecast_json"])
    if mutation == "zone":
        forecast["zone_id"] = "other"
    elif mutation == "dates":
        forecast["forecast"][0]["date"] = "2026-02-31"
    elif mutation == "order":
        forecast["forecast"].reverse()
    elif mutation == "duplicate":
        forecast["forecast"][1]["date"] = forecast["forecast"][0]["date"]
    elif mutation == "horizon":
        forecast["metadata"]["horizon_days"] = 11
    elif mutation == "nan":
        forecast["forecast"][0]["temperature_2m"] = float("nan")
    elif mutation == "error":
        forecast["error"] = "provider error containing a secret"
    else:
        forecast["forecast"] = [
            {"date": (date(2026, 1, 1) + timedelta(days=i)).isoformat()} for i in range(257)
        ]
    payload["forecast_json"] = json.dumps(forecast)
    payload["forecast_sha256"] = hashlib.sha256(payload["forecast_json"].encode()).hexdigest()
    result = await tool()(zone_id="bassin_du_lez", mode="saved_inputs_j1", saved_inputs=payload)
    assert result["error"] == "Saved B6 inputs refused"


async def test_semantic_utf8_envelope_limit_and_row_limit():
    payload = request()
    payload["expected_library"]["patterns"][0]["antecedent"] = "é" * 65536
    result = await tool()(zone_id="bassin_du_lez", mode="saved_inputs_j1", saved_inputs=payload)
    assert result["metadata"]["code"] == "B6_SAVED_INPUTS_REFUSED"
    payload = request()
    row = payload["expected_library"]["patterns"][0]
    payload["expected_library"]["patterns"] = [
        {**copy.deepcopy(row), "pattern_id": i} for i in range(257)
    ]
    result = await tool()(zone_id="bassin_du_lez", mode="saved_inputs_j1", saved_inputs=payload)
    assert result["metadata"]["code"] == "B6_SAVED_INPUTS_REFUSED"


async def test_saved_mode_requires_mesh_secret_on_http_but_operator_stdio_works(monkeypatch):
    monkeypatch.setattr(tool_module.settings, "GEOAI_MESH_AUTH_TOKEN", "")
    result = await tool()(zone_id="bassin_du_lez", mode="saved_inputs_j1", saved_inputs=request())
    assert result["metadata"]["code"] == "B6_SAVED_INPUTS_REFUSED"
    monkeypatch.setattr(tool_module.settings, "MCP_TRANSPORT", "stdio")
    result = await tool()(zone_id="bassin_du_lez", mode="saved_inputs_j1", saved_inputs=request())
    assert result["level2_result"]["status"] == "no_conclusion"


async def test_saved_mode_refuses_legacy_threshold_and_path_flags():
    for extra in (
        {"min_support": 0.1},
        {"min_stability": 0.1},
        {"patterns_source": "/etc/passwd"},
        {"provider": "physical"},
        {"allow_local_source": True},
    ):
        result = await tool()(
            zone_id="bassin_du_lez", mode="saved_inputs_j1", saved_inputs=request(), **extra
        )
        assert result["metadata"]["code"] == "B6_SAVED_INPUTS_REFUSED"


@pytest.mark.parametrize("mutation", ["hash", "extra", "nested"])
async def test_framework_refusal_boundary_does_not_echo_invalid_saved_values(mutation):
    from fastmcp import Client, FastMCP

    payload = request()
    private_value = "private-submitted-value-must-not-be-quoted"
    if mutation == "hash":
        payload["forecast_sha256"] = private_value
    elif mutation == "extra":
        payload["private_extra"] = private_value
    else:
        payload["expected_library"]["patterns"][0]["variables"] = {"secret": private_value}
    server = FastMCP("saved-invalid-boundary-test")
    tool_module.register_pattern_alert_tools(server)
    async with Client(server) as client:
        result = await client.call_tool(
            "pattern_alert",
            {
                "zone_id": "bassin_du_lez",
                "mode": "saved_inputs_j1",
                "saved_inputs": payload,
            },
        )
    assert result.structured_content == {
        "error": "Saved B6 inputs refused",
        "metadata": {"mode": "saved_inputs_j1", "code": "B6_SAVED_INPUTS_REFUSED"},
    }
    assert private_value not in str(result)


@pytest.mark.parametrize("mutation", ["string", "list", "depth"])
async def test_saved_preflight_refuses_before_nested_validation(monkeypatch, mutation):
    from fastmcp import Client, FastMCP

    payload = request()
    record = payload["expected_library"]["patterns"][0]
    if mutation == "string":
        record["antecedent"] = "é" * (128 * 1024)
    elif mutation == "list":
        record["variables"] = ["private"] * 65536
    else:
        value = "private"
        for _ in range(40):
            value = [value]
        record["variables"] = value

    def forbidden(*_args, **_kwargs):
        pytest.fail("full nested model validation ran before bounded preflight")

    monkeypatch.setattr(tool_module, "check_input", forbidden)
    server = FastMCP("saved-bounded-preflight-test")
    tool_module.register_pattern_alert_tools(server)
    async with Client(server) as client:
        result = await client.call_tool(
            "pattern_alert",
            {
                "zone_id": "bassin_du_lez",
                "mode": "saved_inputs_j1",
                "saved_inputs": payload,
            },
        )
    assert result.structured_content["metadata"]["code"] == "B6_SAVED_INPUTS_REFUSED"


async def test_saved_horizon_smaller_than_scored_rows_is_refused(monkeypatch, synthetic_package):
    from src import b6_saved

    package, digest = synthetic_package
    payload = request(package, digest)
    monkeypatch.setattr(
        b6_saved,
        "_CATALOG",
        {
            payload["library_version"]: b6_saved.CatalogEntry(package, digest),
        },
    )
    forecast = json.loads(payload["forecast_json"])
    forecast["metadata"]["horizon_days"] = 8
    payload["forecast_json"] = json.dumps(forecast)
    payload["forecast_sha256"] = hashlib.sha256(payload["forecast_json"].encode()).hexdigest()
    result = await tool()(zone_id="bassin_du_lez", mode="saved_inputs_j1", saved_inputs=payload)
    assert result == {
        "error": "Saved B6 inputs refused",
        "metadata": {"mode": "saved_inputs_j1", "code": "B6_SAVED_INPUTS_REFUSED"},
    }


async def test_every_original_catalog_library_fits_preflight_bounds():
    manifest = json.loads((PACKAGE / "manifest.json").read_bytes())
    counts = []
    for entry in manifest["libraries"]:
        payload = request()
        library = json.loads((PACKAGE / entry["json_path"]).read_bytes())
        payload.update(
            algorithm=entry["algorithm"],
            expected_library=library,
            expected_json_sha256=manifest["files"][entry["json_path"]]["sha256"],
            expected_parquet_sha256=manifest["files"][entry["parquet_path"]]["sha256"],
        )
        forecast = json.loads(payload["forecast_json"])
        forecast["zone_id"] = entry["zone_id"]
        payload["forecast_json"] = json.dumps(forecast)
        payload["forecast_sha256"] = hashlib.sha256(payload["forecast_json"].encode()).hexdigest()
        result = await tool()(
            zone_id=entry["zone_id"], mode="saved_inputs_j1", saved_inputs=payload
        )
        assert "error" not in result, (entry["zone_id"], entry["algorithm"], result)
        assert result["level2_result"]["n_patterns_read"] == len(library["patterns"])
        counts.append(len(library["patterns"]))
    assert sorted(counts) == [6, 7, 10, 108, 135, 152]
