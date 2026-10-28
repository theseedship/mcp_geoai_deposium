from __future__ import annotations

import asyncio
import hashlib
import json
from unittest.mock import AsyncMock

import pytest

from src import solar_pilot as pilot
from src.tools import solar as tools
from tests.test_solar_portfolio import _payload


def request(**updates):
    return {
        "action": "validate",
        "sites": [{"site_id": "00123", "latitude": 0, "longitude": 0}],
        "source": "era5",
        "start_date": "2026-10-11",
        "end_date": "2026-10-12",
        "variables": ["shortwave_radiation"],
        **updates,
    }


def test_validation_works_disabled_without_tempfiles_or_fetch(monkeypatch):
    monkeypatch.setattr(pilot.tempfile, "TemporaryDirectory", lambda **kw: pytest.fail("tempfile"))
    result = pilot.run_solar_pilot(request(), fetcher=lambda query: pytest.fail("provider"))
    assert result["schema_version"] == "geoai-solar-pilot/v1"
    assert result["items"][0]["status"] == "pending"
    assert result["sites"][0]["site_id"] == "00123"
    assert result["summary"]["http_acquisitions"] == 0


@pytest.mark.parametrize(
    "updates",
    [
        {"sites": []},
        {"sites": [{"site_id": "a", "latitude": 0, "longitude": 0}] * 3},
        {"end_date": "2026-10-11"},
        {"end_date": "2026-10-13"},
        {"start_date": "2026-10-11T00:00:00Z"},
        {"sites": [{"site_id": 123, "latitude": 0, "longitude": 0}]},
        {"sites": [{"site_id": "a", "latitude": True, "longitude": 0}]},
        {"source": "satellite", "variables": ["temperature_2m"]},
        {"variables": ["global_tilted_irradiance"]},
        {"output_dir": "/tmp/unsafe"},
    ],
)
def test_invalid_batch_rejected_before_side_effects(updates, monkeypatch):
    monkeypatch.setattr(pilot.tempfile, "TemporaryDirectory", lambda **kw: pytest.fail("tempfile"))
    with pytest.raises(ValueError):
        pilot.run_solar_pilot(request(**updates))


def test_collection_activation_requires_enabled_and_source_proof():
    for config in [
        pilot.SolarPilotSettings(_env_file=None),
        pilot.SolarPilotSettings(_env_file=None, SOLAR_MCP_ENABLED=True),
        pilot.SolarPilotSettings(
            _env_file=None,
            SOLAR_MCP_ENABLED=True,
            SOLAR_MCP_SOURCE_ACCESS_PROOFS={"forecast": "proof"},
        ),
    ]:
        with pytest.raises(ValueError, match="not activated"):
            pilot.run_solar_pilot(request(action="collect"), config=config)


def test_collection_returns_exact_checksums_profiles_partial_failures_and_dedup():
    sites = [
        {"site_id": "00123", "latitude": 0, "longitude": 0},
        {"site_id": "other", "latitude": 0, "longitude": 0},
    ]
    calls = []

    def fetch(query):
        calls.append(query)
        value = _payload(query)
        interval = value["interval_seconds"]
        from datetime import UTC, datetime, timedelta

        start = datetime(2026, 10, 11, tzinfo=UTC)
        value["timestamps"] = [
            (start + timedelta(seconds=i * interval)).isoformat().replace("+00:00", "Z")
            for i in range(48)
        ]
        value["values"] = {"shortwave_radiation": [0.0, None] * 24}
        value["missing_counts"] = {"shortwave_radiation": 24}
        return value

    config = pilot.SolarPilotSettings(
        _env_file=None,
        SOLAR_MCP_ENABLED=True,
        SOLAR_MCP_SOURCE_ACCESS_PROOFS={"era5": "approval-ref"},
    )
    result = pilot.run_solar_pilot(
        request(action="collect", sites=sites), config=config, fetcher=fetch
    )
    assert result["summary"]["succeeded"] == 2
    assert result["summary"]["http_acquisitions"] == 1
    assert len(calls) == 1
    for item in result["items"]:
        raw = item["profile_json"].encode("utf-8")
        assert raw.endswith(b"\n")
        assert hashlib.sha256(raw).hexdigest() == item["artifact_sha256"]
        assert json.loads(raw)["values"]["shortwave_radiation"][:2] == [0.0, None]
    failed = pilot.run_solar_pilot(
        request(action="collect"),
        config=config,
        fetcher=lambda q: (_ for _ in ()).throw(RuntimeError("unsafe")),
    )
    assert failed["summary"]["failed"] == 1
    assert failed["items"][0]["profile_json"] is None
    assert failed["items"][0]["attempts"] == 1
    assert "unsafe" not in json.dumps(failed)


@pytest.mark.asyncio
async def test_disabled_tool_does_not_spawn(monkeypatch):
    monkeypatch.setattr(
        tools, "SolarPilotSettings", lambda: pilot.SolarPilotSettings(_env_file=None)
    )
    spawn = AsyncMock(side_effect=AssertionError("spawn"))
    monkeypatch.setattr(asyncio, "create_subprocess_exec", spawn)
    result = await tools.solar_portfolio_collect(**request(action="collect"))
    assert "error" in result
    spawn.assert_not_called()


class Process:
    def __init__(self, output=b"", blocking=False):
        self.returncode = None
        self.stdin = type(
            "Writer",
            (),
            {"write": lambda self, v: None, "drain": AsyncMock(), "close": lambda self: None},
        )()
        self.stdout = asyncio.StreamReader()
        if not blocking:
            self.stdout.feed_data(output)
            self.stdout.feed_eof()
        self.killed = False

    def kill(self):
        self.killed = True
        self.returncode = -9

    async def wait(self):
        if self.returncode is None:
            self.returncode = 0
        return self.returncode


@pytest.mark.asyncio
async def test_worker_success(monkeypatch):
    expected = pilot.run_solar_pilot(request())
    process = Process(json.dumps(expected).encode())
    monkeypatch.setattr(asyncio, "create_subprocess_exec", AsyncMock(return_value=process))
    result = await tools._collect_isolated(pilot.SolarPilotInput.model_validate(request()))
    assert result["items"] == expected["items"]
    assert not process.killed


@pytest.mark.asyncio
async def test_worker_timeout_kills_process(monkeypatch):
    process = Process(blocking=True)
    monkeypatch.setattr(asyncio, "create_subprocess_exec", AsyncMock(return_value=process))
    monkeypatch.setattr(tools, "PILOT_TIMEOUT_SECONDS", 0.01)
    result = await tools._collect_isolated(pilot.SolarPilotInput.model_validate(request()))
    assert result == {"error": "solar pilot collection timed out"}
    assert process.killed


@pytest.mark.asyncio
async def test_worker_cancellation_kills_process(monkeypatch):
    process = Process(blocking=True)
    monkeypatch.setattr(asyncio, "create_subprocess_exec", AsyncMock(return_value=process))
    task = asyncio.create_task(
        tools._collect_isolated(pilot.SolarPilotInput.model_validate(request()))
    )
    await asyncio.sleep(0.01)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert process.killed


@pytest.mark.asyncio
async def test_worker_oversized_response_kills_process(monkeypatch):
    process = Process(b"x" * (pilot.MAX_RESPONSE_BYTES + 1))
    monkeypatch.setattr(asyncio, "create_subprocess_exec", AsyncMock(return_value=process))
    assert "error" in await tools._collect_isolated(pilot.SolarPilotInput.model_validate(request()))
    assert process.killed


@pytest.mark.asyncio
async def test_worker_spawn_failure_is_safe(monkeypatch):
    monkeypatch.setattr(
        asyncio, "create_subprocess_exec", AsyncMock(side_effect=OSError("private/path/key"))
    )
    result = await tools._collect_isolated(pilot.SolarPilotInput.model_validate(request()))
    assert result == {"error": "solar pilot worker unavailable"}


def test_solar_only_http_requires_mesh_token(monkeypatch):
    from src import server

    monkeypatch.setattr(server.settings, "MCP_TRANSPORT", "http")
    monkeypatch.setattr(server.settings, "GEOAI_MESH_AUTH_TOKEN", "")
    monkeypatch.setattr(server, "_data_plane_enabled", lambda: False)
    monkeypatch.setattr(
        server,
        "SolarPilotSettings",
        lambda: pilot.SolarPilotSettings(_env_file=None, SOLAR_MCP_ENABLED=True),
    )
    with pytest.raises(RuntimeError, match="GEOAI_MESH_AUTH_TOKEN"):
        server._preflight_mesh_auth()


@pytest.mark.asyncio
async def test_native_fastmcp_validation_returns_structured_contract():
    from fastmcp import Client, FastMCP

    mcp = FastMCP("solar-pilot-test")
    tools.register_solar_tools(mcp)
    async with Client(mcp) as client:
        result = await client.call_tool("solar_portfolio_collect", request())
    assert result.structured_content["schema_version"] == "geoai-solar-pilot/v1"
    assert result.structured_content["summary"]["http_acquisitions"] == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel", [False, True])
async def test_parent_removes_worker_files_after_timeout_or_cancellation(monkeypatch, cancel):
    from pathlib import Path

    process = Process(blocking=True)
    directories = []

    async def spawn(*args, **kwargs):
        folder = Path(kwargs["env"]["GEOAI_SOLAR_WORK_DIR"])
        directories.append(folder)
        (folder / "private-profile.json").write_text("private")
        return process

    monkeypatch.setattr(asyncio, "create_subprocess_exec", spawn)
    monkeypatch.setattr(tools, "PILOT_TIMEOUT_SECONDS", 0.02)
    task = asyncio.create_task(
        tools._collect_isolated(pilot.SolarPilotInput.model_validate(request()))
    )
    if cancel:
        await asyncio.sleep(0.01)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    else:
        assert "timed out" in (await task)["error"]
    assert process.killed
    assert directories and all(not folder.exists() for folder in directories)


def test_published_nested_site_schema_is_strict_and_usable():
    from jsonschema import Draft202012Validator
    from jsonschema.exceptions import ValidationError as SchemaValidationError

    from src.contracts import contracts_json_schema

    schema = contracts_json_schema()["solar_portfolio_collect"]["input"]
    site = schema["$defs"]["SolarPilotSite"]
    assert site["additionalProperties"] is False
    assert set(site["required"]) == {"site_id", "latitude", "longitude"}
    assert site["properties"]["site_id"]["type"] == "string"
    assert schema["properties"]["sites"]["minItems"] == 1
    assert schema["properties"]["sites"]["maxItems"] == 2
    validator = Draft202012Validator(schema)
    validator.validate(request())
    for updates in [{"site_id": 123}, {"unexpected": "bad"}, {"latitude": True}]:
        invalid = request(sites=[{"site_id": "00123", "latitude": 0, "longitude": 0, **updates}])
        with pytest.raises(SchemaValidationError):
            validator.validate(invalid)


@pytest.mark.asyncio
async def test_native_tools_list_matches_published_input_contract():
    from fastmcp import Client, FastMCP

    from src.contracts import contracts_json_schema

    mcp = FastMCP("solar-native-schema")
    tools.register_solar_tools(mcp)
    async with Client(mcp) as client:
        registered = await client.list_tools()
    assert len(registered) == 1
    schema = registered[0].inputSchema
    from fastmcp.utilities.json_schema import compress_schema

    # FastMCP removes titles and inlines model definitions when publishing tools/list.
    published = contracts_json_schema()["solar_portfolio_collect"]["input"]
    assert schema == compress_schema(published, prune_titles=True, dereference=True)
    assert schema["additionalProperties"] is False
    assert schema["properties"]["action"]["enum"] == ["validate", "collect"]
    assert schema["properties"]["source"]["enum"] == ["forecast", "era5", "satellite"]
    assert schema["properties"]["sites"]["items"]["additionalProperties"] is False


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "updates",
    [
        {"sites": [{"site_id": 123, "latitude": 0, "longitude": 0}]},
        {"sites": [{"site_id": "00123", "latitude": "0", "longitude": 0}]},
        {"sites": [{"site_id": "00123", "latitude": False, "longitude": 0}]},
        {"sites": [{"site_id": "00123", "latitude": 0, "longitude": 0, "unexpected": "bad"}]},
        {"sites": [{"site_id": "00123", "latitude": 0, "longitude": 0}] * 3},
        {"source": "other"},
        {"action": "other"},
    ],
)
async def test_native_binding_rejects_invalid_strict_input(updates):
    from fastmcp import Client, FastMCP
    from fastmcp.exceptions import ToolError

    mcp = FastMCP("solar-native-rejection")
    tools.register_solar_tools(mcp)
    async with Client(mcp) as client:
        with pytest.raises(ToolError):
            await client.call_tool("solar_portfolio_collect", request(**updates))


@pytest.mark.asyncio
async def test_hot_activation_cannot_enable_tokenless_http_collection(monkeypatch):
    from src import server

    monkeypatch.setattr(tools.settings, "MCP_TRANSPORT", "http")
    monkeypatch.setattr(tools.settings, "GEOAI_MESH_AUTH_TOKEN", "")
    monkeypatch.setattr(server, "_data_plane_enabled", lambda: False)
    current = pilot.SolarPilotSettings(_env_file=None, SOLAR_MCP_ENABLED=False)
    monkeypatch.setattr(server, "SolarPilotSettings", lambda: current)
    monkeypatch.setattr(tools, "SolarPilotSettings", lambda: current)
    server._preflight_mesh_auth()
    current = pilot.SolarPilotSettings(
        _env_file=None,
        SOLAR_MCP_ENABLED=True,
        SOLAR_MCP_SOURCE_ACCESS_PROOFS={"era5": "access-ref"},
    )
    # A newly injected environment token must not replace the server's static auth config.
    monkeypatch.setenv("GEOAI_MESH_AUTH_TOKEN", "new-environment-token")
    spawn = AsyncMock(side_effect=AssertionError("must not spawn"))
    monkeypatch.setattr(asyncio, "create_subprocess_exec", spawn)
    result = await tools.solar_portfolio_collect(**request(action="collect"))
    assert result == {"error": "solar pilot collection requires configured mesh authentication"}
    spawn.assert_not_called()


def test_contract_import_and_validation_work_without_posix_locking(tmp_path):
    import subprocess
    import sys

    script = r"""
import builtins
import sys
from pathlib import Path
original = builtins.__import__
def reject_posix(name, *args, **kwargs):
    if name == "fcntl":
        raise ImportError("simulated non-POSIX platform")
    return original(name, *args, **kwargs)
builtins.__import__ = reject_posix
from src.contracts import contracts_json_schema
from src.solar_pilot import SolarPilotInput, run_solar_pilot
from src.solar_portfolio import SolarPortfolioError, run_solar_portfolio, export_solar_portfolio
request = {
    "action": "validate", "sites": [{"site_id": "00123", "latitude": 0, "longitude": 0}],
    "source": "era5", "start_date": "2026-10-11", "end_date": "2026-10-12",
    "variables": ["shortwave_radiation"],
}
assert "solar_portfolio_collect" in contracts_json_schema()
assert run_solar_pilot(SolarPilotInput.model_validate(request))["sites"][0]["site_id"] == "00123"
output = Path(sys.argv[1])
def forbidden_fetch(query):
    raise AssertionError("provider must not be called")
try:
    run_solar_portfolio(sites=request["sites"], source="era5", start_date=request["start_date"],
                        end_date=request["end_date"], variables=request["variables"],
                        output_dir=output, fetcher=forbidden_fetch)
except SolarPortfolioError as exc:
    assert "POSIX file locking" in str(exc)
else:
    raise AssertionError("unsupported collection must be refused")
try:
    export_solar_portfolio(output)
except SolarPortfolioError as exc:
    assert "POSIX file locking" in str(exc)
else:
    raise AssertionError("unsupported export must be refused")
assert not output.exists()
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path / "unsupported-batch")],
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 0, result.stderr
    assert not (tmp_path / "unsupported-batch").exists()


@pytest.mark.asyncio
async def test_global_worker_admission_refuses_third_before_files_or_spawn(monkeypatch):
    slots = asyncio.Semaphore(tools.PILOT_MAX_WORKERS)
    monkeypatch.setattr(tools, "_COLLECTION_SLOTS", slots)
    entered = asyncio.Event()
    processes = []
    directories = []
    make_directory = tools.tempfile.TemporaryDirectory

    def temporary_directory(**kwargs):
        context = make_directory(**kwargs)
        directories.append(context.name)
        return context

    async def spawn(*args, **kwargs):
        process = Process(blocking=True)
        if len(processes) >= 2:
            process.stdout.feed_data(json.dumps(pilot.run_solar_pilot(request())).encode())
            process.stdout.feed_eof()
        processes.append(process)
        assert sum(child.returncode is None for child in processes) <= 2
        if len(processes) == 2:
            entered.set()
        return process

    monkeypatch.setattr(tools.tempfile, "TemporaryDirectory", temporary_directory)
    monkeypatch.setattr(asyncio, "create_subprocess_exec", spawn)
    parsed = pilot.SolarPilotInput.model_validate(request())
    running = [asyncio.create_task(tools._collect_isolated(parsed)) for _ in range(2)]
    try:
        await asyncio.wait_for(entered.wait(), timeout=1)
        refused = await asyncio.wait_for(tools._collect_isolated(parsed), timeout=0.1)
        assert refused == {"error": "solar pilot collection capacity exhausted"}
        assert len(processes) == len(directories) == 2
        running[0].cancel()
        with pytest.raises(asyncio.CancelledError):
            await running[0]
        assert processes[0].killed
        accepted = await tools._collect_isolated(parsed)
        assert accepted["schema_version"] == "geoai-solar-pilot/v1"
        assert len(processes) == 3
    finally:
        for task in running:
            if not task.done():
                task.cancel()
        await asyncio.gather(*running, return_exceptions=True)
    from pathlib import Path

    assert all(not Path(directory).exists() for directory in directories)
    assert slots._value == tools.PILOT_MAX_WORKERS


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["spawn_failure", "worker_failure", "timeout"])
async def test_admission_slot_released_after_worker_error(monkeypatch, outcome):
    slots = asyncio.Semaphore(tools.PILOT_MAX_WORKERS)
    monkeypatch.setattr(tools, "_COLLECTION_SLOTS", slots)
    monkeypatch.setattr(tools, "PILOT_TIMEOUT_SECONDS", 0.01)
    if outcome == "spawn_failure":
        spawn = AsyncMock(side_effect=OSError("worker unavailable"))
    else:
        spawn = AsyncMock(return_value=Process(b"invalid-json", blocking=outcome == "timeout"))
    monkeypatch.setattr(asyncio, "create_subprocess_exec", spawn)
    result = await tools._collect_isolated(pilot.SolarPilotInput.model_validate(request()))
    assert "error" in result
    assert slots._value == tools.PILOT_MAX_WORKERS


@pytest.mark.asyncio
async def test_repeated_cancellation_during_reap_keeps_worker_slot_and_files(monkeypatch):
    from pathlib import Path

    slots = asyncio.Semaphore(2)
    # One slot represents another active worker; cleanup owns the remaining slot.
    await slots.acquire()
    monkeypatch.setattr(tools, "_COLLECTION_SLOTS", slots)
    monkeypatch.setattr(tools, "PILOT_TIMEOUT_SECONDS", 0.01)
    reap_started = asyncio.Event()
    permit_reap = asyncio.Event()
    directories = []

    class DelayedReapProcess(Process):
        def __init__(self):
            super().__init__(blocking=True)
            self.reaped = False

        async def wait(self):
            reap_started.set()
            await permit_reap.wait()
            self.reaped = True
            return self.returncode

    process = DelayedReapProcess()

    async def spawn(*args, **kwargs):
        directory = Path(kwargs["env"]["GEOAI_SOLAR_WORK_DIR"])
        directories.append(directory)
        (directory / "profile.json").write_text("private")
        return process

    spawn_mock = AsyncMock(side_effect=spawn)
    monkeypatch.setattr(asyncio, "create_subprocess_exec", spawn_mock)
    parsed = pilot.SolarPilotInput.model_validate(request())
    task = asyncio.create_task(tools._collect_isolated(parsed))
    try:
        await asyncio.wait_for(reap_started.wait(), timeout=1)
        assert process.killed and not process.reaped
        for _ in range(3):
            task.cancel()
            await asyncio.sleep(0)
            assert not task.done()
            assert slots.locked()
            assert directories[0].exists()
        refused = await tools._collect_isolated(parsed)
        assert refused == {"error": "solar pilot collection capacity exhausted"}
        spawn_mock.assert_awaited_once()
        permit_reap.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert process.reaped
        assert not directories[0].exists()
        assert slots._value == 1
    finally:
        permit_reap.set()
        await asyncio.gather(task, return_exceptions=True)
        slots.release()
