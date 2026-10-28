"""Hermetic guards for the live-acceptance harness wiring (#171).

The documented command, ``docker compose exec geoai-mcp python tests/live_acceptance.py``,
could not work: the script was not in the container, and the server (which the compose gives
``GEOAI_MESH_AUTH_TOKEN``) rejects a caller with no bearer. Nothing here starts a server or
touches the network; it checks the two pieces of wiring that were missing.
"""

from __future__ import annotations

import asyncio
import importlib.util
import pathlib
import socket
import threading
import time

import httpx
import pytest
import yaml
from fastmcp.client.auth.bearer import BearerAuth

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
_HARNESS = _REPO_ROOT / "tests" / "live_acceptance.py"
_TOKEN_VAR = "GEOAI_MESH_AUTH_TOKEN"


@pytest.fixture
def harness():
    spec = importlib.util.spec_from_file_location("live_acceptance_under_test", _HARNESS)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _geoai_service() -> dict:
    compose = yaml.safe_load((_REPO_ROOT / "docker-compose.yml").read_text(encoding="utf-8"))
    return compose["services"]["geoai-mcp"]


def test_compose_mounts_the_harness_read_only_where_the_documented_command_finds_it():
    dockerfile = (_REPO_ROOT / "Dockerfile").read_text(encoding="utf-8")
    workdir = next(
        line.split(None, 1)[1].strip()
        for line in dockerfile.splitlines()
        if line.startswith("WORKDIR ")
    )
    # `python tests/live_acceptance.py` is relative to the exec working directory, which is
    # the image WORKDIR.
    expected = f"./tests/live_acceptance.py:{workdir}/tests/live_acceptance.py:ro"
    assert expected in _geoai_service()["volumes"]
    assert (_REPO_ROOT / "tests" / "live_acceptance.py").is_file()


def test_the_container_the_command_runs_in_has_the_token_variable():
    env = _geoai_service()["environment"]
    assert any(str(entry).startswith(f"{_TOKEN_VAR}=") for entry in env)


def test_client_sends_the_env_token_as_a_bearer(harness, monkeypatch):
    monkeypatch.setenv(_TOKEN_VAR, "  s3cret-token-value  ")
    client = harness._client()
    auth = client.transport.auth
    assert isinstance(auth, BearerAuth)
    request = _FakeRequest()
    next(auth.auth_flow(request))
    assert request.headers["Authorization"] == "Bearer s3cret-token-value"


@pytest.mark.parametrize("value", [None, "", "   "])
def test_client_connects_without_auth_when_the_token_is_unset_or_blank(harness, monkeypatch, value):
    if value is None:
        monkeypatch.delenv(_TOKEN_VAR, raising=False)
    else:
        monkeypatch.setenv(_TOKEN_VAR, value)
    assert harness._auth() is None
    assert harness._client().transport.auth is None


def _http_401() -> httpx.HTTPStatusError:
    request = httpx.Request("POST", "http://localhost:5001/mcp")
    response = httpx.Response(401, request=request)
    return httpx.HTTPStatusError("401 Unauthorized", request=request, response=response)


@pytest.mark.parametrize(
    "error",
    [
        pytest.param(_http_401(), id="http-401"),
        pytest.param(
            RuntimeError("Client failed to connect: All connection attempts failed"),
            id="connect-failed",
        ),
    ],
)
def test_a_rejected_connection_stops_before_the_store_is_written(
    harness, monkeypatch, capsys, error
):
    """The two exceptions the real client raises for a bad bearer and for a closed port."""
    calls: list[str] = []

    async def rejected():
        raise error

    monkeypatch.setattr(harness, "preflight", rejected)
    monkeypatch.setattr(harness, "setup_geoparquet", lambda: calls.append("setup"))
    monkeypatch.setattr(harness, "run_tests", lambda: calls.append("run"))
    monkeypatch.setenv(_TOKEN_VAR, "s3cret-token-value")

    with pytest.raises(SystemExit) as exc:
        harness.main()

    assert exc.value.code == 2
    assert calls == []
    captured = capsys.readouterr()
    assert "s3cret-token-value" not in captured.out
    assert "s3cret-token-value" not in captured.err


class _Sentinel(Exception):
    pass


class _FailingClient:
    """Stands in for ``harness._client()``: entering the session raises the sentinel."""

    async def __aenter__(self):
        raise _Sentinel

    async def __aexit__(self, *exc_info):
        return False


def _closed_loopback_url() -> str:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    return f"http://127.0.0.1:{port}/mcp"


def test_preflight_connects_through_the_client_and_lets_its_failure_out(harness, monkeypatch):
    monkeypatch.setattr(harness, "_client", lambda: _FailingClient())
    with pytest.raises(_Sentinel):
        asyncio.run(harness.preflight())


def test_the_cases_connect_through_the_authenticated_client(harness, monkeypatch):
    # A closed port: a run_tests() that built its own unauthenticated Client would raise a
    # connection error instead of the sentinel.
    monkeypatch.setattr(harness, "MCP_URL", _closed_loopback_url())
    monkeypatch.setattr(harness, "_client", lambda: _FailingClient())
    with pytest.raises(_Sentinel):
        asyncio.run(harness.run_tests())


def test_a_peer_that_never_answers_ends_in_exit_2_within_the_preflight_limit(harness, monkeypatch):
    calls: list[str] = []
    with socket.socket() as server:
        server.bind(("127.0.0.1", 0))
        server.listen()
        accepted: list[socket.socket] = []

        def accept_and_stay_silent() -> None:
            try:
                accepted.append(server.accept()[0])
            except OSError:
                pass

        thread = threading.Thread(target=accept_and_stay_silent, daemon=True)
        thread.start()
        monkeypatch.setattr(harness, "MCP_URL", f"http://127.0.0.1:{server.getsockname()[1]}/mcp")
        monkeypatch.setattr(harness, "PREFLIGHT_TIMEOUT_S", 1)
        monkeypatch.setattr(harness, "setup_geoparquet", lambda: calls.append("setup"))
        monkeypatch.setattr(harness, "run_tests", lambda: calls.append("run"))

        started = time.monotonic()
        with pytest.raises(SystemExit) as exc:
            harness.main()
        elapsed = time.monotonic() - started
        for conn in accepted:
            conn.close()

    assert exc.value.code == 2
    assert elapsed < 10
    assert calls == []


def test_setup_runs_only_after_a_successful_connection(harness, monkeypatch):
    calls: list[str] = []

    async def ok():
        calls.append("preflight")

    async def run():
        calls.append("run")

    monkeypatch.setattr(harness, "preflight", ok)
    monkeypatch.setattr(harness, "setup_geoparquet", lambda: calls.append("setup"))
    monkeypatch.setattr(harness, "run_tests", run)

    harness.main()

    assert calls == ["preflight", "setup", "run"]


class _FakeRequest:
    def __init__(self) -> None:
        self.headers: dict[str, str] = {}
