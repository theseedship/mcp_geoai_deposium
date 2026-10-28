"""Exercise GeoAI's HTTP boundary without network or data-provider calls."""

from __future__ import annotations

import json

import httpx
import pytest
from fastmcp import FastMCP

from src import server
from src.auth import build_mesh_auth
from src.contracts import tool_capabilities_json

_TOKEN = "geoai-http-test-secret-at-least-32-characters"
_HEADERS = {
    "Authorization": f"Bearer {_TOKEN}",
    "Accept": "application/json, text/event-stream",
}
_INITIALIZE = {
    "jsonrpc": "2.0",
    "id": 1,
    "method": "initialize",
    "params": {
        "protocolVersion": "2025-11-25",
        "capabilities": {},
        "clientInfo": {"name": "geoai-security-test", "version": "1"},
    },
}


@pytest.fixture
def authenticated_app():
    # Mount the production tools/resources rather than a synthetic tool registry. A fresh
    # boundary keeps this independent of import-time environment settings and other tests.
    boundary = FastMCP("GeoAI HTTP qualification", auth=build_mesh_auth(_TOKEN))
    boundary.mount(server.mcp)
    boundary.custom_route("/health", methods=["GET"])(server.health_check)
    return boundary.http_app(json_response=True)


@pytest.mark.parametrize("hostname", ["127.0.0.1:5001", "geoai.test", "healthcheck.railway.app"])
async def test_production_health_accepts_deployment_hosts_without_bearer(
    authenticated_app, hostname
):
    # The former 3.4.2 pin records a 3.4.3 regression: deployed /health returned 421.
    async with authenticated_app.router.lifespan_context(authenticated_app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=authenticated_app),
            base_url=f"http://{hostname}",
        ) as client:
            response = await client.get("/health")
    assert response.status_code == 200
    assert response.text == "OK"


@pytest.mark.parametrize("authorization", [None, "Bearer wrong-secret"])
async def test_http_initialize_rejects_missing_and_wrong_bearer(authenticated_app, authorization):
    headers = {"Accept": _HEADERS["Accept"]}
    if authorization is not None:
        headers["Authorization"] = authorization
    async with authenticated_app.router.lifespan_context(authenticated_app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=authenticated_app),
            base_url="http://geoai.test",
        ) as client:
            response = await client.post("/mcp", headers=headers, json=_INITIALIZE)
    assert response.status_code == 401
    assert "mcp-session-id" not in response.headers


async def test_authenticated_http_surface_and_deleted_session_are_enforced(authenticated_app):
    async with authenticated_app.router.lifespan_context(authenticated_app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=authenticated_app),
            base_url="http://geoai.test",
            headers=_HEADERS,
        ) as client:
            initialized = await client.post("/mcp", json=_INITIALIZE)
            assert initialized.status_code == 200
            assert initialized.json()["result"]["protocolVersion"] == "2025-11-25"
            session_id = initialized.headers["mcp-session-id"]
            client.headers.update(
                {"mcp-session-id": session_id, "mcp-protocol-version": "2025-11-25"}
            )
            ready = await client.post(
                "/mcp", json={"jsonrpc": "2.0", "method": "notifications/initialized"}
            )
            assert ready.status_code == 202

            listed = await client.post(
                "/mcp", json={"jsonrpc": "2.0", "id": 2, "method": "tools/list"}
            )
            assert listed.status_code == 200
            expected_tools = set(tool_capabilities_json()["tools"])
            assert len(expected_tools) == 16
            assert {tool["name"] for tool in listed.json()["result"]["tools"]} == expected_tools

            resource = await client.post(
                "/mcp",
                json={
                    "jsonrpc": "2.0",
                    "id": 3,
                    "method": "resources/read",
                    "params": {"uri": "geoai://capabilities/tools"},
                },
            )
            assert resource.status_code == 200
            assert json.loads(resource.json()["result"]["contents"][0]["text"]) == (
                tool_capabilities_json()
            )

            deleted = await client.delete("/mcp")
            assert deleted.status_code == 200
            stale = await client.post(
                "/mcp", json={"jsonrpc": "2.0", "id": 4, "method": "tools/list"}
            )
            assert stale.status_code == 404


async def test_http_session_capacity_is_reclaimed_after_delete():
    from mcp.server.lowlevel import Server
    from mcp.server.streamable_http_manager import StreamableHTTPSessionManager

    # The SDK owns session admission and cleanup. One slot exercises the security fix
    # through its public API without allocating thousands of sessions or waiting for expiry.
    manager = StreamableHTTPSessionManager(
        Server("bounded-session-test"), json_response=True, max_sessions=1
    )
    async with manager.run():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=manager.handle_request),
            base_url="http://127.0.0.1",
            headers={"Accept": _HEADERS["Accept"]},
        ) as client:
            first = await client.post("/mcp", json=_INITIALIZE)
            assert first.status_code == 200
            full = await client.post("/mcp", json=_INITIALIZE)
            assert full.status_code == 503
            removed = await client.delete(
                "/mcp", headers={"mcp-session-id": first.headers["mcp-session-id"]}
            )
            assert removed.status_code == 200
            replacement = await client.post("/mcp", json=_INITIALIZE)
            assert replacement.status_code == 200
            assert replacement.headers["mcp-session-id"] != first.headers["mcp-session-id"]
