"""FastMCP server entry point for GeoAI MCP Server."""

import structlog
from fastmcp import FastMCP
from starlette.requests import Request
from starlette.responses import PlainTextResponse

from src.auth import build_mesh_auth
from src.config import settings
from src.resources import SERVER_INSTRUCTIONS, register_agent_resources
from src.solar_pilot import SolarPilotSettings
from src.tools import (
    register_change_tools,
    register_climate_tools,
    register_forecast_tools,
    register_pattern_alert_tools,
    register_pattern_tools,
    register_solar_tools,
    register_spatial_tools,
    register_stac_tools,
)
from src.utils.duckdb_conn import get_duckdb_connection
from src.utils.logging import setup_logging
from src.utils.mcp_redaction import CredentialRedactionMiddleware
from src.utils.s3_endpoint import sanitize_stac_url

# Setup structured logging
setup_logging()
log = structlog.get_logger(__name__)

# Create MCP server
mcp = FastMCP(
    "Deposium GeoAI",
    instructions=SERVER_INSTRUCTIONS,
    auth=build_mesh_auth(settings.GEOAI_MESH_AUTH_TOKEN),
)

# A tool that catches an exception and answers {"error": str(e)} — or lets one escape — hands the
# client a message httpfs and pystac_client have already quoted the configured credential into
# (#148).
#
# Placement relative to register_*_tools below is irrelevant, contrary to what this comment used
# to claim: add_middleware only appends to FastMCP.middleware, and the chain is rebuilt from that
# list on every request. What DOES matter is order among middlewares — first added is outermost —
# so anything that rewrites tool results must be added before this one, never after, or it writes
# credentials back in after the scrub.
mcp.add_middleware(CredentialRedactionMiddleware())

# Register tools
register_stac_tools(mcp)
register_spatial_tools(mcp)
register_change_tools(mcp)
register_pattern_tools(mcp)
register_climate_tools(mcp)
register_forecast_tools(mcp)
register_pattern_alert_tools(mcp)
register_solar_tools(mcp)
register_agent_resources(mcp)


# Health check endpoint for Railway/Docker
@mcp.custom_route("/health", methods=["GET"])
async def health_check(request: Request) -> PlainTextResponse:
    """Health check endpoint for container orchestration."""
    return PlainTextResponse("OK")


def _data_plane_enabled() -> bool:
    return settings.ALLOW_LOCAL_PARQUET_SOURCE or any(
        value.strip()
        for value in (
            settings.S3_ENDPOINT,
            settings.S3_ACCESS_KEY,
            settings.S3_SECRET_KEY,
            settings.S3_BUCKET,
        )
    )


def _preflight_mesh_auth() -> None:
    """Reject unauthenticated HTTP data-plane startup before side effects."""
    if (
        settings.MCP_TRANSPORT.lower() != "stdio"
        and (_data_plane_enabled() or SolarPilotSettings().SOLAR_MCP_ENABLED)
        and not settings.GEOAI_MESH_AUTH_TOKEN.strip()
    ):
        raise RuntimeError(
            "GEOAI_MESH_AUTH_TOKEN is required for an HTTP/SSE data-plane deployment"
        )


def _prepare_mesh_extensions() -> bool:
    """Install/load worker extensions once, outside every tenant request deadline."""
    if not _data_plane_enabled():
        return False
    connection = get_duckdb_connection(
        ":memory:",
        load_httpfs=bool(settings.S3_BUCKET.strip()),
    )
    connection.close()
    return True


def main() -> None:
    """Run the MCP server on the configured transport.

    - http (default): mesh / Railway / docker — binds HOST:PORT.
    - stdio: Claude Desktop (or any MCP client) spawns this process; no socket.
    - sse: legacy SSE transport, also binds HOST:PORT.
    """
    transport = settings.MCP_TRANSPORT.lower()
    _stac_url = sanitize_stac_url(settings.STAC_URL)
    log.info(
        "Starting GeoAI MCP Server",
        transport=transport,
        host=settings.HOST,
        port=settings.PORT,
        # Sanitized, exactly like the run manifests. A private STAC deployment is reached as
        # `https://user:token@stac.internal/v1`, and a log line published into structlog, a
        # container log or a CI transcript is not a smaller audience than a manifest -- it is
        # usually a larger one. `sanitize_stac_url`, not `sanitize_endpoint`: a catalogue root
        # has a path (`.../v1`) and an object-store endpoint may not, so the two contracts
        # differ below the authority and are identical above it. `None` means STAC_URL is unset,
        # or that it does not match the grammar -- an unparseable URL is not logged at all.
        #
        # The reductions are logged beside it for the same reason the manifests publish them: an
        # operator comparing this line against the configured value must be able to tell a
        # dropped `?token=` from a value that was already an address.
        stac_url=_stac_url[0],
        stac_url_reductions=list(_stac_url[1]),
    )

    _preflight_mesh_auth()

    if _prepare_mesh_extensions():
        log.info("Prepared DuckDB extensions for isolated mesh workers")

    if transport == "stdio":
        mcp.run(transport="stdio")
    elif transport == "sse":
        mcp.run(transport="sse")
    else:
        mcp.run(transport=transport, host=settings.HOST, port=settings.PORT)


if __name__ == "__main__":
    main()
