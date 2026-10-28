"""Bounded solar acquisition tool using a killable isolated worker."""

from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
from typing import Annotated, Any, Literal

from pydantic import Field

from src.config import settings
from src.contracts import output_schema_for
from src.solar_contracts import SolarResolution, SolarSource, SolarVariable
from src.solar_pilot import (
    MAX_REQUEST_BYTES,
    MAX_RESPONSE_BYTES,
    SolarPilotInput,
    SolarPilotOutput,
    SolarPilotSettings,
    SolarPilotSite,
    run_solar_pilot,
)

PILOT_TIMEOUT_SECONDS = 30.0
PILOT_MAX_WORKERS = 2
_COLLECTION_SLOTS = asyncio.Semaphore(PILOT_MAX_WORKERS)


async def _collect_isolated(request: SolarPilotInput) -> dict[str, Any]:
    encoded = json.dumps(request.model_dump(mode="json"), ensure_ascii=False).encode("utf-8")
    if len(encoded) > MAX_REQUEST_BYTES:
        return {"error": "solar pilot request exceeds byte limit"}
    # There is no queued admission: acquire completes without suspension while
    # unlocked, so no other task can enter between this check and acquisition.
    if _COLLECTION_SLOTS.locked():
        return {"error": "solar pilot collection capacity exhausted"}
    await _COLLECTION_SLOTS.acquire()
    try:
        with tempfile.TemporaryDirectory(prefix="geoai-solar-worker-") as temporary:
            try:
                process = await asyncio.create_subprocess_exec(
                    sys.executable,
                    "-m",
                    "src.solar_pilot",
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.DEVNULL,
                    env={**os.environ, "GEOAI_SOLAR_WORK_DIR": temporary},
                )
            except Exception:
                return {"error": "solar pilot worker unavailable"}

            async def exchange() -> bytes:
                assert process.stdin is not None and process.stdout is not None
                process.stdin.write(encoded)
                await process.stdin.drain()
                process.stdin.close()
                chunks = []
                size = 0
                while chunk := await process.stdout.read(min(65536, MAX_RESPONSE_BYTES + 1 - size)):
                    chunks.append(chunk)
                    size += len(chunk)
                    if size > MAX_RESPONSE_BYTES:
                        raise ValueError("oversized solar response")
                await process.wait()
                return b"".join(chunks)

            try:
                raw = await asyncio.wait_for(exchange(), timeout=PILOT_TIMEOUT_SECONDS)
                if process.returncode != 0:
                    return {"error": "solar pilot worker failed"}
                result = json.loads(raw)
                if isinstance(result, dict) and set(result) == {"error"}:
                    return {"error": "solar pilot collection failed"}
                return SolarPilotOutput.model_validate(result).model_dump(
                    mode="json", exclude={"error"}
                )
            except asyncio.CancelledError:
                raise
            except TimeoutError:
                return {"error": "solar pilot collection timed out"}
            except Exception:
                return {"error": "solar pilot worker failed"}
            finally:
                if process.returncode is None:
                    try:
                        process.kill()
                    except ProcessLookupError:
                        pass
                # Keep ownership until the child is reaped, even if the caller
                # cancels again while cleanup is waiting for process termination.
                reap = asyncio.create_task(process.wait())
                cancelled_during_reap = False
                while not reap.done():
                    try:
                        await asyncio.shield(reap)
                    except asyncio.CancelledError:
                        cancelled_during_reap = True
                reap.result()
                if cancelled_during_reap:
                    raise asyncio.CancelledError
    finally:
        _COLLECTION_SLOTS.release()


async def solar_portfolio_collect(
    action: Literal["validate", "collect"],
    sites: Annotated[list[SolarPilotSite], Field(min_length=1, max_length=2)],
    source: SolarSource,
    start_date: str,
    end_date: str,
    variables: Annotated[list[SolarVariable], Field(min_length=1)],
    model: str | None = None,
    temporal_resolution: SolarResolution | None = None,
) -> dict[str, Any]:
    """Validate or collect at most two sites over two consecutive UTC dates.

    Collection requires server activation and access proof. Returns exact provider
    profiles for caller-owned persistence. A timed-out call has no usable result;
    the caller checkpoints successful queries individually before further work.
    """
    try:
        request = SolarPilotInput.model_validate(
            {
                "action": action,
                "sites": sites,
                "source": source,
                "start_date": start_date,
                "end_date": end_date,
                "variables": variables,
                "model": model,
                "temporal_resolution": temporal_resolution,
            }
        )
        if action == "validate":
            return run_solar_pilot(request)
        # Auth is configured once when the server is constructed. A reloadable
        # activation file cannot enable collection on an unauthenticated HTTP server.
        if settings.MCP_TRANSPORT.lower() != "stdio" and not settings.GEOAI_MESH_AUTH_TOKEN.strip():
            return {"error": "solar pilot collection requires configured mesh authentication"}
        SolarPilotSettings().require_access(request.source)
    except Exception:
        return {"error": "solar pilot input or source activation rejected"}
    return await _collect_isolated(request)


def register_solar_tools(mcp: Any) -> None:
    mcp.tool(output_schema=output_schema_for("solar_portfolio_collect"))(solar_portfolio_collect)
