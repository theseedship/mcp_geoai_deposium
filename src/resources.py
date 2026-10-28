"""MCP resources and prompts that guide agents through GeoAI workflows."""

import json
from collections.abc import Awaitable, Callable
from pathlib import Path

from fastmcp import FastMCP

from src.contracts import tool_capabilities_json
from src.manifests import dataset_manifest_contract, dataset_manifest_json_schema
from src.zone_catalogue import zone_catalogue

_REPO_ROOT = Path(__file__).resolve().parents[1]

SERVER_INSTRUCTIONS = """Deposium GeoAI is a standalone MCP server for geospatial and climate tools.
Use storage-free workflows first when no S3/RustFS store is configured:
forecast_query, stac_search, stac_download, ndvi_profile, vegetation_stress,
and change_detect. Data-plane tools and persisted climate outputs require
S3/RustFS or an explicitly enabled, root-confined local Parquet path:
spatial_query, discover_patterns, discover_patterns_cv, era5_ingest (when
write_output=true), ndvi_ingest (when write_output=true), climate_enrich_ndvi,
climate_percentiles, climate_normals, and pattern_alert. Read geoai://guide/agent before
planning multi-step work. MCP clients should read geoai://capabilities/tools for
profiles, persistence defaults, timeout classes, stability, and strict schemas.
Clients must read geoai://contracts/dataset-manifest-v1 before
producing or consuming spatial dataset manifests; the schema-only resource does
not enforce cross-field semantic rules.
solar_portfolio_collect is an experimental, default-off solar acquisition pilot:
validate 1-2 structured sites or collect complete profiles over two UTC days.
Collection requires server-recorded source access; durable private lots belong
to the caller, and this tool does not authorize user/space/chat access.
Prefer explicit tool names, inputs, and caveats in final answers."""

_AGENT_GUIDE = """# Deposium GeoAI Agent Guide

Use this guide before planning a task with the GeoAI MCP tools.

Connect any MCP client through HTTP, stdio, or legacy SSE using `MCP_TRANSPORT`.
The server runs independently. Storage-free tools need only their public data
providers. HTTP/SSE deployments that enable storage or solar collection require
a locally configured `GEOAI_MESH_AUTH_TOKEN`; stdio uses the launching process's
access. This token does not depend on an external gateway or application.

Read `geoai://capabilities/tools` before selecting or exposing tools to a client. It is the machine-readable source for profiles, persistence defaults,
descriptive timeout classes, stability status, and strict input/output schemas.
Those fields are metadata only: each runtime handler owns its authentication
and execution limits. The experimental `solar_portfolio_collect` wrapper
enforces a 30-second worker deadline separately from these metadata fields.

## First decision: storage-free or data-plane

Storage-free tools work without S3/RustFS:

- `forecast_query`: Open-Meteo forecast for a `zone_id` or explicit point.
- `stac_search`: public STAC catalog search.
- `stac_download`: STAC asset URL resolution.
- `ndvi_profile`: Sentinel-2 NDVI time series for a geometry.
- `vegetation_stress`: NDVI stress heuristic over a lookback window.
- `change_detect`: bbox-level vegetation gain/loss between two dates.

`solar_portfolio_collect` is a separate experimental acquisition pilot, disabled by
default for collection. `validate` checks 1-2 structured site/scenario rows and
two consecutive UTC dates without acquisition. `collect` requires server
activation through `SOLAR_MCP_ENABLED` and a source-specific access record in
`SOLAR_MCP_SOURCE_ACCESS_PROOFS`. It returns complete checksummed profiles
without durable storage. The caller owns source access authorization and durable
lot persistence. The tool does not grant access to private workbooks or lots.

Data-plane tools use an S3/RustFS store for normal persisted workflows. An
explicitly enabled local Parquet path is also supported when it is confined by
`LOCAL_PARQUET_SOURCE_ROOT`:

- `spatial_query`: GeoParquet spatial predicates.
- `discover_patterns`: so4gp gradual pattern mining over stored tabular data.
- `discover_patterns_cv`: cross-zone validation over B4 unified tables.
- `era5_ingest`: Open-Meteo Archive to GeoParquet when `write_output=true`.
- `ndvi_ingest`: Sentinel-2 NDVI zonal statistics to GeoParquet when
  `write_output=true`.
- `climate_enrich_ndvi`: unified ERA5 plus NDVI table.
- `climate_percentiles`: climatological P10/P50/P90 normals.
- `climate_normals`: read-only view of the stored normals, with an optional
  per-day classification of forecast values against the month of each day.
  Precipitation is banded against the wet-day distribution; `confidence` is
  always `null`.
- `pattern_alert`: experimental O3-library plus forecast scoring.

If no S3/RustFS or confined local path is configured, keep `write_output=false`
for archive/forecast experiments and avoid pattern-mining/storage tasks unless
a seeded store exists.
For persisted workflows, preserve authenticated tenant and space boundaries and
verify the dataset manifest before consuming stored data. Read
`geoai://contracts/dataset-manifest-v1` for structural and semantic requirements.
Keep data locators, access authorization, and scientific evidence distinct.

## Good answer structure

1. State which tool(s) you used and why.
2. Echo the important inputs: `zone_id` or bbox/geometry, dates, horizon,
   provider, cloud-cover limit, and whether output persistence was requested.
3. Summarize the result in domain language, not raw JSON only.
4. Surface caveats: cloud cover, coarse bbox-level change detection, missing S3,
   forecast provider fallback, or unavailable persisted sources.
5. Suggest the next tool only when it naturally follows from the result.

## Example storage-free tasks

- Forecast: call `forecast_query(zone_id="nairobi", forecast_days=7,
  write_output=false)`.
- STAC search: call `stac_search` with a small bbox, date range, collection,
  cloud-cover cap, and limit.
- NDVI profile: call `ndvi_profile` with a Point or Polygon geometry, dates,
  and `interval_days`.
- ERA5 probe without persistence: call `era5_ingest(..., write_output=false)`
  for a short period.

## Example data-plane task

With the dev harness seeded, call `discover_patterns(source="auto",
tenant_id="dev-tenant", space_id="dev-space", algorithm="graank",
min_support=0.5)` and explain the strongest gradual patterns.
"""

_DOC_RESOURCES: tuple[tuple[str, str, str, str], ...] = (
    (
        "geoai://docs/readme",
        "README",
        "Top-level project overview and tool inventory.",
        "README.md",
    ),
)


def _read_repo_file(relative_path: str) -> str:
    path = (_REPO_ROOT / relative_path).resolve()
    if not path.is_file() or _REPO_ROOT not in path.parents:
        raise FileNotFoundError(relative_path)
    return path.read_text(encoding="utf-8")


def _doc_reader(relative_path: str) -> Callable[[], Awaitable[str]]:
    async def read_doc() -> str:
        return _read_repo_file(relative_path)

    read_doc.__name__ = "read_" + relative_path.replace("/", "_").replace(".", "_")
    return read_doc


def register_agent_resources(mcp: FastMCP) -> None:
    """Register read-only resources/prompts that help MCP clients plan tasks."""

    @mcp.resource(
        "geoai://guide/agent",
        name="agent_guide",
        title="Deposium GeoAI Agent Guide",
        description="How to choose tools, structure answers, and handle missing S3/RustFS.",
        mime_type="text/markdown",
    )
    async def agent_guide() -> str:
        return _AGENT_GUIDE

    @mcp.resource(
        "geoai://capabilities/tools",
        name="tool_capabilities",
        title="GeoAI Tool Capabilities",
        description=(
            "Strict input/output schemas plus profile, persistence, timeout, and stability metadata."
        ),
        mime_type="application/json",
    )
    async def tool_capabilities() -> str:
        return json.dumps(tool_capabilities_json(), indent=2, sort_keys=True)

    @mcp.resource(
        "geoai://zones/catalogue",
        name="zone_catalogue",
        title="GeoAI Scientific Zone Catalogue",
        description=(
            "Read-only panel and pilot reference points at zone bbox midpoints; "
            "excludes client sites and personal GPS locations."
        ),
        mime_type="application/json",
    )
    async def scientific_zone_catalogue() -> str:
        return json.dumps(zone_catalogue(), indent=2, sort_keys=True, allow_nan=False)

    @mcp.resource(
        "geoai://schemas/dataset-manifest-v1",
        name="dataset_manifest_v1_schema",
        title="Dataset Manifest V1 Structural Schema",
        description=(
            "Structural JSON Schema; use the contract resource for required semantic rules."
        ),
        mime_type="application/schema+json",
    )
    async def dataset_manifest_v1_schema() -> str:
        return json.dumps(dataset_manifest_json_schema(), indent=2, sort_keys=True)

    @mcp.resource(
        "geoai://contracts/dataset-manifest-v1",
        name="dataset_manifest_v1_contract",
        title="Dataset Manifest V1 Contract",
        description=(
            "Normative structural schema, cross-field semantic rules, and conformance fixtures."
        ),
        mime_type="application/json",
    )
    async def dataset_manifest_v1_contract() -> str:
        return json.dumps(dataset_manifest_contract(), indent=2, sort_keys=True)

    for uri, name, description, relative_path in _DOC_RESOURCES:
        mcp.resource(
            uri,
            name=name,
            title=name,
            description=description,
            mime_type="text/markdown",
        )(_doc_reader(relative_path))

    @mcp.prompt(
        name="geoai_task",
        title="Plan and execute a GeoAI task",
        description="Prompt template for choosing GeoAI tools and reporting results.",
    )
    async def geoai_task(question: str, storage_available: bool = False) -> str:
        storage_note = (
            "S3/RustFS is available; persisted data-plane tools can be used."
            if storage_available
            else "Assume S3/RustFS is not available unless the user explicitly says it is."
        )
        return f"""Use the Deposium GeoAI MCP server to answer this task:

{question}

Before calling tools, read geoai://guide/agent if you need workflow guidance.
{storage_note}

When answering, mention the tool calls, key inputs, result summary, and caveats.
Prefer storage-free tools unless the task requires stored GeoParquet or so4gp
pattern mining."""
