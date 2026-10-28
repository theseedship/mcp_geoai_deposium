"""Functional checks run INSIDE a candidate GeoAI image, offline (#173).

``scripts/image_smoke.sh`` mounts this file and ``tests/fixtures/image-smoke`` into a
disposable ``docker run --network none`` container of the image and runs it from the image
WORKDIR (``/app``), exactly where ``python -m src.server`` runs. It needs only what the
runtime image ships: no pytest, no dev extra.

Every check exercises real code, not just imports, on tiny synthetic fixtures generated
here at run time. A tool that needs the network (STAC API, forecast provider, S3/RustFS)
is checked through its local compute path and reported as ``limitation`` with the reason.

Usage (inside the container): ``python /smoke/image_smoke_checks.py /smoke/fixtures``.
The last stdout line is ``IMAGE_SMOKE_JSON=<json>``; the exit code is 1 if any check fails.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any

APP_ROOT = Path.cwd()
FIXTURES = APP_ROOT / "tests/fixtures/image-smoke"
WORK = Path(tempfile.gettempdir()) / "image-smoke"  # replaced by a fresh directory in main()

CHECKS: list[tuple[str, Any]] = []


def check(name: str):
    def deco(fn):
        CHECKS.append((name, fn))
        return fn

    return deco


def _run(name: str, fn) -> dict[str, Any]:
    started = time.monotonic()
    try:
        status, detail = fn()
    except Exception as exc:  # noqa: BLE001 - a failed check is reported, not raised
        status = "fail"
        detail = {"error": f"{type(exc).__name__}: {exc}", "trace": traceback.format_exc()}
    return {
        "check": name,
        "status": status,
        "seconds": round(time.monotonic() - started, 3),
        "detail": detail,
    }


def _call(client, tool: str, arguments: dict[str, Any]) -> dict[str, Any]:
    async def run():
        async with client:
            result = await client.call_tool(tool, arguments, raise_on_error=False)
            return result.structured_content or {}

    return asyncio.run(run())


@check("package_origin")
def _package_origin():
    import src

    origin = Path(src.__file__).resolve()
    ok = origin.is_relative_to((APP_ROOT / "src").resolve())
    return ("pass" if ok else "fail"), {
        "src": origin.as_posix(),
        "python": sys.executable,
        "home": os.environ.get("HOME"),
    }


def _inline_refs(schema: Any, defs: dict[str, Any] | None = None) -> Any:
    """Inline local ``#/$defs/...`` references, as FastMCP does when it publishes a schema."""
    if isinstance(schema, dict):
        defs = schema.get("$defs", defs)
        ref = schema.get("$ref")
        if isinstance(ref, str) and ref.startswith("#/$defs/") and defs is not None:
            siblings = {k: v for k, v in schema.items() if k != "$ref"}
            return _inline_refs({**defs[ref.removeprefix("#/$defs/")], **siblings}, defs)
        return {k: _inline_refs(v, defs) for k, v in schema.items() if k != "$defs"}
    if isinstance(schema, list):
        return [_inline_refs(v, defs) for v in schema]
    return schema


# `-dev` packages that a kept runtime package hard-depends on, so apt cannot leave them out.
_REQUIRED_DEV_PACKAGES = {
    "python3-numpy-dev": (
        "headers-only Depends of python3-numpy, required by python3-gdal, a Depends of "
        "gdal-bin (Debian trixie); kept with the GDAL CLI"
    ),
}


@check("runtime_hygiene")
def _runtime_hygiene():
    tools = sorted(
        t for t in ("gcc", "cc", "g++", "c++", "make", "ld", "uv", "uvx") if shutil.which(t)
    )
    packages = subprocess.run(
        ["dpkg-query", "-W", "-f", "${Package}\n"], capture_output=True, text=True, check=True
    ).stdout.split()
    build_packages = sorted(
        p
        for p in packages
        if (p.endswith("-dev") and p not in _REQUIRED_DEV_PACKAGES)
        or p in {"build-essential", "gcc", "g++", "make"}
    )
    caches = sorted(
        p
        for p in ("/root/.cache", "/opt/duckdb-home/.cache", "/tmp/requirements.txt")
        if Path(p).exists()
    )
    ok = not tools and not build_packages and not caches
    return ("pass" if ok else "fail"), {
        "build_tools_on_path": tools,
        "build_packages": build_packages,
        "build_caches": caches,
        "dpkg_packages": len(packages),
        "required_dev_packages_present": sorted(set(packages) & set(_REQUIRED_DEV_PACKAGES)),
        "required_dev_packages_reason": _REQUIRED_DEV_PACKAGES,
    }


@check("tool_contracts")
def _tool_contracts():
    from fastmcp import Client

    from src.contracts import TOOL_CONTRACTS, output_schema_for, tool_capabilities_json
    from src.server import mcp

    async def list_tools():
        async with Client(mcp) as client:
            return await client.list_tools()

    tools = {tool.name: tool for tool in asyncio.run(list_tools())}
    problems = []
    if set(tools) != set(TOOL_CONTRACTS):
        problems.append(f"names differ: {sorted(set(tools) ^ set(TOOL_CONTRACTS))}")
    if set(tools) != set(tool_capabilities_json()["tools"]):
        problems.append("names differ from geoai://capabilities/tools")
    for name, tool in sorted(tools.items()):
        if name not in TOOL_CONTRACTS:
            continue
        if _inline_refs(tool.outputSchema) != _inline_refs(output_schema_for(name)):
            problems.append(f"{name}: output schema differs from its contract")
        expected = set(TOOL_CONTRACTS[name]["input"].model_fields)
        published = set((tool.inputSchema or {}).get("properties", {}))
        if published != expected:
            problems.append(f"{name}: input fields {sorted(published ^ expected)} differ")
    return ("fail" if problems else "pass"), {"tools": sorted(tools), "problems": problems}


@check("resources_and_readme")
def _resources():
    from fastmcp import Client

    from src.server import mcp

    async def read_all():
        async with Client(mcp) as client:
            resources = await client.list_resources()
            texts = {}
            for resource in resources:
                contents = await client.read_resource(resource.uri)
                texts[str(resource.uri)] = "".join(getattr(c, "text", "") or "" for c in contents)
            return texts

    texts = asyncio.run(read_all())
    empty = sorted(uri for uri, text in texts.items() if not text.strip())
    readme = texts.get("geoai://docs/readme", "")
    readme_ok = readme == (APP_ROOT / "README.md").read_text(encoding="utf-8")
    private_docs = sorted(
        uri for uri in texts if uri.startswith("geoai://docs/") and uri != "geoai://docs/readme"
    )
    ok = bool(texts) and not empty and readme_ok and not private_docs
    return ("pass" if ok else "fail"), {
        "resources_read": len(texts),
        "empty": empty,
        "readme_bytes": len(readme.encode("utf-8")),
        "readme_matches_app_README": readme_ok,
        "private_document_resources": private_docs,
    }


@check("gdal")
def _gdal():
    import rasterio

    cli = subprocess.run(["gdalinfo", "--version"], capture_output=True, text=True, check=True)
    drivers = rasterio.drivers.raster_driver_extensions()
    with rasterio.Env() as env:
        gtiff = "GTiff" in env.drivers()
    try:
        import osgeo  # noqa: F401

        osgeo_status = "present"
    except ImportError:
        osgeo_status = "absent (no GDAL Python bindings are locked; rasterio is the binding)"
    return ("pass" if gtiff else "fail"), {
        "gdalinfo_version": cli.stdout.strip(),
        "rasterio": rasterio.__version__,
        "rasterio_gdal": rasterio.__gdal_version__,
        "rasterio_gtiff_driver": gtiff,
        "raster_extensions": len(drivers),
        "osgeo": osgeo_status,
    }


@check("torch_cpu")
def _torch():
    from importlib.metadata import distributions

    import torch

    gpu = sorted(
        name
        for name in {(d.metadata["Name"] or "").lower() for d in distributions()}
        if name.startswith(("nvidia-", "cuda-")) or name == "triton"
    )
    ok = (
        torch.__version__ == "2.13.0+cpu"
        and not torch.cuda.is_available()
        and torch.version.cuda is None
        and not gpu
    )
    value = float(torch.tensor([1.0, 2.0, 3.0]).sum())
    return ("pass" if ok and value == 6.0 else "fail"), {
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "torch_version_cuda": torch.version.cuda,
        "gpu_distributions": gpu,
        "tensor_sum": value,
    }


@check("stac_rasterio_ndvi")
def _stac_ndvi():
    import numpy as np
    import pystac
    import rasterio
    from rasterio.transform import from_origin
    from rasterio.warp import transform

    from src.tools._ndvi import _compute_ndvi_for_item

    stac_dir = WORK / "stac"
    stac_dir.mkdir()
    shutil.copy(FIXTURES / "stac-item.json", stac_dir / "item.json")
    item = pystac.Item.from_file((stac_dir / "item.json").as_posix())
    # An 8x8, 10 m, UTM 31N raster centred on the fixture footprint: read_band reprojects
    # the WGS84 query geometry into the raster CRS, as for real Sentinel-2 tiles.
    west, south, east, north = item.bbox
    xs, ys = transform("EPSG:4326", "EPSG:32631", [(west + east) / 2], [(south + north) / 2])
    origin = from_origin(xs[0] - 40, ys[0] + 40, 10, 10)
    red = np.full((8, 8), 1000, dtype="uint16")
    nir = np.full((8, 8), 3000, dtype="uint16")
    for name, band in (("red", red), ("nir", nir)):
        with rasterio.open(
            stac_dir / f"{name}.tif",
            "w",
            driver="GTiff",
            width=8,
            height=8,
            count=1,
            dtype="uint16",
            crs="EPSG:32631",
            transform=origin,
            nodata=0,
        ) as dst:
            dst.write(band, 1)
    item.make_asset_hrefs_absolute()
    stats = _compute_ndvi_for_item(item, item.geometry)
    ok = stats is not None and abs(stats["ndvi_mean"] - 0.5) < 1e-9
    return ("pass" if ok else "fail"), {
        "item_id": item.id,
        "asset_hrefs": {key: asset.href for key, asset in item.assets.items()},
        "ndvi": stats,
        "expected_ndvi_mean": 0.5,
        "limitation": (
            "stac_search and stac_download query a remote STAC API and are not exercised "
            "offline; ndvi_profile, vegetation_stress and change_detect search the same API, "
            "so only their NDVI compute path (_compute_ndvi_for_item: STAC item fixture, "
            "rasterio read, NDVI) is exercised"
        ),
    }


@check("duckdb_extensions_offline")
def _duckdb():
    import duckdb

    from src.utils.duckdb_conn import get_duckdb_connection

    con = duckdb.connect(":memory:")
    try:
        con.execute("LOAD spatial")
        con.execute("LOAD httpfs")
        point = con.execute("SELECT ST_AsText(ST_Point(1, 2))").fetchone()[0]
        installed = con.execute(
            "SELECT extension_name, extension_version, install_path FROM duckdb_extensions() "
            "WHERE installed ORDER BY extension_name"
        ).fetchall()
    finally:
        con.close()
    # The server's startup path (INSTALL + LOAD) must also stay offline once cached.
    get_duckdb_connection(":memory:", load_httpfs=True).close()
    cached = [
        {"name": n, "version": v, "path": p} for n, v, p in installed if n in {"spatial", "httpfs"}
    ]
    under_home = all(e["path"].startswith(os.environ.get("HOME", "~")) for e in cached)
    ok = point == "POINT (1 2)" and len(cached) == 2 and under_home
    return ("pass" if ok else "fail"), {
        "duckdb": duckdb.__version__,
        "extensions": cached,
        "st_point": point,
        "under_home": under_home,
    }


def _write_mesh_parquet(path: Path) -> None:
    from src.utils.duckdb_conn import get_duckdb_connection

    con = get_duckdb_connection(":memory:", install_extensions=False)
    try:
        con.execute(
            """
            CREATE TABLE zones AS
            SELECT 'z0' AS zone_id, ST_AsWKB(ST_GeomFromText(
                'POLYGON((0 0, 1 0, 1 1, 0 1, 0 0))')) AS geometry
            UNION ALL
            SELECT 'z1', ST_AsWKB(ST_GeomFromText('POLYGON((5 5, 6 5, 6 6, 5 6, 5 5))'))
            """
        )
        con.execute(f"COPY zones TO '{path.as_posix()}' (FORMAT PARQUET)")
    finally:
        con.close()


@check("mesh_subprocess_deadline")
def _mesh():
    from fastmcp import Client

    from src.config import settings
    from src.server import mcp
    from src.tools import spatial

    source = WORK / "mesh.parquet"
    _write_mesh_parquet(source)
    arguments = {
        "source": source.as_posix(),
        "geometry": {"type": "Point", "coordinates": [0.5, 0.5]},
        "select_columns": ["geometry", "zone_id"],
        "limit": 5,
        "evidence_context": {
            "dataset_id": "image_smoke",
            "artifact_version": "sha256:" + "a" * 64,
            "scope_fingerprint": "sha256:" + "b" * 64,
            "query_fingerprint": "sha256:" + "c" * 64,
            "geometry_column": "geometry",
            "crs": "EPSG:4326",
        },
    }
    started = time.monotonic()
    ok_result = _call(Client(mcp), "spatial_query", arguments)
    elapsed = time.monotonic() - started
    rows = (ok_result.get("evidence") or {}).get("selection", {}).get("row_offsets")
    configured = settings.SPATIAL_MESH_QUERY_TIMEOUT_S
    # The same real worker under a deadline it cannot meet must be killed, not awaited.
    settings.SPATIAL_MESH_QUERY_TIMEOUT_S = 0.001
    try:
        timeout_result = _call(Client(mcp), "spatial_query", arguments)
    finally:
        settings.SPATIAL_MESH_QUERY_TIMEOUT_S = configured
    ok = (
        rows == [0]
        and elapsed < configured
        and timeout_result.get("error_code") == "SPATIAL_QUERY_TIMEOUT"
    )
    return ("pass" if ok else "fail"), {
        "worker_command": list(spatial._MESH_WORKER_COMMAND),
        "row_offsets": rows,
        "features": len(ok_result.get("features") or []),
        "error": ok_result.get("error_code"),
        "seconds": round(elapsed, 3),
        "configured_deadline_s": configured,
        "forced_deadline_result": timeout_result.get("error_code"),
    }


def _write_zone_tables(root: Path, zones: list[str]) -> None:
    import pandas as pd

    from src.utils.duckdb_conn import duckdb_session

    root.mkdir(parents=True)
    for index, zone in enumerate(zones):
        days = 12
        frame = pd.DataFrame(
            {
                "date": pd.date_range("2026-06-01", periods=days, freq="D"),
                "temperature_2m": [18.0 + 0.6 * d + 0.1 * index for d in range(days)],
                "evapotranspiration": [2.0 + 0.3 * d + 0.05 * index for d in range(days)],
                "ndvi_mean": [0.70 - 0.02 * d for d in range(days)],
            }
        )
        with duckdb_session(":memory:") as con:
            con.register("frame", frame)
            con.execute(f"COPY frame TO '{(root / f'{zone}.parquet').as_posix()}' (FORMAT PARQUET)")


@check("patterns_mining_cv_alert")
def _patterns():
    from fastmcp import Client

    from src.config import settings
    from src.server import mcp
    from src.tools import pattern_alert as alert_module
    from src.tools import patterns as patterns_module

    zones = ["z1", "z2", "z3", "z4", "z5"]
    data = WORK / "patterns" / "climate"
    _write_zone_tables(data, zones)
    bucket_prefix = "s3://image-smoke/"
    real_load = patterns_module.load_dataframe
    real_write = patterns_module._write_validated_patterns_parquet

    def local(uri: str) -> str:
        # The only substitution: the RustFS bucket is mapped onto local files. DuckDB,
        # so4gp, cross-validation and the parquet writer below are the real ones.
        return (WORK / "patterns" / uri.removeprefix(bucket_prefix)).as_posix()

    def load_dataframe(uri, **kwargs):
        return real_load(local(uri), **kwargs)

    def write_validated(library, uri):
        Path(local(uri)).parent.mkdir(parents=True, exist_ok=True)  # an S3 key has no parent
        return real_write(library, local(uri))

    async def synthetic_forecast(**_kwargs):
        rows = [
            {
                "date": f"2026-07-{d + 1:02d}",
                "temperature_2m": 25.0 + d,
                "evapotranspiration": 3.0 + d,
            }
            for d in range(7)
        ]
        return {
            "zone_id": "z1",
            "forecast": rows,
            "metadata": {"source": "image_smoke_synthetic", "horizon_days": len(rows)},
        }

    saved = (
        settings.S3_BUCKET,
        patterns_module.load_dataframe,
        patterns_module._write_validated_patterns_parquet,
        alert_module._run_forecast_query,
    )
    settings.S3_BUCKET = "image-smoke"
    patterns_module.load_dataframe = load_dataframe
    patterns_module._write_validated_patterns_parquet = write_validated
    alert_module._run_forecast_query = synthetic_forecast
    try:
        mined = _call(
            Client(mcp),
            "discover_patterns",
            {
                "source": "climate/z1.parquet",
                "order_by": "date",
                "min_support": 0.5,
                "write_output": False,
            },
        )
        cv = _call(
            Client(mcp),
            "discover_patterns_cv",
            {
                "zone_ids": zones,
                "source_template": "climate/{zone_id}.parquet",
                "min_support": 0.5,
                "n_folds": 5,
                "write_output": True,
                "output_key": "o3/validated.parquet",
            },
        )
        library = WORK / "patterns" / "o3" / "validated.parquet"
        alert = _call(
            Client(mcp),
            "pattern_alert",
            {
                "zone_id": "z1",
                "patterns_source": library.as_posix(),
                "allow_local_source": True,
                "forecast_horizon_days": 7,
            },
        )
    finally:
        (
            settings.S3_BUCKET,
            patterns_module.load_dataframe,
            patterns_module._write_validated_patterns_parquet,
            alert_module._run_forecast_query,
        ) = saved
    n_mined = len(mined.get("patterns") or [])
    n_validated = (cv.get("statistics") or {}).get("n_validated_patterns", 0)
    n_alerts = (alert.get("statistics") or {}).get("n_alerts", 0)
    ok = (
        "error" not in mined
        and "error" not in cv
        and "error" not in alert
        and n_mined > 0
        and n_validated > 0
        and library.is_file()
        and n_alerts > 0
    )
    return ("pass" if ok else "fail"), {
        "discover_patterns": {"patterns": n_mined, "error": mined.get("error")},
        "discover_patterns_cv": {"validated": n_validated, "error": cv.get("error")},
        "pattern_alert": {
            "alerts": n_alerts,
            "top_action": ((alert.get("alerts") or [{}])[0]).get("recommended_action"),
            "error": alert.get("error"),
        },
        "limitation": (
            "the RustFS/S3 source and output are mapped onto local parquet files, and "
            "pattern_alert's forecast_query (remote provider) returns a synthetic forecast; "
            "mining (so4gp), cross-validation, the parquet writer and alert scoring are real"
        ),
    }


def main() -> int:
    global FIXTURES, WORK
    if len(sys.argv) > 1:
        FIXTURES = Path(sys.argv[1])
    WORK = Path(tempfile.mkdtemp(prefix="image-smoke-"))
    # Before any `src` import: the settings object reads the environment once. Local parquet
    # sources are confined to the throwaway work directory.
    os.environ["ALLOW_LOCAL_PARQUET_SOURCE"] = "true"
    os.environ["LOCAL_PARQUET_SOURCE_ROOT"] = WORK.as_posix()
    # Import `src` from the working directory, as `python -m src.server` does, not from the
    # copy installed in the virtualenv (whose force-included README would mask a missing one).
    sys.path.insert(0, APP_ROOT.as_posix())
    from src.utils.logging import setup_logging

    # The tool handlers log errors; install the credential redactor as the server does.
    setup_logging()
    results = [_run(name, fn) for name, fn in CHECKS]
    failed = [r["check"] for r in results if r["status"] == "fail"]
    report = {
        "checks": results,
        "summary": {
            "pass": sum(r["status"] == "pass" for r in results),
            "limitation": sum("limitation" in r["detail"] for r in results),
            "fail": len(failed),
            "failed": failed,
        },
    }
    shutil.rmtree(WORK, ignore_errors=True)
    print("IMAGE_SMOKE_JSON=" + json.dumps(report, sort_keys=True, default=str))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
