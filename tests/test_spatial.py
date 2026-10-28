"""Network-free tests for DuckDB spatial_query."""

from __future__ import annotations

import asyncio
import json
import math
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from threading import Event

import pytest
from pydantic import ValidationError
from rasterio.warp import transform as rasterio_transform

from src.config import settings
from src.contracts import EvidenceContextV1, SpatialQueryInput
from src.tools import spatial
from src.tools.spatial import (
    _legacy_geometry_expression,
    _mesh_geometry_expression,
    _run_legacy_spatial_query,
    register_spatial_tools,
)
from src.utils.duckdb_conn import get_duckdb_connection

FORMAL_GEOPARQUET = Path(__file__).parent / "fixtures/gai3/formal-wkb.parquet"

# The exact strings `_legacy_geometry_expression` raises, pinned so that one cannot stand in for
# another.
UNSUPPORTED_TYPE_MESSAGE = "unsupported geometry column type"
UNSUPPORTED_CRS_MESSAGE = (
    "unsupported geometry column CRS: a native GEOMETRY column must be EPSG:4326, OGC:CRS84 "
    "or EPSG:2154"
)
MISSING_CRS_MESSAGE = (
    "unsupported geometry column CRS: a native GEOMETRY column declares no CRS and must be "
    "EPSG:4326, OGC:CRS84 or EPSG:2154"
)


class _CaptureMCP:
    """Records tool handlers without a live FastMCP server."""

    def __init__(self) -> None:
        self.tools: dict[str, object] = {}

    def tool(self, *_args, **_kwargs):
        def _decorator(fn):
            self.tools[fn.__name__] = fn
            return fn

        return _decorator


def _load_spatial_tools() -> dict:
    cap = _CaptureMCP()
    register_spatial_tools(cap)
    return cap.tools


def _write_wkb_parquet(path: Path) -> None:
    con = get_duckdb_connection(":memory:")
    try:
        con.execute(
            """
            CREATE TABLE zones AS
            SELECT
                'z1' AS zone_id,
                'Zone 1' AS name,
                ST_AsWKB(ST_GeomFromText(
                    'POLYGON((0 0, 1 0, 1 1, 0 1, 0 0))'
                )) AS geometry
            UNION ALL
            SELECT
                'z2' AS zone_id,
                'Zone 2' AS name,
                ST_AsWKB(ST_GeomFromText(
                    'POLYGON((5 5, 6 5, 6 6, 5 6, 5 5))'
                )) AS geometry
            """
        )
        con.execute(f"COPY zones TO '{path.as_posix()}' (FORMAT PARQUET)")
    finally:
        con.close()


def _write_mesh_parquet(path: Path, *, geometry_column: str = "geometry") -> None:
    con = get_duckdb_connection(":memory:")
    try:
        con.execute(
            f'''
            CREATE TABLE zones AS
            SELECT 'z0' AS zone_id, 'Zero' AS name, ST_AsWKB(ST_GeomFromText(
                'POLYGON((0 0, 1 0, 1 1, 0 1, 0 0))')) AS "{geometry_column}"
            UNION ALL
            SELECT 'z1', 'One', ST_AsWKB(ST_GeomFromText(
                'POLYGON((0 0, 1 0, 1 1, 0 1, 0 0))'))
            UNION ALL
            SELECT 'z2', 'Two', ST_AsWKB(ST_GeomFromText(
                'POLYGON((5 5, 6 5, 6 6, 5 6, 5 5))'))
            '''
        )
        con.execute(f"COPY zones TO '{path.as_posix()}' (FORMAT PARQUET)")
    finally:
        con.close()


def _write_typed_mesh_parquet(path: Path) -> None:
    con = get_duckdb_connection(":memory:")
    try:
        con.execute(
            """
            CREATE TABLE typed_zone AS
            SELECT
                ST_AsWKB(ST_GeomFromText(
                    'POLYGON((0 0, 1 0, 1 1, 0 1, 0 0))'
                )) AS geometry,
                TIMESTAMP '2026-07-12 09:10:11.123456' AS observed_at,
                DECIMAL '12.340' AS score,
                UUID '12345678-1234-5678-1234-567812345678' AS record_id,
                BLOB 'AB' AS payload
            """
        )
        con.execute(f"COPY typed_zone TO '{path.as_posix()}' (FORMAT PARQUET)")
    finally:
        con.close()


def _write_live_acceptance_geoparquet(path: Path) -> None:
    """Write the file `tests/live_acceptance.py::setup_geoparquet` uploads, to a local path.

    Same table and same `COPY ... (FORMAT PARQUET)` from a `GEOMETRY` column: with the spatial
    extension loaded DuckDB tags it as GeoParquet and reads it back as a native `GEOMETRY`.
    """
    con = get_duckdb_connection(":memory:")
    try:
        con.execute("""
            CREATE TABLE cities AS
            SELECT * FROM (VALUES
                ('Paris',       'France',  2161000, ST_GeomFromText('POLYGON((2.25 48.81, 2.42 48.81, 2.42 48.90, 2.25 48.90, 2.25 48.81))')),
                ('Lyon',        'France',   516000, ST_GeomFromText('POLYGON((4.80 45.72, 4.90 45.72, 4.90 45.80, 4.80 45.80, 4.80 45.72))')),
                ('Marseille',   'France',   870000, ST_GeomFromText('POLYGON((5.33 43.25, 5.42 43.25, 5.42 43.33, 5.33 43.33, 5.33 43.25))')),
                ('Toulouse',    'France',   493000, ST_GeomFromText('POLYGON((1.38 43.55, 1.50 43.55, 1.50 43.65, 1.38 43.65, 1.38 43.55))')),
                ('Bordeaux',    'France',   260000, ST_GeomFromText('POLYGON((-0.63 44.81, -0.53 44.81, -0.53 44.88, -0.63 44.88, -0.63 44.81))')),
                ('Montpellier', 'France',   295000, ST_GeomFromText('POLYGON((3.82 43.58, 3.93 43.58, 3.93 43.65, 3.82 43.65, 3.82 43.58))')),
                ('Nantes',      'France',   320000, ST_GeomFromText('POLYGON((-1.60 47.19, -1.50 47.19, -1.50 47.26, -1.60 47.26, -1.60 47.19))')),
                ('Strasbourg',  'France',   284000, ST_GeomFromText('POLYGON((7.72 48.55, 7.82 48.55, 7.82 48.62, 7.72 48.62, 7.72 48.55))')),
                ('Lille',       'France',   236000, ST_GeomFromText('POLYGON((3.02 50.60, 3.10 50.60, 3.10 50.66, 3.02 50.66, 3.02 50.60))')),
                ('Nice',        'France',   342000, ST_GeomFromText('POLYGON((7.22 43.68, 7.30 43.68, 7.30 43.73, 7.22 43.73, 7.22 43.68))'))
            ) AS t(name, country, population, geometry)
        """)
        con.execute(f"COPY cities TO '{path.as_posix()}' (FORMAT PARQUET)")
    finally:
        con.close()


def _read_geometry_type(path: Path) -> str:
    con = get_duckdb_connection(":memory:")
    try:
        described = con.execute(
            "DESCRIBE SELECT geometry FROM read_parquet($source)", {"source": path.as_posix()}
        ).fetchone()
        return str(described[1])
    finally:
        con.close()


def _allow_local_sources(monkeypatch, root: Path) -> None:
    monkeypatch.setattr(settings, "ALLOW_LOCAL_PARQUET_SOURCE", True)
    monkeypatch.setattr(settings, "LOCAL_PARQUET_SOURCE_ROOT", root.as_posix())


def _mesh_context(geometry_column: str = "geometry") -> dict:
    return EvidenceContextV1(
        dataset_id="native_spatial_demo",
        artifact_version="sha256:" + "a" * 64,
        scope_fingerprint="sha256:" + "b" * 64,
        query_fingerprint="sha256:" + "c" * 64,
        geometry_column=geometry_column,
        crs="EPSG:4326",
    ).model_dump(mode="json")


@pytest.mark.parametrize(
    ("encoding", "mesh"),
    [
        ("WKB", False),
        ("EPSG:4326", False),
        ("OGC:CRS84", False),
        ("WKB", True),
        ("EPSG:4326", True),
    ],
)
@pytest.mark.parametrize("query_coordinates", [(2.35, 48.86), (2.36, 48.85)])
def test_distance_uses_lon_lat_web_mercator(
    monkeypatch, tmp_path, encoding, mesh, query_coordinates
):
    # Independent EPSG:3857 formula: a northward step at Paris is ~1692 m;
    # an eastward step is ~1113 m. Neither is a raw-degree or geodesic distance.
    longitude, latitude = 2.35, 48.85
    query_longitude, query_latitude = query_coordinates
    radius = 6_378_137
    delta_x = radius * math.radians(query_longitude - longitude)
    delta_y = radius * (
        math.log(math.tan(math.pi / 4 + math.radians(query_latitude) / 2))
        - math.log(math.tan(math.pi / 4 + math.radians(latitude) / 2))
    )
    expected_distance = math.hypot(delta_x, delta_y)
    src = tmp_path / "point.parquet"
    point = "ST_Point(2.35, 48.85)"
    geometry_expression = (
        f"ST_AsWKB({point})" if encoding == "WKB" else f"ST_SetCRS({point}, '{encoding}')"
    )
    con = get_duckdb_connection(":memory:")
    try:
        con.execute(
            f"COPY (SELECT 'paris' AS zone_id, {geometry_expression} AS geometry) "
            "TO ? (FORMAT PARQUET)",
            [src.as_posix()],
        )
    finally:
        con.close()
    assert _read_geometry_type(src) == ("BLOB" if encoding == "WKB" else f"GEOMETRY('{encoding}')")
    _allow_local_sources(monkeypatch, tmp_path)
    tool = _load_spatial_tools()["spatial_query"]

    # Bracket the formula within 10 cm so swapped axes fail in both directions,
    # including the smaller eastward projection error. Exercise the real runners.
    for threshold, expected_ids in [
        (expected_distance - 0.1, []),
        (expected_distance + 0.1, ["paris"]),
    ]:
        result = asyncio.run(
            tool(
                source=src.as_posix(),
                geometry={"type": "Point", "coordinates": list(query_coordinates)},
                operation="distance",
                distance_m=threshold,
                select_columns=["zone_id", "geometry"],
                evidence_context=_mesh_context() if mesh else None,
            )
        )
        assert "error" not in result, result
        assert [feature["properties"]["zone_id"] for feature in result["features"]] == (
            expected_ids
        ), (threshold, expected_distance, result)
        if expected_ids:
            assert result["features"][0]["geometry"] == {
                "type": "Point",
                "coordinates": [longitude, latitude],
            }
        if mesh:
            assert result["evidence"]["operation"] == "spatial.distance"
            assert result["evidence"]["selection"]["row_offsets"] == ([0] if expected_ids else [])


class TestSpatialQuery:
    @pytest.mark.parametrize(
        ("described_type", "expected_expression"),
        [
            ("GEOMETRY('EPSG:4326')", '"geometry"'),
            ("BLOB", 'ST_GeomFromWKB("geometry")'),
        ],
    )
    def test_mesh_accepts_supported_geometry_types(
        self, described_type: str, expected_expression: str
    ) -> None:
        assert _mesh_geometry_expression(described_type, '"geometry"') == expected_expression

    @pytest.mark.parametrize(
        "described_type",
        [
            "GEOMETRY",
            "GEOMETRY('EPSG:3857')",
            "GEOMETRY('ESRI:102003')",
            "GEOMETRY('EPSG:4326'",
        ],
    )
    def test_mesh_rejects_unknown_or_malformed_geometry_types(self, described_type: str) -> None:
        with pytest.raises(ValueError, match="unsupported mesh geometry column type"):
            _mesh_geometry_expression(described_type, '"geometry"')

    def test_mesh_returns_physical_offsets_and_native_evidence(self, monkeypatch, tmp_path):
        src = tmp_path / "zones.parquet"
        _write_mesh_parquet(src)
        _allow_local_sources(monkeypatch, tmp_path)

        result = asyncio.run(
            _load_spatial_tools()["spatial_query"](
                source=src.as_posix(),
                geometry={"type": "Point", "coordinates": [0.5, 0.5]},
                select_columns=["geometry", "zone_id", "name"],
                limit=2,
                evidence_context=_mesh_context(),
            )
        )

        assert "error" not in result, result
        assert result["evidence"]["selection"]["feature_ids"] == []
        assert result["evidence"]["selection"]["row_offsets"] == [0, 1]
        assert result["evidence"]["derived"] is False
        assert result["evidence"]["truncated"] is False
        assert "evidence" not in result["metadata"]

    def test_mesh_probe_reports_truncation_and_exact_limit(self, monkeypatch, tmp_path):
        src = tmp_path / "zones.parquet"
        _write_mesh_parquet(src)
        _allow_local_sources(monkeypatch, tmp_path)

        result = asyncio.run(
            _load_spatial_tools()["spatial_query"](
                source=src.as_posix(),
                geometry={"type": "Point", "coordinates": [0.5, 0.5]},
                select_columns=["geometry", "zone_id"],
                limit=1,
                evidence_context=_mesh_context(),
            )
        )
        assert result["evidence"]["selection"]["row_offsets"] == [0]
        assert result["evidence"]["truncated"] is True
        assert len(result["features"]) == 1

    def test_mesh_uses_manifest_geometry_column(self, monkeypatch, tmp_path):
        src = tmp_path / "zones.parquet"
        _write_mesh_parquet(src, geometry_column="shape")
        _allow_local_sources(monkeypatch, tmp_path)

        result = asyncio.run(
            _load_spatial_tools()["spatial_query"](
                source=src.as_posix(),
                geometry={"type": "Point", "coordinates": [0.5, 0.5]},
                select_columns=["shape", "zone_id"],
                evidence_context=_mesh_context("shape"),
            )
        )
        assert "error" not in result, result
        assert result["features"][0]["geometry"]["type"] == "Polygon"
        assert result["evidence"]["selection"]["geometry_column"] == "shape"

    def test_mesh_reads_formal_geoparquet_logical_geometry(self, monkeypatch):
        _allow_local_sources(monkeypatch, FORMAL_GEOPARQUET.parent)

        result = asyncio.run(
            _load_spatial_tools()["spatial_query"](
                source=FORMAL_GEOPARQUET.as_posix(),
                geometry={"type": "Point", "coordinates": [3.8767, 43.6108]},
                select_columns=["geometry", "id", "label"],
                limit=2,
                evidence_context=_mesh_context(),
            )
        )

        assert "error" not in result, result
        assert result["features"][0]["geometry"]["type"] == "Point"
        assert result["evidence"]["selection"]["row_offsets"] == [0]

        distance_result = asyncio.run(
            _load_spatial_tools()["spatial_query"](
                source=FORMAL_GEOPARQUET.as_posix(),
                geometry={"type": "Point", "coordinates": [3.8767, 43.6108]},
                operation="distance",
                distance_m=1,
                select_columns=["geometry", "id"],
                limit=2,
                evidence_context=_mesh_context(),
            )
        )
        assert "error" not in distance_result, distance_result
        assert distance_result["evidence"]["operation"] == "spatial.distance"

    def test_mesh_normalizes_common_duckdb_property_types(self, monkeypatch, tmp_path):
        src = tmp_path / "typed.parquet"
        _write_typed_mesh_parquet(src)
        _allow_local_sources(monkeypatch, tmp_path)

        result = asyncio.run(
            _load_spatial_tools()["spatial_query"](
                source=src.as_posix(),
                geometry={"type": "Point", "coordinates": [0.5, 0.5]},
                select_columns=["geometry", "observed_at", "score", "record_id", "payload"],
                evidence_context=_mesh_context(),
            )
        )

        assert "error" not in result, result
        assert result["features"][0]["properties"] == {
            "observed_at": "2026-07-12T09:10:11.123456",
            "score": "12.340",
            "record_id": "12345678-1234-5678-1234-567812345678",
            "payload": "base64:QUI=",
        }

    def test_mesh_omitted_limit_uses_lowered_configured_default(self, monkeypatch, tmp_path):
        src = tmp_path / "zones.parquet"
        _write_mesh_parquet(src)
        _allow_local_sources(monkeypatch, tmp_path)
        monkeypatch.setattr(settings, "SPATIAL_MESH_MAX_LIMIT", 1)

        direct = SpatialQueryInput(
            source=src.as_posix(),
            geometry={"type": "Point", "coordinates": [0.5, 0.5]},
            select_columns=["geometry", "zone_id"],
            evidence_context=_mesh_context(),
        )
        assert direct.limit == 1
        legacy = SpatialQueryInput(
            source=src.as_posix(),
            geometry={"type": "Point", "coordinates": [0.5, 0.5]},
        )
        assert legacy.limit == 100

        result = asyncio.run(
            _load_spatial_tools()["spatial_query"](
                source=src.as_posix(),
                geometry={"type": "Point", "coordinates": [0.5, 0.5]},
                select_columns=["geometry", "zone_id"],
                evidence_context=_mesh_context(),
            )
        )
        assert "error" not in result, result
        assert len(result["features"]) == 1
        assert result["evidence"]["truncated"] is True

    def test_mesh_metadata_and_errors_do_not_expose_source(self, monkeypatch, tmp_path):
        src = tmp_path / "private-path-name.parquet"
        _write_mesh_parquet(src)
        _allow_local_sources(monkeypatch, tmp_path)
        result = asyncio.run(
            _load_spatial_tools()["spatial_query"](
                source=src.as_posix(),
                geometry={"type": "Point", "coordinates": [0.5, 0.5]},
                select_columns=["geometry", "zone_id"],
                evidence_context=_mesh_context(),
            )
        )
        assert src.as_posix() not in json.dumps(result["metadata"])

        error = asyncio.run(
            _load_spatial_tools()["spatial_query"](
                source="https://private.example/secret.parquet",
                geometry={"type": "Point", "coordinates": [0, 0]},
                select_columns=["geometry"],
                evidence_context=_mesh_context(),
            )
        )
        assert "error_code" in error
        assert "private.example" not in json.dumps(error)

    def test_mesh_timeout_hard_kills_uncooperative_process(self, monkeypatch, tmp_path):
        src = tmp_path / "zones.parquet"
        _allow_local_sources(monkeypatch, tmp_path)
        monkeypatch.setattr(settings, "SPATIAL_MESH_QUERY_TIMEOUT_S", 0.02)
        monkeypatch.setattr(
            "src.tools.spatial._MESH_WORKER_COMMAND",
            (sys.executable, "-c", "import time; time.sleep(60)"),
        )
        real_spawn = subprocess.Popen
        worker_pids: list[int] = []

        def recording_spawn(*args, **kwargs):
            process = real_spawn(*args, **kwargs)
            worker_pids.append(process.pid)
            return process

        monkeypatch.setattr("src.tools.spatial.subprocess.Popen", recording_spawn)
        started = time.monotonic()
        result = asyncio.run(
            _load_spatial_tools()["spatial_query"](
                source=src.as_posix(),
                geometry={"type": "Point", "coordinates": [0, 0]},
                select_columns=["geometry"],
                evidence_context=_mesh_context(),
            )
        )
        assert time.monotonic() - started < 1.0
        assert result["error_code"] == "SPATIAL_QUERY_TIMEOUT"
        assert worker_pids
        for pid in worker_pids:
            with pytest.raises(ProcessLookupError):
                os.kill(pid, 0)

    def test_mesh_response_byte_cap(self, monkeypatch, tmp_path):
        src = tmp_path / "zones.parquet"
        _write_mesh_parquet(src)
        _allow_local_sources(monkeypatch, tmp_path)
        monkeypatch.setattr(settings, "SPATIAL_MESH_MAX_RESPONSE_BYTES", 200)
        result = asyncio.run(
            _load_spatial_tools()["spatial_query"](
                source=src.as_posix(),
                geometry={"type": "Point", "coordinates": [0.5, 0.5]},
                select_columns=["geometry", "zone_id", "name"],
                evidence_context=_mesh_context(),
            )
        )
        assert result["error_code"] == "SPATIAL_QUERY_RESPONSE_TOO_LARGE"

    def test_mesh_input_rejects_invalid_identity_columns_and_geometry_budgets(self, monkeypatch):
        context = _mesh_context()
        with pytest.raises(ValidationError):
            SpatialQueryInput(
                source="data.parquet",
                geometry={"type": "Point", "coordinates": [0, 0]},
                select_columns=["file_row_number"],
                evidence_context={**context, "geometry_column": "file_row_number"},
            )
        with pytest.raises(ValidationError):
            SpatialQueryInput(
                source="data.parquet",
                geometry={"type": "Point", "coordinates": [181, 0]},
                select_columns=["geometry"],
                evidence_context=context,
            )
        for allowed_column in ("source_id", "source_code", "key_indicator", "uri_count"):
            result = SpatialQueryInput(
                source="data.parquet",
                geometry={"type": "Point", "coordinates": [0, 0]},
                select_columns=["geometry", allowed_column],
                evidence_context=context,
            )
            assert allowed_column in result.select_columns
        for unsafe_column in (
            "source",
            "path",
            "uri",
            "url",
            "bucket",
            "key",
            "etag",
            "file_path",
            "source_uri",
            "object_key",
            "artifact_etag",
            "filename",
            "internal_value",
            "__private",
            "file_row_number",
        ):
            with pytest.raises(ValidationError):
                SpatialQueryInput(
                    source="data.parquet",
                    geometry={"type": "Point", "coordinates": [0, 0]},
                    select_columns=["geometry", unsafe_column],
                    evidence_context=context,
                )

        with pytest.raises(ValidationError):
            SpatialQueryInput(
                source="data.parquet",
                geometry={"type": "Point", "coordinates": [0, 0]},
                operation="distance",
                select_columns=["geometry"],
                evidence_context=context,
            )
        with pytest.raises(ValidationError):
            SpatialQueryInput(
                source="data.parquet",
                geometry={"type": "Point", "coordinates": [0, 0]},
                operation="intersects",
                distance_m=1,
                select_columns=["geometry"],
                evidence_context=context,
            )
        with pytest.raises(ValidationError):
            SpatialQueryInput(
                source="data.parquet",
                geometry={"type": "Point", "coordinates": [0, 0]},
                operation="distance",
                distance_m=1_000_001,
                select_columns=["geometry"],
                evidence_context=context,
            )
        with pytest.raises(ValidationError):
            SpatialQueryInput(
                source="data.parquet",
                geometry={"type": "Point", "coordinates": [0, 0, 0, 0, 0]},
                select_columns=["geometry"],
                evidence_context=context,
            )
        with pytest.raises(ValidationError):
            SpatialQueryInput(
                source="data.parquet",
                geometry={"type": "Polygon", "coordinates": [[0, 0], [1, 1]]},
                select_columns=["geometry"],
                evidence_context=context,
            )
        monkeypatch.setattr(settings, "SPATIAL_MESH_MAX_GEOJSON_BYTES", 10)
        with pytest.raises(ValidationError):
            SpatialQueryInput(
                source="data.parquet",
                geometry={"type": "Point", "coordinates": [0, 0]},
                select_columns=["geometry"],
                evidence_context=context,
            )

    def test_mesh_input_refuses_a_coordinate_past_the_float_range(self):
        # #221: float() raised OverflowError on this int out of the mesh model validator, and
        # Pydantic does not report that as a validation error, where a longitude of 181 is one.
        with pytest.raises(ValidationError):
            SpatialQueryInput(
                source="data.parquet",
                geometry={"type": "Point", "coordinates": [10**400, 0]},
                select_columns=["geometry"],
                evidence_context=_mesh_context(),
            )

    def test_legacy_cancellation_does_not_wait_for_blocked_worker(self, monkeypatch):
        started = Event()
        release = Event()
        finished = Event()
        cancelled = Event()

        def blocked_query(*args, **kwargs):
            started.set()
            release.wait()
            finished.set()
            return [], []

        original_cancel = spatial._MeshQueryControl.cancel

        def recording_cancel(control):
            cancelled.set()
            original_cancel(control)

        monkeypatch.setattr(spatial, "_execute_spatial_query", blocked_query)
        monkeypatch.setattr(spatial._MeshQueryControl, "cancel", recording_cancel)

        async def exercise() -> None:
            task = asyncio.create_task(
                _run_legacy_spatial_query("SELECT 1", source="data.parquet", geom_json="{}")
            )
            deadline = asyncio.get_running_loop().time() + 1
            while not started.is_set() and asyncio.get_running_loop().time() < deadline:
                await asyncio.sleep(0.01)
            assert started.is_set()
            started_cancelling = time.monotonic()
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert time.monotonic() - started_cancelling < 0.5
            assert cancelled.is_set()
            release.set()
            deadline = asyncio.get_running_loop().time() + 1
            while not finished.is_set() and asyncio.get_running_loop().time() < deadline:
                await asyncio.sleep(0.01)
            assert finished.is_set()

        asyncio.run(exercise())

    def test_wkb_geoparquet_round_trip_default_columns(self, monkeypatch, tmp_path):
        src = tmp_path / "zones.parquet"
        _write_wkb_parquet(src)
        _allow_local_sources(monkeypatch, tmp_path)

        result = asyncio.run(
            _load_spatial_tools()["spatial_query"](
                source=src.as_posix(),
                geometry={"type": "Point", "coordinates": [0.5, 0.5]},
                operation="intersects",
            )
        )

        assert "error" not in result, result
        assert result["metadata"]["total_results"] == 1
        feature = result["features"][0]
        assert feature["geometry"]["type"] == "Polygon"
        assert feature["properties"]["zone_id"] == "z1"
        assert feature["properties"]["name"] == "Zone 1"

    def test_wkb_distance_branch_and_selected_geometry(self, monkeypatch, tmp_path):
        src = tmp_path / "zones.parquet"
        _write_wkb_parquet(src)
        _allow_local_sources(monkeypatch, tmp_path)

        result = asyncio.run(
            _load_spatial_tools()["spatial_query"](
                source=src.as_posix(),
                geometry={"type": "Point", "coordinates": [0.5, 0.5]},
                operation="distance",
                distance_m=10,
                select_columns=["zone_id", "geometry"],
            )
        )

        assert "error" not in result, result
        assert result["metadata"]["total_results"] == 1
        assert result["features"][0]["properties"] == {"zone_id": "z1"}
        assert result["features"][0]["geometry"]["type"] == "Polygon"

    @pytest.mark.parametrize(
        ("described_type", "expected_expression"),
        [
            ("BLOB", 'ST_GeomFromWKB("geometry")'),
            ("GEOMETRY('OGC:CRS84')", '"geometry"'),
            ("GEOMETRY('EPSG:4326')", '"geometry"'),
            ("GEOMETRY('EPSG:2154')", '"geometry"'),
        ],
    )
    def test_legacy_geometry_expression_follows_the_column_type(
        self, described_type: str, expected_expression: str
    ) -> None:
        assert _legacy_geometry_expression(described_type, '"geometry"') == expected_expression

    @pytest.mark.parametrize(
        ("described_type", "message"),
        [
            ("VARCHAR", UNSUPPORTED_TYPE_MESSAGE),
            ("GEOMETRYX", UNSUPPORTED_TYPE_MESSAGE),
            ("BLOB[]", UNSUPPORTED_TYPE_MESSAGE),
            ("", UNSUPPORTED_TYPE_MESSAGE),
            ("GEOMETRY('EPSG:4326'", UNSUPPORTED_TYPE_MESSAGE),
            ("GEOMETRY('EPSG:4326') ", UNSUPPORTED_TYPE_MESSAGE),
            ("GEOMETRY", MISSING_CRS_MESSAGE),
            ("GEOMETRY('EPSG:3857')", UNSUPPORTED_CRS_MESSAGE),
            ("GEOMETRY('EPSG:4978')", UNSUPPORTED_CRS_MESSAGE),
            ("GEOMETRY('EPSG:5703')", UNSUPPORTED_CRS_MESSAGE),
            ("GEOMETRY('EPSG:21781')", UNSUPPORTED_CRS_MESSAGE),
            ("GEOMETRY('epsg:2154')", UNSUPPORTED_CRS_MESSAGE),
            ("GEOMETRY('EPSG:2154h')", UNSUPPORTED_CRS_MESSAGE),
            ("GEOMETRY('')", UNSUPPORTED_CRS_MESSAGE),
            # Lon/lat WGS84 by another name, refused on purpose: only the two exact strings pass.
            ("GEOMETRY('OGC:CRS84h')", UNSUPPORTED_CRS_MESSAGE),
            ("GEOMETRY('EPSG:4979')", UNSUPPORTED_CRS_MESSAGE),
            ("GEOMETRY('epsg:4326')", UNSUPPORTED_CRS_MESSAGE),
        ],
    )
    def test_legacy_geometry_expression_rejects_other_types(
        self, described_type: str, message: str
    ) -> None:
        with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
            _legacy_geometry_expression(described_type, '"geometry"')

    def test_varchar_geometry_column_is_refused_as_an_unsupported_type(self, monkeypatch, tmp_path):
        src = tmp_path / "text.parquet"
        con = get_duckdb_connection(":memory:")
        try:
            con.execute(
                "COPY (SELECT 'z1' AS zone_id, 'POINT (0 0)' AS geometry) "
                f"TO '{src.as_posix()}' (FORMAT PARQUET)"
            )
        finally:
            con.close()
        assert _read_geometry_type(src) == "VARCHAR"
        _allow_local_sources(monkeypatch, tmp_path)

        result = asyncio.run(
            _load_spatial_tools()["spatial_query"](
                source=src.as_posix(), geometry={"type": "Point", "coordinates": [0, 0]}
            )
        )

        assert result == {"error": UNSUPPORTED_TYPE_MESSAGE}

    def test_native_geometry_without_a_crs_is_refused_for_its_missing_crs(
        self, monkeypatch, tmp_path
    ):
        src = tmp_path / "no_crs.parquet"
        con = get_duckdb_connection(":memory:")
        try:
            con.execute(
                "COPY (SELECT 'z1' AS zone_id, "
                "ST_GeomFromText('POLYGON((2 48, 3 48, 3 49, 2 49, 2 48))') AS geometry) "
                f"TO '{src.as_posix()}' (FORMAT PARQUET, GEOPARQUET_VERSION 'V2')"
            )
        finally:
            con.close()
        assert _read_geometry_type(src) == "GEOMETRY"
        _allow_local_sources(monkeypatch, tmp_path)

        result = asyncio.run(
            _load_spatial_tools()["spatial_query"](
                source=src.as_posix(), geometry={"type": "Point", "coordinates": [2.5, 48.5]}
            )
        )

        assert result == {"error": MISSING_CRS_MESSAGE}
        assert "CRS" in result["error"]

    @pytest.mark.parametrize(
        ("crs", "accepted"),
        [
            ("OGC:CRS84", True),
            ("EPSG:4326", True),
            ("EPSG:3857", False),
        ],
    )
    def test_native_geometry_crs_gate(self, monkeypatch, tmp_path, crs, accepted):
        src = tmp_path / "zones.parquet"
        polygon = "ST_GeomFromText('POLYGON((2 48, 3 48, 3 49, 2 49, 2 48))')"
        tagged = {
            "OGC:CRS84": polygon,
            "EPSG:4326": f"ST_SetCRS({polygon}, 'EPSG:4326')",
        }.get(crs, f"ST_Transform(ST_SetCRS({polygon}, 'EPSG:4326'), '{crs}')")
        con = get_duckdb_connection(":memory:")
        try:
            con.execute(
                f"COPY (SELECT 'z1' AS zone_id, {tagged} AS geometry) "
                f"TO '{src.as_posix()}' (FORMAT PARQUET)"
            )
        finally:
            con.close()
        assert _read_geometry_type(src) == f"GEOMETRY('{crs}')"
        _allow_local_sources(monkeypatch, tmp_path)
        tool = _load_spatial_tools()["spatial_query"]
        # The point sits inside the polygon in lon/lat. A projected column used as-is would put
        # the polygon in metres, far from that point in degrees (the polygon is anchored away
        # from 0,0 so that this holds for EPSG:3857 too); the distance branch would also
        # reinterpret metres as degrees.
        point = {"type": "Point", "coordinates": [2.5, 48.5]}

        for operation, distance_m in [
            ("intersects", None),
            ("contains", None),
            ("within", None),
            ("distance", 10),
        ]:
            geometry = (
                {"type": "Polygon", "coordinates": [[[1, 47], [4, 47], [4, 50], [1, 50], [1, 47]]]}
                if operation == "within"
                else point
            )
            result = asyncio.run(
                tool(
                    source=src.as_posix(),
                    geometry=geometry,
                    operation=operation,
                    distance_m=distance_m,
                )
            )
            if accepted:
                assert "error" not in result, (operation, result)
                assert [f["properties"]["zone_id"] for f in result["features"]] == ["z1"]
            else:
                assert "error" in result, (operation, result)
                assert result == {"error": UNSUPPORTED_CRS_MESSAGE}

    def test_native_geometry_fixture_is_read_back_as_geometry(self, tmp_path):
        """Guard: the fixture must stay native GEOMETRY, or the tests below stop covering #241."""
        native = tmp_path / "test_cities.parquet"
        _write_live_acceptance_geoparquet(native)
        wkb = tmp_path / "zones.parquet"
        _write_wkb_parquet(wkb)

        assert _read_geometry_type(native).startswith("GEOMETRY")
        assert _read_geometry_type(wkb) == "BLOB"

    @pytest.mark.parametrize(
        ("operation", "geometry", "distance_m"),
        [
            ("intersects", {"type": "Point", "coordinates": [2.3, 48.85]}, None),
            ("contains", {"type": "Point", "coordinates": [2.3, 48.85]}, None),
            ("distance", {"type": "Point", "coordinates": [2.3, 48.85]}, 100),
            (
                "within",
                {
                    "type": "Polygon",
                    "coordinates": [
                        [[2.0, 48.5], [2.7, 48.5], [2.7, 49.2], [2.0, 49.2], [2.0, 48.5]]
                    ],
                },
                None,
            ),
        ],
    )
    def test_native_geometry_geoparquet_all_operations(
        self, monkeypatch, tmp_path, operation, geometry, distance_m
    ):
        src = tmp_path / "test_cities.parquet"
        _write_live_acceptance_geoparquet(src)
        _allow_local_sources(monkeypatch, tmp_path)

        result = asyncio.run(
            _load_spatial_tools()["spatial_query"](
                source=src.as_posix(),
                geometry=geometry,
                operation=operation,
                distance_m=distance_m,
            )
        )

        assert "error" not in result, result
        assert [f["properties"]["name"] for f in result["features"]] == ["Paris"]
        assert result["features"][0]["geometry"]["type"] == "Polygon"

    @pytest.mark.parametrize(
        ("distance_m", "expected"),
        # 0.01 degree of longitude is about 1113 m in EPSG:3857; 0.01 would pass a raw-degree
        # comparison, so a miss at 1100 shows the column is transformed.
        [(1100, []), (1200, ["Paris"])],
    )
    def test_native_geometry_distance_outside_the_polygon(
        self, monkeypatch, tmp_path, distance_m, expected
    ):
        src = tmp_path / "test_cities.parquet"
        _write_live_acceptance_geoparquet(src)
        _allow_local_sources(monkeypatch, tmp_path)

        result = asyncio.run(
            _load_spatial_tools()["spatial_query"](
                source=src.as_posix(),
                geometry={"type": "Point", "coordinates": [2.24, 48.85]},
                operation="distance",
                distance_m=distance_m,
            )
        )

        assert "error" not in result, result
        assert [f["properties"]["name"] for f in result["features"]] == expected

    @pytest.mark.parametrize(("distance_m", "expected"), [(1100, []), (1200, ["z1"])])
    def test_wkb_distance_outside_the_polygon(self, monkeypatch, tmp_path, distance_m, expected):
        src = tmp_path / "zones.parquet"
        _write_wkb_parquet(src)
        _allow_local_sources(monkeypatch, tmp_path)

        result = asyncio.run(
            _load_spatial_tools()["spatial_query"](
                source=src.as_posix(),
                geometry={"type": "Point", "coordinates": [-0.01, 0.5]},
                operation="distance",
                distance_m=distance_m,
            )
        )

        assert "error" not in result, result
        assert [f["properties"]["zone_id"] for f in result["features"]] == expected

    def test_native_geometry_geoparquet_geojson_and_select_columns(self, monkeypatch, tmp_path):
        src = tmp_path / "test_cities.parquet"
        _write_live_acceptance_geoparquet(src)
        _allow_local_sources(monkeypatch, tmp_path)
        point = {"type": "Point", "coordinates": [2.3, 48.85]}
        tool = _load_spatial_tools()["spatial_query"]

        default = asyncio.run(tool(source=src.as_posix(), geometry=point))
        assert "error" not in default, default
        assert default["type"] == "FeatureCollection"
        assert default["metadata"]["total_results"] == 1
        feature = default["features"][0]
        assert feature["type"] == "Feature"
        assert feature["geometry"]["coordinates"][0][0] == [2.25, 48.81]
        assert feature["properties"] == {
            "name": "Paris",
            "country": "France",
            "population": 2161000,
        }

        selected = asyncio.run(
            tool(
                source=src.as_posix(),
                geometry=point,
                select_columns=["name", "population", "geometry"],
            )
        )
        assert "error" not in selected, selected
        assert selected["features"][0]["properties"] == {"name": "Paris", "population": 2161000}
        assert selected["features"][0]["geometry"]["type"] == "Polygon"

    def test_wkb_geoparquet_contains_and_within(self, monkeypatch, tmp_path):
        src = tmp_path / "zones.parquet"
        _write_wkb_parquet(src)
        _allow_local_sources(monkeypatch, tmp_path)
        tool = _load_spatial_tools()["spatial_query"]

        contains = asyncio.run(
            tool(
                source=src.as_posix(),
                geometry={"type": "Point", "coordinates": [5.5, 5.5]},
                operation="contains",
            )
        )
        assert "error" not in contains, contains
        assert [f["properties"]["zone_id"] for f in contains["features"]] == ["z2"]

        within = asyncio.run(
            tool(
                source=src.as_posix(),
                geometry={
                    "type": "Polygon",
                    "coordinates": [[[4, 4], [7, 4], [7, 7], [4, 7], [4, 4]]],
                },
                operation="within",
            )
        )
        assert "error" not in within, within
        assert [f["properties"]["zone_id"] for f in within["features"]] == ["z2"]

    def test_rejects_unsafe_or_cross_bucket_sources(self, monkeypatch, tmp_path):
        tool = _load_spatial_tools()["spatial_query"]
        geometry = {"type": "Point", "coordinates": [0, 0]}

        unsafe = asyncio.run(tool(source="https://example.com/x.parquet", geometry=geometry))
        assert "error" in unsafe and "unsupported source scheme" in unsafe["error"]

        traversal = asyncio.run(
            tool(source=(tmp_path / "../x.parquet").as_posix(), geometry=geometry)
        )
        assert "error" in traversal and "path traversal" in traversal["error"]

        monkeypatch.setattr(settings, "S3_BUCKET", "allowed")
        cross_bucket = asyncio.run(
            tool(source="s3://other-bucket/private.parquet", geometry=geometry)
        )
        assert "error" in cross_bucket and "configured bucket" in cross_bucket["error"]

    def test_local_sources_can_be_disabled(self, monkeypatch, tmp_path):
        tool = _load_spatial_tools()["spatial_query"]
        geometry = {"type": "Point", "coordinates": [0, 0]}

        monkeypatch.setattr(settings, "ALLOW_LOCAL_PARQUET_SOURCE", False)
        result = asyncio.run(
            tool(source=(tmp_path / "zones.parquet").as_posix(), geometry=geometry)
        )

        assert "error" in result
        assert "local parquet sources are disabled" in result["error"]

    def test_local_sources_can_be_confined_to_root(self, monkeypatch, tmp_path):
        tool = _load_spatial_tools()["spatial_query"]
        geometry = {"type": "Point", "coordinates": [0.5, 0.5]}
        allowed_dir = tmp_path / "allowed"
        allowed_dir.mkdir()
        allowed = allowed_dir / "zones.parquet"
        _write_wkb_parquet(allowed)

        _allow_local_sources(monkeypatch, allowed_dir)

        ok = asyncio.run(tool(source=allowed.as_posix(), geometry=geometry))
        assert "error" not in ok, ok
        assert ok["metadata"]["total_results"] == 1

        outside = asyncio.run(
            tool(source=(tmp_path / "outside.parquet").as_posix(), geometry=geometry)
        )
        assert "error" in outside
        assert "LOCAL_PARQUET_SOURCE_ROOT" in outside["error"]

    def test_local_sources_require_root_when_enabled(self, monkeypatch, tmp_path):
        tool = _load_spatial_tools()["spatial_query"]
        monkeypatch.setattr(settings, "ALLOW_LOCAL_PARQUET_SOURCE", True)
        monkeypatch.setattr(settings, "LOCAL_PARQUET_SOURCE_ROOT", "")

        result = asyncio.run(
            tool(
                source=(tmp_path / "zones.parquet").as_posix(),
                geometry={"type": "Point", "coordinates": [0, 0]},
            )
        )

        assert "error" in result
        assert "LOCAL_PARQUET_SOURCE_ROOT is required" in result["error"]


# --------------------------------------------------------------------------- #
# Native GEOMETRY('EPSG:2154') columns (#245)
# --------------------------------------------------------------------------- #

_TRANSFORM_FAILED = "the result is empty or has non-finite coordinates"
QUERY_TO_LAMBERT_MESSAGE = (
    "Invalid Input Error: query geometry cannot be transformed from EPSG:4326 to EPSG:2154: "
    + _TRANSFORM_FAILED
)
QUERY_TO_WEB_MERCATOR_MESSAGE = (
    "Invalid Input Error: query geometry cannot be transformed from EPSG:4326 to EPSG:3857: "
    + _TRANSFORM_FAILED
)
COLUMN_TO_WEB_MERCATOR_MESSAGE = (
    "Invalid Input Error: geometry column cannot be transformed from EPSG:2154 to EPSG:3857: "
    + _TRANSFORM_FAILED
)
COLUMN_TO_LONLAT_MESSAGE = (
    "Invalid Input Error: geometry column cannot be transformed from EPSG:2154 to EPSG:4326: "
    + _TRANSFORM_FAILED
)

# Rows in Lambert-93 metres. The shapes are off-axis (diamond, skewed parallelogram, triangle) and
# x and y differ everywhere, so swapped axes cannot give the same answers by symmetry.
LAMBERT_ROWS = {
    "a_square": [(651000, 6861000), (653000, 6861000), (653000, 6863000), (651000, 6863000)],
    "b_diamond": [(660000, 6870000), (662000, 6868000), (664000, 6870000), (662000, 6872000)],
    "c_skew": [(680000, 6850000), (684000, 6851000), (685000, 6855000), (681000, 6854000)],
    "d_lyon": [(842000, 6518000), (844000, 6518000), (844000, 6520000), (842000, 6520000)],
    # First vertex is the projection origin: Lambert-93 (700000, 6600000) is exactly 3 E, 46.5 N.
    "e_origin": [(700000, 6600000), (702000, 6600000), (701000, 6602000)],
}
# Query shapes in Lambert-93 metres. No row edge is within 3 km of any query edge, so the lon/lat
# edges that become chords once projected do not decide any case.
LAMBERT_INTERSECTS = [(645000, 6861000), (651000, 6855000), (659000, 6863000), (653000, 6869000)]
LAMBERT_WITHIN = [
    (645000, 6855000),
    (668000, 6862000),
    (668000, 6876000),
    (655000, 6878000),
    (646000, 6868000),
]
LAMBERT_EMPTY = [(670000, 6840000), (674000, 6840000), (672000, 6844000)]


def _lambert_to_lonlat(points: list[tuple[float, float]]) -> list[list[float]]:
    """Independent oracle: rasterio's bundled PROJ, not the DuckDB build under test."""
    xs, ys = rasterio_transform(
        "EPSG:2154", "EPSG:4326", [p[0] for p in points], [p[1] for p in points]
    )
    return [[lon, lat] for lon, lat in zip(xs, ys, strict=True)]


def _lonlat_to_lambert(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    xs, ys = rasterio_transform(
        "EPSG:4326", "EPSG:2154", [p[0] for p in points], [p[1] for p in points]
    )
    return list(zip(xs, ys, strict=True))


def _polygon_wkt(points) -> str:
    ring = [*points, points[0]]
    return "POLYGON((" + ", ".join(f"{x!r} {y!r}" for x, y in ring) + "))"


def _geojson_polygon(points) -> dict:
    return {"type": "Polygon", "coordinates": [[list(p) for p in [*points, points[0]]]]}


def _geojson_point(point) -> dict:
    return {"type": "Point", "coordinates": list(point)}


def _lonlat_shape(lambert_shape: dict) -> dict:
    """The same GeoJSON shape with its vertices given as lon/lat instead of Lambert-93 metres."""
    if lambert_shape["type"] == "Point":
        return _geojson_point(_lambert_to_lonlat([tuple(lambert_shape["coordinates"])])[0])
    ring = [tuple(p) for p in lambert_shape["coordinates"][0]]
    return {"type": "Polygon", "coordinates": [_lambert_to_lonlat(ring)]}


def _web_mercator(lon: float, lat: float) -> tuple[float, float]:
    radius = 6_378_137
    return (
        radius * math.radians(lon),
        radius * math.log(math.tan(math.pi / 4 + math.radians(lat) / 2)),
    )


def _write_geometry_parquet(path: Path, rows, geometry_sql: str) -> None:
    """Write ``rows`` (id, name, score, WKT) with ``geometry_sql`` applied to the WKT column."""
    con = get_duckdb_connection(":memory:")
    try:
        con.execute("CREATE TABLE src (zone_id VARCHAR, name VARCHAR, score INTEGER, wkt VARCHAR)")
        con.executemany("INSERT INTO src VALUES (?, ?, ?, ?)", rows)
        con.execute(
            f"COPY (SELECT zone_id, name, score, {geometry_sql} AS geometry FROM src) "
            "TO ? (FORMAT PARQUET)",
            [path.as_posix()],
        )
    finally:
        con.close()


@pytest.fixture
def lambert_sources(monkeypatch, tmp_path) -> dict[str, Path]:
    """The same rows as native Lambert-93, WKB in Lambert-93 and native lon/lat."""
    _allow_local_sources(monkeypatch, tmp_path)
    lambert_rows, lonlat_rows = [], []
    for index, (zone_id, points) in enumerate(LAMBERT_ROWS.items(), start=1):
        name = zone_id.split("_")[1].title()
        lambert_rows.append((zone_id, name, index, _polygon_wkt(points)))
        lonlat_rows.append((zone_id, name, index, _polygon_wkt(_lambert_to_lonlat(points))))
    sources = {
        "native": tmp_path / "native_2154.parquet",
        "wkb": tmp_path / "wkb_2154.parquet",
        "lonlat": tmp_path / "lonlat_4326.parquet",
    }
    _write_geometry_parquet(
        sources["native"], lambert_rows, "ST_SetCRS(ST_GeomFromText(wkt), 'EPSG:2154')"
    )
    _write_geometry_parquet(sources["wkb"], lambert_rows, "ST_AsWKB(ST_GeomFromText(wkt))")
    _write_geometry_parquet(
        sources["lonlat"], lonlat_rows, "ST_SetCRS(ST_GeomFromText(wkt), 'EPSG:4326')"
    )
    assert _read_geometry_type(sources["native"]) == "GEOMETRY('EPSG:2154')"
    assert _read_geometry_type(sources["wkb"]) == "BLOB"
    assert _read_geometry_type(sources["lonlat"]) == "GEOMETRY('EPSG:4326')"
    return sources


def _run(**tool_kwargs) -> dict:
    return asyncio.run(_load_spatial_tools()["spatial_query"](**tool_kwargs))


def _features(result: dict) -> list[tuple[str, dict]]:
    assert "error" not in result, result
    return [(f["properties"]["zone_id"], f["properties"]) for f in result["features"]]


class TestNativeLambert93:
    PARITY_COLUMNS = ["zone_id", "name", "score"]

    @pytest.mark.parametrize(
        ("operation", "lambert_shape", "expected_ids"),
        [
            ("intersects", _geojson_polygon(LAMBERT_INTERSECTS), ["a_square"]),
            ("intersects", _geojson_polygon(LAMBERT_EMPTY), []),
            ("within", _geojson_polygon(LAMBERT_WITHIN), ["a_square", "b_diamond"]),
            ("within", _geojson_polygon(LAMBERT_EMPTY), []),
            ("contains", _geojson_point((662000, 6870000)), ["b_diamond"]),
            ("contains", _geojson_point((662000, 6860000)), []),
        ],
    )
    def test_predicates_match_the_wkb_copy_queried_in_lambert(
        self, lambert_sources, operation, lambert_shape, expected_ids
    ):
        native = _run(
            source=lambert_sources["native"].as_posix(),
            geometry=_lonlat_shape(lambert_shape),
            operation=operation,
            select_columns=self.PARITY_COLUMNS,
        )
        wkb = _run(
            source=lambert_sources["wkb"].as_posix(),
            geometry=lambert_shape,
            operation=operation,
            select_columns=self.PARITY_COLUMNS,
        )

        assert [zone_id for zone_id, _ in _features(native)] == expected_ids
        assert _features(native) == _features(wkb)
        assert [f["properties"] for f in native["features"]] == [
            {"zone_id": zone_id, "name": zone_id.split("_")[1].title(), "score": score}
            for zone_id, score in [(z, list(LAMBERT_ROWS).index(z) + 1) for z in expected_ids]
        ]

    def test_a_query_with_swapped_axes_finds_nothing(self, lambert_sources):
        """Guard for the fixture: lat/lon order must matter, or the parity tests prove nothing."""
        lon_lat = _lambert_to_lonlat(LAMBERT_INTERSECTS)
        swapped = [[lat, lon] for lon, lat in lon_lat]
        result = _run(
            source=lambert_sources["native"].as_posix(),
            geometry={"type": "Polygon", "coordinates": [[*swapped, swapped[0]]]},
            select_columns=self.PARITY_COLUMNS,
        )
        assert _features(result) == []

    @pytest.mark.parametrize(
        ("distance_m", "expected_ids"),
        [(20000, ["a_square", "b_diamond"]), (60000, ["a_square", "b_diamond", "c_skew"])],
    )
    def test_distance_matches_the_lon_lat_copy(self, lambert_sources, distance_m, expected_ids):
        query = _lambert_to_lonlat([(655000, 6865000)])[0]
        results = {
            key: _run(
                source=lambert_sources[key].as_posix(),
                geometry=_geojson_point(query),
                operation="distance",
                distance_m=distance_m,
                select_columns=self.PARITY_COLUMNS,
            )
            for key in ("native", "lonlat")
        }

        assert [zone_id for zone_id, _ in _features(results["native"])] == expected_ids
        assert _features(results["native"]) == _features(results["lonlat"])

    def test_distance_thresholds_bracket_a_known_separation(self, lambert_sources):
        # The query sits diagonally off the corner (653000, 6863000) of a_square, so that corner
        # is the nearest point. The expected value is the EPSG:3857 distance between the two
        # lon/lat points, from the closed-form Mercator formula.
        corner, query = _lambert_to_lonlat([(653000, 6863000), (655000, 6865000)])
        corner_x, corner_y = _web_mercator(*corner)
        query_x, query_y = _web_mercator(*query)
        separation = math.hypot(query_x - corner_x, query_y - corner_y)
        assert 4000 < separation < 4600  # ~2828 m in Lambert-93, stretched by 1/cos(48.9 deg)

        for threshold, expected_ids in [(separation - 1, []), (separation + 1, ["a_square"])]:
            for key in ("native", "lonlat"):
                result = _run(
                    source=lambert_sources[key].as_posix(),
                    geometry=_geojson_point(query),
                    operation="distance",
                    distance_m=threshold,
                    select_columns=["zone_id"],
                )
                assert [zone_id for zone_id, _ in _features(result)] == expected_ids, (
                    key,
                    threshold,
                    separation,
                )

    def test_default_columns_return_lon_lat_geometry(self, lambert_sources):
        result = _run(
            source=lambert_sources["native"].as_posix(),
            geometry=_geojson_point(_lambert_to_lonlat([(662000, 6870000)])[0]),
            operation="contains",
        )

        assert "error" not in result, result
        feature = result["features"][0]
        assert feature["properties"] == {"zone_id": "b_diamond", "name": "Diamond", "score": 2}
        assert feature["geometry"]["type"] == "Polygon"
        ring = feature["geometry"]["coordinates"][0]
        expected = _lambert_to_lonlat([*LAMBERT_ROWS["b_diamond"], LAMBERT_ROWS["b_diamond"][0]])
        assert len(ring) == len(expected)
        for actual, wanted in zip(ring, expected, strict=True):
            assert actual == pytest.approx(wanted, abs=1e-7)
        # Not Lambert-93 metres: every coordinate is a plausible longitude and latitude.
        assert all(-10 < lon < 10 and 40 < lat < 52 for lon, lat in ring)

    def test_explicit_geometry_column_returns_lon_lat_geometry(self, lambert_sources):
        result = _run(
            source=lambert_sources["native"].as_posix(),
            geometry=_lonlat_shape(
                _geojson_polygon([(699000, 6599000), (704000, 6599000), (704000, 6604000)])
            ),
            operation="intersects",
            select_columns=["score", "geometry", "zone_id"],
        )

        assert "error" not in result, result
        assert len(result["features"]) == 1
        feature = result["features"][0]
        assert feature["properties"] == {"score": 5, "zone_id": "e_origin"}
        ring = feature["geometry"]["coordinates"][0]
        # The projection origin is exactly 3 E, 46.5 N: an analytic value, not a library output.
        assert ring[0] == pytest.approx([3.0, 46.5], abs=1e-9)
        expected = _lambert_to_lonlat([*LAMBERT_ROWS["e_origin"], LAMBERT_ROWS["e_origin"][0]])
        for actual, wanted in zip(ring, expected, strict=True):
            assert actual == pytest.approx(wanted, abs=1e-7)

    def test_attributes_only_selection_does_not_return_geometry(self, lambert_sources):
        result = _run(
            source=lambert_sources["native"].as_posix(),
            geometry=_geojson_point(_lambert_to_lonlat([(662000, 6870000)])[0]),
            operation="contains",
            select_columns=["zone_id", "name"],
        )

        assert "error" not in result, result
        assert result["features"] == [
            {
                "type": "Feature",
                "geometry": None,
                "properties": {"zone_id": "b_diamond", "name": "Diamond"},
            }
        ]

    @pytest.mark.parametrize("operation", ["intersects", "within", "contains"])
    @pytest.mark.parametrize("pole", [(0, -90), (500, 95)])
    def test_non_finite_transformed_query_is_rejected(self, lambert_sources, operation, pole):
        # DuckDB 1.5.4 returns POINT (inf inf) for (0, -90) -> EPSG:2154 without raising, and
        # ST_Intersects on it is simply false: an empty success if nothing checks the transform.
        geometry = (
            {"type": "Polygon", "coordinates": [[[*pole], [pole[0] + 1, pole[1]], [*pole]]]}
            if operation == "within"
            else _geojson_point(pole)
        )
        result = _run(
            source=lambert_sources["native"].as_posix(), geometry=geometry, operation=operation
        )

        assert result == {"error": QUERY_TO_LAMBERT_MESSAGE}

    def test_distance_rejects_a_query_with_no_finite_web_mercator_position(self, lambert_sources):
        result = _run(
            source=lambert_sources["native"].as_posix(),
            geometry=_geojson_point((0, 95)),
            operation="distance",
            distance_m=1000,
        )

        assert result == {"error": QUERY_TO_WEB_MERCATOR_MESSAGE}

    def test_the_non_finite_check_is_scoped_to_the_native_projected_path(self, lambert_sources):
        # WKB and lon/lat columns never transform the query for a predicate, so the same query
        # still succeeds there, as before.
        for key in ("wkb", "lonlat"):
            result = _run(
                source=lambert_sources[key].as_posix(),
                geometry=_geojson_point((0, -90)),
                select_columns=["zone_id"],
            )
            assert result["features"] == [], (key, result)

    @pytest.fixture
    def poisoned_source(self, monkeypatch, tmp_path) -> Path:
        """A native Lambert-93 file whose second row has a NaN vertex, next to a valid row."""
        _allow_local_sources(monkeypatch, tmp_path)
        src = tmp_path / "poisoned_2154.parquet"
        con = get_duckdb_connection(":memory:")
        try:
            con.execute(
                f"""
                COPY (
                    SELECT 'ok' AS zone_id, ST_SetCRS(ST_GeomFromText(
                        '{_polygon_wkt(LAMBERT_ROWS["a_square"])}'), 'EPSG:2154') AS geometry
                    UNION ALL
                    SELECT 'poison', ST_SetCRS(ST_MakeLine([
                        ST_Point(652000, 6862000), ST_Point('nan'::DOUBLE, 6862000)
                    ]), 'EPSG:2154')
                ) TO '{src.as_posix()}' (FORMAT PARQUET)
                """
            )
        finally:
            con.close()
        assert _read_geometry_type(src) == "GEOMETRY('EPSG:2154')"
        return src

    @staticmethod
    def _around_the_poisoned_vertex() -> dict:
        return _lonlat_shape(
            _geojson_polygon([(651500, 6861500), (652500, 6861500), (652500, 6862500)])
        )

    def test_a_row_that_cannot_be_transformed_for_output_is_an_error(self, poisoned_source):
        # Without a check ST_AsGeoJSON returns '' for it and the tool reports a malformed feature.
        query = self._around_the_poisoned_vertex()

        default = _run(source=poisoned_source.as_posix(), geometry=query)
        explicit = _run(
            source=poisoned_source.as_posix(),
            geometry=query,
            select_columns=["zone_id", "geometry"],
        )

        assert default == {"error": COLUMN_TO_LONLAT_MESSAGE}
        assert explicit == {"error": COLUMN_TO_LONLAT_MESSAGE}

    def test_a_row_that_cannot_be_transformed_for_distance_is_an_error(self, poisoned_source):
        result = _run(
            source=poisoned_source.as_posix(),
            geometry=_geojson_point(_lambert_to_lonlat([(652000, 6862000)])[0]),
            operation="distance",
            distance_m=1000,
            select_columns=["zone_id"],
        )

        assert result == {"error": COLUMN_TO_WEB_MERCATOR_MESSAGE}

    def test_attributes_only_selection_keeps_its_behavior_on_such_a_row(self, poisoned_source):
        # The output conversion exists only when geometry is returned: nothing is converted here.
        result = _run(
            source=poisoned_source.as_posix(),
            geometry=self._around_the_poisoned_vertex(),
            select_columns=["zone_id"],
        )

        assert [zone_id for zone_id, _ in _features(result)] == ["ok", "poison"]

    def test_a_row_that_transforms_to_an_empty_geometry_is_an_error(self, monkeypatch, tmp_path):
        # DuckDB 1.5.4 transforms POINT (nan y) to EPSG:3857 as POINT EMPTY, with no inf or nan in
        # the text, and ST_Distance on it is not true: a silent miss if only the text is checked.
        _allow_local_sources(monkeypatch, tmp_path)
        src = tmp_path / "nan_point.parquet"
        con = get_duckdb_connection(":memory:")
        try:
            con.execute(
                "COPY (SELECT 'nan' AS zone_id, ST_SetCRS(ST_Point('nan'::DOUBLE, 6862000), "
                f"'EPSG:2154') AS geometry) TO '{src.as_posix()}' (FORMAT PARQUET)"
            )
        finally:
            con.close()

        result = _run(
            source=src.as_posix(),
            geometry=_geojson_point(_lambert_to_lonlat([(652000, 6862000)])[0]),
            operation="distance",
            distance_m=1000,
            select_columns=["zone_id"],
        )

        assert result == {"error": COLUMN_TO_WEB_MERCATOR_MESSAGE}

    def test_a_row_without_a_geometry_is_not_a_transform_failure(self, monkeypatch, tmp_path):
        _allow_local_sources(monkeypatch, tmp_path)
        src = tmp_path / "null_geometry.parquet"
        con = get_duckdb_connection(":memory:")
        try:
            con.execute(
                "COPY (SELECT zone_id, ST_SetCRS(shape, 'EPSG:2154') AS geometry FROM ("
                "SELECT 'null' AS zone_id, NULL::GEOMETRY AS shape UNION ALL "
                "SELECT 'ok', ST_GeomFromText('POINT (652000 6862000)'))) "
                f"TO '{src.as_posix()}' (FORMAT PARQUET)"
            )
        finally:
            con.close()
        assert _read_geometry_type(src) == "GEOMETRY('EPSG:2154')"
        around = _lonlat_shape(
            _geojson_polygon([(650000, 6860000), (654000, 6860000), (652000, 6864000)])
        )

        distance = _run(
            source=src.as_posix(),
            geometry=_geojson_point(_lambert_to_lonlat([(652000, 6862000)])[0]),
            operation="distance",
            distance_m=5,
        )
        default = _run(source=src.as_posix(), geometry=around)

        assert [zone_id for zone_id, _ in _features(distance)] == ["ok"]
        assert [zone_id for zone_id, _ in _features(default)] == ["ok"]
        assert default["features"][0]["geometry"]["type"] == "Point"

    @pytest.mark.parametrize(
        ("crs", "wkt"),
        [
            ("EPSG:3857", "POINT (278000 6190000)"),
            ("EPSG:4978", "POINT (4200000 190000)"),
            ("EPSG:5703", "POINT (2.5 48.5)"),
        ],
    )
    @pytest.mark.parametrize("operation", ["intersects", "within", "contains", "distance"])
    def test_other_native_crs_are_still_refused(self, monkeypatch, tmp_path, crs, wkt, operation):
        # EPSG:4978 (geocentric) and EPSG:5703 (vertical) both survive a probe ST_Transform from
        # EPSG:4326, and both would be read in the wrong coordinates, so no probe is used.
        _allow_local_sources(monkeypatch, tmp_path)
        src = tmp_path / "other.parquet"
        _write_geometry_parquet(
            src, [("z1", "Z", 1, wkt)], f"ST_SetCRS(ST_GeomFromText(wkt), '{crs}')"
        )
        assert _read_geometry_type(src) == f"GEOMETRY('{crs}')"

        result = _run(
            source=src.as_posix(),
            geometry=_geojson_point((2.5, 48.5)),
            operation=operation,
            distance_m=10 if operation == "distance" else None,
        )

        assert result == {"error": UNSUPPORTED_CRS_MESSAGE}

    @pytest.mark.parametrize("operation", ["intersects", "within", "contains", "distance"])
    def test_a_native_column_without_a_crs_is_still_refused(self, monkeypatch, tmp_path, operation):
        _allow_local_sources(monkeypatch, tmp_path)
        src = tmp_path / "no_crs.parquet"
        con = get_duckdb_connection(":memory:")
        try:
            con.execute(
                "COPY (SELECT 'z1' AS zone_id, ST_GeomFromText('POINT (652000 6862000)') "
                f"AS geometry) TO '{src.as_posix()}' (FORMAT PARQUET, GEOPARQUET_VERSION 'V2')"
            )
        finally:
            con.close()
        assert _read_geometry_type(src) == "GEOMETRY"

        result = _run(
            source=src.as_posix(),
            geometry=_geojson_point((2.5, 48.5)),
            operation=operation,
            distance_m=10 if operation == "distance" else None,
        )

        assert result == {"error": MISSING_CRS_MESSAGE}

    def test_polygon_edge_limit_vertices_are_transformed_and_segments_reconnected(
        self, monkeypatch, tmp_path
    ):
        """Characterization, not a guarantee: a lon/lat edge is not densified before projection.

        A parallel (constant latitude) is an arc in Lambert-93, but only the query polygon's
        vertices are transformed, so its long 49 N edge becomes a straight chord about 7 km north
        of that arc at 3 E. A feature north of the lon/lat edge but south of the chord matches.
        """
        _allow_local_sources(monkeypatch, tmp_path)

        def square_at(lon: float, lat: float) -> list[tuple[float, float]]:
            x, y = _lonlat_to_lambert([(lon, lat)])[0]
            return [(x - 100, y - 100), (x + 100, y - 100), (x + 100, y + 100), (x - 100, y + 100)]

        src = tmp_path / "edge.parquet"
        _write_geometry_parquet(
            src,
            [
                ("inside_edge", "In", 1, _polygon_wkt(square_at(3.0, 48.98))),
                ("beyond_edge", "Out", 2, _polygon_wkt(square_at(3.0, 49.02))),
            ],
            "ST_SetCRS(ST_GeomFromText(wkt), 'EPSG:2154')",
        )
        top_lons = [7 - 0.25 * step for step in range(33)]  # 33 vertices from 7 E to -1 E
        plain = [[-1, 46], [7, 46], [7, 49], [-1, 49], [-1, 46]]
        densified = [[-1, 46], [7, 46], *[[lon, 49] for lon in top_lons], [-1, 46]]

        def ids(ring: list[list[float]]) -> list[str]:
            result = _run(
                source=src.as_posix(),
                geometry={"type": "Polygon", "coordinates": [ring]},
                select_columns=["zone_id"],
            )
            return [zone_id for zone_id, _ in _features(result)]

        # Both squares lie about two kilometres either side of the 49 N edge in lon/lat.
        assert ids(densified) == ["inside_edge"]
        assert ids(plain) == ["inside_edge", "beyond_edge"]
