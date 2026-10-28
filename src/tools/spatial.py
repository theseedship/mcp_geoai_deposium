"""Spatial query tools using DuckDB."""

import asyncio
import base64
import json
import math
import os
import pickle
import re
import subprocess
import sys
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, time, timedelta
from decimal import Decimal
from functools import partial
from typing import Any, Protocol, cast
from uuid import UUID

import structlog

from src.config import settings
from src.contracts import SpatialQueryInput, check_input, output_schema_for
from src.evidence import (
    EvidenceContextV1,
    EvidenceSelection,
    GeoAIEvidenceV1,
    build_evidence_id,
    validate_mesh_projection_column,
)
from src.utils.duckdb_conn import get_duckdb_connection
from src.utils.path_policy import resolve_local_parquet_path

log = structlog.get_logger(__name__)

# Whitelist of valid spatial operations
VALID_OPERATIONS = {"intersects", "within", "contains", "distance"}

# Pattern for valid column names (alphanumeric + underscore only)
VALID_COLUMN_RE = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")
SAFE_PARQUET_SOURCE_RE = re.compile(r"^[A-Za-z0-9._:/-]+\.parquet$")

SpatialRow = tuple[Any, ...]
SpatialQueryResult = tuple[list[SpatialRow], list[str]]
MeshError = dict[str, str]
MESH_MAX_IPC_BYTES = 16 * 1024 * 1024
MESH_MAX_REQUEST_BYTES = 128 * 1024
MESH_GEOMETRY_EXPRESSION_TOKEN = "__GEOAI_GEOMETRY_EXPRESSION__"
LEGACY_GEOMETRY_COLUMN = "geometry"
# Not matchable by VALID_COLUMN_RE, so a caller-supplied column name can never collide with it.
LEGACY_GEOMETRY_EXPRESSION_TOKEN = "@@GEOAI_GEOMETRY@@"
# The GeoJSON query as the predicates compare it, the column as ``ST_Distance`` reads it, and the
# column as ``ST_AsGeoJSON`` returns it. They differ only for a native EPSG:2154 column.
LEGACY_QUERY_EXPRESSION_TOKEN = "@@GEOAI_QUERY@@"
LEGACY_DISTANCE_EXPRESSION_TOKEN = "@@GEOAI_DISTANCE_GEOMETRY@@"
LEGACY_OUTPUT_EXPRESSION_TOKEN = "@@GEOAI_OUTPUT_GEOMETRY@@"
MESH_FORMAL_GEOMETRY_TYPE = "GEOMETRY('EPSG:4326')"
# DuckDB reports OGC:CRS84 (the GeoParquet default, lon/lat WGS84) when the ``geo`` metadata has
# no ``crs`` member, and a plain GEOMETRY when ``crs`` is null or when a Parquet GEOMETRY logical
# type has no CRS. The legacy path accepts OGC:CRS84 next to the mesh path's EPSG:4326, and
# EPSG:2154 (Lambert-93), the only projected CRS it reads.
LEGACY_NATIVE_PROJECTED_CRS = "EPSG:2154"
LEGACY_NATIVE_GEOMETRY_CRS = frozenset({"EPSG:4326", "OGC:CRS84", LEGACY_NATIVE_PROJECTED_CRS})
# DuckDB 1.5.4 does not raise when ST_Transform cannot place a coordinate: (0, -90) to EPSG:2154
# is POINT (inf inf), and a NaN vertex comes back as nan or as an empty geometry.
_NON_FINITE_WKT = r"\b(inf|nan)\b"
_TRANSFORM_FAILED = "the result is empty or has non-finite coordinates"
_NATIVE_GEOMETRY_TYPE_RE = re.compile(r"GEOMETRY\('([^']*)'\)")
_MESH_WORKER_COMMAND = (sys.executable, "-m", "src.spatial_mesh_worker")


class _InterruptibleConnection(Protocol):
    def interrupt(self) -> None: ...


class _MeshQueryControl:
    """Synchronize timeout cancellation with worker-owned DuckDB connection setup."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._connection: _InterruptibleConnection | None = None
        self._cancelled = False

    def attach(self, connection: _InterruptibleConnection) -> bool:
        with self._lock:
            if self._cancelled:
                connection.interrupt()
                return False
            self._connection = connection
            return True

    def cancel(self) -> None:
        with self._lock:
            self._cancelled = True
            if self._connection is not None:
                self._connection.interrupt()

    def detach(self, connection: _InterruptibleConnection) -> None:
        with self._lock:
            if self._connection is connection:
                self._connection = None


def _mesh_geometry_expression(geometry_type: str, quoted_geometry: str) -> str:
    """Select a type-correct expression for the two supported mesh encodings."""
    if geometry_type == "BLOB":
        return f"ST_GeomFromWKB({quoted_geometry})"
    if geometry_type == MESH_FORMAL_GEOMETRY_TYPE:
        return quoted_geometry
    raise ValueError("unsupported mesh geometry column type")


def _legacy_geometry_expression(geometry_type: str, quoted_geometry: str) -> str:
    """Select a type-correct expression for a WKB ``BLOB`` or a native ``GEOMETRY`` column.

    DuckDB reads a GeoParquet geometry column as ``GEOMETRY('<crs>')`` while a plain Parquet WKB
    column stays ``BLOB``; ``ST_GeomFromWKB`` only accepts the latter (no BLOB <-> GEOMETRY cast).
    A native column is used as-is, in its own CRS: lon/lat WGS84, which the GeoJSON input and the
    EPSG:4326 -> 3857 distance branch assume, or EPSG:2154, for which :func:`_legacy_sql_tokens`
    transforms the query, the distance operand and the returned geometry instead. Only the exact
    strings ``GEOMETRY('EPSG:4326')``, ``GEOMETRY('OGC:CRS84')`` and ``GEOMETRY('EPSG:2154')``
    pass; any other CRS, including a geocentric or vertical one that a probe transform would let
    through, and a bare ``GEOMETRY`` that declares none, is refused rather than compared in the
    wrong coordinates.
    Some refusals are lon/lat WGS84 under another spelling and are refused on purpose, with the
    CRS message: ``OGC:CRS84h``, ``EPSG:4979``, a PROJJSON ``crs`` without an ``id`` member, and a
    lower-case authority in metadata written by another tool.

    Limits: the check reads the CRS the file declares. A ``GEOMETRY`` that DuckDB writes without
    a CRS through a plain ``COPY ... (FORMAT PARQUET)`` reads back as ``OGC:CRS84``, so projected
    coordinates in such a file are not detected; and a WKB ``BLOB`` column declares nothing, so it
    is never checked and is compared in its own coordinates.
    """
    if geometry_type == "BLOB":
        return f"ST_GeomFromWKB({quoted_geometry})"
    if geometry_type == "GEOMETRY":
        raise ValueError(
            "unsupported geometry column CRS: a native GEOMETRY column declares no CRS and must "
            "be EPSG:4326, OGC:CRS84 or EPSG:2154"
        )
    native = _NATIVE_GEOMETRY_TYPE_RE.fullmatch(geometry_type)
    if native is not None:
        if native.group(1) in LEGACY_NATIVE_GEOMETRY_CRS:
            return quoted_geometry
        raise ValueError(
            "unsupported geometry column CRS: a native GEOMETRY column must be EPSG:4326, "
            "OGC:CRS84 or EPSG:2154"
        )
    raise ValueError("unsupported geometry column type")


def _transform_failure(subject: str, source_crs: str, target_crs: str) -> str:
    return f"{subject} cannot be transformed from {source_crs} to {target_crs}: {_TRANSFORM_FAILED}"


def _checked_transform(subject: str, geometry: str, source_crs: str, target_crs: str) -> str:
    """SQL for ``geometry`` transformed to ``target_crs``, raising if the result is unusable.

    Only for the native EPSG:2154 path. A NULL geometry passes through unchanged (the ``CASE``
    condition is then NULL), so a row without a geometry never fails the query.
    """
    transformed = f"ST_Transform({geometry}, '{source_crs}', '{target_crs}', always_xy := true)"
    message = _transform_failure(subject, source_crs, target_crs)
    return (
        f"CASE WHEN regexp_matches(ST_AsText({transformed}), '{_NON_FINITE_WKT}', 'i') "
        f"OR (ST_IsEmpty({transformed}) AND NOT ST_IsEmpty({geometry})) "
        f"THEN error('{message}') ELSE {transformed} END"
    )


def _legacy_sql_tokens(
    geometry_type: str, quoted_geometry: str
) -> tuple[dict[str, str], dict[str, str]]:
    """Return the legacy SQL token expressions for the column ``geometry_type`` describes.

    The second mapping holds, per token, a query-only expression that must be evaluated once
    before the main query, so that an unusable query geometry fails even when no row is read.

    A WKB ``BLOB`` and a lon/lat column keep the original expressions. A native EPSG:2154 column
    is compared where it is: the lon/lat query is transformed into Lambert-93 for the predicates
    (vertices only, segments reconnected, no densification), both sides go to EPSG:3857 for
    ``distance``, and a returned geometry goes back to lon/lat. Every one of those transforms is
    checked, because DuckDB does not raise when it cannot place a coordinate.
    """
    column = _legacy_geometry_expression(geometry_type, quoted_geometry)
    query = "ST_GeomFromGeoJSON($geom)"
    tokens = {
        LEGACY_GEOMETRY_EXPRESSION_TOKEN: column,
        LEGACY_QUERY_EXPRESSION_TOKEN: query,
        LEGACY_DISTANCE_EXPRESSION_TOKEN: (
            f"ST_Transform({column}, 'EPSG:4326', 'EPSG:3857', always_xy := true)"
        ),
        LEGACY_OUTPUT_EXPRESSION_TOKEN: column,
    }
    query_checks: dict[str, str] = {}
    if geometry_type == f"GEOMETRY('{LEGACY_NATIVE_PROJECTED_CRS}')":
        crs = LEGACY_NATIVE_PROJECTED_CRS
        tokens[LEGACY_QUERY_EXPRESSION_TOKEN] = (
            f"ST_Transform({query}, 'EPSG:4326', '{crs}', always_xy := true)"
        )
        tokens[LEGACY_DISTANCE_EXPRESSION_TOKEN] = _checked_transform(
            "geometry column", column, crs, "EPSG:3857"
        )
        tokens[LEGACY_OUTPUT_EXPRESSION_TOKEN] = _checked_transform(
            "geometry column", column, crs, "EPSG:4326"
        )
        query_checks[LEGACY_QUERY_EXPRESSION_TOKEN] = _checked_transform(
            "query geometry", query, "EPSG:4326", crs
        )
        query_checks[LEGACY_DISTANCE_EXPRESSION_TOKEN] = _checked_transform(
            "query geometry", query, "EPSG:4326", "EPSG:3857"
        )
    return tokens, query_checks


def _validate_parquet_source(source: str) -> str | None:
    """Return an error message if ``source`` is not a confined parquet source."""
    if not SAFE_PARQUET_SOURCE_RE.fullmatch(source):
        return f"unsafe parquet source: {source!r}"
    if ".." in source:
        return f"path traversal not allowed in source: {source!r}"
    if "://" in source:
        scheme = source.split("://", 1)[0]
        if scheme != "s3":
            return f"unsupported source scheme {scheme!r} — use a local path or s3:// only"
        if not settings.S3_BUCKET:
            return "s3 source requires S3_BUCKET to be configured"
        prefix = f"s3://{settings.S3_BUCKET}/"
        if not source.startswith(prefix):
            return f"s3 source must be within the configured bucket ({prefix}...)"
        return None
    if not settings.ALLOW_LOCAL_PARQUET_SOURCE:
        return "local parquet sources are disabled by ALLOW_LOCAL_PARQUET_SOURCE"
    try:
        resolve_local_parquet_path(source, role="parquet source")
    except ValueError as exc:
        return str(exc)
    return None


def _execute_spatial_query(
    sql: str,
    *,
    source: str,
    geom_json: str,
    query_control: _MeshQueryControl | None = None,
    distance_m: float | None = None,
    install_extensions: bool = True,
    geometry_column: str | None = None,
    legacy_geometry_column: str | None = None,
) -> SpatialQueryResult:
    """Run the DuckDB spatial query off the event loop."""
    conn = get_duckdb_connection(
        ":memory:",
        load_httpfs=source.startswith("s3://"),
        install_extensions=install_extensions,
    )
    try:
        if query_control is not None and not query_control.attach(conn):
            raise TimeoutError("spatial query cancelled before execution")
        if geometry_column is not None:
            validate_mesh_projection_column(geometry_column, field_name="geometry_column")
            quoted_geometry = _quote_identifier(geometry_column)
            geometry_type = _describe_column_type(conn, quoted_geometry, source)
            geometry_expression = _mesh_geometry_expression(geometry_type, quoted_geometry)
            if MESH_GEOMETRY_EXPRESSION_TOKEN not in sql:
                raise ValueError("mesh SQL is missing its geometry expression token")
            sql = sql.replace(MESH_GEOMETRY_EXPRESSION_TOKEN, geometry_expression)
        elif legacy_geometry_column is not None:
            quoted_geometry = _quote_identifier(legacy_geometry_column)
            geometry_type = _describe_column_type(conn, quoted_geometry, source)
            tokens, query_checks = _legacy_sql_tokens(geometry_type, quoted_geometry)
            for token, check in query_checks.items():
                if token in sql:
                    conn.execute(f"SELECT {check}", {"geom": geom_json}).fetchall()
            for token, expression in tokens.items():
                sql = sql.replace(token, expression)
        params: dict[str, object] = {"source": source, "geom": geom_json}
        if distance_m is not None:
            params["distance"] = distance_m
        result = conn.execute(sql, params).fetchall()
        col_names = [desc[0] for desc in conn.description]
        return result, col_names
    finally:
        if query_control is not None:
            query_control.detach(conn)
        conn.close()


def _describe_column_type(conn: Any, quoted_column: str, source: str) -> str:
    """Return the DuckDB type of one already-quoted column of the parquet ``source``."""
    described = conn.execute(
        f"DESCRIBE SELECT {quoted_column} FROM read_parquet($source)",
        {"source": source},
    ).fetchone()
    return str(described[1]) if described and len(described) > 1 else ""


def _quote_identifier(identifier: str) -> str:
    """Quote an identifier after the contract has restricted it to SQL-safe characters."""
    return f'"{identifier}"'


def _mesh_error(error_code: str, message: str) -> MeshError:
    return {"error": message, "error_code": error_code}


def _mesh_sql(
    *,
    geometry_column: str,
    operation: str,
    distance_m: float | None,
    select_columns: list[str],
    limit: int,
) -> str | MeshError:
    try:
        for column in select_columns:
            validate_mesh_projection_column(column, field_name="select_columns")
    except ValueError:
        return _mesh_error("SPATIAL_QUERY_INVALID_COLUMNS", "invalid mesh projection column")
    geometry_expr = MESH_GEOMETRY_EXPRESSION_TOKEN
    projection = [
        (
            f"ST_AsGeoJSON({geometry_expr}) AS __geoai_geometry"
            if column == geometry_column
            else _quote_identifier(column)
        )
        for column in select_columns
    ]
    projection.append("file_row_number AS __geoai_source_row_offset")
    if operation == "intersects":
        where_clause = f"ST_Intersects({geometry_expr}, ST_GeomFromGeoJSON($geom))"
    elif operation == "within":
        where_clause = f"ST_Within({geometry_expr}, ST_GeomFromGeoJSON($geom))"
    elif operation == "contains":
        where_clause = f"ST_Contains({geometry_expr}, ST_GeomFromGeoJSON($geom))"
    elif operation == "distance" and distance_m is not None:
        # GeoParquet and GeoJSON use longitude/latitude, including EPSG:4326 columns.
        where_clause = f"""
            ST_Distance(
                ST_Transform({geometry_expr}, 'EPSG:4326', 'EPSG:3857', always_xy := true),
                ST_Transform(
                    ST_GeomFromGeoJSON($geom), 'EPSG:4326', 'EPSG:3857', always_xy := true
                )
            ) <= $distance
        """
    else:
        return _mesh_error("SPATIAL_QUERY_INVALID_OPERATION", "invalid spatial operation")
    projection_sql = ", ".join(projection)
    return f"""
        SELECT {projection_sql}
        FROM read_parquet($source, file_row_number = true)
        WHERE {where_clause}
        ORDER BY file_row_number
        LIMIT {limit + 1}
        """


async def _run_mesh_query(
    sql: str,
    *,
    source: str,
    geom_json: str,
    distance_m: float | None,
    geometry_column: str,
) -> SpatialQueryResult | MeshError:
    """Run one mesh query in a killable process with a hard wall-clock timeout."""
    request = json.dumps(
        {
            "sql": sql,
            "source": source,
            "geom_json": geom_json,
            "distance_m": distance_m,
            "geometry_column": geometry_column,
        },
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    ).encode("utf-8")
    if len(request) > MESH_MAX_REQUEST_BYTES:
        return _mesh_error("SPATIAL_QUERY_ERROR", "spatial query failed")

    worker_env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith("COV_CORE_") and key != "COVERAGE_PROCESS_START"
    }
    process: subprocess.Popen[bytes] | None = None
    try:
        with (
            tempfile.TemporaryFile(mode="w+b") as request_file,
            tempfile.TemporaryFile(mode="w+b") as response_file,
        ):
            request_file.write(request)
            request_file.seek(0)
            process = subprocess.Popen(
                _MESH_WORKER_COMMAND,
                stdin=request_file,
                stdout=response_file,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
                env=worker_env,
            )
            loop = asyncio.get_running_loop()
            deadline = loop.time() + settings.SPATIAL_MESH_QUERY_TIMEOUT_S
            while process.poll() is None:
                remaining = deadline - loop.time()
                if remaining <= 0:
                    _kill_mesh_process(process)
                    return _mesh_error("SPATIAL_QUERY_TIMEOUT", "spatial query timed out")
                await asyncio.sleep(min(0.01, remaining))
            if process.returncode != 0:
                return _mesh_error("SPATIAL_QUERY_ERROR", "spatial query failed")
            response_file.seek(0, os.SEEK_END)
            if response_file.tell() > MESH_MAX_IPC_BYTES:
                return _mesh_error("SPATIAL_QUERY_ERROR", "spatial query failed")
            response_file.seek(0)
            encoded = response_file.read()
            try:
                status, payload = pickle.loads(encoded)
            except (EOFError, pickle.PickleError, TypeError, ValueError):
                return _mesh_error("SPATIAL_QUERY_ERROR", "spatial query failed")
            if status == "ok" and isinstance(payload, tuple) and len(payload) == 2:
                return cast(SpatialQueryResult, payload)
            if status == "too_large":
                return _mesh_error(
                    "SPATIAL_QUERY_RESPONSE_TOO_LARGE",
                    "spatial query response exceeds byte limit",
                )
            return _mesh_error("SPATIAL_QUERY_ERROR", "spatial query failed")
    except asyncio.CancelledError:
        if process is not None:
            _kill_mesh_process(process)
        raise
    except Exception:
        if process is not None:
            _kill_mesh_process(process)
        return _mesh_error("SPATIAL_QUERY_ERROR", "spatial query failed")


def _kill_mesh_process(process: subprocess.Popen[bytes]) -> None:
    """Kill and reap one isolated worker without exposing its stderr."""
    if process.poll() is None:
        process.kill()
    try:
        process.wait(timeout=0.5)
    except subprocess.TimeoutExpired:
        pass


async def _poll_duckdb_worker(worker: asyncio.Future[SpatialQueryResult]) -> SpatialQueryResult:
    """Await legacy DuckDB work while progressing the Python 3.14 event loop."""
    # DuckDB's Python extension may complete its worker call without promptly waking an
    # asyncio loop that is awaiting the wrapped concurrent Future.
    while not worker.done():
        await asyncio.sleep(0.01)
    return worker.result()


async def _run_legacy_spatial_query(sql: str, *, source: str, geom_json: str) -> SpatialQueryResult:
    """Run the legacy unbounded query without the DuckDB/asyncio completion deadlock."""
    loop = asyncio.get_running_loop()
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="geoai-spatial-legacy")
    query_control = _MeshQueryControl()
    worker = loop.run_in_executor(
        executor,
        partial(
            _execute_spatial_query,
            sql,
            source=source,
            geom_json=geom_json,
            query_control=query_control,
            legacy_geometry_column=LEGACY_GEOMETRY_COLUMN,
        ),
    )
    cancelled = False
    try:
        return await _poll_duckdb_worker(worker)
    except asyncio.CancelledError:
        cancelled = True
        query_control.cancel()
        raise
    finally:
        executor.shutdown(wait=not cancelled, cancel_futures=True)


def _mesh_response_bytes(payload: dict[str, Any]) -> int:
    return len(
        json.dumps(
            payload,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        ).encode("utf-8")
    )


def _json_safe_property(value: Any) -> Any:
    """Normalize common DuckDB values to deterministic JSON-native values."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("non-finite property value")
        return value
    if isinstance(value, Decimal):
        if not value.is_finite():
            raise ValueError("non-finite decimal property value")
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, (date, time)):
        return value.isoformat()
    if isinstance(value, timedelta):
        return str(value)
    if isinstance(value, UUID):
        return str(value)
    if isinstance(value, (bytes, bytearray, memoryview)):
        return "base64:" + base64.b64encode(bytes(value)).decode("ascii")
    if isinstance(value, (list, tuple)):
        return [_json_safe_property(item) for item in value]
    if isinstance(value, dict):
        if any(not isinstance(key, str) for key in value):
            raise ValueError("property object keys must be strings")
        return {key: _json_safe_property(value[key]) for key in sorted(value)}
    raise ValueError("unsupported property value")


async def _spatial_query_mesh(
    *,
    source: str,
    geometry: dict[str, Any],
    operation: str,
    distance_m: float | None,
    select_columns: list[str],
    limit: int,
    context: EvidenceContextV1,
) -> dict[str, Any]:
    if operation not in VALID_OPERATIONS:
        return _mesh_error("SPATIAL_QUERY_INVALID_OPERATION", "invalid spatial operation")
    if context.crs != "EPSG:4326":
        return _mesh_error("SPATIAL_QUERY_INVALID_CRS", "mesh queries require EPSG:4326")
    source_err = _validate_parquet_source(source)
    if source_err:
        return _mesh_error("SPATIAL_QUERY_SOURCE_REJECTED", "spatial source rejected")
    geom_json = json.dumps(geometry, ensure_ascii=False, separators=(",", ":"))
    query = _mesh_sql(
        geometry_column=context.geometry_column,
        operation=operation,
        distance_m=distance_m,
        select_columns=select_columns,
        limit=limit,
    )
    if isinstance(query, dict):
        return query
    result = await _run_mesh_query(
        query,
        source=source,
        geom_json=geom_json,
        distance_m=distance_m,
        geometry_column=context.geometry_column,
    )
    if isinstance(result, dict):
        log.error("spatial_query_mesh_error", error_code=result["error_code"], operation=operation)
        return result
    try:
        rows, col_names = result
        expected_columns = [
            "__geoai_geometry" if column == context.geometry_column else column
            for column in select_columns
        ] + ["__geoai_source_row_offset"]
        if col_names != expected_columns:
            raise ValueError("DuckDB projection does not match the requested mesh columns")
        truncated = len(rows) > limit
        rows = rows[:limit]
        features = []
        row_offsets = []
        for row in rows:
            values = dict(zip(col_names, row, strict=True))
            geom = values.pop("__geoai_geometry")
            row_offsets.append(int(values.pop("__geoai_source_row_offset")))
            if isinstance(geom, str):
                geom = json.loads(geom)
            features.append(
                {
                    "type": "Feature",
                    "geometry": geom,
                    "properties": {
                        key: _json_safe_property(value) for key, value in values.items()
                    },
                }
            )
        if any(left >= right for left, right in zip(row_offsets, row_offsets[1:], strict=False)):
            raise ValueError("mesh row offsets are not strictly increasing")
    except Exception:
        log.error("spatial_query_mesh_error", error_code="SPATIAL_QUERY_ERROR", operation=operation)
        return _mesh_error("SPATIAL_QUERY_ERROR", "spatial query failed")
    selection = EvidenceSelection(
        feature_ids=(),
        row_offsets=tuple(row_offsets),
        columns=tuple(select_columns),
        geometry_column=context.geometry_column,
    )
    evidence_payload = {
        "schema_version": "geoai-evidence/v1",
        "evidence_id": "geoai_ev_" + "0" * 64,
        "scope_fingerprint": context.scope_fingerprint,
        "dataset_id": context.dataset_id,
        "artifact_version": context.artifact_version,
        "operation": f"spatial.{operation}",
        "query_fingerprint": context.query_fingerprint,
        "selection": selection.model_dump(mode="json"),
        "derived": False,
        "truncated": truncated,
        "caveats": ["feature_ids_unavailable", "row_offsets_are_physical_zero_based"],
    }
    evidence_payload["evidence_id"] = build_evidence_id(evidence_payload)
    evidence = GeoAIEvidenceV1.model_validate(evidence_payload)
    response = {
        "type": "FeatureCollection",
        "features": features,
        "evidence": evidence.model_dump(mode="json"),
        "metadata": {
            "dataset_id": context.dataset_id,
            "artifact_version": context.artifact_version,
            "operation": operation,
            "total_results": len(features),
            "truncated": truncated,
        },
    }
    if _mesh_response_bytes(response) > settings.SPATIAL_MESH_MAX_RESPONSE_BYTES:
        log.error("spatial_query_mesh_error", error_code="SPATIAL_QUERY_RESPONSE_TOO_LARGE")
        return _mesh_error(
            "SPATIAL_QUERY_RESPONSE_TOO_LARGE", "spatial query response exceeds byte limit"
        )
    return response


def register_spatial_tools(mcp: Any) -> None:
    """Register spatial tools with the MCP server."""

    @mcp.tool(output_schema=output_schema_for("spatial_query"))  # type: ignore[untyped-decorator]
    async def spatial_query(
        source: str,
        geometry: dict[str, Any],
        operation: str = "intersects",
        distance_m: float | None = None,
        select_columns: list[str] | None = None,
        limit: int | None = None,
        evidence_context: EvidenceContextV1 | None = None,
    ) -> dict[str, Any]:
        """Execute spatial queries on GeoParquet files using DuckDB.

        A native GEOMETRY column must declare EPSG:4326, OGC:CRS84 or EPSG:2154 (Lambert-93);
        every other CRS, and a column that declares none, is refused. For EPSG:2154 the
        longitude/latitude query is transformed into Lambert-93 and compared there, and a returned
        geometry is converted back to longitude/latitude WGS84. Only vertices are transformed and
        the segments between them reconnected, with no densification and no guaranteed tolerance
        along polygon edges, so a feature close to a query edge may differ from the exact
        longitude/latitude result. A transform that yields an empty or non-finite geometry is an
        error, not an empty result. A WKB BLOB column declares no CRS and is not checked:
        intersects, within and contains compare it in its own coordinates. The evidence mode needs
        EPSG:4326.

        Args:
            source: Same-bucket S3 URI, or a trusted local GeoParquet path when
                ALLOW_LOCAL_PARQUET_SOURCE and LOCAL_PARQUET_SOURCE_ROOT permit it.
            geometry: GeoJSON geometry to query against (longitude/latitude for a native column)
            operation: Spatial operation (intersects, within, contains, distance)
            distance_m: Distance in meters (for distance operation), measured in EPSG:3857
            select_columns: Columns to return (default: all)
            limit: Maximum results (default: 100)
            evidence_context: Server-owned context enabling bounded locator-backed evidence mode.

        Returns:
            GeoJSON FeatureCollection with matching features
        """
        resolved_limit = (
            settings.SPATIAL_MESH_MAX_LIMIT
            if evidence_context is not None and limit is None
            else 100
            if limit is None
            else limit
        )
        check_input(
            SpatialQueryInput,
            source=source,
            geometry=geometry,
            operation=operation,
            distance_m=distance_m,
            select_columns=select_columns,
            limit=resolved_limit,
            evidence_context=evidence_context,
        )
        if evidence_context is not None:
            # The validated model is reconstructed here so direct Python calls and FastMCP calls
            # share exactly the same immutable context semantics.
            mesh_context = EvidenceContextV1.model_validate(evidence_context)
            if select_columns is None:
                return _mesh_error(
                    "SPATIAL_QUERY_INVALID_COLUMNS", "mesh select_columns is required"
                )
            return await _spatial_query_mesh(
                source=source,
                geometry=geometry,
                operation=operation,
                distance_m=distance_m,
                select_columns=select_columns,
                limit=resolved_limit,
                context=mesh_context,
            )
        log.info("spatial_query", source=source, operation=operation)

        # Validate operation against whitelist
        if operation not in VALID_OPERATIONS:
            return {
                "error": f"Invalid operation: {operation}. Must be one of: {', '.join(VALID_OPERATIONS)}"
            }
        source_err = _validate_parquet_source(source)
        if source_err:
            return {"error": source_err}

        # Validate and sanitize column names. Geometry is serialized to
        # GeoJSON directly in the SQL so no per-row conversion is needed.
        # The real expression (WKB BLOB or native GEOMETRY) is resolved from the column's type
        # at execution time, so the SQL carries a placeholder until then.
        geom_expr = LEGACY_GEOMETRY_EXPRESSION_TOKEN
        geojson_geom = f"ST_AsGeoJSON({LEGACY_OUTPUT_EXPRESSION_TOKEN}) AS geometry"
        if select_columns:
            safe_columns = [col for col in select_columns if VALID_COLUMN_RE.match(col)]
            if not safe_columns:
                return {"error": "No valid column names provided"}
            columns_sql = ", ".join(
                geojson_geom if col == "geometry" else col for col in safe_columns
            )
        else:
            columns_sql = f"* REPLACE ({geojson_geom})"

        # Clamp limit to safe range
        limit = max(1, min(int(resolved_limit), settings.SPATIAL_QUERY_MAX_LIMIT))

        geom_json = json.dumps(geometry)

        # Build parameterized query — source and geom_json are passed as parameters
        if operation == "intersects":
            where_clause = f"ST_Intersects({geom_expr}, {LEGACY_QUERY_EXPRESSION_TOKEN})"
        elif operation == "within":
            where_clause = f"ST_Within({geom_expr}, {LEGACY_QUERY_EXPRESSION_TOKEN})"
        elif operation == "contains":
            where_clause = f"ST_Contains({geom_expr}, {LEGACY_QUERY_EXPRESSION_TOKEN})"
        elif operation == "distance" and distance_m:
            safe_distance = float(distance_m)
            # GeoParquet and GeoJSON use longitude/latitude, including EPSG:4326 columns.
            where_clause = f"""
                ST_Distance(
                    {LEGACY_DISTANCE_EXPRESSION_TOKEN},
                    ST_Transform(
                        ST_GeomFromGeoJSON($geom), 'EPSG:4326', 'EPSG:3857', always_xy := true
                    )
                ) <= {safe_distance}
            """
        else:
            return {"error": "Distance operation requires distance_m parameter"}

        sql = f"""
            SELECT {columns_sql}
            FROM read_parquet($source)
            WHERE {where_clause}
            LIMIT {limit}
        """

        try:
            result, col_names = await _run_legacy_spatial_query(
                sql, source=source, geom_json=geom_json
            )

            # Geometry already arrives as a GeoJSON string via ST_AsGeoJSON
            features = []
            for row in result:
                props = dict(zip(col_names, row, strict=True))
                geom = props.pop("geometry", None)
                if isinstance(geom, str):
                    try:
                        geom = json.loads(geom)
                    except json.JSONDecodeError:
                        pass
                features.append(
                    {
                        "type": "Feature",
                        "geometry": geom,
                        "properties": props,
                    }
                )

            return {
                "type": "FeatureCollection",
                "features": features,
                "metadata": {
                    "source": source,
                    "operation": operation,
                    "total_results": len(features),
                },
            }

        except Exception as e:
            log.error("spatial_query_error", error=str(e))
            return {"error": str(e)}
