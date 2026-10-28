"""Inspect formal GeoParquet and explicitly declared internal WKB Parquet.

The public function in this module is an internal library boundary, not an MCP
tool. Callers must authorize the source before invocation and apply upload/file
size budgets. Inspection scans the primary geometry column to validate WKB and
derive a trustworthy two-dimensional bbox.
"""

from __future__ import annotations

import json
import math
import os
import re
import stat
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path, PurePosixPath
from typing import Any, Literal
from urllib.parse import urlsplit

import duckdb
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from rasterio.crs import CRS  # type: ignore[import-untyped]
from rasterio.errors import CRSError  # type: ignore[import-untyped]
from shapely import from_wkb, get_coordinates  # type: ignore[import-untyped]
from shapely.errors import GEOSException  # type: ignore[import-untyped]

from src.config import settings
from src.manifests import (
    DatasetKind,
    DatasetManifestV1,
    ManifestArtifact,
    ManifestColumn,
    ManifestProvenance,
    ManifestSpatial,
    ManifestTable,
)
from src.utils.duckdb_conn import duckdb_session
from src.utils.path_policy import resolve_local_parquet_path

_GEOPARQUET_VERSION_RE = re.compile(r"^(?P<major>[0-9]+)\.[0-9]+\.[0-9]+$")
_GEOMETRY_TYPE_RE = re.compile(
    r"^(Point|LineString|Polygon|MultiPoint|MultiLineString|MultiPolygon|"
    r"GeometryCollection)( Z| M| ZM)?$"
)
_SUPPORTED_ENCODINGS = {
    "WKB",
    "point",
    "linestring",
    "polygon",
    "multipoint",
    "multilinestring",
    "multipolygon",
}
_GLOB_CHARS = frozenset("*?[]{}")
_MAX_GEO_METADATA_BYTES = 1_000_000


class GeoParquetInspectionError(ValueError):
    """Raised when a Parquet artifact cannot satisfy the spatial contract."""


def _canonical_crs(value: Any) -> str:
    if value is None:
        raise GeoParquetInspectionError("an explicit, non-null CRS is required")
    if not isinstance(value, (str, dict)):
        raise GeoParquetInspectionError("CRS must be a PROJJSON object or an authority string")
    candidate = json.dumps(value, separators=(",", ":")) if isinstance(value, dict) else value
    try:
        crs = CRS.from_user_input(candidate)
    except (CRSError, TypeError, ValueError) as exc:
        raise GeoParquetInspectionError(f"invalid CRS: {exc}") from exc
    epsg = crs.to_epsg()
    canonical = f"EPSG:{epsg}" if epsg is not None else crs.to_string()
    if not canonical:
        raise GeoParquetInspectionError("CRS cannot be normalized to a stable identifier")
    return canonical


class WkbCompatibility(BaseModel):
    """Explicit semantics for an internal WKB Parquet without ``geo`` metadata."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    geometry_column: str = Field(..., min_length=1)
    crs: str = Field(..., min_length=1)

    @field_validator("crs")
    @classmethod
    def _normalize_crs(cls, value: str) -> str:
        return _canonical_crs(value)


class SpatialParquetInspection(BaseModel):
    """Validated spatial fragment from which a canonical manifest can be built."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    source_kind: Literal["formal_geoparquet", "internal_wkb"]
    artifact_format: Literal["geoparquet", "parquet"]
    geoparquet_version: str | None
    table: ManifestTable
    spatial: ManifestSpatial
    geometry_types: list[str] = Field(..., min_length=1)
    geometry_count: int = Field(..., ge=1)
    null_geometry_count: int = Field(..., ge=0)
    empty_geometry_count: int = Field(..., ge=0)
    metadata_bbox_verified: Literal[True] | None
    caveats: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _counts_fit_table(self) -> SpatialParquetInspection:
        if self.geometry_count + self.null_geometry_count != self.table.row_count:
            raise ValueError("geometry and null counts must equal table row_count")
        if self.empty_geometry_count > self.geometry_count:
            raise ValueError("empty geometry count cannot exceed geometry count")
        if self.source_kind == "formal_geoparquet":
            if self.artifact_format != "geoparquet" or self.geoparquet_version is None:
                raise ValueError(
                    "formal GeoParquet inspection requires format and version metadata"
                )
        elif self.artifact_format != "parquet" or self.geoparquet_version is not None:
            raise ValueError("internal WKB inspection must remain plain parquet without a version")
        return self


def _normalize_source(source: str | Path) -> str:
    raw = source.as_posix() if isinstance(source, Path) else source
    if not raw or any(character in raw for character in _GLOB_CHARS):
        raise GeoParquetInspectionError("source must name one concrete Parquet artifact")
    if raw.startswith("s3://"):
        parsed = urlsplit(raw)
        if parsed.query or parsed.fragment or not parsed.netloc or not parsed.path:
            raise GeoParquetInspectionError("invalid S3 Parquet URI")
        if not settings.S3_BUCKET:
            raise GeoParquetInspectionError("S3_BUCKET is required for S3 inspection")
        if parsed.netloc != settings.S3_BUCKET:
            raise GeoParquetInspectionError(
                f"S3 artifact must be in the configured bucket {settings.S3_BUCKET!r}"
            )
        s3_path = PurePosixPath(parsed.path.lstrip("/"))
        if ".." in s3_path.parts or s3_path.suffix.lower() != ".parquet":
            raise GeoParquetInspectionError("S3 source must be a traversal-free .parquet key")
        return raw
    if "://" in raw:
        raise GeoParquetInspectionError("only local paths and s3:// Parquet URIs are supported")
    try:
        local_path = resolve_local_parquet_path(raw, role="GeoParquet inspection source")
    except ValueError as exc:
        raise GeoParquetInspectionError(str(exc)) from exc
    if local_path.suffix.lower() != ".parquet" or not local_path.is_file():
        raise GeoParquetInspectionError("local source must be an existing .parquet file")
    return local_path.as_posix()


@contextmanager
def _inspection_connection(source: str) -> Iterator[duckdb.DuckDBPyConnection]:
    if source.startswith("s3://"):
        with duckdb_session(":memory:", load_httpfs=True) as conn:
            yield conn
        return
    conn = duckdb.connect(":memory:")
    try:
        yield conn
    finally:
        conn.close()


def _table_metadata(conn: duckdb.DuckDBPyConnection, source: str) -> ManifestTable:
    files = conn.execute(
        "SELECT num_rows FROM parquet_file_metadata(?)",
        [source],
    ).fetchall()
    if len(files) != 1:
        raise GeoParquetInspectionError(
            f"source must resolve to exactly one Parquet file, got {len(files)}"
        )
    described = conn.execute("DESCRIBE SELECT * FROM read_parquet(?)", [source]).fetchall()
    columns = [
        ManifestColumn(
            name=str(row[0]),
            type="BLOB" if str(row[1]).startswith("GEOMETRY") else str(row[1]),
        )
        for row in described
    ]
    return ManifestTable(row_count=int(files[0][0]), columns=columns)


def _decode_geo_metadata(conn: duckdb.DuckDBPyConnection, source: str) -> dict[str, Any] | None:
    matches: list[bytes] = []
    for key, value in conn.execute(
        "SELECT key, value FROM parquet_kv_metadata(?)",
        [source],
    ).fetchall():
        try:
            decoded_key = key if isinstance(key, str) else bytes(key).decode("utf-8")
        except (UnicodeDecodeError, TypeError, ValueError):
            continue
        if decoded_key == "geo":
            if not isinstance(value, (bytes, bytearray, memoryview)):
                raise GeoParquetInspectionError("GeoParquet geo metadata must be binary UTF-8")
            matches.append(bytes(value))
    if not matches:
        return None
    if len(matches) != 1:
        raise GeoParquetInspectionError("Parquet file contains duplicate geo metadata entries")
    raw = matches[0]
    if len(raw) > _MAX_GEO_METADATA_BYTES:
        raise GeoParquetInspectionError("GeoParquet geo metadata exceeds the 1 MB limit")
    try:
        payload = json.loads(raw.decode("utf-8"))
    # ValueError also covers an integer longer than Python's digit limit (#221); the other two
    # are its subclasses, named for the reader.
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise GeoParquetInspectionError("GeoParquet geo metadata is not valid UTF-8 JSON") from exc
    if not isinstance(payload, dict):
        raise GeoParquetInspectionError("GeoParquet geo metadata must be a JSON object")
    return payload


def _metadata_bbox(value: Any, *, label: str, two_dimensional: bool) -> list[float]:
    if not isinstance(value, list) or len(value) not in {4, 6, 8}:
        raise GeoParquetInspectionError(f"{label} bbox must contain 4, 6, or 8 coordinates")
    try:
        bbox = [float(coordinate) for coordinate in value]
    except OverflowError as exc:  # an int past the float range (#221)
        raise GeoParquetInspectionError(f"{label} bbox coordinates must be finite") from exc
    except (TypeError, ValueError) as exc:
        raise GeoParquetInspectionError(f"{label} bbox coordinates must be numeric") from exc
    if not all(math.isfinite(coordinate) for coordinate in bbox):
        raise GeoParquetInspectionError(f"{label} bbox coordinates must be finite")
    dimension = len(bbox) // 2
    if any(bbox[index] > bbox[index + dimension] for index in range(dimension)):
        raise GeoParquetInspectionError(f"{label} bbox minima must not exceed maxima")
    if two_dimensional and len(bbox) != 4:
        raise GeoParquetInspectionError("dataset-manifest/v1 currently supports only 2D bbox")
    return bbox


def _geometry_types(value: Any, *, label: str) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise GeoParquetInspectionError(f"{label} geometry_types must be an array of strings")
    if not value:
        raise GeoParquetInspectionError(f"{label} geometry_types must contain at least one type")
    invalid = [item for item in value if not _GEOMETRY_TYPE_RE.fullmatch(item)]
    if invalid:
        raise GeoParquetInspectionError(f"{label} has unsupported geometry_types: {invalid}")
    if len(value) != len(set(value)):
        raise GeoParquetInspectionError(f"{label} geometry_types must contain unique values")
    return value


def _formal_semantics(
    metadata: dict[str, Any], *, table_columns: set[str] | None
) -> tuple[str, str, str, list[str], list[float] | None]:
    version = metadata.get("version")
    if not isinstance(version, str):
        raise GeoParquetInspectionError(
            f"unsupported GeoParquet version {version!r}; expected a stable 1.x.y version"
        )
    match = _GEOPARQUET_VERSION_RE.fullmatch(version)
    if match is None or match.group("major") != "1":
        raise GeoParquetInspectionError(
            f"unsupported GeoParquet version {version!r}; expected a stable 1.x.y version"
        )
    primary = metadata.get("primary_column")
    columns = metadata.get("columns")
    if not isinstance(primary, str) or not primary:
        raise GeoParquetInspectionError("GeoParquet primary_column is required")
    if not isinstance(columns, dict) or not columns:
        raise GeoParquetInspectionError("GeoParquet columns metadata is required")
    if primary not in columns:
        raise GeoParquetInspectionError("GeoParquet primary_column is absent from columns metadata")

    normalized_crs: dict[str, str] = {}
    for name, column in columns.items():
        label = f"GeoParquet column {name!r}"
        if not isinstance(name, str) or not name or not isinstance(column, dict):
            raise GeoParquetInspectionError("GeoParquet columns must map names to metadata objects")
        if table_columns is not None and name not in table_columns:
            raise GeoParquetInspectionError(f"{label} does not exist in the Parquet schema")
        encoding = column.get("encoding")
        if encoding not in _SUPPORTED_ENCODINGS:
            raise GeoParquetInspectionError(f"{label} has unsupported encoding {encoding!r}")
        _geometry_types(column.get("geometry_types"), label=label)
        if "crs" not in column or column["crs"] is None:
            raise GeoParquetInspectionError(f"{label} must declare an explicit non-null CRS")
        if not isinstance(column["crs"], dict):
            raise GeoParquetInspectionError(f"{label} CRS must be a PROJJSON object")
        normalized_crs[name] = _canonical_crs(column["crs"])
        if "bbox" in column:
            _metadata_bbox(column["bbox"], label=label, two_dimensional=name == primary)

    primary_metadata = columns[primary]
    if primary_metadata["encoding"] != "WKB":
        raise GeoParquetInspectionError(
            "dataset-manifest/v1 inspection currently supports WKB primary geometry only"
        )
    return (
        version,
        primary,
        normalized_crs[primary],
        _geometry_types(primary_metadata["geometry_types"], label=f"GeoParquet column {primary!r}"),
        (
            _metadata_bbox(
                primary_metadata["bbox"],
                label=f"GeoParquet column {primary!r}",
                two_dimensional=True,
            )
            if "bbox" in primary_metadata
            else None
        ),
    )


def _quoted_identifier(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def _scan_wkb(
    conn: duckdb.DuckDBPyConnection,
    source: str,
    geometry_column: str,
    *,
    expected_rows: int,
) -> tuple[list[float], list[str], int, int, int]:
    cursor = conn.execute(
        f"SELECT {_quoted_identifier(geometry_column)} FROM read_parquet(?)",
        [source],
    )
    seen = 0
    geometry_count = 0
    null_count = 0
    empty_count = 0
    geometry_types: set[str] = set()
    bounds: list[float] | None = None

    while rows := cursor.fetchmany(2048):
        for (raw_geometry,) in rows:
            row_index = seen
            seen += 1
            if raw_geometry is None:
                null_count += 1
                continue
            if not isinstance(raw_geometry, (bytes, bytearray, memoryview)):
                raise GeoParquetInspectionError(
                    f"geometry row {row_index} is not WKB bytes ({type(raw_geometry).__name__})"
                )
            try:
                geometry = from_wkb(bytes(raw_geometry), on_invalid="raise")
            except (GEOSException, TypeError, ValueError) as exc:
                raise GeoParquetInspectionError(
                    f"geometry row {row_index} contains malformed WKB"
                ) from exc
            geometry_count += 1
            geometry_types.add(geometry.geom_type)
            if geometry.has_z or geometry.has_m:
                raise GeoParquetInspectionError(
                    "dataset-manifest/v1 currently supports only 2D geometries"
                )
            if geometry.is_empty:
                empty_count += 1
                continue
            coordinates = get_coordinates(geometry)
            if not all(math.isfinite(float(value)) for value in coordinates.flat):
                raise GeoParquetInspectionError(
                    f"geometry row {row_index} contains non-finite coordinates"
                )
            current = [float(value) for value in geometry.bounds]
            if not all(math.isfinite(value) for value in current):
                raise GeoParquetInspectionError(f"geometry row {row_index} has non-finite bounds")
            if bounds is None:
                bounds = current
            else:
                bounds = [
                    min(bounds[0], current[0]),
                    min(bounds[1], current[1]),
                    max(bounds[2], current[2]),
                    max(bounds[3], current[3]),
                ]

    if seen != expected_rows:
        raise GeoParquetInspectionError(
            f"geometry scan row count {seen} does not match Parquet metadata {expected_rows}"
        )
    if geometry_count == 0 or bounds is None:
        raise GeoParquetInspectionError("spatial artifact has no non-empty geometry")
    return bounds, sorted(geometry_types), geometry_count, null_count, empty_count


def _bbox_contains(declared: list[float], actual: list[float]) -> bool:
    tolerance = 1e-9
    return (
        declared[0] <= actual[0] + tolerance
        and declared[1] <= actual[1] + tolerance
        and declared[2] >= actual[2] - tolerance
        and declared[3] >= actual[3] - tolerance
    )


def inspect_spatial_parquet(
    source: str | Path,
    *,
    compatibility: WkbCompatibility | None = None,
) -> SpatialParquetInspection:
    """Inspect one authorized Parquet artifact and derive its spatial manifest fragment.

    Formal GeoParquet requires stable 1.x metadata, WKB primary geometry, and an
    explicit PROJJSON CRS. A file without ``geo`` metadata is accepted only when
    ``compatibility`` explicitly supplies the WKB column and CRS. Every local
    source, whether passed as a string or ``Path``, obeys the deployment allowlist.
    """
    return _inspect_spatial_parquet_source(_normalize_source(source), compatibility=compatibility)


def inspect_spatial_parquet_fd(
    descriptor: int,
    *,
    compatibility: WkbCompatibility | None = None,
) -> SpatialParquetInspection:
    """Inspect a caller-authorized, open regular file without reopening its pathname.

    Internal Linux library entry point, not a tool or a caller-controlled locator.
    The caller owns authorization, size limits and immutable snapshot creation.
    Unlike string/Path inspection, a borrowed descriptor already names the file;
    resolving its procfs path would discard that binding and reintroduce a race.
    Duplicate the descriptor so this function never closes the caller's handle.
    """
    owned = os.dup(descriptor)
    try:
        if not stat.S_ISREG(os.fstat(owned).st_mode):
            raise GeoParquetInspectionError("inspection descriptor must name a regular file")
        source = f"/proc/self/fd/{owned}"
        if not Path(source).is_file():
            raise GeoParquetInspectionError("descriptor inspection requires Linux procfs")
        return _inspect_spatial_parquet_source(source, compatibility=compatibility)
    finally:
        os.close(owned)


def _inspect_spatial_parquet_source(
    normalized_source: str,
    *,
    compatibility: WkbCompatibility | None,
) -> SpatialParquetInspection:
    """Common inspection after pathname policy or descriptor authorization."""
    try:
        with _inspection_connection(normalized_source) as conn:
            geo_metadata = _decode_geo_metadata(conn, normalized_source)
            formal_semantics = None
            if geo_metadata is not None:
                if compatibility is not None:
                    raise GeoParquetInspectionError(
                        "WKB compatibility must not override formal GeoParquet metadata"
                    )
                # Validate untrusted geo JSON before DuckDB interprets it while reading the schema.
                formal_semantics = _formal_semantics(geo_metadata, table_columns=None)
            table = _table_metadata(conn, normalized_source)
            column_names = {column.name for column in table.columns}

            if geo_metadata is not None:
                metadata_columns = set(geo_metadata["columns"])
                missing_columns = sorted(metadata_columns - column_names)
                if missing_columns:
                    raise GeoParquetInspectionError(
                        "GeoParquet columns do not exist in the Parquet schema: "
                        + ", ".join(missing_columns)
                    )
                assert formal_semantics is not None
                version, geometry_column, crs, declared_types, declared_bbox = formal_semantics
                source_kind: Literal["formal_geoparquet", "internal_wkb"] = "formal_geoparquet"
                artifact_format: Literal["geoparquet", "parquet"] = "geoparquet"
                caveats: list[str] = []
            else:
                if compatibility is None:
                    raise GeoParquetInspectionError(
                        "Parquet has no geo metadata; explicit WkbCompatibility is required"
                    )
                version = None
                geometry_column = compatibility.geometry_column
                crs = compatibility.crs
                declared_types = []
                declared_bbox = None
                source_kind = "internal_wkb"
                artifact_format = "parquet"
                caveats = [
                    "GeoParquet geo metadata is absent; geometry encoding and CRS come from "
                    "the explicit internal compatibility contract."
                ]

            if geometry_column not in column_names:
                raise GeoParquetInspectionError(
                    f"geometry column {geometry_column!r} is absent from the Parquet schema"
                )
            bbox, actual_types, geometry_count, null_count, empty_count = _scan_wkb(
                conn,
                normalized_source,
                geometry_column,
                expected_rows=table.row_count,
            )
            if source_kind == "formal_geoparquet" and not set(actual_types).issubset(
                declared_types
            ):
                raise GeoParquetInspectionError(
                    "GeoParquet geometry_types do not cover scanned WKB values: "
                    f"declared={sorted(declared_types)}, actual={actual_types}"
                )
            if declared_bbox is not None and not _bbox_contains(declared_bbox, bbox):
                raise GeoParquetInspectionError(
                    "GeoParquet bbox does not contain the scanned geometry extent"
                )

            return SpatialParquetInspection(
                source_kind=source_kind,
                artifact_format=artifact_format,
                geoparquet_version=version,
                table=table,
                spatial=ManifestSpatial(
                    geometry_column=geometry_column,
                    encoding="WKB",
                    crs=crs,
                    bbox=bbox,
                ),
                geometry_types=actual_types,
                geometry_count=geometry_count,
                null_geometry_count=null_count,
                empty_geometry_count=empty_count,
                metadata_bbox_verified=True if declared_bbox is not None else None,
                caveats=caveats,
            )
    except GeoParquetInspectionError:
        raise
    except duckdb.Error as exc:
        raise GeoParquetInspectionError(f"Parquet inspection failed: {exc}") from exc


def build_dataset_manifest(
    inspection: SpatialParquetInspection,
    *,
    dataset_id: str,
    dataset_kind: DatasetKind,
    artifact_key: str,
    artifact_version: str,
    provenance: ManifestProvenance,
) -> DatasetManifestV1:
    """Attach server-owned identity/provenance to a validated spatial inspection."""
    return DatasetManifestV1(
        schema_version="dataset-manifest/v1",
        dataset_id=dataset_id,
        dataset_kind=dataset_kind,
        artifact=ManifestArtifact(
            format=inspection.artifact_format,
            key=artifact_key,
            version=artifact_version,
        ),
        table=inspection.table,
        spatial=inspection.spatial,
        provenance=provenance,
    )
