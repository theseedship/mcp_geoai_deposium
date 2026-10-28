"""Pydantic input/output contract schemas for the MCP tools (FIERCE DoD lot B7).

Each of the sixteen MCP tools has a declared **input** model and a declared
**output** model. The input models are *enforced* at the handler boundary via
:func:`check_input`, rejecting malformed agent calls before any STAC / raster /
DuckDB I/O. The output models are the published, consumer-facing contract —
exported as JSON Schema via :func:`contracts_json_schema` and verified against
the live tool shapes in ``tests/test_contracts.py``.

Design notes
------------
- **Inputs** use ``extra="forbid"`` — a typo'd argument is an error. **Outputs**
  use ``extra="allow"`` — additive metadata keys never break a consumer.
- Whitelisted enums the handlers *already* validate gracefully (the spatial
  ``operation`` set, the mining ``algorithm`` set, the column-name regex) are
  intentionally kept as plain ``str`` here, so the handlers keep returning their
  structured ``{"error": ...}`` instead of the contract pre-empting them with a
  raise. The contract enforces only the structural/range constraints that
  nothing else checks (bbox arity, numeric bounds, non-empty identifiers).
- Output models **are wired into the tools** via FastMCP 3.4.2's ``@mcp.tool(output_schema=...)``.
  FastMCP validates the return at runtime and **raises on a missing required field** (and on a
  wrong type) — verified — so the *wired* schema (:func:`output_schema_for`) is a **lenient** copy
  of the success model with the top-level ``required`` stripped: the success shape **and** a
  handler's ``{"error": ...}`` return both pass; only a wrong *type* on a present field raises. The
  wire stays backward-compatible (FastMCP adds ``structuredContent`` *alongside* the JSON text).
- The **published** contract (:func:`contracts_json_schema`) keeps the **strict** success model
  (its real ``required`` fields) as ``anyOf(success, ErrorResponse)`` — meaningful for consumers —
  with nested ``$defs`` hoisted so ``$ref`` s resolve under ``anyOf``. MCP requires the wired schema
  to be a single object (no ``anyOf``), hence the strict-published / lenient-wired split.
"""

from __future__ import annotations

import json
import math
from datetime import date
from typing import Any, Literal, get_args

from pydantic import BaseModel, ConfigDict, Field, NonNegativeInt, field_validator, model_validator

from src.b6_saved_contracts import B6HistoricalInputs, B6SavedInputs, Sha256
from src.config import settings
from src.evidence import (
    EvidenceContextV1,
    EvidenceSelection,  # noqa: F401 - re-exported compatibility surface
    GeoAIEvidenceV1,
    build_evidence_id,  # noqa: F401 - re-exported compatibility surface
    validate_mesh_projection_column,
)
from src.solar_pilot import SolarPilotInput, SolarPilotOutput


class _InputBase(BaseModel):
    """Inputs reject unknown fields — a misspelled argument is a hard error."""

    model_config = ConfigDict(extra="forbid")


class _OutputBase(BaseModel):
    """Outputs tolerate extra keys — additive metadata never breaks a consumer."""

    model_config = ConfigDict(extra="allow")


def _validate_bbox(value: list[float] | None) -> list[float] | None:
    """Validate a WGS84 bbox as [west, south, east, north]."""
    if value is None:
        return None
    west, south, east, north = value
    coords = [float(v) for v in (west, south, east, north)]
    if not all(math.isfinite(v) for v in coords):
        raise ValueError("bbox coordinates must be finite")
    west, south, east, north = coords
    if not (-180 <= west < east <= 180):
        raise ValueError("bbox longitude bounds must satisfy -180 <= west < east <= 180")
    if not (-90 <= south < north <= 90):
        raise ValueError("bbox latitude bounds must satisfy -90 <= south < north <= 90")
    return coords


def _parse_iso_date(value: str, *, field_name: str) -> date:
    try:
        parsed = date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"{field_name} must be an ISO-8601 date YYYY-MM-DD") from exc
    if parsed.isoformat() != value:
        raise ValueError(f"{field_name} must be an ISO-8601 date YYYY-MM-DD")
    return parsed


_GEOJSON_GEOMETRY_TYPES = {
    "Point",
    "MultiPoint",
    "LineString",
    "MultiLineString",
    "Polygon",
    "MultiPolygon",
    "GeometryCollection",
}

_MESH_MAX_GEOMETRY_DEPTH = 12
_MESH_COORDINATE_NESTING = {
    "Point": 0,
    "MultiPoint": 1,
    "LineString": 1,
    "MultiLineString": 2,
    "Polygon": 2,
    "MultiPolygon": 3,
}


def _validate_mesh_geometry(geometry: dict[str, Any]) -> dict[str, Any]:
    """Apply bounded, finite WGS84 checks used only by the mesh evidence mode."""
    try:
        encoded = json.dumps(
            geometry, ensure_ascii=False, allow_nan=False, separators=(",", ":")
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError("geometry must be finite JSON") from exc
    if len(encoded) > settings.SPATIAL_MESH_MAX_GEOJSON_BYTES:
        raise ValueError("geometry exceeds the configured JSON byte limit")

    positions = 0

    def visit_position(value: Any) -> None:
        nonlocal positions
        if (
            not isinstance(value, list)
            or not 2 <= len(value) <= 4
            or any(isinstance(item, bool) or not isinstance(item, (int, float)) for item in value)
        ):
            raise ValueError("coordinate positions must contain two to four finite numbers")
        try:
            coordinates = [float(item) for item in value]
        except OverflowError as exc:  # an int beyond the float range (#221)
            raise ValueError("coordinate positions must contain finite numbers") from exc
        if not all(math.isfinite(item) for item in coordinates):
            raise ValueError("coordinate positions must contain finite numbers")
        lon, lat = coordinates[:2]
        if not (-180 <= lon <= 180) or not (-90 <= lat <= 90):
            raise ValueError("coordinate positions must be finite WGS84 lon/lat")
        positions += 1
        if positions > settings.SPATIAL_MESH_MAX_COORDINATE_POSITIONS:
            raise ValueError("geometry exceeds the configured coordinate-position limit")

    def visit_coordinates(value: Any, expected_nesting: int, depth: int) -> None:
        if depth > _MESH_MAX_GEOMETRY_DEPTH:
            raise ValueError("geometry nesting exceeds the configured depth limit")
        if expected_nesting == 0:
            visit_position(value)
            return
        if not isinstance(value, list) or not value:
            raise ValueError("geometry coordinates must be non-empty arrays")
        for item in value:
            visit_coordinates(item, expected_nesting - 1, depth + 1)

    def visit_geometry(value: Any, depth: int) -> None:
        if depth > _MESH_MAX_GEOMETRY_DEPTH:
            raise ValueError("geometry nesting exceeds the configured depth limit")
        if not isinstance(value, dict) or value.get("type") not in _GEOJSON_GEOMETRY_TYPES:
            raise ValueError("geometry must contain valid GeoJSON geometry members")
        if set(value) - {"type", "coordinates", "geometries", "bbox"}:
            raise ValueError("geometry contains unsupported members")
        if value["type"] == "GeometryCollection":
            children = value.get("geometries")
            if not isinstance(children, list) or not 1 <= len(children) <= 100:
                raise ValueError("GeometryCollection requires 1 to 100 geometries")
            for child in children:
                visit_geometry(child, depth + 1)
        else:
            visit_coordinates(
                value.get("coordinates"), _MESH_COORDINATE_NESTING[value["type"]], depth + 1
            )

    visit_geometry(geometry, 0)
    return geometry


def _check_coordinate_numbers(coordinates: list) -> None:
    """Refuse a coordinate number that is not a finite float (#221).

    A JSON integer beyond the float range reaches the handler as a Python ``int``, and shapely
    raises ``OverflowError`` on it, which Pydantic does not turn into a validation error. No
    lon/lat range is checked here: ``spatial_query`` compares a WKB ``BLOB`` source in its own
    coordinates, reads a native ``GEOMETRY`` column that declares lon/lat WGS84 (``EPSG:4326``,
    ``OGC:CRS84``) or ``EPSG:2154``, and refuses every other declared CRS. A lon/lat query that
    cannot be placed in ``EPSG:2154`` is refused by the handler, not here.
    Non-numeric members are left to shapely, as before.
    """
    stack = [coordinates]
    while stack:
        item = stack.pop()
        if isinstance(item, list):
            stack.extend(item)
        elif isinstance(item, (int, float)) and not isinstance(item, bool):
            try:
                finite = math.isfinite(float(item))
            except OverflowError:
                finite = False
            if not finite:
                raise ValueError("geometry coordinates must be finite numbers")


def _validate_geometry(value: dict[str, Any]) -> dict[str, Any]:
    """Shape-level GeoJSON *geometry* check (a geometry, not a Feature/FeatureCollection).

    Structural (like ``_validate_bbox``), plus one numeric rule: every number in ``coordinates``
    must be a finite float (:func:`_check_coordinate_numbers`). It rejects the malformed shapes
    that would otherwise fail deep inside shapely / ``ST_GeomFromGeoJSON``.
    """
    if not isinstance(value, dict):
        raise ValueError("geometry must be a GeoJSON geometry object")
    gtype = value.get("type")
    if gtype not in _GEOJSON_GEOMETRY_TYPES:
        raise ValueError(
            f"geometry.type must be one of {sorted(_GEOJSON_GEOMETRY_TYPES)} (got {gtype!r}); "
            "pass a geometry, not a Feature/FeatureCollection"
        )
    if gtype == "GeometryCollection":
        children = value.get("geometries")
        if not isinstance(children, list) or not children:
            raise ValueError("GeometryCollection requires a non-empty 'geometries' list")
        for child in children:
            # Each member must itself be a fully valid geometry (type AND coordinates),
            # else a coordinate-less child would still fail later in shapely / ST_GeomFromGeoJSON.
            _validate_geometry(child)
    elif not isinstance(value.get("coordinates"), list) or not value["coordinates"]:
        raise ValueError(f"{gtype} requires a non-empty 'coordinates' list")
    else:
        _check_coordinate_numbers(value["coordinates"])
    return value


# --------------------------------------------------------------------------- #
# Input contracts (enforced at the handler boundary)
# --------------------------------------------------------------------------- #


class StacSearchInput(_InputBase):
    bbox: list[float] = Field(
        ..., min_length=4, max_length=4, description="[min_lon, min_lat, max_lon, max_lat]"
    )
    datetime: str = Field(..., min_length=1, description="RFC3339 instant or interval")
    collections: list[str] = Field(..., min_length=1)
    cloud_cover_max: int = Field(settings.DEFAULT_CLOUD_COVER_MAX, ge=0, le=100)
    limit: int = Field(10, ge=1)

    @field_validator("bbox")
    @classmethod
    def _bbox_bounds(cls, v: list[float]) -> list[float]:
        return _validate_bbox(v)


class StacDownloadInput(_InputBase):
    item_id: str = Field(..., min_length=1)
    collection: str = Field(..., min_length=1)
    bands: list[str] = Field(..., min_length=1)
    bbox: list[float] | None = Field(None, min_length=4, max_length=4)
    resolution_m: int = Field(10, gt=0)

    @field_validator("bbox")
    @classmethod
    def _bbox_bounds(cls, v: list[float] | None) -> list[float] | None:
        return _validate_bbox(v)


class SpatialQueryInput(_InputBase):
    source: str = Field(
        ...,
        min_length=1,
        description=(
            "same-bucket S3 URI, or a trusted local Parquet path only when the server enables "
            "ALLOW_LOCAL_PARQUET_SOURCE with a confining LOCAL_PARQUET_SOURCE_ROOT"
        ),
    )
    geometry: dict[str, Any] = Field(
        ...,
        description=(
            "GeoJSON geometry in longitude/latitude WGS84 for a native GEOMETRY column that "
            "declares EPSG:4326, OGC:CRS84 or EPSG:2154 (any other CRS is refused). For EPSG:2154 "
            "its vertices are transformed to Lambert-93 and the segments between them "
            "reconnected, without densification or a guaranteed tolerance along polygon edges; "
            "returned geometries are longitude/latitude. distance_m is measured in EPSG:3857. A "
            "WKB BLOB column is compared in its own coordinates by intersects, within and contains"
        ),
    )
    operation: str = "intersects"  # handler whitelists intersects/within/contains/distance
    distance_m: float | None = Field(None, gt=0)
    select_columns: list[str] | None = None
    limit: int = Field(100, ge=1)
    evidence_context: EvidenceContextV1 | None = None

    @field_validator("geometry")
    @classmethod
    def _geometry_shape(cls, v: dict[str, Any]) -> dict[str, Any]:
        return _validate_geometry(v)

    @model_validator(mode="before")
    @classmethod
    def _resolve_limit_default(cls, values: Any) -> Any:
        if not isinstance(values, dict) or values.get("limit") is not None:
            return values
        resolved = dict(values)
        resolved["limit"] = (
            settings.SPATIAL_MESH_MAX_LIMIT if resolved.get("evidence_context") is not None else 100
        )
        return resolved

    @model_validator(mode="after")
    def _mesh_constraints(self) -> SpatialQueryInput:
        if self.evidence_context is None:
            return self
        if self.operation not in {"intersects", "within", "contains", "distance"}:
            raise ValueError("mesh operation must be intersects, within, contains, or distance")
        if self.operation == "distance":
            if self.distance_m is None:
                raise ValueError("mesh distance operation requires distance_m")
            if self.distance_m > 1_000_000:
                raise ValueError("mesh distance_m must be <= 1000000")
        elif self.distance_m is not None:
            raise ValueError("mesh distance_m is only valid for the distance operation")
        if self.select_columns is None or not 1 <= len(self.select_columns) <= 16:
            raise ValueError("mesh mode requires 1 to 16 select_columns")
        if len(set(self.select_columns)) != len(self.select_columns):
            raise ValueError("mesh select_columns must contain unique values")
        for column in self.select_columns:
            validate_mesh_projection_column(column, field_name="select_columns")
        if self.evidence_context.geometry_column not in self.select_columns:
            raise ValueError("mesh select_columns must include the context geometry_column")
        if self.limit > settings.SPATIAL_MESH_MAX_LIMIT:
            raise ValueError("mesh limit exceeds SPATIAL_MESH_MAX_LIMIT")
        _validate_mesh_geometry(self.geometry)
        return self


class NdviProfileInput(_InputBase):
    geometry: dict[str, Any]
    start_date: str = Field(..., min_length=1)
    end_date: str = Field(..., min_length=1)
    # 0 is valid and documented: "keep every observation" (no minimum spacing)
    interval_days: int = Field(16, ge=0)
    cloud_cover_max: int = Field(settings.DEFAULT_CLOUD_COVER_MAX, ge=0, le=100)

    @field_validator("geometry")
    @classmethod
    def _geometry_shape(cls, v: dict[str, Any]) -> dict[str, Any]:
        return _validate_geometry(v)


class VegetationStressInput(_InputBase):
    geometry: dict[str, Any]
    # A guard, not a product limit (#221): about 740,000 days would take the start date past
    # year 1 and raise OverflowError. A hundred years keeps the whole Sentinel-2 archive (2015 on)
    # in reach for decades, so no window that could find imagery is refused.
    lookback_days: int = Field(60, ge=1, le=36500)
    ndvi_threshold: float = Field(settings.DEFAULT_NDVI_STRESS_THRESHOLD, ge=-1.0, le=1.0)
    drop_threshold: float = Field(settings.DEFAULT_NDVI_DROP_THRESHOLD, ge=0.0, le=2.0)
    threshold_mode: Literal["static", "percentile"] = "static"  # FIERCE B6 level-1
    percentile_level: int = Field(10, ge=1, le=99)  # the percentile used in "percentile" mode

    @field_validator("geometry")
    @classmethod
    def _geometry_shape(cls, v: dict[str, Any]) -> dict[str, Any]:
        return _validate_geometry(v)


class ChangeDetectInput(_InputBase):
    bbox: list[float] = Field(..., min_length=4, max_length=4)
    date_before: str = Field(..., min_length=1)
    date_after: str = Field(..., min_length=1)
    change_types: list[str] | None = None
    min_change_area_ha: float = Field(settings.DEFAULT_CHANGE_MIN_AREA_HA, ge=0.0)
    confidence_threshold: float = Field(settings.DEFAULT_CHANGE_CONFIDENCE, ge=0.0, le=1.0)

    @field_validator("bbox")
    @classmethod
    def _bbox_bounds(cls, v: list[float]) -> list[float]:
        return _validate_bbox(v)

    @model_validator(mode="after")
    def _dates_are_iso_and_ordered(self) -> ChangeDetectInput:
        before = _parse_iso_date(self.date_before, field_name="date_before")
        after = _parse_iso_date(self.date_after, field_name="date_after")
        if before >= after:
            raise ValueError("date_before must be earlier than date_after")
        return self


class DiscoverPatternsInput(_InputBase):
    source: str = Field("auto", min_length=1)  # empty source is malformed (handler branches on it)
    tenant_id: str | None = None
    space_id: str | None = None
    output_key: str | None = None
    algorithm: str = "graank"  # handler whitelists the supported algorithms
    # min_support / min_rep ranges are validated *gracefully* by the handler
    # (it returns a structured {"error": ...} for out-of-range), so the contract
    # keeps them as plain floats rather than pre-empting that with a raise.
    min_support: float = 0.5
    target_column: str | None = None
    min_rep: float = 0.5
    columns: list[str] | None = None
    order_by: str | None = None
    drop_columns: list[str] | None = None
    source_kind: str | None = None
    encode_equal: bool = False
    limit: int | None = Field(None, ge=1)
    write_output: bool = True


class DiscoverPatternsCvInput(_InputBase):
    # FIERCE O3: cross-zone validation over already-materialized B4 unified tables.
    zone_ids: list[str] = Field(..., min_length=5)
    # Explicit per-zone sources. Values may be one URI/key or a list of segment URIs/keys.
    sources: dict[str, str | list[str]] | None = None
    # If sources is omitted, format this template with {zone_id} and optionally {window_label}.
    source_template: str | None = "climate/unified/{zone_id}.parquet"
    window_labels: list[str] | None = None
    algorithm: str = "graank"  # handler whitelists graank/tgrad for this first O3 ship
    min_support: float = 0.5
    n_folds: int = Field(5, ge=2)
    min_validated_zones: int = Field(1, ge=1)
    target_column: str | None = None
    min_rep: float = 0.5
    columns: list[str] | None = None
    order_by: str | None = "date"
    drop_columns: list[str] | None = None
    source_kind: str | None = None
    encode_equal: bool = False
    limit: int | None = Field(None, ge=1)
    period_start: str | None = None
    period_end: str | None = None
    # Echo source-quality caveats such as {"svalbard": ["2023_07"]}; the handler does
    # not hide those rows, it preserves the caveat in metadata for reviewers.
    truncated_sources: dict[str, list[str]] | None = None
    write_output: bool = False
    output_key: str | None = None

    @model_validator(mode="after")
    def _cv_shape(self) -> DiscoverPatternsCvInput:
        if len(set(self.zone_ids)) != len(self.zone_ids):
            raise ValueError("zone_ids must be unique")
        if self.n_folds > len(self.zone_ids):
            raise ValueError("n_folds must be <= number of zones")
        if self.sources is None and not self.source_template:
            raise ValueError("provide sources or source_template")
        if self.sources is not None:
            missing = [zone_id for zone_id in self.zone_ids if zone_id not in self.sources]
            if missing:
                raise ValueError(f"sources missing zone_ids: {missing}")
        if self.period_start is not None or self.period_end is not None:
            if self.period_start is None or self.period_end is None:
                raise ValueError("period_start and period_end must both be set or both be None")
            start = _parse_iso_date(self.period_start, field_name="period_start")
            end = _parse_iso_date(self.period_end, field_name="period_end")
            if start > end:
                raise ValueError("period_start must be <= period_end")
        return self


class Era5IngestInput(_InputBase):
    # FIERCE B2: ERA5 daily ingestion via the Open-Meteo Archive API for one zone.
    zone_id: str = Field(..., min_length=1)  # resolved to a centroid from zones/zones.geojson
    start_date: str = Field(..., min_length=1, description="ISO-8601 YYYY-MM-DD (inclusive)")
    end_date: str = Field(..., min_length=1, description="ISO-8601 YYYY-MM-DD (inclusive)")
    # None => all six terra-cognita §5.1 variables; the handler validates names gracefully.
    variables: list[str] | None = None
    temporal: Literal["daily"] = "daily"  # only daily this iteration (ERA5 daily grid)
    overwrite: bool = False  # False skips an existing parquet (idempotent re-runs)
    write_output: bool = True
    output_path: str | None = None  # explicit destination (else derived from S3 settings)


class ClimateEnrichNdviInput(_InputBase):
    # FIERCE B4: align NDVI onto the ERA5 daily grid + materialize the unified table.
    zone_id: str = Field(..., min_length=1)
    start_date: str = Field(..., min_length=1)
    end_date: str = Field(..., min_length=1)
    tolerance_days: int = Field(3, ge=0, le=31)  # nearest-date NDVI↔ERA5 join window (±N days)
    cloud_cover_max: int = Field(settings.DEFAULT_CLOUD_COVER_MAX, ge=0, le=100)
    # explicit ERA5 GeoParquet source (else derived from S3 settings)
    era5_source: str | None = None
    # explicit B3 NDVI GeoParquet source. When set, the stored series IS the input: no STAC search
    # and no COG read happen, and an unusable source is an error rather than a fallback (#177).
    # `min_length=1` so a supplied-but-empty source is refused AT THE BOUNDARY rather than being
    # read as "omitted" and silently sent to the provider (PR #188 review).
    ndvi_source: str | None = Field(None, min_length=1)
    # refuse the run when the ERA5 source cannot cover the requested window, instead of returning
    # the intersection of the two (#133).
    strict_window: bool = False
    write_output: bool = True
    output_path: str | None = None


class NdviIngestInput(_InputBase):
    # FIERCE B3: standalone Sentinel-2 NDVI zonal-stats time series for one zone via COG streaming.
    zone_id: str = Field(
        ..., min_length=1
    )  # resolved to bbox + mask geometry from zones/zones.geojson
    start_date: str = Field(..., min_length=1, description="ISO-8601 YYYY-MM-DD (inclusive)")
    end_date: str = Field(..., min_length=1, description="ISO-8601 YYYY-MM-DD (inclusive)")
    cloud_cover_max: int = Field(settings.DEFAULT_CLOUD_COVER_MAX, ge=0, le=100)
    # True => one least-cloudy scene per month (backfill-friendly cadence); False => every scene in
    # range up to the STAC item cap.
    monthly: bool = True
    write_output: bool = True
    output_path: str | None = None  # explicit destination (else climate/ndvi/{zone_id}.parquet)


class ClimatePercentilesInput(_InputBase):
    # B6 climatological percentile baseline: P10/P50/P90 ERA5 normals by zone x month.
    zone_id: str = Field(..., min_length=1)
    start_date: str | None = Field(None, min_length=1)
    end_date: str | None = Field(None, min_length=1)
    variables: list[str] | None = None
    era5_source: str | None = None
    overwrite: bool = False
    write_output: bool = True
    output_path: str | None = None
    classify_observed: dict[str, float] | None = None
    classify_month: int | None = Field(None, ge=1, le=12)


class ClimateNormalsValue(_InputBase):
    """One forecast day to classify: ``(variable, date, value)``.

    There is deliberately **no request-level month**. Each entry's month comes from its own
    ``date``, so a 7-day horizon that crosses a month boundary is classified day by day
    against the right normals instead of all seven days against one month (site-watch V0,
    §G2).

    ``value: None`` is how a provider gap is expressed. It yields ``status: "missing"`` with
    no band and no category — never a zero, which would classify as a real dry day.
    """

    variable: str = Field(..., min_length=1)
    date: str = Field(..., min_length=1)  # ISO YYYY-MM-DD; the handler parses and reports
    value: float | None = None


class ClimateNormalsInput(_InputBase):
    # Site-watch V0 §G2: the read-only view of climate/normals/{zone_id}.parquet, plus an
    # optional per-entry classification.
    #
    # `science_shared_read` means "an allowlisted scientific dataset through a STABLE ALIAS",
    # so there is no source parameter here and no destination either. An earlier revision
    # took a `normals_source` path by analogy with `climate_enrich_ndvi`'s `ndvi_source`
    # (#188); that analogy does not hold. `ndvi_source` sits on an `internal_write` operator
    # tool, where a physical locator is the point. On a facade read it would let a caller
    # select a sibling artifact inside the same storage prefix, which is exactly what the
    # profile promises callers cannot do (#203 review). The zone resolves to its canonical
    # artifact and nothing else.
    zone_id: str = Field(..., min_length=1)
    variables: list[str] | None = None  # None => every variable stored for the zone
    values: list[ClimateNormalsValue] | None = None  # None => normals only, no series


class ForecastQueryInput(_InputBase):
    # FIERCE B5: short-term forecast for a zone centroid OR an explicit point, via the
    # provider router with automatic fallback. EXACTLY ONE of {zone_id} / {lat+lon}.
    zone_id: str | None = Field(None, min_length=1)  # resolved to a centroid (climate plane)
    latitude: float | None = Field(None, ge=-90, le=90)
    longitude: float | None = Field(None, ge=-180, le=180)
    variables: list[str] | None = None  # None => all six §5.1 vars; handler validates names
    # default = the max horizon; handler clamps the UPPER bound gracefully to FORECAST_MAX_DAYS.
    forecast_days: int = Field(settings.FORECAST_MAX_DAYS, ge=1)
    provider: str = "auto"  # handler whitelists auto/ai/openmeteo/nim (graceful {"error"})
    write_output: bool = False  # a forecast is ephemeral — default no persist
    output_path: str | None = None  # REQUIRED for write_output in point-mode (no zone_id)

    @model_validator(mode="after")
    def _zone_xor_point(self) -> ForecastQueryInput:
        has_zone = self.zone_id is not None
        # latitude/longitude must be provided together (a half-point is malformed)
        if (self.latitude is None) != (self.longitude is None):
            raise ValueError("latitude and longitude must both be provided for a point")
        has_point = self.latitude is not None and self.longitude is not None
        if has_zone == has_point:  # both, or neither — no implicit precedence
            raise ValueError("provide exactly one of zone_id OR (latitude AND longitude)")
        return self


class PatternAlertInput(_InputBase):
    mode: Literal["live", "saved_inputs_j1", "historical_case_j1"] = "live"
    saved_inputs: B6SavedInputs | None = None
    historical_inputs: B6HistoricalInputs | None = None
    # FIERCE Palier 4 MVP: read a validated O3 library, query B5 forecast, and
    # return experimental alerts. Calibrated confidence intentionally remains unset.
    zone_id: str = Field(..., min_length=1)
    patterns_source: str = Field(
        "climate/o3/holdout/q3_2023/tgrad_validated.parquet",
        min_length=1,
        description=(
            "same-bucket S3 key/URI for an O3 validated pattern library, or a local path only "
            "under the server-configured LOCAL_PARQUET_SOURCE_ROOT"
        ),
    )
    source_kind: str = "parquet"
    allow_local_source: bool = Field(
        False,
        description=(
            "per-call local-source opt-in; the server must also enable "
            "ALLOW_LOCAL_PARQUET_SOURCE and configure LOCAL_PARQUET_SOURCE_ROOT"
        ),
    )
    forecast_horizon_days: int = Field(settings.FORECAST_MAX_DAYS, ge=1)
    provider: str = "auto"
    min_support: float = Field(0.5, ge=0.0, le=1.0)
    min_stability: float = Field(0.5, ge=0.0, le=1.0)
    min_validated_zones: int = Field(1, ge=1)
    require_zone_validated: bool = True
    decay_halflife_days: float = Field(10.0, gt=0.0)
    max_alerts: int = Field(10, ge=1, le=100)

    @model_validator(mode="after")
    def _mode_inputs(self):
        if self.mode == "live":
            if self.saved_inputs is not None or self.historical_inputs is not None:
                raise ValueError("private inputs require their explicit mode")
            return self
        if self.mode == "saved_inputs_j1":
            if self.saved_inputs is None or self.historical_inputs is not None:
                raise ValueError("saved_inputs_j1 requires only saved_inputs")
        elif self.historical_inputs is None or self.saved_inputs is not None:
            raise ValueError("historical_case_j1 requires only historical_inputs")
        # The saved mode has frozen J1 gates and no acquisition options. Reject
        # legacy flags rather than silently accepting a requested threshold/path.
        for name, field in type(self).model_fields.items():
            if (
                name not in {"zone_id", "mode", "saved_inputs", "historical_inputs"}
                and getattr(self, name) != field.default
            ):
                raise ValueError("legacy options are unavailable in saved mode")
        return self


# --------------------------------------------------------------------------- #
# Output contracts (published schema; verified against live shapes in tests)
# --------------------------------------------------------------------------- #


class FeatureCollectionOutput(_OutputBase):
    """The shared GeoJSON shape returned by stac_search / spatial_query / change_detect."""

    # `features` is required on success — a meaningful published contract (the *wired* schema
    # strips `required` so the {"error": ...} return still validates; see output_schema_for).
    type: str = "FeatureCollection"
    features: list[dict[str, Any]]
    metadata: dict[str, Any] = Field(default_factory=dict)


class StacSearchOutput(FeatureCollectionOutput):
    pass


class SpatialQueryOutput(FeatureCollectionOutput):
    evidence: GeoAIEvidenceV1 | None = None


class ChangeDetectOutput(FeatureCollectionOutput):
    pass


class StacDownloadOutput(_OutputBase):
    item_id: str  # required on success (the wired schema strips required — see output_schema_for)
    collection: str
    datetime: str | None = None
    bands: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class NdviProfileEntry(_OutputBase):
    # date is None when the STAC item carries no datetime (producer emits
    # `item.datetime.isoformat() if item.datetime else None`), so the published
    # contract must allow null — else a strict consumer rejects a valid payload.
    date: str | None = None
    ndvi_mean: float
    ndvi_std: float | None = None
    ndvi_min: float | None = None
    ndvi_max: float | None = None
    cloud_cover: float | None = None
    item_id: str | None = None


class NdviProfileOutput(_OutputBase):
    geometry: dict[str, Any]  # required on success (the wired schema strips required)
    profile: list[NdviProfileEntry] = Field(default_factory=list)
    statistics: dict[str, Any] = Field(default_factory=dict)
    alerts: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class VegetationStressOutput(_OutputBase):
    geometry: dict[str, Any]  # required on success (the wired schema strips required)
    analysis: dict[str, Any] = Field(default_factory=dict)
    alerts: list[dict[str, Any]] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class DiscoverPatternsOutput(_OutputBase):
    patterns: list[dict[str, Any]]  # required on success (the wired schema strips required)
    statistics: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class DiscoverPatternsCvOutput(_OutputBase):
    patterns: list[dict[str, Any]]  # validated PatternLibrary records on success
    folds: list[dict[str, Any]] = Field(default_factory=list)
    output: str | None = None
    statistics: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class Era5IngestOutput(_OutputBase):
    zone_id: str  # required on success (the wired schema strips required — see output_schema_for)
    start_date: str | None = None
    end_date: str | None = None
    n_days: int = 0
    variables: list[str] = Field(default_factory=list)  # normalized names actually ingested
    output: str | None = None  # the GeoParquet destination written (None if write_output=False)
    completeness: dict[str, Any] = Field(default_factory=dict)  # expected vs received per variable
    statistics: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ClimateEnrichNdviOutput(_OutputBase):
    zone_id: str  # required on success (the wired schema strips required)
    rows: list[dict[str, Any]] = Field(default_factory=list)  # unified (zone,date) DataFrame-ready
    output: str | None = None  # the unified GeoParquet destination (None if write_output=False)
    statistics: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class NdviIngestOutput(_OutputBase):
    zone_id: str  # required on success (the wired schema strips required — see output_schema_for)
    observations: list[dict[str, Any]] = Field(default_factory=list)  # per-scene NDVI zonal stats
    n_observations: int = 0
    output: str | None = None  # the NDVI GeoParquet destination (None if write_output=False)
    statistics: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ClimatePercentilesOutput(_OutputBase):
    zone_id: str  # required on success (the wired schema strips required)
    normals: list[dict[str, Any]] = Field(default_factory=list)
    output: str | None = None
    variables: list[str] = Field(default_factory=list)
    statistics: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
    anomaly: dict[str, Any] | None = None


class ClimateNormalsReference(_OutputBase):
    """The `watch-input/v1` ``reference`` block GeoAI owns, and nothing more than that block.

    The schema's ``reference`` object is `additionalProperties: false`, so this is exactly
    ``kind``, ``grain``, ``source`` and ``window`` (the "1991-2020" label) minus ``evidence_id``
    — the only part of the contract the tool fills in wholesale. ``evidence_id`` is deliberately
    absent — a real ``geoai_ev_…`` comes from an authorized ``dataset.describe`` over a
    registered dataset. An authorized client can supply it after exact read binding.
    The window's dates
    (``window_start``/``window_end``) are NOT part of this block; they travel in
    ``metadata.normals_window`` instead (#218, #172).
    """

    kind: Literal["climatological_percentiles"] = "climatological_percentiles"
    grain: str
    source: Literal["climate_normals"] = "climate_normals"
    window: str | None = None  # e.g. "1991-2020"


class ClimateNormalsUnclassified(_OutputBase):
    """One ``values[]`` entry that could not become a schema-valid ``series[]`` row.

    Declared rather than left to ``extra="allow"``: these ARE the result for an arid month or
    a v1 artifact, and a published schema that documents only the successes describes a
    payload the tool does not return (#203 review).
    """

    variable: str
    date: str
    reason: str


class ClimateNormalsMetadata(_OutputBase):
    """Logical identity and original-byte binding required on every successful read."""

    dataset: str = Field(pattern=r"^science:normals:[A-Za-z0-9._-]+$")
    artifact_version: str = Field(pattern=r"^sha256:[0-9a-f]{64}$", min_length=71, max_length=71)
    artifact_size_bytes: int = Field(strict=True, gt=0)


class ClimateNormalsOutput(_OutputBase):
    zone_id: str  # required on success (the wired schema strips required)
    normals: list[dict[str, Any]] = Field(default_factory=list)  # monthly P10/P50/P90 rows
    # `watch-input/v1` `$defs/entry` objects. Left as mappings on purpose: the authority for
    # their shape is `docs/design/contracts/watch-input.v1.schema.json`, the file Deposium
    # holds byte-identically, and `tests/test_climate_normals.py` validates every emitted
    # entry against it. Restating that schema as a Pydantic model would create a second
    # source of truth for a cross-repo contract, and the two would drift on the first
    # revision.
    series: list[dict[str, Any]] = Field(default_factory=list)
    unclassified: list[ClimateNormalsUnclassified] = Field(default_factory=list)
    reference: ClimateNormalsReference | None = None
    # Mandatory and always null: R15 and the contract both require it, and there is no measured
    # skill behind a raw-forecast classification. Typed as the literal rather than defaulted to
    # it, so a future handler cannot put a number here without the model rejecting it.
    confidence: Literal[None] = None
    caveats: list[str] = Field(default_factory=list)  # the doctrine limits, plain sentences
    statistics: dict[str, Any] = Field(default_factory=dict)
    metadata: ClimateNormalsMetadata


class ForecastQueryOutput(_OutputBase):
    zone_id: str | None = None  # set in zone mode
    point: dict[str, float] | None = None  # {latitude, longitude} in point mode
    # `forecast` is required on success — the per-day rows (the wired schema strips required so a
    # handler {"error": ...} still validates; see output_schema_for).
    forecast: list[dict[str, Any]]
    statistics: dict[str, Any] = Field(default_factory=dict)
    output: str | None = None  # dated GeoParquet destination (None if write_output=False)
    metadata: dict[str, Any] = Field(default_factory=dict)  # source, provider_chain, horizon_days…


class B6HistoricalSource(_InputBase):
    """Exact pinned native JSON bytes for independent connector result binding."""

    library_json: str = Field(min_length=1, max_length=256 * 1024)
    forecast_json: str = Field(min_length=1, max_length=256 * 1024)


class PatternAlertOutput(_OutputBase):
    historical_source: B6HistoricalSource | None = Field(
        default=None, exclude_if=lambda v: v is None
    )
    level2_result: B6Level2Output | None = None
    zone_id: str
    alerts: list[dict[str, Any]] = Field(default_factory=list)
    statistics: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


# --------------------------------------------------------------------------- #
# B6 level 2 — explanatory index (#30, contract b6-explanatory/1.0.0)
#
# The offline `src.cli.b6_explanatory` command writes this contract; the explicit
# saved_inputs_j1 mode returns it through pattern_alert for the private N2 adapter.
# The index ranks how strongly the current forecast supports
# cross-validated library patterns; it is NOT a probability or a calibrated confidence,
# which is why both of those fields are typed as permanently null here.
# --------------------------------------------------------------------------- #

B6_FORMULA_ID = "b6-explanatory/1.0.0"
B6_FORMULA = "explanatory_index = support x stability x forecast_agreement x lag_decay"
B6ExclusionReason = Literal[
    "not_cross_validated",
    "zone_not_validated",
    "low_support",
    "low_stability",
    "lag_outside_horizon",
    "no_forecast_variable",
    "insufficient_forecast_values",
]
# Evaluation order: a pattern is counted under the FIRST gate it fails.
B6_EXCLUSION_REASONS: tuple[str, ...] = get_args(B6ExclusionReason)
B6NoConclusionReason = Literal[
    "empty_library",
    "no_admissible_pattern",
    "all_lags_outside_horizon",
    "missing_forecast_data",
]


class _B6Result(BaseModel):
    """Unlike tool outputs, a B6 result forbids extra keys and is immutable: an added
    ``probability`` or ``confidence`` on any typed field must fail validation, not travel as
    tolerated metadata. Free dict fields (``forecast_evidence``, ``trace.forecast``,
    ``code_ref``) still accept arbitrary keys."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    def model_copy(self, *, update=None, deep: bool = False):
        # Pydantic does not validate ``update``; revalidate so a copy obeys the same contract.
        if not update:
            return super().model_copy(deep=deep)
        return type(self).model_validate({**self.model_dump(), **update})


class B6Params(_InputBase):
    """Gates and decay. Defaults are proposed, not ratified (#30), so every trace records them."""

    min_folds: int = Field(5, ge=1)
    min_support: float = Field(0.5, ge=0.0, le=1.0)
    min_stability: float = Field(0.5, ge=0.0, le=1.0)
    halflife_days: float = Field(10.0, gt=0.0)
    max_patterns: int = Field(10, ge=1, le=100)


class B6Factor(_B6Result):
    name: Literal["support", "stability", "forecast_agreement", "lag_decay"]
    value: float = Field(..., ge=0.0, le=1.0)
    derivation: str


class B6PatternResult(_B6Result):
    rule_id: str
    pattern_id: int
    pattern: str
    antecedent: str
    consequent: str
    lag_days: float
    explanatory_index: float = Field(..., ge=0.0, le=1.0)
    factors: list[B6Factor]
    forecast_evidence: list[dict[str, Any]] = Field(default_factory=list)
    explanation: str


class B6LibraryIdentity(_B6Result):
    """Which verified D-A bytes were scored; refused upstream if any digest differs."""

    library_version: str
    manifest_sha256: str
    json_sha256: str
    parquet_sha256: str
    schema_version: Literal["1.0.0"]
    zone_id: str
    algorithm: str
    area_status: str
    qualification: str


class B6PatternTrace(_B6Result):
    """Per-pattern validation provenance, kept for every read pattern, eligible or not."""

    pattern_id: int
    n_folds: int | None
    zones_validated: list[str]
    period_start: date | None
    period_end: date | None
    excluded: B6ExclusionReason | None


class B6HistoricalPoint(_B6Result):
    lat: Literal[43.445]
    lon: Literal[4.92]


class B6HistoricalCaseTrace(_B6Result):
    """Engineering replay context, explicitly separate from scientific adoption."""

    case_id: Literal["q3-2023-fos-native-recurrence"]
    case_version: Literal["2026.10.27.1"]
    case_manifest_sha256: Sha256
    preparation_manifest_sha256: Sha256
    forecast_profile: Literal["rule15-temperature"]
    producer_zone_id: Literal["cross_zone"]
    target_zone_id: Literal["fos_sur_mer"]
    forecast_point: B6HistoricalPoint
    registry_sha256: Sha256
    source_bundle_sha256: Sha256
    native_parquet_sha256: Sha256
    extraction_manifest_sha256: Sha256
    forecast_sha256: Sha256
    model_issued_at: str
    retrieved_at: str
    acquisition: Literal["none"] = "none"
    scientific_adoption: Literal["pending"] = "pending"
    ready_for_scoring: Literal[False] = False
    live_activation: Literal[False] = False
    limitations: list[str]


class B6Trace(_B6Result):
    historical_case: B6HistoricalCaseTrace | None = Field(
        default=None, exclude_if=lambda v: v is None
    )
    formula_id: Literal["b6-explanatory/1.0.0"]
    formula_version: Literal["1.0.0"]
    formula: str = B6_FORMULA
    params: B6Params
    params_status: Literal["proposed_unratified"] = "proposed_unratified"
    library: B6LibraryIdentity
    forecast: dict[str, Any]
    effective_horizon_days: float
    code_ref: dict[str, Any] | None
    patterns: list[B6PatternTrace]
    computed_at: str


class B6Level2Output(_B6Result):
    zone_id: str
    algorithm: str
    status: Literal["explained", "no_conclusion"]
    reason: B6NoConclusionReason | None
    patterns: list[B6PatternResult]
    n_patterns_read: int = Field(..., ge=0)
    n_eligible: int = Field(..., ge=0)
    exclusions: dict[B6ExclusionReason, NonNegativeInt]
    # Typed as None, not Optional[float]: any value is a validation error, by design.
    confidence: None = None
    event_probability: None = None
    trace: B6Trace

    @model_validator(mode="after")
    def _consistent(self):
        if set(self.exclusions) != set(B6_EXCLUSION_REASONS):
            raise ValueError("every exclusion reason must be counted, even when zero")
        if (self.status == "explained") != (self.reason is None):
            raise ValueError("status 'explained' has no reason; 'no_conclusion' needs one")
        if (self.status == "explained") != bool(self.patterns):
            raise ValueError("only an 'explained' result lists patterns")
        if self.n_eligible + sum(self.exclusions.values()) != self.n_patterns_read:
            raise ValueError("eligible plus excluded patterns must equal the patterns read")
        if not len(self.patterns) <= self.n_eligible <= len(self.trace.patterns):
            raise ValueError("listed, eligible and traced pattern counts are inconsistent")
        if len(self.trace.patterns) != self.n_patterns_read:
            raise ValueError("every read pattern must be traced")
        return self


class ErrorResponse(_OutputBase):
    """The error variant every tool may return instead of its success shape."""

    error: str
    metadata: dict[str, Any] | None = None


# --------------------------------------------------------------------------- #
# Registry + helpers
# --------------------------------------------------------------------------- #

ToolProfile = Literal[
    "storage_free_read",
    "tenant_read",
    "tenant_derive",
    "science_shared_read",
    "internal_write",
    "experimental",
]
TimeoutClass = Literal["short", "standard", "long"]
StabilityStatus = Literal["stable", "partial", "experimental"]


class ToolCapability(BaseModel):
    """Connector-facing metadata; authorization and timeout enforcement live upstream."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    profile: ToolProfile
    timeout_class: TimeoutClass
    stability_status: StabilityStatus


TOOL_CONTRACTS: dict[str, dict[str, type[BaseModel]]] = {
    "solar_portfolio_collect": {"input": SolarPilotInput, "output": SolarPilotOutput},
    "stac_search": {"input": StacSearchInput, "output": StacSearchOutput},
    "stac_download": {"input": StacDownloadInput, "output": StacDownloadOutput},
    "spatial_query": {"input": SpatialQueryInput, "output": SpatialQueryOutput},
    "ndvi_profile": {"input": NdviProfileInput, "output": NdviProfileOutput},
    "vegetation_stress": {"input": VegetationStressInput, "output": VegetationStressOutput},
    "change_detect": {"input": ChangeDetectInput, "output": ChangeDetectOutput},
    "discover_patterns": {"input": DiscoverPatternsInput, "output": DiscoverPatternsOutput},
    "discover_patterns_cv": {
        "input": DiscoverPatternsCvInput,
        "output": DiscoverPatternsCvOutput,
    },
    "era5_ingest": {"input": Era5IngestInput, "output": Era5IngestOutput},
    "ndvi_ingest": {"input": NdviIngestInput, "output": NdviIngestOutput},
    "climate_enrich_ndvi": {"input": ClimateEnrichNdviInput, "output": ClimateEnrichNdviOutput},
    "climate_percentiles": {
        "input": ClimatePercentilesInput,
        "output": ClimatePercentilesOutput,
    },
    "climate_normals": {"input": ClimateNormalsInput, "output": ClimateNormalsOutput},
    "forecast_query": {"input": ForecastQueryInput, "output": ForecastQueryOutput},
    "pattern_alert": {"input": PatternAlertInput, "output": PatternAlertOutput},
}


TOOL_CAPABILITIES: dict[str, ToolCapability] = {
    "solar_portfolio_collect": ToolCapability(
        profile="experimental", timeout_class="standard", stability_status="experimental"
    ),
    "stac_search": ToolCapability(
        profile="storage_free_read", timeout_class="short", stability_status="stable"
    ),
    "stac_download": ToolCapability(
        profile="storage_free_read", timeout_class="short", stability_status="stable"
    ),
    "spatial_query": ToolCapability(
        profile="tenant_read", timeout_class="standard", stability_status="stable"
    ),
    "ndvi_profile": ToolCapability(
        profile="storage_free_read", timeout_class="standard", stability_status="partial"
    ),
    "vegetation_stress": ToolCapability(
        profile="storage_free_read", timeout_class="standard", stability_status="partial"
    ),
    "change_detect": ToolCapability(
        profile="storage_free_read", timeout_class="standard", stability_status="partial"
    ),
    "discover_patterns": ToolCapability(
        profile="tenant_derive", timeout_class="long", stability_status="stable"
    ),
    "discover_patterns_cv": ToolCapability(
        profile="internal_write", timeout_class="long", stability_status="partial"
    ),
    "era5_ingest": ToolCapability(
        profile="internal_write", timeout_class="long", stability_status="partial"
    ),
    "ndvi_ingest": ToolCapability(
        profile="internal_write", timeout_class="long", stability_status="partial"
    ),
    "climate_enrich_ndvi": ToolCapability(
        profile="internal_write", timeout_class="long", stability_status="partial"
    ),
    "climate_percentiles": ToolCapability(
        profile="internal_write", timeout_class="long", stability_status="stable"
    ),
    # The read-only counterpart: the normals were reachable only through the operator tool
    # above, which writes. This one reads a stored artifact and returns; `partial` because the
    # V0 ships without the client-zone path (§G1) and the archived replay (§G3).
    "climate_normals": ToolCapability(
        profile="science_shared_read", timeout_class="short", stability_status="partial"
    ),
    "forecast_query": ToolCapability(
        profile="storage_free_read", timeout_class="short", stability_status="stable"
    ),
    "pattern_alert": ToolCapability(
        profile="experimental", timeout_class="standard", stability_status="experimental"
    ),
}

TOOL_CAPABILITIES_SCHEMA_VERSION = "geoai-tool-capabilities/v1"

TOOL_PROFILE_DEFINITIONS: dict[str, str] = {
    "storage_free_read": "Read/compute from public upstreams; persistence must remain disabled for facade reads.",
    "tenant_read": "Read tenant data only through an authenticated, server-resolved locator.",
    "tenant_derive": "Derive tenant artifacts only with an authorized locator and server-owned destination.",
    "science_shared_read": "Read an allowlisted scientific dataset through a stable alias.",
    "internal_write": "Operator/worker workflow; not enabled for general facade callers.",
    "experimental": "Explicitly experimental and disabled by default in product facades.",
}

TIMEOUT_CLASS_DEFINITIONS: dict[str, str] = {
    "short": "Bounded lookup or provider fallback suitable for a short synchronous budget.",
    "standard": "Bounded query or mixed I/O suitable for an explicit synchronous budget.",
    "long": "Batch/CPU/archive workflow requiring an operator or job-specific budget.",
}

STABILITY_STATUS_DEFINITIONS: dict[str, str] = {
    "stable": "Shipped for its documented scope.",
    "partial": "Shipped primitive with explicit scientific or operational scope still incomplete.",
    "experimental": "Prototype semantics; no production-grade or calibrated claim.",
}


def check_input(_model: type[BaseModel], /, **kwargs: Any) -> None:
    """Validate tool inputs against the declared contract.

    Raises ``pydantic.ValidationError`` on violation. Values are not rebound:
    FastMCP already binds the signature defaults/types from the handler's type
    hints, so this enforces only the *additional* structural/range constraints
    the contract declares (bbox arity, numeric bounds, non-empty identifiers).
    """
    _model(**kwargs)


def output_schema_for(name: str) -> dict[str, Any]:
    """The single-object JSON Schema to wire into ``@mcp.tool(output_schema=...)``.

    MCP requires the wired output schema to be a single object type (no ``anyOf``). FastMCP
    validates the tool's return against it and **raises on a missing required field** (verified),
    so we hand it a **lenient** copy of the success model: the top-level ``required`` list is
    stripped, leaving ``extra="allow"`` + all-optional. The success shape passes, and a handler's
    ``{"error": ...}`` return also passes (only a wrong *type* on a present field raises). The
    *strict* success model (with its real ``required`` fields) is what :func:`contracts_json_schema`
    publishes to consumers.
    """
    schema = TOOL_CONTRACTS[name]["output"].model_json_schema()
    schema.pop("required", None)
    return schema


def _hoist_defs(*schemas: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Pop each schema's ``$defs`` into one shared mapping so a ``$ref: '#/$defs/X'`` embedded
    under ``anyOf`` still resolves from the document root (Pydantic puts ``$defs`` at the schema
    top level, which would otherwise be orphaned once nested)."""
    defs: dict[str, Any] = {}
    branches: list[dict[str, Any]] = []
    for s in schemas:
        s = dict(s)
        defs.update(s.pop("$defs", {}) or {})
        branches.append(s)
    return branches, defs


def contracts_json_schema() -> dict[str, dict[str, Any]]:
    """Export every tool's input + output JSON Schema (the consumer-facing contract).

    The ``output`` is published as the **strict** success model ``anyOf`` :class:`ErrorResponse` —
    every tool may return its success shape (with its real required fields) **or** a structured
    ``{"error": ...}`` — so a consumer validating against this contract accepts both while still
    getting a meaningful success schema. (The schema *wired into the MCP tool* via
    :func:`output_schema_for` is the lenient single-object variant, because MCP output schemas must
    be a single object type and FastMCP validates at runtime.) Nested ``$defs`` are hoisted to the
    per-tool root so ``$ref`` s resolve under ``anyOf``.
    """
    error_schema = ErrorResponse.model_json_schema()
    out: dict[str, dict[str, Any]] = {}
    for name, models in TOOL_CONTRACTS.items():
        branches, defs = _hoist_defs(models["output"].model_json_schema(), error_schema)
        output: dict[str, Any] = {"anyOf": branches}
        if defs:
            output["$defs"] = defs
        out[name] = {"input": models["input"].model_json_schema(), "output": output}
    return out


def tool_capabilities_json() -> dict[str, Any]:
    """Export the connector catalog with strict consumer schemas and descriptive metadata."""
    contract_names = set(TOOL_CONTRACTS)
    capability_names = set(TOOL_CAPABILITIES)
    if contract_names != capability_names:
        missing = sorted(contract_names - capability_names)
        extra = sorted(capability_names - contract_names)
        raise RuntimeError(f"tool capability registry drift: missing={missing}, extra={extra}")

    schemas = contracts_json_schema()
    tools: dict[str, Any] = {}
    for name in sorted(TOOL_CONTRACTS):
        input_model = TOOL_CONTRACTS[name]["input"]
        write_output = input_model.model_fields.get("write_output")
        persistence_supported = write_output is not None
        persistence_default = False
        if write_output is not None:
            if not isinstance(write_output.default, bool):
                raise RuntimeError(f"{name}.write_output must have an explicit boolean default")
            persistence_default = write_output.default

        tools[name] = {
            **TOOL_CAPABILITIES[name].model_dump(mode="json"),
            "persistence": {
                "supported": persistence_supported,
                "default": persistence_default,
            },
            "input_schema": schemas[name]["input"],
            "output_schema": schemas[name]["output"],
        }

    return {
        "schema_version": TOOL_CAPABILITIES_SCHEMA_VERSION,
        "tool_count": len(tools),
        "metadata_only": True,
        "timeout_classes_enforced_by_geoai": False,
        "profiles": TOOL_PROFILE_DEFINITIONS,
        "timeout_classes": TIMEOUT_CLASS_DEFINITIONS,
        "stability_statuses": STABILITY_STATUS_DEFINITIONS,
        "tools": tools,
    }
