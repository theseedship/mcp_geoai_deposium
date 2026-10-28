"""Offline preflight for a canonical WMO normals science manifest (#223).

This command neither registers a dataset nor reads a provider, object store, or database.
The operator supplies an already-authorized local copy, its independently trusted digest,
and real registration provenance. Inspection uses a private snapshot of the verified bytes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from calendar import monthrange
from pathlib import Path
from typing import Any

import duckdb
from pydantic import ValidationError
from shapely import from_wkb
from shapely.errors import GEOSException

from src.cli._normals_local_io import ConfinedLocalIO
from src.config import settings
from src.manifests import ManifestProvenance
from src.utils.geoparquet import (
    WkbCompatibility,
    build_dataset_manifest,
    inspect_spatial_parquet_fd,
)
from src.utils.logging import setup_logging
from src.utils.openmeteo import NORMALIZED_VARS
from src.utils.s3_endpoint import redact_text

_ROOT = Path(__file__).resolve().parents[2]
_ZONES = _ROOT / "zones" / "zones.geojson"
_SHA256 = re.compile(r"[0-9a-fA-F]{64}\Z")
_ZONE_ID = re.compile(r"[A-Za-z0-9._-]+\Z")
_GLOBS = frozenset("*?[]{}")
_MAX_SOURCE_BYTES = 8 * 1024 * 1024
_MAX_PROVENANCE_BYTES = 64 * 1024
_START, _END, _LABEL = "1991-01-01", "2020-12-31", "WMO_1991_2020"
_MONTH_COUNTS = {
    month: sum(monthrange(year, month)[1] for year in range(1991, 2021)) for month in range(1, 13)
}
_COORD_ABS_TOL = 1e-9
_REQUIRED_COLUMNS = frozenset(
    {
        "zone_id",
        "month",
        "variable",
        "p10",
        "p50",
        "p90",
        "n",
        "mean",
        "window_start",
        "window_end",
        "window_label",
        "lat",
        "lon",
        "p10_wet",
        "p50_wet",
        "p90_wet",
        "n_wet",
        "wet_day_threshold_mm",
        "schema_version",
        "geometry",
    }
)
_ROW_COLUMNS = (
    "zone_id",
    "month",
    "variable",
    "p10",
    "p50",
    "p90",
    "n",
    "mean",
    "window_start",
    "window_end",
    "window_label",
    "lat",
    "lon",
    "p10_wet",
    "p50_wet",
    "p90_wet",
    "n_wet",
    "wet_day_threshold_mm",
    "schema_version",
    "geometry",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="normals-science-manifest")
    parser.add_argument(
        "--source", required=True, help="authorized local canonical normals Parquet"
    )
    parser.add_argument("--zone-id", required=True, help="public panel or pilot zone ID")
    parser.add_argument(
        "--expected-sha256", required=True, help="independently trusted source digest"
    )
    parser.add_argument(
        "--provenance-json", required=True, help="operator-supplied real IDs and dates"
    )
    parser.add_argument(
        "--work-root", required=True, help="existing local directory for private snapshot"
    )
    parser.add_argument("--output", required=True, help="new local dataset-manifest JSON path")
    return parser


def _local_path(raw: str, *, role: str, suffix: str | None = None) -> Path:
    if not raw or any(char in raw for char in _GLOBS) or "://" in raw:
        raise ValueError(f"{role} must be one concrete local path without glob or URI syntax")
    given = Path(raw)
    if not given.is_absolute():
        raise ValueError(f"{role} must be an absolute local path")
    if suffix is not None and given.suffix.lower() != suffix:
        raise ValueError(f"{role} must end in {suffix}")
    # Keep the operator's path intact: resolving symlinks here would discard
    # the no-follow traversal enforced by ConfinedLocalIO at the actual I/O.
    return given


def _check_zone(zone_id: str) -> tuple[float, float]:
    if not _ZONE_ID.fullmatch(zone_id):
        raise ValueError("zone_id is not a safe scientific zone key")
    registry = json.loads(_ZONES.read_text(encoding="utf-8"))
    for feature in registry.get("features", []):
        properties = feature.get("properties", {})
        if properties.get("zone_id") != zone_id or properties.get("role") not in {"panel", "pilot"}:
            continue
        bbox = properties.get("bbox")
        if (
            not isinstance(bbox, list)
            or len(bbox) != 4
            or not all(type(value) in {int, float} and math.isfinite(value) for value in bbox)
        ):
            raise ValueError("scientific zone must have a finite four-coordinate bbox")
        return ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)
    raise ValueError("zone_id must be a known public panel or pilot scientific zone")


def _validate_rows(snapshot: Path, inspection: Any, *, zone_id: str) -> None:
    expected_lon, expected_lat = _check_zone(zone_id)
    columns = {column.name for column in inspection.table.columns}
    missing = _REQUIRED_COLUMNS - columns
    if missing:
        raise ValueError(f"normals v2 columns missing: {', '.join(sorted(missing))}")
    if inspection.source_kind != "internal_wkb" or inspection.geometry_types != ["Point"]:
        raise ValueError("canonical normals require internal WKB Point geometry")
    if inspection.null_geometry_count or inspection.table.row_count != 72:
        raise ValueError("canonical normals require 72 non-null geometry rows")

    with duckdb.connect(":memory:") as conn:
        selected = ", ".join(_ROW_COLUMNS)
        rows = conn.execute(f"SELECT {selected} FROM read_parquet(?)", [str(snapshot)]).fetchall()

    expected_grid = {(variable, month) for variable in NORMALIZED_VARS for month in range(1, 13)}
    found_grid: set[tuple[str, int]] = set()
    for values in rows:
        row = dict(zip(_ROW_COLUMNS, values, strict=True))
        if row["zone_id"] != zone_id:
            raise ValueError("normal row belongs to another zone")
        variable, month = row["variable"], row["month"]
        if type(month) is not int or (variable, month) not in expected_grid:
            raise ValueError("normal row has an unexpected variable or month")
        key = (variable, month)
        if key in found_grid:
            raise ValueError("normal row duplicates a variable/month key")
        found_grid.add(key)
        if row["schema_version"] != 2 or type(row["schema_version"]) is not int:
            raise ValueError("all normals rows must be schema v2")
        if (row["window_start"], row["window_end"], row["window_label"]) != (_START, _END, _LABEL):
            raise ValueError("all normals rows must use the canonical WMO 1991-2020 window")
        count = row["n"]
        if type(count) is not int or count != _MONTH_COUNTS[month]:
            raise ValueError("each normal month requires full WMO 1991-2020 calendar-day coverage")
        all_days = (row["p10"], row["p50"], row["p90"])
        if not all(type(value) in {int, float} and math.isfinite(value) for value in all_days):
            raise ValueError("all-days percentiles must be finite")
        if not all_days[0] <= all_days[1] <= all_days[2]:
            raise ValueError("all-days percentiles must be ordered")
        if not all(
            type(row[name]) in {int, float} and math.isfinite(row[name])
            for name in ("mean", "lat", "lon")
        ):
            raise ValueError("normal mean and coordinates must be finite")
        if not (
            math.isclose(row["lon"], expected_lon, rel_tol=0, abs_tol=_COORD_ABS_TOL)
            and math.isclose(row["lat"], expected_lat, rel_tol=0, abs_tol=_COORD_ABS_TOL)
        ):
            raise ValueError("normal coordinates must match the scientific zone bbox midpoint")
        try:
            point = from_wkb(row["geometry"], on_invalid="raise")
            if point.geom_type != "Point" or point.is_empty or point.has_z:
                raise ValueError("normal geometry must be a non-empty 2D Point")
            geometry_matches = math.isclose(
                point.x, expected_lon, rel_tol=0, abs_tol=_COORD_ABS_TOL
            ) and math.isclose(point.y, expected_lat, rel_tol=0, abs_tol=_COORD_ABS_TOL)
        except (GEOSException, TypeError) as exc:
            raise ValueError("normal geometry must be a valid WKB Point") from exc
        if not geometry_matches:
            raise ValueError("normal WKB Point must match the scientific zone bbox midpoint")
        wet_count = row["n_wet"]
        if variable == "precipitation":
            if type(wet_count) is not int or not 0 <= wet_count <= count:
                raise ValueError("precipitation n_wet is required and must fit n (zero is valid)")
            if row["wet_day_threshold_mm"] != 1.0:
                raise ValueError("precipitation wet-day threshold must be 1.0 mm")
            wet_percentiles = (row["p10_wet"], row["p50_wet"], row["p90_wet"])
            if wet_count == 0 and any(value is not None for value in wet_percentiles):
                raise ValueError("zero wet days must not have wet-day percentiles")
            if wet_count > 0 and any(value is None for value in wet_percentiles):
                raise ValueError("measured wet days require wet-day percentiles")
            if wet_count > 0 and (
                not all(
                    type(value) in {int, float} and math.isfinite(value)
                    for value in wet_percentiles
                )
                or not 1.0 <= wet_percentiles[0] <= wet_percentiles[1] <= wet_percentiles[2]
            ):
                raise ValueError("wet-day percentiles must be finite, at least 1.0 mm and ordered")
        elif any(
            row[name] is not None
            for name in ("n_wet", "p10_wet", "p50_wet", "p90_wet", "wet_day_threshold_mm")
        ):
            raise ValueError("non-precipitation wet-day columns must be null")
    if found_grid != expected_grid:
        raise ValueError("normal variable/month grid is incomplete")


def run(args: argparse.Namespace) -> dict[str, Any]:
    _check_zone(args.zone_id)
    if not _SHA256.fullmatch(args.expected_sha256):
        raise ValueError("expected-sha256 must contain exactly 64 hex digits")
    expected = args.expected_sha256.lower()
    source = _local_path(args.source, role="source", suffix=".parquet")
    provenance_path = _local_path(args.provenance_json, role="provenance JSON", suffix=".json")
    work_root = _local_path(args.work_root, role="work root")
    output = _local_path(args.output, role="output", suffix=".json")
    if len({source, provenance_path, output}) != 3:
        raise ValueError("source, provenance JSON and output must be distinct paths")
    if not settings.ALLOW_LOCAL_PARQUET_SOURCE:
        raise ValueError("local preflight is disabled by ALLOW_LOCAL_PARQUET_SOURCE")
    raw_root = settings.LOCAL_PARQUET_SOURCE_ROOT.strip()
    if not raw_root:
        raise ValueError("LOCAL_PARQUET_SOURCE_ROOT is required for local preflight")

    with ConfinedLocalIO(Path(raw_root)) as local:
        source_bytes = local.read_bytes(source, role="source", limit=_MAX_SOURCE_BYTES)
        digest = hashlib.sha256(source_bytes).hexdigest()
        if digest != expected:
            raise ValueError("source SHA-256 does not match the independently expected digest")
        provenance_bytes = local.read_bytes(
            provenance_path, role="provenance JSON", limit=_MAX_PROVENANCE_BYTES
        )
        provenance = ManifestProvenance.model_validate_json(provenance_bytes)
        with local.private_snapshot(work_root, source_bytes) as snapshot:
            inspection = inspect_spatial_parquet_fd(
                snapshot.fd,
                compatibility=WkbCompatibility(geometry_column="geometry", crs="EPSG:4326"),
            )
            _validate_rows(snapshot.path, inspection, zone_id=args.zone_id)
            manifest = build_dataset_manifest(
                inspection,
                dataset_id=f"science:normals:{args.zone_id}",
                dataset_kind="science_dataset",
                artifact_key=f"climate/normals/{args.zone_id}.parquet",
                artifact_version=f"sha256:{digest}",
                provenance=provenance,
            )

        payload = (manifest.model_dump_json(indent=2) + "\n").encode("utf-8")
        local.publish_new(output, payload)
    return {"status": "preflight_only", "output": str(output), "sha256": digest}


def main(argv: list[str] | None = None) -> int:
    setup_logging()
    args = build_parser().parse_args(argv)
    try:
        result = run(args)
    except ValidationError:
        # Pydantic's rendered error includes input values, potentially the whole provenance.
        print(json.dumps({"status": "refused", "error": "invalid manifest or provenance schema"}))
        return 1
    except (ValueError, OSError, duckdb.Error) as exc:
        error = redact_text(str(exc), settings.S3_ENDPOINT)
        print(json.dumps({"status": "refused", "error": error}, sort_keys=True))
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
