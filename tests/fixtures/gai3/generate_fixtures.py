"""Regenerate the portable GAI-3 Parquet and expected-manifest fixtures."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import duckdb
from rasterio.crs import CRS
from shapely import Point, to_wkb

ROOT = Path(__file__).resolve().parent
CREATED_AT = "2026-07-11T00:00:00Z"
BBOX = [3.8767, -1.2921, 36.8219, 43.6108]
ROWS = [
    (1, "montpellier", bytes(to_wkb(Point(3.8767, 43.6108)))),
    (2, "nairobi", bytes(to_wkb(Point(36.8219, -1.2921)))),
    (3, "unknown", None),
]


def _geo_metadata() -> dict[str, Any]:
    return {
        "version": "1.1.0",
        "primary_column": "geometry",
        "columns": {
            "geometry": {
                "encoding": "WKB",
                "geometry_types": ["Point"],
                "crs": json.loads(CRS.from_epsg(4326)._projjson()),
                "bbox": BBOX,
            }
        },
    }


def _write_parquet(path: Path, *, formal: bool) -> None:
    path.unlink(missing_ok=True)
    conn = duckdb.connect(":memory:")
    try:
        conn.execute("CREATE TABLE fixture(id INTEGER, label VARCHAR, geometry BLOB)")
        conn.executemany("INSERT INTO fixture VALUES (?, ?, ?)", ROWS)
        if formal:
            conn.execute(
                "COPY fixture TO ? (FORMAT PARQUET, KV_METADATA {geo: ?})",
                [path.as_posix(), json.dumps(_geo_metadata(), separators=(",", ":"))],
            )
        else:
            conn.execute("COPY fixture TO ? (FORMAT PARQUET)", [path.as_posix()])
    finally:
        conn.close()


def _inspection(*, formal: bool) -> dict[str, Any]:
    return {
        "source_kind": "formal_geoparquet" if formal else "internal_wkb",
        "artifact_format": "geoparquet" if formal else "parquet",
        "geoparquet_version": "1.1.0" if formal else None,
        "table": {
            "row_count": 3,
            "columns": [
                {"name": "id", "type": "INTEGER"},
                {"name": "label", "type": "VARCHAR"},
                {"name": "geometry", "type": "BLOB"},
            ],
        },
        "spatial": {
            "geometry_column": "geometry",
            "encoding": "WKB",
            "crs": "EPSG:4326",
            "bbox": BBOX,
        },
        "geometry_types": ["Point"],
        "geometry_count": 2,
        "null_geometry_count": 1,
        "empty_geometry_count": 0,
        "metadata_bbox_verified": True if formal else None,
        "caveats": (
            []
            if formal
            else [
                "GeoParquet geo metadata is absent; geometry encoding and CRS come from "
                "the explicit internal compatibility contract."
            ]
        ),
    }


def _expected(path: Path, *, formal: bool, file_id: int) -> dict[str, Any]:
    inspection = _inspection(formal=formal)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    stem = "formal-wkb" if formal else "internal-wkb"
    return {
        "fixture_version": "gai3-fixture/v1",
        "compatibility": None if formal else {"geometry_column": "geometry", "crs": "EPSG:4326"},
        "inspection": inspection,
        "manifest": {
            "schema_version": "dataset-manifest/v1",
            "dataset_id": f"fixture-{stem}",
            "dataset_kind": "app_geoparquet" if formal else "science_dataset",
            "artifact": {
                "format": inspection["artifact_format"],
                "key": f"fixtures/gai3/{stem}.parquet",
                "version": f"sha256:{digest}",
            },
            "table": inspection["table"],
            "spatial": inspection["spatial"],
            "provenance": {
                "tenant_id": "fixture-tenant",
                "folder_id": "fixture-space",
                "file_id": file_id,
                "document_id": f"00000000-0000-4000-8000-{file_id:012d}",
                "producer": "gai3-fixture-generator",
                "created_at": CREATED_AT,
            },
        },
    }


def main() -> None:
    for formal, stem, file_id in (
        (True, "formal-wkb", 4201),
        (False, "internal-wkb", 4202),
    ):
        parquet_path = ROOT / f"{stem}.parquet"
        expected_path = ROOT / f"{stem}.expected.json"
        _write_parquet(parquet_path, formal=formal)
        expected_path.write_text(
            json.dumps(_expected(parquet_path, formal=formal, file_id=file_id), indent=2) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
