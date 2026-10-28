"""The internal descriptor entry point never reopens a replaced source pathname."""

import os
from pathlib import Path

import duckdb
import pytest
from shapely import Point, to_wkb

from src.utils.geoparquet import (
    GeoParquetInspectionError,
    WkbCompatibility,
    inspect_spatial_parquet_fd,
)


@pytest.fixture
def parquet(tmp_path):
    source = tmp_path / "source.parquet"
    with duckdb.connect(":memory:") as conn:
        conn.execute("CREATE TABLE points (value INTEGER, geometry BLOB)")
        conn.execute("INSERT INTO points VALUES (1, ?)", [bytes(to_wkb(Point(3.8, 43.6)))])
        conn.execute("COPY points TO ? (FORMAT PARQUET)", [str(source)])
    return source


def test_descriptor_inspection_survives_source_path_replacement(parquet):
    with parquet.open("rb") as source:
        parquet.unlink()
        parquet.write_bytes(b"not-the-verified-file")
        inspection = inspect_spatial_parquet_fd(
            source.fileno(),
            compatibility=WkbCompatibility(geometry_column="geometry", crs="EPSG:4326"),
        )
        assert inspection.table.row_count == 1
        assert inspection.spatial.bbox == [3.8, 43.6, 3.8, 43.6]
        assert source.read(4) == b"PAR1"  # borrowed handle remains open
    assert parquet.read_bytes() == b"not-the-verified-file"


def test_descriptor_inspection_requires_regular_file(tmp_path):
    descriptor = os.open(tmp_path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        with pytest.raises(GeoParquetInspectionError, match="regular file"):
            inspect_spatial_parquet_fd(descriptor)
        assert Path(f"/proc/self/fd/{descriptor}").is_dir()
    finally:
        os.close(descriptor)


def test_descriptor_inspection_keeps_spatial_contract(parquet):
    with parquet.open("rb") as source:
        with pytest.raises(GeoParquetInspectionError, match="WkbCompatibility"):
            inspect_spatial_parquet_fd(source.fileno())
        assert not source.closed
