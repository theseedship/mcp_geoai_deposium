import hashlib
import json
from pathlib import Path

import duckdb
import pytest
from rasterio.crs import CRS
from shapely import Point, to_wkb

from src.manifests import ManifestProvenance
from src.utils.geoparquet import (
    GeoParquetInspectionError,
    WkbCompatibility,
    build_dataset_manifest,
    inspect_spatial_parquet,
)
from src.utils.path_policy import resolve_local_parquet_path as enforce_local_path_policy

FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "gai3"
_NO_GEO_METADATA = object()


@pytest.fixture(autouse=True)
def _isolate_inspection_tests_from_deployment_path_policy(monkeypatch):
    def resolve_fixture_path(value: str, *, role: str) -> Path:
        del role
        return Path(value).expanduser().resolve(strict=False)

    monkeypatch.setattr(
        "src.utils.geoparquet.resolve_local_parquet_path",
        resolve_fixture_path,
    )


def _fixture(stem: str) -> dict:
    return json.loads((FIXTURE_ROOT / f"{stem}.expected.json").read_text())


def _manifest_from_fixture(inspection, fixture: dict):
    expected = fixture["manifest"]
    return build_dataset_manifest(
        inspection,
        dataset_id=expected["dataset_id"],
        dataset_kind=expected["dataset_kind"],
        artifact_key=expected["artifact"]["key"],
        artifact_version=expected["artifact"]["version"],
        provenance=ManifestProvenance.model_validate(expected["provenance"]),
    )


def _valid_geo_metadata() -> dict:
    return {
        "version": "1.1.0",
        "primary_column": "geometry",
        "columns": {
            "geometry": {
                "encoding": "WKB",
                "geometry_types": ["Point"],
                "crs": json.loads(CRS.from_epsg(4326)._projjson()),
                "bbox": [3.0, 43.0, 5.0, 45.0],
            }
        },
    }


def _write_parquet(
    tmp_path: Path,
    *,
    geo_metadata=_NO_GEO_METADATA,
    geometry: bytes | None = None,
) -> Path:
    path = tmp_path / "fixture.parquet"
    geometry = bytes(to_wkb(Point(4.0, 44.0))) if geometry is None else geometry
    conn = duckdb.connect(":memory:")
    try:
        conn.execute("CREATE TABLE fixture(id INTEGER, geometry BLOB)")
        conn.execute("INSERT INTO fixture VALUES (1, ?)", [geometry])
        if geo_metadata is _NO_GEO_METADATA:
            conn.execute("COPY fixture TO ? (FORMAT PARQUET)", [path.as_posix()])
        else:
            raw_metadata = (
                geo_metadata if isinstance(geo_metadata, str) else json.dumps(geo_metadata)
            )
            conn.execute(
                "COPY fixture TO ? (FORMAT PARQUET, KV_METADATA {geo: ?})",
                [path.as_posix(), raw_metadata],
            )
    finally:
        conn.close()
    return path


def test_formal_geoparquet_fixture_produces_frozen_manifest():
    fixture = _fixture("formal-wkb")
    fixture_path = FIXTURE_ROOT / "formal-wkb.parquet"

    inspection = inspect_spatial_parquet(fixture_path)
    manifest = _manifest_from_fixture(inspection, fixture)

    assert inspection.model_dump(mode="json") == fixture["inspection"]
    assert manifest.model_dump(mode="json") == fixture["manifest"]
    assert (
        manifest.artifact.version
        == f"sha256:{hashlib.sha256(fixture_path.read_bytes()).hexdigest()}"
    )


def test_internal_wkb_fixture_produces_frozen_manifest_with_explicit_contract():
    fixture = _fixture("internal-wkb")
    fixture_path = FIXTURE_ROOT / "internal-wkb.parquet"
    compatibility = WkbCompatibility.model_validate(fixture["compatibility"])

    inspection = inspect_spatial_parquet(
        fixture_path,
        compatibility=compatibility,
    )
    manifest = _manifest_from_fixture(inspection, fixture)

    assert inspection.model_dump(mode="json") == fixture["inspection"]
    assert manifest.model_dump(mode="json") == fixture["manifest"]
    assert manifest.artifact.format == "parquet"
    assert (
        manifest.artifact.version
        == f"sha256:{hashlib.sha256(fixture_path.read_bytes()).hexdigest()}"
    )


def test_internal_wkb_cannot_be_mislabeled_as_app_geoparquet():
    fixture = _fixture("internal-wkb")
    inspection = inspect_spatial_parquet(
        FIXTURE_ROOT / "internal-wkb.parquet",
        compatibility=WkbCompatibility.model_validate(fixture["compatibility"]),
    )
    expected = fixture["manifest"]

    with pytest.raises(ValueError, match="app_geoparquet"):
        build_dataset_manifest(
            inspection,
            dataset_id=expected["dataset_id"],
            dataset_kind="app_geoparquet",
            artifact_key=expected["artifact"]["key"],
            artifact_version=expected["artifact"]["version"],
            provenance=ManifestProvenance.model_validate(expected["provenance"]),
        )


def test_internal_wkb_without_explicit_compatibility_fails_closed():
    with pytest.raises(GeoParquetInspectionError, match="WkbCompatibility"):
        inspect_spatial_parquet(FIXTURE_ROOT / "internal-wkb.parquet")


def test_compatibility_cannot_override_formal_metadata():
    compatibility = WkbCompatibility(geometry_column="geometry", crs="EPSG:3857")

    with pytest.raises(GeoParquetInspectionError, match="must not override"):
        inspect_spatial_parquet(
            FIXTURE_ROOT / "formal-wkb.parquet",
            compatibility=compatibility,
        )


@pytest.mark.parametrize("crs_value", [_NO_GEO_METADATA, None])
def test_formal_geoparquet_requires_explicit_non_null_crs(tmp_path, crs_value):
    metadata = _valid_geo_metadata()
    if crs_value is _NO_GEO_METADATA:
        del metadata["columns"]["geometry"]["crs"]
    else:
        metadata["columns"]["geometry"]["crs"] = crs_value
    path = _write_parquet(tmp_path, geo_metadata=metadata)

    with pytest.raises(GeoParquetInspectionError, match="explicit non-null CRS"):
        inspect_spatial_parquet(path)


def test_formal_geoparquet_rejects_non_projjson_crs(tmp_path):
    metadata = _valid_geo_metadata()
    metadata["columns"]["geometry"]["crs"] = "EPSG:4326"
    path = _write_parquet(tmp_path, geo_metadata=metadata)

    with pytest.raises(GeoParquetInspectionError, match="PROJJSON object|invalid CRS"):
        inspect_spatial_parquet(path)


def test_malformed_geo_metadata_is_rejected_before_compatibility(tmp_path):
    path = _write_parquet(tmp_path, geo_metadata="{not-json")

    with pytest.raises(GeoParquetInspectionError, match="valid UTF-8 JSON"):
        inspect_spatial_parquet(
            path,
            compatibility=WkbCompatibility(geometry_column="geometry", crs="EPSG:4326"),
        )


def test_geo_metadata_integer_past_the_digit_limit_is_rejected_as_invalid_json(tmp_path):
    # json.dumps cannot write an integer past Python's 4300-digit limit either, so it is
    # spliced into the text; json.loads then raises a plain ValueError, not a JSONDecodeError.
    metadata = _valid_geo_metadata()
    metadata["columns"]["geometry"]["bbox"] = [3.0, 43.0, "HUGE", 45.0]
    raw_metadata = json.dumps(metadata).replace('"HUGE"', "9" * 5000)
    path = _write_parquet(tmp_path, geo_metadata=raw_metadata)

    with pytest.raises(GeoParquetInspectionError, match="valid UTF-8 JSON"):
        inspect_spatial_parquet(path)


def test_declared_geometry_types_must_cover_scanned_wkb(tmp_path):
    metadata = _valid_geo_metadata()
    metadata["columns"]["geometry"]["geometry_types"] = ["Polygon"]
    path = _write_parquet(tmp_path, geo_metadata=metadata)

    with pytest.raises(GeoParquetInspectionError, match="do not cover"):
        inspect_spatial_parquet(path)


def test_broad_declared_geometry_types_may_cover_narrower_actual_values(tmp_path):
    metadata = _valid_geo_metadata()
    metadata["columns"]["geometry"]["geometry_types"] = ["Point", "Polygon"]
    path = _write_parquet(tmp_path, geo_metadata=metadata)

    inspection = inspect_spatial_parquet(path)

    assert inspection.geometry_types == ["Point"]


def test_declared_geometry_types_must_be_unique(tmp_path):
    metadata = _valid_geo_metadata()
    metadata["columns"]["geometry"]["geometry_types"] = ["Point", "Point"]
    path = _write_parquet(tmp_path, geo_metadata=metadata)

    with pytest.raises(GeoParquetInspectionError, match="unique values"):
        inspect_spatial_parquet(path)


def test_declared_geometry_types_must_not_be_empty(tmp_path):
    metadata = _valid_geo_metadata()
    metadata["columns"]["geometry"]["geometry_types"] = []
    path = _write_parquet(tmp_path, geo_metadata=metadata)

    with pytest.raises(GeoParquetInspectionError, match="at least one type"):
        inspect_spatial_parquet(path)


def test_declared_bbox_must_contain_scanned_extent(tmp_path):
    metadata = _valid_geo_metadata()
    metadata["columns"]["geometry"]["bbox"] = [0.0, 0.0, 1.0, 1.0]
    path = _write_parquet(tmp_path, geo_metadata=metadata)

    with pytest.raises(GeoParquetInspectionError, match="does not contain"):
        inspect_spatial_parquet(path)


def test_declared_bbox_beyond_the_float_range_is_rejected_not_overflowed(tmp_path):
    metadata = _valid_geo_metadata()
    metadata["columns"]["geometry"]["bbox"] = [3.0, 43.0, 10**400, 45.0]
    path = _write_parquet(tmp_path, geo_metadata=metadata)

    with pytest.raises(GeoParquetInspectionError, match="bbox coordinates must be finite"):
        inspect_spatial_parquet(path)


def test_malformed_wkb_is_rejected_with_row_context(tmp_path):
    path = _write_parquet(tmp_path, geometry=b"not-wkb")

    with pytest.raises(GeoParquetInspectionError, match="row 0 contains malformed WKB"):
        inspect_spatial_parquet(
            path,
            compatibility=WkbCompatibility(geometry_column="geometry", crs="EPSG:4326"),
        )


def test_three_dimensional_wkb_is_rejected_without_lossy_bbox(tmp_path):
    path = _write_parquet(tmp_path, geometry=bytes(to_wkb(Point(4.0, 44.0, 10.0))))

    with pytest.raises(GeoParquetInspectionError, match="only 2D geometries"):
        inspect_spatial_parquet(
            path,
            compatibility=WkbCompatibility(geometry_column="geometry", crs="EPSG:4326"),
        )


def test_primary_native_encoding_is_explicitly_out_of_scope(tmp_path):
    metadata = _valid_geo_metadata()
    metadata["columns"]["geometry"]["encoding"] = "point"
    path = _write_parquet(tmp_path, geo_metadata=metadata)

    with pytest.raises(GeoParquetInspectionError, match="WKB primary geometry"):
        inspect_spatial_parquet(path)


def test_internal_wkb_preserves_nested_top_level_column_types(tmp_path):
    path = tmp_path / "nested.parquet"
    conn = duckdb.connect(":memory:")
    try:
        conn.execute(
            "CREATE TABLE fixture(id INTEGER, properties STRUCT(name VARCHAR, scores DOUBLE[]), "
            "geometry BLOB)"
        )
        conn.execute(
            "INSERT INTO fixture VALUES (1, {name: 'sample', scores: [1.0, 2.0]}, ?)",
            [bytes(to_wkb(Point(4.0, 44.0)))],
        )
        conn.execute("COPY fixture TO ? (FORMAT PARQUET)", [path.as_posix()])
    finally:
        conn.close()

    inspection = inspect_spatial_parquet(
        path,
        compatibility=WkbCompatibility(geometry_column="geometry", crs="EPSG:4326"),
    )

    types = {column.name: column.type for column in inspection.table.columns}
    assert types["properties"] == 'STRUCT("name" VARCHAR, scores DOUBLE[])'
    assert types["geometry"] == "BLOB"


@pytest.mark.parametrize(
    "source",
    ["https://example.test/data.parquet", "tests/fixtures/gai3/*.parquet"],
)
def test_inspector_rejects_remote_urls_and_globs(source):
    with pytest.raises(GeoParquetInspectionError):
        inspect_spatial_parquet(source)


@pytest.mark.parametrize("source_type", [str, Path])
def test_every_local_source_type_obeys_deployment_policy(monkeypatch, source_type):
    source_path = FIXTURE_ROOT / "formal-wkb.parquet"
    source = source_type(source_path.as_posix())
    monkeypatch.setattr(
        "src.utils.geoparquet.resolve_local_parquet_path",
        enforce_local_path_policy,
    )
    monkeypatch.setattr("src.utils.path_policy.settings.ALLOW_LOCAL_PARQUET_SOURCE", False)

    with pytest.raises(GeoParquetInspectionError, match="disabled"):
        inspect_spatial_parquet(source)


def test_local_string_source_works_with_explicit_allowlist(monkeypatch):
    source = (FIXTURE_ROOT / "formal-wkb.parquet").resolve()
    monkeypatch.setattr(
        "src.utils.geoparquet.resolve_local_parquet_path",
        enforce_local_path_policy,
    )
    monkeypatch.setattr("src.utils.path_policy.settings.ALLOW_LOCAL_PARQUET_SOURCE", True)
    monkeypatch.setattr(
        "src.utils.path_policy.settings.LOCAL_PARQUET_SOURCE_ROOT",
        FIXTURE_ROOT.resolve().as_posix(),
    )

    inspection = inspect_spatial_parquet(source.as_posix())

    assert inspection.source_kind == "formal_geoparquet"


@pytest.mark.parametrize(
    "source",
    ["s3://other-bucket/data.parquet", "s3://allowed-bucket/../data.parquet"],
)
def test_inspector_rejects_cross_bucket_and_traversing_s3_sources(monkeypatch, source):
    monkeypatch.setattr("src.utils.geoparquet.settings.S3_BUCKET", "allowed-bucket")

    with pytest.raises(GeoParquetInspectionError):
        inspect_spatial_parquet(source)
