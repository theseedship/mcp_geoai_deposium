"""Configuration policy tests."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.config import Settings

_LOCAL_SOURCE_ENV = (
    "ALLOW_LOCAL_PARQUET_SOURCE",
    "LOCAL_PARQUET_SOURCE_ROOT",
    "GEOAI_MESH_AUTH_TOKEN",
    "S3_ENDPOINT",
    "S3_ACCESS_KEY",
    "S3_SECRET_KEY",
    "S3_BUCKET",
    "FORECAST_ENSEMBLE_EVIDENCE_BUCKET",
    "FORECAST_ARCHIVE_API_BASE_URL",
    "FORECAST_ARCHIVE_API_KEY",
    "FORECAST_ENSEMBLE_API_BASE_URL",
    "FORECAST_ENSEMBLE_API_KEY",
)


def _settings(monkeypatch, **values) -> Settings:
    for name in _LOCAL_SOURCE_ENV:
        monkeypatch.delenv(name, raising=False)
    return Settings(_env_file=None, **values)


@pytest.mark.parametrize("transport", ["http", "stdio"])
def test_local_parquet_is_disabled_by_default_for_every_transport(monkeypatch, transport):
    configured = _settings(monkeypatch, MCP_TRANSPORT=transport)

    assert configured.ALLOW_LOCAL_PARQUET_SOURCE is False
    assert configured.LOCAL_PARQUET_SOURCE_ROOT == ""


def test_enabling_local_parquet_requires_a_root(monkeypatch):
    with pytest.raises(ValidationError, match="LOCAL_PARQUET_SOURCE_ROOT is required"):
        _settings(monkeypatch, ALLOW_LOCAL_PARQUET_SOURCE=True)


def test_local_parquet_root_must_be_absolute(monkeypatch):
    with pytest.raises(ValidationError, match="must be an absolute path"):
        _settings(
            monkeypatch,
            ALLOW_LOCAL_PARQUET_SOURCE=True,
            LOCAL_PARQUET_SOURCE_ROOT="fixtures",
        )


def test_local_parquet_root_must_not_be_filesystem_root(monkeypatch):
    with pytest.raises(ValidationError, match="must not be the filesystem root"):
        _settings(
            monkeypatch,
            ALLOW_LOCAL_PARQUET_SOURCE=True,
            LOCAL_PARQUET_SOURCE_ROOT="/",
        )


def test_rooted_local_parquet_can_be_enabled_explicitly(monkeypatch, tmp_path):
    configured = _settings(
        monkeypatch,
        ALLOW_LOCAL_PARQUET_SOURCE=True,
        LOCAL_PARQUET_SOURCE_ROOT=tmp_path.as_posix(),
        MCP_TRANSPORT="stdio",
    )

    assert configured.ALLOW_LOCAL_PARQUET_SOURCE is True
    assert configured.LOCAL_PARQUET_SOURCE_ROOT == tmp_path.as_posix()


def test_mesh_token_shape_is_validated_when_present(monkeypatch):
    with pytest.raises(ValidationError, match="at least 32 characters"):
        _settings(
            monkeypatch,
            MCP_TRANSPORT="http",
            S3_BUCKET="deposium",
            GEOAI_MESH_AUTH_TOKEN="too-short",
        )

    configured = _settings(
        monkeypatch,
        MCP_TRANSPORT="http",
        S3_BUCKET="deposium",
        GEOAI_MESH_AUTH_TOKEN="g" * 32,
    )
    assert configured.GEOAI_MESH_AUTH_TOKEN == "g" * 32


def test_http_data_plane_can_be_configured_without_a_token(monkeypatch):
    configured = _settings(monkeypatch, MCP_TRANSPORT="http", S3_BUCKET="deposium")

    assert configured.GEOAI_MESH_AUTH_TOKEN == ""


def test_stdio_data_plane_does_not_require_http_auth(monkeypatch):
    configured = _settings(monkeypatch, MCP_TRANSPORT="stdio", S3_BUCKET="deposium")
    assert configured.GEOAI_MESH_AUTH_TOKEN == ""


def test_mesh_limits_have_bounded_defaults(monkeypatch):
    configured = _settings(monkeypatch)

    assert configured.SPATIAL_MESH_MAX_LIMIT == 100
    assert configured.SPATIAL_MESH_QUERY_TIMEOUT_S == 5.0
    assert configured.SPATIAL_MESH_MAX_RESPONSE_BYTES == 8_388_608
    assert configured.SPATIAL_MESH_MAX_GEOJSON_BYTES == 65_536
    assert configured.SPATIAL_MESH_MAX_COORDINATE_POSITIONS == 5000


def test_previous_runs_archive_has_explicit_endpoint_and_key(monkeypatch):
    configured = _settings(monkeypatch)

    assert configured.FORECAST_ARCHIVE_API_BASE_URL == (
        "https://previous-runs-api.open-meteo.com/v1/forecast"
    )
    assert configured.FORECAST_ARCHIVE_API_KEY == ""

    configured = _settings(
        monkeypatch,
        FORECAST_ARCHIVE_API_BASE_URL=(
            "https://customer-previous-runs-api.open-meteo.com/v1/forecast"
        ),
        FORECAST_ARCHIVE_API_KEY="archive-secret",
    )
    assert configured.FORECAST_ARCHIVE_API_BASE_URL == (
        "https://customer-previous-runs-api.open-meteo.com/v1/forecast"
    )
    assert configured.FORECAST_ARCHIVE_API_KEY == "archive-secret"


def test_previous_runs_archive_rejects_key_for_unapproved_endpoint(monkeypatch):
    with pytest.raises(ValidationError, match="approved Open-Meteo Previous Runs endpoint"):
        _settings(
            monkeypatch,
            FORECAST_ARCHIVE_API_KEY="archive-secret",
        )


def test_ensemble_archive_has_explicit_endpoint_and_key(monkeypatch):
    configured = _settings(monkeypatch)

    assert configured.FORECAST_ENSEMBLE_API_BASE_URL == (
        "https://ensemble-api.open-meteo.com/v1/ensemble"
    )
    assert configured.FORECAST_ENSEMBLE_API_KEY == ""

    configured = _settings(
        monkeypatch,
        FORECAST_ENSEMBLE_API_BASE_URL=("https://customer-ensemble-api.open-meteo.com/v1/ensemble"),
        FORECAST_ENSEMBLE_API_KEY="ensemble-secret",
    )
    assert configured.FORECAST_ENSEMBLE_API_BASE_URL == (
        "https://customer-ensemble-api.open-meteo.com/v1/ensemble"
    )
    assert configured.FORECAST_ENSEMBLE_API_KEY == "ensemble-secret"


def test_ensemble_evidence_bucket_is_explicit_and_disabled_by_default(monkeypatch):
    configured = _settings(monkeypatch)
    assert configured.FORECAST_ENSEMBLE_EVIDENCE_BUCKET == ""

    configured = _settings(
        monkeypatch,
        FORECAST_ENSEMBLE_EVIDENCE_BUCKET="geoai-ensemble-evidence",
    )
    assert configured.FORECAST_ENSEMBLE_EVIDENCE_BUCKET == "geoai-ensemble-evidence"


def test_ensemble_archive_rejects_key_for_unapproved_endpoint(monkeypatch):
    with pytest.raises(ValidationError, match="approved Open-Meteo Ensemble endpoint"):
        _settings(
            monkeypatch,
            FORECAST_ENSEMBLE_API_KEY="ensemble-secret",
        )


@pytest.mark.parametrize(
    "field,value",
    [
        ("SPATIAL_MESH_MAX_LIMIT", 101),
        ("SPATIAL_MESH_QUERY_TIMEOUT_S", 0),
        ("SPATIAL_MESH_MAX_RESPONSE_BYTES", 0),
        ("SPATIAL_MESH_MAX_GEOJSON_BYTES", 0),
        ("SPATIAL_MESH_MAX_COORDINATE_POSITIONS", 0),
        ("SPATIAL_MESH_MAX_RESPONSE_BYTES", 8_388_609),
        ("SPATIAL_MESH_MAX_GEOJSON_BYTES", 65_537),
        ("SPATIAL_MESH_MAX_COORDINATE_POSITIONS", 5001),
    ],
)
def test_mesh_limits_reject_unsafe_values(monkeypatch, field, value):
    with pytest.raises(ValidationError):
        _settings(monkeypatch, **{field: value})
