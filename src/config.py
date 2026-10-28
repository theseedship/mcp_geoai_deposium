"""Configuration management for GeoAI MCP Server."""

from pathlib import Path
from typing import Self
from urllib.parse import urlsplit

from pydantic import ValidationInfo, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_MCP_TRANSPORTS = {"http", "stdio", "sse"}
FORECAST_ARCHIVE_AUTH_HOST = "customer-previous-runs-api.open-meteo.com"
FORECAST_ENSEMBLE_AUTH_HOST = "customer-ensemble-api.open-meteo.com"


def validate_forecast_archive_auth_endpoint(base_url: str, apikey: str) -> None:
    """Reject archive credentials unless they target the approved HTTPS endpoint."""
    if not apikey.strip():
        return
    parsed = urlsplit(base_url.strip())
    approved = (
        parsed.scheme.lower() == "https"
        and parsed.hostname == FORECAST_ARCHIVE_AUTH_HOST
        and parsed.port in (None, 443)
        and parsed.path.rstrip("/") == "/v1/forecast"
        and parsed.username is None
        and parsed.password is None
        and not parsed.query
        and not parsed.fragment
    )
    if not approved:
        raise ValueError(
            "FORECAST_ARCHIVE_API_KEY requires the approved Open-Meteo Previous Runs "
            f"endpoint: https://{FORECAST_ARCHIVE_AUTH_HOST}/v1/forecast"
        )


def validate_forecast_ensemble_auth_endpoint(base_url: str, apikey: str) -> None:
    """Reject ensemble credentials unless they target the approved HTTPS endpoint."""
    if not apikey.strip():
        return
    parsed = urlsplit(base_url.strip())
    approved = (
        parsed.scheme.lower() == "https"
        and parsed.hostname == FORECAST_ENSEMBLE_AUTH_HOST
        and parsed.port in (None, 443)
        and parsed.path.rstrip("/") == "/v1/ensemble"
        and parsed.username is None
        and parsed.password is None
        and not parsed.query
        and not parsed.fragment
    )
    if not approved:
        raise ValueError(
            "FORECAST_ENSEMBLE_API_KEY requires the approved Open-Meteo Ensemble "
            f"endpoint: https://{FORECAST_ENSEMBLE_AUTH_HOST}/v1/ensemble"
        )


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
    )

    # Server - Railway provides PORT env var
    # Use 0.0.0.0 for IPv4 compatibility (Railway prefers this)
    HOST: str = "0.0.0.0"
    PORT: int = 5001

    # MCP transport: "http" (mesh / Railway / docker), "stdio" (Claude Desktop
    # spawns the process directly), or "sse". Lets a developer connect this
    # repo's MCP server standalone without the mesh.
    MCP_TRANSPORT: str = "http"

    @field_validator("MCP_TRANSPORT")
    @classmethod
    def _check_transport(cls, v: str) -> str:
        """Fail fast at startup on an invalid transport rather than deep inside
        mcp.run() at runtime."""
        lv = v.lower()
        if lv not in _MCP_TRANSPORTS:
            raise ValueError(f"MCP_TRANSPORT must be one of {sorted(_MCP_TRANSPORTS)}, got {v!r}")
        return lv

    # STAC Catalogs
    STAC_URL: str = "https://earth-search.aws.element84.com/v1"
    PLANETARY_COMPUTER_URL: str = "https://planetarycomputer.microsoft.com/api/stac/v1"

    # Storage
    IMAGE_CACHE_DIR: str = "/data/images"
    MODEL_CACHE_DIR: str = "/data/models"
    DUCKDB_PATH: str = "/data/geoai.duckdb"

    # S3 (for GeoParquet)
    S3_ENDPOINT: str = ""
    S3_ACCESS_KEY: str = ""
    S3_SECRET_KEY: str = ""
    S3_BUCKET: str = ""
    S3_FORCE_PATH_STYLE: bool = True
    FORECAST_ENSEMBLE_EVIDENCE_BUCKET: str = ""
    GEOAI_MESH_AUTH_TOKEN: str = ""

    # Query Guard Rails
    SPATIAL_QUERY_MAX_LIMIT: int = 10000
    SPATIAL_MESH_MAX_LIMIT: int = 100
    SPATIAL_MESH_QUERY_TIMEOUT_S: float = 5.0
    SPATIAL_MESH_MAX_RESPONSE_BYTES: int = 8_388_608
    SPATIAL_MESH_MAX_GEOJSON_BYTES: int = 65_536
    SPATIAL_MESH_MAX_COORDINATE_POSITIONS: int = 5000
    STAC_SEARCH_MAX_ITEMS: int = 50
    # Caller-controlled local Parquet reads/writes are a trusted developer/CI
    # capability, never a transport-derived permission. Enabling them requires
    # an absolute, non-filesystem root.
    ALLOW_LOCAL_PARQUET_SOURCE: bool = False
    LOCAL_PARQUET_SOURCE_ROOT: str = ""

    @model_validator(mode="after")
    def _check_local_parquet_policy(self) -> Self:
        if not self.ALLOW_LOCAL_PARQUET_SOURCE:
            return self

        raw_root = self.LOCAL_PARQUET_SOURCE_ROOT.strip()
        if not raw_root:
            raise ValueError(
                "LOCAL_PARQUET_SOURCE_ROOT is required when ALLOW_LOCAL_PARQUET_SOURCE=true"
            )
        root = Path(raw_root).expanduser()
        if not root.is_absolute():
            raise ValueError("LOCAL_PARQUET_SOURCE_ROOT must be an absolute path")
        resolved_root = root.resolve(strict=False)
        if resolved_root == Path(resolved_root.anchor):
            raise ValueError("LOCAL_PARQUET_SOURCE_ROOT must not be the filesystem root")
        return self

    @model_validator(mode="after")
    def _check_http_data_plane_auth(self) -> Self:
        token = self.GEOAI_MESH_AUTH_TOKEN.strip()
        if token and len(token) < 32:
            raise ValueError("GEOAI_MESH_AUTH_TOKEN must contain at least 32 characters")
        return self

    @field_validator(
        "SPATIAL_MESH_MAX_LIMIT",
        "SPATIAL_MESH_MAX_RESPONSE_BYTES",
        "SPATIAL_MESH_MAX_GEOJSON_BYTES",
        "SPATIAL_MESH_MAX_COORDINATE_POSITIONS",
    )
    @classmethod
    def _check_mesh_positive_limits(cls, value: int, info: ValidationInfo) -> int:
        minimum = 1
        if value < minimum:
            raise ValueError(f"{info.field_name} must be >= {minimum}")
        if info.field_name == "SPATIAL_MESH_MAX_LIMIT" and value > 100:
            raise ValueError("SPATIAL_MESH_MAX_LIMIT must be <= 100")
        if info.field_name == "SPATIAL_MESH_MAX_RESPONSE_BYTES" and value > 8_388_608:
            raise ValueError("SPATIAL_MESH_MAX_RESPONSE_BYTES must be <= 8388608")
        if info.field_name == "SPATIAL_MESH_MAX_GEOJSON_BYTES" and value > 65_536:
            raise ValueError("SPATIAL_MESH_MAX_GEOJSON_BYTES is too large")
        if info.field_name == "SPATIAL_MESH_MAX_COORDINATE_POSITIONS" and value > 5000:
            raise ValueError("SPATIAL_MESH_MAX_COORDINATE_POSITIONS must be <= 5000")
        return value

    @field_validator("SPATIAL_MESH_QUERY_TIMEOUT_S")
    @classmethod
    def _check_mesh_timeout(cls, value: float) -> float:
        if not 0 < value <= 60:
            raise ValueError("SPATIAL_MESH_QUERY_TIMEOUT_S must be > 0 and <= 60")
        return value

    # Scientific Defaults (overridable per-call by agents)
    DEFAULT_CLOUD_COVER_MAX: int = 20
    DEFAULT_NDVI_STRESS_THRESHOLD: float = 0.3
    DEFAULT_NDVI_DROP_THRESHOLD: float = 0.2
    DEFAULT_CHANGE_CONFIDENCE: float = 0.7
    DEFAULT_CHANGE_MIN_AREA_HA: float = 0.5

    # Processing Guard Rails
    MAX_BBOX_AREA_DEG2: float = 1.0  # ~10,000 km2 max processing area
    POINT_BUFFER_DEG: float = 0.005  # ~500m buffer for point geometries
    GDAL_CONCURRENCY: int = 4  # concurrent COG band reads per request

    # Forecast plane (FIERCE B5) — short-term forecast + provider router/automatic fallback.
    # The "auto" chain is [AIFS (AI primary), physical (fallback)]: AIFS (ecmwf_aifs025_single,
    # live-verified all 6 vars incl. radiation, 5 cities, 2026-06-21) is tried first; on any outage
    # the router falls back to the reliable Open-Meteo physical best-match (models OMITTED).
    # models=auto is INVALID. AIFS horizon is ~15 d; physical is 16 d (pin provider="openmeteo").
    FORECAST_API_BASE_URL: str = (
        "https://api.open-meteo.com/v1/forecast"  # → customer-api.* for prod
    )
    FORECAST_API_KEY: str = ""  # apikey for the commercial tier — FORECAST client ONLY (the Archive
    #   client uses the separate, explicit settings below)
    FORECAST_ARCHIVE_API_BASE_URL: str = "https://previous-runs-api.open-meteo.com/v1/forecast"
    FORECAST_ARCHIVE_API_KEY: str = ""  # Set with the matching commercial Previous Runs endpoint.
    FORECAST_ENSEMBLE_API_BASE_URL: str = "https://ensemble-api.open-meteo.com/v1/ensemble"
    FORECAST_ENSEMBLE_API_KEY: str = ""  # Separate entitlement and customer endpoint.

    @model_validator(mode="after")
    def _check_forecast_archive_auth_endpoint(self) -> Self:
        validate_forecast_archive_auth_endpoint(
            self.FORECAST_ARCHIVE_API_BASE_URL,
            self.FORECAST_ARCHIVE_API_KEY,
        )
        return self

    @model_validator(mode="after")
    def _check_forecast_ensemble_auth_endpoint(self) -> Self:
        validate_forecast_ensemble_auth_endpoint(
            self.FORECAST_ENSEMBLE_API_BASE_URL,
            self.FORECAST_ENSEMBLE_API_KEY,
        )
        return self

    FORECAST_ENABLE_AI: bool = (
        True  # AIFS AI-primary ON (verified 2026-06-21); physical is the fallback
    )
    FORECAST_AI_MODEL: str = (
        "ecmwf_aifs025_single"  # the VERIFIED daily AIFS token (ecmwf_aifs025=null)
    )
    FORECAST_PHYSICAL_MODEL: str = (
        ""  # "" => OMIT models (Open-Meteo best-match); models=auto is INVALID
    )
    FORECAST_TIMEOUT_S: float = (
        5.0  # short — a forecast is ~16 rows; fail fast to the next provider
    )
    FORECAST_MAX_DAYS: int = 16  # Open-Meteo Forecast horizon cap (AIFS caps lower, ~15 d)
    NVIDIA_NIM_FORECAST_URL: str = ""  # gates the NIM adapter seam (design §4.7); empty = skipped

    # GPU
    USE_GPU: bool = True
    CUDA_VISIBLE_DEVICES: str = "0"

    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "json"


settings = Settings()
