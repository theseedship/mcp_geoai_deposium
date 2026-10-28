"""Server startup preparation tests."""

from __future__ import annotations

import pytest

from src import server


class _FakeConnection:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


def _clear_data_plane(monkeypatch) -> None:
    monkeypatch.setattr(server.settings, "ALLOW_LOCAL_PARQUET_SOURCE", False)
    for name in ("S3_ENDPOINT", "S3_ACCESS_KEY", "S3_SECRET_KEY", "S3_BUCKET"):
        monkeypatch.setattr(server.settings, name, "")


def test_public_only_startup_skips_duckdb_extension_preparation(monkeypatch) -> None:
    _clear_data_plane(monkeypatch)

    def unexpected_connection(*_args, **_kwargs):
        raise AssertionError("public-only startup must not open DuckDB")

    monkeypatch.setattr(server, "get_duckdb_connection", unexpected_connection)

    assert server._prepare_mesh_extensions() is False


def test_http_data_plane_preflight_rejects_missing_token(monkeypatch) -> None:
    _clear_data_plane(monkeypatch)
    monkeypatch.setattr(server.settings, "MCP_TRANSPORT", "http")
    monkeypatch.setattr(server.settings, "GEOAI_MESH_AUTH_TOKEN", "")
    monkeypatch.setattr(server.settings, "S3_BUCKET", "deposium")

    with pytest.raises(RuntimeError, match="GEOAI_MESH_AUTH_TOKEN is required"):
        server._preflight_mesh_auth()


def test_http_data_plane_preflight_accepts_present_token(monkeypatch) -> None:
    _clear_data_plane(monkeypatch)
    monkeypatch.setattr(server.settings, "MCP_TRANSPORT", "http")
    monkeypatch.setattr(server.settings, "GEOAI_MESH_AUTH_TOKEN", "g" * 32)
    monkeypatch.setattr(server.settings, "S3_BUCKET", "deposium")

    server._preflight_mesh_auth()


def test_stdio_data_plane_preflight_allows_missing_token(monkeypatch) -> None:
    _clear_data_plane(monkeypatch)
    monkeypatch.setattr(server.settings, "MCP_TRANSPORT", "stdio")
    monkeypatch.setattr(server.settings, "GEOAI_MESH_AUTH_TOKEN", "")
    monkeypatch.setattr(server.settings, "S3_BUCKET", "deposium")

    server._preflight_mesh_auth()


def test_public_only_http_preflight_allows_missing_token(monkeypatch) -> None:
    _clear_data_plane(monkeypatch)
    monkeypatch.setattr(server.settings, "MCP_TRANSPORT", "http")
    monkeypatch.setattr(server.settings, "GEOAI_MESH_AUTH_TOKEN", "")

    server._preflight_mesh_auth()


def test_s3_data_plane_prepares_spatial_and_httpfs_extensions(monkeypatch) -> None:
    _clear_data_plane(monkeypatch)
    monkeypatch.setattr(server.settings, "S3_BUCKET", "deposium")
    connection = _FakeConnection()
    calls: list[tuple[str, bool]] = []

    def fake_connection(path: str, *, load_httpfs: bool):
        calls.append((path, load_httpfs))
        return connection

    monkeypatch.setattr(server, "get_duckdb_connection", fake_connection)

    assert server._prepare_mesh_extensions() is True
    assert calls == [(":memory:", True)]
    assert connection.closed is True


class _RecordingLog:
    """Captures what the startup line would have handed structlog."""

    def __init__(self) -> None:
        self.events: list[tuple[str, dict]] = []

    def info(self, event: str, **fields) -> None:
        self.events.append((event, fields))


def test_the_startup_line_does_not_log_the_stac_userinfo(monkeypatch) -> None:
    """A private STAC deployment is reached as `https://user:token@stac.internal/v1`.

    Logging `settings.STAC_URL` verbatim put that token into structlog output, the container
    log and every CI transcript that captures stdout. "A log line is not a run manifest" is a
    distinction the attacker reading the log does not make; it is the same one-line fix.
    """
    _clear_data_plane(monkeypatch)
    monkeypatch.setattr(server.settings, "MCP_TRANSPORT", "stdio")
    monkeypatch.setattr(
        server.settings, "STAC_URL", "https://stac-reader:sup3r-s3cret@stac.internal/v1"
    )
    recorded = _RecordingLog()
    monkeypatch.setattr(server, "log", recorded)
    monkeypatch.setattr(server.mcp, "run", lambda **_kwargs: None)

    server.main()

    startup = [fields for event, fields in recorded.events if event == "Starting GeoAI MCP Server"]
    assert len(startup) == 1
    assert "sup3r-s3cret" not in repr(startup[0])
    assert "stac-reader" not in repr(startup[0])
    # The address itself still gets logged — that is what the field is for.
    assert startup[0]["stac_url"] == "https://stac.internal/v1"
