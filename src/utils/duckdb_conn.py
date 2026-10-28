"""Shared DuckDB connection factory with the spatial extension + S3/RustFS.

Used by both ``src/tools/spatial.py`` (GeoParquet spatial queries) and
``src/patterns/parquet_bridge.py`` (loading tabular Parquet for so4gp). The
object store is RustFS (S3-compatible); credentials come from the S3_*
settings and are sent via parameterized SET commands.

``s3_endpoint`` receives an ADDRESS and nothing else. httpfs quotes whatever it
was given IN FULL in every error it raises, so a ``user:password@`` userinfo in
``S3_ENDPOINT`` put the store's password into any traceback, stderr line or CI
log produced by an exception that does not pass through a redacting boundary.
httpfs has no use for it either: the credentials go in separately, three lines
later, via ``s3_access_key_id`` and ``s3_secret_access_key``.

``S3_ENDPOINT`` is therefore parsed against the ACCEPT grammar in
:mod:`src.utils.s3_endpoint` — ``[scheme://]host[:port]`` and nothing else —
and a value outside it is REFUSED rather than trimmed down to the part of it
that looked address-shaped. See :func:`~src.utils.s3_endpoint.sanitize_endpoint`
for the grammar and for the table of shapes that trimming used to accept.

Refusing is a security measure, so it FAILS CLOSED. Three states have to be told
apart, and the middle one used to be silently merged into the first: no endpoint
configured (empty ``S3_ENDPOINT`` — the legitimate AWS-default deployment), an
endpoint that parses (configure it), and an endpoint that was configured but
does not parse. Skipping the ``SET`` in that last case leaves httpfs on its
built-in AWS default, and the very next read sends ``S3_ACCESS_KEY``/
``S3_SECRET_KEY`` to ``s3.amazonaws.com`` in a SigV4 ``Authorization`` header —
the store's credentials, over the public internet, to a third party, because a
value was misspelled. It raises instead.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

import duckdb

from src.config import settings
from src.utils.s3_endpoint import (
    ENDPOINT_REFUSAL,
    S3EndpointConfigurationError,
    names_an_authority,
    sanitize_endpoint,
)

__all__ = [
    "S3EndpointConfigurationError",
    "duckdb_session",
    "get_duckdb_connection",
]

_spatial_installed = False
_httpfs_installed = False


def _httpfs_endpoint_setting() -> tuple[str, bool] | None:
    """``(the value for SET s3_endpoint, whether TLS is on)``, or ``None`` when none configured.

    ``None`` means one thing only — ``S3_ENDPOINT`` is unset — and the caller then leaves httpfs
    on its AWS default, which is the correct behaviour for a real AWS deployment. A configured
    endpoint that yields no address is NOT that case and must not be allowed to look like it, so
    it raises :class:`S3EndpointConfigurationError` before any connection is opened.

    The scheme strip happens HERE, before guard 2, so that the guard sees the same string the
    ``SET`` will. That ordering is the fix: ``http://`` passed the old truthiness test as an
    address and became ``''`` one line later, and ``SET s3_endpoint = ''`` is not "no endpoint",
    it is "the AWS default" — with the store's credentials attached.
    """
    raw = settings.S3_ENDPOINT
    # GUARD 0 — the entry condition, and the one the other two sit above. "Unset" has to mean
    # exactly unset: `raw.strip() or ""` merged a whitespace-only value ('   ', a tab, an NBSP,
    # an ideographic space) and a non-str into the same `return None` as an absent one, so a
    # misspelled setting took the AWS-default path with the credentials attached — the very
    # defect the guards below exist to stop, entered above them. Only None and the empty string
    # are unset; anything else was configured and has to parse.
    if raw is None or raw == "":
        return None
    if not isinstance(raw, str):
        raise S3EndpointConfigurationError(ENDPOINT_REFUSAL)
    configured = raw.strip()
    if not configured:
        raise S3EndpointConfigurationError(ENDPOINT_REFUSAL)
    # Guard 1: the parser refuses anything it cannot read a host out of.
    address, _credentials_removed = sanitize_endpoint(configured)
    if not address:
        raise S3EndpointConfigurationError(ENDPOINT_REFUSAL)

    # DuckDB s3_endpoint expects host:port, not a full URL. The accept grammar normalised the
    # scheme to lower case, so this is an exact prefix test.
    # A protocol-relative `//host:9000` keeps its marker in the PUBLISHED address, but `//` is
    # address punctuation and not part of a host: httpfs used to receive `//rustfs:9000`
    # verbatim, marker and all.
    endpoint, use_ssl = address, True
    if address.startswith("http://"):
        endpoint, use_ssl = address[len("http://") :], False
    elif address.startswith("https://"):
        endpoint = address[len("https://") :]
    elif address.startswith("//"):
        endpoint = address[len("//") :]

    # Guard 2: re-derived from the EXACT string that is about to be executed, not trusted from
    # guard 1. The fail-open has been closed three times and reopened twice, each time because
    # the single guard in place was reasoned about at a different level than the value DuckDB
    # finally received. It reuses the accept grammar rather than keeping a second, hand-written
    # host test: the second copy is what drifted — it validated the host and NEVER the port, so
    # `http://host:99999`, `http://host:abc` and `http://admin:secret` all passed it.
    if not names_an_authority(endpoint):
        raise S3EndpointConfigurationError(ENDPOINT_REFUSAL)
    return endpoint, use_ssl


def get_duckdb_connection(
    db_path: str | None = None,
    *,
    load_httpfs: bool = False,
    install_extensions: bool = True,
) -> duckdb.DuckDBPyConnection:
    """Get a DuckDB connection with the spatial extension loaded and S3 configured.

    Args:
        db_path: database file to open; defaults to ``settings.DUCKDB_PATH`` (the
            persistent prod path). Pass ``":memory:"`` for ephemeral read-only work
            (e.g. the mining CLI on a local file or S3 source — no persistence needed).
        load_httpfs: force DuckDB's S3 filesystem extension for explicit ``s3://`` paths even
            when global S3 bucket/endpoint settings are not configured.
        install_extensions: install missing extensions before loading them. Isolated mesh
            workers disable this because the service startup/build preinstalls extensions and
            request deadlines must never contain extension downloads.

    Raises:
        S3EndpointConfigurationError: ``load_httpfs`` is set and ``S3_ENDPOINT`` holds a value
            no address can be parsed out of. Raised BEFORE the connection is opened, so nothing
            is sent anywhere on this path.
    """
    global _httpfs_installed, _spatial_installed

    # Resolve the endpoint first. A misconfigured one must abort before any connection exists,
    # not after a connection has been opened and left to fall back to the AWS default.
    endpoint_setting = _httpfs_endpoint_setting() if load_httpfs else None

    conn = duckdb.connect(db_path or settings.DUCKDB_PATH)

    # Install spatial extension once, load per-connection
    if not _spatial_installed and install_extensions:
        conn.execute("INSTALL spatial;")
    conn.execute("LOAD spatial;")
    _spatial_installed = True

    # Load httpfs before any s3:// read/write. Deployment S3 settings may exist
    # during local-only work, so callers opt in when they actually touch S3.
    use_httpfs = load_httpfs
    if use_httpfs:
        if not _httpfs_installed and install_extensions:
            conn.execute("INSTALL httpfs;")
        conn.execute("LOAD httpfs;")
        _httpfs_installed = True

    # Configure S3 if credentials are set — use parameterized SET commands.
    # `endpoint_setting` was sanitized BEFORE the scheme was stripped, by the same parser the
    # manifests and the archive CLI use, so the userinfo is gone from the value httpfs is
    # configured with and not merely from the value a manifest publishes. It is None only when
    # S3_ENDPOINT is unset; an unparseable one — and one out of which no HOST survives — raised
    # above rather than arriving here as an empty string to be SET.
    if endpoint_setting:
        endpoint, use_ssl = endpoint_setting
        conn.execute("SET s3_endpoint = ?", [endpoint])
        conn.execute(f"SET s3_use_ssl = {use_ssl}")
        if settings.S3_FORCE_PATH_STYLE:
            conn.execute("SET s3_url_style = 'path'")
    if use_httpfs and settings.S3_ACCESS_KEY:
        conn.execute("SET s3_access_key_id = ?", [settings.S3_ACCESS_KEY])
    if use_httpfs and settings.S3_SECRET_KEY:
        conn.execute("SET s3_secret_access_key = ?", [settings.S3_SECRET_KEY])

    return conn


@contextmanager
def duckdb_session(
    db_path: str | None = None, *, load_httpfs: bool = False
) -> Iterator[duckdb.DuckDBPyConnection]:
    """Yield a configured DuckDB connection and always close it.

    Replaces the repeated ``conn = get_duckdb_connection(...); try: … finally:
    conn.close()`` boilerplate. ``db_path`` follows :func:`get_duckdb_connection`
    (``None`` → persistent ``settings.DUCKDB_PATH``; ``":memory:"`` → ephemeral).
    The connection still opens per call, so callers wrapping this in
    ``asyncio.to_thread`` keep their thread-safety.
    """
    conn = get_duckdb_connection(db_path, load_httpfs=load_httpfs)
    try:
        yield conn
    finally:
        conn.close()
