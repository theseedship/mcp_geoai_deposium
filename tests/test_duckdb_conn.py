"""Tests for DuckDB connection setup.

Everything here is offline: the fake-connection tests never open a socket, and the one
end-to-end test dials a CLOSED loopback port on purpose, because the only way to see what
httpfs puts into an error message is to make httpfs raise one.
"""

from __future__ import annotations

import socket
from typing import Any

import duckdb
import pytest

from src.utils import duckdb_conn


class _FakeDuckDBConnection:
    def __init__(self) -> None:
        self.calls: list[tuple[str, list[Any] | None]] = []

    def execute(self, sql: str, parameters: list[Any] | None = None) -> _FakeDuckDBConnection:
        self.calls.append((sql, parameters))
        return self


def test_duckdb_connection_does_not_load_httpfs_for_bucket_only(monkeypatch):
    fake = _FakeDuckDBConnection()
    monkeypatch.setattr(duckdb_conn.duckdb, "connect", lambda _path: fake)
    monkeypatch.setattr(duckdb_conn, "_spatial_installed", False)
    monkeypatch.setattr(duckdb_conn, "_httpfs_installed", False)
    monkeypatch.setattr(duckdb_conn.settings, "S3_BUCKET", "deposium")
    monkeypatch.setattr(duckdb_conn.settings, "S3_ENDPOINT", "")
    monkeypatch.setattr(duckdb_conn.settings, "S3_ACCESS_KEY", "")
    monkeypatch.setattr(duckdb_conn.settings, "S3_SECRET_KEY", "")

    conn = duckdb_conn.get_duckdb_connection(":memory:")

    assert conn is fake
    assert [sql for sql, _params in fake.calls] == [
        "INSTALL spatial;",
        "LOAD spatial;",
    ]


def test_duckdb_connection_can_force_httpfs_for_explicit_s3_paths(monkeypatch):
    fake = _FakeDuckDBConnection()
    monkeypatch.setattr(duckdb_conn.duckdb, "connect", lambda _path: fake)
    monkeypatch.setattr(duckdb_conn, "_spatial_installed", False)
    monkeypatch.setattr(duckdb_conn, "_httpfs_installed", False)
    monkeypatch.setattr(duckdb_conn.settings, "S3_BUCKET", "")
    monkeypatch.setattr(duckdb_conn.settings, "S3_ENDPOINT", "")
    monkeypatch.setattr(duckdb_conn.settings, "S3_ACCESS_KEY", "")
    monkeypatch.setattr(duckdb_conn.settings, "S3_SECRET_KEY", "")

    duckdb_conn.get_duckdb_connection(":memory:", load_httpfs=True)

    assert [sql for sql, _params in fake.calls] == [
        "INSTALL spatial;",
        "LOAD spatial;",
        "INSTALL httpfs;",
        "LOAD httpfs;",
    ]


def test_isolated_worker_loads_preinstalled_extensions_without_installing(monkeypatch):
    fake = _FakeDuckDBConnection()
    monkeypatch.setattr(duckdb_conn.duckdb, "connect", lambda _path: fake)
    monkeypatch.setattr(duckdb_conn, "_spatial_installed", False)
    monkeypatch.setattr(duckdb_conn, "_httpfs_installed", False)
    monkeypatch.setattr(duckdb_conn.settings, "S3_ENDPOINT", "")
    monkeypatch.setattr(duckdb_conn.settings, "S3_ACCESS_KEY", "")
    monkeypatch.setattr(duckdb_conn.settings, "S3_SECRET_KEY", "")

    duckdb_conn.get_duckdb_connection(
        ":memory:",
        load_httpfs=True,
        install_extensions=False,
    )

    assert [sql for sql, _params in fake.calls] == ["LOAD spatial;", "LOAD httpfs;"]


def test_duckdb_connection_does_not_load_httpfs_for_credentials_only(monkeypatch):
    fake = _FakeDuckDBConnection()
    monkeypatch.setattr(duckdb_conn.duckdb, "connect", lambda _path: fake)
    monkeypatch.setattr(duckdb_conn, "_spatial_installed", False)
    monkeypatch.setattr(duckdb_conn, "_httpfs_installed", False)
    monkeypatch.setattr(duckdb_conn.settings, "S3_BUCKET", "")
    monkeypatch.setattr(duckdb_conn.settings, "S3_ENDPOINT", "")
    monkeypatch.setattr(duckdb_conn.settings, "S3_ACCESS_KEY", "access")
    monkeypatch.setattr(duckdb_conn.settings, "S3_SECRET_KEY", "secret")
    monkeypatch.setattr(duckdb_conn.settings, "S3_FORCE_PATH_STYLE", False)

    duckdb_conn.get_duckdb_connection(":memory:")

    assert [sql for sql, _params in fake.calls] == [
        "INSTALL spatial;",
        "LOAD spatial;",
    ]


def test_duckdb_connection_loads_httpfs_before_s3_settings(monkeypatch):
    fake = _FakeDuckDBConnection()
    monkeypatch.setattr(duckdb_conn.duckdb, "connect", lambda _path: fake)
    monkeypatch.setattr(duckdb_conn, "_spatial_installed", False)
    monkeypatch.setattr(duckdb_conn, "_httpfs_installed", False)
    monkeypatch.setattr(duckdb_conn.settings, "S3_BUCKET", "deposium")
    monkeypatch.setattr(duckdb_conn.settings, "S3_ENDPOINT", "http://rustfs:9000")
    monkeypatch.setattr(duckdb_conn.settings, "S3_ACCESS_KEY", "access")
    monkeypatch.setattr(duckdb_conn.settings, "S3_SECRET_KEY", "secret")
    monkeypatch.setattr(duckdb_conn.settings, "S3_FORCE_PATH_STYLE", True)

    duckdb_conn.get_duckdb_connection(":memory:", load_httpfs=True)

    sql_calls = [sql for sql, _params in fake.calls]
    assert sql_calls.index("LOAD httpfs;") < sql_calls.index("SET s3_endpoint = ?")
    assert ("SET s3_endpoint = ?", ["rustfs:9000"]) in fake.calls
    assert ("SET s3_access_key_id = ?", ["access"]) in fake.calls
    assert ("SET s3_secret_access_key = ?", ["secret"]) in fake.calls


def test_duckdb_connection_sets_s3_credentials_without_custom_endpoint(monkeypatch):
    fake = _FakeDuckDBConnection()
    monkeypatch.setattr(duckdb_conn.duckdb, "connect", lambda _path: fake)
    monkeypatch.setattr(duckdb_conn, "_spatial_installed", False)
    monkeypatch.setattr(duckdb_conn, "_httpfs_installed", False)
    monkeypatch.setattr(duckdb_conn.settings, "S3_BUCKET", "deposium")
    monkeypatch.setattr(duckdb_conn.settings, "S3_ENDPOINT", "")
    monkeypatch.setattr(duckdb_conn.settings, "S3_ACCESS_KEY", "access")
    monkeypatch.setattr(duckdb_conn.settings, "S3_SECRET_KEY", "secret")
    monkeypatch.setattr(duckdb_conn.settings, "S3_FORCE_PATH_STYLE", False)

    duckdb_conn.get_duckdb_connection(":memory:", load_httpfs=True)

    sql_calls = [sql for sql, _params in fake.calls]
    assert "SET s3_endpoint = ?" not in sql_calls
    assert sql_calls.index("LOAD httpfs;") < sql_calls.index("SET s3_access_key_id = ?")
    assert ("SET s3_access_key_id = ?", ["access"]) in fake.calls
    assert ("SET s3_secret_access_key = ?", ["secret"]) in fake.calls


# --------------------------------------------------------------------------- #
# The endpoint handed to httpfs is an address, never a credential
# --------------------------------------------------------------------------- #


def test_an_endpoint_with_userinfo_is_refused_before_httpfs_sees_anything(monkeypatch):
    """`SET s3_endpoint` used to receive `S3_ENDPOINT` with only the scheme stripped.

    httpfs quotes the endpoint IN FULL in every connection error it raises — measured on
    duckdb 1.5.4 against the dev RustFS:

        IO Error: Could not connect to server error for HTTP HEAD to
        'http://deposium-dev:deposium-dev-secret@127.0.0.1:9999/deposium/…'

    so the userinfo came back out of the engine and into stderr, tickets and CI logs on any
    exit that does not pass through a redacting boundary. The round before this one answered by
    STRIPPING the userinfo and configuring the remainder. It is refused outright now: DuckDB
    never needed the userinfo (the credentials are supplied separately by `s3_access_key_id` /
    `s3_secret_access_key`), so a value carrying one is a misconfiguration, and configuring the
    part of a misconfigured value that looked address-shaped is the behaviour under review.
    """
    fake = _FakeDuckDBConnection()

    def unexpected_connect(*_args, **_kwargs):
        raise AssertionError("userinfo in the endpoint must abort before a connection is opened")

    monkeypatch.setattr(duckdb_conn.duckdb, "connect", unexpected_connect)
    monkeypatch.setattr(duckdb_conn, "_spatial_installed", False)
    monkeypatch.setattr(duckdb_conn, "_httpfs_installed", False)
    monkeypatch.setattr(duckdb_conn.settings, "S3_BUCKET", "deposium")
    monkeypatch.setattr(
        duckdb_conn.settings, "S3_ENDPOINT", "http://rustfs-admin:sup3r-s3cret@rustfs:9000"
    )
    monkeypatch.setattr(duckdb_conn.settings, "S3_ACCESS_KEY", "rustfs-admin")
    monkeypatch.setattr(duckdb_conn.settings, "S3_SECRET_KEY", "sup3r-s3cret")
    monkeypatch.setattr(duckdb_conn.settings, "S3_FORCE_PATH_STYLE", True)

    with pytest.raises(duckdb_conn.S3EndpointConfigurationError) as raised:
        duckdb_conn.get_duckdb_connection(":memory:", load_httpfs=True)

    assert "sup3r-s3cret" not in str(raised.value)
    assert "rustfs-admin" not in str(raised.value)
    assert fake.calls == []


@pytest.mark.parametrize(
    ("endpoint", "expected", "use_ssl"),
    [
        ("http://rustfs:9000", "rustfs:9000", False),
        ("https://mesh.internal", "mesh.internal", True),
        ("rustfs:9000", "rustfs:9000", True),
        ("http://[::1]:9000", "[::1]:9000", False),
        # `//` is address punctuation, not part of a host. httpfs used to be handed
        # `//rustfs:9000` verbatim, marker and all, because the strip only knew the two schemes.
        ("//rustfs:9000", "rustfs:9000", True),
    ],
)
def test_the_address_httpfs_receives_is_the_authority_alone(
    endpoint, expected, use_ssl, monkeypatch
):
    fake = _FakeDuckDBConnection()
    monkeypatch.setattr(duckdb_conn.duckdb, "connect", lambda _path: fake)
    monkeypatch.setattr(duckdb_conn, "_spatial_installed", False)
    monkeypatch.setattr(duckdb_conn, "_httpfs_installed", False)
    monkeypatch.setattr(duckdb_conn.settings, "S3_BUCKET", "deposium")
    monkeypatch.setattr(duckdb_conn.settings, "S3_ENDPOINT", endpoint)
    monkeypatch.setattr(duckdb_conn.settings, "S3_ACCESS_KEY", "access")
    monkeypatch.setattr(duckdb_conn.settings, "S3_SECRET_KEY", "secret")
    monkeypatch.setattr(duckdb_conn.settings, "S3_FORCE_PATH_STYLE", True)

    duckdb_conn.get_duckdb_connection(":memory:", load_httpfs=True)

    assert ("SET s3_endpoint = ?", [expected]) in fake.calls
    assert (f"SET s3_use_ssl = {use_ssl}", None) in fake.calls


# --------------------------------------------------------------------------- #
# A configured-but-unparseable endpoint fails CLOSED
# --------------------------------------------------------------------------- #

# Every shape `sanitize_endpoint` cannot turn into an address. Each one used to skip the whole
# `SET s3_endpoint` / `s3_use_ssl` / `s3_url_style` block, which leaves httpfs on its built-in
# AWS endpoint — so the next read shipped S3_ACCESS_KEY/S3_SECRET_KEY to s3.amazonaws.com in a
# SigV4 Authorization header.
_UNPARSEABLE_ENDPOINTS = [
    # Nothing but userinfo: no address left at all.
    "rustfs-admin:sup3r-s3cret@",
    # An unencoded "/", "?" or "#" inside the password ends the authority early, so
    # `account:password-prefix` masquerades as `host:port` and neither reading may be used.
    "http://rustfs-admin:sup3r/s3cret@mesh.internal:9000",
    "http://rustfs-admin:sup3r?s3cret@mesh.internal:9000",
    "http://rustfs-admin:sup3r#s3cret@mesh.internal:9000",
    # The percent-encoded delimiter, same shape.
    "http://rustfs-admin:sup3r/s3cret%40mesh.internal:9000",
]


@pytest.mark.parametrize("endpoint", _UNPARSEABLE_ENDPOINTS)
def test_a_configured_endpoint_that_does_not_parse_refuses_to_connect(endpoint, monkeypatch):
    """The fail-OPEN regression: no address meant no `SET`, and no `SET` means AWS.

    `sanitize_endpoint` returning `None` for a NON-EMPTY `S3_ENDPOINT` is a misconfiguration,
    not "no endpoint configured". Treating the two the same left DuckDB httpfs pointed at its
    default `s3.amazonaws.com` while `SET s3_access_key_id` / `SET s3_secret_access_key` ran
    three lines later — the store's credentials, signed and sent to Amazon over the internet.
    """
    fake = _FakeDuckDBConnection()

    def unexpected_connect(*_args, **_kwargs):
        raise AssertionError("an unparseable endpoint must abort before a connection is opened")

    monkeypatch.setattr(duckdb_conn.duckdb, "connect", unexpected_connect)
    monkeypatch.setattr(duckdb_conn, "_spatial_installed", False)
    monkeypatch.setattr(duckdb_conn, "_httpfs_installed", False)
    monkeypatch.setattr(duckdb_conn.settings, "S3_ENDPOINT", endpoint)
    monkeypatch.setattr(duckdb_conn.settings, "S3_ACCESS_KEY", "rustfs-admin")
    monkeypatch.setattr(duckdb_conn.settings, "S3_SECRET_KEY", "sup3r-s3cret")

    with pytest.raises(duckdb_conn.S3EndpointConfigurationError) as raised:
        duckdb_conn.get_duckdb_connection(":memory:", load_httpfs=True)

    message = str(raised.value)
    # Useful to an operator: it names the setting and says what to do.
    assert "S3_ENDPOINT" in message
    # And it does not quote the value, because the value is why it could not be parsed.
    assert "sup3r-s3cret" not in message
    assert "sup3r" not in message
    assert "rustfs-admin" not in message
    # Nothing was sent anywhere: no connection, so no SQL, so no signed request.
    assert fake.calls == []


@pytest.mark.parametrize("endpoint", ["", None])
def test_an_unset_endpoint_still_uses_the_aws_default_with_credentials(endpoint, monkeypatch):
    """The legitimate AWS deployment must keep working — unset is not a misconfiguration.

    Only these two values are unset. This test used to pass `"   "` here, which is how the
    fail-open survived two rounds of fixing: the suite itself asserted that a whitespace-only
    setting SHOULD take the AWS-default path, so every guard added below it was measured
    against a table that had already blessed the hole.
    """
    fake = _FakeDuckDBConnection()
    monkeypatch.setattr(duckdb_conn.duckdb, "connect", lambda _path: fake)
    monkeypatch.setattr(duckdb_conn, "_spatial_installed", False)
    monkeypatch.setattr(duckdb_conn, "_httpfs_installed", False)
    monkeypatch.setattr(duckdb_conn.settings, "S3_ENDPOINT", endpoint)
    monkeypatch.setattr(duckdb_conn.settings, "S3_ACCESS_KEY", "AKIAEXAMPLE")
    monkeypatch.setattr(duckdb_conn.settings, "S3_SECRET_KEY", "secret")

    duckdb_conn.get_duckdb_connection(":memory:", load_httpfs=True)

    sql_calls = [sql for sql, _params in fake.calls]
    assert "SET s3_endpoint = ?" not in sql_calls
    assert ("SET s3_access_key_id = ?", ["AKIAEXAMPLE"]) in fake.calls


# A value that empties out is a value someone TYPED. Merging it into "unset" is the whole
# defect, entered above both guards: `raw.strip() or ""` cannot tell a misspelled setting from
# an absent one, and the interpretation it picks is the one that ships credentials to Amazon.
# str.strip() removes every Unicode space, so NBSP and the ideographic space arrive here too.
_EMPTIES_OUT = ["   ", " ", "\t", "\n", "\r\n", "\x0b", "\xa0", "　", " "]


@pytest.mark.parametrize("endpoint", _EMPTIES_OUT)
def test_a_whitespace_only_endpoint_is_a_misconfiguration_not_an_unset_one(endpoint, monkeypatch):
    """Measured reaching AWS before this guard: 'https://deposium.s3.us-east-1.amazonaws.com/…'."""
    fake = _FakeDuckDBConnection()

    def unexpected_connect(*_args, **_kwargs):
        raise AssertionError("a value that empties out must abort before a connection is opened")

    monkeypatch.setattr(duckdb_conn.duckdb, "connect", unexpected_connect)
    monkeypatch.setattr(duckdb_conn, "_spatial_installed", False)
    monkeypatch.setattr(duckdb_conn, "_httpfs_installed", False)
    monkeypatch.setattr(duckdb_conn.settings, "S3_ENDPOINT", endpoint)
    monkeypatch.setattr(duckdb_conn.settings, "S3_ACCESS_KEY", "rustfs-admin")
    monkeypatch.setattr(duckdb_conn.settings, "S3_SECRET_KEY", "sup3r-s3cret")

    with pytest.raises(duckdb_conn.S3EndpointConfigurationError):
        duckdb_conn.get_duckdb_connection(":memory:", load_httpfs=True)

    assert fake.calls == []


@pytest.mark.parametrize("endpoint", [9000, b"http://rustfs:9000", ["http://rustfs:9000"]])
def test_a_non_string_endpoint_is_refused_rather_than_read_as_unset(endpoint, monkeypatch):
    """Unreachable through pydantic today (the field is typed `str = ""`), guarded anyway.

    The old `isinstance(raw, str) else ""` swallowed a type error into "no endpoint configured",
    and of the two ways to be wrong about a type, that is the one that costs credentials.
    """
    fake = _FakeDuckDBConnection()

    def unexpected_connect(*_args, **_kwargs):
        raise AssertionError("a non-string endpoint must abort before a connection is opened")

    monkeypatch.setattr(duckdb_conn.duckdb, "connect", unexpected_connect)
    monkeypatch.setattr(duckdb_conn, "_spatial_installed", False)
    monkeypatch.setattr(duckdb_conn, "_httpfs_installed", False)
    monkeypatch.setattr(duckdb_conn.settings, "S3_ENDPOINT", endpoint)
    monkeypatch.setattr(duckdb_conn.settings, "S3_ACCESS_KEY", "rustfs-admin")
    monkeypatch.setattr(duckdb_conn.settings, "S3_SECRET_KEY", "sup3r-s3cret")

    with pytest.raises(duckdb_conn.S3EndpointConfigurationError):
        duckdb_conn.get_duckdb_connection(":memory:", load_httpfs=True)

    assert fake.calls == []


@pytest.mark.parametrize(
    ("endpoint", "expected", "use_ssl"),
    [
        ("HTTP://rustfs:9000", "rustfs:9000", False),
        ("Http://rustfs:9000", "rustfs:9000", False),
        ("HTTPS://mesh.internal:9000", "mesh.internal:9000", True),
    ],
)
def test_the_scheme_is_stripped_case_insensitively(endpoint, expected, use_ssl, monkeypatch):
    """RFC 3986 schemes are case-insensitive; a case-sensitive strip fed httpfs its own scheme.

    Observed before the fix: `Could not resolve hostname error for HTTP HEAD to
    'http://HTTP://rustfs%3A9000/deposium/…'` — accepted by the guards, unusable in practice.
    """
    fake = _FakeDuckDBConnection()
    monkeypatch.setattr(duckdb_conn.duckdb, "connect", lambda _path: fake)
    monkeypatch.setattr(duckdb_conn, "_spatial_installed", False)
    monkeypatch.setattr(duckdb_conn, "_httpfs_installed", False)
    monkeypatch.setattr(duckdb_conn.settings, "S3_ENDPOINT", endpoint)
    monkeypatch.setattr(duckdb_conn.settings, "S3_ACCESS_KEY", "access")
    monkeypatch.setattr(duckdb_conn.settings, "S3_SECRET_KEY", "secret")

    duckdb_conn.get_duckdb_connection(":memory:", load_httpfs=True)

    assert ("SET s3_endpoint = ?", [expected]) in fake.calls
    # The operator wrote the scheme in capitals; the TLS decision must follow what they wrote.
    ssl_calls = [sql for sql, _params in fake.calls if "s3_use_ssl" in sql]
    assert ssl_calls == [f"SET s3_use_ssl = {use_ssl}"]


@pytest.mark.parametrize(
    "setting",
    [
        "?://h",
        "#h",
        "?h:9000",
        "/?h",
        "[]:9000",
        "[ ]:9000",
        # `//` is punctuation the caller strips; it is not part of an authority.
        "//rustfs:9000",
        # The half guard 2 never had. Its predecessor validated the host and NEVER the port, so
        # every one of these passed it — and `admin:secret` passing it means the PASSWORD was
        # configured as the port and quoted back by httpfs.
        "admin:secret",
        "host:99999",
        "host:abc",
        "host:0",
        "host:09000",
        "host:\u0669\u0660\u0660\u0660",
        # And the shapes that are not an authority at all.
        "mesh.internal:9000/rustfs",
        "[::1:9000",
        "[::1]secret",
        "",
    ],
)
def test_guard_two_wants_a_whole_authority_not_merely_a_host_shaped_prefix(setting):
    """Cutting the authority on '/' alone read a "host" of '?' out of '?://h'.

    The guard's postcondition is that the string is exactly `host[:port]`, and the port half of
    it never existed: the second, hand-written copy of the host test drifted from the parser it
    was meant to be independent of. It is the same accept grammar now, applied to the final
    string, which is the only part of "independent" that was ever load-bearing.
    """
    assert not duckdb_conn.names_an_authority(setting)


@pytest.mark.parametrize(
    "setting", ["rustfs:9000", "mesh.internal", "[::1]:9000", "127.0.0.1", "host:65535"]
)
def test_guard_two_still_accepts_a_real_address(setting):
    assert duckdb_conn.names_an_authority(setting)


def test_an_unparseable_endpoint_is_irrelevant_when_httpfs_is_not_loaded(monkeypatch):
    """Local-only work never touches S3, so a broken S3_ENDPOINT must not break it."""
    fake = _FakeDuckDBConnection()
    monkeypatch.setattr(duckdb_conn.duckdb, "connect", lambda _path: fake)
    monkeypatch.setattr(duckdb_conn, "_spatial_installed", False)
    monkeypatch.setattr(duckdb_conn, "_httpfs_installed", False)
    monkeypatch.setattr(duckdb_conn.settings, "S3_ENDPOINT", "rustfs-admin:sup3r-s3cret@")
    monkeypatch.setattr(duckdb_conn.settings, "S3_ACCESS_KEY", "")
    monkeypatch.setattr(duckdb_conn.settings, "S3_SECRET_KEY", "")

    duckdb_conn.get_duckdb_connection(":memory:")

    assert [sql for sql, _params in fake.calls] == ["INSTALL spatial;", "LOAD spatial;"]


def _closed_loopback_port() -> int:
    """A loopback port nothing is listening on, so connecting to it fails immediately."""
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _httpfs_loadable() -> bool:
    conn = duckdb.connect(":memory:")
    try:
        conn.execute("INSTALL httpfs;")
        conn.execute("LOAD httpfs;")
        return True
    except duckdb.Error:  # extension not present and not downloadable here
        return False
    finally:
        conn.close()


@pytest.mark.skipif(not _httpfs_loadable(), reason="duckdb httpfs extension unavailable")
def test_an_httpfs_connection_error_raised_through_duckdb_session_has_no_password(monkeypatch):
    """The exit the manifest/CLI redactors do not cover: an exception nobody caught.

    This is the whole point of sanitizing the endpoint at configuration time rather than at
    publication time. The message below goes to stderr and into the CI log verbatim.
    """
    endpoint_host = f"127.0.0.1:{_closed_loopback_port()}"
    # The endpoint is well-formed and carries no userinfo — the deployment shape that actually
    # runs. The credentials go in through `s3_access_key_id`/`s3_secret_access_key`, and the
    # engine's own error must not name them even so.
    monkeypatch.setattr(duckdb_conn.settings, "S3_ENDPOINT", f"http://{endpoint_host}")
    monkeypatch.setattr(duckdb_conn.settings, "S3_ACCESS_KEY", "rustfs-admin")
    monkeypatch.setattr(duckdb_conn.settings, "S3_SECRET_KEY", "sup3r-s3cret")
    monkeypatch.setattr(duckdb_conn.settings, "S3_FORCE_PATH_STYLE", True)

    with pytest.raises(duckdb.Error) as raised:  # noqa: PT012 — the session is part of the subject
        with duckdb_conn.duckdb_session(":memory:", load_httpfs=True) as conn:
            conn.execute("SELECT * FROM read_parquet('s3://deposium/no-such-object.parquet')")

    message = str(raised.value)
    # The engine did try to reach the configured address — otherwise this proves nothing.
    assert endpoint_host in message
    assert "sup3r-s3cret" not in message
    assert "rustfs-admin" not in message


# --------------------------------------------------------------------------- #
# TWO INDEPENDENT GUARDS against `SET s3_endpoint = ''`
# --------------------------------------------------------------------------- #

# The third appearance of the same fail-open, and the first two fixes each guarded ONE place.
# Nineteen non-empty `S3_ENDPOINT` values sanitized to the TRUTHY scheme-only address
# `'http://'`, which passed `if endpoint_address:` and then had its scheme stripped into `''`.
# `SET s3_endpoint = ''` is not "no endpoint": DuckDB httpfs falls back to
# `s3.us-east-1.amazonaws.com` and signs the configured credentials into the next request.
# Observed against the real engine, with this repo's dev credentials:
#
#     HTTP GET error reading 'http://s3.us-east-1.amazonaws.com/deposium/…' in region
#     'us-east-1' (HTTP 403 Forbidden) … InvalidAccessKeyId … Invalid Access Key: "deposium-dev"
#
# So the table below is driven through the REAL `get_duckdb_connection` and every row must end
# in exactly one of two states: refused, or configured with an address that names a host.
_CONFIGURED_ENDPOINTS = [
    # --- the 19 reported scheme-only shapes ------------------------------------------------ #
    "http://",
    "https://",
    "//",
    "http:///",
    "http:////",
    "http://@",
    "http://user@",
    "http://user:pass@",
    "http://%40",
    "http://user:pass%40",
    "http://?@",
    "http://?",
    "http://#",
    "http:// ",
    "http://\t",
    "http:///@",
    "http://:@",
    "http://@@",
    "//@",
    # --- shapes added here, because this is the third round and the list keeps being short -- #
    "http://:",  # a colon with nothing on either side of it
    "http://:/",  # …and the same with a trailing slash, which `rstrip("/")` eats
    "http://:9000",  # a bare port: truthy, plausible, and nameless
    "//:9000",
    "http://:9000/",
    "://",  # empty scheme
    "http://   ",  # whitespace where the host goes
    "http:// /",
    "http://#@",
    "http://?#@",
    "http://%40%40",
    "http://a:b@",  # nothing but userinfo, with a scheme
    "a:b@",  # nothing but userinfo, without one
    "/",
    "///",
    "/@",
    "http://\n",
    # --- and the shapes that must still WORK, in the same table, checked the same way ------ #
    "http://rustfs:9000",
    "https://s3.example.com",
    "mesh.internal:9000",
    "//mesh.internal:9000",
    "http://[::1]:9000",
    "http://127.0.0.1:9000/rustfs",
    "http://rustfs-admin:sup3r-s3cret@rustfs:9000",
    # …and the shapes the previous two rounds closed, which must stay closed.
    "rustfs-admin:sup3r-s3cret@",
    "http://rustfs-admin:sup3r/s3cret@mesh.internal:9000",
    "http://rustfs-admin:sup3r?s3cret@mesh.internal:9000",
    "http://rustfs-admin:sup3r#s3cret@mesh.internal:9000",
    "http://rustfs-admin:sup3r/s3cret%40mesh.internal:9000",
]


def _host_in(setting: str) -> str:
    """The host of a `host[:port]`, re-derived HERE so the test does not grade its own subject.

    Deliberately not `duckdb_conn._names_a_host`: an assertion that calls the guard it is
    checking passes for any pair of matching mistakes.
    """
    authority = setting.lstrip("/").split("/")[0]
    if authority.startswith("["):
        return authority[1:].split("]")[0].strip()
    return authority.split(":")[0].strip()


@pytest.mark.parametrize("endpoint", _CONFIGURED_ENDPOINTS)
def test_a_configured_endpoint_is_either_refused_or_names_a_host(endpoint, monkeypatch):
    """Exactly one of two outcomes per row, and never credentials without an address."""
    fake = _FakeDuckDBConnection()
    monkeypatch.setattr(duckdb_conn.duckdb, "connect", lambda _path: fake)
    monkeypatch.setattr(duckdb_conn, "_spatial_installed", False)
    monkeypatch.setattr(duckdb_conn, "_httpfs_installed", False)
    monkeypatch.setattr(duckdb_conn.settings, "S3_ENDPOINT", endpoint)
    monkeypatch.setattr(duckdb_conn.settings, "S3_ACCESS_KEY", "deposium-dev")
    monkeypatch.setattr(duckdb_conn.settings, "S3_SECRET_KEY", "deposium-dev-secret")
    monkeypatch.setattr(duckdb_conn.settings, "S3_FORCE_PATH_STYLE", True)

    try:
        duckdb_conn.get_duckdb_connection(":memory:", load_httpfs=True)
    except duckdb_conn.S3EndpointConfigurationError:
        # Outcome A: refused. Nothing ran at all, so no credential was signed anywhere.
        assert fake.calls == []
        assert "deposium-dev" not in repr(fake.calls)
        return

    # Outcome B: configured — and configured with an ADDRESS.
    sql_calls = [sql for sql, _params in fake.calls]
    endpoint_values = [params for sql, params in fake.calls if sql == "SET s3_endpoint = ?"]
    assert len(endpoint_values) == 1, f"{endpoint!r} configured no endpoint but did not refuse"
    [configured] = endpoint_values[0]
    assert configured, f"{endpoint!r} ran SET s3_endpoint = '' — that is the AWS default"
    assert _host_in(configured), f"{endpoint!r} configured {configured!r}, which names no host"
    # The credentials go out AFTER the address, never before it and never without it.
    assert sql_calls.index("SET s3_endpoint = ?") < sql_calls.index("SET s3_access_key_id = ?")


@pytest.mark.parametrize(
    "endpoint,expected,expected_ssl",
    [
        ("http://rustfs:9000/", "rustfs:9000", False),
        ("https://s3.example.com/", "s3.example.com", True),
        ("//mesh.internal:9000/", "mesh.internal:9000", True),
        ("mesh.internal:9000/", "mesh.internal:9000", True),
    ],
)
def test_a_trailing_slash_still_connects_the_way_it_did_before_the_accept_grammar(
    endpoint, expected, expected_ssl, monkeypatch
):
    """The regression this branch could have shipped, pinned at the connection boundary.

    `main`'s `_httpfs_endpoint_setting` ran `endpoint.rstrip("/")`, so
    `S3_ENDPOINT=http://rustfs:9000/` is a deployment that is running right now. An accept grammar
    that refuses it whole turns that deployment into an `S3EndpointConfigurationError` at startup
    — for a spelling that carries no credential, since a credential needs the `:` or `@` the
    authority already validates. So the slash is accepted and normalised away, and httpfs receives
    exactly the `host:port` it received before.
    """
    fake = _FakeDuckDBConnection()
    monkeypatch.setattr(duckdb_conn.duckdb, "connect", lambda _path: fake)
    monkeypatch.setattr(duckdb_conn, "_spatial_installed", False)
    monkeypatch.setattr(duckdb_conn, "_httpfs_installed", False)
    monkeypatch.setattr(duckdb_conn.settings, "S3_ENDPOINT", endpoint)
    monkeypatch.setattr(duckdb_conn.settings, "S3_ACCESS_KEY", "deposium-dev")
    monkeypatch.setattr(duckdb_conn.settings, "S3_SECRET_KEY", "deposium-dev-secret")
    monkeypatch.setattr(duckdb_conn.settings, "S3_FORCE_PATH_STYLE", True)

    duckdb_conn.get_duckdb_connection(":memory:", load_httpfs=True)

    assert ("SET s3_endpoint = ?", [expected]) in fake.calls
    assert (f"SET s3_use_ssl = {expected_ssl}", None) in fake.calls


@pytest.mark.parametrize("endpoint", ["", None])
def test_no_endpoint_configured_still_means_the_aws_default_with_credentials(endpoint, monkeypatch):
    """The ONE legitimate AWS case, and the whole reason the two states must stay distinct.

    An empty (or whitespace-only, or non-str) `S3_ENDPOINT` says "no endpoint configured": httpfs
    keeps its built-in AWS endpoint, the credentials are sent to it on purpose, and nothing
    raises. That is a real deployment, not the misconfiguration above.
    """
    fake = _FakeDuckDBConnection()
    monkeypatch.setattr(duckdb_conn.duckdb, "connect", lambda _path: fake)
    monkeypatch.setattr(duckdb_conn, "_spatial_installed", False)
    monkeypatch.setattr(duckdb_conn, "_httpfs_installed", False)
    monkeypatch.setattr(duckdb_conn.settings, "S3_ENDPOINT", endpoint)
    monkeypatch.setattr(duckdb_conn.settings, "S3_ACCESS_KEY", "AKIAEXAMPLE")
    monkeypatch.setattr(duckdb_conn.settings, "S3_SECRET_KEY", "secret")

    duckdb_conn.get_duckdb_connection(":memory:", load_httpfs=True)

    sql_calls = [sql for sql, _params in fake.calls]
    assert "SET s3_endpoint = ?" not in sql_calls
    assert ("SET s3_access_key_id = ?", ["AKIAEXAMPLE"]) in fake.calls


def test_guard_two_refuses_a_scheme_only_endpoint_with_guard_one_disabled(monkeypatch):
    """Guard 2 alone. The sanitizer is replaced by the REGRESSED one that returned `'http://'`.

    This is the version of `sanitize_endpoint` that shipped: scheme-only in, truthy scheme-only
    out. If closing the hole depended on the parser being right, this test connects.
    """
    fake = _FakeDuckDBConnection()

    def unexpected_connect(*_args, **_kwargs):
        raise AssertionError("guard 2 must refuse before a connection is opened")

    monkeypatch.setattr(duckdb_conn.duckdb, "connect", unexpected_connect)
    # Guard 1 disabled: whatever comes in is handed back as a truthy "address".
    monkeypatch.setattr(duckdb_conn, "sanitize_endpoint", lambda value: (value, False))
    monkeypatch.setattr(duckdb_conn.settings, "S3_ENDPOINT", "http://")
    monkeypatch.setattr(duckdb_conn.settings, "S3_ACCESS_KEY", "deposium-dev")
    monkeypatch.setattr(duckdb_conn.settings, "S3_SECRET_KEY", "deposium-dev-secret")

    with pytest.raises(duckdb_conn.S3EndpointConfigurationError) as raised:
        duckdb_conn.get_duckdb_connection(":memory:", load_httpfs=True)

    assert "S3_ENDPOINT" in str(raised.value)
    assert fake.calls == []


def test_guard_one_refuses_a_scheme_only_endpoint_with_guard_two_disabled(monkeypatch):
    """Guard 1 alone. The host check at the `SET` site is stubbed out to accept anything."""
    fake = _FakeDuckDBConnection()

    def unexpected_connect(*_args, **_kwargs):
        raise AssertionError("guard 1 must refuse before a connection is opened")

    monkeypatch.setattr(duckdb_conn.duckdb, "connect", unexpected_connect)
    # Guard 2 disabled: every string "names a host".
    monkeypatch.setattr(duckdb_conn, "names_an_authority", lambda _setting: True)
    monkeypatch.setattr(duckdb_conn.settings, "S3_ENDPOINT", "http://")
    monkeypatch.setattr(duckdb_conn.settings, "S3_ACCESS_KEY", "deposium-dev")
    monkeypatch.setattr(duckdb_conn.settings, "S3_SECRET_KEY", "deposium-dev-secret")

    with pytest.raises(duckdb_conn.S3EndpointConfigurationError) as raised:
        duckdb_conn.get_duckdb_connection(":memory:", load_httpfs=True)

    assert "S3_ENDPOINT" in str(raised.value)
    assert fake.calls == []


def test_disabling_both_guards_is_what_reproduces_the_hole(monkeypatch):
    """The control. Without it, the two tests above prove only that SOMETHING refuses.

    With both guards out, `S3_ENDPOINT='http://'` reaches `SET s3_endpoint = ''` and the
    credentials follow it — which is the AWS-default fallback, in this fake, exactly.
    """
    fake = _FakeDuckDBConnection()
    monkeypatch.setattr(duckdb_conn.duckdb, "connect", lambda _path: fake)
    monkeypatch.setattr(duckdb_conn, "_spatial_installed", False)
    monkeypatch.setattr(duckdb_conn, "_httpfs_installed", False)
    monkeypatch.setattr(duckdb_conn, "sanitize_endpoint", lambda value: (value, False))
    monkeypatch.setattr(duckdb_conn, "names_an_authority", lambda _setting: True)
    monkeypatch.setattr(duckdb_conn.settings, "S3_ENDPOINT", "http://")
    monkeypatch.setattr(duckdb_conn.settings, "S3_ACCESS_KEY", "deposium-dev")
    monkeypatch.setattr(duckdb_conn.settings, "S3_SECRET_KEY", "deposium-dev-secret")
    monkeypatch.setattr(duckdb_conn.settings, "S3_FORCE_PATH_STYLE", True)

    duckdb_conn.get_duckdb_connection(":memory:", load_httpfs=True)

    assert ("SET s3_endpoint = ?", [""]) in fake.calls
    assert ("SET s3_access_key_id = ?", ["deposium-dev"]) in fake.calls
