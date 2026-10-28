"""The serving reader binds a single bounded capture to its original Parquet bytes."""

from __future__ import annotations

import asyncio
import hashlib
import io
import os
from contextlib import contextmanager
from dataclasses import FrozenInstanceError
from pathlib import Path
from threading import Event
from types import SimpleNamespace

import duckdb
import pytest

from src.config import settings
from src.utils import normals_reader as reader

_LIMIT = 8 * 1024 * 1024
_SOURCE = "s3://normals-test/climate/normals/montpellier.parquet"


@pytest.fixture
def artifact(tmp_path):
    path = tmp_path / "normals.parquet"
    conn = duckdb.connect()
    try:
        conn.execute(
            "CREATE TABLE normals AS SELECT zone_id, month, variable, month * 1.0 AS p50 "
            "FROM (VALUES ('montpellier'), ('other')) zones(zone_id), "
            "range(1, 13) months(month), "
            "(VALUES ('precipitation'), ('wind_speed')) variables(variable)"
        )
        conn.execute("COPY normals TO ? (FORMAT PARQUET)", [str(path)])
    finally:
        conn.close()
    return path.read_bytes()


class _Body:
    def __init__(self, content, *, partial=None, fail=False):
        self.stream = io.BytesIO(content)
        self.partial = partial
        self.fail = fail
        self.read_sizes = []
        self.closed = False

    def read(self, size):
        assert isinstance(size, int) and 0 < size <= _LIMIT + 1
        self.read_sizes.append(size)
        if self.fail:
            raise OSError("private endpoint and credential failure")
        return self.stream.read(min(size, self.partial) if self.partial else size)

    def close(self):
        self.closed = True
        self.stream.close()


class _Client:
    def __init__(self, response=None, *, fail=False):
        self.response = response
        self.fail = fail
        self.gets = []
        self.closed = False

    def get_object(self, **kwargs):
        self.gets.append(kwargs)
        if self.fail:
            raise OSError("store unavailable")
        return self.response

    def close(self):
        self.closed = True

    def __getattr__(self, name):
        raise AssertionError(f"unexpected object-store operation: {name}")


def _store(monkeypatch, response, *, fail=False):
    client = _Client(response, fail=fail)
    monkeypatch.setattr(settings, "S3_BUCKET", "normals-test")
    monkeypatch.setattr(settings, "ALLOW_LOCAL_PARQUET_SOURCE", False)
    monkeypatch.setattr(reader, "_object_store_client", lambda: client)
    return client


def _read(src=_SOURCE, *, zone_id="montpellier", variables=None):
    return reader.read_normals_artifact(src, zone_id=zone_id, variables=variables)


def _assert_binding(result, content):
    assert result.artifact_version == "sha256:" + hashlib.sha256(content).hexdigest()
    assert result.artifact_size_bytes == len(content) > 0


@pytest.mark.parametrize("partial", [None, 7])
def test_one_get_no_head_or_writes_with_bounded_partial_reads(monkeypatch, artifact, partial):
    body = _Body(artifact, partial=partial)
    client = _store(monkeypatch, {"Body": body, "ContentLength": len(artifact)})
    result = _read(variables=["precipitation"])
    assert len(result.rows) == 12
    assert {row["zone_id"] for row in result.rows} == {"montpellier"}
    assert {row["variable"] for row in result.rows} == {"precipitation"}
    _assert_binding(result, artifact)
    assert client.gets == [{"Bucket": "normals-test", "Key": "climate/normals/montpellier.parquet"}]
    assert body.read_sizes
    if partial:
        assert len(body.read_sizes) > 2
    assert body.closed and client.closed
    with pytest.raises(FrozenInstanceError):
        result.artifact_version = "changed"


@pytest.mark.parametrize("variables", [None, ["precipitation"], ["wind_speed"]])
def test_filters_keep_the_whole_artifact_digest(monkeypatch, artifact, variables):
    body = _Body(artifact)
    _store(monkeypatch, {"Body": body, "ContentLength": len(artifact)})
    result = _read(variables=variables)
    assert len(result.rows) == (24 if variables is None else 12)
    _assert_binding(result, artifact)


@pytest.mark.parametrize("declared_delta", [-1, 1])
def test_content_length_must_match_actual_bytes(monkeypatch, artifact, declared_delta):
    body = _Body(artifact)
    client = _store(monkeypatch, {"Body": body, "ContentLength": len(artifact) + declared_delta})
    with pytest.raises((ValueError, OSError)):
        _read()
    assert body.closed and client.closed


def test_declared_oversize_is_refused_before_reading(monkeypatch):
    body = _Body(b"unused")
    client = _store(monkeypatch, {"Body": body, "ContentLength": _LIMIT + 1})
    with pytest.raises((ValueError, OSError)):
        _read()
    assert body.read_sizes == []
    assert body.closed and client.closed


def test_actual_oversize_is_bounded_even_with_an_untruthful_length(monkeypatch):
    body = _Body(b"x" * (_LIMIT + 2))
    client = _store(monkeypatch, {"Body": body, "ContentLength": 1})
    with pytest.raises((ValueError, OSError)):
        _read()
    assert body.stream.closed and client.closed
    assert body.read_sizes and sum(body.read_sizes) <= _LIMIT + 1


@pytest.mark.parametrize("declared", [None, True, -1, "100", 1.5])
def test_invalid_content_length_is_refused_without_reading(monkeypatch, declared):
    body = _Body(b"unused")
    client = _store(monkeypatch, {"Body": body, "ContentLength": declared})
    with pytest.raises(ValueError):
        _read()
    assert body.read_sizes == []
    assert body.closed and client.closed


@pytest.mark.parametrize("content", [b"", b"not parquet", b"PAR1brokenPAR1"])
def test_corrupt_or_empty_input_is_refused_and_closed(monkeypatch, content):
    body = _Body(content)
    client = _store(monkeypatch, {"Body": body, "ContentLength": len(content)})
    with pytest.raises((ValueError, OSError, duckdb.Error)):
        _read()
    assert body.closed and client.closed


def test_truncated_parquet_is_refused(monkeypatch, artifact):
    body = _Body(artifact[:-16])
    client = _store(monkeypatch, {"Body": body, "ContentLength": len(artifact)})
    with pytest.raises((ValueError, OSError, duckdb.Error)):
        _read()
    assert body.closed and client.closed


@pytest.mark.parametrize("response", [{}, {"Body": None}, {"Body": b"not a stream"}])
def test_missing_or_invalid_body_is_refused_and_client_closed(monkeypatch, response):
    client = _store(monkeypatch, response)
    with pytest.raises((ValueError, OSError, TypeError)):
        _read()
    assert client.closed


def test_nonbytes_body_chunk_is_refused_and_closed(monkeypatch):
    body = _Body(b"")
    body.read = lambda size: "not bytes"
    client = _store(monkeypatch, {"Body": body, "ContentLength": 1})
    with pytest.raises((ValueError, OSError, TypeError)):
        _read()
    assert body.closed and client.closed


@pytest.mark.parametrize("failure", ["get", "read"])
def test_store_failure_closes_resources(monkeypatch, failure):
    body = _Body(b"unused", fail=failure == "read")
    client = _store(monkeypatch, {"Body": body, "ContentLength": 6}, fail=failure == "get")
    with pytest.raises((ValueError, OSError)):
        _read()
    assert client.closed
    if failure == "read":
        assert body.closed


@pytest.mark.parametrize(
    "source",
    [
        "s3://other/climate/normals/montpellier.parquet",
        "s3://normals-test/climate/normals/other.parquet",
        "s3://normals-test/climate/forecast/montpellier.parquet",
        "s3://normals-test/climate/normals/../montpellier.parquet",
        "https://private.invalid/normals.parquet",
    ],
)
def test_disallowed_store_source_refused_before_client_creation(monkeypatch, source):
    monkeypatch.setattr(settings, "S3_BUCKET", "normals-test")
    monkeypatch.setattr(settings, "ALLOW_LOCAL_PARQUET_SOURCE", False)
    monkeypatch.setattr(reader, "_object_store_client", lambda: pytest.fail("client created"))
    with pytest.raises((ValueError, OSError)):
        _read(source)


def test_local_disabled_is_refused_before_descriptor_access(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "ALLOW_LOCAL_PARQUET_SOURCE", False)
    monkeypatch.setattr(reader, "ConfinedLocalIO", lambda *_: pytest.fail("local I/O opened"))
    with pytest.raises(ValueError):
        _read(str(tmp_path / "normals.parquet"))


@pytest.mark.parametrize("replacement", ["source", "ancestor", "root"])
def test_local_symlink_replacements_are_never_followed(
    monkeypatch, tmp_path, artifact, replacement
):
    root = tmp_path / "root"
    root.mkdir()
    directory = root / "nested"
    directory.mkdir()
    source = directory / "normals.parquet"
    source.write_bytes(artifact)
    external = tmp_path / "external"
    external.mkdir()
    (external / "normals.parquet").write_bytes(artifact)
    monkeypatch.setattr(settings, "ALLOW_LOCAL_PARQUET_SOURCE", True)
    monkeypatch.setattr(settings, "LOCAL_PARQUET_SOURCE_ROOT", str(root))
    if replacement == "source":
        source.unlink()
        source.symlink_to(external / "normals.parquet")
    elif replacement == "ancestor":
        directory.rename(root / "original")
        directory.symlink_to(external, target_is_directory=True)
    else:
        root.rename(tmp_path / "original-root")
        root.symlink_to(external, target_is_directory=True)
    with pytest.raises((ValueError, OSError)):
        _read(str(source))


def test_local_read_hashes_original_bytes(monkeypatch, tmp_path, artifact):
    path = tmp_path / "normals.parquet"
    path.write_bytes(artifact)
    monkeypatch.setattr(settings, "ALLOW_LOCAL_PARQUET_SOURCE", True)
    monkeypatch.setattr(settings, "LOCAL_PARQUET_SOURCE_ROOT", str(tmp_path))
    result = _read(str(path), variables=["precipitation"])
    _assert_binding(result, artifact)
    assert len(result.rows) == 12


@pytest.mark.parametrize("corrupt", [False, True])
def test_snapshot_descriptor_is_anonymous_and_closed_on_success_or_query_error(
    monkeypatch, artifact, corrupt
):
    captured_fds = []
    real_connect = duckdb.connect

    class _Connection:
        def __init__(self):
            self.conn = real_connect(":memory:")

        def execute(self, sql, parameters=None):
            assert "INSTALL" not in sql.upper() and "LOAD" not in sql.upper()
            assert parameters is not None
            snapshot = next(
                str(value) for value in parameters if str(value).startswith("/proc/self/fd/")
            )
            fd = int(snapshot.rsplit("/", 1)[1])
            captured_fds.append(fd)
            assert os.fstat(fd).st_nlink == 0
            assert Path(snapshot).read_bytes() == content
            return self.conn.execute(sql, parameters)

        def close(self):
            self.conn.close()

        def __getattr__(self, name):
            return getattr(self.conn, name)

        def __enter__(self):
            return self

        def __exit__(self, *_):
            self.close()

    content = b"not parquet" if corrupt else artifact
    body = _Body(content)
    client = _store(monkeypatch, {"Body": body, "ContentLength": len(content)})
    monkeypatch.setattr(reader.duckdb, "connect", lambda *_args, **_kwargs: _Connection())
    if corrupt:
        with pytest.raises((ValueError, OSError, duckdb.Error)):
            _read()
    else:
        _assert_binding(_read(), content)
    assert captured_fds
    for fd in captured_fds:
        with pytest.raises(OSError):
            os.fstat(fd)
    assert body.closed and client.closed


@pytest.mark.parametrize(
    ("endpoint", "expected"),
    [
        ("rustfs:9000", "https://rustfs:9000"),
        ("//rustfs:9000", "https://rustfs:9000"),
        ("http://rustfs:9000", "http://rustfs:9000"),
        ("HTTPS://rustfs:9000", "https://rustfs:9000"),
        (None, None),
    ],
)
def test_reader_client_preserves_valid_endpoint_tls_policy(monkeypatch, endpoint, expected):
    import boto3

    captured = []
    events = SimpleNamespace(register_last=lambda *_args, **_kwargs: None)
    client = SimpleNamespace(meta=SimpleNamespace(events=events))

    def create_client(service, **kwargs):
        assert service == "s3"
        captured.append(kwargs)
        return client

    monkeypatch.setattr(settings, "S3_ENDPOINT", endpoint)
    monkeypatch.setattr(boto3, "client", create_client)
    assert reader._object_store_client() is client
    assert len(captured) == 1
    assert captured[0]["endpoint_url"] == expected


@pytest.mark.parametrize(
    "endpoint",
    [
        "http://account:private-secret@rustfs:9000",
        "account:private-secret@rustfs:9000",
        "http://[::1:9000",
        "rustfs:99999",
        "https://rustfs:9000/private/path",
        "ftp://rustfs:9000",
    ],
)
def test_reader_client_never_repairs_a_disallowed_endpoint(monkeypatch, endpoint):
    import boto3

    from src.utils.s3_endpoint import S3EndpointConfigurationError

    monkeypatch.setattr(settings, "S3_ENDPOINT", endpoint)
    monkeypatch.setattr(boto3, "client", lambda *_args, **_kwargs: pytest.fail("client created"))
    with pytest.raises(S3EndpointConfigurationError) as raised:
        reader._object_store_client()
    assert "private-secret" not in str(raised.value)


def test_cancelled_async_read_retains_worker_snapshot_until_completion(monkeypatch, artifact):
    entered = Event()
    release = Event()
    snapshot_closed = Event()
    connection_closed = Event()
    descriptors = []
    real_temporary_file = reader.tempfile.TemporaryFile
    real_connect = duckdb.connect

    @contextmanager
    def tracked_snapshot(*args, **kwargs):
        with real_temporary_file(*args, **kwargs) as snapshot:
            descriptors.append(snapshot.fileno())
            yield snapshot
        snapshot_closed.set()

    class _Connection:
        def __init__(self):
            self.conn = real_connect(":memory:")

        def execute(self, sql, parameters):
            assert os.fstat(descriptors[0]).st_nlink == 0
            entered.set()
            if not release.wait(5):
                raise AssertionError("test did not release the cancelled worker")
            return self.conn.execute(sql, parameters)

        def __getattr__(self, name):
            return getattr(self.conn, name)

        def close(self):
            self.conn.close()
            connection_closed.set()

    body = _Body(artifact)
    client = _store(monkeypatch, {"Body": body, "ContentLength": len(artifact)})
    monkeypatch.setattr(reader.tempfile, "TemporaryFile", tracked_snapshot)
    monkeypatch.setattr(reader.duckdb, "connect", lambda *_args, **_kwargs: _Connection())

    async def cancel_during_query():
        task = asyncio.create_task(asyncio.to_thread(_read))
        try:
            assert await asyncio.to_thread(entered.wait, 5), "worker never reached the query"
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            # Cancelling the awaiter cannot close a descriptor still used by the worker.
            assert not snapshot_closed.is_set() and not connection_closed.is_set()
            os.fstat(descriptors[0])
        finally:
            release.set()
        assert await asyncio.to_thread(snapshot_closed.wait, 5)

    asyncio.run(cancel_during_query())
    assert connection_closed.is_set() and snapshot_closed.is_set()
    assert body.closed and client.closed
    with pytest.raises(OSError):
        os.fstat(descriptors[0])


def test_real_streaming_body_incomplete_read_has_safe_reason_and_closes(monkeypatch, artifact):
    from botocore.response import StreamingBody

    stream = io.BytesIO(artifact)
    body = StreamingBody(stream, len(artifact) + 1)
    client = _store(monkeypatch, {"Body": body, "ContentLength": len(artifact) + 1})
    with pytest.raises(reader.NormalsReadError) as raised:
        _read()
    assert raised.value.reason == "incomplete_download"
    assert reader.normals_read_error_reason(raised.value) == "incomplete_download"
    assert stream.closed and client.closed
    assert len(client.gets) == 1


def test_sparse_local_oversize_has_safe_reason_without_loading_bytes(monkeypatch, tmp_path):
    path = tmp_path / "normals.parquet"
    with path.open("wb") as stream:
        stream.truncate(_LIMIT + 1)
    monkeypatch.setattr(settings, "ALLOW_LOCAL_PARQUET_SOURCE", True)
    monkeypatch.setattr(settings, "LOCAL_PARQUET_SOURCE_ROOT", str(tmp_path))
    with pytest.raises(reader.NormalsReadError) as raised:
        _read(str(path))
    assert raised.value.reason == "source_too_large"
    assert reader.normals_read_error_reason(raised.value) == "source_too_large"


def test_safe_reason_ignores_attributes_on_untrusted_exceptions():
    error = OSError("https://private-endpoint.invalid/ secret-value")
    error.reason = "parquet_read_failed"
    assert reader.normals_read_error_reason(error) == "source_read_failed"


@pytest.mark.parametrize("reason", ["https://private-endpoint.invalid/secret", None, []])
def test_safe_reason_refuses_tampered_typed_errors(reason):
    error = reader.NormalsReadError("source_too_large")
    error.reason = reason
    assert reader.normals_read_error_reason(error) == "source_read_failed"


def test_snapshot_creation_failure_has_safe_reason_and_closes_descriptor(monkeypatch, artifact):
    descriptors = []
    real_temporary_file = reader.tempfile.TemporaryFile

    def tracked_snapshot(*args, **kwargs):
        snapshot = real_temporary_file(*args, **kwargs)
        descriptors.append(snapshot.fileno())
        return snapshot

    def fail_connect(*_args, **_kwargs):
        raise OSError("https://private-endpoint.invalid/ secret-value")

    body = _Body(artifact)
    client = _store(monkeypatch, {"Body": body, "ContentLength": len(artifact)})
    monkeypatch.setattr(reader.tempfile, "TemporaryFile", tracked_snapshot)
    monkeypatch.setattr(reader.duckdb, "connect", fail_connect)
    with pytest.raises(reader.NormalsReadError) as raised:
        _read()
    assert raised.value.reason == "snapshot_failed"
    assert reader.normals_read_error_reason(raised.value) == "snapshot_failed"
    assert body.closed and client.closed
    assert len(descriptors) == 1
    with pytest.raises(OSError):
        os.fstat(descriptors[0])
