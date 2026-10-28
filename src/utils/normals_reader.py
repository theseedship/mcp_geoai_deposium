"""Bind a normals query to one bounded capture of its original Parquet bytes.

The fixed 8 MiB cap bounds source bytes, not decompressed rows or DuckDB memory.
Linux procfs is required: DuckDB reads an anonymous, held snapshot descriptor,
never the mutable canonical source. No unbound httpfs fallback is permitted.
"""

from __future__ import annotations

import hashlib
import re
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import duckdb

from src.cli._normals_local_io import ConfinedLocalIO
from src.config import settings

MAX_NORMALS_ARTIFACT_BYTES = 8 * 1024 * 1024
_SAFE_ZONE_RE = re.compile(r"[A-Za-z0-9._-]+")
_READ_ERROR_REASONS = frozenset(
    {
        "source_too_large",
        "incomplete_download",
        "parquet_read_failed",
        "source_read_failed",
        "snapshot_failed",
    }
)


class NormalsReadError(ValueError):
    """A read refusal whose diagnostic category contains no source/provider text."""

    def __init__(self, reason: str) -> None:
        self.reason = (
            reason
            if type(reason) is str and reason in _READ_ERROR_REASONS
            else "source_read_failed"
        )
        super().__init__(self.reason)


def normals_read_error_reason(error: Exception) -> str:
    """Return only an allowlisted diagnostic, even for unknown or altered errors."""
    if isinstance(error, NormalsReadError):
        try:
            reason = error.reason
        except Exception:
            return "source_read_failed"
        if type(reason) is str and reason in _READ_ERROR_REASONS:
            return reason
    return "source_read_failed"


@dataclass(frozen=True)
class NormalsArtifactRead:
    """Rows and identity from the same original artifact capture."""

    rows: list[dict]
    artifact_version: str
    artifact_size_bytes: int


def _object_store_client() -> Any:
    """Reuse endpoint policy and signing capture without an import-time cycle."""
    # The archive module imports climate._load_zones. Importing it only when a
    # request needs S3 leaves climate's serving-reader import acyclic.
    from src.cli.forecast_ensemble_archive import _object_store_client as create_client

    return create_client(normalize_scheme_less_endpoint=True)


def _read_source_bytes(src: str, *, zone_id: str) -> bytes:
    """Capture at most the cap from one confined local read or one S3 GET."""
    if not _SAFE_ZONE_RE.fullmatch(zone_id) or ".." in zone_id:
        raise ValueError("invalid normals zone")
    if src.startswith("s3://"):
        bucket = settings.S3_BUCKET
        key = f"climate/normals/{zone_id}.parquet"
        if not bucket or src != f"s3://{bucket}/{key}":
            raise ValueError("normals source must be the configured canonical object")
        client = _object_store_client()
        body = None
        try:
            from botocore.exceptions import IncompleteReadError

            response = client.get_object(Bucket=bucket, Key=key)
            body = response.get("Body")
            if body is None or not callable(getattr(body, "read", None)):
                raise ValueError("normals object has no readable body")
            declared = response.get("ContentLength")
            if type(declared) is not int or declared < 0:
                raise ValueError("normals object has an invalid content length")
            if declared > MAX_NORMALS_ARTIFACT_BYTES:
                raise NormalsReadError("source_too_large")
            chunks: list[bytes] = []
            remaining = MAX_NORMALS_ARTIFACT_BYTES + 1
            while remaining:
                requested = min(remaining, 1024 * 1024)
                try:
                    chunk = body.read(requested)
                except IncompleteReadError as error:
                    raise NormalsReadError("incomplete_download") from error
                if not isinstance(chunk, bytes) or len(chunk) > requested:
                    raise ValueError("normals object body returned invalid bytes")
                if not chunk:
                    break
                chunks.append(chunk)
                remaining -= len(chunk)
            content = b"".join(chunks)
            if len(content) > MAX_NORMALS_ARTIFACT_BYTES:
                raise NormalsReadError("source_too_large")
            if len(content) != declared:
                raise NormalsReadError("incomplete_download")
            return content
        finally:
            try:
                close_body = getattr(body, "close", None)
                if callable(close_body):
                    close_body()
            finally:
                client.close()

    if not settings.ALLOW_LOCAL_PARQUET_SOURCE:
        raise ValueError("local normals sources are disabled")
    raw_root = settings.LOCAL_PARQUET_SOURCE_ROOT.strip()
    if not raw_root:
        raise ValueError("local normals source root is required")
    # Keep raw components: resolve() would follow symlinks before the anchored
    # O_NOFOLLOW reader can reject them. absolute() only supplies the cwd.
    path = Path(src).expanduser().absolute()
    with ConfinedLocalIO(Path(raw_root).expanduser()) as local_io:
        try:
            return local_io.read_bytes(
                path, role="normals source", limit=MAX_NORMALS_ARTIFACT_BYTES
            )
        except ValueError as error:
            # Translate only the helper's exact first-party cap refusal. Other
            # confinement/read failures keep the generic source category.
            cap_refusal = f"normals source exceeds the {MAX_NORMALS_ARTIFACT_BYTES}-byte limit"
            if error.args == (cap_refusal,):
                raise NormalsReadError("source_too_large") from error
            raise


def read_normals_artifact(
    src: str, *, zone_id: str, variables: list[str] | None
) -> NormalsArtifactRead:
    """Hash one bounded capture and select v1/v2 rows from that same private snapshot."""
    if sys.platform != "linux" or not Path("/proc/self/fd").is_dir():
        raise NormalsReadError("snapshot_failed")
    content = _read_source_bytes(src, zone_id=zone_id)
    if not content:
        raise ValueError("normals artifact is empty")
    version = f"sha256:{hashlib.sha256(content).hexdigest()}"
    # TemporaryFile is unlinked (or O_TMPFILE) before it is returned. Keeping
    # its descriptor open through fetchall binds the query to the held bytes.
    try:
        with tempfile.TemporaryFile(mode="w+b") as snapshot:
            snapshot.write(content)
            snapshot.flush()
            snapshot_path = f"/proc/self/fd/{snapshot.fileno()}"
            conn = duckdb.connect(
                ":memory:",
                config={"autoinstall_known_extensions": False, "autoload_known_extensions": False},
            )
            try:
                # SELECT * preserves v1 artifacts whose wet-day columns are absent.
                sql = "SELECT * FROM read_parquet(?) WHERE zone_id = ?"
                params: list = [snapshot_path, zone_id]
                if variables:
                    placeholders = ", ".join("?" for _ in variables)
                    sql += f" AND variable IN ({placeholders})"  # noqa: S608 - bound parameters
                    params.extend(variables)
                sql += " ORDER BY variable, month"
                rows = conn.execute(sql, params).fetchall()
                names = [description[0] for description in conn.description]
            except Exception as error:
                raise NormalsReadError("parquet_read_failed") from error
            finally:
                conn.close()
    except NormalsReadError:
        raise
    except Exception as error:
        raise NormalsReadError("snapshot_failed") from error
    return NormalsArtifactRead(
        rows=[
            {key: value for key, value in zip(names, row, strict=True) if key != "geometry"}
            for row in rows
        ],
        artifact_version=version,
        artifact_size_bytes=len(content),
    )
