"""Descriptor-anchored local I/O for the offline normals preflight.

This is a POSIX/Linux guard against pathname replacement, not an OS sandbox.
The operator must control the configured root and its ancestor directories:
someone able to rename an already-open directory can move its inode outside
the root while an operation is in progress. All operations still target the
held inode, never a substituted symlink or its destination.
"""

from __future__ import annotations

import os
import secrets
import stat
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

_FORBIDDEN = frozenset("*?[]{}")
_DIRECTORY_FLAGS = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
_FILE_FLAGS = os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC | os.O_NONBLOCK


@dataclass(frozen=True)
class Snapshot:
    """An anonymous, verified-byte Parquet file held open for its consumer."""

    fd: int
    path: Path


class ConfinedLocalIO:
    """Perform local file operations relative to a held, no-follow root fd.

    Paths passed to methods must be raw absolute operator paths, not paths
    returned by ``Path.resolve`` or another symlink-following resolver.
    ``/proc/self/fd`` is intentionally required for stable snapshot paths.
    """

    def __init__(self, root: Path) -> None:
        if sys.platform != "linux" or not Path("/proc/self/fd").is_dir():
            raise OSError("confined local I/O requires Linux procfs")
        root_path = Path(root)
        if not root_path.is_absolute() or root_path == Path("/"):
            raise ValueError("LOCAL_PARQUET_SOURCE_ROOT must be an absolute non-root directory")
        if ".." in root_path.parts or any(char in str(root_path) for char in _FORBIDDEN):
            raise ValueError("LOCAL_PARQUET_SOURCE_ROOT must be a concrete traversal-free path")

        descriptor = os.open("/", _DIRECTORY_FLAGS)
        try:
            for component in root_path.parts[1:]:
                next_descriptor = os.open(component, _DIRECTORY_FLAGS, dir_fd=descriptor)
                os.close(descriptor)
                descriptor = next_descriptor
        except BaseException:
            os.close(descriptor)
            raise
        self.root = root_path
        self._root_fd: int | None = descriptor

    def __enter__(self) -> ConfinedLocalIO:
        self._require_open()
        return self

    def __exit__(self, _type: object, _value: object, _traceback: object) -> None:
        self.close()

    def close(self) -> None:
        descriptor, self._root_fd = self._root_fd, None
        if descriptor is not None:
            os.close(descriptor)

    def _require_open(self) -> int:
        if self._root_fd is None:
            raise ValueError("confined local I/O is closed")
        return self._root_fd

    def _parts(self, path: Path, *, role: str, allow_root: bool = False) -> tuple[str, ...]:
        raw = str(path)
        if not raw or "://" in raw or any(char in raw for char in _FORBIDDEN):
            raise ValueError(f"{role} must be one concrete local path without glob or URI syntax")
        candidate = Path(path)
        if not candidate.is_absolute() or ".." in candidate.parts:
            raise ValueError(f"{role} must be an absolute traversal-free path")
        if not candidate.is_relative_to(self.root):
            raise ValueError(f"{role} must be under LOCAL_PARQUET_SOURCE_ROOT")
        relative = candidate.relative_to(self.root).parts
        if not relative and not allow_root:
            raise ValueError(f"{role} must name a path below LOCAL_PARQUET_SOURCE_ROOT")
        return relative

    def _directory(self, parts: tuple[str, ...]) -> int:
        descriptor = os.dup(self._require_open())
        try:
            for component in parts:
                next_descriptor = os.open(component, _DIRECTORY_FLAGS, dir_fd=descriptor)
                os.close(descriptor)
                descriptor = next_descriptor
            return descriptor
        except BaseException:
            os.close(descriptor)
            raise

    @staticmethod
    def _unlink_if_owned(name: str, parent_fd: int, file_fd: int) -> None:
        """Only clean a name while it still denotes our open regular file."""
        try:
            named = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        except FileNotFoundError:
            return
        opened = os.fstat(file_fd)
        if (named.st_dev, named.st_ino) == (opened.st_dev, opened.st_ino):
            os.unlink(name, dir_fd=parent_fd)

    @staticmethod
    def _rmdir_if_owned(name: str, parent_fd: int, directory_fd: int) -> None:
        """Only remove our private directory, never a substituted symlink."""
        try:
            named = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        except FileNotFoundError:
            return
        opened = os.fstat(directory_fd)
        if (named.st_dev, named.st_ino) == (opened.st_dev, opened.st_ino):
            os.rmdir(name, dir_fd=parent_fd)

    def read_bytes(self, path: Path, *, role: str, limit: int) -> bytes:
        """Read at most ``limit`` bytes from a regular, no-follow file."""
        if limit < 0:
            raise ValueError("read limit must not be negative")
        parts = self._parts(path, role=role)
        parent_fd = self._directory(parts[:-1])
        try:
            descriptor = os.open(parts[-1], _FILE_FLAGS, dir_fd=parent_fd)
            try:
                details = os.fstat(descriptor)
                if not stat.S_ISREG(details.st_mode):
                    raise ValueError(f"{role} must be a regular file")
                if details.st_size > limit:
                    raise ValueError(f"{role} exceeds the {limit}-byte limit")
                chunks: list[bytes] = []
                remaining = limit + 1
                while remaining:
                    chunk = os.read(descriptor, min(remaining, 1024 * 1024))
                    if not chunk:
                        break
                    chunks.append(chunk)
                    remaining -= len(chunk)
                content = b"".join(chunks)
                if len(content) > limit:
                    raise ValueError(f"{role} exceeds the {limit}-byte limit")
                return content
            finally:
                os.close(descriptor)
        finally:
            os.close(parent_fd)

    @contextmanager
    def private_snapshot(self, work_root: Path, content: bytes) -> Iterator[Snapshot]:
        """Yield an unlinked file fd and stable procfd path for DuckDB.

        The private 0700 directory and 0600 file are unlinked before yielding,
        so replacing the work-root pathname cannot replace the snapshot and no
        predictable snapshot name remains for a concurrent writer to mutate.
        """
        parts = self._parts(work_root, role="work root", allow_root=True)
        work_fd = self._directory(parts)
        private_name: str | None = None
        private_fd: int | None = None
        snapshot_fd: int | None = None
        source_created = False
        source_unlinked = False
        try:
            for _ in range(10):
                name = f".normals-preflight-{secrets.token_hex(16)}"
                try:
                    os.mkdir(name, mode=0o700, dir_fd=work_fd)
                except FileExistsError:
                    continue
                private_name = name
                break
            if private_name is None:
                raise FileExistsError("could not create a private snapshot directory")
            private_fd = os.open(private_name, _DIRECTORY_FLAGS, dir_fd=work_fd)
            snapshot_fd = os.open(
                "source.parquet",
                os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC,
                0o600,
                dir_fd=private_fd,
            )
            source_created = True
            view = memoryview(content)
            while view:
                written = os.write(snapshot_fd, view)
                if written <= 0:
                    raise OSError("snapshot write made no progress")
                view = view[written:]
            os.fsync(snapshot_fd)
            os.lseek(snapshot_fd, 0, os.SEEK_SET)
            os.unlink("source.parquet", dir_fd=private_fd)
            source_unlinked = True
            self._rmdir_if_owned(private_name, work_fd, private_fd)
            private_name = None
            yield Snapshot(snapshot_fd, Path(f"/proc/self/fd/{snapshot_fd}"))
        finally:
            try:
                if private_fd is not None:
                    try:
                        if source_created and not source_unlinked and snapshot_fd is not None:
                            self._unlink_if_owned("source.parquet", private_fd, snapshot_fd)
                        if private_name is not None:
                            self._rmdir_if_owned(private_name, work_fd, private_fd)
                    finally:
                        os.close(private_fd)
            finally:
                try:
                    if snapshot_fd is not None:
                        os.close(snapshot_fd)
                finally:
                    os.close(work_fd)

    def publish_new(self, output: Path, payload: bytes) -> None:
        """Atomically publish a new file; never replace an existing name."""
        parts = self._parts(output, role="output")
        parent_fd = self._directory(parts[:-1])
        private_name: str | None = None
        private_fd: int | None = None
        temporary_fd: int | None = None
        temporary_created = False
        try:
            for _ in range(10):
                name = f".normals-manifest-{secrets.token_hex(16)}"
                try:
                    os.mkdir(name, mode=0o700, dir_fd=parent_fd)
                except FileExistsError:
                    continue
                private_name = name
                break
            if private_name is None:
                raise FileExistsError("could not create a private manifest directory")
            private_fd = os.open(private_name, _DIRECTORY_FLAGS, dir_fd=parent_fd)
            temporary_fd = os.open(
                "manifest.json",
                os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC,
                0o600,
                dir_fd=private_fd,
            )
            temporary_created = True
            view = memoryview(payload)
            while view:
                written = os.write(temporary_fd, view)
                if written <= 0:
                    raise OSError("manifest write made no progress")
                view = view[written:]
            os.fsync(temporary_fd)
            # The procfd source denotes the held inode, not a replaceable
            # temporary pathname. AT_SYMLINK_FOLLOW resolves this procfs fd.
            os.link(
                f"/proc/self/fd/{temporary_fd}",
                parts[-1],
                dst_dir_fd=parent_fd,
                follow_symlinks=True,
            )
        finally:
            try:
                if private_fd is not None:
                    try:
                        if temporary_created and temporary_fd is not None:
                            self._unlink_if_owned("manifest.json", private_fd, temporary_fd)
                        if private_name is not None:
                            self._rmdir_if_owned(private_name, parent_fd, private_fd)
                    finally:
                        os.close(private_fd)
            finally:
                try:
                    if temporary_fd is not None:
                        os.close(temporary_fd)
                finally:
                    os.close(parent_fd)
