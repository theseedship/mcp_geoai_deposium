"""POSIX pathname-swap regressions for the offline normals local I/O guard."""

from __future__ import annotations

import os
from pathlib import Path

import duckdb
import pytest

from src.cli import _normals_local_io as local_io


def test_bounded_read_snapshot_and_publish(tmp_path: Path):
    source = tmp_path / "source.parquet"
    content = Path("tests/fixtures/gai3/internal-wkb.parquet").read_bytes()
    source.write_bytes(content)
    work = tmp_path / "work"
    work.mkdir()
    output = tmp_path / "manifest.json"

    with local_io.ConfinedLocalIO(tmp_path) as confined:
        assert confined.read_bytes(source, role="source", limit=len(content)) == content
        with confined.private_snapshot(work, content) as snapshot:
            assert snapshot.path.read_bytes() == content
            assert list(work.iterdir()) == []
            count = (
                duckdb.connect(":memory:")
                .execute("SELECT count(*) FROM read_parquet(?)", [str(snapshot.path)])
                .fetchone()
            )
            assert count == (3,)
        assert not snapshot.path.exists()
        confined.publish_new(output, b'{"ok":true}\n')

    assert output.read_bytes() == b'{"ok":true}\n'
    assert list(work.iterdir()) == []
    assert list(tmp_path.glob(".normals-manifest-*")) == []


def test_bad_paths_symlinks_nonregular_and_oversize_are_refused(tmp_path: Path):
    source = tmp_path / "source.parquet"
    source.write_bytes(b"1234")
    folder = tmp_path / "folder"
    folder.mkdir()
    (folder / "nested.parquet").write_bytes(b"data")
    (tmp_path / "linked.parquet").symlink_to(source)
    (tmp_path / "linked-folder").symlink_to(folder, target_is_directory=True)
    fifo = tmp_path / "not-a-file.parquet"
    os.mkfifo(fifo)

    with local_io.ConfinedLocalIO(tmp_path) as confined:
        with pytest.raises(ValueError, match="limit"):
            confined.read_bytes(source, role="source", limit=3)
        with pytest.raises(ValueError, match="regular file"):
            confined.read_bytes(folder, role="source", limit=10)
        with pytest.raises(ValueError, match="regular file"):
            confined.read_bytes(fifo, role="source", limit=10)
        for path in (
            tmp_path / "linked.parquet",
            tmp_path / "linked-folder" / "nested.parquet",
        ):
            with pytest.raises(OSError):
                confined.read_bytes(path, role="source", limit=10)
        for path in (tmp_path.parent / "outside", tmp_path / ".." / "escape"):
            with pytest.raises(ValueError):
                confined.read_bytes(path, role="source", limit=10)
        for path in (tmp_path / "*.parquet", Path("s3://bucket/object")):
            with pytest.raises(ValueError):
                confined.read_bytes(path, role="source", limit=10)


@pytest.mark.parametrize("filename", ["source.parquet", "provenance.json"])
def test_read_uses_held_parent_after_rename_and_symlink_swap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, filename: str
):
    root = tmp_path / "root"
    root.mkdir()
    trusted = root / "inputs"
    trusted.mkdir()
    (trusted / filename).write_bytes(b"authorized")
    moved = root / "moved-inputs"
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / filename).write_bytes(b"attacker")
    requested = trusted / filename
    real_open = os.open
    swapped = False

    def open_after_swap(path, flags, *args, **kwargs):
        nonlocal swapped
        if not swapped and path == filename and kwargs.get("dir_fd") is not None:
            trusted.rename(moved)
            trusted.symlink_to(outside, target_is_directory=True)
            swapped = True
        return real_open(path, flags, *args, **kwargs)

    with local_io.ConfinedLocalIO(root) as confined:
        monkeypatch.setattr(local_io.os, "open", open_after_swap)
        assert confined.read_bytes(requested, role="source", limit=20) == b"authorized"

    assert swapped
    assert (outside / filename).read_bytes() == b"attacker"


def test_root_replacement_cannot_redirect_any_read(tmp_path: Path):
    root = tmp_path / "root"
    root.mkdir()
    (root / "source.parquet").write_bytes(b"authorized")
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "source.parquet").write_bytes(b"attacker")

    with local_io.ConfinedLocalIO(root) as confined:
        root.rename(tmp_path / "moved-root")
        root.symlink_to(outside, target_is_directory=True)
        assert (
            confined.read_bytes(root / "source.parquet", role="source", limit=20) == b"authorized"
        )


def test_snapshot_work_parent_swap_cannot_write_to_symlink_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    root = tmp_path / "root"
    root.mkdir()
    work = root / "work"
    work.mkdir()
    moved = root / "moved-work"
    outside = tmp_path / "outside"
    outside.mkdir()
    real_mkdir = os.mkdir
    swapped = False

    def mkdir_after_swap(path, mode=0o777, *, dir_fd=None):
        nonlocal swapped
        if not swapped and str(path).startswith(".normals-preflight-"):
            work.rename(moved)
            work.symlink_to(outside, target_is_directory=True)
            swapped = True
        return real_mkdir(path, mode=mode, dir_fd=dir_fd)

    with local_io.ConfinedLocalIO(root) as confined:
        monkeypatch.setattr(local_io.os, "mkdir", mkdir_after_swap)
        with confined.private_snapshot(work, b"verified") as snapshot:
            assert snapshot.path.read_bytes() == b"verified"
            assert list(outside.iterdir()) == []

    assert swapped
    assert list(moved.iterdir()) == []
    assert list(outside.iterdir()) == []


def test_publish_parent_swap_cannot_write_to_symlink_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    root = tmp_path / "root"
    root.mkdir()
    parent = root / "output"
    parent.mkdir()
    moved = root / "moved-output"
    outside = tmp_path / "outside"
    outside.mkdir()
    real_link = os.link
    swapped = False

    def link_after_swap(source, destination, **kwargs):
        nonlocal swapped
        if not swapped:
            parent.rename(moved)
            parent.symlink_to(outside, target_is_directory=True)
            swapped = True
        return real_link(source, destination, **kwargs)

    with local_io.ConfinedLocalIO(root) as confined:
        monkeypatch.setattr(local_io.os, "link", link_after_swap)
        confined.publish_new(parent / "manifest.json", b"authorized")

    assert swapped
    assert (moved / "manifest.json").read_bytes() == b"authorized"
    assert list(outside.iterdir()) == []
    assert list(moved.glob(".normals-manifest-*")) == []


def test_publication_never_clobbers_file_or_symlink_and_cleans_temp(tmp_path: Path):
    output = tmp_path / "manifest.json"
    output.write_bytes(b"existing")
    with local_io.ConfinedLocalIO(tmp_path) as confined:
        with pytest.raises(FileExistsError):
            confined.publish_new(output, b"new")
        assert output.read_bytes() == b"existing"
        output.unlink()
        target = tmp_path / "provenance.json"
        target.write_bytes(b"secret")
        output.symlink_to(target)
        with pytest.raises(FileExistsError):
            confined.publish_new(output, b"new")
        assert target.read_bytes() == b"secret"
        assert output.is_symlink()
    assert list(tmp_path.glob(".normals-manifest-*")) == []


def test_racing_symlink_publication_is_not_followed_or_cleaned(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    output = tmp_path / "manifest.json"
    target = tmp_path / "target.json"
    target.write_bytes(b"original")
    real_link = os.link

    def link_after_other_writer(source, destination, **kwargs):
        os.symlink("target.json", destination, dir_fd=kwargs["dst_dir_fd"])
        return real_link(source, destination, **kwargs)

    with local_io.ConfinedLocalIO(tmp_path) as confined:
        monkeypatch.setattr(local_io.os, "link", link_after_other_writer)
        with pytest.raises(FileExistsError):
            confined.publish_new(output, b"new")

    assert output.is_symlink()
    assert target.read_bytes() == b"original"
    assert list(tmp_path.glob(".normals-manifest-*")) == []


def test_snapshot_exclusive_create_collision_preserves_other_file_and_closes_fd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    work = tmp_path / "work"
    work.mkdir()
    real_open = os.open
    private_descriptors: list[int] = []
    injected = False

    def open_with_collision(path, flags, *args, **kwargs):
        nonlocal injected
        if str(path).startswith(".normals-preflight-"):
            descriptor = real_open(path, flags, *args, **kwargs)
            private_descriptors.append(descriptor)
            return descriptor
        if path == "source.parquet" and flags & os.O_EXCL and not injected:
            competing = real_open(
                path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600, dir_fd=kwargs["dir_fd"]
            )
            os.write(competing, b"other-writer")
            os.close(competing)
            injected = True
        return real_open(path, flags, *args, **kwargs)

    with local_io.ConfinedLocalIO(tmp_path) as confined:
        with monkeypatch.context() as patcher:
            patcher.setattr(local_io.os, "open", open_with_collision)
            with pytest.raises(OSError):
                with confined.private_snapshot(work, b"verified"):
                    pytest.fail("exclusive create unexpectedly succeeded")

    assert injected
    assert len(private_descriptors) == 1
    with pytest.raises(OSError):
        os.fstat(private_descriptors[0])
    private = next(work.glob(".normals-preflight-*"))
    assert (private / "source.parquet").read_bytes() == b"other-writer"


def test_publish_exclusive_create_collision_preserves_other_file_and_closes_fd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    output = tmp_path / "output.json"
    real_open = os.open
    private_descriptors: list[int] = []
    injected = False

    def open_with_collision(path, flags, *args, **kwargs):
        nonlocal injected
        if str(path).startswith(".normals-manifest-"):
            descriptor = real_open(path, flags, *args, **kwargs)
            private_descriptors.append(descriptor)
            return descriptor
        if path == "manifest.json" and flags & os.O_EXCL and not injected:
            competing = real_open(
                path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600, dir_fd=kwargs["dir_fd"]
            )
            os.write(competing, b"other-writer")
            os.close(competing)
            injected = True
        return real_open(path, flags, *args, **kwargs)

    with local_io.ConfinedLocalIO(tmp_path) as confined:
        with monkeypatch.context() as patcher:
            patcher.setattr(local_io.os, "open", open_with_collision)
            with pytest.raises(OSError):
                confined.publish_new(output, b"ours")

    assert injected
    assert len(private_descriptors) == 1
    with pytest.raises(OSError):
        os.fstat(private_descriptors[0])
    private = next(tmp_path.glob(".normals-manifest-*"))
    assert (private / "manifest.json").read_bytes() == b"other-writer"
    assert not output.exists()
