"""Bound untrusted handoff enumeration before retaining its complete tree."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from stat import S_IFIFO
from types import SimpleNamespace

import pytest

from src.patterns import library_handoff


@pytest.mark.parametrize("kind", ["file", "directory"])
def test_verification_stops_consuming_at_the_entry_budget(tmp_path, monkeypatch, kind):
    """Removing the incremental budget must overrun this synthetic iterator."""
    child = tmp_path / "child"
    if kind == "directory":
        child.mkdir()
    else:
        child.write_bytes(b"x")
    metadata = child.stat()
    consumed = 0
    closed = False

    def paths():
        nonlocal consumed
        for _ in range(1044):
            consumed += 1
            if consumed > 5:
                pytest.fail("verification exhausted traversal beyond the entry budget")
            yield child

    @contextmanager
    def scan(directory):
        nonlocal closed
        assert Path(directory) == tmp_path
        try:
            yield (
                SimpleNamespace(
                    path=str(path),
                    is_symlink=lambda: False,
                    stat=lambda *, follow_symlinks: metadata,
                )
                for path in paths()
            )
        finally:
            closed = True

    # Exercise the former rglob implementation and the streaming filesystem
    # boundary with the same finite, instrumented source; no huge tree is needed.
    monkeypatch.setattr(Path, "rglob", lambda *_: paths())
    monkeypatch.setattr(library_handoff, "scandir", scan, raising=False)
    monkeypatch.setattr(library_handoff, "MAX_PACKAGE_ENTRIES", 4, raising=False)
    with pytest.raises(ValueError, match="entry limit"):
        library_handoff.verify_library_handoff(tmp_path, "0" * 64)
    assert consumed == 5  # One overflow entry is observed but never retained.
    assert closed


@pytest.mark.parametrize("target_kind", ["file", "directory", "missing"])
def test_verification_refuses_symlinks_without_scanning_targets(tmp_path, monkeypatch, target_kind):
    target = tmp_path / "target"
    if target_kind == "directory":
        target.mkdir()
    elif target_kind == "file":
        target.write_bytes(b"outside package")
    root = tmp_path / "handoff"
    root.mkdir()
    link = root / "linked"
    try:
        link.symlink_to(target, target_is_directory=target_kind == "directory")
    except (OSError, NotImplementedError) as error:
        pytest.skip(f"symlink creation is unavailable: {error}")

    from os import scandir

    @contextmanager
    def scan(directory):
        assert Path(directory) == root, "verification descended into a symlink target"
        with scandir(directory) as entries:
            yield entries

    monkeypatch.setattr(library_handoff, "scandir", scan, raising=False)
    with pytest.raises(ValueError, match="Symlinks"):
        library_handoff.verify_library_handoff(root, "0" * 64)


def test_verification_refuses_a_symlink_root_before_traversal(tmp_path, monkeypatch):
    target = tmp_path / "target"
    target.mkdir()
    root = tmp_path / "handoff"
    try:
        root.symlink_to(target, target_is_directory=True)
    except (OSError, NotImplementedError) as error:
        pytest.skip(f"symlink creation is unavailable: {error}")

    def unexpected_scan(*_):
        pytest.fail("verification scanned a symlink root")

    monkeypatch.setattr(library_handoff, "scandir", unexpected_scan, raising=False)
    monkeypatch.setattr(Path, "rglob", unexpected_scan)
    with pytest.raises(ValueError, match="Symlinks"):
        library_handoff.verify_library_handoff(root, "0" * 64)


@pytest.mark.parametrize("budget, error", [(1, "entry limit"), (2, "Manifest SHA-256")])
def test_verification_allows_exactly_the_entry_budget(tmp_path, monkeypatch, budget, error):
    (tmp_path / "empty").mkdir()
    (tmp_path / "manifest.json").write_bytes(b"{}")
    monkeypatch.setattr(library_handoff, "MAX_PACKAGE_ENTRIES", budget)
    with pytest.raises(ValueError, match=error):
        library_handoff.verify_library_handoff(tmp_path, "0" * 64)


@pytest.mark.parametrize("limit", ["files", "bytes"])
def test_verification_retains_independent_file_and_byte_limits(tmp_path, monkeypatch, limit):
    for index in range(3):
        (tmp_path / f"{index}.json").write_bytes(b"123")
    if limit == "files":
        monkeypatch.setattr(library_handoff, "MAX_PACKAGE_FILES", 2)
    else:
        monkeypatch.setattr(library_handoff, "MAX_PACKAGE_BYTES", 8)
    with pytest.raises(ValueError, match="file or byte limit"):
        library_handoff.verify_library_handoff(tmp_path, "0" * 64)


def test_verification_rechecks_a_queued_directory_before_scanning(tmp_path, monkeypatch):
    from os import scandir

    target = tmp_path / "target"
    target.mkdir()
    root = tmp_path / "handoff"
    root.mkdir()
    queued = root / "child"
    queued.mkdir()
    probe = tmp_path / "probe"
    try:
        probe.symlink_to(target, target_is_directory=True)
    except (OSError, NotImplementedError) as error:
        pytest.skip(f"symlink creation is unavailable: {error}")
    probe.unlink()

    @contextmanager
    def scan(directory):
        assert Path(directory) == root, "verification scanned a replaced directory"
        with scandir(directory) as entries:
            yield entries
        queued.rmdir()
        queued.symlink_to(target, target_is_directory=True)

    monkeypatch.setattr(library_handoff, "scandir", scan)
    with pytest.raises(ValueError, match="Symlinks"):
        library_handoff.verify_library_handoff(root, "0" * 64)


def test_verification_refuses_nonregular_entries_before_reading(tmp_path, monkeypatch):
    @contextmanager
    def scan(directory):
        assert Path(directory) == tmp_path
        yield iter(
            [
                SimpleNamespace(
                    path=str(tmp_path / "manifest.json"),
                    is_symlink=lambda: False,
                    stat=lambda *, follow_symlinks: SimpleNamespace(st_mode=S_IFIFO),
                )
            ]
        )

    def unexpected_read(*_):
        pytest.fail("verification read an unsupported entry")

    monkeypatch.setattr(library_handoff, "scandir", scan)
    monkeypatch.setattr(library_handoff, "_bounded_read", unexpected_read)
    with pytest.raises(ValueError, match="Unsupported handoff entry"):
        library_handoff.verify_library_handoff(tmp_path, "0" * 64)
