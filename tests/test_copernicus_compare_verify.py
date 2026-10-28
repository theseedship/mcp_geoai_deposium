"""Tampering must fail before any recorded-source or comparison replay."""

import hashlib
import json

import pytest

from benchmarks import copernicus_compare_verify
from benchmarks.copernicus_compare import compare_outputs
from benchmarks.copernicus_compare_verify import verify_package


@pytest.fixture
def package(tmp_path):
    artifact = tmp_path / "evidence.txt"
    artifact.write_bytes(b"retained evidence\n")
    manifest = {
        "artifact_inventory": [
            {
                "path": "evidence.txt",
                "bytes": artifact.stat().st_size,
                "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
            }
        ]
    }
    encoded = json.dumps(manifest).encode()
    (tmp_path / "manifest.json").write_bytes(encoded)
    return tmp_path, hashlib.sha256(encoded).hexdigest()


def test_wrong_trusted_manifest_digest_is_rejected(package):
    directory, _ = package
    with pytest.raises(ValueError, match="manifest"):
        verify_package(directory, "0" * 64)


def test_modified_inventoried_byte_is_rejected(package):
    directory, digest = package
    # Preserve the byte length so the digest check, rather than size alone, matters.
    (directory / "evidence.txt").write_bytes(b"Retained evidence\n")
    with pytest.raises(ValueError, match="artifact"):
        verify_package(directory, digest)


@pytest.mark.parametrize("extra", ["extra.txt", "nested/manifest.json"])
def test_unlisted_extra_artifact_is_rejected(package, extra):
    directory, digest = package
    path = directory / extra
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"unlisted")
    with pytest.raises(ValueError, match="inventory"):
        verify_package(directory, digest)


@pytest.mark.parametrize("native_returned", [False, True])
def test_native_threshold_violation_is_reported_without_rejecting_boundary_evidence(
    native_returned,
):
    comparison = compare_outputs(
        [[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 1.0]],
        ["x", "y"],
        {"Patterns": [["x+, y+", "sup=0.5, conf=1.0"]] if native_returned else []},
        {"itemsets": [{"items": ["x+", "y+"], "support": 0.25}]},
    )

    violations = copernicus_compare_verify._native_threshold_violations(
        "synthetic", comparison["identities"]
    )

    assert violations == (
        [
            {
                "pilot": "synthetic",
                "items": ["x+", "y+"],
                "native_raw_indexes": [0],
                "hit_count": 3,
                "unordered_pair_count": 6,
                "native_support": 0.5,
            }
        ]
        if native_returned
        else []
    )


@pytest.fixture
def source_snapshot(tmp_path, monkeypatch):
    snapshot = tmp_path / "source"
    package = tmp_path / "package"
    package.mkdir()
    sources = {}
    for name in copernicus_compare_verify.SOURCES:
        normalized = (copernicus_compare_verify.ROOT / name).read_bytes().replace(b"\r\n", b"\n")
        destination = snapshot / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(normalized.replace(b"\n", b"\r\n"))
        sources[name] = hashlib.sha256(normalized).hexdigest()
    config = json.dumps({"pilots": [], "columns": []}).encode()
    (package / "configuration.json").write_bytes(config)
    config_digest = hashlib.sha256(config).hexdigest()
    manifest = {
        "artifact_inventory": [
            {"path": "configuration.json", "bytes": len(config), "sha256": config_digest}
        ],
        "configuration_sha256": config_digest,
        "sources": sources,
        "git_commit": "1" * 40,
        "pilots": [],
        "runs": [],
    }
    encoded = json.dumps(manifest).encode()
    (package / "manifest.json").write_bytes(encoded)
    monkeypatch.setattr(copernicus_compare_verify, "ROOT", snapshot)
    monkeypatch.setenv("PATH", "")
    return snapshot, package, hashlib.sha256(encoded).hexdigest()


def test_authenticated_source_snapshot_verifies_without_git_or_commit_history(source_snapshot):
    snapshot, package, digest = source_snapshot
    assert not (snapshot / ".git").exists()

    result = verify_package(package, digest)

    assert result["source_commit"] == "1" * 40
    assert result["verified_files"] == 1
    assert result["verified_pilots"] == 0


@pytest.mark.parametrize(
    "source",
    [
        "benchmarks/copernicus_compare.py",
        "benchmarks/copernicus_compare_run.py",
        "benchmarks/copernicus_pair_reference.py",
    ],
)
def test_changed_source_snapshot_is_rejected_even_without_git(source_snapshot, source):
    snapshot, package, digest = source_snapshot
    (snapshot / source).write_bytes(b"# changed source\n")

    with pytest.raises(ValueError, match="Replay source differs"):
        verify_package(package, digest)
