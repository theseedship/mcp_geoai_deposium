"""D-A to MCP projection: preserve native records and publish only complete files."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path
from unittest.mock import Mock

import pytest

from src.patterns import scientific_projection as projection
from src.patterns.library_handoff import verify_library_handoff

PACKAGE = Path(__file__).resolve().parents[1] / "src/data/pattern-library-2026-10-15/package"
MANIFEST_SHA = "bfed1a04a176e9fba288fa822b3a02a3f280591c8fd9c4942bb245854cfa47e2"


@pytest.fixture(scope="module")
def verified():
    return verify_library_handoff(PACKAGE, MANIFEST_SHA)


@pytest.fixture
def authenticated(monkeypatch, verified):
    manifest, libraries = verified
    result = (
        manifest.model_copy(deep=True),
        [library.model_copy(deep=True) for library in libraries],
    )
    verifier = Mock(return_value=result)
    monkeypatch.setattr(projection, "verify_library_handoff", verifier)
    return result, verifier


def test_real_package_projection_preserves_all_records_and_evidence(tmp_path):
    output = tmp_path / "projection.json"
    result = projection.project_library_handoff(PACKAGE, MANIFEST_SHA, "native-pilots", output)
    data = output.read_bytes()
    value = json.loads(data)
    assert result["projection_sha256"] == hashlib.sha256(data).hexdigest()
    assert result["patterns"] == 418 and result["libraries"] == 6
    assert result["package_version"] == "2026.10.15-native-pilots.1"
    assert set(value) == {
        "schema_version",
        "package_id",
        "package_version",
        "kind",
        "source_digests",
        "provenance",
        "libraries",
    }
    manifest = json.loads((PACKAGE / "manifest.json").read_bytes())
    assert value["libraries"] == [
        json.loads((PACKAGE / entry["json_path"]).read_bytes()) for entry in manifest["libraries"]
    ]
    records = [record for library in value["libraries"] for record in library["patterns"]]
    for field in ("confidence", "stability", "n_folds", "period_start", "period_end", "created_at"):
        assert all(record[field] is None for record in records)
    assert all(record["zones_validated"] == [] for record in records)
    assert sum(record["extra_metrics"].get("conf", 0) > 1 for record in records) == 244
    assert any(record["lag_days"] % 1 != 0 for record in records)
    digests = {item["label"]: item["sha256"] for item in value["source_digests"]}
    assert len(digests) == 28
    assert digests["handoff_manifest"] == MANIFEST_SHA
    assert digests["source_config"] == manifest["source_config_sha256"]
    for name, identity in manifest["files"].items():
        assert digests[f"package_file:{name}"] == identity["sha256"]
    for item in manifest["inputs"]:
        assert digests[f"input:{item['input_id']}"] == item["sha256"]
    for item in manifest["libraries"]:
        assert (
            digests[f"source_library:{item['zone_id']}:{item['algorithm']}"]
            == item["source_sha256"]
        )
    assert value["provenance"]["generated_at"] is None
    assert value["provenance"]["period"] == {
        "start_date": "2021-01-24",
        "end_date": "2024-12-29",
    }
    limitations = " ".join(value["provenance"]["limitations"])
    assert "unqualified" in limitations and "reference-only" in limitations
    assert manifest["producer"]["command"] not in data.decode()
    assert "owuor_lab" not in data.decode() and "source_uri" not in value
    assert len(data) <= projection.MAX_PROJECTION_BYTES
    assert output.stat().st_mode & 0o777 == 0o600


def test_projection_uses_only_verified_models_after_source_replacement(
    tmp_path, authenticated, monkeypatch
):
    (manifest, libraries), verifier = authenticated
    copied = tmp_path / "package"
    shutil.copytree(PACKAGE, copied)

    def verify_then_replace(directory, digest):
        assert Path(directory) == copied and digest == MANIFEST_SHA
        shutil.rmtree(copied)
        copied.mkdir()
        (copied / "manifest.json").write_text('{"untrusted":true}')
        return manifest, libraries

    verifier.side_effect = verify_then_replace
    output = tmp_path / "projection.json"
    projection.project_library_handoff(copied, MANIFEST_SHA, "native", output)
    verifier.assert_called_once_with(copied, MANIFEST_SHA)
    assert sum(len(lib["patterns"]) for lib in json.loads(output.read_bytes())["libraries"]) == 418


def test_deterministic_output_and_source_version(tmp_path, authenticated):
    _, verifier = authenticated
    first, second = tmp_path / "first.json", tmp_path / "second.json"
    result = projection.project_library_handoff(PACKAGE, MANIFEST_SHA, "native", first)
    projection.project_library_handoff(
        PACKAGE, MANIFEST_SHA, "native", second, result["package_version"]
    )
    assert first.read_bytes() == second.read_bytes()
    with pytest.raises(ValueError, match="must match"):
        projection.project_library_handoff(
            PACKAGE, MANIFEST_SHA, "native", tmp_path / "bad", "other"
        )
    assert verifier.call_count == 3
    assert not (tmp_path / "bad").exists()


@pytest.mark.parametrize("identifier", ["", "../native", "a b", "a" * 129, "é", "a\n"])
def test_invalid_identifiers_fail_before_verification(tmp_path, authenticated, identifier):
    _, verifier = authenticated
    with pytest.raises(ValueError, match="identifier"):
        projection.project_library_handoff(PACKAGE, MANIFEST_SHA, identifier, tmp_path / "out")
    verifier.assert_not_called()


@pytest.mark.parametrize("digest", ["", "a" * 63, "A" * 64, "g" * 64])
def test_explicit_manifest_digest_required(tmp_path, authenticated, digest):
    _, verifier = authenticated
    with pytest.raises(ValueError, match="SHA-256"):
        projection.project_library_handoff(PACKAGE, digest, "native", tmp_path / "out")
    verifier.assert_not_called()


def test_refuses_existing_dangling_and_inside_package_outputs(tmp_path, authenticated):
    _, verifier = authenticated
    output = tmp_path / "existing"
    output.write_bytes(b"keep")
    with pytest.raises(ValueError, match="already exists"):
        projection.project_library_handoff(PACKAGE, MANIFEST_SHA, "native", output)
    assert output.read_bytes() == b"keep"
    link = tmp_path / "dangling"
    link.symlink_to(tmp_path / "absent")
    with pytest.raises(ValueError, match="already exists"):
        projection.project_library_handoff(PACKAGE, MANIFEST_SHA, "native", link)
    with pytest.raises(ValueError, match="outside"):
        projection.project_library_handoff(PACKAGE, MANIFEST_SHA, "native", PACKAGE / "projection")
    verifier.assert_not_called()


@pytest.mark.parametrize(
    "component",
    ["manifest.json", "libraries/bassin_du_lez/graank.json", "validation/reference-002.json"],
)
def test_corrupted_package_never_creates_projection(tmp_path, component):
    copied = tmp_path / "package"
    shutil.copytree(PACKAGE, copied)
    path = copied / component
    path.write_bytes(path.read_bytes() + b" ")
    output = tmp_path / "out.json"
    with pytest.raises(ValueError, match="SHA-256|length|byte limit"):
        projection.project_library_handoff(copied, MANIFEST_SHA, "native", output)
    assert not output.exists()


@pytest.mark.parametrize(
    "case",
    [
        "digests",
        "libraries",
        "records",
        "variables",
        "blank_variable",
        "bom_variable",
        "mixed_js_whitespace",
        "text",
        "utf16",
        "nonfinite",
        "bytes",
    ],
)
def test_projection_bounds_refuse_without_partial_output(
    tmp_path, authenticated, monkeypatch, case
):
    (manifest, libraries), _ = authenticated
    if case == "digests":
        manifest.inputs = [
            manifest.inputs[0].model_copy(update={"input_id": f"input_{i}"}) for i in range(101)
        ]
    elif case == "libraries":
        libraries.extend([libraries[0]] * 1000)
    elif case == "records":
        libraries[0].patterns.extend([libraries[0].patterns[0]] * 10_000)
    elif case == "variables":
        libraries[0].patterns[0].variables = ["x"] * 101
    elif case == "blank_variable":
        libraries[0].patterns[0].variables = [" \t"]
    elif case == "bom_variable":
        libraries[0].patterns[0].variables = ["\ufeff"]
    elif case == "mixed_js_whitespace":
        libraries[0].patterns[0].variables = ["\u00a0\u200a\ufeff\u3000\t"]
    elif case == "text":
        libraries[0].patterns[0].antecedent = "x" * 4097
    elif case == "utf16":
        libraries[0].patterns[0].antecedent = "🧪" * 2049
    elif case == "nonfinite":
        libraries[0].patterns[0].extra_metrics = {"conf": float("inf")}
    else:
        monkeypatch.setattr(projection, "MAX_PROJECTION_BYTES", 10)
    with pytest.raises(ValueError, match="limit|finite|blank"):
        projection.project_library_handoff(PACKAGE, MANIFEST_SHA, "native", tmp_path / "out")
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("failure", ["fsync", "link"])
def test_atomic_publish_failure_cleans_temporary_file(
    tmp_path, authenticated, monkeypatch, failure
):
    def fail(*args, **kwargs):
        raise OSError("simulated publish failure")

    monkeypatch.setattr(projection.os, failure, fail)
    with pytest.raises(OSError, match="simulated"):
        projection.project_library_handoff(PACKAGE, MANIFEST_SHA, "native", tmp_path / "out")
    assert list(tmp_path.iterdir()) == []


def test_concurrent_output_creation_cannot_be_overwritten(tmp_path, authenticated, monkeypatch):
    output = tmp_path / "out"
    original_link = projection.os.link

    def competing_link(source, destination):
        output.write_bytes(b"concurrent operator output")
        return original_link(source, destination)

    monkeypatch.setattr(projection.os, "link", competing_link)
    with pytest.raises(FileExistsError):
        projection.project_library_handoff(PACKAGE, MANIFEST_SHA, "native", output)
    assert output.read_bytes() == b"concurrent operator output"
    assert list(tmp_path.iterdir()) == [output]


def test_cli_project_success_and_refusal(tmp_path):
    output = tmp_path / "projection.json"
    command = [
        sys.executable,
        "-m",
        "src.cli.pattern_library_handoff",
        "project",
        str(PACKAGE),
        "--expected-manifest-sha256",
        MANIFEST_SHA,
        "--package-id",
        "native",
        "--output",
        str(output),
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["patterns"] == 418
    original = output.read_bytes()
    refusal = subprocess.run(command, capture_output=True, text=True, check=False)
    assert refusal.returncode == 2 and "already exists" in refusal.stderr
    assert output.read_bytes() == original
