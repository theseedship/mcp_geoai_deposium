"""D-A: immutable library identity, honest evidence, and the real alert read seam."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path

import pytest

from src.patterns.library_schema import PatternLibrary
from src.patterns.offline_bundle import _read_library

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DELIVERED_PACKAGE = _REPO_ROOT / "src/data/pattern-library-2026-10-15/package"
_DELIVERED_MANIFEST_SHA256 = "bfed1a04a176e9fba288fa822b3a02a3f280591c8fd9c4942bb245854cfa47e2"


def test_delivered_package_preserves_native_inventory_and_reference_only_evidence():
    from src.patterns.library_handoff import verify_library_handoff

    manifest, libraries = verify_library_handoff(_DELIVERED_PACKAGE, _DELIVERED_MANIFEST_SHA256)
    assert manifest.library_version == "2026.10.15-native-pilots.1"
    assert len(manifest.files) == 17
    expected_counts = {
        ("bassin_du_lez", "graank"): 6,
        ("bassin_du_lez", "tgrad"): 135,
        ("fos_sur_mer", "graank"): 10,
        ("fos_sur_mer", "tgrad"): 152,
        ("montpellier_metropole", "graank"): 7,
        ("montpellier_metropole", "tgrad"): 108,
    }
    assert len(libraries) == 6
    assert {(lib.zone_id, lib.algorithm): len(lib.patterns) for lib in libraries} == expected_counts
    records = [record for library in libraries for record in library.patterns]
    assert len(records) == 418
    assert all(record.confidence is None and record.n_folds is None for record in records)
    assert all(record.stability is None and record.zones_validated == [] for record in records)
    assert sum((record.extra_metrics or {}).get("conf", 0) > 1 for record in records) == 244
    assert all(entry.qualification == "unqualified" for entry in manifest.libraries)
    assert all(entry.confidence_method is None for entry in manifest.libraries)
    assert {
        (area.zone_id, area.algorithm): area.rule_count for area in manifest.areas
    } == expected_counts
    assert all(area.status == "unqualified" for area in manifest.areas)

    # These are the original 0.70 study bytes, not the exploratory 0.50 study
    # or CV/calibration metadata for the separately mined native records.
    expected_evidence = {
        "manifest.json": "3f57ec7cf52afe071afe14f7daee6d8a640b9d2b75ec2e0eaae8fe36242f481a",
        "spatial/result.json": "c6d1479724821b3f91a835405dd03b0782279861b5294b0f293ade4cb372844b",
        "temporal/result.json": "518ec52245ba213c9b0bdca9c1930db792ed98af724e4198f408aac7f4561b3d",
    }
    original_root = _REPO_ROOT / "tests/fixtures/heldout-validation-2026-10-13"
    source_prefix = (
        "git:745d8929c559ef80112dbc127f52f0fceab5dfe1:"
        "docs/research/heldout-validation-2026-10-13/evidence/"
    )
    assert len(manifest.evidence) == 3
    assert {entry.source_uri: entry.sha256 for entry in manifest.evidence} == {
        source_prefix + name: digest for name, digest in expected_evidence.items()
    }
    for entry in manifest.evidence:
        assert entry.relationship == "reference_only"
        source_name = entry.source_uri.removeprefix(source_prefix)
        assert (_DELIVERED_PACKAGE / entry.package_path).read_bytes() == (
            original_root / source_name
        ).read_bytes()
    study = json.loads((_DELIVERED_PACKAGE / "validation/reference-000.json").read_bytes())
    assert study["parameters"]["min_confidence"] == 0.7
    assert study["parameters"]["n_folds"] == 5
    assert study["parameters"]["temporal_lag_steps"] == [0, 1, 2]


def test_delivered_package_refuses_byte_tampering(tmp_path):
    from src.patterns.library_handoff import verify_library_handoff

    copied = tmp_path / "delivered-package"
    shutil.copytree(_DELIVERED_PACKAGE, copied)
    verify_library_handoff(copied, _DELIVERED_MANIFEST_SHA256)
    library = copied / "libraries/bassin_du_lez/tgrad.json"
    original = library.read_bytes()
    assert original.endswith(b"\n")
    library.write_bytes(original[:-1] + b" ")
    with pytest.raises(ValueError, match="SHA-256"):
        verify_library_handoff(copied, _DELIVERED_MANIFEST_SHA256)


def _json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _cli(*args):
    return subprocess.run(
        [sys.executable, "-m", "src.cli.pattern_library_handoff", *map(str, args)],
        capture_output=True,
        text=True,
        encoding="utf-8",
    )


@pytest.fixture
def source_package(tmp_path):
    source = tmp_path / "native.json"
    library = PatternLibrary.model_validate(
        {
            "schema_version": "1.0.0",
            "zone_id": "lez",
            "algorithm": "tgrad",
            "generated_at": "2026-10-14T12:00:00Z",
            "patterns": [
                {
                    "pattern_id": 7,
                    "antecedent": "(temperature_2m-) -12.0 hours",
                    "consequent": "ndvi_mean+",
                    "pattern": "ndvi_mean+, (temperature_2m-) -12.0 hours",
                    "variables": ["ndvi_mean", "temperature_2m"],
                    "support": 0.75,
                    "confidence": None,
                    "is_temporal": True,
                    "lag_days": -0.5,
                    "algorithm": "tgrad",
                    "zone_id": "lez",
                    "created_at": "2026-10-13T01:02:03Z",
                    "extra_metrics": {"conf": 1.25},
                }
            ],
        }
    )
    source_sha = _json(source, library.model_dump(mode="json"))
    data = tmp_path / "pilot.csv"
    data.write_bytes(b"date,ndvi_mean,temperature_2m\n2021-02-11,0.3,12\n2024-12-27,0.4,11\n")
    evidence = tmp_path / "heldout.json"
    evidence_sha = _json(
        evidence,
        {"producer": "exhaustive_rule_scorer", "lag_steps": 2, "status": "fail"},
    )
    config = {
        "schema_version": "geoai-pattern-library-export/v1",
        "library_version": "2026.10.15-test.1",
        "producer": {
            "name": "so4gp",
            "version": "1.1.2",
            "commit": "a" * 40,
            "command": "python -m src.cli.mine_patterns --algorithm tgrad ...",
        },
        "areas": ["lez", "fos"],
        "inputs": [
            {
                "input_id": "pilot-lez",
                "path": "pilot.csv",
                "source_uri": "fixture:pilot-lez",
                "sha256": hashlib.sha256(data.read_bytes()).hexdigest(),
                "zone_id": "lez",
                "period_start": "2021-02-11",
                "period_end": "2024-12-27",
            }
        ],
        "libraries": [{"path": "native.json", "sha256": source_sha, "input_ids": ["pilot-lez"]}],
        "evidence": [
            {
                "path": "heldout.json",
                "source_uri": "repo:accepted-report-290",
                "sha256": evidence_sha,
                "relationship": "reference_only",
            }
        ],
    }
    config_path = tmp_path / "export.json"
    _json(config_path, config)
    return config_path, config, library


def _export(source_package, tmp_path):
    config_path, _, _ = source_package
    output = tmp_path / "handoff"
    result = _cli("export", "--config", config_path, "--output-dir", output)
    assert result.returncode == 0, result.stderr
    receipt = json.loads(result.stdout)
    return output, receipt["manifest_sha256"]


def test_export_preserves_native_identity_and_separates_report_evidence(source_package, tmp_path):
    output, digest = _export(source_package, tmp_path)
    manifest = json.loads((output / "manifest.json").read_text())
    assert hashlib.sha256((output / "manifest.json").read_bytes()).hexdigest() == digest
    assert manifest["schema_version"] == "geoai-pattern-library-handoff/v1"
    assert manifest["library_schema_version"] == "1.0.0"
    assert manifest["producer"]["version"] == "1.1.2"
    entry = manifest["libraries"][0]
    assert entry["rule_ids"] == ["lez/tgrad/7"]
    assert entry["input_ids"] == ["pilot-lez"]
    assert entry["qualification"] == "unqualified"
    assert entry["confidence_method"] is None
    restored = PatternLibrary.model_validate_json((output / entry["json_path"]).read_bytes())
    assert restored == source_package[2]
    parquet = _read_library(output / entry["parquet_path"], "tgrad")
    assert parquet.patterns == restored.patterns
    assert parquet.patterns[0].lag_days == -0.5
    assert parquet.patterns[0].n_folds is None
    assert parquet.patterns[0].extra_metrics == {"conf": 1.25}
    assert manifest["areas"] == [
        {"zone_id": "lez", "algorithm": "tgrad", "rule_count": 1, "status": "unqualified"},
        {"zone_id": "fos", "algorithm": "tgrad", "rule_count": 0, "status": "absent"},
    ]
    evidence = manifest["evidence"][0]
    assert evidence["relationship"] == "reference_only"
    assert json.loads((output / evidence["package_path"]).read_text())["lag_steps"] == 2
    assert _cli("verify", output, "--expected-manifest-sha256", digest).returncode == 0


@pytest.mark.parametrize("kind", ["library", "input", "evidence"])
def test_export_refuses_changed_sources_before_creating_output(source_package, tmp_path, kind):
    config_path, config, _ = source_package
    section = {"library": "libraries", "input": "inputs", "evidence": "evidence"}[kind]
    (tmp_path / config[section][0]["path"]).write_bytes(b"tampered")
    result = _cli("export", "--config", config_path, "--output-dir", tmp_path / "handoff")
    assert result.returncode == 2
    assert "SHA-256" in result.stderr
    assert not (tmp_path / "handoff").exists()


@pytest.mark.parametrize("defect", ["scorer", "unknown_schema", "unsafe_zone", "unknown_input"])
def test_export_refuses_unfaithful_or_untraceable_records(source_package, tmp_path, defect):
    config_path, config, library = source_package
    raw = library.model_dump(mode="json")
    if defect == "scorer":
        raw = {"candidates": [{"rule_id": "rule", "lag_steps": 2, "confidence": 0.8}]}
    elif defect == "unknown_schema":
        raw["schema_version"] = "2.0.0"
    elif defect == "unsafe_zone":
        raw["zone_id"] = raw["patterns"][0]["zone_id"] = "../escape"
    else:
        config["libraries"][0]["input_ids"] = ["missing"]
    config["libraries"][0]["sha256"] = _json(tmp_path / "native.json", raw)
    _json(config_path, config)
    result = _cli("export", "--config", config_path, "--output-dir", tmp_path / "handoff")
    assert result.returncode == 2
    assert not (tmp_path / "handoff").exists()
    assert not (tmp_path / "escape").exists()


@pytest.mark.parametrize("defect", ["changed", "missing", "extra", "wrong_anchor", "checksum_list"])
def test_verify_refuses_incomplete_or_changed_package(source_package, tmp_path, defect):
    output, digest = _export(source_package, tmp_path)
    library = output / "libraries/lez/tgrad.json"
    if defect == "changed":
        library.write_bytes(library.read_bytes() + b" ")
    elif defect == "missing":
        library.unlink()
    elif defect == "extra":
        (output / "extra.json").write_text("{}")
    elif defect == "wrong_anchor":
        digest = "0" * 64
    else:
        (output / "SHA256SUMS").write_text("")
    result = _cli("verify", output, "--expected-manifest-sha256", digest)
    assert result.returncode == 2
    assert result.stderr


def test_export_does_not_replace_an_existing_handoff(source_package, tmp_path):
    output, digest = _export(source_package, tmp_path)
    result = _cli("export", "--config", source_package[0], "--output-dir", output)
    assert result.returncode == 2
    assert hashlib.sha256((output / "manifest.json").read_bytes()).hexdigest() == digest


@pytest.mark.parametrize("metrics", [None, {}])
def test_absent_and_empty_native_metrics_survive_both_formats(source_package, tmp_path, metrics):
    from src.patterns import library_handoff

    config_path, config, library = source_package
    library.patterns[0].extra_metrics = metrics
    config["libraries"][0]["sha256"] = _json(
        tmp_path / "native.json", library.model_dump(mode="json")
    )
    _json(config_path, config)
    digest = library_handoff.export_library_handoff(config_path, tmp_path / "handoff")
    _, libraries = library_handoff.verify_library_handoff(tmp_path / "handoff", digest)
    assert libraries[0].patterns[0].extra_metrics == metrics


def test_calibrated_value_requires_its_method_without_granting_acceptance(source_package, tmp_path):
    from src.patterns import library_handoff

    config_path, config, library = source_package
    record = library.patterns[0]
    record.n_folds = 5
    record.zones_validated = ["lez"]
    record.stability = 0.8
    record.confidence = 0.6
    config["libraries"][0]["sha256"] = _json(
        tmp_path / "native.json", library.model_dump(mode="json")
    )
    _json(config_path, config)
    with pytest.raises(ValueError, match="confidence_method"):
        library_handoff.export_library_handoff(config_path, tmp_path / "handoff")
    assert not (tmp_path / "handoff").exists()
    config["libraries"][0]["confidence_method"] = "explicit fixture method, not scientific evidence"
    _json(config_path, config)
    digest = library_handoff.export_library_handoff(config_path, tmp_path / "handoff")
    manifest, libraries = library_handoff.verify_library_handoff(tmp_path / "handoff", digest)
    assert libraries[0].patterns[0].confidence == 0.6
    assert libraries[0].patterns[0].n_folds == 5
    assert manifest.libraries[0].qualification == "unqualified"


def test_export_parses_the_same_native_snapshot_it_authenticates(
    source_package, tmp_path, monkeypatch
):
    from src.patterns import library_handoff

    source = tmp_path / "native.json"
    changed = source_package[2].model_dump(mode="json")
    changed["patterns"][0]["support"] = 0.01
    original_open = Path.open
    replaced = False

    @contextmanager
    def replace_after_read(path, *args, **kwargs):
        nonlocal replaced
        mode = args[0] if args else kwargs.get("mode", "r")
        with original_open(path, *args, **kwargs) as stream:
            yield stream
        if path == source and mode == "rb" and not replaced:
            replaced = True
            _json(source, changed)

    monkeypatch.setattr(Path, "open", replace_after_read)
    digest = library_handoff.export_library_handoff(source_package[0], tmp_path / "handoff")
    _, libraries = library_handoff.verify_library_handoff(tmp_path / "handoff", digest)
    assert libraries[0].patterns[0].support == 0.75
    assert json.loads(source.read_text())["patterns"][0]["support"] == 0.01


def test_verify_returns_the_authenticated_snapshot_after_files_are_replaced(
    source_package, tmp_path, monkeypatch
):
    from src.patterns import library_handoff

    output, digest = _export(source_package, tmp_path)
    config_path, config, library = source_package
    library.patterns[0].support = 0.01
    config["libraries"][0]["sha256"] = _json(
        tmp_path / "native.json", library.model_dump(mode="json")
    )
    _json(config_path, config)
    other = tmp_path / "other"
    library_handoff.export_library_handoff(config_path, other)
    evidence = (output / "validation/reference-000.json").read_bytes()
    original_sha = library_handoff._sha

    def replace_after_hash(data):
        result = original_sha(data)
        if data == evidence:
            for suffix in ("json", "parquet"):
                name = f"libraries/lez/tgrad.{suffix}"
                (output / name).write_bytes((other / name).read_bytes())
        return result

    monkeypatch.setattr(library_handoff, "_sha", replace_after_hash)
    _, libraries = library_handoff.verify_library_handoff(output, digest)
    assert libraries[0].patterns[0].support == 0.75


@pytest.mark.parametrize("empty", [False, True])
@pytest.mark.skipif(
    sys.platform == "win32",
    reason="The existing alert tool imports Linux-only normals I/O via src.tools.__init__",
)
async def test_exported_parquet_reaches_real_alert_reader_without_fabricated_validation(
    source_package, tmp_path, monkeypatch, empty
):
    from src.tools import pattern_alert

    config_path, config, library = source_package
    if empty:
        library.patterns = []
        config["libraries"][0]["sha256"] = _json(
            tmp_path / "native.json", library.model_dump(mode="json")
        )
        _json(config_path, config)
    output, _ = _export(source_package, tmp_path)
    monkeypatch.setattr(pattern_alert.settings, "DUCKDB_PATH", ":memory:")
    monkeypatch.setattr(pattern_alert.settings, "ALLOW_LOCAL_PARQUET_SOURCE", True)
    monkeypatch.setattr(pattern_alert.settings, "LOCAL_PARQUET_SOURCE_ROOT", tmp_path.as_posix())

    class Capture:
        def tool(self, **_kwargs):
            def capture(fn):
                self.call = fn
                return fn

            return capture

    capture = Capture()
    pattern_alert.register_pattern_alert_tools(capture)
    result = await capture.call(
        zone_id="lez",
        patterns_source=(output / "libraries/lez/tgrad.parquet").as_posix(),
        allow_local_source=True,
    )
    if empty:
        assert result["alerts"] == []
        assert result["statistics"]["n_patterns_read"] == 0
    else:
        assert "O3 validated library" in result["error"]
        assert result["metadata"]["confidence_status"] == "experimental_unratified"
