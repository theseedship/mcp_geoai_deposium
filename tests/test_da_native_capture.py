"""Pin capture input identity at the loader and evidence handover boundaries."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import duckdb
import pytest

ROOT = Path(__file__).resolve().parents[1]
CAPTURE_SOURCE = ROOT / "tests/fixtures/pattern-library-2026-10-15/capture_native.py"


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(value, sort_keys=True) + "\n").encode()
    path.write_bytes(data)
    return _sha(data)


def _parquet(path, ndvi):
    with duckdb.connect(config={"threads": "1"}) as connection:
        connection.execute(
            "CREATE TABLE pilot AS SELECT '2021-01-0' || i::VARCHAR AS date, "
            "? + i / 10.0 AS ndvi_mean, i::DOUBLE AS temperature_2m, "
            "i::DOUBLE AS precipitation, i::DOUBLE AS humidity, "
            "i::DOUBLE AS evapotranspiration, i::DOUBLE AS wind_speed, "
            "i::DOUBLE AS solar_radiation FROM range(1, 4) t(i)",
            [ndvi],
        )
        connection.execute("COPY pilot TO ? (FORMAT PARQUET)", [str(path)])


@pytest.fixture
def capture_case(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("da_native_capture", CAPTURE_SOURCE)
    capture = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, capture)
    spec.loader.exec_module(capture)
    pilots, study = tmp_path / "pilots", tmp_path / "study"
    pilots.mkdir()
    study.mkdir()
    pilot = pilots / "bassin_du_lez.parquet"
    _parquet(pilot, 0.0)
    provenance_sha = _json(pilots / "provenance.json", {"status": "complete"})
    output_hashes = {
        name: _json(study / name, {"experiment": kind, "lag_steps": 2, "status": "fail"})
        for kind, name in (("spatial", "spatial/result.json"), ("temporal", "temporal/result.json"))
    }
    study_sha = _json(
        study / "manifest.json",
        {
            "schema_version": "geoai-heldout-research/v1",
            "status": "complete",
            "pilot_provenance": {"sha256": provenance_sha},
            "pilot_inputs": [
                {
                    "zone_id": "bassin_du_lez",
                    "sha256": _sha(pilot.read_bytes()),
                    "bytes": pilot.stat().st_size,
                    "min_date": "2021-01-01",
                    "max_date": "2021-01-03",
                    "rows": 3,
                }
            ],
            "output_hashes": output_hashes,
        },
    )
    output = tmp_path / "capture"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(CAPTURE_SOURCE),
            "--pilots",
            str(pilots),
            "--study",
            str(study),
            "--expected-study-manifest-sha256",
            study_sha,
            "--output-dir",
            str(output),
        ],
    )
    monkeypatch.setattr(capture, "version", lambda _name: "1.1.2")
    monkeypatch.setattr(capture.subprocess, "check_output", lambda *_a, **_kw: "a" * 40 + "\n")
    mined = []

    def mine(frame, **_kwargs):
        mined.append(frame["ndvi_mean"].tolist())
        return {"patterns": []}

    monkeypatch.setattr(capture, "mine_gradual_patterns", mine)
    monkeypatch.setattr(capture, "mine_temporal_patterns", mine)
    return capture, pilots, study, output, mined


def test_capture_mines_the_authenticated_pilot_snapshot_after_source_replacement(
    capture_case, tmp_path, monkeypatch
):
    capture, pilots, _, output, mined = capture_case
    original = pilots / "bassin_du_lez.parquet"
    expected_sha = _sha(original.read_bytes())
    replacement = tmp_path / "replacement.parquet"
    _parquet(replacement, 0.6)
    real_loader = capture.load_dataframe

    def replace_before_load(source, **kwargs):
        original.write_bytes(replacement.read_bytes())
        return real_loader(source, **kwargs)

    monkeypatch.setattr(capture, "load_dataframe", replace_before_load)
    capture.main()
    assert mined == [[0.1, 0.2, 0.3], [0.1, 0.2, 0.3]]
    config = json.loads((output / "export.json").read_text())
    identity = config["inputs"][0]
    assert identity["sha256"] == expected_sha
    assert _sha((output / identity["path"]).read_bytes()) == expected_sha


@pytest.mark.parametrize(
    "replaced_name", ["manifest.json", "spatial/result.json", "temporal/result.json"]
)
def test_capture_keeps_authenticated_study_bytes_when_source_changes_during_mining(
    capture_case, monkeypatch, replaced_name
):
    capture, _, study, output, _ = capture_case
    original = (study / replaced_name).read_bytes()
    real_mine = capture.mine_gradual_patterns

    def replace_during_mining(frame, **kwargs):
        _json(study / replaced_name, {"replaced": True, "lag_steps": 30})
        return real_mine(frame, **kwargs)

    monkeypatch.setattr(capture, "mine_gradual_patterns", replace_during_mining)
    capture.main()
    config = json.loads((output / "export.json").read_text())
    reference = next(
        item for item in config["evidence"] if item["source_uri"].endswith("/" + replaced_name)
    )
    assert reference["sha256"] == _sha(original)
    assert reference["relationship"] == "reference_only"
    assert (output / reference["path"]).read_bytes() == original


@pytest.mark.parametrize("source", ["pilot", "provenance", "study_output"])
def test_capture_refuses_source_hash_mismatches_without_publishing(
    capture_case, monkeypatch, source
):
    capture, pilots, study, output, _ = capture_case
    path = {
        "pilot": pilots / "bassin_du_lez.parquet",
        "provenance": pilots / "provenance.json",
        "study_output": study / "spatial/result.json",
    }[source]
    path.write_bytes(b"changed")

    def must_not_mine(*_args, **_kwargs):
        pytest.fail("Mining ran before all capture inputs were authenticated")

    monkeypatch.setattr(capture, "mine_gradual_patterns", must_not_mine)
    with pytest.raises(ValueError, match="SHA-256"):
        capture.main()
    assert not output.exists()


def test_delivered_receipt_resolves_the_original_capture_script():
    folder = CAPTURE_SOURCE.parent
    receipt = json.loads((folder / "verification.json").read_text())
    assert (
        _sha((folder / receipt["capture_script_path"]).read_bytes())
        == receipt["capture_script_sha256"]
    )
    assert (
        receipt["capture_script_sha256"]
        == "7e784175cd21d8c8f8b567b8daf8a401ea2405acc7d2b7719a7bad8e010b97c3"
    )
    fixed = receipt["fixed_capture_script"]
    assert fixed["used_for_delivered_capture"] is False
    assert _sha((folder / fixed["path"]).read_bytes()) == fixed["sha256"]


@pytest.mark.parametrize("limit", ["MAX_JSON_BYTES", "MAX_PILOT_BYTES"])
def test_capture_applies_byte_limits_before_publishing(capture_case, monkeypatch, limit):
    capture, _, _, output, mined = capture_case
    monkeypatch.setattr(capture, limit, 4)
    with pytest.raises(ValueError, match="byte limit"):
        capture.main()
    assert not output.exists()
    assert mined == []


def test_retained_capture_snapshots_feed_the_real_exporter_after_sources_change(
    capture_case, tmp_path
):
    from src.patterns.library_handoff import export_library_handoff, verify_library_handoff

    capture, pilots, study, output, _ = capture_case
    original_sha = _sha((pilots / "bassin_du_lez.parquet").read_bytes())
    capture.main()
    (pilots / "bassin_du_lez.parquet").write_bytes(b"replaced after capture")
    (study / "manifest.json").write_bytes(b"replaced after capture")
    (study / "temporal/result.json").write_bytes(b"replaced after capture")
    package = tmp_path / "package"
    manifest_sha = export_library_handoff(output / "export.json", package)
    manifest, libraries = verify_library_handoff(package, manifest_sha)
    assert manifest.inputs[0].sha256 == original_sha
    assert len(libraries) == 2
    assert all(not library.patterns for library in libraries)
    assert all(item.relationship == "reference_only" for item in manifest.evidence)


def test_failed_mining_does_not_publish_a_partial_capture(capture_case, monkeypatch):
    capture, _, _, output, _ = capture_case

    def fail_mining(*_args, **_kwargs):
        raise RuntimeError("fixture miner failure")

    monkeypatch.setattr(capture, "mine_gradual_patterns", fail_mining)
    with pytest.raises(RuntimeError, match="fixture miner failure"):
        capture.main()
    assert not output.exists()
