import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from scripts import o3_heldout_validation as runner
from src.patterns.heldout_validation import CLIMATE, candidate_universe, spatial_validation


def test_cli_default_confidence_bar_is_one_half():
    args = runner.build_parser().parse_args(
        ["--q3-zones-geojson", "zones.json", "--output-dir", "output"]
    )
    assert args.min_confidence == 0.5


def test_q3_directory_requires_pinned_manifest_before_parquet_read(tmp_path):
    (tmp_path / "bundle.json").write_text("{}")
    with pytest.raises(ValueError, match="SHA-256"):
        runner.snapshot_q3(tmp_path, "0" * 64, tmp_path / "snapshot.zip")


def test_output_manifest_hashes_every_table_and_strict_nulls(tmp_path):
    data = pd.DataFrame(
        {
            "date": pd.date_range("2023-07-01", periods=3),
            "ndvi_mean": [1.0, 2.0, 3.0],
            **{c: [1.0, 1.0, 1.0] for c in CLIMATE},
        }
    )
    result = spatial_validation(
        {str(i): data for i in range(5)}, [[str(i)] for i in range(5)], candidate_universe()
    )
    manifest = runner.write_outputs(
        tmp_path / "out", {"spatial": result}, {"inputs": [], "parameters": {}}
    )
    assert manifest["status"] == "complete"
    assert "spatial/candidates.json" in manifest["output_hashes"]
    for name, digest in manifest["output_hashes"].items():
        assert hashlib.sha256((tmp_path / "out" / name).read_bytes()).hexdigest() == digest
    assert json.loads((tmp_path / "out" / "spatial/candidates.json").read_text()) == []
    assert (tmp_path / "out" / "spatial/fold_scores.csv").read_text().startswith("rule_id,")
    with pytest.raises(FileExistsError):
        runner.write_outputs(tmp_path / "out", {}, {})


def test_artifact_bytes_reproducible_with_same_inputs(tmp_path):
    result = {
        "candidates": [],
        "fold_scores": [{"rule_id": "r", "confidence": None, "status": "non_evaluable"}],
    }
    first = runner.write_outputs(tmp_path / "one", {"spatial": result}, {})
    second = runner.write_outputs(tmp_path / "two", {"spatial": result}, {})
    assert first["output_hashes"] == second["output_hashes"]


def test_nonfinite_serialization_revokes_complete_manifest(tmp_path):
    with pytest.raises(ValueError):
        runner.write_outputs(
            tmp_path / "out", {"spatial": {"scores": [{"confidence": float("nan")}]}}, {}
        )
    manifest = json.loads((tmp_path / "out/manifest.json").read_text())
    assert manifest["status"] == "failed" and "output_hashes" not in manifest


def test_pilot_provenance_must_match_loaded_parquet_hashes():
    provenance = {
        "schema_version": "geoai-heldout-pilot-inputs/v1",
        "status": "complete",
        "outputs": [{"zone_id": z, "sha256": "a" * 64} for z in runner.PILOTS],
    }
    inventory = [{"zone_id": z, "sha256": "b" * 64} for z in runner.PILOTS]
    with pytest.raises(ValueError, match="provenance"):
        runner.verify_pilot_provenance(provenance, inventory)


def test_pilot_loading_is_local_and_hashes_exact_bytes(tmp_path):
    data = pd.DataFrame(
        {
            "date": pd.to_datetime(["2021-01-01", "2023-01-01"]),
            "ndvi_mean": [1.0, 2.0],
            **{c: [1.0, 2.0] for c in CLIMATE},
        }
    )
    import duckdb

    with duckdb.connect() as conn:
        for z in runner.PILOTS:
            conn.register("input_table", data)
            conn.execute("COPY input_table TO ? (FORMAT PARQUET)", [str(tmp_path / f"{z}.parquet")])
    frames, inventory = runner.load_pilots(str(tmp_path / "{zone_id}.parquet"))
    assert set(frames) == set(runner.PILOTS)
    assert len(inventory) == 3
    for entry in inventory:
        assert entry["sha256"] == hashlib.sha256(Path(entry["path"]).read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="local"):
        runner.load_pilots("s3://bucket/{zone_id}.parquet")
