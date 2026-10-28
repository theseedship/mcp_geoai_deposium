"""Hermetic regression of the offline export, using synthetic pinned ZIPs."""

import copy
import hashlib
import importlib.util
import json
import struct
import sys
from datetime import date, timedelta
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import duckdb
import pytest

REPO = Path(__file__).resolve().parents[1]
PANEL = REPO / "tests/fixtures/b5a-replacement-20261005"


def encoded(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def digest(value):
    return hashlib.sha256(value).hexdigest()


@pytest.fixture(scope="module")
def preparation():
    spec = importlib.util.spec_from_file_location(
        "offline_b5a_preparation", REPO / "scripts/prepare_b5a_replacement_inputs.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def frozen_inputs(tmp_path_factory, preparation):
    root = tmp_path_factory.mktemp("b5a-fixtures")
    sources = root / "sources"
    panel_dir = root / "panel"
    sources.mkdir()
    panel_dir.mkdir()
    template = json.loads((PANEL / "gfs-panel.json").read_bytes())
    zones = template["zones"]
    variables = template["variables"]
    assert len(zones) == 15 and len(variables) == 6
    spatial = json.loads((PANEL / "sampling-points.json").read_bytes())
    spatial["current_geometry_midpoint_discrepancies"] = []
    spatial["source_releases"] = {}
    spatial["sampling_points"] = []
    specs = []
    archive_members = {"holdout": {}, "wmo": {}}
    objects = []
    conn = duckdb.connect(":memory:")
    for index, zone in enumerate(zones):
        lat, lon = 10.0 + index, 20.0 + index
        geometry = struct.pack("<BI2d", 1, 1, lon, lat)
        zone_sources = []
        for kind, role, start, count, member in [
            (
                "holdout",
                "holdout",
                "2021-01-01",
                1826,
                f"objects/climate/holdout/era5/2021_2025/{zone}.parquet",
            ),
            ("wmo", "era5", "1991-01-01", 10958, f"climate/era5/{zone}.parquet"),
            ("wmo", "normals", None, 72, f"climate/normals/{zone}.parquet"),
        ]:
            path = root / f"{kind}-{role}-{zone}.parquet"
            if role != "normals":
                values = ", ".join(
                    f"CAST({index + variable_index + 1} AS DOUBLE) AS {variable}"
                    for variable_index, variable in enumerate(variables)
                )
                conn.execute(
                    f"CREATE OR REPLACE TABLE data AS SELECT ? AS zone_id, ? AS lat, "
                    f"? AS lon, ? AS geometry, CAST(DATE '{start}' + i * INTERVAL 1 DAY "
                    f"AS DATE) AS date, {values} FROM range({count}) t(i)",
                    [zone, lat, lon, geometry],
                )
            else:
                conn.execute(
                    "CREATE OR REPLACE TABLE data (zone_id VARCHAR, lat DOUBLE, lon DOUBLE, "
                    "geometry BLOB, variable VARCHAR, month INTEGER, p50 DOUBLE, "
                    "p50_wet DOUBLE, window_start VARCHAR, window_end VARCHAR, window_label VARCHAR)"
                )
                conn.executemany(
                    "INSERT INTO data VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    [
                        (
                            zone,
                            lat,
                            lon,
                            geometry,
                            variable,
                            month,
                            100 + index + variable_index + month / 100,
                            -999.0,
                            "1991-01-01",
                            "2020-12-31",
                            "WMO_1991_2020",
                        )
                        for variable_index, variable in enumerate(variables)
                        for month in range(1, 13)
                    ],
                )
            conn.execute("COPY data TO ? (FORMAT PARQUET)", [str(path)])
            data = path.read_bytes()
            archive_members[kind][member] = data
            record = {
                "release": kind,
                "member": member,
                "logical_s3_key": member,
                "sha256": digest(data),
                "bytes": len(data),
                "rows": count,
            }
            if kind == "holdout":
                record["source_s3_last_modified"] = "2026-06-27T00:00:00Z"
                objects.append(
                    {
                        "bundle_path": member,
                        "logical_s3_key": member,
                        "sha256": digest(data),
                        "bytes": len(data),
                        "source_s3_last_modified": record["source_s3_last_modified"],
                    }
                )
            zone_sources.append(record)
        spatial["sampling_points"].append(
            {
                "zone_id": zone,
                "latitude": lat,
                "longitude": lon,
                "geometry_wkb_hex": geometry.hex(),
                "agreement": "exact_float_and_wkb_across_holdout_wmo_era5_wmo_normals",
                "sources": zone_sources,
            }
        )
    conn.close()
    manifest = encoded({"expected": {"zones": zones}, "objects": objects})
    archive_members["holdout"]["bundle-manifest.json"] = manifest
    archive_members["wmo"]["manifest.json"] = encoded({"fixture": "wmo-manifest"})
    archive_members["wmo"]["bundle.json"] = encoded({"fixture": "wmo-bundle"})
    spatial["holdout_bundle_manifest_sha256"] = digest(manifest)
    spatial["wmo_run_manifest_sha256"] = digest(archive_members["wmo"]["manifest.json"])
    spatial["wmo_bundle_manifest_sha256"] = digest(archive_members["wmo"]["bundle.json"])
    for kind, name, tag, _pin in preparation.RELEASE_SPECS:
        archive = sources / name
        with ZipFile(archive, "w", compression=ZIP_DEFLATED) as handle:
            for member, data in archive_members[kind].items():
                handle.writestr(member, data)
        data = archive.read_bytes()
        specs.append((kind, name, tag, digest(data)))
        spatial["source_releases"][kind] = {
            "release_url": f"https://github.com/theseedship/deposium_geoai/releases/tag/{tag}",
            "asset": name,
            "sha256": digest(data),
            "bytes": len(data),
        }
    sampling = encoded(spatial)
    (panel_dir / "sampling-points.json").write_bytes(sampling)
    for provider in ("gfs", "aifs"):
        panel = json.loads((PANEL / f"{provider}-panel.json").read_bytes())
        panel["configuration"]["sampling_points_sha256"] = digest(sampling)
        (panel_dir / f"{provider}-panel.json").write_bytes(encoded(panel))
    return {
        "sources": sources,
        "panel": panel_dir,
        "specs": specs,
        "zones": zones,
        "variables": variables,
    }


def run(preparation, inputs, output, monkeypatch):
    monkeypatch.setattr(preparation, "RELEASE_SPECS", inputs["specs"])
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prepare",
            "--source-dir",
            str(inputs["sources"]),
            "--panel-dir",
            str(inputs["panel"]),
            "--output-dir",
            str(output),
        ],
    )
    return preparation.main()


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_export_values_keys_receipt_and_reproduction(
    preparation, frozen_inputs, tmp_path, monkeypatch
):
    first, second = tmp_path / "first", tmp_path / "second"
    assert run(preparation, frozen_inputs, first, monkeypatch) == 0
    assert run(preparation, frozen_inputs, second, monkeypatch) == 0
    reference = read_jsonl(first / "reference.jsonl")
    baseline = read_jsonl(first / "p50.jsonl")
    days = [(date(2025, 2, 25) + timedelta(days=i)).isoformat() for i in range(239)]
    expected = {
        (zone, day, variable): float(zone_index + variable_index + 1)
        for zone_index, zone in enumerate(frozen_inputs["zones"])
        for day in days
        for variable_index, variable in enumerate(frozen_inputs["variables"])
    }
    actual = {(r["zone_id"], r["valid_date"], r["variable"]): r["value"] for r in reference}
    assert len(reference) == 21510 and actual == expected
    expected_p50 = {
        (zone, variable, month): 100 + zone_index + variable_index + month / 100
        for zone_index, zone in enumerate(frozen_inputs["zones"])
        for variable_index, variable in enumerate(frozen_inputs["variables"])
        for month in range(1, 13)
    }
    assert len(baseline) == 1080
    assert {(r["zone_id"], r["variable"], r["month"]): r["p50"] for r in baseline} == expected_p50
    receipt = json.loads((first / "preparation-receipt.json").read_bytes())
    assert receipt["counts"]["expected_pairs_per_provider"] == 86040
    assert receipt["checks"]["forecast_rows_collected"] is False
    for name in ("reference.jsonl", "p50.jsonl", "preparation-receipt.json"):
        assert (first / name).read_bytes() == (second / name).read_bytes()
    for key, name in (("reference", "reference.jsonl"), ("baseline_p50", "p50.jsonl")):
        assert receipt["artifacts"][key]["sha256"] == digest((first / name).read_bytes())


@pytest.mark.parametrize("corruption", ["archive", "frozen_panel"])
def test_refuses_corruption_before_output(
    preparation, frozen_inputs, tmp_path, monkeypatch, corruption
):
    import shutil

    inputs = copy.copy(frozen_inputs)
    if corruption == "archive":
        sources = tmp_path / "sources"
        shutil.copytree(inputs["sources"], sources)
        (sources / inputs["specs"][0][1]).write_bytes(b"corrupt ZIP")
        inputs["sources"] = sources
    else:
        panel = tmp_path / "panel"
        shutil.copytree(inputs["panel"], panel)
        value = json.loads((panel / "gfs-panel.json").read_bytes())
        value["horizons_days"] = [1]
        (panel / "gfs-panel.json").write_bytes(encoded(value))
        inputs["panel"] = panel
    output = tmp_path / "output"
    with pytest.raises(ValueError):
        run(preparation, inputs, output, monkeypatch)
    assert not output.exists()


def test_existing_directory_is_never_overwritten(preparation, frozen_inputs, tmp_path, monkeypatch):
    output = tmp_path / "existing"
    output.mkdir()
    sentinel = output / "reference.jsonl"
    sentinel.write_bytes(b"retained user data")
    with pytest.raises(ValueError, match="already exists"):
        run(preparation, frozen_inputs, output, monkeypatch)
    assert sentinel.read_bytes() == b"retained user data"
    assert list(output.iterdir()) == [sentinel]


def test_publication_failure_retains_partial_and_requires_fresh_retry(
    preparation, frozen_inputs, tmp_path, monkeypatch, capsys
):
    output = tmp_path / "partial"
    original_open = Path.open

    def fail_p50(path, mode="r", *args, **kwargs):
        if path == output / "p50.jsonl" and mode == "xb":
            raise OSError("fixture publication failure")
        return original_open(path, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", fail_p50)
    assert run(preparation, frozen_inputs, output, monkeypatch) == 1
    message = json.loads(capsys.readouterr().err)
    assert message["status"] == "incomplete"
    assert message["output_directory"] == str(output)
    assert "new --output-dir" in message["recovery"]
    original = (output / "reference.jsonl").read_bytes()
    assert not (output / "p50.jsonl").exists()
    assert not (output / "preparation-receipt.json").exists()
    with pytest.raises(ValueError, match="already exists"):
        run(preparation, frozen_inputs, output, monkeypatch)
    assert (output / "reference.jsonl").read_bytes() == original
    assert run(preparation, frozen_inputs, tmp_path / "retry", monkeypatch) == 0


def test_verified_archive_snapshot_survives_source_replacement(
    preparation, frozen_inputs, tmp_path, monkeypatch
):
    import shutil

    inputs = copy.copy(frozen_inputs)
    sources = tmp_path / "sources"
    shutil.copytree(inputs["sources"], sources)
    inputs["sources"] = sources
    archive = sources / inputs["specs"][0][1]
    original_read = Path.read_bytes
    replacements = []

    def replace_after_read(path):
        data = original_read(path)
        if path == archive and not replacements:
            archive.write_bytes(b"replaced after verified snapshot")
            replacements.append(True)
        return data

    monkeypatch.setattr(Path, "read_bytes", replace_after_read)
    output = tmp_path / "output"
    assert run(preparation, inputs, output, monkeypatch) == 0
    assert replacements == [True]
    assert len(read_jsonl(output / "reference.jsonl")) == 21510
    receipt = json.loads((output / "preparation-receipt.json").read_bytes())
    assert receipt["source_releases"]["holdout"]["sha256"] == inputs["specs"][0][3]


def test_refuses_cross_source_sampling_disagreement(
    preparation, frozen_inputs, tmp_path, monkeypatch
):
    import shutil

    inputs = copy.copy(frozen_inputs)
    sources = tmp_path / "sources"
    shutil.copytree(inputs["sources"], sources)
    inputs["sources"] = sources
    inputs["specs"] = list(inputs["specs"])
    kind, name, tag, _pin = inputs["specs"][1]
    archive = sources / name
    member = f"climate/normals/{inputs['zones'][0]}.parquet"
    with ZipFile(archive) as handle:
        members = {info.filename: handle.read(info.filename) for info in handle.infolist()}
    parquet = tmp_path / "normals.parquet"
    parquet.write_bytes(members[member])
    conn = duckdb.connect(":memory:")
    conn.execute("CREATE TABLE normals AS SELECT * FROM read_parquet(?)", [str(parquet)])
    # A coherent new point is still incompatible with the retained holdout/WMO point.
    conn.execute("UPDATE normals SET lat = 42, geometry = ?", [struct.pack("<BI2d", 1, 1, 20, 42)])
    changed = tmp_path / "changed.parquet"
    conn.execute("COPY normals TO ? (FORMAT PARQUET)", [str(changed)])
    conn.close()
    members[member] = changed.read_bytes()
    with ZipFile(archive, "w", compression=ZIP_DEFLATED) as handle:
        for member_name, data in members.items():
            handle.writestr(member_name, data)
    inputs["specs"][1] = (kind, name, tag, digest(archive.read_bytes()))
    output = tmp_path / "output"
    with pytest.raises(ValueError, match="sampling disagreement"):
        run(preparation, inputs, output, monkeypatch)
    assert not output.exists()
