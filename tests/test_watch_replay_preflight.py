"""Local synthetic readiness fixtures; never a labelled operational replay."""

import hashlib
import json
import os
import subprocess
import sys
from datetime import date, timedelta
from pathlib import Path
from types import SimpleNamespace

import duckdb
import httpx
import pytest

from src.cli import forecast_archive_panel as archive
from src.cli import watch_replay_preflight as operator
from src.forecast_archive import (
    PREVIOUS_RUNS_HOURLY_UNITS,
    PreviousRunsArchiveClient,
    RawResponse,
    _previous_field,
    normalize_previous_runs,
)

ZONE = "fos_sur_mer"
POINT = (43.445, 4.92)


def _pin(root, name):
    path = root / name
    return {"path": name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def _json(root, name, value):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")
    return _pin(root, name)


def _normal_files(root, point):
    path, source = root / "normals.parquet", root / "era5.parquet"
    with duckdb.connect(":memory:") as conn:
        conn.execute("""CREATE TABLE normals (
            zone_id VARCHAR, variable VARCHAR, month INTEGER, window_label VARCHAR,
            schema_version INTEGER, p10 DOUBLE, p50 DOUBLE, p90 DOUBLE,
            p10_wet DOUBLE, p50_wet DOUBLE, p90_wet DOUBLE, n_wet INTEGER,
            wet_day_threshold_mm DOUBLE, lat DOUBLE, lon DOUBLE, geometry BLOB, n INTEGER)""")
        conn.executemany(
            "INSERT INTO normals VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (
                    ZONE,
                    "precipitation",
                    m,
                    "1991-2020",
                    2,
                    0,
                    0,
                    5,
                    1,
                    2,
                    3,
                    10,
                    1,
                    point[0],
                    point[1],
                    b"synthetic geometry",
                    100,
                )
                for m in range(1, 13)
            ],
        )
        conn.execute("COPY normals TO ? (FORMAT PARQUET)", [str(path)])
        conn.execute(
            "CREATE TABLE source AS SELECT ?::VARCHAR zone_id, ?::DOUBLE lat, ?::DOUBLE lon",
            [ZONE, *POINT],
        )
        conn.execute("COPY source TO ? (FORMAT PARQUET)", [str(source)])
    return {**_pin(root, path.name), "source": _pin(root, source.name)}


def _fixture(
    root,
    *,
    horizons=(1, 3, 5, 7),
    labels=None,
    normals_point=POINT,
    gap=False,
    variable="precipitation",
    hourly_value=0.0,
):
    root.mkdir(parents=True, exist_ok=True)
    points = _json(
        root,
        "sampling.json",
        {"sampling_points": [{"zone_id": ZONE, "latitude": POINT[0], "longitude": POINT[1]}]},
    )
    configuration = {
        "source": "openmeteo_previous_runs",
        "timezone": "UTC",
        "wind_speed_unit": "ms",
        "day_grain": "utc_day",
        "lead_semantics": "fixed_hourly_offset_24_times_horizon_days",
        "panel_id": "synthetic-preflight",
        "sampling_points_sha256": points["sha256"],
    }
    dates = [(date(2025, 6, 1) + timedelta(days=k)).isoformat() for k in range(5)]
    panel = _json(
        root,
        "panel.json",
        {
            "schema_version": "forecast-skill-panel/v1",
            "provider": "gfs",
            "model": "gfs_seamless",
            "configuration": configuration,
            "zones": [ZONE],
            "variables": [variable],
            "horizons_days": list(horizons),
            "valid_dates": dates,
        },
    )
    frozen = archive._load_frozen_panel([root / panel["path"]], root / points["path"])
    identity = frozen.identities[0].panel
    times = [f"{day}T{hour:02d}:00" for day in dates for hour in range(24)]
    hourly, units = {"time": times}, {"time": "iso8601"}
    for horizon in horizons:
        field = _previous_field(variable, horizon)
        hourly[field] = [
            None if gap and horizon == 1 and k < 24 else hourly_value for k in range(len(times))
        ]
        units[field] = PREVIOUS_RUNS_HOURLY_UNITS[variable]
    body = json.dumps(
        {
            "latitude": POINT[0],
            "longitude": POINT[1],
            "utc_offset_seconds": 0,
            "timezone": "GMT",
            "hourly_units": units,
            "hourly": hourly,
        }
    ).encode()
    raw_digest = hashlib.sha256(body).hexdigest()
    raw = RawResponse(body, raw_digest, "2026-10-21T00:00:00Z", {"latitude": POINT[0]})
    directory = root / "archive"
    (directory / "raw").mkdir(parents=True)
    (directory / "raw" / f"{raw_digest}.json").write_bytes(body)
    provenance_name = archive._provenance_name(frozen, ZONE, identity, raw_digest)
    _json(
        directory / "raw",
        provenance_name,
        archive._provenance(
            SimpleNamespace(base_url="https://example.invalid"), frozen, ZONE, identity, raw
        ),
    )
    for horizon in horizons:
        rows, coverage = normalize_previous_runs(
            json.loads(body),
            zone_id=ZONE,
            provider="gfs",
            model="gfs_seamless",
            horizon_days=horizon,
            variables=[variable],
            expected_start_date=dates[0],
            expected_end_date=dates[-1],
        )
        path = directory / f"{archive._artifact_key(ZONE, identity, frozen, horizon)}.json"
        path.write_bytes(
            archive._json_bytes(
                archive._artifact(frozen, ZONE, identity, rows, coverage, raw_digest)
            )
        )
    label_pin = None
    if labels is not None:
        label_pin = _json(
            root,
            "labels.json",
            {
                "schema_version": "watch-labelset/v1",
                "variable": "precipitation",
                "window_start": dates[0],
                "window_end": dates[-1],
                "exhaustive": False,
                "coverage_statement": "synthetic declarations only",
                "labels": [
                    {
                        "schema_version": "watch-label/v1",
                        "event_id": f"synthetic-{k}",
                        "place_ref": f"zone:{ZONE}",
                        "variable": "precipitation",
                        "start": dates[k],
                        "end": dates[k],
                        "timezone": "UTC",
                        "match_tolerance_days": 1,
                        "source": "synthetic fixture",
                        "level": "orange",
                        "provenance_url": "https://example.invalid/label",
                        "licence": "synthetic",
                        "spatial_match": "zone_within_department",
                    }
                    for k in range(labels)
                ],
            },
        )
    descriptor = {
        "schema_version": "geoai-watch-replay-inputs/v1",
        "zone_id": ZONE,
        "panel": panel,
        "sampling_points": points,
        "archive_dir": "archive",
        "normals": _normal_files(root, normals_point),
        "labelset": label_pin,
    }
    _json(root, "inputs.json", descriptor)
    return root / "inputs.json", descriptor


def _run(path):
    return operator.preflight_watch_inputs(path, hashlib.sha256(path.read_bytes()).hexdigest())


def _codes(report):
    return {reason["code"] for reason in report["reasons"]}


def test_four_horizon_inputs_inventory_missing_horizons_and_labels(tmp_path):
    path, _ = _fixture(tmp_path)
    report = _run(path)
    assert report["status"] == "blocked"
    assert report["required_horizons"] == list(range(1, 8))
    assert report["missing_horizons"] == [2, 4, 6]
    assert {"missing_required_horizons", "labelset_absent"} <= _codes(report)
    assert len(report["archives"]) == 4
    assert report["normals"]["rows"] == 12


def test_source_point_mismatch_is_exposed_even_with_identical_zone_id(tmp_path):
    path, _ = _fixture(tmp_path, normals_point=(43.435, 4.91))
    report = _run(path)
    assert {"normals_frozen_point_mismatch", "normals_source_point_mismatch"} <= _codes(report)
    assert report["normals"]["source_point"] == report["frozen_point"]
    assert report["normals"]["declared_point"] != report["frozen_point"]


def test_absent_source_blocks_geometry_lineage(tmp_path):
    path, descriptor = _fixture(tmp_path)
    descriptor["normals"].pop("source")
    _json(tmp_path, path.name, descriptor)
    assert "normals_source_absent" in _codes(_run(path))


@pytest.mark.parametrize("field", ["panel", "sampling_points", "normals", "labelset"])
def test_pinned_input_hash_mismatch_is_refused(tmp_path, field):
    path, descriptor = _fixture(tmp_path, labels=5)
    descriptor[field]["sha256"] = "0" * 64
    _json(tmp_path, path.name, descriptor)
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        _run(path)


def test_descriptor_hash_mismatch_is_refused(tmp_path):
    path, _ = _fixture(tmp_path)
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        operator.preflight_watch_inputs(path, "0" * 64)


def test_provenance_tampering_is_refused_by_existing_verifier(tmp_path):
    path, _ = _fixture(tmp_path)
    provenance = next((tmp_path / "archive/raw").glob("*provenance*.json"))
    document = json.loads(provenance.read_bytes())
    document["provider"] = "other"
    _json(provenance.parent, provenance.name, document)
    with pytest.raises(ValueError, match="another request"):
        _run(path)


def test_missing_archive_artifact_is_aggregated(tmp_path):
    path, _ = _fixture(tmp_path)
    next((tmp_path / "archive").glob("*__3.json")).unlink()
    report = _run(path)
    assert "archive_artifact_absent" in _codes(report)
    assert report["missing_horizons"] == [2, 3, 4, 6]
    assert "labelset_absent" in _codes(report)


def test_empty_labels_are_counted_without_claiming_a_measurement(tmp_path):
    path, _ = _fixture(tmp_path, labels=0)
    report = _run(path)
    assert report["labelset"]["label_count"] == 0
    assert "label_reporting_floor_not_met" in _codes(report)


def test_label_schema_rejects_numeric_boolean_and_main_returns_two(tmp_path, capsys):
    path, descriptor = _fixture(tmp_path, labels=5)
    labels = json.loads((tmp_path / "labels.json").read_bytes())
    labels["exhaustive"] = 1
    descriptor["labelset"] = _json(tmp_path, "labels.json", labels)
    _json(tmp_path, path.name, descriptor)
    assert (
        operator.main(
            ["--inputs", str(path), "--expected-inputs-sha256", _pin(tmp_path, path.name)["sha256"]]
        )
        == 2
    )
    assert json.loads(capsys.readouterr().out)["status"] == "refused"


@pytest.mark.parametrize("gap", [False, True])
def test_zero_rain_is_retained_and_missing_days_are_not_filled(tmp_path, gap):
    path, _ = _fixture(tmp_path, gap=gap)
    report = _run(path)
    first = next(row for row in report["archives"] if row["horizon"] == 1)
    assert first["coverage"]["available_days"] == (4 if gap else 5)
    assert first["coverage"]["status"] == ("partial_archive" if gap else "available")


def test_full_synthetic_input_inventory_calls_no_provider_or_rule(tmp_path, monkeypatch):
    path, _ = _fixture(tmp_path, horizons=tuple(range(1, 8)), labels=5)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("Preflight must not execute providers or weather rules")

    import src.tools.climate as climate
    import src.watch_replay as replay

    monkeypatch.setattr(httpx.Client, "send", forbidden)
    monkeypatch.setattr(httpx.AsyncClient, "send", forbidden)
    monkeypatch.setattr(PreviousRunsArchiveClient, "fetch_daily", forbidden)
    monkeypatch.setattr(PreviousRunsArchiveClient, "fetch_daily_multi", forbidden)
    monkeypatch.setattr(replay, "replay_labelset", forbidden)
    monkeypatch.setattr(climate, "build_watch_series", forbidden)
    report = _run(path)
    assert report["status"] == "inputs_verified"
    assert report["reasons"] == report["missing_horizons"] == []
    assert report["scientific_acceptance"] == "pending"
    assert report["operational_rule_parity"] == "unverified"
    assert report["live_activation"] is False
    assert report["forecast_origin_authentication"] == "not_established_by_hashes"
    assert not any(key in json.dumps(report) for key in ('"hit_rate"', '"score"', '"probability"'))


def test_completed_blocked_inventory_returns_zero_and_output_is_create_only(tmp_path, capsys):
    path, _ = _fixture(tmp_path)
    output = tmp_path / "report.json"
    args = [
        "--inputs",
        str(path),
        "--expected-inputs-sha256",
        _pin(tmp_path, path.name)["sha256"],
        "--output",
        str(output),
    ]
    assert operator.main(args) == 0
    original = output.read_bytes()
    assert json.loads(original)["status"] == "blocked"
    assert operator.main(args) == 2
    assert output.read_bytes() == original
    assert json.loads(capsys.readouterr().out)["status"] == "refused"


@pytest.mark.parametrize("bad_path", ["../panel.json", "/tmp/panel.json", "link.json"])
def test_traversal_absolute_and_symlink_inputs_are_refused(tmp_path, bad_path):
    path, descriptor = _fixture(tmp_path)
    (tmp_path / "link.json").symlink_to(tmp_path / "panel.json")
    descriptor["panel"]["path"] = bad_path
    _json(tmp_path, path.name, descriptor)
    with pytest.raises(ValueError):
        _run(path)


@pytest.mark.parametrize("raw", ['{"a":1,"a":2}', '{"a":NaN}', '{"a":Infinity}'])
def test_nonfinite_and_duplicate_json_keys_are_refused(tmp_path, raw):
    path = tmp_path / "inputs.json"
    path.write_text(raw)
    with pytest.raises(ValueError, match="Invalid JSON"):
        _run(path)


def test_overflowing_json_exponent_is_refused(tmp_path):
    path = tmp_path / "inputs.json"
    path.write_text('{"a":1e309}')
    with pytest.raises(ValueError, match="Nonfinite JSON"):
        _run(path)


def test_future_labels_outside_declared_archive_window_block_readiness(tmp_path):
    path, descriptor = _fixture(tmp_path, horizons=tuple(range(1, 8)), labels=5)
    labels = json.loads((tmp_path / "labels.json").read_bytes())
    for key in ("window_start", "window_end"):
        labels[key] = labels[key].replace("2025", "2030")
    for label in labels["labels"]:
        for key in ("start", "end"):
            label[key] = label[key].replace("2025", "2030")
    descriptor["labelset"] = _json(tmp_path, "labels.json", labels)
    _json(tmp_path, path.name, descriptor)
    report = _run(path)
    assert report["status"] == "blocked"
    assert "label_window_outside_panel" in _codes(report)


def test_verified_archive_with_missing_rain_date_blocks_readiness(tmp_path):
    path, _ = _fixture(tmp_path, horizons=tuple(range(1, 8)), labels=5, gap=True)
    report = _run(path)
    assert report["status"] == "blocked"
    assert {
        "code": "archive_precipitation_dates_missing",
        "horizon": 1,
        "dates": ["2025-06-01"],
    } in report["reasons"]


def test_temperature_only_panel_cannot_supply_a_rain_replay(tmp_path):
    path, _ = _fixture(tmp_path, horizons=tuple(range(1, 8)), labels=5, variable="temperature_2m")
    report = _run(path)
    assert report["status"] == "blocked"
    assert "panel_precipitation_absent" in _codes(report)


def _change_normals(root, descriptor, assignment, values=(), *, prelude=None):
    original = root / descriptor["normals"]["path"]
    replacement = root / "replacement.parquet"
    with duckdb.connect(":memory:") as conn:
        conn.execute("CREATE TABLE n AS SELECT * FROM read_parquet(?)", [str(original)])
        if prelude:
            conn.execute(prelude)
        conn.execute(f"UPDATE n SET {assignment}", values)
        conn.execute("COPY n TO ? (FORMAT PARQUET)", [str(replacement)])
    original.write_bytes(replacement.read_bytes())
    descriptor["normals"]["sha256"] = hashlib.sha256(original.read_bytes()).hexdigest()
    _json(root, "inputs.json", descriptor)


@pytest.mark.parametrize(
    ("assignment", "values", "code"),
    [
        ("p90_wet = ?", [None], "wet_day_reference_invalid"),
        ("p90_wet = ?", [float("nan")], "wet_day_reference_invalid"),
        ("p90_wet = ?", [1.0], "wet_day_reference_invalid"),
        ("p10_wet = ?", [0.5], "wet_day_reference_invalid"),
        ("n_wet = ?", [-1], "wet_day_reference_invalid"),
        ("n_wet = ?", [None], "wet_day_reference_invalid"),
        ("wet_day_threshold_mm = ?", [-1], "wet_day_reference_invalid"),
        ("wet_day_threshold_mm = ?", [float("nan")], "wet_day_reference_invalid"),
        ("schema_version = ?", [1], "precipitation_v2_reference_incomplete"),
    ],
)
def test_unusable_required_wet_reference_blocks_readiness(tmp_path, assignment, values, code):
    path, descriptor = _fixture(tmp_path, horizons=tuple(range(1, 8)), labels=5)
    _change_normals(tmp_path, descriptor, assignment, values)
    report = _run(path)
    assert report["status"] == "blocked"
    assert code in _codes(report)
    assert report["normals"]["required_months"] == [6]


def test_genuine_no_wet_days_is_unavailable_not_corrupt_or_zero_filled(tmp_path):
    path, descriptor = _fixture(tmp_path, horizons=tuple(range(1, 8)), labels=5)
    _change_normals(tmp_path, descriptor, "n_wet=0,p10_wet=NULL,p50_wet=NULL,p90_wet=NULL")
    report = _run(path)
    assert "wet_day_reference_unavailable" in _codes(report)
    assert "wet_day_reference_invalid" not in _codes(report)
    assert report["status"] == "blocked"


def test_unused_month_does_not_require_fabricated_wet_day_bands(tmp_path):
    path, descriptor = _fixture(tmp_path, horizons=tuple(range(1, 8)), labels=5)
    _change_normals(tmp_path, descriptor, "p90_wet=NULL WHERE month=1")
    assert _run(path)["status"] == "inputs_verified"


def test_valid_nondefault_wet_day_threshold_is_recorded(tmp_path):
    path, descriptor = _fixture(tmp_path, horizons=tuple(range(1, 8)), labels=5)
    _change_normals(tmp_path, descriptor, "wet_day_threshold_mm=0")
    report = _run(path)
    assert report["status"] == "inputs_verified"
    assert report["normals"]["wet_day_threshold_mm"] == [0.0]


@pytest.mark.parametrize(
    "selected", ["panel", "sampling", "labels", "normals", "era5", "artifact", "raw", "provenance"]
)
def test_mutating_original_after_capture_cannot_change_verified_snapshot(
    tmp_path, monkeypatch, selected
):
    path, descriptor = _fixture(tmp_path, horizons=tuple(range(1, 8)), labels=5)
    targets = {
        "panel": tmp_path / "panel.json",
        "sampling": tmp_path / "sampling.json",
        "labels": tmp_path / "labels.json",
        "normals": tmp_path / "normals.parquet",
        "era5": tmp_path / "era5.parquet",
        "artifact": sorted((tmp_path / "archive").glob("*.json"))[0],
        "raw": next(
            p for p in (tmp_path / "archive/raw").glob("*.json") if ".provenance." not in p.name
        ),
        "provenance": next((tmp_path / "archive/raw").glob("*.provenance.json")),
    }
    target = targets[selected]
    captured_digest = hashlib.sha256(target.read_bytes()).hexdigest()
    read = operator._preflight_bytes
    reads = []

    def replace_after_read(file, limit=operator._FILE_LIMIT):
        body = read(file, limit)
        if file == target:
            reads.append(file)
            file.write_bytes(b"mutated after capture")
        return body

    monkeypatch.setattr(operator, "_preflight_bytes", replace_after_read)
    report = _run(path)
    assert report["status"] == "inputs_verified"
    assert reads == [target]  # Shared raw/provenance are captured once, not once per horizon.
    assert hashlib.sha256(target.read_bytes()).hexdigest() != captured_digest
    if selected == "labels":
        assert report["labelset"]["sha256"] == descriptor["labelset"]["sha256"]
        assert report["labelset"]["label_count"] == 5
    elif selected == "normals":
        assert report["normals"]["sha256"] == captured_digest
    elif selected == "era5":
        assert report["normals"]["source_sha256"] == captured_digest
    elif selected == "panel":
        assert report["panel_sha256"] == captured_digest
    elif selected == "sampling":
        assert report["sampling_points_sha256"] == captured_digest
    else:
        key = {"artifact": "sha256", "raw": "raw_sha256", "provenance": "provenance_sha256"}[
            selected
        ]
        assert captured_digest in {artifact[key] for artifact in report["archives"]}


def test_label_hash_cannot_describe_old_empty_bytes_and_new_five_labels(tmp_path, monkeypatch):
    path, descriptor = _fixture(tmp_path, horizons=tuple(range(1, 8)), labels=0)
    original = tmp_path / "labels.json"
    read = operator._preflight_bytes

    def replace_labels(file, limit=operator._FILE_LIMIT):
        body = read(file, limit)
        if file == original:
            altered = json.loads(body)
            altered["labels"] = [{"event_id": "not_the_hashed_labels"}] * 5
            original.write_text(json.dumps(altered))
        return body

    monkeypatch.setattr(operator, "_preflight_bytes", replace_labels)
    report = _run(path)
    assert report["status"] == "blocked"
    assert report["labelset"]["sha256"] == descriptor["labelset"]["sha256"]
    assert report["labelset"]["label_count"] == 0


def test_snapshot_has_an_aggregate_byte_bound(tmp_path, monkeypatch):
    path, _ = _fixture(tmp_path)
    monkeypatch.setattr(operator, "_SNAPSHOT_LIMIT", 1)
    with pytest.raises(ValueError, match="snapshot limit"):
        _run(path)


def test_changed_descriptor_is_reauthenticated_before_it_can_select_inputs(tmp_path, monkeypatch):
    path, descriptor = _fixture(tmp_path)
    read = operator._preflight_bytes
    calls = []

    def change_after_initial_pin(file, limit=operator._FILE_LIMIT):
        body = read(file, limit)
        if file == path:
            calls.append(file)
            if len(calls) == 1:
                descriptor["panel"]["path"] = "must-not-be-selected.json"
                path.write_text(json.dumps(descriptor))
        assert file.name != "must-not-be-selected.json"
        return body

    expected = _pin(tmp_path, path.name)["sha256"]
    monkeypatch.setattr(operator, "_preflight_bytes", change_after_initial_pin)
    with pytest.raises(ValueError, match="SHA-256 mismatch: inputs.json"):
        operator.preflight_watch_inputs(path, expected)


def test_copied_panel_is_authenticated_before_loading_its_configuration(tmp_path, monkeypatch):
    path, descriptor = _fixture(tmp_path)
    descriptor["panel"]["sha256"] = "0" * 64
    _json(tmp_path, path.name, descriptor)

    def no_unpinned_panel(*_args, **_kwargs):
        raise AssertionError("Frozen panel loader must not see an unauthenticated panel")

    monkeypatch.setattr(operator, "_load_frozen_panel", no_unpinned_panel)
    with pytest.raises(ValueError, match="SHA-256 mismatch: panel.json"):
        _run(path)


def test_oversized_sampling_coordinate_is_a_cli_refusal_after_valid_hash_checks(tmp_path, capsys):
    path, descriptor = _fixture(tmp_path)
    sampling = json.loads((tmp_path / "sampling.json").read_bytes())
    sampling["sampling_points"][0]["latitude"] = 10**400
    descriptor["sampling_points"] = _json(tmp_path, "sampling.json", sampling)
    panel = json.loads((tmp_path / "panel.json").read_bytes())
    panel["configuration"]["sampling_points_sha256"] = descriptor["sampling_points"]["sha256"]
    descriptor["panel"] = _json(tmp_path, "panel.json", panel)
    _json(tmp_path, path.name, descriptor)
    assert (
        operator.main(
            ["--inputs", str(path), "--expected-inputs-sha256", _pin(tmp_path, path.name)["sha256"]]
        )
        == 2
    )
    report = json.loads(capsys.readouterr().out)
    assert report["status"] == "refused"
    assert report["reasons"][0]["code"] == "input_or_publication_refused"
    assert report["live_activation"] is False


def test_negative_rain_is_refused_after_native_artifact_rederivation(tmp_path):
    path, _ = _fixture(tmp_path, horizons=tuple(range(1, 8)), labels=5, hourly_value=-0.1)
    # The fixture re-normalizes genuine negative raw hourly values and writes matching
    # raw digests, provenance and canonical native artifacts; this is not hash corruption.
    frozen = archive._load_frozen_panel([tmp_path / "panel.json"], tmp_path / "sampling.json")
    native = archive._verify_artifact(
        sorted((tmp_path / "archive").glob("*.json"))[0],
        frozen,
        frozen.identities[0].panel,
        ZONE,
        1,
    )
    assert native["rows"][0]["value"] < 0
    with pytest.raises(operator.PreflightRefusal, match="precipitation cannot be negative"):
        _run(path)


def test_zero_archive_rain_remains_usable(tmp_path):
    path, _ = _fixture(tmp_path, horizons=tuple(range(1, 8)), labels=5, hourly_value=0.0)
    report = _run(path)
    assert report["status"] == "inputs_verified"
    assert all(row["coverage"]["available_days"] == 5 for row in report["archives"])


@pytest.mark.parametrize(
    ("assignment", "values", "prelude"),
    [
        ("n=?", [0], None),
        ("n=?", [None], None),
        ("n=?", [-1], None),
        ("n=?", [9], None),
        ("n_wet=?", [101], None),
        ("n=?", [100.5], "ALTER TABLE n ALTER COLUMN n TYPE DOUBLE"),
        ("n=?", [float("nan")], "ALTER TABLE n ALTER COLUMN n TYPE DOUBLE"),
        ("n=?", [float("inf")], "ALTER TABLE n ALTER COLUMN n TYPE DOUBLE"),
        ("p90_wet=p90_wet", [], "ALTER TABLE n DROP COLUMN n"),
    ],
)
def test_wet_count_cannot_exceed_total_or_use_an_invalid_total(
    tmp_path, assignment, values, prelude
):
    path, descriptor = _fixture(tmp_path, horizons=tuple(range(1, 8)), labels=5)
    _change_normals(tmp_path, descriptor, assignment, values, prelude=prelude)
    report = _run(path)
    assert report["status"] == "blocked"
    assert "wet_day_reference_invalid" in _codes(report)


def test_total_zero_and_wet_zero_remain_unavailable(tmp_path):
    path, descriptor = _fixture(tmp_path, horizons=tuple(range(1, 8)), labels=5)
    _change_normals(tmp_path, descriptor, "n=0,n_wet=0,p10_wet=NULL,p50_wet=NULL,p90_wet=NULL")
    report = _run(path)
    assert report["status"] == "blocked"
    assert "wet_day_reference_unavailable" in _codes(report)
    assert "wet_day_reference_invalid" not in _codes(report)


def test_all_wet_days_can_equal_positive_total(tmp_path):
    path, descriptor = _fixture(tmp_path, horizons=tuple(range(1, 8)), labels=5)
    _change_normals(tmp_path, descriptor, "n=n_wet")
    assert _run(path)["status"] == "inputs_verified"


def _rewrite_raw_timezone(root, timezone):
    frozen = archive._load_frozen_panel([root / "panel.json"], root / "sampling.json")
    panel = frozen.identities[0].panel
    directory = root / "archive"
    artifacts = sorted(directory.glob("*.json"))
    first = json.loads(artifacts[0].read_bytes())
    payload = json.loads((directory / "raw" / f"{first['raw_sha256']}.json").read_bytes())
    payload["timezone"] = timezone
    body = json.dumps(payload).encode()
    digest = hashlib.sha256(body).hexdigest()
    raw = RawResponse(body, digest, "2026-10-21T00:00:00Z", {"latitude": POINT[0]})
    (directory / "raw" / f"{digest}.json").write_bytes(body)
    provenance = archive._provenance_name(frozen, ZONE, panel, digest)
    _json(
        directory / "raw",
        provenance,
        archive._provenance(
            SimpleNamespace(base_url="https://example.invalid"), frozen, ZONE, panel, raw
        ),
    )
    for path in artifacts:
        stored = json.loads(path.read_bytes())
        stored.update(raw_sha256=digest, provenance=provenance)
        path.write_bytes(archive._json_bytes(stored))


def test_unhashable_raw_timezone_is_a_cli_input_refusal(tmp_path):
    path, _ = _fixture(tmp_path, horizons=tuple(range(1, 8)), labels=5)
    # Preserve the hourly data, canonical rows and all identities; retain the malformed
    # raw timezone in a newly hashed response and its newly bound provenance/artifacts.
    _rewrite_raw_timezone(tmp_path, [])
    repo = Path(operator.__file__).resolve().parents[2]
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.cli.watch_replay_preflight",
            "--inputs",
            str(path),
            "--expected-inputs-sha256",
            _pin(tmp_path, path.name)["sha256"],
        ],
        cwd=repo,
        env={**os.environ, "PYTHONPATH": str(repo)},
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )
    assert completed.returncode == 2
    report = json.loads(completed.stdout)
    assert report["status"] == "refused"
    assert "Invalid retained archive input" in report["reasons"][0]["detail"]
    assert "Traceback" not in completed.stderr


def test_programming_type_error_outside_input_boundary_is_not_masked(tmp_path, monkeypatch):
    def programming_error(*_args):
        raise TypeError("programming sentinel")

    monkeypatch.setattr(operator, "preflight_watch_inputs", programming_error)
    with pytest.raises(TypeError, match="programming sentinel"):
        operator.main(
            ["--inputs", str(tmp_path / "unused.json"), "--expected-inputs-sha256", "0" * 64]
        )
