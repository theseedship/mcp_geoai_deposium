"""Authenticated local fixtures and fake mining; no providers or real Q3 mining."""

import argparse
import hashlib
import json
import subprocess
import zipfile
from pathlib import Path

import duckdb
import pandas as pd
import pytest

from scripts import o3_q3_offline_replay as replay
from src.patterns.offline_bundle import OUTPUT_KEYS, OfflineBundleError


@pytest.fixture(autouse=True)
def supported_replay_version(monkeypatch):
    """Fake miners exercise replay bookkeeping in the production-pinned env."""
    original = replay.importlib.metadata.version
    monkeypatch.setattr(
        replay.importlib.metadata,
        "version",
        lambda name: "1.0.9" if name == "so4gp" else original(name),
    )


@pytest.mark.parametrize("version", ["0.9.1", "1.0.8", None])
def test_replay_refuses_wrong_or_absent_so4gp_before_mining(fixture_bundle, monkeypatch, version):
    write, _, _ = fixture_bundle
    args = write()
    original = replay.importlib.metadata.version

    def version_lookup(name):
        if name == "so4gp":
            if version is None:
                raise replay.importlib.metadata.PackageNotFoundError(name)
            return version
        return original(name)

    monkeypatch.setattr(replay.importlib.metadata, "version", version_lookup)
    monkeypatch.setattr(replay, "_mine_zone", lambda *a, **k: pytest.fail("Mining must not begin"))
    with pytest.raises(OfflineBundleError, match=r"requires so4gp==1\.0\.9"):
        replay.run(args)
    manifest = json.loads((args.output_dir / "manifest.json").read_text())
    assert manifest["status"] == "failed"
    assert manifest["error"]["type"] == "OfflineBundleError"


@pytest.mark.parametrize("populated", [False, True])
def test_validated_parquet_has_stable_flat_schema(fixture_bundle, monkeypatch, populated):
    write, _, _ = fixture_bundle
    args = write()

    def mine(frame, **kwargs):
        result = fake_pattern("target_col" in kwargs)
        if not populated:
            result["patterns"] = []
        kwargs["capture_raw_output"](json.dumps({"Patterns": result["patterns"]}))
        return result

    monkeypatch.setattr(replay, "mine_gradual_patterns", mine)
    monkeypatch.setattr(replay, "mine_temporal_patterns", mine)
    replay.run(args)
    expected = [
        ("pattern_id", "BIGINT"),
        ("antecedent", "VARCHAR"),
        ("consequent", "VARCHAR"),
        ("pattern", "VARCHAR"),
        ("variables", "VARCHAR"),
        ("support", "DOUBLE"),
        ("confidence", "DOUBLE"),
        ("is_temporal", "BOOLEAN"),
        ("lag_days", "DOUBLE"),
        ("algorithm", "VARCHAR"),
        ("zone_id", "VARCHAR"),
        ("zones_validated", "VARCHAR"),
        ("period_start", "VARCHAR"),
        ("period_end", "VARCHAR"),
        ("n_folds", "BIGINT"),
        ("stability", "DOUBLE"),
        ("created_at", "VARCHAR"),
        ("extra_metrics", "VARCHAR"),
    ]
    with duckdb.connect(":memory:", config=replay.DUCKDB_CONFIG) as conn:
        for algorithm in ("graank", "tgrad"):
            path = args.output_dir / f"{algorithm}_validated.parquet"
            schema = conn.execute("DESCRIBE SELECT * FROM read_parquet(?)", [str(path)]).fetchall()
            assert [(row[0], row[1]) for row in schema] == expected
            rows = conn.execute(
                "SELECT confidence, is_temporal, lag_days, period_start, n_folds FROM read_parquet(?)",
                [str(path)],
            ).fetchall()
            if populated:
                assert rows and all(row[0] is None for row in rows)
                assert all(
                    row[1] == (algorithm == "tgrad") and row[3] == "2023-07-01" and row[4] == 5
                    for row in rows
                )
            else:
                assert rows == []
    manifest = json.loads((args.output_dir / "manifest.json").read_text())
    assert manifest["status"] == "provisional_replay_complete"


@pytest.mark.parametrize("field", [None, "zone_artifact", "parameters"])
@pytest.mark.parametrize("value", [[], None, 1, "invalid"])
def test_checksum_valid_nonobject_monthly_manifest_rejected(
    fixture_bundle, monkeypatch, field, value
):
    write, payloads, _ = fixture_bundle
    key = "climate/runs/2023_07/manifest.json"
    name = f"objects/{key}"
    monthly = json.loads(payloads[name])
    if field is None:
        monthly = value
    else:
        monthly[field] = value
    payloads[name] = json.dumps(monthly).encode()
    bundle = json.loads(payloads["bundle.json"])
    obj = next(obj for obj in bundle["objects"] if obj["key"] == key)
    old_bytes = obj["bytes"]
    obj.update(bytes=len(payloads[name]), sha256=hashlib.sha256(payloads[name]).hexdigest())
    bundle["inventory"]["total_source_bytes"] += obj["bytes"] - old_bytes
    payloads["bundle.json"] = json.dumps(bundle).encode()
    monkeypatch.setattr(replay, "_mine_zone", lambda *a, **k: pytest.fail("Mining must not begin"))
    args = write()
    with pytest.raises(OfflineBundleError, match="must be.*JSON object"):
        replay.run(args)
    assert not args.output_dir.exists()


@pytest.mark.parametrize("stage", ["mining", "summary_hash", "completion_manifest"])
def test_keyboard_interrupt_revokes_running_or_completed_status(fixture_bundle, monkeypatch, stage):
    write, _, _ = fixture_bundle
    args = write()
    previous = Path.cwd()

    def mine(frame, **kwargs):
        kwargs["capture_raw_output"]('{"Patterns": []}')
        if stage == "mining":
            raise KeyboardInterrupt()
        return fake_pattern("target_col" in kwargs)

    original_read = Path.read_bytes

    def read(path):
        if stage == "summary_hash" and path == args.output_dir / "summary.json":
            raise KeyboardInterrupt()
        return original_read(path)

    original_write = replay._write_json

    def write_json(path, value):
        if (
            stage == "completion_manifest"
            and path.name == "manifest.json"
            and value["status"] == "provisional_replay_complete"
        ):
            raise KeyboardInterrupt()
        original_write(path, value)

    monkeypatch.setattr(replay, "mine_gradual_patterns", mine)
    monkeypatch.setattr(replay, "mine_temporal_patterns", mine)
    monkeypatch.setattr(Path, "read_bytes", read)
    monkeypatch.setattr(replay, "_write_json", write_json)
    with pytest.raises(KeyboardInterrupt):
        replay.run(args)
    assert Path.cwd() == previous
    manifest = json.loads((args.output_dir / "manifest.json").read_text())
    assert manifest["status"] == "failed"
    assert manifest["error"]["type"] == "KeyboardInterrupt"
    assert "output_hashes" not in manifest
    if stage != "mining":
        assert json.loads((args.output_dir / "summary.json").read_text())["status"] == "failed"
    assert list((args.output_dir / "zones").rglob("*.raw.json"))


@pytest.mark.parametrize("dirty", [False, True])
def test_git_source_snapshot_precedes_output_allocation(
    fixture_bundle, monkeypatch, tmp_path, dirty
):
    write, _, _ = fixture_bundle
    args = write()
    root = tmp_path / "checkout"
    root.mkdir()

    def git(*args):
        return subprocess.run(
            ["git", "-C", str(root), *args], check=True, capture_output=True, text=True
        )

    git("init", "-q")
    git(
        "-c",
        "user.name=Fixture",
        "-c",
        "user.email=fixture@example.invalid",
        "commit",
        "--allow-empty",
        "-qm",
        "fixture",
    )
    if dirty:
        (root / "existing-wip").write_text("preserve")
    args.output_dir = root / "new-output"
    monkeypatch.setattr(replay, "ROOT", root)

    def fail_environment():
        raise RuntimeError("stop after snapshot")

    monkeypatch.setattr(replay, "_environment", fail_environment)
    with pytest.raises(RuntimeError, match="stop after snapshot"):
        replay.run(args)
    evidence = json.loads((args.output_dir / "manifest.json").read_text())
    assert evidence["git"] == {"head": git("rev-parse", "HEAD").stdout.strip(), "dirty": dirty}
    assert evidence["status"] == "failed"
    assert git("status", "--porcelain").stdout.strip()


@pytest.mark.parametrize("missing_git", [False, True])
def test_git_preflight_is_actionable_and_allocates_no_output(
    fixture_bundle, monkeypatch, tmp_path, missing_git
):
    write, _, _ = fixture_bundle
    args = write()
    monkeypatch.setattr(replay, "ROOT", tmp_path)
    if missing_git:

        def unavailable(*args, **kwargs):
            raise FileNotFoundError("git")

        monkeypatch.setattr(replay.subprocess, "run", unavailable)
    with pytest.raises(OfflineBundleError, match="Git checkout.*cloned repository"):
        replay.run(args)
    assert not args.output_dir.exists()


def test_replay_writes_exact_raw_and_hashes_alongside_normalized(fixture_bundle, monkeypatch):
    write, _, zones = fixture_bundle
    args = write()
    raw = '{ "Algorithm": "native", "Patterns": [], "producer_field": 42 }\n'

    def mine(frame, **kwargs):
        kwargs["capture_raw_output"](raw)
        return fake_pattern("target_col" in kwargs)

    monkeypatch.setattr(replay, "mine_gradual_patterns", mine)
    monkeypatch.setattr(replay, "mine_temporal_patterns", mine)
    replay.run(args)
    evidence = json.loads((args.output_dir / "manifest.json").read_text())
    for zone in zones[:-1]:
        for algorithm in ("graank", "tgrad"):
            path = args.output_dir / "zones" / zone / f"{algorithm}.raw.json"
            assert path.read_text() == raw
            assert (
                evidence["output_hashes"][str(path.relative_to(args.output_dir))]
                == hashlib.sha256(raw.encode()).hexdigest()
            )
            normalized = json.loads(path.with_name(f"{algorithm}.normalized.json").read_text())
            assert normalized["result"] == fake_pattern(algorithm == "tgrad")


def test_native_error_raw_survives_failed_normalization(fixture_bundle, monkeypatch):
    write, _, zones = fixture_bundle
    args = write()
    raw = '{ "Error": "native failure" }\n'
    previous = Path.cwd()

    def fail(frame, **kwargs):
        kwargs["capture_raw_output"](raw)
        raise ValueError("native failure")

    monkeypatch.setattr(replay, "mine_gradual_patterns", fail)
    with pytest.raises(ValueError, match="native failure"):
        replay.run(args)
    assert Path.cwd() == previous
    destination = args.output_dir / "zones" / zones[0]
    assert (destination / "graank.raw.json").read_text() == raw
    assert not (destination / "graank.normalized.json").exists()
    manifest = json.loads((args.output_dir / "manifest.json").read_text())
    assert manifest["status"] == "failed"
    assert manifest["error"]["message"] == "native failure"


def test_cli_git_preflight_uses_concise_error_handler(
    fixture_bundle, monkeypatch, tmp_path, capsys
):
    write, _, _ = fixture_bundle
    args = write()
    monkeypatch.setattr(replay, "ROOT", tmp_path)
    assert (
        replay.main(
            [
                "--archive",
                str(args.archive),
                "--expected-sha256",
                args.expected_sha256,
                "--zones-geojson",
                str(args.zones_geojson),
                "--output-dir",
                str(args.output_dir),
            ]
        )
        == 1
    )
    assert "Offline replay failed: Replay requires Git" in capsys.readouterr().err
    assert not args.output_dir.exists()


@pytest.fixture
def fixture_bundle(tmp_path):
    zones = ["svalbard", *[f"zone_{i}" for i in range(13)], "kerguelen"]
    geo = tmp_path / "zones.geojson"
    geo.write_text(
        json.dumps(
            {
                "type": "FeatureCollection",
                "features": [
                    {"type": "Feature", "properties": {"zone_id": zone}, "geometry": None}
                    for zone in zones
                ],
            }
        )
    )
    zone_sha = hashlib.sha256(geo.read_bytes()).hexdigest()
    objects, payloads = [], {}

    def add(key, role, data, readback):
        objects.append(
            {
                "key": key,
                "role": role,
                "bytes": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
                "readback": readback,
            }
        )
        payloads[f"objects/{key}"] = data

    for key in OUTPUT_KEYS:
        add(key, "o3_output", b"historical bytes only", {})
    for month in replay.MONTHS:
        for zone in zones:
            dates = [month.replace("_", "-") + "-20", month.replace("_", "-") + "-02"]
            if zone == "kerguelen":
                dates = ["2023-08-15"] if month == "2023_08" else []
            frame = pd.DataFrame(
                {
                    "date": dates,
                    **{
                        column: [float(i + 1) for i in range(len(dates))]
                        for column in replay.FEATURES
                    },
                }
            )
            parquet = tmp_path / "fixture.parquet"
            with duckdb.connect(":memory:") as conn:
                conn.register("fixture", frame)
                conn.execute("COPY fixture TO ? (FORMAT PARQUET)", [str(parquet)])
            add(
                f"climate/holdout/unified/{month}/{zone}.parquet",
                "o3_input_unified",
                parquet.read_bytes(),
                {
                    "rows": len(dates),
                    "min_date": min(dates) if dates else None,
                    "max_date": max(dates) if dates else None,
                },
            )
        number = int(month[-2:])
        last_day = 30 if number == 9 else 31
        manifest = {
            "schema_version": 1,
            "kind": "geoai_run_manifest",
            "zone_artifact": {"sha256": zone_sha, "selected_zone_ids": zones, "n_selected": 15},
            "parameters": {
                "artifact_window_label": month,
                "zone_ids": zones,
                "start_date": f"2023-{number:02d}-01",
                "end_date": f"2023-{number:02d}-{last_day}",
            },
        }
        add(
            f"climate/runs/{month}/manifest.json",
            "b4_input_manifest",
            json.dumps(manifest).encode(),
            {},
        )
    manifest = {
        "schema_version": "geoai-o3-offline-handoff/v1",
        "generated_at": "2026-10-05",
        "historical_producers": {},
        "verification": {},
        "caveats": [],
        "objects": objects,
        "inventory": {
            "object_count": len(objects),
            "total_source_bytes": sum(x["bytes"] for x in objects),
        },
    }
    payloads["bundle.json"] = json.dumps(manifest).encode()
    payloads["README-O3-HANDOFF.md"] = b"fixture"

    def write():
        archive = tmp_path / "fixture.zip"
        with zipfile.ZipFile(archive, "w") as bundle:
            for name, data in payloads.items():
                bundle.writestr(name, data)
        return argparse.Namespace(
            archive=archive,
            expected_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
            zones_geojson=geo,
            output_dir=tmp_path / "result",
        )

    return write, payloads, zones


def fake_pattern(temporal=False):
    items = [
        {"attribute": "ndvi_mean", "direction": "+"},
        {"attribute": "precipitation", "direction": "+"},
    ]
    if temporal:
        items[0].update(role="target", lag_days=0.0)
        items[1].update(role="lagged", lag_days=2.0)
    return {
        "patterns": [
            {
                "gradual_items": items,
                "support": 0.8,
                "is_temporal": temporal,
                "metrics": {"conf": 0.93, "density": 0.51},
            }
        ]
    }


def test_replay_preserves_order_and_keeps_native_metrics_per_zone(fixture_bundle, monkeypatch):
    write, _, zones = fixture_bundle
    calls = []

    def graank(frame, **kwargs):
        assert list(frame.columns) == list(replay.FEATURES)
        assert "target_col" not in kwargs
        assert frame["ndvi_mean"].tolist() == [2.0, 1.0, 2.0, 1.0, 2.0, 1.0]
        assert Path.cwd().name.startswith("geoai-offline-mine-")
        calls.append("graank")
        return fake_pattern()

    def temporal(frame, **kwargs):
        assert list(frame.columns) == ["date", *replay.FEATURES]
        assert kwargs["target_col"] == 1 and kwargs["parallel"] is False
        assert frame["date"].is_monotonic_increasing
        calls.append("tgrad")
        return fake_pattern(True)

    monkeypatch.setattr(replay, "mine_gradual_patterns", graank)
    monkeypatch.setattr(replay, "mine_temporal_patterns", temporal)
    args = write()
    previous = Path.cwd()
    replay.run(args)
    assert Path.cwd() == previous
    assert calls == ["graank"] * 14 + ["tgrad"] * 14
    manifest = json.loads((args.output_dir / "manifest.json").read_text())
    assert manifest["zone_ids"] == zones
    assert manifest["eligible_zone_ids"] == zones[:-1]
    assert manifest["status"] == "provisional_replay_complete"
    folds = json.loads((args.output_dir / "folds.json").read_text())
    assert folds["folds"][0]["validation_zones"] == zones[:-1][::5]
    assert folds["historical_assignment_available"] is False
    native = json.loads((args.output_dir / "zones/svalbard/graank.normalized.json").read_text())
    assert native["result"]["patterns"][0]["metrics"]["conf"] == 0.93
    for algorithm in ("graank", "tgrad"):
        library = json.loads((args.output_dir / f"{algorithm}_validated.json").read_text())
        assert library["patterns"]
        for pattern in library["patterns"]:
            assert pattern["confidence"] is None
            assert "conf" not in pattern["extra_metrics"]
            assert "density" not in pattern["extra_metrics"]
        with duckdb.connect(":memory:") as conn:
            count = conn.execute(
                "SELECT count(*) FROM read_parquet(?) WHERE confidence IS NOT NULL",
                [str(args.output_dir / f"{algorithm}_validated.parquet")],
            ).fetchone()[0]
            assert count == 0
    with pytest.raises(FileExistsError):
        replay.run(args)


@pytest.mark.parametrize(
    "damage",
    [
        "archive_hash",
        "member_hash",
        "undeclared",
        "traversal",
        "missing_field",
        "extra_field",
        "row_drift",
        "zone_order",
    ],
)
def test_rejects_integrity_and_order_drift_before_mining(fixture_bundle, monkeypatch, damage):
    write, payloads, _ = fixture_bundle
    manifest = json.loads(payloads["bundle.json"])
    if damage == "member_hash":
        manifest["objects"][0]["sha256"] = "0" * 64
    elif damage == "undeclared":
        payloads["unexpected"] = b"data"
    elif damage == "traversal":
        payloads["../escape"] = b"data"
    elif damage == "missing_field":
        manifest["objects"][0].pop("bytes")
    elif damage == "extra_field":
        manifest["objects"][0]["unexpected"] = True
    elif damage == "row_drift":
        manifest["objects"][2]["readback"]["rows"] = 999
    elif damage == "zone_order":
        geo = json.loads(write().zones_geojson.read_text())
        geo["features"].reverse()
        write().zones_geojson.write_text(json.dumps(geo))
    payloads["bundle.json"] = json.dumps(manifest).encode()
    args = write()
    if damage == "archive_hash":
        args.expected_sha256 = "0" * 64
    monkeypatch.setattr(replay, "_mine_zone", lambda *a: pytest.fail("Mining must not begin"))
    with pytest.raises(OfflineBundleError):
        replay.run(args)
    assert not args.output_dir.exists()


def test_failed_mining_restores_cwd_and_records_failure(fixture_bundle, monkeypatch):
    write, _, _ = fixture_bundle
    previous = Path.cwd()

    def fail(*args, **kwargs):
        assert Path.cwd() != previous
        raise ValueError("fake mining failure")

    monkeypatch.setattr(replay, "mine_gradual_patterns", fail)
    args = write()
    with pytest.raises(ValueError, match="fake mining failure"):
        replay.run(args)
    assert Path.cwd() == previous
    manifest = json.loads((args.output_dir / "manifest.json").read_text())
    assert manifest["status"] == "failed"
    assert not (args.output_dir / "summary.json").exists()


def test_cv_copy_does_not_mutate_sidecar():
    source = {"zone": fake_pattern()}
    source["zone"]["patterns"][0].update(extra_metrics={"conf": 0.5}, confidence=0.7, conf=0.8)
    clean = replay._cv_input_without_native_metrics(source)
    assert clean["zone"]["patterns"][0].keys() == {"gradual_items", "support", "is_temporal"}
    assert source["zone"]["patterns"][0]["metrics"]["conf"] == 0.93


def test_environment_setup_failure_records_failed_manifest(fixture_bundle, monkeypatch):
    write, _, _ = fixture_bundle
    args = write()

    def fail_environment():
        raise RuntimeError("environment setup failed")

    monkeypatch.setattr(replay, "_environment", fail_environment)
    monkeypatch.setattr(replay, "_mine_zone", lambda *a: pytest.fail("Mining must not begin"))
    with pytest.raises(RuntimeError, match="environment setup failed"):
        replay.run(args)
    manifest = json.loads((args.output_dir / "manifest.json").read_text())
    assert manifest["status"] == "failed"
    assert manifest["error"]["type"] == "RuntimeError"
    assert not (args.output_dir / "summary.json").exists()


def test_hash_read_failure_does_not_publish_completed_summary(fixture_bundle, monkeypatch):
    write, _, _ = fixture_bundle
    args = write()
    original_read = Path.read_bytes

    def fail_summary_read(path):
        if path == args.output_dir / "summary.json":
            raise OSError("summary hash read failed")
        return original_read(path)

    monkeypatch.setattr(Path, "read_bytes", fail_summary_read)
    monkeypatch.setattr(replay, "mine_gradual_patterns", lambda *a, **k: fake_pattern())
    monkeypatch.setattr(replay, "mine_temporal_patterns", lambda *a, **k: fake_pattern(True))
    with pytest.raises(OSError, match="summary hash read failed"):
        replay.run(args)
    manifest = json.loads((args.output_dir / "manifest.json").read_text())
    summary = json.loads((args.output_dir / "summary.json").read_text())
    assert manifest["status"] == "failed"
    assert summary["status"] == "failed"
    assert "output_hashes" not in manifest
    assert manifest["error"]["message"] == "summary hash read failed"


def test_final_manifest_failure_revokes_completion(fixture_bundle, monkeypatch):
    write, _, _ = fixture_bundle
    args = write()
    original_write = replay._write_json

    def fail_completion_manifest(path, value):
        if path.name == "manifest.json" and value["status"] == "provisional_replay_complete":
            raise OSError("final manifest write failed")
        original_write(path, value)

    monkeypatch.setattr(replay, "_write_json", fail_completion_manifest)
    monkeypatch.setattr(replay, "mine_gradual_patterns", lambda *a, **k: fake_pattern())
    monkeypatch.setattr(replay, "mine_temporal_patterns", lambda *a, **k: fake_pattern(True))
    with pytest.raises(OSError, match="final manifest write failed"):
        replay.run(args)
    assert json.loads((args.output_dir / "manifest.json").read_text())["status"] == "failed"
    assert json.loads((args.output_dir / "summary.json").read_text())["status"] == "failed"


@pytest.mark.parametrize(
    "damage", ["null", "nonfinite", "duplicate_date", "outside_month", "missing_column"]
)
def test_rejects_invalid_observations_even_when_hashes_match(
    fixture_bundle, monkeypatch, damage, tmp_path
):
    write, payloads, _ = fixture_bundle
    manifest = json.loads(payloads["bundle.json"])
    obj = manifest["objects"][2]
    local = tmp_path / "bad.parquet"
    local.write_bytes(payloads[f"objects/{obj['key']}"])
    with duckdb.connect(":memory:") as conn:
        frame = conn.execute("SELECT * FROM read_parquet(?)", [str(local)]).df()
        if damage == "null":
            frame.loc[0, "ndvi_mean"] = None
        elif damage == "nonfinite":
            frame.loc[0, "ndvi_mean"] = float("inf")
        elif damage == "duplicate_date":
            frame.loc[0, "date"] = frame.loc[1, "date"]
        elif damage == "outside_month":
            frame.loc[0, "date"] = "2023-06-20"
        else:
            frame = frame.drop(columns=["ndvi_mean"])
        conn.register("bad", frame)
        conn.execute("COPY bad TO ? (FORMAT PARQUET)", [str(local)])
    data = local.read_bytes()
    payloads[f"objects/{obj['key']}"] = data
    obj.update(bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
    obj["readback"].update(min_date=min(frame["date"]), max_date=max(frame["date"]))
    manifest["inventory"]["total_source_bytes"] = sum(item["bytes"] for item in manifest["objects"])
    payloads["bundle.json"] = json.dumps(manifest).encode()
    args = write()
    monkeypatch.setattr(replay, "_mine_zone", lambda *a: pytest.fail("Mining must not begin"))
    with pytest.raises(OfflineBundleError):
        replay.run(args)
    assert not args.output_dir.exists()
