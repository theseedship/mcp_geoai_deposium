"""Offline integrity boundary, structural readback and safe deterministic reports."""

from __future__ import annotations

import errno
import hashlib
import json
import stat
import struct
import subprocess
import sys
import zipfile
import zlib
from pathlib import Path
from unittest.mock import Mock

import duckdb
import pytest

from src.cli.o3_bundle_report import main
from src.patterns import offline_bundle as reader


@pytest.fixture
def test_rows():
    return {
        "pattern_id": 0,
        "antecedent": "ndvi_mean+",
        "consequent": "",
        "pattern": '</script><script>alert("XSS")</script>&\u2028\u2029',
        "variables": json.dumps(["ndvi_mean", "temperature_2m"]),
        "support": 0.6,
        "confidence": None,
        "is_temporal": False,
        "lag_days": 0.0,
        "algorithm": "graank",
        "zone_id": "cross_zone",
        "zones_validated": json.dumps(["zone_a", "zone_b"]),
        "period_start": "2023-07-01",
        "period_end": "2023-09-30",
        "n_folds": 5,
        "stability": 0.4,
        "created_at": "2026-06-27T00:00:00Z",
        "extra_metrics": json.dumps({"density": 0.5}),
    }


def _rewrite_test_archive(path, members):
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        for name, data in members:
            bundle.writestr(name, data)
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def test_archive(tmp_path, test_rows):
    zones = {
        "type": "FeatureCollection",
        "features": [
            {"type": "Feature", "properties": {"zone_id": zone, "koppen": code}, "geometry": None}
            for zone, code in [("zone_a", "Csa"), ("zone_b", "Af")]
        ],
    }
    zones_path = tmp_path / "zones.geojson"
    zones_path.write_text(json.dumps(zones), encoding="utf-8")
    objects = []
    members = []
    for key, algorithm in reader.OUTPUT_KEYS.items():
        row = test_rows | {
            "algorithm": algorithm,
            "is_temporal": algorithm == "tgrad",
            "lag_days": 2.0 if algorithm == "tgrad" else 0.0,
        }
        path = tmp_path / f"{algorithm}.parquet"
        with duckdb.connect() as con:
            con.execute(
                "CREATE TABLE patterns AS SELECT "
                + ", ".join(f'? AS "{column}"' for column in row),
                list(row.values()),
            )
            con.execute("COPY patterns TO ? (FORMAT PARQUET)", [str(path)])
        data = path.read_bytes()
        objects.append(
            {
                "key": key,
                "role": "o3_output",
                "bytes": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
            }
        )
        members.append((f"objects/{key}", data))
    b4 = {
        "schema_version": 1,
        "kind": "geoai_run_manifest",
        "zone_artifact": {
            "sha256": hashlib.sha256(zones_path.read_bytes()).hexdigest(),
            "selected_zone_ids": ["zone_a", "zone_b"],
            "n_selected": 2,
        },
        "parameters": {"start_date": "2023-07-01", "end_date": "2023-07-31"},
        "outputs": [
            {"kind": "unified", "zone_id": "zone_a", "n_matched": 13},
            {"kind": "unified", "zone_id": "zone_b", "n_matched": 0},
        ],
    }
    key = "climate/runs/test/manifest.json"
    data = json.dumps(b4).encode()
    objects.append(
        {
            "key": key,
            "role": "b4_input_manifest",
            "bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
        }
    )
    members.append((f"objects/{key}", data))
    manifest = {
        "schema_version": "geoai-o3-offline-handoff/v1",
        "generated_at": "2026-09-06T22:18:35Z",
        "historical_producers": {"o3_output_base_commit": "8c3f8d7"},
        "verification": {"repository_commit": "6b48981"},
        "inventory": {
            "object_count": len(objects),
            "total_source_bytes": sum(o["bytes"] for o in objects),
        },
        "objects": objects,
        "caveats": ["Scientific thresholds unratified"],
    }
    members += [
        ("bundle.json", json.dumps(manifest).encode()),
        ("README-O3-HANDOFF.md", b"offline handoff"),
    ]
    path = tmp_path / "bundle.zip"
    sha = _rewrite_test_archive(path, members)
    return path, sha, manifest, members, zones_path


def test_verified_bundle(test_archive):
    path, sha, manifest, _, _ = test_archive
    report = reader.load_offline_bundle(path, sha)
    assert report["integrity"]["verified_objects"] == len(manifest["objects"])
    assert [p["algorithm"] for p in report["patterns"]] == ["graank", "tgrad"]
    assert all(p["confidence"] is None for p in report["patterns"])
    assert report["patterns"][1]["lag_days"] == 2
    assert report["patterns"][0]["variables"] == ["ndvi_mean", "temperature_2m"]
    assert report["source_bundle"]["generated_at"] == manifest["generated_at"]
    assert report["zone_class_coverage"] is None
    assert report["manifest_reported_rows"][1]["n_matched"] == 0
    page = reader.render_bundle_html(report)
    assert '</script><script>alert("XSS")' not in page
    assert "\\u003c/script\\u003e" in page and "\\u2028" in page and "\\u2029" in page
    assert "textContent" in page and "innerHTML" not in page and "eval(" not in page
    assert "null (non renseignée)" in page
    assert "&lt;/script&gt;" in reader.render_bundle_markdown(report)


@pytest.mark.parametrize(
    "issue",
    [
        "archive_hash",
        "missing_archive_hash",
        "missing_hash",
        "bad_hash",
        "bad_length",
        "missing_member",
        "extra_member",
        "duplicate_key",
    ],
)
def test_integrity_refused_before_parquet(test_archive, monkeypatch, issue):
    path, sha, manifest, members, _ = test_archive
    mock = Mock(side_effect=AssertionError("Parquet read before all integrity checks"))
    monkeypatch.setattr(reader, "_read_library", mock)
    if issue == "archive_hash":
        sha = "0" * 64
    elif issue == "missing_archive_hash":
        sha = ""
    else:
        if issue == "missing_hash":
            del manifest["objects"][-1]["sha256"]
        elif issue == "bad_hash":
            manifest["objects"][-1]["sha256"] = "0" * 64
        elif issue == "bad_length":
            manifest["objects"][-1]["bytes"] += 1
        elif issue == "missing_member":
            members = members[1:]
        elif issue == "extra_member":
            members.append(("objects/undeclared.txt", b"extra"))
        encoded = json.dumps(manifest).encode()
        if issue == "duplicate_key":
            encoded = encoded.replace(
                b'"schema_version":', b'"schema_version":"bad", "schema_version":', 1
            )
        members = [(name, encoded if name == "bundle.json" else data) for name, data in members]
        sha = _rewrite_test_archive(path, members)
    with pytest.raises(reader.OfflineBundleError):
        reader.load_offline_bundle(path, sha)
    mock.assert_not_called()


@pytest.mark.parametrize(
    "name",
    [
        "../escape",
        "/absolute",
        "objects/../escape",
        "objects\\escape",
        "objects//escape",
        "C:/escape",
        "objects/./escape",
    ],
)
def test_unsafe_zip(test_archive, name):
    path, _, _, members, _ = test_archive
    members.append((name, b"bad"))
    sha = _rewrite_test_archive(path, members)
    with pytest.raises(reader.OfflineBundleError, match="Chemin ZIP"):
        reader.load_offline_bundle(path, sha)


@pytest.mark.parametrize(
    "issue", ["duplicate", "symlink", "schema", "role", "envelope", "inventory", "missing_manifest"]
)
def test_manifest_refusals(test_archive, issue):
    path, _, manifest, members, _ = test_archive
    if issue == "duplicate":
        members.append(members[0])
    elif issue == "symlink":
        info = zipfile.ZipInfo("symlink")
        info.create_system = 3
        info.external_attr = (stat.S_IFLNK | 0o777) << 16
        members.append((info, b"target"))
    elif issue == "schema":
        manifest["schema_version"] = "v99"
    elif issue == "role":
        manifest["objects"][0]["role"] = "o3_input_unified"
    elif issue == "envelope":
        manifest["caveats"] = "not a list"
    elif issue == "inventory":
        manifest["inventory"]["object_count"] += 1
    elif issue == "role_counts":
        manifest["inventory"]["role_counts"] = {"o3_output": 99}
    elif issue == "limits":
        manifest["inventory"]["limits"] = {"max_objects": 1}
    elif issue == "missing_manifest":
        members = [(n, d) for n, d in members if n != "bundle.json"]
    members = [
        (name, json.dumps(manifest).encode() if name == "bundle.json" else data)
        for name, data in members
    ]
    with (
        pytest.warns(UserWarning)
        if issue == "duplicate"
        else __import__("contextlib").nullcontext()
    ):
        sha = _rewrite_test_archive(path, members)
    with pytest.raises(reader.OfflineBundleError):
        reader.load_offline_bundle(path, sha)


@pytest.mark.parametrize(
    "limit,value",
    [
        ("MAX_ENTRIES", 1),
        ("MAX_ARCHIVE_BYTES", 100),
        ("MAX_JSON_BYTES", 10),
        ("MAX_LIBRARY_ROWS", 0),
    ],
)
def test_archive_bounds(test_archive, monkeypatch, limit, value):
    path, sha, _, _, _ = test_archive
    monkeypatch.setattr(reader, limit, value)
    with pytest.raises(reader.OfflineBundleError):
        reader.load_offline_bundle(path, sha)


def test_library_validation(test_archive, tmp_path):
    source = test_archive[0].parent / "graank.parquet"
    invalid = tmp_path / "invalid.parquet"
    with duckdb.connect() as con:
        con.execute("CREATE TABLE p AS SELECT * FROM read_parquet(?)", [str(source)])
        con.execute("UPDATE p SET support=1.1")
        con.execute("COPY p TO ? (FORMAT PARQUET)", [str(invalid)])
    with pytest.raises(ValueError, match="support"):
        reader._read_library(invalid, "graank")


@pytest.mark.parametrize(
    "issue", ["valid", "hash", "duplicate", "missing", "unknown", "second_manifest"]
)
def test_zone_join(test_archive, issue):
    path, sha, _, _, zones_path = test_archive
    report = reader.load_offline_bundle(path, sha, zones_path)
    classes = report["zone_class_coverage"]["classes"]
    assert [row["koppen"] for row in classes] == ["Af", "Csa"]
    assert all(row["pattern_count"] == 2 for row in classes)
    assert classes[0]["validated_zones"] == ["zone_b"]
    if issue == "valid":
        return
    manifests = report["input_manifests"]
    zones = json.loads(zones_path.read_bytes())
    if issue == "hash":
        zones_path.write_bytes(zones_path.read_bytes() + b"\n")
    elif issue == "second_manifest":
        other = json.loads(json.dumps(next(iter(manifests.values()))))
        other["zone_artifact"]["sha256"] = "0" * 64
        manifests["second"] = other
    else:
        if issue == "duplicate":
            zones["features"].append(zones["features"][0])
        elif issue == "missing":
            zones["features"].pop()
        elif issue == "unknown":
            report["patterns"][0]["zones_validated"].append("unknown")
        zones_path.write_text(json.dumps(zones), encoding="utf-8")
        for manifest in manifests.values():
            manifest["zone_artifact"]["sha256"] = hashlib.sha256(
                zones_path.read_bytes()
            ).hexdigest()
    with pytest.raises(reader.OfflineBundleError):
        reader._join_zones(zones_path, manifests, report["patterns"])


def test_manifest_coverage(test_archive):
    path, sha, _, _, _ = test_archive
    report = reader.load_offline_bundle(path, sha)
    rows = report["manifest_reported_rows"]
    assert [(row["zone_id"], row["n_matched"]) for row in rows] == [("zone_a", 13), ("zone_b", 0)]
    assert all(row["evidence"] == "manifest-reported rows; inputs not reread" for row in rows)
    manifest = next(iter(report["input_manifests"].values()))
    manifest["outputs"][0]["n_matched"] = True
    with pytest.raises(reader.OfflineBundleError, match="Lignes rapportées"):
        reader._manifest_coverage({"manifest.json": manifest})


def test_cli_outputs(test_archive, tmp_path, capsys):
    path, sha, _, _, zones_path = test_archive
    first, second = tmp_path / "first", tmp_path / "second"
    for destination in (first, second):
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "src.cli.o3_bundle_report",
                str(path),
                "--expected-sha256",
                sha,
                "--output-dir",
                str(destination),
                "--zones-geojson",
                str(zones_path),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        assert json.loads(result.stdout)["patterns"] == 2
    for filename in reader.REPORT_FILES:
        assert (first / filename).read_bytes() == (second / filename).read_bytes()
    original = (first / "report.json").read_bytes()
    assert main([str(path), "--expected-sha256", sha, "--output-dir", str(first)]) == 1
    assert "existant" in capsys.readouterr().err
    assert (first / "report.json").read_bytes() == original
    with pytest.raises(SystemExit) as error:
        main([str(path), "--output-dir", str(first)])
    assert error.value.code == 2


def test_archive_snapshot(test_archive, tmp_path, monkeypatch):
    path, sha, manifest, members, _ = test_archive
    original_date = manifest["generated_at"]
    manifest["generated_at"] = "2026-10-05T00:00:00Z"
    rewritten = [
        (name, json.dumps(manifest).encode() if name == "bundle.json" else data)
        for name, data in members
    ]
    changed_path = tmp_path / "changed.zip"
    _rewrite_test_archive(changed_path, rewritten)
    changed_bytes = changed_path.read_bytes()
    original_zip = zipfile.ZipFile
    # Rewrite the source in place after SHA verification and before ZIP parsing.
    monkeypatch.setattr(
        reader.zipfile,
        "ZipFile",
        lambda source: (path.write_bytes(changed_bytes), original_zip(source))[1],
    )
    report = reader.load_offline_bundle(path, sha)
    assert hashlib.sha256(path.read_bytes()).hexdigest() != sha
    assert report["archive"]["sha256"] == sha
    assert report["source_bundle"]["generated_at"] == original_date


def test_corrupt_deflate(test_archive, tmp_path, monkeypatch, capsys):
    path, _, _, members, _ = test_archive
    name = members[0][0]
    with zipfile.ZipFile(path) as bundle:
        info = bundle.getinfo(name)
        assert info.compress_type == zipfile.ZIP_DEFLATED
    data = bytearray(path.read_bytes())
    name_length, extra_length = struct.unpack_from("<HH", data, info.header_offset + 26)
    offset = info.header_offset + 30 + name_length + extra_length
    # BTYPE=3 is reserved by RFC1951, giving a real zlib.error during member.read.
    data[offset] = (data[offset] & ~7) | 7
    path.write_bytes(data)
    sha = hashlib.sha256(data).hexdigest()
    with zipfile.ZipFile(path) as bundle, pytest.raises(zlib.error):
        bundle.read(name)
    parquet = Mock(side_effect=AssertionError("corrupt ZIP must never reach Parquet"))
    monkeypatch.setattr(reader, "_read_library", parquet)
    with pytest.raises(reader.OfflineBundleError, match="Bundle refusé"):
        reader.load_offline_bundle(path, sha)
    assert (
        main([str(path), "--expected-sha256", sha, "--output-dir", str(tmp_path / "output")]) == 1
    )
    assert json.loads(capsys.readouterr().err)["status"] == "refused"
    parquet.assert_not_called()


@pytest.mark.parametrize("phase", ["write", "publish_open", "publish_empty", "publish_partial"])
@pytest.mark.parametrize("failed_index", [1, 2])
def test_report_write_rollback(test_archive, tmp_path, monkeypatch, phase, failed_index):
    report = reader.load_offline_bundle(test_archive[0], test_archive[1])
    destination = tmp_path / "output"
    destination.mkdir()
    preserved = destination / "notes.txt"
    preserved.write_bytes(b"pre-existing unrelated file")
    original_open, original_copy = Path.open, reader.shutil.copyfileobj
    calls = 0

    def fail_write(path, *args, **kwargs):
        nonlocal calls
        is_stage = path.parent.name.startswith(".geoai-o3-report-")
        if (phase == "write" and is_stage) or (
            phase == "publish_open" and path.parent == destination
        ):
            calls += 1
            if calls == failed_index + 1:
                raise OSError("injected exclusive-open failure")
        return original_open(path, *args, **kwargs)

    def fail_publish(source, output):
        nonlocal calls
        calls += 1
        if calls == failed_index + 1:
            if phase == "publish_partial":
                output.write(b"partial published content")
                output.flush()
            raise OSError("injected publication-copy failure")
        return original_copy(source, output)

    with monkeypatch.context() as patch:
        if phase in ("write", "publish_open"):
            patch.setattr(Path, "open", fail_write)
        else:
            patch.setattr(reader.shutil, "copyfileobj", fail_publish)
        with pytest.raises(reader.OfflineBundleError, match="Écriture du rapport refusée"):
            reader.write_bundle_reports(report, destination)
    assert list(destination.iterdir()) == [preserved]
    assert preserved.read_bytes() == b"pre-existing unrelated file"
    paths = reader.write_bundle_reports(report, destination)
    assert all(path.is_file() and path.stat().st_size for path in paths)
    assert preserved.read_bytes() == b"pre-existing unrelated file"


def test_report_publication_conflict(test_archive, tmp_path, monkeypatch):
    report = reader.load_offline_bundle(test_archive[0], test_archive[1])
    destination = tmp_path / "output"
    original_open = Path.open

    def fail_publish(path, *args, **kwargs):
        if path.parent == destination and path.name == "report.md" and args and args[0] == "xb":
            # A concurrent creator wins before the exclusive publication open.
            with original_open(path, "xb") as concurrent:
                concurrent.write(b"concurrent existing report")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", fail_publish)
    with pytest.raises(reader.OfflineBundleError):
        reader.write_bundle_reports(report, destination)
    assert sorted(p.name for p in destination.iterdir()) == ["report.md"]
    assert (destination / "report.md").read_bytes() == b"concurrent existing report"


def test_report_filesystem_without_hardlinks(test_archive, tmp_path, monkeypatch):
    report = reader.load_offline_bundle(test_archive[0], test_archive[1])
    unavailable = Mock(side_effect=OSError(errno.EOPNOTSUPP, "hard links unsupported"))
    monkeypatch.setattr(reader.os, "link", unavailable)
    paths = reader.write_bundle_reports(report, tmp_path / "portable")
    assert {path.name for path in paths} == set(reader.REPORT_FILES)
    assert json.loads(paths[0].read_bytes()) == report
    assert all(path.stat().st_size for path in paths)
    unavailable.assert_not_called()
