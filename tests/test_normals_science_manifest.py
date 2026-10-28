"""Offline canonical normals manifest preflight: no registration or provider path."""

from __future__ import annotations

import hashlib
import json
import os
import socket
from calendar import monthrange
from datetime import date, timedelta
from pathlib import Path

import duckdb
import pytest
from shapely import Point, to_wkb

from src.cli import _normals_local_io as local_io
from src.cli import normals_science_manifest as cli
from src.config import settings

_VARIABLES = (
    "temperature_2m",
    "precipitation",
    "humidity",
    "evapotranspiration",
    "wind_speed",
    "solar_radiation",
)
_COLUMNS = (
    ("zone_id", "VARCHAR"),
    ("month", "INTEGER"),
    ("variable", "VARCHAR"),
    ("p10", "DOUBLE"),
    ("p50", "DOUBLE"),
    ("p90", "DOUBLE"),
    ("n", "INTEGER"),
    ("mean", "DOUBLE"),
    ("window_start", "VARCHAR"),
    ("window_end", "VARCHAR"),
    ("window_label", "VARCHAR"),
    ("lat", "DOUBLE"),
    ("lon", "DOUBLE"),
    ("p10_wet", "DOUBLE"),
    ("p50_wet", "DOUBLE"),
    ("p90_wet", "DOUBLE"),
    ("n_wet", "INTEGER"),
    ("wet_day_threshold_mm", "DOUBLE"),
    ("schema_version", "INTEGER"),
    ("geometry", "BLOB"),
)


def _rows() -> list[dict]:
    from src.tools.climate import _load_zones

    zone = _load_zones()["montpellier"]
    rows = []
    for variable in _VARIABLES:
        for month in range(1, 13):
            precipitation = variable == "precipitation"
            rows.append(
                {
                    "zone_id": "montpellier",
                    "month": month,
                    "variable": variable,
                    "p10": 1.0,
                    "p50": 2.0,
                    "p90": 3.0,
                    "n": sum(monthrange(year, month)[1] for year in range(1991, 2021)),
                    "mean": 2.0,
                    "window_start": "1991-01-01",
                    "window_end": "2020-12-31",
                    "window_label": "WMO_1991_2020",
                    "lat": zone["lat"],
                    "lon": zone["lon"],
                    "p10_wet": 1.0 if precipitation else None,
                    "p50_wet": 2.0 if precipitation else None,
                    "p90_wet": 3.0 if precipitation else None,
                    "n_wet": 0
                    if precipitation and month == 1
                    else (200 if precipitation else None),
                    "wet_day_threshold_mm": 1.0 if precipitation else None,
                    "schema_version": 2,
                    "geometry": bytes(to_wkb(Point(zone["lon"], zone["lat"]))),
                }
            )
    january = next(row for row in rows if row["variable"] == "precipitation" and row["month"] == 1)
    january["p10_wet"] = january["p50_wet"] = january["p90_wet"] = None
    return rows


def _write_source(path: Path, rows: list[dict]) -> str:
    connection = duckdb.connect(":memory:")
    try:
        definition = ", ".join(f"{name} {kind}" for name, kind in _COLUMNS)
        connection.execute(f"CREATE TABLE normals ({definition})")
        placeholders = ", ".join("?" for _ in _COLUMNS)
        connection.executemany(
            f"INSERT INTO normals VALUES ({placeholders})",
            [[row.get(name) for name, _ in _COLUMNS] for row in rows],
        )
        connection.execute("COPY normals TO ? (FORMAT PARQUET)", [str(path)])
    finally:
        connection.close()
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def case(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(settings, "ALLOW_LOCAL_PARQUET_SOURCE", True)
    monkeypatch.setattr(settings, "LOCAL_PARQUET_SOURCE_ROOT", str(tmp_path))
    source = tmp_path / "normals.parquet"
    digest = _write_source(source, _rows())
    provenance = tmp_path / "provenance.json"
    provenance.write_text(
        json.dumps(
            {
                "tenant_id": "operator-tenant",
                "folder_id": "operator-folder",
                "file_id": 123,
                "producer": "operator-reviewed-import",
                "created_at": "2026-09-29T10:00:00Z",
            }
        ),
        encoding="utf-8",
    )
    work = tmp_path / "work"
    work.mkdir()
    output = tmp_path / "manifest.json"
    arguments = [
        "--source",
        str(source),
        "--zone-id",
        "montpellier",
        "--expected-sha256",
        digest,
        "--provenance-json",
        str(provenance),
        "--work-root",
        str(work),
        "--output",
        str(output),
    ]
    return {
        "root": tmp_path,
        "source": source,
        "provenance": provenance,
        "work": work,
        "output": output,
        "digest": digest,
        "args": arguments,
    }


def _replace(case: dict, option: str, value: str) -> list[str]:
    arguments = list(case["args"])
    arguments[arguments.index(option) + 1] = value
    return arguments


def test_valid_v2_builds_only_a_local_science_manifest(case, monkeypatch):
    def forbidden_network(*_args, **_kwargs):
        raise AssertionError("provider or network access is not part of preflight")

    monkeypatch.setattr(socket.socket, "connect", forbidden_network)
    original = case["source"].read_bytes()
    assert cli.main(case["args"]) == 0
    manifest = json.loads(case["output"].read_text(encoding="utf-8"))
    assert manifest["schema_version"] == "dataset-manifest/v1"
    assert manifest["dataset_id"] == "science:normals:montpellier"
    assert manifest["dataset_kind"] == "science_dataset"
    assert manifest["artifact"] == {
        "format": "parquet",
        "key": "climate/normals/montpellier.parquet",
        "version": f"sha256:{case['digest']}",
    }
    assert manifest["table"]["row_count"] == 72
    assert manifest["spatial"]["geometry_column"] == "geometry"
    assert manifest["spatial"]["crs"] == "EPSG:4326"
    assert manifest["provenance"]["file_id"] == 123
    assert "evidence_id" not in manifest
    assert case["source"].read_bytes() == original
    assert list(case["work"].iterdir()) == []


def test_wrong_digest_refuses_before_inspection(case, monkeypatch):
    monkeypatch.setattr(cli, "inspect_spatial_parquet_fd", lambda *_a, **_kw: pytest.fail("parsed"))
    assert cli.main(_replace(case, "--expected-sha256", "0" * 64)) == 1
    assert not case["output"].exists()


@pytest.mark.parametrize("invalid", [{"file_id": 0}, {"producer": ""}, {"created_at": "today"}])
def test_invalid_provenance_refuses(case, invalid):
    provenance = json.loads(case["provenance"].read_text(encoding="utf-8"))
    provenance.update(invalid)
    case["provenance"].write_text(json.dumps(provenance), encoding="utf-8")
    assert cli.main(case["args"]) == 1
    assert not case["output"].exists()


def test_missing_provenance_argument_refuses(case):
    with pytest.raises(SystemExit):
        cli.main(
            case["args"][: case["args"].index("--provenance-json")]
            + case["args"][case["args"].index("--work-root") :]
        )


@pytest.mark.parametrize(
    "change",
    [
        lambda rows: rows[0].__setitem__("schema_version", 1),
        lambda rows: rows[0].__setitem__("zone_id", "nairobi"),
        lambda rows: rows.pop(),
        lambda rows: rows[0].__setitem__("month", 2),
        lambda rows: rows[0].__setitem__("p50", float("nan")),
        lambda rows: rows[0].__setitem__("window_label", "1991-2020"),
        lambda rows: rows[0].__setitem__("n", 1),
        lambda rows: next(row for row in rows if row["month"] == 2).__setitem__("n", 900),
        lambda rows: rows[0].__setitem__("lon", rows[0]["lon"] + 0.1),
        lambda rows: rows[0].__setitem__(
            "geometry", bytes(to_wkb(Point(rows[0]["lon"] + 0.1, rows[0]["lat"])))
        ),
        lambda rows: next(
            row for row in rows if row["variable"] == "precipitation" and row["month"] == 2
        ).__setitem__("p10_wet", 0.5),
        lambda rows: next(
            row for row in rows if row["variable"] == "precipitation" and row["month"] == 2
        ).__setitem__("n_wet", None),
    ],
    ids=[
        "v1",
        "wrong-zone",
        "missing-month",
        "duplicate-month",
        "nonfinite",
        "wrong-label",
        "sparse-n",
        "february-wrong-n",
        "wrong-lon",
        "wrong-wkb",
        "wet-percentile-below-threshold",
        "missing-n-wet",
    ],
)
def test_bad_canonical_grid_refuses(case, change):
    rows = _rows()
    change(rows)
    case["source"].unlink()
    digest = _write_source(case["source"], rows)
    assert cli.main(_replace(case, "--expected-sha256", digest)) == 1
    assert not case["output"].exists()


def test_zero_wet_days_is_valid(case):
    assert cli.main(case["args"]) == 0  # January precipitation has measured n_wet=0.


def test_unknown_or_client_zone_refuses(case, monkeypatch):
    assert cli.main(_replace(case, "--zone-id", "client-site-1")) == 1
    registry = case["root"] / "client-zones.json"
    registry.write_text(
        json.dumps({"features": [{"properties": {"zone_id": "client-site-1", "role": "client"}}]}),
        encoding="utf-8",
    )
    monkeypatch.setattr(cli, "_ZONES", registry)
    assert cli.main(_replace(case, "--zone-id", "client-site-1")) == 1
    assert not case["output"].exists()


def test_existing_output_and_symlink_are_never_overwritten(case):
    case["output"].write_text("existing", encoding="utf-8")
    assert cli.main(case["args"]) == 1
    assert case["output"].read_text(encoding="utf-8") == "existing"
    case["output"].unlink()
    case["output"].symlink_to(case["provenance"])
    assert cli.main(case["args"]) == 1
    assert case["provenance"].is_file()


def test_symlink_source_is_refused(case):
    linked = case["root"] / "linked.parquet"
    linked.symlink_to(case["source"])
    assert cli.main(_replace(case, "--source", str(linked))) == 1
    assert not case["output"].exists()


@pytest.mark.parametrize("option", ["--source", "--provenance-json", "--work-root", "--output"])
def test_remote_and_glob_paths_refuse(case, option):
    assert cli.main(_replace(case, option, "s3://bucket/normals.parquet")) == 1
    assert cli.main(_replace(case, option, str(case["root"] / "*.parquet"))) == 1


def test_outside_allowlist_and_input_output_collision_refuse(case, tmp_path):
    outside = tmp_path.parent / "outside-source.parquet"
    assert cli.main(_replace(case, "--source", str(outside))) == 1
    assert cli.main(_replace(case, "--output", str(case["provenance"]))) == 1


def test_local_access_must_be_enabled(case, monkeypatch):
    monkeypatch.setattr(settings, "ALLOW_LOCAL_PARQUET_SOURCE", False)
    assert cli.main(case["args"]) == 1
    assert not case["output"].exists()


def test_invalid_provenance_does_not_echo_input(case, capsys):
    case["provenance"].write_text('{"private-value": "do-not-publish-this"}', encoding="utf-8")
    assert cli.main(case["args"]) == 1
    captured = capsys.readouterr()
    assert "do-not-publish-this" not in captured.out + captured.err
    assert json.loads(captured.out)["error"] == "invalid manifest or provenance schema"


def test_diagnostics_redact_configured_credentials(case, monkeypatch, capsys):
    secret = "synthetic-test-credential-not-for-publication"
    monkeypatch.setattr(settings, "S3_SECRET_KEY", secret)

    def fail(_args):
        raise OSError(f"failed for {secret}")

    monkeypatch.setattr(cli, "run", fail)
    assert cli.main(case["args"]) == 1
    captured = capsys.readouterr()
    assert secret not in captured.out + captured.err
    assert "[REDACTED]" in captured.out


def test_atomic_publication_refuses_a_racing_writer(case, monkeypatch):
    original_link = local_io.os.link

    def publish_other_first(source, destination, **kwargs):
        descriptor = os.open(
            destination,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
            dir_fd=kwargs["dst_dir_fd"],
        )
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(b"other-writer")
        original_link(source, destination, **kwargs)

    monkeypatch.setattr(local_io.os, "link", publish_other_first)
    assert cli.main(case["args"]) == 1
    assert case["output"].read_bytes() == b"other-writer"
    assert list(case["root"].glob(".normals-manifest-*")) == []


def test_validation_uses_verified_snapshot_if_source_changes(case, monkeypatch):
    inspect = cli.inspect_spatial_parquet_fd

    def inspect_after_source_change(descriptor, **kwargs):
        case["source"].write_bytes(b"changed-after-snapshot")
        snapshot = Path(f"/proc/self/fd/{descriptor}")
        assert hashlib.sha256(snapshot.read_bytes()).hexdigest() == case["digest"]
        return inspect(descriptor, **kwargs)

    monkeypatch.setattr(cli, "inspect_spatial_parquet_fd", inspect_after_source_change)
    assert cli.main(case["args"]) == 0
    manifest = json.loads(case["output"].read_text(encoding="utf-8"))
    assert manifest["artifact"]["version"] == f"sha256:{case['digest']}"


def test_real_v2_producer_rows_are_accepted(case):
    from src.tools.climate import (
        _load_zones,
        _resolve_normals_window,
        _write_geoparquet,
        compute_monthly_normals,
    )

    start, end, label = _resolve_normals_window(None, None)
    zone = _load_zones()["montpellier"]
    first, last = date.fromisoformat(start), date.fromisoformat(end)
    daily = []
    for offset in range((last - first).days + 1):
        day = first + timedelta(days=offset)
        values = dict.fromkeys(_VARIABLES, 2.0)
        values["precipitation"] = 0.0 if day.month == 1 else 2.0
        daily.append({"date": day.isoformat(), **values})
    rows = compute_monthly_normals(
        daily,
        zone_id="montpellier",
        lat=zone["lat"],
        lon=zone["lon"],
        requested=list(_VARIABLES),
        window_start=start,
        window_end=end,
        window_label=label,
    )
    assert len(daily) == 10_958
    assert {row["n"] for row in rows if row["month"] == 2} == {848}
    assert all(row["window_label"] == "WMO_1991_2020" for row in rows)
    case["source"].unlink()
    _write_geoparquet(
        rows,
        columns=[(name, kind) for name, kind in _COLUMNS if name != "geometry"]
        + [("wkt", "VARCHAR")],
        dst=str(case["source"]),
        date_cols=(),
    )
    digest = hashlib.sha256(case["source"].read_bytes()).hexdigest()
    assert cli.main(_replace(case, "--expected-sha256", digest)) == 0
