import json
from datetime import date, timedelta
from pathlib import Path

import duckdb
import pytest

from src.cli import forecast_forward_archive as cli


def _row(zone: str, valid: str, horizon: int, variable: str) -> dict:
    return {
        "zone_id": zone,
        "origin_date": (date.fromisoformat(valid) - timedelta(days=horizon)).isoformat(),
        "valid_date": valid,
        "horizon_days": horizon,
        "variable": variable,
        "value": 1.0,
        "provider": cli.PROVIDER,
        "model": cli.MODEL,
    }


def _seed_artifacts(root: Path, valid: str, *, missing: str | None = None) -> None:
    run_dir = root / cli._run_id(valid)
    run_dir.mkdir(parents=True)
    for zone in cli.CANONICAL_ZONES:
        for horizon in cli.HORIZONS:
            key = cli._key(zone, valid, horizon)
            if key == missing:
                continue
            run_dir.joinpath(f"{key}.json").write_text(
                json.dumps(
                    {
                        "coverage": {
                            "zone_id": zone,
                            "provider": cli.PROVIDER,
                            "model": cli.MODEL,
                            "horizon_days": horizon,
                            "start_date": valid,
                            "end_date": valid,
                            "expected_days": 1,
                            "available_days": 1,
                            "status": "available",
                        },
                        "rows": [
                            _row(zone, valid, horizon, variable) for variable in cli.NORMALIZED_VARS
                        ],
                    },
                    sort_keys=True,
                ),
                encoding="utf-8",
            )


def _run_args(root: Path, valid: str = "2026-07-15") -> object:
    return cli.build_parser().parse_args(["run", "--valid-date", valid, "--output-dir", str(root)])


def _health_args(root: Path, end: str = "2026-07-15", days: int = 7) -> object:
    return cli.build_parser().parse_args(
        [
            "health",
            "--end-date",
            end,
            "--window-days",
            str(days),
            "--output-dir",
            str(root),
        ]
    )


def test_run_derives_date_run_id_and_writes_exact_gate(tmp_path, monkeypatch):
    captured = {}

    def fake_panel(args):
        captured.update(vars(args))
        _seed_artifacts(
            args.output_dir.parent if args.storage_mode == "local" else args.output_dir,
            args.start_date,
        )
        return {"entries": []}

    monkeypatch.setattr(cli, "run_panel", fake_panel)
    manifest = cli.run(_run_args(tmp_path))

    assert captured["run_id"] == "aifs-forward-20260715"
    assert captured["provider_model"] == [(cli.PROVIDER, cli.MODEL)]
    assert captured["horizon_days"] == [1, 3, 5, 7]
    assert captured["write_manifest"] is False
    assert manifest["valid_date"] == "2026-07-15"
    assert "issue_date" not in manifest
    assert manifest["counts"] == {
        "zones": 15,
        "horizons": 4,
        "variables": 6,
        "artifacts": 60,
        "rows": 360,
        "expected_artifacts": 60,
        "readable_artifacts": 60,
        "valid_artifacts": 60,
        "expected_rows": 360,
        "readable_rows": 360,
        "valid_rows": 360,
    }
    assert manifest["checks"]["ok"] is True


def test_run_nonzero_gate_and_health_never_calls_provider(tmp_path, monkeypatch):
    valid = "2026-07-15"
    missing = cli._key(cli.CANONICAL_ZONES[0], valid, 1)
    _seed_artifacts(tmp_path, valid, missing=missing)
    monkeypatch.setattr(cli, "run_panel", lambda _args: {"entries": []})
    manifest = cli._enhanced_manifest(
        valid_date=valid,
        run_id=cli._run_id(valid),
        storage_mode="local",
        output_dir=tmp_path / cli._run_id(valid),
        bucket="",
        s3_prefix="climate/forecast",
        existing=None,
    )
    assert manifest["checks"]["ok"] is False
    assert manifest["counts"]["expected_artifacts"] == 60
    assert manifest["counts"]["readable_artifacts"] == 59
    assert manifest["counts"]["valid_artifacts"] == 59
    assert manifest["counts"]["expected_rows"] == 360
    assert manifest["counts"]["readable_rows"] == 354
    assert manifest["counts"]["valid_rows"] == 354
    assert cli.main(["run", "--valid-date", valid, "--output-dir", str(tmp_path)]) == 1

    def provider_must_not_run(_args):
        raise AssertionError("health called provider")

    monkeypatch.setattr(cli, "run_panel", provider_must_not_run)
    result = cli.health(_health_args(tmp_path, valid, 7))
    assert result["checks"]["ok"] is False
    assert result["counts"]["missing_run"] == 6


def test_created_at_is_preserved_and_digest_changes_are_detected(tmp_path, monkeypatch):
    valid = "2026-07-15"

    def fake_panel(args):
        if not args.output_dir.exists():
            _seed_artifacts(tmp_path, valid)
        return {"entries": []}

    monkeypatch.setattr(cli, "run_panel", fake_panel)
    first = cli.run(_run_args(tmp_path))
    second = cli.run(_run_args(tmp_path))
    assert second["created_at"] == first["created_at"]
    assert second["artifacts"] == first["artifacts"]

    artifact = tmp_path / cli._run_id(valid) / f"{cli._key(cli.CANONICAL_ZONES[0], valid, 1)}.json"
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    payload["rows"][0]["value"] = 99.0
    artifact.write_text(json.dumps(payload), encoding="utf-8")
    health = cli.health(_health_args(tmp_path, valid, 7))
    assert health["days"][-1]["status"] == "invalid_manifest"


def test_rerun_rejects_digest_drift_and_preserves_prior_digest(tmp_path, monkeypatch):
    valid = "2026-07-15"

    def fake_panel(args):
        if not args.output_dir.exists():
            _seed_artifacts(tmp_path, valid)
        return {"entries": []}

    monkeypatch.setattr(cli, "run_panel", fake_panel)
    first = cli.run(_run_args(tmp_path))
    artifact = tmp_path / cli._run_id(valid) / f"{cli._key(cli.CANONICAL_ZONES[0], valid, 1)}.json"
    original_digest = first["artifacts"][0]["sha256"]
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    payload["rows"][0]["value"] = 99.0
    artifact.write_text(json.dumps(payload), encoding="utf-8")

    rerun = cli.run(_run_args(tmp_path))

    assert rerun["checks"]["ok"] is False
    assert rerun["artifacts"][0]["sha256"] == original_digest
    assert any("digest drifted" in error for error in rerun["errors"])
    assert cli.health(_health_args(tmp_path, valid, 7))["days"][-1]["status"] == "invalid_manifest"


def test_local_health_accepts_absolute_path_after_relative_run(tmp_path, monkeypatch):
    valid = "2026-07-15"
    monkeypatch.chdir(tmp_path)

    def fake_panel(args):
        _seed_artifacts(args.output_dir.parent, valid)
        return {"entries": []}

    monkeypatch.setattr(cli, "run_panel", fake_panel)
    manifest = cli.run(_run_args(Path("relative-archive")))

    assert all(Path(item["output"]).is_absolute() for item in manifest["artifacts"])
    result = cli.health(_health_args(tmp_path / "relative-archive", valid, 7))
    assert result["days"][-1]["status"] == "complete"


def test_health_distinguishes_partial_and_missing_run(tmp_path):
    valid = "2026-07-15"
    _seed_artifacts(tmp_path, valid)
    manifest = cli._enhanced_manifest(
        valid_date=valid,
        run_id=cli._run_id(valid),
        storage_mode="local",
        output_dir=tmp_path / cli._run_id(valid),
        bucket="",
        s3_prefix="climate/forecast",
        existing=None,
    )
    manifest["checks"]["ok"] = False
    manifest["counts"]["rows"] = 359
    (tmp_path / cli._run_id(valid) / "manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    result = cli.health(_health_args(tmp_path, valid, 7))
    assert result["days"][-1]["status"] == "partial"
    assert result["counts"]["missing_run"] == 6


def test_health_marks_corrupt_manifest_invalid_and_rejects_tampered_output(tmp_path):
    valid = "2026-07-15"
    _seed_artifacts(tmp_path, valid)
    manifest = cli._enhanced_manifest(
        valid_date=valid,
        run_id=cli._run_id(valid),
        storage_mode="local",
        output_dir=tmp_path / cli._run_id(valid),
        bucket="",
        s3_prefix="climate/forecast",
        existing=None,
    )
    manifest_path = tmp_path / cli._run_id(valid) / "manifest.json"
    expected_output = manifest["artifacts"][0]["output"]
    manifest["artifacts"][0]["output"] = str(tmp_path / "outside.json")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    result = cli.health(_health_args(tmp_path, valid, 7))
    assert result["days"][-1]["status"] == "invalid_manifest"

    manifest["artifacts"][0]["output"] = expected_output
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    artifact_path = Path(expected_output)
    artifact_path.unlink()
    result = cli.health(_health_args(tmp_path, valid, 7))
    assert result["days"][-1]["status"] == "partial"

    manifest_path.write_text("{not-json", encoding="utf-8")
    result = cli.health(_health_args(tmp_path, valid, 7))
    assert result["days"][-1]["status"] == "invalid_manifest"

    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    original_read_json = cli._read_json

    def access_denied(path):
        if path == manifest_path:
            raise PermissionError("permission denied")
        return original_read_json(path)

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(cli, "_read_json", access_denied)
    try:
        result = cli.health(_health_args(tmp_path, valid, 7))
    finally:
        monkeypatch.undo()
    assert result["days"][-1]["status"] == "invalid_manifest"


def test_s3_health_classifies_duckdb_style_storage_errors(tmp_path, monkeypatch):
    monkeypatch.setattr(
        cli,
        "_read_json",
        lambda _path: (_ for _ in ()).throw(duckdb.HTTPException("HTTP 404 Not Found")),
    )
    missing = cli._health_day(
        root=tmp_path,
        valid_date="2026-07-15",
        storage_mode="s3",
        bucket="deposium",
        s3_prefix="climate/forecast",
    )
    assert missing["status"] == "missing_run"
    assert cli._read_existing("s3://deposium/climate/runs/missing/manifest.json") is None

    monkeypatch.setattr(
        cli,
        "_read_json",
        lambda _path: (_ for _ in ()).throw(duckdb.HTTPException("HTTP 403 Access Denied")),
    )
    denied = cli._health_day(
        root=tmp_path,
        valid_date="2026-07-15",
        storage_mode="s3",
        bucket="deposium",
        s3_prefix="climate/forecast",
    )
    assert denied["status"] == "invalid_manifest"
    with pytest.raises(duckdb.HTTPException, match="Access Denied"):
        cli._read_existing("s3://deposium/climate/runs/forbidden/manifest.json")


@pytest.mark.parametrize("window_days", [7, 14])
def test_health_complete_window(tmp_path, window_days):
    end = date(2026, 7, 15)
    for offset in range(window_days):
        valid = (end - timedelta(days=window_days - offset - 1)).isoformat()
        _seed_artifacts(tmp_path, valid)
        manifest = cli._enhanced_manifest(
            valid_date=valid,
            run_id=cli._run_id(valid),
            storage_mode="local",
            output_dir=tmp_path / cli._run_id(valid),
            bucket="",
            s3_prefix="climate/forecast",
            existing=None,
        )
        (tmp_path / cli._run_id(valid) / "manifest.json").write_text(
            json.dumps(manifest), encoding="utf-8"
        )
    result = cli.health(_health_args(tmp_path, end.isoformat(), window_days))
    assert result["checks"]["ok"] is True
    assert result["counts"] == {
        "complete": window_days,
        "missing_run": 0,
        "partial": 0,
        "invalid_manifest": 0,
    }


def test_parser_rejects_non_supported_windows(tmp_path):
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(
            [
                "health",
                "--end-date",
                "2026-07-15",
                "--window-days",
                "8",
                "--output-dir",
                str(tmp_path),
            ]
        )


def test_non_finite_row_value_fails_exact_gate(tmp_path):
    valid = "2026-07-15"
    _seed_artifacts(tmp_path, valid)
    artifact = tmp_path / cli._run_id(valid) / f"{cli._key(cli.CANONICAL_ZONES[0], valid, 1)}.json"
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    payload["rows"][0]["value"] = True
    artifact.write_text(json.dumps(payload), encoding="utf-8")
    manifest = cli._enhanced_manifest(
        valid_date=valid,
        run_id=cli._run_id(valid),
        storage_mode="local",
        output_dir=tmp_path / cli._run_id(valid),
        bucket="",
        s3_prefix="climate/forecast",
        existing=None,
    )
    assert manifest["checks"]["finite_values"] is False
    assert manifest["checks"]["ok"] is False


def test_oversized_integer_is_invalid_instead_of_crashing(tmp_path):
    valid = "2026-07-15"
    _seed_artifacts(tmp_path, valid)
    artifact = tmp_path / cli._run_id(valid) / f"{cli._key(cli.CANONICAL_ZONES[0], valid, 1)}.json"
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    payload["rows"][0]["value"] = 10**400
    artifact.write_text(json.dumps(payload), encoding="utf-8")

    manifest = cli._enhanced_manifest(
        valid_date=valid,
        run_id=cli._run_id(valid),
        storage_mode="local",
        output_dir=tmp_path / cli._run_id(valid),
        bucket="",
        s3_prefix="climate/forecast",
        existing=None,
    )

    assert manifest["checks"]["finite_values"] is False
    assert manifest["checks"]["ok"] is False


def test_non_array_rows_writes_failed_manifest_instead_of_crashing(tmp_path):
    valid = "2026-07-15"
    _seed_artifacts(tmp_path, valid)
    artifact = tmp_path / cli._run_id(valid) / f"{cli._key(cli.CANONICAL_ZONES[0], valid, 1)}.json"
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    payload["rows"] = None
    artifact.write_text(json.dumps(payload), encoding="utf-8")

    manifest = cli._enhanced_manifest(
        valid_date=valid,
        run_id=cli._run_id(valid),
        storage_mode="local",
        output_dir=tmp_path / cli._run_id(valid),
        bucket="",
        s3_prefix="climate/forecast",
        existing=None,
    )

    affected = next(item for item in manifest["artifacts"] if item["output"] == str(artifact))
    assert affected["rows"] == 0
    assert manifest["counts"]["readable_rows"] == cli.EXPECTED_ROWS - len(cli.NORMALIZED_VARS)
    assert manifest["checks"]["ok"] is False
