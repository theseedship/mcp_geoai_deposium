"""Network-free tests for the multi-zone climate baseline panel runner."""

from __future__ import annotations

import asyncio
import contextlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from src.utils.s3_endpoint import S3EndpointConfigurationError

_PANEL_PATH = Path(__file__).resolve().parents[1] / "scripts" / "climate_baseline_panel.py"
_spec = importlib.util.spec_from_file_location("climate_baseline_panel", _PANEL_PATH)
panel = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(panel)

pytestmark = pytest.mark.usefixtures("allow_local_parquet_paths")


def _ok_payload(args: Any) -> dict[str, Any]:
    variables = getattr(args, "variables", None) or list(panel.NORMALIZED_VARS)
    start_date = getattr(args, "start_date", "1991-01-01")
    end_date = getattr(args, "end_date", "2020-12-31")
    for output in (args.era5_output, args.normals_output):
        if isinstance(output, str) and "://" not in output:
            path = Path(output)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("parquet\n")
    return {
        "ok": True,
        "zone_id": args.zone_id,
        "window": {
            "start_date": start_date,
            "end_date": end_date,
            "label": "wmo_1991_2020"
            if start_date == "1991-01-01" and end_date == "2020-12-31"
            else "custom",
        },
        "variables": variables,
        "storage_mode": getattr(args, "storage_mode", "local"),
        "duration_ms": 12,
        "requested_outputs": {"era5": args.era5_output, "normals": args.normals_output},
        "resolved_outputs": {"era5": args.era5_output, "normals": args.normals_output},
        "stages": {
            "era5_ingest": {
                "output": args.era5_output,
                "start_date": start_date,
                "end_date": end_date,
                "n_days": 10958,
                "pct_days": 100.0,
                "min_variable_pct_complete": 100.0,
            },
            "climate_percentiles": {
                "output": args.normals_output,
                "n_normals": 12 * len(variables),
                "n_months": 12,
                "n_variables": len(variables),
                "source": "era5_parquet",
            },
        },
        "gates": {
            "completeness": {
                "passed": True,
                "min_completeness_pct": getattr(args, "min_completeness_pct", 98.0),
            }
        },
    }


def test_panel_runs_all_zones_and_writes_per_zone_summaries(tmp_path: Path):
    calls: list[Any] = []
    sleeps: list[float] = []

    async def fake_runner(args: Any) -> dict[str, Any]:
        calls.append(args)
        return _ok_payload(args)

    async def fake_sleep(delay: float) -> None:
        sleeps.append(delay)

    args = panel.build_parser().parse_args(
        [
            "--zone-id",
            "bassin_du_lez",
            "fos_sur_mer",
            "--output-dir",
            tmp_path.as_posix(),
            "--overwrite",
            "--archive-chunk-days",
            "10958",
            "--zone-delay-s",
            "1.5",
        ]
    )

    payload = asyncio.run(panel.run_panel(args, runner=fake_runner, sleeper=fake_sleep))

    assert payload["ok"] is True
    assert payload["zone_ids"] == ["bassin_du_lez", "fos_sur_mer"]
    assert payload["n_zones_completed"] == 2
    assert payload["zone_delay_s"] == 1.5
    assert [c.zone_id for c in calls] == ["bassin_du_lez", "fos_sur_mer"]
    assert sleeps == [1.5]
    assert all(c.min_completeness_pct == 98.0 for c in calls)
    assert all(c.archive_chunk_days == 10958 for c in calls)
    assert calls[0].era5_output.endswith("bassin_du_lez-era5-1991-01-01_2020-12-31.parquet")
    assert (tmp_path / "panel-baseline-summary.json").exists()
    assert (tmp_path / "panel-baseline-manifest.json").exists()
    assert (tmp_path / "bassin_du_lez-baseline-1991-01-01_2020-12-31.json").exists()

    written = json.loads((tmp_path / "panel-baseline-summary.json").read_text())
    assert written["zones"][0]["gate_passed"] is True
    assert written["zones"][0]["era5"]["n_days"] == 10958
    manifest = json.loads((tmp_path / "panel-baseline-manifest.json").read_text())
    assert manifest["kind"] == "climate_baseline_panel_manifest"
    assert manifest["storage_mode"] == "local"
    assert manifest["artifacts"][0]["era5_output"].endswith(
        "bassin_du_lez-era5-1991-01-01_2020-12-31.parquet"
    )


def test_panel_s3_storage_mode_delegates_parquet_outputs_and_writes_manifest(tmp_path: Path):
    calls: list[Any] = []

    async def fake_runner(args: Any) -> dict[str, Any]:
        calls.append(args)
        era5 = f"s3://deposium/climate/era5/{args.zone_id}.parquet"
        normals = f"s3://deposium/climate/normals/{args.zone_id}.parquet"
        payload = _ok_payload(args)
        payload["resolved_outputs"] = {"era5": era5, "normals": normals}
        payload["stages"]["era5_ingest"]["output"] = era5
        payload["stages"]["climate_percentiles"]["output"] = normals
        return payload

    args = panel.build_parser().parse_args(
        [
            "--zone-id",
            "montpellier",
            "--storage-mode",
            "s3",
            "--output-dir",
            tmp_path.as_posix(),
        ]
    )

    payload = asyncio.run(
        panel.run_panel(args, runner=fake_runner, run_manifest_writer=lambda *_args: None)
    )

    assert payload["ok"] is True
    assert payload["storage_mode"] == "s3"
    assert calls[0].storage_mode == "s3"
    assert calls[0].era5_output is None
    assert calls[0].normals_output is None

    aggregate = json.loads((tmp_path / "panel-baseline-summary.json").read_text())
    assert aggregate["zones"][0]["requested_outputs"] == {"era5": None, "normals": None}
    assert aggregate["zones"][0]["era5"]["file_size_bytes"] is None

    manifest = json.loads((tmp_path / "panel-baseline-manifest.json").read_text())
    assert manifest["storage_mode"] == "s3"
    assert manifest["artifacts"] == [
        {
            "zone_id": "montpellier",
            "ok": True,
            "preflight_only": False,
            "gate_passed": True,
            "summary_path": (
                tmp_path / "montpellier-baseline-1991-01-01_2020-12-31.json"
            ).as_posix(),
            "planned_outputs": None,
            "era5_output": "s3://deposium/climate/era5/montpellier.parquet",
            "normals_output": "s3://deposium/climate/normals/montpellier.parquet",
        }
    ]


def test_panel_s3_storage_mode_writes_reproducibility_manifest(tmp_path: Path, monkeypatch):
    calls: list[Any] = []
    writes: list[tuple[str, dict[str, Any]]] = []

    async def fake_runner(args: Any) -> dict[str, Any]:
        calls.append(args)
        era5 = f"s3://deposium/climate/era5/{args.zone_id}.parquet"
        normals = f"s3://deposium/climate/normals/{args.zone_id}.parquet"
        payload = _ok_payload(args)
        payload["resolved_outputs"] = {"era5": era5, "normals": normals}
        payload["stages"]["era5_ingest"]["output"] = era5
        payload["stages"]["climate_percentiles"]["output"] = normals
        return payload

    def capture_run_manifest(uri: str, payload: dict[str, Any]) -> None:
        writes.append((uri, payload))

    monkeypatch.setattr(panel.settings, "S3_BUCKET", "deposium")
    args = panel.build_parser().parse_args(
        [
            "--zone-id",
            "montpellier",
            "--storage-mode",
            "s3",
            "--output-dir",
            tmp_path.as_posix(),
            "--variables",
            "temperature_2m",
            "temperature_2m",
            "precipitation",
            "--run-id",
            "test-run",
        ]
    )

    payload = asyncio.run(
        panel.run_panel(args, runner=fake_runner, run_manifest_writer=capture_run_manifest)
    )

    assert payload["ok"] is True
    assert payload["run_id"] == "test-run"
    assert payload["run_manifest_uri"] == ("s3://deposium/climate/runs/test-run/manifest.json")
    assert len(writes) == 1
    assert {uri for uri, _payload in writes} == {payload["run_manifest_uri"]}
    manifest = writes[-1][1]
    assert manifest["kind"] == "geoai_run_manifest"
    assert manifest["run_id"] == "test-run"
    assert manifest["tool_or_script"] == "scripts/climate_baseline_panel.py"
    assert manifest["parameters"]["storage_mode"] == "s3"
    assert manifest["parameters"]["zone_set"] == "explicit"
    assert manifest["parameters"]["zone_ids"] == ["montpellier"]
    assert manifest["parameters"]["variables"] == ["temperature_2m", "precipitation"]
    assert manifest["inputs"][0]["variables"] == ["temperature_2m", "precipitation"]
    assert manifest["zone_artifact"]["path"] == "zones/zones.geojson"
    assert len(manifest["zone_artifact"]["sha256"]) == 64
    assert payload["run_manifest_written"] is True
    assert manifest["checks"]["run_manifest_written"] is True
    assert manifest["outputs"] == [
        {
            "kind": "era5",
            "zone_id": "montpellier",
            "uri": "s3://deposium/climate/era5/montpellier.parquet",
        },
        {
            "kind": "normals",
            "zone_id": "montpellier",
            "uri": "s3://deposium/climate/normals/montpellier.parquet",
        },
    ]
    assert manifest["checks"]["ok"] is True
    assert manifest["errors"] == []

    local_manifest = json.loads((tmp_path / "panel-baseline-manifest.json").read_text())
    assert local_manifest["run_manifest_uri"] == payload["run_manifest_uri"]
    assert local_manifest["zone_artifact"]["sha256"] == manifest["zone_artifact"]["sha256"]


def test_panel_holdout_s3_routes_era5_only_to_holdout_prefix(tmp_path: Path, monkeypatch):
    calls: list[Any] = []
    writes: list[tuple[str, dict[str, Any]]] = []

    async def fake_runner(args: Any) -> dict[str, Any]:
        calls.append(args)
        return {
            "ok": True,
            "zone_id": args.zone_id,
            "window": {
                "start_date": args.start_date,
                "end_date": args.end_date,
                "label": "custom",
            },
            "variables": list(panel.NORMALIZED_VARS),
            "write_output": True,
            "dry_run": False,
            "skip_normals": True,
            "archive_chunk_days": args.archive_chunk_days,
            "storage_mode": args.storage_mode,
            "requested_outputs": {"era5": args.era5_output, "normals": args.normals_output},
            "resolved_outputs": {"era5": args.era5_output, "normals": args.normals_output},
            "duration_ms": 12,
            "stages": {
                "era5_ingest": {
                    "output": args.era5_output,
                    "start_date": args.start_date,
                    "end_date": args.end_date,
                    "n_days": 1826,
                    "pct_days": 100.0,
                    "min_variable_pct_complete": 100.0,
                }
            },
            "gates": {"completeness": {"passed": True, "min_completeness_pct": 98.0}},
            "notes": ["skip-normals: climate_percentiles was not run"],
        }

    monkeypatch.setattr(panel.settings, "S3_BUCKET", "deposium")
    args = panel.build_parser().parse_args(
        [
            "--zone-id",
            "montpellier",
            "--storage-mode",
            "s3",
            "--artifact-scope",
            "holdout",
            "--start-date",
            "2021-01-01",
            "--end-date",
            "2025-12-31",
            "--output-dir",
            tmp_path.as_posix(),
            "--run-id",
            "recent-holdout",
        ]
    )

    payload = asyncio.run(
        panel.run_panel(
            args,
            runner=fake_runner,
            run_manifest_writer=lambda uri, manifest: writes.append((uri, manifest)),
        )
    )

    holdout_era5 = "s3://deposium/climate/holdout/era5/2021_2025/montpellier.parquet"
    assert payload["ok"] is True
    assert payload["artifact_scope"] == "holdout"
    assert payload["artifact_window_label"] == "2021_2025"
    assert payload["skip_normals"] is True
    assert calls[0].skip_normals is True
    assert calls[0].era5_output == holdout_era5
    assert calls[0].normals_output is None
    assert payload["zones"][0]["era5"]["output"] == holdout_era5
    assert payload["zones"][0]["normals"]["output"] is None

    local_manifest = json.loads((tmp_path / "panel-baseline-manifest.json").read_text())
    assert local_manifest["artifact_scope"] == "holdout"
    assert local_manifest["artifacts"][0]["era5_output"] == holdout_era5
    assert local_manifest["artifacts"][0]["normals_output"] is None

    assert len(writes) == 1
    remote_manifest = writes[0][1]
    assert remote_manifest["parameters"]["artifact_scope"] == "holdout"
    assert remote_manifest["parameters"]["skip_normals"] is True
    assert remote_manifest["outputs"] == [
        {"kind": "era5", "zone_id": "montpellier", "uri": holdout_era5}
    ]


def test_panel_s3_run_manifest_skips_failed_zone_outputs(tmp_path: Path, monkeypatch):
    writes: list[tuple[str, dict[str, Any]]] = []

    async def fake_runner(args: Any) -> dict[str, Any]:
        era5 = f"s3://deposium/climate/era5/{args.zone_id}.parquet"
        normals = f"s3://deposium/climate/normals/{args.zone_id}.parquet"
        payload = _ok_payload(args)
        payload["ok"] = False
        payload["failed_stage"] = "climate_percentiles"
        payload["error"] = "normal computation failed"
        payload["resolved_outputs"] = {"era5": era5, "normals": normals}
        payload["stages"]["era5_ingest"]["output"] = era5
        payload["stages"]["climate_percentiles"] = {"error": "normal computation failed"}
        payload["gates"]["completeness"]["passed"] = False
        return payload

    monkeypatch.setattr(panel.settings, "S3_BUCKET", "deposium")
    args = panel.build_parser().parse_args(
        [
            "--zone-id",
            "montpellier",
            "--storage-mode",
            "s3",
            "--output-dir",
            tmp_path.as_posix(),
            "--run-id",
            "partial-failed-zone",
        ]
    )

    payload = asyncio.run(
        panel.run_panel(
            args,
            runner=fake_runner,
            run_manifest_writer=lambda uri, manifest: writes.append((uri, manifest)),
        )
    )

    assert payload["ok"] is False
    assert payload["failed_zone_ids"] == ["montpellier"]
    assert len(writes) == 1
    manifest = writes[0][1]
    assert manifest["checks"]["ok"] is False
    assert manifest["outputs"] == []
    assert manifest["errors"] == [
        {
            "zone_id": "montpellier",
            "failed_stage": "climate_percentiles",
            "error": "normal computation failed",
        }
    ]


def test_default_run_id_includes_zone_selection_fingerprint():
    created_at = "2026-06-24T12:00:00.000001Z"
    args_a = panel.build_parser().parse_args(
        ["--zone-id", "montpellier", "marseille", "--storage-mode", "s3"]
    )
    args_b = panel.build_parser().parse_args(
        ["--zone-id", "nairobi", "mombasa", "--storage-mode", "s3"]
    )

    run_id_a = panel._resolve_run_id(
        args_a,
        zone_ids=["montpellier", "marseille"],
        created_at=created_at,
    )
    run_id_b = panel._resolve_run_id(
        args_b,
        zone_ids=["nairobi", "mombasa"],
        created_at=created_at,
    )

    assert run_id_a != run_id_b
    assert "_2zones_" in run_id_a
    assert "_2zones_" in run_id_b
    assert run_id_a.endswith("_1991_01_01_2020_12_31")
    assert run_id_b.endswith("_1991_01_01_2020_12_31")


def test_panel_rejects_unsafe_run_id_before_writing(tmp_path: Path, monkeypatch):
    calls: list[str] = []
    writes: list[tuple[str, dict[str, Any]]] = []

    async def fake_runner(args: Any) -> dict[str, Any]:
        calls.append(args.zone_id)
        return _ok_payload(args)

    def capture_run_manifest(uri: str, payload: dict[str, Any]) -> None:
        writes.append((uri, payload))

    monkeypatch.setattr(panel.settings, "S3_BUCKET", "deposium")
    args = panel.build_parser().parse_args(
        [
            "--zone-id",
            "montpellier",
            "--storage-mode",
            "s3",
            "--output-dir",
            tmp_path.as_posix(),
            "--run-id",
            "bad..id",
        ]
    )

    try:
        asyncio.run(
            panel.run_panel(args, runner=fake_runner, run_manifest_writer=capture_run_manifest)
        )
    except ValueError as exc:
        assert "--run-id must not contain consecutive dots" in str(exc)
    else:  # pragma: no cover - defensive assertion message
        raise AssertionError("unsafe run_id should be rejected")

    assert calls == []
    assert writes == []


def test_panel_manifest_write_error_is_recorded_after_all_zones(tmp_path: Path, monkeypatch):
    calls: list[str] = []
    writes: list[tuple[str, dict[str, Any]]] = []

    async def fake_runner(args: Any) -> dict[str, Any]:
        calls.append(args.zone_id)
        payload = _ok_payload(args)
        payload["resolved_outputs"] = {
            "era5": f"s3://deposium/climate/era5/{args.zone_id}.parquet",
            "normals": f"s3://deposium/climate/normals/{args.zone_id}.parquet",
        }
        payload["stages"]["era5_ingest"]["output"] = payload["resolved_outputs"]["era5"]
        payload["stages"]["climate_percentiles"]["output"] = payload["resolved_outputs"]["normals"]
        return payload

    def flaky_run_manifest(uri: str, payload: dict[str, Any]) -> None:
        writes.append((uri, payload))
        raise RuntimeError("transient rustfs outage")

    monkeypatch.setattr(panel.settings, "S3_BUCKET", "deposium")
    args = panel.build_parser().parse_args(
        [
            "--zone-id",
            "montpellier",
            "marseille",
            "--storage-mode",
            "s3",
            "--output-dir",
            tmp_path.as_posix(),
            "--run-id",
            "test-run",
        ]
    )

    payload = asyncio.run(
        panel.run_panel(args, runner=fake_runner, run_manifest_writer=flaky_run_manifest)
    )

    assert payload["ok"] is False
    assert payload["zone_ok"] is True
    assert payload["run_manifest_written"] is False
    assert calls == ["montpellier", "marseille"]
    assert len(writes) == 1
    assert writes[0][1]["checks"]["n_zones_completed"] == 2
    assert len(payload["run_manifest_errors"]) == 1
    assert payload["run_manifest_errors"][0]["stage"] == "run_manifest_write"
    assert "transient rustfs outage" in payload["run_manifest_errors"][0]["error"]
    local_manifest = json.loads((tmp_path / "panel-baseline-manifest.json").read_text())
    assert local_manifest["run_manifest_written"] is False
    assert local_manifest["run_manifest_errors"] == payload["run_manifest_errors"]


def test_run_reports_failed_run_manifest_write(tmp_path: Path, monkeypatch, capsys):
    async def fake_runner(args: Any) -> dict[str, Any]:
        payload = _ok_payload(args)
        payload["resolved_outputs"] = {
            "era5": f"s3://deposium/climate/era5/{args.zone_id}.parquet",
            "normals": f"s3://deposium/climate/normals/{args.zone_id}.parquet",
        }
        payload["stages"]["era5_ingest"]["output"] = payload["resolved_outputs"]["era5"]
        payload["stages"]["climate_percentiles"]["output"] = payload["resolved_outputs"]["normals"]
        return payload

    def fail_run_manifest(_uri: str, _payload: dict[str, Any]) -> None:
        raise RuntimeError("rustfs unavailable")

    monkeypatch.setattr(panel.settings, "S3_BUCKET", "deposium")
    args = panel.build_parser().parse_args(
        [
            "--zone-id",
            "montpellier",
            "--storage-mode",
            "s3",
            "--output-dir",
            tmp_path.as_posix(),
            "--run-id",
            "test-run",
        ]
    )

    exit_code = panel.run(args, runner=fake_runner, run_manifest_writer=fail_run_manifest)

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "Wrote climate baseline panel manifest" in captured.out
    assert "Wrote climate baseline run manifest" not in captured.out
    assert "Could not write climate baseline run manifest" in captured.err
    assert "s3://deposium/climate/runs/test-run/manifest.json" in captured.err


def test_code_ref_falls_back_to_railway_env_without_git(monkeypatch):
    monkeypatch.setattr(panel, "_git_output", lambda *_args, **_kwargs: None)
    monkeypatch.setenv("RAILWAY_GIT_COMMIT_SHA", "abc123")
    monkeypatch.setenv("RAILWAY_GIT_BRANCH", "main")

    code_ref = panel._code_ref()

    assert code_ref == {
        "repo": "deposium_geoai",
        "commit": "abc123",
        "branch": "main",
        "dirty": None,
    }


def test_code_ref_marks_clean_git_checkout_not_dirty(monkeypatch):
    def fake_git_output(*args: str, **kwargs: Any) -> str | None:
        if args == ("status", "--porcelain"):
            return "" if kwargs.get("empty_ok") else None
        if args == ("rev-parse", "HEAD"):
            return "abc123"
        if args == ("rev-parse", "--abbrev-ref", "HEAD"):
            return "main"
        return None

    monkeypatch.setattr(panel, "_git_output", fake_git_output)

    assert panel._code_ref() == {
        "repo": "deposium_geoai",
        "commit": "abc123",
        "branch": "main",
        "dirty": False,
    }


def test_full_panel_zone_set_loads_the_b1_artifact():
    zone_ids = panel._load_full_panel_zone_ids()

    assert len(zone_ids) == 15
    assert len(set(zone_ids)) == 15
    assert zone_ids[:4] == ["montpellier", "marseille", "nairobi", "mombasa"]
    assert zone_ids[-3:] == ["montpellier_metropole", "bassin_du_lez", "fos_sur_mer"]


def test_full_panel_zone_set_runs_all_artifact_zones(tmp_path: Path):
    calls: list[str] = []

    async def fake_runner(args: Any) -> dict[str, Any]:
        calls.append(args.zone_id)
        return _ok_payload(args)

    args = panel.build_parser().parse_args(
        [
            "--zone-set",
            "full-panel",
            "--output-dir",
            tmp_path.as_posix(),
        ]
    )

    payload = asyncio.run(panel.run_panel(args, runner=fake_runner))

    assert payload["ok"] is True
    assert payload["zone_set"] == "full-panel"
    assert payload["n_zones_requested"] == 15
    assert payload["n_zones_completed"] == 15
    assert calls == panel._load_full_panel_zone_ids()
    assert (tmp_path / "panel-baseline-summary.json").exists()


def test_preflight_full_panel_writes_paths_without_calling_runner_or_sleeping(tmp_path: Path):
    calls: list[str] = []
    sleeps: list[float] = []

    async def fake_runner(args: Any) -> dict[str, Any]:
        calls.append(args.zone_id)
        raise AssertionError("preflight must not call the climate runner")

    async def fake_sleep(delay: float) -> None:
        sleeps.append(delay)
        raise AssertionError("preflight must not sleep between zones")

    args = panel.build_parser().parse_args(
        [
            "--zone-set",
            "full-panel",
            "--output-dir",
            tmp_path.as_posix(),
            "--preflight-only",
            "--zone-delay-s",
            "90",
        ]
    )

    payload = asyncio.run(panel.run_panel(args, runner=fake_runner, sleeper=fake_sleep))

    assert payload["ok"] is True
    assert payload["preflight_only"] is True
    assert payload["zone_set"] == "full-panel"
    assert payload["n_zones_requested"] == 15
    assert payload["n_zones_completed"] == 15
    assert calls == []
    assert sleeps == []

    aggregate = json.loads((tmp_path / "panel-baseline-summary.json").read_text())
    first = aggregate["zones"][0]
    assert first["zone_id"] == "montpellier"
    assert first["preflight_only"] is True
    assert first["summary_path"].endswith("montpellier-baseline-1991-01-01_2020-12-31.json")
    assert first["requested_outputs"]["era5"].endswith(
        "montpellier-era5-1991-01-01_2020-12-31.parquet"
    )
    assert first["requested_outputs"]["normals"].endswith(
        "montpellier-normals-1991-01-01_2020-12-31.parquet"
    )
    manifest = json.loads((tmp_path / "panel-baseline-manifest.json").read_text())
    first_artifact = manifest["artifacts"][0]
    assert first_artifact["preflight_only"] is True
    assert first_artifact["era5_output"] is None
    assert first_artifact["normals_output"] is None
    assert first_artifact["planned_outputs"]["era5"].endswith(
        "montpellier-era5-1991-01-01_2020-12-31.parquet"
    )


def test_preflight_rejects_unknown_safe_zone_without_calling_runner(tmp_path: Path):
    calls: list[str] = []

    async def fake_runner(args: Any) -> dict[str, Any]:
        calls.append(args.zone_id)
        raise AssertionError("preflight must not call the climate runner")

    args = panel.build_parser().parse_args(
        [
            "--zone-id",
            "atlantis",
            "--output-dir",
            tmp_path.as_posix(),
            "--preflight-only",
        ]
    )

    payload = asyncio.run(panel.run_panel(args, runner=fake_runner))

    assert payload["ok"] is False
    assert payload["failed_zone_ids"] == ["atlantis"]
    assert calls == []
    zone = payload["zones"][0]
    assert zone["failed_stage"] == "preflight"
    assert "Unknown zone_id" in zone["error"]


def test_preflight_rejects_generated_output_paths_live_cli_would_reject(tmp_path: Path):
    calls: list[str] = []

    async def fake_runner(args: Any) -> dict[str, Any]:
        calls.append(args.zone_id)
        raise AssertionError("preflight must not call the climate runner")

    output_dir = tmp_path / "path with spaces"
    args = panel.build_parser().parse_args(
        [
            "--zone-id",
            "montpellier",
            "--output-dir",
            output_dir.as_posix(),
            "--preflight-only",
        ]
    )

    payload = asyncio.run(panel.run_panel(args, runner=fake_runner))

    assert payload["ok"] is False
    assert payload["failed_zone_ids"] == ["montpellier"]
    assert calls == []
    zone = payload["zones"][0]
    assert zone["failed_stage"] == "preflight"
    assert "Invalid era5 output destination before Archive fetch" in zone["error"]


def test_preflight_s3_storage_mode_requires_bucket_before_runner(tmp_path: Path, monkeypatch):
    calls: list[str] = []
    writes: list[tuple[str, dict[str, Any]]] = []

    async def fake_runner(args: Any) -> dict[str, Any]:
        calls.append(args.zone_id)
        raise AssertionError("preflight must not call the climate runner")

    monkeypatch.setattr("src.cli.climate_baseline._default_output_uri", lambda *_args: None)
    args = panel.build_parser().parse_args(
        [
            "--zone-id",
            "montpellier",
            "--storage-mode",
            "s3",
            "--output-dir",
            tmp_path.as_posix(),
            "--preflight-only",
        ]
    )

    def capture_run_manifest(uri: str, payload: dict[str, Any]) -> None:
        writes.append((uri, payload))

    payload = asyncio.run(
        panel.run_panel(args, runner=fake_runner, run_manifest_writer=capture_run_manifest)
    )

    assert payload["ok"] is False
    assert payload["run_manifest_uri"] is None
    assert payload["run_manifest_written"] is None
    assert writes == []
    assert calls == []
    zone = payload["zones"][0]
    assert zone["requested_outputs"] == {"era5": None, "normals": None}
    assert zone["resolved_outputs"] == {"era5": None, "normals": None}
    assert "Missing output destination" in zone["error"]


def test_preflight_holdout_zone_entry_reports_skip_normals(tmp_path: Path, monkeypatch):
    async def fake_runner(args: Any) -> dict[str, Any]:
        raise AssertionError("preflight must not call the climate runner")

    monkeypatch.setattr(panel.settings, "S3_BUCKET", "deposium")
    args = panel.build_parser().parse_args(
        [
            "--zone-id",
            "montpellier",
            "--storage-mode",
            "s3",
            "--artifact-scope",
            "holdout",
            "--start-date",
            "2021-01-01",
            "--end-date",
            "2021-12-31",
            "--output-dir",
            tmp_path.as_posix(),
            "--preflight-only",
        ]
    )

    payload = asyncio.run(panel.run_panel(args, runner=fake_runner))

    holdout_era5 = "s3://deposium/climate/holdout/era5/2021_2021/montpellier.parquet"
    assert payload["ok"] is True
    assert payload["skip_normals"] is True
    assert payload["zones"][0]["skip_normals"] is True
    assert payload["zones"][0]["resolved_outputs"] == {"era5": holdout_era5, "normals": None}

    manifest = json.loads((tmp_path / "panel-baseline-manifest.json").read_text())
    assert manifest["skip_normals"] is True
    assert manifest["artifacts"][0]["planned_outputs"] == {"era5": holdout_era5, "normals": None}


def test_panel_summary_is_written_incrementally_before_zone_sleep(tmp_path: Path):
    async def fake_runner(args: Any) -> dict[str, Any]:
        return _ok_payload(args)

    async def fake_sleep(delay: float) -> None:
        raise RuntimeError("operator interrupted during delay")

    args = panel.build_parser().parse_args(
        [
            "--zone-id",
            "montpellier",
            "marseille",
            "--output-dir",
            tmp_path.as_posix(),
            "--zone-delay-s",
            "90",
        ]
    )

    try:
        asyncio.run(panel.run_panel(args, runner=fake_runner, sleeper=fake_sleep))
    except RuntimeError as exc:
        assert str(exc) == "operator interrupted during delay"
    else:  # pragma: no cover - defensive assertion message
        raise AssertionError("sleep interruption should propagate")

    aggregate = json.loads((tmp_path / "panel-baseline-summary.json").read_text())
    assert aggregate["ok"] is False
    assert aggregate["n_zones_requested"] == 2
    assert aggregate["n_zones_completed"] == 1
    assert aggregate["completed_zone_ids"] == ["montpellier"]
    assert aggregate["zones"][0]["summary_path"].endswith(
        "montpellier-baseline-1991-01-01_2020-12-31.json"
    )


def test_stop_on_rate_limit_writes_partial_panel_and_stops(tmp_path: Path):
    calls: list[str] = []

    async def fake_runner(args: Any) -> dict[str, Any]:
        calls.append(args.zone_id)
        if args.zone_id == "riyadh":
            return {
                "ok": False,
                "zone_id": args.zone_id,
                "duration_ms": 12,
                "failed_stage": "era5_ingest",
                "error": "Client error '429 Too Many Requests'",
                "stages": {},
            }
        return _ok_payload(args)

    args = panel.build_parser().parse_args(
        [
            "--zone-id",
            "montpellier",
            "riyadh",
            "seoul",
            "--output-dir",
            tmp_path.as_posix(),
            "--stop-on-rate-limit",
        ]
    )

    payload = asyncio.run(panel.run_panel(args, runner=fake_runner))

    assert payload["ok"] is False
    assert payload["stop_reason"] == "rate_limit"
    assert payload["failed_zone_ids"] == ["riyadh"]
    assert calls == ["montpellier", "riyadh"]
    assert payload["completed_zone_ids"] == ["montpellier", "riyadh"]

    aggregate = json.loads((tmp_path / "panel-baseline-summary.json").read_text())
    assert aggregate["stop_on_rate_limit"] is True
    assert aggregate["stop_reason"] == "rate_limit"
    assert aggregate["n_zones_completed"] == 2
    assert not (tmp_path / "seoul-baseline-1991-01-01_2020-12-31.json").exists()


def test_failed_zone_manifest_keeps_planned_outputs(tmp_path: Path):
    async def fake_runner(_args: Any) -> dict[str, Any]:
        raise RuntimeError("archive unavailable")

    args = panel.build_parser().parse_args(
        [
            "--zone-id",
            "montpellier",
            "--output-dir",
            tmp_path.as_posix(),
        ]
    )

    payload = asyncio.run(panel.run_panel(args, runner=fake_runner))

    assert payload["ok"] is False
    zone = payload["zones"][0]
    assert zone["resolved_outputs"]["era5"].endswith(
        "montpellier-era5-1991-01-01_2020-12-31.parquet"
    )
    assert zone["era5"]["output"] is None

    manifest = json.loads((tmp_path / "panel-baseline-manifest.json").read_text())
    artifact = manifest["artifacts"][0]
    assert artifact["ok"] is False
    assert artifact["planned_outputs"] == zone["resolved_outputs"]
    assert artifact["era5_output"] is None
    assert artifact["normals_output"] is None


def test_summarize_existing_ignores_past_rate_limit_for_full_aggregate(tmp_path: Path):
    riyadh_path = panel._zone_summary_path(
        tmp_path,
        "riyadh",
        "1991-01-01",
        "2020-12-31",
        index=0,
    )
    riyadh_path.write_text(
        json.dumps(
            {
                "ok": False,
                "zone_id": "riyadh",
                "failed_stage": "era5_ingest",
                "error": "Client error '429 Too Many Requests'",
                "stages": {"era5_ingest": {"error": "429"}},
            }
        )
        + "\n"
    )
    seoul_args = type(
        "Args",
        (),
        {
            "zone_id": "seoul",
            "era5_output": (tmp_path / "seoul-era5.parquet").as_posix(),
            "normals_output": (tmp_path / "seoul-normals.parquet").as_posix(),
        },
    )()
    seoul_path = panel._zone_summary_path(
        tmp_path,
        "seoul",
        "1991-01-01",
        "2020-12-31",
        index=1,
    )
    seoul_path.write_text(json.dumps(_ok_payload(seoul_args)) + "\n")

    async def fake_runner(args: Any) -> dict[str, Any]:
        raise AssertionError("summarize-existing must not call the climate runner")

    args = panel.build_parser().parse_args(
        [
            "--zone-id",
            "riyadh",
            "seoul",
            "--output-dir",
            tmp_path.as_posix(),
            "--summarize-existing",
            "--stop-on-rate-limit",
        ]
    )

    payload = asyncio.run(panel.run_panel(args, runner=fake_runner))

    assert payload["ok"] is False
    assert payload["stop_reason"] is None
    assert payload["completed_zone_ids"] == ["riyadh", "seoul"]
    assert payload["failed_zone_ids"] == ["riyadh"]


def test_summarize_existing_reads_zone_summaries_without_runner_or_sleep(tmp_path: Path):
    calls: list[str] = []
    sleeps: list[float] = []

    for index, zone_id in enumerate(["montpellier", "marseille"]):
        zone_args = type(
            "Args",
            (),
            {
                "zone_id": zone_id,
                "era5_output": (tmp_path / f"{zone_id}-era5.parquet").as_posix(),
                "normals_output": (tmp_path / f"{zone_id}-normals.parquet").as_posix(),
            },
        )()
        summary_path = panel._zone_summary_path(
            tmp_path,
            zone_id,
            "1991-01-01",
            "2020-12-31",
            index=index,
        )
        summary_path.write_text(json.dumps(_ok_payload(zone_args)) + "\n")

    async def fake_runner(args: Any) -> dict[str, Any]:
        calls.append(args.zone_id)
        raise AssertionError("summarize-existing must not call the climate runner")

    async def fake_sleep(delay: float) -> None:
        sleeps.append(delay)
        raise AssertionError("summarize-existing must not sleep between zones")

    args = panel.build_parser().parse_args(
        [
            "--zone-id",
            "montpellier",
            "marseille",
            "--output-dir",
            tmp_path.as_posix(),
            "--summarize-existing",
            "--zone-delay-s",
            "90",
        ]
    )

    payload = asyncio.run(panel.run_panel(args, runner=fake_runner, sleeper=fake_sleep))

    assert payload["ok"] is True
    assert payload["summarize_existing"] is True
    assert payload["completed_zone_ids"] == ["montpellier", "marseille"]
    assert calls == []
    assert sleeps == []

    aggregate = json.loads((tmp_path / "panel-baseline-summary.json").read_text())
    assert aggregate["ok"] is True
    assert aggregate["n_zones_requested"] == 2
    assert aggregate["n_zones_completed"] == 2


def test_summarize_existing_preserves_existing_summary_variables(tmp_path: Path):
    subset = ["temperature_2m", "precipitation"]
    for index, zone_id in enumerate(["montpellier", "marseille"]):
        zone_args = type(
            "Args",
            (),
            {
                "zone_id": zone_id,
                "variables": subset,
                "era5_output": (tmp_path / f"{zone_id}-era5.parquet").as_posix(),
                "normals_output": (tmp_path / f"{zone_id}-normals.parquet").as_posix(),
            },
        )()
        summary_path = panel._zone_summary_path(
            tmp_path,
            zone_id,
            "1991-01-01",
            "2020-12-31",
            index=index,
        )
        summary_path.write_text(json.dumps(_ok_payload(zone_args)) + "\n")

    async def fake_runner(args: Any) -> dict[str, Any]:
        raise AssertionError("summarize-existing must not call the climate runner")

    args = panel.build_parser().parse_args(
        [
            "--zone-id",
            "montpellier",
            "marseille",
            "--output-dir",
            tmp_path.as_posix(),
            "--summarize-existing",
        ]
    )

    payload = asyncio.run(panel.run_panel(args, runner=fake_runner))

    assert payload["ok"] is True
    assert payload["variables"] == subset
    local_manifest = json.loads((tmp_path / "panel-baseline-manifest.json").read_text())
    assert local_manifest["variables"] == subset


def test_summarize_existing_rejects_mismatched_summary_variables(tmp_path: Path):
    variables_by_zone = {
        "montpellier": ["temperature_2m"],
        "marseille": ["precipitation"],
    }
    for index, (zone_id, variables) in enumerate(variables_by_zone.items()):
        zone_args = type(
            "Args",
            (),
            {
                "zone_id": zone_id,
                "variables": variables,
                "era5_output": (tmp_path / f"{zone_id}-era5.parquet").as_posix(),
                "normals_output": (tmp_path / f"{zone_id}-normals.parquet").as_posix(),
            },
        )()
        summary_path = panel._zone_summary_path(
            tmp_path,
            zone_id,
            "1991-01-01",
            "2020-12-31",
            index=index,
        )
        summary_path.write_text(json.dumps(_ok_payload(zone_args)) + "\n")

    async def fake_runner(args: Any) -> dict[str, Any]:
        raise AssertionError("summarize-existing must not call the climate runner")

    args = panel.build_parser().parse_args(
        [
            "--zone-id",
            "montpellier",
            "marseille",
            "--output-dir",
            tmp_path.as_posix(),
            "--summarize-existing",
        ]
    )

    payload = asyncio.run(panel.run_panel(args, runner=fake_runner))

    assert payload["ok"] is False
    assert payload["failed_zone_ids"] == ["marseille"]
    assert "variables" in payload["zones"][1]["error"]


def test_summarize_existing_uses_variables_from_successful_summaries(tmp_path: Path):
    failed_path = panel._zone_summary_path(
        tmp_path,
        "riyadh",
        "1991-01-01",
        "2020-12-31",
        index=0,
    )
    failed_path.write_text(
        json.dumps(
            {
                "ok": False,
                "zone_id": "riyadh",
                "variables": list(panel.NORMALIZED_VARS),
                "failed_stage": "era5_ingest",
                "error": "Client error '429 Too Many Requests'",
                "stages": {"era5_ingest": {"error": "429"}},
            }
        )
        + "\n"
    )

    subset = ["temperature_2m"]
    success_args = type(
        "Args",
        (),
        {
            "zone_id": "seoul",
            "variables": subset,
            "era5_output": (tmp_path / "seoul-era5.parquet").as_posix(),
            "normals_output": (tmp_path / "seoul-normals.parquet").as_posix(),
        },
    )()
    success_path = panel._zone_summary_path(
        tmp_path,
        "seoul",
        "1991-01-01",
        "2020-12-31",
        index=1,
    )
    success_path.write_text(json.dumps(_ok_payload(success_args)) + "\n")

    async def fake_runner(args: Any) -> dict[str, Any]:
        raise AssertionError("summarize-existing must not call the climate runner")

    args = panel.build_parser().parse_args(
        [
            "--zone-id",
            "riyadh",
            "seoul",
            "--output-dir",
            tmp_path.as_posix(),
            "--summarize-existing",
        ]
    )

    payload = asyncio.run(panel.run_panel(args, runner=fake_runner))

    assert payload["ok"] is False
    assert payload["failed_zone_ids"] == ["riyadh"]
    assert payload["zones"][1]["ok"] is True
    assert payload["variables"] == subset
    local_manifest = json.loads((tmp_path / "panel-baseline-manifest.json").read_text())
    assert local_manifest["variables"] == subset


def test_summarize_existing_s3_rejects_local_zone_summaries(tmp_path: Path, monkeypatch):
    zone_args = type(
        "Args",
        (),
        {
            "zone_id": "montpellier",
            "era5_output": (tmp_path / "montpellier-era5.parquet").as_posix(),
            "normals_output": (tmp_path / "montpellier-normals.parquet").as_posix(),
        },
    )()
    summary_path = panel._zone_summary_path(
        tmp_path,
        "montpellier",
        "1991-01-01",
        "2020-12-31",
        index=0,
    )
    summary_path.write_text(json.dumps(_ok_payload(zone_args)) + "\n")
    writes: list[tuple[str, dict[str, Any]]] = []

    async def fake_runner(args: Any) -> dict[str, Any]:
        raise AssertionError("summarize-existing must not call the climate runner")

    monkeypatch.setattr(panel.settings, "S3_BUCKET", "deposium")
    args = panel.build_parser().parse_args(
        [
            "--zone-id",
            "montpellier",
            "--storage-mode",
            "s3",
            "--output-dir",
            tmp_path.as_posix(),
            "--summarize-existing",
            "--run-id",
            "storage-mode-check",
        ]
    )

    payload = asyncio.run(
        panel.run_panel(
            args,
            runner=fake_runner,
            run_manifest_writer=lambda uri, manifest: writes.append((uri, manifest)),
        )
    )

    assert payload["ok"] is False
    assert payload["failed_zone_ids"] == ["montpellier"]
    assert (
        "artifact storage mode 'local' does not match requested 's3'"
        in payload["zones"][0]["error"]
    )
    assert writes[0][1]["checks"]["ok"] is False
    assert writes[0][1]["outputs"] == []


def test_summarize_existing_s3_accepts_explicit_s3_zone_summaries(tmp_path: Path, monkeypatch):
    zone_args = type(
        "Args",
        (),
        {
            "zone_id": "montpellier",
            "era5_output": "s3://deposium/climate/era5/montpellier.parquet",
            "normals_output": "s3://deposium/climate/normals/montpellier.parquet",
        },
    )()
    summary_path = panel._zone_summary_path(
        tmp_path,
        "montpellier",
        "1991-01-01",
        "2020-12-31",
        index=0,
    )
    summary_path.write_text(json.dumps(_ok_payload(zone_args)) + "\n")
    writes: list[tuple[str, dict[str, Any]]] = []

    async def fake_runner(args: Any) -> dict[str, Any]:
        raise AssertionError("summarize-existing must not call the climate runner")

    monkeypatch.setattr(panel.settings, "S3_BUCKET", "deposium")
    args = panel.build_parser().parse_args(
        [
            "--zone-id",
            "montpellier",
            "--storage-mode",
            "s3",
            "--output-dir",
            tmp_path.as_posix(),
            "--summarize-existing",
            "--run-id",
            "explicit-s3-summary",
        ]
    )

    payload = asyncio.run(
        panel.run_panel(
            args,
            runner=fake_runner,
            run_manifest_writer=lambda uri, manifest: writes.append((uri, manifest)),
        )
    )

    assert payload["ok"] is True
    assert payload["failed_zone_ids"] == []
    manifest = writes[0][1]
    assert manifest["checks"]["ok"] is True
    assert manifest["code_ref"] is None
    assert manifest["zone_artifact"] is None
    assert manifest["provenance"]["mode"] == "aggregation_only"
    assert manifest["provenance"]["artifact_code_ref"] is None
    assert manifest["provenance"]["artifact_zone_artifact"] is None
    assert manifest["provenance"]["aggregation_code_ref"] == payload["code_ref"]
    assert manifest["provenance"]["aggregation_zone_artifact"] == payload["zone_artifact"]
    assert manifest["outputs"] == [
        {
            "kind": "era5",
            "zone_id": "montpellier",
            "uri": "s3://deposium/climate/era5/montpellier.parquet",
        },
        {
            "kind": "normals",
            "zone_id": "montpellier",
            "uri": "s3://deposium/climate/normals/montpellier.parquet",
        },
    ]


def test_summarize_existing_holdout_rejects_canonical_s3_artifact_prefix(
    tmp_path: Path, monkeypatch
):
    zone_args = type(
        "Args",
        (),
        {
            "zone_id": "montpellier",
            "storage_mode": "s3",
            "start_date": "2021-01-01",
            "end_date": "2021-12-31",
            "era5_output": "s3://deposium/climate/era5/montpellier.parquet",
            "normals_output": None,
        },
    )()
    payload = _ok_payload(zone_args)
    payload["skip_normals"] = True
    payload["requested_outputs"] = {"era5": zone_args.era5_output, "normals": None}
    payload["resolved_outputs"] = {"era5": zone_args.era5_output, "normals": None}
    payload["stages"].pop("climate_percentiles")
    summary_path = panel._zone_summary_path(
        tmp_path,
        "montpellier",
        "2021-01-01",
        "2021-12-31",
        index=0,
    )
    summary_path.write_text(json.dumps(payload) + "\n")
    writes: list[tuple[str, dict[str, Any]]] = []

    async def fake_runner(args: Any) -> dict[str, Any]:
        raise AssertionError("summarize-existing must not call the climate runner")

    monkeypatch.setattr(panel.settings, "S3_BUCKET", "deposium")
    args = panel.build_parser().parse_args(
        [
            "--zone-id",
            "montpellier",
            "--storage-mode",
            "s3",
            "--artifact-scope",
            "holdout",
            "--start-date",
            "2021-01-01",
            "--end-date",
            "2021-12-31",
            "--output-dir",
            tmp_path.as_posix(),
            "--summarize-existing",
            "--run-id",
            "holdout-prefix-check",
        ]
    )

    aggregate = asyncio.run(
        panel.run_panel(
            args,
            runner=fake_runner,
            run_manifest_writer=lambda uri, manifest: writes.append((uri, manifest)),
        )
    )

    assert aggregate["ok"] is False
    assert aggregate["failed_zone_ids"] == ["montpellier"]
    assert "does not match expected" in aggregate["zones"][0]["error"]
    assert "climate/holdout/era5/2021_2021/montpellier.parquet" in aggregate["zones"][0]["error"]
    assert writes[0][1]["outputs"] == []
    assert writes[0][1]["planned_outputs"] == [
        {
            "kind": "era5",
            "zone_id": "montpellier",
            "uri": "s3://deposium/climate/holdout/era5/2021_2021/montpellier.parquet",
        }
    ]


def test_summarize_existing_holdout_rejects_canonical_s3_summary_with_normals(
    tmp_path: Path, monkeypatch
):
    zone_args = type(
        "Args",
        (),
        {
            "zone_id": "montpellier",
            "storage_mode": "s3",
            "start_date": "2021-01-01",
            "end_date": "2021-12-31",
            "era5_output": "s3://deposium/climate/era5/montpellier.parquet",
            "normals_output": "s3://deposium/climate/normals/montpellier.parquet",
        },
    )()
    summary_path = panel._zone_summary_path(
        tmp_path,
        "montpellier",
        "2021-01-01",
        "2021-12-31",
        index=0,
    )
    summary_path.write_text(json.dumps(_ok_payload(zone_args)) + "\n")
    writes: list[tuple[str, dict[str, Any]]] = []

    async def fake_runner(args: Any) -> dict[str, Any]:
        raise AssertionError("summarize-existing must not call the climate runner")

    monkeypatch.setattr(panel.settings, "S3_BUCKET", "deposium")
    args = panel.build_parser().parse_args(
        [
            "--zone-id",
            "montpellier",
            "--storage-mode",
            "s3",
            "--artifact-scope",
            "holdout",
            "--start-date",
            "2021-01-01",
            "--end-date",
            "2021-12-31",
            "--output-dir",
            tmp_path.as_posix(),
            "--summarize-existing",
            "--run-id",
            "holdout-canonical-reject",
        ]
    )

    aggregate = asyncio.run(
        panel.run_panel(
            args,
            runner=fake_runner,
            run_manifest_writer=lambda uri, manifest: writes.append((uri, manifest)),
        )
    )

    assert aggregate["ok"] is False
    assert aggregate["failed_zone_ids"] == ["montpellier"]
    assert "skip_normals true" in aggregate["zones"][0]["error"]
    assert aggregate["zones"][0]["normals"]["output"] is None
    assert writes[0][1]["outputs"] == []


def test_summarize_existing_holdout_accepts_era5_only_holdout_summary(tmp_path: Path, monkeypatch):
    holdout_era5 = "s3://deposium/climate/holdout/era5/2021_2021/montpellier.parquet"
    zone_args = type(
        "Args",
        (),
        {
            "zone_id": "montpellier",
            "storage_mode": "s3",
            "start_date": "2021-01-01",
            "end_date": "2021-12-31",
            "era5_output": holdout_era5,
            "normals_output": None,
        },
    )()
    payload = _ok_payload(zone_args)
    payload["skip_normals"] = True
    payload["requested_outputs"] = {"era5": holdout_era5, "normals": None}
    payload["resolved_outputs"] = {"era5": holdout_era5, "normals": None}
    payload["stages"].pop("climate_percentiles")
    summary_path = panel._zone_summary_path(
        tmp_path,
        "montpellier",
        "2021-01-01",
        "2021-12-31",
        index=0,
    )
    summary_path.write_text(json.dumps(payload) + "\n")
    writes: list[tuple[str, dict[str, Any]]] = []

    async def fake_runner(args: Any) -> dict[str, Any]:
        raise AssertionError("summarize-existing must not call the climate runner")

    monkeypatch.setattr(panel.settings, "S3_BUCKET", "deposium")
    args = panel.build_parser().parse_args(
        [
            "--zone-id",
            "montpellier",
            "--storage-mode",
            "s3",
            "--artifact-scope",
            "holdout",
            "--start-date",
            "2021-01-01",
            "--end-date",
            "2021-12-31",
            "--output-dir",
            tmp_path.as_posix(),
            "--summarize-existing",
            "--run-id",
            "holdout-existing-summary",
        ]
    )

    aggregate = asyncio.run(
        panel.run_panel(
            args,
            runner=fake_runner,
            run_manifest_writer=lambda uri, manifest: writes.append((uri, manifest)),
        )
    )

    assert aggregate["ok"] is True
    assert aggregate["skip_normals"] is True
    assert aggregate["zones"][0]["skip_normals"] is True
    assert aggregate["zones"][0]["normals"]["output"] is None
    assert writes[0][1]["parameters"]["skip_normals"] is True
    assert writes[0][1]["outputs"] == [
        {"kind": "era5", "zone_id": "montpellier", "uri": holdout_era5}
    ]


def test_summarize_existing_s3_rejects_cross_bucket_artifacts(tmp_path: Path, monkeypatch):
    zone_args = type(
        "Args",
        (),
        {
            "zone_id": "montpellier",
            "storage_mode": "s3",
            "era5_output": "s3://other-bucket/climate/era5/montpellier.parquet",
            "normals_output": "s3://other-bucket/climate/normals/montpellier.parquet",
        },
    )()
    summary_path = panel._zone_summary_path(
        tmp_path,
        "montpellier",
        "1991-01-01",
        "2020-12-31",
        index=0,
    )
    summary_path.write_text(json.dumps(_ok_payload(zone_args)) + "\n")

    async def fake_runner(args: Any) -> dict[str, Any]:
        raise AssertionError("summarize-existing must not call the climate runner")

    monkeypatch.setattr(panel.settings, "S3_BUCKET", "deposium")
    args = panel.build_parser().parse_args(
        [
            "--zone-id",
            "montpellier",
            "--storage-mode",
            "s3",
            "--output-dir",
            tmp_path.as_posix(),
            "--summarize-existing",
        ]
    )

    aggregate = asyncio.run(
        panel.run_panel(args, runner=fake_runner, run_manifest_writer=lambda *_args: None)
    )

    assert aggregate["ok"] is False
    assert aggregate["failed_zone_ids"] == ["montpellier"]
    assert "s3 destination must be under s3://deposium/climate/" in aggregate["zones"][0]["error"]


def test_summarize_existing_local_rejects_legacy_s3_zone_summaries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    zone_args = type(
        "Args",
        (),
        {
            "zone_id": "montpellier",
            "era5_output": "s3://deposium/climate/era5/montpellier.parquet",
            "normals_output": "s3://deposium/climate/normals/montpellier.parquet",
        },
    )()
    payload = _ok_payload(zone_args)
    payload.pop("storage_mode")
    summary_path = panel._zone_summary_path(
        tmp_path,
        "montpellier",
        "1991-01-01",
        "2020-12-31",
        index=0,
    )
    summary_path.write_text(json.dumps(payload) + "\n")

    async def fake_runner(args: Any) -> dict[str, Any]:
        raise AssertionError("summarize-existing must not call the climate runner")

    monkeypatch.setattr(panel.settings, "S3_BUCKET", "deposium")
    args = panel.build_parser().parse_args(
        [
            "--zone-id",
            "montpellier",
            "--output-dir",
            tmp_path.as_posix(),
            "--summarize-existing",
        ]
    )

    aggregate = asyncio.run(panel.run_panel(args, runner=fake_runner))

    assert aggregate["ok"] is False
    assert aggregate["failed_zone_ids"] == ["montpellier"]
    assert (
        "artifact storage mode 's3' does not match requested 'local'"
        in aggregate["zones"][0]["error"]
    )


def test_summarize_existing_local_rejects_unsupported_uri_schemes(tmp_path: Path):
    zone_args = type(
        "Args",
        (),
        {
            "zone_id": "montpellier",
            "era5_output": "https://example.com/era5.parquet",
            "normals_output": "file:///tmp/normals.parquet",
        },
    )()
    payload = _ok_payload(zone_args)
    payload.pop("storage_mode")
    summary_path = panel._zone_summary_path(
        tmp_path,
        "montpellier",
        "1991-01-01",
        "2020-12-31",
        index=0,
    )
    summary_path.write_text(json.dumps(payload) + "\n")

    async def fake_runner(args: Any) -> dict[str, Any]:
        raise AssertionError("summarize-existing must not call the climate runner")

    args = panel.build_parser().parse_args(
        [
            "--zone-id",
            "montpellier",
            "--output-dir",
            tmp_path.as_posix(),
            "--summarize-existing",
        ]
    )

    aggregate = asyncio.run(panel.run_panel(args, runner=fake_runner))

    assert aggregate["ok"] is False
    assert aggregate["failed_zone_ids"] == ["montpellier"]
    assert "unsupported destination scheme 'https'" in aggregate["zones"][0]["error"]


def test_summarize_existing_local_rejects_missing_local_artifacts(tmp_path: Path):
    zone_args = type(
        "Args",
        (),
        {
            "zone_id": "montpellier",
            "era5_output": (tmp_path / "montpellier-era5.parquet").as_posix(),
            "normals_output": (tmp_path / "montpellier-normals.parquet").as_posix(),
        },
    )()
    payload = _ok_payload(zone_args)
    Path(zone_args.era5_output).unlink()
    summary_path = panel._zone_summary_path(
        tmp_path,
        "montpellier",
        "1991-01-01",
        "2020-12-31",
        index=0,
    )
    summary_path.write_text(json.dumps(payload) + "\n")

    async def fake_runner(args: Any) -> dict[str, Any]:
        raise AssertionError("summarize-existing must not call the climate runner")

    args = panel.build_parser().parse_args(
        [
            "--zone-id",
            "montpellier",
            "--output-dir",
            tmp_path.as_posix(),
            "--summarize-existing",
        ]
    )

    aggregate = asyncio.run(panel.run_panel(args, runner=fake_runner))

    assert aggregate["ok"] is False
    assert aggregate["failed_zone_ids"] == ["montpellier"]
    assert "existing local artifact output" in aggregate["zones"][0]["error"]
    assert "montpellier-era5.parquet" in aggregate["zones"][0]["error"]
    assert "does not exist" in aggregate["zones"][0]["error"]


def test_summarize_existing_local_resolves_relative_artifacts_from_summary_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    output_dir = tmp_path / "bundle"
    output_dir.mkdir()
    run_cwd = output_dir / "work"
    run_cwd.mkdir()
    zone_args = type(
        "Args",
        (),
        {
            "zone_id": "montpellier",
            "era5_output": (output_dir / "montpellier-era5.parquet").as_posix(),
            "normals_output": (output_dir / "montpellier-normals.parquet").as_posix(),
        },
    )()
    payload = _ok_payload(zone_args)
    payload["requested_outputs"] = {
        "era5": "montpellier-era5.parquet",
        "normals": "montpellier-normals.parquet",
    }
    payload["resolved_outputs"] = dict(payload["requested_outputs"])
    payload["stages"]["era5_ingest"]["output"] = "montpellier-era5.parquet"
    payload["stages"]["climate_percentiles"]["output"] = "montpellier-normals.parquet"
    summary_path = panel._zone_summary_path(
        output_dir,
        "montpellier",
        "1991-01-01",
        "2020-12-31",
        index=0,
    )
    summary_path.write_text(json.dumps(payload) + "\n")
    monkeypatch.chdir(run_cwd)

    async def fake_runner(args: Any) -> dict[str, Any]:
        raise AssertionError("summarize-existing must not call the climate runner")

    args = panel.build_parser().parse_args(
        [
            "--zone-id",
            "montpellier",
            "--output-dir",
            output_dir.as_posix(),
            "--summarize-existing",
        ]
    )

    aggregate = asyncio.run(panel.run_panel(args, runner=fake_runner))

    expected_era5 = (output_dir / "montpellier-era5.parquet").resolve().as_posix()
    expected_normals = (output_dir / "montpellier-normals.parquet").resolve().as_posix()
    assert aggregate["ok"] is True
    assert aggregate["failed_zone_ids"] == []
    assert aggregate["zones"][0]["era5"]["output"] == expected_era5
    assert aggregate["zones"][0]["normals"]["output"] == expected_normals
    manifest = json.loads((output_dir / "panel-baseline-manifest.json").read_text())
    assert manifest["artifacts"][0]["era5_output"] == expected_era5
    assert manifest["artifacts"][0]["normals_output"] == expected_normals
    assert panel._run_manifest_planned_outputs(aggregate) == [
        {"kind": "era5", "zone_id": "montpellier", "uri": expected_era5},
        {"kind": "normals", "zone_id": "montpellier", "uri": expected_normals},
    ]


def test_summarize_existing_local_resolves_nested_relative_artifacts_from_summary_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    output_dir = tmp_path / "bundle"
    output_dir.mkdir()
    (output_dir / "artifacts").mkdir()
    (output_dir / "artifacts" / "montpellier-era5.parquet").write_text("era5\n")
    (output_dir / "artifacts" / "montpellier-normals.parquet").write_text("normals\n")
    run_cwd = output_dir / "work"
    run_cwd.mkdir()
    (run_cwd / "artifacts").mkdir()
    (run_cwd / "artifacts" / "montpellier-era5.parquet").write_text("wrong era5\n")
    (run_cwd / "artifacts" / "montpellier-normals.parquet").write_text("wrong normals\n")
    zone_args = type(
        "Args",
        (),
        {
            "zone_id": "montpellier",
            "era5_output": (output_dir / "artifacts" / "montpellier-era5.parquet").as_posix(),
            "normals_output": (output_dir / "artifacts" / "montpellier-normals.parquet").as_posix(),
        },
    )()
    payload = _ok_payload(zone_args)
    payload["requested_outputs"] = {
        "era5": "artifacts/montpellier-era5.parquet",
        "normals": "artifacts/montpellier-normals.parquet",
    }
    payload["resolved_outputs"] = dict(payload["requested_outputs"])
    payload["stages"]["era5_ingest"]["output"] = "artifacts/montpellier-era5.parquet"
    payload["stages"]["climate_percentiles"]["output"] = "artifacts/montpellier-normals.parquet"
    summary_path = panel._zone_summary_path(
        output_dir,
        "montpellier",
        "1991-01-01",
        "2020-12-31",
        index=0,
    )
    summary_path.write_text(json.dumps(payload) + "\n")
    monkeypatch.chdir(run_cwd)

    async def fake_runner(args: Any) -> dict[str, Any]:
        raise AssertionError("summarize-existing must not call the climate runner")

    args = panel.build_parser().parse_args(
        [
            "--zone-id",
            "montpellier",
            "--output-dir",
            output_dir.as_posix(),
            "--summarize-existing",
        ]
    )

    aggregate = asyncio.run(panel.run_panel(args, runner=fake_runner))

    expected_era5 = (output_dir / "artifacts" / "montpellier-era5.parquet").resolve().as_posix()
    expected_normals = (
        (output_dir / "artifacts" / "montpellier-normals.parquet").resolve().as_posix()
    )
    assert aggregate["ok"] is True
    assert aggregate["failed_zone_ids"] == []
    assert aggregate["zones"][0]["era5"]["output"] == expected_era5
    assert aggregate["zones"][0]["normals"]["output"] == expected_normals


def test_relative_output_dir_paths_are_not_rebased_twice(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.chdir(tmp_path)
    output_dir = Path("out")
    slug = "1991-01-01_2020-12-31"
    expected_era5 = (tmp_path / "out" / f"montpellier-era5-{slug}.parquet").as_posix()
    expected_normals = (tmp_path / "out" / f"montpellier-normals-{slug}.parquet").as_posix()

    async def fake_runner(args: Any) -> dict[str, Any]:
        return _ok_payload(args)

    args = panel.build_parser().parse_args(
        [
            "--zone-id",
            "montpellier",
            "--output-dir",
            output_dir.as_posix(),
        ]
    )

    payload = asyncio.run(panel.run_panel(args, runner=fake_runner))

    assert payload["ok"] is True
    assert payload["zones"][0]["era5"]["output"] == expected_era5
    assert payload["zones"][0]["normals"]["output"] == expected_normals
    manifest = json.loads((output_dir / "panel-baseline-manifest.json").read_text())
    assert manifest["artifacts"][0]["era5_output"] == expected_era5
    assert manifest["artifacts"][0]["normals_output"] == expected_normals

    async def forbidden_runner(args: Any) -> dict[str, Any]:
        raise AssertionError("summarize-existing must not call the climate runner")

    summarize_args = panel.build_parser().parse_args(
        [
            "--zone-id",
            "montpellier",
            "--output-dir",
            output_dir.as_posix(),
            "--summarize-existing",
        ]
    )

    aggregate = asyncio.run(panel.run_panel(summarize_args, runner=forbidden_runner))

    assert aggregate["ok"] is True
    assert aggregate["failed_zone_ids"] == []
    assert aggregate["zones"][0]["era5"]["output"] == expected_era5
    assert aggregate["zones"][0]["normals"]["output"] == expected_normals


def test_relative_output_dir_preflight_planned_paths_are_not_rebased_twice(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.chdir(tmp_path)
    output_dir = Path("out")
    slug = "1991-01-01_2020-12-31"
    expected_era5 = (tmp_path / "out" / f"montpellier-era5-{slug}.parquet").as_posix()
    expected_normals = (tmp_path / "out" / f"montpellier-normals-{slug}.parquet").as_posix()

    async def forbidden_runner(args: Any) -> dict[str, Any]:
        raise AssertionError("preflight must not call the climate runner")

    args = panel.build_parser().parse_args(
        [
            "--zone-id",
            "montpellier",
            "--output-dir",
            output_dir.as_posix(),
            "--preflight-only",
        ]
    )

    payload = asyncio.run(panel.run_panel(args, runner=forbidden_runner))

    assert payload["ok"] is True
    assert payload["zones"][0]["resolved_outputs"] == {
        "era5": expected_era5,
        "normals": expected_normals,
    }
    assert not Path(expected_era5).exists()
    manifest = json.loads((output_dir / "panel-baseline-manifest.json").read_text())
    assert manifest["artifacts"][0]["planned_outputs"] == {
        "era5": expected_era5,
        "normals": expected_normals,
    }


def test_relative_output_dir_failed_planned_paths_are_not_rebased_twice(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.chdir(tmp_path)
    output_dir = Path("out")
    slug = "1991-01-01_2020-12-31"
    expected_era5 = (tmp_path / "out" / f"montpellier-era5-{slug}.parquet").as_posix()
    expected_normals = (tmp_path / "out" / f"montpellier-normals-{slug}.parquet").as_posix()

    async def fake_runner(_args: Any) -> dict[str, Any]:
        raise RuntimeError("archive unavailable")

    args = panel.build_parser().parse_args(
        [
            "--zone-id",
            "montpellier",
            "--output-dir",
            output_dir.as_posix(),
        ]
    )

    payload = asyncio.run(panel.run_panel(args, runner=fake_runner))

    assert payload["ok"] is False
    assert payload["zones"][0]["resolved_outputs"] == {
        "era5": expected_era5,
        "normals": expected_normals,
    }
    assert not Path(expected_era5).exists()
    manifest = json.loads((output_dir / "panel-baseline-manifest.json").read_text())
    assert manifest["artifacts"][0]["planned_outputs"] == {
        "era5": expected_era5,
        "normals": expected_normals,
    }


def test_summarize_existing_rejects_unknown_zone_before_handoff(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    zone_id = "definitely_unknown_zone"
    zone_args = type(
        "Args",
        (),
        {
            "zone_id": zone_id,
            "era5_output": f"s3://deposium/climate/era5/{zone_id}.parquet",
            "normals_output": f"s3://deposium/climate/normals/{zone_id}.parquet",
        },
    )()
    summary_path = panel._zone_summary_path(
        tmp_path,
        zone_id,
        "1991-01-01",
        "2020-12-31",
        index=0,
    )
    summary_path.write_text(json.dumps(_ok_payload(zone_args)) + "\n")

    async def fake_runner(args: Any) -> dict[str, Any]:
        raise AssertionError("summarize-existing must not call the climate runner")

    monkeypatch.setattr(panel.settings, "S3_BUCKET", "deposium")
    args = panel.build_parser().parse_args(
        [
            "--zone-id",
            zone_id,
            "--storage-mode",
            "s3",
            "--output-dir",
            tmp_path.as_posix(),
            "--summarize-existing",
            "--run-id",
            "unknown-zone-summary",
        ]
    )

    aggregate = asyncio.run(
        panel.run_panel(args, runner=fake_runner, run_manifest_writer=lambda *_args: None)
    )

    assert aggregate["ok"] is False
    assert aggregate["failed_zone_ids"] == [zone_id]
    assert "Unknown zone_id" in aggregate["zones"][0]["error"]


def test_summarize_existing_rejects_lower_completeness_gate(tmp_path: Path):
    zone_args = type(
        "Args",
        (),
        {
            "zone_id": "montpellier",
            "era5_output": (tmp_path / "montpellier-era5.parquet").as_posix(),
            "normals_output": (tmp_path / "montpellier-normals.parquet").as_posix(),
            "min_completeness_pct": 50.0,
        },
    )()
    payload = _ok_payload(zone_args)
    summary_path = panel._zone_summary_path(
        tmp_path,
        "montpellier",
        "1991-01-01",
        "2020-12-31",
        index=0,
    )
    summary_path.write_text(json.dumps(payload) + "\n")

    async def fake_runner(args: Any) -> dict[str, Any]:
        raise AssertionError("summarize-existing must not call the climate runner")

    args = panel.build_parser().parse_args(
        [
            "--zone-id",
            "montpellier",
            "--output-dir",
            tmp_path.as_posix(),
            "--summarize-existing",
            "--min-completeness-pct",
            "98",
        ]
    )

    aggregate = asyncio.run(panel.run_panel(args, runner=fake_runner))

    assert aggregate["ok"] is False
    assert aggregate["failed_zone_ids"] == ["montpellier"]
    assert "min_completeness_pct 50.0 < requested 98.0" in aggregate["zones"][0]["error"]


def test_summarize_existing_rejects_stale_completeness_metrics(tmp_path: Path):
    zone_args = type(
        "Args",
        (),
        {
            "zone_id": "montpellier",
            "era5_output": (tmp_path / "montpellier-era5.parquet").as_posix(),
            "normals_output": (tmp_path / "montpellier-normals.parquet").as_posix(),
            "min_completeness_pct": 98.0,
        },
    )()
    payload = _ok_payload(zone_args)
    payload["stages"]["era5_ingest"]["pct_days"] = 60.0
    summary_path = panel._zone_summary_path(
        tmp_path,
        "montpellier",
        "1991-01-01",
        "2020-12-31",
        index=0,
    )
    summary_path.write_text(json.dumps(payload) + "\n")

    async def fake_runner(args: Any) -> dict[str, Any]:
        raise AssertionError("summarize-existing must not call the climate runner")

    args = panel.build_parser().parse_args(
        [
            "--zone-id",
            "montpellier",
            "--output-dir",
            tmp_path.as_posix(),
            "--summarize-existing",
            "--min-completeness-pct",
            "98",
        ]
    )

    aggregate = asyncio.run(panel.run_panel(args, runner=fake_runner))

    assert aggregate["ok"] is False
    assert aggregate["failed_zone_ids"] == ["montpellier"]
    assert "era5 pct_days 60.0 < completeness threshold 98.0" in aggregate["zones"][0]["error"]


def test_summarize_existing_rejects_short_era5_day_count(tmp_path: Path):
    zone_args = type(
        "Args",
        (),
        {
            "zone_id": "montpellier",
            "era5_output": (tmp_path / "montpellier-era5.parquet").as_posix(),
            "normals_output": (tmp_path / "montpellier-normals.parquet").as_posix(),
            "min_completeness_pct": 98.0,
        },
    )()
    payload = _ok_payload(zone_args)
    payload["stages"]["era5_ingest"]["n_days"] = 365
    payload["stages"]["era5_ingest"]["pct_days"] = 100.0
    payload["stages"]["era5_ingest"]["min_variable_pct_complete"] = 100.0
    summary_path = panel._zone_summary_path(
        tmp_path,
        "montpellier",
        "1991-01-01",
        "2020-12-31",
        index=0,
    )
    summary_path.write_text(json.dumps(payload) + "\n")

    async def fake_runner(args: Any) -> dict[str, Any]:
        raise AssertionError("summarize-existing must not call the climate runner")

    args = panel.build_parser().parse_args(
        [
            "--zone-id",
            "montpellier",
            "--output-dir",
            tmp_path.as_posix(),
            "--summarize-existing",
            "--min-completeness-pct",
            "98",
        ]
    )

    aggregate = asyncio.run(panel.run_panel(args, runner=fake_runner))

    assert aggregate["ok"] is False
    assert aggregate["failed_zone_ids"] == ["montpellier"]
    assert "era5 n_days 365 < expected 10958" in aggregate["zones"][0]["error"]


def test_summarize_existing_rejects_mismatched_summary_window(tmp_path: Path):
    zone_args = type(
        "Args",
        (),
        {
            "zone_id": "montpellier",
            "era5_output": (tmp_path / "montpellier-era5.parquet").as_posix(),
            "normals_output": (tmp_path / "montpellier-normals.parquet").as_posix(),
        },
    )()
    payload = _ok_payload(zone_args)
    payload["window"]["start_date"] = "2000-01-01"
    payload["window"]["end_date"] = "2020-12-31"
    summary_path = panel._zone_summary_path(
        tmp_path,
        "montpellier",
        "1991-01-01",
        "2020-12-31",
        index=0,
    )
    summary_path.write_text(json.dumps(payload) + "\n")

    async def fake_runner(args: Any) -> dict[str, Any]:
        raise AssertionError("summarize-existing must not call the climate runner")

    args = panel.build_parser().parse_args(
        [
            "--zone-id",
            "montpellier",
            "--output-dir",
            tmp_path.as_posix(),
            "--summarize-existing",
        ]
    )

    aggregate = asyncio.run(panel.run_panel(args, runner=fake_runner))

    assert aggregate["ok"] is False
    assert aggregate["failed_zone_ids"] == ["montpellier"]
    assert (
        "summary window '2000-01-01'..'2020-12-31' does not match requested"
        in aggregate["zones"][0]["error"]
    )


def test_summarize_existing_rejects_mismatched_era5_window(tmp_path: Path):
    zone_args = type(
        "Args",
        (),
        {
            "zone_id": "montpellier",
            "era5_output": (tmp_path / "montpellier-era5.parquet").as_posix(),
            "normals_output": (tmp_path / "montpellier-normals.parquet").as_posix(),
        },
    )()
    payload = _ok_payload(zone_args)
    payload["stages"]["era5_ingest"]["start_date"] = "2000-01-01"
    payload["stages"]["era5_ingest"]["end_date"] = "2020-12-31"
    summary_path = panel._zone_summary_path(
        tmp_path,
        "montpellier",
        "1991-01-01",
        "2020-12-31",
        index=0,
    )
    summary_path.write_text(json.dumps(payload) + "\n")

    async def fake_runner(args: Any) -> dict[str, Any]:
        raise AssertionError("summarize-existing must not call the climate runner")

    args = panel.build_parser().parse_args(
        [
            "--zone-id",
            "montpellier",
            "--output-dir",
            tmp_path.as_posix(),
            "--summarize-existing",
        ]
    )

    aggregate = asyncio.run(panel.run_panel(args, runner=fake_runner))

    assert aggregate["ok"] is False
    assert aggregate["failed_zone_ids"] == ["montpellier"]
    assert (
        "era5 window '2000-01-01'..'2020-12-31' does not match requested"
        in aggregate["zones"][0]["error"]
    )


def test_summarize_existing_rejects_normals_shape_mismatch(tmp_path: Path):
    zone_args = type(
        "Args",
        (),
        {
            "zone_id": "montpellier",
            "era5_output": (tmp_path / "montpellier-era5.parquet").as_posix(),
            "normals_output": (tmp_path / "montpellier-normals.parquet").as_posix(),
            "variables": ["temperature_2m"],
        },
    )()
    payload = _ok_payload(zone_args)
    payload["stages"]["climate_percentiles"]["n_variables"] = 6
    payload["stages"]["climate_percentiles"]["n_normals"] = 72
    summary_path = panel._zone_summary_path(
        tmp_path,
        "montpellier",
        "1991-01-01",
        "2020-12-31",
        index=0,
    )
    summary_path.write_text(json.dumps(payload) + "\n")

    async def fake_runner(args: Any) -> dict[str, Any]:
        raise AssertionError("summarize-existing must not call the climate runner")

    args = panel.build_parser().parse_args(
        [
            "--zone-id",
            "montpellier",
            "--variables",
            "temperature_2m",
            "--output-dir",
            tmp_path.as_posix(),
            "--summarize-existing",
        ]
    )

    aggregate = asyncio.run(panel.run_panel(args, runner=fake_runner))

    assert aggregate["ok"] is False
    assert aggregate["failed_zone_ids"] == ["montpellier"]
    assert "normals n_variables 6.0 != variables count 1" in aggregate["zones"][0]["error"]


def test_summarize_existing_rejects_preflight_only_summaries(tmp_path: Path):
    preflight_args = panel.build_parser().parse_args(
        [
            "--zone-id",
            "montpellier",
            "--output-dir",
            tmp_path.as_posix(),
            "--preflight-only",
        ]
    )
    asyncio.run(panel.run_panel(preflight_args))

    async def fake_runner(args: Any) -> dict[str, Any]:
        raise AssertionError("summarize-existing must not call the climate runner")

    args = panel.build_parser().parse_args(
        [
            "--zone-id",
            "montpellier",
            "--output-dir",
            tmp_path.as_posix(),
            "--summarize-existing",
        ]
    )

    payload = asyncio.run(panel.run_panel(args, runner=fake_runner))

    assert payload["ok"] is False
    assert payload["failed_zone_ids"] == ["montpellier"]
    zone = payload["zones"][0]
    assert zone["failed_stage"] == "existing_summary"
    assert "preflight-only" in zone["error"]


def test_summarize_existing_rejects_success_without_materialized_gate(tmp_path: Path):
    summary_path = panel._zone_summary_path(
        tmp_path,
        "montpellier",
        "1991-01-01",
        "2020-12-31",
        index=0,
    )
    summary_path.write_text(json.dumps({"ok": True, "zone_id": "montpellier", "stages": {}}) + "\n")

    async def fake_runner(args: Any) -> dict[str, Any]:
        raise AssertionError("summarize-existing must not call the climate runner")

    args = panel.build_parser().parse_args(
        [
            "--zone-id",
            "montpellier",
            "--output-dir",
            tmp_path.as_posix(),
            "--summarize-existing",
        ]
    )

    payload = asyncio.run(panel.run_panel(args, runner=fake_runner))

    assert payload["ok"] is False
    assert payload["failed_zone_ids"] == ["montpellier"]
    zone = payload["zones"][0]
    assert zone["failed_stage"] == "existing_summary"
    assert "no era5_ingest and climate_percentiles stages" in zone["error"]


def test_panel_collects_failures_without_fail_fast(tmp_path: Path):
    async def fake_runner(args: Any) -> dict[str, Any]:
        if args.zone_id == "bad_zone":
            return {
                "ok": False,
                "zone_id": args.zone_id,
                "failed_stage": "completeness_gate",
                "error": "below threshold",
                "stages": {},
            }
        return _ok_payload(args)

    args = panel.build_parser().parse_args(
        [
            "--zone-id",
            "bad_zone",
            "nairobi",
            "--output-dir",
            tmp_path.as_posix(),
        ]
    )

    payload = asyncio.run(panel.run_panel(args, runner=fake_runner))

    assert payload["ok"] is False
    assert payload["completed_zone_ids"] == ["bad_zone", "nairobi"]
    assert payload["failed_zone_ids"] == ["bad_zone"]
    assert (tmp_path / "bad_zone-baseline-1991-01-01_2020-12-31.json").exists()
    assert (tmp_path / "nairobi-baseline-1991-01-01_2020-12-31.json").exists()


def test_panel_fail_fast_stops_after_first_failure(tmp_path: Path):
    calls: list[str] = []

    async def fake_runner(args: Any) -> dict[str, Any]:
        calls.append(args.zone_id)
        return {
            "ok": False,
            "zone_id": args.zone_id,
            "failed_stage": "era5_ingest",
            "error": "offline",
            "stages": {},
        }

    args = panel.build_parser().parse_args(
        [
            "--zone-id",
            "bad_zone",
            "nairobi",
            "--output-dir",
            tmp_path.as_posix(),
            "--fail-fast",
        ]
    )

    payload = asyncio.run(panel.run_panel(args, runner=fake_runner))

    assert payload["ok"] is False
    assert calls == ["bad_zone"]
    assert payload["completed_zone_ids"] == ["bad_zone"]


def test_invalid_zone_id_is_reported_without_path_escape(tmp_path: Path):
    calls: list[str] = []

    async def fake_runner(args: Any) -> dict[str, Any]:
        calls.append(args.zone_id)
        return _ok_payload(args)

    args = panel.build_parser().parse_args(
        [
            "--zone-id",
            "../bad",
            "bad/zone",
            "nairobi",
            "--output-dir",
            tmp_path.as_posix(),
        ]
    )

    payload = asyncio.run(panel.run_panel(args, runner=fake_runner))

    assert payload["ok"] is False
    assert calls == ["nairobi"]
    assert payload["failed_zone_ids"] == ["../bad", "bad/zone"]
    assert (tmp_path / "invalid-zone-1-baseline-1991-01-01_2020-12-31.json").exists()
    assert (tmp_path / "invalid-zone-2-baseline-1991-01-01_2020-12-31.json").exists()
    assert not (tmp_path.parent / "bad-baseline-1991-01-01_2020-12-31.json").exists()

    aggregate = json.loads((tmp_path / "panel-baseline-summary.json").read_text())
    invalid_entries = aggregate["zones"][:2]
    assert all(entry["failed_stage"] == "preflight" for entry in invalid_entries)
    assert all(Path(entry["summary_path"]).parent == tmp_path for entry in invalid_entries)


def test_zone_exception_is_contained_and_following_zone_runs(tmp_path: Path):
    calls: list[str] = []

    async def fake_runner(args: Any) -> dict[str, Any]:
        calls.append(args.zone_id)
        if args.zone_id == "fos_sur_mer":
            raise RuntimeError("archive timeout")
        return _ok_payload(args)

    args = panel.build_parser().parse_args(
        [
            "--zone-id",
            "fos_sur_mer",
            "nairobi",
            "--output-dir",
            tmp_path.as_posix(),
        ]
    )

    payload = asyncio.run(panel.run_panel(args, runner=fake_runner))

    assert payload["ok"] is False
    assert calls == ["fos_sur_mer", "nairobi"]
    assert payload["failed_zone_ids"] == ["fos_sur_mer"]
    assert payload["zones"][0]["failed_stage"] == "exception"
    assert "RuntimeError: archive timeout" in payload["zones"][0]["error"]
    assert (tmp_path / "fos_sur_mer-baseline-1991-01-01_2020-12-31.json").exists()
    assert (tmp_path / "nairobi-baseline-1991-01-01_2020-12-31.json").exists()


def test_non_finite_zone_delay_is_rejected():
    for value in ("nan", "inf", "-inf"):
        try:
            panel.build_parser().parse_args(["--zone-id", "nairobi", "--zone-delay-s", value])
        except SystemExit as exc:
            assert exc.code == 2
        else:  # pragma: no cover - defensive assertion message
            raise AssertionError(f"{value} should be rejected")


def test_script_help_runs_from_outside_repo(tmp_path: Path):
    result = subprocess.run(
        [sys.executable, _PANEL_PATH.as_posix(), "--help"],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0
    assert "climate_baseline_panel" in result.stdout


def test_run_manifest_records_the_object_store_it_wrote_to(monkeypatch):
    """Bucket-relative URIs cannot tell two RustFS instances apart; the endpoint can (issue #27)."""
    monkeypatch.setattr(panel.settings, "S3_ENDPOINT", "http://127.0.0.1:9000")
    monkeypatch.setattr(panel.settings, "S3_BUCKET", "deposium")
    monkeypatch.setattr(panel.settings, "S3_FORCE_PATH_STYLE", True)

    manifest = panel._run_manifest_payload({"run_id": "env-proof", "zones": []})

    assert manifest["environment"] == {
        "s3_endpoint": "http://127.0.0.1:9000",
        "s3_endpoint_credentials_removed": False,
        "s3_bucket": "deposium",
        "s3_force_path_style": True,
    }


def test_unset_endpoint_is_recorded_as_null(monkeypatch):
    # Only the endpoint half is asserted: manifest publication is gated on a non-empty S3_BUCKET,
    # so no published manifest can carry a null bucket.
    monkeypatch.setattr(panel.settings, "S3_ENDPOINT", "")
    monkeypatch.setattr(panel.settings, "S3_BUCKET", "deposium")

    manifest = panel._run_manifest_payload({"run_id": "env-proof", "zones": []})

    # An empty string would read like a configured address; null says "none configured".
    assert manifest["environment"]["s3_endpoint"] is None


def test_run_manifest_never_carries_a_credential(monkeypatch):
    monkeypatch.setattr(panel.settings, "S3_ENDPOINT", "http://127.0.0.1:9000")
    monkeypatch.setattr(panel.settings, "S3_BUCKET", "deposium")
    monkeypatch.setattr(panel.settings, "S3_ACCESS_KEY", "deposium-dev-access-key")
    monkeypatch.setattr(panel.settings, "S3_SECRET_KEY", "deposium-dev-secret-key")

    serialized = json.dumps(
        panel._run_manifest_payload({"run_id": "env-proof", "zones": []}), default=str
    )

    assert "deposium-dev-access-key" not in serialized
    assert "deposium-dev-secret-key" not in serialized
    assert "127.0.0.1:9000" in serialized


def test_credentials_embedded_in_the_endpoint_never_reach_the_manifest(monkeypatch):
    """`S3_ENDPOINT` may legally carry userinfo, and this manifest is published for handoff."""
    monkeypatch.setattr(
        panel.settings, "S3_ENDPOINT", "http://rustfs-admin:sup3r-s3cret@mesh.internal:9000"
    )
    monkeypatch.setattr(panel.settings, "S3_BUCKET", "deposium")

    manifest = panel._run_manifest_payload({"run_id": "env-proof", "zones": []})
    serialized = json.dumps(manifest, default=str)

    # Assert on the serialized artifact: nesting must not be able to hide the secret.
    assert "sup3r-s3cret" not in serialized
    assert "rustfs-admin" not in serialized
    # Refused whole, not stripped back to `http://mesh.internal:9000`: the accept grammar has no
    # partial recovery. The flag is what tells a reader an endpoint WAS configured.
    assert manifest["environment"]["s3_endpoint"] is None
    assert manifest["environment"]["s3_endpoint_credentials_removed"] is True


def test_write_s3_json_round_trips_the_nested_manifest_shapes(tmp_path, monkeypatch):
    """The real DuckDB `COPY ... (FORMAT JSON)` projection, not an injected fake writer.

    Every panel test stubs `run_manifest_writer`, so nothing else in CI exercises the writer that
    actually serializes these payloads — including `checks.month_coverage.zones`, the first
    list-of-structs nested inside a struct any manifest carries.
    """
    destination = tmp_path / "manifest.json"
    # The URI guard confines writes to s3://{bucket}/climate/runs/; the projection under test is
    # storage-agnostic, so only the guard is relaxed to reach a temp path.
    monkeypatch.setattr(panel, "_validate_run_manifest_uri", lambda uri: None)

    payload = {
        "schema_version": 1,
        "environment": {
            "s3_endpoint": "http://mesh.internal:9000",
            "s3_endpoint_credentials_removed": True,
            "s3_bucket": "deposium",
            "s3_force_path_style": True,
        },
        "checks": {
            "ok": True,
            "month_coverage": {
                "months_requested": 12,
                "complete_evaluated": True,
                "zones": [
                    {"zone_id": "bassin_du_lez", "months_observed": 12, "complete": True},
                    {"zone_id": "fos_sur_mer", "months_observed": 7, "complete": False},
                ],
            },
        },
    }
    panel._write_s3_json(destination.as_posix(), payload)

    with panel.duckdb_session(":memory:") as conn:
        (raw,) = conn.execute(
            "SELECT to_json(j) FROM read_json_auto(?) AS j", [destination.as_posix()]
        ).fetchone()

    assert json.loads(raw) == payload


def test_write_s3_json_round_trips_the_degraded_month_coverage_shapes(tmp_path, monkeypatch):
    """An empty `zones` list and a null `months_requested` are real states, not hypotheticals."""
    destination = tmp_path / "manifest.json"
    monkeypatch.setattr(panel, "_validate_run_manifest_uri", lambda uri: None)

    payload = {
        "schema_version": 1,
        "environment": {
            "s3_endpoint": None,
            "s3_endpoint_credentials_removed": False,
            "s3_bucket": "deposium",
            "s3_force_path_style": False,
        },
        "checks": {
            "ok": False,
            "month_coverage": {
                "months_requested": None,
                "complete_evaluated": False,
                "zones": [],
            },
        },
    }
    panel._write_s3_json(destination.as_posix(), payload)

    with panel.duckdb_session(":memory:") as conn:
        (raw,) = conn.execute(
            "SELECT to_json(j) FROM read_json_auto(?) AS j", [destination.as_posix()]
        ).fetchone()

    assert json.loads(raw) == payload


# A real DuckDB httpfs failure, verbatim. `src.utils.duckdb_conn` hands `S3_ENDPOINT` to
# `SET s3_endpoint` with only the scheme stripped, so httpfs quotes the userinfo back at us.
_ACCESS_KEY = "deposium-dev"
_SECRET_KEY = "deposium-dev-secret"
# The shape that actually runs. `docker-compose.yml` sets `S3_ENDPOINT=http://rustfs:9000` and
# supplies the credentials SEPARATELY, and httpfs quotes the configured `s3_access_key_id` back
# in its 403 bodies. A userinfo-bearing endpoint no longer reaches this test at all: it is
# refused by the accept grammar before `duckdb_session` opens, so nothing gets published.
_HTTPFS_FAILURE = (
    "IOException: IO Error: Could not connect to server error for HTTP HEAD to "
    "'http://mesh.internal:9000/deposium/climate/era5/z1.parquet' (HTTP 403) "
    f"<AWSAccessKeyId>{_ACCESS_KEY}</AWSAccessKeyId>"
)
# The botocore spelling of the same thing, from `src/cli/forecast_ensemble_archive.py`'s client.
_BOTOCORE_FAILURE = (
    "ClientError: An error occurred (SignatureDoesNotMatch) calling the PutObject operation: "
    f"the signature computed from key {_SECRET_KEY} does not match"
)


def test_a_failed_zone_cannot_smuggle_the_store_password_into_the_published_manifest(
    tmp_path, monkeypatch
):
    """`environment.s3_endpoint` is sanitized; `errors[].error` is verbatim exception text.

    A zone that fails against the object store reports `f"{type(exc).__name__}: {exc}"`, and both
    clients this repo uses quote the configured credential in that message — httpfs echoes
    `s3_access_key_id` inside a 403 body, botocore names the signing key. So a single failed zone
    publishes the store's credentials into the same manifest whose `environment` block was
    carefully built from a sanitizer. The scrub belongs at the publication boundary, not at each
    `except`: there are a dozen of those and one more arrives with every new stage.
    """
    destination = tmp_path / "manifest.json"
    monkeypatch.setattr(panel, "_validate_run_manifest_uri", lambda uri: None)
    monkeypatch.setattr(panel.settings, "S3_ENDPOINT", "http://mesh.internal:9000")
    monkeypatch.setattr(panel.settings, "S3_ACCESS_KEY", _ACCESS_KEY)
    monkeypatch.setattr(panel.settings, "S3_SECRET_KEY", _SECRET_KEY)

    payload = {
        "schema_version": 1,
        "environment": panel._run_manifest_environment(),
        "errors": [{"zone_id": "z1", "failed_stage": "exception", "error": _HTTPFS_FAILURE}],
        "run_manifest_errors": [{"uri": "s3://deposium/x", "error": _BOTOCORE_FAILURE}],
    }
    panel._write_s3_json(destination.as_posix(), payload)

    published = destination.read_text()

    # Assert on the bytes that reached the bucket, not on the dict: a credential hides in whatever
    # key no assertion happens to read.
    assert _SECRET_KEY not in published
    assert _ACCESS_KEY not in published
    assert "[REDACTED]" in published
    # The address itself is still on the record — this redacts the credential, not the provenance.
    assert "mesh.internal:9000" in published


# A real pystac_client failure, verbatim. Measured on the pinned pystac_client/requests with
# `STAC_URL=https://stac-reader:stac-s3cret@stac.internal:notaport/v1`: the URL-parse family of
# failures echoes the CONFIGURED URL in full, userinfo included. (A connection-level failure does
# not — urllib3 rebuilds its message from host and port alone.)
_STAC_TOKEN = "stac-s3cret"
_STAC_URL = f"https://stac-reader:{_STAC_TOKEN}@stac.internal:notaport/v1"
_STAC_FAILURE = f"APIError: Failed to parse: {_STAC_URL}"


def test_a_private_stac_token_cannot_ride_out_in_an_error_message(tmp_path, monkeypatch):
    """A catalogue credential is a SECOND credential, and this writer publishes it too.

    `_write_s3_json` is the default run-manifest writer for `scripts/ndvi_pilot_panel.py`, which
    does reach STAC. The redactor was built from `S3_ENDPOINT` plus the configured S3 keys, so a
    private `STAC_URL`'s userinfo matched nothing at all and a failed catalogue request published
    the token verbatim into the destination bucket. Sanitizing `inputs[].endpoint` does not cover
    this: `errors[].error` is a different field.
    """
    destination = tmp_path / "manifest.json"
    monkeypatch.setattr(panel, "_validate_run_manifest_uri", lambda uri: None)
    monkeypatch.setattr(panel.settings, "S3_ENDPOINT", "http://mesh.internal:9000")
    monkeypatch.setattr(panel.settings, "S3_ACCESS_KEY", _ACCESS_KEY)
    monkeypatch.setattr(panel.settings, "S3_SECRET_KEY", _SECRET_KEY)
    monkeypatch.setattr(panel.settings, "STAC_URL", _STAC_URL)

    payload = {
        "schema_version": 1,
        "errors": [{"zone_id": "z1", "failed_stage": "ndvi", "error": _STAC_FAILURE}],
        "run_manifest_errors": [{"uri": "s3://deposium/x", "error": _STAC_FAILURE}],
    }
    panel._write_s3_json(destination.as_posix(), payload)

    published = destination.read_text()

    assert _STAC_TOKEN not in published
    assert "stac-reader" not in published
    # The catalogue host stays on the record: this redacts the credential, not the provenance.
    assert "stac.internal" in published


def test_a_userinfo_endpoint_is_refused_before_any_manifest_is_published(tmp_path, monkeypatch):
    """The other half of the same guarantee, and it is now closed one layer earlier.

    `http://user:pass@host` no longer publishes a stripped address beside a redacted error — it
    matches no accept-grammar production, so `duckdb_session` refuses to open at all and no object
    is written. The password cannot be in a manifest that does not exist.
    """
    monkeypatch.setattr(panel, "_validate_run_manifest_uri", lambda uri: None)
    monkeypatch.setattr(
        panel.settings, "S3_ENDPOINT", "http://rustfs-admin:sup3r-s3cret@mesh.internal:9000"
    )
    destination = tmp_path / "manifest.json"

    with pytest.raises(S3EndpointConfigurationError):
        panel._write_s3_json(destination.as_posix(), {"schema_version": 1})

    assert not destination.exists()


def test_a_clean_endpoint_leaves_every_error_message_exactly_as_it_was(tmp_path, monkeypatch):
    """The normal deployment. Redaction must not rewrite diagnostics it has no business touching."""
    destination = tmp_path / "manifest.json"
    monkeypatch.setattr(panel, "_validate_run_manifest_uri", lambda uri: None)
    monkeypatch.setattr(panel.settings, "S3_ENDPOINT", "http://mesh.internal:9000")

    payload = {
        "schema_version": 1,
        "errors": [{"zone_id": "z1", "failed_stage": "exception", "error": _HTTPFS_FAILURE}],
    }
    panel._write_s3_json(destination.as_posix(), payload)

    with panel.duckdb_session(":memory:") as conn:
        (raw,) = conn.execute(
            "SELECT to_json(j) FROM read_json_auto(?) AS j", [destination.as_posix()]
        ).fetchone()

    assert json.loads(raw) == payload


# --------------------------------------------------------------------------- #
# End to end: a credential in an error message never reaches the published object
# --------------------------------------------------------------------------- #


class _RecordingConnection:
    """Stands in for the DuckDB connection `_write_s3_json` opens. Records the COPY binding."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, list | None]] = []

    def execute(self, sql: str, params: list | None = None) -> None:
        self.calls.append((sql, params))


def test_a_credential_in_a_zone_error_never_reaches_the_published_manifest(
    tmp_path: Path, monkeypatch
):
    """The end-to-end guarantee, through THIS panel's real publication path.

    `climate_baseline_panel._write_s3_json` is the DEFAULT run-manifest writer here and the one
    the ndvi, holdout and segmented panels import and reuse, so it is the wider of the two write
    sites. `run_manifest_writer` is deliberately NOT passed: the real writer must run, and only
    the DuckDB engine is stubbed — at the last possible point, so what the fake records is
    exactly the list of values DuckDB would have bound and written into the bucket.

    The message is the one httpfs actually produces against the dev store: it quotes the
    configured `s3_access_key_id` back inside a 403 body.
    """
    access_key = "deposium-dev"
    secret_key = "deposium-dev-secret"
    monkeypatch.setattr(panel.settings, "S3_BUCKET", "deposium")
    monkeypatch.setattr(panel.settings, "S3_ENDPOINT", "http://rustfs:9000")
    monkeypatch.setattr(panel.settings, "S3_ACCESS_KEY", access_key)
    monkeypatch.setattr(panel.settings, "S3_SECRET_KEY", secret_key)

    connection = _RecordingConnection()

    @contextlib.contextmanager
    def _fake_session(_db_path, *, load_httpfs=False):
        yield connection

    monkeypatch.setattr(panel, "duckdb_session", _fake_session)

    async def fake_runner(args: Any) -> dict[str, Any]:
        payload = _ok_payload(args)
        payload["ok"] = False
        payload["failed_stage"] = "era5_ingest"
        payload["error"] = (
            "IOException: IO Error: Unable to connect to URL "
            "'http://rustfs:9000/deposium/climate/era5/montpellier.parquet': 403 (Forbidden) "
            f"<AWSAccessKeyId>{access_key}</AWSAccessKeyId> "
            f"signature for {secret_key} rejected"
        )
        return payload

    args = panel.build_parser().parse_args(
        [
            "--zone-id",
            "montpellier",
            "--storage-mode",
            "s3",
            "--output-dir",
            tmp_path.as_posix(),
            "--run-id",
            "credential-leak-guard",
        ]
    )

    result = asyncio.run(panel.run_panel(args, runner=fake_runner))

    copies = [(sql, params) for sql, params in connection.calls if sql.startswith("COPY ")]
    assert len(copies) == 1, "the real _write_s3_json must have run"
    sql, values = copies[0]
    # The object really was going to the confined run-manifest prefix.
    assert result["run_manifest_uri"] in sql
    assert "s3://deposium/climate/runs/" in sql
    published = json.dumps(values, default=str)

    assert access_key not in published
    assert secret_key not in published
    assert "[REDACTED]" in published
    # The address is not a secret and must survive, or the manifest loses its provenance.
    assert "rustfs:9000" in published
    # And the structural fields are intact: the run id must still be the one the run used.
    assert result["run_id"] in published


def test_the_same_credential_never_reaches_the_local_run_manifest_either(
    tmp_path: Path, monkeypatch
):
    """The S3 copy was scrubbed and the LOCAL copy of the same payload was not.

    `_write_s3_json` redacted; `_write_json` was a bare `path.write_text(json.dumps(...))`.
    A local run manifest is an artifact too — it is what an operator opens, attaches to a
    ticket, or a CI job uploads — so the two spellings of one run manifest must not disagree
    about what they carry. Measured before the fix, with the repo's own dev credentials:

        local file bytes: LEAK[deposium-dev-secret] = True

    Every JSON file the panel wrote is checked, not just the manifest: `run_panel` also writes
    the panel summary and a per-zone summary out of the same payload.
    """
    access_key = "deposium-dev"
    secret_key = "deposium-dev-secret"
    monkeypatch.setattr(panel.settings, "S3_BUCKET", "deposium")
    monkeypatch.setattr(panel.settings, "S3_ENDPOINT", "http://rustfs:9000")
    monkeypatch.setattr(panel.settings, "S3_ACCESS_KEY", access_key)
    monkeypatch.setattr(panel.settings, "S3_SECRET_KEY", secret_key)

    async def fake_runner(args: Any) -> dict[str, Any]:
        payload = _ok_payload(args)
        payload["ok"] = False
        payload["failed_stage"] = "era5_ingest"
        payload["error"] = (
            "IOException: IO Error: Unable to connect to URL "
            "'http://rustfs:9000/deposium/climate/era5/montpellier.parquet': 403 (Forbidden) "
            f"<AWSAccessKeyId>{access_key}</AWSAccessKeyId> "
            f"signature for {secret_key} rejected"
        )
        return payload

    args = panel.build_parser().parse_args(
        [
            "--zone-id",
            "montpellier",
            "--storage-mode",
            "local",
            "--output-dir",
            tmp_path.as_posix(),
            "--run-id",
            "local-credential-leak-guard",
        ]
    )

    result = asyncio.run(panel.run_panel(args, runner=fake_runner))

    written = sorted(tmp_path.rglob("*.json"))
    assert written, "the panel must have written local JSON artifacts"
    for path in written:
        text = path.read_text()
        assert secret_key not in text, f"{path.name} leaks the configured S3 secret key"
        assert access_key not in text, f"{path.name} leaks the configured S3 access key"
    # The redaction is scoped to prose: the run id and the endpoint address still resolve.
    joined = "\n".join(p.read_text() for p in written)
    assert "[REDACTED]" in joined
    assert "rustfs:9000" in joined
    assert result["run_id"] in joined


# --- full-panel is the panel + pilot roles; a client site never enters it ---

_COMMITTED_ZONES = _PANEL_PATH.parents[1] / "zones" / "zones.geojson"


def _committed_features() -> list[dict[str, Any]]:
    return json.loads(_COMMITTED_ZONES.read_text())["features"]


def _client_feature(zone_id: str = "client_probe", role: str | None = "client") -> dict:
    properties: dict[str, Any] = {
        "zone_id": zone_id,
        "name": "Client probe",
        "koppen": "Csa",
        "bbox": [4.8, 43.35, 5.0, 43.5],
    }
    if role is not None:
        properties["role"] = role
    ring = [[4.8, 43.35], [5.0, 43.35], [5.0, 43.5], [4.8, 43.5], [4.8, 43.35]]
    return {
        "type": "Feature",
        "properties": properties,
        "geometry": {"type": "Polygon", "coordinates": [ring]},
    }


def _write_zones(tmp_path: Path, features: list[dict[str, Any]]) -> Path:
    payload = json.loads(_COMMITTED_ZONES.read_text())
    payload["features"] = features
    path = tmp_path / "zones.geojson"
    path.write_text(json.dumps(payload))
    return path


def test_full_panel_excludes_a_client_site_and_keeps_the_b1_order(tmp_path: Path):
    path = _write_zones(tmp_path, [*_committed_features(), _client_feature()])

    zone_ids = panel._load_full_panel_zone_ids(path)

    assert zone_ids == panel._load_full_panel_zone_ids()
    assert "client_probe" not in zone_ids


@pytest.mark.parametrize("role", [None, "panle"], ids=["missing-role", "typo-role"])
def test_full_panel_rejects_a_missing_or_unknown_role(tmp_path: Path, role: str | None):
    path = _write_zones(tmp_path, [*_committed_features(), _client_feature(role=role)])

    with pytest.raises(RuntimeError, match="client_probe.*role"):
        panel._load_full_panel_zone_ids(path)


def test_full_panel_rejects_a_panel_zone_relabelled_client(tmp_path: Path):
    features = _committed_features()
    features[0]["properties"]["role"] = "client"
    path = _write_zones(tmp_path, [*features, _client_feature()])

    with pytest.raises(RuntimeError, match=r"12 panel \+ 3 pilot"):
        panel._load_full_panel_zone_ids(path)


def test_full_panel_rejects_a_pilot_zone_relabelled_panel(tmp_path: Path):
    # 13 panel + 2 pilot still totals 15: only the per-role count can refuse it
    features = _committed_features()
    next(f for f in features if f["properties"]["role"] == "pilot")["properties"]["role"] = "panel"
    path = _write_zones(tmp_path, [*features, _client_feature()])

    with pytest.raises(RuntimeError, match=r"12 panel \+ 3 pilot zones, got 13 panel \+ 2 pilot"):
        panel._load_full_panel_zone_ids(path)


def test_full_panel_rejects_a_client_reusing_a_panel_zone_id(tmp_path: Path):
    # normals are keyed by zone_id alone (climate/normals/{zone_id}.parquet), so a client that
    # reuses a panel id would overwrite shared scientific normals
    path = _write_zones(tmp_path, [*_committed_features(), _client_feature("montpellier")])

    with pytest.raises(RuntimeError, match="duplicate zone_id"):
        panel._load_full_panel_zone_ids(path)


def test_a_generated_client_site_stays_out_of_full_panel(tmp_path: Path, monkeypatch):
    spec = importlib.util.spec_from_file_location(
        "generate_zones", _PANEL_PATH.parent / "generate_zones.py"
    )
    generator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(generator)
    monkeypatch.setattr(
        generator, "CLIENTS", [("client_probe", "Client probe", 4.8, 43.35, 5.0, 43.5, "Csa")]
    )
    generator.write_geojson(generator.build_features(), tmp_path)

    zone_ids = panel._load_full_panel_zone_ids(tmp_path / "zones.geojson")

    assert zone_ids == panel._load_full_panel_zone_ids()
    assert "client_probe" not in zone_ids
