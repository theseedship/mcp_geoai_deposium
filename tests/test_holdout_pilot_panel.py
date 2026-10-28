"""Network-free tests for scripts/holdout_pilot_panel.py."""

from __future__ import annotations

import asyncio
import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest

_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "holdout_pilot_panel.py"
_spec = importlib.util.spec_from_file_location("holdout_pilot_panel", _SCRIPT_PATH)
panel = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(panel)

pytestmark = pytest.mark.usefixtures("allow_local_parquet_paths")


def _args(*values: str):
    return panel.build_parser().parse_args(list(values))


async def _must_not_run(*_args, **_kwargs):
    raise AssertionError("preflight must not call stage runners")


async def _no_sleep(_seconds: float) -> None:
    return None


def _ok_era5(args: Any) -> dict[str, Any]:
    era5_output = (
        args.era5_output or f"s3://deposium/climate/holdout/era5/2021_2025/{args.zone_id}.parquet"
    )
    return {
        "ok": True,
        "zone_id": args.zone_id,
        "window": {
            "start_date": args.start_date,
            "end_date": args.end_date,
            "label": "custom",
        },
        "variables": args.variables or list(panel.climate_baseline_panel.NORMALIZED_VARS),
        "skip_normals": True,
        "storage_mode": args.storage_mode,
        "duration_ms": 1,
        "requested_outputs": {"era5": args.era5_output, "normals": None},
        "resolved_outputs": {"era5": era5_output, "normals": None},
        "stages": {
            "era5_ingest": {
                "output": era5_output,
                "start_date": args.start_date,
                "end_date": args.end_date,
                "n_days": panel.climate_baseline_panel._expected_days(
                    args.start_date, args.end_date
                ),
                "pct_days": 100.0,
                "min_variable_pct_complete": 100.0,
            }
        },
        "gates": {"completeness": {"passed": True, "min_completeness_pct": 98.0}},
    }


def _ok_unified(kwargs: dict[str, Any]) -> dict[str, Any]:
    return {
        "zone_id": kwargs["zone_id"],
        "output": kwargs["output_path"],
        "statistics": {
            "n_era5_days": 1826,
            "n_ndvi_observations": 8,
            "n_matched": 5,
            "images_searched": 8,
        },
        "metadata": {"era5_source": kwargs["era5_source"]},
    }


class TestHoldoutPilotPanel:
    def test_preflight_default_plans_era5_and_unified_holdout_paths(
        self, tmp_path: Path, monkeypatch
    ):
        monkeypatch.setattr(panel.unified_pilot_panel.settings, "S3_BUCKET", "deposium")
        monkeypatch.setattr(panel.climate_baseline_panel.settings, "S3_BUCKET", "deposium")
        args = _args("--preflight-only", "--output-dir", tmp_path.as_posix())

        result = asyncio.run(
            panel.run_panel(
                args,
                era5_runner=_must_not_run,
                unified_runner=_must_not_run,
                sleeper=_no_sleep,
                run_manifest_writer=lambda *_args: None,
            )
        )

        assert result["ok"] is True
        assert result["artifact_window_label"] == "2021_2025"
        assert result["era5_artifact_window_label"] == "2021_2025"
        assert result["unified_artifact_window_label"] == "2023_07_01_2023_07_31"
        assert result["holdout_window"] == {"start_date": "2021-01-01", "end_date": "2025-12-31"}
        assert result["unified_window"] == {"start_date": "2023-07-01", "end_date": "2023-07-31"}
        assert result["stages"]["era5_holdout"]["ok"] is True
        assert result["stages"]["unified_holdout"]["ok"] is True

        era5_manifest = json.loads(
            (tmp_path / "era5-holdout" / "era5-holdout-manifest.json").read_text()
        )
        unified_manifest = json.loads(
            (tmp_path / "unified-holdout" / "unified-holdout-manifest.json").read_text()
        )
        assert era5_manifest["artifacts"][0]["planned_outputs"] == {
            "era5": "s3://deposium/climate/holdout/era5/2021_2025/montpellier_metropole.parquet",
            "normals": None,
        }
        assert unified_manifest["artifacts"][0]["era5_source"] == (
            "s3://deposium/climate/holdout/era5/2021_2025/montpellier_metropole.parquet"
        )
        assert unified_manifest["artifacts"][0]["unified_output"] == (
            "s3://deposium/climate/holdout/unified/2023_07_01_2023_07_31/montpellier_metropole.parquet"
        )

    def test_run_materializes_era5_then_unified_with_suffixed_run_ids(
        self, tmp_path: Path, monkeypatch
    ):
        monkeypatch.setattr(panel.unified_pilot_panel.settings, "S3_BUCKET", "deposium")
        monkeypatch.setattr(panel.climate_baseline_panel.settings, "S3_BUCKET", "deposium")
        era5_calls: list[Any] = []
        unified_calls: list[dict[str, Any]] = []

        async def fake_era5(args: Any) -> dict[str, Any]:
            era5_calls.append(args)
            return _ok_era5(args)

        async def fake_unified(**kwargs) -> dict[str, Any]:
            unified_calls.append(kwargs)
            return _ok_unified(kwargs)

        args = _args(
            "--zone-id",
            "montpellier",
            "--output-dir",
            tmp_path.as_posix(),
            "--run-id",
            "holdout-proof",
        )

        result = asyncio.run(
            panel.run_panel(
                args,
                era5_runner=fake_era5,
                unified_runner=fake_unified,
                sleeper=_no_sleep,
                run_manifest_writer=lambda *_args: None,
            )
        )

        assert result["ok"] is True
        assert result["run_id"] == "holdout-proof"
        assert result["stages"]["era5_holdout"]["run_id"] == "holdout-proof-era5"
        assert result["stages"]["unified_holdout"]["run_id"] == "holdout-proof-unified"
        assert era5_calls[0].skip_normals is True
        assert era5_calls[0].era5_output == (
            "s3://deposium/climate/holdout/era5/2021_2025/montpellier.parquet"
        )
        assert era5_calls[0].archive_chunk_days == 1826
        assert unified_calls[0]["era5_source"] == (
            "s3://deposium/climate/holdout/era5/2021_2025/montpellier.parquet"
        )
        assert unified_calls[0]["output_path"] == (
            "s3://deposium/climate/holdout/unified/2023_07_01_2023_07_31/montpellier.parquet"
        )

        summary = json.loads((tmp_path / "holdout-pilot-summary.json").read_text())
        manifest = json.loads((tmp_path / "holdout-pilot-manifest.json").read_text())
        assert summary["kind"] == "holdout_pilot_panel_summary"
        assert manifest["kind"] == "holdout_pilot_panel_manifest"
        assert manifest["stages"]["era5_holdout"]["run_manifest_uri"] == (
            "s3://deposium/climate/runs/holdout-proof-era5/manifest.json"
        )

    def test_skip_era5_runs_only_unified_against_existing_holdout(
        self, tmp_path: Path, monkeypatch
    ):
        monkeypatch.setattr(panel.unified_pilot_panel.settings, "S3_BUCKET", "deposium")
        monkeypatch.setattr(panel.climate_baseline_panel.settings, "S3_BUCKET", "deposium")
        unified_calls: list[dict[str, Any]] = []

        async def fake_unified(**kwargs) -> dict[str, Any]:
            unified_calls.append(kwargs)
            return _ok_unified(kwargs)

        args = _args(
            "--skip-era5",
            "--zone-id",
            "montpellier",
            "--output-dir",
            tmp_path.as_posix(),
        )

        result = asyncio.run(
            panel.run_panel(
                args,
                era5_runner=_must_not_run,
                unified_runner=fake_unified,
                sleeper=_no_sleep,
                run_manifest_writer=lambda *_args: None,
            )
        )

        assert result["ok"] is True
        assert result["skip_era5"] is True
        assert result["stages"]["era5_holdout"] is None
        assert len(unified_calls) == 1
        assert unified_calls[0]["era5_source"].endswith(
            "/climate/holdout/era5/2021_2025/montpellier.parquet"
        )

    def test_explicit_era5_source_template_is_passed_to_unified_stage(
        self, tmp_path: Path, monkeypatch
    ):
        monkeypatch.setattr(panel.unified_pilot_panel.settings, "S3_BUCKET", "deposium")
        monkeypatch.setattr(panel.climate_baseline_panel.settings, "S3_BUCKET", "deposium")
        unified_calls: list[dict[str, Any]] = []

        async def fake_unified(**kwargs) -> dict[str, Any]:
            unified_calls.append(kwargs)
            return _ok_unified(kwargs)

        args = _args(
            "--skip-era5",
            "--zone-id",
            "montpellier",
            "--era5-source-template",
            "s3://deposium/climate/holdout/era5/custom/{zone_id}.parquet",
            "--output-dir",
            tmp_path.as_posix(),
        )

        result = asyncio.run(
            panel.run_panel(
                args,
                era5_runner=_must_not_run,
                unified_runner=fake_unified,
                sleeper=_no_sleep,
                run_manifest_writer=lambda *_args: None,
            )
        )

        assert result["ok"] is True
        assert unified_calls[0]["era5_source"] == (
            "s3://deposium/climate/holdout/era5/custom/montpellier.parquet"
        )

    def test_local_explicit_era5_source_template_disables_run_manifest(
        self, tmp_path: Path, monkeypatch
    ):
        monkeypatch.setattr(panel.unified_pilot_panel.settings, "S3_BUCKET", "deposium")
        monkeypatch.setattr(panel.climate_baseline_panel.settings, "S3_BUCKET", "deposium")
        run_manifest_calls: list[tuple[str, dict[str, Any]]] = []

        async def fake_unified(**kwargs) -> dict[str, Any]:
            return _ok_unified(kwargs)

        args = _args(
            "--storage-mode",
            "local",
            "--skip-era5",
            "--zone-id",
            "montpellier",
            "--era5-source-template",
            (tmp_path / "era5" / "{zone_id}.parquet").as_posix(),
            "--output-dir",
            tmp_path.as_posix(),
        )

        result = asyncio.run(
            panel.run_panel(
                args,
                era5_runner=_must_not_run,
                unified_runner=fake_unified,
                sleeper=_no_sleep,
                run_manifest_writer=lambda uri, payload: run_manifest_calls.append((uri, payload)),
            )
        )

        assert result["ok"] is True
        assert run_manifest_calls == []
        assert result["stages"]["unified_holdout"]["run_manifest_uri"] is None

    def test_era5_failure_stops_before_unified(self, tmp_path: Path, monkeypatch):
        monkeypatch.setattr(panel.unified_pilot_panel.settings, "S3_BUCKET", "deposium")
        monkeypatch.setattr(panel.climate_baseline_panel.settings, "S3_BUCKET", "deposium")

        async def failing_era5(_args: Any) -> dict[str, Any]:
            return {"error": "429 Too Many Requests"}

        args = _args("--zone-id", "montpellier", "--output-dir", tmp_path.as_posix())

        result = asyncio.run(
            panel.run_panel(
                args,
                era5_runner=failing_era5,
                unified_runner=_must_not_run,
                sleeper=_no_sleep,
                run_manifest_writer=lambda *_args: None,
            )
        )

        assert result["ok"] is False
        assert result["stop_reason"] == "era5_failed"
        assert result["stages"]["unified_holdout"] is None

    def test_rate_limit_stops_era5_stage_before_later_zones(self, tmp_path: Path, monkeypatch):
        monkeypatch.setattr(panel.unified_pilot_panel.settings, "S3_BUCKET", "deposium")
        monkeypatch.setattr(panel.climate_baseline_panel.settings, "S3_BUCKET", "deposium")
        calls: list[str] = []

        async def fake_era5(args: Any) -> dict[str, Any]:
            calls.append(args.zone_id)
            return {
                "ok": False,
                "zone_id": args.zone_id,
                "duration_ms": 1,
                "failed_stage": "era5_ingest",
                "error": "Client error '429 Too Many Requests'",
                "stages": {},
            }

        args = _args(
            "--zone-id",
            "riyadh",
            "seoul",
            "--output-dir",
            tmp_path.as_posix(),
        )

        result = asyncio.run(
            panel.run_panel(
                args,
                era5_runner=fake_era5,
                unified_runner=_must_not_run,
                sleeper=_no_sleep,
                run_manifest_writer=lambda *_args: None,
            )
        )

        assert result["ok"] is False
        assert result["stop_reason"] == "era5_failed"
        assert result["stages"]["era5_holdout"]["stop_reason"] == "rate_limit"
        assert calls == ["riyadh"]

    def test_local_unified_stage_does_not_publish_remote_run_manifest(
        self, tmp_path: Path, monkeypatch
    ):
        monkeypatch.setattr(panel.unified_pilot_panel.settings, "S3_BUCKET", "deposium")
        monkeypatch.setattr(panel.climate_baseline_panel.settings, "S3_BUCKET", "deposium")

        async def fake_era5(args: Any) -> dict[str, Any]:
            return _ok_era5(args)

        async def fake_unified(**kwargs) -> dict[str, Any]:
            return _ok_unified(kwargs)

        def fail_on_manifest_write(_uri: str, _payload: dict[str, Any]) -> None:
            raise AssertionError("local wrapper runs must not publish remote run manifests")

        args = _args(
            "--storage-mode",
            "local",
            "--zone-id",
            "montpellier",
            "--output-dir",
            tmp_path.as_posix(),
        )

        result = asyncio.run(
            panel.run_panel(
                args,
                era5_runner=fake_era5,
                unified_runner=fake_unified,
                sleeper=_no_sleep,
                run_manifest_writer=fail_on_manifest_write,
            )
        )

        assert result["ok"] is True
        assert result["stages"]["unified_holdout"]["run_manifest_uri"] is None
        assert result["stages"]["unified_holdout"]["run_manifest_written"] is None

    def test_era5_only_does_not_require_unified_window_inside_holdout(
        self, tmp_path: Path, monkeypatch
    ):
        monkeypatch.setattr(panel.unified_pilot_panel.settings, "S3_BUCKET", "deposium")
        monkeypatch.setattr(panel.climate_baseline_panel.settings, "S3_BUCKET", "deposium")
        era5_calls: list[Any] = []

        async def fake_era5(args: Any) -> dict[str, Any]:
            era5_calls.append(args)
            return _ok_era5(args)

        args = _args(
            "--era5-only",
            "--zone-id",
            "montpellier",
            "--holdout-start-date",
            "2024-01-01",
            "--holdout-end-date",
            "2024-12-31",
            "--output-dir",
            tmp_path.as_posix(),
        )

        result = asyncio.run(
            panel.run_panel(
                args,
                era5_runner=fake_era5,
                unified_runner=_must_not_run,
                sleeper=_no_sleep,
                run_manifest_writer=lambda *_args: None,
            )
        )

        assert result["ok"] is True
        assert result["era5_only"] is True
        assert result["stages"]["unified_holdout"] is None
        assert era5_calls[0].start_date == "2024-01-01"
        assert era5_calls[0].end_date == "2024-12-31"

    def test_era5_only_failure_drives_panel_failure(self, tmp_path: Path, monkeypatch):
        monkeypatch.setattr(panel.unified_pilot_panel.settings, "S3_BUCKET", "deposium")
        monkeypatch.setattr(panel.climate_baseline_panel.settings, "S3_BUCKET", "deposium")

        async def failing_era5(_args: Any) -> dict[str, Any]:
            return {"ok": False, "error": "archive unavailable"}

        args = _args(
            "--era5-only",
            "--zone-id",
            "montpellier",
            "--output-dir",
            tmp_path.as_posix(),
        )

        result = asyncio.run(
            panel.run_panel(
                args,
                era5_runner=failing_era5,
                unified_runner=_must_not_run,
                sleeper=_no_sleep,
                run_manifest_writer=lambda *_args: None,
            )
        )

        assert result["ok"] is False
        assert result["stop_reason"] == "era5_failed"
        assert result["stages"]["unified_holdout"] is None

    def test_unified_window_must_be_inside_holdout_window(self, tmp_path: Path, monkeypatch):
        monkeypatch.setattr(panel.unified_pilot_panel.settings, "S3_BUCKET", "deposium")
        monkeypatch.setattr(panel.climate_baseline_panel.settings, "S3_BUCKET", "deposium")
        args = _args(
            "--unified-start-date",
            "2020-12-01",
            "--unified-end-date",
            "2021-01-31",
            "--output-dir",
            tmp_path.as_posix(),
        )

        try:
            asyncio.run(
                panel.run_panel(
                    args,
                    era5_runner=_must_not_run,
                    unified_runner=_must_not_run,
                    sleeper=_no_sleep,
                    run_manifest_writer=lambda *_args: None,
                )
            )
        except ValueError as exc:
            assert "unified window must be inside" in str(exc)
        else:
            raise AssertionError("out-of-range unified window should fail before stage runs")
