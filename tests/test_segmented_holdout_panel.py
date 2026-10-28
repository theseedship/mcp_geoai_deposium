"""Network-free tests for scripts/segmented_holdout_panel.py."""

from __future__ import annotations

import asyncio
import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest

_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "segmented_holdout_panel.py"
_spec = importlib.util.spec_from_file_location("segmented_holdout_panel", _SCRIPT_PATH)
panel = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(panel)

pytestmark = pytest.mark.usefixtures("allow_local_parquet_paths")


def _args(*values: str):
    return panel.build_parser().parse_args(list(values))


async def _must_not_run(*_args, **_kwargs):
    raise AssertionError("preflight must not call stage runners")


async def _no_sleep(_seconds: float) -> None:
    return None


def _set_s3_bucket(monkeypatch, bucket: str = "deposium") -> None:
    monkeypatch.setattr(panel.holdout_pilot_panel.unified_pilot_panel.settings, "S3_BUCKET", bucket)
    monkeypatch.setattr(
        panel.holdout_pilot_panel.climate_baseline_panel.settings, "S3_BUCKET", bucket
    )
    monkeypatch.setattr(panel.climate_baseline_panel.settings, "S3_BUCKET", bucket)


def _ok_era5(args: Any) -> dict[str, Any]:
    era5_output = (
        args.era5_output or f"s3://deposium/climate/holdout/era5/2021_2025/{args.zone_id}.parquet"
    )
    return {
        "ok": True,
        "zone_id": args.zone_id,
        "window": {"start_date": args.start_date, "end_date": args.end_date},
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


def _ok_unified(kwargs: dict[str, Any], *, warning: str | None = None) -> dict[str, Any]:
    metadata: dict[str, Any] = {"era5_source": kwargs["era5_source"]}
    if warning:
        metadata["ndvi_truncation_warning"] = warning
    return {
        "zone_id": kwargs["zone_id"],
        "output": kwargs["output_path"],
        "statistics": {
            "n_era5_days": 1826,
            "n_ndvi_observations": 8,
            "n_matched": 5,
            "images_searched": 8,
        },
        "metadata": metadata,
    }


class TestSegmentedHoldoutPanel:
    def test_preflight_plans_monthly_q3_segments_and_holdout_paths(
        self, tmp_path: Path, monkeypatch
    ):
        _set_s3_bucket(monkeypatch)
        args = _args(
            "--preflight-only",
            "--skip-era5",
            "--zone-id",
            "riyadh",
            "--output-dir",
            tmp_path.as_posix(),
            "--run-id",
            "segmented-proof",
        )

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
        assert result["planned_segments"] == [
            {"label": "2023_07", "window": {"start_date": "2023-07-01", "end_date": "2023-07-31"}},
            {"label": "2023_08", "window": {"start_date": "2023-08-01", "end_date": "2023-08-31"}},
            {"label": "2023_09", "window": {"start_date": "2023-09-01", "end_date": "2023-09-30"}},
        ]
        assert result["completed_segment_labels"] == ["2023_07", "2023_08", "2023_09"]
        assert result["stages"]["era5_holdout"] is None

        segment_manifest = json.loads(
            (
                tmp_path
                / "segments"
                / "2023_07"
                / "unified-holdout"
                / "unified-holdout-manifest.json"
            ).read_text()
        )
        assert segment_manifest["artifacts"][0]["era5_source"] == (
            "s3://deposium/climate/holdout/era5/2021_2025/riyadh.parquet"
        )
        assert segment_manifest["artifacts"][0]["unified_output"] == (
            "s3://deposium/climate/holdout/unified/2023_07/riyadh.parquet"
        )

    def test_preflight_labels_partial_month_segments_with_date_ranges(
        self, tmp_path: Path, monkeypatch
    ):
        _set_s3_bucket(monkeypatch)
        args = _args(
            "--preflight-only",
            "--skip-era5",
            "--zone-id",
            "riyadh",
            "--unified-start-date",
            "2023-07-15",
            "--unified-end-date",
            "2023-08-15",
            "--output-dir",
            tmp_path.as_posix(),
            "--run-id",
            "segmented-partial-proof",
        )

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
        assert result["planned_segments"] == [
            {
                "label": "2023_07_15_2023_07_31",
                "window": {"start_date": "2023-07-15", "end_date": "2023-07-31"},
            },
            {
                "label": "2023_08_01_2023_08_15",
                "window": {"start_date": "2023-08-01", "end_date": "2023-08-15"},
            },
        ]

        segment_manifest = json.loads(
            (
                tmp_path
                / "segments"
                / "2023_07_15_2023_07_31"
                / "unified-holdout"
                / "unified-holdout-manifest.json"
            ).read_text()
        )
        assert segment_manifest["artifacts"][0]["unified_output"] == (
            "s3://deposium/climate/holdout/unified/2023_07_15_2023_07_31/riyadh.parquet"
        )

    def test_run_materializes_era5_once_then_runs_each_segment(self, tmp_path: Path, monkeypatch):
        _set_s3_bucket(monkeypatch)
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
            "riyadh",
            "--output-dir",
            tmp_path.as_posix(),
            "--run-id",
            "segmented-proof",
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
        assert len(era5_calls) == 1
        assert era5_calls[0].start_date == "2021-01-01"
        assert era5_calls[0].end_date == "2025-12-31"
        assert len(unified_calls) == 3
        assert [(c["start_date"], c["end_date"]) for c in unified_calls] == [
            ("2023-07-01", "2023-07-31"),
            ("2023-08-01", "2023-08-31"),
            ("2023-09-01", "2023-09-30"),
        ]
        assert unified_calls[0]["era5_source"] == (
            "s3://deposium/climate/holdout/era5/2021_2025/riyadh.parquet"
        )
        assert unified_calls[1]["output_path"] == (
            "s3://deposium/climate/holdout/unified/2023_08/riyadh.parquet"
        )
        assert result["totals"] == {
            "images_searched": 24,
            "n_ndvi_observations": 24,
            "n_matched": 15,
        }

    def test_skip_era5_runs_only_segmented_b4(self, tmp_path: Path, monkeypatch):
        _set_s3_bucket(monkeypatch)
        unified_calls: list[dict[str, Any]] = []

        async def fake_unified(**kwargs) -> dict[str, Any]:
            unified_calls.append(kwargs)
            return _ok_unified(kwargs)

        args = _args(
            "--skip-era5",
            "--zone-id",
            "riyadh",
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
        assert len(unified_calls) == 3

    def test_local_segments_use_shared_era5_stage_source(self, tmp_path: Path, monkeypatch):
        _set_s3_bucket(monkeypatch, "")
        unified_calls: list[dict[str, Any]] = []

        async def fake_era5(args: Any) -> dict[str, Any]:
            return _ok_era5(args)

        async def fake_unified(**kwargs) -> dict[str, Any]:
            unified_calls.append(kwargs)
            return _ok_unified(kwargs)

        args = _args(
            "--storage-mode",
            "local",
            "--zone-id",
            "riyadh",
            "--output-dir",
            tmp_path.as_posix(),
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
        assert (
            unified_calls[0]["era5_source"]
            == (
                tmp_path
                / "era5-stage"
                / "era5-holdout"
                / "riyadh-era5-2021-01-01_2025-12-31.parquet"
            ).as_posix()
        )
        assert unified_calls[1]["era5_source"] == unified_calls[0]["era5_source"]
        assert "/segments/2023_08/era5-holdout/" not in unified_calls[1]["era5_source"]

    def test_truncation_warnings_are_aggregated_by_segment_and_zone(
        self, tmp_path: Path, monkeypatch
    ):
        _set_s3_bucket(monkeypatch)

        async def fake_era5(args: Any) -> dict[str, Any]:
            return _ok_era5(args)

        async def fake_unified(**kwargs) -> dict[str, Any]:
            warning = None
            if kwargs["start_date"] == "2023-08-01":
                warning = "NDVI search hit the 50-item STAC cap"
            return _ok_unified(kwargs, warning=warning)

        args = _args(
            "--zone-id",
            "riyadh",
            "--output-dir",
            tmp_path.as_posix(),
            "--run-id",
            "segmented-proof",
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
        assert result["truncation_warning_count"] == 1
        assert result["truncation_warnings"] == [
            {
                "segment_label": "2023_08",
                "zone_id": "riyadh",
                "warning": "NDVI search hit the 50-item STAC cap",
            }
        ]

    def test_fail_fast_stops_after_first_failed_segment(self, tmp_path: Path, monkeypatch):
        _set_s3_bucket(monkeypatch)
        unified_calls: list[str] = []

        async def fake_era5(args: Any) -> dict[str, Any]:
            return _ok_era5(args)

        async def fake_unified(**kwargs) -> dict[str, Any]:
            unified_calls.append(kwargs["start_date"])
            return {"error": "STAC unavailable"}

        args = _args(
            "--zone-id",
            "riyadh",
            "--output-dir",
            tmp_path.as_posix(),
            "--fail-fast",
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

        assert result["ok"] is False
        assert result["stop_reason"] == "segment_failed"
        assert result["completed_segment_labels"] == ["2023_07"]
        assert result["failed_segment_labels"] == ["2023_07"]
        assert unified_calls == ["2023-07-01"]

    def test_full_panel_zone_set_loads_all_b1_zones(self, tmp_path: Path, monkeypatch):
        _set_s3_bucket(monkeypatch)
        args = _args(
            "--preflight-only",
            "--era5-only",
            "--zone-set",
            "full-panel",
            "--output-dir",
            tmp_path.as_posix(),
        )

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
        assert result["zone_set"] == "full-panel"
        assert result["zone_ids"] == panel.climate_baseline_panel._load_full_panel_zone_ids()
        assert len(result["zone_ids"]) == 15
        assert result["planned_segments"] == []
        assert result["stages"]["era5_holdout"]["ok"] is True

    def test_unified_window_must_be_inside_holdout_window(self, tmp_path: Path, monkeypatch):
        _set_s3_bucket(monkeypatch)
        args = _args(
            "--skip-era5",
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
