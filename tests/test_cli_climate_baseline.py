"""Tests for the FIERCE climate baseline CLI."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

from src.cli.climate_baseline import build_parser, run, run_pipeline
from src.utils import openmeteo as openmeteo_utils
from src.utils.openmeteo import NORMALIZED_VARS

pytestmark = pytest.mark.usefixtures("allow_local_parquet_paths")


def test_parser_defaults_to_wmo_window_and_all_variables():
    args = build_parser().parse_args(["--zone-id", "montpellier"])

    assert args.zone_id == "montpellier"
    assert args.start_date is None and args.end_date is None
    assert args.variables is None
    assert args.overwrite is False
    assert args.dry_run is False
    assert args.skip_normals is False


def test_pipeline_orchestrates_ingest_then_normals(tmp_path: Path):
    calls: dict[str, dict[str, Any]] = {}
    era5 = (tmp_path / "era5.parquet").as_posix()
    normals = (tmp_path / "normals.parquet").as_posix()

    async def fake_ingest(**kwargs: Any) -> dict[str, Any]:
        calls["ingest"] = kwargs
        return {
            "zone_id": kwargs["zone_id"],
            "start_date": kwargs["start_date"],
            "end_date": kwargs["end_date"],
            "n_days": 3,
            "variables": kwargs["variables"],
            "output": kwargs["output_path"],
            "completeness": {
                "expected_days": 3,
                "received_days": 3,
                "pct_days": 100.0,
                "per_variable": {
                    "temperature_2m": {"received": 3, "pct_complete": 100.0},
                    "precipitation": {"received": 2, "pct_complete": 66.67},
                },
            },
            "metadata": {"elapsed_ms": 12},
        }

    async def fake_percentiles(**kwargs: Any) -> dict[str, Any]:
        calls["percentiles"] = kwargs
        return {
            "zone_id": kwargs["zone_id"],
            "output": kwargs["output_path"],
            "variables": kwargs["variables"],
            "statistics": {"n_normals": 24, "n_months": 12, "n_variables": 2},
            "metadata": {
                "elapsed_ms": 9,
                "source": "era5_parquet",
                "normal_grain": "daily_values_by_calendar_month",
            },
        }

    args = build_parser().parse_args(
        [
            "--zone-id",
            "montpellier",
            "--start-date",
            "2020-01-01",
            "--end-date",
            "2020-01-03",
            "--variables",
            "temperature_2m",
            "precipitation",
            "--era5-output",
            era5,
            "--normals-output",
            normals,
            "--overwrite",
        ]
    )

    payload = asyncio.run(
        run_pipeline(
            args,
            tools={"era5_ingest": fake_ingest, "climate_percentiles": fake_percentiles},
        )
    )

    assert payload["ok"] is True
    assert payload["window"] == {
        "start_date": "2020-01-01",
        "end_date": "2020-01-03",
        "label": "custom",
    }
    assert payload["variables"] == ["temperature_2m", "precipitation"]
    assert calls["ingest"]["write_output"] is True
    assert calls["ingest"]["output_path"] == era5
    assert calls["percentiles"]["era5_source"] == era5
    assert calls["percentiles"]["output_path"] == normals
    assert payload["stages"]["era5_ingest"]["min_variable_pct_complete"] == 66.67
    assert payload["stages"]["climate_percentiles"]["source"] == "era5_parquet"
    assert payload["stages"]["climate_percentiles"]["n_normals"] == 24


def test_pipeline_skip_normals_materializes_era5_only(tmp_path: Path):
    calls: dict[str, dict[str, Any]] = {}
    era5 = (tmp_path / "era5.parquet").as_posix()

    async def fake_ingest(**kwargs: Any) -> dict[str, Any]:
        calls["ingest"] = kwargs
        return {
            "zone_id": kwargs["zone_id"],
            "start_date": kwargs["start_date"],
            "end_date": kwargs["end_date"],
            "n_days": 3,
            "variables": kwargs["variables"],
            "output": kwargs["output_path"],
            "completeness": {
                "expected_days": 3,
                "received_days": 3,
                "pct_days": 100.0,
                "per_variable": {
                    "temperature_2m": {"received": 3, "pct_complete": 100.0},
                },
            },
            "metadata": {"elapsed_ms": 12},
        }

    async def should_not_call_percentiles(**_kwargs: Any) -> dict[str, Any]:
        raise AssertionError("climate_percentiles must not run with --skip-normals")

    args = build_parser().parse_args(
        [
            "--zone-id",
            "montpellier",
            "--start-date",
            "2021-01-01",
            "--end-date",
            "2021-01-03",
            "--variables",
            "temperature_2m",
            "--era5-output",
            era5,
            "--skip-normals",
            "--min-completeness-pct",
            "98",
        ]
    )

    payload = asyncio.run(
        run_pipeline(
            args,
            tools={"era5_ingest": fake_ingest, "climate_percentiles": should_not_call_percentiles},
        )
    )

    assert payload["ok"] is True
    assert payload["skip_normals"] is True
    assert payload["requested_outputs"] == {"era5": era5, "normals": None}
    assert payload["resolved_outputs"] == {"era5": era5, "normals": None}
    assert calls["ingest"]["output_path"] == era5
    assert "climate_percentiles" not in payload["stages"]
    assert "skip-normals" in payload["notes"][0]
    assert payload["gates"]["completeness"]["passed"] is True
    assert payload["gates"]["completeness"]["era5"]["expected_days"] == 3


def test_skip_normals_requires_explicit_era5_output(monkeypatch):
    async def should_not_call_ingest(**_kwargs: Any) -> dict[str, Any]:
        raise AssertionError("era5_ingest must not run without explicit ERA5 holdout output")

    async def should_not_call_percentiles(**_kwargs: Any) -> dict[str, Any]:
        raise AssertionError("climate_percentiles must not run with --skip-normals")

    monkeypatch.setattr(
        "src.cli.climate_baseline._default_output_uri",
        lambda zone_id, kind: f"s3://deposium/climate/{kind}/{zone_id}.parquet",
    )
    args = build_parser().parse_args(
        [
            "--zone-id",
            "montpellier",
            "--start-date",
            "2021-01-01",
            "--end-date",
            "2025-12-31",
            "--skip-normals",
        ]
    )

    payload = asyncio.run(
        run_pipeline(
            args,
            tools={
                "era5_ingest": should_not_call_ingest,
                "climate_percentiles": should_not_call_percentiles,
            },
        )
    )

    assert payload["ok"] is False
    assert payload["failed_stage"] == "preflight"
    assert "--skip-normals never uses the canonical climate/era5 default" in payload["error"]
    assert payload["resolved_outputs"] == {"era5": None, "normals": None}


def test_archive_chunk_days_temporarily_overrides_openmeteo_chunk_guard(tmp_path: Path):
    calls: dict[str, int] = {}
    original = openmeteo_utils._MAX_DAYS_PER_REQUEST
    era5 = (tmp_path / "era5.parquet").as_posix()
    normals = (tmp_path / "normals.parquet").as_posix()

    async def fake_ingest(**kwargs: Any) -> dict[str, Any]:
        calls["ingest_chunk_days"] = openmeteo_utils._MAX_DAYS_PER_REQUEST
        assert openmeteo_utils.split_date_range("1991-01-01", "2020-12-31") == [
            ("1991-01-01", "2020-12-31")
        ]
        return {
            "zone_id": kwargs["zone_id"],
            "start_date": kwargs["start_date"],
            "end_date": kwargs["end_date"],
            "n_days": 10958,
            "variables": kwargs["variables"],
            "output": kwargs["output_path"],
            "completeness": {"pct_days": 100.0, "per_variable": {}},
            "metadata": {},
        }

    async def fake_percentiles(**kwargs: Any) -> dict[str, Any]:
        calls["percentiles_chunk_days"] = openmeteo_utils._MAX_DAYS_PER_REQUEST
        return {
            "zone_id": kwargs["zone_id"],
            "output": kwargs["output_path"],
            "variables": kwargs["variables"],
            "statistics": {"n_normals": 72, "n_months": 12, "n_variables": 6},
            "metadata": {"source": "era5_parquet"},
        }

    args = build_parser().parse_args(
        [
            "--zone-id",
            "montpellier",
            "--era5-output",
            era5,
            "--normals-output",
            normals,
            "--archive-chunk-days",
            "10958",
        ]
    )

    payload = asyncio.run(
        run_pipeline(
            args,
            tools={"era5_ingest": fake_ingest, "climate_percentiles": fake_percentiles},
        )
    )

    assert payload["ok"] is True
    assert payload["archive_chunk_days"] == 10958
    assert calls == {"ingest_chunk_days": 10958, "percentiles_chunk_days": 10958}
    assert openmeteo_utils._MAX_DAYS_PER_REQUEST == original


def test_completeness_gate_fails_after_materialization_when_below_threshold(tmp_path: Path):
    era5 = (tmp_path / "era5.parquet").as_posix()
    normals = (tmp_path / "normals.parquet").as_posix()

    async def fake_ingest(**kwargs: Any) -> dict[str, Any]:
        return {
            "zone_id": kwargs["zone_id"],
            "start_date": kwargs["start_date"],
            "end_date": kwargs["end_date"],
            "n_days": 3,
            "variables": kwargs["variables"],
            "output": kwargs["output_path"],
            "completeness": {
                "expected_days": 3,
                "received_days": 3,
                "pct_days": 100.0,
                "per_variable": {
                    "temperature_2m": {"received": 3, "pct_complete": 100.0},
                    "precipitation": {"received": 2, "pct_complete": 66.67},
                },
            },
            "metadata": {},
        }

    async def fake_percentiles(**kwargs: Any) -> dict[str, Any]:
        return {
            "zone_id": kwargs["zone_id"],
            "output": kwargs["output_path"],
            "variables": kwargs["variables"],
            "statistics": {"n_normals": 24, "n_months": 12, "n_variables": 2},
            "metadata": {"source": "era5_parquet"},
        }

    args = build_parser().parse_args(
        [
            "--zone-id",
            "montpellier",
            "--start-date",
            "2020-01-01",
            "--end-date",
            "2020-01-03",
            "--variables",
            "temperature_2m",
            "precipitation",
            "--era5-output",
            era5,
            "--normals-output",
            normals,
            "--min-completeness-pct",
            "98",
        ]
    )

    payload = asyncio.run(
        run_pipeline(
            args,
            tools={"era5_ingest": fake_ingest, "climate_percentiles": fake_percentiles},
        )
    )

    assert payload["ok"] is False
    assert payload["failed_stage"] == "completeness_gate"
    assert "min_variable_pct_complete 66.67 < 98.0" in payload["error"]
    assert payload["gates"]["completeness"]["passed"] is False
    assert payload["gates"]["completeness"]["era5"]["min_variable_pct_complete"] == 66.67


def test_completeness_gate_passes_for_full_normals(tmp_path: Path):
    era5 = (tmp_path / "era5.parquet").as_posix()
    normals = (tmp_path / "normals.parquet").as_posix()

    async def fake_ingest(**kwargs: Any) -> dict[str, Any]:
        return {
            "zone_id": kwargs["zone_id"],
            "start_date": kwargs["start_date"],
            "end_date": kwargs["end_date"],
            "n_days": 366,
            "variables": kwargs["variables"],
            "output": kwargs["output_path"],
            "completeness": {
                "expected_days": 366,
                "received_days": 366,
                "pct_days": 100.0,
                "per_variable": {
                    "temperature_2m": {"received": 366, "pct_complete": 100.0},
                },
            },
            "metadata": {},
        }

    async def fake_percentiles(**kwargs: Any) -> dict[str, Any]:
        return {
            "zone_id": kwargs["zone_id"],
            "output": kwargs["output_path"],
            "variables": kwargs["variables"],
            "statistics": {"n_normals": 12, "n_months": 12, "n_variables": 1},
            "metadata": {"source": "era5_parquet"},
        }

    args = build_parser().parse_args(
        [
            "--zone-id",
            "montpellier",
            "--start-date",
            "2020-01-01",
            "--end-date",
            "2020-12-31",
            "--variables",
            "temperature_2m",
            "--era5-output",
            era5,
            "--normals-output",
            normals,
            "--min-completeness-pct",
            "98",
        ]
    )

    payload = asyncio.run(
        run_pipeline(
            args,
            tools={"era5_ingest": fake_ingest, "climate_percentiles": fake_percentiles},
        )
    )

    assert payload["ok"] is True
    assert payload["gates"]["completeness"] == {
        "passed": True,
        "min_completeness_pct": 98.0,
        "era5": {
            "expected_days": 366,
            "n_days": 366,
            "pct_days": 100.0,
            "min_variable_pct_complete": 100.0,
        },
        "normals": {
            "expected_normals": 12,
            "n_normals": 12,
            "n_months": 12,
            "n_variables": 1,
        },
    }


def test_completeness_gate_passes_for_skipped_era5_with_stored_completeness(tmp_path: Path):
    era5 = (tmp_path / "era5.parquet").as_posix()
    normals = (tmp_path / "normals.parquet").as_posix()

    async def fake_ingest(**kwargs: Any) -> dict[str, Any]:
        return {
            "zone_id": kwargs["zone_id"],
            "start_date": kwargs["start_date"],
            "end_date": kwargs["end_date"],
            "n_days": 366,
            "variables": kwargs["variables"],
            "output": kwargs["output_path"],
            "completeness": {
                "expected_days": 366,
                "received_days": 366,
                "pct_days": 100.0,
                "per_variable": {
                    "temperature_2m": {
                        "received": 366,
                        "expected": 366,
                        "pct_complete": 100.0,
                    },
                },
            },
            "metadata": {
                "skipped": "output exists and covers the request (overwrite=False) — not re-fetched",
                "stored_coverage": {
                    "start": "2020-01-01",
                    "end": "2020-12-31",
                    "n_days": 366,
                },
            },
        }

    async def fake_percentiles(**kwargs: Any) -> dict[str, Any]:
        return {
            "zone_id": kwargs["zone_id"],
            "output": kwargs["output_path"],
            "variables": kwargs["variables"],
            "statistics": {"n_normals": 12, "n_months": 12, "n_variables": 1},
            "metadata": {"source": "era5_parquet"},
        }

    args = build_parser().parse_args(
        [
            "--zone-id",
            "montpellier",
            "--start-date",
            "2020-01-01",
            "--end-date",
            "2020-12-31",
            "--variables",
            "temperature_2m",
            "--era5-output",
            era5,
            "--normals-output",
            normals,
            "--min-completeness-pct",
            "98",
        ]
    )

    payload = asyncio.run(
        run_pipeline(
            args,
            tools={"era5_ingest": fake_ingest, "climate_percentiles": fake_percentiles},
        )
    )

    assert payload["ok"] is True
    assert payload["stages"]["era5_ingest"]["skipped"] is True
    assert payload["gates"]["completeness"]["passed"] is True


def test_dry_run_does_not_reuse_unwritten_era5_output(tmp_path: Path):
    calls: dict[str, dict[str, Any]] = {}
    era5 = (tmp_path / "era5.parquet").as_posix()

    async def fake_ingest(**kwargs: Any) -> dict[str, Any]:
        calls["ingest"] = kwargs
        return {
            "zone_id": kwargs["zone_id"],
            "start_date": kwargs["start_date"],
            "end_date": kwargs["end_date"],
            "n_days": 1,
            "variables": list(NORMALIZED_VARS),
            "output": None,
            "completeness": {"pct_days": 100.0, "per_variable": {}},
            "metadata": {},
        }

    async def fake_percentiles(**kwargs: Any) -> dict[str, Any]:
        calls["percentiles"] = kwargs
        return {
            "zone_id": kwargs["zone_id"],
            "output": None,
            "variables": list(NORMALIZED_VARS),
            "statistics": {"n_normals": 72, "n_months": 12, "n_variables": 6},
            "metadata": {"source": "archive_fetch"},
        }

    args = build_parser().parse_args(
        ["--zone-id", "montpellier", "--dry-run", "--era5-output", era5]
    )

    payload = asyncio.run(
        run_pipeline(
            args,
            tools={"era5_ingest": fake_ingest, "climate_percentiles": fake_percentiles},
        )
    )

    assert payload["ok"] is True
    assert payload["dry_run"] is True
    assert payload["write_output"] is False
    assert calls["ingest"]["write_output"] is False
    assert calls["percentiles"]["write_output"] is False
    assert calls["percentiles"]["era5_source"] is None
    assert "dry-run does not persist ERA5 output" in payload["notes"][0]


def test_dry_run_with_completeness_gate_fails_before_fetch(tmp_path: Path):
    era5 = (tmp_path / "era5.parquet").as_posix()
    normals = (tmp_path / "normals.parquet").as_posix()

    async def should_not_call_ingest(**_kwargs: Any) -> dict[str, Any]:
        raise AssertionError("era5_ingest must not run for dry-run gated preflight failure")

    async def should_not_call_percentiles(**_kwargs: Any) -> dict[str, Any]:
        raise AssertionError("climate_percentiles must not run for dry-run gated preflight failure")

    args = build_parser().parse_args(
        [
            "--zone-id",
            "montpellier",
            "--dry-run",
            "--era5-output",
            era5,
            "--normals-output",
            normals,
            "--min-completeness-pct",
            "98",
        ]
    )

    payload = asyncio.run(
        run_pipeline(
            args,
            tools={
                "era5_ingest": should_not_call_ingest,
                "climate_percentiles": should_not_call_percentiles,
            },
        )
    )

    assert payload["ok"] is False
    assert payload["failed_stage"] == "preflight"
    assert "--min-completeness-pct requires materialized outputs" in payload["error"]
    assert payload["dry_run"] is True
    assert "gates" not in payload
    assert payload["stages"] == {}


def test_preflight_missing_normals_destination_fails_before_ingest(tmp_path: Path, monkeypatch):
    era5 = (tmp_path / "era5.parquet").as_posix()

    async def should_not_call_ingest(**_kwargs: Any) -> dict[str, Any]:
        raise AssertionError("era5_ingest must not run when normals destination is missing")

    async def should_not_call_percentiles(**_kwargs: Any) -> dict[str, Any]:
        raise AssertionError("climate_percentiles must not run after preflight failure")

    monkeypatch.setattr("src.cli.climate_baseline._default_output_uri", lambda *_args: None)
    args = build_parser().parse_args(["--zone-id", "montpellier", "--era5-output", era5])

    payload = asyncio.run(
        run_pipeline(
            args,
            tools={
                "era5_ingest": should_not_call_ingest,
                "climate_percentiles": should_not_call_percentiles,
            },
        )
    )

    assert payload["ok"] is False
    assert payload["failed_stage"] == "preflight"
    assert "--normals-output" in payload["error"]
    assert payload["resolved_outputs"] == {"era5": era5, "normals": None}
    assert payload["stages"] == {}


def test_run_writes_failure_summary(tmp_path: Path, monkeypatch, capsys):
    out = tmp_path / "summary.json"

    async def failing_ingest(**_kwargs: Any) -> dict[str, Any]:
        return {"error": "No output destination - set S3_BUCKET or pass output_path"}

    async def should_not_call_percentiles(**_kwargs: Any) -> dict[str, Any]:
        raise AssertionError("climate_percentiles must not run after ingest failure")

    monkeypatch.setattr(
        "src.cli.climate_baseline._load_climate_tools",
        lambda: {
            "era5_ingest": failing_ingest,
            "climate_percentiles": should_not_call_percentiles,
        },
    )
    monkeypatch.setattr("src.cli.climate_baseline._default_output_uri", lambda *_args: None)
    args = build_parser().parse_args(["--zone-id", "montpellier", "-o", str(out)])

    assert run(args) == 1
    assert "Wrote climate baseline summary" in capsys.readouterr().out
    payload = json.loads(out.read_text())
    assert payload["ok"] is False
    assert payload["failed_stage"] == "preflight"
    assert "Missing output destination" in payload["error"]
    assert payload["window"]["label"] == "WMO_1991_2020"
