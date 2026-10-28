"""CLI for FIERCE pilot ERA5 backfill + climatological baseline materialization.

This is a thin batch wrapper over the shipped MCP handlers:

1. ``era5_ingest`` materializes daily ERA5 reanalysis for a zone.
2. ``climate_percentiles`` materializes WMO/default climatological normals
   unless ``--skip-normals`` is used for ERA5-only holdout materialization.

It gives operators a reproducible command for the "30-year baseline" quickwin
without duplicating the climate tool logic or requiring an MCP client.

Examples:
    uv run --extra patterns python -m src.cli.climate_baseline \\
        --zone-id montpellier --overwrite

    uv run --extra patterns python -m src.cli.climate_baseline \\
        --zone-id nairobi --start-date 1995-01-01 --end-date 2025-12-31 \\
        --era5-output /tmp/nairobi-era5.parquet \\
        --normals-output /tmp/nairobi-normals.parquet -o /tmp/nairobi-baseline.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import sys
import time
from collections.abc import Awaitable, Callable, Iterable
from contextlib import contextmanager
from datetime import date
from pathlib import Path
from typing import Any

from src.tools.climate import (
    _default_output_uri,
    _resolve_normals_window,
    _validate_dst,
    register_climate_tools,
)
from src.utils import openmeteo as openmeteo_utils
from src.utils.logging import setup_logging
from src.utils.openmeteo import NORMALIZED_VARS

ToolHandler = Callable[..., Awaitable[dict[str, Any]]]


class _ToolCapture:
    """Small FastMCP-compatible capture used to call registered handlers directly."""

    def __init__(self) -> None:
        self.tools: dict[str, ToolHandler] = {}

    def tool(self, *_args: Any, **_kwargs: Any) -> Callable[[ToolHandler], ToolHandler]:
        def _decorator(fn: ToolHandler) -> ToolHandler:
            self.tools[fn.__name__] = fn
            return fn

        return _decorator


def _load_climate_tools() -> dict[str, ToolHandler]:
    capture = _ToolCapture()
    register_climate_tools(capture)
    return capture.tools


def _percent_threshold(value: str) -> float:
    try:
        pct = float(value)
    except ValueError:
        raise argparse.ArgumentTypeError("must be a number between 0 and 100") from None
    if not math.isfinite(pct) or pct < 0.0 or pct > 100.0:
        raise argparse.ArgumentTypeError("must be a number between 0 and 100")
    return pct


def _positive_int(value: str) -> int:
    try:
        number = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError("must be a positive integer") from None
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


@contextmanager
def _archive_chunk_days(max_days: int | None):
    if max_days is None:
        yield
        return
    previous = openmeteo_utils._MAX_DAYS_PER_REQUEST
    openmeteo_utils._MAX_DAYS_PER_REQUEST = max_days
    try:
        yield
    finally:
        openmeteo_utils._MAX_DAYS_PER_REQUEST = previous


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="climate_baseline",
        description=(
            "Run the FIERCE ERA5 baseline pilot: era5_ingest followed by "
            "climate_percentiles, with a compact JSON run summary."
        ),
    )
    p.add_argument("--zone-id", required=True, help="zone id from zones/zones.geojson")
    p.add_argument(
        "--start-date",
        help="ISO date, inclusive; defaults to WMO normal start 1991-01-01",
    )
    p.add_argument(
        "--end-date",
        help="ISO date, inclusive; defaults to WMO normal end 2020-12-31",
    )
    p.add_argument(
        "--variables",
        nargs="+",
        choices=list(NORMALIZED_VARS),
        help="normalized ERA5 variables; default is all six terra-cognita section 5.1 variables",
    )
    p.add_argument(
        "--era5-output",
        help="explicit ERA5 parquet destination; otherwise uses s3://$S3_BUCKET/climate/era5/{zone}.parquet",
    )
    p.add_argument(
        "--normals-output",
        help=(
            "explicit normals parquet destination; otherwise uses "
            "s3://$S3_BUCKET/climate/normals/{zone}.parquet"
        ),
    )
    p.add_argument(
        "--skip-normals",
        action="store_true",
        help="materialize ERA5 only; do not run climate_percentiles or require a normals output",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="replace existing ERA5/normals outputs instead of using coverage-aware skips",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="fetch and summarize without writing parquets",
    )
    p.add_argument(
        "--min-completeness-pct",
        type=_percent_threshold,
        help=(
            "operator gate: fail the CLI after materialization unless ERA5 day/variable "
            "completeness is at least this percentage and normals cover 12 months × variables"
        ),
    )
    p.add_argument(
        "--archive-chunk-days",
        type=_positive_int,
        help=(
            "operator override for Open-Meteo Archive request chunking; default is the "
            "guard-rail in src.utils.openmeteo (366). For long full-panel runs, use "
            "10958 to request one WMO 1991-2020 zone in a single Archive call."
        ),
    )
    p.add_argument("-o", "--output", help="write compact JSON run summary here (default: stdout)")
    return p


def _finite_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _min_pct_complete(per_variable: dict[str, Any]) -> float | None:
    values: list[float] = []
    for item in per_variable.values():
        if not isinstance(item, dict):
            continue
        pct = _finite_float(item.get("pct_complete"))
        if pct is not None:
            values.append(pct)
    return min(values) if values else None


def _expected_days(start_date: str, end_date: str) -> int:
    return (date.fromisoformat(end_date) - date.fromisoformat(start_date)).days + 1


def _compact_stage_common(result: dict[str, Any]) -> dict[str, Any]:
    metadata = result.get("metadata") if isinstance(result.get("metadata"), dict) else {}
    payload: dict[str, Any] = {
        "output": result.get("output"),
        "skipped": bool(metadata.get("skipped")),
    }
    elapsed_ms = _finite_float(metadata.get("elapsed_ms"))
    if elapsed_ms is not None:
        payload["elapsed_ms"] = elapsed_ms
    if "error" in result:
        payload["error"] = result["error"]
    return payload


def _compact_era5_summary(result: dict[str, Any]) -> dict[str, Any]:
    payload = _compact_stage_common(result)
    completeness = (
        result.get("completeness") if isinstance(result.get("completeness"), dict) else {}
    )
    per_variable = (
        completeness.get("per_variable")
        if isinstance(completeness.get("per_variable"), dict)
        else {}
    )
    payload.update(
        {
            "start_date": result.get("start_date"),
            "end_date": result.get("end_date"),
            "n_days": result.get("n_days"),
            "pct_days": _finite_float(completeness.get("pct_days")),
            "min_variable_pct_complete": _min_pct_complete(per_variable),
        }
    )
    return payload


def _compact_normals_summary(result: dict[str, Any]) -> dict[str, Any]:
    payload = _compact_stage_common(result)
    metadata = result.get("metadata") if isinstance(result.get("metadata"), dict) else {}
    statistics = result.get("statistics") if isinstance(result.get("statistics"), dict) else {}
    payload.update(
        {
            "source": metadata.get("source"),
            "normal_grain": metadata.get("normal_grain"),
            "n_normals": statistics.get("n_normals"),
            "n_months": statistics.get("n_months"),
            "n_variables": statistics.get("n_variables"),
        }
    )
    return payload


def _base_summary(
    *,
    args: argparse.Namespace,
    resolved_start: str,
    resolved_end: str,
    window_label: str,
    variables: list[str],
    write_output: bool,
    started_at: float,
) -> dict[str, Any]:
    resolved_outputs = _resolve_output_destinations(args)
    return {
        "ok": False,
        "zone_id": args.zone_id,
        "window": {
            "start_date": resolved_start,
            "end_date": resolved_end,
            "label": window_label,
        },
        "variables": variables,
        "write_output": write_output,
        "dry_run": args.dry_run,
        "skip_normals": bool(getattr(args, "skip_normals", False)),
        "archive_chunk_days": args.archive_chunk_days,
        "storage_mode": getattr(args, "storage_mode", None)
        or _storage_mode_for_outputs(resolved_outputs),
        "requested_outputs": {
            "era5": args.era5_output,
            "normals": args.normals_output,
        },
        "resolved_outputs": resolved_outputs,
        "duration_ms": round((time.monotonic() - started_at) * 1000),
        "stages": {},
        "notes": [],
    }


def _failure_summary(
    *,
    args: argparse.Namespace,
    resolved_start: str,
    resolved_end: str,
    window_label: str,
    variables: list[str],
    write_output: bool,
    started_at: float,
    stage: str,
    result: dict[str, Any],
) -> dict[str, Any]:
    summary = _base_summary(
        args=args,
        resolved_start=resolved_start,
        resolved_end=resolved_end,
        window_label=window_label,
        variables=variables,
        write_output=write_output,
        started_at=started_at,
    )
    summary["failed_stage"] = stage
    summary["error"] = result.get("error", "unknown error")
    summary["stages"] = {
        stage: _compact_era5_summary(result)
        if stage == "era5_ingest"
        else _compact_normals_summary(result)
    }
    return summary


def _resolve_output_destinations(args: argparse.Namespace) -> dict[str, str | None]:
    era5 = args.era5_output or (
        None if args.skip_normals else _default_output_uri(args.zone_id, "era5")
    )
    normals = (
        None
        if args.skip_normals
        else args.normals_output or _default_output_uri(args.zone_id, "normals")
    )
    return {
        "era5": era5,
        "normals": normals,
    }


def _storage_mode_for_outputs(outputs: dict[str, str | None]) -> str:
    values = [value for value in outputs.values() if value]
    if values and all(value.startswith("s3://") for value in values):
        return "s3"
    if values and any(value.startswith("s3://") for value in values):
        return "mixed"
    return "local"


def _preflight_output_destinations(args: argparse.Namespace, *, write_output: bool) -> str | None:
    if args.dry_run and args.min_completeness_pct is not None:
        return (
            "--min-completeness-pct requires materialized outputs; remove --dry-run "
            "to run the gate, or omit --min-completeness-pct for a preview"
        )

    if not write_output:
        return None

    outputs = _resolve_output_destinations(args)
    required_outputs = [("era5", "--era5-output")]
    if not args.skip_normals:
        required_outputs.append(("normals", "--normals-output"))
    missing = [flag for key, flag in required_outputs if not outputs[key]]
    if missing:
        if args.skip_normals:
            return (
                "Missing output destination(s) before Archive fetch: "
                f"{', '.join(missing)}. Pass --era5-output explicitly; "
                "--skip-normals never uses the canonical climate/era5 default."
            )
        return (
            "Missing output destination(s) before Archive fetch: "
            f"{', '.join(missing)}. Pass both --era5-output and --normals-output, "
            "or set S3_BUCKET so both default climate outputs can be derived. "
            "Use --dry-run to skip write preflight."
        )

    for key, dst in outputs.items():
        if dst is None:
            continue
        try:
            _validate_dst(dst)
        except ValueError as exc:
            return f"Invalid {key} output destination before Archive fetch: {exc}"
    return None


def _preflight_failure_summary(
    *,
    args: argparse.Namespace,
    resolved_start: str,
    resolved_end: str,
    window_label: str,
    variables: list[str],
    write_output: bool,
    started_at: float,
    error: str,
) -> dict[str, Any]:
    summary = _base_summary(
        args=args,
        resolved_start=resolved_start,
        resolved_end=resolved_end,
        window_label=window_label,
        variables=variables,
        write_output=write_output,
        started_at=started_at,
    )
    summary["failed_stage"] = "preflight"
    summary["error"] = error
    return summary


def _add_dry_run_note(summary: dict[str, Any]) -> None:
    notes = summary.setdefault("notes", [])
    notes.append(
        "dry-run does not persist ERA5 output; climate_percentiles may fetch Archive independently"
    )


def _apply_completeness_gate(summary: dict[str, Any], threshold: float | None) -> None:
    """Apply an operator gate to a successful run summary, mutating it in place."""
    if threshold is None or not summary.get("ok"):
        return

    window = summary.get("window") if isinstance(summary.get("window"), dict) else {}
    variables = summary.get("variables") if isinstance(summary.get("variables"), list) else []
    stages = summary.get("stages") if isinstance(summary.get("stages"), dict) else {}
    era5 = stages.get("era5_ingest") if isinstance(stages.get("era5_ingest"), dict) else {}
    normals = (
        stages.get("climate_percentiles")
        if isinstance(stages.get("climate_percentiles"), dict)
        else {}
    )
    skip_normals = bool(summary.get("skip_normals"))
    expected_normals = 12 * len(variables)
    expected_days = _expected_days(str(window["start_date"]), str(window["end_date"]))

    gate: dict[str, Any] = {
        "passed": True,
        "min_completeness_pct": threshold,
        "era5": {
            "expected_days": expected_days,
            "n_days": era5.get("n_days"),
            "pct_days": era5.get("pct_days"),
            "min_variable_pct_complete": era5.get("min_variable_pct_complete"),
        },
        "normals": {
            "expected_normals": expected_normals,
            "n_normals": normals.get("n_normals"),
            "n_months": normals.get("n_months"),
            "n_variables": normals.get("n_variables"),
        },
    }
    failures: list[str] = []

    pct_days = _finite_float(era5.get("pct_days"))
    min_variable_pct = _finite_float(era5.get("min_variable_pct_complete"))
    n_days = _finite_float(era5.get("n_days"))
    n_normals = _finite_float(normals.get("n_normals"))
    n_months = _finite_float(normals.get("n_months"))
    n_variables = _finite_float(normals.get("n_variables"))

    if pct_days is None:
        failures.append("era5_ingest.pct_days unavailable; re-run with --overwrite to recompute")
    elif pct_days < threshold:
        failures.append(f"era5_ingest.pct_days {pct_days} < {threshold}")

    if min_variable_pct is None:
        failures.append(
            "era5_ingest.min_variable_pct_complete unavailable; "
            "re-run with --overwrite to recompute"
        )
    elif min_variable_pct < threshold:
        failures.append(f"era5_ingest.min_variable_pct_complete {min_variable_pct} < {threshold}")

    if n_days is None or int(n_days) < expected_days:
        failures.append(f"era5_ingest.n_days {era5.get('n_days')} < expected {expected_days}")

    if not skip_normals:
        if n_months is None or int(n_months) != 12:
            failures.append(f"climate_percentiles.n_months {normals.get('n_months')} != 12")
        if n_variables is None or int(n_variables) != len(variables):
            failures.append(
                f"climate_percentiles.n_variables {normals.get('n_variables')} != {len(variables)}"
            )
        if n_normals is None or int(n_normals) != expected_normals:
            failures.append(
                f"climate_percentiles.n_normals {normals.get('n_normals')} != {expected_normals}"
            )

    if failures:
        gate["passed"] = False
        summary["ok"] = False
        summary["failed_stage"] = "completeness_gate"
        summary["error"] = "; ".join(failures)
    summary.setdefault("gates", {})["completeness"] = gate


async def run_pipeline(
    args: argparse.Namespace,
    *,
    tools: dict[str, ToolHandler] | None = None,
) -> dict[str, Any]:
    started_at = time.monotonic()
    resolved_start, resolved_end, window_label = _resolve_normals_window(
        args.start_date, args.end_date
    )
    variables = list(dict.fromkeys(args.variables)) if args.variables else list(NORMALIZED_VARS)
    requested_variables = variables if args.variables else None
    write_output = not args.dry_run

    preflight_error = _preflight_output_destinations(args, write_output=write_output)
    if preflight_error:
        return _preflight_failure_summary(
            args=args,
            resolved_start=resolved_start,
            resolved_end=resolved_end,
            window_label=window_label,
            variables=variables,
            write_output=write_output,
            started_at=started_at,
            error=preflight_error,
        )

    loaded_tools = tools or _load_climate_tools()

    with _archive_chunk_days(args.archive_chunk_days):
        ingest = await loaded_tools["era5_ingest"](
            zone_id=args.zone_id,
            start_date=resolved_start,
            end_date=resolved_end,
            variables=requested_variables,
            overwrite=args.overwrite,
            write_output=write_output,
            output_path=args.era5_output,
        )
        if "error" not in ingest:
            if args.skip_normals:
                normals = None
            else:
                era5_source = ingest.get("output") if write_output else None
                normals = await loaded_tools["climate_percentiles"](
                    zone_id=args.zone_id,
                    start_date=resolved_start,
                    end_date=resolved_end,
                    variables=requested_variables,
                    era5_source=era5_source,
                    overwrite=args.overwrite,
                    write_output=write_output,
                    output_path=args.normals_output,
                )
    if "error" in ingest:
        summary = _failure_summary(
            args=args,
            resolved_start=resolved_start,
            resolved_end=resolved_end,
            window_label=window_label,
            variables=variables,
            write_output=write_output,
            started_at=started_at,
            stage="era5_ingest",
            result=ingest,
        )
        if args.dry_run:
            _add_dry_run_note(summary)
        return summary

    if normals is not None and "error" in normals:
        summary = _failure_summary(
            args=args,
            resolved_start=resolved_start,
            resolved_end=resolved_end,
            window_label=window_label,
            variables=variables,
            write_output=write_output,
            started_at=started_at,
            stage="climate_percentiles",
            result=normals,
        )
        summary["stages"]["era5_ingest"] = _compact_era5_summary(ingest)
        if args.dry_run:
            _add_dry_run_note(summary)
        return summary

    summary = _base_summary(
        args=args,
        resolved_start=resolved_start,
        resolved_end=resolved_end,
        window_label=window_label,
        variables=variables,
        write_output=write_output,
        started_at=started_at,
    )
    summary["ok"] = True
    summary["stages"] = {"era5_ingest": _compact_era5_summary(ingest)}
    if normals is None:
        summary["notes"].append("skip-normals: climate_percentiles was not run")
    else:
        summary["stages"]["climate_percentiles"] = _compact_normals_summary(normals)
    summary["duration_ms"] = round((time.monotonic() - started_at) * 1000)
    if args.dry_run:
        _add_dry_run_note(summary)
    _apply_completeness_gate(summary, args.min_completeness_pct)
    return summary


def _write_summary(payload: dict[str, Any], output: str | None) -> None:
    text = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if output:
        Path(output).write_text(text, encoding="utf-8")
        print(f"Wrote climate baseline summary to {output}")
    else:
        print(text, end="")


def run(args: argparse.Namespace) -> int:
    try:
        payload = asyncio.run(run_pipeline(args))
        _write_summary(payload, args.output)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"ERROR: climate baseline failed: {exc}", file=sys.stderr)
        return 1
    return 0 if payload.get("ok") else 1


def main(argv: Iterable[str] | None = None) -> int:
    # Install the credential-redaction processor (#148) before anything can log: structlog's
    # DEFAULT chain has no redactor (#202).
    setup_logging()
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
