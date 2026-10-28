#!/usr/bin/env python3
"""Run the first FIERCE O3 cross-zone validation proof over Q3 holdout B4 data.

This is an operational wrapper around the existing ``discover_patterns_cv`` MCP
handler. It does not implement new mining logic; it preflights the segmented B4
artifacts, excludes zones that cannot be mined, runs GRAANK/TGrad validation,
and writes local JSON evidence plus optional RustFS/S3 pattern-library Parquets.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# Allow `python scripts/o3_q3_validation.py` from anywhere.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts import climate_baseline_panel  # noqa: E402
from src.config import settings  # noqa: E402
from src.patterns.parquet_bridge import load_dataframe  # noqa: E402
from src.tools.patterns import register_pattern_tools  # noqa: E402
from src.utils.logging import setup_logging  # noqa: E402

DEFAULT_WINDOW_LABELS = ("2023_07", "2023_08", "2023_09")
DEFAULT_SOURCE_TEMPLATE = "climate/holdout/unified/{window_label}/{zone_id}.parquet"
DEFAULT_OUTPUT_PREFIX = "climate/o3/holdout/q3_2023"
DEFAULT_COLUMNS = (
    "ndvi_mean",
    "temperature_2m",
    "precipitation",
    "humidity",
    "evapotranspiration",
    "wind_speed",
    "solar_radiation",
)
DEFAULT_TRUNCATED_SOURCES = ("svalbard:2023_07",)
ALGORITHMS = ("graank", "tgrad")

FrameLoader = Callable[[str, str, list[str] | None, str | None], Any]
ToolRunner = Callable[..., Awaitable[dict[str, Any]]]


class _CaptureMCP:
    def __init__(self) -> None:
        self.tools: dict[str, Any] = {}

    def tool(self, *_args, **_kwargs):
        def deco(fn):
            self.tools[fn.__name__] = fn
            return fn

        return deco


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="o3_q3_validation",
        description=(
            "Preflight segmented Q3 B4 unified tables, then run discover_patterns_cv "
            "for GRAANK and/or TGrad over eligible zones."
        ),
    )
    zone_selection = p.add_mutually_exclusive_group()
    zone_selection.add_argument("--zone-id", nargs="+", help="one or more zone ids")
    zone_selection.add_argument(
        "--zone-set",
        choices=("full-panel",),
        default="full-panel",
        help="predefined zone set; default loads the 15-zone B1 artifact",
    )
    p.add_argument("--window-label", nargs="+", default=list(DEFAULT_WINDOW_LABELS))
    p.add_argument("--source-template", default=DEFAULT_SOURCE_TEMPLATE)
    p.add_argument("--algorithm", nargs="+", choices=ALGORITHMS, default=list(ALGORITHMS))
    p.add_argument("--columns", nargs="+", default=list(DEFAULT_COLUMNS))
    p.add_argument(
        "--source-kind", choices=("parquet", "geoparquet", "csv", "json"), default="parquet"
    )
    p.add_argument("--order-by", default="date")
    p.add_argument("--target-column", default="ndvi_mean", help="TGrad target column")
    p.add_argument("--min-support", type=float, default=0.5)
    p.add_argument("--min-rep", type=float, default=0.5)
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--min-validated-zones", type=int, default=1)
    p.add_argument("--min-rows-per-zone", type=int, default=2)
    p.add_argument("--period-start", default="2023-07-01")
    p.add_argument("--period-end", default="2023-09-30")
    p.add_argument(
        "--truncated-source",
        action="append",
        default=list(DEFAULT_TRUNCATED_SOURCES),
        metavar="ZONE:LABEL",
        help="source-quality caveat to echo into O3 metadata; repeatable",
    )
    p.add_argument("--output-prefix", default=DEFAULT_OUTPUT_PREFIX)
    p.add_argument("--output-dir", default="/tmp/geoai-o3-q3-validation")
    p.add_argument("--summary-name", default="o3-q3-validation-summary.json")
    p.add_argument("--no-write-output", action="store_true")
    p.add_argument(
        "--top-n", type=int, default=20, help="number of top patterns to keep in summary"
    )
    return p


def _resolve_zone_ids(args: argparse.Namespace) -> list[str]:
    if args.zone_id:
        return list(dict.fromkeys(args.zone_id))
    return climate_baseline_panel._load_full_panel_zone_ids()


def _source_uri(template: str, *, zone_id: str, window_label: str) -> str:
    value = template.format(zone_id=zone_id, window_label=window_label)
    if "://" in value or value.startswith(("/", ".")):
        return value
    if settings.S3_BUCKET:
        return f"s3://{settings.S3_BUCKET}/{value}"
    return value


def _parse_truncated_sources(values: list[str]) -> dict[str, list[str]]:
    parsed: dict[str, list[str]] = {}
    for value in values:
        if ":" not in value:
            raise ValueError(f"--truncated-source must be ZONE:LABEL, got {value!r}")
        zone_id, label = value.split(":", 1)
        if not zone_id or not label:
            raise ValueError(f"--truncated-source must be ZONE:LABEL, got {value!r}")
        parsed.setdefault(zone_id, []).append(label)
    return parsed


def _default_frame_loader(
    uri: str,
    source_kind: str,
    columns: list[str] | None,
    order_by: str | None,
):
    return load_dataframe(uri, kind=source_kind, columns=columns, order_by=order_by)


def _preflight_sources(
    *,
    zone_ids: list[str],
    window_labels: list[str],
    source_template: str,
    source_kind: str,
    columns: list[str] | None,
    order_by: str | None,
    min_rows_per_zone: int,
    frame_loader: FrameLoader,
) -> tuple[list[str], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    eligible: list[str] = []
    for zone_id in zone_ids:
        total_rows = 0
        per_source: list[dict[str, Any]] = []
        status = "eligible"
        error = None
        for label in window_labels:
            uri = _source_uri(source_template, zone_id=zone_id, window_label=label)
            try:
                frame = frame_loader(uri, source_kind, columns, order_by)
                n_rows = len(frame)
            except Exception as exc:  # noqa: BLE001 - operator summary should show the bad source
                n_rows = 0
                status = "error"
                error = str(exc)
            total_rows += n_rows
            per_source.append({"window_label": label, "source": uri, "rows": n_rows})
        if status != "error" and total_rows < min_rows_per_zone:
            status = "excluded_insufficient_rows"
        if status == "eligible":
            eligible.append(zone_id)
        row = {
            "zone_id": zone_id,
            "status": status,
            "total_rows": total_rows,
            "sources": per_source,
        }
        if error:
            row["error"] = error
        rows.append(row)
    return eligible, rows


def _discover_patterns_cv_tool() -> ToolRunner:
    mcp = _CaptureMCP()
    register_pattern_tools(mcp)
    return mcp.tools["discover_patterns_cv"]


async def _run_algorithm(
    *,
    tool_runner: ToolRunner,
    algorithm: str,
    args: argparse.Namespace,
    eligible_zone_ids: list[str],
    truncated_sources: dict[str, list[str]],
) -> dict[str, Any]:
    write_output = not args.no_write_output
    output_key = f"{args.output_prefix}/{algorithm}_validated.parquet" if write_output else None
    kwargs: dict[str, Any] = {
        "zone_ids": eligible_zone_ids,
        "source_template": args.source_template,
        "window_labels": args.window_label,
        "algorithm": algorithm,
        "min_support": args.min_support,
        "n_folds": args.n_folds,
        "min_validated_zones": args.min_validated_zones,
        "columns": args.columns,
        "order_by": args.order_by,
        "source_kind": args.source_kind,
        "period_start": args.period_start,
        "period_end": args.period_end,
        "truncated_sources": truncated_sources,
        "write_output": write_output,
        "output_key": output_key,
    }
    if algorithm == "tgrad":
        kwargs.update({"target_column": args.target_column, "min_rep": args.min_rep})
    started = time.monotonic()
    result = await tool_runner(**kwargs)
    result.setdefault("statistics", {})
    if isinstance(result["statistics"], dict):
        result["statistics"]["wrapper_runtime_ms"] = round((time.monotonic() - started) * 1000)
    return result


async def run_async(
    args: argparse.Namespace,
    *,
    tool_runner: ToolRunner | None = None,
    frame_loader: FrameLoader = _default_frame_loader,
) -> dict[str, Any]:
    zone_ids = _resolve_zone_ids(args)
    truncated_sources = _parse_truncated_sources(args.truncated_source or [])
    eligible_zone_ids, preflight = _preflight_sources(
        zone_ids=zone_ids,
        window_labels=args.window_label,
        source_template=args.source_template,
        source_kind=args.source_kind,
        columns=args.columns,
        order_by=args.order_by,
        min_rows_per_zone=args.min_rows_per_zone,
        frame_loader=frame_loader,
    )
    for row in preflight:
        if row["status"] == "excluded_insufficient_rows":
            truncated_sources.setdefault(row["zone_id"], []).append("insufficient_q3_rows")

    if len(eligible_zone_ids) < args.n_folds:
        raise ValueError(
            f"need at least n_folds={args.n_folds} eligible zones; got {len(eligible_zone_ids)}"
        )
    if any(row["status"] == "error" for row in preflight):
        failed = [row["zone_id"] for row in preflight if row["status"] == "error"]
        raise ValueError(f"source preflight failed for zones: {failed}")

    runner = tool_runner or _discover_patterns_cv_tool()
    runs: dict[str, Any] = {}
    for algorithm in args.algorithm:
        result = await _run_algorithm(
            tool_runner=runner,
            algorithm=algorithm,
            args=args,
            eligible_zone_ids=eligible_zone_ids,
            truncated_sources=truncated_sources,
        )
        if result.get("error"):
            raise RuntimeError(f"{algorithm} O3 validation failed: {result['error']}")
        runs[algorithm] = result

    return {
        "kind": "o3_q3_validation_summary",
        "created_at": _utc_now(),
        "inputs": {
            "source_template": args.source_template,
            "window_labels": args.window_label,
            "columns": args.columns,
            "period_start": args.period_start,
            "period_end": args.period_end,
            "min_support": args.min_support,
            "n_folds": args.n_folds,
            "min_validated_zones": args.min_validated_zones,
            "min_rows_per_zone": args.min_rows_per_zone,
        },
        "zone_ids": zone_ids,
        "eligible_zone_ids": eligible_zone_ids,
        "excluded_zone_ids": [row["zone_id"] for row in preflight if row["status"] != "eligible"],
        "truncated_sources": truncated_sources,
        "preflight": preflight,
        "runs": runs,
    }


def _write_outputs(summary: dict[str, Any], args: argparse.Namespace) -> Path:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    top_n = max(0, args.top_n)
    for algorithm, result in summary["runs"].items():
        full_path = out_dir / f"{algorithm}-result.json"
        full_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        result["top_patterns"] = result.get("patterns", [])[:top_n]
        result["n_patterns"] = len(result.get("patterns", []))
        result.pop("patterns", None)
    summary_path = out_dir / args.summary_name
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary_path


def run(args: argparse.Namespace) -> int:
    try:
        summary = asyncio.run(run_async(args))
        summary_path = _write_outputs(summary, args)
    except Exception as exc:  # noqa: BLE001 - CLI surface stays concise
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(f"OK=True summary={summary_path}")
    for algorithm, result in summary["runs"].items():
        stats = result.get("statistics") or {}
        print(
            f"{algorithm}: patterns={result.get('n_patterns', 0)} "
            f"candidates={stats.get('n_candidate_patterns')} "
            f"validated={stats.get('n_validated_patterns')} "
            f"output={result.get('output')}"
        )
    if summary["excluded_zone_ids"]:
        print(f"excluded_zones={','.join(summary['excluded_zone_ids'])}")
    return 0


def main(argv: list[str] | None = None) -> int:
    # Install the credential-redaction processor (#148). Only src/server.py called this, so
    # every `log.error(..., error=str(e))` these handlers reach went to stderr raw when a
    # script drove them in-process -- structlog's DEFAULT chain has no redactor (#161 review).
    setup_logging()
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
