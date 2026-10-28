#!/usr/bin/env python3
"""Run the recent-holdout ERA5 -> B4 unified pilot chain.

This is an operator wrapper around the two existing panel runners:

1. ``climate_baseline_panel.py --artifact-scope holdout`` materializes
   ERA5-only recent artifacts under ``climate/holdout/era5/{label}/``.
2. ``unified_pilot_panel.py --artifact-scope holdout`` joins a bounded NDVI
   window against those ERA5 artifacts and writes
   ``climate/holdout/unified/{label}/``.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import re
import sys
import time
from collections.abc import Awaitable, Callable, Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# Allow `python scripts/holdout_pilot_panel.py` from anywhere.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts import climate_baseline_panel, unified_pilot_panel  # noqa: E402

PanelRunner = Callable[[argparse.Namespace], Awaitable[dict[str, Any]]]
UnifiedRunner = Callable[..., Awaitable[dict[str, Any]]]
Sleeper = Callable[[float], Awaitable[None]]
RunManifestWriter = Callable[[str, dict[str, Any]], None]

PILOT_ZONE_IDS = unified_pilot_panel.PILOT_ZONE_IDS
ZONE_SET_IDS = {"pilot-3": PILOT_ZONE_IDS}
ZONE_SETS = tuple(ZONE_SET_IDS)
DEFAULT_HOLDOUT_START_DATE = "2021-01-01"
DEFAULT_HOLDOUT_END_DATE = "2025-12-31"
DEFAULT_UNIFIED_START_DATE = "2023-07-01"
DEFAULT_UNIFIED_END_DATE = "2023-07-31"
_SAFE_RUN_ID_RE = re.compile(r"^[A-Za-z0-9._-]+$")


def _nonnegative_float(value: str) -> float:
    try:
        number = float(value)
    except ValueError:
        raise argparse.ArgumentTypeError("must be a non-negative number") from None
    if not math.isfinite(number) or number < 0:
        raise argparse.ArgumentTypeError("must be a finite non-negative number")
    return number


def _positive_int(value: str) -> int:
    try:
        number = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError("must be a positive integer") from None
    if number <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def _percent_threshold(value: str) -> float:
    try:
        number = float(value)
    except ValueError:
        raise argparse.ArgumentTypeError("must be a percentage in [0, 100]") from None
    if not math.isfinite(number) or not 0 <= number <= 100:
        raise argparse.ArgumentTypeError("must be a percentage in [0, 100]")
    return number


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="holdout_pilot_panel",
        description=(
            "Materialize recent holdout ERA5 artifacts and, unless disabled, "
            "run the B4 unified pilot join against those holdout artifacts."
        ),
    )
    zone_selection = p.add_mutually_exclusive_group()
    zone_selection.add_argument("--zone-id", nargs="+", help="one or more zone ids")
    zone_selection.add_argument(
        "--zone-set",
        choices=ZONE_SETS,
        help="predefined zone set; default is pilot-3 when --zone-id is omitted",
    )
    p.add_argument("--holdout-start-date", default=DEFAULT_HOLDOUT_START_DATE)
    p.add_argument("--holdout-end-date", default=DEFAULT_HOLDOUT_END_DATE)
    p.add_argument("--unified-start-date", default=DEFAULT_UNIFIED_START_DATE)
    p.add_argument("--unified-end-date", default=DEFAULT_UNIFIED_END_DATE)
    p.add_argument(
        "--artifact-window-label",
        help=(
            "safe ERA5 holdout source label; default is YYYY_YYYY for whole-year "
            "holdout windows, else holdout start_end with dashes replaced by underscores"
        ),
    )
    p.add_argument(
        "--unified-window-label",
        help=(
            "safe B4 unified output label; default is YYYY_YYYY for whole-year "
            "unified windows, else unified start_end with dashes replaced by underscores"
        ),
    )
    p.add_argument(
        "--era5-source-template",
        help=(
            "explicit ERA5 source template for the B4 stage; may include {zone_id}. "
            "Defaults to the resolved holdout ERA5 path for the selected storage mode"
        ),
    )
    p.add_argument(
        "--storage-mode",
        choices=("local", "s3"),
        default="s3",
        help="artifact destination for both stages; default s3/RustFS",
    )
    p.add_argument(
        "--variables",
        nargs="+",
        choices=list(climate_baseline_panel.NORMALIZED_VARS),
        help="normalized ERA5 variables; default is all six locked variables",
    )
    p.add_argument(
        "--output-dir",
        default="/tmp/geoai-holdout-pilot",
        help="directory for wrapper and per-stage JSON summaries",
    )
    p.add_argument("--summary-name", default="holdout-pilot-summary.json")
    p.add_argument("--manifest-name", default="holdout-pilot-manifest.json")
    p.add_argument("--run-id", help="stable root id; stage manifests append -era5/-unified")
    p.add_argument(
        "--archive-chunk-days",
        type=_positive_int,
        help=(
            "passed to the ERA5 stage; default is the full holdout window length "
            "so each zone can be requested in one Archive call"
        ),
    )
    p.add_argument(
        "--min-completeness-pct",
        type=_percent_threshold,
        default=98.0,
        help="ERA5 completeness gate; default 98",
    )
    p.add_argument(
        "--tolerance-days",
        type=lambda value: unified_pilot_panel._bounded_int(
            value, minimum=0, maximum=31, label="tolerance_days"
        ),
        default=3,
        help="nearest-date NDVI-to-ERA5 join window in days; default 3",
    )
    p.add_argument(
        "--cloud-cover-max",
        type=lambda value: unified_pilot_panel._bounded_int(
            value, minimum=0, maximum=100, label="cloud_cover_max"
        ),
        default=unified_pilot_panel.settings.DEFAULT_CLOUD_COVER_MAX,
        help=f"max cloud cover percentage for NDVI; default {unified_pilot_panel.settings.DEFAULT_CLOUD_COVER_MAX}",
    )
    p.add_argument("--preflight-only", action="store_true")
    p.add_argument("--era5-only", action="store_true", help="materialize/check ERA5 but skip B4")
    p.add_argument(
        "--skip-era5",
        action="store_true",
        help="assume holdout ERA5 already exists and run only the B4 stage",
    )
    p.add_argument("--fail-fast", action="store_true")
    p.add_argument(
        "--zone-delay-s",
        type=_nonnegative_float,
        default=0.0,
        help="seconds to wait between zones in each stage",
    )
    return p


def _flatten_zone_ids(values: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(values))


def _resolve_zone_ids(args: argparse.Namespace) -> list[str]:
    if args.zone_id:
        return _flatten_zone_ids(args.zone_id)
    zone_set = args.zone_set or "pilot-3"
    try:
        return list(ZONE_SET_IDS[zone_set])
    except KeyError:
        raise ValueError(f"unsupported zone set: {zone_set!r}") from None


def _safe_run_id_error(run_id: str) -> str | None:
    if not _SAFE_RUN_ID_RE.fullmatch(run_id):
        return "--run-id must contain only letters, digits, dot, underscore, or dash"
    if ".." in run_id:
        return "--run-id must not contain consecutive dots"
    return None


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _run_id_timestamp(created_at: str) -> str:
    return created_at.replace("-", "").replace(":", "").replace(".", "")


def _zone_selection_fingerprint(zone_ids: list[str]) -> str:
    payload = json.dumps(zone_ids, ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def _window_label(start_date: str, end_date: str) -> str:
    if start_date.endswith("-01-01") and end_date.endswith("-12-31"):
        return f"{start_date[:4]}_{end_date[:4]}"
    slug = f"{start_date}_{end_date}"
    return slug.replace("-", "_")


def _artifact_window_label(args: argparse.Namespace) -> str:
    if args.artifact_window_label:
        if error := unified_pilot_panel._safe_path_label_error(args.artifact_window_label):
            raise ValueError(error)
        return args.artifact_window_label
    return _window_label(args.holdout_start_date, args.holdout_end_date)


def _unified_window_label(args: argparse.Namespace) -> str:
    if args.unified_window_label:
        if error := unified_pilot_panel._safe_path_label_error(
            args.unified_window_label,
            flag="--unified-window-label",
        ):
            raise ValueError(error)
        return args.unified_window_label
    return _window_label(args.unified_start_date, args.unified_end_date)


def _resolve_run_id(args: argparse.Namespace, *, zone_ids: list[str], created_at: str) -> str:
    if args.run_id:
        if error := _safe_run_id_error(args.run_id):
            raise ValueError(error)
        return args.run_id
    zone_scope = args.zone_set or ("pilot-3" if not args.zone_id else f"{len(zone_ids)}zones")
    zone_scope = f"{zone_scope}_{_zone_selection_fingerprint(zone_ids)}"
    return (
        f"{_run_id_timestamp(created_at)}_holdout_{zone_scope}_"
        f"{_artifact_window_label(args)}_{_unified_window_label(args)}"
    )


def _stage_run_id(run_id: str, stage: str) -> str:
    return f"{run_id}-{stage}"


def _validate_args(args: argparse.Namespace) -> None:
    if args.era5_only and args.skip_era5:
        raise ValueError("--era5-only and --skip-era5 cannot be combined")
    if error := unified_pilot_panel._validate_date_range(
        args.holdout_start_date,
        args.holdout_end_date,
    ):
        raise ValueError(f"holdout window invalid: {error}")
    if not args.era5_only:
        if error := unified_pilot_panel._validate_date_range(
            args.unified_start_date,
            args.unified_end_date,
        ):
            raise ValueError(f"unified window invalid: {error}")
        if (
            args.unified_start_date < args.holdout_start_date
            or args.unified_end_date > args.holdout_end_date
        ):
            raise ValueError("unified window must be inside the holdout ERA5 window")
    if args.storage_mode == "s3" and not unified_pilot_panel.settings.S3_BUCKET:
        raise ValueError("--storage-mode s3 requires S3_BUCKET")
    _artifact_window_label(args)
    _unified_window_label(args)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _holdout_slug(args: argparse.Namespace) -> str:
    return f"{args.holdout_start_date}_{args.holdout_end_date}"


def _era5_source_template_for_local(output_dir: Path, args: argparse.Namespace) -> str:
    return (
        output_dir / "era5-holdout" / f"{{zone_id}}-era5-{_holdout_slug(args)}.parquet"
    ).as_posix()


def _era5_source_template_for_s3(era5_label: str) -> str:
    bucket = unified_pilot_panel.settings.S3_BUCKET
    return f"s3://{bucket}/climate/holdout/era5/{era5_label}/{{zone_id}}.parquet"


def _stage_summary(payload: dict[str, Any] | None) -> dict[str, Any] | None:
    if payload is None:
        return None
    return {
        "ok": payload.get("ok"),
        "run_id": payload.get("run_id"),
        "summary_path": payload.get("summary_path"),
        "manifest_path": payload.get("manifest_path"),
        "run_manifest_uri": payload.get("run_manifest_uri"),
        "run_manifest_written": payload.get("run_manifest_written"),
        "n_zones_requested": payload.get("n_zones_requested"),
        "n_zones_completed": payload.get("n_zones_completed"),
        "failed_zone_ids": payload.get("failed_zone_ids"),
        "stop_reason": payload.get("stop_reason"),
    }


def _era5_cli_args(
    args: argparse.Namespace,
    *,
    zone_ids: list[str],
    output_dir: Path,
    era5_label: str,
    run_id: str,
) -> list[str]:
    values = [
        "--zone-id",
        *zone_ids,
        "--storage-mode",
        args.storage_mode,
        "--artifact-scope",
        "holdout",
        "--artifact-window-label",
        era5_label,
        "--start-date",
        args.holdout_start_date,
        "--end-date",
        args.holdout_end_date,
        "--output-dir",
        (output_dir / "era5-holdout").as_posix(),
        "--summary-name",
        "era5-holdout-summary.json",
        "--manifest-name",
        "era5-holdout-manifest.json",
        "--run-id",
        _stage_run_id(run_id, "era5"),
        "--min-completeness-pct",
        str(args.min_completeness_pct),
        "--archive-chunk-days",
        str(
            args.archive_chunk_days
            or climate_baseline_panel._expected_days(args.holdout_start_date, args.holdout_end_date)
        ),
        "--stop-on-rate-limit",
    ]
    if args.variables:
        values.extend(["--variables", *args.variables])
    if args.preflight_only:
        values.append("--preflight-only")
    if args.fail_fast:
        values.append("--fail-fast")
    if args.zone_delay_s:
        values.extend(["--zone-delay-s", str(args.zone_delay_s)])
    return values


def _unified_cli_args(
    args: argparse.Namespace,
    *,
    zone_ids: list[str],
    output_dir: Path,
    era5_label: str,
    unified_label: str,
    run_id: str,
) -> list[str]:
    values = [
        "--zone-id",
        *zone_ids,
        "--storage-mode",
        args.storage_mode,
        "--artifact-scope",
        "holdout",
        "--artifact-window-label",
        unified_label,
        "--start-date",
        args.unified_start_date,
        "--end-date",
        args.unified_end_date,
        "--output-dir",
        (output_dir / "unified-holdout").as_posix(),
        "--summary-name",
        "unified-holdout-summary.json",
        "--manifest-name",
        "unified-holdout-manifest.json",
        "--run-id",
        _stage_run_id(run_id, "unified"),
        "--tolerance-days",
        str(args.tolerance_days),
        "--cloud-cover-max",
        str(args.cloud_cover_max),
    ]
    if args.era5_source_template:
        values.extend(["--era5-source-template", args.era5_source_template])
        if args.storage_mode == "local":
            values.append("--no-run-manifest")
    elif args.storage_mode == "s3":
        values.extend(
            [
                "--era5-source-template",
                _era5_source_template_for_s3(era5_label),
            ]
        )
    elif args.storage_mode == "local":
        values.extend(
            [
                "--era5-source-template",
                _era5_source_template_for_local(output_dir, args),
                "--no-run-manifest",
            ]
        )
    if args.preflight_only:
        values.append("--preflight-only")
    if args.fail_fast:
        values.append("--fail-fast")
    if args.zone_delay_s:
        values.extend(["--zone-delay-s", str(args.zone_delay_s)])
    return values


def _manifest_payload(panel: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "kind": "holdout_pilot_panel_manifest",
        "run_id": panel.get("run_id"),
        "created_at": panel.get("created_at"),
        "ok": panel.get("ok"),
        "storage_mode": panel.get("storage_mode"),
        "artifact_scope": "holdout",
        "artifact_window_label": panel.get("artifact_window_label"),
        "era5_artifact_window_label": panel.get("era5_artifact_window_label"),
        "unified_artifact_window_label": panel.get("unified_artifact_window_label"),
        "zone_ids": panel.get("zone_ids"),
        "holdout_window": panel.get("holdout_window"),
        "unified_window": panel.get("unified_window"),
        "preflight_only": panel.get("preflight_only"),
        "era5_only": panel.get("era5_only"),
        "skip_era5": panel.get("skip_era5"),
        "stages": panel.get("stages"),
    }


async def run_panel(
    args: argparse.Namespace,
    *,
    era5_runner: PanelRunner | None = None,
    unified_runner: UnifiedRunner | None = None,
    sleeper: Sleeper = asyncio.sleep,
    run_manifest_writer: RunManifestWriter = climate_baseline_panel._write_s3_json,
) -> dict[str, Any]:
    _validate_args(args)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    started_at = time.monotonic()
    created_at = _utc_now()
    zone_ids = _resolve_zone_ids(args)
    era5_label = _artifact_window_label(args)
    unified_label = _unified_window_label(args)
    run_id = _resolve_run_id(args, zone_ids=zone_ids, created_at=created_at)

    era5_payload: dict[str, Any] | None = None
    unified_payload: dict[str, Any] | None = None
    stop_reason: str | None = None

    if not args.skip_era5:
        era5_args = climate_baseline_panel.build_parser().parse_args(
            _era5_cli_args(
                args,
                zone_ids=zone_ids,
                output_dir=output_dir,
                era5_label=era5_label,
                run_id=run_id,
            )
        )
        era5_payload = await climate_baseline_panel.run_panel(
            era5_args,
            runner=era5_runner or climate_baseline_panel.run_pipeline,
            sleeper=sleeper,
            run_manifest_writer=run_manifest_writer,
        )
        if not era5_payload.get("ok"):
            stop_reason = "era5_failed"

    if not args.era5_only and stop_reason is None:
        unified_args = unified_pilot_panel.build_parser().parse_args(
            _unified_cli_args(
                args,
                zone_ids=zone_ids,
                output_dir=output_dir,
                era5_label=era5_label,
                unified_label=unified_label,
                run_id=run_id,
            )
        )
        unified_payload = await unified_pilot_panel.run_panel(
            unified_args,
            enrich_runner=unified_runner,
            sleeper=sleeper,
            run_manifest_writer=run_manifest_writer,
        )
        if not unified_payload.get("ok"):
            stop_reason = "unified_failed"

    ok = (era5_payload is None or bool(era5_payload.get("ok"))) and (
        unified_payload is None or bool(unified_payload.get("ok"))
    )
    panel = {
        "schema_version": 1,
        "kind": "holdout_pilot_panel_summary",
        "ok": ok,
        "run_id": run_id,
        "created_at": created_at,
        "zone_set": args.zone_set or ("pilot-3" if not args.zone_id else "explicit"),
        "zone_ids": zone_ids,
        "storage_mode": args.storage_mode,
        "artifact_scope": "holdout",
        "artifact_window_label": era5_label,
        "era5_artifact_window_label": era5_label,
        "unified_artifact_window_label": unified_label,
        "holdout_window": {
            "start_date": args.holdout_start_date,
            "end_date": args.holdout_end_date,
        },
        "unified_window": {
            "start_date": args.unified_start_date,
            "end_date": args.unified_end_date,
        },
        "variables": args.variables or list(climate_baseline_panel.NORMALIZED_VARS),
        "archive_chunk_days": args.archive_chunk_days
        or climate_baseline_panel._expected_days(args.holdout_start_date, args.holdout_end_date),
        "min_completeness_pct": args.min_completeness_pct,
        "tolerance_days": args.tolerance_days,
        "cloud_cover_max": args.cloud_cover_max,
        "preflight_only": args.preflight_only,
        "era5_only": args.era5_only,
        "skip_era5": args.skip_era5,
        "stop_reason": stop_reason,
        "output_dir": output_dir.as_posix(),
        "summary_path": (output_dir / args.summary_name).as_posix(),
        "manifest_path": (output_dir / args.manifest_name).as_posix(),
        "duration_ms": round((time.monotonic() - started_at) * 1000),
        "stages": {
            "era5_holdout": _stage_summary(era5_payload),
            "unified_holdout": _stage_summary(unified_payload),
        },
    }
    _write_json(output_dir / args.summary_name, panel)
    _write_json(output_dir / args.manifest_name, _manifest_payload(panel))
    return panel


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        panel = asyncio.run(run_panel(args))
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    print(f"OK={panel['ok']} summary={panel['summary_path']}")
    if not panel["ok"]:
        print(f"stop_reason={panel.get('stop_reason')}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
