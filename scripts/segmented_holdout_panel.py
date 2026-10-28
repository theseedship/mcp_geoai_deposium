#!/usr/bin/env python3
"""Run recent-holdout ERA5 once, then B4 unified joins over segmented windows.

This wrapper is intentionally operational glue. It does not implement climate,
NDVI, or joins; it orchestrates ``scripts/holdout_pilot_panel.py`` so a wide B4
holdout window can be split into smaller artifacts such as monthly windows.
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
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any, NamedTuple

# Allow `python scripts/segmented_holdout_panel.py` from anywhere.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts import climate_baseline_panel, holdout_pilot_panel  # noqa: E402

PanelRunner = Callable[[argparse.Namespace], Awaitable[dict[str, Any]]]
UnifiedRunner = Callable[..., Awaitable[dict[str, Any]]]
Sleeper = Callable[[float], Awaitable[None]]
RunManifestWriter = Callable[[str, dict[str, Any]], None]

PILOT_ZONE_IDS = holdout_pilot_panel.PILOT_ZONE_IDS
ZONE_SETS = ("pilot-3", "full-panel")
DEFAULT_HOLDOUT_START_DATE = holdout_pilot_panel.DEFAULT_HOLDOUT_START_DATE
DEFAULT_HOLDOUT_END_DATE = holdout_pilot_panel.DEFAULT_HOLDOUT_END_DATE
DEFAULT_UNIFIED_START_DATE = "2023-07-01"
DEFAULT_UNIFIED_END_DATE = "2023-09-30"
_SAFE_RUN_ID_RE = re.compile(r"^[A-Za-z0-9._-]+$")


class Segment(NamedTuple):
    label: str
    start_date: str
    end_date: str

    def as_window(self) -> dict[str, str]:
        return {"start_date": self.start_date, "end_date": self.end_date}


def _nonnegative_float(value: str) -> float:
    try:
        number = float(value)
    except ValueError:
        raise argparse.ArgumentTypeError("must be a non-negative number") from None
    if not math.isfinite(number) or number < 0:
        raise argparse.ArgumentTypeError("must be a finite non-negative number")
    return number


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="segmented_holdout_panel",
        description=(
            "Materialize/check recent holdout ERA5 once, then run B4 NDVI x ERA5 "
            "holdout joins over segmented windows such as monthly 2023 Q3 slices."
        ),
    )
    zone_selection = p.add_mutually_exclusive_group()
    zone_selection.add_argument("--zone-id", nargs="+", help="one or more zone ids")
    zone_selection.add_argument(
        "--zone-set",
        choices=ZONE_SETS,
        help=(
            "predefined zone set; pilot-3 is the default, full-panel loads the "
            "15-zone B1 artifact from zones/zones.geojson"
        ),
    )
    p.add_argument("--holdout-start-date", default=DEFAULT_HOLDOUT_START_DATE)
    p.add_argument("--holdout-end-date", default=DEFAULT_HOLDOUT_END_DATE)
    p.add_argument("--unified-start-date", default=DEFAULT_UNIFIED_START_DATE)
    p.add_argument("--unified-end-date", default=DEFAULT_UNIFIED_END_DATE)
    p.add_argument(
        "--segment-grain",
        choices=("monthly",),
        default="monthly",
        help="B4 segmentation grain; monthly is the first supported quota-safe mode",
    )
    p.add_argument(
        "--artifact-window-label",
        help=(
            "safe ERA5 holdout source label; default is YYYY_YYYY for whole-year "
            "holdout windows, else holdout start_end with dashes replaced by underscores"
        ),
    )
    p.add_argument(
        "--era5-source-template",
        help=(
            "explicit ERA5 source template for segmented B4 stages; may include {zone_id}. "
            "Defaults to S3 holdout paths in s3 mode, or the shared local ERA5 stage "
            "directory in local mode"
        ),
    )
    p.add_argument(
        "--storage-mode",
        choices=("local", "s3"),
        default="s3",
        help="artifact destination for child holdout stages; default s3/RustFS",
    )
    p.add_argument(
        "--variables",
        nargs="+",
        choices=list(climate_baseline_panel.NORMALIZED_VARS),
        help="normalized ERA5 variables; default is all six locked variables",
    )
    p.add_argument(
        "--output-dir",
        default="/tmp/geoai-segmented-holdout",
        help="directory for aggregate and per-segment JSON summaries",
    )
    p.add_argument("--summary-name", default="segmented-holdout-summary.json")
    p.add_argument("--manifest-name", default="segmented-holdout-manifest.json")
    p.add_argument("--run-id", help="stable root id for the segmented run")
    p.add_argument(
        "--archive-chunk-days",
        type=holdout_pilot_panel._positive_int,
        help=(
            "passed to the ERA5 stage; default is the full holdout window length "
            "so each zone can be requested in one Archive call"
        ),
    )
    p.add_argument(
        "--min-completeness-pct",
        type=holdout_pilot_panel._percent_threshold,
        default=98.0,
        help="ERA5 completeness gate; default 98",
    )
    p.add_argument(
        "--tolerance-days",
        type=lambda value: holdout_pilot_panel.unified_pilot_panel._bounded_int(
            value, minimum=0, maximum=31, label="tolerance_days"
        ),
        default=3,
        help="nearest-date NDVI-to-ERA5 join window in days; default 3",
    )
    p.add_argument(
        "--cloud-cover-max",
        type=lambda value: holdout_pilot_panel.unified_pilot_panel._bounded_int(
            value, minimum=0, maximum=100, label="cloud_cover_max"
        ),
        default=holdout_pilot_panel.unified_pilot_panel.settings.DEFAULT_CLOUD_COVER_MAX,
        help=(
            "max cloud cover percentage for NDVI; default "
            f"{holdout_pilot_panel.unified_pilot_panel.settings.DEFAULT_CLOUD_COVER_MAX}"
        ),
    )
    p.add_argument("--preflight-only", action="store_true")
    p.add_argument("--era5-only", action="store_true", help="materialize/check ERA5 but skip B4")
    p.add_argument(
        "--skip-era5",
        action="store_true",
        help="assume holdout ERA5 already exists and run only segmented B4 stages",
    )
    p.add_argument("--fail-fast", action="store_true")
    p.add_argument(
        "--zone-delay-s",
        type=_nonnegative_float,
        default=0.0,
        help="seconds to wait between zones inside each child stage",
    )
    p.add_argument(
        "--segment-delay-s",
        type=_nonnegative_float,
        default=0.0,
        help="seconds to wait between segmented B4 windows",
    )
    return p


def _flatten_zone_ids(values: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(values))


def _resolve_zone_ids(args: argparse.Namespace) -> list[str]:
    if args.zone_id:
        return _flatten_zone_ids(args.zone_id)
    zone_set = args.zone_set or "pilot-3"
    if zone_set == "pilot-3":
        return list(PILOT_ZONE_IDS)
    if zone_set == "full-panel":
        return climate_baseline_panel._load_full_panel_zone_ids()
    raise ValueError(f"unsupported zone set: {zone_set!r}")


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


def _parse_date(value: str, *, label: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError:
        raise ValueError(f"{label} must be an ISO date") from None


def _last_day_of_month(day: date) -> date:
    if day.month == 12:
        return date(day.year, 12, 31)
    return date(day.year, day.month + 1, 1) - timedelta(days=1)


def _monthly_segments(start_date: str, end_date: str) -> list[Segment]:
    start = _parse_date(start_date, label="unified-start-date")
    end = _parse_date(end_date, label="unified-end-date")
    segments: list[Segment] = []
    current = start
    while current <= end:
        segment_end = min(_last_day_of_month(current), end)
        full_month = current.day == 1 and segment_end == _last_day_of_month(current)
        label = (
            f"{current.year:04d}_{current.month:02d}"
            if full_month
            else f"{current.isoformat()}_{segment_end.isoformat()}".replace("-", "_")
        )
        segments.append(
            Segment(label=label, start_date=current.isoformat(), end_date=segment_end.isoformat())
        )
        current = segment_end + timedelta(days=1)
    return segments


def _segments(args: argparse.Namespace) -> list[Segment]:
    if args.segment_grain == "monthly":
        return _monthly_segments(args.unified_start_date, args.unified_end_date)
    raise ValueError(f"unsupported segment grain: {args.segment_grain!r}")


def _artifact_window_label(args: argparse.Namespace) -> str:
    if args.artifact_window_label:
        if error := holdout_pilot_panel.unified_pilot_panel._safe_path_label_error(
            args.artifact_window_label
        ):
            raise ValueError(error)
        return args.artifact_window_label
    return holdout_pilot_panel._window_label(args.holdout_start_date, args.holdout_end_date)


def _resolve_run_id(
    args: argparse.Namespace,
    *,
    zone_ids: list[str],
    segments: list[Segment],
    created_at: str,
) -> str:
    if args.run_id:
        if error := _safe_run_id_error(args.run_id):
            raise ValueError(error)
        return args.run_id
    zone_scope = args.zone_set or ("pilot-3" if not args.zone_id else f"{len(zone_ids)}zones")
    zone_scope = f"{zone_scope}_{_zone_selection_fingerprint(zone_ids)}"
    segment_scope = "era5_only" if args.era5_only else f"{segments[0].label}_{segments[-1].label}"
    return (
        f"{_run_id_timestamp(created_at)}_segmented_holdout_{zone_scope}_"
        f"{_artifact_window_label(args)}_{segment_scope}"
    )


def _validate_args(args: argparse.Namespace) -> None:
    if args.era5_only and args.skip_era5:
        raise ValueError("--era5-only and --skip-era5 cannot be combined")
    if error := holdout_pilot_panel.unified_pilot_panel._validate_date_range(
        args.holdout_start_date,
        args.holdout_end_date,
    ):
        raise ValueError(f"holdout window invalid: {error}")
    if not args.era5_only:
        if error := holdout_pilot_panel.unified_pilot_panel._validate_date_range(
            args.unified_start_date,
            args.unified_end_date,
        ):
            raise ValueError(f"unified window invalid: {error}")
        if (
            args.unified_start_date < args.holdout_start_date
            or args.unified_end_date > args.holdout_end_date
        ):
            raise ValueError("unified window must be inside the holdout ERA5 window")
    if args.storage_mode == "s3" and not holdout_pilot_panel.unified_pilot_panel.settings.S3_BUCKET:
        raise ValueError("--storage-mode s3 requires S3_BUCKET")
    _artifact_window_label(args)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text())
    except FileNotFoundError:
        return None
    if isinstance(payload, dict):
        return payload
    return None


def _child_common_args(
    args: argparse.Namespace,
    *,
    zone_ids: list[str],
    era5_label: str,
) -> list[str]:
    values = [
        "--zone-id",
        *zone_ids,
        "--storage-mode",
        args.storage_mode,
        "--artifact-window-label",
        era5_label,
        "--holdout-start-date",
        args.holdout_start_date,
        "--holdout-end-date",
        args.holdout_end_date,
        "--min-completeness-pct",
        str(args.min_completeness_pct),
        "--tolerance-days",
        str(args.tolerance_days),
        "--cloud-cover-max",
        str(args.cloud_cover_max),
    ]
    if args.variables:
        values.extend(["--variables", *args.variables])
    if args.archive_chunk_days:
        values.extend(["--archive-chunk-days", str(args.archive_chunk_days)])
    if args.preflight_only:
        values.append("--preflight-only")
    if args.fail_fast:
        values.append("--fail-fast")
    if args.zone_delay_s:
        values.extend(["--zone-delay-s", str(args.zone_delay_s)])
    return values


def _local_era5_source_template(output_dir: Path, args: argparse.Namespace) -> str:
    return (
        output_dir
        / "era5-stage"
        / "era5-holdout"
        / f"{{zone_id}}-era5-{args.holdout_start_date}_{args.holdout_end_date}.parquet"
    ).as_posix()


def _era5_cli_args(
    args: argparse.Namespace,
    *,
    zone_ids: list[str],
    output_dir: Path,
    era5_label: str,
    run_id: str,
) -> list[str]:
    return [
        *_child_common_args(args, zone_ids=zone_ids, era5_label=era5_label),
        "--era5-only",
        "--output-dir",
        output_dir.as_posix(),
        "--summary-name",
        "era5-stage-summary.json",
        "--manifest-name",
        "era5-stage-manifest.json",
        "--run-id",
        run_id,
    ]


def _segment_cli_args(
    args: argparse.Namespace,
    *,
    zone_ids: list[str],
    output_dir: Path,
    root_output_dir: Path,
    era5_label: str,
    run_id: str,
    segment: Segment,
) -> list[str]:
    values = [
        *_child_common_args(args, zone_ids=zone_ids, era5_label=era5_label),
        "--skip-era5",
        "--unified-start-date",
        segment.start_date,
        "--unified-end-date",
        segment.end_date,
        "--unified-window-label",
        segment.label,
        "--output-dir",
        output_dir.as_posix(),
        "--summary-name",
        f"{segment.label}-summary.json",
        "--manifest-name",
        f"{segment.label}-manifest.json",
        "--run-id",
        f"{run_id}-{segment.label}",
    ]
    if args.era5_source_template:
        values.extend(["--era5-source-template", args.era5_source_template])
    elif args.storage_mode == "local":
        values.extend(
            ["--era5-source-template", _local_era5_source_template(root_output_dir, args)]
        )
    return values


def _number(value: Any) -> int | float | None:
    return value if isinstance(value, int | float) and math.isfinite(value) else None


def _sum_zone_stage(zones: list[dict[str, Any]], key: str) -> int | float | None:
    total: int | float = 0
    saw_value = False
    for zone in zones:
        stage = zone.get("stage") if isinstance(zone.get("stage"), dict) else {}
        value = _number(stage.get(key))
        if value is None:
            continue
        total += value
        saw_value = True
    return total if saw_value else None


def _load_unified_child(segment_dir: Path) -> dict[str, Any] | None:
    return _read_json(segment_dir / "unified-holdout" / "unified-holdout-summary.json")


def _segment_summary(
    *,
    segment: Segment,
    output_dir: Path,
    payload: dict[str, Any],
) -> dict[str, Any]:
    unified_payload = _load_unified_child(output_dir)
    zones = []
    if isinstance(unified_payload, dict) and isinstance(unified_payload.get("zones"), list):
        zones = [zone for zone in unified_payload["zones"] if isinstance(zone, dict)]

    truncation_warnings: list[dict[str, str]] = []
    for zone in zones:
        stage = zone.get("stage") if isinstance(zone.get("stage"), dict) else {}
        warning = stage.get("ndvi_truncation_warning")
        if isinstance(warning, str) and warning:
            truncation_warnings.append({"zone_id": str(zone.get("zone_id")), "warning": warning})

    stage = payload.get("stages") if isinstance(payload.get("stages"), dict) else {}
    unified_stage = (
        stage.get("unified_holdout") if isinstance(stage.get("unified_holdout"), dict) else None
    )
    return {
        "label": segment.label,
        "window": segment.as_window(),
        "ok": bool(payload.get("ok")),
        "run_id": payload.get("run_id"),
        "summary_path": payload.get("summary_path"),
        "manifest_path": payload.get("manifest_path"),
        "unified_stage": unified_stage,
        "n_zones_requested": payload.get("zone_ids") and len(payload.get("zone_ids")),
        "n_zones_completed": unified_payload.get("n_zones_completed") if unified_payload else None,
        "failed_zone_ids": unified_payload.get("failed_zone_ids") if unified_payload else None,
        "totals": {
            "images_searched": _sum_zone_stage(zones, "images_searched"),
            "n_ndvi_observations": _sum_zone_stage(zones, "n_ndvi_observations"),
            "n_matched": _sum_zone_stage(zones, "n_matched"),
        },
        "truncation_warnings": truncation_warnings,
    }


def _stage_summary(payload: dict[str, Any] | None) -> dict[str, Any] | None:
    if payload is None:
        return None
    return {
        "ok": payload.get("ok"),
        "run_id": payload.get("run_id"),
        "summary_path": payload.get("summary_path"),
        "manifest_path": payload.get("manifest_path"),
        "stages": payload.get("stages"),
        "stop_reason": payload.get("stop_reason"),
    }


def _segment_totals(segments: list[dict[str, Any]], key: str) -> int | float | None:
    total: int | float = 0
    saw_value = False
    for segment in segments:
        totals = segment.get("totals") if isinstance(segment.get("totals"), dict) else {}
        value = _number(totals.get(key))
        if value is None:
            continue
        total += value
        saw_value = True
    return total if saw_value else None


def _truncation_warnings(segments: list[dict[str, Any]]) -> list[dict[str, str]]:
    warnings: list[dict[str, str]] = []
    for segment in segments:
        label = segment.get("label")
        for warning in segment.get("truncation_warnings") or []:
            if not isinstance(warning, dict):
                continue
            warnings.append(
                {
                    "segment_label": str(label),
                    "zone_id": str(warning.get("zone_id")),
                    "warning": str(warning.get("warning")),
                }
            )
    return warnings


def _panel_payload(
    *,
    args: argparse.Namespace,
    output_dir: Path,
    started_at: float,
    created_at: str,
    run_id: str,
    zone_ids: list[str],
    era5_label: str,
    planned_segments: list[Segment],
    era5_payload: dict[str, Any] | None,
    segment_payloads: list[dict[str, Any]],
    stop_reason: str | None,
) -> dict[str, Any]:
    failed_segments = [segment["label"] for segment in segment_payloads if not segment.get("ok")]
    warnings = _truncation_warnings(segment_payloads)
    segment_ok = args.era5_only or (
        len(segment_payloads) == len(planned_segments) and not failed_segments
    )
    era5_ok = args.skip_era5 or era5_payload is None or bool(era5_payload.get("ok"))
    return {
        "schema_version": 1,
        "kind": "segmented_holdout_panel_summary",
        "ok": bool(era5_ok and segment_ok),
        "run_id": run_id,
        "created_at": created_at,
        "zone_set": args.zone_set or ("pilot-3" if not args.zone_id else "explicit"),
        "zone_ids": zone_ids,
        "storage_mode": args.storage_mode,
        "artifact_scope": "holdout",
        "era5_artifact_window_label": era5_label,
        "segment_grain": args.segment_grain,
        "holdout_window": {
            "start_date": args.holdout_start_date,
            "end_date": args.holdout_end_date,
        },
        "unified_window": None
        if args.era5_only
        else {
            "start_date": args.unified_start_date,
            "end_date": args.unified_end_date,
        },
        "planned_segments": [
            {"label": segment.label, "window": segment.as_window()} for segment in planned_segments
        ],
        "completed_segment_labels": [segment["label"] for segment in segment_payloads],
        "failed_segment_labels": failed_segments,
        "n_segments_requested": len(planned_segments),
        "n_segments_completed": len(segment_payloads),
        "variables": args.variables or list(climate_baseline_panel.NORMALIZED_VARS),
        "archive_chunk_days": args.archive_chunk_days
        or climate_baseline_panel._expected_days(args.holdout_start_date, args.holdout_end_date),
        "min_completeness_pct": args.min_completeness_pct,
        "tolerance_days": args.tolerance_days,
        "cloud_cover_max": args.cloud_cover_max,
        "preflight_only": args.preflight_only,
        "era5_only": args.era5_only,
        "skip_era5": args.skip_era5,
        "fail_fast": args.fail_fast,
        "zone_delay_s": args.zone_delay_s,
        "segment_delay_s": args.segment_delay_s,
        "stop_reason": stop_reason,
        "output_dir": output_dir.as_posix(),
        "summary_path": (output_dir / args.summary_name).as_posix(),
        "manifest_path": (output_dir / args.manifest_name).as_posix(),
        "duration_ms": round((time.monotonic() - started_at) * 1000),
        "stages": {
            "era5_holdout": _stage_summary(era5_payload),
            "segmented_unified_holdout": segment_payloads,
        },
        "totals": {
            "images_searched": _segment_totals(segment_payloads, "images_searched"),
            "n_ndvi_observations": _segment_totals(segment_payloads, "n_ndvi_observations"),
            "n_matched": _segment_totals(segment_payloads, "n_matched"),
        },
        "truncation_warning_count": len(warnings),
        "truncation_warnings": warnings,
    }


def _manifest_payload(panel: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "kind": "segmented_holdout_panel_manifest",
        "run_id": panel.get("run_id"),
        "created_at": panel.get("created_at"),
        "ok": panel.get("ok"),
        "storage_mode": panel.get("storage_mode"),
        "artifact_scope": panel.get("artifact_scope"),
        "era5_artifact_window_label": panel.get("era5_artifact_window_label"),
        "segment_grain": panel.get("segment_grain"),
        "holdout_window": panel.get("holdout_window"),
        "unified_window": panel.get("unified_window"),
        "zone_set": panel.get("zone_set"),
        "zone_ids": panel.get("zone_ids"),
        "planned_segments": panel.get("planned_segments"),
        "completed_segment_labels": panel.get("completed_segment_labels"),
        "failed_segment_labels": panel.get("failed_segment_labels"),
        "truncation_warning_count": panel.get("truncation_warning_count"),
        "truncation_warnings": panel.get("truncation_warnings"),
        "totals": panel.get("totals"),
        "stages": panel.get("stages"),
        "stop_reason": panel.get("stop_reason"),
    }


def _write_panel_outputs(output_dir: Path, args: argparse.Namespace, panel: dict[str, Any]) -> None:
    _write_json(output_dir / args.summary_name, panel)
    _write_json(output_dir / args.manifest_name, _manifest_payload(panel))


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
    planned_segments = [] if args.era5_only else _segments(args)
    era5_label = _artifact_window_label(args)
    run_id = _resolve_run_id(
        args,
        zone_ids=zone_ids,
        segments=planned_segments
        or [Segment("era5_only", args.holdout_start_date, args.holdout_end_date)],
        created_at=created_at,
    )

    era5_payload: dict[str, Any] | None = None
    segment_payloads: list[dict[str, Any]] = []
    stop_reason: str | None = None

    if not args.skip_era5:
        era5_dir = output_dir / "era5-stage"
        era5_args = holdout_pilot_panel.build_parser().parse_args(
            _era5_cli_args(
                args,
                zone_ids=zone_ids,
                output_dir=era5_dir,
                era5_label=era5_label,
                run_id=run_id,
            )
        )
        era5_payload = await holdout_pilot_panel.run_panel(
            era5_args,
            era5_runner=era5_runner,
            unified_runner=unified_runner,
            sleeper=sleeper,
            run_manifest_writer=run_manifest_writer,
        )
        if not era5_payload.get("ok"):
            stop_reason = "era5_failed"
        panel = _panel_payload(
            args=args,
            output_dir=output_dir,
            started_at=started_at,
            created_at=created_at,
            run_id=run_id,
            zone_ids=zone_ids,
            era5_label=era5_label,
            planned_segments=planned_segments,
            era5_payload=era5_payload,
            segment_payloads=segment_payloads,
            stop_reason=stop_reason,
        )
        _write_panel_outputs(output_dir, args, panel)

    if not args.era5_only and stop_reason is None:
        for index, segment in enumerate(planned_segments):
            segment_dir = output_dir / "segments" / segment.label
            segment_args = holdout_pilot_panel.build_parser().parse_args(
                _segment_cli_args(
                    args,
                    zone_ids=zone_ids,
                    output_dir=segment_dir,
                    root_output_dir=output_dir,
                    era5_label=era5_label,
                    run_id=run_id,
                    segment=segment,
                )
            )
            child_payload = await holdout_pilot_panel.run_panel(
                segment_args,
                era5_runner=era5_runner,
                unified_runner=unified_runner,
                sleeper=sleeper,
                run_manifest_writer=run_manifest_writer,
            )
            segment_payloads.append(
                _segment_summary(segment=segment, output_dir=segment_dir, payload=child_payload)
            )
            if not child_payload.get("ok") and args.fail_fast:
                stop_reason = "segment_failed"
            panel = _panel_payload(
                args=args,
                output_dir=output_dir,
                started_at=started_at,
                created_at=created_at,
                run_id=run_id,
                zone_ids=zone_ids,
                era5_label=era5_label,
                planned_segments=planned_segments,
                era5_payload=era5_payload,
                segment_payloads=segment_payloads,
                stop_reason=stop_reason,
            )
            _write_panel_outputs(output_dir, args, panel)
            if stop_reason is not None:
                break
            if index < len(planned_segments) - 1 and args.segment_delay_s:
                await sleeper(args.segment_delay_s)

    panel = _panel_payload(
        args=args,
        output_dir=output_dir,
        started_at=started_at,
        created_at=created_at,
        run_id=run_id,
        zone_ids=zone_ids,
        era5_label=era5_label,
        planned_segments=planned_segments,
        era5_payload=era5_payload,
        segment_payloads=segment_payloads,
        stop_reason=stop_reason,
    )
    _write_panel_outputs(output_dir, args, panel)
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
