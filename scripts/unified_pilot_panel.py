#!/usr/bin/env python3
"""Run the B4 unified ERA5 x NDVI pilot panel.

This is an operator wrapper around the existing ``climate_enrich_ndvi`` tool.
It does not implement climate or NDVI logic; it only makes the pilot-zone run
repeatable, emits JSON summaries, and optionally publishes a RustFS/S3 run
manifest for downstream agents.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import time
from collections.abc import Awaitable, Callable, Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# Allow `python scripts/unified_pilot_panel.py` from anywhere.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts import climate_baseline_panel  # noqa: E402
from src.config import settings  # noqa: E402
from src.tools.climate import (  # noqa: E402
    _resolve_zone,
    _validate_date_range,
    _validate_dst,
    register_climate_tools,
)
from src.utils.duckdb_conn import duckdb_session  # noqa: E402
from src.utils.logging import setup_logging  # noqa: E402
from src.utils.s3_endpoint import (  # noqa: E402
    redact_credentials,
    redact_text,
    sanitize_stac_url,
)

PILOT_ZONE_IDS = ("montpellier_metropole", "bassin_du_lez", "fos_sur_mer")
ZONE_SETS = ("pilot-3",)
CANONICAL_BASELINE_START = "1991-01-01"
CANONICAL_BASELINE_END = "2020-12-31"
DEFAULT_START_DATE = "2020-07-01"
DEFAULT_END_DATE = "2020-07-31"
_SAFE_ZONE_RE = re.compile(r"^[A-Za-z0-9._-]+$")
_SAFE_RUN_ID_RE = re.compile(r"^[A-Za-z0-9._-]+$")
_SAFE_S3_JSON_URI_RE = re.compile(r"^[A-Za-z0-9._:/=-]+\.json$")
_SAFE_SQL_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

EnrichRunner = Callable[..., Awaitable[dict[str, Any]]]
Sleeper = Callable[[float], Awaitable[None]]
RunManifestWriter = Callable[[str, dict[str, Any]], None]


class _CaptureMCP:
    """Collect registered tool handlers without starting a FastMCP server."""

    def __init__(self) -> None:
        self.tools: dict[str, object] = {}

    def tool(self, *_args, **_kwargs):
        def _decorator(fn):
            self.tools[fn.__name__] = fn
            return fn

        return _decorator


def _load_enrich_tool() -> EnrichRunner:
    cap = _CaptureMCP()
    register_climate_tools(cap)
    return cap.tools["climate_enrich_ndvi"]  # type: ignore[return-value]


def _bounded_int(value: str, *, minimum: int, maximum: int, label: str) -> int:
    try:
        number = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{label} must be an integer") from None
    if not minimum <= number <= maximum:
        raise argparse.ArgumentTypeError(f"{label} must be in [{minimum}, {maximum}]")
    return number


def _nonnegative_float(value: str) -> float:
    try:
        number = float(value)
    except ValueError:
        raise argparse.ArgumentTypeError("must be a non-negative number") from None
    if not math.isfinite(number) or number < 0:
        raise argparse.ArgumentTypeError("must be a finite non-negative number")
    return number


def _tolerance_days(value: str) -> int:
    return _bounded_int(value, minimum=0, maximum=31, label="tolerance_days")


def _cloud_cover_max(value: str) -> int:
    return _bounded_int(value, minimum=0, maximum=100, label="cloud_cover_max")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="unified_pilot_panel",
        description=(
            "Run climate_enrich_ndvi over the B4 pilot zones, writing one JSON "
            "summary per zone plus an aggregate panel manifest."
        ),
    )
    zone_selection = p.add_mutually_exclusive_group()
    zone_selection.add_argument("--zone-id", nargs="+", help="one or more zone ids")
    zone_selection.add_argument(
        "--zone-set",
        choices=ZONE_SETS,
        help="predefined zone set; default is pilot-3 when --zone-id is omitted",
    )
    p.add_argument(
        "--start-date",
        default=DEFAULT_START_DATE,
        help=(
            "ISO date, inclusive; default is a bounded canonical-baseline proof window "
            f"({DEFAULT_START_DATE})"
        ),
    )
    p.add_argument(
        "--end-date",
        default=DEFAULT_END_DATE,
        help=(
            "ISO date, inclusive; default is a bounded canonical-baseline proof window "
            f"({DEFAULT_END_DATE})"
        ),
    )
    p.add_argument(
        "--storage-mode",
        choices=("local", "s3"),
        default="s3",
        help="unified parquet destination: local under --output-dir, or s3://$S3_BUCKET/climate/unified/{zone}.parquet",
    )
    p.add_argument(
        "--artifact-scope",
        choices=("canonical", "holdout"),
        default="canonical",
        help=(
            "canonical uses climate/{era5,unified}/{zone}.parquet; holdout uses "
            "climate/holdout/{era5,unified}/{window_label}/{zone}.parquet"
        ),
    )
    p.add_argument(
        "--artifact-window-label",
        help=(
            "safe path label for holdout artifacts; default is YYYY_YYYY for whole-year "
            "windows, else start_end with dashes replaced by underscores"
        ),
    )
    p.add_argument(
        "--era5-source-template",
        help=(
            "explicit ERA5 source template; may include {zone_id}. If omitted, "
            "uses s3://$S3_BUCKET/climate/era5/{zone_id}.parquet"
        ),
    )
    p.add_argument(
        "--ndvi-source-template",
        help=(
            "opt-in stored B3 NDVI source template; may include {zone_id}. When given, "
            "climate_enrich_ndvi reads that series and makes no STAC call; an absent, invalid "
            "or mismatched source fails the zone and never falls back to STAC. If omitted, NDVI "
            "is streamed from STAC as before"
        ),
    )
    p.add_argument(
        "--output-template",
        help=(
            "explicit unified output template; may include {zone_id}. If omitted, "
            "storage-mode chooses the canonical local or s3 destination"
        ),
    )
    p.add_argument(
        "--output-dir",
        default="/tmp/geoai-unified-pilot",
        help="directory for JSON summaries and, in local storage mode, unified parquets",
    )
    p.add_argument(
        "--summary-name",
        default="unified-pilot-summary.json",
        help="aggregate JSON filename inside --output-dir",
    )
    p.add_argument(
        "--manifest-name",
        default="unified-pilot-manifest.json",
        help="operator artifact manifest JSON filename inside --output-dir",
    )
    p.add_argument(
        "--run-id",
        help="stable id for the production run manifest; default is generated from UTC time and zone/window",
    )
    p.add_argument(
        "--tolerance-days",
        type=_tolerance_days,
        default=3,
        help="nearest-date NDVI-to-ERA5 join window in days; default 3",
    )
    p.add_argument(
        "--cloud-cover-max",
        type=_cloud_cover_max,
        default=settings.DEFAULT_CLOUD_COVER_MAX,
        help=f"max cloud cover percentage for NDVI observations; default {settings.DEFAULT_CLOUD_COVER_MAX}",
    )
    p.add_argument(
        "--fail-fast",
        action="store_true",
        help="stop after the first failed zone instead of collecting all zone results",
    )
    p.add_argument(
        "--preflight-only",
        action="store_true",
        help="validate zone selection and planned paths without reading ERA5, calling STAC, or writing parquets",
    )
    p.add_argument(
        "--zone-delay-s",
        type=_nonnegative_float,
        default=0.0,
        help="seconds to wait between zone runs; useful for STAC/raster rate limits",
    )
    p.add_argument(
        "--no-run-manifest",
        action="store_true",
        help="do not publish the production run manifest to s3://$S3_BUCKET/climate/runs/{run_id}/manifest.json",
    )
    return p


def _flatten_zone_ids(values: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(values))


def _resolve_zone_ids(args: argparse.Namespace) -> list[str]:
    if args.zone_id:
        return _flatten_zone_ids(args.zone_id)
    return list(PILOT_ZONE_IDS)


def _safe_zone_error(zone_id: str) -> str | None:
    if not _SAFE_ZONE_RE.fullmatch(zone_id):
        return f"Unsafe zone_id {zone_id!r}; use only letters, digits, dot, underscore, or dash"
    if _resolve_zone(zone_id) is None:
        return f"Unknown zone_id {zone_id!r} (not in zones/zones.geojson)"
    return None


def _run_id_error(run_id: str) -> str | None:
    if not _SAFE_RUN_ID_RE.fullmatch(run_id):
        return "--run-id must contain only letters, digits, dot, underscore, or dash"
    if ".." in run_id:
        return "--run-id must not contain consecutive dots"
    return None


def _safe_path_label_error(label: str, *, flag: str = "--artifact-window-label") -> str | None:
    if not _SAFE_RUN_ID_RE.fullmatch(label):
        return f"{flag} must contain only letters, digits, dot, underscore, or dash"
    if ".." in label:
        return f"{flag} must not contain consecutive dots"
    return None


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _run_id_timestamp(created_at: str) -> str:
    return created_at.replace("-", "").replace(":", "").replace(".", "").replace("+00:00", "Z")


def _zone_selection_fingerprint(zone_ids: list[str]) -> str:
    payload = json.dumps(zone_ids, ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def _resolve_run_id(args: argparse.Namespace, *, zone_ids: list[str], created_at: str) -> str:
    if args.run_id:
        if error := _run_id_error(args.run_id):
            raise ValueError(error)
        return args.run_id
    zone_scope = args.zone_set or ("pilot-3" if not args.zone_id else f"{len(zone_ids)}zones")
    zone_scope = f"{zone_scope}_{_zone_selection_fingerprint(zone_ids)}"
    return (
        f"{_run_id_timestamp(created_at)}_unified_{zone_scope}_{args.start_date}_{args.end_date}"
    ).replace("-", "_")


def _git_output(*args: str, empty_ok: bool = False) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", _REPO_ROOT.as_posix(), *args],
            text=True,
            capture_output=True,
            check=True,
        )
    except Exception:
        return None
    value = result.stdout.strip()
    if value or empty_ok:
        return value
    return None


def _env_first(*names: str) -> str | None:
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    return None


def _code_ref() -> dict[str, Any]:
    status = _git_output("status", "--porcelain", empty_ok=True)
    commit = _git_output("rev-parse", "HEAD") or _env_first(
        "RAILWAY_GIT_COMMIT_SHA",
        "GIT_COMMIT",
        "SOURCE_VERSION",
    )
    branch = _git_output("rev-parse", "--abbrev-ref", "HEAD") or _env_first(
        "RAILWAY_GIT_BRANCH",
        "GIT_BRANCH",
        "BRANCH_NAME",
    )
    return {
        "repo": "deposium_geoai",
        "commit": commit,
        "branch": branch,
        "dirty": None if status is None else bool(status),
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _zone_artifact(zone_ids: list[str]) -> dict[str, Any]:
    path = _REPO_ROOT / "zones" / "zones.geojson"
    return {
        "path": "zones/zones.geojson",
        "sha256": _sha256_file(path),
        "selected_zone_ids": zone_ids,
        "n_selected": len(zone_ids),
    }


def _format_template(template: str, *, zone_id: str) -> str:
    try:
        return template.format(zone_id=zone_id)
    except (IndexError, KeyError, ValueError) as exc:
        raise ValueError(f"Invalid path template {template!r}: {exc}") from exc


def _window_slug(start_date: str, end_date: str) -> str:
    return f"{start_date}_{end_date}"


def _artifact_window_label(args: argparse.Namespace) -> str:
    if args.artifact_window_label:
        if error := _safe_path_label_error(args.artifact_window_label):
            raise ValueError(error)
        return args.artifact_window_label
    if args.start_date.endswith("-01-01") and args.end_date.endswith("-12-31"):
        return f"{args.start_date[:4]}_{args.end_date[:4]}"
    return _window_slug(args.start_date, args.end_date).replace("-", "_")


def _resolve_artifact_paths(
    args: argparse.Namespace, *, zone_id: str, output_dir: Path
) -> dict[str, str | None]:
    artifact_label = _artifact_window_label(args)
    if args.era5_source_template:
        era5_source = _format_template(args.era5_source_template, zone_id=zone_id)
    elif args.artifact_scope == "holdout" and settings.S3_BUCKET:
        era5_source = (
            f"s3://{settings.S3_BUCKET}/climate/holdout/era5/{artifact_label}/{zone_id}.parquet"
        )
    elif settings.S3_BUCKET:
        era5_source = f"s3://{settings.S3_BUCKET}/climate/era5/{zone_id}.parquet"
    else:
        era5_source = None

    if args.output_template:
        unified_output = _format_template(args.output_template, zone_id=zone_id)
    elif args.artifact_scope == "holdout" and args.storage_mode == "s3":
        unified_output = (
            f"s3://{settings.S3_BUCKET}/climate/holdout/unified/{artifact_label}/{zone_id}.parquet"
            if settings.S3_BUCKET
            else None
        )
    elif args.storage_mode == "s3":
        unified_output = (
            f"s3://{settings.S3_BUCKET}/climate/unified/{zone_id}.parquet"
            if settings.S3_BUCKET
            else None
        )
    else:
        unified_output = (
            output_dir / f"{zone_id}-unified-{_window_slug(args.start_date, args.end_date)}.parquet"
        ).as_posix()

    # `is not None`, not truthiness: an empty template is an opt-in that resolved to nothing,
    # and reading it as "omitted" would route the zone to STAC — the fallback the stored-source
    # contract forbids (same defect as PR #188 on the tool side).
    ndvi_source = (
        _format_template(args.ndvi_source_template, zone_id=zone_id)
        if args.ndvi_source_template is not None
        else None
    )
    return {
        "era5_source": era5_source,
        "ndvi_source": ndvi_source,
        "unified_output": unified_output,
    }


def _path_validation_error(path: str | None, *, label: str) -> str | None:
    if not path:
        return f"No {label} resolved — set S3_BUCKET or pass an explicit template"
    try:
        _validate_dst(path)
    except ValueError as exc:
        return f"Invalid {label}: {exc}"
    return None


def _ndvi_source_error(args: argparse.Namespace, paths: dict[str, str | None]) -> str | None:
    """Preflight an opted-in stored NDVI source; ``None`` when the caller did not opt in."""
    if args.ndvi_source_template is None:
        return None
    if not paths["ndvi_source"]:
        return "Empty ndvi_source from --ndvi-source-template; refusing rather than using STAC"
    return _path_validation_error(paths["ndvi_source"], label="ndvi_source")


def _artifact_window_validation_error(args: argparse.Namespace) -> str | None:
    if args.era5_source_template or args.artifact_scope != "canonical":
        return None
    if args.start_date < CANONICAL_BASELINE_START or args.end_date > CANONICAL_BASELINE_END:
        return (
            "canonical ERA5 artifacts cover "
            f"{CANONICAL_BASELINE_START}..{CANONICAL_BASELINE_END}; use --artifact-scope holdout "
            "after materializing recent ERA5, or pass --era5-source-template"
        )
    return None


def _run_manifest_uri(args: argparse.Namespace, run_id: str) -> str | None:
    if args.no_run_manifest or not settings.S3_BUCKET:
        return None
    if _run_id_error(run_id):
        return None
    return f"s3://{settings.S3_BUCKET}/climate/runs/{run_id}/manifest.json"


def _validate_run_manifest_uri(uri: str) -> None:
    if not _SAFE_S3_JSON_URI_RE.fullmatch(uri):
        raise ValueError(f"unsafe run manifest URI: {uri!r}")
    if ".." in uri:
        raise ValueError(f"path traversal not allowed in run manifest URI: {uri!r}")
    prefix = f"s3://{settings.S3_BUCKET}/climate/runs/" if settings.S3_BUCKET else None
    if not prefix or not uri.startswith(prefix):
        raise ValueError(f"run manifest URI must be under s3://{settings.S3_BUCKET}/climate/runs/")


def _write_s3_json(uri: str, payload: dict[str, Any]) -> None:
    """Write one JSON object to a confined S3/RustFS URI through DuckDB httpfs.

    Redacted at THIS boundary, not at each `except` site: there are a dozen of those across the
    panels and one more arrives with every new stage, whereas this is the one place a manifest
    becomes an object in a bucket. `redact_credentials` touches only the diagnostic fields --
    `errors[].error`, `run_manifest_errors[].error`, the notes and warnings -- and leaves every
    identifier, URI and path exactly as written, so the published manifest still resolves.
    """
    _validate_run_manifest_uri(uri)
    payload = redact_credentials(payload, settings.S3_ENDPOINT)
    columns = list(payload)
    unsafe_columns = [c for c in columns if not _SAFE_SQL_IDENTIFIER_RE.fullmatch(c)]
    if unsafe_columns:
        raise ValueError(f"unsafe run manifest columns: {unsafe_columns}")
    projections = ", ".join(f'? AS "{column}"' for column in columns)
    values = [payload[column] for column in columns]
    with duckdb_session(":memory:", load_httpfs=True) as conn:
        conn.execute(f"COPY (SELECT {projections}) TO '{uri}' (FORMAT JSON)", values)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    """Serialize one panel payload to a LOCAL file, redacted the same way the S3 copy is.

    `_write_s3_json` was the only redacting writer, on the reading that the object in the bucket
    is the artifact handed to another party. A local run manifest is an artifact too: it is what
    an operator opens, attaches to a ticket, uploads out of a CI job, or feeds back in through
    `--summarize-existing`. Measured on this panel with the repo's own dev credentials
    (`S3_ACCESS_KEY=deposium-dev` / `S3_SECRET_KEY=deposium-dev-secret`) and one httpfs failure
    copied verbatim into `errors[].error`:

        local file bytes: LEAK[deposium-dev-secret] = True

    while the S3 copy of the same payload came out clean. Both spellings of one run manifest now
    go through the same redactor, so they cannot disagree about what they carry.

    `redact_credentials` is scoped to the diagnostic fields, so every identifier, URI and path in
    the file is byte-for-byte what it was and a later `--summarize-existing` read still resolves.
    """
    payload = redact_credentials(payload, settings.S3_ENDPOINT)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _local_file_size(path: str | None) -> int | None:
    if not path or "://" in path:
        return None
    p = Path(path)
    if not p.exists() or not p.is_file():
        return None
    return p.stat().st_size


def _zone_summary_path(
    output_dir: Path,
    zone_id: str,
    start_date: str,
    end_date: str,
    *,
    index: int,
) -> Path:
    key = zone_id if _SAFE_ZONE_RE.fullmatch(zone_id) else f"invalid-zone-{index + 1}"
    return output_dir / f"{key}-unified-{_window_slug(start_date, end_date)}.json"


def _failed_zone_payload(
    *,
    args: argparse.Namespace,
    zone_id: str,
    output_dir: Path,
    started_at: float,
    failed_stage: str,
    error: str,
) -> dict[str, Any]:
    paths = _resolve_artifact_paths(args, zone_id=zone_id, output_dir=output_dir)
    return {
        "ok": False,
        "zone_id": zone_id,
        "window": {"start_date": args.start_date, "end_date": args.end_date},
        "storage_mode": args.storage_mode,
        "preflight_only": False,
        "tolerance_days": args.tolerance_days,
        "cloud_cover_max": args.cloud_cover_max,
        "era5_source": paths["era5_source"],
        "ndvi_source": paths["ndvi_source"],
        "unified_output": paths["unified_output"],
        "duration_ms": round((time.monotonic() - started_at) * 1000),
        "failed_stage": failed_stage,
        "error": error,
        "stage": {},
    }


def _preflight_zone_payload(
    *,
    args: argparse.Namespace,
    zone_id: str,
    output_dir: Path,
    started_at: float,
) -> dict[str, Any]:
    paths = _resolve_artifact_paths(args, zone_id=zone_id, output_dir=output_dir)
    error = _safe_zone_error(zone_id)
    if error is None:
        error = _path_validation_error(paths["era5_source"], label="era5_source")
    if error is None:
        error = _ndvi_source_error(args, paths)
    if error is None:
        error = _path_validation_error(paths["unified_output"], label="unified_output")
    if error is None:
        error = _artifact_window_validation_error(args)
    return {
        "ok": error is None,
        "zone_id": zone_id,
        "window": {"start_date": args.start_date, "end_date": args.end_date},
        "storage_mode": args.storage_mode,
        "preflight_only": True,
        "tolerance_days": args.tolerance_days,
        "cloud_cover_max": args.cloud_cover_max,
        "era5_source": paths["era5_source"],
        "ndvi_source": paths["ndvi_source"],
        "unified_output": paths["unified_output"],
        "duration_ms": round((time.monotonic() - started_at) * 1000),
        "failed_stage": "preflight" if error else None,
        "error": error,
        "stage": {},
        "notes": ["preflight-only: ERA5 read, STAC, raster IO, and parquet writes were not called"],
    }


def _compact_zone_payload(
    *,
    args: argparse.Namespace,
    zone_id: str,
    output_dir: Path,
    started_at: float,
    result: dict[str, Any],
) -> dict[str, Any]:
    paths = _resolve_artifact_paths(args, zone_id=zone_id, output_dir=output_dir)
    stats = result.get("statistics") if isinstance(result.get("statistics"), dict) else {}
    metadata = result.get("metadata") if isinstance(result.get("metadata"), dict) else {}
    output = result.get("output")
    stage = {
        "output": output,
        "file_size_bytes": _local_file_size(output if isinstance(output, str) else None),
        "n_era5_days": stats.get("n_era5_days"),
        "n_ndvi_observations": stats.get("n_ndvi_observations"),
        "n_matched": stats.get("n_matched"),
        "images_searched": stats.get("images_searched"),
        "ndvi_truncation_warning": metadata.get("ndvi_truncation_warning"),
        "ndvi_input": metadata.get("ndvi_input"),
        "ndvi_coverage": metadata.get("ndvi_coverage"),
        "note": metadata.get("note"),
    }
    return {
        "ok": "error" not in result,
        "zone_id": zone_id,
        "window": {"start_date": args.start_date, "end_date": args.end_date},
        "storage_mode": args.storage_mode,
        "preflight_only": False,
        "tolerance_days": args.tolerance_days,
        "cloud_cover_max": args.cloud_cover_max,
        "era5_source": paths["era5_source"],
        "ndvi_source": paths["ndvi_source"],
        "unified_output": paths["unified_output"],
        "duration_ms": round((time.monotonic() - started_at) * 1000),
        "failed_stage": "climate_enrich_ndvi" if "error" in result else None,
        "error": result.get("error"),
        "stage": stage,
    }


def _panel_payload(
    *,
    args: argparse.Namespace,
    output_dir: Path,
    started_at: float,
    created_at: str,
    run_id: str,
    run_manifest_uri: str | None,
    code_ref: dict[str, Any],
    zone_artifact: dict[str, Any],
    zone_ids: list[str],
    zones: list[dict[str, Any]],
    run_manifest_errors: list[dict[str, Any]],
    run_manifest_written: bool | None,
    stop_reason: str | None = None,
) -> dict[str, Any]:
    failed = [z["zone_id"] for z in zones if not z["ok"]]
    zone_ok = not failed and len(zones) == len(zone_ids)
    manifest_ok = args.preflight_only or run_manifest_uri is None or run_manifest_written is True
    return {
        "schema_version": 1,
        "kind": "unified_pilot_panel_summary",
        "ok": zone_ok and manifest_ok,
        "zone_ok": zone_ok,
        "run_id": run_id,
        "created_at": created_at,
        "window": {"start_date": args.start_date, "end_date": args.end_date},
        "zone_set": args.zone_set or ("pilot-3" if not args.zone_id else "explicit"),
        "zone_ids": zone_ids,
        "completed_zone_ids": [z["zone_id"] for z in zones],
        "failed_zone_ids": failed,
        "n_zones_requested": len(zone_ids),
        "n_zones_completed": len(zones),
        "storage_mode": args.storage_mode,
        "artifact_scope": args.artifact_scope,
        "artifact_window_label": _artifact_window_label(args),
        "preflight_only": args.preflight_only,
        "tolerance_days": args.tolerance_days,
        "cloud_cover_max": args.cloud_cover_max,
        "ndvi_source_template": args.ndvi_source_template,
        "zone_delay_s": args.zone_delay_s,
        "stop_reason": stop_reason,
        "output_dir": output_dir.as_posix(),
        "summary_path": (output_dir / args.summary_name).as_posix(),
        "manifest_path": (output_dir / args.manifest_name).as_posix(),
        "run_manifest_uri": run_manifest_uri,
        "run_manifest_written": run_manifest_written if run_manifest_uri else None,
        "run_manifest_errors": list(run_manifest_errors),
        "code_ref": code_ref,
        "zone_artifact": zone_artifact,
        "duration_ms": round((time.monotonic() - started_at) * 1000),
        "zones": zones,
    }


def _refresh_panel_ok(panel: dict[str, Any]) -> None:
    failed = panel.get("failed_zone_ids")
    zone_ok = not failed and panel.get("n_zones_completed") == panel.get("n_zones_requested")
    manifest_ok = not panel.get("run_manifest_uri") or panel.get("run_manifest_written") is True
    panel["zone_ok"] = zone_ok
    panel["ok"] = zone_ok and manifest_ok


def _manifest_payload(panel: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "kind": "unified_pilot_panel_manifest",
        "run_id": panel.get("run_id"),
        "created_at": panel.get("created_at"),
        "ok": panel.get("ok"),
        "zone_ok": panel.get("zone_ok"),
        "storage_mode": panel.get("storage_mode"),
        "artifact_scope": panel.get("artifact_scope"),
        "artifact_window_label": panel.get("artifact_window_label"),
        "window": panel.get("window"),
        "zone_set": panel.get("zone_set"),
        "zone_ids": panel.get("zone_ids"),
        "n_zones_requested": panel.get("n_zones_requested"),
        "n_zones_completed": panel.get("n_zones_completed"),
        "failed_zone_ids": panel.get("failed_zone_ids"),
        "tolerance_days": panel.get("tolerance_days"),
        "cloud_cover_max": panel.get("cloud_cover_max"),
        "run_manifest_uri": panel.get("run_manifest_uri"),
        "run_manifest_written": panel.get("run_manifest_written"),
        "run_manifest_errors": panel.get("run_manifest_errors") or [],
        "code_ref": panel.get("code_ref"),
        "zone_artifact": panel.get("zone_artifact"),
        "artifacts": [
            {
                "zone_id": zone.get("zone_id"),
                "ok": zone.get("ok"),
                "preflight_only": zone.get("preflight_only"),
                "summary_path": zone.get("summary_path"),
                "era5_source": zone.get("era5_source"),
                "ndvi_source": zone.get("ndvi_source"),
                "unified_output": zone.get("unified_output"),
                "n_matched": (zone.get("stage") or {}).get("n_matched"),
                "ndvi_truncation_warning": (zone.get("stage") or {}).get("ndvi_truncation_warning"),
            }
            for zone in panel.get("zones", [])
        ],
    }


def _run_manifest_outputs(panel: dict[str, Any]) -> list[dict[str, Any]]:
    outputs: list[dict[str, Any]] = []
    for zone in panel.get("zones", []):
        if not isinstance(zone, dict) or not zone.get("ok") or zone.get("preflight_only"):
            continue
        stage = zone.get("stage") if isinstance(zone.get("stage"), dict) else {}
        uri = stage.get("output")
        if uri:
            outputs.append(
                {
                    "kind": "unified",
                    "zone_id": zone.get("zone_id"),
                    "uri": uri,
                    "n_matched": stage.get("n_matched"),
                }
            )
    return outputs


def _run_manifest_errors(panel: dict[str, Any]) -> list[dict[str, Any]]:
    errors: list[dict[str, Any]] = []
    for zone in panel.get("zones", []):
        if not isinstance(zone, dict) or not zone.get("error"):
            continue
        errors.append(
            {
                "zone_id": zone.get("zone_id"),
                "failed_stage": zone.get("failed_stage"),
                "error": zone.get("error"),
            }
        )
    return errors


def _run_manifest_payload(panel: dict[str, Any]) -> dict[str, Any]:
    window = panel.get("window") if isinstance(panel.get("window"), dict) else {}
    # A list, not the tuple `sanitize_stac_url` returns: this payload goes through `json.dumps`
    # and a DuckDB `COPY ... (FORMAT JSON)`, and a tuple is not a JSON type.
    stac_address, stac_reductions = sanitize_stac_url(settings.STAC_URL)
    zones = [zone for zone in panel.get("zones", []) if isinstance(zone, dict)]
    if panel.get("ndvi_source_template") is not None:
        # The stored series WAS the input; advertising the STAC catalogue here would name a
        # provider the run never called.
        ndvi_input: dict[str, Any] = {
            "kind": "b3_ndvi_parquet",
            "uris": [zone["ndvi_source"] for zone in zones if zone.get("ndvi_source")],
        }
    else:
        ndvi_input = {
            "kind": "sentinel2_stac_ndvi",
            # The STAC policy, not the object-store one: a catalogue root has a path and
            # the endpoint grammar refuses a path outright. `source_reductions` names the
            # components removed to get the published address — userinfo always, a query and
            # a fragment too — so a reader can tell "this IS the configured value" from
            # "this is a prefix of it". The boundary redactor only knows `S3_ENDPOINT`.
            "source": stac_address,
            "source_reductions": list(stac_reductions),
            "license": "source-specific STAC item license",
        }
    return {
        "schema_version": 1,
        "kind": "geoai_run_manifest",
        "run_id": panel.get("run_id"),
        "created_at": panel.get("created_at"),
        "tool_or_script": "scripts/unified_pilot_panel.py",
        "code_ref": panel.get("code_ref"),
        "zone_artifact": panel.get("zone_artifact"),
        "provenance": {
            "mode": "materialization",
            "artifact_code_ref": panel.get("code_ref"),
            "artifact_zone_artifact": panel.get("zone_artifact"),
        },
        "environment": climate_baseline_panel._run_manifest_environment(),
        "parameters": {
            "zone_set": panel.get("zone_set"),
            "zone_ids": panel.get("zone_ids"),
            "start_date": window.get("start_date"),
            "end_date": window.get("end_date"),
            "storage_mode": panel.get("storage_mode"),
            "artifact_scope": panel.get("artifact_scope"),
            "artifact_window_label": panel.get("artifact_window_label"),
            "tolerance_days": panel.get("tolerance_days"),
            "cloud_cover_max": panel.get("cloud_cover_max"),
            "ndvi_source_template": panel.get("ndvi_source_template"),
            "preflight_only": panel.get("preflight_only"),
        },
        "inputs": [
            {
                "kind": "era5_parquet",
                "uris": [
                    zone.get("era5_source")
                    for zone in panel.get("zones", [])
                    if isinstance(zone, dict) and zone.get("era5_source")
                ],
            },
            ndvi_input,
        ],
        "outputs": _run_manifest_outputs(panel),
        "checks": {
            "ok": panel.get("ok"),
            "zone_ok": panel.get("zone_ok"),
            "run_manifest_written": panel.get("run_manifest_written"),
            "run_manifest_error_count": len(panel.get("run_manifest_errors") or []),
            "n_zones_requested": panel.get("n_zones_requested"),
            "n_zones_completed": panel.get("n_zones_completed"),
            "failed_zone_ids": panel.get("failed_zone_ids"),
        },
        "run_manifest_errors": panel.get("run_manifest_errors") or [],
        "errors": _run_manifest_errors(panel),
    }


async def run_panel(
    args: argparse.Namespace,
    *,
    enrich_runner: EnrichRunner | None = None,
    sleeper: Sleeper = asyncio.sleep,
    run_manifest_writer: RunManifestWriter = _write_s3_json,
) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    started_at = time.monotonic()
    created_at = _utc_now()
    if date_error := _validate_date_range(args.start_date, args.end_date):
        raise ValueError(date_error)
    zone_ids = _resolve_zone_ids(args)
    run_id = _resolve_run_id(args, zone_ids=zone_ids, created_at=created_at)
    run_manifest_uri = _run_manifest_uri(args, run_id)
    code_ref = _code_ref()
    zone_artifact = _zone_artifact(zone_ids)
    zones: list[dict[str, Any]] = []
    run_manifest_errors: list[dict[str, Any]] = []
    stop_reason: str | None = None
    runner = enrich_runner or _load_enrich_tool()

    for index, zone_id in enumerate(zone_ids):
        zone_started = time.monotonic()
        summary_path = _zone_summary_path(
            output_dir, zone_id, args.start_date, args.end_date, index=index
        )

        if args.preflight_only:
            zone_payload = _preflight_zone_payload(
                args=args,
                zone_id=zone_id,
                output_dir=output_dir,
                started_at=zone_started,
            )
        elif error := _safe_zone_error(zone_id):
            zone_payload = _failed_zone_payload(
                args=args,
                zone_id=zone_id,
                output_dir=output_dir,
                started_at=zone_started,
                failed_stage="preflight",
                error=error,
            )
        else:
            paths = _resolve_artifact_paths(args, zone_id=zone_id, output_dir=output_dir)
            error = _path_validation_error(paths["era5_source"], label="era5_source")
            if error is None:
                error = _ndvi_source_error(args, paths)
            if error is None:
                error = _path_validation_error(paths["unified_output"], label="unified_output")
            if error is None:
                error = _artifact_window_validation_error(args)
            if error is not None:
                zone_payload = _failed_zone_payload(
                    args=args,
                    zone_id=zone_id,
                    output_dir=output_dir,
                    started_at=zone_started,
                    failed_stage="preflight",
                    error=error,
                )
            else:
                # Passed only on opt-in, so a caller that did not ask keeps the exact call
                # (and the STAC path) it had before.
                ndvi_kwargs = (
                    {"ndvi_source": paths["ndvi_source"]}
                    if args.ndvi_source_template is not None
                    else {}
                )
                result = await runner(
                    zone_id=zone_id,
                    start_date=args.start_date,
                    end_date=args.end_date,
                    tolerance_days=args.tolerance_days,
                    cloud_cover_max=args.cloud_cover_max,
                    era5_source=paths["era5_source"],
                    write_output=True,
                    output_path=paths["unified_output"],
                    **ndvi_kwargs,
                )
                zone_payload = _compact_zone_payload(
                    args=args,
                    zone_id=zone_id,
                    output_dir=output_dir,
                    started_at=zone_started,
                    result=result,
                )

        zone_payload["summary_path"] = summary_path.as_posix()
        _write_json(summary_path, zone_payload)
        zones.append(zone_payload)

        if not zone_payload["ok"] and args.fail_fast:
            stop_reason = "fail_fast"
            break
        if index < len(zone_ids) - 1 and args.zone_delay_s:
            await sleeper(args.zone_delay_s)

    panel = _panel_payload(
        args=args,
        output_dir=output_dir,
        started_at=started_at,
        created_at=created_at,
        run_id=run_id,
        run_manifest_uri=run_manifest_uri,
        code_ref=code_ref,
        zone_artifact=zone_artifact,
        zone_ids=zone_ids,
        zones=zones,
        run_manifest_errors=run_manifest_errors,
        run_manifest_written=None,
        stop_reason=stop_reason,
    )
    _write_json(output_dir / args.summary_name, panel)
    _write_json(output_dir / args.manifest_name, _manifest_payload(panel))

    if run_manifest_uri and not args.preflight_only:
        try:
            remote_panel = dict(panel)
            remote_panel["run_manifest_written"] = True
            _refresh_panel_ok(remote_panel)
            run_manifest_writer(run_manifest_uri, _run_manifest_payload(remote_panel))
            panel["run_manifest_written"] = True
        except Exception as exc:
            run_manifest_errors.append({"uri": run_manifest_uri, "error": str(exc)})
            panel["run_manifest_errors"] = run_manifest_errors
            panel["run_manifest_written"] = False
        _refresh_panel_ok(panel)
        _write_json(output_dir / args.summary_name, panel)
        _write_json(output_dir / args.manifest_name, _manifest_payload(panel))

    return panel


def main(argv: list[str] | None = None) -> int:
    # Install the credential-redaction processor (#148). Only src/server.py called this, so
    # every `log.error(..., error=str(e))` these handlers reach went to stderr raw when a
    # script drove them in-process -- structlog's DEFAULT chain has no redactor (#161 review).
    setup_logging()
    args = build_parser().parse_args(argv)
    try:
        panel = asyncio.run(run_panel(args))
    except ValueError as exc:
        # `redact_text`, not `redact_credentials`: the latter decides whether a string is prose
        # from the key it hangs under, and a bare exception message has no key — passing one
        # through it returns it untouched.
        print(f"ERROR: {redact_text(str(exc), settings.S3_ENDPOINT)}", file=sys.stderr)
        return 2

    print(
        f"OK={panel['ok']} zones={panel['n_zones_completed']}/{panel['n_zones_requested']} "
        f"summary={panel['summary_path']}"
    )
    if panel.get("failed_zone_ids"):
        print(f"failed_zone_ids={panel['failed_zone_ids']}", file=sys.stderr)
        return 1
    if panel.get("run_manifest_uri") and panel.get("run_manifest_written") is False:
        # `panel` is the IN-MEMORY dict, so it never passed through a JSON writer, and this
        # line is verbatim writer-exception text. Measured with the repo's dev credentials
        # before this call: `stderr LEAK[deposium-dev-secret] = True`. A CI log keeps that line
        # exactly as long as the file on disk keeps its copy.
        print(
            redact_text(
                f"run_manifest_errors={panel.get('run_manifest_errors')}", settings.S3_ENDPOINT
            ),
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
