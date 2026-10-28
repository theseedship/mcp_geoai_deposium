#!/usr/bin/env python3
"""Run the FIERCE climate baseline gate over multiple zones.

This is an operator wrapper around ``src.cli.climate_baseline.run_pipeline``.
It does not implement new climate logic; it only gives the one-zone gate a
repeatable panel runner with per-zone summaries and one aggregate JSON report.
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

# Allow `python scripts/climate_baseline_panel.py` from anywhere.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.cli.climate_baseline import (  # noqa: E402
    _expected_days,
    _percent_threshold,
    _positive_int,
    _preflight_output_destinations,
    _resolve_output_destinations,
    run_pipeline,
)
from src.config import settings  # noqa: E402
from src.tools.climate import (  # noqa: E402
    SCIENTIFIC_ZONE_ROLES,
    ZONE_ROLES,
    _resolve_zone,
    _validate_dst,
)
from src.utils.duckdb_conn import duckdb_session  # noqa: E402
from src.utils.logging import setup_logging  # noqa: E402
from src.utils.openmeteo import NORMALIZED_VARS  # noqa: E402
from src.utils.s3_endpoint import redact_credentials, sanitize_endpoint  # noqa: E402

PanelRunner = Callable[[argparse.Namespace], Awaitable[dict[str, Any]]]
Sleeper = Callable[[float], Awaitable[None]]
RunManifestWriter = Callable[[str, dict[str, Any]], None]
_SAFE_ZONE_RE = re.compile(r"^[A-Za-z0-9._-]+$")
_SAFE_RUN_ID_RE = re.compile(r"^[A-Za-z0-9._-]+$")
_SAFE_S3_JSON_URI_RE = re.compile(r"^[A-Za-z0-9._:/=-]+\.json$")
_SAFE_SQL_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
ZONE_SETS = ("full-panel",)


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
        prog="climate_baseline_panel",
        description=(
            "Run the ERA5 -> climatological baseline gate over several zones, "
            "writing one JSON summary per zone plus an aggregate panel summary."
        ),
    )
    zone_selection = p.add_mutually_exclusive_group(required=True)
    zone_selection.add_argument("--zone-id", nargs="+", help="one or more zone ids")
    zone_selection.add_argument(
        "--zone-set",
        choices=ZONE_SETS,
        help="predefined zone set; full-panel loads the 15-zone B1 artifact from zones/zones.geojson",
    )
    p.add_argument(
        "--start-date",
        default="1991-01-01",
        help="ISO date, inclusive; default is the WMO normal start",
    )
    p.add_argument(
        "--end-date",
        default="2020-12-31",
        help="ISO date, inclusive; default is the WMO normal end",
    )
    p.add_argument(
        "--variables",
        nargs="+",
        choices=list(NORMALIZED_VARS),
        help="normalized ERA5 variables; default is all six locked variables",
    )
    p.add_argument(
        "--output-dir",
        default="/tmp/geoai-baseline-panel",
        help="directory for JSON summaries and, in local storage mode, per-zone parquets",
    )
    p.add_argument(
        "--storage-mode",
        choices=("local", "s3"),
        default="local",
        help=(
            "parquet artifact destination: local writes under --output-dir; "
            "s3 uses s3://$S3_BUCKET/climate/{era5,normals}/{zone}.parquet"
        ),
    )
    p.add_argument(
        "--artifact-scope",
        choices=("baseline", "holdout"),
        default="baseline",
        help=(
            "baseline keeps canonical ERA5/normals outputs; holdout writes ERA5-only "
            "recent artifacts under climate/holdout/era5/{window_label}/{zone}.parquet"
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
        "--summary-name",
        default="panel-baseline-summary.json",
        help="aggregate JSON filename inside --output-dir",
    )
    p.add_argument(
        "--manifest-name",
        default="panel-baseline-manifest.json",
        help="operator artifact manifest JSON filename inside --output-dir",
    )
    p.add_argument(
        "--run-id",
        help=(
            "stable id for the production run manifest; default is generated from "
            "UTC start time, zone selection, and date window"
        ),
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="replace existing ERA5/normals outputs instead of using coverage-aware skips",
    )
    p.add_argument(
        "--min-completeness-pct",
        type=_percent_threshold,
        default=98.0,
        help="operator gate passed to each zone run; default 98",
    )
    p.add_argument(
        "--archive-chunk-days",
        type=_positive_int,
        help=(
            "operator override passed to each zone run; use 10958 to request one "
            "WMO 1991-2020 zone in a single Open-Meteo Archive call"
        ),
    )
    p.add_argument(
        "--fail-fast",
        action="store_true",
        help="stop after the first failed zone instead of collecting all zone results",
    )
    p.add_argument(
        "--preflight-only",
        action="store_true",
        help="validate zone selection and output paths without calling Open-Meteo or writing parquets",
    )
    p.add_argument(
        "--summarize-existing",
        action="store_true",
        help="aggregate existing per-zone JSON summaries from --output-dir without running climate tools",
    )
    p.add_argument(
        "--stop-on-rate-limit",
        action="store_true",
        help="stop cleanly after a 429/rate-limit zone failure instead of hammering later zones",
    )
    p.add_argument(
        "--zone-delay-s",
        type=_nonnegative_float,
        default=0.0,
        help="seconds to wait between zone runs; useful for Open-Meteo free-tier rate limits",
    )
    return p


def _flatten_zone_ids(values: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(values))


def _load_full_panel_zone_ids(path: Path | None = None) -> list[str]:
    """The B1 scientific zones (12 panel + 3 pilot) in file order; a client site never enters."""
    path = path or _REPO_ROOT / "zones" / "zones.geojson"
    try:
        payload = json.loads(path.read_text())
    except FileNotFoundError:
        raise RuntimeError(f"full-panel zone artifact not found: {path}") from None
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"full-panel zone artifact is invalid JSON: {path}: {exc}") from exc

    features = payload.get("features")
    if not isinstance(features, list):
        raise RuntimeError(f"full-panel zone artifact has no features list: {path}")

    all_zone_ids: list[str] = []
    zone_ids: list[str] = []
    roles: list[str] = []
    for index, feature in enumerate(features):
        properties = feature.get("properties") if isinstance(feature, dict) else None
        zone_id = properties.get("zone_id") if isinstance(properties, dict) else None
        if not isinstance(zone_id, str) or not zone_id:
            raise RuntimeError(f"full-panel zone artifact feature {index} has no zone_id")
        role = properties.get("role")
        if not isinstance(role, str) or role not in ZONE_ROLES:
            raise RuntimeError(
                f"full-panel zone artifact feature {zone_id!r} has role {role!r}; "
                f"expected one of {sorted(ZONE_ROLES)}"
            )
        all_zone_ids.append(zone_id)
        if role in SCIENTIFIC_ZONE_ROLES:
            zone_ids.append(zone_id)
            roles.append(role)

    # Across every role: a client reusing a panel id would share its climate/normals/ artifact.
    if len(_flatten_zone_ids(all_zone_ids)) != len(all_zone_ids):
        raise RuntimeError("full-panel zone artifact contains duplicate zone_id values")
    n_panel, n_pilot = roles.count("panel"), roles.count("pilot")
    if (n_panel, n_pilot) != (12, 3):
        raise RuntimeError(
            "full-panel zone artifact must contain 12 panel + 3 pilot zones, "
            f"got {n_panel} panel + {n_pilot} pilot"
        )
    return zone_ids


def _resolve_zone_ids(args: argparse.Namespace) -> list[str]:
    if args.zone_set == "full-panel":
        return _load_full_panel_zone_ids()
    return _flatten_zone_ids(args.zone_id)


def _safe_zone_error(zone_id: str) -> str | None:
    if not _SAFE_ZONE_RE.fullmatch(zone_id):
        return f"Unsafe zone_id {zone_id!r}; use only letters, digits, dot, underscore, or dash"
    return None


def _summary_zone_key(zone_id: str, index: int) -> str:
    return zone_id if _safe_zone_error(zone_id) is None else f"invalid-zone-{index + 1}"


def _window_slug(start_date: str, end_date: str) -> str:
    return f"{start_date}_{end_date}"


def _safe_path_label_error(label: str, *, flag: str = "--artifact-window-label") -> str | None:
    if not _SAFE_RUN_ID_RE.fullmatch(label):
        return f"{flag} must contain only letters, digits, dot, underscore, or dash"
    if ".." in label:
        return f"{flag} must not contain consecutive dots"
    return None


def _artifact_window_label(args: argparse.Namespace) -> str:
    if args.artifact_window_label:
        if error := _safe_path_label_error(args.artifact_window_label):
            raise ValueError(error)
        return args.artifact_window_label
    if args.start_date.endswith("-01-01") and args.end_date.endswith("-12-31"):
        return f"{args.start_date[:4]}_{args.end_date[:4]}"
    return _window_slug(args.start_date, args.end_date).replace("-", "_")


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _run_id_timestamp(created_at: str) -> str:
    return created_at.replace("-", "").replace(":", "").replace(".", "").replace("+00:00", "Z")


def _zone_selection_fingerprint(zone_ids: list[str]) -> str:
    payload = json.dumps(zone_ids, ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def _run_id_error(run_id: str) -> str | None:
    if not _SAFE_RUN_ID_RE.fullmatch(run_id):
        return "--run-id must contain only letters, digits, dot, underscore, or dash"
    if ".." in run_id:
        return "--run-id must not contain consecutive dots"
    return None


def _resolve_run_id(args: argparse.Namespace, *, zone_ids: list[str], created_at: str) -> str:
    if args.run_id:
        if error := _run_id_error(args.run_id):
            raise ValueError(error)
        return args.run_id
    zone_scope = args.zone_set or f"{len(zone_ids)}zones"
    zone_scope = f"{zone_scope}_{_zone_selection_fingerprint(zone_ids)}"
    return (
        f"{_run_id_timestamp(created_at)}_{zone_scope}_{args.start_date}_{args.end_date}"
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


def _run_manifest_uri(run_id: str) -> str | None:
    if not settings.S3_BUCKET:
        return None
    if _run_id_error(run_id):
        return None
    return f"s3://{settings.S3_BUCKET}/climate/runs/{run_id}/manifest.json"


def _holdout_era5_uri(*, zone_id: str, window_label: str) -> str | None:
    if not settings.S3_BUCKET:
        return None
    return f"s3://{settings.S3_BUCKET}/climate/holdout/era5/{window_label}/{zone_id}.parquet"


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

    The destination is interpolated into COPY because DuckDB does not accept it
    as a parameter. Validate the URI first and parameter-bind every JSON value.

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


def _local_file_size(path: str | None) -> int | None:
    if not path:
        return None
    if "://" in path:
        return None
    p = Path(path)
    if not p.exists() or not p.is_file():
        return None
    return p.stat().st_size


def _rate_limited(payload: dict[str, Any]) -> bool:
    error = payload.get("error")
    if not isinstance(error, str):
        return False
    lower = error.lower()
    return "429" in lower or "too many requests" in lower or "rate limit" in lower


def _zone_args(args: argparse.Namespace, *, zone_id: str, output_dir: Path) -> argparse.Namespace:
    slug = _window_slug(args.start_date, args.end_date)
    skip_normals = args.artifact_scope == "holdout"
    artifact_label = _artifact_window_label(args)
    local_outputs = args.storage_mode == "local"
    if local_outputs:
        era5_output = (output_dir / f"{zone_id}-era5-{slug}.parquet").as_posix()
        normals_output = (
            None if skip_normals else (output_dir / f"{zone_id}-normals-{slug}.parquet").as_posix()
        )
    elif skip_normals and settings.S3_BUCKET:
        era5_output = _holdout_era5_uri(zone_id=zone_id, window_label=artifact_label)
        normals_output = None
    else:
        era5_output = None
        normals_output = None
    return argparse.Namespace(
        zone_id=zone_id,
        start_date=args.start_date,
        end_date=args.end_date,
        variables=args.variables,
        era5_output=era5_output,
        normals_output=normals_output,
        skip_normals=skip_normals,
        overwrite=args.overwrite,
        dry_run=False,
        min_completeness_pct=args.min_completeness_pct,
        archive_chunk_days=args.archive_chunk_days,
        output=None,
        storage_mode=args.storage_mode,
    )


def _zone_summary_path(
    output_dir: Path,
    zone_id: str,
    start_date: str,
    end_date: str,
    *,
    index: int,
) -> Path:
    summary_zone_key = _summary_zone_key(zone_id, index)
    return output_dir / f"{summary_zone_key}-baseline-{_window_slug(start_date, end_date)}.json"


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


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _finite_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _artifact_storage_error(
    outputs: tuple[Any, ...],
    *,
    expected_storage_mode: str,
    artifact_base_dir: Path | None = None,
) -> str | None:
    if not outputs or not all(isinstance(output, str) and output for output in outputs):
        return "existing summary has no materialized artifact output"
    modes: list[str] = []
    for output in outputs:
        try:
            _validate_dst(output)
        except ValueError as exc:
            return f"existing artifact output {output!r} is not allowed: {exc}"
        modes.append("s3" if output.startswith("s3://") else "local")
    if len(set(modes)) != 1:
        return f"existing summary mixes artifact storage modes {modes!r}"
    artifact_storage_mode = modes[0]
    if artifact_storage_mode != expected_storage_mode:
        return (
            f"existing summary artifact storage mode {artifact_storage_mode!r} "
            f"does not match requested {expected_storage_mode!r}"
        )
    if artifact_storage_mode == "local":
        for output in outputs:
            output_path = _local_artifact_path(
                output,
                base_dir=artifact_base_dir,
            )
            if not output_path.is_file():
                return f"existing local artifact output {output!r} does not exist"
    return None


def _failed_zone_payload(
    *,
    args: argparse.Namespace,
    zone_id: str,
    started_at: float,
    failed_stage: str,
    error: str,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    requested_outputs = {"era5": None, "normals": None}
    resolved_outputs = {"era5": None, "normals": None}
    if output_dir is not None and _safe_zone_error(zone_id) is None and _resolve_zone(zone_id):
        zone_args = _zone_args(args, zone_id=zone_id, output_dir=output_dir)
        requested_outputs = {
            "era5": zone_args.era5_output,
            "normals": zone_args.normals_output,
        }
        resolved_outputs = _resolve_output_destinations(zone_args)
    return {
        "ok": False,
        "zone_id": zone_id,
        "window": {
            "start_date": args.start_date,
            "end_date": args.end_date,
            "label": "WMO_1991_2020"
            if args.start_date == "1991-01-01" and args.end_date == "2020-12-31"
            else "custom",
        },
        "variables": args.variables or list(NORMALIZED_VARS),
        "write_output": True,
        "dry_run": False,
        "skip_normals": args.artifact_scope == "holdout",
        "archive_chunk_days": args.archive_chunk_days,
        "storage_mode": args.storage_mode,
        "requested_outputs": requested_outputs,
        "resolved_outputs": resolved_outputs,
        "duration_ms": round((time.monotonic() - started_at) * 1000),
        "failed_stage": failed_stage,
        "error": error,
        "stages": {},
        "notes": [],
    }


def _preflight_zone_payload(
    *,
    args: argparse.Namespace,
    zone_id: str,
    started_at: float,
    output_dir: Path,
) -> dict[str, Any]:
    zone_error = _safe_zone_error(zone_id)
    requested_outputs = {"era5": None, "normals": None}
    resolved_outputs = {"era5": None, "normals": None}
    if zone_error is None and _resolve_zone(zone_id) is None:
        zone_error = f"Unknown zone_id {zone_id!r} (not in zones/zones.geojson)"
    if not zone_error:
        zone_args = _zone_args(args, zone_id=zone_id, output_dir=output_dir)
        requested_outputs = {
            "era5": zone_args.era5_output,
            "normals": zone_args.normals_output,
        }
        resolved_outputs = _resolve_output_destinations(zone_args)
        zone_error = _preflight_output_destinations(zone_args, write_output=True)
    return {
        "ok": zone_error is None,
        "zone_id": zone_id,
        "window": {
            "start_date": args.start_date,
            "end_date": args.end_date,
            "label": "WMO_1991_2020"
            if args.start_date == "1991-01-01" and args.end_date == "2020-12-31"
            else "custom",
        },
        "variables": args.variables or list(NORMALIZED_VARS),
        "write_output": False,
        "dry_run": False,
        "preflight_only": True,
        "skip_normals": args.artifact_scope == "holdout",
        "archive_chunk_days": args.archive_chunk_days,
        "storage_mode": args.storage_mode,
        "requested_outputs": requested_outputs,
        "resolved_outputs": resolved_outputs,
        "duration_ms": round((time.monotonic() - started_at) * 1000),
        "failed_stage": "preflight" if zone_error else None,
        "error": zone_error,
        "stages": {},
        "notes": ["preflight-only: Open-Meteo, DuckDB, and parquet writes were not called"],
    }


def _existing_summary_error(
    payload: dict[str, Any],
    *,
    expected_zone_id: str,
    expected_start_date: str,
    expected_end_date: str,
    expected_variables: list[str] | None = None,
    expected_storage_mode: str = "local",
    expected_min_completeness_pct: float | None = None,
    expected_skip_normals: bool = False,
    expected_artifact_window_label: str | None = None,
    artifact_base_dir: Path | None = None,
) -> str | None:
    if payload.get("preflight_only"):
        return "existing summary is preflight-only and has no materialized climate stages"
    if payload.get("zone_id") != expected_zone_id:
        return (
            f"existing summary zone_id {payload.get('zone_id')!r} "
            f"does not match requested {expected_zone_id!r}"
        )

    stages = payload.get("stages") if isinstance(payload.get("stages"), dict) else {}
    era5 = stages.get("era5_ingest") if isinstance(stages.get("era5_ingest"), dict) else None
    normals = (
        stages.get("climate_percentiles")
        if isinstance(stages.get("climate_percentiles"), dict)
        else None
    )
    if not payload.get("ok"):
        return None
    if era5 is None or (not expected_skip_normals and normals is None):
        expected_stages = (
            "era5_ingest stage"
            if expected_skip_normals
            else ("era5_ingest and climate_percentiles stages")
        )
        return f"existing successful summary has no {expected_stages}"
    if expected_skip_normals:
        if payload.get("skip_normals") is not True:
            return "existing holdout summary must have skip_normals true"
        if normals is not None:
            return "existing holdout summary must not have climate_percentiles stage"
        for field in ("requested_outputs", "resolved_outputs"):
            outputs = payload.get(field)
            if isinstance(outputs, dict) and outputs.get("normals"):
                return f"existing holdout summary {field}.normals must be null"
    window = payload.get("window") if isinstance(payload.get("window"), dict) else {}
    if (
        window.get("start_date") != expected_start_date
        or window.get("end_date") != expected_end_date
    ):
        return (
            "existing successful summary window "
            f"{window.get('start_date')!r}..{window.get('end_date')!r} "
            f"does not match requested {expected_start_date!r}..{expected_end_date!r}"
        )
    if era5.get("start_date") != expected_start_date or era5.get("end_date") != expected_end_date:
        return (
            "existing successful summary era5 window "
            f"{era5.get('start_date')!r}..{era5.get('end_date')!r} "
            f"does not match requested {expected_start_date!r}..{expected_end_date!r}"
        )
    variables = _payload_variables(payload)
    if variables is None:
        return "existing successful summary has no valid variables list"
    if expected_variables is not None and variables != expected_variables:
        return (
            f"existing summary variables {variables!r} do not match expected {expected_variables!r}"
        )
    # Summarized runs may publish a production handoff manifest. Treat the
    # materialized artifact URIs as canonical so legacy summaries without a
    # storage_mode field, or explicit s3:// one-zone runs, cannot be relabeled.
    artifact_error = _artifact_storage_error(
        (era5.get("output"),)
        if expected_skip_normals
        else (era5.get("output"), normals.get("output")),
        expected_storage_mode=expected_storage_mode,
        artifact_base_dir=artifact_base_dir,
    )
    if artifact_error:
        return artifact_error
    if (
        expected_skip_normals
        and expected_storage_mode == "s3"
        and expected_artifact_window_label is not None
    ):
        expected_era5_output = _holdout_era5_uri(
            zone_id=expected_zone_id,
            window_label=expected_artifact_window_label,
        )
        if era5.get("output") != expected_era5_output:
            return (
                "existing holdout summary era5 output "
                f"{era5.get('output')!r} does not match expected {expected_era5_output!r}"
            )
    gates = payload.get("gates") if isinstance(payload.get("gates"), dict) else {}
    completeness = gates.get("completeness") if isinstance(gates.get("completeness"), dict) else {}
    if completeness.get("passed") is not True:
        return "existing successful summary has no passed completeness gate"
    if expected_min_completeness_pct is not None:
        recorded_threshold = _finite_float(completeness.get("min_completeness_pct"))
        if recorded_threshold is None:
            return "existing successful summary has no recorded completeness threshold"
        if recorded_threshold < expected_min_completeness_pct:
            return (
                "existing successful summary min_completeness_pct "
                f"{recorded_threshold} < requested {expected_min_completeness_pct}"
            )

    required_era5 = ("output", "n_days", "pct_days", "min_variable_pct_complete")
    missing_era5 = [field for field in required_era5 if era5.get(field) is None]
    required_normals = ("output", "n_normals", "n_months", "n_variables")
    missing_normals = (
        []
        if expected_skip_normals
        else [field for field in required_normals if normals.get(field) is None]
    )
    if missing_era5 or missing_normals:
        return (
            "existing successful summary is missing materialized metrics: "
            f"era5={missing_era5}, normals={missing_normals}"
        )

    expected_days = _expected_days(expected_start_date, expected_end_date)
    n_days = _finite_float(era5.get("n_days"))
    if n_days is None:
        return "existing successful summary has no finite era5 n_days"
    if int(n_days) < expected_days:
        return f"existing successful summary era5 n_days {era5.get('n_days')} < expected {expected_days}"

    threshold_for_metrics = expected_min_completeness_pct
    if threshold_for_metrics is None:
        threshold_for_metrics = _finite_float(completeness.get("min_completeness_pct"))
    if threshold_for_metrics is not None:
        for field in ("pct_days", "min_variable_pct_complete"):
            metric = _finite_float(era5.get(field))
            if metric is None:
                return f"existing successful summary has no finite era5 {field}"
            if metric < threshold_for_metrics:
                return (
                    f"existing successful summary era5 {field} "
                    f"{metric} < completeness threshold {threshold_for_metrics}"
                )

    if not expected_skip_normals:
        n_months = _finite_float(normals.get("n_months"))
        n_variables = _finite_float(normals.get("n_variables"))
        n_normals = _finite_float(normals.get("n_normals"))
        expected_n_variables = len(variables)
        expected_n_normals = 12 * expected_n_variables
        if n_months != 12.0:
            return f"existing successful summary normals n_months {n_months} != 12"
        if n_variables != float(expected_n_variables):
            return (
                "existing successful summary normals n_variables "
                f"{n_variables} != variables count {expected_n_variables}"
            )
        if n_normals != float(expected_n_normals):
            return (
                "existing successful summary normals n_normals "
                f"{n_normals} != expected {expected_n_normals}"
            )
    return None


def _payload_variables(payload: dict[str, Any]) -> list[str] | None:
    variables = payload.get("variables")
    if not isinstance(variables, list) or not variables:
        return None
    if not all(isinstance(variable, str) and variable for variable in variables):
        return None
    return variables


def _local_artifact_path(output: str, *, base_dir: Path | None) -> Path:
    output_path = Path(output)
    if output_path.is_absolute() or base_dir is None:
        return output_path
    base_dir = Path(base_dir)
    summary_path = base_dir / output_path
    if summary_path.is_file():
        return summary_path
    try:
        base_resolved = base_dir.resolve()
        output_resolved = output_path.resolve()
        if output_resolved.is_relative_to(base_resolved):
            expected_from_base = base_dir / output_resolved.relative_to(base_resolved)
            if output_path == expected_from_base or output_path.is_file():
                return output_path
    except RuntimeError:
        pass
    except ValueError:
        pass
    return summary_path


def _resolved_local_output(output: Any, *, base_dir: Path) -> Any:
    if not isinstance(output, str) or not output or "://" in output:
        return output
    output_path = _local_artifact_path(output, base_dir=base_dir)
    return output_path.resolve().as_posix()


def _resolved_local_outputs(outputs: Any, *, base_dir: Path) -> Any:
    if not isinstance(outputs, dict):
        return outputs
    resolved = dict(outputs)
    for kind in ("era5", "normals"):
        resolved[kind] = _resolved_local_output(resolved.get(kind), base_dir=base_dir)
    return resolved


def _compact_zone_entry(payload: dict[str, Any], *, summary_path: Path) -> dict[str, Any]:
    stages = payload.get("stages") if isinstance(payload.get("stages"), dict) else {}
    era5 = stages.get("era5_ingest") if isinstance(stages.get("era5_ingest"), dict) else {}
    normals = (
        stages.get("climate_percentiles")
        if isinstance(stages.get("climate_percentiles"), dict)
        else {}
    )
    gates = payload.get("gates") if isinstance(payload.get("gates"), dict) else {}
    completeness = (
        gates.get("completeness") if isinstance(gates.get("completeness"), dict) else None
    )

    era5_output = _resolved_local_output(era5.get("output"), base_dir=summary_path.parent)
    normals_output = _resolved_local_output(normals.get("output"), base_dir=summary_path.parent)

    entry: dict[str, Any] = {
        "zone_id": payload.get("zone_id"),
        "ok": bool(payload.get("ok")),
        "summary_path": summary_path.as_posix(),
        "duration_ms": payload.get("duration_ms"),
        "failed_stage": payload.get("failed_stage"),
        "error": payload.get("error"),
        "preflight_only": bool(payload.get("preflight_only")),
        "skip_normals": bool(payload.get("skip_normals")),
        "variables": _payload_variables(payload),
        "requested_outputs": payload.get("requested_outputs"),
        "resolved_outputs": _resolved_local_outputs(
            payload.get("resolved_outputs"), base_dir=summary_path.parent
        ),
        "gate_passed": completeness.get("passed") if completeness else None,
        "era5": {
            "output": era5_output,
            "n_days": era5.get("n_days"),
            "pct_days": era5.get("pct_days"),
            "min_variable_pct_complete": era5.get("min_variable_pct_complete"),
        },
        "normals": {
            "output": normals_output,
            "n_normals": normals.get("n_normals"),
            "n_months": normals.get("n_months"),
            "n_variables": normals.get("n_variables"),
            "source": normals.get("source"),
        },
    }
    entry["era5"]["file_size_bytes"] = _local_file_size(era5_output)
    entry["normals"]["file_size_bytes"] = _local_file_size(normals_output)
    return entry


def _panel_variables(args: argparse.Namespace, zones: list[dict[str, Any]]) -> list[str]:
    if args.summarize_existing:
        for zone in zones:
            variables = zone.get("variables")
            if (
                zone.get("ok") is True
                and isinstance(variables, list)
                and all(isinstance(variable, str) for variable in variables)
            ):
                return variables
    return list(dict.fromkeys(args.variables)) if args.variables else list(NORMALIZED_VARS)


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
    manifest_ok = run_manifest_uri is None or run_manifest_written is True
    return {
        "ok": zone_ok and manifest_ok,
        "zone_ok": zone_ok,
        "run_id": run_id,
        "created_at": created_at,
        "window": {
            "start_date": args.start_date,
            "end_date": args.end_date,
            "label": "WMO_1991_2020"
            if args.start_date == "1991-01-01" and args.end_date == "2020-12-31"
            else "custom",
        },
        "zone_ids": zone_ids,
        "variables": _panel_variables(args, zones),
        "completed_zone_ids": [z["zone_id"] for z in zones],
        "failed_zone_ids": failed,
        "n_zones_requested": len(zone_ids),
        "n_zones_completed": len(zones),
        "min_completeness_pct": args.min_completeness_pct,
        "archive_chunk_days": args.archive_chunk_days,
        "zone_delay_s": args.zone_delay_s,
        "storage_mode": args.storage_mode,
        "artifact_scope": args.artifact_scope,
        "artifact_window_label": _artifact_window_label(args),
        "skip_normals": args.artifact_scope == "holdout",
        "zone_set": args.zone_set,
        "overwrite": args.overwrite,
        "preflight_only": args.preflight_only,
        "summarize_existing": args.summarize_existing,
        "stop_on_rate_limit": args.stop_on_rate_limit,
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


def _artifact_output(zone: dict[str, Any], kind: str) -> str | None:
    if zone.get("preflight_only"):
        return None
    if kind == "normals" and zone.get("skip_normals"):
        return None
    stage = zone.get(kind) if isinstance(zone.get(kind), dict) else {}
    return stage.get("output")


def _manifest_planned_outputs(zone: dict[str, Any]) -> dict[str, Any] | None:
    resolved = zone.get("resolved_outputs")
    if not isinstance(resolved, dict):
        return None
    if zone.get("preflight_only"):
        return resolved
    if not zone.get("ok") and any(resolved.get(kind) for kind in ("era5", "normals")):
        return resolved
    return None


def _manifest_payload(panel: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "kind": "climate_baseline_panel_manifest",
        "run_id": panel.get("run_id"),
        "created_at": panel.get("created_at"),
        "ok": panel.get("ok"),
        "zone_ok": panel.get("zone_ok"),
        "storage_mode": panel.get("storage_mode"),
        "artifact_scope": panel.get("artifact_scope"),
        "artifact_window_label": panel.get("artifact_window_label"),
        "skip_normals": panel.get("skip_normals"),
        "window": panel.get("window"),
        "zone_set": panel.get("zone_set"),
        "zone_ids": panel.get("zone_ids"),
        "variables": panel.get("variables"),
        "n_zones_requested": panel.get("n_zones_requested"),
        "n_zones_completed": panel.get("n_zones_completed"),
        "failed_zone_ids": panel.get("failed_zone_ids"),
        "min_completeness_pct": panel.get("min_completeness_pct"),
        "archive_chunk_days": panel.get("archive_chunk_days"),
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
                "gate_passed": zone.get("gate_passed"),
                "summary_path": zone.get("summary_path"),
                "planned_outputs": _manifest_planned_outputs(zone),
                "era5_output": _artifact_output(zone, "era5"),
                "normals_output": _artifact_output(zone, "normals"),
            }
            for zone in panel.get("zones", [])
        ],
    }


def _run_manifest_outputs(panel: dict[str, Any]) -> list[dict[str, Any]]:
    outputs: list[dict[str, Any]] = []
    kinds = ("era5",) if panel.get("skip_normals") else ("era5", "normals")
    for zone in panel.get("zones", []):
        if not isinstance(zone, dict):
            continue
        # Failed zones may carry partial stage outputs for diagnostics/resume,
        # but production handoff manifests must only advertise consumable zones.
        if not zone.get("ok"):
            continue
        for kind in kinds:
            stage = zone.get(kind) if isinstance(zone.get(kind), dict) else {}
            uri = stage.get("output")
            if not uri:
                continue
            outputs.append(
                {
                    "kind": kind,
                    "zone_id": zone.get("zone_id"),
                    "uri": uri,
                }
            )
    return outputs


def _run_manifest_planned_outputs(panel: dict[str, Any]) -> list[dict[str, Any]]:
    planned: list[dict[str, Any]] = []
    kinds = ("era5",) if panel.get("skip_normals") else ("era5", "normals")
    for zone in panel.get("zones", []):
        if not isinstance(zone, dict):
            continue
        resolved = zone.get("resolved_outputs")
        if not isinstance(resolved, dict):
            continue
        for kind in kinds:
            uri = resolved.get(kind)
            if uri:
                planned.append(
                    {
                        "kind": kind,
                        "zone_id": zone.get("zone_id"),
                        "uri": uri,
                    }
                )
    return planned


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


def _run_manifest_environment() -> dict[str, Any]:
    """Record WHICH object store a run wrote to — endpoint, bucket, addressing style.

    Manifest URIs are bucket-relative, so two different RustFS instances produce byte-identical
    ``s3://`` paths and nothing in the artifact says which one holds the objects (issue #27: one
    reviewer found nothing where the author found everything). ``S3_ACCESS_KEY``/``S3_SECRET_KEY``
    are deliberately never read here. An unset endpoint is recorded as ``null`` — "no explicit
    endpoint configured" — rather than an empty string, which would read like a real address.

    The endpoint itself may carry credentials: ``http://user:password@host:9000`` is a valid
    ``S3_ENDPOINT``, and this manifest is published for downstream handoff and packed verbatim into
    export bundles. So the userinfo is stripped through the shared
    ``src.utils.s3_endpoint.sanitize_endpoint`` — the same parser the archive CLI redacts with —
    and its removal is stated in ``s3_endpoint_credentials_removed`` rather than done silently, so
    an operator can see that the recorded address is shorter than the configured one.

    Shared by every panel so a consumer can read it without knowing which script wrote the
    manifest.
    """
    endpoint, credentials_removed = sanitize_endpoint(settings.S3_ENDPOINT)
    return {
        "s3_endpoint": endpoint,
        "s3_endpoint_credentials_removed": credentials_removed,
        "s3_bucket": settings.S3_BUCKET or None,
        "s3_force_path_style": settings.S3_FORCE_PATH_STYLE,
    }


def _run_manifest_provenance(panel: dict[str, Any]) -> dict[str, Any]:
    if panel.get("summarize_existing"):
        return {
            "mode": "aggregation_only",
            "artifact_code_ref": None,
            "artifact_zone_artifact": None,
            "aggregation_code_ref": panel.get("code_ref"),
            "aggregation_zone_artifact": panel.get("zone_artifact"),
            "note": (
                "summarize_existing reuses already materialized zone artifacts; "
                "the current checkout only produced the aggregate manifests"
            ),
        }
    return {
        "mode": "materialization",
        "artifact_code_ref": panel.get("code_ref"),
        "artifact_zone_artifact": panel.get("zone_artifact"),
    }


def _run_manifest_payload(panel: dict[str, Any]) -> dict[str, Any]:
    window = panel.get("window") if isinstance(panel.get("window"), dict) else {}
    variables = panel.get("variables") or list(NORMALIZED_VARS)
    summarize_existing = bool(panel.get("summarize_existing"))
    return {
        "schema_version": 1,
        "kind": "geoai_run_manifest",
        "run_id": panel.get("run_id"),
        "created_at": panel.get("created_at"),
        "tool_or_script": "scripts/climate_baseline_panel.py",
        "code_ref": None if summarize_existing else panel.get("code_ref"),
        "zone_artifact": None if summarize_existing else panel.get("zone_artifact"),
        "provenance": _run_manifest_provenance(panel),
        "environment": _run_manifest_environment(),
        "parameters": {
            "zone_set": panel.get("zone_set") or "explicit",
            "zone_ids": panel.get("zone_ids"),
            "start_date": window.get("start_date"),
            "end_date": window.get("end_date"),
            "window_label": window.get("label"),
            "variables": variables,
            "storage_mode": panel.get("storage_mode"),
            "artifact_scope": panel.get("artifact_scope"),
            "artifact_window_label": panel.get("artifact_window_label"),
            "skip_normals": panel.get("skip_normals"),
            "overwrite": panel.get("overwrite"),
            "min_completeness_pct": panel.get("min_completeness_pct"),
            "archive_chunk_days": panel.get("archive_chunk_days"),
            "zone_delay_s": panel.get("zone_delay_s"),
            "preflight_only": panel.get("preflight_only"),
            "summarize_existing": panel.get("summarize_existing"),
            "stop_on_rate_limit": panel.get("stop_on_rate_limit"),
        },
        "inputs": [
            {
                "kind": "openmeteo_archive",
                "license": "CC-BY 4.0 (Open-Meteo)",
                "variables": variables,
            }
        ],
        "outputs": _run_manifest_outputs(panel),
        "planned_outputs": _run_manifest_planned_outputs(panel),
        "checks": {
            "ok": panel.get("ok"),
            "zone_ok": panel.get("zone_ok"),
            "run_manifest_written": panel.get("run_manifest_written"),
            "run_manifest_error_count": len(panel.get("run_manifest_errors") or []),
            "n_zones_requested": panel.get("n_zones_requested"),
            "n_zones_completed": panel.get("n_zones_completed"),
            "failed_zone_ids": panel.get("failed_zone_ids"),
            "min_completeness_pct": panel.get("min_completeness_pct"),
            "stop_reason": panel.get("stop_reason"),
        },
        "local_artifacts": {
            "summary_path": panel.get("summary_path"),
            "manifest_path": panel.get("manifest_path"),
        },
        "run_manifest_errors": panel.get("run_manifest_errors") or [],
        "errors": _run_manifest_errors(panel),
    }


def _run_manifest_write_error(uri: str, exc: Exception) -> dict[str, Any]:
    return {
        "stage": "run_manifest_write",
        "uri": uri,
        "created_at": _utc_now(),
        "error": f"{type(exc).__name__}: {exc}",
    }


def _write_panel_outputs(
    *,
    summary_path: Path,
    manifest_path: Path,
    panel: dict[str, Any],
    run_manifest_writer: RunManifestWriter,
    write_run_manifest: bool = True,
) -> None:
    _write_json(summary_path, panel)
    _write_json(manifest_path, _manifest_payload(panel))
    run_manifest_uri = panel.get("run_manifest_uri")
    if write_run_manifest and isinstance(run_manifest_uri, str) and run_manifest_uri:
        remote_panel = dict(panel)
        remote_panel["run_manifest_written"] = True
        _refresh_panel_ok(remote_panel)
        remote_payload = _run_manifest_payload(remote_panel)
        try:
            run_manifest_writer(run_manifest_uri, remote_payload)
        except Exception as exc:
            error = _run_manifest_write_error(run_manifest_uri, exc)
            errors = panel.setdefault("run_manifest_errors", [])
            if isinstance(errors, list):
                errors.append(error)
            else:
                panel["run_manifest_errors"] = [error]
            panel["run_manifest_written"] = False
            _refresh_panel_ok(panel)
        else:
            panel["run_manifest_written"] = True
            _refresh_panel_ok(panel)
        _write_json(summary_path, panel)
        _write_json(manifest_path, _manifest_payload(panel))


async def run_panel(
    args: argparse.Namespace,
    *,
    runner: PanelRunner = run_pipeline,
    sleeper: Sleeper = asyncio.sleep,
    run_manifest_writer: RunManifestWriter = _write_s3_json,
) -> dict[str, Any]:
    started_at = time.monotonic()
    created_at = _utc_now()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.preflight_only and args.summarize_existing:
        raise ValueError("--preflight-only and --summarize-existing cannot be combined")

    zone_ids = _resolve_zone_ids(args)
    run_id = _resolve_run_id(args, zone_ids=zone_ids, created_at=created_at)
    run_manifest_uri = (
        _run_manifest_uri(run_id) if args.storage_mode == "s3" and not args.preflight_only else None
    )
    code_ref = _code_ref()
    zone_artifact = _zone_artifact(zone_ids)
    zones: list[dict[str, Any]] = []
    run_manifest_errors: list[dict[str, Any]] = []
    run_manifest_written = True if run_manifest_uri is None else False
    expected_existing_variables = (
        list(dict.fromkeys(args.variables)) if args.summarize_existing and args.variables else None
    )
    stop_reason: str | None = None
    panel_summary_path = output_dir / args.summary_name
    panel_manifest_path = output_dir / args.manifest_name

    for index, zone_id in enumerate(zone_ids):
        zone_started_at = time.monotonic()
        zone_error = _safe_zone_error(zone_id)
        zone_summary_path = _zone_summary_path(
            output_dir,
            zone_id,
            args.start_date,
            args.end_date,
            index=index,
        )
        if args.summarize_existing:
            if zone_error is None and _resolve_zone(zone_id) is None:
                zone_error = f"Unknown zone_id {zone_id!r} (not in zones/zones.geojson)"
            if zone_error:
                payload = _failed_zone_payload(
                    args=args,
                    zone_id=zone_id,
                    started_at=zone_started_at,
                    failed_stage="existing_summary",
                    error=zone_error,
                    output_dir=output_dir,
                )
                payload["write_output"] = False
            else:
                try:
                    payload = _read_json(zone_summary_path)
                    existing_error = _existing_summary_error(
                        payload,
                        expected_zone_id=zone_id,
                        expected_start_date=args.start_date,
                        expected_end_date=args.end_date,
                        expected_variables=expected_existing_variables,
                        expected_storage_mode=args.storage_mode,
                        expected_min_completeness_pct=args.min_completeness_pct,
                        expected_skip_normals=args.artifact_scope == "holdout",
                        expected_artifact_window_label=_artifact_window_label(args),
                        artifact_base_dir=zone_summary_path.parent,
                    )
                    if (
                        existing_error is None
                        and expected_existing_variables is None
                        and payload.get("ok") is True
                    ):
                        expected_existing_variables = _payload_variables(payload)
                    if existing_error:
                        payload = _failed_zone_payload(
                            args=args,
                            zone_id=zone_id,
                            started_at=zone_started_at,
                            failed_stage="existing_summary",
                            error=f"Rejected existing zone summary {zone_summary_path}: {existing_error}",
                            output_dir=output_dir,
                        )
                        payload["write_output"] = False
                except Exception as exc:
                    payload = _failed_zone_payload(
                        args=args,
                        zone_id=zone_id,
                        started_at=zone_started_at,
                        failed_stage="existing_summary",
                        error=f"Could not read existing zone summary {zone_summary_path}: {exc}",
                        output_dir=output_dir,
                    )
                    payload["write_output"] = False
        elif args.preflight_only:
            payload = _preflight_zone_payload(
                args=args,
                zone_id=zone_id,
                started_at=zone_started_at,
                output_dir=output_dir,
            )
        elif zone_error:
            payload = _failed_zone_payload(
                args=args,
                zone_id=zone_id,
                started_at=zone_started_at,
                failed_stage="preflight",
                error=zone_error,
                output_dir=output_dir,
            )
        else:
            zone_args = _zone_args(args, zone_id=zone_id, output_dir=output_dir)
            try:
                payload = await runner(zone_args)
            except Exception as exc:
                payload = _failed_zone_payload(
                    args=args,
                    zone_id=zone_id,
                    started_at=zone_started_at,
                    failed_stage="exception",
                    error=f"{type(exc).__name__}: {exc}",
                    output_dir=output_dir,
                )

        if not args.summarize_existing:
            _write_json(zone_summary_path, payload)
        zones.append(_compact_zone_entry(payload, summary_path=zone_summary_path))
        if not args.summarize_existing and args.stop_on_rate_limit and _rate_limited(payload):
            stop_reason = "rate_limit"
            break
        incremental_panel = _panel_payload(
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
            run_manifest_written=run_manifest_written,
            stop_reason=stop_reason,
        )
        _write_panel_outputs(
            summary_path=panel_summary_path,
            manifest_path=panel_manifest_path,
            panel=incremental_panel,
            run_manifest_writer=run_manifest_writer,
            write_run_manifest=False,
        )
        run_manifest_errors = list(incremental_panel.get("run_manifest_errors") or [])
        if args.fail_fast and not payload.get("ok"):
            break
        if (
            not args.preflight_only
            and not args.summarize_existing
            and args.zone_delay_s
            and index < len(zone_ids) - 1
        ):
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
        run_manifest_written=run_manifest_written,
        stop_reason=stop_reason,
    )
    _write_panel_outputs(
        summary_path=panel_summary_path,
        manifest_path=panel_manifest_path,
        panel=panel,
        run_manifest_writer=run_manifest_writer,
    )
    return panel


def run(
    args: argparse.Namespace,
    *,
    runner: PanelRunner = run_pipeline,
    run_manifest_writer: RunManifestWriter = _write_s3_json,
) -> int:
    payload = asyncio.run(run_panel(args, runner=runner, run_manifest_writer=run_manifest_writer))
    summary_path = Path(args.output_dir) / args.summary_name
    manifest_path = Path(args.output_dir) / args.manifest_name
    print(f"Wrote climate baseline panel summary to {summary_path}")
    print(f"Wrote climate baseline panel manifest to {manifest_path}")
    run_manifest_uri = payload.get("run_manifest_uri")
    if run_manifest_uri and payload.get("run_manifest_written") is True:
        print(f"Wrote climate baseline run manifest to {run_manifest_uri}")
    elif run_manifest_uri and payload.get("run_manifest_written") is False:
        print(
            f"Could not write climate baseline run manifest to {run_manifest_uri}; "
            f"see {manifest_path}",
            file=sys.stderr,
        )
    return 0 if payload.get("ok") else 1


def main(argv: Iterable[str] | None = None) -> int:
    # Install the credential-redaction processor (#148). Only src/server.py called this, so
    # every `log.error(..., error=str(e))` these handlers reach went to stderr raw when a
    # script drove them in-process -- structlog's DEFAULT chain has no redactor (#161 review).
    setup_logging()
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
