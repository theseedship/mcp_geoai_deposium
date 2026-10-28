"""Operational AIFS forward-archive runner and health checker."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import duckdb

from src.cli.forecast_archive_panel import _read_json, _s3_uri, _write_json, run_panel
from src.config import settings
from src.utils.logging import setup_logging
from src.utils.openmeteo import NORMALIZED_VARS

PROVIDER = "aifs"
MODEL = "ecmwf_aifs025_single"
HORIZONS = (1, 3, 5, 7)
CANONICAL_ZONES = (
    "bassin_du_lez",
    "fos_sur_mer",
    "kerguelen",
    "manaus",
    "marseille",
    "mombasa",
    "montpellier",
    "montpellier_metropole",
    "nairobi",
    "niamey",
    "rennes",
    "riyadh",
    "seoul",
    "svalbard",
    "tromso",
)
EXPECTED_ARTIFACTS = len(CANONICAL_ZONES) * len(HORIZONS)
EXPECTED_ROWS = EXPECTED_ARTIFACTS * len(NORMALIZED_VARS)
SCHEMA_VERSION = "forecast-forward-archive/v1"


def _date_arg(value: str) -> str:
    try:
        return date.fromisoformat(value).isoformat()
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected an ISO date YYYY-MM-DD") from exc


def _run_id(valid_date: str) -> str:
    return f"aifs-forward-{valid_date.replace('-', '')}"


def _created_at(existing: dict[str, Any] | None) -> str:
    value = existing.get("created_at") if existing else None
    if isinstance(value, str) and value:
        return value
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def _bucket(args: argparse.Namespace) -> str:
    return getattr(args, "s3_bucket", None) or settings.S3_BUCKET


def _manifest_path(
    *,
    storage_mode: str,
    output_dir: Path,
    run_id: str,
    bucket: str,
) -> str | Path:
    if storage_mode == "s3":
        if not bucket:
            raise ValueError(
                "--s3-bucket or configured S3_BUCKET is required with --storage-mode s3"
            )
        return _s3_uri(bucket=bucket, prefix="climate/runs", run_id=run_id, name="manifest.json")
    return output_dir / "manifest.json"


def _artifact_path(
    *,
    storage_mode: str,
    output_dir: Path,
    run_id: str,
    bucket: str,
    s3_prefix: str,
    key: str,
) -> str | Path:
    if storage_mode == "s3":
        if not bucket:
            raise ValueError(
                "--s3-bucket or configured S3_BUCKET is required with --storage-mode s3"
            )
        return _s3_uri(bucket=bucket, prefix=s3_prefix, run_id=run_id, name=f"{key}.json")
    return (output_dir / f"{key}.json").resolve()


def _key(zone_id: str, valid_date: str, horizon: int) -> str:
    return f"{zone_id}__{PROVIDER}__{MODEL}__{valid_date}__{valid_date}__{horizon}"


def _missing_remote_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return "404" in message or "not found" in message or "no files found" in message


def _read_existing(path: str | Path) -> dict[str, Any] | None:
    try:
        payload = _read_json(path)
    except json.JSONDecodeError:
        return None
    except (OSError, RuntimeError, ValueError, duckdb.Error) as exc:
        if isinstance(exc, FileNotFoundError) or (
            isinstance(path, str) and _missing_remote_error(exc)
        ):
            return None
        raise
    return payload if isinstance(payload, dict) else None


def _digest(payload: dict[str, Any]) -> str:
    """Hash canonical JSON so local and S3 artifacts have the same digest contract."""
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _validate_artifact(
    *,
    payload: Any,
    zone_id: str,
    valid_date: str,
    horizon: int,
) -> tuple[dict[str, Any], list[str], bool, bool, bool, bool]:
    errors: list[str] = []
    coverage_available = True
    six_variables = True
    identity_consistent = True
    finite_values = True
    if not isinstance(payload, dict):
        return {}, ["artifact payload must be an object"], False, False, False, False
    coverage = payload.get("coverage")
    rows = payload.get("rows")
    if not isinstance(coverage, dict):
        errors.append("coverage must be an object")
        coverage = {}
    if not isinstance(rows, list):
        errors.append("rows must be an array")
        rows = []
    expected_coverage = {
        "zone_id": zone_id,
        "provider": PROVIDER,
        "model": MODEL,
        "horizon_days": horizon,
        "start_date": valid_date,
        "end_date": valid_date,
        "expected_days": 1,
        "available_days": 1,
        "status": "available",
    }
    for field, expected in expected_coverage.items():
        if coverage.get(field) != expected:
            errors.append(f"coverage {field} is inconsistent")
            coverage_available = False
    if len(rows) != len(NORMALIZED_VARS):
        errors.append("artifact must contain exactly six rows")
        six_variables = False
    variables: list[str] = []
    origin_date = (date.fromisoformat(valid_date) - timedelta(days=horizon)).isoformat()
    for row in rows:
        if not isinstance(row, dict):
            errors.append("row must be an object")
            finite_values = False
            continue
        variables.append(row.get("variable", ""))
        for field, expected in {
            "zone_id": zone_id,
            "provider": PROVIDER,
            "model": MODEL,
            "horizon_days": horizon,
            "valid_date": valid_date,
            "origin_date": origin_date,
        }.items():
            if row.get(field) != expected:
                errors.append(f"row {field} is inconsistent")
                identity_consistent = False
        value = row.get("value")
        try:
            value_is_finite = (
                not isinstance(value, bool)
                and isinstance(value, (int, float))
                and math.isfinite(value)
            )
        except OverflowError:
            value_is_finite = False
        if not value_is_finite:
            errors.append("row value must be a finite number")
            finite_values = False
    if set(variables) != set(NORMALIZED_VARS) or len(variables) != len(set(variables)):
        errors.append("artifact variables are not the six normalized variables")
        six_variables = False
    return (
        coverage,
        sorted(set(errors)),
        coverage_available,
        six_variables,
        identity_consistent,
        finite_values,
    )


def _enhanced_manifest(
    *,
    valid_date: str,
    run_id: str,
    storage_mode: str,
    output_dir: Path,
    bucket: str,
    s3_prefix: str,
    existing: dict[str, Any] | None,
) -> dict[str, Any]:
    artifacts: list[dict[str, Any]] = []
    errors: list[str] = []
    prior_digests: dict[str, str] = {}
    if (
        existing
        and existing.get("schema_version") == SCHEMA_VERSION
        and existing.get("run_id") == run_id
        and existing.get("valid_date") == valid_date
        and isinstance(existing.get("artifacts"), list)
    ):
        for item in existing["artifacts"]:
            if not isinstance(item, dict):
                continue
            key = item.get("key")
            digest = item.get("sha256")
            if (
                isinstance(key, str)
                and isinstance(digest, str)
                and len(digest) == 64
                and all(character in "0123456789abcdef" for character in digest)
            ):
                prior_digests[key] = digest
    total_rows = 0
    readable_count = 0
    valid_count = 0
    valid_rows = 0
    coverage_available = True
    six_variables = True
    identity_consistent = True
    finite_values = True
    for zone_id in CANONICAL_ZONES:
        for horizon in HORIZONS:
            key = _key(zone_id, valid_date, horizon)
            payload: Any = None
            artifact_rows = 0
            path = _artifact_path(
                storage_mode=storage_mode,
                output_dir=output_dir,
                run_id=run_id,
                bucket=bucket,
                s3_prefix=s3_prefix,
                key=key,
            )
            try:
                payload = _read_json(path)
                (
                    coverage,
                    artifact_errors,
                    artifact_coverage,
                    artifact_variables,
                    artifact_identity,
                    artifact_finite_values,
                ) = _validate_artifact(
                    payload=payload,
                    zone_id=zone_id,
                    valid_date=valid_date,
                    horizon=horizon,
                )
                rows = payload.get("rows") if isinstance(payload, dict) else None
                artifact_rows = len(rows) if isinstance(rows, list) else 0
                digest = _digest(payload) if isinstance(payload, dict) else ""
                prior_digest = prior_digests.get(key)
                if prior_digest and digest != prior_digest:
                    artifact_errors.append("artifact digest drifted from previous manifest")
                    digest = prior_digest
                readable_count += 1
                total_rows += artifact_rows
                if not artifact_errors:
                    valid_count += 1
                    valid_rows += len(NORMALIZED_VARS)
                coverage_available = coverage_available and artifact_coverage
                six_variables = six_variables and artifact_variables
                identity_consistent = identity_consistent and artifact_identity
                finite_values = finite_values and artifact_finite_values
            except (OSError, RuntimeError, ValueError, KeyError, TypeError, duckdb.Error):
                coverage = {}
                artifact_errors = ["artifact is missing or invalid JSON"]
                digest = ""
                coverage_available = False
                six_variables = False
                identity_consistent = False
                finite_values = False
            errors.extend(f"{key}: {error}" for error in artifact_errors)
            artifacts.append(
                {
                    "key": key,
                    "output": str(path),
                    "sha256": digest,
                    "coverage": coverage,
                    "rows": artifact_rows,
                }
            )
    artifact_count = valid_count
    checks = {
        "readable_artifacts": readable_count,
        "valid_artifacts": valid_count,
        "artifact_count": artifact_count,
        "expected_artifacts": EXPECTED_ARTIFACTS,
        "readable_rows": total_rows,
        "valid_rows": valid_rows,
        "row_count": valid_rows,
        "expected_rows": EXPECTED_ROWS,
        "six_variables_per_slot": six_variables,
        "coverage_available": coverage_available,
        "identity_consistent": identity_consistent,
        "finite_values": finite_values,
        "ok": not errors and artifact_count == EXPECTED_ARTIFACTS and valid_rows == EXPECTED_ROWS,
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "valid_date": valid_date,
        "storage_mode": storage_mode,
        "s3_prefix": s3_prefix if storage_mode == "s3" else None,
        "zones": list(CANONICAL_ZONES),
        "provider": PROVIDER,
        "model": MODEL,
        "horizons": list(HORIZONS),
        "variables": list(NORMALIZED_VARS),
        "created_at": _created_at(existing),
        "artifacts": artifacts,
        "counts": {
            "zones": len(CANONICAL_ZONES),
            "horizons": len(HORIZONS),
            "variables": len(NORMALIZED_VARS),
            "artifacts": artifact_count,
            "rows": valid_rows,
            "expected_artifacts": EXPECTED_ARTIFACTS,
            "readable_artifacts": readable_count,
            "valid_artifacts": valid_count,
            "expected_rows": EXPECTED_ROWS,
            "readable_rows": total_rows,
            "valid_rows": valid_rows,
        },
        "checks": checks,
        "errors": sorted(set(errors)),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="forecast-forward-archive")
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run")
    run.add_argument("--valid-date", "--date", dest="valid_date", type=_date_arg)
    run.add_argument("--output-dir", required=True, type=Path)
    run.add_argument("--storage-mode", choices=("local", "s3"), default="local")
    run.add_argument("--s3-bucket", default=None)
    run.add_argument("--s3-prefix", default="climate/forecast")
    health = subparsers.add_parser("health")
    health.add_argument("--end-date", required=True, type=_date_arg)
    health.add_argument(
        "--window-days",
        "--days",
        dest="window_days",
        required=True,
        type=int,
        choices=(7, 14),
    )
    health.add_argument("--output-dir", required=True, type=Path)
    health.add_argument("--storage-mode", choices=("local", "s3"), default="local")
    health.add_argument("--s3-bucket", default=None)
    health.add_argument("--s3-prefix", default="climate/forecast")
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    valid_date = args.valid_date or datetime.now(UTC).date().isoformat()
    run_id = _run_id(valid_date)
    bucket = _bucket(args)
    run_dir = args.output_dir / run_id if args.storage_mode == "local" else args.output_dir
    manifest_path = _manifest_path(
        storage_mode=args.storage_mode,
        output_dir=run_dir,
        run_id=run_id,
        bucket=bucket,
    )
    existing = _read_existing(manifest_path)
    panel_args = argparse.Namespace(
        start_date=valid_date,
        end_date=valid_date,
        zone_id=list(CANONICAL_ZONES),
        provider_model=[(PROVIDER, MODEL)],
        horizon_days=list(HORIZONS),
        output_dir=run_dir,
        storage_mode=args.storage_mode,
        run_id=run_id,
        s3_bucket=args.s3_bucket,
        s3_prefix=args.s3_prefix,
        write_manifest=False,
    )
    run_panel(panel_args)
    manifest = _enhanced_manifest(
        valid_date=valid_date,
        run_id=run_id,
        storage_mode=args.storage_mode,
        output_dir=run_dir,
        bucket=bucket,
        s3_prefix=args.s3_prefix,
        existing=existing,
    )
    _write_json(manifest_path, manifest)
    return manifest


def _local_health_manifest(root: Path, run_id: str, valid_date: str) -> Path:
    candidates = (root / run_id / "manifest.json", root / valid_date / "manifest.json")
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return candidates[0]


def _health_day(
    *,
    root: Path,
    valid_date: str,
    storage_mode: str,
    bucket: str,
    s3_prefix: str,
) -> dict[str, Any]:
    run_id = _run_id(valid_date)
    manifest_path: str | Path = (
        _local_health_manifest(root, run_id, valid_date)
        if storage_mode == "local"
        else _manifest_path(storage_mode="s3", output_dir=root, run_id=run_id, bucket=bucket)
    )
    if storage_mode == "local":
        local_manifest = Path(manifest_path)
        if not local_manifest.exists():
            return {"valid_date": valid_date, "run_id": run_id, "status": "missing_run"}
        if not local_manifest.is_file():
            return {"valid_date": valid_date, "run_id": run_id, "status": "invalid_manifest"}
    try:
        manifest = _read_json(manifest_path)
    except json.JSONDecodeError:
        return {"valid_date": valid_date, "run_id": run_id, "status": "invalid_manifest"}
    except (OSError, RuntimeError, ValueError, duckdb.Error) as exc:
        status = (
            "missing_run"
            if storage_mode == "s3" and _missing_remote_error(exc)
            else "invalid_manifest"
        )
        return {"valid_date": valid_date, "run_id": run_id, "status": status}
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema_version") != SCHEMA_VERSION
        or not isinstance(manifest.get("created_at"), str)
    ):
        return {"valid_date": valid_date, "run_id": run_id, "status": "invalid_manifest"}
    expected = {
        "run_id": run_id,
        "valid_date": valid_date,
        "storage_mode": storage_mode,
        "zones": list(CANONICAL_ZONES),
        "provider": PROVIDER,
        "model": MODEL,
        "horizons": list(HORIZONS),
        "variables": list(NORMALIZED_VARS),
    }
    if any(manifest.get(field) != value for field, value in expected.items()):
        return {"valid_date": valid_date, "run_id": run_id, "status": "invalid_manifest"}
    if storage_mode == "s3" and manifest.get("s3_prefix") != s3_prefix:
        return {"valid_date": valid_date, "run_id": run_id, "status": "invalid_manifest"}
    counts = manifest.get("counts")
    checks = manifest.get("checks")
    if not isinstance(counts, dict) or not isinstance(checks, dict):
        return {"valid_date": valid_date, "run_id": run_id, "status": "invalid_manifest"}
    expected_counts = {
        "zones": len(CANONICAL_ZONES),
        "horizons": len(HORIZONS),
        "variables": len(NORMALIZED_VARS),
        "artifacts": EXPECTED_ARTIFACTS,
        "rows": EXPECTED_ROWS,
    }
    if checks.get("ok") is True and any(
        counts.get(field) != value for field, value in expected_counts.items()
    ):
        return {"valid_date": valid_date, "run_id": run_id, "status": "invalid_manifest"}
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list) or len(artifacts) != EXPECTED_ARTIFACTS:
        return {"valid_date": valid_date, "run_id": run_id, "status": "partial"}
    by_key = {item.get("key"): item for item in artifacts if isinstance(item, dict)}
    if len(by_key) != EXPECTED_ARTIFACTS:
        return {"valid_date": valid_date, "run_id": run_id, "status": "partial"}
    integrity_error = False
    partial_error = False
    for zone_id in CANONICAL_ZONES:
        for horizon in HORIZONS:
            key = _key(zone_id, valid_date, horizon)
            item = by_key.get(key)
            if not isinstance(item, dict):
                partial_error = True
                continue
            output = item.get("output")
            if not isinstance(output, str):
                integrity_error = True
                continue
            expected_path = _artifact_path(
                storage_mode=storage_mode,
                output_dir=Path(manifest_path).parent,
                run_id=run_id,
                bucket=bucket,
                s3_prefix=s3_prefix,
                key=key,
            )
            if output != str(expected_path):
                integrity_error = True
                continue
            path = expected_path
            try:
                payload = _read_json(path)
                _coverage, errors, _available, _variables, _identity, _finite = _validate_artifact(
                    payload=payload,
                    zone_id=zone_id,
                    valid_date=valid_date,
                    horizon=horizon,
                )
                if errors:
                    partial_error = True
                if item.get("sha256") != _digest(payload):
                    integrity_error = True
            except json.JSONDecodeError:
                integrity_error = True
            except FileNotFoundError:
                partial_error = True
            except (OSError, RuntimeError, ValueError, KeyError, TypeError, duckdb.Error) as exc:
                if storage_mode == "local" or not _missing_remote_error(exc):
                    integrity_error = True
                else:
                    partial_error = True
    if integrity_error:
        status = "invalid_manifest"
    elif partial_error or checks.get("ok") is not True:
        status = "partial"
    else:
        status = "complete"
    return {"valid_date": valid_date, "run_id": run_id, "status": status}


def health(args: argparse.Namespace) -> dict[str, Any]:
    bucket = _bucket(args)
    end = date.fromisoformat(args.end_date)
    start = end - timedelta(days=args.window_days - 1)
    days = [
        _health_day(
            root=args.output_dir,
            valid_date=(start + timedelta(days=offset)).isoformat(),
            storage_mode=args.storage_mode,
            bucket=bucket,
            s3_prefix=args.s3_prefix,
        )
        for offset in range(args.window_days)
    ]
    status_counts = {
        status: sum(day["status"] == status for day in days)
        for status in (
            "complete",
            "missing_run",
            "partial",
            "invalid_manifest",
        )
    }
    return {
        "schema_version": "forecast-forward-archive-health/v1",
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "window_days": args.window_days,
        "days": days,
        "counts": status_counts,
        "checks": {"ok": status_counts["complete"] == args.window_days},
    }


def main(argv: list[str] | None = None) -> int:
    # Install the credential-redaction processor (#148) before anything can log: structlog's
    # DEFAULT chain has no redactor (#202).
    setup_logging()
    args = build_parser().parse_args(argv)
    try:
        result = run(args) if args.command == "run" else health(args)
    except (OSError, RuntimeError, ValueError, duckdb.Error):
        print(json.dumps({"error": "operation failed"}))
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["checks"]["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
