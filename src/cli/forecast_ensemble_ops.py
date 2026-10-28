"""Operational gates for scheduled AIFS ensemble archive captures."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import re
import sys
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any
from uuid import uuid4

from src.cli import forecast_ensemble_archive as archive
from src.config import settings
from src.forecast_ensemble import OpenMeteoEnsembleClient
from src.tools.climate import _load_zones
from src.utils.logging import setup_logging

HEALTH_SCHEMA_VERSION = "forecast-ensemble-archive-health/v1"
STORAGE_PREFLIGHT_SCHEMA_VERSION = "forecast-ensemble-storage-preflight/v1"
STORAGE_CANARY_SCHEMA_VERSION = "forecast-ensemble-storage-canary/v1"
STORAGE_CANARY_OBJECT_SCHEMA_VERSION = "forecast-ensemble-storage-canary-object/v1"
SCHEDULED_CAPTURE_SCHEMA_VERSION = "forecast-ensemble-scheduled-capture/v1"
HEALTH_STATUSES = ("complete", "missing_run", "partial", "invalid_manifest", "error")
STORAGE_CANARY_PREFIX = f"{archive.DEFAULT_S3_PREFIX}/_storage-canary"
MAX_STORAGE_CANARY_BYTES = 64 * 1024


class StorageCanaryError(ValueError):
    """Preserve the immutable object receipt when a post-write probe fails."""

    def __init__(
        self,
        message: str,
        *,
        canary: dict[str, Any],
        checks: dict[str, bool],
    ) -> None:
        super().__init__(message)
        self.canary = dict(canary)
        self.checks = {**checks, "ok": False}


def _add_storage_arguments(parser: argparse.ArgumentParser, *, output_required: bool) -> None:
    parser.add_argument(
        "--output-dir",
        required=output_required,
        type=Path,
        default=None if output_required else Path("/tmp/forecast-ensemble"),
    )
    parser.add_argument("--storage-mode", choices=("local", "s3"), default="local")
    parser.add_argument("--s3-bucket", default=None)
    parser.add_argument("--s3-prefix", default=archive.DEFAULT_S3_PREFIX)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="forecast-ensemble-ops")
    subparsers = parser.add_subparsers(dest="command", required=True)

    health_parser = subparsers.add_parser("health")
    health_parser.add_argument("--end-date", required=True, type=archive._date_arg)
    health_parser.add_argument("--window-days", required=True, type=int, choices=(7, 14))
    _add_storage_arguments(health_parser, output_required=True)

    preflight_parser = subparsers.add_parser("storage-preflight")
    _add_storage_arguments(preflight_parser, output_required=False)
    preflight_parser.add_argument("--require-retention", action="store_true")

    canary_parser = subparsers.add_parser("storage-canary")
    canary_parser.add_argument("--storage-mode", choices=("s3",), default="s3")
    canary_parser.add_argument("--evidence-bucket", default=None)
    canary_parser.add_argument("--canary-id", default=None)
    canary_parser.set_defaults(require_retention=True)

    capture_parser = subparsers.add_parser("scheduled-capture")
    capture_parser.add_argument("--snapshot-date", type=archive._date_arg)
    _add_storage_arguments(capture_parser, output_required=True)
    capture_parser.add_argument("--require-retention", action="store_true")
    return parser


def _manifest_location(
    *, root: Path, snapshot_date: date, storage_mode: str, bucket: str
) -> str | Path:
    return archive._manifest_path(
        storage_mode=storage_mode,
        output_dir=root,
        run_id=archive._run_id(snapshot_date),
        bucket=bucket,
    )


def _day_result(snapshot_date: date, status: str, **details: Any) -> dict[str, Any]:
    return {
        "snapshot_date_utc": snapshot_date.isoformat(),
        "run_id": archive._run_id(snapshot_date),
        "status": status,
        **details,
    }


def _safe_operation_error(exc: Exception) -> str:
    return archive._safe_error_message(exc)


def _health_day(
    *,
    root: Path,
    snapshot_date: date,
    storage_mode: str,
    bucket: str,
    s3_prefix: str,
    expected_endpoint: str,
) -> dict[str, Any]:
    run_id = archive._run_id(snapshot_date)
    manifest_location = _manifest_location(
        root=root,
        snapshot_date=snapshot_date,
        storage_mode=storage_mode,
        bucket=bucket,
    )
    try:
        manifest = archive._read_existing(manifest_location)
    except json.JSONDecodeError:
        return _day_result(snapshot_date, "invalid_manifest")
    except Exception as exc:
        return _day_result(snapshot_date, "error", error=_safe_operation_error(exc))
    if manifest is None:
        readable_artifacts = 0
        missing_artifacts: list[str] = []
        zones: dict[str, dict[str, Any]] | None = None
        for zone_id in archive.CANONICAL_ZONES:
            artifact_location = archive._artifact_path(
                storage_mode=storage_mode,
                output_dir=root,
                run_id=run_id,
                bucket=bucket,
                s3_prefix=s3_prefix,
                zone_id=zone_id,
                snapshot_date=snapshot_date,
            )
            try:
                artifact = archive._read_existing(artifact_location)
            except json.JSONDecodeError:
                return _day_result(snapshot_date, "invalid_manifest")
            except Exception as exc:
                return _day_result(snapshot_date, "error", error=_safe_operation_error(exc))
            if artifact is None:
                missing_artifacts.append(zone_id)
            else:
                try:
                    if zones is None:
                        zones = _load_zones()
                    zone = zones[zone_id]
                    archive._validate_artifact(
                        artifact,
                        zone_id=zone_id,
                        snapshot_date=snapshot_date,
                        run_id=run_id,
                        expected_endpoint=expected_endpoint,
                        expected_latitude=zone["lat"],
                        expected_longitude=zone["lon"],
                    )
                except (KeyError, TypeError, ValueError, archive.EnsembleArchiveError):
                    return _day_result(snapshot_date, "invalid_manifest")
                except Exception as exc:
                    return _day_result(snapshot_date, "error", error=_safe_operation_error(exc))
                readable_artifacts += 1
        if readable_artifacts == 0:
            # A wholly refused provider response set is still an attempted run.
            # The immutable receipt is diagnostic only; it cannot commit a day.
            try:
                receipt_location = _health_capture_path(
                    storage_mode=storage_mode,
                    root=root,
                    run_id=run_id,
                    bucket=bucket,
                    s3_prefix=s3_prefix,
                    name="partial.json",
                )
                try:
                    receipt, _raw = archive._read_document(receipt_location)
                except FileNotFoundError:
                    return _day_result(snapshot_date, "missing_run")
                _validate_partial_receipt(
                    receipt,
                    snapshot_date=snapshot_date,
                    root=root,
                    storage_mode=storage_mode,
                    bucket=bucket,
                    s3_prefix=s3_prefix,
                )
            except (json.JSONDecodeError, TypeError, ValueError):
                return _day_result(snapshot_date, "invalid_manifest")
            except Exception as exc:
                return _day_result(snapshot_date, "error", error=_safe_operation_error(exc))
        return _day_result(
            snapshot_date,
            "partial",
            readable_artifacts=readable_artifacts,
            expected_artifacts=archive.EXPECTED_ARTIFACTS,
            missing_artifacts=missing_artifacts,
        )

    try:
        loaded = archive.load_validated_archive(manifest_location)
    except (json.JSONDecodeError, ValueError, archive.EnsembleArchiveError):
        return _day_result(snapshot_date, "invalid_manifest")
    except Exception as exc:
        return _day_result(snapshot_date, "error", error=_safe_operation_error(exc))
    if len(loaded.rows) != archive.EXPECTED_ROWS:
        return _day_result(snapshot_date, "partial")
    return _day_result(
        snapshot_date,
        "complete",
        counts={
            "artifacts": archive.EXPECTED_ARTIFACTS,
            "rows": archive.EXPECTED_ROWS,
            "member_values": archive.EXPECTED_MEMBER_VALUES,
        },
        manifest_sha256=loaded.manifest_sha256,
    )


def _health_capture_path(
    *,
    storage_mode: str,
    root: Path,
    run_id: str,
    bucket: str,
    s3_prefix: str,
    name: str,
) -> str | Path:
    location = archive._capture_path(
        storage_mode=storage_mode,
        output_dir=root,
        run_id=run_id,
        bucket=bucket,
        s3_prefix=s3_prefix,
        name=name,
    )
    if storage_mode == "local" and location != root.resolve() / run_id / "_captures" / name:
        raise ValueError("partial capture evidence path is not canonical")
    return location


def _validate_capture_reference(reference: Any, *, expected_location: str | Path) -> None:
    """Accept only the generated reference in the health command's known run scope."""
    if (
        not isinstance(reference, dict)
        or set(reference) != {"output", "sha256"}
        or reference.get("output") != str(expected_location)
        or not isinstance(reference.get("sha256"), str)
        or re.fullmatch(r"[0-9a-f]{64}", reference["sha256"]) is None
    ):
        raise ValueError("invalid partial capture reference")


def _read_capture_reference(reference: dict[str, str], location: str | Path) -> Any:
    try:
        document, raw = archive._read_document(location)
    except FileNotFoundError as exc:
        raise ValueError("partial capture evidence is missing") from exc
    if hashlib.sha256(raw).hexdigest() != reference["sha256"]:
        raise ValueError("partial capture evidence digest mismatch")
    archive._ensure_capture_secret_free(document, None)
    return document


def _validate_partial_receipt(
    receipt: Any,
    *,
    snapshot_date: date,
    root: Path,
    storage_mode: str,
    bucket: str,
    s3_prefix: str,
) -> None:
    """Verify a wholly refused capture's evidence without accepting scientific rows."""
    if (
        not isinstance(receipt, dict)
        or receipt.get("schema_version") != "forecast-ensemble-capture-report/v1"
        or receipt.get("run_id") != archive._run_id(snapshot_date)
        or receipt.get("snapshot_date_utc") != snapshot_date.isoformat()
        or receipt.get("status") != "partial"
        or receipt.get("checks") != {"ok": False}
        or receipt["checks"]["ok"] is not False
        or set(receipt)
        != {
            "schema_version",
            "run_id",
            "snapshot_date_utc",
            "status",
            "checks",
            "attempt_id",
            "artifacts",
            "rejected_zones",
            "raw_captures",
            "counts",
            "capture_report",
        }
        or not isinstance(receipt.get("attempt_id"), str)
        or re.fullmatch(r"[0-9a-f]{32}", receipt["attempt_id"]) is None
        or receipt.get("artifacts") != []
    ):
        raise ValueError("invalid partial capture receipt")
    expected_counts = {
        "artifacts": 0,
        "rejected_zones": archive.EXPECTED_ARTIFACTS,
        "raw_captures": archive.EXPECTED_ARTIFACTS,
        "rows": 0,
        "member_values": 0,
        "expected_artifacts": archive.EXPECTED_ARTIFACTS,
    }
    counts = receipt.get("counts")
    if counts != expected_counts or any(type(value) is not int for value in counts.values()):
        raise ValueError("partial capture receipt has inconsistent counts")
    rejected = receipt.get("rejected_zones")
    captures = receipt.get("raw_captures")
    if (
        not isinstance(rejected, list)
        or len(rejected) != archive.EXPECTED_ARTIFACTS
        or not isinstance(captures, list)
        or len(captures) != archive.EXPECTED_ARTIFACTS
    ):
        raise ValueError("partial capture receipt has inconsistent evidence counts")

    def capture_location(name: str) -> str | Path:
        return _health_capture_path(
            storage_mode=storage_mode,
            root=root,
            run_id=archive._run_id(snapshot_date),
            bucket=bucket,
            s3_prefix=s3_prefix,
            name=name,
        )

    report_location = capture_location(f"report-{receipt['attempt_id']}.json")
    _validate_capture_reference(receipt["capture_report"], expected_location=report_location)
    references: dict[str, dict[str, str]] = {}
    locations: dict[str, str | Path] = {}
    for capture in captures:
        zone_id = capture.get("zone_id") if isinstance(capture, dict) else None
        if (
            not isinstance(zone_id, str)
            or zone_id not in archive.CANONICAL_ZONES
            or zone_id in references
            or set(capture) != {"zone_id", "output", "sha256"}
        ):
            raise ValueError("partial capture receipt has invalid raw capture zones")
        reference = {key: capture[key] for key in ("output", "sha256")}
        location = capture_location(f"{zone_id}_{receipt['attempt_id']}.json")
        _validate_capture_reference(reference, expected_location=location)
        references[zone_id] = reference
        locations[zone_id] = location
    seen: set[str] = set()
    for rejection in rejected:
        zone_id = rejection.get("zone_id") if isinstance(rejection, dict) else None
        if (
            not isinstance(zone_id, str)
            or zone_id not in archive.CANONICAL_ZONES
            or zone_id in seen
            or set(rejection) != {"zone_id", "error", "capture"}
            or not isinstance(rejection.get("error"), str)
            or not rejection["error"].strip()
            or rejection.get("capture") != references[zone_id]
        ):
            raise ValueError("partial capture receipt has invalid zones")
        seen.add(zone_id)
    archive._ensure_capture_secret_free(receipt, None)
    # Validate every reference before any referenced read: never follow receipt URLs.
    report = _read_capture_reference(receipt["capture_report"], report_location)
    expected_report = {key: value for key, value in receipt.items() if key != "capture_report"}
    if archive._json_bytes(report) != archive._json_bytes(expected_report):
        raise ValueError("partial capture report differs from its receipt")
    for zone_id, reference in references.items():
        envelope = _read_capture_reference(reference, locations[zone_id])
        if (
            not isinstance(envelope, dict)
            or envelope.get("schema_version") != "forecast-ensemble-capture/v1"
            or envelope.get("zone_id") != zone_id
            or envelope.get("snapshot_date_utc") != snapshot_date.isoformat()
            or "payload" not in envelope
        ):
            raise ValueError("invalid partial raw capture envelope")


def health(args: argparse.Namespace) -> dict[str, Any]:
    bucket = archive._bucket(args)
    if args.storage_mode == "s3":
        archive._validate_s3_scope(bucket=bucket, s3_prefix=args.s3_prefix)
    end_date = args.end_date
    start_date = end_date - timedelta(days=args.window_days - 1)
    days = [
        _health_day(
            root=args.output_dir,
            snapshot_date=start_date + timedelta(days=offset),
            storage_mode=args.storage_mode,
            bucket=bucket,
            s3_prefix=args.s3_prefix,
            expected_endpoint=settings.FORECAST_ENSEMBLE_API_BASE_URL.strip(),
        )
        for offset in range(args.window_days)
    ]
    counts = {status: sum(day["status"] == status for day in days) for status in HEALTH_STATUSES}
    return {
        "schema_version": HEALTH_SCHEMA_VERSION,
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "window_days": args.window_days,
        "storage_mode": args.storage_mode,
        "days": days,
        "counts": counts,
        "checks": {
            "no_operational_errors": counts["error"] == 0,
            "no_integrity_errors": counts["invalid_manifest"] == 0,
            "complete_window": counts["complete"] == args.window_days,
            "ok": counts["complete"] == args.window_days,
        },
    }


def _s3_control_error(exc: Exception) -> str:
    status, code = archive._s3_error_status(exc)
    normalized_code = code.lower()
    if status == 404 or normalized_code in {
        "404",
        "nosuchobjectlockconfiguration",
        "objectlockconfigurationnotfounderror",
    }:
        return "disabled"
    if status == 501 or normalized_code in {"501", "notimplemented", "xnotimplemented"}:
        return "unsupported"
    if status in {401, 403} or normalized_code in {
        "accessdenied",
        "allaccessdisabled",
        "unauthorized",
    }:
        return "unknown"
    raise exc


def _bucket_versioning(client: Any, bucket: str) -> tuple[str, list[str]]:
    method = getattr(client, "get_bucket_versioning", None)
    if not callable(method):
        return "unsupported", ["bucket_versioning_api_unsupported"]
    try:
        response = method(Bucket=bucket)
    except Exception as exc:
        state = _s3_control_error(exc)
        return state, [f"bucket_versioning_{state}"]
    raw_status = response.get("Status") if isinstance(response, dict) else None
    if raw_status == "Enabled":
        return "enabled", []
    if raw_status == "Suspended":
        return "suspended", ["bucket_versioning_suspended"]
    return "disabled", ["bucket_versioning_disabled"]


def _object_lock(client: Any, bucket: str) -> tuple[dict[str, Any], list[str]]:
    method = getattr(client, "get_object_lock_configuration", None)
    if not callable(method):
        return {"state": "unsupported"}, ["object_lock_api_unsupported"]
    try:
        response = method(Bucket=bucket)
    except Exception as exc:
        state = _s3_control_error(exc)
        return {"state": state}, [f"object_lock_{state}"]
    config = response.get("ObjectLockConfiguration") if isinstance(response, dict) else None
    if not isinstance(config, dict) or config.get("ObjectLockEnabled") != "Enabled":
        return {"state": "disabled"}, ["object_lock_disabled"]
    rule = config.get("Rule")
    if not isinstance(rule, dict):
        rule = {}
    default_retention = rule.get("DefaultRetention", {})
    if not isinstance(default_retention, dict):
        default_retention = {}
    mode = default_retention.get("Mode")
    days = default_retention.get("Days")
    years = default_retention.get("Years")
    normalized_days = (
        days if isinstance(days, int) and not isinstance(days, bool) and days > 0 else None
    )
    normalized_years = (
        years if isinstance(years, int) and not isinstance(years, bool) and years > 0 else None
    )
    retention = {
        "state": "enabled",
        "default_mode": mode.lower() if isinstance(mode, str) else None,
        "default_days": normalized_days,
        "default_years": normalized_years,
    }
    has_duration = (normalized_days is not None) != (normalized_years is not None)
    warnings: list[str] = []
    if mode != "COMPLIANCE" or not has_duration:
        warnings.append("default_compliance_retention_missing")
    return retention, warnings


def _nearest_existing_parent(path: Path) -> Path:
    candidate = path.resolve()
    while not candidate.exists() and candidate.parent != candidate:
        candidate = candidate.parent
    return candidate


def _s3_storage_preflight(
    object_store: Any,
    *,
    bucket: str,
    require_retention: bool,
) -> dict[str, Any]:
    object_store.head_bucket(Bucket=bucket)
    versioning, versioning_warnings = _bucket_versioning(object_store, bucket)
    object_lock, lock_warnings = _object_lock(object_store, bucket)
    has_retention_duration = bool(
        object_lock.get("default_days") or object_lock.get("default_years")
    )
    retention_configured = (
        versioning == "enabled"
        and object_lock.get("state") == "enabled"
        and object_lock.get("default_mode") == "compliance"
        and has_retention_duration
    )
    warnings = sorted({*versioning_warnings, *lock_warnings})
    remaining_gates = [
        "scoped_conditional_put_probe",
        "version_bound_manifest",
        "effective_delete_denial_canary",
    ]
    if not retention_configured:
        remaining_gates.insert(0, "versioning_and_default_compliance_retention")
    return {
        "schema_version": STORAGE_PREFLIGHT_SCHEMA_VERSION,
        "storage_mode": "s3",
        "bucket": bucket,
        "storage_capability": (
            "retention_configured_not_opposable" if retention_configured else "prototype_only"
        ),
        "versioning": versioning,
        "object_lock": object_lock,
        "code_ref": archive._runtime_code_ref(),
        "warnings": warnings,
        "remaining_opposable_gates": remaining_gates,
        "checks": {
            "target_reachable": True,
            "application_create_only": True,
            "write_permission_verified": False,
            "retention_configured": retention_configured,
            "opposable_evidence_ready": False,
            "capture_attempt_allowed": True,
            "ok": not require_retention or retention_configured,
        },
    }


def storage_preflight(args: argparse.Namespace, *, client: Any | None = None) -> dict[str, Any]:
    code_ref = archive._runtime_code_ref()
    if args.storage_mode == "local":
        parent = _nearest_existing_parent(args.output_dir)
        writable = parent.is_dir() and os.access(parent, os.W_OK)
        return {
            "schema_version": STORAGE_PREFLIGHT_SCHEMA_VERSION,
            "storage_mode": "local",
            "storage_capability": "prototype_only",
            "versioning": "not_applicable",
            "object_lock": {"state": "not_applicable"},
            "code_ref": code_ref,
            "warnings": ["local_storage_has_no_deny_delete_retention"],
            "remaining_opposable_gates": [
                "versioned_object_storage",
                "default_compliance_retention",
                "scoped_conditional_put_probe",
                "version_bound_manifest",
                "effective_delete_denial_canary",
            ],
            "checks": {
                "target_reachable": writable,
                "application_create_only": True,
                "write_permission_verified": False,
                "retention_configured": False,
                "opposable_evidence_ready": False,
                "capture_attempt_allowed": writable,
                "ok": writable and not args.require_retention,
            },
        }

    bucket = archive._bucket(args)
    archive._validate_s3_scope(bucket=bucket, s3_prefix=args.s3_prefix)
    object_store = client or archive._object_store_client()
    return _s3_storage_preflight(
        object_store,
        bucket=bucket,
        require_retention=args.require_retention,
    )


def _canary_timestamp(value: Any, field: str) -> datetime:
    if isinstance(value, str):
        try:
            value = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError(f"storage canary {field} is invalid") from exc
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"storage canary {field} must be timezone-aware")
    return value.astimezone(UTC)


def _read_canary_version(client: Any, *, bucket: str, key: str, version_id: str) -> bytes:
    response = client.get_object(Bucket=bucket, Key=key, VersionId=version_id)
    if not isinstance(response, dict) or response.get("VersionId") != version_id:
        raise ValueError("storage canary read did not preserve the object version")
    body = response.get("Body")
    if body is None or not callable(getattr(body, "read", None)):
        raise ValueError("storage canary read returned no body")
    try:
        raw = body.read(MAX_STORAGE_CANARY_BYTES + 1)
    finally:
        close = getattr(body, "close", None)
        if callable(close):
            close()
    if not isinstance(raw, bytes):
        raise ValueError("storage canary body must be bytes")
    if len(raw) > MAX_STORAGE_CANARY_BYTES:
        raise ValueError("storage canary body exceeds the maximum allowed size")
    return raw


def _is_conditional_conflict(exc: Exception) -> bool:
    status, code = archive._s3_error_status(exc)
    normalized_code = code.lower()
    return (status == 412 and normalized_code == "preconditionfailed") or (
        status == 409 and normalized_code == "conditionalrequestconflict"
    )


def _is_retention_delete_denial(exc: Exception) -> bool:
    status, code = archive._s3_error_status(exc)
    return status == 403 and code.lower() in {
        "accessdenied",
        "allaccessdisabled",
        "objectlocked",
    }


def _evidence_bucket(args: argparse.Namespace) -> str:
    configured = settings.FORECAST_ENSEMBLE_EVIDENCE_BUCKET.strip()
    if not configured:
        raise ValueError("FORECAST_ENSEMBLE_EVIDENCE_BUCKET must be configured")
    shared_bucket = settings.S3_BUCKET.strip()
    if shared_bucket and configured == shared_bucket:
        raise ValueError("forecast ensemble evidence bucket must differ from S3_BUCKET")
    requested = getattr(args, "evidence_bucket", None)
    if requested is not None and requested.strip() != configured:
        raise ValueError("--evidence-bucket must match FORECAST_ENSEMBLE_EVIDENCE_BUCKET")
    return configured


def _canary_caller() -> dict[str, Any]:
    access_key = settings.S3_ACCESS_KEY.strip()
    railway = {
        name.lower(): value
        for name in (
            "RAILWAY_PROJECT_ID",
            "RAILWAY_ENVIRONMENT_ID",
            "RAILWAY_SERVICE_ID",
            "RAILWAY_DEPLOYMENT_ID",
        )
        if (value := os.getenv(name, "").strip())
    }
    return {
        "access_key_sha256": (
            hashlib.sha256(access_key.encode("utf-8")).hexdigest() if access_key else None
        ),
        "railway": railway,
    }


def _canary_receipt(
    *,
    bucket: str,
    key: str,
    version_id: str | None,
    digest: str,
    caller: dict[str, Any],
    write_outcome: str,
) -> dict[str, Any]:
    return {
        "bucket": bucket,
        "key": key,
        "version_id": version_id,
        "sha256": digest,
        "caller": caller,
        "write_outcome": write_outcome,
    }


def _storage_canary_error(
    exc: Exception,
    *,
    context: str,
    canary: dict[str, Any],
    checks: dict[str, bool],
) -> StorageCanaryError:
    return StorageCanaryError(
        f"storage canary {context}: {_safe_operation_error(exc)}",
        canary=canary,
        checks=checks,
    )


def storage_canary(
    args: argparse.Namespace,
    *,
    client: Any | None = None,
    clock: Any | None = None,
    canary_id: str | None = None,
) -> dict[str, Any]:
    if args.storage_mode != "s3":
        raise ValueError("storage canary requires S3 storage mode")
    bucket = _evidence_bucket(args)
    object_store = client or archive._object_store_client()
    preflight = _s3_storage_preflight(
        object_store,
        bucket=bucket,
        require_retention=True,
    )
    if not preflight["checks"]["ok"]:
        return {
            "schema_version": STORAGE_CANARY_SCHEMA_VERSION,
            "preflight": preflight,
            "canary": None,
            "checks": {"canary_started": False, "ok": False},
        }

    now = _canary_timestamp((clock or (lambda: datetime.now(UTC)))(), "clock")
    identifier = canary_id or getattr(args, "canary_id", None) or uuid4().hex
    if (
        not isinstance(identifier, str)
        or not 8 <= len(identifier) <= 64
        or any(character not in "abcdefghijklmnopqrstuvwxyz0123456789-" for character in identifier)
    ):
        raise ValueError("storage canary id is invalid")

    key = f"{STORAGE_CANARY_PREFIX}/{now.date().isoformat()}/{identifier}.json"
    caller = _canary_caller()
    payload = {
        "schema_version": STORAGE_CANARY_OBJECT_SCHEMA_VERSION,
        "canary_id": identifier,
        "created_at_utc": now.isoformat(timespec="seconds").replace("+00:00", "Z"),
        "purpose": "aifs_ensemble_storage_evidence",
        "code_ref": archive._runtime_code_ref(),
        "caller": caller,
    }
    encoded = archive._json_bytes(payload)
    digest = hashlib.sha256(encoded).hexdigest()
    content_md5 = base64.b64encode(hashlib.md5(encoded, usedforsecurity=False).digest()).decode(
        "ascii"
    )
    checks = {
        "canary_started": True,
        "conditional_create": False,
        "version_bound_read": False,
        "conditional_rewrite_denied": False,
        "caller_version_delete_denied": False,
        "post_denial_read": False,
    }
    version_id: str | None = None
    receipt = _canary_receipt(
        bucket=bucket,
        key=key,
        version_id=version_id,
        digest=digest,
        caller=caller,
        write_outcome="unknown",
    )
    try:
        put_response = object_store.put_object(
            Bucket=bucket,
            Key=key,
            Body=encoded,
            ContentLength=len(encoded),
            ContentType="application/json",
            ContentMD5=content_md5,
            IfNoneMatch="*",
        )
    except Exception as exc:
        if _is_conditional_conflict(exc):
            receipt["write_outcome"] = "conditional_conflict"
            context = "initial conditional write was rejected"
        else:
            context = "initial write outcome is unknown"
        raise _storage_canary_error(
            exc,
            context=context,
            canary=receipt,
            checks=checks,
        ) from None
    checks["conditional_create"] = True
    raw_version_id = put_response.get("VersionId") if isinstance(put_response, dict) else None
    version_id = raw_version_id if isinstance(raw_version_id, str) else None
    receipt["version_id"] = version_id
    receipt["write_outcome"] = "accepted"
    if not isinstance(version_id, str) or not version_id or version_id.lower() == "null":
        raise _storage_canary_error(
            ValueError("storage canary write returned no version id"),
            context="failed after write",
            canary=receipt,
            checks=checks,
        )
    receipt["write_outcome"] = "versioned"

    try:
        head = object_store.head_object(Bucket=bucket, Key=key, VersionId=version_id)
        if not isinstance(head, dict) or head.get("VersionId") != version_id:
            raise ValueError("storage canary head did not preserve the object version")
        if head.get("ObjectLockMode") != "COMPLIANCE":
            raise ValueError("storage canary object is not under COMPLIANCE retention")
        retain_until = _canary_timestamp(
            head.get("ObjectLockRetainUntilDate"), "retain-until timestamp"
        )
        if retain_until <= now:
            raise ValueError("storage canary retention has already expired")

        first_read = _read_canary_version(
            object_store,
            bucket=bucket,
            key=key,
            version_id=version_id,
        )
        if hashlib.sha256(first_read).hexdigest() != digest:
            raise ValueError("storage canary read-back digest is inconsistent")
        checks["version_bound_read"] = True

        try:
            object_store.put_object(
                Bucket=bucket,
                Key=key,
                Body=encoded,
                ContentLength=len(encoded),
                ContentType="application/json",
                ContentMD5=content_md5,
                IfNoneMatch="*",
            )
        except Exception as exc:
            if not _is_conditional_conflict(exc):
                raise
        else:
            raise ValueError("storage canary conditional rewrite unexpectedly succeeded")
        checks["conditional_rewrite_denied"] = True

        try:
            object_store.delete_object(Bucket=bucket, Key=key, VersionId=version_id)
        except Exception as exc:
            if not _is_retention_delete_denial(exc):
                raise
        else:
            raise ValueError("storage canary retained version was unexpectedly deleted")
        checks["caller_version_delete_denied"] = True

        final_read = _read_canary_version(
            object_store,
            bucket=bucket,
            key=key,
            version_id=version_id,
        )
        if hashlib.sha256(final_read).hexdigest() != digest:
            raise ValueError("storage canary changed after the delete-denial probe")
        checks["post_denial_read"] = True
    except Exception as exc:
        raise _storage_canary_error(
            exc,
            context="failed after write",
            canary=receipt,
            checks=checks,
        ) from None

    return {
        "schema_version": STORAGE_CANARY_SCHEMA_VERSION,
        "preflight": preflight,
        "canary": {
            "bucket": bucket,
            "key": key,
            "version_id": version_id,
            "sha256": digest,
            "object_lock_mode": "compliance",
            "retain_until_utc": retain_until.isoformat().replace("+00:00", "Z"),
            "caller": caller,
            "write_outcome": "versioned",
        },
        "remaining_opposable_gates": [
            "version_bound_manifest",
            "privileged_retention_delete_denial",
            "iam_policy_review",
        ],
        "checks": {**checks, "ok": True},
    }


def scheduled_capture(
    args: argparse.Namespace,
    *,
    object_store_client: Any | None = None,
    provider_client: OpenMeteoEnsembleClient | None = None,
) -> dict[str, Any]:
    preflight = storage_preflight(args, client=object_store_client)
    if not preflight["checks"]["ok"]:
        return {
            "schema_version": SCHEDULED_CAPTURE_SCHEMA_VERSION,
            "preflight": preflight,
            "capture": None,
            "checks": {"capture_started": False, "ok": False},
        }
    manifest = archive.run(args, client=provider_client)
    capture_ok = manifest.get("checks", {}).get("ok") is True
    return {
        "schema_version": SCHEDULED_CAPTURE_SCHEMA_VERSION,
        "preflight": preflight,
        "capture": manifest,
        "checks": {"capture_started": True, "ok": capture_ok},
    }


def _exit_code(command: str, result: dict[str, Any]) -> int:
    if command == "health":
        checks = result["checks"]
        if not checks["no_operational_errors"] or not checks["no_integrity_errors"]:
            return 2
    if command == "scheduled-capture":
        capture = result.get("capture")
        if (
            result.get("preflight", {}).get("checks", {}).get("ok") is True
            and result["checks"].get("capture_started") is True
            and isinstance(capture, dict)
            and capture.get("schema_version") == archive.CAPTURE_REPORT_SCHEMA_VERSION
            and capture.get("status") == "partial"
            and isinstance(capture.get("capture_report"), dict)
        ):
            # The collector returns this report only after attempting all zones and
            # retaining the refusal evidence. A completed invocation may be partial;
            # checks.ok still records scientific completeness, and health stays nonzero.
            return 0
    return 0 if result["checks"]["ok"] else 1


def main(argv: list[str] | None = None) -> int:
    # Install the credential-redaction processor (#148) before anything can log: structlog's
    # DEFAULT chain has no redactor (#202).
    setup_logging()
    args = build_parser().parse_args(argv)
    try:
        if args.command == "health":
            result = health(args)
        elif args.command == "storage-preflight":
            result = storage_preflight(args)
        elif args.command == "storage-canary":
            result = storage_canary(args)
        else:
            result = scheduled_capture(args)
    except StorageCanaryError as exc:
        print(
            json.dumps(
                {
                    "schema_version": STORAGE_CANARY_SCHEMA_VERSION,
                    "status": "error",
                    "error": _safe_operation_error(exc),
                    "canary": exc.canary,
                    "checks": exc.checks,
                },
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2
    except Exception as exc:
        print(
            json.dumps({"status": "error", "error": _safe_operation_error(exc)}),
            file=sys.stderr,
        )
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return _exit_code(args.command, result)


if __name__ == "__main__":
    raise SystemExit(main())
