from __future__ import annotations

import argparse
import base64
import hashlib
import json
from datetime import UTC, date, datetime, timedelta
from io import BytesIO
from pathlib import Path
from typing import Any

import httpx
import pytest

from src.cli import forecast_ensemble_archive as archive
from src.cli import forecast_ensemble_ops as ops
from src.forecast_ensemble import MEMBER_FIELDS, OpenMeteoEnsembleClient

SNAPSHOT_DATE = date(2026, 7, 15)
NOW = datetime(2026, 7, 15, 19, 7, tzinfo=UTC)
PUBLIC_ENDPOINT = "https://ensemble-api.open-meteo.com/v1/ensemble"


def _payload(*, snapshot_date: date = SNAPSHOT_DATE, days: int = 8) -> dict[str, Any]:
    dates = [(snapshot_date + timedelta(days=offset)).isoformat() for offset in range(1, days + 1)]
    daily: dict[str, Any] = {"time": dates}
    units = {"time": "iso8601"}
    for member_id, field in enumerate(MEMBER_FIELDS):
        daily[field] = [0.1 if member_id < 11 else 1.0] * days
        units[field] = "mm"
    return {
        "latitude": 43.6,
        "longitude": 3.875,
        "elevation": 42.0,
        "timezone": "GMT",
        "utc_offset_seconds": 0,
        "daily_units": units,
        "daily": daily,
    }


def _client(calls: list[dict[str, Any]]) -> OpenMeteoEnsembleClient:
    def transport(params: dict[str, Any]) -> dict[str, Any]:
        calls.append(dict(params))
        return {
            **_payload(),
            "latitude": params["latitude"],
            "longitude": params["longitude"],
        }

    return OpenMeteoEnsembleClient(
        base_url=PUBLIC_ENDPOINT,
        transport=transport,
        clock=lambda: NOW,
    )


def _archive_args(root: Path) -> argparse.Namespace:
    return archive.build_parser().parse_args(
        ["--snapshot-date", SNAPSHOT_DATE.isoformat(), "--output-dir", str(root)]
    )


def _health_args(root: Path) -> argparse.Namespace:
    return ops.build_parser().parse_args(
        [
            "health",
            "--end-date",
            SNAPSHOT_DATE.isoformat(),
            "--window-days",
            "7",
            "--output-dir",
            str(root),
        ]
    )


def _s3_args(
    *,
    command: str,
    require_retention: bool = False,
    evidence_bucket: str | None = None,
) -> argparse.Namespace:
    if command == "storage-canary":
        argv = [command, "--storage-mode", "s3"]
        if evidence_bucket is not None:
            argv.extend(["--evidence-bucket", evidence_bucket])
        return ops.build_parser().parse_args(argv)
    argv = [
        command,
        "--storage-mode",
        "s3",
        "--s3-bucket",
        "geoai-archive",
        "--output-dir",
        "/tmp/forecast-ensemble",
    ]
    if require_retention:
        argv.append("--require-retention")
    return ops.build_parser().parse_args(argv)


def _configure_evidence_bucket(
    monkeypatch,
    *,
    evidence_bucket: str = "geoai-evidence",
    shared_bucket: str = "geoai-archive",
) -> None:
    monkeypatch.setattr(archive.settings, "S3_BUCKET", shared_bucket)
    monkeypatch.setattr(
        archive.settings,
        "FORECAST_ENSEMBLE_EVIDENCE_BUCKET",
        evidence_bucket,
    )
    monkeypatch.setattr(archive.settings, "S3_ACCESS_KEY", "scheduler-access-key")
    for name in (
        "RAILWAY_PROJECT_ID",
        "RAILWAY_ENVIRONMENT_ID",
        "RAILWAY_SERVICE_ID",
        "RAILWAY_DEPLOYMENT_ID",
    ):
        monkeypatch.delenv(name, raising=False)


class FakeObjectStore:
    def __init__(self, *, retained: bool = False) -> None:
        self.retained = retained
        self.head_requests: list[dict[str, Any]] = []

    def head_bucket(self, **request: Any) -> dict[str, Any]:
        self.head_requests.append(request)
        return {}

    def get_bucket_versioning(self, **_request: Any) -> dict[str, Any]:
        return {"Status": "Enabled" if self.retained else "Suspended"}

    def get_object_lock_configuration(self, **_request: Any) -> dict[str, Any]:
        if not self.retained:
            return {"ObjectLockConfiguration": {}}
        return {
            "ObjectLockConfiguration": {
                "ObjectLockEnabled": "Enabled",
                "Rule": {"DefaultRetention": {"Mode": "COMPLIANCE", "Days": 30}},
            }
        }


class FakeS3Error(RuntimeError):
    def __init__(self, status: int, code: str, message: str) -> None:
        super().__init__(message)
        self.response = {
            "ResponseMetadata": {"HTTPStatusCode": status},
            "Error": {"Code": code},
        }


class CanaryObjectStore(FakeObjectStore):
    def __init__(self) -> None:
        super().__init__(retained=True)
        self.version_id = "version-0001"
        self.body: bytes | None = None
        self.put_requests: list[dict[str, Any]] = []
        self.object_head_requests: list[dict[str, Any]] = []
        self.get_requests: list[dict[str, Any]] = []
        self.delete_requests: list[dict[str, Any]] = []

    def put_object(self, **request: Any) -> dict[str, Any]:
        self.put_requests.append(request)
        if self.body is not None:
            raise FakeS3Error(412, "PreconditionFailed", "already exists")
        self.body = request["Body"]
        return {"VersionId": self.version_id}

    def head_object(self, **request: Any) -> dict[str, Any]:
        self.object_head_requests.append(request)
        return {
            "VersionId": self.version_id,
            "ObjectLockMode": "COMPLIANCE",
            "ObjectLockRetainUntilDate": NOW + timedelta(days=30),
        }

    def get_object(self, **request: Any) -> dict[str, Any]:
        self.get_requests.append(request)
        assert self.body is not None
        return {"VersionId": self.version_id, "Body": BytesIO(self.body)}

    def delete_object(self, **request: Any) -> dict[str, Any]:
        self.delete_requests.append(request)
        raise FakeS3Error(403, "AccessDenied", "retained version")


def test_health_validates_real_archive_without_new_provider_calls(tmp_path: Path) -> None:
    calls: list[dict[str, Any]] = []
    archive.run(_archive_args(tmp_path), client=_client(calls))
    assert len(calls) == len(archive.CANONICAL_ZONES)

    result = ops.health(_health_args(tmp_path))

    assert len(calls) == len(archive.CANONICAL_ZONES)
    assert result["counts"] == {
        "complete": 1,
        "missing_run": 6,
        "partial": 0,
        "invalid_manifest": 0,
        "error": 0,
    }
    complete = result["days"][-1]
    assert complete["status"] == "complete"
    assert complete["counts"] == {"artifacts": 15, "rows": 60, "member_values": 3060}
    assert len(complete["manifest_sha256"]) == 64
    assert result["checks"] == {
        "no_operational_errors": True,
        "no_integrity_errors": True,
        "complete_window": False,
        "ok": False,
    }


def test_health_distinguishes_partial_and_corrupt_manifest(tmp_path: Path) -> None:
    manifest = archive.run(_archive_args(tmp_path), client=_client([]))
    manifest_path = tmp_path / archive._run_id(SNAPSHOT_DATE) / "manifest.json"
    manifest_path.unlink()
    for item in manifest["artifacts"][1:]:
        Path(item["output"]).unlink()

    partial = ops.health(_health_args(tmp_path))
    assert partial["days"][-1]["status"] == "partial"
    assert partial["days"][-1]["readable_artifacts"] == 1
    assert partial["days"][-1]["missing_artifacts"] == list(archive.CANONICAL_ZONES[1:])

    manifest_path.write_text("{}", encoding="utf-8")
    invalid = ops.health(_health_args(tmp_path))
    assert invalid["days"][-1]["status"] == "invalid_manifest"
    assert invalid["checks"]["no_integrity_errors"] is False

    manifest_path.write_text("{not-json", encoding="utf-8")
    malformed = ops.health(_health_args(tmp_path))
    assert malformed["days"][-1]["status"] == "invalid_manifest"


def test_health_treats_missing_artifact_after_manifest_as_integrity_error(
    tmp_path: Path,
) -> None:
    manifest = archive.run(_archive_args(tmp_path), client=_client([]))
    Path(manifest["artifacts"][0]["output"]).unlink()

    result = ops.health(_health_args(tmp_path))

    assert result["days"][-1]["status"] == "invalid_manifest"
    assert ops._exit_code("health", result) == 2


def test_health_treats_invalid_orphan_artifact_as_integrity_error(tmp_path: Path) -> None:
    manifest = archive.run(_archive_args(tmp_path), client=_client([]))
    manifest_path = tmp_path / archive._run_id(SNAPSHOT_DATE) / "manifest.json"
    manifest_path.unlink()
    Path(manifest["artifacts"][0]["output"]).write_text("{}", encoding="utf-8")

    result = ops.health(_health_args(tmp_path))

    assert result["days"][-1]["status"] == "invalid_manifest"
    assert ops._exit_code("health", result) == 2


def test_health_rejects_orphan_artifact_from_unexpected_endpoint(tmp_path: Path) -> None:
    manifest = archive.run(_archive_args(tmp_path), client=_client([]))
    manifest_path = tmp_path / archive._run_id(SNAPSHOT_DATE) / "manifest.json"
    manifest_path.unlink()
    artifact_path = Path(manifest["artifacts"][0]["output"])
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    artifact["endpoint"] = "https://customer-ensemble-api.open-meteo.com/v1/ensemble"
    artifact_path.write_text(json.dumps(artifact), encoding="utf-8")

    result = ops.health(_health_args(tmp_path))

    assert result["days"][-1]["status"] == "invalid_manifest"
    assert ops._exit_code("health", result) == 2


def test_health_operational_error_uses_exit_two(tmp_path: Path, monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        ops.archive,
        "_read_existing",
        lambda _path: (_ for _ in ()).throw(PermissionError("permission denied")),
    )

    result = ops.health(_health_args(tmp_path))
    assert result["counts"]["error"] == 7
    assert result["checks"]["no_operational_errors"] is False

    exit_code = ops.main(
        [
            "health",
            "--end-date",
            SNAPSHOT_DATE.isoformat(),
            "--window-days",
            "7",
            "--output-dir",
            str(tmp_path),
        ]
    )
    output = json.loads(capsys.readouterr().out)
    assert exit_code == 2
    assert output["counts"]["error"] == 7


def test_s3_health_treats_not_found_as_gap_and_access_denied_as_error(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(archive.settings, "S3_BUCKET", "geoai-archive")
    args = ops.build_parser().parse_args(
        [
            "health",
            "--end-date",
            SNAPSHOT_DATE.isoformat(),
            "--window-days",
            "7",
            "--output-dir",
            str(tmp_path),
            "--storage-mode",
            "s3",
            "--s3-bucket",
            "geoai-archive",
        ]
    )

    class ErrorStore:
        error = FakeS3Error(404, "NoSuchKey", "404 Not Found")

        def get_object(self, **_kwargs):
            raise self.error

    store = ErrorStore()
    monkeypatch.setattr(archive, "_object_store_client", lambda: store)
    missing = ops.health(args)
    assert missing["counts"]["missing_run"] == 7
    assert missing["counts"]["error"] == 0

    store.error = FakeS3Error(403, "AccessDenied", "access denied")
    denied = ops.health(args)
    assert denied["counts"]["missing_run"] == 0
    assert denied["counts"]["error"] == 7
    assert denied["checks"]["no_operational_errors"] is False


def test_storage_preflight_reports_prototype_without_overclaim(monkeypatch) -> None:
    monkeypatch.setattr(archive.settings, "S3_BUCKET", "geoai-archive")
    store = FakeObjectStore()

    result = ops.storage_preflight(_s3_args(command="storage-preflight"), client=store)

    assert store.head_requests == [{"Bucket": "geoai-archive"}]
    assert result["storage_capability"] == "prototype_only"
    assert result["versioning"] == "suspended"
    assert result["object_lock"] == {"state": "disabled"}
    assert result["checks"]["capture_attempt_allowed"] is True
    assert result["checks"]["write_permission_verified"] is False
    assert result["checks"]["retention_configured"] is False
    assert result["checks"]["opposable_evidence_ready"] is False
    assert result["checks"]["ok"] is True


def test_storage_preflight_accepts_visible_compliance_retention(monkeypatch) -> None:
    monkeypatch.setattr(archive.settings, "S3_BUCKET", "geoai-archive")

    result = ops.storage_preflight(
        _s3_args(command="storage-preflight", require_retention=True),
        client=FakeObjectStore(retained=True),
    )

    assert result["storage_capability"] == "retention_configured_not_opposable"
    assert result["versioning"] == "enabled"
    assert result["object_lock"] == {
        "state": "enabled",
        "default_mode": "compliance",
        "default_days": 30,
        "default_years": None,
    }
    assert result["checks"]["retention_configured"] is True
    assert result["checks"]["write_permission_verified"] is False
    assert result["checks"]["opposable_evidence_ready"] is False
    assert result["checks"]["ok"] is True


def test_storage_preflight_keeps_unreadable_controls_inconclusive(monkeypatch) -> None:
    monkeypatch.setattr(archive.settings, "S3_BUCKET", "geoai-archive")

    class RestrictedControlPlane(FakeObjectStore):
        def get_bucket_versioning(self, **_request: Any) -> dict[str, Any]:
            raise FakeS3Error(403, "AccessDenied", "access denied")

        def get_object_lock_configuration(self, **_request: Any) -> dict[str, Any]:
            raise FakeS3Error(403, "AccessDenied", "access denied")

    prototype = ops.storage_preflight(
        _s3_args(command="storage-preflight"),
        client=RestrictedControlPlane(),
    )
    strict = ops.storage_preflight(
        _s3_args(command="storage-preflight", require_retention=True),
        client=RestrictedControlPlane(),
    )

    assert prototype["versioning"] == "unknown"
    assert prototype["object_lock"] == {"state": "unknown"}
    assert prototype["checks"]["ok"] is True
    assert strict["checks"]["retention_configured"] is False
    assert strict["checks"]["ok"] is False


def test_storage_canary_proves_versioned_compliance_and_delete_denial(monkeypatch) -> None:
    _configure_evidence_bucket(monkeypatch)
    store = CanaryObjectStore()

    result = ops.storage_canary(
        _s3_args(command="storage-canary"),
        client=store,
        clock=lambda: NOW,
        canary_id="canary-1234",
    )

    expected_key = "climate/forecast-ensemble/_storage-canary/2026-07-15/canary-1234.json"
    assert result["schema_version"] == ops.STORAGE_CANARY_SCHEMA_VERSION
    assert result["preflight"]["checks"]["retention_configured"] is True
    assert result["canary"] == {
        "bucket": "geoai-evidence",
        "key": expected_key,
        "version_id": "version-0001",
        "sha256": result["canary"]["sha256"],
        "object_lock_mode": "compliance",
        "retain_until_utc": "2026-08-14T19:07:00Z",
        "caller": {
            "access_key_sha256": hashlib.sha256(b"scheduler-access-key").hexdigest(),
            "railway": {},
        },
        "write_outcome": "versioned",
    }
    assert len(result["canary"]["sha256"]) == 64
    assert result["remaining_opposable_gates"] == [
        "version_bound_manifest",
        "privileged_retention_delete_denial",
        "iam_policy_review",
    ]
    assert result["checks"] == {
        "canary_started": True,
        "conditional_create": True,
        "version_bound_read": True,
        "conditional_rewrite_denied": True,
        "caller_version_delete_denied": True,
        "post_denial_read": True,
        "ok": True,
    }
    assert store.head_requests == [{"Bucket": "geoai-evidence"}]
    assert len(store.put_requests) == 2
    assert all(request["IfNoneMatch"] == "*" for request in store.put_requests)
    assert all(request["Bucket"] == "geoai-evidence" for request in store.put_requests)
    expected_md5 = base64.b64encode(
        hashlib.md5(store.body or b"", usedforsecurity=False).digest()
    ).decode("ascii")
    assert all(request["ContentMD5"] == expected_md5 for request in store.put_requests)
    assert store.object_head_requests == [
        {"Bucket": "geoai-evidence", "Key": expected_key, "VersionId": "version-0001"}
    ]
    assert store.delete_requests == [
        {"Bucket": "geoai-evidence", "Key": expected_key, "VersionId": "version-0001"}
    ]
    expected_get = {
        "Bucket": "geoai-evidence",
        "Key": expected_key,
        "VersionId": "version-0001",
    }
    assert store.get_requests == [expected_get, expected_get]


def test_storage_canary_blocks_before_write_without_retention(monkeypatch) -> None:
    _configure_evidence_bucket(monkeypatch)

    result = ops.storage_canary(
        _s3_args(command="storage-canary"),
        client=FakeObjectStore(),
        clock=lambda: NOW,
        canary_id="canary-1234",
    )

    assert result["canary"] is None
    assert result["checks"] == {"canary_started": False, "ok": False}
    assert ops._exit_code("storage-canary", result) == 1


def test_storage_canary_fails_if_retained_version_can_be_deleted(monkeypatch) -> None:
    _configure_evidence_bucket(monkeypatch)

    class DeletableObjectStore(CanaryObjectStore):
        def delete_object(self, **request: Any) -> dict[str, Any]:
            self.delete_requests.append(request)
            self.body = None
            return {}

    with pytest.raises(ops.StorageCanaryError, match="unexpectedly deleted") as caught:
        ops.storage_canary(
            _s3_args(command="storage-canary"),
            client=DeletableObjectStore(),
            clock=lambda: NOW,
            canary_id="canary-1234",
        )
    assert caught.value.canary["version_id"] == "version-0001"
    assert caught.value.checks["caller_version_delete_denied"] is False


def test_storage_canary_failure_preserves_post_write_receipt(monkeypatch) -> None:
    _configure_evidence_bucket(monkeypatch)

    class GovernanceObjectStore(CanaryObjectStore):
        def head_object(self, **request: Any) -> dict[str, Any]:
            result = super().head_object(**request)
            result["ObjectLockMode"] = "GOVERNANCE"
            return result

    with pytest.raises(ops.StorageCanaryError) as caught:
        ops.storage_canary(
            _s3_args(command="storage-canary"),
            client=GovernanceObjectStore(),
            clock=lambda: NOW,
            canary_id="canary-1234",
        )

    error = str(caught.value)
    assert "failed after write" in error
    assert "not under COMPLIANCE retention" in error
    assert caught.value.canary["bucket"] == "geoai-evidence"
    assert caught.value.canary["key"].endswith("canary-1234.json")
    assert caught.value.canary["version_id"] == "version-0001"
    assert caught.value.checks["version_bound_read"] is False


def test_storage_canary_rejects_null_version_with_object_key_receipt(monkeypatch) -> None:
    _configure_evidence_bucket(monkeypatch)

    class NullVersionObjectStore(CanaryObjectStore):
        def put_object(self, **request: Any) -> dict[str, Any]:
            result = super().put_object(**request)
            return {**result, "VersionId": "null"}

    with pytest.raises(ops.StorageCanaryError) as caught:
        ops.storage_canary(
            _s3_args(command="storage-canary"),
            client=NullVersionObjectStore(),
            clock=lambda: NOW,
            canary_id="canary-1234",
        )

    error = str(caught.value)
    assert "no version id" in error
    assert caught.value.canary["bucket"] == "geoai-evidence"
    assert caught.value.canary["key"].endswith("canary-1234.json")
    assert caught.value.canary["version_id"] == "null"
    assert caught.value.canary["write_outcome"] == "accepted"


def test_storage_canary_initial_put_timeout_keeps_an_ambiguous_write_receipt(
    monkeypatch,
) -> None:
    _configure_evidence_bucket(monkeypatch)

    class TimedOutObjectStore(CanaryObjectStore):
        def put_object(self, **request: Any) -> dict[str, Any]:
            self.put_requests.append(request)
            raise TimeoutError("write response timed out")

    store = TimedOutObjectStore()
    with pytest.raises(ops.StorageCanaryError, match="outcome is unknown") as caught:
        ops.storage_canary(
            _s3_args(command="storage-canary"),
            client=store,
            clock=lambda: NOW,
            canary_id="canary-1234",
        )

    assert caught.value.canary == {
        "bucket": "geoai-evidence",
        "key": "climate/forecast-ensemble/_storage-canary/2026-07-15/canary-1234.json",
        "version_id": None,
        "sha256": caught.value.canary["sha256"],
        "caller": {
            "access_key_sha256": hashlib.sha256(b"scheduler-access-key").hexdigest(),
            "railway": {},
        },
        "write_outcome": "unknown",
    }
    assert caught.value.checks["conditional_create"] is False
    assert store.put_requests[0]["IfNoneMatch"] == "*"
    assert store.put_requests[0]["ContentMD5"]


@pytest.mark.parametrize(
    ("status", "code"),
    [
        (412, "PreconditionFailed"),
        (409, "ConditionalRequestConflict"),
    ],
)
def test_storage_canary_initial_conditional_conflict_is_not_ambiguous(
    monkeypatch,
    status: int,
    code: str,
) -> None:
    _configure_evidence_bucket(monkeypatch)

    class ConflictingObjectStore(CanaryObjectStore):
        def put_object(self, **request: Any) -> dict[str, Any]:
            self.put_requests.append(request)
            raise FakeS3Error(status, code, "conditional write rejected")

    store = ConflictingObjectStore()
    with pytest.raises(ops.StorageCanaryError, match="conditional write was rejected") as caught:
        ops.storage_canary(
            _s3_args(command="storage-canary"),
            client=store,
            clock=lambda: NOW,
            canary_id="canary-1234",
        )

    assert caught.value.canary["write_outcome"] == "conditional_conflict"
    assert caught.value.canary["version_id"] is None
    assert caught.value.checks["conditional_create"] is False
    assert len(store.put_requests) == 1
    assert store.put_requests[0]["IfNoneMatch"] == "*"
    assert store.object_head_requests == []
    assert store.get_requests == []
    assert store.delete_requests == []


def test_storage_canary_initial_unrelated_409_remains_ambiguous(monkeypatch) -> None:
    _configure_evidence_bucket(monkeypatch)

    class InvalidRequestObjectStore(CanaryObjectStore):
        def put_object(self, **request: Any) -> dict[str, Any]:
            self.put_requests.append(request)
            raise FakeS3Error(409, "InvalidRequest", "unrelated 409")

    store = InvalidRequestObjectStore()
    with pytest.raises(ops.StorageCanaryError, match="outcome is unknown") as caught:
        ops.storage_canary(
            _s3_args(command="storage-canary"),
            client=store,
            clock=lambda: NOW,
            canary_id="canary-1234",
        )

    assert caught.value.canary["write_outcome"] == "unknown"
    assert caught.value.checks["conditional_create"] is False
    assert len(store.put_requests) == 1
    assert store.object_head_requests == []
    assert store.get_requests == []
    assert store.delete_requests == []


def test_storage_canary_rejects_missing_or_shared_evidence_bucket(monkeypatch) -> None:
    store = CanaryObjectStore()
    _configure_evidence_bucket(monkeypatch, evidence_bucket="")
    with pytest.raises(ValueError, match="must be configured"):
        ops.storage_canary(_s3_args(command="storage-canary"), client=store)

    _configure_evidence_bucket(monkeypatch, evidence_bucket="geoai-archive")
    with pytest.raises(ValueError, match="must differ"):
        ops.storage_canary(_s3_args(command="storage-canary"), client=store)

    assert store.head_requests == []
    assert store.put_requests == []


def test_storage_canary_rejects_cli_bucket_mismatch_before_write(monkeypatch) -> None:
    _configure_evidence_bucket(monkeypatch)
    store = CanaryObjectStore()

    with pytest.raises(ValueError, match="must match"):
        ops.storage_canary(
            _s3_args(command="storage-canary", evidence_bucket="other-evidence"),
            client=store,
        )

    assert store.head_requests == []
    assert store.put_requests == []


def test_storage_canary_does_not_accept_unrelated_409_as_conditional_conflict(
    monkeypatch,
) -> None:
    _configure_evidence_bucket(monkeypatch)

    class InvalidRequestObjectStore(CanaryObjectStore):
        def put_object(self, **request: Any) -> dict[str, Any]:
            if self.body is not None:
                self.put_requests.append(request)
                raise FakeS3Error(409, "InvalidRequest", "unrelated 409")
            return super().put_object(**request)

    with pytest.raises(ops.StorageCanaryError, match="unrelated 409") as caught:
        ops.storage_canary(
            _s3_args(command="storage-canary"),
            client=InvalidRequestObjectStore(),
            clock=lambda: NOW,
            canary_id="canary-1234",
        )

    assert caught.value.checks["conditional_rewrite_denied"] is False


def test_storage_canary_main_emits_structured_post_write_receipt(monkeypatch, capsys) -> None:
    failure = ops.StorageCanaryError(
        "storage canary failed after write: denied",
        canary={
            "bucket": "geoai-evidence",
            "key": "climate/forecast-ensemble/_storage-canary/test.json",
            "version_id": "version-0001",
            "sha256": "a" * 64,
            "caller": {"access_key_sha256": None, "railway": {}},
        },
        checks={"canary_started": True, "conditional_create": True},
    )
    monkeypatch.setattr(
        ops,
        "storage_canary",
        lambda _args: (_ for _ in ()).throw(failure),
    )

    assert ops.main(["storage-canary", "--canary-id", "canary-1234"]) == 2
    payload = json.loads(capsys.readouterr().err)
    assert payload["schema_version"] == ops.STORAGE_CANARY_SCHEMA_VERSION
    assert payload["canary"]["version_id"] == "version-0001"
    assert payload["checks"] == {
        "canary_started": True,
        "conditional_create": True,
        "ok": False,
    }


def test_scheduled_capture_blocks_before_provider_when_retention_is_required(
    monkeypatch,
) -> None:
    monkeypatch.setattr(archive.settings, "S3_BUCKET", "geoai-archive")
    monkeypatch.setattr(
        ops.archive,
        "run",
        lambda *_args, **_kwargs: pytest.fail("retention failure started capture"),
    )

    result = ops.scheduled_capture(
        _s3_args(command="scheduled-capture", require_retention=True),
        object_store_client=FakeObjectStore(),
    )

    assert result["capture"] is None
    assert result["checks"] == {"capture_started": False, "ok": False}
    assert ops._exit_code("scheduled-capture", result) == 1


def test_scheduled_capture_allows_explicit_prototype_path(monkeypatch) -> None:
    monkeypatch.setattr(archive.settings, "S3_BUCKET", "geoai-archive")
    manifest = {"checks": {"ok": True}, "counts": {"artifacts": 15, "rows": 60}}
    monkeypatch.setattr(ops.archive, "run", lambda _args, client=None: manifest)

    result = ops.scheduled_capture(
        _s3_args(command="scheduled-capture"),
        object_store_client=FakeObjectStore(),
    )

    assert result["preflight"]["storage_capability"] == "prototype_only"
    assert result["capture"] is manifest
    assert result["checks"] == {"capture_started": True, "ok": True}


@pytest.mark.parametrize("all_refused", [False, True])
def test_scheduled_capture_quality_refusal_is_partial_in_logs_and_health(
    tmp_path: Path, all_refused: bool, monkeypatch, capsys
) -> None:
    calls = []

    def transport(params):
        calls.append(params)
        payload = {
            **_payload(),
            "latitude": params["latitude"],
            "longitude": params["longitude"],
        }
        if all_refused or len(calls) == 1:
            payload["daily"][MEMBER_FIELDS[2]][0] = -0.1
        return payload

    args = ops.build_parser().parse_args(
        [
            "scheduled-capture",
            "--snapshot-date",
            SNAPSHOT_DATE.isoformat(),
            "--output-dir",
            str(tmp_path),
        ]
    )
    client = OpenMeteoEnsembleClient(
        base_url=PUBLIC_ENDPOINT, apikey="", transport=transport, clock=lambda: NOW
    )
    monkeypatch.setattr(archive, "OpenMeteoEnsembleClient", lambda: client)
    assert (
        ops.main(
            [
                "scheduled-capture",
                "--snapshot-date",
                args.snapshot_date.isoformat(),
                "--output-dir",
                str(tmp_path),
            ]
        )
        == 0
    )
    result = json.loads(capsys.readouterr().out)

    assert len(calls) == 15
    assert result["checks"] == {"capture_started": True, "ok": False}
    assert result["capture"]["status"] == "partial"
    assert len(result["capture"]["rejected_zones"]) == (15 if all_refused else 1)
    report_receipt = result["capture"]["capture_report"]
    report_bytes = Path(report_receipt["output"]).read_bytes()
    assert hashlib.sha256(report_bytes).hexdigest() == report_receipt["sha256"]
    assert json.loads(report_bytes)["checks"] == {"ok": False}
    assert ops._exit_code("scheduled-capture", result) == 0
    manifest_path = tmp_path / archive._run_id(SNAPSHOT_DATE) / "manifest.json"
    assert not manifest_path.exists()
    with pytest.raises(FileNotFoundError):
        archive.load_validated_archive(manifest_path)

    health = ops.health(_health_args(tmp_path))
    assert len(calls) == 15  # The diagnostic read must remain provider-free.
    assert health["days"][-1]["status"] == "partial"
    assert health["days"][-1]["readable_artifacts"] == (0 if all_refused else 14)
    assert health["checks"]["no_integrity_errors"] is True
    assert health["checks"]["ok"] is False
    assert ops._exit_code("health", health) == 1


@pytest.mark.parametrize("failed_document", ["raw_capture", "report", "partial_receipt"])
def test_scheduled_capture_evidence_write_failure_uses_exit_two(
    tmp_path: Path, failed_document: str, monkeypatch, capsys
) -> None:
    calls = []

    def transport(params):
        calls.append(params)
        payload = {
            **_payload(),
            "latitude": params["latitude"],
            "longitude": params["longitude"],
        }
        payload["daily"][MEMBER_FIELDS[0]][0] = -0.1
        return payload

    client = OpenMeteoEnsembleClient(
        base_url=PUBLIC_ENDPOINT, apikey="", transport=transport, clock=lambda: NOW
    )
    monkeypatch.setattr(archive, "OpenMeteoEnsembleClient", lambda: client)
    publish = archive._publish_json_if_absent
    failed_paths = []

    def publish_or_fail(path, payload):
        name = Path(path).name
        document = (
            "partial_receipt"
            if name == "partial.json"
            else "report"
            if name.startswith("report-")
            else "raw_capture"
        )
        if document == failed_document:
            failed_paths.append(path)
            raise PermissionError("capture evidence write denied")
        return publish(path, payload)

    monkeypatch.setattr(archive, "_publish_json_if_absent", publish_or_fail)

    assert (
        ops.main(
            [
                "scheduled-capture",
                "--snapshot-date",
                SNAPSHOT_DATE.isoformat(),
                "--output-dir",
                str(tmp_path),
            ]
        )
        == 2
    )
    output = capsys.readouterr()
    assert json.loads(output.err)["status"] == "error"
    assert not output.out
    assert len(failed_paths) == 1
    assert len(calls) == (1 if failed_document == "raw_capture" else 15)
    run_dir = tmp_path / archive._run_id(SNAPSHOT_DATE)
    assert not (run_dir / "manifest.json").exists()
    assert not (run_dir / "_captures" / "partial.json").exists()


def test_scheduled_capture_http_failure_uses_exit_two(tmp_path: Path, monkeypatch, capsys) -> None:
    calls = []

    def get_response(url, **kwargs):
        calls.append((url, kwargs))
        return httpx.Response(403, request=httpx.Request("GET", url))

    client = OpenMeteoEnsembleClient(base_url=PUBLIC_ENDPOINT, apikey="", clock=lambda: NOW)
    monkeypatch.setattr(archive, "OpenMeteoEnsembleClient", lambda: client)
    monkeypatch.setattr(httpx, "get", get_response)

    assert (
        ops.main(
            [
                "scheduled-capture",
                "--snapshot-date",
                SNAPSHOT_DATE.isoformat(),
                "--output-dir",
                str(tmp_path),
            ]
        )
        == 2
    )
    output = capsys.readouterr()
    assert json.loads(output.err)["status"] == "error"
    assert not output.out
    assert len(calls) == 1
    assert not list(tmp_path.rglob("*.json"))


@pytest.mark.parametrize(
    "mutation",
    [
        {"run_id": "wrong-run"},
        {"snapshot_date_utc": "2026-07-14"},
        {"status": "complete"},
        {"checks": {"ok": True}},
        {"rejected_zones": []},
        {"rejected_zones": [{"zone_id": "unknown-zone"}]},
    ],
)
def test_health_does_not_trust_malformed_partial_capture_receipts(tmp_path: Path, mutation) -> None:
    receipt = {
        "schema_version": "forecast-ensemble-capture-report/v1",
        "run_id": archive._run_id(SNAPSHOT_DATE),
        "snapshot_date_utc": SNAPSHOT_DATE.isoformat(),
        "status": "partial",
        "checks": {"ok": False},
        "rejected_zones": [{"zone_id": archive.CANONICAL_ZONES[0]}],
        **mutation,
    }
    path = tmp_path / archive._run_id(SNAPSHOT_DATE) / "_captures" / "partial.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(receipt), encoding="utf-8")

    result = ops.health(_health_args(tmp_path))

    assert result["days"][-1]["status"] == "invalid_manifest"
    assert ops._exit_code("health", result) == 2


@pytest.mark.parametrize("storage_mode", ["local", "s3"])
@pytest.mark.parametrize("receipt_present", [False, True])
def test_health_distinguishes_missing_receipt_from_json_null(
    tmp_path: Path, storage_mode: str, receipt_present: bool, monkeypatch
) -> None:
    args = _health_args(tmp_path)
    args.storage_mode = storage_mode
    if storage_mode == "local":
        if receipt_present:
            path = tmp_path / archive._run_id(SNAPSHOT_DATE) / "_captures" / "partial.json"
            path.parent.mkdir(parents=True)
            path.write_bytes(b"null\n")
    else:
        args.s3_bucket = "geoai-archive"
        monkeypatch.setattr(archive.settings, "S3_BUCKET", "geoai-archive")
        receipt_key = (
            f"{archive.DEFAULT_S3_PREFIX}/{archive._run_id(SNAPSHOT_DATE)}/_captures/partial.json"
        )

        class ReceiptStore:
            def get_object(self, *, Bucket, Key):
                assert Bucket == "geoai-archive"
                if receipt_present and Key == receipt_key:
                    return {"Body": BytesIO(b"null\n")}
                raise FakeS3Error(404, "NoSuchKey", "missing run")

        monkeypatch.setattr(archive, "_object_store_client", lambda: ReceiptStore())

    result = ops.health(args)

    assert result["days"][-1]["status"] == (
        "invalid_manifest" if receipt_present else "missing_run"
    )
    assert result["checks"]["no_integrity_errors"] is not receipt_present
    assert result["checks"]["no_operational_errors"] is True
    assert ops._exit_code("health", result) == (2 if receipt_present else 1)


def _all_refused_archive(root: Path, *, payload_kind: str = "negative") -> dict[str, Any]:
    def transport(params):
        if payload_kind == "non_object":
            return ["provider returned a non-object"]
        payload = {
            **_payload(),
            "latitude": params["latitude"],
            "longitude": params["longitude"],
        }
        payload["daily"][MEMBER_FIELDS[0]][0] = -1.0
        return payload

    return archive.run(
        _archive_args(root),
        client=OpenMeteoEnsembleClient(
            base_url=PUBLIC_ENDPOINT, apikey="", transport=transport, clock=lambda: NOW
        ),
    )


def _rewrite_partial_receipt(root: Path, receipt: dict[str, Any]) -> None:
    report = {key: value for key, value in receipt.items() if key != "capture_report"}
    raw = archive._json_bytes(report)
    Path(receipt["capture_report"]["output"]).write_bytes(raw)
    receipt["capture_report"]["sha256"] = hashlib.sha256(raw).hexdigest()
    path = root / archive._run_id(SNAPSHOT_DATE) / "_captures" / "partial.json"
    path.write_bytes(archive._json_bytes(receipt))


@pytest.mark.parametrize("payload_kind", ["negative", "non_object"])
def test_health_verifies_all_refused_evidence_without_normalizing_payload(
    tmp_path: Path, payload_kind: str, monkeypatch
) -> None:
    receipt = _all_refused_archive(tmp_path, payload_kind=payload_kind)
    monkeypatch.setattr(
        archive, "run", lambda *_args, **_kwargs: pytest.fail("health started a capture")
    )
    before = {path: path.read_bytes() for path in tmp_path.rglob("*.json")}

    result = ops.health(_health_args(tmp_path))

    assert result["days"][-1]["status"] == "partial"
    assert result["checks"]["no_integrity_errors"] is True
    assert ops._exit_code("health", result) == 1
    assert {path: path.read_bytes() for path in tmp_path.rglob("*.json")} == before
    assert not receipt["artifacts"]
    assert not (tmp_path / archive._run_id(SNAPSHOT_DATE) / "manifest.json").exists()


@pytest.mark.parametrize(
    "damage",
    [
        "missing_report",
        "corrupt_report",
        "report_digest",
        "different_report",
        "missing_capture",
        "corrupt_capture",
        "capture_digest",
        "capture_schema",
        "capture_zone",
        "capture_date",
    ],
)
def test_health_rejects_broken_all_refused_evidence_chain(tmp_path: Path, damage: str) -> None:
    receipt = _all_refused_archive(tmp_path)
    report_path = Path(receipt["capture_report"]["output"])
    capture = receipt["raw_captures"][0]
    capture_path = Path(capture["output"])
    if damage == "missing_report":
        report_path.unlink()
    elif damage == "corrupt_report":
        report_path.write_bytes(b"{broken")
    elif damage == "report_digest":
        receipt["capture_report"]["sha256"] = "0" * 64
        partial = tmp_path / archive._run_id(SNAPSHOT_DATE) / "_captures" / "partial.json"
        partial.write_bytes(archive._json_bytes(receipt))
    elif damage == "different_report":
        report = json.loads(report_path.read_bytes())
        report["unexpected"] = True
        raw = archive._json_bytes(report)
        report_path.write_bytes(raw)
        receipt["capture_report"]["sha256"] = hashlib.sha256(raw).hexdigest()
        partial = tmp_path / archive._run_id(SNAPSHOT_DATE) / "_captures" / "partial.json"
        partial.write_bytes(archive._json_bytes(receipt))
    elif damage == "missing_capture":
        capture_path.unlink()
    elif damage == "corrupt_capture":
        # Valid JSON with a changed byte must still fail the retained digest.
        capture_path.write_bytes(capture_path.read_bytes() + b" ")
    else:
        if damage == "capture_digest":
            capture["sha256"] = "0" * 64
        else:
            envelope = json.loads(capture_path.read_bytes())
            key, value = {
                "capture_schema": ("schema_version", "wrong-schema"),
                "capture_zone": ("zone_id", archive.CANONICAL_ZONES[1]),
                "capture_date": ("snapshot_date_utc", "2026-07-14"),
            }[damage]
            envelope[key] = value
            raw = archive._json_bytes(envelope)
            capture_path.write_bytes(raw)
            capture["sha256"] = hashlib.sha256(raw).hexdigest()
        receipt["rejected_zones"][0]["capture"] = {
            key: capture[key] for key in ("output", "sha256")
        }
        _rewrite_partial_receipt(tmp_path, receipt)

    result = ops.health(_health_args(tmp_path))

    assert result["days"][-1]["status"] == "invalid_manifest"
    assert result["checks"]["no_integrity_errors"] is False
    assert ops._exit_code("health", result) == 2


@pytest.mark.parametrize("reference_kind", ["report", "capture"])
@pytest.mark.parametrize(
    "output",
    [
        "https://untrusted.example/capture.json",
        "s3://other-bucket/climate/forecast-ensemble/capture.json",
        "/tmp/another-run/_captures/capture.json",
        "../_captures/capture.json",
    ],
)
def test_partial_receipt_refuses_out_of_scope_references_before_reads(
    tmp_path: Path, reference_kind: str, output: str, monkeypatch
) -> None:
    receipt = _all_refused_archive(tmp_path)
    if reference_kind == "report":
        receipt["capture_report"]["output"] = output
    else:
        receipt["raw_captures"][-1]["output"] = output
        receipt["rejected_zones"][-1]["capture"]["output"] = output
    monkeypatch.setattr(
        archive, "_read_document", lambda _path: pytest.fail("unscoped reference caused a read")
    )

    with pytest.raises(ValueError, match="reference"):
        ops._validate_partial_receipt(
            receipt,
            snapshot_date=SNAPSHOT_DATE,
            root=tmp_path,
            storage_mode="local",
            bucket="",
            s3_prefix=archive.DEFAULT_S3_PREFIX,
        )


@pytest.mark.parametrize(
    "damage",
    [
        "counts",
        "accepted_artifacts",
        "duplicate_capture",
        "duplicate_rejection",
        "missing_error",
        "mismatched_rejection",
        "unbounded_captures",
        "attempt_id",
    ],
)
def test_partial_receipt_rejects_inconsistent_metadata_before_reads(
    tmp_path: Path, damage: str, monkeypatch
) -> None:
    receipt = _all_refused_archive(tmp_path)
    if damage == "counts":
        receipt["counts"]["rows"] = 1
    elif damage == "accepted_artifacts":
        receipt["artifacts"] = [{"zone_id": archive.CANONICAL_ZONES[0]}]
    elif damage == "duplicate_capture":
        receipt["raw_captures"][-1] = receipt["raw_captures"][0]
    elif damage == "duplicate_rejection":
        receipt["rejected_zones"][-1] = receipt["rejected_zones"][0]
    elif damage == "missing_error":
        receipt["rejected_zones"][0].pop("error")
    elif damage == "mismatched_rejection":
        receipt["rejected_zones"][0]["capture"] = receipt["rejected_zones"][1]["capture"]
    elif damage == "unbounded_captures":
        receipt["raw_captures"] *= 100
    else:
        receipt["attempt_id"] = "../another-run"
    monkeypatch.setattr(
        archive, "_read_document", lambda _path: pytest.fail("invalid metadata caused a read")
    )

    with pytest.raises(ValueError):
        ops._validate_partial_receipt(
            receipt,
            snapshot_date=SNAPSHOT_DATE,
            root=tmp_path,
            storage_mode="local",
            bucket="",
            s3_prefix=archive.DEFAULT_S3_PREFIX,
        )


def test_health_complete_manifest_precedes_missing_historical_capture_evidence(
    tmp_path: Path,
) -> None:
    receipt = _all_refused_archive(tmp_path)
    archive.run(_archive_args(tmp_path), client=_client([]))
    Path(receipt["capture_report"]["output"]).unlink()
    for capture in receipt["raw_captures"]:
        Path(capture["output"]).unlink()

    result = ops.health(_health_args(tmp_path))

    assert result["days"][-1]["status"] == "complete"
    assert result["checks"]["no_integrity_errors"] is True


@pytest.mark.parametrize(
    "failure_kind", ["intact", "missing_report", "missing_capture", "permission", "network"]
)
def test_health_scoped_s3_receipt_preserves_integrity_and_operational_errors(
    tmp_path: Path, failure_kind: str, monkeypatch
) -> None:
    receipt = _all_refused_archive(tmp_path)
    documents = {}
    prefix = f"{archive.DEFAULT_S3_PREFIX}/{archive._run_id(SNAPSHOT_DATE)}/_captures/"
    for capture in receipt["raw_captures"]:
        raw = Path(capture["output"]).read_bytes()
        capture["output"] = f"s3://geoai-archive/{prefix}{Path(capture['output']).name}"
        documents[capture["output"]] = raw
    for rejection, capture in zip(receipt["rejected_zones"], receipt["raw_captures"], strict=True):
        rejection["capture"] = {key: capture[key] for key in ("output", "sha256")}
    report_uri = f"s3://geoai-archive/{prefix}report-{receipt['attempt_id']}.json"
    raw_report = archive._json_bytes(
        {key: value for key, value in receipt.items() if key != "capture_report"}
    )
    receipt["capture_report"] = {
        "output": report_uri,
        "sha256": hashlib.sha256(raw_report).hexdigest(),
    }
    documents[report_uri] = raw_report
    documents[f"s3://geoai-archive/{prefix}partial.json"] = archive._json_bytes(receipt)
    reads = []
    failure_uri = (
        receipt["raw_captures"][0]["output"] if failure_kind == "missing_capture" else report_uri
    )

    class ReceiptStore:
        def get_object(self, *, Bucket, Key):
            assert Bucket == "geoai-archive"
            uri = f"s3://{Bucket}/{Key}"
            reads.append(uri)
            if uri == failure_uri and failure_kind != "intact":
                if failure_kind == "permission":
                    raise FakeS3Error(403, "AccessDenied", "access denied")
                if failure_kind == "network":
                    raise ConnectionError("store unreachable")
                raise FakeS3Error(404, "NoSuchKey", "missing evidence")
            if uri not in documents:
                raise FakeS3Error(404, "NoSuchKey", "missing run")
            return {"Body": BytesIO(documents[uri])}

    monkeypatch.setattr(archive.settings, "S3_BUCKET", "geoai-archive")
    monkeypatch.setattr(archive, "_object_store_client", lambda: ReceiptStore())
    args = _health_args(tmp_path)
    args.storage_mode = "s3"
    args.s3_bucket = "geoai-archive"

    result = ops.health(args)

    if failure_kind == "intact":
        assert result["days"][-1]["status"] == "partial"
        assert result["checks"]["no_integrity_errors"] is True
        assert result["checks"]["no_operational_errors"] is True
        assert ops._exit_code("health", result) == 1
        assert all(capture["output"] in reads for capture in receipt["raw_captures"])
        return
    integrity_error = failure_kind.startswith("missing")
    assert result["days"][-1]["status"] == ("invalid_manifest" if integrity_error else "error")
    assert result["checks"]["no_integrity_errors"] is not integrity_error
    assert result["checks"]["no_operational_errors"] is integrity_error
    assert ops._exit_code("health", result) == 2
    assert failure_uri in reads


@pytest.mark.parametrize("evidence_kind", ["receipt", "report", "capture"])
@pytest.mark.parametrize("target_scope", ["outside_run", "within_capture_folder"])
def test_health_refuses_symlinked_capture_evidence_before_target_read(
    tmp_path: Path, evidence_kind: str, target_scope: str, monkeypatch
) -> None:
    root = tmp_path / "archive"
    receipt = _all_refused_archive(root)
    partial_path = root / archive._run_id(SNAPSHOT_DATE) / "_captures" / "partial.json"
    if evidence_kind == "receipt":
        evidence_path = partial_path
    elif evidence_kind == "report":
        evidence_path = Path(receipt["capture_report"]["output"])
    else:
        evidence_path = Path(receipt["raw_captures"][0]["output"])
    target = (
        tmp_path / "outside.json"
        if target_scope == "outside_run"
        else evidence_path.parent / "noncanonical-alias.json"
    )
    evidence_path.rename(target)
    evidence_path.symlink_to(target)
    if evidence_kind == "report":
        receipt["capture_report"]["output"] = str(target)
        _rewrite_partial_receipt(root, receipt)
    elif evidence_kind == "capture":
        receipt["raw_captures"][0]["output"] = str(target)
        receipt["rejected_zones"][0]["capture"]["output"] = str(target)
        _rewrite_partial_receipt(root, receipt)
    read_document = archive._read_document
    reads = []

    def guarded_read(path):
        assert not isinstance(path, Path) or path.resolve() != target
        reads.append(path)
        return read_document(path)

    monkeypatch.setattr(archive, "_read_document", guarded_read)

    result = ops.health(_health_args(root))

    assert result["days"][-1]["status"] == "invalid_manifest"
    assert result["checks"]["no_integrity_errors"] is False
    assert ops._exit_code("health", result) == 2
    assert target not in reads


@pytest.mark.parametrize("coercion", ["false_row_count", "integer_check", "float_count"])
def test_health_report_receipt_comparison_preserves_json_types(
    tmp_path: Path, coercion: str
) -> None:
    receipt = _all_refused_archive(tmp_path)
    report_path = Path(receipt["capture_report"]["output"])
    report = json.loads(report_path.read_bytes())
    if coercion == "false_row_count":
        report["counts"]["rows"] = False
    elif coercion == "integer_check":
        report["checks"]["ok"] = 0
    else:
        report["counts"]["rejected_zones"] = 15.0
    raw = archive._json_bytes(report)
    report_path.write_bytes(raw)
    receipt["capture_report"]["sha256"] = hashlib.sha256(raw).hexdigest()
    partial_path = tmp_path / archive._run_id(SNAPSHOT_DATE) / "_captures" / "partial.json"
    partial_path.write_bytes(archive._json_bytes(receipt))

    result = ops.health(_health_args(tmp_path))

    assert result["days"][-1]["status"] == "invalid_manifest"
    assert result["checks"]["no_integrity_errors"] is False
    assert ops._exit_code("health", result) == 2
