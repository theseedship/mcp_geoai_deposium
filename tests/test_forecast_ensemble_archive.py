from __future__ import annotations

import hashlib
import importlib
import json
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, date, datetime, timedelta
from io import BytesIO
from pathlib import Path
from threading import Barrier
from urllib.parse import quote, quote_plus

import botocore
import pytest
from botocore.awsrequest import AWSRequest
from botocore.credentials import ReadOnlyCredentials
from botocore.hooks import HierarchicalEmitter
from botocore.model import ServiceId
from botocore.signers import RequestSigner

from src.cli import forecast_ensemble_archive as cli
from src.forecast_backtest import load_rows
from src.forecast_ensemble import MEMBER_FIELDS, OpenMeteoEnsembleClient
from src.utils import s3_endpoint

SNAPSHOT_DATE = date(2026, 7, 15)
NOW = datetime(2026, 7, 15, 19, 7, tzinfo=UTC)
PUBLIC_ENDPOINT = "https://ensemble-api.open-meteo.com/v1/ensemble"
COMMERCIAL_ENDPOINT = "https://customer-ensemble-api.open-meteo.com/v1/ensemble"


def _payload(*, snapshot_date: date = SNAPSHOT_DATE, days: int = 8) -> dict:
    dates = [(snapshot_date + timedelta(days=offset)).isoformat() for offset in range(1, days + 1)]
    daily = {"time": dates}
    units = {"time": "iso8601"}
    for member_id, field in enumerate(MEMBER_FIELDS):
        value = 0.1 if member_id < 11 else 1.0
        daily[field] = [value] * days
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


def _client(transport, *, apikey: str = "") -> OpenMeteoEnsembleClient:
    def coordinate_aware_transport(params):
        payload = transport(params)
        if isinstance(payload, dict):
            payload = {
                **payload,
                "latitude": params["latitude"],
                "longitude": params["longitude"],
            }
        return payload

    return OpenMeteoEnsembleClient(
        base_url=COMMERCIAL_ENDPOINT if apikey else PUBLIC_ENDPOINT,
        apikey=apikey,
        transport=coordinate_aware_transport,
        clock=lambda: NOW,
    )


def _args(
    root: Path,
    *,
    storage_mode: str = "local",
    bucket: str | None = None,
    prefix: str = cli.DEFAULT_S3_PREFIX,
) -> object:
    argv = [
        "--snapshot-date",
        SNAPSHOT_DATE.isoformat(),
        "--output-dir",
        str(root),
        "--storage-mode",
        storage_mode,
        "--s3-prefix",
        prefix,
    ]
    if bucket:
        argv.extend(["--s3-bucket", bucket])
    return cli.build_parser().parse_args(argv)


def _run_dir(root: Path) -> Path:
    return root / cli._run_id(SNAPSHOT_DATE)


def test_local_run_writes_exact_archive_gate_and_backtest_rows(tmp_path, monkeypatch):
    calls: list[dict] = []

    def transport(params):
        calls.append(dict(params))
        return _payload()

    monkeypatch.setenv("RAILWAY_GIT_COMMIT_SHA", "abc123")
    monkeypatch.setenv("RAILWAY_GIT_BRANCH", "feat/aifs-ensemble-probabilities")
    manifest = cli.run(_args(tmp_path), client=_client(transport))

    assert len(calls) == 15
    assert manifest["schema_version"] == cli.MANIFEST_SCHEMA_VERSION
    assert manifest["artifact_schema_version"] == cli.ARTIFACT_SCHEMA_VERSION
    assert manifest["run_id"] == "aifs-ensemble-forward-20260715"
    assert manifest["snapshot_date_utc"] == "2026-07-15"
    assert manifest["zones"] == list(cli.CANONICAL_ZONES)
    assert manifest["horizons"] == [1, 3, 5, 7]
    assert manifest["event_contract"] == {
        "variable": "precipitation",
        "operator": ">",
        "threshold_mm": 0.1,
        "probability_method": "raw_ensemble_relative_frequency",
        "calibration_status": "uncalibrated",
    }
    assert manifest["counts"] == {
        "zones": 15,
        "horizons": 4,
        "artifacts": 15,
        "rows": 60,
        "members_per_row": 51,
        "member_values": 3060,
        "expected_artifacts": 15,
        "expected_rows": 60,
        "expected_member_values": 3060,
    }
    assert manifest["checks"] == {
        "all_artifacts_valid": True,
        "all_rows_valid": True,
        "all_members_complete": True,
        "ok": True,
    }
    assert manifest["code_ref"] == {
        "repo": "deposium_geoai",
        "commit": "abc123",
        "branch": "feat/aifs-ensemble-probabilities",
    }
    assert len(manifest["artifacts"]) == 15
    assert all(Path(item["output"]).is_absolute() for item in manifest["artifacts"])

    artifact_path = Path(manifest["artifacts"][0]["output"])
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    assert artifact["run_id"] == manifest["run_id"]
    assert len(artifact["rows"]) == 4
    assert sum(len(row["members"]) for row in artifact["rows"]) == 204
    assert artifact["rows"][0]["members"][0] == {
        "member_id": 0,
        "source_field": "precipitation_sum",
        "daily_precipitation_mm": 0.1,
    }
    assert (_run_dir(tmp_path) / "manifest.json").is_file()

    backtest_rows = load_rows(artifact_path, kind="prediction", provider=cli.PROVIDER)
    assert len(backtest_rows) == 4
    assert all(row.probability == pytest.approx(40 / 51) for row in backtest_rows)
    assert all(row.probability_event is not None for row in backtest_rows)


def test_completed_manifest_resumes_without_provider_call(tmp_path):
    first = cli.run(_args(tmp_path), client=_client(lambda _params: _payload()))

    def provider_must_not_run(_params):
        raise AssertionError("completed archive called provider")

    second = cli.run(_args(tmp_path), client=_client(provider_must_not_run))

    assert second == first


def test_partial_failure_writes_no_manifest_and_next_run_resumes(tmp_path):
    first_calls = 0

    def fail_on_fourth(_params):
        nonlocal first_calls
        first_calls += 1
        if first_calls == 4:
            raise RuntimeError("provider unavailable")
        return _payload()

    with pytest.raises(cli.EnsembleArchiveError, match="provider unavailable"):
        cli.run(_args(tmp_path), client=_client(fail_on_fourth))

    assert first_calls == 4
    assert not (_run_dir(tmp_path) / "manifest.json").exists()
    assert len(list(_run_dir(tmp_path).glob("*.json"))) == 3

    resume_calls = 0

    def resume_transport(_params):
        nonlocal resume_calls
        resume_calls += 1
        return _payload()

    manifest = cli.run(_args(tmp_path), client=_client(resume_transport))

    assert resume_calls == 12
    assert manifest["checks"]["ok"] is True
    assert len(list(_run_dir(tmp_path).glob("*.json"))) == 16


def test_invalid_existing_artifact_fails_closed_without_overwrite_or_provider(tmp_path):
    cli.run(_args(tmp_path), client=_client(lambda _params: _payload()))
    manifest_path = _run_dir(tmp_path) / "manifest.json"
    manifest_path.unlink()
    artifact_path = sorted(_run_dir(tmp_path).glob("*.json"))[0]
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    artifact["rows"][0]["members"].pop()
    artifact_path.write_text(json.dumps(artifact, sort_keys=True), encoding="utf-8")
    invalid_bytes = artifact_path.read_bytes()

    def provider_must_not_run(_params):
        raise AssertionError("invalid existing artifact called provider")

    with pytest.raises(cli.EnsembleArchiveError, match="retain all ensemble members"):
        cli.run(_args(tmp_path), client=_client(provider_must_not_run))

    assert artifact_path.read_bytes() == invalid_bytes
    assert not manifest_path.exists()


def test_existing_artifact_rejects_tampered_zone_provenance(tmp_path):
    manifest = cli.run(_args(tmp_path), client=_client(lambda _params: _payload()))
    manifest_path = _run_dir(tmp_path) / "manifest.json"
    manifest_path.unlink()
    artifact_path = Path(manifest["artifacts"][0]["output"])
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    artifact["requested_coordinates"]["latitude"] = 0.0
    artifact["request_parameters"]["latitude"] = 0.0
    artifact_path.write_text(json.dumps(artifact, sort_keys=True), encoding="utf-8")

    with pytest.raises(cli.EnsembleArchiveError, match="requested coordinates"):
        cli.run(
            _args(tmp_path),
            client=_client(lambda _params: pytest.fail("tampered artifact called provider")),
        )

    assert not manifest_path.exists()


def test_existing_artifact_rejects_provider_coordinates_from_another_zone(tmp_path):
    manifest = cli.run(_args(tmp_path), client=_client(lambda _params: _payload()))
    manifest_path = _run_dir(tmp_path) / "manifest.json"
    manifest_path.unlink()
    artifact_path = Path(manifest["artifacts"][0]["output"])
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    artifact["provider_coordinates"]["latitude"] = -1.3
    artifact["provider_coordinates"]["longitude"] = 36.8
    artifact_path.write_text(json.dumps(artifact, sort_keys=True), encoding="utf-8")
    invalid_bytes = artifact_path.read_bytes()
    capture_bytes = {
        path: path.read_bytes() for path in (_run_dir(tmp_path) / "_captures").glob("*.json")
    }

    with pytest.raises(cli.EnsembleArchiveError, match="too far from the requested zone"):
        cli.run(
            _args(tmp_path),
            client=_client(lambda _params: pytest.fail("tampered artifact called provider")),
        )

    assert artifact_path.read_bytes() == invalid_bytes
    assert {
        path: path.read_bytes() for path in (_run_dir(tmp_path) / "_captures").glob("*.json")
    } == capture_bytes
    assert not manifest_path.exists()


def test_completed_manifest_detects_structurally_valid_artifact_digest_drift(tmp_path):
    manifest = cli.run(_args(tmp_path), client=_client(lambda _params: _payload()))
    artifact_path = Path(manifest["artifacts"][0]["output"])
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    artifact["retrieved_at_utc"] = "2026-07-15T19:08:00Z"
    artifact_path.write_text(json.dumps(artifact, sort_keys=True), encoding="utf-8")

    def provider_must_not_run(_params):
        raise AssertionError("digest validation called provider")

    with pytest.raises(cli.EnsembleArchiveError, match="digest drift detected"):
        cli.run(_args(tmp_path), client=_client(provider_must_not_run))


def test_s3_paths_use_shared_helpers_and_manifest_is_written_last(tmp_path, monkeypatch):
    monkeypatch.setattr(cli.settings, "S3_BUCKET", "geoai-archive")
    store: dict[str, dict] = {}
    writes: list[str] = []

    def read_document(path):
        key = str(path)
        if key not in store:
            raise FileNotFoundError(key)
        payload = json.loads(json.dumps(store[key]))
        return payload, cli._json_bytes(payload)

    def publish_json_if_absent(path, payload):
        key = str(path)
        if key in store:
            return False
        writes.append(key)
        store[key] = json.loads(json.dumps(payload))
        return True

    monkeypatch.setattr(cli, "_read_document", read_document)
    monkeypatch.setattr(cli, "_publish_json_if_absent", publish_json_if_absent)
    calls = 0

    def transport(_params):
        nonlocal calls
        calls += 1
        return _payload()

    manifest = cli.run(
        _args(
            tmp_path,
            storage_mode="s3",
            bucket="geoai-archive",
            prefix="climate/forecast-ensemble",
        ),
        client=_client(transport),
    )

    run_id = cli._run_id(SNAPSHOT_DATE)
    manifest_uri = f"s3://geoai-archive/climate/runs/{run_id}/manifest.json"
    assert calls == 15
    assert len(store) == 31
    assert writes[-1] == manifest_uri
    assert writes.count(manifest_uri) == 1
    assert manifest["artifacts"][0]["output"].startswith(
        f"s3://geoai-archive/climate/forecast-ensemble/{run_id}/"
    )
    assert store[manifest_uri]["checks"]["ok"] is True

    def provider_must_not_run(_params):
        raise AssertionError("S3 completed archive called provider")

    resumed = cli.run(
        _args(tmp_path, storage_mode="s3", bucket="geoai-archive"),
        client=_client(provider_must_not_run),
    )
    assert resumed == manifest
    assert writes.count(manifest_uri) == 1


@pytest.mark.parametrize("rejected_index", [0, 7])
def test_data_quality_rejection_retains_negative_value_and_resumes_one_zone(
    tmp_path, rejected_index
):
    calls = 0

    def transport(_params):
        nonlocal calls
        payload = _payload()
        if calls == rejected_index:
            payload["daily"][MEMBER_FIELDS[0]][0] = -0.1
        calls += 1
        return payload

    report = cli.run(_args(tmp_path), client=_client(transport))
    assert calls == 15
    assert report["schema_version"] == cli.CAPTURE_REPORT_SCHEMA_VERSION
    assert report["status"] == "partial"
    assert report["checks"]["ok"] is False
    assert report["counts"]["artifacts"] == 14
    assert report["counts"]["raw_captures"] == 15
    assert len(report["artifacts"]) == 14
    assert not (_run_dir(tmp_path) / "manifest.json").exists()
    rejection = report["rejected_zones"][0]
    assert rejection["zone_id"] == cli.CANONICAL_ZONES[rejected_index]
    capture_path = Path(rejection["capture"]["output"])
    raw = capture_path.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == rejection["capture"]["sha256"]
    capture = json.loads(raw)
    assert capture["payload"]["daily"][MEMBER_FIELDS[0]][0] == -0.1
    assert capture["schema_version"] == "forecast-ensemble-capture/v1"
    report_path = Path(report["capture_report"]["output"])
    assert report_path.parent == _run_dir(tmp_path) / "_captures"
    assert (
        hashlib.sha256(report_path.read_bytes()).hexdigest() == report["capture_report"]["sha256"]
    )
    saved_report = json.loads(report_path.read_bytes())
    assert saved_report == {key: value for key, value in report.items() if key != "capture_report"}
    receipts = {path: path.read_bytes() for path in report_path.parent.glob("*.json")}
    resume_calls = 0

    def valid_transport(_params):
        nonlocal resume_calls
        resume_calls += 1
        return _payload()

    manifest = cli.run(_args(tmp_path), client=_client(valid_transport))
    assert resume_calls == 1
    assert manifest["checks"]["ok"] is True
    assert manifest["counts"]["artifacts"] == 15
    assert manifest["counts"]["member_values"] == 3060
    assert all(path.read_bytes() == raw for path, raw in receipts.items())
    assert len(list(report_path.parent.glob("*.json"))) == len(receipts) + 1
    artifact = json.loads(Path(manifest["artifacts"][rejected_index]["output"]).read_bytes())
    assert len(artifact["rows"][0]["members"]) == 51
    assert artifact["rows"][0]["probability"] == pytest.approx(40 / 51)


@pytest.mark.parametrize("rejected_index", [0, 7])
@pytest.mark.parametrize("field", ["latitude", "longitude"])
def test_far_fresh_coordinates_retain_capture_and_continue_other_zones(
    tmp_path, rejected_index, field
):
    calls = []

    def transport(params):
        payload = {
            **_payload(),
            "latitude": params["latitude"],
            "longitude": params["longitude"],
        }
        if len(calls) == rejected_index:
            payload[field] += 0.250001
        calls.append(dict(params))
        return payload

    client = OpenMeteoEnsembleClient(transport=transport, clock=lambda: NOW)
    report = cli.run(_args(tmp_path), client=client)

    assert len(calls) == 15
    assert report["status"] == "partial"
    assert report["checks"]["ok"] is False
    assert report["counts"]["artifacts"] == 14
    assert report["counts"]["raw_captures"] == 15
    assert not (_run_dir(tmp_path) / "manifest.json").exists()
    assert len(report["rejected_zones"]) == 1
    rejection = report["rejected_zones"][0]
    assert rejection["zone_id"] == cli.CANONICAL_ZONES[rejected_index]
    assert "resolved coordinates" in rejection["error"]
    assert "too far" in rejection["error"]
    assert {artifact["zone_id"] for artifact in report["artifacts"]} == (
        set(cli.CANONICAL_ZONES) - {rejection["zone_id"]}
    )
    capture_path = Path(rejection["capture"]["output"])
    raw = capture_path.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == rejection["capture"]["sha256"]
    capture = json.loads(raw)
    assert capture["payload"][field] == calls[rejected_index][field] + 0.250001
    assert len(capture["payload"]["daily"]) == 52
    report_path = Path(report["capture_report"]["output"])
    assert (
        hashlib.sha256(report_path.read_bytes()).hexdigest() == report["capture_report"]["sha256"]
    )
    assert json.loads(report_path.read_bytes()) == {
        key: value for key, value in report.items() if key != "capture_report"
    }


def test_s3_data_quality_rejection_preserves_captures_without_credentials(tmp_path, monkeypatch):
    monkeypatch.setattr(cli.settings, "S3_BUCKET", "geoai-archive")
    secret = "ensemble-private-key+/?"
    store = {}

    def read_document(path):
        if str(path) not in store:
            raise FileNotFoundError(str(path))
        raw = store[str(path)]
        return json.loads(raw), raw

    def publish(path, payload):
        if str(path) in store:
            return False
        store[str(path)] = cli._json_bytes(payload)
        return True

    def transport(_params):
        payload = _payload()
        payload["daily"][MEMBER_FIELDS[0]][0] = -0.1
        return payload

    monkeypatch.setattr(cli, "_read_document", read_document)
    monkeypatch.setattr(cli, "_publish_json_if_absent", publish)
    report = cli.run(
        _args(tmp_path, storage_mode="s3", bucket="geoai-archive"),
        client=_client(transport, apikey=secret),
    )
    assert report["counts"]["artifacts"] == 0
    assert len(report["rejected_zones"]) == 15
    prefix = f"s3://geoai-archive/climate/forecast-ensemble/{cli._run_id(SNAPSHOT_DATE)}/_captures/"
    assert len(store) == 17
    assert all(path.startswith(prefix) for path in store)
    assert all(secret.encode() not in raw and b'"apikey"' not in raw for raw in store.values())
    assert all(
        hashlib.sha256(store[item["output"]]).hexdigest() == item["sha256"]
        for item in report["raw_captures"]
    )
    assert not any(path.endswith("manifest.json") for path in store)


@pytest.mark.parametrize("invalid", [False, True])
def test_secret_tainted_raw_response_fails_closed(tmp_path, invalid):
    secret = "ensemble-secret-payload"

    def transport(_params):
        payload = _payload()
        payload["unexpected_secret"] = secret
        if invalid:
            payload["daily"][MEMBER_FIELDS[0]][0] = -0.1
        return payload

    with pytest.raises(cli.EnsembleArchiveError, match="contains configured credentials"):
        cli.run(_args(tmp_path), client=_client(transport, apikey=secret))
    assert not list(_run_dir(tmp_path).rglob("*.json"))


def test_oversized_capture_fails_before_publish(tmp_path):
    def transport(_params):
        payload = _payload()
        payload["extra"] = "x" * cli.MAX_ARCHIVE_JSON_BYTES
        return payload

    with pytest.raises(cli.EnsembleArchiveError, match="maximum allowed size"):
        cli.run(_args(tmp_path), client=_client(transport))
    assert not list(_run_dir(tmp_path).rglob("*.json"))


def test_capture_publish_failure_aborts_before_quality_rejection(tmp_path, monkeypatch):
    calls = 0

    def transport(_params):
        nonlocal calls
        calls += 1
        payload = _payload()
        payload["daily"][MEMBER_FIELDS[0]][0] = -0.1
        return payload

    def publish(_path, _payload):
        raise OSError("capture store unavailable")

    monkeypatch.setattr(cli, "_publish_json_if_absent", publish)
    with pytest.raises(cli.EnsembleArchiveError, match="capture store unavailable"):
        cli.run(_args(tmp_path), client=_client(transport))
    assert calls == 1
    assert not list(_run_dir(tmp_path).rglob("*.json"))


def test_capture_collision_with_different_bytes_fails_closed(tmp_path, monkeypatch):
    def publish(path, _payload):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}\n", encoding="utf-8")
        return False

    monkeypatch.setattr(cli, "_publish_json_if_absent", publish)
    with pytest.raises(cli.EnsembleArchiveError, match="collision or stored bytes mismatch"):
        cli.run(_args(tmp_path), client=_client(lambda _params: _payload()))
    assert not (_run_dir(tmp_path) / "manifest.json").exists()


def test_quality_error_without_retained_capture_fails_closed(tmp_path):
    class UncapturedClient:
        base_url = PUBLIC_ENDPOINT

        def fetch_snapshot(self, **_kwargs):
            raise cli.EnsembleDataQualityError("invalid provider content")

    with pytest.raises(cli.EnsembleArchiveError, match="no retained capture"):
        cli.run(_args(tmp_path), client=UncapturedClient())
    assert not list(_run_dir(tmp_path).rglob("*.json"))


def test_partial_cli_prints_report_and_returns_one(tmp_path, monkeypatch, capsys):
    def transport(_params):
        payload = _payload()
        payload["daily"][MEMBER_FIELDS[0]][0] = -0.1
        return payload

    monkeypatch.setattr(cli, "OpenMeteoEnsembleClient", lambda: _client(transport))
    status = cli.main(["--snapshot-date", SNAPSHOT_DATE.isoformat(), "--output-dir", str(tmp_path)])
    assert status == 1
    assert json.loads(capsys.readouterr().out)["status"] == "partial"


def test_repeated_partial_attempt_retains_first_receipt_and_adds_unique_report(tmp_path):
    calls = 0

    def transport(_params):
        nonlocal calls
        payload = _payload()
        if calls in {0, 15}:
            payload["daily"][MEMBER_FIELDS[0]][0] = -0.1
        calls += 1
        return payload

    first = cli.run(_args(tmp_path), client=_client(transport))
    capture_dir = _run_dir(tmp_path) / "_captures"
    original_receipts = {path: path.read_bytes() for path in capture_dir.glob("*.json")}
    second = cli.run(_args(tmp_path), client=_client(transport))
    assert calls == 16
    assert second["status"] == "partial"
    assert second["counts"]["artifacts"] == 14
    assert second["counts"]["raw_captures"] == 1
    assert first["attempt_id"] != second["attempt_id"]
    assert first["capture_report"]["output"] != second["capture_report"]["output"]
    assert all(path.read_bytes() == raw for path, raw in original_receipts.items())
    assert len(list(capture_dir.glob("report-*.json"))) == 2
    assert not (_run_dir(tmp_path) / "manifest.json").exists()


def test_cli_failure_redacts_raw_and_encoded_api_key(tmp_path, monkeypatch, capsys):
    secret = "ensemble-secret+/?"

    def failing_transport(_params):
        raise RuntimeError(
            f"failed token={secret} encoded={quote(secret, safe='')} plus={quote_plus(secret, safe='')}"
        )

    client = _client(failing_transport, apikey=secret)
    with pytest.raises(cli.EnsembleArchiveError) as caught:
        cli.run(_args(tmp_path), client=client)
    error = str(caught.value)
    assert secret not in error
    assert quote(secret, safe="") not in error
    assert quote_plus(secret, safe="") not in error
    assert "[REDACTED]" in error
    assert not (_run_dir(tmp_path) / "manifest.json").exists()

    monkeypatch.setattr(cli, "OpenMeteoEnsembleClient", lambda: client)
    exit_code = cli.main(
        [
            "--snapshot-date",
            SNAPSHOT_DATE.isoformat(),
            "--output-dir",
            str(tmp_path / "main"),
        ]
    )
    stderr = capsys.readouterr().err
    assert exit_code == 1
    assert secret not in stderr
    assert quote(secret, safe="") not in stderr
    assert quote_plus(secret, safe="") not in stderr
    assert "[REDACTED]" in stderr


def test_error_redaction_survives_malformed_s3_endpoint(monkeypatch):
    secret = "session-secret"
    endpoint_secret = "endpoint-secret"
    monkeypatch.setattr(
        cli.settings,
        "S3_ENDPOINT",
        f"https://{endpoint_secret}@[malformed",
    )
    monkeypatch.setenv("AWS_SESSION_TOKEN", secret)

    message = cli._safe_error_message(
        RuntimeError(f"failure token={secret} endpoint={endpoint_secret}")
    )

    assert secret not in message
    assert endpoint_secret not in message
    assert "[REDACTED]" in message


def test_an_account_name_that_prefixes_its_password_does_not_leave_the_tail(monkeypatch):
    """`_secret_values` is a SET, so it used to be redacted in hash order.

    These secrets overlap: this repo's own store uses account `deposium-dev` with password
    `deposium-dev-secret`. Redacting the short one first rewrote the password to
    `[REDACTED]-secret` and published the tail — and invisibly, because `_ensure_secret_free`
    then finds no WHOLE secret in the payload and waves it through.
    """
    monkeypatch.setattr(cli.settings, "S3_ENDPOINT", "http://127.0.0.1:9000")
    monkeypatch.setattr(cli.settings, "S3_ACCESS_KEY", "deposium-dev")
    monkeypatch.setattr(cli.settings, "S3_SECRET_KEY", "deposium-dev-secret")
    monkeypatch.delenv("AWS_ACCESS_KEY_ID", raising=False)
    monkeypatch.delenv("AWS_SECRET_ACCESS_KEY", raising=False)
    monkeypatch.delenv("AWS_SESSION_TOKEN", raising=False)
    monkeypatch.delenv("AWS_SECURITY_TOKEN", raising=False)

    message = cli._safe_error_message(RuntimeError("auth failed for deposium-dev-secret"))

    assert message == "auth failed for [REDACTED]"
    assert "-secret" not in message


def test_a_configured_store_key_below_the_url_floor_is_still_redacted(monkeypatch):
    """`main` redacted every non-empty configured key; the shared secret set must too.

    `configured_store_secrets` grew a length floor to stop a `?token=1` from blanking every
    digit of a diagnostic, and applied it to the explicitly configured `S3_ACCESS_KEY` /
    `S3_SECRET_KEY` as well. Rewiring `_secret_values` onto that set then dropped a seven-character
    store key from the redaction, invisibly: `_ensure_secret_free` checks the same set, so it found
    nothing to object to and waved the payload through.
    """
    access_key = "acc-key"  # seven characters, below `_MIN_SECRET_LENGTH`
    secret_key = "s3cr3t7"  # seven characters, below `_MIN_SECRET_LENGTH`
    monkeypatch.setattr(cli.settings, "S3_ENDPOINT", "http://127.0.0.1:9000")
    monkeypatch.setattr(cli.settings, "S3_ACCESS_KEY", access_key)
    monkeypatch.setattr(cli.settings, "S3_SECRET_KEY", secret_key)
    monkeypatch.setattr(cli.settings, "STAC_URL", "")
    monkeypatch.delenv("AWS_ACCESS_KEY_ID", raising=False)
    monkeypatch.delenv("AWS_SECRET_ACCESS_KEY", raising=False)
    monkeypatch.delenv("AWS_SESSION_TOKEN", raising=False)
    monkeypatch.delenv("AWS_SECURITY_TOKEN", raising=False)

    assert {access_key, secret_key} <= cli._secret_values(None)

    message = cli._safe_error_message(
        RuntimeError(f"AccessDenied for {access_key}: signature with {secret_key} rejected")
    )
    assert message == "AccessDenied for [REDACTED]: signature with [REDACTED] rejected"

    with pytest.raises(ValueError, match="configured credentials"):
        cli._ensure_secret_free({"error": f"signature with {secret_key} rejected"}, None)
    with pytest.raises(ValueError, match="configured credentials"):
        cli._ensure_secret_free({"error": f"AccessDenied for {access_key}"}, None)
    # The gate is discriminating, not merely armed: a clean payload still passes.
    cli._ensure_secret_free({"error": "AccessDenied for [REDACTED]"}, None)


def test_corrupt_remote_artifact_is_not_classified_as_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(cli.settings, "S3_BUCKET", "geoai-archive")
    run_id = cli._run_id(SNAPSHOT_DATE)
    first_artifact = cli._artifact_path(
        storage_mode="s3",
        output_dir=tmp_path,
        run_id=run_id,
        bucket="geoai-archive",
        s3_prefix=cli.DEFAULT_S3_PREFIX,
        zone_id=cli.CANONICAL_ZONES[0],
        snapshot_date=SNAPSHOT_DATE,
    )

    def read_document(path):
        if str(path) == str(first_artifact):
            raise json.JSONDecodeError("invalid", "{", 1)
        raise FileNotFoundError(str(path))

    monkeypatch.setattr(cli, "_read_document", read_document)

    with pytest.raises(cli.EnsembleArchiveError, match="invalid"):
        cli.run(
            _args(tmp_path, storage_mode="s3", bucket="geoai-archive"),
            client=_client(lambda _params: _payload()),
        )


def test_local_publish_is_atomic_and_create_only(tmp_path):
    target = tmp_path / "artifact.json"
    payloads = ({"writer": "one", "values": list(range(1000))}, {"writer": "two"})

    with ThreadPoolExecutor(max_workers=2) as executor:
        outcomes = list(
            executor.map(lambda payload: cli._publish_json_if_absent(target, payload), payloads)
        )

    assert sorted(outcomes) == [False, True]
    assert json.loads(target.read_text(encoding="utf-8")) in payloads
    assert list(tmp_path.glob(".*.tmp")) == []


def test_local_directory_creation_fsyncs_each_new_parent(tmp_path, monkeypatch):
    fsynced: list[Path] = []
    monkeypatch.setattr(cli, "_fsync_directory", fsynced.append)
    target = tmp_path / "archive" / "run"

    cli._mkdir_durable(target)

    assert target.is_dir()
    assert fsynced == [tmp_path, tmp_path / "archive"]


def test_s3_publish_uses_create_only_precondition_and_preserves_winner(monkeypatch):
    class ConditionalConflict(RuntimeError):
        response = {
            "ResponseMetadata": {"HTTPStatusCode": 412},
            "Error": {"Code": "PreconditionFailed"},
        }

    class FakeObjectStore:
        def __init__(self):
            self.objects = {}
            self.requests = []

        def put_object(self, **request):
            self.requests.append(request)
            identity = (request["Bucket"], request["Key"])
            if identity in self.objects:
                raise ConditionalConflict()
            self.objects[identity] = request["Body"]

        def get_object(self, **request):
            return {"Body": BytesIO(self.objects[(request["Bucket"], request["Key"])])}

    store = FakeObjectStore()
    monkeypatch.setattr(cli, "_object_store_client", lambda: store)
    path = "s3://geoai-archive/climate/run/artifact.json"

    assert cli._publish_json_if_absent(path, {"writer": "one"}) is True
    assert cli._publish_json_if_absent(path, {"writer": "two"}) is False
    assert all(request["IfNoneMatch"] == "*" for request in store.requests)
    assert cli._read_existing(path) == {"writer": "one"}


def test_s3_manifest_loader_rejects_cross_bucket_before_read(monkeypatch):
    monkeypatch.setattr(cli.settings, "S3_BUCKET", "authorized")
    monkeypatch.setattr(
        cli,
        "_read_document",
        lambda _path: pytest.fail("cross-bucket manifest triggered an object read"),
    )

    with pytest.raises(ValueError, match="configured S3 bucket"):
        cli.load_validated_archive(
            "s3://other/climate/runs/aifs-ensemble-forward-20260715/manifest.json"
        )


def test_s3_writer_rejects_bucket_and_prefix_overrides_before_provider(tmp_path, monkeypatch):
    monkeypatch.setattr(cli.settings, "S3_BUCKET", "authorized")
    client = _client(lambda _params: pytest.fail("invalid S3 scope called provider"))

    with pytest.raises(cli.EnsembleArchiveError, match="configured S3 bucket"):
        cli.run(
            _args(tmp_path, storage_mode="s3", bucket="other"),
            client=client,
        )

    with pytest.raises(cli.EnsembleArchiveError, match="canonical S3 artifact prefix"):
        cli.run(
            _args(
                tmp_path,
                storage_mode="s3",
                bucket="authorized",
                prefix="another/namespace",
            ),
            client=client,
        )


def test_archive_json_reads_are_bounded(tmp_path, monkeypatch):
    oversized = b"{" + b" " * cli.MAX_ARCHIVE_JSON_BYTES + b"}"
    local = tmp_path / "oversized.json"
    local.write_bytes(oversized)

    with pytest.raises(ValueError, match="maximum allowed size"):
        cli._read_document(local)

    body = BytesIO(oversized)

    class FakeObjectStore:
        def get_object(self, **_request):
            return {"Body": body, "ContentLength": len(oversized)}

    monkeypatch.setattr(cli, "_object_store_client", lambda: FakeObjectStore())
    with pytest.raises(ValueError, match="maximum allowed size"):
        cli._read_document("s3://archive/climate/run/oversized.json")
    assert body.closed


def test_concurrent_local_runs_converge_on_one_valid_manifest(tmp_path):
    barrier = Barrier(2)

    def concurrent_client(retrieved_at):
        first_call = True

        def transport(params):
            nonlocal first_call
            if first_call:
                first_call = False
                barrier.wait(timeout=5)
            return {
                **_payload(),
                "latitude": params["latitude"],
                "longitude": params["longitude"],
            }

        return OpenMeteoEnsembleClient(
            transport=transport,
            clock=lambda: retrieved_at,
        )

    clients = (
        concurrent_client(NOW),
        concurrent_client(NOW + timedelta(minutes=1)),
    )
    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(
            executor.map(lambda client: cli.run(_args(tmp_path), client=client), clients)
        )

    assert results[0] == results[1]
    loaded = cli.load_validated_archive(_run_dir(tmp_path) / "manifest.json")
    assert len(loaded.rows) == 60


def test_existing_manifest_cannot_echo_configured_secret(tmp_path):
    secret = "ensemble-secret"
    client = _client(lambda _params: _payload(), apikey=secret)
    manifest = cli.run(_args(tmp_path), client=client)
    manifest_path = _run_dir(tmp_path) / "manifest.json"
    manifest["unexpected_debug_value"] = secret
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(cli.EnsembleArchiveError, match="configured credentials") as caught:
        cli.run(_args(tmp_path), client=client)

    assert secret not in str(caught.value)


def test_past_incomplete_archive_is_explicitly_not_resumable(tmp_path):
    yesterday = SNAPSHOT_DATE
    args = _args(tmp_path)
    args.snapshot_date = yesterday
    client = OpenMeteoEnsembleClient(
        transport=lambda _params: pytest.fail("expired run called provider"),
        clock=lambda: datetime.combine(
            yesterday + timedelta(days=1),
            datetime.min.time().replace(hour=19, minute=7),
            tzinfo=UTC,
        ),
    )

    with pytest.raises(
        cli.EnsembleArchiveError, match="snapshot_date must equal the current UTC date"
    ):
        cli.run(args, client=client)


def test_validated_loader_rejects_member_probability_tampering_even_with_updated_digest(tmp_path):
    manifest = cli.run(_args(tmp_path), client=_client(lambda _params: _payload()))
    manifest_path = _run_dir(tmp_path) / "manifest.json"
    artifact_path = Path(manifest["artifacts"][0]["output"])
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    artifact["rows"][0]["probability"] = 0.0
    artifact_path.write_text(json.dumps(artifact, sort_keys=True), encoding="utf-8")
    manifest["artifacts"][0]["sha256"] = cli._digest(artifact)
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")

    with pytest.raises(ValueError, match="probability is inconsistent"):
        cli.load_validated_archive(manifest_path)


# --------------------------------------------------------------------------- #
# The boto3 client never receives the configured endpoint verbatim
# --------------------------------------------------------------------------- #


class _RecordingBoto3:
    """Just enough of the boto3 module to capture what `client()` was asked for."""

    def __init__(self) -> None:
        self.kwargs: dict[str, object] | None = None
        self.events = _RecordingEvents()

    def client(self, _service: str, **kwargs: object) -> object:
        self.kwargs = kwargs
        return _RecordingBotoClient(self.events)


class _RecordingEvents:
    def __init__(self) -> None:
        self.registrations: list[tuple[str, object]] = []

    def register_last(self, event: str, handler: object) -> None:
        self.registrations.append((event, handler))


class _RecordingBotoClient:
    def __init__(self, events: _RecordingEvents) -> None:
        self.meta = type("Meta", (), {"events": events})()


@pytest.fixture
def _fake_boto3(monkeypatch):
    recorder = _RecordingBoto3()
    real_import = importlib.import_module

    def fake_import(name: str, *args, **kwargs):
        return recorder if name == "boto3" else real_import(name, *args, **kwargs)

    monkeypatch.setattr(cli.importlib, "import_module", fake_import)
    return recorder


@pytest.mark.parametrize(
    "endpoint",
    [
        # botocore quotes `endpoint_url` back in EndpointConnectionError, so every one of these
        # put its userinfo into a traceback raised out of `_read_document`. Measured on
        # botocore 1.43.57:
        #   Could not connect to the endpoint URL:
        #   "http://rustfs-admin:sup3r-s3cret@127.0.0.1:1/deposium/x.json"
        "http://rustfs-admin:sup3r-s3cret@127.0.0.1:9000",
        "rustfs-admin:sup3r-s3cret@127.0.0.1:9000",
        "http://rustfs-admin:sup3r-s3cret%40127.0.0.1:9000",
        # No "@" at all: the PASSWORD reads as the port and goes over as `endpoint_url`.
        "http://rustfs-admin:sup3r-s3cret",
        # Unparseable for other reasons. `None` is not an acceptable answer for any of them:
        # boto3's default endpoint is AWS, and the store credentials would be signed to Amazon.
        "http://[::1:9000",
        "http://127.0.0.1:99999",
        "ftp://127.0.0.1:9000",
        "   ",
    ],
)
def test_a_refused_endpoint_never_reaches_boto3(endpoint, monkeypatch, _fake_boto3):
    monkeypatch.setattr(cli.settings, "S3_ENDPOINT", endpoint)

    with pytest.raises(cli.S3EndpointConfigurationError) as raised:
        cli._object_store_client()

    assert _fake_boto3.kwargs is None, "a refused endpoint must not construct a client"
    assert "sup3r-s3cret" not in str(raised.value)
    assert "rustfs-admin" not in str(raised.value)


@pytest.mark.parametrize(
    ("endpoint", "expected"),
    [
        ("http://rustfs:9000", "http://rustfs:9000"),
        ("HTTPS://mesh.internal:9000", "https://mesh.internal:9000"),
        ("http://[::1]:9000", "http://[::1]:9000"),
        # Unset stays unset: boto3 then uses its AWS default on purpose.
        ("", None),
        (None, None),
    ],
)
def test_a_configured_endpoint_reaches_boto3_sanitized(
    endpoint, expected, monkeypatch, _fake_boto3
):
    monkeypatch.setattr(cli.settings, "S3_ENDPOINT", endpoint)

    cli._object_store_client()

    assert _fake_boto3.kwargs is not None
    assert _fake_boto3.kwargs["endpoint_url"] == expected
    assert [event for event, _handler in _fake_boto3.events.registrations] == ["before-sign.s3"]


class _RotatingCredentials:
    def __init__(self, *generations: tuple[str, str, str | None]) -> None:
        self.generations = generations
        self.calls = 0

    def get_frozen_credentials(self) -> ReadOnlyCredentials:
        generation = self.generations[min(self.calls, len(self.generations) - 1)]
        self.calls += 1
        return ReadOnlyCredentials(*generation, account_id=None)


def _request_signer(
    credentials,
    *,
    signature_version="v4",
    before_capture=None,
) -> RequestSigner:
    emitter = HierarchicalEmitter()
    if before_capture is not None:
        emitter.register("before-sign.s3", before_capture)
    emitter.register_last(
        "before-sign.s3",
        lambda **kwargs: cli._capture_effective_store_credentials(
            **kwargs,
            unsigned_signature=botocore.UNSIGNED,
            auth_type_maps=botocore.auth.AUTH_TYPE_MAPS,
        ),
    )
    signer = RequestSigner(
        ServiceId("s3"),
        "us-east-1",
        "s3",
        signature_version,
        credentials,
        emitter,
    )
    # RequestSigner deliberately stores only a weak proxy to the emitter; a real client owns the
    # strong reference through ``client.meta.events``. Keep the same lifetime in this unit harness.
    signer._test_event_emitter = emitter
    return signer


@pytest.fixture
def _clean_effective_store_credentials():
    with s3_endpoint._effective_store_values_lock:
        s3_endpoint._effective_store_values.clear()
    yield
    with s3_endpoint._effective_store_values_lock:
        s3_endpoint._effective_store_values.clear()


def test_real_request_signer_uses_exact_observed_generation_and_refreshes_on_retry(
    _clean_effective_store_credentials,
):
    first = ("AKIAFIRSTGENERATION1", "first-secret-abcdefgh", "first-token-abcdefgh")
    second = ("AKIASECONDGENERATION", "second-secret-abcdefg", "second-token-abcdefg")
    credentials = _RotatingCredentials(first, second)
    signer = _request_signer(credentials)
    request = AWSRequest(method="GET", url="https://s3.amazonaws.com/bucket/key")

    signer.sign("GetObject", request)

    assert credentials.calls == 1, "the immutable envelope must prevent a second refresh"
    assert f"Credential={first[0]}/" in request.headers["Authorization"]
    assert request.headers["X-Amz-Security-Token"] == first[2]
    assert set(first) <= set(s3_endpoint.effective_store_values())
    signing_context_repr = repr(request.context)
    assert all(value not in signing_context_repr for value in first)

    signer.sign("GetObject", request)

    assert credentials.calls == 2, "a retry must return to the dynamic source, not pin attempt 1"
    assert f"Credential={second[0]}/" in request.headers["Authorization"]
    assert request.headers["X-Amz-Security-Token"] == second[2]
    assert set(first + second) <= set(s3_endpoint.effective_store_values())

    before_redaction = credentials.calls
    message = cli._safe_error_message(RuntimeError(f"denied {first[0]} then {second[0]}"))
    assert first[0] not in message
    assert second[0] not in message
    assert credentials.calls == before_redaction, "redaction must never refresh the provider"


def test_real_s3express_signer_uses_exact_identity_cache_generation_on_each_retry(
    _clean_effective_store_credentials,
):
    default = _RotatingCredentials(("AKIADEFAULT00000001", "default-secret-abcdefgh", None))
    first = ("AKIAEXPRESS00000001", "express-secret-abcdef1", "express-token-abcdef1")
    second = ("AKIAEXPRESS00000002", "express-secret-abcdef2", "express-token-abcdef2")

    class RotatingIdentityCache:
        def __init__(self) -> None:
            self.calls = 0
            self.sources = [_RotatingCredentials(first), _RotatingCredentials(second)]

        def get_credentials(self, cache_key):
            assert cache_key == "directory-bucket"
            source = self.sources[min(self.calls, len(self.sources) - 1)]
            self.calls += 1
            return source

    identity_cache = RotatingIdentityCache()

    def inject_identity_cache(request, **_kwargs):
        # This is the ordering used by botocore's S3ExpressIdentityResolver: the cache does not
        # exist before the broad before-sign event and our register_last handler must run after it.
        request.context["signing"] = {
            "identity_cache": identity_cache,
            "cache_key": "directory-bucket",
        }

    signer = _request_signer(
        default,
        signature_version="v4-s3express",
        before_capture=inject_identity_cache,
    )
    request = AWSRequest(method="GET", url="https://s3.amazonaws.com/bucket/key")

    signer.sign("GetObject", request)

    assert identity_cache.calls == 1
    assert identity_cache.sources[0].calls == 1
    assert default.calls == 0
    assert f"Credential={first[0]}/" in request.headers["Authorization"]
    assert request.headers["x-amz-s3session-token"] == first[2]
    assert set(first) <= set(s3_endpoint.effective_store_values())
    assert all(value not in repr(request.context) for value in first)

    # Botocore leaves an existing S3 Express token header untouched. Its retry preparation removes
    # attempt headers before signing again; mirror that boundary while retaining request.context.
    del request.headers["x-amz-s3session-token"]
    signer.sign("GetObject", request)

    assert identity_cache.calls == 2
    assert identity_cache.sources[1].calls == 1
    assert f"Credential={second[0]}/" in request.headers["Authorization"]
    assert request.headers["x-amz-s3session-token"] == second[2]
    assert set(first + second) <= set(s3_endpoint.effective_store_values())

    before_redaction = identity_cache.calls
    message = cli._safe_error_message(RuntimeError(f"denied {first[0]} then {second[0]}"))
    assert first[0] not in message
    assert second[0] not in message
    assert identity_cache.calls == before_redaction


def test_signing_identity_cache_rejects_a_changed_key_without_echoing_credentials():
    snapshot = cli._SigningCredentialSnapshot(
        "AKIAEXPRESSMISMATCH",
        "express-mismatch-secret",
        "express-mismatch-token",
    )
    cache = cli._SigningIdentityCache(snapshot, "expected-directory-bucket")

    assert cache.get_credentials("expected-directory-bucket") is snapshot
    with pytest.raises(ValueError) as captured:
        cache.get_credentials("different-directory-bucket")

    message = str(captured.value)
    assert "different-directory-bucket" not in message
    assert all(
        value not in message for value in (snapshot.access_key, snapshot.secret_key, snapshot.token)
    )


def test_real_request_signer_preserves_preexisting_request_credentials_across_retry(
    _clean_effective_store_credentials,
):
    default = _RotatingCredentials(("AKIADEFAULT00000001", "default-secret-abcdefgh", None))
    first = ("AKIAOVERRIDE0000001", "override-secret-abcdef1", "override-token-abcdef1")
    second = ("AKIAOVERRIDE0000002", "override-secret-abcdef2", "override-token-abcdef2")
    override = _RotatingCredentials(first, second)
    signer = _request_signer(default)
    request = AWSRequest(method="GET", url="https://s3.amazonaws.com/bucket/key")
    request.context["signing"] = {"request_credentials": override}

    signer.sign("GetObject", request)
    assert f"Credential={first[0]}/" in request.headers["Authorization"]
    signer.sign("GetObject", request)
    assert f"Credential={second[0]}/" in request.headers["Authorization"]

    assert override.calls == 2
    assert default.calls == 0


def test_real_request_signer_unsigned_request_never_reads_or_records_credentials(
    _clean_effective_store_credentials,
):
    credentials = _RotatingCredentials(
        ("AKIAUNSIGNED0000001", "unsigned-secret-abcdefgh", "unsigned-token-abcdefgh")
    )
    signer = _request_signer(credentials, signature_version=botocore.UNSIGNED)
    request = AWSRequest(method="GET", url="https://s3.amazonaws.com/bucket/key")

    signer.sign("GetObject", request)

    assert credentials.calls == 0
    assert "Authorization" not in request.headers
    assert s3_endpoint.effective_store_values() == ()


@pytest.mark.parametrize(
    "key,echo",
    [
        ("synthetic-key+/?", quote("synthetic-key+/?", safe="")),
        ("synthetic-key+/?", "synthetic-key%2b%2f%3f"),
        ("synthetic-key+/?", "synthetic-key%2b%2F%3f"),
        ("synthetic-key+/?", "synthetic-key%2B%2f%3F"),
        ('synthetic-"key', 'synthetic-"key'),
        ('synthetic-"key', json.dumps('synthetic-"key')[1:-1]),
        ("synthetic-été-key", "synthetic-été-key"),
        ("synthetic-été-key", json.dumps("synthetic-été-key")[1:-1]),
    ],
)
def test_capture_refuses_encoded_or_escaped_api_key_echo_before_any_publication(
    tmp_path, key, echo
):
    calls = []

    def transport(params):
        calls.append(params)
        return {**_payload(), "reason": f"provider error: apikey={echo}"}

    with pytest.raises(cli.EnsembleArchiveError, match="configured credentials") as caught:
        cli.run(_args(tmp_path), client=_client(transport, apikey=key))

    assert len(calls) == 1
    assert key not in str(caught.value)
    assert echo not in str(caught.value)
    assert list(tmp_path.rglob("*.json")) == []


def test_capture_key_comparison_preserves_literal_case_and_original_payload(tmp_path):
    key = "synthetic-key+/?"
    payload = {"reason": "SYNTHETIC-KEY%2b%2F%3f"}
    client = _client(lambda _params: _payload(), apikey=key)
    path = tmp_path / "capture.json"

    receipt = cli._publish_capture_document(path, payload, client)

    assert json.loads(path.read_text()) == payload
    assert payload == {"reason": "SYNTHETIC-KEY%2b%2F%3f"}
    assert receipt["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
