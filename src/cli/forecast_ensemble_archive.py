"""Collect one create-only daily AIFS ensemble snapshot for the canonical zones."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
import re
import sys
import tempfile
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any
from urllib.parse import quote, quote_plus, urlsplit
from uuid import uuid4

from src.cli.forecast_archive_panel import _s3_uri
from src.config import settings
from src.forecast_backtest import (
    DEFAULT_PRECIPITATION_EVENT_THRESHOLD_MM,
    PRECIPITATION_EVENT_OPERATOR,
)
from src.forecast_ensemble import (
    CAPTURE_WINDOW_END_HOUR_UTC,
    CAPTURE_WINDOW_START_HOUR_UTC,
    DEFAULT_HORIZONS,
    EXPECTED_MEMBER_COUNT,
    MAX_PROVIDER_COORDINATE_DELTA_DEGREES,
    MEMBER_FIELDS,
    MODEL,
    PROVIDER,
    EnsembleDataQualityError,
    OpenMeteoEnsembleClient,
)
from src.forecast_ensemble import (
    SCHEMA_VERSION as ARTIFACT_SCHEMA_VERSION,
)
from src.tools.climate import _load_zones
from src.utils.logging import setup_logging
from src.utils.s3_endpoint import (
    ENDPOINT_REFUSAL,
    S3EndpointConfigurationError,
    _spellings,
    configured_store_secrets,
    remember_effective_store_credentials,
    sanitize_endpoint,
)

HORIZONS = DEFAULT_HORIZONS
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
MANIFEST_SCHEMA_VERSION = "forecast-ensemble-forward-manifest/v1"
CAPTURE_REPORT_SCHEMA_VERSION = "forecast-ensemble-capture-report/v1"
DEFAULT_S3_PREFIX = "climate/forecast-ensemble"
MANIFEST_S3_PREFIX = "climate/runs"
EXPECTED_ARTIFACTS = len(CANONICAL_ZONES)
EXPECTED_ROWS = EXPECTED_ARTIFACTS * len(HORIZONS)
EXPECTED_MEMBER_VALUES = EXPECTED_ROWS * EXPECTED_MEMBER_COUNT
MAX_ARCHIVE_JSON_BYTES = 2 * 1024 * 1024
OFFICIAL_ENSEMBLE_HOSTS = {
    "ensemble-api.open-meteo.com",
    "customer-ensemble-api.open-meteo.com",
}


class EnsembleArchiveError(RuntimeError):
    """Archive failure whose message has already been stripped of credentials."""


@dataclass(frozen=True)
class ValidatedEnsembleArchive:
    """Fully verified archive input ready for conversion to backtest rows."""

    manifest: dict[str, Any]
    manifest_location: str
    manifest_sha256: str
    provider: str
    model: str
    rows: tuple[dict[str, Any], ...]


def _date_arg(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected an ISO date YYYY-MM-DD") from exc


def _run_id(snapshot_date: date) -> str:
    return f"aifs-ensemble-forward-{snapshot_date.strftime('%Y%m%d')}"


def _artifact_key(zone_id: str, snapshot_date: date) -> str:
    return f"{zone_id}__{MODEL}__{snapshot_date.isoformat()}"


def _bucket(args: argparse.Namespace) -> str:
    return getattr(args, "s3_bucket", None) or settings.S3_BUCKET


def _manifest_path(*, storage_mode: str, output_dir: Path, run_id: str, bucket: str) -> str | Path:
    if storage_mode == "s3":
        if not bucket:
            raise ValueError(
                "--s3-bucket or configured S3_BUCKET is required with --storage-mode s3"
            )
        return _s3_uri(
            bucket=bucket,
            prefix=MANIFEST_S3_PREFIX,
            run_id=run_id,
            name="manifest.json",
        )
    return (output_dir / run_id / "manifest.json").resolve()


def _artifact_path(
    *,
    storage_mode: str,
    output_dir: Path,
    run_id: str,
    bucket: str,
    s3_prefix: str,
    zone_id: str,
    snapshot_date: date,
) -> str | Path:
    name = f"{_artifact_key(zone_id, snapshot_date)}.json"
    if storage_mode == "s3":
        if not bucket:
            raise ValueError(
                "--s3-bucket or configured S3_BUCKET is required with --storage-mode s3"
            )
        return _s3_uri(bucket=bucket, prefix=s3_prefix, run_id=run_id, name=name)
    return (output_dir / run_id / name).resolve()


def _missing_remote_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return "404" in message or "not found" in message or "no files found" in message


def _s3_error_status(exc: Exception) -> tuple[int | None, str]:
    response = getattr(exc, "response", None)
    if not isinstance(response, dict):
        return None, ""
    metadata = response.get("ResponseMetadata")
    error = response.get("Error")
    status = metadata.get("HTTPStatusCode") if isinstance(metadata, dict) else None
    code = error.get("Code") if isinstance(error, dict) else ""
    return status if isinstance(status, int) else None, str(code)


def _s3_parts(path: str) -> tuple[str, str]:
    parsed = urlsplit(path)
    if (
        parsed.scheme != "s3"
        or not parsed.netloc
        or not parsed.path.lstrip("/")
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("invalid S3 archive URI")
    return parsed.netloc, parsed.path.lstrip("/")


def _object_store_endpoint() -> str | None:
    """The endpoint boto3 may see, or ``None`` when none is configured.

    botocore quotes ``endpoint_url`` back in its exceptions — measured on botocore 1.43.57::

        EndpointConnectionError: Could not connect to the endpoint URL:
        "http://rustfs-admin:sup3r-s3cret@127.0.0.1:1/deposium/x.json"

    so ``endpoint_url=settings.S3_ENDPOINT`` put the store's password into any traceback this
    CLI raised out of a read, on every call path that does not pass through
    :func:`_safe_error_message`. DuckDB httpfs stopped receiving the raw value; boto3 was still
    getting it.

    Routed through the same accept grammar the connection factory uses, and a value that does
    not parse RAISES rather than being handed over: an unparseable ``S3_ENDPOINT`` reaching
    boto3 as ``None`` is the fail-open in its other spelling — boto3's default endpoint is AWS,
    and ``S3_ACCESS_KEY``/``S3_SECRET_KEY`` would be signed into a request to Amazon.
    """
    raw = settings.S3_ENDPOINT
    if raw is None or raw == "":
        return None
    address, _differed = sanitize_endpoint(raw)
    if address is None:
        raise S3EndpointConfigurationError(ENDPOINT_REFUSAL)
    return address


@dataclass(frozen=True, repr=False)
class _SigningCredentialSnapshot:
    """Immutable credential object accepted by ``RequestSigner.get_auth_instance``."""

    access_key: str
    secret_key: str
    token: str | None = None
    account_id: str | None = None

    def get_frozen_credentials(self) -> _SigningCredentialSnapshot:
        # RequestSigner calls this even when ``request_credentials`` is already immutable.  A bare
        # botocore ReadOnlyCredentials tuple has no such method, hence this small compatible shell.
        return self


@dataclass(frozen=True, repr=False)
class _SigningIdentityCache:
    snapshot: _SigningCredentialSnapshot
    cache_key: Any

    def get_credentials(self, cache_key: Any) -> _SigningCredentialSnapshot:
        if cache_key != self.cache_key:
            raise ValueError("signing identity cache key changed after credential capture")
        return self.snapshot


@dataclass(frozen=True, repr=False)
class _CredentialCaptureState:
    source: Any
    snapshot: _SigningCredentialSnapshot
    identity_cache: _SigningIdentityCache | None = None


_CREDENTIAL_CAPTURE_STATE = "_geoai_effective_store_credentials"


def _capture_effective_store_credentials(
    *,
    request: Any,
    request_signer: Any,
    signature_version: Any,
    unsigned_signature: Any,
    auth_type_maps: Any,
    **_kwargs: Any,
) -> None:
    """Freeze, remember and pin exactly the credentials used for one signing attempt.

    ``before-sign`` runs again for every retry.  The state below remembers the original dynamic
    source, not the snapshot injected on the previous attempt, so a refreshable ECS/IMDS/IRSA
    credential may rotate between attempts. A ``request_credentials`` supplied by another
    botocore component remains the source of truth instead of being replaced by the signer's
    default credentials. S3 Express is special: botocore selects a session credential from its
    identity cache after ``before-sign``. For that auth type we freeze the original cache result
    once and install a repr-safe, one-attempt cache so the signer uses the exact observed value.
    """
    if signature_version is unsigned_signature:
        return

    signing_context = request.context.setdefault("signing", {})
    prior = request.context.get(_CREDENTIAL_CAPTURE_STATE)
    auth_type = auth_type_maps.get(signature_version)
    if getattr(auth_type, "REQUIRES_IDENTITY_CACHE", False):
        existing_cache = signing_context.get("identity_cache")
        if existing_cache is None or "cache_key" not in signing_context:
            return
        if isinstance(prior, _CredentialCaptureState) and existing_cache is prior.identity_cache:
            source = prior.source
        else:
            source = existing_cache
        get_credentials = getattr(source, "get_credentials", None)
        if not callable(get_credentials):
            return
        selected = get_credentials(signing_context["cache_key"])
    else:
        existing = signing_context.get("request_credentials")
        if isinstance(prior, _CredentialCaptureState) and existing is prior.snapshot:
            source = prior.source
        elif existing is not None:
            source = existing
        else:
            # Botocore's public before-sign event exposes the RequestSigner but not its
            # credentials. This is the sole private attribute used by the bridge and is pinned by
            # a real-signer contract test. Reading it does not resolve or refresh the provider.
            source = getattr(request_signer, "_credentials", None)
        selected = source
    if selected is None:
        return

    freeze = getattr(selected, "get_frozen_credentials", None)
    frozen = freeze() if callable(freeze) else selected
    access_key = getattr(frozen, "access_key", None)
    secret_key = getattr(frozen, "secret_key", None)
    if not isinstance(access_key, str) or not isinstance(secret_key, str):
        return
    token = getattr(frozen, "token", None)
    account_id = getattr(frozen, "account_id", None)
    snapshot = _SigningCredentialSnapshot(
        access_key=access_key,
        secret_key=secret_key,
        token=token if isinstance(token, str) else None,
        account_id=account_id if isinstance(account_id, str) else None,
    )
    remember_effective_store_credentials(snapshot.access_key, snapshot.secret_key, snapshot.token)
    if getattr(auth_type, "REQUIRES_IDENTITY_CACHE", False):
        identity_cache = _SigningIdentityCache(snapshot, signing_context["cache_key"])
        signing_context["identity_cache"] = identity_cache
        state = _CredentialCaptureState(source, snapshot, identity_cache)
    else:
        signing_context["request_credentials"] = snapshot
        state = _CredentialCaptureState(source, snapshot)
    request.context[_CREDENTIAL_CAPTURE_STATE] = state


def _object_store_client(*, normalize_scheme_less_endpoint: bool = False) -> Any:
    # Resolve the endpoint BEFORE importing boto3: a refusal must not be reachable only after a
    # client exists, and nothing should be constructed for a configuration that cannot be used.
    endpoint_url = _object_store_endpoint()
    # The serving normals reader replaces DuckDB httpfs, which uses TLS for a
    # validated bare/protocol-relative authority. Preserve that spelling there
    # without changing existing CLI callers. Validation above always runs first.
    if normalize_scheme_less_endpoint and endpoint_url is not None:
        if endpoint_url.startswith("//"):
            endpoint_url = f"https:{endpoint_url}"
        elif not endpoint_url.startswith(("http://", "https://")):
            endpoint_url = f"https://{endpoint_url}"
    boto3 = importlib.import_module("boto3")
    config_type = importlib.import_module("botocore.config").Config
    addressing_style = "path" if settings.S3_FORCE_PATH_STYLE else "virtual"
    client = boto3.client(
        "s3",
        endpoint_url=endpoint_url,
        aws_access_key_id=settings.S3_ACCESS_KEY or None,
        aws_secret_access_key=settings.S3_SECRET_KEY or None,
        region_name=os.environ.get("AWS_REGION", "us-east-1"),
        config=config_type(s3={"addressing_style": addressing_style}),
    )
    # Capture at signing rather than client construction.  The latter would pin the first
    # refreshable generation forever; before-sign runs on retries too and lets the provider rotate.
    unsigned_signature = importlib.import_module("botocore").UNSIGNED
    auth_type_maps = importlib.import_module("botocore.auth").AUTH_TYPE_MAPS
    # S3ExpressIdentityResolver is registered on the broad ``before-sign.s3`` event. A more
    # specific wildcard runs before that resolver, when its identity cache does not exist yet.
    # Register last on the same event so the exact session credential can be captured and pinned.
    client.meta.events.register_last(
        "before-sign.s3",
        lambda **kwargs: _capture_effective_store_credentials(
            **kwargs,
            unsigned_signature=unsigned_signature,
            auth_type_maps=auth_type_maps,
        ),
    )
    return client


def _json_bytes(payload: dict[str, Any]) -> bytes:
    return (json.dumps(payload, sort_keys=True, ensure_ascii=True) + "\n").encode()


def _read_document(path: str | Path) -> tuple[Any, bytes]:
    if isinstance(path, Path):
        if path.stat().st_size > MAX_ARCHIVE_JSON_BYTES:
            raise ValueError("archive JSON exceeds the maximum allowed size")
        with path.open("rb") as stream:
            raw = stream.read(MAX_ARCHIVE_JSON_BYTES + 1)
    else:
        bucket, key = _s3_parts(path)
        try:
            response = _object_store_client().get_object(Bucket=bucket, Key=key)
        except Exception as exc:
            status, code = _s3_error_status(exc)
            if (
                status == 404
                or code in {"404", "NoSuchKey", "NotFound"}
                or _missing_remote_error(exc)
            ):
                raise FileNotFoundError(path) from exc
            raise
        body = response.get("Body") if isinstance(response, dict) else None
        if body is None or not hasattr(body, "read"):
            raise ValueError("S3 archive response body is missing")
        content_length = response.get("ContentLength") if isinstance(response, dict) else None
        try:
            if isinstance(content_length, int) and content_length > MAX_ARCHIVE_JSON_BYTES:
                raise ValueError("archive JSON exceeds the maximum allowed size")
            raw = body.read(MAX_ARCHIVE_JSON_BYTES + 1)
        finally:
            close = getattr(body, "close", None)
            if callable(close):
                close()
        if not isinstance(raw, bytes):
            raise ValueError("S3 archive response body must be bytes")
    if len(raw) > MAX_ARCHIVE_JSON_BYTES:
        raise ValueError("archive JSON exceeds the maximum allowed size")
    return json.loads(raw), raw


def _read_existing(path: str | Path) -> Any | None:
    try:
        payload, _raw = _read_document(path)
        return payload
    except FileNotFoundError:
        return None
    except Exception as exc:
        if not isinstance(path, Path) and _missing_remote_error(exc):
            return None
        raise


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    directory_fd = os.open(path, flags)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _mkdir_durable(path: Path) -> None:
    missing: list[Path] = []
    cursor = path
    while not cursor.exists() and cursor.parent != cursor:
        missing.append(cursor)
        cursor = cursor.parent
    path.mkdir(parents=True, exist_ok=True)
    for created in reversed(missing):
        _fsync_directory(created.parent)


def _publish_json_if_absent(path: str | Path, payload: dict[str, Any]) -> bool:
    """Publish one complete create-only object, returning whether this caller won."""
    encoded = _json_bytes(payload)
    if isinstance(path, Path):
        _mkdir_durable(path.parent)
        file_descriptor, temporary_name = tempfile.mkstemp(
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
        )
        temporary_path = Path(temporary_name)
        try:
            with os.fdopen(file_descriptor, "wb") as stream:
                os.fchmod(stream.fileno(), 0o644)
                stream.write(encoded)
                stream.flush()
                os.fsync(stream.fileno())
            try:
                os.link(temporary_path, path)
            except FileExistsError:
                return False
            _fsync_directory(path.parent)
            return True
        finally:
            temporary_path.unlink(missing_ok=True)

    bucket, key = _s3_parts(path)
    try:
        _object_store_client().put_object(
            Bucket=bucket,
            Key=key,
            Body=encoded,
            ContentLength=len(encoded),
            ContentType="application/json",
            IfNoneMatch="*",
        )
    except Exception as exc:
        status, code = _s3_error_status(exc)
        if status in {409, 412} or code in {"ConditionalRequestConflict", "PreconditionFailed"}:
            return False
        raise
    return True


def _digest(payload: dict[str, Any]) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _secret_values(client: Any | None) -> set[str]:
    """This CLI's secrets: the weather API key and the ambient AWS env, plus the store's.

    The object-store half is not assembled here any more. ``configured_store_secrets`` is the
    single definition of "what credentials is this process configured with" — endpoint userinfo,
    the separately supplied ``S3_ACCESS_KEY``/``S3_SECRET_KEY``, AND the ambient ``AWS_*``
    environment that used to be read raw right here — shared with the manifest
    redactor, which used to know only about the userinfo and was consequently a no-op in the
    deployment shape this repo actually runs.
    """
    values = (
        getattr(client, "apikey", "") if client is not None else "",
        getattr(settings, "FORECAST_ENSEMBLE_API_KEY", ""),
        # The ambient AWS_* reads used to be spelled out here, raw. They now live in
        # configured_store_secrets, which also runs them through _spellings — so this CLI gains
        # the percent-encoded and JSON-escaped forms it silently lacked, and the panels gain the
        # ambient credentials they never had at all (#147).
        *configured_store_secrets(settings.S3_ENDPOINT),
    )
    return {value.strip() for value in values if isinstance(value, str) and value.strip()}


def _safe_error_message(exc: Exception, client: Any | None = None) -> str:
    message = str(exc) or type(exc).__name__
    # Longest secret first. `_secret_values` is a SET, so it iterated in hash order, and these
    # secrets overlap: this repo's own store uses account `deposium-dev` with password
    # `deposium-dev-secret`, so the account name is a prefix of the password. Redacting the short
    # one first rewrote the password to `[REDACTED]-secret` and published the tail — on 7 of 8
    # PYTHONHASHSEED values, and invisibly, because `_ensure_secret_free` then finds no WHOLE
    # secret in the payload and passes it.
    for secret in sorted(_secret_values(client), key=len, reverse=True):
        for candidate in (secret, quote(secret, safe=""), quote_plus(secret, safe="")):
            message = message.replace(candidate, "[REDACTED]")
    return message


def _ensure_secret_free(payload: dict[str, Any], client: Any | None) -> None:
    serialized = json.dumps(payload, sort_keys=True, ensure_ascii=True)
    if any(secret in serialized for secret in _secret_values(client)):
        raise ValueError("archive payload contains configured credentials")


def _finite_number(value: Any, field: str, *, nonnegative: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a finite number")
    try:
        finite = math.isfinite(value)
    except OverflowError as exc:
        raise ValueError(f"{field} must be a finite number") from exc
    if not finite or (nonnegative and value < 0):
        suffix = " >= 0" if nonnegative else ""
        raise ValueError(f"{field} must be a finite number{suffix}")
    return float(value)


def _event_contract() -> dict[str, Any]:
    return {
        "variable": "precipitation",
        "operator": PRECIPITATION_EVENT_OPERATOR,
        "threshold_mm": DEFAULT_PRECIPITATION_EVENT_THRESHOLD_MM,
        "probability_method": "raw_ensemble_relative_frequency",
        "calibration_status": "uncalibrated",
    }


def _validate_endpoint(endpoint: Any, expected_endpoint: str) -> None:
    if endpoint != expected_endpoint or not isinstance(endpoint, str) or not endpoint:
        raise ValueError("artifact endpoint is inconsistent")
    parsed = urlsplit(endpoint)
    if (
        parsed.scheme.lower() != "https"
        or not parsed.hostname
        or parsed.port not in (None, 443)
        or parsed.path.rstrip("/") != "/v1/ensemble"
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
        or parsed.hostname not in OFFICIAL_ENSEMBLE_HOSTS
    ):
        raise ValueError(
            "artifact endpoint must be an official credential-free HTTPS /v1/ensemble URL"
        )


def _validate_retrieved_at(
    value: Any,
    snapshot_date: date,
    *,
    request_started_at: Any,
) -> None:
    timestamps: dict[str, datetime] = {}
    for field, raw in (
        ("request_started_at_utc", request_started_at),
        ("retrieved_at_utc", value),
    ):
        if not isinstance(raw, str) or not raw:
            raise ValueError(f"artifact {field} is missing")
        try:
            timestamp = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError(f"artifact {field} is invalid") from exc
        if timestamp.utcoffset() != timedelta(0) or timestamp.date() != snapshot_date:
            raise ValueError(f"artifact {field} must be on the UTC snapshot date")
        if not CAPTURE_WINDOW_START_HOUR_UTC <= timestamp.hour < CAPTURE_WINDOW_END_HOUR_UTC:
            raise ValueError(f"artifact {field} is outside the fixed UTC capture window")
        timestamps[field] = timestamp
    if timestamps["retrieved_at_utc"] < timestamps["request_started_at_utc"]:
        raise ValueError("artifact acquisition timestamps move backwards")


def _validate_artifact(
    payload: Any,
    *,
    zone_id: str,
    snapshot_date: date,
    run_id: str,
    expected_endpoint: str,
    expected_latitude: float,
    expected_longitude: float,
) -> dict[str, int]:
    if not isinstance(payload, dict):
        raise ValueError("artifact payload must be an object")
    expected_fields = {
        "schema_version",
        "run_id",
        "zone_id",
        "snapshot_date_utc",
        "request_started_at_utc",
        "retrieved_at_utc",
        "provider",
        "model",
        "requested_coordinates",
        "provider_coordinates",
        "endpoint",
        "request_parameters",
        "member_contract",
        "rows",
    }
    if set(payload) != expected_fields:
        raise ValueError("artifact fields are inconsistent")
    expected_identity = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "run_id": run_id,
        "zone_id": zone_id,
        "snapshot_date_utc": snapshot_date.isoformat(),
        "provider": PROVIDER,
        "model": MODEL,
    }
    for field, expected in expected_identity.items():
        if payload.get(field) != expected:
            raise ValueError(f"artifact {field} is inconsistent")
    _validate_endpoint(payload.get("endpoint"), expected_endpoint)
    _validate_retrieved_at(
        payload.get("retrieved_at_utc"),
        snapshot_date,
        request_started_at=payload.get("request_started_at_utc"),
    )
    requested_coordinates = payload.get("requested_coordinates")
    provider_coordinates = payload.get("provider_coordinates")
    if not isinstance(requested_coordinates, dict) or not isinstance(provider_coordinates, dict):
        raise ValueError("artifact coordinates are missing")
    if requested_coordinates != {
        "latitude": expected_latitude,
        "longitude": expected_longitude,
    }:
        raise ValueError("artifact requested coordinates are inconsistent")
    provider_latitude = _finite_number(provider_coordinates.get("latitude"), "provider latitude")
    provider_longitude = _finite_number(provider_coordinates.get("longitude"), "provider longitude")
    provider_elevation = provider_coordinates.get("elevation")
    if set(provider_coordinates) != {"latitude", "longitude", "elevation"}:
        raise ValueError("artifact provider coordinate fields are inconsistent")
    if provider_elevation is not None:
        _finite_number(provider_elevation, "provider elevation")
    if not -90 <= provider_latitude <= 90 or not -180 <= provider_longitude <= 180:
        raise ValueError("artifact provider coordinates are invalid")
    latitude_delta = abs(provider_latitude - expected_latitude)
    longitude_delta = abs((provider_longitude - expected_longitude + 180) % 360 - 180)
    if (
        latitude_delta > MAX_PROVIDER_COORDINATE_DELTA_DEGREES
        or longitude_delta > MAX_PROVIDER_COORDINATE_DELTA_DEGREES
    ):
        raise ValueError("artifact provider coordinates are too far from the requested zone")

    request = payload.get("request_parameters")
    if not isinstance(request, dict):
        raise ValueError("artifact request parameters are invalid")
    expected_request = {
        "latitude": expected_latitude,
        "longitude": expected_longitude,
        "models": MODEL,
        "daily": "precipitation_sum",
        "timezone": "UTC",
        "precipitation_unit": "mm",
        "start_date": (snapshot_date + timedelta(days=min(HORIZONS))).isoformat(),
        "end_date": (snapshot_date + timedelta(days=max(HORIZONS))).isoformat(),
    }
    if request != expected_request:
        raise ValueError("artifact request parameters are inconsistent")

    member_contract = payload.get("member_contract")
    expected_member_contract = {
        "expected": EXPECTED_MEMBER_COUNT,
        "weighting": "equal",
        "source_fields": list(MEMBER_FIELDS),
    }
    if member_contract != expected_member_contract:
        raise ValueError("artifact member contract is inconsistent")

    rows = payload.get("rows")
    if not isinstance(rows, list) or len(rows) != len(HORIZONS):
        raise ValueError("artifact must contain exactly four rows")
    by_horizon = {row.get("horizon_days"): row for row in rows if isinstance(row, dict)}
    if len(by_horizon) != len(HORIZONS) or set(by_horizon) != set(HORIZONS):
        raise ValueError("artifact rows must contain the four canonical horizons exactly once")

    member_values = 0
    event_contract = _event_contract()
    for horizon in HORIZONS:
        row = by_horizon[horizon]
        expected_row = {
            "zone_id": zone_id,
            "origin_date": snapshot_date.isoformat(),
            "valid_date": (snapshot_date + timedelta(days=horizon)).isoformat(),
            "horizon_days": horizon,
            "variable": "precipitation",
            "provider": PROVIDER,
            "model": MODEL,
            "member_count_expected": EXPECTED_MEMBER_COUNT,
            "member_count_used": EXPECTED_MEMBER_COUNT,
            "probability_method": event_contract["probability_method"],
            "calibration_status": event_contract["calibration_status"],
            "origin_semantics": "snapshot_date_utc_not_model_initialization",
        }
        expected_row_fields = {
            *expected_row,
            "value",
            "probability",
            "probability_event",
            "event_member_count",
            "members",
            "daily_aggregation",
        }
        if set(row) != expected_row_fields:
            raise ValueError(f"artifact row fields are inconsistent for horizon {horizon}")
        if any(row.get(field) != expected for field, expected in expected_row.items()):
            raise ValueError(f"artifact row identity is inconsistent for horizon {horizon}")
        if row.get("probability_event") != {
            "operator": event_contract["operator"],
            "threshold_mm": event_contract["threshold_mm"],
        }:
            raise ValueError(f"artifact event contract is inconsistent for horizon {horizon}")
        if row.get("daily_aggregation") != {
            "source": "provider_daily",
            "operation": "sum",
            "window_hours": 24,
            "timezone": "UTC",
            "unit": "mm",
        }:
            raise ValueError(f"artifact daily aggregation is inconsistent for horizon {horizon}")

        members = row.get("members")
        if not isinstance(members, list) or len(members) != EXPECTED_MEMBER_COUNT:
            raise ValueError(f"artifact horizon {horizon} must retain all ensemble members")
        values: list[float] = []
        for member_id, (member, source_field) in enumerate(
            zip(members, MEMBER_FIELDS, strict=True)
        ):
            if (
                not isinstance(member, dict)
                or set(member) != {"member_id", "source_field", "daily_precipitation_mm"}
                or member.get("member_id") != member_id
            ):
                raise ValueError(f"artifact member identity is inconsistent for horizon {horizon}")
            if member.get("source_field") != source_field:
                raise ValueError(f"artifact member source is inconsistent for horizon {horizon}")
            values.append(
                _finite_number(
                    member.get("daily_precipitation_mm"),
                    f"horizon {horizon} member {member_id}",
                    nonnegative=True,
                )
            )
        event_count = sum(value > DEFAULT_PRECIPITATION_EVENT_THRESHOLD_MM for value in values)
        if row.get("event_member_count") != event_count:
            raise ValueError(f"artifact event count is inconsistent for horizon {horizon}")
        probability = _finite_number(row.get("probability"), "row probability")
        if not math.isclose(
            probability,
            event_count / EXPECTED_MEMBER_COUNT,
            rel_tol=0.0,
            abs_tol=1e-15,
        ):
            raise ValueError(f"artifact probability is inconsistent for horizon {horizon}")
        value = _finite_number(row.get("value"), "row value", nonnegative=True)
        if not math.isclose(
            value,
            sum(values) / EXPECTED_MEMBER_COUNT,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(f"artifact mean is inconsistent for horizon {horizon}")
        member_values += len(values)
    return {"rows": len(rows), "member_values": member_values}


def _runtime_code_ref() -> dict[str, str] | None:
    def first(*names: str) -> str | None:
        return next((os.environ[name] for name in names if os.environ.get(name)), None)

    commit = first("RAILWAY_GIT_COMMIT_SHA", "GIT_COMMIT", "SOURCE_VERSION")
    branch = first("RAILWAY_GIT_BRANCH", "GIT_BRANCH", "BRANCH_NAME")
    if commit is None and branch is None:
        return None
    result = {"repo": "deposium_geoai"}
    if commit is not None:
        result["commit"] = commit
    if branch is not None:
        result["branch"] = branch
    return result


def _build_manifest(
    *,
    snapshot_date: date,
    run_id: str,
    storage_mode: str,
    s3_prefix: str,
    endpoint: str,
    artifacts: list[dict[str, Any]],
) -> dict[str, Any]:
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "run_id": run_id,
        "snapshot_date_utc": snapshot_date.isoformat(),
        "created_at_utc": datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "storage_mode": storage_mode,
        "s3_prefix": s3_prefix if storage_mode == "s3" else None,
        "zones": list(CANONICAL_ZONES),
        "horizons": list(HORIZONS),
        "provider": PROVIDER,
        "model": MODEL,
        "endpoint": endpoint,
        "event_contract": _event_contract(),
        "artifacts": artifacts,
        "counts": {
            "zones": len(CANONICAL_ZONES),
            "horizons": len(HORIZONS),
            "artifacts": len(artifacts),
            "rows": sum(item["rows"] for item in artifacts),
            "members_per_row": EXPECTED_MEMBER_COUNT,
            "member_values": sum(item["member_values"] for item in artifacts),
            "expected_artifacts": EXPECTED_ARTIFACTS,
            "expected_rows": EXPECTED_ROWS,
            "expected_member_values": EXPECTED_MEMBER_VALUES,
        },
        "checks": {
            "all_artifacts_valid": len(artifacts) == EXPECTED_ARTIFACTS,
            "all_rows_valid": sum(item["rows"] for item in artifacts) == EXPECTED_ROWS,
            "all_members_complete": (
                sum(item["member_values"] for item in artifacts) == EXPECTED_MEMBER_VALUES
            ),
            "ok": True,
        },
    }
    if code_ref := _runtime_code_ref():
        manifest["code_ref"] = code_ref
    return manifest


def _validate_completed_manifest(
    payload: Any,
    *,
    snapshot_date: date,
    run_id: str,
    storage_mode: str,
    output_dir: Path,
    bucket: str,
    s3_prefix: str,
    expected_endpoint: str,
    client: Any | None = None,
    collected_rows: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("existing manifest must be an object")
    if storage_mode == "s3":
        _validate_s3_scope(bucket=bucket, s3_prefix=s3_prefix)
    _ensure_secret_free(payload, client)
    required_fields = {
        "schema_version",
        "artifact_schema_version",
        "run_id",
        "snapshot_date_utc",
        "created_at_utc",
        "storage_mode",
        "s3_prefix",
        "zones",
        "horizons",
        "provider",
        "model",
        "endpoint",
        "event_contract",
        "artifacts",
        "counts",
        "checks",
    }
    actual_fields = set(payload)
    if actual_fields != required_fields and actual_fields != required_fields | {"code_ref"}:
        raise ValueError("existing manifest fields are inconsistent")
    expected_identity = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "run_id": run_id,
        "snapshot_date_utc": snapshot_date.isoformat(),
        "storage_mode": storage_mode,
        "s3_prefix": s3_prefix if storage_mode == "s3" else None,
        "zones": list(CANONICAL_ZONES),
        "horizons": list(HORIZONS),
        "provider": PROVIDER,
        "model": MODEL,
        "endpoint": expected_endpoint,
        "event_contract": _event_contract(),
    }
    if any(payload.get(field) != expected for field, expected in expected_identity.items()):
        raise ValueError("existing manifest identity or contract is inconsistent")
    if not isinstance(payload.get("created_at_utc"), str) or not payload["created_at_utc"]:
        raise ValueError("existing manifest created_at_utc is missing")
    try:
        created_at = datetime.fromisoformat(payload["created_at_utc"].replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("existing manifest created_at_utc is invalid") from exc
    if created_at.utcoffset() != timedelta(0):
        raise ValueError("existing manifest created_at_utc must be UTC")
    code_ref = payload.get("code_ref")
    if code_ref is not None:
        if (
            not isinstance(code_ref, dict)
            or set(code_ref) - {"repo", "commit", "branch"}
            or code_ref.get("repo") != "deposium_geoai"
            or any(not isinstance(value, str) or not value for value in code_ref.values())
        ):
            raise ValueError("existing manifest code_ref is invalid")
    counts = payload.get("counts")
    expected_counts = {
        "zones": len(CANONICAL_ZONES),
        "horizons": len(HORIZONS),
        "artifacts": EXPECTED_ARTIFACTS,
        "rows": EXPECTED_ROWS,
        "members_per_row": EXPECTED_MEMBER_COUNT,
        "member_values": EXPECTED_MEMBER_VALUES,
        "expected_artifacts": EXPECTED_ARTIFACTS,
        "expected_rows": EXPECTED_ROWS,
        "expected_member_values": EXPECTED_MEMBER_VALUES,
    }
    if counts != expected_counts:
        raise ValueError("existing manifest counts are inconsistent")
    if payload.get("checks") != {
        "all_artifacts_valid": True,
        "all_rows_valid": True,
        "all_members_complete": True,
        "ok": True,
    }:
        raise ValueError("existing manifest checks are incomplete")

    raw_artifacts = payload.get("artifacts")
    if not isinstance(raw_artifacts, list) or len(raw_artifacts) != EXPECTED_ARTIFACTS:
        raise ValueError("existing manifest must reference exactly 15 artifacts")
    artifacts_by_key = {item.get("key"): item for item in raw_artifacts if isinstance(item, dict)}
    if len(artifacts_by_key) != EXPECTED_ARTIFACTS:
        raise ValueError("existing manifest artifact identities are invalid")

    zones = _load_zones()
    if missing_zones := sorted(set(CANONICAL_ZONES) - set(zones)):
        raise ValueError(f"canonical zones are missing: {missing_zones}")

    for zone_id in CANONICAL_ZONES:
        key = _artifact_key(zone_id, snapshot_date)
        item = artifacts_by_key.get(key)
        if not isinstance(item, dict):
            raise ValueError(f"existing manifest is missing artifact {key}")
        if set(item) != {"key", "zone_id", "output", "sha256", "rows", "member_values"}:
            raise ValueError(f"existing manifest artifact fields are inconsistent for {key}")
        path = _artifact_path(
            storage_mode=storage_mode,
            output_dir=output_dir,
            run_id=run_id,
            bucket=bucket,
            s3_prefix=s3_prefix,
            zone_id=zone_id,
            snapshot_date=snapshot_date,
        )
        expected_entry = {
            "key": key,
            "zone_id": zone_id,
            "output": str(path),
            "rows": len(HORIZONS),
            "member_values": len(HORIZONS) * EXPECTED_MEMBER_COUNT,
        }
        if any(item.get(field) != expected for field, expected in expected_entry.items()):
            raise ValueError(f"existing manifest artifact metadata is inconsistent for {key}")
        digest = item.get("sha256")
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError(f"existing manifest digest is invalid for {key}")
        artifact = _read_existing(path)
        if artifact is None:
            raise ValueError(f"existing manifest artifact is missing: {key}")
        if not isinstance(artifact, dict):
            raise ValueError(f"existing manifest artifact must be an object: {key}")
        _ensure_secret_free(artifact, client)
        _validate_artifact(
            artifact,
            zone_id=zone_id,
            snapshot_date=snapshot_date,
            run_id=run_id,
            expected_endpoint=expected_endpoint,
            expected_latitude=zones[zone_id]["lat"],
            expected_longitude=zones[zone_id]["lon"],
        )
        if _digest(artifact) != digest:
            raise ValueError(f"artifact digest drift detected: {key}")
        if collected_rows is not None:
            collected_rows.extend(json.loads(json.dumps(artifact["rows"])))
    return payload


def _manifest_location(value: str | Path) -> str | Path:
    if isinstance(value, Path):
        return value.resolve()
    if value.startswith("s3://"):
        bucket, key = _s3_parts(value)
        segments = key.split("/")
        _validate_s3_scope(bucket=bucket, s3_prefix=DEFAULT_S3_PREFIX)
        if (
            len(segments) != 4
            or segments[:2] != MANIFEST_S3_PREFIX.split("/")
            or segments[3] != "manifest.json"
        ):
            raise ValueError("ensemble archive manifest must use the canonical S3 run path")
        return value
    return Path(value).resolve()


def _validate_s3_scope(*, bucket: str, s3_prefix: str) -> None:
    if not settings.S3_BUCKET or bucket != settings.S3_BUCKET:
        raise ValueError("ensemble archive must use the configured S3 bucket")
    if s3_prefix != DEFAULT_S3_PREFIX:
        raise ValueError("ensemble archive must use the canonical S3 artifact prefix")


def load_validated_archive(manifest_location: str | Path) -> ValidatedEnsembleArchive:
    """Load all 15 create-only artifacts after manifest and digest verification."""
    location = _manifest_location(manifest_location)
    manifest, raw_manifest = _read_document(location)
    if not isinstance(manifest, dict):
        raise ValueError("ensemble archive manifest must be an object")
    raw_snapshot_date = manifest.get("snapshot_date_utc")
    raw_run_id = manifest.get("run_id")
    raw_endpoint = manifest.get("endpoint")
    if not isinstance(raw_snapshot_date, str) or not isinstance(raw_run_id, str):
        raise ValueError("ensemble archive manifest identity is missing")
    try:
        snapshot_date = date.fromisoformat(raw_snapshot_date)
    except ValueError as exc:
        raise ValueError("ensemble archive snapshot_date_utc is invalid") from exc
    if raw_run_id != _run_id(snapshot_date):
        raise ValueError("ensemble archive run_id is inconsistent")
    if not isinstance(raw_endpoint, str):
        raise ValueError("ensemble archive endpoint is missing")
    _validate_endpoint(raw_endpoint, raw_endpoint)

    if isinstance(location, Path):
        storage_mode = "local"
        output_dir = location.parent.parent
        bucket = ""
        s3_prefix = DEFAULT_S3_PREFIX
    else:
        storage_mode = "s3"
        bucket, _key = _s3_parts(location)
        output_dir = Path(".")
        raw_s3_prefix = manifest.get("s3_prefix")
        if not isinstance(raw_s3_prefix, str):
            raise ValueError("ensemble archive S3 prefix is missing")
        s3_prefix = raw_s3_prefix
        _validate_s3_scope(bucket=bucket, s3_prefix=s3_prefix)

    expected_manifest_location = _manifest_path(
        storage_mode=storage_mode,
        output_dir=output_dir,
        run_id=raw_run_id,
        bucket=bucket,
    )
    if str(location) != str(expected_manifest_location):
        raise ValueError("ensemble archive manifest location is non-canonical")

    rows: list[dict[str, Any]] = []
    validated_manifest = _validate_completed_manifest(
        manifest,
        snapshot_date=snapshot_date,
        run_id=raw_run_id,
        storage_mode=storage_mode,
        output_dir=output_dir,
        bucket=bucket,
        s3_prefix=s3_prefix,
        expected_endpoint=raw_endpoint,
        collected_rows=rows,
    )
    if len(rows) != EXPECTED_ROWS:
        raise ValueError("ensemble archive did not yield exactly 60 prediction rows")
    return ValidatedEnsembleArchive(
        manifest=validated_manifest,
        manifest_location=str(location),
        manifest_sha256=hashlib.sha256(raw_manifest).hexdigest(),
        provider=PROVIDER,
        model=MODEL,
        rows=tuple(rows),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="forecast-ensemble-archive")
    parser.add_argument("--snapshot-date", type=_date_arg)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--storage-mode", choices=("local", "s3"), default="local")
    parser.add_argument("--s3-bucket", default=None)
    parser.add_argument("--s3-prefix", default=DEFAULT_S3_PREFIX)
    return parser


def _capture_path(
    *,
    storage_mode: str,
    output_dir: Path,
    run_id: str,
    bucket: str,
    s3_prefix: str,
    name: str,
) -> str | Path:
    if storage_mode == "s3":
        return _s3_uri(bucket=bucket, prefix=f"{s3_prefix}/{run_id}", run_id="_captures", name=name)
    return (output_dir / run_id / "_captures" / name).resolve()


def _publish_capture_document(
    path: str | Path, payload: dict[str, Any], client: Any
) -> dict[str, str]:
    """Publish bounded, credential-free evidence and verify the exact stored bytes."""
    _ensure_capture_secret_free(payload, client)
    encoded = _json_bytes(payload)
    if len(encoded) > MAX_ARCHIVE_JSON_BYTES:
        raise ValueError("capture JSON exceeds the maximum allowed size")
    expected_digest = hashlib.sha256(encoded).hexdigest()
    _publish_json_if_absent(path, payload)
    _winner, stored_bytes = _read_document(path)
    stored_digest = hashlib.sha256(stored_bytes).hexdigest()
    if stored_bytes != encoded or stored_digest != expected_digest:
        raise ValueError("capture evidence collision or stored bytes mismatch")
    return {"output": str(path), "sha256": stored_digest}


def _ensure_capture_secret_free(payload: dict[str, Any], client: Any | None) -> None:
    """Refuse credential echoes in decoded JSON, including encoded URL/error text."""
    serialized = _canonical_percent_escape_case(
        json.dumps(payload, sort_keys=True, ensure_ascii=True)
    )
    spellings = _spellings(configured=tuple(_secret_values(client)), derived=())
    if any(
        _canonical_percent_escape_case(candidate) in serialized
        for spelling in spellings
        for candidate in (spelling, json.dumps(spelling, ensure_ascii=True)[1:-1])
    ):
        raise ValueError("capture payload contains configured credentials")


def _canonical_percent_escape_case(value: str) -> str:
    # Only encoded hex digits are case-insensitive. Key text remains case-sensitive,
    # and this comparison copy never changes the captured provider payload.
    return re.sub(r"%[0-9a-fA-F]{2}", lambda match: match.group(0).upper(), value)


def _retain_first_partial_report(path: str | Path, report: dict[str, Any], client: Any) -> None:
    """Keep the first partial receipt discoverable without listing capture objects."""
    _ensure_capture_secret_free(report, client)
    if len(_json_bytes(report)) > MAX_ARCHIVE_JSON_BYTES:
        raise ValueError("capture report JSON exceeds the maximum allowed size")
    created = _publish_json_if_absent(path, report)
    retained, raw = _read_document(path)
    if created and raw != _json_bytes(report):
        raise ValueError("first partial receipt stored bytes mismatch")
    if not isinstance(retained, dict):
        raise ValueError("first partial receipt must be an object")
    _ensure_capture_secret_free(retained, client)
    expected = {
        "schema_version": CAPTURE_REPORT_SCHEMA_VERSION,
        "status": "partial",
        "run_id": report["run_id"],
        "snapshot_date_utc": report["snapshot_date_utc"],
        "checks": {"ok": False},
    }
    if any(retained.get(key) != value for key, value in expected.items()) or not retained.get(
        "rejected_zones"
    ):
        raise ValueError("first partial receipt is inconsistent")


def run(
    args: argparse.Namespace, *, client: OpenMeteoEnsembleClient | None = None
) -> dict[str, Any]:
    snapshot_date = args.snapshot_date or datetime.now(UTC).date()
    run_id = _run_id(snapshot_date)
    storage_mode = args.storage_mode
    output_dir = args.output_dir
    s3_prefix = args.s3_prefix
    bucket = _bucket(args)
    client = client or OpenMeteoEnsembleClient()
    endpoint = client.base_url
    try:
        _validate_endpoint(endpoint, endpoint)
        if storage_mode == "s3":
            _validate_s3_scope(bucket=bucket, s3_prefix=s3_prefix)
        zones = _load_zones()
        if missing_zones := sorted(set(CANONICAL_ZONES) - set(zones)):
            raise ValueError(f"canonical zones are missing: {missing_zones}")
        if storage_mode == "local":
            _mkdir_durable(output_dir / run_id)
        manifest_path = _manifest_path(
            storage_mode=storage_mode,
            output_dir=output_dir,
            run_id=run_id,
            bucket=bucket,
        )
        existing_manifest = _read_existing(manifest_path)
        if existing_manifest is not None:
            return _validate_completed_manifest(
                existing_manifest,
                snapshot_date=snapshot_date,
                run_id=run_id,
                storage_mode=storage_mode,
                output_dir=output_dir,
                bucket=bucket,
                s3_prefix=s3_prefix,
                expected_endpoint=endpoint,
                client=client,
            )

        attempt_id = uuid4().hex
        artifacts: list[dict[str, Any]] = []
        captures: list[dict[str, Any]] = []
        rejected_zones: list[dict[str, Any]] = []
        for zone_id in CANONICAL_ZONES:
            path = _artifact_path(
                storage_mode=storage_mode,
                output_dir=output_dir,
                run_id=run_id,
                bucket=bucket,
                s3_prefix=s3_prefix,
                zone_id=zone_id,
                snapshot_date=snapshot_date,
            )
            artifact = _read_existing(path)
            if artifact is None:
                zone = zones[zone_id]
                capture_reference: dict[str, str] | None = None

                def retain_capture(envelope: dict[str, Any], *, _zone_id: str = zone_id) -> None:
                    nonlocal capture_reference
                    if capture_reference is not None:
                        raise ValueError("provider supplied more than one capture for a zone")
                    capture_path = _capture_path(
                        storage_mode=storage_mode,
                        output_dir=output_dir,
                        run_id=run_id,
                        bucket=bucket,
                        s3_prefix=s3_prefix,
                        name=f"{_zone_id}_{attempt_id}.json",
                    )
                    capture_reference = _publish_capture_document(capture_path, envelope, client)
                    captures.append({"zone_id": _zone_id, **capture_reference})

                try:
                    snapshot = client.fetch_snapshot(
                        zone_id=zone_id,
                        latitude=zone["lat"],
                        longitude=zone["lon"],
                        snapshot_date=snapshot_date,
                        horizons=HORIZONS,
                        event_threshold_mm=DEFAULT_PRECIPITATION_EVENT_THRESHOLD_MM,
                        on_capture=retain_capture,
                    )
                except EnsembleDataQualityError as exc:
                    if capture_reference is None:
                        raise ValueError("provider data rejection has no retained capture") from exc
                    rejected_zones.append(
                        {
                            "zone_id": zone_id,
                            "error": _safe_error_message(exc, client),
                            "capture": capture_reference,
                        }
                    )
                    continue
                artifact = json.loads(
                    json.dumps({**snapshot.to_dict(), "run_id": run_id}, sort_keys=True)
                )
                _ensure_secret_free(artifact, client)
                _validate_artifact(
                    artifact,
                    zone_id=zone_id,
                    snapshot_date=snapshot_date,
                    run_id=run_id,
                    expected_endpoint=endpoint,
                    expected_latitude=zone["lat"],
                    expected_longitude=zone["lon"],
                )
                _publish_json_if_absent(path, artifact)
                artifact = _read_existing(path)
                if artifact is None:
                    raise ValueError("published artifact is missing")
            if not isinstance(artifact, dict):
                raise ValueError("existing artifact must be an object")
            _ensure_secret_free(artifact, client)
            stats = _validate_artifact(
                artifact,
                zone_id=zone_id,
                snapshot_date=snapshot_date,
                run_id=run_id,
                expected_endpoint=endpoint,
                expected_latitude=zones[zone_id]["lat"],
                expected_longitude=zones[zone_id]["lon"],
            )
            artifacts.append(
                {
                    "key": _artifact_key(zone_id, snapshot_date),
                    "zone_id": zone_id,
                    "output": str(path),
                    "sha256": _digest(artifact),
                    **stats,
                }
            )

        if rejected_zones:
            report = {
                "schema_version": CAPTURE_REPORT_SCHEMA_VERSION,
                "status": "partial",
                "run_id": run_id,
                "attempt_id": attempt_id,
                "snapshot_date_utc": snapshot_date.isoformat(),
                "artifacts": artifacts,
                "rejected_zones": rejected_zones,
                "raw_captures": captures,
                "counts": {
                    "artifacts": len(artifacts),
                    "rejected_zones": len(rejected_zones),
                    "raw_captures": len(captures),
                    "rows": sum(item["rows"] for item in artifacts),
                    "member_values": sum(item["member_values"] for item in artifacts),
                    "expected_artifacts": EXPECTED_ARTIFACTS,
                },
                "checks": {"ok": False},
            }
            report_path = _capture_path(
                storage_mode=storage_mode,
                output_dir=output_dir,
                run_id=run_id,
                bucket=bucket,
                s3_prefix=s3_prefix,
                name=f"report-{attempt_id}.json",
            )
            receipt = _publish_capture_document(report_path, report, client)
            result = {**report, "capture_report": receipt}
            first_partial_path = _capture_path(
                storage_mode=storage_mode,
                output_dir=output_dir,
                run_id=run_id,
                bucket=bucket,
                s3_prefix=s3_prefix,
                name="partial.json",
            )
            _retain_first_partial_report(first_partial_path, result, client)
            return result

        manifest = _build_manifest(
            snapshot_date=snapshot_date,
            run_id=run_id,
            storage_mode=storage_mode,
            s3_prefix=s3_prefix,
            endpoint=endpoint,
            artifacts=artifacts,
        )
        _ensure_secret_free(manifest, client)
        _publish_json_if_absent(manifest_path, manifest)
        written_manifest = _read_existing(manifest_path)
        if written_manifest is None:
            raise ValueError("published manifest is missing")
        return _validate_completed_manifest(
            written_manifest,
            snapshot_date=snapshot_date,
            run_id=run_id,
            storage_mode=storage_mode,
            output_dir=output_dir,
            bucket=bucket,
            s3_prefix=s3_prefix,
            expected_endpoint=endpoint,
            client=client,
        )
    except EnsembleArchiveError:
        raise
    except Exception as exc:
        raise EnsembleArchiveError(_safe_error_message(exc, client)) from None


def main(argv: list[str] | None = None) -> int:
    # Install the credential-redaction processor (#148) before anything can log: structlog's
    # DEFAULT chain has no redactor (#202).
    setup_logging()
    args = build_parser().parse_args(argv)
    try:
        manifest = run(args)
    except Exception as exc:
        print(json.dumps({"status": "error", "error": _safe_error_message(exc)}), file=sys.stderr)
        return 1
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0 if manifest.get("checks", {}).get("ok") is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
