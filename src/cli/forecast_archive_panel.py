"""Collect and export a frozen Previous Runs panel, resumably and offline-verifiably (#193).

``collect`` reads the frozen panel manifests (one per provider/model identity) and the frozen
sampling points their configuration binds by SHA-256, and requests each zone at its FROZEN
point, never at the current zone geometry. It keeps the provider's raw bytes beside a
provenance sidecar, writes every file create-only, and on resume trusts an artifact only if it
re-derives byte for byte from its retained raw response; anything else is an
``integrity_error`` that is never refetched or replaced. ``export`` flattens ONE identity into
the JSONL forecast rows ``forecast_skill_matrix`` already reads.

``run_panel`` below is the earlier current-coordinate collector. It is kept unchanged for its
two library callers (``forecast_forward_archive`` and ``forecast_benchmark_panel``); this CLI no
longer reaches it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Any
from urllib.parse import quote, quote_plus, urlsplit, urlunsplit

from src.cli.forecast_skill_matrix import _panel_manifest, _publish_create_only
from src.config import settings
from src.forecast_archive import (
    ArchiveResult,
    ArchiveRow,
    Coverage,
    PreviousRunsArchiveClient,
    RawResponse,
    ResponseRefused,
    normalize_previous_runs,
)
from src.forecast_backtest import BacktestValidationError
from src.forecast_skill import PanelManifest
from src.tools.climate import _load_zones, _scientific_zone_ids
from src.utils.duckdb_conn import duckdb_session
from src.utils.logging import setup_logging
from src.utils.openmeteo import NORMALIZED_VARS
from src.utils.s3_endpoint import publication_secrets, redact_credentials

_SAFE_S3_PART = re.compile(r"^[-A-Za-z0-9._=]+$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")

MANIFEST_SCHEMA_VERSION = "forecast-archive-panel/v2"
ARTIFACT_SCHEMA_VERSION = "forecast-archive-artifact/v1"
PROVENANCE_SCHEMA_VERSION = "forecast-archive-provenance/v1"
# What the client actually sends and how it reads the answer. A panel frozen under anything
# else describes a different collection, so it is refused rather than silently reinterpreted.
_CLIENT_CONFIGURATION = {
    "source": "openmeteo_previous_runs",
    "timezone": "UTC",
    "wind_speed_unit": "ms",
    "day_grain": "utc_day",
    "lead_semantics": "fixed_hourly_offset_24_times_horizon_days",
}


class ArchiveRefusal(ValueError):
    """A frozen input or an archived file that must stop the run, never be repaired."""


@dataclass(frozen=True)
class FrozenIdentity:
    panel: PanelManifest
    panel_sha256: str


@dataclass(frozen=True)
class FrozenPanel:
    """Every identity's manifest, proven to share one panel, plus its frozen points."""

    panel_id: str
    configuration: dict[str, Any]
    points: dict[str, tuple[float, float]]
    zones: tuple[str, ...]
    variables: tuple[str, ...]
    horizons: tuple[int, ...]
    start_date: str
    end_date: str
    identities: tuple[FrozenIdentity, ...]


def _safe_name(value: str, what: str) -> str:
    """Panel values become file names; one that could leave the archive directory is refused."""
    if ".." in value or not _SAFE_S3_PART.fullmatch(value):
        raise ArchiveRefusal(f"unsafe {what} for an archive file name: {value!r}")
    return value


def _sampling_points(decoded: Any) -> dict[str, tuple[float, float]]:
    items = decoded.get("sampling_points") if isinstance(decoded, dict) else None
    if not isinstance(items, list):
        raise ArchiveRefusal("sampling points file must hold a sampling_points list")
    points: dict[str, tuple[float, float]] = {}
    for item in items:
        zone = item.get("zone_id") if isinstance(item, dict) else None
        coordinates = (item.get("latitude"), item.get("longitude")) if zone else ()
        if (
            not isinstance(zone, str)
            or len(coordinates) != 2
            or any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                for value in coordinates
            )
        ):
            raise ArchiveRefusal(f"malformed sampling point: {item!r}")
        if zone in points:
            raise ArchiveRefusal(f"duplicate sampling point for zone {zone!r}")
        points[zone] = (float(coordinates[0]), float(coordinates[1]))
    return points


def _load_frozen_panel(panel_paths: list[Path], sampling_points: Path) -> FrozenPanel:
    """Load the frozen panel and refuse anything that is not exactly one panel.

    The manifests are parsed by the matrix engine's own loader, so collection and scoring
    cannot disagree on what a panel is. Across identities, every axis and the whole
    configuration must match; the configuration must be what the client sends; and the
    sampling file must hash to the digest the configuration froze, with a point for every zone.
    The points are requested coordinates from the accepted source archives, not grid cells.
    """
    if not panel_paths:
        raise ArchiveRefusal("at least one --panel-manifest is required")
    identities = tuple(FrozenIdentity(*_panel_manifest(path)) for path in panel_paths)
    first = identities[0].panel
    for identity in identities[1:]:
        for axis in ("configuration", "zones", "variables", "horizons_days", "valid_dates"):
            if getattr(identity.panel, axis) != getattr(first, axis):
                raise ArchiveRefusal(f"panel manifests disagree on {axis}")
    names = [(identity.panel.provider, identity.panel.model) for identity in identities]
    if len(set(names)) != len(names):
        raise ArchiveRefusal("a provider/model identity is declared by more than one manifest")
    for provider, model in names:
        _safe_name(provider, "provider")
        _safe_name(model, "model")
    for zone in first.zones:
        _safe_name(zone, "zone id")
    configuration = first.configuration
    for field, expected in _CLIENT_CONFIGURATION.items():
        if configuration.get(field) != expected:
            raise ArchiveRefusal(
                f"panel configuration {field} must be {expected!r}, got {configuration.get(field)!r}"
            )
    panel_id = configuration.get("panel_id")
    if not isinstance(panel_id, str) or not panel_id:
        raise ArchiveRefusal("panel configuration has no panel_id")
    if unknown := sorted(set(first.variables) - set(NORMALIZED_VARS)):
        raise ArchiveRefusal(f"panel declares variables Previous Runs is not read for: {unknown}")
    if any(horizon > 7 for horizon in first.horizons_days):
        raise ArchiveRefusal("Previous Runs stores horizons 1..7 days only")
    dates = sorted(first.valid_dates)
    if any((later - earlier).days != 1 for earlier, later in zip(dates, dates[1:], strict=False)):
        raise ArchiveRefusal(
            "panel valid_dates must be one contiguous range: Previous Runs is requested by "
            "start and end date"
        )
    body = sampling_points.read_bytes()
    digest = hashlib.sha256(body).hexdigest()
    if digest != configuration.get("sampling_points_sha256"):
        raise ArchiveRefusal(
            f"sampling points file hashes to {digest}, not the panel's sampling_points_sha256"
        )
    try:
        points = _sampling_points(json.loads(body))
    except ValueError as exc:
        if isinstance(exc, ArchiveRefusal):
            raise
        raise ArchiveRefusal("sampling points file is not valid JSON") from exc
    if missing := [zone for zone in first.zones if zone not in points]:
        raise ArchiveRefusal(f"no frozen sampling point for zone(s): {missing}")
    return FrozenPanel(
        panel_id=panel_id,
        configuration=dict(configuration),
        points={zone: points[zone] for zone in first.zones},
        zones=first.zones,
        variables=first.variables,
        horizons=first.horizons_days,
        start_date=dates[0].isoformat(),
        end_date=dates[-1].isoformat(),
        identities=identities,
    )


def _provider_error_message(exc: Exception, client: Any) -> str:
    """Serialize provider failures without retaining ANY configured credential.

    Two things were wrong with the previous two-item allowlist, and both are leaks rather than
    tidiness (#146). It knew only the weather API key, so an object-store credential — which
    httpfs quotes in full in its errors — passed straight through; and it iterated a ``set``,
    whose order is arbitrary, so a secret that is a PREFIX of a longer one could be replaced
    first and publish the longer one's tail. This repo's own dev store is exactly that shape:
    account ``deposium-dev``, password ``deposium-dev-secret``.

    ``publication_secrets`` answers both: it is the shared definition of every credential a
    published document may have to have punched out of it — store keys, the ambient ``AWS_*``
    environment (#147), and a private ``STAC_URL`` — already ordered LONGEST FIRST and already
    carrying the percent-encoded and JSON-escaped spellings this function open-coded for one
    value. The API key stays local, since it is this CLI's own and not an object-store concern.
    """
    message = str(exc)
    local = tuple(
        value.strip()
        for value in (getattr(client, "apikey", ""), settings.FORECAST_ARCHIVE_API_KEY)
        if isinstance(value, str) and value.strip()
    )
    # Longest first across BOTH sets, for the prefix reason above.
    secrets = sorted({*local, *publication_secrets(settings.S3_ENDPOINT)}, key=len, reverse=True)
    for secret in secrets:
        for candidate in (secret, quote(secret, safe=""), quote_plus(secret, safe="")):
            message = message.replace(candidate, "[REDACTED]")
    return message


def _s3_uri(*, bucket: str, prefix: str, run_id: str, name: str) -> str:
    parts = (bucket, run_id, name)
    if any(not value or ".." in value or not _SAFE_S3_PART.fullmatch(value) for value in parts):
        raise ValueError("unsafe S3 archive path")
    if (
        not prefix
        or ".." in prefix
        or any(not _SAFE_S3_PART.fullmatch(part) for part in prefix.split("/"))
    ):
        raise ValueError("unsafe S3 archive prefix")
    return f"s3://{bucket}/{prefix}/{run_id}/{name}"


def _read_json(path: str | Path) -> dict[str, Any]:
    if isinstance(path, Path):
        return json.loads(path.read_text(encoding="utf-8"))
    with duckdb_session(":memory:", load_httpfs=True) as conn:
        text = conn.execute("select content from read_text(?)", [path]).fetchone()[0]
    return json.loads(text)


def _s3_exists(path: str) -> bool:
    try:
        _read_json(path)
    except json.JSONDecodeError:
        return False
    except Exception as exc:
        message = str(exc).lower()
        if "404" in message or "not found" in message or "no files found" in message:
            return False
        raise
    return True


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    """Publish ``payload``, with every configured credential punched out of its prose first.

    This is the publication boundary — the one place a manifest becomes a file or an object —
    which is where the other three publishers in this repo redact
    (``climate_baseline_panel._write_s3_json``, ``unified_pilot_panel``,
    ``climate_manifest_consumer``). This one did not, and it writes to the same
    ``climate/runs/{run_id}/manifest.json`` prefix meant for downstream handoff (#146).

    BOTH branches, not just ``s3://``: a local manifest is routinely committed or uploaded
    later, and a redactor that depends on the destination is a redactor that fails on the path
    nobody tested. ``redact_credentials`` only rewrites the diagnostic fields, so run ids, URIs
    and zone ids survive verbatim and the published manifest still resolves.
    """
    payload = redact_credentials(payload, settings.S3_ENDPOINT)
    if isinstance(path, Path):
        path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
        return
    columns = list(payload)
    projections = []
    values: list[Any] = []
    for column in columns:
        value = payload[column]
        if isinstance(value, (dict, list)):
            projections.append(f'?::JSON AS "{column}"')
            values.append(json.dumps(value, sort_keys=True))
        else:
            projections.append(f'? AS "{column}"')
            values.append(value)
    with duckdb_session(":memory:", load_httpfs=True) as conn:
        conn.execute(f"COPY (SELECT {', '.join(projections)}) TO '{path}' (FORMAT JSON)", values)


def _provider_model(value: str) -> tuple[str, str]:
    provider, separator, model = value.partition(":")
    if not separator or not provider or not model:
        raise argparse.ArgumentTypeError("expected provider:model")
    return provider, model


def build_parser() -> argparse.ArgumentParser:
    """The frozen panel is the only input: zones, dates, horizons and identities come from it."""
    parser = argparse.ArgumentParser(prog="forecast_archive_panel")
    commands = parser.add_subparsers(dest="command", required=True)
    collect = commands.add_parser("collect", help="collect the frozen panel into a local archive")
    collect.add_argument(
        "--panel-manifest",
        action="append",
        required=True,
        type=Path,
        help="repeatable: one frozen forecast-skill-panel/v1 manifest per provider/model",
    )
    collect.add_argument(
        "--sampling-points",
        required=True,
        type=Path,
        help="frozen sampling points; must hash to configuration.sampling_points_sha256",
    )
    collect.add_argument("--output-dir", required=True, type=Path)
    collect.add_argument(
        "--run-id",
        required=True,
        help="names the create-only run manifest; a used run id is refused",
    )
    export = commands.add_parser(
        "export", help="flatten ONE identity into forecast_skill_matrix JSONL rows"
    )
    export.add_argument("--panel-manifest", required=True, type=Path, help="one identity")
    export.add_argument("--sampling-points", required=True, type=Path)
    export.add_argument("--archive-dir", required=True, type=Path)
    export.add_argument("--output", required=True, type=Path, help="create-only JSONL")
    return parser


def _json_bytes(payload: dict[str, Any]) -> bytes:
    """Canonical bytes, credentials punched out of diagnostic prose (see ``_write_json``)."""
    redacted = redact_credentials(payload, settings.S3_ENDPOINT)
    return (json.dumps(redacted, sort_keys=True) + "\n").encode("utf-8")


def _artifact_key(zone: str, panel: PanelManifest, frozen: FrozenPanel, horizon: int) -> str:
    return f"{zone}__{panel.provider}__{panel.model}__{frozen.start_date}__{frozen.end_date}__{horizon}"


def _provenance_name(frozen: FrozenPanel, zone: str, panel: PanelManifest, raw_sha256: str) -> str:
    """One provenance per REQUEST (zone x identity x response), not per response digest.

    Two requests can return identical bytes (the raw file is then shared), yet each was sent
    with its own model and parameters; keying provenance by digest alone would credit the
    second request with the first one's.
    """
    return (
        f"{zone}__{panel.provider}__{panel.model}__{frozen.start_date}__{frozen.end_date}"
        f"__{raw_sha256}.provenance.json"
    )


def _artifact(
    frozen: FrozenPanel,
    zone: str,
    panel: PanelManifest,
    rows: list[ArchiveRow],
    coverage: Coverage,
    raw_sha256: str,
) -> dict[str, Any]:
    latitude, longitude = frozen.points[zone]
    return {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "panel_id": frozen.panel_id,
        "configuration": frozen.configuration,
        "latitude": latitude,
        "longitude": longitude,
        "raw_sha256": raw_sha256,
        "provenance": _provenance_name(frozen, zone, panel, raw_sha256),
        "coverage": asdict(coverage),
        "rows": [
            {
                **asdict(row),
                "origin_date": row.origin_date.isoformat(),
                "valid_date": row.valid_date.isoformat(),
            }
            for row in rows
        ],
    }


def _public_endpoint(url: str) -> str:
    """Scheme, host and path only: a URL's userinfo or query never reaches the provenance."""
    parts = urlsplit(url)
    netloc = (parts.hostname or "") + (f":{parts.port}" if parts.port else "")
    return urlunsplit((parts.scheme, netloc, parts.path, "", ""))


def _provenance(
    client: Any, frozen: FrozenPanel, zone: str, panel: PanelManifest, raw: RawResponse
) -> dict[str, Any]:
    try:
        returned = json.loads(raw.body)
    except ValueError:
        returned = None  # a refused, non-JSON response: its bytes are still retained
    if not isinstance(returned, dict):
        returned = {}
    latitude, longitude = frozen.points[zone]
    return {
        "schema_version": PROVENANCE_SCHEMA_VERSION,
        "raw_sha256": raw.sha256,
        "raw_bytes": len(raw.body),
        "retrieved_at": raw.retrieved_at,
        "endpoint": _public_endpoint(getattr(client, "base_url", "") or ""),
        "params": raw.params,
        "panel_id": frozen.panel_id,
        "configuration": frozen.configuration,
        "zone_id": zone,
        "requested_coordinates": {"latitude": latitude, "longitude": longitude},
        "provider": panel.provider,
        "requested_model": panel.model,
        # The grid cell the provider actually answered for, and the frame it declared.
        "returned": {
            key: returned.get(key)
            for key in (
                "latitude",
                "longitude",
                "elevation",
                "utc_offset_seconds",
                "timezone",
                "hourly_units",
            )
        },
        # A Previous Runs response states neither, and its fixed 24 x horizon hour offsets can
        # mix initializations inside one UTC day. Null is the truthful value, not an inference
        # from the request.
        "attested_model": None,
        "attested_initialization": None,
    }


def _publish_raw(
    raw_dir: Path, raw: RawResponse, provenance: dict[str, Any], provenance_name: str
) -> None:
    """Retain the exact bytes, content-addressed, then this request's provenance; create-only.

    The raw name IS the digest, so a second retrieval of identical bytes (several horizons
    share one batch response; two models can answer alike) lands on the same file and only
    has to prove it is identical. Provenance is per request (``_provenance_name``): the
    horizons of one batch share it, and a retry of the same request keeps the first one's.
    """
    raw_path = raw_dir / f"{raw.sha256}.json"
    try:
        _publish_create_only(raw_path, raw.body)
    except FileExistsError:
        if hashlib.sha256(raw_path.read_bytes()).hexdigest() != raw.sha256:
            raise ArchiveRefusal(f"raw/{raw_path.name} exists with other bytes") from None
    try:
        _publish_create_only(raw_dir / provenance_name, _json_bytes(provenance))
    except FileExistsError:
        pass


def _verify_artifact(
    path: Path, frozen: FrozenPanel, panel: PanelManifest, zone: str, horizon: int
) -> dict[str, Any]:
    """Trust an archived artifact only if it re-derives, byte for byte, from its raw response.

    Resume and export both come through here. The raw bytes it names must still exist and
    still hash to the recorded digest, their provenance must name the same digest, and
    normalizing them under the frozen panel must reproduce the stored file exactly — rows,
    coverage, configuration and coordinates. Any difference is an ``ArchiveRefusal``: the
    artifact is never repaired, refetched or replaced, because a silent rewrite is how a
    published archive changes under its readers.
    """
    stored = path.read_bytes()
    try:
        artifact = json.loads(stored)
    except ValueError as exc:
        raise ArchiveRefusal(f"{path.name} is not valid JSON") from exc
    raw_sha256 = artifact.get("raw_sha256") if isinstance(artifact, dict) else None
    if not isinstance(raw_sha256, str) or not _SHA256.fullmatch(raw_sha256):
        raise ArchiveRefusal(f"{path.name} names no raw response")
    raw_path = path.parent / "raw" / f"{raw_sha256}.json"
    try:
        body = raw_path.read_bytes()
    except OSError as exc:
        raise ArchiveRefusal(f"raw response raw/{raw_path.name} is missing") from exc
    if hashlib.sha256(body).hexdigest() != raw_sha256:
        raise ArchiveRefusal(f"raw/{raw_path.name} no longer matches its digest")
    provenance_name = _provenance_name(frozen, zone, panel, raw_sha256)
    try:
        provenance = json.loads((raw_path.parent / provenance_name).read_bytes())
    except (OSError, ValueError) as exc:
        raise ArchiveRefusal(f"provenance raw/{provenance_name} is missing or unreadable") from exc
    if not isinstance(provenance, dict) or any(
        provenance.get(field) != expected
        for field, expected in (
            ("raw_sha256", raw_sha256),
            ("zone_id", zone),
            ("provider", panel.provider),
            ("requested_model", panel.model),
        )
    ):
        raise ArchiveRefusal(f"provenance raw/{provenance_name} names another request")
    try:
        payload = json.loads(body)
        if not isinstance(payload, dict):
            raise ValueError("raw response is not a JSON object")
        rows, coverage = normalize_previous_runs(
            payload,
            zone_id=zone,
            provider=panel.provider,
            model=panel.model,
            horizon_days=horizon,
            variables=list(frozen.variables),
            expected_start_date=frozen.start_date,
            expected_end_date=frozen.end_date,
        )
    except ValueError as exc:
        raise ArchiveRefusal(f"raw/{raw_path.name} does not normalize: {exc}") from exc
    if stored != _json_bytes(_artifact(frozen, zone, panel, rows, coverage, raw_sha256)):
        raise ArchiveRefusal(f"{path.name} does not re-derive from its raw response")
    return artifact


def collect(args: argparse.Namespace, *, client: Any | None = None) -> dict[str, Any]:
    """Collect the frozen panel into ``--output-dir``; every file is written exactly once.

    Layout: ``raw/<sha256>.json`` (the provider's bytes, shared by identical responses) and
    ``raw/<zone>__<provider>__<model>__<start>__<end>__<sha256>.provenance.json`` per request;
    ``<key>.json`` per zone x identity x horizon (rows, coverage, ``raw_sha256``); and
    ``manifest-<run_id>.json``. Local only: the handoff collects locally, with no S3 writes.
    A rerun resumes: a verified artifact is ``resumed``, a failed one is ``integrity_error``
    and stays untouched, and only absent artifacts are requested.
    """
    run_id = _safe_name(args.run_id, "run id")
    frozen = _load_frozen_panel(args.panel_manifest, args.sampling_points)
    output_dir: Path = args.output_dir
    manifest_path = output_dir / f"manifest-{run_id}.json"
    if os.path.lexists(manifest_path):
        raise ArchiveRefusal(
            f"{manifest_path.name} already exists; a run manifest is never replaced — "
            "resume under a new --run-id"
        )
    client = client or PreviousRunsArchiveClient()
    output_dir.mkdir(parents=True, exist_ok=True)
    entries: list[dict[str, Any]] = []
    for zone in frozen.zones:
        latitude, longitude = frozen.points[zone]
        for identity in frozen.identities:
            panel = identity.panel
            pending: dict[int, tuple[Path, dict[str, Any]]] = {}
            for horizon in frozen.horizons:
                path = output_dir / f"{_artifact_key(zone, panel, frozen, horizon)}.json"
                entry = {
                    "key": path.stem,
                    "zone_id": zone,
                    "provider": panel.provider,
                    "model": panel.model,
                    "horizon_days": horizon,
                    "output": path.name,
                }
                if not os.path.lexists(path):
                    pending[horizon] = (path, entry)
                    continue
                try:
                    artifact = _verify_artifact(path, frozen, panel, zone, horizon)
                    entries.append({**entry, "status": "resumed", "coverage": artifact["coverage"]})
                except (ArchiveRefusal, OSError) as exc:
                    entries.append({**entry, "status": "integrity_error", "error": str(exc)})
            if not pending:
                continue
            try:
                results = client.fetch_daily_multi(
                    zone_id=zone,
                    latitude=latitude,
                    longitude=longitude,
                    start_date=frozen.start_date,
                    end_date=frozen.end_date,
                    horizons=list(pending),
                    variables=list(frozen.variables),
                    provider=panel.provider,
                    model=panel.model,
                )
            except Exception as exc:
                results = dict.fromkeys(pending, exc)
            for horizon, (path, entry) in pending.items():
                result = results.get(horizon, ArchiveRefusal("no result for this horizon"))
                raw = result.raw if isinstance(result, (ArchiveResult, ResponseRefused)) else None
                # The bytes are retained BEFORE the result is judged: a response the normalizer
                # refused is kept too, so a refusal at the preflight leaves something to inspect.
                provenance_name = (
                    _provenance_name(frozen, zone, panel, raw.sha256) if raw is not None else None
                )
                try:
                    if raw is not None:
                        _publish_raw(
                            output_dir / "raw",
                            raw,
                            _provenance(client, frozen, zone, panel, raw),
                            provenance_name,
                        )
                except (ArchiveRefusal, OSError) as exc:
                    entries.append({**entry, "status": "integrity_error", "error": str(exc)})
                    continue
                if isinstance(result, ResponseRefused):
                    message = _provider_error_message(result, client)
                    entries.append(
                        {
                            **entry,
                            "status": "refused",
                            "error": message,
                            "raw_sha256": raw.sha256,
                            "provenance": provenance_name,
                        }
                    )
                    continue
                if isinstance(result, Exception):
                    message = _provider_error_message(result, client)
                    entries.append({**entry, "status": "provider_error", "error": message})
                    continue
                rows, coverage, raw = result
                artifact = _artifact(frozen, zone, panel, rows, coverage, raw.sha256)
                try:
                    _publish_create_only(path, _json_bytes(artifact))
                except (ArchiveRefusal, OSError) as exc:
                    entries.append({**entry, "status": "integrity_error", "error": str(exc)})
                    continue
                entries.append(
                    {**entry, "status": coverage.status, "coverage": artifact["coverage"]}
                )
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "run_id": run_id,
        "panel_id": frozen.panel_id,
        "sampling_points_sha256": frozen.configuration["sampling_points_sha256"],
        "valid_dates": {"start": frozen.start_date, "end": frozen.end_date},
        "zones": list(frozen.zones),
        "identities": [_identity_summary(identity, entries) for identity in frozen.identities],
        "entries": entries,
    }
    data = _json_bytes(manifest)
    try:
        _publish_create_only(manifest_path, data)
    except FileExistsError as exc:
        raise ArchiveRefusal(f"{manifest_path.name} appeared during the run; not replaced") from exc
    # Return what was published, not the pre-redaction object it was serialized from.
    return json.loads(data)


def export(args: argparse.Namespace) -> dict[str, Any]:
    """Flatten one identity's verified artifacts into the matrix engine's forecast JSONL.

    The scoring CLI's ``--provider``/``--model`` arguments are a declaration, not proof, so the
    proof happens here, before flattening: every artifact must pass ``_verify_artifact``, and
    every row must carry the panel's provider and model, be a declared pair and be unique.
    One failure refuses the whole export; nothing partial is published. An artifact that was
    never collected is listed, not refused: its pairs stay in the panel's denominator and the
    matrix reports them as ``not_submitted``.
    """
    frozen = _load_frozen_panel([args.panel_manifest], args.sampling_points)
    panel = frozen.identities[0].panel
    if os.path.lexists(args.output):
        raise ArchiveRefusal(f"{args.output} already exists; an export is never replaced")
    declared = panel.declares()
    seen: set[tuple[str, str, str, str]] = set()
    lines: list[dict[str, Any]] = []
    missing_artifacts: list[str] = []
    provenance: set[str] = set()
    for zone in frozen.zones:
        for horizon in frozen.horizons:
            path = args.archive_dir / f"{_artifact_key(zone, panel, frozen, horizon)}.json"
            if not os.path.lexists(path):
                missing_artifacts.append(path.name)
                continue
            artifact = _verify_artifact(path, frozen, panel, zone, horizon)
            # Verified above as THIS identity's request; named so the export traces to it.
            provenance.add(artifact["provenance"])
            for row in artifact["rows"]:
                if (row["provider"], row["model"]) != (panel.provider, panel.model):
                    raise ArchiveRefusal(
                        f"{path.name} holds a {row['provider']}/{row['model']} row, "
                        f"not {panel.provider}/{panel.model}"
                    )
                key = (row["zone_id"], row["origin_date"], row["valid_date"], row["variable"])
                if key in seen:
                    raise ArchiveRefusal(f"duplicate forecast key: {key}")
                origin, valid = date.fromisoformat(key[1]), date.fromisoformat(key[2])
                if (valid - origin).days != row["horizon_days"] or not declared(
                    key[0], origin, valid, key[3]
                ):
                    raise ArchiveRefusal(f"forecast key outside the frozen panel: {key}")
                seen.add(key)
                lines.append(
                    {
                        field: row[field]
                        for field in (
                            "zone_id",
                            "origin_date",
                            "valid_date",
                            "horizon_days",
                            "variable",
                            "value",
                        )
                    }
                )
    if not lines:
        raise ArchiveRefusal("no verified forecast row to export")
    data = "".join(json.dumps(line, sort_keys=True) + "\n" for line in lines).encode("utf-8")
    try:
        _publish_create_only(args.output, data)
    except FileExistsError as exc:
        raise ArchiveRefusal(f"{args.output} appeared during the export; not replaced") from exc
    return {
        "status": "exported",
        "provider": panel.provider,
        "model": panel.model,
        "panel_id": frozen.panel_id,
        "expected_pairs": panel.expected_pairs,
        "rows": len(lines),
        "missing_artifacts": missing_artifacts,
        "provenance": sorted(provenance),
        "sha256": hashlib.sha256(data).hexdigest(),
    }


def _identity_summary(identity: FrozenIdentity, entries: list[dict[str, Any]]) -> dict[str, Any]:
    """The panel's declared denominator beside what this run holds; never a shrunk one."""
    panel = identity.panel
    own = [
        entry
        for entry in entries
        if (entry["provider"], entry["model"]) == (panel.provider, panel.model)
    ]
    return {
        "provider": panel.provider,
        "model": panel.model,
        "panel_manifest_sha256": identity.panel_sha256,
        "expected_pairs": panel.expected_pairs,
        "available_pairs": sum(
            entry["coverage"]["available_pairs"] for entry in own if "coverage" in entry
        ),
        "statuses": {
            status: sum(entry["status"] == status for entry in own)
            for status in sorted({entry["status"] for entry in own})
        },
    }


def run_panel(
    args: argparse.Namespace, *, client: PreviousRunsArchiveClient | None = None
) -> dict[str, Any]:
    """Earlier current-coordinate collector, kept for its library callers only (see module)."""
    zones = _load_zones()
    zone_ids = args.zone_id or _scientific_zone_ids(zones)
    if unknown := sorted(set(zone_ids) - set(zones)):
        raise ValueError(f"unknown zone ids: {unknown}")
    if any(horizon < 1 or horizon > 7 for horizon in args.horizon_days):
        raise ValueError("horizon-days must be in 1..7")
    client = client or PreviousRunsArchiveClient()
    provider_models = args.provider_model or [
        ("gfs", "gfs_seamless"),
        ("aifs", "ecmwf_aifs025_single"),
    ]
    storage_mode = getattr(args, "storage_mode", "local")
    run_id = getattr(args, "run_id", None)
    if storage_mode == "s3" and not run_id:
        raise ValueError("--run-id is required with --storage-mode s3")
    bucket = getattr(args, "s3_bucket", None) or settings.S3_BUCKET
    if storage_mode == "s3" and not bucket:
        raise ValueError("--s3-bucket or configured S3_BUCKET is required with --storage-mode s3")
    prefix = getattr(args, "s3_prefix", "climate/forecast")
    if storage_mode == "local":
        args.output_dir.mkdir(parents=True, exist_ok=True)
    entries: list[dict[str, Any]] = []
    for zone_id in zone_ids:
        zone = zones[zone_id]
        for provider, model in provider_models:
            pending: dict[int, tuple[str | Path, str]] = {}
            for horizon in args.horizon_days:
                key = (
                    f"{zone_id}__{provider}__{model}__{args.start_date}__{args.end_date}__{horizon}"
                )
                output: str | Path = (
                    _s3_uri(bucket=bucket, prefix=prefix, run_id=run_id, name=f"{key}.json")
                    if storage_mode == "s3"
                    else args.output_dir / f"{key}.json"
                )
                cache_exists = output.exists() if isinstance(output, Path) else _s3_exists(output)
                if cache_exists:
                    try:
                        existing = _read_json(output)
                        coverage = existing["coverage"]
                        if not isinstance(coverage, dict) or any(
                            coverage.get(field) != expected
                            for field, expected in (
                                ("zone_id", zone_id),
                                ("provider", provider),
                                ("model", model),
                                ("horizon_days", horizon),
                                ("start_date", args.start_date),
                                ("end_date", args.end_date),
                            )
                        ):
                            raise ValueError("cached coverage does not match requested panel")
                        entries.append(
                            {
                                "key": key,
                                "status": "resumed",
                                "output": str(output),
                                "coverage": coverage,
                            }
                        )
                    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                        entries.append(
                            {
                                "key": key,
                                "status": "provider_error",
                                "output": str(output),
                                "error": f"invalid existing artifact: {exc}",
                            }
                        )
                    continue
                pending[horizon] = (output, key)
            if not pending:
                continue
            try:
                results = client.fetch_daily_multi(
                    zone_id=zone_id,
                    latitude=zone["lat"],
                    longitude=zone["lon"],
                    start_date=args.start_date,
                    end_date=args.end_date,
                    provider=provider,
                    model=model,
                    horizons=list(pending),
                )
                for horizon, (output, key) in pending.items():
                    try:
                        result = results[horizon]
                        if isinstance(result, Exception):
                            raise result
                        rows, coverage = result[0], result[1]
                        serialized_rows = [
                            {
                                **asdict(row),
                                "origin_date": row.origin_date.isoformat(),
                                "valid_date": row.valid_date.isoformat(),
                            }
                            for row in rows
                        ]
                        payload = {"coverage": asdict(coverage), "rows": serialized_rows}
                        _write_json(output, payload)
                        entries.append(
                            {
                                "key": key,
                                "status": coverage.status,
                                "output": str(output),
                                "coverage": asdict(coverage),
                            }
                        )
                    except Exception as exc:
                        entries.append(
                            {
                                "key": key,
                                "status": "provider_error",
                                "error": _provider_error_message(exc, client),
                            }
                        )
            except Exception as exc:
                for _output, key in pending.values():
                    entries.append(
                        {
                            "key": key,
                            "status": "provider_error",
                            "error": _provider_error_message(exc, client),
                        }
                    )
    manifest = {
        "schema_version": "forecast-archive-panel/v1",
        "start_date": args.start_date,
        "end_date": args.end_date,
        "zones": zone_ids,
        "entries": entries,
    }
    manifest_path: str | Path = (
        _s3_uri(bucket=bucket, prefix="climate/runs", run_id=run_id, name="manifest.json")
        if storage_mode == "s3"
        else args.output_dir / "manifest.json"
    )
    manifest["storage_mode"] = storage_mode
    manifest["run_id"] = run_id if storage_mode == "s3" else None
    if getattr(args, "write_manifest", True):
        _write_json(manifest_path, manifest)
    return manifest


def main(argv: list[str] | None = None) -> int:
    # Install the credential-redaction processor (#148) before anything can log: structlog's
    # DEFAULT chain has no redactor (#202).
    setup_logging()
    args = build_parser().parse_args(argv)
    try:
        result = collect(args) if args.command == "collect" else export(args)
    except (ArchiveRefusal, BacktestValidationError, OSError) as exc:
        refusal = {"status": "refused", "error": str(exc)}
        print(json.dumps(redact_credentials(refusal, settings.S3_ENDPOINT), sort_keys=True))
        return 1
    # stdout is a publication too: the values land in terminal scrollback and CI logs (#146).
    print(json.dumps(redact_credentials(result, settings.S3_ENDPOINT), indent=2, sort_keys=True))
    # A run that left any entry unusable is not a success, even though its manifest is written.
    usable = {"available", "partial_archive", "resumed"}
    return 0 if all(entry["status"] in usable for entry in result.get("entries", [])) else 1


if __name__ == "__main__":
    raise SystemExit(main())
