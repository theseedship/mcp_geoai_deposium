"""Local input inventory only: never execute a weather rule or contact a provider.

Requires the dev extra for local JSON Schema validation. Hashes bind supplied
bytes; they do not authenticate model initialization, labels or scientific use.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import stat
from datetime import date
from pathlib import Path
from tempfile import TemporaryDirectory

import duckdb
from jsonschema import Draft202012Validator, FormatChecker
from referencing import Registry, Resource

from src.cli.forecast_archive_panel import (
    _artifact_key,
    _load_frozen_panel,
    _provenance_name,
    _verify_artifact,
)
from src.cli.forecast_skill_matrix import (
    _publish_create_only,
    _reject_constant,
    _reject_duplicate_keys,
)
from src.utils.logging import setup_logging
from src.watch_replay import HORIZONS, _validate_archives, _validate_labelset, _validate_normals

_SHA256 = re.compile(r"[0-9a-f]{64}")
_JSON_LIMIT = 2 * 1024 * 1024
_FILE_LIMIT = 64 * 1024 * 1024
_SNAPSHOT_LIMIT = 128 * 1024 * 1024
_GUARDS = {
    "scientific_acceptance": "pending",
    "operational_rule_parity": "unverified",
    "live_activation": False,
    "forecast_origin_authentication": "not_established_by_hashes",
}


class PreflightRefusal(ValueError):
    """Malformed or integrity-invalid input; no completed inventory is claimed."""


def _preflight_path(root: Path, value: object) -> Path:
    if not isinstance(value, str) or not value or len(value) > 4096:
        raise PreflightRefusal("Input paths must be nonempty relative strings")
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts or "\x00" in value:
        raise PreflightRefusal("Input paths must stay under the descriptor directory")
    path = root / relative
    if any(part.is_symlink() for part in (path, *path.parents)):
        raise PreflightRefusal("Symlink inputs are refused")
    return path


def _preflight_bytes(path: Path, limit: int = _FILE_LIMIT) -> bytes:
    if not stat.S_ISREG(path.stat().st_mode) or path.stat().st_size > limit:
        raise PreflightRefusal(f"Input is not a bounded regular file: {path.name}")
    with path.open("rb") as stream:
        body = stream.read(limit + 1)
    if len(body) > limit:
        raise PreflightRefusal(f"Input exceeds byte limit: {path.name}")
    return body


def _preflight_json(path: Path, limit: int = _JSON_LIMIT, *, body: bytes | None = None) -> dict:
    try:
        value = json.loads(
            _preflight_bytes(path, limit) if body is None else body,
            object_pairs_hook=_reject_duplicate_keys("preflight JSON"),
            parse_constant=_reject_constant,
        )
    except (ValueError, RecursionError, UnicodeError) as error:
        raise PreflightRefusal(f"Invalid JSON: {path.name}") from error
    if not isinstance(value, dict):
        raise PreflightRefusal(f"Expected a JSON object: {path.name}")
    pending = [value]
    while pending:
        item = pending.pop()
        if isinstance(item, dict):
            pending.extend(item.values())
        elif isinstance(item, list):
            pending.extend(item)
        elif isinstance(item, float) and not math.isfinite(item):
            raise PreflightRefusal(f"Nonfinite JSON number: {path.name}")
    return value


def _preflight_pin(
    root: Path, spec: object, *, json_file: bool = False
) -> tuple[Path, str, dict | None]:
    if not isinstance(spec, dict) or set(spec) - {"path", "sha256", "source"}:
        raise PreflightRefusal("Expected a pinned {path, sha256} input")
    digest = spec.get("sha256")
    if not isinstance(digest, str) or not _SHA256.fullmatch(digest):
        raise PreflightRefusal("Expected a lowercase SHA-256 digest")
    path = _preflight_path(root, spec.get("path"))
    body = _preflight_bytes(path, _JSON_LIMIT if json_file else _FILE_LIMIT)
    if hashlib.sha256(body).hexdigest() != digest:
        raise PreflightRefusal(f"SHA-256 mismatch: {path.name}")
    return path, digest, _preflight_json(path, body=body) if json_file else None


def _preflight_parquet(path: Path, *, source: bool = False) -> list[dict]:
    # No shared spatial/httpfs session: no extension loading or installation.
    with duckdb.connect(
        ":memory:",
        config={"autoinstall_known_extensions": False, "autoload_known_extensions": False},
    ) as conn:
        query = (
            "SELECT DISTINCT zone_id, lat, lon FROM read_parquet(?) LIMIT 3"
            if source
            else "SELECT * FROM read_parquet(?) LIMIT 73"
        )
        result = conn.execute(query, [str(path)])
        columns = [field[0] for field in result.description]
        rows = [dict(zip(columns, row, strict=True)) for row in result.fetchall()]
    if not rows or len(rows) > (1 if source else 72):
        raise PreflightRefusal("Parquet needs one source point or at most 72 normals rows")
    return rows


def _preflight_point(rows: list[dict], zone: str) -> dict:
    points = set()
    for row in rows:
        latitude, longitude = row.get("lat"), row.get("lon")
        if row.get("zone_id") != zone or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            for value in (latitude, longitude)
        ):
            raise PreflightRefusal("Parquet has missing/nonfinite coordinates or another zone")
        if not -90 <= latitude <= 90 or not -180 <= longitude <= 180:
            raise PreflightRefusal("Parquet coordinates are out of range")
        points.add((latitude, longitude))
    if len(points) != 1:
        raise PreflightRefusal("Parquet declares multiple sampling points")
    latitude, longitude = points.pop()
    return {"latitude": latitude, "longitude": longitude}


def _preflight_normals(
    root: Path, spec: dict, zone: str, frozen: dict, required_months: set[int], reasons: list
) -> dict:
    path, digest, _ = _preflight_pin(root, spec)
    rows = _preflight_parquet(path)
    window = _validate_normals(rows, zone)
    point = _preflight_point(rows, zone)
    if any(type(row.get("month")) is not int or not 1 <= row["month"] <= 12 for row in rows):
        raise PreflightRefusal("Normals months must be integers in 1..12")
    rain = [row for row in rows if row["variable"] == "precipitation"]
    months = sorted(row["month"] for row in rain)
    needed = [row for row in rain if row["month"] in required_months]
    if not required_months <= set(months) or any(
        type(row.get("schema_version")) is not int or row["schema_version"] != 2 for row in needed
    ):
        reasons.append({"code": "precipitation_v2_reference_incomplete", "months": months})
    invalid, unavailable = [], []
    for row in needed:
        values = [row.get(field) for field in ("p10_wet", "p50_wet", "p90_wet")]
        count, total, threshold = row.get("n_wet"), row.get("n"), row.get("wet_day_threshold_mm")
        finite_threshold = type(threshold) in (int, float) and math.isfinite(threshold)
        if (
            type(count) is not int
            or type(total) is not int
            or not 0 <= count <= total
            or not finite_threshold
            or threshold < 0
        ):
            invalid.append(row["month"])
        elif count == 0 and all(value is None for value in values):
            unavailable.append(row["month"])
        elif (
            count == 0
            or not all(type(value) in (int, float) and math.isfinite(value) for value in values)
            or not threshold <= values[0] <= values[1] <= values[2]
        ):
            invalid.append(row["month"])
    if invalid:
        reasons.append({"code": "wet_day_reference_invalid", "months": invalid})
    if unavailable:
        reasons.append({"code": "wet_day_reference_unavailable", "months": unavailable})
    report = {
        "sha256": digest,
        "rows": len(rows),
        "window_label": window,
        "declared_point": point,
        "precipitation_months": months,
        "required_months": sorted(required_months),
        "wet_day_threshold_mm": sorted(
            {
                row["wet_day_threshold_mm"]
                for row in needed
                if type(row.get("wet_day_threshold_mm")) in (int, float)
                and math.isfinite(row["wet_day_threshold_mm"])
            }
        ),
        "geometry_lineage": "unverified",
    }
    if point != frozen:
        reasons.append({"code": "normals_frozen_point_mismatch"})
    source_spec = spec.get("source")
    if source_spec is None:
        reasons.append({"code": "normals_source_absent"})
    else:
        source, source_digest, _ = _preflight_pin(root, source_spec)
        source_point = _preflight_point(_preflight_parquet(source, source=True), zone)
        report.update(source_sha256=source_digest, source_point=source_point)
        if source_point != point:
            reasons.append({"code": "normals_source_point_mismatch"})
        if source_point != frozen:
            reasons.append({"code": "normals_source_frozen_point_mismatch"})
        if source_point == point == frozen:
            report["geometry_lineage"] = "coordinate_match_only_not_computation_proof"
    return report


def _preflight_verify_artifact(path: Path, frozen, panel, zone: str, horizon: int) -> dict:
    """Confine malformed retained-input errors to the native decoding boundary."""
    try:
        artifact = _verify_artifact(path, frozen, panel, zone, horizon)
    except (TypeError, OverflowError) as error:
        raise PreflightRefusal(
            f"Invalid retained archive input at horizon {horizon}: {error}"
        ) from error
    if any(row["variable"] == "precipitation" and row["value"] < 0 for row in artifact["rows"]):
        raise PreflightRefusal(f"Archived precipitation cannot be negative (horizon {horizon})")
    return artifact


def _preflight_labels(root: Path, spec: dict, zone: str, reasons: list) -> dict:
    _, digest, document = _preflight_pin(root, spec, json_file=True)
    # Exactly these local resources; the registry has no network retriever.
    contracts = Path(__file__).resolve().parents[1] / "contracts_data/watch"
    schemas = [
        _preflight_json(contracts / f"{name}.v1.schema.json")
        for name in ("watch-label", "watch-labelset")
    ]
    registry = Registry().with_resources(
        (schema["$id"], Resource.from_contents(schema)) for schema in schemas
    )
    validator = Draft202012Validator(schemas[1], registry=registry, format_checker=FormatChecker())
    if next(validator.iter_errors(document), None) is not None:
        raise PreflightRefusal("Labelset does not satisfy the committed local schemas")
    variable, start, end, labels = _validate_labelset(document, zone)
    if variable != "precipitation":
        raise PreflightRefusal("This preflight accepts precipitation labels only")
    if len(labels) < 5:
        reasons.append({"code": "label_reporting_floor_not_met", "label_count": len(labels)})
    return {
        "sha256": digest,
        "label_count": len(labels),
        "variable": variable,
        "window_start": start.isoformat(),
        "window_end": end.isoformat(),
        "exhaustive": document["exhaustive"],
        "provenance": "declarations_not_authenticated",
    }


def _preflight_inventory(inputs: Path, expected_sha256: str) -> dict:
    """Inventory one local identity. Missing prerequisites block; malformed bytes refuse."""
    inputs = inputs.absolute()
    root = inputs.parent
    _, digest, document = _preflight_pin(
        root, {"path": inputs.name, "sha256": expected_sha256}, json_file=True
    )
    if document.get("schema_version") != "geoai-watch-replay-inputs/v1" or set(document) - {
        "schema_version",
        "zone_id",
        "panel",
        "sampling_points",
        "archive_dir",
        "normals",
        "labelset",
    }:
        raise PreflightRefusal("Expected geoai-watch-replay-inputs/v1 descriptor")
    zone = document.get("zone_id")
    if not isinstance(zone, str) or not zone:
        raise PreflightRefusal("zone_id must be a nonempty string")
    panel_path, panel_digest, _ = _preflight_pin(root, document.get("panel"), json_file=True)
    points_path, points_digest, _ = _preflight_pin(
        root, document.get("sampling_points"), json_file=True
    )
    frozen = _load_frozen_panel([panel_path], points_path)
    if zone not in frozen.zones:
        raise PreflightRefusal("zone_id is not declared by the frozen panel")
    panel = frozen.identities[0].panel
    latitude, longitude = frozen.points[zone]
    point = {"latitude": latitude, "longitude": longitude}
    reasons, artifacts, available = [], [], []
    if "precipitation" not in frozen.variables:
        reasons.append({"code": "panel_precipitation_absent"})
    missing = sorted(set(HORIZONS) - set(frozen.horizons))
    if missing:
        reasons.append({"code": "missing_required_horizons", "horizons": missing})
    archive_dir = (
        _preflight_path(root, document["archive_dir"])
        if document.get("archive_dir") is not None
        else None
    )
    if archive_dir is None:
        reasons.append({"code": "archive_directory_absent"})
    for horizon in frozen.horizons:
        name = f"{_artifact_key(zone, panel, frozen, horizon)}.json"
        path = (
            _preflight_path(root, str((archive_dir / name).relative_to(root)))
            if archive_dir
            else None
        )
        if path is None or not path.exists():
            reasons.append({"code": "archive_artifact_absent", "horizon": horizon})
            continue
        stored = _preflight_json(path)
        raw_digest = stored.get("raw_sha256")
        if not isinstance(raw_digest, str) or not _SHA256.fullmatch(raw_digest):
            raise PreflightRefusal("Archive artifact has no valid raw digest")
        for child in (f"{raw_digest}.json", _provenance_name(frozen, zone, panel, raw_digest)):
            _preflight_json(
                _preflight_path(root, str((path.parent / "raw" / child).relative_to(root))),
                _FILE_LIMIT,
            )
        artifact = _preflight_verify_artifact(path, frozen, panel, zone, horizon)
        artifacts.append(artifact)
        available.append(
            {
                "horizon": horizon,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "raw_sha256": raw_digest,
                "provenance_sha256": hashlib.sha256(
                    (
                        path.parent / "raw" / _provenance_name(frozen, zone, panel, raw_digest)
                    ).read_bytes()
                ).hexdigest(),
                "coverage": artifact["coverage"],
            }
        )
    if len(artifacts) == len(HORIZONS):
        _validate_archives(artifacts)
    report = {
        "schema_version": "watch-replay-preflight/v1",
        **_GUARDS,
        "inputs_sha256": digest,
        "zone_id": zone,
        "identity": {"provider": panel.provider, "model": panel.model},
        "panel_sha256": panel_digest,
        "sampling_points_sha256": points_digest,
        "frozen_point": point,
        "required_horizons": list(HORIZONS),
        "declared_horizons": list(frozen.horizons),
        "missing_horizons": sorted(set(HORIZONS) - {row["horizon"] for row in available}),
        "archives": available,
        "normals": None,
        "labelset": None,
    }
    required_dates = set(panel.valid_dates)
    if document.get("labelset") is None:
        reasons.append({"code": "labelset_absent"})
    else:
        report["labelset"] = _preflight_labels(root, document["labelset"], zone, reasons)
        required_dates = _preflight_label_coverage(report["labelset"], panel, artifacts, reasons)
    if document.get("normals") is None:
        reasons.append({"code": "normals_absent"})
    else:
        report["normals"] = _preflight_normals(
            root, document["normals"], zone, point, {day.month for day in required_dates}, reasons
        )
    report.update(status="blocked" if reasons else "inputs_verified", reasons=reasons)
    return report


def _preflight_label_coverage(labelset: dict, panel, artifacts: list, reasons: list) -> set[date]:
    start, end = (date.fromisoformat(labelset[key]) for key in ("window_start", "window_end"))
    dates = set(panel.valid_dates)
    if start < min(dates) or end > max(dates):
        reasons.append({"code": "label_window_outside_panel"})
    required = {day for day in dates if start <= day <= end}
    for artifact in artifacts:
        present = {
            date.fromisoformat(row["valid_date"])
            for row in artifact["rows"]
            if row["variable"] == "precipitation"
        }
        missing = sorted(required - present)
        if missing:
            reasons.append(
                {
                    "code": "archive_precipitation_dates_missing",
                    "horizon": artifact["coverage"]["horizon_days"],
                    "dates": [day.isoformat() for day in missing],
                }
            )
    return required or dates


def _preflight_snapshot(inputs: Path, destination: Path, expected_sha256: str) -> Path:
    """Capture each selected local file once; reused readers see only read-only copies."""
    root, captured, total = inputs.parent, set(), 0

    def capture(relative: object, *, json_file: bool = False, limit: int = _FILE_LIMIT) -> Path:
        nonlocal total
        source = _preflight_path(root, relative)
        target = destination / source.relative_to(root)
        if target not in captured:
            body = _preflight_bytes(source, limit)
            total += len(body)
            if total > _SNAPSHOT_LIMIT:
                raise PreflightRefusal("Selected inputs exceed the 128 MiB snapshot limit")
            if json_file:
                _preflight_json(source, limit, body=body)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(body)
            target.chmod(0o400)
            captured.add(target)
        return target

    def capture_pin(spec: object, *, json_file: bool = False) -> Path:
        if not isinstance(spec, dict):
            raise PreflightRefusal("Expected a pinned input object")
        target = capture(
            spec.get("path"), json_file=json_file, limit=_JSON_LIMIT if json_file else _FILE_LIMIT
        )
        _preflight_pin(destination, spec, json_file=json_file)
        return target

    descriptor = capture(inputs.name, json_file=True, limit=_JSON_LIMIT)
    _, _, document = _preflight_pin(
        destination, {"path": descriptor.name, "sha256": expected_sha256}, json_file=True
    )
    panel = capture_pin(document.get("panel"), json_file=True)
    points = capture_pin(document.get("sampling_points"), json_file=True)
    frozen = _load_frozen_panel([panel], points)
    zone = document.get("zone_id")
    if not isinstance(zone, str) or zone not in frozen.zones:
        raise PreflightRefusal("zone_id is not declared by the frozen panel")
    directory = document.get("archive_dir")
    if directory is not None:
        source_dir = _preflight_path(root, directory)
        for horizon in frozen.horizons:
            name = f"{_artifact_key(zone, frozen.identities[0].panel, frozen, horizon)}.json"
            relative = str((source_dir / name).relative_to(root))
            if not _preflight_path(root, relative).exists():
                continue
            stored = _preflight_json(capture(relative, json_file=True, limit=_JSON_LIMIT))
            raw_digest = stored.get("raw_sha256")
            if not isinstance(raw_digest, str) or not _SHA256.fullmatch(raw_digest):
                raise PreflightRefusal("Archive artifact has no valid raw digest")
            for child in (
                f"{raw_digest}.json",
                _provenance_name(frozen, zone, frozen.identities[0].panel, raw_digest),
            ):
                capture(str((source_dir / "raw" / child).relative_to(root)), json_file=True)
    for key in ("normals", "labelset"):
        spec = document.get(key)
        if spec is not None:
            capture_pin(spec, json_file=key == "labelset")
            if key == "normals" and spec.get("source") is not None:
                capture_pin(spec["source"])
    for directory in sorted(
        (path for path in destination.rglob("*") if path.is_dir()),
        key=lambda path: len(path.parts),
        reverse=True,
    ):
        directory.chmod(0o500)
    destination.chmod(0o500)
    return descriptor


def preflight_watch_inputs(inputs: Path, expected_sha256: str) -> dict:
    """Verify mechanically usable inputs from a bounded private snapshot, never run a rule."""
    inputs = inputs.absolute()
    # Authenticate the descriptor before it can select any other local input.
    _preflight_pin(inputs.parent, {"path": inputs.name, "sha256": expected_sha256}, json_file=True)
    with TemporaryDirectory(prefix="geoai-watch-preflight-") as temporary:
        descriptor = _preflight_snapshot(inputs, Path(temporary), expected_sha256)
        return _preflight_inventory(descriptor, expected_sha256)


def main(argv: list[str] | None = None) -> int:
    setup_logging()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", required=True, type=Path)
    parser.add_argument("--expected-inputs-sha256", required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        report = preflight_watch_inputs(args.inputs, args.expected_inputs_sha256)
        encoded = (json.dumps(report, sort_keys=True, allow_nan=False, indent=2) + "\n").encode()
        if args.output:
            _publish_create_only(args.output, encoded)
        else:
            print(encoded.decode(), end="")
        return 0
    except (OSError, ValueError, OverflowError, duckdb.Error) as error:
        report = {
            "schema_version": "watch-replay-preflight/v1",
            **_GUARDS,
            "status": "refused",
            "reasons": [{"code": "input_or_publication_refused", "detail": str(error)}],
        }
        print(json.dumps(report, sort_keys=True, allow_nan=False))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
