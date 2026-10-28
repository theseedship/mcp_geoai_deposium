#!/usr/bin/env python3
"""Offline, provider-free B4 pilot over stored ERA5 and B3 NDVI files (#133).

A bounded local proof, not a panel delivery. Three subcommands:

``freeze``  write the source manifest: every ERA5 and B3 NDVI input pinned by sha256, the
            zones, the window, one output path per (zone, proposed tolerance) unit, and the
            code/config identity. Refuses to overwrite an existing manifest.
``run``     the #201 protocol, literally: ``run_resume plan`` → materialize each pending unit
            with ``unified_pilot_panel.py --ndvi-source-template`` → ``run_resume record`` →
            ``run_resume verify``. A blocked plan or a code/config drift stops before anything is
            written; a second run over unchanged bytes reuses every unit and writes nothing.
``report``  DuckDB readback of every source and output against the manifest and checkpoint log,
            per-zone coverage and the alignment sensitivity, and the digest of every file.

Provider-free by construction: the object-store bucket is cleared, local paths are confined to
``--local-root``, and the STAC collector is replaced by a function that raises, so a regression
that routed a unit to STAC would fail the unit instead of fetching. Nothing here can reach
Open-Meteo, ERA5/CDS or STAC, and nothing is written outside the output directory.

The tolerances are PROPOSALS for a sensitivity readout (ADR 0002 §3, FIERCE Decision 3). Nothing
in this script ratifies one.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import sys
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import duckdb  # noqa: E402

from scripts import unified_pilot_panel  # noqa: E402
from src.cli import run_resume as resume_cli  # noqa: E402
from src.config import settings  # noqa: E402
from src.run_resume import (  # noqa: E402
    SOURCE_MANIFEST_SCHEMA_VERSION,
    ResumeValidationError,
    SourceManifest,
)
from src.utils.code_ref import code_ref  # noqa: E402
from src.utils.dates import month_ranges  # noqa: E402
from src.utils.logging import setup_logging  # noqa: E402

KIND = "b4_local_offline_pilot"
DEFAULT_START = "2021-01-01"
DEFAULT_END = "2024-12-31"
PROPOSED_TOLERANCES = (1, 3, 5)
#: The code whose bytes decide what a unit produces. `run` refuses when any of them changed since
#: `freeze`, because the resume fingerprint covers the inputs and not the code that reads them.
CODE_FILES = (
    "scripts/b4_local_pilot.py",
    "scripts/unified_pilot_panel.py",
    "src/tools/climate.py",
    "src/run_resume.py",
    "src/cli/run_resume.py",
    "zones/zones.geojson",
)
SOURCES_NAME = "sources.json"
CHECKPOINTS_NAME = "checkpoints.jsonl"
PLAN_NAME = "plan.json"
REPORT_NAME = "report.json"
SUMMARY_DIGESTS_NAME = "summary-digests.json"


class PilotError(RuntimeError):
    """A refusal: the pilot stops rather than guess."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _code_identity() -> dict[str, Any]:
    return {
        "code_ref": code_ref(),
        "files": {name: _sha256(_REPO_ROOT / name) for name in CODE_FILES},
        "duckdb_version": duckdb.__version__,
    }


def _unit_id(zone_id: str, start: str, end: str, tolerance: int) -> str:
    return f"{zone_id}__{start[:4]}_{end[:4]}__pm{tolerance}d"


def _published_digests(args: argparse.Namespace) -> dict[str, dict[str, str]]:
    """``{kind: {zone_id: sha256}}`` from the release manifests, when the caller supplies them."""
    published: dict[str, dict[str, str]] = {}
    if args.era5_bundle_manifest:
        manifest = json.loads(Path(args.era5_bundle_manifest).read_text())
        published["era5"] = {item["zone_id"]: item["sha256"] for item in manifest["objects"]}
    if args.b3_verification:
        verification = json.loads(Path(args.b3_verification).read_text())
        published["b3_ndvi"] = {
            Path(item["relative_path"]).stem: item["sha256"] for item in verification["artifacts"]
        }
    return published


def freeze(args: argparse.Namespace) -> dict[str, Any]:
    out = Path(args.out).resolve()
    manifest_path = out / SOURCES_NAME
    if manifest_path.exists():
        raise PilotError(f"{manifest_path} is frozen; use a new --out to declare a new run")
    for day in (args.start_date, args.end_date):
        date.fromisoformat(day)
    published = _published_digests(args)
    units = []
    for zone_id in args.zone_id:
        if error := unified_pilot_panel._safe_zone_error(zone_id):
            raise PilotError(error)
        sources = []
        for kind, template in (("era5", args.era5_template), ("b3_ndvi", args.b3_template)):
            path = Path(template.format(zone_id=zone_id)).resolve()
            if not path.is_file():
                raise PilotError(f"{kind} source for {zone_id!r} is absent: {path}")
            digest = _sha256(path)
            expected = published.get(kind, {}).get(zone_id)
            if kind in published and expected != digest:
                raise PilotError(
                    f"{kind} source for {zone_id!r} does not match its published digest "
                    f"({digest} != {expected})"
                )
            sources.append({"kind": kind, "uri": path.as_posix(), "sha256": digest})
        for tolerance in args.tolerance_days:
            units.append(
                {
                    "unit_id": _unit_id(zone_id, args.start_date, args.end_date, tolerance),
                    "zone_id": zone_id,
                    "start_date": args.start_date,
                    "end_date": args.end_date,
                    "tolerance_days": tolerance,
                    "output_uri": (out / "unified" / f"pm{tolerance}d" / f"{zone_id}.parquet")
                    .resolve()
                    .as_posix(),
                    "sources": sources,
                }
            )
    payload = {
        "schema_version": SOURCE_MANIFEST_SCHEMA_VERSION,
        "kind": KIND,
        "run_id": args.run_id,
        "frozen_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "zone_ids": list(args.zone_id),
        "window": {"start_date": args.start_date, "end_date": args.end_date},
        "alignment": {
            "method": "nearest-date within ±N days (climate_enrich_ndvi)",
            "proposed_tolerance_days": list(args.tolerance_days),
            "status": "proposed for sensitivity readout only; not ratified (Decision 3)",
            "window_contract": "intersection",
        },
        "archives": [
            {"name": Path(path).name, "sha256": _sha256(Path(path))} for path in args.archive
        ],
        "published_digests_checked": sorted(published),
        "provenance_notes": list(args.note),
        "code_identity": _code_identity(),
        "output_dir": out.as_posix(),
        "units": units,
    }
    serialized = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    try:
        SourceManifest.from_mapping(
            payload, digest=hashlib.sha256(serialized.encode("utf-8")).hexdigest()
        )
    except ResumeValidationError as exc:
        raise PilotError(f"invalid frozen source manifest: {exc}") from exc
    out.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(serialized)
    return {"sources": manifest_path.as_posix(), "sha256": _sha256(manifest_path)}


def _load(out: Path) -> dict[str, Any]:
    manifest_path = out / SOURCES_NAME
    if not manifest_path.is_file():
        raise PilotError(f"no frozen manifest at {manifest_path}; run `freeze` first")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("kind") != KIND:
        raise PilotError(f"{manifest_path} is not a {KIND} manifest")
    return manifest


def _refuse_code_drift(manifest: dict[str, Any]) -> None:
    frozen = manifest["code_identity"]["files"]
    current_identity = _code_identity()
    current = current_identity["files"]
    drifted = sorted(name for name in frozen if frozen[name] != current.get(name))
    if drifted:
        raise PilotError(f"code/config drift since freeze: {', '.join(drifted)}")
    frozen_version = manifest["code_identity"].get("duckdb_version")
    if frozen_version != current_identity["duckdb_version"]:
        raise PilotError(
            "DuckDB runtime drift since freeze: "
            f"{frozen_version!r} != {current_identity['duckdb_version']!r}"
        )


def _provider_free(local_root: Path) -> None:
    """Pin the process to local files and make any STAC collection an error."""
    from src.tools import climate

    async def _no_stac(*_args: Any, **_kwargs: Any) -> Any:
        raise PilotError("STAC collection attempted in the provider-free pilot")

    settings.S3_BUCKET = ""
    settings.ALLOW_LOCAL_PARQUET_SOURCE = True
    settings.LOCAL_PARQUET_SOURCE_ROOT = local_root.as_posix()
    climate._collect_ndvi_observations = _no_stac


def _resume(command: str, out: Path, *extra: str) -> int:
    return resume_cli.main(
        [
            command,
            "--source-manifest",
            (out / SOURCES_NAME).as_posix(),
            "--checkpoints",
            (out / CHECKPOINTS_NAME).as_posix(),
            *extra,
        ]
    )


def _materialize(unit: dict[str, Any], out: Path) -> dict[str, Any]:
    sources = {source["kind"]: source["uri"] for source in unit["sources"]}
    wrapper_args = unified_pilot_panel.build_parser().parse_args(
        [
            "--zone-id",
            unit["zone_id"],
            "--start-date",
            unit["start_date"],
            "--end-date",
            unit["end_date"],
            "--storage-mode",
            "local",
            "--era5-source-template",
            sources["era5"],
            "--ndvi-source-template",
            sources["b3_ndvi"],
            "--output-template",
            unit["output_uri"],
            "--tolerance-days",
            str(unit["tolerance_days"]),
            "--output-dir",
            (out / "summaries" / unit["unit_id"]).as_posix(),
            "--run-id",
            unit["unit_id"],
            "--no-run-manifest",
            "--fail-fast",
        ]
    )
    Path(unit["output_uri"]).parent.mkdir(parents=True, exist_ok=True)
    panel = asyncio.run(unified_pilot_panel.run_panel(wrapper_args))
    zone = panel["zones"][0] if panel.get("zones") else {}
    stage = zone.get("stage") or {}
    if not panel.get("ok") or stage.get("ndvi_input") != "stored_b3_series":
        raise PilotError(f"unit {unit['unit_id']!r} failed: {zone.get('error') or stage}")
    if stage.get("images_searched") != 0:
        raise PilotError(f"unit {unit['unit_id']!r} reports a STAC search")
    return panel


def _summary_stage(unit: dict[str, Any], out: Path) -> tuple[dict[str, Any], str]:
    """Read the one expected wrapper summary and validate its unit identity."""
    summary_dir = out / "summaries" / unit["unit_id"]
    expected = summary_dir / (
        f"{unit['zone_id']}-unified-{unit['start_date']}_{unit['end_date']}.json"
    )
    candidates = list(summary_dir.glob(f"{unit['zone_id']}-unified-*.json"))
    if candidates != [expected]:
        raise PilotError(f"unit {unit['unit_id']!r} needs exactly its expected wrapper summary")
    try:
        raw = expected.read_bytes()
        zone = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PilotError(f"invalid wrapper summary for {unit['unit_id']!r}: {exc}") from exc
    sources = {source["kind"]: source["uri"] for source in unit["sources"]}
    if not isinstance(zone, dict) or not isinstance(zone.get("stage"), dict):
        raise PilotError(f"invalid wrapper summary shape for {unit['unit_id']!r}")
    stage = zone["stage"]
    expected_fields = {
        "ok": True,
        "zone_id": unit["zone_id"],
        "window": {"start_date": unit["start_date"], "end_date": unit["end_date"]},
        "tolerance_days": unit["tolerance_days"],
        "era5_source": sources["era5"],
        "ndvi_source": sources["b3_ndvi"],
        "unified_output": unit["output_uri"],
        "summary_path": expected.as_posix(),
    }
    if any(zone.get(key) != value for key, value in expected_fields.items()) or any(
        stage.get(key) != value
        for key, value in {
            "output": unit["output_uri"],
            "ndvi_input": "stored_b3_series",
            "images_searched": 0,
        }.items()
    ):
        raise PilotError(f"wrapper summary identity disagrees with unit {unit['unit_id']!r}")
    return stage, hashlib.sha256(raw).hexdigest()


def _summary_digests(
    out: Path, manifest: dict[str, Any], *, allow_missing: bool
) -> dict[str, dict[str, str]]:
    path = out / SUMMARY_DIGESTS_NAME
    if not path.is_file():
        if allow_missing:
            return {}
        raise PilotError(f"no frozen wrapper-summary digests at {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PilotError(f"invalid wrapper-summary digests at {path}: {exc}") from exc
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != 1
        or payload.get("source_manifest_sha256") != _sha256(out / SOURCES_NAME)
        or not isinstance(payload.get("summaries"), dict)
    ):
        raise PilotError(f"wrapper-summary digests at {path} do not bind to this manifest")
    digests = payload["summaries"]
    declared = {unit["unit_id"] for unit in manifest["units"]}
    if not set(digests) <= declared or any(
        not isinstance(value, dict)
        or set(value) != {"summary_sha256", "output_sha256"}
        or any(not isinstance(digest, str) for digest in value.values())
        for value in digests.values()
    ):
        raise PilotError(f"wrapper-summary digests at {path} declare invalid units")
    return digests


def _write_summary_digests(out: Path, digests: dict[str, dict[str, str]]) -> None:
    path = out / SUMMARY_DIGESTS_NAME
    payload = {
        "schema_version": 1,
        "source_manifest_sha256": _sha256(out / SOURCES_NAME),
        "summaries": digests,
    }
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def run(args: argparse.Namespace) -> int:
    out = Path(args.out).resolve()
    manifest = _load(out)
    _refuse_code_drift(manifest)
    _provider_free(Path(args.local_root).resolve())
    code = _resume("plan", out, "--output", (out / PLAN_NAME).as_posix())
    if code != 0:
        return code
    plan = json.loads((out / PLAN_NAME).read_text())
    units = {unit["unit_id"]: unit for unit in manifest["units"]}
    digests = _summary_digests(out, manifest, allow_missing=True)
    for entry in plan["units"]:
        if entry["status"] == "reuse":
            unit_id = entry["unit_id"]
            _, actual = _summary_stage(units[unit_id], out)
            pinned = digests.get(unit_id)
            if (
                pinned is None
                or pinned["summary_sha256"] != actual
                or pinned["output_sha256"] != _sha256(Path(units[unit_id]["output_uri"]))
            ):
                raise PilotError(
                    f"wrapper summary for reused unit {unit_id!r} is unpinned or modified"
                )
    pending = [entry["unit_id"] for entry in plan["units"] if entry["status"] == "pending"]
    completed_at = args.completed_at or datetime.now(UTC).isoformat(timespec="seconds")
    for unit_id in pending:
        _materialize(units[unit_id], out)
        _, summary_digest = _summary_stage(units[unit_id], out)
        digests[unit_id] = {
            "summary_sha256": summary_digest,
            "output_sha256": _sha256(Path(units[unit_id]["output_uri"])),
        }
        _write_summary_digests(out, digests)
        code = _resume(
            "record",
            out,
            "--plan",
            (out / PLAN_NAME).as_posix(),
            "--unit-id",
            unit_id,
            "--completed-at",
            completed_at,
        )
        if code != 0:
            return code
    return _resume("verify", out)


def _one(connection: duckdb.DuckDBPyConnection, sql: str, params: list[Any]) -> tuple:
    row = connection.execute(sql, params).fetchone()
    assert row is not None
    return row


def _point(connection: duckdb.DuckDBPyConnection, path: str) -> list[float] | None:
    """The sampling point a file carries, or ``None`` when it has no lat/lon columns."""
    columns = {
        item[0]
        for item in connection.execute("SELECT * FROM read_parquet(?) LIMIT 0", [path]).description
    }
    if not {"lat", "lon"} <= columns:
        return None
    return list(_one(connection, "SELECT min(lat), min(lon) FROM read_parquet(?)", [path]))


def _readback(manifest: dict[str, Any], checkpoints: dict[str, dict[str, Any]]) -> dict[str, Any]:
    start, end = manifest["window"]["start_date"], manifest["window"]["end_date"]
    requested_days = (date.fromisoformat(end) - date.fromisoformat(start)).days + 1
    requested_months = [slice_start[:7] for slice_start, _ in month_ranges(start, end)]
    zones: dict[str, Any] = {}
    config = {"autoinstall_known_extensions": False, "autoload_known_extensions": False}
    with duckdb.connect(":memory:", config=config) as connection:
        for unit in manifest["units"]:
            sources = {source["kind"]: source for source in unit["sources"]}
            zone_id = unit["zone_id"]
            for source in sources.values():
                if _sha256(Path(source["uri"])) != source["sha256"]:
                    raise PilotError(f"{source['uri']} no longer matches the manifest")
            if zone_id not in zones:
                era5 = sources["era5"]["uri"]
                b3 = sources["b3_ndvi"]["uri"]
                source_days, first, last, days, zone_ids, points = _one(
                    connection,
                    "SELECT count(DISTINCT date), min(date), max(date), "
                    "count(DISTINCT date) FILTER (WHERE date BETWEEN ? AND ?), "
                    "count(DISTINCT zone_id), count(DISTINCT (lat, lon)) FROM read_parquet(?)",
                    [start, end, era5],
                )
                present = {
                    str(row[0])
                    for row in connection.execute(
                        "SELECT DISTINCT date FROM read_parquet(?) WHERE date BETWEEN ? AND ?",
                        [era5, start, end],
                    ).fetchall()
                }
                b3_total, b3_first, b3_last = _one(
                    connection,
                    "SELECT count(*), min(date), max(date) FROM read_parquet(?) WHERE zone_id = ?",
                    [b3, zone_id],
                )
                observations = [
                    str(row[0])
                    for row in connection.execute(
                        "SELECT date FROM read_parquet(?) WHERE zone_id = ? "
                        "AND date BETWEEN ? AND ? ORDER BY date",
                        [b3, zone_id, start, end],
                    ).fetchall()
                ]
                observed_months = sorted({day[:7] for day in observations})
                all_days = [
                    date.fromordinal(date.fromisoformat(start).toordinal() + offset).isoformat()
                    for offset in range(requested_days)
                ]
                zones[zone_id] = {
                    "requested_period": f"{start}..{end}",
                    "era5": {
                        "source_span": f"{first}..{last}",
                        "source_days": source_days,
                        "days_in_window": days,
                        "days_requested": requested_days,
                        "missing_days_in_window": [d for d in all_days if d not in present],
                        "distinct_zone_ids": zone_ids,
                        "distinct_points": points,
                        "point_lat_lon": _point(connection, era5),
                    },
                    "b3_ndvi": {
                        "observations_total": b3_total,
                        "source_span": f"{b3_first}..{b3_last}",
                        "observations_in_window": len(observations),
                        "first_in_window": observations[0] if observations else None,
                        "last_in_window": observations[-1] if observations else None,
                        "months_requested": len(requested_months),
                        "months_observed": len(observed_months),
                        "months_without_observation": [
                            m for m in requested_months if m not in observed_months
                        ],
                        "point_lat_lon": _point(connection, b3),
                    },
                    "tolerances": {},
                    "_observations": observations,
                }
            output = Path(unit["output_uri"])
            checkpoint = checkpoints.get(unit["unit_id"])
            digest = _sha256(output)
            if checkpoint is None or checkpoint["output_sha256"] != digest:
                raise PilotError(f"{output} does not match its checkpoint")
            rows, first_day, last_day, dup, nulls = _one(
                connection,
                "SELECT count(*), min(date), max(date), count(*) - count(DISTINCT date), "
                "count(*) FILTER (WHERE ndvi_mean IS NULL OR ndvi_date IS NULL "
                "OR temperature_2m IS NULL) FROM read_parquet(?)",
                [output.as_posix()],
            )
            offsets = dict(
                connection.execute(
                    "SELECT day_delta, count(*) FROM read_parquet(?) GROUP BY 1 ORDER BY 1",
                    [output.as_posix()],
                ).fetchall()
            )
            matched = {
                str(row[0])
                for row in connection.execute(
                    "SELECT ndvi_date FROM read_parquet(?)", [output.as_posix()]
                ).fetchall()
            }
            observations = zones[zone_id]["_observations"]
            zones[zone_id]["tolerances"][f"pm{unit['tolerance_days']}d"] = {
                "unit_id": unit["unit_id"],
                "output": output.as_posix(),
                "sha256": digest,
                "rows": rows,
                "row_span": f"{first_day}..{last_day}",
                "duplicate_dates": dup,
                "required_nulls": nulls,
                "day_delta_counts": {str(k): v for k, v in offsets.items()},
                "matched_ndvi_dates": len(matched),
                "unmatched_ndvi_dates": [d for d in observations if d not in matched],
            }
    for zone in zones.values():
        zone.pop("_observations")
        digests = {entry["sha256"] for entry in zone["tolerances"].values()}
        zone["tolerance_outputs_byte_identical"] = len(digests) == 1
    return zones


def report(args: argparse.Namespace) -> int:
    out = Path(args.out).resolve()
    manifest = _load(out)
    _refuse_code_drift(manifest)
    code = _resume("verify", out)
    if code != 0:
        return code
    checkpoint_path = out / CHECKPOINTS_NAME
    if not checkpoint_path.is_file():
        raise PilotError(f"no checkpoints at {checkpoint_path}")
    records = [
        json.loads(line) for line in checkpoint_path.read_text().splitlines() if line.strip()
    ]
    checkpoints = {record["unit_id"]: record for record in records}
    declared = {unit["unit_id"] for unit in manifest["units"]}
    if set(checkpoints) != declared:
        raise PilotError("checkpoint log does not cover exactly the frozen units")
    digests = _summary_digests(out, manifest, allow_missing=False)
    if set(digests) != declared:
        raise PilotError("wrapper-summary digests do not cover exactly the frozen units")
    summaries = {}
    for unit in manifest["units"]:
        stage, actual = _summary_stage(unit, out)
        pinned = digests[unit["unit_id"]]
        if (
            pinned["summary_sha256"] != actual
            or pinned["output_sha256"] != checkpoints[unit["unit_id"]]["output_sha256"]
        ):
            raise PilotError(f"wrapper summary for {unit['unit_id']!r} was modified")
        summaries[unit["unit_id"]] = stage
    payload = {
        "kind": f"{KIND}_report",
        "run_id": manifest["run_id"],
        "source_manifest_sha256": _sha256(out / SOURCES_NAME),
        "checkpoints_sha256": _sha256(out / CHECKPOINTS_NAME),
        "zones": _readback(manifest, checkpoints),
        "wrapper_stages": summaries,
        "files": {
            path.relative_to(out).as_posix(): _sha256(path)
            for path in sorted(out.rglob("*"))
            if path.is_file() and path.name != REPORT_NAME
        },
    }
    (out / REPORT_NAME).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": "ok", "report": (out / REPORT_NAME).as_posix()}))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="b4_local_pilot", description=__doc__.split("\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)

    f = commands.add_parser("freeze", help="write the immutable source manifest")
    f.add_argument("--out", required=True)
    f.add_argument("--era5-template", required=True, help="local ERA5 path with {zone_id}")
    f.add_argument("--b3-template", required=True, help="local B3 NDVI path with {zone_id}")
    f.add_argument("--zone-id", nargs="+", default=list(unified_pilot_panel.PILOT_ZONE_IDS))
    f.add_argument("--start-date", default=DEFAULT_START)
    f.add_argument("--end-date", default=DEFAULT_END)
    f.add_argument("--tolerance-days", nargs="+", type=int, default=list(PROPOSED_TOLERANCES))
    f.add_argument("--run-id", default="b4-local-pilot")
    f.add_argument("--era5-bundle-manifest", help="release bundle-manifest.json to check against")
    f.add_argument("--b3-verification", help="release b3-verification JSON to check against")
    f.add_argument("--archive", action="append", default=[], help="release archive to pin")
    f.add_argument("--note", action="append", default=[], help="provenance note to carry")

    for name, text in (("run", "plan, materialize, record, verify"), ("report", "readback")):
        sub = commands.add_parser(name, help=text)
        sub.add_argument("--out", required=True)
        if name == "run":
            sub.add_argument("--local-root", required=True, help="root local reads/writes obey")
            sub.add_argument("--completed-at", help="checkpoint timestamp; default now (UTC)")
    return parser


def main(argv: list[str] | None = None) -> int:
    # The credential-redaction processor (#148): this drives src/ handlers in-process.
    setup_logging()
    args = build_parser().parse_args(argv)
    try:
        if args.command == "freeze":
            print(json.dumps({"status": "ok", **freeze(args)}))
            return 0
        return run(args) if args.command == "run" else report(args)
    except PilotError as exc:
        print(json.dumps({"status": "refused", "error": str(exc)}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
