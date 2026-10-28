#!/usr/bin/env python3
"""Provisional CPU replay of an authenticated local Q3 handoff, without providers."""

from __future__ import annotations

import argparse
import calendar
import copy
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import subprocess
import sys
import tempfile
import time
import zipfile
from datetime import UTC, date, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.patterns.cross_validation import (  # noqa: E402
    cross_validate_mined_patterns,
    validated_library_to_dataframe,
)
from src.patterns.library_schema import PatternLibrary  # noqa: E402
from src.patterns.offline_bundle import (  # noqa: E402
    MAX_ARCHIVE_BYTES,
    MAX_JSON_BYTES,
    OfflineBundleError,
    _strict_json,
    _verify_bundle,
)
from src.patterns.so4gp_adapter import (  # noqa: E402
    mine_gradual_patterns,
    mine_temporal_patterns,
)

FEATURES = (
    "ndvi_mean",
    "temperature_2m",
    "precipitation",
    "humidity",
    "evapotranspiration",
    "wind_speed",
    "solar_radiation",
)
MONTHS = ("2023_07", "2023_08", "2023_09")
START, END = date(2023, 7, 1), date(2023, 9, 30)
CAVEATS = [
    "Provisional replay; no scientific reception or calibrated confidence.",
    "Round-robin folds reconstructed from authenticated zone order; historical folds absent.",
    "GRAANK has no target constraint; native conf may use an arbitrary first-item denominator.",
    "Native producer metrics remain per zone, never copied into cross-zone aggregates.",
    "NDVI temporal target and validation thresholds remain provisional.",
    "Svalbard July retains the historical STAC truncation warning.",
    "Kerguelen remains excluded because its complete Q3 input contains one row.",
    "Sparse NDVI observation dates do not establish daily coverage or temporal generalisation.",
]
DUCKDB_CONFIG = {"autoinstall_known_extensions": "false", "autoload_known_extensions": "false"}
# The existing flat exchange uses JSON/ISO strings for nested/date fields.
# Cast both empty and populated tables: pandas cannot infer null-only types.
VALIDATED_PARQUET_TYPES = {
    "pattern_id": "BIGINT",
    "antecedent": "VARCHAR",
    "consequent": "VARCHAR",
    "pattern": "VARCHAR",
    "variables": "VARCHAR",
    "support": "DOUBLE",
    "confidence": "DOUBLE",
    "is_temporal": "BOOLEAN",
    "lag_days": "DOUBLE",
    "algorithm": "VARCHAR",
    "zone_id": "VARCHAR",
    "zones_validated": "VARCHAR",
    "period_start": "VARCHAR",
    "period_end": "VARCHAR",
    "n_folds": "BIGINT",
    "stability": "DOUBLE",
    "created_at": "VARCHAR",
    "extra_metrics": "VARCHAR",
}


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--zones-geojson", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser


def _write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _zone_order(zones_path, input_manifests):
    if not zones_path.is_file() or zones_path.stat().st_size > MAX_JSON_BYTES:
        raise OfflineBundleError("Local zones GeoJSON absent or too large")
    data = zones_path.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    geo = _strict_json(data)
    if not isinstance(geo, dict) or geo.get("type") != "FeatureCollection":
        raise OfflineBundleError("Invalid zones GeoJSON")
    features = geo.get("features")
    if not isinstance(features, list):
        raise OfflineBundleError("Zones features missing")
    ids = []
    for feature in features:
        props = feature.get("properties") if isinstance(feature, dict) else None
        zone = props.get("zone_id") if isinstance(props, dict) else None
        if not isinstance(zone, str) or not re.fullmatch(r"[a-z0-9_]+", zone):
            raise OfflineBundleError("Invalid zone identifier")
        ids.append(zone)
    if len(ids) != 15 or len(set(ids)) != 15 or "kerguelen" not in ids or "svalbard" not in ids:
        raise OfflineBundleError("Expected the complete historical 15-zone inventory")
    windows = set()
    for manifest in input_manifests.values():
        if not isinstance(manifest, dict):
            raise OfflineBundleError("Monthly input manifest must be a JSON object")
        artifact = manifest.get("zone_artifact", {})
        params = manifest.get("parameters", {})
        if not isinstance(artifact, dict) or not isinstance(params, dict):
            raise OfflineBundleError("Monthly zone_artifact and parameters must be JSON objects")
        month = params.get("artifact_window_label")
        if month not in MONTHS or month in windows:
            raise OfflineBundleError("Missing, duplicate or unexpected monthly manifest")
        windows.add(month)
        year, number = map(int, month.split("_"))
        if (
            manifest.get("schema_version") != 1
            or manifest.get("kind") != "geoai_run_manifest"
            or artifact.get("sha256") != digest
            or artifact.get("selected_zone_ids") != ids
            or artifact.get("n_selected") != 15
            or params.get("zone_ids") != ids
            or params.get("start_date") != date(year, number, 1).isoformat()
            or params.get("end_date")
            != date(year, number, calendar.monthrange(year, number)[1]).isoformat()
        ):
            raise OfflineBundleError(
                "Zones order/hash or requested monthly period differs from manifests"
            )
    if windows != set(MONTHS):
        raise OfflineBundleError("All three monthly manifests required")
    return ids, digest


def _load_frames(bundle, manifest, zone_ids, scratch):
    """Read every declared monthly table, checking dates and finite numeric features."""
    import duckdb
    import numpy as np
    import pandas as pd

    objects = manifest["objects"]
    for obj in objects:
        if set(obj) != {"key", "role", "bytes", "sha256", "readback"}:
            raise OfflineBundleError("Unexpected or missing object fields")
    sources = {obj["key"]: obj for obj in objects if obj["role"] == "o3_input_unified"}
    expected = {
        f"climate/holdout/unified/{month}/{zone}.parquet" for month in MONTHS for zone in zone_ids
    }
    if set(sources) != expected:
        raise OfflineBundleError("Expected exactly 45 monthly tables for the authenticated zones")
    frames, inventory = {}, []
    with duckdb.connect(":memory:", config=DUCKDB_CONFIG) as conn:
        for zone in zone_ids:
            parts = []
            for month in MONTHS:
                key = f"climate/holdout/unified/{month}/{zone}.parquet"
                obj = sources[key]
                local = scratch / "input.parquet"
                local.write_bytes(bundle.read(f"objects/{key}"))
                frame = conn.execute("SELECT * FROM read_parquet(?)", [str(local)]).df()
                readback = obj["readback"]
                if not isinstance(readback, dict) or set(readback) != {
                    "rows",
                    "min_date",
                    "max_date",
                }:
                    raise OfflineBundleError(f"Invalid table readback: {key}")
                if type(readback["rows"]) is not int or len(frame) != readback["rows"]:
                    raise OfflineBundleError(f"Table row count drift: {key}")
                if frame.empty:
                    if readback["min_date"] is not None or readback["max_date"] is not None:
                        raise OfflineBundleError(f"Empty table date drift: {key}")
                    inventory.append({**obj, "actual_rows": 0})
                    continue  # Historical zero-row Parquets have no columns.
                if not {"date", *FEATURES} <= set(frame.columns):
                    raise OfflineBundleError(f"Required columns missing: {key}")
                frame = frame[["date", *FEATURES]].copy()
                dates = pd.to_datetime(frame["date"], errors="raise")
                if (
                    dates.isna().any()
                    or dates.dt.tz is not None
                    or not dates.eq(dates.dt.normalize()).all()
                ):
                    raise OfflineBundleError(f"Invalid daily dates: {key}")
                if not dates.dt.strftime("%Y_%m").eq(month).all():
                    raise OfflineBundleError(f"Date outside declared month/Q3: {key}")
                actual_min, actual_max = (
                    dates.min().date().isoformat(),
                    dates.max().date().isoformat(),
                )
                if (actual_min, actual_max) != (readback["min_date"], readback["max_date"]):
                    raise OfflineBundleError(f"Table date readback drift: {key}")
                for column in FEATURES:
                    if not pd.api.types.is_numeric_dtype(
                        frame[column]
                    ) or pd.api.types.is_bool_dtype(frame[column]):
                        raise OfflineBundleError(f"Non-numeric feature {column}: {key}")
                if not np.isfinite(frame[list(FEATURES)].to_numpy(dtype=float)).all():
                    raise OfflineBundleError(f"Null or nonfinite feature: {key}")
                frame["date"] = dates
                parts.append(frame)
                inventory.append({**obj, "actual_rows": len(frame)})
            if not parts:
                raise OfflineBundleError(f"No Q3 observations: {zone}")
            frame = (
                pd.concat(parts, ignore_index=True)
                .sort_values("date", kind="mergesort")
                .reset_index(drop=True)
            )
            if frame["date"].duplicated().any():
                raise OfflineBundleError(f"Duplicate observation dates: {zone}")
            if frame["date"].min().date() < START or frame["date"].max().date() > END:
                raise OfflineBundleError(f"Observation outside requested Q3 period: {zone}")
            if (zone == "kerguelen" and len(frame) != 1) or (
                zone != "kerguelen" and len(frame) < 2
            ):
                raise OfflineBundleError(f"Historical eligibility changed: {zone}")
            frames[zone] = frame
    return frames, inventory


def _mine_zone(frame, algorithm, *, capture_raw_output=None):
    """Sequential caller-owned cwd confinement around the public adapter."""
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="geoai-offline-mine-") as folder:
        previous = Path.cwd()
        try:
            os.chdir(folder)
            if algorithm == "graank":
                result = mine_gradual_patterns(
                    frame[list(FEATURES)],
                    min_support=0.5,
                    encode_equal=False,
                    capture_raw_output=capture_raw_output,
                )
                target_idx = None
            else:
                temporal = frame[["date", *FEATURES]].copy()
                temporal["date"] = temporal["date"].dt.strftime("%Y-%m-%d")
                target_idx = list(temporal.columns).index("ndvi_mean")
                result = mine_temporal_patterns(
                    temporal,
                    target_col=target_idx,
                    min_support=0.5,
                    min_rep=0.5,
                    encode_equal=False,
                    parallel=False,
                    capture_raw_output=capture_raw_output,
                )
        finally:
            os.chdir(previous)
    return result, {
        "rows": len(frame),
        "patterns": len(result["patterns"]),
        "runtime_seconds": time.perf_counter() - started,
        "target_col": target_idx,
        "columns": list(FEATURES) if algorithm == "graank" else ["date", *FEATURES],
    }


def _cv_input_without_native_metrics(results):
    clean = copy.deepcopy(results)
    for result in clean.values():
        for pattern in result.get("patterns", []):
            pattern.pop("metrics", None)
            pattern.pop("extra_metrics", None)
            pattern.pop("confidence", None)
            pattern.pop("conf", None)
    return clean


def _git_state():
    def git(*args):
        return subprocess.run(
            ["git", "-C", str(ROOT), *args], check=True, capture_output=True, text=True
        ).stdout.strip()

    try:
        return {"head": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain"))}
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        raise OfflineBundleError(
            "Replay requires Git and a Git checkout with a HEAD commit; "
            "install Git and run from a cloned repository, not a source archive."
        ) from exc


def _environment():
    versions = {}
    for name in ("so4gp", "torch", "numpy", "pandas", "duckdb", "pydantic", "structlog"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    if versions["so4gp"] != "1.0.9":
        raise OfflineBundleError(
            "This provisional CPU replay requires so4gp==1.0.9; "
            f"found {versions['so4gp'] or 'not installed'}. "
            "Use the documented isolated replay environment before mining."
        )
    torch_state = None
    if versions["torch"] is not None:
        import torch

        torch.set_num_threads(1)
        torch_state = {"cuda_build": torch.version.cuda, "num_threads": torch.get_num_threads()}
    # No GPU initialization or automatic device selection.
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "packages": versions,
        "device": "cpu",
        "torch": torch_state,
        "python_hash_seed": os.environ.get("PYTHONHASHSEED", "random"),
    }


def run(args):
    import duckdb

    archive = args.archive.resolve()
    if not archive.is_file() or archive.stat().st_size > MAX_ARCHIVE_BYTES:
        raise OfflineBundleError("Archive absent or too large")
    output = args.output_dir.absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Fresh output directory required: {output}")
    git_state = _git_state()  # Snapshot source provenance before creating any output.
    with tempfile.TemporaryDirectory(prefix="geoai-offline-input-") as folder:
        scratch = Path(folder)
        snapshot = scratch / "handoff.zip"
        with archive.open("rb") as source:
            data = source.read(MAX_ARCHIVE_BYTES + 1)
        if len(data) > MAX_ARCHIVE_BYTES:
            raise OfflineBundleError("Archive too large")
        snapshot.write_bytes(data)
        manifest, _, input_manifests, archive_bytes = _verify_bundle(snapshot, args.expected_sha256)
        zone_ids, zones_sha = _zone_order(args.zones_geojson, input_manifests)
        with zipfile.ZipFile(snapshot) as bundle:
            frames, inventory = _load_frames(bundle, manifest, zone_ids, scratch)
        eligible = [zone for zone in zone_ids if zone != "kerguelen"]
        output.mkdir(mode=0o700)  # Exclusive allocation, including a concurrent creator.
        evidence = {
            "schema_version": "geoai-q3-offline-replay/v1",
            "status": "running",
            "provisional_replay": True,
            "git": git_state,
        }
        try:
            _write_json(output / "manifest.json", evidence)
            generated = datetime.now(UTC)
            evidence.update(
                {
                    "archive_sha256": args.expected_sha256.lower(),
                    "archive_bytes": archive_bytes,
                    "zones_sha256": zones_sha,
                    "zone_ids": zone_ids,
                    "eligible_zone_ids": eligible,
                    "excluded_zones": {"kerguelen": {"rows": 1, "reason": "insufficient Q3 rows"}},
                    "truncated_sources": {"svalbard": ["2023_07"]},
                    "parameters": {
                        "min_support": 0.5,
                        "min_rep": 0.5,
                        "n_folds": 5,
                        "min_validated_zones": 1,
                        "encode_equal": False,
                        "period_start": START.isoformat(),
                        "period_end": END.isoformat(),
                        "features": list(FEATURES),
                        "months": list(MONTHS),
                        "graank_target": None,
                        "temporal_target": "ndvi_mean",
                        "temporal_transformations": "all",
                        "search": "apriori",
                    },
                    "environment": _environment(),
                    "generated_at": generated.isoformat(),
                    "inputs": inventory,
                    "source_bundle": manifest,
                    "caveats": CAVEATS + manifest["caveats"],
                }
            )
            _write_json(output / "manifest.json", evidence)
            stats, counts = {}, {}
            for algorithm in ("graank", "tgrad"):
                mined, counts[algorithm] = {}, {}
                for zone in eligible:
                    destination = output / "zones" / zone
                    destination.mkdir(parents=True, exist_ok=True)
                    raw_path = destination / f"{algorithm}.raw.json"
                    result, details = _mine_zone(
                        frames[zone],
                        algorithm,
                        capture_raw_output=lambda raw, path=raw_path: path.write_text(
                            raw, encoding="utf-8"
                        ),
                    )
                    mined[zone], counts[algorithm][zone] = result, details
                    _write_json(
                        destination / f"{algorithm}.normalized.json", {"result": result, **details}
                    )
                library, folds, stats[algorithm] = cross_validate_mined_patterns(
                    _cv_input_without_native_metrics(mined),
                    zone_ids=eligible,
                    algorithm=algorithm,
                    min_support=0.5,
                    n_folds=5,
                    period_start=START,
                    period_end=END,
                    generated_at=generated,
                )
                library = PatternLibrary.model_validate(library.model_dump())
                if any(pattern.confidence is not None for pattern in library.patterns):
                    raise ValueError("Calibrated confidence must remain null")
                _write_json(output / f"{algorithm}_validated.json", library.model_dump(mode="json"))
                table = validated_library_to_dataframe(library)
                if list(table.columns) != list(VALIDATED_PARQUET_TYPES):
                    raise ValueError(
                        "Validated pattern columns differ from the flat exchange schema"
                    )
                typed_columns = ", ".join(
                    f'CAST("{column}" AS {dtype}) AS "{column}"'
                    for column, dtype in VALIDATED_PARQUET_TYPES.items()
                )
                with duckdb.connect(":memory:", config=DUCKDB_CONFIG) as conn:
                    conn.register("validated", table)
                    conn.execute(
                        f"COPY (SELECT {typed_columns} FROM validated) TO ? (FORMAT PARQUET)",
                        [str(output / f"{algorithm}_validated.parquet")],
                    )
                _write_json(
                    output / "folds.json",
                    {
                        "status": "reconstructed_round_robin",
                        "historical_assignment_available": False,
                        "zone_order": eligible,
                        "folds": folds,
                    },
                )
            summary = {
                "status": "running",
                "statistics": stats,
                "per_zone": counts,
                "calibrated_confidence": None,
                "caveats": evidence["caveats"],
            }
            _write_json(output / "summary.json", summary)
            evidence["output_hashes"] = {
                str(path.relative_to(output)): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in sorted(output.rglob("*"))
                if path.is_file() and path.name != "manifest.json"
            }
            # Publish completion only after mining and artifact bookkeeping succeeded.
            summary["status"] = "provisional_replay_complete"
            _write_json(output / "summary.json", summary)
            evidence["output_hashes"]["summary.json"] = hashlib.sha256(
                (output / "summary.json").read_bytes()
            ).hexdigest()
            evidence["status"] = "provisional_replay_complete"
            _write_json(output / "manifest.json", evidence)
        except (Exception, KeyboardInterrupt) as exc:
            evidence["status"] = "failed"
            evidence["error"] = {"type": type(exc).__name__, "message": str(exc)}
            evidence.pop("output_hashes", None)
            # Failure reporting is best effort; preserve the original exception.
            try:
                if (output / "summary.json").exists():
                    _write_json(
                        output / "summary.json", {"status": "failed", "error": evidence["error"]}
                    )
            except Exception as report_error:
                exc.add_note(f"Could not mark summary failed: {report_error}")
            try:
                _write_json(output / "manifest.json", evidence)
            except Exception as report_error:
                exc.add_note(f"Could not write failure manifest: {report_error}")
            raise
    return output


def main(argv=None):
    from src.utils.logging import setup_logging

    setup_logging()
    args = build_parser().parse_args(argv)
    try:
        print(run(args))
    except (OfflineBundleError, ValueError, OSError, zipfile.BadZipFile) as exc:
        print(f"Offline replay failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
