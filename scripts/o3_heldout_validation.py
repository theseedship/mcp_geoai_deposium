#!/usr/bin/env python3
"""Offline held-out research experiment, separate from recurrence and #19.

Reads authenticated Q3 artifacts and optional local three-pilot B4 tables.
No mining/provider/MCP calls or object-store writes are performed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import o3_q3_offline_replay as replay  # noqa: E402
from src.patterns.heldout_validation import (  # noqa: E402
    CLIMATE,
    TARGET,
    candidate_universe,
    geometry_folds,
    spatial_validation,
    temporal_validation,
)
from src.patterns.offline_bundle import MAX_JSON_BYTES, _strict_json, _verify_bundle  # noqa: E402

PILOTS = ("bassin_du_lez", "fos_sur_mer", "montpellier_metropole")
Q3_MANIFEST_SHA256 = "e0164cf24138ae5396064b0665ed35389577c2f538cfe588e9e502fa84297c8c"


def _digest(data):
    return hashlib.sha256(data).hexdigest()


def snapshot_q3(directory, expected_manifest_sha256, destination):
    """Authenticate the manifest, then reuse the existing strict ZIP verifier.

    The archive is a temporary byte snapshot of the checked local handoff. Its
    generated ZIP hash is not represented as the published archive's hash.
    """
    directory = Path(directory).resolve()
    raw = (directory / "bundle.json").read_bytes()
    if len(raw) > MAX_JSON_BYTES or _digest(raw) != expected_manifest_sha256:
        raise ValueError("Q3 manifest SHA-256 mismatch")
    manifest = _strict_json(raw)
    names = [
        "bundle.json",
        "README-O3-HANDOFF.md",
        *("objects/" + o["key"] for o in manifest["objects"]),
    ]
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        for name in names:
            relative = PurePosixPath(name)
            if "\\" in name or ":" in name or relative.is_absolute() or ".." in relative.parts:
                raise ValueError("Invalid local Q3 member path")
            path = (directory / name).resolve()
            if not path.is_relative_to(directory):
                raise ValueError("Local Q3 member outside handoff")
            bundle.writestr(name, raw if name == "bundle.json" else path.read_bytes())
    digest = _digest(Path(destination).read_bytes())
    verified, _, monthly, _ = _verify_bundle(Path(destination), digest)
    return verified, monthly


def load_pilots(template):
    import duckdb
    import numpy as np
    import pandas as pd

    if "://" in template or "{zone_id}" not in template:
        raise ValueError("Pilot template must be a local path containing {zone_id}")
    frames, inventory = {}, []
    with (
        tempfile.TemporaryDirectory(prefix="geoai-pilots-") as folder,
        duckdb.connect(":memory:", config=replay.DUCKDB_CONFIG) as conn,
    ):
        for zone in PILOTS:
            source = Path(template.format(zone_id=zone)).resolve()
            data = source.read_bytes()
            snapshot = Path(folder) / "input.parquet"
            snapshot.write_bytes(data)
            frame = conn.execute("SELECT * FROM read_parquet(?)", [str(snapshot)]).df()
            if not {"date", TARGET, *CLIMATE} <= set(frame):
                raise ValueError(f"Missing pilot features: {zone}")
            frame = frame[["date", TARGET, *CLIMATE]].copy()
            dates = pd.to_datetime(frame["date"], errors="raise")
            if (
                dates.isna().any()
                or dates.dt.tz is not None
                or dates.duplicated().any()
                or not dates.eq(dates.dt.normalize()).all()
            ):
                raise ValueError(f"Invalid daily pilot dates: {zone}")
            for column in (TARGET, *CLIMATE):
                if not pd.api.types.is_numeric_dtype(frame[column]) or pd.api.types.is_bool_dtype(
                    frame[column]
                ):
                    raise ValueError(f"Non-numeric pilot feature: {zone}/{column}")
            if not np.isfinite(frame[[TARGET, *CLIMATE]].to_numpy(dtype=float)).all():
                raise ValueError(f"Nonfinite pilot features: {zone}")
            frame["date"] = dates
            frames[zone] = frame.sort_values("date", kind="stable").reset_index(drop=True)
            inventory.append(
                {
                    "zone_id": zone,
                    "path": str(source),
                    "bytes": len(data),
                    "sha256": _digest(data),
                    "rows": len(frame),
                    "min_date": dates.min().date().isoformat() if len(frame) else None,
                    "max_date": dates.max().date().isoformat() if len(frame) else None,
                    "outside_experiment_rows": int(
                        (~dates.between("2021-01-01", "2024-12-31")).sum()
                    ),
                }
            )
    return frames, inventory


def verify_pilot_provenance(provenance, inventory):
    if (
        not isinstance(provenance, dict)
        or provenance.get("schema_version") != "geoai-heldout-pilot-inputs/v1"
        or provenance.get("status") != "complete"
    ):
        raise ValueError("Invalid pilot provenance schema or status")
    outputs = provenance.get("outputs")
    if (
        not isinstance(outputs, list)
        or len(outputs) != len(PILOTS)
        or not all(isinstance(row, dict) for row in outputs)
    ):
        raise ValueError("Invalid pilot provenance output inventory")
    by_zone = {row.get("zone_id"): row for row in outputs}
    if set(by_zone) != set(PILOTS):
        raise ValueError("Invalid pilot provenance zone inventory")
    for row in inventory:
        declared = by_zone[row["zone_id"]]
        if declared.get("sha256") != row["sha256"] or declared.get("rows") != row["rows"]:
            raise ValueError(f"Pilot provenance hash/row mismatch: {row['zone_id']}")


def _write_json(path, data):
    path.write_text(
        json.dumps(data, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def write_outputs(output, results, provenance):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    manifest = {"schema_version": "geoai-heldout-research/v1", **provenance, "status": "running"}
    _write_json(output / "manifest.json", manifest)
    try:
        for scope, result in sorted(results.items()):
            folder = output / scope
            folder.mkdir()
            _write_json(folder / "result.json", result)
            for name in (
                "candidates",
                "donor_scores",
                "fold_scores",
                "zone_scores",
                "koppen_scores",
                "scores",
                "coverage",
                "folds",
            ):
                if name not in result:
                    continue
                rows = result[name]
                _write_json(folder / f"{name}.json", rows)
                fields = (
                    ["rule_id", *sorted({k for row in rows for k in row} - {"rule_id"})]
                    if rows
                    else [
                        "rule_id",
                        "status",
                        "pairs",
                        "antecedent",
                        "joint",
                        "support",
                        "confidence",
                    ]
                )
                with (folder / f"{name}.csv").open("w", encoding="utf-8", newline="") as file:
                    writer = csv.DictWriter(file, fieldnames=fields, lineterminator="\n")
                    writer.writeheader()
                    for row in rows:
                        writer.writerow(
                            {
                                k: json.dumps(v, sort_keys=True)
                                if isinstance(v, (list, dict))
                                else v
                                for k, v in row.items()
                            }
                        )
        manifest["output_hashes"] = {
            p.relative_to(output).as_posix(): _digest(p.read_bytes())
            for p in sorted(output.rglob("*"))
            if p.is_file() and p != output / "manifest.json"
        }
        manifest["status"] = "complete"
        _write_json(output / "manifest.json", manifest)
    except BaseException as exc:
        manifest.update(status="failed", error={"type": type(exc).__name__, "message": str(exc)})
        manifest.pop("output_hashes", None)
        _write_json(output / "manifest.json", manifest)
        raise
    return manifest


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--q3-dir", type=Path, default=ROOT / "examples/so4gp_parquet/geoai-o3-q3-2023-handoff"
    )
    parser.add_argument("--expected-bundle-sha256", default=Q3_MANIFEST_SHA256)
    parser.add_argument(
        "--q3-zones-geojson",
        type=Path,
        required=True,
        help="Original geometry matching Q3 monthly manifests",
    )
    parser.add_argument("--pilot-template", help="Local B4 2021–2024 Parquets containing {zone_id}")
    parser.add_argument(
        "--pilot-provenance",
        type=Path,
        help="Required with pilots: source/alignment/geometry provenance JSON",
    )
    parser.add_argument("--lag-steps", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--min-support", type=float, default=0.5)
    parser.add_argument("--min-confidence", type=float, default=0.5)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def run(args):
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if bool(args.pilot_template) != bool(args.pilot_provenance):
        raise ValueError("Pilot template and provenance must be supplied together")
    # Snapshot all authenticated inputs before allocating an output directory.
    with tempfile.TemporaryDirectory(prefix="geoai-heldout-") as folder:
        scratch = Path(folder)
        archive = scratch / "q3.zip"
        manifest, monthly = snapshot_q3(args.q3_dir, args.expected_bundle_sha256, archive)
        geometry_bytes = args.q3_zones_geojson.read_bytes()
        geometry = scratch / "zones.geojson"
        geometry.write_bytes(geometry_bytes)
        ids, geometry_digest = replay._zone_order(geometry, monthly)
        geojson = _strict_json(geometry_bytes)
        koppen = {
            f["properties"]["zone_id"]: f["properties"]["koppen"] for f in geojson["features"]
        }
        with zipfile.ZipFile(archive) as bundle:
            frames, inventory = replay._load_frames(bundle, manifest, ids, scratch)
    eligible = {z: f for z, f in frames.items() if len(f) >= 2}
    folds = geometry_folds(geojson, list(eligible))
    thresholds = {"min_support": args.min_support, "min_confidence": args.min_confidence}
    results = {
        "spatial": spatial_validation(
            eligible, folds, candidate_universe(), koppen=koppen, **thresholds
        )
    }
    pilot_inputs, pilot_provenance = [], None
    if args.pilot_template:
        pilot_frames, pilot_inputs = load_pilots(args.pilot_template)
        pilot_bytes = args.pilot_provenance.read_bytes()
        pilot_provenance = {
            "path": str(args.pilot_provenance.resolve()),
            "sha256": _digest(pilot_bytes),
            "value": _strict_json(pilot_bytes),
        }
        verify_pilot_provenance(pilot_provenance["value"], pilot_inputs)
        results["temporal"] = temporal_validation(
            pilot_frames,
            candidate_universe(lags=args.lag_steps),
            koppen=dict.fromkeys(PILOTS, "Csa"),
            **thresholds,
        )
    git = subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
    provenance = {
        "code": {
            "commit": git,
            "branch": subprocess.run(
                ["git", "-C", str(ROOT), "branch", "--show-current"],
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip(),
            "dirty": bool(
                subprocess.run(
                    ["git", "-C", str(ROOT), "status", "--porcelain"],
                    capture_output=True,
                    text=True,
                    check=True,
                ).stdout.strip()
            ),
            "files": {
                str(p.relative_to(ROOT)).replace("\\", "/"): _digest(p.read_bytes())
                for p in [
                    Path(__file__),
                    ROOT / "src/patterns/heldout_validation.py",
                    ROOT / "scripts/o3_q3_offline_replay.py",
                    ROOT / "src/patterns/offline_bundle.py",
                ]
            },
        },
        "runtime": {
            "python": platform.python_version(),
            "packages": {
                n: importlib.metadata.version(n) for n in ("numpy", "pandas", "duckdb", "shapely")
            },
        },
        "parameters": {
            **thresholds,
            "passing_spatial_folds": 4,
            "n_folds": 5,
            "target": TARGET,
            "climate": list(CLIMATE),
            "temporal_lag_steps": sorted(set(args.lag_steps)),
            "preprocessing": "identity; strict finite features, no fitting or imputation",
            "candidate_selection": "pooled donor joint/pair support only",
            "pair_population": "within-zone unordered pairs with both strict orientations tested",
            "lag_policy": "observation-step shifts, split periods before transformation",
        },
        "q3": {
            "bundle_sha256": args.expected_bundle_sha256,
            "geometry_sha256": geometry_digest,
            "eligible_zone_ids": sorted(eligible),
            "excluded": [
                {"zone_id": z, "rows": len(f), "reason": "fewer than two observations"}
                for z, f in frames.items()
                if z not in eligible
            ],
            "inventory": inventory,
        },
        "pilot_inputs": pilot_inputs,
        "pilot_provenance": pilot_provenance,
        "limits": [
            "Research thresholds are proposed; no scientific acceptance inferred.",
            "#19 recurrence outputs are separate and unchanged.",
            "Svalbard July remains truncated; Kerguelen has one Q3 row.",
            "Köppen labels retained historically; Manaus, Nairobi and Niamey remain under review.",
            "Temporal transfer is restricted to three French Csa pilots; overlapping pilots are geographically dependent.",
            "Pilot cadence and geometry differ from Q3; do not pool experiments.",
            "Positive lags count observations, not calendar days; sparse intervals differ across zones and periods.",
        ],
        "temporal_status": "evaluated" if args.pilot_template else "not_requested",
    }
    return write_outputs(args.output_dir, results, provenance)


def main(argv=None):
    from src.utils.logging import setup_logging

    setup_logging()
    args = build_parser().parse_args(argv)
    try:
        manifest = run(args)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(f"status={manifest['status']} manifest={args.output_dir / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
