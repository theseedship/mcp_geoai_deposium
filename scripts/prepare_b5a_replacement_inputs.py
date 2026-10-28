#!/usr/bin/env python3
"""Prepare point-skill inputs from already downloaded pinned releases, entirely offline.

Frozen panels and sampling points are verified byte-for-byte and never rewritten.
Choose a fresh output directory: reference.jsonl, p50.jsonl and the portable
preparation-receipt.json are create-only. No forecasts are collected.
If publication is interrupted, partial artifacts are retained and status is incomplete.
Retry with a new output directory; never overwrite or clear an existing directory.
"""

import argparse
import hashlib
import io
import json
import math
import struct
import sys
import tempfile
from datetime import date, timedelta
from pathlib import Path, PurePosixPath
from zipfile import ZipFile

import duckdb

RELEASE_SPECS = [
    (
        "holdout",
        "geoai-era5-holdout-2021-2025-handoff.zip",
        "era5-holdout-handoff-2026-09-19",
        "3a03059e6a0f75e77fb355b68e2535b0e841e2bb460ac0d2574a0104c7f5937d",
    ),
    (
        "wmo",
        "geoai-canonical-full-panel-wmo-1991-2020-20260624.zip",
        "climate-baseline-full-panel-2026-06-26",
        "44f6831db4029e4df9769604092ed52e63316ae865dd94b9c836c76aed9fa7a8",
    ),
]


def main():
    parser = argparse.ArgumentParser(
        description="Offline preparation from pinned published ZIPs; no downloads or provider calls."
    )
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--panel-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "tests/fixtures/b5a-replacement-20261005",
    )
    args = parser.parse_args()
    args.repo = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(args.repo))
    from src.forecast_backtest import parse_rows
    from src.forecast_skill import MonthlyP50Baseline, PanelManifest
    from src.utils.logging import setup_logging

    setup_logging()

    source_dir = args.source_dir
    output_dir = args.output_dir
    require_new_output = not output_dir.exists()
    if not require_new_output:
        raise ValueError("Output directory already exists; choose a fresh directory")
    panel_dir = args.panel_dir
    frozen_spatial = json.loads((panel_dir / "sampling-points.json").read_bytes())

    def sha(b):
        return hashlib.sha256(b).hexdigest()

    def encode(x):
        return (json.dumps(x, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()

    def publish(path, b):
        if path.parent == panel_dir:
            if not path.read_bytes() == b:
                raise ValueError(f"Frozen candidate bytes differ: {path.name}")
        else:
            with path.open("xb") as f:
                f.write(b)
        return {"file": path.name, "sha256": sha(b), "bytes": len(b)}

    archives = {}
    releases = {}
    for kind, name, tag, digest in RELEASE_SPECS:
        b = (source_dir / name).read_bytes()
        if not sha(b) == digest:
            raise ValueError("Input validation failed: sha(b) == digest")
        archives[kind] = ZipFile(io.BytesIO(b))
        releases[kind] = {
            "release_url": f"https://github.com/theseedship/deposium_geoai/releases/tag/{tag}",
            "asset": name,
            "sha256": digest,
            "bytes": len(b),
        }
    manifest = json.loads(archives["holdout"].read("bundle-manifest.json"))
    zones = sorted(manifest["expected"]["zones"])
    if not len(zones) == 15:
        raise ValueError("Input validation failed: len(zones) == 15")
    variables = [
        "temperature_2m",
        "precipitation",
        "humidity",
        "evapotranspiration",
        "wind_speed",
        "solar_radiation",
    ]
    dates = [date(2025, 2, 25) + timedelta(days=i) for i in range(239)]
    if not dates[-1] == date(2025, 10, 21):
        raise ValueError("Input validation failed: dates[-1] == date(2025, 10, 21)")
    connection = duckdb.connect(":memory:")
    objects = []
    points = []
    references = []
    baselines = []
    temporary = tempfile.TemporaryDirectory(prefix="b5a-offline-")
    extract = Path(temporary.name)

    def read_parquet_object(kind, member, zone, role):
        if member not in archives[kind].namelist():
            raise ValueError("Input invariant failed")
        name = PurePosixPath(member)
        if not (not name.is_absolute() and ".." not in name.parts):
            raise ValueError(
                "Input validation failed: not name.is_absolute() and '..' not in name.parts"
            )
        b = archives[kind].read(member)
        if kind == "holdout":
            declared = next(o for o in manifest["objects"] if o["bundle_path"] == member)
            if not (sha(b) == declared["sha256"] and len(b) == declared["bytes"]):
                raise ValueError(
                    "Input validation failed: sha(b) == declared['sha256'] and len(b) == declared['bytes']"
                )
        else:
            declared = {}
        path = extract / (kind + "-" + role + "-" + zone + ".parquet")
        publish(path, b)
        columns = [
            r[0]
            for r in connection.execute(
                "describe select * from read_parquet(?)", [str(path)]
            ).fetchall()
        ]
        rows = [
            dict(zip(columns, r, strict=True))
            for r in connection.execute("select * from read_parquet(?)", [str(path)]).fetchall()
        ]
        if not {r["zone_id"] for r in rows} == {zone}:
            raise ValueError("Input validation failed: {r['zone_id'] for r in rows} == {zone}")
        spatial = {(r["lat"], r["lon"], bytes(r["geometry"])) for r in rows}
        if not len(spatial) == 1:
            raise ValueError("Input validation failed: len(spatial) == 1")
        lat, lon, wkb = next(iter(spatial))
        if not (math.isfinite(lat) and math.isfinite(lon)):
            raise ValueError("Input validation failed: math.isfinite(lat) and math.isfinite(lon)")
        if not (len(wkb) == 21 and wkb[0] == 1 and (struct.unpack("<I", wkb[1:5])[0] == 1)):
            raise ValueError(
                "Input validation failed: len(wkb) == 21 and wkb[0] == 1 and (struct.unpack('<I', wkb[1:5])[0] == 1)"
            )
        if not struct.unpack("<dd", wkb[5:]) == (lon, lat):
            raise ValueError(f"geometry disagrees {kind}/{zone}")
        rec = {
            "release": kind,
            "member": member,
            "logical_s3_key": declared.get("logical_s3_key", member),
            "sha256": sha(b),
            "bytes": len(b),
            "rows": len(rows),
        }
        if declared:
            rec["source_s3_last_modified"] = declared["source_s3_last_modified"]
        objects.append(rec)
        return (rows, (lat, lon, wkb), rec)

    for zone in zones:
        hold, hpoint, hrec = read_parquet_object(
            "holdout", f"objects/climate/holdout/era5/2021_2025/{zone}.parquet", zone, "holdout"
        )
        wmo, wpoint, wrec = read_parquet_object("wmo", f"climate/era5/{zone}.parquet", zone, "era5")
        norms, npoint, nrec = read_parquet_object(
            "wmo", f"climate/normals/{zone}.parquet", zone, "normals"
        )
        if not hpoint == wpoint == npoint:
            raise ValueError(f"STOP: sampling disagreement for {zone}")
        if not (len(hold) == 1826 and len({r["date"] for r in hold}) == 1826):
            raise ValueError(
                "Input validation failed: len(hold) == 1826 and len({r['date'] for r in hold}) == 1826"
            )
        if not (
            min(r["date"] for r in hold) == date(2021, 1, 1)
            and max(r["date"] for r in hold) == date(2025, 12, 31)
        ):
            raise ValueError(
                "Input validation failed: min((r['date'] for r in hold)) == date(2021, 1, 1) and max((r['date'] for r in hold)) == date(2025, 12, 31)"
            )
        if not (len(wmo) == 10958 and len({r["date"] for r in wmo}) == 10958):
            raise ValueError(
                "Input validation failed: len(wmo) == 10958 and len({r['date'] for r in wmo}) == 10958"
            )
        if not (
            min(r["date"] for r in wmo) == date(1991, 1, 1)
            and max(r["date"] for r in wmo) == date(2020, 12, 31)
        ):
            raise ValueError(
                "Input validation failed: min((r['date'] for r in wmo)) == date(1991, 1, 1) and max((r['date'] for r in wmo)) == date(2020, 12, 31)"
            )
        if not (
            len(norms) == 72
            and {(r["variable"], r["month"]) for r in norms}
            == {(v, m) for v in variables for m in range(1, 13)}
        ):
            raise ValueError(
                "Input validation failed: len(norms) == 72 and {(r['variable'], r['month']) for r in norms} == {(v, m) for v in variables for m in range(1, 13)}"
            )
        if not {(r["window_start"], r["window_end"], r["window_label"]) for r in norms} == {
            ("1991-01-01", "2020-12-31", "WMO_1991_2020")
        }:
            raise ValueError(
                "Input validation failed: {(r['window_start'], r['window_end'], r['window_label']) for r in norms} == {('1991-01-01', '2020-12-31', 'WMO_1991_2020')}"
            )
        for row in hold:
            for var in variables:
                if not (row[var] is not None and math.isfinite(row[var])):
                    raise ValueError(
                        "Input validation failed: row[var] is not None and math.isfinite(row[var])"
                    )
            if row["date"] in dates:
                for var in variables:
                    references.append(
                        {
                            "zone_id": zone,
                            "valid_date": row["date"].isoformat(),
                            "variable": var,
                            "value": row[var],
                        }
                    )
        for row in norms:
            if not (row["p50"] is not None and math.isfinite(row["p50"])):
                raise ValueError(
                    "Input validation failed: row['p50'] is not None and math.isfinite(row['p50'])"
                )
            baselines.append(
                {
                    "zone_id": zone,
                    "variable": row["variable"],
                    "month": row["month"],
                    "p50": row["p50"],
                }
            )
        points.append(
            {
                "zone_id": zone,
                "latitude": hpoint[0],
                "longitude": hpoint[1],
                "geometry_wkb_hex": hpoint[2].hex(),
                "agreement": "exact_float_and_wkb_across_holdout_wmo_era5_wmo_normals",
                "sources": [hrec, wrec, nrec],
            }
        )
    references.sort(key=lambda r: (r["zone_id"], r["valid_date"], r["variable"]))
    baselines.sort(key=lambda r: (r["zone_id"], r["variable"], r["month"]))
    if not (len(references) == 21510 and len(baselines) == 1080):
        raise ValueError(
            "Input validation failed: len(references) == 21510 and len(baselines) == 1080"
        )
    parse_rows(references, kind="reference")
    MonthlyP50Baseline.from_records(baselines)
    if not len({(r["zone_id"], r["valid_date"], r["variable"]) for r in references}) == len(
        references
    ):
        raise ValueError(
            "Input validation failed: len({(r['zone_id'], r['valid_date'], r['variable']) for r in references}) == len(references)"
        )
    if not len({(r["zone_id"], r["variable"], r["month"]) for r in baselines}) == len(baselines):
        raise ValueError(
            "Input validation failed: len({(r['zone_id'], r['variable'], r['month']) for r in baselines}) == len(baselines)"
        )
    mismatches = frozen_spatial["current_geometry_midpoint_discrepancies"]
    writer_runs = [
        {
            "run_id": "holdout-pilot-mesh-2021-2025-proof-era5",
            "zones": ["bassin_du_lez", "fos_sur_mer", "montpellier_metropole"],
            "proof": "holdout-pilot-mesh-rustfs-2026-06-27.md",
        },
        {
            "run_id": "holdout-representative-mesh-2023-q3-widening-proof-era5",
            "zones": ["mombasa", "nairobi", "niamey", "rennes", "riyadh"],
            "proof": "holdout-b4-representative-q3-mesh-rustfs-2026-06-27.md",
        },
        {
            "run_id": "holdout-full-panel-mesh-2023-q3-segmented-proof-era5",
            "zones": [
                "kerguelen",
                "manaus",
                "marseille",
                "montpellier",
                "seoul",
                "svalbard",
                "tromso",
            ],
            "proof": "holdout-b4-full-panel-q3-segmented-mesh-rustfs-2026-06-27.md",
        },
    ]
    for r in writer_runs:
        r["logical_manifest_key"] = f"climate/runs/{r['run_id']}/manifest.json"
        r["proof_url"] = (
            "https://github.com/theseedship/deposium_geoai/blob/b9028f0/docs/demo/" + r.pop("proof")
        )
    spatial = {
        "schema_version": "b5a-replacement-sampling-points/v1",
        "status": "candidate_not_agreed_no_forecasts_collected",
        "source_releases": releases,
        "holdout_writer_runs": writer_runs,
        "holdout_bundle_manifest_sha256": sha(archives["holdout"].read("bundle-manifest.json")),
        "wmo_run_manifest_sha256": sha(archives["wmo"].read("manifest.json")),
        "wmo_bundle_manifest_sha256": sha(archives["wmo"].read("bundle.json")),
        "sampling_points": points,
        "source_discrepancies": [],
        "current_geometry_midpoint_discrepancies": mismatches,
        "notes": [
            "Sampling points are extracted from the published source Parquets, never inferred from current boundaries.",
            "These stored lat/lon points identify requested sampling coordinates, not model grid-cell coordinates.",
            "Replacement retrieval must use these exact coordinates; current three French pilot bbox midpoints differ.",
        ],
    }
    artifacts = {}
    artifacts["sampling_points"] = publish(panel_dir / "sampling-points.json", encode(spatial))
    configuration = {
        "panel_id": "b5a-replacement-20261005-v1",
        "source": "openmeteo_previous_runs",
        "day_grain": "utc_day",
        "timezone": "UTC",
        "lead_semantics": "fixed_hourly_offset_24_times_horizon_days",
        "wind_speed_unit": "ms",
        "sampling_points_sha256": artifacts["sampling_points"]["sha256"],
    }
    for provider, model in [("gfs", "gfs_seamless"), ("aifs", "ecmwf_aifs025_single")]:
        panel = {
            "schema_version": "forecast-skill-panel/v1",
            "provider": provider,
            "model": model,
            "configuration": configuration,
            "zones": zones,
            "variables": variables,
            "horizons_days": [1, 3, 5, 7],
            "valid_dates": [d.isoformat() for d in dates],
        }
        parsed = PanelManifest.from_record(panel)
        if not parsed.expected_pairs == 86040:
            raise ValueError("Input validation failed: parsed.expected_pairs == 86040")
        parsed.require_within(
            training_end=date(2020, 12, 31),
            holdout_start=date(2021, 1, 1),
            holdout_end=date(2025, 12, 31),
        )
        parsed.require_identity(provider=provider, model=model, configuration=configuration)
        artifacts["panel_" + provider] = publish(
            panel_dir / (provider + "-panel.json"), encode(panel)
        )

    def jsonl(rows):
        return "".join(
            json.dumps(r, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
            for r in rows
        ).encode()

    output_dir.mkdir(parents=True, exist_ok=False)
    try:
        artifacts["reference"] = publish(output_dir / "reference.jsonl", jsonl(references))
        artifacts["baseline_p50"] = publish(output_dir / "p50.jsonl", jsonl(baselines))
        summary = {
            "schema_version": "b5a-replacement-offline-preparation/v1",
            "status": "prepared_candidate_not_agreed_forecast_rows_absent",
            "source_documentation_commit": "b9028f05c7d02d97d43d5eed8e3d273a4a185e37",
            "source_releases": releases,
            "source_objects": objects,
            "holdout_writer_runs": writer_runs,
            "artifacts": artifacts,
            "counts": {
                "zones": 15,
                "variables": 6,
                "valid_dates": 239,
                "horizons": 4,
                "expected_pairs_per_provider": 86040,
                "reference_rows": 21510,
                "baseline_rows": 1080,
                "source_point_discrepancies": 0,
                "current_geometry_midpoint_discrepancies": len(mismatches),
            },
            "training_end": "2020-12-31",
            "holdout_start": "2021-01-01",
            "holdout_end": "2025-12-31",
            "panel_first_valid_date": "2025-02-25",
            "panel_last_valid_date": "2025-10-21",
            "configuration": configuration,
            "checks": {
                "source_zip_digests": True,
                "holdout_object_digests": True,
                "spatial_exact_cross_source_match": True,
                "geometry_point_matches_lat_lon": True,
                "reference_dates_finite_unique": True,
                "all_day_monthly_p50_finite_unique": True,
                "panel_contract_and_leakage": True,
                "forecast_rows_collected": False,
            },
            "preparation_script": {
                "file": Path(__file__).name,
                "sha256": sha(Path(__file__).read_bytes()),
            },
        }
        publish(output_dir / "preparation-receipt.json", encode(summary))
    except OSError as exc:
        temporary.cleanup()
        connection.close()
        print(
            json.dumps(
                {
                    "status": "incomplete",
                    "output_directory": str(output_dir),
                    "error": str(exc),
                    "recovery": "Partial artifacts retained; retry with a new --output-dir path.",
                },
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 1
    temporary.cleanup()
    connection.close()
    print(
        json.dumps(
            {"status": summary["status"], "artifacts": artifacts, "counts": summary["counts"]},
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, KeyError, TypeError, duckdb.Error) as exc:
        print(json.dumps({"status": "refused", "error": str(exc)}), file=sys.stderr)
        raise SystemExit(1) from None
