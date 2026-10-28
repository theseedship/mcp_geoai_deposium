#!/usr/bin/env python3
"""Materialize a local three-pilot panel from the two pinned release archives.

This portable offline preparation mirrors nearest-date alignment without
loading MCP tools or spatial extensions. Donor and evaluation periods are
aligned separately, so climate matches cannot cross 2022/2023.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import stat
import sys
import tempfile
import zipfile
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.o3_heldout_validation import PILOTS, _write_json  # noqa: E402
from src.patterns.heldout_validation import CLIMATE  # noqa: E402

ERA5_SHA256 = "3a03059e6a0f75e77fb355b68e2535b0e841e2bb460ac0d2574a0104c7f5937d"
B3_SHA256 = "0cd6dd981bb78916026fc1585cd856ba81e89deeaea24158973513bf6ad5526c"


def read_archive(path, expected_sha256):
    data = Path(path).read_bytes()
    if len(data) > 100 * 1024 * 1024 or hashlib.sha256(data).hexdigest() != expected_sha256:
        raise ValueError("Archive SHA-256 mismatch or size limit exceeded")
    result = {}
    with zipfile.ZipFile(io.BytesIO(data)) as bundle:
        entries = bundle.infolist()
        if len(entries) > 1000 or sum(e.file_size for e in entries) > 100 * 1024 * 1024:
            raise ValueError("Archive expanded size limit exceeded")
        for entry in entries:
            name = PurePosixPath(entry.filename)
            if (
                name.is_absolute()
                or ".." in name.parts
                or ":" in entry.filename
                or "\\" in entry.filename
                or entry.filename in result
                or stat.S_ISLNK(entry.external_attr >> 16)
                or entry.flag_bits & 1
            ):
                raise ValueError("Invalid archive member")
            content = bundle.read(entry)  # Checks ZIP CRC as well.
            if len(content) != entry.file_size:
                raise ValueError("Archive member size mismatch")
            result[entry.filename] = content
    return result


def align(era5, ndvi, zone_id, tolerance_days=3):
    import pandas as pd

    parts = []
    for start, end in [("2021-01-01", "2022-12-31"), ("2023-01-01", "2024-12-31")]:
        climate = era5.loc[(era5.zone_id == zone_id) & era5.date.between(start, end)].copy()
        observations = ndvi.loc[(ndvi.zone_id == zone_id) & ndvi.date.between(start, end)].copy()
        if climate.date.duplicated().any():
            raise ValueError("Duplicate ERA5 dates")
        if climate.empty or observations.empty:
            continue
        climate = climate.drop(columns=["geometry", "geometry_wkt"], errors="ignore")
        observations = observations[["date", "ndvi_mean"]].rename(columns={"date": "ndvi_date"})
        merged = pd.merge_asof(
            observations.sort_values("ndvi_date"),
            climate.sort_values("date"),
            left_on="ndvi_date",
            right_on="date",
            direction="nearest",
            tolerance=pd.Timedelta(days=tolerance_days),
        ).dropna(subset=["date"])
        merged["day_delta"] = (merged.date - merged.ndvi_date).abs().dt.days
        # The existing aligner prefers smallest delta, then highest NDVI.
        merged = merged.sort_values(
            ["date", "day_delta", "ndvi_mean"], ascending=[True, True, False], kind="stable"
        ).drop_duplicates("date")
        parts.append(merged)
    return (
        pd.concat(parts, ignore_index=True).sort_values("date").reset_index(drop=True)
        if parts
        else pd.DataFrame(
            columns=["date", "ndvi_date", "zone_id", "day_delta", "ndvi_mean", *CLIMATE]
        )
    )


def prepare(era5_archive, b3_archive, output):
    import duckdb
    import numpy as np
    import pandas as pd

    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    era5 = read_archive(era5_archive, ERA5_SHA256)
    b3 = read_archive(b3_archive, B3_SHA256)
    provenance = {
        "schema_version": "geoai-heldout-pilot-inputs/v1",
        "status": "complete",
        "archives": [
            {"path": str(Path(era5_archive).resolve()), "sha256": ERA5_SHA256},
            {"path": str(Path(b3_archive).resolve()), "sha256": B3_SHA256},
        ],
        "alignment": {
            "method": "nearest ERA5 date within ±3 days, tie earlier; collision nearest then higher NDVI",
            "tolerance_days": 3,
            "periods_aligned_separately": True,
            "fitted_parameters": [],
            "status": "proposed; no alignment ratification",
        },
        "source_manifests": {
            "era5_sha256": hashlib.sha256(era5["bundle-manifest.json"]).hexdigest(),
            "b3_sha256": hashlib.sha256(b3["manifest.json"]).hexdigest(),
        },
        "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "runtime": {"duckdb": duckdb.__version__, "pandas": pd.__version__},
        "notes": [
            "Fresh flat numeric materialization; not recovery of historical B4 Parquet bytes.",
            "B3 later pilot geometries differ from Q3; ERA5 sample points predate the boundary change.",
            "Geography is limited to three French Csa pilots; Metropole and Lez overlap.",
            "Sparse monthly B3 observations; no inference of daily NDVI coverage.",
        ],
    }
    tables, inputs = {}, []
    with (
        tempfile.TemporaryDirectory(prefix="geoai-pilot-preparation-") as folder,
        duckdb.connect(
            ":memory:",
            config={"autoinstall_known_extensions": "false", "autoload_known_extensions": "false"},
        ) as conn,
    ):
        for zone in PILOTS:
            frames = []
            for archive, key in [
                (era5, f"objects/climate/holdout/era5/2021_2025/{zone}.parquet"),
                (b3, f"climate/ndvi/{zone}.parquet"),
            ]:
                raw = archive[key]
                path = Path(folder) / "source.parquet"
                path.write_bytes(raw)
                frame = conn.execute("SELECT * FROM read_parquet(?)", [str(path)]).df()
                frame["date"] = pd.to_datetime(frame["date"], errors="raise")
                frames.append(frame)
                inputs.append(
                    {"key": key, "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}
                )
            table = align(*frames, zone)
            if not np.isfinite(table[["ndvi_mean", *CLIMATE]].to_numpy(dtype=float)).all():
                raise ValueError(f"Nonfinite aligned features: {zone}")
            tables[zone] = table
        output.mkdir(parents=True)
        for zone, table in tables.items():
            conn.register("unified", table)
            path = output / f"{zone}.parquet"
            conn.execute("COPY unified TO ? (FORMAT PARQUET)", [str(path)])
    provenance["inputs"] = inputs
    provenance["outputs"] = [
        {
            "zone_id": z,
            "path": str((output / f"{z}.parquet").resolve()),
            "sha256": hashlib.sha256((output / f"{z}.parquet").read_bytes()).hexdigest(),
            "rows": len(f),
            "train_rows": int(f.date.between("2021-01-01", "2022-12-31").sum()),
            "evaluation_rows": int(f.date.between("2023-01-01", "2024-12-31").sum()),
            "maximum_day_delta": int(f.day_delta.max()) if len(f) else None,
        }
        for z, f in tables.items()
    ]
    _write_json(output / "provenance.json", provenance)
    return provenance


def main(argv=None):
    from src.utils.logging import setup_logging

    setup_logging()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--era5-archive", type=Path, required=True)
    parser.add_argument("--b3-archive", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    result = prepare(args.era5_archive, args.b3_archive, args.output_dir)
    print(json.dumps(result["outputs"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
