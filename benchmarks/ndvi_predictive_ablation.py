#!/usr/bin/env python3
"""Frozen exploratory NDVI ablation, entirely offline and create-only.

The production adapter supplies native GRAANK; this module alone constructs
calendar prediction examples, training-fitted features and scored artifacts.
Importing the module never runs mining, fitting, scoring or archive loading.
"""

from __future__ import annotations

import argparse
import calendar
import csv
import hashlib
import importlib.metadata
import io
import json
import math
import os
import platform
import stat
import subprocess
import sys
import tempfile
import time
from datetime import UTC, date, datetime
from itertools import combinations
from pathlib import Path
from zipfile import ZipFile

import numpy as np

try:
    import resource
except ImportError:  # Windows has no resource module; memory reporting is optional.
    resource = None

SITES = ("fos_sur_mer", "bassin_du_lez", "montpellier_metropole")
CLIMATE = (
    "temperature_2m",
    "precipitation",
    "humidity",
    "evapotranspiration",
    "wind_speed",
    "solar_radiation",
)
DRIVERS = ["origin_ndvi"] + [f"{v}_t{lag}" for lag in (0, 1) for v in CLIMATE]
ARMS = (
    "persistence",
    "seasonal_median",
    "previous_year",
    "raw_ridge",
    "rank_ridge",
    "so4gp_ridge",
    "generic_ridge",
)
UNITS = {
    "temperature_2m": "degree Celsius",
    "precipitation": "mm",
    "humidity": "%",
    "evapotranspiration": "mm",
    "wind_speed": "m/s",
    "solar_radiation": "W/m2",
    "ndvi": "dimensionless",
}
ROOT = Path(__file__).resolve().parents[1]
ARCHIVE_MAX_BYTES = 64 * 1024 * 1024
MEMBER_MAX_BYTES = 16 * 1024 * 1024
PROTOCOL_PINS = {
    "PROTOCOL.md": "88aa965f3f588e9f7b7db1773962380adb372f402f00c39fcdee23382d33bbff",
    "protocol.json": "bb93a903239c045f3cf08e76cb18ff9d53049d11e5f788a77291d0edee141711",
}

EXECUTION_PINS = {
    "EXECUTION_AMENDMENT.md": "dcbbe7f00d8d2daa6013bf0ad56fbf74e04ef1578ceac8ef155656e20c39c8b8",
    "execution-amendment.json": "b80ae2f7b6788c11b1173412766374c17b0e4a7f1642031f5efd8a95d00cd2ab",
}
EXECUTION_FREEZE_SHA256 = "bf5dd74f100e05204db677cb1d34a10d540cea88a2752367d2e9a02c2dc2691a"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def encode(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")


def verify_protocol(path):
    path = Path(path)
    freeze_bytes = (path / "protocol-freeze.json").read_bytes()
    freeze = json.loads(freeze_bytes)
    if sha(freeze_bytes) != "7da7456311018fc419828ca26e83bfb6228ecf212baf98df9e3619fc6556a5e8":
        raise ValueError("Frozen protocol receipt mutation")
    if freeze["files"] != PROTOCOL_PINS:
        raise ValueError("Protocol freeze differs from code-pinned protocol")
    for name, digest in PROTOCOL_PINS.items():
        if sha((path / name).read_bytes()) != digest:
            raise ValueError(f"Frozen protocol mutation: {name}")
    execution_freeze_bytes = (path / "execution-freeze.json").read_bytes()
    if sha(execution_freeze_bytes) != EXECUTION_FREEZE_SHA256:
        raise ValueError("Frozen execution receipt mutation")
    execution_freeze = json.loads(execution_freeze_bytes)
    if execution_freeze["files"] != EXECUTION_PINS:
        raise ValueError("Execution freeze differs from code-pinned amendment")
    for name, digest in EXECUTION_PINS.items():
        if sha((path / name).read_bytes()) != digest:
            raise ValueError(f"Frozen execution amendment mutation: {name}")
    amendment = json.loads((path / "execution-amendment.json").read_bytes())
    config = json.loads((path / "protocol.json").read_bytes())
    config.update(amendment["overrides"])
    if (
        config.get("native_strategy") != "all_driver_pairs_plus_target"
        or config.get("maximum_antecedent_size") != 2
    ):
        raise ValueError("Unsupported frozen execution amendment")
    return config, {
        "files": PROTOCOL_PINS,
        "freeze_sha256": sha(freeze_bytes),
        "freeze": freeze,
        "execution_files": EXECUTION_PINS,
        "execution_freeze_sha256": sha(execution_freeze_bytes),
        "execution_freeze": execution_freeze,
        "execution_amendment": amendment,
    }


def verify_runtime(config):
    versions = {
        name: importlib.metadata.version(name)
        for name in ("so4gp", "numpy", "pandas", "duckdb", "torch")
    }
    if versions["so4gp"] != config["native_version"]:
        raise ValueError(f"Native version mismatch: {versions['so4gp']}")
    import torch

    if torch.version.cuda is not None:
        raise ValueError("Requires the pinned CPU-only torch runtime")
    return {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "dependencies": versions,
        "torch_cuda_build": torch.version.cuda,
        "device": "cpu",
        "platform": platform.platform(),
    }


def create_output_dir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=False)
    return path


def snapshot_archive(path, expected):
    """Hash the same bounded immutable byte snapshot later parsed by ZipFile."""
    with Path(path).open("rb") as stream:
        before = os.fstat(stream.fileno())
        if not stat.S_ISREG(before.st_mode) or before.st_size > ARCHIVE_MAX_BYTES:
            raise ValueError("Archive must be a bounded regular file")
        data = stream.read(ARCHIVE_MAX_BYTES + 1)
        after = os.fstat(stream.fileno())

    def identity(s):
        return (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)

    if identity(before) != identity(after) or len(data) != before.st_size:
        raise ValueError("Archive changed during snapshot")
    if sha(data) != expected:
        raise ValueError("Archive SHA-256 mismatch")
    return data


def _day(value):
    if isinstance(value, datetime):
        if value.tzinfo is not None:
            value = value.astimezone(UTC)
        if value.hour or value.minute or value.second or value.microsecond:
            raise ValueError("Only UTC daily midnight source dates are supported")
        return value.date()
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value)[:10])


def _shift(month, offset):
    y, m = map(int, month.split("-"))
    i = y * 12 + m - 1 + offset
    return f"{i // 12:04d}-{i % 12 + 1:02d}"


def _month_end(month):
    y, m = map(int, month.split("-"))
    return date(y, m, calendar.monthrange(y, m)[1])


def _finite(value):
    try:
        return math.isfinite(float(value))
    except (ValueError, TypeError):
        return False


def load_inputs(ndvi_archive, era5_archive, config, private_dir):
    """Only six fixed Parquet member paths are materialized; never extractall."""
    import duckdb

    # No extension autoload or installation; local read_parquet needs no extension.
    conn = duckdb.connect(
        ":memory:",
        config={"autoload_known_extensions": "false", "autoinstall_known_extensions": "false"},
    )
    rows = {"ndvi": [], "era5": []}
    provenance = {"archives": {}, "members": [], "units": UNITS}
    snapshots = {
        kind: snapshot_archive(supplied, config["input_archives"][kind]["sha256"])
        for kind, supplied in (("ndvi", ndvi_archive), ("era5", era5_archive))
    }
    try:
        for kind, data in snapshots.items():
            spec = config["input_archives"][kind]
            provenance["archives"][kind] = {
                "name": spec["name"],
                "sha256": sha(data),
                "bytes": len(data),
            }
            with ZipFile(io.BytesIO(data)) as archive:
                names = archive.namelist()
                for site in SITES:
                    member = (
                        f"climate/ndvi/{site}.parquet"
                        if kind == "ndvi"
                        else f"objects/climate/holdout/era5/2021_2025/{site}.parquet"
                    )
                    if names.count(member) != 1:
                        raise ValueError(f"Missing or duplicate allowed member: {member}")
                    info = archive.getinfo(member)
                    if info.file_size > MEMBER_MAX_BYTES or info.flag_bits & 1:
                        raise ValueError("Unbounded or encrypted archive member")
                    payload = archive.read(member)
                    private_path = Path(private_dir) / f"{kind}-{site}.parquet"
                    with private_path.open("xb") as stream:
                        stream.write(payload)
                    frame = conn.execute(
                        "SELECT * FROM read_parquet(?)", [str(private_path)]
                    ).fetchdf()
                    schema = {str(c): str(t) for c, t in frame.dtypes.items()}
                    required = (
                        {"date", "zone_id", "lat", "lon", "ndvi_mean"}
                        if kind == "ndvi"
                        else {"date", "zone_id", "lat", "lon", *CLIMATE}
                    )
                    if not required <= set(frame.columns) or set(frame.zone_id) != {site}:
                        raise ValueError(f"Unsupported source schema/site in {member}")
                    coordinates = sorted({(float(r.lat), float(r.lon)) for r in frame.itertuples()})
                    if len(coordinates) != 1 or not all(map(math.isfinite, coordinates[0])):
                        raise ValueError("Invalid or changing source coordinates")
                    provenance["members"].append(
                        {
                            "kind": kind,
                            "site": site,
                            "path": member,
                            "sha256": sha(payload),
                            "bytes": len(payload),
                            "rows": len(frame),
                            "schema": schema,
                            "source_coordinates": {
                                "latitude": coordinates[0][0],
                                "longitude": coordinates[0][1],
                            },
                        }
                    )
                    for raw in frame.to_dict("records"):
                        row = {
                            "site": site,
                            "date": _day(raw["date"]).isoformat(),
                            "latitude": float(raw["lat"]),
                            "longitude": float(raw["lon"]),
                        }
                        if kind == "ndvi":
                            row["ndvi"] = float(raw["ndvi_mean"])
                            row["item_id"] = str(raw.get("item_id", ""))
                        else:
                            row.update({v: float(raw[v]) for v in CLIMATE})
                        rows[kind].append(row)
    finally:
        conn.close()
    return rows["ndvi"], rows["era5"], provenance


def build_examples(ndvi_rows, era5_rows, config):
    """Exact month joins; climate is already in the declared normalized units."""
    history = {site: {} for site in SITES}
    daily = {site: {} for site in SITES}
    invalid = []
    for row in ndvi_rows:
        site = row["site"]
        if site not in history:
            raise ValueError("Unsupported NDVI site")
        day = _day(row["date"])
        month = day.strftime("%Y-%m")
        if month in history[site]:
            raise ValueError(f"Multiple retained NDVI observations: {site}/{month}")
        if not _finite(row["ndvi"]) or not -1 <= float(row["ndvi"]) <= 1:
            invalid.append({"site": site, "date": day.isoformat(), "reason": "invalid_ndvi"})
        history[site][month] = {**row, "date": day.isoformat()}
    for row in era5_rows:
        site = row["site"]
        if site not in daily:
            raise ValueError("Unsupported climate site")
        day = _day(row["date"])
        month = day.strftime("%Y-%m")
        daily[site].setdefault(month, []).append({**row, "date": day.isoformat()})
    coverage, aggregate = {}, {}
    for site in SITES:
        coverage[site], aggregate[site] = {}, {}
        for month, rows in sorted(daily[site].items()):
            y, m = map(int, month.split("-"))
            days_expected = calendar.monthrange(y, m)[1]
            dates = [r["date"] for r in rows]
            reasons = []
            if len(set(dates)) != len(dates):
                reasons.append("duplicate_climate_day")
            if {int(d[-2:]) for d in dates} != set(range(1, days_expected + 1)):
                reasons.append("incomplete_climate_month")
            for r in rows:
                for v in CLIMATE:
                    val = r.get(v)
                    if not _finite(val):
                        reasons.append(f"nonfinite_{v}")
                    elif (
                        v
                        in ("precipitation", "evapotranspiration", "wind_speed", "solar_radiation")
                        and float(val) < 0
                    ) or (v == "humidity" and not 0 <= float(val) <= 100):
                        reasons.append(f"invalid_range_{v}")
            coverage[site][month] = {
                "days_expected": days_expected,
                "rows": len(rows),
                "distinct_days": len(set(dates)),
                "reasons": sorted(set(reasons)),
                "complete": not reasons,
            }
            if not reasons:
                values = {}
                for v, agg in config["monthly_aggregations"].items():
                    x = np.asarray([r[v] for r in rows], dtype=float)
                    values[v] = float({"mean": np.mean, "sum": np.sum, "max": np.max}[agg](x))
                aggregate[site][month] = values
    if invalid:
        raise ValueError(f"Rejected NDVI values: {invalid}")
    examples, exclusions = [], []
    years = sorted(set(config["train_target_years"] + config["evaluation_target_years"]))
    for site in SITES:
        for year in years:
            for month_num in range(1, 13):
                target_month = f"{year:04d}-{month_num:02d}"
                origin_month, previous_month = _shift(target_month, -1), _shift(target_month, -2)
                reasons = []
                for label, month in (("origin", origin_month), ("target", target_month)):
                    if month not in history[site]:
                        reasons.append(f"missing_{label}_ndvi")
                for lag, month in ((0, origin_month), (1, previous_month)):
                    if month not in aggregate[site]:
                        reasons.append(f"unavailable_climate_t{lag}")
                        reasons.extend(
                            f"t{lag}:{r}"
                            for r in coverage[site]
                            .get(month, {})
                            .get("reasons", ["missing_climate_month"])
                        )
                base = {
                    "site": site,
                    "target_month": target_month,
                    "origin_month": origin_month,
                    "origin_date": _month_end(origin_month).isoformat(),
                    "split": "train" if year in config["train_target_years"] else "evaluation",
                }
                if reasons:
                    exclusions.append({**base, "reasons": reasons})
                    continue
                origin, target = history[site][origin_month], history[site][target_month]
                drivers = [float(origin["ndvi"])] + [
                    aggregate[site][month][v]
                    for month in (origin_month, previous_month)
                    for v in CLIMATE
                ]
                examples.append(
                    {
                        **base,
                        "origin_ndvi": float(origin["ndvi"]),
                        "origin_ndvi_date": origin["date"],
                        "target": float(target["ndvi"]),
                        "target_date": target["date"],
                        "horizon_days": (_day(target["date"]) - _month_end(origin_month)).days,
                        "drivers": drivers,
                        "seasonal": [
                            math.sin(2 * math.pi * month_num / 12),
                            math.cos(2 * math.pi * month_num / 12),
                        ],
                        "climate_months": [origin_month, previous_month],
                        "monthly_coverage": [
                            coverage[site][origin_month],
                            coverage[site][previous_month],
                        ],
                    }
                )
    return {
        "examples": examples,
        "exclusions": exclusions,
        "ndvi_history": history,
        "monthly_coverage": coverage,
    }


def empirical_ranks(train_values, values):
    train = np.asarray(train_values, dtype=float)
    values = np.asarray(values, dtype=float)
    if (
        train.ndim != 1
        or not len(train)
        or not np.isfinite(train).all()
        or not np.isfinite(values).all()
    ):
        raise ValueError("CDF requires nonempty finite training vector and finite inputs")
    reference = np.sort(train)
    return (
        np.searchsorted(reference, values, side="left")
        + np.searchsorted(reference, values, side="right")
    ) / (2.0 * len(reference))


def strict_support(matrix, items):
    matrix = np.asarray(matrix, dtype=float)
    n = len(matrix)
    if matrix.ndim != 2 or n < 2 or not np.isfinite(matrix).all() or not items:
        raise ValueError("Support requires finite matrix with two rows and nonempty items")
    concordance = np.ones((n, n), dtype=bool)
    for column, direction in items:
        if direction not in ("+", "-"):
            raise ValueError("Invalid gradual direction")
        x = matrix[:, int(column)]
        concordance &= (x[:, None] < x[None, :]) if direction == "+" else (x[:, None] > x[None, :])
    return float(concordance.sum() / (n * (n - 1) / 2))


def select_patterns(normalized_patterns, train_matrix, feature_names, config):
    """Canonicalize inverse itemsets and independently recount native support."""
    candidates, unique = [], {}
    for index, pattern in enumerate(normalized_patterns):
        record = {"native_index": index, "native_pattern": pattern, "reason": None}
        raw = pattern.get("gradual_items", [])
        attributes = [i.get("attribute") for i in raw]
        if len(attributes) != len(set(attributes)) or any(
            i.get("direction") not in ("+", "-") or i.get("attribute") not in feature_names
            for i in raw
        ):
            raise ValueError("Malformed native itemset")
        target = next((i for i in raw if i["attribute"] == "target_next_ndvi"), None)
        if target is None:
            record["reason"] = "no_target"
        elif len(raw) - 1 < config["minimum_antecedent_size"]:
            record["reason"] = "too_few_antecedents"
        elif len(raw) - 1 > config.get("maximum_antecedent_size", float("inf")):
            record["reason"] = "too_many_antecedents"
        else:
            inverse = target["direction"] == "-"
            canonical = sorted(
                (
                    {
                        "attribute": i["attribute"],
                        "direction": ("-" if i["direction"] == "+" else "+")
                        if inverse
                        else i["direction"],
                    }
                    for i in raw
                ),
                key=lambda i: (i["attribute"], i["direction"]),
            )
            signature = ",".join(i["attribute"] + i["direction"] for i in canonical)
            exact = strict_support(
                train_matrix,
                [(feature_names.index(i["attribute"]), i["direction"]) for i in canonical],
            )
            reported = pattern.get("support")
            if not _finite(reported):
                raise ValueError("Missing finite native support")
            record.update(
                signature=signature,
                items=[i for i in canonical if i["attribute"] != "target_next_ndvi"],
                native_support=float(reported),
                exact_support=exact,
                support_discrepancy=abs(exact - float(reported)),
            )
            if record["support_discrepancy"] > config["support_recount_tolerance"] + 1e-12:
                record["reason"] = "support_recount_failure"
            elif exact <= config["native_min_support"]:
                record["reason"] = "support_not_strictly_above_threshold"
            else:
                previous = unique.get(signature)
                if previous is None:
                    unique[signature] = record
                elif float(reported) > previous["native_support"]:
                    previous["reason"] = "duplicate_inverse"
                    unique[signature] = record
                else:
                    record["reason"] = "duplicate_inverse"
        candidates.append(record)
    if any(c["reason"] == "support_recount_failure" for c in candidates):
        return {
            "selected": [],
            "candidates": candidates,
            "failure": "native_support_recount_discrepancy",
        }
    ranked = sorted(unique.values(), key=lambda r: (-r["native_support"], r["signature"]))
    selected = []
    for index, r in enumerate(ranked):
        r["reason"] = "selected" if index < config["maximum_mined_features"] else "feature_limit"
        if r["reason"] == "selected":
            selected.append(
                {k: r[k] for k in ("items", "signature", "native_support", "exact_support")}
            )
    return {"selected": selected, "candidates": candidates, "failure": None}


def generic_interactions(selected, feature_names, seed):
    rng = np.random.default_rng(seed)
    result = []
    mined_identities = {
        tuple((i["attribute"], i["direction"]) for i in p["items"]) for p in selected
    }
    seen = set()
    for pattern in selected:
        arity = len(pattern["items"])
        positives = sum(i["direction"] == "+" for i in pattern["items"])
        attrs = rng.choice(feature_names, size=arity, replace=False).tolist()
        signs = np.array(["+"] * positives + ["-"] * (arity - positives))
        rng.shuffle(signs)
        items = sorted(
            ({"attribute": a, "direction": str(s)} for a, s in zip(attrs, signs, strict=True)),
            key=lambda i: i["attribute"],
        )
        identity = tuple((i["attribute"], i["direction"]) for i in items)
        result.append(
            {
                "items": items,
                "signature": ",".join(a + s for a, s in identity),
                "matches_mined": identity in mined_identities,
                "duplicate_generic": identity in seen,
                "arity": arity,
                "positive_signs": positives,
            }
        )
        seen.add(identity)
    return result


def interaction_matrix(ranks, patterns, feature_names=DRIVERS):
    ranks = np.asarray(ranks, dtype=float)
    if not patterns:
        return np.empty((len(ranks), 0), dtype=float)
    columns = []
    for pattern in patterns:
        signed = [
            ranks[:, feature_names.index(i["attribute"])]
            if i["direction"] == "+"
            else 1 - ranks[:, feature_names.index(i["attribute"])]
            for i in pattern["items"]
        ]
        columns.append(np.min(np.column_stack(signed), axis=1))
    return np.column_stack(columns)


def fit_ridge(X, y, alpha=10.0, feature_names=None):
    X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=float)
    if (
        X.ndim != 2
        or y.shape != (len(X),)
        or not len(X)
        or not np.isfinite(X).all()
        or not np.isfinite(y).all()
        or alpha <= 0
    ):
        raise ValueError("Invalid finite ridge data or alpha")
    mean, std = X.mean(axis=0), X.std(axis=0)
    keep = np.flatnonzero(np.ptp(X, axis=0) > 0)
    Z = (X[:, keep] - mean[keep]) / std[keep]
    intercept = float(y.mean())
    coef = np.linalg.solve(Z.T @ Z + alpha * np.eye(len(keep)), Z.T @ (y - intercept))
    names = (
        list(feature_names) if feature_names is not None else [str(i) for i in range(X.shape[1])]
    )
    return {
        "mean": mean.tolist(),
        "std": std.tolist(),
        "keep": keep.tolist(),
        "coefficients": coef.tolist(),
        "intercept": intercept,
        "alpha": float(alpha),
        "input_feature_names": names,
        "kept_feature_names": [names[i] for i in keep],
        "dropped_constant_features": [names[i] for i in range(X.shape[1]) if i not in keep],
        "training_rows": len(X),
    }


def predict_ridge(model, X):
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or X.shape[1] != len(model["mean"]) or not np.isfinite(X).all():
        raise ValueError("Invalid finite ridge predictors")
    keep = np.asarray(model["keep"], dtype=int)
    Z = (X[:, keep] - np.asarray(model["mean"])[keep]) / np.asarray(model["std"])[keep]
    return model["intercept"] + Z @ np.asarray(model["coefficients"])


def native_mine(matrix, feature_names, config, artifact_dir):
    """Separate bounded process, JSON transport only, private cwd, real adapter."""
    if config.get("native_strategy") != "all_driver_pairs_plus_target":
        raise ValueError("Unsupported native strategy; verified execution amendment required")
    artifact_dir = Path(artifact_dir)
    artifact_dir.mkdir()
    job = {
        "matrix": np.asarray(matrix).tolist(),
        "feature_names": feature_names,
        "min_support": config["native_min_support"],
        "encode_equal": config["encode_equal"],
        "version": config["native_version"],
        "native_strategy": config["native_strategy"],
    }
    (artifact_dir / "input.json").write_bytes(encode(job))
    start = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="ndvi-native-") as cwd:
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--native-worker",
            str((artifact_dir / "input.json").resolve()),
            str(artifact_dir.resolve()),
        ]
        environment = dict(
            os.environ,
            CUDA_VISIBLE_DEVICES="",
            OMP_NUM_THREADS="1",
            MKL_NUM_THREADS="1",
            OPENBLAS_NUM_THREADS="1",
            PYTHONHASHSEED="0",
            MPLCONFIGDIR=str(Path(cwd) / "matplotlib"),
        )
        try:
            run = subprocess.run(
                command,
                cwd=cwd,
                env=environment,
                capture_output=True,
                timeout=config["native_timeout_seconds"],
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            (artifact_dir / "stdout.txt").write_bytes(exc.stdout or b"")
            (artifact_dir / "stderr.txt").write_bytes(exc.stderr or b"")
            raise RuntimeError("Native mining timed out at frozen timeout") from exc
    (artifact_dir / "stdout.txt").write_bytes(run.stdout)
    (artifact_dir / "stderr.txt").write_bytes(run.stderr)
    if run.returncode:
        raise RuntimeError(f"Native mining failed with exit {run.returncode}; see retained stderr")
    parsed = json.loads((artifact_dir / "parsed.json").read_bytes())
    parsed["wall_seconds"] = time.monotonic() - start
    return parsed


def bounded_subsets(feature_names):
    """Every unordered driver pair, with target last; no target-guided pruning."""
    names = list(feature_names)
    target = "target_next_ndvi"
    if len(names) != len(set(names)) or names.count(target) != 1:
        raise ValueError("Need unique feature names and exactly one target")
    drivers = [n for n in names if n != target]
    if len(drivers) < 2:
        raise ValueError("Need at least two antecedent drivers")
    return [[a, b, target] for a, b in combinations(drivers, 2)]


def native_pair_mining(matrix, feature_names, config, artifact_dir, adapter=None):
    """Unchanged native discovery on each declared triplet; optional test injection.

    The actual worker always supplies the production adapter. No CLI mechanism
    accepts a replacement adapter or synthetic native output.
    """
    import pandas as pd

    if config.get("native_strategy") != "all_driver_pairs_plus_target":
        raise ValueError("Unsupported native strategy; verified execution amendment required")
    if adapter is None:
        sys.path.insert(0, str(ROOT))
        from src.patterns.so4gp_adapter import mine_gradual_patterns

        adapter = mine_gradual_patterns
    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim != 2 or matrix.shape[1] != len(feature_names) or not np.isfinite(matrix).all():
        raise ValueError("Finite matching native matrix required")
    artifact_dir = Path(artifact_dir)
    inventory, patterns, raw_patterns = [], [], []
    for index, subset in enumerate(bounded_subsets(feature_names)):
        call_dir = artifact_dir / f"call-{index:03d}"
        call_dir.mkdir()
        submatrix = matrix[:, [feature_names.index(name) for name in subset]]
        call_input = {
            "matrix": submatrix.tolist(),
            "feature_names": subset,
            "min_support": config["native_min_support"],
            "encode_equal": config["encode_equal"],
            "algorithm": "graank",
            "device": "cpu",
        }
        input_bytes = encode(call_input)
        (call_dir / "input.json").write_bytes(input_bytes)
        captured = []

        def capture(raw, captured=captured, raw_path=call_dir / "raw-native.json"):
            captured.append(raw)
            with raw_path.open("xb") as stream:
                stream.write(raw.encode("utf-8"))

        result = adapter(
            pd.DataFrame(submatrix, columns=subset),
            min_support=config["native_min_support"],
            encode_equal=config["encode_equal"],
            algorithm="graank",
            capture_raw_output=capture,
        )
        if len(captured) != 1:
            raise ValueError("Native adapter must capture exactly one raw output per call")
        raw_result = json.loads(captured[0])
        if raw_result.get("Algorithm") != "GRAANK" or raw_result.get("Error"):
            raise ValueError("Native pair discovery failed or returned another algorithm")
        raw_patterns.extend(raw_result.get("Patterns", []))
        (call_dir / "parsed.json").write_bytes(encode(result))
        for pattern in result["patterns"]:
            patterns.append({**pattern, "native_call_index": index, "native_call_columns": subset})
        inventory.append(
            {
                "index": index,
                "feature_names": subset,
                "input_sha256": sha(input_bytes),
                "raw_sha256": sha(captured[0].encode("utf-8")),
                "normalized_patterns": len(result["patterns"]),
                "runtime_seconds": result.get("runtime_seconds"),
            }
        )
    combined = {
        "algorithm": "GRAANK",
        "algorithm_key": "graank",
        "native_strategy": config["native_strategy"],
        "native_call_count": len(inventory),
        "patterns": patterns,
        "min_support": config["native_min_support"],
        "encode_equal": config["encode_equal"],
    }
    (artifact_dir / "call-inventory.json").write_bytes(encode(inventory))
    (artifact_dir / "raw-native.json").write_bytes(
        encode(
            {
                "Algorithm": "GRAANK",
                "Patterns": raw_patterns,
                "Native call count": len(inventory),
                "Note": "Aggregate only; exact native raw bytes are retained separately per call.",
            }
        )
    )
    return combined


def _native_worker(input_path, output_path):
    sys.path.insert(0, str(ROOT))
    from src.patterns.so4gp_adapter import mine_gradual_patterns

    job = json.loads(Path(input_path).read_bytes())
    if importlib.metadata.version("so4gp") != job["version"]:
        raise ValueError("Worker native version mismatch")
    import torch

    if torch.version.cuda is not None:
        raise ValueError("Worker requires CPU-only torch")
    torch.set_num_threads(1)
    output_path = Path(output_path)
    config = {
        "native_strategy": job["native_strategy"],
        "native_min_support": job["min_support"],
        "encode_equal": job["encode_equal"],
    }
    result = native_pair_mining(
        job["matrix"], job["feature_names"], config, output_path, adapter=mine_gradual_patterns
    )
    (output_path / "parsed.json").write_bytes(encode(result))


def _metrics(rows):
    error = np.asarray([r["prediction"] - r["actual"] for r in rows])
    return {
        "n": len(rows),
        "mae": float(np.abs(error).mean()),
        "rmse": float(np.sqrt(np.square(error).mean())),
        "clipped": sum(r["clipped"] for r in rows),
    }


def paired_bootstrap(differences, config):
    differences = np.asarray(differences, dtype=float)
    n = len(differences)
    rng = np.random.default_rng(config["bootstrap_seed"])
    block, draws = config["bootstrap_block_months"], config["bootstrap_draws"]
    samples = []
    for _ in range(draws):
        starts = rng.integers(0, n, size=math.ceil(n / block))
        indices = np.concatenate([(s + np.arange(block)) % n for s in starts])[:n]
        samples.append(float(differences[indices].mean()))
    return {
        "method": "paired circular block bootstrap conditional on fitted models",
        "block_months": block,
        "draws": draws,
        "seed": config["bootstrap_seed"],
        "percentiles": [2.5, 97.5],
        "interval": np.percentile(samples, [2.5, 97.5]).tolist(),
        "interpretation": "Exploratory inspected-holdout descriptive interval; omits model-selection uncertainty; no confirmatory or calibration claim.",
    }


def evaluate_site(examples, ndvi_history, config, native_result=None):
    examples = sorted(examples, key=lambda e: e["target_month"])
    train = [e for e in examples if e["split"] == "train"]
    evaluation = [e for e in examples if e["split"] == "evaluation"]
    if not train or not evaluation or len({e["site"] for e in examples}) != 1:
        raise ValueError("Need training and evaluation examples for one site")
    if native_result is None:
        raise ValueError("Actual native result required; no fixture fallback")
    X = np.asarray([e["drivers"] for e in examples], dtype=float)
    seasons = np.asarray([e["seasonal"] for e in examples], dtype=float)
    mask = np.array([e["split"] == "train" for e in examples])
    y = np.asarray([e["target"] for e in train], dtype=float)
    mining_matrix = np.column_stack([X[mask], y])
    selection = select_patterns(
        native_result["patterns"], mining_matrix, DRIVERS + ["target_next_ndvi"], config
    )
    if selection["failure"]:
        return {"status": "mining_failure", "selection": selection}
    generic = generic_interactions(
        selection["selected"], DRIVERS, config["generic_interaction_seed"]
    )
    ranks = np.column_stack([empirical_ranks(X[mask, j], X[:, j]) for j in range(X.shape[1])])
    raw = np.column_stack([X, seasons])
    ranked = np.column_stack([raw, ranks])
    raw_names = DRIVERS + ["target_month_sin", "target_month_cos"]
    ranked_names = raw_names + [f"rank({n})" for n in DRIVERS]
    matrices = {
        "raw_ridge": raw,
        "rank_ridge": ranked,
        "so4gp_ridge": np.column_stack([ranked, interaction_matrix(ranks, selection["selected"])]),
        "generic_ridge": np.column_stack([ranked, interaction_matrix(ranks, generic)]),
    }
    names = {
        "raw_ridge": raw_names,
        "rank_ridge": ranked_names,
        "so4gp_ridge": ranked_names + [p["signature"] for p in selection["selected"]],
        "generic_ridge": ranked_names + [p["signature"] for p in generic],
    }
    models = {
        arm: fit_ridge(matrix[mask], y, config["ridge_alpha"], names[arm])
        for arm, matrix in matrices.items()
    }
    learned = {arm: predict_ridge(models[arm], matrix) for arm, matrix in matrices.items()}
    seasonal_values = {
        m: [
            float(row["ndvi"])
            for month, row in ndvi_history.items()
            if int(month[:4]) in config["train_target_years"] and int(month[5:]) == m
        ]
        for m in range(1, 13)
    }
    all_training_obs = [v for vals in seasonal_values.values() for v in vals]
    if not all_training_obs:
        raise ValueError("No observed training seasonal baseline data")
    global_median = float(np.median(all_training_obs))
    medians = {m: float(np.median(v)) if v else global_median for m, v in seasonal_values.items()}
    predictions = []
    bounds = config["ndvi_prediction_bounds"]
    for index, example in enumerate(examples):
        # Retain training fitted values as diagnostics; metrics use evaluation only.
        seasonal = medians[int(example["target_month"][5:])]
        previous = ndvi_history.get(_shift(example["target_month"], -12))
        previous_ok = previous is not None and _day(previous["date"]) <= _day(
            example["origin_date"]
        )
        values = {
            "persistence": example["origin_ndvi"],
            "seasonal_median": seasonal,
            "previous_year": float(previous["ndvi"]) if previous_ok else seasonal,
            **{a: float(v[index]) for a, v in learned.items()},
        }
        for arm in ARMS:
            value = values[arm]
            prediction = float(np.clip(value, *bounds))
            predictions.append(
                {
                    k: example[k]
                    for k in (
                        "site",
                        "split",
                        "target_month",
                        "origin_month",
                        "origin_date",
                        "origin_ndvi_date",
                        "target_date",
                        "horizon_days",
                        "climate_months",
                        "monthly_coverage",
                    )
                }
                | {
                    "arm": arm,
                    "actual": example["target"],
                    "prediction": prediction,
                    "unclipped_prediction": value,
                    "clipped": value != prediction,
                    "absolute_error": abs(prediction - example["target"]),
                    "squared_error": (prediction - example["target"]) ** 2,
                    "previous_year_fallback": arm == "previous_year" and not previous_ok,
                }
            )
    eval_rows = [r for r in predictions if r["split"] == "evaluation"]
    metrics = {a: _metrics([r for r in eval_rows if r["arm"] == a]) for a in ARMS}
    yearly = {
        str(year): {
            a: _metrics(
                [r for r in eval_rows if r["arm"] == a and int(r["target_month"][:4]) == year]
            )
            for a in ARMS
        }
        for year in config["evaluation_target_years"]
        if any(int(e["target_month"][:4]) == year for e in evaluation)
    }
    paired = []
    for e in evaluation:
        rows = {r["arm"]: r for r in eval_rows if r["target_month"] == e["target_month"]}
        paired.append(
            {
                "target_month": e["target_month"],
                **{
                    f"{a}_minus_so4gp_absolute_error": rows[a]["absolute_error"]
                    - rows["so4gp_ridge"]["absolute_error"]
                    for a in ARMS
                    if a != "so4gp_ridge"
                },
            }
        )
    contrasts = {
        f"{a}_mae_minus_so4gp_mae": metrics[a]["mae"] - metrics["so4gp_ridge"]["mae"]
        for a in ARMS
        if a != "so4gp_ridge"
    }
    contiguous = all(
        _shift(a["target_month"], 1) == b["target_month"]
        for a, b in zip(evaluation, evaluation[1:], strict=False)
    )
    bootstrap = (
        paired_bootstrap([p["rank_ridge_minus_so4gp_absolute_error"] for p in paired], config)
        if examples[0]["site"] == config["primary_site"] and contiguous
        else None
    )
    return {
        "status": "complete",
        "site": examples[0]["site"],
        "train_n": len(train),
        "evaluation_n": len(evaluation),
        "models": models,
        "rank_references": {name: sorted(X[mask, j].tolist()) for j, name in enumerate(DRIVERS)},
        "seasonal_baseline": {
            "observed_training_values": {str(k): v for k, v in seasonal_values.items()},
            "monthly_medians": {str(k): v for k, v in medians.items()},
            "global_median": global_median,
        },
        "selection": selection,
        "generic_interactions": generic,
        "predictions": predictions,
        "metrics": metrics,
        "yearly_metrics": yearly,
        "contrasts": contrasts,
        "paired_error_differences": paired,
        "bootstrap": bootstrap,
        "evaluation_contiguous": contiguous,
        "native_normalized": native_result,
    }


def _write_csv(path, rows):
    if not rows:
        with Path(path).open("xb") as stream:
            stream.write(b"")
        return
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\r\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    k: json.dumps(v, sort_keys=True) if isinstance(v, (dict, list)) else v
                    for k, v in row.items()
                }
            )


def _code_identity():
    files = [Path(__file__).resolve(), ROOT / "src/patterns/so4gp_adapter.py"]
    git = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=True,
    ).stdout.strip()
    return {
        "git_head": git,
        "files": {str(p.relative_to(ROOT)): sha(p.read_bytes()) for p in files},
        "native_distribution": {
            str(p): sha(importlib.metadata.distribution("so4gp").locate_file(p).read_bytes())
            for p in importlib.metadata.distribution("so4gp").files or []
            if str(p).endswith(".py")
        },
    }


def scientific_payload(result):
    """Portable numerical/source identity excluding paths, clocks and native timing."""
    sites = {}
    for site, value in result["sites"].items():
        clean = dict(value)
        native = clean.pop("native_normalized", {})
        clean["native_patterns"] = native.get("patterns", [])
        sites[site] = clean
    return {
        "protocol_files": result["protocol"]["files"],
        "execution_files": result["protocol"]["execution_files"],
        "execution_freeze_sha256": result["protocol"]["execution_freeze_sha256"],
        "code": result["code"],
        "runtime": result["runtime"],
        "inputs": result["inputs"],
        "examples": result["examples"],
        "exclusions": result["exclusions"],
        "monthly_coverage": result["monthly_coverage"],
        "sites": sites,
    }


def report_markdown(result):
    primary = result["sites"][result["config"]["primary_site"]]
    contrast = primary["contrasts"]["rank_ridge_mae_minus_so4gp_mae"]
    generic = primary["contrasts"]["generic_ridge_mae_minus_so4gp_mae"]
    observed = (
        "a measured incremental gain"
        if contrast > 0
        else "no measured incremental gain"
        if contrast == 0
        else "a measured loss"
    )
    lines = [
        "# Exploratory NDVI prediction ablation",
        "",
        f"Fos-sur-Mer shows {observed}: MAE(rank control) − MAE(native-so4gp) = **{contrast:.8f} NDVI** on {primary['evaluation_n']} paired evaluation targets. The corresponding generic-interaction contrast is {generic:.8f} NDVI. Positive contrasts favor native-so4gp.",
        "",
        "This is a retrospective exploratory comparison on an already inspected holdout. The question is the added information in this fixed feature package. It does not establish TGrad forecasting, causality, a stress probability, global climatic transfer, scientific/FIERCE reception, or the utility of the full pattern library.",
        "",
        "## Data, origin and provenance",
        "",
        "Training targets are 2021–2022; evaluation targets are 2023–2024. Features use origin-month observed NDVI and complete daily ERA5 summaries in origin and previous calendar months. The target acquisition day is retained only for reporting. A December 2022 origin may predict January 2023; January 2023 is never a training label. Exact calendar joins do not bridge missing observations or substitute nearby sites.",
        "",
        "The source data are processed retrospective archives, with upstream filtering and retention already applied. Their processing/publication timestamps are unavailable. Month-end availability is an explicit simulation assumption, not verified operational availability. Earlier historical observations are not added to learned training rows. The fixed previous-year reference can use a prior observation once it is before the current origin.",
        "",
        "The ERA5 and NDVI sampling points differ; monthly joins here do not ratify geographic equivalence or the earlier daily alignment decision. Locations, units, schema and archive/member digests are in inputs.json. Eligible candidate target months and input coverage are in examples.json; excluded candidate months and their reasons are in exclusions.csv.",
        "",
        "## Evaluation metrics",
        "",
        "| Site | Arm | Evaluation n | MAE | RMSE | Clipped |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for site, value in result["sites"].items():
        for arm, metrics in value["metrics"].items():
            lines.append(
                f"| {site} | {arm} | {metrics['n']} | {metrics['mae']:.8f} | {metrics['rmse']:.8f} | {metrics['clipped']} |"
            )
    lines.extend(
        [
            "",
            "## Mining and fitted features",
            "",
            "The initial unrestricted 14-column native search timed out at 60 seconds before model fitting or scoring; its failure is preserved separately under evidence/initial-timeout. A frozen pre-score execution amendment narrows the question to pair interactions. Each site uses all 78 unordered pairs of the 13 original drivers, each combined with the training target in one unchanged production-adapter CPU so4gp 1.1.2 GRAANK call. All 78 calls run in a single worker under the original total 60-second bound, with strict comparisons and support 0.50. No dates, seasonal encoding or held-out rows enter the miner. Retained itemsets contain the target and exactly two antecedents, are canonicalized to positive target direction, independently recounted, then sorted by reported support and lexical signature. All selection decisions and exact native output are retained. The model uses empirical training CDF ranks and signed-rank minima, with all learned features scaled using training moments and constant training columns dropped. Alpha is fixed at 10 and the intercept is unpenalized.",
            "",
        ]
    )
    for site, value in result["sites"].items():
        count = len(value["selection"]["selected"])
        lines.append(
            f"- {site}: {value['train_n']} training / {value['evaluation_n']} evaluation examples; {count} selected mined interactions; {sum(p['matches_mined'] for p in value['generic_interactions'])} generic/mined coincidences; {sum(p['duplicate_generic'] for p in value['generic_interactions'])} repeated generic identities."
        )
    lines.extend(
        [
            "",
            "The generic comparator uses one label-independent fixed-seed draw matched on arity and sign counts. It controls this ordinary interaction construction, not all possible engineering. If mining selects no interactions, rank, native and generic models coincide. A gain over ranks without a gain over this generic comparator is weaker evidence for mining itself.",
            "",
            "## Yearly paired contrasts",
            "",
            "| Site | Year | Rank MAE − native MAE | Generic MAE − native MAE |",
            "| --- | --- | ---: | ---: |",
        ]
    )
    for site, value in result["sites"].items():
        for year, metrics in value["yearly_metrics"].items():
            lines.append(
                f"| {site} | {year} | {metrics['rank_ridge']['mae'] - metrics['so4gp_ridge']['mae']:.8f} | {metrics['generic_ridge']['mae'] - metrics['so4gp_ridge']['mae']:.8f} |"
            )
    interval = primary["bootstrap"]
    lines.extend(["", "## Uncertainty and limits", ""])
    if interval:
        lo, hi = interval["interval"]
        lines.append(
            f"The exploratory paired circular bootstrap interval for the primary MAE difference is [{lo:.8f}, {hi:.8f}] NDVI (3 months/block, 2,000 draws, seed 42, 2.5–97.5 percentiles). This conditions on already fitted models and an inspected holdout, assumes resampling is meaningful, and omits model-selection uncertainty. It is not a calibrated or confirmatory significance claim."
        )
    else:
        lines.append(
            "No bootstrap interval is reported because the primary evaluation series is not contiguous."
        )
    lines.extend(
        [
            "",
            "Secondary sites are descriptive checks with overlapping footprints and calendar gaps; they are not independent replications and are not pooled. Individual paired error differences and yearly metrics remain available. The training sample is small, the native selection is label-dependent, and one deterministic ridge and generic construction do not cover other algorithms. Negative effects and failures are retained without tuning the split, support, seed, feature cap or alpha.",
            "",
            "## Proposed next tests",
            "",
            "A future protocol could evaluate a newly reserved time period and operationally timestamped observations, repeat across genuinely distinct landscapes, and compare additional predeclared interaction constructors. Such work requires a new protocol; the present result does not change deployment or scientific acceptance.",
            "",
            "## Artifacts and reproducibility",
            "",
            "result.json contains fitted parameters, training CDF references, all feature identities, predictions, paired errors, metrics, source/runtime/code identities and timing/memory. predictions.csv preserves actual and unclipped/clipped forecasts and dates. artifact-manifest.json authenticates output bytes. scientific-digest.json hashes portable source identities and numerical results while excluding timing and local paths; reruns must use a new directory.",
            "",
            "Weather data by Open-Meteo.com, derived from ERA5 (CC BY 4.0); contains modified Copernicus Climate Change Service information 2021–2025. Neither the European Commission nor ECMWF is responsible for its use. Sentinel-2 observations retain the archive's existing provenance and rights. No provider endorsement is implied.",
            "",
        ]
    )
    return "\n".join(lines)


def peak_memory_kib():
    """Report optional process peaks in KiB; unavailable measurements remain null."""
    peaks = {"parent_peak_rss_kib": None, "native_child_peak_rss_kib": None}
    if resource is None:
        return peaks
    for key, usage_name in (
        ("parent_peak_rss_kib", "RUSAGE_SELF"),
        ("native_child_peak_rss_kib", "RUSAGE_CHILDREN"),
    ):
        try:
            maximum = resource.getrusage(getattr(resource, usage_name)).ru_maxrss
        except (AttributeError, OSError, ValueError):
            continue
        peaks[key] = maximum / 1024 if sys.platform == "darwin" else maximum
    return peaks


def run_experiment(ndvi_archive, era5_archive, protocol_dir, output_dir):
    start = time.monotonic()
    started_at_utc = datetime.now(UTC).isoformat()
    config, protocol_receipt = verify_protocol(protocol_dir)
    runtime = verify_runtime(config)
    code = _code_identity()
    out = create_output_dir(output_dir)
    try:
        for name in (
            *PROTOCOL_PINS,
            "protocol-freeze.json",
            *EXECUTION_PINS,
            "execution-freeze.json",
        ):
            (out / name).write_bytes((Path(protocol_dir) / name).read_bytes())
        with tempfile.TemporaryDirectory(prefix="ndvi-inputs-") as temporary:
            ndvi, era5, inputs = load_inputs(ndvi_archive, era5_archive, config, temporary)
        prepared = build_examples(ndvi, era5, config)
        eligibility = {}
        for site in SITES:
            eligibility[site] = {
                split: sum(e["site"] == site and e["split"] == split for e in prepared["examples"])
                for split in ("train", "evaluation")
            }
        if eligibility != config["expected_eligibility"]:
            raise ValueError(f"Eligibility differs from frozen inspected counts: {eligibility}")
        (out / "inputs.json").write_bytes(encode(inputs))
        (out / "examples.json").write_bytes(encode(prepared))
        _write_csv(out / "exclusions.csv", prepared["exclusions"])
        sites = {}
        for site in SITES:
            examples = [e for e in prepared["examples"] if e["site"] == site]
            training = [e for e in examples if e["split"] == "train"]
            matrix = np.asarray([e["drivers"] + [e["target"]] for e in training], dtype=float)
            native = native_mine(
                matrix, DRIVERS + ["target_next_ndvi"], config, out / f"native-{site}"
            )
            value = evaluate_site(examples, prepared["ndvi_history"][site], config, native)
            (out / f"site-{site}.json").write_bytes(encode(value))
            if value["status"] != "complete":
                raise RuntimeError(
                    f"Native support recount failure for {site}; retained all candidates"
                )
            sites[site] = value
        config_after, receipt_after = verify_protocol(protocol_dir)
        if receipt_after != protocol_receipt or config_after != config or _code_identity() != code:
            raise ValueError("Protocol or code changed during run")
        result = {
            "schema_version": config["schema_version"],
            "status": "complete",
            "config": config,
            "protocol": protocol_receipt,
            "runtime": runtime,
            "code": code,
            "inputs": inputs,
            "examples": prepared["examples"],
            "exclusions": prepared["exclusions"],
            "monthly_coverage": prepared["monthly_coverage"],
            "sites": sites,
            "execution": {
                "started_at_utc": started_at_utc,
                "elapsed_seconds": time.monotonic() - start,
                **peak_memory_kib(),
                "input_paths": {
                    "ndvi": str(Path(ndvi_archive).resolve()),
                    "era5": str(Path(era5_archive).resolve()),
                },
            },
        }
        digest = sha(encode(scientific_payload(result)))
        result["scientific_sha256"] = digest
        (out / "result.json").write_bytes(encode(result))
        (out / "scientific-digest.json").write_bytes(
            encode(
                {
                    "sha256": digest,
                    "method": "SHA-256 of canonical JSON from scientific_payload(result); excludes clocks, local paths, memory and native runtime; includes source/code/protocol/runtime identities and numerical outputs",
                }
            )
        )
        _write_csv(
            out / "predictions.csv", [p for value in sites.values() for p in value["predictions"]]
        )
        (out / "report.md").write_bytes(report_markdown(result).encode("utf-8"))
        manifest = {
            "schema_version": "ndvi-ablation-artifacts/1",
            "scientific_sha256": digest,
            "files": [
                {
                    "path": p.relative_to(out).as_posix(),
                    "bytes": p.stat().st_size,
                    "sha256": sha(p.read_bytes()),
                }
                for p in sorted(out.rglob("*"))
                if p.is_file()
            ],
        }
        (out / "artifact-manifest.json").write_bytes(encode(manifest))
        return result
    except BaseException as exc:
        (out / "failure.json").write_bytes(
            encode(
                {
                    "status": "failed",
                    "exception": type(exc).__name__,
                    "message": str(exc),
                    "elapsed_seconds": time.monotonic() - start,
                    "partial_artifacts_retained": True,
                }
            )
        )
        if not (out / "report.md").exists():
            failure_report = f"# NDVI ablation failed\n\n{type(exc).__name__}: {exc}\n\nPartial artifacts are retained. No successful predictive conclusion is reported.\n"
            (out / "report.md").write_bytes(failure_report.encode("utf-8"))
        if not (out / "artifact-manifest.json").exists():
            (out / "artifact-manifest.json").write_bytes(
                encode(
                    {
                        "status": "failed",
                        "files": [
                            {
                                "path": p.relative_to(out).as_posix(),
                                "sha256": sha(p.read_bytes()),
                                "bytes": p.stat().st_size,
                            }
                            for p in sorted(out.rglob("*"))
                            if p.is_file()
                        ],
                    }
                )
            )
        raise


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == "--native-worker":
        if len(argv) != 3:
            raise ValueError("Worker expects input JSON and output directory")
        _native_worker(argv[1], argv[2])
        return 0
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ndvi-archive", type=Path, required=True)
    parser.add_argument("--era5-archive", type=Path, required=True)
    parser.add_argument("--protocol-dir", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Must not already exist; partial failures remain",
    )
    args = parser.parse_args(argv)
    result = run_experiment(
        args.ndvi_archive, args.era5_archive, args.protocol_dir, args.output_dir
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "scientific_sha256": result["scientific_sha256"],
                "output_dir": str(args.output_dir),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
