"""Execute the agreed bounded WP0 comparison; no provider access or application hooks."""

import argparse
import csv
import hashlib
import importlib.metadata as metadata
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def validate_matrix(matrix):
    columns, rows = matrix["columns"], matrix["rows"]
    if not 1 <= len(columns) <= 16 or len(set(columns)) != len(columns):
        raise ValueError("Invalid feature identities")
    if any(not isinstance(c, str) or not c for c in columns):
        raise ValueError("Invalid feature names")
    if not 2 <= len(rows) <= 64:
        raise ValueError("Expected 2 to 64 unchanged observations")
    for row in rows:
        if len(row) != len(columns):
            raise ValueError("Ragged matrix")
        for value in row:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError("Non-numeric or missing input")
            if not math.isfinite(value):
                raise ValueError("Nonfinite input")


def matrix_digest(matrix):
    data = {"columns": matrix["columns"], "rows": matrix["rows"]}
    return sha256(json.dumps(data, separators=(",", ":"), allow_nan=False).encode())


def run_bounded(command, timeout, cwd=None, env=None):
    if isinstance(timeout, bool) or not math.isfinite(timeout) or not 0 < timeout <= 60:
        raise ValueError("Timeout must be positive and at most 60 seconds")
    started = time.perf_counter()
    try:
        process = subprocess.run(
            command,
            timeout=timeout,
            cwd=cwd,
            env=env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        result = {
            "status": "ok" if process.returncode == 0 else "error",
            "returncode": process.returncode,
            "stdout": process.stdout,
            "stderr": process.stderr,
        }
    except subprocess.TimeoutExpired as error:

        def decoded(value):
            return (
                value.decode("utf-8", errors="replace") if isinstance(value, bytes) else value or ""
            )

        result = {
            "status": "timeout",
            "returncode": None,
            "stdout": decoded(error.stdout),
            "stderr": decoded(error.stderr),
        }
    result["elapsed_seconds"] = time.perf_counter() - started
    return result


def installed_sources(package):
    distribution = metadata.distribution(package)
    selected = []
    for entry in distribution.files or []:
        name = str(entry).replace("\\", "/")
        if not name.endswith(".py"):
            continue
        if package == "so4gp" and not name.startswith("so4gp/"):
            continue
        if package == "mlxtend" and not name.startswith("mlxtend/frequent_patterns/"):
            continue
        raw = Path(distribution.locate_file(entry)).read_bytes()
        selected.append(
            {"path": name, "sha256": sha256(raw), "lf_sha256": sha256(raw.replace(b"\r\n", b"\n"))}
        )
    return sorted(selected, key=lambda row: row["path"])


def worker(method, matrix_path, output):
    matrix = json.loads(Path(matrix_path).read_text(encoding="utf-8"))
    validate_matrix(matrix)
    expected = matrix_digest(matrix)
    import numpy as np
    import pandas as pd

    from benchmarks.copernicus_pair_reference import canonical_items, encode_ordered_pairs

    package = "so4gp" if method == "graank" else "mlxtend"
    required = "1.1.2" if method == "graank" else "0.25.0"
    if metadata.version(package) != required:
        raise ValueError("Wrong comparator version")
    if method == "graank":
        import torch
        from so4gp.algorithms import GRAANK

        if torch.version.cuda is not None:
            raise ValueError("Execution profile must have no CUDA runtime")
    else:
        from mlxtend.frequent_patterns import apriori

    sources = installed_sources(package)
    started = time.perf_counter()
    frame = pd.DataFrame(matrix["rows"], columns=matrix["columns"], dtype="float64")
    actual = {"columns": list(frame.columns), "rows": frame.to_numpy(dtype="float64").tolist()}
    if matrix_digest(actual) != expected:
        raise ValueError("DataFrame changed the recorded matrix")
    canonical = []
    extra = {}
    if method == "graank":
        miner = GRAANK(frame, min_sup=0.5, eq=False, device="cpu")
        engine = miner.mining_engine
        observed = {
            "columns": matrix["columns"],
            "rows": np.asarray(engine.data, dtype="float64").tolist(),
        }
        if engine.row_count != len(matrix["rows"]) or list(engine.attr_cols) != list(
            range(len(frame.columns))
        ):
            raise ValueError("Native reader changed the population or feature selection")
        if matrix_digest(observed) != expected:
            raise ValueError("Native reader changed matrix values/order")
        native_text = miner.discover(
            search_type="apriori", max_iteration=None, compute_descriptors=True, save_results=False
        )
        payload = json.loads(native_text)
        from benchmarks.copernicus_compare import parse_native_patterns

        records, errors = parse_native_patterns(payload, matrix["columns"])
        canonical = [
            [name + sign for name, sign in canonical_items(row["items"])] for row in records
        ]
        extra = {
            "native_raw_json": native_text,
            "native_parser_errors": errors,
            "native_row_count": engine.row_count,
            "native_attribute_columns": [int(i) for i in engine.attr_cols],
            "native_matrix_digest": matrix_digest(observed),
        }
        status = "reported_error" if "Error" in payload else "parse_error" if errors else "ok"
    else:
        encoding = encode_ordered_pairs(actual["rows"], actual["columns"])
        labels = [name + sign for name in matrix["columns"] for sign in ("+", "-")]
        label_index = {label: i for i, label in enumerate(labels)}
        one_hot = np.zeros((len(encoding.transactions), len(labels)), dtype=bool)
        for i, transaction in enumerate(encoding.transactions):
            for name, sign in transaction:
                one_hot[i, label_index[name + sign]] = True
        transactions = pd.DataFrame(one_hot, columns=labels)
        frequent = apriori(
            transactions, min_support=0.25, use_colnames=True, max_len=7, low_memory=False, n_jobs=1
        )
        payload = {
            "itemsets": [
                {"items": sorted(row.itemsets), "support": float(row.support)}
                for row in frequent.itertuples(index=False)
            ]
        }
        for row in payload["itemsets"]:
            if len(row["items"]) >= 2:
                items = [(item[:-1], item[-1]) for item in row["items"]]
                canonical.append([name + sign for name, sign in canonical_items(items)])
        extra = {
            "transaction_count": len(encoding.transactions),
            "empty_transaction_count": sum(not row for row in encoding.transactions),
            "raw_itemset_serialization": "DataFrame order; frozenset items listed lexically",
        }
        status = "ok"
    elapsed = time.perf_counter() - started
    write_json(
        output,
        {
            "method": method,
            "status": status,
            "package": package,
            "version": required,
            "matrix_digest": expected,
            "method_seconds": elapsed,
            "raw_payload": payload,
            "canonical_output_identities": canonical,
            "sources": sources,
            **extra,
        },
    )


def write_rows_csv(path, rows):
    flattened = []
    for row in rows:
        result = {**row.get("reference", {}), **{k: v for k, v in row.items() if k != "reference"}}
        flattened.append(
            {
                key: json.dumps(value, separators=(",", ":"))
                if isinstance(value, (list, dict))
                else value
                for key, value in result.items()
            }
        )
    keys = sorted({key for row in flattened for key in row})
    with Path(path).open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys, lineterminator="\n")
        writer.writeheader()
        writer.writerows(flattened)


def load_pilot(path, specification, columns):
    raw = path.read_bytes()
    if sha256(raw) != specification["sha256"]:
        raise ValueError("Frozen input digest mismatch: " + path.name)
    import duckdb

    selection = ", ".join('"' + c + '"' for c in ["date", *columns])
    with duckdb.connect(
        ":memory:",
        config={
            "autoload_known_extensions": False,
            "autoinstall_known_extensions": False,
            "preserve_insertion_order": True,
        },
    ) as connection:
        records = connection.execute(
            f"SELECT {selection} FROM read_parquet(?)", [str(path)]
        ).fetchall()
    dates = [row[0].isoformat() if hasattr(row[0], "isoformat") else str(row[0]) for row in records]
    rows = [
        [
            float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else v
            for v in row[1:]
        ]
        for row in records
    ]
    matrix = {
        "columns": columns,
        "rows": rows,
        "dates": dates,
        "source_sha256": specification["sha256"],
        "source_file": path.name,
    }
    validate_matrix(matrix)
    if len(rows) != specification["rows"]:
        raise ValueError("Input population differs from agreed count")
    return matrix


def run(config_path, input_dir, output_dir):
    config_bytes = Path(config_path).read_bytes()
    config = json.loads(config_bytes)
    fixed = {
        "timeout_seconds": 60,
        "native_min_support": 0.5,
        "apriori_min_support": 0.25,
        "minimum_itemset_size": 2,
        "maximum_itemset_size": 7,
        "encode_equal": False,
        "device": "cpu",
    }
    if any(config.get(key) != value for key, value in fixed.items()) or len(config["columns"]) != 7:
        raise ValueError("Execution must use the agreed fixed configuration")
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    for name in ("inputs", "matrices", "raw", "derived", "logs"):
        (output / name).mkdir()
    (output / "configuration.json").write_bytes(config_bytes)
    from benchmarks.copernicus_compare import compare_outputs

    environment = {
        **os.environ,
        "PYTHONHASHSEED": "0",
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
    }
    manifest = {
        "schema": "geoai-wp0-executed-comparison/v1",
        "configuration_sha256": sha256(config_bytes),
        "platform": platform.platform(),
        "python": sys.version,
        "python_executable": sys.executable,
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "versions": {d.metadata["Name"]: d.version for d in metadata.distributions()},
        "thread_environment": {
            k: environment[k]
            for k in (
                "PYTHONHASHSEED",
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
            )
        },
        "timing_boundary": {
            "process_seconds": "child wall time including interpreter/imports, JSON read and output write; hard 60 second limit",
            "method_seconds": "float64 matrix ingestion, native construction/mining or transaction encoding/Apriori, raw result parsing/serialization and inverse canonicalization; excludes imports, file read/write and independent parent recount",
        },
        "sources": {
            path: sha256((ROOT / path).read_bytes().replace(b"\r\n", b"\n"))
            for path in (
                "benchmarks/copernicus_compare.py",
                "benchmarks/copernicus_compare_run.py",
                "benchmarks/copernicus_pair_reference.py",
            )
        },
        "runs": [],
        "pilots": [],
    }
    for spec in config["pilots"]:
        zone = spec["zone"]
        original = Path(input_dir) / (zone + ".parquet")
        matrix = load_pilot(original, spec, config["columns"])
        shutil.copyfile(original, output / "inputs" / original.name)
        matrix_path = output / "matrices" / (zone + ".json")
        write_json(matrix_path, matrix)
        actual = {}
        statuses = {}
        for method in ("graank", "apriori"):
            raw_path = output / "raw" / f"{zone}-{method}.json"
            command = [
                sys.executable,
                "-m",
                "benchmarks.copernicus_compare_run",
                "worker",
                "--method",
                method,
                "--matrix",
                str(matrix_path),
                "--output",
                str(raw_path),
            ]
            outcome = run_bounded(command, timeout=60, cwd=ROOT, env=environment)
            for stream in ("stdout", "stderr"):
                (output / "logs" / f"{zone}-{method}.{stream}.txt").write_text(
                    outcome.pop(stream), encoding="utf-8", newline="\n"
                )
            outcome.update(
                {
                    "pilot": zone,
                    "method": method,
                    "command": command,
                    "matrix_sha256": sha256(matrix_path.read_bytes()),
                    "matrix_digest": matrix_digest(matrix),
                }
            )
            if outcome["status"] == "ok":
                try:
                    result = json.loads(raw_path.read_text(encoding="utf-8"))
                    outcome["status"] = result["status"]
                    outcome["method_seconds"] = result["method_seconds"]
                    actual[method] = result["raw_payload"]
                except Exception as error:
                    outcome["status"] = "invalid_result"
                    outcome["error"] = str(error)
            statuses[method] = outcome["status"]
            manifest["runs"].append(outcome)
            print(
                f"{zone} {method}: {outcome['status']} ({outcome['elapsed_seconds']:.3f}s)",
                flush=True,
            )
        if statuses == {"graank": "ok", "apriori": "ok"}:
            comparison = compare_outputs(
                matrix["rows"], config["columns"], actual["graank"], actual["apriori"]
            )
            reference = comparison.pop("reference_rows")
            write_rows_csv(output / "derived" / (zone + "-reference.csv"), reference)
            write_rows_csv(
                output / "derived" / (zone + "-comparison.csv"), comparison["identities"]
            )
            write_json(output / "derived" / (zone + "-comparison.json"), comparison)
            manifest["pilots"].append(
                {
                    "zone": zone,
                    "status": comparison["status"],
                    "rows": len(matrix["rows"]),
                    "summary": comparison["summary"],
                }
            )
        else:
            manifest["pilots"].append(
                {"zone": zone, "status": "incomplete", "method_statuses": statuses}
            )
        write_json(output / f"checkpoint-{zone}.json", manifest["pilots"][-1])
    manifest["artifact_inventory"] = [
        {
            "path": path.relative_to(output).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256(path.read_bytes()),
        }
        for path in sorted(output.rglob("*"))
        if path.is_file()
    ]
    manifest["scientific_reception"] = False
    manifest["speed_ranking_claimed"] = False
    write_json(output / "manifest.json", manifest)
    return 0 if all(p["status"] == "complete" for p in manifest["pilots"]) else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    execution = commands.add_parser("run")
    execution.add_argument("--config", type=Path, required=True)
    execution.add_argument("--input-dir", type=Path, required=True)
    execution.add_argument("--output-dir", type=Path, required=True)
    child = commands.add_parser("worker")
    child.add_argument("--method", choices=("graank", "apriori"), required=True)
    child.add_argument("--matrix", type=Path, required=True)
    child.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.action == "worker":
        try:
            worker(args.method, args.matrix, args.output)
        except Exception:
            traceback.print_exc()
            return 1
        return 0
    return run(args.config, args.input_dir, args.output_dir)


if __name__ == "__main__":
    raise SystemExit(main())
