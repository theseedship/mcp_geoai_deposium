"""Audit retained WP0 artifacts and replay counting; never execute a miner."""

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
SOURCES = {
    "benchmarks/copernicus_compare.py",
    "benchmarks/copernicus_compare_run.py",
    "benchmarks/copernicus_pair_reference.py",
}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _digest(data):
    return hashlib.sha256(data).hexdigest()


def _json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _semantic(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _csv_bytes(rows):
    flattened = []
    for row in rows:
        values = {**row.get("reference", {}), **{k: v for k, v in row.items() if k != "reference"}}
        flattened.append(
            {
                k: json.dumps(v, separators=(",", ":")) if isinstance(v, (list, dict)) else v
                for k, v in values.items()
            }
        )
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(
        stream, fieldnames=sorted({k for row in flattened for k in row}), lineterminator="\n"
    )
    writer.writeheader()
    writer.writerows(flattened)
    return stream.getvalue().encode("utf-8")


def _native_threshold_violations(zone, identities):
    return [
        {
            "pilot": zone,
            "items": row["items"],
            "native_raw_indexes": row["native_raw_indexes"],
            **{
                key: row["reference"][key]
                for key in ("hit_count", "unordered_pair_count", "native_support")
            },
        }
        for row in identities
        if row["native_raw_indexes"] and not row["strict"]
    ]


def verify_package(directory, expected_digest):
    """Require a trusted manifest digest, closed inventory and pinned local sources.

    The authenticated manifest pins exact LF-normalized replay source bytes;
    its commit remains recorded metadata, so Git history is unnecessary.
    Reading copied Parquets requires DuckDB; no miner import or execution occurs.
    """
    root = Path(directory).absolute()
    _require(
        root.is_dir() and not root.is_symlink() and not root.is_junction(),
        "Invalid package directory",
    )
    root = root.resolve()
    manifest_bytes = (root / "manifest.json").read_bytes()
    _require(_digest(manifest_bytes) == expected_digest, "Trusted manifest digest mismatch")
    manifest = json.loads(manifest_bytes)
    inventory = manifest["artifact_inventory"]
    indexed = {}
    for entry in inventory:
        name = entry["path"]
        relative = PurePosixPath(name)
        _require(
            isinstance(name, str)
            and str(relative) == name
            and not relative.is_absolute()
            and ".." not in relative.parts
            and "\\" not in name
            and name != "manifest.json",
            "Invalid inventory path",
        )
        _require(name.casefold() not in indexed, "Duplicate inventory path")
        indexed[name.casefold()] = name
        path = root / name
        _require(path.resolve().is_relative_to(root), "Escaping inventory path")
        _require(path.is_file() and not path.is_symlink(), "Missing or linked artifact: " + name)
        raw = path.read_bytes()
        _require(
            type(entry["bytes"]) is int
            and len(raw) == entry["bytes"]
            and _digest(raw) == entry["sha256"],
            "Inventoried artifact mismatch: " + name,
        )
    actual = set()
    for path in root.rglob("*"):
        _require(not path.is_symlink() and not path.is_junction(), "Linked artifact in package")
        if path.is_file() and path != root / "manifest.json":
            actual.add(path.relative_to(root).as_posix())
    _require(actual == set(indexed.values()), "Closed artifact inventory mismatch")
    config_path = root / "configuration.json"
    _require(
        _digest(config_path.read_bytes()) == manifest["configuration_sha256"],
        "Configuration digest mismatch",
    )
    config = _json(config_path)
    _require(set(manifest["sources"]) == SOURCES, "Expected exactly three measured source modules")
    commit = manifest["git_commit"]
    _require(
        isinstance(commit, str)
        and len(commit) == 40
        and all(c in "0123456789abcdef" for c in commit),
        "Invalid source commit",
    )
    for name, digest in manifest["sources"].items():
        _require(
            _digest((ROOT / name).read_bytes().replace(b"\r\n", b"\n")) == digest,
            "Replay source differs from measurement: " + name,
        )
    from benchmarks.copernicus_compare import compare_outputs
    from benchmarks.copernicus_compare_run import load_pilot, matrix_digest, validate_matrix

    specs = {spec["zone"]: spec for spec in config["pilots"]}
    pilots = {pilot["zone"]: pilot for pilot in manifest["pilots"]}
    runs = {(run["pilot"], run["method"]): run for run in manifest["runs"]}
    _require(
        len(specs) == len(config["pilots"])
        and len(pilots) == len(manifest["pilots"])
        and set(specs) == set(pilots),
        "Duplicate or missing pilot identities",
    )
    _require(
        len(runs) == len(manifest["runs"])
        and set(runs) == {(zone, method) for zone in specs for method in ("graank", "apriori")},
        "Duplicate or missing method runs",
    )
    verified, threshold_violations = 0, []
    for zone, spec in specs.items():
        _require(
            isinstance(zone, str)
            and zone
            and "/" not in zone
            and "\\" not in zone
            and zone not in (".", ".."),
            "Invalid pilot identity",
        )
        matrix_path = root / "matrices" / (zone + ".json")
        matrix = _json(matrix_path)
        validate_matrix(matrix)
        expected = load_pilot(root / "inputs" / (zone + ".parquet"), spec, config["columns"])
        _require(
            _semantic(matrix) == _semantic(expected),
            "Copied input/matrix/date order mismatch: " + zone,
        )
        _require(
            _semantic(_json(root / f"checkpoint-{zone}.json")) == _semantic(pilots[zone]),
            "Pilot checkpoint mismatch: " + zone,
        )
        payloads = {}
        for method in ("graank", "apriori"):
            run = runs[zone, method]
            _require(
                run["matrix_digest"] == matrix_digest(matrix)
                and run["matrix_sha256"] == _digest(matrix_path.read_bytes()),
                "Run matrix identity mismatch",
            )
            raw_path = root / "raw" / f"{zone}-{method}.json"
            if raw_path.exists():
                raw = _json(raw_path)
                _require(
                    raw["method"] == method
                    and raw["matrix_digest"] == matrix_digest(matrix)
                    and raw["status"] == run["status"],
                    "Raw method identity/status mismatch",
                )
                payloads[method] = raw["raw_payload"]
        if pilots[zone]["status"] != "complete":
            continue
        _require(
            all(runs[zone, method]["status"] == "ok" for method in ("graank", "apriori")),
            "Complete pilot has a non-successful run",
        )
        replay = compare_outputs(
            matrix["rows"], matrix["columns"], payloads["graank"], payloads["apriori"]
        )
        threshold_violations.extend(_native_threshold_violations(zone, replay["identities"]))
        reference = replay.pop("reference_rows")
        prefix = root / "derived" / zone
        _require(
            _semantic(replay) == _semantic(_json(Path(str(prefix) + "-comparison.json"))),
            "Derived comparison replay mismatch: " + zone,
        )
        _require(
            pilots[zone]["summary"] == replay["summary"]
            and pilots[zone]["rows"] == len(matrix["rows"]),
            "Pilot summary mismatch",
        )
        for suffix, rows in (("reference", reference), ("comparison", replay["identities"])):
            _require(
                Path(str(prefix) + f"-{suffix}.csv").read_bytes() == _csv_bytes(rows),
                "Derived CSV replay mismatch: " + zone,
            )
        verified += 1
    return {
        "audit_only": True,
        "manifest_sha256": expected_digest,
        "source_commit": commit,
        "verified_files": len(inventory),
        "verified_pilots": verified,
        "native_threshold_violations": threshold_violations,
        "pilot_statuses": {zone: pilot["status"] for zone, pilot in pilots.items()},
        "non_success_runs": [
            {"pilot": run["pilot"], "method": run["method"], "status": run["status"]}
            for run in manifest["runs"]
            if run["status"] != "ok"
        ],
        "scientific_reception": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--expected-manifest-sha256", required=True)
    args = parser.parse_args()
    try:
        result = verify_package(args.directory, args.expected_manifest_sha256)
    except (ValueError, KeyError, TypeError, OSError) as error:
        parser.error(str(error))
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
