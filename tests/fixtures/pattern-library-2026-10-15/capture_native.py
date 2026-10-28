"""Reproduce the bounded D-A native capture; never rerun the held-out scorer.

Run from the repository root after preparing the three supplied pilot inputs.
The output directory must be new. Export its export.json with the D-A CLI.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from src.patterns.library_adapter import library_from_mining_result  # noqa: E402
from src.patterns.offline_bundle import MAX_JSON_BYTES, _strict_json  # noqa: E402
from src.patterns.parquet_bridge import build_temporal_frame, load_dataframe  # noqa: E402
from src.patterns.so4gp_adapter import mine_gradual_patterns, mine_temporal_patterns  # noqa: E402

FEATURES = [
    "ndvi_mean",
    "temperature_2m",
    "precipitation",
    "humidity",
    "evapotranspiration",
    "wind_speed",
    "solar_radiation",
]
MAX_PILOT_BYTES = 64 * 1024 * 1024
MAX_PILOTS = 3


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
    )


def authenticated_snapshot(source, expected_sha256, destination, *, max_bytes):
    """Hash and retain one bounded read; never reopen an authenticated source."""
    if not isinstance(expected_sha256, str) or not re.fullmatch(
        r"[a-fA-F0-9]{64}", expected_sha256
    ):
        raise ValueError(f"Invalid expected SHA-256: {source.name}")
    if source.is_symlink():
        raise ValueError(f"Capture source symlink refused: {source.name}")
    with source.open("rb") as stream:
        data = stream.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise ValueError(f"Capture source exceeds byte limit: {source.name}")
    actual = hashlib.sha256(data).hexdigest()
    if actual != expected_sha256.lower():
        raise ValueError(f"Capture source SHA-256 mismatch: {source.name}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(data)
    return actual


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilots", type=Path, default=ROOT / "research-data/pilots")
    parser.add_argument(
        "--study", type=Path, default=ROOT / "docs/research/heldout-validation-2026-10-13/evidence"
    )
    parser.add_argument("--expected-study-manifest-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if version("so4gp") != "1.1.2":
        raise ValueError("This capture requires the locked so4gp 1.1.2 CPU environment")
    if args.output_dir.exists() or args.output_dir.is_symlink():
        raise ValueError("Choose a new capture output directory")
    output = args.output_dir.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".geoai-native-capture-", dir=output.parent) as scratch:
        stage = Path(scratch) / "capture"
        stage.mkdir(mode=0o700)
        capture = capture_snapshots(args, stage)
        if output.exists():
            raise ValueError("Choose a new capture output directory")
        stage.rename(output)
    print(json.dumps(capture))


def capture_snapshots(args, output):
    """Authenticate the whole input set before loading/mining any of it."""
    identities = {}
    manifest_name = "validation/manifest.json"
    identities[manifest_name] = authenticated_snapshot(
        args.study / "manifest.json",
        args.expected_study_manifest_sha256,
        output / manifest_name,
        max_bytes=MAX_JSON_BYTES,
    )
    study = _strict_json((output / manifest_name).read_bytes())
    if (
        study.get("schema_version") != "geoai-heldout-research/v1"
        or study.get("status") != "complete"
    ):
        raise ValueError("Unsupported or incomplete study manifest")
    identities["inputs/provenance.json"] = authenticated_snapshot(
        args.pilots / "provenance.json",
        study["pilot_provenance"]["sha256"],
        output / "inputs/provenance.json",
        max_bytes=MAX_JSON_BYTES,
    )
    items = study["pilot_inputs"]
    if not isinstance(items, list) or not 1 <= len(items) <= MAX_PILOTS:
        raise ValueError("Capture requires one to three pilot inputs")
    zones = [item["zone_id"] for item in items]
    if len(set(zones)) != len(zones) or any(
        not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", zone) for zone in zones
    ):
        raise ValueError("Unsafe or duplicate pilot zone_id")
    for item in items:
        name = f"inputs/{item['zone_id']}.parquet"
        identities[name] = authenticated_snapshot(
            args.pilots / f"{item['zone_id']}.parquet",
            item["sha256"],
            output / name,
            max_bytes=MAX_PILOT_BYTES,
        )
    evidence_names = ("manifest.json", "spatial/result.json", "temporal/result.json")
    for name in evidence_names[1:]:
        identities["validation/" + name] = authenticated_snapshot(
            args.study / name,
            study["output_hashes"][name],
            output / "validation" / name,
            max_bytes=MAX_JSON_BYTES,
        )

    (output / "native").mkdir()
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    config = {
        "schema_version": "geoai-pattern-library-export/v1",
        "library_version": "2026.10.15-native-pilots.1",
        "producer": {
            "name": "so4gp",
            "version": version("so4gp"),
            "commit": commit,
            "command": subprocess.list2cmdline([sys.executable, *sys.argv]),
        },
        "areas": zones,
        "inputs": [],
        "libraries": [],
        "evidence": [],
    }
    capture = {
        "features": FEATURES,
        "min_support": 0.5,
        "min_rep": 0.5,
        "encode_equal": False,
        "python": sys.version,
        "capture_script_sha256": digest(Path(__file__)),
        "authenticated_snapshots": identities,
        "libraries": [],
    }
    for item in items:
        zone = item["zone_id"]
        name = f"inputs/{zone}.parquet"
        frame = load_dataframe(str(output / name), order_by="date", db_path=":memory:")
        config["inputs"].append(
            {
                "input_id": zone,
                "zone_id": zone,
                "path": name,
                "source_uri": f"geoai-heldout-pilot-inputs/v1:{zone}",
                "sha256": identities[name],
                "period_start": item["min_date"],
                "period_end": item["max_date"],
            }
        )
        for algorithm in ("graank", "tgrad"):
            temporal, _, target = build_temporal_frame(
                frame[["date", *FEATURES]], time_column="date", target_column="ndvi_mean"
            )
            started = time.monotonic()
            previous = Path.cwd()
            with tempfile.TemporaryDirectory(prefix="geoai-native-da-") as scratch:
                os.chdir(scratch)
                try:
                    result = (
                        mine_gradual_patterns(frame[FEATURES], min_support=0.5, encode_equal=False)
                        if algorithm == "graank"
                        else mine_temporal_patterns(
                            temporal,
                            target_col=target,
                            min_support=0.5,
                            min_rep=0.5,
                            encode_equal=False,
                        )
                    )
                finally:
                    os.chdir(previous)
            library = library_from_mining_result(
                result,
                zone_id=zone,
                algorithm=algorithm,
                generated_at=datetime.now(UTC),
                strict=True,
            )
            name = f"native/{zone}-{algorithm}.json"
            write_json(output / name, library.model_dump(mode="json"))
            config["libraries"].append(
                {"path": name, "sha256": digest(output / name), "input_ids": [zone]}
            )
            capture["libraries"].append(
                {
                    "zone_id": zone,
                    "algorithm": algorithm,
                    "patterns": len(library.patterns),
                    "elapsed_seconds": time.monotonic() - started,
                }
            )
    for name in evidence_names:
        path = "validation/" + name
        config["evidence"].append(
            {
                "path": path,
                "source_uri": f"git:{commit}:docs/research/heldout-validation-2026-10-13/evidence/{name}",
                "sha256": identities[path],
                "relationship": "reference_only",
            }
        )
    write_json(output / "export.json", config)
    write_json(output / "capture.json", capture)
    return capture


if __name__ == "__main__":
    main()
