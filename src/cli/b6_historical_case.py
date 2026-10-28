"""Prepare and check a historical B6 case offline; never score or activate it."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from src.b6_saved_contracts import MAX_SAVED_ENVELOPE_BYTES
from src.patterns.historical_case import (
    export_historical_case,
    preflight_historical_forecast,
    verify_historical_case,
)
from src.utils.logging import setup_logging


def main(argv: list[str] | None = None) -> int:
    setup_logging()
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    export = commands.add_parser("export", help="Create a verified historical preparation package")
    export.add_argument("--bundle-dir", type=Path, required=True)
    export.add_argument("--zones-geojson", type=Path, required=True)
    export.add_argument("--output-dir", type=Path, required=True)
    verify = commands.add_parser(
        "verify", help="Verify the pinned package and unchanged native rules"
    )
    preflight = commands.add_parser(
        "preflight", help="Check forecast compatibility without granting adoption or scoring"
    )
    for command in (verify, preflight):
        command.add_argument("directory", type=Path)
        command.add_argument("--expected-manifest-sha256", required=True)
    preflight.add_argument("--forecast-json", type=Path, required=True)
    preflight.add_argument("--expected-forecast-sha256", required=True)
    preflight.add_argument(
        "--forecast-profile",
        choices=("full-shortlist", "rule15-temperature"),
        default="full-shortlist",
        help="Compatibility requirements only; preserves the complete native library",
    )
    args = parser.parse_args(argv)
    try:
        if args.command == "export":
            result = export_historical_case(args.bundle_dir, args.zones_geojson, args.output_dir)
        elif args.command == "verify":
            result = verify_historical_case(args.directory, args.expected_manifest_sha256)
        else:
            with args.forecast_json.open("rb") as stream:
                raw = stream.read(MAX_SAVED_ENVELOPE_BYTES + 1)
            if len(raw) > MAX_SAVED_ENVELOPE_BYTES:
                raise ValueError("historical forecast exceeds 128 KiB")
            result = preflight_historical_forecast(
                args.directory,
                args.expected_manifest_sha256,
                raw,
                args.expected_forecast_sha256,
                forecast_profile=args.forecast_profile,
            )
        print(json.dumps(result, ensure_ascii=False, allow_nan=False, sort_keys=True))
        return 0
    except (ValueError, OSError) as exc:
        print(f"Historical B6 preparation refused: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
