"""Operator CLI for bounded solar portfolio collection and export."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from src.solar_portfolio import (
    ALLOWED_VARIABLES,
    SolarPortfolioError,
    export_solar_portfolio,
    load_solar_sites,
    run_solar_portfolio,
)
from src.utils.logging import setup_logging

DEFAULT_VARIABLES = (
    "shortwave_radiation",
    "diffuse_radiation",
    "direct_normal_irradiance",
)


def _variables(value: str) -> tuple[str, ...]:
    variables = tuple(part.strip() for part in value.split(","))
    if not variables or any(not item for item in variables):
        raise argparse.ArgumentTypeError("variables must be a comma-separated list")
    unknown = set(variables) - ALLOWED_VARIABLES
    if unknown:
        raise argparse.ArgumentTypeError(f"unknown variables: {', '.join(sorted(unknown))}")
    if len(variables) != len(set(variables)):
        raise argparse.ArgumentTypeError("variables must not contain duplicates")
    return variables


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="solar-portfolio")
    commands = parser.add_subparsers(dest="command", required=True)

    validate = commands.add_parser(
        "validate", help="validate a site/scenario CSV without network calls"
    )
    validate.add_argument("--sites", required=True, type=Path)
    validate.add_argument("--variables", type=_variables, default=DEFAULT_VARIABLES)

    run = commands.add_parser("run", help="fetch solar series and resume a checksummed batch")
    run.add_argument("--sites", required=True, type=Path)
    run.add_argument("--source", required=True, choices=("forecast", "era5", "satellite"))
    run.add_argument("--start-date", required=True)
    run.add_argument("--end-date", required=True)
    run.add_argument("--variables", type=_variables, default=DEFAULT_VARIABLES)
    run.add_argument("--model")
    run.add_argument("--temporal-resolution", choices=("hourly", "native"))
    run.add_argument("--output-dir", required=True, type=Path)
    run.add_argument("--concurrency", type=int, choices=range(1, 9), default=1)
    run.add_argument("--max-attempts", type=int, choices=range(1, 4), default=3)
    run.add_argument("--max-age-hours", type=float)

    export = commands.add_parser("export", help="export a completed batch as long-form CSV")
    export.add_argument("--output-dir", required=True, type=Path)
    export.add_argument("--destination", type=Path)
    return parser


def _dispatch(args: argparse.Namespace) -> dict[str, object]:
    if args.command == "validate":
        sites = load_solar_sites(args.sites, variables=args.variables)
        return {
            "valid": True,
            "sites": len({site.site_id for site in sites}),
            "scenarios": len(sites),
        }
    if args.command == "run":
        return run_solar_portfolio(
            sites_path=args.sites,
            source=args.source,
            start_date=args.start_date,
            end_date=args.end_date,
            variables=args.variables,
            model=args.model,
            output_dir=args.output_dir,
            concurrency=args.concurrency,
            max_attempts=args.max_attempts,
            max_age_hours=args.max_age_hours,
            temporal_resolution=args.temporal_resolution,
        )
    destination = export_solar_portfolio(args.output_dir, args.destination)
    return {"exported": True, "destination": str(destination)}


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    setup_logging()
    try:
        summary = _dispatch(args)
    except SolarPortfolioError as exc:
        print(json.dumps({"error": str(exc)}, sort_keys=True), file=sys.stderr)
        return 2
    except Exception as exc:
        # Adapter errors are intentionally reduced to a stable exception class.
        print(json.dumps({"error": type(exc).__name__}, sort_keys=True), file=sys.stderr)
        return 1
    print(json.dumps(summary, sort_keys=True))
    if args.command == "run" and summary.get("failed", 0):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
