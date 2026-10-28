"""Offline CLI for the leakage-safe forecast benchmark (FIERCE B5a)."""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

from src.cli.forecast_ensemble_archive import load_validated_archive
from src.forecast_backtest import (
    DEFAULT_PRECIPITATION_EVENT_THRESHOLD_MM,
    DEFAULT_RELIABILITY_BINS,
    load_rows,
    load_rows_with_digest,
    parse_rows,
    run_backtest,
)
from src.utils.logging import setup_logging


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="forecast_backtest")
    predictions = parser.add_mutually_exclusive_group(required=True)
    predictions.add_argument("--predictions", help="JSON array or JSONL forecast archive")
    predictions.add_argument(
        "--predictions-manifest",
        help="validated forecast-ensemble-forward-manifest/v1 local path or S3 URI",
    )
    parser.add_argument("--reference", required=True, help="JSON array or JSONL ERA5 reference")
    parser.add_argument(
        "--provider",
        default=None,
        help="provider label for raw --predictions only (default: archived)",
    )
    parser.add_argument("--training-end", default="2020-12-31")
    parser.add_argument("--holdout-start", default="2021-01-01")
    parser.add_argument("--holdout-end", default="2025-12-31")
    parser.add_argument(
        "--precipitation-event-threshold-mm",
        type=float,
        default=DEFAULT_PRECIPITATION_EVENT_THRESHOLD_MM,
        help="Observed/predicted daily precipitation strictly above this value counts as rain.",
    )
    parser.add_argument(
        "--reliability-bin-count",
        type=int,
        default=DEFAULT_RELIABILITY_BINS,
        help="Number of equal-width bins used for precipitation reliability metrics (2..100).",
    )
    parser.add_argument("-o", "--output", help="write result JSON instead of stdout")
    return parser


def main(argv: list[str] | None = None) -> int:
    # Install the credential-redaction processor (#148) before anything can log: structlog's
    # DEFAULT chain has no redactor (#202).
    setup_logging()
    parser = build_parser()
    args = parser.parse_args(argv)
    archive = None
    if args.predictions_manifest:
        if args.provider is not None:
            parser.error("--provider cannot override a validated predictions manifest")
        archive = load_validated_archive(args.predictions_manifest)
        predictions = parse_rows(
            archive.rows,
            kind="prediction",
            provider=archive.provider,
        )
    else:
        predictions = load_rows(
            args.predictions,
            kind="prediction",
            provider=args.provider or "archived",
        )
    reference_path = Path(args.reference).resolve()
    reference_rows, reference_sha256 = load_rows_with_digest(
        reference_path,
        kind="reference",
    )
    result = run_backtest(
        predictions,
        reference_rows,
        training_end=date.fromisoformat(args.training_end),
        holdout_start=date.fromisoformat(args.holdout_start),
        holdout_end=date.fromisoformat(args.holdout_end),
        precipitation_event_threshold_mm=args.precipitation_event_threshold_mm,
        reliability_bin_count=args.reliability_bin_count,
    )
    if archive is not None:
        result["prediction_archive"] = {
            "schema_version": archive.manifest["schema_version"],
            "run_id": archive.manifest["run_id"],
            "manifest": archive.manifest_location,
            "manifest_sha256": archive.manifest_sha256,
            "provider": archive.provider,
            "model": archive.model,
            "rows": len(archive.rows),
        }
    result["reference_input"] = {
        "path": str(reference_path),
        "sha256": reference_sha256,
        "provenance_status": "caller_supplied_unverified",
    }
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        Path(args.output).write_text(encoded, encoding="utf-8")
    else:
        print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
