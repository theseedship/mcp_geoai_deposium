"""Run the resumable B5a archive collection and offline benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import date
from pathlib import Path
from typing import Any

from src.cli.forecast_archive_panel import _provider_model, run_panel
from src.forecast_backtest import (
    DEFAULT_PRECIPITATION_EVENT_THRESHOLD_MM,
    DEFAULT_RELIABILITY_BINS,
    BacktestValidationError,
    load_rows,
    probability_evaluation_metadata,
    run_backtest,
)
from src.tools.climate import _load_zones, _scientific_zone_ids
from src.utils.logging import setup_logging


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="forecast-benchmark-panel")
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--end-date", required=True)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--archive-dir",
        type=Path,
        help="Use an existing forecast-archive-panel directory instead of collecting archives.",
    )
    parser.add_argument("--zone-id", nargs="+")
    parser.add_argument("--provider-model", action="append", type=_provider_model)
    parser.add_argument("--horizon-days", type=int, nargs="+", default=[1])
    parser.add_argument("--training-end", default="2020-12-31")
    parser.add_argument("--holdout-start", default="2021-01-01")
    parser.add_argument("--holdout-end", default="2025-12-31")
    parser.add_argument(
        "--precipitation-event-threshold-mm",
        type=float,
        default=DEFAULT_PRECIPITATION_EVENT_THRESHOLD_MM,
        help="Daily precipitation strictly above this value counts as the event.",
    )
    parser.add_argument(
        "--reliability-bin-count",
        type=int,
        default=DEFAULT_RELIABILITY_BINS,
        help="Equal-width precipitation reliability bins (2..100).",
    )
    parser.add_argument("--preflight", action="store_true")
    return parser


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _archive_files(output_dir: Path) -> list[Path]:
    return sorted(path for path in output_dir.glob("*.json") if path.name != "manifest.json")


def _validate_panel_args(args: argparse.Namespace) -> None:
    if any(horizon < 1 or horizon > 7 for horizon in args.horizon_days):
        raise ValueError("horizon-days must be in 1..7")
    zones = _load_zones()
    if args.zone_id and (unknown := sorted(set(args.zone_id) - set(zones))):
        raise ValueError(f"unknown zone ids: {unknown}")


def _requested_provider_models(args: argparse.Namespace) -> set[tuple[str, str]]:
    return set(
        args.provider_model
        or [
            ("gfs", "gfs_seamless"),
            ("aifs", "ecmwf_aifs025_single"),
        ]
    )


def _validate_existing_manifest(
    manifest: dict[str, Any], *, args: argparse.Namespace, archive_dir: Path
) -> None:
    expected_zones = set(args.zone_id or _scientific_zone_ids(_load_zones()))
    if manifest.get("start_date") != args.start_date or manifest.get("end_date") != args.end_date:
        raise ValueError("archive manifest date window does not match benchmark arguments")
    if set(manifest.get("zones", [])) != expected_zones:
        raise ValueError("archive manifest zones do not match benchmark arguments")
    expected_models = _requested_provider_models(args)
    expected_horizons = set(args.horizon_days)
    for entry in manifest.get("entries", []):
        if entry.get("status") not in {"available", "resumed"}:
            continue
        coverage = entry.get("coverage")
        if not isinstance(coverage, dict):
            raise ValueError(f"archive entry has no coverage metadata: {entry.get('output')}")
        identity = (
            coverage.get("zone_id"),
            coverage.get("provider"),
            coverage.get("model"),
            coverage.get("horizon_days"),
        )
        if identity[0] not in expected_zones or identity[1:3] not in expected_models:
            raise ValueError(f"archive entry is outside requested panel: {entry.get('output')}")
        if identity[3] not in expected_horizons:
            raise ValueError(
                f"archive entry horizon is outside requested panel: {entry.get('output')}"
            )
        output = Path(entry.get("output", ""))
        if not output.is_absolute():
            output = archive_dir / output
        if not output.is_file():
            raise ValueError(f"archive artifact does not exist: {output}")


def _report(result: dict[str, Any], *, archive_manifest: dict[str, Any]) -> str:
    lines = [
        "# B5a Forecast Benchmark",
        "",
        "> ERA5 is an independent reanalysis reference, not station ground truth.",
        "",
        f"- Training end: `{result['training_end']}`",
        f"- Holdout: `{result['holdout']['start']}..{result['holdout']['end']}`",
        f"- Status: `{result['status']}`",
        f"- Prediction rows: `{result['prediction_rows']}`",
        f"- Reference rows: `{result['reference_rows']}`",
        f"- Archive manifest: `{_digest(Path(archive_manifest['path']))}`",
        "",
        "## Metrics",
        "",
        "| Provider | Variable | Horizon | N | MAE | RMSE | Bias | Status |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for metric in result["metrics"]:
        lines.append(
            f"| {metric['provider']} | {metric['variable']} | {metric['horizon_days']} "
            f"| {metric['n']} | {metric['mae']:.4f} | {metric['rmse']:.4f} "
            f"| {metric['bias']:.4f} | {metric.get('status', 'complete')} |"
        )
    lines += [
        "",
        "## Explicit gaps",
        "",
        "- Persistence and WMO climatology baselines are not included until their "
        "versioned inputs are supplied.",
        "- Providers with incomplete archive coverage are excluded from scoring and "
        "remain `archive_required` in the archive manifest.",
    ]
    invalid = result.get("invalid_artifacts", [])
    if invalid:
        lines += ["", "### Invalid artifacts", ""]
        lines.extend(f"- `{item.get('artifact')}`: {item.get('error')}" for item in invalid)
    return "\n".join(lines) + "\n"


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    if date.fromisoformat(args.start_date) > date.fromisoformat(args.end_date):
        raise ValueError("start-date must be before end-date")
    if not args.reference.is_file():
        raise ValueError(f"reference does not exist: {args.reference}")
    probability_evaluation = probability_evaluation_metadata(
        args.precipitation_event_threshold_mm,
        args.reliability_bin_count,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    archive_dir = args.archive_dir or args.output_dir / "archives"
    archive_args = argparse.Namespace(
        start_date=args.start_date,
        end_date=args.end_date,
        zone_id=args.zone_id,
        provider_model=args.provider_model,
        horizon_days=args.horizon_days,
        output_dir=archive_dir,
    )
    _validate_panel_args(archive_args)
    if args.preflight:
        return {
            "schema_version": "forecast-benchmark-panel/v1",
            "status": "preflight_ok",
            "reference": str(args.reference),
            "output_dir": str(args.output_dir),
            "probability_evaluation": probability_evaluation,
        }
    if args.archive_dir is not None:
        archive_manifest_path = archive_dir / "manifest.json"
        if not archive_manifest_path.is_file():
            raise ValueError(f"archive manifest does not exist: {archive_manifest_path}")
        archive_manifest = json.loads(archive_manifest_path.read_text(encoding="utf-8"))
        _validate_existing_manifest(archive_manifest, args=args, archive_dir=archive_dir)
    else:
        archive_manifest = run_panel(archive_args)
        archive_manifest_path = archive_dir / "manifest.json"
    archive_manifest["path"] = str(archive_manifest_path)
    reference = load_rows(args.reference, kind="reference")
    predictions = []
    scored: list[dict[str, Any]] = []
    current_paths = {
        Path(entry["output"])
        for entry in archive_manifest.get("entries", [])
        if entry.get("output") and entry.get("status") in {"available", "resumed"}
    }
    for path in sorted(current_paths):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            coverage = payload["coverage"]
            if not isinstance(coverage, dict):
                raise ValueError("coverage must be an object")
            if coverage.get("status") != "available":
                continue
            provider = f"{coverage['provider']}:{coverage['model']}"
            predictions.extend(load_rows(path, kind="prediction", provider=provider))
        except (
            OSError,
            KeyError,
            TypeError,
            ValueError,
            json.JSONDecodeError,
            BacktestValidationError,
        ) as exc:
            scored.append({"artifact": str(path), "status": "invalid_artifact", "error": str(exc)})
    if predictions:
        result = run_backtest(
            predictions,
            reference,
            training_end=date.fromisoformat(args.training_end),
            holdout_start=date.fromisoformat(args.holdout_start),
            holdout_end=date.fromisoformat(args.holdout_end),
            precipitation_event_threshold_mm=args.precipitation_event_threshold_mm,
            reliability_bin_count=args.reliability_bin_count,
        )
    else:
        result = {
            "schema_version": "forecast-backtest/v2",
            "status": "archive_required",
            "training_end": args.training_end,
            "holdout": {"start": args.holdout_start, "end": args.holdout_end},
            "prediction_rows": 0,
            "reference_rows": len(reference),
            "probability_evaluation": probability_evaluation,
            "metrics": [],
            "provenance": {"prediction_providers": []},
        }
    result["archive_manifest"] = archive_manifest
    result["invalid_artifacts"] = scored
    if scored:
        result["status"] = "partial_artifacts" if predictions else "invalid_artifacts"
    result_path = args.output_dir / "benchmark.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (args.output_dir / "benchmark.md").write_text(
        _report(result, archive_manifest=archive_manifest), encoding="utf-8"
    )
    return result


def main(argv: list[str] | None = None) -> int:
    # Install the credential-redaction processor (#148) before anything can log: structlog's
    # DEFAULT chain has no redactor (#202).
    setup_logging()
    args = build_parser().parse_args(argv)
    print(json.dumps(run_benchmark(args), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
