"""Score one verified D-A library against one saved forecast: B6 level 2, offline (#30).

The forecast is a saved ``forecast_query`` result, not a live call: a live forecast changes
between runs, so a result computed from it could never be reproduced. The library is read
only through ``verify_library_handoff`` with a manifest digest supplied by the operator.

Publication is create-only, like ``forecast_skill_matrix``: ``--output`` is written once and
never replaced. A ``no_conclusion`` result is a valid result (exit 0); a refusal (bad digest,
unknown scope, unusable forecast, existing output) publishes nothing and exits 1.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.b6_explanatory import load_verified_library, score_level2
from src.b6_forecast import parse_saved_forecast
from src.cli.forecast_skill_matrix import _publish_create_only
from src.contracts import B6Params
from src.utils.code_ref import code_ref
from src.utils.logging import setup_logging

_DEFAULTS = B6Params()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="b6-explanatory", description=__doc__)
    parser.add_argument("--package-dir", required=True, type=Path)
    parser.add_argument(
        "--expected-manifest-sha256",
        required=True,
        help="manifest digest from a trusted channel, never read from the package",
    )
    parser.add_argument("--zone-id", required=True)
    parser.add_argument("--algorithm", required=True)
    parser.add_argument(
        "--forecast-json",
        required=True,
        type=Path,
        help="a saved forecast_query result: {zone_id, forecast: [...], metadata: {...}}",
    )
    parser.add_argument("--min-folds", type=int, default=_DEFAULTS.min_folds)
    parser.add_argument("--min-support", type=float, default=_DEFAULTS.min_support)
    parser.add_argument("--min-stability", type=float, default=_DEFAULTS.min_stability)
    parser.add_argument("--halflife-days", type=float, default=_DEFAULTS.halflife_days)
    parser.add_argument("--max-patterns", type=int, default=_DEFAULTS.max_patterns)
    parser.add_argument(
        "--computed-at",
        help="ISO timestamp recorded in the trace (default: now, UTC); fix it to compare runs",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="create-only: refused if anything already exists at this path",
    )
    return parser


def _forecast(path: Path, zone_id: str) -> tuple[list[dict[str, Any]], dict[str, Any], float]:
    """``(rows, identity, effective_horizon_days)`` from the exact bytes read."""
    return parse_saved_forecast(path.read_bytes(), zone_id, filename=path.name, label=str(path))


def run(args: argparse.Namespace) -> tuple[dict[str, Any], str]:
    output: Path = args.output
    if os.path.lexists(output):
        raise ValueError(f"{output} already exists; a published result is never replaced")
    if output.resolve() == args.forecast_json.resolve():
        raise ValueError("--output and --forecast-json name the same file")
    if output.resolve().is_relative_to(args.package_dir.resolve()):
        raise ValueError("--output must be outside the verified package")
    params = B6Params(
        min_folds=args.min_folds,
        min_support=args.min_support,
        min_stability=args.min_stability,
        halflife_days=args.halflife_days,
        max_patterns=args.max_patterns,
    )
    library, identity = load_verified_library(
        args.package_dir, args.expected_manifest_sha256, args.zone_id, args.algorithm
    )
    rows, forecast_identity, horizon = _forecast(args.forecast_json, args.zone_id)
    result = score_level2(
        library,
        library_identity=identity,
        forecast_rows=rows,
        forecast_identity=forecast_identity,
        zone_id=args.zone_id,
        effective_horizon_days=horizon,
        params=params,
        code_ref=code_ref(),
        computed_at=args.computed_at or datetime.now(UTC).isoformat(timespec="seconds"),
    )
    value = result.model_dump(mode="json")
    data = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")
    try:
        _publish_create_only(output, data)
    except FileExistsError as exc:  # a concurrent writer won after the early check
        raise ValueError(f"{output} already exists; a published result is never replaced") from exc
    return value, hashlib.sha256(data).hexdigest()


def main(argv: list[str] | None = None) -> int:
    setup_logging()
    args = build_parser().parse_args(argv)
    try:
        value, digest = run(args)
    except (ValueError, OSError) as exc:  # pydantic's ValidationError is a ValueError
        print(json.dumps({"status": "refused", "error": str(exc)}, sort_keys=True))
        return 1
    print(
        json.dumps(
            {
                "status": value["status"],
                "reason": value["reason"],
                "output": str(args.output),
                "output_sha256": digest,
                "n_patterns_read": value["n_patterns_read"],
                "n_eligible": value["n_eligible"],
                "exclusions": value["exclusions"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
