#!/usr/bin/env python3
"""Generate leakage-safe persistence and WMO P50 forecast baselines."""

from __future__ import annotations

import argparse
import glob
import json
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import duckdb

from src.forecast_backtest import BacktestRow, run_backtest
from src.utils.logging import setup_logging


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="forecast-baseline-panel")
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--dates", required=True, type=Path)
    parser.add_argument("--normals-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--holdout-start", required=True)
    parser.add_argument("--holdout-end", required=True)
    return parser


def _load_reference(
    path: Path, dates: set[str]
) -> tuple[list[BacktestRow], dict[tuple[str, str, str], float]]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    selected = [row for row in rows if row["valid_date"] in dates]
    reference = [
        BacktestRow(
            row["zone_id"],
            date.fromisoformat(row["valid_date"]),
            date.fromisoformat(row["valid_date"]),
            0,
            row["variable"],
            float(row["value"]),
            "reference",
        )
        for row in selected
    ]
    history = {
        (row["zone_id"], row["valid_date"], row["variable"]): float(row["value"]) for row in rows
    }
    return reference, history


def _load_normals(normals_dir: Path) -> dict[tuple[str, str, int], float]:
    connection = duckdb.connect()
    normals: dict[tuple[str, str, int], float] = {}
    for path in sorted(glob.glob(str(normals_dir / "*.parquet"))):
        for zone, variable, month, p50 in connection.execute(
            "select zone_id, variable, month, p50 from read_parquet(?)", [path]
        ).fetchall():
            normals[(zone, variable, month)] = float(p50)
    return normals


def run(args: argparse.Namespace) -> dict[str, dict[str, Any]]:
    dates = set(json.loads(args.dates.read_text(encoding="utf-8")))
    start = date.fromisoformat(args.holdout_start)
    end = date.fromisoformat(args.holdout_end)
    reference, history = _load_reference(args.reference, dates)
    variables = sorted({row.variable for row in reference})
    zones = sorted({row.zone_id for row in reference})
    normals = _load_normals(args.normals_dir)
    persistence: list[BacktestRow] = []
    climatology: list[BacktestRow] = []
    for valid_text in sorted(dates):
        valid = date.fromisoformat(valid_text)
        if valid <= start:
            continue
        origin = valid - timedelta(days=1)
        for zone in zones:
            for variable in variables:
                if (zone, valid_text, variable) not in history:
                    continue
                previous = history.get((zone, origin.isoformat(), variable))
                if previous is not None:
                    persistence.append(
                        BacktestRow(zone, origin, valid, 1, variable, previous, "persistence")
                    )
                normal = normals.get((zone, variable, valid.month))
                if normal is not None:
                    climatology.append(
                        BacktestRow(zone, origin, valid, 1, variable, normal, "climatology_wmo_p50")
                    )
    result = {
        "persistence": run_backtest(
            persistence,
            reference,
            training_end=date(2020, 12, 31),
            holdout_start=start,
            holdout_end=end,
        ),
        "climatology": run_backtest(
            climatology,
            reference,
            training_end=date(2020, 12, 31),
            holdout_start=start,
            holdout_end=end,
        ),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "baselines.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def main() -> int:
    # Install the credential-redaction processor (#148). Only src/server.py called this, so
    # every `log.error(..., error=str(e))` these handlers reach went to stderr raw when a
    # script drove them in-process -- structlog's DEFAULT chain has no redactor (#161 review).
    setup_logging()
    result = run(build_parser().parse_args())
    print(
        json.dumps(
            {
                name: {"status": value["status"], "prediction_rows": value["prediction_rows"]}
                for name, value in result.items()
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
