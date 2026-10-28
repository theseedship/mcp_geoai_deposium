"""CLI for so4gp gradual-pattern mining (FIERCE DoD lot O5, issue #23 — WP4 `fierce1-mining`).

A thin wrapper over the existing stack — the DuckDB ingestion bridge + the so4gp adapter +
the frozen exchange schema — so a pattern library can be produced from the command line
(batch / offline), not only via the `discover_patterns` MCP tool. Output is the validated
`PatternLibrary` as JSON.

Examples:
    uv run --extra patterns python -m src.cli.mine_patterns \\
        --source scripts/sample/tabular_data.parquet --algorithm graank \\
        --zone-id montpellier -o patterns.json

    uv run --extra patterns python -m src.cli.mine_patterns \\
        --source s3://deposium/climate/unified/montpellier.parquet --algorithm tgrad \\
        --zone-id montpellier --order-by date --target-column ndvi_mean -o patterns.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.patterns.library_adapter import library_from_mining_result
from src.utils.logging import setup_logging


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="mine_patterns",
        description="Mine gradual patterns (so4gp) and emit the frozen exchange-schema library as JSON.",
    )
    p.add_argument("--source", required=True, help="parquet/csv path or s3:// URI")
    p.add_argument("--zone-id", required=True, help="producing zone (science-plane scope)")
    p.add_argument("--algorithm", choices=["graank", "tgrad"], default="graank")
    p.add_argument("--kind", default="parquet", choices=["parquet", "geoparquet", "csv", "json"])
    p.add_argument("--min-support", type=float, default=0.5)
    p.add_argument("--min-rep", type=float, default=0.5, help="tgrad: min representativity")
    p.add_argument("--order-by", help="time column to sort rows by (required for tgrad)")
    p.add_argument("--target-column", help="tgrad: target column NAME (resolved to index)")
    p.add_argument("--eq", action="store_true", help="treat equal consecutive values as gradual")
    p.add_argument("--limit", type=int, help="optional row cap")
    p.add_argument(
        "--db-path", default=":memory:", help="DuckDB file (default :memory: — reads are ephemeral)"
    )
    p.add_argument("-o", "--output", help="write JSON here (default: stdout)")
    return p


def run(args: argparse.Namespace) -> int:
    # Fail fast on bad args BEFORE any I/O (and before importing the patterns extra).
    if args.algorithm == "tgrad" and (not args.target_column or not args.order_by):
        print("ERROR: --algorithm tgrad requires --target-column and --order-by", file=sys.stderr)
        return 2
    if not 0.0 <= args.min_support <= 1.0:
        print(f"ERROR: --min-support must be in [0, 1]; got {args.min_support}", file=sys.stderr)
        return 2
    if args.algorithm == "tgrad" and not 0.0 < args.min_rep <= 1.0:
        print(f"ERROR: --min-rep must be in (0, 1]; got {args.min_rep}", file=sys.stderr)
        return 2

    # Imports deferred so --help / arg-validation work without the optional `patterns` extra.
    from src.patterns import so4gp_adapter
    from src.patterns.parquet_bridge import build_temporal_frame, load_dataframe

    # Same confinement as the MCP tools (#214): so4gp litters the cwd with GRAANK_* / TGrad_*.
    from src.tools.patterns import _mine_in_tempdir

    try:
        df = load_dataframe(
            args.source,
            kind=args.kind,
            order_by=args.order_by,
            limit=args.limit,
            db_path=args.db_path,
        )
        if args.algorithm == "tgrad":
            # Curate exactly like the MCP tool: date→str at index 0, numeric attrs only,
            # target resolved to its index (raises ValueError on any structural problem).
            frame, _attr_cols, target_idx = build_temporal_frame(
                df, time_column=args.order_by, target_column=args.target_column, drop_columns=None
            )
            # Only the mining call runs in the temp cwd: --source / --db-path / -o resolve
            # against the caller's cwd (load above, write below), so they stay outside it.
            result = _mine_in_tempdir(
                so4gp_adapter.mine_temporal_patterns,
                frame,
                target_col=target_idx,
                min_support=args.min_support,
                min_rep=args.min_rep,
                encode_equal=args.eq,
            )
        else:
            # GRAANK is contemporaneous and needs >= 2 numeric columns and >= 2 rows — drop
            # date / geometry / string columns (row order, set by --order-by, is preserved).
            numeric = df.select_dtypes(include="number").drop(
                columns=[args.order_by], errors="ignore"
            )
            if numeric.shape[1] < 2 or len(numeric) < 2:
                print(
                    f"ERROR: graank needs >= 2 numeric columns and >= 2 rows; got shape {numeric.shape}",
                    file=sys.stderr,
                )
                return 2
            result = _mine_in_tempdir(
                so4gp_adapter.mine_gradual_patterns,
                numeric,
                min_support=args.min_support,
                encode_equal=args.eq,
                algorithm="graank",
            )
    except so4gp_adapter.So4gpNotInstalled as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    except ValueError as exc:  # bad structure / columns (incl. build_temporal_frame guards)
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:  # load / mining failure — concise message, no raw traceback
        print(f"ERROR: mining failed: {exc}", file=sys.stderr)
        return 1

    library = library_from_mining_result(result, zone_id=args.zone_id, algorithm=args.algorithm)
    payload = library.model_dump_json(indent=2)

    if args.output:
        Path(args.output).write_text(payload + "\n", encoding="utf-8")
        print(f"Wrote {len(library.patterns)} patterns ({args.algorithm}) to {args.output}")
    else:
        print(payload)
    return 0


def main(argv: list[str] | None = None) -> int:
    # Install the credential-redaction processor (#148) before anything can log: structlog's
    # DEFAULT chain has no redactor (#202).
    setup_logging()
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
