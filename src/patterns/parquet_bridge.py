"""Load tabular data from RustFS/local into a DataFrame for so4gp.

so4gp's GRAANK only accepts a pandas DataFrame or a CSV path — it cannot
read Parquet, GeoParquet, or S3 directly. DuckDB is the universal reader
already used by ``spatial_query``: it reads Parquet / GeoParquet / CSV /
JSON from RustFS (S3) and hands back a pandas DataFrame.

This module is the ingestion bridge: RustFS object -> DuckDB -> numeric
DataFrame ordered by time -> so4gp adapter.

Requires pandas (ships with the ``patterns`` extra); imported at call time
so the module loads in the base runtime.
"""

from __future__ import annotations

import re

from src.config import settings
from src.utils.duckdb_conn import duckdb_session

# DuckDB readers per source kind. GeoParquet is read as Parquet — the
# geometry column is dropped for mining (so4gp needs numeric columns only).
_READERS = {
    "parquet": "read_parquet",
    "geoparquet": "read_parquet",
    "csv": "read_csv_auto",
    "json": "read_json_auto",
}

_VALID_IDENTIFIER_RE = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")


def _quote_identifier(name: str, *, role: str) -> str:
    """Validate (alnum+underscore) and double-quote a SQL identifier.

    The regex guarantees no embedded quote can break out; double-quoting then
    lets a column whose name is a DuckDB reserved word (order, end, group, …)
    parse as an identifier instead of raising a ParserException.
    """
    if not _VALID_IDENTIFIER_RE.match(name):
        raise ValueError(f"Invalid {role} identifier: {name!r}")
    return f'"{name}"'


def build_select_query(
    kind: str,
    *,
    columns: list[str] | None = None,
    order_by: str | None = None,
    limit: int | None = None,
) -> str:
    """Build the parameterized DuckDB SELECT (``$source`` left as a parameter).

    Pure — no DB connection — so identifier quoting/validation is unit-testable.
    """
    if kind not in _READERS:
        raise ValueError(f"Unknown source kind '{kind}'. Supported: {sorted(_READERS)}")
    reader = _READERS[kind]

    if columns:
        select_sql = ", ".join(_quote_identifier(c, role="column") for c in columns)
    else:
        select_sql = "*"

    sql = f"SELECT {select_sql} FROM {reader}($source)"

    if order_by is not None:
        sql += f" ORDER BY {_quote_identifier(order_by, role='order_by')}"

    if limit is not None:
        safe_limit = max(1, min(int(limit), settings.SPATIAL_QUERY_MAX_LIMIT))
        sql += f" LIMIT {safe_limit}"

    return sql


def load_dataframe(
    source: str,
    *,
    kind: str = "parquet",
    columns: list[str] | None = None,
    order_by: str | None = None,
    drop_columns: list[str] | None = None,
    limit: int | None = None,
    db_path: str | None = None,
):
    """Read a tabular source from RustFS/local into a pandas DataFrame.

    Args:
        source: S3 URI (s3://deposium/climate/unified/{zone}.parquet) or a
            local path. Passed as a DuckDB parameter, never interpolated.
        kind: one of _READERS ('parquet', 'geoparquet', 'csv', 'json').
        columns: explicit numeric columns to keep (validated as identifiers).
            None keeps all columns the source provides.
        order_by: column to sort rows by (e.g. 'date') — gradual patterns are
            interpreted along row order, so time ordering matters. Validated
            as an identifier.
        drop_columns: columns to exclude (e.g. 'geometry' from GeoParquet).
        limit: optional row cap (clamped to SPATIAL_QUERY_MAX_LIMIT).

    Returns:
        pandas.DataFrame.

    Raises:
        ValueError: on an unknown kind or an invalid column/order identifier.
    """
    sql = build_select_query(kind, columns=columns, order_by=order_by, limit=limit)

    with duckdb_session(db_path, load_httpfs=source.startswith("s3://")) as conn:
        df = conn.execute(sql, {"source": source}).df()

    if drop_columns:
        df = df.drop(columns=[c for c in drop_columns if c in df.columns])

    return df


def build_temporal_frame(df, *, time_column, target_column, drop_columns=None):
    """Assemble the DataFrame so4gp's TGrad needs: the date-time column (as a STRING —
    a datetime64 dtype raises "No date-time datasets found") first, then the numeric
    attributes; resolve the target column to its positional index.

    Returns (frame, attr_cols, target_idx). Raises ValueError with a readable message on
    any structural problem. Shared by the ``discover_patterns`` MCP tool and the
    ``mine_patterns`` CLI — lives here (the mining bridge) so neither depends on the other.
    """
    import pandas as pd

    if time_column is None:
        raise ValueError("temporal mining requires order_by (the date-time column).")
    if target_column is None:
        raise ValueError("temporal mining requires target_column.")

    work = df.drop(columns=[c for c in (drop_columns or []) if c in df.columns])

    if time_column not in work.columns:
        raise ValueError(f"order_by/time column {time_column!r} not found in source.")
    if target_column not in work.columns:
        raise ValueError(f"target_column {target_column!r} not found in source.")

    # Numeric attributes, minus the time column (which may itself be numeric, e.g. an
    # epoch — it is the time axis, never a mined attribute).
    attr_cols = [c for c in work.select_dtypes(include=["number"]).columns if c != time_column]
    if target_column not in attr_cols:
        raise ValueError(
            f"target_column {target_column!r} must be a numeric attribute (not the time column)."
        )
    if len(attr_cols) < 2:
        raise ValueError(
            "temporal mining needs the target plus at least one other numeric attribute."
        )

    time_series = work[time_column].astype(str).rename(time_column)
    frame = pd.concat([time_series, work[attr_cols]], axis=1)
    target_idx = list(frame.columns).index(target_column)
    return frame, attr_cols, target_idx
