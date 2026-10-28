"""Gradual pattern mining MCP tools (so4gp) — Paliers 1 & 2.

Prefix-agnostic by design (ADR 0001 §2/§6): the tool reads a `source` parquet/
geoparquet/csv/geojson from RustFS via DuckDB, mines gradual patterns with
so4gp, and writes the result parquet back to RustFS. The caller supplies the
scope — `{tenant_id}/{space_id}/…` (app) or `climate/{kind}/…` (science) —
so the same tool serves both data planes.

- Palier 1 (`algorithm="graank"`): plain GRAANK, contemporaneous co-monotonic
  patterns only.
- Palier 2 (`algorithm="tgrad"`): T-GRAANK temporal gradual patterns — emits a
  `lag_days` between a chosen `target_column` and the other attributes.

NO calibrated `confidence` is emitted in either mode — that requires
5-fold/cross-zone validation (Paliers 3/4). See ADR §5/§7.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import tempfile
import threading
import time
from datetime import date
from string import Formatter

import structlog

from src.config import settings
from src.contracts import (
    DiscoverPatternsCvInput,
    DiscoverPatternsInput,
    check_input,
    output_schema_for,
)
from src.patterns.cross_validation import (
    cross_validate_mined_patterns,
    validated_library_to_dataframe,
)
from src.patterns.parquet_bridge import build_temporal_frame, load_dataframe
from src.patterns.render import render_gradual_item
from src.patterns.so4gp_adapter import (
    SUPPORTED_ALGORITHMS,
    SUPPORTED_TEMPORAL_ALGORITHMS,
    So4gpNotInstalled,
    mine_gradual_patterns,
    mine_temporal_patterns,
)
from src.utils.duckdb_conn import duckdb_session

log = structlog.get_logger(__name__)

# Safe S3 key: path segments of alnum/._- joined by '/', no '..', no quotes.
_SAFE_KEY_RE = re.compile(r"^[A-Za-z0-9._\-/]+$")

_EXT_KIND = {
    ".parquet": "parquet",
    ".geoparquet": "geoparquet",
    ".csv": "csv",
    ".tsv": "csv",
    # .json = a FLAT array of records (read_json_auto). A GeoJSON
    # FeatureCollection nests data under features[].properties and is NOT
    # tabular here — flattening it is a later palier, so .geojson is not
    # advertised. Spatial inputs go through GeoParquet (tabular) instead.
    ".json": "json",
}

# so4gp's discover() writes GRAANK_<ts>.csv/.txt into the cwd with no flag to
# disable (ADR §8). Serialize mining and run it inside a temp dir so the server
# cwd is never littered. chdir is process-global, hence the lock.
_mine_lock = threading.Lock()


def _infer_kind(uri: str, explicit: str | None) -> str:
    if explicit:
        if explicit not in ("parquet", "geoparquet", "csv", "json"):
            raise ValueError(f"Invalid source_kind: {explicit!r}")
        return explicit
    lower = uri.lower()
    for ext, kind in _EXT_KIND.items():
        if lower.endswith(ext):
            return kind
    return "parquet"


def _validate_key(key: str, *, role: str) -> str:
    if not key or ".." in key or not _SAFE_KEY_RE.match(key):
        raise ValueError(f"Invalid {role}: {key!r}")
    return key


def _mine_in_tempdir(fn, *args, **kwargs):
    """Run a so4gp mining call ``fn(*args, **kwargs)`` inside a throwaway cwd so its
    ``GRAANK_*`` / ``TGrad_*`` output never lands in the server's working directory.

    Both ``discover()`` (GRAANK) and ``discover_tgp()`` (TGrad) litter the cwd with
    timestamped files and expose no flag to disable it (ADR §8). chdir is process-global,
    so the call is serialized under ``_mine_lock`` and run in a per-call temp dir. The
    mining function and its DataFrame are forwarded verbatim — the DataFrame is ``fn``'s
    first positional arg, so it must travel through ``*args`` (a kwargs-only helper would
    drop it and raise a missing-argument TypeError before so4gp ever runs)."""
    with _mine_lock:
        prev = os.getcwd()
        with tempfile.TemporaryDirectory(prefix="so4gp_") as td:
            # Invariant: while the lock is held the cwd is this temp dir, so any worker
            # thread under it must use ABSOLUTE paths — no cwd-relative I/O (none today).
            os.chdir(td)
            try:
                return fn(*args, **kwargs)
            finally:
                os.chdir(prev)


# Fixed output schema so a zero-pattern result still writes a typed, readable
# parquet (not a column-less empty frame). The temporal columns (is_temporal,
# target, max_lag_days) are null for contemporaneous GRAANK patterns.
_PATTERN_COLUMNS = [
    "pattern_id",
    "pattern",
    "support",
    "n_items",
    "is_temporal",
    "target",
    "max_lag_days",
    "gradual_items",
    "metrics",
]


def _patterns_to_dataframe(result: dict):
    """Flatten normalized patterns into an algorithm-agnostic table.

    Always carries _PATTERN_COLUMNS, even for an empty pattern set, so a
    legitimate "0 patterns" result persists as a valid empty parquet.
    """
    import pandas as pd

    rows = []
    for i, pat in enumerate(result["patterns"]):
        items = pat["gradual_items"]
        is_temporal = bool(pat.get("is_temporal"))
        target = next((gi["attribute"] for gi in items if gi.get("role") == "target"), None)
        lags = [
            abs(gi["lag_days"])
            for gi in items
            if gi.get("role") == "lagged" and gi.get("lag_days") is not None
        ]
        rows.append(
            {
                "pattern_id": i,
                "pattern": ", ".join(render_gradual_item(gi) for gi in items),
                "support": pat["support"],
                "n_items": len(items),
                "is_temporal": is_temporal,
                "target": target,
                "max_lag_days": max(lags) if lags else None,
                "gradual_items": json.dumps(items),
                "metrics": json.dumps(pat["metrics"]),
            }
        )
    return pd.DataFrame(rows, columns=_PATTERN_COLUMNS)


def _write_patterns_parquet(result: dict, output_uri: str) -> None:
    """Write patterns via DuckDB's native parquet writer (no pyarrow) using the
    RustFS S3 config. output_uri is built from validated key + trusted bucket.

    Runs on an ephemeral ``:memory:`` connection — it only registers a frame and
    ``COPY``s it to an external parquet path, so it never needs the persistent
    ``settings.DUCKDB_PATH`` database file."""
    patterns_df = _patterns_to_dataframe(result)
    with duckdb_session(":memory:", load_httpfs=output_uri.startswith("s3://")) as conn:
        conn.register("patterns_out", patterns_df)
        conn.execute(f"COPY patterns_out TO '{output_uri}' (FORMAT PARQUET)")


def _resolve_bucket_uri(value: str, *, bucket: str, role: str) -> str:
    """Resolve a raw bucket key or same-bucket s3:// URI into an s3:// URI."""
    if value.startswith("s3://"):
        prefix = f"s3://{bucket}/"
        if not value.startswith(prefix):
            raise ValueError(f"{role} must be within the configured bucket ({prefix}…).")
        key = value[len(prefix) :]
        return f"s3://{bucket}/{_validate_key(key, role=role)}"
    if "://" in value:
        raise ValueError(f"Unsupported {role} URI scheme: {value!r}")
    return f"s3://{bucket}/{_validate_key(value, role=role)}"


def _as_source_list(value: str | list[str]) -> list[str]:
    if isinstance(value, str):
        return [value]
    return value


def _format_source_template(
    template: str,
    *,
    zone_id: str,
    window_label: str | None,
) -> str:
    try:
        return template.format(zone_id=zone_id, window_label=window_label)
    except KeyError as exc:
        raise ValueError(f"Unknown source_template placeholder: {exc}") from exc


def _source_template_fields(template: str) -> set[str]:
    return {
        field_name.split(".", 1)[0].split("[", 1)[0]
        for _, field_name, _, _ in Formatter().parse(template)
        if field_name
    }


def _resolve_cv_sources(
    *,
    zone_ids: list[str],
    sources: dict[str, str | list[str]] | None,
    source_template: str | None,
    window_labels: list[str] | None,
    bucket: str,
) -> dict[str, list[str]]:
    """Resolve O3 per-zone source lists, bucket-confined like discover_patterns."""
    resolved: dict[str, list[str]] = {}
    if sources is not None:
        for zone_id in zone_ids:
            resolved[zone_id] = [
                _resolve_bucket_uri(item, bucket=bucket, role=f"source for zone {zone_id}")
                for item in _as_source_list(sources[zone_id])
            ]
        return resolved

    if source_template is None:
        raise ValueError("provide sources or source_template")
    template_fields = _source_template_fields(source_template)
    if "zone_id" not in template_fields:
        raise ValueError("source_template must include {zone_id} unless sources are explicit")
    if "window_label" in template_fields and not window_labels:
        raise ValueError("source_template uses {window_label}; provide window_labels")
    labels = window_labels or [None]
    for zone_id in zone_ids:
        zone_sources = [
            _format_source_template(source_template, zone_id=zone_id, window_label=label)
            for label in labels
        ]
        resolved[zone_id] = [
            _resolve_bucket_uri(item, bucket=bucket, role=f"source for zone {zone_id}")
            for item in zone_sources
        ]
    return resolved


def _load_cv_zone_frame(
    source_uris: list[str],
    *,
    source_kind: str | None,
    columns: list[str] | None,
    order_by: str | None,
    drop_columns: list[str] | None,
    limit: int | None,
):
    frames = [
        load_dataframe(
            uri,
            kind=_infer_kind(uri, source_kind),
            columns=columns,
            order_by=order_by,
            drop_columns=drop_columns,
            limit=limit,
        )
        for uri in source_uris
    ]
    if not frames:
        raise ValueError("no sources provided")
    if len(frames) == 1:
        frame = frames[0]
    else:
        import pandas as pd

        frame = pd.concat(frames, ignore_index=True)
        if order_by and order_by in frame.columns:
            frame = frame.sort_values(order_by, kind="mergesort").reset_index(drop=True)
    if limit is not None and len(frame) > limit:
        frame = frame.head(limit).reset_index(drop=True)
    return frame


def _mine_cv_frame(
    df,
    *,
    algorithm: str,
    min_support: float,
    target_column: str | None,
    min_rep: float,
    order_by: str | None,
    drop_columns: list[str] | None,
    encode_equal: bool,
) -> tuple[dict, int, list[str]]:
    algo_key = algorithm.lower()
    is_temporal = algo_key in SUPPORTED_TEMPORAL_ALGORITHMS
    if is_temporal:
        frame, mined_columns, target_idx = build_temporal_frame(
            df,
            time_column=order_by,
            target_column=target_column,
            drop_columns=drop_columns,
        )
        if len(frame) < 2:
            raise ValueError("Need at least 2 rows to mine temporal patterns.")
        result = _mine_in_tempdir(
            mine_temporal_patterns,
            frame,
            min_support=min_support,
            min_rep=min_rep,
            encode_equal=encode_equal,
            algorithm=algo_key,
            target_col=target_idx,
        )
        return result, int(len(frame)), mined_columns

    numeric_df = df.select_dtypes(include=["number"])
    if order_by:
        numeric_df = numeric_df.drop(columns=[order_by], errors="ignore")
    if numeric_df.shape[1] < 2 or len(numeric_df) < 2:
        raise ValueError("Need at least 2 numeric columns and 2 rows to mine gradual patterns.")
    result = _mine_in_tempdir(
        mine_gradual_patterns,
        numeric_df,
        min_support=min_support,
        encode_equal=encode_equal,
        algorithm=algo_key,
    )
    return result, int(len(numeric_df)), list(numeric_df.columns)


def _write_validated_patterns_parquet(library, output_uri: str) -> None:
    validated_df = validated_library_to_dataframe(library)
    with duckdb_session(":memory:", load_httpfs=output_uri.startswith("s3://")) as conn:
        conn.register("validated_patterns_out", validated_df)
        conn.execute(f"COPY validated_patterns_out TO '{output_uri}' (FORMAT PARQUET)")


def _parse_date_or_none(value: str | None) -> date | None:
    return date.fromisoformat(value) if value is not None else None


def register_pattern_tools(mcp):
    """Register gradual pattern mining tools with the MCP server."""

    @mcp.tool(output_schema=output_schema_for("discover_patterns"))
    async def discover_patterns(
        source: str = "auto",
        tenant_id: str | None = None,
        space_id: str | None = None,
        output_key: str | None = None,
        algorithm: str = "graank",
        min_support: float = 0.5,
        target_column: str | None = None,
        min_rep: float = 0.5,
        columns: list[str] | None = None,
        order_by: str | None = None,
        drop_columns: list[str] | None = None,
        source_kind: str | None = None,
        encode_equal: bool = False,
        limit: int | None = None,
        write_output: bool = True,
    ) -> dict:
        """Mine gradual (co-monotonic) patterns from a dataset with so4gp.

        A gradual pattern is a set of attributes that rise/fall together across
        rows, e.g. {precipitation↑, ndvi↑} with a support score.

        Two modes, selected by ``algorithm``:

        - ``graank`` (Palier 1, default): contemporaneous co-monotonic
          patterns. NO temporal lag, NO calibrated confidence.
        - ``tgrad`` (Palier 2): TEMPORAL gradual patterns — surfaces a time lag
          between a chosen target attribute and the others, e.g.
          "ndvi+, (precipitation+) +7.0 days". Requires ``target_column`` and
          ``order_by`` (the date-time column); the latter must hold real
          date/time values (so4gp parses them — an epoch integer or plain index
          will not be recognized as time). Still NO calibrated confidence (that
          is Palier 4).

        Args:
            source: "auto" | an s3:// URI (must be within the configured
                bucket) | a raw bucket key. "auto" requires tenant_id+space_id
                and reads
                s3://{bucket}/{tenant_id}/{space_id}/parquet/tabular_data.parquet
                (what rust_worker/not_stirling already materializes for Excel/CSV).
                Supported formats: parquet, geoparquet (geometry dropped), csv,
                and FLAT json record arrays. Raw GeoJSON FeatureCollections are
                not flattened here (later palier) — use GeoParquet for spatial.
            tenant_id: scope segment 1 (e.g. user UUID). Needed for "auto" and
                for the default output_key.
            space_id: scope segment 2 (e.g. space/directory id). Same.
            output_key: explicit RustFS key to write patterns to. Default:
                {tenant_id}/{space_id}/patterns/{algorithm}.parquet.
            algorithm: "graank" (contemporaneous) or "tgrad" (temporal).
            min_support: minimum support threshold (0-1).
            target_column: [tgrad only, REQUIRED] the reference/target attribute
                that the other attributes are lagged against.
            min_rep: [tgrad only] minimum representativity (0-1) — bounds how
                many lag steps are explored.
            columns: numeric columns to mine (default: all numeric columns). For
                tgrad, order_by + target_column are always kept regardless.
            order_by: time/sequence column to ORDER BY before mining. REQUIRED
                for tgrad (the time axis); recommended for graank.
            drop_columns: columns to exclude (geometry is dropped by default for
                geoparquet/geojson sources).
            source_kind: parquet|geoparquet|csv|json (inferred from extension).
            encode_equal: so4gp `eq` — treat equal consecutive values as gradual.
            limit: row cap (clamped to SPATIAL_QUERY_MAX_LIMIT).
            write_output: persist the patterns parquet to RustFS.

        Returns:
            {patterns, statistics, metadata}. Patterns are itemsets with support
            + raw metrics (density, singularity_scr, …); temporal patterns add a
            per-item role/lag_days and a top-level target + max_lag_days.
        """
        check_input(
            DiscoverPatternsInput,
            source=source,
            tenant_id=tenant_id,
            space_id=space_id,
            output_key=output_key,
            algorithm=algorithm,
            min_support=min_support,
            target_column=target_column,
            min_rep=min_rep,
            columns=columns,
            order_by=order_by,
            drop_columns=drop_columns,
            source_kind=source_kind,
            encode_equal=encode_equal,
            limit=limit,
            write_output=write_output,
        )
        log.info(
            "discover_patterns",
            source=source,
            tenant_id=tenant_id,
            space_id=space_id,
            algorithm=algorithm,
        )
        t0 = time.monotonic()

        algo_key = algorithm.lower()
        is_temporal = algo_key in SUPPORTED_TEMPORAL_ALGORITHMS
        if algo_key not in SUPPORTED_ALGORITHMS and not is_temporal:
            all_algos = sorted([*SUPPORTED_ALGORITHMS, *SUPPORTED_TEMPORAL_ALGORITHMS])
            return {"error": f"Unsupported algorithm '{algorithm}'. Supported: {all_algos}"}
        # Reject out-of-range thresholds up front — so4gp does not validate them
        # and silently returns an empty pattern set (which reads as a legitimate
        # "no patterns found" rather than the bad input it is).
        if not (0.0 <= min_support <= 1.0):
            return {"error": "min_support must be between 0 and 1."}
        if is_temporal:
            if not target_column:
                return {"error": f"algorithm='{algo_key}' requires target_column."}
            if not order_by:
                return {
                    "error": f"algorithm='{algo_key}' requires order_by (the date-time column)."
                }
            if not (0.0 < min_rep <= 1.0):
                return {"error": "min_rep must be in (0, 1] for temporal mining."}
            # order_by / target_column drive the mining frame; dropping them via
            # drop_columns would surface later as a confusing "column not found"
            # data error instead of the configuration mistake it is.
            dropped = set(drop_columns or [])
            clashing = [c for c in (order_by, target_column) if c in dropped]
            if clashing:
                return {
                    "error": f"drop_columns must not include order_by/target_column: {clashing}"
                }

        bucket = settings.S3_BUCKET
        if not bucket:
            return {"error": "S3_BUCKET is not configured — cannot reach RustFS."}

        # --- Resolve the source URI + kind (bucket-restricted) -----------
        try:
            if source == "auto":
                if not (tenant_id and space_id):
                    return {"error": "source='auto' requires tenant_id and space_id."}
                key = _validate_key(
                    f"{tenant_id}/{space_id}/parquet/tabular_data.parquet", role="source key"
                )
                source_uri = f"s3://{bucket}/{key}"
                kind = "parquet"
            elif source.startswith("s3://"):
                # Confine reads to the configured bucket — never let a caller
                # exfiltrate from an arbitrary bucket via the server's creds.
                prefix = f"s3://{bucket}/"
                if not source.startswith(prefix):
                    return {
                        "error": f"s3:// source must be within the configured bucket ({prefix}…)."
                    }
                source_uri = source
                kind = _infer_kind(source, source_kind)
            else:
                source_uri = f"s3://{bucket}/{_validate_key(source, role='source key')}"
                kind = _infer_kind(source, source_kind)
        except ValueError as e:
            return {"error": str(e)}

        # --- Resolve + validate the output destination BEFORE loading/mining
        # (fail fast — don't burn an expensive mine then discard it) -------
        output_uri = None
        if write_output:
            try:
                if output_key is None:
                    if not (tenant_id and space_id):
                        return {
                            "error": "write_output=True needs output_key or tenant_id+space_id."
                        }
                    output_key = f"{tenant_id}/{space_id}/patterns/{algorithm.lower()}.parquet"
                output_key = _validate_key(output_key, role="output_key")
            except ValueError as e:
                return {"error": str(e)}
            output_uri = f"s3://{bucket}/{output_key}"

        if drop_columns is None and kind == "geoparquet":
            drop_columns = ["geometry"]

        # Never load unbounded: GRAANK is O(n²) in rows, so default to the
        # configured guard rail unless the caller passes an explicit limit.
        effective_limit = min(
            limit if limit is not None else settings.SPATIAL_QUERY_MAX_LIMIT,
            settings.SPATIAL_QUERY_MAX_LIMIT,
        )

        # For tgrad, the time + target columns must survive an explicit column
        # projection — they are the time axis + reference, not optional. Only an
        # explicit, NON-EMPTY list is a projection: an empty list (how some
        # clients serialize an omitted optional) means "all columns", same as
        # None — augmenting it would wrongly drop every other numeric attribute.
        load_columns = columns
        if is_temporal and columns:
            load_columns = list(dict.fromkeys([*columns, order_by, target_column]))

        # --- Load via DuckDB --------------------------------------------
        try:
            df = await asyncio.to_thread(
                load_dataframe,
                source_uri,
                kind=kind,
                columns=load_columns,
                order_by=order_by,
                drop_columns=drop_columns,
                limit=effective_limit,
            )
        except Exception as e:  # noqa: BLE001 — surface load failures to the agent
            log.warning("discover_patterns_load_failed", source_uri=source_uri, error=str(e))
            return {"error": f"Failed to read source {source_uri}: {e}"}

        # --- Prepare the mining frame + mine (in a temp cwd) -------------
        if is_temporal:
            try:
                frame, mined_columns, target_idx = build_temporal_frame(
                    df,
                    time_column=order_by,
                    target_column=target_column,
                    drop_columns=drop_columns,
                )
            except ValueError as e:
                return {
                    "error": str(e),
                    "metadata": {"source_uri": source_uri, "rows": int(len(df))},
                }
            if len(frame) < 2:
                return {
                    "error": "Need at least 2 rows to mine temporal patterns.",
                    "metadata": {"source_uri": source_uri, "rows": int(len(frame))},
                }
            rows_mined = int(len(frame))
            try:
                result = await asyncio.to_thread(
                    _mine_in_tempdir,
                    mine_temporal_patterns,
                    frame,
                    min_support=min_support,
                    min_rep=min_rep,
                    encode_equal=encode_equal,
                    algorithm=algo_key,
                    target_col=target_idx,
                )
            except So4gpNotInstalled as e:
                return {"error": str(e)}
            except Exception as e:  # noqa: BLE001 — so4gp internal / MemoryError
                log.warning(
                    "discover_patterns_mine_failed",
                    source_uri=source_uri,
                    rows=rows_mined,
                    error=str(e),
                )
                return {"error": f"Temporal pattern mining failed ({type(e).__name__}): {e}"}
        else:
            numeric_df = df.select_dtypes(include=["number"])
            if order_by:
                numeric_df = numeric_df.drop(columns=[order_by], errors="ignore")
            if numeric_df.shape[1] < 2 or len(numeric_df) < 2:
                return {
                    "error": "Need at least 2 numeric columns and 2 rows to mine gradual patterns.",
                    "metadata": {
                        "source_uri": source_uri,
                        "rows": int(len(df)),
                        "numeric_columns": list(numeric_df.columns),
                    },
                }
            mined_columns = list(numeric_df.columns)
            rows_mined = int(len(numeric_df))
            try:
                result = await asyncio.to_thread(
                    _mine_in_tempdir,
                    mine_gradual_patterns,
                    numeric_df,
                    min_support=min_support,
                    encode_equal=encode_equal,
                    algorithm=algo_key,
                )
            except So4gpNotInstalled as e:
                return {"error": str(e)}
            except Exception as e:  # noqa: BLE001 — MemoryError (O(n²)) / so4gp internal failure
                log.warning(
                    "discover_patterns_mine_failed",
                    source_uri=source_uri,
                    rows=rows_mined,
                    error=str(e),
                )
                return {"error": f"Pattern mining failed ({type(e).__name__}): {e}"}

        # --- Write output (destination already resolved above) -----------
        if output_uri is not None:
            try:
                await asyncio.to_thread(_write_patterns_parquet, result, output_uri)
            except Exception as e:  # noqa: BLE001
                log.warning("discover_patterns_write_failed", output_uri=output_uri, error=str(e))
                return {
                    "error": f"Mined {len(result['patterns'])} patterns but failed to write {output_uri}: {e}"
                }

        metadata = {
            "algorithm": result["algorithm"],
            "algorithm_key": result["algorithm_key"],
            "min_support": min_support,
            "source_uri": source_uri,
            "output_uri": output_uri,
            "tenant_id": tenant_id,
            "space_id": space_id,
            "rows_mined": rows_mined,
            "row_limit": effective_limit,
            "columns_mined": mined_columns,
            "elapsed_ms": round((time.monotonic() - t0) * 1000),
        }
        if is_temporal:
            metadata.update(
                {
                    "target_column": target_column,
                    "time_column": order_by,
                    "min_rep": min_rep,
                    "note": (
                        "Palier 2: temporal gradual patterns with lag_days; "
                        "no calibrated confidence yet (see ADR 0001 §5/§7)."
                    ),
                }
            )
        else:
            metadata["note"] = (
                "Palier 1: contemporaneous co-monotonic patterns; "
                "no lag_days/confidence (see ADR 0001 §5/§7)."
            )

        return {
            "patterns": result["patterns"],
            "statistics": {
                "n_patterns": len(result["patterns"]),
                "runtime_seconds": result["runtime_seconds"],
                "invalid_count": result["invalid_count"],
            },
            "metadata": metadata,
        }

    @mcp.tool(output_schema=output_schema_for("discover_patterns_cv"))
    async def discover_patterns_cv(
        zone_ids: list[str],
        sources: dict[str, str | list[str]] | None = None,
        source_template: str | None = "climate/unified/{zone_id}.parquet",
        window_labels: list[str] | None = None,
        algorithm: str = "graank",
        min_support: float = 0.5,
        n_folds: int = 5,
        min_validated_zones: int = 1,
        target_column: str | None = None,
        min_rep: float = 0.5,
        columns: list[str] | None = None,
        order_by: str | None = "date",
        drop_columns: list[str] | None = None,
        source_kind: str | None = None,
        encode_equal: bool = False,
        limit: int | None = None,
        period_start: str | None = None,
        period_end: str | None = None,
        truncated_sources: dict[str, list[str]] | None = None,
        write_output: bool = False,
        output_key: str | None = None,
    ) -> dict:
        """Cross-validate gradual patterns across zones (FIERCE O3 minimal).

        The tool reads already-materialized B4 unified tables, mines patterns per
        zone, and reports which patterns survive held-out-zone validation. It
        fills O3 provenance fields (`zones_validated`, `n_folds`, `stability`)
        but deliberately leaves calibrated `confidence` unset for Palier 4.
        """
        check_input(
            DiscoverPatternsCvInput,
            zone_ids=zone_ids,
            sources=sources,
            source_template=source_template,
            window_labels=window_labels,
            algorithm=algorithm,
            min_support=min_support,
            n_folds=n_folds,
            min_validated_zones=min_validated_zones,
            target_column=target_column,
            min_rep=min_rep,
            columns=columns,
            order_by=order_by,
            drop_columns=drop_columns,
            source_kind=source_kind,
            encode_equal=encode_equal,
            limit=limit,
            period_start=period_start,
            period_end=period_end,
            truncated_sources=truncated_sources,
            write_output=write_output,
            output_key=output_key,
        )
        log.info("discover_patterns_cv", zones=len(zone_ids), algorithm=algorithm)
        t0 = time.monotonic()

        algo_key = algorithm.lower()
        is_temporal = algo_key in SUPPORTED_TEMPORAL_ALGORITHMS
        if algo_key not in SUPPORTED_ALGORITHMS and not is_temporal:
            all_algos = sorted([*SUPPORTED_ALGORITHMS, *SUPPORTED_TEMPORAL_ALGORITHMS])
            return {"error": f"Unsupported algorithm '{algorithm}'. Supported: {all_algos}"}
        if not (0.0 <= min_support <= 1.0):
            return {"error": "min_support must be between 0 and 1."}
        if is_temporal:
            if not target_column:
                return {"error": f"algorithm='{algo_key}' requires target_column."}
            if not order_by:
                return {
                    "error": f"algorithm='{algo_key}' requires order_by (the date-time column)."
                }
            if not (0.0 < min_rep <= 1.0):
                return {"error": "min_rep must be in (0, 1] for temporal mining."}
            dropped = set(drop_columns or [])
            clashing = [c for c in (order_by, target_column) if c in dropped]
            if clashing:
                return {
                    "error": f"drop_columns must not include order_by/target_column: {clashing}"
                }

        bucket = settings.S3_BUCKET
        if not bucket:
            return {"error": "S3_BUCKET is not configured — cannot reach RustFS."}

        try:
            source_map = _resolve_cv_sources(
                zone_ids=zone_ids,
                sources=sources,
                source_template=source_template,
                window_labels=window_labels,
                bucket=bucket,
            )
            output_uri = None
            if write_output:
                if not output_key:
                    return {"error": "write_output=True requires output_key for O3 artifacts."}
                output_uri = _resolve_bucket_uri(output_key, bucket=bucket, role="output_key")
        except ValueError as e:
            return {"error": str(e)}

        effective_limit = min(
            limit if limit is not None else settings.SPATIAL_QUERY_MAX_LIMIT,
            settings.SPATIAL_QUERY_MAX_LIMIT,
        )
        load_columns = columns
        if is_temporal and columns:
            load_columns = list(dict.fromkeys([*columns, order_by, target_column]))

        mined_by_zone: dict[str, dict] = {}
        zone_statistics: list[dict] = []
        for zone_id in zone_ids:
            try:
                df = await asyncio.to_thread(
                    _load_cv_zone_frame,
                    source_map[zone_id],
                    source_kind=source_kind,
                    columns=load_columns,
                    order_by=order_by,
                    drop_columns=drop_columns,
                    limit=effective_limit,
                )
                result, rows_mined, mined_columns = await asyncio.to_thread(
                    _mine_cv_frame,
                    df,
                    algorithm=algo_key,
                    min_support=min_support,
                    target_column=target_column,
                    min_rep=min_rep,
                    order_by=order_by,
                    drop_columns=drop_columns,
                    encode_equal=encode_equal,
                )
            except So4gpNotInstalled as e:
                return {"error": str(e)}
            except Exception as e:  # noqa: BLE001 — surface per-zone failures to the agent
                log.warning("discover_patterns_cv_zone_failed", zone_id=zone_id, error=str(e))
                return {
                    "error": f"Failed to mine zone {zone_id}: {e}",
                    "metadata": {"zone_id": zone_id, "sources": source_map.get(zone_id, [])},
                }
            mined_by_zone[zone_id] = result
            zone_statistics.append(
                {
                    "zone_id": zone_id,
                    "sources": source_map[zone_id],
                    "rows_mined": rows_mined,
                    "columns_mined": mined_columns,
                    "n_patterns": len(result.get("patterns", [])),
                }
            )

        try:
            library, folds, cv_stats = cross_validate_mined_patterns(
                mined_by_zone,
                zone_ids=zone_ids,
                algorithm=algo_key,
                min_support=min_support,
                n_folds=n_folds,
                min_validated_zones=min_validated_zones,
                period_start=_parse_date_or_none(period_start),
                period_end=_parse_date_or_none(period_end),
            )
        except ValueError as e:
            return {"error": str(e)}

        if output_uri is not None:
            try:
                await asyncio.to_thread(_write_validated_patterns_parquet, library, output_uri)
            except Exception as e:  # noqa: BLE001
                log.warning(
                    "discover_patterns_cv_write_failed", output_uri=output_uri, error=str(e)
                )
                return {
                    "error": (
                        f"Validated {len(library.patterns)} patterns but failed to write "
                        f"{output_uri}: {e}"
                    )
                }

        statistics = {
            **cv_stats,
            "runtime_ms": round((time.monotonic() - t0) * 1000),
            "zone_statistics": zone_statistics,
        }
        return {
            "patterns": [record.model_dump(mode="json") for record in library.patterns],
            "folds": folds,
            "output": output_uri,
            "statistics": statistics,
            "metadata": {
                "algorithm": algo_key,
                "min_support": min_support,
                "row_limit": effective_limit,
                "source_template": source_template,
                "window_labels": window_labels,
                "truncated_sources": truncated_sources or {},
                "note": (
                    "Palier 3: cross-zone validation fields are populated; "
                    "confidence remains null until Palier 4 pattern_alert."
                ),
            },
        }
