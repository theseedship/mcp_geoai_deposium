"""Climate ingestion + enrichment tools (FIERCE lots B2 / B4 / B6).

Three MCP tools:

- ``era5_ingest`` — fetch ERA5 **daily** reanalysis for a zone centroid from the Open-Meteo
  Archive API (:mod:`src.utils.openmeteo`), normalize to the terra-cognita §5.1 schema, and write a
  DuckDB-spatial-readable **WKB GeoParquet** (``climate/era5/{zone_id}.parquet``). Reports
  completeness; does not *gate* on it (ADR 0002 §3 — MVP).
- ``climate_enrich_ndvi`` — load that ERA5 parquet, collect the zone's NDVI profile, align NDVI onto
  the ERA5 daily grid by nearest date (±N days, *proposed* — ADR 0002 §3), and **materialize** the
  unified ``climate/unified/{zone_id}.parquet`` (terra-cognita §5.3), also returning a
  DataFrame-ready row set for ``discover_patterns``.
- ``climate_percentiles`` — compute climatological P10/P50/P90 ERA5 normals by
  calendar month × variable × zone, defaulting to the WMO 1991-2020 window and
  materializing ``climate/normals/{zone_id}.parquet``.

The analysis core is pure + network-free (``_align_ndvi_to_era5``, ``_completeness_report``,
``normalize_archive_response``, ``compute_monthly_normals``, ``classify_against_normals``) so it is
unit-testable without the live API; the handlers add the zone lookup, the Open-Meteo fetch, and the
GeoParquet I/O. Built on the #35 contracts registry.

*Terminology:* "GeoParquet" here means a Parquet with a **WKB geometry BLOB** (EPSG:4326 implicit),
read via ``ST_GeomFromWKB`` — exactly like ``scripts/generate_zones.py``. It is **not** a
GeoParquet-``geo``-metadata-tagged file (that needs the GDAL Parquet driver, absent here), so it is
not consumable as GeoParquet by GDAL/geopandas.
"""

from __future__ import annotations

import asyncio
import json
import math
import re
import time
from datetime import UTC, date, datetime
from functools import lru_cache
from pathlib import Path

import structlog

from src.config import settings
from src.contracts import (
    ClimateEnrichNdviInput,
    ClimateNormalsInput,
    ClimatePercentilesInput,
    Era5IngestInput,
    NdviIngestInput,
    check_input,
    output_schema_for,
)
from src.tools._ndvi import _collect_ndvi_monthly, _collect_ndvi_observations
from src.utils.dates import month_ranges
from src.utils.duckdb_conn import duckdb_session
from src.utils.normals_reader import (
    NormalsArtifactRead,
    normals_read_error_reason,
    read_normals_artifact,
)
from src.utils.openmeteo import (
    NORMALIZED_VARS,
    OPEN_METEO_DATA_LICENSE,
    fetch_archive_daily,
    normalize_openmeteo_daily,
    provider_var,
    split_date_range,
)
from src.utils.path_policy import resolve_local_parquet_path
from src.utils.stats import percentile_thresholds

log = structlog.get_logger(__name__)

_ZONES_GEOJSON = Path(__file__).resolve().parents[2] / "zones" / "zones.geojson"

# Allowed zone roles in zones.geojson. The scientific zones (FIERCE B1: 12 panel + 3 pilot) are the
# panel + pilot roles; a client site is tenant-scoped and never enters a default zone set
# (docs/design/b1-pilot-boundaries.md, "Client sites are not pilot zones").
ZONE_ROLES = frozenset({"panel", "pilot", "client"})
SCIENTIFIC_ZONE_ROLES = frozenset({"panel", "pilot"})

# zone_id is a safe key; the destination must be a quote-free parquet path/URI (it is interpolated
# into a DuckDB COPY target, which cannot be parameter-bound).
_SAFE_ZONE_RE = re.compile(r"^[A-Za-z0-9._-]+$")
_SAFE_DST_RE = re.compile(r"^[A-Za-z0-9._:/-]+\.parquet$")

# The canonical climate artifacts this module persists, one prefix per kind:
# ``s3://{bucket}/climate/{kind}/{zone_id}.parquet``. Source of truth for
# ``_default_output_uri`` *and* for the storage-contract drift guards in
# ``tests/test_docs_drift.py``, which import it rather than re-typing it — adding a kind
# here is what makes those guards demand the docs list it.
PERSISTED_CLIMATE_KINDS = ("era5", "unified", "normals", "ndvi")

_COMPLETENESS_THRESHOLD_PCT = 98.0  # ADR 0002 §3 — reported, NOT enforced at MVP
_WMO_NORMAL_START = "1991-01-01"
_WMO_NORMAL_END = "2020-12-31"
_NORMAL_GRAIN = "daily_values_by_calendar_month"


# --------------------------------------------------------------------------- #
# Zone resolution
# --------------------------------------------------------------------------- #


@lru_cache(maxsize=1)
def _load_zones() -> dict[str, dict]:
    """Parse zones/zones.geojson once → {zone_id: {lon, lat, bbox, geometry, role}}."""
    with open(_ZONES_GEOJSON, encoding="utf-8") as f:
        fc = json.load(f)
    zones: dict[str, dict] = {}
    for feature in fc.get("features", []):
        props = feature.get("properties", {})
        zid = props.get("zone_id")
        bbox = props.get("bbox")
        if not zid or not bbox or len(bbox) != 4:
            continue
        zones[zid] = {
            "lon": (bbox[0] + bbox[2]) / 2.0,  # centroid of the zone bbox (WGS84)
            "lat": (bbox[1] + bbox[3]) / 2.0,
            "bbox": list(bbox),
            "geometry": feature.get("geometry"),
            "role": props.get("role"),
        }
    return zones


def _resolve_zone(zone_id: str) -> dict | None:
    """Resolve a zone_id to its centroid + bbox + polygon, or None if unknown."""
    return _load_zones().get(zone_id)


def _scientific_zone_ids(zones: dict[str, dict]) -> list[str]:
    """Sorted panel + pilot zone ids: the default zone set, which never includes a client site.

    Raises ``ValueError`` on a missing or unknown role, so a malformed registry fails
    instead of silently shrinking the default panel.
    """
    bad = sorted(zid for zid, zone in zones.items() if zone.get("role") not in ZONE_ROLES)
    if bad:
        raise ValueError(
            f"zones with a missing or unknown role: {bad}; expected one of {sorted(ZONE_ROLES)}"
        )
    return sorted(zid for zid, zone in zones.items() if zone.get("role") in SCIENTIFIC_ZONE_ROLES)


# --------------------------------------------------------------------------- #
# Normalization (pure)
# --------------------------------------------------------------------------- #


def normalize_archive_response(
    raws: list[dict], normalized_variables: list[str]
) -> tuple[list[str], dict[str, list[float | None]]]:
    """Merge one-or-more Archive JSON chunks → (dates, {normalized_var: [stored-unit values]}).

    Concatenates the ``daily`` blocks in order and applies the documented unit conversions
    (only ``solar_radiation`` MJ/m² → W/m²). Raises ``KeyError`` if a requested provider variable
    is absent from the response (an upstream rename — caught by the live contract test).

    Thin wrapper over :func:`src.utils.openmeteo.normalize_openmeteo_daily` (``strict=True``) —
    a behaviour-preserving refactor (audit B5 §4.4): the per-chunk fail-loud body moved DOWN to the
    ``openmeteo`` layer so the forecast client can reuse it (with ``strict=False``) without importing
    ``climate.py`` (a cycle). The public name + KeyError-on-rename guarantee are unchanged.
    """
    dates: list[str] = []
    columns: dict[str, list[float | None]] = {v: [] for v in normalized_variables}
    for raw in raws:
        chunk_dates, chunk_columns = normalize_openmeteo_daily(
            raw, normalized_variables, strict=True
        )
        dates.extend(chunk_dates)
        for v in normalized_variables:
            columns[v].extend(chunk_columns[v])
    return dates, columns


def _completeness_report(
    dates: list[str],
    columns: dict[str, list[float | None]],
    start_date: str,
    end_date: str,
) -> dict:
    """Expected vs received daily records, per variable + overall (reported, not gated)."""
    expected = (date.fromisoformat(end_date) - date.fromisoformat(start_date)).days + 1
    per_variable = {}
    for v, series in columns.items():
        received = sum(1 for x in series if x is not None)
        per_variable[v] = {
            "received": received,
            "expected": expected,
            "pct_complete": round(100.0 * received / expected, 2) if expected else 0.0,
        }
    return {
        "expected_days": expected,
        "received_days": len(dates),
        "pct_days": round(100.0 * len(dates) / expected, 2) if expected else 0.0,
        "per_variable": per_variable,
        "threshold_pct": _COMPLETENESS_THRESHOLD_PCT,
        "gate_enforced": False,  # ADR 0002 §3 — funded ESP deliverable enforces this
    }


def _value_stats(columns: dict[str, list[float | None]]) -> dict:
    """min / max / mean / n per variable, ignoring missing (None) records."""
    stats: dict[str, dict] = {}
    for v, series in columns.items():
        vals = [x for x in series if x is not None]
        if vals:
            stats[v] = {
                "min": round(min(vals), 4),
                "max": round(max(vals), 4),
                "mean": round(sum(vals) / len(vals), 4),
                "n": len(vals),
            }
        else:
            stats[v] = {"min": None, "max": None, "mean": None, "n": 0}
    return stats


def _provider_map(normalized_variables: list[str]) -> dict:
    """Traceability: normalized → {provider var, stored unit, conversion note}."""
    out = {}
    for v in normalized_variables:
        pv = provider_var(v)
        out[v] = {"provider": pv.provider, "unit": pv.unit, "note": pv.note}
    return out


def _align_ndvi_to_era5(
    era5_rows: list[dict], ndvi_obs: list[dict], *, tolerance_days: int
) -> list[dict]:
    """Snap each NDVI observation onto its nearest ERA5 daily-grid row within ±tolerance_days.

    Pure. One unified row per *matched* NDVI observation (NDVI obs with no ERA5 day inside the
    window are dropped — the ±N d alignment is *proposed*, ADR 0002 §3). The row carries the ERA5
    day's climate columns + ``ndvi_date`` (the Sentinel-2 acquisition date) / ``ndvi_mean`` /
    ``ndvi_std`` (uncertainty, terra §5.3) / ``day_delta``.
    """
    grid: list[tuple[date, dict]] = []
    for r in era5_rows:
        try:
            grid.append((date.fromisoformat(str(r["date"])[:10]), r))
        except (ValueError, KeyError, TypeError):
            continue

    # One row per ERA5 grid date (the unified key is (zone, date)). When several NDVI obs snap to
    # the SAME ERA5 day — e.g. two STAC items from one overpass in an MGRS tile-overlap, or two
    # dates collapsing onto one day across an ERA5 gap — keep the closest (smallest day_delta,
    # tie-broken by higher NDVI) so the table has no duplicate (zone, date) rows.
    best_by_date: dict[str, dict] = {}
    for obs in ndvi_obs:
        raw = obs.get("date")
        if not raw:
            continue
        try:
            od = date.fromisoformat(str(raw)[:10])
        except ValueError:
            continue
        best_row: dict | None = None
        best_delta: int | None = None
        for gd, row in grid:
            delta = abs((gd - od).days)
            if delta <= tolerance_days and (best_delta is None or delta < best_delta):
                best_row, best_delta = row, delta
        if best_row is None:
            continue
        merged = {k: v for k, v in best_row.items() if k != "geometry_wkt"}
        merged["ndvi_date"] = od.isoformat()  # the Sentinel-2 acquisition date
        merged["ndvi_mean"] = obs.get("ndvi_mean")
        merged["ndvi_std"] = obs.get("ndvi_std")  # uncertainty (terra §5.3) — None if absent
        merged["day_delta"] = best_delta

        key = merged["date"]
        incumbent = best_by_date.get(key)
        if incumbent is None or _better_match(merged, incumbent):
            best_by_date[key] = merged
    return [best_by_date[k] for k in sorted(best_by_date)]


def _ndvi_alignment_census(
    era5_rows: list[dict], ndvi_obs: list[dict], *, tolerance_days: int
) -> dict:
    """How each NDVI observation fared against the ERA5 grid, counted independently of the rows.

    `len(ndvi_obs) - len(unified)` is NOT the number of observations with no ERA5 day, and using it
    as one made the metadata state something false. `_align_ndvi_to_era5` keeps ONE row per ERA5
    date, so two observations either side of a single ERA5 day collapse into one row — measured:
    2 observations, 1 unified row, "1 had no ERA5 day within ±3 d" when both did (Greptile P1 /
    Copilot, PR #188).

    The two outcomes have opposite meanings and must not share a number. An observation past the
    end of the climate record is a COVERAGE gap — the table is narrower than the window asked for.
    An observation collapsed onto a shared ERA5 day is a RESOLUTION limit — the unified key is
    `(zone, date)`, so the closest observation wins and the table is complete for the days it has.
    """
    grid: set[date] = set()
    for row in era5_rows:
        try:
            grid.add(date.fromisoformat(str(row["date"])[:10]))
        except (ValueError, KeyError, TypeError):
            continue

    matchable = unreadable = 0
    for obs in ndvi_obs:
        raw = obs.get("date")
        if not raw:
            unreadable += 1
            continue
        try:
            od = date.fromisoformat(str(raw)[:10])
        except (ValueError, TypeError):
            unreadable += 1
            continue
        if any(abs((gd - od).days) <= tolerance_days for gd in grid):
            matchable += 1

    return {
        "matchable": matchable,
        "unreadable": unreadable,
        "out_of_range": len(ndvi_obs) - matchable - unreadable,
    }


def _better_match(cand: dict, incumbent: dict) -> bool:
    """Tie-break two NDVI obs snapping to the same ERA5 day: smaller day_delta wins, then higher
    NDVI (None treated as worst)."""
    if cand["day_delta"] != incumbent["day_delta"]:
        return cand["day_delta"] < incumbent["day_delta"]
    cand_ndvi = cand["ndvi_mean"] if cand["ndvi_mean"] is not None else float("-inf")
    inc_ndvi = incumbent["ndvi_mean"] if incumbent["ndvi_mean"] is not None else float("-inf")
    return cand_ndvi > inc_ndvi


# --------------------------------------------------------------------------- #
# GeoParquet I/O (WKB, EPSG:4326) — mirrors scripts/generate_zones.py
# --------------------------------------------------------------------------- #


def _validate_dst(dst: str) -> None:
    """Confine an interpolated COPY/read_parquet target (it cannot be parameter-bound).

    ``output_path`` / ``era5_source`` are agent-controlled, so beyond the charset/.parquet SQLi
    guard we also (a) reject path traversal, and (b) allow only a policy-enabled local path under
    ``LOCAL_PARQUET_SOURCE_ROOT`` or an ``s3://`` URI **inside the configured bucket's
    ``climate/`` prefix** — blocking ``http(s)://`` / ``file://`` (SSRF to
    cloud-metadata/internal hosts) and cross-bucket object-store reads/writes.
    """
    # fullmatch (not match) so a trailing newline can't slip past the `$` anchor.
    if not _SAFE_DST_RE.fullmatch(dst):
        raise ValueError(f"unsafe parquet destination: {dst!r}")
    if ".." in dst:
        raise ValueError(f"path traversal not allowed in destination: {dst!r}")
    if "://" in dst:
        scheme = dst.split("://", 1)[0]
        if scheme != "s3":
            raise ValueError(
                f"unsupported destination scheme {scheme!r} — use a local path or s3:// only"
            )
        prefix = f"s3://{settings.S3_BUCKET}/climate/" if settings.S3_BUCKET else None
        if not prefix or not dst.startswith(prefix):
            raise ValueError(f"s3 destination must be under s3://{settings.S3_BUCKET}/climate/")
    else:
        resolve_local_parquet_path(dst, role="climate parquet path")


def _default_output_uri(zone_id: str, kind: str) -> str | None:
    """``s3://{bucket}/climate/{kind}/{zone_id}.parquet`` if a bucket is configured, else None."""
    if kind not in PERSISTED_CLIMATE_KINDS or not _SAFE_ZONE_RE.fullmatch(zone_id):
        return None
    if not settings.S3_BUCKET:
        return None
    return f"s3://{settings.S3_BUCKET}/climate/{kind}/{zone_id}.parquet"


def _write_geoparquet(
    records: list[dict],
    *,
    columns: list[tuple[str, str]],
    dst: str,
    date_cols: tuple[str, ...] = ("date",),
    wkt_col: str = "wkt",
) -> None:
    """Write ``records`` to a WKB GeoParquet at ``dst`` (local path or ``s3://`` URI).

    ``columns`` is the ordered staging schema ``(name, ddl_type)`` (must include ``wkt_col`` as
    VARCHAR holding ``POINT(lon lat)``). On write: each ``date_cols`` entry is ``CAST … AS DATE`` and
    ``wkt_col`` becomes ``ST_AsWKB(ST_GeomFromText(...)) AS geometry`` — read back via
    ``ST_GeomFromWKB(geometry)``. One statement per ``execute()`` (DuckDB cross-version safety).
    """
    _validate_dst(dst)
    names = [c for c, _ in columns]
    ddl = ", ".join(f"{c} {t}" for c, t in columns)
    placeholders = ", ".join(["?"] * len(names))
    rows = [tuple(rec.get(c) for c in names) for rec in records]

    projections: list[str] = []
    for c in names:
        if c == wkt_col:
            continue
        projections.append(f"CAST({c} AS DATE) AS {c}" if c in date_cols else c)
    projections.append(f"ST_AsWKB(ST_GeomFromText({wkt_col})) AS geometry")
    order_by = date_cols[0] if date_cols else names[0]

    with duckdb_session(":memory:", load_httpfs=dst.startswith("s3://")) as conn:
        conn.execute(f"CREATE TABLE _stg ({ddl})")
        if rows:
            conn.executemany(f"INSERT INTO _stg VALUES ({placeholders})", rows)
        conn.execute(
            f"COPY (SELECT {', '.join(projections)} FROM _stg ORDER BY {order_by}) "
            f"TO '{dst}' (FORMAT PARQUET)"
        )


def _read_era5_records(path: str) -> list[dict]:
    """Read an ERA5 GeoParquet back into dict rows (``date`` as ISO str, geometry as WKT)."""
    _validate_dst(path)
    with duckdb_session(":memory:", load_httpfs=path.startswith("s3://")) as conn:
        res = conn.execute(
            "SELECT * EXCLUDE (geometry, date), CAST(date AS VARCHAR) AS date, "
            f"ST_AsText(ST_GeomFromWKB(geometry)) AS geometry_wkt "
            f"FROM read_parquet('{path}') ORDER BY date"
        ).fetchall()
        col_names = [d[0] for d in conn.description]
        return [dict(zip(col_names, row, strict=True)) for row in res]


class NdviSourceError(ValueError):
    """An explicit ``ndvi_source`` could not be used as the B4 NDVI input.

    Its own class because the whole point of `ndvi_source` is that it does NOT degrade: the caller
    named a durable B3 artifact, so a source that cannot be read, does not hold the zone, or does
    not carry an NDVI series is an error to report, never a reason to go back to the provider. An
    exception type keeps that decision at the call site instead of encoding it in a return value
    the collection path would have to be taught to distinguish.
    """


# The columns a B3 series must actually carry to be an NDVI input. `ndvi_std` is deliberately NOT
# here: `_align_ndvi_to_era5` reads it with `.get`, publishes `None` when absent, and terra §5.3
# treats the uncertainty as optional — requiring it would refuse a usable series over a column the
# aligner already handles missing.
_NDVI_SOURCE_REQUIRED_COLUMNS = ("date", "zone_id", "ndvi_mean")


def _read_ndvi_records(
    path: str, *, zone_id: str, start_date: str, end_date: str
) -> tuple[list[dict], dict]:
    """``(the zone's NDVI observations in window, what the source turned out to be)``.

    The read side of B3. `ndvi_ingest` writes `climate/ndvi/{zone_id}.parquet` and, until this
    existed, nothing read it: `climate_enrich_ndvi` re-derived NDVI from STAC on every call, so
    #133's "depends on the durable B3 materialization" was a dependency the code could not express
    (#177).

    Confined by `_validate_dst`, the same guard `era5_source` gets and for the same reason — the
    path is agent-controlled and is interpolated into `read_parquet`, so it is restricted to the
    configured bucket's `climate/` prefix or a policy-enabled local root.

    Four things are checked before a row is returned, and each one is a way the run could otherwise
    publish a unified table built on something other than what was asked for:

    * the SCHEMA — a parquet that is readable but carries no `ndvi_mean` is not an NDVI series;
    * the ZONE — the file is filtered to `zone_id` exactly as the ERA5 side is, so a multi-zone or
      mislabelled artifact cannot align one zone's NDVI onto another's climate;
    * the WINDOW — rows outside `[start_date, end_date]` are dropped, so pointing at a decade-long
      artifact for a one-year run reports a one-year table and not a decade of it;
    * the RESULT — zero rows in window is an error, not an empty unified table. The caller named a
      source expecting it to hold the series; silently writing nothing would look like "no NDVI
      matched" rather than "you pointed at the wrong window".

    Acquisition dates are preserved verbatim. `date` in the B3 artifact IS the Sentinel-2 scene
    date, and it becomes `ndvi_date` downstream — no re-derivation, no snapping on this side.

    The second half of the return is the coverage the caller has to publish. Reading a stored
    series removes the STAC item cap and therefore the truncation warning, and a warning
    disappearing must not be allowed to read as "coverage is complete": the pilot archives have
    real missing months (81 of 120 for montpellier_metropole), and that gap is a property of the
    data, not of the reader.
    """
    _validate_dst(path)
    with duckdb_session(":memory:", load_httpfs=path.startswith("s3://")) as conn:
        try:
            described = conn.execute(f"DESCRIBE SELECT * FROM read_parquet('{path}')").fetchall()
        except Exception as e:
            raise NdviSourceError(f"could not read NDVI source {path!r}: {e}") from e
        columns = {row[0] for row in described}
        missing = [c for c in _NDVI_SOURCE_REQUIRED_COLUMNS if c not in columns]
        if missing:
            raise NdviSourceError(
                f"NDVI source {path!r} is not a B3 series — missing column(s) "
                f"{', '.join(missing)}; expected at least "
                f"{', '.join(_NDVI_SOURCE_REQUIRED_COLUMNS)}"
            )
        # The NDVI values are CAST rather than taken as stored. This repo's own B3 artifacts are
        # DOUBLE, but a foreign parquet may spell them DECIMAL, and DuckDB then hands back
        # `Decimal('0.71')` — which compares unequal to `0.71`, serializes differently in the tool
        # result, and only becomes a float again when `_write_geoparquet` stages it. Normalizing
        # here keeps the returned rows and the written table agreeing about their own type.
        optional = [
            f"CAST({c} AS DOUBLE) AS {c}" if c == "ndvi_std" else c
            for c in ("ndvi_std", "cloud_cover", "item_id")
            if c in columns
        ]
        selected = ", ".join(
            [
                "CAST(date AS VARCHAR) AS date",
                "zone_id",
                "CAST(ndvi_mean AS DOUBLE) AS ndvi_mean",
                *optional,
            ]
        )
        try:
            res = conn.execute(
                f"SELECT {selected} FROM read_parquet('{path}') "  # noqa: S608 - _validate_dst
                "WHERE zone_id = ? AND CAST(date AS VARCHAR) BETWEEN ? AND ? "
                "ORDER BY date",
                [zone_id, start_date, end_date],
            ).fetchall()
            names = [d[0] for d in conn.description]
        except Exception as e:
            raise NdviSourceError(f"could not read NDVI source {path!r}: {e}") from e

    rows = [dict(zip(names, row, strict=True)) for row in res]
    if not rows:
        raise NdviSourceError(
            f"NDVI source {path!r} has no rows for zone {zone_id!r} in "
            f"{start_date}..{end_date} — check the zone and the window, or run ndvi_ingest"
        )

    per_month: dict[str, int] = {}
    for row in rows:
        per_month[str(row["date"])[:7]] = per_month.get(str(row["date"])[:7], 0) + 1
    requested = len(month_ranges(start_date, end_date))
    coverage = {
        "months_requested": requested,
        "months_observed": len(per_month),
        "max_observations_in_a_month": max(per_month.values()),
        "first_observation": str(rows[0]["date"])[:10],
        "last_observation": str(rows[-1]["date"])[:10],
    }
    return rows, coverage


def _output_exists(dst: str) -> bool:
    """True if ``dst`` is a readable parquet (idempotent-skip check for overwrite=False).

    Best-effort: any read error (including a transient object-store blip) is treated as "absent",
    so at worst the deterministic ERA5 pull is re-fetched and the identical file re-written — never
    silent corruption. This is a skip *optimization*, not a data-integrity guarantee.
    """
    _validate_dst(dst)
    with duckdb_session(":memory:", load_httpfs=dst.startswith("s3://")) as conn:
        try:
            conn.execute(f"SELECT 1 FROM read_parquet('{dst}') LIMIT 1")
            return True
        except Exception:
            return False


def _stored_coverage(
    dst: str,
    *,
    zone_id: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
) -> dict:
    """Best-effort stored coverage at ``dst`` for one zone.

    ``n_days`` is a distinct date count inside the requested window, not a whole-file row count, so
    duplicate rows or out-of-window dates cannot make a sparse existing artifact look complete.
    """
    _validate_dst(dst)
    predicates = []
    params = []
    if zone_id:
        predicates.append("zone_id = ?")
        params.append(zone_id)
    if start_date and end_date:
        predicates.append("CAST(date AS DATE) BETWEEN CAST(? AS DATE) AND CAST(? AS DATE)")
        params.extend([start_date, end_date])
    where = f" WHERE {' AND '.join(predicates)}" if predicates else ""
    with duckdb_session(":memory:", load_httpfs=dst.startswith("s3://")) as conn:
        try:
            row = conn.execute(
                "SELECT CAST(MIN(date) AS VARCHAR), CAST(MAX(date) AS VARCHAR), "
                "COUNT(DISTINCT CAST(date AS DATE)) "
                f"FROM read_parquet('{dst}')" + where,
                params,
            ).fetchone()
            cols = [
                d[0]
                for d in conn.execute(f"SELECT * FROM read_parquet('{dst}') LIMIT 0").description
            ]
            return {"start": row[0], "end": row[1], "n_days": row[2], "columns": cols}
        except Exception:
            return {}


def _stored_completeness(
    dst: str,
    *,
    zone_id: str,
    start_date: str,
    end_date: str,
    requested: list[str],
) -> dict:
    """Completeness report for an existing ERA5 parquet clipped to the requested window."""
    _validate_dst(dst)
    expected = (date.fromisoformat(end_date) - date.fromisoformat(start_date)).days + 1
    if not expected:
        return {}
    select_cols = ["COUNT(DISTINCT CAST(date AS DATE))"] + [
        f"COUNT(DISTINCT CASE WHEN {v} IS NOT NULL THEN CAST(date AS DATE) END)" for v in requested
    ]
    query = (
        f"SELECT {', '.join(select_cols)} FROM read_parquet('{dst}') "
        f"WHERE zone_id = ? AND CAST(date AS DATE) BETWEEN DATE '{start_date}' AND DATE '{end_date}'"
    )
    with duckdb_session(":memory:", load_httpfs=dst.startswith("s3://")) as conn:
        try:
            row = conn.execute(query, [zone_id]).fetchone()
        except Exception:
            return {}

    received_days = int(row[0] or 0)
    per_variable = {}
    for index, variable in enumerate(requested, start=1):
        received = int(row[index] or 0)
        per_variable[variable] = {
            "received": received,
            "expected": expected,
            "pct_complete": round(100.0 * received / expected, 2),
        }
    return {
        "expected_days": expected,
        "received_days": received_days,
        "pct_days": round(100.0 * received_days / expected, 2),
        "per_variable": per_variable,
    }


def _coverage_satisfies(stored: dict, start_date: str, end_date: str, requested: list[str]) -> bool:
    """True iff the stored parquet's date range superset-covers [start_date, end_date] AND holds
    every requested variable column (ISO date strings compare chronologically)."""
    s, e = stored.get("start"), stored.get("end")
    if not s or not e or s > start_date or e < end_date:
        return False
    expected_days = (date.fromisoformat(end_date) - date.fromisoformat(start_date)).days + 1
    if int(stored.get("n_days") or 0) < expected_days:
        return False
    cols = set(stored.get("columns") or [])
    return all(v in cols for v in requested)


# --------------------------------------------------------------------------- #
# Climate normals / percentile baseline (FIERCE B6 Palier-3)
# --------------------------------------------------------------------------- #


def _resolve_normals_window(start_date: str | None, end_date: str | None) -> tuple[str, str, str]:
    """Resolve optional dates to the default WMO 1991-2020 normal window."""
    resolved_start = start_date or _WMO_NORMAL_START
    resolved_end = end_date or _WMO_NORMAL_END
    label = (
        "WMO_1991_2020"
        if resolved_start == _WMO_NORMAL_START and resolved_end == _WMO_NORMAL_END
        else "custom"
    )
    return resolved_start, resolved_end, label


def _filter_era5_rows_for_window(
    rows: list[dict], *, zone_id: str, start_date: str, end_date: str
) -> list[dict]:
    """Keep only rows for ``zone_id`` whose daily date is inside the requested window."""
    kept = []
    for row in rows:
        d = str(row.get("date", ""))[:10]
        if row.get("zone_id") != zone_id or not d or d < start_date or d > end_date:
            continue
        kept.append(row)
    kept.sort(key=lambda r: str(r.get("date", ""))[:10])
    return kept


def _era5_rows_cover_window(
    rows: list[dict], *, start_date: str, end_date: str, requested: list[str]
) -> bool:
    """Coverage check for an already-read ERA5 row set, after zone/window filtering.

    Requires not just the right [start, end] span but FULL daily density: a real ``era5_ingest`` output
    has one row per calendar day, so a sparse / boundary-only parquet (e.g. just the first + last day)
    must NOT count as 'covering' the window — otherwise the normals would be computed from a handful of
    rows (mostly-null monthly buckets) under ``source="era5_parquet"`` instead of fetching the real days
    (Greptile P1)."""
    if not rows:
        return False
    dates = {str(r.get("date", ""))[:10] for r in rows if r.get("date")}
    if not dates or min(dates) > start_date or max(dates) < end_date:
        return False
    expected_days = (date.fromisoformat(end_date) - date.fromisoformat(start_date)).days + 1
    if len(dates) < expected_days:  # rows are clipped to the window → this means a missing-day gap
        return False
    cols = set().union(*(r.keys() for r in rows))
    return all(v in cols for v in requested)


def _records_from_archive(
    dates: list[str],
    columns: dict[str, list[float | None]],
    *,
    zone_id: str,
    lat: float,
    lon: float,
    requested: list[str],
) -> list[dict]:
    """Build ERA5-like row dicts from normalized Archive response columns."""
    return [
        {
            "date": d,
            "zone_id": zone_id,
            "lat": lat,
            "lon": lon,
            **{v: columns[v][i] for v in requested},
        }
        for i, d in enumerate(dates)
    ]


# The Climdex definition of a wet day. Named and published rather than inlined, because the
# resulting wet-day P10/P50/P90 is a PROJECT-SPECIFIC classification and not the Climdex R95p
# "very wet days" index — a reader who assumes the latter would attribute a standard meaning to
# numbers that do not have it (`docs/design/site-watch-v0.md` G2).
WET_DAY_THRESHOLD_MM = 1.0

# Stamped on every normals row so a reader can tell a v1 artifact (all-days percentiles only) from
# a v2 one (all-days plus the precipitation wet-day triplet) without inspecting its column list.
# v1 files carry no such column; `_normals_coverage` reads it defensively and reports 1 for them.
NORMALS_SCHEMA_VERSION = 2

# The variable the wet-day triplet exists for. The all-days percentiles of precipitation include
# dry days, which drag P90 down: a month that is dry for 25 days and pours on 5 has an all-days P90
# below several of its own wet days, so banding heavy rain against it understates every one of
# them. Temperature has no equivalent — there is no "dry day" to exclude — so the wet columns are
# NULL for every other variable rather than 0, which would claim a measured absence.
_WET_DAY_VARIABLE = "precipitation"


def compute_monthly_normals(
    rows: list[dict],
    *,
    zone_id: str,
    lat: float,
    lon: float,
    requested: list[str],
    window_start: str,
    window_end: str,
    window_label: str,
) -> list[dict]:
    """Compute P10/P50/P90 + n/mean per normalized variable and calendar month.

    The resolved climate-normal window (start/end/label) is stamped on every row so the materialized
    parquet records *which window produced these normals*; the idempotent-skip then refuses to return
    normals computed over a different window (a custom-window re-run would otherwise silently poison the
    per-zone default path — review P1).

    Schema v2 adds the precipitation WET-DAY triplet beside the all-days one — never in place of it
    (#172, Site-Watch V0 G2). Four states have to stay distinguishable in the output, and merging
    any two of them is the defect this shape exists to avoid:

    ==========================  ==========  =======  ===========================================
    row                         p*_wet      n_wet    meaning
    ==========================  ==========  =======  ===========================================
    any non-precipitation       ``None``    ``None`` not applicable — there is no dry day to drop
    precipitation, no samples   ``None``    ``None`` not observed — no valid values in this bucket
    precipitation, wet days     computed    ``k``    the heavy-rain distribution
    precipitation, no wet days  ``None``    ``0``    MEASURED absence: 30 years, not one wet day
    ==========================  ==========  =======  ===========================================

    The last row is not a hypothetical. `niamey` records 0.0 mm for January, February, November and
    December across the whole 1991-2020 window, so a coverage gate that demanded a wet-day
    percentile would reject valid arid-zone data (`docs/CONSOLIDATION_ROADMAP.md` "Arid-zone trap").
    `n_wet = 0` with `p*_wet = None` says "we looked and there were none"; `n_wet = None` means
    either not applicable (non-precipitation) or not observed (precipitation with `n = 0`)."""
    buckets: dict[tuple[str, int], list[float]] = {
        (v, m): [] for v in requested for m in range(1, 13)
    }
    for row in rows:
        try:
            month = date.fromisoformat(str(row["date"])[:10]).month
        except (KeyError, TypeError, ValueError):
            continue
        for v in requested:
            value = row.get(v)
            if value is None:
                continue
            try:
                numeric = float(value)
            except (TypeError, ValueError, OverflowError):  # OverflowError: an int past float range
                continue
            if math.isfinite(numeric):
                buckets[(v, month)].append(numeric)

    wkt = f"POINT({lon} {lat})"
    normals = []
    for variable in requested:
        is_wet_day_variable = variable == _WET_DAY_VARIABLE
        for month in range(1, 13):
            values = buckets[(variable, month)]
            if values:
                pct = percentile_thresholds(values, level=50)
                mean = round(sum(values) / len(values), 4)
                p10, p50, p90 = pct["p10"], pct["p50"], pct["p90"]
            else:
                mean = p10 = p50 = p90 = None

            # The wet-day pass reads the SAME bucket, filtered — not a second accumulation, so the
            # two distributions cannot drift apart over the same days.
            p10_wet = p50_wet = p90_wet = None
            n_wet = None
            if is_wet_day_variable and values:
                wet = [v for v in values if v >= WET_DAY_THRESHOLD_MM]
                n_wet = len(wet)
                if wet:
                    wet_pct = percentile_thresholds(wet, level=50)
                    p10_wet, p50_wet, p90_wet = (
                        wet_pct["p10"],
                        wet_pct["p50"],
                        wet_pct["p90"],
                    )

            normals.append(
                {
                    "zone_id": zone_id,
                    "month": month,
                    "variable": variable,
                    "p10": p10,
                    "p50": p50,
                    "p90": p90,
                    "n": len(values),
                    "mean": mean,
                    "p10_wet": p10_wet,
                    "p50_wet": p50_wet,
                    "p90_wet": p90_wet,
                    "n_wet": n_wet,
                    # Carried on the precipitation rows only: the threshold is what defines THIS
                    # row's wet subset, and putting 1.0 on a temperature row would imply a
                    # wet-day reading of a variable that has none.
                    "wet_day_threshold_mm": WET_DAY_THRESHOLD_MM if is_wet_day_variable else None,
                    "schema_version": NORMALS_SCHEMA_VERSION,
                    "window_start": window_start,
                    "window_end": window_end,
                    "window_label": window_label,
                    "lat": lat,
                    "lon": lon,
                    "wkt": wkt,
                }
            )
    return normals


def classify_against_normals(value: float, *, p10: float, p50: float, p90: float) -> dict:
    """Classify one observed value against P10/P50/P90 climatological normals.

    Raises ``ValueError`` on inconsistent percentiles, and on a number past the float range
    (whose ``float()`` raises ``OverflowError``), so callers handle both the same way (#221).
    """
    try:
        value = float(value)
        p10, p50, p90 = float(p10), float(p50), float(p90)
    except OverflowError as exc:
        raise ValueError("value and percentiles must be within the float range") from exc
    if not (p10 <= p50 <= p90):
        raise ValueError(f"Expected p10 <= p50 <= p90, got {p10}, {p50}, {p90}")
    if value < p10:
        category = "extreme_low"
        band = "<p10"
    elif value < p50:
        category = "below_normal"
        band = "p10_p50"
    elif value <= p90:
        category = "above_normal"
        band = "p50_p90"
    else:
        category = "extreme_high"
        band = ">p90"
    return {"category": category, "percentile_band": band, "vs_p50": round(value - p50, 4)}


# --------------------------------------------------------------------------- #
# Site-watch V0 §G2 — the read-only normals view and its per-day classification
# --------------------------------------------------------------------------- #

#: The two variables the V0 watch predicate classifies. `watch-input/v1`'s `entry` enum is
#: exactly these, so a normals row for any other variable is returned under ``normals[]`` and
#: never becomes a ``series[]`` entry — emitting one would produce a document the committed
#: schema rejects.
WATCH_SERIES_VARIABLES = ("precipitation", "wind_speed")

#: `reference_thin: true` below this. Thirty years is the WMO normal; anything thinner is a
#: percentile a reader should weigh differently, and the contract makes them say so.
REFERENCE_THIN_N = 30

#: The doctrine limits, carried in the output and not only in the design note (#172
#: acceptance): one plain-text entry per bullet of `docs/design/site-watch-v0.md` section
#: "Data limits", in the same order, each starting with that bullet's title. Plain on purpose:
#: this text is read by Deposium and by clients, so no markdown, file path or internal symbol.
#: The offline replay report copies the same tuple, so a limit that only holds for the live
#: watch (the 7-day request, the adapter-stamped retrieval time) is worded as one.
#: `test_the_caveats_state_the_design_note_data_limits_as_plain_text_entries` parses the titles
#: from the note, so a bullet added, renamed or dropped there fails until this tuple follows.
NORMALS_CAVEATS: tuple[str, ...] = (
    "Grid, not site: a forecast value describes a provider grid cell at one point per zone (the "
    "midpoint of the zone's bounding box in the live watch contract), of the order of 10 km or "
    "coarser depending on the model, not the site itself. There is no downscaling and no "
    "orographic correction.",
    "Daily aggregates only: values are daily; precipitation is a daily total and wind speed is "
    "the daily maximum of the 10 m wind. There is no hourly step.",
    "UTC days: every daily value is a UTC day. A French local day (CET/CEST) is not a UTC day.",
    "No gusts, no wind direction: neither is ingested. A wind vigilance is a gust criterion, so "
    "it cannot be reproduced here and wind is indicative only.",
    "Monthly normals on daily values: P10/P50/P90 of the 1991-2020 window by calendar month, "
    "compared with daily values. For precipitation the all-days P90 includes dry days and "
    "understates heavy rain, so the band is classified against the wet-day distribution (days "
    f"of at least {WET_DAY_THRESHOLD_MM} mm); the wet-day percentiles are a project-specific "
    "classification, not the Climdex R95p index.",
    "Horizon 7 days requested, served horizon reported: In live watch use, a forecast is "
    "requested for 7 days but a provider may return a shorter usable horizon; the forecast "
    "provider or adapter reports the served horizon and any message must use that served "
    "value. A replay instead reads archived forecasts, one per lead day from 1 to 7. The stored "
    "normals carry no horizon; it comes with the forecast.",
    "No measured skill: there is no measured skill behind this classification on the production "
    "model, so confidence is null, always, by construction.",
    "Two timestamps, no evidence id: In live watch use, the retrieval time of a forecast is "
    "stamped by the adapter after the fetch and the model initialisation time is null until the "
    "provider exposes it; in a replay nothing is fetched and the run time follows from the lead "
    "day. This output reports neither timestamp. Neither is an evidence identifier.",
)

#: Why an entry could not become a schema-valid `series[]` row. Each one is a state of the
#: STORED normals, not of the request, and each is reported rather than papered over with a
#: fallback percentile — a band computed from a distribution that does not exist is worse than
#: no band at all.
UNCLASSIFIED_REASONS = (
    "reference_unavailable",
    "wet_reference_unavailable",
    "reference_inconsistent",
    "variable_outside_watch_contract",
    "date_unparseable",
)


def _stored_int(value, default: int | None = None) -> int | None:
    """``int(value)`` for a value read out of a stored artifact, or ``default``.

    Stored columns are not a trusted source. A legacy or hand-edited file can hold a string,
    a float or a null where an integer is declared, and a bare ``int(row["month"])`` turns
    that into an uncaught exception from inside the response builder — past the read guard,
    so the tool crashes instead of returning its documented error (#203 review). A stored
    ``inf`` raises ``OverflowError`` rather than ``ValueError``, and gets the default too (#221).
    """
    if value is None or isinstance(value, bool):
        return default
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return default


def _normals_index(rows: list[dict]) -> dict[tuple[str, int], dict]:
    """Key the stored rows by ``(variable, month)``, skipping any that cannot be keyed.

    A row with no usable month is not a reference for any day, so it is dropped here rather
    than raising: the entries that wanted it then report ``reference_unavailable``, which is
    the honest answer, and the rest of the panel still classifies.
    """
    index: dict[tuple[str, int], dict] = {}
    for row in rows:
        variable = row.get("variable")
        month = _stored_int(row.get("month"))
        if not isinstance(variable, str) or not variable or month is None or not 1 <= month <= 12:
            continue
        index[(variable, month)] = row
    return index


def _finite(value) -> float | None:
    """``float(value)`` when it is a real number, else ``None``.

    A stored NULL and a stored NaN mean the same thing here — the bucket produced no
    percentile — and the classifier must not be handed either. Neither may an integer beyond
    the float range, whose ``float()`` raises ``OverflowError`` (#221).
    """
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return numeric if math.isfinite(numeric) else None


def _watch_entry_skeleton(variable: str, day: date, normal: dict) -> dict:
    pv = provider_var(variable)
    n = _stored_int(normal.get("n"), 0) or 0
    entry = {
        "variable": variable,
        "provider_variable": pv.provider,
        "unit": pv.unit,
        "day": day.isoformat(),
        "n": n,
        # A judgement the provider owes the reader, not one the reader has to recompute.
        "reference_thin": n < REFERENCE_THIN_N,
    }
    for key in ("p10", "p50", "p90"):
        bound = _finite(normal.get(key))
        if bound is not None:
            entry[key] = bound
    return entry


def classify_watch_entry(
    *, variable: str, day: date, value: float | None, normal: dict
) -> tuple[dict | None, str | None]:
    """One `watch-input/v1` ``series[]`` entry, or ``(None, reason)`` when none is possible.

    The month is NOT a parameter: the caller derives it from ``day`` and hands in that month's
    normal, which is what makes a horizon crossing a month boundary classify day by day
    against the right reference instead of all of it against one month (§G2).

    Precipitation is banded against the WET-day triplet, never the all-days one, and never a
    mix of the two: the all-days P90 of a variable that is zero on most days sits far below
    the heavy-rain threshold a reader would assume it marks. A day under
    ``wet_day_threshold_mm`` is ``dry`` — the value is kept, the band is not invented, and
    dry spells stay a V1 derived variable.

    Returns ``(None, reason)`` rather than falling back when the stored reference cannot
    support a band. The case that matters is a genuinely arid month: ``n_wet = 0`` is valid
    data (niamey, January, over 1991-2020), so the wet triplet is NULL and a forecast day at
    5 mm has no wet-day distribution to be placed in. Banding it against the all-days
    percentiles instead would answer a different question while looking like an answer to
    this one.
    """
    if variable not in WATCH_SERIES_VARIABLES:
        return None, "variable_outside_watch_contract"
    entry = _watch_entry_skeleton(variable, day, normal)
    is_precipitation = variable == _WET_DAY_VARIABLE
    threshold = _finite(normal.get("wet_day_threshold_mm")) if is_precipitation else None
    if is_precipitation:
        for key in ("p10_wet", "p50_wet", "p90_wet"):
            bound = _finite(normal.get(key))
            if bound is not None:
                entry[key] = bound
        n_wet = _stored_int(normal.get("n_wet"))
        if n_wet is not None:  # a v1 artifact stores no wet-day count; never invent a zero
            entry["n_wet"] = n_wet
        if threshold is not None:
            entry["wet_day_threshold_mm"] = threshold

    if value is None:
        # `missing` forbids value/percentile_band/category in the schema, and the contract is
        # explicit that a gap is never a zero: a zero would classify as a real dry day.
        entry["status"] = "missing"
        return entry, None

    numeric = _finite(value)
    if numeric is None:
        entry["status"] = "missing"
        return entry, None

    entry["value"] = numeric
    if is_precipitation:
        if threshold is None:
            return None, "reference_unavailable"
        if numeric < threshold:
            entry["status"] = "dry"
            return entry, None
        bounds = (entry.get("p10_wet"), entry.get("p50_wet"), entry.get("p90_wet"))
        if any(bound is None for bound in bounds):
            return None, "wet_reference_unavailable"
    else:
        bounds = (entry.get("p10"), entry.get("p50"), entry.get("p90"))
        if any(bound is None for bound in bounds):
            return None, "reference_unavailable"

    p10, p50, p90 = bounds
    try:
        # The SAME evaluator the operator tool and the offline replay use. Neither Deposium
        # nor a replay script gets a second scientific formula (§G2, §G3).
        classified = classify_against_normals(numeric, p10=p10, p50=p50, p90=p90)
    except ValueError:
        return None, "reference_inconsistent"
    entry["status"] = "available"
    entry["percentile_band"] = classified["percentile_band"]
    entry["category"] = classified["category"]
    return entry, None


def build_watch_series(values: list[dict], *, normals: list[dict]) -> tuple[list[dict], list[dict]]:
    """``(series, unclassified)`` for a list of ``(variable, date, value)`` entries."""
    index = _normals_index(normals)
    series: list[dict] = []
    unclassified: list[dict] = []
    for item in values:
        variable = str(item.get("variable") or "")
        raw_day = str(item.get("date") or "")
        try:
            day = date.fromisoformat(raw_day)
        except ValueError:
            unclassified.append(
                {"variable": variable, "date": raw_day, "reason": "date_unparseable"}
            )
            continue
        # The month comes from THIS entry's day. A 7-day horizon spanning 31 Jan -> 2 Feb
        # reads January's normals for the first days and February's for the rest.
        normal = index.get((variable, day.month))
        if normal is None:
            unclassified.append(
                {
                    "variable": variable,
                    "date": day.isoformat(),
                    "reason": (
                        "variable_outside_watch_contract"
                        if variable not in WATCH_SERIES_VARIABLES
                        else "reference_unavailable"
                    ),
                }
            )
            continue
        entry, reason = classify_watch_entry(
            variable=variable, day=day, value=item.get("value"), normal=normal
        )
        if entry is None:
            unclassified.append({"variable": variable, "date": day.isoformat(), "reason": reason})
            continue
        series.append(entry)
    return series, unclassified


def _read_normals_rows(src: str, *, zone_id: str, variables: list[str] | None) -> list[dict]:
    """The stored monthly rows for a zone, v1 and v2 alike.

    ``SELECT *`` and not a pinned column list: a v1 artifact has no wet-day columns at all, so
    naming them would raise on exactly the files the version-aware path exists to keep
    readable. The caller reads what is present and reports the rest as absent.
    """
    _validate_dst(src)
    with duckdb_session(":memory:", load_httpfs=src.startswith("s3://")) as conn:
        sql = f"SELECT * FROM read_parquet('{src}') WHERE zone_id = ?"  # noqa: S608 - _validate_dst
        params: list = [zone_id]
        if variables:
            placeholders = ", ".join("?" for _ in variables)
            sql += f" AND variable IN ({placeholders})"  # noqa: S608 - bound parameters
            params.extend(variables)
        sql += " ORDER BY variable, month"
        rows = conn.execute(sql, params).fetchall()
        names = [d[0] for d in conn.description]
    return [
        {key: value for key, value in zip(names, row, strict=True) if key != "geometry"}
        for row in rows
    ]


def _read_bound_normals(
    src: str, *, zone_id: str, variables: list[str] | None
) -> NormalsArtifactRead:
    """Serving-only reader; retain the operator's unbound row reader separately."""
    _validate_dst(src)
    return read_normals_artifact(src, zone_id=zone_id, variables=variables)


def _normals_reference(rows: list[dict]) -> dict:
    """The `watch-input/v1` ``reference`` block, and nothing more than that block minus the
    evidence id.

    The schema's ``reference`` object is `additionalProperties: false`, so this returns exactly
    ``kind``, ``window`` (year span), ``grain`` and ``source`` — never ``window_start``/``window_end``,
    which are dates, not part of that block, and live in ``metadata.normals_window`` instead
    (#218, #172).

    ``evidence_id`` is NOT produced here. A real `geoai_ev_...` is a `dataset.describe` over a
    registered dataset with a server-owned evidence context. An authorized client
    supplies it after authorization and exact read binding. Emitting a
    plausible-looking identifier computed inside the tool would be worse than omitting it: it
    would satisfy the schema's pattern while citing nothing a reader can resolve.
    """
    window_dates = _normals_window_dates(rows)
    if window_dates is None:
        raise ValueError(
            "incomplete stored normals window: window_start and window_end are required"
        )
    reference = {
        "kind": "climatological_percentiles",
        "grain": _NORMAL_GRAIN,
        "source": "climate_normals",
    }
    if window_dates:
        start, end = window_dates["start"], window_dates["end"]
        error = _validate_date_range(start, end)
        if error:
            raise ValueError(error)
        # Storage keeps the scientific label (WMO_1991_2020 or custom). The watch
        # contract needs years; the exact dates remain in metadata.normals_window.
        reference["window"] = (
            f"{date.fromisoformat(start).year:04d}-{date.fromisoformat(end).year:04d}"
        )
    return reference


def _normals_window_dates(rows: list[dict]) -> dict | None:
    """``metadata.normals_window``: the stored window's dates, absent from ``reference`` (#172).

    ``None`` when either date is missing. ``_normals_reference`` refuses that incomplete
    window through the tool's existing error path instead of publishing a reference without it.
    """
    start = next((str(row["window_start"]) for row in rows if row.get("window_start")), None)
    end = next((str(row["window_end"]) for row in rows if row.get("window_end")), None)
    if start and end:
        return {"start": start, "end": end}
    return None


def _normals_refusal(message: str) -> dict:
    """An ``{"error": ...}`` return that still carries ``confidence: null`` (#172).

    A refusal is not a finding, but the key stays present on every handler return so no consumer meets a
    payload where the absence of ``confidence`` could be filled with a default.
    """
    return {"error": message, "confidence": None}


def _normals_provenance() -> dict:
    """``metadata.provenance``: the §G3 provenance the tool can state, and the rest as null.

    ``attribution`` is the runtime data attribution every tool here already publishes as
    ``license`` (#99). The forecast-side fields (``provider_chain``, ``horizon_days_requested``,
    ``horizon_days_served``, ``model_run_at``, ``retrieved_at``) describe a forecast fetch. The
    tool fetches none: ``values[]`` come from the caller, so it cannot know which provider
    served them, over which horizon, or when. Deriving a horizon from ``len(values)`` would pass
    the caller's selection off as the provider's served horizon; they stay ``None``. Where each
    one really comes from: ``provider_chain`` and the served horizon from ``forecast_query``
    metadata (``metadata.provider_chain``, ``metadata.horizon_days``); the requested horizon is
    the adapter's own request; ``retrieved_at`` is stamped by the adapter after the fetch;
    ``model_run_at`` stays null until a provider exposes it (``forecast_query`` returns only
    ``issued_at``). Path: ``metadata.provenance.forecast`` (site-watch V0, "Data limits").
    """
    return {
        "attribution": OPEN_METEO_DATA_LICENSE,
        "forecast": {
            "provider_chain": None,
            "horizon_days_requested": None,
            "horizon_days_served": None,
            "model_run_at": None,
            "retrieved_at": None,
            "reason": (
                "values[] are supplied by the caller and climate_normals fetches no forecast, "
                "so it cannot know which provider served them, over which horizon, or when. "
                "provider_chain and horizon_days_served come from forecast_query metadata; "
                "horizon_days_requested is the adapter's own request; retrieved_at is stamped "
                "by the adapter after the fetch; model_run_at stays null until a provider "
                "exposes it."
            ),
        },
    }


def _normals_coverage(dst: str, *, zone_id: str) -> dict:
    """Return per-variable month coverage + the stored window for a normals parquet, or {} on an
    unreadable schema. ``n_months`` counts only months with a NON-NULL ``p50`` — every (variable,
    month) pair is materialized even for empty buckets, so a plain ``COUNT(DISTINCT month)`` would be
    a tautological 12 and let a sparse/partial file pass the idempotent-skip as 'covered' (review P2).
    It also counts only a month that is one of 1..12: a stored ``inf``, ``NaN``, 0 or 11.5 must not
    make up the twelfth month of a file the skip then trusts (#221).

    Version-aware, and DEFENSIVELY so (#172). ``schema_version`` is read only when the column is
    actually present, because selecting it unconditionally would raise on every v1 artifact, return
    ``{}``, and make the idempotent-skip treat a perfectly good file as absent — re-running the
    Archive fetch to regenerate normals, which is the one thing the normals-only operator path
    exists to avoid. A v1 file reports ``schema_version: 1``.

    The wet-day columns are REPORTED and never gated on: see
    :func:`_normals_coverage_satisfies` for why ``n_wet = 0`` must not fail a coverage check."""
    _validate_dst(dst)
    with duckdb_session(":memory:", load_httpfs=dst.startswith("s3://")) as conn:
        try:
            stored_columns = {
                row[0]
                for row in conn.execute(
                    f"DESCRIBE SELECT * FROM read_parquet('{dst}')"  # noqa: S608 - _validate_dst
                ).fetchall()
            }
            rows = conn.execute(
                # Only a month in 1..12 counts (#221). TRY_CAST keeps a VARCHAR month readable.
                "SELECT variable, COUNT(DISTINCT month) FILTER (WHERE p50 IS NOT NULL AND "
                "TRY_CAST(month AS DOUBLE) IN (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12)) AS n_months, "
                "COUNT(*) AS n_rows "
                f"FROM read_parquet('{dst}') WHERE zone_id = ? GROUP BY variable ORDER BY variable",
                [zone_id],
            ).fetchall()
            col_names = [d[0] for d in conn.description]
            by_variable = {
                row[0]: dict(zip(col_names[1:], row[1:], strict=True))
                for row in rows
                if row and row[0] is not None
            }
            total_rows = conn.execute(
                f"SELECT COUNT(*) FROM read_parquet('{dst}') WHERE zone_id = ?", [zone_id]
            ).fetchone()[0]
            win = conn.execute(
                "SELECT window_start, window_end, window_label "
                f"FROM read_parquet('{dst}') WHERE zone_id = ? LIMIT 1",
                [zone_id],
            ).fetchone()
            window = {"start": win[0], "end": win[1], "label": win[2]} if win else {}
            schema_version = 1
            if "schema_version" in stored_columns:
                stored = conn.execute(
                    # MIN, not MAX: a file carrying rows from two versions is as old as its oldest
                    # row. COALESCE inside the MIN, because SQL MIN SKIPS nulls — a file with legacy
                    # NULL rows beside v2 ones reported 2, which is the opposite of what the MIN was
                    # put there to guarantee (Greptile P2, PR #189). A NULL version IS v1: it is
                    # what a row written before the column existed looks like once the column does.
                    "SELECT MIN(COALESCE(schema_version, 1)) "  # noqa: S608 - _validate_dst
                    f"FROM read_parquet('{dst}') WHERE zone_id = ?",
                    [zone_id],
                ).fetchone()
                schema_version = stored[0] if stored and stored[0] is not None else 1
            wet_months = None
            if "n_wet" in stored_columns:
                wet = conn.execute(
                    "SELECT COUNT(*) FILTER (WHERE n_wet IS NOT NULL), "  # noqa: S608
                    "COUNT(*) FILTER (WHERE n_wet = 0) "
                    f"FROM read_parquet('{dst}') WHERE zone_id = ?",
                    [zone_id],
                ).fetchone()
                wet_months = {"rows_with_n_wet": wet[0], "rows_with_no_wet_day": wet[1]}
            return {
                "variables": by_variable,
                "n_rows": total_rows,
                "window": window,
                "schema_version": schema_version,
                "wet_day": wet_months,
            }
        except Exception:
            return {}


def _normals_coverage_satisfies(coverage: dict, requested: list[str]) -> bool:
    """True iff every requested variable has all 12 calendar months WITH data (non-null p50).

    Gated on the ALL-DAYS ``p50`` alone, and deliberately not on anything wet-day (#172). A gate
    that also demanded a non-null ``p90_wet`` would reject `niamey`, whose January, February,
    November and December are 0.0 mm across the entire 1991-2020 window: there is no wet-day
    percentile to compute because there were no wet days, and that is valid data rather than a
    coverage hole (`docs/CONSOLIDATION_ROADMAP.md` "Arid-zone trap"). The wet-day counts live in
    the coverage dict for a reader to inspect; nothing decides on them here.

    The schema version is likewise reported and not gated: an existing v1 artifact still satisfies
    the all-days contract it was written for, so this returns True for it rather than forcing an
    Archive refetch. Promoting v1 normals to v2 is the normals-only operator path's job, with the
    explicit promotion/rollback that path is required to have — not a side effect of a coverage
    check."""
    by_variable = coverage.get("variables") or {}
    return all((by_variable.get(v) or {}).get("n_months") == 12 for v in requested)


def _normals_window_matches(coverage: dict, start_date: str, end_date: str) -> bool:
    """True iff the stored normals were computed over the requested resolved window — so the
    idempotent-skip never returns normals from a *different* window under the new request's labels."""
    window = coverage.get("window") or {}
    return window.get("start") == start_date and window.get("end") == end_date


def _read_normals_records(path: str, *, zone_id: str) -> list[dict]:
    """Read a normals WKB parquet into row dicts for returning/classifying."""
    _validate_dst(path)
    with duckdb_session(":memory:", load_httpfs=path.startswith("s3://")) as conn:
        res = conn.execute(
            "SELECT * EXCLUDE (geometry), ST_AsText(ST_GeomFromWKB(geometry)) AS geometry_wkt "
            f"FROM read_parquet('{path}') WHERE zone_id = ? ORDER BY variable, month",
            [zone_id],
        ).fetchall()
        col_names = [d[0] for d in conn.description]
        return [dict(zip(col_names, row, strict=True)) for row in res]


def _classify_observed_values(
    normals: list[dict], *, observed: dict[str, float] | None, month: int | None
) -> dict | None:
    """Classify optional observed values against matching month normals."""
    if observed is None:
        return None
    if month is None:
        return {"error": "classify_month is required when classify_observed is provided"}
    by_key = {(row.get("variable"), row.get("month")): row for row in normals}
    out = {}
    for variable, value in observed.items():
        normal = by_key.get((variable, month))
        if normal is None:
            out[variable] = {"error": f"No normals for variable={variable!r}, month={month}"}
            continue
        if normal.get("p50") is None or not normal.get("n"):
            # Every (variable, month) is materialized, so an empty bucket exists with null
            # percentiles — surface a clear no-data message rather than a cryptic float() error.
            out[variable] = {
                "error": f"No climatology data for variable={variable!r}, month={month} (n=0)",
                "n": normal.get("n", 0),
            }
            continue
        try:
            out[variable] = classify_against_normals(
                value, p10=normal["p10"], p50=normal["p50"], p90=normal["p90"]
            )
            out[variable]["value"] = value
            out[variable]["month"] = month
        except (TypeError, ValueError) as e:
            out[variable] = {"error": str(e)}
    return out


_NORMALS_INTERNAL_COLS = ("wkt", "geometry_wkt", "window_start", "window_end", "window_label")


def _public_normals(records: list[dict]) -> list[dict]:
    """Strip internal columns (WKT geometry + the per-row window stamp) from returned normals rows —
    the window is surfaced once in ``metadata`` (resolved_*/window_label), not repeated per row."""
    return [{k: v for k, v in row.items() if k not in _NORMALS_INTERNAL_COLS} for row in records]


def _normals_month_count(records: list[dict]) -> int:
    """Minimum per-variable count of months with non-null P50.

    A month is read like :func:`_normals_index` reads it: a stored value that is not a month
    in 1..12 (``inf``, ``NaN``, 13) counts for nothing instead of raising (#221).
    """
    months_by_var: dict[str, set[int]] = {}
    for row in records:
        if row.get("p50") is None:
            continue
        variable = row.get("variable")
        month = _stored_int(row.get("month"))
        if variable is None or month is None or not 1 <= month <= 12:
            continue
        months_by_var.setdefault(str(variable), set()).add(month)
    if not months_by_var:
        return 0
    return min(len(months) for months in months_by_var.values())


# --------------------------------------------------------------------------- #
# Shared validation
# --------------------------------------------------------------------------- #


def _validate_date_range(start_date: str, end_date: str) -> str | None:
    """ISO-8601, start ≤ end, no future dates. Returns an error message or None."""
    try:
        s = date.fromisoformat(start_date)
        e = date.fromisoformat(end_date)
    except ValueError:
        return f"start_date/end_date must be ISO-8601 YYYY-MM-DD (got {start_date!r}, {end_date!r})"
    if e < s:
        return f"end_date {end_date} precedes start_date {start_date}"
    if e > datetime.now(UTC).date():
        return f"end_date {end_date} is in the future (ERA5 archive is historical)"
    return None


def _percentiles_metadata(
    *,
    resolved_start: str,
    resolved_end: str,
    window_label: str,
    source: str,
    era5_src: str | None,
    source_fallback_reason: str | None,
    requested: list[str],
    t0: float,
    extra: dict | None = None,
) -> dict:
    """Build the shared ``climate_percentiles`` provenance block.

    Called identically by the idempotent-skip return and the fresh-success return so the
    metadata shape can no longer drift between the two paths (audit R5). ``extra`` carries
    the skip-only keys (``skipped``, ``stored_coverage``)."""
    metadata = {
        "resolved_start_date": resolved_start,
        "resolved_end_date": resolved_end,
        "window_label": window_label,
        "source": source,
        "era5_source": era5_src,
        "source_fallback_reason": source_fallback_reason,
        "normal_grain": _NORMAL_GRAIN,
        "percentile_method": "numpy_linear_p10_p50_p90",
        "percentile_levels": [10, 50, 90],
        "provider_map": _provider_map(requested),
        "license": OPEN_METEO_DATA_LICENSE,
        "elapsed_ms": round((time.monotonic() - t0) * 1000),
    }
    if extra:
        metadata.update(extra)
    return metadata


# --------------------------------------------------------------------------- #
# Tool registration
# --------------------------------------------------------------------------- #


def register_climate_tools(mcp):
    """Register the climate ingestion + enrichment tools with the MCP server."""

    @mcp.tool(output_schema=output_schema_for("era5_ingest"))
    async def era5_ingest(
        zone_id: str,
        start_date: str,
        end_date: str,
        variables: list[str] | None = None,
        temporal: str = "daily",
        overwrite: bool = False,
        write_output: bool = True,
        output_path: str | None = None,
    ) -> dict:
        """Ingest ERA5 daily reanalysis for a zone via the Open-Meteo Archive API (FIERCE B2).

        Resolves the zone centroid from zones/zones.geojson, fetches the six terra-cognita §5.1
        daily variables (``models=era5`` forced), normalizes units, and writes a WKB GeoParquet
        (``climate/era5/{zone_id}.parquet``) readable by DuckDB Spatial.

        Args:
            zone_id: A zone identifier present in zones/zones.geojson (e.g. "montpellier").
            start_date: ISO-8601 "YYYY-MM-DD" (inclusive).
            end_date: ISO-8601 "YYYY-MM-DD" (inclusive, not in the future).
            variables: Normalized variable names (default: all six §5.1 variables).
            temporal: Only "daily" this iteration (the ERA5 daily grid).
            overwrite: If False (default), skip when the target parquet already exists.
            write_output: If False, fetch + report without writing the parquet.
            output_path: Explicit destination (else derived from S3_BUCKET).

        Returns:
            Ingestion summary: n_days, variables, output path, a completeness report, and stats.
        """
        check_input(
            Era5IngestInput,
            zone_id=zone_id,
            start_date=start_date,
            end_date=end_date,
            variables=variables,
            temporal=temporal,
            overwrite=overwrite,
            write_output=write_output,
            output_path=output_path,
        )
        log.info("era5_ingest", zone_id=zone_id, start_date=start_date, end_date=end_date)
        t0 = time.monotonic()

        # De-duplicate (preserve order) so a repeated name can't double a series / inflate
        # completeness past 100% / collide the staging column.
        requested = list(dict.fromkeys(variables)) if variables else list(NORMALIZED_VARS)
        unknown = [v for v in requested if v not in NORMALIZED_VARS]
        if unknown:
            return {"error": f"Unknown variables {unknown}; supported: {list(NORMALIZED_VARS)}"}

        date_err = _validate_date_range(start_date, end_date)
        if date_err:
            return {"error": date_err}

        zone = _resolve_zone(zone_id)
        if zone is None:
            return {"error": f"Unknown zone_id {zone_id!r} (not in zones/zones.geojson)"}
        lon, lat = zone["lon"], zone["lat"]

        dst = output_path or _default_output_uri(zone_id, "era5")
        if write_output and not dst:
            return {
                "error": "No output destination — set S3_BUCKET or pass output_path",
            }
        # Validate the (agent-controlled) destination once, gracefully — before it reaches the
        # unwrapped _output_exists probe, which would otherwise raise instead of returning an error.
        if write_output and dst:
            try:
                _validate_dst(dst)
            except ValueError as e:
                return {"error": f"Invalid output_path: {e}"}

        # Idempotent skip — ONLY when the stored parquet actually covers the requested range +
        # variables (else a wider/different re-run with overwrite=False would silently keep the old
        # slice). Echo the stored coverage so the consumer sees what is on disk. A non-covering
        # existing file is a hard error advising overwrite=True (rather than a misleading skip).
        exists = (
            await asyncio.to_thread(_output_exists, dst)
            if (write_output and dst and not overwrite)
            else False
        )
        if exists:
            stored = await asyncio.to_thread(
                _stored_coverage,
                dst,
                zone_id=zone_id,
                start_date=start_date,
                end_date=end_date,
            )
            if not _coverage_satisfies(stored, start_date, end_date, requested):
                return {
                    "error": (
                        f"{dst} exists but its stored coverage "
                        f"({stored.get('start')}..{stored.get('end')}, "
                        f"{len(stored.get('columns') or [])} cols) does not cover the requested "
                        f"{start_date}..{end_date} / {requested} — re-run with overwrite=True to replace"
                    )
                }
            completeness = await asyncio.to_thread(
                _stored_completeness,
                dst,
                zone_id=zone_id,
                start_date=start_date,
                end_date=end_date,
                requested=requested,
            )
            return {
                "zone_id": zone_id,
                "start_date": stored.get("start"),
                "end_date": stored.get("end"),
                "n_days": stored.get("n_days", 0),  # keep the top-level shape across all paths
                "output": dst,
                "variables": requested,
                "completeness": completeness,
                "metadata": {
                    "skipped": "output exists and covers the request (overwrite=False) — not re-fetched",
                    "requested_start": start_date,
                    "requested_end": end_date,
                    "stored_coverage": stored,
                    "elapsed_ms": round((time.monotonic() - t0) * 1000),
                },
            }

        try:
            raws = []
            for chunk_start, chunk_end in split_date_range(start_date, end_date):
                raw = await asyncio.to_thread(
                    fetch_archive_daily,
                    latitude=lat,
                    longitude=lon,
                    start_date=chunk_start,
                    end_date=chunk_end,
                    normalized_variables=requested,
                )
                raws.append(raw)
            dates, columns = normalize_archive_response(raws, requested)
        except Exception as e:
            log.error("era5_ingest_fetch_error", zone_id=zone_id, error=str(e))
            return {"error": f"Open-Meteo Archive fetch/normalize failed: {e}"}

        if not dates:
            return {"error": "Open-Meteo returned no daily records for the requested range"}

        wkt = f"POINT({lon} {lat})"
        records = []
        for i, d in enumerate(dates):
            rec = {"date": d, "zone_id": zone_id, "lat": lat, "lon": lon, "wkt": wkt}
            for v in requested:
                rec[v] = columns[v][i]
            records.append(rec)

        written = None
        if write_output and dst:
            staging = [
                ("date", "VARCHAR"),
                ("zone_id", "VARCHAR"),
                ("lat", "DOUBLE"),
                ("lon", "DOUBLE"),
                ("wkt", "VARCHAR"),
                *[(v, "DOUBLE") for v in requested],
            ]
            try:
                await asyncio.to_thread(_write_geoparquet, records, columns=staging, dst=dst)
                written = dst
            except Exception as e:
                log.error("era5_ingest_write_error", zone_id=zone_id, error=str(e))
                return {"error": f"GeoParquet write failed: {e}"}

        return {
            "zone_id": zone_id,
            "start_date": start_date,
            "end_date": end_date,
            "n_days": len(dates),
            "variables": requested,
            "output": written,
            "completeness": _completeness_report(dates, columns, start_date, end_date),
            "statistics": _value_stats(columns),
            "metadata": {
                "lat": lat,
                "lon": lon,
                "models": "era5",
                "temporal": temporal,
                "source": "open-meteo-archive",
                "license": OPEN_METEO_DATA_LICENSE,
                "provider_map": _provider_map(requested),
                "elapsed_ms": round((time.monotonic() - t0) * 1000),
            },
        }

    @mcp.tool(output_schema=output_schema_for("ndvi_ingest"))
    async def ndvi_ingest(
        zone_id: str,
        start_date: str,
        end_date: str,
        cloud_cover_max: int = settings.DEFAULT_CLOUD_COVER_MAX,
        monthly: bool = True,
        write_output: bool = True,
        output_path: str | None = None,
    ) -> dict:
        """Materialize a standalone Sentinel-2 NDVI zonal-stats time series for a zone (FIERCE B3).

        Streams the red + NIR COGs per Sentinel-2 scene via GDAL /vsicurl/ (no COG storage), masks
        to the zone geometry, and writes a per-zone WKB GeoParquet (``climate/ndvi/{zone_id}.parquet``)
        keyed on ``zone_id``. Unlike ``climate_enrich_ndvi`` (B4) this needs no ERA5 source and does
        no join — it is the NDVI series on its own.

        Args:
            zone_id: A zone identifier present in zones/zones.geojson.
            start_date: ISO-8601 "YYYY-MM-DD" (inclusive).
            end_date: ISO-8601 "YYYY-MM-DD" (inclusive, not in the future).
            cloud_cover_max: Max scene cloud cover %.
            monthly: True (default) keeps the least-cloudy scene per month — the tractable cadence
                for a multi-year backfill; False collects every scene in range up to the STAC cap.
            write_output: If False, collect + return without writing the parquet.
            output_path: Explicit destination (else climate/ndvi/{zone_id}.parquet from S3_BUCKET).

        Returns:
            The per-scene NDVI observations + the materialized output path + stats.
        """
        check_input(
            NdviIngestInput,
            zone_id=zone_id,
            start_date=start_date,
            end_date=end_date,
            cloud_cover_max=cloud_cover_max,
            monthly=monthly,
            write_output=write_output,
            output_path=output_path,
        )
        log.info("ndvi_ingest", zone_id=zone_id, start_date=start_date, end_date=end_date)
        t0 = time.monotonic()

        date_err = _validate_date_range(start_date, end_date)
        if date_err:
            return {"error": date_err}

        zone = _resolve_zone(zone_id)
        if zone is None:
            return {"error": f"Unknown zone_id {zone_id!r} (not in zones/zones.geojson)"}

        # Resolve + validate the destination EARLY (mirrors era5_ingest) so an invalid output_path
        # fails fast — before the expensive multi-month COG streaming — with the right message.
        dst = output_path or _default_output_uri(zone_id, "ndvi")
        if write_output and not dst:
            return {"error": "No output destination — set S3_BUCKET or pass output_path"}
        if write_output and dst:
            try:
                _validate_dst(dst)
            except ValueError as e:
                return {"error": f"Invalid output_path: {e}"}

        try:
            if monthly:
                searched, observations = await _collect_ndvi_monthly(
                    zone["bbox"], zone["geometry"], start_date, end_date, cloud_cover_max
                )
            else:
                searched, observations = await _collect_ndvi_observations(
                    zone["bbox"], f"{start_date}/{end_date}", zone["geometry"], cloud_cover_max
                )
        except Exception as e:
            log.error("ndvi_ingest_collect_error", zone_id=zone_id, error=str(e))
            return {"error": f"NDVI collection failed: {e}"}

        lon, lat = zone["lon"], zone["lat"]
        records = [
            {
                "date": (obs.get("date") or "")[:10],  # scene datetime → ISO date
                "zone_id": zone_id,
                "lat": lat,
                "lon": lon,
                "wkt": f"POINT({lon} {lat})",
                "ndvi_mean": obs.get("ndvi_mean"),
                "ndvi_std": obs.get("ndvi_std"),
                "ndvi_min": obs.get("ndvi_min"),
                "ndvi_max": obs.get("ndvi_max"),
                "cloud_cover": obs.get("cloud_cover"),
                "item_id": obs.get("item_id"),
            }
            for obs in observations
        ]

        staging = [
            ("date", "VARCHAR"),
            ("zone_id", "VARCHAR"),
            ("lat", "DOUBLE"),
            ("lon", "DOUBLE"),
            ("wkt", "VARCHAR"),
            ("ndvi_mean", "DOUBLE"),
            ("ndvi_std", "DOUBLE"),
            ("ndvi_min", "DOUBLE"),
            ("ndvi_max", "DOUBLE"),
            ("cloud_cover", "DOUBLE"),
            ("item_id", "VARCHAR"),
        ]

        written = None
        if write_output and dst:
            if not records:
                return {
                    "error": f"No NDVI observations for {zone_id!r} in {start_date}..{end_date}"
                }
            try:
                await asyncio.to_thread(_write_geoparquet, records, columns=staging, dst=dst)
                written = dst
            except Exception as e:
                log.error("ndvi_ingest_write_error", zone_id=zone_id, error=str(e))
                return {"error": f"NDVI GeoParquet write failed: {e}"}

        means = [r["ndvi_mean"] for r in records if r["ndvi_mean"] is not None]
        meta = {
            "start_date": start_date,
            "end_date": end_date,
            "cadence": "monthly" if monthly else "all-scenes",
            "cloud_cover_max": cloud_cover_max,
            "months_or_scenes_searched": searched,
            "source": "sentinel-2-l2a",
            "streaming": "STAC/COG via GDAL /vsicurl/ (no COG storage)",
            "elapsed_ms": round((time.monotonic() - t0) * 1000),
        }
        if not monthly and searched >= settings.STAC_SEARCH_MAX_ITEMS:
            # An all-scenes search is capped, so reaching the boundary cannot prove full coverage.
            meta["ndvi_truncation_warning"] = (
                f"NDVI search hit the {settings.STAC_SEARCH_MAX_ITEMS}-item STAC cap — observations "
                "may be truncated; narrow the date range for full coverage"
            )
        return {
            "zone_id": zone_id,
            "observations": records,
            "n_observations": len(records),
            "output": written,
            "statistics": {
                "ndvi_mean_avg": round(sum(means) / len(means), 4) if means else None,
                "ndvi_mean_min": round(min(means), 4) if means else None,
                "ndvi_mean_max": round(max(means), 4) if means else None,
                "first_date": records[0]["date"] if records else None,
                "last_date": records[-1]["date"] if records else None,
            },
            "metadata": meta,
        }

    @mcp.tool(output_schema=output_schema_for("climate_enrich_ndvi"))
    async def climate_enrich_ndvi(
        zone_id: str,
        start_date: str,
        end_date: str,
        tolerance_days: int = 3,
        cloud_cover_max: int = settings.DEFAULT_CLOUD_COVER_MAX,
        era5_source: str | None = None,
        write_output: bool = True,
        output_path: str | None = None,
        # APPENDED, not slotted in beside `era5_source` where it reads better. Inserting it there
        # rebound a positional caller's `write_output` onto this parameter and its `output_path`
        # onto `write_output` — silently, since both are truthy (Copilot, PR #188). Nothing in the
        # repo calls this positionally, but a signature is a contract with callers that are not in
        # the repo. Every later parameter is appended for the same reason.
        ndvi_source: str | None = None,
        strict_window: bool = False,
    ) -> dict:
        """Align NDVI onto the ERA5 daily grid + materialize the unified table (FIERCE B4).

        Loads the ERA5 GeoParquet (``era5_source``, else ``climate/era5/{zone_id}.parquet`` — run
        ``era5_ingest`` first), obtains the zone's NDVI profile, snaps each NDVI observation to its
        nearest ERA5 day within ±tolerance_days, and materializes
        ``climate/unified/{zone_id}.parquet`` (terra-cognita §5.3) — also returning the unified rows
        for ``discover_patterns``.

        Args:
            zone_id: A zone identifier present in zones/zones.geojson.
            start_date: ISO-8601 "YYYY-MM-DD" (inclusive).
            end_date: ISO-8601 "YYYY-MM-DD" (inclusive, not in the future).
            tolerance_days: Nearest-date NDVI↔ERA5 join window (±N days; proposed, ADR 0002 §3).
            cloud_cover_max: Max cloud cover % for NDVI observations.
            era5_source: Explicit ERA5 GeoParquet source (else derived from S3_BUCKET).
            ndvi_source: Explicit B3 NDVI GeoParquet (``climate/ndvi/{zone_id}.parquet``). When
                given, the stored series IS the input and NOTHING is collected — no STAC search,
                no COG read. A source that cannot be read, does not hold the zone, or has no rows
                in the window is an ERROR; it never degrades to collection, because a caller who
                named a durable artifact asked for that artifact and not for a fresh one that
                happens to resemble it. Omit it to keep the streaming behaviour (#177).
            strict_window: refuse the run when the ERA5 source does not cover
                ``[start_date, end_date]`` in full, instead of returning the intersection. Default
                False, which keeps "give me what you have" working; set it for a run whose output
                is going to be presented as covering the window it asked for (#133).
            write_output: If False, align + return without materializing the parquet.
            output_path: Explicit unified destination (else derived from S3_BUCKET).

        Returns:
            The unified (zone, date) rows + the materialized output path + stats.
        """
        check_input(
            ClimateEnrichNdviInput,
            zone_id=zone_id,
            start_date=start_date,
            end_date=end_date,
            tolerance_days=tolerance_days,
            cloud_cover_max=cloud_cover_max,
            era5_source=era5_source,
            ndvi_source=ndvi_source,
            strict_window=strict_window,
            write_output=write_output,
            output_path=output_path,
        )
        log.info(
            "climate_enrich_ndvi",
            zone_id=zone_id,
            start_date=start_date,
            end_date=end_date,
            ndvi_input="stored_b3_series" if ndvi_source else "stac_cog_stream",
        )
        t0 = time.monotonic()

        date_err = _validate_date_range(start_date, end_date)
        if date_err:
            return {"error": date_err}

        zone = _resolve_zone(zone_id)
        if zone is None:
            return {"error": f"Unknown zone_id {zone_id!r} (not in zones/zones.geojson)"}

        # Resolve + validate the (agent-controlled) destination EARLY — before the expensive ERA5
        # read and NDVI collection — so an invalid output_path fails fast with the right message
        # (mirrors era5_ingest), rather than wasting the fetch and surfacing a write error.
        dst = output_path or _default_output_uri(zone_id, "unified")
        if write_output and not dst:
            return {"error": "No output destination — set S3_BUCKET or pass output_path"}
        if write_output and dst:
            try:
                _validate_dst(dst)
            except ValueError as e:
                return {"error": f"Invalid output_path: {e}"}

        era5_src = era5_source or _default_output_uri(zone_id, "era5")
        if not era5_src:
            return {
                "error": "No ERA5 source — set S3_BUCKET or pass era5_source (run era5_ingest first)"
            }
        try:
            era5_rows = await asyncio.to_thread(_read_era5_records, era5_src)
        except Exception as e:
            return {
                "error": f"Could not read ERA5 parquet for {zone_id!r} — run era5_ingest first: {e}"
            }
        if not era5_rows:
            return {"error": f"ERA5 parquet for {zone_id!r} has no rows"}
        # era5_source may be a multi-zone/combined parquet — keep only the requested zone so
        # alignment never mixes one zone's NDVI with another zone's climate (Codex #37).
        era5_rows = [r for r in era5_rows if r.get("zone_id") == zone_id]
        if not era5_rows:
            return {"error": f"ERA5 source has no rows for zone {zone_id!r}"}

        # The strict gate, deliberately BEFORE the NDVI read or collection: a run that cannot
        # produce the window it was asked for should not first spend a STAC search or a parquet
        # read finding that out. It reuses `_era5_rows_cover_window` rather than comparing min/max
        # here, because that helper already requires FULL daily density — a source holding only the
        # first and last day of the window spans it without covering it, and a second hand-written
        # check is how two coverage rules drift apart.
        #
        # Off by default. Returning the intersection is legitimate — "give me what you have" is
        # most of the use — and the defect #133 surfaced was never that the intersection is
        # returned, it is that it used to be returned SILENTLY. That is now reported either way
        # (`ndvi_out_of_era5_range`, `era5_window`). This flag is for the other case: a run whose
        # output will be presented as covering the window it asked for, where a narrower table is
        # not a smaller answer but a wrong one.
        if strict_window:
            published = [v for v in NORMALIZED_VARS if v in era5_rows[0]]
            # CLIPPED first. `_era5_rows_cover_window` compares its distinct-day count against the
            # window length and says so in its own docstring — "rows are clipped to the window" —
            # so handing it the whole artifact breaks the precondition and inverts the guarantee:
            # a 1991-2020 source has 10,958 distinct days, which clears the density bar for ANY
            # window inside it, hole or no hole. A 2016 request missing three days in June would
            # have been accepted and stamped `strict` (Greptile P1, PR #191).
            in_window = _filter_era5_rows_for_window(
                era5_rows, zone_id=zone_id, start_date=start_date, end_date=end_date
            )
            if not _era5_rows_cover_window(
                in_window, start_date=start_date, end_date=end_date, requested=published
            ):
                stored = sorted(str(r.get("date"))[:10] for r in era5_rows if r.get("date"))
                span = f"{stored[0]}..{stored[-1]}" if stored else "empty"
                # BOTH numbers, because the two ways this fails need different answers and the
                # message has to say which one happened. The span alone described the artifact
                # rather than the gap: for a January request against a file running into the next
                # year it read "it holds 2024-01-01..2025-03-31 (61 distinct days)", which sounds
                # like ample coverage while the actual fault was a missing 10 January.
                present = len({d for d in stored if start_date <= d <= end_date})
                expected = (date.fromisoformat(end_date) - date.fromisoformat(start_date)).days + 1
                return {
                    "error": (
                        f"strict_window: the ERA5 source does not cover "
                        f"{start_date}..{end_date} for zone {zone_id!r} — {present} of "
                        f"{expected} days are present inside that window (the source spans "
                        f"{span}). Narrow the window, materialize the missing ERA5, or drop "
                        f"strict_window to accept the intersection."
                    ),
                    "era5_window": span,
                    "requested_window": f"{start_date}..{end_date}",
                    "days_present_in_window": present,
                    "days_requested": expected,
                }

        ndvi_coverage: dict | None = None
        # `is not None`, not truthiness. The contract now refuses an empty string, but this branch
        # must not DEPEND on that: `if ndvi_source:` treated a supplied `""` as omitted and called
        # the STAC collector, which is the no-fallback contract broken by the one value most likely
        # to arrive from an unset variable or an empty form field (Greptile P1 / Copilot, PR #188).
        if ndvi_source is not None:
            # The explicit-source path. `_collect_ndvi_observations` is not reached at all, and
            # that is the contract rather than an optimization: this branch must make no provider
            # call, so a failure here returns rather than falling through to the collector below.
            images_searched = 0
            try:
                ndvi_obs, ndvi_coverage = await asyncio.to_thread(
                    _read_ndvi_records,
                    ndvi_source,
                    zone_id=zone_id,
                    start_date=start_date,
                    end_date=end_date,
                )
            except Exception as e:
                # Deliberately broad, like the ERA5 read beside it. Three different failures reach
                # here and only two are subclasses of one another: `NdviSourceError` (the source is
                # not a usable B3 series), `ValueError` from `_validate_dst` (the location is
                # refused), and anything raised while OPENING the connection — notably
                # `S3EndpointConfigurationError`, a RuntimeError, which a misconfigured
                # `S3_ENDPOINT` raises before the inner try is entered. Catching only the first two
                # let that one escape as an unhandled exception instead of a tool error response
                # (Copilot, PR #188).
                log.error("climate_enrich_ndvi_source_error", zone_id=zone_id, error=str(e))
                return {"error": f"Invalid ndvi_source: {e}"}
        else:
            try:
                images_searched, ndvi_obs = await _collect_ndvi_observations(
                    zone["bbox"], f"{start_date}/{end_date}", zone["geometry"], cloud_cover_max
                )
            except Exception as e:
                log.error("climate_enrich_ndvi_collect_error", zone_id=zone_id, error=str(e))
                return {"error": f"NDVI collection failed: {e}"}

        unified = _align_ndvi_to_era5(era5_rows, ndvi_obs, tolerance_days=tolerance_days)

        written = None
        value_cols = [v for v in NORMALIZED_VARS if v in era5_rows[0]]
        staging = [
            ("date", "VARCHAR"),
            ("zone_id", "VARCHAR"),
            ("lat", "DOUBLE"),
            ("lon", "DOUBLE"),
            ("wkt", "VARCHAR"),
            ("ndvi_date", "VARCHAR"),
            ("ndvi_mean", "DOUBLE"),
            ("ndvi_std", "DOUBLE"),
            ("day_delta", "INTEGER"),
            *[(v, "DOUBLE") for v in value_cols],
        ]
        if write_output and dst:
            for row in unified:
                row["wkt"] = f"POINT({row['lon']} {row['lat']})"
            try:
                await asyncio.to_thread(
                    _write_geoparquet,
                    unified,
                    columns=staging,
                    dst=dst,
                    date_cols=("date", "ndvi_date"),
                )
                written = dst
            except Exception as e:
                log.error("climate_enrich_ndvi_write_error", zone_id=zone_id, error=str(e))
                return {"error": f"Unified GeoParquet write failed: {e}"}

        meta = {
            "start_date": start_date,
            "end_date": end_date,
            "tolerance_days": tolerance_days,
            "alignment": "nearest-date ±N days (proposed, not ratified — ADR 0002 §3)",
            "era5_source": era5_src,
            "ndvi_input": "stored_b3_series" if ndvi_source else "stac_cog_stream",
            # On the row beside the data, because "this table covers its window" and "this table is
            # the intersection" are different claims and the rows look identical either way.
            "window_contract": "strict" if strict_window else "intersection",
            "elapsed_ms": round((time.monotonic() - t0) * 1000),
        }
        if ndvi_source:
            meta["ndvi_source"] = ndvi_source
            # No silent caps, and no silent COMPLETENESS either. Reading a stored series removes
            # the STAC item cap, so `ndvi_truncation_warning` below cannot fire — and a warning
            # that stops appearing must not be read as coverage becoming complete. The B3 pilots
            # have a monthly least-cloudy cadence with 81 of 120 requested months represented for
            # montpellier_metropole. The artifact does not encode why the other months are absent.
            # Publishing requested-vs-observed months beside the rows keeps that difference visible.
            meta["ndvi_coverage"] = ndvi_coverage
            if (
                ndvi_coverage
                and ndvi_coverage["months_observed"] < ndvi_coverage["months_requested"]
            ):
                meta["ndvi_coverage_note"] = (
                    f"{ndvi_coverage['months_observed']} of "
                    f"{ndvi_coverage['months_requested']} requested months carry an observation "
                    "in the stored series; the remainder are absent from the stored series. "
                    "The artifact does not encode why; absence is not zero"
                )
        # No silent losses. An NDVI observation the aligner does not turn into a row is invisible
        # in a row count, and the two reasons that happens mean opposite things — so they are
        # counted separately, against the ERA5 grid rather than against the deduplicated rows.
        # Measured on the dev harness: the stored ERA5 is the WMO 1991-2020 baseline while the B3
        # pilot series runs to 2024, so a 2016-2024 request aligned 40 of 81 observations and said
        # nothing about the 41 that fell past the end of the climate record. The count and the
        # ERA5 window together are what let a reader see that the table is a 2016-2020 table,
        # whatever window was asked for.
        census = _ndvi_alignment_census(era5_rows, ndvi_obs, tolerance_days=tolerance_days)
        collapsed = census["matchable"] - len(unified)
        if census["out_of_range"] or collapsed or census["unreadable"]:
            era5_dates = [str(r.get("date"))[:10] for r in era5_rows if r.get("date")]
            meta["era5_window"] = (
                f"{min(era5_dates)}..{max(era5_dates)}" if era5_dates else "unknown"
            )
        if census["out_of_range"]:
            # A coverage gap: the table is narrower than the window that was requested.
            meta["ndvi_out_of_era5_range"] = census["out_of_range"]
            meta["ndvi_out_of_era5_range_note"] = (
                f"{census['out_of_range']} of {len(ndvi_obs) - census['unreadable']} NDVI "
                f"observations with parseable dates had no ERA5 day within ±{tolerance_days} d "
                f"and are absent from the unified table; the ERA5 source covers "
                f"{meta['era5_window']}"
            )
        if collapsed > 0:
            # A resolution limit, NOT a gap: the unified key is (zone, date), so several
            # observations around one ERA5 day yield one row. Selection minimizes day_delta,
            # then prefers the higher NDVI value when deltas tie.
            meta["ndvi_collapsed_onto_shared_days"] = collapsed
            meta["ndvi_collapsed_note"] = (
                f"{collapsed} further observation(s) matched an ERA5 day already represented; "
                "the unified key is (zone, date), so selection uses minimum day_delta, then "
                "higher NDVI when day_delta ties"
            )
        if census["unreadable"]:
            meta["ndvi_unreadable_dates"] = census["unreadable"]
        if not unified:
            meta["note"] = "no NDVI observations matched within ±tolerance_days" + (
                " — wrote an empty schema-valid unified table"
                if written
                else " — no unified table written"
            )
        if images_searched >= settings.STAC_SEARCH_MAX_ITEMS:
            # No silent caps: NDVI collection is bounded by the STAC item limit; flag possible
            # truncation on long ranges rather than reporting a misleadingly complete table.
            meta["ndvi_truncation_warning"] = (
                f"NDVI search hit the {settings.STAC_SEARCH_MAX_ITEMS}-item STAC cap — observations "
                "may be truncated; narrow the date range for full coverage"
            )
        return {
            "zone_id": zone_id,
            "rows": unified,
            "output": written,
            "statistics": {
                "n_era5_days": len(era5_rows),
                "n_ndvi_observations": len(ndvi_obs),
                "n_matched": len(unified),
                "images_searched": images_searched,
            },
            "metadata": meta,
        }

    @mcp.tool(output_schema=output_schema_for("climate_percentiles"))
    async def climate_percentiles(
        zone_id: str,
        start_date: str | None = None,
        end_date: str | None = None,
        variables: list[str] | None = None,
        era5_source: str | None = None,
        overwrite: bool = False,
        write_output: bool = True,
        output_path: str | None = None,
        classify_observed: dict[str, float] | None = None,
        classify_month: int | None = None,
    ) -> dict:
        """Compute ERA5 climatological P10/P50/P90 normals by calendar month (FIERCE B6).

        Defaults to the WMO 1991-2020 climate-normal window, reuses an existing ERA5 parquet
        when it covers the requested zone/window, otherwise fetches the Archive API through the B2
        client. Materializes ``climate/normals/{zone_id}.parquet`` in long format.
        """
        check_input(
            ClimatePercentilesInput,
            zone_id=zone_id,
            start_date=start_date,
            end_date=end_date,
            variables=variables,
            era5_source=era5_source,
            overwrite=overwrite,
            write_output=write_output,
            output_path=output_path,
            classify_observed=classify_observed,
            classify_month=classify_month,
        )
        log.info("climate_percentiles", zone_id=zone_id, start_date=start_date, end_date=end_date)
        t0 = time.monotonic()

        # `variables is None` (omitted) → default to all six; an EXPLICIT empty list is a malformed
        # request and must be rejected, not silently widened to all variables (Codex P2).
        requested = (
            list(dict.fromkeys(variables)) if variables is not None else list(NORMALIZED_VARS)
        )
        if not requested:
            return {"error": "variables must not be empty"}
        unknown = [v for v in requested if v not in NORMALIZED_VARS]
        if unknown:
            return {"error": f"Unknown variables {unknown}; supported: {list(NORMALIZED_VARS)}"}

        if classify_observed is not None and classify_month is None:
            return {"error": "classify_month is required when classify_observed is provided"}
        if classify_observed is not None:
            observed_unknown = [v for v in classify_observed if v not in NORMALIZED_VARS]
            if observed_unknown:
                return {
                    "error": (
                        f"Unknown classify_observed variables {observed_unknown}; "
                        f"supported: {list(NORMALIZED_VARS)}"
                    )
                }
            excluded = [v for v in classify_observed if v not in requested]
            if excluded:
                return {"error": f"classify_observed variables must be requested too: {excluded}"}

        resolved_start, resolved_end, window_label = _resolve_normals_window(start_date, end_date)
        date_err = _validate_date_range(resolved_start, resolved_end)
        if date_err:
            return {"error": date_err}

        zone = _resolve_zone(zone_id)
        if zone is None:
            return {"error": f"Unknown zone_id {zone_id!r} (not in zones/zones.geojson)"}
        lon, lat = zone["lon"], zone["lat"]

        dst = output_path or _default_output_uri(zone_id, "normals")
        if write_output and not dst:
            return {"error": "No output destination — set S3_BUCKET or pass output_path"}
        if write_output and dst:
            try:
                _validate_dst(dst)
            except ValueError as e:
                return {"error": f"Invalid output_path: {e}"}

        era5_src = era5_source or _default_output_uri(zone_id, "era5")
        exists = (
            await asyncio.to_thread(_output_exists, dst)
            if (write_output and dst and not overwrite)
            else False
        )
        if exists:
            coverage = await asyncio.to_thread(_normals_coverage, dst, zone_id=zone_id)
            # Skip ONLY if the stored file fully covers the requested variables (12 months with data)
            # AND was computed over the SAME resolved window — otherwise a custom-window re-run would
            # return stale normals mislabeled with the new window (review P1/P2).
            if _normals_coverage_satisfies(coverage, requested) and _normals_window_matches(
                coverage, resolved_start, resolved_end
            ):
                stored_records = await asyncio.to_thread(
                    _read_normals_records, dst, zone_id=zone_id
                )
                normals = [row for row in stored_records if row.get("variable") in requested]
                anomaly = _classify_observed_values(
                    normals, observed=classify_observed, month=classify_month
                )
                stored_window = coverage.get("window") or {}
                return {
                    "zone_id": zone_id,
                    "normals": _public_normals(normals),
                    "output": dst,
                    "variables": requested,
                    "statistics": {
                        "n_normals": len(normals),
                        "n_variables": len(requested),
                        "n_months": _normals_month_count(normals),
                    },
                    "metadata": _percentiles_metadata(
                        # echo the STORED window, never re-derive from the new request (review P1)
                        resolved_start=stored_window.get("start", resolved_start),
                        resolved_end=stored_window.get("end", resolved_end),
                        window_label=stored_window.get("label", window_label),
                        source="normals_parquet",
                        era5_src=era5_src,
                        source_fallback_reason=None,
                        requested=requested,
                        t0=t0,
                        extra={
                            "skipped": (
                                "normals output exists and covers the request (overwrite=False)"
                            ),
                            "stored_coverage": coverage,
                        },
                    ),
                    **({"anomaly": anomaly} if anomaly is not None else {}),
                }
            return {
                "error": (
                    f"{dst} exists but its stored normals do not cover the requested variables "
                    f"{requested} across 12 months with data, or were computed over a different "
                    f"window than {resolved_start}..{resolved_end} — re-run with overwrite=True to replace"
                ),
                "metadata": {"stored_coverage": coverage},
            }

        era5_rows: list[dict] = []
        source = "archive_fetch"
        source_fallback_reason = None
        if era5_src:
            try:
                candidate_rows = await asyncio.to_thread(_read_era5_records, era5_src)
                clipped_rows = _filter_era5_rows_for_window(
                    candidate_rows,
                    zone_id=zone_id,
                    start_date=resolved_start,
                    end_date=resolved_end,
                )
                if _era5_rows_cover_window(
                    clipped_rows,
                    start_date=resolved_start,
                    end_date=resolved_end,
                    requested=requested,
                ):
                    era5_rows = clipped_rows
                    source = "era5_parquet"
                elif era5_source:
                    # An EXPLICIT source that doesn't fully cover must error, not silently trigger a
                    # multi-year Archive fetch from a different source than the caller asked for
                    # (Greptile P2). Only the DERIVED default source falls back to fetch.
                    return {
                        "error": (
                            f"era5_source {era5_source!r} does not fully cover "
                            f"{resolved_start}..{resolved_end} for variables {requested} "
                            "(needs full daily coverage) — provide a covering source or omit "
                            "era5_source to fetch from the Archive"
                        )
                    }
                else:
                    source_fallback_reason = "default_era5_does_not_cover_window_or_variables"
            except Exception as e:
                if era5_source:
                    return {"error": f"Could not read ERA5 source {era5_source!r}: {e}"}
                source_fallback_reason = f"default_era5_source_unreadable: {e}"

        if not era5_rows:
            try:
                raws = []
                for chunk_start, chunk_end in split_date_range(resolved_start, resolved_end):
                    raw = await asyncio.to_thread(
                        fetch_archive_daily,
                        latitude=lat,
                        longitude=lon,
                        start_date=chunk_start,
                        end_date=chunk_end,
                        normalized_variables=requested,
                    )
                    raws.append(raw)
                dates, columns = normalize_archive_response(raws, requested)
            except Exception as e:
                log.error("climate_percentiles_fetch_error", zone_id=zone_id, error=str(e))
                return {"error": f"Open-Meteo Archive fetch/normalize failed: {e}"}
            if not dates:
                return {"error": "Open-Meteo returned no daily records for the requested range"}
            era5_rows = _records_from_archive(
                dates, columns, zone_id=zone_id, lat=lat, lon=lon, requested=requested
            )
            source = "archive_fetch"

        normals = compute_monthly_normals(
            era5_rows,
            zone_id=zone_id,
            lat=lat,
            lon=lon,
            requested=requested,
            window_start=resolved_start,
            window_end=resolved_end,
            window_label=window_label,
        )
        anomaly = _classify_observed_values(
            normals, observed=classify_observed, month=classify_month
        )

        written = None
        if write_output and dst:
            staging = [
                ("month", "INTEGER"),
                ("zone_id", "VARCHAR"),
                ("variable", "VARCHAR"),
                ("p10", "DOUBLE"),
                ("p50", "DOUBLE"),
                ("p90", "DOUBLE"),
                ("n", "INTEGER"),
                ("mean", "DOUBLE"),
                ("window_start", "VARCHAR"),
                ("window_end", "VARCHAR"),
                ("window_label", "VARCHAR"),
                ("lat", "DOUBLE"),
                ("lon", "DOUBLE"),
                ("wkt", "VARCHAR"),
                # v2, after the whole v1 block. They were grouped beside `mean`, next to the
                # all-days percentiles they belong with, which reads better and made the "additive"
                # claim in this comment false: every v1 column from `window_start` on shifted
                # ordinal position, so a positional reader got `p10_wet` where it expected
                # `window_start` (Greptile P2, PR #189).
                #
                # What is achievable, and what this ordering guarantees:
                #
                #   * the 13 v1 DATA columns (`month` … `lon`) keep their v1 index exactly;
                #   * `geometry` remains the LAST column, as it is in v1.
                #
                # A literal byte-prefix is NOT achievable and is not claimed. `_write_geoparquet`
                # emits `wkt` as `geometry` and always appends it last, wherever it sits in this
                # staging list, so `geometry` moves from index 13 to index 19. That writer is shared
                # with the era5/unified/ndvi artifacts and teaching it to honour a position here
                # would relayout all of them — a far larger change than the ordinal it would fix.
                # A reader that takes "the last column" as the geometry, which is what the writer
                # has always guaranteed, is unaffected; one pinned to index 13 is not.
                ("p10_wet", "DOUBLE"),
                ("p50_wet", "DOUBLE"),
                ("p90_wet", "DOUBLE"),
                ("n_wet", "INTEGER"),
                ("wet_day_threshold_mm", "DOUBLE"),
                ("schema_version", "INTEGER"),
            ]
            try:
                await asyncio.to_thread(
                    _write_geoparquet, normals, columns=staging, dst=dst, date_cols=()
                )
                written = dst
            except Exception as e:
                log.error("climate_percentiles_write_error", zone_id=zone_id, error=str(e))
                return {"error": f"Normals GeoParquet write failed: {e}"}

        metadata = _percentiles_metadata(
            resolved_start=resolved_start,
            resolved_end=resolved_end,
            window_label=window_label,
            source=source,
            era5_src=era5_src,
            source_fallback_reason=source_fallback_reason,
            requested=requested,
            t0=t0,
        )
        return {
            "zone_id": zone_id,
            "normals": _public_normals(normals),
            "output": written,
            "variables": requested,
            "statistics": {
                "n_era5_days": len(era5_rows),
                "n_normals": len(normals),
                "n_variables": len(requested),
                "n_months": _normals_month_count(normals),
            },
            "metadata": metadata,
            **({"anomaly": anomaly} if anomaly is not None else {}),
        }

    @mcp.tool(output_schema=output_schema_for("climate_normals"))
    async def climate_normals(
        zone_id: str,
        variables: list[str] | None = None,
        values: list[dict] | None = None,
    ) -> dict:
        """Read the stored climate normals for a zone, and optionally classify forecast days.

        The read-only counterpart of ``climate_percentiles`` (site-watch V0 §G2). That tool is
        an ``internal_write`` operator: it fetches, computes and materializes. This one only
        reads the zone's canonical ``climate/normals/{zone_id}.parquet`` and returns. There is
        no ``write_output``, no destination and **no source parameter**: ``science_shared_read``
        means an allowlisted dataset through a stable alias, so a caller names a zone, never a
        path.

        With ``values``, each entry is classified against the normals of **its own month**.
        There is no request-level month: a 7-day horizon spanning 31 January to 2 February
        reads January's normals for the first days and February's for the rest.

        Precipitation is banded against the wet-day distribution (``p10_wet``/``p50_wet``/
        ``p90_wet``, days ≥ 1.0 mm), never the all-days percentiles, which include dry days and
        understate heavy rain. A day below the threshold is ``dry``: the value is kept and no
        band is invented.

        This is a raw-forecast classification, not an event watch and not a calibrated risk
        (RISK_REGISTER R14/R15). ``confidence`` is present and always ``null``; there is no
        probability, risk or severity field, by construction.

        Args:
            zone_id: A zone identifier present in zones/zones.geojson.
            variables: Normalized variable names to return (default: everything stored). Every
                variable named in ``values`` must be inside this narrowing, or the request is
                refused — filtering the reference away and then reporting the classification
                as ``reference_unavailable`` would state something false about the corpus.
            values: Optional ``[{variable, date, value}]`` forecast days to classify. A ``null``
                ``value`` is a provider gap and yields ``status: "missing"``, never a zero.

        Returns:
            ``normals`` (the monthly rows), ``series`` (schema-valid ``watch-input/v1`` entries),
            ``unclassified`` (entries the stored reference cannot support, with a reason),
            ``reference``, ``confidence: null``, ``caveats``, ``statistics`` and ``metadata``
            (``metadata.provenance`` holds the attribution and the forecast-side fields this
            tool cannot know, as ``null`` with a reason). ``metadata.artifact_version`` is the
            SHA-256 of the original Parquet bytes queried, with their ``artifact_size_bytes``.
            Reads require Linux procfs and at most 8 MiB of source bytes (decompressed data
            is not bounded by this cap). A refusal is ``{"error", "confidence": null}``.
        """
        check_input(
            ClimateNormalsInput,
            zone_id=zone_id,
            variables=variables,
            values=values,
        )
        log.info("climate_normals", zone_id=zone_id, n_values=len(values or []))
        t0 = time.monotonic()

        # An EXPLICIT empty list is a malformed request, not "everything" — the same distinction
        # `climate_percentiles` draws, and for the same reason: silently widening a caller's
        # narrowing is how a facade returns more than was asked for.
        if variables is not None and not variables:
            return _normals_refusal("variables must be a non-empty list when provided")
        if variables:
            try:
                for name in variables:
                    provider_var(name)
            except ValueError as e:
                return _normals_refusal(str(e))
            # Refuse rather than classify against a reference the narrowing removed. Loading
            # it anyway would be friendlier and worse: the two inputs would disagree and the
            # output would not say so.
            outside = sorted(
                {
                    str(item.get("variable"))
                    for item in (values or [])
                    if str(item.get("variable")) not in set(variables)
                }
            )
            if outside:
                return _normals_refusal(
                    f"values reference variables outside the requested set: {', '.join(outside)}"
                )

        src = _default_output_uri(zone_id, "normals")
        if not src:
            return _normals_refusal("No normals source configured for this zone")
        try:
            artifact = await asyncio.to_thread(
                _read_bound_normals, src, zone_id=zone_id, variables=variables
            )
        except Exception as e:
            # Storage exceptions can contain locators or credentials. Publish only
            # the safe exception type and a fixed diagnostic category in logs,
            # and a generic refusal to the caller.
            log.error(
                "climate_normals_read_error",
                zone_id=zone_id,
                error_type=type(e).__name__,
                reason=normals_read_error_reason(e),
            )
            return _normals_refusal(f"Could not read the stored normals for zone {zone_id!r}")
        rows = artifact.rows
        if not rows:
            return _normals_refusal(f"No stored normals for zone {zone_id!r}")

        # Inside the guard, not after it: every value below comes out of the artifact, and a
        # readable file with a malformed column must produce the documented error rather than
        # a traceback from the response builder (#203 review).
        try:
            series, unclassified = build_watch_series(
                [item if isinstance(item, dict) else dict(item) for item in (values or [])],
                normals=rows,
            )
            stored_variables = sorted(
                {str(row["variable"]) for row in rows if isinstance(row.get("variable"), str)}
            )
            # A file written before #189 has no `schema_version` column at all; a row written
            # before it does, but with a NULL. Both are v1 — see `_normals_coverage`.
            schema_versions = sorted(
                {_stored_int(row.get("schema_version"), 1) or 1 for row in rows}
            ) or [1]
            metadata = {
                # The logical dataset, not the physical locator: publishing the path would
                # hand back exactly the locator the profile keeps out of the input.
                "dataset": f"science:normals:{zone_id}",
                "artifact_version": artifact.artifact_version,
                "artifact_size_bytes": artifact.artifact_size_bytes,
                "normal_grain": _NORMAL_GRAIN,
                "percentile_levels": [10, 50, 90],
                "wet_day_threshold_mm": WET_DAY_THRESHOLD_MM,
                "normals_schema_version": schema_versions,
                "variables": stored_variables,
                "watch_series_variables": list(WATCH_SERIES_VARIABLES),
                "provider_map": _provider_map(stored_variables),
                "license": OPEN_METEO_DATA_LICENSE,
                "provenance": _normals_provenance(),
                "day_grain": "utc_day",
                "elapsed_ms": round((time.monotonic() - t0) * 1000),
            }
            # The window dates do not belong in `reference` (schema `additionalProperties:
            # false`); omitted rather than published half-empty when either date is absent —
            # see `_normals_window_dates`.
            normals_window = _normals_window_dates(rows)
            if normals_window:
                metadata["normals_window"] = normals_window
            payload = {
                "zone_id": zone_id,
                "normals": _public_normals(rows),
                "series": series,
                "unclassified": unclassified,
                "reference": _normals_reference(rows),
                "confidence": None,
                "caveats": list(NORMALS_CAVEATS),
                "metadata": metadata,
                "statistics": {
                    "n_normals": len(rows),
                    "n_variables": len(stored_variables),
                    "n_values": len(values or []),
                    "n_classified": len(series),
                    "n_unclassified": len(unclassified),
                },
            }
        except Exception as e:
            log.error("climate_normals_build_error", zone_id=zone_id, error=str(e))
            return _normals_refusal(f"Stored normals for zone {zone_id!r} are unusable: {e}")
        return payload
