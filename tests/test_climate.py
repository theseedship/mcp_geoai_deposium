"""Tests for FIERCE B2/B4 climate ingestion + enrichment (network-free + a live contract probe).

Layers:
1. **Pure helpers** — unit conversion, date-range chunking, Archive normalization, completeness,
   and the NDVI↔ERA5 ±N-day alignment — exercised with no network.
2. **era5_ingest** end-to-end with the Open-Meteo client monkeypatched by a captured fixture:
   asserts the WKB GeoParquet round-trips via DuckDB (``ST_GeomFromWKB``) with the normalized
   terra-cognita §5.1 schema, the correct ``n_days``, and the solar-radiation unit conversion.
3. **climate_enrich_ndvi** with NDVI collection monkeypatched: asserts the unified table
   materializes and the ±N-day join pairs the right rows.
4. A **live** Archive contract probe (skipped unless ``OPENMETEO_LIVE=1``) that fails loudly on an
   upstream variable rename.
"""

from __future__ import annotations

import asyncio
import math
import os
from datetime import date, timedelta
from pathlib import Path

import pytest
from pydantic import ValidationError

from src.config import settings
from src.contracts import ClimateEnrichNdviInput
from src.tools.climate import (
    _align_ndvi_to_era5,
    _completeness_report,
    _era5_rows_cover_window,
    _filter_era5_rows_for_window,
    _ndvi_alignment_census,
    _normals_coverage,
    _normals_coverage_satisfies,
    _normals_month_count,
    _validate_dst,
    _write_geoparquet,
    classify_against_normals,
    compute_monthly_normals,
    normalize_archive_response,
    register_climate_tools,
)
from src.utils.duckdb_conn import get_duckdb_connection
from src.utils.openmeteo import (
    NORMALIZED_VARS,
    PROVIDER_VARS,
    convert_to_stored_unit,
    split_date_range,
)

_RUN_LIVE = os.environ.get("OPENMETEO_LIVE") == "1"
pytestmark = pytest.mark.usefixtures("allow_local_parquet_paths")


def test_local_climate_path_rejects_when_global_policy_is_off(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "ALLOW_LOCAL_PARQUET_SOURCE", False)

    with pytest.raises(ValueError, match="ALLOW_LOCAL_PARQUET_SOURCE"):
        _validate_dst((tmp_path / "era5.parquet").as_posix())


def test_local_climate_path_must_be_under_configured_root(monkeypatch, tmp_path):
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    monkeypatch.setattr(settings, "LOCAL_PARQUET_SOURCE_ROOT", allowed.as_posix())

    with pytest.raises(ValueError, match="LOCAL_PARQUET_SOURCE_ROOT"):
        _validate_dst((tmp_path / "outside.parquet").as_posix())


# A faithful captured Open-Meteo Archive **daily** response (3 days, all six §5.1 provider vars).
# shortwave_radiation_sum is in MJ/m² (the API's native daily unit) — the ingest converts to W/m².
ARCHIVE_FIXTURE = {
    "latitude": 43.6,
    "longitude": 3.5,
    "generationtime_ms": 1.2,
    "utc_offset_seconds": 0,
    "timezone": "GMT",
    "elevation": 50.0,
    "daily_units": {
        "time": "iso8601",
        "temperature_2m_mean": "°C",
        "precipitation_sum": "mm",
        "relative_humidity_2m_mean": "%",
        "et0_fao_evapotranspiration": "mm",
        "wind_speed_10m_max": "m/s",
        "shortwave_radiation_sum": "MJ/m²",
    },
    "daily": {
        "time": ["2024-01-01", "2024-01-02", "2024-01-03"],
        "temperature_2m_mean": [8.5, 9.1, 7.8],
        "precipitation_sum": [0.0, 2.3, 0.1],
        "relative_humidity_2m_mean": [80.0, 78.0, 82.0],
        "et0_fao_evapotranspiration": [0.5, 0.4, 0.6],
        "wind_speed_10m_max": [3.2, 4.1, 2.8],
        "shortwave_radiation_sum": [5.0, 6.2, 4.5],
    },
}


class _CaptureMCP:
    """Records tool handlers (drops the decorator kwargs) without a live FastMCP server."""

    def __init__(self) -> None:
        self.tools: dict[str, object] = {}

    def tool(self, *_args, **_kwargs):
        def _decorator(fn):
            self.tools[fn.__name__] = fn
            return fn

        return _decorator


def _load_climate_tools() -> dict:
    cap = _CaptureMCP()
    register_climate_tools(cap)
    return cap.tools


def _obs(dates_and_ndvi: list[tuple[str, float]]) -> list[dict]:
    return [
        {"date": d, "ndvi_mean": n, "item_id": f"i{i}"} for i, (d, n) in enumerate(dates_and_ndvi)
    ]


def _archive_fixture_for_range(start_date: str, end_date: str) -> dict:
    """Synthetic Archive response with deterministic daily values for every §5.1 variable."""
    start = date.fromisoformat(start_date)
    end = date.fromisoformat(end_date)
    days = []
    cur = start
    while cur <= end:
        days.append(cur)
        cur += timedelta(days=1)

    def _series(base: float) -> list[float]:
        return [base + d.month * 10.0 + d.day / 100.0 + (d.year - 2020) for d in days]

    return {
        "daily_units": ARCHIVE_FIXTURE["daily_units"],
        "daily": {
            "time": [d.isoformat() for d in days],
            "temperature_2m_mean": _series(0.0),
            "precipitation_sum": _series(100.0),
            "relative_humidity_2m_mean": _series(200.0),
            "et0_fao_evapotranspiration": _series(300.0),
            "wind_speed_10m_max": _series(400.0),
            # Still in MJ/m², as the real Archive API reports it.
            "shortwave_radiation_sum": _series(500.0),
        },
    }


# --------------------------------------------------------------------------- #
# 1. Pure helpers
# --------------------------------------------------------------------------- #


class TestProviderContract:
    def test_fixture_units_match_pinned_source_units(self):
        # The captured fixture's daily_units must match each ProviderVar.source_unit — the same
        # assertion the live probe makes against the real API, pinned here network-free so the
        # constants and the fixture cannot silently drift apart.
        units = ARCHIVE_FIXTURE["daily_units"]
        for pv in PROVIDER_VARS:
            assert units[pv.provider] == pv.source_unit, pv.provider

    def test_only_solar_radiation_is_converted(self):
        # source_unit == stored unit for every variable EXCEPT solar_radiation (MJ/m² → W/m²)
        for pv in PROVIDER_VARS:
            if pv.normalized == "solar_radiation":
                assert pv.source_unit == "MJ/m²" and pv.unit == "W/m²"
            else:
                assert pv.source_unit == pv.unit


class TestUnitConversion:
    def test_solar_radiation_mj_to_w_per_m2(self):
        # 5 MJ/m² daily sum → W/m² daily mean = 5e6 / 86400 ≈ 57.87
        assert convert_to_stored_unit("solar_radiation", 5.0) == pytest.approx(57.870, abs=1e-3)

    def test_other_variables_passthrough(self):
        assert convert_to_stored_unit("temperature_2m", 8.5) == 8.5
        assert convert_to_stored_unit("precipitation", 0.0) == 0.0

    def test_none_passthrough(self):
        assert convert_to_stored_unit("solar_radiation", None) is None
        assert convert_to_stored_unit("temperature_2m", None) is None


class TestSplitDateRange:
    def test_single_chunk_when_within_max(self):
        assert split_date_range("2024-01-01", "2024-01-10", max_days=366) == [
            ("2024-01-01", "2024-01-10")
        ]

    def test_default_max_days_is_runtime_configurable(self, monkeypatch):
        monkeypatch.setattr("src.utils.openmeteo._MAX_DAYS_PER_REQUEST", 10)

        assert split_date_range("2024-01-01", "2024-01-25") == [
            ("2024-01-01", "2024-01-10"),
            ("2024-01-11", "2024-01-20"),
            ("2024-01-21", "2024-01-25"),
        ]

    def test_contiguous_non_overlapping_cover(self):
        chunks = split_date_range("2024-01-01", "2024-03-01", max_days=31)
        assert chunks[0][0] == "2024-01-01"
        assert chunks[-1][1] == "2024-03-01"
        # contiguous: each chunk starts the day after the previous ends, no gaps/overlaps
        from datetime import date, timedelta

        for (_, prev_end), (next_start, _) in zip(chunks, chunks[1:], strict=False):
            assert date.fromisoformat(next_start) == date.fromisoformat(prev_end) + timedelta(
                days=1
            )

    def test_rejects_inverted_range(self):
        with pytest.raises(ValueError):
            split_date_range("2024-02-01", "2024-01-01")


class TestNormalize:
    def test_normalize_applies_conversion_and_concatenates(self):
        dates, columns = normalize_archive_response([ARCHIVE_FIXTURE], list(NORMALIZED_VARS))
        assert dates == ["2024-01-01", "2024-01-02", "2024-01-03"]
        assert columns["temperature_2m"] == [8.5, 9.1, 7.8]
        assert columns["solar_radiation"][0] == pytest.approx(57.870, abs=1e-3)

    def test_two_chunks_concatenate_in_order(self):
        dates, columns = normalize_archive_response(
            [ARCHIVE_FIXTURE, ARCHIVE_FIXTURE], ["temperature_2m"]
        )
        assert len(dates) == 6
        assert columns["temperature_2m"] == [8.5, 9.1, 7.8, 8.5, 9.1, 7.8]

    def test_missing_provider_key_raises(self):
        broken = {"daily": {"time": ["2024-01-01"]}}  # no provider series
        with pytest.raises(KeyError):
            normalize_archive_response([broken], ["temperature_2m"])


class TestCompleteness:
    def test_reports_expected_received_and_does_not_gate(self):
        dates, columns = normalize_archive_response([ARCHIVE_FIXTURE], list(NORMALIZED_VARS))
        report = _completeness_report(dates, columns, "2024-01-01", "2024-01-03")
        assert report["expected_days"] == 3
        assert report["received_days"] == 3
        assert report["per_variable"]["temperature_2m"]["pct_complete"] == 100.0
        assert report["gate_enforced"] is False

    def test_gap_case_counts_non_null_and_short_window(self):
        # 2 of 3 expected days received; one temperature value missing (None must not be counted)
        dates = ["2024-01-01", "2024-01-02"]
        columns = {"temperature_2m": [8.5, None], "precipitation": [0.0, 2.3]}
        report = _completeness_report(dates, columns, "2024-01-01", "2024-01-03")
        assert report["expected_days"] == 3
        assert report["received_days"] == 2
        assert report["pct_days"] == round(100 * 2 / 3, 2)  # 66.67 — ratio orientation pinned
        assert report["per_variable"]["temperature_2m"]["received"] == 1
        assert report["per_variable"]["temperature_2m"]["pct_complete"] == round(100 * 1 / 3, 2)
        assert report["per_variable"]["precipitation"]["received"] == 2


class TestClimateNormalsPure:
    def _two_year_rows(self) -> list[dict]:
        fixture = _archive_fixture_for_range("2021-01-01", "2022-12-31")
        dates, columns = normalize_archive_response([fixture], list(NORMALIZED_VARS))
        return [
            {
                "date": d,
                "zone_id": "montpellier",
                "lat": 43.6,
                "lon": 3.5,
                **{v: columns[v][i] for v in NORMALIZED_VARS},
            }
            for i, d in enumerate(dates)
        ]

    def test_monthly_normals_span_years(self):
        normals = compute_monthly_normals(
            self._two_year_rows(),
            zone_id="montpellier",
            lat=43.6,
            lon=3.5,
            requested=list(NORMALIZED_VARS),
            window_start="2021-01-01",
            window_end="2022-12-31",
            window_label="custom",
        )
        assert len(normals) == 6 * 12
        jan_temp = next(r for r in normals if r["variable"] == "temperature_2m" and r["month"] == 1)
        assert jan_temp["n"] == 62  # Jan 2021 + Jan 2022
        assert jan_temp["p10"] <= jan_temp["p50"] <= jan_temp["p90"]
        # Pin the exact computed climatology so a percentile-method or month-bucketing regression is
        # caught: Jan temps are {11.01..11.31} ∪ {12.01..12.31} → median 11.66 (numpy linear).
        assert jan_temp["p50"] == pytest.approx(11.66)
        assert jan_temp["mean"] == pytest.approx(11.66)
        # the resolved window is stamped on every row (so the parquet records its provenance)
        assert jan_temp["window_start"] == "2021-01-01"
        assert jan_temp["window_end"] == "2022-12-31"
        assert jan_temp["wkt"] == "POINT(3.5 43.6)"

    @pytest.mark.parametrize("unusable", [float("nan"), 10**400], ids=["nan", "past-float-range"])
    def test_monthly_normals_ignore_nan_values(self, unusable):
        # An int past the float range is skipped like NaN, not raised as OverflowError (#221).
        normals = compute_monthly_normals(
            [
                {
                    "date": "2021-01-01",
                    "zone_id": "montpellier",
                    "lat": 43.6,
                    "lon": 3.5,
                    "temperature_2m": unusable,
                },
                {
                    "date": "2021-01-02",
                    "zone_id": "montpellier",
                    "lat": 43.6,
                    "lon": 3.5,
                    "temperature_2m": 12.0,
                },
            ],
            zone_id="montpellier",
            lat=43.6,
            lon=3.5,
            requested=["temperature_2m"],
            window_start="2021-01-01",
            window_end="2021-01-31",
            window_label="custom",
        )
        january = next(r for r in normals if r["month"] == 1)
        assert january["n"] == 1
        assert january["p10"] == january["p50"] == january["p90"] == 12.0

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            (9.9, "extreme_low"),
            (10.0, "below_normal"),
            (19.9, "below_normal"),
            (20.0, "above_normal"),
            (30.0, "above_normal"),
            (30.1, "extreme_high"),
        ],
    )
    def test_classify_against_normals_edges(self, value, expected):
        result = classify_against_normals(value, p10=10.0, p50=20.0, p90=30.0)
        assert result["category"] == expected
        assert result["vs_p50"] == pytest.approx(value - 20.0)

    @pytest.mark.parametrize("field", ["value", "p10", "p50", "p90"])
    def test_classify_against_normals_refuses_a_number_past_the_float_range(self, field):
        # A ValueError, like inconsistent percentiles, so the callers' `except ValueError`
        # handles it; float() alone raised OverflowError past them (#221).
        args = {"value": 15.0, "p10": 10.0, "p50": 20.0, "p90": 30.0, field: 10**400}
        with pytest.raises(ValueError, match="within the float range"):
            classify_against_normals(
                args["value"], p10=args["p10"], p50=args["p50"], p90=args["p90"]
            )

    def test_filter_era5_rows_by_zone_and_window(self):
        rows = [
            {"date": "2020-12-31", "zone_id": "montpellier", "temperature_2m": 1.0},
            {"date": "2021-01-01", "zone_id": "montpellier", "temperature_2m": 2.0},
            {"date": "2021-01-02", "zone_id": "marseille", "temperature_2m": 3.0},
            {"date": "2021-01-03", "zone_id": "montpellier", "temperature_2m": 4.0},
        ]
        clipped = _filter_era5_rows_for_window(
            rows, zone_id="montpellier", start_date="2021-01-01", end_date="2021-01-03"
        )
        assert [r["date"] for r in clipped] == ["2021-01-01", "2021-01-03"]

    def test_era5_cover_window_requires_full_daily_density(self):
        # boundary-only rows span the window but are sparse → must NOT count as covering, else the
        # normals are computed from 2 rows instead of fetching the real days (Greptile P1).
        sparse = [
            {"date": "2021-01-01", "zone_id": "z", "temperature_2m": 1.0},
            {"date": "2021-12-31", "zone_id": "z", "temperature_2m": 2.0},
        ]
        assert not _era5_rows_cover_window(
            sparse, start_date="2021-01-01", end_date="2021-12-31", requested=["temperature_2m"]
        )
        dense = [
            {"date": f"2021-01-0{d}", "zone_id": "z", "temperature_2m": float(d)} for d in (1, 2, 3)
        ]
        assert _era5_rows_cover_window(
            dense, start_date="2021-01-01", end_date="2021-01-03", requested=["temperature_2m"]
        )


class TestAlignNdvi:
    ERA5 = [
        {"date": "2024-01-01", "zone_id": "z", "lat": 43.6, "lon": 3.5, "temperature_2m": 8.5},
        {"date": "2024-01-05", "zone_id": "z", "lat": 43.6, "lon": 3.5, "temperature_2m": 9.1},
        {"date": "2024-01-10", "zone_id": "z", "lat": 43.6, "lon": 3.5, "temperature_2m": 7.8},
    ]

    def test_matches_within_tolerance_and_carries_climate(self):
        # NDVI on 2024-01-06 → nearest ERA5 day 2024-01-05 (delta 1 ≤ 3)
        unified = _align_ndvi_to_era5(self.ERA5, _obs([("2024-01-06", 0.7)]), tolerance_days=3)
        assert len(unified) == 1
        assert unified[0]["date"] == "2024-01-05"
        assert unified[0]["ndvi_mean"] == 0.7
        assert unified[0]["day_delta"] == 1
        assert unified[0]["temperature_2m"] == 9.1  # climate carried from the matched ERA5 day

    def test_drops_observations_outside_tolerance(self):
        # NDVI on 2024-01-15 → nearest ERA5 day is 2024-01-10 (delta 5 > 3) → dropped
        unified = _align_ndvi_to_era5(self.ERA5, _obs([("2024-01-15", 0.4)]), tolerance_days=3)
        assert unified == []

    def test_skips_null_dates(self):
        unified = _align_ndvi_to_era5(
            self.ERA5, [{"date": None, "ndvi_mean": 0.5}], tolerance_days=3
        )
        assert unified == []

    def test_iso_datetime_ndvi_dates_are_truncated(self):
        unified = _align_ndvi_to_era5(
            self.ERA5, [{"date": "2024-01-01T10:30:00+00:00", "ndvi_mean": 0.6}], tolerance_days=3
        )
        assert len(unified) == 1
        assert unified[0]["date"] == "2024-01-01"
        assert unified[0]["day_delta"] == 0

    def test_inclusive_tolerance_boundary(self):
        # delta exactly == tolerance is KEPT (NDVI 01-13 → 01-10, delta 3)
        kept = _align_ndvi_to_era5(self.ERA5, _obs([("2024-01-13", 0.5)]), tolerance_days=3)
        assert len(kept) == 1 and kept[0]["date"] == "2024-01-10" and kept[0]["day_delta"] == 3
        # delta one past tolerance is DROPPED (NDVI 01-14 → 01-10, delta 4)
        dropped = _align_ndvi_to_era5(self.ERA5, _obs([("2024-01-14", 0.5)]), tolerance_days=3)
        assert dropped == []

    def test_picks_nearest_not_first_among_multiple(self):
        # NDVI 01-04 is within ±3 of BOTH 01-01 (delta 3) and 01-05 (delta 1) → must snap to 01-05
        unified = _align_ndvi_to_era5(self.ERA5, _obs([("2024-01-04", 0.6)]), tolerance_days=3)
        assert len(unified) == 1
        assert unified[0]["date"] == "2024-01-05"
        assert unified[0]["day_delta"] == 1
        assert unified[0]["temperature_2m"] == 9.1

    def test_dedup_same_calendar_date_keeps_higher_ndvi(self):
        # Two NDVI items on the same day (MGRS tile-overlap) snap to one ERA5 day → one row, best NDVI
        unified = _align_ndvi_to_era5(
            self.ERA5, _obs([("2024-01-05", 0.40), ("2024-01-05", 0.70)]), tolerance_days=3
        )
        assert len(unified) == 1
        assert unified[0]["date"] == "2024-01-05"
        assert unified[0]["ndvi_mean"] == 0.70  # higher NDVI wins the day_delta tie

    def test_dedup_distinct_dates_collapsing_to_one_era5_day_keeps_closest(self):
        # 01-04 (delta 1) and 01-06 (delta 1) both snap to 01-05 → one row; tie broken by higher NDVI
        unified = _align_ndvi_to_era5(
            self.ERA5, _obs([("2024-01-04", 0.50), ("2024-01-06", 0.80)]), tolerance_days=3
        )
        assert len(unified) == 1
        assert unified[0]["date"] == "2024-01-05"
        assert unified[0]["ndvi_mean"] == 0.80

    def test_carries_ndvi_std_uncertainty(self):
        obs = [{"date": "2024-01-05", "ndvi_mean": 0.7, "ndvi_std": 0.05}]
        unified = _align_ndvi_to_era5(self.ERA5, obs, tolerance_days=3)
        assert unified[0]["ndvi_std"] == 0.05  # terra §5.3 uncertainty carried through


# --------------------------------------------------------------------------- #
# 2. era5_ingest — network-free (fixture-driven), full WKB GeoParquet round-trip
# --------------------------------------------------------------------------- #


class TestEra5IngestNetworkFree:
    def test_ingest_writes_geoparquet_round_trip(self, tmp_path, monkeypatch):
        def _fake_fetch(**_kwargs):
            return ARCHIVE_FIXTURE

        monkeypatch.setattr("src.tools.climate.fetch_archive_daily", _fake_fetch)
        dst = (tmp_path / "era5_montpellier.parquet").as_posix()
        ingest = _load_climate_tools()["era5_ingest"]

        result = asyncio.run(
            ingest(
                zone_id="montpellier",
                start_date="2024-01-01",
                end_date="2024-01-03",
                output_path=dst,
            )
        )

        assert "error" not in result, result
        assert result["n_days"] == 3
        assert result["output"] == dst
        assert result["variables"] == list(NORMALIZED_VARS)
        assert result["completeness"]["expected_days"] == 3
        assert result["metadata"]["models"] == "era5"
        assert result["metadata"]["license"] == "CC-BY 4.0 (Open-Meteo)"
        assert "non-commercial" not in result["metadata"]["license"]

        con = get_duckdb_connection(":memory:")
        n = con.execute(f"SELECT count(*) FROM read_parquet('{dst}')").fetchone()[0]
        assert n == 3

        cols = {
            d[0] for d in con.execute(f"SELECT * FROM read_parquet('{dst}') LIMIT 0").description
        }
        for normalized in NORMALIZED_VARS:
            assert normalized in cols
        assert {"date", "zone_id", "lat", "lon", "geometry"} <= cols

        # Geometry round-trips as the zone centroid POINT(lon lat)
        wkt = con.execute(
            f"SELECT ST_AsText(ST_GeomFromWKB(geometry)) FROM read_parquet('{dst}') LIMIT 1"
        ).fetchone()[0]
        assert wkt == "POINT (3.5 43.6)"

        # solar_radiation stored as W/m² (converted from the fixture's 5.0 MJ/m²)
        solar = con.execute(
            f"SELECT solar_radiation FROM read_parquet('{dst}') ORDER BY date LIMIT 1"
        ).fetchone()[0]
        assert solar == pytest.approx(57.870, abs=1e-3)
        con.close()

    def test_unknown_zone_is_a_graceful_error(self, monkeypatch):
        monkeypatch.setattr("src.tools.climate.fetch_archive_daily", lambda **_k: ARCHIVE_FIXTURE)
        ingest = _load_climate_tools()["era5_ingest"]
        result = asyncio.run(
            ingest(
                zone_id="atlantis",
                start_date="2024-01-01",
                end_date="2024-01-03",
                output_path="/tmp/should_not_write.parquet",
            )
        )
        assert "error" in result and "atlantis" in result["error"]

    def test_future_end_date_is_rejected(self, monkeypatch):
        # monkeypatch fetch with a fail-loud stub so a regressed date guard fails on the "future"
        # assertion, not on a live network error.
        monkeypatch.setattr(
            "src.tools.climate.fetch_archive_daily",
            lambda **_k: (_ for _ in ()).throw(AssertionError("must not fetch")),
        )
        ingest = _load_climate_tools()["era5_ingest"]
        result = asyncio.run(
            ingest(
                zone_id="montpellier",
                start_date="2099-01-01",
                end_date="2099-01-03",
                output_path="/tmp/should_not_write.parquet",
            )
        )
        assert "error" in result and "future" in result["error"]

    def test_duplicate_variables_are_deduped(self, tmp_path, monkeypatch):
        monkeypatch.setattr("src.tools.climate.fetch_archive_daily", lambda **_k: ARCHIVE_FIXTURE)
        dst = (tmp_path / "dup.parquet").as_posix()
        result = asyncio.run(
            _load_climate_tools()["era5_ingest"](
                zone_id="montpellier",
                start_date="2024-01-01",
                end_date="2024-01-03",
                variables=["temperature_2m", "temperature_2m"],
                output_path=dst,
            )
        )
        assert "error" not in result, result
        assert result["variables"] == ["temperature_2m"]  # deduped
        assert result["completeness"]["per_variable"]["temperature_2m"]["pct_complete"] == 100.0

    def test_unknown_variables_graceful_error(self):
        result = asyncio.run(
            _load_climate_tools()["era5_ingest"](
                zone_id="montpellier",
                start_date="2024-01-01",
                end_date="2024-01-03",
                variables=["bogus"],
                output_path="/tmp/should_not_write.parquet",
            )
        )
        assert "error" in result and "Unknown variables" in result["error"]

    def test_fetch_failure_graceful_error(self, tmp_path, monkeypatch):
        def _boom(**_k):
            raise RuntimeError("boom")

        monkeypatch.setattr("src.tools.climate.fetch_archive_daily", _boom)
        result = asyncio.run(
            _load_climate_tools()["era5_ingest"](
                zone_id="montpellier",
                start_date="2024-01-01",
                end_date="2024-01-03",
                output_path=(tmp_path / "x.parquet").as_posix(),
            )
        )
        assert "error" in result and "fetch/normalize failed" in result["error"]

    def test_empty_response_graceful_error(self, tmp_path, monkeypatch):
        empty = {
            "daily": {
                "time": [],
                "temperature_2m_mean": [],
                "precipitation_sum": [],
                "relative_humidity_2m_mean": [],
                "et0_fao_evapotranspiration": [],
                "wind_speed_10m_max": [],
                "shortwave_radiation_sum": [],
            }
        }
        monkeypatch.setattr("src.tools.climate.fetch_archive_daily", lambda **_k: empty)
        result = asyncio.run(
            _load_climate_tools()["era5_ingest"](
                zone_id="montpellier",
                start_date="2024-01-01",
                end_date="2024-01-03",
                output_path=(tmp_path / "x.parquet").as_posix(),
            )
        )
        assert "error" in result and "no daily records" in result["error"]

    @pytest.mark.parametrize(
        "bad_path",
        [
            "http://169.254.169.254/latest/x.parquet",  # SSRF
            "file:///etc/passwd.parquet",  # local file scheme
            "s3://other-bucket/secret.parquet",  # cross-bucket (no S3_BUCKET configured)
            "../../escape.parquet",  # traversal
        ],
    )
    def test_unsafe_output_path_rejected_gracefully(self, bad_path, monkeypatch):
        # validation runs before any fetch; fail loud if a fetch is ever attempted
        monkeypatch.setattr(
            "src.tools.climate.fetch_archive_daily",
            lambda **_k: (_ for _ in ()).throw(AssertionError("must not fetch")),
        )
        ingest = _load_climate_tools()["era5_ingest"]
        for overwrite in (False, True):  # False is the path that previously raised uncaught
            result = asyncio.run(
                ingest(
                    zone_id="montpellier",
                    start_date="2024-01-01",
                    end_date="2024-01-03",
                    output_path=bad_path,
                    overwrite=overwrite,
                )
            )
            assert "error" in result and "output_path" in result["error"], (bad_path, overwrite)

    def test_write_failure_graceful_error(self, tmp_path, monkeypatch):
        monkeypatch.setattr("src.tools.climate.fetch_archive_daily", lambda **_k: ARCHIVE_FIXTURE)
        # a regex-valid but unwritable destination: a path component that is actually a file
        afile = tmp_path / "afile"
        afile.write_text("not a dir")
        dst = (afile / "out.parquet").as_posix()
        result = asyncio.run(
            _load_climate_tools()["era5_ingest"](
                zone_id="montpellier",
                start_date="2024-01-01",
                end_date="2024-01-03",
                output_path=dst,
            )
        )
        assert "error" in result and "GeoParquet write failed" in result["error"]

    def test_idempotent_skip_then_overwrite(self, tmp_path, monkeypatch):
        calls = {"n": 0}

        def _counting_fetch(**_k):
            calls["n"] += 1
            return ARCHIVE_FIXTURE

        monkeypatch.setattr("src.tools.climate.fetch_archive_daily", _counting_fetch)
        dst = (tmp_path / "era5.parquet").as_posix()
        ingest = _load_climate_tools()["era5_ingest"]
        args = {"zone_id": "montpellier", "start_date": "2024-01-01", "end_date": "2024-01-03"}

        first = asyncio.run(ingest(**args, output_path=dst))
        assert first["output"] == dst and calls["n"] == 1

        # default overwrite=False → skip, no second fetch, stored coverage echoed
        skipped = asyncio.run(ingest(**args, output_path=dst))
        assert calls["n"] == 1
        assert "skipped" in skipped["metadata"]
        assert skipped["n_days"] == 3  # top-level shape consistent across all non-error paths
        assert skipped["metadata"]["stored_coverage"]["n_days"] == 3
        assert skipped["completeness"]["expected_days"] == 3
        assert skipped["completeness"]["received_days"] == 3
        assert skipped["completeness"]["pct_days"] == 100.0
        assert skipped["completeness"]["per_variable"]["temperature_2m"]["pct_complete"] == 100.0

        # overwrite=True → re-fetch + re-write
        rewritten = asyncio.run(ingest(**args, output_path=dst, overwrite=True))
        assert calls["n"] == 2
        assert "skipped" not in rewritten.get("metadata", {})
        assert rewritten["n_days"] == 3

    def test_idempotent_skip_rejects_existing_output_for_other_zone(self, tmp_path, monkeypatch):
        calls = {"n": 0}

        def _counting_fetch(**_k):
            calls["n"] += 1
            return ARCHIVE_FIXTURE

        monkeypatch.setattr("src.tools.climate.fetch_archive_daily", _counting_fetch)
        dst = (tmp_path / "era5.parquet").as_posix()
        ingest = _load_climate_tools()["era5_ingest"]

        marseille = asyncio.run(
            ingest(
                zone_id="marseille",
                start_date="2024-01-01",
                end_date="2024-01-03",
                variables=["temperature_2m"],
                output_path=dst,
            )
        )
        assert "error" not in marseille, marseille
        assert calls["n"] == 1

        montpellier = asyncio.run(
            ingest(
                zone_id="montpellier",
                start_date="2024-01-01",
                end_date="2024-01-03",
                variables=["temperature_2m"],
                output_path=dst,
            )
        )

        assert calls["n"] == 1
        assert "error" in montpellier
        assert "does not cover the requested" in montpellier["error"]

    def test_idempotent_skip_counts_distinct_dates_not_rows(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "src.tools.climate.fetch_archive_daily",
            lambda **_k: (_ for _ in ()).throw(AssertionError("must not fetch")),
        )
        dst = (tmp_path / "duplicate_sparse.parquet").as_posix()
        con = get_duckdb_connection(":memory:")
        con.execute(
            """
            CREATE TABLE existing AS
            SELECT * FROM (
              VALUES
                ('2024-01-01', 'montpellier', 8.5),
                ('2024-01-01', 'montpellier', 8.6),
                ('2024-01-03', 'montpellier', 7.8)
            ) AS t(date, zone_id, temperature_2m)
            """
        )
        con.execute(f"COPY existing TO '{dst}' (FORMAT PARQUET)")
        con.close()

        result = asyncio.run(
            _load_climate_tools()["era5_ingest"](
                zone_id="montpellier",
                start_date="2024-01-01",
                end_date="2024-01-03",
                variables=["temperature_2m"],
                output_path=dst,
            )
        )

        assert "error" in result
        assert "does not cover the requested" in result["error"]

    def test_idempotent_skip_counts_only_requested_window_days(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "src.tools.climate.fetch_archive_daily",
            lambda **_k: (_ for _ in ()).throw(AssertionError("must not fetch")),
        )
        dst = (tmp_path / "outside_dates_hide_gap.parquet").as_posix()
        con = get_duckdb_connection(":memory:")
        con.execute(
            """
            CREATE TABLE existing AS
            SELECT * FROM (
              VALUES
                ('2024-01-01', 'montpellier', 8.5),
                ('2024-01-03', 'montpellier', 7.8),
                ('2024-01-04', 'montpellier', 7.9)
            ) AS t(date, zone_id, temperature_2m)
            """
        )
        con.execute(f"COPY existing TO '{dst}' (FORMAT PARQUET)")
        con.close()

        result = asyncio.run(
            _load_climate_tools()["era5_ingest"](
                zone_id="montpellier",
                start_date="2024-01-01",
                end_date="2024-01-03",
                variables=["temperature_2m"],
                output_path=dst,
            )
        )

        assert "error" in result
        assert "does not cover the requested" in result["error"]

    def test_skip_refused_when_stored_coverage_insufficient(self, tmp_path, monkeypatch):
        # ingest a 3-day slice, then request a WIDER range with overwrite=False — must NOT silently
        # skip (which would leave only the old slice); it errors and advises overwrite=True.
        monkeypatch.setattr("src.tools.climate.fetch_archive_daily", lambda **_k: ARCHIVE_FIXTURE)
        dst = (tmp_path / "era5.parquet").as_posix()
        ingest = _load_climate_tools()["era5_ingest"]
        asyncio.run(
            ingest(
                zone_id="montpellier",
                start_date="2024-01-01",
                end_date="2024-01-03",
                output_path=dst,
            )
        )
        wider = asyncio.run(
            ingest(
                zone_id="montpellier",
                start_date="2024-01-01",
                end_date="2024-06-30",
                output_path=dst,
            )
        )
        assert "error" in wider and "does not cover" in wider["error"]
        assert "overwrite=True" in wider["error"]

    def test_write_output_false_skips_write_but_reports(self, monkeypatch):
        monkeypatch.setattr("src.tools.climate.fetch_archive_daily", lambda **_k: ARCHIVE_FIXTURE)
        result = asyncio.run(
            _load_climate_tools()["era5_ingest"](
                zone_id="montpellier",
                start_date="2024-01-01",
                end_date="2024-01-03",
                write_output=False,
            )
        )
        assert "error" not in result, result
        assert result["output"] is None
        assert result["n_days"] == 3
        assert result["completeness"]["expected_days"] == 3


# --------------------------------------------------------------------------- #
# 3. climate_enrich_ndvi — network-free (NDVI collection monkeypatched)
# --------------------------------------------------------------------------- #


class TestClimateEnrichNetworkFree:
    def test_enrich_materializes_unified_table(self, tmp_path, monkeypatch):
        # First materialize an ERA5 parquet for the zone (fixture-driven ingest).
        monkeypatch.setattr("src.tools.climate.fetch_archive_daily", lambda **_k: ARCHIVE_FIXTURE)
        era5_dst = (tmp_path / "era5.parquet").as_posix()
        unified_dst = (tmp_path / "unified.parquet").as_posix()
        tools = _load_climate_tools()
        asyncio.run(
            tools["era5_ingest"](
                zone_id="montpellier",
                start_date="2024-01-01",
                end_date="2024-01-03",
                output_path=era5_dst,
            )
        )

        # Point the enrich tool at our local ERA5 file (era5_source) and stub NDVI collection —
        # no network, no S3, no monkeypatch of the path resolver.
        async def _fake_collect(bbox, datetime_str, geometry, cloud_cover_max):
            return 2, _obs([("2024-01-02", 0.71), ("2024-01-03", 0.69)])

        monkeypatch.setattr("src.tools.climate._collect_ndvi_observations", _fake_collect)

        result = asyncio.run(
            tools["climate_enrich_ndvi"](
                zone_id="montpellier",
                start_date="2024-01-01",
                end_date="2024-01-03",
                tolerance_days=3,
                era5_source=era5_dst,
                output_path=unified_dst,
            )
        )

        assert "error" not in result, result
        assert result["statistics"]["n_matched"] == 2
        assert result["output"] == unified_dst
        assert len(result["rows"]) == 2
        assert result["rows"][0]["ndvi_mean"] in (0.71, 0.69)
        assert "temperature_2m" in result["rows"][0]  # climate carried into the unified row

        con = get_duckdb_connection(":memory:")
        n = con.execute(f"SELECT count(*) FROM read_parquet('{unified_dst}')").fetchone()[0]
        assert n == 2
        cols = {
            d[0]
            for d in con.execute(f"SELECT * FROM read_parquet('{unified_dst}') LIMIT 0").description
        }
        assert {
            "date",
            "ndvi_date",
            "ndvi_mean",
            "ndvi_std",
            "day_delta",
            "temperature_2m",
            "geometry",
        } <= cols
        # Read VALUES back (not just columns) to prove the ndvi↔climate pairing survived the write.
        paired = con.execute(
            f"SELECT ndvi_mean, day_delta, temperature_2m FROM read_parquet('{unified_dst}') "
            "WHERE date='2024-01-02'"
        ).fetchone()
        assert paired[0] == pytest.approx(0.71)
        assert paired[1] == 0
        assert paired[2] == pytest.approx(9.1)  # 2024-01-02 temperature from the fixture
        con.close()

    def _make_era5(self, tmp_path, monkeypatch) -> str:
        """Materialize a local ERA5 parquet via the fixture-driven ingest; return its path."""
        monkeypatch.setattr("src.tools.climate.fetch_archive_daily", lambda **_k: ARCHIVE_FIXTURE)
        era5_dst = (tmp_path / "era5.parquet").as_posix()
        asyncio.run(
            _load_climate_tools()["era5_ingest"](
                zone_id="montpellier",
                start_date="2024-01-01",
                end_date="2024-01-03",
                output_path=era5_dst,
            )
        )
        return era5_dst

    def test_missing_era5_source_is_a_graceful_error(self, tmp_path):
        # the ERA5 read fails before NDVI collection, so the specific cause must surface.
        # output_path must sit under the temp root of `allow_local_parquet_paths`, else the
        # path guard fires first and masks the ERA5 error — `/tmp` is not that root on macOS.
        result = asyncio.run(
            _load_climate_tools()["climate_enrich_ndvi"](
                zone_id="montpellier",
                start_date="2024-01-01",
                end_date="2024-01-03",
                era5_source="/nonexistent/era5.parquet",
                output_path=(tmp_path / "should_not_write.parquet").as_posix(),
            )
        )
        assert "error" in result and "Could not read ERA5 parquet" in result["error"]

    def test_future_date_rejected(self):
        result = asyncio.run(
            _load_climate_tools()["climate_enrich_ndvi"](
                zone_id="montpellier", start_date="2099-01-01", end_date="2099-01-03"
            )
        )
        assert "error" in result and "future" in result["error"]

    def test_unknown_zone_graceful_error(self):
        result = asyncio.run(
            _load_climate_tools()["climate_enrich_ndvi"](
                zone_id="atlantis", start_date="2024-01-01", end_date="2024-01-03"
            )
        )
        assert "error" in result and "atlantis" in result["error"]

    def test_ndvi_collection_failure_graceful_error(self, tmp_path, monkeypatch):
        era5_dst = self._make_era5(tmp_path, monkeypatch)

        async def _raise(*_a, **_k):
            raise RuntimeError("stac down")

        monkeypatch.setattr("src.tools.climate._collect_ndvi_observations", _raise)
        result = asyncio.run(
            _load_climate_tools()["climate_enrich_ndvi"](
                zone_id="montpellier",
                start_date="2024-01-01",
                end_date="2024-01-03",
                era5_source=era5_dst,
                output_path=(tmp_path / "unified.parquet").as_posix(),
            )
        )
        assert "error" in result and "NDVI collection failed" in result["error"]

    def test_no_destination_graceful_error(self, tmp_path, monkeypatch):
        era5_dst = self._make_era5(tmp_path, monkeypatch)
        monkeypatch.setattr("src.config.settings.S3_BUCKET", "")  # no derivable default destination

        async def _fake_collect(*_a, **_k):
            return 1, _obs([("2024-01-02", 0.7)])

        monkeypatch.setattr("src.tools.climate._collect_ndvi_observations", _fake_collect)
        # write_output=True (default) but no output_path and no S3_BUCKET → nowhere to write
        result = asyncio.run(
            _load_climate_tools()["climate_enrich_ndvi"](
                zone_id="montpellier",
                start_date="2024-01-01",
                end_date="2024-01-03",
                era5_source=era5_dst,
            )
        )
        assert "error" in result and "No output destination" in result["error"]

    def test_era5_source_for_other_zone_is_rejected(self, tmp_path, monkeypatch):
        # an ERA5 file holding only montpellier rows must NOT be aligned against marseille's NDVI
        era5_dst = self._make_era5(tmp_path, monkeypatch)  # montpellier rows

        async def _must_not_collect(*_a, **_k):
            raise AssertionError("must not collect — zone filter should fail first")

        monkeypatch.setattr("src.tools.climate._collect_ndvi_observations", _must_not_collect)
        result = asyncio.run(
            _load_climate_tools()["climate_enrich_ndvi"](
                zone_id="marseille",
                start_date="2024-01-01",
                end_date="2024-01-03",
                era5_source=era5_dst,
                output_path=(tmp_path / "u.parquet").as_posix(),
            )
        )
        assert "error" in result and "no rows for zone 'marseille'" in result["error"]

    def test_no_match_writes_empty_table_and_signals(self, tmp_path, monkeypatch):
        era5_dst = self._make_era5(tmp_path, monkeypatch)
        unified_dst = (tmp_path / "unified.parquet").as_posix()

        async def _far_obs(*_a, **_k):
            return 1, _obs([("2024-06-01", 0.7)])  # outside the ±3d window of the Jan ERA5 grid

        monkeypatch.setattr("src.tools.climate._collect_ndvi_observations", _far_obs)
        result = asyncio.run(
            _load_climate_tools()["climate_enrich_ndvi"](
                zone_id="montpellier",
                start_date="2024-01-01",
                end_date="2024-01-03",
                era5_source=era5_dst,
                output_path=unified_dst,
            )
        )
        assert "error" not in result, result
        assert result["statistics"]["n_matched"] == 0
        assert result["output"] == unified_dst
        assert "empty schema-valid unified table" in result["metadata"]["note"]
        assert Path(unified_dst).exists()

        con = get_duckdb_connection(":memory:")
        n = con.execute(f"SELECT count(*) FROM read_parquet('{unified_dst}')").fetchone()[0]
        cols = {
            d[0]
            for d in con.execute(f"SELECT * FROM read_parquet('{unified_dst}') LIMIT 0").description
        }
        con.close()
        assert n == 0
        assert {"date", "zone_id", "ndvi_date", "ndvi_mean", "geometry"} <= cols

    def test_ndvi_truncation_warning_when_stac_cap_hit(self, tmp_path, monkeypatch):
        from src.config import settings

        era5_dst = self._make_era5(tmp_path, monkeypatch)

        async def _capped(*_a, **_k):
            return settings.STAC_SEARCH_MAX_ITEMS, _obs([("2024-01-02", 0.7)])

        monkeypatch.setattr("src.tools.climate._collect_ndvi_observations", _capped)
        result = asyncio.run(
            _load_climate_tools()["climate_enrich_ndvi"](
                zone_id="montpellier",
                start_date="2024-01-01",
                end_date="2024-01-03",
                era5_source=era5_dst,
                output_path=(tmp_path / "u.parquet").as_posix(),
            )
        )
        assert "ndvi_truncation_warning" in result["metadata"]

    def test_unsafe_output_path_rejected_before_collection(self, monkeypatch):
        # an invalid output_path must fail fast (before the expensive NDVI collection) with the
        # right message — mirrors era5_ingest (Greptile #37)
        async def _must_not_collect(*_a, **_k):
            raise AssertionError("must not collect when output_path is invalid")

        monkeypatch.setattr("src.tools.climate._collect_ndvi_observations", _must_not_collect)
        result = asyncio.run(
            _load_climate_tools()["climate_enrich_ndvi"](
                zone_id="montpellier",
                start_date="2024-01-01",
                end_date="2024-01-03",
                era5_source="/tmp/whatever.parquet",
                output_path="s3://other-bucket/secret.parquet",
            )
        )
        assert "error" in result and "Invalid output_path" in result["error"]


# --------------------------------------------------------------------------- #
# 4. climate_percentiles — network-free climatological baseline
# --------------------------------------------------------------------------- #


class TestClimatePercentilesNetworkFree:
    def _fake_archive(self, **kwargs):
        return _archive_fixture_for_range(kwargs["start_date"], kwargs["end_date"])

    def test_percentiles_fetch_write_round_trip_and_classify(self, tmp_path, monkeypatch):
        monkeypatch.setattr("src.tools.climate.fetch_archive_daily", self._fake_archive)
        dst = (tmp_path / "normals.parquet").as_posix()
        result = asyncio.run(
            _load_climate_tools()["climate_percentiles"](
                zone_id="montpellier",
                start_date="2021-01-01",
                end_date="2022-12-31",
                output_path=dst,
                classify_observed={"temperature_2m": 15.0},
                classify_month=1,
            )
        )

        assert "error" not in result, result
        assert result["output"] == dst
        assert result["statistics"]["n_normals"] == 72
        assert result["statistics"]["n_era5_days"] == 730
        assert result["statistics"]["n_months"] == 12
        assert result["metadata"]["source"] == "archive_fetch"
        assert result["metadata"]["window_label"] == "custom"
        assert result["metadata"]["normal_grain"] == "daily_values_by_calendar_month"
        assert result["metadata"]["license"] == "CC-BY 4.0 (Open-Meteo)"
        assert "non-commercial" not in result["metadata"]["license"]
        # observed 15.0 is above Jan temperature_2m P90 (~12.25) → extreme_high band
        assert result["anomaly"]["temperature_2m"]["category"] == "extreme_high"
        assert result["anomaly"]["temperature_2m"]["value"] == 15.0
        assert result["anomaly"]["temperature_2m"]["month"] == 1
        # public rows carry no internal columns (wkt / geometry_wkt / per-row window stamp)
        assert {"wkt", "geometry_wkt", "window_start", "window_end", "window_label"}.isdisjoint(
            result["normals"][0]
        )

        jan_temp = next(
            r for r in result["normals"] if r["variable"] == "temperature_2m" and r["month"] == 1
        )
        assert jan_temp["n"] == 62
        assert jan_temp["p10"] <= jan_temp["p50"] <= jan_temp["p90"]

        coverage = _normals_coverage(dst, zone_id="montpellier")
        assert coverage["variables"]["temperature_2m"]["n_months"] == 12

        con = get_duckdb_connection(":memory:")
        n = con.execute(f"SELECT count(*) FROM read_parquet('{dst}')").fetchone()[0]
        assert n == 72
        cols = {
            d[0] for d in con.execute(f"SELECT * FROM read_parquet('{dst}') LIMIT 0").description
        }
        assert {
            "zone_id",
            "month",
            "variable",
            "p10",
            "p50",
            "p90",
            "n",
            "mean",
            "lat",
            "lon",
            "p10_wet",
            "p50_wet",
            "p90_wet",
            "n_wet",
            "wet_day_threshold_mm",
            "schema_version",
            "geometry",
        } <= cols
        persisted_wet = con.execute(
            "SELECT p10_wet, p50_wet, p90_wet, n_wet, wet_day_threshold_mm, schema_version "
            f"FROM read_parquet('{dst}') WHERE variable = 'precipitation' AND month = 1"
        ).fetchone()
        jan_precipitation = next(
            r for r in result["normals"] if r["variable"] == "precipitation" and r["month"] == 1
        )
        assert persisted_wet == tuple(
            jan_precipitation[key]
            for key in (
                "p10_wet",
                "p50_wet",
                "p90_wet",
                "n_wet",
                "wet_day_threshold_mm",
                "schema_version",
            )
        )
        persisted_not_applicable = con.execute(
            "SELECT p10_wet, p50_wet, p90_wet, n_wet, wet_day_threshold_mm, schema_version "
            f"FROM read_parquet('{dst}') WHERE variable = 'temperature_2m' AND month = 1"
        ).fetchone()
        assert persisted_not_applicable == (None, None, None, None, None, 2)
        wkt = con.execute(
            f"SELECT ST_AsText(ST_GeomFromWKB(geometry)) FROM read_parquet('{dst}') LIMIT 1"
        ).fetchone()[0]
        assert wkt == "POINT (3.5 43.6)"
        con.close()

    def test_idempotent_skip_then_overwrite(self, tmp_path, monkeypatch):
        calls = {"n": 0}

        def _counting_fetch(**kwargs):
            calls["n"] += 1
            return _archive_fixture_for_range(kwargs["start_date"], kwargs["end_date"])

        monkeypatch.setattr("src.tools.climate.fetch_archive_daily", _counting_fetch)
        dst = (tmp_path / "normals.parquet").as_posix()
        tool = _load_climate_tools()["climate_percentiles"]
        args = {
            "zone_id": "montpellier",
            "start_date": "2021-01-01",
            "end_date": "2021-12-31",
            "output_path": dst,
        }

        first = asyncio.run(tool(**args))
        assert "error" not in first, first
        first_calls = calls["n"]
        assert first_calls >= 1

        skipped = asyncio.run(tool(**args))
        assert calls["n"] == first_calls
        assert "skipped" in skipped["metadata"]
        assert skipped["statistics"]["n_normals"] == 72
        assert skipped["statistics"]["n_months"] == 12
        assert skipped["metadata"]["normal_grain"] == "daily_values_by_calendar_month"
        assert skipped["metadata"]["percentile_levels"] == [10, 50, 90]
        assert skipped["metadata"]["license"] == "CC-BY 4.0 (Open-Meteo)"
        assert "non-commercial" not in skipped["metadata"]["license"]

        subset_skip = asyncio.run(tool(**args, variables=["temperature_2m"]))
        assert calls["n"] == first_calls
        assert "skipped" in subset_skip["metadata"]
        assert subset_skip["variables"] == ["temperature_2m"]
        assert subset_skip["statistics"]["n_normals"] == 12
        assert {row["variable"] for row in subset_skip["normals"]} == {"temperature_2m"}

        rewritten = asyncio.run(tool(**args, overwrite=True))
        assert calls["n"] > first_calls
        assert "skipped" not in rewritten.get("metadata", {})

    def test_read_existing_era5_source_filters_zone_and_window_without_fetch(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.setattr("src.tools.climate.fetch_archive_daily", self._fake_archive)
        tools = _load_climate_tools()
        era5_dst = (tmp_path / "era5.parquet").as_posix()
        asyncio.run(
            tools["era5_ingest"](
                zone_id="montpellier",
                start_date="2021-01-01",
                end_date="2022-12-31",
                output_path=era5_dst,
            )
        )

        def _must_not_fetch(**_kwargs):
            raise AssertionError("covered ERA5 source should avoid Archive fetch")

        monkeypatch.setattr("src.tools.climate.fetch_archive_daily", _must_not_fetch)
        result = asyncio.run(
            tools["climate_percentiles"](
                zone_id="montpellier",
                start_date="2021-06-01",
                end_date="2021-06-30",
                variables=["temperature_2m"],
                era5_source=era5_dst,
                output_path=(tmp_path / "normals.parquet").as_posix(),
            )
        )

        assert "error" not in result, result
        assert result["metadata"]["source"] == "era5_parquet"
        assert result["statistics"]["n_era5_days"] == 30
        june = next(r for r in result["normals"] if r["month"] == 6)
        january = next(r for r in result["normals"] if r["month"] == 1)
        assert june["n"] == 30
        assert january["n"] == 0

    def test_classify_observed_requires_explicit_month(self, monkeypatch):
        monkeypatch.setattr(
            "src.tools.climate.fetch_archive_daily",
            lambda **_k: (_ for _ in ()).throw(AssertionError("must fail before fetch")),
        )
        result = asyncio.run(
            _load_climate_tools()["climate_percentiles"](
                zone_id="montpellier",
                classify_observed={"temperature_2m": 15.0},
                write_output=False,
            )
        )
        assert "error" in result and "classify_month is required" in result["error"]

    def test_unsafe_output_path_rejected_before_fetch(self, monkeypatch):
        monkeypatch.setattr(
            "src.tools.climate.fetch_archive_daily",
            lambda **_k: (_ for _ in ()).throw(AssertionError("must not fetch")),
        )
        result = asyncio.run(
            _load_climate_tools()["climate_percentiles"](
                zone_id="montpellier",
                output_path="s3://other-bucket/secret.parquet",
            )
        )
        assert "error" in result and "Invalid output_path" in result["error"]

    def test_skip_refused_for_different_window_no_mislabel(self, tmp_path, monkeypatch):
        # Write the default WMO window, then request a CUSTOM window against the SAME path with
        # overwrite=False. The normals parquet records its window, so the skip must REFUSE rather than
        # return the stored (default-window) normals mislabeled as the custom window (review P1).
        monkeypatch.setattr("src.tools.climate.fetch_archive_daily", self._fake_archive)
        dst = (tmp_path / "normals.parquet").as_posix()
        tool = _load_climate_tools()["climate_percentiles"]
        first = asyncio.run(
            tool(
                zone_id="montpellier",
                start_date="2021-01-01",
                end_date="2021-12-31",
                output_path=dst,
            )
        )
        assert "error" not in first, first

        def _must_not_fetch(**_k):
            raise AssertionError(
                "must not fetch — different window should error, not recompute here"
            )

        monkeypatch.setattr("src.tools.climate.fetch_archive_daily", _must_not_fetch)
        clash = asyncio.run(
            tool(
                zone_id="montpellier",
                start_date="2010-01-01",
                end_date="2020-12-31",
                output_path=dst,
            )
        )
        assert "error" in clash and "overwrite=True" in clash["error"]
        assert "different window" in clash["error"]

    def test_sparse_normals_not_treated_as_covered(self, tmp_path, monkeypatch):
        # A narrow (single-month) prior run leaves most months null. The coverage probe counts
        # non-null months, so a same-path re-run must NOT skip it as a complete climatology (review P2).
        monkeypatch.setattr("src.tools.climate.fetch_archive_daily", self._fake_archive)
        dst = (tmp_path / "normals.parquet").as_posix()
        tool = _load_climate_tools()["climate_percentiles"]
        sparse = asyncio.run(
            tool(
                zone_id="montpellier",
                start_date="2021-06-01",
                end_date="2021-06-30",
                variables=["temperature_2m"],
                output_path=dst,
            )
        )
        assert "error" not in sparse, sparse
        assert sparse["statistics"]["n_months"] == 1
        coverage = _normals_coverage(dst, zone_id="montpellier")
        assert coverage["variables"]["temperature_2m"]["n_months"] == 1  # only June has data
        rerun = asyncio.run(
            tool(
                zone_id="montpellier",
                start_date="2021-06-01",
                end_date="2021-06-30",
                variables=["temperature_2m"],
                output_path=dst,
            )
        )
        assert "error" in rerun and "overwrite=True" in rerun["error"]

    def _rerun_with_december(
        self, tmp_path, monkeypatch, december: str | None, month_type: str = "DOUBLE"
    ) -> dict:
        """Write a full year of normals, then re-run over it with December's month replaced.

        ``december`` is a literal for the stored month 12, written into a copy whose whole
        ``month`` column is ``month_type`` — DOUBLE is the only way a Parquet file holds ``inf``
        there. ``None`` re-runs over the file as written.
        """
        monkeypatch.setattr("src.tools.climate.fetch_archive_daily", self._fake_archive)
        tool = _load_climate_tools()["climate_percentiles"]
        args = {
            "zone_id": "montpellier",
            "start_date": "2021-01-01",
            "end_date": "2021-12-31",
            "variables": ["temperature_2m"],
        }
        dst = (tmp_path / "normals.parquet").as_posix()
        written = dst if december is None else (tmp_path / "written.parquet").as_posix()
        first = asyncio.run(tool(**args, output_path=written))
        assert "error" not in first, first
        if december is not None:
            con = get_duckdb_connection(":memory:")
            con.execute(
                "COPY (SELECT * REPLACE (CASE WHEN month = 12 THEN "
                f"CAST('{december}' AS {month_type}) ELSE CAST(month AS {month_type}) END AS month) "
                f"FROM read_parquet('{written}')) TO '{dst}' (FORMAT PARQUET)"
            )
            con.close()

        def _must_not_fetch(**_k):
            raise AssertionError("an existing output is skipped or refused, never recomputed")

        monkeypatch.setattr("src.tools.climate.fetch_archive_daily", _must_not_fetch)
        return asyncio.run(tool(**args, output_path=dst))

    @pytest.mark.parametrize(
        ("december", "month_type"),
        [
            ("inf", "DOUBLE"),
            ("nan", "DOUBLE"),
            ("0", "DOUBLE"),
            ("11.5", "DOUBLE"),
            ("December", "VARCHAR"),
        ],
        ids=["inf", "nan", "0", "11.5", "varchar-word"],
    )
    def test_a_stored_month_that_is_not_a_month_is_refused_not_raised(
        self, tmp_path, monkeypatch, december, month_type
    ):
        # A legacy or hand-edited file holding inf (or NaN) where December's month should be.
        # COUNT(DISTINCT month) counted it as the twelfth month, the skip trusted the file, and
        # `_normals_month_count` then raised OverflowError (ValueError for NaN) out of the tool
        # instead of its documented refusal (#221). A 0, an 11.5 or a word is no December
        # either; the word must still leave the other eleven months countable.
        result = self._rerun_with_december(tmp_path, monkeypatch, december, month_type)

        assert "exists but its stored normals do not cover" in result["error"]
        assert "overwrite=True" in result["error"]
        coverage = result["metadata"]["stored_coverage"]["variables"]["temperature_2m"]
        assert coverage["n_months"] == 11

    @pytest.mark.parametrize(
        ("december", "month_type"),
        [(None, "DOUBLE"), ("12", "DOUBLE"), ("12", "VARCHAR")],
        ids=["as-written", "double-12", "varchar-12"],
    )
    def test_a_full_year_still_takes_the_skip_whatever_the_month_column_type(
        self, tmp_path, monkeypatch, december, month_type
    ):
        # The control for the refusal above: the same file with a real December — as written, or
        # with a DOUBLE or a legacy VARCHAR month column — is still covered and skipped.
        result = self._rerun_with_december(tmp_path, monkeypatch, december, month_type)

        assert "error" not in result, result
        assert "skipped" in result["metadata"]
        assert result["statistics"]["n_months"] == 12

    @pytest.mark.parametrize(
        "stray", [math.inf, math.nan, 13, 0, 11.5], ids=["inf", "nan", "13", "0", "11.5"]
    )
    def test_the_month_count_counts_only_months_one_to_twelve(self, stray):
        # A stored month outside 1..12 counts for nothing, as `_normals_index` already reads it:
        # it neither raises (inf: OverflowError, NaN: ValueError) nor makes up a twelfth month.
        eleven = [{"variable": "temperature_2m", "month": m, "p50": 1.0} for m in range(1, 12)]

        def with_month(month):
            return [*eleven, {"variable": "temperature_2m", "month": month, "p50": 1.0}]

        assert _normals_month_count(with_month(stray)) == 11
        assert _normals_month_count(with_month(12.0)) == 12

    def test_classify_on_empty_month_returns_clear_no_data(self, tmp_path, monkeypatch):
        # Normals over a June-only window have n=0 for January; classifying a Jan value must return a
        # clear no-data message, not a cryptic float() TypeError (review P3).
        monkeypatch.setattr("src.tools.climate.fetch_archive_daily", self._fake_archive)
        result = asyncio.run(
            _load_climate_tools()["climate_percentiles"](
                zone_id="montpellier",
                start_date="2021-06-01",
                end_date="2021-06-30",
                variables=["temperature_2m"],
                write_output=False,
                classify_observed={"temperature_2m": 15.0},
                classify_month=1,
            )
        )
        assert "error" not in result, result
        anomaly = result["anomaly"]["temperature_2m"]
        assert "No climatology data" in anomaly["error"] and anomaly["n"] == 0

    def test_unknown_variables_graceful_error(self):
        result = asyncio.run(
            _load_climate_tools()["climate_percentiles"](
                zone_id="montpellier", variables=["bogus"], write_output=False
            )
        )
        assert "error" in result and "Unknown variables" in result["error"]

    def test_unknown_zone_graceful_error(self, monkeypatch):
        monkeypatch.setattr(
            "src.tools.climate.fetch_archive_daily",
            lambda **_k: (_ for _ in ()).throw(AssertionError("must not fetch")),
        )
        result = asyncio.run(
            _load_climate_tools()["climate_percentiles"](zone_id="atlantis", write_output=False)
        )
        assert "error" in result and "atlantis" in result["error"]

    def test_explicit_unreadable_era5_source_errors(self):
        result = asyncio.run(
            _load_climate_tools()["climate_percentiles"](
                zone_id="montpellier",
                start_date="2021-01-01",
                end_date="2021-12-31",
                era5_source="/nonexistent/era5.parquet",
                write_output=False,
            )
        )
        assert "error" in result and "Could not read ERA5 source" in result["error"]

    def test_empty_variables_list_rejected(self, monkeypatch):
        # an EXPLICIT empty list is malformed — must error, not silently widen to all six (Codex P2)
        monkeypatch.setattr(
            "src.tools.climate.fetch_archive_daily",
            lambda **_k: (_ for _ in ()).throw(AssertionError("must not fetch")),
        )
        result = asyncio.run(
            _load_climate_tools()["climate_percentiles"](
                zone_id="montpellier", variables=[], write_output=False
            )
        )
        assert "error" in result and "variables must not be empty" in result["error"]

    def test_explicit_era5_source_not_covering_errors(self, tmp_path, monkeypatch):
        # an explicit, readable era5_source that does NOT fully cover the window must error rather than
        # silently trigger a multi-year Archive fetch from a different source (Greptile P2).
        monkeypatch.setattr("src.tools.climate.fetch_archive_daily", self._fake_archive)
        tools = _load_climate_tools()
        era5_dst = (tmp_path / "era5.parquet").as_posix()
        asyncio.run(
            tools["era5_ingest"](
                zone_id="montpellier",
                start_date="2021-06-01",
                end_date="2021-06-30",  # only June
                output_path=era5_dst,
            )
        )

        def _must_not_fetch(**_k):
            raise AssertionError("explicit non-covering source must error, not fetch")

        monkeypatch.setattr("src.tools.climate.fetch_archive_daily", _must_not_fetch)
        result = asyncio.run(
            tools["climate_percentiles"](
                zone_id="montpellier",
                start_date="2021-01-01",
                end_date="2021-12-31",  # wider than the source
                variables=["temperature_2m"],
                era5_source=era5_dst,
                write_output=False,
            )
        )
        assert "error" in result and "does not fully cover" in result["error"]


# --------------------------------------------------------------------------- #
# 5. Live Archive contract probe (skipped offline)
# --------------------------------------------------------------------------- #


@pytest.mark.skipif(not _RUN_LIVE, reason="set OPENMETEO_LIVE=1 to hit the live Archive API")
class TestArchiveLiveContract:
    def test_daily_variable_names_unchanged(self):
        from src.utils.openmeteo import PROVIDER_VARS, fetch_archive_daily

        raw = fetch_archive_daily(
            latitude=43.6,
            longitude=3.5,
            start_date="2024-01-01",
            end_date="2024-01-03",
            normalized_variables=list(NORMALIZED_VARS),
        )
        daily = raw["daily"]
        units = raw.get("daily_units", {})
        assert "time" in daily and daily["time"]
        # The exact provider variable names AND units are the CONTRACT — a rename or a unit change
        # fails loudly here (e.g. shortwave_radiation_sum drifting from MJ/m² would corrupt the
        # MJ→W conversion silently otherwise).
        for pv in PROVIDER_VARS:
            assert pv.provider in daily, f"Open-Meteo rename: missing {pv.provider!r}"
            assert len(daily[pv.provider]) == len(daily["time"])
            assert units.get(pv.provider) == pv.source_unit, (
                f"Open-Meteo unit drift on {pv.provider!r}: "
                f"expected {pv.source_unit!r}, got {units.get(pv.provider)!r}"
            )


# --------------------------------------------------------------------------- #
# normals schema v2 — the precipitation wet-day triplet (#172, Site-Watch V0 G2)
# --------------------------------------------------------------------------- #


def _precip_rows(zone_id: str, month_values: dict[int, list[float]]) -> list[dict]:
    """Daily rows carrying only `precipitation`, one calendar month at a time."""
    rows = []
    for month, values in month_values.items():
        for day, value in enumerate(values, start=1):
            rows.append(
                {"date": f"2021-{month:02d}-{day:02d}", "zone_id": zone_id, "precipitation": value}
            )
    return rows


def _normals_for(rows: list[dict], requested: list[str]) -> list[dict]:
    return compute_monthly_normals(
        rows,
        zone_id="z",
        lat=0.0,
        lon=0.0,
        requested=requested,
        window_start="2021-01-01",
        window_end="2021-12-31",
        window_label="custom",
    )


class TestTheWetDayTripletSitsBesideTheAllDaysOne:
    """The all-days percentiles of precipitation include dry days, which drag P90 down.

    A month that is dry for most of itself and pours on a few days has an all-days P90 below several
    of its own wet days, so banding heavy rain against it understates every one of them. The wet-day
    distribution is added BESIDE the all-days one, never in place of it — the all-days figures stay
    the answer for anything that is not a heavy-rain question.
    """

    def test_the_wet_percentiles_exclude_the_dry_days(self):
        """25 dry days and 5 wet ones — and the all-days median is therefore 0.0 mm."""
        rows = _precip_rows("z", {1: [0.0] * 25 + [4.0, 6.0, 8.0, 12.0, 30.0]})

        jan = next(r for r in _normals_for(rows, ["precipitation"]) if r["month"] == 1)

        assert jan["n"] == 30
        assert jan["n_wet"] == 5
        assert jan["wet_day_threshold_mm"] == 1.0
        # The dry days own the low half of the all-days distribution outright.
        assert (jan["p10"], jan["p50"], jan["p90"]) == (0.0, 0.0, 6.2)
        assert (jan["p10_wet"], jan["p50_wet"], jan["p90_wet"]) == (4.8, 8.0, 22.8)
        assert jan["p10_wet"] <= jan["p50_wet"] <= jan["p90_wet"]

    def test_banding_rain_against_the_all_days_distribution_calls_a_normal_day_extreme(self):
        """The consequence, stated as a classification rather than as a percentile comparison.

        With an all-days P90 of 6.2 mm, an ordinary 8 mm day reads `extreme_high` — and so does a
        12 mm one, and so would anything wetter. On its own wet-day distribution 8 mm is exactly the
        median. A digest banding precipitation against the all-days figures would therefore call
        every rainy day extreme, which is not a threshold being slightly off but a variable whose
        low half is dry days.
        """
        rows = _precip_rows("z", {1: [0.0] * 25 + [4.0, 6.0, 8.0, 12.0, 30.0]})
        jan = next(r for r in _normals_for(rows, ["precipitation"]) if r["month"] == 1)

        all_days = classify_against_normals(8.0, p10=jan["p10"], p50=jan["p50"], p90=jan["p90"])
        wet_days = classify_against_normals(
            8.0, p10=jan["p10_wet"], p50=jan["p50_wet"], p90=jan["p90_wet"]
        )

        assert all_days["category"] == "extreme_high"
        assert wet_days["category"] == "above_normal"

    def test_the_threshold_is_inclusive(self):
        """`>= 1.0 mm` is the Climdex wet day; exactly 1.0 mm is wet, not a boundary omission."""
        rows = _precip_rows("z", {1: [0.9, 1.0, 1.1]})

        jan = next(r for r in _normals_for(rows, ["precipitation"]) if r["month"] == 1)

        assert jan["n"] == 3
        assert jan["n_wet"] == 2

    def test_a_non_precipitation_variable_reports_not_applicable_not_zero(self):
        """`None`, not 0. Temperature has no dry day to exclude, so the question does not apply.

        Writing `n_wet: 0` here would claim a measured absence of wet days for a temperature row.
        """
        rows = [
            {"date": f"2021-01-{d:02d}", "zone_id": "z", "temperature_2m": 10.0 + d}
            for d in range(1, 31)
        ]

        jan = next(r for r in _normals_for(rows, ["temperature_2m"]) if r["month"] == 1)

        assert jan["n"] == 30
        assert jan["n_wet"] is None
        assert jan["p10_wet"] is None
        assert jan["p50_wet"] is None
        assert jan["p90_wet"] is None
        assert jan["wet_day_threshold_mm"] is None

    def test_an_arid_month_reports_a_measured_absence(self):
        """The arid-zone trap, and the distinction a single NULL would destroy.

        `niamey` records 0.0 mm for January, February, November and December across the whole
        1991-2020 window. `n_wet = 0` with `p*_wet = None` says "we looked and there were none";
        `n_wet = None` says "the question does not apply". Collapsing them would make valid
        arid-zone data indistinguishable from a variable that has no wet-day reading at all.
        """
        rows = _precip_rows("z", {1: [0.0] * 31})

        jan = next(r for r in _normals_for(rows, ["precipitation"]) if r["month"] == 1)

        assert jan["n"] == 31
        assert jan["n_wet"] == 0  # measured, not missing
        assert jan["p90_wet"] is None  # nothing to compute a percentile over
        assert jan["wet_day_threshold_mm"] == 1.0  # the threshold still applied

    @pytest.mark.parametrize(
        "rows",
        [
            pytest.param([], id="no-rows"),
            pytest.param(
                [
                    {"date": "2021-01-01", "zone_id": "z", "precipitation": None},
                    {"date": "2021-01-02", "zone_id": "z", "precipitation": "invalid"},
                    {"date": "2021-01-03", "zone_id": "z", "precipitation": float("nan")},
                ],
                id="all-values-invalid",
            ),
        ],
    )
    def test_an_unobserved_precipitation_month_is_not_reported_as_all_dry(self, rows):
        """No valid sample is missing data, not a measured absence of wet days."""
        jan = next(r for r in _normals_for(rows, ["precipitation"]) if r["month"] == 1)

        assert jan["n"] == 0
        assert jan["n_wet"] is None
        assert jan["p10_wet"] is None
        assert jan["p50_wet"] is None
        assert jan["p90_wet"] is None
        assert jan["wet_day_threshold_mm"] == 1.0

    def test_every_row_is_stamped_with_the_schema_version(self):
        rows = _precip_rows("z", {1: [2.0] * 31})

        normals = _normals_for(rows, ["precipitation", "temperature_2m"])

        assert {r["schema_version"] for r in normals} == {2}


class TestTheCoverageGateIsVersionAwareAndDoesNotJudgeWetDays:
    """Two ways this could silently force an Archive refetch, which is what #172 forbids."""

    def _write(self, path: str, *, with_v2_columns: bool) -> None:
        con = get_duckdb_connection(":memory:")
        v2 = ", 0 AS n_wet, NULL::DOUBLE AS p90_wet, 2 AS schema_version" if with_v2_columns else ""
        con.execute(
            "COPY (SELECT m AS month, 'z' AS zone_id, 'precipitation' AS variable, "
            "1.0 AS p10, 2.0 AS p50, 3.0 AS p90, 30 AS n, 2.0 AS mean, "
            "'1991-01-01' AS window_start, '2020-12-31' AS window_end, "
            f"'WMO_1991_2020' AS window_label{v2} "
            f"FROM range(1, 13) AS t(m)) TO '{path}' (FORMAT PARQUET)"
        )
        con.close()

    def test_a_v1_artifact_reports_version_one_and_still_satisfies_coverage(self, tmp_path):
        """Selecting `schema_version` unconditionally would raise, return {}, and refetch.

        A v1 file is not broken — it satisfies the all-days contract it was written for. Promoting
        it to v2 belongs to the normals-only operator path, with the promotion/rollback that path is
        required to have; it must not happen as a side effect of a coverage check.
        """
        dst = (tmp_path / "normals_v1.parquet").as_posix()
        self._write(dst, with_v2_columns=False)

        coverage = _normals_coverage(dst, zone_id="z")

        assert coverage != {}, "a v1 artifact must not read as unreadable"
        assert coverage["schema_version"] == 1
        assert coverage["wet_day"] is None
        assert _normals_coverage_satisfies(coverage, ["precipitation"]) is True

    def test_a_v2_artifact_whose_every_month_is_dry_still_satisfies_coverage(self, tmp_path):
        """`n_wet = 0` on all twelve months is `niamey`, not a coverage hole."""
        dst = (tmp_path / "normals_v2.parquet").as_posix()
        self._write(dst, with_v2_columns=True)

        coverage = _normals_coverage(dst, zone_id="z")

        assert coverage["schema_version"] == 2
        assert coverage["wet_day"] == {"rows_with_n_wet": 12, "rows_with_no_wet_day": 12}
        assert _normals_coverage_satisfies(coverage, ["precipitation"]) is True


class TestTheV2LayoutIsAPrefixExtensionOfV1:
    """PR #189 review — "additive" has to be true of the bytes, not only of the readers we know."""

    def test_the_written_column_order_keeps_every_v1_column_at_its_v1_index(
        self, tmp_path, monkeypatch
    ):
        """Every v1 DATA column keeps its index, and `geometry` stays last.

        Grouped beside `mean` — where they read better — the wet columns pushed `window_start` and
        everything after it to new ordinals, so a positional v1 reader got `p10_wet` where it
        expected `window_start`.

        A literal byte-prefix is not achievable and this does not assert one: `_write_geoparquet`
        always appends the WKB `geometry` column last, so it moves from index 13 to 19. Relaying out
        that shared writer to fix one ordinal would change the era5, unified and ndvi artifacts too.
        """
        monkeypatch.setattr("src.tools.climate.fetch_archive_daily", lambda **_k: ARCHIVE_FIXTURE)
        dst = (tmp_path / "normals.parquet").as_posix()
        asyncio.run(
            _load_climate_tools()["climate_percentiles"](
                zone_id="montpellier",
                start_date="2024-01-01",
                end_date="2024-01-03",
                output_path=dst,
            )
        )

        con = get_duckdb_connection(":memory:")
        names = [
            r[0] for r in con.execute(f"DESCRIBE SELECT * FROM read_parquet('{dst}')").fetchall()
        ]
        missing_month = con.execute(
            "SELECT n, p10_wet, p50_wet, p90_wet, n_wet, wet_day_threshold_mm, schema_version "
            f"FROM read_parquet('{dst}') WHERE variable = 'precipitation' AND month = 2"
        ).fetchone()
        con.close()

        v1_data_columns = [
            "month",
            "zone_id",
            "variable",
            "p10",
            "p50",
            "p90",
            "n",
            "mean",
            "window_start",
            "window_end",
            "window_label",
            "lat",
            "lon",
        ]
        # Every v1 data column at its v1 index.
        assert names[: len(v1_data_columns)] == v1_data_columns, f"a v1 column moved: {names}"
        # `geometry` still last, as `_write_geoparquet` has always guaranteed — it appends the WKB
        # column at the end wherever `wkt` sits in the staging list, which is why a literal
        # byte-prefix is not achievable here and is not claimed.
        assert names[-1] == "geometry"
        assert names[len(v1_data_columns) : -1] == [
            "p10_wet",
            "p50_wet",
            "p90_wet",
            "n_wet",
            "wet_day_threshold_mm",
            "schema_version",
        ]
        assert missing_month == (0, None, None, None, None, 1.0, 2)

    def test_a_null_schema_version_reads_as_v1_rather_than_being_skipped(self, tmp_path):
        """SQL `MIN` skips nulls, so a mixed file reported the NEWEST version it contained.

        That is the opposite of what taking the minimum was for. A NULL version is what a row
        written before the column existed looks like once the column does: it is v1.
        """
        dst = (tmp_path / "mixed.parquet").as_posix()
        con = get_duckdb_connection(":memory:")
        con.execute(
            "COPY (SELECT m AS month, 'z' AS zone_id, 'precipitation' AS variable, "
            "1.0 AS p10, 2.0 AS p50, 3.0 AS p90, 30 AS n, 2.0 AS mean, "
            "'1991-01-01' AS window_start, '2020-12-31' AS window_end, "
            "'WMO_1991_2020' AS window_label, "
            # one legacy row with a NULL version, the rest stamped v2
            "CASE WHEN m = 1 THEN NULL ELSE 2 END::INTEGER AS schema_version "
            f"FROM range(1, 13) AS t(m)) TO '{dst}' (FORMAT PARQUET)"
        )
        con.close()

        coverage = _normals_coverage(dst, zone_id="z")

        assert coverage["schema_version"] == 1


# --------------------------------------------------------------------------- #
# 3b. climate_enrich_ndvi(ndvi_source=…) — the B3 read path (#177)
# --------------------------------------------------------------------------- #


def _write_b3_series(path: str, rows: list[tuple[str, str, float]]) -> None:
    """A B3-shaped `climate/ndvi/{zone}.parquet` — `(date, zone_id, ndvi_mean)` triples.

    Written with DuckDB rather than through `ndvi_ingest` on purpose: the tool needs STAC and COG
    reads, and the point of every test below is that the read path touches NEITHER.
    """
    con = get_duckdb_connection(":memory:")
    values = ", ".join(f"(DATE '{d}', '{z}', {n})" for d, z, n in rows)
    con.execute(
        f"COPY (SELECT * FROM (VALUES {values}) AS t(date, zone_id, ndvi_mean)) "
        f"TO '{path}' (FORMAT PARQUET)"
    )
    con.close()


class TestTheB3SeriesIsReadableAsAB4Input:
    """#177 — `ndvi_ingest` wrote `climate/ndvi/{zone}.parquet` and nothing read it.

    `climate_enrich_ndvi` re-derived NDVI from STAC on every call, so #133's "depends on the
    durable B3 materialization" named a dependency the code had no way to express. Worse on the
    multi-year window #133 is about: the B4 collector's 50-item STAC cap applies to the WHOLE
    window while B3's applies per month, so streaming got strictly LESS data than the artifact
    already in the bucket.
    """

    ERA5_FIXTURE_ZONE = "montpellier"

    def _era5(self, tmp_path, monkeypatch) -> str:
        monkeypatch.setattr("src.tools.climate.fetch_archive_daily", lambda **_k: ARCHIVE_FIXTURE)
        dst = (tmp_path / "era5.parquet").as_posix()
        asyncio.run(
            _load_climate_tools()["era5_ingest"](
                zone_id=self.ERA5_FIXTURE_ZONE,
                start_date="2024-01-01",
                end_date="2024-01-03",
                output_path=dst,
            )
        )
        return dst

    def _explode_on_collection(self, monkeypatch) -> dict:
        """Make ANY provider call a test failure, and record that it never happened."""
        seen = {"called": False}

        async def _boom(*_a, **_k):
            seen["called"] = True
            raise AssertionError("the explicit-source path must not reach a provider")

        monkeypatch.setattr("src.tools.climate._collect_ndvi_observations", _boom)
        return seen

    def test_an_explicit_source_is_read_without_touching_a_provider(self, tmp_path, monkeypatch):
        """The contract, not an optimization: `ndvi_source` means "use THIS", not "prefer this"."""
        era5 = self._era5(tmp_path, monkeypatch)
        seen = self._explode_on_collection(monkeypatch)
        ndvi = (tmp_path / "ndvi.parquet").as_posix()
        _write_b3_series(
            ndvi,
            [
                ("2024-01-02", self.ERA5_FIXTURE_ZONE, 0.71),
                ("2024-01-03", self.ERA5_FIXTURE_ZONE, 0.69),
            ],
        )

        result = asyncio.run(
            _load_climate_tools()["climate_enrich_ndvi"](
                zone_id=self.ERA5_FIXTURE_ZONE,
                start_date="2024-01-01",
                end_date="2024-01-03",
                era5_source=era5,
                ndvi_source=ndvi,
                write_output=False,
            )
        )

        assert "error" not in result, result
        assert seen["called"] is False
        assert result["statistics"]["n_ndvi_observations"] == 2
        assert result["statistics"]["images_searched"] == 0
        assert result["metadata"]["ndvi_input"] == "stored_b3_series"
        assert result["metadata"]["ndvi_source"] == ndvi
        # The cap belongs to the STAC search; no search happened, so no warning may be invented.
        assert "ndvi_truncation_warning" not in result["metadata"]

    def test_acquisition_dates_are_carried_through_verbatim(self, tmp_path, monkeypatch):
        """`date` in the B3 artifact IS the Sentinel-2 scene date and must survive as `ndvi_date`."""
        era5 = self._era5(tmp_path, monkeypatch)
        self._explode_on_collection(monkeypatch)
        ndvi = (tmp_path / "ndvi.parquet").as_posix()
        _write_b3_series(ndvi, [("2024-01-02", self.ERA5_FIXTURE_ZONE, 0.71)])

        result = asyncio.run(
            _load_climate_tools()["climate_enrich_ndvi"](
                zone_id=self.ERA5_FIXTURE_ZONE,
                start_date="2024-01-01",
                end_date="2024-01-03",
                era5_source=era5,
                ndvi_source=ndvi,
                write_output=False,
            )
        )

        assert result["rows"][0]["ndvi_date"] == "2024-01-02"
        assert result["rows"][0]["ndvi_mean"] == 0.71

    def test_an_unreadable_source_is_an_error_and_never_a_fallback(self, tmp_path, monkeypatch):
        """The whole point: a caller who named an artifact asked for THAT artifact.

        Degrading to collection would answer a different question with data that merely resembles
        the answer — and would do it silently, since the result shape is identical.
        """
        era5 = self._era5(tmp_path, monkeypatch)
        seen = self._explode_on_collection(monkeypatch)

        result = asyncio.run(
            _load_climate_tools()["climate_enrich_ndvi"](
                zone_id=self.ERA5_FIXTURE_ZONE,
                start_date="2024-01-01",
                end_date="2024-01-03",
                era5_source=era5,
                ndvi_source=(tmp_path / "absent.parquet").as_posix(),
                write_output=False,
            )
        )

        assert "error" in result
        assert "ndvi_source" in result["error"]
        assert seen["called"] is False

    def test_a_parquet_that_is_not_an_ndvi_series_is_refused(self, tmp_path, monkeypatch):
        """Readable is not the same as usable — an ERA5 file has no `ndvi_mean`."""
        era5 = self._era5(tmp_path, monkeypatch)
        self._explode_on_collection(monkeypatch)

        result = asyncio.run(
            _load_climate_tools()["climate_enrich_ndvi"](
                zone_id=self.ERA5_FIXTURE_ZONE,
                start_date="2024-01-01",
                end_date="2024-01-03",
                era5_source=era5,
                ndvi_source=era5,  # a real parquet, the wrong KIND of one
                write_output=False,
            )
        )

        assert "error" in result
        assert "ndvi_mean" in result["error"]

    def test_a_source_holding_another_zone_is_refused(self, tmp_path, monkeypatch):
        """The same guard the ERA5 side has, and for the same reason.

        Aligning one zone's NDVI onto another's climate produces a table that is wrong in a way
        nothing downstream can detect.
        """
        era5 = self._era5(tmp_path, monkeypatch)
        self._explode_on_collection(monkeypatch)
        ndvi = (tmp_path / "ndvi.parquet").as_posix()
        _write_b3_series(ndvi, [("2024-01-02", "marseille", 0.71)])

        result = asyncio.run(
            _load_climate_tools()["climate_enrich_ndvi"](
                zone_id=self.ERA5_FIXTURE_ZONE,
                start_date="2024-01-01",
                end_date="2024-01-03",
                era5_source=era5,
                ndvi_source=ndvi,
                write_output=False,
            )
        )

        assert "error" in result
        assert "no rows for zone" in result["error"]

    def test_observations_outside_the_requested_window_are_dropped(self, tmp_path, monkeypatch):
        """Pointing at a decade-long artifact for a three-day run must report three days."""
        era5 = self._era5(tmp_path, monkeypatch)
        self._explode_on_collection(monkeypatch)
        ndvi = (tmp_path / "ndvi.parquet").as_posix()
        _write_b3_series(
            ndvi,
            [
                ("2019-06-15", self.ERA5_FIXTURE_ZONE, 0.40),  # long before the window
                ("2024-01-02", self.ERA5_FIXTURE_ZONE, 0.71),
                ("2031-01-02", self.ERA5_FIXTURE_ZONE, 0.90),  # long after it
            ],
        )

        result = asyncio.run(
            _load_climate_tools()["climate_enrich_ndvi"](
                zone_id=self.ERA5_FIXTURE_ZONE,
                start_date="2024-01-01",
                end_date="2024-01-03",
                era5_source=era5,
                ndvi_source=ndvi,
                write_output=False,
            )
        )

        assert "error" not in result, result
        assert result["statistics"]["n_ndvi_observations"] == 1
        assert result["metadata"]["ndvi_coverage"]["first_observation"] == "2024-01-02"

    def test_coverage_reports_observed_months_against_requested_months(self, tmp_path, monkeypatch):
        """A truncation warning that stops appearing must not read as complete coverage.

        The B3 pilots are a monthly least-cloudy cadence over the months that HAVE a usable scene
        — 81 of 120 for montpellier_metropole. The gap is a property of the data, and removing the
        STAC cap does not fill it.
        """
        era5 = self._era5(tmp_path, monkeypatch)
        self._explode_on_collection(monkeypatch)
        ndvi = (tmp_path / "ndvi.parquet").as_posix()
        _write_b3_series(ndvi, [("2024-01-02", self.ERA5_FIXTURE_ZONE, 0.71)])

        result = asyncio.run(
            _load_climate_tools()["climate_enrich_ndvi"](
                zone_id=self.ERA5_FIXTURE_ZONE,
                start_date="2024-01-01",
                end_date="2024-03-31",  # three months requested, one observed
                era5_source=era5,
                ndvi_source=ndvi,
                write_output=False,
            )
        )

        coverage = result["metadata"]["ndvi_coverage"]
        assert coverage["months_requested"] == 3
        assert coverage["months_observed"] == 1
        assert coverage["max_observations_in_a_month"] == 1
        note = result["metadata"]["ndvi_coverage_note"]
        assert "1 of 3 requested months" in note
        assert "absent from the stored series" in note
        assert "does not encode why" in note
        assert "no usable scene" not in note

    def test_collapsed_note_states_both_selection_steps_for_a_day_delta_tie(
        self, tmp_path, monkeypatch
    ):
        era5 = self._era5(tmp_path, monkeypatch)
        self._explode_on_collection(monkeypatch)
        ndvi = (tmp_path / "ndvi.parquet").as_posix()
        _write_b3_series(
            ndvi,
            [
                ("2024-01-02", self.ERA5_FIXTURE_ZONE, 0.40),
                ("2024-01-02", self.ERA5_FIXTURE_ZONE, 0.70),
            ],
        )

        result = asyncio.run(
            _load_climate_tools()["climate_enrich_ndvi"](
                zone_id=self.ERA5_FIXTURE_ZONE,
                start_date="2024-01-01",
                end_date="2024-01-03",
                era5_source=era5,
                ndvi_source=ndvi,
                write_output=False,
            )
        )

        assert result["statistics"]["n_ndvi_observations"] == 2
        assert result["statistics"]["n_matched"] == 1
        assert result["rows"][0]["ndvi_mean"] == 0.70
        assert result["metadata"]["ndvi_collapsed_onto_shared_days"] == 1
        note = result["metadata"]["ndvi_collapsed_note"]
        assert "minimum day_delta" in note
        assert "higher NDVI when day_delta ties" in note

    def test_observations_past_the_end_of_the_era5_record_are_reported(self, tmp_path, monkeypatch):
        """Measured on the dev harness, in miniature.

        The stored ERA5 is the WMO 1991-2020 baseline while the B3 pilot series runs to 2024, so a
        2016-2024 request aligned **40 of 81** observations — and the unified table reported its
        own row count as though that were the whole story.
        """
        era5 = self._era5(tmp_path, monkeypatch)  # covers 2024-01-01..2024-01-03 only
        self._explode_on_collection(monkeypatch)
        ndvi = (tmp_path / "ndvi.parquet").as_posix()
        _write_b3_series(
            ndvi,
            [
                ("2024-01-02", self.ERA5_FIXTURE_ZONE, 0.71),  # inside the ERA5 record
                ("2024-06-15", self.ERA5_FIXTURE_ZONE, 0.55),  # past its end
                ("2024-07-15", self.ERA5_FIXTURE_ZONE, 0.52),  # past its end
            ],
        )

        result = asyncio.run(
            _load_climate_tools()["climate_enrich_ndvi"](
                zone_id=self.ERA5_FIXTURE_ZONE,
                start_date="2024-01-01",
                end_date="2024-07-31",
                era5_source=era5,
                ndvi_source=ndvi,
                write_output=False,
            )
        )

        assert result["statistics"]["n_ndvi_observations"] == 3
        assert result["statistics"]["n_matched"] == 1
        assert result["metadata"]["ndvi_out_of_era5_range"] == 2
        assert result["metadata"]["era5_window"] == "2024-01-01..2024-01-03"
        note = result["metadata"]["ndvi_out_of_era5_range_note"]
        assert "2 of 3 NDVI observations with parseable dates" in note
        # These two really were outside the record, so nothing may be filed as "collapsed".
        assert "ndvi_collapsed_onto_shared_days" not in result["metadata"]

    def test_a_source_outside_the_confined_locations_is_refused(self, tmp_path, monkeypatch):
        """`ndvi_source` is agent-controlled and interpolated into `read_parquet`.

        It gets `_validate_dst` — the same confinement `era5_source` has — so an `http(s)://`
        source cannot turn a manifest run into a request to a cloud-metadata or internal host.
        """
        era5 = self._era5(tmp_path, monkeypatch)
        seen = self._explode_on_collection(monkeypatch)

        result = asyncio.run(
            _load_climate_tools()["climate_enrich_ndvi"](
                zone_id=self.ERA5_FIXTURE_ZONE,
                start_date="2024-01-01",
                end_date="2024-01-03",
                era5_source=era5,
                ndvi_source="http://169.254.169.254/latest/meta-data/iam/ndvi.parquet",
                write_output=False,
            )
        )

        assert "error" in result
        assert "ndvi_source" in result["error"]
        assert seen["called"] is False


class TestTheAlignmentCensusSeparatesAGapFromAResolutionLimit:
    """PR #188 review — `len(ndvi_obs) - len(unified)` conflated two opposite outcomes.

    `_align_ndvi_to_era5` keeps ONE row per ERA5 date, so observations either side of a single ERA5
    day collapse into one row. Subtracting row count from observation count therefore labelled them
    "had no ERA5 day", which was simply false — and false in the direction that invents a coverage
    gap where the data is complete.
    """

    def test_two_observations_around_one_era5_day_are_not_called_unmatched(self):
        """The measured case: 2 observations, 1 row, and both matched."""
        era5 = [
            {"date": "2024-01-02", "zone_id": "z", "lat": 0.0, "lon": 0.0, "temperature_2m": 5.0}
        ]
        obs = [{"date": "2024-01-01", "ndvi_mean": 0.5}, {"date": "2024-01-03", "ndvi_mean": 0.6}]

        census = _ndvi_alignment_census(era5, obs, tolerance_days=3)

        assert census["matchable"] == 2
        assert census["out_of_range"] == 0
        assert census["unreadable"] == 0

    def test_an_observation_past_the_record_is_out_of_range(self):
        era5 = [{"date": "2024-01-02", "zone_id": "z", "lat": 0.0, "lon": 0.0}]
        obs = [{"date": "2024-01-02", "ndvi_mean": 0.5}, {"date": "2024-09-02", "ndvi_mean": 0.6}]

        census = _ndvi_alignment_census(era5, obs, tolerance_days=3)

        assert census["matchable"] == 1
        assert census["out_of_range"] == 1

    def test_an_unreadable_date_is_its_own_bucket(self):
        """Neither a coverage gap nor a resolution limit — counting it as either would mislabel it."""
        era5 = [{"date": "2024-01-02", "zone_id": "z", "lat": 0.0, "lon": 0.0}]
        obs = [{"date": "not-a-date", "ndvi_mean": 0.5}, {"date": None, "ndvi_mean": 0.6}]

        census = _ndvi_alignment_census(era5, obs, tolerance_days=3)

        assert census == {"matchable": 0, "unreadable": 2, "out_of_range": 0}


class TestASuppliedButEmptySourceIsNotTreatedAsOmitted:
    """PR #188 review — `if ndvi_source:` read `""` as "not supplied" and called the provider.

    That is the no-fallback contract broken by the one value most likely to arrive by accident: an
    unset environment variable or an empty form field.
    """

    def test_the_contract_refuses_an_empty_source_at_the_boundary(self):
        with pytest.raises(ValidationError):
            ClimateEnrichNdviInput(
                zone_id="montpellier",
                start_date="2024-01-01",
                end_date="2024-01-03",
                ndvi_source="",
            )

    def test_the_boundary_refusal_happens_before_any_provider_call(self, tmp_path, monkeypatch):
        """`check_input` raises rather than returning, which is the repo's fail-closed convention.

        What matters here is the ORDER: the contract is checked before any I/O, so the refusal
        costs nothing and reaches nothing.
        """
        called = {"yes": False}

        async def _boom(*_a, **_k):
            called["yes"] = True
            raise AssertionError("an empty ndvi_source must not reach a provider")

        monkeypatch.setattr("src.tools.climate._collect_ndvi_observations", _boom)
        tools = _load_climate_tools()

        with pytest.raises(ValidationError):
            asyncio.run(
                tools["climate_enrich_ndvi"](
                    zone_id="montpellier",
                    start_date="2024-01-01",
                    end_date="2024-01-03",
                    ndvi_source="",
                    write_output=False,
                )
            )

        assert called["yes"] is False

    def test_the_branch_does_not_depend_on_the_contract_having_refused_it(
        self, tmp_path, monkeypatch
    ):
        """Defence in depth, with the boundary check disabled.

        `if ndvi_source:` was the actual defect, and a contract constraint sitting in front of it
        hides that rather than fixing it. With `check_input` neutralised, an empty source must
        still be treated as SUPPLIED — reaching the reader, which refuses it — and never as
        omitted.
        """
        called = {"yes": False}

        async def _boom(*_a, **_k):
            called["yes"] = True
            raise AssertionError("an empty ndvi_source must not reach a provider")

        monkeypatch.setattr("src.tools.climate._collect_ndvi_observations", _boom)
        monkeypatch.setattr("src.tools.climate.fetch_archive_daily", lambda **_k: ARCHIVE_FIXTURE)
        monkeypatch.setattr("src.tools.climate.check_input", lambda *_a, **_k: None)
        era5_dst = (tmp_path / "era5.parquet").as_posix()
        tools = _load_climate_tools()
        asyncio.run(
            tools["era5_ingest"](
                zone_id="montpellier",
                start_date="2024-01-01",
                end_date="2024-01-03",
                output_path=era5_dst,
            )
        )

        result = asyncio.run(
            tools["climate_enrich_ndvi"](
                zone_id="montpellier",
                start_date="2024-01-01",
                end_date="2024-01-03",
                era5_source=era5_dst,
                ndvi_source="",
                write_output=False,
            )
        )

        assert "error" in result
        assert "ndvi_source" in result["error"]
        assert called["yes"] is False


class TestStrictWindowRefusesAnUncoverableRequest:
    """#133 — a unified table narrower than the window it was asked for is not a smaller answer.

    Off by default: returning the intersection is legitimate, and "give me what you have" is most
    of the use. The defect #133 surfaced was never that the intersection is returned, it is that it
    used to be returned silently. This flag is for the other case — a run whose output will be
    presented as covering the window it requested.
    """

    def _era5(self, tmp_path, monkeypatch) -> str:
        monkeypatch.setattr("src.tools.climate.fetch_archive_daily", lambda **_k: ARCHIVE_FIXTURE)
        dst = (tmp_path / "era5.parquet").as_posix()
        asyncio.run(
            _load_climate_tools()["era5_ingest"](
                zone_id="montpellier",
                start_date="2024-01-01",
                end_date="2024-01-03",
                output_path=dst,
            )
        )
        return dst

    def _ndvi(self, tmp_path) -> str:
        dst = (tmp_path / "ndvi.parquet").as_posix()
        _write_b3_series(dst, [("2024-01-02", "montpellier", 0.71)])
        return dst

    def _run(self, tmp_path, monkeypatch, **kwargs) -> dict:
        era5, ndvi = self._era5(tmp_path, monkeypatch), self._ndvi(tmp_path)

        async def _boom(*_a, **_k):
            raise AssertionError("no provider call is expected in these tests")

        monkeypatch.setattr("src.tools.climate._collect_ndvi_observations", _boom)
        return asyncio.run(
            _load_climate_tools()["climate_enrich_ndvi"](
                zone_id="montpellier",
                era5_source=era5,
                ndvi_source=ndvi,
                write_output=False,
                **kwargs,
            )
        )

    def test_a_window_past_the_end_of_the_record_is_refused(self, tmp_path, monkeypatch):
        """The dev-harness shape in miniature: ERA5 to 2024-01-03, a request running to July."""
        result = self._run(
            tmp_path,
            monkeypatch,
            start_date="2024-01-01",
            end_date="2024-07-31",
            strict_window=True,
        )

        assert "error" in result
        assert "strict_window" in result["error"]
        # The two windows are named, so the caller is told what to fix rather than that it failed.
        assert result["era5_window"] == "2024-01-01..2024-01-03"
        assert result["requested_window"] == "2024-01-01..2024-07-31"
        # Three of the 213 requested days exist: the narrowing is stated as a proportion, which
        # is what tells a caller whether to shorten the window or go materialize ERA5.
        assert (result["days_present_in_window"], result["days_requested"]) == (3, 213)

    def test_the_same_request_succeeds_without_the_flag(self, tmp_path, monkeypatch):
        """Default behaviour is unchanged — and the narrowing it accepts is reported, not silent.

        The NDVI series deliberately carries one observation inside the ERA5 record and one past
        its end, so the intersection is a real loss rather than a nominal one.
        """
        era5 = self._era5(tmp_path, monkeypatch)
        ndvi = (tmp_path / "ndvi_two.parquet").as_posix()
        _write_b3_series(
            ndvi, [("2024-01-02", "montpellier", 0.71), ("2024-06-15", "montpellier", 0.55)]
        )

        async def _boom(*_a, **_k):
            raise AssertionError("no provider call is expected here")

        monkeypatch.setattr("src.tools.climate._collect_ndvi_observations", _boom)
        result = asyncio.run(
            _load_climate_tools()["climate_enrich_ndvi"](
                zone_id="montpellier",
                start_date="2024-01-01",
                end_date="2024-07-31",
                era5_source=era5,
                ndvi_source=ndvi,
                write_output=False,
            )
        )

        assert "error" not in result, result
        assert result["metadata"]["window_contract"] == "intersection"
        assert result["metadata"]["era5_window"] == "2024-01-01..2024-01-03"
        assert result["metadata"]["ndvi_out_of_era5_range"] == 1

    def test_a_covered_window_passes_the_gate(self, tmp_path, monkeypatch):
        result = self._run(
            tmp_path,
            monkeypatch,
            start_date="2024-01-01",
            end_date="2024-01-03",
            strict_window=True,
        )

        assert "error" not in result, result
        assert result["metadata"]["window_contract"] == "strict"
        assert result["statistics"]["n_matched"] == 1

    def test_a_sparse_source_spanning_the_window_is_still_refused(self, tmp_path, monkeypatch):
        """Spanning is not covering, and this is why the existing helper is reused.

        A parquet holding only the first and last day has the right min and max. A hand-written
        min/max comparison would pass it, and the unified table would be built on two days of
        climate while reporting the whole window.
        """
        era5 = (tmp_path / "sparse.parquet").as_posix()
        # Written through the real writer: `_read_era5_records` does `SELECT * EXCLUDE (geometry,
        # date)`, so a hand-rolled parquet without a geometry column fails the READ and never
        # reaches the gate this test is about.
        _write_geoparquet(
            [
                {
                    "date": d,
                    "zone_id": "montpellier",
                    "lat": 43.6,
                    "lon": 3.5,
                    "wkt": "POINT(3.5 43.6)",
                    "temperature_2m": t,
                }
                for d, t in (("2024-01-01", 5.0), ("2024-01-31", 6.0))
            ],
            columns=[
                ("date", "VARCHAR"),
                ("zone_id", "VARCHAR"),
                ("lat", "DOUBLE"),
                ("lon", "DOUBLE"),
                ("wkt", "VARCHAR"),
                ("temperature_2m", "DOUBLE"),
            ],
            dst=era5,
        )

        async def _boom(*_a, **_k):
            raise AssertionError("no provider call is expected here")

        monkeypatch.setattr("src.tools.climate._collect_ndvi_observations", _boom)
        result = asyncio.run(
            _load_climate_tools()["climate_enrich_ndvi"](
                zone_id="montpellier",
                start_date="2024-01-01",
                end_date="2024-01-31",
                era5_source=era5,
                ndvi_source=self._ndvi(tmp_path),
                strict_window=True,
                write_output=False,
            )
        )

        assert "error" in result
        assert "2 of 31 days are present" in result["error"], result["error"]

    def test_the_refusal_happens_before_any_ndvi_work(self, tmp_path, monkeypatch):
        """A run that cannot produce its window should not first pay for a source read.

        Pointed at an `ndvi_source` that would itself error, the strict refusal must still be the
        answer — which is only true if the gate runs first.
        """
        era5 = self._era5(tmp_path, monkeypatch)

        async def _boom(*_a, **_k):
            raise AssertionError("collection must not be reached")

        monkeypatch.setattr("src.tools.climate._collect_ndvi_observations", _boom)
        result = asyncio.run(
            _load_climate_tools()["climate_enrich_ndvi"](
                zone_id="montpellier",
                start_date="2024-01-01",
                end_date="2024-07-31",
                era5_source=era5,
                ndvi_source=(tmp_path / "does-not-exist.parquet").as_posix(),
                strict_window=True,
                write_output=False,
            )
        )

        assert "strict_window" in result["error"], (
            f"the NDVI source was read before the window gate: {result['error']}"
        )


class TestStrictWindowIsNotSatisfiedByDaysOutsideTheWindow:
    """PR #191 review — the coverage helper counts distinct days and assumes they are CLIPPED.

    Its docstring says so. Handing it the whole artifact inverted the guarantee: a 1991-2020
    source has 10,958 distinct days, which clears the density bar for any window inside it,
    hole or no hole — so `strict_window=True` would have accepted an incomplete window and
    labelled the result `strict`, which is worse than not offering the flag.
    """

    def test_a_hole_inside_the_window_is_refused_even_when_the_source_is_much_larger(
        self, tmp_path, monkeypatch
    ):
        era5 = (tmp_path / "holed.parquet").as_posix()
        # January 2024 with the 10th missing, plus a long tail of 2025 days that exist only to
        # inflate the distinct-day count past the 31 a January request needs.
        days = [f"2024-01-{d:02d}" for d in range(1, 32) if d != 10]
        days += [f"2025-03-{d:02d}" for d in range(1, 32)]
        _write_geoparquet(
            [
                {
                    "date": d,
                    "zone_id": "montpellier",
                    "lat": 43.6,
                    "lon": 3.5,
                    "wkt": "POINT(3.5 43.6)",
                    "temperature_2m": 5.0,
                }
                for d in days
            ],
            columns=[
                ("date", "VARCHAR"),
                ("zone_id", "VARCHAR"),
                ("lat", "DOUBLE"),
                ("lon", "DOUBLE"),
                ("wkt", "VARCHAR"),
                ("temperature_2m", "DOUBLE"),
            ],
            dst=era5,
        )
        ndvi = (tmp_path / "ndvi.parquet").as_posix()
        _write_b3_series(ndvi, [("2024-01-02", "montpellier", 0.71)])

        async def _boom(*_a, **_k):
            raise AssertionError("no provider call is expected here")

        monkeypatch.setattr("src.tools.climate._collect_ndvi_observations", _boom)
        result = asyncio.run(
            _load_climate_tools()["climate_enrich_ndvi"](
                zone_id="montpellier",
                start_date="2024-01-01",
                end_date="2024-01-31",
                era5_source=era5,
                ndvi_source=ndvi,
                strict_window=True,
                write_output=False,
            )
        )

        assert "error" in result, (
            "61 distinct days in the file, 30 inside a 31-day window — the out-of-window tail "
            f"must not satisfy the density check: {result.get('metadata')}"
        )
        assert "strict_window" in result["error"]
        # The counts in the message are the IN-WINDOW ones, so it describes the gap and not the
        # artifact: 30 of the 31 days are there, and the source spanning into 2025 is context.
        assert "30 of 31 days are present" in result["error"], result["error"]
        assert result["days_present_in_window"] == 30
        assert result["days_requested"] == 31
