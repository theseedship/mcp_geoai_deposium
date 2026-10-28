"""Network-free tests for the standalone NDVI ingest tool (FIERCE B3, issue #27).

Layers:
1. **Pure helper** — ``month_ranges`` monthly slicing (full/partial/boundary), no network.
2. **ndvi_ingest** with the Sentinel-2 collection monkeypatched by synthetic observations:
   asserts the per-zone WKB GeoParquet round-trips via DuckDB (``ST_GeomFromWKB``) with the NDVI
   zonal-stats schema, the return/stats shape, and the graceful error paths (unknown zone, future
   date, invalid output_path, empty result).
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
import structlog.testing

from src.config import settings
from src.tools._ndvi import _collect_ndvi_monthly
from src.tools.climate import register_climate_tools
from src.utils.dates import month_ranges
from src.utils.duckdb_conn import get_duckdb_connection

pytestmark = pytest.mark.usefixtures("allow_local_parquet_paths")


class _CaptureMCP:
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


def _synthetic_obs(dates: list[str]) -> list[dict]:
    return [
        {
            "date": f"{d}T10:00:00+00:00",
            "ndvi_mean": 0.5 + i * 0.01,
            "ndvi_std": 0.05,
            "ndvi_min": 0.2,
            "ndvi_max": 0.8,
            "cloud_cover": 3.0,
            "item_id": f"S2_{d}",
        }
        for i, d in enumerate(dates)
    ]


# --------------------------------------------------------------------------- #
# 1. month_ranges (pure)
# --------------------------------------------------------------------------- #


class TestMonthRanges:
    def test_full_year_is_twelve_months(self):
        ranges = month_ranges("2020-01-01", "2020-12-31")
        assert len(ranges) == 12
        assert ranges[0] == ("2020-01-01", "2020-01-31")
        assert ranges[1] == ("2020-02-01", "2020-02-29")  # leap year
        assert ranges[-1] == ("2020-12-01", "2020-12-31")

    def test_partial_first_and_last_months_are_clamped(self):
        ranges = month_ranges("2020-01-15", "2020-03-10")
        assert ranges == [
            ("2020-01-15", "2020-01-31"),
            ("2020-02-01", "2020-02-29"),
            ("2020-03-01", "2020-03-10"),
        ]

    def test_single_month(self):
        assert month_ranges("2021-07-05", "2021-07-20") == [("2021-07-05", "2021-07-20")]

    def test_year_boundary(self):
        ranges = month_ranges("2020-11-01", "2021-02-28")
        assert [r[0] for r in ranges] == ["2020-11-01", "2020-12-01", "2021-01-01", "2021-02-01"]
        assert ranges[-1] == ("2021-02-01", "2021-02-28")

    def test_contiguous_non_overlapping_cover(self):
        ranges = month_ranges("2019-06-10", "2022-03-15")
        # every slice end is the day before the next slice start
        from datetime import date, timedelta

        for (_, end), (nxt_start, _) in zip(ranges, ranges[1:], strict=False):
            assert date.fromisoformat(nxt_start) == date.fromisoformat(end) + timedelta(days=1)
        assert ranges[0][0] == "2019-06-10"
        assert ranges[-1][1] == "2022-03-15"


class TestMonthlyCollection:
    def test_falls_back_when_least_cloudy_scene_is_unusable(self, monkeypatch):
        least_cloudy = SimpleNamespace(id="least-cloudy", properties={"eo:cloud_cover": 2.0})
        fallback = SimpleNamespace(id="fallback", properties={"eo:cloud_cover": 12.0})
        monkeypatch.setattr(
            "src.tools._ndvi._search_sentinel2", lambda *_a, **_k: [fallback, least_cloudy]
        )

        attempted: list[str] = []

        def _fake_compute(item, _geometry):
            attempted.append(item.id)
            if item is least_cloudy:
                return None
            observation = _synthetic_obs(["2023-07-15"])[0]
            observation["item_id"] = item.id
            return observation

        monkeypatch.setattr("src.tools._ndvi._compute_ndvi_for_item", _fake_compute)

        searched, observations = asyncio.run(
            _collect_ndvi_monthly(
                [3.0, 43.0, 4.0, 44.0],
                {"type": "Polygon", "coordinates": []},
                "2023-07-01",
                "2023-07-31",
                20,
                max_concurrency=1,
            )
        )

        assert searched == 1
        assert attempted == ["least-cloudy", "fallback"]
        assert [obs["item_id"] for obs in observations] == ["fallback"]


# --------------------------------------------------------------------------- #
# 2. ndvi_ingest (collection monkeypatched)
# --------------------------------------------------------------------------- #


class TestNdviIngest:
    async def _fake_monthly(self, *_a, **_k):
        return 3, _synthetic_obs(["2023-01-15", "2023-02-12", "2023-03-18"])

    def test_writes_per_zone_ndvi_geoparquet(self, tmp_path, monkeypatch):
        monkeypatch.setattr("src.tools.climate._collect_ndvi_monthly", self._fake_monthly)
        dst = (tmp_path / "ndvi.parquet").as_posix()
        result = asyncio.run(
            _load_climate_tools()["ndvi_ingest"](
                zone_id="montpellier",
                start_date="2023-01-01",
                end_date="2023-03-31",
                output_path=dst,
            )
        )
        assert "error" not in result, result
        assert result["n_observations"] == 3
        assert result["output"] == dst
        assert result["metadata"]["cadence"] == "monthly"
        assert result["metadata"]["streaming"].startswith("STAC/COG")
        assert result["statistics"]["first_date"] == "2023-01-15"
        assert result["statistics"]["last_date"] == "2023-03-18"

        # the parquet round-trips through DuckDB with a readable WKB geometry + NDVI columns
        con = get_duckdb_connection(":memory:")
        rows = con.execute(
            f"SELECT date, zone_id, ndvi_mean, ST_AsText(ST_GeomFromWKB(geometry)) "
            f"FROM read_parquet('{dst}') ORDER BY date"
        ).fetchall()
        con.close()
        assert len(rows) == 3
        assert str(rows[0][0]) == "2023-01-15"
        assert rows[0][1] == "montpellier"
        assert rows[0][3].startswith("POINT")

    def test_write_output_false_collects_without_writing(self, tmp_path, monkeypatch):
        monkeypatch.setattr("src.tools.climate._collect_ndvi_monthly", self._fake_monthly)
        result = asyncio.run(
            _load_climate_tools()["ndvi_ingest"](
                zone_id="montpellier",
                start_date="2023-01-01",
                end_date="2023-03-31",
                write_output=False,
            )
        )
        assert "error" not in result, result
        assert result["output"] is None
        assert result["n_observations"] == 3

    def test_non_monthly_uses_all_scenes_collector(self, tmp_path, monkeypatch):
        async def _fake_all(*_a, **_k):
            return 5, _synthetic_obs(["2023-07-03", "2023-07-08"])

        monkeypatch.setattr("src.tools.climate._collect_ndvi_observations", _fake_all)
        # guard: the monthly collector must NOT be called when monthly=False
        monkeypatch.setattr(
            "src.tools.climate._collect_ndvi_monthly",
            lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not use monthly")),
        )
        dst = (tmp_path / "ndvi.parquet").as_posix()
        result = asyncio.run(
            _load_climate_tools()["ndvi_ingest"](
                zone_id="montpellier",
                start_date="2023-07-01",
                end_date="2023-07-31",
                monthly=False,
                output_path=dst,
            )
        )
        assert "error" not in result, result
        assert result["metadata"]["cadence"] == "all-scenes"
        assert "ndvi_truncation_warning" not in result["metadata"]
        assert result["n_observations"] == 2

    def test_non_monthly_warns_when_stac_cap_is_reached(self, monkeypatch):
        async def _at_cap(*_a, **_k):
            return settings.STAC_SEARCH_MAX_ITEMS, _synthetic_obs(["2023-07-03"])

        monkeypatch.setattr("src.tools.climate._collect_ndvi_observations", _at_cap)
        result = asyncio.run(
            _load_climate_tools()["ndvi_ingest"](
                zone_id="montpellier",
                start_date="2023-07-01",
                end_date="2023-07-31",
                monthly=False,
                write_output=False,
            )
        )

        assert "error" not in result, result
        assert "may be truncated" in result["metadata"]["ndvi_truncation_warning"]

    def test_monthly_cadence_does_not_report_stac_cap_warning(self, monkeypatch):
        async def _many_months(*_a, **_k):
            return settings.STAC_SEARCH_MAX_ITEMS, _synthetic_obs(["2023-07-03"])

        monkeypatch.setattr("src.tools.climate._collect_ndvi_monthly", _many_months)
        result = asyncio.run(
            _load_climate_tools()["ndvi_ingest"](
                zone_id="montpellier",
                start_date="2019-06-01",
                end_date="2023-07-31",
                write_output=False,
            )
        )

        assert "error" not in result, result
        assert result["metadata"]["cadence"] == "monthly"
        assert "ndvi_truncation_warning" not in result["metadata"]

    def test_unknown_zone_graceful_error(self):
        result = asyncio.run(
            _load_climate_tools()["ndvi_ingest"](
                zone_id="atlantis", start_date="2023-01-01", end_date="2023-03-31"
            )
        )
        assert "error" in result and "atlantis" in result["error"]

    def test_future_date_rejected(self):
        result = asyncio.run(
            _load_climate_tools()["ndvi_ingest"](
                zone_id="montpellier", start_date="2099-01-01", end_date="2099-03-31"
            )
        )
        assert "error" in result and "future" in result["error"]

    def test_invalid_output_path_fails_before_collection(self, monkeypatch):
        # collection must not run when output_path is invalid (mirrors era5_ingest)
        async def _must_not_collect(*_a, **_k):
            raise AssertionError("must not collect when output_path is invalid")

        monkeypatch.setattr("src.tools.climate._collect_ndvi_monthly", _must_not_collect)
        result = asyncio.run(
            _load_climate_tools()["ndvi_ingest"](
                zone_id="montpellier",
                start_date="2023-01-01",
                end_date="2023-03-31",
                output_path="s3://other-bucket/secret.parquet",
            )
        )
        assert "error" in result and "Invalid output_path" in result["error"]

    def test_empty_result_is_a_graceful_error_when_writing(self, tmp_path, monkeypatch):
        async def _empty(*_a, **_k):
            return 3, []

        monkeypatch.setattr("src.tools.climate._collect_ndvi_monthly", _empty)
        result = asyncio.run(
            _load_climate_tools()["ndvi_ingest"](
                zone_id="montpellier",
                start_date="2023-01-01",
                end_date="2023-03-31",
                output_path=(tmp_path / "ndvi.parquet").as_posix(),
            )
        )
        assert "error" in result and "No NDVI observations" in result["error"]


class TestASystematicFailureIsNotReportedAsNoData:
    """#161 — the monthly collector used to discard exceptions silently.

    ``asyncio.gather(..., return_exceptions=True)`` hands failures back as objects, and the
    ``isinstance(r, dict)`` filter dropped them without a word. A STAC search failure — an
    unreachable catalogue, a misconfigured ``STAC_URL``, an expired credential — is raised inside
    ``_one_month``, so every month failed identically, the caller got ``(months, [])``, and
    ``ndvi_ingest`` answered "No NDVI observations for 'montpellier' in ...".

    This is the DEFAULT path (``monthly=True``). Nothing distinguished a broken catalogue from a
    genuinely cloud-covered window, so a B3/B4 backfill produced confidently empty series instead
    of failing — a far more expensive outcome than the credential leak filed alongside it.
    """

    @staticmethod
    def _run_monthly(monkeypatch, search):
        monkeypatch.setattr("src.tools._ndvi._search_sentinel2", search)
        return asyncio.run(
            _collect_ndvi_monthly(
                [3.0, 43.0, 4.0, 44.0],
                {"type": "Polygon", "coordinates": []},
                "2023-01-01",
                "2023-03-31",
                20,
                max_concurrency=1,
            )
        )

    def test_every_month_failing_raises_instead_of_returning_nothing(self, monkeypatch):
        def _unreachable(*_a, **_k):
            raise RuntimeError("APIError: Failed to parse: https://stac.internal/v1")

        with pytest.raises(RuntimeError, match="Failed to parse"):
            self._run_monthly(monkeypatch, _unreachable)

    def test_the_tool_reports_the_real_cause_rather_than_an_empty_window(self, monkeypatch):
        # End to end through the REAL collector — the fault is injected at the STAC leaf, not by
        # stubbing `_collect_ndvi_monthly` itself. Stubbing the collector would only prove that a
        # raise reaches ndvi_ingest's handler, which was never in doubt; what has to be pinned is
        # that the collector raises at all instead of returning an empty list.
        def _unreachable(*_a, **_k):
            raise RuntimeError("APIError: Failed to parse: https://stac.internal/v1")

        monkeypatch.setattr("src.tools._ndvi._search_sentinel2", _unreachable)
        result = asyncio.run(
            _load_climate_tools()["ndvi_ingest"](
                zone_id="montpellier",
                start_date="2023-01-01",
                end_date="2023-03-31",
                write_output=False,
            )
        )
        assert "error" in result
        assert "NDVI collection failed" in result["error"]
        assert "No NDVI observations" not in result["error"], (
            "a systematic failure is still being reported as a legitimate empty window"
        )

    def test_a_genuinely_empty_window_is_still_not_an_exception(self, monkeypatch):
        # The other half of the distinction, and the reason this is not just "raise on failure":
        # a month with no qualifying scene returns None and must stay a skipped month, not an
        # error. Collapsing the two in the other direction would be just as wrong.
        searched, observations = self._run_monthly(monkeypatch, lambda *_a, **_k: [])
        assert searched == 3
        assert observations == []

    def test_a_partial_failure_keeps_its_data_and_says_so(self, monkeypatch):
        # Partial failure IS partial data and must not be fatal — but it must not be invisible
        # either: the count alone cannot tell an operator which of the two happened, which is the
        # whole complaint against the old behaviour.
        calls = {"n": 0}

        def _fails_only_first(*_a, **_k):
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("transient upstream 503")
            return []

        with structlog.testing.capture_logs() as logs:
            searched, observations = self._run_monthly(monkeypatch, _fails_only_first)

        assert searched == 3
        assert observations == []  # the surviving months genuinely had no scenes
        warnings = [entry for entry in logs if entry["event"] == "ndvi_monthly_partial_failure"]
        assert warnings, "a partial failure went by without a word"
        assert warnings[0]["failed"] == 1
        assert "transient upstream 503" in warnings[0]["error"], (
            "the warning carries a count but not the cause, which is what made this undiagnosable"
        )

    def test_a_cancellation_is_propagated_not_filed_as_a_failed_month(self, monkeypatch):
        # `asyncio.gather(..., return_exceptions=True)` does not hand back only Exception
        # subclasses. Measured: KeyboardInterrupt and SystemExit propagate out of gather and
        # never reach the results list, but asyncio.CancelledError DOES land in it, and it is a
        # BaseException that is not an Exception.
        #
        # That is the case worth guarding — cancellation is how an abandoned MCP call tears its
        # tasks down. Filed as an ordinary per-month failure it would log a partial-failure
        # warning and carry on producing a result nobody is waiting for.
        calls = {"n": 0}

        def _cancel_first(*_a, **_k):
            calls["n"] += 1
            if calls["n"] == 1:
                raise asyncio.CancelledError("client went away")
            return []

        with pytest.raises(asyncio.CancelledError):
            self._run_monthly(monkeypatch, _cancel_first)
