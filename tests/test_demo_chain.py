"""Network-free tests for the FIERCE Montpellier demonstrator (scripts/demo_montpellier.py).

Two layers, matching the demonstrator design (docs/design/montpellier-demonstrator.md §5):

1. **Reporting + serialization (pure, base runtime).** The demo's `build_report` / `write_artifacts`
   / `_patterns_table` helpers + `library_from_mining_result` are pure — no so4gp / network / S3 — so
   they run everywhere. Asserts the report carries the design §2 provenance and that the emitted
   library is schema-valid JSON.
2. **End-to-end mining seam (gated on the `patterns` extra).** Drives the **real**
   `discover_patterns(tgrad)` tool with the S3 source resolver (`load_dataframe`) **monkeypatched**
   to a synthetic unified DataFrame — so the chain's stage-3→4 wiring (discover_patterns →
   PatternLibrary) is exercised with NO network and NO S3, but through the actual tool (no bypass,
   design §4.2(a)). Skipped without so4gp; run it in the 3.14 patterns image:
       uv run --extra patterns pytest tests/test_demo_chain.py -o addopts=""
"""

from __future__ import annotations

import asyncio
import importlib.util
from pathlib import Path

import pytest

from src.patterns.library_adapter import library_from_mining_result
from src.patterns.library_schema import PatternLibrary

# Load the standalone demo script by path (scripts/ is not an installed package).
_DEMO_PATH = Path(__file__).resolve().parents[1] / "scripts" / "demo_montpellier.py"
_spec = importlib.util.spec_from_file_location("demo_montpellier", _DEMO_PATH)
demo = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(demo)


# A normalized tgrad mining result (the shape so4gp_adapter / discover_patterns emit) — pure input.
_TGRAD_RESULT = {
    "patterns": [
        {
            "is_temporal": True,
            "gradual_items": [
                {"attribute": "ndvi_mean", "direction": "+", "role": "target", "lag_days": 0.0},
                {
                    "attribute": "precipitation",
                    "direction": "+",
                    "role": "lagged",
                    "lag_days": 7.0,
                    "lag_raw": "+7.0 days",
                },
            ],
            "support": 0.889,
            "metrics": {"density": 0.9},
        },
    ],
    "algorithm_key": "tgrad",
}


def _ingest_stub() -> dict:
    return {
        "zone_id": demo.ZONE_ID,
        "start_date": demo.START_DATE,
        "end_date": demo.END_DATE,
        "n_days": 365,
        "completeness": {"pct_days": 100.0},
    }


def _enrich_stub(*, truncated: bool = False) -> dict:
    meta = {}
    if truncated:
        meta["ndvi_truncation_warning"] = "NDVI search hit the 50-item STAC cap"
    return {
        "output": f"s3://test-bucket/climate/unified/{demo.ZONE_ID}.parquet",
        "statistics": {"n_ndvi_observations": 40, "n_matched": 31},
        "metadata": meta,
    }


def _mining_stub(n_patterns: int = 1) -> dict:
    return {"statistics": {"n_patterns": n_patterns}}


# --------------------------------------------------------------------------- #
# 1. Reporting + serialization — pure, base runtime
# --------------------------------------------------------------------------- #


class TestDemoReport:
    def test_library_is_schema_valid_json(self, tmp_path):
        lib = library_from_mining_result(_TGRAD_RESULT, zone_id=demo.ZONE_ID, algorithm="tgrad")
        report = demo.build_report(
            library=lib,
            ingest=_ingest_stub(),
            enrich=_enrich_stub(),
            mining=_mining_stub(),
            used_min_support=0.5,
            used_min_rep=0.5,
            generated_at="2026-06-17 12:00 UTC",
        )
        json_path, report_path = demo.write_artifacts(lib, report, tmp_path / "demo")

        # the JSON artifact round-trips through the frozen exchange schema
        reloaded = PatternLibrary.model_validate_json(json_path.read_text())
        assert reloaded.zone_id == demo.ZONE_ID and reloaded.algorithm == "tgrad"
        assert len(reloaded.patterns) == 1
        assert reloaded.patterns[0].confidence is None  # discovery-grade, ADR 0001 §5
        assert report_path.read_text() == report

    def test_report_carries_design_provenance(self):
        lib = library_from_mining_result(_TGRAD_RESULT, zone_id=demo.ZONE_ID, algorithm="tgrad")
        report = demo.build_report(
            library=lib,
            ingest=_ingest_stub(),
            enrich=_enrich_stub(),
            mining=_mining_stub(),
            used_min_support=0.5,
            used_min_rep=0.5,
            generated_at="2026-06-17 12:00 UTC",
        )
        # design §2 provenance: n_days, n_matched, support, signed lag, and the honesty caveats
        assert demo.ZONE_ID in report
        assert "365" in report  # n_days
        assert "31 unified rows" in report  # n_matched
        assert "0.889" in report  # per-pattern support
        assert "+7.0" in report  # signed lag_days
        assert "confidence=None" in report  # no calibrated confidence
        assert "One year" in report  # pipeline-proof caveat
        assert "pipeline knob" in report  # thresholds-are-a-non-claim caveat

    def test_truncation_warning_surfaced_when_present(self):
        lib = library_from_mining_result(_TGRAD_RESULT, zone_id=demo.ZONE_ID, algorithm="tgrad")
        report = demo.build_report(
            library=lib,
            ingest=_ingest_stub(),
            enrich=_enrich_stub(truncated=True),
            mining=_mining_stub(),
            used_min_support=0.5,
            used_min_rep=0.5,
            generated_at="2026-06-17 12:00 UTC",
        )
        assert "NDVI truncation" in report and "STAC cap" in report

    def test_empty_library_reports_zero_patterns_honestly(self):
        empty = library_from_mining_result(
            {"patterns": []}, zone_id=demo.ZONE_ID, algorithm="tgrad"
        )
        report = demo.build_report(
            library=empty,
            ingest=_ingest_stub(),
            enrich=_enrich_stub(),
            mining=_mining_stub(n_patterns=0),
            used_min_support=0.2,
            used_min_rep=0.2,
            generated_at="2026-06-17 12:00 UTC",
        )
        assert "No schema-mappable patterns" in report

    def test_skip_path_completeness_not_fabricated(self):
        # era5_ingest's idempotent-skip return (overwrite=False, parquet already covers the range)
        # carries NO `completeness` block — only stored coverage. build_report must NOT print a
        # fabricated "0%/None% of expected days" on such a cached re-run (the P2 review finding).
        lib = library_from_mining_result(_TGRAD_RESULT, zone_id=demo.ZONE_ID, algorithm="tgrad")
        skipped_ingest = {
            "zone_id": demo.ZONE_ID,
            "start_date": demo.START_DATE,
            "end_date": demo.END_DATE,
            "n_days": 365,  # the real stored COUNT(*), echoed by the skip path
            "metadata": {"skipped": "output exists and covers the request (overwrite=False)"},
        }
        report = demo.build_report(
            library=lib,
            ingest=skipped_ingest,
            enrich=_enrich_stub(),
            mining=_mining_stub(),
            used_min_support=0.5,
            used_min_rep=0.5,
            generated_at="2026-06-17 12:00 UTC",
        )
        assert "365 daily records" in report
        assert "0% of expected days" not in report  # no fabricated completeness
        assert "None%" not in report
        assert "cached" in report  # the honest skip phrasing


# --------------------------------------------------------------------------- #
# 2. End-to-end mining seam — real discover_patterns, monkeypatched resolver
# --------------------------------------------------------------------------- #

try:
    import pandas as pd
    import so4gp  # noqa: F401

    _HAS_PATTERNS = True
except ImportError:
    _HAS_PATTERNS = False


def _synthetic_unified_df():
    """A faithful unified-table DataFrame (B4 §5.3 schema): the WKB-blob geometry + identifiers that
    `drop_columns` removes, plus the monotonic signal columns tgrad mines. Strictly co-monotonic so
    the seam reliably emits ≥1 temporal pattern (ndvi_mean ↑ with precipitation ↑, temperature ↓)."""
    n = 14
    dates = [f"2023-{1 + i // 4:02d}-{1 + (i % 4) * 7:02d}" for i in range(n)]
    return pd.DataFrame(
        {
            "date": dates,
            "zone_id": [demo.ZONE_ID] * n,
            "lat": [43.6] * n,
            "lon": [3.86] * n,
            "geometry": [b"\x01\x01\x00\x00\x00"] * n,  # WKB blob stand-in (dropped before mining)
            "ndvi_date": dates,
            "day_delta": [0] * n,
            "ndvi_std": [0.01 + 0.001 * i for i in range(n)],
            "ndvi_mean": [0.30 + 0.03 * i for i in range(n)],  # target, rising
            "precipitation": [10.0 + 5.0 * i for i in range(n)],  # rising → concordant with ndvi
            "temperature_2m": [25.0 - 0.5 * i for i in range(n)],  # falling → discordant
        }
    )


@pytest.mark.skipif(not _HAS_PATTERNS, reason="patterns extra not installed (needs Python >=3.14)")
class TestDemoChainSeam:
    """Stage 3→4 of the demo (discover_patterns(tgrad) → PatternLibrary) with NO network/S3: the
    S3 source resolver is monkeypatched, but the call goes through the REAL tool (no bypass)."""

    def _setup(self, monkeypatch):
        """Monkeypatch the S3 resolver + bucket; return (real discover_patterns tool, unified URI).

        discover_patterns hard-fails on an empty S3_BUCKET (it is bucket-restricted by design), so
        give it a dummy bucket and monkeypatch the RustFS→DataFrame resolver (load_dataframe) — the
        call then goes through the REAL tool with no network/S3 (design §4.2(a), no bypass)."""
        from src.config import settings

        df = _synthetic_unified_df()

        def _fake_load_dataframe(
            source, *, kind="parquet", columns=None, order_by=None, drop_columns=None, **_kw
        ):
            # faithful stand-in for src.patterns.parquet_bridge.load_dataframe: honor the explicit
            # columns projection, order_by, and drop_columns exactly as the real bridge does, but
            # read no S3. (The demo passes columns=None today; mirrored anyway to avoid drift.)
            out = df.copy()
            if columns:
                out = out[[c for c in columns if c in out.columns]]
            if order_by is not None:
                out = out.sort_values(order_by).reset_index(drop=True)
            if drop_columns:
                out = out.drop(columns=[c for c in drop_columns if c in out.columns])
            return out

        monkeypatch.setattr("src.config.settings.S3_BUCKET", "test-bucket")
        assert settings.S3_BUCKET == "test-bucket"
        monkeypatch.setattr("src.tools.patterns.load_dataframe", _fake_load_dataframe)

        discover_patterns = demo._load_tools()["discover_patterns"]
        source = f"s3://test-bucket/climate/unified/{demo.ZONE_ID}.parquet"
        return discover_patterns, source

    def _mine(self, monkeypatch):
        """Drive the demo's REAL stage-3 helper (mine_with_stepping) — exercising the live
        threshold-stepping loop + the exact discover_patterns call-site kwargs, network/S3-free."""
        discover_patterns, source = self._setup(monkeypatch)
        return asyncio.run(demo.mine_with_stepping(discover_patterns, source=source))

    def test_chain_emits_schema_valid_library(self, monkeypatch, tmp_path):
        result, used_min_support, used_min_rep = self._mine(monkeypatch)
        assert "error" not in result, result
        assert result["statistics"]["n_patterns"] >= 1
        # the stepping loop stopped at one of the documented tiers (strongly-monotonic data → tier 0)
        assert (used_min_support, used_min_rep) in demo.THRESHOLD_TIERS
        # the result is marked temporal (tgrad metadata), not contemporaneous
        assert result["metadata"]["target_column"] == demo.TARGET_COLUMN
        assert result["metadata"]["time_column"] == demo.ORDER_BY

        # stage 4: the discover_patterns RESULT maps straight into a schema-valid PatternLibrary
        lib = library_from_mining_result(result, zone_id=demo.ZONE_ID, algorithm="tgrad")
        assert isinstance(lib, PatternLibrary)
        assert len(lib.patterns) >= 1
        for rec in lib.patterns:
            assert rec.algorithm == "tgrad"
            assert rec.zone_id == demo.ZONE_ID
            assert rec.is_temporal is True
            assert isinstance(rec.lag_days, float)  # lag_days present for tgrad (design §5)
            assert rec.confidence is None  # uncalibrated at paliers 1-2 (ADR 0001 §5)
            assert 0.0 <= rec.support <= 1.0

        # the library serializes to schema-valid JSON the consumer (BEJOF) can read back
        assert (
            PatternLibrary.model_validate_json(lib.model_dump_json()).model_dump()
            == lib.model_dump()
        )

        # and the demo's report renders from the real result without error
        report = demo.build_report(
            library=lib,
            ingest=_ingest_stub(),
            enrich=_enrich_stub(),
            mining=result,
            used_min_support=used_min_support,
            used_min_rep=used_min_rep,
            generated_at="2026-06-17 12:00 UTC",
        )
        assert demo.ZONE_ID in report and "tgrad" in report

    def test_drop_columns_keeps_only_signal_columns(self, monkeypatch):
        # the geometry/identifier columns must NOT appear among the mined columns (design §4.3)
        result, _used_ms, _used_mr = self._mine(monkeypatch)
        assert "error" not in result, result  # a tool-level error has no metadata to index
        mined = set(result["metadata"]["columns_mined"])
        assert "ndvi_mean" in mined
        assert mined.isdisjoint({"zone_id", "lat", "lon", "geometry", "ndvi_date", "day_delta"})
