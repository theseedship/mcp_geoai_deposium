"""End-to-end seam test for the BEJOF ↔ Owuor pattern-library contract.

This is the project's **#1 integration risk** (FIERCE_ALIGNMENT "The critical
integration point"): BEJOF's alert engine (lot B6) consumes the pattern library
Owuor's mining (lot O2) produces. If the exchange format drifts, the two
deliverables silently stop fitting together — and under Art. 4.4 (global
evaluation) that sinks *both*.

This test exercises the full chain **producer → adapter → PatternLibrary →
consumer** and asserts:

1. **No silent loss** — every mined pattern with a parseable support survives
   into a schema-valid record, or is *loudly* skipped (never dropped silently).
2. **Round-trip fidelity** — the library survives a JSON serialize/deserialize
   without losing or coercing fields (the wire contract BEJOF reads).
3. **Consumer satisfiability** — a stand-in B6 consumer can score every record
   using only the exchange-schema fields (no producer-internal state needed).

`TestSeamSynthetic` runs everywhere (the adapter + schema are pure). `TestSeamReal`
drives the *real* so4gp producer and is skipped without the ``patterns``
extra — run it in the 3.14 patterns image:
    uv run --extra patterns pytest tests/test_seam_bejof_owuor.py -o addopts=""
"""

from __future__ import annotations

import pytest

from src.patterns.library_adapter import library_from_mining_result
from src.patterns.library_schema import PatternLibrary

# --- A stand-in for the future B6 `pattern_alert` consumer ------------------


def _consumer_score(rec) -> float:
    """Score a record the way the B6 alert engine eventually will — using ONLY
    exchange-schema fields (support + signed lag). Proves the consumer side of
    the contract is satisfiable from the library alone, with no producer-internal
    state. (`confidence` is None through paliers 1-2 by design, so the consumer
    must not depend on it yet.)"""
    temporal_decay = 1.0 / (1.0 + abs(rec.lag_days) / 7.0)
    return rec.support * temporal_decay


# --- Synthetic normalized mining results (shape of so4gp_adapter output) ----

GRAANK_RESULT = {
    "patterns": [
        {
            "gradual_items": [
                {"attribute": "precipitation", "direction": "+"},
                {"attribute": "ndvi", "direction": "+"},
            ],
            "support": 0.82,
            "is_temporal": False,
            "metrics": {"density": 0.82, "singularity_scr": 0.12, "avg_dev": "2.9"},
        },
        {
            "gradual_items": [
                {"attribute": "temperature", "direction": "-"},
                {"attribute": "ndvi", "direction": "+"},
            ],
            "support": 0.64,
            "is_temporal": False,
            "metrics": {},
        },
    ],
    "algorithm_key": "graank",
    "min_support": 0.5,
}

TGRAD_RESULT = {
    "patterns": [
        {
            "is_temporal": True,
            "gradual_items": [
                {
                    "attribute": "ndvi",
                    "direction": "+",
                    "role": "target",
                    "lag_days": 0.0,
                    "lag_raw": None,
                },
                {
                    "attribute": "precipitation",
                    "direction": "+",
                    "role": "lagged",
                    "lag_days": 7.0,
                    "lag_raw": "+7.0 days",
                },
            ],
            "support": 0.889,
            "metrics": {},
        },
    ],
    "algorithm_key": "tgrad",
    "min_rep": 0.5,
}


class TestSeamSynthetic:
    """The contract chain on hand-built normalized input — no so4gp needed."""

    def test_graank_round_trip(self):
        lib = library_from_mining_result(GRAANK_RESULT, zone_id="montpellier", algorithm="graank")
        assert isinstance(lib, PatternLibrary)
        assert lib.zone_id == "montpellier"
        assert lib.algorithm == "graank"
        # no silent loss: every pattern with a numeric support survived
        assert len(lib.patterns) == 2
        for rec in lib.patterns:
            # GRAANK is contemporaneous and uncalibrated at paliers 1-2
            assert rec.confidence is None
            assert rec.is_temporal is False
            assert rec.lag_days == 0.0
            assert 0.0 <= rec.support <= 1.0
            # the consumer-facing rule form is derivable
            assert rec.variables and all(v.strip() for v in rec.variables)
            assert rec.antecedent  # itemset goes in antecedent for plain GRAANK
            assert rec.consequent == ""

    def test_tgrad_round_trip(self):
        lib = library_from_mining_result(TGRAD_RESULT, zone_id="montpellier", algorithm="tgrad")
        assert len(lib.patterns) == 1
        rec = lib.patterns[0]
        assert rec.is_temporal is True
        assert rec.algorithm == "tgrad"
        # the lag survives the seam (O4 -> O2) as a signed scalar the consumer scores on
        assert rec.lag_days == 7.0
        # rule form: target -> consequent, lagged item -> antecedent (with its lag)
        assert rec.consequent == "ndvi+"
        assert "(precipitation+)" in rec.antecedent and "+7.0 days" in rec.antecedent

    def test_json_serialization_is_faithful(self):
        lib = library_from_mining_result(TGRAD_RESULT, zone_id="montpellier", algorithm="tgrad")
        round_tripped = PatternLibrary.model_validate_json(lib.model_dump_json())
        # the wire contract BEJOF reads must reproduce the library byte-for-meaning
        assert round_tripped.model_dump() == lib.model_dump()
        # spot-check the fields the consumer scores on are not coerced (float stays float)
        assert isinstance(round_tripped.patterns[0].lag_days, float)
        assert round_tripped.patterns[0].support == 0.889

    def test_no_silent_loss_malformed_pattern(self):
        # a malformed pattern (support=None) sits BETWEEN two valid ones
        result = {
            "patterns": [
                GRAANK_RESULT["patterns"][0],
                {
                    "gradual_items": [{"attribute": "x", "direction": "+"}],
                    "support": None,
                    "is_temporal": False,
                },
                GRAANK_RESULT["patterns"][1],
            ],
        }
        # non-strict: the bad one is skipped, the good ones survive, ids keep the source index
        lib = library_from_mining_result(
            result, zone_id="montpellier", algorithm="graank", strict=False
        )
        assert len(lib.patterns) == 2
        assert [rec.pattern_id for rec in lib.patterns] == [0, 2]  # index 1 dropped, not renumbered
        # strict: a single unmappable pattern aborts loudly rather than shipping a partial library
        with pytest.raises(ValueError):
            library_from_mining_result(
                result, zone_id="montpellier", algorithm="graank", strict=True
            )

    def test_consumer_can_score_every_record(self):
        graank = library_from_mining_result(
            GRAANK_RESULT, zone_id="montpellier", algorithm="graank"
        )
        tgrad = library_from_mining_result(TGRAD_RESULT, zone_id="montpellier", algorithm="tgrad")
        for lib in (graank, tgrad):
            for rec in lib.patterns:
                score = _consumer_score(rec)
                assert 0.0 <= score <= 1.0  # finite, bounded, computable from schema alone
        # the temporal pattern is decayed by its lag; the contemporaneous one is not
        assert _consumer_score(tgrad.patterns[0]) < tgrad.patterns[0].support


# --- The same chain, driven by the REAL so4gp producer ----------------------
# NOTE: a module-level importorskip would skip TestSeamSynthetic too, so detect
# availability instead and gate only the real-producer class on it.

try:
    import pandas as pd
    import so4gp  # noqa: F401

    from src.patterns.so4gp_adapter import mine_gradual_patterns, mine_temporal_patterns

    _HAS_PATTERNS = True
except ImportError:
    _HAS_PATTERNS = False


@pytest.fixture
def climate_df():
    return pd.DataFrame(
        {
            "date_idx": [0, 1, 2, 3, 4, 5, 6, 7],
            "precip": [45, 52, 38, 25, 18, 42, 55, 48],
            "temp": [8, 9, 11, 14, 17, 15, 13, 12],
            "ndvi": [0.72, 0.68, 0.65, 0.58, 0.52, 0.61, 0.69, 0.74],
        }
    )


@pytest.fixture
def temporal_df():
    return pd.DataFrame(
        [
            ["2020-01-01", 10, 0.30, 25],
            ["2020-01-08", 20, 0.31, 24],
            ["2020-01-15", 35, 0.40, 22],
            ["2020-01-22", 50, 0.55, 20],
            ["2020-01-29", 40, 0.62, 21],
            ["2020-02-05", 60, 0.58, 19],
            ["2020-02-12", 80, 0.70, 17],
            ["2020-02-19", 70, 0.85, 18],
            ["2020-02-26", 90, 0.80, 16],
            ["2020-03-04", 100, 0.92, 15],
        ],
        columns=["date", "precipitation", "ndvi", "temperature"],
    )


@pytest.mark.skipif(not _HAS_PATTERNS, reason="patterns extra not installed (needs Python >=3.14)")
class TestSeamReal:
    """Drive the seam with real so4gp output (skipped without the extra)."""

    def test_graank_seam_real(self, climate_df, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)  # so4gp writes GRAANK_* to cwd — contain it
        result = mine_gradual_patterns(climate_df, min_support=0.5, algorithm="graank")
        lib = library_from_mining_result(
            result, zone_id="montpellier", algorithm="graank", strict=False
        )
        # no silent loss: library count == #patterns that carried a parseable support
        mappable = sum(1 for p in result["patterns"] if p.get("support") is not None)
        assert len(lib.patterns) == mappable >= 1
        for rec in lib.patterns:
            assert rec.confidence is None  # ADR 0001 §5 — uncalibrated at paliers 1-2
            assert rec.is_temporal is False and rec.lag_days == 0.0
            assert 0.0 <= _consumer_score(rec) <= 1.0
        # wire round-trip stays faithful on real output
        assert (
            PatternLibrary.model_validate_json(lib.model_dump_json()).model_dump()
            == lib.model_dump()
        )

    def test_tgrad_seam_real(self, temporal_df, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        # ndvi is the target (column index 2); precipitation/temperature are the lagged attrs
        result = mine_temporal_patterns(
            temporal_df, target_col=2, min_support=0.4, min_rep=0.5, algorithm="tgrad"
        )
        lib = library_from_mining_result(
            result, zone_id="montpellier", algorithm="tgrad", strict=False
        )
        mappable = sum(1 for p in result["patterns"] if p.get("support") is not None)
        assert len(lib.patterns) == mappable >= 1
        for rec in lib.patterns:
            assert rec.is_temporal is True and rec.algorithm == "tgrad"
            assert rec.confidence is None
        # the temporal lag (O4) survives into the exchange library (O2) on at least one pattern
        assert any(rec.lag_days != 0.0 for rec in lib.patterns)
        assert (
            PatternLibrary.model_validate_json(lib.model_dump_json()).model_dump()
            == lib.model_dump()
        )
