"""Contract tests for the frozen pattern-library exchange schema (DoD O2 / kickoff #18, #20)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.patterns.library_schema import (
    SCHEMA_VERSION,
    GradualPatternRecord,
    PatternLibrary,
    pattern_library_json_schema,
)


def _rec(**kw):
    base = {
        "pattern_id": 0,
        "antecedent": "precipitation+",
        "consequent": "ndvi_mean+",
        "pattern": "precipitation+, ndvi_mean+",
        "variables": ["precipitation", "ndvi_mean"],
        "support": 0.82,
        "algorithm": "graank",
        "zone_id": "montpellier",
    }
    base.update(kw)
    return GradualPatternRecord(**base)


def test_minimal_record_valid_and_confidence_defaults_none():
    r = _rec()
    # ADR 0001 §5: no calibrated confidence at paliers 1-2.
    assert r.confidence is None
    assert r.lag_days == 0 and r.is_temporal is False


def test_temporal_record_carries_lag():
    r = _rec(
        is_temporal=True, lag_days=7, algorithm="tgrad", pattern="ndvi+, (precipitation+) +7.0 days"
    )
    assert r.is_temporal and r.lag_days == 7


def test_support_must_be_in_unit_interval():
    with pytest.raises(ValidationError):
        _rec(support=1.5)


def test_confidence_when_set_must_be_in_unit_interval():
    with pytest.raises(ValidationError):
        _rec(confidence=2.0)
    # a valid confidence also needs validation provenance (see next test)
    ok = _rec(confidence=0.7, n_folds=5, zones_validated=["montpellier", "marseille"])
    assert ok.confidence == 0.7


def test_confidence_requires_validation_provenance():
    # ADR 0001 §5 — ENFORCED: confidence is illegitimate before cross-validation ran.
    with pytest.raises(ValidationError):
        _rec(confidence=0.9)  # no n_folds / zones_validated
    with pytest.raises(ValidationError):
        _rec(confidence=0.9, n_folds=5)  # zones_validated still empty
    with pytest.raises(ValidationError):
        _rec(confidence=0.9, n_folds=0, zones_validated=["a"])  # 0 folds = no CV ran
    with pytest.raises(ValidationError):
        _rec(confidence=0.9, n_folds=1, zones_validated=["a"])  # < MIN_CV_FOLDS (5)
    assert _rec(confidence=0.9, n_folds=5, zones_validated=["a"]).confidence == 0.9


def test_nonzero_lag_requires_temporal():
    with pytest.raises(ValidationError):
        _rec(lag_days=7, is_temporal=False, algorithm="tgrad")
    assert _rec(lag_days=7, is_temporal=True, algorithm="tgrad").lag_days == 7


def test_temporal_records_require_temporal_algorithm():
    # graank is contemporaneous — a temporal flag or non-zero lag under it is rejected.
    with pytest.raises(ValidationError):
        _rec(is_temporal=True, lag_days=7, algorithm="graank")
    with pytest.raises(ValidationError):
        _rec(is_temporal=True, algorithm="graank")
    assert _rec(is_temporal=True, lag_days=7, algorithm="tgrad").algorithm == "tgrad"


def test_fractional_lag_preserved():
    r = _rec(is_temporal=True, lag_days=0.5, algorithm="tgrad")  # e.g. 12 hours
    assert r.lag_days == 0.5


def test_validation_period_both_or_neither():
    with pytest.raises(ValidationError):
        _rec(period_start="2020-01-01")  # period_end missing
    r = _rec(period_start="2020-01-01", period_end="2020-12-31")
    assert r.period_start is not None and r.period_end is not None


def test_extra_metrics_optional_dict():
    r = _rec(extra_metrics={"density": 0.82, "singularity_scr": 0.12})
    assert r.extra_metrics["density"] == 0.82
    assert _rec().extra_metrics is None


def test_validation_period_order_enforced():
    with pytest.raises(ValidationError):
        _rec(period_start="2020-01-01", period_end="2019-01-01")


def test_extra_fields_forbidden():
    with pytest.raises(ValidationError):
        _rec(unexpected_field=1)


def test_library_rejects_scope_mismatch():
    ok = PatternLibrary(zone_id="montpellier", algorithm="graank", patterns=[_rec()])
    assert len(ok.patterns) == 1
    with pytest.raises(ValidationError):
        PatternLibrary(zone_id="marseille", algorithm="graank", patterns=[_rec()])
    with pytest.raises(ValidationError):
        PatternLibrary(zone_id="montpellier", algorithm="aco", patterns=[_rec()])


def test_library_rejects_duplicate_pattern_ids():
    with pytest.raises(ValidationError):
        PatternLibrary(
            zone_id="montpellier",
            algorithm="graank",
            patterns=[_rec(pattern_id=0), _rec(pattern_id=0)],
        )


def test_json_schema_exports_key_fields():
    js = pattern_library_json_schema()
    assert js.get("$defs", {}).get("GradualPatternRecord")
    props = js["$defs"]["GradualPatternRecord"]["properties"]
    for f in (
        "pattern_id",
        "antecedent",
        "consequent",
        "support",
        "confidence",
        "lag_days",
        "algorithm",
        "zone_id",
        "zones_validated",
        "period_start",
        "period_end",
    ):
        assert f in props, f"exchange schema missing {f}"


def test_schema_version_pinned():
    assert PatternLibrary(zone_id="z", algorithm="graank").schema_version == SCHEMA_VERSION
