"""B6 level 2: pure explanatory composition over a verified library and a saved forecast (#30)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from src import b6_explanatory as b6
from src.contracts import B6_EXCLUSION_REASONS, B6_FORMULA_ID, B6Params
from src.patterns.library_schema import PatternLibrary
from src.tools import pattern_alert


def _rows(variable: str, values: list[float]) -> list[dict]:
    return [{"date": f"2026-10-{i + 1:02d}", variable: v} for i, v in enumerate(values)]


def test_pattern_alert_reuses_the_level2_helpers():
    # One implementation of the arithmetic: level 1 alerts and level 2 cannot drift apart.
    assert pattern_alert._series_delta is b6.series_delta
    assert pattern_alert._forecast_alignment is b6.forecast_alignment
    assert pattern_alert._lag_decay is b6.lag_decay


def test_helpers_keep_their_behaviour():
    assert b6.lag_decay(-10.0, 10.0) == pytest.approx(0.5)
    rising = _rows("evapotranspiration", [1.0, 2.0])
    assert b6.series_delta(rising, "evapotranspiration")["direction"] == "+"
    assert b6.series_delta(rising[:1], "evapotranspiration") == {
        "status": "insufficient_forecast_values",
        "n": 1,
    }
    agreement, evidence = b6.forecast_alignment(
        antecedent="(evapotranspiration+) +7.0 days", forecast_rows=rising
    )
    assert agreement == 1.0 and evidence[0]["matched"] is True
    assert b6.forecast_alignment(antecedent="ndvi_mean+", forecast_rows=rising) == (0.0, [])


# --------------------------------------------------------------------------- #
# score_level2 on a synthetic library
# --------------------------------------------------------------------------- #

COMPUTED_AT = "2026-10-16T08:00:00+00:00"
LIBRARY_IDENTITY = {
    "library_version": "2026.10.16-fixture.1",
    "manifest_sha256": "a" * 64,
    "json_sha256": "b" * 64,
    "parquet_sha256": "c" * 64,
    "schema_version": "1.0.0",
    "zone_id": "lez",
    "algorithm": "tgrad",
    "area_status": "unqualified",
    "qualification": "unqualified",
}
FORECAST_IDENTITY = {"sha256": "d" * 64, "source": "fixture", "issued_at": "2026-10-16T00:00:00"}
RISING_ET = _rows("evapotranspiration", [float(i) for i in range(10)])


def _record(pattern_id: int = 1, **overrides) -> dict:
    record = {
        "pattern_id": pattern_id,
        "antecedent": "(evapotranspiration+) +7.0 days",
        "consequent": "ndvi_mean+",
        "pattern": "ndvi_mean+, (evapotranspiration+) +7.0 days",
        "variables": ["ndvi_mean", "evapotranspiration"],
        "support": 0.8,
        "is_temporal": True,
        "lag_days": 7.0,
        "algorithm": "tgrad",
        "zone_id": "lez",
        "zones_validated": ["lez", "fos"],
        "period_start": "2021-01-01",
        "period_end": "2024-12-31",
        "n_folds": 5,
        "stability": 0.9,
    }
    record.update(overrides)
    return record


def _library(*records: dict) -> PatternLibrary:
    return PatternLibrary.model_validate(
        {"zone_id": "lez", "algorithm": "tgrad", "patterns": list(records)}
    )


def _score(library, *, forecast_rows=RISING_ET, horizon=10, **params):
    return b6.score_level2(
        library,
        library_identity=LIBRARY_IDENTITY,
        forecast_rows=forecast_rows,
        forecast_identity=FORECAST_IDENTITY,
        zone_id="lez",
        effective_horizon_days=horizon,
        params=B6Params(**params),
        code_ref={"repo": "deposium_geoai", "commit": "f" * 40, "dirty": False},
        computed_at=COMPUTED_AT,
    )


def test_cross_validated_pattern_with_matching_forecast_is_explained():
    result = _score(_library(_record()))
    assert result.status == "explained" and result.reason is None
    assert result.confidence is None and result.event_probability is None
    [pattern] = result.patterns
    expected = 0.8 * 0.9 * 1.0 * 0.5 ** (7.0 / 10.0)
    assert pattern.explanatory_index == pytest.approx(expected)
    assert [(f.name, f.value) for f in pattern.factors] == [
        ("support", 0.8),
        ("stability", 0.9),
        ("forecast_agreement", 1.0),
        ("lag_decay", pytest.approx(0.5**0.7)),
    ]
    assert pattern.rule_id == "lez/tgrad/1"
    assert pattern.forecast_evidence[0]["matched"] is True
    assert "not a probability" in pattern.explanation
    assert result.exclusions == dict.fromkeys(B6_EXCLUSION_REASONS, 0)
    assert (result.n_patterns_read, result.n_eligible) == (1, 1)


def test_level2_scores_the_supported_minimum_temperature_driver():
    record = _record(
        antecedent="temperature_2m_min+",
        pattern="temperature_2m_min+, ndvi_mean+",
        variables=["temperature_2m_min", "ndvi_mean"],
    )
    result = _score(_library(record), forecast_rows=_rows("temperature_2m_min", [1.0, 2.0]))
    assert result.status == "explained"
    [pattern] = result.patterns
    assert pattern.explanatory_index == pytest.approx(0.8 * 0.9 * 0.5**0.7)
    assert pattern.forecast_evidence[0]["variable"] == "temperature_2m_min"
    assert pattern.forecast_evidence[0]["matched"] is True
    assert result.exclusions == dict.fromkeys(B6_EXCLUSION_REASONS, 0)


@pytest.mark.parametrize(
    "forecast_rows",
    [
        _rows("temperature_2m_min", [1.0]),
        _rows("temperature_2m_min", [None, 2.0]),
        _rows("temperature_2m", [1.0, 2.0]),
    ],
)
def test_minimum_temperature_without_two_values_is_missing_data(forecast_rows):
    record = _record(antecedent="temperature_2m_min+")
    result = _score(_library(record), forecast_rows=forecast_rows)
    assert result.status == "no_conclusion"
    assert result.reason == "missing_forecast_data"
    assert result.exclusions["insufficient_forecast_values"] == 1
    assert result.exclusions["no_forecast_variable"] == 0


def test_level2_counts_minimum_temperature_in_mixed_forecast_agreement():
    record = _record(antecedent="temperature_2m_min+, evapotranspiration-, ndvi_mean+")
    forecast_rows = [
        {"date": "2026-10-01", "temperature_2m_min": 1.0, "evapotranspiration": 1.0},
        {"date": "2026-10-02", "temperature_2m_min": 2.0, "evapotranspiration": 2.0},
    ]
    [pattern] = _score(_library(record), forecast_rows=forecast_rows).patterns
    assert pattern.factors[2].value == 0.5
    assert pattern.factors[2].derivation.startswith("1/2")
    assert pattern.factors[2].derivation.endswith(
        "partial driver agreement, not verification of the full antecedent; not counted: ndvi_mean"
    )
    assert {item["variable"] for item in pattern.forecast_evidence} == {
        "temperature_2m_min",
        "evapotranspiration",
    }


def test_legacy_pattern_alert_keeps_its_original_forecast_variable_set():
    forecast_rows = [
        {"date": "2026-10-01", "temperature_2m_min": 1.0, "evapotranspiration": 1.0},
        {"date": "2026-10-02", "temperature_2m_min": 2.0, "evapotranspiration": 2.0},
    ]
    assert pattern_alert._forecast_alignment(
        antecedent="temperature_2m_min+", forecast_rows=forecast_rows
    ) == (0.0, [])
    agreement, evidence = pattern_alert._forecast_alignment(
        antecedent="temperature_2m_min+, evapotranspiration-", forecast_rows=forecast_rows
    )
    assert agreement == 0.0
    assert [item["variable"] for item in evidence] == ["evapotranspiration"]


@pytest.mark.parametrize(
    "overrides, forecast_rows, excluded, reason",
    [
        ({"n_folds": 4}, RISING_ET, "not_cross_validated", "no_admissible_pattern"),
        ({"n_folds": None}, RISING_ET, "not_cross_validated", "no_admissible_pattern"),
        ({"zones_validated": []}, RISING_ET, "not_cross_validated", "no_admissible_pattern"),
        ({"zones_validated": ["fos"]}, RISING_ET, "zone_not_validated", "no_admissible_pattern"),
        ({"support": 0.4}, RISING_ET, "low_support", "no_admissible_pattern"),
        ({"stability": None}, RISING_ET, "low_stability", "no_admissible_pattern"),
        ({"stability": 0.4}, RISING_ET, "low_stability", "no_admissible_pattern"),
        ({"lag_days": -30.0}, RISING_ET, "lag_outside_horizon", "all_lags_outside_horizon"),
        (
            {"antecedent": "ndvi_mean+", "pattern": "ndvi_mean+"},
            RISING_ET,
            "no_forecast_variable",
            "missing_forecast_data",
        ),
        (
            {},
            _rows("precipitation", [1.0, 2.0]),
            "insufficient_forecast_values",
            "missing_forecast_data",
        ),
        ({}, [], "insufficient_forecast_values", "missing_forecast_data"),
    ],
)
def test_each_exclusion_is_typed_and_gives_a_typed_non_conclusion(
    overrides, forecast_rows, excluded, reason
):
    result = _score(_library(_record(**overrides)), forecast_rows=forecast_rows)
    assert result.status == "no_conclusion" and result.reason == reason
    assert result.patterns == [] and result.n_eligible == 0
    assert result.exclusions == {**dict.fromkeys(B6_EXCLUSION_REASONS, 0), excluded: 1}
    assert result.trace.patterns[0].excluded == excluded
    assert result.confidence is None and result.event_probability is None


def test_empty_library_is_a_typed_non_conclusion():
    result = _score(_library())
    assert (result.status, result.reason, result.n_patterns_read) == (
        "no_conclusion",
        "empty_library",
        0,
    )
    assert result.exclusions == dict.fromkeys(B6_EXCLUSION_REASONS, 0)


def test_gates_are_parameters_not_constants():
    # #30 point 2: the n_folds gate is a recorded choice; a different value changes the result.
    library = _library(_record(n_folds=2))
    assert _score(library).status == "no_conclusion"
    relaxed = _score(library, min_folds=2)
    assert relaxed.status == "explained"
    assert relaxed.trace.params.min_folds == 2


def test_disagreeing_forecast_stays_explained_with_a_zero_index():
    falling = _rows("evapotranspiration", [9.0, 1.0])
    [pattern] = _score(_library(_record()), forecast_rows=falling).patterns
    assert pattern.explanatory_index == 0.0
    assert pattern.factors[2].derivation.startswith("0/1")


def test_ranking_truncation_and_counts():
    library = _library(
        _record(1, support=0.6),
        _record(2, support=0.9),
        _record(3, support=0.9, lag_days=1.0),
        _record(4, n_folds=1),
        _record(5, lag_days=40.0),
    )
    result = _score(library, max_patterns=2)
    assert [p.pattern_id for p in result.patterns] == [3, 2]
    assert result.n_eligible == 3 and result.n_patterns_read == 5
    assert result.exclusions["not_cross_validated"] == 1
    assert result.exclusions["lag_outside_horizon"] == 1


def test_trace_records_formula_inputs_and_per_pattern_provenance():
    result = _score(_library(_record(), _record(2, n_folds=None, zones_validated=[])))
    trace = result.trace
    assert trace.formula_id == B6_FORMULA_ID and trace.formula_version == "1.0.0"
    assert trace.params == B6Params() and trace.params_status == "proposed_unratified"
    assert trace.library.model_dump() == LIBRARY_IDENTITY
    assert trace.forecast == FORECAST_IDENTITY
    assert trace.effective_horizon_days == 10
    assert trace.code_ref["commit"] == "f" * 40
    assert trace.computed_at == COMPUTED_AT
    first, second = trace.patterns
    assert (first.n_folds, first.zones_validated, first.excluded) == (5, ["lez", "fos"], None)
    assert (str(first.period_start), str(first.period_end)) == ("2021-01-01", "2024-12-31")
    assert (second.n_folds, second.excluded) == (None, "not_cross_validated")


def test_flat_forecast_stays_explained_with_a_zero_index():
    # #30 agreed point 6: measured but flat evidence is a zero index, not a non-conclusion.
    flat = _rows("evapotranspiration", [3.0, 2.0, 3.0])
    result = _score(_library(_record()), forecast_rows=flat)
    assert (result.status, result.reason) == ("explained", None)
    [pattern] = result.patterns
    assert pattern.explanatory_index == 0.0
    assert pattern.forecast_evidence[0]["direction"] == "0"
    assert pattern.explanation.startswith("0/1 forecast drivers match")
    assert "partial driver agreement" not in pattern.factors[2].derivation


def test_multi_driver_agreement_is_partial_and_names_uncounted_items():
    rows = [
        {"date": "2026-10-01", "evapotranspiration": 1.0, "temperature_2m": 20.0},
        {"date": "2026-10-02", "evapotranspiration": 2.0, "temperature_2m": 21.0},
    ]
    antecedent = "(evapotranspiration+), (temperature_2m-), (ndvi_mean+) +7.0 days"
    record = _record(antecedent=antecedent, pattern=antecedent)
    [pattern] = _score(_library(record), forecast_rows=rows).patterns
    agreement = pattern.factors[2]
    assert agreement.value == 0.5
    assert agreement.derivation.startswith("1/2")
    assert agreement.derivation.endswith("not counted: ndvi_mean")
    assert "partial driver agreement" in agreement.derivation


@pytest.mark.parametrize(
    "second, reason",
    [
        ({"lag_days": 40.0}, "all_lags_outside_horizon"),
        ({"antecedent": "ndvi_mean+", "pattern": "ndvi_mean+"}, "missing_forecast_data"),
    ],
)
def test_mixed_exclusions_pick_the_reason_from_admissible_patterns(second, reason):
    library = _library(_record(1, lag_days=30.0), _record(2, n_folds=1), _record(3, **second))
    result = _score(library)
    assert (result.status, result.reason) == ("no_conclusion", reason)
    assert result.exclusions["not_cross_validated"] == 1


def test_identity_of_another_library_is_refused():
    with pytest.raises(ValueError, match="scored library"):
        b6.score_level2(
            _library(_record()),
            library_identity={**LIBRARY_IDENTITY, "zone_id": "fos"},
            forecast_rows=RISING_ET,
            forecast_identity=FORECAST_IDENTITY,
            zone_id="lez",
            effective_horizon_days=10,
            code_ref=None,
            computed_at=COMPUTED_AT,
        )


def test_same_inputs_give_the_same_result():
    library = _library(_record(), _record(2, support=0.7))
    assert _score(library).model_dump(mode="json") == _score(library).model_dump(mode="json")


# --------------------------------------------------------------------------- #
# load_verified_library on the delivered #302 package
# --------------------------------------------------------------------------- #

PACKAGE = Path(__file__).resolve().parents[1] / "src/data/pattern-library-2026-10-15/package"
MANIFEST_SHA = "bfed1a04a176e9fba288fa822b3a02a3f280591c8fd9c4942bb245854cfa47e2"


def test_real_package_verifies_and_reports_its_identity():
    library, identity = b6.load_verified_library(PACKAGE, MANIFEST_SHA, "bassin_du_lez", "tgrad")
    manifest = json.loads((PACKAGE / "manifest.json").read_bytes())
    assert (library.zone_id, library.algorithm, len(library.patterns)) == (
        "bassin_du_lez",
        "tgrad",
        135,
    )
    assert identity.model_dump() == {
        "library_version": "2026.10.15-native-pilots.1",
        "manifest_sha256": MANIFEST_SHA,
        "json_sha256": manifest["files"]["libraries/bassin_du_lez/tgrad.json"]["sha256"],
        "parquet_sha256": manifest["files"]["libraries/bassin_du_lez/tgrad.parquet"]["sha256"],
        "schema_version": "1.0.0",
        "zone_id": "bassin_du_lez",
        "algorithm": "tgrad",
        "area_status": "unqualified",
        "qualification": "unqualified",
    }
    assert (
        identity.json_sha256
        == hashlib.sha256((PACKAGE / "libraries/bassin_du_lez/tgrad.json").read_bytes()).hexdigest()
    )


def test_wrong_manifest_digest_is_refused():
    with pytest.raises(ValueError, match="Manifest SHA-256"):
        b6.load_verified_library(PACKAGE, "0" * 64, "bassin_du_lez", "tgrad")


@pytest.mark.parametrize("zone, algorithm", [("nairobi", "tgrad"), ("bassin_du_lez", "aco")])
def test_unknown_scope_is_refused(zone, algorithm):
    with pytest.raises(ValueError, match="No library"):
        b6.load_verified_library(PACKAGE, MANIFEST_SHA, zone, algorithm)
