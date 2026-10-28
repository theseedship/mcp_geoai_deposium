"""Paired MAE/P50 skill matrix (#193).

The hermetic fixture proof lives at the bottom: four frozen input artifacts (the forecast rows,
the reference, the monthly P50 and the declared panel manifest), one frozen expected matrix,
and a pinned ``result_digest``. Everything above it pins the policies the
fixture alone could not distinguish from an accident.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import socket
import subprocess
import threading
from datetime import date, datetime, timedelta
from pathlib import Path

import httpx
import pytest

from src import forecast_backtest, forecast_skill
from src.cli import forecast_skill_matrix as cli
from src.forecast_backtest import BacktestValidationError, parse_rows
from src.forecast_skill import (
    EXCLUSION_REASONS,
    REQUIRED_INPUTS,
    SKILL_MATRIX_SCHEMA_VERSION,
    MonthlyP50Baseline,
    paired_skill_matrix,
    result_digest,
)

FIXTURES = Path(__file__).parent / "fixtures" / "skill-matrix-v2"
_REPO_ROOT = Path(__file__).resolve().parent.parent

TRAINING_END = date(2020, 12, 31)
HOLDOUT_START = date(2021, 1, 1)
HOLDOUT_END = date(2021, 1, 31)

DIGEST = "a" * 64
INPUTS = dict.fromkeys(("forecast", "reference", "baseline_p50", "panel_manifest"), DIGEST)


def _forecast(**overrides):
    row = {
        "zone_id": "zone_a",
        "origin_date": "2021-01-04",
        "valid_date": "2021-01-05",
        "horizon_days": 1,
        "variable": "temperature_2m",
        "value": 10.0,
    }
    row.update(overrides)
    return row


def _reference(**overrides):
    row = {
        "zone_id": "zone_a",
        "valid_date": "2021-01-05",
        "variable": "temperature_2m",
        "value": 12.0,
    }
    row.update(overrides)
    return row


def _baseline(p50=20.0, **overrides):
    record = {"zone_id": "zone_a", "variable": "temperature_2m", "month": 1, "p50": p50}
    record.update(overrides)
    return record


def _panel_record(forecast_rows, *, provider="synthetic", model="unit", configuration=None):
    """A panel manifest record: the full rectangle spanned by the rows' own attributes.

    Tests that need a pair to be ABSENT from the submitted rows, or a row to fall outside the
    declared panel, override single fields through ``_matrix(..., panel={...})``.
    """
    return {
        "schema_version": "forecast-skill-panel/v1",
        "provider": provider,
        "model": model,
        "configuration": dict(configuration or {}),
        "zones": sorted({row["zone_id"] for row in forecast_rows}),
        "variables": sorted({row["variable"] for row in forecast_rows}),
        "horizons_days": sorted({row["horizon_days"] for row in forecast_rows}),
        "valid_dates": sorted({row["valid_date"] for row in forecast_rows}),
    }


def _panel(record):
    # Attribute access at call time: the module under test owns the type.
    return forecast_skill.PanelManifest.from_record(record)


def _matrix(forecast_rows, reference_rows, baseline_records, panel=None, **kwargs):
    options = {
        "training_end": TRAINING_END,
        "holdout_start": HOLDOUT_START,
        "holdout_end": HOLDOUT_END,
        "provider": "synthetic",
        "model": "unit",
        "inputs": INPUTS,
    }
    options.update(kwargs)
    record = _panel_record(
        forecast_rows,
        provider=options["provider"],
        model=options["model"],
        configuration=options.get("configuration"),
    )
    record.update(panel or {})
    return paired_skill_matrix(
        parse_rows(forecast_rows, kind="prediction", provider=options["provider"]),
        parse_rows(reference_rows, kind="reference"),
        MonthlyP50Baseline.from_records(baseline_records),
        panel=_panel(record),
        **options,
    )


# ---------------------------------------------------------------------------------------
# The paired contract. This is the whole point of the module: the published panel divided a
# 239-date provider aggregate by a 238-date baseline aggregate, and nothing in either
# artifact recorded that the two numerators covered different days.
# ---------------------------------------------------------------------------------------


def test_forecast_and_baseline_share_every_scored_pair():
    matrix = _matrix([_forecast()], [_reference()], [_baseline()])
    cell = matrix["cells"][0]
    assert cell["n"] == 1
    # One `n` per cell, used by both sides. There is no shape in which they can differ.
    assert cell["mae_forecast"] == cell["abs_error_sum_forecast"] / cell["n"]
    assert cell["mae_baseline"] == cell["abs_error_sum_baseline"] / cell["n"]
    assert cell["mae_skill"] == pytest.approx(1 - 2.0 / 8.0)


def test_a_pair_missing_on_either_side_is_dropped_from_both():
    """A gap removes the pair, never just the side that has it.

    Keeping the baseline half of an unmatched pair is exactly how a denominator ends up
    computed over more days than its numerator.
    """
    rows = [_forecast(), _forecast(zone_id="zone_b")]
    matrix = _matrix(rows, [_reference()], [_baseline(), _baseline(zone_id="zone_b")])
    assert matrix["census"]["exclusions"]["missing_reference"] == 1
    assert matrix["census"]["scored_pairs"] == 1
    assert [cell["n"] for cell in matrix["cells"]] == [1]


def test_missing_baseline_month_excludes_the_pair():
    matrix = _matrix([_forecast()], [_reference()], [_baseline(month=7)])
    assert matrix["census"]["exclusions"]["missing_baseline"] == 1
    assert matrix["cells"] == []
    assert matrix["status"] == "no_scored_pairs"


def test_null_p50_is_a_recorded_gap_not_a_crash():
    """``null``/NaN is a month with no usable normal — a count, not a fatal parse."""
    matrix = _matrix([_forecast()], [_reference()], [_baseline(p50=None)])
    assert matrix["census"]["exclusions"]["non_finite_baseline"] == 1
    assert matrix["census"]["scored_pairs"] == 0


def test_every_exclusion_reason_is_published_even_at_zero():
    matrix = _matrix([_forecast()], [_reference()], [_baseline()])
    assert set(matrix["census"]["exclusions"]) == set(EXCLUSION_REASONS)
    assert set(matrix["census"]["exclusions"].values()) == {0}


# ---------------------------------------------------------------------------------------
# Expected pairs come from the declared panel (v2), not from the rows that were submitted.
# ---------------------------------------------------------------------------------------

_THREE_ZONES = {"zones": ["zone_a", "zone_b", "zone_c"]}


def _three_zone_baseline():
    return [_baseline(zone_id=zone) for zone in _THREE_ZONES["zones"]]


def test_the_census_counts_expected_pairs_from_the_declared_panel():
    rows = [_forecast(), _forecast(zone_id="zone_b")]  # zone_b has no reference
    matrix = _matrix(rows, [_reference()], _three_zone_baseline(), panel=_THREE_ZONES)
    census = matrix["census"]  # zone_c is declared and never submitted
    assert census["expected_pairs"] == 3
    assert census["submitted_forecast_rows"] == 2
    assert census["eligible_pairs"] == 2
    assert census["scored_pairs"] == 1
    assert census["missing_pairs"] == 2
    # zone_c has neither a forecast nor a reference: it is filed under the FIRST gap only.
    assert census["exclusions"] == {
        "not_submitted": 1,
        "missing_reference": 1,
        "missing_baseline": 0,
        "non_finite_baseline": 0,
    }
    assert census["expected_pairs"] == census["scored_pairs"] + census["missing_pairs"]
    assert census["missing_pairs"] == sum(census["exclusions"].values())
    assert (
        census["eligible_pairs"] == census["expected_pairs"] - census["exclusions"]["not_submitted"]
    )


def test_missing_submitted_rows_leave_the_expected_denominator_unchanged():
    """The point of v2: a short archive cannot shrink the number of pairs expected."""
    reference = [_reference(zone_id=zone) for zone in _THREE_ZONES["zones"]]
    full = _matrix(
        [_forecast(zone_id=zone) for zone in _THREE_ZONES["zones"]],
        reference,
        _three_zone_baseline(),
    )
    short = _matrix([_forecast()], reference, _three_zone_baseline(), panel=_THREE_ZONES)
    assert full["census"]["expected_pairs"] == short["census"]["expected_pairs"] == 3
    assert (full["census"]["submitted_forecast_rows"], full["census"]["missing_pairs"]) == (3, 0)
    assert (short["census"]["submitted_forecast_rows"], short["census"]["scored_pairs"]) == (1, 1)
    assert short["census"]["missing_pairs"] == 2
    assert short["census"]["exclusions"]["not_submitted"] == 2
    # The declared panel is part of the result, so it is part of its fingerprint.
    assert full["result_digest"] != short["result_digest"]


def test_a_declared_date_with_no_submitted_row_is_reported_not_dropped():
    matrix = _matrix(
        [_forecast()],
        [_reference()],
        [_baseline()],
        panel={"valid_dates": ["2021-01-05", "2021-01-06"]},
    )
    assert matrix["census"]["expected_pairs"] == 2
    assert matrix["census"]["exclusions"]["not_submitted"] == 1
    assert matrix["census"]["scored_pairs"] == 1


_GAP_REASONS = {
    "not_submitted": "not_submitted",
    "missing_reference": "missing_reference",
    "missing_baseline": "missing_baseline",
    "nan_baseline": "non_finite_baseline",
}


@pytest.mark.parametrize("gap", _GAP_REASONS)
def test_forecast_and_p50_are_dropped_together_on_the_same_keys(gap):
    """Whatever the gap, the pair leaves BOTH sides: n, MAE_forecast and MAE_p50 agree.

    zone_b is wildly wrong on every side (forecast 100, reference 50, P50 -30), so if any
    one side kept its half of the pair the cell MAEs would move away from zone_a's alone.
    """
    forecast_b = _forecast(zone_id="zone_b", value=100.0)
    reference_b = _reference(zone_id="zone_b", value=50.0)
    baseline_b = _baseline(zone_id="zone_b", p50=math.nan if gap == "nan_baseline" else -30.0)
    matrix = _matrix(
        [_forecast()] + ([] if gap == "not_submitted" else [forecast_b]),
        [_reference()] + ([] if gap == "missing_reference" else [reference_b]),
        [_baseline()] + ([] if gap == "missing_baseline" else [baseline_b]),
        panel={"zones": ["zone_a", "zone_b"]},
    )
    cell = matrix["cells"][0]
    assert matrix["census"]["expected_pairs"] == 2
    assert matrix["census"]["scored_pairs"] == cell["n"] == 1
    assert matrix["census"]["exclusions"][_GAP_REASONS[gap]] == 1
    assert cell["mae_forecast"] == pytest.approx(2.0)  # |10 - 12|, zone_a alone
    assert cell["mae_baseline"] == pytest.approx(8.0)  # |20 - 12|, zone_a alone
    assert [zone["zone_id"] for zone in cell["by_zone"]] == ["zone_a"]
    assert matrix["variable_aggregates"][0]["n"] == 1


def test_the_v2_census_has_exactly_the_documented_fields():
    census = _matrix([_forecast()], [_reference()], [_baseline()])["census"]
    assert set(census) == {
        "expected_pairs",
        "submitted_forecast_rows",
        "eligible_pairs",
        "scored_pairs",
        "missing_pairs",
        "exclusions",
        "reference_rows",
        "reference_rows_outside_holdout",
        "baseline_months",
    }


def test_the_contract_is_v2_and_names_the_panel_manifest():
    assert SKILL_MATRIX_SCHEMA_VERSION == "forecast-skill-matrix/v2"
    assert REQUIRED_INPUTS == ("forecast", "reference", "baseline_p50", "panel_manifest")
    assert "not_submitted" in EXCLUSION_REASONS
    matrix = _matrix([_forecast()], [_reference()], [_baseline()])
    assert matrix["schema_version"] == SKILL_MATRIX_SCHEMA_VERSION
    assert matrix["inputs"]["panel_manifest"] == DIGEST


# ---------------------------------------------------------------------------------------
# Scalar policy.
# ---------------------------------------------------------------------------------------


def test_negative_skill_is_published_unclamped():
    """A forecast worse than climatology publishes how much worse. B6 multiplies this."""
    matrix = _matrix(
        [_forecast(value=30.0)],
        [_reference()],
        [_baseline(p50=13.0)],
    )
    assert matrix["cells"][0]["mae_skill"] == pytest.approx(1 - 18.0 / 1.0)
    assert matrix["baseline"]["clamped"] is False


def test_zero_denominator_is_undefined_and_the_cell_survives():
    """A perfect climatology has no ratio — and a cell that vanished would read as unscored."""
    matrix = _matrix([_forecast()], [_reference()], [_baseline(p50=12.0)])
    cell = matrix["cells"][0]
    assert cell["mae_baseline"] == 0
    assert cell["mae_skill"] is None
    assert cell["skill_status"] == "undefined_zero_denominator"
    assert cell["n"] == 1


def test_aggregates_are_sample_weighted_not_averaged_ratios():
    """Two horizons with different coverage: the aggregate is the ratio of the paired sums."""
    forecast = [
        _forecast(),
        _forecast(origin_date="2021-01-03", horizon_days=2, value=4.0),
        _forecast(valid_date="2021-01-06", origin_date="2021-01-05", value=12.5),
    ]
    reference = [_reference(), _reference(valid_date="2021-01-06", value=12.0)]
    matrix = _matrix(forecast, reference, [_baseline()])
    aggregate = matrix["variable_aggregates"][0]
    by_horizon = {cell["horizon_days"]: cell for cell in matrix["cells"]}
    assert by_horizon[1]["n"] == 2
    assert by_horizon[2]["n"] == 1
    expected = 1 - sum(cell["abs_error_sum_forecast"] for cell in matrix["cells"]) / sum(
        cell["abs_error_sum_baseline"] for cell in matrix["cells"]
    )
    assert aggregate["mae_skill"] == pytest.approx(expected)
    macro = sum(cell["mae_skill"] for cell in matrix["cells"]) / 2
    assert aggregate["diagnostics"]["macro_mae_skill"] == pytest.approx(macro)
    # Different questions, different answers. The fixture below shows them diverging on a
    # realistic panel too, so this is not an artefact of a three-row test.
    assert aggregate["mae_skill"] != pytest.approx(macro)


def test_no_aggregate_spans_variables():
    """Summing millimetres and degrees into one numerator is a unit error, not an aggregate."""
    matrix = _matrix(
        [_forecast(), _forecast(variable="precipitation", value=1.0)],
        [_reference(), _reference(variable="precipitation", value=3.0)],
        [_baseline(), _baseline(variable="precipitation", p50=2.0)],
    )
    assert [item["variable"] for item in matrix["variable_aggregates"]] == [
        "precipitation",
        "temperature_2m",
    ]
    assert "mae_skill" not in matrix


def test_baseline_applies_at_every_horizon():
    """The existing panel has no denominator above h=1; the monthly P50 is horizon-free."""
    forecast = [
        _forecast(origin_date="2021-01-04", valid_date="2021-01-05", horizon_days=1),
        _forecast(origin_date="2021-01-01", valid_date="2021-01-05", horizon_days=4),
    ]
    matrix = _matrix(forecast, [_reference()], [_baseline()])
    assert {cell["horizon_days"] for cell in matrix["cells"]} == {1, 4}
    assert all(cell["mae_baseline"] == 8.0 for cell in matrix["cells"])


def test_per_zone_values_sit_beside_every_pooled_cell():
    """A pooled positive can hide a majority of negative zones (#98 acceptance criterion 3)."""
    forecast = [_forecast(), _forecast(zone_id="zone_b", value=40.0)]
    reference = [_reference(), _reference(zone_id="zone_b")]
    baseline = [_baseline(), _baseline(zone_id="zone_b")]
    cell = _matrix(forecast, reference, baseline)["cells"][0]
    assert [zone["zone_id"] for zone in cell["by_zone"]] == ["zone_a", "zone_b"]
    assert cell["by_zone"][0]["mae_skill"] > 0
    assert cell["by_zone"][1]["mae_skill"] < 0
    assert sum(zone["n"] for zone in cell["by_zone"]) == cell["n"]


# ---------------------------------------------------------------------------------------
# What refuses rather than counts.
# ---------------------------------------------------------------------------------------


def test_prediction_originating_in_the_leakage_gap_is_refused():
    with pytest.raises(BacktestValidationError, match="baseline/holdout gap"):
        _matrix(
            [_forecast(origin_date="2020-12-30", valid_date="2021-01-05", horizon_days=6)],
            [_reference()],
            [_baseline()],
            training_end=date(2020, 12, 20),
        )


def test_a_panel_pair_originating_in_the_leakage_gap_is_refused_at_the_manifest():
    """The gap is policed where the pair is DECLARED, so an unsubmitted pair cannot hide it."""
    with pytest.raises(BacktestValidationError, match="baseline/holdout gap"):
        _matrix(
            [_forecast()],
            [_reference()],
            [_baseline()],
            panel={"horizons_days": [1, 6]},  # h=6 from 2021-01-05 originates 2020-12-30
            training_end=date(2020, 12, 20),
        )


@pytest.mark.parametrize(
    ("row", "panel"),
    [
        pytest.param({"zone_id": "zone_x"}, {}, id="zone"),
        pytest.param({"variable": "precipitation"}, {}, id="variable"),
        pytest.param({"valid_date": "2021-01-06", "origin_date": "2021-01-05"}, {}, id="date"),
        pytest.param(
            {"valid_date": "2021-02-10", "origin_date": "2021-02-09"}, {}, id="past-the-holdout"
        ),
        pytest.param(
            {"origin_date": "2021-01-03", "horizon_days": 2},
            {"horizons_days": [1]},
            id="horizon",
        ),
        pytest.param(
            {"origin_date": "2021-01-03", "horizon_days": 2},
            {"horizons_days": [1, 3]},
            id="horizon-between-declared-ones",
        ),
    ],
)
def test_a_submitted_row_outside_the_declared_panel_is_refused(row, panel):
    """v1 silently counted these as `out_of_panel`; a row nobody declared is now an error."""
    with pytest.raises(BacktestValidationError, match="outside the declared panel"):
        _matrix_with_extra_row(_forecast(**row), panel)


def _matrix_with_extra_row(extra, panel):
    """The declared rectangle is the ONE-row panel; ``extra`` is submitted beyond it."""
    record = _panel_record([_forecast()])
    record.update(panel)
    return paired_skill_matrix(
        parse_rows([_forecast(), extra], kind="prediction", provider="synthetic"),
        parse_rows([_reference()], kind="reference"),
        MonthlyP50Baseline.from_records([_baseline()]),
        panel=_panel(record),
        training_end=TRAINING_END,
        holdout_start=HOLDOUT_START,
        holdout_end=HOLDOUT_END,
        provider="synthetic",
        model="unit",
        inputs=INPUTS,
    )


def test_duplicate_prediction_key_is_refused():
    with pytest.raises(BacktestValidationError, match="duplicate prediction key"):
        _matrix([_forecast(), _forecast(value=11.0)], [_reference()], [_baseline()])


def test_duplicate_reference_key_is_refused():
    with pytest.raises(BacktestValidationError, match="duplicate reference key"):
        _matrix([_forecast()], [_reference(), _reference(value=13.0)], [_baseline()])


def test_duplicate_baseline_month_is_refused():
    with pytest.raises(BacktestValidationError, match="duplicate baseline key"):
        MonthlyP50Baseline.from_records([_baseline(), _baseline(p50=21.0)])


@pytest.mark.parametrize(
    ("records", "message"),
    [
        ([["zone_a", "temperature_2m", 1, 20.0]], "must be an object"),
        ([{"zone_id": "zone_a", "variable": "temperature_2m", "month": 1}], "missing: p50"),
        ([_baseline(zone_id="")], "zone_id must be a non-empty string"),
        ([_baseline(variable=None)], "variable must be a non-empty string"),
        ([_baseline(month=0)], "month must be an integer in 1..12"),
        ([_baseline(month=True)], "month must be an integer in 1..12"),
        ([_baseline(p50="warm")], "p50 must be numeric or null"),
        ([_baseline(p50=10**400)], "p50 must be numeric or null"),  # float() overflows (#221)
        ([], "at least one monthly P50"),
    ],
)
def test_malformed_baseline_records_are_refused(records, message):
    """The denominator is the half a reviewer cannot re-derive from the forecast archive."""
    with pytest.raises(BacktestValidationError, match=message):
        MonthlyP50Baseline.from_records(records)


def test_reversed_holdout_window_is_refused():
    with pytest.raises(BacktestValidationError, match="holdout_start must be before"):
        _matrix([_forecast()], [_reference()], [_baseline()], holdout_end=date(2020, 12, 31 - 1))


@pytest.mark.parametrize(("field", "value"), [("provider", ""), ("model", "")])
def test_the_frozen_identity_cannot_be_blank(field, value):
    with pytest.raises(BacktestValidationError, match=f"{field} must be a non-empty string"):
        paired_skill_matrix(
            parse_rows([_forecast()], kind="prediction", provider="synthetic"),
            parse_rows([_reference()], kind="reference"),
            MonthlyP50Baseline.from_records([_baseline()]),
            training_end=TRAINING_END,
            holdout_start=HOLDOUT_START,
            holdout_end=HOLDOUT_END,
            inputs=INPUTS,
            panel=_panel(_panel_record([_forecast(), _forecast(zone_id="zone_b")])),
            **{"provider": "synthetic", "model": "unit", field: value},
        )


def test_a_blank_input_name_is_refused():
    with pytest.raises(BacktestValidationError, match="input name must be a non-empty string"):
        _matrix([_forecast()], [_reference()], [_baseline()], inputs={**INPUTS, "": DIGEST})


def test_reference_rows_outside_the_panel_are_counted_not_scored():
    """A wider truth archive is usable; the census says how much of it the panel ignored."""
    reference = [_reference(), _reference(valid_date="2020-11-05", value=4.0)]
    census = _matrix([_forecast()], reference, [_baseline()])["census"]
    assert census["reference_rows"] == 2
    assert census["reference_rows_outside_holdout"] == 1
    assert census["scored_pairs"] == 1


def test_mixed_providers_are_refused():
    rows = parse_rows([_forecast()], kind="prediction", provider="synthetic")
    rows += parse_rows([_forecast(zone_id="zone_b")], kind="prediction", provider="other")
    with pytest.raises(BacktestValidationError, match="mix providers"):
        paired_skill_matrix(
            rows,
            parse_rows([_reference()], kind="reference"),
            MonthlyP50Baseline.from_records([_baseline()]),
            training_end=TRAINING_END,
            holdout_start=HOLDOUT_START,
            holdout_end=HOLDOUT_END,
            provider="synthetic",
            model="unit",
            inputs=INPUTS,
            panel=_panel(_panel_record([_forecast(), _forecast(zone_id="zone_b")])),
        )


def test_declared_provider_must_match_the_rows():
    with pytest.raises(BacktestValidationError, match="does not match forecast rows"):
        paired_skill_matrix(
            parse_rows([_forecast()], kind="prediction", provider="other"),
            parse_rows([_reference()], kind="reference"),
            MonthlyP50Baseline.from_records([_baseline()]),
            training_end=TRAINING_END,
            holdout_start=HOLDOUT_START,
            holdout_end=HOLDOUT_END,
            provider="synthetic",
            model="unit",
            inputs=INPUTS,
            panel=_panel(_panel_record([_forecast(), _forecast(zone_id="zone_b")])),
        )


@pytest.mark.parametrize(
    ("declared", "manifest"),
    [
        pytest.param({"provider": "other"}, {}, id="provider"),
        pytest.param({"model": "other"}, {}, id="model"),
        pytest.param({"configuration": {"run": "12z"}}, {}, id="configuration"),
        pytest.param({}, {"configuration": {"run": "12z"}}, id="configuration-only-in-manifest"),
        pytest.param(
            {"configuration": {"run": 1}}, {"configuration": {"run": True}}, id="1-is-not-true"
        ),
        pytest.param(
            {"configuration": {"run": 1}}, {"configuration": {"run": 1.0}}, id="1-is-not-1.0"
        ),
    ],
)
def test_an_identity_that_contradicts_the_manifest_is_refused(declared, manifest):
    """The matrix is published under the identity it was DECLARED with; the manifest froze it."""
    rows = [_forecast()]
    provider = declared.get("provider", "synthetic")
    record = _panel_record(rows, **{k: v for k, v in manifest.items() if k == "configuration"})
    with pytest.raises(BacktestValidationError, match="contradicts the panel manifest"):
        paired_skill_matrix(
            parse_rows(rows, kind="prediction", provider=provider),
            parse_rows([_reference()], kind="reference"),
            MonthlyP50Baseline.from_records([_baseline()]),
            panel=_panel(record),
            training_end=TRAINING_END,
            holdout_start=HOLDOUT_START,
            holdout_end=HOLDOUT_END,
            provider=provider,
            model=declared.get("model", "unit"),
            configuration=declared.get("configuration"),
            inputs=INPUTS,
        )


def test_an_identity_that_matches_the_manifest_is_accepted():
    matrix = _matrix(
        [_forecast()],
        [_reference()],
        [_baseline()],
        configuration={"run": "00z", "nested": {"a": [1, 2]}},
    )
    assert matrix["frozen_configuration"]["configuration"] == {
        "run": "00z",
        "nested": {"a": [1, 2]},
    }


def test_a_nan_in_the_declared_configuration_is_a_refusal_not_a_crash():
    with pytest.raises(BacktestValidationError, match="configuration"):
        _matrix([_forecast()], [_reference()], [_baseline()], configuration={"x": math.nan})


def test_panel_dates_must_sit_inside_the_declared_holdout():
    with pytest.raises(BacktestValidationError, match="outside the holdout"):
        _matrix(
            [_forecast()],
            [_reference()],
            [_baseline()],
            panel={"valid_dates": ["2021-01-05", "2021-02-10"]},
        )


_PANEL_OK = _panel_record([_forecast()])


@pytest.mark.parametrize(
    ("change", "message"),
    [
        pytest.param({"zones": ["zone_a", "zone_a"]}, "duplicate zone", id="duplicate-zone"),
        pytest.param(
            {"variables": ["temperature_2m", "temperature_2m"]},
            "duplicate variable",
            id="duplicate-variable",
        ),
        pytest.param({"horizons_days": [1, 1]}, "duplicate horizon", id="duplicate-horizon"),
        pytest.param(
            {"valid_dates": ["2021-01-05", "2021-01-05"]}, "duplicate valid date", id="dup-date"
        ),
        pytest.param(
            {"valid_dates": ["2021-01-05", "20210105"]},
            "duplicate valid date",
            id="same-date-spelled-twice",
        ),
        pytest.param({"schema_version": "forecast-skill-panel/v0"}, "schema_version", id="schema"),
        pytest.param({"extra": 1}, "unknown field", id="unknown-field"),
        pytest.param({"provider": None}, "provider", id="provider-none"),
        pytest.param({"provider": ["synthetic"]}, "provider", id="provider-list"),
        pytest.param({"provider": ""}, "provider", id="provider-blank"),
        pytest.param({"model": 3}, "model", id="model-int"),
        pytest.param({"configuration": None}, "configuration", id="configuration-none"),
        pytest.param({"configuration": ["a"]}, "configuration", id="configuration-list"),
        pytest.param({"zones": "zone_a"}, "zones", id="zones-string"),
        pytest.param({"zones": None}, "zones", id="zones-none"),
        pytest.param({"zones": []}, "zones", id="zones-empty"),
        pytest.param({"zones": [["zone_a"]]}, "zones", id="zones-nested-list"),
        pytest.param({"zones": [{}]}, "zones", id="zones-dict"),
        pytest.param({"zones": [""]}, "zones", id="zones-blank"),
        pytest.param({"variables": [None]}, "variables", id="variables-none-item"),
        pytest.param({"variables": []}, "variables", id="variables-empty"),
        pytest.param({"horizons_days": "1"}, "horizons_days", id="horizons-string"),
        pytest.param({"horizons_days": [True]}, "horizons_days", id="horizons-bool"),
        pytest.param({"horizons_days": [0]}, "horizons_days", id="horizons-zero"),
        pytest.param({"horizons_days": ["1"]}, "horizons_days", id="horizons-str-item"),
        pytest.param({"horizons_days": [1.5]}, "horizons_days", id="horizons-float"),
        pytest.param({"horizons_days": [[1]]}, "horizons_days", id="horizons-nested"),
        pytest.param({"horizons_days": [10**400]}, "horizons_days", id="horizons-beyond-float"),
        pytest.param({"horizons_days": [10**9]}, "horizons_days", id="horizons-beyond-calendar"),
        pytest.param({"horizons_days": []}, "horizons_days", id="horizons-empty"),
        pytest.param({"valid_dates": "2021-01-05"}, "valid_dates", id="dates-string"),
        pytest.param({"valid_dates": [None]}, "valid_dates", id="dates-none-item"),
        pytest.param({"valid_dates": [20210105]}, "valid_dates", id="dates-int-item"),
        pytest.param({"valid_dates": [["2021-01-05"]]}, "valid_dates", id="dates-nested"),
        pytest.param({"valid_dates": ["not a date"]}, "valid_dates", id="dates-garbage"),
        pytest.param({"valid_dates": []}, "valid_dates", id="dates-empty"),
        pytest.param(
            {"valid_dates": ["0001-01-02"], "horizons_days": [5]},
            "horizons_days",
            id="origin-before-year-1",
        ),
    ],
)
def test_a_malformed_panel_manifest_is_a_validation_error(change, message):
    """Wrong types must reach the module's own error, never a TypeError or an OverflowError."""
    with pytest.raises(BacktestValidationError, match=message):
        _panel({**_PANEL_OK, **change})


def _direct(**overrides):
    """A ``PanelManifest`` built through its constructor, bypassing ``from_record``."""
    fields = {
        "provider": "synthetic",
        "model": "unit",
        "configuration": {},
        "zones": ("zone_a",),
        "variables": ("temperature_2m",),
        "horizons_days": (1,),
        "valid_dates": (date(2021, 1, 5),),
    }
    fields.update(overrides)
    return forecast_skill.PanelManifest(**fields)


def test_a_directly_constructed_panel_is_valid_when_its_axes_are():
    panel = _direct(zones=["zone_a", "zone_b"], horizons_days=[1, 3])  # lists are normalised
    assert panel.expected_pairs == 2 * 1 * 1 * 2


@pytest.mark.parametrize(
    ("change", "message"),
    [
        pytest.param({"zones": ("zone_a", "zone_a")}, "duplicate zone", id="duplicate-zone"),
        pytest.param(
            {"variables": ("temperature_2m", "temperature_2m")},
            "duplicate variable",
            id="duplicate-variable",
        ),
        pytest.param({"horizons_days": (1, 1)}, "duplicate horizon", id="duplicate-horizon"),
        pytest.param(
            {"valid_dates": (date(2021, 1, 5), date(2021, 1, 5))},
            "duplicate valid date",
            id="duplicate-date",
        ),
        pytest.param({"zones": ()}, "zones", id="no-zones"),
        pytest.param({"zones": "zone_a"}, "zones", id="zones-string"),
        pytest.param({"zones": (["zone_a"],)}, "zones", id="zones-nested"),
        pytest.param({"horizons_days": (True,)}, "horizons_days", id="horizon-bool"),
        pytest.param({"horizons_days": (0,)}, "horizons_days", id="horizon-zero"),
        pytest.param({"valid_dates": ("2021-01-05",)}, "valid_dates", id="date-as-string"),
        pytest.param({"valid_dates": ()}, "valid_dates", id="no-dates"),
        pytest.param({"provider": ""}, "provider", id="provider-blank"),
        pytest.param({"model": None}, "model", id="model-none"),
        pytest.param({"configuration": {"x": math.nan}}, "configuration", id="configuration-nan"),
        pytest.param({"configuration": ["a"]}, "configuration", id="configuration-list"),
        pytest.param(
            {"valid_dates": (date(1, 1, 2),), "horizons_days": (5,)},
            "horizons_days",
            id="origin-before-year-1",
        ),
    ],
)
def test_direct_construction_cannot_bypass_the_manifest_invariants(change, message):
    """``expected_pairs`` must equal the number of distinct declared pairs for ANY instance.

    A repeated zone would otherwise be counted twice by the arithmetic denominator while
    the pair itself exists once, and ``missing_pairs`` would exceed the sum of exclusions.
    """
    with pytest.raises(BacktestValidationError, match=message):
        _direct(**change)


def test_a_duplicated_zone_cannot_reach_the_census_through_direct_construction():
    with pytest.raises(BacktestValidationError, match="duplicate zone"):
        paired_skill_matrix(
            parse_rows([_forecast()], kind="prediction", provider="synthetic"),
            parse_rows([_reference()], kind="reference"),
            MonthlyP50Baseline.from_records([_baseline()]),
            panel=_direct(zones=("zone_a", "zone_a")),
            training_end=TRAINING_END,
            holdout_start=HOLDOUT_START,
            holdout_end=HOLDOUT_END,
            provider="synthetic",
            model="unit",
            inputs=INPUTS,
        )


@pytest.mark.parametrize(
    ("horizon", "allowed"),
    [
        pytest.param(16, True, id="origin-on-training-end"),
        pytest.param(15, False, id="first-day-of-the-gap"),
        pytest.param(5, False, id="last-day-of-the-gap"),
        pytest.param(4, True, id="origin-on-holdout-start"),
    ],
)
def test_the_leakage_gap_boundaries_are_exact(horizon, allowed):
    """Gap is training_end < origin < holdout_start, counted without enumerating pairs."""
    origin = (date(2021, 1, 5) - timedelta(days=horizon)).isoformat()
    row = _forecast(origin_date=origin, horizon_days=horizon)
    kwargs = {"training_end": date(2020, 12, 20)}
    if allowed:
        assert _matrix([row], [_reference()], [_baseline()], **kwargs)
    else:
        with pytest.raises(BacktestValidationError, match="baseline/holdout gap"):
            _matrix([row], [_reference()], [_baseline()], **kwargs)


def test_a_declared_date_before_the_holdout_is_refused():
    """The lower bound of the window: both window tests above use a date past `holdout_end`."""
    with pytest.raises(BacktestValidationError, match="2020-12-31 is outside the holdout"):
        _matrix(
            [_forecast()],
            [_reference()],
            [_baseline()],
            panel={"valid_dates": ["2020-12-31", "2021-01-05"]},
        )


def _exactly(message):
    return f"^{re.escape(message)}$"


_NON_EMPTY_LIST = "panel manifest {} must be a non-empty list"


@pytest.mark.parametrize(
    ("change", "message"),
    [
        pytest.param({"valid_dates": None}, _NON_EMPTY_LIST.format("valid_dates"), id="dates-null"),
        pytest.param({"valid_dates": 5}, _NON_EMPTY_LIST.format("valid_dates"), id="dates-int"),
        pytest.param(
            {"valid_dates": {"2021-01-05": 1}},
            _NON_EMPTY_LIST.format("valid_dates"),
            id="dates-object",
        ),
        pytest.param({"horizons_days": 7}, _NON_EMPTY_LIST.format("horizons_days"), id="hz-int"),
        pytest.param(
            {"horizons_days": [-1]},
            "panel manifest horizons_days must contain positive integers",
            id="hz-negative",
        ),
    ],
)
def test_the_manifest_axes_refuse_with_the_documented_message(change, message):
    """The whole message, not the field name: a later check must not be able to stand in."""
    with pytest.raises(BacktestValidationError, match=_exactly(message)):
        _panel({**_PANEL_OK, **change})


@pytest.mark.parametrize(
    ("change", "message"),
    [
        pytest.param(
            {"valid_dates": {date(2021, 1, 5)}},
            _NON_EMPTY_LIST.format("valid_dates"),
            id="dates-set",
        ),
        pytest.param(
            {"horizons_days": {1}}, _NON_EMPTY_LIST.format("horizons_days"), id="horizons-set"
        ),
        pytest.param(
            {"valid_dates": (datetime(2021, 1, 5, 12, 0),)},
            "panel manifest valid_dates must contain ISO dates",
            id="dates-datetime",
        ),
        pytest.param(
            {"horizons_days": (-1,)},
            "panel manifest horizons_days must contain positive integers",
            id="hz-negative",
        ),
        pytest.param(
            {"configuration": {"x": object()}},
            "configuration must be finite, JSON-serializable data",
            id="configuration-object",
        ),
        pytest.param(
            {"valid_dates": (date(1, 1, 2), date(2021, 1, 5)), "horizons_days": (1, 5)},
            "panel manifest horizons_days: 5 days before 0001-01-02 is outside the calendar",
            id="calendar-uses-the-longest-horizon",
        ),
    ],
)
def test_direct_construction_refuses_with_the_documented_message(change, message):
    with pytest.raises(BacktestValidationError, match=_exactly(message)):
        _direct(**change)


def test_the_panel_keeps_its_own_copy_of_the_configuration():
    configuration = {"run": "00z", "nested": {"a": [1, 2]}}
    panel = _direct(configuration=configuration)
    configuration["nested"]["a"].append(3)
    configuration["run"] = "12z"
    assert panel.configuration == {"run": "00z", "nested": {"a": [1, 2]}}


def test_an_empty_configuration_that_is_not_an_object_is_refused():
    """`[]` is falsy: `configuration or {}` would have let it through as `{}`."""
    with pytest.raises(BacktestValidationError, match="configuration must be a JSON object"):
        paired_skill_matrix(
            parse_rows([_forecast()], kind="prediction", provider="synthetic"),
            parse_rows([_reference()], kind="reference"),
            MonthlyP50Baseline.from_records([_baseline()]),
            panel=_panel(_PANEL_OK),
            training_end=TRAINING_END,
            holdout_start=HOLDOUT_START,
            holdout_end=HOLDOUT_END,
            provider="synthetic",
            model="unit",
            configuration=[],
            inputs=INPUTS,
        )


@pytest.mark.parametrize("horizon", [7, 2])
def test_a_row_whose_horizon_contradicts_its_dates_is_refused(horizon):
    """Built directly, bypassing `parse_rows`: origin 01-04 -> valid 01-05 is h=1, whatever it says.

    Membership derives the horizon from the dates, so the row is DECLARED when the panel
    lists h=1; labelling the cell with the row's own claim would file it under h=7.
    """
    row = forecast_backtest.BacktestRow(
        "zone_a", date(2021, 1, 4), date(2021, 1, 5), horizon, "temperature_2m", 10.0, "synthetic"
    )
    panel = _panel({**_PANEL_OK, "horizons_days": [1, 7]})
    with pytest.raises(BacktestValidationError, match=f"horizon_days {horizon} contradicts"):
        _score_rows([row], panel)


def test_a_row_whose_horizon_matches_its_dates_is_scored_in_that_cell():
    row = forecast_backtest.BacktestRow(
        "zone_a", date(2021, 1, 4), date(2021, 1, 5), 1, "temperature_2m", 10.0, "synthetic"
    )
    matrix = _score_rows([row], _panel({**_PANEL_OK, "horizons_days": [1, 7]}))
    assert [cell["horizon_days"] for cell in matrix["cells"]] == [1]


def _score_rows(rows, panel):
    return paired_skill_matrix(
        rows,
        parse_rows([_reference()], kind="reference"),
        MonthlyP50Baseline.from_records([_baseline()]),
        panel=panel,
        training_end=TRAINING_END,
        holdout_start=HOLDOUT_START,
        holdout_end=HOLDOUT_END,
        provider="synthetic",
        model="unit",
        inputs=INPUTS,
    )


def test_a_gap_hit_by_only_one_date_and_one_horizon_is_found():
    """Every date is checked against every horizon, not just the first date."""
    with pytest.raises(BacktestValidationError, match="originating 2020-12-26 .*gap"):
        _matrix(
            [_forecast(origin_date="2020-12-11", horizon_days=25)],
            [_reference()],
            [_baseline()],
            panel={"valid_dates": ["2021-01-05", "2021-01-20"], "horizons_days": [1, 25]},
            training_end=date(2020, 12, 20),
            holdout_end=date(2021, 1, 31),
        )


def _huge_panel(zones=1000, variables=1000, dates=1000):
    day = date(2021, 1, 5)
    return _direct(
        zones=tuple(f"zone_{i}" for i in range(zones)),
        variables=tuple(f"var_{i}" for i in range(variables)),
        valid_dates=tuple(day + timedelta(days=i) for i in range(dates)),
    )


@pytest.fixture
def deadline():
    """Fail, rather than hang or exhaust memory, if the body runs past ``seconds``."""
    import signal

    def _expired(*_args):
        raise AssertionError("did not finish in time: the declared panel is being enumerated")

    previous = signal.signal(signal.SIGALRM, _expired)
    signal.alarm(10)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)


def test_a_huge_declared_panel_is_counted_without_being_materialised(deadline):
    """10^9 expected pairs, 3 submitted: the absent ones are arithmetic, not a set."""
    panel = _huge_panel()
    assert panel.expected_pairs == 10**9
    day = "2021-01-05"
    rows = [
        _forecast(
            zone_id=f"zone_{i}", variable=f"var_{i}", origin_date="2021-01-04", valid_date=day
        )
        for i in range(3)
    ]
    matrix = paired_skill_matrix(
        parse_rows(rows, kind="prediction", provider="synthetic"),
        parse_rows(
            [_reference(zone_id=f"zone_{i}", variable=f"var_{i}") for i in range(2)],
            kind="reference",
        ),
        MonthlyP50Baseline.from_records(
            [_baseline(zone_id=f"zone_{i}", variable=f"var_{i}") for i in range(3)]
        ),
        panel=panel,
        training_end=TRAINING_END,
        holdout_start=HOLDOUT_START,
        holdout_end=date(2024, 12, 31),
        provider="synthetic",
        model="unit",
        inputs=INPUTS,
    )
    census = matrix["census"]
    assert census["expected_pairs"] == 10**9
    assert census["submitted_forecast_rows"] == census["eligible_pairs"] == 3
    assert census["scored_pairs"] == 2
    assert census["missing_pairs"] == 10**9 - 2
    assert census["exclusions"] == {
        "not_submitted": 10**9 - 3,
        "missing_reference": 1,
        "missing_baseline": 0,
        "non_finite_baseline": 0,
    }
    assert census["missing_pairs"] == sum(census["exclusions"].values())


def test_a_row_outside_a_huge_declared_panel_is_still_refused(deadline):
    row = _forecast(zone_id="zone_1000", origin_date="2021-01-04", valid_date="2021-01-05")
    with pytest.raises(BacktestValidationError, match="outside the declared panel"):
        paired_skill_matrix(
            parse_rows([row], kind="prediction", provider="synthetic"),
            parse_rows([_reference()], kind="reference"),
            MonthlyP50Baseline.from_records([_baseline()]),
            panel=_huge_panel(),
            training_end=TRAINING_END,
            holdout_start=HOLDOUT_START,
            holdout_end=date(2024, 12, 31),
            provider="synthetic",
            model="unit",
            inputs=INPUTS,
        )


def test_the_panel_must_be_a_parsed_manifest_not_a_raw_record():
    """A raw dict would skip every check ``from_record`` makes."""
    with pytest.raises(BacktestValidationError, match="panel must be a PanelManifest"):
        paired_skill_matrix(
            parse_rows([_forecast()], kind="prediction", provider="synthetic"),
            parse_rows([_reference()], kind="reference"),
            MonthlyP50Baseline.from_records([_baseline()]),
            panel=_PANEL_OK,
            training_end=TRAINING_END,
            holdout_start=HOLDOUT_START,
            holdout_end=HOLDOUT_END,
            provider="synthetic",
            model="unit",
            inputs=INPUTS,
        )


def test_a_configuration_that_is_not_an_object_is_refused():
    with pytest.raises(BacktestValidationError, match="configuration must be a JSON object"):
        paired_skill_matrix(
            parse_rows([_forecast()], kind="prediction", provider="synthetic"),
            parse_rows([_reference()], kind="reference"),
            MonthlyP50Baseline.from_records([_baseline()]),
            panel=_panel(_PANEL_OK),
            training_end=TRAINING_END,
            holdout_start=HOLDOUT_START,
            holdout_end=HOLDOUT_END,
            provider="synthetic",
            model="unit",
            configuration=["run", "00z"],
            inputs=INPUTS,
        )


@pytest.mark.parametrize("record", [None, [], "panel", 3, [_PANEL_OK]])
def test_a_panel_manifest_that_is_not_an_object_is_refused(record):
    with pytest.raises(BacktestValidationError, match="panel manifest must be a JSON object"):
        _panel(record)


@pytest.mark.parametrize("field", sorted(_PANEL_OK))
def test_a_panel_manifest_missing_a_field_is_refused(field):
    record = {key: value for key, value in _PANEL_OK.items() if key != field}
    with pytest.raises(BacktestValidationError, match=f"missing: {field}"):
        _panel(record)


def test_overlapping_training_and_holdout_is_refused():
    with pytest.raises(BacktestValidationError, match="overlap"):
        _matrix([_forecast()], [_reference()], [_baseline()], training_end=date(2021, 1, 2))


@pytest.mark.parametrize(
    "inputs",
    [
        {},
        {**INPUTS, "forecast": "short"},
        {**INPUTS, "forecast": "z" * 64},
    ],
)
def test_a_matrix_without_verifiable_input_digests_is_refused(inputs):
    """Without them the immutability claim names no bytes and cannot be checked."""
    with pytest.raises(BacktestValidationError):
        _matrix([_forecast()], [_reference()], [_baseline()], inputs=inputs)


@pytest.mark.parametrize("omitted", REQUIRED_INPUTS)
def test_a_partial_input_manifest_is_refused(omitted):
    """All four jointly determine every cell, so any three of them leave one unchecked.

    A manifest carrying only the forecast digest looks complete and lets nobody verify the
    truth it was compared against or the denominator it was divided by.
    """
    inputs = {name: DIGEST for name in REQUIRED_INPUTS if name != omitted}
    with pytest.raises(BacktestValidationError, match=f"missing digests for: {omitted}"):
        _matrix([_forecast()], [_reference()], [_baseline()], inputs=inputs)


def test_extra_input_digests_are_kept():
    """A caller recording the archive manifest too is adding provenance, not breaking it."""
    matrix = _matrix(
        [_forecast()],
        [_reference()],
        [_baseline()],
        inputs={**INPUTS, "archive_manifest": DIGEST},
    )
    assert matrix["inputs"]["archive_manifest"] == DIGEST


# ---------------------------------------------------------------------------------------
# Identity of the artifact.
# ---------------------------------------------------------------------------------------


def test_result_digest_ignores_the_producing_commit():
    """Same inputs, later commit: same result. The commit is recorded beside it, not in it."""
    first = _matrix([_forecast()], [_reference()], [_baseline()], code_ref={"commit": "aaa"})
    second = _matrix([_forecast()], [_reference()], [_baseline()], code_ref={"commit": "bbb"})
    assert first["result_digest"] == second["result_digest"]
    assert first["code_ref"] != second["code_ref"]


def test_result_digest_changes_with_the_frozen_configuration():
    first = _matrix([_forecast()], [_reference()], [_baseline()])
    second = _matrix([_forecast()], [_reference()], [_baseline()], configuration={"run": "12z"})
    assert first["result_digest"] != second["result_digest"]


def test_result_digest_is_independent_of_row_order():
    """The sums are order-independent, so the fingerprint is too."""
    rows = [_forecast(), _forecast(zone_id="zone_b", value=9.5)]
    reference = [_reference(), _reference(zone_id="zone_b")]
    baseline = [_baseline(), _baseline(zone_id="zone_b")]
    forward = _matrix(rows, reference, baseline)
    backward = _matrix(rows[::-1], reference[::-1], baseline[::-1])
    assert forward["result_digest"] == backward["result_digest"]


def test_result_digest_recomputes_from_the_published_body():
    matrix = _matrix([_forecast()], [_reference()], [_baseline()], code_ref={"commit": "aaa"})
    assert result_digest(matrix) == matrix["result_digest"]


# ---------------------------------------------------------------------------------------
# The hermetic fixture proof (#193 acceptance check 5).
#
# Four frozen artifacts (rows, reference, monthly P50, declared panel manifest), no network,
# no provider, no clock. The expected matrix is
# committed with `code_ref: null` on purpose: pinning a commit would break the fixture on
# the next commit while the result it describes had not moved at all.
# ---------------------------------------------------------------------------------------

# v1 was 544ed8d8c623283b5d5095aa13b42df05d7ed12a3cd38d51c640f2009d016d1f. The cells and
# aggregates are byte-identical; the body gained the declared-panel census, `not_submitted`
# and the `panel_manifest` input digest, and the forecast input lost its 3 out-of-panel rows.
FIXTURE_DIGEST = "f45a033b8e24277b73949bd3ba87f8183b0d310322eef9135362175c544290f1"


def _fixture_argv(tmp_path: Path, *extra: str) -> list[str]:
    """The exact invocation that produced the committed expectation, minus ``--allow-dirty``.

    The override is here because the proof has to run from a developer's modified checkout;
    the artifact it writes still records ``dirty: true``, so nothing claims otherwise.
    """
    return [
        "--forecast",
        str(FIXTURES / "forecast.jsonl"),
        "--reference",
        str(FIXTURES / "reference.jsonl"),
        "--baseline-p50",
        str(FIXTURES / "baseline_p50.json"),
        "--panel-manifest",
        str(FIXTURES / "panel_manifest.json"),
        "--provider",
        "synthetic",
        "--model",
        "hermetic-v1",
        "--configuration",
        '{"panel": "synthetic-14d"}',
        "--training-end",
        "2021-01-24",
        "--holdout-start",
        "2021-01-25",
        "--holdout-end",
        "2021-02-07",
        "--output",
        str(tmp_path / "matrix.json"),
        "--allow-dirty",
        *extra,
    ]


def _run_fixture(tmp_path: Path, *extra: str) -> dict:
    return cli.run(cli.build_parser().parse_args(_fixture_argv(tmp_path, *extra)))


def test_hermetic_fixture_reproduces_the_frozen_matrix(tmp_path):
    produced = _run_fixture(tmp_path)
    expected = json.loads((FIXTURES / "expected_matrix.json").read_text(encoding="utf-8"))
    assert produced["result_digest"] == FIXTURE_DIGEST
    assert {**produced, "code_ref": None} == expected
    assert (
        json.loads((tmp_path / "matrix.json").read_text(encoding="utf-8"))["result_digest"]
        == FIXTURE_DIGEST
    )


def test_hermetic_fixture_exercises_every_exclusion_reason(tmp_path):
    """A fixture where the counters are all zero proves only that they exist."""
    census = _run_fixture(tmp_path)["census"]
    assert all(census["exclusions"][reason] > 0 for reason in EXCLUSION_REASONS)
    assert census["missing_pairs"] == sum(census["exclusions"].values())


def test_hermetic_fixture_expected_denominator_comes_from_the_manifest(tmp_path):
    """3 zones x 2 variables x 3 horizons x 14 valid dates, whatever was submitted."""
    census = _run_fixture(tmp_path)["census"]
    assert census["expected_pairs"] == 3 * 2 * 3 * 14 == 252
    assert census["submitted_forecast_rows"] < census["expected_pairs"]
    assert census["exclusions"]["not_submitted"] == (
        census["expected_pairs"] - census["submitted_forecast_rows"]
    )
    assert census["eligible_pairs"] == census["submitted_forecast_rows"]
    assert census["expected_pairs"] == census["scored_pairs"] + census["missing_pairs"]


def test_hermetic_fixture_shows_both_scalar_policies(tmp_path):
    matrix = _run_fixture(tmp_path)
    skills = [cell["mae_skill"] for cell in matrix["cells"]]
    assert max(skills) > 0 and min(skills) < -1  # unclamped, both directions
    undefined = [
        zone
        for cell in matrix["cells"]
        for zone in cell["by_zone"]
        if zone["skill_status"] == "undefined_zero_denominator"
    ]
    assert undefined and all(zone["mae_skill"] is None for zone in undefined)


def test_hermetic_fixture_separates_the_aggregate_from_the_macro_average(tmp_path):
    """They coincide when every horizon scores the same pairs; real coverage differs."""
    aggregates = {item["variable"]: item for item in _run_fixture(tmp_path)["variable_aggregates"]}
    temperature = aggregates["temperature_2m"]
    assert temperature["mae_skill"] != pytest.approx(temperature["diagnostics"]["macro_mae_skill"])


def test_hermetic_fixture_is_stable_across_runs(tmp_path):
    first = _run_fixture(tmp_path)
    second = _run_fixture(tmp_path / "again")
    assert first["result_digest"] == second["result_digest"]
    assert first["schema_version"] == SKILL_MATRIX_SCHEMA_VERSION


def test_cli_records_the_digest_of_the_bytes_it_parsed(tmp_path):
    matrix = _run_fixture(tmp_path)
    for name in REQUIRED_INPUTS:
        assert len(matrix["inputs"][name]) == 64
    assert matrix["inputs"]["panel_manifest"] == _sha256(FIXTURES / "panel_manifest.json")
    assert matrix["code_ref"]["repo"] == "deposium_geoai"


# Longer than Python's 4,300-digit int limit: ``json.loads`` raises a plain ``ValueError`` on
# it, not a ``JSONDecodeError`` (#221).
_PAST_THE_DIGIT_LIMIT = "9" * 5000


@pytest.mark.parametrize(
    "extra",
    [
        ("--configuration", "not json"),
        ("--configuration", "[1, 2]"),
        ("--training-end", "2021-13-40"),
        pytest.param(
            ("--configuration", '{"x": ' + _PAST_THE_DIGIT_LIMIT + "}"),
            id="configuration-past-the-digit-limit",
        ),
    ],
)
def test_cli_refuses_and_reports_rather_than_traces(tmp_path, capsys, extra):
    """A refusal is one machine-readable line, not a traceback a CI log swallows."""
    assert cli.main(_fixture_argv(tmp_path, *extra)) == 1
    assert json.loads(capsys.readouterr().out)["status"] == "refused"
    assert not (tmp_path / "matrix.json").exists()


def test_cli_refuses_a_miswired_input_rather_than_scoring_it(tmp_path, capsys):
    """The forecast archive handed in as the reference: refused, not quietly half-scored.

    Reading a multi-horizon archive as truth collapses three rows onto one
    ``(zone, day, variable)`` key. Taking the last one would produce a plausible matrix
    whose denominator came from the file being scored.
    """
    argv = _fixture_argv(tmp_path)
    argv[argv.index("--reference") + 1] = str(FIXTURES / "forecast.jsonl")
    assert cli.main(argv) == 1
    refusal = json.loads(capsys.readouterr().out)
    assert refusal["status"] == "refused"
    assert "duplicate reference key" in refusal["error"]


def test_cli_requires_the_panel_manifest(tmp_path, capsys):
    """The fourth input is not optional: without it the denominator would be the rows again."""
    argv = _fixture_argv(tmp_path)
    at = argv.index("--panel-manifest")
    del argv[at : at + 2]
    with pytest.raises(SystemExit) as stopped:
        cli.main(argv)
    assert stopped.value.code == 2
    assert "--panel-manifest" in capsys.readouterr().err
    assert not (tmp_path / "matrix.json").exists()


def _with_panel_manifest(tmp_path: Path, content: bytes | None) -> list[str]:
    argv = _fixture_argv(tmp_path)
    path = tmp_path / "panel.json"
    if content is not None:
        path.write_bytes(content)
    return _replace_flag(argv, "--panel-manifest", str(path))


@pytest.mark.parametrize(
    ("content", "message"),
    [
        pytest.param(None, "cannot read", id="absent"),
        pytest.param(b"\xff\xfe\x00not utf-8", "not UTF-8", id="binary"),
        pytest.param(b"{ not json", "not valid JSON", id="invalid-json"),
        pytest.param(b"", "not valid JSON", id="empty"),
        pytest.param(b"[1, 2]", "panel manifest must be a JSON object", id="array"),
        pytest.param(b"42", "panel manifest must be a JSON object", id="scalar"),
        pytest.param(b"null", "panel manifest must be a JSON object", id="null"),
        pytest.param(
            b'{"zones": ["zone_a"], "zones": ["zone_b"]}', "duplicate key", id="duplicate-json-key"
        ),
        pytest.param(
            b'{"configuration": {}, "x": ' + _PAST_THE_DIGIT_LIMIT.encode() + b"}",
            "a number has more than",
            id="past-the-digit-limit",
        ),
        pytest.param(b'{"configuration": {"x": NaN}}', "non-finite", id="nan-literal"),
        pytest.param(b"[" * 200_000, "nested too deeply", id="deeply-nested"),
    ],
)
def test_cli_refuses_an_unusable_panel_manifest(tmp_path, capsys, content, message):
    """Every one of these is a routine input problem: the JSON refusal, never a traceback."""
    assert cli.main(_with_panel_manifest(tmp_path, content)) == 1
    refusal = _refusal(capsys)
    assert message in refusal["error"]
    assert not (tmp_path / "matrix.json").exists()


@pytest.mark.parametrize(
    "configuration",
    [
        pytest.param('{"panel": "x", "panel": "synthetic-14d"}', id="top-level"),
        pytest.param('{"panel": "synthetic-14d", "nested": {"a": 1, "a": 2}}', id="nested"),
    ],
)
def test_cli_refuses_a_repeated_key_in_the_configuration(tmp_path, capsys, configuration):
    """The manifest refuses a repeated key; `--configuration` must not keep the last one."""
    argv = _replace_flag(_fixture_argv(tmp_path), "--configuration", configuration)
    assert cli.main(argv) == 1
    error = _refusal(capsys)["error"]
    assert "--configuration" in error and "duplicate key" in error
    assert not (tmp_path / "matrix.json").exists()


def test_cli_refuses_an_identity_that_contradicts_the_manifest(tmp_path, capsys):
    argv = _replace_flag(_fixture_argv(tmp_path), "--model", "hermetic-v2")
    assert cli.main(argv) == 1
    assert "contradicts the panel manifest" in _refusal(capsys)["error"]
    assert not (tmp_path / "matrix.json").exists()


def test_cli_refuses_a_manifest_configuration_that_differs_from_the_flag(tmp_path, capsys):
    argv = _replace_flag(_fixture_argv(tmp_path), "--configuration", '{"panel": "another"}')
    assert cli.main(argv) == 1
    assert "contradicts the panel manifest" in _refusal(capsys)["error"]


def _forecast_lines() -> list[str]:
    return (FIXTURES / "forecast.jsonl").read_text(encoding="utf-8").splitlines()


def _with_forecast(tmp_path: Path, lines: list[str]) -> list[str]:
    forecast = tmp_path / "forecast_variant.jsonl"
    forecast.write_text("".join(line + "\n" for line in lines), encoding="utf-8")
    return _replace_flag(_fixture_argv(tmp_path), "--forecast", str(forecast))


def test_cli_refuses_a_duplicate_submitted_row(tmp_path, capsys):
    lines = _forecast_lines()
    assert cli.main(_with_forecast(tmp_path, [*lines, lines[0]])) == 1
    assert "duplicate prediction key" in _refusal(capsys)["error"]


def test_cli_refuses_a_submitted_row_outside_the_panel(tmp_path, capsys):
    outside = json.loads(_forecast_lines()[0])
    outside.update(zone_id="zone_z")
    assert cli.main(_with_forecast(tmp_path, [*_forecast_lines(), json.dumps(outside)])) == 1
    assert "outside the declared panel" in _refusal(capsys)["error"]
    assert not (tmp_path / "matrix.json").exists()


def test_cli_a_short_archive_keeps_the_expected_denominator(tmp_path):
    """Half the rows submitted: fewer scored pairs, the SAME 252 expected."""
    lines = _forecast_lines()
    argv = _with_forecast(tmp_path, lines[: len(lines) // 2])
    census = cli.run(cli.build_parser().parse_args(argv))["census"]
    assert census["expected_pairs"] == 252
    assert census["submitted_forecast_rows"] == len(lines) // 2
    assert census["scored_pairs"] < 203
    assert census["missing_pairs"] == census["expected_pairs"] - census["scored_pairs"]


def test_cli_refuses_a_manifest_window_that_the_flags_contradict(tmp_path, capsys):
    argv = _replace_flag(_fixture_argv(tmp_path), "--holdout-end", "2021-02-06")
    assert cli.main(argv) == 1
    assert "outside the holdout" in _refusal(capsys)["error"]


def test_cli_refuses_a_forecast_value_beyond_the_float_range(tmp_path, capsys):
    """``"value": 10**400`` in the forecast archive: the JSON refusal, not a traceback (#221).

    ``json.dumps`` writes that integer as a valid literal; ``float()`` of it raises
    ``OverflowError``, which ``main`` does not catch.
    """
    rows = [
        json.loads(line)
        for line in (FIXTURES / "forecast.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    rows[-1]["value"] = 10**400
    forecast = tmp_path / "forecast.jsonl"
    forecast.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    argv = _fixture_argv(tmp_path)
    argv[argv.index("--forecast") + 1] = str(forecast)
    assert cli.main(argv) == 1
    refusal = json.loads(capsys.readouterr().out)
    assert refusal["status"] == "refused"
    assert "prediction value must be numeric" in refusal["error"]
    assert not (tmp_path / "matrix.json").exists()


def test_cli_refuses_a_dirty_tree_without_the_override(tmp_path, monkeypatch):
    """An artifact that claims to be reproducible from its commit has to be."""
    monkeypatch.setattr(
        cli, "code_ref", lambda: {"repo": "deposium_geoai", "commit": "a", "dirty": True}
    )
    args = cli.build_parser().parse_args(
        [
            "--forecast",
            str(FIXTURES / "forecast.jsonl"),
            "--reference",
            str(FIXTURES / "reference.jsonl"),
            "--baseline-p50",
            str(FIXTURES / "baseline_p50.json"),
            "--panel-manifest",
            str(FIXTURES / "panel_manifest.json"),
            "--provider",
            "synthetic",
            "--model",
            "hermetic-v1",
            "--training-end",
            "2021-01-24",
            "--holdout-start",
            "2021-01-25",
            "--holdout-end",
            "2021-02-07",
            "--output",
            str(tmp_path / "matrix.json"),
        ]
    )
    with pytest.raises(BacktestValidationError, match="dirty"):
        cli.run(args)
    assert not (tmp_path / "matrix.json").exists()


def test_cli_refuses_an_unconfirmed_clean_checkout(tmp_path, monkeypatch, capsys):
    """``dirty: None`` (git unavailable) must refuse exactly like ``dirty: True`` does."""
    monkeypatch.setattr(
        cli, "code_ref", lambda: {"repo": "deposium_geoai", "commit": "a", "dirty": None}
    )
    argv = _fixture_argv(tmp_path)
    argv.remove("--allow-dirty")
    assert cli.main(argv) == 1
    assert json.loads(capsys.readouterr().out)["status"] == "refused"
    assert cli.main(_fixture_argv(tmp_path)) == 0


def test_cli_main_prints_the_census_beside_the_digest(tmp_path, capsys):
    """The one line a CI log keeps has to carry the counts, not just a verdict."""
    assert cli.main(_fixture_argv(tmp_path)) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["result_digest"] == FIXTURE_DIGEST
    assert summary["status"] == "completed"
    assert summary["expected_pairs"] == summary["scored_pairs"] + summary["missing_pairs"]
    assert summary["eligible_pairs"] <= summary["expected_pairs"]


def _with_baseline(tmp_path: Path, path: Path) -> dict:
    argv = _fixture_argv(tmp_path)
    argv[argv.index("--baseline-p50") + 1] = str(path)
    return cli.run(cli.build_parser().parse_args(argv))


def test_cli_reads_a_parquet_baseline_identically(tmp_path):
    """The real panel's normals are Parquet; the hermetic proof's are JSON.

    They have to produce the same cells, or the fixture proof says nothing about the run it
    stands in for. The ``result_digest`` legitimately differs: it covers the input digests,
    and the Parquet bytes are not the JSON bytes. The numbers must not.
    """
    duckdb = pytest.importorskip("duckdb")
    records = json.loads((FIXTURES / "baseline_p50.json").read_text(encoding="utf-8"))
    parquet = tmp_path / "baseline_p50.parquet"
    connection = duckdb.connect()
    try:
        connection.execute(
            "create table normals (zone_id varchar, variable varchar, month integer, p50 double)"
        )
        connection.executemany(
            "insert into normals values (?, ?, ?, ?)",
            [(item["zone_id"], item["variable"], item["month"], item["p50"]) for item in records],
        )
        connection.execute("copy normals to ? (format parquet)", [str(parquet)])
    finally:
        connection.close()

    from_parquet = _with_baseline(tmp_path, parquet)
    expected = json.loads((FIXTURES / "expected_matrix.json").read_text(encoding="utf-8"))
    for section in ("cells", "variable_aggregates", "census"):
        assert from_parquet[section] == expected[section]
    assert from_parquet["inputs"]["baseline_p50"] != expected["inputs"]["baseline_p50"]
    assert from_parquet["result_digest"] != expected["result_digest"]


@pytest.mark.parametrize(
    ("name", "content", "message"),
    [
        ("absent.json", None, "cannot read"),
        ("bad.jsonl", '{"zone_id": "a"}\n{ not json\n', "invalid JSONL"),
        ("scalar.json", "42", "JSON array, a rows wrapper, or JSONL"),
        ("empty.json", "   \n\n", "JSON array, a rows wrapper, or JSONL"),
        ("binary.json", None, "not UTF-8 text"),
        ("notparquet.parquet", "definitely not parquet", "monthly P50 table"),
        pytest.param(
            "overflow.json",
            json.dumps([_baseline(p50=10**400)]),
            "p50 must be numeric or null",
            id="p50-beyond-the-float-range",
        ),
        pytest.param(
            "huge.json",
            '[{"zone_id": "zone_a", "variable": "temperature_2m", "month": 1, "p50": '
            + _PAST_THE_DIGIT_LIMIT
            + "}]",
            "huge.json: a number has more than 4300 digits",
            id="array-past-the-digit-limit",
        ),
        pytest.param(
            "huge.jsonl",
            json.dumps(_baseline())
            + '\n{"zone_id": "zone_b", "variable": "temperature_2m", "month": 1, "p50": '
            + _PAST_THE_DIGIT_LIMIT
            + "}\n",
            "at line 2: a number has more than 4300 digits",
            id="jsonl-past-the-digit-limit",
        ),
    ],
)
def test_cli_refuses_an_unusable_baseline_file(tmp_path, capsys, name, content, message):
    """Every one of these is a routine input problem and must arrive as the JSON refusal.

    ``main`` catches ``BacktestValidationError`` and nothing else, so a DuckDB error or a
    decode error escaping as a traceback would leave a caller with neither a result nor a
    refusal to parse.
    """
    path = tmp_path / name
    if content is not None:
        path.write_text(content, encoding="utf-8")
    elif name == "binary.json":
        path.write_bytes(b"\xff\xfe\x00not utf-8")
    argv = _fixture_argv(tmp_path)
    argv[argv.index("--baseline-p50") + 1] = str(path)
    assert cli.main(argv) == 1
    assert message in json.loads(capsys.readouterr().out)["error"]


def test_cli_reads_a_jsonl_baseline(tmp_path):
    """JSONL is what the forecast and reference archives are; the baseline must not differ.

    An input that alone needed reshaping into an array first is the asymmetry that ends in
    a hand-edited file next to the two that did not.
    """
    records = json.loads((FIXTURES / "baseline_p50.json").read_text(encoding="utf-8"))
    lines = tmp_path / "baseline_p50.jsonl"
    lines.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )
    expected = json.loads((FIXTURES / "expected_matrix.json").read_text(encoding="utf-8"))
    assert _with_baseline(tmp_path, lines)["cells"] == expected["cells"]


def test_cli_accepts_the_rows_wrapper_form(tmp_path):
    """``{"rows": [...]}`` is how the other archives in this repo are written."""
    records = json.loads((FIXTURES / "baseline_p50.json").read_text(encoding="utf-8"))
    wrapped = tmp_path / "wrapped.json"
    wrapped.write_text(json.dumps({"rows": records}), encoding="utf-8")
    expected = json.loads((FIXTURES / "expected_matrix.json").read_text(encoding="utf-8"))
    assert _with_baseline(tmp_path, wrapped)["cells"] == expected["cells"]


def test_code_ref_falls_back_to_the_deployment_environment(monkeypatch):
    """A container image has no ``.git``; the platform injects the commit it was built from.

    ``dirty`` stays ``None`` there, because nothing observed a tree. Reporting it as clean
    would be the only lie this record can tell.
    """
    from src.utils import code_ref as module

    monkeypatch.setattr(module, "_git", lambda *args, **kwargs: None)
    monkeypatch.setenv("GIT_COMMIT", "deadbeef")
    monkeypatch.setenv("GIT_BRANCH", "main")
    assert module.code_ref() == {
        "repo": "deposium_geoai",
        "commit": "deadbeef",
        "branch": "main",
        "dirty": None,
    }


def test_code_ref_reads_the_local_checkout():
    from src.utils.code_ref import code_ref as read

    reference = read(cwd=Path(__file__).parent)
    assert reference["repo"] == "deposium_geoai"
    assert reference["dirty"] in (True, False)
    assert len(reference["commit"]) == 40


def test_code_ref_ignores_the_process_working_directory(tmp_path, monkeypatch):
    """The default target is this checkout, located from the module's own path — not cwd."""
    monkeypatch.chdir(tmp_path)
    from src.utils.code_ref import code_ref as read

    head = subprocess.run(
        ["git", "-C", str(_REPO_ROOT), "rev-parse", "HEAD"],
        text=True,
        capture_output=True,
        check=True,
    ).stdout.strip()
    assert read()["commit"] == head


def test_code_ref_does_not_trust_a_foreign_repository(tmp_path, monkeypatch):
    """Pointed at some OTHER git repo, it must not report that repo's commit as ours."""
    monkeypatch.delenv("RAILWAY_GIT_COMMIT_SHA", raising=False)
    monkeypatch.delenv("GIT_COMMIT", raising=False)
    monkeypatch.delenv("SOURCE_VERSION", raising=False)
    foreign = tmp_path / "foreign"
    foreign.mkdir()
    subprocess.run(["git", "init"], cwd=foreign, check=True, capture_output=True)
    (foreign / "f.txt").write_text("x", encoding="utf-8")
    subprocess.run(["git", "add", "f.txt"], cwd=foreign, check=True, capture_output=True)
    subprocess.run(
        ["git", "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-m", "x"],
        cwd=foreign,
        check=True,
        capture_output=True,
    )
    from src.utils.code_ref import code_ref as read

    reference = read(cwd=foreign)
    assert reference["commit"] is None
    assert reference["dirty"] is None


def test_baseline_nan_from_a_numeric_source_is_the_same_state_as_null():
    """Parquet delivers NaN, strict JSON delivers null; the matrix must not tell them apart."""
    from_nan = MonthlyP50Baseline.from_records([_baseline(p50=math.nan)])
    from_null = MonthlyP50Baseline.from_records([_baseline(p50=None)])
    assert math.isnan(from_nan.lookup("zone_a", "temperature_2m", 1))
    assert math.isnan(from_null.lookup("zone_a", "temperature_2m", 1))


# ---------------------------------------------------------------------------------------
# Create-only publication and the receipt (#193).
#
# `result_digest` says "same scientific result" and ignores code_ref on purpose; the
# receipt's `matrix_sha256` says "these exact bytes". A published matrix is never
# truncated or replaced, so a rerun, a changed input or configuration, or a different
# commit publishes to a NEW path.
# ---------------------------------------------------------------------------------------


@pytest.fixture
def no_network(monkeypatch):
    """The CLI scores an archive; any provider or socket use is a bug."""

    def _forbidden(*_args, **_kwargs):
        raise AssertionError("network access attempted")

    monkeypatch.setattr(socket.socket, "connect", _forbidden)
    monkeypatch.setattr(httpx.Client, "send", _forbidden)
    monkeypatch.setattr(httpx.AsyncClient, "send", _forbidden)


def _receipt_path(tmp_path: Path) -> Path:
    return tmp_path / "matrix.json.receipt.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _published(tmp_path: Path) -> tuple[Path, Path]:
    return tmp_path / "matrix.json", _receipt_path(tmp_path)


def _snapshot(*paths: Path) -> dict[Path, bytes]:
    return {path: path.read_bytes() for path in paths}


def _replace_flag(argv: list[str], flag: str, value: str) -> list[str]:
    argv[argv.index(flag) + 1] = value
    return argv


def _refusal(capsys) -> dict:
    refusal = json.loads(capsys.readouterr().out)
    assert refusal["status"] == "refused"
    return refusal


def test_publication_writes_the_matrix_and_a_receipt_that_verifies(tmp_path, no_network):
    assert cli.main(_fixture_argv(tmp_path)) == 0
    matrix_path, receipt_path = _published(tmp_path)
    matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))

    assert receipt["schema_version"] == "forecast-skill-matrix-receipt/v1"
    assert receipt["matrix_file"] == "matrix.json"
    # The exact bytes on disk, not a re-serialisation of the parsed matrix.
    assert receipt["matrix_sha256"] == _sha256(matrix_path)
    assert receipt["matrix_bytes"] == matrix_path.stat().st_size
    # The scientific identity is unchanged and is a different thing from the byte hash.
    assert receipt["result_digest"] == matrix["result_digest"] == FIXTURE_DIGEST
    assert receipt["matrix_sha256"] != receipt["result_digest"]
    assert receipt["inputs"] == {
        "forecast": _sha256(FIXTURES / "forecast.jsonl"),
        "reference": _sha256(FIXTURES / "reference.jsonl"),
        "baseline_p50": _sha256(FIXTURES / "baseline_p50.json"),
        "panel_manifest": _sha256(FIXTURES / "panel_manifest.json"),
    }
    assert receipt["inputs"]["panel_manifest"] == matrix["inputs"]["panel_manifest"]
    assert receipt["inputs"] == matrix["inputs"]
    assert receipt["frozen_configuration"] == {
        "provider": "synthetic",
        "model": "hermetic-v1",
        "configuration": {"panel": "synthetic-14d"},
    }
    assert receipt["code_ref"] == matrix["code_ref"]
    assert receipt["code_ref"]["repo"] == "deposium_geoai"


def test_rerun_to_the_same_target_refuses_and_changes_nothing(tmp_path, capsys, no_network):
    assert cli.main(_fixture_argv(tmp_path)) == 0
    capsys.readouterr()
    before = _snapshot(*_published(tmp_path))

    assert cli.main(_fixture_argv(tmp_path)) == 1
    assert "already exists" in _refusal(capsys)["error"]
    assert _snapshot(*_published(tmp_path)) == before


@pytest.mark.parametrize(
    "change",
    [
        pytest.param(("--configuration", '{"panel": "another"}'), id="configuration"),
        pytest.param(("--model", "hermetic-v2"), id="model"),
        pytest.param(("--holdout-end", "2021-02-06"), id="window"),
    ],
)
def test_a_changed_configuration_cannot_reuse_the_target(tmp_path, capsys, change):
    assert cli.main(_fixture_argv(tmp_path)) == 0
    capsys.readouterr()
    before = _snapshot(*_published(tmp_path))

    assert cli.main(_fixture_argv(tmp_path, *change)) == 1
    _refusal(capsys)
    assert _snapshot(*_published(tmp_path)) == before


def test_a_changed_input_cannot_reuse_the_target(tmp_path, capsys):
    assert cli.main(_fixture_argv(tmp_path)) == 0
    capsys.readouterr()
    before = _snapshot(*_published(tmp_path))
    baseline = tmp_path / "baseline_p50.json"
    baseline.write_bytes((FIXTURES / "baseline_p50.json").read_bytes() + b"\n")

    argv = _replace_flag(_fixture_argv(tmp_path), "--baseline-p50", str(baseline))
    assert cli.main(argv) == 1
    _refusal(capsys)
    assert _snapshot(*_published(tmp_path)) == before


def test_a_different_commit_cannot_reuse_the_target(tmp_path, capsys, monkeypatch):
    """Same scientific result, later commit: same `result_digest`, but a NEW path."""
    assert cli.main(_fixture_argv(tmp_path)) == 0
    capsys.readouterr()
    matrix_path, _ = _published(tmp_path)
    before = _snapshot(*_published(tmp_path))
    first = json.loads(matrix_path.read_text(encoding="utf-8"))

    later = {"repo": "deposium_geoai", "commit": "f" * 40, "branch": "main", "dirty": False}
    monkeypatch.setattr(cli, "code_ref", lambda: later)
    assert cli.main(_fixture_argv(tmp_path)) == 1
    _refusal(capsys)
    assert _snapshot(*_published(tmp_path)) == before

    # The retry path: a new target. Identical result, different bytes and receipt.
    new = tmp_path / "later"
    argv = _replace_flag(_fixture_argv(tmp_path), "--output", str(new / "matrix.json"))
    assert cli.main(argv) == 0
    second = json.loads((new / "matrix.json").read_text(encoding="utf-8"))
    receipt = json.loads((new / "matrix.json.receipt.json").read_text(encoding="utf-8"))
    assert second["result_digest"] == first["result_digest"] == FIXTURE_DIGEST
    assert receipt["code_ref"]["commit"] == "f" * 40
    assert receipt["matrix_sha256"] == _sha256(new / "matrix.json")
    assert receipt["matrix_sha256"] != _sha256(matrix_path)


@pytest.mark.parametrize("dangling", [True, False], ids=["dangling", "live"])
def test_a_symlink_target_is_refused_and_not_followed(tmp_path, capsys, dangling):
    elsewhere = tmp_path / "elsewhere.json"
    if not dangling:
        elsewhere.write_text("accepted\n", encoding="utf-8")
    matrix_path, receipt_path = _published(tmp_path)
    matrix_path.symlink_to(elsewhere)

    assert cli.main(_fixture_argv(tmp_path)) == 1
    _refusal(capsys)
    assert matrix_path.is_symlink()
    assert not receipt_path.exists()
    if dangling:
        assert not elsewhere.exists()
    else:
        assert elsewhere.read_text(encoding="utf-8") == "accepted\n"


def test_a_symlink_receipt_is_refused_before_any_matrix_is_published(tmp_path, capsys):
    matrix_path, receipt_path = _published(tmp_path)
    receipt_path.symlink_to(tmp_path / "elsewhere.json")

    assert cli.main(_fixture_argv(tmp_path)) == 1
    _refusal(capsys)
    assert not matrix_path.exists()
    assert not (tmp_path / "elsewhere.json").exists()


def test_a_stray_receipt_blocks_publication_rather_than_being_replaced(tmp_path, capsys):
    matrix_path, receipt_path = _published(tmp_path)
    receipt_path.write_text("stray\n", encoding="utf-8")

    assert cli.main(_fixture_argv(tmp_path)) == 1
    _refusal(capsys)
    assert not matrix_path.exists()
    assert receipt_path.read_text(encoding="utf-8") == "stray\n"


def _copied_inputs(tmp_path: Path) -> dict[str, Path]:
    copies = {}
    for flag, name in (
        ("--forecast", "forecast.jsonl"),
        ("--reference", "reference.jsonl"),
        ("--baseline-p50", "baseline_p50.json"),
        ("--panel-manifest", "panel_manifest.json"),
    ):
        copies[flag] = tmp_path / name
        copies[flag].write_bytes((FIXTURES / name).read_bytes())
    return copies


_INPUT_FLAGS = ["--forecast", "--reference", "--baseline-p50", "--panel-manifest"]


@pytest.mark.parametrize("flag", _INPUT_FLAGS)
def test_an_output_path_naming_an_input_is_refused(tmp_path, capsys, flag):
    """`--output` is what gets written; aimed at an input it would replace the evidence."""
    inputs = _copied_inputs(tmp_path)
    argv = _fixture_argv(tmp_path)
    for name, path in inputs.items():
        _replace_flag(argv, name, str(path))
    _replace_flag(argv, "--output", str(inputs[flag]))
    before = inputs[flag].read_bytes()

    assert cli.main(argv) == 1
    assert "same file" in _refusal(capsys)["error"]
    assert inputs[flag].read_bytes() == before


@pytest.mark.parametrize("flag", _INPUT_FLAGS)
def test_a_receipt_path_naming_an_input_is_refused(tmp_path, capsys, flag):
    inputs = _copied_inputs(tmp_path)
    output = tmp_path / "x.json"
    inputs[flag] = Path(str(output) + ".receipt.json")
    argv = _fixture_argv(tmp_path)
    for name, path in inputs.items():
        _replace_flag(argv, name, str(path))
    inputs[flag].write_bytes(b"input\n")
    _replace_flag(argv, "--output", str(output))

    assert cli.main(argv) == 1
    assert "same file" in _refusal(capsys)["error"]
    assert inputs[flag].read_bytes() == b"input\n"
    assert not output.exists()


def test_an_input_reached_through_a_symlink_or_dot_segments_still_collides(tmp_path, capsys):
    inputs = _copied_inputs(tmp_path)
    alias = tmp_path / "alias.jsonl"
    alias.symlink_to(inputs["--forecast"])
    before = inputs["--forecast"].read_bytes()
    argv = _fixture_argv(tmp_path)
    for name, path in inputs.items():
        _replace_flag(argv, name, str(path))
    _replace_flag(argv, "--output", str(tmp_path / "sub" / ".." / "alias.jsonl"))

    assert cli.main(argv) == 1
    assert "same file" in _refusal(capsys)["error"]
    assert inputs["--forecast"].read_bytes() == before


def test_a_concurrent_writer_that_wins_the_race_is_not_overwritten(tmp_path, capsys, monkeypatch):
    """The target does not exist when the CLI starts, and does by the time it publishes."""
    real = cli.paired_skill_matrix
    matrix_path, receipt_path = _published(tmp_path)

    def _score_then_lose_the_race(*args, **kwargs):
        matrix_path.write_text("the other writer\n", encoding="utf-8")
        return real(*args, **kwargs)

    monkeypatch.setattr(cli, "paired_skill_matrix", _score_then_lose_the_race)
    assert cli.main(_fixture_argv(tmp_path)) == 1
    _refusal(capsys)
    assert matrix_path.read_text(encoding="utf-8") == "the other writer\n"
    assert not receipt_path.exists()
    assert [p.name for p in tmp_path.iterdir()] == ["matrix.json"]  # no temp file left behind


def test_exactly_one_of_many_simultaneous_writers_publishes(tmp_path):
    target = tmp_path / "out.json"
    start = threading.Barrier(8)
    outcomes: list[str] = []

    def _writer(index: int) -> None:
        start.wait()
        try:
            cli._publish_create_only(target, f"writer {index}\n".encode())
            outcomes.append("won")
        except FileExistsError:
            outcomes.append("lost")

    threads = [threading.Thread(target=_writer, args=(i,)) for i in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert sorted(outcomes) == ["lost"] * 7 + ["won"]
    assert target.read_text(encoding="utf-8").startswith("writer ")
    assert [p.name for p in tmp_path.iterdir()] == ["out.json"]


def test_a_receipt_that_cannot_be_written_leaves_the_matrix_and_says_so(
    tmp_path, capsys, monkeypatch
):
    """Matrix first, receipt second. If the receipt fails, the published matrix is kept.

    Deleting it would be a second, unreviewed mutation of a published path; the operator is
    told the matrix has no receipt and must publish again to a new path.
    """
    real = cli._publish_create_only
    matrix_path, receipt_path = _published(tmp_path)

    def _receipt_slot_taken(path: Path, data: bytes) -> None:
        if path == receipt_path:
            path.write_text("taken\n", encoding="utf-8")
        real(path, data)

    monkeypatch.setattr(cli, "_publish_create_only", _receipt_slot_taken)
    assert cli.main(_fixture_argv(tmp_path)) == 1
    outcome = json.loads(capsys.readouterr().out)
    assert outcome["status"] == "incomplete"
    assert outcome["matrix"] == str(matrix_path)
    assert "no receipt" in outcome["error"]
    published = json.loads(matrix_path.read_text(encoding="utf-8"))
    assert published["result_digest"] == FIXTURE_DIGEST
    assert receipt_path.read_text(encoding="utf-8") == "taken\n"


def test_the_cli_documents_the_two_digests_and_the_retry_path():
    text = " ".join(cli.__doc__.split())
    for phrase in ("result_digest", "matrix_sha256", ".receipt.json", "new path", "panel manifest"):
        assert phrase in text


def test_a_reformatted_manifest_changes_the_input_digest_not_the_cells(tmp_path):
    """The digest fingerprints the BYTES; the numbers only depend on what the bytes declare."""
    reformatted = tmp_path / "panel.json"
    reformatted.write_text(
        json.dumps(json.loads((FIXTURES / "panel_manifest.json").read_text(encoding="utf-8"))),
        encoding="utf-8",
    )
    argv = _replace_flag(_fixture_argv(tmp_path), "--panel-manifest", str(reformatted))
    matrix = cli.run(cli.build_parser().parse_args(argv))
    assert matrix["inputs"]["panel_manifest"] == _sha256(reformatted)
    assert matrix["inputs"]["panel_manifest"] != _sha256(FIXTURES / "panel_manifest.json")
    assert matrix["result_digest"] != FIXTURE_DIGEST
    expected = json.loads((FIXTURES / "expected_matrix.json").read_text(encoding="utf-8"))
    for section in ("cells", "variable_aggregates", "census"):
        assert matrix[section] == expected[section]
