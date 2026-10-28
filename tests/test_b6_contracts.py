"""B6 level-2 result contract: parameters stay explicit, confidence stays null (#30)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.contracts import (
    B6_EXCLUSION_REASONS,
    B6_FORMULA_ID,
    B6Factor,
    B6Level2Output,
    B6Params,
)


def _trace() -> dict:
    return {
        "formula_id": B6_FORMULA_ID,
        "formula_version": "1.0.0",
        "params": B6Params().model_dump(),
        "library": {
            "library_version": "v1",
            "manifest_sha256": "a" * 64,
            "json_sha256": "b" * 64,
            "parquet_sha256": "c" * 64,
            "schema_version": "1.0.0",
            "zone_id": "lez",
            "algorithm": "tgrad",
            "area_status": "unqualified",
            "qualification": "unqualified",
        },
        "forecast": {"sha256": "d" * 64},
        "effective_horizon_days": 16,
        "code_ref": None,
        "patterns": [],
        "computed_at": "2026-10-16T00:00:00Z",
    }


def _no_conclusion(**overrides) -> dict:
    value = {
        "zone_id": "lez",
        "algorithm": "tgrad",
        "status": "no_conclusion",
        "reason": "empty_library",
        "patterns": [],
        "n_patterns_read": 0,
        "n_eligible": 0,
        "exclusions": dict.fromkeys(B6_EXCLUSION_REASONS, 0),
        "trace": _trace(),
    }
    value.update(overrides)
    return value


def test_params_defaults_are_the_proposed_unratified_values():
    params = B6Params()
    assert params.model_dump() == {
        "min_folds": 5,
        "min_support": 0.5,
        "min_stability": 0.5,
        "halflife_days": 10.0,
        "max_patterns": 10,
    }


@pytest.mark.parametrize(
    "field, value",
    [
        ("min_folds", 0),
        ("min_support", 1.5),
        ("min_stability", -0.1),
        ("halflife_days", 0),
        ("max_patterns", 0),
        ("unknown", 1),
    ],
)
def test_params_refuse_out_of_bounds_or_unknown(field, value):
    with pytest.raises(ValidationError):
        B6Params(**{field: value})


def test_output_round_trips_and_keeps_confidence_and_probability_null():
    output = B6Level2Output.model_validate(_no_conclusion())
    dumped = output.model_dump(mode="json")
    assert dumped["confidence"] is None and dumped["event_probability"] is None
    assert B6Level2Output.model_validate(dumped) == output
    for field in ("confidence", "event_probability"):
        with pytest.raises(ValidationError):
            B6Level2Output.model_validate(_no_conclusion(**{field: 0.4}))


def test_output_requires_every_exclusion_count():
    partial = dict.fromkeys(B6_EXCLUSION_REASONS[:-1], 0)
    with pytest.raises(ValidationError, match="exclusion"):
        B6Level2Output.model_validate(_no_conclusion(exclusions=partial))


def test_status_and_reason_must_agree():
    with pytest.raises(ValidationError):
        B6Level2Output.model_validate(_no_conclusion(reason=None))
    with pytest.raises(ValidationError):
        B6Level2Output.model_validate(_no_conclusion(status="explained"))


def test_no_extra_probability_or_confidence_key_is_tolerated():
    with pytest.raises(ValidationError):
        B6Level2Output.model_validate(_no_conclusion(probability=0.7))
    with pytest.raises(ValidationError):
        B6Factor(name="support", value=0.5, derivation="x", confidence=0.8)


def test_results_cannot_be_given_a_confidence_after_construction():
    output = B6Level2Output.model_validate(_no_conclusion())
    with pytest.raises(ValidationError):
        output.confidence = 0.7
    with pytest.raises(ValidationError):
        output.model_copy(update={"event_probability": 0.9})
    with pytest.raises(ValidationError):
        output.trace.library.model_copy(update={"confidence": 0.9})
    assert output.model_copy() == output


def test_counts_must_add_up():
    with pytest.raises(ValidationError, match="patterns read"):
        B6Level2Output.model_validate(_no_conclusion(n_eligible=99))
    with pytest.raises(ValidationError, match="patterns read"):
        B6Level2Output.model_validate(_no_conclusion(n_patterns_read=3))


def test_factor_value_is_bounded():
    with pytest.raises(ValidationError):
        B6Factor(name="support", value=1.2, derivation="library support")


def test_negative_exclusion_counts_cannot_cancel_each_other():
    exclusions = dict.fromkeys(B6_EXCLUSION_REASONS, 0)
    exclusions.update(not_cross_validated=-1, zone_not_validated=1)
    with pytest.raises(ValidationError, match="greater than or equal to 0"):
        B6Level2Output.model_validate(_no_conclusion(exclusions=exclusions))


def test_exclusion_schema_requires_nonnegative_counts():
    schema = B6Level2Output.model_json_schema()
    assert schema["properties"]["exclusions"]["additionalProperties"]["minimum"] == 0
