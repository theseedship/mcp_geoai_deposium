"""Unit tests for the so4gp output parser — pure, no so4gp/pandas needed.

Runs in the base runtime. The sample is REAL GRAANK `Patterns` discover() output
captured during API verification, so the parser is pinned to the actual
JSON-string shape so4gp emits (not the README's implied dict).
"""

import json

import pytest

from src.patterns.so4gp_adapter import parse_discover_output

# Verbatim GRAANK discover() output (a JSON *string*).
REAL_DISCOVER_OUTPUT = """{
    "Algorithm": "GRAANK",
    "Run-time": "1.357852 seconds",
    "Patterns": [
        ["precip+, date_idx+", "sup=0.536", "density=0.536", "avg_dev=3.267", "dispersion=1.806", "connect=1", "singularity_scr=0.45"],
        ["date_idx+, temp+", "sup=0.714", "density=0.714", "avg_dev=3.35", "dispersion=1.824", "connect=1", "singularity_scr=0.55"],
        ["precip+, temp-, ndvi+", "sup=0.643", "density=0.643", "avg_dev=2.444", "dispersion=1.212", "connect=1", "singularity_scr=0.389"]
    ],
    "Invalid Count": "11"
}"""


class TestParseDiscoverOutput:
    def test_top_level_fields(self):
        result = parse_discover_output(REAL_DISCOVER_OUTPUT)
        assert result["algorithm"] == "GRAANK"
        assert result["runtime_seconds"] == pytest.approx(1.357852)
        assert result["invalid_count"] == 11
        assert len(result["patterns"]) == 3

    def test_support_extracted_as_float(self):
        result = parse_discover_output(REAL_DISCOVER_OUTPUT)
        supports = [p["support"] for p in result["patterns"]]
        assert supports == pytest.approx([0.536, 0.714, 0.643])

    def test_gradual_items_attribute_and_direction(self):
        result = parse_discover_output(REAL_DISCOVER_OUTPUT)
        # third pattern: "precip+, temp-, ndvi+"
        items = result["patterns"][2]["gradual_items"]
        assert items == [
            {"attribute": "precip", "direction": "+"},
            {"attribute": "temp", "direction": "-"},
            {"attribute": "ndvi", "direction": "+"},
        ]

    def test_metrics_kept_without_support(self):
        result = parse_discover_output(REAL_DISCOVER_OUTPUT)
        metrics = result["patterns"][0]["metrics"]
        assert "sup" not in metrics  # support is hoisted out
        assert metrics["density"] == pytest.approx(0.536)
        assert metrics["singularity_scr"] == pytest.approx(0.45)
        assert metrics["connect"] == pytest.approx(1.0)

    def test_empty_patterns(self):
        result = parse_discover_output('{"Algorithm": "GRAANK", "Patterns": []}')
        assert result["patterns"] == []
        assert result["invalid_count"] is None
        assert result["runtime_seconds"] is None

    def test_attribute_without_direction_is_not_dropped(self):
        # Defensive: a token with no trailing +/- is kept with direction None
        result = parse_discover_output('{"Patterns": [["foo", "sup=0.9"]]}')
        items = result["patterns"][0]["gradual_items"]
        assert items == [{"attribute": "foo", "direction": None}]


@pytest.mark.parametrize("confidence", [0.0, 0.9])
def test_combined_metrics_match_separate_assignments(confidence):
    def parse(tokens):
        raw = json.dumps({"Patterns": [["precip+, ndvi+", *tokens]]})
        return parse_discover_output(raw)["patterns"][0]

    combined = parse([f"sup=0.8, conf={confidence}", "density=0.7"])
    separate = parse(["sup=0.8", f"conf={confidence}", "density=0.7"])
    assert combined == separate
    assert combined["support"] == 0.8
    assert combined["metrics"] == {"conf": confidence, "density": 0.7}


def test_missing_native_confidence_is_not_fabricated():
    pat = parse_discover_output('{"Patterns": [["precip+, ndvi+", "sup=0.8"]]}')["patterns"][0]
    assert "conf" not in pat["metrics"]


def test_commas_inside_string_metric_values_survive():
    raw = json.dumps({"Patterns": [["precip+, ndvi+", "sup=0.8, note=a,b, conf=0.9"]]})
    pat = parse_discover_output(raw)["patterns"][0]
    assert pat["support"] == 0.8
    assert pat["metrics"] == {"note": "a,b", "conf": 0.9}


def test_temporal_combined_metrics_preserve_signed_lag():
    raw = json.dumps({"Patterns": [["ndvi+, (precipitation-) -2.0 weeks", "sup=0.8, conf=0.9"]]})
    pat = parse_discover_output(raw, is_temporal=True)["patterns"][0]
    assert pat["support"] == 0.8
    assert pat["metrics"]["conf"] == 0.9
    assert pat["is_temporal"] is True
    assert pat["gradual_items"][1]["lag_days"] == -14.0
    assert pat["gradual_items"][1]["lag_raw"] == "-2.0 weeks"


@pytest.mark.parametrize("patterns", [None, [], [["precip+, ndvi+", "sup=0.8"]]])
def test_upstream_error_is_explicit_even_with_patterns(patterns):
    payload = {"Error": "invalid target column"}
    if patterns is not None:
        payload["Patterns"] = patterns
    with pytest.raises(ValueError, match="so4gp discovery failed: invalid target column"):
        parse_discover_output(json.dumps(payload))


@pytest.mark.parametrize("error", [None, ""])
def test_empty_patterns_with_no_error_are_successful(error):
    result = parse_discover_output(json.dumps({"Error": error, "Patterns": []}))
    assert result["patterns"] == []
