"""Unit tests for the TGrad (temporal) output parser — pure, no so4gp needed.

Runs in the base runtime. The sample is REAL TGrad ``discover_tgp()``
output captured during API verification (FIERCE Palier 2), so the temporal
parser is pinned to the actual lag-bearing JSON-string shape so4gp emits:
a target gradual item followed by parenthesized, time-lagged items.
"""

import pytest

from src.patterns.so4gp_adapter import (
    _lag_to_days,
    _parse_temporal_items,
    parse_discover_output,
)

# Verbatim TGrad discover_tgp() output (a JSON *string*), trimmed to a
# representative subset covering days / weeks / months and single + multi
# lagged-item patterns. target_col=2 (ndvi).
REAL_TGRAD_OUTPUT = """{
    "Algorithm": "TGrad",
    "Minimum Representation": "0.50",
    "Target Column": "2",
    "Run-time": "1.688166 seconds",
    "Patterns": [
        ["ndvi+, (precipitation+) +7.0 days", "sup=0.889", "density=0.711", "avg_dev=3.625", "dispersion=1.9", "connect=1", "singularity_scr=0.756"],
        ["ndvi+, (temperature-) +7.0 days, (precipitation+) +7.0 days", "sup=0.889", "density=0.711", "avg_dev=3.625", "dispersion=1.9", "connect=1", "singularity_scr=0.756"],
        ["ndvi+, (precipitation+) +2.0 weeks", "sup=0.964", "density=0.6", "avg_dev=3.074", "dispersion=1.72", "connect=1", "singularity_scr=1.378"],
        ["ndvi+, (precipitation+) +1.0 months", "sup=0.9", "density=0.4", "avg_dev=2.9", "dispersion=1.4", "connect=1", "singularity_scr=1.6"]
    ]
}"""


class TestParseTemporalOutput:
    def test_top_level_fields(self):
        result = parse_discover_output(REAL_TGRAD_OUTPUT, is_temporal=True)
        assert result["algorithm"] == "TGrad"
        assert result["runtime_seconds"] == pytest.approx(1.688166)
        assert result["target_column"] == "2"
        assert result["min_representativity"] == pytest.approx(0.5)
        assert len(result["patterns"]) == 4

    def test_patterns_flagged_temporal(self):
        result = parse_discover_output(REAL_TGRAD_OUTPUT, is_temporal=True)
        assert all(p["is_temporal"] for p in result["patterns"])

    def test_target_item_first_with_zero_lag(self):
        result = parse_discover_output(REAL_TGRAD_OUTPUT, is_temporal=True)
        target = result["patterns"][0]["gradual_items"][0]
        assert target == {
            "attribute": "ndvi",
            "direction": "+",
            "role": "target",
            "lag_days": 0.0,
            "lag_raw": None,
        }

    def test_lagged_item_days(self):
        result = parse_discover_output(REAL_TGRAD_OUTPUT, is_temporal=True)
        lagged = result["patterns"][0]["gradual_items"][1]
        assert lagged == {
            "attribute": "precipitation",
            "direction": "+",
            "role": "lagged",
            "lag_days": pytest.approx(7.0),
            "lag_raw": "+7.0 days",
        }

    def test_weeks_and_months_normalized_to_days(self):
        result = parse_discover_output(REAL_TGRAD_OUTPUT, is_temporal=True)
        # pattern[2]: "+2.0 weeks" -> 14 days
        assert result["patterns"][2]["gradual_items"][1]["lag_days"] == pytest.approx(14.0)
        # pattern[3]: "+1.0 months" -> ~30.42 days (so4gp month constant)
        assert result["patterns"][3]["gradual_items"][1]["lag_days"] == pytest.approx(
            30.41666, abs=1e-3
        )

    def test_multi_lagged_items(self):
        result = parse_discover_output(REAL_TGRAD_OUTPUT, is_temporal=True)
        # "ndvi+, (temperature-) +7.0 days, (precipitation+) +7.0 days"
        items = result["patterns"][1]["gradual_items"]
        assert len(items) == 3  # 1 target + 2 lagged
        assert items[0]["role"] == "target"
        roles = {i["attribute"]: i["role"] for i in items}
        assert roles == {"ndvi": "target", "temperature": "lagged", "precipitation": "lagged"}
        assert items[1] == {
            "attribute": "temperature",
            "direction": "-",
            "role": "lagged",
            "lag_days": pytest.approx(7.0),
            "lag_raw": "+7.0 days",
        }


class TestTemporalItemEdgeCases:
    def test_negative_sign_lag(self):
        items = _parse_temporal_items("ndvi+, (precip-) -3.0 days")
        assert items[1]["lag_days"] == pytest.approx(-3.0)
        assert items[1]["lag_raw"] == "-3.0 days"

    def test_unknown_unit_yields_none(self):
        # Defensive: an unmapped unit must not fabricate a lag.
        items = _parse_temporal_items("ndvi+, (precip+) +5.0 fortnights")
        assert items[1]["lag_days"] is None
        assert items[1]["lag_raw"] == "+5.0 fortnights"

    def test_lag_to_days_units(self):
        assert _lag_to_days(1, "days") == pytest.approx(1.0)
        assert _lag_to_days(2, "weeks") == pytest.approx(14.0)
        assert _lag_to_days(24, "hours") == pytest.approx(1.0)
        assert _lag_to_days(1, "years") == pytest.approx(365.25)
        assert _lag_to_days(1, "parsecs") is None


class TestGraankBackwardCompatibility:
    def test_non_temporal_pattern_has_no_lag_keys(self):
        # A plain GRAANK pattern (no parens) keeps the legacy item shape.
        result = parse_discover_output(
            '{"Algorithm": "GRAANK", "Patterns": [["precip+, ndvi+", "sup=0.8"]]}'
        )
        pattern = result["patterns"][0]
        assert pattern["is_temporal"] is False
        assert pattern["gradual_items"] == [
            {"attribute": "precip", "direction": "+"},
            {"attribute": "ndvi", "direction": "+"},
        ]

    def test_paren_in_column_name_stays_gradual(self):
        # Regression: a GRAANK column name containing '(' (e.g. units) must NOT
        # be mis-parsed as temporal. With is_temporal=False both items survive;
        # the old "(" in pattern_str heuristic dropped "ndvi+" here.
        raw = '{"Algorithm": "GRAANK", "Patterns": [["co2(ppm)+, ndvi+", "sup=0.8"]]}'
        result = parse_discover_output(raw)  # default is_temporal=False
        pattern = result["patterns"][0]
        assert pattern["is_temporal"] is False
        assert pattern["gradual_items"] == [
            {"attribute": "co2(ppm)", "direction": "+"},
            {"attribute": "ndvi", "direction": "+"},
        ]


class TestTemporalTargetName:
    """The so4gp envelope 'Target Column' is a numeric INDEX (e.g. "2"), not a name;
    the real target name is threaded from the miner (``df.columns[target_col]``). The
    leading target token is split on the first top-level comma, so a name containing
    '(' survives (the original ``find("(")`` truncation bug)."""

    def test_threaded_name_used_not_envelope_index(self):
        # Envelope 'Target Column' is "2"; the threaded name must win — never "2".
        result = parse_discover_output(REAL_TGRAD_OUTPUT, is_temporal=True, target_name="ndvi")
        target = result["patterns"][0]["gradual_items"][0]
        assert target["attribute"] == "ndvi"
        assert target["role"] == "target"
        assert target["direction"] == "+"
        assert result["target_column"] == "2"  # index metadata preserved untouched

    def test_parenthesised_target_name_threaded(self):
        raw = (
            '{"Algorithm": "TGrad", "Target Column": "3", '
            '"Patterns": [["co2(ppm)+, (precipitation+) +7.0 days", "sup=0.9"]]}'
        )
        result = parse_discover_output(raw, is_temporal=True, target_name="co2(ppm)")
        items = result["patterns"][0]["gradual_items"]
        assert items[0]["attribute"] == "co2(ppm)"
        assert items[0]["direction"] == "+"
        assert items[0]["role"] == "target"
        assert items[1]["attribute"] == "precipitation"  # lagged item still parsed

    def test_parenthesised_target_fallback_without_name(self):
        # Even with no threaded name, the top-level-comma boundary (not find("(")) keeps
        # a parenthesised target intact — this is the original bug.
        items = _parse_temporal_items("co2(ppm)+, (precipitation+) +7.0 days")
        assert items[0]["attribute"] == "co2(ppm)"
        assert items[0]["direction"] == "+"
        assert items[0]["role"] == "target"
        assert items[1]["attribute"] == "precipitation"

    def test_comma_inside_target_parens_not_truncated(self):
        # A comma INSIDE the target's parentheses must not split the token: the boundary
        # is the first TOP-LEVEL comma, so name + direction both survive (Greptile P2).
        items = _parse_temporal_items("co2(ppm,avg)+, (precipitation+) +7.0 days")
        assert items[0]["attribute"] == "co2(ppm,avg)"
        assert items[0]["direction"] == "+"  # direction not lost to truncation
        assert items[0]["role"] == "target"
        assert items[1]["attribute"] == "precipitation"
        items_threaded = _parse_temporal_items(
            "co2(ppm,avg)+, (precipitation+) +7.0 days", target_name="co2(ppm,avg)"
        )
        assert items_threaded[0]["attribute"] == "co2(ppm,avg)"
        assert items_threaded[0]["direction"] == "+"
