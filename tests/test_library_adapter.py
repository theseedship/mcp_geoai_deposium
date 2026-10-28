"""Tests for the producer→exchange adapter (FIERCE DoD O2, issue #20).

Pure — feeds normalized so4gp dicts (the shape `parse_discover_output` emits), so it runs
in the base runtime without so4gp installed.
"""

from __future__ import annotations

import json

import pytest

from src.patterns.library_adapter import (
    library_from_mining_result,
    record_from_normalized,
)
from src.patterns.render import render_gradual_item
from src.patterns.so4gp_adapter import parse_discover_output

GRAANK_PAT = {
    "gradual_items": [
        {"attribute": "precipitation", "direction": "+"},
        {"attribute": "ndvi", "direction": "+"},
    ],
    "support": 0.82,
    "metrics": {"density": 0.82, "singularity_scr": 0.12, "avg_dev": "2.9"},
    "is_temporal": False,
}


@pytest.mark.parametrize("confidence", [None, 0.0, 0.9])
def test_native_confidence_survives_strict_library_without_calibration(confidence):
    metric_token = "sup=0.8"
    if confidence is not None:
        metric_token += f", conf={confidence}"
    raw = json.dumps({"Patterns": [["precipitation+, ndvi+", metric_token]]})
    result = parse_discover_output(raw)
    library = library_from_mining_result(result, zone_id="z", algorithm="graank", strict=True)
    assert len(library.patterns) == 1
    record = library.patterns[0]
    assert record.support == 0.8
    assert record.extra_metrics == (None if confidence is None else {"conf": confidence})
    assert record.confidence is None
    assert record.n_folds is None
    assert record.zones_validated == []


TGRAD_PAT = {
    "gradual_items": [
        {"attribute": "ndvi", "direction": "+", "role": "target", "lag_days": 0.0, "lag_raw": None},
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
    "is_temporal": True,
}


def test_graank_record_maps_itemset_to_antecedent():
    r = record_from_normalized(GRAANK_PAT, pattern_id=0, algorithm="graank", zone_id="montpellier")
    assert r.algorithm == "graank" and r.zone_id == "montpellier"
    assert r.is_temporal is False and r.lag_days == 0.0
    assert r.antecedent == "precipitation+, ndvi+" and r.consequent == ""
    assert r.variables == ["precipitation", "ndvi"]
    assert r.support == 0.82
    assert r.confidence is None  # never fabricated at paliers 1-2
    # only numeric so4gp measures survive as extra_metrics; the string token is dropped
    assert r.extra_metrics == {"density": 0.82, "singularity_scr": 0.12}


def test_tgrad_record_splits_target_consequent_and_signed_lag():
    r = record_from_normalized(TGRAD_PAT, pattern_id=3, algorithm="tgrad", zone_id="rennes")
    assert r.pattern_id == 3 and r.algorithm == "tgrad"
    assert r.is_temporal is True and r.lag_days == 7.0
    assert r.consequent == "ndvi+"
    assert r.antecedent == "(precipitation+) +7.0 days"
    assert r.variables == ["ndvi", "precipitation"]
    assert r.extra_metrics is None  # no numeric metrics in this sample


def test_dominant_signed_lag_keeps_sign_of_largest_abs():
    pat = {
        "is_temporal": True,
        "support": 0.5,
        "metrics": {},
        "gradual_items": [
            {"attribute": "ndvi", "direction": "+", "role": "target", "lag_days": 0.0},
            {
                "attribute": "a",
                "direction": "+",
                "role": "lagged",
                "lag_days": 7.0,
                "lag_raw": "+7.0 days",
            },
            {
                "attribute": "b",
                "direction": "-",
                "role": "lagged",
                "lag_days": -14.0,
                "lag_raw": "-14.0 days",
            },
        ],
    }
    r = record_from_normalized(pat, pattern_id=0, algorithm="tgrad", zone_id="z")
    assert r.lag_days == -14.0  # largest |lag|, sign preserved


def test_library_from_result_scopes_and_enumerates():
    result = {"patterns": [GRAANK_PAT, dict(GRAANK_PAT)]}
    lib = library_from_mining_result(result, zone_id="montpellier", algorithm="graank")
    assert lib.zone_id == "montpellier" and lib.algorithm == "graank"
    assert [p.pattern_id for p in lib.patterns] == [0, 1]
    assert all(p.zone_id == "montpellier" and p.algorithm == "graank" for p in lib.patterns)


def test_empty_result_yields_empty_library():
    lib = library_from_mining_result({"patterns": []}, zone_id="z", algorithm="graank")
    assert lib.patterns == []


def test_unmappable_pattern_is_skipped_not_fatal():
    # parse_discover_output legitimately emits support=None — it must not nuke the library.
    bad = {
        "gradual_items": [{"attribute": "x", "direction": "+"}],
        "support": None,
        "is_temporal": False,
        "metrics": {},
    }
    result = {"patterns": [GRAANK_PAT, bad, dict(GRAANK_PAT)]}
    lib = library_from_mining_result(result, zone_id="montpellier", algorithm="graank")
    assert len(lib.patterns) == 2  # the support=None pattern is skipped; valid ones survive
    # strict mode surfaces the error instead of skipping
    with pytest.raises(ValueError):
        library_from_mining_result(result, zone_id="montpellier", algorithm="graank", strict=True)


def test_empty_itemset_pattern_is_skipped():
    result = {
        "patterns": [{"gradual_items": [], "support": 0.5, "is_temporal": False, "metrics": {}}]
    }
    assert library_from_mining_result(result, zone_id="z", algorithm="graank").patterns == []


def test_nonfinite_metrics_dropped():
    pat = {
        **GRAANK_PAT,
        "metrics": {"density": float("inf"), "singularity_scr": 0.12, "x": float("nan")},
    }
    r = record_from_normalized(pat, pattern_id=0, algorithm="graank", zone_id="z")
    assert r.extra_metrics == {"singularity_scr": 0.12}  # inf / nan dropped


def test_signed_lag_tiebreak_is_deterministic():
    pat = {
        "is_temporal": True,
        "support": 0.5,
        "metrics": {},
        "gradual_items": [
            {"attribute": "t", "direction": "+", "role": "target", "lag_days": 0.0},
            {
                "attribute": "a",
                "direction": "+",
                "role": "lagged",
                "lag_days": 7.0,
                "lag_raw": "+7.0 days",
            },
            {
                "attribute": "b",
                "direction": "-",
                "role": "lagged",
                "lag_days": -7.0,
                "lag_raw": "-7.0 days",
            },
        ],
    }
    # equal |lag| → tie-break toward the larger signed value (+7), regardless of item order
    assert record_from_normalized(pat, pattern_id=0, algorithm="tgrad", zone_id="z").lag_days == 7.0


class TestSharedRendererStrictDirection:
    """R4 — the producer parquet renderer (patterns.py) and the exchange-record renderer
    (library_adapter.py) now share one strict ``render_gradual_item``. A normalized item
    always carries a 'direction' key (possibly ``None`` → '?'); a MISSING key is malformed
    and raises rather than silently rendering '?'. This changed library_adapter's old
    lenient ``.get('direction')`` → '?' path to fail-loud."""

    def test_renders_contemporaneous_and_lagged(self):
        assert render_gradual_item({"attribute": "ndvi", "direction": "+"}) == "ndvi+"
        assert (
            render_gradual_item(
                {"attribute": "precip", "direction": "+", "role": "lagged", "lag_raw": "+7.0 days"}
            )
            == "(precip+) +7.0 days"
        )

    def test_none_direction_renders_question_mark(self):
        # A present-but-None direction is the normal so4gp shape for an unparsed sign — kept.
        assert render_gradual_item({"attribute": "x", "direction": None}) == "x?"

    def test_missing_direction_key_raises(self):
        # The fail-loud contract: a malformed item with no 'direction' key must raise, not '?'.
        with pytest.raises(KeyError):
            render_gradual_item({"attribute": "x"})

    def test_record_from_normalized_raises_on_direction_less_item(self):
        # The exchange-record caller (was lenient '?') now fails loud on the malformed item.
        bad = {
            "gradual_items": [{"attribute": "x"}],  # no 'direction' key
            "support": 0.5,
            "is_temporal": False,
            "metrics": {},
        }
        with pytest.raises(KeyError):
            record_from_normalized(bad, pattern_id=0, algorithm="graank", zone_id="z")

    def test_library_skips_direction_less_item_nonstrict_and_raises_strict(self):
        # Observable behaviour change (R4): a direction-less item used to render '?' and be
        # INCLUDED; it is now skipped non-strict (KeyError caught) and raises under strict.
        bad = {
            "gradual_items": [{"attribute": "x"}],
            "support": 0.5,
            "is_temporal": False,
            "metrics": {},
        }
        result = {"patterns": [GRAANK_PAT, bad, dict(GRAANK_PAT)]}
        lib = library_from_mining_result(result, zone_id="z", algorithm="graank")
        assert len(lib.patterns) == 2  # malformed item skipped; valid ones survive
        with pytest.raises(KeyError):
            library_from_mining_result(result, zone_id="z", algorithm="graank", strict=True)
