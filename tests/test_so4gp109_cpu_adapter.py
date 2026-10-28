"""Hermetic contracts for the legacy and public CPU adapter invocation paths."""

import json
import sys
from types import ModuleType, SimpleNamespace

import pandas as pd
import pytest

from src.patterns.so4gp_adapter import mine_gradual_patterns, mine_temporal_patterns


@pytest.mark.parametrize("temporal", [False, True])
@pytest.mark.parametrize("legacy", [False, True])
@pytest.mark.parametrize("error", [False, True])
def test_raw_capture_precedes_parsing_without_changing_result(
    fake_so4gp, frame, temporal, legacy, error
):
    raw = (
        '{ "Error": "native failure" }\n'
        if error
        else '{ "Algorithm": "fixture", "Patterns": [] }\n'
    )

    class Miner:
        def __init__(self, **kwargs):
            pass

        def discover(self, **kwargs):
            return raw

    if temporal and legacy:
        Miner.discover_tgp = Miner.discover
    setattr(
        fake_so4gp, "TGrad" if temporal and legacy else "TGRAANK" if temporal else "GRAANK", Miner
    )
    # Exercise the public GRAANK signature detection as well as legacy kwargs.
    if not temporal and not legacy:

        def public_constructor(self, *, data_source, min_sup, eq, device):
            assert device == "cpu"

        Miner.__init__ = public_constructor
    mine = mine_temporal_patterns if temporal else mine_gradual_patterns
    kwargs = {"target_col": 2} if temporal else {}
    captured = []
    if error:
        with pytest.raises(ValueError, match="native failure"):
            mine(frame, **kwargs, capture_raw_output=captured.append)
    else:
        expected = mine(frame, **kwargs)
        actual = mine(frame, **kwargs, capture_raw_output=captured.append)
        assert actual == expected
    assert captured == [raw]


@pytest.fixture
def fake_so4gp(monkeypatch):
    package = ModuleType("so4gp")
    package.algorithms = SimpleNamespace()
    monkeypatch.setitem(sys.modules, "so4gp", package)
    return package.algorithms


@pytest.fixture
def frame():
    # The public temporal adapter also validates the physical date-time axis.
    # pandas is a base dependency; the so4gp algorithms remain entirely fake.
    return pd.DataFrame(
        {"date": ["2020-01-01", "2020-01-08"], "precip": [10, 20], "ndvi": [0.3, 0.4]}
    )


@pytest.mark.parametrize("parallel", [False, True])
def test_legacy_temporal_preferred_and_invocation_preserved(fake_so4gp, frame, parallel):
    calls = []

    class LegacyTGrad:
        def __init__(self, **kwargs):
            calls.append(kwargs)

        def discover_tgp(self, **kwargs):
            calls.append(kwargs)
            return json.dumps({"Algorithm": "TGrad", "Patterns": []})

    fake_so4gp.TGrad = LegacyTGrad
    # If both APIs exist, the legacy class must win.
    fake_so4gp.TGRAANK = object
    result = mine_temporal_patterns(
        frame,
        target_col=2,
        min_support=0.6,
        min_rep=0.7,
        encode_equal=True,
        parallel=parallel,
    )
    assert calls == [
        {"data_source": frame, "min_sup": 0.6, "eq": True, "target_col": 2, "min_rep": 0.7},
        {"parallel": parallel},
    ]
    assert result["algorithm"] == "TGrad"
    assert result["algorithm_key"] == "tgrad"


def test_public_temporal_cpu_apriori_all_transformations(fake_so4gp, frame):
    calls = []

    class PublicTGRAANK:
        def __init__(self, *, data_source, min_sup, min_rep, eq, device):
            calls.append((data_source, min_sup, min_rep, eq, device))

        def discover(self, *, target_col, transformations, search_algorithm, save_results):
            calls.append((target_col, transformations, search_algorithm, save_results))
            return json.dumps(
                {
                    "Algorithm": "TGRAANK",
                    "Target Column": "2",
                    "Patterns": [["ndvi+, (precip+) +7.0 days", "sup=0.8, conf=0.9"]],
                }
            )

    fake_so4gp.TGRAANK = PublicTGRAANK
    result = mine_temporal_patterns(
        frame,
        target_col=2,
        min_support=0.6,
        min_rep=0.7,
        encode_equal=True,
        algorithm="TGRAD",
    )
    assert calls == [(frame, 0.6, 0.7, True, "cpu"), (2, "all", "apriori", False)]
    pattern = result["patterns"][0]
    assert pattern["is_temporal"] is True
    assert pattern["support"] == 0.8
    assert pattern["metrics"]["conf"] == 0.9
    assert pattern["gradual_items"][0]["attribute"] == "ndvi"
    assert pattern["gradual_items"][1]["lag_days"] == 7.0
    assert result["algorithm_key"] == "TGRAD"
    assert result["target_col"] == 2


def test_public_temporal_parallel_rejected_before_constructor(fake_so4gp, frame):
    class PublicTGRAANK:
        def __init__(self, **kwargs):
            pytest.fail("unsupported parallel request should fail before mining")

    fake_so4gp.TGRAANK = PublicTGRAANK
    with pytest.raises(ValueError, match="parallel=True.*TGRAANK CPU"):
        mine_temporal_patterns(frame, target_col=2, parallel=True)


def test_public_temporal_error_envelope_is_not_empty_success(fake_so4gp, frame):
    class PublicTGRAANK:
        def __init__(self, **kwargs):
            pass

        def discover(self, **kwargs):
            return json.dumps({"Error": "No date-time datasets found"})

    fake_so4gp.TGRAANK = PublicTGRAANK
    with pytest.raises(ValueError, match="No date-time datasets found"):
        mine_temporal_patterns(frame, target_col=2)


@pytest.mark.parametrize("dates", [["invalid", "invalid"], ["2020-01-01", None], ["0", "1"]])
def test_public_temporal_refuses_missing_clock_before_constructor(fake_so4gp, frame, dates):
    class PublicTGRAANK:
        def __init__(self, **kwargs):
            pytest.fail("invalid dates must be refused before the upstream cleaner")

    fake_so4gp.TGRAANK = PublicTGRAANK
    frame["date"] = dates
    with pytest.raises(ValueError, match="complete, parseable date-time"):
        mine_temporal_patterns(frame, target_col=2)


@pytest.mark.parametrize("pattern", ["ndvi+, (precip+) +1.0 lag", "ndvi+, precip+"])
def test_temporal_refuses_nonphysical_lag_and_keeps_raw_capture(fake_so4gp, frame, pattern):
    raw = json.dumps({"Algorithm": "TGrad", "Patterns": [[pattern, "sup=0.8"]]})

    class PublicTGRAANK:
        def __init__(self, **kwargs):
            pass

        def discover(self, **kwargs):
            return raw

    fake_so4gp.TGRAANK = PublicTGRAANK
    captured = []
    with pytest.raises(ValueError, match="supported physical time unit"):
        mine_temporal_patterns(frame, target_col=2, capture_raw_output=captured.append)
    assert captured == [raw]


def test_public_graank_cpu_stays_unrestricted_and_keeps_native_conf(fake_so4gp, frame):
    calls = []

    class PublicGRAANK:
        def __init__(self, *, data_source, min_sup, eq, device):
            calls.append((data_source, min_sup, eq, device))

        def discover(self, *, target_col=None):
            calls.append(target_col)
            # A pattern without NDVI proves it must not be a required target.
            return json.dumps(
                {
                    "Algorithm": "GRAANK",
                    "Patterns": [
                        ["precip+, temp-", "sup=0.8, conf=0.875"],
                    ],
                }
            )

    fake_so4gp.GRAANK = PublicGRAANK
    result = mine_gradual_patterns(frame, min_support=0.6, encode_equal=True)
    assert calls == [(frame, 0.6, True, "cpu"), None]
    pattern = result["patterns"][0]
    assert pattern["support"] == 0.8
    assert pattern["metrics"]["conf"] == 0.875
    assert pattern["gradual_items"] == [
        {"attribute": "precip", "direction": "+"},
        {"attribute": "temp", "direction": "-"},
    ]


def test_legacy_graank_kwargs_preserved(fake_so4gp, frame):
    calls = []

    class LegacyGRAANK:
        def __init__(self, *, data_source, min_sup, eq):
            calls.append((data_source, min_sup, eq))

        def discover(self):
            calls.append("discover")
            return json.dumps({"Algorithm": "GRAANK", "Patterns": []})

    fake_so4gp.GRAANK = LegacyGRAANK
    mine_gradual_patterns(frame, min_support=0.6, encode_equal=True)
    assert calls == [(frame, 0.6, True), "discover"]
