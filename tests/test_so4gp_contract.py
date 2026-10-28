"""Real installed so4gp 1.1.2 CPU contracts, not scientific acceptance.

so4gp's import path, constructor kwargs, and discover() return type all
CHANGED across recent releases (top-level sgp.GRAANK -> sgp.algorithms.GRAANK
at 0.7.2; Python floor 3.12 -> 3.14 for 0.9.x). This test binds the contract against the
*actually installed* library so a silent upstream change fails CI loudly
rather than fabricating patterns at runtime.

Skipped when the ``patterns`` extra is not installed.
Run it in the 3.14 patterns image build:
    uv run --extra patterns pytest tests/test_so4gp_contract.py -o addopts=""
"""

import inspect
import json
import math
from importlib.metadata import version

import pytest

so4gp = pytest.importorskip("so4gp", reason="patterns extra not installed (needs Python >=3.14)")
pd = pytest.importorskip("pandas")

from src.patterns.library_adapter import library_from_mining_result  # noqa: E402
from src.patterns.so4gp_adapter import (  # noqa: E402  (after importorskip by design)
    mine_gradual_patterns,
    mine_temporal_patterns,
    parse_discover_output,
)


def test_installed_default_version():
    assert version("so4gp") == "1.1.2"


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


def test_graank_lives_under_algorithms_namespace():
    # The pre-0.7.2 top-level `from so4gp import GRAANK` must NOT be relied on.
    from so4gp.algorithms import GRAANK  # noqa: F401

    assert not hasattr(so4gp, "GRAANK"), (
        "so4gp.GRAANK reappeared at top level — adapter import path may need review"
    )


def test_graank_accepts_data_source_min_sup_eq():
    from so4gp.algorithms import GRAANK

    params = inspect.signature(GRAANK.__init__).parameters
    assert "data_source" in params
    assert "min_sup" in params
    assert "eq" in params
    assert params["device"].default == "cpu"


def test_discover_returns_json_string(climate_df, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from so4gp.algorithms import GRAANK

    miner = GRAANK(data_source=climate_df, min_sup=0.5, eq=False, device="cpu")
    assert miner.mining_engine._device == "cpu"
    raw = miner.discover(search_type="apriori", target_col=None, save_results=False)
    assert isinstance(raw, str), "discover() must return a JSON string"
    parsed = parse_discover_output(raw)
    assert parsed["algorithm"] == "GRAANK"
    assert isinstance(parsed["patterns"], list)
    assert not list(tmp_path.iterdir())


def test_adapter_end_to_end(climate_df, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # so4gp writes GRAANK_* to cwd — contain it
    result = mine_gradual_patterns(climate_df, min_support=0.5, algorithm="graank")
    assert result["algorithm_key"] == "graank"
    assert result["min_support"] == 0.5
    # precip+ and ndvi+ co-vary in the fixture; at least one pattern found
    assert len(result["patterns"]) >= 1
    for pat in result["patterns"]:
        assert isinstance(pat["support"], float) and 0.0 <= pat["support"] <= 1.0
        assert isinstance(pat["gradual_items"], list)
        assert math.isfinite(pat["metrics"]["conf"])
    library = library_from_mining_result(result, zone_id="cpu-fixture", algorithm="graank")
    assert len(library.patterns) == len(result["patterns"])
    assert all(record.confidence is None for record in library.patterns)
    assert not list(tmp_path.iterdir())


# --- Public TGRAANK facade, exposed by GeoAI's stable tgrad key -------------


@pytest.fixture
def temporal_df():
    # TGrad needs a parseable date-time column rendered as a STRING + numeric
    # attributes. precipitation rises, ndvi follows, temperature falls.
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


def test_tgraank_lives_under_algorithms_namespace():
    from so4gp.algorithms import TGRAANK  # noqa: F401

    assert not hasattr(so4gp.algorithms, "TGrad")


def test_tgraank_requires_target_on_discover():
    from so4gp.algorithms import TGRAANK

    params = inspect.signature(TGRAANK.__init__).parameters
    assert "target_col" not in params
    assert "min_rep" in params
    assert params["device"].default == "cpu"
    discover = inspect.signature(TGRAANK.discover).parameters
    assert discover["target_col"].default is inspect.Parameter.empty
    assert discover["transformations"].default == "ami"
    assert discover["save_results"].default is False


def test_tgraank_discovers_temporal_patterns(temporal_df, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from so4gp.algorithms import TGRAANK

    miner = TGRAANK(temporal_df, min_sup=0.4, min_rep=0.5, device="cpu")
    assert miner.mining_engine._device == "cpu"
    raw = miner.discover(
        target_col=2, transformations="all", search_algorithm="apriori", save_results=False
    )
    assert isinstance(raw, str)
    parsed = parse_discover_output(raw, is_temporal=True)
    # The public facade retains the engine's TGrad algorithm label.
    assert parsed["algorithm"] == "TGrad"
    assert parsed["patterns"]
    assert not list(tmp_path.iterdir())


def test_temporal_adapter_end_to_end(temporal_df, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # so4gp writes TGrad_* to cwd — contain it
    captured = []
    result = mine_temporal_patterns(
        temporal_df,
        target_col=2,
        min_support=0.4,
        min_rep=0.5,
        algorithm="tgrad",
        capture_raw_output=captured.append,
    )
    assert result["algorithm_key"] == "tgrad"
    assert result["target_col"] == 2
    assert len(result["patterns"]) >= 1

    # Every pattern is temporal: a target item (lag 0) + ≥1 lagged item whose
    # lag normalizes to a real number of days.
    sample = result["patterns"][0]
    assert sample["is_temporal"] is True
    roles = [gi["role"] for gi in sample["gradual_items"]]
    assert "target" in roles and "lagged" in roles
    lagged = [gi for gi in sample["gradual_items"] if gi["role"] == "lagged"]
    assert all(isinstance(gi["lag_days"], (int, float)) for gi in lagged)
    assert any(gi["lag_days"] > 0 for gi in lagged)
    native = json.loads(captured[0])
    assert len(native["Patterns"]) == len(result["patterns"])
    for entry, pattern in zip(native["Patterns"], result["patterns"], strict=True):
        # Pin the actual combined native assignments, not a second call to the
        # same parser that could repeat its own normalization mistake.
        assignments = dict(part.split("=", 1) for part in entry[1].split(", "))
        assert pattern["support"] == float(assignments["sup"])
        assert pattern["metrics"]["conf"] == float(assignments["conf"])
    assert all(
        math.isfinite(pattern["metrics"]["conf"]) and 0.0 <= pattern["support"] <= 1.0
        for pattern in result["patterns"]
    )
    assert any(
        item["attribute"] == "precipitation" and item["lag_days"] == 7.0
        for pattern in result["patterns"]
        for item in pattern["gradual_items"]
        if item["role"] == "lagged"
    )
    library = library_from_mining_result(result, zone_id="cpu-fixture", algorithm="tgrad")
    assert len(library.patterns) == len(result["patterns"])
    assert all(record.confidence is None for record in library.patterns)
    assert any(record.lag_days > 0 for record in library.patterns)
    assert not list(tmp_path.iterdir())


def test_tgraank_accepts_datetime64_dtype(temporal_df, tmp_path, monkeypatch):
    # GeoAI converts typed dates to UTC strings before native TGRAANK mining.
    # That conversion must preserve the same patterns and physical lag scale.
    monkeypatch.chdir(tmp_path)
    string_result = mine_temporal_patterns(temporal_df, target_col=2, min_support=0.4)
    dt_df = temporal_df.copy()
    dt_df["date"] = pd.to_datetime(dt_df["date"])
    original = dt_df.copy(deep=True)
    result = mine_temporal_patterns(dt_df, target_col=2, min_support=0.4)
    assert result["patterns"]
    assert {key: value for key, value in result.items() if key != "runtime_seconds"} == {
        key: value for key, value in string_result.items() if key != "runtime_seconds"
    }
    assert all(
        isinstance(item["lag_days"], (int, float)) and math.isfinite(item["lag_days"])
        for pattern in result["patterns"]
        for item in pattern["gradual_items"]
        if item["role"] == "lagged"
    )
    assert any(
        item["attribute"] == "precipitation" and item["lag_days"] == 7.0
        for pattern in result["patterns"]
        for item in pattern["gradual_items"]
        if item["role"] == "lagged"
    )
    pd.testing.assert_frame_equal(dt_df, original)
    assert not list(tmp_path.iterdir())


def test_temporal_adapter_rejects_missing_physical_time(temporal_df, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    invalid = temporal_df.copy()
    invalid["date"] = "invalid"
    # Upstream can substitute NoTime and row-step 'lag' units. GeoAI must not
    # relabel that output as physical-day temporal evidence.
    with pytest.raises(ValueError, match="date|time|lag"):
        mine_temporal_patterns(invalid, target_col=2, min_support=0.4)


def test_temporal_adapter_rejects_multiple_clocks(temporal_df, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    ambiguous = temporal_df.assign(second_date=temporal_df["date"])
    # Native TGRAANK sums both timestamps and doubles the physical lag.
    with pytest.raises(ValueError):
        mine_temporal_patterns(ambiguous, target_col=2, min_support=0.4)
    assert not list(tmp_path.iterdir())


def test_temporal_adapter_rejects_column_removal_before_target(temporal_df, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    invalid = temporal_df.copy()
    invalid.insert(1, "text", "unparseable")
    assert invalid.columns[3] == "ndvi"
    # Native cleaning drops text: unchanged target_col=3 would mine temperature
    # while GeoAI labels the result ndvi. Reject the input before that shift.
    with pytest.raises(ValueError):
        mine_temporal_patterns(invalid, target_col=3, min_support=0.4)
    assert not list(tmp_path.iterdir())


def test_temporal_adapter_rejects_missing_numeric_row(temporal_df, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    invalid = temporal_df.copy()
    invalid.loc[invalid.index[4], "precipitation"] = float("nan")
    # Native dropna silently changes the population and temporal transformations.
    with pytest.raises(ValueError):
        mine_temporal_patterns(invalid, target_col=2, min_support=0.4)
    assert not list(tmp_path.iterdir())
