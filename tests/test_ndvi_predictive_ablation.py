"""Synthetic, offline guards for the frozen predictive ablation protocol.

These tests never open the historical archives. Small counterexamples check
information boundaries and independent statistical invariants; mocked mining
results are deliberately restricted to this synthetic test path.
"""

from __future__ import annotations

import builtins
import calendar
import copy
import csv
import importlib.util
import io
import itertools
import json
import shutil
import subprocess
from datetime import date
from pathlib import Path, PureWindowsPath
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "ndvi_predictive_ablation", ROOT / "benchmarks" / "ndvi_predictive_ablation.py"
)
ablation = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ablation)


@pytest.fixture
def config():
    return ablation.verify_protocol(ROOT / "tests/fixtures/ndvi-predictive-ablation-2026-10-25")[0]


@pytest.fixture
def windows_text_defaults(monkeypatch):
    """Simulate Windows locale/newline defaults without replacing binary I/O."""
    original_open = Path.open
    original_text_encoding = io.text_encoding

    def windows_text_encoding(encoding, stacklevel=2):
        if encoding in (None, "locale"):
            return "cp1252"
        return original_text_encoding(encoding, stacklevel)

    def windows_open(path, mode="r", buffering=-1, encoding=None, errors=None, newline=None):
        if "b" not in mode:
            if encoding in (None, "locale"):
                encoding = "cp1252"
            if newline is None:
                newline = "\r\n"
        return original_open(
            path, mode, buffering=buffering, encoding=encoding, errors=errors, newline=newline
        )

    monkeypatch.setattr(Path, "open", windows_open)
    # Path.write_text resolves defaults through io.text_encoding before open;
    # override that resolution as well, including on a UTF-8-mode interpreter.
    monkeypatch.setattr(io, "text_encoding", windows_text_encoding)


def test_benchmark_imports_when_platform_has_no_resource_module(monkeypatch):
    original_import = builtins.__import__
    unavailable_imports = []

    def without_resource(name, *args, **kwargs):
        if name == "resource":
            unavailable_imports.append(name)
            raise ModuleNotFoundError("resource is unavailable on this platform", name="resource")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_resource)
    spec = importlib.util.spec_from_file_location("ablation_without_resource", SPEC.origin)
    portable = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(portable)
    assert unavailable_imports == ["resource"]
    assert portable.resource is None
    assert portable.peak_memory_kib() == {
        "parent_peak_rss_kib": None,
        "native_child_peak_rss_kib": None,
    }
    assert callable(portable.build_examples)


@pytest.mark.parametrize(
    "platform, expected",
    [
        ("linux", {"parent_peak_rss_kib": 5632, "native_child_peak_rss_kib": 2048}),
        ("darwin", {"parent_peak_rss_kib": 5.5, "native_child_peak_rss_kib": 2.0}),
    ],
)
def test_peak_memory_units_are_normalized_for_platform(platform, expected, monkeypatch):
    observed = []

    def getrusage(who):
        observed.append(who)
        return SimpleNamespace(ru_maxrss={101: 5632, 202: 2048}[who])

    monkeypatch.setattr(
        ablation,
        "resource",
        SimpleNamespace(
            RUSAGE_SELF=101,
            RUSAGE_CHILDREN=202,
            getrusage=getrusage,
        ),
    )
    monkeypatch.setattr(ablation.sys, "platform", platform)
    assert ablation.peak_memory_kib() == expected
    assert observed == [101, 202]


def test_one_unavailable_memory_metric_does_not_discard_the_other(monkeypatch):
    def getrusage(who):
        if who == 101:
            raise OSError("self memory accounting unavailable")
        return SimpleNamespace(ru_maxrss=2048)

    monkeypatch.setattr(
        ablation,
        "resource",
        SimpleNamespace(
            RUSAGE_SELF=101,
            RUSAGE_CHILDREN=202,
            getrusage=getrusage,
        ),
    )
    monkeypatch.setattr(ablation.sys, "platform", "linux")
    assert ablation.peak_memory_kib() == {
        "parent_peak_rss_kib": None,
        "native_child_peak_rss_kib": 2048,
    }


def climate_month(month: str, *, value: float = 1.0):
    year, number = map(int, month.split("-"))
    return [
        {
            "site": "fos_sur_mer",
            "date": date(year, number, day).isoformat(),
            "temperature_2m": value,
            "precipitation": value,
            "humidity": 50.0,
            "evapotranspiration": value,
            "wind_speed": value,
            "solar_radiation": value,
        }
        for day in range(1, calendar.monthrange(year, number)[1] + 1)
    ]


def ndvi(month: str, value: float, day: int = 15):
    return {"site": "fos_sur_mer", "date": f"{month}-{day:02d}", "ndvi": value}


def by_target(result):
    return {row["target_month"]: row for row in result["examples"]}


def test_split_is_target_month_and_december_origin_is_legitimate(config):
    observations = [ndvi("2022-11", 0.2), ndvi("2022-12", 0.3, 31), ndvi("2023-01", 0.4, 2)]
    weather = climate_month("2022-10") + climate_month("2022-11") + climate_month("2022-12")
    rows = by_target(ablation.build_examples(observations, weather, config))
    assert rows["2022-12"]["split"] == "train"
    assert rows["2022-12"]["origin_month"] == "2022-11"
    assert rows["2023-01"]["split"] == "evaluation"
    assert rows["2023-01"]["origin_month"] == "2022-12"
    assert rows["2023-01"]["origin_date"].startswith("2022-12-31")
    assert rows["2023-01"]["origin_ndvi"] == 0.3
    assert rows["2023-01"]["origin_ndvi_date"] <= rows["2023-01"]["origin_date"]
    assert rows["2023-01"]["target_date"].startswith("2023-01-02")


def test_actual_target_day_and_future_weather_do_not_change_predictors(config):
    observations = [ndvi("2022-11", 0.2), ndvi("2022-12", 0.3), ndvi("2023-01", 0.4, 2)]
    weather = climate_month("2022-11", value=2) + climate_month("2022-12", value=3)
    original = by_target(ablation.build_examples(observations, weather, config))["2023-01"]
    changed = copy.deepcopy(observations)
    changed[-1]["date"] = "2023-01-30"
    changed_weather = weather + climate_month("2023-01", value=999)
    altered = by_target(ablation.build_examples(changed, changed_weather, config))["2023-01"]
    assert altered["drivers"] == original["drivers"]
    assert altered["seasonal"] == original["seasonal"]
    assert altered["origin_ndvi"] == original["origin_ndvi"] == 0.3
    assert altered["target_date"] != original["target_date"]
    assert len(original["drivers"]) == 13
    # Aggregations can be checked by hand and must use November/December only.
    assert original["drivers"] == pytest.approx([0.3, 3, 93, 50, 93, 3, 3, 2, 60, 50, 60, 2, 2])


@pytest.mark.parametrize(
    "fault", ["missing_day", "duplicate_day", "nonfinite", "negative", "humidity"]
)
def test_incomplete_or_invalid_weather_excludes_target_without_repair(config, fault):
    observations = [ndvi("2023-01", 0.2), ndvi("2023-02", 0.3)]
    weather = climate_month("2022-12") + climate_month("2023-01")
    if fault == "missing_day":
        weather.pop()
    elif fault == "duplicate_day":
        weather.append(copy.deepcopy(weather[-1]))
    elif fault == "nonfinite":
        weather[-1]["temperature_2m"] = float("nan")
    elif fault == "negative":
        weather[-1]["precipitation"] = -1
    else:
        weather[-1]["humidity"] = 101
    result = ablation.build_examples(observations, weather, config)
    assert "2023-02" not in by_target(result)
    assert any(row["target_month"] == "2023-02" for row in result["exclusions"])


def test_calendar_gap_is_not_bridged_and_future_ndvi_is_not_origin(config):
    observations = [ndvi("2022-12", 0.1), ndvi("2023-02", 0.9)]
    weather = climate_month("2022-12") + climate_month("2023-01")
    result = ablation.build_examples(observations, weather, config)
    assert "2023-02" not in by_target(result)
    assert any(row["target_month"] == "2023-02" for row in result["exclusions"])


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1.01, 1.01])
def test_invalid_ndvi_is_rejected_instead_of_clipped_or_imputed(config, value):
    observations = [ndvi("2022-12", value), ndvi("2023-01", 0.3)]
    weather = climate_month("2022-11") + climate_month("2022-12")
    with pytest.raises(ValueError, match="Rejected NDVI"):
        ablation.build_examples(observations, weather, config)


def test_leap_february_requires_29_days(config):
    observations = [ndvi("2024-02", 0.2), ndvi("2024-03", 0.3)]
    weather = climate_month("2024-01") + climate_month("2024-02")
    assert "2024-03" in by_target(ablation.build_examples(observations, weather, config))
    weather = [row for row in weather if row["date"] != "2024-02-29"]
    assert "2024-03" not in by_target(ablation.build_examples(observations, weather, config))


def test_empirical_rank_uses_training_ties_and_out_of_range_limits():
    # Four training observations: two ties at 2. Holdout values never affect n.
    expected = [0, 0.125, 0.5, 0.75, 0.875, 1]
    actual = ablation.empirical_ranks([1, 2, 2, 4], [-100, 1, 2, 3, 4, 100])
    assert actual == pytest.approx(expected)
    assert ablation.empirical_ranks([1, 2, 2, 4], [2])[0] == actual[2]


def test_strict_support_counts_ties_as_failures_and_simultaneous_inverse_equally():
    # Six unordered pairs; four are strictly ordered in all columns, one ties
    # in x, and one disagrees with the target. Ties stay in the denominator.
    matrix = np.asarray([[0, 0, 0], [0, 1, 1], [1, 2, 3], [2, 3, 2]], dtype=float)
    items = [(0, "+"), (1, "+"), (2, "+")]
    assert ablation.strict_support(matrix, items) == pytest.approx(4 / 6)
    assert ablation.strict_support(matrix, [(i, "-") for i, _ in items]) == pytest.approx(4 / 6)


def pattern(items, support):
    return {
        "gradual_items": [{"attribute": attribute, "direction": sign} for attribute, sign in items],
        "support": support,
        "metrics": {},
    }


def test_selection_canonicalizes_inverse_deduplicates_and_independently_recounts(config):
    matrix = np.asarray([[0, 0, 0], [0, 1, 1], [1, 2, 3], [2, 3, 2]], dtype=float)
    names = ["a", "b", "target_next_ndvi"]
    positive = [(name, "+") for name in names]
    inverse = [(name, "-") for name in reversed(names)]
    result = ablation.select_patterns(
        [pattern(inverse, 0.667), pattern(positive, 0.667)], matrix, names, config
    )
    assert len(result["selected"]) == 1
    selected = result["selected"][0]
    assert selected["exact_support"] == pytest.approx(4 / 6)
    assert selected["native_support"] == 0.667
    # The target is normalized positive and not included in a feature antecedent.
    assert {item["attribute"] for item in selected["items"]} == {"a", "b"}
    assert all(item["direction"] == "+" for item in selected["items"])
    assert len(result["candidates"]) == 2


def test_exact_threshold_is_excluded_and_native_discrepancy_is_a_failure(config):
    names = ["a", "b", "target_next_ndvi"]
    items = [(name, "+") for name in names]
    # Three tied high rows yield only three strictly concordant pairs of six.
    matrix = np.asarray([[0, 0, 0], [1, 1, 1], [1, 1, 1], [1, 1, 1]], dtype=float)
    assert ablation.strict_support(matrix, [(i, "+") for i in range(3)]) == 0.5
    result = ablation.select_patterns([pattern(items, 0.5)], matrix, names, config)
    assert not result["selected"]
    assert result["candidates"][0]["reason"]
    failed = ablation.select_patterns([pattern(items, 0.7)], matrix, names, config)
    assert failed["failure"] == "native_support_recount_discrepancy"
    assert failed["selected"] == []
    assert failed["candidates"][0]["reason"] == "support_recount_failure"


def test_selection_uses_fixed_lexical_ties_cap_and_target_arity_rules(config):
    drivers = ["a", "b", "c", "d", "e"]
    names = drivers + ["target_next_ndvi"]
    matrix = np.repeat(np.arange(4, dtype=float)[:, None], len(names), axis=1)
    candidates = [
        pattern([(a, "+"), (b, "+"), ("target_next_ndvi", "+")], 1)
        for a, b in itertools.combinations(drivers, 2)
    ]
    candidates += [pattern([("a", "+"), ("b", "+")], 1)]
    candidates += [pattern([("a", "+"), ("target_next_ndvi", "+")], 1)]
    result = ablation.select_patterns(list(reversed(candidates)), matrix, names, config)
    assert len(result["selected"]) == config["maximum_mined_features"] == 8
    signatures = [record["signature"] for record in result["selected"]]
    assert signatures == sorted(signatures)
    reasons = [record["reason"] for record in result["candidates"]]
    assert reasons.count("feature_limit") == 2
    assert reasons.count("no_target") == 1
    assert reasons.count("too_few_antecedents") == 1
    assert (
        result["selected"]
        == ablation.select_patterns(candidates, matrix, names, config)["selected"]
    )


def test_amended_selection_rejects_higher_arity_without_changing_general_helper(config):
    names = ["a", "b", "c", "target_next_ndvi"]
    matrix = np.repeat(np.arange(4, dtype=float)[:, None], len(names), axis=1)
    candidate = pattern([(name, "+") for name in names], 1)
    unbounded = copy.deepcopy(config)
    unbounded.pop("maximum_antecedent_size", None)
    assert len(ablation.select_patterns([candidate], matrix, names, unbounded)["selected"]) == 1
    amended = copy.deepcopy(config)
    amended["maximum_antecedent_size"] = 2
    result = ablation.select_patterns([candidate], matrix, names, amended)
    assert result["selected"] == []
    assert result["candidates"][0]["reason"] == "too_many_antecedents"


def test_signed_interaction_is_minimum_of_training_fitted_ranks():
    ranks = np.asarray([[0, 0], [0.5, 0.75], [1, 1]])
    features = [
        {
            "items": [
                {"attribute": "a", "direction": "+"},
                {"attribute": "b", "direction": "-"},
            ]
        }
    ]
    assert ablation.interaction_matrix(ranks, features, ["a", "b"])[:, 0] == pytest.approx(
        [0, 0.25, 0]
    )


def test_ridge_training_scaling_intercept_and_constant_drop_invariants():
    X = np.asarray([[1, 8, 3], [2, 8, 1], [4, 8, 2], [7, 8, 5]], dtype=float)
    y = np.asarray([0.1, 0.4, 0.3, 0.7])
    model = ablation.fit_ridge(X, y, feature_names=["x", "constant", "z"])
    assert model["mean"] == pytest.approx([3.5, 8, 2.75])
    assert model["std"][0] == pytest.approx(np.std([1, 2, 4, 7]))
    base = ablation.predict_ridge(model, X)
    assert np.mean(base) == pytest.approx(np.mean(y))
    # Changing a training-constant column at evaluation must have no effect.
    altered = X.copy()
    altered[:, 1] = [1e6, -1e6, 0, 4]
    assert ablation.predict_ridge(model, altered) == pytest.approx(base)
    # Fixed standardized ridge must be invariant to units and unpenalized y shifts.
    units = np.asarray([100, 1, 0.01])
    rescaled = ablation.fit_ridge(X * units, y, feature_names=["x", "constant", "z"])
    assert ablation.predict_ridge(rescaled, X * units) == pytest.approx(base)
    shifted = ablation.fit_ridge(X, y + 0.2)
    assert ablation.predict_ridge(shifted, X) == pytest.approx(np.asarray(base) + 0.2)


def test_constant_only_ridge_predicts_training_mean_without_singular_failure():
    model = ablation.fit_ridge([[4, 9], [4, 9], [4, 9]], [0.1, 0.3, 0.5])
    assert model["keep"] == []
    assert ablation.predict_ridge(model, [[-999, 999], [4, 9]]) == pytest.approx([0.3, 0.3])


def test_generic_interactions_match_complexity_are_deterministic_and_need_no_labels():
    selected = [
        {"items": [{"attribute": "a", "direction": "+"}, {"attribute": "b", "direction": "-"}]},
        {
            "items": [
                {"attribute": "a", "direction": "+"},
                {"attribute": "b", "direction": "+"},
                {"attribute": "c", "direction": "-"},
            ]
        },
    ]
    names = ["a", "b", "c", "d", "e"]
    result = ablation.generic_interactions(selected, names, seed=20261025)
    assert result == ablation.generic_interactions(copy.deepcopy(selected), names, seed=20261025)
    assert len(result) == 2
    for source, generic in zip(selected, result, strict=True):
        assert len(generic["items"]) == len(source["items"])
        assert sum(item["direction"] == "+" for item in generic["items"]) == sum(
            item["direction"] == "+" for item in source["items"]
        )
        assigned = [item["attribute"] for item in generic["items"]]
        assert len(assigned) == len(set(assigned))
        assert set(assigned) <= set(names)
    # Irrelevant supervised metadata cannot alter this label-independent draw.
    for interaction in selected:
        interaction["native_support"] = -999
        interaction["target_values"] = [999, -999]
    assert result == ablation.generic_interactions(selected, names, seed=20261025)


def synthetic_examples(config):
    """Use calendar builder to avoid manufacturing an undocumented row schema."""
    observations = []
    weather = []
    for year in (2021, 2022, 2023, 2024):
        for month in range(1, 13):
            label = f"{year}-{month:02d}"
            observations.append(ndvi(label, 0.1 + month / 40 + (year - 2021) / 30))
            weather.extend(climate_month(label, value=float(month)))
    return ablation.build_examples(observations, weather, config)


def test_all_arms_score_same_origins_and_empty_mining_exactly_falls_back(config):
    built = synthetic_examples(config)
    result = ablation.evaluate_site(
        built["examples"],
        built["ndvi_history"]["fos_sur_mer"],
        config,
        native_result={"patterns": []},
    )
    per_arm = {}
    for row in result["predictions"]:
        if row["split"] != "evaluation":
            continue
        per_arm.setdefault(row["arm"], {})[row["target_month"]] = row
        assert -1 <= row["prediction"] <= 1
    assert len(per_arm) == 7
    scored = [set(rows) for rows in per_arm.values()]
    assert all(origins == scored[0] for origins in scored)
    assert scored[0] == {
        row["target_month"] for row in built["examples"] if row["split"] == "evaluation"
    }
    assert not result["selection"]["selected"]
    assert result["generic_interactions"] == []
    # These names are the protocol arm identifiers, shared by forecasts/models.
    for target in scored[0]:
        rank = per_arm["rank_ridge"][target]
        for arm in ("so4gp_ridge", "generic_ridge"):
            assert per_arm[arm][target]["prediction"] == rank["prediction"]
            assert per_arm[arm][target]["unclipped_prediction"] == rank["unclipped_prediction"]


def test_changing_only_evaluation_labels_cannot_change_fits_ranks_or_predictions(config):
    built = synthetic_examples(config)
    changed = copy.deepcopy(built["examples"])
    for row in changed:
        if row["split"] == "evaluation":
            row["target"] = -0.99
    history = built["ndvi_history"]["fos_sur_mer"]
    original = ablation.evaluate_site(
        built["examples"], history, config, native_result={"patterns": []}
    )
    altered = ablation.evaluate_site(changed, history, config, native_result={"patterns": []})
    assert original["models"] == altered["models"]
    assert original["rank_references"] == altered["rank_references"]
    assert original["selection"] == altered["selection"]
    assert original["generic_interactions"] == altered["generic_interactions"]
    prediction_keys = ("site", "target_month", "arm", "prediction", "unclipped_prediction")
    original_predictions = [
        tuple(row[key] for key in prediction_keys) for row in original["predictions"]
    ]
    altered_predictions = [
        tuple(row[key] for key in prediction_keys) for row in altered["predictions"]
    ]
    assert original_predictions == altered_predictions
    assert original["metrics"] != altered["metrics"]


def test_mining_selection_gets_only_training_drivers_and_labels(config, monkeypatch):
    built = synthetic_examples(config)
    captured = []
    original_select = ablation.select_patterns

    def checked_selection(patterns, matrix, feature_names, config):
        captured.append((np.asarray(matrix).copy(), list(feature_names)))
        return original_select(patterns, matrix, feature_names, config)

    monkeypatch.setattr(ablation, "select_patterns", checked_selection)
    # A strictly monotone training feature pair gives a nonempty selected model.
    examples = copy.deepcopy(built["examples"])
    train = [row for row in examples if row["split"] == "train"]
    for index, row in enumerate(train):
        row["drivers"][0] = index / len(train)
        row["drivers"][1] = index
        row["target"] = index / len(train)
    native = {
        "patterns": [
            pattern(
                [(ablation.DRIVERS[0], "+"), (ablation.DRIVERS[1], "+"), ("target_next_ndvi", "+")],
                1,
            )
        ]
    }
    first = ablation.evaluate_site(
        examples, built["ndvi_history"]["fos_sur_mer"], config, native_result=native
    )
    changed = copy.deepcopy(examples)
    for row in changed:
        if row["split"] == "evaluation":
            row["target"] = -0.9
    second = ablation.evaluate_site(
        changed, built["ndvi_history"]["fos_sur_mer"], config, native_result=native
    )
    assert len(first["selection"]["selected"]) == 1
    assert first["models"] == second["models"]
    assert first["selection"] == second["selection"]
    assert first["generic_interactions"] == second["generic_interactions"]
    assert [(r["arm"], r["prediction"]) for r in first["predictions"]] == [
        (r["arm"], r["prediction"]) for r in second["predictions"]
    ]
    for matrix, names in captured:
        assert names == ablation.DRIVERS + ["target_next_ndvi"]
        assert matrix.shape == (len(train), 14)
        assert matrix[:, :-1] == pytest.approx(np.asarray([row["drivers"] for row in train]))
        assert matrix[:, -1] == pytest.approx([row["target"] for row in train])
    assert np.array_equal(captured[0][0], captured[1][0])


def test_evaluation_covariates_do_not_change_training_scaling_in_any_arm(config):
    built = synthetic_examples(config)
    examples = copy.deepcopy(built["examples"])
    train = [row for row in examples if row["split"] == "train"]
    for index, row in enumerate(train):
        row["drivers"][0] = index / len(train)
        row["drivers"][1] = index
        row["target"] = index / len(train)
    native = {
        "patterns": [
            pattern(
                [(ablation.DRIVERS[0], "+"), (ablation.DRIVERS[1], "+"), ("target_next_ndvi", "+")],
                1,
            )
        ]
    }
    first = ablation.evaluate_site(
        examples, built["ndvi_history"]["fos_sur_mer"], config, native_result=native
    )
    changed = copy.deepcopy(examples)
    for row in changed:
        if row["split"] == "evaluation":
            row["drivers"] = [value + 1e6 for value in row["drivers"]]
            row["seasonal"] = [0.7, -0.7]
    second = ablation.evaluate_site(
        changed, built["ndvi_history"]["fos_sur_mer"], config, native_result=native
    )
    assert first["models"] == second["models"]
    assert first["rank_references"] == second["rank_references"]
    assert first["selection"] == second["selection"]
    # Including a genuine selected interaction makes this protect the additional
    # rank/minimum columns as well as the raw and seasonal feature scaling.
    assert len(first["models"]["so4gp_ridge"]["mean"]) == 29
    clipped = [row for row in second["predictions"] if row["clipped"]]
    assert clipped
    assert all(row["prediction"] in (-1, 1) for row in clipped)
    assert all(abs(row["unclipped_prediction"]) > 1 for row in clipped)


def test_previous_year_uses_only_observed_history_before_origin(config):
    built = synthetic_examples(config)
    history = copy.deepcopy(built["ndvi_history"]["fos_sur_mer"])
    # January 2023 is an earlier evaluation observation, legitimately known
    # by the December 2023 origin of the January 2024 forecast.
    history["2023-01"]["ndvi"] = 0.88
    result = ablation.evaluate_site(
        built["examples"], history, config, native_result={"patterns": []}
    )

    def forecast(value, arm):
        return next(
            row
            for row in value["predictions"]
            if row["target_month"] == "2024-01" and row["arm"] == arm
        )

    assert forecast(result, "previous_year")["prediction"] == 0.88
    assert not forecast(result, "previous_year")["previous_year_fallback"]
    history["2023-01"]["date"] = "2024-01-01"
    late = ablation.evaluate_site(
        built["examples"], history, config, native_result={"patterns": []}
    )
    assert forecast(late, "previous_year")["previous_year_fallback"]
    assert (
        forecast(late, "previous_year")["prediction"]
        == forecast(late, "seasonal_median")["prediction"]
    )
    assert (
        forecast(late, "seasonal_median")["prediction"]
        == forecast(result, "seasonal_median")["prediction"]
    )


def test_create_output_directory_never_overwrites_existing_evidence(tmp_path):
    output = tmp_path / "new-run"
    assert ablation.create_output_dir(output) == output
    evidence = output / "evidence.txt"
    evidence.write_bytes(b"first immutable run")
    with pytest.raises((FileExistsError, ValueError)):
        ablation.create_output_dir(output)
    assert evidence.read_bytes() == b"first immutable run"


def test_archive_snapshot_verifies_the_loaded_bytes_and_survives_later_source_mutation(tmp_path):
    source = tmp_path / "synthetic.zip"
    source.write_bytes(b"authenticated synthetic bytes")
    expected = ablation.sha(source.read_bytes())
    snapshot = ablation.snapshot_archive(source, expected)
    source.write_bytes(b"changed after snapshot")
    assert snapshot == b"authenticated synthetic bytes"
    with pytest.raises(ValueError, match="SHA-256"):
        ablation.snapshot_archive(source, expected)


def test_runtime_rejects_wrong_native_version_before_mining(config, monkeypatch):
    monkeypatch.setattr(
        ablation.importlib.metadata,
        "version",
        lambda package: "0.0.0",
    )
    with pytest.raises(ValueError, match="Native version mismatch"):
        ablation.verify_runtime(config)


def test_no_implicit_mock_or_fixture_fallback_is_accepted(config):
    built = synthetic_examples(config)
    with pytest.raises(ValueError, match="native result required|fixture fallback"):
        ablation.evaluate_site(built["examples"], built["ndvi_history"]["fos_sur_mer"], config)


@pytest.mark.parametrize("flag", ["--mock", "--fixture", "--synthetic"])
def test_real_cli_does_not_offer_fixture_substitution(flag, monkeypatch, tmp_path):
    def must_not_run(*args, **kwargs):
        raise AssertionError("invalid fixture flag must be rejected before experiment execution")

    monkeypatch.setattr(ablation, "run_experiment", must_not_run)
    with pytest.raises(SystemExit) as exc:
        ablation.main(
            [
                "--ndvi-archive",
                str(tmp_path / "unused-ndvi.zip"),
                "--era5-archive",
                str(tmp_path / "unused-era5.zip"),
                "--protocol-dir",
                str(tmp_path / "unused-protocol"),
                "--output-dir",
                str(tmp_path / "unused-output"),
                flag,
            ]
        )
    assert exc.value.code == 2


def test_actual_native_worker_preserves_output_from_pinned_cpu_library(
    config, tmp_path, monkeypatch
):
    pytest.importorskip("so4gp", reason="native CPU contract needs the optional patterns extra")
    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path / "matplotlib"))
    output = tmp_path / "native"
    ablation.verify_runtime(config)
    returned = ablation.native_mine(
        [[0, 0, 0], [1, 1, 1], [2, 2, 2], [3, 3, 3]],
        ["a", "b", "target_next_ndvi"],
        config,
        output,
    )
    raw = json.loads((output / "raw-native.json").read_text(encoding="utf-8"))
    parsed = json.loads((output / "parsed.json").read_text(encoding="utf-8"))
    assert raw["Algorithm"] == "GRAANK"
    assert raw["Patterns"]
    assert returned["patterns"] == parsed["patterns"]
    assert parsed["algorithm_key"] == "graank"
    assert parsed["patterns"]
    selected = ablation.select_patterns(
        parsed["patterns"],
        [[0, 0, 0], [1, 1, 1], [2, 2, 2], [3, 3, 3]],
        ["a", "b", "target_next_ndvi"],
        config,
    )
    assert selected["failure"] is None
    assert selected["selected"][0]["exact_support"] == 1


def test_bounded_subsets_cover_every_unordered_driver_pair_once():
    names = ablation.DRIVERS + ["target_next_ndvi"]
    subsets = ablation.bounded_subsets(names)
    expected = [(a, b, "target_next_ndvi") for a, b in itertools.combinations(ablation.DRIVERS, 2)]
    assert len(subsets) == 78
    assert [tuple(subset) for subset in subsets] == expected
    assert len({frozenset(subset[:2]) for subset in subsets}) == 78
    assert set().union(*(set(subset[:2]) for subset in subsets)) == set(ablation.DRIVERS)


def test_all_pairs_share_one_cpu_worker_and_one_site_timeout(config, tmp_path, monkeypatch):
    calls = []

    def subprocess_run(command, *, cwd, env, capture_output, timeout, check):
        calls.append(command)
        assert timeout == 60
        assert capture_output is True
        assert check is False
        assert env["CUDA_VISIBLE_DEVICES"] == ""
        assert env["OMP_NUM_THREADS"] == "1"
        assert Path(cwd) != tmp_path
        output = Path(command[-1])
        (output / "parsed.json").write_text(
            json.dumps({"patterns": [], "native_call_count": 78}), encoding="utf-8"
        )
        return ablation.subprocess.CompletedProcess(command, 0, stdout=b"", stderr=b"")

    monkeypatch.setattr(ablation.subprocess, "run", subprocess_run)
    result = ablation.native_mine(
        np.ones((3, 14)), ablation.DRIVERS + ["target_next_ndvi"], config, tmp_path / "worker"
    )
    assert len(calls) == 1
    assert calls[0][2] == "--native-worker"
    assert result["native_call_count"] == 78


def test_site_timeout_preserves_partial_native_call_evidence(config, tmp_path, monkeypatch):
    output = tmp_path / "timed-out"

    def timeout_run(command, **kwargs):
        partial = output / "call-000"
        partial.mkdir()
        (partial / "raw-native.json").write_bytes(b"native evidence before timeout")
        raise ablation.subprocess.TimeoutExpired(
            command, kwargs["timeout"], output=b"partial stdout", stderr=b"partial stderr"
        )

    monkeypatch.setattr(ablation.subprocess, "run", timeout_run)
    with pytest.raises(RuntimeError, match="frozen timeout"):
        ablation.native_mine(
            np.ones((3, 14)), ablation.DRIVERS + ["target_next_ndvi"], config, output
        )
    assert (output / "call-000/raw-native.json").read_bytes() == b"native evidence before timeout"
    assert (output / "stdout.txt").read_bytes() == b"partial stdout"
    assert (output / "stderr.txt").read_bytes() == b"partial stderr"


@pytest.mark.parametrize(
    "names",
    [
        ["a", "b"],
        ["a", "target_next_ndvi"],
        ["a", "b", "target_next_ndvi", "target_next_ndvi"],
        ["a", "a", "target_next_ndvi"],
    ],
)
def test_bounded_subsets_reject_ambiguous_or_insufficient_schema(names):
    with pytest.raises(ValueError):
        ablation.bounded_subsets(names)


def test_each_native_pair_gets_exact_driver_columns_and_unchanged_training_labels(config, tmp_path):
    names = ablation.DRIVERS + ["target_next_ndvi"]
    matrix = np.arange(5 * len(names), dtype=float).reshape(5, len(names))
    captured = []
    output = tmp_path / "native"
    output.mkdir()

    def adapter(frame, *, min_support, encode_equal, algorithm, capture_raw_output):
        captured.append((list(frame.columns), frame.to_numpy().copy()))
        assert min_support == 0.5
        assert encode_equal is False
        assert algorithm == "graank"
        capture_raw_output(json.dumps({"Algorithm": "GRAANK", "Patterns": []}))
        return {"algorithm": "GRAANK", "algorithm_key": "graank", "patterns": []}

    amended = copy.deepcopy(config)
    amended["native_strategy"] = "all_driver_pairs_plus_target"
    ablation.native_pair_mining(matrix, names, amended, output, adapter=adapter)
    assert len(captured) == 78
    for (columns, values), expected in zip(captured, ablation.bounded_subsets(names), strict=True):
        assert columns == expected
        assert values.shape == (5, 3)
        assert np.array_equal(values, matrix[:, [names.index(column) for column in columns]])
        assert np.array_equal(values[:, -1], matrix[:, -1])
    assert len(list(output.glob("call-*/raw-native.json"))) == 78
    assert len(list(output.glob("call-*/parsed.json"))) == 78
    assert len(json.loads((output / "call-inventory.json").read_text(encoding="utf-8"))) == 78


@pytest.mark.parametrize("resource_available", [True, False])
def test_experiment_passes_only_training_rows_to_native_miner(
    config, tmp_path, monkeypatch, resource_available, windows_text_defaults
):
    # Exercise orchestration using exclusively invented source rows. The native
    # boundary is captured before mining; no archive is opened by this test.
    base = synthetic_examples(config)
    prepared = {"examples": [], "exclusions": [], "ndvi_history": {}, "monthly_coverage": {}}
    for site in ablation.SITES:
        prepared["ndvi_history"][site] = copy.deepcopy(base["ndvi_history"]["fos_sur_mer"])
        for split in ("train", "evaluation"):
            rows = [row for row in base["examples"] if row["split"] == split]
            for row in rows[: config["expected_eligibility"][site][split]]:
                row = copy.deepcopy(row)
                row["site"] = site
                if split == "evaluation":
                    row["target"] = -0.987654321
                prepared["examples"].append(row)
    captured = []

    def native_boundary(matrix, names, settings, artifact_dir):
        captured.append((np.asarray(matrix).copy(), list(names)))
        assert settings["native_strategy"] == "all_driver_pairs_plus_target"
        Path(artifact_dir).mkdir()
        (Path(artifact_dir) / "synthetic-proof.txt").write_bytes(b"synthetic native boundary")
        return {"patterns": []}

    monkeypatch.setattr(ablation, "load_inputs", lambda *args: ([], [], {"synthetic": True}))
    monkeypatch.setattr(ablation, "build_examples", lambda *args: prepared)
    monkeypatch.setattr(ablation, "verify_runtime", lambda *args: {"synthetic": True})
    monkeypatch.setattr(ablation, "_code_identity", lambda: {"synthetic": True})
    monkeypatch.setattr(ablation, "native_mine", native_boundary)
    if not resource_available:
        monkeypatch.setattr(ablation, "resource", None)
    output = tmp_path / "synthetic-output"
    original_relative_to = Path.relative_to

    def windows_output_relative(path, *other):
        relative = original_relative_to(path, *other)
        if len(other) == 1 and Path(other[0]) == output:
            return PureWindowsPath(*relative.parts)
        return relative

    monkeypatch.setattr(Path, "relative_to", windows_output_relative)
    result = ablation.run_experiment(
        tmp_path / "unused-ndvi.zip",
        tmp_path / "unused-era5.zip",
        ROOT / "tests/fixtures/ndvi-predictive-ablation-2026-10-25",
        output,
    )
    assert len(captured) == len(ablation.SITES)
    for site, (matrix, names) in zip(ablation.SITES, captured, strict=True):
        training = [
            row for row in prepared["examples"] if row["site"] == site and row["split"] == "train"
        ]
        expected = np.asarray([row["drivers"] + [row["target"]] for row in training])
        assert names == ablation.DRIVERS + ["target_next_ndvi"]
        assert np.array_equal(matrix, expected)
        assert matrix.shape[1] == 14
        assert -0.987654321 not in matrix[:, -1]
    # This uses the complete synthetic fitted result, including twelve seasonal
    # medians. Integer month keys used to sort differently after JSON reload,
    # so an in-memory hash could not authenticate its own saved result.json.
    saved = json.loads((output / "result.json").read_text(encoding="utf-8"))
    if not resource_available:
        for metric in ("parent_peak_rss_kib", "native_child_peak_rss_kib"):
            assert result["execution"][metric] is None
            assert saved["execution"][metric] is None
    reloaded_digest = ablation.sha(ablation.encode(ablation.scientific_payload(saved)))
    assert reloaded_digest == saved["scientific_sha256"] == result["scientific_sha256"]
    for site in ablation.SITES:
        medians = result["sites"][site]["seasonal_baseline"]["monthly_medians"]
        assert set(medians) == {str(month) for month in range(1, 13)}
        assert saved["sites"][site]["predictions"] == result["sites"][site]["predictions"]
    report_bytes = (output / "report.md").read_bytes()
    report = report_bytes.decode("utf-8")
    assert "−" in report
    assert b"\r\n" not in report_bytes
    assert (
        "Eligible candidate target months and input coverage are in examples.json; "
        "excluded candidate months and their reasons are in exclusions.csv."
    ) in report
    manifest = json.loads((output / "artifact-manifest.json").read_text(encoding="utf-8"))
    assert any("/" in item["path"] for item in manifest["files"])
    for item in manifest["files"]:
        assert "\\" not in item["path"]
        artifact = output / item["path"]
        assert artifact.is_file()
        assert ablation.sha(artifact.read_bytes()) == item["sha256"]


def test_autocrlf_checkout_preserves_all_frozen_protocol_bytes(tmp_path):
    repo = tmp_path / "git-repo"
    repo.mkdir()
    protocol_relative = Path("tests/fixtures/ndvi-predictive-ablation-2026-10-25")
    staged_protocol = repo / protocol_relative
    staged_protocol.mkdir(parents=True)
    frozen = (
        *ablation.PROTOCOL_PINS,
        "protocol-freeze.json",
        *ablation.EXECUTION_PINS,
        "execution-freeze.json",
    )
    assert len(frozen) == 6
    originals = {name: (ROOT / protocol_relative / name).read_bytes() for name in frozen}
    for name, content in originals.items():
        (staged_protocol / name).write_bytes(content)
    (repo / ".gitattributes").write_bytes((ROOT / ".gitattributes").read_bytes())
    (repo / "autocrlf-control.txt").write_bytes(b"first line\nsecond line\n")

    def git(*args):
        return subprocess.run(
            ["git", *args], cwd=repo, capture_output=True, check=True, encoding="utf-8"
        )

    git("init")
    git("config", "core.autocrlf", "true")
    git("config", "core.safecrlf", "false")
    git("add", ".gitattributes", "autocrlf-control.txt", protocol_relative.as_posix())
    checkout = tmp_path / "checkout"
    git("checkout-index", "--all", "--prefix=" + checkout.as_posix() + "/")
    # The control demonstrates actual Git newline conversion is active. The
    # six authenticated files must keep their original LF bytes despite it.
    assert (checkout / "autocrlf-control.txt").read_bytes() == b"first line\r\nsecond line\r\n"
    for name, content in originals.items():
        assert (checkout / protocol_relative / name).read_bytes() == content
    effective, _ = ablation.verify_protocol(checkout / protocol_relative)
    assert effective["native_strategy"] == "all_driver_pairs_plus_target"


@pytest.mark.parametrize("note", ["native ASCII output", "température −1 °C"])
def test_native_raw_bytes_and_inventory_hash_ignore_text_defaults(
    config, tmp_path, windows_text_defaults, note
):
    # Prove the simulation exercises both locale and newline traps.
    with pytest.raises(UnicodeEncodeError):
        (tmp_path / "locale-control.txt").write_text("−")
    (tmp_path / "newline-control.txt").write_text("first\nsecond\n", encoding="utf-8")
    assert (tmp_path / "newline-control.txt").read_bytes() == b"first\r\nsecond\r\n"
    raw = (
        json.dumps(
            {"Algorithm": "GRAANK", "Patterns": [], "Note": note},
            ensure_ascii=False,
            indent=2,
        )
        + "\n"
    )
    output = tmp_path / "native"
    output.mkdir()

    def adapter(frame, *, capture_raw_output, **kwargs):
        capture_raw_output(raw)
        return {"patterns": [], "algorithm_key": "graank"}

    ablation.native_pair_mining(
        [[0, 1, 0.1], [1, 2, 0.2]],
        ["a", "b", "target_next_ndvi"],
        config,
        output,
        adapter=adapter,
    )
    exact = (output / "call-000/raw-native.json").read_bytes()
    inventory = json.loads((output / "call-inventory.json").read_text(encoding="utf-8"))
    assert exact == raw.encode("utf-8")
    assert inventory[0]["raw_sha256"] == ablation.sha(exact)


def test_csv_is_utf8_with_exact_line_endings_under_windows_defaults(
    tmp_path, windows_text_defaults
):
    output = tmp_path / "non-ascii.csv"
    rows = [{"site": "Fos-sur-Mer", "note": "température −1 °C"}]
    ablation._write_csv(output, rows)
    raw = output.read_bytes()
    assert raw == "site,note\r\nFos-sur-Mer,température −1 °C\r\n".encode()
    assert list(csv.DictReader(io.StringIO(raw.decode("utf-8"), newline=""))) == rows
    empty = tmp_path / "empty.csv"
    ablation._write_csv(empty, [])
    assert empty.read_bytes() == b""


def test_protocol_hash_mutation_fails_even_if_local_freeze_is_rewritten(tmp_path):
    source = ROOT / "tests/fixtures/ndvi-predictive-ablation-2026-10-25"
    copied = tmp_path / "protocol"
    shutil.copytree(source, copied)
    loaded, receipt = ablation.verify_protocol(copied)
    assert loaded["native_version"] == "1.1.2"
    assert receipt["files"] == ablation.PROTOCOL_PINS
    changed = copied / "protocol.json"
    changed.write_bytes(changed.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="mutation"):
        ablation.verify_protocol(copied)
    freeze_path = copied / "protocol-freeze.json"
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    freeze["files"]["protocol.json"] = ablation.sha(changed.read_bytes())
    freeze_path.write_text(json.dumps(freeze), encoding="utf-8")
    with pytest.raises(ValueError, match="code-pinned|receipt mutation"):
        ablation.verify_protocol(copied)


@pytest.mark.parametrize(
    "filename",
    [
        "EXECUTION_AMENDMENT.md",
        "execution-amendment.json",
        "execution-freeze.json",
    ],
)
def test_amendment_or_receipt_mutation_is_rejected_before_any_execution(tmp_path, filename):
    source = ROOT / "tests/fixtures/ndvi-predictive-ablation-2026-10-25"
    copied = tmp_path / "amendment"
    shutil.copytree(source, copied)
    effective, receipt = ablation.verify_protocol(copied)
    assert effective["native_strategy"] == "all_driver_pairs_plus_target"
    assert effective["minimum_antecedent_size"] == effective["maximum_antecedent_size"] == 2
    assert receipt["execution_files"] == ablation.EXECUTION_PINS
    assert receipt["execution_freeze"]["predictive_model_fitting_or_scoring_started"] is False
    changed = copied / filename
    changed.write_bytes(changed.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="execution.*mutation"):
        ablation.verify_protocol(copied)
    if filename != "execution-freeze.json":
        receipt_path = copied / "execution-freeze.json"
        edited_receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        edited_receipt["files"][filename] = ablation.sha(changed.read_bytes())
        receipt_path.write_text(json.dumps(edited_receipt), encoding="utf-8")
        with pytest.raises(ValueError, match="execution receipt mutation"):
            ablation.verify_protocol(copied)


def test_scientific_digest_ignores_clocks_but_detects_predictions_and_source_changes():
    result = {
        "protocol": {
            "files": {"protocol.json": "frozen"},
            "execution_files": {"execution-amendment.json": "amended"},
            "execution_freeze_sha256": "amendment-receipt",
        },
        "code": {"commit": "code"},
        "runtime": {"so4gp": "1.1.2"},
        "inputs": {"sha256": "source"},
        "examples": [],
        "exclusions": [],
        "monthly_coverage": {},
        "execution": {"elapsed_seconds": 1},
        "sites": {
            "fos_sur_mer": {
                "predictions": [{"prediction": 0.3}],
                "native_normalized": {
                    "patterns": [{"support": 0.7}],
                    "runtime_seconds": 1,
                    "wall_seconds": 2,
                },
            }
        },
    }

    def digest(value):
        return ablation.sha(ablation.encode(ablation.scientific_payload(value)))

    changed = copy.deepcopy(result)
    changed["execution"] = {"elapsed_seconds": 99, "input_paths": {"archive": "/other/path"}}
    changed["sites"]["fos_sur_mer"]["native_normalized"]["runtime_seconds"] = 99
    changed["sites"]["fos_sur_mer"]["native_normalized"]["wall_seconds"] = 999
    assert digest(result) == digest(changed)
    changed["sites"]["fos_sur_mer"]["predictions"][0]["prediction"] += 0.01
    assert digest(result) != digest(changed)
    changed = copy.deepcopy(result)
    changed["inputs"]["sha256"] = "other source"
    assert digest(result) != digest(changed)
    changed = copy.deepcopy(result)
    changed["protocol"]["execution_files"]["execution-amendment.json"] = "other amendment"
    assert digest(result) != digest(changed)
