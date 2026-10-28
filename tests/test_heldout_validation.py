import json

import pandas as pd
import pytest

from src.patterns import heldout_validation as cv


def frame(x, y, dates=None):
    return pd.DataFrame(
        {
            "date": pd.to_datetime(dates or pd.date_range("2021-01-01", periods=len(x)).tolist()),
            "temperature_2m": x,
            "ndvi_mean": y,
        }
    )


def rule(target="+", lag=0):
    return cv.Rule((("temperature_2m", "+"),), target, lag)


def test_fixed_direction_scores_failures_and_ties():
    assert cv.score(frame([1, 2, 3], [3, 2, 1]), rule()).joint == 0
    assert cv.score(frame([1, 2, 3], [3, 2, 1]), rule("-")).joint == 3
    tied = cv.score(frame([1, 1, 1], [1, 2, 3]), rule())
    assert tied.pairs == 3 and tied.antecedent == 0
    assert tied.as_dict()["confidence"] is None
    assert tied.as_dict()["status"] == "non_evaluable"


def test_pool_counts_not_ratios():
    result = cv.Counts(1, 1, 1) + cv.Counts(9, 9, 0)
    assert result.as_dict()["confidence"] == 0.1
    assert result.as_dict()["support"] == 0.1


def test_default_confidence_bar_is_one_half():
    assert cv.Counts(10, 10, 5).as_dict()["status"] == "pass"


def test_lower_confidence_does_not_bypass_support_bar():
    # Confidence is 0.80, but joint support is only 0.40.
    assert cv.Counts(10, 5, 4).as_dict(min_confidence=0.5)["status"] == "fail"


def test_transitive_geometry_groups_and_deterministic_folds():
    from shapely.geometry import box, mapping

    geo = {
        "features": [
            {
                "properties": {"zone_id": str(i), "koppen": "Csa"},
                "geometry": mapping(box(a, 0, b, 1)),
            }
            for i, (a, b) in enumerate(
                [(0, 2), (1, 3), (2.5, 4), (10, 11), (20, 21), (30, 31), (40, 41)]
            )
        ]
    }
    folds = cv.geometry_folds(geo, list("0123456"))
    assert folds == cv.geometry_folds(
        {"features": list(reversed(geo["features"]))}, list("6543210")
    )
    assert any(set("012") <= set(fold) for fold in folds)
    assert sorted(z for fold in folds for z in fold) == list("0123456")
    with pytest.raises(ValueError, match="independent"):
        cv.geometry_folds(geo, list("012"))


def test_heldout_values_cannot_select_candidates_or_flip_direction():
    frames = {str(i): frame([1, 2, 3], [1, 2, 3]) for i in range(5)}
    folds = [[str(i)] for i in range(5)]
    first = cv.spatial_validation(frames, folds, [rule(), rule("-")])
    frames["0"] = frame([1, 2, 3], [3, 2, 1])
    second = cv.spatial_validation(frames, folds, [rule(), rule("-")])
    a = [r for r in first["fold_scores"] if r["fold"] == 1]
    b = [r for r in second["fold_scores"] if r["fold"] == 1]
    assert [r["selected"] for r in a] == [r["selected"] for r in b]
    assert b[0]["joint"] == 0 and b[0]["status"] == "fail"
    assert not any(r["rule_id"] == rule("-").id and r["selected"] for r in b)


def test_candidate_not_selected_in_one_fold_is_visible():
    frames = {str(i): frame([1, 2, 3], [3, 2, 1]) for i in range(5)}
    frames["0"] = frame(list(range(10)), list(range(10)))
    result = cv.spatial_validation(frames, [[str(i)] for i in range(5)], [rule()])
    assert result["fold_scores"][0]["status"] == "not_selected"
    assert result["candidates"][0]["passing_folds"] == 0
    assert result["candidates"][0]["stability"] == 0


def test_nonevaluable_fold_cannot_inflate_stability():
    frames = {str(i): frame([1, 2, 3], [1, 2, 3]) for i in range(5)}
    frames["0"] = frame([1, 1, 1], [1, 2, 3])
    frames["1"] = frame([1, 1, 1], [1, 2, 3])
    result = cv.spatial_validation(frames, [[str(i)] for i in range(5)], [rule()])
    candidate = result["candidates"][0]
    assert candidate["passing_folds"] == 3
    assert candidate["stability"] == 0.6 and not candidate["validated"]
    assert len(result["fold_scores"]) == 5


def test_lag_split_excludes_cross_boundary_window():
    data = frame(
        [1, 2, 3, 4], [1, 2, 3, 4], ["2022-11-01", "2022-12-01", "2023-01-01", "2023-02-01"]
    )
    train, test = cv.temporal_split(data)
    assert len(cv.transform(test, 1)) == 1
    assert cv.score(test, rule(lag=1)).pairs == 0
    assert len(cv.transform(train, 1)) == 1
    assert cv.transform(test, 1)["temperature_2m"].tolist() == [3]


def test_temporal_donor_freeze_and_strict_serialization():
    dates = ["2021-01-01", "2021-02-01", "2022-01-01", "2023-01-01", "2023-02-01", "2024-01-01"]
    frames = {"pilot": frame([1, 2, 3, 4, 5, 6], [1, 2, 3, 6, 5, 4], dates)}
    result = cv.temporal_validation(frames, [rule(), rule("-"), rule(lag=1)])
    selected = {r["rule_id"] for r in result["candidates"]}
    assert rule().id in selected and rule("-").id not in selected
    assert next(r for r in result["scores"] if r["rule_id"] == rule().id)["joint"] == 0
    assert json.loads(json.dumps(result, allow_nan=False)) == result


def test_lag_elapsed_days_are_reported_separately_by_period():
    data = frame(
        [1, 2, 3, 4], [1, 2, 3, 4], ["2022-11-01", "2022-12-01", "2023-01-01", "2023-03-01"]
    )
    result = cv.temporal_validation({"pilot": data}, [rule(lag=1)])
    intervals = result["coverage"][0]["lag_elapsed_days"]["1"]
    assert intervals["train"] == {"min": 30, "max": 30}
    assert intervals["evaluation"] == {"min": 59, "max": 59}


def test_spatial_candidate_library_preserves_rule_items_with_count_metrics():
    frames = {str(i): frame([1, 2, 3], [1, 2, 3]) for i in range(5)}
    result = cv.spatial_validation(frames, [[str(i)] for i in range(5)], [rule()])
    candidate = result["candidates"][0]
    assert candidate["antecedent_items"] == [["temperature_2m", "+"]]
    recovered = cv.Rule(
        tuple(tuple(i) for i in candidate["antecedent_items"]),
        candidate["consequent"][1],
        candidate["lag_steps"],
    )
    assert recovered == rule() and recovered.id == candidate["rule_id"]


def test_rule_rejects_ndvi_antecedent_and_invalid_lags():
    with pytest.raises(ValueError):
        cv.Rule((("ndvi_mean", "+"),), "+", 0)
    with pytest.raises(ValueError):
        rule(lag=-1)


def test_missing_and_nonfinite_observations_are_explicitly_rejected():
    with pytest.raises(ValueError, match="finite"):
        cv.score(frame([1, float("nan")], [1, 2]), rule())
