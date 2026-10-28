"""Hand-counted comparison contracts; no miners, pilot files or providers."""

import pytest

from benchmarks.copernicus_compare import compare_outputs, parse_native_patterns


def identity(result, items):
    """Select a reported identity without reproducing comparison arithmetic."""
    matches = [row for row in result["identities"] if row["items"] == items]
    assert len(matches) == 1
    return matches[0]


@pytest.mark.parametrize(
    "metadata",
    [
        ["sup=0.667, conf=1.0"],
        ["sup=0.667", "conf=1.0", "density=0.667", "connect=1"],
    ],
)
def test_native_parser_accepts_both_actual_descriptor_shapes(metadata):
    raw_row = ["y-, x+", *metadata]
    records, errors = parse_native_patterns({"Patterns": [raw_row]}, ["x", "y"])

    assert errors == []
    assert len(records) == 1
    assert records[0]["raw_index"] == 0
    assert set(records[0]["items"]) == {("x", "+"), ("y", "-")}
    assert records[0]["native_support"] == 0.667
    assert records[0]["raw_row"] == raw_row


@pytest.mark.parametrize(
    "raw_row",
    [
        ["x+, unknown+", "sup=1.0, conf=1.0"],
        ["x+, x-", "sup=1.0, conf=1.0"],
        ["x+, x+", "sup=1.0, conf=1.0"],
        ["x?, y+", "sup=1.0, conf=1.0"],
        ["x+, y+", "conf=1.0"],
        ["x+, y+", "sup=not-a-number, conf=1.0"],
        ["x+, y+", "sup=nan, conf=1.0"],
        ["x+, y+", "sup=inf, conf=1.0"],
        [],
    ],
)
def test_invalid_native_row_keeps_raw_position_and_does_not_erase_valid_rows(raw_row):
    valid = ["x+, y+", "sup=1.0, conf=1.0"]
    records, errors = parse_native_patterns({"Patterns": [valid, raw_row]}, ["x", "y"])

    assert len(records) == len(errors) == 1
    assert records[0]["raw_index"] == 0
    assert records[0]["raw_row"] == valid
    assert errors[0]["raw_index"] == 1
    assert errors[0]["raw_row"] == raw_row


def test_common_monotone_identity_has_two_explicit_denominators():
    result = compare_outputs(
        [[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]],
        ["x", "y"],
        {"Patterns": [["x+, y+", "sup=1.0, conf=1.0"]]},
        {"itemsets": [{"items": ["x+", "y+"], "support": 0.5}]},
    )

    row = identity(result, ["x+", "y+"])
    assert result["status"] == "complete"
    assert row["reference"] == {
        "hit_count": 3,
        "ordered_pair_count": 6,
        "unordered_pair_count": 3,
        "apriori_support": 0.5,
        "native_support": 1.0,
    }
    assert row["strict"] is True
    assert row["disposition"] == "common"
    assert row["native_raw_indexes"] == [0]
    assert row["apriori_raw_indexes"] == [0]


def test_inverse_outputs_share_one_identity_without_adding_hits_or_support():
    result = compare_outputs(
        [[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]],
        ["x", "y"],
        {
            "Patterns": [
                ["x+, y+", "sup=1.0, conf=1.0"],
                ["y-, x-", "sup=1.0, conf=1.0"],
            ]
        },
        {
            "itemsets": [
                {"items": ["x+", "y+"], "support": 0.5},
                {"items": ["y-", "x-"], "support": 0.5},
            ]
        },
    )

    assert len(result["identities"]) == 1
    row = identity(result, ["x+", "y+"])
    assert row["reference"]["hit_count"] == 3
    assert row["reference"]["apriori_support"] == 0.5
    assert row["reference"]["native_support"] == 1.0
    assert row["native_raw_indexes"] == row["apriori_raw_indexes"] == [0, 1]


def test_tie_transactions_keep_denominator_and_native_rounding_is_checked_separately():
    result = compare_outputs(
        [[1.0, 1.0], [1.0, 1.0], [2.0, 2.0]],
        ["x", "y"],
        {
            "Patterns": [
                ["x+, y+", "sup=0.667, conf=1.0"],
                ["x-, y-", "sup=0.666, conf=1.0"],
            ]
        },
        {"itemsets": [{"items": ["x+", "y+"], "support": 1 / 3}]},
    )

    row = identity(result, ["x+", "y+"])
    assert row["reference"]["hit_count"] == 2
    assert row["reference"]["ordered_pair_count"] == 6
    assert row["reference"]["unordered_pair_count"] == 3
    assert row["reference"]["apriori_support"] == pytest.approx(1 / 3)
    assert row["reference"]["native_support"] == pytest.approx(2 / 3)
    assert row["native_support_rounding_mismatches"] == [1]
    assert row["apriori_support_mismatches"] == []


def test_apriori_raw_support_mismatch_is_visible_after_independent_recount():
    result = compare_outputs(
        [[1.0, 1.0], [1.0, 1.0], [2.0, 2.0]],
        ["x", "y"],
        {"Patterns": [["x+, y+", "sup=0.667, conf=1.0"]]},
        {"itemsets": [{"items": ["x+", "y+"], "support": 0.5}]},
    )

    row = identity(result, ["x+", "y+"])
    assert row["reference"]["hit_count"] == 2
    assert row["reference"]["apriori_support"] == pytest.approx(1 / 3)
    assert row["apriori_support_mismatches"] == [0]
    assert row["native_support_rounding_mismatches"] == []


def test_apriori_inclusive_boundary_is_retained_and_excluded_from_strict_results():
    result = compare_outputs(
        [[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 1.0]],
        ["x", "y"],
        {"Patterns": []},
        {"itemsets": [{"items": ["x+", "y+"], "support": 0.25}]},
    )

    row = identity(result, ["x+", "y+"])
    assert row["reference"]["hit_count"] == 3
    assert row["reference"]["ordered_pair_count"] == 12
    assert row["reference"]["unordered_pair_count"] == 6
    assert row["reference"]["apriori_support"] == 0.25
    assert row["reference"]["native_support"] == 0.5
    assert row["strict"] is False
    assert row["disposition"] == "boundary"
    assert row["apriori_raw_indexes"] == [0]


@pytest.mark.parametrize(
    ("rows", "pruned", "disposition"),
    [
        (
            [[1.0, 1.0], [2.0, 2.0], [1.0, 1.0]],
            ["x", "y"],
            "apriori_only_singleton_gate",
        ),
        (
            [[1.0, 1.0], [1.0, 1.0], [2.0, 2.0]],
            [],
            "apriori_only_unexplained",
        ),
    ],
)
def test_reordering_changes_gate_explanation_without_changing_joint_count(
    rows, pruned, disposition
):
    result = compare_outputs(
        rows,
        ["x", "y"],
        {"Patterns": []},
        {"itemsets": [{"items": ["x+", "y+"], "support": 1 / 3}]},
    )

    row = identity(result, ["x+", "y+"])
    assert row["reference"]["hit_count"] == 2
    assert row["reference"]["native_support"] == pytest.approx(2 / 3)
    assert row["strict"] is True
    assert row["singleton_pruned"] == pruned
    assert row["disposition"] == disposition
    assert row["native_superset_witnesses"] == []


def test_returned_superset_witness_checks_inverse_after_canonicalization():
    result = compare_outputs(
        [[1.0, 3.0, 3.0], [2.0, 2.0, 2.0], [3.0, 1.0, 1.0]],
        ["a", "b", "c"],
        {"Patterns": [["a+, b-, c-", "sup=1.0, conf=1.0"]]},
        {"itemsets": [{"items": ["b+", "c+"], "support": 0.5}]},
    )

    row = identity(result, ["b+", "c+"])
    assert row["strict"] is True
    assert row["singleton_pruned"] == []
    assert row["native_superset_witnesses"] == [0]
    assert row["disposition"] == "apriori_only_returned_superset"


def test_singleton_gate_equal_to_threshold_is_not_a_pruning_explanation():
    # Three upper and three lower comparisons give a gate of exactly 0.5;
    # the two features still agree on all six unordered pairs.
    result = compare_outputs(
        [[1.0, 1.0], [4.0, 4.0], [3.0, 3.0], [2.0, 2.0]],
        ["x", "y"],
        {"Patterns": []},
        {"itemsets": [{"items": ["x+", "y+"], "support": 0.5}]},
    )

    row = identity(result, ["x+", "y+"])
    assert row["reference"]["hit_count"] == 6
    assert row["reference"]["native_support"] == 1.0
    assert row["singleton_pruned"] == []
    assert row["disposition"] == "apriori_only_unexplained"


def test_independent_reference_keeps_strict_identity_missing_from_both_methods():
    result = compare_outputs(
        [[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]],
        ["x", "y"],
        {"Patterns": []},
        {"itemsets": []},
    )

    row = identity(result, ["x+", "y+"])
    assert row["strict"] is True
    assert row["disposition"] == "reference_only"
    assert row["native_raw_indexes"] == row["apriori_raw_indexes"] == []
    assert len(result["reference_rows"]) == 2


def test_reference_covers_all_canonical_signs_and_seven_feature_itemset_sizes():
    result = compare_outputs(
        [[1.0] * 7, [2.0] * 7],
        ["a", "b", "c", "d", "e", "f", "g"],
        {"Patterns": []},
        {"itemsets": []},
    )

    # Sum(C(7,k) * 2**(k-1), k=2..7) = (3**7 - 1 - 14) / 2.
    assert len(result["reference_rows"]) == 1086
    # Only the all-aligned orientation qualifies for each chosen feature subset.
    assert len(result["identities"]) == 120
    assert {len(row["items"]) for row in result["identities"]} == {2, 3, 4, 5, 6, 7}
    row = identity(result, ["a+", "b+", "c+", "d+", "e+", "f+", "g+"])
    assert row["reference"]["hit_count"] == 1
    assert row["reference"]["ordered_pair_count"] == 2
    assert row["reference"]["native_support"] == 1.0


def test_native_only_qualifying_identity_is_not_hidden_by_empty_apriori_output():
    result = compare_outputs(
        [[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]],
        ["x", "y"],
        {"Patterns": [["x+, y+", "sup=1.0, conf=1.0"]]},
        {"itemsets": []},
    )

    row = identity(result, ["x+", "y+"])
    assert row["disposition"] == "native_only"
    assert row["native_raw_indexes"] == [0]
    assert row["apriori_raw_indexes"] == []


def test_native_error_is_incomplete_even_when_patterns_is_empty():
    result = compare_outputs(
        [[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]],
        ["x", "y"],
        {"Error": "Search space is empty!", "Patterns": []},
        {"itemsets": []},
    )

    assert result["status"] == "incomplete"


def test_invalid_native_output_makes_comparison_parse_error():
    result = compare_outputs(
        [[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]],
        ["x", "y"],
        {"Patterns": [["x+, unknown+", "sup=1.0, conf=1.0"]]},
        {"itemsets": []},
    )

    assert result["status"] == "parse_error"


def test_apriori_singletons_keep_raw_evidence_without_entering_multivariate_comparison():
    singleton = {"items": ["x+"], "support": 0.5}
    result = compare_outputs(
        [[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]],
        ["x", "y"],
        {"Patterns": []},
        {
            "itemsets": [
                singleton,
                {"items": ["y+"], "support": 0.5},
                {"items": ["x+", "y+"], "support": 0.5},
            ]
        },
    )

    assert result["status"] == "complete"
    assert len(result["identities"]) == 1
    assert result["summary"]["apriori_records"] == 1
    assert result["summary"]["apriori_ignored_singletons"] == 2
    assert result["apriori_ignored_singletons"][0]["raw_row"] == singleton
    assert identity(result, ["x+", "y+"])["apriori_raw_indexes"] == [2]


@pytest.mark.parametrize("items", [["x+", "x-"], ["x+", "x+"], ["x+", "unknown+"]])
def test_invalid_apriori_multi_itemset_preserves_error_and_valid_raw_indexes(items):
    invalid = {"items": items, "support": 0.5}
    result = compare_outputs(
        [[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]],
        ["x", "y"],
        {"Patterns": [["x+, y+", "sup=1.0, conf=1.0"]]},
        {"itemsets": [{"items": ["x+", "y+"], "support": 0.5}, invalid]},
    )

    assert result["status"] == "parse_error"
    assert result["summary"]["parse_errors"] == 1
    assert result["apriori_parse_errors"][0]["raw_index"] == 1
    assert result["apriori_parse_errors"][0]["raw_row"] == invalid
    assert identity(result, ["x+", "y+"])["apriori_raw_indexes"] == [0]
