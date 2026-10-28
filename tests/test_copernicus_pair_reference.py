"""Hand-countable synthetic arithmetic; no so4gp, mlxtend or pilot inputs."""

import hashlib
import json
from collections.abc import Sequence
from dataclasses import asdict
from itertools import count, repeat
from pathlib import Path

import pytest

from benchmarks import copernicus_pair_reference
from benchmarks.copernicus_pair_reference import (
    canonical_items,
    count_pattern,
    encode_ordered_pairs,
    inverse_items,
    passes_native_threshold,
    singleton_gate,
)

PLUS = (("x", "+"), ("y", "+"))


@pytest.mark.parametrize(
    ("rows", "pattern"),
    [([(1, 1), (2, 2), (3, 3)], PLUS), ([(1, 3), (2, 2), (3, 1)], (("x", "+"), ("y", "-")))],
)
def test_monotone_pairs_and_inverse_have_explicit_denominators(rows, pattern):
    encoded = encode_ordered_pairs(rows, ["x", "y"])
    counts = count_pattern(encoded, pattern)
    assert counts.hit_count == counts.unordered_pair_count == 3
    assert counts.ordered_pair_count == len(encoded.transactions) == 6
    assert counts.apriori_support == 0.5
    assert counts.native_support == 2 * counts.apriori_support == 1
    assert count_pattern(encoded, inverse_items(pattern)) == counts
    assert inverse_items(inverse_items(pattern)) == tuple(sorted(pattern))
    assert canonical_items(pattern) == canonical_items(reversed(inverse_items(pattern)))
    assert (
        count_pattern(encoded, (("x", "+"), ("y", "-" if pattern == PLUS else "+"))).hit_count == 0
    )


def test_ties_retain_empty_transactions_and_denominator():
    encoded = encode_ordered_pairs([(1, 1), (1, 1), (2, 2)], ["x", "y"])
    counts = count_pattern(encoded, PLUS)
    assert len(encoded.transactions) == 6
    assert encoded.transactions.count(frozenset()) == 2
    assert (counts.hit_count, counts.unordered_pair_count) == (2, 3)
    assert counts.apriori_support == pytest.approx(1 / 3)
    assert counts.native_support == pytest.approx(2 / 3)
    # A tie in x must not erase the transaction's strict y comparison.
    partial = encode_ordered_pairs([(1, 1), (1, 2), (2, 3)], ["x", "y"])
    assert partial.transactions[0] == frozenset({("y", "+")})
    assert count_pattern(partial, PLUS).hit_count == 2


def test_all_constant_is_evaluable_zero_support():
    encoded = encode_ordered_pairs([(1, 1)] * 3, ["x", "y"])
    assert encoded.transactions == (frozenset(),) * 6
    counts = count_pattern(encoded, PLUS)
    assert counts.hit_count == counts.apriori_support == counts.native_support == 0
    assert singleton_gate(encoded, "x") == 0
    assert passes_native_threshold(counts, 0) is False


@pytest.mark.parametrize("rows", [[], [(1, 1)]])
def test_zero_pairs_are_explicitly_non_evaluable(rows):
    encoded = encode_ordered_pairs(rows, ["x", "y"])
    counts = count_pattern(encoded, PLUS)
    assert counts.ordered_pair_count == counts.unordered_pair_count == counts.hit_count == 0
    assert counts.apriori_support is counts.native_support is None
    assert singleton_gate(encoded, "x") is None
    assert passes_native_threshold(counts, 0.5) is None


def test_exact_threshold_is_counted_then_strictly_filtered():
    encoded = encode_ordered_pairs([(1, 2), (2, 3), (3, 4), (4, 1)], ["x", "y"])
    counts = count_pattern(encoded, PLUS)
    assert (counts.hit_count, counts.unordered_pair_count) == (3, 6)
    assert counts.native_support == 0.5
    assert counts.apriori_support == 0.25
    assert passes_native_threshold(counts, 0.5) is False
    assert passes_native_threshold(counts, 0.49) is True


def test_source_derived_singleton_gate_exposes_row_order_artifact():
    alternating = encode_ordered_pairs([(1, 1), (2, 2), (1, 1)], ["x", "y"])
    grouped = encode_ordered_pairs([(1, 1), (1, 1), (2, 2)], ["x", "y"])
    assert count_pattern(alternating, PLUS) == count_pattern(grouped, PLUS)
    assert singleton_gate(alternating, "x") == pytest.approx(1 / 3)
    assert singleton_gate(grouped, "x") == pytest.approx(2 / 3)
    assert passes_native_threshold(count_pattern(alternating, PLUS), 0.5) is True
    assert singleton_gate(alternating, "x") < 0.5 < singleton_gate(grouped, "x")


@pytest.mark.parametrize(
    ("rows", "columns"),
    [
        ([(1,)], ["x", "y"]),
        ([(True,)], ["x"]),
        ([("1",)], ["x"]),
        ([(float("nan"),)], ["x"]),
        ([(float("inf"),)], ["x"]),
        ([(10**10000,)], ["x"]),
        ([(1, 2)], ["x", "x"]),
        ([(1,)], [""]),
        ([(1,)], [1]),
        ([], []),
        ([(1,)] * 65, ["x"]),
        ([], [str(k) for k in range(17)]),
    ],
)
def test_invalid_input_is_refused(rows, columns):
    with pytest.raises(ValueError):
        encode_ordered_pairs(rows, columns)


@pytest.mark.parametrize(
    "items",
    [
        (),
        (("unknown", "+"),),
        (("x", "+"), ("x", "-")),
        (("x", "+"), ("x", "+")),
        (("x", "?"),),
        ("x+",),
    ],
)
def test_invalid_itemsets_are_refused(items):
    with pytest.raises(ValueError):
        count_pattern(encode_ordered_pairs([(1,), (2,)], ["x"]), items)


@pytest.mark.parametrize(
    "threshold", [True, "0.5", -0.1, 1.1, float("nan"), pytest.param(10**10000, id="huge")]
)
def test_invalid_thresholds_are_refused(threshold):
    counts = count_pattern(encode_ordered_pairs([(1,), (2,)], ["x"]), (("x", "+"),))
    with pytest.raises(ValueError):
        passes_native_threshold(counts, threshold)


def test_unknown_singleton_column_is_refused():
    with pytest.raises(ValueError, match="Unknown column"):
        singleton_gate(encode_ordered_pairs([(1,)], ["x"]), "unknown")


class UntouchedSequence(Sequence):
    """Length-only sentinel: oversized inputs must fail before reading values."""

    def __init__(self, size):
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        raise AssertionError("Oversized sequence was accessed")

    def __iter__(self):
        raise AssertionError("Oversized sequence was iterated")


@pytest.mark.parametrize(
    ("rows", "columns"),
    [
        (UntouchedSequence(10**6), ["x"]),
        ([], UntouchedSequence(10**6)),
        ([UntouchedSequence(10**6)], ["x"]),
    ],
)
def test_oversized_shapes_fail_before_materialization(rows, columns):
    with pytest.raises(ValueError):
        encode_ordered_pairs(rows, columns)


@pytest.mark.parametrize(
    ("rows", "columns"),
    [("12", ["x"]), (b"12", ["x"]), ([], "x"), ([], b"x"), (["1"], ["x"]), ([b"1"], ["x"])],
)
def test_text_containers_are_refused(rows, columns):
    with pytest.raises(ValueError):
        encode_ordered_pairs(rows, columns)


@pytest.mark.parametrize("items", [UntouchedSequence(10**6), repeat(("x", "+"))])
def test_oversized_and_infinite_patterns_are_refused(items):
    with pytest.raises(ValueError, match="at most 16"):
        inverse_items(items)


def test_pattern_iterator_consumption_stops_at_seventeen():
    consumed = []
    items = (consumed.append(k) or ("x", "+") for k in count())
    with pytest.raises(ValueError, match="at most 16"):
        canonical_items(items)
    assert consumed == list(range(17))


def test_supplied_integer_comparisons_do_not_silently_cast_to_float64():
    # float64 would merge these two values; this mathematical reference must not.
    encoded = encode_ordered_pairs([(2**53,), (2**53 + 1,)], ["x"])
    counts = count_pattern(encoded, (("x", "+"),))
    assert counts.hit_count == 1
    assert counts.native_support == 1


def test_published_synthetic_receipt_replays_and_matches_helper():
    root = Path(__file__).resolve().parents[1]
    receipt = json.loads(
        (
            root / "tests/fixtures/db-comparison-preparation-2026-10-21/synthetic-receipt.json"
        ).read_text(encoding="utf-8")
    )
    helper_source = Path(copernicus_pair_reference.__file__).read_bytes()
    assert hashlib.sha256(helper_source).hexdigest() == receipt["implementation_sha256"]
    assert receipt["schema_version"] == "geoai-db-synthetic-reference/v1"
    for flag in (
        "native_mining_executed",
        "alternative_mining_executed",
        "pilot_inputs_read",
        "scientific_acceptance",
    ):
        assert receipt[flag] is False
    assert len(receipt["cases"]) == 5
    assert {case["case"] for case in receipt["cases"]} == {
        "tie_reorder_a",
        "tie_reorder_b",
        "monotone",
        "constant",
        "exact_threshold",
    }
    replayed = []
    for case in receipt["cases"]:
        encoded = encode_ordered_pairs(case["synthetic_rows"], ["x", "y"])
        counts = count_pattern(encoded, PLUS)
        result = {
            "case": case["case"],
            "synthetic_rows": case["synthetic_rows"],
            "reference": asdict(counts),
            "source_derived_candidate_gt_0_5": passes_native_threshold(counts, 0.5),
        }
        if case["case"] == "exact_threshold":
            result["inclusive_transaction_ge_0_25"] = counts.apriori_support >= 0.25
        else:
            result["source_derived_singleton_gate_x"] = singleton_gate(encoded, "x")
        replayed.append(result)
    # Python considers True == 1; serialized JSON preserves their distinct types.
    assert json.dumps(replayed, sort_keys=True, allow_nan=False) == json.dumps(
        receipt["cases"], sort_keys=True, allow_nan=False
    )
