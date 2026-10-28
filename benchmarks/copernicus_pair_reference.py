"""Synthetic preparation for strict, contemporaneous gradual-item support.

No miner or dataset loader is provided. For every i != j, a transaction contains
(column, '+') if x[j] > x[i], or '-' if x[j] < x[i]. Ties contribute neither
item; even empty transactions remain in the n(n-1) denominator. Thus the strict
pair support H / C(n, 2) equals twice transaction support H / (n(n-1)).

The singleton diagnostic and strict threshold reflect inspected so4gp 1.1.2
source, not executed mining: data_gp.py:348-385 and
algorithms/base/graank_alg.py:86-90. Source hashes and scope are recorded in
docs/research/db-comparison-preparation-2026-10-21/source-receipt.json and
protocol.md beside it. Comparisons preserve supplied numeric values without
float64 coercion; this is not bitwise native-engine parity. These helpers do
not reproduce full native search, subset removal, descriptors or confidence.
"""

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from itertools import islice
from math import isfinite
from numbers import Real

Item = tuple[str, str]
MAX_ROWS = 64
MAX_COLUMNS = 16


@dataclass(frozen=True)
class PairEncoding:
    columns: tuple[str, ...]
    row_count: int
    transactions: tuple[frozenset[Item], ...]
    upper_counts: tuple[int, ...]
    lower_counts: tuple[int, ...]


@dataclass(frozen=True)
class SupportCounts:
    hit_count: int
    ordered_pair_count: int
    unordered_pair_count: int
    apriori_support: float | None
    native_support: float | None


def encode_ordered_pairs(rows: Sequence[Sequence[Real]], columns: Sequence[str]) -> PairEncoding:
    """Encode at most 64 rows and 16 columns; preserve input order and tie rows."""
    if not isinstance(columns, Sequence) or isinstance(columns, (str, bytes, bytearray)):
        raise ValueError("Columns must be a non-text sequence")
    if not 1 <= len(columns) <= MAX_COLUMNS:
        raise ValueError(f"Expected 1 to {MAX_COLUMNS} columns")
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes, bytearray)):
        raise ValueError("Rows must be a non-text sequence")
    if len(rows) > MAX_ROWS:
        raise ValueError(f"Expected at most {MAX_ROWS} rows")
    names = tuple(columns[k] for k in range(len(columns)))
    if any(not isinstance(name, str) or not name for name in names):
        raise ValueError("Column names must be nonempty strings")
    if len(set(names)) != len(names):
        raise ValueError("Column names must be unique")
    data = []
    for i in range(len(rows)):
        row = rows[i]
        if not isinstance(row, Sequence) or isinstance(row, (str, bytes, bytearray)):
            raise ValueError("Each row must be a non-text sequence")
        if len(row) != len(names):
            raise ValueError("Rows must be rectangular and match the columns")
        values = tuple(row[k] for k in range(len(names)))
        for value in values:
            if isinstance(value, bool) or not isinstance(value, Real):
                raise ValueError("Cells must be finite real numbers, excluding bool")
            try:
                finite = isfinite(value)
            except (OverflowError, ValueError) as error:
                raise ValueError("Cells must be finite representable real numbers") from error
            if not finite:
                raise ValueError("Cells must be finite real numbers")
        data.append(values)

    upper, lower = [0] * len(names), [0] * len(names)
    transactions = []
    for i, left in enumerate(data):
        for j, right in enumerate(data):
            if i == j:
                continue
            items = set()
            for k, name in enumerate(names):
                if right[k] > left[k]:
                    items.add((name, "+"))
                    if i < j:
                        upper[k] += 1
                    else:
                        lower[k] += 1
                elif right[k] < left[k]:
                    items.add((name, "-"))
            transactions.append(frozenset(items))
    return PairEncoding(names, len(data), tuple(transactions), tuple(upper), tuple(lower))


def _validated_items(items: Iterable[Item]) -> tuple[Item, ...]:
    if isinstance(items, (str, bytes, bytearray)):
        raise ValueError("Items must be a non-text iterable")
    if isinstance(items, Sequence) and len(items) > MAX_COLUMNS:
        raise ValueError(f"A pattern must contain at most {MAX_COLUMNS} items")
    pattern = tuple(islice(items, MAX_COLUMNS + 1))
    if len(pattern) > MAX_COLUMNS:
        raise ValueError(f"A pattern must contain at most {MAX_COLUMNS} items")
    if not pattern:
        raise ValueError("A pattern must contain at least one item")
    seen = set()
    for item in pattern:
        if not isinstance(item, tuple) or len(item) != 2:
            raise ValueError("Items must be (column, direction) tuples")
        name, sign = item
        if not isinstance(name, str) or not name or sign not in ("+", "-"):
            raise ValueError("Items need a nonempty column and '+' or '-' direction")
        if name in seen:
            raise ValueError("Duplicate or contradictory items for one column")
        seen.add(name)
    return tuple(sorted(pattern))


def inverse_items(items: Iterable[Item]) -> tuple[Item, ...]:
    """Invert every sign; reversing a row pair gives this inverse transaction."""
    return tuple((name, "-" if sign == "+" else "+") for name, sign in _validated_items(items))


def canonical_items(items: Iterable[Item]) -> tuple[Item, ...]:
    """Use the smaller sorted signature modulo simultaneous sign inversion."""
    pattern = _validated_items(items)
    return min(pattern, inverse_items(pattern))


def count_pattern(encoding: PairEncoding, items: Iterable[Item]) -> SupportCounts:
    """Count one itemset exactly; zero pairs have null support, not zero support.

    ``native_support`` names the source-derived denominator conversion only;
    it does not claim a corresponding native pattern was discovered.
    """
    pattern = frozenset(_validated_items(items))
    if any(name not in encoding.columns for name, _ in pattern):
        raise ValueError("Pattern contains an unknown column")
    hits = sum(pattern <= transaction for transaction in encoding.transactions)
    ordered = len(encoding.transactions)
    unordered = encoding.row_count * (encoding.row_count - 1) // 2
    return SupportCounts(
        hits,
        ordered,
        unordered,
        hits / ordered if ordered else None,
        hits / unordered if unordered else None,
    )


def singleton_gate(encoding: PairEncoding, column: str) -> float | None:
    """Source-derived max(upper, lower)/C(n,2); row-order diagnostic, not mining."""
    if column not in encoding.columns:
        raise ValueError("Unknown column")
    denominator = encoding.row_count * (encoding.row_count - 1) // 2
    if not denominator:
        return None
    k = encoding.columns.index(column)
    return max(encoding.upper_counts[k], encoding.lower_counts[k]) / denominator


def passes_native_threshold(counts: SupportCounts, min_support: float) -> bool | None:
    """Apply the inspected strict > candidate filter separately from counting.

    This is not a full native acceptance test; singleton pruning and search
    output conventions are deliberately not applied here.
    """
    if isinstance(min_support, bool) or not isinstance(min_support, Real):
        raise ValueError("min_support must be finite and between zero and one")
    try:
        valid = isfinite(min_support) and 0 <= min_support <= 1
    except (OverflowError, ValueError) as error:
        raise ValueError("min_support must be finite and between zero and one") from error
    if not valid:
        raise ValueError("min_support must be finite and between zero and one")
    return None if counts.native_support is None else counts.native_support > min_support
