"""Pure, independent analysis of bounded signed-pair mining outputs.

Neither miner is imported or executed here. Raw observations remain in their
supplied order. Both mining consumers must receive that same finite float64
matrix; the caller records its provenance and enforces execution limits.
"""

from collections import Counter, defaultdict
from dataclasses import asdict
from itertools import combinations, product
from math import isclose, isfinite

from benchmarks.copernicus_pair_reference import (
    Item,
    canonical_items,
    count_pattern,
    encode_ordered_pairs,
    inverse_items,
    singleton_gate,
)

MAX_FEATURES = 7
NATIVE_THRESHOLD = 0.5
SUPPORT_TOLERANCE = 1e-12


def _parse_items(labels: object, columns: list[str]) -> tuple[Item, ...]:
    if not isinstance(labels, (list, tuple)) or not labels:
        raise ValueError("Expected a nonempty list of signed feature labels")
    if len(labels) > MAX_FEATURES:
        raise ValueError("Itemset exceeds the agreed seven-feature scope")
    items = []
    seen = set()
    for label in labels:
        if not isinstance(label, str):
            raise ValueError("Signed feature labels must be strings")
        label = label.strip()
        if len(label) < 2 or label[-1] not in ("+", "-"):
            raise ValueError("Feature label must end with '+' or '-'")
        name, sign = label[:-1], label[-1]
        if name not in columns:
            raise ValueError(f"Unknown feature: {name}")
        if name in seen:
            raise ValueError(f"Duplicate or contradictory feature: {name}")
        seen.add(name)
        items.append((name, sign))
    return tuple(sorted(items))


def _parse_support(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (str, int, float)):
        raise ValueError("Support must be a finite numeric value")
    try:
        support = float(value)
    except (ValueError, OverflowError) as error:
        raise ValueError("Support must be a finite numeric value") from error
    if not isfinite(support) or not 0 <= support <= 1:
        raise ValueError("Support must be finite and between zero and one")
    return support


def _error(raw_index: int | None, raw_row: object, error: object) -> dict:
    return {"raw_index": raw_index, "raw_row": raw_row, "error": str(error)}


def parse_native_patterns(payload: dict, columns: list[str]) -> tuple[list[dict], list[dict]]:
    """Parse both actual so4gp descriptor layouts without dropping bad rows.

    Each valid record retains its original row, original index, signed items and
    reported rounded support. Unknown, duplicate and contradictory items are
    errors; simultaneous inverse patterns remain separate raw records.
    """
    records, errors = [], []
    if not isinstance(payload, dict) or not isinstance(payload.get("Patterns"), list):
        return [], [_error(None, payload, "Expected a native Patterns list")]
    for raw_index, raw_row in enumerate(payload["Patterns"]):
        try:
            if not isinstance(raw_row, (list, tuple)) or len(raw_row) < 2:
                raise ValueError("Expected a native pattern and support descriptors")
            if not isinstance(raw_row[0], str):
                raise ValueError("Native pattern must be a string")
            items = _parse_items(raw_row[0].split(","), columns)
            if len(items) < 2:
                raise ValueError("Native itemsets must have sizes two through seven")
            supports = []
            for descriptor in raw_row[1:]:
                if not isinstance(descriptor, str):
                    raise ValueError("Native descriptors must be strings")
                for field in descriptor.split(","):
                    key, separator, value = field.strip().partition("=")
                    if separator and key.strip() == "sup":
                        supports.append(_parse_support(value.strip()))
            if len(supports) != 1:
                raise ValueError("Expected exactly one native support descriptor")
            records.append(
                {
                    "raw_index": raw_index,
                    "items": items,
                    "native_support": supports[0],
                    "raw_row": raw_row,
                }
            )
        except ValueError as error:
            errors.append(_error(raw_index, raw_row, error))
    return records, errors


def _parse_apriori_patterns(payload: dict, columns: list[str]) -> tuple[list, list, list]:
    records, errors, singletons = [], [], []
    if not isinstance(payload, dict) or not isinstance(payload.get("itemsets"), list):
        return [], [_error(None, payload, "Expected an Apriori itemsets list")], []
    for raw_index, raw_row in enumerate(payload["itemsets"]):
        try:
            if not isinstance(raw_row, dict):
                raise ValueError("Expected an Apriori itemset record")
            items = _parse_items(raw_row.get("items"), columns)
            support = _parse_support(raw_row.get("support"))
            record = {
                "raw_index": raw_index,
                "items": items,
                "apriori_support": support,
                "raw_row": raw_row,
            }
            if len(items) == 1:
                singletons.append(record)
            else:
                records.append(record)
        except ValueError as error:
            errors.append(_error(raw_index, raw_row, error))
    return records, errors, singletons


def _labels(items: tuple[Item, ...]) -> list[str]:
    return [name + sign for name, sign in items]


def _index_records(records: list[dict]) -> dict:
    indexed = defaultdict(list)
    for record in records:
        indexed[canonical_items(record["items"])].append(record)
    return indexed


def _same_support(reported: float, expected: float | None) -> bool:
    return expected is not None and isclose(
        reported, expected, rel_tol=0, abs_tol=SUPPORT_TOLERANCE
    )


def compare_outputs(
    rows: list[list[float]],
    columns: list[str],
    native_payload: dict,
    apriori_payload: dict,
) -> dict:
    """Recount the complete bounded universe and explain observed differences.

    ``reference_rows`` contains every canonical signed itemset of sizes 2--7.
    ``identities`` retains every strict reference-qualified or observed itemset.
    Native subset witnesses describe actual returned supersets; they do not
    certify native maximality or reproduce its complete search procedure.
    """
    if len(columns) > MAX_FEATURES:
        raise ValueError("The bounded comparison supports at most seven features")
    encoding = encode_ordered_pairs(rows, columns)
    native_records, native_errors = parse_native_patterns(native_payload, columns)
    apriori_records, apriori_errors, apriori_singletons = _parse_apriori_patterns(
        apriori_payload, columns
    )
    native_index = _index_records(native_records)
    apriori_index = _index_records(apriori_records)
    gates = {column: singleton_gate(encoding, column) for column in columns}
    reference_rows, identities = [], []

    for size in range(2, len(columns) + 1):
        for names in combinations(sorted(columns), size):
            # Fix the first feature to '+' to enumerate exactly one orientation.
            for signs in product(("+", "-"), repeat=size - 1):
                items = canonical_items(((names[0], "+"), *zip(names[1:], signs, strict=True)))
                counts = count_pattern(encoding, items)
                strict = counts.unordered_pair_count > 0 and (
                    2 * counts.hit_count > counts.unordered_pair_count
                )
                reference = {
                    "items": _labels(items),
                    "reference": asdict(counts),
                    "strict": strict,
                }
                reference_rows.append(reference)
                native = native_index.get(items, [])
                apriori = apriori_index.get(items, [])
                if not strict and not native and not apriori:
                    continue
                pruned = [
                    name
                    for name, _ in items
                    if gates[name] is not None and gates[name] < NATIVE_THRESHOLD
                ]
                direct, inverse = set(items), set(inverse_items(items))
                witnesses = [
                    record["raw_index"]
                    for record in native_records
                    if direct < set(record["items"]) or inverse < set(record["items"])
                ]
                native_mismatches = [
                    record["raw_index"]
                    for record in native
                    if not _same_support(
                        record["native_support"],
                        round(counts.native_support, 3)
                        if counts.native_support is not None
                        else None,
                    )
                ]
                apriori_mismatches = [
                    record["raw_index"]
                    for record in apriori
                    if not _same_support(record["apriori_support"], counts.apriori_support)
                ]
                boundary = (
                    counts.unordered_pair_count > 0
                    and 2 * counts.hit_count == counts.unordered_pair_count
                )
                if not strict:
                    disposition = "boundary" if boundary else "below_threshold"
                elif native and apriori:
                    disposition = "common"
                elif native:
                    disposition = "native_only"
                elif not apriori:
                    disposition = "reference_only"
                elif pruned:
                    disposition = "apriori_only_singleton_gate"
                elif witnesses:
                    disposition = "apriori_only_returned_superset"
                else:
                    disposition = "apriori_only_unexplained"
                identities.append(
                    {
                        **reference,
                        "native_raw_indexes": [record["raw_index"] for record in native],
                        "apriori_raw_indexes": [record["raw_index"] for record in apriori],
                        "native_support_rounding_mismatches": native_mismatches,
                        "apriori_support_mismatches": apriori_mismatches,
                        "singleton_pruned": pruned,
                        "native_superset_witnesses": witnesses,
                        "disposition": disposition,
                    }
                )

    dispositions = Counter(row["disposition"] for row in identities)
    native_mismatch_count = sum(
        len(row["native_support_rounding_mismatches"]) for row in identities
    )
    apriori_mismatch_count = sum(len(row["apriori_support_mismatches"]) for row in identities)
    method_errors = {
        "native": native_payload.get("Error") if isinstance(native_payload, dict) else None,
        "apriori": apriori_payload.get("Error") if isinstance(apriori_payload, dict) else None,
    }
    summary = {
        "native_records": len(native_records),
        "native_identities": len(native_index),
        "apriori_records": len(apriori_records),
        "apriori_identities": len(apriori_index),
        "apriori_ignored_singletons": len(apriori_singletons),
        "reference_strict_identities": sum(row["strict"] for row in reference_rows),
        "common": dispositions["common"],
        "native_only": dispositions["native_only"],
        "apriori_only": sum(
            count
            for disposition, count in dispositions.items()
            if disposition.startswith("apriori_only_")
        ),
        "singleton_gate_omissions": dispositions["apriori_only_singleton_gate"],
        "returned_superset_omissions": dispositions["apriori_only_returned_superset"],
        "reference_only": dispositions["reference_only"],
        "unexplained_omissions": (
            dispositions["apriori_only_unexplained"]
            + dispositions["native_only"]
            + dispositions["reference_only"]
        ),
        "support_mismatches": native_mismatch_count + apriori_mismatch_count,
        "native_support_rounding_mismatches": native_mismatch_count,
        "apriori_support_mismatches": apriori_mismatch_count,
        "boundary_identities": dispositions["boundary"],
        "below_threshold_identities": dispositions["below_threshold"],
        "parse_errors": len(native_errors) + len(apriori_errors),
    }
    failure_flags = []
    if summary["parse_errors"]:
        failure_flags.append("parse_errors")
    if any(method_errors.values()):
        failure_flags.append("method_errors")
    if encoding.row_count < 2:
        failure_flags.append("non_evaluable_input")
    if summary["support_mismatches"]:
        failure_flags.append("support_mismatches")
    if summary["unexplained_omissions"]:
        failure_flags.append("unexplained_omissions")
    if summary["below_threshold_identities"]:
        failure_flags.append("below_threshold_output")
    status = "complete"
    if method_errors["native"] or method_errors["apriori"] or encoding.row_count < 2:
        status = "incomplete"
    if native_errors or apriori_errors:
        status = "parse_error"
    return {
        "status": status,
        "summary": summary,
        "failure_flags": failure_flags,
        "method_errors": method_errors,
        "native_parse_errors": native_errors,
        "apriori_parse_errors": apriori_errors,
        "native_records": native_records,
        "apriori_records": apriori_records,
        "apriori_ignored_singletons": apriori_singletons,
        "singleton_gates": gates,
        "reference_rows": reference_rows,
        "identities": identities,
    }
