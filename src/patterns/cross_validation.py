"""Cross-zone validation helpers for gradual pattern libraries.

This module is intentionally independent from the MCP layer and from DuckDB I/O:
callers pass normalized so4gp mining results per zone, and receive a validated
``PatternLibrary`` plus fold metadata. The current Palier-3 scope fills O3
provenance fields (zones, folds, stability) but still leaves calibrated
``confidence`` unset for Palier 4.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from datetime import UTC, date, datetime
from typing import Any

from src.patterns.library_adapter import record_from_normalized
from src.patterns.library_schema import GradualPatternRecord, PatternLibrary


def make_zone_folds(zone_ids: list[str], n_folds: int) -> list[list[str]]:
    """Build deterministic round-robin cross-zone folds.

    The caller controls zone order. For a 15-zone panel and ``n_folds=5``, this
    yields five folds of three validation zones each.
    """
    if n_folds < 2:
        raise ValueError("n_folds must be >= 2")
    if len(zone_ids) < n_folds:
        raise ValueError("n_folds must be <= number of zones")
    folds: list[list[str]] = [[] for _ in range(n_folds)]
    for idx, zone_id in enumerate(zone_ids):
        folds[idx % n_folds].append(zone_id)
    return folds


def _finite_support(value: Any) -> float | None:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return None
    support = float(value)
    if not math.isfinite(support):
        return None
    return support


def _canonical_item(item: Mapping[str, Any], *, temporal: bool) -> tuple:
    attr = str(item["attribute"])
    direction = str(item["direction"])
    if not temporal:
        return (attr, direction)
    lag = item.get("lag_days")
    lag_value = 0.0 if lag is None else round(float(lag), 6)
    return (str(item.get("role") or ""), attr, direction, lag_value)


def canonical_pattern_key(pattern: Mapping[str, Any], *, algorithm: str) -> str:
    """Stable identity for the same gradual pattern across zones.

    GRAANK itemsets are order-insensitive, so their item tuples are sorted.
    Temporal patterns include role and lag in the identity.
    """
    items = pattern.get("gradual_items") or []
    if not items:
        raise ValueError("pattern has no gradual_items")
    temporal = bool(pattern.get("is_temporal"))
    item_keys = sorted(_canonical_item(item, temporal=temporal) for item in items)
    payload = {
        "algorithm": algorithm.lower(),
        "is_temporal": temporal,
        "items": item_keys,
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _zone_ordered_subset(zone_ids: list[str], present: set[str]) -> list[str]:
    return [zone_id for zone_id in zone_ids if zone_id in present]


def _record_with_validation(
    base_pattern: Mapping[str, Any],
    *,
    pattern_id: int,
    algorithm: str,
    library_zone_id: str,
    support: float,
    zones_validated: list[str],
    period_start: date | None,
    period_end: date | None,
    n_folds: int,
    stability: float,
    extra_metrics: dict[str, float],
    generated_at: datetime,
) -> GradualPatternRecord:
    base = record_from_normalized(
        dict(base_pattern),
        pattern_id=pattern_id,
        algorithm=algorithm,
        zone_id=library_zone_id,
    )
    merged_metrics: dict[str, float] = {}
    if base.extra_metrics:
        merged_metrics.update(base.extra_metrics)
    merged_metrics.update(extra_metrics)
    data = base.model_dump()
    data.update(
        {
            "support": support,
            "zones_validated": zones_validated,
            "period_start": period_start,
            "period_end": period_end,
            "n_folds": n_folds,
            "stability": stability,
            "created_at": generated_at,
            "extra_metrics": merged_metrics or None,
        }
    )
    return GradualPatternRecord.model_validate(data)


def cross_validate_mined_patterns(
    mined_by_zone: Mapping[str, Mapping[str, Any]],
    *,
    zone_ids: list[str],
    algorithm: str,
    min_support: float,
    n_folds: int = 5,
    min_validated_zones: int = 1,
    period_start: date | None = None,
    period_end: date | None = None,
    library_zone_id: str = "cross_zone",
    generated_at: datetime | None = None,
) -> tuple[PatternLibrary, list[dict[str, Any]], dict[str, Any]]:
    """Aggregate per-zone mining output into a cross-zone validated library.

    A candidate pattern is mined independently per zone. For each fold, the
    validation zones are held out. A pattern validates for a fold only if it is
    present in at least one training zone and at least one validation zone. The
    record ``stability`` is the share of folds that validate; ``zones_validated``
    lists the held-out zones where the pattern was observed.
    """
    if not 0.0 <= min_support <= 1.0:
        raise ValueError("min_support must be between 0 and 1")
    if min_validated_zones < 1:
        raise ValueError("min_validated_zones must be >= 1")
    if len(set(zone_ids)) != len(zone_ids):
        raise ValueError("zone_ids must be unique")
    missing = [zone_id for zone_id in zone_ids if zone_id not in mined_by_zone]
    if missing:
        raise ValueError(f"missing mined results for zones: {missing}")

    folds = make_zone_folds(zone_ids, n_folds)
    generated_at = generated_at or datetime.now(UTC)

    by_key: dict[str, dict[str, Any]] = {}
    skipped_patterns = 0
    for zone_id in zone_ids:
        result = mined_by_zone[zone_id]
        for pat in result.get("patterns", []) or []:
            support = _finite_support(pat.get("support"))
            if support is None or support < min_support:
                skipped_patterns += 1
                continue
            try:
                key = canonical_pattern_key(pat, algorithm=algorithm)
            except (KeyError, TypeError, ValueError):
                skipped_patterns += 1
                continue
            bucket = by_key.setdefault(
                key,
                {
                    "pattern": pat,
                    "supports": {},
                },
            )
            bucket["supports"][zone_id] = support

    fold_rows: list[dict[str, Any]] = []
    for idx, validation_zones in enumerate(folds, start=1):
        train_zones = [zone_id for zone_id in zone_ids if zone_id not in validation_zones]
        fold_rows.append(
            {
                "fold": idx,
                "train_zones": train_zones,
                "validation_zones": validation_zones,
            }
        )

    ranked: list[dict[str, Any]] = []
    for key, bucket in by_key.items():
        supports: dict[str, float] = bucket["supports"]
        present_zones = set(supports)
        zones_validated: set[str] = set()
        validated_folds = 0
        for validation_zones in folds:
            validation_set = set(validation_zones)
            train_set = set(zone_ids) - validation_set
            train_hits = present_zones & train_set
            validation_hits = present_zones & validation_set
            if train_hits and validation_hits:
                validated_folds += 1
                zones_validated.update(validation_hits)

        ordered_validated = _zone_ordered_subset(zone_ids, zones_validated)
        if len(ordered_validated) < min_validated_zones or validated_folds == 0:
            continue
        validated_supports = [supports[zone_id] for zone_id in ordered_validated]
        all_supports = list(supports.values())
        stability = validated_folds / n_folds
        ranked.append(
            {
                "key": key,
                "pattern": bucket["pattern"],
                "support": _mean(validated_supports),
                "zones_validated": ordered_validated,
                "stability": stability,
                "validated_folds": validated_folds,
                "zone_coverage": len(ordered_validated) / len(zone_ids),
                "mean_support_all_zones": _mean(all_supports),
                "min_support_all_zones": min(all_supports),
                "max_support_all_zones": max(all_supports),
            }
        )

    ranked.sort(
        key=lambda row: (
            -row["stability"],
            -len(row["zones_validated"]),
            -row["support"],
            row["key"],
        )
    )

    records: list[GradualPatternRecord] = []
    for idx, row in enumerate(ranked):
        records.append(
            _record_with_validation(
                row["pattern"],
                pattern_id=idx,
                algorithm=algorithm,
                library_zone_id=library_zone_id,
                support=row["support"],
                zones_validated=row["zones_validated"],
                period_start=period_start,
                period_end=period_end,
                n_folds=n_folds,
                stability=row["stability"],
                generated_at=generated_at,
                extra_metrics={
                    "validated_folds": float(row["validated_folds"]),
                    "zone_coverage": float(row["zone_coverage"]),
                    "mean_support_all_zones": float(row["mean_support_all_zones"]),
                    "min_support_all_zones": float(row["min_support_all_zones"]),
                    "max_support_all_zones": float(row["max_support_all_zones"]),
                },
            )
        )

    stats = {
        "n_zones": len(zone_ids),
        "n_folds": n_folds,
        "n_candidate_patterns": len(by_key),
        "n_validated_patterns": len(records),
        "skipped_patterns": skipped_patterns,
        "min_support": min_support,
        "min_validated_zones": min_validated_zones,
    }
    library = PatternLibrary(
        zone_id=library_zone_id,
        algorithm=algorithm,
        generated_at=generated_at,
        patterns=records,
    )
    return library, fold_rows, stats


_VALIDATED_PATTERN_COLUMNS = [
    "pattern_id",
    "antecedent",
    "consequent",
    "pattern",
    "variables",
    "support",
    "confidence",
    "is_temporal",
    "lag_days",
    "algorithm",
    "zone_id",
    "zones_validated",
    "period_start",
    "period_end",
    "n_folds",
    "stability",
    "created_at",
    "extra_metrics",
]


def validated_library_to_dataframe(library: PatternLibrary):
    """Flatten a validated library into a typed DataFrame for Parquet output."""
    import pandas as pd

    rows = []
    for record in library.patterns:
        rows.append(
            {
                "pattern_id": record.pattern_id,
                "antecedent": record.antecedent,
                "consequent": record.consequent,
                "pattern": record.pattern,
                "variables": json.dumps(record.variables),
                "support": record.support,
                "confidence": record.confidence,
                "is_temporal": record.is_temporal,
                "lag_days": record.lag_days,
                "algorithm": record.algorithm,
                "zone_id": record.zone_id,
                "zones_validated": json.dumps(record.zones_validated),
                "period_start": record.period_start.isoformat() if record.period_start else None,
                "period_end": record.period_end.isoformat() if record.period_end else None,
                "n_folds": record.n_folds,
                "stability": record.stability,
                "created_at": record.created_at.isoformat() if record.created_at else None,
                "extra_metrics": json.dumps(record.extra_metrics or {}),
            }
        )
    return pd.DataFrame(rows, columns=_VALIDATED_PATTERN_COLUMNS)
