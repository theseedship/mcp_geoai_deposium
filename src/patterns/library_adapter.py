"""Producer → exchange adapter (FIERCE DoD lot O2, issue #20).

Maps the normalized so4gp mining result (from `so4gp_adapter.parse_discover_output` /
`mine_gradual_patterns` / `mine_temporal_patterns`) into the frozen exchange schema
(`GradualPatternRecord` / `PatternLibrary`). This is the producer→exchange adapter the
schema module flagged as the *outstanding O2 work*: it derives the consumer-facing
`antecedent`/`consequent` rule form and the signed `lag_days` from the producer's
itemset form, so today's `discover_patterns` output validates against the frozen contract.

Pure: no so4gp / duckdb / pandas import — unit-testable in the base runtime.

The `confidence` / `zones_validated` / `period` / `n_folds` / `stability` fields are left
unset here — they are filled only by cross-zone validation (palier 3), per ADR 0001 §5.
"""

from __future__ import annotations

import math
from datetime import datetime

import structlog
from pydantic import ValidationError

from .library_schema import GradualPatternRecord, PatternLibrary
from .render import render_gradual_item

logger = structlog.get_logger(__name__)


def _dominant_signed_lag(items: list[dict]) -> float:
    """The single signed lag for the exchange field = the lagged item with the largest
    |lag| (sign kept). 0.0 if there is no lagged item. Multi-lag patterns keep their full
    per-item detail in `pattern` / `variables`; this is the scalar the consumer scores on."""
    lagged = [gi for gi in items if gi.get("role") == "lagged" and gi.get("lag_days") is not None]
    if not lagged:
        return 0.0
    # deterministic: largest |lag|, tie-break toward the larger signed value (+7 over -7)
    return float(max(lagged, key=lambda gi: (abs(gi["lag_days"]), gi["lag_days"]))["lag_days"])


def _numeric_metrics(metrics: dict | None) -> dict[str, float] | None:
    """Keep only the numeric so4gp measures (density, singularity_scr, …) as extra_metrics;
    drop string-valued tokens. Returns None if nothing numeric remains."""
    if not metrics:
        return None
    out = {
        k: float(v)
        for k, v in metrics.items()
        if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)
    }
    return out or None


def record_from_normalized(
    pat: dict, *, pattern_id: int, algorithm: str, zone_id: str
) -> GradualPatternRecord:
    """Map one normalized pattern dict → a validated GradualPatternRecord."""
    items = pat.get("gradual_items") or []
    if not items:
        raise ValueError("pattern has no gradual_items")
    # parse_discover_output emits support=None when the sup= token is missing/non-numeric;
    # such a pattern is malformed — raise so the library loop can isolate/skip it.
    support = pat.get("support")
    if support is None:
        raise ValueError("pattern has no parseable support")
    is_temporal = bool(pat.get("is_temporal"))
    variables = [gi["attribute"] for gi in items]
    pattern_str = ", ".join(render_gradual_item(gi) for gi in items)

    if is_temporal and any(gi.get("role") == "target" for gi in items):
        consequent = ", ".join(
            render_gradual_item(gi) for gi in items if gi.get("role") == "target"
        )
        # lagged items only; "" for a (degenerate) target-only temporal pattern
        antecedent = ", ".join(
            render_gradual_item(gi) for gi in items if gi.get("role") != "target"
        )
    else:
        consequent = ""
        antecedent = pattern_str

    return GradualPatternRecord(
        pattern_id=pattern_id,
        antecedent=antecedent,
        consequent=consequent,
        pattern=pattern_str,
        variables=variables,
        support=support,
        is_temporal=is_temporal,
        lag_days=_dominant_signed_lag(items) if is_temporal else 0.0,
        algorithm=algorithm,
        zone_id=zone_id,
        extra_metrics=_numeric_metrics(pat.get("metrics")),
    )


def library_from_mining_result(
    result: dict,
    *,
    zone_id: str,
    algorithm: str,
    generated_at: datetime | None = None,
    strict: bool = False,
) -> PatternLibrary:
    """Map a full normalized mining result → a validated PatternLibrary.

    Args:
        result: the dict returned by `mine_gradual_patterns` / `mine_temporal_patterns`
            (or `parse_discover_output`) — must carry a "patterns" list.
        zone_id: the producing zone (the science-plane scope, e.g. "montpellier").
        algorithm: the exchange-schema algorithm key ("graank" / "tgrad" / "aco" / …) —
            pass the SUPPORTED key, not so4gp's display name ("GRAANK").
        generated_at: optional library timestamp.
        strict: if True, a single unmappable pattern aborts with the raised error.
            Default False — skip unmappable patterns (e.g. support=None, empty itemset,
            empty-attribute token) and log them, so one malformed pattern never discards
            the whole library. `pattern_id` keeps the source index (ids may have gaps).
    """
    records: list[GradualPatternRecord] = []
    for i, pat in enumerate(result.get("patterns", [])):
        try:
            records.append(
                record_from_normalized(pat, pattern_id=i, algorithm=algorithm, zone_id=zone_id)
            )
        except (ValidationError, ValueError, KeyError, TypeError) as exc:
            if strict:
                raise
            logger.warning("skipping unmappable pattern", index=i, zone_id=zone_id, error=str(exc))
    return PatternLibrary(
        zone_id=zone_id, algorithm=algorithm, generated_at=generated_at, patterns=records
    )
