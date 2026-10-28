"""Frozen BEJOF <-> Owuor pattern-library exchange schema (FIERCE DoD lots O2 / kickoff #18, #20).

This is the **target contract** the so4gp producer (`discover_patterns`, paliers 1-2 + the
planned ACO/GA/PSO of palier 3) must be **adapted to write**, and that the consumer
(`pattern_alert`, palier 4) reads. The producer tool still writes its support-only per-run Parquet
(`_PATTERN_COLUMNS`); the **`library_adapter`** module bridges that output to this contract —
deriving the `antecedent`/`consequent` rule form + the signed `lag_days` — so conversion happens at
the exchange boundary (it backs the `mine_patterns` CLI). It reconciles three sources:

* the documented target schema — ``docs/design/terra-cognita-addendum.md`` §5.4
  (``pattern_id / antecedent / consequent / support / variables / lag_days / zone_id /
  algorithm / created_at``);
* the *current* `discover_patterns` output — ``src/tools/patterns.py`` ``_PATTERN_COLUMNS``
  (``pattern / gradual_items / support / is_temporal / target / max_lag_days / metrics``);
* the DoD lot O2 requirements — each pattern carries **support + confidence + zones/period
  of validation**, and the lot O3 stability fields (folds + robustness).

Design rule (ADR 0001 §5): ``confidence`` MUST stay ``None`` until it is calibrated by
T-GRAANK lag + 5-fold cross-zone validation (paliers 3/4). Emitting a calibrated
``support x forecast x decay`` confidence before then replays the original defect, so the
field defaults to ``None`` and producers at paliers 1-2 leave it unset. Which method produces
the O2 ``confidence`` once calibrated is still an open question (issue #20, 2026-09-19
checkpoint: not automatically B6's composite; Decision 1 unsigned).

Storage path (unchanged): ``s3://deposium/patterns/{zone_id}/{algorithm}.parquet``.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

SCHEMA_VERSION = "1.0.0"

Algorithm = Literal["graank", "tgrad", "aco", "ga", "pso"]

# Algorithms that legitimately carry a temporal lag — only T-GRAANK at palier 2; the GRAANK
# metaheuristic variants (ACO/GA/PSO) are contemporaneous.
TEMPORAL_ALGORITHMS: frozenset[str] = frozenset({"tgrad"})

# Minimum cross-validation folds before a calibrated `confidence` is legitimate. Proposed
# operational value (ADR 0002 §3 — 5-fold cross-zone, ratify at kickoff).
MIN_CV_FOLDS = 5


class GradualPatternRecord(BaseModel):
    """One gradual pattern in the exchange format. One row of the patterns Parquet."""

    model_config = ConfigDict(extra="forbid")

    pattern_id: int = Field(..., ge=0)
    # Rule form for the consumer (pattern_alert). For plain co-monotone itemsets
    # (GRAANK), `consequent` is "" and the full itemset lives in `antecedent`/`pattern`.
    antecedent: str = Field(..., description='e.g. "temperature_2m+, precipitation-"')
    consequent: str = Field("", description='e.g. "ndvi_mean-"; "" for itemsets')
    # Producer form (so4gp raw): the human-readable itemset string + parsed items.
    pattern: str = Field(..., description="raw so4gp itemset/temporal string")
    variables: list[str] = Field(default_factory=list)

    support: float = Field(..., ge=0.0, le=1.0, description="fraction of concordant row pairs")
    # Calibrated confidence — None until paliers 3/4 (ADR 0001 §5). Never fabricated earlier.
    confidence: float | None = Field(None, ge=0.0, le=1.0)

    is_temporal: bool = False
    # Signed lag in days (0 if contemporaneous), float to preserve sub-day / fractional-month
    # lags from the so4gp adapter (12 h → 0.5, 1 month → ~30.42). Extends §5.4's INTEGER.
    # Derived from the target item's signed lag — NOT the producer's unsigned `max_lag_days`.
    lag_days: float = Field(0.0, description="signed temporal lag in days; 0 if contemporaneous")

    algorithm: Algorithm
    zone_id: str = Field(..., min_length=1, description="producing zone")

    # Validation provenance (DoD O2 + O3) — empty/None until cross-zone validation runs.
    zones_validated: list[str] = Field(default_factory=list, description="cross-zone fold set (O3)")
    period_start: date | None = Field(None, description="validation period start")
    period_end: date | None = Field(None, description="validation period end")
    n_folds: int | None = Field(None, ge=0, description="cross-validation folds (O3)")
    stability: float | None = Field(None, ge=0.0, le=1.0, description="per-pattern robustness (O3)")

    created_at: datetime | None = None

    # Producer-internal raw measures (e.g. so4gp density / singularity_scr). Carried for
    # provenance only; NOT part of the support x forecast x decay scoring.
    extra_metrics: dict[str, float] | None = None

    @field_validator("variables")
    @classmethod
    def _non_empty_vars(cls, v: list[str]) -> list[str]:
        if any(not s.strip() for s in v):
            raise ValueError("variables must not contain empty strings")
        return v

    @field_validator("period_end")
    @classmethod
    def _period_order(cls, v: date | None, info):
        start = info.data.get("period_start")
        if v is not None and start is not None and v < start:
            raise ValueError("period_end must be >= period_start")
        return v

    @model_validator(mode="after")
    def _cross_field_invariants(self):
        # ADR 0001 §5 (ENFORCED): a calibrated `confidence` may exist ONLY once genuine
        # cross-validation ran — the support x forecast x decay value is illegitimate before
        # T-GRAANK lag + >= MIN_CV_FOLDS cross-zone validation. None / 0 / < MIN are all "no CV".
        if self.confidence is not None and (
            (self.n_folds or 0) < MIN_CV_FOLDS or not self.zones_validated
        ):
            raise ValueError(
                f"confidence may be set only after cross-validation "
                f"(requires n_folds >= {MIN_CV_FOLDS} + non-empty zones_validated)"
            )
        # only temporal algorithms may carry a temporal flag / non-zero lag
        if (self.is_temporal or self.lag_days != 0) and self.algorithm not in TEMPORAL_ALGORITHMS:
            raise ValueError(
                f"temporal/lagged records require a temporal algorithm "
                f"{sorted(TEMPORAL_ALGORITHMS)}, got {self.algorithm!r}"
            )
        # a non-zero lag implies the pattern is temporal
        if self.lag_days != 0 and not self.is_temporal:
            raise ValueError("a non-zero lag_days requires is_temporal=True")
        # the validation period must be fully specified or fully absent
        if (self.period_start is None) != (self.period_end is None):
            raise ValueError("period_start and period_end must both be set or both be None")
        return self


class PatternLibrary(BaseModel):
    """A per-(zone, algorithm) pattern library — the unit written to one Parquet file."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = SCHEMA_VERSION
    zone_id: str = Field(..., min_length=1)
    algorithm: Algorithm
    generated_at: datetime | None = None
    patterns: list[GradualPatternRecord] = Field(default_factory=list)

    @field_validator("patterns")
    @classmethod
    def _consistent_scope(cls, v: list[GradualPatternRecord], info):
        zone = info.data.get("zone_id")
        algo = info.data.get("algorithm")
        seen: set[int] = set()
        for p in v:
            if zone and p.zone_id != zone:
                raise ValueError(
                    f"pattern {p.pattern_id} zone_id {p.zone_id!r} != library {zone!r}"
                )
            if algo and p.algorithm != algo:
                raise ValueError(
                    f"pattern {p.pattern_id} algorithm {p.algorithm!r} != library {algo!r}"
                )
            if p.pattern_id in seen:
                raise ValueError(f"duplicate pattern_id {p.pattern_id} in library")
            seen.add(p.pattern_id)
        return v


def pattern_library_json_schema() -> dict:
    """The frozen exchange schema as JSON Schema (for non-Python consumers / contract tests)."""
    return PatternLibrary.model_json_schema()
