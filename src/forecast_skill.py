"""Paired MAE/P50 skill matrix (#193, the engineering slice of #98a).

``run_backtest`` scores one row set at a time. Running it twice — once on a provider, once
on a climatology baseline — and dividing the two aggregate MAEs is NOT a skill score, and
that is the specific mistake this module exists to make impossible. The published
2026-07-15 panel divided a 239-valid-date provider aggregate by a 238-valid-date baseline
aggregate; the extra date moves the ratio and nothing in either artifact records that the
two numerators were computed over different days.

Here the two MAEs are computed from THE SAME list of pairs, built once:

    for every expected pair that survives eligibility, the reference value and the monthly
    P50 for that (zone, variable, month) enter the same pair. A pair that cannot be
    completed on either side is excluded from BOTH sides and counted by reason.

So ``mae_forecast`` and ``mae_baseline`` always share an ``n``, and their ratio is a paired
skill score by construction rather than by a reviewer's assumption.

*Expected pairs are declared, not inferred.* A frozen panel manifest (:class:`PanelManifest`)
lists the zones, valid dates, variables and horizons the run is about, and the provider,
model and configuration it is frozen under. The denominator of every count is that declared
set. Counting from the forecast rows that happened to be submitted lets a short archive
shrink its own denominator and still read as complete; here a pair nobody submitted is
reported (``not_submitted``), not forgotten. A submitted row the manifest did not declare, a
duplicate row, or an identity that contradicts the manifest is refused.

Three policies are deliberate and each one is a place a well-meaning change would silently
destroy the result:

*Unclamped.* ``mae_skill = 1 - mae_forecast / mae_baseline`` is published as computed. A
forecast three times worse than climatology publishes ``-2.0``. Clamping at zero is how an
unskilled horizon becomes indistinguishable from a merely-no-better one, and B6 multiplies
this number.

*A zero denominator is undefined, not zero and not infinite.* ``mae_baseline == 0`` means
the climatology reproduced every observation exactly over those pairs; the ratio has no
value. The cell publishes ``mae_skill: null`` with ``skill_status:
"undefined_zero_denominator"`` and stays non-ratifiable. It is not dropped from the matrix
either — a disappearing cell reads as "not evaluated".

*Sample-weighted from the paired absolute-error sums.* Aggregating across horizons averages
the SUMS, never the per-cell ratios. The macro average of the cell skills is published too,
as a diagnostic, because it answers a different question (how the horizons behave on
average, each given equal weight) and must not be mistaken for the aggregate.

There is no all-variable scalar. Summing ``|Δmm|`` and ``|Δ°C|`` into one numerator is a
unit error; aggregates stop at the variable.

Out of scope here, on purpose: the block bootstrap over forecast origins (#98 keeps that as
an unsigned threshold obligation), CRPS/probabilistic scope, and any ratification claim.
This module computes and publishes; ratification is recorded in #98 by named signers.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from bisect import bisect_right
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any

from src.forecast_backtest import BacktestRow, BacktestValidationError

__all__ = [
    "EXCLUSION_REASONS",
    "PANEL_SCHEMA_VERSION",
    "REQUIRED_INPUTS",
    "SKILL_MATRIX_SCHEMA_VERSION",
    "MonthlyP50Baseline",
    "PanelManifest",
    "paired_skill_matrix",
    "result_digest",
]

#: v2 adds the declared-panel accounting (``census.expected_pairs``, ``not_submitted``) and the
#: ``panel_manifest`` input digest. There is no v1 output path: a v1 body cannot say how many
#: pairs were expected.
SKILL_MATRIX_SCHEMA_VERSION = "forecast-skill-matrix/v2"

#: The format of the frozen panel manifest, see :class:`PanelManifest`.
PANEL_SCHEMA_VERSION = "forecast-skill-panel/v1"

#: Every reason an expected pair can fail to become a scored pair. Each expected pair is
#: filed under the FIRST reason that applies, in this order, so the reasons sum to
#: ``missing_pairs``. The matrix always publishes all of them, zeros included: a reason that
#: only appears when it fires cannot be distinguished from a reason the producing code forgot
#: to implement.
EXCLUSION_REASONS = (
    "not_submitted",
    "missing_reference",
    "missing_baseline",
    "non_finite_baseline",
)

#: The four artifacts that jointly determine every cell. A matrix must fingerprint all of
#: them: the forecast alone cannot be re-scored without the truth it was compared against, the
#: denominator it was divided by, or the panel that says which pairs were expected.
REQUIRED_INPUTS = ("forecast", "reference", "baseline_p50", "panel_manifest")

_UNDEFINED_ZERO_DENOMINATOR = "undefined_zero_denominator"
_SCORED = "scored"


@dataclass(frozen=True)
class MonthlyP50Baseline:
    """The WMO 1991–2020 monthly P50 denominator, keyed ``(zone_id, variable, month)``.

    A mapping and not a row set, because the baseline is horizon-independent: the same
    monthly value is the denominator at h=1 and at h=7. ``forecast_baseline_panel.py`` builds
    it as ``BacktestRow``s with a literal ``horizon_days=1``, which is why the existing panel
    has no denominator at all above h=1 (#98 gap 2). Looking it up per pair fixes that
    without inventing per-horizon baseline rows.
    """

    values: Mapping[tuple[str, str, int], float]

    @classmethod
    def from_records(cls, records: Iterable[Mapping[str, Any]]) -> MonthlyP50Baseline:
        """Build from ``{zone_id, variable, month, p50}`` records, rejecting ambiguity.

        A duplicate ``(zone_id, variable, month)`` is fatal rather than last-write-wins: two
        different P50s for one month means the denominator depends on file order, and the
        matrix would be reproducible only by accident.

        A non-finite ``p50`` is KEPT. It is a real state of a normals table (a month with no
        usable observations) and the pairing counts it as ``non_finite_baseline``; rejecting
        it here would turn a recordable data gap into a crash on the whole panel. ``null`` is
        read as that same state and not as a type error: it is how a Parquet NaN survives a
        round trip through strict JSON, and the two spellings have to mean one thing.
        """
        values: dict[tuple[str, str, int], float] = {}
        for record in records:
            if not isinstance(record, Mapping):
                raise BacktestValidationError("baseline record must be an object")
            missing = [key for key in ("zone_id", "variable", "month", "p50") if key not in record]
            if missing:
                raise BacktestValidationError(f"baseline record missing: {', '.join(missing)}")
            zone = record["zone_id"]
            variable = record["variable"]
            if not isinstance(zone, str) or not zone:
                raise BacktestValidationError("baseline zone_id must be a non-empty string")
            if not isinstance(variable, str) or not variable:
                raise BacktestValidationError("baseline variable must be a non-empty string")
            month = record["month"]
            if isinstance(month, bool) or not isinstance(month, int) or not 1 <= month <= 12:
                raise BacktestValidationError("baseline month must be an integer in 1..12")
            raw_p50 = record["p50"]
            if raw_p50 is None:
                p50 = math.nan
            else:
                try:
                    p50 = float(raw_p50)
                except (TypeError, ValueError, OverflowError) as exc:
                    raise BacktestValidationError("baseline p50 must be numeric or null") from exc
            key = (zone, variable, month)
            if key in values:
                raise BacktestValidationError(f"duplicate baseline key: {key}")
            values[key] = p50
        if not values:
            raise BacktestValidationError("baseline must contain at least one monthly P50")
        return cls(values=values)

    def lookup(self, zone_id: str, variable: str, month: int) -> float | None:
        return self.values.get((zone_id, variable, month))


_PANEL_FIELDS = (
    "schema_version",
    "provider",
    "model",
    "configuration",
    "zones",
    "variables",
    "horizons_days",
    "valid_dates",
)

_PairKey = tuple[str, date, date, str]  # (zone_id, origin_date, valid_date, variable)


def _canonical_configuration(configuration: Any) -> str:
    """The configuration as canonical JSON, so identity is compared on JSON types.

    Python equality would call ``{"run": 1}``, ``{"run": 1.0}`` and ``{"run": True}`` the same
    configuration; as published JSON they are three different ones. A non-finite float is a
    refusal here rather than a ``ValueError`` from ``result_digest`` after the run.
    """
    if not isinstance(configuration, Mapping):
        raise BacktestValidationError("configuration must be a JSON object")
    try:
        return json.dumps(configuration, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise BacktestValidationError(
            "configuration must be finite, JSON-serializable data"
        ) from exc


def _string_axis(field: str, noun: str, items: Any) -> tuple[str, ...]:
    if not isinstance(items, (list, tuple)) or not items:
        raise BacktestValidationError(f"panel manifest {field} must be a non-empty list")
    seen: set[str] = set()
    for item in items:
        if not isinstance(item, str) or not item:
            raise BacktestValidationError(f"panel manifest {field} must contain non-empty strings")
        if item in seen:
            raise BacktestValidationError(f"duplicate {noun} in panel manifest: {item!r}")
        seen.add(item)
    return tuple(items)


def _horizon_axis(items: Any) -> tuple[int, ...]:
    if not isinstance(items, (list, tuple)) or not items:
        raise BacktestValidationError("panel manifest horizons_days must be a non-empty list")
    seen: set[int] = set()
    for item in items:
        if isinstance(item, bool) or not isinstance(item, int) or item < 1:
            raise BacktestValidationError(
                "panel manifest horizons_days must contain positive integers"
            )
        if item in seen:
            raise BacktestValidationError(f"duplicate horizon in panel manifest: {item}")
        seen.add(item)
    return tuple(items)


def _date_axis(items: Any) -> tuple[date, ...]:
    if not isinstance(items, (list, tuple)) or not items:
        raise BacktestValidationError("panel manifest valid_dates must be a non-empty list")
    seen: set[date] = set()
    for item in items:
        if type(item) is not date:  # a datetime is a date subclass, and not a calendar day
            raise BacktestValidationError("panel manifest valid_dates must contain ISO dates")
        if item in seen:
            raise BacktestValidationError(
                f"duplicate valid date in panel manifest: {item.isoformat()}"
            )
        seen.add(item)
    return tuple(items)


@dataclass(frozen=True)
class PanelManifest:
    """The frozen panel a matrix is about: what was EXPECTED, independent of what was submitted.

    The expected pairs are the cross product ``zones x variables x valid_dates x
    horizons_days``. The origin date of a pair is derived (``valid_date - horizon_days``);
    it is not a fourth free axis, because a forecast row already has to satisfy
    ``horizon_days == valid_date - origin_date``. ``provider``, ``model`` and
    ``configuration`` are the identity the panel was frozen under.

    Only the four axes are stored, never the pairs: ``expected_pairs`` is their product and
    membership is four set lookups, so a panel of 10^9 pairs costs what its axes cost.

    The invariants — non-empty axes, no duplicate entry on any axis, positive integer
    horizons, real dates, an identity of finite JSON data, and origins that stay inside the
    calendar — are checked in ``__post_init__``, so they hold for ANY instance, including one
    built directly and not through :meth:`from_record`. That is what makes ``expected_pairs``
    the number of distinct declared pairs and not merely an upper bound of it.

    As JSON (``schema_version`` is ``forecast-skill-panel/v1``; every field is required and
    unknown fields are refused)::

        {"schema_version": "forecast-skill-panel/v1",
         "provider": "aifs", "model": "aifs-single-v1", "configuration": {"run": "00z"},
         "zones": ["zone_a", "zone_b"], "variables": ["temperature_2m"],
         "horizons_days": [1, 3, 7], "valid_dates": ["2021-01-25", "2021-01-26"]}

    A rectangle is the only shape the format can declare. A panel that is genuinely ragged
    (a zone with no h=7) is declared as the full rectangle, and the pairs it never had show up
    as ``not_submitted`` rather than disappearing from the denominator.
    """

    provider: str
    model: str
    configuration: Mapping[str, Any]
    zones: tuple[str, ...]
    variables: tuple[str, ...]
    horizons_days: tuple[int, ...]
    valid_dates: tuple[date, ...]

    def __post_init__(self) -> None:
        for field in ("provider", "model"):
            value = getattr(self, field)
            if not isinstance(value, str) or not value:
                raise BacktestValidationError(f"panel manifest {field} must be a non-empty string")
        _canonical_configuration(self.configuration)
        # Frozen, so normalised through object.__setattr__: lists become tuples, and the
        # configuration is deep-copied so nothing the caller does to its dict, at any depth,
        # can change the panel afterwards.
        object.__setattr__(self, "configuration", copy.deepcopy(dict(self.configuration)))
        object.__setattr__(self, "zones", _string_axis("zones", "zone", self.zones))
        object.__setattr__(self, "variables", _string_axis("variables", "variable", self.variables))
        object.__setattr__(self, "horizons_days", _horizon_axis(self.horizons_days))
        object.__setattr__(self, "valid_dates", _date_axis(self.valid_dates))
        # The earliest date with the longest horizon is the only origin that can leave the
        # calendar, so this is one subtraction and not one per pair.
        earliest, longest = min(self.valid_dates), max(self.horizons_days)
        try:
            earliest - timedelta(days=longest)
        except OverflowError as exc:
            raise BacktestValidationError(
                f"panel manifest horizons_days: {longest} days before "
                f"{earliest.isoformat()} is outside the calendar"
            ) from exc

    @classmethod
    def from_record(cls, record: Any) -> PanelManifest:
        """Build from the decoded JSON object, refusing anything ambiguous or malformed.

        Every malformed shape — wrong container, wrong element type, an empty axis, a
        duplicate, an unparsable date, a horizon that leaves the calendar — is a
        :class:`BacktestValidationError`; none may escape as a ``TypeError`` or an
        ``OverflowError``. This only decodes (ISO strings to dates); the checks themselves are
        the constructor's.
        """
        if not isinstance(record, Mapping):
            raise BacktestValidationError("panel manifest must be a JSON object")
        missing = [field for field in _PANEL_FIELDS if field not in record]
        if missing:
            raise BacktestValidationError(f"panel manifest missing: {', '.join(missing)}")
        unknown = sorted(str(key) for key in record if key not in _PANEL_FIELDS)
        if unknown:
            raise BacktestValidationError(f"panel manifest has unknown field(s): {unknown}")
        if record["schema_version"] != PANEL_SCHEMA_VERSION:
            raise BacktestValidationError(
                f"panel manifest schema_version must be {PANEL_SCHEMA_VERSION!r}"
            )
        raw_dates = record["valid_dates"]
        if not isinstance(raw_dates, (list, tuple)):
            raise BacktestValidationError("panel manifest valid_dates must be a non-empty list")
        valid_dates = []
        for raw in raw_dates:
            try:
                if not isinstance(raw, str):
                    raise ValueError(raw)
                valid_dates.append(date.fromisoformat(raw))
            except ValueError as exc:
                raise BacktestValidationError(
                    "panel manifest valid_dates must contain ISO dates"
                ) from exc
        return cls(
            provider=record["provider"],
            model=record["model"],
            configuration=record["configuration"],
            zones=record["zones"],
            variables=record["variables"],
            horizons_days=record["horizons_days"],
            valid_dates=tuple(valid_dates),
        )

    @property
    def expected_pairs(self) -> int:
        """The number of distinct declared pairs; arithmetic, since no axis holds a duplicate."""
        return (
            len(self.zones) * len(self.variables) * len(self.valid_dates) * len(self.horizons_days)
        )

    def declares(self) -> Callable[[str, date, date, str], bool]:
        """A membership test ``(zone, origin, valid, variable) -> bool`` over the four axes."""
        zones, variables = frozenset(self.zones), frozenset(self.variables)
        dates, horizons = frozenset(self.valid_dates), frozenset(self.horizons_days)

        def _declared(zone: str, origin: date, valid: date, variable: str) -> bool:
            return (
                zone in zones
                and variable in variables
                and valid in dates
                and (valid - origin).days in horizons
            )

        return _declared

    def require_identity(
        self, *, provider: str, model: str, configuration: Mapping[str, Any]
    ) -> None:
        """A matrix is published under the identity it was frozen with; a mismatch is fatal."""
        declared = {
            "provider": provider,
            "model": model,
            "configuration": _canonical_configuration(configuration),
        }
        frozen = {
            "provider": self.provider,
            "model": self.model,
            "configuration": _canonical_configuration(self.configuration),
        }
        differing = [name for name in declared if declared[name] != frozen[name]]
        if differing:
            raise BacktestValidationError(
                f"declared {', '.join(differing)} contradicts the panel manifest"
            )

    def require_within(self, *, training_end: date, holdout_start: date, holdout_end: date) -> None:
        """The declared pairs must respect the window and the leakage barrier.

        Policed where the pair is DECLARED and not where it is submitted: a pair nobody
        submitted still counts in the denominator, so a manifest that expects a leaked
        origin or a date outside the holdout describes a panel this matrix cannot score.

        Done without enumerating pairs. The window only depends on the earliest and latest
        date. A pair is in the gap when ``training_end < valid - h < holdout_start``, i.e.
        ``valid - holdout_start < h < valid - training_end``: for each date, one binary
        search over the sorted horizons finds the smallest ``h`` above the lower bound and
        the gap holds a pair exactly when that ``h`` is below the upper bound.
        """
        first, last = min(self.valid_dates), max(self.valid_dates)
        for outside in (first, last):
            if not holdout_start <= outside <= holdout_end:
                raise BacktestValidationError(
                    f"panel valid date {outside.isoformat()} is outside the holdout window"
                )
        horizons = sorted(self.horizons_days)
        for valid in sorted(self.valid_dates):
            above = bisect_right(horizons, (valid - holdout_start).days)
            if above < len(horizons) and horizons[above] < (valid - training_end).days:
                origin = valid - timedelta(days=horizons[above])
                raise BacktestValidationError(
                    f"panel pair originating {origin.isoformat()} falls in the baseline/holdout gap"
                )


@dataclass(frozen=True)
class _Pair:
    zone_id: str
    origin_date: date
    valid_date: date
    variable: str
    horizon_days: int
    forecast: float
    baseline: float
    reference: float


def _window(training_end: date, holdout_start: date, holdout_end: date) -> None:
    if training_end >= holdout_start:
        raise BacktestValidationError("training and holdout periods overlap")
    if holdout_start > holdout_end:
        raise BacktestValidationError("holdout_start must be before holdout_end")


def _reference_index(
    reference: list[BacktestRow],
    *,
    holdout_start: date,
    holdout_end: date,
) -> tuple[dict[tuple[str, date, str], float], int]:
    index: dict[tuple[str, date, str], float] = {}
    outside = 0
    for row in reference:
        key = (row.zone_id, row.valid_date, row.variable)
        if key in index:
            raise BacktestValidationError(f"duplicate reference key: {key}")
        index[key] = row.value
        if not holdout_start <= row.valid_date <= holdout_end:
            outside += 1
    return index, outside


def _frozen_provider(forecast: list[BacktestRow], *, provider: str) -> None:
    """One matrix, one provider. A mixed row set is refused, not silently pooled.

    ``run_backtest`` groups by provider and can carry several in one result. A skill matrix
    cannot: it is published as the immutable evidence for ONE frozen provider/model/config,
    and pooling two providers into cells labelled with one of them would misattribute the
    other's errors.
    """
    providers = sorted({row.provider for row in forecast})
    if len(providers) > 1:
        raise BacktestValidationError(
            f"forecast rows mix providers {providers}; one matrix scores one frozen provider"
        )
    if providers and providers[0] != provider:
        raise BacktestValidationError(
            f"declared provider {provider!r} does not match forecast rows {providers[0]!r}"
        )


def _pairs(
    forecast: list[BacktestRow],
    reference_index: Mapping[tuple[str, date, str], float],
    baseline: MonthlyP50Baseline,
    panel: PanelManifest,
) -> tuple[list[_Pair], int, dict[str, int]]:
    """``(pairs, eligible, exclusions)``, counted against the DECLARED panel.

    The split between what raises and what is merely counted is the point. An IDENTITY
    violation raises: a submitted row the manifest never declared, or a duplicate prediction
    key that would double-weight a cell, makes the whole matrix wrong and no count can repair
    it. (The leakage barrier is policed on the manifest itself, in
    ``PanelManifest.require_within``: every submitted row is a declared pair, so its origin
    was already checked.) A DATA GAP — no forecast submitted, no reference for that day, no P50
    for that month — is counted and excluded from both sides, because that is an ordinary and
    honest state of a panel.

    Every expected pair is filed under the first gap that applies, in ``EXCLUSION_REASONS``
    order, so the reasons partition ``expected - scored``.
    """
    declared = panel.declares()
    submitted: dict[_PairKey, BacktestRow] = {}
    for row in forecast:
        key = (row.zone_id, row.origin_date, row.valid_date, row.variable)
        if key in submitted:
            raise BacktestValidationError(f"duplicate prediction key: {key}")
        if row.horizon_days != (row.valid_date - row.origin_date).days:
            # Panel membership derives the horizon from the dates while the cell is labelled
            # with `row.horizon_days`; `parse_rows` guarantees they agree, a row built by hand
            # does not have to.
            raise BacktestValidationError(
                f"prediction row horizon_days {row.horizon_days} contradicts its dates: {key}"
            )
        if not declared(*key):
            raise BacktestValidationError(f"prediction row outside the declared panel: {key}")
        submitted[key] = row

    # Only the submitted rows are visited. Every one of them is a distinct declared pair (a
    # duplicate and an undeclared row were refused above), so the pairs nobody submitted are
    # counted, not enumerated: a panel of 10^9 expected pairs costs what its rows cost.
    exclusions = dict.fromkeys(EXCLUSION_REASONS, 0)
    exclusions["not_submitted"] = panel.expected_pairs - len(submitted)
    pairs: list[_Pair] = []
    for row in submitted.values():
        actual = reference_index.get((row.zone_id, row.valid_date, row.variable))
        if actual is None:
            exclusions["missing_reference"] += 1
            continue
        normal = baseline.lookup(row.zone_id, row.variable, row.valid_date.month)
        if normal is None:
            exclusions["missing_baseline"] += 1
            continue
        if not math.isfinite(normal):
            exclusions["non_finite_baseline"] += 1
            continue
        pairs.append(
            _Pair(
                zone_id=row.zone_id,
                origin_date=row.origin_date,
                valid_date=row.valid_date,
                variable=row.variable,
                horizon_days=row.horizon_days,
                forecast=row.value,
                baseline=normal,
                reference=actual,
            )
        )
    # Sorted so the floating-point sums below are order-independent: the artifact carries a
    # digest over its own cells, and a digest that changes with the order rows happened to
    # arrive in is not a fingerprint of the result.
    pairs.sort(key=lambda pair: (pair.variable, pair.horizon_days, pair.zone_id, pair.valid_date))
    return pairs, len(submitted), exclusions


def _skill(forecast_sum: float, baseline_sum: float) -> tuple[float | None, str]:
    if baseline_sum == 0:
        return None, _UNDEFINED_ZERO_DENOMINATOR
    return 1 - forecast_sum / baseline_sum, _SCORED


def _summary(pairs: list[_Pair]) -> dict[str, Any]:
    """Paired sums first, means derived from them. ``n`` is shared by construction."""
    n = len(pairs)
    forecast_errors = [pair.forecast - pair.reference for pair in pairs]
    baseline_errors = [pair.baseline - pair.reference for pair in pairs]
    forecast_sum = sum(abs(error) for error in forecast_errors)
    baseline_sum = sum(abs(error) for error in baseline_errors)
    # The ratio of the sums and the ratio of the means are the same number only because the
    # two sides share `n` — which is exactly the property this module enforces.
    skill, status = _skill(forecast_sum, baseline_sum)
    return {
        "n": n,
        "abs_error_sum_forecast": forecast_sum,
        "abs_error_sum_baseline": baseline_sum,
        "mae_forecast": forecast_sum / n,
        "mae_baseline": baseline_sum / n,
        "mae_skill": skill,
        "skill_status": status,
        "diagnostics": {
            "rmse_forecast": math.sqrt(sum(e * e for e in forecast_errors) / n),
            "rmse_baseline": math.sqrt(sum(e * e for e in baseline_errors) / n),
            "bias_forecast": sum(forecast_errors) / n,
            "bias_baseline": sum(baseline_errors) / n,
        },
    }


def _by_zone(pairs: list[_Pair]) -> list[dict[str, Any]]:
    """Per-zone values beside every pooled cell (#98 gap 4).

    A pooled cell answers "does this horizon beat climatology overall"; it cannot answer "on
    how many zones", and a pooled positive built from three strong zones and nine negative
    ones is the failure mode the acceptance criteria name explicitly.
    """
    grouped: dict[str, list[_Pair]] = defaultdict(list)
    for pair in pairs:
        grouped[pair.zone_id].append(pair)
    return [
        {"zone_id": zone_id, **_summary(zone_pairs)}
        for zone_id, zone_pairs in sorted(grouped.items())
    ]


def _macro(cells: list[dict[str, Any]]) -> dict[str, Any]:
    defined = [cell["mae_skill"] for cell in cells if cell["mae_skill"] is not None]
    return {
        "macro_mae_skill": (sum(defined) / len(defined)) if defined else None,
        "defined_cells": len(defined),
        "undefined_cells": len(cells) - len(defined),
    }


def _validate_inputs_manifest(inputs: Mapping[str, str]) -> dict[str, str]:
    """All four input digests are required, not optional metadata.

    A matrix without them names no bytes, so "the same inputs" is unverifiable and the
    immutability claim is decoration. "At least one" was not enough: the four arguments
    jointly determine every cell, so a manifest carrying only the forecast digest describes
    a quarter of what produced the numbers while looking complete — and a reviewer re-running
    it would have nothing to check the reference, the denominator or the declared panel
    against.

    Extra names are allowed on purpose. A caller that also digests the archive manifest or
    the zone boundaries is recording more provenance, not less, and refusing that would push
    the extra fingerprints somewhere the artifact does not carry them.
    """
    missing = [name for name in REQUIRED_INPUTS if name not in inputs]
    if missing:
        raise BacktestValidationError(f"inputs missing digests for: {', '.join(missing)}")
    resolved: dict[str, str] = {}
    for name, digest in inputs.items():
        if not isinstance(name, str) or not name:
            raise BacktestValidationError("input name must be a non-empty string")
        if not isinstance(digest, str) or len(digest) != 64:
            raise BacktestValidationError(f"input {name!r} digest must be a sha256 hex string")
        try:
            int(digest, 16)
        except ValueError as exc:
            raise BacktestValidationError(
                f"input {name!r} digest must be a sha256 hex string"
            ) from exc
        resolved[name] = digest.lower()
    return resolved


def result_digest(matrix: Mapping[str, Any]) -> str:
    """SHA-256 over the scientific content of a matrix.

    ``code_ref`` and the digest field itself are excluded, and that exclusion is the point:
    the digest answers "is this the same result", which must stay ``True`` when the identical
    inputs are re-scored from a later commit. The commit that produced the artifact is
    recorded beside the digest, not inside it, so a reviewer can see both answers separately.
    """
    body = {key: value for key, value in matrix.items() if key not in ("code_ref", "result_digest")}
    canonical = json.dumps(body, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def paired_skill_matrix(
    forecast: list[BacktestRow],
    reference: list[BacktestRow],
    baseline: MonthlyP50Baseline,
    *,
    panel: PanelManifest,
    training_end: date,
    holdout_start: date,
    holdout_end: date,
    provider: str,
    model: str,
    configuration: Mapping[str, Any] | None = None,
    inputs: Mapping[str, str],
    code_ref: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Score one frozen provider/model/configuration against the monthly P50 baseline.

    Args:
        forecast: archived predictions for a single provider. Mixed providers are refused, as
            is a duplicate row or a row the ``panel`` did not declare.
        reference: the independently materialized ERA5 truth rows.
        baseline: the monthly P50 denominator.
        panel: the frozen panel manifest. It, and not ``forecast``, defines the expected pairs.
        training_end/holdout_start/holdout_end: the panel window; the gap between
            ``training_end`` and ``holdout_start`` is a leakage barrier, and a panel that
            declares a pair originating inside it, or a valid date outside the holdout,
            is refused.
        provider/model/configuration: the frozen identity the matrix is published under. It
            must equal the identity declared in ``panel``.
        inputs: ``name -> sha256`` for every artifact the cells were computed from, the panel
            manifest included. Required.
        code_ref: the producing checkout, recorded beside the cells and excluded from
            ``result_digest``. See :mod:`src.utils.code_ref`.

    Returns:
        The matrix payload, with ``cells`` keyed by ``(variable, horizon_days)``,
        ``variable_aggregates`` sample-weighted across horizons, and a ``census`` whose
        ``expected -> eligible -> scored`` chain accounts for every declared pair.
    """
    if not isinstance(provider, str) or not provider:
        raise BacktestValidationError("provider must be a non-empty string")
    if not isinstance(model, str) or not model:
        raise BacktestValidationError("model must be a non-empty string")
    _window(training_end, holdout_start, holdout_end)
    if not isinstance(panel, PanelManifest):
        raise BacktestValidationError("panel must be a PanelManifest")
    panel.require_identity(
        provider=provider, model=model, configuration={} if configuration is None else configuration
    )
    panel.require_within(
        training_end=training_end, holdout_start=holdout_start, holdout_end=holdout_end
    )
    _frozen_provider(forecast, provider=provider)
    digested = _validate_inputs_manifest(inputs)
    reference_index, reference_outside = _reference_index(
        reference,
        holdout_start=holdout_start,
        holdout_end=holdout_end,
    )
    pairs, eligible, exclusions = _pairs(forecast, reference_index, baseline, panel)

    grouped: dict[tuple[str, int], list[_Pair]] = defaultdict(list)
    for pair in pairs:
        grouped[(pair.variable, pair.horizon_days)].append(pair)
    cells = [
        {
            "variable": variable,
            "horizon_days": horizon,
            **_summary(cell_pairs),
            "by_zone": _by_zone(cell_pairs),
        }
        for (variable, horizon), cell_pairs in sorted(grouped.items())
    ]

    by_variable: dict[str, list[_Pair]] = defaultdict(list)
    for pair in pairs:
        by_variable[pair.variable].append(pair)
    aggregates = []
    for variable, variable_pairs in sorted(by_variable.items()):
        summary = _summary(variable_pairs)
        summary["diagnostics"].update(
            _macro([cell for cell in cells if cell["variable"] == variable])
        )
        aggregates.append(
            {
                "variable": variable,
                "horizons": sorted({pair.horizon_days for pair in variable_pairs}),
                **summary,
            }
        )

    matrix: dict[str, Any] = {
        "schema_version": SKILL_MATRIX_SCHEMA_VERSION,
        # A matrix with no scored pair is not a failed run and not a completed one: the
        # inputs parsed, the window held, and nothing survived eligibility. Publishing it as
        # `completed` beside an empty `cells` array is how an empty result gets read as "the
        # provider was evaluated", so it says so instead.
        "status": "completed" if pairs else "no_scored_pairs",
        "baseline": {
            "kind": "monthly_p50",
            "definition": "mae_skill = 1 - MAE_forecast / MAE_climatology",
            "clamped": False,
            "zero_denominator": _UNDEFINED_ZERO_DENOMINATOR,
        },
        "frozen_configuration": {
            "provider": provider,
            "model": model,
            "configuration": {} if configuration is None else dict(configuration),
        },
        "training_end": training_end.isoformat(),
        "holdout": {"start": holdout_start.isoformat(), "end": holdout_end.isoformat()},
        "inputs": digested,
        "census": {
            # expected = scored + missing, and missing = the sum of `exclusions`, always.
            # `eligible` is the expected pairs that HAD a submitted row (expected minus
            # `not_submitted`); it can still lose pairs to a reference or P50 gap.
            # `submitted_forecast_rows` is a diagnostic on the input, never a denominator.
            "expected_pairs": panel.expected_pairs,
            "submitted_forecast_rows": len(forecast),
            "eligible_pairs": eligible,
            "scored_pairs": len(pairs),
            "missing_pairs": panel.expected_pairs - len(pairs),
            "exclusions": exclusions,
            "reference_rows": len(reference),
            "reference_rows_outside_holdout": reference_outside,
            "baseline_months": len(baseline.values),
        },
        "cells": cells,
        "variable_aggregates": aggregates,
        # No all-variable scalar: summing |Δmm| and |Δ°C| into one numerator is a unit error.
        "notes": [
            "forecast and baseline share every scored pair by construction",
            "mae_skill is unclamped; negative values are real and published",
            "no aggregate spans variables",
            "expected pairs come from the declared panel manifest, not from submitted rows",
        ],
    }
    matrix["result_digest"] = result_digest(matrix)
    matrix["code_ref"] = dict(code_ref) if code_ref is not None else None
    return matrix
