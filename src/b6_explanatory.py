"""B6 level 2: the explanatory index ``b6-explanatory/1.0.0`` (#30).

``explanatory_index = support x stability x forecast_agreement x lag_decay``. It says how
strongly the current forecast supports cross-validated patterns from a verified D-A library
for one zone. It is the arithmetic ``pattern_alert`` already reports as ``provisional_score``,
now named, versioned, gated and traced; it is NOT a probability, a calibrated confidence or a
warning, so the result keeps ``confidence`` and ``event_probability`` null.

The forecast helpers live here, not in ``pattern_alert``, so both levels share one
implementation of the alignment and decay arithmetic; ``pattern_alert`` imports them back.
"""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any

from src.contracts import (
    B6_EXCLUSION_REASONS,
    B6_FORMULA_ID,
    B6Factor,
    B6Level2Output,
    B6LibraryIdentity,
    B6Params,
    B6PatternResult,
    B6PatternTrace,
    B6Trace,
)
from src.patterns.library_handoff import verify_library_handoff
from src.patterns.library_schema import GradualPatternRecord, PatternLibrary
from src.utils.openmeteo import FORECAST_NORMALIZED_VARS, NORMALIZED_VARS

_ITEM_RE = re.compile(r"\(?\s*([A-Za-z_][A-Za-z0-9_]*)\s*([+-])\s*\)?")


def finite_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def parse_items(text: str) -> list[dict[str, str]]:
    return [
        {"variable": match.group(1), "direction": match.group(2)}
        for match in _ITEM_RE.finditer(text or "")
    ]


def series_delta(forecast_rows: list[dict[str, Any]], variable: str) -> dict[str, Any]:
    values: list[float] = []
    dates: list[str | None] = []
    for row in forecast_rows:
        value = finite_float(row.get(variable))
        if value is not None:
            values.append(value)
            dates.append(row.get("date"))
    if len(values) < 2:
        return {"status": "insufficient_forecast_values", "n": len(values)}
    delta = values[-1] - values[0]
    direction = "+" if delta > 0 else "-" if delta < 0 else "0"
    return {
        "status": "ok",
        "first_date": dates[0],
        "last_date": dates[-1],
        "first": values[0],
        "last": values[-1],
        "delta": delta,
        "direction": direction,
        "n": len(values),
    }


def forecast_alignment(
    *,
    antecedent: str,
    forecast_rows: list[dict[str, Any]],
    supported_variables: tuple[str, ...] = NORMALIZED_VARS,
) -> tuple[float, list[dict[str, Any]]]:
    # Preserve pattern_alert's historical driver set by default. Level 2 explicitly
    # supplies the complete forecast set, including forecast-only minimum temperature.
    evidence: list[dict[str, Any]] = []
    items = [item for item in parse_items(antecedent) if item["variable"] in supported_variables]
    if not items:
        return 0.0, evidence
    matches = 0
    for item in items:
        delta = series_delta(forecast_rows, item["variable"])
        matched = delta.get("direction") == item["direction"]
        if matched:
            matches += 1
        evidence.append({**item, "matched": matched, **delta})
    return matches / len(items), evidence


def lag_decay(lag_days: float, halflife_days: float) -> float:
    return 0.5 ** (abs(lag_days) / halflife_days)


# --------------------------------------------------------------------------- #
# Level 2 composition
# --------------------------------------------------------------------------- #

# Gates that make a pattern admissible at all, before the forecast is looked at. If none
# survives them the answer is "no admissible pattern", whatever the forecast says.
_ADMISSIBILITY = ("not_cross_validated", "zone_not_validated", "low_support", "low_stability")


def _gate(
    record: GradualPatternRecord,
    *,
    zone_id: str,
    params: B6Params,
    effective_horizon_days: float,
    forecast_rows: list[dict[str, Any]],
) -> tuple[str | None, float, list[dict[str, Any]]]:
    """``(first failed gate or None, forecast_agreement, evidence)``, gates in contract order.

    The cross-validation gate is the exchange-schema one (``n_folds`` plus a non-empty
    ``zones_validated``), stricter than ``pattern_alert``'s screening gate on purpose: a
    pattern nobody cross-validated must not be ranked, however well the forecast fits it.
    """
    if (record.n_folds or 0) < params.min_folds or not record.zones_validated:
        return "not_cross_validated", 0.0, []
    if zone_id not in record.zones_validated:
        return "zone_not_validated", 0.0, []
    if record.support < params.min_support:
        return "low_support", 0.0, []
    if record.stability is None or record.stability < params.min_stability:
        return "low_stability", 0.0, []
    if abs(record.lag_days) > effective_horizon_days:
        return "lag_outside_horizon", 0.0, []
    agreement, evidence = forecast_alignment(
        antecedent=record.antecedent,
        forecast_rows=forecast_rows,
        supported_variables=FORECAST_NORMALIZED_VARS,
    )
    if not evidence:
        return "no_forecast_variable", 0.0, []
    # A driver without two forecast values has no direction; counting it as a mismatch
    # would silently lower the index instead of saying the data is missing.
    if any(item["status"] != "ok" for item in evidence):
        return "insufficient_forecast_values", 0.0, evidence
    return None, agreement, evidence


def _result(
    record: GradualPatternRecord,
    *,
    library: PatternLibrary,
    agreement: float,
    evidence: list[dict[str, Any]],
    params: B6Params,
) -> B6PatternResult:
    stability = record.stability or 0.0  # admitted patterns always carry one
    decay = lag_decay(record.lag_days, params.halflife_days)
    index = record.support * stability * agreement * decay
    matches = sum(1 for item in evidence if item["matched"])
    # Agreed on #30: only forecast drivers are compared, and the variables left out are named,
    # because agreement on part of the antecedent does not verify the whole antecedent.
    omitted = sorted(
        {item["variable"] for item in parse_items(record.antecedent)}
        - {item["variable"] for item in evidence}
    )
    skipped = (
        "; partial driver agreement, not verification of the full antecedent; "
        f"not counted: {', '.join(omitted)}"
        if omitted
        else ""
    )
    factors = [
        B6Factor(name="support", value=record.support, derivation="library support"),
        B6Factor(name="stability", value=stability, derivation="library stability (O3 robustness)"),
        B6Factor(
            name="forecast_agreement",
            value=agreement,
            derivation=(
                f"{matches}/{len(evidence)} antecedent drivers change in the pattern direction "
                f"over the forecast (direction agreement, not forecast skill){skipped}"
            ),
        ),
        B6Factor(
            name="lag_decay",
            value=decay,
            derivation=f"0.5 ** (|{record.lag_days:g}| / {params.halflife_days:g})",
        ),
    ]
    return B6PatternResult(
        rule_id=f"{library.zone_id}/{library.algorithm}/{record.pattern_id}",
        pattern_id=record.pattern_id,
        pattern=record.pattern,
        antecedent=record.antecedent,
        consequent=record.consequent,
        lag_days=record.lag_days,
        explanatory_index=index,
        factors=factors,
        forecast_evidence=evidence,
        explanation=(
            f"{matches}/{len(evidence)} forecast drivers match the pattern direction; "
            f"explanatory_index {index:.3f} = support {record.support:.3f} x stability "
            f"{stability:.3f} x forecast_agreement {agreement:.3f} x lag_decay {decay:.3f}. "
            "Explanatory ranking only: not a probability or a calibrated confidence."
        ),
    )


def _no_conclusion_reason(excluded: list[str | None]) -> str:
    if not excluded:
        return "empty_library"
    admissible = [reason for reason in excluded if reason not in _ADMISSIBILITY]
    if not admissible:
        return "no_admissible_pattern"
    if all(reason == "lag_outside_horizon" for reason in admissible):
        return "all_lags_outside_horizon"
    return "missing_forecast_data"


def score_level2(
    library: PatternLibrary,
    *,
    library_identity: B6LibraryIdentity | dict[str, Any],
    forecast_rows: list[dict[str, Any]],
    forecast_identity: dict[str, Any],
    zone_id: str,
    effective_horizon_days: float,
    params: B6Params | None = None,
    code_ref: dict[str, Any] | None,
    computed_at: str,
) -> B6Level2Output:
    """Score one verified library against one saved forecast; no I/O, no clock, no network.

    ``computed_at`` and ``code_ref`` are inputs so that the same inputs give byte-identical
    output. Every read pattern appears in the trace with its first failed gate (or none).
    """
    params = params or B6Params()
    identity = B6LibraryIdentity.model_validate(library_identity)
    # The trace must name the library that was scored, not another one's digests.
    if (identity.zone_id, identity.algorithm) != (library.zone_id, library.algorithm):
        raise ValueError("library_identity does not describe the scored library")
    eligible: list[B6PatternResult] = []
    traces: list[B6PatternTrace] = []
    excluded: list[str | None] = []
    for record in library.patterns:
        reason, agreement, evidence = _gate(
            record,
            zone_id=zone_id,
            params=params,
            effective_horizon_days=effective_horizon_days,
            forecast_rows=forecast_rows,
        )
        excluded.append(reason)
        traces.append(
            B6PatternTrace(
                pattern_id=record.pattern_id,
                n_folds=record.n_folds,
                zones_validated=record.zones_validated,
                period_start=record.period_start,
                period_end=record.period_end,
                excluded=reason,
            )
        )
        if reason is None:
            eligible.append(
                _result(
                    record,
                    library=library,
                    agreement=agreement,
                    evidence=evidence,
                    params=params,
                )
            )
    eligible.sort(
        key=lambda p: (
            -p.explanatory_index,
            -p.factors[2].value,
            -p.factors[1].value,
            -p.factors[0].value,
            p.pattern_id,
        )
    )
    return B6Level2Output(
        zone_id=zone_id,
        algorithm=library.algorithm,
        status="explained" if eligible else "no_conclusion",
        reason=None if eligible else _no_conclusion_reason(excluded),
        patterns=eligible[: params.max_patterns],
        n_patterns_read=len(library.patterns),
        n_eligible=len(eligible),
        exclusions={name: excluded.count(name) for name in B6_EXCLUSION_REASONS},
        trace=B6Trace(
            formula_id=B6_FORMULA_ID,
            formula_version="1.0.0",
            params=params,
            library=identity,
            forecast=forecast_identity,
            effective_horizon_days=effective_horizon_days,
            code_ref=code_ref,
            patterns=traces,
            computed_at=computed_at,
        ),
    )


# --------------------------------------------------------------------------- #
# Authenticated input
# --------------------------------------------------------------------------- #


def load_verified_library(
    package_dir: str | Path,
    expected_manifest_sha256: str,
    zone_id: str,
    algorithm: str,
) -> tuple[PatternLibrary, B6LibraryIdentity]:
    """One library from a D-A package, only after ``verify_library_handoff`` accepts it all.

    The manifest digest must come from a trusted channel, never from the package itself.
    The returned identity is copied from the verified manifest, so the trace names exactly
    the bytes that were scored. An unknown scope is refused rather than scored as empty:
    "this zone has no patterns" and "this zone is not in the package" are different answers.
    """
    manifest, libraries = verify_library_handoff(package_dir, expected_manifest_sha256)
    entry = next(
        (e for e in manifest.libraries if (e.zone_id, e.algorithm) == (zone_id, algorithm)),
        None,
    )
    library = next(
        (lib for lib in libraries if (lib.zone_id, lib.algorithm) == (zone_id, algorithm)),
        None,
    )
    if entry is None or library is None:
        raise ValueError(f"No library for zone {zone_id!r} and algorithm {algorithm!r}")
    # A library's own zone need not be a declared area; say so instead of guessing a status.
    area = next(
        (
            row.status
            for row in manifest.areas
            if (row.zone_id, row.algorithm) == (zone_id, algorithm)
        ),
        "not_declared",
    )
    return library, B6LibraryIdentity(
        library_version=manifest.library_version,
        manifest_sha256=expected_manifest_sha256,
        json_sha256=manifest.files[entry.json_path].sha256,
        parquet_sha256=manifest.files[entry.parquet_path].sha256,
        schema_version=manifest.library_schema_version,
        zone_id=zone_id,
        algorithm=algorithm,
        area_status=area,
        qualification=entry.qualification,
    )
