"""Leakage-safe forecast backtest primitives.

The runner consumes archived predictions and an independently materialized ERA5
reference. It deliberately has no Open-Meteo client dependency: a current
forecast endpoint cannot be used to fabricate historical predictions.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

DEFAULT_PRECIPITATION_EVENT_THRESHOLD_MM = 0.1
DEFAULT_RELIABILITY_BINS = 10
PRECIPITATION_EVENT_OPERATOR = ">"


class BacktestValidationError(ValueError):
    """Input or leakage-policy violation."""


@dataclass(frozen=True)
class ProbabilityEvent:
    operator: str
    threshold_mm: float


@dataclass(frozen=True)
class BacktestRow:
    zone_id: str
    origin_date: date
    valid_date: date
    horizon_days: int
    variable: str
    value: float
    provider: str
    probability: float | None = None
    probability_event: ProbabilityEvent | None = None


def _date(value: Any, field: str) -> date:
    if not isinstance(value, str):
        raise BacktestValidationError(f"{field} must be an ISO date")
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise BacktestValidationError(f"{field} must be an ISO date") from exc


def _row(payload: dict[str, Any], *, kind: str, provider: str = "reference") -> BacktestRow:
    required = ("zone_id", "valid_date", "variable", "value")
    missing = [key for key in required if key not in payload]
    if kind == "prediction" and "origin_date" not in payload:
        missing.append("origin_date")
    if missing:
        raise BacktestValidationError(f"{kind} row missing: {', '.join(missing)}")
    try:
        value = float(payload["value"])
    except (TypeError, ValueError, OverflowError) as exc:
        raise BacktestValidationError(f"{kind} value must be numeric") from exc
    if not math.isfinite(value):
        raise BacktestValidationError(f"{kind} value must be finite")
    if kind == "prediction" and not provider:
        raise BacktestValidationError("prediction provider is required")
    origin = (
        _date(payload["origin_date"], "origin_date")
        if kind == "prediction"
        else _date(payload["valid_date"], "valid_date")
    )
    valid = _date(payload["valid_date"], "valid_date")
    raw_horizon = payload.get("horizon_days", (valid - origin).days)
    if kind == "prediction" and (
        isinstance(raw_horizon, bool) or not isinstance(raw_horizon, int) or raw_horizon < 1
    ):
        raise BacktestValidationError("horizon_days must be a positive integer")
    horizon = raw_horizon if kind == "prediction" else 0
    if kind == "prediction" and (horizon < 1 or (valid - origin).days != horizon):
        raise BacktestValidationError(
            "horizon_days must equal valid_date - origin_date and be >= 1"
        )
    zone = payload["zone_id"]
    variable = payload["variable"]
    if not isinstance(zone, str) or not zone or not isinstance(variable, str) or not variable:
        raise BacktestValidationError("zone_id and variable must be non-empty strings")
    probability = payload.get("probability") if kind == "prediction" else None
    if probability is not None:
        if isinstance(probability, bool):
            raise BacktestValidationError("probability must be numeric, not boolean")
        try:
            probability = float(probability)
        except (TypeError, ValueError, OverflowError) as exc:
            raise BacktestValidationError("probability must be numeric") from exc
        if not math.isfinite(probability) or not 0 <= probability <= 1:
            raise BacktestValidationError("probability must be finite and between 0 and 1")
    raw_probability_event = payload.get("probability_event") if kind == "prediction" else None
    probability_event = None
    if probability is not None:
        if variable != "precipitation":
            raise BacktestValidationError(
                "probability is only supported for precipitation predictions"
            )
        if not isinstance(raw_probability_event, dict):
            raise BacktestValidationError(
                "precipitation probability requires probability_event metadata"
            )
        if set(raw_probability_event) != {"operator", "threshold_mm"}:
            raise BacktestValidationError(
                "probability_event must contain only operator and threshold_mm"
            )
        operator = raw_probability_event["operator"]
        if operator != PRECIPITATION_EVENT_OPERATOR:
            raise BacktestValidationError(
                f"probability_event.operator must be {PRECIPITATION_EVENT_OPERATOR!r}"
            )
        raw_threshold = raw_probability_event["threshold_mm"]
        if isinstance(raw_threshold, bool):
            raise BacktestValidationError(
                "probability_event.threshold_mm must be a finite number >= 0"
            )
        try:
            threshold = float(raw_threshold)
        except (TypeError, ValueError, OverflowError) as exc:
            raise BacktestValidationError(
                "probability_event.threshold_mm must be a finite number >= 0"
            ) from exc
        if not math.isfinite(threshold) or threshold < 0:
            raise BacktestValidationError(
                "probability_event.threshold_mm must be a finite number >= 0"
            )
        probability_event = ProbabilityEvent(operator=operator, threshold_mm=threshold)
    elif raw_probability_event is not None:
        raise BacktestValidationError("probability_event requires probability")
    return BacktestRow(
        zone,
        origin,
        valid,
        horizon,
        variable,
        value,
        provider,
        probability,
        probability_event,
    )


def manifest_digest(path: str | Path) -> str:
    """Return the SHA-256 digest of an archive manifest, byte-for-byte."""
    digest = hashlib.sha256()
    try:
        with Path(path).open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise BacktestValidationError(f"cannot read manifest {path}: {exc}") from exc
    return digest.hexdigest()


def _rows_from_text(
    text: str,
    *,
    source: Path,
    kind: str,
    provider: str | None,
) -> list[BacktestRow]:
    try:
        decoded = json.loads(text)
    except json.JSONDecodeError:
        items = []
        for line_number, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                continue
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise BacktestValidationError(
                    f"invalid JSONL at line {line_number}: {exc.msg}"
                ) from exc
            except ValueError as exc:  # an integer longer than Python's digit limit (#221)
                raise BacktestValidationError(
                    f"invalid JSONL at line {line_number}: {digit_limit_message()}"
                ) from exc
    except ValueError as exc:  # an integer longer than Python's digit limit (#221)
        raise BacktestValidationError(f"{source}: {digit_limit_message()}") from exc
    else:
        if isinstance(decoded, list):
            items = decoded
        elif isinstance(decoded, dict) and isinstance(decoded.get("rows"), list):
            items = decoded["rows"]
        elif isinstance(decoded, dict):
            items = [decoded]
        else:
            items = None
    if not isinstance(items, list) or not items:
        raise BacktestValidationError(f"{source} must contain a non-empty rows array")
    if kind == "prediction" and not provider:
        raise BacktestValidationError("provider is required when loading predictions")
    return [_row(item, kind=kind, provider=provider or "reference") for item in items]


def digit_limit_message() -> str:
    """The refusal for a JSON integer longer than Python parses (``json.loads`` ValueError, #221)."""
    return f"a number has more than {sys.get_int_max_str_digits()} digits"


def load_rows_with_digest(
    path: str | Path,
    *,
    kind: str,
    provider: str | None = None,
) -> tuple[list[BacktestRow], str]:
    """Load rows and fingerprint the exact bytes parsed by the loader."""
    source = Path(path)
    try:
        raw = source.read_bytes()
    except OSError as exc:
        raise BacktestValidationError(f"cannot read {source}: {exc}") from exc
    rows = _rows_from_text(
        raw.decode("utf-8"),
        source=source,
        kind=kind,
        provider=provider,
    )
    return rows, hashlib.sha256(raw).hexdigest()


def load_rows(path: str | Path, *, kind: str, provider: str | None = None) -> list[BacktestRow]:
    """Load a JSON array or JSONL archive, rejecting malformed rows."""
    rows, _digest = load_rows_with_digest(path, kind=kind, provider=provider)
    return rows


def parse_rows(
    items: list[dict[str, Any]] | tuple[dict[str, Any], ...],
    *,
    kind: str,
    provider: str | None = None,
) -> list[BacktestRow]:
    """Convert already verified row objects using the regular row contract."""
    if not items:
        raise BacktestValidationError("verified archive must contain non-empty rows")
    if kind == "prediction" and not provider:
        raise BacktestValidationError("provider is required when parsing predictions")
    return [_row(item, kind=kind, provider=provider or "reference") for item in items]


def validate_inputs(
    predictions: list[BacktestRow],
    reference: list[BacktestRow],
    *,
    training_end: date,
    holdout_start: date,
    holdout_end: date,
) -> None:
    """Fail closed on overlap, leakage, missing reference values, or duplicate keys."""
    if training_end >= holdout_start:
        raise BacktestValidationError("training and holdout periods overlap")
    if holdout_start > holdout_end:
        raise BacktestValidationError("holdout_start must be before holdout_end")
    ref_keys: set[tuple[str, date, str]] = set()
    for row in reference:
        key = (row.zone_id, row.valid_date, row.variable)
        if key in ref_keys:
            raise BacktestValidationError(f"duplicate reference key: {key}")
        ref_keys.add(key)
        if not holdout_start <= row.valid_date <= holdout_end:
            raise BacktestValidationError(f"reference row outside holdout: {key}")
    pred_keys: set[tuple[str, date, date, str, str]] = set()
    for row in predictions:
        if row.origin_date > training_end and row.origin_date < holdout_start:
            raise BacktestValidationError("prediction origin falls in the baseline/holdout gap")
        if row.valid_date < holdout_start or row.valid_date > holdout_end:
            raise BacktestValidationError("prediction valid_date outside holdout")
        if row.origin_date >= row.valid_date:
            raise BacktestValidationError("prediction origin_date must precede valid_date")
        key = (row.zone_id, row.origin_date, row.valid_date, row.variable, row.provider)
        if key in pred_keys:
            raise BacktestValidationError(f"duplicate prediction key: {key}")
        pred_keys.add(key)
        if (row.zone_id, row.valid_date, row.variable) not in ref_keys:
            raise BacktestValidationError(f"missing reference value for prediction: {key}")


def _metrics(pairs: list[tuple[float, float]]) -> dict[str, float | int]:
    errors = [prediction - actual for prediction, actual in pairs]
    n = len(errors)
    return {
        "n": n,
        "mae": sum(abs(error) for error in errors) / n,
        "rmse": math.sqrt(sum(error * error for error in errors) / n),
        "bias": sum(errors) / n,
    }


def _validate_probability_settings(
    precipitation_event_threshold_mm: float,
    reliability_bin_count: int,
) -> tuple[float, int]:
    if isinstance(precipitation_event_threshold_mm, bool):
        raise BacktestValidationError(
            "precipitation_event_threshold_mm must be a finite number >= 0"
        )
    try:
        threshold = float(precipitation_event_threshold_mm)
    except (TypeError, ValueError) as exc:
        raise BacktestValidationError(
            "precipitation_event_threshold_mm must be a finite number >= 0"
        ) from exc
    if not math.isfinite(threshold) or threshold < 0:
        raise BacktestValidationError(
            "precipitation_event_threshold_mm must be a finite number >= 0"
        )
    if (
        isinstance(reliability_bin_count, bool)
        or not isinstance(reliability_bin_count, int)
        or not 2 <= reliability_bin_count <= 100
    ):
        raise BacktestValidationError("reliability_bin_count must be an integer in 2..100")
    return threshold, reliability_bin_count


def probability_evaluation_metadata(
    precipitation_event_threshold_mm: float = DEFAULT_PRECIPITATION_EVENT_THRESHOLD_MM,
    reliability_bin_count: int = DEFAULT_RELIABILITY_BINS,
) -> dict[str, Any]:
    threshold, bin_count = _validate_probability_settings(
        precipitation_event_threshold_mm,
        reliability_bin_count,
    )
    return {
        "precipitation_event": {
            "operator": PRECIPITATION_EVENT_OPERATOR,
            "threshold_mm": threshold,
        },
        "reliability_bin_count": bin_count,
        "event_rate_scope": "evaluation_holdout_only_not_training_climatology",
    }


def _validate_probability_contract(
    predictions: list[BacktestRow],
    *,
    precipitation_event_threshold_mm: float,
) -> None:
    for row in predictions:
        if row.probability is None:
            if row.probability_event is not None:
                raise BacktestValidationError("probability_event requires probability")
            continue
        if row.variable != "precipitation":
            raise BacktestValidationError(
                "probability is only supported for precipitation predictions"
            )
        if (
            isinstance(row.probability, bool)
            or not isinstance(row.probability, (int, float))
            or not math.isfinite(row.probability)
            or not 0 <= row.probability <= 1
        ):
            raise BacktestValidationError("probability must be finite and between 0 and 1")
        event = row.probability_event
        if not isinstance(event, ProbabilityEvent):
            raise BacktestValidationError(
                "precipitation probability requires probability_event metadata"
            )
        if event.operator != PRECIPITATION_EVENT_OPERATOR:
            raise BacktestValidationError(
                f"probability_event.operator must be {PRECIPITATION_EVENT_OPERATOR!r}"
            )
        if (
            isinstance(event.threshold_mm, bool)
            or not isinstance(event.threshold_mm, (int, float))
            or not math.isfinite(event.threshold_mm)
            or event.threshold_mm < 0
        ):
            raise BacktestValidationError(
                "probability_event.threshold_mm must be a finite number >= 0"
            )
        if not math.isclose(
            event.threshold_mm,
            precipitation_event_threshold_mm,
            rel_tol=0,
            abs_tol=1e-12,
        ):
            raise BacktestValidationError(
                "probability_event.threshold_mm does not match the evaluation threshold"
            )


def _probability_metrics(
    probabilities: list[float],
    observed: list[bool],
    *,
    reliability_bin_count: int,
) -> dict[str, Any]:
    buckets: list[list[tuple[float, bool]]] = [[] for _ in range(reliability_bin_count)]
    for probability, event in zip(probabilities, observed, strict=True):
        bucket_index = min(int(probability * reliability_bin_count), reliability_bin_count - 1)
        buckets[bucket_index].append((probability, event))

    reliability_bins: list[dict[str, Any]] = []
    expected_calibration_error = 0.0
    maximum_calibration_error = 0.0
    total = len(probabilities)
    for index, bucket in enumerate(buckets):
        count = len(bucket)
        lower = index / reliability_bin_count
        upper = (index + 1) / reliability_bin_count
        if count:
            mean_probability = sum(probability for probability, _ in bucket) / count
            observed_frequency = sum(event for _, event in bucket) / count
            calibration_gap = mean_probability - observed_frequency
            absolute_gap = abs(calibration_gap)
            expected_calibration_error += count / total * absolute_gap
            maximum_calibration_error = max(maximum_calibration_error, absolute_gap)
        else:
            mean_probability = None
            observed_frequency = None
            calibration_gap = None
        reliability_bins.append(
            {
                "index": index,
                "lower_bound": lower,
                "upper_bound": upper,
                "upper_bound_inclusive": index == reliability_bin_count - 1,
                "n": count,
                "mean_probability": mean_probability,
                "observed_frequency": observed_frequency,
                "calibration_gap": calibration_gap,
            }
        )

    event_rate = sum(observed) / total
    mean_probability = sum(probabilities) / total
    return {
        "brier_score": sum(
            (probability - event) ** 2
            for probability, event in zip(probabilities, observed, strict=True)
        )
        / total,
        "evaluation_event_rate": event_rate,
        "mean_probability": mean_probability,
        "calibration_bias": mean_probability - event_rate,
        "expected_calibration_error": expected_calibration_error,
        "maximum_calibration_error": maximum_calibration_error,
        "reliability_bins": reliability_bins,
    }


def _bootstrap_ci(
    errors: list[float], *, seed: int, samples: int = 200, max_rows: int = 10_000
) -> tuple[float, float] | None:
    if len(errors) < 2:
        return None
    rng = random.Random(seed)
    if len(errors) > max_rows:
        # Keep a deterministic, evenly spaced subset without duplicating or dropping the
        # distribution through a second sampling-with-replacement pass.
        offset = rng.randrange(len(errors))
        errors = [
            errors[(offset + i * len(errors) // max_rows) % len(errors)] for i in range(max_rows)
        ]
    means: list[float] = []
    for _ in range(samples):
        selected: list[float] = []
        for _ in errors:
            selected.append(errors[rng.randrange(len(errors))])
        means.append(sum(selected) / len(selected))
    means.sort()
    return means[int(samples * 0.025)], means[int(samples * 0.975)]


def run_backtest(
    predictions: list[BacktestRow],
    reference: list[BacktestRow],
    *,
    training_end: date,
    holdout_start: date,
    holdout_end: date,
    precipitation_event_threshold_mm: float = DEFAULT_PRECIPITATION_EVENT_THRESHOLD_MM,
    reliability_bin_count: int = DEFAULT_RELIABILITY_BINS,
) -> dict[str, Any]:
    probability_evaluation = probability_evaluation_metadata(
        precipitation_event_threshold_mm,
        reliability_bin_count,
    )
    precipitation_event_threshold_mm = probability_evaluation["precipitation_event"]["threshold_mm"]
    reliability_bin_count = probability_evaluation["reliability_bin_count"]
    _validate_probability_contract(
        predictions,
        precipitation_event_threshold_mm=precipitation_event_threshold_mm,
    )
    validate_inputs(
        predictions,
        reference,
        training_end=training_end,
        holdout_start=holdout_start,
        holdout_end=holdout_end,
    )
    ref = {(row.zone_id, row.valid_date, row.variable): row.value for row in reference}
    groups: dict[tuple[str, str, int], list[tuple[BacktestRow, float]]] = defaultdict(list)
    for row in predictions:
        groups[(row.provider, row.variable, row.horizon_days)].append(
            (row, ref[(row.zone_id, row.valid_date, row.variable)])
        )
    metrics = []
    for (provider, variable, horizon), pairs in sorted(groups.items()):
        numeric = [(row.value, actual) for row, actual in pairs]
        item: dict[str, Any] = {
            "provider": provider,
            "variable": variable,
            "horizon_days": horizon,
            **_metrics(numeric),
        }
        seed = int(hashlib.sha256(f"{provider}:{variable}:{horizon}".encode()).hexdigest()[:8], 16)
        ci = _bootstrap_ci(
            [prediction - actual for prediction, actual in numeric],
            seed=seed,
        )
        if ci:
            item["bias_ci_95"] = list(ci)
        else:
            item["bias_ci_95"] = None
        if variable == "precipitation":
            observed = [actual > precipitation_event_threshold_mm for _, actual in numeric]
            item["occurrence_accuracy"] = sum(
                (prediction > precipitation_event_threshold_mm) == actual_flag
                for (prediction, _), actual_flag in zip(numeric, observed, strict=True)
            ) / len(numeric)
            probabilities = [row.probability for row, _ in pairs]
            item["probability_rows"] = sum(probability is not None for probability in probabilities)
            item["probability_completeness"] = item["probability_rows"] / len(probabilities)
            if item["probability_rows"] == 0:
                item["status"] = "point_only"
                item["probability_status"] = "unavailable"
                item["warnings"] = [
                    "probability_unavailable",
                    "probabilistic_metrics_omitted",
                ]
            elif item["probability_completeness"] < 1:
                item["status"] = "partial_probability"
                item["probability_status"] = "partial"
                item["warnings"] = [
                    "partial_probability",
                    "probabilistic_metrics_omitted",
                ]
            else:
                item["status"] = "complete"
                item["probability_status"] = "complete"
                item.update(
                    _probability_metrics(
                        [probability for probability in probabilities if probability is not None],
                        observed,
                        reliability_bin_count=reliability_bin_count,
                    )
                )
        metrics.append(item)
    return {
        "schema_version": "forecast-backtest/v2",
        "status": "completed",
        "training_end": training_end.isoformat(),
        "holdout": {"start": holdout_start.isoformat(), "end": holdout_end.isoformat()},
        "prediction_rows": len(predictions),
        "reference_rows": len(reference),
        "probability_evaluation": probability_evaluation,
        "metrics": metrics,
        "provenance": {"prediction_providers": sorted({row.provider for row in predictions})},
    }
