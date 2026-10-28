"""Offline rule replay for the site-watch V0 (§G3, #172): a pure core, no I/O.

Would the documented watch rule have fired on the forecast issued ``h`` days before a
labelled event, and at which horizons? This module answers that for one
`watch-labelset/v1` document, from archived Previous Runs artifacts and the zone's stored
normals, all handed in by the caller. It reads no file, calls no provider and touches no
object store: loading the inputs is the caller's job.

Every day goes through :func:`src.tools.climate.build_watch_series`, the evaluator the
read-only ``climate_normals`` tool uses. There is no second formula here, only the counting
the design note defines (``docs/design/forecast-backtest.md`` §Rule replay mode).

It measures rule triggers, not skill: ``confidence`` is null, and no probability, risk or
severity field exists. The report is GeoAI-internal (``watch-replay/v1``), not a shared
contract.
"""

from __future__ import annotations

from collections import Counter
from datetime import date, timedelta
from typing import Any

from src.forecast_backtest import BacktestValidationError, parse_rows
from src.tools.climate import (
    NORMALS_CAVEATS,
    WATCH_SERIES_VARIABLES,
    _finite,
    _stored_int,
    build_watch_series,
)
from src.utils.openmeteo import OPEN_METEO_DATA_LICENSE

REPORT_SCHEMA_VERSION = "watch-replay/v1"

#: Every horizon the Previous Runs archive serves, no sampling: a subset under-reports lead.
HORIZONS = tuple(range(1, 8))

#: Fixed by the docs (forecast-backtest.md, site-watch-v0.md §G3). The schema allows 0..2;
#: a wider window would inflate hits, so anything else is refused, not honoured.
MATCH_TOLERANCE_DAYS = 1

#: Below this many labels per variable the replay is "not measured", never a rate.
MIN_LABELS = 5

#: A mean Gregorian month, so a per-month rate does not depend on which months the window spans.
DAYS_PER_MONTH = 365.25 / 12

#: The two `Coverage.status` values the Previous Runs client writes (`src/forecast_archive.py`).
_ARCHIVE_STATUSES = ("available", "partial_archive")

_IDENTITY = ("zone_id", "provider", "model")

#: The five bounds of `docs/design/forecast-backtest.md` §Rule replay mode, copied verbatim
#: (whitespace-normalised). `test_the_five_bounds_are_verbatim_from_the_design_note` pins them.
REPLAY_BOUNDS: tuple[str, ...] = (
    "**Lead time is bounded by the archive.** Previous Runs stores 1..7 lead days "
    "(`fetch_daily` rejects anything else); no claim beyond 7 days.",
    "**Archived model ≠ production model.** The archive client defaults to `gfs_seamless`; "
    "the production `forecast_query` chain is AIFS (`ecmwf_aifs025_single`) first, "
    "Open-Meteo physical fallback. A replay on GFS says nothing about the AIFS path that the "
    "live watch will use. Where an AIFS Previous Runs archive is used, record its actual "
    "coverage window in the report (the Deposium brief notes 2025-02 → 2025-10 at writing "
    "time; confirm against the provider's coverage response, do not copy the figure).",
    "**Labels are not the reference.** Skill is measured against the ERA5 holdout above; "
    "vigilance bulletins are event labels for the trigger question only. They must not be "
    "read as a verification dataset.",
    "**Wind labels are indicative only.** A wind vigilance is issued on gusts; the archived "
    "series is `wind_speed` (provider: `wind_speed_10m_max`, daily maximum of the 10 m wind, "
    "`src/utils/openmeteo.py`). The two are not commensurable; report wind separately and "
    "say so.",
    "**No skill claim.** A hit rate is a rule-trigger count on `n_labels` episodes. It is not "
    "recall of a calibrated detector and does not lift the R13/R14 gates; "
    "`pattern_alert.confidence` stays `null`.",
)

#: Which predicate this report replayed. Deposium's WatchRule owns the business mapping; the
#: replay applies the documented one offline and cannot check it against Deposium's code.
RULE = {
    "predicate": (
        "notify: an entry with status 'available' and category 'extreme_high' "
        "(watch_level == notify in the documented watch-finding/v1 rule)"
    ),
    "counts": "notify only; the watch tier (above_normal and value >= 0.9·p90) is not replayed",
    "hit_anchor": (
        "origin_date = start - h; a UTC day d in [start - 1, end + 1] within the labelset "
        "window is read from the artifact at horizon d - origin_date, in 1..7"
    ),
    "source": "docs/design/forecast-backtest.md, section Rule replay mode",
    "parity": "Deposium WatchRule parity not verified",
}


class ReplayValidationError(ValueError):
    """An input the replay refuses rather than reports on."""


def _is_notify(entry: dict) -> bool:
    """A ``notify`` under the documented rule: ``extreme_high`` on a classified day.

    ``extreme_high`` ⇒ ``notify`` (forecast-backtest.md §Rule replay mode, *The rule*). Every
    G3 metric counts notify only, so the ``watch`` tier is never computed here. A ``dry``
    precipitation day is classified but carries no category, so it is never a notify.
    """
    return entry.get("status") == "available" and entry.get("category") == "extreme_high"


def _date(value: Any, field: str) -> date:
    if not isinstance(value, str):
        raise ReplayValidationError(f"{field} must be an ISO date")
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise ReplayValidationError(f"{field} must be an ISO date") from exc


def _check_coverage_claims(horizon: int, coverage: dict, rows: list) -> None:
    """An artifact's rows must agree with its own `Coverage` (`src/forecast_archive.py`).

    Every row's `valid_date` lies in `[start_date, end_date]`, the number of distinct
    `valid_date`s equals `available_days`, `available_days` is at most `expected_days`, and
    `status` is `available` iff the two are equal (`partial_archive` otherwise). An artifact
    with no archived day is valid when its coverage says so (``available_days == 0``). A
    coverage that carries pairs (``expected_pairs``) is checked on pairs instead: the rows
    equal ``available_pairs``, ``available_days`` counts the days with every variable, and
    ``status`` is ``available`` iff every pair is.
    """
    start = _date(coverage.get("start_date"), f"horizon {horizon} coverage.start_date")
    end = _date(coverage.get("end_date"), f"horizon {horizon} coverage.end_date")
    expected_days = coverage.get("expected_days")
    if isinstance(expected_days, bool) or not isinstance(expected_days, int):
        raise ReplayValidationError(f"horizon {horizon}: coverage.expected_days must be an integer")
    available_days = coverage.get("available_days")
    if isinstance(available_days, bool) or not isinstance(available_days, int):
        raise ReplayValidationError(
            f"horizon {horizon}: coverage.available_days must be an integer"
        )
    valid_dates = {row.valid_date for row in rows}
    if any(day < start or day > end for day in valid_dates):
        raise ReplayValidationError(
            f"horizon {horizon}: an archived row's valid_date lies outside its own "
            f"coverage.start_date/end_date"
        )
    # Pair coverage (#193): the normalizer keeps a day's complete variables when another one
    # has a gap, so a day can hold rows without being complete. ``available_days`` then counts
    # the days carrying every variable, and the status follows the pairs, not the days.
    expected_pairs = coverage.get("expected_pairs")
    pairs = isinstance(expected_pairs, int) and not isinstance(expected_pairs, bool)
    if pairs and expected_pairs > 0:
        available_pairs = coverage.get("available_pairs")
        if available_pairs != len(rows) or available_pairs > expected_pairs:
            raise ReplayValidationError(
                f"horizon {horizon}: coverage.available_pairs disagrees with the rows archived"
            )
        # A rectangle of whole days x whole variables, or the per-day count below is a guess.
        if expected_days <= 0:
            raise ReplayValidationError(
                f"horizon {horizon}: coverage.expected_days must be positive with pairs"
            )
        if expected_pairs % expected_days:
            raise ReplayValidationError(
                f"horizon {horizon}: coverage.expected_pairs is not a whole number of "
                f"variables per day over coverage.expected_days"
            )
        variables_per_day = expected_pairs // expected_days
        by_day: dict[date, set[str]] = {}
        for row in rows:
            by_day.setdefault(row.valid_date, set()).add(row.variable)
        archived_days = sum(len(names) == variables_per_day for names in by_day.values())
        complete = available_pairs == expected_pairs
    else:
        archived_days = len(valid_dates)
        complete = available_days == expected_days
    if archived_days != available_days:
        raise ReplayValidationError(
            f"horizon {horizon}: coverage.available_days disagrees with the distinct "
            f"valid_date count actually archived"
        )
    if available_days > expected_days:
        raise ReplayValidationError(
            f"horizon {horizon}: coverage.available_days exceeds coverage.expected_days"
        )
    wants = "available" if complete else "partial_archive"
    if coverage.get("status") != wants:
        raise ReplayValidationError(
            f"horizon {horizon}: coverage.status must be {wants!r} for "
            f"{available_days}/{expected_days} archived days"
        )


def _validate_archives(archives: Any) -> tuple[dict[str, str], dict[int, dict], dict[int, list]]:
    """One Previous Runs artifact per horizon 1..7, one zone/provider/model, clean rows.

    Rows must also agree with their own artifact's coverage claims (`_check_coverage_claims`).
    """
    if not isinstance(archives, (list, tuple)):
        raise ReplayValidationError("archives must be a list of forecast-archive-panel artifacts")
    by_h: dict[int, dict] = {}
    for artifact in archives:
        coverage = artifact.get("coverage") if isinstance(artifact, dict) else None
        if not isinstance(coverage, dict) or not isinstance(artifact.get("rows"), list):
            raise ReplayValidationError(
                "each archive needs a Previous Runs coverage block and a rows list"
            )
        horizon = coverage.get("horizon_days")
        if isinstance(horizon, bool) or not isinstance(horizon, int):
            raise ReplayValidationError("coverage.horizon_days must be an integer")
        if coverage.get("status") not in _ARCHIVE_STATUSES:
            raise ReplayValidationError(
                f"coverage.status must be one of {_ARCHIVE_STATUSES} (a Previous Runs archive)"
            )
        for key in _IDENTITY:
            value = coverage.get(key)
            if not isinstance(value, str) or not value:
                raise ReplayValidationError(f"coverage {key} must be a non-empty string")
        if horizon in by_h:
            raise ReplayValidationError(f"duplicate artifact for horizon {horizon}")
        by_h[horizon] = artifact
    if sorted(by_h) != list(HORIZONS):
        raise ReplayValidationError(
            f"horizons must be exactly 1..7, one artifact each; got {sorted(by_h)}"
        )
    # Every value is already a non-empty string (checked above), so this set never has to
    # hash anything else.
    identities = {tuple(a["coverage"].get(key) for key in _IDENTITY) for a in by_h.values()}
    if len(identities) != 1:
        raise ReplayValidationError("archives must share one zone_id, provider and model")
    identity = dict(zip(_IDENTITY, identities.pop(), strict=True))

    rows_by_h: dict[int, list] = {}
    for horizon, artifact in by_h.items():
        expected = {**identity, "horizon_days": horizon}
        rows = artifact["rows"]
        for row in rows:
            if not isinstance(row, dict) or any(row.get(k) != v for k, v in expected.items()):
                raise ReplayValidationError(
                    f"a row of the horizon-{horizon} artifact does not match its coverage"
                )
        # `parse_rows` refuses an empty list; an artifact with no archived day is valid and
        # simply covers nothing.
        try:
            parsed = (
                parse_rows(rows, kind="prediction", provider=identity["provider"]) if rows else []
            )
        except BacktestValidationError as exc:
            raise ReplayValidationError(f"horizon {horizon}: {exc}") from exc
        seen: set[tuple[date, str]] = set()
        for row in parsed:
            key = (row.valid_date, row.variable)
            if key in seen:
                raise ReplayValidationError(
                    f"horizon {horizon}: duplicate row for {row.variable} on {row.valid_date}"
                )
            seen.add(key)
        _check_coverage_claims(horizon, artifact["coverage"], parsed)
        rows_by_h[horizon] = parsed
    return identity, by_h, rows_by_h


def _validate_normals(normals: Any, zone_id: str) -> str:
    """The archive zone's rows, one per (variable, month), from one normal window.

    `build_watch_series` keys normals by (variable, month) only and keeps the last duplicate,
    so another zone's rows or a mixed file would classify silently under this zone's name.
    """
    if not isinstance(normals, (list, tuple)) or not all(isinstance(r, dict) for r in normals):
        raise ReplayValidationError("normals must be a list of stored normals rows")
    seen: set[tuple[Any, int | None]] = set()
    for row in normals:
        if row.get("zone_id") != zone_id:
            raise ReplayValidationError(
                f"normals row for zone {row.get('zone_id')!r} does not match the archive "
                f"zone {zone_id!r}"
            )
        variable = row.get("variable")
        if not isinstance(variable, str) or not variable:
            raise ReplayValidationError(
                f"normals row variable must be a non-empty string: {variable!r}"
            )
        window_label = row.get("window_label")
        if not isinstance(window_label, str) or not window_label:
            raise ReplayValidationError(
                f"window_label must be a non-empty string: {window_label!r}"
            )
        # `variable` and `window_label` are checked above: this key is always hashable.
        key = (variable, _stored_int(row.get("month")))
        if key in seen:
            raise ReplayValidationError(f"duplicate normals row for {key}")
        seen.add(key)
    window_labels = {row.get("window_label") for row in normals}
    if len(window_labels) != 1:
        raise ReplayValidationError(
            f"normals must carry exactly one window_label; got {sorted(window_labels)}"
        )
    return window_labels.pop()


def _validate_labelset(
    labelset: Any, zone_id: str
) -> tuple[str, date, date, list[tuple[str, date, date]]]:
    """What the committed schema does not enforce; the schema itself is checked by the caller."""
    if not isinstance(labelset, dict) or labelset.get("schema_version") != "watch-labelset/v1":
        raise ReplayValidationError("labelset schema_version must be 'watch-labelset/v1'")
    variable = labelset.get("variable")
    if variable not in WATCH_SERIES_VARIABLES:
        raise ReplayValidationError(f"labelset variable must be one of {WATCH_SERIES_VARIABLES}")
    window_start = _date(labelset.get("window_start"), "window_start")
    window_end = _date(labelset.get("window_end"), "window_end")
    if window_start > window_end:
        raise ReplayValidationError("labelset window_start is after window_end")
    labels = labelset.get("labels")
    if not isinstance(labels, list):
        raise ReplayValidationError("labelset labels must be a list")
    parsed: list[tuple[str, date, date]] = []
    seen: set[str] = set()
    for label in labels:
        if not isinstance(label, dict) or label.get("schema_version") != "watch-label/v1":
            raise ReplayValidationError("every label schema_version must be 'watch-label/v1'")
        event_id = label.get("event_id")
        if not isinstance(event_id, str) or not event_id or event_id in seen:
            raise ReplayValidationError(f"event_id must be non-empty and unique: {event_id!r}")
        seen.add(event_id)
        if label.get("variable") != variable:
            raise ReplayValidationError(
                f"label {event_id}: variable differs from the labelset variable"
            )
        if label.get("place_ref") != f"zone:{zone_id}":
            raise ReplayValidationError(
                f"label {event_id}: place_ref must be 'zone:{zone_id}', the archive zone"
            )
        tolerance = label.get("match_tolerance_days")
        if isinstance(tolerance, bool) or tolerance != MATCH_TOLERANCE_DAYS:
            raise ReplayValidationError(
                f"label {event_id}: match_tolerance_days must be {MATCH_TOLERANCE_DAYS}"
            )
        start = _date(label.get("start"), f"label {event_id} start")
        end = _date(label.get("end"), f"label {event_id} end")
        if start > end:
            raise ReplayValidationError(f"label {event_id}: start is after end")
        if start < window_start or end > window_end:
            raise ReplayValidationError(f"label {event_id} lies outside the labelset window")
        parsed.append((event_id, start, end))
    if labelset.get("exhaustive") is True:
        coverage_statement = labelset.get("coverage_statement")
        if not isinstance(coverage_statement, str) or not coverage_statement:
            raise ReplayValidationError(
                "an exhaustive labelset must carry a non-empty coverage_statement: "
                "false_triggers_per_month is only meaningful when a curator states what "
                "the labelset covers"
            )
    return variable, window_start, window_end, parsed


def _classify_horizon(
    rows: list, *, variable: str, window_start: date, window_end: date, normals: list[dict]
) -> tuple[list[date], dict[date, bool], dict[str, int]]:
    """``(sorted archived days, {evaluable day: is_notify}, unclassified counts)`` for one h.

    A day is evaluable when the evaluator classified it (``available``) or kept it as ``dry``.
    A missing or unclassified day is absent from the map: it is never a quiet day.
    """
    kept = [
        row
        for row in rows
        if row.variable == variable and window_start <= row.valid_date <= window_end
    ]
    values = [
        {"variable": row.variable, "date": row.valid_date.isoformat(), "value": row.value}
        for row in kept
    ]
    series, unclassified = build_watch_series(values, normals=normals)
    evaluable = {
        date.fromisoformat(entry["day"]): _is_notify(entry)
        for entry in series
        if entry.get("status") in ("available", "dry")
    }
    reasons = Counter(item["reason"] for item in unclassified)
    return sorted(row.valid_date for row in kept), evaluable, dict(sorted(reasons.items()))


def _hit(
    days_by_h: dict[int, dict[date, bool]],
    *,
    start: date,
    end: date,
    horizon: int,
    window_start: date,
    window_end: date,
) -> bool | None:
    """Did the run issued ``horizon`` days before ``start`` notify inside the label window?

    The origin is ``start - horizon``, so ``origin < start`` holds by construction. Each day of
    ``[start - 1, end + 1]`` (inside the labelset window) is read from the artifact whose
    horizon is that day minus the origin, when it is 1..7. ``True`` if any such evaluable
    day is a notify, ``False`` if some are evaluable and none is, ``None`` if none is.
    """
    origin = start - timedelta(days=horizon)
    day = max(start - timedelta(days=MATCH_TOLERANCE_DAYS), window_start)
    last = min(end + timedelta(days=MATCH_TOLERANCE_DAYS), window_end)
    observed: list[bool] = []
    while day <= last:
        served = (day - origin).days  # the horizon at which this run forecast `day`
        if served in days_by_h and day in days_by_h[served]:
            observed.append(days_by_h[served][day])
        day += timedelta(days=1)
    return any(observed) if observed else None


def _rate(numerator: int, denominator: float, *, measured: bool) -> float | None:
    return round(numerator / denominator, 4) if measured and denominator else None


def replay_labelset(archives: Any, *, normals: Any, labelset: Any) -> dict:
    """Replay the documented rule for one `watch-labelset/v1` and return a `watch-replay/v1`.

    Args:
        archives: The seven `forecast-archive-panel` artifacts (``{coverage, rows}``), one per
            horizon 1..7, for one zone, provider and model.
        normals: The zone's stored monthly normals rows (v1 or v2), as read from its parquet.
        labelset: One `watch-labelset/v1` document; its labels name the archive zone.

    Raises:
        ReplayValidationError: on any input the replay refuses rather than reports on.
    """
    identity, by_h, rows_by_h = _validate_archives(archives)
    zone_id = identity["zone_id"]
    window_label = _validate_normals(normals, zone_id)
    variable, window_start, window_end, labels = _validate_labelset(labelset, zone_id)
    exhaustive = labelset.get("exhaustive") is True
    trigger_key = "false_triggers_per_month" if exhaustive else "unmatched_triggers_per_month"
    window_days = (window_end - window_start).days + 1

    archived: dict[int, list[date]] = {}
    days_by_h: dict[int, dict[date, bool]] = {}
    reasons: dict[int, dict[str, int]] = {}
    for horizon in HORIZONS:
        archived[horizon], days_by_h[horizon], reasons[horizon] = _classify_horizon(
            rows_by_h[horizon],
            variable=variable,
            window_start=window_start,
            window_end=window_end,
            normals=list(normals),
        )

    label_reports = []
    widened: set[date] = set()
    for event_id, start, end in labels:
        hits = {
            horizon: _hit(
                days_by_h,
                start=start,
                end=end,
                horizon=horizon,
                window_start=window_start,
                window_end=window_end,
            )
            for horizon in HORIZONS
        }
        lead = max((h for h, hit in hits.items() if hit), default=None)
        label_reports.append(
            {
                "event_id": event_id,
                "start": start.isoformat(),
                "end": end.isoformat(),
                "hit": {str(h): hit for h, hit in hits.items()},
                "lead": lead,
            }
        )
        day = start - timedelta(days=MATCH_TOLERANCE_DAYS)
        while day <= end + timedelta(days=MATCH_TOLERANCE_DAYS):
            widened.add(day)
            day += timedelta(days=1)

    evaluated = {
        h: sum(report["hit"][str(h)] is not None for report in label_reports) for h in HORIZONS
    }
    # Five labels with no evaluable day at any horizon (e.g. precipitation on v1 normals) are
    # not a measurement of zero: no rate, the raw counts only.
    if len(labels) < MIN_LABELS:
        status = "not_measured"
    elif not any(evaluated.values()):
        status = "not_evaluable"
    else:
        status = "measured"
    measured = status == "measured"

    horizons: dict[str, dict] = {}
    for horizon in HORIZONS:
        hits = sum(report["hit"][str(horizon)] is True for report in label_reports)
        outside = {d: notify for d, notify in days_by_h[horizon].items() if d not in widened}
        notify_days = sum(outside.values())
        horizons[str(horizon)] = {
            "hits": hits,
            "evaluated": evaluated[horizon],
            "hit_rate": _rate(
                hits, evaluated[horizon], measured=measured and evaluated[horizon] >= MIN_LABELS
            ),
            "outside_label_notify_days": notify_days,
            "outside_label_evaluable_days": len(outside),
            trigger_key: _rate(notify_days, len(outside) / DAYS_PER_MONTH, measured=measured),
        }

    precipitation_thresholds = {
        threshold
        for row in normals
        if row.get("variable") == "precipitation"
        and (threshold := _finite(row.get("wet_day_threshold_mm"))) is not None
    }
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "variable": variable,
        "zone_id": zone_id,
        "status": status,
        "n_labels": len(labels),
        "labels": label_reports,
        "leads": [report["lead"] for report in label_reports],
        "horizons": horizons,
        "coverage": {
            "window_days": window_days,
            "horizons_served": [h for h in HORIZONS if archived[h]],
            "per_horizon": {
                str(h): {
                    "archived_days": len(archived[h]),
                    # The days actually archived; `archive_coverage` holds the requested span.
                    "archived_first_day": archived[h][0].isoformat() if archived[h] else None,
                    "archived_last_day": archived[h][-1].isoformat() if archived[h] else None,
                    "share": round(len(archived[h]) / window_days, 4),
                    "evaluable_days": len(days_by_h[h]),
                    "unclassified": reasons[h],
                    "archive_coverage": dict(by_h[h]["coverage"]),
                }
                for h in HORIZONS
            },
        },
        "provenance": {
            **identity,
            "horizons": list(HORIZONS),
            "normals_window_label": window_label,
            # As the climate_normals tool reports it: a missing or NULL column is v1.
            "normals_schema_version": sorted(
                {_stored_int(row.get("schema_version"), 1) or 1 for row in normals}
            )
            or [1],
            # Read from the precipitation rows only; a wind row never carries one.
            "wet_day_threshold_mm": sorted(precipitation_thresholds),
            "license": OPEN_METEO_DATA_LICENSE,
            "window_start": window_start.isoformat(),
            "window_end": window_end.isoformat(),
            "exhaustive": exhaustive,
            "coverage_statement": labelset.get("coverage_statement"),
            "match_tolerance_days": MATCH_TOLERANCE_DAYS,
            "day_grain": "utc_day",
        },
        "rule": dict(RULE),
        "bounds": list(REPLAY_BOUNDS),
        "caveats": list(NORMALS_CAVEATS),
        "confidence": None,
    }
