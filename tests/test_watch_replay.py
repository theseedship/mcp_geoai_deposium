"""The offline rule-replay core (site-watch V0 §G3, #172).

Synthetic fixtures only: no real label, no archive fetched from a provider, no network and
no object store. Every `event_id` is ``synthetic-NNN`` and every provenance URL points at
``example.invalid``. What this file pins is the replay's arithmetic and its refusals, not a
lead time: nothing here is evidence about any real zone.
"""

from __future__ import annotations

import json
import math
from datetime import date, timedelta
from pathlib import Path
from unittest import mock

import pytest
from jsonschema import Draft202012Validator, FormatChecker
from referencing import Registry, Resource

import src.tools.climate as climate
import src.watch_replay as watch_replay
from src.tools.climate import NORMALS_CAVEATS
from src.watch_replay import (
    DAYS_PER_MONTH,
    REPLAY_BOUNDS,
    ReplayValidationError,
    replay_labelset,
)

_REPO_ROOT = Path(__file__).resolve().parents[1]
_CONTRACTS = _REPO_ROOT / "src/contracts_data/watch"
_EXPECTED_BOUNDS = json.loads(
    (_REPO_ROOT / "tests/fixtures/watch/replay-bounds.json").read_text(encoding="utf-8")
)

ZONE = "fos_sur_mer"
PROVIDER = "gfs"
MODEL = "gfs_seamless"
WINDOW_START = date(2025, 1, 1)
WINDOW_END = date(2025, 3, 31)
QUIET = 6.0  # between p50 and p90 of the wind normals below: above_normal, never a notify
NOTIFY = 20.0  # above p90: extreme_high, a notify


def _days(start: date, end: date) -> list[date]:
    return [start + timedelta(days=i) for i in range((end - start).days + 1)]


def _normal(month: int, variable: str = "wind_speed", **overrides) -> dict:
    """One stored v2 normals row; the wet-day columns exist on precipitation only."""
    is_precipitation = variable == "precipitation"
    row = {
        "zone_id": ZONE,
        "month": month,
        "variable": variable,
        "p10": 2.0,
        "p50": 5.0,
        "p90": 12.0,
        "n": 930,
        "window_start": "1991-01-01",
        "window_end": "2020-12-31",
        "window_label": "1991-2020",
        "p10_wet": 2.0 if is_precipitation else None,
        "p50_wet": 6.0 if is_precipitation else None,
        "p90_wet": 20.0 if is_precipitation else None,
        "n_wet": 210 if is_precipitation else None,
        "wet_day_threshold_mm": 1.0 if is_precipitation else None,
        "schema_version": 2,
    }
    row.update(overrides)
    return row


def _wind_normals(**overrides) -> list[dict]:
    return [_normal(month, **overrides) for month in range(1, 13)]


def _artifact(
    horizon: int,
    values: dict[date, float],
    *,
    variable: str = "wind_speed",
    zone_id: str = ZONE,
    model: str = MODEL,
) -> dict:
    """One `forecast-archive-panel` artifact: ``{coverage, rows}``, as the panel writes it."""
    rows = [
        {
            "zone_id": zone_id,
            "origin_date": (day - timedelta(days=horizon)).isoformat(),
            "valid_date": day.isoformat(),
            "horizon_days": horizon,
            "variable": variable,
            "value": value,
            "provider": PROVIDER,
            "model": model,
        }
        for day, value in sorted(values.items())
    ]
    expected = (WINDOW_END - WINDOW_START).days + 1
    return {
        "coverage": {
            "zone_id": zone_id,
            "provider": PROVIDER,
            "model": model,
            "horizon_days": horizon,
            "start_date": WINDOW_START.isoformat(),
            "end_date": WINDOW_END.isoformat(),
            "expected_days": expected,
            "available_days": len(values),
            "status": "available" if len(values) == expected else "partial_archive",
        },
        "rows": rows,
    }


def _archives(
    *,
    variable: str = "wind_speed",
    base=QUIET,
    notify: dict[int, set[date]] | None = None,
    missing: dict[int, set[date]] | None = None,
    horizons=range(1, 8),
    start: date = WINDOW_START,
    end: date = WINDOW_END,
) -> list[dict]:
    """Seven artifacts over the window. ``base`` is a value or a ``day -> value`` callable."""
    archives = []
    for h in horizons:
        values = {}
        for day in _days(start, end):
            if day in (missing or {}).get(h, set()):
                continue
            value = base(day) if callable(base) else base
            if day in (notify or {}).get(h, set()):
                value = NOTIFY
            values[day] = value
        archives.append(_artifact(h, values, variable=variable))
    return archives


def _label(n: int, start: date, end: date, *, variable: str = "wind_speed", **overrides) -> dict:
    label = {
        "schema_version": "watch-label/v1",
        "event_id": f"synthetic-{n:03d}",
        "place_ref": f"zone:{ZONE}",
        "variable": variable,
        "start": start.isoformat(),
        "end": end.isoformat(),
        "timezone": "Europe/Paris",
        "match_tolerance_days": 1,
        "source": "synthetic test fixture",
        "level": "orange",
        "provenance_url": f"https://example.invalid/synthetic-{n:03d}",
        "licence": "synthetic-test-fixture",
        "spatial_match": "zone_within_department",
    }
    label.update(overrides)
    return label


def _labelset(
    labels: list[dict],
    *,
    variable: str = "wind_speed",
    exhaustive: bool = False,
    window_start: date = WINDOW_START,
    window_end: date = WINDOW_END,
) -> dict:
    return {
        "schema_version": "watch-labelset/v1",
        "variable": variable,
        "window_start": window_start.isoformat(),
        "window_end": window_end.isoformat(),
        "exhaustive": exhaustive,
        "coverage_statement": "synthetic fixture; says nothing about any real bulletin archive",
        "labels": labels,
    }


#: Five labels, far enough apart that no widened window touches another. Widened windows
#: cover 5 + 3 + 4 + 5 + 3 = 20 of the 90 window days, so 70 days lie outside every label.
FIVE = [
    (date(2025, 1, 10), date(2025, 1, 12)),
    (date(2025, 1, 25), date(2025, 1, 25)),
    (date(2025, 2, 10), date(2025, 2, 11)),
    (date(2025, 2, 25), date(2025, 2, 27)),
    (date(2025, 3, 15), date(2025, 3, 15)),
]


def _five_labels(variable: str = "wind_speed") -> list[dict]:
    return [_label(i + 1, s, e, variable=variable) for i, (s, e) in enumerate(FIVE)]


def _replay(archives=None, *, normals=None, labelset=None) -> dict:
    return replay_labelset(
        _archives() if archives is None else archives,
        normals=_wind_normals() if normals is None else normals,
        labelset=_labelset(_five_labels()) if labelset is None else labelset,
    )


def _hits(report: dict, event_id: str) -> dict[int, bool | None]:
    label = next(item for item in report["labels"] if item["event_id"] == event_id)
    return {int(h): hit for h, hit in label["hit"].items()}


def _lead(report: dict, event_id: str) -> int | None:
    return next(item for item in report["labels"] if item["event_id"] == event_id)["lead"]


# ---------------------------------------------------------------------------------------
# hit(h) is anchored on the forecast ISSUED h days before start.
# ---------------------------------------------------------------------------------------


def test_hit_is_anchored_on_issuance_not_on_the_artifact_horizon():
    """The counter-example that ruled out 'any notify day with origin before start'.

    Label 10-15 January; one notify on the 16th (end + 1) in the h=7 artifact. That run was
    issued on the 9th, one day before the event, so it is lead 1, not lead 7.
    """
    start, end = date(2025, 1, 10), date(2025, 1, 15)
    report = _replay(
        _archives(notify={7: {date(2025, 1, 16)}}),
        labelset=_labelset([_label(1, start, end)]),
    )
    hits = _hits(report, "synthetic-001")
    assert hits[7] is False
    assert hits[1] is True
    assert all(hits[h] is False for h in range(2, 8))
    assert _lead(report, "synthetic-001") == 1


def test_hit_and_lead_read_start_from_each_horizon():
    start, end = date(2025, 1, 10), date(2025, 1, 12)
    report = _replay(
        _archives(notify={3: {start}, 5: {start}}),
        labelset=_labelset([_label(1, start, end)]),
    )
    assert _hits(report, "synthetic-001") == {
        1: False,
        2: False,
        3: True,
        4: False,
        5: True,
        6: False,
        7: False,
    }
    assert _lead(report, "synthetic-001") == 5
    assert report["leads"] == [5]


def test_the_tolerance_is_one_day_either_side_and_no_more():
    start, end = date(2025, 2, 10), date(2025, 2, 12)
    labelset = _labelset([_label(1, start, end)])

    # start - 1 read one day after issuance: the run issued at start - 2, i.e. h = 2.
    before = _replay(_archives(notify={1: {date(2025, 2, 9)}}), labelset=labelset)
    assert [h for h, hit in _hits(before, "synthetic-001").items() if hit] == [2]

    # end + 1 read at horizon 7: issued on 6 Feb, four days before start.
    after = _replay(_archives(notify={7: {date(2025, 2, 13)}}), labelset=labelset)
    assert [h for h, hit in _hits(after, "synthetic-001").items() if hit] == [4]

    # start - 2 and end + 2 are outside the tolerance at every horizon.
    outside = {h: {date(2025, 2, 8), date(2025, 2, 14)} for h in range(1, 8)}
    far = _replay(_archives(notify=outside), labelset=labelset)
    assert all(hit is False for hit in _hits(far, "synthetic-001").values())
    assert _lead(far, "synthetic-001") is None


@pytest.mark.parametrize("tolerance", [0, 2, True])
def test_a_tolerance_other_than_one_is_refused(tolerance):
    """The docs fix 1; the schema allows 0..2. A wider window would inflate hits silently."""
    labelset = _labelset(
        [_label(1, date(2025, 1, 10), date(2025, 1, 12), match_tolerance_days=tolerance)]
    )
    with pytest.raises(ReplayValidationError, match="match_tolerance_days"):
        _replay(labelset=labelset)


# ---------------------------------------------------------------------------------------
# One evaluator: the replay goes through #203's build_watch_series, never a copy of it.
# ---------------------------------------------------------------------------------------


def test_the_replay_calls_the_same_evaluator_as_the_live_tool():
    start, end = date(2025, 1, 10), date(2025, 1, 12)
    archives = _archives()
    # Exactly p90 is above_normal, not a notify; just above it is extreme_high.
    for artifact, value in ((archives[0], 12.0), (archives[1], 12.0001)):
        for row in artifact["rows"]:
            if row["valid_date"] == start.isoformat():
                row["value"] = value
    assert watch_replay.build_watch_series is climate.build_watch_series
    with (
        mock.patch.object(
            climate, "classify_against_normals", wraps=climate.classify_against_normals
        ) as classifier,
        mock.patch.object(
            watch_replay, "build_watch_series", wraps=climate.build_watch_series
        ) as series,
    ):
        report = _replay(archives, labelset=_labelset([_label(1, start, end)]))
    assert series.call_count == 7
    assert classifier.call_count == 7 * 90
    hits = _hits(report, "synthetic-001")
    assert hits[1] is False and hits[2] is True


# ---------------------------------------------------------------------------------------
# Triggers outside the labels: unmatched unless the set is declared exhaustive.
# ---------------------------------------------------------------------------------------


@pytest.mark.parametrize("exhaustive", [False, True])
def test_triggers_outside_labels_are_unmatched_unless_exhaustive(exhaustive):
    notify = {2: {date(2025, 3, 1), date(2025, 3, 5)}, 3: {date(2025, 3, 1)}}
    missing = {2: {date(2025, 3, 20)}}
    report = _replay(
        _archives(notify=notify, missing=missing),
        labelset=_labelset(_five_labels(), exhaustive=exhaustive),
    )
    kept = "false_triggers_per_month" if exhaustive else "unmatched_triggers_per_month"
    other = "unmatched_triggers_per_month" if exhaustive else "false_triggers_per_month"
    per_h = report["horizons"]
    for block in per_h.values():
        assert kept in block and other not in block
    # The missing day leaves the denominator: a gap is never a quiet day.
    assert per_h["2"]["outside_label_notify_days"] == 2
    assert per_h["2"]["outside_label_evaluable_days"] == 69
    assert per_h["2"][kept] == round(2 / (69 / DAYS_PER_MONTH), 4)
    assert per_h["3"]["outside_label_evaluable_days"] == 70
    assert per_h["3"][kept] == round(1 / (70 / DAYS_PER_MONTH), 4)
    assert per_h["1"][kept] == 0.0
    assert DAYS_PER_MONTH == 30.4375


@pytest.mark.parametrize("exhaustive", ["true", 1])
def test_only_a_boolean_true_unlocks_false_triggers(exhaustive):
    """The schema requires a boolean; the replay reads anything else as not exhaustive."""
    labelset = {**_labelset(_five_labels()), "exhaustive": exhaustive}
    report = _replay(labelset=labelset)
    for block in report["horizons"].values():
        assert "unmatched_triggers_per_month" in block
        assert "false_triggers_per_month" not in block
    assert report["provenance"]["exhaustive"] is False


# ---------------------------------------------------------------------------------------
# Coverage is reported, never filled.
# ---------------------------------------------------------------------------------------


def test_coverage_is_per_horizon_and_a_gap_is_never_filled():
    gap = set(_days(date(2025, 1, 1), date(2025, 1, 9)))
    report = _replay(_archives(missing={4: gap}))
    per_h = report["coverage"]["per_horizon"]
    assert report["coverage"]["window_days"] == 90
    assert per_h["4"]["archived_days"] == 81
    assert per_h["4"]["share"] == round(81 / 90, 4)
    assert per_h["4"]["archive_coverage"]["status"] == "partial_archive"
    assert all(per_h[str(h)]["share"] == 1.0 for h in (1, 2, 3, 5, 6, 7))
    assert report["coverage"]["horizons_served"] == [1, 2, 3, 4, 5, 6, 7]


def test_coverage_records_the_first_and_last_archived_day_not_the_requested_span():
    """Bound 2: the panel's start_date/end_date are what was requested, not what was archived."""
    missing = {
        4: set(_days(date(2025, 1, 1), date(2025, 1, 9))),
        6: set(_days(date(2025, 3, 25), date(2025, 3, 31))),
        7: set(_days(WINDOW_START, WINDOW_END)),
    }
    per_h = _replay(_archives(missing=missing))["coverage"]["per_horizon"]
    assert per_h["4"]["archived_first_day"] == "2025-01-10"
    assert per_h["4"]["archive_coverage"]["start_date"] == "2025-01-01"
    assert per_h["6"]["archived_last_day"] == "2025-03-24"
    assert per_h["6"]["archive_coverage"]["end_date"] == "2025-03-31"
    assert (per_h["1"]["archived_first_day"], per_h["1"]["archived_last_day"]) == (
        "2025-01-01",
        "2025-03-31",
    )
    assert per_h["7"]["archived_days"] == 0
    assert per_h["7"]["archived_first_day"] is None and per_h["7"]["archived_last_day"] is None


def test_a_label_with_no_evaluable_day_is_null_never_false():
    first_start, first_end = FIVE[0]
    widened = set(_days(first_start - timedelta(days=1), first_end + timedelta(days=1)))
    report = _replay(_archives(missing=dict.fromkeys(range(1, 8), widened)))
    assert all(hit is None for hit in _hits(report, "synthetic-001").values())
    assert _lead(report, "synthetic-001") is None
    assert all(block["evaluated"] == 4 for block in report["horizons"].values())


# ---------------------------------------------------------------------------------------
# Precipitation: v1 normals cannot classify; v2 bands against the wet-day triplet.
# ---------------------------------------------------------------------------------------


def test_precipitation_on_v1_normals_is_reference_unavailable_not_zero():
    v1 = [
        {
            "zone_id": ZONE,
            "month": month,
            "variable": "precipitation",
            "p10": 0.0,
            "p50": 0.0,
            "p90": 5.0,
            "n": 930,
            "window_label": "1991-2020",
        }
        for month in range(1, 13)
    ]
    report = replay_labelset(
        _archives(variable="precipitation", base=30.0),
        normals=v1,
        labelset=_labelset(_five_labels("precipitation"), variable="precipitation"),
    )
    # Five labels, but no day was evaluable at any horizon: not a measurement of zero.
    assert report["status"] == "not_evaluable"
    assert report["n_labels"] == 5
    for label in report["labels"]:
        assert all(hit is None for hit in label["hit"].values())
    for h in range(1, 8):
        cov = report["coverage"]["per_horizon"][str(h)]
        assert cov["unclassified"] == {"reference_unavailable": 90}
        assert cov["evaluable_days"] == 0
        block = report["horizons"][str(h)]
        assert block["hits"] == 0 and block["evaluated"] == 0 and block["hit_rate"] is None
        assert block["outside_label_notify_days"] == 0
        assert block["unmatched_triggers_per_month"] is None
    assert report["provenance"]["normals_schema_version"] == [1]
    assert report["provenance"]["wet_day_threshold_mm"] == []


def test_precipitation_on_v2_normals_bands_wet_days_and_keeps_dry_days():
    normals = [_normal(m, "precipitation") for m in (1, 3)] + [
        _normal(2, "precipitation", p10_wet=None, p50_wet=None, p90_wet=None, n_wet=0)
    ]
    normals += [_normal(m, "precipitation") for m in range(4, 13)]
    january, february = date(2025, 1, 10), date(2025, 2, 10)

    def base(day: date) -> float:
        return 5.0 if day.month == 2 else 0.5  # dry in Jan/Mar, wet but unbandable in Feb

    archives = _archives(variable="precipitation", base=base)
    for row in archives[0]["rows"]:
        if row["valid_date"] == january.isoformat():
            row["value"] = 25.0  # > p90_wet: a notify
    labels = [
        _label(1, january, date(2025, 1, 12), variable="precipitation"),
        _label(2, february, date(2025, 2, 11), variable="precipitation"),
    ]
    report = replay_labelset(
        archives,
        normals=normals,
        labelset=_labelset(labels, variable="precipitation"),
    )
    # Dry days are evaluable (a real "no"), never a notify.
    assert _hits(report, "synthetic-001") == {1: True, **dict.fromkeys(range(2, 8), False)}
    assert all(hit is None for hit in _hits(report, "synthetic-002").values())
    cov = report["coverage"]["per_horizon"]["3"]
    assert cov["unclassified"] == {"wet_reference_unavailable": 28}
    assert cov["evaluable_days"] == 62
    assert report["provenance"]["normals_schema_version"] == [2]
    assert report["provenance"]["wet_day_threshold_mm"] == [1.0]


def test_a_label_crossing_a_month_reads_each_days_own_month():
    """January's reference for 30-31 Jan, February's for 1-3 Feb (§G2)."""
    normals = [_normal(1, p90=20.0), _normal(2, p90=12.0)]
    normals += [_normal(m) for m in range(3, 13)]
    start, end = date(2025, 1, 31), date(2025, 2, 2)
    report = replay_labelset(
        _archives(base=15.0),
        normals=normals,
        labelset=_labelset([_label(1, start, end)]),
    )
    assert _hits(report, "synthetic-001") == {**dict.fromkeys(range(1, 7), True), 7: False}
    assert _lead(report, "synthetic-001") == 6


# ---------------------------------------------------------------------------------------
# Real panel artifacts: every variable in one file, and a span wider than the labelset.
# ---------------------------------------------------------------------------------------


def test_a_mixed_variable_artifact_replays_only_the_labelset_variable():
    """`forecast-archive-panel` writes every NORMALIZED_VARS variable into each artifact.

    A wind notify on the label start is not a precipitation hit, and wind rows are not
    archived precipitation days; the same artifacts replayed for wind do see the notify.
    """
    start, end = date(2025, 1, 10), date(2025, 1, 12)
    wind = _archives(notify={1: {start}})
    rain = _archives(variable="precipitation", base=0.5)  # dry every day
    mixed = [
        {"coverage": w["coverage"], "rows": r["rows"] + w["rows"]}
        for w, r in zip(wind, rain, strict=True)
    ]
    normals = [_normal(m, "precipitation") for m in range(1, 13)] + _wind_normals()
    rain_labelset = _labelset(
        [_label(1, start, end, variable="precipitation")], variable="precipitation"
    )

    rain_report = replay_labelset(mixed, normals=normals, labelset=rain_labelset)
    assert all(hit is False for hit in _hits(rain_report, "synthetic-001").values())
    for cov in rain_report["coverage"]["per_horizon"].values():
        assert cov["archived_days"] == 90 and cov["share"] == 1.0

    wind_report = replay_labelset(
        mixed, normals=normals, labelset=_labelset([_label(1, start, end)])
    )
    assert _hits(wind_report, "synthetic-001")[1] is True
    assert wind_report["coverage"]["per_horizon"]["1"]["archived_days"] == 90


def test_an_archive_wider_than_the_labelset_window_is_read_inside_it_only():
    """The labelset speaks for its own window; archived days outside it are not replayed.

    The archive spans January to March, the labelset February only. January notifies,
    including 31 January (start - 1 of the first label), neither hit nor count as triggers.
    """
    january = {date(2025, 1, 20), date(2025, 1, 21), date(2025, 1, 31)}
    starts = [date(2025, 2, day) for day in (1, 8, 14, 20, 26)]
    report = _replay(
        _archives(notify=dict.fromkeys(range(1, 8), january)),
        labelset=_labelset(
            [_label(i + 1, day, day) for i, day in enumerate(starts)],
            window_start=date(2025, 2, 1),
            window_end=date(2025, 2, 28),
        ),
    )
    assert report["coverage"]["window_days"] == 28
    for h in range(1, 8):
        cov = report["coverage"]["per_horizon"][str(h)]
        assert cov["archived_days"] == 28 and cov["share"] == 1.0
        assert (cov["archived_first_day"], cov["archived_last_day"]) == ("2025-02-01", "2025-02-28")
        block = report["horizons"][str(h)]
        assert block["hits"] == 0 and block["evaluated"] == 5
        assert block["outside_label_notify_days"] == 0
        # Widened windows inside February: 2 days for the 1st (31 Jan is outside) + 4 × 3.
        assert block["outside_label_evaluable_days"] == 28 - 14
        assert block["unmatched_triggers_per_month"] == 0.0


# ---------------------------------------------------------------------------------------
# Fewer than five labels: not measured, never a rate.
# ---------------------------------------------------------------------------------------


def test_four_labels_are_not_measured_and_publish_no_rate():
    notify = {1: {FIVE[0][0], date(2025, 3, 1)}}
    report = _replay(_archives(notify=notify), labelset=_labelset(_five_labels()[:4]))
    assert report["n_labels"] == 4
    assert report["status"] == "not_measured"
    for block in report["horizons"].values():
        assert block["hit_rate"] is None
        assert block["unmatched_triggers_per_month"] is None
        assert block["evaluated"] == 4
    # The raw counts stay, so a reader can see what was and was not measured.
    assert report["horizons"]["1"]["hits"] == 1
    assert report["horizons"]["1"]["outside_label_notify_days"] == 1


def test_five_labels_are_measured_but_a_thin_horizon_is_not():
    notify = {1: {FIVE[0][0]}}
    # Remove the two days hit(7) reads for the last two labels: start - 1 at h=6, start at h=7.
    missing = {
        6: {FIVE[3][0] - timedelta(days=1), FIVE[4][0] - timedelta(days=1)},
        7: {FIVE[3][0], FIVE[4][0]},
    }
    report = _replay(_archives(notify=notify, missing=missing))
    assert report["status"] == "measured"
    per_h = report["horizons"]
    assert per_h["1"]["hits"] == 1 and per_h["1"]["evaluated"] == 5
    assert per_h["1"]["hit_rate"] == 0.2
    assert per_h["6"]["evaluated"] == 5 and per_h["6"]["hit_rate"] == 0.0
    assert per_h["7"]["evaluated"] == 3 and per_h["7"]["hit_rate"] is None
    assert per_h["7"]["unmatched_triggers_per_month"] == 0.0


def test_five_labels_none_evaluable_at_any_horizon_are_not_evaluable_and_publish_no_rate():
    """Every widened label day missing at every horizon: rates stay null, raw counts stay."""
    widened = {d for s, e in FIVE for d in _days(s - timedelta(days=1), e + timedelta(days=1))}
    notify = {1: {date(2025, 3, 1)}}
    report = _replay(_archives(notify=notify, missing=dict.fromkeys(range(1, 8), widened)))
    assert report["n_labels"] == 5
    assert report["status"] == "not_evaluable"
    for block in report["horizons"].values():
        assert block["evaluated"] == 0 and block["hit_rate"] is None
        assert block["outside_label_evaluable_days"] == 70
        assert block["unmatched_triggers_per_month"] is None
    assert report["horizons"]["1"]["outside_label_notify_days"] == 1

    # One horizon with no evaluable label is not enough: h=7 reads only artifacts 6 and 7.
    report = _replay(_archives(notify=notify, missing=dict.fromkeys((6, 7), widened)))
    assert report["status"] == "measured"
    assert report["horizons"]["7"]["evaluated"] == 0
    assert report["horizons"]["6"]["evaluated"] == 5
    assert report["horizons"]["7"]["unmatched_triggers_per_month"] == 0.0


def test_an_empty_labelset_is_valid_and_not_measured():
    """`labels` has no minItems: an exhaustive window with no bulletin is a legitimate set."""
    report = _replay(labelset=_labelset([], exhaustive=True))
    assert report["n_labels"] == 0
    assert report["status"] == "not_measured"
    assert report["labels"] == [] and report["leads"] == []
    assert report["horizons"]["1"]["false_triggers_per_month"] is None
    assert report["horizons"]["1"]["outside_label_evaluable_days"] == 90


# ---------------------------------------------------------------------------------------
# Fail closed on every input the schema does not already guard.
# ---------------------------------------------------------------------------------------


def _drop_horizons(*keep):
    return [a for a in _archives() if a["coverage"]["horizon_days"] in keep]


def _with(archives, index, mutate):
    mutate(archives[index])
    return archives


def _duplicate_row(artifact):
    artifact["rows"].append(dict(artifact["rows"][0]))


def _bad_origin(artifact):
    artifact["rows"][0]["origin_date"] = artifact["rows"][0]["valid_date"]


def _row_horizon(artifact):
    artifact["rows"][0]["horizon_days"] = 2
    artifact["rows"][0]["origin_date"] = (
        date.fromisoformat(artifact["rows"][0]["valid_date"]) - timedelta(days=2)
    ).isoformat()


def _thin_rows(artifact):
    """Coverage still claims the full window; only a few rows are actually archived."""
    artifact["rows"] = artifact["rows"][:3]


def _row_outside_window(artifact):
    """The first row moved a day before the coverage's own claimed start_date.

    The artifact still has as many distinct days as `available_days` claims, so only the
    window check can refuse it.
    """
    horizon = artifact["coverage"]["horizon_days"]
    day = WINDOW_START - timedelta(days=1)
    artifact["rows"][0].update(
        origin_date=(day - timedelta(days=horizon)).isoformat(), valid_date=day.isoformat()
    )


def _inflate_available_days(artifact):
    """`available_days` matches the archived rows but now exceeds `expected_days`."""
    artifact["coverage"]["expected_days"] = artifact["coverage"]["available_days"] - 1


def _status_disagrees_with_the_counts(artifact):
    """`available_days == expected_days`, yet `status` still says `partial_archive`."""
    artifact["coverage"]["status"] = "partial_archive"


@pytest.mark.parametrize(
    ("archives", "match"),
    [
        (lambda: _drop_horizons(1, 3, 5, 7), "horizons"),
        (lambda: _archives() + [_archives()[0]], "horizon"),
        (lambda: _with(_archives(), 2, lambda a: a.pop("coverage")), "coverage"),
        (
            lambda: _with(
                _archives(), 2, lambda a: a["coverage"].update(model="ecmwf_aifs025_single")
            ),
            "one zone",
        ),
        (lambda: _with(_archives(), 2, lambda a: a["coverage"].update(status="current")), "status"),
        (lambda: _with(_archives(), 0, _duplicate_row), "duplicate"),
        (lambda: _with(_archives(), 0, _bad_origin), "horizon_days"),
        (lambda: _with(_archives(), 0, _row_horizon), "coverage"),
        (lambda: _with(_archives(), 0, lambda a: a["rows"][0].update(model="other")), "coverage"),
        (
            lambda: _with(_archives(), 0, lambda a: a["rows"][0].update(zone_id="montpellier")),
            "coverage",
        ),
        (lambda: _with(_archives(), 0, lambda a: a["rows"][0].update(provider="aifs")), "coverage"),
        (lambda: _with(_archives(), 0, _thin_rows), "horizon 1"),
        (lambda: _with(_archives(), 0, _row_outside_window), "outside its own coverage"),
        (lambda: _with(_archives(), 0, _inflate_available_days), "exceeds"),
        (lambda: _with(_archives(), 0, _status_disagrees_with_the_counts), "coverage.status"),
        # float() of an int past ~1.8e308 raises OverflowError, not ValueError (#221).
        (
            lambda: _with(_archives(), 0, lambda a: a["rows"][0].update(value=10**400)),
            "horizon 1: prediction value must be numeric",
        ),
    ],
    ids=[
        "sampled-horizons",
        "duplicate-horizon",
        "no-coverage",
        "mixed-models",
        "not-previous-runs",
        "duplicate-day",
        "origin-not-before-valid",
        "row-horizon-mismatch",
        "row-model-mismatch",
        "row-zone-mismatch",
        "row-provider-mismatch",
        "coverage-claim-few-rows",
        "coverage-claim-row-outside-window",
        "coverage-claim-available-exceeds-expected",
        "coverage-claim-status-disagrees-with-counts",
        "row-value-beyond-float-range",
    ],
)
def test_archives_fail_closed(archives, match):
    with pytest.raises(ReplayValidationError, match=match):
        _replay(archives())


def test_an_archive_for_another_zone_is_refused():
    archives = [
        _artifact(h, dict.fromkeys(_days(WINDOW_START, WINDOW_END), QUIET), zone_id="montpellier")
        for h in range(1, 8)
    ]
    with pytest.raises(ReplayValidationError, match="place_ref"):
        replay_labelset(
            archives,
            normals=_wind_normals(zone_id="montpellier"),
            labelset=_labelset(_five_labels()),
        )


@pytest.mark.parametrize(
    ("normals", "match"),
    [
        (lambda: _wind_normals()[:11] + [_normal(12, zone_id="montpellier")], "zone"),
        (lambda: _wind_normals() + [_normal(3, p90=30.0)], "duplicate"),
        (lambda: _wind_normals()[:11] + [_normal(12, window_label="1981-2010")], "window_label"),
        (lambda: [], "window_label"),
    ],
    ids=["other-zone", "duplicate-month", "mixed-windows", "empty"],
)
def test_normals_fail_closed(normals, match):
    with pytest.raises(ReplayValidationError, match=match):
        _replay(normals=normals())


@pytest.mark.parametrize(
    ("variable", "field", "out_of_range", "status"),
    [
        ("wind_speed", "n", math.inf, "measured"),
        ("wind_speed", "schema_version", math.inf, "measured"),
        ("wind_speed", "p90", 10**400, "not_evaluable"),
        ("precipitation", "n_wet", math.inf, "measured"),
        ("precipitation", "wet_day_threshold_mm", 10**400, "not_evaluable"),
    ],
    ids=[
        "n-inf",
        "schema-version-inf",
        "p90-beyond-float",
        "n-wet-inf",
        "wet-threshold-beyond-float",
    ],
)
def test_an_out_of_range_stored_number_reads_like_nan_not_an_overflow(
    variable, field, out_of_range, status
):
    """``int(inf)`` and ``float(10**400)`` raise OverflowError, not ValueError (#221).

    A stored NaN in the same field already reads as absent: the report is the one a NaN
    gives, on all twelve rows, never an uncaught exception.
    """

    def replay(value) -> dict:
        return replay_labelset(
            _archives(variable=variable, notify={1: {FIVE[0][0]}}),
            normals=[_normal(month, variable, **{field: value}) for month in range(1, 13)],
            labelset=_labelset(_five_labels(variable), variable=variable),
        )

    report = replay(out_of_range)
    assert report == replay(math.nan)
    assert report["status"] == status


def test_an_extra_normals_row_with_an_infinite_month_is_dropped_like_a_nan_month():
    """``_validate_normals`` keys it ``(variable, None)``; ``build_watch_series`` skips it.

    No day's month is ``inf``, so the row is a reference for nothing and the report is the
    one without it (#221). Before, ``int(inf)`` raised inside the normals check.
    """
    report = _replay(normals=_wind_normals() + [_normal(math.inf)])
    assert report == _replay(normals=_wind_normals() + [_normal(math.nan)])
    assert report == _replay()


def _labels_with(index, **overrides):
    labels = _five_labels()
    labels[index] = {**labels[index], **overrides}
    return labels


@pytest.mark.parametrize(
    ("labelset", "match"),
    [
        (
            lambda: {**_labelset(_five_labels()), "schema_version": "watch-labelset/v2"},
            "schema_version",
        ),
        (lambda: _labelset([], variable="temperature_2m"), "variable"),
        # No label, so only the window order itself can refuse it.
        (lambda: _labelset([], window_start=date(2025, 4, 1)), "after window_end"),
        (lambda: _labelset(_labels_with(0, variable="precipitation")), "variable"),
        (lambda: _labelset(_labels_with(0, start="2024-12-31")), "window"),
        (lambda: _labelset(_labels_with(4, end="2025-04-01")), "outside the labelset window"),
        (lambda: _labelset(_labels_with(0, start="2025-01-13")), "start"),
        (lambda: _labelset(_labels_with(1, event_id="synthetic-001")), "event_id"),
        (lambda: _labelset(_labels_with(0, place_ref="zone:montpellier")), "place_ref"),
        (lambda: _labelset(_labels_with(0, schema_version="watch-label/v2")), "schema_version"),
        (
            lambda: {
                k: v
                for k, v in _labelset(_five_labels(), exhaustive=True).items()
                if k != "coverage_statement"
            },
            "coverage_statement",
        ),
        (
            lambda: {**_labelset(_five_labels(), exhaustive=True), "coverage_statement": ""},
            "coverage_statement",
        ),
        (
            lambda: {**_labelset(_five_labels(), exhaustive=True), "coverage_statement": 5},
            "coverage_statement",
        ),
    ],
    ids=[
        "set-version",
        "variable-outside-watch",
        "window-reversed",
        "label-variable",
        "label-outside-window",
        "label-past-window-end",
        "label-reversed",
        "duplicate-event",
        "other-zone",
        "label-version",
        "exhaustive-no-coverage-statement",
        "exhaustive-empty-coverage-statement",
        "exhaustive-non-string-coverage-statement",
    ],
)
def test_labelset_fails_closed(labelset, match):
    with pytest.raises(ReplayValidationError, match=match):
        _replay(labelset=labelset())


def _inputs(**overrides) -> dict:
    """Explicit inputs: `_replay` would swap a None for the default fixture."""
    return {
        "archives": _archives(),
        "normals": _wind_normals(),
        "labelset": _labelset(_five_labels()),
        **overrides,
    }


def _one_archived_day(**coverage) -> list[dict]:
    """Horizon 1 archives a single day, then `coverage` is applied to its claims.

    With one day, `True` equals the count it replaces, so only the type guard can refuse it.
    """
    archives = _archives()
    archives[0] = _artifact(1, {WINDOW_START: QUIET})
    archives[0]["coverage"].update(coverage)
    return archives


def _blank_model() -> list[dict]:
    """Coverage and rows agree on an empty model, so only the non-empty guard can refuse."""
    quiet = dict.fromkeys(_days(WINDOW_START, WINDOW_END), QUIET)
    return [_artifact(h, quiet, model="") for h in range(1, 8)]


@pytest.mark.parametrize(
    ("inputs", "match"),
    [
        (lambda: _inputs(archives=None), "archives must be a list"),
        (lambda: _inputs(archives=_with(_archives(), 3, lambda a: a.pop("rows"))), "rows list"),
        (lambda: _inputs(normals=None), "normals must be a list"),
        (lambda: _inputs(labelset={**_labelset(_five_labels()), "labels": None}), "labels must"),
        (
            lambda: _inputs(labelset={**_labelset(_five_labels()), "window_start": None}),
            "window_start",
        ),
        (
            lambda: _inputs(
                archives=_with(_archives(), 0, lambda a: a["coverage"].update(horizon_days=True))
            ),
            "integer",
        ),
        (
            lambda: _inputs(
                archives=_with(_archives(), 0, lambda a: a["coverage"].update(horizon_days=1.0))
            ),
            "integer",
        ),
        (lambda: _inputs(archives=_blank_model()), "non-empty"),
        (lambda: _inputs(normals=_wind_normals(window_label=None)), "window_label"),
        (lambda: _inputs(labelset=_labelset(_labels_with(0, event_id=""))), "event_id"),
        (
            lambda: _inputs(
                archives=_with(_archives(), 0, lambda a: a["coverage"].update(zone_id=["fos"]))
            ),
            "zone_id",
        ),
        (
            lambda: _inputs(
                archives=_with(_archives(), 0, lambda a: a["coverage"].update(provider=["gfs"]))
            ),
            "provider",
        ),
        (
            lambda: _inputs(
                archives=_with(_archives(), 0, lambda a: a["coverage"].update(model=["gfs"]))
            ),
            "model",
        ),
        (lambda: _inputs(normals=_wind_normals(window_label=["1991-2020"])), "window_label"),
        (
            lambda: _inputs(
                archives=_with(_archives(), 0, lambda a: a["coverage"].update(expected_days=90.0))
            ),
            "expected_days",
        ),
        (
            lambda: _inputs(
                archives=_with(_archives(), 0, lambda a: a["coverage"].update(available_days=90.0))
            ),
            "available_days must be an integer",
        ),
        (
            lambda: _inputs(archives=_one_archived_day(available_days=True)),
            "available_days must be an integer",
        ),
        (
            lambda: _inputs(archives=_one_archived_day(expected_days=True, status="available")),
            "expected_days must be an integer",
        ),
        (lambda: _inputs(normals=_wind_normals(variable=["wind_speed"])), "variable"),
        (lambda: _inputs(normals=_wind_normals(variable="")), "variable"),
        (lambda: _inputs(normals=_wind_normals(window_label="")), "window_label"),
    ],
    ids=[
        "archives-none",
        "artifact-without-rows",
        "normals-none",
        "labels-none",
        "window-start-none",
        "horizon-bool",
        "horizon-float",
        "model-empty",
        "window-label-none",
        "event-id-empty",
        "coverage-zone-id-list",
        "coverage-provider-list",
        "coverage-model-list",
        "normals-window-label-list",
        "coverage-expected-days-float",
        "coverage-available-days-float",
        "coverage-available-days-bool",
        "coverage-expected-days-bool",
        "normals-variable-list",
        "normals-variable-empty",
        "normals-window-label-empty",
    ],
)
def test_wrong_types_are_refused_not_crashed_on_or_accepted(inputs, match):
    """Each case is a ReplayValidationError, never a TypeError, a KeyError or a report."""
    kwargs = inputs()
    with pytest.raises(ReplayValidationError, match=match):
        replay_labelset(kwargs.pop("archives"), **kwargs)


# ---------------------------------------------------------------------------------------
# Doctrine: the bounds, the caveats, confidence null, no skill vocabulary, no I/O.
# ---------------------------------------------------------------------------------------


def test_the_five_bounds_are_verbatim_from_the_design_note():
    """Every replay report preserves all five independent expected bounds verbatim."""
    assert len(_EXPECTED_BOUNDS) == 5
    assert list(REPLAY_BOUNDS) == _EXPECTED_BOUNDS
    report = _replay()
    assert report["bounds"] == _EXPECTED_BOUNDS
    assert report["caveats"] == list(NORMALS_CAVEATS)


def test_the_replay_report_does_not_claim_a_seven_day_forecast_was_requested():
    """The report copies the tool caveats; a replay reads archived forecasts per lead day."""
    caveats = _replay()["caveats"]
    horizon = next(c for c in caveats if c.startswith("Horizon 7 days requested"))
    assert "In live watch use, a forecast is requested for 7 days" in horizon
    assert "A replay instead reads archived forecasts" in horizon
    # The report itself publishes `coverage.horizons_served`, so it does know the served horizon.
    assert "does not know the served horizon" not in horizon
    assert _replay()["coverage"]["horizons_served"]
    timestamps = next(c for c in caveats if c.startswith("Two timestamps"))
    assert timestamps.split(": ", 1)[1].startswith("In live watch use")
    assert "replay" in timestamps


def test_the_report_names_the_rule_it_replayed_and_its_provenance():
    report = _replay()
    assert report["schema_version"] == "watch-replay/v1"
    assert report["rule"]["parity"] == "Deposium WatchRule parity not verified"
    assert "extreme_high" in report["rule"]["predicate"]
    provenance = report["provenance"]
    assert provenance["zone_id"] == ZONE
    assert (provenance["provider"], provenance["model"]) == (PROVIDER, MODEL)
    assert provenance["horizons"] == [1, 2, 3, 4, 5, 6, 7]
    assert provenance["normals_window_label"] == "1991-2020"
    assert provenance["match_tolerance_days"] == 1
    assert provenance["exhaustive"] is False
    assert provenance["coverage_statement"] == _labelset([])["coverage_statement"]
    assert provenance["license"] == "CC-BY 4.0 (Open-Meteo)"
    # Wind normals carry no threshold, even when they are v2.
    assert provenance["wet_day_threshold_mm"] == []
    assert provenance["normals_schema_version"] == [2]


def _keys(value):
    if isinstance(value, dict):
        for key, item in value.items():
            yield key
            yield from _keys(item)
    elif isinstance(value, list):
        for item in value:
            yield from _keys(item)


def test_confidence_is_null_and_no_skill_vocabulary_is_emitted():
    report = _replay()
    assert "confidence" in report and report["confidence"] is None
    forbidden = ("probability", "risk", "severity", "skill", "score", "recall", "calibrat")
    for key in _keys(report):
        assert not any(word in key for word in forbidden), key


def test_the_report_is_json_native():
    report = _replay()
    assert json.loads(json.dumps(report)) == report


def test_the_replay_does_no_io(monkeypatch):
    def _refuse(*_args, **_kwargs):
        raise AssertionError("the replay core must not do I/O")

    import duckdb
    import httpx

    monkeypatch.setattr(httpx.Client, "send", _refuse)
    monkeypatch.setattr(httpx.AsyncClient, "send", _refuse)
    monkeypatch.setattr(duckdb, "connect", _refuse)
    report = _replay(_archives(notify={1: {FIVE[0][0]}}))
    assert report["horizons"]["1"]["hits"] == 1


def test_the_synthetic_labelset_is_valid_against_the_committed_contracts():
    schemas = {
        path.name: json.loads(path.read_text(encoding="utf-8"))
        for path in _CONTRACTS.glob("watch-label*.schema.json")
    }
    registry = Registry().with_resources(
        (schema["$id"], Resource.from_contents(schema)) for schema in schemas.values()
    )
    validator = Draft202012Validator(
        schemas["watch-labelset.v1.schema.json"],
        registry=registry,
        format_checker=FormatChecker(),
    )
    for document in (_labelset(_five_labels()), _labelset([], exhaustive=True)):
        assert list(validator.iter_errors(document)) == []
    # The registry really resolves the label schema: a bad label is caught.
    bad = _labelset(_labels_with(0, level="jaune"))
    assert list(validator.iter_errors(bad))


# ---------------------------------------------------------------------------------------
# #193 — the Previous Runs normalizer keeps the complete variables of a day with one gap, so
# an artifact can hold rows for a day that is not complete. The replay must accept exactly
# what the normalizer writes.
# ---------------------------------------------------------------------------------------


def _normalized_archives(gap_day: date) -> list[dict]:
    """Seven artifacts from the real normalizer; temperature is blank on ``gap_day``."""
    from dataclasses import asdict

    from src.forecast_archive import (
        PREVIOUS_RUNS_HOURLY_UNITS,
        _previous_field,
        normalize_previous_runs,
    )

    variables = ["wind_speed", "temperature_2m"]
    days = _days(WINDOW_START, WINDOW_END)
    times = [f"{day.isoformat()}T{hour:02d}:00" for day in days for hour in range(24)]
    archives = []
    for horizon in range(1, 8):
        hourly = {"time": times}
        units = {"time": "iso8601"}
        for variable in variables:
            field = _previous_field(variable, horizon)
            hourly[field] = [
                None
                if variable == "temperature_2m" and stamp.startswith(gap_day.isoformat())
                else QUIET
                for stamp in times
            ]
            units[field] = PREVIOUS_RUNS_HOURLY_UNITS[variable]
        rows, coverage = normalize_previous_runs(
            {"utc_offset_seconds": 0, "timezone": "GMT", "hourly_units": units, "hourly": hourly},
            zone_id=ZONE,
            provider=PROVIDER,
            model=MODEL,
            horizon_days=horizon,
            variables=variables,
            expected_start_date=WINDOW_START.isoformat(),
            expected_end_date=WINDOW_END.isoformat(),
        )
        archives.append(
            json.loads(
                json.dumps(
                    {
                        "coverage": asdict(coverage),
                        "rows": [
                            {
                                **asdict(row),
                                "origin_date": row.origin_date.isoformat(),
                                "valid_date": row.valid_date.isoformat(),
                            }
                            for row in rows
                        ],
                    }
                )
            )
        )
    return archives


def test_an_artifact_with_one_variable_gap_is_replayed_not_refused():
    gap_day = date(2025, 2, 10)
    archives = _normalized_archives(gap_day)
    coverage = archives[0]["coverage"]
    assert coverage["status"] == "partial_archive"
    assert coverage["available_days"] == 89  # the gap day is not complete
    assert len({row["valid_date"] for row in archives[0]["rows"]}) == 90  # but has wind rows

    report = _replay(
        archives,
        normals=_wind_normals() + [_normal(month, "temperature_2m") for month in range(1, 13)],
    )

    assert report["coverage"]["per_horizon"]["1"]["archive_coverage"]["status"] == (
        "partial_archive"
    )


def test_pair_coverage_must_still_match_the_rows():
    archives = _normalized_archives(date(2025, 2, 10))
    archives[0]["coverage"]["available_pairs"] += 1
    with pytest.raises(ReplayValidationError, match="available_pairs"):
        _replay(
            archives,
            normals=_wind_normals() + [_normal(month, "temperature_2m") for month in range(1, 13)],
        )


@pytest.mark.parametrize(
    ("change", "match"),
    [
        ({"expected_pairs": 179}, "whole number of variables per day"),
        # The review's exact case: 179 rows over 90 days and two variables, declared as
        # 179 expected pairs, one available day and status "available".
        (
            {"expected_pairs": 179, "available_days": 1, "status": "available"},
            "whole number of variables per day",
        ),
        ({"expected_days": 0}, "expected_days"),
    ],
)
def test_pair_coverage_must_divide_into_whole_days(change, match):
    archives = _normalized_archives(date(2025, 2, 10))
    archives[0]["coverage"].update(change)
    with pytest.raises(ReplayValidationError, match=match):
        _replay(
            archives,
            normals=_wind_normals() + [_normal(month, "temperature_2m") for month in range(1, 13)],
        )
