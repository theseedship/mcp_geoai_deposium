"""Derive Koppen-Geiger classes from the shipped WMO normals and diff them against the
hand-written constants in ``scripts/generate_zones.py`` (issue #186).

The 15 ``koppen`` values in ``zones/zones.geojson`` are literals typed into ``PANEL`` /
``PILOTS``, and the generator says so itself: *"best-known representative values, pending
geomatics validation"*. ``tests/test_zones.py`` guards their FORMAT — a valid Koppen code,
at least 8 distinct classes — and nothing guards whether a code is TRUE of its zone.
Relabelling ``riyadh`` from ``BWh`` to ``Cfb`` keeps that suite green.

This is the cheap half of the answer, and it needs no external source: a Koppen class is
DEFINED from monthly temperature and precipitation normals, and the repo already holds
those at 100% completeness for all 15 zones.

    python scripts/koppen_crosscheck.py --bundle <dir> --zones zones/zones.geojson

``--bundle`` is any directory holding ``climate/normals/*.parquet`` — the published
``climate-baseline-full-panel-2026-06-26`` package unzipped, or a local materialization.
Offline: no S3, no provider, no network.

WHAT THIS DOES NOT SETTLE. The normals are sampled at a single BBOX MIDPOINT per zone
(``src/tools/climate.py``), while the zones are polygons ranging from a 1-degree panel box
to a 589 km2 watershed. Deriving a class from that point validates THAT POINT. Closing the
B1 residual needs a Koppen raster overlay or spatial sampling across each polygon, which is
issue #194 and is neither implemented nor implied here.

Measured on 2026-09-12 from a local dev store and again on 2026-09-23 from the published
package, with identical results: 12 of 15 agree, and all three divergences (manaus, nairobi,
niamey) turn on a PRECIPITATION threshold. No temperature-driven criterion diverges on any
zone, which is why #194 has three zones to settle rather than fifteen.
"""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

import duckdb

#: 1991-2020 inclusive. Used to turn a mean DAILY value plus its contributing day count
#: back into a monthly TOTAL: ``n / YEARS`` recovers the days in that calendar month
#: (February comes out 28.27, i.e. 8 leap years over the window).
BASELINE_YEARS = 30

#: The two half-years Peel chooses between. Summer is whichever of them is WARMER — that is
#: the rule, and latitude is not an input to it. This module used to pick by hemisphere
#: instead, which agrees with the warmer half on all 15 panel zones (nairobi, at 1.3 S, has
#: ONDJFM at 19.5 C against AMJJAS at 18.0 C) but is not what the criteria say and would
#: diverge on a near-equatorial or bimodal climate — exactly where the s/w letter is already
#: fragile (Greptile P2, PR #198).
_AMJJAS = (4, 5, 6, 7, 8, 9)
_ONDJFM = (10, 11, 12, 1, 2, 3)


def monthly_totals(
    mean_per_day: float, contributing_days: int, *, years: int = BASELINE_YEARS
) -> float:
    """A monthly precipitation TOTAL from the stored mean daily value.

    The normals hold mm/day, Koppen's thresholds are mm/month, and the conversion is not a
    constant 30: ``contributing_days / years`` is the real length of that calendar month
    across the window, leap years included.
    """
    return mean_per_day * (contributing_days / years)


def classify_koppen(
    temperature_by_month: dict[int, float],
    precipitation_by_month: dict[int, float],
) -> tuple[str, dict[str, float]]:
    """``(the Koppen code, the intermediate values it was decided on)``.

    Criteria: Peel, Finlayson & McMahon (2007), with the Kottek ``As``/``Aw`` summer/winter
    dry-season split — the panel declares ``As`` for mombasa, so collapsing both onto ``Aw``
    would report a divergence that is only a convention difference. The C/D boundary is the
    0 degC isotherm, matching ``generate_zones.py``.

    The second half of the return is not decoration: a bare class tells a reader nothing
    about WHY a zone diverges, and every divergence measured so far has been one threshold.

    Latitude is deliberately NOT a parameter: summer is the warmer half-year, which the
    temperatures already state.
    """
    months = list(range(1, 13))
    temps = [temperature_by_month[m] for m in months]
    precip = [precipitation_by_month[m] for m in months]
    mat, mean_annual_precip = sum(temps) / 12, sum(precip)
    t_hot, t_cold, p_driest = max(temps), min(temps), min(precip)

    def _half_mean(half: tuple[int, ...]) -> float:
        return sum(temperature_by_month[m] for m in half) / len(half)

    summer = _AMJJAS if _half_mean(_AMJJAS) >= _half_mean(_ONDJFM) else _ONDJFM
    winter = _ONDJFM if summer is _AMJJAS else _AMJJAS
    p_summer = sum(precipitation_by_month[m] for m in summer)
    p_winter = sum(precipitation_by_month[m] for m in winter)
    p_summer_dry = min(precipitation_by_month[m] for m in summer)
    p_summer_wet = max(precipitation_by_month[m] for m in summer)
    p_winter_dry = min(precipitation_by_month[m] for m in winter)
    p_winter_wet = max(precipitation_by_month[m] for m in winter)

    # The aridity threshold moves with WHEN the rain falls, not only how much: the same
    # annual total supports vegetation differently in a winter-rain and a summer-rain regime.
    if p_winter >= 0.7 * mean_annual_precip:
        threshold = 2 * mat
    elif p_summer >= 0.7 * mean_annual_precip:
        threshold = 2 * mat + 28
    else:
        threshold = 2 * mat + 14

    decided_on = {
        "mat": round(mat, 2),
        "map": round(mean_annual_precip, 1),
        "t_hot": round(t_hot, 2),
        "t_cold": round(t_cold, 2),
        "p_driest": round(p_driest, 1),
        "threshold": round(threshold, 1),
    }

    def summer_letter() -> str:
        if t_cold < -38:
            return "d"
        if t_hot >= 22:
            return "a"
        return "b" if sum(1 for t in temps if t >= 10) >= 4 else "c"

    def seasonality_letter() -> str:
        if p_summer_dry < 40 and p_summer_dry < p_winter_wet / 3:
            return "s"
        return "w" if p_winter_dry < p_summer_wet / 10 else "f"

    # B is tested FIRST and overrides: an arid zone is arid whatever its temperature says.
    if mean_annual_precip < 10 * threshold:
        desert_or_steppe = "W" if mean_annual_precip < 5 * threshold else "S"
        return "B" + desert_or_steppe + ("h" if mat >= 18 else "k"), decided_on
    if t_cold >= 18:
        if p_driest >= 60:
            return "Af", decided_on
        if p_driest >= 100 - mean_annual_precip / 25:
            return "Am", decided_on
        dry_month_in_summer = p_driest in [precipitation_by_month[m] for m in summer]
        return ("As" if dry_month_in_summer else "Aw"), decided_on
    if t_hot <= 10:
        return ("ET" if t_hot > 0 else "EF"), decided_on
    if t_cold > 0:
        return "C" + seasonality_letter() + summer_letter(), decided_on
    return "D" + seasonality_letter() + summer_letter(), decided_on


def window_years(window_start: str, window_end: str) -> int:
    """Whole calendar years in a stored normals window, or raise.

    The divisor used to be the literal 30. Pointing ``--bundle`` at a five-year holdout
    materialization then understated every monthly precipitation total roughly sixfold, which
    moves the aridity threshold and can change a class silently (Greptile P1, PR #198).

    Whole years only, and it REFUSES anything else rather than approximating: ``n / years`` is
    meaningful as "the days in this calendar month" only when every month appears the same
    number of times. A window starting in March does not satisfy that, and quietly returning a
    wrong total is the failure this check exists to prevent.
    """
    start, end = date.fromisoformat(window_start), date.fromisoformat(window_end)
    if (start.month, start.day) != (1, 1) or (end.month, end.day) != (12, 31):
        raise ValueError(
            f"normals window {window_start}..{window_end} is not whole calendar years — "
            "a monthly total cannot be recovered from a mean daily value over it"
        )
    return end.year - start.year + 1


def read_normals(bundle: Path) -> dict[str, dict]:
    """Per-zone monthly temperature and precipitation totals out of a normals bundle."""
    pattern = (bundle / "climate" / "normals" / "*.parquet").as_posix()
    con = duckdb.connect(":memory:")
    try:
        # Bound, not interpolated: a directory name may legitimately contain an apostrophe,
        # which terminates a single-quoted SQL literal and makes DuckDB reject a path that
        # `Path` accepts (Greptile P2, PR #198).
        rows = con.execute(
            "SELECT zone_id, month, variable, mean, n, lat, window_label, window_start, "
            "window_end FROM read_parquet(?) "
            "WHERE variable IN ('temperature_2m', 'precipitation') ORDER BY zone_id, month",
            [pattern],
        ).fetchall()
    finally:
        con.close()

    zones: dict[str, dict] = {}
    for zone_id, month, variable, mean, n, lat, window, start, end in rows:
        zone = zones.setdefault(
            zone_id,
            {
                "temperature": {},
                "precipitation": {},
                "lat": lat,
                "window": window,
                "span": (start, end),
            },
        )
        # A bundle holding two window-suffixed materializations for one zone used to merge
        # them: the first row supplied the window, later rows overwrote the monthly values,
        # and the hybrid still passed the 12-month check (Greptile P1, PR #198).
        if zone["span"] != (start, end):
            raise ValueError(
                f"zone {zone_id!r} carries two different normals windows in this bundle — "
                f"{zone['span'][0]}..{zone['span'][1]} and {start}..{end}. Point --bundle at "
                "one materialization; merging them classifies a climate that never existed"
            )
        if variable == "temperature_2m":
            zone["temperature"][month] = mean
        else:
            years = window_years(start, end)
            days = n / years
            # The window is whole years, so this has to be a real month length. It is the one
            # cheap check that catches a mis-scaled divisor before it moves a threshold.
            if not 27.0 <= days <= 31.5:
                raise ValueError(
                    f"zone {zone_id!r} month {month}: {n} contributing days over {years} years "
                    f"is {days:.2f} days/month, which is not a calendar month — the stored "
                    "window and the row counts disagree"
                )
            zone["precipitation"][month] = monthly_totals(mean, n, years=years)
    return zones


def declared_labels(zones_geojson: Path) -> dict[str, tuple[str, str]]:
    """``zone_id -> (declared koppen, role)`` from the canonical zones file."""
    features = json.loads(zones_geojson.read_text(encoding="utf-8"))["features"]
    return {
        f["properties"]["zone_id"]: (f["properties"]["koppen"], f["properties"]["role"])
        for f in features
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True, help="dir with climate/normals/")
    parser.add_argument("--zones", type=Path, required=True, help="zones/zones.geojson")
    args = parser.parse_args()

    zones = read_normals(args.bundle)
    declared = declared_labels(args.zones)

    header = (
        f"{'zone':24} {'role':6} {'declared':9} {'derived':9} {'':4} "
        f"{'MAT':>7} {'MAP':>8} {'Tcold':>7} {'Thot':>7} {'Pdry':>7} {'Pthr':>7}"
    )
    print(header)
    print("-" * len(header))
    agree = disagree = unclassified = 0
    # Iterate the DECLARED labels, not the bundle. Iterating the bundle skipped a declared zone
    # with no parquet entirely: it never reached the INCOMPLETE branch and never entered the
    # totals, so a partial input read as a clean agreement unless someone noticed the zone
    # count (Greptile P2, PR #198).
    for zone_id in sorted(set(declared) | set(zones)):
        zone = zones.get(zone_id)
        label, role = declared.get(zone_id, ("?", "?"))
        if zone is None:
            unclassified += 1
            print(f"{zone_id:24} {role:6} {label:9} {'-':9} MISSING — no normals in the bundle")
            continue
        if len(zone["temperature"]) != 12 or len(zone["precipitation"]) != 12:
            unclassified += 1
            print(f"{zone_id:24} {role:6} {label:9} {'-':9} INCOMPLETE — not 12 months of both")
            continue
        derived, decided = classify_koppen(zone["temperature"], zone["precipitation"])
        matches = derived == label
        agree, disagree = agree + matches, disagree + (not matches)
        print(
            f"{zone_id:24} {role:6} {label:9} {derived:9} {'ok' if matches else 'DIFF':4} "
            f"{decided['mat']:7} {decided['map']:8} {decided['t_cold']:7} "
            f"{decided['t_hot']:7} {decided['p_driest']:7} {decided['threshold']:7}"
        )
    print("-" * len(header))
    window = next(iter(zones.values()))["window"] if zones else "?"
    print(
        f"agree {agree} / disagree {disagree} / unclassified {unclassified}   "
        f"window={window}   declared={len(declared)}   in_bundle={len(zones)}"
    )
    # A run that could not classify a declared zone is not a clean run, and the exit status is
    # the only part of this a CI step would read.
    return 1 if unclassified else 0


if __name__ == "__main__":
    raise SystemExit(main())
