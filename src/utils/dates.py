"""Dependency-free calendar helpers shared by the tools and the ops scripts.

Kept free of raster/STAC imports on purpose: an ops script that only needs date arithmetic must
not drag rasterio, shapely, numpy and pystac_client in behind it.
"""

from __future__ import annotations

from datetime import date, timedelta


def month_ranges(start_date: str, end_date: str) -> list[tuple[str, str]]:
    """Split ``[start_date, end_date]`` (inclusive ISO dates) into per-month ``(start, end)`` slices.

    Each slice is clamped to the requested window, so the first/last months are partial when the
    range does not begin/end on a month boundary. Used to drive a monthly-cadence NDVI backfill:
    Sentinel-2's ~5-day revisit means a decade-long single search would blow past the STAC item cap,
    so we walk month by month and keep the least-cloudy scene per month instead.
    """
    start = date.fromisoformat(start_date)
    end = date.fromisoformat(end_date)
    ranges: list[tuple[str, str]] = []
    cur = start
    while cur <= end:
        # December is spelled out rather than reached through "the day before the 1st of next
        # month": that construction needs `date(cur.year + 1, 1, 1)`, which for year 9999 is out
        # of `datetime`'s range and raised `ValueError: year must be in 1..9999, not 10000` — on a
        # window whose dates are both perfectly legal ISO. Callers then had to guard a helper that
        # should not have been raising: `9999-12-31` is a date, not an error.
        last_in_month = (
            date(cur.year, 12, 31)
            if cur.month == 12
            else date(cur.year, cur.month + 1, 1) - timedelta(days=1)
        )
        slice_end = min(last_in_month, end)
        ranges.append((cur.isoformat(), slice_end.isoformat()))
        if last_in_month == date.max:
            break
        cur = last_in_month + timedelta(days=1)
    return ranges
