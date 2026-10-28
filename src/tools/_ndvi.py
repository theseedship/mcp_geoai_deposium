"""Shared Sentinel-2 NDVI collection helpers.

The STAC-search → band-read → NDVI-stats pipeline used by both ``change.py``
(change_detect / ndvi_profile / vegetation_stress) and ``climate.py``
(climate_enrich_ndvi). It lives in its own module so the climate plane no longer
imports ``change.py``'s underscore-private helpers (audit R2 — hidden cross-tool
coupling). Behaviour is identical to the original ``change.py`` definitions.
"""

from __future__ import annotations

import asyncio

import structlog

from src.config import settings
from src.utils.dates import month_ranges
from src.utils.raster import compute_ndvi, read_band
from src.utils.stac_client import search_items_sync

log = structlog.get_logger(__name__)


def _search_sentinel2(
    bbox: list[float],
    datetime_str: str,
    cloud_cover_max: int = settings.DEFAULT_CLOUD_COVER_MAX,
    limit: int = 10,
) -> list:
    """Search STAC for Sentinel-2 L2A items."""
    return search_items_sync(
        bbox=bbox,
        datetime_str=datetime_str,
        collections=["sentinel-2-l2a"],
        cloud_cover_max=cloud_cover_max,
        max_items=min(limit, settings.STAC_SEARCH_MAX_ITEMS),
    )


def _get_band_url(item, band_keys: tuple[str, ...]) -> str | None:
    """Resolve a band URL from a STAC item, trying multiple key names."""
    for key in band_keys:
        if key in item.assets:
            return item.assets[key].href
    return None


def _compute_ndvi_for_item(item, geometry: dict) -> dict | None:
    """Read red + NIR bands from a STAC item and compute NDVI stats.

    Returns:
        Dict with date, ndvi_mean, ndvi_std, ndvi_min, ndvi_max,
        cloud_cover, item_id — or None on failure.
    """
    red_url = _get_band_url(item, ("red", "B04"))
    nir_url = _get_band_url(item, ("nir", "B08"))

    if not red_url or not nir_url:
        log.warning("missing_bands", item_id=item.id)
        return None

    red = read_band(red_url, geometry)
    nir = read_band(nir_url, geometry)

    if red is None or nir is None:
        return None

    stats = compute_ndvi(red, nir)
    if stats is None:
        return None

    mean, std, vmin, vmax = stats
    return {
        "date": item.datetime.isoformat() if item.datetime else None,
        "ndvi_mean": round(mean, 4),
        "ndvi_std": round(std, 4),
        "ndvi_min": round(vmin, 4),
        "ndvi_max": round(vmax, 4),
        "cloud_cover": item.properties.get("eo:cloud_cover"),
        "item_id": item.id,
    }


def _reraise_non_exceptions(results: list) -> None:
    """Let a cancellation through instead of filing it as a failed month.

    ``asyncio.gather(..., return_exceptions=True)`` does not hand back only ``Exception``
    subclasses. Measured on this interpreter: ``KeyboardInterrupt`` and ``SystemExit`` propagate
    out of ``gather`` and never reach the results list, but **``asyncio.CancelledError`` lands in
    it**, and it is a ``BaseException`` that is not an ``Exception``.

    That is the case worth guarding, and not the one the review named. Cancellation is how an
    abandoned MCP call or a shutting-down worker tears its tasks down; classing it as an ordinary
    per-month failure would log a partial-failure warning and carry on producing a result nobody
    is waiting for — and, when every month is cancelled, would re-raise it as though the catalogue
    were broken.
    """
    for result in results:
        if isinstance(result, BaseException) and not isinstance(result, Exception):
            raise result


async def _collect_ndvi_observations(
    bbox: list[float],
    datetime_str: str,
    mask_geometry: dict,
    cloud_cover_max: int,
) -> tuple[int, list[dict]]:
    """Search Sentinel-2 and compute NDVI for every item found.

    Blocking I/O (STAC search, GDAL reads) is offloaded to threads; a
    semaphore caps concurrent GDAL reads at settings.GDAL_CONCURRENCY.

    Returns:
        (items_searched, observations sorted by date). Items whose bands
        could not be read are dropped (and counted in a warning log).
    """
    items = await asyncio.to_thread(
        _search_sentinel2,
        bbox,
        datetime_str,
        cloud_cover_max,
        settings.STAC_SEARCH_MAX_ITEMS,
    )

    sem = asyncio.Semaphore(settings.GDAL_CONCURRENCY)

    async def _ndvi_with_limit(item):
        async with sem:
            return await asyncio.to_thread(_compute_ndvi_for_item, item, mask_geometry)

    results = await asyncio.gather(
        *[_ndvi_with_limit(item) for item in items],
        return_exceptions=True,
    )
    observations = [r for r in results if isinstance(r, dict)]

    failed = len(items) - len(observations)
    if failed:
        # The cause, not just the count (#161). This path already logged that N items were
        # dropped, which tells an operator something is wrong but never what — and when every
        # item fails for one systemic reason (an unreachable band store, a GDAL misconfiguration)
        # the count on its own reads like unlucky imagery.
        _reraise_non_exceptions(results)
        first_error = next((r for r in results if isinstance(r, Exception)), None)
        log.warning(
            "ndvi_observations_failed",
            failed=failed,
            searched=len(items),
            error=str(first_error) if first_error is not None else None,
        )

    observations.sort(key=lambda x: x["date"] or "")
    return len(items), observations


async def _collect_ndvi_monthly(
    bbox: list[float],
    mask_geometry: dict,
    start_date: str,
    end_date: str,
    cloud_cover_max: int,
    *,
    max_concurrency: int = 6,
) -> tuple[int, list[dict]]:
    """Collect one least-cloudy NDVI observation per month over ``[start_date, end_date]``.

    For each month the STAC catalogue is searched and qualifying items are tried from lowest to
    highest ``eo:cloud_cover`` until one yields NDVI. This normally streams one COG pair per month,
    and only falls back to another scene when the preferred item is unusable. Months with no usable
    qualifying scene are skipped.

    Returns:
        (months_searched, observations sorted by date). ``max_concurrency`` caps the number of
        months processed at once, bounding both STAC search fan-out and concurrent GDAL reads.

    Raises:
        Whatever the per-month work raised, when EVERY month failed. A month with no usable scene
        returns ``None`` and is skipped; a month that *failed* is a different fact, and collapsing
        the two turns a broken catalogue into an empty result (#161).
    """
    months = month_ranges(start_date, end_date)
    sem = asyncio.Semaphore(max_concurrency)

    async def _one_month(mstart: str, mend: str) -> dict | None:
        async with sem:
            items = await asyncio.to_thread(
                _search_sentinel2, bbox, f"{mstart}/{mend}", cloud_cover_max, 50
            )
            if not items:
                return None
            ordered = sorted(items, key=lambda it: it.properties.get("eo:cloud_cover", 100.0))
            for item in ordered:
                observation = await asyncio.to_thread(_compute_ndvi_for_item, item, mask_geometry)
                if observation is not None:
                    return observation
            return None

    results = await asyncio.gather(
        *[_one_month(ms, me) for ms, me in months],
        return_exceptions=True,
    )
    observations = [r for r in results if isinstance(r, dict)]

    # `return_exceptions=True` hands failures back as objects, and the `isinstance(r, dict)`
    # filter above used to DISCARD them without a word (#161). A STAC search failure — an
    # unreachable catalogue, a misconfigured STAC_URL, an expired credential — is raised inside
    # `_one_month`, so every month failed identically and the caller received
    # `(months, [])`. `ndvi_ingest` reports that as
    # "No NDVI observations for 'montpellier' in ...": a configuration error dressed up as a
    # legitimate empty result, on the DEFAULT path (`monthly=True`). Nothing distinguishes it
    # from a genuinely cloud-covered window, so a B3/B4 backfill against a broken catalogue
    # produces confidently empty series instead of an error.
    _reraise_non_exceptions(results)
    failures = [r for r in results if isinstance(r, Exception)]
    if failures and len(failures) == len(results):
        # Every month, the same way: systematic, not meteorological. Raising lets `ndvi_ingest`'s
        # existing handler answer "NDVI collection failed: {e}" with the real cause.
        raise failures[0]
    if failures:
        # A partial failure IS partial data and must not be fatal, but it must not be invisible
        # either — the count alone cannot tell an operator which of the two happened.
        log.warning(
            "ndvi_monthly_partial_failure",
            months=len(results),
            failed=len(failures),
            error=str(failures[0]),
        )

    observations.sort(key=lambda x: x["date"] or "")
    return len(months), observations
