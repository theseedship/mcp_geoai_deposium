"""Singleton STAC client to avoid reconnecting on every search."""

from __future__ import annotations

from pystac_client import Client

from src.config import settings

_client: Client | None = None


def get_stac_client() -> Client:
    """Return a reusable STAC catalog client."""
    global _client
    if _client is None:
        _client = Client.open(settings.STAC_URL)
    return _client


def search_items_sync(
    *,
    bbox: list[float] | None = None,
    datetime_str: str | None = None,
    collections: list[str] | None = None,
    cloud_cover_max: int | None = None,
    max_items: int | None = None,
    ids: list[str] | None = None,
) -> list:
    """Blocking STAC search — call via asyncio.to_thread from async code.

    Args:
        bbox: Bounding box [west, south, east, north].
        datetime_str: Date range "YYYY-MM-DD/YYYY-MM-DD".
        collections: STAC collections to search.
        cloud_cover_max: If set, filter on eo:cloud_cover < this value.
        max_items: Maximum number of items to return.
        ids: Specific item IDs to look up.

    Returns:
        List of pystac Items.
    """
    catalog = get_stac_client()
    query = {"eo:cloud_cover": {"lt": cloud_cover_max}} if cloud_cover_max is not None else None
    search = catalog.search(
        bbox=bbox,
        datetime=datetime_str,
        collections=collections,
        query=query,
        max_items=max_items,
        ids=ids,
    )
    return list(search.items())
