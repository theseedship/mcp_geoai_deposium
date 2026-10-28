"""STAC catalog search and download tools."""

import asyncio
import time

import structlog

from src.config import settings
from src.contracts import (
    StacDownloadInput,
    StacSearchInput,
    check_input,
    output_schema_for,
)
from src.utils.stac_client import search_items_sync

log = structlog.get_logger(__name__)


def register_stac_tools(mcp):
    """Register STAC tools with the MCP server."""

    @mcp.tool(output_schema=output_schema_for("stac_search"))
    async def stac_search(
        bbox: list[float],
        datetime: str,
        collections: list[str],
        cloud_cover_max: int = settings.DEFAULT_CLOUD_COVER_MAX,
        limit: int = 10,
    ) -> dict:
        """Search satellite imagery in STAC catalogs.

        Args:
            bbox: Bounding box [west, south, east, north]
            datetime: Date range "YYYY-MM-DD/YYYY-MM-DD"
            collections: STAC collections to search (e.g., ["sentinel-2-l2a"])
            cloud_cover_max: Maximum cloud cover percentage (default: 20)
            limit: Maximum number of results (default: 10)

        Returns:
            GeoJSON FeatureCollection with matching items
        """
        check_input(
            StacSearchInput,
            bbox=bbox,
            datetime=datetime,
            collections=collections,
            cloud_cover_max=cloud_cover_max,
            limit=limit,
        )
        log.info("stac_search", bbox=bbox, datetime=datetime, collections=collections)
        t0 = time.monotonic()

        # Clamp limit to safe range
        limit = max(1, min(int(limit), settings.STAC_SEARCH_MAX_ITEMS))

        # Search STAC (blocking I/O → offload to thread)
        items = await asyncio.to_thread(
            search_items_sync,
            bbox=bbox,
            datetime_str=datetime,
            collections=collections,
            cloud_cover_max=cloud_cover_max,
            max_items=limit,
        )

        # Convert to GeoJSON
        features = []
        for item in items:
            features.append(
                {
                    "type": "Feature",
                    "id": item.id,
                    "geometry": item.geometry,
                    "properties": {
                        "datetime": item.datetime.isoformat() if item.datetime else None,
                        "eo:cloud_cover": item.properties.get("eo:cloud_cover"),
                        "platform": item.properties.get("platform"),
                        "collection": item.collection_id,
                    },
                    "assets": {
                        name: {"href": asset.href, "type": asset.media_type}
                        for name, asset in item.assets.items()
                    },
                }
            )

        return {
            "type": "FeatureCollection",
            "features": features,
            "metadata": {
                "total_results": len(features),
                "bbox": bbox,
                "datetime": datetime,
                "collections": collections,
                "elapsed_ms": round((time.monotonic() - t0) * 1000),
            },
        }

    @mcp.tool(output_schema=output_schema_for("stac_download"))
    async def stac_download(
        item_id: str,
        collection: str,
        bands: list[str],
        bbox: list[float] | None = None,
        resolution_m: int = 10,
    ) -> dict:
        """Resolve STAC asset URLs for the requested bands of a STAC item.

        Despite the historical name `stac_download`, this tool does NOT
        download files locally. It looks up the STAC item and returns the
        HTTPS URLs of each requested band so the caller can read them
        directly via GDAL /vsicurl/ — which is what the geoai NDVI tools
        already do internally.

        Use the returned URLs with rasterio/GDAL on the consumer side, or
        let `ndvi_profile` / `vegetation_stress` / `change_detect` consume
        the imagery directly.

        Args:
            item_id: STAC item ID
            collection: STAC collection name
            bands: Band names to resolve (e.g., ["B04", "B03", "B02"])
            bbox: Echoed back in metadata; no server-side cropping is applied.
            resolution_m: Currently ignored — assets are exposed at their
                native resolution. Reserved for a future rasterio-based
                download/crop implementation.

        Returns:
            Per-band URLs and item metadata.
        """
        check_input(
            StacDownloadInput,
            item_id=item_id,
            collection=collection,
            bands=bands,
            bbox=bbox,
            resolution_m=resolution_m,
        )
        log.info("stac_download", item_id=item_id, bands=bands)
        t0 = time.monotonic()

        # Get the specific item (blocking I/O → offload to thread)
        items = await asyncio.to_thread(search_items_sync, ids=[item_id], collections=[collection])

        if not items:
            return {"error": f"Item {item_id} not found in collection {collection}"}

        item = items[0]

        bands_out = {}
        for band in bands:
            if band in item.assets:
                bands_out[band] = {
                    "href": item.assets[band].href,
                    "status": "url_ready",
                }
            else:
                bands_out[band] = {"error": f"Band {band} not found"}

        return {
            "item_id": item_id,
            "collection": collection,
            "datetime": item.datetime.isoformat() if item.datetime else None,
            "bands": bands_out,
            "metadata": {
                "bbox": bbox or item.bbox,
                "elapsed_ms": round((time.monotonic() - t0) * 1000),
            },
        }
