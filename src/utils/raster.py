"""Raster reading and NDVI computation utilities."""

from __future__ import annotations

import numpy as np
import rasterio
import structlog
from rasterio.mask import mask as rio_mask
from rasterio.warp import transform_geom
from shapely.geometry import box, mapping, shape

from src.config import settings

log = structlog.get_logger(__name__)


def geometry_to_bbox(geometry: dict) -> list[float]:
    """Extract bounding box from GeoJSON geometry, buffering Points.

    Args:
        geometry: GeoJSON geometry dict.

    Returns:
        [west, south, east, north] in EPSG:4326.
    """
    geom = shape(geometry)
    if geom.geom_type == "Point":
        buf = settings.POINT_BUFFER_DEG
        return [geom.x - buf, geom.y - buf, geom.x + buf, geom.y + buf]
    bounds = geom.bounds  # (minx, miny, maxx, maxy)
    return list(bounds)


def bbox_to_polygon(bbox: list[float]) -> dict:
    """Convert [west, south, east, north] bbox to GeoJSON Polygon.

    Args:
        bbox: Bounding box as [west, south, east, north].

    Returns:
        GeoJSON Polygon dict.
    """
    return mapping(box(*bbox))


def read_band(url: str, geometry: dict) -> np.ndarray | None:
    """Read a COG band clipped to geometry via GDAL /vsicurl/.

    Reprojects geometry from EPSG:4326 to the raster's native CRS before
    masking, since Sentinel-2 COGs are in UTM.

    Args:
        url: HTTP(S) URL to a Cloud-Optimized GeoTIFF band.
        geometry: GeoJSON geometry in EPSG:4326.

    Returns:
        2-D numpy array with nodata replaced by NaN, or None on failure.
    """
    try:
        with rasterio.open(url) as src:
            # Reproject geometry from WGS84 to raster CRS
            geom_native = transform_geom("EPSG:4326", src.crs, geometry)

            out_image, _ = rio_mask(src, [geom_native], crop=True, filled=False)

            # Squeeze to 2-D (single band), masked pixels become NaN
            data = out_image[0].astype(np.float64).filled(np.nan)

            # Replace nodata with NaN (use raster metadata, fallback to 0 for Sentinel-2 L2A)
            nodata_val = src.nodata if src.nodata is not None else 0
            data[data == nodata_val] = np.nan

            return data
    except Exception:
        log.warning("read_band_failed", url=url, exc_info=True)
        return None


def compute_ndvi(red: np.ndarray, nir: np.ndarray) -> tuple[float, float, float, float] | None:
    """Compute NDVI statistics from red and NIR bands.

    NDVI = (NIR - Red) / (NIR + Red), with zero-division protection.

    Args:
        red: 2-D array of red band values.
        nir: 2-D array of NIR band values.

    Returns:
        (mean, std, min, max) of valid NDVI pixels, or None if no valid data.
    """
    denominator = nir + red
    # np.where still evaluates the division everywhere; silence the
    # divide-by-zero warning for the positions masked out by the condition.
    with np.errstate(divide="ignore", invalid="ignore"):
        ndvi = np.where(denominator != 0, (nir - red) / denominator, np.nan)

    # Mask out NaN from input bands
    valid = ndvi[~np.isnan(ndvi)]
    if valid.size == 0:
        return None

    return (
        float(np.mean(valid)),
        float(np.std(valid)),
        float(np.min(valid)),
        float(np.max(valid)),
    )
