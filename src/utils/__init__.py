"""Utility modules for GeoAI MCP Server — import them by name, not from here.

This package deliberately re-exports NOTHING. It used to hoist
``src.utils.raster`` and ``src.utils.stac_client`` into the package root for
convenience, which meant that importing ANY sibling — ``src.utils.dates`` for
date arithmetic, ``src.utils.duckdb_conn`` for a connection — executed this file
first and dragged rasterio, shapely, numpy and pystac_client in with it. An ops
script that only needs to split a window into months paid for the whole raster
stack, and the dependency-free homes those helpers were moved into never paid
off. No caller ever used the re-exported names: every one of them imports the
submodule it wants directly (``from src.utils.raster import compute_ndvi``).

Keep it empty. A convenience re-export here is a mandatory import for every
consumer of every other module in the package.
"""
