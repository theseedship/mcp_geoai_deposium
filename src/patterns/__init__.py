"""Gradual pattern mining domain (so4gp).

Pure domain logic for feeding tabular/timeseries data into so4gp and
normalizing its output. No MCP wiring lives here — the future
``src/tools/patterns.py`` MCP tool imports from this package, the same way
``src/tools/change.py`` imports ``src/utils/raster.py``.

so4gp is an OPTIONAL dependency (extra ``patterns``) and requires Python
>=3.14. Modules here import it lazily; only actually mining requires the extra
installed.
"""
