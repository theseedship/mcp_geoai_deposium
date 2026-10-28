"""MCP Tools for GeoAI Server."""

from src.tools.change import register_change_tools
from src.tools.climate import register_climate_tools
from src.tools.forecast import register_forecast_tools
from src.tools.pattern_alert import register_pattern_alert_tools
from src.tools.patterns import register_pattern_tools
from src.tools.solar import register_solar_tools
from src.tools.spatial import register_spatial_tools
from src.tools.stac import register_stac_tools

__all__ = [
    "register_solar_tools",
    "register_stac_tools",
    "register_spatial_tools",
    "register_change_tools",
    "register_pattern_tools",
    "register_climate_tools",
    "register_forecast_tools",
    "register_pattern_alert_tools",
]
