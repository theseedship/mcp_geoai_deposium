"""Short-term forecast tool (FIERCE B5) — ``forecast_query`` (MCP tool #11).

A per-point daily forecast for the six terra-cognita §5.1 variables, served through the
:class:`~src.utils.forecast.ForecastRouter` with **automatic fallback**. The shipped ``auto`` chain
is AIFS AI-primary (``ecmwf_aifs025_single``) → Open-Meteo physical best-match, unless
``FORECAST_ENABLE_AI=false`` forces physical-only. The router is the durable contract; the provider
behind each slot is swappable (design §4.2/§4.3). The forecast is a **point-in-time snapshot** —
no ``confidence`` (Palier 4).

Network-free testable: the router/providers are constructed from ``settings`` and the only live I/O
is the Open-Meteo HTTP call inside ``fetch_forecast_daily`` (which the tests monkeypatch). The zone
resolution + GeoParquet I/O reuse the climate plane (one implementation, design §4.8).
"""

from __future__ import annotations

import asyncio
import time
from datetime import UTC, datetime

import structlog

from src.config import settings
from src.contracts import ForecastQueryInput, check_input, output_schema_for

# Reuse the climate plane's zone resolution + WKB GeoParquet I/O (one implementation — §4.8).
# climate.py does NOT import forecast.py, so this is a one-directional dependency (no cycle).
from src.tools.climate import (
    _SAFE_ZONE_RE,
    _resolve_zone,
    _validate_dst,
    _value_stats,
    _write_geoparquet,
)
from src.utils.forecast import (
    ALLOWED_PROVIDERS,
    ForecastProviderUnavailable,
    build_router,
)
from src.utils.openmeteo import (
    FORECAST_NORMALIZED_VARS,
    NORMALIZED_VARS,
    OPEN_METEO_DATA_LICENSE,
    forecast_provider_map,
)

log = structlog.get_logger(__name__)


def _forecast_output_uri(zone_id: str, issued_date: str) -> str | None:
    """``s3://{bucket}/climate/forecast/{zone_id}_{issued_date}.parquet`` if a bucket is configured.

    Built in-handler (not via ``_default_output_uri``, which has no date param — §4.8): a forecast is
    dated by its issue day, so re-issuing on a new day writes a new file rather than clobbering."""
    if not _SAFE_ZONE_RE.fullmatch(zone_id) or not settings.S3_BUCKET:
        return None
    return f"s3://{settings.S3_BUCKET}/climate/forecast/{zone_id}_{issued_date}.parquet"


def register_forecast_tools(mcp):
    """Register the forecast tool with the MCP server."""

    @mcp.tool(output_schema=output_schema_for("forecast_query"))
    async def forecast_query(
        zone_id: str | None = None,
        latitude: float | None = None,
        longitude: float | None = None,
        variables: list[str] | None = None,
        forecast_days: int = settings.FORECAST_MAX_DAYS,
        provider: str = "auto",
        write_output: bool = False,
        output_path: str | None = None,
    ) -> dict:
        """Short-term daily forecast for a zone (or explicit point) with automatic fallback (B5).

        Asks the ForecastRouter for a daily forecast of the six terra-cognita §5.1 variables: it
        tries providers in order and on outage (raise / timeout / any requested variable
        missing-or-all-null / zero usable days) falls back to the next — no error surfaced — and
        records the served ``source`` + the full ``provider_chain`` in metadata.

        Args:
            zone_id: A zone identifier in zones/zones.geojson (resolves a centroid). Provide this
                XOR (latitude AND longitude).
            latitude: Explicit point latitude (with longitude). XOR zone_id.
            longitude: Explicit point longitude (with latitude). XOR zone_id.
            variables: Normalized variable names (default: all six §5.1 variables). Explicit
                temperature_2m_min adds daily minimum air temperature at 2 m in Celsius over
                the full UTC calendar day, not a night-specific minimum.
            forecast_days: Horizon in days (clamped to [1, FORECAST_MAX_DAYS]; per-provider capped).
            provider: "auto" (router chain) | "ai" | "openmeteo" (physical) | "nim" (if configured).
            write_output: If True, persist a dated WKB GeoParquet (default False — forecasts are
                ephemeral). Point-mode requires an explicit output_path.
            output_path: Explicit destination (zone-mode default is derived from S3_BUCKET + date).

        Returns:
            {forecast[], statistics, metadata{source, provider_chain, horizon_days, …}, [output]}.
        """
        check_input(
            ForecastQueryInput,
            zone_id=zone_id,
            latitude=latitude,
            longitude=longitude,
            variables=variables,
            forecast_days=forecast_days,
            provider=provider,
            write_output=write_output,
            output_path=output_path,
        )
        log.info("forecast_query", zone_id=zone_id, provider=provider, forecast_days=forecast_days)
        t0 = time.monotonic()

        if provider not in ALLOWED_PROVIDERS:
            return {
                "error": f"Unknown provider {provider!r}; supported: {sorted(ALLOWED_PROVIDERS)}"
            }

        # Variables: default all six; dedup (order-preserving); reject unknowns gracefully.
        requested = (
            list(dict.fromkeys(variables)) if variables is not None else list(NORMALIZED_VARS)
        )
        if not requested:
            return {"error": "variables must not be empty"}
        unknown = [v for v in requested if v not in FORECAST_NORMALIZED_VARS]
        if unknown:
            return {
                "error": f"Unknown variables {unknown}; supported: {list(FORECAST_NORMALIZED_VARS)}"
            }

        days = max(
            1, min(forecast_days, settings.FORECAST_MAX_DAYS)
        )  # clamp upper bound gracefully

        # Resolve the point: zone centroid OR explicit lat/lon (contract already enforced the XOR;
        # re-checked here so a direct handler call also fails gracefully rather than raising).
        if zone_id is not None:
            zone = _resolve_zone(zone_id)
            if zone is None:
                return {"error": f"Unknown zone_id {zone_id!r} (not in zones/zones.geojson)"}
            lon, lat = zone["lon"], zone["lat"]
        elif latitude is not None and longitude is not None:
            lat, lon = float(latitude), float(longitude)
        else:
            return {"error": "provide exactly one of zone_id OR (latitude AND longitude)"}

        # Persistence destination resolved + validated BEFORE the fetch (fail fast — mirrors climate).
        issued_at = datetime.now(UTC)
        issued_date = issued_at.date().isoformat()
        dst = None
        if write_output:
            if zone_id is not None:
                dst = output_path or _forecast_output_uri(zone_id, issued_date)
                if not dst:
                    return {"error": "No output destination — set S3_BUCKET or pass output_path"}
            else:
                # Point-mode has no stable name → an explicit output_path is REQUIRED (§4.8).
                if not output_path:
                    return {
                        "error": "write_output in point-mode requires an explicit output_path "
                        "(no zone_id to derive a stable name from)"
                    }
                dst = output_path
            try:
                _validate_dst(dst)
            except ValueError as e:
                return {"error": f"Invalid output_path: {e}"}

        router = build_router(provider)
        try:
            result, chain = await asyncio.to_thread(
                router.fetch_daily,
                latitude=lat,
                longitude=lon,
                variables=requested,
                forecast_days=days,
            )
        except ForecastProviderUnavailable as exc:
            log.warning("forecast_all_providers_down", provider_chain=exc.provider_chain)
            return {
                "error": "all forecast providers unavailable",
                "metadata": {
                    "provider_chain": exc.provider_chain,
                    "requested_days": days,
                    "elapsed_ms": round((time.monotonic() - t0) * 1000),
                },
            }
        except Exception as e:  # an unexpected non-outage failure — surface, never raise to caller
            log.error("forecast_query_error", error=str(e))
            return {"error": f"Forecast failed: {e}"}

        dates, columns = result["dates"], result["columns"]
        forecast = [
            {"date": d, **{v: columns[v][i] for v in requested}} for i, d in enumerate(dates)
        ]

        written = None
        if write_output and dst:
            records = [
                {
                    "date": d,
                    "zone_id": zone_id,  # None in point-mode
                    "lat": lat,
                    "lon": lon,
                    "wkt": f"POINT({lon} {lat})",
                    **{v: columns[v][i] for v in requested},
                }
                for i, d in enumerate(dates)
            ]
            staging = [
                ("date", "VARCHAR"),
                ("zone_id", "VARCHAR"),
                ("lat", "DOUBLE"),
                ("lon", "DOUBLE"),
                ("wkt", "VARCHAR"),
                *[(v, "DOUBLE") for v in requested],
            ]
            try:
                await asyncio.to_thread(
                    _write_geoparquet, records, columns=staging, dst=dst, date_cols=("date",)
                )
                written = dst
            except Exception as e:
                log.error("forecast_query_write_error", error=str(e))
                return {"error": f"Forecast GeoParquet write failed: {e}"}

        out: dict = {
            "forecast": forecast,
            "statistics": _value_stats(columns),
            "output": written,
            "metadata": {
                "source": result.get("source"),
                "provider_chain": chain,
                "horizon_days": len(
                    dates
                ),  # actual usable days served (a shorter horizon ≠ outage)
                "requested_days": days,
                "lat": lat,
                "lon": lon,
                "variables": requested,
                "issued_at": issued_at.isoformat(),  # a forecast is a snapshot, not reproducible
                "provider_map": forecast_provider_map(requested),
                "license": OPEN_METEO_DATA_LICENSE,
                "elapsed_ms": round((time.monotonic() - t0) * 1000),
            },
        }
        if zone_id is not None:
            out["zone_id"] = zone_id
        else:
            out["point"] = {"latitude": lat, "longitude": lon}
        return out
