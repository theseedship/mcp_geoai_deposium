"""Change detection and NDVI analysis tools.

Implements real satellite band reading and NDVI computation using
Sentinel-2 COG bands (B04=Red, B08=NIR) via GDAL /vsicurl/.
"""

from __future__ import annotations

import asyncio
import time
from datetime import UTC, datetime, timedelta

import structlog
from shapely.geometry import shape

from src.config import settings
from src.contracts import (
    ChangeDetectInput,
    NdviProfileInput,
    VegetationStressInput,
    check_input,
    output_schema_for,
)

# The Sentinel-2 NDVI collection cluster lives in a neutral module (audit R2) so the
# climate plane no longer imports change.py privates. Referenced through the MODULE
# (``_ndvi._search_sentinel2`` etc.), never re-imported as bare names: ``_ndvi`` is the
# single, honest patch home, so a test patching ``src.tools._ndvi._search_sentinel2``
# affects BOTH change_detect's direct search and the shared _collect_ndvi_observations
# internals (a bare re-import here would shadow only the former — Greptile #49 P2).
from src.tools import _ndvi
from src.utils.raster import (
    bbox_to_polygon,
    geometry_to_bbox,
)
from src.utils.stats import percentile_thresholds

log = structlog.get_logger(__name__)

MAX_NDVI_DECLINE_GAP_DAYS = 32
NDVI_DECLINE_RULE = "absolute_drop_with_max_gap"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _ensure_polygon_geometry(geometry: dict) -> dict:
    """If geometry is a Point, buffer it into a Polygon."""
    geom = shape(geometry)
    if geom.geom_type == "Point":
        bbox = geometry_to_bbox(geometry)
        return bbox_to_polygon(bbox)
    return geometry


def _find_best_ndvi(items: list, geometry: dict) -> tuple[dict | None, dict | None]:
    """Try items in order, return (ndvi_stats, image_info) for the first success."""
    for item in items:
        result = _ndvi._compute_ndvi_for_item(item, geometry)
        if result is not None:
            image_info = {
                "item_id": item.id,
                "date": item.datetime.isoformat() if item.datetime else None,
                "cloud_cover": item.properties.get("eo:cloud_cover"),
            }
            return result, image_info
    return None, None


def _parse_observation_datetime(value: str | None) -> datetime | None:
    """Parse a STAC observation timestamp for ordering checks."""
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed


def _observation_gap_days(date_from: str | None, date_to: str | None) -> int | None:
    """Return a positive UTC calendar-day gap, or None for invalid/non-increasing dates."""
    from_dt = _parse_observation_datetime(date_from)
    to_dt = _parse_observation_datetime(date_to)
    if from_dt is None or to_dt is None:
        return None
    gap_days = (to_dt.astimezone(UTC).date() - from_dt.astimezone(UTC).date()).days
    return gap_days if gap_days > 0 else None


def _validate_area(geometry: dict) -> str | None:
    """Return an error message if the geometry exceeds the area guard rail."""
    bbox = geometry_to_bbox(geometry)
    w, s, e, n = bbox
    area_deg2 = abs(e - w) * abs(n - s)
    if area_deg2 > settings.MAX_BBOX_AREA_DEG2:
        return (
            f"Area too large: {area_deg2:.2f} deg² exceeds limit of "
            f"{settings.MAX_BBOX_AREA_DEG2} deg² (~10,000 km²). "
            "Use a smaller geometry."
        )
    return None


# Confidence formula saturates at |dNDVI| = 0.2, which matches typical thresholds
# for "clear vegetation loss/gain" in remote sensing literature. Combined with the
# default confidence_threshold=0.7, this means changes >= ~0.14 NDVI are flagged.
_CONFIDENCE_SATURATION = 0.2

# Small NDVI swings are dominated by atmospheric/sensor noise; require a hard
# minimum delta in addition to the confidence threshold.
_NOISE_FLOOR = 0.05


def _classify_ndvi_change(ndvi_diff: float, confidence_threshold: float) -> tuple[str, float]:
    """Classify an NDVI difference into vegetation_loss / vegetation_gain / no_change.

    Returns (change_class, confidence). Pure function — no I/O.
    """
    abs_diff = abs(ndvi_diff)
    confidence = min(1.0, abs_diff / _CONFIDENCE_SATURATION)

    if ndvi_diff < -_NOISE_FLOOR and confidence >= confidence_threshold:
        return "vegetation_loss", confidence
    if ndvi_diff > _NOISE_FLOOR and confidence >= confidence_threshold:
        return "vegetation_gain", confidence
    return "no_change", confidence


def _resample_by_interval(profile: list[dict], interval_days: int) -> list[dict]:
    """Keep observations at least `interval_days` apart, in chronological order.

    Sentinel-2 has a ~5-day revisit cycle, so a 1-year window can return 60+
    items. Callers passing interval_days=30 expect ~12 points; this helper
    enforces that contract. Pure function — assumes input is already sorted
    by ISO 8601 date string.

    Args:
        profile: List of observation dicts with a "date" key (ISO 8601 or None).
        interval_days: Minimum gap in days. <=0 disables resampling.

    Returns:
        Filtered list, in the same order. Entries with date=None are dropped
        when resampling is active (no way to enforce the interval on them).
    """
    if interval_days <= 0 or len(profile) <= 1:
        return profile

    filtered: list[dict] = []
    last_dt = None
    for entry in profile:
        if entry.get("date") is None:
            continue
        cur_dt = datetime.fromisoformat(entry["date"])
        if last_dt is None or (cur_dt - last_dt).days >= interval_days:
            filtered.append(entry)
            last_dt = cur_dt
    return filtered


def _profile_statistics(profile: list[dict]) -> dict:
    """Aggregate mean/min/max/std over the ndvi_mean of each observation."""
    if not profile:
        return {"mean": None, "min": None, "max": None, "std": None}

    means = [p["ndvi_mean"] for p in profile]
    mean = sum(means) / len(means)
    std = (sum((m - mean) ** 2 for m in means) / len(means)) ** 0.5
    return {
        "mean": round(mean, 4),
        "min": round(min(means), 4),
        "max": round(max(means), 4),
        "std": round(std, 4),
    }


def _detect_profile_stress(
    profile: list[dict],
) -> tuple[bool, str | None, float | None, int | None]:
    """Scan a sorted NDVI profile for stress signals.

    Two triggers: any observation below the stress threshold, or an absolute
    drop greater than the configured threshold between consecutive observations
    no more than 32 days apart (which takes precedence for the reported
    period/drop). Pure function — no I/O.

    Returns:
        (stress_detected, stress_period, ndvi_drop, ndvi_drop_interval_days)
    """
    stress_detected = False
    stress_period = None
    ndvi_drop = None
    ndvi_drop_interval_days = None

    for entry in profile:
        if entry["ndvi_mean"] < settings.DEFAULT_NDVI_STRESS_THRESHOLD:
            stress_detected = True
            stress_period = entry["date"]
            break

    for i in range(1, len(profile)):
        gap_days = _observation_gap_days(profile[i - 1].get("date"), profile[i].get("date"))
        if gap_days is None or gap_days > MAX_NDVI_DECLINE_GAP_DAYS:
            continue
        drop = profile[i - 1]["ndvi_mean"] - profile[i]["ndvi_mean"]
        if drop > settings.DEFAULT_NDVI_DROP_THRESHOLD:
            stress_detected = True
            ndvi_drop = round(drop, 4)
            ndvi_drop_interval_days = gap_days
            stress_period = profile[i]["date"]
            break

    return stress_detected, stress_period, ndvi_drop, ndvi_drop_interval_days


def _low_ndvi_alert(current: float, threshold: float) -> list[dict]:
    """Return a low_ndvi alert if current NDVI is below threshold, else []."""
    if current >= threshold:
        return []
    return [
        {
            "type": "low_ndvi",
            "severity": "warning",
            "message": f"Current NDVI ({current:.3f}) below threshold ({threshold})",
            "ndvi_value": round(current, 4),
        }
    ]


def _first_rapid_decline(observations: list[dict], drop_threshold: float) -> dict | None:
    """Return the first above-threshold NDVI drop within the 32-day gap, or None."""
    for i in range(1, len(observations)):
        gap_days = _observation_gap_days(
            observations[i - 1].get("date"), observations[i].get("date")
        )
        if gap_days is None or gap_days > MAX_NDVI_DECLINE_GAP_DAYS:
            continue
        drop = observations[i - 1]["ndvi_mean"] - observations[i]["ndvi_mean"]
        if drop > drop_threshold:
            return {
                "type": "rapid_decline",
                "severity": "alert",
                "message": (
                    f"NDVI dropped {drop:.3f} between "
                    f"{observations[i - 1]['date']} and {observations[i]['date']}"
                ),
                "ndvi_drop": round(drop, 4),
                "date_from": observations[i - 1]["date"],
                "date_to": observations[i]["date"],
                "gap_days": gap_days,
                "max_gap_days": MAX_NDVI_DECLINE_GAP_DAYS,
                "rule": NDVI_DECLINE_RULE,
            }
    return None


# Below this many observations, percentiles over the lookback window are too noisy to be
# meaningful — fall back to the static threshold (FIERCE B6 level-1, design doc §4 Q2).
_PERCENTILE_MIN_POINTS = 8


def _resolve_ndvi_threshold(
    observations: list[dict],
    *,
    static_threshold: float,
    threshold_mode: str,
    percentile_level: int,
    min_points: int,
) -> tuple[float, dict]:
    """Resolve the effective low-NDVI threshold + a traceability metadata block (FIERCE B6).

    In ``percentile`` mode with >= ``min_points`` observations, the threshold is P{level} over the
    observed ``ndvi_mean`` series; otherwise it falls back to the static threshold (recorded with a
    ``fallback_reason``). The value compared against the threshold upstream stays ``recent_mean``
    (``trigger_value``) — the percentile only changes how the *threshold* is derived.
    """
    n = len(observations)
    # what the threshold is compared against upstream: the single-observation path uses the raw
    # value, the multi-observation path uses recent_mean, and the empty path compares nothing
    # (Greptile/Codex #36).
    if n == 0:
        trigger_value = None
    elif n == 1:
        trigger_value = "current_ndvi"
    else:
        trigger_value = "recent_mean"
    meta: dict = {
        "threshold_mode": threshold_mode,
        "static_ndvi_threshold": static_threshold,
        "percentile_level": percentile_level,
        "percentile_min_points": min_points,
        "n_points": n,
        "trigger_value": trigger_value,
        "percentile_values": None,
        "fallback_reason": None,
    }
    if threshold_mode != "percentile":
        meta["threshold_source"] = "static"
        meta["effective_ndvi_threshold"] = static_threshold
        return static_threshold, meta
    if n < min_points:
        meta["threshold_source"] = "static_fallback"
        meta["fallback_reason"] = "insufficient_points"
        meta["effective_ndvi_threshold"] = static_threshold
        return static_threshold, meta
    pct = percentile_thresholds([o["ndvi_mean"] for o in observations], level=percentile_level)
    meta["threshold_source"] = "percentile"
    meta["percentile_values"] = {"p10": pct["p10"], "p50": pct["p50"], "p90": pct["p90"]}
    meta["effective_ndvi_threshold"] = pct["selected_threshold"]
    return pct["selected_threshold"], meta


def _analyze_vegetation_stress_observations(
    observations: list[dict],
    *,
    ndvi_threshold: float,
    drop_threshold: float,
    threshold_mode: str = "static",
    percentile_level: int = 10,
    min_points: int = _PERCENTILE_MIN_POINTS,
) -> tuple[float, float | None, float | None, list[dict], dict]:
    """Pure NDVI stress analysis (no I/O) for >= 1 observation — the testable core of
    ``vegetation_stress`` (FIERCE B6 level-1).

    Resolves the (optionally percentile-based) low-NDVI threshold, splits baseline (first 75%) /
    recent (last 25%), and assembles alerts. The low-NDVI alert compares ``recent_mean`` (the
    existing trigger) against the *resolved* threshold; ``drop_threshold`` stays static.
    Returns ``(current_ndvi, baseline_ndvi, ndvi_change, alerts, threshold_meta)``.
    """
    effective, tmeta = _resolve_ndvi_threshold(
        observations,
        static_threshold=ndvi_threshold,
        threshold_mode=threshold_mode,
        percentile_level=percentile_level,
        min_points=min_points,
    )
    if len(observations) == 1:
        current = observations[0]["ndvi_mean"]
        return round(current, 4), None, None, _low_ndvi_alert(current, effective), tmeta

    split_idx = max(1, int(len(observations) * 0.75))
    baseline_obs = observations[:split_idx]
    recent_obs = observations[split_idx:] or [observations[-1]]
    baseline_mean = sum(o["ndvi_mean"] for o in baseline_obs) / len(baseline_obs)
    recent_mean = sum(o["ndvi_mean"] for o in recent_obs) / len(recent_obs)
    ndvi_change = recent_mean - baseline_mean

    alerts = _low_ndvi_alert(recent_mean, effective)
    rapid_decline = _first_rapid_decline(observations, drop_threshold)
    if rapid_decline:
        alerts.append(rapid_decline)
    if ndvi_change < -drop_threshold:
        alerts.append(
            {
                "type": "declining_trend",
                "severity": "warning",
                "message": (
                    f"Overall declining trend: baseline {baseline_mean:.3f} → "
                    f"recent {recent_mean:.3f} (change: {ndvi_change:.3f})"
                ),
                "baseline_ndvi": round(baseline_mean, 4),
                "recent_ndvi": round(recent_mean, 4),
            }
        )
    tmeta["baseline_observations"] = len(baseline_obs)
    tmeta["recent_observations"] = len(recent_obs)
    tmeta["ndvi_decline_rule"] = NDVI_DECLINE_RULE
    tmeta["ndvi_decline_max_gap_days"] = MAX_NDVI_DECLINE_GAP_DAYS
    return round(recent_mean, 4), round(baseline_mean, 4), round(ndvi_change, 4), alerts, tmeta


def _stress_response(
    geometry: dict,
    current_ndvi: float | None,
    baseline_ndvi: float | None,
    ndvi_change: float | None,
    alerts: list[dict],
    metadata: dict,
) -> dict:
    """Assemble the vegetation_stress response payload."""
    return {
        "geometry": geometry,
        "analysis": {
            "current_ndvi": current_ndvi,
            "baseline_ndvi": baseline_ndvi,
            "ndvi_change": ndvi_change,
        },
        "alerts": alerts,
        "metadata": metadata,
    }


# ---------------------------------------------------------------------------
# Tool registration
# ---------------------------------------------------------------------------


def register_change_tools(mcp):
    """Register change detection tools with the MCP server."""

    @mcp.tool(output_schema=output_schema_for("ndvi_profile"))
    async def ndvi_profile(
        geometry: dict,
        start_date: str,
        end_date: str,
        interval_days: int = 16,
        cloud_cover_max: int = settings.DEFAULT_CLOUD_COVER_MAX,
    ) -> dict:
        """Extract NDVI time-series for a given geometry.

        Reads Sentinel-2 L2A bands (B04=Red, B08=NIR) as Cloud-Optimized
        GeoTIFFs and computes NDVI = (NIR - Red) / (NIR + Red) for each
        available observation within the date range.

        Args:
            geometry: GeoJSON geometry (Point or Polygon)
            start_date: Start date "YYYY-MM-DD"
            end_date: End date "YYYY-MM-DD"
            interval_days: Minimum days between kept observations after STAC
                search (default: 16). Sentinel-2 has a ~5-day revisit cycle,
                so 16 keeps roughly one observation per Sentinel-2 cycle pair.
                Set to 0 to keep every observation.
            cloud_cover_max: Maximum cloud cover percentage

        Returns:
            NDVI time-series profile with statistics and alerts
        """
        check_input(
            NdviProfileInput,
            geometry=geometry,
            start_date=start_date,
            end_date=end_date,
            interval_days=interval_days,
            cloud_cover_max=cloud_cover_max,
        )
        log.info("ndvi_profile", start_date=start_date, end_date=end_date)
        t0 = time.monotonic()

        # Area guard rail
        area_err = _validate_area(geometry)
        if area_err:
            return {"error": area_err}

        # Ensure polygon for masking
        mask_geometry = _ensure_polygon_geometry(geometry)
        bbox = geometry_to_bbox(geometry)
        datetime_str = f"{start_date}/{end_date}"

        images_searched, profile = await _ndvi._collect_ndvi_observations(
            bbox,
            datetime_str,
            mask_geometry,
            cloud_cover_max,
        )
        images_processed = len(profile)

        # Resample to enforce minimum spacing between observations
        profile = _resample_by_interval(profile, interval_days)

        statistics = _profile_statistics(profile)
        (
            stress_detected,
            stress_period,
            ndvi_drop,
            ndvi_drop_interval_days,
        ) = _detect_profile_stress(profile)

        return {
            "geometry": geometry,
            "profile": profile,
            "statistics": statistics,
            "alerts": {
                "vegetation_stress_detected": stress_detected,
                "stress_period": stress_period,
                "ndvi_drop": ndvi_drop,
                "ndvi_drop_interval_days": ndvi_drop_interval_days,
                "ndvi_drop_max_gap_days": MAX_NDVI_DECLINE_GAP_DAYS,
                "ndvi_drop_rule": NDVI_DECLINE_RULE,
            },
            "metadata": {
                "start_date": start_date,
                "end_date": end_date,
                "interval_days": interval_days,
                "cloud_cover_max": cloud_cover_max,
                "images_searched": images_searched,
                "images_processed": images_processed,
                "elapsed_ms": round((time.monotonic() - t0) * 1000),
            },
        }

    @mcp.tool(output_schema=output_schema_for("vegetation_stress"))
    async def vegetation_stress(
        geometry: dict,
        lookback_days: int = 60,
        ndvi_threshold: float = settings.DEFAULT_NDVI_STRESS_THRESHOLD,
        drop_threshold: float = settings.DEFAULT_NDVI_DROP_THRESHOLD,
        threshold_mode: str = "static",
        percentile_level: int = 10,
    ) -> dict:
        """Detect vegetation stress based on NDVI analysis.

        Computes NDVI over a lookback window, splits into baseline (first 75%)
        and recent (last 25%), and generates stress alerts.

        Args:
            geometry: GeoJSON geometry to analyze
            lookback_days: Days to look back (default: 60, at most 36500)
            ndvi_threshold: NDVI below this value indicates stress (the *static* level-1 threshold).
            drop_threshold: Rapid drop greater than this indicates alert
            threshold_mode: "static" (default) | "percentile". In "percentile" mode the low-NDVI
                threshold is P{percentile_level} over the observed NDVI series (FIERCE B6 level-1)
                — a *statistical* threshold rather than a fixed one — with an automatic fall back to
                the static threshold when there are fewer than 8 observations. Percentiles are over
                the observed lookback window, NOT a climatology. ``drop_threshold`` stays static.
            percentile_level: percentile used as the low-NDVI threshold in "percentile" mode (1-99,
                default 10 = P10).

        Returns:
            Stress analysis results with alerts. ``metadata`` records the threshold provenance
            (``threshold_mode`` / ``threshold_source`` / ``effective_ndvi_threshold`` /
            ``percentile_values`` / ``fallback_reason`` / ``trigger_value`` …).
        """
        check_input(
            VegetationStressInput,
            geometry=geometry,
            lookback_days=lookback_days,
            ndvi_threshold=ndvi_threshold,
            drop_threshold=drop_threshold,
            threshold_mode=threshold_mode,
            percentile_level=percentile_level,
        )
        log.info(
            "vegetation_stress",
            lookback_days=lookback_days,
            ndvi_threshold=ndvi_threshold,
        )
        t0 = time.monotonic()

        # Area guard rail
        area_err = _validate_area(geometry)
        if area_err:
            return {"error": area_err}

        mask_geometry = _ensure_polygon_geometry(geometry)
        bbox = geometry_to_bbox(geometry)

        # Compute lookback window
        end_dt = datetime.now(UTC)
        start_dt = end_dt - timedelta(days=lookback_days)
        datetime_str = f"{start_dt.strftime('%Y-%m-%d')}/{end_dt.strftime('%Y-%m-%d')}"

        images_searched, observations = await _ndvi._collect_ndvi_observations(
            bbox,
            datetime_str,
            mask_geometry,
            settings.DEFAULT_CLOUD_COVER_MAX,
        )

        def _metadata(**extra) -> dict:
            return {
                "lookback_days": lookback_days,
                "ndvi_threshold": ndvi_threshold,
                "drop_threshold": drop_threshold,
                "ndvi_decline_rule": NDVI_DECLINE_RULE,
                "ndvi_decline_max_gap_days": MAX_NDVI_DECLINE_GAP_DAYS,
                "images_searched": images_searched,
                "images_processed": len(observations),
                **extra,
                "elapsed_ms": round((time.monotonic() - t0) * 1000),
            }

        if not observations:
            # Echo the threshold provenance even with zero observations, so a percentile call over an
            # empty lookback window stays distinguishable from a static one (Codex #36). With n=0 this
            # records the requested mode/level and the static fallback that WOULD apply.
            _, empty_meta = _resolve_ndvi_threshold(
                observations,
                static_threshold=ndvi_threshold,
                threshold_mode=threshold_mode,
                percentile_level=percentile_level,
                min_points=_PERCENTILE_MIN_POINTS,
            )
            return _stress_response(
                geometry,
                None,
                None,
                None,
                alerts=[],
                metadata=_metadata(
                    message="No valid observations found in lookback window",
                    **empty_meta,
                ),
            )

        # Core analysis is a pure, network-free function (testable in isolation): it resolves the
        # static-or-percentile low-NDVI threshold, splits baseline/recent, and assembles alerts.
        current, baseline, change, alerts, threshold_meta = _analyze_vegetation_stress_observations(
            observations,
            ndvi_threshold=ndvi_threshold,
            drop_threshold=drop_threshold,
            threshold_mode=threshold_mode,
            percentile_level=percentile_level,
        )
        if len(observations) == 1:
            # One observation is a current snapshot, not a baseline comparison.
            threshold_meta["message"] = (
                "Only 1 valid observation — baseline comparison requires >= 2"
            )
        return _stress_response(
            geometry,
            current,
            baseline,
            change,
            alerts=alerts,
            metadata=_metadata(**threshold_meta),
        )

    @mcp.tool(output_schema=output_schema_for("change_detect"))
    async def change_detect(
        bbox: list[float],
        date_before: str,
        date_after: str,
        change_types: list[str] | None = None,
        min_change_area_ha: float = settings.DEFAULT_CHANGE_MIN_AREA_HA,
        confidence_threshold: float = settings.DEFAULT_CHANGE_CONFIDENCE,
    ) -> dict:
        """Detect vegetation change between two dates using NDVI differencing.

        Searches for the best Sentinel-2 image in a +/- 30 day window around
        each date, computes mean NDVI over the bbox for both, and classifies
        the bbox-level change as vegetation_loss, vegetation_gain, or
        no_change.

        Note: this returns a *bbox-level summary*, not per-pixel change
        polygons. The single feature in the FeatureCollection (if change is
        detected) has the input bbox as its geometry. Per-pixel vectorization
        and the change_types/min_change_area_ha filters require the torchange
        integration tracked under ROADMAP Phase 2B.

        Args:
            bbox: Bounding box [west, south, east, north]
            date_before: Before date "YYYY-MM-DD"
            date_after: After date "YYYY-MM-DD"
            change_types: Currently ignored — only vegetation NDVI differencing
                is implemented. Reserved for torchange integration.
            min_change_area_ha: Currently ignored — no per-pixel area filter
                is applied. Reserved for torchange integration.
            confidence_threshold: Minimum confidence score (0-1) — confidence
                saturates at |dNDVI| = 0.2

        Returns:
            GeoJSON FeatureCollection with at most one bbox-level feature.
        """
        check_input(
            ChangeDetectInput,
            bbox=bbox,
            date_before=date_before,
            date_after=date_after,
            change_types=change_types,
            min_change_area_ha=min_change_area_ha,
            confidence_threshold=confidence_threshold,
        )
        log.info(
            "change_detect",
            bbox=bbox,
            date_before=date_before,
            date_after=date_after,
        )
        t0 = time.monotonic()

        # change_types and min_change_area_ha accepted for proxy compatibility
        # but currently ignored — see docstring.
        _ = change_types
        _ = min_change_area_ha

        # Area guard rail
        geometry = bbox_to_polygon(bbox)
        area_err = _validate_area(geometry)
        if area_err:
            return {"error": area_err}

        # Create +/- 30 day search windows
        window_days = 30
        before_dt = datetime.strptime(date_before, "%Y-%m-%d")
        after_dt = datetime.strptime(date_after, "%Y-%m-%d")

        before_start = (before_dt - timedelta(days=window_days)).strftime("%Y-%m-%d")
        before_end = (before_dt + timedelta(days=window_days)).strftime("%Y-%m-%d")
        after_start = (after_dt - timedelta(days=window_days)).strftime("%Y-%m-%d")
        after_end = (after_dt + timedelta(days=window_days)).strftime("%Y-%m-%d")

        # Search both windows in parallel (blocking I/O → offload to threads)
        before_items, after_items = await asyncio.gather(
            asyncio.to_thread(
                _ndvi._search_sentinel2,
                bbox,
                f"{before_start}/{before_end}",
                settings.DEFAULT_CLOUD_COVER_MAX,
                10,
            ),
            asyncio.to_thread(
                _ndvi._search_sentinel2,
                bbox,
                f"{after_start}/{after_end}",
                settings.DEFAULT_CLOUD_COVER_MAX,
                10,
            ),
        )

        # Sort by cloud cover (lowest first)
        before_items.sort(key=lambda i: i.properties.get("eo:cloud_cover", 100))
        after_items.sort(key=lambda i: i.properties.get("eo:cloud_cover", 100))

        # Find best NDVI for each window in parallel (blocking I/O → threads)
        (before_ndvi, before_image), (after_ndvi, after_image) = await asyncio.gather(
            asyncio.to_thread(_find_best_ndvi, before_items, geometry),
            asyncio.to_thread(_find_best_ndvi, after_items, geometry),
        )

        # If we couldn't get NDVI for either window
        if before_ndvi is None or after_ndvi is None:
            return {
                "type": "FeatureCollection",
                "features": [],
                "metadata": {
                    "bbox": bbox,
                    "date_before": date_before,
                    "date_after": date_after,
                    "method": "ndvi_differencing",
                    "before_image": before_image,
                    "after_image": after_image,
                    "ndvi_difference": None,
                    "message": "Could not compute NDVI for one or both time windows",
                    "before_items_searched": len(before_items),
                    "after_items_searched": len(after_items),
                    "elapsed_ms": round((time.monotonic() - t0) * 1000),
                },
            }

        before_acquired = _parse_observation_datetime(before_ndvi.get("date"))
        after_acquired = _parse_observation_datetime(after_ndvi.get("date"))
        if (
            before_acquired is not None
            and after_acquired is not None
            and before_acquired >= after_acquired
        ):
            return {
                "error": (
                    "Selected before image is not earlier than selected after image; "
                    "narrow the date windows or choose dates farther apart"
                ),
                "metadata": {
                    "bbox": bbox,
                    "date_before": date_before,
                    "date_after": date_after,
                    "before_image": before_image,
                    "after_image": after_image,
                    "before_observation_date": before_ndvi.get("date"),
                    "after_observation_date": after_ndvi.get("date"),
                    "elapsed_ms": round((time.monotonic() - t0) * 1000),
                },
            }

        # Compute NDVI difference and classify
        ndvi_diff = after_ndvi["ndvi_mean"] - before_ndvi["ndvi_mean"]
        change_class, confidence = _classify_ndvi_change(ndvi_diff, confidence_threshold)

        # Build feature if change detected
        features = []
        if change_class != "no_change":
            features.append(
                {
                    "type": "Feature",
                    "geometry": geometry,
                    "properties": {
                        "change_type": change_class,
                        "confidence": round(confidence, 4),
                        "ndvi_before": before_ndvi["ndvi_mean"],
                        "ndvi_after": after_ndvi["ndvi_mean"],
                        "ndvi_difference": round(ndvi_diff, 4),
                        "date_before": before_ndvi["date"],
                        "date_after": after_ndvi["date"],
                    },
                }
            )

        return {
            "type": "FeatureCollection",
            "features": features,
            "metadata": {
                "bbox": bbox,
                "date_before": date_before,
                "date_after": date_after,
                "confidence_threshold": confidence_threshold,
                "method": "ndvi_differencing",
                "before_image": before_image,
                "after_image": after_image,
                "ndvi_difference": round(ndvi_diff, 4),
                "elapsed_ms": round((time.monotonic() - t0) * 1000),
            },
        }
