"""Experimental Palier-4 alerts from O3 pattern libraries plus B5 forecasts.

This tool is deliberately conservative: it returns an explainable
``provisional_score`` but leaves ``confidence`` as ``None``. Product-grade
confidence still needs ratified thresholds and backtested calibration.
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any, Literal

import structlog
from pydantic import SkipValidation

from src.b6_explanatory import finite_float as _finite_float
from src.b6_explanatory import forecast_alignment as _forecast_alignment
from src.b6_explanatory import lag_decay as _lag_decay
from src.b6_explanatory import series_delta as _series_delta  # noqa: F401 - historical name
from src.b6_historical import score_historical_case
from src.b6_saved import score_saved_inputs
from src.b6_saved_contracts import B6HistoricalInputs, B6SavedInputs, preflight_saved_inputs
from src.config import settings
from src.contracts import PatternAlertInput, check_input, output_schema_for
from src.patterns.parquet_bridge import load_dataframe
from src.tools.forecast import register_forecast_tools
from src.tools.patterns import _resolve_bucket_uri
from src.utils.path_policy import resolve_local_parquet_path

log = structlog.get_logger(__name__)

DEFAULT_PATTERNS_SOURCE = "climate/o3/holdout/q3_2023/tgrad_validated.parquet"
CONFIDENCE_STATUS = "experimental_unratified"

_SUPPORTED_SOURCE_KINDS = {"parquet", "geoparquet", "csv", "json"}


class _CaptureMCP:
    def __init__(self) -> None:
        self.tools: dict[str, Any] = {}

    def tool(self, *_args, **_kwargs):
        def deco(fn):
            self.tools[fn.__name__] = fn
            return fn

        return deco


def _forecast_query_tool():
    mcp = _CaptureMCP()
    register_forecast_tools(mcp)
    return mcp.tools["forecast_query"]


async def _run_forecast_query(**kwargs) -> dict[str, Any]:
    return await _forecast_query_tool()(**kwargs)


def _resolve_patterns_source(source: str, *, allow_local_source: bool) -> str:
    if source.startswith(("s3://", "http://", "https://")):
        if not settings.S3_BUCKET:
            raise ValueError("S3_BUCKET is not configured — cannot resolve patterns_source")
        return _resolve_bucket_uri(source, bucket=settings.S3_BUCKET, role="patterns_source")
    if source.startswith(("/", ".")):
        if not allow_local_source:
            raise ValueError("local patterns_source requires allow_local_source=True")
        return resolve_local_parquet_path(source, role="patterns_source").as_posix()
    if "://" in source:
        raise ValueError(f"Unsupported patterns_source URI scheme: {source!r}")
    if not settings.S3_BUCKET:
        raise ValueError("S3_BUCKET is not configured — cannot resolve relative patterns_source")
    return _resolve_bucket_uri(source, bucket=settings.S3_BUCKET, role="patterns_source")


def _json_list(value: Any) -> list[Any]:
    if value is None:
        return []
    try:
        import pandas as pd

        if pd.isna(value):
            return []
    except (ImportError, TypeError, ValueError):
        pass
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return []
        return parsed if isinstance(parsed, list) else []
    return []


def _int_value(value: Any, default: int = 0) -> int:
    numeric = _finite_float(value)
    return default if numeric is None else int(numeric)


def _recommend_action(
    *,
    provisional_score: float,
    horizon_lag_match: bool,
    forecast_alignment: float,
) -> str:
    if not horizon_lag_match:
        return "monitor_historical_pattern"
    if forecast_alignment <= 0:
        return "no_action"
    if provisional_score >= 0.25:
        return "watch_conditions"
    return "monitor_conditions"


def _explain(alert: dict[str, Any]) -> str:
    if not alert["horizon_lag_match"]:
        return (
            f"Pattern lag {alert['lag_days']:.2f} d is outside the forecast horizon; "
            "kept for context with provisional_score=0."
        )
    return (
        f"{alert['forecast_matches']}/{alert['forecast_items']} forecast drivers match the "
        f"pattern direction; support={alert['support']:.3f}, stability={alert['stability']:.3f}, "
        f"lag_decay={alert['lag_decay']:.3f}."
    )


def _row_value(row: dict[str, Any], key: str, default: Any = None) -> Any:
    value = row.get(key, default)
    try:
        import pandas as pd

        if pd.isna(value):
            return default
    except (ImportError, TypeError, ValueError):
        pass
    return value


def _alert_from_row(
    row: dict[str, Any],
    *,
    zone_id: str,
    forecast_rows: list[dict[str, Any]],
    forecast_horizon_days: int,
    decay_halflife_days: float,
) -> dict[str, Any]:
    support = _finite_float(row.get("support")) or 0.0
    stability = _finite_float(row.get("stability")) or 0.0
    lag_days = _finite_float(row.get("lag_days")) or 0.0
    antecedent = str(_row_value(row, "antecedent", "") or "")
    zones_validated = [str(z) for z in _json_list(row.get("zones_validated"))]
    forecast_alignment, evidence = _forecast_alignment(
        antecedent=antecedent,
        forecast_rows=forecast_rows,
    )
    horizon_lag_match = abs(lag_days) <= float(forecast_horizon_days)
    lag_decay = _lag_decay(lag_days, decay_halflife_days)
    score = support * stability * forecast_alignment * lag_decay
    if not horizon_lag_match:
        score = 0.0
    alert = {
        "pattern_id": _int_value(_row_value(row, "pattern_id", 0)),
        "pattern": str(_row_value(row, "pattern", "") or ""),
        "antecedent": antecedent,
        "consequent": str(_row_value(row, "consequent", "") or ""),
        "algorithm": str(_row_value(row, "algorithm", "") or ""),
        "support": support,
        "stability": stability,
        "zones_validated": zones_validated,
        "n_folds": _int_value(_row_value(row, "n_folds", 0)),
        "lag_days": lag_days,
        "lag_decay": lag_decay,
        "horizon_lag_match": horizon_lag_match,
        "forecast_alignment": forecast_alignment,
        "forecast_items": len(evidence),
        "forecast_matches": sum(1 for item in evidence if item.get("matched")),
        "forecast_evidence": evidence,
        "provisional_score": score,
        "confidence": None,
        "confidence_status": CONFIDENCE_STATUS,
        "zone_validated": zone_id in zones_validated,
    }
    alert["recommended_action"] = _recommend_action(
        provisional_score=score,
        horizon_lag_match=horizon_lag_match,
        forecast_alignment=forecast_alignment,
    )
    alert["explanation"] = _explain(alert)
    return alert


def _is_o3_validated(row: dict[str, Any], *, min_validated_zones: int) -> bool:
    zones_validated = _json_list(row.get("zones_validated"))
    n_folds = _int_value(_row_value(row, "n_folds", 0))
    stability = _finite_float(row.get("stability"))
    return n_folds >= 2 and stability is not None and len(zones_validated) >= min_validated_zones


def register_pattern_alert_tools(mcp):
    """Register the experimental Palier-4 alert tool."""

    @mcp.tool(output_schema=output_schema_for("pattern_alert"))
    async def pattern_alert(
        zone_id: str,
        patterns_source: str = DEFAULT_PATTERNS_SOURCE,
        source_kind: str = "parquet",
        allow_local_source: bool = False,
        forecast_horizon_days: int = settings.FORECAST_MAX_DAYS,
        provider: str = "auto",
        min_support: float = 0.5,
        min_stability: float = 0.5,
        min_validated_zones: int = 1,
        require_zone_validated: bool = True,
        decay_halflife_days: float = 10.0,
        max_alerts: int = 10,
        mode: Literal["live", "saved_inputs_j1", "historical_case_j1"] = "live",
        # Preserve the published typed schema while passing the raw decoded value
        # into our bounded/generic refusal boundary before nested model validation.
        saved_inputs: SkipValidation[B6SavedInputs | None] = None,
        historical_inputs: SkipValidation[B6HistoricalInputs | None] = None,
    ) -> dict[str, Any]:
        """Experimental alert scoring over validated O3 patterns and short-term forecast.

        Returns explainable provisional scores only. ``confidence`` deliberately
        remains ``null`` until thresholds and calibration are ratified.
        Explicit ``saved_inputs_j1`` mode verifies the pinned original package and
        scores the supplied saved forecast without provider acquisition or writes.
        ``historical_case_j1`` instead selects one pinned server-owned historical
        case for contextual engineering arithmetic; scientific adoption and
        operational readiness remain pending.
        """
        if mode == "historical_case_j1":
            try:
                if (
                    settings.MCP_TRANSPORT.lower() != "stdio"
                    and not settings.GEOAI_MESH_AUTH_TOKEN.strip()
                ):
                    raise ValueError("authenticated mesh is required")
                if saved_inputs is not None:
                    raise ValueError("historical mode refuses saved inputs")
                preflight_saved_inputs(zone_id, historical_inputs)
                check_input(
                    PatternAlertInput,
                    zone_id=zone_id,
                    mode=mode,
                    saved_inputs=saved_inputs,
                    historical_inputs=historical_inputs,
                    patterns_source=patterns_source,
                    source_kind=source_kind,
                    allow_local_source=allow_local_source,
                    forecast_horizon_days=forecast_horizon_days,
                    provider=provider,
                    min_support=min_support,
                    min_stability=min_stability,
                    min_validated_zones=min_validated_zones,
                    require_zone_validated=require_zone_validated,
                    decay_halflife_days=decay_halflife_days,
                    max_alerts=max_alerts,
                )
                return await asyncio.to_thread(score_historical_case, zone_id, historical_inputs)
            except Exception:  # noqa: BLE001 - fixed private refusal, no caller content logged
                log.warning("b6_historical_inputs_refused")
                return {
                    "error": "Historical B6 inputs refused",
                    "metadata": {
                        "mode": "historical_case_j1",
                        "code": "B6_HISTORICAL_INPUTS_REFUSED",
                    },
                }

        if mode == "saved_inputs_j1":
            failure_category = "mesh_auth_unavailable"
            try:
                # FastMCP enforces the configured bearer token on HTTP requests.
                # Compute-only deployments may start without it: refuse this mode.
                if (
                    settings.MCP_TRANSPORT.lower() != "stdio"
                    and not settings.GEOAI_MESH_AUTH_TOKEN.strip()
                ):
                    raise ValueError("authenticated mesh is required")
                failure_category = "invalid_request"
                if historical_inputs is not None:
                    raise ValueError("saved mode refuses historical inputs")
                preflight_saved_inputs(zone_id, saved_inputs)
                check_input(
                    PatternAlertInput,
                    zone_id=zone_id,
                    mode=mode,
                    saved_inputs=saved_inputs,
                    historical_inputs=historical_inputs,
                    patterns_source=patterns_source,
                    source_kind=source_kind,
                    allow_local_source=allow_local_source,
                    forecast_horizon_days=forecast_horizon_days,
                    provider=provider,
                    min_support=min_support,
                    min_stability=min_stability,
                    min_validated_zones=min_validated_zones,
                    require_zone_validated=require_zone_validated,
                    decay_halflife_days=decay_halflife_days,
                    max_alerts=max_alerts,
                )
                failure_category = "saved_scoring"
                return await asyncio.to_thread(score_saved_inputs, zone_id, saved_inputs)
            except Exception as exc:  # noqa: BLE001 - fixed client boundary and safe internal categories
                if failure_category == "saved_scoring":
                    if isinstance(exc, OSError):
                        failure_category = "package_io_failed"
                    elif isinstance(exc, ValueError):
                        failure_category = "input_or_package_verification_failed"
                    else:
                        failure_category = "scorer_failed"
                # No exception message/traceback, path, request, or forecast enters logs.
                log.warning("b6_saved_inputs_refused", category=failure_category)
                return {
                    "error": "Saved B6 inputs refused",
                    "metadata": {"mode": "saved_inputs_j1", "code": "B6_SAVED_INPUTS_REFUSED"},
                }

        check_input(
            PatternAlertInput,
            mode=mode,
            saved_inputs=saved_inputs,
            historical_inputs=historical_inputs,
            zone_id=zone_id,
            patterns_source=patterns_source,
            source_kind=source_kind,
            allow_local_source=allow_local_source,
            forecast_horizon_days=forecast_horizon_days,
            provider=provider,
            min_support=min_support,
            min_stability=min_stability,
            min_validated_zones=min_validated_zones,
            require_zone_validated=require_zone_validated,
            decay_halflife_days=decay_halflife_days,
            max_alerts=max_alerts,
        )
        log.info("pattern_alert", zone_id=zone_id, patterns_source=patterns_source)
        t0 = time.monotonic()
        if source_kind not in _SUPPORTED_SOURCE_KINDS:
            return {"error": f"Invalid source_kind {source_kind!r}"}
        try:
            resolved_source = _resolve_patterns_source(
                patterns_source,
                allow_local_source=allow_local_source,
            )
        except ValueError as exc:
            return {"error": str(exc)}

        try:
            frame = await asyncio.to_thread(
                load_dataframe,
                resolved_source,
                kind=source_kind,
                order_by=None,
            )
        except Exception as exc:  # noqa: BLE001 - surface source/read failures to the agent
            return {"error": f"Failed to read patterns_source: {exc}"}

        rows = frame.to_dict(orient="records")
        if not rows:
            return {
                "zone_id": zone_id,
                "alerts": [],
                "statistics": {"n_patterns_read": 0, "n_alerts": 0},
                "metadata": {
                    "patterns_source": resolved_source,
                    "confidence_status": CONFIDENCE_STATUS,
                    "note": "empty pattern library",
                    "elapsed_ms": round((time.monotonic() - t0) * 1000),
                },
            }

        if not any(_is_o3_validated(row, min_validated_zones=min_validated_zones) for row in rows):
            return {
                "error": "patterns_source does not look like an O3 validated library",
                "metadata": {
                    "required": ["zones_validated", "n_folds", "stability"],
                    "confidence_status": CONFIDENCE_STATUS,
                },
            }

        forecast = await _run_forecast_query(
            zone_id=zone_id,
            variables=None,
            forecast_days=forecast_horizon_days,
            provider=provider,
            write_output=False,
        )
        if forecast.get("error"):
            return {
                "error": f"forecast_query failed: {forecast['error']}",
                "metadata": forecast.get("metadata") or {},
            }
        forecast_rows = forecast.get("forecast") or []
        forecast_metadata = forecast.get("metadata") or {}
        effective_horizon_days = _int_value(
            forecast_metadata.get("horizon_days"), len(forecast_rows) or forecast_horizon_days
        )
        alerts: list[dict[str, Any]] = []
        rejected = {
            "low_support": 0,
            "low_stability": 0,
            "not_o3_validated": 0,
            "zone_not_validated": 0,
        }
        for row in rows:
            support = _finite_float(row.get("support"))
            stability = _finite_float(row.get("stability"))
            if not _is_o3_validated(row, min_validated_zones=min_validated_zones):
                rejected["not_o3_validated"] += 1
                continue
            if support is None or support < min_support:
                rejected["low_support"] += 1
                continue
            if stability is None or stability < min_stability:
                rejected["low_stability"] += 1
                continue
            if require_zone_validated and zone_id not in _json_list(row.get("zones_validated")):
                rejected["zone_not_validated"] += 1
                continue
            alerts.append(
                _alert_from_row(
                    row,
                    zone_id=zone_id,
                    forecast_rows=forecast_rows,
                    forecast_horizon_days=effective_horizon_days,
                    decay_halflife_days=decay_halflife_days,
                )
            )

        alerts.sort(
            key=lambda alert: (
                -alert["provisional_score"],
                -alert["forecast_alignment"],
                -alert["stability"],
                -alert["support"],
                alert["pattern_id"],
            )
        )
        alerts = alerts[:max_alerts]
        return {
            "zone_id": zone_id,
            "alerts": alerts,
            "statistics": {
                "n_patterns_read": len(rows),
                "n_patterns_considered": len(rows) - sum(rejected.values()),
                "n_alerts": len(alerts),
                "rejected": rejected,
                "forecast_days": len(forecast_rows),
                "effective_horizon_days": effective_horizon_days,
                "elapsed_ms": round((time.monotonic() - t0) * 1000),
            },
            "metadata": {
                "patterns_source": resolved_source,
                "provider": provider,
                "forecast_source": forecast_metadata.get("source"),
                "provider_chain": forecast_metadata.get("provider_chain", []),
                "requested_forecast_horizon_days": forecast_horizon_days,
                "effective_forecast_horizon_days": effective_horizon_days,
                "decay_halflife_days": decay_halflife_days,
                "confidence_status": CONFIDENCE_STATUS,
                "note": (
                    "Experimental Palier-4 MVP: provisional_score is support x stability "
                    "x forecast_alignment x lag_decay; confidence remains null."
                ),
            },
        }
