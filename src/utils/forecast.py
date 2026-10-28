"""Forecast provider plane + router with automatic fallback (FIERCE B5).

The **durable contract** of B5 (DoD #29 criterion 2): a tiny ``ForecastProvider`` interface, a
``ForecastRouter`` that tries providers in order and **falls back on outage** (raise / timeout /
any-requested-variable-missing-or-all-null / zero usable days), and concrete Open-Meteo providers —
a **reliable physical-NWP** one (``models`` omitted = best-match) and an **AI** one
(``models=ecmwf_aifs025_single``, AIFS — live-verified all 6 vars incl. radiation 2026-06-21, ON by
default as the primary; physical is the fallback). The provider behind each slot is
swappable, so a different AI model (or a literal-FourCastNet adapter — :mod:`src.utils.nvidia_nim`)
drops in later with **no** change to the router, the ``forecast_query`` tool, or ``pattern_alert``.

A forecast is a **point-in-time snapshot** — no ``confidence`` is emitted here (that is Palier 4).
Sync (mirrors the B2 client); the async tool handler offloads via ``asyncio.to_thread``.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import httpx
import structlog

from src.config import settings
from src.utils.openmeteo import fetch_forecast_daily, normalize_forecast_daily

log = structlog.get_logger(__name__)

# AIFS publishes a slightly shorter horizon than the physical best-match (design §4.5 / §6).
_AIFS_MAX_DAYS = 15

# Provider tokens accepted by the tool's `provider=` pin (handler whitelists; "auto" = the router).
PROVIDER_AUTO = "auto"
PROVIDER_AI = "ai"
PROVIDER_PHYSICAL = "openmeteo"
PROVIDER_NIM = "nim"
ALLOWED_PROVIDERS = frozenset({PROVIDER_AUTO, PROVIDER_AI, PROVIDER_PHYSICAL, PROVIDER_NIM})


class ForecastProviderUnavailable(RuntimeError):
    """A provider is down / unconfigured / returned no usable data — the router advances.

    When raised by the **router** (all providers failed) it carries the ``provider_chain`` so the
    tool handler can surface it in the graceful ``{"error", "metadata": {"provider_chain": …}}``."""

    def __init__(self, message: str, *, provider_chain: list[dict] | None = None) -> None:
        super().__init__(message)
        self.provider_chain: list[dict] = provider_chain or []


@runtime_checkable
class ForecastProvider(Protocol):
    """One forecast source. ``fetch_daily`` returns a NORMALIZED result or raises on outage."""

    name: str
    models: str
    max_days: int

    def fetch_daily(
        self, *, latitude: float, longitude: float, variables: list[str], forecast_days: int
    ) -> dict:
        """Return ``{"dates": [...], "columns": {var: [stored-unit values]}, "max_days": int}``
        (post-normalization, trailing-null-trimmed, §4.5-usable) or raise
        :class:`ForecastProviderUnavailable` / a transport error on outage."""
        ...


# --------------------------------------------------------------------------- #
# §4.5 usability helpers (pure)
# --------------------------------------------------------------------------- #


def _trim_trailing_null_days(
    dates: list[str], columns: dict[str, list[float | None]]
) -> tuple[list[str], dict[str, list[float | None]]]:
    """Drop trailing days where EVERY requested variable is null (the live 16th-day case).

    A *leading*/interior all-null day is kept (it is a genuine gap inside the horizon); only the
    tail is trimmed so ``horizon_days`` reflects the real usable window."""
    if any(len(col) != len(dates) for col in columns.values()):
        # Ragged provider payloads are invalid/outage material; leave them unchanged so _is_usable
        # can reject them without this trim helper indexing past a short column.
        return dates, columns
    n = len(dates)
    while n > 0 and all(col[n - 1] is None for col in columns.values()):
        n -= 1
    if n == len(dates):
        return dates, columns
    return dates[:n], {v: col[:n] for v, col in columns.items()}


def _is_usable(
    dates: list[str], columns: dict[str, list[float | None]], requested: list[str]
) -> bool:
    """§4.5: usable iff ≥1 day AND every requested variable has a full-length column with ≥1 value.

    A per-day null on an otherwise-populated variable is a tolerated gap; a variable that is
    **absent, length-mismatched (ragged), or all-null across the window** makes the whole result an
    outage (→ router fallback). The ragged check (``len(col) != len(dates)``) is load-bearing: a
    *non-leading* short column slips past ``_trim_trailing_null_days`` (whose ``all(...)`` short-
    circuits on an earlier populated column) and would otherwise IndexError later while building rows
    instead of falling back (Codex P2). This also catches the AIFS all-null failure mode."""
    if not dates:
        return False
    n = len(dates)
    for v in requested:
        col = columns.get(v)
        if not col or len(col) != n or all(x is None for x in col):
            return False
    return True


# --------------------------------------------------------------------------- #
# Concrete provider — Open-Meteo (physical best-match OR an AI model)
# --------------------------------------------------------------------------- #


class OpenMeteoForecastProvider:
    """Open-Meteo ``/v1/forecast`` provider — physical best-match (``models=""``) or an AI model.

    Reads its endpoint/key from ``settings`` at construction (so a test/env override applies),
    requests ``min(forecast_days, max_days)`` (Codex #128 per-provider cap), normalizes with
    ``strict=False`` (a missing variable → outage, not a crash), trims trailing-null days, and
    raises :class:`ForecastProviderUnavailable` when the result is not §4.5-usable."""

    def __init__(
        self,
        *,
        name: str,
        models: str = "",
        max_days: int | None = None,
        base_url: str = "",
        apikey: str = "",
        timeout_s: float | None = None,
        require_models: bool = False,
    ) -> None:
        self.name = name
        self.models = models
        self.max_days = max_days if max_days is not None else settings.FORECAST_MAX_DAYS
        self.base_url = base_url or settings.FORECAST_API_BASE_URL
        self.apikey = apikey or settings.FORECAST_API_KEY
        self.timeout_s = timeout_s if timeout_s is not None else settings.FORECAST_TIMEOUT_S
        # An AI slot REQUIRES a model token: an empty token would silently send NO `models` param
        # = the physical best-match, which would serve real data MISLABELED as the AI provider and
        # defeat the experimental gating. require_models=True makes that a clean outage instead.
        self.require_models = require_models

    def fetch_daily(
        self, *, latitude: float, longitude: float, variables: list[str], forecast_days: int
    ) -> dict:
        if self.require_models and not self.models:
            raise ForecastProviderUnavailable(
                f"{self.name}: AI provider is enabled but no model token is configured "
                "(set FORECAST_AI_MODEL) — refusing to silently serve the physical best-match"
            )
        days = max(1, min(forecast_days, self.max_days))  # per-provider horizon cap (Codex #128)
        try:
            raw = fetch_forecast_daily(
                latitude=latitude,
                longitude=longitude,
                normalized_variables=variables,
                forecast_days=days,
                models=self.models,
                timeout_s=self.timeout_s,
                apikey=self.apikey,
                base_url=self.base_url,
            )
            dates, columns = normalize_forecast_daily(raw, variables)
            dates, columns = _trim_trailing_null_days(dates, columns)
        except (ValueError, IndexError) as exc:
            # A malformed payload — non-JSON 200 body (resp.json() → JSONDecodeError, a ValueError)
            # or a ragged daily series (IndexError in the trim) — is an OUTAGE, not a crash: convert
            # it so the router advances to the next provider (genuine httpx transport/5xx errors keep
            # propagating to the router's httpx.HTTPError arm). §4.5/§4.6.
            raise ForecastProviderUnavailable(
                f"{self.name}: malformed forecast response ({type(exc).__name__}: {exc})"
            ) from exc
        if not _is_usable(dates, columns, variables):
            raise ForecastProviderUnavailable(
                f"{self.name}: no usable forecast "
                "(a requested variable is missing/all-null, or zero usable days)"
            )
        return {"dates": dates, "columns": columns, "max_days": self.max_days}


# --------------------------------------------------------------------------- #
# Router — try each provider, fall back on outage (the DoD star)
# --------------------------------------------------------------------------- #


class ForecastRouter:
    """Try providers in order; on outage advance to the next. The served provider and the full
    attempted chain are returned in the tuple — the handler stamps them into ``metadata``."""

    def __init__(self, providers: list[ForecastProvider]) -> None:
        self.providers = list(providers)

    def fetch_daily(
        self, *, latitude: float, longitude: float, variables: list[str], forecast_days: int
    ) -> tuple[dict, list[dict]]:
        """Return ``(normalized_result, provider_chain)`` from the first usable provider.

        **Raises** :class:`ForecastProviderUnavailable` (carrying the chain) when ALL providers
        fail — the ``{"error"}`` translation is the handler's job (Codex #294), so this tuple
        contract stays unambiguous (it never returns a dict-error)."""
        chain: list[dict] = []
        for provider in self.providers:
            entry: dict = {
                "provider": provider.name,
                "models": provider.models or "(best-match)",
                "status": "ok",
            }
            try:
                result = provider.fetch_daily(
                    latitude=latitude,
                    longitude=longitude,
                    variables=variables,
                    forecast_days=forecast_days,
                )
            except ForecastProviderUnavailable as exc:
                entry["status"] = "outage"
                entry["reason"] = str(exc)
                chain.append(entry)
                log.warning("forecast_provider_outage", provider=provider.name, reason=str(exc))
                continue
            except httpx.HTTPError as exc:  # transport / 5xx after the client's own backoff
                entry["status"] = "outage"
                entry["reason"] = f"{type(exc).__name__}: {exc}"
                chain.append(entry)
                log.warning("forecast_provider_transport", provider=provider.name, error=str(exc))
                continue
            except Exception as exc:
                entry["status"] = "outage"
                entry["reason"] = f"{type(exc).__name__}: {exc}"
                chain.append(entry)
                log.warning("forecast_provider_error", provider=provider.name, error=str(exc))
                continue
            # Enforce the usable contract at the router boundary: a provider that *returns* (rather
            # than raises) a non-usable result — missing/all-null/length-mismatched variable or zero
            # days — is treated as an outage and we advance, instead of serving it and IndexError-ing
            # later in the handler (Codex P2). OpenMeteoForecastProvider self-validates; this also
            # guards any other ForecastProvider implementation that doesn't.
            if not _is_usable(result.get("dates") or [], result.get("columns") or {}, variables):
                entry["status"] = "outage"
                entry["reason"] = (
                    "provider returned a non-usable result "
                    "(missing/all-null/length-mismatched variable or zero days)"
                )
                chain.append(entry)
                log.warning("forecast_provider_nonusable", provider=provider.name)
                continue
            chain.append(entry)
            result["source"] = provider.name
            return result, chain

        raise ForecastProviderUnavailable(
            "all forecast providers unavailable", provider_chain=chain
        )


# --------------------------------------------------------------------------- #
# Provider chain construction (reads settings at call time)
# --------------------------------------------------------------------------- #


def _physical_provider() -> OpenMeteoForecastProvider:
    """The RELIABLE default — Open-Meteo best-match (``models`` omitted)."""
    return OpenMeteoForecastProvider(
        name="openmeteo_physical",
        models=settings.FORECAST_PHYSICAL_MODEL,  # "" => omit (best-match)
        max_days=settings.FORECAST_MAX_DAYS,
    )


def _ai_provider() -> OpenMeteoForecastProvider:
    """The AI-primary (AIFS, ``ecmwf_aifs025_single``) — ON by default; ~15-day horizon.

    ``require_models=True``: an empty ``FORECAST_AI_MODEL`` makes this provider a clean outage rather
    than silently degrading to the physical best-match mislabeled as ``openmeteo_ai`` — so in the
    ``"auto"`` chain the physical fallback serves (correctly labeled) and the ``"ai"`` pin returns a
    graceful all-down ``{"error"}``. (Guards against a future operator blanking the token; the shipped
    default is the verified ``ecmwf_aifs025_single``.)"""
    return OpenMeteoForecastProvider(
        name="openmeteo_ai",
        models=settings.FORECAST_AI_MODEL,
        max_days=min(settings.FORECAST_MAX_DAYS, _AIFS_MAX_DAYS),
        require_models=True,
    )


def build_router(provider: str = PROVIDER_AUTO) -> ForecastRouter:
    """Construct the router for a ``provider=`` pin (reads ``settings`` at call time).

    - ``"auto"`` → ``[AI, physical]`` when ``FORECAST_ENABLE_AI`` (AI primary, physical fallback —
      matching the DoD's "FourCastNet ↔ Open-Meteo" roles), else ``[physical]``.
    - ``"ai"`` / ``"openmeteo"`` → that single provider (no fallback).
    - ``"nim"`` → the config-gated NVIDIA NIM seam (raises unavailable unless configured — §4.7).
    """
    if provider == PROVIDER_PHYSICAL:
        return ForecastRouter([_physical_provider()])
    if provider == PROVIDER_AI:
        return ForecastRouter([_ai_provider()])
    if provider == PROVIDER_NIM:
        from src.utils.nvidia_nim import NvidiaNimForecastProvider

        return ForecastRouter([NvidiaNimForecastProvider()])
    # "auto" — the router chain
    chain: list[ForecastProvider] = []
    if settings.FORECAST_ENABLE_AI:
        chain.append(_ai_provider())
    chain.append(_physical_provider())
    return ForecastRouter(chain)
