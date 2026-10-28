"""NVIDIA FourCastNet adapter seam — config-gated, NOT implemented in B5 (design §4.7).

This is a deliberate, named :class:`~src.utils.forecast.ForecastProvider` that **locks the router's
pluggability** (and gives the outage test a real, deliberately-unavailable provider) **without**
building the heavy IC/grid/precip pipeline (out of scope — design §3). It is gated on
``settings.NVIDIA_NIM_FORECAST_URL``: unset (the v1 default) → ``fetch_daily`` raises
:class:`ForecastProviderUnavailable`, so the router skips it and it is **not** in the default
``"auto"`` chain.

Recon recorded so the future increment has the facts and the honest "why neither is a drop-in"
(re-verified against NVIDIA's primary sources 2026-07-22 — design §1.2 carries the corrections):

- **Hosted** ``https://climate.api.nvidia.com/v1/nvidia/fourcastnet`` (the "Earth-2 FourCastNet
  Preview API", packaging FourCastNet-2 / SFNO) — the published OpenAPI requires ``input_id``, an
  integer 0-3 selecting one of four canned sample inputs, with ``additionalProperties: false``, so
  **no lat/lon or date can be supplied at all**. Output variables are documented as "Present
  supported options are [w10m,t2m,msl,tcwv,z500]", and the 200 response is a tar of global contour
  **PNG images** on a 6-hour step. Not a point × daily × six-variable numeric forecast.
- **Self-host NIM** ``POST /v1/infer`` — the caller must supply their own ERA5 initial-condition
  tensor as an uploaded ``.npy``; the response is a tar of per-timestep global 721x1440 (0.25 deg)
  grids at a 6-hour step. Channel count is version-dependent (NIM 1.1.0 = SFNO ERA5 73ch;
  NIM 2.0.0, 2026-06-05, defaults to FourCastNet 3 ERA5 72ch), so do not quote a bare number.

**The decisive fact, true of every packaging:** neither channel list contains precipitation, surface
solar radiation, 2 m relative humidity or reference evapotranspiration. FourCastNet reaches only
**two** of our six §5.1 variables — ``t2m`` and 10 m wind. Self-hosting shifts the burden without
unlocking the missing ones. (Do NOT write "FourCastNet has no precipitation" unqualified: the
FourCastNet 1 research release does ship a separate precipitation *diagnostic* model; it is the
Earth-2 productized packagings that omit it.)

So a *literal* FourCastNet is an adapter-shaped later increment (Meteomatics the cheapest hosted
path, self-host NIM the expensive one) — see the design §7 ESP decision. Note that Meteomatics
documents its FourCastNet and GraphCast products as "Most relevant weather parameters except
radiation and cloud cover", so even that path does not close the radiation gap. When a viable
adapter is wired, this class grows the real client; the router/tool/contract above it do not change.
"""

from __future__ import annotations

from src.config import settings
from src.utils.forecast import ForecastProviderUnavailable

# Literal FourCastNet horizon (10 d per the FIERCE spec) — declared so the interface is complete
# even while the body is a gated stub.
_FOURCASTNET_MAX_DAYS = 10


class NvidiaNimForecastProvider:
    """Gated FourCastNet provider. Unconfigured → raises so the router falls back (§4.7)."""

    name = "nvidia_nim"
    models = "fourcastnet"

    def __init__(self) -> None:
        self.url = settings.NVIDIA_NIM_FORECAST_URL
        self.max_days = _FOURCASTNET_MAX_DAYS

    def fetch_daily(
        self, *, latitude: float, longitude: float, variables: list[str], forecast_days: int
    ) -> dict:
        if not self.url:
            raise ForecastProviderUnavailable(
                "NVIDIA NIM not configured (set NVIDIA_NIM_FORECAST_URL to enable the seam)"
            )
        # Configured but not implemented in B5 — surfaced as an outage so a misconfiguration never
        # crashes the router; the real client lands in a later increment (design §3 / §7).
        raise ForecastProviderUnavailable(
            "NVIDIA NIM FourCastNet adapter is not implemented in B5 (the model carries none of "
            "radiation, 2m relative humidity or ET0, and needs an ERA5-IC pipeline — design §4.7); "
            "use the Open-Meteo providers"
        )
