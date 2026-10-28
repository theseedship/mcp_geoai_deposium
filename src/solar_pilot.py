"""Ephemeral, bounded solar mesh acquisition; persistence belongs to the caller."""

from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
from collections.abc import Callable
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from src.solar_contracts import SolarQuery, SolarResolution, SolarSource, SolarVariable
from src.solar_portfolio import SolarPortfolioError, run_solar_portfolio, validate_solar_sites

MAX_REQUEST_BYTES = 64 * 1024
MAX_RESPONSE_BYTES = 2 * 1024 * 1024


class SolarPilotSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env.solar-pilot", case_sensitive=True, extra="forbid"
    )
    SOLAR_MCP_ENABLED: bool = False
    SOLAR_MCP_SOURCE_ACCESS_PROOFS: dict[str, str] = Field(default_factory=dict)

    def require_access(self, source: str) -> None:
        proof = self.SOLAR_MCP_SOURCE_ACCESS_PROOFS.get(source, "")
        if not self.SOLAR_MCP_ENABLED or not proof.strip():
            raise SolarPortfolioError("solar source is not activated for mesh collection")


class SolarPilotSite(BaseModel):
    """Strict coordinates and text identifiers shared by native and published schemas."""

    model_config = ConfigDict(extra="forbid", strict=True)
    site_id: StrictStr = Field(min_length=1, max_length=128)
    latitude: float = Field(ge=-90, le=90, allow_inf_nan=False)
    longitude: float = Field(ge=-180, le=180, allow_inf_nan=False)
    scenario_id: StrictStr = Field(default="default", max_length=128)
    tilt: float | None = Field(default=None, ge=0, le=90, allow_inf_nan=False)
    azimuth: float | None = Field(default=None, ge=-180, le=180, allow_inf_nan=False)

    @field_validator("site_id", "scenario_id", mode="before")
    @classmethod
    def trim_identifier(cls, value: Any) -> Any:
        return value.strip() if isinstance(value, str) else value


class SolarPilotInput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    action: Literal["validate", "collect"]
    sites: list[SolarPilotSite] = Field(min_length=1, max_length=2)
    source: SolarSource
    start_date: str
    end_date: str
    variables: list[SolarVariable] = Field(min_length=1)
    model: str | None = None
    temporal_resolution: SolarResolution | None = None

    @model_validator(mode="after")
    def validate_batch(self) -> SolarPilotInput:
        normalized = validate_solar_sites(
            [site.model_dump() for site in self.sites], variables=self.variables
        )
        for site in normalized:
            query = SolarQuery.model_validate(
                {
                    "source": self.source,
                    "latitude": site.latitude,
                    "longitude": site.longitude,
                    "start_date": self.start_date,
                    "end_date": self.end_date,
                    "variables": self.variables,
                    "model": self.model,
                    "tilt": site.tilt,
                    "azimuth": site.azimuth,
                    "temporal_resolution": self.temporal_resolution
                    or ("native" if self.source == "satellite" else "hourly"),
                }
            )
            if (query.end_date - query.start_date).days != 1:
                raise ValueError("solar pilot requires exactly two consecutive UTC dates")
        self.sites = [SolarPilotSite.model_validate(site.canonical()) for site in normalized]
        return self


class SolarPilotSummary(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    total: int
    succeeded: int
    failed: int
    http_acquisitions: int
    cache_reuse: int


class SolarPilotItem(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    site_id: str
    scenario_id: str
    query_hash: str
    query: dict[str, Any]
    status: Literal["pending", "succeeded", "failed"]
    attempts: int
    error: dict[str, Any] | None
    artifact_sha256: str | None
    profile_json: str | None


class SolarPilotOutput(BaseModel):
    error: str | None = None
    model_config = ConfigDict(extra="forbid", strict=True)
    schema_version: Literal["geoai-solar-pilot/v1"] = "geoai-solar-pilot/v1"
    action: Literal["validate", "collect"]
    sites: list[SolarPilotSite]
    summary: SolarPilotSummary
    items: list[SolarPilotItem]


def run_solar_pilot(
    request: SolarPilotInput | dict[str, Any],
    *,
    config: SolarPilotSettings | None = None,
    fetcher: Callable[[SolarQuery], dict[str, Any]] | None = None,
    work_dir: str | None = None,
) -> dict[str, Any]:
    request = SolarPilotInput.model_validate(
        request.model_dump() if isinstance(request, SolarPilotInput) else request
    )
    sites = validate_solar_sites(
        [site.model_dump() for site in request.sites], variables=request.variables
    )
    queries = [
        SolarQuery.model_validate(
            {
                "source": request.source,
                "latitude": s.latitude,
                "longitude": s.longitude,
                "start_date": request.start_date,
                "end_date": request.end_date,
                "variables": request.variables,
                "model": request.model,
                "tilt": s.tilt if "global_tilted_irradiance" in request.variables else None,
                "azimuth": s.azimuth if "global_tilted_irradiance" in request.variables else None,
                "temporal_resolution": request.temporal_resolution
                or ("native" if request.source == "satellite" else "hourly"),
            }
        )
        for s in sites
    ]
    if request.action == "validate":
        items = [
            {
                "site_id": s.site_id,
                "scenario_id": s.scenario_id,
                "query_hash": "",
                "query": q.model_dump(mode="json"),
                "status": "pending",
                "attempts": 0,
                "error": None,
                "artifact_sha256": None,
                "profile_json": None,
            }
            for s, q in zip(sites, queries, strict=True)
        ]
        return SolarPilotOutput(
            action="validate",
            sites=request.sites,
            summary=SolarPilotSummary(
                total=len(sites), succeeded=0, failed=0, http_acquisitions=0, cache_reuse=0
            ),
            items=[SolarPilotItem.model_validate(item) for item in items],
        ).model_dump(mode="json", exclude={"error"})
    (config or SolarPilotSettings()).require_access(request.source)
    with (
        nullcontext(work_dir)
        if work_dir is not None
        else tempfile.TemporaryDirectory(prefix="geoai-solar-pilot-")
    ) as temporary:
        summary = run_solar_portfolio(
            sites=sites,
            source=request.source,
            start_date=request.start_date,
            end_date=request.end_date,
            variables=request.variables,
            model=request.model,
            temporal_resolution=request.temporal_resolution,
            output_dir=temporary,
            concurrency=2,
            max_attempts=1,
            fetcher=fetcher,
        )
        manifest = json.loads((Path(temporary) / "manifest.json").read_bytes())
        items = []
        for item in manifest["items"].values():
            digest = item.get("artifact_sha256")
            profile = None
            if item["status"] == "succeeded":
                raw = (Path(temporary) / (item["query_hash"] + ".json")).read_bytes()
                if hashlib.sha256(raw).hexdigest() != digest:
                    raise SolarPortfolioError("solar artifact checksum mismatch")
                profile = raw.decode("utf-8")
            items.append(
                {
                    "site_id": item["site_id"],
                    "scenario_id": item["scenario_id"],
                    "query_hash": item["query_hash"],
                    "query": item["query"],
                    "status": item["status"],
                    "attempts": item["attempts"],
                    "error": item.get("error"),
                    "artifact_sha256": digest,
                    "profile_json": profile,
                }
            )
        result = SolarPilotOutput(
            action="collect",
            sites=request.sites,
            summary=SolarPilotSummary.model_validate(
                {key: summary[key] for key in SolarPilotSummary.model_fields}
            ),
            items=[SolarPilotItem.model_validate(item) for item in items],
        ).model_dump(mode="json", exclude={"error"})
        if len(json.dumps(result, ensure_ascii=False).encode("utf-8")) > MAX_RESPONSE_BYTES:
            raise SolarPortfolioError("solar pilot response exceeds byte limit")
        return result


def main() -> None:
    try:
        raw = sys.stdin.buffer.read(MAX_REQUEST_BYTES + 1)
        if not raw or len(raw) > MAX_REQUEST_BYTES:
            raise ValueError("invalid solar pilot request")
        result = run_solar_pilot(json.loads(raw), work_dir=os.environ.get("GEOAI_SOLAR_WORK_DIR"))
    except Exception:
        result = {"error": "solar pilot collection failed"}
    encoded = json.dumps(result, ensure_ascii=False).encode("utf-8")
    if len(encoded) > MAX_RESPONSE_BYTES:
        encoded = b'{"error":"solar pilot response exceeds byte limit"}'
    sys.stdout.buffer.write(encoded)
    sys.stdout.buffer.flush()


if __name__ == "__main__":
    main()
