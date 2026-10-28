"""Input contract for synchronous Open-Meteo solar requests."""

from __future__ import annotations

import re
from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

SolarSource = Literal["forecast", "era5", "satellite"]
SolarVariable = Literal[
    "shortwave_radiation",
    "diffuse_radiation",
    "direct_normal_irradiance",
    "global_tilted_irradiance",
    "temperature_2m",
    "wind_speed_10m",
]
SolarResolution = Literal["hourly", "native"]

DEFAULT_SOLAR_VARIABLES: tuple[SolarVariable, ...] = (
    "shortwave_radiation",
    "diffuse_radiation",
    "direct_normal_irradiance",
)


class SolarQuery(BaseModel):
    """A bounded point query for solar conditions and irradiance."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    source: SolarSource
    latitude: float = Field(ge=-90, le=90, allow_inf_nan=False)
    longitude: float = Field(ge=-180, le=180, allow_inf_nan=False)
    start_date: date
    end_date: date
    variables: list[SolarVariable] = Field(default_factory=lambda: list(DEFAULT_SOLAR_VARIABLES))
    model: str | None = None
    tilt: float | None = Field(default=None, ge=0, le=90, allow_inf_nan=False)
    azimuth: float | None = Field(default=None, ge=-180, le=180, allow_inf_nan=False)
    temporal_resolution: SolarResolution = "hourly"

    @field_validator("start_date", "end_date", mode="before")
    @classmethod
    def _iso_calendar_date(cls, value: object) -> object:
        if isinstance(value, datetime):
            raise ValueError("dates must be ISO calendar dates without a time")
        if isinstance(value, date):
            return value
        if isinstance(value, str) and re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
            return value
        raise ValueError("dates must use YYYY-MM-DD format")

    @field_validator("variables")
    @classmethod
    def _unique_variables(cls, values: list[SolarVariable]) -> list[SolarVariable]:
        if not values or len(values) != len(set(values)):
            raise ValueError("variables must be a non-empty list without duplicates")
        return values

    @field_validator("model")
    @classmethod
    def _non_empty_model(cls, value: str | None) -> str | None:
        if value is not None and (not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}", value)):
            raise ValueError("model must be one provider model identifier of at most 64 characters")
        return value

    @model_validator(mode="after")
    def _validate_request(self) -> SolarQuery:
        if self.end_date < self.start_date:
            raise ValueError("end_date must be on or after start_date")
        days = (self.end_date - self.start_date).days + 1
        maximum = 16 if self.source == "forecast" else 366
        if days > maximum:
            raise ValueError(f"date range cannot exceed {maximum} days for {self.source}")

        if "global_tilted_irradiance" in self.variables and (
            self.tilt is None or self.azimuth is None
        ):
            raise ValueError("global_tilted_irradiance requires explicit tilt and azimuth")
        if self.source == "satellite" and any(
            variable in self.variables for variable in ("temperature_2m", "wind_speed_10m")
        ):
            raise ValueError("satellite source does not provide temperature or wind")
        if self.temporal_resolution == "native" and self.source != "satellite":
            raise ValueError("native temporal resolution is only available for satellite")

        if self.source == "era5" and self.model not in (None, "era5"):
            raise ValueError("era5 source only supports model era5")
        if self.source == "satellite" and self.model not in (None, "eumetsat_sarah3"):
            raise ValueError("satellite source only supports model eumetsat_sarah3")
        return self
