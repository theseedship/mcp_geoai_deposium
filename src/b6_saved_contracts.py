"""Closed input envelope and pre-validation processing bounds for saved B6."""

import json
from datetime import date, datetime
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from src.patterns.library_schema import Algorithm, GradualPatternRecord, PatternLibrary

MAX_SAVED_ENVELOPE_BYTES = 128 * 1024
MAX_SAVED_RECORDS = 256
MAX_SAVED_JSON_DEPTH = 16
MAX_SAVED_JSON_NODES = 32768
MAX_SAVED_CONTAINER_ITEMS = 4096
Sha256 = Annotated[str, Field(pattern=r"^[a-f0-9]{64}$")]


def preflight_saved_inputs(zone_id: str, inputs: Any) -> None:
    """Bound raw decoded input before constructing any nested Pydantic models.

    Walk one child at a time with bounded depth/work, without copying containers
    or serializing the complete request. This is a semantic processing bound;
    the MCP transport has already decoded the incoming JSON object.
    """
    remaining = MAX_SAVED_ENVELOPE_BYTES
    nodes = 0

    def charge(size: int) -> None:
        nonlocal remaining
        remaining -= size
        if remaining < 0:
            raise ValueError("saved envelope exceeds 128 KiB")

    def visit(value: Any, depth: int) -> None:
        nonlocal nodes
        nodes += 1
        if depth > MAX_SAVED_JSON_DEPTH or nodes > MAX_SAVED_JSON_NODES:
            raise ValueError("saved envelope exceeds nesting or node limit")
        if isinstance(value, BaseModel):
            value = value.__dict__  # bounded traversal; no full model_dump first
        if isinstance(value, (date, datetime)):
            value = value.isoformat()
        if isinstance(value, str):
            # UTF-8/JSON escaping cannot shorten a string. Reject large values
            # before allocating an encoded copy; any encoded scalar is bounded.
            if len(value) > remaining:
                raise ValueError("saved envelope exceeds 128 KiB")
            charge(len(json.dumps(value, ensure_ascii=False).encode("utf-8")))
        elif value is None or isinstance(value, (bool, int, float)):
            if isinstance(value, int) and value.bit_length() > 4 * remaining:
                raise ValueError("saved integer exceeds byte limit")
            charge(len(json.dumps(value, allow_nan=False)))
        elif isinstance(value, dict):
            if len(value) > MAX_SAVED_CONTAINER_ITEMS:
                raise ValueError("saved object exceeds item limit")
            charge(2 + max(0, len(value) - 1))
            for key, child in value.items():
                if not isinstance(key, str):
                    raise ValueError("saved object keys must be strings")
                visit(key, depth + 1)
                charge(1)
                visit(child, depth + 1)
        elif isinstance(value, list):
            if len(value) > MAX_SAVED_CONTAINER_ITEMS:
                raise ValueError("saved list exceeds item limit")
            charge(2 + max(0, len(value) - 1))
            for child in value:
                visit(child, depth + 1)
        else:
            raise ValueError("saved envelope must contain JSON values")

    visit({"zone_id": zone_id, "mode": "saved_inputs_j1", "saved_inputs": inputs}, 0)


class B6SavedPatternLibrary(PatternLibrary):
    """The original exchange model with bounds specific to synchronous saved scoring."""

    schema_version: Literal["1.0.0"] = "1.0.0"
    patterns: list[GradualPatternRecord] = Field(min_length=1, max_length=MAX_SAVED_RECORDS)


class B6SavedInputs(BaseModel):
    """No caller paths, acquisition options or mutable scoring thresholds.

    ``packet_sha256`` is an upstream N2 fingerprint, retained as provenance only.
    GeoAI does not receive that whole packet and cannot authenticate its tenant scope.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    schema_version: Literal["geoai-b6-saved-request/v1"]
    library_version: str = Field(min_length=1, max_length=128)
    algorithm: Algorithm
    expected_manifest_sha256: Sha256
    expected_json_sha256: Sha256
    expected_parquet_sha256: Sha256
    expected_library: B6SavedPatternLibrary
    forecast_json: str = Field(min_length=1, max_length=MAX_SAVED_ENVELOPE_BYTES)
    forecast_sha256: Sha256
    packet_sha256: Sha256


class B6HistoricalInputs(BaseModel):
    """Private fixed-case selector; no caller data, acquisition or scoring options."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    schema_version: Literal["geoai-b6-historical-request/v1"]
    case_id: Literal["q3-2023-fos-native-recurrence"]
    case_version: Literal["2026.10.27.1"]
    expected_case_manifest_sha256: Sha256
    packet_sha256: Sha256
