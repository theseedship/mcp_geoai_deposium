"""Strict immutable GeoAI native evidence contracts."""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator, model_validator

_SAFE_LOGICAL_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,254}$")
_SAFE_SQL_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_SHA256_RE = r"^sha256:[0-9a-f]{64}$"
MESH_RESERVED_COLUMNS = frozenset(
    {
        "file_row_number",
        "filename",
        "__geoai_geometry",
        "__geoai_row_offset",
        "__geoai_source_row_offset",
    }
)
_MESH_RESERVED_COLUMNS_CASEFOLDED = frozenset(column.casefold() for column in MESH_RESERVED_COLUMNS)
MESH_STORAGE_COLUMN_NAMES = frozenset({"source", "path", "uri", "url", "bucket", "key", "etag"})
MESH_STORAGE_COLUMN_SUFFIXES = ("_path", "_uri", "_url", "_bucket", "_key", "_etag")


def _validate_safe_logical_id(value: str, *, field_name: str) -> str:
    if (
        not _SAFE_LOGICAL_ID_RE.fullmatch(value)
        or ".." in value
        or "://" in value
        or "/" in value
        or "\\" in value
    ):
        raise ValueError(f"{field_name} must be a safe logical ID, not a path or URL")
    return value


def validate_sql_identifier(value: str, *, field_name: str) -> str:
    if not _SAFE_SQL_IDENTIFIER_RE.fullmatch(value):
        raise ValueError(f"{field_name} must be a SQL-safe identifier")
    if value.casefold() in _MESH_RESERVED_COLUMNS_CASEFOLDED:
        raise ValueError(f"{field_name} uses a reserved spatial name")
    return value


def validate_mesh_projection_column(value: str, *, field_name: str) -> str:
    """Reject SQL-safe columns whose names advertise server-owned storage identity."""
    validated = validate_sql_identifier(value, field_name=field_name)
    normalized = validated.lower()
    if (
        validated.startswith("__")
        or normalized.startswith("internal_")
        or normalized in MESH_STORAGE_COLUMN_NAMES
        or normalized.endswith(MESH_STORAGE_COLUMN_SUFFIXES)
    ):
        raise ValueError(f"{field_name} uses a storage-reference name")
    return validated


def _validate_manifest_column(value: str, *, field_name: str) -> str:
    """Normalize a manifest column name without imposing query SQL syntax on it."""
    trimmed = value.strip()
    if (
        not trimmed
        or len(trimmed) > 255
        or "/" in trimmed
        or "\\" in trimmed
        or "://" in trimmed
        or trimmed.casefold() in _MESH_RESERVED_COLUMNS_CASEFOLDED
        or any(ord(char) < 32 or ord(char) == 127 for char in trimmed)
    ):
        raise ValueError(f"{field_name} must be a bounded nonempty manifest column name")
    return trimmed


def _validate_opaque_identifier(value: str, *, field_name: str) -> str:
    trimmed = value.strip()
    if (
        not trimmed
        or len(trimmed) > 255
        or "/" in trimmed
        or "\\" in trimmed
        or "://" in trimmed
        or ".." in trimmed
        or any(ord(char) < 32 or ord(char) == 127 for char in trimmed)
    ):
        raise ValueError(f"{field_name} must be an opaque identifier, not a path or URL")
    return trimmed


def _validate_caveat(value: str) -> str:
    trimmed = value.strip()
    if (
        not trimmed
        or len(trimmed) > 500
        or "/" in trimmed
        or "\\" in trimmed
        or "://" in trimmed
        or ".." in trimmed
        or trimmed.startswith("~")
        or re.match(r"^[A-Za-z]:[\\/]", trimmed)
        or any(ord(char) < 32 or ord(char) == 127 for char in trimmed)
    ):
        raise ValueError("caveats must be bounded logical labels, not storage paths or URLs")
    return trimmed


def _validate_unique_strings(value: tuple[str, ...], *, field_name: str) -> tuple[str, ...]:
    if len(set(value)) != len(value):
        raise ValueError(f"{field_name} must contain unique values")
    if any(not item for item in value):
        raise ValueError(f"{field_name} must not contain empty values")
    return value


class _EvidenceBase(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class EvidenceContextV1(_EvidenceBase):
    schema_version: Literal["geoai-evidence-context/v1"] = "geoai-evidence-context/v1"
    dataset_id: str = Field(..., min_length=1, max_length=255)
    artifact_version: str = Field(..., pattern=_SHA256_RE)
    scope_fingerprint: str = Field(..., pattern=_SHA256_RE)
    query_fingerprint: str = Field(..., pattern=_SHA256_RE)
    geometry_column: str = Field(..., min_length=1, max_length=255)
    crs: Literal["EPSG:4326"]

    @field_validator("dataset_id")
    @classmethod
    def _dataset_id_is_logical(cls, value: str) -> str:
        return _validate_safe_logical_id(value, field_name="dataset_id")

    @field_validator("geometry_column")
    @classmethod
    def _geometry_column_is_safe(cls, value: str) -> str:
        return validate_mesh_projection_column(value, field_name="geometry_column")


class EvidenceSelection(_EvidenceBase):
    feature_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=100)
    row_offsets: tuple[int, ...] = Field(default_factory=tuple, max_length=100)
    columns: tuple[str, ...] = Field(..., min_length=1, max_length=256)
    geometry_column: str = Field(..., min_length=1, max_length=256)

    @field_validator("feature_ids", "row_offsets", "columns", mode="before")
    @classmethod
    def _json_lists_are_tuple_backed(cls, value: Any) -> Any:
        return tuple(value) if isinstance(value, list) else value

    @field_validator("feature_ids", "columns")
    @classmethod
    def _string_lists_are_unique(
        cls, value: tuple[str, ...], info: ValidationInfo
    ) -> tuple[str, ...]:
        field_name = info.field_name or "selection"
        if info.field_name == "columns":
            value = tuple(_validate_manifest_column(item, field_name=field_name) for item in value)
        else:
            value = tuple(
                _validate_opaque_identifier(item, field_name=field_name) for item in value
            )
        return _validate_unique_strings(value, field_name=field_name)

    @field_validator("row_offsets")
    @classmethod
    def _row_offsets_are_unique_and_nonnegative(cls, value: tuple[int, ...]) -> tuple[int, ...]:
        if len(set(value)) != len(value):
            raise ValueError("row_offsets must contain unique values")
        if any(offset < 0 for offset in value):
            raise ValueError("row_offsets must be nonnegative")
        if any(left >= right for left, right in zip(value, value[1:], strict=False)):
            raise ValueError("row_offsets must be strictly increasing")
        return value

    @field_validator("geometry_column")
    @classmethod
    def _geometry_column_is_bounded(cls, value: str, info: ValidationInfo) -> str:
        return _validate_manifest_column(value, field_name=info.field_name or "geometry_column")

    @model_validator(mode="after")
    def _geometry_column_is_selected(self) -> EvidenceSelection:
        if self.geometry_column not in self.columns:
            raise ValueError("columns must include geometry_column")
        return self


class _GeoAIEvidencePayload(_EvidenceBase):
    schema_version: Literal["geoai-evidence/v1"] = "geoai-evidence/v1"
    scope_fingerprint: str = Field(..., pattern=_SHA256_RE)
    dataset_id: str = Field(..., min_length=1, max_length=255)
    artifact_version: str = Field(..., pattern=_SHA256_RE)
    operation: Literal[
        "dataset.describe",
        "spatial.intersects",
        "spatial.within",
        "spatial.contains",
        "spatial.distance",
    ]
    query_fingerprint: str = Field(..., pattern=_SHA256_RE)
    selection: EvidenceSelection
    derived: Literal[False] = False
    truncated: bool = False
    caveats: tuple[str, ...] = Field(default_factory=tuple, max_length=16)

    @field_validator("dataset_id")
    @classmethod
    def _dataset_id_is_logical(cls, value: str) -> str:
        return _validate_safe_logical_id(value, field_name="dataset_id")

    @field_validator("caveats", mode="before")
    @classmethod
    def _json_lists_are_tuple_backed(cls, value: Any) -> Any:
        return tuple(value) if isinstance(value, list) else value

    @field_validator("caveats")
    @classmethod
    def _caveats_are_unique(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        value = tuple(_validate_caveat(item) for item in value)
        return _validate_unique_strings(value, field_name="caveats")


class GeoAIEvidenceV1(_GeoAIEvidencePayload):
    evidence_id: str = Field(..., pattern=r"^geoai_ev_[0-9a-f]{64}$")

    @model_validator(mode="after")
    def _evidence_id_matches_payload(self) -> GeoAIEvidenceV1:
        expected = _compute_evidence_id(self.model_dump(mode="json", exclude={"evidence_id"}))
        if self.evidence_id != expected:
            raise ValueError("evidence_id does not match the canonical evidence payload")
        return self


def build_evidence_id(payload: GeoAIEvidenceV1 | dict[str, Any]) -> str:
    """Validate, normalize, and hash an evidence payload without ``evidence_id``."""
    if isinstance(payload, BaseModel):
        value = payload.model_dump(mode="json", exclude={"evidence_id"})
    else:
        value = dict(payload)
        value.pop("evidence_id", None)
    normalized = _GeoAIEvidencePayload.model_validate(value).model_dump(mode="json")
    return _compute_evidence_id(normalized)


def _compute_evidence_id(value: dict[str, Any]) -> str:
    """Hash an already-normalized payload without invoking model validation recursively."""
    canonical = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return "geoai_ev_" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()
