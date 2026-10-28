"""Typed ``dataset-manifest/v1`` contract shared with Deposium integrations.

GeoAI owns the table and spatial semantics in this module. Server-owned identity,
artifact versioning, and provenance are supplied by the worker or gateway; GeoAI
never derives them from a caller-controlled path.
"""

from __future__ import annotations

import math
import re
from typing import Any, Literal

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, field_validator, model_validator

type ManifestFormat = Literal["geoparquet", "parquet"]
type DatasetKind = Literal["app_geoparquet", "science_dataset"]

_EXTENSION_KEY_RE = re.compile(r"^[a-z][a-z0-9_.-]*:[A-Za-z_][A-Za-z0-9_.-]*$")
_EXTENSION_KEY_JSON_PATTERN = r"^[a-z][a-z0-9_.-]*:[A-Za-z_][A-Za-z0-9_.-]*$"
_SEMANTIC_RULES: tuple[dict[str, str], ...] = (
    {
        "id": "table-column-names-unique",
        "assertion": "unique_by",
        "collection_pointer": "/table/columns",
        "field_pointer": "/name",
        "error": "manifest table columns must have unique names",
    },
    {
        "id": "geometry-column-exists",
        "assertion": "value_in_collection",
        "value_pointer": "/spatial/geometry_column",
        "collection_pointer": "/table/columns",
        "collection_field_pointer": "/name",
        "error": "manifest spatial.geometry_column must exist in table.columns",
    },
)


class _ManifestModel(BaseModel):
    """Freeze contract fields while allowing explicitly namespaced extensions."""

    model_config = ConfigDict(extra="allow", frozen=True, str_strip_whitespace=True)

    @model_validator(mode="after")
    def _extensions_are_namespaced(self) -> _ManifestModel:
        extras = self.__pydantic_extra__ or {}
        invalid = sorted(key for key in extras if not _EXTENSION_KEY_RE.fullmatch(key))
        if invalid:
            raise ValueError(
                "unknown manifest fields must be namespaced as 'namespace:field': "
                + ", ".join(invalid)
            )
        return self


class ManifestColumn(_ManifestModel):
    name: str = Field(..., min_length=1)
    type: str = Field(..., min_length=1)


class ManifestTable(_ManifestModel):
    row_count: int = Field(..., gt=0)
    columns: list[ManifestColumn] = Field(
        ...,
        min_length=1,
        json_schema_extra={"uniqueItems": True},
    )

    @model_validator(mode="after")
    def _column_names_are_unique(self) -> ManifestTable:
        names = [column.name for column in self.columns]
        if len(names) != len(set(names)):
            raise ValueError("manifest table columns must have unique names")
        return self


class ManifestSpatial(_ManifestModel):
    geometry_column: str = Field(..., min_length=1)
    encoding: Literal["WKB"]
    crs: str = Field(..., min_length=1)
    bbox: list[float] = Field(..., min_length=4, max_length=4)

    @field_validator("bbox")
    @classmethod
    def _bbox_is_finite_and_ordered(cls, value: list[float]) -> list[float]:
        bbox = [float(coordinate) for coordinate in value]
        if not all(math.isfinite(coordinate) for coordinate in bbox):
            raise ValueError("manifest bbox coordinates must be finite")
        min_x, min_y, max_x, max_y = bbox
        if min_x > max_x or min_y > max_y:
            raise ValueError("manifest bbox minima must not exceed maxima")
        return bbox


class ManifestArtifact(_ManifestModel):
    format: ManifestFormat
    key: str = Field(..., min_length=1)
    version: str = Field(..., min_length=1)


class ManifestProvenance(_ManifestModel):
    tenant_id: str = Field(..., min_length=1)
    folder_id: str = Field(..., min_length=1)
    file_id: int = Field(..., gt=0)
    document_id: str | None = Field(None, min_length=1)
    producer: str = Field(..., min_length=1)
    created_at: AwareDatetime


class DatasetManifestV1(_ManifestModel):
    schema_version: Literal["dataset-manifest/v1"]
    dataset_id: str = Field(..., min_length=1)
    dataset_kind: DatasetKind
    artifact: ManifestArtifact
    table: ManifestTable
    spatial: ManifestSpatial
    provenance: ManifestProvenance

    @model_validator(mode="after")
    def _spatial_and_artifact_semantics_are_consistent(self) -> DatasetManifestV1:
        column_names = {column.name for column in self.table.columns}
        if self.spatial.geometry_column not in column_names:
            raise ValueError("manifest spatial.geometry_column must exist in table.columns")
        if self.dataset_kind == "app_geoparquet" and self.artifact.format != "geoparquet":
            raise ValueError("app_geoparquet manifests require artifact.format='geoparquet'")
        return self


def _restrict_json_schema_extensions(value: Any) -> None:
    if isinstance(value, dict):
        if value.get("type") == "object" and value.get("additionalProperties") is True:
            value["additionalProperties"] = False
            value["patternProperties"] = {_EXTENSION_KEY_JSON_PATTERN: {}}
        for child in value.values():
            _restrict_json_schema_extensions(child)
    elif isinstance(value, list):
        for child in value:
            _restrict_json_schema_extensions(child)


def dataset_manifest_json_schema() -> dict[str, Any]:
    """Return the structural schema plus explicit portable semantic-rule metadata."""
    schema = DatasetManifestV1.model_json_schema(mode="serialization")
    _restrict_json_schema_extensions(schema)
    schema["$schema"] = "https://json-schema.org/draft/2020-12/schema"
    schema["$id"] = "https://deposium.ai/schemas/dataset-manifest/v1"
    schema["$comment"] = (
        "Structural JSON Schema validation is necessary but not sufficient. "
        "Consumers MUST also enforce x-deposium-semantic-rules or validate "
        "against the runtime model."
    )
    schema["x-deposium-semantic-rules"] = dataset_manifest_semantic_rules()
    schema["allOf"] = [
        {
            "if": {
                "properties": {"dataset_kind": {"const": "app_geoparquet"}},
                "required": ["dataset_kind"],
            },
            "then": {
                "properties": {
                    "artifact": {
                        "properties": {"format": {"const": "geoparquet"}},
                        "required": ["format"],
                    }
                }
            },
        }
    ]
    return schema


def dataset_manifest_semantic_rules() -> dict[str, Any]:
    """Return cross-field rules that standard JSON Schema cannot express."""
    return {
        "rules_version": "dataset-manifest-semantic-rules/v1",
        "rules": [dict(rule) for rule in _SEMANTIC_RULES],
    }


def dataset_manifest_contract() -> dict[str, Any]:
    """Return the normative portable contract consumed outside Python."""
    semantic_rules = dataset_manifest_semantic_rules()
    return {
        "contract_version": "dataset-manifest-contract/v1",
        "structural_schema": dataset_manifest_json_schema(),
        "semantic_rules": semantic_rules,
        "validation_order": ["structural_schema", "semantic_rules"],
        "conformance_fixtures": {
            "repository_path": "tests/fixtures/gai3",
            "positive": ["formal-wkb.expected.json", "internal-wkb.expected.json"],
            "negative": "invalid-manifest-cases.json",
        },
    }
