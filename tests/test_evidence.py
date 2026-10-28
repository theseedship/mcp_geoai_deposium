"""Evidence V1 contract and canonical-ID tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from src.evidence import (
    EvidenceContextV1,
    EvidenceSelection,
    GeoAIEvidenceV1,
    build_evidence_id,
    validate_mesh_projection_column,
)

FIXTURE = Path(__file__).parent / "fixtures/evidence-v1/native-spatial.json"


def test_native_fixture_is_byte_identical_to_python_contract() -> None:
    payload = json.loads(FIXTURE.read_text())
    context = EvidenceContextV1.model_validate(payload["context"])
    evidence = GeoAIEvidenceV1.model_validate(payload["expected_evidence"])

    assert context.model_dump(mode="json") == payload["context"]
    assert evidence.model_dump(mode="json") == payload["expected_evidence"]
    assert build_evidence_id(evidence) == payload["expected_evidence"]["evidence_id"]


def test_evidence_sequences_are_tuple_backed_and_json_arrays() -> None:
    payload = json.loads(FIXTURE.read_text())["expected_evidence"]
    evidence = GeoAIEvidenceV1.model_validate(payload)

    assert isinstance(evidence.selection.feature_ids, tuple)
    assert isinstance(evidence.selection.row_offsets, tuple)
    assert isinstance(evidence.selection.columns, tuple)
    assert isinstance(evidence.caveats, tuple)
    assert evidence.model_dump(mode="json") == payload


def test_evidence_sequences_cannot_mutate_or_desynchronize_canonical_id() -> None:
    payload = json.loads(FIXTURE.read_text())["expected_evidence"]
    evidence = GeoAIEvidenceV1.model_validate(payload)
    original_id = evidence.evidence_id

    with pytest.raises(AttributeError):
        evidence.selection.feature_ids.append("new_feature")
    with pytest.raises(TypeError):
        evidence.selection.row_offsets[0] = 1
    with pytest.raises(AttributeError):
        evidence.caveats.append("new_caveat")

    assert evidence.evidence_id == original_id
    assert build_evidence_id(evidence) == original_id
    assert evidence.model_dump(mode="json") == payload


def test_evidence_id_tampering_is_rejected() -> None:
    payload = json.loads(FIXTURE.read_text())["expected_evidence"]
    payload["selection"]["row_offsets"] = [0, 3]

    with pytest.raises(ValidationError, match="evidence_id"):
        GeoAIEvidenceV1.model_validate(payload)


def test_evidence_id_builder_hashes_normalized_payload() -> None:
    payload = json.loads(FIXTURE.read_text())["expected_evidence"]
    payload.pop("evidence_id")
    payload["selection"]["columns"] = [" geometry ", "zone_id", "name"]
    payload["selection"]["geometry_column"] = " geometry "
    payload["caveats"] = [
        " feature_ids_unavailable ",
        "row_offsets_are_physical_zero_based",
    ]

    evidence = GeoAIEvidenceV1.model_validate(
        {**payload, "evidence_id": build_evidence_id(payload)}
    )

    assert evidence.selection.columns[0] == "geometry"
    assert evidence.selection.geometry_column == "geometry"
    assert evidence.caveats[0] == "feature_ids_unavailable"


def test_evidence_id_builder_applies_payload_defaults() -> None:
    payload = json.loads(FIXTURE.read_text())["expected_evidence"]
    payload.pop("evidence_id")
    payload["truncated"] = False
    payload["caveats"] = []
    explicit_id = build_evidence_id(payload)
    payload.pop("derived")
    payload.pop("truncated")
    payload.pop("caveats")

    assert build_evidence_id(payload) == explicit_id


def test_evidence_models_are_immutable_and_strict() -> None:
    payload = json.loads(FIXTURE.read_text())
    selection = EvidenceSelection.model_validate(payload["expected_evidence"]["selection"])
    with pytest.raises(ValidationError):
        selection.columns = ["geometry"]
    with pytest.raises(ValidationError):
        EvidenceContextV1.model_validate({**payload["context"], "dataset_id": "/tmp/data"})
    with pytest.raises(ValidationError):
        EvidenceContextV1.model_validate({**payload["context"], "artifact_version": "sha256:ABC"})
    with pytest.raises(ValidationError):
        EvidenceContextV1.model_validate({**payload["context"], "crs": "EPSG:3857"})
    without_crs = dict(payload["context"])
    without_crs.pop("crs")
    with pytest.raises(ValidationError):
        EvidenceContextV1.model_validate(without_crs)
    with pytest.raises(ValidationError):
        EvidenceContextV1.model_validate(
            {**payload["context"], "geometry_column": "__geoai_source_row_offset"}
        )
    for geometry_column in ("source_uri", "internal_geometry", "__private"):
        with pytest.raises(ValidationError):
            EvidenceContextV1.model_validate(
                {**payload["context"], "geometry_column": geometry_column}
            )


@pytest.mark.parametrize("column", ["", "name/path", "name\nvalue"])
def test_evidence_selection_rejects_empty_path_or_control_columns(column: str) -> None:
    payload = json.loads(FIXTURE.read_text())["expected_evidence"]["selection"]
    payload["columns"] = [column]
    payload["geometry_column"] = column
    with pytest.raises(ValidationError):
        EvidenceSelection.model_validate(payload)


def test_evidence_selection_rejects_duckdb_virtual_filename_column() -> None:
    payload = json.loads(FIXTURE.read_text())["expected_evidence"]["selection"]
    payload["columns"] = ["geometry", "filename"]
    with pytest.raises(ValidationError):
        EvidenceSelection.model_validate(payload)

    payload["columns"] = ["geometry"]
    payload["geometry_column"] = "filename"
    with pytest.raises(ValidationError):
        EvidenceSelection.model_validate(payload)


@pytest.mark.parametrize(
    "column",
    [
        "FILE_ROW_NUMBER",
        "FileName",
        "__GEOAI_GEOMETRY",
        "__GeoAI_Row_Offset",
        "__GEOAI_SOURCE_ROW_OFFSET",
    ],
)
def test_mesh_projection_rejects_case_insensitive_reserved_spatial_names(column: str) -> None:
    with pytest.raises(ValueError, match="reserved spatial name"):
        validate_mesh_projection_column(column, field_name="column")


@pytest.mark.parametrize(
    "column",
    [
        "FILE_ROW_NUMBER",
        "FileName",
        "__GEOAI_GEOMETRY",
        "__GeoAI_Row_Offset",
        "__GEOAI_SOURCE_ROW_OFFSET",
    ],
)
def test_evidence_selection_rejects_case_insensitive_reserved_spatial_names(column: str) -> None:
    payload = json.loads(FIXTURE.read_text())["expected_evidence"]["selection"]
    payload["columns"] = ["geometry", column]
    with pytest.raises(ValidationError, match="bounded nonempty manifest column name"):
        EvidenceSelection.model_validate(payload)

    payload["columns"] = [column]
    payload["geometry_column"] = column
    with pytest.raises(ValidationError, match="bounded nonempty manifest column name"):
        EvidenceSelection.model_validate(payload)


def test_evidence_selection_accepts_non_sql_manifest_columns() -> None:
    selection = EvidenceSelection(
        feature_ids=[],
        row_offsets=[],
        columns=["  Parcel Name  "],
        geometry_column=" Parcel Name ",
    )
    assert selection.columns == ("Parcel Name",)
    assert selection.geometry_column == "Parcel Name"


@pytest.mark.parametrize(
    "column",
    [
        "source_id",
        "key_indicator",
        "Source_Id",
        "KEY_INDICATOR",
        "source_code",
        "uri_count",
    ],
)
def test_mesh_projection_accepts_benign_underscore_names(column: str) -> None:
    assert validate_mesh_projection_column(column, field_name="column") == column


@pytest.mark.parametrize(
    "column",
    [
        "source",
        "path",
        "uri",
        "url",
        "bucket",
        "key",
        "etag",
        "file_path",
        "source_uri",
        "object_key",
        "artifact_etag",
        "internal_value",
        "__private",
        "file_row_number",
        "filename",
    ],
)
def test_mesh_projection_rejects_storage_locators_and_internal_names(column: str) -> None:
    with pytest.raises(ValueError, match="storage-reference name|reserved spatial name"):
        validate_mesh_projection_column(column, field_name="column")


def test_evidence_rejects_unbounded_or_locator_like_selection_metadata() -> None:
    payload = json.loads(FIXTURE.read_text())["expected_evidence"]
    payload["selection"]["feature_ids"] = ["s3://bucket/private.parquet"]
    with pytest.raises(ValidationError):
        GeoAIEvidenceV1.model_validate(payload)

    payload = json.loads(FIXTURE.read_text())["expected_evidence"]
    payload["selection"]["row_offsets"] = [2, 0]
    with pytest.raises(ValidationError, match="strictly increasing"):
        GeoAIEvidenceV1.model_validate(payload)

    payload = json.loads(FIXTURE.read_text())["expected_evidence"]
    payload["selection"]["row_offsets"] = list(range(101))
    with pytest.raises(ValidationError):
        GeoAIEvidenceV1.model_validate(payload)

    payload = json.loads(FIXTURE.read_text())["expected_evidence"]
    payload["caveats"] = ["read from /private/path"]
    with pytest.raises(ValidationError):
        GeoAIEvidenceV1.model_validate(payload)


def test_dataset_describe_is_part_of_the_shared_operation_contract() -> None:
    payload = json.loads(FIXTURE.read_text())["expected_evidence"]
    payload["operation"] = "dataset.describe"
    payload["evidence_id"] = build_evidence_id(payload)
    assert GeoAIEvidenceV1.model_validate(payload).operation == "dataset.describe"
