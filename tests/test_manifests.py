import copy
import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator, ValidationError
from pydantic import ValidationError as PydanticValidationError

from src.manifests import (
    DatasetManifestV1,
    dataset_manifest_contract,
    dataset_manifest_json_schema,
    dataset_manifest_semantic_rules,
)

FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "gai3"


def _manifest_payload(stem: str = "formal-wkb") -> dict:
    fixture = json.loads((FIXTURE_ROOT / f"{stem}.expected.json").read_text())
    return fixture["manifest"]


def _resolve_path(payload: dict, dotted_path: str) -> tuple[dict | list, str | int]:
    parts = dotted_path.split(".")
    target: dict | list = payload
    for part in parts[:-1]:
        target = target[int(part)] if isinstance(target, list) else target[part]
    final: str | int = int(parts[-1]) if isinstance(target, list) else parts[-1]
    return target, final


def _apply_invalid_case(payload: dict, case: dict) -> dict:
    mutated = copy.deepcopy(payload)
    for dotted_path in case.get("remove", []):
        target, key = _resolve_path(mutated, dotted_path)
        del target[key]
    for dotted_path, value in case.get("set", {}).items():
        target, key = _resolve_path(mutated, dotted_path)
        target[key] = value
    for dotted_path, value in case.get("append", {}).items():
        target, key = _resolve_path(mutated, dotted_path)
        target[key].append(value)
    return mutated


@pytest.mark.parametrize("stem", ["formal-wkb", "internal-wkb"])
def test_dataset_manifest_v1_round_trips_portable_fixture(stem):
    payload = _manifest_payload(stem)

    manifest = DatasetManifestV1.model_validate(payload)

    assert manifest.model_dump(mode="json") == payload


@pytest.mark.parametrize("stem", ["formal-wkb", "internal-wkb"])
def test_dataset_manifest_schema_is_valid_and_accepts_fixture(stem):
    schema = dataset_manifest_json_schema()
    Draft202012Validator.check_schema(schema)

    Draft202012Validator(schema).validate(_manifest_payload(stem))

    assert schema["$id"] == "https://deposium.ai/schemas/dataset-manifest/v1"
    assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
    assert schema["x-deposium-semantic-rules"] == dataset_manifest_semantic_rules()


def test_dataset_manifest_contract_publishes_both_validation_layers():
    contract = dataset_manifest_contract()

    assert contract["contract_version"] == "dataset-manifest-contract/v1"
    assert contract["structural_schema"] == dataset_manifest_json_schema()
    assert contract["semantic_rules"] == dataset_manifest_semantic_rules()
    assert contract["validation_order"] == ["structural_schema", "semantic_rules"]
    assert contract["conformance_fixtures"]["negative"] == "invalid-manifest-cases.json"


def test_namespaced_manifest_extensions_are_forward_compatible():
    payload = _manifest_payload()
    payload["geoai:inspection_version"] = "1"
    payload["spatial"]["geoai:geometry_types"] = ["Point"]

    manifest = DatasetManifestV1.model_validate(payload)
    Draft202012Validator(dataset_manifest_json_schema()).validate(payload)

    assert manifest.model_dump(mode="json")["geoai:inspection_version"] == "1"


def test_unnamespaced_unknown_manifest_fields_fail_runtime_and_schema():
    payload = _manifest_payload()
    payload["dataset_typo"] = "silent-drift"

    with pytest.raises(PydanticValidationError, match="namespace:field"):
        DatasetManifestV1.model_validate(payload)
    with pytest.raises(ValidationError):
        Draft202012Validator(dataset_manifest_json_schema()).validate(payload)


@pytest.mark.parametrize(
    "field_path",
    [("schema_version",), ("spatial", "encoding")],
)
def test_required_discriminator_fields_cannot_be_omitted(field_path):
    payload = _manifest_payload()
    target = payload
    for key in field_path[:-1]:
        target = target[key]
    del target[field_path[-1]]

    with pytest.raises(PydanticValidationError, match="Field required"):
        DatasetManifestV1.model_validate(payload)
    with pytest.raises(ValidationError):
        Draft202012Validator(dataset_manifest_json_schema()).validate(payload)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda payload: payload["table"]["columns"].append(
                copy.deepcopy(payload["table"]["columns"][0])
            ),
            "unique names",
        ),
        (
            lambda payload: payload["spatial"].update({"bbox": [2.0, 0.0, 1.0, 3.0]}),
            "minima",
        ),
        (
            lambda payload: payload["provenance"].update({"created_at": "2026-07-11T00:00:00"}),
            "timezone",
        ),
        (
            lambda payload: payload["spatial"].update({"geometry_column": "missing"}),
            "must exist",
        ),
        (
            lambda payload: payload["artifact"].update({"format": "parquet"}),
            "app_geoparquet",
        ),
        (
            lambda payload: payload["table"].update({"row_count": 0}),
            "greater than 0",
        ),
    ],
)
def test_manifest_semantic_invariants_are_enforced(mutate, message):
    payload = _manifest_payload()
    mutate(payload)

    with pytest.raises(PydanticValidationError, match=message):
        DatasetManifestV1.model_validate(payload)


def test_schema_rejects_app_geoparquet_with_plain_parquet_format():
    payload = _manifest_payload()
    payload["artifact"]["format"] = "parquet"

    with pytest.raises(ValidationError):
        Draft202012Validator(dataset_manifest_json_schema()).validate(payload)


def test_portable_invalid_manifest_corpus_matches_runtime_and_json_schema():
    corpus = json.loads((FIXTURE_ROOT / "invalid-manifest-cases.json").read_text())
    base = _manifest_payload()
    validator = Draft202012Validator(dataset_manifest_json_schema())

    assert corpus["fixture_version"] == "gai3-invalid-manifest-cases/v1"
    assert corpus["base_fixture"] == "formal-wkb.expected.json"
    for case in corpus["cases"]:
        payload = _apply_invalid_case(base, case)
        with pytest.raises(PydanticValidationError, match=case["semantic_error_contains"]):
            DatasetManifestV1.model_validate(payload)
        assert validator.is_valid(payload) is case["schema_valid"], case["id"]
