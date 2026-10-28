import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator, FormatChecker, ValidationError

FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "rw2"


def _load(name: str) -> dict:
    return json.loads((FIXTURE_ROOT / name).read_text())


def _contract() -> dict:
    return _load("contract.json")


def _schemas() -> dict[str, dict]:
    return {name: _load(path) for name, path in _contract()["schemas"].items()}


def _validator(schema: dict) -> Draft202012Validator:
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema, format_checker=FormatChecker())


def test_rw2_contract_versions_and_state_machine_are_frozen():
    contract = _contract()

    assert contract["contract_version"] == "geoai-rw2-contract/v1"
    assert contract["transport_schema_version"] == "geoai-rw2/v1"
    assert contract["successful_progression"] == [
        "queued",
        "validating",
        "validated",
        "manifest_published",
        "catalog_pending",
        "registered",
    ]
    assert contract["terminal_states"] == ["registered", "failed", "cancelled"]
    assert "catalog acknowledgement" in contract["registered_definition"]


def test_all_rw2_json_schemas_are_valid_and_have_stable_ids():
    for name, schema in _schemas().items():
        Draft202012Validator.check_schema(schema)
        assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
        assert schema["$id"].endswith("/v1"), name


def test_positive_rw2_corpus_validates_against_named_schemas():
    schemas = _schemas()
    corpus = _load("positive-cases.json")

    assert corpus["fixture_version"] == "geoai-rw2-positive-cases/v1"
    for case in corpus["cases"]:
        _validator(schemas[case["schema"]]).validate(case["payload"])


def test_public_locator_contract_has_no_physical_or_scope_overrides():
    schema_text = json.dumps(_schemas()["dataset_locator"])
    forbidden = {
        "tenant_id",
        "space_id",
        "bucket",
        "key",
        "path",
        "uri",
        "url",
        "output_key",
        "output_path",
    }

    for field in forbidden:
        assert f'"{field}"' not in schema_text


def test_internal_source_and_manifest_keys_are_relative_and_non_traversing():
    cases = _load("positive-cases.json")["cases"]
    request = next(case["payload"] for case in cases if case["id"] == "registration-request")
    ack = next(case["payload"] for case in cases if case["id"] == "catalog-ack-request")

    for key in [request["source"]["key"], ack["manifest_key"]]:
        assert not key.startswith("/")
        assert "://" not in key
        assert ".." not in key.split("/")


def test_internal_registration_source_requires_a_strong_etag():
    validator = _validator(_schemas()["registration_request"])
    cases = _load("positive-cases.json")["cases"]
    request = next(case["payload"] for case in cases if case["id"] == "registration-request")

    for invalid_etag in [None, "", '"', '""', '"""', 'W/"weak-etag"']:
        invalid = json.loads(json.dumps(request))
        if invalid_etag is None:
            invalid["source"].pop("etag")
        else:
            invalid["source"]["etag"] = invalid_etag
        with pytest.raises(ValidationError):
            validator.validate(invalid)


def test_authorization_failures_do_not_disclose_foreign_dataset_existence():
    corpus = _load("cross-tenant-negative-cases.json")

    assert corpus["fixture_version"] == "geoai-rw2-cross-tenant-negative-cases/v1"
    authorization_cases = [case for case in corpus["cases"] if case["stage"] == "authorization"]
    assert authorization_cases
    for case in authorization_cases:
        assert case["expected"] == {
            "code": "DATASET_NOT_FOUND",
            "http_status": 404,
            "disclose_existence": False,
            "upstream_calls": 0,
        }


@pytest.mark.parametrize("case", _load("cross-tenant-negative-cases.json")["cases"])
def test_negative_locator_corpus_matches_schema_stage(case):
    validator = _validator(_schemas()["dataset_locator"])

    if case["stage"] == "schema":
        with pytest.raises(ValidationError):
            validator.validate(case["locator"])
    else:
        validator.validate(case["locator"])


def test_registered_status_requires_digests_and_is_not_retryable():
    validator = _validator(_schemas()["registration_status"])
    cases = _load("positive-cases.json")["cases"]
    registered = next(case["payload"] for case in cases if case["id"] == "registered-status")

    invalid = {**registered, "artifact_version": None, "retryable": True}
    with pytest.raises(ValidationError):
        validator.validate(invalid)


def test_internal_key_schemas_reject_absolute_uri_and_traversing_shapes():
    """The SCHEMAS (not just the positive fixture) must reject unsafe keys:
    absolute paths, URI schemes, and `..` traversal segments — a consumer
    vendoring the frozen schema is the boundary this contract protects."""
    schemas = _schemas()
    cases = _load("positive-cases.json")["cases"]
    request = next(case["payload"] for case in cases if case["id"] == "registration-request")
    ack = next(case["payload"] for case in cases if case["id"] == "catalog-ack-request")

    unsafe_keys = [
        "/absolute/path.parquet",
        "../escape.parquet",
        "tenant-alpha/../tenant-bravo/private.parquet",
        "tenant-alpha/481/..",
        "s3://deposium/tenant-alpha/x.parquet",
        "https://attacker.example/file.parquet",
        "tenant alpha/space.parquet",
        "tenant-alpha/./source.parquet",
        "tenant-alpha//source.parquet",
        "tenant-alpha/source.parquet/",
        ".",
    ]
    for unsafe in unsafe_keys:
        bad_request = json.loads(json.dumps(request))
        bad_request["source"]["key"] = unsafe
        with pytest.raises(ValidationError):
            _validator(schemas["registration_request"]).validate(bad_request)

        bad_ack = {**ack, "manifest_key": unsafe}
        with pytest.raises(ValidationError):
            _validator(schemas["catalog_ack_request"]).validate(bad_ack)

    # Dotted-but-safe segments stay valid (only `..` path segments are banned).
    ok_request = json.loads(json.dumps(request))
    ok_request["source"]["key"] = "tenant-alpha/481/geo.spatial/source..v2.parquet"
    _validator(schemas["registration_request"]).validate(ok_request)


def test_status_digests_follow_the_state_machine():
    """Digests appear exactly where the state machine produces them: none
    before validation, no manifest digest before publication, both mandatory
    once the manifest is published (the lost-ACK resume tuple needs them)."""
    validator = _validator(_schemas()["registration_status"])
    cases = _load("positive-cases.json")["cases"]
    validating = next(case["payload"] for case in cases if case["id"] == "validating-status")
    pending = next(case["payload"] for case in cases if case["id"] == "catalog-pending-status")

    # Pre-validation states must not carry digests.
    with pytest.raises(ValidationError):
        validator.validate({**validating, "artifact_version": pending["artifact_version"]})
    with pytest.raises(ValidationError):
        validator.validate(
            {**validating, "status": "queued", "manifest_sha256": pending["manifest_sha256"]}
        )

    # The manifest digest cannot exist before publication.
    with pytest.raises(ValidationError):
        validator.validate(
            {
                **validating,
                "status": "validated",
                "artifact_version": pending["artifact_version"],
                "manifest_sha256": pending["manifest_sha256"],
            }
        )
    # validated with the artifact digest only is legal.
    validator.validate(
        {**validating, "status": "validated", "artifact_version": pending["artifact_version"]}
    )

    # Post-publication states must carry BOTH digests.
    for status in ["manifest_published", "catalog_pending"]:
        with pytest.raises(ValidationError):
            validator.validate({**pending, "status": status, "artifact_version": None})
        with pytest.raises(ValidationError):
            validator.validate({**pending, "status": status, "manifest_sha256": None})
        validator.validate({**pending, "status": status})


def test_failed_status_retry_flags_must_agree():
    """Top-level `retryable` and `error.retryable` must never disagree on a
    failed status — one consumer must not retry while another stops."""
    validator = _validator(_schemas()["registration_status"])
    cases = _load("positive-cases.json")["cases"]
    validating = next(case["payload"] for case in cases if case["id"] == "validating-status")

    def failed(top: bool, inner: bool) -> dict:
        return {
            **validating,
            "status": "failed",
            "retryable": top,
            "error": {
                "code": "VALIDATION_ERROR",
                "stage": "validating",
                "message": "bounded byte validation failed",
                "retryable": inner,
            },
        }

    validator.validate(failed(True, True))
    validator.validate(failed(False, False))
    with pytest.raises(ValidationError):
        validator.validate(failed(False, True))
    with pytest.raises(ValidationError):
        validator.validate(failed(True, False))
