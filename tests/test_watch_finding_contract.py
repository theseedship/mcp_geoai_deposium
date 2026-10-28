"""Exercise the published manual finding boundary with synthetic data only."""

import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator, FormatChecker, ValidationError

SCHEMA_PATH = (
    Path(__file__).resolve().parents[1] / "src/contracts_data/watch/watch-finding.v2.schema.json"
)


@pytest.fixture
def validator():
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema, format_checker=FormatChecker())


@pytest.fixture
def manual_finding():
    return {
        "schema_version": "watch-finding/v2",
        "registre": "raw_forecast_classification",
        "rule": {"id": "out_of_normal", "version": "v1", "predicate": "watch_level != none"},
        "place": {"ref": "zone:synthetic"},
        "day": "2026-10-02",
        "day_grain": "utc_day",
        "variable": "wind_speed",
        "provider_variable": "wind_speed_10m_max",
        "value": 15,
        "p90": 12,
        "percentile_band": ">p90",
        "category": "extreme_high",
        "watch_level": "notify",
        "confidence": None,
        "finding_key": "synthetic-finding",
        "input_hash": "sha256:" + "a" * 64,
        "retrieved_at": "2026-10-02T10:00:00Z",
        "model_run_at": None,
        "horizon_days_served": 1,
        "revisions": [
            {
                "input_hash": "sha256:" + "a" * 64,
                "retrieved_at": "2026-10-02T10:00:00Z",
                "value": 15,
                "watch_level": "notify",
            }
        ],
        "delivery": {
            "chat": {"state": "pending", "at": None, "ref": None},
            "mail": {"state": "pending", "at": None, "ref": None},
        },
        "context": {"kind": "manual", "delivery_authorized": False},
    }


def test_manual_finding_is_valid(validator, manual_finding):
    validator.validate(manual_finding)


@pytest.mark.parametrize("container", ["root", "context"])
def test_manual_finding_rejects_consent_injection(validator, manual_finding, container):
    target = manual_finding if container == "root" else manual_finding["context"]
    target["consent"] = {"status": "granted"}
    with pytest.raises(ValidationError, match="Additional properties"):
        validator.validate(manual_finding)


@pytest.mark.parametrize(
    "context",
    [
        {"kind": "manual", "delivery_authorized": True},
        {"kind": "scheduled", "delivery_authorized": False},
        {"kind": "manual"},
        {"delivery_authorized": False},
        {"kind": "manual", "delivery_authorized": False, "recipient": "test@example.invalid"},
    ],
)
def test_manual_finding_rejects_other_authority(validator, manual_finding, context):
    manual_finding["context"] = context
    with pytest.raises(ValidationError):
        validator.validate(manual_finding)


def test_manual_context_is_required(validator, manual_finding):
    del manual_finding["context"]
    with pytest.raises(ValidationError, match="context"):
        validator.validate(manual_finding)


def test_manual_finding_cannot_claim_v1(validator, manual_finding):
    manual_finding["schema_version"] = "watch-finding/v1"
    with pytest.raises(ValidationError, match="watch-finding/v2"):
        validator.validate(manual_finding)


@pytest.mark.parametrize(
    ("field", "invalid"),
    [("day", "2026-02-30")],
)
def test_manual_finding_checks_time_formats(validator, manual_finding, field, invalid):
    manual_finding[field] = invalid
    with pytest.raises(ValidationError):
        validator.validate(manual_finding)
