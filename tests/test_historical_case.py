"""Real pinned archive roundtrips and explicitly synthetic forecast refusals."""

import copy
import hashlib
import json
import shutil
from pathlib import Path

import pytest

from src.patterns import historical_case as case

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "examples/so4gp_parquet/geoai-o3-q3-2023-handoff"


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def encode(value):
    return (json.dumps(value, sort_keys=True, indent=2) + "\n").encode()


@pytest.fixture(scope="module")
def registry_bytes():
    # Historical source is immutable. Never substitute the current registry.
    return (
        ROOT / "tests/fixtures/pattern-qualification-2026-10-16/zones-q3-2023.geojson"
    ).read_bytes()


@pytest.fixture
def registry(tmp_path, registry_bytes):
    path = tmp_path / "zones.geojson"
    path.write_bytes(registry_bytes)
    return path


@pytest.fixture
def exported(tmp_path, registry):
    output = tmp_path / "package"
    receipt = case.export_historical_case(BUNDLE, registry, output)
    return output, receipt["manifest_sha256"]


def forecast():
    return {
        "zone_id": "fos_sur_mer",
        "forecast": [
            {
                "date": f"2023-07-{day:02d}",
                "temperature_2m": 20.0 + day,
                "humidity": 70.0 - day,
                "evapotranspiration": 1.0 + day,
            }
            for day in range(2, 7)
        ],
        "metadata": {
            "synthetic": True,
            "source": "synthetic-technical-fixture",
            "model_issued_at": "2023-07-01T18:00:00Z",
            "issued_at": "2026-10-21T00:00:00Z",
            "lat": 43.445,
            "lon": 4.92,
            "horizon_days": 5,
            "registry_sha256": case.ZONES_SHA256,
        },
    }


def preflight(exported, value):
    raw = encode(value)
    return case.preflight_historical_forecast(*exported, raw, digest(raw))


def repin(package):
    manifest_path = package / "manifest.json"
    manifest = json.loads(manifest_path.read_bytes())
    for name in manifest["files"]:
        raw = (package / name).read_bytes()
        manifest["files"][name] = {"sha256": digest(raw), "bytes": len(raw)}
    raw = encode(manifest)
    manifest_path.write_bytes(raw)
    return digest(raw)


def test_roundtrip_preserves_native_records_sources_and_scope(exported, registry_bytes):
    package, pin = exported
    result = case.verify_historical_case(package, pin)
    records = result["library"]["patterns"]
    native = json.loads(
        (
            ROOT / "tests/fixtures/pattern-qualification-2026-10-16/q3-native-readback.json"
        ).read_bytes()
    )["libraries"]["tgrad"]["records"]
    # Native timestamps normalize to Z in exchange JSON; compare model values.
    from src.patterns.library_schema import GradualPatternRecord

    assert [GradualPatternRecord.model_validate(r) for r in records] == [
        GradualPatternRecord.model_validate(r) for r in native
    ]
    assert len(records) == 149
    assert all(r["zone_id"] == "cross_zone" and r["confidence"] is None for r in records)
    assert (package / "source/tgrad.parquet").read_bytes() == (
        BUNDLE / "objects" / case.PARQUET_KEY
    ).read_bytes()
    assert (package / "source/zones.geojson").read_bytes() == registry_bytes
    descriptor = result["descriptor"]
    assert descriptor["target_zone_id"] == "fos_sur_mer"
    assert descriptor["primary_rule_ids"] == [
        "cross_zone/tgrad/15",
        "cross_zone/tgrad/16",
        "cross_zone/tgrad/22",
    ]
    assert descriptor["shorter_lag_alternative_rule_ids"] == ["cross_zone/tgrad/23"]
    assert descriptor["bbox"] == [4.85, 43.4, 4.99, 43.49]
    assert descriptor["ready_for_scoring"] is descriptor["live_activation"] is False
    assert descriptor["scientific_adoption"] == "pending"
    assert len(result["provenance"]["b4_manifests"]) == 3


def test_export_is_deterministic_and_does_not_replace_existing(exported, registry, tmp_path):
    other = tmp_path / "other"
    receipt = case.export_historical_case(BUNDLE, registry, other)
    assert receipt["manifest_sha256"] == exported[1]
    assert {p.relative_to(other): p.read_bytes() for p in other.rglob("*") if p.is_file()} == {
        p.relative_to(exported[0]): p.read_bytes() for p in exported[0].rglob("*") if p.is_file()
    }
    sentinel = exported[0] / "sentinel"
    sentinel.write_bytes(b"preserved")
    with pytest.raises(ValueError, match="already exists"):
        case.export_historical_case(BUNDLE, registry, exported[0])
    assert sentinel.read_bytes() == b"preserved"


@pytest.mark.parametrize("source", ["registry", "bundle", "parquet", "b4"])
def test_bad_source_never_publishes(source, registry, tmp_path):
    bundle = tmp_path / "source"
    shutil.copytree(BUNDLE, bundle)
    if source == "registry":
        target = registry
    elif source == "bundle":
        target = bundle / "bundle.json"
    elif source == "parquet":
        target = bundle / "objects" / case.PARQUET_KEY
    else:
        target = next((bundle / "objects/climate/runs").glob("*/manifest.json"))
    target.write_bytes(target.read_bytes() + b" ")
    output = tmp_path / "refused"
    with pytest.raises(ValueError, match="SHA-256"):
        case.export_historical_case(bundle, registry, output)
    assert not output.exists()


@pytest.mark.parametrize(
    "mutation", ["target", "geometry", "period", "candidate", "adoption", "bool_type", "extra"]
)
def test_descriptor_cannot_be_promoted_by_repinning(exported, mutation):
    package, _ = exported
    path = package / "manifest.json"
    manifest = json.loads(path.read_bytes())
    descriptor = manifest["descriptor"]
    if mutation == "target":
        descriptor["target_zone_id"] = "bassin_du_lez"
    elif mutation == "geometry":
        descriptor["geometry"]["coordinates"][0][0][0] += 0.01
    elif mutation == "period":
        descriptor["period_end"] = "2026-10-21"
    elif mutation == "candidate":
        descriptor["primary_rule_ids"].append("cross_zone/tgrad/23")
    elif mutation == "adoption":
        descriptor["scientific_adoption"] = "accepted"
    elif mutation == "bool_type":
        descriptor["live_activation"] = 0
    else:
        descriptor["surprise"] = True
    raw = encode(manifest)
    path.write_bytes(raw)
    with pytest.raises(ValueError, match="descriptor"):
        case.verify_historical_case(package, digest(raw))


@pytest.mark.parametrize(
    "mutation", ["subset", "zone", "fold", "confidence", "bool_type", "id_type"]
)
def test_derived_library_cannot_replace_native_records(exported, mutation):
    package, _ = exported
    path = package / "library.json"
    library = json.loads(path.read_bytes())
    if mutation == "subset":
        library["patterns"] = library["patterns"][:3]
    elif mutation == "zone":
        library["zone_id"] = "fos_sur_mer"
    elif mutation == "fold":
        library["patterns"][0]["n_folds"] = 9
    elif mutation == "bool_type":
        library["patterns"][0]["is_temporal"] = 1
    elif mutation == "id_type":
        library["patterns"][0]["pattern_id"] = False
    else:
        library["patterns"][0]["confidence"] = 0.9
    path.write_bytes(encode(library))
    with pytest.raises(ValueError, match="native records"):
        case.verify_historical_case(package, repin(package))


@pytest.mark.parametrize(
    "mutation", ["extra_file", "extra_dir", "symlink", "missing", "traversal", "oversized", "pin"]
)
def test_closed_package_refusals(exported, mutation, tmp_path):
    package, pin = exported
    if mutation == "extra_file":
        (package / "unexpected").write_bytes(b"x")
    elif mutation == "extra_dir":
        (package / "unexpected").mkdir()
    elif mutation == "symlink":
        path = package / "library.json"
        raw = path.read_bytes()
        path.unlink()
        outside = tmp_path / "outside"
        outside.write_bytes(raw)
        path.symlink_to(outside)
    elif mutation == "missing":
        (package / "library.json").unlink()
    elif mutation == "traversal":
        path = package / "manifest.json"
        value = json.loads(path.read_bytes())
        value["files"]["../outside"] = value["files"].pop("library.json")
        raw = encode(value)
        path.write_bytes(raw)
        pin = digest(raw)
    elif mutation == "oversized":
        (package / "library.json").write_bytes(b" " * (case.MAX_JSON_BYTES + 1))
    else:
        pin = "0" * 64
    with pytest.raises(ValueError):
        case.verify_historical_case(package, pin)


def test_preflight_is_explicitly_synthetic_and_never_ready(exported):
    result = preflight(exported, forecast())
    assert result["compatible"]
    assert result["synthetic"]
    assert result["evidence_kind"] == "synthetic_technical_fixture"
    assert result["ready_for_scoring"] is result["live_activation"] is False
    assert result["scientific_adoption"] == "pending"
    assert result["model_issued_at"].startswith("2023-")
    assert result["retrieved_at"].startswith("2026-")


def test_default_full_shortlist_report_remains_exact_v1(exported):
    value = forecast()
    raw = encode(value)
    result = case.preflight_historical_forecast(*exported, raw, digest(raw))
    assert result == case.preflight_historical_forecast(
        *exported, raw, digest(raw), forecast_profile="full-shortlist"
    )
    assert result == {
        "schema_version": "geoai-b6-historical-forecast-preflight/v1",
        "manifest_sha256": exported[1],
        "forecast_sha256": digest(raw),
        "compatible": True,
        "target_zone_id": "fos_sur_mer",
        "producer_zone_id": "cross_zone",
        "horizon_days": 5,
        "model_issued_at": "2023-07-01T18:00:00Z",
        "forecast_origin_authentication": "not_established_by_metadata",
        "retrieved_at": "2026-10-21T00:00:00Z",
        "synthetic": True,
        "evidence_kind": "synthetic_technical_fixture",
        "ready_for_scoring": False,
        "scientific_adoption": "pending",
        "live_activation": False,
    }


def test_temperature_profile_is_explicit_compatibility_only_and_preserves_package(exported):
    value = forecast()
    for index, row in enumerate(value["forecast"]):
        row.pop("evapotranspiration")
        row["humidity"] = None
        if index >= 2:
            row["temperature_2m"] = None
    raw = encode(value)
    with pytest.raises(ValueError, match="two finite values"):
        case.preflight_historical_forecast(*exported, raw, digest(raw))
    package, pin = exported
    before = {p.relative_to(package): p.read_bytes() for p in package.rglob("*") if p.is_file()}
    result = case.preflight_historical_forecast(
        *exported, raw, digest(raw), forecast_profile="rule15-temperature"
    )
    assert result["schema_version"] == "geoai-b6-historical-forecast-preflight/v2"
    assert result["profile_id"] == "q3-2023-fos-rule15-temperature/v1"
    assert result["required_forecast_variables"] == ["temperature_2m"]
    assert result["primary_case_rule_ids"] == ["cross_zone/tgrad/15"]
    assert result["manifest_sha256"] == pin
    assert result["compatible"] is True
    assert result["ready_for_scoring"] is result["live_activation"] is False
    assert result["scientific_adoption"] == "pending"
    assert result["forecast_origin_authentication"] == "not_established_by_metadata"
    assert {
        p.relative_to(package): p.read_bytes() for p in package.rglob("*") if p.is_file()
    } == before
    verified = case.verify_historical_case(package, pin)
    assert len(verified["library"]["patterns"]) == 149
    assert verified["descriptor"]["primary_rule_ids"] == [
        "cross_zone/tgrad/15",
        "cross_zone/tgrad/16",
        "cross_zone/tgrad/22",
    ]


@pytest.mark.parametrize("finite_count", [0, 1])
def test_temperature_profile_requires_two_finite_temperature_values(exported, finite_count):
    value = forecast()
    for row in value["forecast"][finite_count:]:
        row.pop("temperature_2m")
    raw = encode(value)
    with pytest.raises(ValueError, match="two finite values"):
        case.preflight_historical_forecast(
            *exported, raw, digest(raw), forecast_profile="rule15-temperature"
        )


@pytest.mark.parametrize("driver", case.DRIVERS)
@pytest.mark.parametrize("invalid", [True, "70", float("nan"), float("inf"), 10**1000])
def test_temperature_profile_rejects_invalid_present_known_drivers(exported, driver, invalid):
    value = forecast()
    value["forecast"][0][driver] = invalid
    raw = encode(value)
    with pytest.raises(ValueError):
        case.preflight_historical_forecast(
            *exported, raw, digest(raw), forecast_profile="rule15-temperature"
        )


@pytest.mark.parametrize("profile", ["", "rule15", "full_shortlist", None, True])
def test_unknown_forecast_profile_is_refused(exported, profile):
    raw = encode(forecast())
    with pytest.raises(ValueError, match="unknown historical forecast profile"):
        case.preflight_historical_forecast(*exported, raw, digest(raw), forecast_profile=profile)


def test_daily_model_issuance_can_be_on_first_forecast_day(exported):
    value = forecast()
    value["metadata"]["model_issued_at"] = "2023-07-02T12:00:00Z"
    result = preflight(exported, value)
    assert result["compatible"]
    assert result["forecast_origin_authentication"] == "not_established_by_metadata"


def test_nonsynthetic_midday_issue_refused_for_full_daily_row(exported):
    value = forecast()
    value["metadata"].update(synthetic=False, model_issued_at="2023-07-02T00:00:01Z")
    with pytest.raises(ValueError, match="first full UTC forecast day"):
        preflight(exported, value)


@pytest.mark.parametrize("issued_at", ["2023-07-01T23:59:59Z", "2023-07-02T00:00:00Z"])
def test_nonsynthetic_declared_issue_precedes_full_daily_row(exported, issued_at):
    # Technical test of declarations only; no provider evidence is supplied.
    value = forecast()
    value["metadata"].update(synthetic=False, model_issued_at=issued_at)
    result = preflight(exported, value)
    assert result["compatible"]
    assert result["forecast_origin_authentication"] == "not_established_by_metadata"
    assert result["ready_for_scoring"] is False


@pytest.mark.parametrize(
    "issued_at",
    [
        "0001-01-01T00:00:00+00:01",
        "0001-01-01T00:00:00+23:59",
        "9999-12-31T23:59:59-00:01",
        "9999-12-31T23:59:59-23:59",
    ],
)
@pytest.mark.parametrize("synthetic", [False, True])
def test_utc_issuance_overflow_is_a_stable_core_and_cli_refusal(
    exported, issued_at, synthetic, tmp_path, capsys
):
    from src.cli import b6_historical_case as cli

    value = forecast()
    value["metadata"].update(synthetic=synthetic, model_issued_at=issued_at)
    with pytest.raises(ValueError, match="valid and representable in UTC"):
        preflight(exported, value)
    raw = encode(value)
    source = tmp_path / "boundary-forecast.json"
    source.write_bytes(raw)
    package, manifest_sha256 = exported
    assert (
        cli.main(
            [
                "preflight",
                str(package),
                "--expected-manifest-sha256",
                manifest_sha256,
                "--forecast-json",
                str(source),
                "--expected-forecast-sha256",
                digest(raw),
            ]
        )
        == 2
    )
    result = capsys.readouterr()
    assert result.out == ""
    assert result.err == (
        "Historical B6 preparation refused: historical model issuance must be valid "
        "and representable in UTC\n"
    )


@pytest.mark.parametrize(
    "mutation",
    [
        "target",
        "current_point",
        "missing_registry",
        "nan_coord",
        "bool_coord",
        "four_days",
        "horizon",
        "bool_horizon",
        "gap",
        "duplicate",
        "order",
        "invalid_date",
        "current_date",
        "missing_driver",
        "one_value",
        "string_value",
        "nan_driver",
        "bool_driver",
        "missing_issuance",
        "late_issuance",
        "naive_issuance",
        "invalid_issuance",
        "missing_source",
        "synthetic_marker",
    ],
)
@pytest.mark.parametrize("profile", ["full-shortlist", "rule15-temperature"])
def test_forecast_compatibility_refusals(exported, mutation, profile):
    value = copy.deepcopy(forecast())
    meta, rows = value["metadata"], value["forecast"]
    required_driver = "humidity" if profile == "full-shortlist" else "temperature_2m"
    if mutation == "target":
        value["zone_id"] = "cross_zone"
    elif mutation == "current_point":
        meta.update(lat=43.454, lon=4.901)
    elif mutation == "missing_registry":
        meta.pop("registry_sha256")
    elif mutation == "nan_coord":
        meta["lat"] = float("nan")
    elif mutation == "bool_coord":
        meta["lat"] = True
    elif mutation == "four_days":
        value["forecast"] = rows[:4]
        meta["horizon_days"] = 4
    elif mutation == "horizon":
        meta["horizon_days"] = 6
    elif mutation == "bool_horizon":
        meta["horizon_days"] = True
    elif mutation == "gap":
        rows[-1]["date"] = "2023-07-07"
    elif mutation == "duplicate":
        rows[-1]["date"] = rows[-2]["date"]
    elif mutation == "order":
        rows.reverse()
    elif mutation == "invalid_date":
        rows[-1]["date"] = "2023-07-32"
    elif mutation == "current_date":
        rows[-1]["date"] = "2026-10-21"
    elif mutation == "missing_driver":
        for row in rows:
            row.pop(required_driver)
    elif mutation == "one_value":
        for row in rows[1:]:
            row[required_driver] = None
    elif mutation in ("string_value", "nan_driver", "bool_driver"):
        rows[0]["humidity"] = {
            "string_value": "70",
            "nan_driver": float("nan"),
            "bool_driver": True,
        }[mutation]
    elif mutation == "missing_issuance":
        meta.pop("model_issued_at")
    elif mutation == "late_issuance":
        meta["model_issued_at"] = "2026-10-21T00:00:00Z"
    elif mutation == "naive_issuance":
        meta["model_issued_at"] = "2023-07-01T00:00:00"
    elif mutation == "invalid_issuance":
        meta["model_issued_at"] = "not a timestamp"
    elif mutation == "missing_source":
        meta.pop("source")
    else:
        meta["synthetic"] = "true"
    raw = encode(value)
    with pytest.raises(ValueError):
        case.preflight_historical_forecast(*exported, raw, digest(raw), forecast_profile=profile)


def test_forecast_digest_and_duplicate_keys_refused(exported):
    raw = encode(forecast())
    with pytest.raises(ValueError, match="SHA-256"):
        case.preflight_historical_forecast(*exported, raw, "0" * 64)
    raw = b'{"zone_id":"fos_sur_mer","zone_id":"fos_sur_mer"}'
    with pytest.raises(ValueError, match="dupliqu"):
        case.preflight_historical_forecast(*exported, raw, digest(raw))


def test_standalone_forecast_api_shares_saved_input_byte_bound(exported):
    raw = b" " * (case.MAX_SAVED_ENVELOPE_BYTES + 1)
    with pytest.raises(ValueError, match="byte limit"):
        case.preflight_historical_forecast(*exported, raw, digest(raw))


@pytest.mark.parametrize("coordinate", [True, False])
def test_huge_json_numbers_refused_as_nonfinite(exported, coordinate):
    value = forecast()
    if coordinate:
        value["metadata"]["lat"] = 10**1000
    else:
        value["forecast"][0]["humidity"] = 10**1000
    with pytest.raises(ValueError):
        preflight(exported, value)


def test_publication_failure_cleans_only_created_output(registry, tmp_path, monkeypatch):
    output = tmp_path / "failed"
    original = Path.open

    def fail(path, mode="r", *args, **kwargs):
        if path == output / "library.json" and mode == "xb":
            raise OSError("simulated publication failure")
        return original(path, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", fail)
    with pytest.raises(OSError, match="publication failure"):
        case.export_historical_case(BUNDLE, registry, output)
    assert not output.exists()
