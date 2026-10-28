"""Operator smoke over actual Q3 bytes; forecast data is explicitly synthetic."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from src.b6_saved_contracts import MAX_SAVED_ENVELOPE_BYTES
from src.cli import b6_historical_case as cli
from src.patterns.historical_case import ZONES_SHA256

REPO = Path(__file__).resolve().parents[1]


def test_cli_installs_redaction_before_driving_src_handlers(tmp_path, monkeypatch, capsys):
    initialized = []
    monkeypatch.setattr(cli, "setup_logging", lambda: initialized.append(True))

    def verified(*_args):
        assert initialized == [True]
        return {"verified": True}

    monkeypatch.setattr(cli, "verify_historical_case", verified)
    assert cli.main(["verify", str(tmp_path), "--expected-manifest-sha256", "0" * 64]) == 0
    assert json.loads(capsys.readouterr().out) == {"verified": True}


def test_cli_export_verify_and_synthetic_preflight(tmp_path, capsys):
    package = tmp_path / "case"
    assert (
        cli.main(
            [
                "export",
                "--bundle-dir",
                str(REPO / "examples/so4gp_parquet/geoai-o3-q3-2023-handoff"),
                "--zones-geojson",
                str(REPO / "tests/fixtures/pattern-qualification-2026-10-16/zones-q3-2023.geojson"),
                "--output-dir",
                str(package),
            ]
        )
        == 0
    )
    receipt = json.loads(capsys.readouterr().out)
    digest = receipt["manifest_sha256"]
    assert receipt["native_records"] == 149
    assert receipt["ready_for_scoring"] is False

    assert cli.main(["verify", str(package), "--expected-manifest-sha256", digest]) == 0
    verified = json.loads(capsys.readouterr().out)
    assert verified["library"]["zone_id"] == "cross_zone"
    assert verified["descriptor"]["target_zone_id"] == "fos_sur_mer"
    assert len(verified["library"]["patterns"]) == 149

    forecast = {
        "zone_id": "fos_sur_mer",
        "forecast": [
            {
                "date": f"2023-07-{day:02d}",
                "temperature_2m": 20 + day,
                "humidity": 70 - day,
                "evapotranspiration": 2 + day / 10,
            }
            for day in range(1, 6)
        ],
        "metadata": {
            "synthetic": True,
            "source": "synthetic-test-fixture",
            "lat": 43.445,
            "lon": 4.92,
            "registry_sha256": ZONES_SHA256,
            "horizon_days": 5,
            "issued_at": "2026-10-21T00:00:00Z",
            "model_issued_at": "2023-06-30T18:00:00Z",
        },
    }
    raw = json.dumps(forecast, allow_nan=False).encode()
    source = tmp_path / "synthetic-forecast.json"
    source.write_bytes(raw)
    args = [
        "preflight",
        str(package),
        "--expected-manifest-sha256",
        digest,
        "--forecast-json",
        str(source),
        "--expected-forecast-sha256",
        hashlib.sha256(raw).hexdigest(),
    ]
    assert cli.main(args) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["ready_for_scoring"] is False
    assert report["scientific_adoption"] == "pending"
    assert report["schema_version"] == "geoai-b6-historical-forecast-preflight/v1"

    for row in forecast["forecast"]:
        row.pop("humidity")
        row.pop("evapotranspiration")
    raw = json.dumps(forecast, allow_nan=False).encode()
    source.write_bytes(raw)
    args[-1] = hashlib.sha256(raw).hexdigest()
    assert cli.main(args) == 2
    refused = capsys.readouterr()
    assert refused.out == ""
    assert "two finite values" in refused.err
    assert cli.main([*args, "--forecast-profile", "rule15-temperature"]) == 0
    temperature_report = json.loads(capsys.readouterr().out)
    assert temperature_report["schema_version"] == "geoai-b6-historical-forecast-preflight/v2"
    assert temperature_report["profile_id"] == "q3-2023-fos-rule15-temperature/v1"
    assert temperature_report["required_forecast_variables"] == ["temperature_2m"]
    assert temperature_report["primary_case_rule_ids"] == ["cross_zone/tgrad/15"]
    assert temperature_report["ready_for_scoring"] is False
    assert temperature_report["live_activation"] is False
    assert temperature_report["scientific_adoption"] == "pending"
    assert cli.main(["verify", str(package), "--expected-manifest-sha256", "0" * 64]) == 2
    failed = capsys.readouterr()
    assert failed.out == ""
    assert "refused" in failed.err


def test_cli_rejects_unknown_forecast_profile(tmp_path, capsys):
    with pytest.raises(SystemExit) as refused:
        cli.main(
            [
                "preflight",
                str(tmp_path),
                "--expected-manifest-sha256",
                "0" * 64,
                "--forecast-json",
                str(tmp_path / "unused.json"),
                "--expected-forecast-sha256",
                "0" * 64,
                "--forecast-profile",
                "rule15",
            ]
        )
    assert refused.value.code == 2
    assert "invalid choice" in capsys.readouterr().err


def test_cli_bounds_forecast_before_preflight(tmp_path, capsys, monkeypatch):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("oversized forecast reached the verifier")

    monkeypatch.setattr(cli, "preflight_historical_forecast", forbidden)
    source = tmp_path / "oversized.json"
    source.write_bytes(b" " * (MAX_SAVED_ENVELOPE_BYTES + 1))
    assert (
        cli.main(
            [
                "preflight",
                str(tmp_path / "not-a-package"),
                "--expected-manifest-sha256",
                "0" * 64,
                "--forecast-json",
                str(source),
                "--expected-forecast-sha256",
                "0" * 64,
            ]
        )
        == 2
    )
    result = capsys.readouterr()
    assert result.out == ""
    assert "128 KiB" in result.err
