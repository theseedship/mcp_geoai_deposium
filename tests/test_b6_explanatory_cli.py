"""Offline B6 level-2 CLI: verified library + saved forecast -> one create-only result (#30)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from src.cli import b6_explanatory as cli
from src.contracts import B6_EXCLUSION_REASONS, B6Level2Output
from src.patterns.library_handoff import export_library_handoff

_REPO = Path(__file__).resolve().parents[1]
PACKAGE = _REPO / "src/data/pattern-library-2026-10-15/package"
MANIFEST_SHA = "bfed1a04a176e9fba288fa822b3a02a3f280591c8fd9c4942bb245854cfa47e2"
FORECAST = _REPO / "tests/fixtures/b6/forecast-bassin_du_lez-fixture.json"
COMPUTED_AT = "2026-10-16T08:00:00+00:00"


def _write_json(path: Path, value) -> str:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def synthetic_package(tmp_path) -> tuple[Path, str]:
    """A real D-A export whose one record is cross-validated on bassin_du_lez."""
    source = tmp_path / "src"
    source.mkdir()
    library = {
        "schema_version": "1.0.0",
        "zone_id": "bassin_du_lez",
        "algorithm": "tgrad",
        "patterns": [
            {
                "pattern_id": 3,
                "antecedent": "(temperature_2m-) +7.0 days",
                "consequent": "ndvi_mean+",
                "pattern": "ndvi_mean+, (temperature_2m-) +7.0 days",
                "variables": ["ndvi_mean", "temperature_2m"],
                "support": 0.8,
                "is_temporal": True,
                "lag_days": 7.0,
                "algorithm": "tgrad",
                "zone_id": "bassin_du_lez",
                "zones_validated": ["bassin_du_lez", "fos_sur_mer"],
                "period_start": "2021-01-01",
                "period_end": "2024-12-31",
                "n_folds": 5,
                "stability": 0.9,
            }
        ],
    }
    library_sha = _write_json(source / "library.json", library)
    data = source / "pilot.csv"
    data.write_bytes(b"date,ndvi_mean,temperature_2m\n2021-01-01,0.3,12\n2024-12-31,0.4,11\n")
    config = {
        "schema_version": "geoai-pattern-library-export/v1",
        "library_version": "2026.10.16-b6-fixture.1",
        "producer": {
            "name": "so4gp",
            "version": "1.1.2",
            "commit": "a" * 40,
            "command": "fixture",
        },
        "areas": ["bassin_du_lez"],
        "inputs": [
            {
                "input_id": "pilot-lez",
                "path": "pilot.csv",
                "source_uri": "fixture:pilot-lez",
                "sha256": hashlib.sha256(data.read_bytes()).hexdigest(),
                "zone_id": "bassin_du_lez",
                "period_start": "2021-01-01",
                "period_end": "2024-12-31",
            }
        ],
        "libraries": [{"path": "library.json", "sha256": library_sha, "input_ids": ["pilot-lez"]}],
    }
    _write_json(source / "export.json", config)
    package = tmp_path / "package"
    return package, export_library_handoff(source / "export.json", package)


def _run(capsys, package, digest, output, *extra, forecast=FORECAST):
    code = cli.main(
        [
            "--package-dir",
            str(package),
            "--expected-manifest-sha256",
            digest,
            "--zone-id",
            "bassin_du_lez",
            "--algorithm",
            "tgrad",
            "--forecast-json",
            str(forecast),
            "--computed-at",
            COMPUTED_AT,
            "--output",
            str(output),
            *extra,
        ]
    )
    return code, json.loads(capsys.readouterr().out)


def test_synthetic_cross_validated_package_is_explained(capsys, synthetic_package, tmp_path):
    package, digest = synthetic_package
    output = tmp_path / "result.json"
    code, summary = _run(capsys, package, digest, output)
    assert code == 0 and summary["status"] == "explained"
    data = output.read_bytes()
    assert summary["output_sha256"] == hashlib.sha256(data).hexdigest()
    result = B6Level2Output.model_validate_json(data)
    [pattern] = result.patterns
    # Temperature falls over the fixture forecast, as the pattern antecedent says.
    assert pattern.factors[2].value == 1.0
    assert pattern.explanatory_index == pytest.approx(0.8 * 0.9 * 0.5**0.7)
    assert result.trace.library.manifest_sha256 == digest
    assert result.trace.forecast["sha256"] == hashlib.sha256(FORECAST.read_bytes()).hexdigest()
    assert result.trace.forecast["source"] == "fixture_not_a_real_forecast"
    assert result.trace.effective_horizon_days == 16
    assert result.trace.computed_at == COMPUTED_AT
    assert result.trace.code_ref["repo"] == "deposium_geoai"
    assert result.confidence is None and result.event_probability is None


def test_real_302_package_gives_the_honest_non_conclusion(capsys, tmp_path):
    output = tmp_path / "real.json"
    code, summary = _run(capsys, PACKAGE, MANIFEST_SHA, output)
    assert code == 0
    assert (summary["status"], summary["reason"]) == ("no_conclusion", "no_admissible_pattern")
    expected = {**dict.fromkeys(B6_EXCLUSION_REASONS, 0), "not_cross_validated": 135}
    assert summary["exclusions"] == expected
    result = B6Level2Output.model_validate_json(output.read_bytes())
    assert result.n_patterns_read == 135 and result.patterns == []
    assert result.trace.library.library_version == "2026.10.15-native-pilots.1"
    assert result.trace.library.area_status == "unqualified"


def test_parameter_overrides_are_recorded(capsys, synthetic_package, tmp_path):
    package, digest = synthetic_package
    output = tmp_path / "strict.json"
    code, summary = _run(capsys, package, digest, output, "--min-folds", "6")
    assert code == 0 and summary["exclusions"]["not_cross_validated"] == 1
    result = B6Level2Output.model_validate_json(output.read_bytes())
    assert result.trace.params.min_folds == 6


def test_output_is_create_only(capsys, synthetic_package, tmp_path):
    package, digest = synthetic_package
    output = tmp_path / "once.json"
    assert _run(capsys, package, digest, output)[0] == 0
    before = output.read_bytes()
    code, summary = _run(capsys, package, digest, output)
    assert code == 1 and summary["status"] == "refused" and "exists" in summary["error"]
    assert output.read_bytes() == before


@pytest.mark.parametrize(
    "forecast, message",
    [
        ({"error": "provider down"}, "forecast_query error"),
        ({"zone_id": "fos_sur_mer", "forecast": [], "metadata": {}}, "zone"),
        ({"forecast": [], "metadata": {}}, "zone"),
        ({"zone_id": "bassin_du_lez", "metadata": {}}, "forecast rows"),
        (
            {
                "zone_id": "bassin_du_lez",
                "forecast": [{"date": "2026-10-02"}, {"date": "2026-10-01"}],
            },
            "ascending",
        ),
        (
            {"zone_id": "bassin_du_lez", "forecast": [], "metadata": {"horizon_days": 0}},
            "horizon",
        ),
        (
            {
                "zone_id": "bassin_du_lez",
                "forecast": [{"date": "2026-10-01"}, {"date": "2026-10-02"}],
                "metadata": {"horizon_days": 16},
            },
            "exceeds",
        ),
    ],
)
def test_unusable_forecast_is_refused_before_publishing(
    capsys, synthetic_package, tmp_path, forecast, message
):
    package, digest = synthetic_package
    saved = tmp_path / "forecast.json"
    _write_json(saved, forecast)
    output = tmp_path / "refused.json"
    code, summary = _run(capsys, package, digest, output, forecast=saved)
    assert code == 1 and summary["status"] == "refused" and message in summary["error"]
    assert not output.exists()


def test_output_inside_the_package_is_refused(capsys, synthetic_package):
    package, digest = synthetic_package
    code, summary = _run(capsys, package, digest, package / "result.json")
    assert code == 1 and "outside the verified package" in summary["error"]
    assert not (package / "result.json").exists()


def test_dangling_symlink_at_output_is_refused(capsys, synthetic_package, tmp_path):
    package, digest = synthetic_package
    target = tmp_path / "elsewhere.json"
    output = tmp_path / "link.json"
    output.symlink_to(target)
    code, summary = _run(capsys, package, digest, output)
    assert code == 1 and "exists" in summary["error"]
    assert not target.exists()


def test_wrong_manifest_digest_is_refused(capsys, tmp_path):
    output = tmp_path / "wrong.json"
    code, summary = _run(capsys, PACKAGE, "0" * 64, output)
    assert code == 1 and "Manifest SHA-256" in summary["error"]
    assert not output.exists()
