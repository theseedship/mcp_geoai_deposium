import json

import pytest

from src.cli import forecast_benchmark_panel
from src.cli.forecast_benchmark_panel import build_parser, run_benchmark
from src.cli.forecast_forward_archive import CANONICAL_ZONES
from src.tools.climate import _load_zones


def test_preflight_is_network_free(tmp_path):
    reference = tmp_path / "reference.json"
    reference.write_text(
        '[{"zone_id":"montpellier","valid_date":"2021-01-01",'
        '"variable":"temperature_2m","value":10}]',
        encoding="utf-8",
    )
    args = build_parser().parse_args(
        [
            "--start-date",
            "2021-01-01",
            "--end-date",
            "2021-01-01",
            "--reference",
            str(reference),
            "--output-dir",
            str(tmp_path / "out"),
            "--preflight",
        ]
    )
    result = run_benchmark(args)
    assert result["status"] == "preflight_ok"
    assert not (tmp_path / "out" / "archives").exists()


def test_empty_archive_is_explicitly_not_a_score(tmp_path, monkeypatch):
    reference = tmp_path / "reference.json"
    reference.write_text(
        '[{"zone_id":"montpellier","valid_date":"2021-01-01",'
        '"variable":"temperature_2m","value":10}]',
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "src.cli.forecast_benchmark_panel.run_panel",
        lambda args: (
            args.output_dir.mkdir(parents=True, exist_ok=True),
            (args.output_dir / "manifest.json").write_text(
                json.dumps({"schema_version": "forecast-archive-panel/v1"}),
                encoding="utf-8",
            ),
            {"schema_version": "forecast-archive-panel/v1", "entries": []},
        )[-1],
    )
    args = build_parser().parse_args(
        [
            "--start-date",
            "2021-01-01",
            "--end-date",
            "2021-01-01",
            "--reference",
            str(reference),
            "--output-dir",
            str(tmp_path / "out"),
            "--precipitation-event-threshold-mm",
            "1.5",
            "--reliability-bin-count",
            "4",
        ]
    )
    result = run_benchmark(args)
    assert result["status"] == "archive_required"
    assert result["schema_version"] == "forecast-backtest/v2"
    assert result["probability_evaluation"] == {
        "precipitation_event": {"operator": ">", "threshold_mm": 1.5},
        "reliability_bin_count": 4,
        "event_rate_scope": "evaluation_holdout_only_not_training_climatology",
    }
    assert json.loads((tmp_path / "out" / "benchmark.json").read_text())["prediction_rows"] == 0


def test_invalid_artifact_makes_partial_score_explicit(tmp_path, monkeypatch):
    reference = tmp_path / "reference.json"
    reference.write_text(
        '[{"zone_id":"montpellier","valid_date":"2021-01-02",'
        '"variable":"temperature_2m","value":10}]',
        encoding="utf-8",
    )

    def fake_panel(args):
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "manifest.json").write_text("{}", encoding="utf-8")
        (args.output_dir / "valid.json").write_text(
            json.dumps(
                {
                    "coverage": {
                        "status": "available",
                        "provider": "gfs",
                        "model": "gfs_seamless",
                    },
                    "rows": [
                        {
                            "zone_id": "montpellier",
                            "origin_date": "2021-01-01",
                            "valid_date": "2021-01-02",
                            "horizon_days": 1,
                            "variable": "temperature_2m",
                            "value": 11,
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        (args.output_dir / "broken.json").write_text("{broken", encoding="utf-8")
        return {
            "entries": [
                {"status": "available", "output": str(args.output_dir / "valid.json")},
                {"status": "available", "output": str(args.output_dir / "broken.json")},
            ]
        }

    monkeypatch.setattr("src.cli.forecast_benchmark_panel.run_panel", fake_panel)
    args = build_parser().parse_args(
        [
            "--start-date",
            "2021-01-01",
            "--end-date",
            "2021-01-02",
            "--reference",
            str(reference),
            "--output-dir",
            str(tmp_path / "out"),
        ]
    )
    result = run_benchmark(args)
    assert result["status"] == "partial_artifacts"
    assert "broken.json" in (tmp_path / "out" / "benchmark.md").read_text()


def test_existing_archive_dir_is_scored_without_collecting(tmp_path, monkeypatch):
    reference = tmp_path / "reference.json"
    reference.write_text(
        '[{"zone_id":"montpellier","valid_date":"2021-01-02",'
        '"variable":"temperature_2m","value":10}]',
        encoding="utf-8",
    )
    archive_dir = tmp_path / "archives"
    archive_dir.mkdir()
    artifact = archive_dir / "forecast.json"
    artifact.write_text(
        json.dumps(
            {
                "coverage": {
                    "status": "available",
                    "zone_id": "montpellier",
                    "provider": "gfs",
                    "model": "gfs_seamless",
                    "horizon_days": 1,
                },
                "rows": [
                    {
                        "zone_id": "montpellier",
                        "origin_date": "2021-01-01",
                        "valid_date": "2021-01-02",
                        "horizon_days": 1,
                        "variable": "temperature_2m",
                        "value": 11,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    (archive_dir / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "forecast-archive-panel/v1",
                "start_date": "2021-01-01",
                "end_date": "2021-01-02",
                "zones": ["montpellier"],
                "entries": [
                    {
                        "status": "available",
                        "output": str(artifact),
                        "coverage": {
                            "zone_id": "montpellier",
                            "provider": "gfs",
                            "model": "gfs_seamless",
                            "horizon_days": 1,
                        },
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "src.cli.forecast_benchmark_panel.run_panel",
        lambda _args: (_ for _ in ()).throw(AssertionError("network collection should be skipped")),
    )
    args = build_parser().parse_args(
        [
            "--start-date",
            "2021-01-01",
            "--end-date",
            "2021-01-02",
            "--reference",
            str(reference),
            "--output-dir",
            str(tmp_path / "out"),
            "--archive-dir",
            str(archive_dir),
            "--zone-id",
            "montpellier",
            "--precipitation-event-threshold-mm",
            "2.5",
            "--reliability-bin-count",
            "5",
        ]
    )
    result = run_benchmark(args)
    assert result["status"] == "completed"
    assert result["prediction_rows"] == 1
    assert result["probability_evaluation"]["precipitation_event"]["threshold_mm"] == 2.5
    assert result["probability_evaluation"]["reliability_bin_count"] == 5


def test_an_archived_value_beyond_the_float_range_is_an_invalid_artifact(tmp_path, monkeypatch):
    """One artifact holding ``"value": 10**400`` is reported, and the others still score (#221).

    ``float()`` of that integer raises ``OverflowError``, which used to escape the
    per-artifact ``except`` and abort the whole run.
    """
    reference = tmp_path / "reference.json"
    reference.write_text(
        '[{"zone_id":"montpellier","valid_date":"2021-01-02",'
        '"variable":"temperature_2m","value":10}]',
        encoding="utf-8",
    )
    archive_dir = tmp_path / "archives"
    archive_dir.mkdir()
    entries = []
    for name, (provider, model), value in [
        ("valid.json", ("gfs", "gfs_seamless"), 11),
        ("overflow.json", ("aifs", "ecmwf_aifs025_single"), 10**400),
    ]:
        coverage = {
            "zone_id": "montpellier",
            "provider": provider,
            "model": model,
            "horizon_days": 1,
        }
        artifact = archive_dir / name
        artifact.write_text(
            json.dumps(
                {
                    "coverage": {"status": "available", **coverage},
                    "rows": [
                        {
                            "zone_id": "montpellier",
                            "origin_date": "2021-01-01",
                            "valid_date": "2021-01-02",
                            "horizon_days": 1,
                            "variable": "temperature_2m",
                            "value": value,
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        entries.append({"status": "available", "output": str(artifact), "coverage": coverage})
    (archive_dir / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "forecast-archive-panel/v1",
                "start_date": "2021-01-01",
                "end_date": "2021-01-02",
                "zones": ["montpellier"],
                "entries": entries,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "src.cli.forecast_benchmark_panel.run_panel",
        lambda _args: (_ for _ in ()).throw(AssertionError("network collection should be skipped")),
    )
    args = build_parser().parse_args(
        [
            "--start-date",
            "2021-01-01",
            "--end-date",
            "2021-01-02",
            "--reference",
            str(reference),
            "--output-dir",
            str(tmp_path / "out"),
            "--archive-dir",
            str(archive_dir),
            "--zone-id",
            "montpellier",
        ]
    )
    result = run_benchmark(args)
    assert result["status"] == "partial_artifacts"
    assert result["prediction_rows"] == 1
    assert result["invalid_artifacts"] == [
        {
            "artifact": str(archive_dir / "overflow.json"),
            "status": "invalid_artifact",
            "error": "prediction value must be numeric",
        }
    ]
    assert "overflow.json" in (tmp_path / "out" / "benchmark.md").read_text()


def test_existing_archive_dir_rejects_mismatched_window(tmp_path):
    reference = tmp_path / "reference.json"
    reference.write_text("[]", encoding="utf-8")
    archive_dir = tmp_path / "archives"
    archive_dir.mkdir()
    (archive_dir / "manifest.json").write_text(
        json.dumps(
            {
                "start_date": "2020-01-01",
                "end_date": "2020-01-02",
                "zones": ["montpellier"],
                "entries": [],
            }
        ),
        encoding="utf-8",
    )
    args = build_parser().parse_args(
        [
            "--start-date",
            "2021-01-01",
            "--end-date",
            "2021-01-02",
            "--reference",
            str(reference),
            "--output-dir",
            str(tmp_path / "out"),
            "--archive-dir",
            str(archive_dir),
            "--zone-id",
            "montpellier",
        ]
    )
    import pytest

    with pytest.raises(ValueError, match="date window"):
        run_benchmark(args)


def _resume_args(tmp_path):
    # the documented B5a resume command: no --zone-id, so the default zone set applies
    return build_parser().parse_args(
        [
            "--start-date",
            "2021-01-01",
            "--end-date",
            "2021-01-02",
            "--reference",
            str(tmp_path / "reference.json"),
            "--output-dir",
            str(tmp_path / "out"),
        ]
    )


def _manifest(zones):
    return {"start_date": "2021-01-01", "end_date": "2021-01-02", "zones": zones, "entries": []}


@pytest.fixture
def zones_with_a_client(monkeypatch):
    zones = {**_load_zones(), "client_probe": {"lon": 4.9, "lat": 43.4, "role": "client"}}
    monkeypatch.setattr(forecast_benchmark_panel, "_load_zones", lambda: zones)


@pytest.mark.usefixtures("zones_with_a_client")
def test_a_fifteen_zone_archive_still_resumes_once_a_client_site_exists(tmp_path):
    forecast_benchmark_panel._validate_existing_manifest(
        _manifest(list(CANONICAL_ZONES)), args=_resume_args(tmp_path), archive_dir=tmp_path
    )


@pytest.mark.usefixtures("zones_with_a_client")
def test_an_archive_that_collected_a_client_site_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="zones do not match"):
        forecast_benchmark_panel._validate_existing_manifest(
            _manifest([*CANONICAL_ZONES, "client_probe"]),
            args=_resume_args(tmp_path),
            archive_dir=tmp_path,
        )


def test_resume_check_refuses_a_malformed_role(tmp_path, monkeypatch):
    # Without the check, a 14-zone archive would match a 14-zone "expected" set and resume.
    zones = {zid: dict(zone) for zid, zone in _load_zones().items()}
    zones["bassin_du_lez"]["role"] = "panle"
    monkeypatch.setattr(forecast_benchmark_panel, "_load_zones", lambda: zones)
    shortened = [zid for zid in CANONICAL_ZONES if zid != "bassin_du_lez"]
    with pytest.raises(ValueError, match="bassin_du_lez"):
        forecast_benchmark_panel._validate_existing_manifest(
            _manifest(shortened), args=_resume_args(tmp_path), archive_dir=tmp_path
        )
