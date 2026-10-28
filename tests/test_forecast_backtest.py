import json
from datetime import UTC, date, datetime, timedelta

import pytest

from src.cli import forecast_ensemble_archive as ensemble_cli
from src.cli.forecast_backtest import build_parser
from src.cli.forecast_backtest import main as backtest_main
from src.cli.forecast_ensemble_archive import load_validated_archive
from src.forecast_backtest import (
    BacktestRow,
    BacktestValidationError,
    ProbabilityEvent,
    load_rows,
    load_rows_with_digest,
    manifest_digest,
    run_backtest,
)
from src.forecast_ensemble import MEMBER_FIELDS, OpenMeteoEnsembleClient


def _ensemble_payload() -> dict:
    snapshot = date(2026, 7, 15)
    dates = [(snapshot + timedelta(days=offset)).isoformat() for offset in range(1, 9)]
    daily = {"time": dates}
    units = {"time": "iso8601"}
    for member_id, field in enumerate(MEMBER_FIELDS):
        daily[field] = [0.1 if member_id < 11 else 1.0] * len(dates)
        units[field] = "mm"
    return {
        "latitude": 43.6,
        "longitude": 3.875,
        "elevation": 42.0,
        "timezone": "GMT",
        "utc_offset_seconds": 0,
        "daily_units": units,
        "daily": daily,
    }


def _ensemble_run(root):
    args = ensemble_cli.build_parser().parse_args(
        ["--snapshot-date", "2026-07-15", "--output-dir", str(root)]
    )

    def transport(params):
        return {
            **_ensemble_payload(),
            "latitude": params["latitude"],
            "longitude": params["longitude"],
        }

    client = OpenMeteoEnsembleClient(
        transport=transport,
        clock=lambda: datetime(2026, 7, 15, 19, 7, tzinfo=UTC),
    )
    return ensemble_cli.run(args, client=client)


def _rows(tmp_path):
    reference = tmp_path / "reference.json"
    predictions = tmp_path / "predictions.json"
    reference.write_text(
        '[{"zone_id":"montpellier","valid_date":"2021-01-02","variable":"temperature_2m","value":12},'
        '{"zone_id":"montpellier","valid_date":"2021-01-03","variable":"temperature_2m","value":14}]',
        encoding="utf-8",
    )
    predictions.write_text(
        '[{"zone_id":"montpellier","origin_date":"2021-01-01","valid_date":"2021-01-02",'
        '"horizon_days":1,"variable":"temperature_2m","value":13},'
        '{"zone_id":"montpellier","origin_date":"2021-01-01","valid_date":"2021-01-03",'
        '"horizon_days":2,"variable":"temperature_2m","value":12}]',
        encoding="utf-8",
    )
    return predictions, reference


def test_backtest_is_deterministic_and_groups_metrics(tmp_path):
    predictions, reference = _rows(tmp_path)
    result = run_backtest(
        load_rows(predictions, kind="prediction", provider="fixture"),
        load_rows(reference, kind="reference"),
        training_end=date(2020, 12, 31),
        holdout_start=date(2021, 1, 1),
        holdout_end=date(2021, 12, 31),
    )
    assert result["schema_version"] == "forecast-backtest/v2"
    assert [item["mae"] for item in result["metrics"]] == [1.0, 2.0]


def test_cli_consumes_verified_full_ensemble_manifest(tmp_path):
    archive_root = tmp_path / "archive"
    manifest = _ensemble_run(archive_root)
    manifest_path = archive_root / manifest["run_id"] / "manifest.json"
    archive = load_validated_archive(manifest_path)
    reference_path = tmp_path / "reference.json"
    reference_path.write_text(
        json.dumps(
            [
                {
                    "zone_id": row["zone_id"],
                    "valid_date": row["valid_date"],
                    "variable": row["variable"],
                    "value": float(index % 2),
                }
                for index, row in enumerate(archive.rows)
            ]
        ),
        encoding="utf-8",
    )
    output_path = tmp_path / "result.json"

    exit_code = backtest_main(
        [
            "--predictions-manifest",
            str(manifest_path),
            "--reference",
            str(reference_path),
            "--training-end",
            "2026-07-14",
            "--holdout-start",
            "2026-07-15",
            "--holdout-end",
            "2026-07-31",
            "--output",
            str(output_path),
        ]
    )

    assert exit_code == 0
    result = json.loads(output_path.read_text(encoding="utf-8"))
    assert result["prediction_archive"] == {
        "schema_version": "forecast-ensemble-forward-manifest/v1",
        "run_id": "aifs-ensemble-forward-20260715",
        "manifest": str(manifest_path.resolve()),
        "manifest_sha256": archive.manifest_sha256,
        "provider": "openmeteo_aifs_ensemble",
        "model": "ecmwf_aifs025_ensemble",
        "rows": 60,
    }
    assert result["reference_input"] == {
        "path": str(reference_path.resolve()),
        "sha256": manifest_digest(reference_path),
        "provenance_status": "caller_supplied_unverified",
    }
    assert sum(metric["n"] for metric in result["metrics"]) == 60


def test_cli_rejects_provider_override_for_verified_manifest(tmp_path):
    archive_root = tmp_path / "archive"
    manifest = _ensemble_run(archive_root)
    manifest_path = archive_root / manifest["run_id"] / "manifest.json"

    with pytest.raises(SystemExit) as caught:
        backtest_main(
            [
                "--predictions-manifest",
                str(manifest_path),
                "--provider",
                "spoofed",
                "--reference",
                str(tmp_path / "unused.json"),
            ]
        )
    assert caught.value.code == 2


def test_cli_refuses_a_prediction_beyond_the_float_range(tmp_path):
    """``--predictions`` holding ``"value": 10**400`` ends in the domain error (#221).

    It used to end in an ``OverflowError`` traceback from inside the row parser.
    """
    predictions, reference = _rows(tmp_path)
    predictions.write_text(
        json.dumps(
            [
                {
                    "zone_id": "montpellier",
                    "origin_date": "2021-01-01",
                    "valid_date": "2021-01-02",
                    "horizon_days": 1,
                    "variable": "temperature_2m",
                    "value": 10**400,
                }
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(BacktestValidationError, match="prediction value must be numeric"):
        backtest_main(["--predictions", str(predictions), "--reference", str(reference)])


def test_backtest_rejects_missing_reference(tmp_path):
    predictions, reference = _rows(tmp_path)
    reference.write_text(
        '[{"zone_id":"montpellier","valid_date":"2021-01-02","variable":"temperature_2m","value":12}]',
        encoding="utf-8",
    )
    with pytest.raises(BacktestValidationError, match="missing reference"):
        run_backtest(
            load_rows(predictions, kind="prediction", provider="fixture"),
            load_rows(reference, kind="reference"),
            training_end=date(2020, 12, 31),
            holdout_start=date(2021, 1, 1),
            holdout_end=date(2021, 12, 31),
        )


def test_backtest_rejects_prediction_origin_in_future_gap(tmp_path):
    predictions, reference = _rows(tmp_path)
    predictions.write_text(
        '[{"zone_id":"montpellier","origin_date":"2021-01-01","valid_date":"2021-01-02",'
        '"horizon_days":1,"variable":"temperature_2m","value":13}]',
        encoding="utf-8",
    )
    with pytest.raises(BacktestValidationError):
        run_backtest(
            load_rows(predictions, kind="prediction", provider="fixture"),
            load_rows(reference, kind="reference"),
            training_end=date(2020, 12, 31),
            holdout_start=date(2021, 1, 3),
            holdout_end=date(2021, 12, 31),
        )


@pytest.mark.parametrize("raw_horizon", [None, "", 1.5, True])
def test_prediction_horizon_is_strict(tmp_path, raw_horizon):
    path = tmp_path / "prediction.json"
    path.write_text(
        json.dumps(
            [
                {
                    "zone_id": "montpellier",
                    "origin_date": "2021-01-01",
                    "valid_date": "2021-01-02",
                    "horizon_days": raw_horizon,
                    "variable": "temperature_2m",
                    "value": 13,
                }
            ]
        ),
        encoding="utf-8",
    )
    with pytest.raises(BacktestValidationError, match="horizon"):
        load_rows(path, kind="prediction", provider="fixture")


def test_jsonl_errors_are_domain_errors_and_single_object_is_accepted(tmp_path):
    single = tmp_path / "single.jsonl"
    single.write_text(
        '{"zone_id":"montpellier","valid_date":"2021-01-02",'
        '"variable":"temperature_2m","value":12}\n',
        encoding="utf-8",
    )
    assert len(load_rows(single, kind="reference")) == 1

    rows, digest = load_rows_with_digest(single, kind="reference")
    assert len(rows) == 1
    assert digest == manifest_digest(single)

    malformed = tmp_path / "malformed.jsonl"
    malformed.write_text(
        '{"zone_id":"montpellier","valid_date":"2021-01-02",'
        '"variable":"temperature_2m","value":12}\nnot-json\n',
        encoding="utf-8",
    )
    with pytest.raises(BacktestValidationError, match="invalid JSONL at line 2"):
        load_rows(malformed, kind="reference")


def _precipitation_prediction(**overrides):
    row = {
        "zone_id": "x",
        "origin_date": "2021-01-01",
        "valid_date": "2021-01-02",
        "horizon_days": 1,
        "variable": "precipitation",
        "value": 0.8,
        "probability": 0.9,
        "probability_event": {"operator": ">", "threshold_mm": 0.1},
    }
    row.update(overrides)
    return row


@pytest.mark.parametrize("layout", ["array", "jsonl"])
@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"value": 10**400}, "prediction value must be numeric"),
        ({"probability": 10**400}, "probability must be numeric"),
        (
            {"probability_event": {"operator": ">", "threshold_mm": 10**400}},
            "threshold_mm must be a finite number",
        ),
    ],
    ids=["value", "probability", "threshold_mm"],
)
@pytest.mark.parametrize("loader", [load_rows, load_rows_with_digest])
def test_a_number_beyond_the_float_range_is_a_domain_error(
    tmp_path, loader, overrides, message, layout
):
    """``json.dumps(10**400)`` writes a valid 401-digit literal that ``float()`` overflows (#221).

    ``OverflowError`` is not a ``ValueError``, so it used to escape the row parser and every
    CLI that loads archives through it.
    """
    row = _precipitation_prediction(**overrides)
    path = tmp_path / f"prediction.{'json' if layout == 'array' else 'jsonl'}"
    if layout == "array":
        path.write_text(json.dumps([row]), encoding="utf-8")
    else:
        path.write_text(
            json.dumps(_precipitation_prediction()) + "\n" + json.dumps(row) + "\n",
            encoding="utf-8",
        )

    with pytest.raises(BacktestValidationError, match=message):
        loader(path, kind="prediction", provider="fixture")


@pytest.mark.parametrize(
    ("name", "message"),
    [
        ("reference.json", "reference.json: a number has more than 4300 digits"),
        ("reference.jsonl", "invalid JSONL at line 2: a number has more than 4300 digits"),
    ],
)
@pytest.mark.parametrize("loader", [load_rows, load_rows_with_digest])
def test_an_integer_past_the_digit_limit_is_a_domain_error(tmp_path, loader, name, message):
    """Past 4,300 digits ``json.loads`` itself raises a plain ``ValueError`` (#221).

    It is not a ``JSONDecodeError``: neither the array branch nor the per-line JSONL
    fallback recognised it as malformed input.
    """
    valid = (
        '{"zone_id":"montpellier","valid_date":"2021-01-02","variable":"temperature_2m","value":12}'
    )
    huge = (
        '{"zone_id":"montpellier","valid_date":"2021-01-03",'
        '"variable":"temperature_2m","value":' + "9" * 5000 + "}"
    )
    path = tmp_path / name
    if name.endswith(".jsonl"):
        # the long literal on line 2, where the per-line fallback parses it
        path.write_text(f"{valid}\n{huge}\n", encoding="utf-8")
    else:
        path.write_text(f"[{valid},{huge}]", encoding="utf-8")

    with pytest.raises(BacktestValidationError, match=message):
        loader(path, kind="reference")


def test_prediction_provider_is_required(tmp_path):
    predictions, _ = _rows(tmp_path)
    with pytest.raises(BacktestValidationError, match="provider is required"):
        load_rows(predictions, kind="prediction")


def test_precipitation_metrics_and_manifest_digest(tmp_path):
    reference = tmp_path / "reference.json"
    predictions = tmp_path / "predictions.json"
    reference.write_text(
        json.dumps(
            [
                {
                    "zone_id": "x",
                    "valid_date": "2021-01-02",
                    "variable": "precipitation",
                    "value": 1.0,
                },
                {
                    "zone_id": "x",
                    "valid_date": "2021-01-03",
                    "variable": "precipitation",
                    "value": 0.0,
                },
            ]
        ),
        encoding="utf-8",
    )
    predictions.write_text(
        json.dumps(
            [
                {
                    "zone_id": "x",
                    "origin_date": "2021-01-01",
                    "valid_date": "2021-01-02",
                    "horizon_days": 1,
                    "variable": "precipitation",
                    "value": 0.8,
                    "probability": 0.9,
                    "probability_event": {"operator": ">", "threshold_mm": 0.1},
                },
                {
                    "zone_id": "x",
                    "origin_date": "2021-01-01",
                    "valid_date": "2021-01-03",
                    "horizon_days": 2,
                    "variable": "precipitation",
                    "value": 0.0,
                    "probability": 0.1,
                    "probability_event": {"operator": ">", "threshold_mm": 0.1},
                },
            ]
        ),
        encoding="utf-8",
    )
    result = run_backtest(
        load_rows(predictions, kind="prediction", provider="fixture"),
        load_rows(reference, kind="reference"),
        training_end=date(2020, 12, 31),
        holdout_start=date(2021, 1, 1),
        holdout_end=date(2021, 12, 31),
    )
    assert all(
        "occurrence_accuracy" in item and "brier_score" in item for item in result["metrics"]
    )
    assert result["probability_evaluation"] == {
        "precipitation_event": {"operator": ">", "threshold_mm": 0.1},
        "reliability_bin_count": 10,
        "event_rate_scope": "evaluation_holdout_only_not_training_climatology",
    }
    manifest = tmp_path / "manifest.json"
    manifest.write_bytes(b'{"manifest":"v1"}')
    assert len(manifest_digest(manifest)) == 64


def _precipitation_rows(probabilities, *, event_threshold_mm=0.1):
    values = [0.0, 0.1, 1.0, 2.0]
    predictions = []
    reference = []
    for offset, (actual, probability) in enumerate(
        zip(values, probabilities, strict=True), start=2
    ):
        valid = date(2021, 1, offset)
        predictions.append(
            BacktestRow(
                "x",
                date(2021, 1, offset - 1),
                valid,
                1,
                "precipitation",
                actual,
                "fixture",
                probability,
                (
                    ProbabilityEvent(operator=">", threshold_mm=event_threshold_mm)
                    if probability is not None
                    else None
                ),
            )
        )
        reference.append(
            BacktestRow(
                "x",
                valid,
                valid,
                0,
                "precipitation",
                actual,
                "reference",
            )
        )
    return predictions, reference


def _run_precipitation(probabilities, *, row_event_threshold_mm=None, **kwargs):
    event_threshold_mm = (
        kwargs.get("precipitation_event_threshold_mm", 0.1)
        if row_event_threshold_mm is None
        else row_event_threshold_mm
    )
    predictions, reference = _precipitation_rows(
        probabilities,
        event_threshold_mm=event_threshold_mm,
    )
    return run_backtest(
        predictions,
        reference,
        training_end=date(2020, 12, 31),
        holdout_start=date(2021, 1, 1),
        holdout_end=date(2021, 12, 31),
        **kwargs,
    )


def test_probability_metrics_emit_deterministic_reliability_bins():
    result = _run_precipitation([0.1, 0.2, 0.8, 0.9], reliability_bin_count=2)
    metric = result["metrics"][0]

    assert metric["status"] == "complete"
    assert metric["probability_status"] == "complete"
    assert metric["brier_score"] == pytest.approx(0.025)
    assert metric["evaluation_event_rate"] == 0.5
    assert metric["mean_probability"] == 0.5
    assert metric["calibration_bias"] == 0.0
    assert metric["expected_calibration_error"] == pytest.approx(0.15)
    assert metric["maximum_calibration_error"] == pytest.approx(0.15)
    assert metric["reliability_bins"] == [
        {
            "index": 0,
            "lower_bound": 0.0,
            "upper_bound": 0.5,
            "upper_bound_inclusive": False,
            "n": 2,
            "mean_probability": pytest.approx(0.15),
            "observed_frequency": 0.0,
            "calibration_gap": pytest.approx(0.15),
        },
        {
            "index": 1,
            "lower_bound": 0.5,
            "upper_bound": 1.0,
            "upper_bound_inclusive": True,
            "n": 2,
            "mean_probability": pytest.approx(0.85),
            "observed_frequency": 1.0,
            "calibration_gap": pytest.approx(-0.15),
        },
    ]


def test_custom_event_threshold_changes_probability_metrics():
    metric = _run_precipitation(
        [0.1, 0.2, 0.8, 0.9],
        precipitation_event_threshold_mm=1.0,
        reliability_bin_count=2,
    )["metrics"][0]

    assert metric["brier_score"] == pytest.approx(0.175)
    assert metric["evaluation_event_rate"] == 0.25
    assert metric["calibration_bias"] == 0.25
    assert metric["expected_calibration_error"] == pytest.approx(0.25)


def test_probability_event_threshold_must_match_evaluation():
    with pytest.raises(BacktestValidationError, match="does not match"):
        _run_precipitation(
            [0.1, 0.2, 0.8, 0.9],
            row_event_threshold_mm=1.0,
            precipitation_event_threshold_mm=0.1,
        )


@pytest.mark.parametrize(
    "row",
    [
        {
            "zone_id": "x",
            "origin_date": "2021-01-01",
            "valid_date": "2021-01-02",
            "horizon_days": 1,
            "variable": "precipitation",
            "value": 0.0,
            "probability": 0.2,
        },
        {
            "zone_id": "x",
            "origin_date": "2021-01-01",
            "valid_date": "2021-01-02",
            "horizon_days": 1,
            "variable": "temperature_2m",
            "value": 10.0,
            "probability": 0.2,
            "probability_event": {"operator": ">", "threshold_mm": 0.1},
        },
    ],
)
def test_probability_rows_require_supported_event_metadata(tmp_path, row):
    path = tmp_path / "prediction.json"
    path.write_text(json.dumps([row]), encoding="utf-8")

    with pytest.raises(BacktestValidationError):
        load_rows(path, kind="prediction", provider="fixture")


@pytest.mark.parametrize(
    ("probabilities", "status", "probability_status"),
    [
        ([None, None, None, None], "point_only", "unavailable"),
        ([0.1, None, 0.8, 0.9], "partial_probability", "partial"),
    ],
)
def test_incomplete_probabilities_never_emit_partial_scores(
    probabilities, status, probability_status
):
    metric = _run_precipitation(probabilities)["metrics"][0]

    assert metric["status"] == status
    assert metric["probability_status"] == probability_status
    assert "brier_score" not in metric
    assert "reliability_bins" not in metric
    assert "probabilistic_metrics_omitted" in metric["warnings"]


def test_probability_one_is_in_the_final_inclusive_bin():
    metric = _run_precipitation([0.0, 0.5, 0.75, 1.0], reliability_bin_count=2)["metrics"][0]

    assert [bucket["n"] for bucket in metric["reliability_bins"]] == [1, 3]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"precipitation_event_threshold_mm": -0.1},
        {"precipitation_event_threshold_mm": float("nan")},
        {"precipitation_event_threshold_mm": True},
        {"reliability_bin_count": 1},
        {"reliability_bin_count": 101},
        {"reliability_bin_count": 2.5},
        {"reliability_bin_count": True},
    ],
)
def test_probability_settings_fail_closed(kwargs):
    with pytest.raises(BacktestValidationError):
        _run_precipitation([0.1, 0.2, 0.8, 0.9], **kwargs)


def test_cli_exposes_probability_evaluation_settings():
    args = build_parser().parse_args(
        [
            "--predictions",
            "predictions.jsonl",
            "--reference",
            "reference.jsonl",
            "--precipitation-event-threshold-mm",
            "0.2",
            "--reliability-bin-count",
            "5",
        ]
    )

    assert args.precipitation_event_threshold_mm == 0.2
    assert args.reliability_bin_count == 5
