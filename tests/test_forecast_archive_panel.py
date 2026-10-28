import argparse
import hashlib
import json
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pytest

from src.cli import forecast_archive_panel, forecast_skill_matrix
from src.forecast_archive import (
    PREVIOUS_RUNS_HOURLY_UNITS,
    ArchiveRow,
    Coverage,
    PreviousRunsArchiveClient,
    _previous_field,
)
from src.utils.openmeteo import NORMALIZED_VARS


def _legacy_parser() -> argparse.ArgumentParser:
    """The Namespace ``run_panel``'s library callers build; the CLI no longer exposes it."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--end-date", required=True)
    parser.add_argument("--zone-id", nargs="+")
    parser.add_argument(
        "--provider-model", action="append", type=forecast_archive_panel._provider_model
    )
    parser.add_argument("--horizon-days", type=int, nargs="+", default=[1])
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--storage-mode", choices=("local", "s3"), default="local")
    parser.add_argument("--run-id")
    parser.add_argument("--s3-bucket", default=None)
    parser.add_argument("--s3-prefix", default="climate/forecast")
    return parser


def _args(tmp_path: Path, *extra: str):
    return _legacy_parser().parse_args(
        [
            "--start-date",
            "2025-06-01",
            "--end-date",
            "2025-06-01",
            "--zone-id",
            "montpellier",
            "--output-dir",
            str(tmp_path),
            *extra,
        ]
    )


class _Client:
    def fetch_daily(self, **kwargs):
        coverage = Coverage(
            kwargs["zone_id"],
            kwargs["provider"],
            kwargs["model"],
            kwargs["horizon_days"],
            kwargs["start_date"],
            kwargs["end_date"],
            1,
            1,
            "available",
        )
        row = ArchiveRow(
            kwargs["zone_id"],
            date(2025, 5, 31),
            date(2025, 6, 1),
            kwargs["horizon_days"],
            "temperature_2m",
            20.0,
            kwargs["provider"],
            kwargs["model"],
        )
        return [row], coverage

    def fetch_daily_multi(self, **kwargs):
        return {
            horizon: self.fetch_daily(**kwargs, horizon_days=horizon)
            for horizon in kwargs.pop("horizons")
        }


def test_custom_provider_models_replace_defaults_and_model_is_in_key(tmp_path, monkeypatch):
    monkeypatch.setattr(
        forecast_archive_panel,
        "_load_zones",
        lambda: {"montpellier": {"lat": 43.6, "lon": 3.5}},
    )
    args = _args(tmp_path, "--provider-model", "custom:custom_model")
    manifest = forecast_archive_panel.run_panel(args, client=_Client())

    assert [entry["key"] for entry in manifest["entries"]] == [
        "montpellier__custom__custom_model__2025-06-01__2025-06-01__1"
    ]
    artifact = tmp_path / "montpellier__custom__custom_model__2025-06-01__2025-06-01__1.json"
    payload = json.loads(artifact.read_text())
    assert payload["rows"][0]["valid_date"] == "2025-06-01"


def test_default_models_do_not_collide_and_resume_preserves_coverage(tmp_path, monkeypatch):
    monkeypatch.setattr(
        forecast_archive_panel,
        "_load_zones",
        lambda: {"montpellier": {"lat": 43.6, "lon": 3.5}},
    )
    manifest = forecast_archive_panel.run_panel(_args(tmp_path), client=_Client())
    assert len(manifest["entries"]) == 2
    assert len({entry["key"] for entry in manifest["entries"]}) == 2

    resumed = forecast_archive_panel.run_panel(_args(tmp_path), client=_Client())
    assert all(entry["status"] == "resumed" for entry in resumed["entries"])
    assert all(entry["coverage"]["status"] == "available" for entry in resumed["entries"])


def test_invalid_existing_artifact_is_reported(tmp_path, monkeypatch):
    monkeypatch.setattr(
        forecast_archive_panel,
        "_load_zones",
        lambda: {"montpellier": {"lat": 43.6, "lon": 3.5}},
    )
    path = tmp_path / "montpellier__gfs__gfs_seamless__2025-06-01__2025-06-01__1.json"
    path.write_text("{}")
    manifest = forecast_archive_panel.run_panel(_args(tmp_path), client=_Client())
    assert manifest["entries"][0]["status"] == "provider_error"
    assert "invalid existing artifact" in manifest["entries"][0]["error"]


def test_provider_errors_are_redacted_before_manifest_persistence(tmp_path, monkeypatch):
    class LeakyClient:
        apikey = "archive/secret+"

        def fetch_daily_multi(self, **_kwargs):
            raise RuntimeError("request failed: apikey=archive/secret+ encoded=archive%2Fsecret%2B")

    monkeypatch.setattr(
        forecast_archive_panel,
        "_load_zones",
        lambda: {"montpellier": {"lat": 43.6, "lon": 3.5}},
    )
    manifest = forecast_archive_panel.run_panel(_args(tmp_path), client=LeakyClient())
    serialized = json.dumps(manifest)

    assert "archive/secret+" not in serialized
    assert "archive%2Fsecret%2B" not in serialized
    assert "[REDACTED]" in serialized


def test_default_zone_set_never_collects_a_client_site(tmp_path, monkeypatch):
    # Without --zone-id the panel defaults to the scientific zones (panel + pilot): a client
    # site would otherwise be collected, and every later B5a resume check would reject it.
    monkeypatch.setattr(
        forecast_archive_panel,
        "_load_zones",
        lambda: {
            "montpellier": {"lat": 43.6, "lon": 3.5, "role": "panel"},
            "client_probe": {"lat": 43.4, "lon": 4.9, "role": "client"},
        },
    )
    args = _legacy_parser().parse_args(
        [
            "--start-date",
            "2025-06-01",
            "--end-date",
            "2025-06-01",
            "--output-dir",
            str(tmp_path),
            "--provider-model",
            "gfs:gfs_seamless",
        ]
    )

    manifest = forecast_archive_panel.run_panel(args, client=_Client())

    assert manifest["zones"] == ["montpellier"]
    assert [entry["coverage"]["zone_id"] for entry in manifest["entries"]] == ["montpellier"]


def test_s3_uri_is_confined_to_archive_namespace():
    assert (
        forecast_archive_panel._s3_uri(
            bucket="deposium",
            prefix="climate/forecast",
            run_id="aifs-forward-20260715",
            name="montpellier.json",
        )
        == "s3://deposium/climate/forecast/aifs-forward-20260715/montpellier.json"
    )

    with pytest.raises(ValueError, match="unsafe S3"):
        forecast_archive_panel._s3_uri(
            bucket="deposium",
            prefix="climate/../raw",
            run_id="run",
            name="artifact.json",
        )


def test_s3_exists_treats_missing_and_corrupt_objects_as_cache_misses(monkeypatch):
    def missing(_path):
        raise RuntimeError("HTTP Error 404: Not Found")

    monkeypatch.setattr(forecast_archive_panel, "_read_json", missing)
    assert not forecast_archive_panel._s3_exists("s3://deposium/missing.json")

    def corrupt(_path):
        raise json.JSONDecodeError("invalid", "{", 0)

    monkeypatch.setattr(forecast_archive_panel, "_read_json", corrupt)
    assert not forecast_archive_panel._s3_exists("s3://deposium/corrupt.json")

    def denied(_path):
        raise RuntimeError("access denied")

    monkeypatch.setattr(forecast_archive_panel, "_read_json", denied)
    with pytest.raises(RuntimeError, match="access denied"):
        forecast_archive_panel._s3_exists("s3://deposium/forbidden.json")


def test_s3_mode_uses_configured_bucket_by_default(tmp_path, monkeypatch):
    monkeypatch.setattr(
        forecast_archive_panel,
        "_load_zones",
        lambda: {"montpellier": {"lat": 43.6, "lon": 3.5}},
    )
    monkeypatch.setattr(forecast_archive_panel.settings, "S3_BUCKET", "configured-bucket")
    monkeypatch.setattr(forecast_archive_panel, "_s3_exists", lambda _path: False)
    written: list[str | Path] = []

    def capture(path, payload):
        written.append(path)

    monkeypatch.setattr(forecast_archive_panel, "_write_json", capture)
    args = _args(tmp_path, "--storage-mode", "s3", "--run-id", "run-1")
    manifest = forecast_archive_panel.run_panel(args, client=_Client())

    assert manifest["storage_mode"] == "s3"
    assert all(str(path).startswith("s3://configured-bucket/") for path in written)


def test_s3_mode_does_not_require_a_local_output_directory(tmp_path, monkeypatch):
    monkeypatch.setattr(
        forecast_archive_panel,
        "_load_zones",
        lambda: {"montpellier": {"lat": 43.6, "lon": 3.5}},
    )
    monkeypatch.setattr(forecast_archive_panel.settings, "S3_BUCKET", "configured-bucket")
    monkeypatch.setattr(forecast_archive_panel, "_s3_exists", lambda _path: False)
    monkeypatch.setattr(forecast_archive_panel, "_write_json", lambda _path, _payload: None)
    blocked_parent = tmp_path / "not-a-directory"
    blocked_parent.write_text("blocked", encoding="utf-8")
    args = _args(
        blocked_parent / "out",
        "--storage-mode",
        "s3",
        "--run-id",
        "run-1",
    )

    manifest = forecast_archive_panel.run_panel(args, client=_Client())

    assert manifest["storage_mode"] == "s3"
    assert manifest["entries"][0]["status"] == "available"


def test_s3_mode_refetches_and_replaces_a_corrupt_artifact(tmp_path, monkeypatch):
    monkeypatch.setattr(
        forecast_archive_panel,
        "_load_zones",
        lambda: {"montpellier": {"lat": 43.6, "lon": 3.5}},
    )
    monkeypatch.setattr(forecast_archive_panel.settings, "S3_BUCKET", "configured-bucket")

    def corrupt(_path):
        raise json.JSONDecodeError("invalid", "{", 0)

    monkeypatch.setattr(forecast_archive_panel, "_read_json", corrupt)
    written: list[str | Path] = []
    monkeypatch.setattr(
        forecast_archive_panel,
        "_write_json",
        lambda path, _payload: written.append(path),
    )
    args = _args(tmp_path, "--storage-mode", "s3", "--run-id", "run-1")

    manifest = forecast_archive_panel.run_panel(args, client=_Client())

    assert manifest["entries"][0]["status"] == "available"
    assert any("climate/forecast/run-1/" in str(path) for path in written)


def test_internal_caller_can_suppress_low_level_manifest(tmp_path, monkeypatch):
    monkeypatch.setattr(
        forecast_archive_panel,
        "_load_zones",
        lambda: {"montpellier": {"lat": 43.6, "lon": 3.5}},
    )
    written: list[str | Path] = []
    monkeypatch.setattr(
        forecast_archive_panel,
        "_write_json",
        lambda path, _payload: written.append(path),
    )
    args = _args(tmp_path, "--provider-model", "custom:custom_model")
    args.write_manifest = False

    manifest = forecast_archive_panel.run_panel(args, client=_Client())

    assert manifest["entries"][0]["status"] == "available"
    assert len(written) == 1
    assert Path(written[0]).name != "manifest.json"


class TestPublishedManifestCarriesNoCredentials:
    """#146 — the only run-manifest publication boundary in the repo that did not scrub.

    It writes to the same ``climate/runs/{run_id}/manifest.json`` prefix as every other
    publisher, i.e. the artifact meant for downstream handoff, and ``import-zip`` republishes
    that into a destination bucket handed to another party.
    """

    ACCESS_KEY = "rustfs-admin"
    # Deliberately a SUPERSTRING of the access key: replacing the short one first would rewrite
    # this to "[REDACTED]-secret" and publish the tail, which a naive "the whole secret is
    # absent" assertion still passes. Longest-first is what makes it fail properly.
    SECRET_KEY = "rustfs-admin-secret"
    ENDPOINT = "http://rustfs:9000"

    @pytest.fixture(autouse=True)
    def _store_configured(self, monkeypatch):
        from src.utils import s3_endpoint

        monkeypatch.setattr(s3_endpoint.settings, "S3_ACCESS_KEY", self.ACCESS_KEY)
        monkeypatch.setattr(s3_endpoint.settings, "S3_SECRET_KEY", self.SECRET_KEY)
        monkeypatch.setattr(s3_endpoint.settings, "S3_ENDPOINT", self.ENDPOINT)
        monkeypatch.setattr(forecast_archive_panel.settings, "S3_ENDPOINT", self.ENDPOINT)
        for name in s3_endpoint.AMBIENT_STORE_ENV_VARS:
            monkeypatch.delenv(name, raising=False)

    def test_a_written_manifest_carries_no_store_credential(self, tmp_path):
        path = tmp_path / "manifest.json"
        forecast_archive_panel._write_json(
            path,
            {
                "run_id": f"{self.ACCESS_KEY}-2026-08-30",
                "errors": [
                    {
                        "zone_id": "montpellier",
                        "error": f"IO Error: HTTP HEAD 'http://{self.ACCESS_KEY}:{self.SECRET_KEY}@rustfs:9000/b/k'",
                    }
                ],
            },
        )
        written = path.read_text(encoding="utf-8")
        assert self.SECRET_KEY not in written
        assert f"{self.ACCESS_KEY}:" not in written

    def test_the_redactor_does_not_corrupt_the_artifact_it_protects(self, tmp_path):
        # The run id legitimately embeds the account name. A redactor that rewrites identifiers
        # makes every URI in the manifest point at an object that does not exist — and a redactor
        # that corrupts its own artifact gets switched off, which protects nothing.
        path = tmp_path / "manifest.json"
        run_id = f"{self.ACCESS_KEY}-2026-08-30"
        forecast_archive_panel._write_json(path, {"run_id": run_id, "errors": []})
        assert json.loads(path.read_text(encoding="utf-8"))["run_id"] == run_id

    def test_the_provider_error_covers_store_credentials_not_just_the_api_key(self):
        exc = Exception(
            f"IO Error: HTTP HEAD 'http://{self.ACCESS_KEY}:{self.SECRET_KEY}@rustfs:9000/b/k'"
        )
        message = forecast_archive_panel._provider_error_message(exc, client=None)
        assert self.SECRET_KEY not in message
        assert f"{self.ACCESS_KEY}:" not in message

    def test_the_longer_secret_is_replaced_before_its_own_prefix(self):
        # The set the previous implementation iterated had arbitrary order, so the access key
        # could be replaced first and publish "-secret" as the tail of the password.
        exc = Exception(f"denied: {self.SECRET_KEY}")
        message = forecast_archive_panel._provider_error_message(exc, client=None)
        assert message == "denied: [REDACTED]"
        assert "-secret" not in message

    def test_the_api_key_is_still_covered(self, monkeypatch):
        monkeypatch.setattr(
            forecast_archive_panel.settings, "FORECAST_ARCHIVE_API_KEY", "weather-api-key-xyz"
        )
        message = forecast_archive_panel._provider_error_message(
            Exception("401 for key weather-api-key-xyz"), client=None
        )
        assert "weather-api-key-xyz" not in message


def test_default_zone_set_refuses_a_malformed_role(tmp_path, monkeypatch):
    # A typo'd role must stop the run, not drop the zone from the default panel.
    monkeypatch.setattr(
        forecast_archive_panel,
        "_load_zones",
        lambda: {
            "montpellier": {"lat": 43.6, "lon": 3.5, "role": "panel"},
            "bassin_du_lez": {"lat": 43.6, "lon": 3.9, "role": "panle"},
        },
    )
    args = _legacy_parser().parse_args(
        ["--start-date", "2025-06-01", "--end-date", "2025-06-01", "--output-dir", str(tmp_path)]
    )
    with pytest.raises(ValueError, match="bassin_du_lez"):
        forecast_archive_panel.run_panel(args, client=_Client())
    assert list(tmp_path.iterdir()) == []


# ---------------------------------------------------------------------------------------
# #193 — the frozen panel: the accepted manifests and the frozen sampling points, never the
# current zone geometry. Synthetic inputs below; the real handoff files are only READ.
# ---------------------------------------------------------------------------------------

HANDOFF = Path(__file__).resolve().parent.parent / "tests/fixtures/b5a-replacement-20261005"
IDENTITIES = (("gfs", "gfs_seamless"), ("aifs", "ecmwf_aifs025_single"))


def _frozen_files(
    root: Path,
    *,
    zones=("zone_a", "zone_b"),
    points=None,
    dates=("2025-06-01", "2025-06-02"),
    horizons=(1, 3),
    variables=("temperature_2m", "precipitation"),
    identities=IDENTITIES,
    configuration=None,
    panel_overrides=None,
):
    """Synthetic frozen panel manifests plus the sampling-points file their digest binds."""
    root.mkdir(parents=True, exist_ok=True)
    if points is None:
        points = [
            {"zone_id": zone, "latitude": 40.0 + index, "longitude": 2.0 + index}
            for index, zone in enumerate(zones)
        ]
    points_path = root / "sampling-points.json"
    points_path.write_text(json.dumps({"sampling_points": points}), encoding="utf-8")
    config = {
        "day_grain": "utc_day",
        "lead_semantics": "fixed_hourly_offset_24_times_horizon_days",
        "panel_id": "synthetic-v1",
        "sampling_points_sha256": hashlib.sha256(points_path.read_bytes()).hexdigest(),
        "source": "openmeteo_previous_runs",
        "timezone": "UTC",
        "wind_speed_unit": "ms",
        **(configuration or {}),
    }
    panels = []
    for provider, model in identities:
        record = {
            "schema_version": "forecast-skill-panel/v1",
            "provider": provider,
            "model": model,
            "configuration": config,
            "zones": list(zones),
            "variables": list(variables),
            "horizons_days": list(horizons),
            "valid_dates": list(dates),
        }
        record.update((panel_overrides or {}).get(provider, {}))
        path = root / f"{provider}-panel.json"
        path.write_text(json.dumps(record), encoding="utf-8")
        panels.append(path)
    return panels, points_path


def test_the_accepted_handoff_panel_loads_with_its_frozen_points():
    frozen = forecast_archive_panel._load_frozen_panel(
        [HANDOFF / "gfs-panel.json", HANDOFF / "aifs-panel.json"],
        HANDOFF / "sampling-points.json",
    )

    assert frozen.panel_id == "b5a-replacement-20261005-v1"
    assert [identity.panel.expected_pairs for identity in frozen.identities] == [86_040, 86_040]
    assert (frozen.start_date, frozen.end_date) == ("2025-02-25", "2025-10-21")
    # The three zones whose current bbox midpoint differs: the FROZEN point is what is used.
    assert frozen.points["bassin_du_lez"] == (43.72, 3.8149999999999995)
    assert frozen.points["fos_sur_mer"] == (43.445, 4.92)
    assert frozen.points["montpellier_metropole"] == (43.605000000000004, 3.865)


def test_a_sampling_file_with_another_digest_is_refused(tmp_path):
    panels, points = _frozen_files(tmp_path)
    points.write_text(points.read_text(encoding="utf-8") + " ", encoding="utf-8")
    with pytest.raises(forecast_archive_panel.ArchiveRefusal, match="sampling_points_sha256"):
        forecast_archive_panel._load_frozen_panel(panels, points)


def test_a_zone_without_a_frozen_point_is_refused(tmp_path):
    panels, points = _frozen_files(
        tmp_path, points=[{"zone_id": "zone_a", "latitude": 40.0, "longitude": 2.0}]
    )
    with pytest.raises(forecast_archive_panel.ArchiveRefusal, match="zone_b"):
        forecast_archive_panel._load_frozen_panel(panels, points)


def test_a_duplicated_frozen_point_is_refused(tmp_path):
    point = {"zone_id": "zone_a", "latitude": 40.0, "longitude": 2.0}
    panels, points = _frozen_files(tmp_path, zones=("zone_a",), points=[point, point])
    with pytest.raises(forecast_archive_panel.ArchiveRefusal, match="duplicate sampling point"):
        forecast_archive_panel._load_frozen_panel(panels, points)


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"zones": ["zone_a"]}, "disagree on zones"),
        ({"horizons_days": [1]}, "disagree on horizons_days"),
        ({"valid_dates": ["2025-06-01"]}, "disagree on valid_dates"),
        ({"variables": ["temperature_2m"]}, "disagree on variables"),
    ],
)
def test_panel_manifests_that_disagree_are_refused(tmp_path, override, message):
    panels, points = _frozen_files(tmp_path, panel_overrides={"aifs": override})
    with pytest.raises(forecast_archive_panel.ArchiveRefusal, match=message):
        forecast_archive_panel._load_frozen_panel(panels, points)


def test_a_repeated_identity_is_refused(tmp_path):
    panels, points = _frozen_files(tmp_path, identities=(IDENTITIES[0],))
    with pytest.raises(forecast_archive_panel.ArchiveRefusal, match="identity"):
        forecast_archive_panel._load_frozen_panel([panels[0], panels[0]], points)


@pytest.mark.parametrize(
    ("configuration", "message"),
    [
        ({"wind_speed_unit": "kmh"}, "wind_speed_unit"),
        ({"timezone": "Europe/Paris"}, "timezone"),
        ({"lead_semantics": "single_00z_run"}, "lead_semantics"),
    ],
)
def test_a_configuration_the_client_does_not_send_is_refused(tmp_path, configuration, message):
    panels, points = _frozen_files(tmp_path, configuration=configuration)
    with pytest.raises(forecast_archive_panel.ArchiveRefusal, match=message):
        forecast_archive_panel._load_frozen_panel(panels, points)


def test_non_contiguous_valid_dates_are_refused(tmp_path):
    panels, points = _frozen_files(tmp_path, dates=("2025-06-01", "2025-06-03"))
    with pytest.raises(forecast_archive_panel.ArchiveRefusal, match="contiguous"):
        forecast_archive_panel._load_frozen_panel(panels, points)


def test_a_zone_id_that_would_escape_the_archive_is_refused(tmp_path):
    panels, points = _frozen_files(tmp_path, zones=("../zone_a",))
    with pytest.raises(forecast_archive_panel.ArchiveRefusal, match="unsafe"):
        forecast_archive_panel._load_frozen_panel(panels, points)


# ---------------------------------------------------------------------------------------
# #193 — collect: frozen coordinates, raw bytes + provenance, create-only files, and a resume
# that trusts an artifact only when it re-derives from its raw response. Transport injected;
# no provider is ever called.
# ---------------------------------------------------------------------------------------

FIXED_NOW = datetime(2026, 10, 15, 9, 30, tzinfo=UTC)
_FIELD_VARIABLE = {
    _previous_field(variable, horizon): variable
    for variable in NORMALIZED_VARS
    for horizon in range(1, 8)
}


def _transport(calls=None, *, gap=None):
    """Answer like Previous Runs from the request alone: UTC hours, declared units, a grid cell.

    ``gap=(field, "YYYY-MM-DD")`` blanks one field for one day, the partial-coverage case.
    """

    def transport(params):
        if calls is not None:
            calls.append(dict(params))
        start = date.fromisoformat(params["start_date"])
        days = (date.fromisoformat(params["end_date"]) - start).days + 1
        times = [
            f"{(start + timedelta(days=day)).isoformat()}T{hour:02d}:00"
            for day in range(days)
            for hour in range(24)
        ]
        hourly = {"time": times}
        units = {"time": "iso8601"}
        for index, field in enumerate(params["hourly"].split(",")):
            values = [round(1.0 + index + (n % 24) / 10, 3) for n in range(len(times))]
            if gap and gap[0] == field:
                values = [None if times[n].startswith(gap[1]) else v for n, v in enumerate(values)]
            hourly[field] = values
            units[field] = PREVIOUS_RUNS_HOURLY_UNITS[_FIELD_VARIABLE[field]]
        payload = {
            "latitude": round(params["latitude"], 1) + 0.05,
            "longitude": round(params["longitude"], 1) + 0.05,
            "elevation": 12.0,
            "generationtime_ms": 0.1,
            "utc_offset_seconds": 0,
            "timezone": "GMT",
            "timezone_abbreviation": "GMT",
            "hourly_units": units,
            "hourly": hourly,
        }
        return json.dumps(payload).encode("utf-8")

    return transport


def _client(transport):
    return PreviousRunsArchiveClient(apikey="", transport=transport, now=lambda: FIXED_NOW)


def _collect_args(panels, points, output, run_id="run-1"):
    argv = ["collect"]
    for panel in panels:
        argv += ["--panel-manifest", str(panel)]
    argv += ["--sampling-points", str(points), "--output-dir", str(output), "--run-id", run_id]
    return forecast_archive_panel.build_parser().parse_args(argv)


def _snapshot(root: Path) -> dict[str, bytes]:
    return {
        str(path.relative_to(root)): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


@pytest.fixture
def no_current_geometry(monkeypatch):
    def refuse():
        raise AssertionError("the frozen collection must not read the current zone geometry")

    monkeypatch.setattr(forecast_archive_panel, "_load_zones", refuse)


def test_collect_sends_the_frozen_points_of_the_accepted_panel(tmp_path, no_current_geometry):
    calls = []

    def offline(params):
        calls.append(dict(params))
        raise RuntimeError("offline test transport")

    args = _collect_args(
        [HANDOFF / "gfs-panel.json", HANDOFF / "aifs-panel.json"],
        HANDOFF / "sampling-points.json",
        tmp_path / "out",
    )
    manifest = forecast_archive_panel.collect(args, client=_client(offline))

    sent = {(round(call["latitude"], 12), round(call["longitude"], 12)) for call in calls}
    for zone, frozen in {
        "bassin_du_lez": (43.72, 3.8149999999999995),
        "fos_sur_mer": (43.445, 4.92),
        "montpellier_metropole": (43.605000000000004, 3.865),
    }.items():
        assert (round(frozen[0], 12), round(frozen[1], 12)) in sent, zone
    # The current bbox midpoints the handoff names must never be requested.
    assert (round(43.65965, 12), round(3.795, 12)) not in sent
    assert {call["start_date"] for call in calls} == {"2025-02-25"}
    assert {call["end_date"] for call in calls} == {"2025-10-21"}
    assert [identity["expected_pairs"] for identity in manifest["identities"]] == [86_040] * 2
    assert all(identity["available_pairs"] == 0 for identity in manifest["identities"])


def test_collect_keeps_raw_bytes_provenance_and_rows(tmp_path, no_current_geometry):
    panels, points = _frozen_files(tmp_path / "inputs")
    calls = []
    output = tmp_path / "out"
    manifest = forecast_archive_panel.collect(
        _collect_args(panels, points, output), client=_client(_transport(calls))
    )

    assert {entry["status"] for entry in manifest["entries"]} == {"available"}
    assert len(manifest["entries"]) == 2 * 2 * 2  # zones x identities x horizons
    assert len(calls) == 4  # one multi-horizon request per zone x identity
    assert {(call["latitude"], call["longitude"]) for call in calls} == {(40.0, 2.0), (41.0, 3.0)}

    artifact = json.loads((output / f"{manifest['entries'][0]['key']}.json").read_text())
    raw = (output / "raw" / f"{artifact['raw_sha256']}.json").read_bytes()
    assert hashlib.sha256(raw).hexdigest() == artifact["raw_sha256"]
    provenance = json.loads((output / "raw" / artifact["provenance"]).read_text())
    assert provenance["retrieved_at"] == "2026-10-15T09:30:00+00:00"
    assert provenance["requested_coordinates"] == {"latitude": 40.0, "longitude": 2.0}
    assert provenance["returned"]["latitude"] == 40.05
    assert provenance["returned"]["hourly_units"]["temperature_2m_previous_day1"] == "°C"
    assert provenance["params"]["models"] == "gfs_seamless"
    assert "apikey" not in provenance["params"]
    assert provenance["attested_model"] is None
    assert provenance["attested_initialization"] is None
    assert (
        provenance["configuration"]["sampling_points_sha256"]
        == hashlib.sha256(points.read_bytes()).hexdigest()
    )
    assert artifact["coverage"]["expected_pairs"] == artifact["coverage"]["available_pairs"] == 4
    assert (output / "manifest-run-1.json").is_file()


def test_partial_coverage_lists_the_missing_pair_and_keeps_the_denominator(tmp_path):
    panels, points = _frozen_files(tmp_path / "inputs", zones=("zone_a",), horizons=(1,))
    gap = (_previous_field("precipitation", 1), "2025-06-02")
    manifest = forecast_archive_panel.collect(
        _collect_args(panels, points, tmp_path / "out"),
        client=_client(_transport(gap=gap)),
    )

    entry = manifest["entries"][0]
    assert entry["status"] == "partial_archive"
    assert entry["coverage"]["missing"] == [
        {"valid_date": "2025-06-02", "variable": "precipitation", "horizon_days": 1}
    ]
    gfs = manifest["identities"][0]
    assert (gfs["expected_pairs"], gfs["available_pairs"]) == (4, 3)


def test_a_second_run_resumes_identically_without_calling_the_provider(tmp_path):
    panels, points = _frozen_files(tmp_path / "inputs")
    output = tmp_path / "out"
    forecast_archive_panel.collect(
        _collect_args(panels, points, output), client=_client(_transport())
    )
    before = _snapshot(output)
    calls = []

    resumed = forecast_archive_panel.collect(
        _collect_args(panels, points, output, run_id="run-2"), client=_client(_transport(calls))
    )

    assert calls == []
    assert {entry["status"] for entry in resumed["entries"]} == {"resumed"}
    after = _snapshot(output)
    assert {name: data for name, data in after.items() if name != "manifest-run-2.json"} == before


def test_a_tampered_raw_response_is_an_integrity_error_left_untouched(tmp_path):
    panels, points = _frozen_files(
        tmp_path / "inputs", zones=("zone_a",), identities=IDENTITIES[:1]
    )
    output = tmp_path / "out"
    first = forecast_archive_panel.collect(
        _collect_args(panels[:1], points, output), client=_client(_transport())
    )
    artifact = json.loads((output / f"{first['entries'][0]['key']}.json").read_text())
    raw_path = output / "raw" / f"{artifact['raw_sha256']}.json"
    raw_path.write_bytes(raw_path.read_bytes().replace(b"1.0", b"9.0", 1))
    tampered = raw_path.read_bytes()
    calls = []

    second = forecast_archive_panel.collect(
        _collect_args(panels[:1], points, output, run_id="run-2"),
        client=_client(_transport(calls)),
    )

    assert calls == []  # never refetched
    assert {entry["status"] for entry in second["entries"]} == {"integrity_error"}
    assert "no longer matches its digest" in second["entries"][0]["error"]
    assert raw_path.read_bytes() == tampered  # never replaced


def test_an_edited_artifact_is_an_integrity_error_left_untouched(tmp_path):
    panels, points = _frozen_files(
        tmp_path / "inputs", zones=("zone_a",), identities=IDENTITIES[:1]
    )
    output = tmp_path / "out"
    first = forecast_archive_panel.collect(
        _collect_args(panels[:1], points, output), client=_client(_transport())
    )
    path = output / f"{first['entries'][0]['key']}.json"
    artifact = json.loads(path.read_text())
    artifact["rows"][0]["value"] += 1.0
    path.write_text(json.dumps(artifact, sort_keys=True) + "\n")
    edited = path.read_bytes()

    second = forecast_archive_panel.collect(
        _collect_args(panels[:1], points, output, run_id="run-2"),
        client=_client(_transport()),
    )

    assert second["entries"][0]["status"] == "integrity_error"
    assert "does not re-derive" in second["entries"][0]["error"]
    assert path.read_bytes() == edited


def test_a_used_run_id_is_refused_before_any_request(tmp_path):
    panels, points = _frozen_files(tmp_path / "inputs")
    output = tmp_path / "out"
    output.mkdir()
    (output / "manifest-run-1.json").write_text("{}\n")
    calls = []

    with pytest.raises(forecast_archive_panel.ArchiveRefusal, match="never replaced"):
        forecast_archive_panel.collect(
            _collect_args(panels, points, output), client=_client(_transport(calls))
        )

    assert calls == []
    assert (output / "manifest-run-1.json").read_text() == "{}\n"


def test_collect_redacts_the_api_key_from_provider_errors(tmp_path):
    class LeakyClient:
        apikey = "archive/secret+"

        def fetch_daily_multi(self, **_kwargs):
            raise RuntimeError("request failed: apikey=archive/secret+ encoded=archive%2Fsecret%2B")

    panels, points = _frozen_files(tmp_path / "inputs")
    manifest = forecast_archive_panel.collect(
        _collect_args(panels, points, tmp_path / "out"), client=LeakyClient()
    )
    written = (tmp_path / "out" / "manifest-run-1.json").read_text()

    assert {entry["status"] for entry in manifest["entries"]} == {"provider_error"}
    for text in (json.dumps(manifest), written):
        assert "archive/secret+" not in text
        assert "archive%2Fsecret%2B" not in text


def test_main_prints_a_refusal_for_a_wrong_sampling_digest(tmp_path, capsys):
    panels, points = _frozen_files(tmp_path / "inputs")
    points.write_text("{}", encoding="utf-8")
    argv = ["collect", "--panel-manifest", str(panels[0]), "--sampling-points", str(points)]
    argv += ["--output-dir", str(tmp_path / "out"), "--run-id", "run-1"]

    assert forecast_archive_panel.main(argv) == 1
    assert json.loads(capsys.readouterr().out)["status"] == "refused"
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize(
    "removed",
    ["--zone-id", "--provider-model", "--start-date", "--end-date", "--horizon-days"],
)
def test_the_current_geometry_flags_are_gone(removed):
    with pytest.raises(SystemExit):
        forecast_archive_panel.build_parser().parse_args(["collect", removed, "x"])


# ---------------------------------------------------------------------------------------
# #193 — export: one identity at a time, verified before flattening, into the JSONL the
# delivered matrix engine already reads. End to end on synthetic inputs, no provider call.
# ---------------------------------------------------------------------------------------


def _export_args(panel, points, archive, output):
    return forecast_archive_panel.build_parser().parse_args(
        [
            "export",
            "--panel-manifest",
            str(panel),
            "--sampling-points",
            str(points),
            "--archive-dir",
            str(archive),
            "--output",
            str(output),
        ]
    )


def _collected(tmp_path, **transport):
    panels, points = _frozen_files(tmp_path / "inputs")
    archive = tmp_path / "archive"
    forecast_archive_panel.collect(
        _collect_args(panels, points, archive), client=_client(_transport(**transport))
    )
    return panels, points, archive


def test_export_end_to_end_feeds_the_matrix_engine_with_an_unshrunk_denominator(tmp_path):
    gap = (_previous_field("precipitation", 1), "2025-06-02")
    panels, points, archive = _collected(tmp_path, gap=gap)
    forecast = tmp_path / "gfs-forecast.jsonl"

    summary = forecast_archive_panel.export(_export_args(panels[0], points, archive, forecast))

    assert summary["status"] == "exported"
    assert (summary["provider"], summary["model"]) == ("gfs", "gfs_seamless")
    assert summary["expected_pairs"] == 16 and summary["rows"] == 14  # gap in both zones
    rows = [json.loads(line) for line in forecast.read_text().splitlines()]
    assert set(rows[0]) == {
        "zone_id",
        "origin_date",
        "valid_date",
        "horizon_days",
        "variable",
        "value",
    }
    reference = tmp_path / "reference.jsonl"
    reference.write_text(
        "".join(
            json.dumps({"zone_id": zone, "valid_date": day, "variable": variable, "value": 2.0})
            + "\n"
            for zone in ("zone_a", "zone_b")
            for day in ("2025-06-01", "2025-06-02")
            for variable in ("temperature_2m", "precipitation")
        )
    )
    baseline = tmp_path / "p50.json"
    baseline.write_text(
        json.dumps(
            [
                {"zone_id": zone, "variable": variable, "month": 6, "p50": 1.0}
                for zone in ("zone_a", "zone_b")
                for variable in ("temperature_2m", "precipitation")
            ]
        )
    )
    configuration = json.loads(panels[0].read_text())["configuration"]
    matrix_path = tmp_path / "matrix.json"

    status = forecast_skill_matrix.main(
        [
            "--forecast",
            str(forecast),
            "--reference",
            str(reference),
            "--baseline-p50",
            str(baseline),
            "--panel-manifest",
            str(panels[0]),
            "--provider",
            "gfs",
            "--model",
            "gfs_seamless",
            "--configuration",
            json.dumps(configuration),
            "--training-end",
            "2024-12-31",
            "--holdout-start",
            "2025-01-01",
            "--holdout-end",
            "2025-12-31",
            "--output",
            str(matrix_path),
            "--allow-dirty",
        ]
    )

    assert status == 0
    matrix = json.loads(matrix_path.read_text())
    census = matrix["census"]
    assert census["expected_pairs"] == 16
    assert census["scored_pairs"] == 14
    assert census["exclusions"]["not_submitted"] == 2
    assert matrix.get("confidence") is None


def test_each_identity_exports_separately_and_identically_keyed(tmp_path):
    panels, points, archive = _collected(tmp_path)
    gfs, aifs = tmp_path / "gfs.jsonl", tmp_path / "aifs.jsonl"

    forecast_archive_panel.export(_export_args(panels[0], points, archive, gfs))
    forecast_archive_panel.export(_export_args(panels[1], points, archive, aifs))

    def keys(path):
        return [
            tuple(json.loads(line)[field] for field in ("zone_id", "valid_date", "variable"))
            for line in path.read_text().splitlines()
        ]

    assert keys(gfs) == keys(aifs) and len(keys(gfs)) == 16


def test_an_export_target_that_exists_is_refused(tmp_path):
    panels, points, archive = _collected(tmp_path)
    output = tmp_path / "gfs.jsonl"
    output.write_text("keep\n")

    with pytest.raises(forecast_archive_panel.ArchiveRefusal, match="never replaced"):
        forecast_archive_panel.export(_export_args(panels[0], points, archive, output))
    assert output.read_text() == "keep\n"


def test_an_export_over_a_tampered_archive_is_refused_whole(tmp_path):
    panels, points, archive = _collected(tmp_path)
    artifact = json.loads(sorted(archive.glob("zone_b__gfs__*.json"))[0].read_text())
    raw = archive / "raw" / f"{artifact['raw_sha256']}.json"
    raw.write_bytes(raw.read_bytes() + b" ")
    output = tmp_path / "gfs.jsonl"

    with pytest.raises(forecast_archive_panel.ArchiveRefusal, match="no longer matches"):
        forecast_archive_panel.export(_export_args(panels[0], points, archive, output))
    assert not output.exists()


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"provider": "aifs"}, "not gfs/gfs_seamless"),
        ({"model": "other_model"}, "not gfs/gfs_seamless"),
        ({"valid_date": "2025-07-01", "origin_date": "2025-06-30"}, "outside the frozen panel"),
        ({"zone_id": "zone_z"}, "outside the frozen panel"),
        ({"horizon_days": 3}, "outside the frozen panel"),
    ],
)
def test_a_row_outside_the_identity_or_panel_is_refused(tmp_path, monkeypatch, change, message):
    panels, points, archive = _collected(tmp_path)
    verify = forecast_archive_panel._verify_artifact

    def tampered(*args):
        artifact = verify(*args)
        artifact["rows"][0].update(change)
        return artifact

    monkeypatch.setattr(forecast_archive_panel, "_verify_artifact", tampered)
    output = tmp_path / "gfs.jsonl"
    with pytest.raises(forecast_archive_panel.ArchiveRefusal, match=message):
        forecast_archive_panel.export(_export_args(panels[0], points, archive, output))
    assert not output.exists()


def test_a_duplicate_forecast_key_is_refused(tmp_path, monkeypatch):
    panels, points, archive = _collected(tmp_path)
    verify = forecast_archive_panel._verify_artifact

    def doubled(*args):
        artifact = verify(*args)
        artifact["rows"].append(dict(artifact["rows"][0]))
        return artifact

    monkeypatch.setattr(forecast_archive_panel, "_verify_artifact", doubled)
    output = tmp_path / "gfs.jsonl"
    with pytest.raises(forecast_archive_panel.ArchiveRefusal, match="duplicate forecast key"):
        forecast_archive_panel.export(_export_args(panels[0], points, archive, output))
    assert not output.exists()


def test_an_uncollected_artifact_is_listed_not_invented(tmp_path):
    panels, points, archive = _collected(tmp_path)
    removed = sorted(archive.glob("zone_b__gfs__*__3.json"))[0]
    removed.unlink()
    output = tmp_path / "gfs.jsonl"

    summary = forecast_archive_panel.export(_export_args(panels[0], points, archive, output))

    assert summary["missing_artifacts"] == [removed.name]
    assert summary["rows"] == 12 and summary["expected_pairs"] == 16


def test_the_same_inputs_and_clock_reproduce_the_same_bytes(tmp_path):
    """Repeatability: two collections from one fixture are byte-identical, export included."""
    panels, points = _frozen_files(tmp_path / "inputs")
    gap = (_previous_field("temperature_2m", 3), "2025-06-01")
    for name in ("first", "second"):
        forecast_archive_panel.collect(
            _collect_args(panels, points, tmp_path / name), client=_client(_transport(gap=gap))
        )
        forecast_archive_panel.export(
            _export_args(panels[1], points, tmp_path / name, tmp_path / name / "aifs.jsonl")
        )

    first, second = _snapshot(tmp_path / "first"), _snapshot(tmp_path / "second")
    assert first == second
    assert "aifs.jsonl" in first and "manifest-run-1.json" in first


# ---------------------------------------------------------------------------------------
# #193 review — a refused response keeps its bytes (a wrong unit guess at the preflight must
# leave something to inspect), and a run with any unusable entry exits non-zero.
# ---------------------------------------------------------------------------------------


def _wrong_unit(transport):
    def wrapped(params):
        payload = json.loads(transport(params))
        field = _previous_field("temperature_2m", 1)
        if field in payload["hourly_units"]:
            payload["hourly_units"][field] = "°F"
        return json.dumps(payload).encode("utf-8")

    return wrapped


def test_a_refused_response_keeps_its_raw_bytes_and_provenance(tmp_path):
    panels, points = _frozen_files(
        tmp_path / "inputs", zones=("zone_a",), horizons=(1,), identities=IDENTITIES[:1]
    )
    output = tmp_path / "out"
    sent = []

    def recording(params):
        body = _wrong_unit(_transport())(params)
        sent.append(body)
        return body

    manifest = forecast_archive_panel.collect(
        _collect_args(panels, points, output), client=_client(recording)
    )

    (entry,) = manifest["entries"]
    assert entry["status"] == "refused"
    assert "must use '°C'" in entry["error"]
    raw = output / "raw" / f"{entry['raw_sha256']}.json"
    assert raw.read_bytes() in sent
    provenance = json.loads(raw.with_name(entry["provenance"]).read_text())
    assert provenance["returned"]["hourly_units"]["temperature_2m_previous_day1"] == "°F"
    assert not (output / f"{entry['key']}.json").exists()


@pytest.mark.parametrize(
    ("transport", "code"),
    [(_transport, 0), (lambda: _wrong_unit(_transport()), 1)],
)
def test_main_exits_non_zero_when_an_entry_is_not_usable(tmp_path, monkeypatch, transport, code):
    panels, points = _frozen_files(tmp_path / "inputs", zones=("zone_a",), horizons=(1,))
    monkeypatch.setattr(
        forecast_archive_panel,
        "PreviousRunsArchiveClient",
        lambda: _client(transport()),
    )
    argv = ["collect", "--sampling-points", str(points), "--output-dir", str(tmp_path / "out")]
    for panel in panels:
        argv += ["--panel-manifest", str(panel)]

    assert forecast_archive_panel.main([*argv, "--run-id", "run-1"]) == code


def test_main_exits_non_zero_on_a_provider_error(tmp_path, monkeypatch):
    def offline(_params):
        raise RuntimeError("offline test transport")

    panels, points = _frozen_files(tmp_path / "inputs", zones=("zone_a",), horizons=(1,))
    monkeypatch.setattr(
        forecast_archive_panel, "PreviousRunsArchiveClient", lambda: _client(offline)
    )
    argv = ["collect", "--panel-manifest", str(panels[0]), "--sampling-points", str(points)]
    argv += ["--output-dir", str(tmp_path / "out"), "--run-id", "run-1"]

    assert forecast_archive_panel.main(argv) == 1


@pytest.mark.parametrize("cell", [{"nested": 1}, [1.0, 2.0]], ids=["object", "array"])
def test_a_non_numeric_hourly_value_is_refused_and_its_bytes_kept(tmp_path, cell):
    panels, points = _frozen_files(
        tmp_path / "inputs", zones=("zone_a",), horizons=(1,), identities=IDENTITIES[:1]
    )
    sent = []

    def malformed(params):
        payload = json.loads(_transport()(params))
        payload["hourly"][_previous_field("temperature_2m", 1)][0] = cell
        sent.append(json.dumps(payload).encode("utf-8"))
        return sent[-1]

    output = tmp_path / "out"
    manifest = forecast_archive_panel.collect(
        _collect_args(panels, points, output), client=_client(malformed)
    )

    (entry,) = manifest["entries"]
    assert entry["status"] == "refused"
    assert "non-numeric" in entry["error"]
    assert (output / "raw" / f"{entry['raw_sha256']}.json").read_bytes() == sent[0]
    provenance = json.loads((output / "raw" / entry["provenance"]).read_text(encoding="utf-8"))
    assert provenance["raw_sha256"] == entry["raw_sha256"]
    assert provenance["requested_model"] == "gfs_seamless"
    assert not (output / entry["output"]).exists()


def test_identical_bytes_keep_one_raw_file_but_each_request_its_own_provenance(tmp_path):
    """The synthetic transport ignores the model, so both identities get identical bytes."""
    panels, points = _frozen_files(tmp_path / "inputs", zones=("zone_a",), horizons=(1,))
    output = tmp_path / "out"
    manifest = forecast_archive_panel.collect(
        _collect_args(panels, points, output), client=_client(_transport())
    )

    artifacts = {
        (entry["provider"], entry["model"]): json.loads(
            (output / entry["output"]).read_text(encoding="utf-8")
        )
        for entry in manifest["entries"]
    }
    assert len({artifact["raw_sha256"] for artifact in artifacts.values()}) == 1
    assert len(list((output / "raw").glob("*.provenance.json"))) == 2
    for (provider, model), artifact in artifacts.items():
        provenance = json.loads(
            (output / "raw" / artifact["provenance"]).read_text(encoding="utf-8")
        )
        assert provenance["raw_sha256"] == artifact["raw_sha256"]
        assert (provenance["provider"], provenance["requested_model"]) == (provider, model)
        assert provenance["params"]["models"] == model
        assert provenance["zone_id"] == "zone_a"
        assert provenance["requested_coordinates"] == {"latitude": 40.0, "longitude": 2.0}


def test_a_provenance_naming_another_request_is_an_integrity_error(tmp_path):
    panels, points = _frozen_files(tmp_path / "inputs", zones=("zone_a",), horizons=(1,))
    output = tmp_path / "out"
    first = forecast_archive_panel.collect(
        _collect_args(panels, points, output), client=_client(_transport())
    )
    gfs, aifs = (
        json.loads((output / entry["output"]).read_text(encoding="utf-8"))
        for entry in first["entries"]
    )
    # Swap the gfs request's provenance for the aifs one: same digest, another request.
    (output / "raw" / gfs["provenance"]).write_bytes(
        (output / "raw" / aifs["provenance"]).read_bytes()
    )

    second = forecast_archive_panel.collect(
        _collect_args(panels, points, output, run_id="run-2"), client=_client(_transport())
    )

    statuses = {entry["provider"]: entry["status"] for entry in second["entries"]}
    assert statuses == {"gfs": "integrity_error", "aifs": "resumed"}


def test_identical_bodies_keep_each_request_identity_through_collect_resume_and_export(tmp_path):
    """Review item 1: identical bytes, distinct model/params/retrieval time, kept end to end."""
    panels, points = _frozen_files(tmp_path / "inputs", zones=("zone_a",), horizons=(1,))
    ticks = iter(range(1000))
    ticking = PreviousRunsArchiveClient(
        apikey="",
        transport=_transport(),
        now=lambda: FIXED_NOW + timedelta(minutes=next(ticks)),
    )
    output = tmp_path / "out"

    first = forecast_archive_panel.collect(_collect_args(panels, points, output), client=ticking)

    def provenance_of(name):
        return json.loads((output / "raw" / name).read_text(encoding="utf-8"))

    artifacts = {
        entry["model"]: json.loads((output / entry["output"]).read_text(encoding="utf-8"))
        for entry in first["entries"]
    }
    assert len({artifact["raw_sha256"] for artifact in artifacts.values()}) == 1
    requests = {model: provenance_of(a["provenance"]) for model, a in artifacts.items()}
    assert {model: p["requested_model"] for model, p in requests.items()} == {
        "gfs_seamless": "gfs_seamless",
        "ecmwf_aifs025_single": "ecmwf_aifs025_single",
    }
    assert {model: p["params"]["models"] for model, p in requests.items()} == {
        "gfs_seamless": "gfs_seamless",
        "ecmwf_aifs025_single": "ecmwf_aifs025_single",
    }
    assert len({p["retrieved_at"] for p in requests.values()}) == 2

    # Resume: both artifacts verify against their OWN provenance; nothing is rewritten.
    before = _snapshot(output)
    calls = []
    resumed = forecast_archive_panel.collect(
        _collect_args(panels, points, output, run_id="run-2"), client=_client(_transport(calls))
    )
    assert calls == []
    assert {entry["status"] for entry in resumed["entries"]} == {"resumed"}
    assert {k: v for k, v in _snapshot(output).items() if k != "manifest-run-2.json"} == before

    # Export: each identity is exported from, and names, its own request's provenance.
    for panel, (provider, model) in zip(panels, IDENTITIES, strict=True):
        summary = forecast_archive_panel.export(
            _export_args(panel, points, output, tmp_path / f"{provider}.jsonl")
        )
        assert summary["provenance"] == [artifacts[model]["provenance"]]
        assert provenance_of(summary["provenance"][0])["requested_model"] == model
        assert (
            provenance_of(summary["provenance"][0])["retrieved_at"]
            == (requests[model]["retrieved_at"])
        )
