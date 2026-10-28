"""Network-free tests for scripts/unified_pilot_panel.py."""

from __future__ import annotations

import asyncio
import contextlib
import importlib.util
import json
from pathlib import Path

import pytest

from src.utils.s3_endpoint import S3EndpointConfigurationError

_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "unified_pilot_panel.py"
_spec = importlib.util.spec_from_file_location("unified_pilot_panel", _SCRIPT_PATH)
panel = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(panel)

pytestmark = pytest.mark.usefixtures("allow_local_parquet_paths")


def _args(*values: str):
    return panel.build_parser().parse_args(list(values))


async def _must_not_run(**_kwargs):
    raise AssertionError("preflight must not call climate_enrich_ndvi")


async def _no_sleep(_seconds: float) -> None:
    return None


class TestUnifiedPilotPanel:
    def test_preflight_default_pilot_set_resolves_s3_paths(self, tmp_path, monkeypatch):
        monkeypatch.setattr(panel.settings, "S3_BUCKET", "deposium")
        args = _args("--preflight-only", "--output-dir", tmp_path.as_posix())

        result = asyncio.run(panel.run_panel(args, enrich_runner=_must_not_run, sleeper=_no_sleep))

        assert result["ok"] is True
        assert result["preflight_only"] is True
        assert result["zone_ids"] == list(panel.PILOT_ZONE_IDS)
        assert result["run_manifest_uri"].startswith("s3://deposium/climate/runs/")
        assert result["run_manifest_written"] is None
        assert all(zone["preflight_only"] for zone in result["zones"])
        assert result["zones"][0]["era5_source"].startswith("s3://deposium/climate/era5/")
        assert result["zones"][0]["unified_output"].startswith("s3://deposium/climate/unified/")
        assert result["window"] == {
            "start_date": panel.DEFAULT_START_DATE,
            "end_date": panel.DEFAULT_END_DATE,
        }

        summary = json.loads((tmp_path / "unified-pilot-summary.json").read_text())
        manifest = json.loads((tmp_path / "unified-pilot-manifest.json").read_text())
        assert summary["kind"] == "unified_pilot_panel_summary"
        assert manifest["kind"] == "unified_pilot_panel_manifest"

    def test_holdout_preflight_resolves_holdout_s3_paths(self, tmp_path, monkeypatch):
        monkeypatch.setattr(panel.settings, "S3_BUCKET", "deposium")
        args = _args(
            "--preflight-only",
            "--zone-id",
            "montpellier",
            "--artifact-scope",
            "holdout",
            "--start-date",
            "2021-01-01",
            "--end-date",
            "2025-12-31",
            "--output-dir",
            tmp_path.as_posix(),
        )

        result = asyncio.run(panel.run_panel(args, enrich_runner=_must_not_run, sleeper=_no_sleep))

        holdout_era5 = "s3://deposium/climate/holdout/era5/2021_2025/montpellier.parquet"
        holdout_unified = "s3://deposium/climate/holdout/unified/2021_2025/montpellier.parquet"
        assert result["ok"] is True
        assert result["artifact_scope"] == "holdout"
        assert result["artifact_window_label"] == "2021_2025"
        assert result["zones"][0]["era5_source"] == holdout_era5
        assert result["zones"][0]["unified_output"] == holdout_unified

        manifest = json.loads((tmp_path / "unified-pilot-manifest.json").read_text())
        assert manifest["artifact_scope"] == "holdout"
        assert manifest["artifact_window_label"] == "2021_2025"
        assert manifest["artifacts"][0]["era5_source"] == holdout_era5
        assert manifest["artifacts"][0]["unified_output"] == holdout_unified

    def test_local_run_calls_real_tool_boundary_with_resolved_paths(self, tmp_path, monkeypatch):
        monkeypatch.setattr(panel.settings, "S3_BUCKET", "")
        era5_template = (tmp_path / "era5" / "{zone_id}.parquet").as_posix()
        args = _args(
            "--zone-id",
            "montpellier",
            "--storage-mode",
            "local",
            "--era5-source-template",
            era5_template,
            "--output-dir",
            tmp_path.as_posix(),
            "--no-run-manifest",
        )
        calls = []

        async def _fake_enrich(**kwargs):
            calls.append(kwargs)
            return {
                "zone_id": kwargs["zone_id"],
                "output": kwargs["output_path"],
                "statistics": {
                    "n_era5_days": 365,
                    "n_ndvi_observations": 24,
                    "n_matched": 21,
                    "images_searched": 24,
                },
                "metadata": {"era5_source": kwargs["era5_source"]},
            }

        result = asyncio.run(panel.run_panel(args, enrich_runner=_fake_enrich, sleeper=_no_sleep))

        assert result["ok"] is True
        assert result["run_manifest_uri"] is None
        assert len(calls) == 1
        assert calls[0]["zone_id"] == "montpellier"
        assert calls[0]["era5_source"] == (tmp_path / "era5" / "montpellier.parquet").as_posix()
        assert calls[0]["output_path"].endswith("montpellier-unified-2020-07-01_2020-07-31.parquet")
        assert result["zones"][0]["stage"]["n_matched"] == 21

    def test_canonical_s3_preflight_rejects_recent_window_without_recent_era5(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.setattr(panel.settings, "S3_BUCKET", "deposium")
        args = _args(
            "--preflight-only",
            "--zone-id",
            "montpellier",
            "--start-date",
            "2023-01-01",
            "--end-date",
            "2023-12-31",
            "--output-dir",
            tmp_path.as_posix(),
        )

        result = asyncio.run(panel.run_panel(args, enrich_runner=_must_not_run, sleeper=_no_sleep))

        assert result["ok"] is False
        assert result["failed_zone_ids"] == ["montpellier"]
        assert result["zones"][0]["failed_stage"] == "preflight"
        assert (
            "canonical ERA5 artifacts cover 1991-01-01..2020-12-31" in result["zones"][0]["error"]
        )

    def test_canonical_recent_window_allows_explicit_era5_template(self, tmp_path, monkeypatch):
        monkeypatch.setattr(panel.settings, "S3_BUCKET", "deposium")
        args = _args(
            "--preflight-only",
            "--zone-id",
            "montpellier",
            "--start-date",
            "2023-01-01",
            "--end-date",
            "2023-12-31",
            "--era5-source-template",
            "s3://deposium/climate/custom-era5/{zone_id}.parquet",
            "--output-dir",
            tmp_path.as_posix(),
        )

        result = asyncio.run(panel.run_panel(args, enrich_runner=_must_not_run, sleeper=_no_sleep))

        assert result["ok"] is True
        assert result["zones"][0]["era5_source"] == (
            "s3://deposium/climate/custom-era5/montpellier.parquet"
        )

    def test_fail_fast_stops_after_first_tool_error(self, tmp_path, monkeypatch):
        monkeypatch.setattr(panel.settings, "S3_BUCKET", "")
        args = _args(
            "--zone-id",
            "montpellier",
            "marseille",
            "--storage-mode",
            "local",
            "--era5-source-template",
            (tmp_path / "era5" / "{zone_id}.parquet").as_posix(),
            "--output-dir",
            tmp_path.as_posix(),
            "--no-run-manifest",
            "--fail-fast",
        )
        calls = []

        async def _fake_enrich(**kwargs):
            calls.append(kwargs["zone_id"])
            return {"error": "NDVI collection failed: STAC unavailable"}

        result = asyncio.run(panel.run_panel(args, enrich_runner=_fake_enrich, sleeper=_no_sleep))

        assert result["ok"] is False
        assert result["stop_reason"] == "fail_fast"
        assert result["completed_zone_ids"] == ["montpellier"]
        assert calls == ["montpellier"]

    def test_s3_run_manifest_payload_advertises_unified_outputs(self, tmp_path, monkeypatch):
        monkeypatch.setattr(panel.settings, "S3_BUCKET", "deposium")
        args = _args("--zone-id", "montpellier", "--output-dir", tmp_path.as_posix())
        written = []

        async def _fake_enrich(**kwargs):
            return {
                "zone_id": kwargs["zone_id"],
                "output": kwargs["output_path"],
                "statistics": {
                    "n_era5_days": 365,
                    "n_ndvi_observations": 24,
                    "n_matched": 21,
                    "images_searched": 24,
                },
                "metadata": {},
            }

        def _capture_manifest(uri, payload):
            written.append((uri, payload))

        result = asyncio.run(
            panel.run_panel(
                args,
                enrich_runner=_fake_enrich,
                sleeper=_no_sleep,
                run_manifest_writer=_capture_manifest,
            )
        )

        assert result["ok"] is True
        assert result["run_manifest_written"] is True
        assert len(written) == 1
        uri, payload = written[0]
        assert uri == result["run_manifest_uri"]
        assert payload["tool_or_script"] == "scripts/unified_pilot_panel.py"
        assert payload["checks"]["run_manifest_written"] is True
        assert payload["outputs"] == [
            {
                "kind": "unified",
                "zone_id": "montpellier",
                "uri": "s3://deposium/climate/unified/montpellier.parquet",
                "n_matched": 21,
            }
        ]

    def test_s3_preflight_main_returns_zero_with_planned_manifest(self, tmp_path, monkeypatch):
        monkeypatch.setattr(panel.settings, "S3_BUCKET", "deposium")

        code = panel.main(
            [
                "--preflight-only",
                "--zone-id",
                "montpellier",
                "--output-dir",
                tmp_path.as_posix(),
            ]
        )

        assert code == 0

    def test_code_ref_falls_back_to_runtime_env_without_git(self, monkeypatch):
        monkeypatch.setattr(panel, "_git_output", lambda *_args, **_kwargs: None)
        monkeypatch.setenv("GIT_COMMIT", "abc123")
        monkeypatch.setenv("GIT_BRANCH", "codex/b4")

        assert panel._code_ref() == {
            "repo": "deposium_geoai",
            "commit": "abc123",
            "branch": "codex/b4",
            "dirty": None,
        }

    def test_invalid_window_fails_before_path_or_run_id_materialization(self, tmp_path):
        args = _args(
            "--preflight-only",
            "--zone-id",
            "montpellier",
            "--start-date",
            "2023-02-01",
            "--end-date",
            "2023-01-01",
            "--storage-mode",
            "local",
            "--era5-source-template",
            (tmp_path / "era5" / "{zone_id}.parquet").as_posix(),
            "--output-dir",
            tmp_path.as_posix(),
            "--no-run-manifest",
        )

        try:
            asyncio.run(panel.run_panel(args, enrich_runner=_must_not_run, sleeper=_no_sleep))
        except ValueError as exc:
            assert "precedes start_date" in str(exc)
        else:
            raise AssertionError(
                "invalid date window should fail before preflight paths are written"
            )

    def test_run_manifest_records_the_object_store_it_wrote_to(self, monkeypatch):
        """Same block, same shape as the B3/ERA5 panels — a consumer reads it panel-agnostically."""
        monkeypatch.setattr(panel.settings, "S3_ENDPOINT", "http://localhost:9100")
        monkeypatch.setattr(panel.settings, "S3_BUCKET", "deposium")
        monkeypatch.setattr(panel.settings, "S3_FORCE_PATH_STYLE", True)

        manifest = panel._run_manifest_payload({"run_id": "env-proof", "zones": []})

        assert manifest["environment"] == {
            "s3_endpoint": "http://localhost:9100",
            "s3_endpoint_credentials_removed": False,
            "s3_bucket": "deposium",
            "s3_force_path_style": True,
        }

    def test_unset_endpoint_is_recorded_as_null(self, monkeypatch):
        # Only the endpoint half: manifest publication is gated on a non-empty S3_BUCKET, so a
        # published manifest can never carry a null bucket.
        monkeypatch.setattr(panel.settings, "S3_ENDPOINT", "")
        monkeypatch.setattr(panel.settings, "S3_BUCKET", "deposium")

        manifest = panel._run_manifest_payload({"run_id": "env-proof", "zones": []})

        assert manifest["environment"]["s3_endpoint"] is None

    def test_credentials_embedded_in_the_endpoint_never_reach_the_manifest(self, monkeypatch):
        """`S3_ENDPOINT` may legally carry userinfo, and this manifest is published for handoff."""
        monkeypatch.setattr(
            panel.settings, "S3_ENDPOINT", "http://rustfs-admin:sup3r-s3cret@mesh.internal:9000"
        )
        monkeypatch.setattr(panel.settings, "S3_BUCKET", "deposium")

        manifest = panel._run_manifest_payload({"run_id": "env-proof", "zones": []})
        serialized = json.dumps(manifest, default=str)

        assert "sup3r-s3cret" not in serialized
        assert "rustfs-admin" not in serialized
        # Refused whole rather than stripped: the accept grammar offers no partial reading.
        assert manifest["environment"]["s3_endpoint"] is None
        assert manifest["environment"]["s3_endpoint_credentials_removed"] is True

    def test_the_stac_source_is_published_as_an_address_too(self, monkeypatch):
        """`inputs[].source` is `STAC_URL` verbatim; the redactor only knows `S3_ENDPOINT`."""
        monkeypatch.setattr(panel.settings, "S3_ENDPOINT", "http://localhost:9100")
        monkeypatch.setattr(panel.settings, "S3_BUCKET", "deposium")
        monkeypatch.setattr(
            panel.settings, "STAC_URL", "https://stac-reader:stac-s3cret@stac.internal/v1"
        )

        manifest = panel._run_manifest_payload({"run_id": "env-proof", "zones": []})
        serialized = json.dumps(manifest, default=str)

        assert "stac-s3cret" not in serialized
        assert "stac-reader" not in serialized
        stac_input = next(i for i in manifest["inputs"] if i["kind"] == "sentinel2_stac_ndvi")
        assert stac_input["source"] == "https://stac.internal/v1"
        assert stac_input["source_reductions"] == ["userinfo"]

    def test_a_stac_query_is_dropped_and_the_manifest_SAYS_so(self, monkeypatch):
        """This panel needs its own pin: it writes `source`, not `endpoint`, and had its own
        `sanitize_endpoint` call. A query on a STAC URL is dropped — a `?token=` is how a
        catalogue takes a credential on the URL — and `source_reductions` is what keeps that from
        being a silent misstatement."""
        monkeypatch.setattr(panel.settings, "S3_ENDPOINT", "http://localhost:9100")
        monkeypatch.setattr(panel.settings, "S3_BUCKET", "deposium")
        monkeypatch.setattr(panel.settings, "STAC_URL", "https://stac.internal/v1?token=abc")

        manifest = panel._run_manifest_payload({"run_id": "env-proof", "zones": []})

        stac_input = next(i for i in manifest["inputs"] if i["kind"] == "sentinel2_stac_ndvi")
        assert stac_input["source"] == "https://stac.internal/v1"
        assert stac_input["source_reductions"] == ["query"]
        assert "token=abc" not in json.dumps(manifest, default=str)

    def test_an_untouched_stac_url_publishes_an_empty_reduction_list(self, monkeypatch):
        monkeypatch.setattr(panel.settings, "S3_ENDPOINT", "http://localhost:9100")
        monkeypatch.setattr(panel.settings, "S3_BUCKET", "deposium")
        monkeypatch.setattr(panel.settings, "STAC_URL", "https://earth-search.aws.element84.com/v1")

        manifest = panel._run_manifest_payload({"run_id": "env-proof", "zones": []})

        stac_input = next(i for i in manifest["inputs"] if i["kind"] == "sentinel2_stac_ndvi")
        assert stac_input["source"] == "https://earth-search.aws.element84.com/v1"
        assert stac_input["source_reductions"] == []

    def test_run_manifest_never_carries_a_credential(self, monkeypatch):
        monkeypatch.setattr(panel.settings, "S3_ENDPOINT", "http://localhost:9100")
        monkeypatch.setattr(panel.settings, "S3_BUCKET", "deposium")
        monkeypatch.setattr(panel.settings, "S3_ACCESS_KEY", "deposium-dev-access-key")
        monkeypatch.setattr(panel.settings, "S3_SECRET_KEY", "deposium-dev-secret-key")

        serialized = json.dumps(
            panel._run_manifest_payload({"run_id": "env-proof", "zones": []}), default=str
        )

        assert "deposium-dev-access-key" not in serialized
        assert "deposium-dev-secret-key" not in serialized
        assert "localhost:9100" in serialized


def test_a_userinfo_endpoint_is_refused_before_any_manifest_is_published(tmp_path, monkeypatch):
    """This panel owns its own copy of `_write_s3_json`, so it needs its own boundary test.

    The credential-in-`errors[].error` half is covered end to end below, through the real writer,
    on the shape that actually deploys (`S3_ACCESS_KEY`/`S3_SECRET_KEY` supplied separately). This
    is the other half, and it is now closed one layer earlier than a redactor: a userinfo-bearing
    `S3_ENDPOINT` matches no accept-grammar production, so `duckdb_session` refuses to open and NO
    object is written. There used to be a published manifest here carrying a stripped address; the
    password cannot be in a manifest that does not exist.
    """
    destination = tmp_path / "manifest.json"
    # The URI guard confines writes to s3://{bucket}/climate/runs/; only the guard is relaxed.
    monkeypatch.setattr(panel, "_validate_run_manifest_uri", lambda uri: None)
    monkeypatch.setattr(
        panel.settings, "S3_ENDPOINT", "http://rustfs-admin:sup3r-s3cret@mesh.internal:9000"
    )

    with pytest.raises(S3EndpointConfigurationError):
        panel._write_s3_json(destination.as_posix(), {"schema_version": 1})

    assert not destination.exists()


# --------------------------------------------------------------------------- #
# End to end: a credential in an error message never reaches the published object
# --------------------------------------------------------------------------- #


class _RecordingConnection:
    """Stands in for the DuckDB connection `_write_s3_json` opens. Records the COPY binding."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, list | None]] = []

    def execute(self, sql: str, params: list | None = None) -> None:
        self.calls.append((sql, params))


def test_a_credential_in_a_zone_error_never_reaches_the_published_manifest(tmp_path, monkeypatch):
    """The end-to-end guarantee, through the REAL publication path.

    `run_panel` -> `_run_manifest_payload` -> `_write_s3_json` -> `COPY … TO 's3://…'`. Only the
    engine is stubbed, and it is stubbed at the last possible point: what the fake records is
    exactly the list of values DuckDB would have bound and written into the bucket.

    The message is the one httpfs actually produces against the dev store — it quotes the
    configured `s3_access_key_id` inside a 403 body — and `import-zip` republishes this manifest
    into the DESTINATION bucket, the one handed to another party.
    """
    access_key = "deposium-dev"
    secret_key = "deposium-dev-secret"
    monkeypatch.setattr(panel.settings, "S3_BUCKET", "deposium")
    monkeypatch.setattr(panel.settings, "S3_ENDPOINT", "http://rustfs:9000")
    monkeypatch.setattr(panel.settings, "S3_ACCESS_KEY", access_key)
    monkeypatch.setattr(panel.settings, "S3_SECRET_KEY", secret_key)

    connection = _RecordingConnection()

    @contextlib.contextmanager
    def _fake_session(_db_path, *, load_httpfs=False):
        yield connection

    monkeypatch.setattr(panel, "duckdb_session", _fake_session)

    async def _failing_enrich(**kwargs):
        return {
            "error": (
                "IOException: IO Error: Unable to connect to URL "
                "'http://rustfs:9000/deposium/climate/z.parquet': 403 (Forbidden) "
                f"<AWSAccessKeyId>{access_key}</AWSAccessKeyId> "
                f"signature for {secret_key} rejected"
            )
        }

    args = _args("--zone-id", "montpellier", "--output-dir", tmp_path.as_posix())
    result = asyncio.run(panel.run_panel(args, enrich_runner=_failing_enrich, sleeper=_no_sleep))

    copies = [(sql, params) for sql, params in connection.calls if sql.startswith("COPY ")]
    assert len(copies) == 1, "the real _write_s3_json must have run"
    sql, values = copies[0]
    # The object really was going to the confined run-manifest prefix.
    assert result["run_manifest_uri"] in sql
    published = json.dumps(values, default=str)

    assert access_key not in published
    assert secret_key not in published
    assert "[REDACTED]" in published
    # The address is not a secret and must survive, or the manifest loses its provenance.
    assert "rustfs:9000" in published
    # And the structural fields are intact: `deposium-dev` is a substring of nothing here, but
    # the run id and the URI must still be the ones the run actually used.
    assert result["run_id"] in published
    assert "s3://deposium/climate/runs/" in sql


def test_the_writer_failure_line_on_stderr_is_redacted_too(monkeypatch, capsys):
    """`panel` is the IN-MEMORY dict, so it never went through either JSON writer.

    `main` prints `run_manifest_errors=` verbatim out of it, and that value is writer-exception
    text. Measured with the repo's own dev credentials before the fix:

        stderr LEAK[deposium-dev-secret] = True

    A CI log keeps that line exactly as long as a file on disk keeps its copy.
    """
    access_key = "deposium-dev"
    secret_key = "deposium-dev-secret"
    monkeypatch.setattr(panel.settings, "S3_ACCESS_KEY", access_key)
    monkeypatch.setattr(panel.settings, "S3_SECRET_KEY", secret_key)
    monkeypatch.setattr(panel.settings, "S3_ENDPOINT", "http://rustfs:9000")

    published = {
        "ok": False,
        "n_zones_completed": 0,
        "n_zones_requested": 1,
        "summary_path": "summary.json",
        "failed_zone_ids": [],
        "run_manifest_uri": "s3://deposium/climate/runs/r/manifest.json",
        "run_manifest_written": False,
        "run_manifest_errors": [
            {
                "uri": "s3://deposium/climate/runs/r/manifest.json",
                "error": (
                    "IOException: HTTP 403 "
                    f"<AWSAccessKeyId>{access_key}</AWSAccessKeyId> "
                    f"signature for {secret_key} rejected"
                ),
            }
        ],
    }
    monkeypatch.setattr(panel.asyncio, "run", lambda _coro: published)

    exit_code = panel.main(["--zone-id", "montpellier"])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert secret_key not in captured.err
    assert access_key not in captured.err
    assert "[REDACTED]" in captured.err
    # The URI is provenance and survives.
    assert "s3://deposium/climate/runs/r/manifest.json" in captured.err


def test_the_same_credential_never_reaches_the_local_run_manifest_either(tmp_path, monkeypatch):
    """The S3 copy was scrubbed and the LOCAL copy of the same payload was not.

    `_write_s3_json` redacted; `_write_json` was a bare `path.write_text(json.dumps(...))`, and
    `run_panel` writes the panel summary and the local manifest out of the same payload the
    remote writer is handed. Measured before the fix, with the repo's own dev credentials:

        local file bytes: LEAK[deposium-dev-secret] = True
    """
    access_key = "deposium-dev"
    secret_key = "deposium-dev-secret"
    monkeypatch.setattr(panel.settings, "S3_BUCKET", "deposium")
    monkeypatch.setattr(panel.settings, "S3_ENDPOINT", "http://rustfs:9000")
    monkeypatch.setattr(panel.settings, "S3_ACCESS_KEY", access_key)
    monkeypatch.setattr(panel.settings, "S3_SECRET_KEY", secret_key)

    async def _failing_enrich(**kwargs):
        return {
            "error": (
                "IOException: IO Error: Unable to connect to URL "
                "'http://rustfs:9000/deposium/climate/z.parquet': 403 (Forbidden) "
                f"<AWSAccessKeyId>{access_key}</AWSAccessKeyId> "
                f"signature for {secret_key} rejected"
            )
        }

    args = _args(
        "--zone-id",
        "montpellier",
        "--output-dir",
        tmp_path.as_posix(),
        "--storage-mode",
        "local",
    )
    result = asyncio.run(panel.run_panel(args, enrich_runner=_failing_enrich, sleeper=_no_sleep))

    written = sorted(tmp_path.rglob("*.json"))
    assert written, "the panel must have written local JSON artifacts"
    for path in written:
        text = path.read_text()
        assert secret_key not in text, f"{path.name} leaks the configured S3 secret key"
        assert access_key not in text, f"{path.name} leaks the configured S3 access key"
    joined = "\n".join(p.read_text() for p in written)
    assert "[REDACTED]" in joined
    assert "rustfs:9000" in joined
    assert result["run_id"] in joined


# --------------------------------------------------------------------------- #
# --ndvi-source-template: the stored B3 series through the wrapper (#133)
# --------------------------------------------------------------------------- #

_B3_ZONE = "montpellier_metropole"


def _write_era5(path: Path, zone_id: str, days: list[str]) -> None:
    """A minimal ERA5-shaped GeoParquet, written by the tool's own writer."""
    from src.tools.climate import _write_geoparquet

    rows = [
        {
            "date": day,
            "zone_id": zone_id,
            "lat": 43.6,
            "lon": 3.86,
            "wkt": "POINT(3.86 43.6)",
            "temperature_2m": 10.0 + index,
            "precipitation": 0.0,
        }
        for index, day in enumerate(days)
    ]
    _write_geoparquet(
        rows,
        columns=[
            ("date", "VARCHAR"),
            ("zone_id", "VARCHAR"),
            ("lat", "DOUBLE"),
            ("lon", "DOUBLE"),
            ("wkt", "VARCHAR"),
            ("temperature_2m", "DOUBLE"),
            ("precipitation", "DOUBLE"),
        ],
        dst=path.as_posix(),
    )


def _write_b3(path: Path, rows: list[tuple[str, str, float]]) -> None:
    """A B3-shaped `climate/ndvi/{zone}.parquet` — `(date, zone_id, ndvi_mean)` triples."""
    from src.utils.duckdb_conn import get_duckdb_connection

    con = get_duckdb_connection(":memory:")
    values = ", ".join(f"(DATE '{d}', '{z}', {n})" for d, z, n in rows)
    con.execute(
        f"COPY (SELECT * FROM (VALUES {values}) AS t(date, zone_id, ndvi_mean)) "
        f"TO '{path.as_posix()}' (FORMAT PARQUET)"
    )
    con.close()


@pytest.fixture
def stac_must_not_run(monkeypatch):
    """Any STAC collection is a test failure; the flag records that it never happened."""
    seen = {"called": False}

    async def _boom(*_a, **_k):
        seen["called"] = True
        raise AssertionError("an opted-in stored NDVI source must never reach STAC")

    monkeypatch.setattr("src.tools.climate._collect_ndvi_observations", _boom)
    return seen


def _b3_args(tmp_path: Path, ndvi_template: str, *extra: str):
    return _args(
        "--zone-id",
        _B3_ZONE,
        "--storage-mode",
        "local",
        "--start-date",
        "2024-01-01",
        "--end-date",
        "2024-01-03",
        "--era5-source-template",
        (tmp_path / "era5" / "{zone_id}.parquet").as_posix(),
        "--ndvi-source-template",
        ndvi_template,
        "--output-template",
        (tmp_path / "unified" / "{zone_id}.parquet").as_posix(),
        "--output-dir",
        tmp_path.as_posix(),
        "--no-run-manifest",
        *extra,
    )


class TestStoredB3NdviSource:
    @pytest.fixture(autouse=True)
    def _era5(self, tmp_path, monkeypatch, allow_local_parquet_paths):
        monkeypatch.setattr(panel.settings, "S3_BUCKET", "")
        (tmp_path / "era5").mkdir()
        (tmp_path / "unified").mkdir()
        _write_era5(
            tmp_path / "era5" / f"{_B3_ZONE}.parquet",
            _B3_ZONE,
            ["2024-01-01", "2024-01-02", "2024-01-03"],
        )

    def test_opt_in_reads_the_stored_series_through_the_real_tool(
        self, tmp_path, stac_must_not_run
    ):
        (tmp_path / "ndvi").mkdir()
        _write_b3(
            tmp_path / "ndvi" / f"{_B3_ZONE}.parquet",
            [("2024-01-02", _B3_ZONE, 0.71), ("2024-01-03", _B3_ZONE, 0.69)],
        )
        args = _b3_args(tmp_path, (tmp_path / "ndvi" / "{zone_id}.parquet").as_posix())

        result = asyncio.run(panel.run_panel(args, sleeper=_no_sleep))

        zone = result["zones"][0]
        assert result["ok"] is True, zone
        assert stac_must_not_run["called"] is False
        assert zone["ndvi_source"] == (tmp_path / "ndvi" / f"{_B3_ZONE}.parquet").as_posix()
        assert zone["stage"]["ndvi_input"] == "stored_b3_series"
        assert zone["stage"]["images_searched"] == 0
        assert zone["stage"]["n_matched"] == 2
        assert zone["stage"]["ndvi_coverage"]["months_observed"] == 1
        assert (tmp_path / "unified" / f"{_B3_ZONE}.parquet").is_file()
        assert result["ndvi_source_template"] == args.ndvi_source_template

    def test_absent_source_fails_closed_without_stac(self, tmp_path, stac_must_not_run):
        args = _b3_args(tmp_path, (tmp_path / "absent" / "{zone_id}.parquet").as_posix())

        result = asyncio.run(panel.run_panel(args, sleeper=_no_sleep))

        assert result["ok"] is False
        assert result["failed_zone_ids"] == [_B3_ZONE]
        assert "ndvi_source" in result["zones"][0]["error"]
        assert stac_must_not_run["called"] is False
        assert not (tmp_path / "unified" / f"{_B3_ZONE}.parquet").exists()

    @pytest.mark.parametrize(
        ("kind", "expected"),
        [
            ("other_zone", "has no rows for zone"),
            ("out_of_window", "has no rows for zone"),
            ("not_ndvi", "ndvi_mean"),
        ],
    )
    def test_mismatched_source_fails_closed_without_stac(
        self, tmp_path, stac_must_not_run, kind, expected
    ):
        source = tmp_path / "ndvi.parquet"
        if kind == "other_zone":
            _write_b3(source, [("2024-01-02", "fos_sur_mer", 0.4)])
        elif kind == "out_of_window":
            _write_b3(source, [("2025-06-02", _B3_ZONE, 0.4)])
        else:
            _write_era5(source, _B3_ZONE, ["2024-01-02"])
        args = _b3_args(tmp_path, source.as_posix())

        result = asyncio.run(panel.run_panel(args, sleeper=_no_sleep))

        assert result["ok"] is False
        assert expected in result["zones"][0]["error"]
        assert stac_must_not_run["called"] is False
        assert not (tmp_path / "unified" / f"{_B3_ZONE}.parquet").exists()

    @pytest.mark.parametrize(
        "template",
        [
            "",
            "http://169.254.169.254/{zone_id}.parquet",
            "/{zone_id}.parquet",
            "{zone_id}.csv",
        ],
    )
    @pytest.mark.parametrize("preflight", [False, True])
    def test_invalid_source_is_refused_before_the_tool_runs(self, tmp_path, template, preflight):
        args = _b3_args(tmp_path, template, *(["--preflight-only"] if preflight else []))

        result = asyncio.run(panel.run_panel(args, enrich_runner=_must_not_run, sleeper=_no_sleep))

        assert result["ok"] is False
        assert result["zones"][0]["failed_stage"] == "preflight"
        assert "ndvi_source" in result["zones"][0]["error"]

    def test_default_call_is_unchanged_without_opt_in(self, tmp_path):
        calls = []

        async def _fake_enrich(**kwargs):
            calls.append(kwargs)
            return {"output": kwargs["output_path"], "statistics": {}, "metadata": {}}

        args = _args(
            "--zone-id",
            _B3_ZONE,
            "--storage-mode",
            "local",
            "--era5-source-template",
            (tmp_path / "era5" / "{zone_id}.parquet").as_posix(),
            "--output-dir",
            tmp_path.as_posix(),
            "--no-run-manifest",
        )

        result = asyncio.run(panel.run_panel(args, enrich_runner=_fake_enrich, sleeper=_no_sleep))

        assert result["ok"] is True
        assert "ndvi_source" not in calls[0]
        assert result["zones"][0]["ndvi_source"] is None
        manifest = panel._run_manifest_payload(result)
        assert manifest["inputs"][1]["kind"] == "sentinel2_stac_ndvi"

    def test_opt_in_run_manifest_names_the_stored_series_not_stac(self, tmp_path):
        async def _fake_enrich(**kwargs):
            return {"output": kwargs["output_path"], "statistics": {}, "metadata": {}}

        template = (tmp_path / "ndvi" / "{zone_id}.parquet").as_posix()
        args = _b3_args(tmp_path, template)

        result = asyncio.run(panel.run_panel(args, enrich_runner=_fake_enrich, sleeper=_no_sleep))
        manifest = panel._run_manifest_payload(result)

        assert manifest["inputs"][1] == {
            "kind": "b3_ndvi_parquet",
            "uris": [(tmp_path / "ndvi" / f"{_B3_ZONE}.parquet").as_posix()],
        }
        assert manifest["parameters"]["ndvi_source_template"] == template
