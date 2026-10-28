"""Network-free tests for the B3 NDVI pilot panel's run manifest (issue #27).

The panel's NDVI work is already covered by ``test_ndvi_ingest.py``; here the ``ndvi_ingest`` tool
and the DuckDB readback are both stubbed so the tests only exercise what the panel itself owns:
run-id derivation, when a manifest is (not) published, the payload contents, and the failure paths.
"""

from __future__ import annotations

import json

import pytest

from scripts import ndvi_pilot_panel


def _args(**overrides):
    defaults = {
        "zones": ["montpellier_metropole", "bassin_du_lez"],
        "start": "2016-01-01",
        "end": "2024-12-31",
        "cloud_cover_max": 20,
        "output_dir": None,
        "run_id": "b3-proof",
        "no_run_manifest": False,
        "overwrite_run_manifest": False,
        "summarize_existing": False,
        "json": False,
    }
    return ndvi_pilot_panel.build_parser().parse_args([]).__class__(**{**defaults, **overrides})


def _fake_ingest(**overrides):
    """An ndvi_ingest stub returning one successful materialization per zone."""

    async def _run(*, zone_id, output_path, **_kwargs):
        if zone_id in overrides.get("failing", ()):
            return {"error": f"boom for {zone_id}"}
        result = {
            "zone_id": zone_id,
            "output": output_path,
            "n_observations": 81,
            # `months_or_scenes_searched` lives in `metadata`, not `statistics` — see
            # src/tools/climate.py's ndvi_ingest return value.
            "statistics": {"ndvi_mean_avg": 0.42},
            "metadata": {
                "months_or_scenes_searched": 108,
                **(overrides.get("metadata") or {}),
            },
        }
        return result

    return _run


@pytest.fixture(autouse=True)
def _stub_readback(monkeypatch):
    monkeypatch.setattr(
        ndvi_pilot_panel,
        "_readback",
        lambda dst: {
            "rows": 81,
            "first_date": "2016-11-14",
            "last_date": "2024-12-27",
            "distinct_points": 1,
        },
    )


@pytest.fixture(autouse=True)
def _bucket(monkeypatch):
    monkeypatch.setattr(ndvi_pilot_panel.settings, "S3_BUCKET", "deposium")


@pytest.fixture(autouse=True)
def _no_existing_manifest(monkeypatch):
    """Default: the run id is free. Collision tests override the seam explicitly."""
    monkeypatch.setattr(ndvi_pilot_panel, "_manifest_exists", lambda _uri: False)


class TestRunId:
    def test_explicit_run_id_is_kept(self):
        run_id = ndvi_pilot_panel._resolve_run_id(
            _args(run_id="b3-proof"), zone_ids=["a"], created_at="2026-07-31T00:00:00.000000Z"
        )
        assert run_id == "b3-proof"

    def test_derived_run_id_is_manifest_uri_safe(self):
        run_id = ndvi_pilot_panel._resolve_run_id(
            _args(run_id=None), zone_ids=["a", "b"], created_at="2026-07-31T01:02:03.000004Z"
        )
        # Dashes and colons would break the confined climate/runs/ URI charset.
        assert ndvi_pilot_panel.climate_baseline_panel._run_id_error(run_id) is None
        assert "2zones" in run_id

    def test_unsafe_explicit_run_id_is_rejected(self):
        with pytest.raises(ValueError):
            ndvi_pilot_panel._resolve_run_id(
                _args(run_id="../escape"), zone_ids=["a"], created_at="2026-07-31T00:00:00.000000Z"
            )


class TestManifestPublication:
    def test_manifest_is_published_under_climate_runs(self):
        calls = []
        panel = ndvi_pilot_panel.run_panel(
            _args(),
            ndvi_runner=_fake_ingest(),
            run_manifest_writer=lambda uri, payload: calls.append((uri, payload)),
        )

        assert panel["ok"] is True
        assert panel["run_manifest_written"] is True
        assert len(calls) == 1
        uri, payload = calls[0]
        assert uri == "s3://deposium/climate/runs/b3-proof/manifest.json"
        assert payload["kind"] == "geoai_run_manifest"
        assert payload["tool_or_script"] == "scripts/ndvi_pilot_panel.py"
        assert payload["parameters"]["start_date"] == "2016-01-01"
        assert payload["parameters"]["end_date"] == "2024-12-31"
        assert payload["parameters"]["zone_ids"] == ["montpellier_metropole", "bassin_du_lez"]
        assert payload["code_ref"]["repo"] == "deposium_geoai"
        assert payload["checks"] == {
            "ok": True,
            "n_zones_requested": 2,
            "n_zones_completed": 2,
            "failed_zone_ids": [],
            "month_coverage": {
                "months_requested": 108,
                "complete_evaluated": True,
                "zones": [
                    {
                        "zone_id": "montpellier_metropole",
                        "months_observed": 81,
                        "complete": False,
                    },
                    {"zone_id": "bassin_du_lez", "months_observed": 81, "complete": False},
                ],
            },
            "truncation_warnings": [],
        }

    def test_outputs_carry_the_readback_evidence(self):
        calls = []
        ndvi_pilot_panel.run_panel(
            _args(),
            ndvi_runner=_fake_ingest(),
            run_manifest_writer=lambda uri, payload: calls.append((uri, payload)),
        )

        outputs = calls[0][1]["outputs"]
        assert [o["zone_id"] for o in outputs] == ["montpellier_metropole", "bassin_du_lez"]
        assert outputs[0]["uri"] == "s3://deposium/climate/ndvi/montpellier_metropole.parquet"
        assert outputs[0]["rows"] == 81
        assert outputs[0]["first_date"] == "2016-11-14"
        assert outputs[0]["last_date"] == "2024-12-27"

    def test_local_runs_publish_no_remote_manifest(self, tmp_path):
        def _must_not_write(*_args):
            raise AssertionError("a local run must not publish a remote manifest")

        panel = ndvi_pilot_panel.run_panel(
            _args(output_dir=tmp_path.as_posix()),
            ndvi_runner=_fake_ingest(),
            run_manifest_writer=_must_not_write,
        )
        assert panel["run_manifest_uri"] is None
        assert panel["run_manifest_written"] is False
        assert panel["storage_mode"] == "local"

    def test_no_run_manifest_flag_skips_publication(self):
        def _must_not_write(*_args):
            raise AssertionError("--no-run-manifest must skip the write")

        panel = ndvi_pilot_panel.run_panel(
            _args(no_run_manifest=True),
            ndvi_runner=_fake_ingest(),
            run_manifest_writer=_must_not_write,
        )
        assert panel["run_manifest_uri"] is None
        assert panel["ok"] is True


class TestEnvironmentProvenance:
    """Which RustFS held the objects must be readable off the manifest (issue #27)."""

    def test_environment_records_endpoint_bucket_and_addressing(self, monkeypatch):
        monkeypatch.setattr(ndvi_pilot_panel.settings, "S3_ENDPOINT", "http://127.0.0.1:9000")
        monkeypatch.setattr(ndvi_pilot_panel.settings, "S3_FORCE_PATH_STYLE", True)
        calls = []
        ndvi_pilot_panel.run_panel(
            _args(),
            ndvi_runner=_fake_ingest(),
            run_manifest_writer=lambda uri, payload: calls.append((uri, payload)),
        )

        assert calls[0][1]["environment"] == {
            "s3_endpoint": "http://127.0.0.1:9000",
            "s3_endpoint_credentials_removed": False,
            "s3_bucket": "deposium",
            "s3_force_path_style": True,
        }

    def test_two_instances_are_distinguishable(self, monkeypatch):
        """The dispute the block exists to settle: same URIs, different store."""

        def _run_against(endpoint):
            monkeypatch.setattr(ndvi_pilot_panel.settings, "S3_ENDPOINT", endpoint)
            calls = []
            ndvi_pilot_panel.run_panel(
                _args(),
                ndvi_runner=_fake_ingest(),
                run_manifest_writer=lambda uri, payload: calls.append((uri, payload)),
            )
            return calls[0][0], calls[0][1]["environment"]["s3_endpoint"]

        dev_harness = _run_against("http://127.0.0.1:9200")
        mesh = _run_against("http://localhost:9100")

        assert dev_harness[0] == mesh[0]  # byte-identical manifest URIs...
        assert dev_harness[1] != mesh[1]  # ...but the store is now on the record

    def test_unset_endpoint_is_null_not_an_empty_string(self, monkeypatch):
        monkeypatch.setattr(ndvi_pilot_panel.settings, "S3_ENDPOINT", "")
        calls = []
        ndvi_pilot_panel.run_panel(
            _args(),
            ndvi_runner=_fake_ingest(),
            run_manifest_writer=lambda uri, payload: calls.append((uri, payload)),
        )

        # "" would read like a configured address; null says "no explicit endpoint".
        assert calls[0][1]["environment"]["s3_endpoint"] is None

    def test_no_credential_ever_reaches_the_payload(self, monkeypatch):
        monkeypatch.setattr(ndvi_pilot_panel.settings, "S3_ENDPOINT", "http://127.0.0.1:9000")
        monkeypatch.setattr(ndvi_pilot_panel.settings, "S3_ACCESS_KEY", "deposium-dev-access-key")
        monkeypatch.setattr(ndvi_pilot_panel.settings, "S3_SECRET_KEY", "deposium-dev-secret-key")
        calls = []
        ndvi_pilot_panel.run_panel(
            _args(),
            ndvi_runner=_fake_ingest(),
            run_manifest_writer=lambda uri, payload: calls.append((uri, payload)),
        )

        serialized = json.dumps(calls[0][1], default=str)
        assert "deposium-dev-access-key" not in serialized
        assert "deposium-dev-secret-key" not in serialized
        assert "127.0.0.1:9000" in serialized

    def test_credentials_embedded_in_the_endpoint_never_reach_the_payload(self, monkeypatch):
        """`S3_ENDPOINT` may carry userinfo, and this manifest is published for handoff."""
        monkeypatch.setattr(
            ndvi_pilot_panel.settings,
            "S3_ENDPOINT",
            "http://rustfs-admin:sup3r-s3cret@mesh.internal:9000",
        )
        calls = []
        ndvi_pilot_panel.run_panel(
            _args(),
            ndvi_runner=_fake_ingest(),
            run_manifest_writer=lambda uri, payload: calls.append((uri, payload)),
        )

        # Assert on the serialized artifact: nesting must not be able to hide the secret.
        serialized = json.dumps(calls[0][1], default=str)
        assert "sup3r-s3cret" not in serialized
        assert "rustfs-admin" not in serialized
        environment = calls[0][1]["environment"]
        # Refused whole rather than stripped: the accept grammar offers no partial reading.
        assert environment["s3_endpoint"] is None
        # Stated, not silent: a shortened address is not a misconfiguration.
        assert environment["s3_endpoint_credentials_removed"] is True

    def test_the_stac_endpoint_is_published_as_an_address_too(self, monkeypatch):
        """`inputs[].endpoint` is `STAC_URL` verbatim, and a private STAC can carry userinfo.

        The publication boundary redacts on `S3_ENDPOINT` only, so nothing downstream catches
        this one: it has to be sanitized where it is written.
        """
        monkeypatch.setattr(
            ndvi_pilot_panel.settings,
            "STAC_URL",
            "https://stac-reader:stac-s3cret@stac.internal/v1",
        )
        calls = []
        ndvi_pilot_panel.run_panel(
            _args(),
            ndvi_runner=_fake_ingest(),
            run_manifest_writer=lambda uri, payload: calls.append((uri, payload)),
        )

        serialized = json.dumps(calls[0][1], default=str)
        assert "stac-s3cret" not in serialized
        assert "stac-reader" not in serialized
        assert calls[0][1]["inputs"][0]["endpoint"] == "https://stac.internal/v1"

    def test_a_stac_query_is_dropped_and_the_manifest_SAYS_so(self, monkeypatch):
        """The N4 defect. A STAC deployment whose URL carries a query was misrecorded in silence.

        `sanitize_endpoint` answered `("https://stac.example/v1", True)` for
        `https://stac.example/v1?token=abc` — the query gone, one boolean raised, and no way for
        a reader of the manifest to tell that from a lowercased scheme. `endpoint_reductions`
        names the component, so what is published is legibly a prefix of what was configured.
        """
        monkeypatch.setattr(
            ndvi_pilot_panel.settings, "STAC_URL", "https://stac.internal/v1?token=abc"
        )
        calls = []
        ndvi_pilot_panel.run_panel(
            _args(),
            ndvi_runner=_fake_ingest(),
            run_manifest_writer=lambda uri, payload: calls.append((uri, payload)),
        )

        stac_input = calls[0][1]["inputs"][0]
        assert stac_input["endpoint"] == "https://stac.internal/v1"
        assert stac_input["endpoint_reductions"] == ["query"]
        # And the token itself is nowhere in the published bytes.
        assert "token=abc" not in json.dumps(calls[0][1], default=str)

    def test_an_untouched_stac_url_publishes_an_empty_reduction_list(self, monkeypatch):
        """`[]` is the load-bearing answer: it is how a reader knows the recorded address IS the
        configured value. Without it, "nothing was removed" and "something was" look the same."""
        monkeypatch.setattr(
            ndvi_pilot_panel.settings, "STAC_URL", "https://earth-search.aws.element84.com/v1"
        )
        calls = []
        ndvi_pilot_panel.run_panel(
            _args(),
            ndvi_runner=_fake_ingest(),
            run_manifest_writer=lambda uri, payload: calls.append((uri, payload)),
        )

        stac_input = calls[0][1]["inputs"][0]
        assert stac_input["endpoint"] == "https://earth-search.aws.element84.com/v1"
        assert stac_input["endpoint_reductions"] == []

    def test_a_stripped_userinfo_is_named_in_the_reductions(self, monkeypatch):
        monkeypatch.setattr(
            ndvi_pilot_panel.settings,
            "STAC_URL",
            "https://stac-reader:stac-s3cret@stac.internal/v1?token=abc",
        )
        calls = []
        ndvi_pilot_panel.run_panel(
            _args(),
            ndvi_runner=_fake_ingest(),
            run_manifest_writer=lambda uri, payload: calls.append((uri, payload)),
        )

        stac_input = calls[0][1]["inputs"][0]
        assert stac_input["endpoint"] == "https://stac.internal/v1"
        assert stac_input["endpoint_reductions"] == ["userinfo", "query"]

    def test_a_refused_stac_url_publishes_null_and_says_it_was_refused(self, monkeypatch):
        monkeypatch.setattr(ndvi_pilot_panel.settings, "STAC_URL", "ftp://stac.internal/v1")
        calls = []
        ndvi_pilot_panel.run_panel(
            _args(),
            ndvi_runner=_fake_ingest(),
            run_manifest_writer=lambda uri, payload: calls.append((uri, payload)),
        )

        stac_input = calls[0][1]["inputs"][0]
        assert stac_input["endpoint"] is None
        assert stac_input["endpoint_reductions"] == ["refused"]

    def test_a_credential_free_stac_endpoint_is_published_unchanged(self, monkeypatch):
        monkeypatch.setattr(
            ndvi_pilot_panel.settings, "STAC_URL", "https://earth-search.aws.element84.com/v1"
        )
        calls = []
        ndvi_pilot_panel.run_panel(
            _args(),
            ndvi_runner=_fake_ingest(),
            run_manifest_writer=lambda uri, payload: calls.append((uri, payload)),
        )

        assert calls[0][1]["inputs"][0]["endpoint"] == "https://earth-search.aws.element84.com/v1"

    def test_a_credential_free_endpoint_is_not_flagged(self, monkeypatch):
        monkeypatch.setattr(ndvi_pilot_panel.settings, "S3_ENDPOINT", "http://127.0.0.1:9000")
        calls = []
        ndvi_pilot_panel.run_panel(
            _args(),
            ndvi_runner=_fake_ingest(),
            run_manifest_writer=lambda uri, payload: calls.append((uri, payload)),
        )

        assert calls[0][1]["environment"]["s3_endpoint_credentials_removed"] is False


class TestMonthCoverage:
    """ "81 observations" must never be readable as "81 of 81" when 108 months were searched."""

    def test_month_ranges_drive_the_requested_count(self):
        # Same split the monthly collector walks, partial edge months included.
        assert (
            ndvi_pilot_panel._months_requested(
                {"start_date": "2016-01-01", "end_date": "2024-12-31"}
            )
            == 108
        )
        assert (
            ndvi_pilot_panel._months_requested(
                {"start_date": "2020-01-15", "end_date": "2020-03-02"}
            )
            == 3
        )

    def test_partial_coverage_records_observed_below_requested(self):
        calls = []
        ndvi_pilot_panel.run_panel(
            _args(),
            ndvi_runner=_fake_ingest(),
            run_manifest_writer=lambda uri, payload: calls.append((uri, payload)),
        )

        coverage = calls[0][1]["checks"]["month_coverage"]
        assert coverage["months_requested"] == 108
        assert coverage["zones"][0] == {
            "zone_id": "montpellier_metropole",
            "months_observed": 81,
            "complete": False,
        }

    def test_full_coverage_is_marked_complete(self, monkeypatch):
        monkeypatch.setattr(
            ndvi_pilot_panel,
            "_readback",
            lambda dst: {
                "rows": 12,
                "first_date": "2020-01-14",
                "last_date": "2020-12-27",
                "distinct_points": 1,
            },
        )

        async def _twelve(*, zone_id, output_path, **_kwargs):
            return {
                "zone_id": zone_id,
                "output": output_path,
                "n_observations": 12,
                "statistics": {},
                "metadata": {},
            }

        calls = []
        ndvi_pilot_panel.run_panel(
            _args(start="2020-01-01", end="2020-12-31"),
            ndvi_runner=_twelve,
            run_manifest_writer=lambda uri, payload: calls.append((uri, payload)),
        )

        coverage = calls[0][1]["checks"]["month_coverage"]
        assert coverage["months_requested"] == 12
        assert all(z["months_observed"] == 12 and z["complete"] for z in coverage["zones"])

    def test_months_observed_is_the_number_the_panel_re_read_from_the_object(self):
        """The numerator is a property of the OBJECT, and this pins the half that is observable.

        `--summarize-existing` has no tool result to ask, so its `months_observed` has always been
        the readback's row count — 40 here, whatever else anything claims. On the COLLECTION path
        the same guarantee holds by a different mechanism: the two counts must be EQUAL or the
        zone fails (see the divergence test below), which is why no test can distinguish the two
        sources there. That equality is the guarantee; this is the direct evidence for it.
        """
        seen = []

        def _forty(dst):
            seen.append(dst)
            return {
                "rows": 40,
                "first_date": "2016-11-14",
                "last_date": "2024-12-27",
                "distinct_points": 1,
            }

        calls = []
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(ndvi_pilot_panel, "_readback", _forty)
            ndvi_pilot_panel.run_panel(
                _args(summarize_existing=True),
                run_manifest_writer=lambda uri, payload: calls.append((uri, payload)),
            )

        assert seen, "the panel must actually have re-read the artifact"
        coverage = calls[0][1]["checks"]["month_coverage"]
        assert all(zone["months_observed"] == 40 for zone in coverage["zones"])

    def test_a_tool_count_that_diverges_from_the_artifact_fails_the_zone(self):
        """`ndvi_ingest` says 81, the parquet holds 40. That is a real defect — a partial write, a
        stale object at that key, a bug in the writer — and a manifest must not paper over it.

        Failing the zone rather than silently preferring the readback is the deliberate half of
        the choice: the readback number alone would publish a manifest that looks entirely
        healthy over an artifact that disagrees with the tool that produced it.
        """
        calls = []
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(
                ndvi_pilot_panel,
                "_readback",
                lambda dst: {
                    "rows": 40,
                    "first_date": "2016-11-14",
                    "last_date": "2024-12-27",
                    "distinct_points": 1,
                },
            )
            result = ndvi_pilot_panel.run_panel(
                _args(),
                ndvi_runner=_fake_ingest(),
                run_manifest_writer=lambda uri, payload: calls.append((uri, payload)),
            )

        assert not result["ok"]
        assert all(not zone["ok"] for zone in result["zones"])
        # Both numbers and the URI, so an operator knows which object to go and look at.
        error = result["zones"][0]["error"]
        assert "row count divergence" in error
        assert "81 observations" in error and "40 rows" in error
        # A failed zone is absent from coverage, so no figure is published for it at all.
        assert calls[0][1]["checks"]["month_coverage"]["zones"] == []
        assert calls[0][1]["checks"]["failed_zone_ids"] == [
            "montpellier_metropole",
            "bassin_du_lez",
        ]

    def test_agreeing_counts_leave_the_zone_healthy(self):
        """The normal case must not become a failure: tool and artifact both say 81."""
        calls = []
        result = ndvi_pilot_panel.run_panel(
            _args(),
            ndvi_runner=_fake_ingest(),
            run_manifest_writer=lambda uri, payload: calls.append((uri, payload)),
        )

        assert result["ok"], [z.get("error") for z in result["zones"]]
        assert all(
            z["months_observed"] == 81 for z in calls[0][1]["checks"]["month_coverage"]["zones"]
        )

    def test_failed_zone_is_absent_from_coverage(self):
        calls = []
        ndvi_pilot_panel.run_panel(
            _args(),
            ndvi_runner=_fake_ingest(failing=("bassin_du_lez",)),
            run_manifest_writer=lambda uri, payload: calls.append((uri, payload)),
        )

        coverage = calls[0][1]["checks"]["month_coverage"]
        assert [z["zone_id"] for z in coverage["zones"]] == ["montpellier_metropole"]

    def test_summarize_existing_still_records_coverage(self):
        calls = []
        ndvi_pilot_panel.run_panel(
            _args(summarize_existing=True),
            run_manifest_writer=lambda uri, payload: calls.append((uri, payload)),
        )

        coverage = calls[0][1]["checks"]["month_coverage"]
        assert coverage["months_requested"] == 108
        assert coverage["zones"][0]["months_observed"] == 81

    def test_summarize_existing_never_claims_completeness(self):
        """`months_observed` is the artifact's TOTAL row count there, not months in the window."""
        calls = []
        ndvi_pilot_panel.run_panel(
            _args(summarize_existing=True, start="2020-01-01", end="2020-12-31"),
            run_manifest_writer=lambda uri, payload: calls.append((uri, payload)),
        )

        coverage = calls[0][1]["checks"]["month_coverage"]
        # 81 observed out of 12 requested is not "complete", it is two incommensurable numbers.
        assert coverage["months_requested"] == 12
        assert coverage["zones"][0]["months_observed"] == 81
        assert coverage["complete_evaluated"] is False
        assert all(z["complete"] is None for z in coverage["zones"])

    def test_an_unusable_window_is_not_an_evaluated_completeness(self):
        """A materialization run can also fail to evaluate completeness: no denominator."""
        calls = []
        ndvi_pilot_panel.run_panel(
            # Inverted window: month_ranges returns [], which _months_requested reports as None
            # rather than 0 so nothing is "complete" against a zero denominator. With the tool's
            # own months_or_scenes_searched absent too, there is no denominator at all.
            _args(
                start="2024-12-31",
                end="2024-01-01",
            ),
            ndvi_runner=_fake_ingest(metadata={"months_or_scenes_searched": None}),
            run_manifest_writer=lambda uri, payload: calls.append((uri, payload)),
        )

        coverage = calls[0][1]["checks"]["month_coverage"]
        assert coverage["months_requested"] is None
        assert all(z["complete"] is None for z in coverage["zones"])
        # The flag must cover this panel-level refusal too: claiming completeness WAS evaluated
        # beside a null denominator and an all-null verdict says the opposite of what happened.
        assert coverage["complete_evaluated"] is False

    def test_a_true_flag_still_allows_a_per_zone_null_verdict(self):
        """`complete_evaluated: true` means the panel could evaluate, not that every zone got one.

        The remaining per-zone reason is `months_observed > months_requested`.
        """
        calls = []
        ndvi_pilot_panel.run_panel(
            _args(),
            ndvi_runner=_fake_ingest(metadata={"months_or_scenes_searched": 12}),
            run_manifest_writer=lambda uri, payload: calls.append((uri, payload)),
        )

        coverage = calls[0][1]["checks"]["month_coverage"]
        assert coverage["complete_evaluated"] is True
        assert coverage["months_requested"] == 12
        assert coverage["zones"][0]["months_observed"] == 81
        assert coverage["zones"][0]["complete"] is None

    def test_observed_above_requested_is_never_complete(self):
        """A numerator larger than the denominator proves they are not counting the same thing."""

        async def _overshoot(*, zone_id, output_path, **_kwargs):
            return {
                "zone_id": zone_id,
                "output": output_path,
                "n_observations": 81,
                "statistics": {},
                "metadata": {"months_or_scenes_searched": 12},
            }

        calls = []
        ndvi_pilot_panel.run_panel(
            _args(start="2020-01-01", end="2020-12-31"),
            ndvi_runner=_overshoot,
            run_manifest_writer=lambda uri, payload: calls.append((uri, payload)),
        )

        coverage = calls[0][1]["checks"]["month_coverage"]
        assert coverage["months_requested"] == 12
        assert all(z["complete"] is None for z in coverage["zones"])

    def test_inverted_window_does_not_produce_a_zero_denominator(self):
        """`month_ranges` returns [] rather than raising, which would make everything complete."""
        assert (
            ndvi_pilot_panel._months_requested(
                {"start_date": "2024-01-01", "end_date": "2020-01-01"}
            )
            is None
        )

        calls = []
        panel = ndvi_pilot_panel.run_panel(
            _args(start="2024-01-01", end="2020-01-01", summarize_existing=True),
            run_manifest_writer=lambda uri, payload: calls.append((uri, payload)),
        )

        assert panel["run_manifest_written"] is True
        coverage = calls[0][1]["checks"]["month_coverage"]
        assert coverage["months_requested"] is None
        assert all(z["complete"] is None for z in coverage["zones"])

    def test_the_denominator_comes_from_the_run_that_produced_the_numerator(self):
        """`ndvi_ingest` already reports what it walked; recomputing it invites drift (F6)."""

        async def _reports_96(*, zone_id, output_path, **_kwargs):
            return {
                "zone_id": zone_id,
                "output": output_path,
                "n_observations": 81,
                "statistics": {},
                "metadata": {"months_or_scenes_searched": 96},
            }

        calls = []
        ndvi_pilot_panel.run_panel(
            # The window recomputes to 108; the tool says it walked 96. The tool wins.
            _args(start="2016-01-01", end="2024-12-31"),
            ndvi_runner=_reports_96,
            run_manifest_writer=lambda uri, payload: calls.append((uri, payload)),
        )

        assert calls[0][1]["checks"]["month_coverage"]["months_requested"] == 96

    def test_unparseable_window_degrades_instead_of_losing_the_manifest(self):
        calls = []
        panel = ndvi_pilot_panel.run_panel(
            _args(start="not-a-date", end="2024-12-31", summarize_existing=True),
            run_manifest_writer=lambda uri, payload: calls.append((uri, payload)),
        )

        assert panel["run_manifest_written"] is True
        coverage = calls[0][1]["checks"]["month_coverage"]
        assert coverage["months_requested"] is None
        assert coverage["zones"][0]["complete"] is None

    def test_no_constant_applicability_flag_is_shipped_alongside_the_warnings(self):
        """The panel hardcodes monthly cadence, so such a flag could only ever be a constant."""
        calls = []
        ndvi_pilot_panel.run_panel(
            _args(),
            ndvi_runner=_fake_ingest(),
            run_manifest_writer=lambda uri, payload: calls.append((uri, payload)),
        )

        checks = calls[0][1]["checks"]
        assert checks["truncation_warnings"] == []
        assert "truncation_warnings_applicable" not in checks


class TestSummarizeExisting:
    def test_readback_only_run_publishes_a_manifest_without_collecting(self):
        def _must_not_collect(**_kwargs):
            raise AssertionError("--summarize-existing must not re-stream any COG")

        calls = []
        panel = ndvi_pilot_panel.run_panel(
            _args(summarize_existing=True),
            ndvi_runner=_must_not_collect,
            run_manifest_writer=lambda uri, payload: calls.append((uri, payload)),
        )

        assert panel["ok"] is True
        assert panel["run_manifest_written"] is True
        outputs = calls[0][1]["outputs"]
        assert [o["zone_id"] for o in outputs] == ["montpellier_metropole", "bassin_du_lez"]
        assert outputs[0]["rows"] == 81

    def test_aggregation_only_run_does_not_claim_materialization(self):
        calls = []
        ndvi_pilot_panel.run_panel(
            _args(summarize_existing=True),
            run_manifest_writer=lambda uri, payload: calls.append((uri, payload)),
        )

        payload = calls[0][1]
        assert payload["parameters"]["summarize_existing"] is True
        assert payload["code_ref"] is None
        assert payload["provenance"]["mode"] == "aggregation_only"
        assert payload["provenance"]["artifact_code_ref"] is None
        assert payload["provenance"]["aggregation_code_ref"]["repo"] == "deposium_geoai"

    def test_materializing_run_claims_materialization(self):
        calls = []
        ndvi_pilot_panel.run_panel(
            _args(),
            ndvi_runner=_fake_ingest(),
            run_manifest_writer=lambda uri, payload: calls.append((uri, payload)),
        )

        payload = calls[0][1]
        assert payload["parameters"]["summarize_existing"] is False
        assert payload["provenance"]["mode"] == "materialization"
        assert payload["provenance"]["artifact_code_ref"]["repo"] == "deposium_geoai"

    def test_unreadable_artifact_fails_the_zone(self, monkeypatch):
        def _boom(dst):
            if "bassin_du_lez" in dst:
                raise RuntimeError("no such object")
            return {
                "rows": 81,
                "first_date": "2016-11-14",
                "last_date": "2024-12-27",
                "distinct_points": 1,
            }

        monkeypatch.setattr(ndvi_pilot_panel, "_readback", _boom)
        calls = []
        panel = ndvi_pilot_panel.run_panel(
            _args(summarize_existing=True),
            run_manifest_writer=lambda uri, payload: calls.append((uri, payload)),
        )

        # A manifest must never advertise an object nobody could read.
        assert panel["ok"] is False
        payload = calls[0][1]
        assert [o["zone_id"] for o in payload["outputs"]] == ["montpellier_metropole"]
        assert payload["checks"]["failed_zone_ids"] == ["bassin_du_lez"]
        assert "no such object" in payload["errors"][0]["error"]


class TestManifestImmutability:
    def test_reused_run_id_refuses_to_replace_an_existing_manifest(self):
        def _must_not_write(*_args):
            raise AssertionError("an existing manifest must never be replaced silently")

        panel = ndvi_pilot_panel.run_panel(
            _args(summarize_existing=True),
            run_manifest_writer=_must_not_write,
            run_manifest_exists=lambda _uri: True,
        )

        assert panel["ok"] is False
        assert panel["run_manifest_written"] is False
        assert "immutable" in panel["run_manifest_errors"][0]["error"]
        assert panel["run_manifest_errors"][0]["uri"] == (
            "s3://deposium/climate/runs/b3-proof/manifest.json"
        )

    def test_summarize_existing_cannot_clobber_a_materialization_manifest(self):
        """The regression Greptile found: aggregation-only provenance replacing materialization."""
        written = []
        ndvi_pilot_panel.run_panel(
            _args(),
            ndvi_runner=_fake_ingest(),
            run_manifest_writer=lambda uri, payload: written.append(payload),
            run_manifest_exists=lambda _uri: False,
        )
        assert written[0]["provenance"]["mode"] == "materialization"

        # Same explicit --run-id, now in aggregate mode: it must be refused, not overwrite.
        panel = ndvi_pilot_panel.run_panel(
            _args(summarize_existing=True),
            run_manifest_writer=lambda uri, payload: written.append(payload),
            run_manifest_exists=lambda _uri: True,
        )
        assert panel["ok"] is False
        assert len(written) == 1
        assert written[0]["provenance"]["artifact_code_ref"]["repo"] == "deposium_geoai"

    def test_overwrite_flag_allows_a_deliberate_replacement(self):
        calls = []
        panel = ndvi_pilot_panel.run_panel(
            _args(summarize_existing=True, overwrite_run_manifest=True),
            run_manifest_writer=lambda uri, payload: calls.append((uri, payload)),
            run_manifest_exists=lambda _uri: True,
        )
        assert panel["ok"] is True
        assert panel["run_manifest_written"] is True
        assert len(calls) == 1


class TestFailurePaths:
    def test_failed_zone_is_excluded_from_outputs_but_recorded(self):
        calls = []
        panel = ndvi_pilot_panel.run_panel(
            _args(),
            ndvi_runner=_fake_ingest(failing=("bassin_du_lez",)),
            run_manifest_writer=lambda uri, payload: calls.append((uri, payload)),
        )

        assert panel["ok"] is False
        payload = calls[0][1]
        assert [o["zone_id"] for o in payload["outputs"]] == ["montpellier_metropole"]
        assert payload["checks"]["failed_zone_ids"] == ["bassin_du_lez"]
        assert payload["checks"]["n_zones_completed"] == 1
        assert payload["errors"] == [
            {"zone_id": "bassin_du_lez", "error": "boom for bassin_du_lez"}
        ]

    def test_truncation_warning_reaches_the_manifest(self):
        calls = []
        panel = ndvi_pilot_panel.run_panel(
            _args(),
            ndvi_runner=_fake_ingest(metadata={"ndvi_truncation_warning": "hit the 50-item cap"}),
            run_manifest_writer=lambda uri, payload: calls.append((uri, payload)),
        )

        checks = calls[0][1]["checks"]
        warnings = checks["truncation_warnings"]
        assert [w["zone_id"] for w in warnings] == ["montpellier_metropole", "bassin_du_lez"]
        assert warnings[0]["warning"] == "hit the 50-item cap"
        assert panel["truncation_warnings"] == warnings
        # A populated list flagged "inapplicable" is exactly the contradiction that got the flag
        # deleted; no consumer may ever see the pair.
        assert "truncation_warnings_applicable" not in checks

    def test_materialization_readback_failure_does_not_abort_the_panel(self, monkeypatch):
        """The regression Greptile found: a readback error escaping run_panel mid-materialization."""

        def _boom(dst):
            if "montpellier_metropole" in dst:
                raise RuntimeError("httpfs timeout")
            return {
                "rows": 81,
                "first_date": "2016-11-14",
                "last_date": "2024-12-27",
                "distinct_points": 1,
            }

        monkeypatch.setattr(ndvi_pilot_panel, "_readback", _boom)
        calls = []
        panel = ndvi_pilot_panel.run_panel(
            _args(),
            ndvi_runner=_fake_ingest(),
            run_manifest_writer=lambda uri, payload: calls.append((uri, payload)),
        )

        # The later zone still ran, and the manifest was still published.
        assert [z["zone_id"] for z in panel["zones"]] == [
            "montpellier_metropole",
            "bassin_du_lez",
        ]
        assert panel["run_manifest_written"] is True
        assert panel["ok"] is False
        payload = calls[0][1]
        assert [o["zone_id"] for o in payload["outputs"]] == ["bassin_du_lez"]
        assert payload["checks"]["failed_zone_ids"] == ["montpellier_metropole"]
        assert "httpfs timeout" in payload["errors"][0]["error"]

    def test_manifest_write_failure_fails_the_run(self):
        def _boom(*_args):
            raise RuntimeError("rustfs unreachable")

        panel = ndvi_pilot_panel.run_panel(
            _args(), ndvi_runner=_fake_ingest(), run_manifest_writer=_boom
        )

        # The artifacts materialized fine; the run still fails so the operator re-publishes
        # instead of shipping unmanifested artifacts.
        assert all(z["ok"] for z in panel["zones"])
        assert panel["ok"] is False
        assert panel["run_manifest_written"] is False
        assert panel["run_manifest_errors"] == [
            {
                "uri": "s3://deposium/climate/runs/b3-proof/manifest.json",
                "error": "rustfs unreachable",
            }
        ]


# --------------------------------------------------------------------------- #
# A private catalogue's userinfo never reaches the published run manifest
# --------------------------------------------------------------------------- #


class TestStacProvenanceIsSanitized:
    """`inputs[].endpoint` is provenance, and provenance is published into a bucket.

    `src/server.py` already sanitizes `STAC_URL` before logging it. The panels copied the same
    setting into the manifest verbatim, and the manifest is the artifact that gets handed to
    another party. Measured on this branch before the fix, with
    `STAC_URL='https://stac-reader:stac-s3cret@stac.internal/v1'`:

        inputs[0].endpoint = 'https://stac-reader:stac-s3cret@stac.internal/v1'
    """

    def test_a_catalogue_credential_never_reaches_the_published_manifest(self, monkeypatch):
        monkeypatch.setattr(
            ndvi_pilot_panel.settings,
            "STAC_URL",
            "https://stac-reader:stac-s3cret@stac.internal/v1",
        )
        calls = []
        ndvi_pilot_panel.run_panel(
            _args(),
            ndvi_runner=_fake_ingest(),
            run_manifest_writer=lambda uri, payload: calls.append((uri, payload)),
        )

        payload = calls[0][1]
        published = json.dumps(payload, default=str)
        assert "stac-s3cret" not in published
        assert "stac-reader" not in published
        assert payload["inputs"][0]["endpoint"] == "https://stac.internal/v1"

    def test_a_query_borne_catalogue_token_is_dropped_too(self, monkeypatch):
        """`?token=` is the other way a catalogue takes a credential on the URL."""
        monkeypatch.setattr(
            ndvi_pilot_panel.settings, "STAC_URL", "https://stac.internal/v1?token=abc123"
        )
        calls = []
        ndvi_pilot_panel.run_panel(
            _args(),
            ndvi_runner=_fake_ingest(),
            run_manifest_writer=lambda uri, payload: calls.append((uri, payload)),
        )

        assert "abc123" not in json.dumps(calls[0][1], default=str)

    def test_the_default_public_catalogue_is_published_unchanged(self, monkeypatch):
        """The sanitizer must not cost provenance: the DEFAULT `STAC_URL` has a path.

        A sanitizer that refused or truncated `.../v1` would publish a value the manifest's
        reader cannot resolve, which is a worse outcome than the leak it fixes.
        """
        monkeypatch.setattr(
            ndvi_pilot_panel.settings,
            "STAC_URL",
            "https://earth-search.aws.element84.com/v1",
        )
        calls = []
        ndvi_pilot_panel.run_panel(
            _args(),
            ndvi_runner=_fake_ingest(),
            run_manifest_writer=lambda uri, payload: calls.append((uri, payload)),
        )

        endpoint = calls[0][1]["inputs"][0]["endpoint"]
        assert endpoint == "https://earth-search.aws.element84.com/v1"


class TestMainStdoutIsRedacted:
    """`main` prints the in-memory panel, which no redacting writer has ever seen.

    Both clients quote the configured credential in their failure text — httpfs echoes
    `s3_access_key_id` inside a 403 body, botocore names the signing key — and stdout is a CI log.
    """

    _FAILURE = (
        "IOException: HTTP 403 <AWSAccessKeyId>deposium-dev</AWSAccessKeyId> "
        "signature for deposium-dev-secret rejected"
    )

    def _panel(self):
        return {
            "ok": False,
            "zones": [{"zone_id": "montpellier_metropole", "ok": False, "error": self._FAILURE}],
            "truncation_warnings": [],
            "run_manifest_written": False,
            "run_manifest_uri": "s3://deposium/climate/runs/r/manifest.json",
            "run_manifest_errors": [
                {"uri": "s3://deposium/climate/runs/r/manifest.json", "error": self._FAILURE}
            ],
        }

    def _run(self, monkeypatch, capsys, json_mode):
        monkeypatch.setattr(ndvi_pilot_panel.settings, "S3_ENDPOINT", "http://rustfs:9000")
        monkeypatch.setattr(ndvi_pilot_panel.settings, "S3_ACCESS_KEY", "deposium-dev")
        monkeypatch.setattr(ndvi_pilot_panel.settings, "S3_SECRET_KEY", "deposium-dev-secret")
        monkeypatch.setattr(ndvi_pilot_panel, "run_panel", lambda _args: self._panel())
        argv = ["--json"] if json_mode else []
        ndvi_pilot_panel.main(argv)
        return capsys.readouterr().out

    def test_the_json_dump_carries_no_credential(self, monkeypatch, capsys):
        out = self._run(monkeypatch, capsys, json_mode=True)
        assert "deposium-dev-secret" not in out
        assert "deposium-dev</AWSAccessKeyId>" not in out
        # The tail must not survive on its own either: the access key prefixes the secret.
        assert "-secret" not in out
        # What must survive: the manifest URI names what failed.
        assert "s3://deposium/climate/runs/r/manifest.json" in out

    def test_the_human_lines_carry_no_credential(self, monkeypatch, capsys):
        out = self._run(monkeypatch, capsys, json_mode=False)
        assert "deposium-dev-secret" not in out
        assert "-secret" not in out
        assert "montpellier_metropole" in out
        assert "FAILED" in out
