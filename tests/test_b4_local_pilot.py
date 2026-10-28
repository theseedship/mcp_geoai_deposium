"""Hermetic tests for scripts/b4_local_pilot.py — the offline B4 pilot (#133).

Every test runs the real `climate_enrich_ndvi` through the real wrapper and the real #201
resume CLI over tiny local Parquets. STAC is replaced by a function that raises, so a unit that
reached a provider would fail rather than fetch.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

from src.config import settings
from src.tools import climate
from tests.test_unified_pilot_panel import _write_b3, _write_era5

_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "b4_local_pilot.py"
_spec = importlib.util.spec_from_file_location("b4_local_pilot", _SCRIPT_PATH)
pilot = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pilot)

ZONE = "montpellier_metropole"
DAYS = [f"2024-01-{day:02d}" for day in range(1, 11)]
OBSERVATIONS = [("2024-01-02", ZONE, 0.61), ("2024-01-05", ZONE, 0.63), ("2024-01-09", ZONE, 0.58)]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def workspace(tmp_path, monkeypatch, allow_local_parquet_paths):
    # Registered with monkeypatch BEFORE the script mutates them, so teardown restores them.
    for name in ("S3_BUCKET", "ALLOW_LOCAL_PARQUET_SOURCE", "LOCAL_PARQUET_SOURCE_ROOT"):
        monkeypatch.setattr(settings, name, getattr(settings, name))
    monkeypatch.setattr(climate, "_collect_ndvi_observations", climate._collect_ndvi_observations)
    (tmp_path / "in" / "era5").mkdir(parents=True)
    (tmp_path / "in" / "b3").mkdir(parents=True)
    _write_era5(tmp_path / "in" / "era5" / f"{ZONE}.parquet", ZONE, DAYS)
    _write_b3(tmp_path / "in" / "b3" / f"{ZONE}.parquet", OBSERVATIONS)
    return tmp_path


def _freeze(root: Path, *extra: str) -> int:
    return pilot.main(
        [
            "freeze",
            "--out",
            (root / "out").as_posix(),
            "--era5-template",
            (root / "in" / "era5" / "{zone_id}.parquet").as_posix(),
            "--b3-template",
            (root / "in" / "b3" / "{zone_id}.parquet").as_posix(),
            "--zone-id",
            ZONE,
            "--start-date",
            DAYS[0],
            "--end-date",
            DAYS[-1],
            "--tolerance-days",
            "1",
            "3",
            *extra,
        ]
    )


def _run(root: Path) -> int:
    return pilot.main(
        [
            "run",
            "--out",
            (root / "out").as_posix(),
            "--local-root",
            root.as_posix(),
            "--completed-at",
            "2026-09-29T00:00:00+00:00",
        ]
    )


def _outputs(root: Path) -> list[Path]:
    return sorted((root / "out" / "unified").rglob("*.parquet"))


def _plan_statuses(root: Path) -> dict[str, str]:
    plan = json.loads((root / "out" / "plan.json").read_text())
    return {entry["unit_id"]: entry["status"] for entry in plan["units"]}


def test_first_run_materializes_every_unit_from_stored_files_only(workspace):
    assert _freeze(workspace) == 0
    assert _run(workspace) == 0

    outputs = _outputs(workspace)
    assert [p.parent.name for p in outputs] == ["pm1d", "pm3d"]
    records = (workspace / "out" / "checkpoints.jsonl").read_text().splitlines()
    assert len(records) == 2

    assert pilot.main(["report", "--out", (workspace / "out").as_posix()]) == 0
    report = json.loads((workspace / "out" / "report.json").read_text())
    zone = report["zones"][ZONE]
    assert zone["era5"]["days_in_window"] == 10
    assert zone["era5"]["missing_days_in_window"] == []
    assert zone["b3_ndvi"]["observations_in_window"] == 3
    for readout in zone["tolerances"].values():
        assert readout["rows"] == 3
        assert readout["day_delta_counts"] == {"0": 3}
        assert readout["unmatched_ndvi_dates"] == []
    # Complete daily ERA5: every join is same-day, so the proposed tolerances cannot differ.
    assert zone["tolerance_outputs_byte_identical"] is True
    for stage in report["wrapper_stages"].values():
        assert stage["ndvi_input"] == "stored_b3_series"
        assert stage["images_searched"] == 0
    assert report["files"]["checkpoints.jsonl"] == _sha(workspace / "out" / "checkpoints.jsonl")


def test_a_second_run_reuses_every_unit_and_writes_nothing(workspace, monkeypatch):
    assert _freeze(workspace) == 0
    assert _run(workspace) == 0
    before = {p: (_sha(p), p.stat().st_mtime_ns) for p in _outputs(workspace)}
    log_before = _sha(workspace / "out" / "checkpoints.jsonl")

    def _must_not_materialize(*_a, **_k):
        raise AssertionError("a reused unit must not be materialized again")

    monkeypatch.setattr(pilot, "_materialize", _must_not_materialize)

    assert _run(workspace) == 0
    assert set(_plan_statuses(workspace).values()) == {"reuse"}
    assert {p: (_sha(p), p.stat().st_mtime_ns) for p in _outputs(workspace)} == before
    assert _sha(workspace / "out" / "checkpoints.jsonl") == log_before


def test_source_drift_blocks_the_run_before_anything_is_written(workspace, monkeypatch):
    assert _freeze(workspace) == 0
    assert _run(workspace) == 0
    before = {p: _sha(p) for p in _outputs(workspace)}
    _write_b3(workspace / "in" / "b3" / f"{ZONE}.parquet", [("2024-01-02", ZONE, 0.99)])
    monkeypatch.setattr(pilot, "_materialize", lambda *_a, **_k: pytest.fail("materialized"))

    assert _run(workspace) == 2
    assert set(_plan_statuses(workspace).values()) == {"source_mismatch"}
    assert {p: _sha(p) for p in _outputs(workspace)} == before


def test_output_drift_blocks_the_run_and_the_report(workspace, monkeypatch):
    assert _freeze(workspace) == 0
    assert _run(workspace) == 0
    tampered = _outputs(workspace)[0]
    tampered.write_bytes(tampered.read_bytes() + b"\0")
    monkeypatch.setattr(pilot, "_materialize", lambda *_a, **_k: pytest.fail("materialized"))

    assert _run(workspace) == 2
    assert "output_modified" in _plan_statuses(workspace).values()
    assert pilot.main(["report", "--out", (workspace / "out").as_posix()]) == 2
    assert not (workspace / "out" / "report.json").exists()


def test_an_unclaimed_output_is_never_overwritten(workspace):
    assert _freeze(workspace) == 0
    squatter = workspace / "out" / "unified" / "pm1d" / f"{ZONE}.parquet"
    squatter.parent.mkdir(parents=True)
    squatter.write_bytes(b"not ours")

    assert _run(workspace) == 2
    assert squatter.read_bytes() == b"not ours"
    assert not (workspace / "out" / "checkpoints.jsonl").exists()


def test_code_drift_since_freeze_is_refused(workspace, monkeypatch):
    assert _freeze(workspace) == 0
    real = pilot._code_identity

    def _drifted():
        identity = real()
        identity["files"]["src/tools/climate.py"] = "0" * 64
        return identity

    monkeypatch.setattr(pilot, "_code_identity", _drifted)

    assert _run(workspace) == 1
    assert _outputs(workspace) == []


def test_a_stac_call_fails_the_unit_instead_of_fetching(workspace, monkeypatch):
    """Belt and braces: even if the wrapper stopped passing the stored source, nothing fetches."""
    assert _freeze(workspace) == 0
    real = pilot.unified_pilot_panel.build_parser

    def _without_ndvi_source():
        parser = real()
        original = parser.parse_args

        def _parse(argv):
            argv = list(argv)
            index = argv.index("--ndvi-source-template")
            del argv[index : index + 2]
            return original(argv)

        parser.parse_args = _parse
        return parser

    monkeypatch.setattr(pilot.unified_pilot_panel, "build_parser", _without_ndvi_source)

    assert _run(workspace) == 1
    assert not (workspace / "out" / "checkpoints.jsonl").exists()


class TestFreeze:
    def test_refuses_to_overwrite_a_frozen_manifest(self, workspace):
        assert _freeze(workspace) == 0
        digest = _sha(workspace / "out" / "sources.json")

        assert _freeze(workspace) == 1
        assert _sha(workspace / "out" / "sources.json") == digest

    def test_refuses_an_absent_source(self, workspace):
        (workspace / "in" / "b3" / f"{ZONE}.parquet").unlink()

        assert _freeze(workspace) == 1
        assert not (workspace / "out" / "sources.json").exists()

    def test_refuses_a_source_that_differs_from_its_published_digest(self, workspace):
        published = workspace / "b3-verification.json"
        published.write_text(
            json.dumps(
                {
                    "artifacts": [
                        {"relative_path": f"climate/ndvi/{ZONE}.parquet", "sha256": "0" * 64}
                    ]
                }
            )
        )

        assert _freeze(workspace, "--b3-verification", published.as_posix()) == 1
        assert not (workspace / "out" / "sources.json").exists()

    def test_pins_sources_window_outputs_tolerances_and_code(self, workspace):
        era5 = workspace / "in" / "era5" / f"{ZONE}.parquet"
        bundle = workspace / "bundle-manifest.json"
        bundle.write_text(json.dumps({"objects": [{"zone_id": ZONE, "sha256": _sha(era5)}]}))

        assert _freeze(workspace, "--era5-bundle-manifest", bundle.as_posix()) == 0
        manifest = json.loads((workspace / "out" / "sources.json").read_text())

        assert manifest["schema_version"] == "run-resume-sources/v1"
        assert manifest["window"] == {"start_date": DAYS[0], "end_date": DAYS[-1]}
        assert manifest["alignment"]["proposed_tolerance_days"] == [1, 3]
        assert "not ratified" in manifest["alignment"]["status"]
        assert manifest["published_digests_checked"] == ["era5"]
        assert set(manifest["code_identity"]["files"]) == set(pilot.CODE_FILES)
        assert [u["unit_id"] for u in manifest["units"]] == [
            f"{ZONE}__2024_2024__pm1d",
            f"{ZONE}__2024_2024__pm3d",
        ]
        assert {s["kind"] for s in manifest["units"][0]["sources"]} == {"era5", "b3_ndvi"}
        assert manifest["units"][0]["sources"][0]["sha256"] == _sha(era5)


@pytest.mark.parametrize(
    "extra",
    [
        ("--zone-id", ZONE, ZONE),
        ("--tolerance-days", "1", "1"),
    ],
)
def test_freeze_refuses_duplicate_units_before_writing(workspace, extra):
    assert _freeze(workspace, *extra) == 1
    assert not (workspace / "out" / "sources.json").exists()


def test_duckdb_runtime_drift_refuses_run_and_report(workspace, monkeypatch):
    assert _freeze(workspace) == 0
    real_identity = pilot._code_identity

    def _new_runtime():
        return {**real_identity(), "duckdb_version": "99.99"}

    monkeypatch.setattr(pilot, "_code_identity", _new_runtime)
    assert _run(workspace) == 1
    assert _outputs(workspace) == []

    monkeypatch.setattr(pilot, "_code_identity", real_identity)
    assert _run(workspace) == 0
    monkeypatch.setattr(pilot, "_code_identity", _new_runtime)
    assert pilot.main(["report", "--out", (workspace / "out").as_posix()]) == 1
    assert not (workspace / "out" / "report.json").exists()


def test_summary_digests_are_pinned_once_and_unchanged_on_reuse(workspace):
    assert _freeze(workspace) == 0
    assert _run(workspace) == 0
    anchor = workspace / "out" / pilot.SUMMARY_DIGESTS_NAME
    before = (anchor.read_bytes(), anchor.stat().st_mtime_ns)
    pinned = json.loads(anchor.read_text())
    assert len(pinned["summaries"]) == 2
    checkpoints = {
        record["unit_id"]: record
        for record in (
            json.loads(line)
            for line in (workspace / "out" / "checkpoints.jsonl").read_text().splitlines()
        )
    }
    for unit_id, digests in pinned["summaries"].items():
        summary = next((workspace / "out" / "summaries" / unit_id).glob(f"{ZONE}-unified-*.json"))
        assert digests["summary_sha256"] == _sha(summary)
        assert digests["output_sha256"] == checkpoints[unit_id]["output_sha256"]

    assert _run(workspace) == 0
    assert (anchor.read_bytes(), anchor.stat().st_mtime_ns) == before


@pytest.mark.parametrize("change", ["modified", "missing", "ambiguous"])
def test_report_refuses_unverified_wrapper_summary(workspace, change):
    assert _freeze(workspace) == 0
    assert _run(workspace) == 0
    unit_id = f"{ZONE}__2024_2024__pm1d"
    summary_dir = workspace / "out" / "summaries" / unit_id
    summary = next(summary_dir.glob(f"{ZONE}-unified-*.json"))
    if change == "modified":
        payload = json.loads(summary.read_text())
        payload["stage"]["n_matched"] = 777
        summary.write_text(json.dumps(payload))
        assert _run(workspace) == 1  # Reuse must not accept changed wrapper evidence either.
    elif change == "missing":
        summary.unlink()
    else:
        (summary_dir / f"{ZONE}-unified-other.json").write_text("{}")

    assert pilot.main(["report", "--out", (workspace / "out").as_posix()]) == 1
    assert not (workspace / "out" / "report.json").exists()


def test_report_refuses_unpinned_legacy_summary(workspace):
    assert _freeze(workspace) == 0
    assert _run(workspace) == 0
    (workspace / "out" / pilot.SUMMARY_DIGESTS_NAME).unlink()

    assert _run(workspace) == 1
    assert pilot.main(["report", "--out", (workspace / "out").as_posix()]) == 1
    assert not (workspace / "out" / "report.json").exists()


def test_report_refuses_summary_anchor_not_bound_to_checkpoint(workspace):
    assert _freeze(workspace) == 0
    assert _run(workspace) == 0
    anchor = workspace / "out" / pilot.SUMMARY_DIGESTS_NAME
    payload = json.loads(anchor.read_text())
    unit_id = next(iter(payload["summaries"]))
    payload["summaries"][unit_id]["output_sha256"] = "0" * 64
    anchor.write_text(json.dumps(payload))

    assert pilot.main(["report", "--out", (workspace / "out").as_posix()]) == 1
    assert not (workspace / "out" / "report.json").exists()


def test_report_requires_checkpoint_for_every_frozen_unit(workspace):
    assert _freeze(workspace) == 0
    assert _run(workspace) == 0
    checkpoints = workspace / "out" / "checkpoints.jsonl"
    checkpoints.write_text(checkpoints.read_text().splitlines()[0] + "\n")

    assert pilot.main(["report", "--out", (workspace / "out").as_posix()]) == 1
    assert not (workspace / "out" / "report.json").exists()
