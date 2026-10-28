"""Provider-free resume protocol (#133).

The end of the file drives the whole protocol on real files: plan, execute part of it,
kill the run, re-plan, finish, and confirm a third plan asks for nothing. Everything above
pins one decision each — mostly the decisions to STOP, which are the ones a helpful
refactor removes first.
"""

from __future__ import annotations

import hashlib
import json
from datetime import date
from pathlib import Path

import pytest

from src.cli import run_resume as cli
from src.run_resume import (
    BLOCKING_STATUSES,
    SOURCE_MANIFEST_SCHEMA_VERSION,
    UNIT_STATUSES,
    ResumeValidationError,
    SourceArtifact,
    SourceManifest,
    WorkUnit,
    bind_plan,
    checkpoint_record,
    checkpoints_from_records,
    plan_run,
    planned_status,
    runnable_units,
    unit_fingerprint,
)

DIGEST_A = "a" * 64
DIGEST_B = "b" * 64
MANIFEST_DIGEST = "c" * 64


def _unit(unit_id="montpellier__2021", **overrides):
    payload = {
        "unit_id": unit_id,
        "zone_id": "montpellier_metropole",
        "start_date": "2021-01-01",
        "end_date": "2021-12-31",
        "output_uri": f"/panel/{unit_id}.parquet",
        "sources": [
            {"kind": "era5_parquet", "uri": f"/stored/era5/{unit_id}.parquet", "sha256": DIGEST_A},
            {"kind": "ndvi_parquet", "uri": f"/stored/ndvi/{unit_id}.parquet", "sha256": DIGEST_B},
        ],
    }
    payload.update(overrides)
    return payload


def _manifest(*units, run_id="panel-2021", digest=MANIFEST_DIGEST):
    return SourceManifest.from_mapping(
        {
            "schema_version": SOURCE_MANIFEST_SCHEMA_VERSION,
            "run_id": run_id,
            "units": list(units) or [_unit()],
        },
        digest=digest,
    )


def _resolver(**overrides):
    """A resolver over a declared world: URI -> digest, with ``None`` for unreachable."""

    def resolve(uri: str) -> str | None:
        if uri in overrides:
            return overrides[uri]
        if uri.startswith("/stored/era5/"):
            return DIGEST_A
        if uri.startswith("/stored/ndvi/"):
            return DIGEST_B
        return None

    return resolve


def _plan(manifest=None, checkpoints=None, resolve=None):
    return plan_run(
        manifest or _manifest(),
        checkpoints or {},
        resolve=resolve or _resolver(),
    )


def _status(plan, unit_id="montpellier__2021"):
    return next(entry["status"] for entry in plan["units"] if entry["unit_id"] == unit_id)


def _record_digest(record: dict) -> str:
    """The digest `supersedes` names a checkpoint by — the same one `checkpoints_from_records`
    computes over a record's canonical JSON."""
    return hashlib.sha256(
        json.dumps(record, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


# ---------------------------------------------------------------------------------------
# What resume is keyed on. `forecast_archive_panel.py` resumes on six `coverage` fields of
# the REQUEST, so a re-materialized source leaves a stale artifact matching its own
# coverage and the panel reports `resumed`.
# ---------------------------------------------------------------------------------------


def test_the_fingerprint_follows_the_inputs_not_the_request():
    same_request = _unit()
    moved_source = _unit()
    moved_source["sources"][0]["sha256"] = "d" * 64
    first = _manifest(same_request).units[0]
    second = _manifest(moved_source).units[0]
    assert first.zone_id == second.zone_id and first.output_uri == second.output_uri
    assert unit_fingerprint(first) != unit_fingerprint(second)


def test_units_can_be_built_without_a_manifest_file():
    """The manifest is one way in, not the only one: a caller may assemble units in code."""
    unit = WorkUnit(
        unit_id="built__2021",
        zone_id="zone_a",
        start_date=date(2021, 1, 1),
        end_date=date(2021, 12, 31),
        output_uri="/panel/built.parquet",
        sources=(SourceArtifact(kind="era5_parquet", uri="/stored/era5/x", sha256=DIGEST_A),),
    )
    manifest = SourceManifest(run_id="ad-hoc", units=(unit,), digest=MANIFEST_DIGEST)
    plan = plan_run(
        manifest, {}, resolve=lambda uri: DIGEST_A if uri.startswith("/stored/") else None
    )
    assert _status(plan, "built__2021") == "pending"
    assert plan["units"][0]["fingerprint"] == unit_fingerprint(unit)


def test_the_fingerprint_ignores_the_order_sources_were_written_in():
    forward = _unit()
    backward = _unit()
    backward["sources"] = list(reversed(backward["sources"]))
    assert unit_fingerprint(_manifest(forward).units[0]) == unit_fingerprint(
        _manifest(backward).units[0]
    )


def test_a_completed_unit_is_reused_when_nothing_moved():
    unit = _manifest().units[0]
    checkpoints = {
        unit.unit_id: checkpoints_from_records(
            [checkpoint_record(unit, output_sha256=DIGEST_A, completed_at="2026-09-23T00:00:00Z")]
        )[unit.unit_id]
    }
    plan = _plan(checkpoints=checkpoints, resolve=_resolver(**{unit.output_uri: DIGEST_A}))
    assert _status(plan) == "reuse"
    assert plan["runnable"] is True
    assert runnable_units(_manifest(), plan) == ()


def test_a_source_that_moved_under_a_completed_unit_stops_the_run():
    """Re-running overwrites an accepted artifact; keeping it publishes a false provenance.

    Neither is a decision this layer may take, so it takes neither and says why.
    """
    unit = _manifest().units[0]
    checkpoints = checkpoints_from_records(
        [checkpoint_record(unit, output_sha256=DIGEST_A, completed_at="2026-09-23T00:00:00Z")]
    )
    moved = _unit()
    moved["sources"][0]["sha256"] = "d" * 64
    plan = plan_run(
        _manifest(moved),
        checkpoints,
        resolve=_resolver(**{unit.output_uri: DIGEST_A, moved["sources"][0]["uri"]: "d" * 64}),
    )
    assert _status(plan) == "source_changed"
    assert plan["runnable"] is False
    assert "overwrite" in plan["blocked"][0]["notes"][0]


def test_an_artifact_written_outside_the_protocol_stops_the_run():
    unit = _manifest().units[0]
    checkpoints = checkpoints_from_records(
        [checkpoint_record(unit, output_sha256=DIGEST_A, completed_at="2026-09-23T00:00:00Z")]
    )
    plan = _plan(checkpoints=checkpoints, resolve=_resolver(**{unit.output_uri: "e" * 64}))
    assert _status(plan) == "output_modified"
    assert plan["runnable"] is False


def test_a_vanished_output_becomes_work_again_rather_than_a_conflict():
    """Nothing to overwrite and nothing to trust: the honest status is `pending`."""
    unit = _manifest().units[0]
    checkpoints = checkpoints_from_records(
        [checkpoint_record(unit, output_sha256=DIGEST_A, completed_at="2026-09-23T00:00:00Z")]
    )
    plan = _plan(checkpoints=checkpoints)  # the output resolves to None
    assert _status(plan) == "pending"
    assert plan["runnable"] is True
    assert "gone" in plan["units"][0]["notes"][0]


def test_an_artifact_no_checkpoint_claims_stops_the_run():
    """An unclaimed artifact is not a free slot.

    It is either an accepted result whose checkpoint log was lost, or the half-written
    remains of a run that died mid-write. Both look identical from here and executing over
    either destroys it, so "must not overwrite accepted artifacts" has to survive a lost
    checkpoint file — not only an intact one.
    """
    plan = _plan(resolve=_resolver(**{"/panel/montpellier__2021.parquet": "e" * 64}))
    assert _status(plan) == "output_unclaimed"
    assert plan["runnable"] is False
    assert "would overwrite it" in plan["blocked"][0]["notes"][0]


def test_a_unit_with_no_checkpoint_and_no_artifact_is_simply_work():
    plan = _plan()
    assert _status(plan) == "pending"
    assert plan["runnable"] is True
    assert runnable_units(_manifest(), plan)[0].unit_id == "montpellier__2021"


# ---------------------------------------------------------------------------------------
# The input side. A planner that cannot read a source must not pronounce on its unit.
# ---------------------------------------------------------------------------------------


def test_an_unreachable_source_blocks_the_unit():
    plan = _plan(resolve=lambda uri: None)
    assert _status(plan) == "source_unreachable"
    assert plan["runnable"] is False


def test_a_source_that_does_not_match_its_declared_digest_blocks_the_unit():
    plan = _plan(resolve=_resolver(**{"/stored/era5/montpellier__2021.parquet": "f" * 64}))
    assert _status(plan) == "source_mismatch"
    assert plan["runnable"] is False


def test_unreadable_sources_outrank_a_checkpoint():
    """Reporting `reuse` over inputs nobody could read would be the worst available answer."""
    unit = _manifest().units[0]
    checkpoints = checkpoints_from_records(
        [checkpoint_record(unit, output_sha256=DIGEST_A, completed_at="2026-09-23T00:00:00Z")]
    )
    plan = plan_run(_manifest(), checkpoints, resolve=lambda uri: None)
    assert _status(plan) == "source_unreachable"


def test_one_blocked_unit_stops_the_whole_run():
    """A partial panel published beside an unexplained conflict is the artifact to prevent."""
    good = _unit("zone_a__2021")
    bad = _unit("zone_b__2021")
    plan = plan_run(
        _manifest(good, bad),
        {},
        resolve=_resolver(**{bad["sources"][0]["uri"]: None}),
    )
    assert plan["counts"]["pending"] == 1
    assert plan["counts"]["source_unreachable"] == 1
    assert plan["runnable"] is False
    assert runnable_units(_manifest(good, bad), plan) == ()


def test_every_status_is_published_even_at_zero():
    counts = _plan()["counts"]
    assert set(counts) == set(UNIT_STATUSES)
    assert counts["pending"] == 1
    assert all(count == 0 for status, count in counts.items() if status != "pending")


def test_every_blocking_status_actually_blocks():
    assert set(BLOCKING_STATUSES) <= set(UNIT_STATUSES)
    assert "pending" not in BLOCKING_STATUSES and "reuse" not in BLOCKING_STATUSES


def test_checkpoints_for_units_the_manifest_dropped_are_reported():
    """Not an error — a manifest may narrow a panel — but silence is how a shrinking run
    stops accounting for artifacts it already produced."""
    retired = _manifest(_unit("retired__2020")).units[0]
    checkpoints = checkpoints_from_records(
        [checkpoint_record(retired, output_sha256=DIGEST_A, completed_at="2026-09-23T00:00:00Z")]
    )
    plan = plan_run(_manifest(), checkpoints, resolve=_resolver())
    assert plan["orphan_checkpoints"] == ["retired__2020"]
    assert plan["runnable"] is True


def test_the_plan_names_the_manifest_it_planned_from():
    plan = plan_run(_manifest(digest=MANIFEST_DIGEST), {}, resolve=_resolver(), code_ref={"c": "1"})
    assert plan["source_manifest_digest"] == MANIFEST_DIGEST
    assert plan["code_ref"] == {"c": "1"}


# ---------------------------------------------------------------------------------------
# A plan is a file, and a file outlives the manifest it was computed from.
# ---------------------------------------------------------------------------------------


def test_a_plan_from_another_manifest_is_refused():
    """Matching on `unit_id` alone hands back units the plan never examined.

    Every blocking check in this module is then bypassed by editing a manifest and reusing
    yesterday's plan, which is what `runnable_units` allowed before #201's review.
    """
    plan = _plan()
    other = _manifest(digest="9" * 64)
    with pytest.raises(ResumeValidationError, match="was computed from manifest"):
        runnable_units(other, plan)


def test_a_plan_whose_unit_changed_since_is_refused():
    """Same id, same manifest digest supplied, different sources: the plan is stale."""
    plan = _plan()
    moved = _unit()
    moved["sources"][0]["sha256"] = "d" * 64
    with pytest.raises(ResumeValidationError, match="changed since this plan was computed"):
        bind_plan(_manifest(moved), plan)


def test_a_plan_that_does_not_cover_a_unit_is_refused():
    plan = _plan()
    with pytest.raises(ResumeValidationError, match="does not cover unit"):
        bind_plan(_manifest(_unit(), _unit("added__2021")), plan)


def test_planned_status_reads_the_status_the_plan_gave():
    plan = _plan()
    assert planned_status(_manifest(), plan, "montpellier__2021") == "pending"
    with pytest.raises(ResumeValidationError, match="does not cover unit 'ghost'"):
        planned_status(_manifest(), plan, "ghost")


# ---------------------------------------------------------------------------------------
# What the manifest refuses.
# ---------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ({"schema_version": "other", "run_id": "r", "units": [_unit()]}, "schema_version"),
        ({"schema_version": SOURCE_MANIFEST_SCHEMA_VERSION, "units": [_unit()]}, "run_id"),
        (
            {"schema_version": SOURCE_MANIFEST_SCHEMA_VERSION, "run_id": "r", "units": []},
            "non-empty units array",
        ),
    ],
)
def test_a_malformed_manifest_is_refused(payload, message):
    with pytest.raises(ResumeValidationError, match=message):
        SourceManifest.from_mapping(payload, digest=MANIFEST_DIGEST)


def test_a_source_without_a_digest_is_refused():
    """A source named only by URI makes "the same inputs" a statement about a path."""
    unit = _unit()
    del unit["sources"][0]["sha256"]
    with pytest.raises(ResumeValidationError, match="sources\\[0\\].sha256"):
        _manifest(unit)


@pytest.mark.parametrize("digest", ["short", "z" * 64, 42, None])
def test_a_source_digest_that_is_not_sha256_is_refused(digest):
    unit = _unit()
    unit["sources"][0]["sha256"] = digest
    with pytest.raises(ResumeValidationError, match="sha256 hex string"):
        _manifest(unit)


def test_two_units_writing_the_same_artifact_are_refused():
    """The second silently destroys the first, and a later checkpoint describes a coin toss."""
    first = _unit("zone_a__2021", output_uri="/panel/shared.parquet")
    second = _unit("zone_b__2021", output_uri="/panel/shared.parquet")
    with pytest.raises(ResumeValidationError, match="same output_uri"):
        _manifest(first, second)


def test_a_duplicate_unit_id_is_refused():
    with pytest.raises(ResumeValidationError, match="duplicate unit_id"):
        _manifest(_unit(), _unit())


def test_the_same_source_uri_declared_twice_is_refused():
    unit = _unit()
    unit["sources"].append(dict(unit["sources"][0], sha256="d" * 64))
    with pytest.raises(ResumeValidationError, match="twice"):
        _manifest(unit)


def test_a_backwards_window_is_refused():
    with pytest.raises(ResumeValidationError, match="must not be after end_date"):
        _manifest(_unit(start_date="2021-12-31", end_date="2021-01-01"))


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda unit: unit.update(start_date="2021-13-40"), "start_date must be an ISO date"),
        (lambda unit: unit.update(end_date=20211231), "end_date must be an ISO date"),
        (lambda unit: unit.update(sources=[]), "sources must be a non-empty array"),
        (lambda unit: unit.update(sources=["era5"]), "sources\\[0\\] must be an object"),
        (lambda unit: unit.pop("output_uri"), "missing: output_uri"),
    ],
)
def test_a_malformed_unit_is_refused(mutate, message):
    unit = _unit()
    mutate(unit)
    with pytest.raises(ResumeValidationError, match=message):
        _manifest(unit)


@pytest.mark.parametrize("payload", [["not", "an", "object"], "text", 7])
def test_a_manifest_that_is_not_an_object_is_refused(payload):
    with pytest.raises(ResumeValidationError, match="schema_version|must be an object"):
        SourceManifest.from_mapping(payload, digest=MANIFEST_DIGEST)


def test_a_unit_that_is_not_an_object_is_refused():
    with pytest.raises(ResumeValidationError, match="units\\[0\\] must be an object"):
        _manifest(["montpellier"])


def test_a_checkpoint_record_that_is_not_an_object_is_refused():
    with pytest.raises(ResumeValidationError, match="checkpoints\\[0\\] must be an object"):
        checkpoints_from_records(["done"])


def test_a_checkpoint_log_describing_one_unit_twice_is_refused():
    """Last-write-wins would make the resume decision depend on append order."""
    unit = _manifest().units[0]
    record = checkpoint_record(unit, output_sha256=DIGEST_A, completed_at="2026-09-23T00:00:00Z")
    with pytest.raises(ResumeValidationError, match="duplicate checkpoint"):
        checkpoints_from_records([record, record])


def test_a_checkpoint_missing_its_fingerprint_is_refused():
    with pytest.raises(ResumeValidationError, match="missing: fingerprint"):
        checkpoints_from_records(
            [
                {
                    "unit_id": "u",
                    "output_uri": "/o",
                    "output_sha256": DIGEST_A,
                    "completed_at": "t",
                }
            ]
        )


# ---------------------------------------------------------------------------------------
# The CLI, and the full crash-and-restart walk at the bottom.
# ---------------------------------------------------------------------------------------


def _write(path: Path, payload: bytes | str) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = payload.encode("utf-8") if isinstance(payload, str) else payload
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


@pytest.fixture
def world(tmp_path):
    """Two units over real files, with a manifest that pins the source bytes."""
    sources = {}
    units = []
    for zone in ("zone_a", "zone_b"):
        era5 = tmp_path / "stored" / f"{zone}-era5.parquet"
        ndvi = tmp_path / "stored" / f"{zone}-ndvi.parquet"
        sources[era5] = _write(era5, f"era5 bytes for {zone}")
        sources[ndvi] = _write(ndvi, f"ndvi bytes for {zone}")
        units.append(
            {
                "unit_id": f"{zone}__2021",
                "zone_id": zone,
                "start_date": "2021-01-01",
                "end_date": "2021-12-31",
                "output_uri": str(tmp_path / "panel" / f"{zone}-2021.parquet"),
                "sources": [
                    {"kind": "era5_parquet", "uri": str(era5), "sha256": sources[era5]},
                    {"kind": "ndvi_parquet", "uri": str(ndvi), "sha256": sources[ndvi]},
                ],
            }
        )
    manifest = tmp_path / "sources.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": SOURCE_MANIFEST_SCHEMA_VERSION,
                "run_id": "panel-2021",
                "units": units,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return {
        "root": tmp_path,
        "manifest": manifest,
        "checkpoints": tmp_path / "checkpoints.jsonl",
        "units": units,
        "sources": sources,
    }


def _cli(world, *argv):
    return cli.main(
        [
            *argv,
            "--source-manifest",
            str(world["manifest"]),
            "--checkpoints",
            str(world["checkpoints"]),
        ]
    )


def _take_plan(world, capsys=None, name="run"):
    """A plan, taken BEFORE anything is written — which is the whole point of its authority.

    `record` only accepts a unit the plan left `pending`, so a plan taken after the
    artifact exists would already read `output_unclaimed` and refuse. One plan per run,
    reused for every unit recorded from it.
    """
    plan_path = world["root"] / "plans" / f"{name}.json"
    code = _cli(world, "plan", "--output", str(plan_path))
    if capsys is not None:
        capsys.readouterr()
    assert code == 0, "the fixture world should plan cleanly"
    return plan_path


def _record(world, plan_path, unit_id, completed_at="t1"):
    return _cli(
        world,
        "record",
        "--plan",
        str(plan_path),
        "--unit-id",
        unit_id,
        "--completed-at",
        completed_at,
    )


def _out(capsys):
    return json.loads(capsys.readouterr().out)


def test_cli_plan_on_a_fresh_run_wants_every_unit(world, capsys):
    assert _cli(world, "plan") == 0
    summary = _out(capsys)
    assert summary["status"] == "ok"
    assert summary["counts"]["pending"] == 2
    assert summary["runnable"] is True


def test_cli_plan_writes_the_full_plan_when_asked(world, capsys):
    plan_path = world["root"] / "plans" / "plan.json"
    assert _cli(world, "plan", "--output", str(plan_path)) == 0
    capsys.readouterr()
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    assert [entry["unit_id"] for entry in plan["units"]] == ["zone_a__2021", "zone_b__2021"]
    assert plan["code_ref"]["repo"] == "deposium_geoai"


def test_cli_refuses_to_record_a_unit_whose_artifact_is_not_there(world, capsys):
    """A checkpoint written ahead of the bytes is how a crashed run claims work it never did."""
    plan = _take_plan(world, capsys)
    assert _record(world, plan, "zone_a__2021", "t") == 1
    assert "record a unit only after its artifact is written" in _out(capsys)["error"]
    assert not world["checkpoints"].exists()


def test_cli_refuses_to_record_a_unit_the_manifest_does_not_declare(world, capsys):
    plan = _take_plan(world, capsys)
    assert _record(world, plan, "ghost", "t") == 1
    assert "not declared by the manifest" in _out(capsys)["error"]


def test_cli_refuses_a_second_checkpoint_for_one_unit(world, capsys):
    plan = _take_plan(world, capsys)
    _write(Path(world["units"][0]["output_uri"]), "panel bytes")
    assert _record(world, plan, "zone_a__2021", "t1") == 0
    capsys.readouterr()
    assert _record(world, plan, "zone_a__2021", "t2") == 1
    assert "already has a checkpoint" in _out(capsys)["error"]


def test_cli_plan_exits_two_when_the_answer_is_stop(world, capsys):
    """Exit 2, not 1: a blocked plan is the answer that was asked for, not a broken input."""
    Path(world["units"][0]["sources"][0]["uri"]).unlink()
    assert _cli(world, "plan") == 2
    summary = _out(capsys)
    assert summary["status"] == "blocked"
    assert summary["blocked"][0]["status"] == "source_unreachable"


def test_cli_verify_reports_an_artifact_that_changed(world, capsys):
    plan = _take_plan(world, capsys)
    output = Path(world["units"][0]["output_uri"])
    _write(output, "panel bytes")
    assert _record(world, plan, "zone_a__2021", "t1") == 0
    capsys.readouterr()
    assert _cli(world, "verify") == 0
    assert _out(capsys)["verified"] == 1
    _write(output, "somebody edited this")
    assert _cli(world, "verify") == 2
    summary = _out(capsys)
    assert summary["status"] == "blocked"
    assert summary["failures"] == [{"unit_id": "zone_a__2021", "reason": "output_modified"}]


def test_cli_refuses_an_unreadable_manifest(tmp_path, capsys):
    assert (
        cli.main(
            [
                "plan",
                "--source-manifest",
                str(tmp_path / "absent.json"),
                "--checkpoints",
                str(tmp_path / "c.jsonl"),
            ]
        )
        == 1
    )
    assert "cannot read source manifest" in _out(capsys)["error"]


def test_cli_refuses_a_corrupt_checkpoint_log(world, capsys):
    world["checkpoints"].write_text('{"unit_id": "a"}\n{ not json\n', encoding="utf-8")
    assert _cli(world, "plan") == 1
    assert "invalid checkpoint JSONL" in _out(capsys)["error"]


def test_cli_refuses_a_manifest_that_is_not_json(world, capsys):
    world["manifest"].write_bytes(b"\xff\xfe not json at all")
    assert _cli(world, "plan") == 1
    assert "must contain a JSON source manifest" in _out(capsys)["error"]


def test_cli_refuses_a_checkpoint_log_it_cannot_decode(world, capsys):
    world["checkpoints"].write_bytes(b"\xff\xfe\x00")
    assert _cli(world, "plan") == 1
    assert "cannot read checkpoints" in _out(capsys)["error"]


def test_cli_ignores_blank_lines_in_the_checkpoint_log(world, capsys):
    """An append that raced a newline must not invalidate a whole run's history."""
    plan = _take_plan(world, capsys)
    _write(Path(world["units"][0]["output_uri"]), "zone_a panel bytes")
    assert _record(world, plan, "zone_a__2021", "t1") == 0
    capsys.readouterr()
    with world["checkpoints"].open("a", encoding="utf-8") as stream:
        stream.write("\n   \n")
    assert _cli(world, "plan") == 0
    assert _out(capsys)["counts"]["reuse"] == 1


def test_cli_verify_reports_an_artifact_that_vanished(world, capsys):
    plan = _take_plan(world, capsys)
    output = Path(world["units"][0]["output_uri"])
    _write(output, "panel bytes")
    assert _record(world, plan, "zone_a__2021", "t1") == 0
    capsys.readouterr()
    output.unlink()
    assert _cli(world, "verify") == 2
    assert _out(capsys)["failures"] == [{"unit_id": "zone_a__2021", "reason": "output_unreadable"}]


def _recorded_unit(world, capsys):
    plan = _take_plan(world, capsys)
    _write(Path(world["units"][0]["output_uri"]), "panel bytes")
    assert _record(world, plan, "zone_a__2021", "t1") == 0
    capsys.readouterr()
    return world["units"][0]


def test_cli_verify_hashes_the_live_sources(world, capsys):
    """A source that no longer matches its declared digest fails verification.

    Before #201's review `verify` compared the checkpoint fingerprint with one derived
    from the same unchanged manifest — two ways of saying the same thing — so it could
    not see the files at all and returned success over stale provenance.
    """
    _recorded_unit(world, capsys)
    payload = json.loads(world["manifest"].read_text(encoding="utf-8"))
    payload["units"][0]["sources"][0]["sha256"] = "d" * 64
    world["manifest"].write_text(json.dumps(payload, indent=2), encoding="utf-8")
    assert _cli(world, "verify") == 2
    assert _out(capsys)["failures"] == [{"unit_id": "zone_a__2021", "reason": "source_mismatch"}]


def test_cli_verify_sees_a_source_re_materialized_and_re_declared(world, capsys):
    """The live files agree again; the recorded artifact still came from the old ones.

    This is why both checks are kept: hashing the files alone would now pass, and the
    fingerprint alone could not have seen the previous case.
    """
    first = _recorded_unit(world, capsys)
    source = Path(first["sources"][0]["uri"])
    _write(source, "era5 bytes, re-materialized")
    payload = json.loads(world["manifest"].read_text(encoding="utf-8"))
    payload["units"][0]["sources"][0]["sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
    world["manifest"].write_text(json.dumps(payload, indent=2), encoding="utf-8")
    assert _cli(world, "verify") == 2
    assert _out(capsys)["failures"] == [{"unit_id": "zone_a__2021", "reason": "source_changed"}]


def test_cli_verify_sees_a_checkpoint_pointing_at_another_artifact(world, capsys):
    """`plan` reads the manifest's output_uri and `verify` reads the checkpoint's.

    Left to diverge, one command approves bytes the other never looked at.
    """
    _recorded_unit(world, capsys)
    lines = world["checkpoints"].read_text(encoding="utf-8").splitlines()
    record = json.loads(lines[0])
    elsewhere = world["root"] / "elsewhere.parquet"
    record["output_uri"] = str(elsewhere)
    record["output_sha256"] = _write(elsewhere, "different bytes entirely")
    world["checkpoints"].write_text(json.dumps(record, sort_keys=True) + "\n", encoding="utf-8")
    assert _cli(world, "verify") == 2
    assert _out(capsys)["failures"] == [
        {"unit_id": "zone_a__2021", "reason": "checkpoint_mismatch"}
    ]
    assert _cli(world, "plan") == 2
    assert _out(capsys)["blocked"][0]["status"] == "checkpoint_mismatch"


@pytest.mark.parametrize(
    ("uri", "local"),
    [
        ("/stored/x.parquet", True),
        ("file:///stored/x.parquet", True),
        ("file://localhost/stored/x.parquet", True),
        ("s3://bucket/stored/x.parquet", False),
        ("https://example.org/x.parquet", False),
    ],
)
def test_only_local_uris_resolve(uri, local):
    """`Path("s3://bucket/key")` collapses to `s3:/bucket/key`, so a Path-based guard would
    treat a remote URI as a relative directory instead of refusing it."""
    assert (cli._local_path(uri) is not None) is local


def test_a_remote_source_blocks_rather_than_being_treated_as_a_local_path(tmp_path, capsys):
    manifest = tmp_path / "sources.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": SOURCE_MANIFEST_SCHEMA_VERSION,
                "run_id": "r",
                "units": [
                    _unit(
                        sources=[
                            {
                                "kind": "era5_parquet",
                                "uri": "s3://mesh/era5.parquet",
                                "sha256": DIGEST_A,
                            }
                        ]
                    )
                ],
            }
        ),
        encoding="utf-8",
    )
    assert (
        cli.main(
            ["plan", "--source-manifest", str(manifest), "--checkpoints", str(tmp_path / "c.jsonl")]
        )
        == 2
    )
    assert _out(capsys)["blocked"][0]["status"] == "source_unreachable"


def test_cli_refuses_a_control_file_pointed_at_declared_data(world, capsys):
    """`--output` truncates its target and `--checkpoints` appends to its own.

    Pointed at a source Parquet or an accepted artifact, either corrupts exactly the bytes
    this layer exists to protect — and no status in the protocol can report damage the
    protocol itself caused (#201 review).
    """
    source = world["units"][0]["sources"][0]["uri"]
    before = Path(source).read_bytes()
    assert _cli(world, "plan", "--output", source) == 1
    assert "points at a source of 'zone_a__2021'" in _out(capsys)["error"]
    assert Path(source).read_bytes() == before


def test_cli_refuses_a_checkpoint_log_pointed_at_an_artifact(world, capsys):
    output = world["units"][0]["output_uri"]
    before = _write(Path(output), "accepted panel bytes")
    assert (
        cli.main(
            [
                "plan",
                "--source-manifest",
                str(world["manifest"]),
                "--checkpoints",
                output,
            ]
        )
        == 1
    )
    assert "points at the output of 'zone_a__2021'" in _out(capsys)["error"]
    assert hashlib.sha256(Path(output).read_bytes()).hexdigest() == before


def test_cli_refuses_plan_output_colliding_with_checkpoints(world, capsys):
    """`plan --output` truncates its target; pointed at `--checkpoints` it destroys the log.

    Both are control files, so `_refuse_writing_over_data` (which only guards declared
    source and output paths) never sees this collision — it needs its own check (N1, #201
    review).
    """
    _write(world["checkpoints"], "an existing checkpoint log")
    before = world["checkpoints"].read_bytes()
    assert _cli(world, "plan", "--output", str(world["checkpoints"])) == 1
    error = _out(capsys)["error"]
    assert "--output" in error and "--checkpoints" in error
    assert world["checkpoints"].read_bytes() == before


def test_cli_refuses_plan_output_colliding_with_the_source_manifest(world, capsys):
    before = world["manifest"].read_bytes()
    assert _cli(world, "plan", "--output", str(world["manifest"])) == 1
    error = _out(capsys)["error"]
    assert "--output" in error and "--source-manifest" in error
    assert world["manifest"].read_bytes() == before


def test_cli_refuses_record_checkpoints_colliding_with_plan(world, capsys):
    plan_path = _take_plan(world, capsys)
    before = plan_path.read_bytes()
    assert (
        cli.main(
            [
                "record",
                "--source-manifest",
                str(world["manifest"]),
                "--checkpoints",
                str(plan_path),
                "--plan",
                str(plan_path),
                "--unit-id",
                "zone_a__2021",
                "--completed-at",
                "t1",
            ]
        )
        == 1
    )
    error = _out(capsys)["error"]
    assert "--checkpoints" in error and "--plan" in error
    assert plan_path.read_bytes() == before


def test_cli_refuses_record_checkpoints_colliding_with_the_source_manifest(world, capsys):
    plan_path = _take_plan(world, capsys)
    before = world["manifest"].read_bytes()
    assert (
        cli.main(
            [
                "record",
                "--source-manifest",
                str(world["manifest"]),
                "--checkpoints",
                str(world["manifest"]),
                "--plan",
                str(plan_path),
                "--unit-id",
                "zone_a__2021",
                "--completed-at",
                "t1",
            ]
        )
        == 1
    )
    error = _out(capsys)["error"]
    assert "--checkpoints" in error and "--source-manifest" in error
    assert world["manifest"].read_bytes() == before


def test_cli_refuses_to_record_a_unit_the_plan_blocked(world, capsys):
    """Otherwise a block is laundered into a `reuse` by the next plan.

    An artifact nothing claimed plans as `output_unclaimed`; recording it anyway would
    stamp it with the manifest-derived fingerprint and the following plan would report
    `reuse` over bytes whose provenance was never established.
    """
    _write(Path(world["units"][0]["output_uri"]), "bytes from nowhere")
    plan_path = world["root"] / "plans" / "blocked.json"
    assert _cli(world, "plan", "--output", str(plan_path)) == 2
    capsys.readouterr()
    assert _record(world, plan_path, "zone_a__2021", "t1") == 1
    error = _out(capsys)["error"]
    assert "output_unclaimed" in error and "only a unit the plan left pending" in error
    assert not world["checkpoints"].exists()


def test_cli_refuses_to_record_against_a_plan_for_another_manifest(world, capsys, tmp_path):
    plan = _take_plan(world, capsys)
    payload = json.loads(world["manifest"].read_text(encoding="utf-8"))
    payload["run_id"] = "renamed"
    world["manifest"].write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _write(Path(world["units"][0]["output_uri"]), "panel bytes")
    assert _record(world, plan, "zone_a__2021", "t1") == 1
    assert "was computed from manifest" in _out(capsys)["error"]


def test_cli_refuses_to_record_when_a_source_moved_since_the_plan(world, capsys):
    """The plan was taken before the unit ran; a source can move in between."""
    plan = _take_plan(world, capsys)
    _write(Path(world["units"][0]["output_uri"]), "panel bytes")
    _write(Path(world["units"][0]["sources"][0]["uri"]), "era5 bytes, re-materialized")
    assert _record(world, plan, "zone_a__2021", "t1") == 1
    assert "source_mismatch" in _out(capsys)["error"]
    assert not world["checkpoints"].exists()


@pytest.mark.parametrize("content", ["{ not json", "[1, 2]"])
def test_cli_refuses_an_unusable_plan_file(world, capsys, content):
    (world["root"] / "bad-plan.json").write_text(content, encoding="utf-8")
    _write(Path(world["units"][0]["output_uri"]), "panel bytes")
    assert _record(world, world["root"] / "bad-plan.json", "zone_a__2021", "t1") == 1
    assert "must contain a JSON plan" in _out(capsys)["error"]


def test_cli_refuses_a_missing_plan_file(world, capsys):
    _write(Path(world["units"][0]["output_uri"]), "panel bytes")
    assert _record(world, world["root"] / "absent.json", "zone_a__2021", "t1") == 1
    assert "cannot read plan" in _out(capsys)["error"]


# ---------------------------------------------------------------------------------------
# Crash and restart, on real files, end to end (#133's "restart after partial failure").
# ---------------------------------------------------------------------------------------


def test_the_protocol_survives_a_run_that_died_halfway(world, capsys):
    first, second = world["units"]
    plan = _take_plan(world, capsys)

    assert _cli(world, "plan") == 0
    assert _out(capsys)["counts"] == dict.fromkeys(UNIT_STATUSES, 0) | {"pending": 2}

    # Unit one completes and is recorded; the process then dies before unit two starts.
    _write(Path(first["output_uri"]), "zone_a panel bytes")
    assert _record(world, plan, first["unit_id"], "t1") == 0
    capsys.readouterr()

    # Restart: exactly one unit is still work, and nothing asks to redo the finished one.
    assert _cli(world, "plan") == 0
    resumed = _out(capsys)
    assert resumed["counts"]["reuse"] == 1
    assert resumed["counts"]["pending"] == 1
    assert resumed["runnable"] is True

    _write(Path(second["output_uri"]), "zone_b panel bytes")
    assert _record(world, plan, second["unit_id"], "t2") == 0
    capsys.readouterr()

    # A third plan asks for nothing: running it again is a no-op, which is idempotence.
    assert _cli(world, "plan") == 0
    final = _out(capsys)
    assert final["counts"]["reuse"] == 2
    assert final["counts"]["pending"] == 0
    assert _cli(world, "verify") == 0
    assert _out(capsys)["verified"] == 2


def test_a_source_re_materialized_after_the_run_stops_the_restart(world, capsys):
    """The defect `forecast_archive_panel.py`'s coverage-field resume cannot see."""
    plan = _take_plan(world, capsys)
    first = world["units"][0]
    _write(Path(first["output_uri"]), "zone_a panel bytes")
    assert _record(world, plan, first["unit_id"], "t1") == 0
    capsys.readouterr()

    # Same path, new bytes: the request is unchanged and the artifact is now stale.
    _write(Path(first["sources"][0]["uri"]), "era5 bytes, re-materialized")

    assert _cli(world, "plan") == 2
    blocked = _out(capsys)["blocked"]
    assert [entry["status"] for entry in blocked] == ["source_mismatch"]

    # And with the manifest updated to the new digest, the conflict becomes the honest one:
    # a completed unit whose inputs moved. Still a stop, now for the right reason.
    payload = json.loads(world["manifest"].read_text(encoding="utf-8"))
    payload["units"][0]["sources"][0]["sha256"] = hashlib.sha256(
        Path(first["sources"][0]["uri"]).read_bytes()
    ).hexdigest()
    world["manifest"].write_text(json.dumps(payload, indent=2), encoding="utf-8")

    assert _cli(world, "plan") == 2
    assert [entry["status"] for entry in _out(capsys)["blocked"]] == ["source_changed"]


def test_a_unit_whose_output_vanished_can_be_recorded_again_from_the_plan_that_saw_it_gone(
    world, capsys
):
    """N2: `_completed_status` marks a checkpointed unit `pending` again when its recorded
    output is gone — the sources did not move, so this is not `source_changed`, and there is
    nothing left on disk to overwrite. Before this fix, re-running such a unit and recording
    a *different* set of bytes at that path collided with `record`'s own "already has a
    checkpoint" refusal, and the plan taken afterwards saw `output_modified` either way.

    The plan now names what a second checkpoint replaces (`supersedes`), and `record` and
    `checkpoints_from_records` accept a second checkpoint only when it matches.
    """
    first = world["units"][0]
    output = Path(first["output_uri"])

    plan1 = _take_plan(world, capsys, name="plan1")
    _write(output, "first bytes")
    assert _record(world, plan1, first["unit_id"], "t1") == 0
    capsys.readouterr()
    first_record = json.loads(world["checkpoints"].read_text(encoding="utf-8").strip())
    first_record_digest = _record_digest(first_record)

    # The artifact vanishes. Nothing moved under it and nothing is left to overwrite, so the
    # honest status is `pending` again — with the checkpoint RECORD it would replace named
    # on it, not that record's output digest.
    output.unlink()
    plan2 = _take_plan(world, capsys, name="plan2")
    plan2_payload = json.loads(plan2.read_text(encoding="utf-8"))
    entry = next(e for e in plan2_payload["units"] if e["unit_id"] == first["unit_id"])
    assert entry["status"] == "pending"
    assert entry["supersedes"] == first_record_digest

    # Reproduced: the regenerated bytes differ from the first run's (a producer that embeds
    # a timestamp or a code_ref, not a byte-identical re-run).
    _write(output, "second, different bytes")
    assert _record(world, plan2, first["unit_id"], "t2") == 0
    capsys.readouterr()

    # `plan` at `reuse` (exit 0): zone_a is settled again; zone_b was never touched.
    assert _cli(world, "plan") == 0
    counts = _out(capsys)["counts"]
    assert counts["reuse"] == 1
    assert counts["pending"] == 1

    # `verify` at exit 0: the live bytes match the second checkpoint, and the fingerprint
    # still matches (the sources never moved).
    assert _cli(world, "verify") == 0
    assert _out(capsys)["verified"] == 1

    # Recording the same unit again with that same (now stale) plan is refused: its
    # `supersedes` names the FIRST checkpoint, not the one currently on file.
    assert _record(world, plan2, first["unit_id"], "t3") == 1
    assert "already has a checkpoint" in _out(capsys)["error"]


def test_a_plan_that_already_recorded_a_replacement_cannot_be_replayed_after_it_is_modified(
    world, capsys
):
    """N2, amended: naming the superseded checkpoint by its `output_sha256` has a gap a
    byte-identical regeneration opens. C1 is recorded, its output vanishes, and plan P names
    C1 as the checkpoint it would replace. The unit is then regenerated with IDENTICAL
    bytes, so `record --plan P` writes C2 with the same `output_sha256` as C1 — correct, and
    unchanged by this fix. But P must not still be usable once C2 exists: with `supersedes`
    naming C1's *output*, C2's `output_sha256` still matches it, so P could be replayed after
    C2's bytes are modified outside the protocol, laundering that modification into a new
    checkpoint. Naming C1's *record* instead closes this, because C2's own record
    (`completed_at`, `supersedes`) differs from C1's, and P no longer matches anything on
    file after step 2.
    """
    first = world["units"][0]
    output = Path(first["output_uri"])

    # 1. record C1 (output hash H), then the output vanishes.
    plan1 = _take_plan(world, capsys, name="plan1")
    _write(output, "identical bytes")
    assert _record(world, plan1, first["unit_id"], "t1") == 0
    capsys.readouterr()
    output.unlink()

    # 2. plan P: the unit is `pending`, `supersedes` names C1's record. The unit is
    # regenerated with bytes IDENTICAL to C1's, and `record --plan P` writes C2 — same
    # `output_sha256` as C1, correct so far.
    plan_p = _take_plan(world, capsys, name="plan_p")
    plan_p_payload = json.loads(plan_p.read_text(encoding="utf-8"))
    entry = next(e for e in plan_p_payload["units"] if e["unit_id"] == first["unit_id"])
    assert entry["status"] == "pending"
    assert entry["supersedes"] is not None
    _write(output, "identical bytes")
    assert _record(world, plan_p, first["unit_id"], "t2") == 0
    capsys.readouterr()

    # 3. the output is changed outside the protocol.
    _write(output, "changed outside the protocol")
    assert _cli(world, "verify") == 2
    assert _out(capsys)["failures"] == [{"unit_id": first["unit_id"], "reason": "output_modified"}]

    # 4. `record --plan P` a second time must be refused: P's `supersedes` still names C1's
    # record, but the checkpoint on file is now C2, whose own record differs from C1's. The
    # change made outside the protocol must not be laundered into a checkpoint.
    capsys.readouterr()
    assert _record(world, plan_p, first["unit_id"], "t3") == 1
    assert "already has a checkpoint" in _out(capsys)["error"]
    assert _cli(world, "verify") == 2
    assert _out(capsys)["failures"] == [{"unit_id": first["unit_id"], "reason": "output_modified"}]


def test_a_second_checkpoint_is_accepted_only_when_it_names_the_first_one_it_replaces():
    """`checkpoints_from_records` itself, without the CLI: the one case a duplicate
    unit_id is not refused is a `supersedes` that matches the digest of the prior record
    (not that record's `output_sha256`, which a byte-identical regeneration would still
    match after the record itself had changed).
    """
    unit = _manifest().units[0]
    first = checkpoint_record(unit, output_sha256=DIGEST_A, completed_at="t1")
    first_digest = _record_digest(first)
    matching = checkpoint_record(
        unit, output_sha256=DIGEST_B, completed_at="t2", supersedes=first_digest
    )
    indexed = checkpoints_from_records([first, matching])
    assert indexed[unit.unit_id].output_sha256 == DIGEST_B
    assert indexed[unit.unit_id].supersedes == first_digest

    mismatched = checkpoint_record(
        unit, output_sha256=DIGEST_B, completed_at="t2", supersedes="d" * 64
    )
    with pytest.raises(ResumeValidationError, match="duplicate checkpoint"):
        checkpoints_from_records([first, mismatched])


_REPO_ROOT = Path(__file__).resolve().parents[1]

#: Anything that can fetch, collect or talk to the object store. #133 forbids the resume
#: layer from invoking ERA5/Open-Meteo/STAC collection, and a docstring saying so is not a
#: guarantee — the next helper added to this module could import one without anyone noticing.
_COLLECTION_MODULES = (
    "httpx",
    "requests",
    "aiohttp",
    "urllib.request",
    "boto3",
    "duckdb",
    "src.utils.openmeteo",
    "src.utils.stac_client",
    "src.utils.duckdb_conn",
    "src.tools.stac",
    "src.tools.climate",
    "src.tools.forecast",
)


def _imported_modules(dotted: str) -> set[str]:
    """Every module name imported by ``dotted``, following first-party edges transitively."""
    import ast

    seen: set[str] = set()
    pending = [dotted]
    imports: set[str] = set()
    while pending:
        current = pending.pop()
        if current in seen:
            continue
        seen.add(current)
        path = _REPO_ROOT / (current.replace(".", "/") + ".py")
        if not path.exists():
            continue
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
                names = [node.module, *(f"{node.module}.{a.name}" for a in node.names)]
            else:
                continue
            imports.update(names)
            pending.extend(name for name in names if name.startswith("src."))
    return imports


@pytest.mark.parametrize("module", ["src.run_resume", "src.cli.run_resume"])
def test_the_resume_layer_cannot_reach_a_provider(module):
    """ "Provider-free" as a property of the import graph, not a line in a docstring.

    Transitive over first-party edges: a helper that imports `src.tools.climate` pulls in
    the whole collection path, and the direct import list would still look clean.
    """
    reached = _imported_modules(module)
    assert not reached & set(_COLLECTION_MODULES), sorted(reached & set(_COLLECTION_MODULES))


def test_the_provider_guard_would_notice():
    """A guard parametrized over a module that imports nothing forbidden proves nothing."""
    assert set(_COLLECTION_MODULES) & _imported_modules("src.tools.climate")
