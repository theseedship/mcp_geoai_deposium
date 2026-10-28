"""The normals-only operator: stored ERA5 in, versioned staging out, explicit promotion (#172).

Every test here is hermetic and local. The operator refuses an object-store root by design —
#172 authorizes no overwrite of an existing environment — so there is nothing to mock away.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import normals_rematerialize as op
from src.config import settings
from src.tools.climate import _write_geoparquet
from src.utils.openmeteo import NORMALIZED_VARS

YEAR_START, YEAR_END = "2020-01-01", "2020-12-31"


def _write_era5_year(root: Path, zone_id: str, *, wet: bool = True) -> Path:
    """One synthetic ERA5 year at full daily density, which is what the coverage gate wants.

    ``wet=False`` gives a zone with no day above the 1 mm wet-day threshold — the arid case
    `niamey` really has for four months, and the one a naive wet-day coverage gate rejects.
    """
    from datetime import date, timedelta

    start, end = date.fromisoformat(YEAR_START), date.fromisoformat(YEAR_END)
    rows, day = [], start
    while day <= end:
        rows.append(
            {
                "date": day.isoformat(),
                "zone_id": zone_id,
                "lat": 13.5,
                "lon": 2.1,
                "wkt": "POINT(2.1 13.5)",
                "temperature_2m": 25.0 + day.month,
                "precipitation": (5.0 if day.day % 3 == 0 else 0.0) if wet else 0.0,
                "humidity": 50.0,
                "evapotranspiration": 3.0,
                "wind_speed": 4.0,
                "solar_radiation": 200.0,
            }
        )
        day += timedelta(days=1)

    dst = root / "climate" / "era5" / f"{zone_id}.parquet"
    dst.parent.mkdir(parents=True, exist_ok=True)
    _write_geoparquet(
        rows,
        columns=[
            ("date", "VARCHAR"),
            ("zone_id", "VARCHAR"),
            ("lat", "DOUBLE"),
            ("lon", "DOUBLE"),
            ("wkt", "VARCHAR"),
            *[(v, "DOUBLE") for v in NORMALIZED_VARS],
        ],
        dst=dst.as_posix(),
    )
    return dst


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "ALLOW_LOCAL_PARQUET_SOURCE", True)
    monkeypatch.setattr(settings, "LOCAL_PARQUET_SOURCE_ROOT", tmp_path.as_posix())
    return tmp_path


def _materialize(store: Path, *zones: str, run_id: str = "r1") -> int:
    return op.main(
        [
            "materialize",
            "--root",
            store.as_posix(),
            "--era5-root",
            store.as_posix(),
            "--run-id",
            run_id,
            "--zone-id",
            *zones,
            "--start-date",
            YEAR_START,
            "--end-date",
            YEAR_END,
        ]
    )


def _canonical(store: Path, zone_id: str) -> Path:
    return store / op._CANONICAL.format(zone_id=zone_id)


def _staged(store: Path, zone_id: str, run_id: str = "r1") -> Path:
    return store / op._STAGING.format(run_id=run_id, zone_id=zone_id)


def _kept(store: Path, zone_id: str, run_id: str = "r1") -> Path:
    return store / op._ROLLBACK.format(run_id=run_id, zone_id=zone_id)


def _schema_version(path: Path) -> int:
    import duckdb

    con = duckdb.connect(":memory:")
    try:
        names = {r[0] for r in con.execute(f"DESCRIBE SELECT * FROM '{path}'").fetchall()}
        if "schema_version" not in names:
            return 1
        return con.execute(f"SELECT MIN(schema_version) FROM '{path}'").fetchone()[0]
    finally:
        con.close()


class TestMaterializeTouchesNothingCanonical:
    def test_it_stages_v2_and_leaves_canonical_alone(self, store):
        _write_era5_year(store, "niamey")
        canonical = _canonical(store, "niamey")
        canonical.parent.mkdir(parents=True, exist_ok=True)
        canonical.write_bytes(b"not a parquet, and must not be read or replaced")

        assert _materialize(store, "niamey") == 0

        assert _schema_version(_staged(store, "niamey")) == op.NORMALS_SCHEMA_VERSION
        # Byte-identical: the step that computes is the step that must not touch canonical.
        assert canonical.read_bytes() == b"not a parquet, and must not be read or replaced"

    def test_a_missing_stored_era5_is_a_refusal_and_not_a_fetch(self, store, capsys):
        """The operator's premise. A missing input must not become a slower path.

        The refusal is about the missing PARQUET and fires before any zone resolution, which is
        why an unknown zone id reaches it at all.
        """
        assert _materialize(store, "atlantis") == 2
        assert "never fetches from the Archive" in capsys.readouterr().err

    def test_the_manifest_records_the_window_version_and_code(self, store):
        _write_era5_year(store, "niamey")
        _materialize(store, "niamey")

        manifest = json.loads(
            (store / op._MANIFEST.format(run_id="r1")).read_text(encoding="utf-8")
        )
        assert manifest["kind"] == "normals_rematerialization"
        assert manifest["target_schema_version"] == op.NORMALS_SCHEMA_VERSION
        assert manifest["window"] == {"start": YEAR_START, "end": YEAR_END}
        assert manifest["promoted"] is False
        assert "code_ref" in manifest

    def test_an_arid_zone_stages_successfully(self, store):
        """`n_wet = 0` is valid data, and a wet-day coverage gate would reject it.

        This is the shape `niamey` really has for January, February, November and December
        across 1991-2020.
        """
        # `niamey` and not a made-up id: `climate_percentiles` resolves the zone against
        # `zones/zones.geojson`, and it is also the zone that really has this shape.
        _write_era5_year(store, "niamey", wet=False)

        assert _materialize(store, "niamey") == 0

        import duckdb

        con = duckdb.connect(":memory:")
        rows = con.execute(
            f"SELECT n_wet, p90_wet FROM '{_staged(store, 'niamey')}' "
            "WHERE variable = 'precipitation'"
        ).fetchall()
        con.close()
        assert all(n_wet == 0 for n_wet, _ in rows)
        assert all(p90_wet is None for _, p90_wet in rows)


class TestMaterializeRefusesAPromotedRun:
    """`materialize` never read an existing manifest, so re-materializing under a promoted run
    id rewrote staging and wrote a fresh manifest saying `promoted: False` while canonical still
    held the earlier promotion's v2 — a false manifest, and the only way back was to promote and
    then roll back a second time (checklist N2, PR #199)."""

    def test_materializing_a_promoted_run_again_is_refused(self, store, capsys):
        _write_era5_year(store, "niamey")
        canonical = _canonical(store, "niamey")
        canonical.parent.mkdir(parents=True, exist_ok=True)
        canonical.write_bytes(b"the v1 artifact")
        _materialize(store, "niamey")
        op.main(["promote", "--root", store.as_posix(), "--run-id", "r1", "--zone-id", "niamey"])

        code = _materialize(store, "niamey")

        assert code == 2
        assert "Roll it back first" in capsys.readouterr().err
        manifest = json.loads(
            (store / op._MANIFEST.format(run_id="r1")).read_text(encoding="utf-8")
        )
        assert manifest["promoted"] is True
        assert manifest["promoted_zones"] == ["niamey"]
        assert (
            op.main(
                ["rollback", "--root", store.as_posix(), "--run-id", "r1", "--zone-id", "niamey"]
            )
            == 0
        )


class TestPromotionIsAllOrNothingAndKeepsWhatItReplaces:
    def test_the_previous_canonical_is_copied_aside_before_being_replaced(self, store):
        _write_era5_year(store, "niamey")
        canonical = _canonical(store, "niamey")
        canonical.parent.mkdir(parents=True, exist_ok=True)
        canonical.write_bytes(b"the v1 artifact")
        _materialize(store, "niamey")

        assert (
            op.main(
                ["promote", "--root", store.as_posix(), "--run-id", "r1", "--zone-id", "niamey"]
            )
            == 0
        )

        assert _kept(store, "niamey").read_bytes() == b"the v1 artifact"
        assert _schema_version(canonical) == op.NORMALS_SCHEMA_VERSION
        # Nothing is deleted: the staged artifact is still there to promote again.
        assert _staged(store, "niamey").exists()

    def test_a_zone_that_was_not_staged_blocks_the_whole_promotion(self, store):
        """Half-promoted is the state with no name — some canonical v2, some v1, and a
        rollback covering only the ones that moved."""
        _write_era5_year(store, "niamey")
        _write_era5_year(store, "montpellier")
        _materialize(store, "niamey")  # montpellier deliberately not staged
        canonical = _canonical(store, "niamey")
        canonical.parent.mkdir(parents=True, exist_ok=True)
        canonical.write_bytes(b"the v1 artifact")

        code = op.main(
            [
                "promote",
                "--root",
                store.as_posix(),
                "--run-id",
                "r1",
                "--zone-id",
                "niamey",
                "montpellier",
            ]
        )

        assert code == 2
        # niamey WAS staged and valid, and still must not have moved.
        assert canonical.read_bytes() == b"the v1 artifact"
        assert not _kept(store, "niamey").exists()

    def test_promoting_the_same_run_twice_is_refused(self, store, capsys):
        """Read from the MANIFEST, not from the rollback copy's existence.

        The guard used to be "a rollback copy exists", which also refused every promotion after
        a rollback — contradicting the documented claim that a rollback is reversible
        (Greptile P1, PR #199).
        """
        _write_era5_year(store, "niamey")
        _canonical(store, "niamey").parent.mkdir(parents=True, exist_ok=True)
        _canonical(store, "niamey").write_bytes(b"the v1 artifact")
        _materialize(store, "niamey")
        op.main(["promote", "--root", store.as_posix(), "--run-id", "r1", "--zone-id", "niamey"])

        code = op.main(
            ["promote", "--root", store.as_posix(), "--run-id", "r1", "--zone-id", "niamey"]
        )

        assert code == 2
        assert "already promoted" in capsys.readouterr().err
        assert _kept(store, "niamey").read_bytes() == b"the v1 artifact"

    def test_the_manifest_records_the_promotion(self, store):
        _write_era5_year(store, "niamey")
        _materialize(store, "niamey")
        op.main(["promote", "--root", store.as_posix(), "--run-id", "r1", "--zone-id", "niamey"])

        manifest = json.loads(
            (store / op._MANIFEST.format(run_id="r1")).read_text(encoding="utf-8")
        )
        assert manifest["promoted"] is True
        assert "promoted_at" in manifest


class TestRollbackRestoresWithoutDeleting:
    def test_it_puts_back_exactly_what_was_kept(self, store):
        _write_era5_year(store, "niamey")
        canonical = _canonical(store, "niamey")
        canonical.parent.mkdir(parents=True, exist_ok=True)
        canonical.write_bytes(b"the v1 artifact")
        _materialize(store, "niamey")
        op.main(["promote", "--root", store.as_posix(), "--run-id", "r1", "--zone-id", "niamey"])

        assert (
            op.main(
                ["rollback", "--root", store.as_posix(), "--run-id", "r1", "--zone-id", "niamey"]
            )
            == 0
        )

        assert canonical.read_bytes() == b"the v1 artifact"
        # Reversible: the v2 is still staged, so the rollback can itself be undone.
        assert _schema_version(_staged(store, "niamey")) == op.NORMALS_SCHEMA_VERSION

    def test_a_run_that_was_never_promoted_cannot_be_rolled_back(self, store, capsys):
        """Inventing a restore would publish an artifact nobody kept."""
        _write_era5_year(store, "niamey")
        _materialize(store, "niamey")

        code = op.main(
            ["rollback", "--root", store.as_posix(), "--run-id", "r1", "--zone-id", "niamey"]
        )

        assert code == 2
        assert "not marked promoted" in capsys.readouterr().err

    def test_a_rollback_can_itself_be_undone(self, store):
        """The claim the old guard contradicted: the staged v2 is untouched, so promote again."""
        _write_era5_year(store, "niamey")
        canonical = _canonical(store, "niamey")
        canonical.parent.mkdir(parents=True, exist_ok=True)
        canonical.write_bytes(b"the v1 artifact")
        _materialize(store, "niamey")
        promote = ["promote", "--root", store.as_posix(), "--run-id", "r1", "--zone-id", "niamey"]
        op.main(promote)
        op.main(["rollback", "--root", store.as_posix(), "--run-id", "r1", "--zone-id", "niamey"])

        assert op.main(promote) == 0

        assert _schema_version(canonical) == op.NORMALS_SCHEMA_VERSION
        # And the kept copy still describes the state before this run EVER promoted.
        assert _kept(store, "niamey").read_bytes() == b"the v1 artifact"


class TestRollbackIsRefusedWhenPartial:
    """Promotion is a run-level operation, so rollback has to undo the whole run too — a
    per-zone rollback leaves some canonical objects v2 and some v1, with a manifest that says
    neither (checklist N1, PR #199)."""

    def test_rolling_back_part_of_the_run_is_refused(self, store):
        _write_era5_year(store, "niamey")
        _write_era5_year(store, "montpellier")
        for zone_id in ("niamey", "montpellier"):
            canonical = _canonical(store, zone_id)
            canonical.parent.mkdir(parents=True, exist_ok=True)
            canonical.write_bytes(b"the v1 artifact")
        _materialize(store, "niamey", "montpellier")
        op.main(
            [
                "promote",
                "--root",
                store.as_posix(),
                "--run-id",
                "r1",
                "--zone-id",
                "niamey",
                "montpellier",
            ]
        )

        code = op.main(
            ["rollback", "--root", store.as_posix(), "--run-id", "r1", "--zone-id", "niamey"]
        )

        assert code == 2
        assert _schema_version(_canonical(store, "niamey")) == op.NORMALS_SCHEMA_VERSION
        assert _schema_version(_canonical(store, "montpellier")) == op.NORMALS_SCHEMA_VERSION
        manifest = json.loads(
            (store / op._MANIFEST.format(run_id="r1")).read_text(encoding="utf-8")
        )
        assert manifest["promoted"] is True

    def test_rolling_back_the_whole_run_restores_every_zone(self, store):
        _write_era5_year(store, "niamey")
        _write_era5_year(store, "montpellier")
        for zone_id in ("niamey", "montpellier"):
            canonical = _canonical(store, zone_id)
            canonical.parent.mkdir(parents=True, exist_ok=True)
            canonical.write_bytes(b"the v1 artifact")
        _materialize(store, "niamey", "montpellier")
        op.main(
            [
                "promote",
                "--root",
                store.as_posix(),
                "--run-id",
                "r1",
                "--zone-id",
                "niamey",
                "montpellier",
            ]
        )

        code = op.main(
            [
                "rollback",
                "--root",
                store.as_posix(),
                "--run-id",
                "r1",
                "--zone-id",
                "niamey",
                "montpellier",
            ]
        )

        assert code == 0
        assert _canonical(store, "niamey").read_bytes() == b"the v1 artifact"
        assert _canonical(store, "montpellier").read_bytes() == b"the v1 artifact"
        manifest = json.loads(
            (store / op._MANIFEST.format(run_id="r1")).read_text(encoding="utf-8")
        )
        assert manifest["promoted"] is False
        assert manifest["promoted_zones"] == []

    def test_a_zone_with_no_prior_canonical_is_skipped_once_its_canonical_is_removed_by_hand(
        self, store, capsys
    ):
        _write_era5_year(store, "niamey")
        _write_era5_year(store, "montpellier")
        canonical_niamey = _canonical(store, "niamey")
        canonical_niamey.parent.mkdir(parents=True, exist_ok=True)
        canonical_niamey.write_bytes(b"the v1 artifact")
        # montpellier deliberately has no canonical before this run promotes.
        _materialize(store, "niamey", "montpellier")
        op.main(
            [
                "promote",
                "--root",
                store.as_posix(),
                "--run-id",
                "r1",
                "--zone-id",
                "niamey",
                "montpellier",
            ]
        )
        rollback_both = [
            "rollback",
            "--root",
            store.as_posix(),
            "--run-id",
            "r1",
            "--zone-id",
            "niamey",
            "montpellier",
        ]

        code = op.main(rollback_both)

        assert code == 2
        assert "does not delete" in capsys.readouterr().err
        assert _schema_version(canonical_niamey) == op.NORMALS_SCHEMA_VERSION

        _canonical(store, "montpellier").unlink()
        code = op.main(rollback_both)

        assert code == 0
        assert canonical_niamey.read_bytes() == b"the v1 artifact"
        assert not _canonical(store, "montpellier").exists()


class TestAnObjectStoreRootIsRefusedByEveryStep:
    """`Path("s3://deposium")` collapses to `s3:/deposium`, so a `"://" in str(root)` guard
    never fired and the operator treated the URI as a RELATIVE LOCAL DIRECTORY — creating
    `s3:/deposium/...` under the working directory."""

    @pytest.mark.parametrize("uri", ["s3://deposium", "https://evil.example/x", "file:///tmp/x"])
    @pytest.mark.parametrize("step", ["materialize", "promote", "rollback"])
    def test_every_scheme_is_refused_before_anything_is_written(self, uri, step, capsys, tmp_path):
        argv = [step, "--root", uri, "--run-id", "r1", "--zone-id", "niamey"]
        if step == "materialize":
            argv += ["--era5-root", tmp_path.as_posix()]

        assert op.main(argv) == 2
        assert "local-filesystem only" in capsys.readouterr().err
        assert not list(Path.cwd().glob("s3:*")), "a scheme-shaped directory was created"


class TestPromotionRequiresTheRunItClaimsToPromote:
    """Promotion used to treat the manifest as optional and stamp it globally, so a staged
    parquet could be promoted with no provenance and a partial promotion could mark a whole run
    as promoted (Greptile P2, PR #199)."""

    def test_a_staged_parquet_without_a_manifest_is_not_promotable(self, store, capsys):
        _write_era5_year(store, "niamey")
        _materialize(store, "niamey")
        (store / op._MANIFEST.format(run_id="r1")).unlink()

        code = op.main(
            ["promote", "--root", store.as_posix(), "--run-id", "r1", "--zone-id", "niamey"]
        )

        assert code == 2
        assert "no manifest" in capsys.readouterr().err

    def test_the_zone_set_must_match_what_the_run_materialized(self, store, capsys):
        _write_era5_year(store, "niamey")
        _write_era5_year(store, "montpellier")
        _materialize(store, "niamey")

        code = op.main(
            [
                "promote",
                "--root",
                store.as_posix(),
                "--run-id",
                "r1",
                "--zone-id",
                "niamey",
                "montpellier",
            ]
        )

        assert code == 2
        assert "Promote the run the manifest describes" in capsys.readouterr().err

    def test_the_promoted_zones_are_recorded_by_name(self, store):
        _write_era5_year(store, "niamey")
        _materialize(store, "niamey")
        op.main(["promote", "--root", store.as_posix(), "--run-id", "r1", "--zone-id", "niamey"])

        manifest = json.loads(
            (store / op._MANIFEST.format(run_id="r1")).read_text(encoding="utf-8")
        )
        assert manifest["promoted_zones"] == ["niamey"]


class TestAnAbsentCanonicalIsAStateAndNotAGap:
    """With no canonical to copy aside, promotion wrote no rollback file at all — so the state
    "there was nothing here" was unrepresentable and a rollback could not restore it."""

    def test_the_absence_is_marked(self, store):
        _write_era5_year(store, "niamey")
        _materialize(store, "niamey")
        assert not _canonical(store, "niamey").exists()

        assert (
            op.main(
                ["promote", "--root", store.as_posix(), "--run-id", "r1", "--zone-id", "niamey"]
            )
            == 0
        )

        assert _kept(store, "niamey").with_suffix(".absent").exists()
        assert not _kept(store, "niamey").exists()

    def test_rolling_that_back_is_refused_rather_than_deleting(self, store, capsys):
        """Restoring "absent" means removing a file, and this operator does not delete."""
        _write_era5_year(store, "niamey")
        _materialize(store, "niamey")
        op.main(["promote", "--root", store.as_posix(), "--run-id", "r1", "--zone-id", "niamey"])

        code = op.main(
            ["rollback", "--root", store.as_posix(), "--run-id", "r1", "--zone-id", "niamey"]
        )

        assert code == 2
        assert "does not delete" in capsys.readouterr().err
        # The promoted artifact is left exactly as it was; nothing is half-undone.
        assert _schema_version(_canonical(store, "niamey")) == op.NORMALS_SCHEMA_VERSION


class TestIdentifiersCannotEscapeTheRoot:
    """`run_id` and `zone_id` are unrestricted CLI strings interpolated into paths."""

    # `""` and `-leading` are rejected by argparse before this guard is reached — it exits
    # rather than returning, so asserting on the guard's message there would be asserting on
    # argparse. The cases below are the ones that reach the path interpolation.
    @pytest.mark.parametrize("bad", ["../../etc", "..", "a/b", "with space", "x/../../y"])
    def test_an_unsafe_run_id_is_refused(self, bad, store, capsys):
        code = op.main(
            ["promote", "--root", store.as_posix(), "--run-id", bad, "--zone-id", "niamey"]
        )

        assert code == 2
        assert "not a safe identifier" in capsys.readouterr().err

    def test_an_unsafe_zone_id_is_refused(self, store, capsys):
        code = op.main(
            [
                "materialize",
                "--root",
                store.as_posix(),
                "--era5-root",
                store.as_posix(),
                "--run-id",
                "r1",
                "--zone-id",
                "../../../etc/passwd",
            ]
        )

        assert code == 2
        assert "not a safe identifier" in capsys.readouterr().err


class TestAReplacementIsNeverPartiallyWritten:
    def test_a_failed_copy_leaves_the_destination_untouched(self, store, monkeypatch):
        """`shutil.copyfile` writes in place: a failure partway leaves a truncated parquet where
        a valid one used to be. Temp-and-rename makes the destination old-or-new, never a
        prefix of new."""
        _write_era5_year(store, "niamey")
        canonical = _canonical(store, "niamey")
        canonical.parent.mkdir(parents=True, exist_ok=True)
        canonical.write_bytes(b"the v1 artifact")
        _materialize(store, "niamey")

        real_copy = op.shutil.copyfile
        calls = {"n": 0}

        def _fail_on_the_second(src, dst):
            calls["n"] += 1
            if calls["n"] == 2:  # the staged -> canonical copy, after the aside
                raise OSError("no space left on device")
            return real_copy(src, dst)

        monkeypatch.setattr(op.shutil, "copyfile", _fail_on_the_second)

        with pytest.raises(OSError):
            op.promote(
                op.build_parser().parse_args(
                    ["promote", "--root", store.as_posix(), "--run-id", "r1", "--zone-id", "niamey"]
                )
            )

        assert canonical.read_bytes() == b"the v1 artifact"
        assert not list(canonical.parent.glob(".*.partial")), "a temp file was left behind"


class TestTheUnwindMessageNamesWhatItCouldNotRestore:
    def test_a_zone_with_no_prior_canonical_is_named_rather_than_claimed_restored(
        self, store, monkeypatch, capsys
    ):
        """The unwind used to print "Canonical is as it was before this invocation" even when a
        zone that had no canonical before the run still held its staged v2 — because a zone with
        only an ".absent" marker has nothing to restore from (checklist N3, PR #199)."""
        _write_era5_year(store, "montpellier")
        _write_era5_year(store, "niamey")
        # montpellier has no canonical before this run promotes; niamey has a v1 one.
        canonical_niamey = _canonical(store, "niamey")
        canonical_niamey.parent.mkdir(parents=True, exist_ok=True)
        canonical_niamey.write_bytes(b"the v1 artifact")
        _materialize(store, "montpellier", "niamey")

        real_copy = op.shutil.copyfile
        calls = {"n": 0}

        def _fail_on_the_third(src, dst):
            calls["n"] += 1
            if calls["n"] == 3:  # the niamey staged -> canonical copy
                raise OSError("no space left on device")
            return real_copy(src, dst)

        monkeypatch.setattr(op.shutil, "copyfile", _fail_on_the_third)

        with pytest.raises(OSError):
            op.promote(
                op.build_parser().parse_args(
                    [
                        "promote",
                        "--root",
                        store.as_posix(),
                        "--run-id",
                        "r1",
                        "--zone-id",
                        "montpellier",
                        "niamey",
                    ]
                )
            )

        err = capsys.readouterr().err
        assert "Canonical is as it was" not in err
        assert _canonical(store, "montpellier").as_posix() in err
        assert canonical_niamey.read_bytes() == b"the v1 artifact"
