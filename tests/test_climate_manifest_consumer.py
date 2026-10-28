"""Network-free tests for scripts/climate_manifest_consumer.py.

Covers the zip-slip containment guard in ``import_zip`` — it must reject members that escape
the extraction root while letting ordinary members through on every platform — the per-kind
row expectations of ``check_manifest`` for era5/normals/ndvi/unified manifests, and the
destination-URI / manifest-rewrite half that ``export-zip`` and ``import-zip`` build paths from.

The manifest fixtures keep real ``s3://`` URIs so the URI validators stay exercised, and a stub
session maps those URIs onto local temp files, so nothing here touches an object store.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import zipfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import duckdb
import pytest

_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "climate_manifest_consumer.py"
_spec = importlib.util.spec_from_file_location("climate_manifest_consumer", _SCRIPT_PATH)
consumer = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(consumer)


def _bundle(tmp_path: Path, members: dict[str, bytes]) -> Path:
    dst = tmp_path / "bundle.zip"
    with zipfile.ZipFile(dst, "w") as zf:
        for name, payload in members.items():
            zf.writestr(name, payload)
    return dst


def _import(bundle: Path):
    return consumer.import_zip(bundle, bucket="deposium", run_id=None, manifest_uri=None)


def test_ordinary_members_pass_the_containment_guard(tmp_path):
    # A plain top-level member must never be read as an escape. The temp root is a symlink
    # on macOS (/var -> /private/var), so an unresolved root rejects everything; guard
    # against that regression by asserting the run fails *later*, on the missing bundle.json.
    bundle = _bundle(tmp_path, {"README.txt": b"hello"})

    with pytest.raises(Exception) as excinfo:  # noqa: PT011 - the type is the assertion
        _import(bundle)

    assert "escapes bundle root" not in str(excinfo.value)


def test_zip_slip_member_is_rejected(tmp_path):
    bundle = _bundle(tmp_path, {"../escaped.txt": b"owned"})

    with pytest.raises(ValueError, match="escapes bundle root"):
        _import(bundle)


def test_nested_zip_slip_member_is_rejected(tmp_path):
    bundle = _bundle(tmp_path, {"climate/era5/../../../escaped.parquet": b"owned"})

    with pytest.raises(ValueError, match="escapes bundle root"):
        _import(bundle)


def _uri(kind: str, zone_id: str) -> str:
    return f"s3://deposium/climate/{kind}/{zone_id}.parquet"


def _write_parquet(path: Path, rows: int) -> None:
    conn = duckdb.connect(":memory:")
    try:
        conn.execute(
            f"COPY (SELECT i FROM range({rows}) t(i)) TO '{path.as_posix()}' (FORMAT PARQUET)"
        )
    finally:
        conn.close()


def _install(
    monkeypatch,
    tmp_path: Path,
    manifest: dict[str, Any],
    rows: dict[tuple[str, str], int],
) -> str:
    """Materialize ``manifest`` and its artifacts locally, then route the consumer's reads there.

    ``rows`` gives the real row count of each ``(kind, zone_id)`` artifact, which is deliberately
    independent of whatever the manifest advertises.
    """
    local_by_uri: dict[str, str] = {}
    for output in manifest["outputs"]:
        parquet = tmp_path / f"{output['kind']}-{output['zone_id']}.parquet"
        _write_parquet(parquet, rows[output["kind"], output["zone_id"]])
        local_by_uri[output["uri"]] = parquet.as_posix()

    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    manifest_uri = "s3://deposium/climate/runs/test-run/manifest.json"
    local_by_uri[manifest_uri] = manifest_path.as_posix()

    @contextmanager
    def _session(*_args, **_kwargs):
        conn = duckdb.connect(":memory:")
        try:
            yield _LocalConn(conn, local_by_uri)
        finally:
            conn.close()

    monkeypatch.setattr(consumer, "duckdb_session", _session)
    return manifest_uri


class _LocalConn:
    """DuckDB connection that substitutes local paths for the fixture's ``s3://`` URIs."""

    def __init__(self, conn: duckdb.DuckDBPyConnection, local_by_uri: dict[str, str]) -> None:
        self._conn = conn
        self._local_by_uri = local_by_uri

    def execute(self, sql: str, parameters: list[Any] | None = None):
        if parameters is None:
            return self._conn.execute(sql)
        return self._conn.execute(sql, [self._local_by_uri.get(p, p) for p in parameters])


def _ndvi_manifest(
    zones: tuple[str, ...] = ("montpellier_metropole", "bassin_du_lez"),
    *,
    monthly: bool = True,
    start_date: str = "2024-01-01",
    end_date: str = "2024-12-31",
    advertised_rows: dict[str, int] | None = None,
) -> dict[str, Any]:
    advertised_rows = advertised_rows or {}
    return {
        "schema_version": 1,
        "kind": "geoai_run_manifest",
        "run_id": "test-ndvi-run",
        "parameters": {
            "zone_ids": list(zones),
            "start_date": start_date,
            "end_date": end_date,
            "cloud_cover_max": 20,
            "monthly": monthly,
        },
        "outputs": [
            {
                "kind": "ndvi",
                "zone_id": zone,
                "uri": _uri("ndvi", zone),
                **({"rows": advertised_rows[zone]} if zone in advertised_rows else {}),
            }
            for zone in zones
        ],
        "checks": {"ok": True, "n_zones_requested": len(zones), "n_zones_completed": len(zones)},
    }


def _unified_manifest(matched: dict[str, int]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "kind": "geoai_run_manifest",
        "run_id": "test-unified-run",
        "parameters": {
            "zone_ids": list(matched),
            "start_date": "2024-01-01",
            "end_date": "2024-12-31",
        },
        "outputs": [
            {
                "kind": "unified",
                "zone_id": zone,
                "uri": _uri("unified", zone),
                "n_matched": n_matched,
            }
            for zone, n_matched in matched.items()
        ],
        "checks": {"ok": True, "zone_ok": True},
    }


def _era5_manifest(
    zones: tuple[str, ...] = ("montpellier", "marseille"),
    *,
    variables: tuple[str, ...] = ("temperature_2m", "precipitation"),
) -> dict[str, Any]:
    outputs: list[dict[str, Any]] = []
    for zone in zones:
        outputs.append({"kind": "era5", "zone_id": zone, "uri": _uri("era5", zone)})
        outputs.append({"kind": "normals", "zone_id": zone, "uri": _uri("normals", zone)})
    return {
        "schema_version": 1,
        "kind": "geoai_run_manifest",
        "run_id": "test-era5-run",
        "parameters": {
            "zone_ids": list(zones),
            "start_date": "2024-01-01",
            "end_date": "2024-01-10",
            "variables": list(variables),
        },
        "outputs": outputs,
        "checks": {"ok": True, "zone_ok": True},
    }


def test_monthly_ndvi_manifest_validates_under_the_month_bound(tmp_path, monkeypatch):
    manifest = _ndvi_manifest()
    uri = _install(
        monkeypatch,
        tmp_path,
        manifest,
        {("ndvi", "montpellier_metropole"): 9, ("ndvi", "bassin_du_lez"): 12},
    )

    result = consumer.check_manifest(uri, expected_zones=None)

    assert result["ok"], result["errors"]
    assert result["n_zones"] == 2
    assert result["row_bounds"]["ndvi"] == "1 <= rows <= 12"
    assert [a["expected_rows"] for a in result["artifacts"]] == [None, None]
    assert result["expected_era5_rows"] is None
    assert result["expected_normals_rows"] is None


def test_monthly_ndvi_manifest_fails_above_the_month_bound(tmp_path, monkeypatch):
    uri = _install(
        monkeypatch,
        tmp_path,
        _ndvi_manifest(zones=("montpellier_metropole",)),
        {("ndvi", "montpellier_metropole"): 13},
    )

    result = consumer.check_manifest(uri, expected_zones=None)

    assert not result["ok"]
    assert result["errors"] == ["ndvi/montpellier_metropole has 13 rows, expected 1 <= rows <= 12"]


def test_empty_ndvi_artifact_fails(tmp_path, monkeypatch):
    uri = _install(
        monkeypatch,
        tmp_path,
        _ndvi_manifest(zones=("montpellier_metropole",)),
        {("ndvi", "montpellier_metropole"): 0},
    )

    result = consumer.check_manifest(uri, expected_zones=None)

    assert not result["ok"]
    assert "has 0 rows" in result["errors"][0]


def test_all_scenes_ndvi_accepts_more_rows_than_months(tmp_path, monkeypatch):
    uri = _install(
        monkeypatch,
        tmp_path,
        _ndvi_manifest(zones=("montpellier_metropole",), monthly=False),
        {("ndvi", "montpellier_metropole"): 400},
    )

    result = consumer.check_manifest(uri, expected_zones=None)

    assert result["ok"], result["errors"]
    assert result["row_bounds"]["ndvi"] == "rows >= 1"


def test_the_era5_pair_gate_names_only_the_zone_that_lacks_its_pair(tmp_path, monkeypatch):
    # A mixed manifest isolates the gate: `bassin_du_lez` carries NDVI only, which is not a
    # missing half of anything, while `montpellier` carries the full era5+normals pair and an
    # NDVI series. Only the era5-less zone may be named, and only because era5 is advertised
    # at all — the same manifest without era5 outputs must stay green.
    manifest = _era5_manifest(zones=("montpellier",))
    manifest["parameters"]["zone_ids"] = ["montpellier", "bassin_du_lez"]
    manifest["parameters"]["monthly"] = True
    manifest["outputs"] += [
        {"kind": "ndvi", "zone_id": "montpellier", "uri": _uri("ndvi", "montpellier")},
        {"kind": "ndvi", "zone_id": "bassin_du_lez", "uri": _uri("ndvi", "bassin_du_lez")},
    ]
    uri = _install(
        monkeypatch,
        tmp_path,
        manifest,
        {
            ("era5", "montpellier"): 10,
            ("normals", "montpellier"): 24,
            ("ndvi", "montpellier"): 1,
            ("ndvi", "bassin_du_lez"): 1,
        },
    )

    result = consumer.check_manifest(uri, expected_zones=None)

    assert result["errors"] == ["zones missing era5/normals pair: ['bassin_du_lez']"]


def test_ndvi_only_manifest_is_not_failed_for_a_missing_era5_pair(tmp_path, monkeypatch):
    uri = _install(
        monkeypatch,
        tmp_path,
        _ndvi_manifest(zones=("montpellier_metropole",)),
        {("ndvi", "montpellier_metropole"): 7},
    )

    result = consumer.check_manifest(uri, expected_zones=None)

    assert result["errors"] == []


def test_unified_manifest_validates_against_its_advertised_match_count(tmp_path, monkeypatch):
    uri = _install(
        monkeypatch,
        tmp_path,
        _unified_manifest({"fos_sur_mer": 88}),
        {("unified", "fos_sur_mer"): 88},
    )

    result = consumer.check_manifest(uri, expected_zones=None)

    assert result["ok"], result["errors"]
    assert result["artifacts"][0]["expected_rows"] == 88


def test_unified_manifest_fails_when_the_artifact_disagrees_with_n_matched(tmp_path, monkeypatch):
    uri = _install(
        monkeypatch,
        tmp_path,
        _unified_manifest({"fos_sur_mer": 88}),
        {("unified", "fos_sur_mer"): 87},
    )

    result = consumer.check_manifest(uri, expected_zones=None)

    assert not result["ok"]
    assert result["errors"] == ["unified/fos_sur_mer has 87 rows, expected 88"]


def test_era5_manifest_still_uses_exact_row_expectations(tmp_path, monkeypatch):
    uri = _install(
        monkeypatch,
        tmp_path,
        _era5_manifest(),
        {
            ("era5", "montpellier"): 10,
            ("normals", "montpellier"): 24,
            ("era5", "marseille"): 10,
            ("normals", "marseille"): 24,
        },
    )

    result = consumer.check_manifest(uri, expected_zones=None)

    assert result["ok"], result["errors"]
    assert result["expected_era5_rows"] == 10
    assert result["expected_normals_rows"] == 24
    assert {a["expected_rows"] for a in result["artifacts"]} == {10, 24}


def test_era5_manifest_still_fails_on_an_off_by_one_row_count(tmp_path, monkeypatch):
    uri = _install(
        monkeypatch,
        tmp_path,
        _era5_manifest(zones=("montpellier",)),
        {("era5", "montpellier"): 9, ("normals", "montpellier"): 24},
    )

    result = consumer.check_manifest(uri, expected_zones=None)

    assert not result["ok"]
    assert result["errors"] == ["era5/montpellier has 9 rows, expected 10"]


def test_era5_manifest_still_requires_the_normals_half_of_the_pair(tmp_path, monkeypatch):
    manifest = _era5_manifest(zones=("montpellier",))
    manifest["outputs"] = [o for o in manifest["outputs"] if o["kind"] == "era5"]
    uri = _install(monkeypatch, tmp_path, manifest, {("era5", "montpellier"): 10})

    result = consumer.check_manifest(uri, expected_zones=None)

    assert not result["ok"]
    assert "zones missing era5/normals pair: ['montpellier']" in result["errors"]
    # Narrowed contract, documented in the module docstring: an expectation is reported only for
    # a kind the manifest advertises, so a manifest with no normals output reports None here.
    assert result["expected_normals_rows"] is None
    assert "normals" not in result["row_bounds"]


def test_unknown_output_kind_is_still_rejected(tmp_path, monkeypatch):
    manifest = _ndvi_manifest(zones=("montpellier_metropole",))
    manifest["outputs"][0]["kind"] = "lidar"
    uri = _install(monkeypatch, tmp_path, manifest, {("lidar", "montpellier_metropole"): 3})

    with pytest.raises(ValueError, match="unsupported output kind"):
        consumer.check_manifest(uri, expected_zones=None)


def test_expected_zones_defaults_to_the_manifest_zone_ids(tmp_path, monkeypatch):
    uri = _install(
        monkeypatch,
        tmp_path,
        _ndvi_manifest(),
        {("ndvi", "montpellier_metropole"): 9, ("ndvi", "bassin_du_lez"): 9},
    )

    result = consumer.check_manifest(uri, expected_zones=None)

    assert result["expected_zones"] == 2
    assert result["expected_zones_source"] == "manifest.parameters.zone_ids"
    assert result["ok"], result["errors"]


def test_expected_zones_flag_overrides_the_manifest(tmp_path, monkeypatch):
    uri = _install(
        monkeypatch,
        tmp_path,
        _ndvi_manifest(),
        {("ndvi", "montpellier_metropole"): 9, ("ndvi", "bassin_du_lez"): 9},
    )

    result = consumer.check_manifest(uri, expected_zones=15)

    assert result["expected_zones"] == 15
    assert result["expected_zones_source"] == "cli"
    assert result["errors"] == ["found 2 zones, expected 15"]


# --------------------------------------------------------------------------- #
# Fail closed: an unusable input is refused, not reinterpreted
# --------------------------------------------------------------------------- #


def _inverted(manifest: dict[str, Any]) -> dict[str, Any]:
    manifest["parameters"]["start_date"] = "2024-12-31"
    manifest["parameters"]["end_date"] = "2024-01-01"
    return manifest


def test_an_inverted_window_is_refused_even_when_no_bound_would_have_read_it(tmp_path, monkeypatch):
    """The silent one. A unified bound is `rows >= 1` and never touches the window, so this
    manifest passed its check — `ok: true`, no errors — while advertising a window that runs
    backwards. Nothing downstream could tell."""
    manifest = _inverted(_ndvi_manifest(zones=("a",)))
    manifest["outputs"] = [{"kind": "unified", "zone_id": "a", "uri": _uri("unified", "a")}]
    uri = _install_manifest_only(monkeypatch, tmp_path, manifest)

    with pytest.raises(consumer.ManifestError, match="window is inverted"):
        consumer.check_manifest(uri, expected_zones=None)


def test_an_inverted_window_no_longer_yields_an_unsatisfiable_ndvi_bound(tmp_path, monkeypatch):
    """`month_ranges` walks `while cur <= end` and returns [] on an inverted window, so the
    ceiling was 0 and the bound was `1 <= rows <= 0` — unsatisfiable, and the error blamed the
    artifact for the parameters."""
    uri = _install_manifest_only(monkeypatch, tmp_path, _inverted(_ndvi_manifest(zones=("a",))))

    with pytest.raises(consumer.ManifestError, match="window is inverted"):
        consumer.check_manifest(uri, expected_zones=None)


def test_an_inverted_window_no_longer_demands_a_negative_era5_row_count(tmp_path, monkeypatch):
    """era5's bound is its day count, and `(start - end).days + 1` over an inverted window is
    -364. A real parquet file was being asked to hold -364 rows."""
    uri = _install_manifest_only(monkeypatch, tmp_path, _inverted(_era5_manifest(zones=("a",))))

    with pytest.raises(consumer.ManifestError, match="window is inverted"):
        consumer.check_manifest(uri, expected_zones=None)


def test_a_one_day_window_is_not_inverted(tmp_path, monkeypatch):
    """The boundary the refusal must not swallow: start == end is a legal one-day run."""
    manifest = _era5_manifest(zones=("a",))
    manifest["parameters"]["end_date"] = manifest["parameters"]["start_date"]
    uri = _install(monkeypatch, tmp_path, manifest, {("era5", "a"): 1, ("normals", "a"): 24})

    result = consumer.check_manifest(uri, expected_zones=None)

    assert result["ok"], result["errors"]
    assert result["expected_era5_rows"] == 1


@pytest.mark.parametrize("key", ["rows", "n_matched"])
@pytest.mark.parametrize("value", ["81", 81.0, True, False, None, -5, [81], {"n": 81}])
def test_an_unusable_row_claim_is_refused_rather_than_ignored(tmp_path, monkeypatch, key, value):
    """Every one of these used to mean "this entry makes no claim", which is a silent downgrade.

    `_advertised_rows` NARROWS the kind bound to the advertised count, so an entry claiming 81
    rows over a 3-row artifact is caught. Reading an unusable claim as absence throws that
    narrowing away and the truncation walks straight through. Measured before this change:
    `"81"`, `81.0`, `true` and `null` all reported `ok: true` against a 3-row parquet.

    `true` is the deliberate one. `isinstance(True, int)` holds in Python, so a JSON `true`
    arrives as the integer 1 and would pin the bound to "exactly one row".
    """
    manifest = _ndvi_manifest(zones=("a",))
    manifest["outputs"][0][key] = value
    uri = _install_manifest_only(monkeypatch, tmp_path, manifest)

    with pytest.raises(consumer.ManifestError, match=f"has a {key} that is not a non-negative"):
        consumer.check_manifest(uri, expected_zones=None)


def test_an_absent_row_claim_is_still_absence_not_a_refusal(tmp_path, monkeypatch):
    """The other side of the same line: a manifest that simply makes no claim is legitimate, and
    the kind bound alone decides. Only a claim that IS made and is wrong gets refused."""
    manifest = _ndvi_manifest(zones=("a",))
    assert "rows" not in manifest["outputs"][0] and "n_matched" not in manifest["outputs"][0]
    uri = _install(monkeypatch, tmp_path, manifest, {("ndvi", "a"): 3})

    result = consumer.check_manifest(uri, expected_zones=None)

    assert result["ok"], result["errors"]
    assert result["row_bounds"] == {"ndvi": "1 <= rows <= 12"}


def test_a_second_row_key_is_validated_even_when_the_first_one_answers(tmp_path, monkeypatch):
    """`rows` is read first and `n_matched` never returned — but a garbage value under it is
    still a manifest stating something untrue, and it is not left unread."""
    manifest = _ndvi_manifest(zones=("a",), advertised_rows={"a": 3})
    manifest["outputs"][0]["n_matched"] = "3"
    uri = _install_manifest_only(monkeypatch, tmp_path, manifest)

    with pytest.raises(consumer.ManifestError, match="has a n_matched that is not a non-negative"):
        consumer.check_manifest(uri, expected_zones=None)


def test_an_advertised_zero_is_still_a_claim_and_still_accepted(tmp_path, monkeypatch):
    """0 is a non-negative integer and it survives the type gate, so the conventional-floor
    exception below it keeps working: a run that legitimately matched nothing says so."""
    manifest = _ndvi_manifest(zones=("a",), advertised_rows={"a": 0})
    uri = _install(monkeypatch, tmp_path, manifest, {("ndvi", "a"): 0})

    result = consumer.check_manifest(uri, expected_zones=None)

    assert result["ok"], result["errors"]


def test_an_all_scenes_run_keeps_a_positive_floor_and_gains_no_ceiling(tmp_path, monkeypatch):
    """`monthly: false` drops the month CEILING — an all-scenes search over a year can return
    hundreds of observations — and must not drop the floor with it. An artifact holding zero rows
    under a manifest claiming a successful run is the under-claim the floor exists to reject."""
    empty = _ndvi_manifest(zones=("a",), monthly=False)
    uri = _install(monkeypatch, tmp_path, empty, {("ndvi", "a"): 0})

    result = consumer.check_manifest(uri, expected_zones=None)

    assert not result["ok"]
    assert result["errors"] == ["ndvi/a has 0 rows, expected rows >= 1"]
    assert result["row_bounds"] == {"ndvi": "rows >= 1"}


def test_an_all_scenes_run_accepts_far_more_rows_than_the_window_has_months(tmp_path, monkeypatch):
    """The ceiling really is gone: 500 observations over a 12-month window is a legal all-scenes
    result, and the monthly bound would have capped it at 12."""
    uri = _install(
        monkeypatch,
        tmp_path,
        _ndvi_manifest(zones=("a",), monthly=False),
        {("ndvi", "a"): 500},
    )

    result = consumer.check_manifest(uri, expected_zones=None)

    assert result["ok"], result["errors"]


# --------------------------------------------------------------------------- #
# Zone IDENTITY, not zone cardinality
# --------------------------------------------------------------------------- #


def _zone_manifest(declared: tuple[str, ...], materialized: tuple[str, ...]) -> dict[str, Any]:
    """A manifest whose declared zone list and whose outputs are set INDEPENDENTLY."""
    manifest = _ndvi_manifest(zones=declared)
    manifest["outputs"] = [
        {"kind": "ndvi", "zone_id": zone, "uri": _uri("ndvi", zone)} for zone in materialized
    ]
    return manifest


def test_an_output_for_a_zone_nobody_asked_for_is_refused(tmp_path, monkeypatch):
    """The substitution a count cannot see: `c` was never materialized and `d` was not requested.

    Three declared, three delivered, and they are not the same three. This reported `ok: true`.
    """
    manifest = _zone_manifest(declared=("a", "b", "c"), materialized=("a", "b", "d"))
    uri = _install(monkeypatch, tmp_path, manifest, {("ndvi", z): 5 for z in ("a", "b", "d")})

    result = consumer.check_manifest(uri, expected_zones=None)

    assert not result["ok"]
    # Both halves, both named. An operator must not have to diff two lists by hand.
    assert "zones in parameters.zone_ids with no output: ['c']" in result["errors"]
    assert "outputs for zones absent from parameters.zone_ids: ['d']" in result["errors"]


def test_the_same_artifact_advertised_twice_is_refused(tmp_path, monkeypatch):
    """Four outputs over three zones. `by_zone[zone][kind] = artifact` overwrites, so the first
    `a` artifact's verdict was silently dropped and the cardinality check saw nothing at all."""
    manifest = _zone_manifest(declared=("a", "b", "c"), materialized=("a", "a", "b", "c"))
    uri = _install(monkeypatch, tmp_path, manifest, {("ndvi", z): 5 for z in ("a", "b", "c")})

    result = consumer.check_manifest(uri, expected_zones=None)

    assert not result["ok"]
    assert "outputs advertise the same artifact more than once: ['ndvi/a']" in result["errors"]


def test_a_duplicate_that_also_breaks_the_count_is_named_not_just_counted(tmp_path, monkeypatch):
    """`{a, a, b}` against `{a, b, c}` did fail — on `found 2 zones, expected 3`, which names
    neither the duplicated `a` nor the missing `c`."""
    manifest = _zone_manifest(declared=("a", "b", "c"), materialized=("a", "a", "b"))
    uri = _install(monkeypatch, tmp_path, manifest, {("ndvi", z): 5 for z in ("a", "b")})

    result = consumer.check_manifest(uri, expected_zones=None)

    assert not result["ok"]
    assert result["errors"] == [
        "outputs advertise the same artifact more than once: ['ndvi/a']",
        "zones in parameters.zone_ids with no output: ['c']",
        "found 2 zones, expected 3",
    ]


def test_one_zone_carrying_two_KINDS_is_not_a_duplicate(tmp_path, monkeypatch):
    """The duplicate key is `(kind, zone_id)`. Every era5 zone legitimately carries a normals
    output beside its era5 one, and keying on the zone alone would fail every baseline run."""
    uri = _install(
        monkeypatch,
        tmp_path,
        _era5_manifest(zones=("montpellier",)),
        {("era5", "montpellier"): 10, ("normals", "montpellier"): 24},
    )

    result = consumer.check_manifest(uri, expected_zones=None)

    assert result["ok"], result["errors"]
    assert result["expected_zone_ids"] == ["montpellier"]


def test_a_manifest_that_declares_no_zone_list_still_gets_the_duplicate_check(
    tmp_path, monkeypatch
):
    """No declaration means no identity to compare against — but a repeated artifact is a defect
    the manifest states entirely by itself, so that half still binds."""
    manifest = _zone_manifest(declared=("a", "b"), materialized=("a", "a"))
    del manifest["parameters"]["zone_ids"]
    uri = _install(monkeypatch, tmp_path, manifest, {("ndvi", "a"): 5})

    result = consumer.check_manifest(uri, expected_zones=None)

    assert "outputs advertise the same artifact more than once: ['ndvi/a']" in result["errors"]
    assert result["expected_zone_ids"] is None
    assert result["expected_zones_source"] == "fallback"


@pytest.mark.parametrize(
    "zone_ids, match",
    [
        ("montpellier", "must be a non-empty list"),
        ([], "must be a non-empty list"),
        ({"a": 1}, "must be a non-empty list"),
        (["a", 2], "unusable entries"),
        (["a", "../etc/passwd"], "unusable entries"),
        (["a", "a", "b"], "names a zone twice"),
    ],
)
def test_an_unusable_zone_declaration_is_refused_rather_than_downgraded(
    tmp_path, monkeypatch, zone_ids, match
):
    """Every one of these used to fall through to the fallback count of 15 in silence.

    `zone_ids: "montpellier"` is the sharp one: a string is not a list, but `len` of it is 12, so
    the old `isinstance(list)` test sent it to the fallback and the run was checked against a
    number nobody chose.
    """
    manifest = _ndvi_manifest(zones=("a",))
    manifest["parameters"]["zone_ids"] = zone_ids
    uri = _install_manifest_only(monkeypatch, tmp_path, manifest)

    with pytest.raises(consumer.ManifestError, match=match):
        consumer.check_manifest(uri, expected_zones=None)


def test_main_reports_a_bad_manifest_on_one_line_without_a_traceback(tmp_path, monkeypatch, capsys):
    manifest = _ndvi_manifest(zones=("montpellier_metropole",))
    manifest["outputs"][0]["kind"] = "lidar"
    uri = _install(monkeypatch, tmp_path, manifest, {("lidar", "montpellier_metropole"): 3})
    monkeypatch.setattr(sys, "argv", ["climate_manifest_consumer", "check", "--manifest-uri", uri])

    exit_code = consumer.main()

    captured = capsys.readouterr()
    assert exit_code != 0
    assert captured.err.splitlines() == ["error: unsupported output kind: 'lidar'"]
    assert "Traceback" not in captured.err


def test_main_returns_zero_on_a_valid_ndvi_manifest(tmp_path, monkeypatch, capsys):
    uri = _install(
        monkeypatch,
        tmp_path,
        _ndvi_manifest(zones=("montpellier_metropole",)),
        {("ndvi", "montpellier_metropole"): 7},
    )
    monkeypatch.setattr(sys, "argv", ["climate_manifest_consumer", "check", "--manifest-uri", uri])

    exit_code = consumer.main()

    assert exit_code == 0
    assert capsys.readouterr().out.startswith("OK: 1 zones, 1 outputs")


# --------------------------------------------------------------------------- #
# Advertised row counts narrow the bound for every kind
# --------------------------------------------------------------------------- #


def test_ndvi_artifact_must_match_the_row_count_its_output_entry_advertises(tmp_path, monkeypatch):
    # The B3 producer writes `rows` on every NDVI output. A manifest advertising 81 observations
    # over a 3-row parquet is truncated, stale or another zone's file; the 1..months kind bound
    # alone would wave it through.
    uri = _install(
        monkeypatch,
        tmp_path,
        _ndvi_manifest(
            zones=("montpellier_metropole",),
            start_date="2016-11-01",
            end_date="2024-12-31",
            advertised_rows={"montpellier_metropole": 81},
        ),
        {("ndvi", "montpellier_metropole"): 3},
    )

    result = consumer.check_manifest(uri, expected_zones=None)

    assert not result["ok"]
    assert result["errors"] == ["ndvi/montpellier_metropole has 3 rows, expected 81"]


def test_ndvi_artifact_matching_its_advertised_row_count_passes(tmp_path, monkeypatch):
    uri = _install(
        monkeypatch,
        tmp_path,
        _ndvi_manifest(
            zones=("montpellier_metropole",),
            start_date="2016-11-01",
            end_date="2024-12-31",
            advertised_rows={"montpellier_metropole": 81},
        ),
        {("ndvi", "montpellier_metropole"): 81},
    )

    result = consumer.check_manifest(uri, expected_zones=None)

    assert result["ok"], result["errors"]
    assert result["artifacts"][0]["expected_rows"] == 81


def test_an_advertised_zero_is_checked_as_zero_not_as_an_impossible_bound(tmp_path, monkeypatch):
    # A unified run that matched nothing advertises n_matched: 0. Intersecting that with the
    # kind floor of 1 would build `1 <= rows <= 0`, which no row count can ever satisfy.
    uri = _install(
        monkeypatch,
        tmp_path,
        _unified_manifest({"fos_sur_mer": 0}),
        {("unified", "fos_sur_mer"): 0},
    )

    result = consumer.check_manifest(uri, expected_zones=None)

    assert result["ok"], result["errors"]
    assert result["artifacts"][0]["expected_rows"] == 0
    assert result["artifacts"][0]["row_bound"] == "0"


def test_an_advertised_zero_still_fails_a_non_empty_artifact_with_a_coherent_message(
    tmp_path, monkeypatch
):
    uri = _install(
        monkeypatch,
        tmp_path,
        _unified_manifest({"fos_sur_mer": 0}),
        {("unified", "fos_sur_mer"): 3},
    )

    result = consumer.check_manifest(uri, expected_zones=None)

    assert result["errors"] == ["unified/fos_sur_mer has 3 rows, expected 0"]


# --------------------------------------------------------------------------- #
# An advertised count NARROWS the kind bound; it never replaces it
# --------------------------------------------------------------------------- #


def _install_manifest_only(monkeypatch, tmp_path: Path, manifest: dict[str, Any]) -> str:
    """Route only the manifest URI to a local file; every artifact URI stays unreachable.

    Any ``read_parquet`` reached during the check therefore fails loudly instead of quietly
    succeeding, which is how these tests prove a self-contradiction is caught before the I/O.
    """
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    manifest_uri = "s3://deposium/climate/runs/test-run/manifest.json"
    local_by_uri = {manifest_uri: manifest_path.as_posix()}

    @contextmanager
    def _session(*_args, **_kwargs):
        conn = duckdb.connect(":memory:")
        try:
            yield _LocalConn(conn, local_by_uri)
        finally:
            conn.close()

    monkeypatch.setattr(consumer, "duckdb_session", _session)
    return manifest_uri


def test_an_advertised_count_above_the_window_ceiling_is_a_contradiction_not_a_wider_bound(
    tmp_path, monkeypatch
):
    # Twelve monthly slices can hold at most twelve observations. A manifest advertising 999 used
    # to REPLACE the ceiling with 999 and pass a 999-row artifact; the manifest disagrees with its
    # own window and that is what the error has to say.
    uri = _install_manifest_only(
        monkeypatch,
        tmp_path,
        _ndvi_manifest(
            zones=("montpellier_metropole",),
            start_date="2024-01-01",
            end_date="2024-12-31",
            advertised_rows={"montpellier_metropole": 999},
        ),
    )

    with pytest.raises(
        consumer.ManifestError,
        match=r"ndvi/montpellier_metropole advertises 999 rows, "
        r"but its own parameters allow 1 <= rows <= 12",
    ):
        consumer.check_manifest(uri, expected_zones=None)


def test_a_consistent_advertised_count_still_pins_the_bound_exactly(tmp_path, monkeypatch):
    # 5 is inside the 1..12 the window allows, so it stands — and it stands as an EXACT
    # expectation, not as a new floor: a 9-row artifact is inside the kind bound and still wrong.
    uri = _install(
        monkeypatch,
        tmp_path,
        _ndvi_manifest(
            zones=("montpellier_metropole",),
            start_date="2024-01-01",
            end_date="2024-12-31",
            advertised_rows={"montpellier_metropole": 5},
        ),
        {("ndvi", "montpellier_metropole"): 9},
    )

    result = consumer.check_manifest(uri, expected_zones=None)

    assert result["errors"] == ["ndvi/montpellier_metropole has 9 rows, expected 5"]
    assert result["artifacts"][0]["row_bound"] == "5"


def test_a_derived_floor_still_binds_because_era5_rows_are_not_a_convention(tmp_path, monkeypatch):
    # era5 is one row per day: the floor is the window's day count, derived, not a "we expect
    # data" convention. An entry advertising 5 over a ten-day window contradicts it exactly as
    # 999 contradicts the NDVI ceiling above.
    manifest = _era5_manifest(zones=("montpellier",))
    manifest["outputs"][0]["rows"] = 5
    uri = _install_manifest_only(monkeypatch, tmp_path, manifest)

    with pytest.raises(
        consumer.ManifestError,
        match=r"era5/montpellier advertises 5 rows, but its own parameters allow 10",
    ):
        consumer.check_manifest(uri, expected_zones=None)


def test_the_zero_exception_does_not_extend_to_a_derived_floor(tmp_path, monkeypatch):
    # The zero exception exists for a floor of exactly 1 — the "we expect data" convention. An
    # era5 entry claiming zero rows for a ten-day window is not a legitimately empty result.
    manifest = _era5_manifest(zones=("montpellier",))
    manifest["outputs"][0]["rows"] = 0
    uri = _install_manifest_only(monkeypatch, tmp_path, manifest)

    with pytest.raises(
        consumer.ManifestError,
        match=r"era5/montpellier advertises 0 rows, but its own parameters allow 10",
    ):
        consumer.check_manifest(uri, expected_zones=None)


def test_an_ndvi_entry_advertising_zero_is_accepted_like_a_unified_one(tmp_path, monkeypatch):
    # Sentinel-2 coverage is opportunistic; "no usable scene in the whole window" is a real
    # outcome, and the NDVI floor of 1 is the same convention the unified floor is.
    uri = _install(
        monkeypatch,
        tmp_path,
        _ndvi_manifest(
            zones=("montpellier_metropole",),
            start_date="2024-01-01",
            end_date="2024-12-31",
            advertised_rows={"montpellier_metropole": 0},
        ),
        {("ndvi", "montpellier_metropole"): 0},
    )

    result = consumer.check_manifest(uri, expected_zones=None)

    assert result["ok"], result["errors"]
    assert result["artifacts"][0]["row_bound"] == "0"


# --------------------------------------------------------------------------- #
# Window and `monthly` are read the way the producer writes them
# --------------------------------------------------------------------------- #


def test_a_unified_manifest_without_a_window_is_rejected(tmp_path, monkeypatch):
    manifest = _unified_manifest({"fos_sur_mer": 3})
    del manifest["parameters"]["start_date"]
    uri = _install(monkeypatch, tmp_path, manifest, {("unified", "fos_sur_mer"): 3})

    with pytest.raises(consumer.ManifestError, match="start_date/end_date are required"):
        consumer.check_manifest(uri, expected_zones=None)


def test_a_garbage_window_is_rejected_on_an_all_scenes_ndvi_manifest(tmp_path, monkeypatch):
    manifest = _ndvi_manifest(zones=("montpellier_metropole",), monthly=False)
    manifest["parameters"]["end_date"] = "2024-13-99"
    uri = _install(monkeypatch, tmp_path, manifest, {("ndvi", "montpellier_metropole"): 3})

    with pytest.raises(consumer.ManifestError, match="end_date is not an ISO date"):
        consumer.check_manifest(uri, expected_zones=None)


def test_an_omitted_monthly_flag_defaults_to_the_producers_default(tmp_path, monkeypatch):
    # `ndvi_ingest(..., monthly: bool = True)`: an omitted field means monthly, so the month
    # bound applies. Truthy-testing the field would hand this manifest `rows >= 1` instead.
    manifest = _ndvi_manifest(zones=("montpellier_metropole",))
    del manifest["parameters"]["monthly"]
    uri = _install(monkeypatch, tmp_path, manifest, {("ndvi", "montpellier_metropole"): 13})

    result = consumer.check_manifest(uri, expected_zones=None)

    assert result["row_bounds"]["ndvi"] == "1 <= rows <= 12"
    assert result["errors"] == ["ndvi/montpellier_metropole has 13 rows, expected 1 <= rows <= 12"]


def test_a_non_boolean_monthly_flag_is_rejected(tmp_path, monkeypatch):
    manifest = _ndvi_manifest(zones=("montpellier_metropole",))
    manifest["parameters"]["monthly"] = "yes"
    uri = _install(monkeypatch, tmp_path, manifest, {("ndvi", "montpellier_metropole"): 3})

    with pytest.raises(consumer.ManifestError, match="monthly must be a boolean"):
        consumer.check_manifest(uri, expected_zones=None)


# --------------------------------------------------------------------------- #
# zone_id confinement
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("kind", ["era5", "normals", "ndvi", "unified"])
@pytest.mark.parametrize(
    "zone_id",
    [
        "../../../../tmp/pwned",
        # Dot-only names pass the charset regex while naming no zone. They are not a traversal
        # (the value lands as one path segment), but they used to be rejected only later, by
        # _copy_parquet's blanket '..' test, under a message that blamed the S3 URI.
        ".",
        "..",
        "...",
    ],
)
def test_a_traversing_or_dot_only_zone_id_is_rejected_before_it_reaches_a_path(kind, zone_id):
    # export_zip builds `climate/<kind>/<zone_id>.parquet` from this value and the local branch
    # of _copy_parquet never reaches an S3 validator, so `_outputs` is the only gate — and it
    # must hold for every kind, not just the ones that happen to go through _destination_uri.
    manifest = {
        "outputs": [
            {
                "kind": kind,
                "zone_id": zone_id,
                "uri": _uri(kind, "montpellier_metropole"),
            }
        ]
    }

    with pytest.raises(consumer.ManifestError, match="unsafe zone_id"):
        consumer._outputs(manifest)


@pytest.mark.parametrize("zone_id", [".", "..", "..."])
def test_a_dot_only_zone_id_is_named_as_such_rather_than_as_a_bad_s3_uri(zone_id):
    # The point of the gate is the message: "unsafe zone_id" sends an operator to the manifest
    # field that is wrong; "unsafe S3 URI" sends them to the bucket layout, which is fine.
    with pytest.raises(consumer.ManifestError, match="unsafe zone_id"):
        consumer._destination_uri("staging", "ndvi", zone_id)


@pytest.mark.parametrize("run_id", [".", "..", "..."])
def test_a_dot_only_run_id_is_named_as_such_too(run_id):
    with pytest.raises(consumer.ManifestError, match="unsafe run_id"):
        consumer._manifest_destination_uri("staging", run_id)


@pytest.mark.parametrize("zone_id", ["bassin_du_lez", "zone.2024", ".hidden", "a..b", "z-1_2"])
def test_a_name_that_merely_contains_dots_is_still_a_name(zone_id):
    # The gate rejects names made of nothing but dots, not names that use one.
    assert consumer._destination_uri("staging", "ndvi", zone_id).endswith(f"/{zone_id}.parquet")


def test_a_plain_zone_id_still_passes_the_outputs_gate():
    manifest = {
        "outputs": [
            {"kind": "ndvi", "zone_id": "bassin_du_lez", "uri": _uri("ndvi", "bassin_du_lez")}
        ]
    }

    assert consumer._outputs(manifest) == manifest["outputs"]


# --------------------------------------------------------------------------- #
# Destination URIs and manifest rewriting for the new kinds
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("kind", "expected"),
    [
        ("era5", "s3://staging/climate/era5/bassin_du_lez.parquet"),
        ("normals", "s3://staging/climate/normals/bassin_du_lez.parquet"),
        ("ndvi", "s3://staging/climate/ndvi/bassin_du_lez.parquet"),
        ("unified", "s3://staging/climate/unified/bassin_du_lez.parquet"),
    ],
)
def test_destination_uri_covers_every_supported_kind(kind, expected):
    assert consumer._destination_uri("staging", kind, "bassin_du_lez") == expected


def test_destination_uri_still_rejects_an_unsupported_kind_and_an_unsafe_zone():
    with pytest.raises(consumer.ManifestError, match="unsupported artifact kind"):
        consumer._destination_uri("staging", "lidar", "bassin_du_lez")
    with pytest.raises(consumer.ManifestError, match="unsafe zone_id"):
        consumer._destination_uri("staging", "ndvi", "../etc/passwd")


def _mixed_manifest() -> dict[str, Any]:
    return {
        "kind": "geoai_run_manifest",
        "run_id": "source-run",
        "outputs": [
            {
                "kind": "ndvi",
                "zone_id": "bassin_du_lez",
                "uri": "s3://source/climate/ndvi/bassin_du_lez.parquet",
                "rows": 81,
            },
            {
                "kind": "unified",
                "zone_id": "fos_sur_mer",
                "uri": "s3://source/climate/unified/fos_sur_mer.parquet",
                "n_matched": 88,
            },
        ],
        "planned_outputs": [
            {
                "kind": "ndvi",
                "zone_id": "montpellier_metropole",
                "uri": "s3://source/climate/ndvi/montpellier_metropole.parquet",
            }
        ],
    }


def test_rewrite_manifest_uris_repoints_ndvi_and_unified_outputs():
    rewritten = consumer._rewrite_manifest_uris(
        _mixed_manifest(), bucket="staging", run_id="imported-run"
    )

    assert rewritten["run_id"] == "imported-run"
    assert [o["uri"] for o in rewritten["outputs"]] == [
        "s3://staging/climate/ndvi/bassin_du_lez.parquet",
        "s3://staging/climate/unified/fos_sur_mer.parquet",
    ]
    assert rewritten["planned_outputs"][0]["uri"] == (
        "s3://staging/climate/ndvi/montpellier_metropole.parquet"
    )
    # the advertised counts travel with the entry: an imported manifest keeps its own claims
    assert rewritten["outputs"][0]["rows"] == 81
    assert rewritten["outputs"][1]["n_matched"] == 88


def test_rewrite_manifest_uris_repoints_the_environment_block_at_the_destination(monkeypatch):
    monkeypatch.setattr(consumer.settings, "S3_ENDPOINT", "http://127.0.0.1:9100")
    monkeypatch.setattr(consumer.settings, "S3_FORCE_PATH_STYLE", True)
    manifest = _mixed_manifest()
    manifest["environment"] = {
        "s3_endpoint": "http://127.0.0.1:9000",
        "s3_bucket": "deposium",
        "s3_force_path_style": True,
        "unknown_future_key": "kept",
    }

    rewritten = consumer._rewrite_manifest_uris(manifest, bucket="staging", run_id="imported-run")

    assert rewritten["environment"] == {
        "s3_endpoint": "http://127.0.0.1:9100",
        "s3_endpoint_credentials_removed": False,
        "s3_bucket": "staging",
        "s3_force_path_style": True,
        "unknown_future_key": "kept",
    }
    assert rewritten["source_environment"] == {
        "s3_endpoint": "http://127.0.0.1:9000",
        "s3_endpoint_credentials_removed": False,
        "s3_bucket": "deposium",
        "s3_force_path_style": True,
        "unknown_future_key": "kept",
    }
    assert rewritten["source_environment_hops"] == 1


def test_a_credential_named_environment_key_is_not_republished(monkeypatch):
    """Unknown keys are carried through on purpose; a credential under one is not.

    The bundle is written by somebody else and `import-zip` republishes its `environment` block
    under `source_environment` in the manifest it writes to the DESTINATION bucket — the object
    being handed to a third party. `redact_credentials` cannot cover this: the origin's key is
    not one of this process's own secrets, and `s3_secret_key` is a structural field rather than
    one of `DIAGNOSTIC_FIELDS`. Measured before the filter existed, the pair was copied verbatim
    into BOTH blocks.
    """
    monkeypatch.setattr(consumer.settings, "S3_ENDPOINT", "http://mine:9000")
    monkeypatch.setattr(consumer.settings, "S3_FORCE_PATH_STYLE", True)
    manifest = _mixed_manifest()
    manifest["environment"] = {
        "s3_endpoint": "http://far.away:9000",
        "s3_bucket": "origin-bucket",
        "s3_secret_key": "origin-s3cret",
        "aws_session_token": "origin-token",
        "nested": {"catalogue_api_key": "origin-catalogue-token"},
        "unknown_future_key": "kept",
    }

    rewritten = consumer._rewrite_manifest_uris(manifest, bucket="dest", run_id="imported-run")

    published = json.dumps(rewritten)
    assert "origin-s3cret" not in published
    assert "origin-token" not in published
    assert "origin-catalogue-token" not in published
    # The KEY survives with a redacted value, so a reader can see the origin shipped one, and the
    # forward-compatible unknown key is still carried through untouched.
    for block in (rewritten["environment"], rewritten["source_environment"]):
        assert block["s3_secret_key"] == consumer.REDACTED
        assert block["aws_session_token"] == consumer.REDACTED
        assert block["nested"]["catalogue_api_key"] == consumer.REDACTED
        assert block["unknown_future_key"] == "kept"
        assert block["s3_bucket"] in {"dest", "origin-bucket"}


def test_rewrite_manifest_uris_is_a_no_op_when_no_environment_block_is_present():
    rewritten = consumer._rewrite_manifest_uris(
        _mixed_manifest(), bucket="staging", run_id="imported-run"
    )

    assert "environment" not in rewritten
    assert "source_environment" not in rewritten


# --------------------------------------------------------------------------- #
# The published manifest never carries the store password
# --------------------------------------------------------------------------- #

_IMPORT_SECRET = "deposium-dev-secret"
_IMPORT_ACCOUNT = "deposium-dev"


def _published_json(manifest: dict, *, bucket: str = "staging") -> str:
    """The manifest exactly as ``import_zip`` writes it to the destination bucket.

    ``import_zip`` serialises the rewritten manifest with ``_write_json`` and COPYs that file to
    ``s3://{bucket}/climate/runs/{run_id}/manifest.json``. Asserting on the serialised bytes
    rather than on the dict is the point: a credential can hide in a nested or unknown key that a
    key-by-key assertion never looks at.
    """
    rewritten = consumer._rewrite_manifest_uris(manifest, bucket=bucket, run_id="imported-run")
    return json.dumps(rewritten, indent=2, sort_keys=True, default=consumer._json_default)


def test_the_configured_endpoint_password_never_reaches_the_published_manifest(monkeypatch):
    # The manifest lands in the DESTINATION bucket — the one being handed to another party.
    monkeypatch.setattr(
        consumer.settings,
        "S3_ENDPOINT",
        f"http://{_IMPORT_ACCOUNT}:{_IMPORT_SECRET}@127.0.0.1:9000",
    )
    monkeypatch.setattr(consumer.settings, "S3_FORCE_PATH_STYLE", True)
    manifest = _mixed_manifest()
    manifest["environment"] = {"s3_endpoint": "http://127.0.0.1:9100", "s3_bucket": "deposium"}

    published = _published_json(manifest)

    assert _IMPORT_SECRET not in published
    assert _IMPORT_ACCOUNT not in published
    # REFUSED WHOLE rather than stripped down to `http://127.0.0.1:9000`. A value carrying
    # userinfo does not match the accept grammar, and the manifest records "no usable address"
    # plus the flag saying one was configured. Recording the host that a recovering parser
    # believed it had found is exactly the guess that leaked a password prefix four times.
    assert json.loads(published)["environment"] == {
        "s3_endpoint": None,
        "s3_endpoint_credentials_removed": True,
        "s3_bucket": "staging",
        "s3_force_path_style": True,
    }


def test_a_credential_named_environment_key_never_reaches_the_published_bytes(monkeypatch):
    """The same guarantee asserted on the JSON that is written, not on the dict.

    `_rewrite_manifest_uris` is only half the path; a credential hides in whatever key no
    assertion happens to read, so this checks the serialized document `import-zip` publishes to
    `s3://{destination}/climate/runs/{run_id}/manifest.json`. A LIST member is covered too — an
    `environment` is free to spell its stores as a list of blocks.
    """
    monkeypatch.setattr(consumer.settings, "S3_ENDPOINT", "http://127.0.0.1:9000")
    monkeypatch.setattr(consumer.settings, "S3_FORCE_PATH_STYLE", True)
    manifest = _mixed_manifest()
    manifest["environment"] = {
        "s3_endpoint": "http://far.away:9000",
        "s3_bucket": "theirs",
        "stac_password": "origin-catalogue-pw",
        "stores": [{"name": "cold", "private_key": "origin-private-key"}],
    }

    published = _published_json(manifest)

    assert "origin-catalogue-pw" not in published
    assert "origin-private-key" not in published
    parsed = json.loads(published)
    for block in (parsed["environment"], parsed["source_environment"]):
        assert block["stac_password"] == consumer.REDACTED
        assert block["stores"] == [{"name": "cold", "private_key": consumer.REDACTED}]
    # Nothing else moved: the origin bucket is still named under source_environment.
    assert parsed["source_environment"]["s3_bucket"] == "theirs"


def test_an_incoming_environment_is_sanitized_too_rather_than_republished_verbatim(monkeypatch):
    # A bundle written elsewhere may already carry credentials; it is not trusted to be clean.
    monkeypatch.setattr(consumer.settings, "S3_ENDPOINT", "http://127.0.0.1:9000")
    monkeypatch.setattr(consumer.settings, "S3_FORCE_PATH_STYLE", True)
    manifest = _mixed_manifest()
    manifest["environment"] = {
        "s3_endpoint": f"http://{_IMPORT_ACCOUNT}:{_IMPORT_SECRET}@far.away:9000",
        "s3_endpoint_credentials_removed": False,
        "s3_bucket": "theirs",
    }

    published = _published_json(manifest)

    assert _IMPORT_SECRET not in published
    assert _IMPORT_ACCOUNT not in published
    assert json.loads(published)["source_environment"] == {
        "s3_endpoint": None,
        "s3_endpoint_credentials_removed": True,
        "s3_bucket": "theirs",
    }


def test_the_ORIGINS_password_does_not_ride_an_inherited_error_into_the_destination_bucket(
    monkeypatch,
):
    """The origin's credentials are not this process's credentials — and they are recoverable.

    `redact_credentials` was only ever handed `settings.S3_ENDPOINT`, whose account and secret
    are a different pair entirely, so an incoming `errors[].error` quoting the ORIGIN store went
    into the destination bucket verbatim. Measured before this change: `"origin-s3cret" in
    published` was True.

    The bundle names its own store in the `environment` block it shipped, so the credential does
    not have to be guessed — it is read out of the document being imported.
    """
    monkeypatch.setattr(consumer.settings, "S3_ENDPOINT", "http://mine:9000")
    monkeypatch.setattr(consumer.settings, "S3_ACCESS_KEY", "my-key")
    monkeypatch.setattr(consumer.settings, "S3_SECRET_KEY", "my-secret")
    manifest = _mixed_manifest()
    manifest["environment"] = {
        "s3_endpoint": "http://origin-admin:origin-s3cret@far.away:9000",
        "s3_bucket": "theirs",
    }
    manifest["errors"] = [
        {
            "zone_id": "z1",
            "error": (
                "IOException: IO Error: HTTP HEAD to "
                "'http://origin-admin:origin-s3cret@far.away:9000/theirs/climate/ndvi/z1.parquet'"
            ),
        }
    ]

    published = _published_json(manifest)

    assert "origin-s3cret" not in published
    assert "origin-admin" not in published
    assert "[REDACTED]" in published
    # The origin's ADDRESS is not a secret and must survive, or the relayed manifest loses the
    # provenance `source_environment` exists to carry.
    assert "far.away:9000" in published


def test_an_imported_stac_input_is_not_republished_with_its_credential(monkeypatch):
    """`inputs[]` had no sanitizer at all on import.

    An imported manifest declaring `inputs[0].endpoint =
    "https://stac-reader:stac-s3cret@stac.origin/v1"` republished that string, credential and
    all, into the destination bucket. `redact_credentials` cannot help here twice over: it does
    not touch structural fields by design, and an origin's catalogue token is not among this
    process's configured secrets.
    """
    monkeypatch.setattr(consumer.settings, "S3_ENDPOINT", "http://mine:9000")
    manifest = _mixed_manifest()
    manifest["inputs"] = [
        {
            "kind": "stac_sentinel2_l2a",
            "endpoint": "https://stac-reader:stac-s3cret@stac.origin/v1",
        }
    ]

    published = _published_json(manifest)

    assert "stac-s3cret" not in published
    assert "stac-reader" not in published
    stac_input = json.loads(published)["inputs"][0]
    assert stac_input["endpoint"] == "https://stac.origin/v1"
    assert stac_input["endpoint_reductions"] == ["userinfo"]


def test_every_declared_input_address_key_is_sanitized_not_just_the_scalar_ones(monkeypatch):
    """All three of `_INPUT_ADDRESS_KEYS`, in ONE manifest, because two of them drifted apart.

    `_incoming_addresses` reads `endpoint`, `source` AND `uris`; `_sanitize_incoming_inputs` kept
    a second, hand-written `("endpoint", "source")`. Measured before this test existed: an
    imported manifest's `inputs[].uris` republished its userinfo VERBATIM into the destination
    bucket while its two siblings had theirs removed. `uris` is a LIST, which is why it was the
    one that got left behind.
    """
    monkeypatch.setattr(consumer.settings, "S3_ENDPOINT", "http://mine:9000")
    manifest = _mixed_manifest()
    manifest["inputs"] = [
        {
            "kind": "era5_parquet",
            "uris": [
                "s3://era5-reader:era5-s3cret@far.away:9000/theirs/climate/era5/z1.parquet",
                "s3://theirs/climate/era5/z2.parquet",
            ],
        },
        {
            "kind": "stac_sentinel2_l2a",
            "endpoint": "https://stac-reader:stac-s3cret@stac.origin/v1",
        },
        {"kind": "sentinel2_stac_ndvi", "source": "https://src-reader:src-s3cret@src.origin/v1"},
    ]

    published = _published_json(manifest)

    for secret in ("era5-s3cret", "era5-reader", "stac-s3cret", "stac-reader", "src-s3cret"):
        assert secret not in published, secret
    inputs = json.loads(published)["inputs"]
    # The credential-bearing `s3://` URI keeps its host, bucket and key: `sanitize_stac_url` is an
    # http(s) policy and refuses that scheme whole, so publishing its `None` would have replaced
    # the artifact's location with nothing at all.
    assert inputs[0]["uris"][0] == (
        f"s3://{consumer.REDACTED}:{consumer.REDACTED}@far.away:9000/theirs/climate/era5/z1.parquet"
    )
    # …and the sibling entry that carried no credential is republished byte for byte.
    assert inputs[0]["uris"][1] == "s3://theirs/climate/era5/z2.parquet"
    assert inputs[0]["uris_reductions"] == ["userinfo"]
    assert inputs[1]["endpoint"] == "https://stac.origin/v1"
    assert inputs[1]["endpoint_reductions"] == ["userinfo"]
    assert inputs[2]["source"] == "https://src.origin/v1"
    assert inputs[2]["source_reductions"] == ["userinfo"]


def test_the_input_sanitizer_and_the_address_reader_cannot_drift_again():
    """The guard on the bug's CAUSE, not just on its symptom.

    The defect was two hand-written key tuples that disagreed. Nothing stops a fourth key being
    added to `_INPUT_ADDRESS_KEYS` and the sanitizer being forgotten again except the sanitizer
    reading that same tuple, so assert it does: every declared key must actually be rewritten.
    """
    manifest = {
        "inputs": [
            {key: f"https://reader:tok3n-{key}@origin/v1" for key in consumer._INPUT_ADDRESS_KEYS}
        ]
    }

    consumer._sanitize_incoming_inputs(manifest)

    item = manifest["inputs"][0]
    for key in consumer._INPUT_ADDRESS_KEYS:
        assert f"tok3n-{key}" not in json.dumps(item), key
        assert item[f"{key}_reductions"] == ["userinfo"], key


def test_an_imported_input_without_a_credential_is_republished_verbatim(monkeypatch):
    """The other side of the line, and it is the important one: a sanitizer that rewrites
    addresses it had no reason to touch destroys the provenance it was added to protect.

    `uris` is a list of era5 parquet locations, `source` here is a catalogue root with no
    credential, and neither is a URL this import has any business editing.
    """
    monkeypatch.setattr(consumer.settings, "S3_ENDPOINT", "http://mine:9000")
    manifest = _mixed_manifest()
    manifest["inputs"] = [
        {"kind": "era5_parquet", "uris": ["s3://theirs/climate/era5/z1.parquet"]},
        {"kind": "sentinel2_stac_ndvi", "source": "https://earth-search.aws.element84.com/v1"},
    ]

    inputs = json.loads(_published_json(manifest))["inputs"]

    assert inputs[0] == {"kind": "era5_parquet", "uris": ["s3://theirs/climate/era5/z1.parquet"]}
    assert inputs[1] == {
        "kind": "sentinel2_stac_ndvi",
        "source": "https://earth-search.aws.element84.com/v1",
    }


def test_a_credential_in_an_imported_stac_input_is_also_scrubbed_from_the_prose(monkeypatch):
    """Both halves of the same secret. The structural field is sanitized and the DIAGNOSTIC field
    that quotes the same catalogue token is redacted — `inputs[].endpoint` is one of the places
    `_incoming_addresses` looks precisely so the error text can be reached too."""
    monkeypatch.setattr(consumer.settings, "S3_ENDPOINT", "http://mine:9000")
    manifest = _mixed_manifest()
    manifest["inputs"] = [
        {"kind": "stac_sentinel2_l2a", "endpoint": "https://reader:cat-tok3n@stac.origin/v1"}
    ]
    manifest["errors"] = [
        {"zone_id": "z1", "error": "401 from https://reader:cat-tok3n@stac.origin/v1/search"}
    ]

    published = _published_json(manifest)

    assert "cat-tok3n" not in published
    assert "[REDACTED]" in published
    assert "stac.origin" in published


def test_an_upstream_removal_flag_is_not_downgraded_by_a_clean_endpoint(monkeypatch):
    monkeypatch.setattr(consumer.settings, "S3_ENDPOINT", "http://127.0.0.1:9000")
    monkeypatch.setattr(consumer.settings, "S3_FORCE_PATH_STYLE", True)
    manifest = _mixed_manifest()
    manifest["environment"] = {
        "s3_endpoint": "http://far.away:9000",
        "s3_endpoint_credentials_removed": True,
    }

    rewritten = consumer._rewrite_manifest_uris(manifest, bucket="staging", run_id="imported-run")

    # The producer already removed credentials; that fact must survive the relay.
    assert rewritten["source_environment"]["s3_endpoint_credentials_removed"] is True


def test_a_non_object_environment_is_refused_rather_than_republished_unchecked():
    manifest = _mixed_manifest()
    manifest["environment"] = "http://deposium-dev:deposium-dev-secret@127.0.0.1:9000"

    with pytest.raises(consumer.ManifestError, match="environment must be a JSON object"):
        consumer._rewrite_manifest_uris(manifest, bucket="staging", run_id="imported-run")


# --------------------------------------------------------------------------- #
# Provenance survives a second import
# --------------------------------------------------------------------------- #


def test_a_re_import_keeps_the_origin_environment_instead_of_clobbering_it(monkeypatch):
    monkeypatch.setattr(consumer.settings, "S3_ENDPOINT", "http://second.host:9000")
    monkeypatch.setattr(consumer.settings, "S3_FORCE_PATH_STYLE", True)
    manifest = _mixed_manifest()
    # What an export of an already-imported manifest looks like: environment names the relay,
    # source_environment names the origin.
    manifest["environment"] = {"s3_endpoint": "http://first.host:9000", "s3_bucket": "relay"}
    manifest["source_environment"] = {"s3_endpoint": "http://origin.host:9000", "s3_bucket": "orig"}
    manifest["source_environment_hops"] = 1

    rewritten = consumer._rewrite_manifest_uris(manifest, bucket="staging", run_id="imported-run")

    assert rewritten["source_environment"]["s3_endpoint"] == "http://origin.host:9000"
    assert rewritten["source_environment"]["s3_bucket"] == "orig"
    assert rewritten["source_environment_hops"] == 2
    assert rewritten["environment"]["s3_endpoint"] == "http://second.host:9000"


def test_a_re_import_re_sanitizes_the_origin_it_inherited(monkeypatch):
    monkeypatch.setattr(consumer.settings, "S3_ENDPOINT", "http://second.host:9000")
    monkeypatch.setattr(consumer.settings, "S3_FORCE_PATH_STYLE", True)
    manifest = _mixed_manifest()
    manifest["environment"] = {"s3_endpoint": "http://first.host:9000"}
    manifest["source_environment"] = {
        "s3_endpoint": f"http://{_IMPORT_ACCOUNT}:{_IMPORT_SECRET}@origin.host:9000"
    }

    published = _published_json(manifest)

    assert _IMPORT_SECRET not in published
    assert json.loads(published)["source_environment"]["s3_endpoint"] is None


# --------------------------------------------------------------------------- #
# main() reports input problems, not bugs
# --------------------------------------------------------------------------- #


def test_main_lets_a_programming_error_surface_instead_of_reporting_it_as_bad_input(monkeypatch):
    def _boom(_args):
        raise ValueError("off-by-one in a helper")

    monkeypatch.setattr(consumer, "_dispatch", _boom)
    monkeypatch.setattr(
        sys, "argv", ["climate_manifest_consumer", "check", "--manifest-uri", "s3://x/y.json"]
    )

    with pytest.raises(ValueError, match="off-by-one in a helper"):
        consumer.main()


# --------------------------------------------------------------------------- #
# The shared month_ranges home stays dependency-free
# --------------------------------------------------------------------------- #


def test_the_shared_month_ranges_helper_pulls_in_no_heavy_dependency():
    """``month_ranges`` lives in a home an ops script can import for date arithmetic alone.

    Loaded straight from its path in a subprocess, so neither this session's already-imported
    modules nor the ``src.utils`` package ``__init__`` (which eagerly imports the raster + STAC
    stack) answers for it: what shows up here is what the module itself needs.
    """
    module = Path(__file__).resolve().parents[1] / "src" / "utils" / "dates.py"
    probe = (
        "import importlib.util, sys;"
        f"spec = importlib.util.spec_from_file_location('d', {str(module)!r});"
        "m = importlib.util.module_from_spec(spec);"
        "spec.loader.exec_module(m);"
        "assert m.month_ranges('2024-01-15', '2024-03-10')[0] == ('2024-01-15', '2024-01-31');"
        "print(','.join(sorted({n.split('.')[0] for n in sys.modules"
        " if n.split('.')[0] in {'rasterio', 'shapely', 'pystac_client', 'numpy', 'duckdb'}})))"
    )
    completed = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, check=True
    )

    assert completed.stdout.strip() == "", completed.stdout


_HEAVY = "{'rasterio', 'shapely', 'pystac_client', 'numpy'}"


def test_the_consumer_itself_loads_without_the_raster_and_stac_stack():
    """`check` on an ERA5 manifest must not pay for rasterio, shapely, numpy or pystac_client.

    The dependency-free homes only pay off if the consumer's OWN import path stays clear. It did
    not: ``src/utils/__init__.py`` re-exported ``src.utils.raster`` and ``src.utils.stac_client``,
    so ``from src.utils.duckdb_conn import duckdb_session`` executed that file and pulled the whole
    stack in by itself, whatever the leaf modules cost.

    Loaded from its path in a subprocess, so this session's already-imported modules do not answer
    for it. ``duckdb`` is expected here and deliberately not asserted against — the consumer
    imports it directly and every command needs it.
    """
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "import importlib.util, sys;"
            f"spec = importlib.util.spec_from_file_location('c', {str(_SCRIPT_PATH)!r});"
            "m = importlib.util.module_from_spec(spec);"
            "spec.loader.exec_module(m);"
            "assert m.check_manifest and m.main;"
            "print(','.join(sorted({n.split('.')[0] for n in sys.modules"
            f" if n.split('.')[0] in {_HEAVY}}})))",
        ],
        capture_output=True,
        text=True,
        check=True,
    )

    assert completed.stdout.strip() == "", completed.stdout


# --------------------------------------------------------------------------- #
# main() tells "wrong kind of object" apart from "engine bug"
# --------------------------------------------------------------------------- #


def _run_main(monkeypatch, raiser):
    monkeypatch.setattr(consumer, "_dispatch", raiser)
    monkeypatch.setattr(
        sys, "argv", ["climate_manifest_consumer", "check", "--manifest-uri", "s3://x/y.json"]
    )
    return consumer.main()


def test_a_uri_whose_object_is_not_valid_json_is_reported_as_input_not_as_a_crash(
    tmp_path, monkeypatch, capsys
):
    # Measured on duckdb 1.5.4: read_json_auto over a non-JSON body raises InvalidInputException,
    # not IOException — only a MISSING object gives IOException. `check` used to dump a traceback.
    not_json = tmp_path / "manifest.json"
    not_json.write_text("<html><body>403 Forbidden</body></html>")

    def _read_it(_args):
        duckdb.connect(":memory:").execute(
            "SELECT * FROM read_json_auto(?)", [not_json.as_posix()]
        ).fetchall()

    with pytest.raises(duckdb.InvalidInputException):
        _read_it(None)  # the fixture really does raise the exception this test is about

    assert _run_main(monkeypatch, _read_it) == 2
    captured = capsys.readouterr()
    assert captured.err.startswith("error: ")
    assert captured.err.count("\n") == 1


def test_a_non_parquet_artifact_is_reported_the_same_way(tmp_path, monkeypatch, capsys):
    not_parquet = tmp_path / "zone.parquet"
    not_parquet.write_text("nope")

    def _read_it(_args):
        duckdb.connect(":memory:").execute(
            "SELECT COUNT(*) FROM read_parquet(?)", [not_parquet.as_posix()]
        ).fetchall()

    assert _run_main(monkeypatch, _read_it) == 2
    assert capsys.readouterr().err.startswith("error: ")


def test_an_engine_bug_still_surfaces_its_traceback(monkeypatch):
    # InternalException is a sibling of InvalidInputException under DatabaseError. Catching the
    # parent would have swallowed the one class that means "this is a bug, not your input".
    def _boom(_args):
        raise duckdb.InternalException("INTERNAL Error: attempted to access index 9 of size 3")

    with pytest.raises(duckdb.InternalException):
        _run_main(monkeypatch, _boom)


# The store password reaches a manifest through prose, not through the address field. Both clients
# this repo talks to the object store with quote the endpoint IN FULL on a failure — measured on
# duckdb 1.5.4 and botocore, not assumed — and `src.utils.duckdb_conn` hands `S3_ENDPOINT` to
# `SET s3_endpoint` with only the scheme stripped, so httpfs has the userinfo to quote.
_HTTPFS_FAILURE = (
    "IO Error: Could not connect to server error for HTTP HEAD to "
    f"'http://{_IMPORT_ACCOUNT}:{_IMPORT_SECRET}@127.0.0.1:9000/theirs/climate/ndvi/z2.parquet'"
)


def test_an_inherited_error_message_cannot_carry_the_password_into_the_destination_bucket(
    monkeypatch,
):
    """`_rewrite_environment` cleans the address field. Every OTHER field is prose.

    A producer's `errors[].error` and `run_manifest_errors[].error` are verbatim exception text,
    and `import-zip` used to copy them through untouched into the manifest it publishes in the
    DESTINATION bucket — the bucket being handed to another party. Sanitizing `environment` while
    republishing the same endpoint two keys further down is not a sanitized manifest.
    """
    monkeypatch.setattr(
        consumer.settings,
        "S3_ENDPOINT",
        f"http://{_IMPORT_ACCOUNT}:{_IMPORT_SECRET}@127.0.0.1:9000",
    )
    monkeypatch.setattr(consumer.settings, "S3_FORCE_PATH_STYLE", True)
    manifest = _mixed_manifest()
    manifest["environment"] = {"s3_endpoint": "http://127.0.0.1:9100", "s3_bucket": "deposium"}
    manifest["errors"] = [{"zone_id": "z2", "error": f"readback failed: {_HTTPFS_FAILURE}"}]
    manifest["run_manifest_errors"] = [{"uri": "s3://theirs/x", "error": _HTTPFS_FAILURE}]

    published = _published_json(manifest)

    assert _IMPORT_SECRET not in published
    assert _IMPORT_ACCOUNT not in published
    # The diagnostic survives with the credential punched out, rather than the entry vanishing.
    assert "Could not connect to server" in json.loads(published)["errors"][0]["error"]


def test_the_one_line_error_report_does_not_print_the_store_password(monkeypatch, capsys):
    """The line an operator pastes into a ticket and CI keeps in its log.

    Every exception this handler catches comes back from DuckDB httpfs, which quotes the endpoint
    in full on a connection failure. Reporting it as one clean line instead of a traceback does not
    make it safe to print.
    """
    monkeypatch.setattr(
        consumer.settings,
        "S3_ENDPOINT",
        f"http://{_IMPORT_ACCOUNT}:{_IMPORT_SECRET}@127.0.0.1:9000",
    )
    monkeypatch.setattr(
        consumer,
        "_dispatch",
        lambda args: (_ for _ in ()).throw(duckdb.IOException(_HTTPFS_FAILURE)),
    )
    monkeypatch.setattr(
        sys, "argv", ["climate_manifest_consumer", "check", "--manifest-uri", "s3://b/c/m.json"]
    )

    exit_code = consumer.main()

    captured = capsys.readouterr()
    assert exit_code == 2
    assert _IMPORT_SECRET not in captured.err
    assert _IMPORT_ACCOUNT not in captured.err
    assert captured.err.startswith("error: ")
    assert captured.err.count("\n") == 1
    # Still says which host could not be reached — this removes the credential, not the diagnosis.
    assert "127.0.0.1:9000" in captured.err


def test_the_zero_exception_does_not_reach_a_one_day_era5_window(tmp_path, monkeypatch):
    """The derived floor that happens to EQUAL the convention is still derived.

    era5's floor is its day count. Over a one-day window that count is 1, the same number the
    "we expect data" convention uses for ndvi/unified — so keying the exception on
    ``bound.minimum == 1`` handed a one-day era5 entry the exception and accepted it advertising
    zero rows, which is exactly the under-claim the floor exists to reject.
    """
    manifest = _era5_manifest(zones=("montpellier",))
    manifest["parameters"]["end_date"] = manifest["parameters"]["start_date"]
    manifest["outputs"][0]["rows"] = 0
    uri = _install_manifest_only(monkeypatch, tmp_path, manifest)

    with pytest.raises(
        consumer.ManifestError,
        match=r"era5/montpellier advertises 0 rows, but its own parameters allow 1",
    ):
        consumer.check_manifest(uri, expected_zones=None)


# --------------------------------------------------------------------------- #
# The exit-code contract the storage doc publishes, pinned here instead of in a bucket
# --------------------------------------------------------------------------- #


def test_a_self_contradictory_manifest_exits_two_with_empty_stdout_and_one_stderr_line(
    tmp_path, monkeypatch, capsys
):
    """`docs/design/rustfs-storage-contract.md` states this row of the exit-code table.

    It used to state it by citing an object an agent had hand-written into one developer's dev
    RustFS (`s3://deposium/climate/runs/c4-probe/manifest.json`) — evidence no reader could
    reproduce, and evidence that would go on being cited after the object was deleted or the
    behaviour changed. The manifest below is that same fixture, self-contradictory in exactly the
    same way: two `era5` entries advertising 5 and 7 rows over a 2020 full-year window, whose own
    `parameters` allow 366.

    Every artifact URI stays unreachable, which is the point: exit 2 is reached at ZERO
    object-store round trips, before a single `read_parquet`.
    """
    manifest = _era5_manifest(zones=("z1", "z2"))
    manifest["parameters"]["start_date"] = "2020-01-01"
    manifest["parameters"]["end_date"] = "2020-12-31"
    manifest["outputs"] = [
        {"kind": "era5", "zone_id": "z1", "uri": _uri("era5", "z1"), "rows": 5},
        {"kind": "era5", "zone_id": "z2", "uri": _uri("era5", "z2"), "rows": 7},
    ]
    uri = _install_manifest_only(monkeypatch, tmp_path, manifest)
    monkeypatch.setattr(
        sys, "argv", ["climate_manifest_consumer", "check", "--manifest-uri", uri, "--json"]
    )

    exit_code = consumer.main()

    captured = capsys.readouterr()
    # Malformed/unsupported/self-contradictory: exit 2, whatever `--json` was asked for.
    assert exit_code == 2
    assert captured.out == ""
    # Exactly one line, and it names the FIRST contradiction only — the trade the doc states.
    assert captured.err.splitlines() == [
        "error: manifest output era5/z1 advertises 5 rows, but its own parameters allow 366"
    ]
    assert "era5/z2" not in captured.err


def test_an_account_name_that_prefixes_its_password_does_not_publish_the_tail():
    """Replacing in split order rewrites the short secret first and leaves the long one's tail.

    `deposium-dev` / `deposium-dev-secret` is the repo's own dev pair, so this is the shape the
    codebase actually runs with. Replacing `deposium-dev` first turns the password into
    `[REDACTED]-secret` — a leak wearing a redaction, which reads as handled in review.
    """
    published, reductions = consumer._sanitized_address(
        "s3://deposium-dev:deposium-dev-secret@far.away:9000/bucket/zone.parquet"
    )

    assert reductions == ("userinfo",)
    assert "deposium-dev-secret" not in published
    assert "-secret" not in published, f"the password tail survived: {published}"
    # What must survive: the location the field exists to record.
    assert published == "s3://[REDACTED]:[REDACTED]@far.away:9000/bucket/zone.parquet"


def test_a_password_that_prefixes_the_account_name_is_handled_the_same_way():
    """The other order, so the fix is length-based rather than a swap of two hard-coded positions."""
    # Both clear MIN_SECRET_LENGTH, so the floor is not what is being tested here — the ordering is.
    published, reductions = consumer._sanitized_address(
        "s3://tok3n-acct:tok3n-acct-and-more@far.away:9000/bucket/zone.parquet"
    )

    assert reductions == ("userinfo",)
    assert "tok3n-acct" not in published
    assert "-and-more" not in published, f"the account tail survived: {published}"


def test_an_all_scenes_ndvi_entry_may_not_advertise_zero(tmp_path, monkeypatch):
    """`monthly: false` drops the CEILING, so the conventional floor is the only bound left.

    Measured on 7b735e6, before this refusal existed: an all-scenes manifest advertising
    `rows: 0` over an empty parquet reported `ok: True`, `errors: []`, against the bound
    `rows >= 1` — the same floor an all-scenes entry with no advertised count is held to by
    `test_an_all_scenes_run_keeps_a_positive_floor_and_gains_no_ceiling`. An entry that can
    waive the only bound its kind has is an entry checked against nothing.

    `_install`, not `_install_manifest_only`: the artifact read is STUBBED so that removing the
    refusal lets the check run all the way to its verdict. Under `_install_manifest_only` the
    reverted code reached an unstubbed `read_parquet` and died on
    `_duckdb.HTTPException ... 404`, so the test failed on a missing network object rather than
    on the thing it is about. With a real, empty parquet behind the URI it now fails on
    `ok=True, errors=[]` -- the acceptance itself.
    """
    manifest = _ndvi_manifest(zones=("a",), monthly=False, advertised_rows={"a": 0})
    uri = _install(monkeypatch, tmp_path, manifest, {("ndvi", "a"): 0})

    try:
        result = consumer.check_manifest(uri, expected_zones=None)
    except consumer.ManifestError as exc:
        assert "ndvi/a advertises 0 rows, but its own parameters allow rows >= 1" in str(exc)
    else:
        raise AssertionError(
            "an all-scenes entry advertising 0 rows over an empty artifact was ACCEPTED: "
            f"ok={result['ok']}, errors={result['errors']}"
        )


def test_a_monthly_ndvi_entry_keeps_the_zero_exception(tmp_path, monkeypatch):
    """The other side of the same rule: a monthly slice keeps its derived ceiling and its
    `checks.month_coverage` denominator, so "no usable scene all window" stays a legal claim."""
    uri = _install(
        monkeypatch,
        tmp_path,
        _ndvi_manifest(zones=("a",), monthly=True, advertised_rows={"a": 0}),
        {("ndvi", "a"): 0},
    )

    result = consumer.check_manifest(uri, expected_zones=None)

    assert result["ok"], result["errors"]
    assert result["artifacts"][0]["row_bound"] == "0"


def test_a_bundle_carrying_only_source_environment_is_still_sanitized():
    """`_rewrite_environment` returned before doing anything when `environment` was absent.

    Measured on 7b735e6, with `_without_credential_fields` already wired into
    `_sanitized_environment`:

        source_environment published -> {"s3_endpoint":
            "http://origin-admin:origin-s3cret@far.away:9000", "s3_secret_key": "origin-s3cret"}

    Both tests added with that filter supply an `environment` block, so the only caller of the
    filter was never reached on this shape. A bundle is untrusted whether or not it shipped a
    sibling block.
    """
    manifest = {
        "kind": "geoai_run_manifest",
        "source_environment": {
            "s3_endpoint": "http://origin-admin:origin-s3cret@far.away:9000",
            "s3_secret_key": "origin-s3cret",
            "s3_bucket": "origin-bucket",
        },
        "source_environment_hops": 1,
    }

    rewritten = consumer._rewrite_manifest_uris(manifest, bucket="staging", run_id="imported-run")

    source = rewritten["source_environment"]
    # `sanitize_endpoint` refuses a userinfo-bearing authority WHOLE rather than trimming it,
    # so the address is dropped and the removal is recorded beside it.
    assert source["s3_endpoint"] is None
    assert source["s3_endpoint_credentials_removed"] is True
    assert source["s3_secret_key"] == consumer.REDACTED
    # Provenance that is not a credential survives, and the hop is counted like any other.
    assert source["s3_bucket"] == "origin-bucket"
    assert rewritten["source_environment_hops"] == 2
    serialized = json.dumps(rewritten)
    assert "origin-s3cret" not in serialized
    assert "origin-admin" not in serialized
    # The absent block is still absent: an empty origin was not invented for it.
    assert "environment" not in rewritten


def test_an_absent_environment_and_an_absent_source_environment_stay_absent():
    rewritten = consumer._rewrite_manifest_uris(
        _mixed_manifest(), bucket="staging", run_id="imported-run"
    )

    assert "environment" not in rewritten
    assert "source_environment" not in rewritten
    assert "source_environment_hops" not in rewritten


def test_an_output_entry_of_an_unsupported_kind_refuses_the_bundle():
    """ "Copied verbatim" was the default for an entry this consumer cannot lay out.

    Measured on 7b735e6: an entry with `kind: "lidar"` republished
    `s3://evil-admin:evil-s3cret@attacker.host/x.parquet` intact into the destination bucket.
    """
    manifest = {
        "kind": "geoai_run_manifest",
        "outputs": [
            {
                "kind": "lidar",
                "zone_id": "z",
                "uri": "s3://evil-admin:evil-s3cret@attacker.host/x.parquet",
            }
        ],
    }

    with pytest.raises(consumer.ManifestError, match=r"unsupported outputs kind: 'lidar'"):
        consumer._rewrite_manifest_uris(manifest, bucket="staging", run_id="imported-run")


def test_an_output_entry_without_a_zone_id_refuses_the_bundle():
    manifest = {
        "kind": "geoai_run_manifest",
        "outputs": [{"kind": "ndvi", "uri": "s3://evil-admin:evil-s3cret@attacker.host/y.parquet"}],
    }

    with pytest.raises(
        consumer.ManifestError,
        match=r"manifest outputs entry has a missing or unsafe zone_id: None",
    ):
        consumer._rewrite_manifest_uris(manifest, bucket="staging", run_id="imported-run")


def test_a_planned_output_entry_gets_the_same_gate():
    manifest = {
        "kind": "geoai_run_manifest",
        "planned_outputs": [
            {
                "kind": "era5",
                "zone_id": "../escape",
                "uri": "s3://evil-admin:evil-s3cret@attacker.host/z.parquet",
            }
        ],
    }

    with pytest.raises(
        consumer.ManifestError,
        match=r"manifest planned_outputs entry has a missing or unsafe zone_id",
    ):
        consumer._rewrite_manifest_uris(manifest, bucket="staging", run_id="imported-run")


def test_an_absolute_manifest_path_in_the_bundle_cannot_read_outside_the_root(tmp_path):
    """The zip-member guard protects EXTRACTION; `manifest_path` was read with no guard at all.

    `root / "/abs/path"` collapses to `/abs/path` — that is what `pathlib` does with an absolute
    right-hand side — so on 7b735e6 `_read_json` opened a file outside the extraction root, an
    arbitrary local JSON read driven by an untrusted bundle. The target here is a real file
    holding a real manifest, so a bundle that got through would be read successfully rather than
    failing on its own accord.
    """
    outside = tmp_path / "outside.json"
    outside.write_text(json.dumps({"kind": "geoai_run_manifest", "run_id": "somebody-elses-run"}))
    bundle = _bundle(
        tmp_path,
        {
            "bundle.json": json.dumps(
                {"manifest_path": outside.as_posix(), "artifacts": []}
            ).encode(),
        },
    )

    with pytest.raises(consumer.ManifestError, match=r"manifest_path escapes bundle root"):
        _import(bundle)


def test_a_traversing_manifest_path_in_the_bundle_is_rejected_too(tmp_path):
    outside = tmp_path / "outside.json"
    outside.write_text(json.dumps({"kind": "geoai_run_manifest", "run_id": "somebody-elses-run"}))
    bundle = _bundle(
        tmp_path,
        {
            "bundle.json": json.dumps(
                {"manifest_path": "../../../../../../etc/hosts", "artifacts": []}
            ).encode(),
        },
    )

    with pytest.raises(consumer.ManifestError, match=r"manifest_path escapes bundle root"):
        _import(bundle)


def test_a_manifest_path_that_is_not_a_string_is_rejected(tmp_path):
    bundle = _bundle(
        tmp_path,
        {"bundle.json": json.dumps({"manifest_path": ["manifest.json"]}).encode()},
    )

    with pytest.raises(consumer.ManifestError, match=r"manifest_path must be a non-empty string"):
        _import(bundle)


def _holdout_era5_manifest(zones: tuple[str, ...] = ("z",)) -> dict[str, Any]:
    """The shape `climate_baseline_panel --artifact-scope holdout` publishes: era5 outputs only,
    with `parameters.skip_normals: true` recording that normals were never promised."""
    return {
        "schema_version": 1,
        "kind": "geoai_run_manifest",
        "run_id": "test-holdout-run",
        "parameters": {
            "zone_ids": list(zones),
            "start_date": "2024-01-01",
            "end_date": "2024-01-10",
            "variables": ["temperature_2m"],
            "skip_normals": True,
        },
        "outputs": [{"kind": "era5", "zone_id": zone, "uri": _uri("era5", zone)} for zone in zones],
        "checks": {"ok": True, "zone_ok": True},
    }


def test_an_era5_only_holdout_manifest_is_not_failed_for_the_missing_normals_half(
    tmp_path, monkeypatch
):
    """The pair gate fired on `"era5" in kinds` without consulting `skip_normals`.

    Measured on 7b735e6 against this manifest:

        ok=False errors=["zones missing era5/normals pair: ['z']"]

    `scripts/climate_baseline_panel.py` sets `skip_normals` for the holdout artifact scope, so
    the consumer rejected manifests this repo's own panel produces.
    """
    uri = _install(monkeypatch, tmp_path, _holdout_era5_manifest(), {("era5", "z"): 10})

    result = consumer.check_manifest(uri, expected_zones=None)

    assert result["ok"], result["errors"]
    assert result["skip_normals"] is True
    assert result["expected_era5_rows"] == 10
    assert result["expected_normals_rows"] is None


def test_the_pair_gate_still_fires_when_the_run_promised_both_halves(tmp_path, monkeypatch):
    """The other side: without `skip_normals`, a missing normals half is still an error.

    HONEST SCOPE. The first two assertions are a NEGATIVE CONTROL, not a pin: the gate behaved
    this way before the fix too, so reverting the production hunk leaves them passing. What this
    test does pin independently is the third assertion — the resolved value is REPORTED, which is
    what lets an operator tell "this run declared it skipped normals" from "the consumer decided
    to ignore the pair". `.get`, not `[...]`: on the reverted code the key is simply absent and a
    `KeyError: 'skip_normals'` says nothing about the contract, whereas
    `assert None is False` names the claim.
    """
    manifest = _holdout_era5_manifest()
    del manifest["parameters"]["skip_normals"]
    uri = _install(monkeypatch, tmp_path, manifest, {("era5", "z"): 10})

    result = consumer.check_manifest(uri, expected_zones=None)

    assert not result["ok"]
    assert "zones missing era5/normals pair: ['z']" in result["errors"]
    assert result.get("skip_normals") is False


def test_a_non_boolean_skip_normals_is_refused(tmp_path, monkeypatch):
    """A `"true"` STRING must be refused, not truthy-tested.

    `_install`, not `_install_manifest_only`: the artifact read is stubbed so that removing
    `_skip_normals` lets the check reach a verdict instead of dying on an unstubbed
    `read_parquet` (`_duckdb.HTTPException ... 404`), which would have failed the test on a
    missing network object rather than on the acceptance it is about. Reverted, the string is
    ignored entirely and the run comes back with the pair-gate verdict, which is what the
    `else` branch prints.
    """
    manifest = _holdout_era5_manifest()
    manifest["parameters"]["skip_normals"] = "true"
    uri = _install(monkeypatch, tmp_path, manifest, {("era5", "z"): 10})

    try:
        result = consumer.check_manifest(uri, expected_zones=None)
    except consumer.ManifestError as exc:
        assert "parameters.skip_normals must be a boolean, got 'true'" in str(exc)
    else:
        raise AssertionError(
            "a non-boolean parameters.skip_normals was not refused: "
            f"ok={result['ok']}, errors={result['errors']}"
        )


def test_a_non_string_output_kind_exits_two_instead_of_dumping_a_traceback(
    tmp_path, monkeypatch, capsys
):
    """`kind in _KIND_TO_PREFIX` is a dict lookup, and a JSON list is not hashable.

    Measured on 7b735e6:

        TypeError: cannot use 'list' as a dict key (unhashable type: 'list')

    `TypeError` is not in `main`'s except tuple, so the contract's "exit 2, one stderr line"
    became a traceback. Fixed at the validation site rather than by widening that tuple: a
    genuine engine bug must keep its traceback.
    """
    manifest = _ndvi_manifest(zones=("a",))
    manifest["outputs"][0]["kind"] = ["ndvi"]
    uri = _install_manifest_only(monkeypatch, tmp_path, manifest)
    monkeypatch.setattr(
        sys, "argv", ["climate_manifest_consumer", "check", "--manifest-uri", uri, "--json"]
    )

    exit_code = consumer.main()

    captured = capsys.readouterr()
    assert exit_code == 2
    assert captured.out == ""
    assert captured.err.splitlines() == ["error: unsupported output kind: ['ndvi']"]


def test_a_dict_output_kind_is_refused_the_same_way(tmp_path, monkeypatch):
    manifest = _ndvi_manifest(zones=("a",))
    manifest["outputs"][0]["kind"] = {"ndvi": True}
    uri = _install_manifest_only(monkeypatch, tmp_path, manifest)

    with pytest.raises(consumer.ManifestError, match=r"unsupported output kind: \{'ndvi': True\}"):
        consumer.check_manifest(uri, expected_zones=None)


def test_a_bundle_output_kind_that_is_not_a_string_is_refused_on_import():
    """`_destination_uri` shares the hazard: it is reached from the bundle artifact loop."""
    with pytest.raises(consumer.ManifestError, match=r"unsupported artifact kind: \['ndvi'\]"):
        consumer._destination_uri("deposium", ["ndvi"], "z")


@pytest.mark.parametrize(
    ("start_date", "end_date", "expected_bound", "rows"),
    [
        ("9999-01-01", "9999-12-31", "1 <= rows <= 12", 3),
        # A one-month window makes floor and ceiling coincide, and the bound renders collapsed.
        ("9999-12-01", "9999-12-31", "1", 1),
    ],
)
def test_a_window_at_the_iso_upper_bound_is_processed_not_refused(
    tmp_path, monkeypatch, start_date, end_date, expected_bound, rows
):
    """This used to assert exit 2, and that assertion encoded a bug rather than a contract.

    `month_ranges` reached the end of a month through `date(cur.year + 1, 1, 1)`, so a
    December-9999 window asked the stdlib for year 10000 and raised
    `ValueError: year must be in 1..9999, not 10000`. The consumer wrapped that into its exit-2
    contract, and this test pinned the wrapper — refusing a window whose dates are both perfectly
    legal ISO, and leaving the helper still raising for `src/tools/_ndvi.py`, which does not wrap it.

    The helper is fixed at the source now, so the window is ordinary: it slices, and the month
    ceiling is derived from it like any other.
    """
    uri = _install(
        monkeypatch,
        tmp_path,
        _ndvi_manifest(zones=("a",), start_date=start_date, end_date=end_date),
        {("ndvi", "a"): rows},
    )

    result = consumer.check_manifest(uri, expected_zones=None)

    assert result["ok"], result["errors"]
    assert result["row_bounds"]["ndvi"] == expected_bound


def test_the_month_count_at_the_iso_upper_bound_is_twelve():
    """`_month_count` no longer wraps a `ValueError` -- there is none to wrap since 86874a7."""
    assert consumer._month_count("9999-01-01", "9999-12-31") == 12
    assert consumer._month_count("9999-12-01", "9999-12-31") == 1


def test_a_window_below_the_overflow_still_derives_its_month_ceiling(tmp_path, monkeypatch):
    """The refusal is bounded to the windows the arithmetic cannot represent: 9998 still works."""
    uri = _install(
        monkeypatch,
        tmp_path,
        _ndvi_manifest(zones=("a",), start_date="9998-01-01", end_date="9998-12-31"),
        {("ndvi", "a"): 3},
    )

    result = consumer.check_manifest(uri, expected_zones=None)

    assert result["ok"], result["errors"]
    assert result["row_bounds"]["ndvi"] == "1 <= rows <= 12"


# --------------------------------------------------------------------------- #
# import-zip: the documented exit-2 contract covers every bad-bundle shape
# --------------------------------------------------------------------------- #
#
# `main`'s except tuple is deliberately narrow so a genuine bug keeps its traceback, which means
# every input problem has to be converted at ITS validation site. Five shapes escaped as exit 1
# with a traceback on f4f9263 -- all of them bundle content, none of them engine faults. Each test
# below names the shape and asserts `ManifestError`, the type `main` keys on; the CLI test at the
# end pins the end of the chain (exit 2, one stderr line, empty stdout) for the whole class.


def _bundle_with_manifest_path(tmp_path: Path, manifest_path, extra=None) -> Path:
    members = {
        "bundle.json": json.dumps(
            {
                "manifest_path": manifest_path,
                "artifacts": [{"kind": "era5", "zone_id": "z", "path": "climate/z.parquet"}],
            }
        ).encode(),
        "manifest.json": json.dumps({"run_id": "r1"}).encode(),
        "climate/z.parquet": b"PAR1 not really a parquet \x00\xff",
    }
    members.update(extra or {})
    return _bundle(tmp_path, members)


def test_a_manifest_path_naming_no_file_is_operator_input(tmp_path):
    """FileNotFoundError: [Errno 2] No such file or directory: '.../nope.json'"""
    bundle = _bundle_with_manifest_path(tmp_path, "nope.json")

    with pytest.raises(consumer.ManifestError, match="cannot be read"):
        _import(bundle)


def test_a_manifest_path_naming_the_bundle_root_is_operator_input(tmp_path):
    """IsADirectoryError: [Errno 21] Is a directory: '.../geoai-climate-import-xxxx'

    `.` resolves to the extraction root itself, so it passes the containment guard one line
    above and dies on the read.
    """
    bundle = _bundle_with_manifest_path(tmp_path, ".")

    with pytest.raises(consumer.ManifestError, match="cannot be read"):
        _import(bundle)


def test_a_manifest_path_naming_a_directory_inside_the_bundle_is_operator_input(tmp_path):
    """IsADirectoryError: [Errno 21] Is a directory: '.../climate'"""
    bundle = _bundle_with_manifest_path(tmp_path, "climate")

    with pytest.raises(consumer.ManifestError, match="cannot be read"):
        _import(bundle)


def test_a_manifest_path_naming_a_binary_member_is_operator_input(tmp_path):
    """UnicodeDecodeError: 'utf-8' codec can't decode byte 0xff in position 21

    Pointing `manifest_path` at one of the bundle's own parquet artifacts is a plausible
    export/import mistake, and `read_text` fails before `json.loads` ever sees the bytes -- so
    the existing JSONDecodeError guard never fires.
    """
    bundle = _bundle_with_manifest_path(tmp_path, "climate/z.parquet")

    with pytest.raises(consumer.ManifestError, match="not UTF-8 text"):
        _import(bundle)


def test_a_manifest_path_with_an_embedded_null_is_operator_input(tmp_path):
    """ValueError: lstat: embedded null character in path

    The only shape that never opens a file: `resolve()` raises from inside `posixpath.realpath`,
    one line ABOVE the containment guard. `ManifestError` subclasses `ValueError`, so this test
    asserts the subclass explicitly -- a bare `pytest.raises(ValueError)` would have passed
    against the unfixed code.
    """
    bundle = _bundle_with_manifest_path(tmp_path, "man\x00ifest.json")

    with pytest.raises(consumer.ManifestError, match="null byte"):
        _import(bundle)


def test_a_bundle_with_no_bundle_json_is_operator_input(tmp_path):
    """FileNotFoundError: [Errno 2] No such file or directory: '.../bundle.json'

    A zip that is simply not an export bundle -- the very first read in `import_zip`.
    """
    bundle = _bundle(tmp_path, {"README.txt": b"hello"})

    with pytest.raises(consumer.ManifestError, match="cannot be read"):
        _import(bundle)


def test_a_bad_bundle_reaches_the_cli_as_exit_2_and_one_stderr_line(tmp_path):
    """The end of the chain: `ManifestError` only helps if `main` turns it into the contract.

    Measured on f4f9263 this exited 1 with a full traceback on stderr.
    """
    bundle = _bundle_with_manifest_path(tmp_path, "climate/z.parquet")

    completed = subprocess.run(
        [
            sys.executable,
            str(_SCRIPT_PATH),
            "import-zip",
            "--bundle",
            str(bundle),
            "--bucket",
            "deposium",
        ],
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 2, completed.stderr
    assert completed.stdout == ""
    assert completed.stderr.count("\n") == 1, completed.stderr
    assert completed.stderr.startswith("error: ")
    assert "Traceback" not in completed.stderr


class TestConfinedBundleMembers:
    r"""Both bundle-supplied paths get one guard, because two copies drifted once already.

    The zip-member loop protects EXTRACTION only. `manifest_path` and every
    `artifacts[].relative_path` are read from `bundle.json` AFTER it, and `pathlib` collapses
    `root / "/etc/passwd"` to the absolute path.
    """

    HOSTILE = [
        pytest.param("/etc/passwd", id="absolute"),
        pytest.param("../../escape.json", id="parent-segments"),
        pytest.param("z\x00.parquet", id="null-byte"),
        pytest.param("\ud800.parquet", id="surrogate"),
        pytest.param("", id="empty"),
        pytest.param(None, id="not-a-string"),
        pytest.param(17, id="not-a-string-either"),
    ]

    @pytest.mark.parametrize("value", HOSTILE)
    def test_a_hostile_member_raises_manifest_error_not_a_traceback(self, value, tmp_path):
        """ManifestError is what `main` turns into exit 2 and one stderr line.

        The null byte and the surrogate had to be refused BEFORE `resolve()`: it raises them
        itself, one line above the containment check, so they escaped as exit 1 with a traceback.
        Neither opens a file, so no read guard catches them either.
        """
        with pytest.raises(consumer.ManifestError):
            consumer._confined_member(tmp_path.resolve(), value, "artifacts[].relative_path")

    def test_the_error_names_the_field_it_came_from(self, tmp_path):
        with pytest.raises(consumer.ManifestError) as raised:
            consumer._confined_member(tmp_path.resolve(), "/etc/passwd", "manifest_path")
        assert "manifest_path" in str(raised.value)

    def test_an_ordinary_member_resolves(self, tmp_path):
        root = tmp_path.resolve()
        (root / "climate" / "ndvi").mkdir(parents=True)
        (root / "climate" / "ndvi" / "z.parquet").write_bytes(b"x")

        resolved = consumer._confined_member(
            root, "climate/ndvi/z.parquet", "artifacts[].relative_path"
        )

        assert resolved == root / "climate" / "ndvi" / "z.parquet"


# --------------------------------------------------------------------------- #
# The last substantive round: six findings that reproduced on 6fbd599
# --------------------------------------------------------------------------- #


class TestACredentialNamedFlagIsNotACredential:
    """`_without_credential_fields` matched the SUBSTRING "credential" and caught its own flag."""

    CLEAN = "http://clean.host:9000"

    def test_the_removal_flag_survives_as_a_boolean(self):
        """Measured on 6fbd599:

           {"s3_endpoint_credentials_removed": False}
        -> {"s3_endpoint_credentials_removed": "[REDACTED]"}
        """
        block = {"s3_endpoint": self.CLEAN, "s3_endpoint_credentials_removed": False}

        assert consumer._without_credential_fields(block)["s3_endpoint_credentials_removed"] is (
            False
        )

    def test_a_clean_origin_endpoint_does_not_report_a_removal_that_never_happened(self):
        """The field exists to answer exactly one question -- "is the recorded address shorter
        than the configured one" -- and it lied in the clean case. `_sanitized_environment` OR-s
        the incoming flag into the one it publishes, and `bool("[REDACTED]")` is True, so a
        non-empty redaction string read as a removal a reader could not disprove."""
        published = consumer._sanitized_environment(
            {"s3_endpoint": self.CLEAN, "s3_endpoint_credentials_removed": False}
        )

        assert published["s3_endpoint"] == self.CLEAN
        assert published["s3_endpoint_credentials_removed"] is False

    def test_a_real_removal_still_reports_one(self):
        """The flag must stay able to say True, or the fix has only moved the lie.

        `sanitize_endpoint` REFUSES a userinfo-bearing endpoint whole rather than trimming it, so
        the published address is `None` and the flag is what carries the news."""
        published = consumer._sanitized_environment(
            {"s3_endpoint": "http://origin-admin:origin-s3cret@far.away:9000"}
        )

        assert published["s3_endpoint"] is None
        assert published["s3_endpoint_credentials_removed"] is True

    def test_an_upstream_true_is_never_degraded_to_false(self):
        """The OR is load-bearing: the incoming flag records a removal at the PRODUCER, this one
        records a removal here, and either answers the question the field exists for."""
        published = consumer._sanitized_environment(
            {"s3_endpoint": self.CLEAN, "s3_endpoint_credentials_removed": True}
        )

        assert published["s3_endpoint_credentials_removed"] is True

    def test_a_string_under_a_credential_named_key_is_still_withheld(self):
        """The case the substring match was written for, unchanged."""
        block = {"s3_secret_key": "origin-s3cret", "aws_session_token": "origin-tok"}

        assert consumer._without_credential_fields(block) == {
            "s3_secret_key": consumer.REDACTED,
            "aws_session_token": consumer.REDACTED,
        }

    def test_a_number_under_a_credential_named_key_is_still_withheld(self):
        """An all-numeric password is a real shape -- it is the same ambiguity
        `sanitize_stac_url` fails closed on -- so only `bool` and `None` are exempt."""
        assert consumer._without_credential_fields({"s3_secret_key": 1234}) == {
            "s3_secret_key": consumer.REDACTED
        }

    def test_a_null_credential_is_not_turned_into_a_claim_that_one_was_shipped(self):
        """This REPLACES rather than deletes so a reader can see the origin shipped a credential.
        `null` says it shipped none, and redacting it manufactures the opposite claim."""
        assert consumer._without_credential_fields({"s3_secret_key": None}) == {
            "s3_secret_key": None
        }


class TestAnOutputUriIsBoundToItsOwnKindAndZone:
    """`_outputs` checked that `uri` was S3-SHAPED and never that it was THIS entry's uri."""

    def _manifest(self, uri: str) -> dict[str, Any]:
        manifest = _ndvi_manifest(zones=("montpellier_metropole",))
        manifest["outputs"][0]["uri"] = uri
        return manifest

    def test_an_entry_pointing_at_another_zones_artifact_is_refused(self):
        """The substitution. The entry is labelled `ndvi`/`montpellier_metropole` while its uri
        names `bassin_du_lez`, and every gate passed: the URI is well-formed, and
        `_zone_identity_errors` compares the declared zones against the entries' own METADATA,
        never against where they point. `check` then read the wrong parquet under the right name
        and `export_zip` bundled those bytes as `climate/ndvi/montpellier_metropole.parquet`."""
        with pytest.raises(consumer.ManifestError, match="its own kind and zone do not imply"):
            consumer._outputs(self._manifest("s3://deposium/climate/ndvi/bassin_du_lez.parquet"))

    def test_an_entry_pointing_at_another_kinds_artifact_is_refused(self):
        with pytest.raises(consumer.ManifestError, match="its own kind and zone do not imply"):
            consumer._outputs(
                self._manifest("s3://deposium/climate/era5/montpellier_metropole.parquet")
            )

    def test_the_canonical_layout_is_accepted(self):
        """`_default_output_uri`: s3://{bucket}/climate/{kind}/{zone_id}.parquet"""
        uri = "s3://deposium/climate/ndvi/montpellier_metropole.parquet"

        assert consumer._outputs(self._manifest(uri))[0]["uri"] == uri

    def test_the_holdout_layout_is_accepted_rather_than_broken(self):
        """A legitimate producer in THIS repo writes outside the canonical tree:
        `climate_baseline_panel._holdout_era5_uri` lays artifacts out as
        s3://{bucket}/climate/holdout/era5/{window_label}/{zone}.parquet. A rule stated as one
        hard-coded prefix would have refused a manifest this repo's own panel produces."""
        manifest = _era5_manifest(zones=("montpellier_metropole",))
        holdout = "s3://deposium/climate/holdout/era5/2019_2020/montpellier_metropole.parquet"
        for output in manifest["outputs"]:
            if output["kind"] == "era5":
                output["uri"] = holdout

        kept = {o["uri"] for o in consumer._outputs(manifest)}

        assert holdout in kept

    def test_the_bucket_is_deliberately_not_bound(self):
        """A manifest legitimately describes objects in a store this process does not write to --
        that is the whole point of import-zip."""
        uri = "s3://someone-elses-bucket/climate/ndvi/montpellier_metropole.parquet"

        assert consumer._outputs(self._manifest(uri))[0]["uri"] == uri


class TestTwoRowClaimsThatDisagree:
    """Both keys were validated; only the first found was ever returned."""

    def test_a_second_claim_that_disagrees_is_refused_rather_than_dropped(self):
        """Measured on 6fbd599: `{"rows": 1, "n_matched": 2}` returned 1 and the 2 vanished, so a
        manifest contradicting itself about the artifact it hands over was accepted as a one-row
        claim -- the consumer choosing which of the two numbers to believe on tuple order."""
        with pytest.raises(consumer.ManifestError, match="two row counts that disagree"):
            consumer._advertised_rows({"kind": "ndvi", "zone_id": "z", "rows": 1, "n_matched": 2})

    def test_the_message_names_both_numbers_and_both_keys(self):
        with pytest.raises(consumer.ManifestError) as excinfo:
            consumer._advertised_rows({"kind": "ndvi", "zone_id": "z", "rows": 1, "n_matched": 2})

        assert "rows=1" in str(excinfo.value)
        assert "n_matched=2" in str(excinfo.value)

    def test_two_claims_that_agree_are_fine(self):
        """A unified entry legitimately carries both keys, and agreeing with itself is not a
        contradiction."""
        assert (
            consumer._advertised_rows(
                {"kind": "unified", "zone_id": "z", "rows": 3, "n_matched": 3}
            )
            == 3
        )

    def test_one_claim_is_still_that_claim(self):
        assert consumer._advertised_rows({"kind": "ndvi", "zone_id": "z", "rows": 3}) == 3

    def test_no_claim_is_still_absence(self):
        assert consumer._advertised_rows({"kind": "ndvi", "zone_id": "z"}) is None


class TestARefusalDoesNotPrintWhatItRefused:
    """Every one of these fires BEFORE `_incoming_addresses` has collected the origin's
    credentials, so no redactor can help -- the value must simply not be echoed."""

    CREDENTIAL = "https://origin-user:origin-pass@host"

    def test_a_string_environment_is_refused_without_quoting_it(self):
        """`_incoming_addresses` skips it for not being a dict, and the refusal then printed the
        credential the ACCEPT path would have removed. Measured on 6fbd599:

            error: manifest environment must be a JSON object, got
                   'https://origin-user:origin-pass@host'
        """
        with pytest.raises(consumer.ManifestError) as excinfo:
            consumer._rewrite_environment({"environment": self.CREDENTIAL}, bucket="dest")

        assert "origin-pass" not in str(excinfo.value)
        assert "got str" in str(excinfo.value)

    def test_a_string_source_environment_is_refused_without_quoting_it(self):
        with pytest.raises(consumer.ManifestError) as excinfo:
            consumer._rewrite_environment({"source_environment": self.CREDENTIAL}, bucket="dest")

        assert "origin-pass" not in str(excinfo.value)
        assert "got str" in str(excinfo.value)

    def test_a_string_outputs_block_is_refused_without_quoting_it(self):
        with pytest.raises(consumer.ManifestError) as excinfo:
            consumer._rewrite_manifest_uris(
                {"outputs": self.CREDENTIAL}, bucket="dest", run_id="r1"
            )

        assert "origin-pass" not in str(excinfo.value)

    def test_a_string_outputs_entry_is_refused_without_quoting_it(self):
        with pytest.raises(consumer.ManifestError) as excinfo:
            consumer._rewrite_manifest_uris(
                {"outputs": [self.CREDENTIAL]}, bucket="dest", run_id="r1"
            )

        assert "origin-pass" not in str(excinfo.value)

    def test_the_field_and_its_type_still_reach_the_operator(self):
        """Refusing silently would be its own defect: the message has to say what to fix."""
        with pytest.raises(consumer.ManifestError, match="manifest environment must be a JSON"):
            consumer._rewrite_environment({"environment": 7}, bucket="dest")


def test_an_unopenable_bundle_is_exit_two_rather_than_a_traceback(tmp_path, monkeypatch, capsys):
    """`zipfile.ZipFile(...)` was the one unconverted `except` escape left in import-zip.

    Measured on 6fbd599, `import-zip --bundle /nonexistent/nope.zip` raised FileNotFoundError,
    which is not in `main`'s except tuple, and exited 1 with a traceback.
    """
    missing = tmp_path / "nope.zip"
    monkeypatch.setattr(
        sys,
        "argv",
        ["climate_manifest_consumer", "import-zip", "--bundle", str(missing), "--bucket", "b"],
    )

    exit_code = consumer.main()

    captured = capsys.readouterr()
    assert exit_code == 2
    assert captured.out == ""
    assert captured.err.splitlines() == [
        f"error: bundle cannot be opened: {missing.as_posix()} (FileNotFoundError)"
    ]


def test_a_bundle_that_is_not_a_zip_is_exit_two_rather_than_a_traceback(
    tmp_path, monkeypatch, capsys
):
    """BadZipFile is not an OSError, so it needed its own conversion: a truncated download or an
    HTML error page saved under the bundle's name is the realistic shape."""
    corrupt = tmp_path / "corrupt.zip"
    corrupt.write_bytes(b"<html>404 Not Found</html>")
    monkeypatch.setattr(
        sys,
        "argv",
        ["climate_manifest_consumer", "import-zip", "--bundle", str(corrupt), "--bucket", "b"],
    )

    exit_code = consumer.main()

    captured = capsys.readouterr()
    assert exit_code == 2
    assert captured.out == ""
    assert captured.err.splitlines() == [
        f"error: bundle is not a readable zip: {corrupt.as_posix()}"
    ]


class TestNothingIsWrittenForABundleThatWasAlwaysGoingToBeRefused:
    """Two defects in one place: the copy loop interleaved validation with writes, and the two
    inventories a bundle carries were never compared to each other."""

    @staticmethod
    def _bundle_for(tmp_path: Path, artifacts, outputs, manifest=None) -> Path:
        manifest = manifest or {
            "kind": "geoai_run_manifest",
            "run_id": "origin-run",
            "parameters": {"zone_ids": sorted({z for _k, z in outputs})},
            "outputs": [
                {"kind": kind, "zone_id": zone, "uri": _uri(kind, zone)} for kind, zone in outputs
            ],
        }
        members = {
            "bundle.json": json.dumps(
                {
                    "manifest_path": "manifest.json",
                    "artifacts": [
                        {
                            "kind": kind,
                            "zone_id": zone,
                            "relative_path": f"climate/{kind}/{zone}.parquet",
                        }
                        for kind, zone in artifacts
                    ],
                }
            ).encode(),
            "manifest.json": json.dumps(manifest).encode(),
        }
        for kind, zone in artifacts:
            members[f"climate/{kind}/{zone}.parquet"] = b"PAR1 stand-in"
        return _bundle(tmp_path, members)

    @staticmethod
    def _record_copies(monkeypatch) -> list[str]:
        """Stub the two sites that would otherwise reach a real object store."""
        copied: list[str] = []

        def _copy(_conn, _source, destination):
            copied.append(destination)

        class _Stub:
            """The manifest write is a plain `execute` on the session; only the copies matter."""

            def execute(self, _sql, _parameters=None):
                return None

        @contextmanager
        def _session(*_args, **_kwargs):
            yield _Stub()

        monkeypatch.setattr(consumer, "_copy_parquet", _copy)
        monkeypatch.setattr(consumer, "duckdb_session", _session)
        return copied

    def test_a_valid_bundle_still_imports_every_artifact(self, tmp_path, monkeypatch):
        """The control. Restructuring the loop must not have stopped it doing its job."""
        copied = self._record_copies(monkeypatch)
        bundle = self._bundle_for(
            tmp_path, [("ndvi", "a"), ("ndvi", "b")], [("ndvi", "a"), ("ndvi", "b")]
        )

        result = _import(bundle)

        assert result["n_imported"] == 2
        assert sorted(copied) == [_uri("ndvi", "a"), _uri("ndvi", "b")]

    def test_a_second_invalid_artifact_leaves_nothing_behind(self, tmp_path, monkeypatch):
        """The partial import. Measured on 6fbd599: the loop validated one artifact, copied it,
        and only then looked at the next -- so a bundle with a valid first entry and an
        unsupported second `kind` wrote `climate/ndvi/a.parquet` into the HANDOFF bucket and
        then raised, leaving an object there with no manifest vouching for it."""
        copied = self._record_copies(monkeypatch)
        bundle = self._bundle_for(tmp_path, [("ndvi", "a"), ("lidar", "b")], [("ndvi", "a")])

        with pytest.raises(consumer.ManifestError, match="unsupported artifact kind"):
            _import(bundle)

        assert copied == []

    def test_an_invalid_manifest_entry_also_leaves_nothing_behind(self, tmp_path, monkeypatch):
        """The other half: the MANIFEST's own entries were not validated until
        `_rewrite_manifest_uris`, which ran after the copy loop had finished."""
        copied = self._record_copies(monkeypatch)
        bundle = self._bundle_for(
            tmp_path,
            [("ndvi", "a")],
            [("ndvi", "a")],
            manifest={
                "kind": "geoai_run_manifest",
                "run_id": "origin-run",
                "outputs": [
                    {"kind": "ndvi", "zone_id": "a", "uri": _uri("ndvi", "a")},
                    {"kind": "ndvi", "zone_id": "..", "uri": _uri("ndvi", "b")},
                ],
            },
        )

        with pytest.raises(consumer.ManifestError, match="unsafe zone_id"):
            _import(bundle)

        assert copied == []

    def test_a_bundle_shipping_one_set_and_advertising_another_is_refused(
        self, tmp_path, monkeypatch
    ):
        """The inventory mismatch. `bundle.artifacts` drives what gets COPIED and
        `manifest.outputs` is what gets ADVERTISED, and nothing compared them: a bundle shipping
        a/c while advertising a/b imported c's parquet with nothing vouching for it, and
        published an entry for `climate/ndvi/b.parquet` that was never written -- or worse, that
        resolves to a STALE object an earlier run left there, which the fresh manifest now
        vouches for as part of this run."""
        copied = self._record_copies(monkeypatch)
        bundle = self._bundle_for(
            tmp_path, [("ndvi", "a"), ("ndvi", "c")], [("ndvi", "a"), ("ndvi", "b")]
        )

        with pytest.raises(consumer.ManifestError) as excinfo:
            _import(bundle)

        assert "different artifact sets" in str(excinfo.value)
        assert "ndvi/b" in str(excinfo.value)
        assert "ndvi/c" in str(excinfo.value)
        assert copied == []

    def test_a_duplicated_shipment_is_a_difference_too(self, tmp_path, monkeypatch):
        """Counted rather than set-compared, so shipping the same artifact twice against a
        manifest that advertises it once is caught."""
        self._record_copies(monkeypatch)
        bundle = self._bundle_for(tmp_path, [("ndvi", "a"), ("ndvi", "a")], [("ndvi", "a")])

        with pytest.raises(consumer.ManifestError, match="different artifact sets"):
            _import(bundle)

    def test_planned_outputs_are_deliberately_not_compared(self, tmp_path, monkeypatch):
        """`planned_outputs` records what the run INTENDED to write. An artifact that was planned
        and not produced is exactly the shape a partial run legitimately hands over, so comparing
        it would refuse a correct bundle."""
        copied = self._record_copies(monkeypatch)
        bundle = self._bundle_for(
            tmp_path,
            [("ndvi", "a")],
            [("ndvi", "a")],
            manifest={
                "kind": "geoai_run_manifest",
                "run_id": "origin-run",
                "outputs": [{"kind": "ndvi", "zone_id": "a", "uri": _uri("ndvi", "a")}],
                "planned_outputs": [
                    {"kind": "ndvi", "zone_id": "a", "uri": _uri("ndvi", "a")},
                    {"kind": "ndvi", "zone_id": "b", "uri": _uri("ndvi", "b")},
                ],
            },
        )

        assert _import(bundle)["n_imported"] == 1
        assert copied == [_uri("ndvi", "a")]


class TestImportAppliesTheStructuralGateCheckApplies:
    """`check` refuses a manifest without a usable `outputs` list before it reads anything;
    `import-zip` copied every artifact and published the manifest anyway.

    Measured on e99100c: a bundle whose manifest had no `outputs` key sailed through
    `_rewrite_manifest_uris` (which validates the entries it FINDS) and
    `_refuse_an_inventory_mismatch` (which returned early on the absent list), had its artifacts
    copied into the destination bucket, and got a manifest published that `check` then refused
    with "manifest outputs must be a non-empty list". A manifest of the wrong `kind` did the
    same: `_read_manifest` asks, `import_zip` never did. Both gates now run before the first
    copy, through the same functions `check` uses.
    """

    _bundle_for = staticmethod(
        TestNothingIsWrittenForABundleThatWasAlwaysGoingToBeRefused._bundle_for
    )

    @staticmethod
    def _record_writes(monkeypatch) -> tuple[list[str], list[str]]:
        """Every write the import can make: the parquet copies and the manifest publish."""
        copied: list[str] = []
        executed: list[str] = []

        def _copy(_conn, _source, destination):
            copied.append(destination)

        class _Stub:
            def execute(self, sql, _parameters=None):
                executed.append(sql)
                return None

        @contextmanager
        def _session(*_args, **_kwargs):
            yield _Stub()

        monkeypatch.setattr(consumer, "_copy_parquet", _copy)
        monkeypatch.setattr(consumer, "duckdb_session", _session)
        return copied, executed

    @pytest.mark.parametrize(
        "manifest",
        [
            pytest.param(
                {"kind": "geoai_run_manifest", "run_id": "origin-run"}, id="no-outputs-key"
            ),
            pytest.param(
                {"kind": "geoai_run_manifest", "run_id": "origin-run", "outputs": []},
                id="empty-outputs",
            ),
        ],
    )
    def test_a_manifest_advertising_nothing_is_refused_before_any_copy(
        self, tmp_path, monkeypatch, manifest
    ):
        copied, executed = self._record_writes(monkeypatch)
        bundle = self._bundle_for(tmp_path, [("ndvi", "a")], [("ndvi", "a")], manifest=manifest)

        with pytest.raises(consumer.ManifestError, match="manifest outputs must be a non-empty"):
            _import(bundle)

        assert copied == []
        assert executed == []

    def test_a_manifest_of_another_kind_is_refused_before_any_copy(self, tmp_path, monkeypatch):
        copied, executed = self._record_writes(monkeypatch)
        bundle = self._bundle_for(
            tmp_path,
            [("ndvi", "a")],
            [("ndvi", "a")],
            manifest={
                "kind": "something_else",
                "run_id": "origin-run",
                "outputs": [{"kind": "ndvi", "zone_id": "a", "uri": _uri("ndvi", "a")}],
            },
        )

        with pytest.raises(
            consumer.ManifestError, match="manifest kind must be 'geoai_run_manifest'"
        ):
            _import(bundle)

        assert copied == []
        assert executed == []

    def test_a_valid_bundle_still_imports_and_publishes(self, tmp_path, monkeypatch):
        """The control: the gate refuses what `check` refuses and nothing more."""
        copied, executed = self._record_writes(monkeypatch)
        bundle = self._bundle_for(
            tmp_path, [("ndvi", "a"), ("ndvi", "b")], [("ndvi", "a"), ("ndvi", "b")]
        )

        result = _import(bundle)

        assert result["n_imported"] == 2
        assert sorted(copied) == [_uri("ndvi", "a"), _uri("ndvi", "b")]
        assert len(executed) == 1
        assert executed[0].startswith("COPY (SELECT * FROM read_json_auto(?)) TO ")


class TestARefusedHttpAddressIsNotReinterpreted:
    """`sanitize_stac_url` gives the same refusal for two different reasons; only one has a fallback.

    "This is not http(s), the policy does not apply" is an `s3://` object location and the userinfo
    fallback is right for it. "This IS http(s) and it is ambiguous" has no address to publish, and
    reinterpreting it asserted a host nobody configured.
    """

    def test_a_legitimate_catalogue_with_an_email_in_its_path_publishes_no_invented_host(self):
        """Measured before the gate: the real host became the redaction and example.com the host."""
        published, reductions = consumer._sanitized_address(
            "https://stac.internal:8443/catalogs/user@example.com/v1"
        )

        assert published == consumer.REDACTED
        assert reductions == (consumer.STAC_REFUSED,)
        # The failure this replaces: `stac.internal:8443` read as userinfo, `example.com` as host.
        assert "example.com" not in published
        assert "stac.internal" not in published

    def test_an_ambiguous_catalogue_credential_publishes_no_address_either(self):
        published, reductions = consumer._sanitized_address(
            "https://stac-reader:1234/5678@stac.example/v1"
        )

        assert published == consumer.REDACTED
        assert reductions == (consumer.STAC_REFUSED,)
        assert "stac-reader" not in published

    def test_an_s3_location_still_keeps_everything_but_its_userinfo(self):
        """The fallback exists for this shape: an object location must survive to be provenance."""
        published, reductions = consumer._sanitized_address(
            "s3://deposium-dev:deposium-dev-secret@far.away:9000/bucket/zone.parquet"
        )

        assert reductions == ("userinfo",)
        assert published == "s3://[REDACTED]:[REDACTED]@far.away:9000/bucket/zone.parquet"
        assert "-secret" not in published

    def test_a_clean_catalogue_root_is_published_unchanged(self):
        assert consumer._sanitized_address("https://stac.example/v1") == (
            "https://stac.example/v1",
            (),
        )

    def test_a_clean_s3_location_is_published_unchanged(self):
        uri = "s3://bucket/climate/ndvi/z.parquet"
        assert consumer._sanitized_address(uri) == (uri, ())


class TestAnIncomingTokenBelowTheAuthority:
    """The gate read USERINFO only, so a catalogue token never reached the sanitizer.

    `sanitize_stac_url` already knew how to reduce it — the gate above it never let the value
    through, so `import-zip` republished `inputs[].endpoint` with the token intact while the
    diagnostic half of the same manifest had it punched out.
    """

    def test_a_query_token_is_reduced_out_of_a_structural_address(self):
        """Reported as `query`, which is all `sanitize_stac_url` can honestly say.

        The policy drops the whole query because SOME queries carry a token; it does not parse
        the parameters, so it cannot tell this `?token=` from a `?collections=`. Naming the
        component is therefore the precise claim — a query was removed — where the old
        `endpoint_credentials_removed` asserted a fact about the query's CONTENTS that nothing
        had established. Whether the policy should keep a non-credential query at all is the open
        question recorded on #178; this test pins the reporting, not that decision.
        """
        manifest = {"inputs": [{"kind": "stac", "endpoint": "https://s.example/v1?token=abcdefgh"}]}

        consumer._sanitize_incoming_inputs(manifest)

        assert manifest["inputs"][0]["endpoint"] == "https://s.example/v1"
        assert manifest["inputs"][0]["endpoint_reductions"] == ["query"]

    def test_a_clean_catalogue_root_is_still_republished_byte_for_byte(self):
        """A sanitizer that rewrites what it had no reason to touch destroys the provenance."""
        uri = "https://earth-search.aws.element84.com/v1"
        manifest = {"inputs": [{"kind": "stac", "endpoint": uri}]}

        consumer._sanitize_incoming_inputs(manifest)

        assert manifest["inputs"][0] == {"kind": "stac", "endpoint": uri}

    def test_a_short_username_does_not_mangle_the_object_location(self):
        """`s3://u:pw@far.away/b/z.parquet` came back with `z.parquet` as `z.parq[REDACTED]et`.

        The one-character username was used as a replacement pattern. The location is the whole
        reason this fallback punches out userinfo instead of dropping the field.
        """
        manifest = {
            "inputs": [{"kind": "era5", "uris": ["s3://u:pw12345678@far.away/b/z.parquet"]}]
        }

        consumer._sanitize_incoming_inputs(manifest)

        published = manifest["inputs"][0]["uris"][0]
        assert published == "s3://u:[REDACTED]@far.away/b/z.parquet"
        assert "pw12345678" not in published


class TestADroppedFilterIsNotReportedAsARemovedCredential:
    """#178 — the import path asserted a credential removal for every reduction it made.

    `sanitize_stac_url` drops the query from EVERY catalogue root, credential or not, so the most
    ordinary STAC endpoint there is came out of `import-zip` flagged as having carried a secret.
    Neither direction had a test, which is why the claim went unchallenged: the assertions only
    ever checked that the *address* was right.
    """

    def test_a_collections_filter_reports_the_query_not_a_credential(self):
        """The defect, stated as the manifest states it.

        Measured on `main` @ 40206b4::

            {'endpoint': 'https://earth-search.aws.element84.com/v1',
             'endpoint_credentials_removed': True}

        `?collections=sentinel-2-l2a` is a provenance filter. Losing it is a real cost and is
        worth recording — as a lost FILTER, which is what happened.
        """
        manifest = {
            "inputs": [
                {
                    "kind": "stac",
                    "endpoint": (
                        "https://earth-search.aws.element84.com/v1?collections=sentinel-2-l2a"
                    ),
                }
            ]
        }

        consumer._sanitize_incoming_inputs(manifest)

        item = manifest["inputs"][0]
        assert item["endpoint"] == "https://earth-search.aws.element84.com/v1"
        assert item["endpoint_reductions"] == ["query"]
        assert "endpoint_credentials_removed" not in item
        assert "userinfo" not in item["endpoint_reductions"]

    def test_a_real_catalogue_credential_still_reports_the_userinfo(self):
        """The other direction, so the fix is not "stop reporting" — it is "report which"."""
        manifest = {
            "inputs": [
                {"kind": "stac", "endpoint": "https://stac-reader:stac-s3cret@stac.origin/v1"}
            ]
        }

        consumer._sanitize_incoming_inputs(manifest)

        item = manifest["inputs"][0]
        assert item["endpoint"] == "https://stac.origin/v1"
        assert item["endpoint_reductions"] == ["userinfo"]

    def test_a_credential_and_a_filter_on_one_address_report_both(self):
        """`https://user:tok@host/v1?collections=…` loses two different things at once."""
        manifest = {
            "inputs": [
                {
                    "kind": "stac",
                    "endpoint": (
                        "https://stac-reader:stac-s3cret@stac.origin/v1?collections=sentinel-2-l2a"
                    ),
                }
            ]
        }

        consumer._sanitize_incoming_inputs(manifest)

        item = manifest["inputs"][0]
        assert item["endpoint"] == "https://stac.origin/v1"
        assert item["endpoint_reductions"] == ["userinfo", "query"]

    def test_the_producers_own_reduction_history_survives_the_import(self):
        """A producer that already recorded `["query"]` must not have it overwritten.

        The address published by the producer is already a PREFIX of what was configured. If the
        import stamps its own `["userinfo"]` over that, the record says the address is complete
        apart from a credential — and it is not.
        """
        manifest = {
            "inputs": [
                {
                    "kind": "stac",
                    "endpoint": "https://stac-reader:stac-s3cret@stac.origin/v1",
                    "endpoint_reductions": ["query"],
                }
            ]
        }

        consumer._sanitize_incoming_inputs(manifest)

        assert manifest["inputs"][0]["endpoint_reductions"] == ["query", "userinfo"]

    def test_an_unknown_reduction_name_from_a_foreign_producer_is_kept(self):
        """Dropping it would be this code committing the deletion it exists to prevent."""
        manifest = {
            "inputs": [
                {
                    "kind": "stac",
                    "endpoint": "https://stac-reader:stac-s3cret@stac.origin/v1",
                    "endpoint_reductions": ["some-future-component"],
                }
            ]
        }

        consumer._sanitize_incoming_inputs(manifest)

        assert manifest["inputs"][0]["endpoint_reductions"] == [
            "some-future-component",
            "userinfo",
        ]

    def test_a_list_field_unions_the_categories_of_its_entries(self):
        """`uris` reduces per entry, and the field-level record must cover all of them."""
        manifest = {
            "inputs": [
                {
                    "kind": "era5",
                    "uris": [
                        "s3://era5-reader:era5-s3cret@far.away:9000/theirs/z1.parquet",
                        "https://stac.origin/v1?collections=sentinel-2-l2a",
                        "s3://theirs/climate/era5/z2.parquet",
                    ],
                }
            ]
        }

        consumer._sanitize_incoming_inputs(manifest)

        item = manifest["inputs"][0]
        assert item["uris_reductions"] == ["userinfo", "query"]
        # The entry with nothing to remove is still republished byte for byte.
        assert item["uris"][2] == "s3://theirs/climate/era5/z2.parquet"

    def test_an_address_with_nothing_removable_gains_no_reductions_field(self):
        """The quiet case, and the one a "report everything" fix would get wrong."""
        manifest = {
            "inputs": [{"kind": "stac", "endpoint": "https://earth-search.aws.element84.com/v1"}]
        }

        consumer._sanitize_incoming_inputs(manifest)

        item = manifest["inputs"][0]
        assert item["endpoint"] == "https://earth-search.aws.element84.com/v1"
        assert "endpoint_reductions" not in item
        assert "endpoint_credentials_removed" not in item


class TestTheLegacyBooleanDoesNotSurviveTheMigration:
    """PR #187 review — the field this change replaces was left in place beside its replacement.

    A manifest written by the pre-#178 importer already carries `{key}_credentials_removed: true`.
    Adding `{key}_reductions` without removing it produced either a contradiction or, worse, the
    original false claim surviving completely untouched.
    """

    def test_the_boolean_is_removed_and_migrated_beside_the_new_categories(self):
        """Measured before the pop: `endpoint_credentials_removed: true` alongside `["query"]`."""
        manifest = {
            "inputs": [
                {
                    "kind": "stac",
                    "endpoint": (
                        "https://earth-search.aws.element84.com/v1?collections=sentinel-2-l2a"
                    ),
                    "endpoint_credentials_removed": True,
                }
            ]
        }

        consumer._sanitize_incoming_inputs(manifest)

        item = manifest["inputs"][0]
        assert "endpoint_credentials_removed" not in item
        # `unspecified` rather than a guess: the old boolean cannot be mapped onto a component, and
        # dropping it would publish the address as though it were the configured value verbatim.
        assert item["endpoint_reductions"] == ["unspecified", "query"]

    def test_the_boolean_is_removed_even_when_the_address_needs_no_rewrite(self):
        """The nastier half: the early exit used to leave the false claim entirely alone.

        A clean address carries nothing to remove, so the loop `continue`d before touching the
        siblings — and `endpoint_credentials_removed: true` went into the destination bucket
        asserting a credential removal on an address that never had one.
        """
        manifest = {
            "inputs": [
                {
                    "kind": "stac",
                    "endpoint": "https://clean.example/v1",
                    "endpoint_credentials_removed": True,
                }
            ]
        }

        consumer._sanitize_incoming_inputs(manifest)

        item = manifest["inputs"][0]
        assert "endpoint_credentials_removed" not in item
        assert item["endpoint_reductions"] == ["unspecified"]
        assert item["endpoint"] == "https://clean.example/v1"

    def test_a_false_legacy_boolean_leaves_no_trace_at_all(self):
        """`false` said nothing was removed, so migrating it would invent a reduction."""
        manifest = {
            "inputs": [
                {
                    "kind": "stac",
                    "endpoint": "https://clean.example/v1",
                    "endpoint_credentials_removed": False,
                }
            ]
        }

        consumer._sanitize_incoming_inputs(manifest)

        item = manifest["inputs"][0]
        assert "endpoint_credentials_removed" not in item
        assert "endpoint_reductions" not in item


class TestASecretBelowTheAuthorityOfANonHttpAddress:
    """PR #187 review — the gate said "this carries a credential" and the sanitizer removed nothing.

    `_carries_a_credential` reads `url_secrets`, which covers the userinfo, the query AND the
    fragment. The non-http fallback read `endpoint_credentials`, which is the userinfo alone. So an
    `s3://` URI with a token on its query was selected for sanitization and then republished byte
    for byte, into the destination bucket, with the token intact.
    """

    TOKEN = "abcdefghij"

    def test_a_query_token_on_an_s3_uri_is_removed_and_reported(self):
        """Measured before the fix: returned verbatim, reductions `()`, token published."""
        published, reductions = consumer._sanitized_address(
            f"s3://bucket/climate/era5/z.parquet?token={self.TOKEN}"
        )

        assert self.TOKEN not in published
        assert published == "s3://bucket/climate/era5/z.parquet?token=[REDACTED]"
        assert reductions == ("query",)

    def test_a_fragment_token_is_reached_too(self):
        """Never sent to the server, but quoted back by anything echoing the configured string."""
        published, reductions = consumer._sanitized_address(
            f"s3://bucket/k.parquet#token={self.TOKEN}"
        )

        assert self.TOKEN not in published
        assert reductions == ("fragment",)

    def test_a_userinfo_and_a_query_secret_are_reported_separately(self):
        """Two regions, two categories, in the order the policy walks the URL."""
        published, reductions = consumer._sanitized_address(
            f"s3://user:passwordlong@far.away/b/z.parquet?token={self.TOKEN}"
        )

        assert "passwordlong" not in published
        assert self.TOKEN not in published
        assert reductions == ("userinfo", "query")

    def test_a_query_token_does_not_rewrite_the_same_text_in_the_host_or_path(self):
        published, reductions = consumer._sanitized_address(
            f"s3://{self.TOKEN}/{self.TOKEN}?token={self.TOKEN}"
        )

        assert published == f"s3://{self.TOKEN}/{self.TOKEN}?token=[REDACTED]"
        assert reductions == ("query",)

    def test_a_path_at_sign_does_not_extend_raw_userinfo_into_the_object_key(self):
        published, reductions = consumer._sanitized_address(
            "s3://user:passwordlong@passwordlong/bucket/passwordlong@copy"
        )

        assert published == "s3://user:[REDACTED]@passwordlong/bucket/passwordlong@copy"
        assert reductions == ("userinfo",)

    def test_an_encoded_path_at_sign_does_not_extend_encoded_userinfo_into_the_key(self):
        published, reductions = consumer._sanitized_address(
            "s3://user:passwordlong%40passwordlong/bucket/passwordlong%40copy"
        )

        assert published == "s3://user:[REDACTED]%40passwordlong/bucket/passwordlong%40copy"
        assert reductions == ("userinfo",)

    def test_the_same_token_in_query_and_fragment_reports_both_reductions(self):
        published, reductions = consumer._sanitized_address(
            f"s3://bucket/k?token={self.TOKEN}#token={self.TOKEN}"
        )

        assert published == "s3://bucket/k?token=[REDACTED]#token=[REDACTED]"
        assert reductions == ("query", "fragment")

    def test_reductions_keep_url_order_when_the_fragment_secret_is_longer(self):
        published, reductions = consumer._sanitized_address(
            "s3://bucket/k?token=abcdefgh#token=abcdefghijklmnop"
        )

        assert published == "s3://bucket/k?token=[REDACTED]#token=[REDACTED]"
        assert reductions == ("query", "fragment")

    def test_encoded_parameter_names_and_tokens_follow_the_same_region_policy(self):
        published, reductions = consumer._sanitized_address(
            "s3://bucket/k?api%5Fkey=abc%2Fdefghij#%74oken=uvw%2Fxyzabc"
        )

        assert published == "s3://bucket/k?api%5Fkey=[REDACTED]#%74oken=[REDACTED]"
        assert reductions == ("query", "fragment")

    def test_the_list_path_no_longer_republishes_the_token(self):
        """`uris` is the field the drift left behind once already; this is the same field again.

        Before the fix `cleaned == value` held — nothing had been rewritten — so the loop
        `continue`d and the entry went out with its token.
        """
        uri = f"s3://bucket/climate/era5/z.parquet?token={self.TOKEN}"
        manifest = {
            "inputs": [{"kind": "era5", "uris": [uri, "s3://bucket/climate/era5/z2.parquet"]}]
        }

        consumer._sanitize_incoming_inputs(manifest)

        item = manifest["inputs"][0]
        assert self.TOKEN not in json.dumps(item)
        assert item["uris_reductions"] == ["query"]
        # The sibling with nothing to remove is still republished byte for byte.
        assert item["uris"][1] == "s3://bucket/climate/era5/z2.parquet"

    def test_a_short_secret_is_still_left_alone(self):
        """The MIN_SECRET_LENGTH floor applies below the authority too.

        A one-character value is a substring of ordinary text, and punching it out mangles the
        location this fallback exists to preserve.
        """
        published, reductions = consumer._sanitized_address("s3://bucket/k.parquet?token=a")

        assert published == "s3://bucket/k.parquet?token=a"
        assert reductions == ()
