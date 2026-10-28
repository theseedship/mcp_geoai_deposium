"""FIERCE zone-panel contract tests (DoD lot B1, issue #25).

Validates the generator output: 15 zones (12 panel + 3 named FR pilots), documented
geometries with a valid Koppen class, an explicit CRS, and bbox/geometry consistency.

The 12 panel zones are 1deg boxes. The 3 pilots are real IGN ADMIN EXPRESS / Sandre
BD TOPAGE boundaries vendored under `zones/sources/`, so they may legitimately be
MultiPolygon (Montpellier Metropole owns a detached islet off Villeneuve-les-Maguelone)
and must carry their provenance so a consumer can honour the LOv2 attribution.
"""

from __future__ import annotations

import importlib.util
import json
import re
from pathlib import Path

import pytest
from shapely.geometry import shape
from shapely.wkt import loads as load_wkt

from src.cli.forecast_forward_archive import CANONICAL_ZONES
from src.tools.climate import (
    SCIENTIFIC_ZONE_ROLES,
    ZONE_ROLES,
    _load_zones,
    _scientific_zone_ids,
)
from src.utils.duckdb_conn import get_duckdb_connection

REPO = Path(__file__).resolve().parent.parent
ZONES = REPO / "zones"

# The exact CRS member the generator emits (RFC 7946 CRS84, longitude/latitude).
EXPECTED_CRS = {"type": "name", "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"}}

# Full Koppen-Geiger code set (main class + valid sub-letters).
KOPPEN = re.compile(
    r"^(Af|Am|Aw|As|BWh|BWk|BSh|BSk|Csa|Csb|Csc|Cwa|Cwb|Cwc|Cfa|Cfb|Cfc|"
    r"Dsa|Dsb|Dsc|Dsd|Dwa|Dwb|Dwc|Dwd|Dfa|Dfb|Dfc|Dfd|ET|EF)$"
)
PILOTS = {"montpellier_metropole", "bassin_du_lez", "fos_sur_mer"}


def _load_generator():
    spec = importlib.util.spec_from_file_location("genz", REPO / "scripts" / "generate_zones.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# Per-zone split folder for each role, as the generator files them.
ROLE_DIRS = _load_generator().ROLE_DIRS


@pytest.fixture(scope="module")
def features():
    fc = json.loads((ZONES / "zones.geojson").read_text())
    return fc["features"]


def test_collection_has_crs():
    fc = json.loads((ZONES / "zones.geojson").read_text())
    assert fc["type"] == "FeatureCollection"
    assert "CRS84" in json.dumps(fc["crs"]), "CRS must be explicit (EPSG:4326 / CRS84)"


def test_fifteen_zones_twelve_panel_three_pilot(features):
    # Count by role, not a total: a client site is tenant-scoped and never part of the 12 + 3.
    roles = [f["properties"]["role"] for f in features]
    assert set(roles) <= ZONE_ROLES, f"unknown zone role in {sorted(set(roles))}"
    assert roles.count("panel") == 12
    assert roles.count("pilot") == 3


def test_three_named_pilots_present(features):
    ids = {f["properties"]["zone_id"] for f in features if f["properties"]["role"] == "pilot"}
    assert ids == PILOTS


def test_every_zone_has_valid_schema_and_geometry(features):
    seen = set()
    for f in features:
        p = f["properties"]
        for key in ("zone_id", "name", "koppen", "bbox", "role"):
            assert key in p, f"{p.get('zone_id')} missing {key}"
        assert p["zone_id"] not in seen, f"duplicate zone_id {p['zone_id']}"
        seen.add(p["zone_id"])
        assert KOPPEN.match(p["koppen"]), f"{p['zone_id']}: invalid Koppen {p['koppen']!r}"
        geom = shape(f["geometry"])
        # panel zones are boxes; pilots are real boundaries and may be multi-part
        expected = ("Polygon",) if p["role"] == "panel" else ("Polygon", "MultiPolygon")
        assert geom.is_valid and geom.geom_type in expected, f"{p['zone_id']}: {geom.geom_type}"
        # bbox property matches geometry bounds
        assert [round(c, 4) for c in geom.bounds] == p["bbox"]
        # plausible WGS84 ranges
        minx, miny, maxx, maxy = geom.bounds
        assert -180 <= minx < maxx <= 180 and -90 <= miny < maxy <= 90


def test_per_zone_files_exist(features):
    for f in features:
        sub = ROLE_DIRS[f["properties"]["role"]]
        assert (ZONES / sub / f"{f['properties']['zone_id']}.geojson").exists()


def test_koppen_spread_is_diverse(features):
    # the panel exists to span climate classes — guard against a degenerate all-same panel
    classes = {f["properties"]["koppen"] for f in features}
    assert len(classes) >= 8, f"expected a diverse Koppen panel, got {sorted(classes)}"


def test_pilots_are_real_boundaries_not_bboxes(features):
    # the point of #25: the pilots must no longer be the 5-vertex rectangles they started as
    for f in features:
        p = f["properties"]
        if p["role"] != "pilot":
            continue
        geom = shape(f["geometry"])
        assert not geom.equals(geom.envelope), f"{p['zone_id']} is still a bounding box"
        # a real admin/hydrographic outline carries hundreds of vertices, a bbox carries 5
        n_coords = sum(len(g.exterior.coords) for g in getattr(geom, "geoms", [geom]))
        assert n_coords > 100, f"{p['zone_id']}: only {n_coords} vertices"
        # and it must be materially smaller than the bbox it used to be approximated by
        assert geom.area < 0.85 * geom.envelope.area, f"{p['zone_id']} nearly fills its envelope"


def test_pilots_carry_provenance_and_area(features):
    # LOv2 is attribution-only: a consumer reading zones.geojson alone must still be able
    # to credit the source, so provenance travels with the feature.
    for f in features:
        p = f["properties"]
        if p["role"] != "pilot":
            continue
        for key in ("area_km2", "source", "dataset", "licence", "attribution", "definition"):
            assert key in p, f"{p['zone_id']} missing {key}"
        assert "Licence Ouverte" in p["licence"], f"{p['zone_id']}: {p['licence']!r}"
        assert p["area_km2"] > 0


def test_pilot_features_match_their_vendored_boundary(features):
    # zones.geojson must be a faithful projection of zones/sources/, not a drifted copy
    for f in features:
        p = f["properties"]
        if p["role"] != "pilot":
            continue
        source = json.loads((ZONES / "sources" / f"{p['zone_id']}.geojson").read_text())
        assert shape(f["geometry"]).equals(shape(source["geometry"]))
        assert p["area_km2"] == source["properties"]["area_km2"]
        assert p["name"] == source["properties"]["name"]


@pytest.fixture(scope="module")
def generated_parquet(features, tmp_path_factory):
    # zones.parquet is gitignored (its Snappy pages are not byte-reproducible across machines,
    # even at an identical DuckDB build), so the Parquet contract is guarded *semantically*:
    # regenerate it once here and let the tests compare values, not committed bytes.
    mod = _load_generator()
    out_dir = tmp_path_factory.mktemp("zones_parquet")
    status = mod.write_geoparquet(features, out_dir)
    assert status.startswith("ok"), f"write_geoparquet did not produce a Parquet: {status}"
    return out_dir / "zones.parquet"


def test_parquet_areas_match_geojson(features, generated_parquet):
    # the Parquet materialization must agree with the canonical GeoJSON: an INTEGER staging
    # column in write_geoparquet() silently rounded the geodesic pilot areas (589.3886 -> 589)
    pq = generated_parquet
    con = get_duckdb_connection(":memory:")
    parquet_areas = dict(
        con.execute(
            f"SELECT zone_id, area_km2 FROM read_parquet('{pq.as_posix()}') WHERE role = 'pilot'"
        ).fetchall()
    )
    con.close()
    geojson_areas = {
        f["properties"]["zone_id"]: f["properties"]["area_km2"]
        for f in features
        if f["properties"]["role"] == "pilot"
    }
    assert parquet_areas == geojson_areas


def test_generator_fails_loudly_when_a_vendored_boundary_is_missing(tmp_path):
    mod = _load_generator()
    with pytest.raises(FileNotFoundError, match="refresh_pilot_boundaries"):
        mod.build_features(sources_dir=tmp_path)


def test_generator_is_deterministic():
    mod = _load_generator()
    a = mod.build_features()
    b = mod.build_features()
    assert a == b
    assert sum(1 for f in a if f["properties"]["role"] in SCIENTIFIC_ZONE_ROLES) == 15


def test_geoparquet_artifact_readable(features, generated_parquet):
    # the B1 artifact set includes a DuckDB-Spatial-readable parquet (geometry as WKB),
    # generated on demand — see the generated_parquet fixture for why it is not committed
    pq = generated_parquet
    con = get_duckdb_connection(":memory:")
    n = con.execute(f"SELECT count(*) FROM read_parquet('{pq.as_posix()}')").fetchone()[0]
    assert n == len(features)
    wkt = con.execute(
        f"SELECT ST_AsText(ST_GeomFromWKB(geometry)) FROM read_parquet('{pq.as_posix()}') LIMIT 1"
    ).fetchone()[0]
    assert wkt.startswith("POLYGON")
    con.close()


def test_generator_reproduces_committed_geojson(tmp_path):
    """The generator must regenerate the committed GeoJSON byte-for-byte.

    Closes the generator<->artifact drift gap: the other tests read the committed
    zones.geojson as their source of truth, so editing scripts/generate_zones.py
    (e.g. a Koppen value) WITHOUT re-running it left every test green. This is the
    only test that runs build_features() + write_geojson() + _crs_member() and proves
    their output equals what is committed. zones.geojson is byte-stable, so an exact
    byte compare is the strongest available reproducibility assertion.
    """
    mod = _load_generator()
    feats = mod.build_features()
    mod.write_geojson(feats, tmp_path)

    # combined collection: byte-identical to the committed canonical artifact
    assert (tmp_path / "zones.geojson").read_bytes() == (ZONES / "zones.geojson").read_bytes()

    # exact split-file inventory, per directory. write_geojson() only writes, it never deletes,
    # so comparing file-by-file over the generated features would leave an orphan committed split
    # (from a renamed or dropped zone) invisible — the very drift this test exists to catch.
    expected: dict[str, set[str]] = {sub: set() for sub in mod.ROLE_DIRS.values()}
    for f in feats:
        sub = mod.ROLE_DIRS[f["properties"]["role"]]
        expected[sub].add(f"{f['properties']['zone_id']}.geojson")
    for sub, names in expected.items():
        assert {p.name for p in (ZONES / sub).glob("*.geojson")} == names, sub

    # every per-zone split file: byte-identical to its committed counterpart
    for sub, names in expected.items():
        for name in sorted(names):
            assert (tmp_path / sub / name).read_bytes() == (ZONES / sub / name).read_bytes(), name


def test_crs_is_exact_crs84_on_collection_and_every_per_zone_file(features):
    """CRS must be the exact CRS84 URN — on the collection AND each per-zone file.

    Replaces the substring check (which 'CRS84-BOGUS' would satisfy) with structural
    equality, and extends coverage to the 15 per-zone files (whose crs member was
    never read by any test).
    """
    fc = json.loads((ZONES / "zones.geojson").read_text())
    assert fc["crs"] == EXPECTED_CRS
    for f in features:
        sub = ROLE_DIRS[f["properties"]["role"]]
        single = json.loads((ZONES / sub / f"{f['properties']['zone_id']}.geojson").read_text())
        assert single.get("crs") == EXPECTED_CRS, f["properties"]["zone_id"]


def test_parquet_matches_geojson_on_every_zone_and_column(features, generated_parquet):
    """Every Parquet row must agree with the canonical GeoJSON — all 15 zones, all columns.

    ``test_parquet_areas_match_geojson`` only guards ``area_km2`` on the 3 pilots, so a wrong
    ``name``/``koppen``/``role`` or a mangled geometry on any zone passes it; and
    ``test_geoparquet_artifact_readable`` only checks a count plus one LIMIT-1 row. This closes
    both gaps by cross-checking `write_geoparquet()`'s WKB/DuckDB round-trip against the source
    GeoJSON for the whole panel.

    There is deliberately no comparison against a committed ``zones.parquet``: the artifact is
    generated on demand and gitignored (its Snappy pages are not byte-reproducible across
    machines), so ``zones.geojson`` is the only available source of truth.
    """
    con = get_duckdb_connection(":memory:")
    rows = con.execute(
        "SELECT zone_id, name, koppen, role, area_km2, "
        "ST_AsText(ST_GeomFromWKB(geometry)) AS wkt "
        f"FROM read_parquet('{generated_parquet.as_posix()}') ORDER BY zone_id"
    ).fetchall()
    con.close()

    geo = {f["properties"]["zone_id"]: f for f in features}
    # Exact zone_id identity, not just a count: 15 rows with one zone duplicated and another
    # omitted would satisfy a length check while every row still resolved against `geo`.
    # The query is ORDER BY zone_id, so this also fails on duplicates.
    assert [r[0] for r in rows] == sorted(geo)
    assert len(rows) == len(features)

    for zone_id, name, koppen, role, area_km2, wkt in rows:
        p = geo[zone_id]["properties"]
        assert (name, koppen, role, area_km2) == (
            p["name"],
            p["koppen"],
            p["role"],
            p.get("area_km2"),
        ), zone_id
        assert load_wkt(wkt).equals(shape(geo[zone_id]["geometry"])), zone_id


# A client site as scripts/generate_zones.py::CLIENTS would declare it:
# (zone_id, name, min_lon, min_lat, max_lon, max_lat, koppen).
_CLIENT_PROBE = ("client_probe", "Client probe", 4.8, 43.35, 5.0, 43.5, "Csa")


def test_generator_files_a_client_apart_and_keeps_the_scientific_panel(
    features, tmp_path, monkeypatch
):
    """A client site survives regeneration, in its own folder, after the unchanged 12 + 3."""
    mod = _load_generator()
    monkeypatch.setattr(mod, "CLIENTS", [_CLIENT_PROBE])
    mod.write_geojson(mod.build_features(), tmp_path)

    generated = json.loads((tmp_path / "zones.geojson").read_text())["features"]
    assert generated[:15] == features
    assert [f["properties"]["role"] for f in generated[15:]] == ["client"]
    assert (tmp_path / "client_sites" / "client_probe.geojson").is_file()
    assert not (tmp_path / "panel_global" / "client_probe.geojson").exists()


def test_generator_refuses_an_unknown_role(tmp_path):
    mod = _load_generator()
    assert set(mod.ROLE_DIRS) == ZONE_ROLES
    feats = mod.build_features()
    feats[0]["properties"]["role"] = "panle"
    with pytest.raises(KeyError, match="panle"):
        mod.write_geojson(feats, tmp_path)


def test_scientific_zone_ids_are_the_canonical_fifteen_and_never_a_client():
    zones = _load_zones()
    assert tuple(_scientific_zone_ids(zones)) == CANONICAL_ZONES
    with_client = {**zones, "client_probe": {"role": "client"}}
    assert tuple(_scientific_zone_ids(with_client)) == CANONICAL_ZONES


@pytest.mark.parametrize("role", [None, "panle", ""])
def test_scientific_zone_ids_refuse_a_missing_or_unknown_role(role):
    zones = {**_load_zones(), "bad_zone": {"role": role}}
    with pytest.raises(ValueError, match="bad_zone"):
        _scientific_zone_ids(zones)
