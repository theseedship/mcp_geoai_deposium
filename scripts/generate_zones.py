#!/usr/bin/env python3
"""Generate the FIERCE 15-zone panel as GeoJSON (FIERCE DoD lot B1, issue #25).

Panel global (12 zones): a 1deg x 1deg `shapely.geometry.box` per city, spanning a
deliberate Koppen spread (Mediterranean -> desert -> tropical -> subarctic -> tundra).
Pilotes France (3 zones): the real administrative / hydrographic boundaries, read from the
vendored `zones/sources/{zone_id}.geojson` produced by `scripts/refresh_pilot_boundaries.py`.

CRS is OGC:CRS84 (WGS84 longitude/latitude) for every geometry — the GeoJSON default per
RFC 7946 — emitted explicitly as a named-CRS member (the FIERCE DoD requires "CRS specified").
Köppen classes use the 0 °C C/D isotherm convention; they are representative values pending
geomatics validation. `area_km2` is carried only on the pilots — a 1° panel box area is
latitude-dependent (~12 300 km² at 43°N vs ~3 600 km² at 78°N), so it is intentionally omitted there.

The script is **idempotent**: re-running it reproduces the exact same files. The canonical
GeoJSON path is offline; the optional Parquet materialization calls DuckDB ``INSTALL spatial``
and therefore requires a cached/vendored extension, or network access on a cold machine.

Caveat on ``zones.parquet``: its bytes are NOT reproducible across machines, which is why it is
gitignored and generated on demand rather than committed. Two independent effects were measured on
PR #124: (a) DuckDB stamps its own version into the file (``created_by = "DuckDB version v1.5.4
(build ...)"``), so a different DuckDB writes different bytes — the #42 D3 version-sensitivity
(at the time ``duckdb>=0.10.0`` with ``uv.lock`` gitignored pinned nothing; since #129 committed
``uv.lock`` and #132 bounded DuckDB to 1.5.4, the lock pins that version); and (b) even at an
*identical* DuckDB build (same ``created_by``, same declared encodings), the Snappy-compressed
data pages differ across machines (360/229 371 bytes on v1.5.4/08e34c447b) while the decoded
content is identical — so pinning alone would not make the binary reproducible either. Runs on
one machine are byte-stable.
``zones.geojson`` is the canonical committed artifact; ``tests/test_zones.py`` guards the Parquet
*semantically* (regenerate → values must match the GeoJSON). Output:

    zones/panel_global/{zone_id}.geojson   # one Feature per zone (design §2.1 format)
    zones/pilot_france/{zone_id}.geojson
    zones/client_sites/{zone_id}.geojson   # only once CLIENTS is non-empty (none today)
    zones/zones.geojson                     # combined FeatureCollection (all 15) — CANONICAL B1 artifact
    zones/zones.parquet                     # on-demand, gitignored: geometry as WKB (ST_GeomFromWKB)

The GeoJSON is the canonical, DoD-satisfying artifact. `zones.parquet` stores geometry as a WKB
blob (not a metadata-tagged GeoParquet — that needs the GDAL Parquet driver, absent here); DuckDB
consumers wrap it with `ST_GeomFromWKB(geometry)` (the B4 unified-table work does its own typing).

Format per zone matches docs/design/extend-geoai-climate.md §2.1:
    {"type":"Feature","properties":{"zone_id","name","koppen","bbox","role","area_km2"},
     "geometry":{"type":"Polygon","coordinates":[...]}}

NOTE — accuracy & ownership: the 12 panel bboxes are 1deg representative boxes and the Koppen
classes are best-known representative values, pending geomatics validation. The 3 pilot
geometries are no longer approximate: they are the authoritative IGN ADMIN EXPRESS / Sandre
BD TOPAGE boundaries, vendored under `zones/sources/` so this generator stays offline and
deterministic (the "ou pre-telecharge" path of docs/design/extend-geoai-climate.md §2.1).
Refresh them with `scripts/refresh_pilot_boundaries.py`; see docs/design/b1-pilot-boundaries.md.
"""

from __future__ import annotations

import json
from pathlib import Path

from shapely.geometry import box, mapping, shape

# Panel global — (zone_id, name, center_lon, center_lat, koppen). 1deg x 1deg box.
PANEL: list[tuple[str, str, float, float, str]] = [
    ("montpellier", "Montpellier", 3.5, 43.6, "Csa"),  # Mediterranean, hot dry summer
    ("marseille", "Marseille", 5.4, 43.3, "Csa"),  # Mediterranean
    ("nairobi", "Nairobi", 36.8, -1.3, "Cwb"),  # subtropical highland
    ("mombasa", "Mombasa", 39.6, -4.0, "As"),  # tropical savanna, dry summer
    ("niamey", "Niamey", 2.1, 13.5, "BSh"),  # hot semi-arid
    ("rennes", "Rennes", -1.7, 48.1, "Cfb"),  # oceanic
    ("tromso", "Tromso", 18.9, 69.6, "Dfc"),  # subarctic
    ("svalbard", "Svalbard (Longyearbyen)", 15.6, 78.2, "ET"),  # tundra
    ("manaus", "Manaus", -60.0, -3.1, "Am"),  # tropical monsoon (pronounced Jun-Sep dry season)
    (
        "kerguelen",
        "Kerguelen",
        70.2,
        -49.35,
        "ET",
    ),  # subantarctic tundra (box covers Port-aux-Francais station)
    ("riyadh", "Riyadh", 46.7, 24.7, "BWh"),  # hot desert
    ("seoul", "Seoul", 127.0, 37.5, "Dwa"),  # monsoon humid continental
]

# Pilotes France — (zone_id, koppen). Name, geometry and area come from the vendored
# boundary under zones/sources/, refreshed by scripts/refresh_pilot_boundaries.py.
PILOTS: list[tuple[str, str]] = [
    ("montpellier_metropole", "Csa"),
    ("bassin_du_lez", "Csa"),
    ("fos_sur_mer", "Csa"),
]

# Client sites — (zone_id, name, min_lon, min_lat, max_lon, max_lat, koppen). A tenant-scoped box
# around one watched site, never part of the scientific 12 panel + 3 pilot zones
# (docs/design/b1-pilot-boundaries.md, "Client sites are not pilot zones"). The Köppen field is
# here only because tests/test_zones.py requires one on every feature. Empty until a site is agreed.
CLIENTS: list[tuple[str, str, float, float, float, float, str]] = []

# Per-zone split folder for each role. An unknown role raises KeyError instead of being filed
# as a panel zone.
ROLE_DIRS = {"panel": "panel_global", "pilot": "pilot_france", "client": "client_sites"}

SOURCES_DIR = Path(__file__).resolve().parent.parent / "zones" / "sources"

# Provenance keys carried from the vendored boundary onto the emitted pilot feature, so a
# consumer of zones.geojson alone can still honour the LOv2 attribution requirement.
_PROVENANCE_KEYS = ("source", "dataset", "licence", "attribution", "definition")


def _feature(zone_id: str, name: str, koppen: str, geom, role: str, extra: dict | None) -> dict:
    minx, miny, maxx, maxy = geom.bounds
    props = {
        "zone_id": zone_id,
        "name": name,
        "koppen": koppen,
        "bbox": [round(minx, 4), round(miny, 4), round(maxx, 4), round(maxy, 4)],
        "role": role,
    }
    if extra:
        props.update(extra)
    return {"type": "Feature", "properties": props, "geometry": mapping(geom)}


def load_pilot_boundary(zone_id: str, sources_dir: Path | None = None) -> dict:
    """Read one vendored pilot boundary. Offline by construction — no network here."""
    path = (sources_dir or SOURCES_DIR) / f"{zone_id}.geojson"
    if not path.is_file():
        raise FileNotFoundError(
            f"missing vendored boundary {path}; run scripts/refresh_pilot_boundaries.py"
        )
    return json.loads(path.read_text())


def build_features(sources_dir: Path | None = None) -> list[dict]:
    feats: list[dict] = []
    for zone_id, name, lon, lat, koppen in PANEL:
        geom = box(lon - 0.5, lat - 0.5, lon + 0.5, lat + 0.5)
        feats.append(_feature(zone_id, name, koppen, geom, "panel", None))
    for zone_id, koppen in PILOTS:
        source = load_pilot_boundary(zone_id, sources_dir)
        props = source["properties"]
        extra = {"area_km2": props["area_km2"]}
        extra.update({k: props[k] for k in _PROVENANCE_KEYS if k in props})
        feats.append(
            _feature(zone_id, props["name"], koppen, shape(source["geometry"]), "pilot", extra)
        )
    for zone_id, name, min_lon, min_lat, max_lon, max_lat, koppen in CLIENTS:
        geom = box(min_lon, min_lat, max_lon, max_lat)
        feats.append(_feature(zone_id, name, koppen, geom, "client", None))
    return feats


def _crs_member() -> dict:
    # GeoJSON named-CRS member (RFC 7946 deprecates this, but it makes the CRS explicit
    # for downstream tooling, as the FIERCE DoD requires "CRS specified").
    return {"type": "name", "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"}}


def _dumps(payload: dict, geometry_key: str) -> str:
    """Readable properties, compact coordinates.

    The pilot boundaries carry thousands of positions; `indent=2` puts each number on its own
    line and inflates the artifacts ~2.6x for no benefit. Same helper as in
    scripts/refresh_pilot_boundaries.py.
    """
    placeholders: dict[str, dict] = {}

    def _stash(geometry: dict) -> str:
        token = f"__GEOM_{len(placeholders)}__"
        placeholders[token] = geometry
        return token

    shell_source = dict(payload)
    if geometry_key == "features":
        shell_source["features"] = [
            {**f, "geometry": _stash(f["geometry"])} for f in payload["features"]
        ]
    else:
        shell_source["geometry"] = _stash(payload["geometry"])
    shell = json.dumps(shell_source, indent=2, ensure_ascii=False)
    for token, geometry in placeholders.items():
        compact = json.dumps(geometry, ensure_ascii=False, separators=(",", ":"))
        shell = shell.replace(f'"{token}"', compact)
    return shell + "\n"


def write_geojson(feats: list[dict], out_dir: Path) -> None:
    for f in feats:
        sub = ROLE_DIRS[f["properties"]["role"]]
        d = out_dir / sub
        d.mkdir(parents=True, exist_ok=True)
        single = {"type": "Feature", "crs": _crs_member(), **f}
        (d / f"{f['properties']['zone_id']}.geojson").write_text(_dumps(single, "geometry"))
    fc = {
        "type": "FeatureCollection",
        "name": "fierce_zones",
        "crs": _crs_member(),
        "features": feats,
    }
    (out_dir / "zones.geojson").write_text(_dumps(fc, "features"))


def write_geoparquet(feats: list[dict], out_dir: Path) -> str:
    """Write a DuckDB-spatial-readable Parquet (geometry as WKB) — no GDAL driver needed.

    Read it back with: ``SELECT *, ST_GeomFromWKB(geometry) FROM read_parquet('zones.parquet')``.
    This is what `spatial_query` / the unified-table work (B4) consume. Returns a status.
    """
    try:
        import duckdb
        from shapely.geometry import shape
    except Exception as e:  # pragma: no cover - core deps, stay defensive
        return f"skipped (import failed: {e})"
    dst = (out_dir / "zones.parquet").as_posix()
    rows = []
    for f in feats:
        p = f["properties"]
        rows.append(
            (
                p["zone_id"],
                p["name"],
                p["koppen"],
                p["role"],
                p.get("area_km2"),
                shape(f["geometry"]).wkt,
            )
        )
    try:
        con = duckdb.connect()
        # one statement per execute() — DuckDB's Python execute() does not reliably run
        # multiple semicolon-separated statements across versions.
        con.execute("INSTALL spatial;")
        con.execute("LOAD spatial;")
        con.execute(
            "CREATE TABLE z (zone_id VARCHAR, name VARCHAR, koppen VARCHAR, "
            # DOUBLE, not INTEGER: pilot areas are geodesic decimals (589.3886), and an INTEGER
            # staging column silently rounds them, so zones.parquet would disagree with
            # zones.geojson and zones/sources. Guarded by test_parquet_areas_match_geojson.
            "role VARCHAR, area_km2 DOUBLE, wkt VARCHAR)"
        )
        con.executemany("INSERT INTO z VALUES (?,?,?,?,?,?)", rows)
        con.execute(
            f"COPY (SELECT zone_id, name, koppen, role, area_km2, "
            f"ST_AsWKB(ST_GeomFromText(wkt)) AS geometry FROM z) "
            f"TO '{dst}' (FORMAT PARQUET);"
        )
        con.close()
        return "ok (WKB geometry)"
    except Exception as e:
        return f"skipped (Parquet write failed: {e})"


def main() -> int:
    out_dir = Path(__file__).resolve().parent.parent / "zones"
    feats = build_features()
    write_geojson(feats, out_dir)
    gp = write_geoparquet(feats, out_dir)
    panel = sum(1 for f in feats if f["properties"]["role"] == "panel")
    pilot = sum(1 for f in feats if f["properties"]["role"] == "pilot")
    client = sum(1 for f in feats if f["properties"]["role"] == "client")
    koppen = sorted({f["properties"]["koppen"] for f in feats})
    print(
        f"Wrote {len(feats)} zones to {out_dir}/ ({panel} panel + {pilot} pilot + {client} client)"
    )
    print(f"  Koppen spread: {', '.join(koppen)}")
    print(f"  GeoParquet: {gp}")
    if not gp.startswith("ok"):
        print("ERROR: zones.parquet was not (re)generated — the B1 artifact set is incomplete.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
