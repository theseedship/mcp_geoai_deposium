#!/usr/bin/env python3
"""Refresh the three FIERCE French pilot boundaries from authoritative sources (DoD lot B1, #25).

This script is **network-bound** and deliberately separate from ``scripts/generate_zones.py``.
The generator must stay offline, deterministic and network-free (its contract tests run with no
network), so it consumes the *vendored* output of this script rather than fetching anything
itself — the "ou pre-telecharge" path of docs/design/extend-geoai-climate.md §2.1.

    scripts/refresh_pilot_boundaries.py   ->  zones/sources/{zone_id}.geojson   (network, rare)
    scripts/generate_zones.py             ->  zones/**                          (offline, idempotent)

Sources — all Licence Ouverte / Open Licence 2.0 (Etalab), i.e. attribution-only and therefore
compatible with this MIT repository:

    montpellier_metropole  IGN ADMIN EXPRESS COG, `epci` layer (Geoplateforme WFS)
    fos_sur_mer            IGN ADMIN EXPRESS COG, `commune` layer (Geoplateforme WFS)
    bassin_du_lez          Sandre BD TOPAGE(R) 2026, bassins versants topographiques

NOTE — do NOT re-source the two admin boundaries from geo.api.gouv.fr. It republishes the same
IGN geometries under the data.gouv.fr "Contours administratifs" dataset, which is declared
**ODbL** (share-alike). Fetching from IGN directly keeps the licence unambiguously LOv2.

NOTE — `bassin_du_lez` is the topographic catchment **at the Lez outlet**, i.e. the 7 BD TOPAGE
units draining to the sea at Palavas: the 2 Lez units plus the Mosson (4) and the Coulazou (1).
The Mosson is a left-bank tributary joining the Lez at Lattes, so it is part of the catchment;
official sources frame the Lez sub-basin as ~30 % of the whole and the Mosson as ~70 %.
The narrower "Lez sensu stricto" reading (2 units, ~193 km2) and the wider SAGE
"Lez, Mosson, Etangs Palavasiens" management perimeter (~637 km2) are both defensible but
different objects; see docs/design/b1-pilot-boundaries.md.
"""

from __future__ import annotations

import argparse
import json
import urllib.parse
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from shapely.geometry import mapping, shape
from shapely.geometry.polygon import orient
from shapely.ops import unary_union

IGN_WFS = "https://data.geopf.fr/wfs/ows"
SANDRE_TOPAGE_WFS = "https://services.sandre.eaufrance.fr/geo/topage2026"

LOV2 = "Licence Ouverte / Open Licence 2.0 (Etalab)"

# The 7 BD TOPAGE elementary catchments draining to the Lez outlet: 2 Lez + 4 Mosson + 1 Coulazou.
LEZ_CDOH = (
    "06B0000002150456086",  # Le Lez de sa source a la Lironde
    "06B0000002150456009",  # Le Lez de la Lironde incluse la mer Mediterranee
    "06B0000002150456066",  # La Mosson de sa source au ruisseau de Miege Sole inclus
    "06B0000002150458403",  # La Mosson du ruisseau de Miege Sole au ruisseau de Lasse deron
    "06B0000002150458401",  # La Mosson du ruisseau de Lasse deron inclus au ruisseau du Coulazou
    "06B0000002150455994",  # Ruisseau du Coulazou
    "06B0000002150456007",  # La Mosson du ruisseau du Coulazou au Lez
)

COORD_DECIMALS = 6  # ~11 cm at this latitude — far below the zonal-statistics grain


def _get(url: str, params: dict[str, str], *, timeout: float) -> dict[str, Any]:
    full = f"{url}?{urllib.parse.urlencode(params)}"
    with urllib.request.urlopen(full, timeout=timeout) as resp:  # noqa: S310 - pinned https hosts
        if resp.status != 200:
            raise RuntimeError(f"{full} -> HTTP {resp.status}")
        payload = json.loads(resp.read())
    if not isinstance(payload, dict) or "features" not in payload:
        raise RuntimeError(f"{full} -> not a GeoJSON FeatureCollection")
    return payload


def _ign_wfs(layer: str, cql: str, *, timeout: float) -> dict[str, Any]:
    return _get(
        IGN_WFS,
        {
            "SERVICE": "WFS",
            "VERSION": "2.0.0",
            "REQUEST": "GetFeature",
            "TYPENAMES": layer,
            "CQL_FILTER": cql,
            "SRSNAME": "EPSG:4326",
            "OUTPUTFORMAT": "application/json",
            "COUNT": "10",
        },
        timeout=timeout,
    )


def _fes_or_equals(prop: str, values: tuple[str, ...]) -> str:
    clauses = "".join(
        f"<fes:PropertyIsEqualTo><fes:ValueReference>{prop}</fes:ValueReference>"
        f"<fes:Literal>{v}</fes:Literal></fes:PropertyIsEqualTo>"
        for v in values
    )
    return (
        '<fes:Filter xmlns:fes="http://www.opengis.net/fes/2.0">'
        f"<fes:Or>{clauses}</fes:Or>"
        "</fes:Filter>"
    )


def _normalize(geom) -> Any:
    """Dissolve a unit list into one shape and orient it RFC 7946-style (CCW exteriors).

    A MultiPolygon carrying a single part is collapsed to a Polygon: IGN serves every commune
    as a MultiPolygon, so the wrapper is a serialization artefact, not a topological fact. A
    genuinely multi-part boundary (Montpellier Metropole has a detached islet) stays multi-part.
    """
    merged = unary_union(geom) if isinstance(geom, list) else geom
    if merged.geom_type == "MultiPolygon" and len(merged.geoms) == 1:
        merged = merged.geoms[0]
    if merged.geom_type == "Polygon":
        return orient(merged, 1.0)
    if merged.geom_type == "MultiPolygon":
        return type(merged)([orient(part, 1.0) for part in merged.geoms])
    raise RuntimeError(f"unexpected geometry type {merged.geom_type}")


def dumps_feature(feature: dict[str, Any]) -> str:
    """Readable properties, compact coordinates.

    A real boundary carries thousands of positions; `indent=2` puts each number on its own
    line and inflates the artifact ~2.6x for no benefit. Keep the envelope pretty-printed and
    the coordinate array on one line. Same helper in scripts/generate_zones.py.
    """
    placeholder = "__GEOMETRY__"
    shell = json.dumps({**feature, "geometry": placeholder}, indent=2, ensure_ascii=False)
    compact = json.dumps(feature["geometry"], ensure_ascii=False, separators=(",", ":"))
    return shell.replace(f'"{placeholder}"', compact) + "\n"


def _round_coords(obj):
    if isinstance(obj, float):
        return round(obj, COORD_DECIMALS)
    if isinstance(obj, list | tuple):  # shapely's mapping() emits tuples, not lists
        return [_round_coords(v) for v in obj]
    return obj


def _area_km2(geom) -> float:
    """Geodesic area on the WGS84 spheroid, via DuckDB spatial (ST_Area_Spheroid).

    ``ST_Area_Spheroid`` reads X as **latitude** and Y as longitude, so a lon/lat geometry
    must be flipped first — otherwise every area is computed as if it sat on the equator
    (a 1x1 deg box at 43.6N returns 12,287 km2 instead of 8,931).
    """
    import duckdb

    con = duckdb.connect()
    try:
        con.execute("INSTALL spatial;")
        con.execute("LOAD spatial;")
        value = con.execute(
            "SELECT ST_Area_Spheroid(ST_FlipCoordinates(ST_GeomFromText(?))) / 1e6", [geom.wkt]
        ).fetchone()[0]
    finally:
        con.close()
    return round(float(value), 4)


def fetch_montpellier_metropole(*, timeout: float) -> tuple[Any, dict[str, Any]]:
    layer = "ADMINEXPRESS-COG.LATEST:epci"
    cql = "code_siren='243400017'"
    fc = _ign_wfs(layer, cql, timeout=timeout)
    features = fc["features"]
    if len(features) != 1:
        raise RuntimeError(f"expected exactly 1 EPCI, got {len(features)}")
    props = features[0]["properties"]
    members = [c for c in (props.get("codes_insee_des_communes_membres") or "").split("/") if c]
    if len(members) != 31:
        raise RuntimeError(f"expected 31 member communes, got {len(members)}")
    meta = {
        "source": "IGN ADMIN EXPRESS COG",
        "source_url": f"{IGN_WFS}?TYPENAMES={layer}&CQL_FILTER={cql}",
        "dataset": layer,
        "licence": LOV2,
        "attribution": "IGN - ADMIN EXPRESS COG",
        "definition": (
            "EPCI perimeter of Montpellier Mediterranee Metropole (SIREN 243400017), "
            "31 member communes"
        ),
        "member_ids": members,
        "official_name": props.get("nom_officiel"),
    }
    return shape(features[0]["geometry"]), meta


def fetch_fos_sur_mer(*, timeout: float) -> tuple[Any, dict[str, Any]]:
    layer = "ADMINEXPRESS-COG.LATEST:commune"
    cql = "code_insee='13039'"
    fc = _ign_wfs(layer, cql, timeout=timeout)
    features = fc["features"]
    if len(features) != 1:
        raise RuntimeError(f"expected exactly 1 commune, got {len(features)}")
    props = features[0]["properties"]
    meta = {
        "source": "IGN ADMIN EXPRESS COG",
        "source_url": f"{IGN_WFS}?TYPENAMES={layer}&CQL_FILTER={cql}",
        "dataset": layer,
        "licence": LOV2,
        "attribution": "IGN - ADMIN EXPRESS COG",
        "definition": "Commune of Fos-sur-Mer (INSEE 13039)",
        "member_ids": ["13039"],
        "official_name": props.get("nom_officiel"),
    }
    return shape(features[0]["geometry"]), meta


def fetch_bassin_du_lez(*, timeout: float) -> tuple[Any, dict[str, Any]]:
    layer = "sa:BassinVersantTopographique_FXX_Topage2026"
    fc = _get(
        SANDRE_TOPAGE_WFS,
        {
            "SERVICE": "WFS",
            "VERSION": "2.0.0",
            "REQUEST": "GetFeature",
            "TYPENAMES": layer,
            "SRSNAME": "EPSG:4326",
            "OUTPUTFORMAT": "application/json; subtype=geojson",
            "FILTER": _fes_or_equals("CdOH", LEZ_CDOH),
        },
        timeout=timeout,
    )
    features = fc["features"]
    if len(features) != len(LEZ_CDOH):
        raise RuntimeError(f"expected {len(LEZ_CDOH)} BD TOPAGE units, got {len(features)}")
    got = {f["properties"]["CdOH"] for f in features}
    if got != set(LEZ_CDOH):
        raise RuntimeError(f"CdOH mismatch: missing {set(LEZ_CDOH) - got}")
    meta = {
        "source": "Sandre BD TOPAGE(R) 2026",
        "source_url": f"{SANDRE_TOPAGE_WFS}?TYPENAMES={layer}&FILTER=CdOH in {LEZ_CDOH}",
        "dataset": layer,
        "licence": LOV2,
        "attribution": "BD TOPAGE(R) - Sandre / OFB / IGN",
        "definition": (
            "Topographic catchment at the Lez outlet: 7 BD TOPAGE elementary units "
            "(Lez x2 + Mosson x4 + Coulazou x1) draining to the sea at Palavas"
        ),
        "member_ids": list(LEZ_CDOH),
        "official_name": "Bassin versant topographique du Lez a l'exutoire",
    }
    return [shape(f["geometry"]) for f in features], meta


FETCHERS = {
    "montpellier_metropole": ("Montpellier Metropole", fetch_montpellier_metropole),
    "bassin_du_lez": ("Bassin du Lez", fetch_bassin_du_lez),
    "fos_sur_mer": ("Fos-sur-Mer", fetch_fos_sur_mer),
}


def refresh(zone_id: str, out_dir: Path, *, timeout: float) -> dict[str, Any]:
    name, fetcher = FETCHERS[zone_id]
    raw, meta = fetcher(timeout=timeout)
    geometry = _round_coords(mapping(_normalize(raw)))
    # rounding can in principle pinch a ring; refuse to vendor an invalid boundary
    rounded = shape(geometry)
    if not rounded.is_valid:
        raise RuntimeError(f"{zone_id}: geometry invalid after rounding to {COORD_DECIMALS} dp")
    feature = {
        "type": "Feature",
        "crs": {"type": "name", "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"}},
        "properties": {
            "zone_id": zone_id,
            "name": name,
            "area_km2": _area_km2(rounded),
            "retrieved_at": datetime.now(UTC).strftime("%Y-%m-%d"),
            **meta,
        },
        "geometry": geometry,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{zone_id}.geojson").write_text(dumps_feature(feature))
    return feature


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Refresh the vendored FIERCE French pilot boundaries (B1, #25)."
    )
    parser.add_argument(
        "--zone",
        action="append",
        choices=sorted(FETCHERS),
        help="refresh only this zone (repeatable); default is all three",
    )
    parser.add_argument("--timeout", type=float, default=120.0, help="per-request timeout, seconds")
    parser.add_argument("--json", action="store_true", help="print machine-readable JSON")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    out_dir = Path(__file__).resolve().parent.parent / "zones" / "sources"
    zones = args.zone or sorted(FETCHERS)
    summary = []
    for zone_id in zones:
        feature = refresh(zone_id, out_dir, timeout=args.timeout)
        props = feature["properties"]
        summary.append(
            {
                "zone_id": zone_id,
                "geom_type": feature["geometry"]["type"],
                "area_km2": props["area_km2"],
                "source": props["source"],
                "licence": props["licence"],
            }
        )
    if args.json:
        print(json.dumps({"ok": True, "zones": summary}, indent=2, ensure_ascii=False))
    else:
        for row in summary:
            print(
                f"{row['zone_id']:<24} {row['geom_type']:<13} "
                f"{row['area_km2']:>10.2f} km2  {row['source']}"
            )
        print(f"\nWrote {len(summary)} boundaries to {out_dir}/")
        print("Now re-run scripts/generate_zones.py to rebuild the zones/ artifacts.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
