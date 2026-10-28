"""Bounded, offline readback of a verified O3 handoff; no scientific revalidation."""

from __future__ import annotations

import hashlib
import html
import json
import os
import re
import shutil
import stat
import tempfile
import zipfile
import zlib
from datetime import date
from io import BytesIO
from pathlib import Path, PurePosixPath
from typing import Any

import duckdb
from pydantic import ValidationError

from src.patterns.library_schema import GradualPatternRecord, PatternLibrary

MAX_ARCHIVE_BYTES = 100 * 1024 * 1024
MAX_ENTRIES = 1000
MAX_JSON_BYTES = 1024 * 1024
MAX_LIBRARY_ROWS = 10_000
OUTPUT_KEYS = {
    f"climate/o3/holdout/q3_2023/{algorithm}_validated.parquet": algorithm
    for algorithm in ("graank", "tgrad")
}
REPORT_FILES = ("report.json", "report.md", "report.html")
CAVEATS = [
    "Lecture offline et validation structurelle uniquement ; aucune revalidation scientifique.",
    "Une confiance null signifie non renseignée, jamais zéro ni une confiance calibrée.",
    "Kerguelen : exclusion historique du mining O3 pour lignes Q3 insuffisantes.",
    "Svalbard juillet 2023 : avertissement historique de troncature STAC conservé.",
    "Les affectations des folds et la correspondance zone–Köppen sont absentes du ZIP ; la jointure externe éventuelle est identifiée séparément.",
    "Les données brutes ERA5/images et le bundle historique de forecast ne sont pas fournis.",
    "Registre historique Q3 : géométries des trois pilotes anciennes et chevauchement avec le panel.",
    "Labels manaus/nairobi/niamey sous revue #194 ; folds non stratifiés par Köppen.",
    "Une couverture descriptive par classe ne prouve pas une généralisation held-out.",
]


class OfflineBundleError(ValueError):
    """An input or output was refused without running a provider or miner."""


def _unique_json_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise OfflineBundleError(f"Clé JSON dupliquée : {key}")
        result[key] = value
    return result


def _strict_json(data: bytes) -> Any:
    if len(data) > MAX_JSON_BYTES:
        raise OfflineBundleError("JSON dépasse la limite de taille")
    try:
        result = json.loads(data, object_pairs_hook=_unique_json_pairs)
        # Also refuses non-standard NaN/Infinity at any nesting depth.
        json.dumps(result, allow_nan=False)
        return result
    except (ValueError, UnicodeError, RecursionError) as exc:
        raise OfflineBundleError(f"JSON invalide : {exc}") from exc


def _safe_member_name(name: str) -> str:
    path = PurePosixPath(name)
    if (
        not name
        or "\\" in name
        or ":" in name
        or path.is_absolute()
        or any(part in ("", ".", "..") for part in name.split("/"))
        or str(path) != name
    ):
        raise OfflineBundleError(f"Chemin ZIP non sûr : {name!r}")
    return name


def _verify_bundle(archive: Path, expected_sha256: str) -> tuple[dict, dict, dict, int]:
    if not re.fullmatch(r"[a-fA-F0-9]{64}", expected_sha256):
        raise OfflineBundleError("SHA-256 explicite requis : 64 caractères hexadécimaux")
    if not archive.is_file() or archive.stat().st_size > MAX_ARCHIVE_BYTES:
        raise OfflineBundleError("Archive absente ou supérieure à 100 MiB")
    with archive.open("rb") as source:
        # Hash and parse the exact same bounded snapshot, even if the file is rewritten.
        snapshot = source.read(MAX_ARCHIVE_BYTES + 1)
        archive_bytes = len(snapshot)
        if archive_bytes > MAX_ARCHIVE_BYTES:
            raise OfflineBundleError("Archive supérieure à 100 MiB")
        digest = hashlib.sha256(snapshot).hexdigest()
        if digest != expected_sha256.lower():
            raise OfflineBundleError("SHA-256 de l'archive différent de la valeur attendue")
        with zipfile.ZipFile(BytesIO(snapshot)) as bundle:
            entries = bundle.infolist()
            if len(entries) > MAX_ENTRIES or sum(e.file_size for e in entries) > MAX_ARCHIVE_BYTES:
                raise OfflineBundleError(
                    "ZIP dépasse les limites d'entrées ou de taille décompressée"
                )
            members = {}
            for entry in entries:
                name = _safe_member_name(entry.filename)
                if name in members:
                    raise OfflineBundleError(f"Membre ZIP dupliqué : {name}")
                if stat.S_ISLNK(entry.external_attr >> 16) or entry.is_dir() or entry.flag_bits & 1:
                    raise OfflineBundleError(f"Membre ZIP non régulier : {name}")
                members[name] = entry
            if "bundle.json" not in members or "README-O3-HANDOFF.md" not in members:
                raise OfflineBundleError("Manifest bundle.json ou README manquant")
            if members["bundle.json"].file_size > MAX_JSON_BYTES:
                raise OfflineBundleError("Manifest trop volumineux")
            manifest = _strict_json(bundle.read("bundle.json"))
            if (
                not isinstance(manifest, dict)
                or manifest.get("schema_version") != "geoai-o3-offline-handoff/v1"
            ):
                raise OfflineBundleError("Version/enveloppe du manifest non prise en charge")
            required = {
                "schema_version",
                "generated_at",
                "historical_producers",
                "verification",
                "inventory",
                "objects",
                "caveats",
            }
            if set(manifest) != required:
                raise OfflineBundleError("Champs de l'enveloppe du manifest invalides")
            if (
                not isinstance(manifest["generated_at"], str)
                or not isinstance(manifest["historical_producers"], dict)
                or not isinstance(manifest["verification"], dict)
                or not isinstance(manifest["caveats"], list)
                or any(not isinstance(c, str) for c in manifest["caveats"])
            ):
                raise OfflineBundleError("Métadonnées du manifest invalides")
            objects = manifest["objects"]
            inventory = manifest["inventory"]
            if not isinstance(objects, list) or not isinstance(inventory, dict):
                raise OfflineBundleError("Inventaire invalide")
            keys = set()
            declared_total = 0
            outputs = {}
            input_manifests = {}
            for obj in objects:
                if (
                    not isinstance(obj, dict)
                    or not {"key", "role", "bytes", "sha256"} <= obj.keys()
                ):
                    raise OfflineBundleError("Objet du manifest incomplet (hash requis)")
                key = obj["key"]
                if not isinstance(key, str):
                    raise OfflineBundleError("Clé d'objet invalide")
                _safe_member_name(key)
                if key in keys:
                    raise OfflineBundleError(f"Objet du manifest dupliqué : {key}")
                keys.add(key)
                size = obj["bytes"]
                sha = obj["sha256"]
                if type(size) is not int or not 0 <= size <= MAX_ARCHIVE_BYTES:
                    raise OfflineBundleError(f"Longueur d'objet invalide : {key}")
                if not isinstance(sha, str) or not re.fullmatch(r"[a-fA-F0-9]{64}", sha):
                    raise OfflineBundleError(f"Hash d'objet absent/invalide : {key}")
                role = obj["role"]
                valid_role = (
                    (role == "o3_output" and key in OUTPUT_KEYS)
                    or (
                        role == "o3_input_unified"
                        and re.fullmatch(r"climate/holdout/unified/\d{4}_\d{2}/[^/]+\.parquet", key)
                    )
                    or (
                        role == "b4_input_manifest"
                        and re.fullmatch(r"climate/runs/[^/]+/manifest\.json", key)
                    )
                )
                if not valid_role:
                    raise OfflineBundleError(f"Rôle/chemin non pris en charge : {key}")
                name = f"objects/{key}"
                if name not in members or members[name].file_size != size:
                    raise OfflineBundleError(f"Membre absent ou longueur différente : {key}")
                declared_total += size
                hasher = hashlib.sha256()
                actual_size = 0
                retained = bytearray()
                with bundle.open(name) as member:
                    while chunk := member.read(64 * 1024):
                        hasher.update(chunk)
                        actual_size += len(chunk)
                        if role in ("o3_output", "b4_input_manifest"):
                            retained.extend(chunk)
                if actual_size != size or hasher.hexdigest() != sha.lower():
                    raise OfflineBundleError(f"Intégrité d'objet invalide : {key}")
                if role == "o3_output":
                    outputs[key] = bytes(retained)
                elif role == "b4_input_manifest":
                    input_manifests[key] = bytes(retained)
            if set(outputs) != set(OUTPUT_KEYS):
                raise OfflineBundleError("Les deux bibliothèques graank/tgrad sont requises")
            expected_members = {"bundle.json", "README-O3-HANDOFF.md"} | {
                f"objects/{k}" for k in keys
            }
            if set(members) != expected_members:
                raise OfflineBundleError("Membre ZIP non déclaré dans le manifest")
            if (
                type(inventory.get("object_count")) is not int
                or inventory["object_count"] != len(objects)
                or type(inventory.get("total_source_bytes")) is not int
                or inventory["total_source_bytes"] != declared_total
            ):
                raise OfflineBundleError("Comptages de l'inventaire différents des objets")
            if "role_counts" in inventory:
                actual_roles = {
                    role: sum(obj["role"] == role for obj in objects)
                    for role in {obj["role"] for obj in objects}
                }
                declared_roles = inventory["role_counts"]
                if (
                    not isinstance(declared_roles, dict)
                    or any(type(n) is not int for n in declared_roles.values())
                    or declared_roles != actual_roles
                ):
                    raise OfflineBundleError("Comptages de rôles invalides dans l'inventaire")
            limits = inventory.get("limits", {})
            if not isinstance(limits, dict) or any(
                type(n) is not int or n <= 0 for n in limits.values()
            ):
                raise OfflineBundleError("Limites de l'inventaire invalides")
            if (
                len(objects) > limits.get("max_objects", MAX_ENTRIES)
                or declared_total > limits.get("max_total_bytes", MAX_ARCHIVE_BYTES)
                or any(
                    obj["bytes"] > limits.get("max_object_bytes", MAX_ARCHIVE_BYTES)
                    for obj in objects
                )
            ):
                raise OfflineBundleError("Objets supérieurs aux limites déclarées du manifest")
            # Every object's bytes/hash have passed before these inputs are interpreted.
            input_manifests = {k: _strict_json(v) for k, v in sorted(input_manifests.items())}
            return manifest, outputs, input_manifests, archive_bytes


def _read_library(path: Path, algorithm: str) -> PatternLibrary:
    with duckdb.connect(
        config={
            "autoinstall_known_extensions": "false",
            "autoload_known_extensions": "false",
            "memory_limit": "256MB",
            "threads": "1",
        }
    ) as connection:
        cursor = connection.execute(
            "SELECT * FROM read_parquet(?) LIMIT ?", [str(path), MAX_LIBRARY_ROWS + 1]
        )
        columns = [column[0] for column in cursor.description]
        if set(columns) != set(GradualPatternRecord.model_fields):
            raise OfflineBundleError("Colonnes Parquet différentes du contrat PatternLibrary")
        rows = cursor.fetchall()
    if len(rows) > MAX_LIBRARY_ROWS:
        raise OfflineBundleError("Bibliothèque supérieure à la limite de patterns")
    records = []
    for row in rows:
        record = dict(zip(columns, row, strict=True))
        for name in ("variables", "zones_validated", "extra_metrics"):
            if isinstance(record[name], str):
                record[name] = _strict_json(record[name].encode("utf-8"))
        records.append(record)
    library = PatternLibrary.model_validate(
        {
            "zone_id": records[0]["zone_id"] if records else "cross_zone",
            "algorithm": algorithm,
            "patterns": records,
        }
    )
    _json_bytes(library.model_dump(mode="json"))
    return library


def _manifest_coverage(input_manifests: dict) -> list[dict]:
    coverage = []
    for key, manifest in sorted(input_manifests.items()):
        if (
            not isinstance(manifest, dict)
            or manifest.get("schema_version") != 1
            or manifest.get("kind") != "geoai_run_manifest"
        ):
            raise OfflineBundleError(f"Manifest B4 non pris en charge : {key}")
        parameters = manifest.get("parameters")
        outputs = manifest.get("outputs")
        if not isinstance(parameters, dict) or not isinstance(outputs, list):
            raise OfflineBundleError(f"Couverture du manifest B4 invalide : {key}")
        start, end = parameters.get("start_date"), parameters.get("end_date")
        if (
            not isinstance(start, str)
            or not isinstance(end, str)
            or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", start)
            or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", end)
        ):
            raise OfflineBundleError(f"Période B4 absente/invalide : {key}")
        try:
            if date.fromisoformat(start) > date.fromisoformat(end):
                raise ValueError("inverted period")
        except ValueError as exc:
            raise OfflineBundleError(f"Période B4 invalide : {key}") from exc
        seen = set()
        for output in outputs:
            if (
                not isinstance(output, dict)
                or output.get("kind") != "unified"
                or not isinstance(output.get("zone_id"), str)
                or type(output.get("n_matched")) is not int
                or output["n_matched"] < 0
            ):
                raise OfflineBundleError(f"Lignes rapportées par le manifest invalides : {key}")
            if not output["zone_id"] or output["zone_id"] in seen:
                raise OfflineBundleError(f"Zone B4 vide/dupliquée : {key}")
            seen.add(output["zone_id"])
            coverage.append(
                {
                    "manifest_key": key,
                    "zone_id": output["zone_id"],
                    "period_start": parameters.get("start_date"),
                    "period_end": parameters.get("end_date"),
                    "n_matched": output["n_matched"],
                    "evidence": "manifest-reported rows; inputs not reread",
                }
            )
    return coverage


def _join_zones(zones_geojson: str | Path, input_manifests: dict, patterns: list[dict]) -> dict:
    path = Path(zones_geojson)
    if not path.is_file() or path.stat().st_size > MAX_JSON_BYTES:
        raise OfflineBundleError("GeoJSON des zones absent ou trop volumineux")
    data = path.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if not input_manifests:
        raise OfflineBundleError("Jointure zones impossible sans manifests B4")
    selected = None
    for manifest in input_manifests.values():
        artifact = manifest.get("zone_artifact")
        if not isinstance(artifact, dict) or artifact.get("sha256") != digest:
            raise OfflineBundleError("SHA-256 zones différent d'un manifest B4")
        ids = artifact.get("selected_zone_ids")
        if (
            not isinstance(ids, list)
            or any(not isinstance(v, str) or not v for v in ids)
            or len(set(ids)) != len(ids)
            or type(artifact.get("n_selected")) is not int
            or artifact["n_selected"] != len(ids)
        ):
            raise OfflineBundleError("Sélection des zones B4 invalide")
        if selected is not None and selected != set(ids):
            raise OfflineBundleError("Sélections des zones différentes entre manifests B4")
        selected = set(ids)
    geojson = _strict_json(data)
    if (
        not isinstance(geojson, dict)
        or geojson.get("type") != "FeatureCollection"
        or not isinstance(geojson.get("features"), list)
    ):
        raise OfflineBundleError("GeoJSON des zones invalide")
    zones = {}
    for feature in geojson["features"]:
        props = feature.get("properties") if isinstance(feature, dict) else None
        if (
            not isinstance(props, dict)
            or not isinstance(props.get("zone_id"), str)
            or not isinstance(props.get("koppen"), str)
            or not props["koppen"]
        ):
            raise OfflineBundleError("Identifiant/classe Köppen manquant dans le GeoJSON")
        if props["zone_id"] in zones:
            raise OfflineBundleError("Identifiant de zone GeoJSON dupliqué")
        zones[props["zone_id"]] = props["koppen"]
    if selected is None or not selected <= zones.keys():
        raise OfflineBundleError("Zones sélectionnées absentes du GeoJSON")
    validated = {zone for pattern in patterns for zone in pattern["zones_validated"]}
    if not validated <= selected:
        raise OfflineBundleError("Zone validée absente de la sélection historique")
    classes = []
    for code in sorted({zones[zone] for zone in selected}):
        selected_ids = sorted(zone for zone in selected if zones[zone] == code)
        validated_ids = sorted(set(selected_ids) & validated)
        classes.append(
            {
                "koppen": code,
                "selected_zones": selected_ids,
                "validated_zones": validated_ids,
                "pattern_count": sum(
                    bool(set(pattern["zones_validated"]) & set(selected_ids))
                    for pattern in patterns
                ),
            }
        )
    return {
        "zones_sha256": digest,
        "classes": classes,
        "zones": {zone: zones[zone] for zone in sorted(selected)},
        "evidence": "Descriptive coverage of stored zones_validated; not held-out generalisation",
    }


def load_offline_bundle(
    archive: str | Path, expected_sha256: str, zones_geojson: str | Path | None = None
) -> dict:
    """Verify all ZIP objects, then validate its two libraries against PatternLibrary."""
    try:
        manifest, outputs, input_manifests, archive_bytes = _verify_bundle(
            Path(archive), expected_sha256
        )
        coverage = _manifest_coverage(input_manifests)
        libraries = []
        with tempfile.TemporaryDirectory(prefix="geoai-o3-bundle-") as temporary:
            for key, algorithm in sorted(OUTPUT_KEYS.items()):
                path = Path(temporary) / f"{algorithm}.parquet"
                path.write_bytes(outputs[key])
                libraries.append(_read_library(path, algorithm).model_dump(mode="json"))
        patterns = sorted(
            (p for lib in libraries for p in lib["patterns"]),
            key=lambda p: (p["algorithm"], p["pattern_id"]),
        )
        return {
            "schema_version": "geoai-o3-bundle-report/v1",
            "archive": {"sha256": expected_sha256.lower(), "bytes": archive_bytes},
            "integrity": {
                "verified_objects": len(manifest["objects"]),
                "verified_source_bytes": manifest["inventory"]["total_source_bytes"],
            },
            "source_bundle": manifest,
            "input_manifests": input_manifests,
            "library_schema_version": libraries[0]["schema_version"],
            "libraries": [
                {k: v for k, v in lib.items() if k != "patterns"}
                | {"pattern_count": len(lib["patterns"])}
                for lib in libraries
            ],
            "patterns": patterns,
            "manifest_reported_rows": coverage,
            "zone_class_coverage": _join_zones(zones_geojson, input_manifests, patterns)
            if zones_geojson is not None
            else None,
            "caveats": CAVEATS + manifest["caveats"],
        }
    except OfflineBundleError:
        raise
    except (
        OSError,
        ValueError,
        TypeError,
        KeyError,
        zipfile.BadZipFile,
        zlib.error,
        NotImplementedError,
        duckdb.Error,
        ValidationError,
    ) as exc:
        raise OfflineBundleError(f"Bundle refusé : {exc}") from exc


def _json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n"
    ).encode("utf-8")


def _markdown_cell(value: Any) -> str:
    text = "null (non renseignée)" if value is None else str(value)
    return (
        html.escape(text)
        .replace("|", "&#124;")
        .replace("\n", " ")
        .replace("\r", " ")
        .replace("`", "&#96;")
    )


def render_bundle_markdown(report: dict) -> str:
    source = report["source_bundle"]
    lines = [
        "# Inventaire O3 offline",
        "",
        f"SHA-256 archive : `{report['archive']['sha256']}`",
        "",
        f"Export source : {_markdown_cell(source['generated_at'])}",
        f"Patterns : {len(report['patterns'])} ; objets vérifiés : {report['integrity']['verified_objects']}.",
        "",
        "## Provenance",
        "",
    ]
    for key, value in sorted(source["historical_producers"].items()):
        lines.append(f"- {_markdown_cell(key)} : {_markdown_cell(value)}")
    lines += ["", "## Limites", ""] + [f"- {_markdown_cell(c)}" for c in report["caveats"]]
    lines += [
        "",
        "## Lignes rapportées par les manifests (inputs non relus)",
        "",
        "| Zone | Début | Fin | n_matched | Manifest |",
        "|---|---|---|---|---|",
    ]
    for row in report["manifest_reported_rows"]:
        lines.append(
            "| "
            + " | ".join(
                _markdown_cell(row[key])
                for key in ("zone_id", "period_start", "period_end", "n_matched", "manifest_key")
            )
            + " |"
        )
    if report["zone_class_coverage"] is not None:
        lines += [
            "",
            "## Couverture descriptive Köppen (pas une généralisation held-out)",
            "",
            f"SHA-256 zones : `{report['zone_class_coverage']['zones_sha256']}`",
            "",
            "| Classe | Zones sélectionnées | Zones validées stockées | Patterns concernés |",
            "|---|---|---|---|",
        ]
        for row in report["zone_class_coverage"]["classes"]:
            lines.append(
                "| "
                + " | ".join(
                    _markdown_cell(v)
                    for v in (
                        row["koppen"],
                        ", ".join(row["selected_zones"]),
                        ", ".join(row["validated_zones"]),
                        row["pattern_count"],
                    )
                )
                + " |"
            )
    lines += [
        "",
        "## Patterns",
        "",
        "| Algorithme | ID | Pattern | Variables | Support | Stabilité | Lag (jours) | Folds | Zones validées | Période | Confiance |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for p in report["patterns"]:
        values = [
            p["algorithm"],
            p["pattern_id"],
            p["pattern"],
            ", ".join(p["variables"]),
            p["support"],
            p["stability"],
            p["lag_days"],
            p["n_folds"],
            ", ".join(p["zones_validated"]),
            f"{p['period_start']} – {p['period_end']}",
            p["confidence"],
        ]
        lines.append("| " + " | ".join(_markdown_cell(v) for v in values) + " |")
    return "\n".join(lines) + "\n"


def _json_script(value: Any) -> str:
    return (
        _json_bytes(value)
        .decode("utf-8")
        .replace("&", "\\u0026")
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("\u2028", "\\u2028")
        .replace("\u2029", "\\u2029")
    )


def render_bundle_html(report: dict) -> str:
    caveats = "".join(f"<li>{html.escape(c)}</li>" for c in report["caveats"])
    supplemental = html.escape(
        json.dumps(
            {
                "manifest-reported rows (inputs non relus)": report["manifest_reported_rows"],
                "couverture descriptive Köppen": report["zone_class_coverage"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    provenance = html.escape(
        json.dumps(
            report["source_bundle"]["historical_producers"], sort_keys=True, ensure_ascii=False
        )
    )
    # Dataset strings only reach textContent; the JSON data block cannot close its script.
    return (
        """<!doctype html>
<html lang="fr"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Inventaire O3 offline</title><style>body{font:16px system-ui;margin:2rem;color:#17202a}label{display:inline-block;margin:0 1rem 1rem 0}table{border-collapse:collapse;width:100%}td,th{border:1px solid #ccd;padding:.5rem;text-align:left}th{background:#eef}small{overflow-wrap:anywhere}select,input{max-width:14rem}td{overflow-wrap:anywhere}</style>
<h1>Inventaire O3 offline</h1><p>Export source : """
        + html.escape(report["source_bundle"]["generated_at"])
        + """</p><p><small>SHA-256 : """
        + html.escape(report["archive"]["sha256"])
        + """</small></p><p>Provenance : """
        + provenance
        + """</p><ul>"""
        + caveats
        + """</ul><details><summary>Couverture et lignes rapportées par manifests</summary><pre>"""
        + supplemental
        + """</pre></details>
<label>Algorithme <select id="algorithm"><option value="">Tous</option></select></label>
<label>Variable <select id="variable"><option value="">Toutes</option></select></label>
<label>Zone validée <select id="zone"><option value="">Toutes</option></select></label>
<label>Support minimum <input id="support" type="number" min="0" max="1" step="0.01"></label>
<label>Stabilité minimum <input id="stability" type="number" min="0" max="1" step="0.01"></label>
<label>Lag absolu maximum (jours) <input id="lag" type="number" min="0" step="any"></label>
<p id="count" aria-live="polite"></p><div style="overflow:auto"><table><thead><tr><th>Algorithme</th><th>ID</th><th>Pattern</th><th>Variables</th><th>Support</th><th>Stabilité</th><th>Lag (jours)</th><th>Folds</th><th>Zones validées</th><th>Période</th><th>Confiance</th></tr></thead><tbody id="rows"></tbody></table></div>
<noscript>Activez JavaScript pour consulter et filtrer la table ; le rapport Markdown contient tous les patterns.</noscript>
<script id="dataset" type="application/json">"""
        + _json_script(report["patterns"])
        + """</script>
<script>
'use strict';
const patterns=JSON.parse(document.getElementById('dataset').textContent);
const ids=['algorithm','variable','zone','support','stability','lag'];
const controls=Object.fromEntries(ids.map(id=>[id,document.getElementById(id)]));
for(const [id,values] of [['algorithm',patterns.map(p=>p.algorithm)],['variable',patterns.flatMap(p=>p.variables)],['zone',patterns.flatMap(p=>p.zones_validated)]]){
 for(const value of [...new Set(values)].sort()){const option=document.createElement('option');option.value=value;option.textContent=value;controls[id].appendChild(option);}
}
function render(){
 const selected=patterns.filter(p=>(!controls.algorithm.value||p.algorithm===controls.algorithm.value)&&(!controls.variable.value||p.variables.includes(controls.variable.value))&&(!controls.zone.value||p.zones_validated.includes(controls.zone.value))&&(!controls.support.value||p.support>=Number(controls.support.value))&&(!controls.stability.value||(p.stability!==null&&p.stability>=Number(controls.stability.value)))&&(!controls.lag.value||Math.abs(p.lag_days)<=Number(controls.lag.value)));
 const body=document.getElementById('rows');body.replaceChildren();
 for(const p of selected){const row=document.createElement('tr');for(const value of [p.algorithm,p.pattern_id,p.pattern,p.variables.join(', '),p.support,p.stability,p.lag_days,p.n_folds,p.zones_validated.join(', '),[p.period_start,p.period_end].join(' – '),p.confidence]){const cell=document.createElement('td');cell.textContent=value===null?'null (non renseignée)':String(value);row.appendChild(cell);}body.appendChild(row);}
 document.getElementById('count').textContent=selected.length+' / '+patterns.length+' patterns';
}
for(const control of Object.values(controls)){control.addEventListener('input',render);}render();
</script></html>
"""
    )


def write_bundle_reports(report: dict, output_dir: str | Path) -> list[Path]:
    """Write deterministic reports exclusively; existing files are never overwritten."""
    destination = Path(output_dir)
    contents = [
        _json_bytes(report),
        render_bundle_markdown(report).encode("utf-8"),
        render_bundle_html(report).encode("utf-8"),
    ]
    paths = [destination / name for name in REPORT_FILES]
    if any(path.exists() or path.is_symlink() for path in paths):
        raise OfflineBundleError("Rapport existant : choisir un nouveau répertoire de sortie")
    try:
        destination.mkdir(parents=True, exist_ok=True)
        # Private staging on the destination filesystem: no visible output until all
        # staging writes succeeded. Exclusive creation avoids clobbering without
        # requiring filesystem hard-link support; visible copies are not atomic.
        with tempfile.TemporaryDirectory(prefix=".geoai-o3-report-", dir=destination) as temporary:
            staged = [Path(temporary) / name for name in REPORT_FILES]
            published = []
            try:
                for path, content in zip(staged, contents, strict=True):
                    with path.open("xb") as output:
                        output.write(content)
                for source, path in zip(staged, paths, strict=True):
                    with source.open("rb") as staged_input, path.open("xb") as output:
                        identity = os.fstat(output.fileno())
                        # Register ownership before copying, including partial failures.
                        published.append((path, identity))
                        shutil.copyfileobj(staged_input, output)
            except BaseException:
                # Only remove outputs this invocation published, and leave any
                # subsequently replaced file/symlink alone.
                for path, identity in reversed(published):
                    try:
                        current = path.lstat()
                        if (current.st_dev, current.st_ino) == (identity.st_dev, identity.st_ino):
                            path.unlink()
                    except FileNotFoundError:
                        pass
                raise
    except OSError as exc:
        raise OfflineBundleError(f"Écriture du rapport refusée : {exc}") from exc
    return paths
