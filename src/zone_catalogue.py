"""Read-only public catalogue of the shipped scientific zone reference points."""

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

_ZONES_GEOJSON = Path(__file__).resolve().parents[1] / "zones" / "zones.geojson"
_SAFE_ZONE_ID = re.compile(r"^[A-Za-z0-9._-]+$")


def zone_catalogue() -> dict[str, Any]:
    """Hash and project the same source bytes, without providers or persistence.

    Coordinates deliberately match climate._load_zones' bbox midpoint, rather
    than a polygon centroid or a personal GPS point. Client sites stay private.
    """
    source = _ZONES_GEOJSON.read_bytes()
    collection = json.loads(source)
    if collection.get("type") != "FeatureCollection" or not isinstance(
        collection.get("features"), list
    ):
        raise ValueError("Invalid zone FeatureCollection")
    zones = []
    seen = set()
    for feature in collection["features"]:
        properties = feature.get("properties", {})
        zone_id = properties.get("zone_id")
        if not isinstance(zone_id, str) or not _SAFE_ZONE_ID.fullmatch(zone_id):
            raise ValueError("Invalid zone_id")
        if zone_id in seen:
            raise ValueError(f"Duplicate zone_id: {zone_id}")
        seen.add(zone_id)
        role = properties.get("role")
        if role not in {"panel", "pilot", "client"}:
            raise ValueError(f"Invalid zone role: {zone_id}")
        if role == "client":
            continue
        label = properties.get("name")
        bbox = properties.get("bbox")
        if not isinstance(label, str) or not label.strip():
            raise ValueError(f"Invalid zone label: {zone_id}")
        if (
            not isinstance(bbox, list)
            or len(bbox) != 4
            or any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                for value in bbox
            )
        ):
            raise ValueError(f"Invalid zone bbox: {zone_id}")
        west, south, east, north = bbox
        if not (-180 <= west <= east <= 180 and -90 <= south <= north <= 90):
            raise ValueError(f"Zone bbox outside WGS84 bounds: {zone_id}")
        zones.append(
            {
                "zone_id": zone_id,
                "label": label,
                "role": role,
                "latitude": (south + north) / 2.0,
                "longitude": (west + east) / 2.0,
            }
        )
    return {
        "schema_version": "geoai-zone-catalogue/v1",
        "catalogue_version": "sha256:" + hashlib.sha256(source).hexdigest(),
        "zones": sorted(zones, key=lambda zone: zone["zone_id"]),
    }
