"""The registered catalogue shares climate reference coordinates, never client sites."""

import asyncio
import hashlib
import io
import json
import math
from pathlib import Path

import pytest
from fastmcp import Client, FastMCP

import src.zone_catalogue as catalogue_module
from src.contracts import tool_capabilities_json
from src.resources import register_agent_resources
from src.tools.climate import _load_zones

_SOURCE = Path(__file__).resolve().parents[1] / "zones" / "zones.geojson"
_URI = "geoai://zones/catalogue"


async def _read_registered_catalogue(mcp: FastMCP | None = None) -> tuple[str, str]:
    if mcp is None:
        mcp = FastMCP("zone-catalogue-test")
        register_agent_resources(mcp)
    async with Client(mcp) as client:
        resources = await client.list_resources()
        resource = next(resource for resource in resources if str(resource.uri) == _URI)
        contents = await client.read_resource(_URI)
        return resource.mimeType, "".join(content.text for content in contents)


def test_registered_catalogue_contains_all_15_climate_reference_points():
    mime_type, text = asyncio.run(_read_registered_catalogue())
    payload = json.loads(text)
    source = json.loads(_SOURCE.read_bytes())
    expected = {
        feature["properties"]["zone_id"]: feature["properties"]
        for feature in source["features"]
        if feature["properties"]["role"] in {"panel", "pilot"}
    }
    climate_zones = _load_zones()

    assert mime_type == "application/json"
    assert set(payload) == {"schema_version", "catalogue_version", "zones"}
    assert payload["schema_version"] == "geoai-zone-catalogue/v1"
    assert (
        payload["catalogue_version"] == "sha256:" + hashlib.sha256(_SOURCE.read_bytes()).hexdigest()
    )
    assert len(payload["zones"]) == len(expected) == 15
    assert [zone["zone_id"] for zone in payload["zones"]] == sorted(expected)
    assert sum(zone["role"] == "panel" for zone in payload["zones"]) == 12
    assert sum(zone["role"] == "pilot" for zone in payload["zones"]) == 3
    for zone in payload["zones"]:
        assert set(zone) == {"zone_id", "label", "role", "latitude", "longitude"}
        zone_id = zone["zone_id"]
        assert zone["label"] == expected[zone_id]["name"]
        assert zone["role"] == expected[zone_id]["role"]
        assert zone["latitude"] == climate_zones[zone_id]["lat"]
        assert zone["longitude"] == climate_zones[zone_id]["lon"]
        assert math.isfinite(zone["latitude"]) and -90 <= zone["latitude"] <= 90
        assert math.isfinite(zone["longitude"]) and -180 <= zone["longitude"] <= 180


def test_actual_server_registers_resource_without_adding_a_tool():
    from src.server import mcp

    async def inspect_server():
        async with Client(mcp) as client:
            return {tool.name for tool in await client.list_tools()}

    _, text = asyncio.run(_read_registered_catalogue(mcp))
    assert len(json.loads(text)["zones"]) == 15
    assert asyncio.run(inspect_server()) == set(tool_capabilities_json()["tools"])


def test_registered_catalogue_is_deterministic_and_performs_no_writes(monkeypatch):
    original_open = io.open

    def read_only_open(file, mode="r", *args, **kwargs):
        assert not any(flag in mode for flag in "wax+"), f"Unexpected write: {file}"
        return original_open(file, mode, *args, **kwargs)

    monkeypatch.setattr(io, "open", read_only_open)
    first = asyncio.run(_read_registered_catalogue())
    assert asyncio.run(_read_registered_catalogue()) == first


def test_version_hashes_source_bytes_and_client_sites_stay_out(tmp_path, monkeypatch):
    source = json.loads(_SOURCE.read_bytes())
    source["features"].append(
        {
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [1, 2]},
            "properties": {"zone_id": "personal_site", "role": "client", "name": "Private"},
        }
    )
    path = tmp_path / "zones.geojson"
    path.write_text(json.dumps(source), encoding="utf-8")
    monkeypatch.setattr(catalogue_module, "_ZONES_GEOJSON", path)
    _, first_text = asyncio.run(_read_registered_catalogue())
    first = json.loads(first_text)
    assert len(first["zones"]) == 15
    assert "personal_site" not in first_text
    assert "Private" not in first_text
    path.write_bytes(path.read_bytes() + b"\n")
    _, second_text = asyncio.run(_read_registered_catalogue())
    second = json.loads(second_text)
    assert first["zones"] == second["zones"]
    assert first["catalogue_version"] != second["catalogue_version"]
    assert second["catalogue_version"] == "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("zone_id", "../personal"),
        ("zone_id", ""),
        ("name", "  "),
        ("role", "unknown"),
        ("bbox", [0, 0, float("nan"), 1]),
        ("bbox", [0, 0, float("inf"), 1]),
        ("bbox", [0, -91, 1, 1]),
        ("bbox", [-181, 0, 1, 1]),
        ("bbox", [2, 0, 1, 1]),
        ("bbox", [0, 2, 1, 1]),
        ("bbox", [False, 0, 1, 1]),
        ("bbox", [0, 0, 1]),
    ],
)
def test_catalogue_rejects_invalid_source_points(tmp_path, monkeypatch, field, value):
    source = json.loads(_SOURCE.read_bytes())
    source["features"][0]["properties"][field] = value
    path = tmp_path / "zones.geojson"
    path.write_text(json.dumps(source), encoding="utf-8")
    monkeypatch.setattr(catalogue_module, "_ZONES_GEOJSON", path)
    with pytest.raises(ValueError):
        catalogue_module.zone_catalogue()


def test_catalogue_rejects_duplicate_ids(tmp_path, monkeypatch):
    source = json.loads(_SOURCE.read_bytes())
    source["features"].append(source["features"][0])
    path = tmp_path / "zones.geojson"
    path.write_text(json.dumps(source), encoding="utf-8")
    monkeypatch.setattr(catalogue_module, "_ZONES_GEOJSON", path)
    with pytest.raises(ValueError, match="Duplicate zone_id"):
        catalogue_module.zone_catalogue()
