"""Minimal subprocess entrypoint for one bounded GeoAI mesh query."""

from __future__ import annotations

import json
import pickle
import sys
from typing import Any

from src.tools.spatial import MESH_MAX_IPC_BYTES, MESH_MAX_REQUEST_BYTES, _execute_spatial_query


def _error_payload(status: str = "error") -> bytes:
    return pickle.dumps((status, None), protocol=pickle.HIGHEST_PROTOCOL)


def _parse_request(raw: bytes) -> tuple[str, str, str, float | None, str]:
    value: Any = json.loads(raw)
    if not isinstance(value, dict) or set(value) != {
        "sql",
        "source",
        "geom_json",
        "distance_m",
        "geometry_column",
    }:
        raise ValueError("invalid mesh worker request")
    sql = value["sql"]
    source = value["source"]
    geom_json = value["geom_json"]
    distance_m = value["distance_m"]
    geometry_column = value["geometry_column"]
    if not all(isinstance(item, str) for item in (sql, source, geom_json, geometry_column)):
        raise ValueError("invalid mesh worker request")
    if distance_m is not None and (
        isinstance(distance_m, bool) or not isinstance(distance_m, (int, float))
    ):
        raise ValueError("invalid mesh worker request")
    return (
        sql,
        source,
        geom_json,
        float(distance_m) if distance_m is not None else None,
        geometry_column,
    )


def main() -> None:
    try:
        raw = sys.stdin.buffer.read(MESH_MAX_REQUEST_BYTES + 1)
        if not raw or len(raw) > MESH_MAX_REQUEST_BYTES:
            raise ValueError("invalid mesh worker request")
        sql, source, geom_json, distance_m, geometry_column = _parse_request(raw)
        result = _execute_spatial_query(
            sql,
            source=source,
            geom_json=geom_json,
            distance_m=distance_m,
            install_extensions=False,
            geometry_column=geometry_column,
        )
        encoded = pickle.dumps(("ok", result), protocol=pickle.HIGHEST_PROTOCOL)
        if len(encoded) > MESH_MAX_IPC_BYTES:
            encoded = _error_payload("too_large")
    except BaseException:
        encoded = _error_payload()
    sys.stdout.buffer.write(encoded)
    sys.stdout.buffer.flush()


if __name__ == "__main__":
    main()
