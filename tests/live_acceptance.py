"""Manual LIVE acceptance harness. Not a test module, and not a CI gate.

Renamed from ``tests/test_integration.py`` (#171). Under that name it contributed
**zero** collected tests while being the only file in the repo whose name claimed to be
the integration suite, and CI had to exclude it by hand — so "integration tests 100%
green" was true of a set nobody could point at. It is now named for what it is, and
pytest does not collect it because the name no longer matches a test module. The
exclusion is structural; there is no ``--ignore`` in the workflow any more.

This file is the **live** half of FIERCE Lot 8 under the maintainer's option 3: the
hermetic pytest suite is the non-regression gate, and this harness is a manual
acceptance step recorded per run. A green hermetic job is not evidence that this ran.

Run it against a live stack, from inside the container::

    docker compose exec geoai-mcp python tests/live_acceptance.py

It needs, and will fail without:

- RustFS/MinIO reachable at ``S3_ENDPOINT`` (the setup step writes a test GeoParquet);
- a STAC catalog reachable at ``STAC_URL``;
- the MCP server answering on ``http://localhost:5001/mcp``.

The dev compose mounts this file read-only into ``geoai-mcp`` (``/app/tests/``), and sets
``GEOAI_MESH_AUTH_TOKEN`` there, so the server only accepts bearer-authenticated callers.
The harness sends that variable as its bearer (``_client()``); if it is unset or blank it
connects without auth, matching a server with auth disabled. It checks it can connect
*before* it writes anything to the store, so a rejected run leaves the store untouched.
The token is read from the process environment only, not from a ``.env`` file. The no-auth
path fits a server started without a data plane, which cannot pass the S3-backed spatial
cases.

It reaches public Sentinel-2 imagery over the network, so its results depend on upstream
availability and can differ between runs for reasons that are not regressions. That is
why it is not a gate.

Recording a run is part of the procedure, not an afterthought: see
``docs/runbooks/live-acceptance.md`` for the evidence a run must carry (commit,
platform, environment, per-case results, limitations). ``docs/TEST_INVENTORY.md`` holds
the case inventory and the Lot 8 wording.

Each assertion goes through ``check()``; the count is inventoried and guarded by
``tests/test_test_inventory.py``.
"""

import asyncio
import json
import os
import sys
import time

# ---------------------------------------------------------------------------
# Setup: upload test GeoParquet to MinIO
# ---------------------------------------------------------------------------

# The harness bucket (`geoai-dev` since 2026-09-27; docs/dev-harness.md, Host ports).
S3_PATH = f"s3://{os.environ.get('S3_BUCKET') or 'geoai-dev'}/geoai/test_cities.parquet"


def setup_geoparquet():
    """Upload test GeoParquet with French cities to MinIO."""
    import duckdb

    conn = duckdb.connect(":memory:")
    conn.execute("INSTALL spatial; LOAD spatial;")
    conn.execute("INSTALL httpfs; LOAD httpfs;")

    endpoint = os.environ.get("S3_ENDPOINT", "")
    if endpoint.startswith("http://"):
        endpoint = endpoint[len("http://") :]
    elif endpoint.startswith("https://"):
        endpoint = endpoint[len("https://") :]
    endpoint = endpoint.rstrip("/")

    conn.execute(f"SET s3_endpoint = '{endpoint}'")
    conn.execute(f"SET s3_access_key_id = '{os.environ.get('S3_ACCESS_KEY', '')}'")
    conn.execute(f"SET s3_secret_access_key = '{os.environ.get('S3_SECRET_KEY', '')}'")
    conn.execute("SET s3_url_style = 'path'")
    conn.execute("SET s3_use_ssl = false")

    conn.execute("""
        CREATE TABLE cities AS
        SELECT * FROM (VALUES
            ('Paris',       'France',  2161000, ST_GeomFromText('POLYGON((2.25 48.81, 2.42 48.81, 2.42 48.90, 2.25 48.90, 2.25 48.81))')),
            ('Lyon',        'France',   516000, ST_GeomFromText('POLYGON((4.80 45.72, 4.90 45.72, 4.90 45.80, 4.80 45.80, 4.80 45.72))')),
            ('Marseille',   'France',   870000, ST_GeomFromText('POLYGON((5.33 43.25, 5.42 43.25, 5.42 43.33, 5.33 43.33, 5.33 43.25))')),
            ('Toulouse',    'France',   493000, ST_GeomFromText('POLYGON((1.38 43.55, 1.50 43.55, 1.50 43.65, 1.38 43.65, 1.38 43.55))')),
            ('Bordeaux',    'France',   260000, ST_GeomFromText('POLYGON((-0.63 44.81, -0.53 44.81, -0.53 44.88, -0.63 44.88, -0.63 44.81))')),
            ('Montpellier', 'France',   295000, ST_GeomFromText('POLYGON((3.82 43.58, 3.93 43.58, 3.93 43.65, 3.82 43.65, 3.82 43.58))')),
            ('Nantes',      'France',   320000, ST_GeomFromText('POLYGON((-1.60 47.19, -1.50 47.19, -1.50 47.26, -1.60 47.26, -1.60 47.19))')),
            ('Strasbourg',  'France',   284000, ST_GeomFromText('POLYGON((7.72 48.55, 7.82 48.55, 7.82 48.62, 7.72 48.62, 7.72 48.55))')),
            ('Lille',       'France',   236000, ST_GeomFromText('POLYGON((3.02 50.60, 3.10 50.60, 3.10 50.66, 3.02 50.66, 3.02 50.60))')),
            ('Nice',        'France',   342000, ST_GeomFromText('POLYGON((7.22 43.68, 7.30 43.68, 7.30 43.73, 7.22 43.73, 7.22 43.68))'))
        ) AS t(name, country, population, geometry)
    """)

    conn.execute(f"COPY cities TO '{S3_PATH}' (FORMAT PARQUET)")
    conn.close()
    print(f"  Uploaded {S3_PATH}")


# ---------------------------------------------------------------------------
# Connection
# ---------------------------------------------------------------------------

MCP_URL = "http://localhost:5001/mcp"

#: Seconds the preflight may take. Applied to the preflight only, never to the client: one
#: case below allows up to 60 s.
PREFLIGHT_TIMEOUT_S = 30


def _auth():
    """The bearer the server expects, or ``None`` when it runs with auth disabled.

    Same normalisation as the server (``src.auth.build_mesh_auth`` strips and treats blank
    as disabled). Never printed.
    """
    return os.environ.get("GEOAI_MESH_AUTH_TOKEN", "").strip() or None


def _client():
    """A fastmcp ``Client`` for ``MCP_URL``; a ``str`` ``auth`` is sent as ``Bearer <str>``."""
    from fastmcp import Client

    return Client(MCP_URL, auth=_auth())


async def preflight():
    """Connect and list tools; raise if the server is unreachable or rejects the bearer."""
    async with _client() as client:
        await client.list_tools()


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------

passed = 0
failed = 0
skipped = 0

#: The number of cases this harness inventories — the figure `docs/TEST_INVENTORY.md`
#: publishes and `tests/test_test_inventory.py` guards. Every case ends in exactly one of
#: pass, fail or skip, so a run's three counters always sum to it.
INVENTORIED_CASES = 56


def call_tool(client, name, args):
    """Call MCP tool and return parsed JSON."""
    result = asyncio.get_event_loop().run_until_complete(client.call_tool(name, args))
    return json.loads(result.content[0].text)


def check(label, condition, detail=""):
    """Assert a condition and print result."""
    global passed, failed
    if condition:
        passed += 1
        print(f"  PASS  {label}")
    else:
        failed += 1
        print(f"  FAIL  {label}  {detail}")


def skip(label, reason):
    """Record an inventoried case the live response gave nothing to check.

    Twelve of the 56 cases sit behind a response actually containing features or a
    profile. They are legitimately conditional — there is no feature shape to verify when
    the provider returned no features — but before #171's review they simply did not run,
    and the summary printed ALL TESTS PASSED over a run that had executed 44 of them. A
    number the acceptance criterion cites has to be accounted for, not silently reduced.
    """
    global skipped
    skipped += 1
    print(f"  SKIP  {label}  ({reason})")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


async def run_tests():
    global passed, failed

    async with _client() as client:
        # --- Tool listing ---
        print("\n=== Tools ===")
        tools = await client.list_tools()
        tool_names = {t.name for t in tools}
        expected = {
            "stac_search",
            "stac_download",
            "spatial_query",
            "change_detect",
            "ndvi_profile",
            "vegetation_stress",
            "discover_patterns",
            "discover_patterns_cv",
            "era5_ingest",
            "climate_enrich_ndvi",
            "climate_percentiles",
            "forecast_query",
        }
        check("Core tools registered", expected <= tool_names, f"got {tool_names}")

        # --- STAC search ---
        print("\n=== STAC Search ===")
        result = await client.call_tool(
            "stac_search",
            {
                "bbox": [-122.5, 37.7, -122.3, 37.9],
                "datetime": "2024-06-01/2024-06-30",
                "collections": ["sentinel-2-l2a"],
                "cloud_cover_max": 30,
                "limit": 3,
            },
        )
        data = json.loads(result.content[0].text)
        check("Returns FeatureCollection", data.get("type") == "FeatureCollection")
        check("Has results", data["metadata"]["total_results"] > 0)
        check("Respects limit", len(data["features"]) <= 3)
        feat = data["features"][0]
        check("Has geometry", "geometry" in feat)
        check("Has cloud_cover", "eo:cloud_cover" in feat["properties"])
        check("Cloud cover within range", feat["properties"]["eo:cloud_cover"] <= 30)

        # --- STAC search limit clamp ---
        result = await client.call_tool(
            "stac_search",
            {
                "bbox": [-122.5, 37.7, -122.3, 37.9],
                "datetime": "2024-01-01/2024-12-31",
                "collections": ["sentinel-2-l2a"],
                "cloud_cover_max": 100,
                "limit": 9999,
            },
        )
        data = json.loads(result.content[0].text)
        check("Limit clamped to 50", data["metadata"]["total_results"] <= 50)

        # --- Spatial: intersects ---
        print("\n=== Spatial Query: intersects ===")
        result = await client.call_tool(
            "spatial_query",
            {
                "source": S3_PATH,
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[-1, 42], [8, 42], [8, 45], [-1, 45], [-1, 42]]],
                },
                "operation": "intersects",
            },
        )
        data = json.loads(result.content[0].text)
        check("No error", "error" not in data, data.get("error", ""))
        names = {f["properties"]["name"] for f in data.get("features", [])}
        expected_south = {"Marseille", "Toulouse", "Bordeaux", "Montpellier", "Nice"}
        check("Found 5 southern cities", names == expected_south, f"got {names}")

        # --- Spatial: contains ---
        print("\n=== Spatial Query: contains ===")
        result = await client.call_tool(
            "spatial_query",
            {
                "source": S3_PATH,
                "geometry": {"type": "Point", "coordinates": [4.85, 45.76]},
                "operation": "contains",
            },
        )
        data = json.loads(result.content[0].text)
        check("No error", "error" not in data, data.get("error", ""))
        check("Found 1 city", data.get("metadata", {}).get("total_results") == 1)
        if data.get("features"):
            check("City is Lyon", data["features"][0]["properties"]["name"] == "Lyon")
        else:
            skip("City is Lyon", "no features returned")

        # --- Spatial: distance ---
        print("\n=== Spatial Query: distance ===")
        result = await client.call_tool(
            "spatial_query",
            {
                "source": S3_PATH,
                "geometry": {"type": "Point", "coordinates": [2.35, 48.86]},
                "operation": "distance",
                "distance_m": 200000,
            },
        )
        data = json.loads(result.content[0].text)
        check("No error", "error" not in data, data.get("error", ""))
        names = {f["properties"]["name"] for f in data.get("features", [])}
        check("Paris in results", "Paris" in names, f"got {names}")

        # --- Spatial: select_columns ---
        print("\n=== Spatial Query: select_columns ===")
        result = await client.call_tool(
            "spatial_query",
            {
                "source": S3_PATH,
                "geometry": {"type": "Point", "coordinates": [4.85, 45.76]},
                "operation": "contains",
                "select_columns": ["name", "geometry"],
            },
        )
        data = json.loads(result.content[0].text)
        check("No error", "error" not in data, data.get("error", ""))
        if data.get("features"):
            props = data["features"][0]["properties"]
            check("Has name", "name" in props)
            check("No population (filtered out)", "population" not in props)
        else:
            skip("Has name", "no features returned")
            skip("No population (filtered out)", "no features returned")

        # --- Spatial: invalid operation ---
        print("\n=== Spatial Query: invalid operation ===")
        result = await client.call_tool(
            "spatial_query",
            {
                "source": S3_PATH,
                "geometry": {"type": "Point", "coordinates": [0, 0]},
                "operation": "INVALID",
            },
        )
        data = json.loads(result.content[0].text)
        check("Returns error", "error" in data)

        # --- Spatial: GeoJSON output format ---
        print("\n=== Spatial Query: GeoJSON output ===")
        result = await client.call_tool(
            "spatial_query",
            {
                "source": S3_PATH,
                "geometry": {"type": "Point", "coordinates": [4.85, 45.76]},
                "operation": "contains",
            },
        )
        data = json.loads(result.content[0].text)
        if data.get("features"):
            feat = data["features"][0]
            check("Feature type correct", feat["type"] == "Feature")
            geom = feat["geometry"]
            # Geometry may be dict, JSON string, or WKB string
            if isinstance(geom, str):
                try:
                    geom = json.loads(geom)
                except (json.JSONDecodeError, ValueError):
                    pass  # WKB or other format
            if isinstance(geom, dict):
                check("Geometry is Polygon", geom["type"] == "Polygon")
                check("Has coordinates", len(geom["coordinates"]) > 0)
                skip("Geometry present (non-GeoJSON format)", "geometry decoded as GeoJSON")
            else:
                check("Geometry present (non-GeoJSON format)", geom is not None)
                skip("Geometry is Polygon", "geometry is not GeoJSON")
                skip("Has coordinates", "geometry is not GeoJSON")
        else:
            for label in (
                "Feature type correct",
                "Geometry is Polygon",
                "Has coordinates",
                "Geometry present (non-GeoJSON format)",
            ):
                skip(label, "no features returned")

        # ---------------------------------------------------------------
        # NDVI Profile
        # ---------------------------------------------------------------
        print("\n=== NDVI Profile ===")
        result = await client.call_tool(
            "ndvi_profile",
            {
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        [
                            [-122.45, 37.75],
                            [-122.40, 37.75],
                            [-122.40, 37.80],
                            [-122.45, 37.80],
                            [-122.45, 37.75],
                        ]
                    ],
                },
                "start_date": "2024-06-01",
                "end_date": "2024-08-31",
                "cloud_cover_max": 30,
            },
        )
        data = json.loads(result.content[0].text)
        check("ndvi_profile: no error", "error" not in data, data.get("error", ""))
        check("ndvi_profile: has profile list", isinstance(data.get("profile"), list))
        check("ndvi_profile: has statistics", "statistics" in data)
        check("ndvi_profile: has alerts", "alerts" in data)
        check("ndvi_profile: has metadata", "metadata" in data)
        check(
            "ndvi_profile: images_processed in metadata",
            "images_processed" in data.get("metadata", {}),
        )

        if data["profile"]:
            entry = data["profile"][0]
            check("ndvi_profile: entry has date", "date" in entry)
            check("ndvi_profile: entry has ndvi_mean", "ndvi_mean" in entry)
            check(
                "ndvi_profile: NDVI in valid range",
                -1 <= entry["ndvi_mean"] <= 1,
                f"got {entry['ndvi_mean']}",
            )
            check("ndvi_profile: images_processed > 0", data["metadata"]["images_processed"] > 0)
        else:
            for label in (
                "entry has date",
                "entry has ndvi_mean",
                "NDVI in valid range",
                "images_processed > 0",
            ):
                skip(f"ndvi_profile: {label}", "the profile came back empty")

        # --- NDVI Profile with Point geometry ---
        print("\n=== NDVI Profile: Point geometry ===")
        result = await client.call_tool(
            "ndvi_profile",
            {
                "geometry": {"type": "Point", "coordinates": [-122.42, 37.77]},
                "start_date": "2024-06-01",
                "end_date": "2024-08-31",
            },
        )
        data = json.loads(result.content[0].text)
        check(
            "ndvi_profile (point): accepted without error",
            "error" not in data,
            data.get("error", ""),
        )
        check("ndvi_profile (point): has profile", isinstance(data.get("profile"), list))

        # ---------------------------------------------------------------
        # Vegetation Stress
        # ---------------------------------------------------------------
        print("\n=== Vegetation Stress ===")
        result = await client.call_tool(
            "vegetation_stress",
            {
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        [
                            [-122.45, 37.75],
                            [-122.40, 37.75],
                            [-122.40, 37.80],
                            [-122.45, 37.80],
                            [-122.45, 37.75],
                        ]
                    ],
                },
                "lookback_days": 90,
            },
        )
        data = json.loads(result.content[0].text)
        check("veg_stress: no error", "error" not in data, data.get("error", ""))
        check("veg_stress: has analysis", "analysis" in data)
        check("veg_stress: has alerts", isinstance(data.get("alerts"), list))
        check("veg_stress: has metadata", "metadata" in data)
        if data.get("analysis", {}).get("current_ndvi") is not None:
            ndvi = data["analysis"]["current_ndvi"]
            check("veg_stress: NDVI in valid range", -1 <= ndvi <= 1, f"got {ndvi}")
        else:
            skip("veg_stress: NDVI in valid range", "no current_ndvi in the analysis")

        # ---------------------------------------------------------------
        # Change Detect
        # ---------------------------------------------------------------
        print("\n=== Change Detect ===")
        result = await client.call_tool(
            "change_detect",
            {
                "bbox": [-122.45, 37.75, -122.40, 37.80],
                "date_before": "2024-03-01",
                "date_after": "2024-08-01",
            },
        )
        data = json.loads(result.content[0].text)
        check("change_detect: no error", "error" not in data, data.get("error", ""))
        check("change_detect: returns FeatureCollection", data.get("type") == "FeatureCollection")
        check("change_detect: has metadata", "metadata" in data)
        meta = data.get("metadata", {})
        check(
            "change_detect: method is ndvi_differencing", meta.get("method") == "ndvi_differencing"
        )
        check("change_detect: has before_image", "before_image" in meta)
        check("change_detect: has after_image", "after_image" in meta)
        check("change_detect: has ndvi_difference", "ndvi_difference" in meta)

        # --- Change Detect: area guard rail ---
        print("\n=== Change Detect: area guard rail ===")
        result = await client.call_tool(
            "change_detect",
            {
                "bbox": [-130, 30, -100, 50],
                "date_before": "2024-01-01",
                "date_after": "2024-06-01",
            },
        )
        data = json.loads(result.content[0].text)
        check(
            "change_detect guard rail: returns error",
            "error" in data,
            f"got keys: {list(data.keys())}",
        )

        # ---------------------------------------------------------------
        # Phase 4: elapsed_ms observability
        # ---------------------------------------------------------------
        print("\n=== Observability: elapsed_ms ===")

        # stac_search elapsed_ms
        result = await client.call_tool(
            "stac_search",
            {
                "bbox": [-122.5, 37.7, -122.3, 37.9],
                "datetime": "2024-06-01/2024-06-30",
                "collections": ["sentinel-2-l2a"],
                "limit": 1,
            },
        )
        data = json.loads(result.content[0].text)
        check(
            "stac_search: has elapsed_ms",
            isinstance(data.get("metadata", {}).get("elapsed_ms"), (int, float)),
        )

        # ndvi_profile elapsed_ms (reuse earlier result would require re-call)
        result = await client.call_tool(
            "ndvi_profile",
            {
                "geometry": {"type": "Point", "coordinates": [-122.42, 37.77]},
                "start_date": "2024-07-01",
                "end_date": "2024-07-31",
            },
        )
        data = json.loads(result.content[0].text)
        check(
            "ndvi_profile: has elapsed_ms",
            isinstance(data.get("metadata", {}).get("elapsed_ms"), (int, float)),
        )

        # change_detect elapsed_ms (reuse earlier bbox)
        result = await client.call_tool(
            "change_detect",
            {
                "bbox": [-122.45, 37.75, -122.40, 37.80],
                "date_before": "2024-03-01",
                "date_after": "2024-08-01",
            },
        )
        data = json.loads(result.content[0].text)
        check(
            "change_detect: has elapsed_ms",
            isinstance(data.get("metadata", {}).get("elapsed_ms"), (int, float)),
        )

        # ---------------------------------------------------------------
        # Phase 5: Performance timing
        # ---------------------------------------------------------------
        print("\n=== Perf: ndvi_profile timing ===")
        t0 = time.time()
        result = await client.call_tool(
            "ndvi_profile",
            {
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        [
                            [-122.45, 37.75],
                            [-122.40, 37.75],
                            [-122.40, 37.80],
                            [-122.45, 37.80],
                            [-122.45, 37.75],
                        ]
                    ],
                },
                "start_date": "2024-06-01",
                "end_date": "2024-08-31",
                "cloud_cover_max": 30,
            },
        )
        elapsed = time.time() - t0
        data = json.loads(result.content[0].text)
        n_images = data["metadata"]["images_processed"]
        check(
            f"ndvi_profile perf: {n_images} images in {elapsed:.1f}s",
            elapsed < 30 or n_images == 0,
            f"took {elapsed:.1f}s for {n_images} images",
        )

        print("\n=== Perf: change_detect timing ===")
        t0 = time.time()
        result = await client.call_tool(
            "change_detect",
            {
                "bbox": [-122.45, 37.75, -122.40, 37.80],
                "date_before": "2024-03-01",
                "date_after": "2024-08-01",
            },
        )
        elapsed = time.time() - t0
        check(
            f"change_detect perf: completed in {elapsed:.1f}s", elapsed < 15, f"took {elapsed:.1f}s"
        )

        # ---------------------------------------------------------------
        # Phase 5: Concurrency — two parallel calls
        # ---------------------------------------------------------------
        print("\n=== Concurrency: parallel calls ===")
        t0 = time.time()
        r1, r2 = await asyncio.gather(
            client.call_tool(
                "ndvi_profile",
                {
                    "geometry": {"type": "Point", "coordinates": [-122.42, 37.77]},
                    "start_date": "2024-07-01",
                    "end_date": "2024-07-31",
                },
            ),
            client.call_tool(
                "ndvi_profile",
                {
                    "geometry": {"type": "Point", "coordinates": [2.35, 48.86]},
                    "start_date": "2024-07-01",
                    "end_date": "2024-07-31",
                },
            ),
        )
        elapsed = time.time() - t0
        d1 = json.loads(r1.content[0].text)
        d2 = json.loads(r2.content[0].text)
        check("Parallel call 1: no error", "error" not in d1, d1.get("error", ""))
        check("Parallel call 2: no error", "error" not in d2, d2.get("error", ""))
        check(f"Parallel calls completed in {elapsed:.1f}s", elapsed < 60, f"took {elapsed:.1f}s")

    # --- Summary ---
    # Against INVENTORIED_CASES, not against `passed + failed`. The old denominator was
    # whatever happened to run, so a run that skipped twelve conditional cases printed
    # ALL TESTS PASSED and looked like 44/44 — while the acceptance criterion cites 56.
    # Coverage is now part of the result, and the runbook's evidence template asks for it.
    executed = passed + failed
    print(f"\n{'=' * 50}")
    print(
        f"Results: {passed} passed, {failed} failed, {skipped} skipped "
        f"of {INVENTORIED_CASES} inventoried cases"
    )
    if executed + skipped != INVENTORIED_CASES:
        print(
            f"  NOTE  {executed + skipped} cases accounted for, {INVENTORIED_CASES} inventoried; "
            "the inventory and this harness have drifted"
        )
    if skipped:
        print(f"  Coverage: {executed}/{INVENTORIED_CASES} cases produced a pass or fail.")
    if failed:
        print("SOME CASES FAILED")
        sys.exit(1)
    print("NO CASE FAILED" if skipped else "ALL CASES PASSED")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    # Connect and authenticate before the setup step writes to the store: a rejected run
    # must not mutate it.
    print(f"Checking the MCP server at {MCP_URL}...")
    try:
        asyncio.run(asyncio.wait_for(preflight(), timeout=PREFLIGHT_TIMEOUT_S))
    except Exception as exc:  # noqa: BLE001 - report any connect/auth failure, then stop
        print(f"  FAIL  cannot connect to {MCP_URL}: {type(exc).__name__}: {exc}")
        print("  Nothing was written to the store.")
        sys.exit(2)
    print("Setting up test GeoParquet...")
    setup_geoparquet()
    print("Running integration tests...")
    asyncio.run(run_tests())


if __name__ == "__main__":
    main()
