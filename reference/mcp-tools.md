# MCP tools

Deposium GeoAI exposes 16 native tools to any compatible MCP client. This page helps you choose one; the [README](../README.md) covers installation and connection. A client can read `geoai://guide/agent` for workflow guidance and `geoai://capabilities/tools` for the authoritative machine-readable inventory, strict input/output schemas, persistence defaults, and stability metadata. These are MCP resource URIs, read through the client rather than ordinary web links.

The standalone server supplies data access and computation. A conversational interface, document retrieval (RAG), scheduled monitoring, notifications, and durable client-side storage require additional client or application features.

## Input conventions and access

- A `bbox` is a WGS84 array `[west, south, east, north]`, with increasing longitude and latitude bounds. A `geometry` is a GeoJSON geometry object with longitude/latitude coordinates, **not** a Feature or FeatureCollection.
- Dates use `YYYY-MM-DD`; STAC `datetime` accepts an instant or interval. Use explicit, short windows for a first query.
- A `zone_id` refers to a configured zone, discoverable through `geoai://zones/catalogue`. An arbitrary place name does not create a zone.
- Storage-free reads need reachable upstream providers. Stored-data workflows require configured object storage or explicitly enabled local sources confined to `LOCAL_PARQUET_SOURCE_ROOT`. Source paths are subject to each handler's validation.
- HTTP/SSE deployments with a data plane or solar collection require `GEOAI_MESH_AUTH_TOKEN`; stdio uses the launching process's access. Keep credentials on the server. See [server startup](../src/server.py), [configuration](../src/config.py), and the [agent resources](../src/resources.py).

The tables list essential inputs, not every option. Consult [input and output contracts](../src/contracts.py) for exact fields, defaults, and error shapes. Capability profiles and timeout classes are descriptive metadata; they do not themselves enforce authorization or execution deadlines.

## Weather and climate

Implementations: [forecast](../src/tools/forecast.py) and [climate](../src/tools/climate.py).

| Tool | Purpose | Essential inputs and conditions |
| --- | --- | --- |
| `forecast_query` | Get a short-term daily forecast for a point or zone. | Supply exactly one of `zone_id` or both `latitude` and `longitude`; choose `forecast_days`, optional `variables` and `provider`. Provider fallback and the served source appear in metadata. `write_output` defaults to `false`. |
| `era5_ingest` | Acquire historical daily weather through the Open-Meteo Archive API. | `zone_id`, `start_date`, `end_date`; optional `variables`. Only `temporal="daily"` is supported. `write_output` defaults to `true`; set it to `false` for an acquisition probe without persistence. This is a zone-centroid series, not a full ERA5 grid download. |
| `climate_enrich_ndvi` | Align NDVI observations with a daily ERA5 table. | `zone_id`, dates, optional `era5_source` and `ndvi_source`; `tolerance_days` controls nearest-date matching. An explicit unusable `ndvi_source` produces an error rather than silently fetching new imagery. `strict_window=true` requires complete ERA5 window coverage. Writes by default. |
| `climate_percentiles` | Build monthly P10/P50/P90 climate reference distributions. | `zone_id`; optional `era5_source`, date window and `variables`. Can classify supplied observations with `classify_observed` and `classify_month`. Writes by default; needs an appropriate historical source. |
| `climate_normals` | Read stored monthly reference distributions and optionally classify dated values. | `zone_id`, optional `variables` and `values` entries containing `variable`, `date`, and `value`. Each entry uses its own calendar month. The zone resolves to a canonical stored artifact; no arbitrary source path is accepted. |

Forecasts are provider snapshots, not calibrated event probabilities. Climate classification requires a stored baseline with suitable coverage; precipitation bands use the wet-day distribution. `climate_normals` reports `confidence: null`.

## Satellite imagery and vegetation

Implementations: [STAC tools](../src/tools/stac.py), [vegetation and change tools](../src/tools/change.py), and [climate ingestion](../src/tools/climate.py).

| Tool | Purpose | Essential inputs and conditions |
| --- | --- | --- |
| `stac_search` | Find satellite scenes in the configured STAC catalogue. | `bbox`, `datetime`, `collections`; narrow results with `cloud_cover_max` and `limit`. Results include item identifiers, footprints, metadata, and asset URLs. The server caps the result count. |
| `stac_download` | Resolve asset URLs for selected bands of a scene. | `item_id`, `collection`, `bands` from a search result. Despite its name, this tool returns URLs without downloading files. `bbox` is echoed without cropping; `resolution_m` does not resample assets. |
| `ndvi_profile` | Compute a Sentinel-2 NDVI time series for a geometry. | `geometry`, `start_date`, `end_date`; optional `interval_days` and `cloud_cover_max`. `interval_days=0` keeps every available observation within the search cap. Needs accessible imagery and usable pixels. |
| `vegetation_stress` | Flag low NDVI or a sharp decline over a lookback window. | `geometry`; tune `lookback_days`, `ndvi_threshold`, `drop_threshold`, and `threshold_mode`. Percentile mode derives its threshold from the observed window and falls back to the static threshold when observations are insufficient. |
| `change_detect` | Compare mean vegetation greenness around two dates. | `bbox`, `date_before`, `date_after`, optional `confidence_threshold`. Searches within ±30 days of each date and returns at most one bbox-level feature. `change_types` and `min_change_area_ha` currently have no effect. |
| `ndvi_ingest` | Acquire a zone's Sentinel-2 NDVI zonal statistics as a reusable table. | `zone_id`, `start_date`, `end_date`; `monthly=true` selects a least-cloudy scene per month, while `false` retains scenes up to the search cap. `write_output` defaults to `true`. |

Cloud cover, scene availability, masking, and observation spacing constrain these results. Stress and change scores are heuristics; the change tool does not delineate per-pixel change polygons. NDVI describes vegetation greenness and does not directly measure carbon storage or biodiversity.

## Spatial queries

Implementation: [spatial tool](../src/tools/spatial.py).

| Tool | Purpose | Essential inputs and conditions |
| --- | --- | --- |
| `spatial_query` | Select stored GeoParquet features using a geometry and spatial predicate. | `source`, `geometry`; `operation` is `intersects`, `within`, `contains`, or `distance`. Distance queries use `distance_m`; `select_columns` and `limit` bound results. Optional `evidence_context` enables a stricter bounded evidence mode. |

For native geometry columns, supported declared coordinate systems are WGS84 (`EPSG:4326` or `OGC:CRS84`) and Lambert-93 (`EPSG:2154`); returned geometries use longitude/latitude. Lambert-93 query vertices are transformed without edge densification. Distance uses Web Mercator (`EPSG:3857`), so it carries projection distortion. WKB blob predicates compare in the source coordinates; do not assume the same CRS handling for every storage representation.

## Patterns and experimental alerts

Implementations: [pattern tools](../src/tools/patterns.py) and [alert tool](../src/tools/pattern_alert.py). Mining requires the optional `patterns` dependency group in [pyproject.toml](../pyproject.toml), including so4gp, and suitable stored tables.

| Tool | Purpose | Essential inputs and conditions |
| --- | --- | --- |
| `discover_patterns` | Find attributes that tend to increase or decrease together, including temporal gradual patterns. | `source` identifies a stored table; `source="auto"` uses `tenant_id` and `space_id`. Choose `algorithm`, `min_support`, and optional `columns`. Temporal `tgrad` needs `target_column` and a time axis through `order_by`. `write_output` defaults to `true`. |
| `discover_patterns_cv` | Check pattern recurrence across zone folds. | At least five unique `zone_ids`, with `sources` or a `source_template`; tune `n_folds`, `algorithm`, `columns`, and optional period bounds. Needs already materialized per-zone tables. `write_output` defaults to `false`. |
| `pattern_alert` | Combine a validated pattern library and forecast into an experimental explanatory score. | In the default `live` mode: `zone_id`, an accessible `patterns_source`, and optional forecast and pattern thresholds. Needs a compatible validated library and reachable forecast provider. Additional replay modes have explicit input contracts; inspect the schema before using them. |

Patterns describe associations and recurrence, not causation. Cross-zone validation is an engineering procedure whose result still needs scientific assessment. Alert scores are explanatory indices, not calibrated confidence or event probabilities; a missing or unsuitable library can yield no conclusion. Scheduling calls and sending notifications are responsibilities of the consuming application.

## Solar acquisition pilot

Implementation: [solar tool](../src/tools/solar.py), with [pilot input models](../src/solar_pilot.py) and [source contracts](../src/solar_contracts.py).

| Tool | Purpose | Essential inputs and conditions |
| --- | --- | --- |
| `solar_portfolio_collect` | Validate structured solar queries or collect bounded provider profiles. | `action` (`validate` or `collect`), 1–2 `sites`, `source`, two consecutive UTC dates (`start_date`, `end_date`), and `variables`; source-specific `model`, `temporal_resolution`, tilt, or azimuth may apply. Each site includes `site_id`, `latitude`, and `longitude`. |

This tool is experimental. Validation performs no acquisition. Collection is disabled by default and requires `SOLAR_MCP_ENABLED` plus a source-specific entry in `SOLAR_MCP_SOURCE_ACCESS_PROOFS`. It uses a separate worker with a 30-second deadline and at most two simultaneous collections. Returned profiles include checksums; the caller owns durable storage and source access authorization. Collection alone does not establish photovoltaic yield or investment suitability.

## First calls

These are MCP `tools/call` parameter objects using native tool names. Submit one through your connected client's tool-call interface; the client supplies the JSON-RPC envelope. Inputs were checked against the declared contracts and registered tool schemas. They require live providers and are examples, not recorded observations.

Get a three-day forecast without persistence:

```json
{
  "name": "forecast_query",
  "arguments": {
    "latitude": 48.85,
    "longitude": 2.35,
    "forecast_days": 3,
    "provider": "auto",
    "write_output": false
  }
}
```

Find a few Sentinel-2 scenes in a small area:

```json
{
  "name": "stac_search",
  "arguments": {
    "bbox": [2.30, 48.80, 2.40, 48.90],
    "datetime": "2025-06-01T00:00:00Z/2025-06-30T23:59:59Z",
    "collections": ["sentinel-2-l2a"],
    "cloud_cover_max": 20,
    "limit": 3
  }
}
```

Request an NDVI profile for a polygon:

```json
{
  "name": "ndvi_profile",
  "arguments": {
    "geometry": {
      "type": "Polygon",
      "coordinates": [[[2.30, 48.80], [2.31, 48.80], [2.31, 48.81], [2.30, 48.81], [2.30, 48.80]]]
    },
    "start_date": "2025-06-01",
    "end_date": "2025-06-30",
    "interval_days": 16,
    "cloud_cover_max": 20
  }
}
```

Check returned metadata, missing observations, provider fallback, and any structured `error` before drawing a conclusion or chaining another call. For stored workflows, also read `geoai://contracts/dataset-manifest-v1` and preserve the dataset's provenance and access boundaries.
