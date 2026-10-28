# Technical evidence 14

Source: `docs/design/deposium-mcps-geoai-handoff.md` at `d4bea08f41b20b337ee8152704c7ed5231876072`.

Original SHA-256: `2c96b4962daf1a2b13d6515e9e2bd76a716d5b22a9343dbd643909f4a9a7e8d3`.

This is a curated technical extract from the dated source, with private navigation and operational identities omitted. It records documented evidence; publication does not independently rerun or certify the experiment.

## Selected source excerpts

**Current readback — 7 October 2026:** the August catalog/deployment facts above
are historical. Use the ROADMAP
and pause/resume handoff for current owner
actions and recovery checks. The manual Site-Watch adapter and pure replay core
are shipped, with a bounded staging manual proof; labelled replay remains gated.
The separate bulletin addendum
records deployed rules controls, a natural legacy V1 staging occurrence and SMTP
acceptance, while live V2 rules/inbox and production activation remain unqualified.

| Resource | Purpose |
|---|---|
| `geoai://capabilities/tools` | 13 strict input/output schemas plus profile, persistence default, timeout class, and stability |
| `geoai://contracts/dataset-manifest-v1` | Normative structural schema, semantic rules, validation order, and fixture locations |
| `geoai://schemas/dataset-manifest-v1` | Structural JSON Schema only; insufficient on its own |
| `geoai://docs/integration-contract` | Cross-repository ownership, locator, evidence, and rollout rules |
| `geoai://docs/mcps-handoff` | This implementation handoff, packaged with the server for connector agents |

This table is the gateway handoff snapshot. The current cross-repository source
of truth is section 3 of the parent contract:
forecast providers own model execution and source terms; GeoAI owns adapters,
routing, normalized science, provenance and scientific mining; Deposium owns
business watch scheduling, predicates, lifecycle and delivery. RustFS/S3 is the
authoritative artifact store, DuckDB is the bounded query plane, and
`deposium_rust_worker` owns application file identities/manifests rather than
every GeoAI scientific write.

| Concern | Owner |
|---|---|
| Forecast model execution, source issue metadata, licence and attribution terms | Weather/data provider |
| Provider adapters, normalized scientific/spatial computation and upstream GeoAI tool contracts | GeoAI |
| Authentication, tenant/space authorization, logical locators, capability enforcement, evidence normalization | MCPs |
| Uploaded-file identity, bounded object fetch, formal GeoParquet validation, canonical app manifest publication | Rust worker |
| Original artifacts and manifests | RustFS/S3 |
| Rebuildable query/catalog index | MCPs; never the source of truth |

**Current checkpoint after MCP-1..4 and MCPs PR [internal record] (lot [internal record]):** MCPs pins and
profiles all 14 upstream capabilities. `ndvi_ingest` is catalogued as
`internal_write`, with `long` timeout and `partial` stability; it remains
non-public, non-discoverable, and disabled for facade callers. Six
`storage_free_read` facade tools are
public: `geo_stac_search`, `geo_stac_download`, `geo_change_detect`,
`geo_ndvi_profile`, `geo_vegetation_stress`, and `geo_forecast_query`.
Locator-backed describe and bounded spatial query are implemented behind the
tenant-read gate; derived, internal-write, science-shared, and experimental
profiles remain disabled for facade callers.

This is intentionally a gateway-only path. It does not assign weather-model
execution to GeoAI, create a second scheduler, or turn an experimental
`pattern_alert` score into calibrated confidence. Scientific archive-capture
jobs and the Deposium business watch scheduler remain separate. The registered
`climate_normals` tool ([internal record]) is not drawn here. This gateway-only diagram also
omits the later shipped `watch-input/v1` adapter and pure replay core; their current
qualification and remaining labelled-replay gates are linked in the readback above.

## Interpretation boundary

Internal engineering reception is distinct from external programme approval. Historical or synthetic qualification does not establish production activation, causal inference, calibrated probabilities or general predictive superiority. Preserve the source date and scope when using these excerpts.
