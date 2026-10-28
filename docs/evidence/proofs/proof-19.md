# Technical evidence 19

Source: `docs/design/rustfs-storage-contract.md` at `d4bea08f41b20b337ee8152704c7ed5231876072`.

Original SHA-256: `9ea4ac67b87ea124a9ed27cdb54b9630409bb8b38ed143eeafdeb21d56ada8c4`.

This is a curated technical extract from the dated source, with private navigation and operational identities omitted. It records documented evidence; publication does not independently rerun or certify the experiment.

## Selected source excerpts

1. **Application data**: user uploads, parsed pages, tabular Parquet, chat
   artifacts, and optional patterns created from a tenant/space dataset.
2. **Scientific data**: climate and EO artifacts keyed by zone, time window,
   variable set, algorithm, and run provenance.

They can share the same RustFS bucket and DuckDB access layer, but they must not
share ambiguous path semantics. The distinction is not "basic vs advanced"
patterns. It is about ownership, lifecycle, query scope, and reproducibility.

Scientific patterns are the same family of so4gp outputs as application patterns,
but they answer a different question:

- App pattern: "what motifs exist in this tenant/space dataset?"
- Scientific pattern: "what climate or EO motif is reproducible for this zone,
  window, variable set, and validation regime?"

- deleting a scientific baseline with a user folder cleanup;
- treating a user-local exploratory pattern as a validated FIERCE pattern
  library;
- losing the source window and run parameters needed to reproduce a result.

  climate/era5/{zone_id}.parquet
  climate/normals/{zone_id}.parquet
  climate/normals/staging/{run_id}/{zone_id}.parquet
  climate/normals/staging/{run_id}/run.json
  climate/normals/rollback/{run_id}/{zone_id}.parquet
  climate/normals/rollback/{run_id}/{zone_id}.absent
  climate/normals/events/zones/{zone_id}/{seq}.json
  climate/normals/events/runs/{run_id}/{seq}.json
  climate/ndvi/{zone_id}.parquet
  climate/unified/{zone_id}.parquet
  climate/holdout/era5/{window_label}/{zone_id}.parquet
  climate/holdout/unified/{window_label}/{zone_id}.parquet
  climate/forecast/{zone_id}_{issued_date}.parquet
  climate/runs/{run_id}/manifest.json

| Prefix | Owner | Typical writer | Typical reader | Lifecycle |
|---|---|---|---|---|
| `raw/` | Deposium app | Solid, edge upload, agents | parsers, audit tools | user/tenant retention |
| `processed/` | Deposium app | MCPs, rust_worker, n8n | RAG/search/tools, GeoAI when scoped | user/tenant retention |
| `chats/` | Deposium app | Solid/MCPs | chat services | shorter retention possible |
| `exports/` | Deposium app | app/tools | users/operators | shorter retention possible |
| `climate/` | GeoAI | GeoAI climate tools | GeoAI, MCPs, agents | durable scientific baseline |
| `patterns/{zone_id}/` | FIERCE/science | GeoAI/Owuor producer | `pattern_alert`, validation, agents | durable scientific library |
| app `patterns/` | Deposium app | GeoAI `discover_patterns` | app workflows | user/tenant retention |

`raw/` is the original evidence. `processed/` is the queryable data plane. App
patterns are tenant/space-scoped and should not be consumed as scientific pattern
libraries unless they are promoted through an explicit validation step.

The canonical climate files are "latest usable artifact" paths. Reproducibility
comes from the run manifest, not from duplicating every Parquet on every run.
If audit pressure increases, add optional run-scoped copies under
`climate/runs/{run_id}/artifacts/...`.

`ndvi_ingest` writes `climate/ndvi/{zone_id}.parquet` and, until [internal record], **nothing read it**.
`climate_enrich_ndvi` re-derived NDVI from STAC on every call, which made [internal record]'s "depends on the
durable B3 materialization" a dependency the code could not express — and on a multi-year window it
was actively worse than the artifact already in the bucket: the B4 collector's STAC item cap applies
to the **whole window** while B3's applies **per month**, so streaming returned strictly less data.

## Interpretation boundary

Internal engineering reception is distinct from external programme approval. Historical or synthetic qualification does not establish production activation, causal inference, calibrated probabilities or general predictive superiority. Preserve the source date and scope when using these excerpts.
