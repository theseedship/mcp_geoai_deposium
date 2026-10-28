# Technical evidence 07

Source: `docs/demo/canonical-full-panel-rustfs-2026-06-26.md` at `d4bea08f41b20b337ee8152704c7ed5231876072`.

Original SHA-256: `491320ee2060a69e050ef9b0768c1f8ae111b5e1ccb1b0c8f1d5d4f1cea7fdaa`.

This is a curated technical extract from the dated source, with private navigation and operational identities omitted. It records documented evidence; publication does not independently rerun or certify the experiment.

## Selected source excerpts

This note records the first complete canonical RustFS/S3 materialization of the
GeoAI B1 full-panel WMO ERA5/normals baseline.

- Repo: `deposium_geoai`
- Branch: `main`
- Canonical bucket: `[storage artifact]`
- Canonical manifest:
  `[storage artifact]`
- Window: `1991-01-01` through `2020-12-31`
- Zone set: `full-panel` (15 B1 zones)
- Variables: the six locked ERA5 variables from terra-cognita section 5.1
- Storage mode: `s3`

`n_outputs = 30` means one ERA5 Parquet and one normals Parquet for each of the
15 zones.

| Zone | ERA5 rows | Normals rows |
|---|---:|---:|
| `montpellier` | 10958 | 72 |
| `marseille` | 10958 | 72 |
| `nairobi` | 10958 | 72 |
| `mombasa` | 10958 | 72 |
| `niamey` | 10958 | 72 |
| `rennes` | 10958 | 72 |
| `tromso` | 10958 | 72 |
| `svalbard` | 10958 | 72 |
| `manaus` | 10958 | 72 |
| `kerguelen` | 10958 | 72 |
| `riyadh` | 10958 | 72 |
| `seoul` | 10958 | 72 |
| `montpellier_metropole` | 10958 | 72 |
| `bassin_du_lez` | 10958 | 72 |
| `fos_sur_mer` | 10958 | 72 |

- The canonical RustFS bucket now contains all 15 B1 zone ERA5 Parquet artifacts.
- The canonical RustFS bucket now contains all 15 B1 zone climatological normals
  Parquet artifacts.
- The canonical run manifest is final and reports `ok = true`.
- Downstream consumers can use the manifest as the stable handoff point instead
  of discovering partial paths by convention.
- For these 15 B1 zones and this WMO window, downstream checks should reuse the
  RustFS/S3 artifacts instead of redownloading Open-Meteo Archive data.
- A new location outside this panel still needs a first materialization run
  before it becomes reusable through the same manifest/artifact pattern.

| Location | Role |
|---|---|
| Git repository | Code, scripts, docs, and reproducible instructions. No generated data bundle is committed. |
| GitHub Release | Versioned downloadable snapshot for demos, onboarding, and isolated RustFS tests. |
| RustFS/S3 | Operational source of truth used by GeoAI and the Deposium mesh. |

- Ratify the production scheduling/retry policy.
- Wire the downstream consumer check into CI or the mesh operational path.
- Extend the same canonical storage path to the full NDVI/unified-table backfill.


## Interpretation boundary

Internal engineering reception is distinct from external programme approval. Historical or synthetic qualification does not establish production activation, causal inference, calibrated probabilities or general predictive superiority. Preserve the source date and scope when using these excerpts.
