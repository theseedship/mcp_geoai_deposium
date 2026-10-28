# Technical evidence 09

Source: `docs/demo/full-panel-baseline-validation-2026-06-23.md` at `d4bea08f41b20b337ee8152704c7ed5231876072`.

Original SHA-256: `34cd39708320be34a656550a22ebb20739ea96c2b6d553b3b525bffd0c60b4da`.

This is a curated technical extract from the dated source, with private navigation and operational identities omitted. It records documented evidence; publication does not independently rerun or certify the experiment.

## Selected source excerpts

Variables: the six locked ERA5 variables from terra-cognita section 5.1:
`temperature_2m`, `precipitation`, `humidity`, `evapotranspiration`,
`wind_speed`, `solar_radiation`.

| Zone | ERA5 rows | Min variable completeness | Normals | Months | Variables | Gate | ERA5 size | Normals size |
|---|---:|---:|---:|---:|---:|---|---:|---:|
| `bassin_du_lez` | 10,958 | 100.0% | 72 | 12 | 6 | pass | 170 KB | 4.9 KB |
| `fos_sur_mer` | 10,958 | 100.0% | 72 | 12 | 6 | pass | 173 KB | 4.9 KB |
| `kerguelen` | 10,958 | 100.0% | 72 | 12 | 6 | pass | 170 KB | 4.8 KB |
| `manaus` | 10,958 | 100.0% | 72 | 12 | 6 | pass | 140 KB | 4.8 KB |
| `marseille` | 10,958 | 100.0% | 72 | 12 | 6 | pass | 171 KB | 4.8 KB |
| `mombasa` | 10,958 | 100.0% | 72 | 12 | 6 | pass | 140 KB | 4.8 KB |
| `montpellier` | 10,958 | 100.0% | 72 | 12 | 6 | pass | 171 KB | 4.8 KB |
| `montpellier_metropole` | 10,958 | 100.0% | 72 | 12 | 6 | pass | 171 KB | 4.9 KB |
| `nairobi` | 10,958 | 100.0% | 72 | 12 | 6 | pass | 138 KB | 4.8 KB |
| `niamey` | 10,958 | 100.0% | 72 | 12 | 6 | pass | 133 KB | 4.8 KB |
| `rennes` | 10,958 | 100.0% | 72 | 12 | 6 | pass | 168 KB | 4.9 KB |
| `riyadh` | 10,958 | 100.0% | 72 | 12 | 6 | pass | 142 KB | 4.8 KB |
| `seoul` | 10,958 | 100.0% | 72 | 12 | 6 | pass | 171 KB | 4.8 KB |
| `svalbard` | 10,958 | 100.0% | 72 | 12 | 6 | pass | 147 KB | 4.7 KB |
| `tromso` | 10,958 | 100.0% | 72 | 12 | 6 | pass | 159 KB | 4.8 KB |

- The B2 -> B6 WMO baseline path now repeats across the full 15-zone B1 panel,
  not only the Montpellier/Kenya/French pilot subset.
- Every zone produced complete ERA5 daily coverage for 1991-2020 and a
  complete `12 months x 6 variables` climatological P10/P50/P90 normal table.
- The panel runner can handle long runs operationally: preflight, per-zone JSON
  summaries, incremental aggregate summaries, stop-on-rate-limit, resumable
  batches, and summarize-existing aggregation.
- For this workload, `--archive-chunk-days 10958` is the practical free-tier
  setting: it avoids turning one zone into about 30 yearly HTTP requests.

### Not established by this dated run

- Production S3/RustFS retention, scheduling, retry policy, or SLA behavior.
- Sentinel-2 NDVI backfill for the same 15 zones.
- Full B4 unified ERA5+NDVI table materialization for the full panel.
- O3 cross-zone pattern validation, 5-fold CV, or calibrated `pattern_alert`.

## Interpretation boundary

Internal engineering reception is distinct from external programme approval. Historical or synthetic qualification does not establish production activation, causal inference, calibrated probabilities or general predictive superiority. Preserve the source date and scope when using these excerpts.
