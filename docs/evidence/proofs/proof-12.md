# Technical evidence 12

Source: `docs/demo/o3-q3-cross-zone-validation-mesh-rustfs-2026-06-27.md` at `d4bea08f41b20b337ee8152704c7ed5231876072`.

Original SHA-256: `0a083111011b5089f6f5ba08258ff78b5213d63af17b035694f305ddb7aeec0e`.

This is a curated technical extract from the dated source, with private navigation and operational identities omitted. It records documented evidence; publication does not independently rerun or certify the experiment.

## Selected source excerpts

This note records the first O3 cross-zone validation run over the full-panel
2023 Q3 B4 holdout artifacts. It uses the `discover_patterns_cv` primitive
merged in PR [internal record] through the wrapper added for this proof:
`scripts/o3_q3_validation.py`.

- Repo: `deposium_geoai`

- Base commit: `8c3f8d7`

- B4 input labels: `2023_07`, `2023_08`, `2023_09`
- B4 source template: `climate/holdout/unified/{window_label}/{zone_id}.parquet`

- Period: `2023-07-01..2023-09-30`
- Algorithms: `graank`, `tgrad`
- Columns mined: `ndvi_mean`, `temperature_2m`, `precipitation`, `humidity`,
  `evapotranspiration`, `wind_speed`, `solar_radiation`
- Thresholds: `min_support=0.5`, `n_folds=5`, `min_validated_zones=1`

The wrapper read all expected Q3 B4 Parquets before mining. All 45 inputs were
readable through DuckDB/httpfs. One zone was excluded before mining because it
does not have enough Q3 rows for gradual-pattern mining.

| Algorithm | Eligible zones | Candidate patterns | Validated patterns | Runtime | Output |
|---|---:|---:|---:|---:|---|
| `graank` | 14 | 61 | 30 | `89.7 s` | `[storage artifact]` |
| `tgrad` | 14 | 384 | 149 | `278.9 s` | `[storage artifact]` |

| Pattern | Support | Stability | Zones validated | Confidence |
|---|---:|---:|---:|---|
| `temperature_2m+, solar_radiation+, evapotranspiration+` | `0.602` | `0.8` | 8 | `null` |
| `solar_radiation-, evapotranspiration-, humidity+` | `0.640` | `0.8` | 6 | `null` |
| `humidity-, ndvi_mean+` | `0.630` | `0.8` | 6 | `null` |

| Pattern | Lag | Support | Stability | Zones validated | Confidence |
|---|---:|---:|---:|---:|---|
| `ndvi_mean+, (evapotranspiration+) +1.0 months` | `30.42 d` | `0.632` | `1.0` | 10 | `null` |
| `ndvi_mean+, (wind_speed-) +1.0 months` | `30.42 d` | `0.617` | `1.0` | 10 | `null` |
| `ndvi_mean+, (humidity-) +1.0 months` | `30.42 d` | `0.654` | `1.0` | 9 | `null` |

`confidence` remains null. The original paragraph assigning “calibrated” scoring to a future pattern-alert stage is omitted because that historical wording is superseded: the accepted later implementation is an explanatory index, not calibrated confidence or event probability.

- The full-panel segmented Q3 B4 holdout artifacts are usable as the first O3
  input without re-downloading Sentinel-2 or ERA5 data.
- `discover_patterns_cv` can produce cross-zone validated pattern libraries on
  real mesh RustFS data for both contemporaneous GRAANK and temporal TGrad.
- TGrad emits lag-aware records over the Q3 panel, including 14-day and
  approximately 1-month lags that Palier 4 can inspect.
- The produced O3 libraries are materialized in the agreed scientific S3 scope
  under `climate/o3/holdout/q3_2023/`.
- The workflow preserves source-quality caveats instead of hiding them:
  `svalbard` July remains truncated and `kerguelen` is excluded for insufficient
  Q3 rows.

- It does not complete full O3. The ratified "validated" thresholds are still a
  kickoff decision, and ACO/GA/PSO benchmarking remains separate.
- It does not prove multi-year NDVI/unified backfill.
- It does not turn stability into calibrated product confidence.
- It does not clear `kerguelen`; that zone needs a wider or better-covered
  holdout slice before it can participate in O3 mining.

## Interpretation boundary

Internal engineering reception is distinct from external programme approval. Historical or synthetic qualification does not establish production activation, causal inference, calibrated probabilities or general predictive superiority. Preserve the source date and scope when using these excerpts.
