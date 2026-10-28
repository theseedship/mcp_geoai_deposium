# Technical evidence 24

Source: `docs/research/pattern-qualification-2026-10-16/README.md` at `d4bea08f41b20b337ee8152704c7ed5231876072`.

Original SHA-256: `28ca85178a2d22ee3677571bc8adf416ba0f861b498251a1f05ae7f0d9cc2290`.

This is a curated technical extract from the dated source, with private navigation and operational identities omitted. It records documented evidence; publication does not independently rerun or certify the experiment.

## Selected source excerpts

**Evidence mapping supplied; scientific review and Deposium agreement pending.**
This audit accounts for all **418** records in native package
`2026.10.15-native-pilots.1`. It links applicable existing [internal record] results and
records why the remaining rules cannot inherit them. No experiment, mining,
observation rescoring, threshold change or package rewrite was performed.

The J1 real-case assessment answers the contributor/maintainer's subsequent
request to identify a real eligible level-2 case or explain the missing
validation. It assesses actual Lez rules, preserves their local and pooled
results, and identifies a separate historical Q3 explanatory candidate without
claiming it is already qualified. J1's minimum explanatory scope does not
automatically require calibrated probability or a nonzero predictive lag.

The broader explanatory rule shortlist now identifies
three actual Fos-sur-Mer five-day TGrad recurrence records, a shorter-lag
alternative, Lez local association evidence and supplementary Seoul/Mombasa
matches. All 179 archived Q3 native records were directly decoded from their
authenticated Parquets and compared field for field with the CSV exports.
The shortlist distinguishes supported historical explanations from accepted
operational J1 qualification; no current package state was promoted.

| Packaged records | Link to the original 0.70 study | Result |
|---|---|---|
| 10 GRAANK | NDVI-as-consequent projections to six temporal lag-0 candidates | All six pooled evaluation results fail; two Lez local results pass |
| 1 GRAANK | Fos wind-speed-positive / NDVI-positive projection to a spatial candidate | All five folds fail; stability 0 |
| 12 GRAANK | No NDVI in the itemset | Outside the study's target scope |
| 395 TGrad | Nonzero calendar-day lags | No faithful equivalence to observation-step shifts established |
| **418 total** | **11 explicit itemset projections; 407 without an applicable rule link** | **0 records acquire accepted qualification** |

Native GRAANK records have an empty consequent. Assigning NDVI as the consequent
is an explicit **directed projection of an undirected itemset**, not an already
stored native rule. Signed items must all match exactly; extra items are never
dropped, inverse directions are never collapsed, and subset/superset matches
are not used. All eleven projections have positive NDVI direction and lag 0.

| Native scoped ID | Projected climate antecedent, NDVI `+` consequent | Original evidence | Pooled result | Producing zone result |
|---|---|---|---|---|
| `bassin_du_lez/graank/0` | wind_speed `-` | Temporal, lag 0 | Fail | Fail |
| `bassin_du_lez/graank/1` | solar_radiation `-`, temperature_2m `-` | Temporal, lag 0 | Fail | **Pass** |
| `bassin_du_lez/graank/4` | evapotranspiration `-`, solar_radiation `-`, temperature_2m `-` | Temporal, lag 0 | Fail | **Pass** |
| `fos_sur_mer/graank/0` | wind_speed `+` | Spatial, Q3 | Fail; 0/5 passing folds | Fail |
| `fos_sur_mer/graank/5` | solar_radiation `-`, temperature_2m `-` | Temporal, lag 0 | Fail | Fail |
| `fos_sur_mer/graank/6` | evapotranspiration `-`, humidity `+` | Temporal, lag 0 | Fail | Fail |
| `fos_sur_mer/graank/8` | evapotranspiration `-`, solar_radiation `-`, temperature_2m `-` | Temporal, lag 0 | Fail | Fail |
| `montpellier_metropole/graank/0` | wind_speed `-` | Temporal, lag 0 | Fail | Fail |
| `montpellier_metropole/graank/1` | evapotranspiration `-`, temperature_2m `-` | Temporal, lag 0 | Fail | Fail |
| `montpellier_metropole/graank/2` | solar_radiation `-`, temperature_2m `-` | Temporal, lag 0 | Fail | Fail |
| `montpellier_metropole/graank/3` | evapotranspiration `-`, solar_radiation `-` | Temporal, lag 0 | Fail | Fail |

## Interpretation boundary

Internal engineering reception is distinct from external programme approval. Historical or synthetic qualification does not establish production activation, causal inference, calibrated probabilities or general predictive superiority. Preserve the source date and scope when using these excerpts.
