# Exploratory NDVI prediction: a native so4gp feature ablation

**Protocol written before mining, model fitting or predictive scoring on
25 October 2026.** Input schemas, identities and eligibility counts have been
inspected. Historical outcomes and earlier association studies are already
known: this is an exploratory retrospective experiment, not an untouched
confirmatory trial. The machine-readable settings are in `protocol.json`.

## Question and scope

Does a small fixed regression gain predictive information from multivariate
features selected by native **so4gp 1.1.2 GRAANK**, beyond the same regression
using the same raw inputs and their univariate ranks?

This tests one supervised use of gradual-pattern discovery. Calendar lags are
explicitly constructed before mining. It does **not** validate native TGrad's
fuzzy day-lag estimator, the existing 418-record library, an event probability,
a weather model, or the entire library's usefulness. A negative outcome must
not be generalized beyond the tested target, inputs and feature construction.

This is separate from PRs #313/#317's mining-comparison preparation and #316's
historical explanatory-case preparation. No new application or MCP interface
is needed for this offline experiment. Scientific/contractual acceptance is
unchanged.

## Frozen population and target

- Primary site: **Fos-sur-Mer**. Lez and Montpellier Métropole are secondary
  descriptive checks; their overlapping footprints prevent treating them as
  independent replications or pooling them into independent evidence.
- Training targets: calendar months in **2021–2022**. Evaluation targets:
  **2023–2024**. Fit separately per site, with no evaluation-driven tuning.
- Origin: the end of UTC calendar month `t`. Target: the archive's retained
  NDVI observation in calendar month `t+1`, identified by its actual acquisition
  date. The future acquisition day is used for reporting only, never a feature.
- Require an observed finite NDVI in the origin and target months and complete
  finite daily climate data in months `t` and `t−1`. No missing-month bridging,
  NDVI interpolation, or cross-zone substitution is permitted.
- Report all candidate origin months and exclusion reasons. All arms score
  exactly the same eligible targets. Expected counts from metadata inspection:
  Fos 22 training/24 evaluation; Lez 20/16; Métropole 20/16.
- An origin in December 2022 may predict January 2023: its predictors are past
  information, while that target must never enter training. Conversely, a
  December 2022 *training target* may use only its November-origin predictors.

## Inputs and availability assumptions

Use the supplied original B3 and ERA5 ZIPs, authenticated before reading members:

| Archive | SHA-256 |
| --- | --- |
| `geoai-b3-ndvi-pilot-3zones-20260822.zip` | `0cd6dd981bb78916026fc1585cd856ba81e89deeaea24158973513bf6ad5526c` |
| `geoai-era5-holdout-2021-2025-handoff.zip` | `3a03059e6a0f75e77fb355b68e2535b0e841e2bb460ac0d2574a0104c7f5937d` |

Read only the three declared sites. Keep source member hashes, acquisition
dates, units, input schemas and source coordinates. Snapshot bounded archive
bytes before verification/loading; write selected Parquets only to a private
temporary directory. No provider collection is part of this experiment.

The raw NDVI archive contains 254 observations from late 2016 through 2024;
the primary period uses 130. The earlier observations are not silently added
to the primary training population. ERA5 has daily rows for 2021–2025; future
months must not enter features. Its original daily values are normalized:

| Variable | Monthly aggregation, separately at `t` and `t−1` |
| --- | --- |
| Temperature at 2 m | Mean, °C |
| Precipitation | Sum, mm |
| Humidity | Mean, % |
| Reference evapotranspiration | Sum, mm |
| Wind speed | Maximum of daily maxima, m/s |
| Solar radiation | Mean of daily mean fluxes, W/m² |

Each month must contain every calendar day exactly once, with finite values
for all six variables. Negative precipitation/evapotranspiration/radiation or
wind and out-of-range humidity/NDVI must be reported and rejected, not repaired.

The ERA5 and NDVI locations differ. In particular Fos uses ERA5 point
43.445/4.92 and NDVI sample point 43.45455/4.89795. Preserve the archive's
geographic provenance; do not claim site equivalence, global transfer, or a
new scientifically ratified alignment. Monthly joins here replace neither the
existing daily alignment decision nor the package's original provenance.

Publication timestamps for retrospective ERA5 and NDVI processing are not
available. Month-end availability is an explicit simulation assumption, not
an attested operational data feed. Forecasts use observed-origin NDVI and
retrospective climate histories; they do not use actual future weather or the
2025 forecast archive. Earlier full-period native libraries are not reused.

## Predictors and seven comparison arms

The 13 numeric origin drivers are current observed NDVI and the six monthly
climate summaries at each of `t` and `t−1`. Add the sine and cosine of the
**known next calendar month's** month-of-year to each learned model.

1. **Persistence:** origin NDVI.
2. **Training seasonal median:** median observed NDVI for the target month of
   year, using only 2021–2022 observations; training global median fallback.
3. **Previous-year seasonal value:** retained observation for the target month
   of the previous year, only if observed before the origin; otherwise arm 2.
   Earlier evaluation observations may become known history at later origins.
4. **Raw ridge:** the 15 predictors above.
5. **Rank-control ridge:** arm 4 plus a training-fitted empirical rank for
   each of the 13 numeric drivers.
6. **Native-so4gp ridge:** arm 5 plus up to eight mined interaction features.
7. **Generic-interaction ridge:** arm 5 plus label-independent interactions
   matched to arm 6's count, arities and sign counts; seed `20261025`.

All learned arms use ordinary ridge with fixed alpha **10**, an unpenalized
intercept, and training-only mean/std scaling for **every** feature, including
ranks and interactions. Drop columns constant in training deterministically.
Predict the NDVI level, not a retrospectively tuned event threshold. Clip all
predictions to the physical interval [−1, 1], retain unclipped learned values,
and report clipping. No hyperparameter/algorithm/seed search is permitted.

## Native mining and interaction construction

Run the actual pinned CPU GRAANK through the existing adapter, once per site's
training matrix: the 13 numeric drivers plus `target_next_ndvi`. This target
column contains training labels only. Dates, calendar encoding and evaluation
rows are excluded from the miner. Settings: support 0.50, strict comparisons
(`eq=False`), default native GRAANK search. Bound each native call at 60 seconds
and confine its working directory. Preserve exact native output and normalize
through the existing adapter without modifying production code.

Retain itemsets containing the target and at least two antecedent drivers.
Canonicalize simultaneous inverses to a positive target sign, deduplicate, and
independently recount strict pair support on training rows. Record native
rounding; a discrepancy larger than 0.001 is a reported mining failure, not
silently corrected evidence. Retain exact support strictly above 0.50. Sort by
reported native support descending, then canonical lexical signature; take at
most eight. Publish all raw candidates and selection/exclusion reasons.

For driver `x`, empirical rank is
`(#training values < x + 0.5 × #training values == x) / n_training`.
Thus values below/above the training range map to 0/1. Fit ranks only on
training rows. A signed antecedent uses `rank(x)` for `+` and `1−rank(x)` for
`−`; its feature is the minimum of its signed ranks. The fitted regression
learns how these candidate features relate to future NDVI.

The generic comparator samples driver assignments and signs without target
values, preserving each selected interaction's arity and number of positive
signs. Use the single fixed seed, record coincidences with mined interactions
and effective feature identities, and never choose a favorable comparator.
This controls one ordinary interaction construction, not all possible feature
engineering. Empty native selection is a valid result: arms 5–7 must coincide.

## Metrics and interpretation fixed before execution

- Primary contrast: Fos **MAE(rank control) − MAE(native-so4gp)** on the same
  24 eligible evaluation origins. Positive means a measured incremental gain
  for this feature package; zero/negative means no measured gain.
- Report every arm's MAE and RMSE, actual/forecast NDVI, absolute error,
  source dates, elapsed horizon, monthly coverage and clipping. Include yearly
  metrics and contrasts against the generic comparator and all references.
- For the contiguous primary series, report an exploratory paired circular
  block-bootstrap interval (3 months/block, 2,000 draws, seed 42). It conditions
  on already fitted models and an inspected holdout, assumes resampling is
  meaningful, and omits model-selection uncertainty. It is **not** a calibration
  or confirmatory significance claim. Secondary series with calendar gaps get
  no such interval; retain their individual error differences instead.
- All results, failures and negative effects are retained. No changing support,
  threshold, target, split, site, alpha or feature count after seeing scores.
- Even a positive primary contrast does not establish native TGrad forecasting,
  causal effects, a probability of stress, broad climatic generalization, or
  scientific/FIERCE reception. A gain over rank control but not the generic
  interaction comparator is weaker evidence for the value of mining itself.
- The report distinguishes observed effect, uncertainty, implementation/data
  limits and proposed next tests. Do not promote an explanatory index to a
  forecast or interpret a source/transport check as predictive skill.

## Verification and deliverables

Before a real run, test calendar joins and origin cutoffs, held-out target
isolation, target-day independence, training-only CDF/scaling/mining/selection,
strict support counts and inverses, zero-feature fallback, constant columns,
and identical scored origins across arms. Tests use synthetic fixtures; the
real run must use actual native mining, with no fixture substitution.

Write one immutable output directory containing protocol/config hashes,
code/runtime/dependency identities, authenticated input hashes, raw native
outputs, selected features, fitted parameters, all forecasts and exclusions,
metrics and a factual report. Repeat into another directory to check numerical
reproducibility, retaining timing differences as such. Compare source identities
and predictive results; never overwrite the first run.
