# Technical evidence 21

Source: `docs/research/ndvi-predictive-ablation-2026-10-25/README.md` at `d4bea08f41b20b337ee8152704c7ed5231876072`.

Original SHA-256: `393071b68411832bd704bddfca9e0899f792c0e2af9b000216f553533ca5c5a3`.

This is a curated technical extract from the dated source, with private navigation and operational identities omitted. It records documented evidence; publication does not independently rerun or certify the experiment.

## Selected source excerpts

**No predictive gain is demonstrated on the primary Fos case.** Adding the
frozen native GRAANK pair-interaction features increases mean absolute error
from **0.02263 to 0.02354 NDVI**, about **4.0%**. Simply reusing the origin month's
retained NDVI gives **0.02237**. Lower error is better.

This is a measured, reproducible negative result for one explicit recipe. It
is not evidence that every so4gp method fails, nor that TGrad cannot ever help
prediction. The experiment tests whether selected gradual associations improve
one small predictor of the **next month's retained NDVI**, using only past inputs.

Native so4gp discovers combinations of variables that increase or decrease
together. Its output is a pattern and its support, not a future numeric NDVI.
Here, an experimental bridge turns patterns mined on training data into extra
features of a fixed ridge regression. Comparing the same regression with and
without those features tests their added predictive contribution.

| Design element | Fixed choice |
| --- | --- |
| Primary case | Fos-sur-Mer: 22 training targets, 24 evaluation targets |
| Secondary checks | Lez and Montpellier Métropole: 20 training / 16 evaluation each |
| Target | Archived retained NDVI acquisition in the next calendar month |
| Information at origin | Origin NDVI, complete weather summaries for the origin and preceding months, known calendar month |
| Training / evaluation | Target months in 2021–2022 / 2023–2024, respectively |
| Mining | Actual CPU so4gp 1.1.2 GRAANK, all 78 pairs of thirteen drivers plus the training target, support > 0.50 after independent recount |
| Comparisons | Persistence, seasonal median, previous-year observation, raw regression, regression with ranks, regression with native interactions, matched generic interactions |

The same eligible target months are scored for every arm. No future weather,
actual target-acquisition day, or evaluation NDVI label enters model fitting,
mining, scaling or feature selection. Missing calendar observations are excluded,
not interpolated. The actual acquisition day determines the reported horizon,
which varies within the next month; this is not a fixed 30-day forecast.

All values below are evaluation MAE, in dimensionless NDVI units. The
full measured report also gives RMSE, all seven arms, year-by-year
comparisons and the descriptive uncertainty interval. Exact numbers and selected
patterns are in summary.json; all 826 training-diagnostic and
evaluation predictions are in predictions.csv.

| Statement | Evidence and limit |
| --- | --- |
| The native library actually mines usable patterns | All 234 three-column native calls and their outputs are retained; selected supports are independently recounted. This checks execution and counting, not scientific causality. |
| Future NDVI can now be evaluated against simple references | A chronological evaluation harness and paired predictions exist. The measured recipe does not improve the primary reference. |
| Adding these native features is useful | **Not demonstrated** on the primary case; small secondary changes do not establish general value. |
| TGrad predicts five-day stress, or B6 is calibrated | **Not tested.** This is GRAANK on explicit calendar-lagged inputs, not the native TGrad fuzzy day-lag estimator, a labelled stress event, or a calibrated probability. |
| A FIERCE scientific lot can be received | **No acceptance decision follows from this experiment.** Existing issue-specific criteria and partner review remain separate. |

## Interpretation boundary

Internal engineering reception is distinct from external programme approval. Historical or synthetic qualification does not establish production activation, causal inference, calibrated probabilities or general predictive superiority. Preserve the source date and scope when using these excerpts.
