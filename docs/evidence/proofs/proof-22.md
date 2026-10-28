# Technical evidence 22

Source: `docs/research/offline-heldout-validation.md` at `d4bea08f41b20b337ee8152704c7ed5231876072`.

Original SHA-256: `1b7189bddf944d3f2d4f0994898fbf85a0d74c13dc6efcbbc41d1f2e087175e1`.

This is a curated technical extract from the dated source, with private navigation and operational identities omitted. It records documented evidence; publication does not independently rerun or certify the experiment.

## Selected source excerpts

This experiment scores climate → `ndvi_mean` rules against observations. It
extends the offline research workflow by reusing Q3 authentication and table
loading from `o3_q3_offline_replay.py`. The recurrence replay, live MCP tool and
[internal record] results remain separate.

For a worked example of observation pairs, `J/A` confidence, training versus
held-out areas, and four-of-five-fold stability, read
Held-out CV explained, the plain-language companion
to this method.

**Integration checkpoint:** [internal record] merged as `ac9df6777e6c8b86dfa79d9ed847595f48f687be`.
Final synchronized-head CI passed 3,250 tests, skipped four and reached 88.50%
coverage. The 14 October readback
records independent reproduction and remaining scientific/ESP receipt. The run
manifests, original threshold and post hoc revision below retain their exact identities.

The search space exhaustively enumerates all nonempty signed itemsets over the
six climate variables, with either NDVI sign: 1,456 directed rules per lag.
This small, deterministic space avoids producer item ordering, producer
confidence and version-dependent mining heuristics. Each spatial fold selects
rules only when pooled **training** joint support is at least 0.50. A rule is
never selected because it appeared in a held-out zone.

For each zone, compare distinct observation pairs with strict inequalities.
Both orientations are examined; at most one can satisfy any antecedent.
Ties fail the affected condition. Let `N` be the number of unordered pairs,
`A` the count satisfying every fixed climate item and `J` the count also
satisfying the fixed NDVI consequent. Report support `J/N` and confidence
`J/A`. Pool `N`, `A` and `J` before division. Never compare observations from
different zones or average zone confidence ratios. Zero `A` gives null
confidence and `non_evaluable`; zero `N` also gives null support.

Each selected rule is scored on **all** held-out observations, including
failures. Every candidate has five fold rows; a fold where donors did not
select it is `not_selected`. Passing requires joint support `J/N` ≥ 0.50 and
conditional confidence `J/A` ≥ 0.50. These are independent observation counts,
not so4gp's native confidence descriptor. Stability is passing folds divided
by **five**, and spatial validation
requires at least four passing folds. Non-evaluable and unselected folds
cannot raise stability. These are proposed research bars, not scientific
acceptance.

The original run used confidence ≥ 0.70. The user requested 0.50 after seeing
those held-out results, so the revised run is an **exploratory sensitivity
analysis**, not independent confirmatory evidence. Since `0 < A ≤ N`,
`J/A ≥ J/N`: confidence ≥ 0.50 adds no restriction when support is already
≥ 0.50. This revision removes the stricter 0.70 confidence screen; it does
not change candidate selection, folds, support or stability requirements.

Temporal selection uses only 2021–2022; evaluation uses only 2023–2024.
Preprocessing is identity with finite numeric features: no learned scaling,
imputation or evaluation-derived parameters. The predeclared transformations
shift climate by 0, 1 or 2 **observation steps**, with NDVI at the later row.
Each period is split before shifting, so lagged windows crossing the boundary
are excluded. Removed windows and elapsed-day ranges are reported separately
for donors and evaluation. These lags are not fuzzy calendar-day TGrad lags;
the sparse sampling does not justify treating them as fixed day intervals.
Candidate inverse signs remain separate directed rules, even when their joint
counts coincide.

## Interpretation boundary

Internal engineering reception is distinct from external programme approval. Historical or synthetic qualification does not establish production activation, causal inference, calibrated probabilities or general predictive superiority. Preserve the source date and scope when using these excerpts.

## Original strict-screen results

On the 13 October evaluation, the original confidence bar 0.70 yielded 2 passing spatial fold entries, 40 failing entries and 8 not selected; 0/10 spatial candidates passed at least four of five folds. Temporal results were 6/52 passes and 46 failures. The exploratory 0.50 rerun was a separate changed threshold, not satisfaction of the original criterion. The six strict temporal passes represent three inverse pairs and fail locally at Fos; neither inverse orientation supplies an independent discovery. Geographic overlap, sparse observations and differing historical geometry prevent class-wide transfer claims.
