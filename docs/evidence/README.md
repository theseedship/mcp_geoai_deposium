# FIERCE — Public evidence index

This is the public technical evidence annex supporting the DEPOSIUM SPACE
Milestone 2 project report. The full milestone report and its confidential
commercial supporting materials are separate documents, not published here.

Use the pages in this order:

1. Read the [technical KPI annex](fierce-kpi-evidence.en.md) for the explanation of
   each KPI, reported status, supporting links and limits. The
   [Word edition](fierce-kpi-evidence.en.docx) contains the same public content.
2. Use [the objective table below](#choose-an-objective) to go directly to an
   implementation or a cited proof.
3. Open an individual excerpt in [the public file index](#public-file-index), or
   [the source map](source-map.json) for its provenance and hashes.

For a new technical recording in DEPOSIUM, use the
[English chat demonstration guide](fierce-chat-demo.en.md) or its
[French edition](fierce-chat-demo.fr.md): copyable messages,
expected outputs, three-pilot coverage and the distinction between recorded
application behavior and scientific evidence. This guide prepares a recording;
it is not itself proof that a new recording or qualification has taken place.

**KPI 1 clarification — 30 October 2026:** the milestone report dated
29 October records **5/5 commercial-validation service components completed**.
**0/5 is the starting baseline, not the reported achieved result.** This public
annex summarizes that declaration; it does not independently audit the
confidential business materials. The five components include the interview
study and **fundraising preparation**, not a completed investment round. See
[KPI 1: metric, reported result and evidence boundary](fierce-kpi-evidence.en.md#3-kpi-1--d1--market-study-and-commercial-readiness).

The technical annex retains the dated quantitative findings, adapted scopes,
negative results and maturity limits.
[Architecture](../architecture.md) explains the independent server;
[the roadmap](../roadmap.md) separates delivered capabilities and remaining work.

This collection contains only the report's cited technical excerpts and bounded
acceptance/integration records. It is not the private project documentation tree
or a copy of every underlying dataset. [The source map](source-map.json) records
original-source and curated-extract hashes, evidence-source commit
`d4bea08f41b20b337ee8152704c7ed5231876072`, and the distinct report-source commit
`fe21e58b8ee5baaa800e41601248563310825fa5`. Rehosting dated evidence does not
independently reproduce its experiments or certify its acceptance.

## Choose an objective

| Objective | Implementation | Public evidence and assessment boundary |
| --- | --- | --- |
| KPI 1 / D1: commercial validation | Five commercial-validation service components; confidential business materials referenced in the milestone report. | **5/5 reported completed** in the 29 October milestone report; 0/5 is the baseline. [Metric and evidence boundary](fierce-kpi-evidence.en.md#3-kpi-1--d1--market-study-and-commercial-readiness). |
| KPI 2 / WP0: real-data comparison | [CPU so4gp adapter](../../src/patterns/so4gp_adapter.py). | [Comparison and reproduction](proofs/proof-26.md); 20 shared identities, explained differences; no general superiority claim. |
| KPI 2 / WP1: native extraction and integration | [Pattern tools](../../src/tools/patterns.py), [packaged libraries](../../src/data/pattern-library-2026-10-15/package). | [Three-pilot capture](proofs/proof-23.md): 418 records, all unqualified. |
| KPI 2 / WP2: robustness and confidence | [Held-out evaluation](../../src/patterns/heldout_validation.py). | [Negative evaluation](proofs/proof-22.md), [scope amendment](proofs/proof-15.md), [NDVI ablation](proofs/proof-21.md); no calibrated confidence or positive primary predictive gain. |
| KPI 2 / WP3: temporal and spatial dimensions | [Pattern adapter](../../src/patterns/so4gp_adapter.py), [cross-validation](../../src/patterns/cross_validation.py). | [Q3 recurrence](proofs/proof-12.md); a native lag is not independently demonstrated forecast lead-time skill. |
| KPI 2 / WP4: callable software | [Server](../../src/server.py), [package metadata](../../pyproject.toml), [tool guide](../../reference/mcp-tools.md). | [Packaging/acceptance](proofs/proof-02.md), [upstream trace](proofs/proof-25.md); distinguish historical acceptance and current pinned 1.1.2 profile. |
| KPI 3 / B1–B4: geospatial pipeline | [Zones](../../zones/zones.geojson), [climate/NDVI tools](../../src/tools/climate.py). | [ERA5 coverage](proofs/proof-09.md), [NDVI evidence](proofs/proof-03.md), [130-row join](proofs/proof-04.md). |
| KPI 3 / B5–B6: forecasts and analysis | [Forecast tools](../../src/tools/forecast.py), [alert tools](../../src/tools/pattern_alert.py), [historical adapter](../../src/b6_historical.py). | [Forecast matrices](proofs/proof-05.md), [bounded B6 reception](proofs/proof-06.md); model substitution and explanatory-index limits remain explicit. |
| KPIs 4 / 7: indicators across three pilots | [Joined climate/NDVI contracts](../../src/contracts.py). | [Joint mining](proofs/proof-23.md), [authorized conversational readback](proofs/proof-11.md); three Copernicus indicators, not three independent satellite measurements. |
| KPI 8: pollution | No direct reduction greater than 5% claimed by the plan. | No measured reduction is inferred from alerts. |
| KPI 9: resource efficiency | Decision-support use cases. | The 5–10% target is projected, not measured; a quantified scenario remains separate. |
| KPI 10: critical raw materials | Outside the stated scope. | No recycling, recovery or extraction claim. |
| Maturity and integrated demonstration | Independent server plus consuming-application integration. | [N1 readback](proofs/proof-11.md), [scheduled bulletin](proofs/proof-10.md), [B6 demonstration](proofs/proof-06.md). Integrated demonstrator: TRL 6 self-assessment, moderate confidence; TRL 7 unproven. Novel predictive alerting: approximately TRL 4 at most. |

## Public file index

The following files are the specifically cited pieces. Technical excerpts record
what their source documents reported. Acceptance records report internal scoped
decisions; integration records report application changes. None is an external
FIERCE approval or a newly executed reproduction.

| File | Evidence | Kind |
| --- | --- | --- |
| [proof-01](proofs/proof-01.md) | Historical software and delivery overview | technical-extract |
| [proof-02](proofs/proof-02.md) | Adopted delivery scoreboard and packaging | technical-extract |
| [proof-03](proofs/proof-03.md) | NDVI shipped-code evidence | technical-extract |
| [proof-04](proofs/proof-04.md) | Unified 130-row pilot data | technical-extract |
| [proof-05](proofs/proof-05.md) | Forecast collection and comparison matrices | technical-extract |
| [proof-06](proofs/proof-06.md) | Bounded historical B6 reception | technical-extract |
| [proof-07](proofs/proof-07.md) | S3 baseline materialization | technical-extract |
| [proof-08](proofs/proof-08.md) | Delivery and acceptance checkpoint | technical-extract |
| [proof-09](proofs/proof-09.md) | Full-panel ERA5 coverage | technical-extract |
| [proof-10](proofs/proof-10.md) | Scheduled bulletin and recipient observation | technical-extract |
| [proof-11](proofs/proof-11.md) | Authorized conversational data readback | technical-extract |
| [proof-12](proofs/proof-12.md) | Q3 cross-zone recurrence | technical-extract |
| [proof-13](proofs/proof-13.md) | Manual staging acquisition and classification | technical-extract |
| [proof-14](proofs/proof-14.md) | Consuming-application integration boundary | technical-extract |
| [proof-15](proofs/proof-15.md) | Historical catalogue scope amendment | technical-extract |
| [proof-16](proofs/proof-16.md) | Application chat publication | technical-extract |
| [proof-17](proofs/proof-17.md) | Application bulletins and monitoring | technical-extract |
| [proof-18](proofs/proof-18.md) | Guided interface specification and evidence | technical-extract |
| [proof-19](proofs/proof-19.md) | S3-compatible storage contract | technical-extract |
| [proof-20](proofs/proof-20.md) | Minimum technical scope correspondence | technical-extract |
| [proof-21](proofs/proof-21.md) | Negative NDVI predictive ablation | technical-extract |
| [proof-22](proofs/proof-22.md) | Held-out evaluation and negative results | technical-extract |
| [proof-23](proofs/proof-23.md) | Native three-pilot pattern libraries | technical-extract |
| [proof-24](proofs/proof-24.md) | Pattern qualification limits | technical-extract |
| [proof-25](proofs/proof-25.md) | so4gp upstream source trace | technical-extract |
| [proof-26](proofs/proof-26.md) | Reproduced WP0 Copernicus comparison | technical-extract |
| [record-01](proofs/record-01.md) | O1/#19 acceptance | acceptance-record |
| [record-02](proofs/record-02.md) | O2 acceptance | acceptance-record |
| [record-03](proofs/record-03.md) | O3 acceptance | acceptance-record |
| [record-04](proofs/record-04.md) | O4/#22 acceptance | acceptance-record |
| [record-05](proofs/record-05.md) | decision #23 | acceptance-record |
| [record-06](proofs/record-06.md) | #193 acceptance | acceptance-record |
| [record-07](proofs/record-07.md) | #30 acceptance | acceptance-record |
| [record-08](proofs/record-08.md) | MCPs #962 | integration-record |
| [record-09](proofs/record-09.md) | #963 | integration-record |
| [record-10](proofs/record-10.md) | Solid #919 | integration-record |
| [record-11](proofs/record-11.md) | #920 | integration-record |
| [record-12](proofs/record-12.md) | #922 | integration-record |
