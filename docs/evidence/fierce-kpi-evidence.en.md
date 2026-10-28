# FIERCE — Evidence Report for DEPOSIUM GeoAI Deliveries

**Project:** DEPOSIUM SPACE / The Seed Ship — FIERCE_1OC_22

**Funding:** Funded by FIERCE through its Financial Support Programme. FIERCE receives European Union funding under Horizon Europe, Grant Agreement 101177496. The authors are responsible for the content; no EU or HADEA endorsement is implied.

**Document status as of:** 28 October 2026

**Purpose:** link each KPI in the Transformation Plan to verifiable deliverables, results and supporting evidence.

## 1. Provenance and how to read this public report

This is a public adaptation of the English FIERCE evidence report dated
28 October 2026. The original report file was recorded at source-report commit
`fe21e58b8ee5baaa800e41601248563310825fa5`, with SHA-256
`ee95442027f62e056c6f84a8a451a4d87747494cc7503cc6818973dc2a014492`.
Its technical citations refer to the separate evidence-source commit
`d4bea08f41b20b337ee8152704c7ed5231876072`. The report itself was not present at
that earlier evidence commit. This DOCX edition is newly generated from the
public English Markdown edition; it is not a recovered original Word document.

Targets below are transcribed from the supplied FIERCE Transformation Plan:
deliverables and KPIs on pages 8–9, impact contributions on page 11, and maturity
targets on page 2. The contractual originals and commercial materials are not
published here. This report does not certify the adapted scope's conformity with
the original plan.

The [evidence index](README.md) and [source map](https://github.com/theseedship/mcp_geoai_deposium/blob/66252f53960d6ac7c9cb6c92587dcb4f8ee8a05a/docs/evidence/source-map.json) identify the
curated technical excerpts and bounded acceptance/integration records supporting
the report. Original-source hashes identify retained provenance; curated excerpts
have their own hashes. Publication does not rerun the experiments, requalify a
deployment or independently certify an acceptance. Unpublished underlying artifacts
and business evidence remain separate from these public excerpts.

Three levels are distinguished: **implementation** (code, data and calculations),
**observed qualification** (dated tests, reproduction and demonstrations), and
**internal acceptance** by Nicolas/Deposium. Internal acceptance is not external
FIERCE approval or positive validation of every scientific hypothesis.

The [acceptance scoreboard](https://github.com/theseedship/mcp_geoai_deposium/blob/66252f53960d6ac7c9cb6c92587dcb4f8ee8a05a/docs/evidence/proofs/proof-02.md) records **15/15 B/O lots accepted
within adopted scopes**. This count is distinct from the plan's five scientific
work packages and six modules; it is neither a predictive success rate nor a
statement that all original stronger objectives were met.

## 2. Summary of the plan’s KPIs

| KPI | Stated target | Implementation and available evidence | Supporting material or limitation to retain in the assessment |
| --- | --- | --- | --- |
| **1 — Commercial validation / D1** | 5/5 service components | The market-study outputs are not audited in these technical repositories. | Attach the CLELYA/Amandine deliverables; do not infer commercial results from code. |
| **2 — Pattern mining / D2** | 5/5 work packages delivered and integrated | Five documented areas: a real-data comparison, integrated native extraction, evaluation, temporal/spatial extensions and packaging. | Acceptance is within adopted scopes; negative results and the absence of calibrated confidence remain explicit. |
| **3 — Geospatial pipeline / D3** | 6/6 modules delivered and operational | B1 through B6 delivered and accepted: zones, ERA5, NDVI, unified data, forecasts with fallback, and a two-level engine. | Explain the FourCastNet/AIFS and explanatory-index/confidence adaptations; attach the requested video on the `.ai` SaaS platform. |
| **4 — Indicators across the pilots** | At least 3 indicators across all 3 pilots, analysed together | NDVI, temperature and precipitation in 130 joined historical observations; a qualified conversational access path and reproducible joint analyses. | This does not prove continuous production monitoring. The combined-use video remains a separate piece of evidence. |
| **7 — Environmental indicators** | More than 2 new indicators using the data covered by FIERCE | Three Copernicus indicators can be used together; Sentinel-2 NDVI and two ERA5 quantities are identified separately. | The plan expressly reserves FIERCE’s interpretation of “satellite data”. These are not three independent satellite observations. |
| **8 — Pollution** | The plan claims no direct reduction >5% | No measured pollution reduction is claimed. | Retain the plan’s answer; no impact value is inferred from alerts. |
| **9 — Resource efficiency** | An expected improvement of 5–10% in applicable use cases | The features can support decision-making scenarios; the technical audit identified no measurement of this gain. | Attach a quantified scenario and its assumptions; distinguish a projection from an observed gain. |
| **10 — Critical raw materials** | Outside the stated scope | No CRM recycling, recovery or extraction objective is claimed. | Retain this scope boundary. |

KPIs 5 and 6 do not appear in the supplied plan; no additional objective is created here.

## 3. KPI 1 / D1 — Market study and commercial readiness

**Responsibility for documentation: CLELYA Conseil, Amandine and Deposium coordination.** The five sets of supporting materials to attach are: the market study; a synthesis of the 15 announced interviews; pricing and revenue-model recommendations; the go-to-market strategy; and fundraising preparation.

Interviews and potential pilot opportunities can explain the intended uses. They do not establish delivery of these five sets of materials. **This report assigns neither 0/5 nor 5/5 to this KPI**: it identifies the required business evidence, outside the repository audit.

## 4. KPI 2 / D2 — The five scientific and software work packages

### WP0 — Compare so4gp with another library on Copernicus data

The adopted comparison runs **GRAANK from so4gp 1.1.2 and Apriori from mlxtend 0.25.0** on the same three ERA5/NDVI datasets, with seven variables. All six CPU jobs finish within their 60-second limits. **20 native identities are present among 91 reference identities; all 71 differences are explained, and support values agree for every shared identity.** An independent Linux reproduction recovers the same identities, supports and classifications.

**Evidence:** [results, artifacts and reproduction](https://github.com/theseedship/mcp_geoai_deposium/blob/66252f53960d6ac7c9cb6c92587dcb4f8ee8a05a/docs/evidence/proofs/proof-26.md), [O1/#19 acceptance](https://github.com/theseedship/mcp_geoai_deposium/blob/66252f53960d6ac7c9cb6c92587dcb4f8ee8a05a/docs/evidence/proofs/record-01.md).

**What this establishes:** a real-data comparison that can be checked and reproduced. It does not establish general superiority, a speed ranking or NDVI forecasting skill.

### WP1 — Extract patterns and integrate them into DEPOSIUM

The native pipeline produces JSON/Parquet libraries that GeoAI tools can consume. The 1.1.2 capture across the three pilots contains **23 GRAANK records and 395 TGrad records, for a total of 418 records**, distributed across six libraries. The consumer reads them and refuses to draw conclusions when they are unqualified.

**Evidence:** [native package, inputs and consumer checks](https://github.com/theseedship/mcp_geoai_deposium/blob/66252f53960d6ac7c9cb6c92587dcb4f8ee8a05a/docs/evidence/proofs/proof-23.md), [tools and integration](https://github.com/theseedship/mcp_geoai_deposium/blob/66252f53960d6ac7c9cb6c92587dcb4f8ee8a05a/docs/evidence/proofs/proof-01.md).

**Limitation:** all 418 records remain unqualified. Their availability and the explicit refusal to draw unsupported conclusions demonstrate integration, not positive predictive validation. These deliveries document the correspondence to WP1 without inventing a separate WP1 acceptance.

### WP2 — Evaluate robustness, stability and confidence

The study uses five groups of zones and a temporal split of the data. At the original thresholds of **support ≥0.50 and conditional confidence ≥0.70**, **0/10 spatial candidates** pass at least four of the five folds, and **6/52 temporal rules** pass the temporal check. These six successes correspond to three inverse pairs and fail locally at Fos. The negative results form part of the scientific deliverable.

Nicolas/Deposium expressly adopted **five groups accounting for overlaps across 14 zones**, training on four groups and evaluating the fifth, as the correspondence to the original criterion expressed in terms of individual zones. This method and its acceptance remain distinct from passing the tests.

The accepted historical O2 catalogue contains **179 original Q3 records**, including **164 containing NDVI** and 15 climate-only itemsets. It preserves supports, dates, recurrence metadata and missing confidence values. This catalogue is distinct from the newer capture of 418 records.

**Evidence:** [held-out evaluation report](https://github.com/theseedship/mcp_geoai_deposium/blob/66252f53960d6ac7c9cb6c92587dcb4f8ee8a05a/docs/evidence/proofs/proof-22.md), [catalogue and adopted scope correspondence](https://github.com/theseedship/mcp_geoai_deposium/blob/66252f53960d6ac7c9cb6c92587dcb4f8ee8a05a/docs/evidence/proofs/proof-15.md), [O2 acceptance](https://github.com/theseedship/mcp_geoai_deposium/blob/66252f53960d6ac7c9cb6c92587dcb4f8ee8a05a/docs/evidence/proofs/record-02.md), [O3 acceptance](https://github.com/theseedship/mcp_geoai_deposium/blob/66252f53960d6ac7c9cb6c92587dcb4f8ee8a05a/docs/evidence/proofs/record-03.md).

**Key limitation:** the evaluation and catalogue were accepted within an adapted scope. The original expectation of a positively validated library with confidence remains unmet; unknown fields are not replaced with a score for presentation purposes.

### WP3 — Add temporal and spatial dimensions

The Q3 run across **14 eligible zones** produced **30 GRAANK and 149 TGrad records**, with lags, supports and recurrence metadata. The ability to extract lagged patterns and its integration were accepted under O4.

**Evidence:** [Q3 execution and results](https://github.com/theseedship/mcp_geoai_deposium/blob/66252f53960d6ac7c9cb6c92587dcb4f8ee8a05a/docs/evidence/proofs/proof-12.md), [O4/#22 acceptance](https://github.com/theseedship/mcp_geoai_deposium/blob/66252f53960d6ac7c9cb6c92587dcb4f8ee8a05a/docs/evidence/proofs/record-04.md).

**Limitation:** cross-zone recurrence and a native lag measure neither causality nor the accuracy of an independent forecast at that number of days.

### WP4 — Deliver an installable, callable tool

The historical O5 acceptance records an MCP interface, API documentation,
serialization, the `fierce1-mining` CLI and tag **v0.1.0**. These are historical
acceptance references, not a newly published release tag or the current standalone
GeoAI entry point. The current public [package entry points](https://github.com/theseedship/mcp_geoai_deposium/blob/860f11568c1888a4d5b461df14df94febe8558eb/pyproject.toml)
provide `geoai-server` and forecast CLIs; they do not declare `fierce1-mining`.
The so4gp 1.1.2 profile and its CPU qualification are later, traceable work.

**Evidence:** [packaging and O5 acceptance](https://github.com/theseedship/mcp_geoai_deposium/blob/66252f53960d6ac7c9cb6c92587dcb4f8ee8a05a/docs/evidence/proofs/proof-02.md), [decision #23](https://github.com/theseedship/mcp_geoai_deposium/blob/66252f53960d6ac7c9cb6c92587dcb4f8ee8a05a/docs/evidence/proofs/record-05.md), [source-inclusion trace for the 22 upstream modules](https://github.com/theseedship/mcp_geoai_deposium/blob/66252f53960d6ac7c9cb6c92587dcb4f8ee8a05a/docs/evidence/proofs/proof-25.md).

### Predictive result to retain in the assessment

A dedicated experiment tested the contribution of native interactions to an NDVI prediction recipe. On the primary criterion at Fos, MAE increases from **0.02263 to 0.02354**, or approximately **4% more error**; persistence achieves 0.02237. This recipe therefore demonstrates no gain on the primary criterion. It tests neither a labelled stress alert nor a calibrated probability.

**Evidence:** [ablation protocol, results and limitations](https://github.com/theseedship/mcp_geoai_deposium/blob/66252f53960d6ac7c9cb6c92587dcb4f8ee8a05a/docs/evidence/proofs/proof-21.md). River benchmarks provide separate supplementary evidence; they do not automatically qualify the climate/NDVI rules.

## 5. KPI 3 / D3 — The six geospatial pipeline modules

| Module in the plan | Verifiable implementation | Main evidence |
| --- | --- | --- |
| **B1 — Zones** | 15 zones: a panel of 12 and three French pilots. WGS84 geometries; identified administrative and hydrological boundaries for the pilots. | [B1 generation and acceptance](https://github.com/theseedship/mcp_geoai_deposium/blob/66252f53960d6ac7c9cb6c92587dcb4f8ee8a05a/docs/evidence/proofs/proof-02.md). |
| **B2 — ERA5** | Six variables, 15 zones, **1991–2020: 30 years**, 10,958 days per zone, **100% observed completeness**, and 72 monthly normals per zone. GeoParquet in S3-compatible storage. | [Full-panel run and per-zone table](https://github.com/theseedship/mcp_geoai_deposium/blob/66252f53960d6ac7c9cb6c92587dcb4f8ee8a05a/docs/evidence/proofs/proof-09.md), [RustFS materialization](https://github.com/theseedship/mcp_geoai_deposium/blob/66252f53960d6ac7c9cb6c92587dcb4f8ee8a05a/docs/evidence/proofs/proof-07.md). |
| **B3 — NDVI** | Sentinel-2 STAC/COG streaming and zonal statistics, without archiving COGs. Available series from November 2016 to December 2024: **81 Montpellier, 81 Lez, 92 Fos**, for a total of 254 observations. | [Shipped-code evidence, manifests and coverage](https://github.com/theseedship/mcp_geoai_deposium/blob/66252f53960d6ac7c9cb6c92587dcb4f8ee8a05a/docs/evidence/proofs/proof-03.md), [B3 acceptance](https://github.com/theseedship/mcp_geoai_deposium/blob/66252f53960d6ac7c9cb6c92587dcb4f8ee8a05a/docs/evidence/proofs/proof-02.md). |
| **B4 — Unified data** | ERA5/NDVI join usable in DuckDB/GeoParquet; a historical package with **130 rows: 41 Lez, 48 Fos, 41 Montpellier**. Dates and gaps are preserved. | [Materialization and matching](https://github.com/theseedship/mcp_geoai_deposium/blob/66252f53960d6ac7c9cb6c92587dcb4f8ee8a05a/docs/evidence/proofs/proof-04.md), [conversational qualification](https://github.com/theseedship/mcp_geoai_deposium/blob/66252f53960d6ac7c9cb6c92587dcb4f8ee8a05a/docs/evidence/proofs/proof-11.md). |
| **B5 — Forecasts and fallback** | AIFS primary through Open-Meteo, with physical-model fallback. GFS/AIFS archive and matrices: **86,040 pairs per identity**, 15 zones, 239 dates, six variables and day-1/day-3/day-5/day-7 horizons; no missing or excluded pairs. | [Collection and independent reproduction](https://github.com/theseedship/mcp_geoai_deposium/blob/66252f53960d6ac7c9cb6c92587dcb4f8ee8a05a/docs/evidence/proofs/proof-05.md), [#193 acceptance](https://github.com/theseedship/mcp_geoai_deposium/blob/66252f53960d6ac7c9cb6c92587dcb4f8ee8a05a/docs/evidence/proofs/record-06.md). |
| **B6 — Two levels of analysis** | Level 1: thresholds/percentiles. Level 2: a traced explanatory index. Historical Fos case: **149 patterns examined, 147 excluded, 2 eligible**; displayed response and verified canonical persistence. | [B6 acceptance, values and protected reads](https://github.com/theseedship/mcp_geoai_deposium/blob/66252f53960d6ac7c9cb6c92587dcb4f8ee8a05a/docs/evidence/proofs/proof-06.md), [#30 acceptance](https://github.com/theseedship/mcp_geoai_deposium/blob/66252f53960d6ac7c9cb6c92587dcb4f8ee8a05a/docs/evidence/proofs/record-07.md). |

### Scope correspondences to explain against the original plan

- **ERA5 period:** 15 years announced; 30 years delivered for the 1991–2020 reference. The recent validation period is a separate archive.
- **Forecast model:** the plan names FourCastNet/Open-Meteo; the internally accepted scope uses **AIFS/physical models**. FourCastNet remains an unavailable extension point. Do not present this substitution as a completed FourCastNet integration or as external approval already obtained.
- **Confidence:** the accepted level 2 calculates `support × stability × forecast agreement × lag decay`. The indices 0.42518 and 0.41861 rank two historical associations; they do not represent probabilities of 42%. Calibrated confidence and probability remain unavailable.
- **Historical-case activation:** the states `scientific_adoption: pending`, `ready_for_scoring: false` and `live_activation: false` remain unchanged. The accepted historical explanatory calculation does not activate a scientific alert in production; bulletins and user-defined thresholds have their own operational path.
- **Horizon:** forecast availability up to 15/16 days does not establish measured skill at those lead times. The cited matrices evaluate day 1/day 3/day 5/day 7.
- **Storage:** RustFS fills the S3-compatible role planned for MinIO. Access to registered artifacts depends on their installation and the permissions in each environment.

The [acceptance checkpoint](https://github.com/theseedship/mcp_geoai_deposium/blob/66252f53960d6ac7c9cb6c92587dcb4f8ee8a05a/docs/evidence/proofs/proof-08.md) and [minimum technical scope correspondence](https://github.com/theseedship/mcp_geoai_deposium/blob/66252f53960d6ac7c9cb6c92587dcb4f8ee8a05a/docs/evidence/proofs/proof-20.md) document these choices. Bounded internal acceptance does not turn a stronger scientific objective into a measured result.

## 6. KPI 4 — Three indicators across the three pilots

The indicators jointly present in the historical package are **dimensionless NDVI, daily mean temperature in °C, and daily precipitation in mm**.

| Pilot | Available joined rows | Period actually represented |
| --- | ---: | --- |
| Bassin du Lez | 41 | 11 February 2021 – 27 December 2024 |
| Fos-sur-Mer | 48 | 24 January 2021 – 29 December 2024 |
| Montpellier Métropole | 41 | 11 February 2021 – 27 December 2024 |

**Evidence of actual access:** [N1 qualification](https://github.com/theseedship/mcp_geoai_deposium/blob/66252f53960d6ac7c9cb6c92587dcb4f8ee8a05a/docs/evidence/proofs/proof-11.md). The two pages of **100 then 30 rows** reproduce the source exactly. Signatures, permissions, identities and retained data were checked; Nicolas attested the reload and readback, corroborated by protected reads.

**Evidence of joint analysis:** the [library package](https://github.com/theseedship/mcp_geoai_deposium/blob/66252f53960d6ac7c9cb6c92587dcb4f8ee8a05a/docs/evidence/proofs/proof-23.md) mines NDVI alongside six ERA5 variables; the [reproduced WP0 comparison](https://github.com/theseedship/mcp_geoai_deposium/blob/66252f53960d6ac7c9cb6c92587dcb4f8ee8a05a/docs/evidence/proofs/proof-26.md) covers all three pilots. Local calculations and conversational reading each have their own evidence; combining them in a video must be identified as a separate demonstration.

**Interpretation limits:** spaced NDVI acquisitions, no daily interpolation, point-based ERA5 reanalysis and zonal NDVI, historical geometry differences, and the Lez/Métropole overlap. These 130 rows constitute neither three complete daily series nor continuous production monitoring. A recorded zero and a missing value remain distinct.

## 7. What is delivered in DEPOSIUM

| Component | Function provided | Evidence |
| --- | --- | --- |
| **GeoAI / Python** | Adapts providers, processes rasters and geometries, calculates indicators/patterns and exposes typed contracts. The current registry contains **16 tools**, including the experimental solar pilot. | [Tool catalogue](https://github.com/theseedship/mcp_geoai_deposium/blob/66252f53960d6ac7c9cb6c92587dcb4f8ee8a05a/docs/evidence/proofs/proof-01.md). |
| **MCPs** | Connects tools to chat, enforces owner/space permissions, retains results and verifies their provenance during readback. | [Integration contract](https://github.com/theseedship/mcp_geoai_deposium/blob/66252f53960d6ac7c9cb6c92587dcb4f8ee8a05a/docs/evidence/proofs/proof-14.md), [MCPs #962](https://github.com/theseedship/mcp_geoai_deposium/blob/66252f53960d6ac7c9cb6c92587dcb4f8ee8a05a/docs/evidence/proofs/record-08.md), [#963](https://github.com/theseedship/mcp_geoai_deposium/blob/66252f53960d6ac7c9cb6c92587dcb4f8ee8a05a/docs/evidence/proofs/record-09.md). |
| **Solid** | Weather/NDVI forms, B6 rendering, tables and charts based on retained results, and collapsible sources and limitations. | [Interface specification and evidence](https://github.com/theseedship/mcp_geoai_deposium/blob/66252f53960d6ac7c9cb6c92587dcb4f8ee8a05a/docs/evidence/proofs/proof-18.md), [Solid #919](https://github.com/theseedship/mcp_geoai_deposium/blob/66252f53960d6ac7c9cb6c92587dcb4f8ee8a05a/docs/evidence/proofs/record-10.md), [#920](https://github.com/theseedship/mcp_geoai_deposium/blob/66252f53960d6ac7c9cb6c92587dcb4f8ee8a05a/docs/evidence/proofs/record-11.md), [#922](https://github.com/theseedship/mcp_geoai_deposium/blob/66252f53960d6ac7c9cb6c92587dcb4f8ee8a05a/docs/evidence/proofs/record-12.md). |
| **DuckDB + GeoParquet + RustFS/S3** | Store and query structured artifacts independently of model-generated text. | [Storage contract](https://github.com/theseedship/mcp_geoai_deposium/blob/66252f53960d6ac7c9cb6c92587dcb4f8ee8a05a/docs/evidence/proofs/proof-19.md). |
| **Bulletins and monitoring** | Daily delivery at a chosen time, threshold rules, enablement/revocation, traces and recovery; optional publication in a conversation. | [Delivered scope](https://github.com/theseedship/mcp_geoai_deposium/blob/66252f53960d6ac7c9cb6c92587dcb4f8ee8a05a/docs/evidence/proofs/proof-17.md), [email actually received on 7 October](https://github.com/theseedship/mcp_geoai_deposium/blob/66252f53960d6ac7c9cb6c92587dcb4f8ee8a05a/docs/evidence/proofs/proof-10.md), [chat publication](https://github.com/theseedship/mcp_geoai_deposium/blob/66252f53960d6ac7c9cb6c92587dcb4f8ee8a05a/docs/evidence/proofs/proof-16.md). |

The main tools linked to the KPIs are `era5_ingest`, `ndvi_ingest`, `climate_enrich_ndvi`, `climate_percentiles`, `climate_normals`, `forecast_query`, `discover_patterns`, `discover_patterns_cv` and `pattern_alert`. The MCP protocol makes these capabilities callable by a client/agent; their availability in a space depends on the connector, permissions, storage and configured sources.

Verifiable displays and calculations remain distinct from document RAG: the model can explain results but is not used as the source of the numbers. Reading back retained results acquires no new weather or satellite data.

### Qualification and environments

The cited historical N1 and B6 evidence concerns **a staging environment and authorized spaces**, at the recorded dates and versions. Deploying the same code on `.ai` does not copy data packages or grant production spaces access to them. The [documented N1 registrations and checks](https://github.com/theseedship/mcp_geoai_deposium/blob/66252f53960d6ac7c9cb6c92587dcb4f8ee8a05a/docs/evidence/proofs/proof-11.md) are specific to the authorized environment and space. A historical demonstration on `.ai` therefore needs its own installation and authorization, without requiring the same archives to be collected again. **This report provides no production qualification of that historical path.**

The forms and rich NDVI views have retention, refusal and Chromium rendering tests. As of this report, a new NDVI acquisition followed by reloading it in this new view remains to be confirmed through connected use; the older N1 reader evidence is retained separately.

## 8. Impact contributions: KPIs 7 to 10

**KPI 7 — More than two indicators.** The shared technical evidence is that of KPI 4. Distinguish **NDVI derived from Sentinel-2** from **temperature and precipitation from ERA5**, a reanalysis. The plan explicitly reserves FIERCE’s interpretation of “satellite data”. The report demonstrates the availability and analysis of three Copernicus indicators; it certifies neither three regulatory CSRD indicators nor three independent satellite measurements.

**KPI 8 — Pollution.** The plan states that no direct reduction above 5% is claimed during the project. Decision-support features do not constitute a measurement of avoided emissions or avoided air, soil or water pollution.

**KPI 9 — Resources.** The **5–10% target is a projection**, to be supported by a pilot scenario specifying the baseline, the operation concerned, assumptions and calculation method. No measurement of this impact was found in the technical materials examined. Interviews can inform these assumptions; they do not become observed gains. This supporting material remains to be completed by the business-study/coordination team.

**KPI 10 — CRM.** The project announces no contribution to the recycling, recovery or extraction of critical raw materials. This KPI remains outside the scope, as stated in the source plan.

## 9. Objective maturity assessment and TRL evidence

### Reference framework and assessed technology

The plan specifies **TRL 4 → 6, IRL 3 → 4 and SRL 3 → 5**. Our technical assessment, based on this dossier's dated evidence, is **a supported TRL 6 for the integrated geospatial demonstrator within its documented scope; TRL 7 is not established.** This is a technical self-assessment with moderate confidence, rather than an external FIERCE decision.

The reference is the [Horizon Europe 2026–2027 General Annexes, section B, page 15 of the current PDF](https://ec.europa.eu/info/funding-tenders/opportunities/docs/2021-2027/horizon/wp-call/2026-2027/wp-15-general-annexes_horizon-2026-2027_en.pdf#page=15), consulted on 28 October. In summary, level 4 concerns validation under laboratory conditions; level 5 concerns validation under representative conditions; level 6 concerns demonstration under those conditions; level 7 concerns demonstration of the system prototype in its operational setting. We use this framework for the assessment; it does not replace any programme-specific appraisal.

The assessed technology is **the geospatial pipeline added to DEPOSIUM**: real-data preparation, indicators, tools, authorized conversational access, retention, forecast-threshold notifications and historical explanatory analysis. Its level is not automatically inherited from the pre-existing document platform. The novel pattern-based environmental prediction capability is assessed separately below.

### Why TRL 6 is defensible

| Criterion considered | Concrete evidence | Contribution to the assessment |
| --- | --- | --- |
| Representative inputs and conditions | [Real ERA5 data across 15 zones](https://github.com/theseedship/mcp_geoai_deposium/blob/66252f53960d6ac7c9cb6c92587dcb4f8ee8a05a/docs/evidence/proofs/proof-09.md) and [real NDVI across three pilots](https://github.com/theseedship/mcp_geoai_deposium/blob/66252f53960d6ac7c9cb6c92587dcb4f8ee8a05a/docs/evidence/proofs/proof-03.md), with geometries, units, gaps and manifests. | Validation is not limited to synthetic examples. |
| Integrated application workflow | [N1: 130 exact rows in chat, pagination, retention and readback after reload](https://github.com/theseedship/mcp_geoai_deposium/blob/66252f53960d6ac7c9cb6c92587dcb4f8ee8a05a/docs/evidence/proofs/proof-11.md). | Demonstration uses the deployed application, actual user identity, space permissions and storage. |
| Actual acquisition and analysis | [Real Site-Watch run on 4 October](https://github.com/theseedship/mcp_geoai_deposium/blob/66252f53960d6ac7c9cb6c92587dcb4f8ee8a05a/docs/evidence/proofs/proof-13.md): provider forecast, normals, classification and retained result. | The service performs environmental processing representative of the intended use. |
| Scheduled execution reaching a recipient | [Natural 7 October bulletin and inbox receipt](https://github.com/theseedship/mcp_geoai_deposium/blob/66252f53960d6ac7c9cb6c92587dcb4f8ee8a05a/docs/evidence/proofs/proof-10.md). | The forecast → scheduled task → retention → delivery chain reaches a real recipient. This occurrence does not establish general reliability or an SLA. |
| Integrated explanatory pattern analysis | [Historical B6: 149 patterns, two displayed results and two protected reads verifying persistence](https://github.com/theseedship/mcp_geoai_deposium/blob/66252f53960d6ac7c9cb6c92587dcb4f8ee8a05a/docs/evidence/proofs/proof-06.md). | The bounded calculation was demonstrated in the application; scientific and activation states remain unchanged. |

The relevant environment is the deployed SaaS with its actual services, sources, permissions and storage, used on real pilot problems. Observing a workflow in staging does not automatically reduce it to a laboratory experiment. **TRL 5 is strongly supported; the integrated-workflow demonstrations justify our TRL 6 assessment for this scope.** Unit tests, CI and deployments alone would not support that conclusion.

Confidence remains moderate: the campaigns are bounded, the owner acts as the pilot user, N1/B6 use historical data, and demonstrations occurred on different dates. This assessment does not extend to every account, site, language or new interface. Neither predictive gain nor commercial acceptance is inferred from the maturity assessment.

### What is not established: TRL 7 and novel prediction

**We do not assign TRL 7 to the complete GeoAI system.** The records do not yet demonstrate the whole prototype performing a representative environmental business workflow in its intended operational context. Production deployment or one received email is insufficient to establish that. A focused next-stage campaign could document use by the intended operators, expected outputs, their use, errors and recovery in that context. We impose no arbitrary duration or additional signature as a universal TRL 7 rule.

**Novel pattern-based predictive alerting is a separate capability: its research components are at approximately TRL 4 at most on the available evidence; TRL 5/6 is not demonstrated for that function.** A positively qualified native library and a validated link between patterns and future environmental impacts have not been supplied. The [pattern-qualification record](https://github.com/theseedship/mcp_geoai_deposium/blob/66252f53960d6ac7c9cb6c92587dcb4f8ee8a05a/docs/evidence/proofs/proof-24.md), [held-out evaluation](https://github.com/theseedship/mcp_geoai_deposium/blob/66252f53960d6ac7c9cb6c92587dcb4f8ee8a05a/docs/evidence/proofs/proof-22.md) and [NDVI ablation](https://github.com/theseedship/mcp_geoai_deposium/blob/66252f53960d6ac7c9cb6c92587dcb4f8ee8a05a/docs/evidence/proofs/proof-21.md) establish this boundary. The negative ablation result alone does not remove TRL 6 from the integrated demonstrator; it also does not demonstrate the missing predictive function.

### Recommended submission wording

“Our self-assessment assigns **TRL 6 to the integrated geospatial demonstrator**, within the scope of workflows actually demonstrated with real data in a relevant pilot environment: indicators, authorized conversational access, retention, historical explanatory analysis and forecast-threshold notifications. This assessment does not cover scientifically validated novel environmental prediction, calibrated probabilities, or TRL 7 for the complete system.”

Conformity of the adapted scope with the initial plan and external FIERCE appraisal remain separate from this technical assessment. **IRL/SRL** require the corresponding commercial, user and business evidence; their final levels are not inferred from the number of pull requests or accepted lots.

## 10. Materials to assemble for submission

| Material | Consolidation responsibility | Expected content |
| --- | --- | --- |
| D1 commercial study and readiness | CLELYA / Amandine / Nicolas | The five sets of materials for KPI 1, including the list and synthesis of interviews actually conducted. |
| D2 scientific evidence package | Nicolas, drawing on Dickson’s deliveries | Reproduced WP0, package/catalogue, validation report, negative results, O1–O5 acceptances and limitations. This report links to those existing materials. |
| D3 pipeline evidence package | Nicolas, drawing on Jonathan’s deliveries and Deposium integration | B1–B6 evidence, versions, acceptances, model adaptations and the interpretation of the score. |
| KPI 3 / KPI 4 video | Nicolas / coordination | Show the SaaS platform specified in the plan, the three pilots, the three indicators and their analysis; retain the URL/date/version and the environment actually used. This audit produces no video and does not claim one has been submitted. |
| KPI 9 impact scenario | Business study / coordination | A justified, quantified projection, clearly distinguished from a measured result. |
| Access to evidence | Nicolas / coordination | The public cited excerpts and source map; any unpublished underlying artifacts, original contractual documents and commercial materials must be supplied separately when required. |

**Suggested assessment wording:** “The project has delivered a geospatial pipeline integrated into DEPOSIUM, traceable ERA5 and Sentinel-2 datasets, pattern-discovery tools and their reproducible evaluation. The modules were accepted within explicitly documented technical scopes. The experiments do not support claims of general predictive superiority or calibrated event probabilities. Demonstrations and business evidence complement this implementation report.”
