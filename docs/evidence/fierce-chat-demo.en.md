# FIERCE — technical demonstration recording guide in DEPOSIUM

Prepared on **31 October 2026**. Audience: technical and scientific evaluators.
Scope: **geospatial capabilities, integration, results and limitations**.

[Version française](fierce-chat-demo.fr.md)

UI labels depend on the interface language. French labels are included where
helpful for navigation; technical command identifiers remain unchanged.

This guide accompanies the [technical annex and its evidence](fierce-kpi-evidence.en.md).
It prepares a new recording; it is neither an already completed video nor a
new qualification of the environment used. The cited evidence retains its
original dates and scope.

## 1. The story to tell

> We make weather and satellite data accessible in a workspace, with their
> dates, units and sources. We show three indicators across three pilots,
> saved analyses and monitoring rules. We have also tested pattern discovery:
> the scientific results, including negative results, remain distinct from
> the operation of the interface.

Three data families must remain clearly identifiable:

| Family | What is shown | What it represents |
| --- | --- | --- |
| N1 joined historical data | NDVI, temperature and precipitation across three pilots | Sentinel-2 acquisitions matched to ERA5 days; observations at intervals, not a forecast. |
| Current forecast | The next three days at an explicit point | Output from a weather provider, with a collection time; not a weather-station measurement. |
| B6 historical example | Pattern library and archived GFS forecast | Traceable explanatory calculation for a fixed historical case; not an event probability. |

**For submission of the deliverable, record on `deposium.ai`**: this is the SaaS
platform specified in the technical dossier. Rehearsing on another instance is
useful, but it must be identified by its actual environment name. On its own,
it does not count as a recording on `.ai`.

The messages below are **DEPOSIUM application commands**. Commands such as
`pilot_data_read`, `geoai_weather` and `level2_historical_case` are not native
tool names to send unchanged to the standalone Python server. The
[guide to the 16 MCP tools](../../reference/mcp-tools.md) describes that second
layer; the [architecture](../architecture.md) distinguishes the server from the
application.

## 2. Prepare a rehearsal that matches the recording

Before recording:

1. In the chosen environment, open a demonstration workspace with the
   geospatial analysis plugin enabled. Check that the account has permission
   to access this workspace. Use dedicated conversations for weather, N1 and B6.
2. Have the operator verify installation and authorization of the
   `fierce-pilots-2021-2024 / n1-local-20261014` package and the
   `q3-2023-fos-native-recurrence / 2026.10.27.1` case in **this** environment.
   Deploying code copies neither these data nor their permissions.
3. Run each selected scenario once. Record the versions actually served, the
   date, the environment and any limitations. Historical staging checks do
   not replace this rehearsal on `.ai`.
4. Wait for the final response and for it to be saved before retrying or
   reloading. A pending save verification can be retried with the designated
   button; it does not require a new acquisition.
5. Prepare tabs for the public evidence cited in each sequence. Hide email
   addresses and account data on screen.

The commands in scenarios A to E can be copied as written when the specified
packages are installed. Examples containing `<...>` are templates: the
operator must fill them in with the identities actually authorized before
recording.

If a result is unavailable, preserve its actual status. Resolve the
prerequisite before recording, or show the dated evidence and identify it as
such. A historical evidence capture does not become a run performed today.

## 3. Proposed main edit — approximately 15 minutes

| Approximate time | Sequence | Main message |
| --- | --- | --- |
| 0:00–0:45 | Context, environment and tools | A specialized MCP server used in DEPOSIUM; three data families. |
| 0:45–4:15 | A — Three pilots, three indicators, reading and descriptive analysis | Data can be accessed together, paginated and read back. |
| 4:15–6:15 | B — Weather, views and thresholds | Provider values, explicit calculations and the same saved result. |
| 6:15–7:45 | C — Satellite NDVI | Actual acquisitions, their dates and their limitations. |
| 7:45–10:00 | D — B6 example | Two explained scores, with exclusions and sources. |
| 10:00–12:15 | F — Scientific evidence | What was compared, reproduced and actually concluded. |
| 12:15–14:00 | G or H — Classification or monitoring | Climatological reference, or bulletin and thresholds, depending on the prepared workflow. |
| 14:00–15:00 | Synthesis and supplementary evidence E | Controlled access, saved result, delivered scope and maturity. |

G and H can be two supplementary clips if preparing them would extend the
video. At a minimum, retain: **`.ai` visible, three pilots, three indicators
together, documented combined analysis, readback and a distinction between
scientific results and software operation**.

## A. Three pilots and three historical indicators together

**Target: KPI 3 / B4, KPI 4 and KPI 7; persistence and authorized access.**
Evidence: [archive join](proofs/proof-04.md),
[N1 qualification](proofs/proof-11.md),
[joint analysis of the pilots](proofs/proof-23.md).

### A1. Open the first page

In a dedicated conversation:

```text
pilot_data_read {"package_id":"fierce-pilots-2021-2024","package_version":"n1-local-20261014","filters":{},"limit":100}
```

**Expected for this specific package:** 130 matches, 100 rows on this page,
continuation available. Show the table and its columns: zone, ERA5 UTC day,
NDVI acquisition, date offset, NDVI, temperature and rainfall. Open the sources
and method: version, fingerprints and limitations are part of the result,
alongside the prose in the response.

Say: “Each row matches a vegetation observation with historical weather data.
The three indicators can be viewed together.”

### A2. Read the next page, then reload

Click **Next page on the response's scientific panel** (**Page suivante**).
A table's internal pagination may only navigate rows already loaded: it is
not the package continuation.

If this button is unavailable, copy the **entire** continuation command
provided in the response, including its actual cursor. Keep the same package,
version, filters and limit. Do not reuse an old cursor from another
conversation.

**Expected:** 30 additional rows, all from Montpellier Métropole, “End of
selection” (“Fin de la sélection”), no next page. The two pages yield
**100 + 30 = 130 rows**, not 130 consecutive days.

Reload the browser page, then send:

```text
scientific_readback
```

**Expected:** readback of the last saved page, therefore **30 rows** at this
stage. The same values, version and sources; no new collection. This message
does not request a new aggregation of all 130 rows.

### A3. Present each pilot without truncating its observations

Send these commands in sequence:

```text
pilot_data_read {"package_id":"fierce-pilots-2021-2024","package_version":"n1-local-20261014","filters":{"zone_id":"bassin_du_lez"},"limit":100}
```

```text
pilot_data_read {"package_id":"fierce-pilots-2021-2024","package_version":"n1-local-20261014","filters":{"zone_id":"fos_sur_mer"},"limit":100}
```

```text
pilot_data_read {"package_id":"fierce-pilots-2021-2024","package_version":"n1-local-20261014","filters":{"zone_id":"montpellier_metropole"},"limit":100}
```

| Pilot | Expected rows | Period actually represented |
| --- | ---: | --- |
| Bassin du Lez | 41 | 2021-02-11 → 2024-12-27 |
| Fos-sur-Mer | 48 | 2021-01-24 → 2024-12-29 |
| Montpellier Métropole | 41 | 2021-02-11 → 2024-12-27 |

Each response must be terminal, with no truncation. In **Sources / method /
full text** (**Sources / méthode / texte intégral**), show the calculated
means for the three variables and their known sample counts `n`. Missing cells
are excluded; zeros are retained. With these filters, one page contains all
selected observations for the pilot. On a truncated global page, these means
would apply only to the rows on that page.

### A4. Show combined analysis without inventing a chat feature

Compare the three summaries above: NDVI, temperature and rainfall are present
in the same selection, with their sample counts. If needed, use the values
**actually returned** in a comparison title card.

Say: “The service calculates descriptive summaries of the selected
observations. These means describe the sample on the acquisition dates.
They are not climate means covering every day from 2021 to 2024.”

For **joint analysis beyond display**, then open the
[native package](../../src/data/pattern-library-2026-10-15/package/manifest.json)
and its [report](proofs/proof-23.md): NDVI and six ERA5 variables were analyzed
together across the same three pilots. This yields six libraries,
**23 GRAANK patterns + 395 TGrad = 418 records**, all unqualified.
The [reproduced WP0 comparison](proofs/proof-26.md) provides a second piece of
evidence of joint analysis on these data.

Make the transition explicit: “Here is the reproducible analysis performed on
the data; its evidence is distinct from reading them in the chat.” Do not
present LLM commentary as a calculated correlation, a new mining run or an
independent validation.

**Limitations to keep visible:** NDVI is unitless; ERA5 daily mean temperature
is in °C; ERA5 daily rainfall is in mm; the acquisition date is distinct from
the ERA5 day; nearest-date matching is retained within ±3 days. No daily
interpolation. Lez and Métropole overlap. Historical geometries and ERA5 points
have documented limitations.

## B. Current forecast, visualizations and calculations using the same values

**Target: KPI 3 / B5 and application integration.**
References: [weather tools](../../reference/mcp-tools.md#weather-and-climate),
[interface scope](proofs/proof-18.md).

### B1. Acquire an explicit forecast

In a new conversation, use the **Weather forecasts** (**Prévisions météo**)
form, or send:

```text
geoai_weather {"version":1,"latitude":43.45455,"longitude":4.89795,"forecast_days":3,"variables":["precipitation","wind_speed","temperature_2m"]}
```

This is a **demonstration point in Fos-sur-Mer**, not a municipality polygon.
A city name does not prove that the provider uses its city center. The older
saved point `montpellier` and the `montpellier_metropole` pilot are also
different entries.

**Expected:** UTC dates, queried coordinates, values, units, served and
requested horizons, provider/model actually used, any fallback, licence and
GeoAI collection time. The figures are those available on the day: no
temperature or threshold exceedance is predetermined in this scenario.

Show the distinction: here, `issued_at` describes GeoAI collection; the model
issue time remains unknown if the provider does not supply it separately.

### B2. Reuse the saved response

Once it has been saved, send these messages one after another:

```text
Show a table of these forecasts.
```

```text
Show a bar chart of these forecasts
```

```text
Map these locations.
```

```text
Which days of these forecasts exceed 20 mm of rain or 10 m/s of wind?
```

```text
Where do these forecasts come from?
```

**Expected:** tables and charts based on the saved forecast, with distinct
variables and units; a map of the queried point; dates and values of strict
`>` exceedances, or no exceedances **among values that can be evaluated**.
No new weather call for these readbacks.

Precipitation values are totals per UTC day; wind is the daily maximum at 10 m
in m/s, **not a gust**. A missing value remains unknown. The map does not
represent a rainfall layer or a pilot boundary.

This conversational threshold reader handles one or two distinct rain/wind
thresholds, strict or inclusive, combined with **OR**. Do not attribute the
broader logic of the monitoring agent in scenario H to it.

### B3. Check continuity

Reload, then send:

```text
Summarize these forecasts.
```

**Expected:** the same values and collection time as B1. A follow-up must
remain in the conversation where this forecast is saved. Ambiguous requests,
a change of subject or a failed acquisition may require a new explicit
request; this scenario does not qualify every possible phrasing in an open
conversation.

This workflow demonstrates acquisition, presentation and reuse of results.
**Forecast accuracy** is evaluated separately in F.

## C. Satellite: NDVI profile and acquisitions actually retained

**Target: KPI 3 / B3 and satellite integration.**
References: [delivered NDVI](proofs/proof-03.md),
[interface and qualification limitations](proofs/proof-18.md).

In a new conversation, open **Satellite NDVI profile** (**Profil NDVI satellite**):

| Field | Demonstration value |
| --- | --- |
| West, south, east, north | `4.88`, `43.44`, `4.91`, `43.47` |
| Start / end | `2026-09-01` / `2026-09-30` |
| Minimum spacing | `16` days |
| Maximum cloud cover | `20` % |

Equivalent command:

```text
geoai_ndvi {"version":1,"bbox":[4.88,43.44,4.91,43.47],"start_date":"2026-09-01","end_date":"2026-09-30","interval_days":16,"cloud_cover_max":20}
```

**Expected:** acquisitions actually found and retained, dates, STAC identities,
NDVI and cloud statistics; a distinction between scenes searched, scenes
processed before spacing and observations retained. An external query for
the same month does not guarantee exactly the figures in an older response.

In this explicit workflow, show the saved result panel: table, scatter plot
and **Sources and method** (**Sources et méthode**). The plot preserves the
time gaps and does not connect acquisitions in a way that suggests a daily
series. If no acquisition is retained, show the status. Without a usable
dated NDVI mean, no chart is expected.
Reload to check the same saved result. If persistence is pending, use
**Retry saved verification** (**Réessayer la vérification enregistrée**).

Say: “NDVI describes the greenness of this area on the observation dates.
We preserve gaps in acquisition. A stress signal alone cannot establish its
cause, a yield loss or a quantity of carbon.”

The filter is **scene cloud cover < 20 %**, not a quality check on every pixel.
The interval is a minimum, not a guarantee of periodicity. A negative NDVI
minimum does not identify water, shadow or an artifact without further checks.
NDVI is unitless.

The published evidence distinguishes interface tests from the connected
workflow. This rehearsal must therefore record whether a **new acquisition
with a rich result panel followed by reload** was actually observed in the
recorded environment. A natural-language query may return text without using
this saved-panel workflow; use the form or command here.

## D. B6: understand a historical explanatory result

**Target: KPI 3 / B6, traceable calculation and acceptance of the adopted scope.**
Evidence: [accepted historical case](proofs/proof-06.md),
[bounded acceptance](proofs/record-07.md).

Open **B6 historical example → View the example** (**Exemple historique B6 →
Consulter l’exemple**), or send:

```text
level2_historical_case {"case_id":"q3-2023-fos-native-recurrence","case_version":"2026.10.27.1"}
```

**Expected for these bytes and this version:** 149 patterns read, 147 excluded,
2 admissible, in this order:

| Pattern | Support | Recurrence | Agreement | Lag | Explanatory score |
| --- | ---: | ---: | ---: | ---: | ---: |
| 39 | 0.6105 | 4/5, or 0.8 | 1 | +2 days | ≈ 0.42518 |
| 15 | 0.592 | 5/5, or 1 | 1 | +5 days | ≈ 0.41861 |

Expand the factors:

```text
score = support × recurrence stability × directional agreement × 0.5^(|lag|/10)
```

Expand the exclusions: **85 target zone, 12 stability, 40 horizon,
10 insufficient forecast values = 147**. These counts use the first failed
filter, rather than cumulative categories.

Then show the dates: GFS issue time **2023-06-30 at 00:00 UTC**, archived
temperatures for **1–5 July 2023**, and a much later archive collection time.
The historical Fos point is **43.445, 4.92**, distinct from the point in B1.
Values are means of eight instantaneous samples at three-hour intervals per
UTC day; humidity and ET0 remain absent.

Say: “We explain why these two patterns pass the engineering filters and how
their scores are calculated. **0.425 does not mean a 42.5 % probability.**
Recurrence across zones is not predictive accuracy. This historical case
activates no operational alert.”

Reload the conversation and reopen the sources/factors of the saved panel.
A verification failure must remain visible; do not certify a score before
verifying that it has been saved. Readback does not retrieve current weather
or rerun pattern extraction.

## E. Controlled refusal, without changing actual permissions

**Target: no invented data and adherence to an authorized version.**
Historical evidence: [N1 refusals and readbacks](proofs/proof-11.md).

In a separate test conversation:

```text
pilot_data_read {"package_id":"fierce-pilots-2021-2024","package_version":"video-version-non-enregistree-20261031","filters":{},"limit":100}
```

**Expected:** unavailable/refused; no values from the requested package and
no substitute external collection. The displayed wording may differ from
the internal code. Check beforehand that this test version is not registered.

This test demonstrates **refusal of an unregistered version**. On its own,
it does not demonstrate all boundaries between users and workspaces:
controlled tests of authorization, altered cursors and modified fingerprints
have separate evidence. Do not spoof a user identifier in the chat.

## F. Show scientific results at their actual level of evidence

**Target: KPI 2 / WP0–WP4 and evaluation of B5.** These sequences open existing
evidence and artifacts. They do not rerun the studies in the chat.

| Title card to show | Result to explain | Public link |
| --- | --- | --- |
| WP0 comparison on data from the three pilots | so4gp 1.1.2 / GRAANK versus mlxtend 0.25.0 / Apriori; 20 shared identities among 91 references, 71 explained differences, identical support values for shared identities; Linux reproduction. Not a universal ranking of algorithms. | [WP0 evidence](proofs/proof-26.md) |
| Native libraries | Six libraries, 418 records; all unqualified. The integration consumes them and refuses an unjustified predictive conclusion. | [Native capture](proofs/proof-23.md), [qualification](proofs/proof-24.md) |
| Robustness and validation | Initial thresholds: support ≥ 0.50 and conditional confidence ≥ 0.70; 0/10 spatial candidates pass at least 4/5 folds; 6/52 temporal rules pass, representing three inverse pairs, which fail at Fos. | [Evaluation](proofs/proof-22.md) |
| Forecasts against the P50 reference | 86,040 pairs per model identity; GFS positive in 11/24 aggregate MAE skill cells, AIFS in 22/24. These are comparison cells, not rates of correct forecasts. | [B5 matrices](proofs/proof-05.md) |
| NDVI predictive test | In the main test at Fos, adding patterns does not reduce MAE: ≈ 0.02263 → 0.02354; persistence ≈ 0.02237. The result is negative under this protocol. | [Ablation](proofs/proof-21.md) |
| Delivered and invoked tool | Server installation, contracts and calls; application interface for presentation, storage and notifications. | [Delivery](proofs/proof-02.md), [tool guide](../../reference/mcp-tools.md) |

Watch the denominators: **130 N1 observations**, **418 native patterns**,
**179 patterns in the Q3 catalogue** and **149 TGrad in B6** are different
objects. The two admissible patterns in D do not scientifically qualify the
418 patterns from the three pilots.

Closing statement: “We have delivered data access, execution and traceability
of calculations. The pattern experiments are reproducible, but do not
establish a usable predictive gain within this scope. Weather threshold
alerts remain a separate, functioning service.”

If a RAG presentation is desired, an **authorized copy of the public annex**
can be placed in the workspace, followed by this request:

```text
In the supplied FIERCE technical annex, summarize the results of WP0, WP2 and the NDVI predictive test. Cite the passages used and distinguish positive, negative and unproven results.
```

The operator must check the response against the source passages before
recording. This sequence shows document retrieval; it does not turn the LLM
into a scientific evaluator or a benchmark executor.

## G. Optional: compare a forecast with climate normals

**Target: B6 level 1.** Evidence: [actual manual run](proofs/proof-13.md).

This workflow requires a **monitoring agent already configured for manual
Site-Watch monitoring** by the operator, in the same workspace and the open
conversation. A generic monitoring agent is insufficient.

The demonstrated historical profile uses `fos_sur_mer`, rain and wind, with
a requested seven-day horizon. In that agent's details, click **Run**
(**Lancer**). This is neither a command to paste into the chat nor an immediate
run of the bulletin cron job.

**Expected if the configuration and normals are authorized:** a “Manual
monitoring” (“Veille manuelle”) response, weather source, horizon actually
served, **1991–2020** reference, available/missing/dry observations, bands and
`notify`, `watch`, `none` classifications. Rain uses the P90 of **wet days**;
wind uses its reference distribution. Display the sample count.

Values and signal counts depend on the retrieved forecast. The three signals
in the documented 4 October run are not promised for a new run. A `notify`
classification does not prove that an email was sent; this manual workflow
states that no sending was requested.

Say: “We place a forecast in the context of a documented historical
distribution. This is neither an official weather warning nor calibrated
predictive skill.”

Without a configured manual agent and normals, show the dated evidence and
say so. Asking the saved-forecast reader alone a question such as “Is this
rainfall unusual?” does not supply the missing normals.

## H. Optional: daily bulletin, monitoring and delivered results

**Target: application operation and rule tracking.** Evidence:
[bulletin and observed email](proofs/proof-10.md),
[publication in the chat](proofs/proof-16.md),
[bulletins and monitoring](proofs/proof-17.md).

In the workspace's agent list, open the configuration for
**GeoAI weather bulletin / Bulletin GeoAI**. Show:

1. **Bulletin and alert location** (**Lieu du bulletin et des alertes**): GPS
   point or reference location.
2. **Daily bulletin** (**Bulletin quotidien**) and its **Daily time
   (Europe/Paris)** (**Heure quotidienne (Europe/Paris)**).
3. **Threshold monitoring** (**Surveillance des seuils**), its enabled status
   and **Check frequency** (**Fréquence des contrôles**).
4. **Add an alert** (**Ajouter une alerte**): for example, rain **at least
   20 mm**, then minimum temperature **at most 0 °C** for indicative frost
   monitoring.
5. Delivery to **my account's email address** (**l’adresse de mon compte**),
   and the destination conversation if that publication is configured.

Prepare the settings and save them with the account owner before recording.
To demonstrate sending, plan an actual scheduled run or use a dated historical
trace. Preparing this guide activates no rule and sends no message.

**Expected:** bulletin time independent of check frequency, settings retained
after reopening, then a trace of the actual run and the result in the selected
channel. With daily use enabled and no alert configured, the daily bulletin
remains available. SMTP acceptance of an email is not, on its own, proof of
receipt; show the recipient inbox if receipt is claimed, hiding personal data.

Rules to explain:

- Independent rules combine with **OR**. **AND** groups, when offered by the
  active configuration, require their conditions to hold on the **same UTC
  day**; groups combine with OR.
- Seven pilot variables: mean temperature, minimum temperature,
  precipitation, humidity, evapotranspiration, wind and solar radiation, with
  their units. This panel's comparators are inclusive ≥ and ≤.
- A forecast minimum temperature ≤ 0 °C is a user-selected threshold, not a
  local measurement or a guarantee of crop damage.
- Monitoring avoids repeated notifications of the same exceedance; a missing
  value does not become a confirmed return below the threshold.
- Hourly checks do not guarantee a new provider calculation every hour. The
  pilot evaluates daily aggregates, not instantaneous gusts, minute-by-minute
  changes or multi-day totals.
- Publication in a conversation depends on its configuration; enabling email
  does not produce it automatically. NDVI is not a variable in this weather
  bulletin.

Say: “The application schedules acquisition, applies explicit rules and keeps
a record of the result. This service does not depend on a scientifically
qualified pattern library.”

## I. Possible supplement: workbook sites and solar data

This capability broadens the use cases, but is not required to demonstrate
the three FIERCE indicators. Prepare a completed, authorized batch before
recording. Replace the following identities with those of the batch:

```text
solar_workbook {"action":"read","space_id":"<ESPACE_UUID>","lot_id":"<LOT_UUID>"}
```

**Expected:** sites, identifiers preserved as text, coordinates from the
workbook, per-site statuses, source cells, source of solar values, UTC dates,
cadence, units and attribution. Reading is not a new collection. A new
collection requires its own parameters, provider permissions and checks; do
not improvise it during recording.

This result alone does not calculate photovoltaic yield, carbon sequestration
or biodiversity.

## 4. What the video covers, and what remains in the supporting evidence

| Technical objective | Sequences to record | Supplementary evidence to retain |
| --- | --- | --- |
| B1: zones | A, point map in B with its scope made explicit | [Zone scope](proofs/proof-02.md): 12 panel zones + 3 pilots; the weather map does not prove their boundaries. |
| B2: ERA5 data and normals | A and G if configured | [Coverage](proofs/proof-09.md), [materialization](proofs/proof-07.md): 15 zones, six variables, 1991–2020; a video showing a few rows does not verify the whole set. |
| B3: Sentinel-2 / NDVI | A and C | [Archive and code](proofs/proof-03.md); acquisitions in C do not, on their own, prove the complete archive. |
| B4: unified data | A | [Join and provenance](proofs/proof-04.md), [readback](proofs/proof-11.md). |
| B5: forecasts and comparison | B and F | [Matrices, coverage and models](proofs/proof-05.md). No gain is inferred from a forecast capture. |
| B6: two analysis levels | D and G | [Acceptance of the explanatory scope](proofs/proof-06.md); no calibrated confidence added. |
| WP0 to WP3: comparison, patterns, evaluation | A4 and F | [WP0](proofs/proof-26.md), [capture](proofs/proof-23.md), [validation](proofs/proof-22.md), [Q3 recurrence](proofs/proof-12.md), [qualification](proofs/proof-24.md). |
| WP4: software delivery | Introduction and functional workflows | [Packaging and adopted scope](proofs/proof-02.md), [so4gp traceability](proofs/proof-25.md), contracts and MCP guide. The video does not measure an ARM64 image size. |
| KPI 4 / KPI 7: three indicators, three pilots, combined analysis | **A1–A4 essential** | [Indicator mapping](fierce-kpi-evidence.en.md#6-kpi-4--three-indicators-across-the-three-pilots). Sentinel-2 NDVI + two ERA5 quantities: not three independent satellite measurements. |
| Integration and maturity | A to H as configured | [Application boundary](proofs/proof-14.md), [TRL assessment](fierce-kpi-evidence.en.md#9-objective-maturity-assessment-and-trl-evidence). Self-assessed TRL 6 for the integrated demonstrator; TRL 7 not established; new pattern-based prediction not validated. |

Scope adaptations must remain explicit: AIFS and physical fallback replace
the initial FourCastNet comparison, 30 years of ERA5 delivered, skill measured
through D7 and an explanatory score instead of calibrated probabilistic
confidence. See the [mapping to the plan](fierce-kpi-evidence.en.md#5-kpi-3--d3--the-six-geospatial-pipeline-modules).
The video must not promise validated quality through D16.

The other impact KPIs do not call for a figure inferred from this video:
displaying an alert establishes no measured savings in resources, pollution
or carbon. The market study is outside the scope of this guide.

## 5. Trace record to complete after rehearsal and after recording

Keep this record with the video. The public version must contain no secrets,
private conversation links or personal addresses.

| Item | Complete with the actual observation |
| --- | --- |
| Date and environment | Date/time with time zone; `.ai` or the actual rehearsal environment name. |
| Versions | GeoAI server and application client revisions used; package/case versions. |
| A — Data | 100 + 30; filters 41 / 48 / 41; three indicators; readback after reload. |
| A4 / F — Analysis | Links to artifacts actually shown; descriptive calculations distinguished from pattern analysis. |
| B — Forecast | Coordinates, served dates, source, collection time and threshold results; values retained after reload. |
| C — NDVI | Area, dates, actual acquisition count; whether the rich panel and reload were actually observed. |
| D — B6 | Version; 149 / 147 / 2; scores; sources and verification of the saved result. |
| E — Refusal | Unregistered test version; no substitution or invented values. |
| G / H — Monitoring | Workflow actually run, or dated historical evidence; trace and receipt distinguished. |
| Video | Final URL and sequence timestamps; any cuts disclosed. |
| Deviations | Incident, missing authorization, missing data, feature not shown and identified replacement evidence. |

The recording demonstrates the actions and results observed in its
environment. Scientific evidence and internal acceptances remain the linked
documents; they do not become external FIERCE approval.
