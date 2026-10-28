# Technical evidence 17

Source: `docs/design/geoai-bulletin-surveillance.md` at `d4bea08f41b20b337ee8152704c7ed5231876072`.

Original SHA-256: `db8e7dcdde28956f3d8766c7274d50f81b7a93ba8c465630b95b5678cb1dee41`.

This is a curated technical extract from the dated source, with private navigation and operational identities omitted. It records documented evidence; publication does not independently rerun or certify the experiment.

## Selected source excerpts

The **threshold surveillance** checks the enabled rules at a selected cadence:
every one, two or six hours, anchored to whole UTC hours. The compatibility
option remains once a day at 08:00 Paris, independently of the daily bulletin's
selected time. It sends only newly detected crossings for a forecast date;
several matches are grouped into one message. A known return below an upper
threshold rearms it; missing values do not. Lower-threshold rules follow the
opposite comparison. Changing the bulletin's hour or variables does not reset
the alert memory.

For example: send a bulletin at 08:30 Paris, and check rain, wind and minimum
temperature every two hours. Both uses share the same agent and account-email
authorization. If due together, one forecast serves both and one grouped email
contains the bulletin and new alert matches.

Legacy scheduled settings retain **Fos-sur-Mer**; the separate
v5 location contract supports a saved preset
or GPS point. The horizon remains the next **three UTC days**.
These are numerical forecasts at a provider grid point, not station observations.

| Variable | Unit | Daily meaning |
|---|---|---|
| Temperature at 2 m | °C | Mean |
| Minimum temperature at 2 m | °C | Minimum over the full UTC day; optional live-forecast extension |
| Precipitation | mm | Total |
| Relative humidity at 2 m | % | Mean |
| Reference evapotranspiration | mm | FAO ET₀ total |
| Wind at 10 m | m/s | Maximum wind speed, not gust speed |
| Solar radiation | W/m² | Daily mean converted from the provider's energy total |

Each rule uses an inclusive upper (`>=`) or lower (`<=`) threshold. There are
up to ten independent rules in v1–v5, combined by OR across rules and forecast dates.
The v6 follow-up
adds AND groups alongside those rules, with ten top-level triggers in total and
at most ten operands per group. Enabled operands must match on the same UTC
forecast date; missing data do not count as normal or rearm a crossing.
Accumulated multiday thresholds and rate-of-change rules remain separate
extensions. A minimum-temperature threshold can identify modelled
freezing risk, not measured frost or damage at a site.

A fresh check retrieves and evaluates a forecast; it may retrieve unchanged
provider output. Open-Meteo documents an **AIFS Single update every six hours**,
with native six-hourly forecast steps. Its API can interpolate hourly points;
those are not independent hourly model calculations. The physical fallback's
update frequency depends on its selected model. Publication delay also matters.
See ECMWF API and
model updates.

The current collection timestamp is not a certified model issuance timestamp.
Checks of daily totals and maxima follow revised expectations for upcoming UTC
days. Detecting a change expected in the next hour requires an additional hourly
or short-range forecast contract; observing a change already happening requires
observations or nowcasting.

| Family | Possible indicators | Resolution and availability |
|---|---|---|
| Weather extensions | Maximum/apparent temperature, gusts, wind direction, rain probability, pressure, cloud cover, UV, soil temperature and moisture | Hourly or daily data depending on model and variable; model update cadence is separate. Availability and normalization must be checked before activation. |
| Satellite observation | NDVI, vegetation variation and GeoAI vegetation-stress signals | These tools already exist in chat. Fresh observations depend on actual Sentinel-2 acquisitions, clouds and sampling; usable observations may be several days apart. Scheduled satellite email is not included in this weather pilot. |
| Climate references | Monthly P10/P50/P90 and historical comparisons | The available 1991–2020 normals are a historical reference, not a continuously changing forecast or real-time observation. ERA5 reanalysis is a separate historical product. |

## Interpretation boundary

Internal engineering reception is distinct from external programme approval. Historical or synthetic qualification does not establish production activation, causal inference, calibrated probabilities or general predictive superiority. Preserve the source date and scope when using these excerpts.
