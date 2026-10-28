# Technical evidence 10

Source: `docs/demo/geoai-bulletin-readiness-2026-10-06.md` at `d4bea08f41b20b337ee8152704c7ed5231876072`.

Original SHA-256: `3b6b426638220aa26c135a87391d766501ae5f0aacb323bccdac910b043d824d`.

This is a curated technical extract from the dated source, with private navigation and operational identities omitted. It records documented evidence; publication does not independently rerun or certify the experiment.

## Selected source excerpts

The scheduled `geoai_bulletin` pilot uses Fos-sur-Mer, the next three UTC forecast
days and one occurrence at **08:00 Europe/Paris**. Settings belong to each space;
the system agent definition is shared. The current space owner enables the
bulletin and optionally email to their account address.

The original configuration chooses daily email or threshold email. The approved
follow-up in MCPs [internal record] and
Solid [internal record] allows daily
email and up to ten independently enabled rainfall/wind
alerts together. Each alert has its own inclusive threshold. Rules share one
forecast and matching alerts are grouped into one email per occurrence.
Migration `073_geoai_bulletin_rules.sql`, SHA-256
`ac200e7c385b5e0c64559bc41ec5c6eacfb3bd0f12944c27c4e94d55f4e96597`,
was applied to local, staging and production on 6 October. Postchecks verified
legacy and V2 acceptance, service-role-only helper access and unchanged settings.
The already enabled legacy staging bulletin was preserved; the migration did not
activate a new assignment or rewrite a frozen execution.

MCPs [internal record] merged as `afcb9c26c949a22923021693a26f0ffacd95baf6` after all twenty
reported checks passed. At 20:52 UTC, staging deployment
`[identifier omitted]` and production deployment
`[identifier omitted]` were successful at that source commit.
In each running service, the compiled contract accepted V1 and V2 public configs
returned by actual PostgREST RPCs, with email disabled in the probe. Health checks
passed. These checks fetched no weather and sent no messages. Solid [internal record]'s owner
controls are a separate source/deployment gate.
Locations, horizons and between-bulletin monitoring remain deferred.

Observe one real scheduled execution at the Paris occurrence, its saved forecast
and visible history, then an explicitly enabled account-email delivery and inbox
receipt. Confirm no-match/missing-only behavior, rule edits, email off and plugin
removal against pending mail. Unit fixtures and SMTP authentication do not replace
these observations.

The pilot uses user thresholds on daily forecast aggregates. It does not establish
official weather warnings, forecast skill or a new FIERCE reception. The manual
Site-Watch seven-day forecast/normal-comparison path retains its own
staging qualification.
The original 072 source guides are the
MCPs runbook at `a96d369`
and Solid controls guide at `343739a`.
Rules-aware release steps are in the
MCPs runbook at `afcb9c2`;
the companion Solid PR above owns the control release.

Read-only staging inspection found the existing owner-enabled legacy V1 bulletin
had naturally executed on **7 October at 06:00 UTC (08:00 Europe/Paris)**. Its slot
completed in one attempt with a saved real `openmeteo_physical` forecast for
7–9 October. The daily-email outbox recorded one delivered attempt and a provider
message receipt: **one SMTP handoff was accepted**. No test email, activation or
manual replay was performed by this inspection. This establishes saved scheduled
execution and SMTP acceptance, without establishing inbox receipt or live combined
V2-rule behavior.

At the 06:56 UTC readback, production had zero enabled bulletins/emails and zero
scheduled slots; neither environment had saved V2 settings. Remaining checks are
authenticated owner setup/Save of combined rules, scheduled V2 matching and
no-match/missing-only behavior, rule edits, email off/plugin removal against pending
mail, visible history and owner inbox confirmation. The fixed location, three UTC
forecast days and morning occurrence remain unchanged.

## Interpretation boundary

Internal engineering reception is distinct from external programme approval. Historical or synthetic qualification does not establish production activation, causal inference, calibrated probabilities or general predictive superiority. Preserve the source date and scope when using these excerpts.

## Later source addendum superseding earlier pending inbox status

Later on 7 October, the owner confirmed receipt of the 08:00 bulletin and supplied its body. The schedule, retrieval timestamps, physical-provider fallback and delivery contents matched the recorded staging V1 occurrence. This is recipient-confirmed arrival of one morning bulletin; no Message-ID was provided. It does not qualify a combined-rule configuration or production activation. At the 11:22 UTC readback, staging had one completed V1 slot and production had no enabled bulletins or scheduled slots.
