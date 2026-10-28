# Technical record 12

Record class: application integration change.

Original record SHA-256: `8b59bbb2fdd5c0fe3498719b5674d512b515fc98e81f20b4b462d3ec363f7d36`.

This is a bounded export of the specifically cited private record, with private navigation and personal/operational identifiers omitted. It is an internal engineering record, not external approval or an independent reproduction.

Integration title: feat(geoai): show retained NDVI acquisitions in verified views.

Merged: True; recorded merge date: 2026-10-28T00:19:36Z.

Changed-file evidence: 27 paths inspected privately; application source is not exported.

Recorded checks: success. Check state is historical engineering evidence, not a scientific validation.

## Selected technical record

Explicit GeoAI NDVI form results can now display their actual acquisitions as a paged table and native MCPUI scatter chart after a fresh authenticated canonical read. The same retained facts survive reload; presentation never reconstructs values from Markdown or interpolates missing days.

Depends on a companion integration change: deploy the backend first. Implements P1b of the existing guided-analysis specification, following the deployed P0/B6 slices ([internal record]/[internal record]).

- Preserve the full bounded signed NDVI envelope (64 KiB / 512 observations) through existing normalization and persistence. Browser structure is not HMAC, source or authorization authority.
- Session-only readback proxy uses server identity and current membership/plan/plugin checks before and after MCPs. Missing persistence has explicit saved-read retry; no acquisition occurs on readback or pagination.
- Reuse actual MCPUI rendering only after canonical verification: local 25-row pages, original observation indices/order/duplicates, exact timestamp strings including microseconds/offsets, zero/null distinction, and scatter points against elapsed UTC days. The caption explains that sorting applies to the current page.
- Retain returned statistics, native stress criteria and unknown provenance honestly. Scene cloud cover and candidate counts establish neither pixel quality nor search completeness. Old text-only and natural-language NDVI retain their existing text path.
- Add a short first-level B6 explanation: the example explains association ranking against archived weather, without predicting an NDVI rise or event probability. B6 values and scientific flags are unchanged.
- Shared chat changes are confined to two render wrappers. Existing forms, send/admission, authorization checks, transports, libraries and database remain unchanged. Both readback proxies now distinguish a removed plugin (403/access denied) from pending canonical persistence (409/read retry).
- The scatter renderer keeps the same canvas, Chart.js instance and full point dataset during local table pagination.

Validation: the initial root combined regression run passed 562 tests across 35 files. Follow-up denial-state coverage passed 130 tests; the chart-remount fix passed 22 focused unit tests and two real Chromium tests that reproduce the defect before the fix and verify stable canvas/instance/data after it. TypeScript and scoped lint/format checks passed. Four real Chromium tests passed for NDVI and B6 at 375/1280 px, including a native scatter canvas with 26 dataset points, a real 25-row table, keyboard pagination/sources, exact values and no page overflow. Shared synthetic fixture parity with MCPs is exact (2,371 compact bytes). Existing CI includes the new retention/proxy and browser suites. SQL runner registration was refreshed only for the one added workflow test line; all other registered inputs, guard, scope and images are unchanged, and its 33 pure guard tests pass.

Delivery note and connected qualification recipe: docs/2026/roadmap-prep/2026-10-28-geoai-ndvi-observations-p1b.md. Local fixtures do not establish deployment or a live NDVI acquisition.

## Retained limitation

Reception applies only to the stated engineering scope. It does not supply calibrated confidence, positive predictive validation, new operational activation or external FIERCE approval.
