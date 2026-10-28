# Technical evidence 13

Source: `docs/demo/site-watch-manual-staging-2026-10-04.md` at `d4bea08f41b20b337ee8152704c7ed5231876072`.

Original SHA-256: `2c61b42ff2a3d97215c484758237b870cd0517b4f15011a30ac90652bdeb845e`.

This is a curated technical extract from the dated source, with private navigation and operational identities omitted. It records documented evidence; publication does not independently rerun or certify the experiment.

## Selected source excerpts

| Day | Variable | Forecast | Reference P90 |
|---|---|---|---|
| 7 October | Precipitation | 36.2 mm | 23.16 mm, wet days |
| 8 October | Maximum wind at 10 m | 9.52 m/s | 9.492 m/s |
| 9 October | Maximum wind at 10 m | 9.60 m/s | 9.492 m/s |

The first HTTP attempt then returned 503 before execution creation: the real
authority RPC returned `PGRST202`, although its installed signature and privileges
were correct. Readback found zero execution, request claim or frozen input.
Restarting only the staging PostgREST service rebuilt its schema cache; the same
RPC then parsed successfully with `delivery_disabled`, which proves owner scope
without authorizing mail. The successful manual request above followed that
check. No forecast was fetched by the refused attempt.

## Interpretation boundary

Internal engineering reception is distinct from external programme approval. Historical or synthetic qualification does not establish production activation, causal inference, calibrated probabilities or general predictive superiority. Preserve the source date and scope when using these excerpts.

## Positive manual run and retained replay evidence

One real manual Site-Watch run and its completed replay passed in staging on 4 October 2026. The authenticated path fetched weather, compared registered normals, persisted input/result and appended one response. The original fourteen conversation messages were unchanged. The accepted request returned HTTP 200 at 10:43:05.102 UTC, with persistence completed at 10:43:05.972936 UTC. Provider: Open-Meteo AI, model ecmwf_aifs025_single; retrieval 10:43:05.351 UTC, with no model issuance time claimed.

At 10:44:02.913 UTC, the identical idempotency key returned HTTP 200 with idempotent:true. Independent readbacks matched one execution, one frozen input and one appended message; zero delivery operations or attempts occurred. The replay reused stored data without new weather acquisition. Maintainer-reported rendering is distinct from an automated browser capture or user-triggered UI qualification. No email was sent.
