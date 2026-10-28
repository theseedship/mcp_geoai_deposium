# Technical evidence 16

Source: `docs/design/geoai-bulletin-chat-publication.md` at `d4bea08f41b20b337ee8152704c7ed5231876072`.

Original SHA-256: `d2e3e11f946f6d813204528acfcea3b72a4bde3efbbbaa5e18ae13d0adea24f6`.

This is a curated technical extract from the dated source, with private navigation and operational identities omitted. It records documented evidence; publication does not independently rerun or certify the experiment.

## Selected source excerpts

The GeoAI agent has a **Publication in a chat** setting: disabled, a new chat
for each qualifying occurrence, or an existing conversation in this Space.
Only the actual Space owner can save the destination. Saving changes this
channel's configuration; it does not execute a forecast or send a message.
Existing settings default to disabled. Email, the daily local time and the
surveillance cadence retain their own settings.
The owner editor has French and English labels. The first slice renders the
scheduled forecast with fixed English prose. The separate
saved-language follow-up
adds a saved message-language preference, distinct from the editor locale;
publication itself does not interpret a natural-language request.

MCPs freezes a publication record with the completed forecast and independent
daily/crossing reasons. It freezes the final message and artifact before writing
the conversation, with stable chat, message and execution identifiers. The
publication worker reads this queue independently of weather acquisition and
the email outbox. Publication retries make no new provider request or email.

- Existing weather settings, forecasts, crossing state and mail history remain
  unchanged when the migration is applied and replayed.
- Chat-only, email-only and both channels behave independently; daily plus
  surveillance produces one acquisition and one chat message.
- Concurrent publishers and recovery after a conversation write retain one
  message and preserve intervening user messages.
- Owner, plugin, assignment, configuration and unavailable-destination guards
  refuse unauthorized new writes.
- Stored receipts preserve all seven pilot variables, zeros and missing values;
  graph and threshold follow-up use the same captured result.
- CI, review, merge, environment migration, deployed commit and the actual owner
  Save/next-occurrence/chat follow-up are recorded as distinct checkpoints.

| Checkpoint | Evidence and limit |
|---|---|
| Runtime implementations | MCPs [internal record] and Solid [internal record] merged with green CI. |
| Migration 080 | Applied and verified in local, staging and production, with preservation checks and private backups. |
| Compiled deployments | MCPs and Solid verified in staging and production; this does not prove the owner browser journey or scheduled delivery. |
| Actual-owner settings read | HTTP 200 in local and staging. The known Space is absent in production; no positive production owner read is qualified. |
| Access refusal | Solid returns HTTP 401 without a session in staging and production; the production settings RPC returns HTTP 403 / `42501` for an invalid actor. Neither is an owner Save check. |
| Local application worker | Still on the earlier runtime; excluded from deployment/publication proof. |
| Owner Save/reload | Pending; no destination is automatically activated by migration or deployment. |
| Next qualifying occurrence | Pending, including persisted chat visibility and saved-result follow-up. |

The common follow-up scope records AND v6,
message-language v7 and multilingual request extraction, then independent
writer/v1. MCPs [internal record] is merged, migrations 081–083 are verified in
local/staging/production, and the backend is deployed in staging/production.
Staging owner reads and production availability/access refusals are verified;
the expected owner Space is absent in production. Solid [internal record] is merged; compiled
frontend deployments are verified in staging/production, without owner UI or
frontend route execution proof.
The writer is optional and off by default: it rereads a named signed forecast,
uses authorized Space documents and adds a separate attributed interpretation
with citations. Publishing the factual message alone
does not qualify that analysis or actual requested-language synthesis.

## Interpretation boundary

Internal engineering reception is distinct from external programme approval. Historical or synthetic qualification does not establish production activation, causal inference, calibrated probabilities or general predictive superiority. Preserve the source date and scope when using these excerpts.
