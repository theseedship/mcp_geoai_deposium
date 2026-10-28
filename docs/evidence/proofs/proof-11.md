# Technical evidence 11

Source: `docs/demo/geoai-n1-chat-qualification-2026-10-14.md` at `d4bea08f41b20b337ee8152704c7ed5231876072`.

Original SHA-256: `43121fd6e513ba171a310a1f71977c5835bea59872c7e93846275492573ea538`.

This is a curated technical extract from the dated source, with private navigation and operational identities omitted. It records documented evidence; publication does not independently rerun or certify the experiment.

## Selected source excerpts

**14–15 October 2026. Status: the selected three-pilot staging chat path is qualified — 100 + 30 exact rows, signed persistence and retained-page readback after maintainer-attested reload.** Timestamps below are UTC; final confirmation occurred on 15 October in Europe/Paris. This is evidence for the pilot-data part of the minimum technical handoff. Real pattern-library consumption, remaining refusal cases, level 2 and contractual/scientific reception remain separate.

- Unfiltered pages returned **100 + 30 = 130** rows, canonically identical to the installed source; no remaining cursor on the final page.
- Each zone filter returned its exact original rows and no remaining cursor.
- Incorrect tenant context refused with `GEOAI_SCIENTIFIC_NOT_AUTHORIZED`.
- Changing filters while continuing a page refused with `GEOAI_SCIENTIFIC_CURSOR_INVALID`.
- An unregistered version refused with `GEOAI_SCIENTIFIC_NOT_REGISTERED`.
- A changed projection digest in a retained binding refused with `GEOAI_SCIENTIFIC_BINDING_MISMATCH`.

NDVI remains dimensionless, temperature a daily mean in Celsius, precipitation a daily total in millimetres. Dates represent historical acquisition matches. Gaps remain gaps; these rows are neither forecasts nor complete daily series.

- Assistant: `assistant-[identifier omitted]`; run: `[identifier omitted]`.
- Receipt captured at `2026-10-14T21:35:48.639Z`; receipt digest `f0a4fe8f4bc642a165dd5b39cbe3eea6d58934a72e98a7e0e4b103730ebc8525`.
- Strict envelope, HMAC, four-identity scope, unique canonical assistant/run identity and final-answer policy checks passed; current registration and connector policy still authorized the binding.
- Exactly **100 rows at offset 0**, 130 matches, limit 100, no filters, continuation present. Rows were canonically identical to the first 100 sorted source observations.
- Page rows SHA-256: `b4218879e561583997ce169734a7924483df21c4728b4c876c891025c92bc957`.
- Conversation object at that read: 113,101 bytes, SHA-256 `399aed3faedaeb3749c8c407e549c0574186047a8813f21f56120f76612cebf2`. This identifies that dated snapshot; later messages naturally change the whole-object hash.

The recorded subsequent user turn requested a new three-day Montpellier forecast. Its assistant reply was saved under `assistant-[identifier omitted]`, with terminal state `answer_streamed`. This corroborates a subsequent saved conversation turn; it is not the scientific page-continuation or retained-page readback test. The forecast's scientific accuracy and receipt are not independently qualified by this record.

- Assistant: `assistant-[identifier omitted]`; run: `[identifier omitted]`; captured at `2026-10-14T21:58:10.602Z`.
- Second receipt digest: `2c220d367e696567d998656e23ce387bb22eafd7d298bf88ec864d707385daad`.
- Offset **100**, exactly **30 rows**, no filters, requested limit 100, no remaining cursor; all rows belong to Montpellier Metropole.
- Second-page rows SHA-256: `18973a1a3d8b35516ab20cb173a7f8df6af7502e15ae6c2ab2f5d70fb3478147`.
- The first and second saved pages concatenate to exactly **130 distinct zone/date rows**, canonically equal to the installed source. Combined rows SHA-256: `65cdd5008b08c79ffe6916b5ee9b6b7aaeaffc93ac4af940594cdfc609072c58`.
- Canonical conversation snapshot at this read: 154,875 bytes, SHA-256 `2b866b01ca26a5a97619920d870ce23e9bfce1b39cd166a5fc9dd878e4f1ea9c`.

- User message: `user-[identifier omitted]`; its exact content is `scientific_readback`.
- Assistant: `assistant-[identifier omitted]`; run: `[identifier omitted]`; terminal state `answer_streamed`.
- Saved assistant text SHA-256: `398fb003a2fa951fe8229ca07afbeda663c11177003cc7ca212b131f17fb9f6b`.
- Executing the deployed default readback helper against the canonical conversation returned a grounded result bound to the exact second-page receipt `2c220d367e696567d998656e23ce387bb22eafd7d298bf88ec864d707385daad`. Its text matched the saved assistant answer exactly.
- Both original page signatures, identities, current grants and source-row matches still passed. The combined 130-row source digest is unchanged.
- Canonical conversation snapshot: 164,228 bytes, SHA-256 `cac4341586f27066a56cd9c282d5ec3371c8bda1e52ebbd77be7af9bfd9e65c9`.

## Interpretation boundary

Internal engineering reception is distinct from external programme approval. Historical or synthetic qualification does not establish production activation, causal inference, calibrated probabilities or general predictive superiority. Preserve the source date and scope when using these excerpts.
