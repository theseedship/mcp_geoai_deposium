# Technical evidence 05

Source: `docs/demo/b5a-forecast-collection-review-2026-10-15.md` at `d4bea08f41b20b337ee8152704c7ed5231876072`.

Original SHA-256: `cee0cb0639aa3316cae32ae66529be8e9c4a4b05aa348d0d45cda1db6736c0c8`.

This is a curated technical extract from the dated source, with private navigation and operational identities omitted. It records documented evidence; publication does not independently rerun or certify the experiment.

## Selected source excerpts

the contributor/maintainer's full collection prerelease
was published at **20:25:20 UTC**, targeting collector commit
`1efa0ff5aa23c5bcfd6c0b00f3b983529fe1351d`. The collection and matrices are
supplied; this review requests no repeated acquisition. No scientific acceptance,
issue closure or B6 probability is established by this record.

- Collection ZIP: **7,368,745 bytes**, SHA-256
  `30c913ce2aec2d3eac16c9391642fbd8c806a2d8cec0208b167270d714a31c74`.
- The ZIP has 209 unique safe entries, no symlinks; all **203** `FILES.sha256`
  entries match the complete file inventory, excluding the checksum file itself.
- All nine files of the reference/P50 bundle
  match its manifest. ZIP SHA-256:
  `e223e48ad14753a8e2a4bc1f3cb86e6525788cac2b8a0352fbdbe21f625ed0c8`.
- **30** hashed raw responses, **30** per-request provenance files, **120**
  available artifacts and **30 saved HTTP-200 events** were checked. This verifies
  archived evidence, not current provider availability.
- Each identity has **86,040 expected/scored pairs**:
  15 zones × 239 dates × six variables × four horizons. There are no missing
  pairs or exclusions. The valid-date window is 25 February–21 October 2025.
- Exports reproduced byte for byte. Two offline scorings per identity reproduce
  all published scientific matrix fields and `result_digest` values.

| Identity | Reproduced `result_digest` | Positive aggregate MAE-skill cells vs P50 |
|---|---|---:|
| GFS | `64274cb058491af829c233e1c83772d56fadede07d31583283788a9089011713` | 11/24 |
| AIFS | `222a3a228e2b15f6bdffb27abc1ecaa32ae407d47e863b7359f3dcf907cd2dbc` | 22/24 |

The machine-readable audit
records full artifact hashes, inventories and replay provenance; its SHA-256 is
`3121885881ce77e656e740230848756a1459c5d98d8ab200601acd1e85f0824f`.
The export/scoring sources match the release target. Local replay used GeoAI
`9999c3a1d629ed3212dc197ad127adac2069649a` with documentation changes preserved
and `--allow-dirty` recorded. Whole matrix byte hashes differ from publication
because `code_ref` records that checkout; scientific fields and result digests
match exactly. The two local replay byte hashes match each other.

These are point-forecast **MAE** comparisons with a monthly P50 baseline on the
frozen panel. Positive aggregate cells do not imply positive skill in every
zone. GFS humidity and wind lose to P50 at every horizon; aggregation and grid
representativeness need scientific interpretation. No CRPS, calibrated event
probability or operational availability was measured. Attested model version
and initialization remain null; `origin_date` is a lead label.

The archived report's opening “not published” status predates publication;
GitHub release metadata establishes the later prerelease. No acquisition rerun
is necessary for that wording difference. Reconcile [internal record]'s criterion-level
acceptance separately, retain [internal record] scientific ratification and agree any B6/J1
consumption explicitly. This does not qualify the contributor/maintainer's native pattern records.


## Interpretation boundary

Internal engineering reception is distinct from external programme approval. Historical or synthetic qualification does not establish production activation, causal inference, calibrated probabilities or general predictive superiority. Preserve the source date and scope when using these excerpts.
