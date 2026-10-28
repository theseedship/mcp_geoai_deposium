# Technical evidence 04

Source: `docs/demo/b4-local-offline-pilot-2026-09-29.md` at `d4bea08f41b20b337ee8152704c7ed5231876072`.

Original SHA-256: `0bbba6f1c641dfc9981b6dd64d04f57fe14c6bce596b73457c144aadc3af039a`.

This is a curated technical extract from the dated source, with private navigation and operational identities omitted. It records documented evidence; publication does not independently rerun or certify the experiment.

## Selected source excerpts

Scope set by the maintainer's [internal record] comment of 2026-09-29: an offline, provider-free test panel
over the existing pilot archives, with no Open-Meteo, ERA5/CDS or STAC call and no write to the
mesh, dev or any object store.

| Asset | SHA-256 | Check |
|---|---|---|
| `geoai-era5-holdout-2021-2025-handoff.zip` | `3a03059e6a0f75e77fb355b68e2535b0e841e2bb460ac0d2574a0104c7f5937d` | `shasum -a 256 -c SHA256SUMS`: 6/6 OK |
| `bundle-manifest.json` | `0653dc43727d0810122f9919b1a6a4232c98f0f1b30d345e28d3067c4bced14a` | in `SHA256SUMS` |
| `verify_era5_holdout_bundle.py` | `66f826dbc5d8da10cdcfd218a2e2e47b46a47a59dd390fb939254ce7da734e8a` | in `SHA256SUMS` |
| `offline-verification.json` | `cb6c3f604dd303d5debdb669a8a4aab6d070d051d104f4ba1b8bd4d26cc8796d` | in `SHA256SUMS` |
| `geoai-b3-ndvi-pilot-3zones-20260822.zip` | `0cd6dd981bb78916026fc1585cd856ba81e89deeaea24158973513bf6ad5526c` | equals `archive_sha256` in the B3 verification JSON |
| `verify_b3.py` | `c5a8e70f9ef20314305a5e441fd5283aea2993ff9413ce56f65a0d147ccd975f` | equals `verification_script_sha256` in the B3 verification JSON |
| `b3-verification-2026-09-05.json` | `c69afb8c2ce7c325ec6b6d8b58d5873695291b393e9fb82ed2321be3912e16df` | no independent published digest; recorded here |

- **ERA5.** All 19 zip members are relative paths; the embedded `OBJECT-SHA256SUMS` passes 18/18,
  and the embedded `bundle-manifest.json`, README and verifier are byte-identical to the release
  assets. `verify_era5_holdout_bundle.py` was read in full (local DuckDB only, no network) and run
  on the extracted directory rather than the zip, so it writes nothing: `ok: true`, 15 objects,
  27,390 rows, and its JSON is **identical** to the published `offline-verification.json`.
- **B3.** All 6 zip members are relative paths; the three Parquets and `manifest.json`
  (`1b2a715055673c5a1d573b5597d86780dd023f13e83f602e4b8b2bb15867a6b5`) hash to the values in the
  verification JSON. `verify_b3.py` was read in full (local DuckDB and two read-only git probes)
  and run **unchanged** against a `git archive` of its required commit `db3ede05`. The agent
  sandbox forbids git outside its worktree, so a wrapper answered only those two probes with the
  archived commit id and a clean status; every other check ran as published. Its JSON equals the
  published one except `verified_at`.

| Zone | ERA5 object SHA-256 | B3 NDVI SHA-256 |
|---|---|---|
| `bassin_du_lez` | `bbd22e1b7ed6d64592702a9e5eb829d84eff6ba10328b9f6dbf0abc517e730c8` | `f4d297b777b5cb2e7baefb13ab3516c680e6922a3df340048930b7e566104c1a` |
| `fos_sur_mer` | `c7230e53fe6ed0634830184ddf233c2bc17f3a4be68a5a00fc1e0d94ea0cff1d` | `b036773526ac5f271b902dd00a5ac2ecdfbc836467595397cadb17f6c033d3cb` |
| `montpellier_metropole` | `c3be7a6096b017cf7afa6cba3a66b47bd5f61d94860490c0c516e7ee892b48cb` | `67f9b1f15ce8e3ab758709a9614efede092a44f4ca4ac7aafcbf74186240a9a8` |

- 9 units, one per (zone, tolerance): `{zone}__2021_2024__pm{1,3,5}d`, each with the zone, the
  window, its output path `$OUT/unified/pm{N}d/{zone}.parquet` and both local sources with their
  SHA-256;
- the two archive digests above, and that both release manifests were checked;
- alignment `nearest-date within ±N days`, `proposed_tolerance_days: [1, 3, 5]`, status
  `proposed for sensitivity readout only; not ratified (Decision 3)`, `window_contract:
  intersection`;
- code/config identity: `code_ref` (commit `568b064`, `dirty: false`), DuckDB `1.5.4`, and the
  SHA-256 of `scripts/b4_local_pilot.py`, `scripts/unified_pilot_panel.py`,
  `src/tools/climate.py`, `src/run_resume.py`, `src/cli/run_resume.py` and
  `zones/zones.geojson`. `run` refuses when any of these changed since `freeze`, because the
  resume fingerprint covers the inputs and not the code that reads them;
- three provenance notes: the ERA5 objects predate `8046597` (old bbox-centre points) and
  accept-versus-resample stays open; the B3
  series is the 2015–2024 run and no 2025 is used; the tolerances are proposals.

## Interpretation boundary

Internal engineering reception is distinct from external programme approval. Historical or synthetic qualification does not establish production activation, causal inference, calibrated probabilities or general predictive superiority. Preserve the source date and scope when using these excerpts.
