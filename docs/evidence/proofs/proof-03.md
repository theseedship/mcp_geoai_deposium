# Technical evidence 03

Source: `docs/demo/b3-ndvi-shipped-code-proof-2026-09-04.md` at `d4bea08f41b20b337ee8152704c7ed5231876072`.

Original SHA-256: `88d0bf4b8b84b3c831bc69329ab9e8186a8775979068cbf30ae0b59772320fb0`.

This is a curated technical extract from the dated source, with private navigation and operational identities omitted. It records documented evidence; publication does not independently rerun or certify the experiment.

## Selected source excerpts

Third and final B3 note. `b3-ndvi-pilot-panel-2026-07-29.md`
proved the mechanics; `b3-ndvi-durable-run-manifest-2026-07-31.md`
added the run manifest. Both were produced from a branch, and the 07-31 manifest was
`provenance.mode: aggregation_only` with `artifact_code_ref: null`, from a **dirty** working tree — so
neither named the code that made the artifacts.

The coverage row is the substantive one. `truncation_warnings` only copies a tool-emitted
`metadata.ndvi_truncation_warning`: the tool emits that warning when an all-scenes search
(`monthly=false`) reaches the STAC item cap. This monthly panel cannot populate it, so an empty list
does not demonstrate coverage. `checks.month_coverage` supplies the missing denominator.

Manifest at `[storage artifact]`:

This is the check that could not run before [internal record]: the consumer raised an unhandled traceback on an NDVI
manifest, because `_manifest_expectations` required an ERA5-only `parameters.variables`.

```text
$ unzip -q … && duckdb -c "SELECT zone_id, count(*), min(date), max(date)
                           FROM read_parquet('climate/ndvi/*.parquet') GROUP BY 1"
bassin_du_lez           81   2016-11-14 .. 2024-12-27
fos_sur_mer             92   2016-11-11 .. 2024-12-29
montpellier_metropole   81   2016-11-14 .. 2024-12-27
```

Row counts and spans reproduce the 07-29 and 07-31 panels exactly. Grepped for the dev store's
credentials in the bundle bytes: absent.

## Interpretation boundary

Internal engineering reception is distinct from external programme approval. Historical or synthetic qualification does not establish production activation, causal inference, calibrated probabilities or general predictive superiority. Preserve the source date and scope when using these excerpts.
