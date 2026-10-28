# Technical evidence 26

Source: `docs/research/wp0-copernicus-2026-10-27/README.md` at `d4bea08f41b20b337ee8152704c7ed5231876072`.

Original SHA-256: `7f4a992f937757717c9d10355007c4c8cd808978864244747f0a3e4e51d52c0c`.

This is a curated technical extract from the dated source, with private navigation and operational identities omitted. It records documented evidence; publication does not independently rerun or certify the experiment.

## Selected source excerpts

The 25 October agreement authorizes native so4gp GRAANK 1.1.2 versus mlxtend Apriori 0.25.0, the three unchanged frozen inputs and seven features, itemsets 2–7, one CPU configuration and 60 seconds per method/pilot. The historical optimizer roster is superseded for this route. All six jobs requested in the delivery instruction completed on 27 October. The handoff below contains the measured outputs and reproducible environment; reception remains the contributor/maintainer/Deposium's independent step.

The manifest records implementation commit `9625867e694eb0749de39d8cd933709f654091e8`, configuration, environment, commands and the hash/size of all 37 retained artifacts. Its SHA-256 is `78236811a8c008ef37f8c5cec36f8579f65b097ad5db7ad99928388e8b84d5a4`. All six child processes exited successfully within their 60-second caps; there were no timeouts, parse errors or support mismatches. The independently counted reference enumerates all 1,086 inverse-canonical signed itemsets of sizes 2–7 per pilot, including nonqualifying ones.

Here “unique” identifies inverse-equivalent itemsets without summing supports. Apriori's raw multivariate records number 72/50/60 before this identity reduction; each pilot also retains 12 raw frequent singletons, excluded from the agreed 2–7-item comparison. Native GRAANK returns 20 unique identities across the pilots, all among the 91 strict reference identities; Apriori covers all 91 on these inputs. There are no native-only, reference-only, exact-threshold or below-threshold identities in the comparison view. This is an observed result for this configuration, not a general search-equivalence claim.

All 71 Apriori-only identities retain evidence for their output difference: 29 contain a feature removed by the native singleton gate; the other 42 have a strict returned-superset witness, aligned modulo inversion. For example, Lez `humidity+ / ndvi_mean+` has 545 hits: native-scale support `545/820 = 0.664634…` and transaction support `545/1640 = 0.332317…`. Its native humidity singleton gate is `403/820 = 0.491463…`, below the native strict 0.5 gate. Lez `evapotranspiration+ / ndvi_mean-` has 554 hits and is a subset, after inversion, of native raw record 4 (zero-based). The per-identity comparison CSV exposes these counts, pruned features and raw witness indexes. A superset witness explains an output convention difference; it does not prove guaranteed maximal output or exhaustiveness. No unexplained residual remains in this run.

Process times include startup/imports and enforce the cap; method times use the boundary below. In particular, the first native process spent most of its wall time outside that method timer. These single observations on Windows 11, Python 3.14.0 and CPU Torch 2.13.0+cpu support no speed ranking. No configuration was tuned or scientific job repeated to obtain a favorable outcome. Native descriptor values, including values called confidence above one, remain verbatim raw output and are not treated as probabilities.

The initial contributor command uses the already verified local `research-data/pilots` input directory and the same committed configuration. The result package retains original 4,274/4,576/4,325-byte Parquets and the actual float64 matrices used, so reproduction need not refetch or serialize source data. No provider collection is part of this command.

The `results/` directory includes `inputs/` (three original Parquets), `matrices/` (exact consumed values and dates), `raw/` (six unfiltered method wrappers and source fingerprints), `logs/` (all twelve stdout/stderr files, empty on success), and `derived/` (complete reference CSVs, comparison CSVs and JSON). Per-pilot checkpoints preserve status. A fresh execution may have different timings, paths and ordering of native output; compare canonical identities, independently counted support and explicit difference classifications. Neither execution nor artifact verification substitutes for independent scientific reception.

## Interpretation boundary

Internal engineering reception is distinct from external programme approval. Historical or synthetic qualification does not establish production activation, causal inference, calibrated probabilities or general predictive superiority. Preserve the source date and scope when using these excerpts.

## Independent Linux reproduction documented on 27 October

An independent execution of reviewed head 6029bb324fadd3d926e7f5486b3ab30e05e3ed40 used fresh Linux CPython 3.14.2 with the same direct numerical pins, so4gp 1.1.2, mlxtend 0.25.0 and Torch 2.13.0+cpu. Original configuration, inputs and measured source bytes were unchanged. All six jobs completed within the original 60-second caps without scientific retries. All three derived comparison JSONs matched identities, supports and difference classifications; reference CSV bytes matched. Native raw-payload differences were confined to runtime. Four focused modules passed 102 isolated tests. This documents reproduction of measured identities/support, not a byte-identical environment replay, speed ranking, predictive validation or scientific acceptance.
