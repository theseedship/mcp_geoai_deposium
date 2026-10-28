# Execution amendment before predictive scoring

The original [protocol](PROTOCOL.md) and its hashes remain unchanged. The
first historical execution on implementation commit `8e08700` reached the
60-second native mining bound at Fos. It produced no fitted predictive model
and no predictive score. Its original output, including the failure and byte
manifest, is retained under `evidence/initial-timeout/`.

Two bounded diagnostic calls on the identical training matrix locate the cost
inside native Apriori candidate generation. Imports finish in about two seconds;
the next levels join every pair of candidate itemsets and scan duplicates.
This is an observed computation failure for that configuration, not a measured
loss of predictive accuracy and not an exhaustive benchmark of the library.

## Narrower mining question

Keep all thirteen original drivers, but mine **every unordered pair of drivers
with the training target**: 78 native three-column calls per site, inside one
worker with the original 60-second bound. Each call uses the unchanged production
adapter and actual CPU so4gp 1.1.2 GRAANK. Retain only target-containing patterns
with exactly two antecedents. Save every input, native output and normalized
result; select from their combined candidates with the original inverse
normalization, independent support recount and top-eight rule.

The resulting question is whether native-selected **pair interactions** help
the fixed small predictor. It excludes higher-arity motifs. It is not equivalent
to completing the initial unrestricted fourteen-column mining run. The generic
interaction comparator receives the same selected count and arity/sign profile;
its driver assignment remains independent of target values conditional on that
profile.

No other scientific settings change: archives, sites, target, split, thirteen
drivers, baselines, scaling, ridge alpha, support threshold, feature cap, seeds,
metrics and clipping stay as originally specified. No predictive score has been
inspected when choosing this amendment. No seed, feature or model search is
introduced. Empty selections and negative outcomes remain valid results.

`execution-amendment.json` gives the machine-readable overrides;
`execution-freeze.json` authenticates both amendment files before fitting or
scoring. The original receipt records its own earlier point in time; it must not
be read as saying that no mining attempt occurred later.
