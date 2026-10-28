# Contributing

This repository is the maintained public source for DEPOSIUM GeoAI. Fork it,
create a branch for a focused change, and open a pull request against `main`.
Explain the problem, resulting behavior, validation and any remaining limits.
Report vulnerabilities through [SECURITY.md](SECURITY.md), rather than a public PR.

## Validate changes

Use Python 3.14 and the locked CPU pattern runtime:

```sh
uv sync --locked --extra dev --extra patterns
uv run --no-sync ruff check .
uv run --no-sync ruff format --check .
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONHASHSEED=0 \
  uv run --no-sync pytest tests/
```

Keep the dependency lock consistent with package metadata. Add meaningful
regression coverage for behavior changes. Normal tests use synthetic or frozen
inputs; live-provider tests and `tests/live_acceptance.py` require separate
explicit configuration and authorized test storage.

## Review and merge policy

All changes to `main` go through a pull request. Do not push directly, force-push
or delete `main`, including from an administrator account.

[@nicolas-geysse](https://github.com/nicolas-geysse) owns review and merge decisions.
Contributor PRs require his approval and merge. New commits invalidate stale
approvals; resolve review threads, update the branch against current `main`,
and wait for the required `Lint & test (Python 3.14)` check to pass before merge.

GitHub does not allow authors to approve their own PRs. The maintainer may merge
his own PR using the PR-only review exception, after reviewing its diff and
validation. This exception does not permit direct pushes or bypass of required
CI. Repository rulesets enforce these controls separately from this document.

## Data, credentials and reuse

Keep service credentials, private project documentation, tenant data and local
research outputs out of commits, fixtures, logs and PR descriptions. Use minimal
synthetic examples and preserve the provenance and scientific limits of retained
examples. See [NOTICE](NOTICE) for data and third-party attribution.

Contributions are provided under the repository's [MIT licence](LICENSE).
Preserve existing copyright and licence notices and obtain permission for any
third-party code or data you include; their separate terms remain applicable.
