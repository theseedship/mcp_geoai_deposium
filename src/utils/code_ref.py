"""The producing-code fingerprint recorded beside an evidence artifact.

An artifact that carries its input digests but not the code that turned them into numbers
is only half reproducible: the same inputs run through a later revision give different
cells with nothing in the file to say so. Every artifact that claims immutability records
this next to the input digests.

``dirty`` is the field that matters and the one that is easiest to omit. A commit id taken
from a tree with uncommitted changes does NOT identify the code that ran, so a caller that
wants "reproducible from this commit" has to be able to see the difference between a clean
checkout and a modified one. ``None`` means the question could not be answered at all (no
git, no repository), which is a third state and must not be reported as clean.

Three private copies of this logic already exist — ``scripts/climate_baseline_panel.py``,
``scripts/unified_pilot_panel.py`` and ``src/cli/forecast_ensemble_archive.py`` (env-only).
This module is their canonical replacement for NEW callers; migrating the existing three is
a separate mechanical change and is deliberately not bundled with a feature.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any

__all__ = ["code_ref"]

_REPO = "deposium_geoai"
_ROOT = Path(__file__).resolve().parents[2]  # this checkout of deposium_geoai


def _git(*args: str, cwd: Path | None, empty_ok: bool = False) -> str | None:
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=cwd,
            text=True,
            capture_output=True,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    value = completed.stdout.strip()
    if value or empty_ok:
        return value
    return None


def _env_first(*names: str) -> str | None:
    return next((os.environ[name] for name in names if os.environ.get(name)), None)


def code_ref(*, cwd: str | Path | None = None) -> dict[str, Any]:
    """``{repo, commit, branch, dirty}`` for the checkout that is producing an artifact.

    Falls back to the deployment environment variables when git is unavailable — a container
    image has no ``.git`` but the platform injects the commit it was built from. ``dirty`` is
    only ever ``True``/``False`` when git answered; an environment-variable commit leaves it
    ``None`` because nothing observed the tree.
    """
    root = Path(cwd) if cwd is not None else _ROOT
    top = _git("rev-parse", "--show-toplevel", cwd=root)
    ours = top is not None and Path(top).resolve() == _ROOT
    status = _git("status", "--porcelain", cwd=root, empty_ok=True) if ours else None
    commit = (_git("rev-parse", "HEAD", cwd=root) if ours else None) or _env_first(
        "RAILWAY_GIT_COMMIT_SHA",
        "GIT_COMMIT",
        "SOURCE_VERSION",
    )
    branch = (_git("rev-parse", "--abbrev-ref", "HEAD", cwd=root) if ours else None) or _env_first(
        "RAILWAY_GIT_BRANCH",
        "GIT_BRANCH",
        "BRANCH_NAME",
    )
    return {
        "repo": _REPO,
        "commit": commit,
        "branch": branch,
        "dirty": None if status is None else bool(status),
    }
