"""Recompute normals from STORED ERA5 and promote them under an explicit protocol (#172).

Normals schema v2 (PR #189) added the precipitation wet-day triplet. Every artifact written
before it is v1, and the fifteen canonical ``climate/normals/{zone_id}.parquet`` objects were
among them: this operator promoted them to v2 on the mesh RustFS on 2026-09-27 (readbacks in
#227), and Railway staging holds no promoted v2 normals yet. Regenerating those must not mean re-running the Archive: the ERA5 days they were
computed from are already materialized, roughly thirty API calls per zone would be spent to
recover numbers we hold, and the Open-Meteo tier decision is explicitly unsigned (R8).

`climate_percentiles(era5_source=...)` already reads a stored parquet and REFUSES to fall back
to the Archive when that source does not cover the window. What did not exist is the operator
layer around it: a versioned destination, a promotion that is not an overwrite, and a rollback
that is not a deletion. `docs/CONSOLIDATION_ROADMAP.md` names the trap directly — *"deletion is
not a versioning or rollback protocol"* — because removing the current object to make an
idempotent skip fire looks like a rematerialization and is actually an unrecoverable delete.

THE LAYOUT
==========

    canonical   climate/normals/{zone_id}.parquet
    staging     climate/normals/staging/{run_id}/{zone_id}.parquet
    rollback    climate/normals/rollback/{run_id}/{zone_id}.parquet
    manifest    climate/normals/staging/{run_id}/manifest.json

THE THREE STEPS, WHICH ARE SEPARATE INVOCATIONS ON PURPOSE
==========================================================

``materialize``
    Reads stored ERA5, computes v2, writes STAGING, validates the v2 coverage contract, and
    writes a manifest. It never reads or touches canonical. Running it is always safe, which is
    why it is the only step that does any computing. Refused while the run is promoted.

``promote --run-id R``
    Copies the CURRENT canonical aside to ``rollback/R/`` first, then copies staging over
    canonical. Refuses unless every requested zone is present in staging AND passes the v2
    contract, so a half-finished run cannot be half-promoted. Refuses if the run's manifest is
    already marked promoted. An existing rollback copy is kept, so it always describes canonical
    before the run first promoted.

``rollback --run-id R``
    Copies ``rollback/R/`` back over canonical and clears the manifest's promoted flag, so the
    run CAN be promoted again — the staged v2 is untouched and the kept copy still holds the
    pre-run state. The flag is what makes that safe: the repeat-promotion guard reads the
    manifest rather than the mere existence of a rollback copy, which would have refused every
    re-promotion and contradicted this paragraph (Greptile P1, PR #199). The whole run only: a
    partial rollback is refused. A zone that had no canonical before is refused until its file
    is removed by hand.

ATOMICITY, STATED HONESTLY
==========================

Each file is replaced through a temp-and-rename, so a destination is either the old bytes or
the new ones and never a prefix of the new ones. Across a zone SET there is no filesystem
transaction: a failure partway leaves a split set, and promotion then restores the zones that
already moved from their rollback copies, and names any zone that had no canonical before,
which it does not delete. That is an unwind, not an atomic commit, and the difference matters
if the unwind itself fails — which is why the kept copies are never removed.

WHAT THE LOCAL MODE DOES NOT DO, DELIBERATELY
=============================================

``--root`` is a local filesystem path in both modes. An ``s3://`` value there is refused rather
than silently attempted, because #172's assignment says in as many words that *no overwrite of an
existing environment is authorized* — so the protocol was implemented and proven offline against
the published ``climate-baseline-full-panel-2026-06-26`` package first. Refusing is the difference
between "not implemented yet" and "implemented, and it wrote somewhere it should not have".

THE OBJECT-STORE MODE (``--store s3``, #223)
============================================

``climate_normals`` reads ``s3://{S3_BUCKET}/climate/normals/{zone_id}.parquet`` through a fresh
DuckDB session on every call, with no pointer and no cache, so promoting means replacing the bytes
at that key in the bucket the serving GeoAI uses. A DuckDB ``COPY`` to ``s3://`` would do that as
an unconditional PUT: it cannot write create-only and verifies no digest. This mode goes through
boto3 instead, behind :class:`_ObjectStore`, which holds its own key allowlist and has no delete.

Its steps, each a separate invocation, all against the same local work root (``--root``):

``fetch``        GET ``climate/era5/{zone_id}.parquet`` into the work root; record etag, size and
                 sha256. No Archive call, ever.
``materialize``  the local computation, over what ``fetch`` wrote, plus the sha256 and size of
                 each staged file and the shared ``code_ref()``. Writes ``run.json`` locally and
                 touches no store.
``publish``      create-only upload of the staged files and ``run.json``, each read back and
                 compared by sha256. An existing object is accepted only if its bytes are
                 identical.
``plan``         HEAD and GET only. Writes ``plan.json`` in the work root, never in the store,
                 with each canonical it read under its sha256, and prints what ``promote``
                 would write and each zone's parity verdict. The rule is EXACT (#223): the
                 all-days p10, p50, p90, n and mean of v2 equal the canonical's on exactly 72
                 unique (variable, month) keys. It does not apply where there is no canonical.
``promote``      the whole run, against the plan. It refuses the run before any write when a
                 zone's parity is not exact, or was judged on a staging object that is not the
                 run record's, with no override. Then, per zone, a create-only rollback copy
                 (or an ``.absent`` marker) read back, a re-HEAD of canonical that refuses if
                 its etag moved since ``plan``, the PUT, a GET that requires the staged sha256,
                 and a create-only event in the zone's journal.
``rollback``     the whole run, only for the zones whose journal says THIS run owns canonical.
                 A zone whose canonical holds this run's staged bytes with no event saying so
                 (a promote stopped between its PUT and its event) is refused before any write.
``verify``       reads canonical through ``climate_normals``' own read path.

The run record is ``staging/{run_id}/run.json`` and not ``manifest.json``: the latter matches the
MCPs science-manifest pattern ``^climate/(?!runs/)(?:[^/]+/)*manifest[.]json$``, so a catalog
rebuild over ``climate/`` would try to parse it as a dataset manifest. The local mode keeps its
``manifest.json`` because nothing rebuilds a catalog over a local directory.

Event journals, one per zone and one per run, are append-only sequences of create-only objects::

    zone journal   climate/normals/events/zones/{zone_id}/{seq:06d}.json
    run journal    climate/normals/events/runs/{run_id}/{seq:06d}.json

The zone journal is what names the run that owns canonical. A sha256 alone cannot: two runs over
the same ERA5 stage identical bytes, and one run's rollback would then undo the other's promotion
(N4 in the #199 review).
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Imported, not moved: `src/cli/forecast_ensemble_ops.py` also uses the client factory, and its
# tests monkeypatch it by module path.
from src.cli.forecast_ensemble_archive import (  # noqa: E402
    _object_store_client,
    _s3_error_status,
)
from src.config import settings  # noqa: E402
from src.tools.climate import (  # noqa: E402
    NORMALS_SCHEMA_VERSION,
    _default_output_uri,
    _load_zones,
    _normals_coverage,
    _normals_coverage_satisfies,
    _read_normals_rows,
    _scientific_zone_ids,
    _validate_dst,
    register_climate_tools,
)
from src.utils.code_ref import code_ref  # noqa: E402
from src.utils.logging import setup_logging  # noqa: E402
from src.utils.openmeteo import NORMALIZED_VARS  # noqa: E402
from src.utils.s3_endpoint import (  # noqa: E402
    names_an_authority,
    redact_text,
    sanitize_endpoint,
    url_secrets,
)

#: The WMO baseline these artifacts were computed over, and the only window this operator
#: rematerializes by default. A different window is a different artifact, not a new version of
#: this one.
WMO_START, WMO_END = "1991-01-01", "2020-12-31"

_ERA5 = "climate/era5/{zone_id}.parquet"
_CANONICAL = "climate/normals/{zone_id}.parquet"
_STAGING = "climate/normals/staging/{run_id}/{zone_id}.parquet"
_ROLLBACK = "climate/normals/rollback/{run_id}/{zone_id}.parquet"
_MANIFEST = "climate/normals/staging/{run_id}/manifest.json"


class OperatorRefusal(RuntimeError):
    """A refusal that is the point of the step, not a failure of it."""


def _code_ref() -> dict[str, Any]:
    """The commit this ran from, and whether the tree was dirty.

    A normals artifact whose producing code cannot be named is not reproducible, and `dirty`
    matters more here than usual: the whole claim is "these numbers come from the stored days
    through the shipped computation".
    """

    def _git(*args: str) -> str | None:
        try:
            out = subprocess.run(
                ["git", *args], capture_output=True, text=True, check=True, timeout=10
            )
            return out.stdout.strip()
        except Exception:
            return None

    commit = _git("rev-parse", "HEAD")
    status = _git("status", "--porcelain")
    return {"commit": commit, "dirty": bool(status) if status is not None else None}


def _tools() -> dict:
    class _Capture:
        def __init__(self) -> None:
            self.tools: dict = {}

        def tool(self, **_kw):
            def deco(fn):
                self.tools[fn.__name__] = fn
                return fn

            return deco

    cap = _Capture()
    register_climate_tools(cap)
    return cap.tools


#: A URI scheme at the front of the value. Matched on the string AFTER argparse has built a
#: `Path`, which is why it does not look for `://`: `Path("s3://deposium")` collapses the double
#: slash to `s3:/deposium`, so a `"://" in str(root)` guard never fired and the operator treated
#: an object-store URI as a RELATIVE LOCAL DIRECTORY — creating `s3:/deposium/...` under the
#: working directory. Two or more characters before the colon, so a Windows drive letter is not
#: mistaken for a scheme.
_URI_SCHEME = re.compile(r"^[A-Za-z][A-Za-z0-9+.\-]+:")


#: `run_id` and `zone_id` are unrestricted CLI strings interpolated into filesystem paths, so a
#: value containing `..` could put the manifest, or a promotion target, outside `--root`
#: (Greptile P2, PR #199). Conservative on purpose: every real zone id and run id in this repo
#: is already within it.
_SAFE_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")


def _require_identifier(value: str, *, role: str) -> str:
    if not _SAFE_IDENTIFIER.match(value) or ".." in value:
        raise OperatorRefusal(
            f"{role} {value!r} is not a safe identifier — letters, digits, dot, dash and "
            "underscore only, and no '..'. These are interpolated into paths under --root"
        )
    return value


def _under(root: Path, relative: str) -> Path:
    """A path inside ``root``, verified after resolution rather than trusted before it."""
    candidate = (root / relative).resolve()
    if not candidate.is_relative_to(root):
        raise OperatorRefusal(f"{candidate} resolves outside --root {root}")
    return candidate


def _require_local(root: Path) -> Path:
    """The local root, or a refusal naming why an object store is not attempted.

    #172 authorizes no overwrite of an existing environment, so the protocol is implemented and
    proven offline and pointing it at a real store is a separate, authorized step. Refusing is
    the difference between "not implemented yet" and "implemented, and it wrote somewhere it
    should not have".
    """
    if _URI_SCHEME.match(str(root)):
        raise OperatorRefusal(
            "--root is not a local path, and is not repeated here: an object-store URI can carry "
            "credentials. --root is local-filesystem only, in both modes: "
            "#172 authorizes no overwrite of an existing environment, so an object-store "
            "destination given as a directory is refused rather than silently attempted. The "
            "object store is reached through --store s3 and its guards"
        )
    return root.expanduser().resolve()


def _validate_staged(path: Path, zone_id: str, requested: list[str]) -> dict:
    """The v2 contract on one staged artifact, or raise.

    Three separate things, because passing two of them is the shape a half-migrated artifact
    has: the file reads, its all-days coverage is complete over the window, and its stamped
    schema version is the one this run targeted.
    """
    coverage = _normals_coverage(path.as_posix(), zone_id=zone_id)
    if not coverage:
        raise OperatorRefusal(f"staged {path} is unreadable as a normals parquet")
    if not _normals_coverage_satisfies(coverage, requested):
        raise OperatorRefusal(
            f"staged {path} does not cover all 12 months for {requested} — "
            f"{coverage.get('variables')}"
        )
    stored_version = coverage.get("schema_version")
    if stored_version != NORMALS_SCHEMA_VERSION:
        raise OperatorRefusal(
            f"staged {path} is schema v{stored_version}, expected v{NORMALS_SCHEMA_VERSION} — "
            "the run computed something other than what it was asked to"
        )
    return coverage


def _stage_zones(
    *,
    root: Path,
    era5_root: Path,
    run_id: str,
    zone_ids: list[str],
    start_date: str,
    end_date: str,
) -> list[dict]:
    """Compute v2 for each zone from its stored ERA5 into staging, validating each as it lands.

    Shared by both modes: the object-store mode stages into its local work root exactly as the
    local mode stages into ``--root``, so there is one computation and one v2 gate.
    """
    requested = list(NORMALIZED_VARS)
    tools = _tools()
    results: list[dict] = []
    for zone_id in zone_ids:
        # Its own template, not a `.replace("normals", "era5")` on the canonical one: that
        # rewrites any zone id containing the substring, and zone ids are operator input.
        era5 = _under(era5_root, _ERA5.format(zone_id=zone_id))
        if not era5.exists():
            raise OperatorRefusal(
                f"no stored ERA5 for {zone_id!r} at {era5} — this operator never fetches from "
                "the Archive, so a missing input is a refusal and not a slower path"
            )
        staged = _under(root, _STAGING.format(run_id=run_id, zone_id=zone_id))
        staged.parent.mkdir(parents=True, exist_ok=True)

        result = asyncio.run(
            tools["climate_percentiles"](
                zone_id=zone_id,
                start_date=start_date,
                end_date=end_date,
                era5_source=era5.as_posix(),
                output_path=staged.as_posix(),
                overwrite=True,
            )
        )
        if "error" in result:
            raise OperatorRefusal(f"{zone_id}: {result['error']}")
        source = (result.get("metadata") or {}).get("source")
        if source != "era5_parquet":
            # The entire premise of this operator. `climate_percentiles` refuses to fall back
            # for an EXPLICIT source, so reaching here would mean that contract changed.
            raise OperatorRefusal(
                f"{zone_id}: computed from {source!r} rather than the stored parquet — refusing "
                "to publish normals this operator cannot claim came from stored ERA5"
            )
        coverage = _validate_staged(staged, zone_id, requested)
        results.append(
            {
                "zone_id": zone_id,
                "era5_source": era5.as_posix(),
                "staged": staged.as_posix(),
                "schema_version": coverage.get("schema_version"),
                "wet_day": coverage.get("wet_day"),
                "window": coverage.get("window"),
            }
        )
        print(f"  staged {zone_id:24} v{coverage.get('schema_version')}  {staged}")
    return results


def materialize(args: argparse.Namespace) -> int:
    root = _require_local(args.root)
    era5_root = _require_local(args.era5_root)
    run_id = _require_identifier(args.run_id, role="run id")
    args.zone_id = [_require_identifier(z, role="zone id") for z in args.zone_id]
    existing = _under(root, _MANIFEST.format(run_id=run_id))
    if existing.exists() and json.loads(existing.read_text(encoding="utf-8")).get("promoted"):
        raise OperatorRefusal(
            f"run {run_id!r} is promoted. Roll it back first, or materialize under a new run id"
        )
    created_at = datetime.now(UTC).isoformat()
    results = _stage_zones(
        root=root,
        era5_root=era5_root,
        run_id=run_id,
        zone_ids=args.zone_id,
        start_date=args.start_date,
        end_date=args.end_date,
    )

    manifest_path = _under(root, _MANIFEST.format(run_id=run_id))
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "kind": "normals_rematerialization",
        "run_id": run_id,
        "created_at": created_at,
        "window": {"start": args.start_date, "end": args.end_date},
        "target_schema_version": NORMALS_SCHEMA_VERSION,
        "input": "stored ERA5 GeoParquet; no Archive call is made by this operator",
        "code_ref": _code_ref(),
        "zones": results,
        "promoted": False,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"\nmanifest {manifest_path}")
    print(f"staged {len(results)} zone(s) at schema v{NORMALS_SCHEMA_VERSION}. Nothing promoted.")
    return 0


def _atomic_copy(src: Path, dst: Path) -> None:
    """Copy so a failure cannot leave a TRUNCATED destination.

    `shutil.copyfile` writes in place: an out-of-space or I/O error partway through leaves a
    half-written parquet where a valid one used to be (Greptile P1, PR #199). Writing a sibling
    temp file and renaming makes the replacement atomic on POSIX — the destination is either the
    old bytes or the new ones, never a prefix of the new ones.
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(f".{dst.name}.partial")
    try:
        shutil.copyfile(src, tmp)
        os.replace(tmp, dst)
    finally:
        tmp.unlink(missing_ok=True)


def _load_manifest(root: Path, run_id: str) -> tuple[Path, dict]:
    """The run's manifest, which promotion REQUIRES rather than treats as optional.

    Promotion used to proceed without one and then mark it ``promoted: true`` if it happened to
    exist, so a staged parquet could be promoted with no provenance, and a partial promotion
    could stamp a whole run as promoted (Greptile P2, PR #199). The manifest is the record of
    what the run computed; promoting outside it publishes an artifact the run does not describe.
    """
    path = _under(root, _MANIFEST.format(run_id=run_id))
    if not path.exists():
        raise OperatorRefusal(
            f"no manifest for run {run_id!r} at {path} — promotion requires the record of what "
            "was materialized; a staged parquet with no manifest has no provenance to promote"
        )
    return path, json.loads(path.read_text(encoding="utf-8"))


def promote(args: argparse.Namespace) -> int:
    root = _require_local(args.root)
    run_id = _require_identifier(args.run_id, role="run id")
    zone_ids = [_require_identifier(z, role="zone id") for z in args.zone_id]
    requested = list(NORMALIZED_VARS)

    manifest_path, manifest = _load_manifest(root, run_id)
    staged_zones = {z["zone_id"] for z in manifest.get("zones", [])}
    if set(zone_ids) != staged_zones:
        raise OperatorRefusal(
            f"run {run_id!r} materialized {sorted(staged_zones)} and this promotion asks for "
            f"{sorted(zone_ids)}. Promote the run the manifest describes: a partial promotion "
            "leaves some canonical objects v2 and some v1, with a manifest that says neither"
        )
    if manifest.get("promoted"):
        raise OperatorRefusal(
            f"run {run_id!r} is already promoted. Roll it back first, or materialize a new run"
        )

    # Validate EVERY zone before copying ANY.
    staged_paths: dict[str, Path] = {}
    for zone_id in zone_ids:
        staged = _under(root, _STAGING.format(run_id=run_id, zone_id=zone_id))
        if not staged.exists():
            raise OperatorRefusal(f"nothing staged for {zone_id!r} under run {run_id!r}")
        _validate_staged(staged, zone_id, requested)
        staged_paths[zone_id] = staged

    # The pre-run canonical, captured once. On a RE-promotion after a rollback the kept copy is
    # already there and holds exactly this state, so it is left alone rather than rewritten —
    # the rollback copy must describe the state before the run EVER promoted.
    promoted: list[str] = []
    try:
        for zone_id, staged in staged_paths.items():
            canonical = _under(root, _CANONICAL.format(zone_id=zone_id))
            kept = _under(root, _ROLLBACK.format(run_id=run_id, zone_id=zone_id))
            absent_marker = kept.with_suffix(".absent")
            if not kept.exists() and not absent_marker.exists():
                kept.parent.mkdir(parents=True, exist_ok=True)
                if canonical.exists():
                    # Copied ASIDE, never moved: a promotion that fails halfway leaves canonical
                    # in place, where a move would leave a zone with no canonical object at all.
                    _atomic_copy(canonical, kept)
                else:
                    # "There was nothing here" is a state, and it has to be representable or a
                    # rollback cannot know it (Greptile P2, PR #199).
                    absent_marker.write_text(
                        "canonical did not exist when this run promoted\n", encoding="utf-8"
                    )
            _atomic_copy(staged, canonical)
            promoted.append(zone_id)
            print(f"  promoted {zone_id:24} -> {canonical}")
    except Exception:
        # Best-effort unwind. Per-file replacement is atomic, so each zone is either fully old or
        # fully new; what a mid-run failure leaves is a SPLIT SET, and restoring the ones that
        # moved is the closest a filesystem gets to the all-or-nothing this step claims.
        restored, left_in_place = [], []
        for zone_id in promoted:
            kept = _under(root, _ROLLBACK.format(run_id=run_id, zone_id=zone_id))
            canonical = _under(root, _CANONICAL.format(zone_id=zone_id))
            if kept.exists():
                _atomic_copy(kept, canonical)
                restored.append(zone_id)
            else:
                # This zone had no canonical before the run promoted, so there is nothing to
                # restore it to — its canonical still holds the staged v2, and this operator does
                # not delete it.
                left_in_place.append(canonical.as_posix())
        print(
            f"promotion failed after {len(promoted)} zone(s); restored from rollback copies: "
            f"{restored or 'none'}.",
            file=sys.stderr,
        )
        if left_in_place:
            print(
                "These had no canonical before this run and now hold its staged v2. This "
                "operator does not delete; remove them yourself to return to the previous "
                "state: " + ", ".join(left_in_place),
                file=sys.stderr,
            )
        else:
            print("Canonical is as it was before this invocation.", file=sys.stderr)
        raise

    manifest["promoted"] = True
    manifest["promoted_at"] = datetime.now(UTC).isoformat()
    manifest["promoted_zones"] = promoted
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"\npromoted {len(promoted)} zone(s). Previous canonical kept under rollback/{run_id}/.")
    return 0


def rollback(args: argparse.Namespace) -> int:
    root = _require_local(args.root)
    run_id = _require_identifier(args.run_id, role="run id")
    zone_ids = [_require_identifier(z, role="zone id") for z in args.zone_id]

    manifest_path, manifest = _load_manifest(root, run_id)
    if not manifest.get("promoted"):
        raise OperatorRefusal(
            f"run {run_id!r} is not marked promoted — there is nothing this run replaced"
        )
    promoted_zones = set(manifest.get("promoted_zones", []))
    if set(zone_ids) != promoted_zones:
        raise OperatorRefusal(
            f"run {run_id!r} promoted {sorted(promoted_zones)} and this rollback asks for "
            f"{sorted(zone_ids)}. Roll back the whole run: a partial rollback leaves some "
            "canonical objects v2 and some v1, with a manifest that says neither"
        )

    # Resolve every source before restoring any, for the same reason promotion validates first.
    sources: dict[str, Path] = {}
    for zone_id in zone_ids:
        kept = _under(root, _ROLLBACK.format(run_id=run_id, zone_id=zone_id))
        if kept.with_suffix(".absent").exists():
            if not _under(root, _CANONICAL.format(zone_id=zone_id)).exists():
                continue  # already back to "no canonical": nothing to restore
            raise OperatorRefusal(
                f"{zone_id!r} had NO canonical object before run {run_id!r} promoted, so "
                "restoring that state means removing a file. This operator does not delete — "
                f"remove {_CANONICAL.format(zone_id=zone_id)} yourself if that is what you want"
            )
        if not kept.exists():
            raise OperatorRefusal(
                f"no rollback copy for {zone_id!r} under run {run_id!r} at {kept} — that run "
                "did not promote this zone, and inventing a restore would publish an artifact "
                "nobody kept"
            )
        sources[zone_id] = kept

    for zone_id, kept in sources.items():
        canonical = _under(root, _CANONICAL.format(zone_id=zone_id))
        _atomic_copy(kept, canonical)
        print(f"  restored {zone_id:24} <- {kept}")

    # Cleared, so the run can be promoted again: the staged v2 is untouched and the kept copy
    # still holds the pre-run state, which is what makes a rollback reversible.
    manifest["promoted"] = False
    manifest["promoted_zones"] = []
    manifest["rolled_back_at"] = datetime.now(UTC).isoformat()
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(
        f"\nrestored {len(sources)} zone(s). The v2 artifacts remain under staging/{run_id}/ "
        "and this run can be promoted again."
    )
    return 0


# --------------------------------------------------------------------------- #
# Object-store mode (#223 A)
# --------------------------------------------------------------------------- #

#: The run record. NOT ``manifest.json``: that name matches the MCPs science-manifest pattern
#: ``^climate/(?!runs/)(?:[^/]+/)*manifest[.]json$`` (deposium_MCPs
#: ``src/services/geoai-object-store.ts``), and a catalog rebuild over ``climate/`` would try to
#: parse this as a dataset manifest.
_RUN_RECORD = "climate/normals/staging/{run_id}/run.json"
_ABSENT = "climate/normals/rollback/{run_id}/{zone_id}.absent"
_ZONE_EVENT = "climate/normals/events/zones/{zone_id}/{seq:06d}.json"
_RUN_EVENT = "climate/normals/events/runs/{run_id}/{seq:06d}.json"

# Local to the work root only; none of these is ever written to the store.
_FETCH_RECORD = "climate/normals/staging/{run_id}/fetch.json"
_PLAN = "climate/normals/staging/{run_id}/plan.json"
_READBACK = "readback/{run_id}/{role}/{zone_id}.parquet"
#: Canonical as ``plan`` read it, named by its sha256. The runbook copies this directory off the
#: store as the only backup of the pre-run canonical: a later ``plan`` that reads other bytes (the
#: ones this run promoted) adds a file next to it and never replaces it. ``promote`` reads
#: canonical again to recompute the parity verdict and accepts only the sha256 the plan recorded,
#: so it writes the same bytes under the same name.
_READBACK_CANONICAL = "readback/{run_id}/canonical/{zone_id}.{sha256}.parquet"

_ABSENT_MARKER = b"canonical did not exist when this run promoted\n"

#: The columns a v2 artifact carries and a v1 one does not. `verify` requires all of them.
_WET_DAY_COLUMNS = ("p10_wet", "p50_wet", "p90_wet", "n_wet", "wet_day_threshold_mm")

#: A journal longer than this is not a journal this operator wrote.
_JOURNAL_CEILING = 10_000

_OBJECT_STORE_STEPS = ("fetch", "publish", "plan", "verify")
_OBJECT_STORE_FLAGS = ("confirm_bucket", "confirm_endpoint", "authorization", "if_match")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _record_bytes(payload: dict[str, Any]) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _atomic_write_bytes(dst: Path, data: bytes) -> None:
    """:func:`_atomic_copy` for bytes already in memory: old-or-new, never a prefix of new."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(f".{dst.name}.partial")
    try:
        tmp.write_bytes(data)
        os.replace(tmp, dst)
    finally:
        tmp.unlink(missing_ok=True)


def _missing(exc: Exception) -> bool:
    status, code = _s3_error_status(exc)
    return status == 404 or code in {"404", "NoSuchKey", "NotFound"}


def _conflict(exc: Exception) -> bool:
    status, code = _s3_error_status(exc)
    return status in {409, 412} or code in {"ConditionalRequestConflict", "PreconditionFailed"}


class _ObjectStore:
    """HEAD, GET and PUT against the bucket, each checked against an allowlist held HERE.

    Boto3 writes do not go through ``_validate_dst``, so the store has to confine itself. What a
    run may read: the ERA5 and canonical normals of ITS zones, and its own staging, rollback and
    event keys. What it may write: its own staging, rollback and event keys, and canonical only
    inside :meth:`canonical_writes`, which ``promote`` and ``rollback`` open and nothing else
    does. There is no delete method, so there is no delete.
    """

    def __init__(self, client: Any, *, bucket: str, run_id: str, zone_ids: list[str]) -> None:
        self._client = client
        self.bucket = bucket
        self._era5 = {_ERA5.format(zone_id=z) for z in zone_ids}
        self._canonical = {_CANONICAL.format(zone_id=z) for z in zone_ids}
        self._own = (
            f"climate/normals/staging/{run_id}/",
            f"climate/normals/rollback/{run_id}/",
            f"climate/normals/events/runs/{run_id}/",
            *(f"climate/normals/events/zones/{z}/" for z in zone_ids),
        )
        self._canonical_open = False

    @contextmanager
    def canonical_writes(self) -> Iterator[None]:
        self._canonical_open = True
        try:
            yield
        finally:
            self._canonical_open = False

    def _allow(self, key: str, *, write: bool) -> None:
        if ".." in key or key.startswith("/"):
            allowed = False
        elif write:
            allowed = key.startswith(self._own) or (key in self._canonical and self._canonical_open)
        else:
            allowed = key in self._era5 or key in self._canonical or key.startswith(self._own)
        if not allowed:
            verb = "PUT" if write else "GET/HEAD"
            raise OperatorRefusal(
                f"{verb} {key!r} is outside this run's key allowlist — the object-store mode "
                "reads only its zones' ERA5 and canonical normals plus its own staging, rollback "
                "and event keys, and writes canonical only during promote and rollback"
            )

    def head(self, key: str) -> dict | None:
        """``{"etag", "size"}``, or ``None`` when there is no object."""
        self._allow(key, write=False)
        try:
            response = self._client.head_object(Bucket=self.bucket, Key=key)
        except Exception as exc:
            if _missing(exc):
                return None
            raise
        return {"etag": response.get("ETag"), "size": response.get("ContentLength")}

    def get(self, key: str) -> tuple[bytes, dict] | None:
        """``(bytes, {"etag", "size", "sha256"})``, or ``None`` when there is no object."""
        self._allow(key, write=False)
        try:
            response = self._client.get_object(Bucket=self.bucket, Key=key)
        except Exception as exc:
            if _missing(exc):
                return None
            raise
        body = response["Body"].read()
        return body, {"etag": response.get("ETag"), "size": len(body), "sha256": _sha256(body)}

    def put_if_absent(self, key: str, body: bytes, *, content_type: str) -> bool:
        """Create-only (``If-None-Match: *``); ``False`` when an object was already there."""
        self._allow(key, write=True)
        try:
            self._client.put_object(
                Bucket=self.bucket,
                Key=key,
                Body=body,
                ContentLength=len(body),
                ContentType=content_type,
                IfNoneMatch="*",
            )
        except Exception as exc:
            if _conflict(exc):
                return False
            raise
        return True

    def put_canonical(self, key: str, body: bytes, *, if_match: str | None) -> None:
        """Replace canonical. ``if_match`` is sent only when the operator opted in (see
        ``--if-match``): whether RustFS honours it is what the harness probe decides."""
        if key not in self._canonical:
            raise OperatorRefusal(f"{key!r} is not a canonical key of this run's zones")
        self._allow(key, write=True)
        request: dict[str, Any] = {
            "Bucket": self.bucket,
            "Key": key,
            "Body": body,
            "ContentLength": len(body),
            "ContentType": "application/vnd.apache.parquet",
        }
        if if_match:
            request["IfMatch"] = if_match
        self._client.put_object(**request)


@dataclass
class _Run:
    """What every object-store step works from, once the guards have passed."""

    args: argparse.Namespace
    root: Path
    run_id: str
    zone_ids: list[str]
    authority: dict[str, Any]
    store: _ObjectStore | None = None


#: A commit named by the environment alone. Git answers with the full sha; an injected
#: ``GIT_COMMIT`` has nothing to check it against, so it must at least be one.
_FULL_SHA = re.compile(r"^[0-9a-f]{40}$")

#: The repository this script runs from, for :func:`_source_digest`.
_REPO_ROOT = Path(__file__).resolve().parent.parent


def _endpoint_authority() -> str | None:
    """``host[:port]`` of ``S3_ENDPOINT`` after the repo's accept grammar, lowercased, or ``None``.

    The port is part of it. On the maintainer's host the harness is published on
    ``127.0.0.1:9200`` and the mesh on ``localhost:9100``, so a host alone would confirm either
    store. Until 2026-09-27 they also shared the bucket name ``deposium``.
    """
    raw = settings.S3_ENDPOINT
    if not raw:
        return None
    address, _differed = sanitize_endpoint(raw)
    if address is None:
        return None
    authority = re.sub(r"^(?:https?:)?//", "", address, flags=re.IGNORECASE).rstrip("/")
    return authority.lower() if names_an_authority(authority) else None


def _require_authorization(value: str | None) -> str:
    """The authorization comment's URL: https on github.com, no userinfo, no query, nothing to
    redact.

    The refusal does not repeat the value: the commonest reason to refuse one is a token in it,
    and the run record, every event and the printed plan would otherwise carry that token into
    the bucket.
    """
    if not value:
        raise OperatorRefusal(
            "--authorization <URL> is required with --store s3: a store write names the "
            "maintainer comment that authorized it, and the run record carries it"
        )
    try:
        parts = urlsplit(value)
    except ValueError:
        # A malformed IPv6 literal, or a netloc that NFKC normalization changes. The error quotes
        # the netloc, userinfo included, so it is refused below like any other value, unechoed.
        parts = None
    if (
        parts is None
        or parts.scheme != "https"
        or not parts.hostname
        or "@" in parts.netloc
        # host[:port] and nothing else. A clone URL with its "@github.com" cut off has no "@",
        # and its token then reads as the port ("github.com:ghp_…") or as the host.
        or not names_an_authority(parts.netloc)
        # The hostname grammar refuses a token in the host only when it carries a "_": a legacy
        # 40-hex classic PAT is a valid label. So the host is pinned to where the authorization
        # comment lives; urlsplit lowercases it, so "GitHub.com" still passes.
        or parts.hostname != "github.com"
        or parts.query
        or url_secrets(value)
        or len(value) > 512
        or any(c.isspace() for c in value)
    ):
        raise OperatorRefusal(
            "--authorization is not an https URL with a host, no credentials and no query, such "
            "as https://github.com/<org>/<repo>/issues/172#issuecomment-<id> (the host must be "
            "github.com). The value given is not repeated here: it may carry a token"
        )
    return value


def _source_digest(root: Path = _REPO_ROOT) -> str:
    """sha256 over every ``src/`` and ``scripts/`` Python file under ``root``, by relative path.

    Recorded because an injected ``GIT_COMMIT`` observes nothing: on the mesh the host's ``src/``
    is mounted over the image's, so the code that runs need not be the commit named. The same
    function run in a clean checkout of that commit gives the digest to compare against.
    """
    digest = hashlib.sha256()
    for path in sorted(p for part in ("src", "scripts") for p in (root / part).rglob("*.py")):
        digest.update(
            f"{path.relative_to(root).as_posix()}\0{_sha256(path.read_bytes())}\n".encode()
        )
    return digest.hexdigest()


def _require_object_store_guards(args: argparse.Namespace) -> dict[str, Any]:
    """The run's authority, or a refusal. Every check runs before any client exists.

    The bucket alone cannot be relied on to tell a rehearsal from a real write — until
    2026-09-27 the harness and the mesh both used ``deposium``, and ``GEOAI_DEV_BUCKET=deposium``
    still makes them share it — so the endpoint's host AND port have to be confirmed too. Neither refusal
    echoes the configured value: a confirmation copied from the error message confirms nothing.
    """
    bucket = settings.S3_BUCKET
    if not bucket:
        raise OperatorRefusal("S3_BUCKET is not configured; there is no bucket to confirm")
    if not args.confirm_bucket:
        raise OperatorRefusal("--confirm-bucket is required with --store s3")
    if args.confirm_bucket != bucket:
        raise OperatorRefusal(
            "--confirm-bucket does not match the configured S3_BUCKET. The value given is not "
            "repeated here: a typed value can carry a credential pasted by mistake"
        )
    endpoint = _endpoint_authority()
    if endpoint is None:
        raise OperatorRefusal(
            "S3_ENDPOINT is unset or not a plain address — boto3 would fall back to AWS, so "
            "the object-store mode refuses rather than guess where it is writing"
        )
    if not args.confirm_endpoint:
        raise OperatorRefusal("--confirm-endpoint <host:port> is required with --store s3")
    confirmed = args.confirm_endpoint.strip()
    if not names_an_authority(confirmed):
        raise OperatorRefusal(
            "--confirm-endpoint takes a bare host[:port], with no scheme, path or credentials. "
            "The value given is not one, and is not repeated here"
        )
    if confirmed.lower() != endpoint:
        raise OperatorRefusal(
            "--confirm-endpoint does not match the host and port of the configured S3_ENDPOINT. "
            "Both are compared: two stores reached through one host differ only by port"
        )
    authorization = _require_authorization(args.authorization)
    ref = code_ref()
    commit = ref.get("commit")
    if not commit:
        raise OperatorRefusal(
            "the producing commit is unknown — no git checkout and no GIT_COMMIT. Normals whose "
            "code cannot be named are not reproducible; run with GIT_COMMIT=<merge sha>"
        )
    if ref.get("dirty") is True:
        raise OperatorRefusal(
            f"the checkout at {commit} has uncommitted changes, so that commit does not "
            "name the code that would run. Commit or stash first"
        )
    tree_observed = ref.get("dirty") is not None
    if not tree_observed and not _FULL_SHA.match(commit):
        raise OperatorRefusal(
            "the producing commit comes from the environment, with no git to observe the tree, "
            "and it is not a full 40-character lowercase sha. Run with GIT_COMMIT=<full merge sha>"
        )
    return {
        "bucket": bucket,
        "endpoint": endpoint,
        "authorization": authorization,
        "code_ref": {**ref, "tree_observed": tree_observed, "source_sha256": _source_digest()},
    }


def _require_scientific_zones(zone_ids: list[str]) -> list[str]:
    """Panel and pilot zones only, each once. A client zone's normals are not this operator's."""
    duplicates = sorted({z for z in zone_ids if zone_ids.count(z) > 1})
    if duplicates:
        raise OperatorRefusal(f"zone ids given more than once: {duplicates}")
    scientific = set(_scientific_zone_ids(_load_zones()))
    outside = [z for z in zone_ids if z not in scientific]
    if outside:
        raise OperatorRefusal(
            f"{outside} are not scientific (panel or pilot) zones. The object-store mode "
            "promotes only those: a client site is not part of the canonical baseline"
        )
    return zone_ids


def _read_local_json(path: Path, *, missing: str) -> dict:
    if not path.exists():
        raise OperatorRefusal(missing)
    return json.loads(path.read_text(encoding="utf-8"))


def _require_zone_set(
    recorded: list[str], asked: list[str], *, what: str, run_id: str, hint: str
) -> None:
    if set(recorded) != set(asked):
        raise OperatorRefusal(
            f"run {run_id!r} {what} {sorted(recorded)} and this invocation asks for "
            f"{sorted(asked)}. {hint}"
        )


def _publish_create_only(store: _ObjectStore, key: str, body: bytes, *, content_type: str) -> bool:
    """Create-only upload, then read back. An existing object passes only if byte-identical."""
    created = store.put_if_absent(key, body, content_type=content_type)
    got = store.get(key)
    if got is None:
        raise OperatorRefusal(f"{key} is not readable back after its upload")
    if got[1]["sha256"] != _sha256(body):
        if created:
            raise OperatorRefusal(f"{key} reads back with a different sha256 than was uploaded")
        raise OperatorRefusal(
            f"{key} already exists with different bytes. Create-only refuses to replace it; "
            "use a new run id"
        )
    return created


def _journal(store: _ObjectStore, template: str, **fields: str) -> list[dict]:
    """Every entry of an append-only journal, read by GET on consecutive sequence numbers.

    GET and not LIST: the store's allowlist names keys, and a sequence is walkable by key.
    """
    entries: list[dict] = []
    for seq in range(1, _JOURNAL_CEILING + 1):
        got = store.get(template.format(seq=seq, **fields))
        if got is None:
            return entries
        entries.append(json.loads(got[0]))
    raise OperatorRefusal(f"journal {template.format(seq=0, **fields)} exceeds its ceiling")


def _append_event(
    store: _ObjectStore, template: str, entries: list[dict], event: dict, **fields: str
) -> dict:
    """Append create-only at the next sequence number. Losing that race is a refusal.

    An existing key holding exactly this event's bytes is this event: a create-only PUT that
    landed and lost its response is retried by botocore, and the store refuses the retry. The
    event carries its own timestamp, so another writer's entry never has the same bytes.
    """
    seq = len(entries) + 1
    event = {**event, "seq": seq}
    body = _record_bytes(event)
    key = template.format(seq=seq, **fields)
    created = store.put_if_absent(key, body, content_type="application/json")
    got = store.get(key)
    if got is None or got[1]["sha256"] != _sha256(body):
        if not created:
            raise OperatorRefusal(
                f"{key} already exists — another writer appended to this journal concurrently"
            )
        raise OperatorRefusal(f"{key} does not read back as the event just written")
    entries.append(event)
    return event


def _owners(entries: list[dict]) -> list[dict]:
    """The promotions still standing in a zone journal, oldest first.

    A stack and not the latest entry: a rollback hands canonical back to the run whose promotion
    it sat on. r2 promoting over r1 and then rolling back restores r1's bytes, and r1 owns them
    again — the latest entry (r2's rollback) would say nobody does.
    """
    stack: list[dict] = []
    for entry in entries:
        if entry.get("kind") == "promote":
            stack.append(entry)
        elif stack and stack[-1].get("run_id") == entry.get("run_id"):
            stack.pop()  # a rollback or an unwind of the promotion on top
    return stack


def _owner(entries: list[dict]) -> dict | None:
    """The promote event whose bytes canonical holds, or ``None`` when no run owns it."""
    stack = _owners(entries)
    return stack[-1] if stack else None


def _latest_by(entries: list[dict], run_id: str) -> dict | None:
    """The latest entry ``run_id`` wrote in a zone journal, whatever came after it."""
    return next((e for e in reversed(entries) if e.get("run_id") == run_id), None)


def _unrecorded_rollbacks(
    journals: dict[str, list[dict]], run_events: list[dict], run_id: str
) -> list[str]:
    """The zones whose latest entry by ``run_id`` is a rollback that no ``rolled_back`` run event
    records: a rollback that stopped before its run event, whatever the promotion before it wrote.

    Only the last run event can record them: any later ``promoted`` event follows a promotion of
    every zone, which leaves none of them on a rollback.
    """
    last = run_events[-1] if run_events else None
    recorded = (last.get("zones") or {}) if _is(last, "rolled_back", run_id) else {}
    unrecorded = []
    for zone_id, entries in journals.items():
        latest = _latest_by(entries, run_id)
        if (
            latest is not None
            and _is(latest, "rollback", run_id)
            and (recorded.get(zone_id) or {}).get("seq") != latest["seq"]
        ):
            unrecorded.append(zone_id)
    return unrecorded


def _load_run_record(run: _Run) -> tuple[dict, str]:
    """The PUBLISHED run record and its sha256. Promotion requires it, like the local manifest."""
    got = run.store.get(_RUN_RECORD.format(run_id=run.run_id))
    if got is None:
        raise OperatorRefusal(
            f"no manifest for run {run.run_id!r} in the store — promotion requires the published "
            "run record (run.json); run materialize then publish"
        )
    record = json.loads(got[0])
    if record.get("run_id") != run.run_id or record.get("kind") != "normals_rematerialization":
        raise OperatorRefusal(f"the run record for {run.run_id!r} does not describe that run")
    return record, got[1]["sha256"]


def _readback_path(run: _Run, role: str, zone_id: str) -> Path:
    return _under(run.root, _READBACK.format(run_id=run.run_id, role=role, zone_id=zone_id))


#: The parity rule (#223, decision 5 of 2026-09-27): EXACT. Per zone, on the all-days fields of
#: every (variable, month) key, v2 staged must equal the canonical it replaces. The wet-day
#: columns are what v2 adds, so a v1 canonical has nothing to compare them with.
_PARITY_FIELDS = ("p10", "p50", "p90", "n", "mean")

#: Six variables by twelve months, on each side.
_PARITY_KEYS = len(NORMALIZED_VARS) * 12

#: The one place the refusals and the plan output send an operator to.
_PARITY_STOP = "the parity stop procedure in docs/runbooks/normals-v2-promotion.md"

#: How many keys a reason names before it only counts the rest.
_PARITY_LISTED = 5


def _parity_label(key: tuple) -> str:
    return f"{key[0]}/{key[1]}"


def _parity_sorted(keys) -> list[tuple]:
    # str() on both parts: a NULL variable or month is a key too, and must still sort.
    return sorted(keys, key=lambda k: (str(k[0]), str(k[1]).zfill(2)))


def _parity_listed(keys: list[tuple]) -> str:
    labels = [_parity_label(k) for k in keys[:_PARITY_LISTED]]
    rest = len(keys) - len(labels)
    return ", ".join(labels) + (f" and {rest} more" if rest else "")


def _parity_not_applicable() -> dict:
    return {
        "rule": "exact",
        "status": "not_applicable",
        "exact": False,
        "reasons": ["no canonical before this run, so parity does not apply"],
    }


def _parity_unreadable(error: str) -> dict:
    """A canonical the comparison cannot read is not a canonical v2 is proven equal to."""
    return {
        "rule": "exact",
        "status": "differs",
        "exact": False,
        "error": error,
        "reasons": [error],
    }


def _parity(rows: list[tuple]) -> dict:
    """The EXACT verdict over the rows of :data:`_PARITY_SQL`.

    Each row pairs one staged row with one canonical row of the same (variable, month) key, or
    carries a single side where the other has no row for that key. A key is exact when it has
    exactly one row on each side and every field in :data:`_PARITY_FIELDS` is NOT DISTINCT
    between them, so NULL equals NULL. The verdict is exact when every key is, and each side has
    exactly :data:`_PARITY_KEYS` unique keys.

    A duplicated key is not exact even when its rows are identical: ``climate_normals`` would
    serve two rows for one month. There is no tolerance: a difference of any size is a STOP.
    """
    keys: dict[tuple, dict[str, Any]] = {}
    for variable, month, staged_row, canonical_row, *rest in rows:
        key = keys.setdefault(
            (variable, month), {"staged": set(), "canonical": set(), "differs": {}}
        )
        if staged_row is not None:
            key["staged"].add(staged_row)
        if canonical_row is not None:
            key["canonical"].add(canonical_row)
        if staged_row is None or canonical_row is None:
            continue
        same, values = rest[: len(_PARITY_FIELDS)], rest[len(_PARITY_FIELDS) :]
        for index, field in enumerate(_PARITY_FIELDS):
            if not same[index] and field not in key["differs"]:
                key["differs"][field] = {
                    "staged": values[2 * index],
                    "canonical": values[2 * index + 1],
                }

    staged_keys = [k for k, v in keys.items() if v["staged"]]
    canonical_keys = [k for k, v in keys.items() if v["canonical"]]
    duplicated = {
        side: _parity_sorted(k for k, v in keys.items() if len(v[side]) > 1)
        for side in ("staged", "canonical")
    }
    only_in_staged = _parity_sorted(k for k, v in keys.items() if not v["canonical"])
    only_in_canonical = _parity_sorted(k for k, v in keys.items() if not v["staged"])
    differing = _parity_sorted(k for k, v in keys.items() if v["differs"])

    reasons: list[str] = []
    for side, present in (("staged", staged_keys), ("canonical", canonical_keys)):
        if len(present) != _PARITY_KEYS:
            reasons.append(
                f"{side} has {len(present)} unique (variable, month) keys, not {_PARITY_KEYS}"
            )
    for side in ("staged", "canonical"):
        if duplicated[side]:
            reasons.append(
                f"{len(duplicated[side])} key(s) duplicated in {side}: "
                f"{_parity_listed(duplicated[side])}"
            )
    if only_in_staged:
        reasons.append(
            f"{len(only_in_staged)} key(s) only in staged: {_parity_listed(only_in_staged)}"
        )
    if only_in_canonical:
        reasons.append(
            f"{len(only_in_canonical)} key(s) only in canonical: "
            f"{_parity_listed(only_in_canonical)}"
        )
    if differing:
        moved = sorted({f for k in differing for f in keys[k]["differs"]})
        reasons.append(
            f"{len(differing)} key(s) differ on {', '.join(moved)}: {_parity_listed(differing)}"
        )

    def _exact_key(k: tuple) -> bool:
        v = keys[k]
        return len(v["staged"]) == 1 and len(v["canonical"]) == 1 and not v["differs"]

    by_variable: dict[str, dict[str, Any]] = {}
    for variable in sorted({str(k[0]) for k in keys}):
        mine = [k for k in keys if str(k[0]) == variable]
        fields: dict[str, int] = {}
        for k in mine:
            for field in keys[k]["differs"]:
                fields[field] = fields.get(field, 0) + 1
        entry = {
            "staged_keys": sum(1 for k in mine if keys[k]["staged"]),
            "canonical_keys": sum(1 for k in mine if keys[k]["canonical"]),
            "identical_keys": sum(1 for k in mine if _exact_key(k)),
            "differing_keys": sum(1 for k in mine if keys[k]["differs"]),
            "differing_fields": dict(sorted(fields.items())),
        }
        entry["exact"] = entry["identical_keys"] == 12 and len(mine) == 12
        by_variable[variable] = entry

    exact = not reasons
    return {
        "rule": "exact",
        "status": "exact" if exact else "differs",
        "exact": exact,
        "fields": list(_PARITY_FIELDS),
        "expected_unique_keys": _PARITY_KEYS,
        "unique_keys": {"staged": len(staged_keys), "canonical": len(canonical_keys)},
        "identical_keys": sum(1 for k in keys if _exact_key(k)),
        "duplicated_keys": {
            side: [_parity_label(k) for k in duplicated[side]] for side in duplicated
        },
        "only_in_staged": [_parity_label(k) for k in only_in_staged],
        "only_in_canonical": [_parity_label(k) for k in only_in_canonical],
        "differing_keys": {_parity_label(k): keys[k]["differs"] for k in differing},
        "by_variable": by_variable,
        "reasons": reasons,
    }


#: Both sides' all-days rows of the zone, numbered so a duplicated key shows as two rows, joined on
#: the key with NOT DISTINCT so that a NULL variable or month is a key like any other. The zone
#: filter is the one ``climate_normals`` reads through (``_read_normals_rows``).
_PARITY_SQL = """
WITH s AS (
    SELECT row_number() OVER () AS row_id, variable, month, {fields}
    FROM read_parquet('{staged}') WHERE zone_id = ?
), c AS (
    SELECT row_number() OVER () AS row_id, variable, month, {fields}
    FROM read_parquet('{canonical}') WHERE zone_id = ?
)
SELECT COALESCE(s.variable, c.variable), COALESCE(s.month, c.month), s.row_id, c.row_id,
    {same}, {values}
FROM s FULL OUTER JOIN c
    ON s.variable IS NOT DISTINCT FROM c.variable AND s.month IS NOT DISTINCT FROM c.month
"""


def _percentile_comparison(staged: Path, canonical: Path, *, zone_id: str) -> dict:
    """All-days p10/p50/p90 of staged against canonical, paired by (variable, month), and the
    parity verdict under the EXACT rule (#223, decision 5), which ``promote`` gates on.

    ``pairs``, ``identical``, ``max_abs_diff`` and ``unpaired`` are the report ``plan`` wrote
    before the rule was decided, kept as they were for anything that reads an older plan. The
    verdict is under ``parity``: see :func:`_parity`.
    """
    import duckdb

    _validate_dst(staged.as_posix())
    _validate_dst(canonical.as_posix())
    con = duckdb.connect(":memory:")
    try:
        try:
            rows = con.execute(
                "SELECT COALESCE(s.variable, c.variable), COALESCE(s.month, c.month), "
                "s.variable IS NOT NULL, c.variable IS NOT NULL, "
                "s.p10, s.p50, s.p90, c.p10, c.p50, c.p90 "
                f"FROM read_parquet('{staged.as_posix()}') s "  # noqa: S608 - _validate_dst
                f"FULL OUTER JOIN read_parquet('{canonical.as_posix()}') c "  # noqa: S608
                "ON s.variable = c.variable AND s.month = c.month ORDER BY 1, 2"
            ).fetchall()
        except Exception:
            error = "canonical is unreadable as a normals parquet"
            return {"error": error, "parity": _parity_unreadable(error)}
        try:
            parity = _parity(
                con.execute(
                    _PARITY_SQL.format(
                        staged=staged.as_posix(),
                        canonical=canonical.as_posix(),
                        fields=", ".join(_PARITY_FIELDS),
                        same=", ".join(f"s.{f} IS NOT DISTINCT FROM c.{f}" for f in _PARITY_FIELDS),
                        values=", ".join(f"s.{f}, c.{f}" for f in _PARITY_FIELDS),
                    ),
                    [zone_id, zone_id],
                ).fetchall()
            )
        except Exception:
            parity = _parity_unreadable(
                "the all-days fields (zone_id, variable, month, "
                f"{', '.join(_PARITY_FIELDS)}) are unreadable on one side"
            )
    finally:
        con.close()
    paired = [r for r in rows if r[2] and r[3]]
    max_abs_diff: dict[str, float | None] = {}
    for offset, name in enumerate(("p10", "p50", "p90")):
        deltas = [
            abs(r[4 + offset] - r[7 + offset])
            for r in paired
            if r[4 + offset] is not None and r[7 + offset] is not None
        ]
        max_abs_diff[name] = max(deltas) if deltas else None
    return {
        "pairs": len(paired),
        "identical": sum(1 for r in paired if r[4:7] == r[7:10]),
        "max_abs_diff": max_abs_diff,
        "unpaired": [f"{r[0]}/{r[1]}" for r in rows if not (r[2] and r[3])],
        "parity": parity,
    }


def _zones_by_id(record: dict) -> dict[str, dict]:
    return {z["zone_id"]: z for z in record.get("zones", [])}


def _store_record(run: _Run) -> dict[str, Any]:
    """What every store-side record carries: where it ran, on whose authority, from which code."""
    return {
        "bucket": run.authority["bucket"],
        "endpoint": run.authority["endpoint"],
        "authorization": run.authority["authorization"],
        "code_ref": run.authority["code_ref"],
    }


#: What binds a record to the store and the authorization a step runs under.
_AUTHORITY_FIELDS = ("bucket", "endpoint", "authorization")

#: What binds rollback and verify: the store only. An emergency rollback may run under its own
#: authorization comment, which its events record; it may not run against another store.
_STORE_FIELDS = ("bucket", "endpoint")

#: The only window the object-store mode stages, plans or promotes. Canonical is the WMO baseline
#: ``climate_normals`` serves; a different window is a different artifact and never replaces it
#: (Codex P1, PR #224). Tests narrow it to the one year their ERA5 covers.
_CANONICAL_WINDOW = {"start": WMO_START, "end": WMO_END}


def _require_canonical_window(window: object, *, what: str) -> None:
    if window != _CANONICAL_WINDOW:
        raise OperatorRefusal(
            f"{what} covers {window}, and canonical holds only the {_CANONICAL_WINDOW['start']} "
            f"to {_CANONICAL_WINDOW['end']} baseline. A different window is a different "
            "artifact, which the object-store mode never writes over canonical"
        )


def _require_same_authority(
    run: _Run, document: dict, *, what: str, fields: tuple[str, ...] = _AUTHORITY_FIELDS
) -> None:
    """Refuse a record made under another bucket, endpoint or authorization than this step's.

    Every step of a run carries the same three confirmations, and the records say which ones they
    were made under. Without this a work root or run id reused from the rehearsal would publish a
    run record naming the harness into the mesh bucket, and a plan reviewed against one store
    would drive a promote on another. The values are not repeated: the endpoint is configuration.
    """
    differ = [f for f in fields if document.get(f) != run.authority[f]]
    if differ:
        raise OperatorRefusal(
            f"{what} for run {run.run_id!r} was made under another {', '.join(differ)} than this "
            "invocation. A run is bound to the store and the authorization it started under: run "
            "this step with the same confirmations, or start again from fetch under a new run id"
        )


def _s3_fetch(run: _Run) -> int:
    records: list[dict] = []
    for zone_id in run.zone_ids:
        key = _ERA5.format(zone_id=zone_id)
        got = run.store.get(key)
        if got is None:
            raise OperatorRefusal(
                f"no stored ERA5 for {zone_id!r} at {key} — this operator never fetches from "
                "the Archive, so a missing input is a refusal and not a slower path"
            )
        body, meta = got
        _atomic_write_bytes(_under(run.root, key), body)
        records.append({"zone_id": zone_id, "key": key, **meta})
        print(f"  fetched {zone_id:24} {meta['size']:>10} B  sha256 {meta['sha256']}")

    record = {
        "kind": "normals_era5_fetch",
        "run_id": run.run_id,
        "fetched_at": datetime.now(UTC).isoformat(),
        "input": "stored ERA5 GeoParquet; no Archive call is made by this operator",
        **_store_record(run),
        "zones": records,
    }
    path = _under(run.root, _FETCH_RECORD.format(run_id=run.run_id))
    _atomic_write_bytes(path, _record_bytes(record))
    print(f"\nfetch record {path}. Nothing computed, nothing written to the store.")
    return 0


def _s3_materialize(run: _Run) -> int:
    _require_canonical_window(
        {"start": run.args.start_date, "end": run.args.end_date}, what="--start-date/--end-date"
    )
    fetch_path = _under(run.root, _FETCH_RECORD.format(run_id=run.run_id))
    fetch_record = _read_local_json(
        fetch_path,
        missing=f"no fetch record for run {run.run_id!r} at {fetch_path} — run fetch first: "
        "the object-store mode computes only from ERA5 it fetched and digested",
    )
    _require_same_authority(run, fetch_record, what="the fetch record")
    fetched = _zones_by_id(fetch_record)
    _require_zone_set(
        list(fetched), run.zone_ids, what="fetched", run_id=run.run_id, hint=_NAME_THE_RUN
    )
    for zone_id in run.zone_ids:
        era5 = _under(run.root, _ERA5.format(zone_id=zone_id))
        if not era5.exists() or _sha256(era5.read_bytes()) != fetched[zone_id]["sha256"]:
            raise OperatorRefusal(
                f"the ERA5 for {zone_id!r} in the work root is not the object fetch digested — "
                "re-run fetch rather than compute from an input the record does not describe"
            )

    created_at = datetime.now(UTC).isoformat()
    results = _stage_zones(
        root=run.root,
        era5_root=run.root,
        run_id=run.run_id,
        zone_ids=run.zone_ids,
        start_date=run.args.start_date,
        end_date=run.args.end_date,
    )
    zones: list[dict] = []
    for result in results:
        zone_id = result["zone_id"]
        staged = Path(result["staged"]).read_bytes()
        source = fetched[zone_id]
        zones.append(
            {
                "zone_id": zone_id,
                "era5": {k: source[k] for k in ("key", "etag", "size", "sha256")},
                "staged_key": _STAGING.format(run_id=run.run_id, zone_id=zone_id),
                "staged_sha256": _sha256(staged),
                "staged_size": len(staged),
                "schema_version": result["schema_version"],
                "wet_day": result["wet_day"],
                "window": result["window"],
            }
        )

    record = {
        "kind": "normals_rematerialization",
        "record": "run",
        "run_id": run.run_id,
        "created_at": created_at,
        "window": {"start": run.args.start_date, "end": run.args.end_date},
        "target_schema_version": NORMALS_SCHEMA_VERSION,
        "input": "stored ERA5 GeoParquet fetched from the store; no Archive call is made",
        **_store_record(run),
        "zones": zones,
    }
    path = _under(run.root, _RUN_RECORD.format(run_id=run.run_id))
    _atomic_write_bytes(path, _record_bytes(record))
    print(f"\nrun record {path}")
    print(f"staged {len(zones)} zone(s) at schema v{NORMALS_SCHEMA_VERSION}. Nothing published.")
    return 0


def _s3_publish(run: _Run) -> int:
    path = _under(run.root, _RUN_RECORD.format(run_id=run.run_id))
    if not path.exists():
        raise OperatorRefusal(
            f"no run record for run {run.run_id!r} at {path} — run materialize first"
        )
    body = path.read_bytes()
    local_record = json.loads(body)
    _require_same_authority(run, local_record, what="the local run record")
    zones = _zones_by_id(local_record)
    _require_zone_set(
        list(zones), run.zone_ids, what="materialized", run_id=run.run_id, hint=_NAME_THE_RUN
    )

    # Every staged file checked against the record before ANY upload.
    staged: dict[str, bytes] = {}
    for zone_id in run.zone_ids:
        local = _under(run.root, _STAGING.format(run_id=run.run_id, zone_id=zone_id))
        data = local.read_bytes() if local.exists() else None
        if data is None or _sha256(data) != zones[zone_id]["staged_sha256"]:
            raise OperatorRefusal(
                f"the staged file for {zone_id!r} is missing or differs from the sha256 the run "
                "record holds — re-run materialize rather than publish bytes it does not describe"
            )
        staged[zone_id] = data

    for zone_id, data in staged.items():
        key = _STAGING.format(run_id=run.run_id, zone_id=zone_id)
        created = _publish_create_only(
            run.store, key, data, content_type="application/vnd.apache.parquet"
        )
        verb = "published" if created else "identical"
        print(f"  {verb:9} {zone_id:24} s3://{run.store.bucket}/{key}")
    record_key = _RUN_RECORD.format(run_id=run.run_id)
    _publish_create_only(run.store, record_key, body, content_type="application/json")
    print(f"\nrun record s3://{run.store.bucket}/{record_key}. Nothing promoted.")
    return 0


def _canonical_precondition(exists: bool) -> str:
    if not exists:
        return "PUT If-None-Match: * (create-only: there was no canonical at plan time)"
    return "PUT (If-Match: <planned etag> only with promote --if-match)"


def _s3_plan(run: _Run) -> int:
    record, record_sha = _load_run_record(run)
    _require_same_authority(run, record, what="the published run record")
    _require_canonical_window(record.get("window"), what="the published run record")
    zones = _zones_by_id(record)
    _require_zone_set(
        list(zones), run.zone_ids, what="materialized", run_id=run.run_id, hint=_NAME_THE_RUN
    )

    planned: dict[str, dict] = {}
    for zone_id in run.zone_ids:
        staged_key = _STAGING.format(run_id=run.run_id, zone_id=zone_id)
        got = run.store.get(staged_key)
        if got is None:
            raise OperatorRefusal(f"nothing published for {zone_id!r} under run {run.run_id!r}")
        staged_local = _readback_path(run, "staged", zone_id)
        _atomic_write_bytes(staged_local, got[0])
        staged = {
            "key": staged_key,
            "sha256": got[1]["sha256"],
            "recorded_sha256": zones[zone_id]["staged_sha256"],
            "matches_record": got[1]["sha256"] == zones[zone_id]["staged_sha256"],
            "schema_version": _normals_coverage(staged_local.as_posix(), zone_id=zone_id).get(
                "schema_version"
            ),
        }

        canonical_key = _CANONICAL.format(zone_id=zone_id)
        current = run.store.get(canonical_key)
        canonical: dict[str, Any] = {"key": canonical_key, "exists": current is not None}
        percentiles = None
        # Not applicable ONLY where there is no canonical: every canonical gets a verdict.
        parity = _parity_not_applicable()
        if current is not None:
            local = _under(
                run.root,
                _READBACK_CANONICAL.format(
                    run_id=run.run_id, zone_id=zone_id, sha256=current[1]["sha256"]
                ),
            )
            _atomic_write_bytes(local, current[0])
            canonical.update(
                {
                    **current[1],
                    "schema_version": _normals_coverage(local.as_posix(), zone_id=zone_id).get(
                        "schema_version"
                    ),
                }
            )
            percentiles = _percentile_comparison(staged_local, local, zone_id=zone_id)
            parity = percentiles.pop("parity")

        kept_key = _ROLLBACK.format(run_id=run.run_id, zone_id=zone_id)
        absent_key = _ABSENT.format(run_id=run.run_id, zone_id=zone_id)
        kept, absent = run.store.head(kept_key), run.store.head(absent_key)
        rollback = {
            "key": kept_key if current is not None else absent_key,
            "state": "kept copy exists"
            if kept
            else "absent marker exists"
            if absent
            else "not written yet",
        }

        entries = _journal(run.store, _ZONE_EVENT, zone_id=zone_id)
        owner = _owner(entries)
        writes: list[dict] = []
        if _is(owner, "promote", run.run_id):
            note = "already promoted by this run; promote skips it"
        else:
            note = None
            if not kept and not absent:
                writes.append(
                    {
                        "op": "PUT If-None-Match: *, then GET and compare sha256",
                        "key": rollback["key"],
                        "sha256": canonical["sha256"]
                        if current is not None
                        else _sha256(_ABSENT_MARKER),
                    }
                )
            writes.append(
                {
                    "op": _canonical_precondition(current is not None)
                    + ", after a HEAD that must still show the planned etag, then GET and "
                    "compare sha256",
                    "key": canonical_key,
                    "sha256": staged["sha256"],
                }
            )
            writes.append(
                {
                    "op": "PUT If-None-Match: * (promote event)",
                    "key": _ZONE_EVENT.format(zone_id=zone_id, seq=len(entries) + 1),
                }
            )
        planned[zone_id] = {
            "staged": staged,
            "canonical": canonical,
            "rollback": rollback,
            "journal": {
                "entries": len(entries),
                "owner": {k: owner[k] for k in ("seq", "run_id", "at")} if owner else None,
            },
            "percentiles_all_days": percentiles,
            "parity": parity,
            "writes": writes,
            "note": note,
        }

    run_events = _journal(run.store, _RUN_EVENT, run_id=run.run_id)
    plan = {
        "kind": "normals_promotion_plan",
        "run_id": run.run_id,
        "planned_at": datetime.now(UTC).isoformat(),
        "run_record_sha256": record_sha,
        **_store_record(run),
        "zones": planned,
        "run_writes": [
            {
                "op": "PUT If-None-Match: * (run event, after every zone)",
                "key": _RUN_EVENT.format(run_id=run.run_id, seq=len(run_events) + 1),
            }
        ],
    }
    path = _under(run.root, _PLAN.format(run_id=run.run_id))
    _atomic_write_bytes(path, _record_bytes(plan))
    print(json.dumps(plan, indent=2, sort_keys=True))
    stopped, unrecorded = _print_parity(planned)
    if not stopped and not unrecorded:
        print(f"\nplan {path}. Nothing was written to the store. Review it, then promote.")
        return 0
    # Exit 0 all the same: the plan is made and is what the investigation starts from. The stop
    # is enforced by promote, which refuses this plan whole.
    print(f"\nplan {path}. Nothing was written to the store.")
    stops: list[str] = []
    if unrecorded:
        stops.append(
            f"STOP: the staged object plan read for {unrecorded} is not the one the run record "
            "holds, so its parity verdict is about other bytes than the ones promote writes. "
            "promote refuses this run before any write. Do not promote: find what wrote the "
            "staging key outside this protocol, then re-run plan on the recorded bytes"
        )
    if stopped:
        stops.append(
            f"STOP: parity is not exact for {stopped}. promote refuses this run before any "
            f"write, and no flag overrides it. Do not promote: investigate following {_PARITY_STOP}"
        )
    for stop in stops:
        print(stop)
        print(stop, file=sys.stderr)
    return 0


def _staged_is_recorded(staged: dict | None, recorded_sha256: str) -> bool:
    """Whether the plan's parity verdict is about the staged bytes the run record holds.

    ``promote`` writes the staging object only when its sha256 is the run record's, so this is
    what binds the verdict to the bytes written, as the planned etag and sha256 bind it on the
    canonical side. A plan that read another staging object judged other bytes.
    """
    staged = staged or {}
    return (
        staged.get("matches_record") is True
        and staged.get("sha256") == recorded_sha256
        and staged.get("recorded_sha256") == recorded_sha256
    )


def _print_parity(planned: dict[str, dict]) -> tuple[list[str], list[str]]:
    """One line per zone under the plan's JSON. Returns the zones whose parity differs, and the
    zones whose staged object is not the recorded one (their verdict is about other bytes)."""
    print(
        f"\nparity (rule: exact; all-days {', '.join(_PARITY_FIELDS)} on {_PARITY_KEYS} unique "
        "(variable, month) keys per side):"
    )
    stopped: list[str] = []
    unrecorded: list[str] = []
    for zone_id, zone in planned.items():
        parity, staged = zone["parity"], zone["staged"]
        if not _staged_is_recorded(staged, staged["recorded_sha256"]):
            unrecorded.append(zone_id)
            label = "NOT RECORDED"
            detail = (
                f"STOP: plan read staged sha256 {staged['sha256']}, the run record holds "
                f"{staged['recorded_sha256']}; this zone's verdict ({parity['status']}) is about "
                "those other bytes"
            )
        elif parity["status"] == "exact":
            label, detail = "exact", f"{parity['identical_keys']}/{_PARITY_KEYS} keys identical"
        elif parity["status"] == "not_applicable":
            label, detail = "not applicable", "no canonical before this run"
        else:
            stopped.append(zone_id)
            label, detail = "DIFFERS", "STOP: " + "; ".join(parity["reasons"])
        print(f"  {label:14} {zone_id:24} {detail}")
    return stopped, unrecorded


def _is(entry: dict | None, kind: str | tuple[str, ...], run_id: str) -> bool:
    kinds = (kind,) if isinstance(kind, str) else kind
    return entry is not None and entry.get("kind") in kinds and entry.get("run_id") == run_id


def _is_exact(parity: dict | None) -> bool:
    parity = parity or {}
    return parity.get("status") == "exact" and parity.get("exact") is True


def _as_recorded(parity: dict) -> dict:
    """``parity`` as ``plan.json`` holds it once written and read back: a verdict recomputed from
    the same bytes compares equal to the reviewed one only in that form."""
    return json.loads(json.dumps(parity))


def _promoted_with_parity(event: dict | None) -> bool:
    """Whether a promote event carries the evidence the EXACT rule asks of a resumed zone: an
    exact verdict, or ``not_applicable`` where there was no canonical before. A promote event
    written by code older than the rule carries none (Codex P2, PR #225)."""
    parity = (event or {}).get("parity") or {}
    if _is_exact(parity):
        return True
    return (
        parity.get("status") == "not_applicable"
        and (event or {}).get("canonical_sha256_before") is None
    )


def _parity_record(parity: dict) -> dict:
    """What a promote event keeps of the verdict promote recomputed for the zone."""
    return {k: parity.get(k) for k in ("rule", "status", "exact", "identical_keys")}


def _require_parity(
    run: _Run,
    plan: dict,
    done: list[str],
    recorded: dict[str, str],
    recomputed: dict[str, dict],
    owners: dict[str, dict | None],
) -> None:
    """The EXACT parity rule (#223, decision 5), for the whole run.

    ``recomputed`` holds, per zone with a canonical, the verdict promote computed itself on the
    staged bytes it writes and the canonical they replace, read again from the store with the
    sha256 the plan saw. That verdict is the one the rule is enforced on. The plan's own verdict
    is read too, as the reviewed signal, but never trusted alone: ``plan.json`` is a local file.

    Refused before any write, and every such zone named:

    - a zone whose plan judged staged bytes other than the ones promote writes: the staging
      object plan read is not the run record's (``recorded``, the sha256 promote requires of the
      staging object it writes). Its verdict, whatever it says, is about those other bytes, as a
      verdict on a canonical whose etag moved since plan is about another canonical;
    - a zone whose recomputed verdict is not exact, or whose plan says ``differs``;
    - a zone whose plan says exact while the recomputed verdict is not: the plan was edited by
      hand, or written by other code;
    - a zone the plan saw a canonical for without an exact verdict (a plan made before the rule,
      or one edited by hand).

    No flag overrides it: a deviation is investigated, never waived.

    A zone this run already promoted is exempt when its promote event (``owners``) carries the
    verdict it was promoted on: its promotion is being resumed, and canonical already holds this
    run's v2. A promote event without one was written by code older than the rule, and is refused.
    A plan whose exact verdict is not, field for field, the one promote recomputes is refused
    too: the reviewed report must be the one the rule is enforced on (Greptile P2, PR #225).
    """
    unbound: list[str] = []
    differing: list[str] = []
    contradicted: list[str] = []
    unjudged: list[str] = []
    unproven: list[str] = []
    for zone_id in run.zone_ids:
        if zone_id in done:
            if not _promoted_with_parity(owners.get(zone_id)):
                unproven.append(zone_id)
            continue
        zone = plan["zones"][zone_id]
        if not _staged_is_recorded(zone.get("staged"), recorded[zone_id]):
            unbound.append(zone_id)
            continue
        parity = zone.get("parity") or {}
        now_exact = zone_id not in recomputed or _is_exact(recomputed[zone_id])
        if parity.get("status") == "differs" or not now_exact:
            differing.append(zone_id)
            if _is_exact(parity):
                contradicted.append(zone_id)
        elif (zone.get("canonical") or {}).get("exists") is not False and not _is_exact(parity):
            unjudged.append(zone_id)
        elif zone_id in recomputed and _as_recorded(recomputed[zone_id]) != parity:
            contradicted.append(zone_id)
    if not unbound and not differing and not unjudged and not contradicted and not unproven:
        return
    found = []
    if unbound:
        found.append(
            f"the plan judged staged bytes other than the ones promote writes for {unbound} (the "
            "staging object plan read is not the one the run record holds, so its verdict is "
            "about other bytes: find what wrote the staging key outside this protocol, then "
            "re-run plan on the recorded bytes and review it)"
        )
    if differing:
        found.append(f"parity is not exact for {differing}")
    if contradicted:
        found.append(
            f"the plan's exact verdict for {contradicted} is not the one promote recomputes from "
            "the staged bytes and the canonical they replace: plan.json was edited, or written by "
            "other code. Do not edit plan.json: re-run plan and review it"
        )
    if unjudged:
        found.append(
            f"the plan has no exact parity verdict for {unjudged}, which have a canonical (a plan "
            "made before the rule records none: re-run plan and review it)"
        )
    if unproven:
        found.append(
            f"{unproven} were promoted by this run with no parity verdict in their promote event "
            "(code older than the rule): roll the run back, then re-run plan and promote"
        )
    tail = "No flag overrides it"
    if differing or unjudged:
        tail += f" and no tolerance waives it: investigate following {_PARITY_STOP}"
    raise OperatorRefusal(
        f"run {run.run_id!r} is refused before any write under the EXACT parity rule: "
        f"{'; and '.join(found)}. {tail}"
    )


def _s3_promote(run: _Run) -> int:
    requested = list(NORMALIZED_VARS)
    record, record_sha = _load_run_record(run)
    _require_same_authority(run, record, what="the published run record")
    _require_canonical_window(record.get("window"), what="the published run record")
    zones = _zones_by_id(record)
    _require_zone_set(
        list(zones), run.zone_ids, what="materialized", run_id=run.run_id, hint=_PROMOTE_THE_RUN
    )
    plan_path = _under(run.root, _PLAN.format(run_id=run.run_id))
    if not plan_path.exists():
        raise OperatorRefusal(
            f"no plan for run {run.run_id!r} at {plan_path} — promote acts on a reviewed plan, "
            "and the plan's etags are what it refuses on; run plan first"
        )
    # Read once: the sha256 each promote event records is that of the bytes acted on.
    plan_body = plan_path.read_bytes()
    plan, plan_sha = json.loads(plan_body), _sha256(plan_body)
    if plan.get("run_record_sha256") != record_sha or set(plan.get("zones", {})) != set(zones):
        raise OperatorRefusal(
            f"the plan at {plan_path} was made for another run record — re-run plan and review it"
        )
    _require_same_authority(run, plan, what="the plan")

    journals = {z: _journal(run.store, _ZONE_EVENT, zone_id=z) for z in run.zone_ids}
    done = [z for z in run.zone_ids if _is(_owner(journals[z]), "promote", run.run_id)]
    run_events = _journal(run.store, _RUN_EVENT, run_id=run.run_id)
    run_promoted = _is(run_events[-1] if run_events else None, "promoted", run.run_id)
    if len(done) == len(run.zone_ids) and run_promoted:
        raise OperatorRefusal(
            f"run {run.run_id!r} is already promoted. Roll it back first, or materialize a new run"
        )
    # This run's promotion still stands in a zone, but another run's sits on top of it: the same
    # predicate as rollback's N4 refusal. Named before the unfinished-rollback case below, which
    # it would otherwise be mistaken for, and whose remedy (`rollback`) N4 refuses here.
    buried = {
        z: _owner(journals[z])
        for z in run.zone_ids
        if z not in done and any(e.get("run_id") == run.run_id for e in _owners(journals[z]))
    }
    if run_promoted and buried:
        described = ", ".join(
            f"{z}: run {o['run_id']!r} (promote, zone event {o['seq']})" for z, o in buried.items()
        )
        raise OperatorRefusal(
            f"run {run.run_id!r} is already promoted, and another run's promotion sits on top of "
            f"it — {described}. Roll the owning run back first"
        )
    if run_promoted or _unrecorded_rollbacks(journals, run_events, run.run_id):
        # The run journal says promoted and the zone journals say otherwise, or a zone's latest
        # entry by this run is a rollback no run event records: a rollback restored the zones and
        # stopped before its run event. Finishing it comes first, or the run journal would record
        # two promotions and no rollback between them, or skip a rollback whose promotion never
        # wrote its own run event.
        raise OperatorRefusal(
            f"run {run.run_id!r} was rolled back without its run event. Run rollback again to "
            "write it, then promote"
        )
    # Every zone promoted and no run event: the promotion stopped after its last zone event. The
    # checks below still run, nothing is written for the zones, and only the run event is added.
    # The plan lists the exact writes, and a journal that moved since makes that list stale: after
    # a promote and a rollback the kept copy exists and the next event is not the one listed. A
    # zone this run already promoted is exempt, so a promotion whose process died resumes on the
    # plan it started from; it is checked below against canonical instead.
    for zone_id in run.zone_ids:
        planned_entries = (plan["zones"][zone_id].get("journal") or {}).get("entries")
        if zone_id not in done and len(journals[zone_id]) != planned_entries:
            raise OperatorRefusal(
                f"the zone journal for {zone_id!r} has {len(journals[zone_id])} entries and the "
                f"plan saw {planned_entries}, so the writes it lists are not the ones promote "
                "would make. Re-run plan and review it again"
            )

    # Validate EVERY zone before writing ANY, as the local mode does.
    work: dict[str, dict] = {}
    recomputed: dict[str, dict] = {}
    for zone_id in run.zone_ids:
        recorded_sha = zones[zone_id]["staged_sha256"]
        canonical_key = _CANONICAL.format(zone_id=zone_id)
        if zone_id in done:
            # A resumed promotion skips this zone, but only while canonical still holds what this
            # run put there.
            current = run.store.get(canonical_key)
            if current is None or current[1]["sha256"] != recorded_sha:
                kept_sha = _owner(journals[zone_id]).get("rollback_sha256")
                if current is not None and kept_sha and current[1]["sha256"] == kept_sha:
                    # An unwind or a rollback of this run restored the kept copy and stopped
                    # before the event that releases the zone: nobody else wrote canonical.
                    raise OperatorRefusal(
                        f"{zone_id!r} was promoted by this run and canonical holds this run's "
                        f"kept copy again (rollback/{run.run_id}/{zone_id}.parquet): an unwind "
                        "or a rollback restored it and stopped before its event. Run rollback, "
                        "which writes only that event for this zone, then re-plan and promote"
                    )
                raise OperatorRefusal(
                    f"{zone_id!r} was promoted by this run and canonical no longer holds its "
                    "staged bytes — it was written outside this protocol since"
                )
            continue

        staged_key = _STAGING.format(run_id=run.run_id, zone_id=zone_id)
        got = run.store.get(staged_key)
        if got is None:
            raise OperatorRefusal(f"nothing staged for {zone_id!r} under run {run.run_id!r}")
        if got[1]["sha256"] != recorded_sha:
            raise OperatorRefusal(
                f"the staged object for {zone_id!r} has sha256 {got[1]['sha256']}, and the run "
                f"record says {recorded_sha}"
            )
        staged_local = _readback_path(run, "staged", zone_id)
        _atomic_write_bytes(staged_local, got[0])
        _validate_staged(staged_local, zone_id, requested)

        planned = plan["zones"][zone_id]["canonical"]
        head = run.store.head(canonical_key)
        if (head or {}).get("etag") != planned.get("etag") or (head is None) == planned["exists"]:
            raise OperatorRefusal(
                f"canonical for {zone_id!r} changed since plan (etag {planned.get('etag')} at "
                f"plan, {(head or {}).get('etag')} now). Re-run plan and review it again"
            )
        # The parity verdict the rule is enforced on is promote's own, computed on the bytes it
        # writes (staged_local, checked above against the run record) and on canonical read
        # again from the store with the sha256 the plan saw. The verdict in plan.json is a local
        # file's claim, and the canonical readback plan left beside it is a local file too.
        parity: dict | None = None
        if planned["exists"]:
            current = run.store.get(canonical_key)
            if current is None or current[1]["sha256"] != planned.get("sha256"):
                raise OperatorRefusal(
                    f"canonical for {zone_id!r} changed since plan (sha256 "
                    f"{planned.get('sha256')} at plan, "
                    f"{current[1]['sha256'] if current else None} now). Re-run plan and review "
                    "it again"
                )
            canonical_local = _under(
                run.root,
                _READBACK_CANONICAL.format(
                    run_id=run.run_id, zone_id=zone_id, sha256=current[1]["sha256"]
                ),
            )
            _atomic_write_bytes(canonical_local, current[0])
            parity = _percentile_comparison(staged_local, canonical_local, zone_id=zone_id)[
                "parity"
            ]

        kept_key = _ROLLBACK.format(run_id=run.run_id, zone_id=zone_id)
        absent_key = _ABSENT.format(run_id=run.run_id, zone_id=zone_id)
        kept, absent = run.store.get(kept_key), run.store.head(absent_key)
        owner = _owner(journals[zone_id])
        if owner is not None and (
            absent or (kept is not None and kept[1]["sha256"] != owner["canonical_sha256_after"])
        ):
            # This run's rollback copy predates that promotion. Promoting over it would let this
            # run's rollback silently erase the other run's promotion. A kept copy that IS the
            # owner's bytes was taken while it owned canonical (this run was rolled back off the
            # top of its promotion), and restoring it cannot undo that promotion.
            raise OperatorRefusal(
                f"{zone_id!r}: canonical is owned by run {owner['run_id']!r} (zone event "
                f"{owner['seq']}), and run {run.run_id!r}'s rollback copy predates that "
                "promotion. Roll the owning run back first, or use a new run id"
            )
        if kept and planned.get("sha256") not in {kept[1]["sha256"], recorded_sha}:
            raise OperatorRefusal(
                f"{zone_id!r}: a rollback copy already exists under run {run.run_id!r} and "
                "describes neither the current canonical nor this run's staged bytes. It is "
                "kept, never rewritten, so this run cannot promote over the current canonical"
            )
        if absent and planned["exists"] and planned.get("sha256") != recorded_sha:
            raise OperatorRefusal(
                f"{zone_id!r}: this run's .absent marker says there was no canonical, and one "
                "now exists with other bytes"
            )
        work[zone_id] = {
            "staged": got[0],
            "staged_sha256": recorded_sha,
            "planned": planned,
            "kept_exists": kept is not None,
            "kept": kept[0] if kept else None,
            "kept_sha256": kept[1]["sha256"] if kept else None,
            "absent_exists": absent is not None,
            "parity": _parity_record(parity or _parity_not_applicable()),
            "plan_sha256": plan_sha,
        }
        if parity is not None:
            recomputed[zone_id] = parity
    # After the checks above, which name a stale plan (whose verdict is stale with it), and before
    # the first write.
    _require_parity(
        run,
        plan,
        done,
        {z: zones[z]["staged_sha256"] for z in run.zone_ids},
        recomputed,
        {z: _owner(journals[z]) for z in done},
    )

    if_match = bool(run.args.if_match)
    touched: list[str] = []
    before: dict[str, bytes | None] = {}
    try:
        with run.store.canonical_writes():
            for zone_id, zone_work in work.items():
                _promote_zone(run, zone_id, zone_work, journals[zone_id], if_match, touched, before)
    except Exception:
        # What this invocation's PUT is known to have left: the staged bytes, or what the readback
        # right after it saw instead. Nothing else is restored over.
        ours = {z: {w["staged_sha256"], w.get("readback_sha256")} - {None} for z, w in work.items()}
        resumed = {z for z, w in work.items() if w.get("resumed")}
        _unwind(run, touched, before, ours, if_match, resumed=resumed)
        raise

    run_events = _journal(run.store, _RUN_EVENT, run_id=run.run_id)
    _append_event(
        run.store,
        _RUN_EVENT,
        run_events,
        {
            "kind": "promoted",
            "run_id": run.run_id,
            "at": datetime.now(UTC).isoformat(),
            "run_record_sha256": record_sha,
            **_store_record(run),
            "zones": {
                z: {
                    k: _owner(journals[z])[k]
                    for k in ("seq", "canonical_sha256_before", "canonical_sha256_after")
                }
                for z in run.zone_ids
            },
        },
        run_id=run.run_id,
    )
    if not work:
        print(
            f"\nevery zone was already promoted by run {run.run_id!r}; wrote only the run event "
            "that promotion stopped before."
        )
        return 0
    print(
        f"\npromoted {len(work)} zone(s)"
        + (f", {len(done)} already promoted by this run" if done else "")
        + f". Previous canonical kept under rollback/{run.run_id}/."
    )
    return 0


def _promote_zone(
    run: _Run,
    zone_id: str,
    work: dict,
    journal: list[dict],
    if_match: bool,
    touched: list[str],
    before: dict[str, bytes | None],
) -> None:
    store = run.store
    planned = work["planned"]
    canonical_key = _CANONICAL.format(zone_id=zone_id)
    current = store.get(canonical_key)
    before_bytes = current[0] if current else None
    before_sha = current[1]["sha256"] if current else None
    # Before ANY write for this zone, so the common race (another zone's turn took long enough
    # for canonical to move) refuses without leaving a rollback copy behind.
    if before_sha != planned.get("sha256") or (current[1]["etag"] if current else None) != (
        planned.get("etag")
    ):
        raise OperatorRefusal(
            f"canonical for {zone_id!r} changed since plan, before anything was written for it. "
            "Re-run plan and review it again"
        )

    # 1. The rollback copy, create-only and read back. An existing one is kept as it is: it
    #    describes canonical before this run EVER promoted.
    #    Its sha256 goes in the event on its own: after a promotion that stopped between the PUT
    #    and its event, canonical at the resumed start is already the staged v2, and only the kept
    #    copy still describes the state a rollback must restore.
    rollback_sha: str | None = None
    if work["kept_exists"]:
        rollback_key = _ROLLBACK.format(run_id=run.run_id, zone_id=zone_id)
        rollback_sha = work["kept_sha256"]
    elif work["absent_exists"]:
        rollback_key = _ABSENT.format(run_id=run.run_id, zone_id=zone_id)
    elif before_bytes is None:
        rollback_key = _ABSENT.format(run_id=run.run_id, zone_id=zone_id)
        _publish_create_only(store, rollback_key, _ABSENT_MARKER, content_type="text/plain")
    else:
        rollback_key = _ROLLBACK.format(run_id=run.run_id, zone_id=zone_id)
        rollback_sha = before_sha
        _publish_create_only(
            store, rollback_key, before_bytes, content_type="application/vnd.apache.parquet"
        )

    # 2. Canonical must still be what the plan saw. A change caught HERE comes after the rollback
    #    copy (or marker) was written, and that copy is kept, never rewritten: it describes the
    #    canonical the plan saw, so this run id cannot promote the zone again unless canonical
    #    returns to those bytes.
    head = store.head(canonical_key)
    if (head or {}).get("etag") != planned.get("etag"):
        raise OperatorRefusal(
            f"canonical for {zone_id!r} changed since plan (etag {planned.get('etag')} at plan, "
            f"{(head or {}).get('etag')} now), after this run's rollback copy for it was written. "
            f"That copy describes the canonical the plan saw, so run {run.run_id!r} cannot "
            "promote this zone unless canonical returns to it: find the other writer, then "
            "materialize under a new run id"
        )

    # 3. The write. Marked touched BEFORE the call: a PUT that raised may still have landed. A
    #    refused precondition is the exception once canonical is read and does not hold this
    #    run's staged bytes — then nothing of this run's landed, so the zone is not touched and
    #    the unwind leaves the other writer's object alone.
    touched.append(zone_id)
    # On a resume after a promotion that stopped between this PUT and its event, canonical already
    # holds this run's staged bytes, and restoring those would release the journal on a zone that
    # still serves this run's v2. The unwind returns such a zone to the pre-run state its kept copy
    # holds; with only the .absent marker there was no canonical, so it is left in place, owned.
    work["resumed"] = before_sha == work["staged_sha256"] and (
        work["kept_exists"] or work["absent_exists"]
    )
    before[zone_id] = work["kept"] if work["resumed"] else before_bytes
    conditional = before_bytes is None or if_match
    try:
        if before_bytes is None:
            landed = store.put_if_absent(
                canonical_key, work["staged"], content_type="application/vnd.apache.parquet"
            )
        else:
            store.put_canonical(
                canonical_key, work["staged"], if_match=planned.get("etag") if if_match else None
            )
            landed = True
    except Exception as exc:
        if not (conditional and _conflict(exc)):
            raise
        landed = False
    if not landed:
        # A refusal alone does not prove that nothing landed. botocore retries a PUT whose
        # response was lost (a timeout, a reset, a 5xx after the commit) with the same
        # precondition, and the store refuses the retry because the first attempt landed. So the
        # key is read before deciding: this invocation's staged bytes there mean the PUT landed.
        now = store.get(canonical_key)
        landed = now is not None and now[1]["sha256"] == work["staged_sha256"]
    if not landed:
        touched.pop()
        del before[zone_id]
        raise OperatorRefusal(
            f"canonical for {zone_id!r} changed since plan: the store refused the conditional "
            f"PUT ({'If-None-Match' if before_bytes is None else 'If-Match'}) and canonical does "
            "not hold this run's staged bytes, so nothing of this run's landed there and it is "
            "left as the other writer made it"
        )

    # 4. Read back what serving will read. Bytes other than the staged ones, seen right after this
    #    run's PUT, are what that PUT produced as far as anything can tell, so the unwind may
    #    restore over them; anything seen later is another writer's.
    got = store.get(canonical_key)
    if got is None or got[1]["sha256"] != work["staged_sha256"]:
        work["readback_sha256"] = got[1]["sha256"] if got else None
        raise OperatorRefusal(
            f"canonical for {zone_id!r} reads back with sha256 "
            f"{got[1]['sha256'] if got else None}, expected the staged {work['staged_sha256']}"
        )

    # 5. The journal names this run as canonical's owner.
    _append_event(
        store,
        _ZONE_EVENT,
        journal,
        {
            "kind": "promote",
            "run_id": run.run_id,
            "zone_id": zone_id,
            "at": datetime.now(UTC).isoformat(),
            "canonical_key": canonical_key,
            "canonical_sha256_before": before_sha,
            "canonical_etag_before": planned.get("etag"),
            "canonical_sha256_after": work["staged_sha256"],
            "canonical_etag_after": got[1]["etag"],
            "rollback_key": rollback_key,
            "rollback_sha256": rollback_sha,
            "if_match": if_match,
            # The verdict this promotion relied on (promote's own recompute, never the plan's
            # alone), and the plan.json it acted on.
            "parity": work["parity"],
            "plan_sha256": work["plan_sha256"],
            **_store_record(run),
        },
        zone_id=zone_id,
    )
    print(
        f"  promoted {zone_id:24} -> s3://{store.bucket}/{canonical_key}  "
        f"(parity {work['parity']['status']}, recomputed by promote)"
    )


def _unwind(
    run: _Run,
    touched: list[str],
    before: dict[str, bytes | None],
    ours: dict[str, set[str]],
    if_match: bool,
    *,
    resumed: set[str],
) -> None:
    """Best-effort return of every touched zone to its bytes before this run promoted it. No delete.

    That is its bytes before this invocation, except in a ``resumed`` zone: one this invocation
    found already holding this run's staged bytes, left by a promotion that stopped between its
    PUT and its event. ``before`` then holds the kept copy, which is what canonical held before
    this run promoted it; restoring the staged bytes instead would release the journal on a zone
    still serving this run's v2.

    Nothing is restored blind. Canonical is read first, and only a zone that still holds what
    THIS invocation's PUT left there (the staged bytes, or the other bytes its readback saw right
    after the PUT) is written: a PUT that raised without landing left canonical as it was, and a
    write that landed later is another writer's, which a restore would erase. The restore carries
    ``If-Match`` on the etag of that read when the operator opted in, and is read back.

    The journal is re-read rather than trusted from memory: a promote event can land and then fail
    its readback. Wherever it names this run as the owner of a canonical that no longer holds this
    run's bytes, an ``unwind`` event says so.
    """
    restored: list[str] = []
    left_in_place: list[str] = []
    foreign: list[str] = []
    failed: list[str] = []
    for zone_id in reversed(touched):
        key = _CANONICAL.format(zone_id=zone_id)
        data = before[zone_id]
        try:
            current = run.store.get(key)
            current_sha = current[1]["sha256"] if current else None
            if current_sha in ours[zone_id]:
                if data is None:
                    # This run's bytes, where there was no canonical: named, not removed. The
                    # journal (if the event landed) rightly names this run as their owner.
                    left_in_place.append(f"s3://{run.store.bucket}/{key}")
                    continue
                with run.store.canonical_writes():
                    try:
                        run.store.put_canonical(
                            key, data, if_match=current[1]["etag"] if if_match else None
                        )
                    except Exception as exc:
                        if not (if_match and _conflict(exc)):
                            raise
                        # A restore that landed and lost its response is retried with the same
                        # If-Match, which the store then refuses: the bytes it restores in
                        # canonical mean it landed, and the readback below confirms it.
                        now = run.store.get(key)
                        if now is None or now[1]["sha256"] != _sha256(data):
                            raise OperatorRefusal(
                                "the store refused the restore's If-Match and canonical does not "
                                "hold the bytes it restores, so the restore did not land"
                            ) from exc
                got = run.store.get(key)
                if got is None or got[1]["sha256"] != _sha256(data):
                    raise OperatorRefusal("the restored canonical reads back with other bytes")
                restored.append(zone_id)
            elif current_sha != (_sha256(data) if data is not None else None):
                foreign.append(f"s3://{run.store.bucket}/{key}")
                continue
            entries = _journal(run.store, _ZONE_EVENT, zone_id=zone_id)
            if _is(_owner(entries), "promote", run.run_id):
                _append_event(
                    run.store,
                    _ZONE_EVENT,
                    entries,
                    {
                        "kind": "unwind",
                        "run_id": run.run_id,
                        "zone_id": zone_id,
                        "at": datetime.now(UTC).isoformat(),
                        "canonical_key": key,
                        "canonical_sha256_after": _sha256(data) if data is not None else None,
                        **_store_record(run),
                    },
                    zone_id=zone_id,
                )
        except Exception as exc:
            failed.append(f"{zone_id} ({redact_text(str(exc), settings.S3_ENDPOINT)})")
    from_kept = sorted(z for z in restored if z in resumed)
    print(
        f"promotion failed after touching {len(touched)} zone(s); restored to their bytes before "
        f"this invocation: {sorted(set(restored) - set(from_kept)) or 'none'}.",
        file=sys.stderr,
    )
    if from_kept:
        print(
            "Restored to this run's kept copy under "
            f"rollback/{run.run_id}/, the canonical before this run promoted them, because they "
            "already held this run's staged bytes when this invocation began (a promotion that "
            f"stopped between its PUT and its event): {from_kept}.",
            file=sys.stderr,
        )
    if failed:
        print(
            "THE UNWIND FAILED for these, on canonical or on the journal entry that releases it; "
            f"their rollback copies are kept under rollback/{run.run_id}/: " + ", ".join(failed),
            file=sys.stderr,
        )
    if foreign:
        print(
            "CHANGED BY ANOTHER WRITER after this run's PUT, so NOT restored and left as they are: "
            + ", ".join(foreign)
            + ". Find that writer before anything else",
            file=sys.stderr,
        )
    if left_in_place:
        print(
            "These had no canonical before this run and now hold its staged v2. This "
            "operator does not delete; remove them yourself to return to the previous state: "
            + ", ".join(left_in_place),
            file=sys.stderr,
        )
    if not failed and not foreign and not left_in_place:
        print("No canonical object holds bytes this invocation wrote.", file=sys.stderr)


def _released_without_event(
    run: _Run, journals: dict[str, list[dict]], owned: list[str]
) -> list[str]:
    """The zones in ``owned`` (this run's promotion stands in each) whose canonical is back in its
    state from before this run: the kept copy, or no object where the promotion's rollback key is
    the ``.absent`` marker. ``promote`` refuses to resume such a zone, and ``rollback`` would only
    write its event."""
    released = []
    for zone_id in owned:
        event = _owner(journals[zone_id])
        got = run.store.get(_CANONICAL.format(zone_id=zone_id))
        sha = got[1]["sha256"] if got else None
        if sha == event["canonical_sha256_after"]:
            continue
        if (got is None and event["rollback_key"].endswith(".absent")) or (
            sha is not None and sha == event.get("rollback_sha256")
        ):
            released.append(zone_id)
    return released


def _restore_canonical(
    run: _Run,
    zone_id: str,
    data: bytes,
    source_key: str,
    verified_etag: str,
    if_match: bool,
    *,
    note: str = "",
) -> None:
    """Put a kept copy back in canonical and read it back. ``verified_etag`` is the etag of the
    read that verified canonical, which is what ``--if-match`` sends: a fresh HEAD would carry the
    etag of whatever landed since that check."""
    canonical_key = _CANONICAL.format(zone_id=zone_id)
    try:
        run.store.put_canonical(canonical_key, data, if_match=verified_etag if if_match else None)
    except Exception as exc:
        if not (if_match and _conflict(exc)):
            raise
        # A restore that landed and lost its response is retried with the same If-Match, which
        # the store then refuses: the kept copy's bytes in canonical mean it landed, and the
        # readback below confirms it.
        now = run.store.get(canonical_key)
        if now is None or now[1]["sha256"] != _sha256(data):
            raise OperatorRefusal(
                f"canonical for {zone_id!r} changed after this rollback verified it: the store "
                "refused If-Match and canonical does not hold the kept copy, so nothing was "
                "restored for it. Find the other writer before anything else"
            ) from exc
    got = run.store.get(canonical_key)
    if got is None or got[1]["sha256"] != _sha256(data):
        raise OperatorRefusal(
            f"canonical for {zone_id!r} reads back with other bytes than the rollback copy at "
            f"{source_key}"
        )
    print(f"  restored {zone_id:24} <- s3://{run.store.bucket}/{source_key}{note}")


def _s3_rollback(run: _Run) -> int:
    record, _record_sha = _load_run_record(run)
    _require_same_authority(run, record, what="the published run record", fields=_STORE_FIELDS)
    zones = _zones_by_id(record)
    _require_zone_set(
        list(zones), run.zone_ids, what="promoted", run_id=run.run_id, hint=_ROLL_BACK_THE_RUN
    )
    journals = {z: _journal(run.store, _ZONE_EVENT, zone_id=z) for z in run.zone_ids}
    to_restore = [z for z in run.zone_ids if _is(_owner(journals[z]), "promote", run.run_id)]
    # This run's promotion still stands in the zone, but another run's sits on top of it.
    foreign = {
        z: _owner(journals[z])
        for z in run.zone_ids
        if z not in to_restore and any(e["run_id"] == run.run_id for e in _owners(journals[z]))
    }
    if foreign:
        described = ", ".join(
            f"{z}: run {o['run_id']!r} (promote, zone event {o['seq']})" for z, o in foreign.items()
        )
        raise OperatorRefusal(
            f"run {run.run_id!r} does not own canonical for every zone it promoted — {described}. "
            "Rolling it back would undo a promotion it does not own (N4, #199 review). Roll back "
            "the owning run first"
        )
    # A promote killed between a zone's PUT and its event (Ctrl-C, a dropped one-shot) leaves
    # canonical on this run's staged bytes, this run's kept copy or marker in place, and no
    # standing promotion of this run accounting for those bytes. Its latest entry there is none,
    # its rollback of an earlier cycle, or its unwind of an earlier attempt: an unwind returns
    # canonical off this run's bytes, so staged bytes after one were written since. Restoring only
    # the journaled zones would report success and leave such a zone serving v2.
    stranded: dict[str, str] = {}  # zone -> the etag of the read that found the staged bytes
    for z in run.zone_ids:
        if z in to_restore:
            continue
        got = run.store.get(_CANONICAL.format(zone_id=z))
        if got is None or got[1]["sha256"] != zones[z]["staged_sha256"]:
            continue
        owner = _owner(journals[z])
        if owner is not None and owner.get("canonical_sha256_after") == got[1]["sha256"]:
            continue  # another run's promotion of identical bytes (N4) accounts for them
        if run.store.head(_ROLLBACK.format(run_id=run.run_id, zone_id=z)) or run.store.head(
            _ABSENT.format(run_id=run.run_id, zone_id=z)
        ):
            stranded[z] = got[1]["etag"]
    # The refusal below names a resume, and promote refuses to resume while a zone this run still
    # owns is back in its state from before this run with no event that says so: its kept copy,
    # put back by an unwind or a rollback that stopped before its event, or no object, where the
    # operator removed this run's v2 by hand as the unwind asks. Each step would name the other,
    # so rollback restores the stranded zones from their kept copies itself. Read only when a zone
    # is stranded: otherwise the only canonical read is the one each restore's If-Match is on.
    released = _released_without_event(run, journals, to_restore) if stranded else []
    unjournaled: dict[str, tuple[bytes, str, str]] = {}
    if stranded and released:
        absent_only = []
        for z, etag in stranded.items():
            kept_key = _ROLLBACK.format(run_id=run.run_id, zone_id=z)
            kept = run.store.get(kept_key)
            if kept is None:
                absent_only.append(z)
            else:
                unjournaled[z] = (kept[0], kept_key, etag)
        if absent_only:
            removals = ", ".join(
                f"s3://{run.store.bucket}/{_CANONICAL.format(zone_id=z)}" for z in absent_only
            )
            raise OperatorRefusal(
                f"canonical for {', '.join(stranded)} holds run {run.run_id!r}'s staged bytes and "
                "no zone event says so: promote stopped between its PUT and its event, and "
                f"promote cannot resume it, because {', '.join(released)} is back in its state "
                f"from before this run with no event that says so. {', '.join(absent_only)} had "
                f"NO canonical object before run {run.run_id!r} (only its .absent marker), so "
                "restoring that state means removing an object, and this operator does not "
                f"delete — remove {removals} yourself, then run rollback again. Nothing was "
                "written"
            )
    elif stranded:
        raise OperatorRefusal(
            f"canonical for {', '.join(stranded)} holds run {run.run_id!r}'s staged bytes and no "
            "zone event says so: promote stopped between its PUT and its event. Nothing was "
            "written. Re-run plan and promote to resume, then roll back"
        )

    run_events = _journal(run.store, _RUN_EVENT, run_id=run.run_id)
    finishing = not to_restore
    latest = {z: _latest_by(journals[z], run.run_id) for z in run.zone_ids}
    # Finishing needs a rollback that no rolled_back run event records yet. A zone an unwind of
    # this run already returned is done as well: a resumed promotion that failed unwinds the zones
    # it promoted, and the rollback written without a failure records that zone by its unwind.
    if finishing and (
        not _unrecorded_rollbacks(journals, run_events, run.run_id)
        or not all(e is None or _is(e, ("rollback", "unwind"), run.run_id) for e in latest.values())
    ):
        raise OperatorRefusal(
            f"run {run.run_id!r} is not marked promoted — no zone journal names it as the owner "
            "of canonical, so there is nothing this run replaced"
        )
    # Otherwise, with nothing left to restore, an earlier rollback restored every zone it had to
    # and wrote every zone event, then stopped before the run event: only that event is written
    # below. The promotion before it need not have written its own run event, and a zone a dead
    # promotion never reached has no entry at all.

    # Resolve every source before restoring any, for the same reason promotion validates first.
    # Each source keeps the etag of the read that verified canonical, which is what --if-match
    # sends: a fresh HEAD would carry the etag of whatever landed since that check.
    sources: dict[str, tuple[bytes, str, str]] = {}
    already_restored: set[str] = set()
    promotions = {z: _owner(journals[z]) for z in to_restore}
    for zone_id in to_restore:
        event = promotions[zone_id]
        canonical_key = _CANONICAL.format(zone_id=zone_id)
        current = run.store.get(canonical_key)
        if event["rollback_key"].endswith(".absent"):
            if current is None:
                continue  # already back to "no canonical": only the journal entry is left
            raise OperatorRefusal(
                f"{zone_id!r} had NO canonical object before run {run.run_id!r} promoted, so "
                "restoring that state means removing an object. This operator does not delete — "
                f"remove s3://{run.store.bucket}/{canonical_key} yourself if that is what you want"
            )
        current_sha = current[1]["sha256"] if current else None
        if (
            current_sha != event["canonical_sha256_after"]
            and current_sha == event["rollback_sha256"]
        ):
            # An earlier rollback or unwind of this run restored the kept copy and stopped
            # before its zone event: canonical already holds the pre-run bytes, and only the
            # event is missing.
            already_restored.add(zone_id)
            continue
        if current is None or current_sha != event["canonical_sha256_after"]:
            raise OperatorRefusal(
                f"canonical for {zone_id!r} is not the object run {run.run_id!r} promoted — it "
                "was written or removed outside this protocol since"
            )
        kept = run.store.get(event["rollback_key"])
        if kept is None or kept[1]["sha256"] != event["rollback_sha256"]:
            raise OperatorRefusal(
                f"the rollback copy for {zone_id!r} at {event['rollback_key']} is missing or is "
                "not the canonical this run replaced — inventing a restore would publish an "
                "artifact nobody kept"
            )
        sources[zone_id] = (kept[0], event["rollback_key"], current[1]["etag"])

    if_match = bool(run.args.if_match)
    with run.store.canonical_writes():
        # The stranded zones first: once a zone event of this rollback has landed, promote waits
        # for its run event, so a rerun must not find a zone still stranded.
        for zone_id, (data, kept_key, verified_etag) in unjournaled.items():
            _restore_canonical(
                run,
                zone_id,
                data,
                kept_key,
                verified_etag,
                if_match,
                note=" (its PUT landed without its promote event)",
            )
            _append_event(
                run.store,
                _ZONE_EVENT,
                journals[zone_id],
                {
                    "kind": "rollback",
                    "run_id": run.run_id,
                    "zone_id": zone_id,
                    "at": datetime.now(UTC).isoformat(),
                    "canonical_key": _CANONICAL.format(zone_id=zone_id),
                    "canonical_sha256_before": zones[zone_id]["staged_sha256"],
                    "canonical_sha256_after": _sha256(data),
                    "restored_from": kept_key,
                    "canonical_written": True,
                    # No promote event of this run names the bytes this replaced, and this
                    # entry releases none: the journal's owner is what it was before this run.
                    "without_promote_event": True,
                    "if_match": if_match,
                    **_store_record(run),
                },
                zone_id=zone_id,
            )
        for zone_id in to_restore:
            canonical_key = _CANONICAL.format(zone_id=zone_id)
            after_sha: str | None = None
            restored_from = promotions[zone_id]["rollback_key"]
            if zone_id in sources:
                data, restored_from, verified_etag = sources[zone_id]
                _restore_canonical(run, zone_id, data, restored_from, verified_etag, if_match)
                after_sha = _sha256(data)
            elif zone_id in already_restored:
                after_sha = promotions[zone_id]["rollback_sha256"]
                print(
                    f"  restored {zone_id:24} already, by a rollback or an unwind that stopped "
                    "before its event"
                )
            else:
                print(f"  released {zone_id:24} (no canonical before; removed by hand since)")
            _append_event(
                run.store,
                _ZONE_EVENT,
                journals[zone_id],
                {
                    "kind": "rollback",
                    "run_id": run.run_id,
                    "zone_id": zone_id,
                    "at": datetime.now(UTC).isoformat(),
                    "canonical_key": canonical_key,
                    "canonical_sha256_before": promotions[zone_id]["canonical_sha256_after"],
                    "canonical_sha256_after": after_sha,
                    "restored_from": restored_from,
                    "canonical_written": zone_id in sources,
                    "if_match": if_match,
                    **_store_record(run),
                },
                zone_id=zone_id,
            )

    _append_event(
        run.store,
        _RUN_EVENT,
        run_events,
        {
            "kind": "rolled_back",
            "run_id": run.run_id,
            "at": datetime.now(UTC).isoformat(),
            **_store_record(run),
            # Every zone, from this run's latest entry in it: a rollback resumed after a failure
            # restores only the rest, and the run event still describes the whole run. A zone a
            # dead promotion never reached has no entry, and says so.
            "zones": {
                z: {
                    k: (_latest_by(journals[z], run.run_id) or {}).get(k)
                    for k in ("seq", "kind", "canonical_sha256_after")
                }
                for z in run.zone_ids
            },
            **({"restored_without_promote_event": sorted(unjournaled)} if unjournaled else {}),
        },
        run_id=run.run_id,
    )
    if finishing:
        print(
            f"\nevery zone was already rolled back by run {run.run_id!r}; wrote only the run "
            "event that rollback stopped before."
        )
        return 0
    print(
        f"\nrestored {len(sources) + len(already_restored) + len(unjournaled)} zone(s). The v2 "
        "artifacts remain "
        f"under staging/{run.run_id}/ "
        "and this run can be promoted again."
    )
    return 0


def _s3_verify(run: _Run) -> int:
    """Read canonical the way ``climate_normals`` does, and check it against the journal.

    After a promote: ``schema_version`` [2], 72 rows, the wet-day columns, and the staged sha256.
    After a rollback: the sha256 of the kept copy it restored. Exit 1 on any failed check — a
    verification that fails is a finding, not a refusal.
    """
    record, _record_sha = _load_run_record(run)
    _require_same_authority(run, record, what="the published run record", fields=_STORE_FIELDS)
    _require_zone_set(
        list(_zones_by_id(record)),
        run.zone_ids,
        what="materialized",
        run_id=run.run_id,
        hint=_NAME_THE_RUN,
    )
    expected_rows = len(NORMALIZED_VARS) * 12
    report: dict[str, dict] = {}
    for zone_id in run.zone_ids:
        entries = _journal(run.store, _ZONE_EVENT, zone_id=zone_id)
        owner = _owner(entries)
        if _is(owner, "promote", run.run_id):
            state = owner
        elif _is(entries[-1] if entries else None, ("rollback", "unwind"), run.run_id):
            state = entries[-1]
        else:
            report[zone_id] = {
                "ok": False,
                "reason": "canonical is in no state this run produced: neither its promotion "
                "nor its latest restore",
                "owner": {k: owner[k] for k in ("seq", "run_id")} if owner else None,
            }
            continue
        got = run.store.get(_CANONICAL.format(zone_id=zone_id))
        entry: dict[str, Any] = {
            "state": state["kind"],
            "sha256": got[1]["sha256"] if got else None,
            "expected_sha256": state.get("canonical_sha256_after"),
        }
        rows: list[dict] = []
        uri = _default_output_uri(zone_id, "normals")
        try:
            rows = _read_normals_rows(uri, zone_id=zone_id, variables=None) if uri else []
        except Exception as exc:
            entry["read_error"] = redact_text(str(exc), settings.S3_ENDPOINT)
        columns = set().union(*(row.keys() for row in rows)) if rows else set()
        entry["rows"] = len(rows)
        entry["schema_versions"] = sorted({int(row.get("schema_version") or 1) for row in rows})
        entry["wet_day_columns"] = all(column in columns for column in _WET_DAY_COLUMNS)
        checks = {
            "sha256": entry["sha256"] == entry["expected_sha256"],
            # For every state, not only after a promote: a verify that is meant to read through
            # climate_normals' own path cannot pass while that path fails. A released zone (no
            # canonical before, removed by hand) is readable when there is nothing to read.
            "readable": got is None
            if entry["expected_sha256"] is None
            else "read_error" not in entry and entry["rows"] > 0,
        }
        if state["kind"] == "promote":
            checks.update(
                {
                    "schema_version": entry["schema_versions"] == [NORMALS_SCHEMA_VERSION],
                    "rows": entry["rows"] == expected_rows,
                    "wet_day_columns": entry["wet_day_columns"],
                }
            )
        entry["checks"] = checks
        entry["ok"] = all(checks.values())
        report[zone_id] = entry

    ok = all(entry["ok"] for entry in report.values())
    print(
        json.dumps(
            {
                "kind": "normals_promotion_verify",
                "run_id": run.run_id,
                "verified_at": datetime.now(UTC).isoformat(),
                "bucket": run.authority["bucket"],
                "endpoint": run.authority["endpoint"],
                "ok": ok,
                "zones": report,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if ok else 1


_NAME_THE_RUN = "Name exactly the run's zones: every step acts on the whole run"
_PROMOTE_THE_RUN = (
    "Promote the run the manifest describes: a partial promotion leaves some canonical objects "
    "v2 and some v1, with a record that says neither"
)
_ROLL_BACK_THE_RUN = (
    "Roll back the whole run: a partial rollback leaves some canonical objects v2 and some v1, "
    "with a record that says neither"
)

_S3_STEPS = {
    "fetch": _s3_fetch,
    "materialize": _s3_materialize,
    "publish": _s3_publish,
    "plan": _s3_plan,
    "promote": _s3_promote,
    "rollback": _s3_rollback,
    "verify": _s3_verify,
}


def _run_object_store_step(args: argparse.Namespace) -> int:
    root = _require_local(args.root)
    run_id = _require_identifier(args.run_id, role="run id")
    zone_ids = _require_scientific_zones(
        [_require_identifier(z, role="zone id") for z in args.zone_id]
    )
    if getattr(args, "era5_root", None) is not None:
        raise OperatorRefusal(
            "--era5-root is local-mode only: with --store s3, materialize reads the ERA5 that "
            "fetch wrote into --root and digested"
        )
    authority = _require_object_store_guards(args)
    run = _Run(args=args, root=root, run_id=run_id, zone_ids=zone_ids, authority=authority)
    try:
        if args.step != "materialize":
            # Only now, after every guard: nothing is constructed for a refused run, and
            # materialize never needs a client at all.
            run.store = _ObjectStore(
                _object_store_client(),
                bucket=authority["bucket"],
                run_id=run_id,
                zone_ids=zone_ids,
            )
        return _S3_STEPS[args.step](run)
    except OperatorRefusal:
        raise
    except Exception as exc:
        # botocore and DuckDB httpfs both quote the endpoint in their errors.
        message = redact_text(str(exc), settings.S3_ENDPOINT)
        print(f"FAILED: {type(exc).__name__}: {message}", file=sys.stderr)
        return 1


def _run_local_step(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    given = [f"--{f.replace('_', '-')}" for f in _OBJECT_STORE_FLAGS if getattr(args, f, None)]
    if args.step in _OBJECT_STORE_STEPS:
        raise OperatorRefusal(
            f"{args.step} is an object-store step: it runs only with --store s3 and the "
            "confirmations that go with it"
        )
    if given:
        raise OperatorRefusal(
            f"{', '.join(given)} apply only with --store s3. Without it this is the local mode, "
            "and a confirmation for a store the run will not touch means the invocation is not "
            "the one that was meant"
        )
    if args.step == "materialize" and args.era5_root is None:
        parser.error("the following arguments are required: --era5-root")
    return args.func(args)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = p.add_subparsers(dest="step", required=True)

    def _common(sp: argparse.ArgumentParser) -> None:
        sp.add_argument(
            "--root",
            type=Path,
            required=True,
            help="local store root; with --store s3, the local work root",
        )
        sp.add_argument("--run-id", required=True, help="names the staging and rollback prefixes")
        sp.add_argument("--zone-id", nargs="+", required=True, help="explicit zone ids")
        sp.add_argument(
            "--store",
            choices=("local", "s3"),
            default="local",
            help="local (default): --root is the store. s3: the bucket is S3_BUCKET",
        )
        sp.add_argument("--confirm-bucket", help="--store s3: must equal S3_BUCKET")
        sp.add_argument(
            "--confirm-endpoint",
            help="--store s3: host:port, must equal S3_ENDPOINT's host and port",
        )
        sp.add_argument(
            "--authorization", help="--store s3: URL of the comment that authorizes this run"
        )

    def _if_match(sp: argparse.ArgumentParser) -> None:
        sp.add_argument(
            "--if-match",
            action="store_true",
            help="--store s3: send If-Match on the canonical PUT. Off by default: whether RustFS "
            "honours it is unproven until the harness probe",
        )

    f = sub.add_parser("fetch", help="--store s3: GET stored ERA5 into the work root")
    _common(f)
    f.set_defaults(func=None)

    m = sub.add_parser("materialize", help="compute v2 into staging/ from stored ERA5")
    _common(m)
    m.add_argument("--era5-root", type=Path, help="local mode: root holding climate/era5/")
    m.add_argument("--start-date", default=WMO_START)
    m.add_argument("--end-date", default=WMO_END)
    m.set_defaults(func=materialize)

    pb = sub.add_parser("publish", help="--store s3: create-only upload of staging and run.json")
    _common(pb)
    pb.set_defaults(func=None)

    pl = sub.add_parser("plan", help="--store s3: HEAD/GET only; what promote would write")
    _common(pl)
    pl.set_defaults(func=None)

    pr = sub.add_parser(
        "promote", help="staging/ -> canonical, keeping the previous under rollback/"
    )
    _common(pr)
    _if_match(pr)
    pr.set_defaults(func=promote)

    rb = sub.add_parser("rollback", help="rollback/ -> canonical")
    _common(rb)
    _if_match(rb)
    rb.set_defaults(func=rollback)

    v = sub.add_parser("verify", help="--store s3: read canonical through climate_normals")
    _common(v)
    v.set_defaults(func=None)
    return p


def main(argv: list[str] | None = None) -> int:
    # Before anything drives a src/ handler. Without it structlog runs its default chain with no
    # redaction processor, and an `error=str(e)` raised inside `src/tools/` — DuckDB httpfs and
    # botocore both quote the endpoint in full, userinfo included — reaches stderr verbatim.
    # `tests/test_scripts_configure_redaction.py` caught this script missing the call, which is
    # the whole reason that guard exists.
    setup_logging()
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.store == "s3":
            return _run_object_store_step(args)
        return _run_local_step(parser, args)
    except OperatorRefusal as e:
        # Redacted like any other diagnostic: a refusal can quote a value the operator typed.
        print(f"REFUSED: {redact_text(str(e), settings.S3_ENDPOINT)}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
