"""Provider-free resume: plan a run from stored bytes, and refuse the ambiguous cases (#133).

`forecast_archive_panel.py` already resumes, and the shape of its resume is the reason this
module exists. It keys a cached artifact by a filename convention and accepts it when six
`coverage` fields match the request. Nothing in that check looks at **what produced** the
artifact. Re-materialize a source and the stale output still matches its own coverage
fields, so the panel reports `resumed` and publishes a result computed from bytes that are
no longer there.

So the unit of resume here is not the request, it is the **fingerprint of the inputs**:

    fingerprint = sha256(unit id, zone, window, output uri, every source digest)

A checkpoint records the fingerprint that produced an output. Reuse requires the
recomputed fingerprint to equal the recorded one AND the output on disk to still hash to
the digest that was recorded for it. Anything else has a name, and three of those names
stop the run.

What stops it, and why stopping is the correct answer rather than a repair:

``source_changed``
    the inputs moved under a completed unit. Re-running would overwrite an accepted
    artifact; keeping it would publish a result whose provenance is a lie. Neither is a
    decision this layer is allowed to take.

``output_modified``
    the artifact exists but no longer hashes to what the checkpoint recorded. Something
    wrote to it out of band. Silently re-running would destroy whatever that was.

``output_unclaimed``
    an artifact sits where a unit with NO checkpoint would write. Nothing in this protocol
    produced it, or something produced it and the log that would say so is gone. Treating
    it as work to redo is how "must not overwrite accepted artifacts" quietly stops being
    true the first time a checkpoint file is lost.

``source_unreachable`` / ``source_mismatch``
    a declared input cannot be read, or does not hash to the digest the manifest declares.
    The manifest is the immutable statement of what the run consumes; a run that proceeds
    past a broken one proves nothing.

This module never writes an artifact and never opens a network connection. It reads a
manifest, hashes what a caller's resolver hands it, and returns a plan. Collection —
ERA5, Open-Meteo, STAC — is not reachable from here, which is the only way "provider-free"
is a property rather than a promise.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from typing import Any

__all__ = [
    "BLOCKING_STATUSES",
    "RESUME_SCHEMA_VERSION",
    "SOURCE_MANIFEST_SCHEMA_VERSION",
    "UNIT_STATUSES",
    "Checkpoint",
    "ResumeValidationError",
    "SourceArtifact",
    "SourceManifest",
    "WorkUnit",
    "checkpoint_record",
    "bind_plan",
    "checkpoints_from_records",
    "plan_run",
    "planned_status",
    "runnable_units",
    "source_status",
    "unit_fingerprint",
]

SOURCE_MANIFEST_SCHEMA_VERSION = "run-resume-sources/v1"
RESUME_SCHEMA_VERSION = "run-resume-plan/v1"

#: Every status a unit can carry. Published in full on every plan, zeros included, so a
#: status that never fires cannot be mistaken for one the planner forgot to implement.
UNIT_STATUSES = (
    "pending",
    "reuse",
    "source_changed",
    "output_modified",
    "output_unclaimed",
    "checkpoint_mismatch",
    "source_unreachable",
    "source_mismatch",
)

#: The statuses that make a plan un-runnable. Each one is a question only a human can
#: answer, and answering it by guessing costs either an accepted artifact or the truth of
#: its provenance.
BLOCKING_STATUSES = (
    "source_changed",
    "output_modified",
    "output_unclaimed",
    "checkpoint_mismatch",
    "source_unreachable",
    "source_mismatch",
)

_HEX = "0123456789abcdef"


class ResumeValidationError(ValueError):
    """A manifest, a checkpoint or a plan input that cannot be trusted."""


def _sha256(value: Any, field: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or set(value.lower()) - set(_HEX):
        raise ResumeValidationError(f"{field} must be a sha256 hex string")
    return value.lower()


def _text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ResumeValidationError(f"{field} must be a non-empty string")
    return value


def _iso(value: Any, field: str) -> date:
    if not isinstance(value, str):
        raise ResumeValidationError(f"{field} must be an ISO date")
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise ResumeValidationError(f"{field} must be an ISO date") from exc


@dataclass(frozen=True)
class SourceArtifact:
    """One stored input, named by URI and pinned by digest.

    The digest is mandatory. A source named only by URI makes "the same inputs" a statement
    about a path, and a path is exactly the thing that keeps pointing at new bytes.
    """

    kind: str
    uri: str
    sha256: str


@dataclass(frozen=True)
class WorkUnit:
    """One zone/window of work and the exact bytes it consumes."""

    unit_id: str
    zone_id: str
    start_date: date
    end_date: date
    output_uri: str
    sources: tuple[SourceArtifact, ...]


@dataclass(frozen=True)
class Checkpoint:
    """What a completed unit produced, and from which fingerprint.

    ``supersedes`` names the checkpoint **record** this one replaces, not its output: the
    sha256 of that record's canonical JSON (see ``record_sha256`` below), when the plan
    found the previously recorded output gone and named it. ``None`` for a unit's first
    checkpoint. Naming the record rather than the output matters because a byte-identical
    regeneration keeps the same ``output_sha256`` — naming that would let a plan already
    used to record a replacement be replayed after the replacement is itself modified
    outside the protocol, laundering that modification into a checkpoint.

    ``record_sha256`` is that same digest computed over THIS checkpoint's own record,
    carried so a later checkpoint (or a plan) can name it in its ``supersedes``.
    """

    unit_id: str
    fingerprint: str
    output_uri: str
    output_sha256: str
    completed_at: str
    record_sha256: str
    supersedes: str | None = None


def unit_fingerprint(unit: WorkUnit) -> str:
    """The identity of a unit's *inputs*, not of its request.

    Sources are sorted before hashing so the fingerprint does not depend on the order they
    were written into the manifest; the digest is then stable for the same bytes however
    the file was assembled.
    """
    body = {
        "unit_id": unit.unit_id,
        "zone_id": unit.zone_id,
        "start_date": unit.start_date.isoformat(),
        "end_date": unit.end_date.isoformat(),
        "output_uri": unit.output_uri,
        "sources": sorted(
            (
                {"kind": source.kind, "uri": source.uri, "sha256": source.sha256}
                for source in unit.sources
            ),
            key=lambda entry: (entry["uri"], entry["kind"], entry["sha256"]),
        ),
    }
    canonical = json.dumps(body, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class SourceManifest:
    """The immutable statement of what a run consumes and what it will produce."""

    run_id: str
    units: tuple[WorkUnit, ...]
    digest: str

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any], *, digest: str) -> SourceManifest:
        if not isinstance(payload, Mapping):
            raise ResumeValidationError("source manifest must be an object")
        version = payload.get("schema_version")
        if version != SOURCE_MANIFEST_SCHEMA_VERSION:
            raise ResumeValidationError(
                f"source manifest schema_version must be {SOURCE_MANIFEST_SCHEMA_VERSION!r},"
                f" got {version!r}"
            )
        run_id = _text(payload.get("run_id"), "run_id")
        raw_units = payload.get("units")
        if not isinstance(raw_units, list) or not raw_units:
            raise ResumeValidationError("source manifest must declare a non-empty units array")
        units: list[WorkUnit] = []
        seen_ids: set[str] = set()
        seen_outputs: dict[str, str] = {}
        for index, raw in enumerate(raw_units):
            unit = _work_unit(raw, index=index)
            if unit.unit_id in seen_ids:
                raise ResumeValidationError(f"duplicate unit_id: {unit.unit_id}")
            seen_ids.add(unit.unit_id)
            # Two units writing the same artifact means the second silently destroys the
            # first, and whichever one a later checkpoint describes is a coin toss.
            if unit.output_uri in seen_outputs:
                raise ResumeValidationError(
                    f"units {seen_outputs[unit.output_uri]!r} and {unit.unit_id!r} declare the"
                    f" same output_uri: {unit.output_uri}"
                )
            seen_outputs[unit.output_uri] = unit.unit_id
            units.append(unit)
        return cls(run_id=run_id, units=tuple(units), digest=_sha256(digest, "manifest digest"))


def _work_unit(raw: Any, *, index: int) -> WorkUnit:
    where = f"units[{index}]"
    if not isinstance(raw, Mapping):
        raise ResumeValidationError(f"{where} must be an object")
    missing = [
        key
        for key in ("unit_id", "zone_id", "start_date", "end_date", "output_uri", "sources")
        if key not in raw
    ]
    if missing:
        raise ResumeValidationError(f"{where} missing: {', '.join(missing)}")
    start = _iso(raw["start_date"], f"{where}.start_date")
    end = _iso(raw["end_date"], f"{where}.end_date")
    if start > end:
        raise ResumeValidationError(f"{where}.start_date must not be after end_date")
    raw_sources = raw["sources"]
    if not isinstance(raw_sources, list) or not raw_sources:
        raise ResumeValidationError(f"{where}.sources must be a non-empty array")
    sources: list[SourceArtifact] = []
    seen: set[str] = set()
    for position, entry in enumerate(raw_sources):
        source_where = f"{where}.sources[{position}]"
        if not isinstance(entry, Mapping):
            raise ResumeValidationError(f"{source_where} must be an object")
        source = SourceArtifact(
            kind=_text(entry.get("kind"), f"{source_where}.kind"),
            uri=_text(entry.get("uri"), f"{source_where}.uri"),
            sha256=_sha256(entry.get("sha256"), f"{source_where}.sha256"),
        )
        # The same URI declared twice with different digests makes the fingerprint depend
        # on which entry a reader happens to look at.
        if source.uri in seen:
            raise ResumeValidationError(f"{where} declares {source.uri} twice")
        seen.add(source.uri)
        sources.append(source)
    return WorkUnit(
        unit_id=_text(raw["unit_id"], f"{where}.unit_id"),
        zone_id=_text(raw["zone_id"], f"{where}.zone_id"),
        start_date=start,
        end_date=end,
        output_uri=_text(raw["output_uri"], f"{where}.output_uri"),
        sources=tuple(sources),
    )


def checkpoints_from_records(records: Iterable[Mapping[str, Any]]) -> dict[str, Checkpoint]:
    """Index checkpoint records by unit, refusing a file that describes one unit twice.

    Last-write-wins would make the resume decision depend on append order, which is the one
    property a crash-and-restart log cannot be assumed to have. The one exception: a second
    record whose ``supersedes`` equals the digest of the first one's own record — not its
    ``output_sha256`` — is the plan naming the checkpoint it replaces (the unit's recorded
    output had gone missing and was reproduced), so it is accepted and takes the first one's
    place. Naming the output would still match after a byte-identical regeneration, letting
    a plan already used to record a replacement be replayed once that replacement is later
    modified outside the protocol. Every other duplicate unit_id is still refused.
    """
    indexed: dict[str, Checkpoint] = {}
    for index, record in enumerate(records):
        where = f"checkpoints[{index}]"
        if not isinstance(record, Mapping):
            raise ResumeValidationError(f"{where} must be an object")
        missing = [
            key
            for key in ("unit_id", "fingerprint", "output_uri", "output_sha256", "completed_at")
            if key not in record
        ]
        if missing:
            raise ResumeValidationError(f"{where} missing: {', '.join(missing)}")
        supersedes = record.get("supersedes")
        record_sha256 = hashlib.sha256(
            json.dumps(record, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        checkpoint = Checkpoint(
            unit_id=_text(record["unit_id"], f"{where}.unit_id"),
            fingerprint=_sha256(record["fingerprint"], f"{where}.fingerprint"),
            output_uri=_text(record["output_uri"], f"{where}.output_uri"),
            output_sha256=_sha256(record["output_sha256"], f"{where}.output_sha256"),
            completed_at=_text(record["completed_at"], f"{where}.completed_at"),
            record_sha256=record_sha256,
            supersedes=_sha256(supersedes, f"{where}.supersedes")
            if supersedes is not None
            else None,
        )
        previous = indexed.get(checkpoint.unit_id)
        if previous is not None and checkpoint.supersedes != previous.record_sha256:
            raise ResumeValidationError(f"duplicate checkpoint for unit: {checkpoint.unit_id}")
        indexed[checkpoint.unit_id] = checkpoint
    return indexed


#: Resolve a URI to the sha256 of the bytes it currently holds, or ``None`` when it cannot
#: be read. ``None`` is not "absent" and not "unchanged" — it blocks the plan, because a
#: planner that cannot see an input cannot say anything true about the unit consuming it.
DigestResolver = Callable[[str], str | None]


def source_status(unit: WorkUnit, resolve: DigestResolver) -> tuple[str | None, list[str]]:
    """``(blocking status or None, notes)`` for a unit's declared inputs.

    Public because ``plan`` is not the only command that has to ask. ``record`` and
    ``verify`` reached the same conclusion from the manifest alone and so could not see a
    source re-materialized behind an unchanged manifest — `plan` would block while the
    other two reported success over the same bytes (#201 review).
    """
    unreachable: list[str] = []
    mismatched: list[str] = []
    for source in unit.sources:
        actual = resolve(source.uri)
        if actual is None:
            unreachable.append(source.uri)
        elif actual.lower() != source.sha256:
            mismatched.append(source.uri)
    if unreachable:
        return "source_unreachable", [f"cannot read {uri}" for uri in unreachable]
    if mismatched:
        return "source_mismatch", [
            f"{uri} does not match its declared digest" for uri in mismatched
        ]
    return None, []


def _unrecorded_status(unit: WorkUnit, resolve: DigestResolver) -> tuple[str, list[str]]:
    """A unit with no checkpoint. Work — unless something is already sitting at its output.

    An artifact nothing in this protocol claims is not a free slot. It is either an
    accepted result whose checkpoint log was lost, or the half-written remains of a run
    that died mid-write; both look identical from here, and executing over either one
    destroys it. The operator has to say which it is.
    """
    if resolve(unit.output_uri) is None:
        return "pending", []
    return "output_unclaimed", [
        f"{unit.output_uri} already exists and no checkpoint claims it; "
        "running this unit would overwrite it"
    ]


def _completed_status(
    unit: WorkUnit,
    checkpoint: Checkpoint,
    fingerprint: str,
    resolve: DigestResolver,
) -> tuple[str, list[str]]:
    if checkpoint.output_uri != unit.output_uri:
        # `plan` inspects the manifest's output_uri and `verify` inspects the checkpoint's.
        # While they may differ, one command can approve bytes the other never looked at,
        # so the divergence is the finding rather than something to pick a winner for.
        return "checkpoint_mismatch", [
            f"the checkpoint records {checkpoint.output_uri} but the manifest declares "
            f"{unit.output_uri}"
        ]
    if checkpoint.fingerprint != fingerprint:
        return "source_changed", [
            "the inputs changed after this unit completed; re-running would overwrite "
            f"{unit.output_uri}"
        ]
    actual = resolve(unit.output_uri)
    if actual is None:
        # Nothing to overwrite and nothing to trust: the unit is simply work again.
        return "pending", [f"recorded output {unit.output_uri} is gone; it will be produced again"]
    if actual.lower() != checkpoint.output_sha256:
        return "output_modified", [f"{unit.output_uri} was written to outside this protocol"]
    return "reuse", []


def plan_run(
    manifest: SourceManifest,
    checkpoints: Mapping[str, Checkpoint],
    *,
    resolve: DigestResolver,
    code_ref: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Decide, per unit, whether to run it, reuse it, or stop.

    Args:
        manifest: the immutable declaration of units, their stored inputs and their outputs.
        checkpoints: completed units indexed by ``unit_id``; empty on a first run.
        resolve: maps a URI to the sha256 of what it currently holds, or ``None``.
        code_ref: the producing checkout, recorded on the plan.

    Returns:
        The plan. ``runnable`` is ``False`` whenever any unit carries a blocking status —
        the whole run stops, not just that unit, because a partial panel published beside
        an unexplained conflict is the artifact this protocol exists to prevent.
    """
    units: list[dict[str, Any]] = []
    for unit in manifest.units:
        fingerprint = unit_fingerprint(unit)
        # Inputs first. A unit whose sources cannot be read says nothing useful about its
        # output, and reporting `reuse` over unreadable inputs would be the worst answer.
        status, notes = source_status(unit, resolve)
        supersedes = None
        if status is None:
            checkpoint = checkpoints.get(unit.unit_id)
            if checkpoint is None:
                status, notes = _unrecorded_status(unit, resolve)
            else:
                status, notes = _completed_status(unit, checkpoint, fingerprint, resolve)
                if status == "pending":
                    # The recorded output is gone. The plan names the checkpoint RECORD it
                    # will replace, so `record` can accept a second checkpoint for this unit
                    # without treating every other duplicate as the last-write-wins it
                    # exists to refuse — and so that a later, out-of-protocol change to a
                    # byte-identical replacement cannot be laundered by replaying this plan.
                    supersedes = checkpoint.record_sha256
        units.append(
            {
                "unit_id": unit.unit_id,
                "zone_id": unit.zone_id,
                "start_date": unit.start_date.isoformat(),
                "end_date": unit.end_date.isoformat(),
                "output_uri": unit.output_uri,
                "fingerprint": fingerprint,
                "status": status,
                "notes": notes,
                "supersedes": supersedes,
                "sources": [
                    {"kind": source.kind, "uri": source.uri, "sha256": source.sha256}
                    for source in unit.sources
                ],
            }
        )

    counts = dict.fromkeys(UNIT_STATUSES, 0)
    for entry in units:
        counts[entry["status"]] += 1
    blocked = [entry for entry in units if entry["status"] in BLOCKING_STATUSES]
    orphans = sorted(set(checkpoints) - {unit.unit_id for unit in manifest.units})
    return {
        "schema_version": RESUME_SCHEMA_VERSION,
        "run_id": manifest.run_id,
        "source_manifest_digest": manifest.digest,
        "code_ref": dict(code_ref) if code_ref is not None else None,
        "runnable": not blocked,
        "counts": counts,
        "units": units,
        "blocked": [
            {"unit_id": entry["unit_id"], "status": entry["status"], "notes": entry["notes"]}
            for entry in blocked
        ],
        # Checkpoints for units this manifest does not declare. Not an error — a manifest
        # may legitimately narrow a panel — but left unreported they are how a shrinking run
        # quietly stops accounting for artifacts it already produced.
        "orphan_checkpoints": orphans,
    }


def checkpoint_record(
    unit: WorkUnit,
    *,
    output_sha256: str,
    completed_at: str,
    code_ref: Mapping[str, Any] | None = None,
    supersedes: str | None = None,
) -> dict[str, Any]:
    """The record a caller appends after a unit's artifact is written and hashed.

    The caller passes ``completed_at`` rather than the module reading a clock: a checkpoint
    log is replayed in tests and compared byte for byte, and an embedded clock makes that
    impossible for no gain.

    ``supersedes`` names the checkpoint **record** this one replaces — the sha256 of that
    record's canonical JSON, i.e. its ``Checkpoint.record_sha256`` — not its output, carried
    over from the plan entry that authorized recording it again (#201 review).
    """
    return {
        "unit_id": unit.unit_id,
        "fingerprint": unit_fingerprint(unit),
        "output_uri": unit.output_uri,
        "output_sha256": _sha256(output_sha256, "output_sha256"),
        "completed_at": _text(completed_at, "completed_at"),
        "code_ref": dict(code_ref) if code_ref is not None else None,
        "supersedes": _sha256(supersedes, "supersedes") if supersedes is not None else None,
    }


def bind_plan(manifest: SourceManifest, plan: Mapping[str, Any]) -> None:
    """Refuse a plan that was not computed from THIS manifest.

    A plan is a file, and a file outlives the manifest it was computed from. Matching the
    two on ``unit_id`` alone — as this did before #201's review — hands back units whose
    sources, window or output path the plan never examined, which is every blocking check
    in this module bypassed by editing a manifest and reusing yesterday's plan.

    The digest binds the documents; the per-unit fingerprints bind their contents, and both
    are checked because a manifest could in principle be rewritten to the same bytes with a
    digest supplied by a careless caller.
    """
    declared = plan.get("source_manifest_digest")
    if declared != manifest.digest:
        raise ResumeValidationError(
            f"this plan was computed from manifest {declared!r}, not {manifest.digest!r}"
        )
    planned = {entry.get("unit_id"): entry.get("fingerprint") for entry in plan.get("units", [])}
    for unit in manifest.units:
        if unit.unit_id not in planned:
            raise ResumeValidationError(f"the plan does not cover unit {unit.unit_id!r}")
        if planned[unit.unit_id] != unit_fingerprint(unit):
            raise ResumeValidationError(
                f"unit {unit.unit_id!r} changed since this plan was computed"
            )


def planned_status(manifest: SourceManifest, plan: Mapping[str, Any], unit_id: str) -> str:
    """The status this plan gave a unit, after binding the plan to the manifest."""
    bind_plan(manifest, plan)
    for entry in plan.get("units", []):
        if entry.get("unit_id") == unit_id:
            return str(entry.get("status"))
    raise ResumeValidationError(f"the plan does not cover unit {unit_id!r}")


def runnable_units(manifest: SourceManifest, plan: Mapping[str, Any]) -> Sequence[WorkUnit]:
    """The units a caller may execute, in manifest order. Empty when the plan is blocked.

    A caller holding both the plan and the manifest could filter on `status == "pending"`
    itself and skip `runnable`. This exists so it does not have to remember to: a blocked
    plan hands back nothing to execute.
    """
    bind_plan(manifest, plan)
    if not plan.get("runnable"):
        return ()
    pending = {
        entry["unit_id"] for entry in plan.get("units", []) if entry.get("status") == "pending"
    }
    return tuple(unit for unit in manifest.units if unit.unit_id in pending)
