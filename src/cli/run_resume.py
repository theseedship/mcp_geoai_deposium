"""Plan, record and verify a provider-free resumable run over stored artifacts (#133).

Three subcommands, and the split between them is the protocol:

``plan``     read the immutable source manifest and the checkpoint log, hash what they
             name, and say per unit: run it, reuse it, or stop. Writes nothing but the
             plan. This is the dry run, and it is the only dry run — there is no mode in
             which planning also executes.
``record``   append one checkpoint after a caller has written and hashed an artifact.
``verify``   re-hash every recorded output and report the ones that no longer match.

The executor is NOT here. This layer decides and remembers; producing the artifact belongs
to whatever tool materializes it. That separation is what makes "must not overwrite
accepted artifacts" structural: nothing in this module opens an output for writing, and
nothing in it can reach ERA5, Open-Meteo or STAC.

URIs are resolved as local paths and ``file://`` only. An object-store resolver is a
separate change; until then an ``s3://`` source is reported ``source_unreachable`` and
blocks the plan, which is the correct answer — a planner that cannot read an input must
not pronounce on the unit consuming it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

from src.run_resume import (
    BLOCKING_STATUSES,
    Checkpoint,
    ResumeValidationError,
    SourceManifest,
    WorkUnit,
    checkpoint_record,
    checkpoints_from_records,
    plan_run,
    planned_status,
    source_status,
    unit_fingerprint,
)
from src.utils.code_ref import code_ref
from src.utils.logging import setup_logging


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="run-resume")
    subcommands = parser.add_subparsers(dest="command", required=True)

    plan = subcommands.add_parser("plan", help="decide what to run, reuse, or stop on")
    plan.add_argument("--source-manifest", required=True, type=Path)
    plan.add_argument("--checkpoints", required=True, type=Path)
    plan.add_argument("--output", type=Path, help="write the full plan here")

    record = subcommands.add_parser("record", help="append one completed unit")
    record.add_argument("--source-manifest", required=True, type=Path)
    record.add_argument("--checkpoints", required=True, type=Path)
    record.add_argument(
        "--plan",
        required=True,
        type=Path,
        help="the plan this unit was authorized by (written by `plan --output`)",
    )
    record.add_argument("--unit-id", required=True)
    record.add_argument("--completed-at", required=True, help="ISO timestamp supplied by caller")

    verify = subcommands.add_parser(
        "verify", help="re-hash every recorded output and its declared sources"
    )
    verify.add_argument("--source-manifest", required=True, type=Path)
    verify.add_argument("--checkpoints", required=True, type=Path)
    return parser


def _local_path(uri: str) -> Path | None:
    """The local path a URI names, or ``None`` when it names something else.

    Parsed, not tested with ``"://" in uri``: ``Path("s3://bucket/key")`` collapses the
    double slash to ``s3:/bucket/key``, so a guard written against the Path silently treats
    a remote URI as a relative directory instead of rejecting it.
    """
    parsed = urlparse(uri)
    # No scheme is a bare path; a one-character scheme is a Windows drive letter.
    if not parsed.scheme or len(parsed.scheme) == 1:
        return Path(uri)
    if parsed.scheme == "file" and parsed.netloc in ("", "localhost"):
        return Path(unquote(parsed.path))
    return None


def _digest_local(uri: str) -> str | None:
    path = _local_path(uri)
    if path is None:
        return None
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError:
        return None
    return digest.hexdigest()


def _protected_paths(manifest: SourceManifest) -> dict[Path, str]:
    """Every local path the manifest names, mapped to what it is.

    A control file is opened for WRITING. ``plan --output`` truncates its target and
    ``record --checkpoints`` appends to its own; pointed at a source Parquet or an accepted
    artifact, either one corrupts exactly the stored bytes this layer exists to protect,
    and no status in the protocol can report damage it caused itself (#201 review).
    """
    protected: dict[Path, str] = {}
    for unit in manifest.units:
        for path, role in (
            (_local_path(unit.output_uri), f"the output of {unit.unit_id!r}"),
            *(
                (_local_path(source.uri), f"a source of {unit.unit_id!r}")
                for source in unit.sources
            ),
        ):
            if path is not None:
                protected.setdefault(path.resolve(), role)
    return protected


def _refuse_writing_over_data(manifest: SourceManifest, **control: Path | None) -> None:
    protected = _protected_paths(manifest)
    for flag, path in control.items():
        if path is None:
            continue
        role = protected.get(path.resolve())
        if role is not None:
            raise ResumeValidationError(
                f"--{flag.replace('_', '-')} points at {role}; "
                "control files must not be written over declared data"
            )


def _refuse_colliding_control_paths(manifest_path: Path, **control: Path | None) -> None:
    """Refuse two control flags that name the same file (N1, #201 review).

    ``plan --output`` truncates its target and ``record --checkpoints`` appends to its own;
    either one aimed at another control file corrupts that file. Checked against
    ``--source-manifest`` too, since that file is also opened for reading only and would be
    silently misread as whichever control file just overwrote it.
    """
    seen: dict[Path, str] = {manifest_path.resolve(): "--source-manifest"}
    for flag, path in control.items():
        if path is None:
            continue
        name = f"--{flag.replace('_', '-')}"
        other = seen.setdefault(path.resolve(), name)
        if other != name:
            raise ResumeValidationError(f"{name} and {other} name the same file")


def _read_plan(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ResumeValidationError(f"cannot read plan {path}: {exc}") from exc
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ResumeValidationError(f"{path} must contain a JSON plan") from exc
    if not isinstance(payload, dict):
        raise ResumeValidationError(f"{path} must contain a JSON plan")
    return payload


def _read_manifest(path: Path) -> SourceManifest:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise ResumeValidationError(f"cannot read source manifest {path}: {exc}") from exc
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ResumeValidationError(f"{path} must contain a JSON source manifest") from exc
    return SourceManifest.from_mapping(payload, digest=hashlib.sha256(raw).hexdigest())


def _read_checkpoints(path: Path) -> list[dict[str, Any]]:
    """JSONL, one record per line. A missing file is an empty log, not an error.

    The first run has no checkpoints, and making the operator create an empty file before
    the protocol will talk to them is the kind of step that gets scripted as `touch` and
    then as `rm -f` when something goes wrong.
    """
    if not path.exists():
        return []
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise ResumeValidationError(f"cannot read checkpoints {path}: {exc}") from exc
    records = []
    for number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ResumeValidationError(
                f"invalid checkpoint JSONL in {path} at line {number}: {exc.msg}"
            ) from exc
    return records


def _plan(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    _refuse_colliding_control_paths(
        args.source_manifest, output=args.output, checkpoints=args.checkpoints
    )
    manifest = _read_manifest(args.source_manifest)
    _refuse_writing_over_data(manifest, output=args.output, checkpoints=args.checkpoints)
    checkpoints = checkpoints_from_records(_read_checkpoints(args.checkpoints))
    plan = plan_run(manifest, checkpoints, resolve=_digest_local, code_ref=code_ref())
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary = {
        "command": "plan",
        "run_id": plan["run_id"],
        "runnable": plan["runnable"],
        "counts": plan["counts"],
        "blocked": plan["blocked"],
        "orphan_checkpoints": plan["orphan_checkpoints"],
    }
    # Exit 2, not 1: a blocked plan is a well-formed answer the operator asked for, and a
    # caller has to be able to tell it apart from "the manifest was unreadable".
    return summary, 2 if not plan["runnable"] else 0


def _record(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    _refuse_colliding_control_paths(
        args.source_manifest, checkpoints=args.checkpoints, plan=args.plan
    )
    manifest = _read_manifest(args.source_manifest)
    _refuse_writing_over_data(manifest, checkpoints=args.checkpoints)
    unit = next((item for item in manifest.units if item.unit_id == args.unit_id), None)
    if unit is None:
        raise ResumeValidationError(f"unit {args.unit_id!r} is not declared by the manifest")

    # The plan is what AUTHORIZED this unit to run, and `record` is the only place that
    # authorization can still be checked. Without it, a unit that `plan` had blocked —
    # unreadable sources, an artifact no checkpoint claimed — could be recorded anyway, and
    # the next plan would report `reuse` over bytes nothing ever established the provenance
    # of. The block would have been laundered into a reuse (#201 review).
    plan_payload = _read_plan(args.plan)
    status = planned_status(manifest, plan_payload, unit.unit_id)
    if status != "pending":
        raise ResumeValidationError(
            f"the plan gave unit {unit.unit_id!r} the status {status!r}; "
            "only a unit the plan left pending may be recorded"
        )
    # The checkpoint this recording replaces, when the plan found the previously recorded
    # output gone (#201 review, N2): a unit whose bytes vanish must be recordable again
    # without treating that second checkpoint as the last-write-wins the log refuses.
    supersedes = next(
        (
            entry.get("supersedes")
            for entry in plan_payload.get("units", [])
            if entry.get("unit_id") == unit.unit_id
        ),
        None,
    )

    # And re-checked NOW, because the plan was computed before the unit ran and a source
    # can move in between.
    blocking, notes = source_status(unit, _digest_local)
    if blocking is not None:
        raise ResumeValidationError(
            f"cannot record {unit.unit_id!r}: {blocking} ({'; '.join(notes)})"
        )

    digest = _digest_local(unit.output_uri)
    if digest is None:
        # The artifact has to exist and be readable BEFORE its completion is recorded.
        # A checkpoint written ahead of the bytes is how a crashed run comes back claiming
        # work it never finished.
        raise ResumeValidationError(
            f"cannot hash the output of {unit.unit_id!r} at {unit.output_uri}: "
            "record a unit only after its artifact is written"
        )
    existing = checkpoints_from_records(_read_checkpoints(args.checkpoints))
    previous = existing.get(unit.unit_id)
    if previous is not None and supersedes != previous.record_sha256:
        raise ResumeValidationError(
            f"unit {unit.unit_id!r} already has a checkpoint; resolve it before recording another"
        )
    record = checkpoint_record(
        unit,
        output_sha256=digest,
        completed_at=args.completed_at,
        code_ref=code_ref(),
        supersedes=supersedes,
    )
    args.checkpoints.parent.mkdir(parents=True, exist_ok=True)
    with args.checkpoints.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, sort_keys=True) + "\n")
    return {
        "command": "record",
        "unit_id": record["unit_id"],
        "fingerprint": record["fingerprint"],
        "output_sha256": record["output_sha256"],
    }, 0


def _verify_unit(unit: WorkUnit | None, checkpoint: Checkpoint) -> str | None:
    """The reason this recorded unit no longer holds, or ``None``.

    Ordered inputs-first like the planner, and for the same reason: a unit whose sources
    cannot be read says nothing useful about its output.

    Both source checks are needed and they answer different questions. Hashing the live
    files catches a source re-materialized behind an unchanged manifest — the case that
    made the old fingerprint-only check report success (#201 review). Comparing the
    checkpoint's fingerprint catches the manifest being updated to the new digest, which
    makes the live files agree again while the recorded artifact still came from the old
    ones. Either alone leaves a way for a stale artifact to verify clean.
    """
    if unit is not None:
        blocking, _notes = source_status(unit, _digest_local)
        if blocking is not None:
            return blocking
        if checkpoint.output_uri != unit.output_uri:
            return "checkpoint_mismatch"
        if checkpoint.fingerprint != unit_fingerprint(unit):
            return "source_changed"
    actual = _digest_local(checkpoint.output_uri)
    if actual is None:
        return "output_unreadable"
    if actual != checkpoint.output_sha256:
        return "output_modified"
    return None


def _verify(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    manifest = _read_manifest(args.source_manifest)
    checkpoints = checkpoints_from_records(_read_checkpoints(args.checkpoints))
    units = {unit.unit_id: unit for unit in manifest.units}
    failures = []
    for unit_id, checkpoint in sorted(checkpoints.items()):
        reason = _verify_unit(units.get(unit_id), checkpoint)
        if reason is not None:
            failures.append({"unit_id": unit_id, "reason": reason})
    return {
        "command": "verify",
        "checkpoints": len(checkpoints),
        "verified": len(checkpoints) - len(failures),
        "failures": failures,
    }, 2 if failures else 0


_COMMANDS = {"plan": _plan, "record": _record, "verify": _verify}


def main(argv: list[str] | None = None) -> int:
    # Install the credential-redaction processor (#148): these readers raise with URIs in
    # the message and structlog's DEFAULT chain has no redactor.
    setup_logging()
    args = build_parser().parse_args(argv)
    try:
        summary, code = _COMMANDS[args.command](args)
    except ResumeValidationError as exc:
        print(json.dumps({"status": "refused", "error": str(exc)}, sort_keys=True))
        return 1
    # Three exit codes and three statuses, kept in step: 0/`ok` the protocol may proceed,
    # 2/`blocked` it answered and the answer is stop, 1/`refused` the inputs were unusable.
    # A verify that found a modified artifact printing `ok` would be the same lie the
    # protocol exists to catch.
    print(json.dumps({"status": "ok" if code == 0 else "blocked", **summary}, sort_keys=True))
    return code


if __name__ == "__main__":
    raise SystemExit(main())


# Re-exported so callers and tests name the blocking set from one place.
__all__ = ["BLOCKING_STATUSES", "build_parser", "main"]
