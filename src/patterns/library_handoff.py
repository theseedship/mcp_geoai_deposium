"""D-A export/integrity boundary; the frozen PatternLibrary contract is unchanged.

This wrapper records provenance, never grants scientific acceptance. Held-out
scorer artifacts are reference-only: their observation lags and conditional
confidence must not be copied into native so4gp exchange records.
"""

from __future__ import annotations

import hashlib
import json
import stat
import tempfile
from datetime import date
from os import scandir
from pathlib import Path
from typing import Annotated, Literal

import duckdb
from pydantic import BaseModel, ConfigDict, Field, model_validator

from .cross_validation import validated_library_to_dataframe
from .library_schema import SCHEMA_VERSION, PatternLibrary, pattern_library_json_schema
from .offline_bundle import (
    MAX_JSON_BYTES,
    MAX_LIBRARY_ROWS,
    _read_library,
    _safe_member_name,
    _strict_json,
)

Digest = Annotated[str, Field(pattern=r"^[a-f0-9]{64}$")]
Identifier = Annotated[str, Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]*$")]
NonEmpty = Annotated[str, Field(min_length=1)]
MAX_PACKAGE_BYTES = 64 * 1024 * 1024
MAX_PACKAGE_FILES = 1024
# Count directories too, so empty/deep trees cannot bypass the file cap.
MAX_PACKAGE_ENTRIES = 2 * MAX_PACKAGE_FILES


class _Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class Producer(_Contract):
    name: Literal["so4gp"]
    version: NonEmpty
    commit: Annotated[str, Field(pattern=r"^[a-f0-9]{40}$")]
    command: NonEmpty


class InputIdentity(_Contract):
    input_id: Identifier
    source_uri: NonEmpty
    sha256: Digest
    zone_id: Identifier
    period_start: date
    period_end: date

    @model_validator(mode="after")
    def _ordered_period(self):
        if self.period_end < self.period_start:
            raise ValueError("input period_end precedes period_start")
        return self


class InputSource(InputIdentity):
    path: NonEmpty


class LibrarySource(_Contract):
    path: NonEmpty
    sha256: Digest
    input_ids: list[Identifier] = Field(min_length=1)
    confidence_method: NonEmpty | None = None


class EvidenceIdentity(_Contract):
    source_uri: NonEmpty
    sha256: Digest
    relationship: Literal["reference_only"]


class EvidenceSource(EvidenceIdentity):
    path: NonEmpty


class ExportConfig(_Contract):
    schema_version: Literal["geoai-pattern-library-export/v1"]
    library_version: NonEmpty
    producer: Producer
    areas: list[Identifier] = Field(min_length=1)
    inputs: list[InputSource] = Field(min_length=1)
    libraries: list[LibrarySource] = Field(min_length=1)
    evidence: list[EvidenceSource] = Field(default_factory=list)


class PackagedEvidence(EvidenceIdentity):
    package_path: str


class LibraryEntry(_Contract):
    zone_id: Identifier
    algorithm: str
    source_sha256: Digest
    json_path: str
    parquet_path: str
    input_ids: list[Identifier] = Field(min_length=1)
    rule_ids: list[str]
    qualification: Literal["unqualified", "empty"]
    exclusions: list[str]
    confidence_method: NonEmpty | None


class AreaStatus(_Contract):
    zone_id: Identifier
    algorithm: str
    rule_count: int = Field(ge=0)
    status: Literal["unqualified", "empty", "absent"]


class FileIdentity(_Contract):
    sha256: Digest
    bytes: int = Field(ge=0, le=MAX_PACKAGE_BYTES)


class HandoffManifest(_Contract):
    schema_version: Literal["geoai-pattern-library-handoff/v1"]
    library_schema_version: Literal["1.0.0"]
    library_version: NonEmpty
    source_config_sha256: Digest
    producer: Producer
    inputs: list[InputIdentity] = Field(min_length=1)
    libraries: list[LibraryEntry] = Field(min_length=1)
    evidence: list[PackagedEvidence]
    areas: list[AreaStatus] = Field(min_length=1)
    files: dict[str, FileIdentity]


def _json_bytes(value) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _bounded_read(path: Path, limit: int) -> bytes:
    with path.open("rb") as stream:
        data = stream.read(limit + 1)
    if len(data) > limit:
        raise ValueError(f"File exceeds byte limit: {path.name}")
    return data


def _authenticated_source(base: Path, source) -> Path:
    path = base / source.path
    with path.open("rb") as stream:
        actual = hashlib.file_digest(stream, "sha256").hexdigest()
    if actual != source.sha256:
        raise ValueError(f"Source SHA-256 mismatch: {source.path}")
    return path


def _authenticated_json(base: Path, source) -> bytes:
    # Authenticate the exact bounded snapshot subsequently parsed/copied. Reopening
    # a path after hashing would let a concurrent replacement falsify provenance.
    data = _bounded_read(base / source.path, MAX_JSON_BYTES)
    if _sha(data) != source.sha256:
        raise ValueError(f"Source SHA-256 mismatch: {source.path}")
    return data


def _library(data: bytes) -> PatternLibrary:
    library = PatternLibrary.model_validate(_strict_json(data))
    if library.schema_version != SCHEMA_VERSION:
        raise ValueError(f"Unsupported library schema_version: {library.schema_version}")
    # Scope strings become path components; do not normalize or rename them.
    if not library.zone_id or any(
        c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
        for c in library.zone_id
    ):
        raise ValueError("Library zone_id is not a safe path component")
    if len(library.patterns) > MAX_LIBRARY_ROWS:
        raise ValueError("Library exceeds pattern row limit")
    return library


def _rule_ids(library: PatternLibrary) -> list[str]:
    return [f"{library.zone_id}/{library.algorithm}/{p.pattern_id}" for p in library.patterns]


def _qualification(library: PatternLibrary) -> tuple[str, list[str]]:
    if not library.patterns:
        return "empty", ["no_patterns"]
    return "unqualified", ["scientific_acceptance_not_recorded"]


def _area_status(libraries: list[PatternLibrary], areas: list[str]) -> list[AreaStatus]:
    rows = []
    for zone in areas:
        for algorithm in sorted({library.algorithm for library in libraries}):
            scoped = [lib for lib in libraries if lib.algorithm == algorithm]
            count = sum(
                1
                for lib in scoped
                for record in lib.patterns
                if zone == lib.zone_id or zone in record.zones_validated
            )
            status = (
                "unqualified"
                if count
                else "empty"
                if any(lib.zone_id == zone for lib in scoped)
                else "absent"
            )
            rows.append(
                AreaStatus(zone_id=zone, algorithm=algorithm, rule_count=count, status=status)
            )
    return rows


def _checksums(files: dict[str, FileIdentity], manifest_sha: str) -> bytes:
    hashes = {name: identity.sha256 for name, identity in files.items()}
    hashes["manifest.json"] = manifest_sha
    return "".join(f"{digest}  {name}\n" for name, digest in sorted(hashes.items())).encode("utf-8")


def export_library_handoff(config_path: str | Path, output_dir: str | Path) -> str:
    """Authenticate all sources and atomically create a new, verified package.

    Relative source paths resolve against the configuration file. Input bytes are
    verified and referenced by digest; library/evidence bytes travel in the package.
    Returns the manifest SHA-256 to convey through a trusted handover channel.
    """
    config_path, output = Path(config_path), Path(output_dir)
    if output.exists():
        raise ValueError(f"Output already exists: {output}")
    raw_config = config_path.read_bytes()
    config = ExportConfig.model_validate(_strict_json(raw_config))
    input_ids = [item.input_id for item in config.inputs]
    if len(set(input_ids)) != len(input_ids) or len(set(config.areas)) != len(config.areas):
        raise ValueError("Duplicate input identity or area")
    base = config_path.resolve().parent
    for item in config.inputs:
        _authenticated_source(base, item)
    sources = []
    for source in config.libraries:
        if not set(source.input_ids) <= set(input_ids):
            raise ValueError("Library references unknown input identity")
        library = _library(_authenticated_json(base, source))
        if any(p.confidence is not None for p in library.patterns) and not source.confidence_method:
            raise ValueError("Non-null confidence requires its actual confidence_method")
        sources.append((source, library))
    scopes = [(lib.zone_id, lib.algorithm) for _, lib in sources]
    if len(set(scopes)) != len(scopes):
        raise ValueError("Duplicate library scope")
    evidence_bytes = [(_authenticated_json(base, item), item) for item in config.evidence]

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".geoai-library-", dir=output.parent) as scratch:
        stage = Path(scratch) / "package"
        stage.mkdir()

        def write(name, data):
            path = stage / _safe_member_name(name)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)

        entries = []
        for source, library in sources:
            stem = f"libraries/{library.zone_id}/{library.algorithm}"
            write(stem + ".json", library.model_dump_json(indent=2).encode("utf-8") + b"\n")
            frame = validated_library_to_dataframe(library)
            # The existing bridge maps absent metrics to {}; retain the original null here.
            frame["extra_metrics"] = [json.dumps(p.extra_metrics) for p in library.patterns]
            with duckdb.connect(config={"threads": "1"}) as connection:
                connection.register("library_rows", frame)
                connection.execute(
                    "COPY library_rows TO ? (FORMAT PARQUET)", [str(stage / (stem + ".parquet"))]
                )
            qualification, exclusions = _qualification(library)
            entries.append(
                LibraryEntry(
                    zone_id=library.zone_id,
                    algorithm=library.algorithm,
                    source_sha256=source.sha256,
                    json_path=stem + ".json",
                    parquet_path=stem + ".parquet",
                    input_ids=source.input_ids,
                    rule_ids=_rule_ids(library),
                    qualification=qualification,
                    exclusions=exclusions,
                    confidence_method=source.confidence_method,
                )
            )
        evidence = []
        for index, (data, item) in enumerate(evidence_bytes):
            name = f"validation/reference-{index:03d}.json"
            _strict_json(data)
            write(name, data)
            evidence.append(
                PackagedEvidence(**item.model_dump(exclude={"path"}), package_path=name)
            )
        write("schemas/pattern-library-1.0.0.json", _json_bytes(pattern_library_json_schema()))
        write("schemas/handoff-v1.json", _json_bytes(HandoffManifest.model_json_schema()))
        files = {
            path.relative_to(stage).as_posix(): FileIdentity(
                sha256=_sha(path.read_bytes()), bytes=path.stat().st_size
            )
            for path in sorted(stage.rglob("*"))
            if path.is_file()
        }
        manifest = HandoffManifest(
            schema_version="geoai-pattern-library-handoff/v1",
            library_schema_version=SCHEMA_VERSION,
            library_version=config.library_version,
            source_config_sha256=_sha(raw_config),
            producer=config.producer,
            inputs=[
                InputIdentity.model_validate(item.model_dump(exclude={"path"}))
                for item in config.inputs
            ],
            libraries=entries,
            evidence=evidence,
            areas=_area_status([lib for _, lib in sources], config.areas),
            files=files,
        )
        raw_manifest = _json_bytes(manifest.model_dump(mode="json"))
        digest = _sha(raw_manifest)
        write("manifest.json", raw_manifest)
        write("SHA256SUMS", _checksums(files, digest))
        verify_library_handoff(stage, digest)
        stage.rename(output)
    return digest


def verify_library_handoff(
    directory: str | Path, expected_manifest_sha256: str
) -> tuple[HandoffManifest, list[PatternLibrary]]:
    """Verify the externally pinned manifest and every file before reading Parquet."""
    root = Path(directory)
    actual_files = {}
    pending = [root]
    visited = total_bytes = 0
    while pending:
        current = pending.pop()
        # Recheck queued directories before scanning; never traverse a symlink
        # root or Windows junction/reparse point, including a replaced directory.
        metadata = current.lstat()
        if stat.S_ISLNK(metadata.st_mode) or getattr(metadata, "st_file_attributes", 0) & (
            stat.FILE_ATTRIBUTE_REPARSE_POINT
        ):
            raise ValueError("Symlinks are not allowed in a handoff")
        with scandir(current) as entries:
            for entry in entries:
                visited += 1
                if visited > MAX_PACKAGE_ENTRIES:
                    raise ValueError("Handoff exceeds entry limit")
                metadata = entry.stat(follow_symlinks=False)
                if entry.is_symlink() or getattr(metadata, "st_file_attributes", 0) & (
                    stat.FILE_ATTRIBUTE_REPARSE_POINT
                ):
                    raise ValueError("Symlinks are not allowed in a handoff")
                path = Path(entry.path)
                if stat.S_ISDIR(metadata.st_mode):
                    pending.append(path)
                elif stat.S_ISREG(metadata.st_mode):
                    total_bytes += metadata.st_size
                    if len(actual_files) >= MAX_PACKAGE_FILES or total_bytes > MAX_PACKAGE_BYTES:
                        raise ValueError("Handoff exceeds file or byte limit")
                    actual_files[path.relative_to(root).as_posix()] = path
                else:
                    raise ValueError(f"Unsupported handoff entry: {path.name}")
    raw = _bounded_read(root / "manifest.json", MAX_JSON_BYTES)
    if _sha(raw) != expected_manifest_sha256:
        raise ValueError("Manifest SHA-256 mismatch (trusted digest required)")
    manifest = HandoffManifest.model_validate(_strict_json(raw))
    expected_files = set(manifest.files) | {"manifest.json", "SHA256SUMS"}
    if set(actual_files) != expected_files:
        raise ValueError("Missing or undeclared handoff file")
    if sum(identity.bytes for identity in manifest.files.values()) > MAX_PACKAGE_BYTES:
        raise ValueError("Manifest inventory exceeds byte limit")
    snapshots = {}
    for name, identity in manifest.files.items():
        path = root / _safe_member_name(name)
        data = _bounded_read(path, identity.bytes)
        if len(data) != identity.bytes or _sha(data) != identity.sha256:
            raise ValueError(f"File SHA-256/length mismatch: {name}")
        snapshots[name] = data
    sums = _checksums(manifest.files, expected_manifest_sha256)
    if _bounded_read(root / "SHA256SUMS", len(sums)) != sums:
        raise ValueError("SHA256SUMS does not match manifest inventory")
    inputs = [item.input_id for item in manifest.inputs]
    if len(set(inputs)) != len(inputs):
        raise ValueError("Duplicate input identity")
    libraries = []
    for entry in manifest.libraries:
        stem = f"libraries/{entry.zone_id}/{entry.algorithm}"
        if entry.json_path != stem + ".json" or entry.parquet_path != stem + ".parquet":
            raise ValueError("Library paths do not match scope")
        if not {entry.json_path, entry.parquet_path} <= set(manifest.files):
            raise ValueError("Library absent from inventory")
        library = _library(snapshots[entry.json_path])
        if (library.zone_id, library.algorithm) != (entry.zone_id, entry.algorithm):
            raise ValueError("Library scope differs from manifest")
        if entry.rule_ids != _rule_ids(library) or not set(entry.input_ids) <= set(inputs):
            raise ValueError("Rule/input identities differ from manifest")
        if (entry.qualification, entry.exclusions) != _qualification(library):
            raise ValueError("Unsubstantiated qualification metadata")
        if any(p.confidence is not None for p in library.patterns) and not entry.confidence_method:
            raise ValueError("Missing confidence method")
        # DuckDB needs a path. Give it a private copy of authenticated bytes,
        # never a public pathname that can change after the inventory check.
        with tempfile.TemporaryDirectory(prefix="geoai-library-read-") as scratch:
            path = Path(scratch) / "library.parquet"
            path.write_bytes(snapshots[entry.parquet_path])
            parquet = _read_library(path, library.algorithm)
        if parquet.patterns != library.patterns:
            raise ValueError("JSON/Parquet records differ")
        libraries.append(library)
    scopes = [(library.zone_id, library.algorithm) for library in libraries]
    if len(set(scopes)) != len(scopes):
        raise ValueError("Duplicate library scope")
    for item in manifest.evidence:
        if (
            item.package_path not in manifest.files
            or item.sha256 != manifest.files[item.package_path].sha256
        ):
            raise ValueError("Evidence identity differs from manifest inventory")
        _strict_json(snapshots[item.package_path])
    areas = list(dict.fromkeys(row.zone_id for row in manifest.areas))
    if manifest.areas != _area_status(libraries, areas):
        raise ValueError("Area qualification differs from library records")
    return manifest, libraries
