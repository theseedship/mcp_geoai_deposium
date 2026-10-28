"""Operator-only D-A projection for the closed MCP scientific package reader.

The existing handoff verifier owns package integrity and its filesystem guarantees.
Only its returned authenticated models enter this conversion; package paths are
never reopened. The original package remains the complete operator evidence.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
from pathlib import Path

from .library_handoff import HandoffManifest, verify_library_handoff
from .library_schema import PatternLibrary

MAX_PROJECTION_BYTES = 1024 * 1024
_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", re.ASCII)
_DIGEST = re.compile(r"[a-f0-9]{64}", re.ASCII)
# ECMAScript trim() includes BOM; Python strip()/\s do not. Keep the
# consumer's explicit WhiteSpace + LineTerminator set without changing values.
_ECMASCRIPT_BLANK = re.compile(
    r"[\u0009-\u000d\u0020\u00a0\u1680\u2000-\u200a\u2028\u2029\u202f\u205f\u3000\ufeff]*"
)


def _identifier(value: str) -> str:
    if not isinstance(value, str) or not _IDENTIFIER.fullmatch(value):
        raise ValueError("Invalid scientific package identifier")
    return value


def _text(value: str, maximum: int = 256, *, empty: bool = False) -> None:
    # MCP measures JavaScript UTF-16 string length, not Python code points.
    if not isinstance(value, str) or (not empty and not value):
        raise ValueError("Invalid scientific projection text")
    if len(value.encode("utf-16-le")) // 2 > maximum:
        raise ValueError("Scientific projection text exceeds limit")


def _bounded_library(library: PatternLibrary) -> dict:
    if len(library.patterns) > 10_000:
        raise ValueError("Scientific library exceeds record limit")
    _text(library.zone_id)
    value = library.model_dump(mode="json")
    if value["generated_at"] is not None:
        _text(value["generated_at"], 64)
    for record in value["patterns"]:
        for field in ("antecedent", "consequent", "pattern"):
            _text(record[field], 4096, empty=True)
        _text(record["zone_id"])
        if len(record["variables"]) > 100 or len(record["zones_validated"]) > 1000:
            raise ValueError("Scientific record exceeds collection limit")
        if any(
            not variable.strip() or _ECMASCRIPT_BLANK.fullmatch(variable)
            for variable in record["variables"]
        ):
            raise ValueError("Scientific variables must not be blank")
        for text in [*record["variables"], *record["zones_validated"]]:
            _text(text)
        if record["created_at"] is not None:
            _text(record["created_at"], 64)
        numbers = [
            record["support"],
            record["confidence"],
            record["lag_days"],
            record["stability"],
            *(record["extra_metrics"] or {}).values(),
        ]
        if any(number is not None and not math.isfinite(number) for number in numbers):
            raise ValueError("Scientific projection requires finite metrics")
        if record["pattern_id"] > 2**53 - 1 or (record["n_folds"] or 0) > 2**53 - 1:
            raise ValueError("Scientific integer cannot be represented exactly by MCP")
    return value


def _build_projection(
    manifest: HandoffManifest,
    libraries: list[PatternLibrary],
    expected_manifest_sha256: str,
    package_id: str,
    package_version: str | None,
) -> dict:
    version = _identifier(manifest.library_version)
    if package_version is not None and _identifier(package_version) != version:
        raise ValueError("Package version must match the verified handoff library_version")
    if len(libraries) > 1000 or sum(len(lib.patterns) for lib in libraries) > 10_000:
        raise ValueError("Scientific projection exceeds library or record limit")
    digests = [
        {"label": "handoff_manifest", "sha256": expected_manifest_sha256},
        {"label": "source_config", "sha256": manifest.source_config_sha256},
        *({"label": f"input:{item.input_id}", "sha256": item.sha256} for item in manifest.inputs),
        *(
            {
                "label": f"source_library:{item.zone_id}:{item.algorithm}",
                "sha256": item.source_sha256,
            }
            for item in manifest.libraries
        ),
        *(
            {"label": f"package_file:{name}", "sha256": identity.sha256}
            for name, identity in sorted(manifest.files.items())
        ),
    ]
    if not 1 <= len(digests) <= 100 or len({d["label"] for d in digests}) != len(digests):
        raise ValueError("Scientific projection exceeds digest inventory limit")
    for digest in digests:
        _text(digest["label"])
        if not _DIGEST.fullmatch(digest["sha256"]):
            raise ValueError("Invalid scientific projection digest")
    return {
        "schema_version": "geoai-scientific-projection/v1",
        "package_id": _identifier(package_id),
        "package_version": version,
        "kind": "patterns",
        "source_digests": digests,
        "provenance": {
            "method": "Offline structural projection of a verified D-A pattern-library handoff.",
            "period": {
                "start_date": min(item.period_start for item in manifest.inputs).isoformat(),
                "end_date": max(item.period_end for item in manifest.inputs).isoformat(),
            },
            "generated_at": None,
            "limitations": [
                "Non-empty D-A libraries remain unqualified; scientific acceptance is not recorded.",
                "Packaged held-out evidence is reference-only; its metrics are not transferred to native records.",
                "Raw producer extra_metrics, including conf, are not calibrated confidence.",
                "The provenance period describes source input coverage, not a record validation period.",
                "Null confidence, stability, folds and validation fields remain absent, not zero.",
                "Integrity and structure checks do not establish scientific acceptance, causality or forecasting skill.",
                "The original verified package retains complete manifest and reference evidence; projection digests bind those artifacts.",
                "Source config, input and source-library digests are declarations bound by the verified manifest; their external originals are not fetched or verified again during projection.",
            ],
        },
        "libraries": [_bounded_library(library) for library in libraries],
    }


def _publish_exclusive(output: Path, data: bytes) -> None:
    # A same-directory hard link publishes only complete bytes and refuses a
    # competing destination, including a dangling symlink. No overwrite fallback.
    descriptor, temporary = tempfile.mkstemp(prefix=".geoai-scientific-", dir=output.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, output)
    finally:
        os.unlink(temporary)


def project_library_handoff(
    directory: str | Path,
    expected_manifest_sha256: str,
    package_id: str,
    output_file: str | Path,
    package_version: str | None = None,
) -> dict:
    """Verify once and exclusively publish a deterministic, bounded MCP projection.

    This does not register the output or grant scientific acceptance. The verifier's
    existing pathname guarantees apply unchanged; only authenticated returned
    library models are serialized after it completes.
    """
    _identifier(package_id)
    if package_version is not None:
        _identifier(package_version)
    if not isinstance(expected_manifest_sha256, str) or not _DIGEST.fullmatch(
        expected_manifest_sha256
    ):
        raise ValueError("Explicit lowercase manifest SHA-256 required")
    root = Path(directory).resolve()
    requested = Path(output_file)
    output = requested.parent.resolve(strict=True) / requested.name
    if output.is_relative_to(root):
        raise ValueError("Projection output must be outside the handoff package")
    if os.path.lexists(output):
        raise ValueError("Projection output already exists")
    manifest, libraries = verify_library_handoff(directory, expected_manifest_sha256)
    projection = _build_projection(
        manifest, libraries, expected_manifest_sha256, package_id, package_version
    )
    data = (json.dumps(projection, ensure_ascii=False, allow_nan=False, indent=2) + "\n").encode(
        "utf-8"
    )
    if len(data) > MAX_PROJECTION_BYTES:
        raise ValueError("Scientific projection exceeds 1 MiB byte limit")
    _publish_exclusive(output, data)
    return {
        "output_file": str(output),
        "projection_sha256": hashlib.sha256(data).hexdigest(),
        "manifest_sha256": expected_manifest_sha256,
        "package_id": projection["package_id"],
        "package_version": projection["package_version"],
        "libraries": len(libraries),
        "patterns": sum(len(library.patterns) for library in libraries),
    }
