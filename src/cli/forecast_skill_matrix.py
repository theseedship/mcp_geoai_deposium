"""Produce one immutable paired MAE/P50 skill matrix from archived inputs (#193).

Reads four artifacts, digests the exact bytes it parsed, and writes the matrix defined in
:mod:`src.forecast_skill`: the forecast rows, the reference, the monthly P50, and the frozen
panel manifest that declares which pairs were expected. The panel manifest is required, not
optional: without it the denominator would be whatever rows happened to be submitted. No
provider client, no network: a current forecast endpoint cannot be used to reconstruct
historical predictions, so this command can only score an archive that already exists.

The CLI refuses unless git confirms a clean checkout of deposium_geoai itself (``dirty`` is
``false``). A dirty or unknown state needs ``--allow-dirty``, and the artifact records the
state it found. In an image without ``.git``, the CLI therefore needs ``--allow-dirty``, and
the artifact records the platform commit with ``dirty: null``.

Publication is create-only. ``--output`` and its receipt, ``<output>.receipt.json``, are
written once and never truncated or replaced: if either path already exists (a file, a
symlink, even a dangling one) or names one of the four inputs, the CLI refuses before
scoring anything. Two digests, two meanings. ``result_digest`` (inside the matrix) identifies
the scientific result and excludes ``code_ref``, so an identical result re-scored from a
later commit keeps it. The receipt's ``matrix_sha256`` is the SHA-256 of the exact bytes
published, so that later matrix has a different ``matrix_sha256`` and its own receipt. The
receipt also carries the four input SHA-256 values (the panel manifest's included), the
frozen provider, model and configuration, and ``code_ref``. The matrix is published first, the receipt second; if the
receipt cannot be written the matrix is kept, the CLI reports ``"status": "incomplete"``
and that matrix has no receipt. Any refusal is retried by publishing to a new path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import uuid
from collections.abc import Callable
from datetime import date
from pathlib import Path
from typing import Any

from src.forecast_backtest import (
    BacktestValidationError,
    digit_limit_message,
    load_rows_with_digest,
)
from src.forecast_skill import MonthlyP50Baseline, PanelManifest, paired_skill_matrix
from src.utils.code_ref import code_ref
from src.utils.logging import setup_logging

RECEIPT_SCHEMA_VERSION = "forecast-skill-matrix-receipt/v1"
RECEIPT_SUFFIX = ".receipt.json"


class PublicationIncomplete(BacktestValidationError):
    """The matrix is published but its receipt is not; the caller must be told which."""

    def __init__(self, message: str, *, matrix: Path, receipt: Path) -> None:
        super().__init__(message)
        self.matrix = matrix
        self.receipt = receipt


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="forecast-skill-matrix")
    parser.add_argument("--forecast", required=True, type=Path)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument(
        "--baseline-p50",
        required=True,
        type=Path,
        help="monthly P50 records: .json/.jsonl as {zone_id,variable,month,p50}, or .parquet",
    )
    parser.add_argument(
        "--panel-manifest",
        required=True,
        type=Path,
        help=(
            "frozen panel manifest (JSON, forecast-skill-panel/v1): zones, valid dates, "
            "variables, horizons, provider, model, configuration. Defines the expected pairs"
        ),
    )
    parser.add_argument("--provider", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--configuration",
        default="{}",
        help='JSON object frozen into the matrix identity (e.g. \'{"run": "00z"}\')',
    )
    parser.add_argument("--training-end", required=True)
    parser.add_argument("--holdout-start", required=True)
    parser.add_argument("--holdout-end", required=True)
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help=(
            f"create-only: refused if it or its receipt (<output>{RECEIPT_SUFFIX}) exists, "
            "or names an input; retry by publishing to a new path"
        ),
    )
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help=(
            "record the matrix even when git cannot confirm a clean deposium_geoai "
            "checkout (dirty or unknown)"
        ),
    )
    return parser


def _iso(value: str, field: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise BacktestValidationError(f"{field} must be an ISO date") from exc


def _parquet_records(path: Path) -> list[dict[str, Any]]:
    """Read the monthly P50 table through DuckDB.

    Every failure here is a routine input problem — no DuckDB installed, a file that is not
    Parquet, a table without these four columns — and each one has to arrive as the CLI's
    machine-readable refusal. ``main`` catches ``BacktestValidationError`` and nothing else,
    so a raw ``duckdb.Error`` would surface as a traceback and a caller parsing the output
    would see neither a result nor a refusal.

    Imported inside the branch so the JSON path — the one the hermetic fixture proof uses —
    never needs the dependency at all.
    """
    try:
        import duckdb
    except ImportError as exc:  # pragma: no cover - duckdb is a hard dependency of the repo
        raise BacktestValidationError(f"reading {path} requires duckdb: {exc}") from exc
    connection = None
    try:
        connection = duckdb.connect()
        rows = connection.execute(
            "select zone_id, variable, month, p50 from read_parquet(?)", [str(path)]
        ).fetchall()
    except duckdb.Error as exc:
        raise BacktestValidationError(f"cannot read {path} as a monthly P50 table: {exc}") from exc
    finally:
        if connection is not None:
            connection.close()
    return [
        {"zone_id": zone, "variable": variable, "month": month, "p50": p50}
        for zone, variable, month, p50 in rows
    ]


def _json_records(path: Path, text: str) -> list[dict[str, Any]]:
    """A JSON array, a ``{"rows": [...]}`` wrapper, or JSONL — one object per line.

    JSONL is not a convenience: it is what the forecast and reference archives in this repo
    are written as, and ``load_rows_with_digest`` accepts it for both of them. A baseline
    that had to be reshaped into an array first would be the only input of the three needing
    a conversion step, which is exactly the kind of asymmetry that ends in a hand-edited file.
    """
    unusable = f"{path} must contain P50 records as a JSON array, a rows wrapper, or JSONL"
    try:
        decoded = json.loads(text)
    except json.JSONDecodeError:
        records = []
        for number, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise BacktestValidationError(
                    f"invalid JSONL in {path} at line {number}: {exc.msg}"
                ) from exc
            except ValueError as exc:  # an integer longer than Python's digit limit (#221)
                raise BacktestValidationError(
                    f"invalid JSONL in {path} at line {number}: {digit_limit_message()}"
                ) from exc
        if not records:
            raise BacktestValidationError(unusable) from None
        return records
    except ValueError as exc:  # an integer longer than Python's digit limit (#221)
        raise BacktestValidationError(f"{path}: {digit_limit_message()}") from exc
    if isinstance(decoded, dict) and isinstance(decoded.get("rows"), list):
        return decoded["rows"]
    if not isinstance(decoded, list):
        raise BacktestValidationError(unusable)
    return decoded


def _baseline_records(path: Path) -> tuple[list[dict[str, Any]], str]:
    """``(records, digest)``. The digest is over the FILE, whatever the format."""
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise BacktestValidationError(f"cannot read {path}: {exc}") from exc
    digest = hashlib.sha256(raw).hexdigest()
    if path.suffix == ".parquet":
        return _parquet_records(path), digest
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise BacktestValidationError(f"{path} is not UTF-8 text") from exc
    return _json_records(path, text), digest


def _reject_duplicate_keys(what: str) -> Callable[[list[tuple[str, Any]]], dict[str, Any]]:
    """``json.loads`` keeps the last of two equal keys; a frozen identity must not."""

    def _hook(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        decoded: dict[str, Any] = {}
        for key, value in pairs:
            if key in decoded:
                raise BacktestValidationError(f"{what} has a duplicate key: {key!r}")
            decoded[key] = value
        return decoded

    return _hook


def _reject_constant(name: str) -> Any:
    raise BacktestValidationError(f"panel manifest contains a non-finite number: {name}")


def _panel_manifest(path: Path) -> tuple[PanelManifest, str]:
    """``(panel, digest)``. The digest is over the FILE's bytes, however it is formatted.

    Every failure here is a routine input problem and arrives as the CLI's refusal.
    ``BacktestValidationError`` is a ``ValueError``, so it is caught first: the generic
    ``ValueError`` below is only Python's digit limit (#221).
    """
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise BacktestValidationError(f"cannot read {path}: {exc}") from exc
    digest = hashlib.sha256(raw).hexdigest()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise BacktestValidationError(f"{path} is not UTF-8 text") from exc
    try:
        decoded = json.loads(
            text,
            object_pairs_hook=_reject_duplicate_keys("panel manifest"),
            parse_constant=_reject_constant,
        )
    except BacktestValidationError:
        raise
    except json.JSONDecodeError as exc:
        raise BacktestValidationError(f"{path} is not valid JSON: {exc.msg}") from exc
    except ValueError as exc:  # an integer longer than Python's digit limit (#221)
        raise BacktestValidationError(f"{path}: {digit_limit_message()}") from exc
    except RecursionError as exc:
        raise BacktestValidationError(f"{path} is nested too deeply") from exc
    return PanelManifest.from_record(decoded), digest


def _receipt_path(output: Path) -> Path:
    return output.with_name(output.name + RECEIPT_SUFFIX)


def _refuse_unsafe_targets(args: argparse.Namespace, receipt: Path) -> None:
    """Refuse before scoring: a target that is an input, or that already exists.

    ``os.path.lexists`` does not follow symlinks, so a dangling link counts as existing.
    This is the early, readable refusal; ``_publish_create_only`` is what makes it race-safe.
    """
    inputs = {
        path.resolve(): flag
        for flag, path in (
            ("--forecast", args.forecast),
            ("--reference", args.reference),
            ("--baseline-p50", args.baseline_p50),
            ("--panel-manifest", args.panel_manifest),
        )
    }
    for name, path in (("--output", args.output), ("the receipt path", receipt)):
        flag = inputs.get(path.resolve())
        if flag is not None:
            raise BacktestValidationError(f"{name} and {flag} name the same file")
    for path in (args.output, receipt):
        if os.path.lexists(path):
            raise BacktestValidationError(
                f"{path} already exists; a published matrix is never replaced — "
                "publish to a new --output path"
            )


def _publish_create_only(path: Path, data: bytes) -> None:
    """Publish ``data`` at ``path`` complete or not at all; ``FileExistsError`` if taken.

    The bytes go to a uniquely named temp file in the same directory, then ``os.link`` gives
    them their final name. Linking is atomic and never replaces: it fails if ANYTHING is
    at ``path`` (a file, a symlink, a dangling symlink), and it does not follow a symlink
    there. So two concurrent writers cannot both win, and a reader never sees a partial
    matrix — which ``O_CREAT | O_EXCL`` on the final path alone would allow.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    fd = os.open(temp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o666)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temp, path)
    finally:
        os.unlink(temp)


def _receipt(matrix: dict[str, Any], output: Path, data: bytes) -> dict[str, Any]:
    return {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "matrix_file": output.name,
        "matrix_sha256": hashlib.sha256(data).hexdigest(),
        "matrix_bytes": len(data),
        "result_digest": matrix["result_digest"],
        "inputs": matrix["inputs"],
        "frozen_configuration": matrix["frozen_configuration"],
        "code_ref": matrix["code_ref"],
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    try:
        configuration = json.loads(
            args.configuration, object_pairs_hook=_reject_duplicate_keys("--configuration")
        )
    except BacktestValidationError:
        raise
    except ValueError as exc:  # JSONDecodeError, or an integer past the digit limit (#221)
        raise BacktestValidationError("--configuration must be a JSON object") from exc
    if not isinstance(configuration, dict):
        raise BacktestValidationError("--configuration must be a JSON object")

    receipt_path = _receipt_path(args.output)
    _refuse_unsafe_targets(args, receipt_path)

    producing = code_ref()
    if producing.get("dirty") is not False and not args.allow_dirty:
        raise BacktestValidationError(
            f"cannot confirm a clean checkout of deposium_geoai (dirty={producing.get('dirty')!r}); "
            "pass --allow-dirty to record it anyway"
        )

    forecast, forecast_digest = load_rows_with_digest(
        args.forecast, kind="prediction", provider=args.provider
    )
    reference, reference_digest = load_rows_with_digest(args.reference, kind="reference")
    records, baseline_digest = _baseline_records(args.baseline_p50)
    panel, panel_digest = _panel_manifest(args.panel_manifest)

    matrix = paired_skill_matrix(
        forecast,
        reference,
        MonthlyP50Baseline.from_records(records),
        panel=panel,
        training_end=_iso(args.training_end, "training_end"),
        holdout_start=_iso(args.holdout_start, "holdout_start"),
        holdout_end=_iso(args.holdout_end, "holdout_end"),
        provider=args.provider,
        model=args.model,
        configuration=configuration,
        inputs={
            "forecast": forecast_digest,
            "reference": reference_digest,
            "baseline_p50": baseline_digest,
            "panel_manifest": panel_digest,
        },
        code_ref=producing,
    )
    data = (json.dumps(matrix, indent=2, sort_keys=True) + "\n").encode("utf-8")
    try:
        _publish_create_only(args.output, data)
    except FileExistsError as exc:  # a concurrent writer got there after the early check
        raise BacktestValidationError(
            f"{args.output} already exists; a published matrix is never replaced — "
            "publish to a new --output path"
        ) from exc
    except OSError as exc:
        raise BacktestValidationError(f"cannot publish {args.output}: {exc}") from exc
    receipt = _receipt(matrix, args.output, data)
    try:
        _publish_create_only(
            receipt_path, (json.dumps(receipt, indent=2, sort_keys=True) + "\n").encode("utf-8")
        )
    except OSError as exc:
        # The matrix stays: deleting a published path is a second mutation nobody asked for.
        # Its scientific content can be checked against result_digest, but without the
        # receipt there is no matrix_sha256 to verify its exact published bytes.
        raise PublicationIncomplete(
            f"matrix published at {args.output} but its receipt {receipt_path} could not be "
            f"written ({exc}); that matrix has no receipt — publish to a new --output path",
            matrix=args.output,
            receipt=receipt_path,
        ) from exc
    return matrix


def main(argv: list[str] | None = None) -> int:
    # Install the credential-redaction processor (#148) before anything can log: these
    # loaders raise with paths in the message and structlog's DEFAULT chain has no redactor.
    setup_logging()
    args = build_parser().parse_args(argv)
    try:
        matrix = run(args)
    except PublicationIncomplete as exc:
        print(
            json.dumps(
                {
                    "status": "incomplete",
                    "error": str(exc),
                    "matrix": str(exc.matrix),
                    "receipt": str(exc.receipt),
                },
                sort_keys=True,
            )
        )
        return 1
    except BacktestValidationError as exc:
        print(json.dumps({"status": "refused", "error": str(exc)}, sort_keys=True))
        return 1
    census = matrix["census"]
    print(
        json.dumps(
            {
                "status": matrix["status"],
                "result_digest": matrix["result_digest"],
                "receipt": str(_receipt_path(args.output)),
                "cells": len(matrix["cells"]),
                "expected_pairs": census["expected_pairs"],
                "submitted_forecast_rows": census["submitted_forecast_rows"],
                "eligible_pairs": census["eligible_pairs"],
                "scored_pairs": census["scored_pairs"],
                "missing_pairs": census["missing_pairs"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
