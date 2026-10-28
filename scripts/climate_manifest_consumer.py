#!/usr/bin/env python3
"""Consumer checks and offline bundles for climate run manifests.

The canonical handoff is a run manifest in RustFS/S3, not a local directory.
This script validates that manifest from a downstream-consumer perspective and
can export/import a zip bundle for isolated RustFS test environments.

``check`` reports ``expected_era5_rows`` / ``expected_normals_rows`` only for the artifact kinds
the manifest actually advertises; either is ``None`` when no output of that kind is listed. An
era5-only manifest therefore reports ``expected_normals_rows: None`` rather than a figure for
artifacts it never claimed.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import tempfile
import zipfile
from collections import Counter, defaultdict
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, NamedTuple

import duckdb

_REPO_ROOT = Path(__file__).resolve().parents[1]

if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.config import settings  # noqa: E402
from src.utils.dates import month_ranges  # noqa: E402
from src.utils.duckdb_conn import duckdb_session  # noqa: E402
from src.utils.logging import setup_logging  # noqa: E402
from src.utils.s3_endpoint import (  # noqa: E402
    MIN_SECRET_LENGTH,
    REDACTED,
    STAC_REFUSED,
    endpoint_credentials,
    redact_credentials,
    redact_text,
    sanitize_endpoint,
    sanitize_stac_url,
    url_secrets,
)

DEFAULT_CANONICAL_MANIFEST = (
    "s3://deposium/climate/runs/canonical-full-panel-wmo-1991-2020-20260624/manifest.json"
)
_SAFE_S3_URI_RE = re.compile(r"^s3://[A-Za-z0-9._-]+/[A-Za-z0-9._/=-]+$")
_SAFE_BUCKET_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_SAFE_ZONE_RE = re.compile(r"^[A-Za-z0-9._-]+$")
_KIND_TO_PREFIX = {
    "era5": "climate/era5",
    "normals": "climate/normals",
    "ndvi": "climate/ndvi",
    "unified": "climate/unified",
}
# Fallback only: a manifest that lists its own zone_ids decides its own zone count.
_FALLBACK_EXPECTED_ZONES = 15

# The floor `_manifest_expectations` gives a kind whose row count cannot be predicted from the
# run's parameters. It is a CONVENTION — "we expect this artifact to hold data" — not a count
# derived from anything, which is what lets `_output_row_bound` accept an advertised 0 against it.
_CONVENTIONAL_FLOOR = 1


_EXPECTED_ZONES_HELP = (
    "override the expected zone count; default is len(manifest parameters.zone_ids)"
)

_CHECK_DESCRIPTION = """\
Validate a run manifest and the artifacts it points at.

Exit codes — automation should branch on these, not on stdout:

  0  every check passed. With --json, the report is on stdout.
  1  the manifest is well-formed but checks failed. With --json, the report is on
     stdout and its "errors" list names EVERY failing artifact.
  2  the manifest is malformed, unsupported, unsafe or unreadable. ONE line goes to
     stderr and NOTHING is written to stdout, --json or not.

Exit 2 covers self-contradiction as well as bad syntax: an output advertising a row
count its own parameters cannot allow is rejected before any object is read, so a
manifest that disagrees with itself costs zero object-store round trips. The price is
that only the FIRST contradiction is named, and that a --json consumer gets empty
stdout for that input class — read the stderr line, or the exit code.

Zones are checked by IDENTITY, not by count. The report fails, naming the zones, when
a zone in parameters.zone_ids has no output, when an output names a zone the
parameters never declared, or when the same (kind, zone_id) artifact is advertised
twice — none of which a matching zone COUNT can see.
"""


class ManifestError(ValueError):
    """A manifest or bundle is malformed, unsupported or unsafe — operator input, not a crash.

    Raised only by the validation sites in this module, so ``main`` can report input problems on
    one line while an incidental ``ValueError`` from a coding mistake still surfaces its traceback.

    A message NEVER quotes a whole unvalidated block, for the reason
    :data:`~src.utils.s3_endpoint.ENDPOINT_REFUSAL` does not quote ``S3_ENDPOINT``: the commonest
    reason a value is refused is that it carries a credential where structure belongs. The import
    path makes that concrete — every one of these fires BEFORE ``_incoming_addresses`` has
    collected the origin's credentials, so no redactor can help. Measured on 6fbd599, a bundle
    whose ``environment`` was the STRING ``"https://origin-user:origin-pass@host"`` was skipped by
    ``_incoming_addresses`` for not being a dict, and ``main`` then printed::

        error: manifest environment must be a JSON object, got
               'https://origin-user:origin-pass@host'

    to stderr and into the CI log — the refusal path printing the credential the ACCEPT path
    would have removed. So a refusal over an unvalidated value names the FIELD and its TYPE,
    which is what an operator needs to fix it, and never the value. A value that has already
    passed a charset gate (``kind``, ``zone_id``, a validated ``uri``) may still be quoted: it is
    known to be a short confined token by the time it is named.
    """


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="climate_manifest_consumer",
        description=(
            "Validate a GeoAI climate run manifest, export its artifacts to a zip bundle, "
            "or import such a bundle into another RustFS/S3 bucket."
        ),
    )
    sub = p.add_subparsers(dest="command", required=True)

    check = sub.add_parser(
        "check",
        help="validate manifest + materialized outputs (exit 0 ok / 1 checks failed / 2 malformed)",
        description=_CHECK_DESCRIPTION,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    check.add_argument("--manifest-uri", default=DEFAULT_CANONICAL_MANIFEST)
    check.add_argument("--expected-zones", type=int, default=None, help=_EXPECTED_ZONES_HELP)
    check.add_argument(
        "--json",
        action="store_true",
        help="print the machine-readable report (exit 0/1 only; exit 2 prints nothing on stdout)",
    )

    export = sub.add_parser("export-zip", help="export manifest outputs to a local zip bundle")
    export.add_argument("--manifest-uri", default=DEFAULT_CANONICAL_MANIFEST)
    export.add_argument("--output", required=True, help="zip path to write")
    export.add_argument("--expected-zones", type=int, default=None, help=_EXPECTED_ZONES_HELP)

    imp = sub.add_parser("import-zip", help="import an exported bundle into S3/RustFS")
    imp.add_argument("--bundle", required=True, help="zip bundle created by export-zip")
    imp.add_argument(
        "--bucket",
        default=settings.S3_BUCKET,
        help="destination bucket; default is S3_BUCKET",
    )
    imp.add_argument(
        "--run-id",
        help="destination run id; default is the bundle manifest run_id",
    )
    imp.add_argument(
        "--manifest-uri",
        help="destination manifest URI; default is s3://{bucket}/climate/runs/{run_id}/manifest.json",
    )
    imp.add_argument("--json", action="store_true", help="print machine-readable JSON")
    return p


def _expected_days(start_date: str, end_date: str) -> int:
    return (date.fromisoformat(end_date) - date.fromisoformat(start_date)).days + 1


def _json_default(value: Any) -> str:
    return str(value)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n")


def _read_json(path: Path) -> dict[str, Any]:
    """Read one JSON object, or raise ``ManifestError``.

    Both call sites are `import_zip`, and both paths are named by an UNTRUSTED zip: `bundle.json`
    may simply be absent, and `manifest_path` is whatever the bundle wrote. Everything that can go
    wrong at those two reads is operator/bundle input, not an engine fault, so it converts HERE
    into the documented "exit 2, one stderr line" rather than widening `main`'s except tuple --
    which would also swallow the genuine bugs that tuple deliberately lets keep their traceback.

    Measured on f4f9263, `import-zip --bundle <zip> --bucket deposium`, all exit 1 with a
    traceback:

        manifest_path 'nope.json'          FileNotFoundError: [Errno 2] No such file or directory
        manifest_path '.'                  IsADirectoryError: [Errno 21] Is a directory
        manifest_path 'climate'            IsADirectoryError: [Errno 21] Is a directory
        manifest_path 'climate/z.parquet'  UnicodeDecodeError: 'utf-8' codec can't decode byte 0xff
        a zip with no bundle.json          FileNotFoundError: [Errno 2] No such file or directory

    `OSError` covers the first three (`FileNotFoundError` and `IsADirectoryError` are both
    subclasses) plus PermissionError and the rest of the errno family; `UnicodeDecodeError` covers
    a member that is not text at all. Neither is caught wholesale further out -- the scope is one
    `read_text` call on a bundle-named path.
    """
    try:
        text = path.read_text()
    except UnicodeDecodeError as exc:
        raise ManifestError(f"{path} is not UTF-8 text: {exc}") from exc
    except OSError as exc:
        raise ManifestError(f"{path} cannot be read: {exc.strerror or exc}") from exc
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ManifestError(f"{path} is not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ManifestError(f"{path} must contain a JSON object")
    return payload


def _safe_sql_literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _validate_s3_uri(uri: str) -> None:
    if not _SAFE_S3_URI_RE.fullmatch(uri) or ".." in uri:
        raise ManifestError(f"unsafe or unsupported S3 URI: {uri!r}")


def _validate_bucket(bucket: str) -> None:
    if not _SAFE_BUCKET_RE.fullmatch(bucket) or ".." in bucket:
        raise ManifestError(f"unsafe bucket name: {bucket!r}")


def _is_safe_name(value: str) -> bool:
    """Charset-confined AND an actual name, rather than a reference to a directory.

    ``_SAFE_ZONE_RE`` allows ``.``, so ``.``, ``..`` and ``...`` used to pass it while naming no
    zone and no run at all. They are not a traversal here — the value always lands as a single
    path SEGMENT, so ``..`` becomes the filename ``...parquet`` — but the S3 branch then rejected
    the assembled URI several frames later, through ``_copy_parquet``'s blanket ``".." in uri``
    test, under the message "unsafe S3 URI". That names the wrong field: the URI is malformed
    because the zone_id was, and an operator reading it goes looking at the bucket layout.
    Reject it here, where the offending field still has a name.
    """
    return bool(_SAFE_ZONE_RE.fullmatch(value)) and value.strip(".") != ""


def _validate_output_binding(uri: str, kind: str, zone_id: str) -> None:
    """The uri an entry advertises must be one its OWN ``kind`` and ``zone_id`` could produce.

    ``_validate_s3_uri`` answers "is this S3-shaped", which is a charset question, and nothing
    tied the answer back to the entry it belongs to. So a manifest could label an entry
    ``ndvi``/``A`` while its ``uri`` read ``s3://b/climate/ndvi/C.parquet``, and every gate
    passed: the URI is well-formed, and ``_zone_identity_errors`` compares
    ``parameters.zone_ids`` against the entries' own ``zone_id`` METADATA, never against where
    they point. ``check`` then read C's parquet, reported it under A's name, and ``export_zip``
    wrote those bytes to ``climate/ndvi/A.parquet`` inside the bundle — C's data republished
    under A's name, at the boundary this whole contract exists to make safe to hand over.

    The rule is stated over the two facts a legitimate producer's layout always carries, not over
    one hard-coded prefix, because a legitimate producer DOES write outside the canonical tree:

    * the last segment is ``{zone_id}.parquet``. That is the file the entry names;
    * ``kind`` appears as one of the segments above it.

    Both canonical and holdout layouts satisfy it::

        s3://{bucket}/climate/{kind}/{zone_id}.parquet                   `_default_output_uri`
        s3://{bucket}/climate/holdout/{kind}/{label}/{zone_id}.parquet   `_holdout_era5_uri`

    and the substitution does not: ``climate/ndvi/C.parquet`` under ``zone_id: A`` fails the
    filename, and ``climate/era5/A.parquet`` under ``kind: ndvi`` fails the segment. The BUCKET
    is deliberately not bound — a manifest legitimately describes objects in a store this process
    does not write to, which is the whole point of ``import-zip``.

    Both values are already charset-confined by the time this runs (``_validate_s3_uri`` and
    ``_is_safe_name``), so the message may quote them.
    """
    key = uri.partition("s3://")[2].partition("/")[2]
    segments = key.split("/")
    if segments[-1] != f"{zone_id}.parquet" or kind not in segments[:-1]:
        raise ManifestError(
            f"manifest output {kind}/{zone_id} advertises a uri its own kind and zone do not "
            f"imply: {uri!r} (expected a path with a {kind!r} segment ending in "
            f"{zone_id + '.parquet'!r})"
        )


def _destination_uri(bucket: str, kind: str, zone_id: str) -> str:
    _validate_bucket(bucket)
    # `isinstance` BEFORE the membership test. `kind` reaches here from a bundle, and a JSON
    # list or object under it makes `kind in _KIND_TO_PREFIX` raise `TypeError: unhashable
    # type: 'list'` — a traceback out of `main`, where the contract promises exit 2 and one
    # stderr line. An unhashable key is not a lookup miss; it never gets as far as being one.
    if not isinstance(kind, str) or kind not in _KIND_TO_PREFIX:
        raise ManifestError(f"unsupported artifact kind: {kind!r}")
    if not _is_safe_name(zone_id):
        raise ManifestError(f"unsafe zone_id: {zone_id!r}")
    return f"s3://{bucket}/{_KIND_TO_PREFIX[kind]}/{zone_id}.parquet"


def _manifest_destination_uri(bucket: str, run_id: str) -> str:
    _validate_bucket(bucket)
    if not _is_safe_name(run_id):
        raise ManifestError(f"unsafe run_id: {run_id!r}")
    return f"s3://{bucket}/climate/runs/{run_id}/manifest.json"


def _read_manifest(conn: duckdb.DuckDBPyConnection, uri: str) -> dict[str, Any]:
    _validate_s3_uri(uri)
    result = conn.execute("SELECT * FROM read_json_auto(?)", [uri])
    rows = result.fetchall()
    if len(rows) != 1:
        raise ManifestError(f"manifest {uri!r} must contain one JSON object, got {len(rows)}")
    columns = [d[0] for d in result.description]
    payload = dict(zip(columns, rows[0], strict=True))
    _require_run_manifest_kind(payload)
    return payload


def _require_run_manifest_kind(payload: dict[str, Any]) -> None:
    """The first structural gate every entry point applies: this document IS a run manifest.

    One function for both readers — ``_read_manifest`` (``check``/``export-zip``, reading from
    the store) and ``import_zip`` (reading from a bundle) — so the two cannot drift on what a
    manifest is. Measured on e99100c, ``import-zip`` never asked: a bundle whose manifest carried
    ``kind: "something_else"`` had its artifacts copied and a manifest published that ``check``
    then refused on this very test.
    """
    if payload.get("kind") != "geoai_run_manifest":
        raise ManifestError(
            f"manifest kind must be 'geoai_run_manifest', got {payload.get('kind')!r}"
        )


class _RowBound(NamedTuple):
    """Accepted row count for one artifact kind: ``minimum <= rows <= maximum``, both inclusive.

    ``maximum`` is None when no upper bound is defensible.

    ``zero_allowed`` says whether an entry of this kind may advertise **0** rows against the
    floor — see :func:`_output_row_bound`. It travels on the bound rather than being re-derived
    from the kind name at the point of use, because for NDVI the answer depends on the run's
    CADENCE and only `_manifest_expectations` has the parameters in hand.
    """

    minimum: int
    maximum: int | None
    zero_allowed: bool = False

    @property
    def exact(self) -> int | None:
        """The single accepted row count when the bound pins one, else None."""
        return self.minimum if self.minimum == self.maximum else None

    def accepts(self, rows: int) -> bool:
        return rows >= self.minimum and (self.maximum is None or rows <= self.maximum)

    def describe(self) -> str:
        if self.exact is not None:
            return str(self.exact)
        if self.maximum is None:
            return f"rows >= {self.minimum}"
        return f"{self.minimum} <= rows <= {self.maximum}"


def _manifest_parameters(manifest: dict[str, Any]) -> dict[str, Any]:
    parameters = manifest.get("parameters")
    if not isinstance(parameters, dict):
        raise ManifestError("manifest has no parameters object")
    return parameters


def _manifest_window(parameters: dict[str, Any]) -> tuple[str, str]:
    start_date = parameters.get("start_date")
    end_date = parameters.get("end_date")
    if isinstance(start_date, date):
        start_date = start_date.isoformat()
    if isinstance(end_date, date):
        end_date = end_date.isoformat()
    if not isinstance(start_date, str) or not isinstance(end_date, str):
        raise ManifestError("manifest parameters.start_date/end_date are required")
    parsed: dict[str, date] = {}
    for name, value in (("start_date", start_date), ("end_date", end_date)):
        try:
            parsed[name] = date.fromisoformat(value)
        except ValueError as exc:
            raise ManifestError(
                f"manifest parameters.{name} is not an ISO date: {value!r}"
            ) from exc
    # An inverted window is not a window. Every derivation below it reads as arithmetic and
    # produces a number, and not one of the numbers means anything. Measured on this consumer
    # with start 2024-12-31 / end 2024-01-01:
    #
    #   unified-only  ok: true. `_RowBound(1, None)` never touches the window at all, so the
    #                 manifest passed its check while advertising a window that runs backwards.
    #   ndvi monthly  "expected 1 <= rows <= 0". `month_ranges` walks `while cur <= end` and
    #                 returns [], so the ceiling is 0 and the bound is unsatisfiable — every
    #                 artifact fails, and the error blames the artifact for the parameters.
    #   era5          "expected -364". A negative row count, demanded of a real parquet file.
    #
    # So it is refused here, once, for every kind — including the kinds whose bounds would never
    # have consulted the window.
    if parsed["end_date"] < parsed["start_date"]:
        raise ManifestError(
            f"manifest parameters window is inverted: end_date {end_date!r} precedes "
            f"start_date {start_date!r}"
        )
    return start_date, end_date


def _month_count(start_date: str, end_date: str) -> int:
    """How many monthly slices the window holds.

    ``month_ranges`` answers for every legal ISO window, December 9999 included — 12 slices for
    ``9999-01-01..9999-12-31``. It did not always: on 7b735e6 it reached the end of a month
    through ``date(cur.year + 1, 1, 1)`` and raised ``ValueError: year must be in 1..9999, not
    10000`` at the ISO upper bound, and this function converted that into a ``ManifestError``
    to honour the exit-2 contract. 86874a7 fixed the helper at the source instead, so the
    conversion had nothing left to catch and is gone: a ``ValueError`` escaping from here is now
    a coding mistake and keeps its traceback, like one from anywhere else in this module.
    """
    return len(month_ranges(start_date, end_date))


def _manifest_expectations(manifest: dict[str, Any], kinds: set[str]) -> dict[str, _RowBound]:
    """Row-count bounds for the artifact kinds this manifest actually advertises.

    era5 and normals are exact: the requested window and variable list fully determine the row
    count. NDVI is not — Sentinel-2 coverage is opportunistic, and cloud, the STAC item cap and
    missing bands all legitimately drop scenes — so a monthly run is only bounded above by its
    month count and an all-scenes run has no defensible upper bound at all. Unified rows fall out
    of the ERA5xNDVI match and cannot be predicted from parameters either.
    """
    parameters = _manifest_parameters(manifest)
    # Every manifest declares the window it covers, whatever it materialized, so validate it once
    # and unconditionally: a normals-, unified- or all-scenes-only run must not carry missing or
    # unparseable dates past this point just because nothing downstream happens to need them.
    start_date, end_date = _manifest_window(parameters)
    bounds: dict[str, _RowBound] = {}
    if kinds & {"era5", "normals"}:
        variables = parameters.get("variables")
        if not isinstance(variables, list) or not variables:
            raise ManifestError("manifest parameters.variables must be a non-empty list")
        if "normals" in kinds:
            bounds["normals"] = _RowBound(12 * len(variables), 12 * len(variables))
    if "era5" in kinds:
        days = _expected_days(start_date, end_date)
        bounds["era5"] = _RowBound(days, days)
    if "ndvi" in kinds:
        # `ndvi_ingest` defaults `monthly` to True (src/tools/climate.py), so an omitted field
        # means monthly here too — truthy-testing it would hand a silent manifest the weakest
        # bound. `month_ranges` is the source of truth for a monthly slice: one observation per
        # month at most, so its length is the only honest upper bound.
        monthly = parameters.get("monthly", True)
        if not isinstance(monthly, bool):
            raise ManifestError(f"manifest parameters.monthly must be a boolean, got {monthly!r}")
        months = _month_count(start_date, end_date) if monthly else None
        # `zero_allowed` follows the cadence. A MONTHLY slice keeps a derived ceiling and the
        # producing panel publishes `checks.month_coverage` beside it, so "no usable scene in
        # the whole window" is a claim a consumer can still weigh against a denominator. An
        # ALL-SCENES run has neither: `monthly: false` drops the ceiling entirely, and the
        # conventional floor of 1 is then the ONLY bound the kind has. Letting the entry
        # advertise 0 against it leaves `_output_row_bound` checking nothing at all — the
        # manifest would be waiving its own last constraint, which is the widening the ceiling
        # rule exists to refuse.
        bounds["ndvi"] = _RowBound(_CONVENTIONAL_FLOOR, months, zero_allowed=monthly)
    if "unified" in kinds:
        # A unified artifact is the ERA5xNDVI INTERSECTION, and an empty intersection of two
        # non-empty inputs is an ordinary result rather than a missing artifact.
        bounds["unified"] = _RowBound(_CONVENTIONAL_FLOOR, None, zero_allowed=True)
    return bounds


# The two keys an output entry may state its own row count under: `rows` on an ndvi entry,
# `n_matched` on a unified one.
_ROW_COUNT_KEYS = ("rows", "n_matched")


def _advertised_rows(output: dict[str, Any]) -> int | None:
    """The row count an output entry claims for itself, if it claims one. Refuses a bad claim.

    ABSENT and UNUSABLE are different answers, and the old ``output.get(key)`` collapsed them.
    Absent is a manifest that simply makes no claim about this artifact — legitimate, and the
    kind bound alone decides. Unusable is a manifest that claims something and got it wrong, and
    reading that as "no claim" is a silent downgrade: measured on this consumer, an ndvi entry
    over a twelve-month window whose ``rows`` was ``"81"``, ``81.0``, ``true`` or ``null`` all
    reported ``ok: true`` against a 3-row artifact, because the narrowing that would have caught
    the truncation never happened. So presence is tested with ``in``, and every key that IS
    present is validated whether or not its value is the one returned.

    ``bool`` is refused explicitly. ``isinstance(True, int)`` is true in Python, so a JSON
    ``true`` arrives here as the integer 1 and would pin the bound to "exactly one row" — a
    number the manifest never meant and that a 1-row artifact would satisfy.

    Negative is refused here rather than downstream. It was already rejected, but by
    ``_output_row_bound`` and with the wrong diagnosis: ``rows: -5`` reported "advertises -5 rows,
    but its own parameters allow 1 <= rows <= 12", which sends an operator to look at the window.
    The window is fine. -5 is not a row count.

    TWO valid claims that DISAGREE are refused, rather than the first one winning. Every key in
    ``_ROW_COUNT_KEYS`` was already validated, but only the first found was ever returned, so
    ``{"rows": 1, "n_matched": 2}`` was accepted as a one-row claim and the second number was
    dropped on the floor. That is a manifest contradicting itself about the artifact it is
    handing over — the same defect class as an advertised count that contradicts the run's own
    window, which the bound check already refuses — and a consumer that silently picks one of the
    two numbers is choosing which of them to believe on tuple order. The same value under both
    keys stays fine: a unified entry legitimately carrying ``rows`` and ``n_matched`` agrees with
    itself.
    """
    claimed: tuple[str, int] | None = None
    for key in _ROW_COUNT_KEYS:
        if key not in output:
            continue
        value = output[key]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ManifestError(
                f"manifest output {output['kind']}/{output['zone_id']} has a {key} that is not a "
                f"non-negative integer: {value!r}"
            )
        if claimed is None:
            claimed = (key, value)
        elif value != claimed[1]:
            raise ManifestError(
                f"manifest output {output['kind']}/{output['zone_id']} advertises two row counts "
                f"that disagree: {claimed[0]}={claimed[1]} and {key}={value}"
            )
    return claimed[1] if claimed is not None else None


def _output_row_bound(output: dict[str, Any], bounds: dict[str, _RowBound]) -> _RowBound:
    """Narrow a kind bound with the row count the entry advertises, whatever the kind.

    An entry that states its own row count (``rows`` on every NDVI output, ``n_matched`` on a
    unified one) is the most specific claim the manifest makes about that artifact, so it pins the
    bound exactly: a 3-row parquet under an entry advertising 81 rows is truncated, stale or simply
    another zone's file, and the kind bound alone (1..months) would wave it through. era5/normals
    bounds are already exact, so this is a no-op there.

    The advertised count NARROWS the kind bound, it does not replace it. Replacing it let a
    manifest widen its own contract: an NDVI entry over a twelve-month window advertising 999 rows
    used to be checked against 999 and pass, where the window alone says at most 12. So the
    advertised count is checked against the kind bound first, and the exact rule is:

    * the CEILING always binds. It is derived from the run's own parameters — days in the window
      for era5, months for a monthly NDVI slice — so an advertised count above it means the
      manifest contradicts itself, which is a malformed manifest, not a wider bound.
    * the FLOOR binds too, with one exception: an advertised **0** is accepted against a
      CONVENTIONAL floor — the ``1`` that ``_manifest_expectations`` hands a kind whose row count
      cannot be predicted from the run's parameters at all. That 1 means "we expect this artifact
      to hold data", not a count derived from anything, and a unified run that legitimately
      matched zero rows must be checked against the 0 it claims rather than against a floor no
      empty result can reach. A DERIVED floor keeps binding, so an era5 entry advertising 5 rows
      for a 365-day window is still rejected.

    The exception is carried by ``bound.zero_allowed``, not inferred from ``bound.minimum == 1``.
    era5's floor is its day count, and over a one-day window that count is 1 — testing the value
    alone handed a one-day era5 entry the convention and accepted it advertising 0 rows.

    It does NOT cover an ALL-SCENES (``monthly: false``) NDVI entry, and that exclusion was
    measured rather than assumed. On 7b735e6 a `monthly: false` manifest advertising ``rows: 0``
    over an empty parquet reported ``ok: True`` with ``errors: []`` against the bound
    ``rows >= 1`` — the same bound whose positive floor an all-scenes run with NO advertised
    count is held to. `monthly: false` removes the ceiling, so that floor is the only constraint
    the kind has left, and an entry allowed to advertise 0 against it is an entry checked against
    nothing. A monthly slice keeps its derived ceiling and its `checks.month_coverage`
    denominator, so it keeps the exception.

    Raises ``ManifestError`` on a contradiction, naming both numbers — the manifest disagrees with
    its own window, and no artifact needs to be read to know it.
    """
    bound = bounds[output["kind"]]
    advertised = _advertised_rows(output)
    if advertised is None:
        return bound
    zero_against_a_conventional_floor = advertised == 0 and bound.zero_allowed
    above_ceiling = bound.maximum is not None and advertised > bound.maximum
    below_floor = advertised < bound.minimum and not zero_against_a_conventional_floor
    if above_ceiling or below_floor:
        raise ManifestError(
            f"manifest output {output['kind']}/{output['zone_id']} advertises {advertised} rows, "
            f"but its own parameters allow {bound.describe()}"
        )
    return _RowBound(advertised, advertised)


def _manifest_zone_ids(manifest: dict[str, Any]) -> list[str] | None:
    """The zone set a run DECLARED, or ``None`` when it declared none.

    ``None`` and a refusal are different answers and both are needed. A manifest that never wrote
    ``parameters.zone_ids`` — the shape the ``_FALLBACK_EXPECTED_ZONES`` count exists for — simply
    offers no identity to compare against, and the zone check downgrades to cardinality. A
    manifest that wrote something *unusable* there is refused instead of downgraded: silently
    falling back turns a typo (``zone_ids: "montpellier"``, a string, whose ``len`` is 12) into a
    weaker check, and a weaker check is exactly what a wrong value must not buy.

    Duplicates inside the declaration are refused too. ``len(zone_ids)`` is the expected COUNT, so
    ``["a", "a", "b"]`` declares three zones and names two — the count and the identity would
    disagree before a single output is looked at, and there is no reading of that which is right.
    """
    parameters = _manifest_parameters(manifest)
    if "zone_ids" not in parameters or parameters["zone_ids"] is None:
        return None
    zone_ids = parameters["zone_ids"]
    if not isinstance(zone_ids, list) or not zone_ids:
        raise ManifestError(
            f"manifest parameters.zone_ids must be a non-empty list, got {zone_ids!r}"
        )
    unusable = [zone for zone in zone_ids if not isinstance(zone, str) or not _is_safe_name(zone)]
    if unusable:
        raise ManifestError(f"manifest parameters.zone_ids has unusable entries: {unusable!r}")
    duplicated = sorted({zone for zone in zone_ids if zone_ids.count(zone) > 1})
    if duplicated:
        raise ManifestError(f"manifest parameters.zone_ids names a zone twice: {duplicated}")
    return zone_ids


def _skip_normals(parameters: dict[str, Any]) -> bool:
    """Whether the run DECLARED that it would not materialize normals.

    `scripts/climate_baseline_panel.py` writes `parameters.skip_normals: true` for its holdout
    artifact scope and then publishes era5 outputs alone. Read as a boolean the same way
    `parameters.monthly` is: a present-but-unusable value is refused rather than truthy-tested,
    since a `"false"` string read as True would quietly withdraw the pair requirement from a run
    that never asked for it.
    """
    skip_normals = parameters.get("skip_normals", False)
    if not isinstance(skip_normals, bool):
        raise ManifestError(
            f"manifest parameters.skip_normals must be a boolean, got {skip_normals!r}"
        )
    return skip_normals


def _resolve_expected_zones(
    zone_ids: list[str] | None, expected_zones: int | None
) -> tuple[int, str]:
    """Explicit CLI override wins; otherwise the run's own zone list decides."""
    if expected_zones is not None:
        return expected_zones, "cli"
    if zone_ids is not None:
        return len(zone_ids), "manifest.parameters.zone_ids"
    return _FALLBACK_EXPECTED_ZONES, "fallback"


def _zone_identity_errors(zone_ids: list[str] | None, outputs: list[dict[str, Any]]) -> list[str]:
    """Which zones the manifest and its outputs disagree about — by NAME, not by count.

    Counting was the whole check, and a count cannot see a substitution. Measured on this
    consumer before this function existed:

    * ``zone_ids: [a, b, c]`` with outputs for ``a, b, d`` — ``d`` is a zone nobody asked for and
      ``c`` was never materialized — reported ``ok: true``, three zones found and three expected.
      The run silently delivered a different set than the one it advertised;
    * ``zone_ids: [a, b, c]`` with outputs for ``a, a, b, c`` reported ``ok: true`` with four
      outputs over three zones. ``by_zone[zone][kind] = artifact`` overwrites, so the first ``a``
      artifact's verdict is dropped on the floor and the duplicate leaves no trace at all;
    * ``zone_ids: [a, b, c]`` with outputs for ``a, a, b`` did fail — but only incidentally, on
      ``found 2 zones, expected 3``, which names neither the duplicated ``a`` nor the missing
      ``c`` and sends an operator looking for the wrong thing.

    So all three are named here. The duplicate check is keyed on ``(kind, zone_id)``, not on the
    zone alone: one zone legitimately carries an era5 output AND a normals output, and it is the
    same artifact advertised twice that is the defect.

    A manifest with no declared zone list gets the duplicate check and nothing else — there is no
    identity to compare against, and inventing one from the outputs would only ever agree with
    itself.
    """
    errors: list[str] = []
    counted = Counter((output["kind"], output["zone_id"]) for output in outputs)
    duplicated = sorted(f"{kind}/{zone}" for (kind, zone), n in counted.items() if n > 1)
    if duplicated:
        errors.append(f"outputs advertise the same artifact more than once: {duplicated}")
    if zone_ids is None:
        return errors
    declared, observed = set(zone_ids), {output["zone_id"] for output in outputs}
    if missing := sorted(declared - observed):
        errors.append(f"zones in parameters.zone_ids with no output: {missing}")
    if unexpected := sorted(observed - declared):
        errors.append(f"outputs for zones absent from parameters.zone_ids: {unexpected}")
    return errors


def _outputs(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    outputs = manifest.get("outputs")
    if not isinstance(outputs, list) or not outputs:
        raise ManifestError("manifest outputs must be a non-empty list")
    for output in outputs:
        if not isinstance(output, dict):
            raise ManifestError("manifest outputs must contain JSON objects")
        kind = output.get("kind")
        # See `_destination_uri`: manifest-supplied, so isinstance-tested before the membership
        # test. Measured on 7b735e6 with `kind: ["ndvi"]`, this raised
        # `TypeError: cannot use 'list' as a dict key (unhashable type: 'list')` and dumped a
        # traceback instead of the one stderr line the exit-code contract promises.
        if not isinstance(kind, str) or kind not in _KIND_TO_PREFIX:
            raise ManifestError(f"unsupported output kind: {kind!r}")
        zone_id = output.get("zone_id")
        # `zone_id` becomes a filesystem path segment in export-zip and a destination URI on
        # import, so it gets the same charset confinement `_destination_uri` applies — one gate
        # here covers every kind.
        if not isinstance(zone_id, str) or not _is_safe_name(zone_id):
            raise ManifestError(f"manifest output has a missing or unsafe zone_id: {zone_id!r}")
        uri = output.get("uri")
        if not isinstance(uri, str):
            raise ManifestError("manifest output missing uri")
        _validate_s3_uri(uri)
        _validate_output_binding(uri, kind, zone_id)
    return outputs


def check_manifest(
    manifest_uri: str,
    *,
    expected_zones: int | None,
) -> dict[str, Any]:
    with duckdb_session(":memory:", load_httpfs=True) as conn:
        manifest = _read_manifest(conn, manifest_uri)
        checks = manifest.get("checks") if isinstance(manifest.get("checks"), dict) else {}
        outputs = _outputs(manifest)
        kinds = {output["kind"] for output in outputs}
        bounds = _manifest_expectations(manifest, kinds)
        skip_normals = _skip_normals(_manifest_parameters(manifest))
        zone_ids = _manifest_zone_ids(manifest)
        expected_zones, expected_zones_source = _resolve_expected_zones(zone_ids, expected_zones)
        by_zone: dict[str, dict[str, Any]] = defaultdict(dict)
        artifacts: list[dict[str, Any]] = []
        errors: list[str] = []

        # Resolve every bound BEFORE reading a single artifact: an advertised count that
        # contradicts the run's own window makes the manifest malformed, and that is knowable
        # without any object-store I/O at all.
        output_bounds = [_output_row_bound(output, bounds) for output in outputs]

        for output, bound in zip(outputs, output_bounds, strict=True):
            kind = output["kind"]
            zone_id = output["zone_id"]
            uri = output["uri"]
            rows = conn.execute("SELECT COUNT(*) FROM read_parquet(?)", [uri]).fetchone()[0]
            ok = bound.accepts(rows)
            if not ok:
                errors.append(f"{kind}/{zone_id} has {rows} rows, expected {bound.describe()}")
            artifact = {
                "uri": uri,
                "rows": rows,
                "expected_rows": bound.exact,
                "row_bound": bound.describe(),
                "ok": ok,
            }
            by_zone[zone_id][kind] = artifact
            artifacts.append({"kind": kind, "zone_id": zone_id, **artifact})

    if "era5" in kinds and not skip_normals:
        # Only a climate baseline run promises both halves of the pair; an NDVI- or unified-only
        # manifest never advertised era5/normals artifacts and must not fail for their absence.
        #
        # `skip_normals` is the run saying it promised only the era5 half. It is not a
        # hypothetical field: `scripts/climate_baseline_panel.py` sets it for the holdout
        # artifact scope, and measured on 7b735e6 the consumer rejected a real era5-only holdout
        # manifest that this repo's own panel produces —
        # `ok=False errors=["zones missing era5/normals pair: [\'z\']"]` — because the gate
        # fired on `"era5" in kinds` alone.
        missing_pairs = [
            zone_id
            for zone_id, zone in sorted(by_zone.items())
            if "era5" not in zone or "normals" not in zone
        ]
        if missing_pairs:
            errors.append(f"zones missing era5/normals pair: {missing_pairs}")
    # Identity BEFORE cardinality: when both fire, the names are the useful half of the report and
    # `found 2 zones, expected 3` is the derived restatement of it.
    errors.extend(_zone_identity_errors(zone_ids, outputs))
    if len(by_zone) != expected_zones:
        errors.append(f"found {len(by_zone)} zones, expected {expected_zones}")
    if checks.get("ok") is not True:
        errors.append("manifest checks.ok is not true")
    # `zone_ok` is a climate-panel field; the NDVI panel never writes one, so demand it only from
    # manifests that carry era5 outputs or that advertise the field themselves.
    if ("zone_ok" in checks or "era5" in kinds) and checks.get("zone_ok") is not True:
        errors.append("manifest checks.zone_ok is not true")

    return {
        "ok": not errors,
        "manifest_uri": manifest_uri,
        "run_id": manifest.get("run_id"),
        "kind": manifest.get("kind"),
        "checks": checks,
        "expected_zones": expected_zones,
        "expected_zones_source": expected_zones_source,
        "expected_zone_ids": sorted(zone_ids) if zone_ids is not None else None,
        "n_zones": len(by_zone),
        "n_outputs": len(outputs),
        "expected_era5_rows": bounds["era5"].exact if "era5" in bounds else None,
        "expected_normals_rows": bounds["normals"].exact if "normals" in bounds else None,
        "skip_normals": skip_normals,
        "row_bounds": {kind: bound.describe() for kind, bound in sorted(bounds.items())},
        "artifacts": sorted(artifacts, key=lambda x: (x["zone_id"], x["kind"])),
        "errors": errors,
    }


def _copy_parquet(conn: duckdb.DuckDBPyConnection, source: str, destination: str) -> None:
    if destination.startswith("s3://"):
        _validate_s3_uri(destination)
    conn.execute(
        f"COPY (SELECT * FROM read_parquet(?)) TO {_safe_sql_literal(destination)} (FORMAT PARQUET)",
        [source],
    )


def export_zip(manifest_uri: str, output: Path, *, expected_zones: int | None) -> dict[str, Any]:
    check = check_manifest(manifest_uri, expected_zones=expected_zones)
    if not check["ok"]:
        raise RuntimeError(f"manifest check failed before export: {check['errors']}")

    output = output.resolve()
    with tempfile.TemporaryDirectory(prefix="geoai-climate-bundle-") as tmp:
        root = Path(tmp)
        artifacts_meta: list[dict[str, Any]] = []
        with duckdb_session(":memory:", load_httpfs=True) as conn:
            manifest = _read_manifest(conn, manifest_uri)
            for artifact in check["artifacts"]:
                rel = Path("climate") / artifact["kind"] / f"{artifact['zone_id']}.parquet"
                local = root / rel
                local.parent.mkdir(parents=True, exist_ok=True)
                _copy_parquet(conn, artifact["uri"], local.as_posix())
                artifacts_meta.append(
                    {
                        **artifact,
                        "relative_path": rel.as_posix(),
                    }
                )

        bundle = {
            "schema_version": 1,
            "kind": "geoai_climate_bundle",
            "created_at": datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z"),
            "source_manifest_uri": manifest_uri,
            "run_id": manifest.get("run_id"),
            "manifest_path": "manifest.json",
            "artifacts": artifacts_meta,
            "copy_method": "duckdb read_parquet -> COPY FORMAT PARQUET",
            "note": (
                "This zip is an offline test bundle. The canonical source of truth remains "
                "the RustFS/S3 manifest URI."
            ),
        }
        _write_json(root / "manifest.json", manifest)
        _write_json(root / "bundle.json", bundle)
        (root / "README.txt").write_text(
            "GeoAI climate baseline offline bundle.\n"
            "Import with: uv run --extra patterns python scripts/climate_manifest_consumer.py "
            "import-zip --bundle <zip> --bucket <bucket>\n"
        )

        output.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for path in sorted(root.rglob("*")):
                if path.is_file():
                    zf.write(path, path.relative_to(root).as_posix())

    return {
        "ok": True,
        "bundle": output.as_posix(),
        "source_manifest_uri": manifest_uri,
        "run_id": check["run_id"],
        "n_outputs": check["n_outputs"],
        "n_zones": check["n_zones"],
        "size_bytes": output.stat().st_size,
    }


# The keys under `inputs[]` that hold an address a producer configured. `uris` is the unified
# panel's list of era5 parquet locations; `endpoint`/`source` are the STAC catalogue roots.
_INPUT_ADDRESS_KEYS = ("endpoint", "source", "uris")


def _incoming_addresses(manifest: dict[str, Any]) -> list[str]:
    """Every address the INCOMING manifest states, wherever a producer may have spelled one.

    A bundle's origin credentials cannot be recognised from this process's own configuration —
    but they do not have to be guessed, because the manifest names its own store. The origin's
    userinfo sits in the `environment.s3_endpoint` it shipped, and its errors quote it:

        environment.s3_endpoint  "http://origin-admin:origin-s3cret@far.away:9000"
        errors[].error           "IOException: HTTP HEAD to
                                  'http://origin-admin:origin-s3cret@far.away:9000/b/z1.parquet'"

    Measured on this consumer before this function existed, that error was republished VERBATIM
    into the destination bucket — the bucket being handed to a third party — because
    `redact_credentials` was only ever handed `settings.S3_ENDPOINT`, whose credentials are a
    different pair entirely.
    """
    addresses: list[str] = []
    for block in ("environment", "source_environment"):
        value = manifest.get(block)
        if isinstance(value, dict) and isinstance(value.get("s3_endpoint"), str):
            addresses.append(value["s3_endpoint"])
    inputs = manifest.get("inputs")
    for item in inputs if isinstance(inputs, list) else []:
        if not isinstance(item, dict):
            continue
        for key in _INPUT_ADDRESS_KEYS:
            value = item.get(key)
            if isinstance(value, str):
                addresses.append(value)
            elif isinstance(value, list):
                addresses.extend(entry for entry in value if isinstance(entry, str))
    return addresses


def _sanitized_address(value: str) -> tuple[str, tuple[str, ...]]:
    """``(the address to publish, the components removed to get there)``.

    The second half of the tuple is why this returns a tuple at all. It used to return the address
    alone, the caller stamped a flat ``{key}_credentials_removed = True`` beside it, and that
    boolean was a claim the function had no basis for: `sanitize_stac_url` drops the query from
    EVERY catalogue root it is given, so importing a manifest whose endpoint carried a
    ``?collections=sentinel-2-l2a`` provenance filter republished it as "a credential was removed
    from this address". Nothing of the sort had happened (#178). Naming the component keeps the
    two apart -- ``("userinfo",)`` is a credential, ``("query",)`` is a filter, and a reader
    decides which one they are looking at instead of being told.

    One credential-bearing address with its userinfo removed, keeping all that can be kept.

    `sanitize_stac_url` is the answer for the catalogue roots under `endpoint`/`source`, and it is
    an http(s) POLICY: it refuses every other scheme whole. An `inputs[].uris` entry is an
    `s3://bucket/key` object location, so handing it to that policy would publish `null` where the
    artifact's location belongs — a sanitizer that destroys the provenance it was added to protect.

    So the fallback punches the userinfo components out of the string instead, which leaves the
    scheme, host, bucket and key intact. The secrets come from `endpoint_credentials(value)` — the
    userinfo of THIS value — not from `configured_store_secrets`, which would also match this
    process's own access key and rewrite a bucket name that merely happens to contain it.

    Longest first, and that ordering is load-bearing rather than tidy: an account name routinely
    prefixes its own password (``deposium-dev`` / ``deposium-dev-secret``). Replacing in split
    order rewrites the short one first and publishes the tail of the long one —
    ``s3://[REDACTED]:[REDACTED]-secret@far.away/...`` — which is a leak wearing a redaction.

    The fallback is gated on the SCHEME, because `sanitize_stac_url` returns the same
    ``(None, ('refused',))`` for "this is not http(s), the policy does not apply" and for "this IS
    http(s) and it is ambiguous". Reinterpreting the second case published a wrong HOST, not
    merely a mangled one — measured before this gate:

        https://stac.internal:8443/catalogs/user@example.com/v1
        -> https://[REDACTED]:[REDACTED]@example.com/v1

    ``stac.internal:8443`` is the real host and port; the ``@`` further down the path made the
    fallback read them as a userinfo, and the published address then names ``example.com``. A
    reader would attribute the artifact to a catalogue it never came from. Where the policy owns
    the scheme and still refuses, there is no address to publish and none is invented.
    """
    address, reductions = sanitize_stac_url(value)
    if address is not None:
        return address, reductions
    if value.lower().startswith(("http://", "https://")):
        # The STAC policy owns this scheme and refused it. Saying so is the honest answer; a
        # reinterpretation would assert a host nobody configured. `STAC_REFUSED` is the same name
        # the policy itself returns and the producers already publish, so a reader does not have
        # to learn a second vocabulary for the consumer path.
        return REDACTED, (STAC_REFUSED,)
    # The same floor the publication secret set uses, and for the same reason: a one-character
    # username is a substring of ordinary text, and punching it out mangles the location this
    # fallback exists to preserve — `s3://u:pw@far.away/b/z.parquet` came out with `z.parquet`
    # rewritten to `z.parq[REDACTED]et`. A username that short is not what is being protected.
    # Rewrite each URL region independently. A query token can legitimately occur in a bucket or
    # object key too; replacing it across the whole address corrupts that provenance. Isolating the
    # regions also lets the same token be reported in BOTH query and fragment when both actually
    # carried it, instead of assigning it to whichever region was inspected first.
    head, fragment_separator, fragment_region = value.partition("#")
    location, query_separator, query_region = head.partition("?")

    def redact_region(region: str, secrets: tuple[str, ...]) -> tuple[str, bool]:
        rewritten = region
        for secret in sorted(set(secrets), key=len, reverse=True):
            if len(secret) >= MIN_SECRET_LENGTH:
                rewritten = rewritten.replace(secret, REDACTED)
        return rewritten, rewritten != region

    removed: list[str] = []

    # Restrict userinfo replacements to the region before its final delimiter, matching the
    # permissive boundary `endpoint_credentials` uses without touching the host or anything after
    # it. Longest first remains load-bearing: an account name often prefixes its password, and the
    # reverse order would publish the password tail behind a redaction marker.
    scheme, scheme_separator, remainder = location.partition("://")
    prefix = f"{scheme}://" if scheme_separator else ""
    if not scheme_separator and location.startswith("//"):
        prefix, remainder = "//", location[2:]
    elif not scheme_separator:
        remainder = location
    authority, path_separator, path = remainder.partition("/")
    userinfo_position, delimiter_width = max(
        (authority.rfind("@"), 1),
        (authority.rfind("%40"), 3),
    )
    if userinfo_position >= 0:
        userinfo = authority[:userinfo_position]
        userinfo, changed = redact_region(userinfo, endpoint_credentials(authority))
        if changed:
            removed.append("userinfo")
        authority = (
            userinfo
            + authority[userinfo_position : userinfo_position + delimiter_width]
            + authority[userinfo_position + delimiter_width :]
        )
    location = prefix + authority + (path_separator + path if path_separator else "")

    # Ask `url_secrets` about each below-authority region separately. Besides preventing a token
    # found in one region from rewriting another, this preserves its raw + percent-decoded spelling
    # policy. Only a spelling present in the original region earns a reduction.
    query_region, changed = redact_region(
        query_region,
        url_secrets(f"s3://redaction.invalid/?{query_region}"),
    )
    if changed:
        removed.append("query")
    fragment_region, changed = redact_region(
        fragment_region,
        url_secrets(f"s3://redaction.invalid/#{fragment_region}"),
    )
    if changed:
        removed.append("fragment")

    value = location
    if query_separator:
        value += query_separator + query_region
    if fragment_separator:
        value += fragment_separator + fragment_region
    # A value whose every secret sat below MIN_SECRET_LENGTH comes back unchanged and reports
    # nothing removed, because nothing was.
    return value, tuple(removed)


def _carries_a_credential(value: str) -> bool:
    """Whether one incoming address holds anything that must not be republished.

    `endpoint_credentials` reads USERINFO only, and it was the whole gate. A catalogue token below
    the authority therefore never reached the sanitizer that already knows how to reduce it:

        endpoint_credentials('https://stac.example/v1?token=abcdefgh')  -> ()
        sanitize_stac_url('https://stac.example/v1?token=abcdefgh')     -> ('https://stac.example/v1', ('query',))

    so `import-zip` republished `inputs[].endpoint` with the token intact while the diagnostic
    half of the same manifest had it punched out — the structural field and the prose field
    disagreeing about the same secret.

    The STAC policy's own reductions answer the question for http(s), and `url_secrets` covers the
    query and fragment spellings for everything else. A value with nothing to remove still returns
    False, so an address this function has no reason to touch is republished byte-for-byte.
    """
    if endpoint_credentials(value) or url_secrets(value):
        return True
    if value.lower().startswith(("http://", "https://")):
        _address, reductions = sanitize_stac_url(value)
        return bool(reductions)
    return False


# What a pre-#178 `{key}_credentials_removed: true` becomes. The old boolean cannot be mapped onto
# a component — that inability IS the defect it is being migrated out of — so it migrates to the
# only true statement left: something was removed and the importer of the day did not record what.
# Dropping it instead would publish the address as though it were the configured value verbatim.
_UNSPECIFIED_REDUCTION = "unspecified"


def _merged_reductions(existing: object, added: list[str]) -> list[str]:
    """The producer's reduction history, with what this import removed appended to it.

    Merged rather than overwritten, because the two are different events and a manifest recording
    only the later one loses the earlier. A producer publishes ``endpoint_reductions: ["query"]``
    to say "the address you are reading is a PREFIX of what was configured"; if the import then
    drops a userinfo and stamps ``["userinfo"]`` over that, the query reduction is gone from the
    record and the published address silently regains a completeness it never had.

    Unknown entries already in the field are kept verbatim. A foreign producer may name a
    component this repo has no constant for, and dropping it would be this function committing the
    very deletion it exists to prevent.

    Order is the policy's, not the alphabet's. `sanitize_stac_url` reports components in the order
    it walks the URL — userinfo, path, query, fragment — and sorting that turned
    ``["userinfo", "query"]`` into ``["query", "userinfo"]``, which reads as though the query went
    first. Nothing depends on the order, but a reader does, and it costs nothing to keep it.
    """
    known = [
        name for name in (existing if isinstance(existing, list) else []) if isinstance(name, str)
    ]
    for name in added:
        if name not in known:
            known.append(name)
    return known


def _sanitize_incoming_inputs(manifest: dict[str, Any]) -> None:
    """Strip a credential out of the STRUCTURAL address fields an incoming manifest carries.

    `environment.s3_endpoint` has `_sanitized_environment`; the `inputs[]` block had nothing, so
    an imported manifest declaring
    `inputs[0].endpoint = "https://stac-reader:stac-s3cret@stac.origin/v1"` republished that
    string, credential and all, into the destination bucket. `redact_credentials` cannot help:
    it deliberately does not touch structural fields, and an origin's catalogue token is not
    among this process's configured secrets anyway.

    Only a value that ACTUALLY carries something removable is rewritten — see
    `_carries_a_credential`, which reads the query and fragment as well as the userinfo. Every other address is republished
    exactly as written — an era5 `s3://…` URI, a bare `openmeteo_archive`, a catalogue root with
    no credential — because a sanitizer that rewrites addresses it had no reason to touch destroys
    the provenance it was added to protect. A rewritten field gains a `{key}_reductions` sibling
    NAMING the components removed — the shape the producers already publish
    (`ndvi_pilot_panel.py` `endpoint_reductions`, `unified_pilot_panel.py` `source_reductions`).

    It used to gain `{key}_credentials_removed = True` instead, and that was wrong for the most
    ordinary input there is: `sanitize_stac_url` drops the query from every catalogue root, so an
    imported `https://earth-search.aws.element84.com/v1?collections=sentinel-2-l2a` came back out
    with a flag asserting a credential had been removed from it. A `?collections=` filter is
    provenance, not a secret. The categories say which of the two happened; the boolean could not
    (#178).

    The loop reads `_INPUT_ADDRESS_KEYS` rather than a tuple of its own. It used to keep a second,
    hand-written `("endpoint", "source")`, the two copies drifted, and the drift WAS the bug:
    `_incoming_addresses` read `uris` and this function did not, so an imported manifest kept its
    userinfo verbatim in the one field whose siblings had theirs removed. `uris` is also the
    reason both shapes are handled here — it is a LIST of addresses, not a scalar.
    """
    inputs = manifest.get("inputs")
    for item in inputs if isinstance(inputs, list) else []:
        if not isinstance(item, dict):
            continue
        for key in _INPUT_ADDRESS_KEYS:
            value = item.get(key)
            # The legacy sibling is popped BEFORE any early exit, and that placement is the fix:
            # a manifest written by the pre-#178 importer already carries
            # `{key}_credentials_removed: true`, and leaving it produces either a contradiction
            # (`true` beside `["query"]`) or — for an address with nothing left to remove — the
            # original false claim surviving untouched, which is exactly what this change exists
            # to delete (Greptile P1 / Copilot, PR #187).
            legacy = bool(item.pop(f"{key}_credentials_removed", None))
            removed: list[str] = [_UNSPECIFIED_REDUCTION] if legacy else []
            if isinstance(value, str):
                if _carries_a_credential(value):
                    item[key], reductions = _sanitized_address(value)
                    removed.extend(reductions)
            elif isinstance(value, list):
                # `uris` is a LIST, and its entries reduce independently — so the categories are
                # UNIONed across them. One entry losing a userinfo while another loses a query is
                # two different reductions to the same field, and reporting either alone would
                # describe the whole list by one of its members.
                cleaned: list[Any] = []
                for entry in value:
                    if isinstance(entry, str) and _carries_a_credential(entry):
                        sanitized, reductions = _sanitized_address(entry)
                        cleaned.append(sanitized)
                        removed.extend(reductions)
                    else:
                        cleaned.append(entry)
                if cleaned != value:
                    item[key] = cleaned
            if removed:
                item[f"{key}_reductions"] = _merged_reductions(
                    item.get(f"{key}_reductions"), removed
                )


def _rewrite_manifest_uris(manifest: dict[str, Any], *, bucket: str, run_id: str) -> dict[str, Any]:
    # An incoming manifest is a bundle someone else wrote, and it is about to become an object in
    # the DESTINATION bucket. `_rewrite_environment` below sanitizes the one field that is meant to
    # hold an address; this scrubs the store password out of every OTHER field, because a producer's
    # `errors[].error` and `run_manifest_errors[].error` are verbatim exception text and both
    # DuckDB httpfs and botocore name the endpoint in full — userinfo included — in their messages.
    #
    # Redacted against the ORIGIN's credentials as well as this process's own. The origin's
    # userinfo is not a secret this process is configured with, but the manifest names its own
    # store, so the credential is recoverable from the document being imported — see
    # `_incoming_addresses`. Collected BEFORE `_sanitize_incoming_inputs` and
    # `_rewrite_environment` remove those addresses from the copy.
    rewritten = json.loads(json.dumps(manifest, default=_json_default))
    for address in _incoming_addresses(rewritten):
        rewritten = redact_credentials(rewritten, address)
    rewritten = redact_credentials(rewritten, settings.S3_ENDPOINT)
    _sanitize_incoming_inputs(rewritten)
    rewritten["run_id"] = run_id
    # Every entry is repointed at the destination, or the bundle is REFUSED. The old shape was
    # "rewrite what I recognise, copy the rest", and the rest is untrusted input: measured on
    # 7b735e6, an entry with `kind: "lidar"` and an entry with no `zone_id` at all both
    # republished `s3://evil-admin:evil-s3cret@attacker.host/x.parquet` intact into the
    # destination bucket, where the URI names neither a real object nor a safe address.
    #
    # Refusing beats sanitizing and beats dropping. There is no destination address to compute
    # for a kind this consumer cannot lay out, so a "sanitized" entry would be an entry pointing
    # nowhere; and dropping it would publish a manifest quietly shorter than the bundle it came
    # from, which is the silent-substitution failure `_zone_identity_errors` exists to catch on
    # the other side. `check` already refuses both shapes through `_outputs` — the import path
    # simply never applied the same gate.
    for key in ("outputs", "planned_outputs"):
        entries = rewritten.get(key)
        if entries is None:
            continue
        if not isinstance(entries, list):
            raise ManifestError(f"manifest {key} must be a list, got {type(entries).__name__}")
        for item in entries:
            if not isinstance(item, dict):
                raise ManifestError(
                    f"manifest {key} must contain JSON objects, got {type(item).__name__}"
                )
            kind = item.get("kind")
            # `isinstance` BEFORE the membership test: `kind` is bundle-supplied, and a JSON
            # list or object there makes `kind in _KIND_TO_PREFIX` raise `TypeError:
            # unhashable type: 'list'` — a traceback where the contract promises one line.
            if not isinstance(kind, str) or kind not in _KIND_TO_PREFIX:
                raise ManifestError(f"unsupported {key} kind: {kind!r}")
            zone_id = item.get("zone_id")
            if not isinstance(zone_id, str) or not _is_safe_name(zone_id):
                raise ManifestError(
                    f"manifest {key} entry has a missing or unsafe zone_id: {zone_id!r}"
                )
            item["uri"] = _destination_uri(bucket, kind, zone_id)
    _rewrite_environment(rewritten, bucket=bucket)
    return rewritten


# The substrings that make a KEY a credential field, whatever else the key is called. Matched
# case-insensitively against the whole key, so `s3_secret_key`, `aws_session_token`,
# `catalogue_api_key` and `stac_password` are all caught while `s3_force_path_style`,
# `s3_bucket` and a forward-compatible `unknown_future_key` are not.
_CREDENTIAL_KEY_MARKERS = (
    "secret",
    "password",
    "passwd",
    "passphrase",
    "token",
    "credential",
    "api_key",
    "apikey",
    "access_key",
    "private_key",
    "signature",
    "authorization",
)


def _names_a_credential(key: object) -> bool:
    return isinstance(key, str) and any(m in key.lower() for m in _CREDENTIAL_KEY_MARKERS)


def _can_hold_a_credential(value: object) -> bool:
    """Whether a value under a credential-NAMED key could be spelling one.

    The name is only half the question, and reading the name alone made this redactor lie about
    its own bookkeeping. Measured on 6fbd599::

        {"s3_endpoint": "http://clean.host:9000", "s3_endpoint_credentials_removed": False}
     -> {"s3_endpoint": "http://clean.host:9000", "s3_endpoint_credentials_removed": "[REDACTED]"}

    `s3_endpoint_credentials_removed` contains "credential", so the substring match caught the
    FLAG this module writes to answer "was anything removed from the recorded address". A reader
    parsing that field gets a non-empty string -- truthy -- so a CLEAN origin endpoint reports
    that credentials were removed when none were. `_sanitized_environment` then OR-ed that
    truthy string into the flag it publishes, so the lie survived into the manifest.

    A `bool` carries one bit and cannot spell a secret, so redacting one destroys information and
    protects nothing. `None` cannot spell one either, and there the redaction MANUFACTURES a
    claim: the whole reason this replaces rather than deletes is so a reader can see the origin
    shipped a credential, and `{"s3_secret_key": null}` says it shipped none.

    Everything else stays covered, including an `int`. An all-numeric password is a real shape --
    it is the same ambiguity `sanitize_stac_url` fails closed on -- so a number under a
    credential-named key is still withheld.
    """
    return value is not None and not isinstance(value, bool)


def _without_credential_fields(block: Any) -> Any:
    """``block`` with the VALUE of every credential-named key replaced by ``REDACTED``.

    An ``environment`` block arrives inside a bundle somebody else wrote, and unknown keys in it
    are carried through on purpose — the block is forward-compatible provenance and a consumer
    that drops what it does not recognise destroys exactly what the field exists to record. That
    decision is kept. What is not kept is the value under a key that says, in its own name, that
    it holds a credential: measured on this consumer before this function existed, a bundle whose
    ``environment`` carried ``"s3_secret_key": "origin-s3cret"`` had that pair copied verbatim
    into BOTH the republished ``environment`` and ``source_environment`` of the manifest written
    to ``s3://{destination}/climate/runs/{run_id}/manifest.json`` — the object being handed to
    another party.

    `redact_credentials` cannot cover this: the origin's key is not one of this process's own
    secrets, and ``s3_secret_key`` is a structural field rather than one of `DIAGNOSTIC_FIELDS`.

    The key survives with a ``REDACTED`` value rather than being deleted, so a reader can still
    see that the origin shipped one. Nested objects are walked, because a block is free to spell
    its store as ``{"store": {"secret_key": …}}``.

    The bound is honest and worth stating: this reads NAMES. A credential an origin hides in a
    field called ``operator_note`` is not recognisable to any consumer and is not covered.

    The name is necessary and not sufficient — see :func:`_can_hold_a_credential` for the values a
    credential-named key can hold without holding a credential.
    """
    if isinstance(block, dict):
        return {
            key: REDACTED
            if _names_a_credential(key) and _can_hold_a_credential(value)
            else _without_credential_fields(value)
            for key, value in block.items()
        }
    if isinstance(block, list):
        return [_without_credential_fields(item) for item in block]
    return block


def _sanitized_environment(environment: dict[str, Any]) -> dict[str, Any]:
    """A copy of ``environment`` whose ``s3_endpoint`` carries no userinfo.

    Applied to the block that ARRIVES in a bundle as well as to the one this import writes: a
    manifest handed over by someone else may already carry ``http://user:pass@host`` in its
    environment, and ``import-zip`` republishes it verbatim under ``source_environment``. Incoming
    data is not trusted to have been sanitized at its source.

    ``s3_endpoint_credentials_removed`` is OR-ed rather than replaced. The upstream flag records
    that a removal happened at the producer; this one records that a removal happened here. Either
    is enough to answer the question the field exists for — "is the recorded address shorter than
    the configured one" — so a ``True`` never degrades to ``False`` on the way through.
    """
    environment = _without_credential_fields(environment)
    if "s3_endpoint" not in environment:
        # Do not invent the key. An environment block that never named an endpoint is not made
        # more truthful by acquiring an explicit "no endpoint" it never claimed.
        return dict(environment)
    endpoint, credentials_removed = sanitize_endpoint(environment["s3_endpoint"])
    return {
        **environment,
        "s3_endpoint": endpoint,
        "s3_endpoint_credentials_removed": (
            bool(environment.get("s3_endpoint_credentials_removed")) or credentials_removed
        ),
    }


def _rewrite_environment(manifest: dict[str, Any], *, bucket: str) -> None:
    """Point an ``environment`` block at the store this import actually wrote to.

    A run manifest records which object store holds its objects (endpoint, bucket, addressing
    style). Import moves those objects to another store, so carrying the block over unchanged would
    leave the imported manifest naming the source store — actively lying about where its data
    lives, which is the one question the block exists to answer. Unknown keys are carried through.
    A no-op when the key is absent.

    The endpoint written here is passed through ``sanitize_endpoint`` first. ``S3_ENDPOINT`` may
    carry ``user:password@`` in its userinfo, and ``import_zip`` publishes this manifest to
    ``s3://{bucket}/climate/runs/{run_id}/manifest.json`` — a durable object in the DESTINATION
    bucket, which is precisely the bucket being handed to another party. Writing the raw setting
    there put the store's password into the handoff artifact.

    ``source_environment`` records the ORIGIN store and is never clobbered. On a re-import
    (import -> export -> import) the incoming manifest's ``source_environment`` already names the
    store the data was first produced in; overwriting it with the previous relay's environment
    would silently truncate the chain to its most recent hop. Keeping the origin and counting the
    hops in ``source_environment_hops`` preserves the answer to "where did this data come from"
    while keeping the field a plain object — accumulating a list instead would change the field's
    type on the second import and break every consumer that reads it as one.

    An ABSENT ``environment`` is a no-op only for the ``environment`` half. It used to be a no-op
    for the whole function — ``if environment is None: return`` sat above everything — which made
    it a hole straight through the sanitizing this function exists to do: a bundle carrying only
    ``source_environment`` never reached ``_sanitized_environment`` at all. Measured on 7b735e6,
    with ``_without_credential_fields`` already in place::

        source_environment published -> {"s3_endpoint":
            "http://origin-admin:origin-s3cret@far.away:9000", "s3_secret_key": "origin-s3cret"}

    into ``s3://{destination}/climate/runs/{run_id}/manifest.json``. The block a bundle arrives
    with is untrusted whether or not a sibling block came with it, so the ``source_environment``
    half now runs unconditionally and the early return covers only the write below it.
    """
    environment = manifest.get("environment")
    if environment is not None and not isinstance(environment, dict):
        # Not an environment block. Refuse rather than republish an uninterpretable blob that
        # cannot be checked for credentials.
        raise ManifestError(
            f"manifest environment must be a JSON object, got {type(environment).__name__}"
        )
    source = manifest.get("source_environment")
    if source is None:
        # Nothing to promote when this manifest carried no environment either: inventing an
        # empty origin block would state a provenance the bundle never claimed.
        if environment is not None:
            manifest["source_environment"] = _sanitized_environment(environment)
            manifest["source_environment_hops"] = 1
    elif isinstance(source, dict):
        # A re-import. Keep the origin, but re-sanitize it: it arrived in a bundle written by
        # someone else and is not trusted to be credential-free.
        manifest["source_environment"] = _sanitized_environment(source)
        hops = manifest.get("source_environment_hops")
        counted = hops if isinstance(hops, int) and not isinstance(hops, bool) else 1
        manifest["source_environment_hops"] = counted + 1
    else:
        raise ManifestError(
            f"manifest source_environment must be a JSON object, got {type(source).__name__}"
        )
    if environment is None:
        return
    endpoint, credentials_removed = sanitize_endpoint(settings.S3_ENDPOINT)
    manifest["environment"] = {
        **_without_credential_fields(environment),
        "s3_endpoint": endpoint,
        "s3_endpoint_credentials_removed": credentials_removed,
        "s3_bucket": bucket,
        "s3_force_path_style": settings.S3_FORCE_PATH_STYLE,
    }


def _confined_member(root: Path, value: object, field: str) -> Path:
    r"""Resolve one BUNDLE-supplied relative path inside ``root``, or raise ManifestError.

    Both `manifest_path` and every `artifacts[].relative_path` come out of an untrusted zip, and
    the extraction loop's guard protects EXTRACTION only — it says nothing about a path the bundle
    names afterwards. `pathlib` collapses `root / "/etc/passwd"` to the absolute path, and `..`
    segments walk out one level at a time.

    Two shapes have to be refused BEFORE `resolve()` rather than after, because `resolve()` itself
    raises and does so one line above the containment check, which is how they escaped as exit 1
    with a traceback:

        "z\x00.parquet"    ValueError: lstat: embedded null character in path
        "\ud800.parquet"   UnicodeEncodeError: surrogates not allowed

    Neither opens a file, so no `read_text` guard can catch them.
    """
    if not isinstance(value, str) or not value:
        raise ManifestError(f"bundle {field} must be a non-empty string: {value!r}")
    if "\x00" in value:
        raise ManifestError(f"bundle {field} contains a null byte: {value!r}")
    try:
        value.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise ManifestError(f"bundle {field} is not encodable UTF-8: {value!r}") from exc
    resolved = (root / value).resolve()
    if not resolved.is_relative_to(root):
        raise ManifestError(f"bundle {field} escapes bundle root: {value!r}")
    return resolved


def _refuse_an_inventory_mismatch(
    outputs: list[dict[str, Any]],
    planned: list[tuple[str, str, Path, str]],
) -> None:
    """The artifacts a bundle SHIPS and the artifacts its manifest ADVERTISES must be the same set.

    Two lists describe the same handover and nothing compared them. ``bundle.artifacts`` drives
    what gets copied; ``manifest.outputs`` is what the republished manifest advertises, and
    ``_rewrite_manifest_uris`` repoints each entry at ``_destination_uri(bucket, kind, zone_id)``
    computed from the MANIFEST's own metadata. So a bundle shipping artifacts for A and C while
    its manifest advertised A and B imported A's and C's parquet, then published a manifest whose
    entries named A and B:

    * ``climate/{kind}/C.parquet`` lands in the handoff bucket with nothing vouching for it;
    * ``climate/{kind}/B.parquet`` is advertised and was never written — so the entry either
      resolves to nothing, or, worse, to a STALE object left there by an earlier run, which the
      freshly published manifest now vouches for as part of this one.

    A bundle this repo produced can never trip this: ``export_zip`` builds ``artifacts`` from
    ``check["artifacts"]``, which is built from ``outputs``, so the two agree by construction.
    That is what makes equality the right relation rather than containment — a mismatch means the
    bundle was assembled somewhere else, or edited after it was.

    Counted rather than set-compared, so a duplicated entry on one side and not the other is a
    difference too. ``planned_outputs`` is deliberately NOT compared: it records what the run
    INTENDED to write, and an artifact that was planned and not produced is exactly the shape a
    partial run legitimately hands over.

    Both halves are charset-confined before this runs — ``kind`` against ``_KIND_TO_PREFIX`` and
    ``zone_id`` against ``_is_safe_name``, on both sides — so the message may name them.
    ``outputs`` is the list ``_outputs`` returned, so it is a non-empty list of objects: a
    manifest with no ``outputs`` at all never reaches this comparison.
    """
    advertised = Counter((output["kind"], output["zone_id"]) for output in outputs)
    shipped = Counter((kind, zone_id) for kind, zone_id, _source, _uri in planned)
    if advertised == shipped:
        return
    missing = sorted(f"{k}/{z}" for k, z in (advertised - shipped).elements())
    extra = sorted(f"{k}/{z}" for k, z in (shipped - advertised).elements())
    raise ManifestError(
        "bundle artifacts and manifest outputs describe different artifact sets: "
        f"advertised but not shipped {missing}, shipped but not advertised {extra}"
    )


def import_zip(
    bundle_path: Path,
    *,
    bucket: str,
    run_id: str | None,
    manifest_uri: str | None,
) -> dict[str, Any]:
    if not bucket:
        raise ManifestError("--bucket is required, or set S3_BUCKET")
    bundle_path = bundle_path.resolve()
    with tempfile.TemporaryDirectory(prefix="geoai-climate-import-") as tmp:
        # resolve() both sides of the containment guards below: on macOS the temp dir is
        # `/var/folders/...`, a symlink to `/private/var/folders/...`, so an unresolved root
        # never contains a resolved member and every extraction would be rejected as an escape.
        root = Path(tmp).resolve()
        # Converted AT THE SITE, the way the other input errors in this module are. This open is
        # the ONE unconverted `except` escape left in `import-zip`: `main`'s tuple catches
        # ManifestError, RuntimeError and three duckdb types, and none of the three ways this
        # line fails is among them. Measured on 6fbd599 with a bundle path that does not exist,
        # `import-zip` exited 1 with a traceback where the contract promises exit 2 and one
        # stderr line:
        #
        #     FileNotFoundError  the bundle is missing, or is a directory
        #     BadZipFile         the bytes are not a zip (a truncated download, an HTML error
        #                        page saved under the name)
        #     OSError            everything else the filesystem refuses
        #
        # Converted here rather than by widening `main`'s tuple: `OSError` in that tuple would
        # swallow every unrelated filesystem bug in the module and turn it into a one-line
        # "operator input" report. A bundle that cannot be opened IS operator input; an OSError
        # from somewhere else is not, and only this site can tell the two apart.
        try:
            archive = zipfile.ZipFile(bundle_path)
        except zipfile.BadZipFile as exc:
            raise ManifestError(f"bundle is not a readable zip: {bundle_path.as_posix()}") from exc
        except OSError as exc:
            # FileNotFoundError, IsADirectoryError and PermissionError are all OSError.
            raise ManifestError(
                f"bundle cannot be opened: {bundle_path.as_posix()} ({type(exc).__name__})"
            ) from exc
        with archive as zf:
            for member in zf.infolist():
                target = (root / member.filename).resolve()
                if not target.is_relative_to(root):
                    raise ManifestError(f"zip member escapes bundle root: {member.filename!r}")
                zf.extract(member, root)
        bundle = _read_json(root / "bundle.json")
        # `manifest_path` is BUNDLE-supplied, so it gets the containment guard the extraction
        # loop above gets. That loop protects EXTRACTION and nothing else: measured on 7b735e6,
        # an absolute `manifest_path` made `root / path` collapse to the absolute path — that is
        # what `pathlib` does with an absolute right-hand side — and `_read_json` opened a file
        # outside the extraction root, an arbitrary local JSON read driven by an untrusted zip.
        # `..` segments do the same thing one level at a time.
        manifest_file = _confined_member(
            root, bundle.get("manifest_path", "manifest.json"), "manifest_path"
        )
        manifest = _read_json(manifest_file)
        _require_run_manifest_kind(manifest)
        final_run_id = run_id or manifest.get("run_id")
        if not isinstance(final_run_id, str) or not final_run_id:
            raise ManifestError("missing run_id; pass --run-id")
        final_manifest_uri = manifest_uri or _manifest_destination_uri(bucket, final_run_id)
        _validate_s3_uri(final_manifest_uri)

        artifacts = bundle.get("artifacts")
        if not isinstance(artifacts, list) or not artifacts:
            raise ManifestError("bundle has no artifacts list")

        # EVERY check the bundle and its manifest have to pass runs before the FIRST object is
        # written. It used to interleave: the loop below validated one artifact, copied it, and
        # only then looked at the next — and the manifest's own entries were not validated at all
        # until `_rewrite_manifest_uris`, which ran after the loop had finished. Measured on
        # 6fbd599, a bundle with a valid first artifact and an invalid second one (an unsupported
        # `kind`, an unsafe `zone_id`, a `relative_path` that escapes the root) copied the first
        # parquet into the destination bucket and THEN raised, leaving a partial import behind:
        # an object in the handoff bucket with no manifest vouching for it, under a name a later
        # run would be expected to overwrite.
        #
        # The honest bound: this closes the VALIDATION half. `_copy_parquet` can still fail on
        # its third object for a reason no amount of pre-checking can see — the network, a full
        # store — so import-zip is not transactional and does not claim to be. What it no longer
        # does is write anything at all for a bundle it was always going to refuse.
        planned: list[tuple[str, str, Path, str]] = []
        for artifact in artifacts:
            if not isinstance(artifact, dict):
                raise ManifestError("bundle artifact must be an object")
            kind = artifact.get("kind")
            zone_id = artifact.get("zone_id")
            rel = artifact.get("relative_path")
            if not isinstance(kind, str) or not isinstance(zone_id, str):
                raise ManifestError(
                    "bundle artifact entry needs a string kind and zone_id, got "
                    f"{type(kind).__name__} and {type(zone_id).__name__}"
                )
            source = _confined_member(root, rel, "artifacts[].relative_path")
            if not source.is_file():
                raise ManifestError(f"bundle artifact path does not exist: {rel!r}")
            planned.append((kind, zone_id, source, _destination_uri(bucket, kind, zone_id)))

        # Validates every `outputs`/`planned_outputs` entry, and now does so before the copies
        # rather than after them.
        patched_manifest = _rewrite_manifest_uris(manifest, bucket=bucket, run_id=final_run_id)
        # The structural gate `check` runs first on every manifest it reads, applied to the
        # manifest about to be PUBLISHED. `_rewrite_manifest_uris` validates the entries it
        # finds and lets a manifest with no `outputs` key, or an empty list, through: measured
        # on e99100c, such a bundle had every artifact copied into the destination bucket and a
        # manifest published that `check` then refused with "manifest outputs must be a
        # non-empty list" — objects in the handoff bucket that nothing readable vouches for.
        # Gated AFTER the rewrite rather than on the raw bundle because `_outputs` also binds
        # each `uri` to its own kind and zone, and the bundle's URIs are the ORIGIN's, replaced
        # wholesale by the line above; the destination URIs are what `check` will read back.
        outputs = _outputs(patched_manifest)
        _refuse_an_inventory_mismatch(outputs, planned)

        with duckdb_session(":memory:", load_httpfs=True) as conn:
            imported: list[dict[str, Any]] = []
            for kind, zone_id, source, destination in planned:
                _copy_parquet(conn, source.as_posix(), destination)
                imported.append({"kind": kind, "zone_id": zone_id, "uri": destination})

            local_manifest = root / "imported-manifest.json"
            _write_json(local_manifest, patched_manifest)
            conn.execute(
                "COPY (SELECT * FROM read_json_auto(?)) TO "
                f"{_safe_sql_literal(final_manifest_uri)} (FORMAT JSON)",
                [local_manifest.as_posix()],
            )

    return {
        "ok": True,
        "bucket": bucket,
        "run_id": final_run_id,
        "manifest_uri": final_manifest_uri,
        "n_imported": len(imported),
    }


def _dispatch(args: argparse.Namespace) -> dict[str, Any]:
    if args.command == "check":
        return check_manifest(args.manifest_uri, expected_zones=args.expected_zones)
    if args.command == "export-zip":
        return export_zip(
            args.manifest_uri,
            Path(args.output),
            expected_zones=args.expected_zones,
        )
    if args.command == "import-zip":
        return import_zip(
            Path(args.bundle),
            bucket=args.bucket,
            run_id=args.run_id,
            manifest_uri=args.manifest_uri,
        )
    raise AssertionError(args.command)  # pragma: no cover


def main() -> int:
    # Install the credential-redaction processor (#148). Only src/server.py called this, so
    # every `log.error(..., error=str(e))` these handlers reach went to stderr raw when a
    # script drove them in-process -- structlog's DEFAULT chain has no redactor (#161 review).
    setup_logging()
    args = _parser().parse_args()
    try:
        result = _dispatch(args)
    except (
        ManifestError,
        RuntimeError,
        duckdb.IOException,
        duckdb.HTTPException,
        # DuckDB reports "the object is missing" and "the object is not what you said it was" with
        # two different exceptions, and only the first is an IOException. Measured on duckdb 1.5.4:
        # read_json_auto over a non-JSON body (an HTML error page, a truncated object) and
        # read_parquet over a non-parquet body both raise InvalidInputException, so `check` on a
        # URI that resolves to the wrong kind of object used to dump a traceback. That is operator
        # input — a mistyped URI, a stale object — exactly like a missing key.
        #
        # duckdb.InternalException is deliberately NOT here. It signals an engine invariant
        # violation, i.e. a genuine bug, and it must keep surfacing its traceback; it is a sibling
        # of these under DatabaseError, so catching the parent class would have swallowed it.
        duckdb.InvalidInputException,
    ) as exc:
        # A malformed, unsupported or unreadable manifest is operator input, not a crash: print
        # the one line that says so. Anything else (a real bug) still raises with its traceback.
        #
        # Redacted first. An object-store client quotes the address it was given IN FULL on a
        # connection failure, and this line is what an operator pastes into a ticket and what CI
        # keeps in its log. `src/utils/duckdb_conn.py` now sanitizes before configuring httpfs, so
        # the DuckDB spelling is closed at the source; the redaction stays because the message can
        # still carry a raw endpoint from anywhere else that quotes one.
        #
        # `redact_text`, not `redact_credentials`: the latter decides whether a string is prose
        # from the KEY it hangs under, and this one has no key. Passing a bare string through it
        # returns the string untouched — the redaction here was a silent no-op.
        print(
            f"error: {' '.join(redact_text(str(exc), settings.S3_ENDPOINT).split())}",
            file=sys.stderr,
        )
        return 2

    if getattr(args, "json", False) or args.command != "check":
        print(json.dumps(result, indent=2, sort_keys=True, default=_json_default))
    else:
        status = "OK" if result["ok"] else "FAILED"
        print(
            f"{status}: {result['n_zones']} zones, {result['n_outputs']} outputs, "
            f"manifest={result['manifest_uri']}"
        )
        if result["errors"]:
            for error in result["errors"]:
                print(f"- {error}")
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
