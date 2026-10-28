#!/usr/bin/env python3
"""Run the B3 standalone Sentinel-2 NDVI zonal-stats panel for the 3 FR pilots (issue #27).

Operator wrapper around the ``ndvi_ingest`` tool: it makes the multi-year, monthly-cadence NDVI
backfill for the three named pilots repeatable, materializes one WKB GeoParquet per zone under
``climate/ndvi/{zone_id}.parquet`` (local path or RustFS/S3), reads each artifact back through
DuckDB to confirm the row count + date span, publishes an immutable run manifest under
``climate/runs/{run_id}/manifest.json``, and prints a JSON summary.

It implements no NDVI logic — the COG streaming, masking and zonal stats all live in
``ndvi_ingest`` / ``src/tools/_ndvi.py``. Storage target follows the standard S3 settings
(``S3_BUCKET`` etc.); pass ``--output-dir`` to write local parquet instead. The run manifest
reuses ``climate_baseline_panel``'s confined writer, so B3 runs land in the same
``climate/runs/`` namespace — and under the same URI validation — as the ERA5 panels.

A full 3-pilot 2015-2024 backfill streams ~10 years x 3 zones of COGs and takes ~20 minutes, so
``--summarize-existing`` re-publishes a manifest over the already materialized artifacts (readback
only, no COG streaming) — the same escape hatch ``climate_baseline_panel.py`` offers, and it marks
the manifest ``provenance.mode = "aggregation_only"`` so an aggregate pass can never pass itself
off as the materializing run.

Example (local RustFS from the dev harness):

    S3_ENDPOINT=http://127.0.0.1:9200 S3_ACCESS_KEY=deposium-dev \\
    S3_SECRET_KEY=deposium-dev-secret S3_BUCKET=geoai-dev S3_FORCE_PATH_STYLE=true \\
    uv run python scripts/ndvi_pilot_panel.py --start 2015-01-01 --end 2024-12-31 --json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

# Allow `python scripts/ndvi_pilot_panel.py` from anywhere.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts import climate_baseline_panel  # noqa: E402
from src.config import settings  # noqa: E402
from src.tools.climate import _default_output_uri, register_climate_tools  # noqa: E402
from src.utils.dates import month_ranges  # noqa: E402
from src.utils.duckdb_conn import duckdb_session  # noqa: E402
from src.utils.logging import setup_logging  # noqa: E402
from src.utils.s3_endpoint import (  # noqa: E402
    redact_credentials,
    redact_text,
    sanitize_stac_url,
)

RunManifestWriter = Callable[[str, dict[str, Any]], None]

PILOT_ZONE_IDS = ("montpellier_metropole", "bassin_du_lez", "fos_sur_mer")
DEFAULT_START_DATE = "2015-01-01"
DEFAULT_END_DATE = "2024-12-31"


class _CaptureMCP:
    def __init__(self) -> None:
        self.tools: dict[str, object] = {}

    def tool(self, *_args, **_kwargs):
        def _decorator(fn):
            self.tools[fn.__name__] = fn
            return fn

        return _decorator


def _load_ndvi_tool():
    cap = _CaptureMCP()
    register_climate_tools(cap)
    return cap.tools["ndvi_ingest"]


def _readback(dst: str) -> dict[str, Any]:
    """Confirm the materialized artifact via DuckDB: row count + date span + geometry decodes."""
    with duckdb_session(":memory:", load_httpfs=dst.startswith("s3://")) as conn:
        n, dmin, dmax, geoms = conn.execute(
            "SELECT count(*), min(date), max(date), "
            "count(DISTINCT ST_AsText(ST_GeomFromWKB(geometry))) "
            f"FROM read_parquet('{dst}')"
        ).fetchone()
    return {"rows": n, "first_date": str(dmin), "last_date": str(dmax), "distinct_points": geoms}


def _manifest_exists(uri: str) -> bool:
    """Whether a run manifest already lives at ``uri``.

    Probes through the same DuckDB/httpfs path the manifest is written and read with. A read that
    fails for any reason — missing key, but also a transient object-store error — is reported as
    "absent", so this guards against accidental reuse of a ``--run-id`` rather than being a
    distributed lock.
    """
    try:
        with duckdb_session(":memory:", load_httpfs=True) as conn:
            conn.execute("SELECT 1 FROM read_json_auto(?) LIMIT 1", [uri]).fetchone()
    except Exception:
        return False
    return True


def _resolve_run_id(args: argparse.Namespace, *, zone_ids: list[str], created_at: str) -> str:
    """Derive the run id, reusing the baseline panel's charset rules and timestamp shape."""
    if args.run_id:
        if error := climate_baseline_panel._run_id_error(args.run_id):
            raise ValueError(error)
        return args.run_id
    stamp = climate_baseline_panel._run_id_timestamp(created_at)
    return f"{stamp}_ndvi_{len(zone_ids)}zones_{args.start}_{args.end}".replace("-", "_")


def _run_manifest_uri(args: argparse.Namespace, run_id: str) -> str | None:
    """The manifest is a data-plane artifact: only published for S3-backed runs."""
    if args.no_run_manifest or args.output_dir or not settings.S3_BUCKET:
        return None
    if climate_baseline_panel._run_id_error(run_id):
        return None
    return f"s3://{settings.S3_BUCKET}/climate/runs/{run_id}/manifest.json"


def _run_manifest_outputs(panel: dict[str, Any]) -> list[dict[str, Any]]:
    """Advertise consumable artifacts only — a failed zone has nothing to hand off."""
    outputs = []
    for zone in panel["zones"]:
        if not zone.get("ok"):
            continue
        readback = zone.get("readback") or {}
        outputs.append(
            {
                "kind": "ndvi",
                "zone_id": zone["zone_id"],
                "uri": zone["output"],
                "rows": readback.get("rows"),
                "first_date": readback.get("first_date"),
                "last_date": readback.get("last_date"),
                "distinct_points": readback.get("distinct_points"),
            }
        )
    return outputs


def _run_manifest_provenance(panel: dict[str, Any]) -> dict[str, Any]:
    """Never let an aggregate-only pass claim it materialized the artifacts it describes."""
    if panel["summarize_existing"]:
        return {
            "mode": "aggregation_only",
            "artifact_code_ref": None,
            "aggregation_code_ref": panel["code_ref"],
            "note": (
                "summarize_existing reuses already materialized zone artifacts; "
                "the current checkout only produced this manifest and its readbacks"
            ),
        }
    return {"mode": "materialization", "artifact_code_ref": panel["code_ref"]}


def _months_requested(window: dict[str, Any]) -> int | None:
    """How many monthly slices the window splits into — ``None`` when it is not usable.

    Reuses ``src.utils.dates.month_ranges``, the same split the monthly collector walks. An
    unparseable date is a ``ValueError``; an inverted window is not — ``month_ranges`` simply
    returns ``[]``, which would make ``months_requested`` 0 and every zone trivially "complete"
    against a zero denominator. An empty split is therefore reported as unusable too.

    This is the fallback denominator. The authoritative one is the tool's own
    ``metadata.months_or_scenes_searched``; see ``_months_searched``.
    """
    try:
        months = month_ranges(window["start_date"], window["end_date"])
    except ValueError:
        return None
    return len(months) or None


def _months_searched(panel: dict[str, Any]) -> int | None:
    """The denominator the run itself reported, falling back to recomputing it from the window.

    ``ndvi_ingest`` returns ``metadata.months_or_scenes_searched`` — for a monthly run that is
    literally ``len(month_ranges(...))`` inside the tool. Preferring it means the manifest quotes
    the number the collector actually walked instead of a second, hand-synchronised computation.
    Every zone runs the same window, so the first reported value speaks for the panel.
    ``--summarize-existing`` has no tool result at all, hence the window fallback.
    """
    for zone in panel["zones"]:
        searched = zone.get("months_searched")
        if isinstance(searched, int):
            return searched or None
    return _months_requested(panel["window"])


def _zone_complete(observed: int, requested: int | None, *, aggregation_only: bool) -> bool | None:
    """``None`` means "not evaluated" — never "incomplete".

    ``--summarize-existing`` has no per-month evidence: ``observed`` there is the artifact's TOTAL
    parquet row count, spanning whatever window materialized it, not the months inside the window
    the aggregate pass was asked about. Twelve months requested against an 81-row artifact is not
    "complete", it is two numbers that do not divide. Completeness is not evaluated in that mode.

    The same incommensurability shows up structurally whenever ``observed`` exceeds ``requested``:
    monthly cadence keeps at most one observation per month, so a larger numerator proves the two
    sides are not counting the same thing. Refuse to claim either way.
    """
    if requested is None or aggregation_only or observed > requested:
        return None
    return observed >= requested


def _run_manifest_month_coverage(panel: dict[str, Any]) -> dict[str, Any]:
    """Record ``N months requested -> M observed`` per zone, so a short series stays visible.

    Monthly cadence keeps at most one observation per month, so a materialized zone's observation
    count IS its number of covered months. Without this the manifest states "81 observations" and
    leaves a consumer unable to tell 81 out of 81 from 81 out of 120.

    ``months_observed`` is ``zone["n_observations"]``, and on BOTH panel paths that number is the
    row count re-read out of the parquet — ``--summarize-existing`` has no tool result to ask, and
    the collection path takes the readback's count after failing the zone on any divergence from
    what ``ndvi_ingest`` reported. The numerator is therefore a property of the object, not the
    tool's recollection of having written it. It is the whole point of the field: a coverage
    figure that a consumer cannot re-derive from the artifact is the unverifiable claim this
    contract exists to remove.

    ``complete_evaluated`` covers every PANEL-LEVEL reason completeness could not be evaluated,
    not just ``--summarize-existing``. There is a second one: no usable denominator. An
    unparseable or inverted window makes ``_months_searched`` return ``None``, which nulls every
    per-zone ``complete``; publishing ``complete_evaluated: true`` beside ``months_requested: null``
    would claim an evaluation happened when nothing was evaluated at all.

    ``true`` therefore means "the panel was in a position to evaluate completeness", NOT "every
    zone got a verdict". A per-zone ``complete`` can still be ``null`` under a true flag, for the
    one remaining reason, which is per zone rather than panel-wide:
    ``months_observed > months_requested`` — a numerator above its denominator proves the two
    sides are not counting the same thing (see ``_zone_complete``).
    """
    aggregation_only = bool(panel["summarize_existing"])
    requested = _months_searched(panel)
    return {
        "months_requested": requested,
        "complete_evaluated": not aggregation_only and requested is not None,
        "zones": [
            {
                "zone_id": zone["zone_id"],
                "months_observed": zone["n_observations"],
                "complete": _zone_complete(
                    zone["n_observations"], requested, aggregation_only=aggregation_only
                ),
            }
            for zone in panel["zones"]
            if zone.get("ok")
        ],
    }


def _run_manifest_payload(panel: dict[str, Any]) -> dict[str, Any]:
    window = panel["window"]
    # A list, not the tuple `sanitize_stac_url` returns: this payload goes through `json.dumps`
    # and a DuckDB `COPY ... (FORMAT JSON)`, and a tuple is not a JSON type.
    stac_address, stac_reductions = sanitize_stac_url(settings.STAC_URL)
    failed = [z["zone_id"] for z in panel["zones"] if not z.get("ok")]
    return {
        "schema_version": 1,
        "kind": "geoai_run_manifest",
        "run_id": panel["run_id"],
        "created_at": panel["created_at"],
        "tool_or_script": "scripts/ndvi_pilot_panel.py",
        "code_ref": None if panel["summarize_existing"] else panel["code_ref"],
        "provenance": _run_manifest_provenance(panel),
        "environment": climate_baseline_panel._run_manifest_environment(),
        "parameters": {
            "zone_ids": panel["zone_ids"],
            "start_date": window["start_date"],
            "end_date": window["end_date"],
            "cloud_cover_max": panel["cloud_cover_max"],
            "monthly": panel["monthly"],
            "storage_mode": panel["storage_mode"],
            "summarize_existing": panel["summarize_existing"],
        },
        "inputs": [
            {
                "kind": "stac_sentinel2_l2a",
                "collection": "sentinel-2-l2a",
                # `sanitize_stac_url`, NOT `sanitize_endpoint`: a catalogue root has a path
                # (`https://earth-search.aws.element84.com/v1`) and the endpoint grammar refuses
                # a path outright, so that helper published `null` here for the DEFAULT
                # deployment — losing the provenance this field exists to record.
                #
                # `endpoint_reductions` is published beside it because the address is not always
                # the configured value: a private catalogue is reached as
                # `https://user:token@stac.internal/v1` and the userinfo is dropped, and a
                # `?token=`/`?collections=` query is dropped too. Recording WHICH components were
                # removed is what keeps that from being a silent misstatement — `[]` means the
                # address is the setting verbatim, `["query"]` means a reader is looking at a
                # prefix of it. `["refused"]` accompanies a null address.
                "endpoint": stac_address,
                "endpoint_reductions": list(stac_reductions),
                "license": "Copernicus Sentinel data (free, full and open)",
                "streaming": "STAC/COG via GDAL /vsicurl/ (no COG storage)",
            }
        ],
        "outputs": _run_manifest_outputs(panel),
        "checks": {
            "ok": panel["ok"],
            "n_zones_requested": len(panel["zone_ids"]),
            "n_zones_completed": sum(1 for z in panel["zones"] if z.get("ok")),
            "failed_zone_ids": failed,
            "month_coverage": _run_manifest_month_coverage(panel),
            # `truncation_warnings` is only ever populated by the all-scenes STAC item cap, which
            # a monthly-cadence run cannot reach — so for this panel the list is structurally
            # empty and must not be read as coverage evidence. `month_coverage` above is what
            # actually carries coverage. The field itself is kept because consumers already read
            # it; no companion "applicable" flag is emitted, because this panel hardcodes
            # `monthly=True` and such a flag could only ever be a constant dressed as a check.
            "truncation_warnings": panel["truncation_warnings"],
        },
        "errors": [
            {"zone_id": z["zone_id"], "error": z["error"]} for z in panel["zones"] if z.get("error")
        ],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="B3 NDVI pilot panel (issue #27).")
    parser.add_argument("--zones", nargs="+", default=list(PILOT_ZONE_IDS), help="zone ids to run")
    parser.add_argument("--start", default=DEFAULT_START_DATE, help="ISO start date (inclusive)")
    parser.add_argument("--end", default=DEFAULT_END_DATE, help="ISO end date (inclusive)")
    parser.add_argument("--cloud-cover-max", type=int, default=settings.DEFAULT_CLOUD_COVER_MAX)
    parser.add_argument(
        "--output-dir", default=None, help="write local parquet here instead of S3 (dir must exist)"
    )
    parser.add_argument("--run-id", default=None, help="explicit run id (else derived from window)")
    parser.add_argument(
        "--no-run-manifest",
        action="store_true",
        help="skip publishing climate/runs/{id}/manifest.json",
    )
    parser.add_argument(
        "--overwrite-run-manifest",
        action="store_true",
        help="replace an existing manifest at the same run id (manifests are immutable by default)",
    )
    parser.add_argument(
        "--summarize-existing",
        action="store_true",
        help=(
            "skip NDVI collection: read back the already materialized per-zone artifacts and "
            "publish the manifest over them (a full 3-pilot backfill takes ~20 min, so "
            "re-publishing a manifest must not require re-streaming every COG)"
        ),
    )
    parser.add_argument("--json", action="store_true", help="print machine-readable JSON")
    return parser


def run_panel(
    args: argparse.Namespace,
    *,
    ndvi_runner=None,
    run_manifest_writer: RunManifestWriter = climate_baseline_panel._write_s3_json,
    run_manifest_exists: Callable[[str], bool] | None = None,
) -> dict[str, Any]:
    """Materialize the NDVI panel, read every artifact back, then publish the run manifest."""
    ndvi_ingest = None if args.summarize_existing else (ndvi_runner or _load_ndvi_tool())
    manifest_exists = run_manifest_exists or _manifest_exists
    out_dir = Path(args.output_dir).resolve() if args.output_dir else None
    created_at = climate_baseline_panel._utc_now()
    zone_ids = list(args.zones)
    run_id = _resolve_run_id(args, zone_ids=zone_ids, created_at=created_at)

    zones: list[dict[str, Any]] = []
    truncation_warnings: list[dict[str, Any]] = []
    for zone_id in zone_ids:
        dst = (
            (out_dir / f"{zone_id}.parquet").as_posix()
            if out_dir
            else _default_output_uri(zone_id, "ndvi")
        )
        if not dst:
            raise SystemExit(f"No destination for {zone_id!r}: set S3_BUCKET or pass --output-dir")
        t0 = time.monotonic()
        if args.summarize_existing:
            # Read back what is already materialized. A missing or unreadable artifact must fail
            # the zone — a manifest may never advertise an object nobody could read.
            try:
                readback = _readback(dst)
            except Exception as e:
                zones.append({"zone_id": zone_id, "ok": False, "error": f"readback failed: {e}"})
                continue
            zones.append(
                {
                    "zone_id": zone_id,
                    "ok": True,
                    "output": dst,
                    "n_observations": readback["rows"],
                    "statistics": None,
                    "readback": readback,
                    "elapsed_s": round(time.monotonic() - t0, 1),
                }
            )
            continue
        result = asyncio.run(
            ndvi_ingest(
                zone_id=zone_id,
                start_date=args.start,
                end_date=args.end,
                cloud_cover_max=args.cloud_cover_max,
                monthly=True,
                output_path=dst,
            )
        )
        if "error" in result:
            zones.append({"zone_id": zone_id, "ok": False, "error": result["error"]})
            continue
        # A capped search cannot prove full coverage — carry the tool's warning into the manifest
        # rather than letting a truncated series look complete downstream.
        warning = (result.get("metadata") or {}).get("ndvi_truncation_warning")
        if warning:
            truncation_warnings.append({"zone_id": zone_id, "warning": warning})
        # An unverifiable artifact is a failed zone, not a dead panel: a readback error here must
        # not abort the remaining zones nor skip manifest publication, or a good materialization
        # ends up with no published run record at all.
        try:
            readback = _readback(result["output"])
        except Exception as e:
            zones.append({"zone_id": zone_id, "ok": False, "error": f"readback failed: {e}"})
            continue
        # The manifest's `month_coverage` figure is derived from `n_observations`, so where that
        # number comes from decides whether the coverage claim is verifiable. The tool's own
        # count is its RECOLLECTION of what it wrote; `readback["rows"]` is the object. They are
        # not the same claim, and a manifest whose coverage comes from memory is exactly the kind
        # of unverifiable assertion this contract exists to remove.
        #
        # Both are used: the readback count is what the zone record carries, and a divergence
        # FAILS the zone. Silently preferring the readback would be the wrong half of the choice
        # — a disagreement between what the tool believes it wrote and what the parquet holds is
        # a real defect (a partial write, a stale object at that key, a bug in the writer), and
        # papering over it publishes a manifest that looks fine over an artifact that is not.
        if result["n_observations"] != readback["rows"]:
            zones.append(
                {
                    "zone_id": zone_id,
                    "ok": False,
                    "error": (
                        f"row count divergence: ndvi_ingest reported "
                        f"{result['n_observations']} observations, the materialized artifact at "
                        f"{result['output']} holds {readback['rows']} rows"
                    ),
                }
            )
            continue
        zones.append(
            {
                "zone_id": zone_id,
                "ok": True,
                "output": result["output"],
                # From the OBJECT, not from the tool's report. Equal to it by the check above;
                # taken from the readback so that the field's provenance is the artifact even
                # when the two agree.
                "n_observations": readback["rows"],
                # The denominator straight from the run that produced the numerator; see
                # `_months_searched`. Absent on the --summarize-existing path (no tool result).
                "months_searched": (result.get("metadata") or {}).get("months_or_scenes_searched"),
                "statistics": result["statistics"],
                "readback": readback,
                "elapsed_s": round(time.monotonic() - t0, 1),
            }
        )

    panel: dict[str, Any] = {
        "ok": all(z.get("ok") for z in zones),
        "run_id": run_id,
        "created_at": created_at,
        "code_ref": climate_baseline_panel._code_ref(),
        "window": {"start_date": args.start, "end_date": args.end},
        "zone_ids": zone_ids,
        "cloud_cover_max": args.cloud_cover_max,
        "monthly": True,
        "storage_mode": "local" if out_dir else "s3",
        "summarize_existing": bool(args.summarize_existing),
        "truncation_warnings": truncation_warnings,
        "zones": zones,
        "run_manifest_uri": None,
        "run_manifest_written": False,
        "run_manifest_errors": [],
    }

    uri = _run_manifest_uri(args, run_id)
    if uri:
        panel["run_manifest_uri"] = uri
        # Run manifests are immutable (issue #27 acceptance criteria). A reused --run-id would
        # otherwise silently replace an existing record — in particular a --summarize-existing pass
        # could overwrite a materialization manifest and destroy its artifact_code_ref.
        if not args.overwrite_run_manifest and manifest_exists(uri):
            panel["run_manifest_errors"].append(
                {
                    "uri": uri,
                    "error": (
                        "a run manifest already exists at this URI and run manifests are "
                        "immutable — use a fresh --run-id, or pass --overwrite-run-manifest to "
                        "deliberately replace it"
                    ),
                }
            )
            panel["ok"] = False
            return panel
        try:
            run_manifest_writer(uri, _run_manifest_payload(panel))
            panel["run_manifest_written"] = True
        except Exception as e:
            # A manifest write failure must not erase a good materialization — surface it and
            # fail the run so the operator re-publishes rather than silently shipping unmanifested
            # artifacts.
            panel["run_manifest_errors"].append({"uri": uri, "error": str(e)})
            panel["ok"] = False
    return panel


def main(argv: list[str] | None = None) -> int:
    # Install the credential-redaction processor (#148). Only src/server.py called this, so
    # every `log.error(..., error=str(e))` these handlers reach went to stderr raw when a
    # script drove them in-process -- structlog's DEFAULT chain has no redactor (#161 review).
    setup_logging()
    args = build_parser().parse_args(argv)
    panel = run_panel(args)

    # `panel` is the IN-MEMORY dict, so it has never passed through `_write_json` or
    # `_write_s3_json` — neither redacting writer has seen it. Every `error` field below is
    # verbatim exception text, and both clients this repo uses quote the configured credential
    # in it: httpfs echoes `s3_access_key_id` inside a 403 body, botocore names the signing key.
    # stdout is a CI log, a terminal scrollback and whatever an operator pipes it into.
    if args.json:
        print(json.dumps(redact_credentials(panel, settings.S3_ENDPOINT), indent=2))
    else:
        for row in panel["zones"]:
            if row.get("ok"):
                rb = row["readback"]
                print(
                    f"{row['zone_id']:<24} {row['n_observations']:>4} obs  "
                    f"{rb['first_date']}..{rb['last_date']}  -> {row['output']}"
                )
            else:
                # `redact_text`, not `redact_credentials`: the latter decides whether a string is
                # prose from the key it hangs under, and a bare message has no key.
                print(
                    f"{row['zone_id']:<24} FAILED: {redact_text(row['error'], settings.S3_ENDPOINT)}"
                )
        for warning in panel["truncation_warnings"]:
            print(f"{warning['zone_id']:<24} WARNING: {warning['warning']}")
        if panel["run_manifest_written"]:
            print(f"run manifest            -> {panel['run_manifest_uri']}")
        for error in panel["run_manifest_errors"]:
            print(
                "run manifest            FAILED: "
                f"{redact_text(error['error'], settings.S3_ENDPOINT)}"
            )
    return 0 if panel["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
