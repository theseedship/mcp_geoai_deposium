#!/usr/bin/env python3
"""End-to-end FIERCE demonstrator on the Montpellier pilot (design: docs/design/montpellier-demonstrator.md).

Chains the **shipped** climate + discovery tools on **one real pilot year** and emits a tangible
artifact — a schema-valid ``PatternLibrary`` (the BEJOF↔Owuor exchange unit) plus a human-readable
markdown report:

    era5_ingest(montpellier_metropole, 2023)         → climate/era5/{zone}.parquet        (B2)
    climate_enrich_ndvi(montpellier_metropole, 2023) → climate/unified/{zone}.parquet      (B4)
    discover_patterns(unified, algorithm="tgrad",    → temporal gradual patterns
                      target_column="ndvi_mean", order_by="date", drop_columns=[…])
    library_from_mining_result(result, zone_id, algorithm="tgrad") → PatternLibrary
    → outputs/demo/montpellier_patterns.json + outputs/demo/montpellier_report.md

It adds **no new lettered lot** — it is the **acceptance proof** that B2/B4 + the Palier-1/2
discovery chain compose on real data, and it surfaces the integration seams.

PASS criterion (owner decision, design §2): success = *the chain runs end-to-end and emits a
schema-valid* ``PatternLibrary``. Finding **robust** patterns is NOT the bar — that needs more years
+ O3 cross-validation. The patterns here are **discovery-grade** (support only; ``confidence=None``)
and the ±N-day NDVI↔ERA5 alignment is *proposed* (ADR 0002 §3).

This is a **real-data** script (design §3): it hits the Open-Meteo Archive + Sentinel-2 STAC over the
network and reads/writes the unified parquet on **S3/RustFS** (the production-faithful path —
``discover_patterns`` is bucket-restricted by design, ``src/tools/patterns.py``). It is NOT part of
the CI suite (the network-free wiring lives in ``tests/test_demo_chain.py``). Run it against the dev
harness (``docker-compose`` RustFS) or a configured bucket, in the 3.14 ``patterns`` image:

    uv run --extra patterns python scripts/demo_montpellier.py            # S3_BUCKET from env/.env

Requires: ``S3_BUCKET`` (+ S3 creds) configured, network access, and the ``patterns`` extra (so4gp).
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from datetime import UTC, datetime
from pathlib import Path

# Allow `python scripts/demo_montpellier.py` from anywhere — make the repo root importable.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.config import settings  # noqa: E402
from src.patterns.library_adapter import library_from_mining_result  # noqa: E402
from src.patterns.library_schema import PatternLibrary  # noqa: E402
from src.tools.climate import register_climate_tools  # noqa: E402
from src.tools.patterns import register_pattern_tools  # noqa: E402
from src.utils.logging import setup_logging  # noqa: E402
from src.utils.s3_endpoint import redact_text  # noqa: E402

# --- Locked demo parameters (design §4.4) ---------------------------------- #
ZONE_ID = "montpellier_metropole"
START_DATE = "2023-01-01"
END_DATE = "2023-12-31"
TOLERANCE_DAYS = 3
ALGORITHM = "tgrad"
TARGET_COLUMN = "ndvi_mean"
ORDER_BY = "date"
# Drop geometry + identifiers EXPLICITLY (design §4.3): the unified parquet is a WKB-blob parquet,
# not a `geo`-metadata GeoParquet, so discover_patterns' auto-drop does NOT fire — we must name them.
DROP_COLUMNS = ["zone_id", "lat", "lon", "geometry", "ndvi_date", "day_delta"]
# Sparse unified table (~one row per matched S2 revisit) → start at the documented thresholds and
# step down until the chain emits ≥1 pattern. Thresholds are a *pipeline knob* here, not a claim.
THRESHOLD_TIERS = [(0.5, 0.5), (0.4, 0.4), (0.3, 0.3), (0.2, 0.2)]


class _CaptureMCP:
    """Collect the tool handlers without a live FastMCP server (drops decorator kwargs).

    Same shim the tests use (``tests/test_climate.py``): the demo invokes the **real** tool handlers
    in-process so it exercises the actual ``discover_patterns`` seam — never a direct bridge call.
    """

    def __init__(self) -> None:
        self.tools: dict[str, object] = {}

    def tool(self, *_args, **_kwargs):
        def _decorator(fn):
            self.tools[fn.__name__] = fn
            return fn

        return _decorator


def _load_tools() -> dict:
    cap = _CaptureMCP()
    register_climate_tools(cap)
    register_pattern_tools(cap)
    return cap.tools


# --------------------------------------------------------------------------- #
# Reporting (pure — no network / so4gp / S3; unit-tested in test_demo_chain.py)
# --------------------------------------------------------------------------- #


def _completeness_clause(ingest: dict) -> str:
    """Human phrasing for the stage-1 completeness, robust to era5_ingest's TWO return shapes.

    The full-fetch path carries a ``completeness`` block; the idempotent-skip path (overwrite=False
    + an existing covering parquet, ``src/tools/climate.py``) does NOT — it returns only the stored
    coverage. Defaulting ``pct_days`` to 0 there would print a fabricated "0%/None% of expected days"
    on every cached re-run, so detect the skip and say so instead of inventing a number.
    """
    completeness = ingest.get("completeness")
    if completeness is not None:
        return f"{completeness.get('pct_days', 0)}% of expected days"
    if ingest.get("metadata", {}).get("skipped"):
        return "cached (overwrite=False; coverage not re-measured)"
    return "completeness not reported"


def _patterns_table(library: PatternLibrary) -> str:
    """Render the library's patterns as a markdown table (the consumer-facing rule form)."""
    if not library.patterns:
        # Neutral wording: an empty library can also mean the loop broke early at a higher tier, or
        # discover_patterns found patterns that were unmappable (support=None / empty itemset) and
        # library_from_mining_result skipped them — not necessarily "the lowest tier".
        return (
            "_No schema-mappable patterns in the library — see the run log above for the "
            "thresholds tried and any skipped patterns._\n"
        )
    header = (
        "| # | antecedent (lagged) | consequent (target) | support | lag (days) | temporal |\n"
        "|---|---------------------|---------------------|---------|------------|----------|\n"
    )
    rows = []
    for p in library.patterns:
        rows.append(
            f"| {p.pattern_id} | {p.antecedent or '—'} | {p.consequent or '—'} | "
            f"{p.support:.3f} | {p.lag_days:+.1f} | {'yes' if p.is_temporal else 'no'} |"
        )
    return header + "\n".join(rows) + "\n"


def build_report(
    *,
    library: PatternLibrary,
    ingest: dict,
    enrich: dict,
    mining: dict,
    used_min_support: float,
    used_min_rep: float,
    generated_at: str,
) -> str:
    """Build the human-readable markdown report from the chain's stage results (PURE).

    Surfaces the provenance the design §2 calls for: ``n_days``, completeness %, ``n_matched``,
    per-pattern support + signed ``lag_days``, and the honesty caveats (discovery-grade, proposed
    alignment, one-year pipeline proof). Takes plain dicts + the validated library so it is
    network-free and unit-testable.
    """
    stats = enrich.get("statistics", {})
    mining_stats = mining.get("statistics", {})
    enrich_meta = enrich.get("metadata", {})
    truncation = enrich_meta.get("ndvi_truncation_warning")

    lines = [
        "# FIERCE demonstrator — Montpellier pilot",
        "",
        f"_Generated {generated_at} · zone `{library.zone_id}` · "
        f"{ingest.get('start_date')} → {ingest.get('end_date')} · algorithm `{library.algorithm}`_",
        "",
        "## What this is",
        "",
        "A **pipeline proof**: the shipped B2 (`era5_ingest`) + B4 (`climate_enrich_ndvi`) climate "
        "plane chained into Palier-2 temporal pattern mining (`discover_patterns(tgrad)`) and "
        "emitted as a schema-valid `PatternLibrary` (the BEJOF↔Owuor exchange unit). Success here "
        "means **the chain ran and the artifact is valid** — not that the patterns are robust "
        "(that needs more years + O3 cross-validation).",
        "",
        "## Chain provenance",
        "",
        "| stage | tool | result |",
        "|-------|------|--------|",
        f"| 1 · ERA5 ingest (B2) | `era5_ingest` | {ingest.get('n_days', 0)} daily records, "
        f"{_completeness_clause(ingest)} |",
        f"| 2 · NDVI⋈ERA5 (B4) | `climate_enrich_ndvi` | "
        f"{stats.get('n_ndvi_observations', 0)} NDVI obs → {stats.get('n_matched', 0)} unified rows "
        f"(±{TOLERANCE_DAYS} d join) |",
        f"| 3 · mining (Palier 2) | `discover_patterns(tgrad)` | "
        f"{mining_stats.get('n_patterns', 0)} pattern(s) at "
        f"min_support={used_min_support}, min_rep={used_min_rep} |",
        f"| 4 · exchange (O2) | `library_from_mining_result` | "
        f"{len(library.patterns)} record(s) in the PatternLibrary |",
        "",
        "## Patterns (discovery-grade)",
        "",
        _patterns_table(library),
        "## Honesty notes",
        "",
        "- **No calibrated confidence.** Every record carries `confidence=None` — support only. A "
        "calibrated `support × forecast × decay` confidence requires T-GRAANK lag + ≥5-fold "
        "cross-zone validation (O3, Palier 4); fabricating it earlier replays the original defect "
        "(ADR 0001 §5).",
        "- **Proposed alignment.** The ±N-day NDVI↔ERA5 join is *proposed*, not ratified "
        "(ADR 0002 §3).",
        "- **One year = first proof.** A single pilot year is deliberate — it tests that the "
        "pipeline runs end-to-end, not that the patterns are statistically robust.",
        f"- **Thresholds are a pipeline knob.** `min_support`/`min_rep` were stepped down to "
        f"{used_min_support}/{used_min_rep} only until the chain emitted ≥1 pattern — a plumbing "
        "setting to exercise the seam, not a tuned scientific parameter (design §4.4).",
    ]
    if truncation:
        lines += ["- **NDVI truncation.** " + truncation + "."]
    lines += [
        "",
        "## Artifact",
        "",
        "The machine-readable library is in `montpellier_patterns.json` (validates against "
        "`PatternLibrary.model_json_schema()`).",
        "",
    ]
    return "\n".join(lines)


def write_artifacts(library: PatternLibrary, report: str, out_dir: Path) -> tuple[Path, Path]:
    """Write the PatternLibrary JSON + the markdown report to ``out_dir``; return both paths."""
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "montpellier_patterns.json"
    report_path = out_dir / "montpellier_report.md"
    json_path.write_text(library.model_dump_json(indent=2) + "\n", encoding="utf-8")
    report_path.write_text(report, encoding="utf-8")
    return json_path, report_path


# --------------------------------------------------------------------------- #
# Chain orchestration (real: network + S3/RustFS)
# --------------------------------------------------------------------------- #


class _ChainError(RuntimeError):
    """A stage returned an ``error`` — abort the chain with its message."""


async def mine_with_stepping(
    discover_patterns,
    *,
    source: str,
    tiers: list[tuple[float, float]] = THRESHOLD_TIERS,
) -> tuple[dict, float, float]:
    """Stage 3: call the REAL ``discover_patterns(tgrad)`` tool, stepping ``min_support``/``min_rep``
    down through ``tiers`` until it emits ≥1 pattern. Returns (mining_result, used_min_support,
    used_min_rep) — the result is whatever the last tier produced (0 patterns is a valid outcome).

    Factored out of :func:`run_chain` so the stepping loop + the exact ``discover_patterns`` call
    kwargs are exercised by the network-free seam test (resolver monkeypatched) — they can no longer
    drift from the live demo undetected. Raises :class:`_ChainError` on a tool-level error.
    """
    mining: dict = {}
    # Fallback only for the degenerate empty-tiers case (the for/else `else` fires and we still
    # return well-formed values); any real tier overwrites these on the first iteration. Using a
    # sentinel rather than tiers[-1] avoids an IndexError when tiers is empty.
    used_min_support, used_min_rep = 0.0, 0.0
    for min_support, min_rep in tiers:
        mining = await discover_patterns(
            source=source,
            algorithm=ALGORITHM,
            target_column=TARGET_COLUMN,
            order_by=ORDER_BY,
            drop_columns=DROP_COLUMNS,
            min_support=min_support,
            min_rep=min_rep,
            write_output=False,  # build the library from the in-memory result (design §4.5)
        )
        if "error" in mining:
            raise _ChainError(f"discover_patterns failed: {mining['error']}")
        n_patterns = mining.get("statistics", {}).get("n_patterns", 0)
        used_min_support, used_min_rep = min_support, min_rep
        print(f"      min_support={min_support}, min_rep={min_rep} → {n_patterns} pattern(s)")
        if n_patterns >= 1:
            break
    else:
        print(
            "      no patterns at any tier — the chain still completes; the report records 0 "
            "patterns honestly (a valid run, just no signal on one sparse year)."
        )
    return mining, used_min_support, used_min_rep


async def run_chain(
    *,
    zone_id: str,
    start_date: str,
    end_date: str,
    overwrite: bool,
) -> tuple[PatternLibrary, dict, dict, dict, float, float]:
    """Run the full B2 → B4 → discover_patterns(tgrad) → library chain. Returns
    (library, ingest, enrich, mining, used_min_support, used_min_rep). Raises _ChainError on a
    stage error (each stage's tool returns ``{"error": …}`` on failure rather than raising)."""
    tools = _load_tools()
    era5_ingest = tools["era5_ingest"]
    climate_enrich_ndvi = tools["climate_enrich_ndvi"]
    discover_patterns = tools["discover_patterns"]

    print(f"[1/4] era5_ingest({zone_id}, {start_date}..{end_date}) → S3 …")
    ingest = await era5_ingest(
        zone_id=zone_id, start_date=start_date, end_date=end_date, overwrite=overwrite
    )
    if "error" in ingest:
        raise _ChainError(f"era5_ingest failed: {ingest['error']}")
    print(
        f"      {ingest.get('n_days')} days → {ingest.get('output')} "
        f"({_completeness_clause(ingest)})"
    )

    print(f"[2/4] climate_enrich_ndvi({zone_id}) → unified S3 parquet …")
    enrich = await climate_enrich_ndvi(
        zone_id=zone_id,
        start_date=start_date,
        end_date=end_date,
        tolerance_days=TOLERANCE_DAYS,
    )
    if "error" in enrich:
        raise _ChainError(f"climate_enrich_ndvi failed: {enrich['error']}")
    unified_source = enrich.get("output")
    n_matched = enrich.get("statistics", {}).get("n_matched", 0)
    print(f"      {n_matched} unified rows → {unified_source}")
    if not unified_source:
        raise _ChainError(
            "climate_enrich_ndvi produced no unified table "
            f"(no NDVI matched within ±{TOLERANCE_DAYS} d): "
            f"{enrich.get('metadata', {}).get('note', 'see metadata')}"
        )

    print("[3/4] discover_patterns(tgrad) — stepping thresholds until ≥1 pattern …")
    mining, used_min_support, used_min_rep = await mine_with_stepping(
        discover_patterns, source=unified_source
    )

    print("[4/4] library_from_mining_result → PatternLibrary …")
    library = library_from_mining_result(mining, zone_id=zone_id, algorithm=ALGORITHM)
    print(f"      {len(library.patterns)} schema-valid record(s)")

    return library, ingest, enrich, mining, used_min_support, used_min_rep


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="demo_montpellier",
        description="End-to-end FIERCE demonstrator (B2 → B4 → tgrad → PatternLibrary) on a pilot zone.",
    )
    p.add_argument("--zone-id", default=ZONE_ID, help=f"pilot zone (default: {ZONE_ID})")
    p.add_argument("--start-date", default=START_DATE, help=f"ISO start (default: {START_DATE})")
    p.add_argument("--end-date", default=END_DATE, help=f"ISO end (default: {END_DATE})")
    p.add_argument(
        "--out-dir",
        default=str(_REPO_ROOT / "outputs" / "demo"),
        help="where to write the JSON + report (default: outputs/demo/)",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="re-fetch + re-write the ERA5 parquet even if it already covers the range",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    # Install the credential-redaction processor (#148). Only src/server.py called this, so
    # every `log.error(..., error=str(e))` these handlers reach went to stderr raw when a
    # script drove them in-process -- structlog's DEFAULT chain has no redactor (#161 review).
    setup_logging()
    args = build_parser().parse_args(argv)

    if not settings.S3_BUCKET:
        print(
            "ERROR: S3_BUCKET is not configured — this demonstrator reads/writes the unified parquet "
            "on S3/RustFS (discover_patterns is bucket-restricted by design). Start the dev-harness "
            "RustFS or set S3_BUCKET (+ S3 creds) and re-run.",
            file=sys.stderr,
        )
        return 2

    generated_at = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    try:
        library, ingest, enrich, mining, used_min_support, used_min_rep = asyncio.run(
            run_chain(
                zone_id=args.zone_id,
                start_date=args.start_date,
                end_date=args.end_date,
                overwrite=args.overwrite,
            )
        )
    except _ChainError as exc:
        # The chain wraps provider exceptions, and STAC/httpfs quote the configured credential in
        # theirs (#161). Same shape already applied at scripts/unified_pilot_panel.py:952.
        print(f"ERROR: {redact_text(str(exc), settings.S3_ENDPOINT)}", file=sys.stderr)
        return 1

    report = build_report(
        library=library,
        ingest=ingest,
        enrich=enrich,
        mining=mining,
        used_min_support=used_min_support,
        used_min_rep=used_min_rep,
        generated_at=generated_at,
    )
    json_path, report_path = write_artifacts(library, report, Path(args.out_dir))

    print()
    print(f"✓ chain complete — {len(library.patterns)} pattern(s) in the PatternLibrary")
    print(f"  library: {json_path}")
    print(f"  report : {report_path}")
    if not library.patterns:
        print(
            "  NOTE: 0 patterns — the chain ran and the artifact is schema-valid (the §2 PASS "
            "criterion), but one sparse year surfaced no temporal signal."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
