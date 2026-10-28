"""The read-only `climate_normals` tool (site-watch V0 §G2, #172).

The acceptance criteria this file answers, one section each: the month comes from each
entry's own date; precipitation is banded against the wet-day distribution; `confidence` is
null and asserted, not assumed; the `series[]` entries validate against the **committed**
`watch-input.v1` schema; and the doctrine limits travel in the output rather than only in
the design note.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import re
from pathlib import Path
from unittest import mock

import duckdb
import pytest
from jsonschema import Draft202012Validator, ValidationError

import src.tools.climate as climate
from src.contracts import (
    TOOL_CAPABILITIES,
    TOOL_CONTRACTS,
    ClimateNormalsOutput,
    output_schema_for,
)
from src.tools.climate import (
    NORMALS_CAVEATS,
    WATCH_SERIES_VARIABLES,
    WET_DAY_THRESHOLD_MM,
    _resolve_normals_window,
    build_watch_series,
    classify_watch_entry,
    register_climate_tools,
)
from src.utils.openmeteo import OPEN_METEO_DATA_LICENSE

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SCHEMA = json.loads(
    (_REPO_ROOT / "src/contracts_data/watch/watch-input.v1.schema.json").read_text(encoding="utf-8")
)
#: The committed `entry` definition, resolved so a `series[]` row can be validated on its own.
#: This is the schema Deposium holds a byte-identical copy of, not a local paraphrase of it.
_ENTRY_VALIDATOR = Draft202012Validator({**_SCHEMA["$defs"]["entry"], "$defs": _SCHEMA["$defs"]})
#: The committed `reference` definition, resolved the same way, for `reference` on its own.
_REFERENCE_VALIDATOR = Draft202012Validator(
    {**_SCHEMA["properties"]["reference"], "$defs": _SCHEMA["$defs"]}
)
_DOCUMENT_VALIDATOR = Draft202012Validator(_SCHEMA)
_DATA_LIMIT_TITLES = json.loads(
    (_REPO_ROOT / "tests/fixtures/watch/data-limit-titles.json").read_text(encoding="utf-8")
)


class _CaptureMCP:
    def __init__(self) -> None:
        self.tools: dict[str, object] = {}

    def tool(self, *_args, **_kwargs):
        def _decorator(fn):
            self.tools[fn.__name__] = fn
            return fn

        return _decorator


def _tool():
    cap = _CaptureMCP()
    register_climate_tools(cap)
    return cap.tools["climate_normals"]


_COLUMNS = (
    ("zone_id", "VARCHAR"),
    ("month", "INTEGER"),
    ("variable", "VARCHAR"),
    ("p10", "DOUBLE"),
    ("p50", "DOUBLE"),
    ("p90", "DOUBLE"),
    ("n", "INTEGER"),
    ("mean", "DOUBLE"),
    ("window_start", "VARCHAR"),
    ("window_end", "VARCHAR"),
    ("window_label", "VARCHAR"),
    ("lat", "DOUBLE"),
    ("lon", "DOUBLE"),
    ("p10_wet", "DOUBLE"),
    ("p50_wet", "DOUBLE"),
    ("p90_wet", "DOUBLE"),
    ("n_wet", "INTEGER"),
    ("wet_day_threshold_mm", "DOUBLE"),
    ("schema_version", "INTEGER"),
)


def _row(month, variable, **overrides):
    """One stored normals row, with the v2 wet-day columns on precipitation only."""
    is_precipitation = variable == "precipitation"
    row = {
        "zone_id": "montpellier",
        "month": month,
        "variable": variable,
        "p10": 0.0,
        "p50": 1.0,
        "p90": 8.0,
        "n": 930,
        "mean": 2.0,
        "window_start": "1991-01-01",
        "window_end": "2020-12-31",
        "window_label": "1991-2020",
        "lat": 43.6,
        "lon": 3.87,
        "p10_wet": 2.0 if is_precipitation else None,
        "p50_wet": 6.0 if is_precipitation else None,
        "p90_wet": 20.0 if is_precipitation else None,
        "n_wet": 210 if is_precipitation else None,
        "wet_day_threshold_mm": WET_DAY_THRESHOLD_MM if is_precipitation else None,
        "schema_version": 2,
    }
    row.update(overrides)
    return row


def _write_normals(path: Path, rows: list[dict], *, columns=_COLUMNS) -> str:
    """A normals parquet with the stored column set, `geometry` last as the writer emits it."""
    conn = duckdb.connect()
    try:
        declared = ", ".join(f"{name} {sql_type}" for name, sql_type in columns)
        conn.execute(f"CREATE TABLE normals ({declared}, geometry VARCHAR)")
        placeholders = ", ".join("?" for _ in range(len(columns) + 1))
        conn.executemany(
            f"INSERT INTO normals VALUES ({placeholders})",
            [[row.get(name) for name, _ in columns] + ["POINT(3.87 43.6)"] for row in rows],
        )
        conn.execute("COPY normals TO ? (FORMAT PARQUET)", [str(path)])
    finally:
        conn.close()
    return path.as_posix()


def _fixture_rows(**overrides):
    """January and February, precipitation and wind, distinguishable month by month."""
    return [
        _row(1, "precipitation", **overrides),
        # February's wet distribution sits far above January's, so a day classified against
        # the wrong month lands in a different band and the test can tell.
        _row(2, "precipitation", p10_wet=10.0, p50_wet=25.0, p90_wet=60.0, n_wet=150, **overrides),
        _row(1, "wind_speed", p10=2.0, p50=5.0, p90=12.0, **overrides),
        _row(2, "wind_speed", p10=3.0, p50=9.0, p90=30.0, **overrides),
    ]


@pytest.fixture
def normals(tmp_path, allow_local_parquet_paths):
    """January and February, precipitation and wind, label `1991-2020` (the schema's shape)."""
    path = Path(allow_local_parquet_paths) / f"{tmp_path.name}-normals.parquet"
    yield _write_normals(path, _fixture_rows())
    path.unlink(missing_ok=True)


@pytest.fixture
def normals_default_label(tmp_path, allow_local_parquet_paths):
    """The same rows, labelled as the default producer path labels them (`WMO_1991_2020`)."""
    label = _resolve_normals_window(None, None)[2]
    path = Path(allow_local_parquet_paths) / f"{tmp_path.name}-normals-default.parquet"
    yield _write_normals(path, _fixture_rows(window_label=label))
    path.unlink(missing_ok=True)


def _call(normals_source, **kwargs):
    """Call the tool with the zone's canonical artifact pointed at the fixture.

    The tool takes no source parameter — `science_shared_read` resolves a zone to its
    allowlisted dataset — so a test supplies the artifact by redirecting that resolution,
    which is also the only path a caller can reach.
    """
    zone_id = kwargs.pop("zone_id", "montpellier")
    with mock.patch.object(climate, "_default_output_uri", lambda zone, kind: normals_source):
        return asyncio.run(_tool()(zone_id=zone_id, **kwargs))


# ---------------------------------------------------------------------------------------
# The month comes from each entry's own date.
# ---------------------------------------------------------------------------------------


def test_a_horizon_crossing_a_month_boundary_is_classified_day_by_day(normals):
    """The defect this guards: one request-level month applied to all seven days.

    31 January and 1 February carry the same 30 mm here. Against January's wet P90 of 20 it
    is `>p90`; against February's 60 it is `p50_p90`. A tool that classified the whole
    horizon against one month would return the same band twice.
    """
    result = _call(
        normals,
        values=[
            {"variable": "precipitation", "date": "2026-01-31", "value": 30.0},
            {"variable": "precipitation", "date": "2026-02-01", "value": 30.0},
        ],
    )
    assert "error" not in result, result
    bands = [(entry["day"], entry["percentile_band"]) for entry in result["series"]]
    assert bands == [("2026-01-31", ">p90"), ("2026-02-01", "p50_p90")]


def test_the_tool_takes_no_request_level_month():
    """A month parameter is the shape the design note rules out; it must not exist."""
    assert "month" not in TOOL_CONTRACTS["climate_normals"]["input"].model_fields


def test_an_unparseable_date_is_reported_not_guessed(normals):
    result = _call(normals, values=[{"variable": "precipitation", "date": "31/01/2026"}])
    assert result["series"] == []
    assert result["unclassified"] == [
        {"variable": "precipitation", "date": "31/01/2026", "reason": "date_unparseable"}
    ]


# ---------------------------------------------------------------------------------------
# Precipitation is banded against the wet-day distribution.
# ---------------------------------------------------------------------------------------


def test_precipitation_is_banded_against_the_wet_day_triplet(normals):
    """The two distributions disagree on 9 mm, and the fixture is built so they must.

    Against the all-days percentiles 9 mm clears the P90 of 8 and reads `>p90` /
    `extreme_high`. Against the wet-day distribution it sits between P50 (6) and P90 (20) —
    `p50_p90` / `above_normal`. The all-days P90 is dragged down by every dry day in the
    month, which is exactly why the contract bands precipitation against wet days.
    """
    result = _call(
        normals, values=[{"variable": "precipitation", "date": "2026-01-15", "value": 9.0}]
    )
    entry = result["series"][0]
    assert entry["percentile_band"] == "p50_p90"
    assert entry["category"] == "above_normal"
    assert entry["p90"] == 8.0 and entry["p90_wet"] == 20.0  # both published, band uses the wet one


def test_a_day_below_the_wet_threshold_is_dry_and_gets_no_band(normals):
    """The value is kept; the band is not invented. Dry spells are a V1 derived variable."""
    result = _call(
        normals, values=[{"variable": "precipitation", "date": "2026-01-15", "value": 0.4}]
    )
    entry = result["series"][0]
    assert entry["status"] == "dry"
    assert entry["value"] == 0.4
    assert entry["wet_day_threshold_mm"] == WET_DAY_THRESHOLD_MM
    assert "percentile_band" not in entry and "category" not in entry


def test_exactly_the_threshold_is_a_wet_day(normals):
    """Climdex defines a wet day as ≥ 1.0 mm, so 1.0 itself is wet."""
    result = _call(
        normals,
        values=[{"variable": "precipitation", "date": "2026-01-15", "value": WET_DAY_THRESHOLD_MM}],
    )
    assert result["series"][0]["status"] == "available"


def test_wind_speed_is_banded_against_the_all_days_percentiles_and_carries_no_wet_fields(normals):
    result = _call(
        normals, values=[{"variable": "wind_speed", "date": "2026-01-15", "value": 13.0}]
    )
    entry = result["series"][0]
    assert entry["percentile_band"] == ">p90"
    assert not any(key.endswith("_wet") or key == "wet_day_threshold_mm" for key in entry)


def test_an_arid_month_with_no_wet_days_refuses_to_band_rather_than_substitute(
    tmp_path, allow_local_parquet_paths
):
    """`n_wet = 0` is valid data, not a defect — niamey, January, over 1991-2020.

    There is no wet-day distribution for a forecast day at 5 mm to be placed in. Banding it
    against the all-days percentiles instead would answer a different question while looking
    like an answer to this one, so the entry is reported unclassified with its reason.
    """
    path = Path(allow_local_parquet_paths) / f"{tmp_path.name}-arid.parquet"
    src = _write_normals(
        path,
        [
            _row(
                1,
                "precipitation",
                p10=0.0,
                p50=0.0,
                p90=0.0,
                p10_wet=None,
                p50_wet=None,
                p90_wet=None,
                n_wet=0,
            )
        ],
    )
    try:
        result = _call(
            src, values=[{"variable": "precipitation", "date": "2026-01-15", "value": 5.0}]
        )
        assert result["series"] == []
        assert result["unclassified"][0]["reason"] == "wet_reference_unavailable"
        # And a dry day in that same month still classifies: the month is arid, not unreadable.
        dry = _call(src, values=[{"variable": "precipitation", "date": "2026-01-15", "value": 0.0}])
        assert dry["series"][0]["status"] == "dry"
    finally:
        path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------------------
# Gaps, and what is never a zero.
# ---------------------------------------------------------------------------------------


def test_a_null_value_is_missing_and_never_a_zero(normals):
    """A zero would classify as a real dry day, which is a different fact about the world."""
    result = _call(
        normals, values=[{"variable": "precipitation", "date": "2026-01-15", "value": None}]
    )
    entry = result["series"][0]
    assert entry["status"] == "missing"
    assert "value" not in entry and "percentile_band" not in entry and "category" not in entry


def test_reference_thin_is_computed_for_the_reader(tmp_path, allow_local_parquet_paths):
    path = Path(allow_local_parquet_paths) / f"{tmp_path.name}-thin.parquet"
    src = _write_normals(path, [_row(1, "wind_speed", n=12, p10=2.0, p50=5.0, p90=12.0)])
    try:
        entry = _call(src, values=[{"variable": "wind_speed", "date": "2026-01-15", "value": 6.0}])[
            "series"
        ][0]
        assert entry["n"] == 12 and entry["reference_thin"] is True
    finally:
        path.unlink(missing_ok=True)


def test_a_variable_outside_the_watch_contract_never_enters_the_series(
    tmp_path, allow_local_parquet_paths
):
    """`watch-input/v1` enumerates two variables; emitting a third would be schema-invalid."""
    path = Path(allow_local_parquet_paths) / f"{tmp_path.name}-temp.parquet"
    src = _write_normals(path, [_row(1, "temperature_2m", p10=2.0, p50=8.0, p90=15.0)])
    try:
        result = _call(
            src, values=[{"variable": "temperature_2m", "date": "2026-01-15", "value": 20.0}]
        )
        assert result["series"] == []
        assert result["unclassified"][0]["reason"] == "variable_outside_watch_contract"
        # The normals themselves are still returned — the restriction is on the classification.
        assert result["normals"][0]["variable"] == "temperature_2m"
        assert set(WATCH_SERIES_VARIABLES) == {"precipitation", "wind_speed"}
    finally:
        path.unlink(missing_ok=True)


def test_an_inconsistent_stored_reference_is_reported_not_raised():
    """`classify_against_normals` raises on p10 > p50; a stored file must not crash the tool."""
    from datetime import date

    entry, reason = classify_watch_entry(
        variable="wind_speed",
        day=date(2026, 1, 15),
        value=6.0,
        normal={"p10": 9.0, "p50": 5.0, "p90": 12.0, "n": 900},
    )
    assert entry is None and reason == "reference_inconsistent"


def test_a_month_with_no_stored_normals_is_reported(normals):
    result = _call(
        normals, values=[{"variable": "precipitation", "date": "2026-07-15", "value": 5.0}]
    )
    assert result["unclassified"][0]["reason"] == "reference_unavailable"


# ---------------------------------------------------------------------------------------
# The committed contract.
# ---------------------------------------------------------------------------------------


def test_every_series_entry_validates_against_the_committed_schema(normals):
    """Against `src/contracts_data/watch/watch-input.v1.schema.json` itself.

    That file is byte-identical to the Deposium reference copy, so this is the contract both
    sides hold and not a local restatement of it.
    """
    result = _call(
        normals,
        values=[
            {"variable": "precipitation", "date": "2026-01-15", "value": 9.0},
            {"variable": "precipitation", "date": "2026-01-16", "value": 0.2},
            {"variable": "precipitation", "date": "2026-02-01", "value": None},
            {"variable": "wind_speed", "date": "2026-01-15", "value": 13.0},
            {"variable": "wind_speed", "date": "2026-02-02", "value": None},
        ],
    )
    assert len(result["series"]) == 5
    for entry in result["series"]:
        _ENTRY_VALIDATOR.validate(entry)
    assert {entry["status"] for entry in result["series"]} == {"available", "dry", "missing"}


#: The `watch-input/v1` fields the ADAPTER (Deposium) supplies: the tool reads stored normals and
#: fetches no forecast, so it cannot know them. Stubbed here on purpose;
#: `test_the_forecast_side_provenance_is_null_with_a_reason` pins that the tool does not answer
#: them itself.
_ADAPTER_OWNED_STUBS = {
    "schema_version": "watch-input/v1",
    "provider": {
        "id": "deposium_geoai",
        "tools": ["geo_forecast_query", "climate_normals"],
        "model": "ecmwf_aifs025_single",
        "provider_chain": ["openmeteo"],
        # `licence` and `attribution` are overwritten from the tool's metadata.
        "licence": "",
        "attribution": "",
    },
    "retrieved_at": "2026-01-14T06:00:00Z",
    "model_run_at": None,
    "horizon_days_requested": 7,
    "horizon_days_served": 7,
    "evidence_ids": [],
}
_ONE_DAY = [{"variable": "precipitation", "date": "2026-01-15", "value": 9.0}]


def _assemble_watch_input(result: dict) -> dict:
    """Build a `watch-input/v1` document, taking from the tool output every field it maps to.

    Copied from ``result``: ``place.ref`` (from ``zone_id``), ``day_grain``, ``confidence``,
    ``series``, ``caveats``, ``reference`` (plus a PLACEHOLDER ``evidence_id``, #223 part C),
    ``provider.licence`` and ``provider.attribution``. Everything else is an adapter stub.
    """
    document = json.loads(json.dumps(_ADAPTER_OWNED_STUBS))
    document["provider"]["licence"] = result["metadata"]["license"]
    document["provider"]["attribution"] = result["metadata"]["provenance"]["attribution"]
    document.update(
        {
            "place": {"ref": f"zone:{result['zone_id']}", "sampling": "bbox_midpoint"},
            "day_grain": result["metadata"]["day_grain"],
            "confidence": result["confidence"],
            "series": result["series"],
            "caveats": result["caveats"],
            # The one field the tool deliberately does not mint - see `_normals_reference`.
            "reference": {**result["reference"], "evidence_id": "geoai_ev_" + "0" * 64},
        }
    )
    return document


def test_the_tool_output_drops_into_a_valid_watch_input_document(normals):
    """Proving the entries validate one by one is not the same as proving the document does.

    The tool's own output is NOT a `watch-input/v1` document: it also carries `zone_id`,
    `normals`, `unclassified`, `statistics` and `metadata`, which the schema (top level
    `additionalProperties: false`) forbids. What is validated is the assembled document: each
    of its top-level fields is either copied from the tool output or a named adapter stub, so
    the schema's whole `required` list is covered and none is left unchecked.

    This fixture writes the window label `1991-2020`. The default producer instead writes
    `WMO_1991_2020`; the tool now derives `1991-2020` from the stored window dates in the
    reference block (#244, repaired in #248). The next test exercises that default-label path with
    complete window dates. Neither fixture proves registration or a real evidence identifier.
    """
    result = _call(normals, values=_ONE_DAY)
    document = _assemble_watch_input(result)
    _DOCUMENT_VALIDATOR.validate(document)
    assert set(document) == set(_SCHEMA["required"])  # nothing required is left unassembled
    assert document["reference"]["window"] == "1991-2020"
    assert document["provider"]["attribution"] == OPEN_METEO_DATA_LICENSE
    assert len(document["caveats"]) == 8


def test_a_document_from_the_default_producer_label_is_valid(normals_default_label):
    """The same check as above, on an artifact labelled the way the producer labels it."""
    result = _call(normals_default_label, values=_ONE_DAY)
    _DOCUMENT_VALIDATOR.validate(_assemble_watch_input(result))


def test_the_document_check_rejects_what_it_claims_to_guard(normals):
    """The validator is live: each mutation is refused by the committed schema itself."""
    document = _assemble_watch_input(_call(normals, values=_ONE_DAY))
    mutations = [
        lambda d: d.update(confidence=0.8),
        lambda d: d.pop("confidence"),
        lambda d: d["reference"].update(window_start="1991-01-01"),
        lambda d: d.update(probability=0.4),
        lambda d: d.update(series=[]),
        lambda d: d["provider"].pop("attribution"),
    ]
    for mutate in mutations:
        broken = json.loads(json.dumps(document))
        mutate(broken)
        with pytest.raises(ValidationError):
            _DOCUMENT_VALIDATOR.validate(broken)


def test_every_top_level_key_is_a_schema_field_or_a_declared_tool_extra(normals):
    """Provenance cannot sit on a top-level key the schema forbids: it lives in `metadata`."""
    result = _call(normals, values=_ONE_DAY)
    tool_extras = {"zone_id", "normals", "unclassified", "statistics", "metadata"}
    assert set(result) - set(_SCHEMA["properties"]) == tool_extras
    assert "provenance" in result["metadata"]


def test_the_reference_block_is_exactly_watch_input_v1_minus_evidence_id(normals):
    """`reference` is `watch-input/v1`'s `reference` object minus `evidence_id` — nothing more.

    The schema's `reference` declares `additionalProperties: false`, so stored
    `window_start`/`window_end` riding along on the block (as opposed to living in
    `metadata.normals_window`, where a reader who wants them can still find them) would make
    the whole block schema-invalid rather than merely carry two harmless extra fields
    (#218, #172).
    """
    result = _call(normals)
    reference = {**result["reference"], "evidence_id": "geoai_ev_" + "0" * 64}
    _REFERENCE_VALIDATOR.validate(reference)
    assert result["metadata"]["normals_window"] == {"start": "1991-01-01", "end": "2020-12-31"}


def test_confidence_is_null_and_cannot_be_a_number(normals):
    """Asserted, not left to convention — and structural: the model rejects a number."""
    result = _call(normals)
    assert "confidence" in result and result["confidence"] is None
    ClimateNormalsOutput.model_validate(result)
    with pytest.raises(Exception, match="confidence"):
        ClimateNormalsOutput.model_validate({**result, "confidence": 0.8})


def test_no_probability_risk_or_severity_field_exists(normals):
    """ "by construction", from the design note. Checked on the model, not only the payload."""
    forbidden = {"probability", "risk", "severity"}
    assert not forbidden & set(ClimateNormalsOutput.model_fields)
    assert not forbidden & set(_call(normals))


def _design_note_data_limit_titles() -> list[str]:
    """Expected public caveat titles, preserved independently of the runtime output."""
    assert len(_DATA_LIMIT_TITLES) == 8
    assert all(isinstance(title, str) and title for title in _DATA_LIMIT_TITLES)
    return _DATA_LIMIT_TITLES


#: What must never reach a reader of the tool output. `re.search` patterns, checked on every
#: caveat: markdown (a backtick, `**`), a file path (`src/`, `dir/file.ext`, `.py`), an issue
#: number, a snake_case or ALL_CAPS identifier, and a call written `name()`.
_LEAKS = (
    r"`",
    r"\*\*",
    r"src/",
    r"\.py",
    r"#\d",
    r"[\w-]+/[\w./-]*\.[A-Za-z]{1,5}\b",
    r"\b[A-Za-z0-9]+(?:_[A-Za-z0-9]+)+\b",
    r"\w\(\)",
)


def test_the_caveats_state_the_design_note_data_limits_as_plain_text_entries():
    """One caveat per bullet, same order, each opening with that bullet's title.

    The titles are preserved in an independent JSON fixture: adding, renaming or dropping
    a runtime caveat fails this check. The body is the output's own plain wording, so it is
    checked for leaks (`_LEAKS`); the key wording is pinned in the tests below.
    """
    titles = _design_note_data_limit_titles()
    assert len(titles) == 8
    assert len(NORMALS_CAVEATS) == len(titles)
    for title, caveat in zip(titles, NORMALS_CAVEATS, strict=True):
        assert caveat.startswith(f"{title}: "), (title, caveat[:60])
        assert len(caveat) > len(title) + 20, title  # a statement, not a bare title
        for leak in _LEAKS:
            assert not re.search(leak, caveat), (title, leak)


def test_the_leak_patterns_catch_what_they_name():
    """Each pattern is live: it matches an example of the thing it is there to keep out."""
    examples = [
        "see `forecast_query`",
        "**bold**",
        "src/tools/x",
        "tool.py",
        "issue #172",
        "docs/design/site-watch-v0.md",
        "NORMALS_CAVEATS applies",
        "build_watch_series applies",
        "build() applies",
    ]
    for example in examples:
        assert any(re.search(leak, example) for leak in _LEAKS), example
    plain = "Grid, not site: P10/P50/P90 (CET/CEST) of at least 1.0 mm on 10 m wind, 1991-2020."
    assert not any(re.search(leak, plain) for leak in _LEAKS)


def _caveat(title: str) -> str:
    return next(c for c in NORMALS_CAVEATS if c.startswith(f"{title}: "))


def test_the_forecast_path_caveats_hold_in_live_use_and_in_replay():
    """The offline replay report copies these caveats, and a replay is not a live watch.

    A replay reads archived forecasts, one per lead day 1 to 7, and stamps no retrieval time. So
    a limit of the live contract (a 7-day request, an adapter-stamped `retrieved_at`) must be
    stated as one: "In live watch use", never as a fact about every consumer of the text
    (PR review, #172).
    """
    horizon = _caveat("Horizon 7 days requested, served horizon reported").split(": ", 1)[1]
    assert horizon.startswith("In live watch use, a forecast is requested for 7 days")
    assert "replay" in horizon and "archived forecasts" in horizon and "lead day" in horizon
    timestamps = _caveat("Two timestamps, no evidence id").split(": ", 1)[1]
    assert timestamps.startswith("In live watch use")
    assert "replay" in timestamps
    # Every "7 days" in the horizon caveat sits after the live qualifier, none before it.
    assert horizon.index("In live watch use") < horizon.index("7 days")
    # True in both outputs: the replay report publishes `coverage.horizons_served` itself.
    assert "does not know the served horizon" not in horizon


def test_the_caveats_pin_the_wording_that_changes_their_meaning():
    """Edits that keep the title and stay plain, but say something else, are caught here."""
    grid = _caveat("Grid, not site")
    assert "in the live watch contract" in grid
    skill = _caveat("No measured skill")
    assert "confidence is null" in skill
    assert not re.search(r"\d", skill)  # no number beside a skill statement
    monthly = _caveat("Monthly normals on daily values")
    assert f"at least {WET_DAY_THRESHOLD_MM} mm" in monthly
    assert "1991-2020" in monthly


def test_the_caveats_travel_in_the_output(normals):
    caveats = _call(normals)["caveats"]
    assert caveats == list(NORMALS_CAVEATS)
    assert [c.split(": ", 1)[0] for c in caveats] == _design_note_data_limit_titles()
    # The two limits the output lacked before #172: the served horizon and the timestamps.
    assert any(c.startswith("Horizon 7 days requested, served horizon reported") for c in caveats)
    assert any(c.startswith("Two timestamps, no evidence id") for c in caveats)


# ---------------------------------------------------------------------------------------
# `confidence: null` and the provenance, on every handler return.
# ---------------------------------------------------------------------------------------


def _every_handler_return(normals):
    """Each dict the handler returns, as ``(label, result)``: one success or refusal per site.

    Not covered, because it is not a handler return: an input-contract violation (an empty
    ``zone_id``, a ``values`` item with an extra key) leaves as an MCP tool error with no
    payload, so with no ``confidence`` key.
    """
    absent = normals.replace(".parquet", "-absent.parquet")
    yield "no values", _call(normals)
    yield "empty values", _call(normals, values=[])
    yield "classified values", _call(normals, values=_ONE_DAY)
    yield (
        "unclassified only",
        _call(normals, values=[{"variable": "precipitation", "date": "2026-03-01", "value": 4.0}]),
    )
    yield "empty variables", _call(normals, variables=[])
    yield "unknown variable", _call(normals, variables=["rainfall"])
    yield "value outside narrowing", _call(normals, variables=["wind_speed"], values=_ONE_DAY)
    yield "no source configured", _call(None)
    yield "unreadable artifact", _call(absent)
    yield "zone without rows", _call(normals, zone_id="not_a_zone")
    with mock.patch.object(climate, "build_watch_series", side_effect=ValueError("boom")):
        yield "build failure", _call(normals, values=_ONE_DAY)


def test_confidence_is_null_on_every_handler_return(normals):
    """Success, empty and every refusal: the null is asserted, not left to convention.

    A refusal is not a finding, but a consumer that reads ``confidence`` must never meet a
    payload where the key is absent and a default could be invented for it.
    """
    outcomes = dict(_every_handler_return(normals))
    assert len(outcomes) == 11
    errors = [result["error"] for result in outcomes.values() if "error" in result]
    assert len(errors) == 7
    # Seven different refusals: a guard that stops firing would fall through to another one and
    # repeat its message, which a bare count of errors would not notice.
    assert len(set(errors)) == 7
    for label, result in outcomes.items():
        assert "confidence" in result, label
        assert result["confidence"] is None, label


def test_every_handler_return_matches_the_wired_output_schema(normals):
    """The schema wired into the tool (`output_schema_for`) accepts each return.

    That is the lenient schema FastMCP validates against, not the published contract
    (`contracts_json_schema()`), which does not declare `confidence` on the error variant and
    accepts `{"error": "x", "confidence": 0.8}`.
    """
    validator = Draft202012Validator(output_schema_for("climate_normals"))
    for _label, result in _every_handler_return(normals):
        validator.validate(result)
    # Live, not vacuous: the wired schema refuses a number where the null belongs.
    with pytest.raises(ValidationError):
        validator.validate({"error": "x", "confidence": 0.8})


_FORECAST_SIDE_FIELDS = (
    "provider_chain",
    "horizon_days_requested",
    "horizon_days_served",
    "model_run_at",
    "retrieved_at",
)


def test_the_provenance_the_tool_can_fill_is_in_metadata(normals):
    """Window and attribution come from what the tool actually read or already publishes."""
    metadata = _call(normals, values=_ONE_DAY)["metadata"]
    assert metadata["provenance"]["attribution"] == OPEN_METEO_DATA_LICENSE
    assert metadata["provenance"]["attribution"] == metadata["license"]
    assert metadata["normals_window"] == {"start": "1991-01-01", "end": "2020-12-31"}
    assert metadata["provider_map"]["precipitation"]["provider"] == "precipitation_sum"


def test_the_forecast_side_provenance_is_null_with_a_reason(normals):
    """`values[]` come from the caller; the tool fetches no forecast, so it must not guess.

    The provider chain, the requested and served horizons and the two timestamps describe a
    forecast fetch. Deriving a horizon from ``len(values)`` would read as the provider's served
    horizon when it is only what the caller chose to send. They are explicit nulls, each with
    the reason, with and without ``values``.
    """
    # Each of them is a key of this test module's own adapter stub table (which the assembled
    # document test checks against the schema's `required` list); the schema is not read here.
    adapter_fields = set(_ADAPTER_OWNED_STUBS) | set(_ADAPTER_OWNED_STUBS["provider"])
    assert set(_FORECAST_SIDE_FIELDS) <= adapter_fields
    for values in (None, _ONE_DAY):
        forecast = _call(normals, values=values)["metadata"]["provenance"]["forecast"]
        assert set(forecast) == {*_FORECAST_SIDE_FIELDS, "reason"}
        assert all(forecast[field] is None for field in _FORECAST_SIDE_FIELDS)
        assert "supplied by the caller" in forecast["reason"]
        assert "fetches no forecast" in forecast["reason"]
        # Where each field really comes from (it is not all `forecast_query`):
        reason = forecast["reason"]
        assert "provider_chain and horizon_days_served come from forecast_query metadata" in reason
        assert "horizon_days_requested is the adapter's own request" in reason
        assert "retrieved_at is stamped by the adapter after the fetch" in reason
        assert "model_run_at stays null until a provider exposes it" in reason
        assert "carries these from" not in reason


# ---------------------------------------------------------------------------------------
# The read-only boundary and the stored artifact.
# ---------------------------------------------------------------------------------------


def test_the_tool_is_registered_read_only_with_no_path_in_or_out():
    """`science_shared_read` is "an allowlisted dataset through a stable alias".

    An earlier revision took a `normals_source` path by analogy with `climate_enrich_ndvi`'s
    `ndvi_source`. The analogy does not hold: that one sits on an `internal_write` operator
    tool, where a physical locator is the point. On a facade read it lets a caller select a
    sibling artifact inside the same storage prefix, which is what the profile promises
    callers cannot do (#203 review).
    """
    capability = TOOL_CAPABILITIES["climate_normals"]
    assert capability.profile == "science_shared_read"
    assert capability.timeout_class == "short"
    fields = set(TOOL_CONTRACTS["climate_normals"]["input"].model_fields)
    assert not fields & {"write_output", "output_path", "overwrite"}
    assert not fields & {"normals_source", "source", "path"}
    assert fields == {"zone_id", "variables", "values"}


def test_narrowing_the_variables_refuses_a_value_outside_them(normals):
    """The defect: the narrowing removes the reference, and the answer then calls it absent.

    `variables=["wind_speed"]` with a precipitation value used to report
    `reference_unavailable` — a false statement about the stored corpus, produced by the
    request's own filter (#203 review).
    """
    result = _call(
        normals,
        variables=["wind_speed"],
        values=[{"variable": "precipitation", "date": "2026-01-15", "value": 9.0}],
    )
    assert "error" in result
    assert "outside the requested set" in result["error"]
    assert "precipitation" in result["error"]


def test_narrowing_is_fine_when_the_values_stay_inside_it(normals):
    result = _call(
        normals,
        variables=["wind_speed"],
        values=[{"variable": "wind_speed", "date": "2026-01-15", "value": 13.0}],
    )
    assert result["series"][0]["percentile_band"] == ">p90"
    assert {row["variable"] for row in result["normals"]} == {"wind_speed"}


def test_the_published_output_schema_documents_the_refusals_and_counts():
    """`unclassified` and `statistics` were undeclared extras, so the published schema
    described a payload the tool does not return (#203 review)."""
    fields = set(TOOL_CONTRACTS["climate_normals"]["output"].model_fields)
    assert {"unclassified", "statistics", "reference", "series", "caveats"} <= fields
    published = TOOL_CONTRACTS["climate_normals"]["output"].model_json_schema()
    assert "unclassified" in published["properties"]
    assert "statistics" in published["properties"]


def test_the_reference_block_is_typed_and_rejects_a_foreign_source():
    """The one part of the cross-repo contract the tool fills in wholesale."""
    from src.contracts import ClimateNormalsReference

    reference = ClimateNormalsReference(grain="daily_values_by_calendar_month", window="1991-2020")
    assert reference.kind == "climatological_percentiles"
    assert "evidence_id" not in ClimateNormalsReference.model_fields
    with pytest.raises(Exception, match="source"):
        ClimateNormalsReference(grain="g", source="somewhere_else")


@pytest.mark.parametrize(
    "broken",
    [
        {"month": "January"},
        {"month": None},
        {"month": 13},
        {"n": "many"},
        {"schema_version": "v2"},
    ],
)
def test_a_malformed_stored_value_does_not_crash_the_tool(
    tmp_path, allow_local_parquet_paths, broken
):
    """A readable artifact with a bad column must produce an answer, not a traceback.

    Only the Parquet read used to sit inside the guard; `int(row["month"])` and friends ran
    after it, from inside the response builder (#203 review).
    """
    path = Path(allow_local_parquet_paths) / f"{tmp_path.name}-broken.parquet"
    columns = tuple(
        (name, "VARCHAR" if name in broken and isinstance(broken[name], str) else sql_type)
        for name, sql_type in _COLUMNS
    )
    row = _row(1, "wind_speed", p10=2.0, p50=5.0, p90=12.0)
    row.update(broken)
    src = _write_normals(path, [row], columns=columns)
    try:
        result = _call(
            src, values=[{"variable": "wind_speed", "date": "2026-01-15", "value": 13.0}]
        )
        # Either a structured refusal or a well-formed answer that reports the gap — never a
        # crash, and never a band built from a value nobody could read.
        if "error" not in result:
            assert result["series"] == [] or result["series"][0]["n"] >= 0
    finally:
        path.unlink(missing_ok=True)


#: An integer past the float range: `float()` raises `OverflowError` on it (#221).
_BEYOND_FLOAT = 10**400


@pytest.mark.parametrize(
    ("variable", "field", "out_of_range"),
    [
        pytest.param("wind_speed", "month", math.inf, id="month-inf"),
        pytest.param("wind_speed", "n", math.inf, id="n-inf"),
        pytest.param("precipitation", "n_wet", math.inf, id="n_wet-inf"),
        pytest.param("wind_speed", "p10", _BEYOND_FLOAT, id="p10-huge"),
        pytest.param("wind_speed", "p50", _BEYOND_FLOAT, id="p50-huge"),
        pytest.param("wind_speed", "p90", _BEYOND_FLOAT, id="p90-huge"),
        pytest.param("precipitation", "p10_wet", _BEYOND_FLOAT, id="p10_wet-huge"),
        pytest.param("precipitation", "p50_wet", _BEYOND_FLOAT, id="p50_wet-huge"),
        pytest.param("precipitation", "p90_wet", _BEYOND_FLOAT, id="p90_wet-huge"),
        pytest.param("precipitation", "wet_day_threshold_mm", _BEYOND_FLOAT, id="threshold-huge"),
        pytest.param("wind_speed", "value", _BEYOND_FLOAT, id="wind-value-huge"),
        pytest.param("precipitation", "value", _BEYOND_FLOAT, id="precipitation-value-huge"),
    ],
)
def test_an_out_of_range_number_reads_exactly_like_a_nan(variable, field, out_of_range):
    """`inf` and an integer past the float range are no more usable than `NaN` (#221).

    `NaN` in any of these fields already read as "no usable value". `int(inf)` and
    `float(10**400)` raise `OverflowError`, not `ValueError`, and `_stored_int` and `_finite`
    caught only the latter, so the same unusable number crashed the evaluator instead. The
    whole return value is compared: answering *something* is not enough.
    """

    def evaluate(**replaced):
        if variable == "precipitation":
            # A wet day, so the wet-day triplet and the threshold are both read.
            normal, value = _row(1, "precipitation"), 9.0
        else:
            normal, value = _row(1, "wind_speed", p10=2.0, p50=5.0, p90=12.0), 13.0
        value = replaced.pop("value", value)
        normal.update(replaced)
        return build_watch_series(
            [{"variable": variable, "date": "2026-01-15", "value": value}], normals=[normal]
        )

    as_nan = evaluate(**{field: math.nan})
    assert evaluate(**{field: out_of_range}) == as_nan
    # The field is one the answer depends on, so the comparison above is not vacuous.
    assert evaluate() != as_nan


@pytest.mark.parametrize("field", ["month", "n", "n_wet", "schema_version"])
def test_a_stored_inf_degrades_like_a_stored_nan_instead_of_refusing_the_panel(
    tmp_path, allow_local_parquet_paths, field
):
    """One stored `inf` refused the whole panel as "unusable"; `NaN` costs a row at most (#221).

    The guard around the response builder caught the `OverflowError`, so the tool did not
    crash, but every entry was lost to one cell. The column is declared DOUBLE: it is the only
    way a Parquet file holds `inf` in a field the writer declares INTEGER.
    """
    columns = tuple((name, "DOUBLE" if name == field else sql_type) for name, sql_type in _COLUMNS)
    values = [
        {"variable": "precipitation", "date": "2026-01-15", "value": 9.0},
        {"variable": "wind_speed", "date": "2026-01-15", "value": 13.0},
    ]

    def answer(number, label):
        rows = [_row(1, "precipitation"), _row(1, "wind_speed", p10=2.0, p50=5.0, p90=12.0)]
        if field == "month":
            # An extra row, so the rows the values need are still there to answer with.
            rows.append(_row(number, "wind_speed", p10=2.0, p50=5.0, p90=12.0))
        else:
            rows[0][field] = number
        path = Path(allow_local_parquet_paths) / f"{tmp_path.name}-{field}-{label}.parquet"
        try:
            return _call(_write_normals(path, rows, columns=columns), values=values)
        finally:
            path.unlink(missing_ok=True)

    def comparable(result):
        # The stored cell is echoed back as read (inf or NaN). Byte identity necessarily
        # differs between these artifacts even though their usable semantics are equal.
        return {
            **result,
            "normals": [{k: v for k, v in row.items() if k != field} for row in result["normals"]],
            "metadata": {
                k: v
                for k, v in result["metadata"].items()
                if k not in {"elapsed_ms", "artifact_version", "artifact_size_bytes"}
            },
        }

    as_inf, as_nan = answer(math.inf, "inf"), answer(math.nan, "nan")

    assert "error" not in as_inf, as_inf
    assert "error" not in as_nan, as_nan
    assert len(as_inf["series"]) == 2  # the panel is answered, not refused
    for result in (as_inf, as_nan):
        assert re.fullmatch(r"sha256:[0-9a-f]{64}", result["metadata"]["artifact_version"])
        assert result["metadata"]["artifact_size_bytes"] > 0
    assert as_inf["metadata"]["artifact_version"] != as_nan["metadata"]["artifact_version"]
    assert comparable(as_inf) == comparable(as_nan)


def test_handler_rows_band_and_version_use_capture_before_canonical_replacement(
    monkeypatch, tmp_path, allow_local_parquet_paths
):
    from src.utils import normals_reader

    source = Path(allow_local_parquet_paths) / f"{tmp_path.name}-canonical.parquet"
    replacement = Path(allow_local_parquet_paths) / f"{tmp_path.name}-replacement.parquet"
    _write_normals(source, [_row(2, "precipitation", p10_wet=2, p50_wet=6, p90_wet=20)])
    _write_normals(replacement, [_row(2, "precipitation", p10_wet=30, p50_wet=40, p90_wet=60)])
    original_bytes = source.read_bytes()
    replacement_bytes = replacement.read_bytes()
    capture = normals_reader._read_source_bytes
    captures = []

    def capture_then_replace(src, *, zone_id):
        content = capture(src, zone_id=zone_id)
        captures.append(content)
        replacement.replace(source)
        return content

    monkeypatch.setattr(normals_reader, "_read_source_bytes", capture_then_replace)
    try:
        result = _call(
            str(source),
            values=[{"variable": "precipitation", "date": "2026-02-01", "value": 12}],
        )
        assert "error" not in result, result
        assert result["normals"][0]["p50_wet"] == 6
        assert result["series"][0]["percentile_band"] == "p50_p90"
        assert captures == [original_bytes]
        assert source.read_bytes() == replacement_bytes
        assert result["metadata"]["artifact_version"] == (
            "sha256:" + hashlib.sha256(original_bytes).hexdigest()
        )
        assert result["metadata"]["artifact_size_bytes"] == len(original_bytes)
        assert result["metadata"]["dataset"] == "science:normals:montpellier"
        assert "artifact_version" not in result["reference"]
        _REFERENCE_VALIDATOR.validate(
            {**result["reference"], "evidence_id": "geoai_ev_" + "0" * 64}
        )
    finally:
        source.unlink(missing_ok=True)
        replacement.unlink(missing_ok=True)


def test_handler_filtering_preserves_whole_artifact_identity(normals):
    content = Path(normals).read_bytes()
    all_rows = _call(normals)
    subset = _call(normals, variables=["precipitation"])
    assert len(all_rows["normals"]) == 4 and len(subset["normals"]) == 2
    for result in (all_rows, subset):
        assert "error" not in result, result
        assert (
            result["metadata"]["artifact_version"]
            == "sha256:" + hashlib.sha256(content).hexdigest()
        )
        assert result["metadata"]["artifact_size_bytes"] == len(content)


def test_changed_artifact_bytes_change_handler_version(normals):
    first = _call(normals)
    path = Path(normals)
    path.unlink()
    _write_normals(path, _fixture_rows(window_label="new-label"))
    second = _call(normals)
    assert "error" not in first and "error" not in second
    assert first["metadata"]["artifact_version"] != second["metadata"]["artifact_version"]
    assert second["metadata"]["artifact_version"] == (
        "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
    )


def test_store_error_is_a_redacted_refusal_with_no_unbound_fallback(monkeypatch):
    from src.config import settings
    from src.utils import normals_reader

    endpoint = "https://private-normals-endpoint.invalid:9000"
    secret = "normals-private-secret-test"
    monkeypatch.setattr(settings, "S3_BUCKET", "normals-test")
    monkeypatch.setattr(settings, "S3_ENDPOINT", endpoint)
    monkeypatch.setattr(settings, "S3_SECRET_KEY", secret)

    def fail_store():
        raise OSError(f"{endpoint}/climate/normals/montpellier.parquet {secret}")

    monkeypatch.setattr(normals_reader, "_object_store_client", fail_store)
    monkeypatch.setattr(
        climate, "_read_normals_rows", lambda *_args, **_kwargs: pytest.fail("unbound fallback")
    )
    with mock.patch.object(climate.log, "error") as logged:
        result = _call("s3://normals-test/climate/normals/montpellier.parquet")
    assert "error" in result and result["confidence"] is None
    logged.assert_called_once_with(
        "climate_normals_read_error",
        zone_id="montpellier",
        error_type="OSError",
        reason="source_read_failed",
    )
    serialized = json.dumps(result) + str(logged.call_args_list)
    for private in (endpoint, secret, "climate/normals/montpellier.parquet", "s3://"):
        assert private not in serialized
    Draft202012Validator(output_schema_for("climate_normals")).validate(result)


@pytest.mark.parametrize(
    ("scenario", "reason"),
    [
        ("declared_oversize", "source_too_large"),
        ("actual_oversize", "source_too_large"),
        ("length_mismatch", "incomplete_download"),
        ("botocore_incomplete", "incomplete_download"),
        ("invalid_parquet", "parquet_read_failed"),
        ("local_oversize", "source_too_large"),
    ],
)
def test_bound_read_logs_distinct_safe_reasons_but_same_public_refusal(
    monkeypatch, tmp_path, scenario, reason
):
    import io

    from botocore.exceptions import IncompleteReadError

    from src.config import settings
    from src.utils import normals_reader

    endpoint = "https://private-normals-reason.invalid:9000"
    secret = "private-normals-reason-secret"
    payload = f"{endpoint}/climate/normals/montpellier.parquet {secret}".encode()
    monkeypatch.setattr(settings, "S3_BUCKET", "normals-test")
    monkeypatch.setattr(settings, "S3_ENDPOINT", endpoint)
    monkeypatch.setattr(settings, "S3_SECRET_KEY", secret)
    # A small test cap exercises the actual-byte refusal without allocating MiBs.
    monkeypatch.setattr(normals_reader, "MAX_NORMALS_ARTIFACT_BYTES", 1024)
    stream = io.BytesIO(payload)
    if scenario == "actual_oversize":
        stream = io.BytesIO(payload + b"x" * 1024)
    body = mock.Mock(wraps=stream)
    if scenario == "botocore_incomplete":
        # The real SDK exception can arise before our byte-count comparison. Its message
        # is deliberately tainted, so retaining its class must not publish its text.
        body.read.side_effect = IncompleteReadError(actual_bytes=endpoint, expected_bytes=secret)
    declared = len(payload)
    if scenario == "declared_oversize":
        declared = 1025
    elif scenario == "length_mismatch":
        declared += 1
    client = mock.Mock()
    client.get_object.return_value = {"Body": body, "ContentLength": declared}
    monkeypatch.setattr(normals_reader, "_object_store_client", lambda: client)
    source = "s3://normals-test/climate/normals/montpellier.parquet"
    if scenario == "local_oversize":
        path = tmp_path / "normals.parquet"
        with path.open("wb") as local:
            local.truncate(1025)
        monkeypatch.setattr(settings, "ALLOW_LOCAL_PARQUET_SOURCE", True)
        monkeypatch.setattr(settings, "LOCAL_PARQUET_SOURCE_ROOT", str(tmp_path))
        source = str(path)
    with mock.patch.object(climate.log, "error") as logged:
        result = _call(source)
    assert result == {
        "error": "Could not read the stored normals for zone 'montpellier'",
        "confidence": None,
    }
    logged.assert_called_once_with(
        "climate_normals_read_error",
        zone_id="montpellier",
        error_type="NormalsReadError",
        reason=reason,
    )
    serialized = json.dumps(result) + str(logged.call_args_list)
    for private in (
        endpoint,
        secret,
        source,
        "climate/normals/montpellier.parquet",
        "/proc/self/fd/",
    ):
        assert private not in serialized
    if scenario == "local_oversize":
        client.get_object.assert_not_called()
        stream.close()
    else:
        client.get_object.assert_called_once_with(
            Bucket="normals-test", Key="climate/normals/montpellier.parquet"
        )
        body.close.assert_called_once()
        client.close.assert_called_once()
        assert stream.closed
    Draft202012Validator(output_schema_for("climate_normals")).validate(result)


def test_normals_only_returns_the_requested_variables(normals):
    result = _call(normals, variables=["wind_speed"])
    assert {row["variable"] for row in result["normals"]} == {"wind_speed"}
    assert result["statistics"]["n_variables"] == 1


def test_an_explicitly_empty_variable_list_is_refused(normals):
    """Silently widening a caller's narrowing is how a facade returns more than was asked."""
    assert "error" in _call(normals, variables=[])


def test_an_unknown_variable_is_refused(normals):
    assert "Unknown climate variable" in _call(normals, variables=["rainfall"])["error"]


def test_a_missing_artifact_is_an_error_not_an_empty_success(tmp_path, allow_local_parquet_paths):
    absent = (Path(allow_local_parquet_paths) / f"{tmp_path.name}-absent.parquet").as_posix()
    result = _call(absent)
    assert "error" in result
    assert absent not in result["error"]
    assert "read_parquet" not in result["error"]


def test_a_zone_with_no_rows_is_an_error(normals):
    result = _call(normals, zone_id="not_a_zone")
    assert "No stored normals" in result["error"]


def test_a_v1_artifact_without_wet_columns_still_reads(tmp_path, allow_local_parquet_paths):
    """Naming the v2 columns in the SELECT would raise on exactly the files v1 support is for."""
    v1_columns = tuple(
        column
        for column in _COLUMNS
        if column[0]
        not in {"p10_wet", "p50_wet", "p90_wet", "n_wet", "wet_day_threshold_mm", "schema_version"}
    )
    path = Path(allow_local_parquet_paths) / f"{tmp_path.name}-v1.parquet"
    src = _write_normals(
        path, [_row(1, "wind_speed", p10=2.0, p50=5.0, p90=12.0)], columns=v1_columns
    )
    try:
        result = _call(
            src, values=[{"variable": "wind_speed", "date": "2026-01-15", "value": 13.0}]
        )
        assert "error" not in result, result
        assert result["metadata"]["normals_schema_version"] == [1]
        assert result["series"][0]["percentile_band"] == ">p90"
    finally:
        path.unlink(missing_ok=True)


def test_a_v1_missing_precipitation_day_does_not_invent_n_wet():
    from datetime import date

    entry, reason = classify_watch_entry(
        variable="precipitation",
        day=date(2026, 1, 5),
        value=None,
        normal={"p10": 0.0, "p50": 0.2, "p90": 6.0, "n": 900},  # v1: no wet-day columns
    )
    assert reason is None
    assert entry["status"] == "missing"
    assert "n_wet" not in entry


def test_precipitation_on_a_v1_artifact_cannot_be_banded(tmp_path, allow_local_parquet_paths):
    """v1 has no wet-day triplet, and the contract bands precipitation against it.

    New-materialization support is not proof that the stored corpus was migrated (#172), so
    the honest answer on a v1 file is a reason, not a band from the all-days percentiles.
    """
    v1_columns = tuple(
        column
        for column in _COLUMNS
        if column[0]
        not in {"p10_wet", "p50_wet", "p90_wet", "n_wet", "wet_day_threshold_mm", "schema_version"}
    )
    path = Path(allow_local_parquet_paths) / f"{tmp_path.name}-v1p.parquet"
    src = _write_normals(path, [_row(1, "precipitation")], columns=v1_columns)
    try:
        result = _call(
            src, values=[{"variable": "precipitation", "date": "2026-01-15", "value": 9.0}]
        )
        assert result["series"] == []
        assert result["unclassified"][0]["reason"] == "reference_unavailable"
    finally:
        path.unlink(missing_ok=True)


def test_the_metadata_names_the_logical_dataset_not_the_path(normals):
    """Publishing the locator would hand back what the profile keeps out of the input."""
    metadata = _call(normals)["metadata"]
    assert metadata["dataset"] == "science:normals:montpellier"
    assert not any("parquet" in str(value) for value in metadata.values())
    assert metadata["wet_day_threshold_mm"] == WET_DAY_THRESHOLD_MM
    assert metadata["day_grain"] == "utc_day"
    assert metadata["normal_grain"] == "daily_values_by_calendar_month"
    assert metadata["normals_schema_version"] == [2]


def test_no_values_means_no_series(normals):
    result = _call(normals)
    assert result["series"] == [] and result["unclassified"] == []
    assert len(result["normals"]) == 4


def test_build_watch_series_is_pure_and_reusable():
    """The offline replay (§G3) has to reach the same evaluator without the tool around it."""
    series, unclassified = build_watch_series(
        [
            {"variable": "wind_speed", "date": "2026-01-15", "value": 13.0},
            {"variable": "wind_speed", "date": "2026-03-15", "value": 13.0},
        ],
        normals=[_row(1, "wind_speed", p10=2.0, p50=5.0, p90=12.0)],
    )
    assert [entry["percentile_band"] for entry in series] == [">p90"]
    assert [item["reason"] for item in unclassified] == ["reference_unavailable"]


@pytest.mark.parametrize(
    ("start", "end"),
    [("not-a-date", "2020-12-31"), ("2020-12-31", "1991-01-01")],
)
def test_stored_normals_invalid_window_is_refused(tmp_path, allow_local_parquet_paths, start, end):
    path = Path(allow_local_parquet_paths) / f"{tmp_path.name}-invalid-window.parquet"
    source = _write_normals(
        path, [_row(1, "precipitation", window_start=start, window_end=end, window_label="custom")]
    )
    result = _call(source)
    assert "error" in result
    assert "Stored normals" in result["error"]


@pytest.mark.parametrize(
    ("start", "end", "label", "expected"),
    [
        ("1991-01-01", "2020-12-31", "1991-2020", "1991-2020"),
        (*climate._resolve_normals_window(None, None), "1991-2020"),
        ("2001-03-01", "2010-09-30", "custom", "2001-2010"),
        ("2021-03-01", "2021-09-30", "custom", "2021-2021"),
    ],
)
def test_stored_window_dates_produce_a_valid_watch_input_document(
    tmp_path, allow_local_parquet_paths, start, end, label, expected
):
    """The adapter's fields are stubbed; every field GeoAI owns comes from the tool.

    Proving the entries validate one by one is not the same as proving the document does —
    this assembles one so the two halves are shown to fit.
    """
    path = Path(allow_local_parquet_paths) / f"{tmp_path.name}-window.parquet"
    normals = _write_normals(
        path, [_row(1, "precipitation", window_start=start, window_end=end, window_label=label)]
    )
    result = _call(
        normals, values=[{"variable": "precipitation", "date": "2026-01-15", "value": 9.0}]
    )
    assert result["metadata"]["normals_window"] == {"start": start, "end": end}
    with duckdb.connect() as conn:
        stored = conn.execute(
            "SELECT window_start, window_end, window_label FROM read_parquet(?)", [normals]
        ).fetchone()
    assert stored == (start, end, label)
    document = {
        "schema_version": "watch-input/v1",
        # Adapter-owned (Deposium): identity, retrieval time, horizon accounting.
        "provider": {
            "id": "deposium_geoai",
            "tools": ["geo_forecast_query", "climate_normals"],
            "model": "ecmwf_aifs025_single",
            "provider_chain": ["openmeteo"],
            "licence": "CC-BY-4.0",
            "attribution": "Open-Meteo.com",
        },
        "place": {"ref": "zone:montpellier", "label": "Montpellier", "sampling": "bbox_midpoint"},
        "retrieved_at": "2026-01-14T06:00:00Z",
        "model_run_at": None,
        "horizon_days_requested": 7,
        "horizon_days_served": 7,
        "day_grain": "utc_day",
        "evidence_ids": [],
        # GeoAI-owned, straight from the tool.
        "confidence": result["confidence"],
        "series": result["series"],
        "caveats": result["caveats"],
        "reference": {
            **{
                k: v
                for k, v in result["reference"].items()
                if k in ("kind", "window", "grain", "source")
            },
            # The one field the tool deliberately does not mint — see `_normals_reference`.
            "evidence_id": "geoai_ev_" + "0" * 64,
        },
    }
    _DOCUMENT_VALIDATOR.validate(document)
    assert document["reference"]["window"] == expected


@pytest.mark.parametrize(
    ("start", "end"),
    [(None, "2020-12-31"), ("1991-01-01", None), (None, None), ("", "2020-12-31")],
)
def test_stored_normals_incomplete_window_is_refused(
    tmp_path, allow_local_parquet_paths, start, end
):
    path = Path(allow_local_parquet_paths) / f"{tmp_path.name}-incomplete-window.parquet"
    source = _write_normals(path, [_row(1, "precipitation", window_start=start, window_end=end)])
    result = _call(source)
    assert "error" in result
    assert "incomplete stored normals window" in result["error"]
