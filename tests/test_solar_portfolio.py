from __future__ import annotations

import csv
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from src import solar_portfolio as portfolio
from src.cli.solar_portfolio import main as cli_main
from src.solar import SolarFetchError
from src.solar_contracts import SolarQuery
from src.solar_portfolio import (
    SolarPortfolioError,
    export_solar_portfolio,
    load_solar_sites,
    run_solar_portfolio,
)

NOW = datetime(2026, 10, 11, 12, tzinfo=UTC)
VARS = ("shortwave_radiation", "diffuse_radiation", "direct_normal_irradiance")


def _csv(path: Path, rows: list[dict[str, object]], headers: list[str] | None = None) -> Path:
    columns = headers or ["site_id", "latitude", "longitude", "scenario_id", "tilt", "azimuth"]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return path


def _site(
    site_id: str,
    *,
    scenario: str = "default",
    lat: float = 43.6,
    lon: float = 3.8,
    tilt: float | None = None,
    azimuth: float | None = None,
) -> dict[str, object]:
    return {
        "site_id": site_id,
        "latitude": lat,
        "longitude": lon,
        "scenario_id": scenario,
        "tilt": "" if tilt is None else tilt,
        "azimuth": "" if azimuth is None else azimuth,
    }


def _payload(
    query: SolarQuery, *, retrieved_at: datetime = NOW, first_value: float | None = 0.0
) -> dict:
    interval_seconds = (
        1800 if query.source == "satellite" and query.temporal_resolution == "native" else 3600
    )
    count = 48 if interval_seconds == 1800 else 24
    irradiance = {
        "shortwave_radiation",
        "diffuse_radiation",
        "direct_normal_irradiance",
        "global_tilted_irradiance",
    }
    return {
        "schema_version": "geoai-solar/v1",
        "query": query.model_dump(mode="json"),
        "source": query.source,
        "product_type": {
            "forecast": "forecast",
            "era5": "reanalysis",
            "satellite": "satellite_estimate",
        }[query.source],
        "requested_coordinates": {"latitude": query.latitude, "longitude": query.longitude},
        "returned_coordinates": {
            "latitude": query.latitude,
            "longitude": query.longitude,
            "elevation": None,
        },
        "timestamps": [
            (datetime(2026, 10, 11, tzinfo=UTC) + timedelta(seconds=interval_seconds * step))
            .isoformat()
            .replace("+00:00", "Z")
            for step in range(count)
        ],
        "values": {
            variable: [first_value, *([None] * (count - 1))] for variable in query.variables
        },
        "units": dict.fromkeys(query.variables, "W/m²"),
        "interval_seconds": interval_seconds,
        "native_model_interval_seconds": interval_seconds,
        "interval_semantics": "provider-served temporal cadence; no additional aggregation or interpolation",
        "variable_semantics": {
            variable: "mean_over_preceding_interval"
            if variable in irradiance
            else "value_at_timestamp"
            for variable in query.variables
        },
        "model_issued_at": None,
        "requested_model": query.model
        or {"forecast": "ecmwf_aifs025_single", "era5": "era5", "satellite": "eumetsat_sarah3"}[
            query.source
        ],
        "actual_model": None,
        "licence": "CC BY 4.0",
        "attribution": {
            "forecast": "Open-Meteo",
            "era5": "Open-Meteo; ERA5",
            "satellite": "Open-Meteo; EUMETSAT SARAH-3",
        }[query.source],
        "missing_counts": dict.fromkeys(query.variables, count - 1),
        "retrieved_at": retrieved_at.isoformat().replace("+00:00", "Z"),
    }


def _run(path: Path, out: Path, fetcher, **kwargs):
    return run_solar_portfolio(
        sites_path=path,
        source="era5",
        start_date="2026-10-11",
        end_date="2026-10-11",
        variables=VARS,
        output_dir=out,
        fetcher=fetcher,
        clock=lambda: NOW,
        **kwargs,
    )


@pytest.mark.parametrize("source", ["era5", "satellite"])
@pytest.mark.parametrize("days", [2, 3])
def test_complete_multiday_series_succeeds_resumes_and_exports(tmp_path, source, days):
    sites = _csv(tmp_path / "sites.csv", [_site("demo")])
    calls = []
    start = datetime(2026, 10, 9, tzinfo=UTC)

    def fetch(query):
        calls.append(query)
        payload = _payload(query)
        interval = payload["interval_seconds"]
        count = days * 86400 // interval
        payload["timestamps"] = [
            (start + timedelta(seconds=interval * step)).isoformat().replace("+00:00", "Z")
            for step in range(count)
        ]
        payload["values"] = {variable: [0.0] * count for variable in query.variables}
        payload["missing_counts"] = dict.fromkeys(query.variables, 0)
        return payload

    arguments = {
        "sites_path": sites,
        "source": source,
        "start_date": start.date().isoformat(),
        "end_date": (start + timedelta(days=days - 1)).date().isoformat(),
        "variables": VARS,
        "output_dir": tmp_path / "batch",
        "fetcher": fetch,
        "clock": lambda: NOW,
    }
    first = run_solar_portfolio(**arguments)
    assert first["succeeded"] == 1 and first["failed"] == 0
    resumed = run_solar_portfolio(**arguments)
    assert resumed["succeeded"] == 1 and len(calls) == 1
    with export_solar_portfolio(arguments["output_dir"]).open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == days * (48 if source == "satellite" else 24) * len(VARS)
    assert rows[-1]["time"] == (
        start + timedelta(days=days) - timedelta(seconds=1800 if source == "satellite" else 3600)
    ).isoformat().replace("+00:00", "Z")


def test_missing_elevation_key_is_rejected_before_export(tmp_path):
    sites = _csv(tmp_path / "sites.csv", [_site("demo")])

    def fetch(query):
        payload = _payload(query)
        del payload["returned_coordinates"]["elevation"]
        return payload

    result = _run(sites, tmp_path / "batch", fetch)
    assert result["succeeded"] == 0 and result["failed"] == 1


def test_validates_csv_and_blocks_network_before_any_fetch(tmp_path):
    bad = _csv(
        tmp_path / "bad.csv",
        [_site("alpha")],
        headers=["site_id", "latitude", "longitude", "token"],
    )
    calls: list[object] = []
    with pytest.raises(SolarPortfolioError, match="unknown columns"):
        _run(bad, tmp_path / "out", lambda query: calls.append(query))
    assert calls == []
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize(
    "rows, message",
    [
        ([_site("a", scenario="one"), _site("a", scenario="two", lat=44)], "changes coordinates"),
        ([_site("a"), _site("a")], "duplicate site_id/scenario_id"),
        ([{**_site("a"), "latitude": "nan"}], "must be finite"),
        ([{**_site("a"), "tilt": 30}], "must be supplied together"),
    ],
)
def test_rejects_invalid_rows_before_fetch(tmp_path, rows, message):
    sites = _csv(tmp_path / "bad.csv", rows)
    calls: list[object] = []
    with pytest.raises(SolarPortfolioError, match=message):
        _run(sites, tmp_path / "out", lambda query: calls.append(query))
    assert calls == []


def test_gti_requires_orientation_and_same_site_keeps_distinct_scenarios(tmp_path):
    missing = _csv(tmp_path / "missing.csv", [_site("roof")])
    with pytest.raises(SolarPortfolioError, match="required for global_tilted"):
        load_solar_sites(missing, variables=(*VARS, "global_tilted_irradiance"))

    sites = _csv(
        tmp_path / "scenarios.csv",
        [
            _site("roof", scenario="south", tilt=30, azimuth=0),
            _site("roof", scenario="west", tilt=30, azimuth=90),
        ],
    )
    calls = []
    result = run_solar_portfolio(
        sites_path=sites,
        source="era5",
        start_date="2026-10-11",
        end_date="2026-10-11",
        variables=(*VARS, "global_tilted_irradiance"),
        output_dir=tmp_path / "out",
        fetcher=lambda query: calls.append(query) or _payload(query),
        clock=lambda: NOW,
    )
    assert result["succeeded"] == 2
    assert len(calls) == 2
    manifest = json.loads((tmp_path / "out" / "manifest.json").read_text())
    assert {entry["scenario_id"] for entry in manifest["items"].values()} == {"south", "west"}
    assert len(list((tmp_path / "out").glob("[0-9a-f]*.json"))) == 2
    target = export_solar_portfolio(tmp_path / "out")
    with target.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    orientation = {
        row["scenario_id"]: (row["tilt"], row["azimuth"])
        for row in rows
        if row["variable"] == "global_tilted_irradiance"
    }
    assert orientation == {"south": ("30.0", "0.0"), "west": ("30.0", "90.0")}


def test_identical_scenario_queries_share_one_acquisition_but_export_both_rows(tmp_path):
    sites = _csv(
        tmp_path / "scenarios.csv",
        [_site("roof", scenario="base"), _site("roof", scenario="lease")],
    )
    calls = []
    result = _run(sites, tmp_path / "out", lambda query: calls.append(query) or _payload(query))
    assert result["succeeded"] == 2
    assert result["http_acquisitions"] == 1
    assert len(calls) == 1
    target = export_solar_portfolio(tmp_path / "out")
    with target.open(encoding="utf-8", newline="") as stream:
        exported = list(csv.DictReader(stream))
    assert {row["scenario_id"] for row in exported} == {"base", "lease"}


def test_adapter_declared_permanent_error_is_not_retried(tmp_path):
    sites = _csv(tmp_path / "sites.csv", [_site("one")])
    calls = 0

    def fetch(_query):
        nonlocal calls
        calls += 1
        raise SolarFetchError("provider rejected request", retryable=False)

    result = _run(sites, tmp_path / "out", fetch, max_attempts=3, sleep=lambda _: None)
    assert result["failed"] == 1
    assert calls == 1
    manifest = json.loads((tmp_path / "out" / "manifest.json").read_text())
    item = next(iter(manifest["items"].values()))
    assert item["error"] == {"type": "SolarFetchError", "retryable": False}


def test_fifty_mock_sites_are_collected_with_bounded_workers(tmp_path):
    sites = _csv(tmp_path / "many.csv", [_site(f"site-{n}", lat=40 + n / 100) for n in range(50)])
    calls: list[str] = []
    result = _run(
        sites,
        tmp_path / "out",
        lambda query: calls.append(str(query.latitude)) or _payload(query),
        concurrency=8,
    )
    assert result["total"] == result["succeeded"] == 50
    assert result["http_acquisitions"] == 50
    assert len(set(calls)) == 50


def test_partial_failure_resumes_without_refetching_successes(tmp_path):
    sites = _csv(
        tmp_path / "sites.csv",
        [_site(name, lat=43.0 + index / 10) for index, name in enumerate("abcd")],
    )
    seen: list[float] = []

    def first_fetch(query):
        seen.append(query.latitude)
        if query.latitude == 43.1:
            raise ValueError("private provider URL with secret")
        return _payload(query)

    first = _run(sites, tmp_path / "out", first_fetch)
    assert first["failed"] == 1
    manifest = json.loads((tmp_path / "out" / "manifest.json").read_text())
    assert "secret" not in json.dumps(manifest)
    assert len(seen) == 4

    second_calls: list[float] = []
    second = _run(
        sites,
        tmp_path / "out",
        lambda query: second_calls.append(query.latitude) or _payload(query),
    )
    assert second["succeeded"] == 4
    assert second_calls == [43.1]
    assert len(seen) == 4


def test_retryable_errors_are_bounded_and_count_every_acquisition(tmp_path):
    sites = _csv(tmp_path / "sites.csv", [_site("one")])
    attempts = 0

    class RetryableError(RuntimeError):
        retryable = True

    def fetch(query):
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise RetryableError("hidden endpoint")
        return _payload(query)

    result = _run(sites, tmp_path / "out", fetch, max_attempts=3, sleep=lambda _: None)
    assert result["http_acquisitions"] == 3
    item = next(
        iter(json.loads((tmp_path / "out" / "manifest.json").read_text())["items"].values())
    )
    assert item["attempts"] == 3


def test_interrupted_running_item_is_reset_and_retried(tmp_path):
    sites = _csv(tmp_path / "sites.csv", [_site("one")])

    class SimulatedCrash(BaseException):
        pass

    with pytest.raises(SimulatedCrash):
        _run(sites, tmp_path / "out", lambda _query: (_ for _ in ()).throw(SimulatedCrash()))
    manifest_path = tmp_path / "out" / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    assert next(iter(manifest["items"].values()))["status"] == "running"
    result = _run(sites, tmp_path / "out", lambda query: _payload(query))
    assert result["succeeded"] == 1


def test_artifact_written_before_manifest_checkpoint_is_salvaged(tmp_path, monkeypatch):
    sites = _csv(tmp_path / "sites.csv", [_site("one")])
    output = tmp_path / "out"
    original_atomic_json = portfolio._atomic_json
    calls = 0

    class SimulatedCrash(BaseException):
        pass

    class RetryableError(RuntimeError):
        retryable = True

    def fetch(query):
        nonlocal calls
        calls += 1
        if calls < 3:
            raise RetryableError("temporary provider failure")
        return _payload(query)

    def crash_after_artifact(path, value):
        digest = original_atomic_json(path, value)
        if path.name != "manifest.json" and path.suffix == ".json":
            raise SimulatedCrash
        return digest

    monkeypatch.setattr(portfolio, "_atomic_json", crash_after_artifact)
    with pytest.raises(SimulatedCrash):
        _run(sites, output, fetch)

    manifest = json.loads((output / "manifest.json").read_text())
    item = next(iter(manifest["items"].values()))
    assert item["status"] == "running"
    assert item["acquisition_counted"] is True
    assert item["attempts"] == 3
    assert manifest["counts"]["http_acquisitions"] == 3

    monkeypatch.setattr(portfolio, "_atomic_json", original_atomic_json)
    resumed = _run(sites, output, fetch)
    assert resumed["succeeded"] == 1
    assert resumed["http_acquisitions"] == 3
    assert resumed["cache_reuse"] == 1
    assert calls == 3


@pytest.mark.parametrize("damage", ["corrupt", "mismatched_query", "numerical_tamper"])
def test_invalid_orphan_artifact_is_not_salvaged(tmp_path, monkeypatch, damage):
    sites = _csv(tmp_path / "sites.csv", [_site("one")])
    output = tmp_path / "out"
    original_atomic_json = portfolio._atomic_json
    calls = 0

    class SimulatedCrash(BaseException):
        pass

    def fetch(query):
        nonlocal calls
        calls += 1
        return _payload(query)

    def crash_after_artifact(path, value):
        digest = original_atomic_json(path, value)
        if path.name != "manifest.json" and path.suffix == ".json":
            if damage == "corrupt":
                path.write_text("{corrupt", encoding="utf-8")
            else:
                payload = json.loads(path.read_text(encoding="utf-8"))
                if damage == "mismatched_query":
                    payload["query"]["latitude"] += 1
                else:
                    payload["values"]["shortwave_radiation"][0] = 999.0
                path.write_bytes(portfolio._canonical_bytes(payload) + b"\n")
            raise SimulatedCrash
        return digest

    monkeypatch.setattr(portfolio, "_atomic_json", crash_after_artifact)
    with pytest.raises(SimulatedCrash):
        _run(sites, output, fetch)
    monkeypatch.setattr(portfolio, "_atomic_json", original_atomic_json)

    resumed = _run(sites, output, fetch)
    assert resumed["succeeded"] == 1
    assert calls == 2


def test_stale_or_corrupt_forecast_artifact_is_fetched_again(tmp_path):
    sites = _csv(tmp_path / "sites.csv", [_site("one")])
    current = [NOW]
    calls = 0

    def fetch(query):
        nonlocal calls
        calls += 1
        return _payload(query, retrieved_at=current[0])

    kwargs = {
        "sites_path": sites,
        "source": "forecast",
        "start_date": "2026-10-11",
        "end_date": "2026-10-11",
        "variables": VARS,
        "output_dir": tmp_path / "out",
        "fetcher": fetch,
        "clock": lambda: current[0],
    }
    run_solar_portfolio(**kwargs)
    current[0] = NOW + timedelta(hours=7)
    run_solar_portfolio(**kwargs)
    assert calls == 2
    artifact = next((tmp_path / "out").glob("[0-9a-f]*.json"))
    artifact.write_text("{corrupt", encoding="utf-8")
    run_solar_portfolio(**kwargs)
    assert calls == 3


@pytest.mark.parametrize("malformation", ["wrong_interval", "truncated"])
def test_malformed_artifacts_never_mark_succeeded(tmp_path, malformation):
    sites = _csv(tmp_path / "sites.csv", [_site("one")])

    def fetch(query):
        payload = _payload(query)
        if malformation == "wrong_interval":
            payload["timestamps"] = payload["timestamps"][:1]
            payload["values"] = {key: values[:1] for key, values in payload["values"].items()}
            payload["missing_counts"] = dict.fromkeys(query.variables, 0)
            payload["interval_seconds"] = 17
        else:
            payload["timestamps"] = payload["timestamps"][:1]
            payload["values"] = {key: values[:1] for key, values in payload["values"].items()}
            payload["missing_counts"] = dict.fromkeys(query.variables, 0)
        return payload

    result = _run(sites, tmp_path / "out", fetch)
    manifest = json.loads((tmp_path / "out" / "manifest.json").read_text())
    item = next(iter(manifest["items"].values()))
    assert result["failed"] == 1
    assert item["status"] == "failed"
    assert not item.get("artifact_sha256")
    assert len(list((tmp_path / "out").glob("[0-9a-f]*.json"))) == 0


def test_satellite_defaults_native_and_accepts_explicit_hourly(tmp_path):
    sites = _csv(tmp_path / "sites.csv", [_site("one")])
    observed: list[str] = []

    def fetch(query):
        observed.append(query.temporal_resolution)
        return _payload(query)

    base = {
        "sites_path": sites,
        "source": "satellite",
        "start_date": "2026-10-11",
        "end_date": "2026-10-11",
        "variables": VARS,
        "fetcher": fetch,
        "clock": lambda: NOW,
    }
    native = run_solar_portfolio(output_dir=tmp_path / "native", **base)
    hourly = run_solar_portfolio(
        output_dir=tmp_path / "hourly", temporal_resolution="hourly", **base
    )
    native_manifest = json.loads((tmp_path / "native" / "manifest.json").read_text())
    hourly_manifest = json.loads((tmp_path / "hourly" / "manifest.json").read_text())
    native_payload = json.loads(next((tmp_path / "native").glob("[0-9a-f]*.json")).read_text())
    hourly_payload = json.loads(next((tmp_path / "hourly").glob("[0-9a-f]*.json")).read_text())
    assert native["succeeded"] == hourly["succeeded"] == 1
    assert observed == ["native", "hourly"]
    assert native_manifest["batch_spec"]["temporal_resolution"] == "native"
    assert hourly_manifest["batch_spec"]["temporal_resolution"] == "hourly"
    assert native_payload["interval_seconds"] == 1800
    assert hourly_payload["interval_seconds"] == 3600


def test_native_resolution_is_rejected_for_non_satellite_source(tmp_path):
    sites = _csv(tmp_path / "sites.csv", [_site("one")])
    with pytest.raises(SolarPortfolioError, match="only available for satellite"):
        _run(
            sites,
            tmp_path / "out",
            lambda query: pytest.fail("invalid resolution called fetcher"),
            temporal_resolution="native",
        )


def test_output_dir_rejects_a_different_batch_spec(tmp_path):
    sites = _csv(tmp_path / "sites.csv", [_site("one")])
    _run(sites, tmp_path / "out", lambda query: _payload(query))
    changed = _csv(tmp_path / "changed.csv", [_site("one", lat=44)])
    with pytest.raises(SolarPortfolioError, match="different batch specification"):
        _run(
            changed, tmp_path / "out", lambda query: pytest.fail("mismatched batch called provider")
        )


def test_concurrent_run_of_same_directory_is_rejected(tmp_path):
    sites = _csv(tmp_path / "sites.csv", [_site("one")])
    entered = threading.Event()
    release = threading.Event()

    def slow(query):
        entered.set()
        assert release.wait(3)
        return _payload(query)

    with ThreadPoolExecutor(max_workers=1) as pool:
        running = pool.submit(_run, sites, tmp_path / "out", slow)
        assert entered.wait(3)
        with pytest.raises(SolarPortfolioError, match="another process"):
            _run(sites, tmp_path / "out", lambda query: pytest.fail("second fetch ran"))
        release.set()
        assert running.result(timeout=3)["succeeded"] == 1


def test_export_preserves_zero_and_null_as_blank(tmp_path):
    sites = _csv(tmp_path / "sites.csv", [_site("one")])
    _run(sites, tmp_path / "out", lambda query: _payload(query, first_value=0.0))
    target = export_solar_portfolio(tmp_path / "out")
    with target.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == len(VARS) * 24
    assert rows[0]["value"] == "0.0"
    assert rows[1]["value"] == ""
    assert rows[0]["scenario_id"] == "default"
    assert rows[0]["tilt"] == rows[0]["azimuth"] == ""
    assert rows[0]["requested_model"] == "era5"
    assert rows[0]["actual_model"] == ""
    assert rows[0]["requested_latitude"] == "43.6"
    assert rows[0]["returned_longitude"] == "3.8"
    assert rows[0]["returned_elevation"] == ""
    assert rows[0]["temporal_resolution"] == "hourly"
    assert rows[0]["interval_seconds"] == "3600"
    assert rows[0]["licence"] == "CC BY 4.0"
    assert rows[0]["attribution"] == "Open-Meteo; ERA5"


def test_cli_validate_and_run_summary(tmp_path, capsys):
    sites = _csv(tmp_path / "sites.csv", [_site("one")])
    assert cli_main(["validate", "--sites", str(sites)]) == 0
    assert json.loads(capsys.readouterr().out)["valid"] is True


def test_structured_sites_match_csv_and_preserve_text_and_zero(tmp_path):
    rows = [
        {"site_id": "00123", "latitude": 0, "longitude": 0, "tilt": 0, "azimuth": 0},
        {"site_id": "other", "latitude": 43.6, "longitude": 3.8},
    ]
    csv_sites = load_solar_sites(_csv(tmp_path / "sites.csv", rows), variables=VARS)
    structured = portfolio.validate_solar_sites(rows, variables=VARS)
    assert [site.canonical() for site in structured] == [site.canonical() for site in csv_sites]
    assert structured[0].site_id == "00123"
    assert structured[0].tilt == 0
    assert structured[1].tilt is None
    assert portfolio.validate_solar_sites(structured) == structured


@pytest.mark.parametrize(
    "updates,message",
    [
        ({"site_id": 123}, "site_id must be text"),
        ({"scenario_id": False}, "scenario_id must be text"),
        ({"latitude": "0"}, "latitude must be numeric"),
        ({"longitude": True}, "longitude must be numeric"),
        ({"latitude": None}, "latitude is required"),
        ({"latitude": float("nan")}, "latitude must be finite"),
        ({"longitude": float("inf")}, "longitude must be finite"),
        ({"tilt": ""}, "tilt must be numeric"),
        ({"tilt": 0}, "supplied together"),
        ({"source": "era5"}, "unknown fields"),
    ],
)
def test_structured_sites_reject_malformed_input(updates, message):
    row = {"site_id": "00123", "latitude": 0, "longitude": 0, **updates}
    with pytest.raises(SolarPortfolioError, match=message):
        portfolio.validate_solar_sites([row])


@pytest.mark.parametrize(
    "second,message",
    [
        ({"site_id": "a", "latitude": 0, "longitude": 0}, "duplicate"),
        (
            {"site_id": "a", "latitude": 1, "longitude": 0, "scenario_id": "other"},
            "changes coordinates",
        ),
    ],
)
def test_structured_sites_enforce_cross_row_rules(second, message):
    with pytest.raises(SolarPortfolioError, match=message):
        portfolio.validate_solar_sites([{"site_id": "a", "latitude": 0, "longitude": 0}, second])


def test_structured_sites_require_orientation_for_gti():
    with pytest.raises(SolarPortfolioError, match="required for global_tilted"):
        portfolio.validate_solar_sites(
            [{"site_id": "a", "latitude": 0, "longitude": 0}],
            variables=["global_tilted_irradiance"],
        )


def test_constructed_site_is_revalidated_before_side_effects(tmp_path):
    out = tmp_path / "batch"
    calls = []
    with pytest.raises(SolarPortfolioError, match="latitude must be finite"):
        run_solar_portfolio(
            sites=[portfolio.SolarSite("a", float("nan"), 0)],
            source="era5",
            start_date="2026-10-11",
            end_date="2026-10-11",
            variables=VARS,
            output_dir=out,
            fetcher=lambda query: calls.append(query),
        )
    assert not out.exists()
    assert not calls


@pytest.mark.parametrize("both", [False, True])
def test_portfolio_requires_exactly_one_site_input(tmp_path, both):
    inputs = {"sites_path": tmp_path / "missing.csv", "sites": []} if both else {}
    with pytest.raises(SolarPortfolioError, match="exactly one"):
        run_solar_portfolio(
            **inputs,
            source="era5",
            start_date="2026-10-11",
            end_date="2026-10-11",
            variables=VARS,
            output_dir=tmp_path / "batch",
        )
    assert not (tmp_path / "batch").exists()


def test_structured_portfolio_reuses_csv_acquisition_identity(tmp_path):
    rows = [{"site_id": "00123", "latitude": 0, "longitude": 0}]
    csv_path = _csv(tmp_path / "sites.csv", rows)
    out = tmp_path / "batch"
    calls = []

    def fetch(query):
        calls.append(query)
        return _payload(query)

    first = _run(csv_path, out, fetch)
    resumed = run_solar_portfolio(
        sites=rows,
        source="era5",
        start_date="2026-10-11",
        end_date="2026-10-11",
        variables=VARS,
        output_dir=out,
        fetcher=fetch,
        clock=lambda: NOW,
    )
    assert first["batch_spec_sha256"] == resumed["batch_spec_sha256"]
    assert resumed["succeeded"] == 1
    assert len(calls) == 1
