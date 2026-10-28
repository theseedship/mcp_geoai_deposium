"""Contract tests for the MCP tool I/O schemas (FIERCE DoD lot B7).

Three layers:

1. **Input models** accept known-good arguments and reject known-bad ones.
2. **Handler wiring** — every registered tool actually enforces its input
   contract at the boundary (a contract-violating call raises *before* any
   network/DuckDB I/O), and the registered tool set matches the contract
   registry exactly (a new tool without a contract fails this test).
3. **Output models** validate representative success payloads + the error
   variant, and the full I/O JSON Schema exports.
"""

from __future__ import annotations

import asyncio
import inspect

import pytest
from jsonschema import Draft202012Validator
from pydantic import ValidationError

from src import contracts as C
from src.config import settings
from src.tools import (
    _ndvi,
    register_change_tools,
    register_climate_tools,
    register_forecast_tools,
    register_pattern_alert_tools,
    register_pattern_tools,
    register_solar_tools,
    register_spatial_tools,
    register_stac_tools,
)


class _CaptureMCP:
    """A stand-in MCP whose ``@tool()`` records the handler and returns it
    unchanged — lets us drive the real tool closures without a live FastMCP
    server or any network."""

    def __init__(self) -> None:
        self.tools: dict[str, object] = {}
        self.output_schemas: dict[str, object] = {}

    def tool(self, *_args, **kwargs):
        def _decorator(fn):
            self.tools[fn.__name__] = fn
            self.output_schemas[fn.__name__] = kwargs.get("output_schema")
            return fn

        return _decorator


def _load_capture() -> _CaptureMCP:
    cap = _CaptureMCP()
    register_stac_tools(cap)
    register_spatial_tools(cap)
    register_change_tools(cap)
    register_pattern_tools(cap)
    register_climate_tools(cap)
    register_forecast_tools(cap)
    register_pattern_alert_tools(cap)
    register_solar_tools(cap)
    return cap


def _load_tools() -> dict:
    return _load_capture().tools


# Minimal valid argument sets per tool (good enough to PASS check_input;
# they would hit the network if fully run, so we only use them to assert the
# contract does NOT reject a healthy call).
GOOD_ARGS = {
    "stac_search": {
        "bbox": [0.0, 0.0, 1.0, 1.0],
        "datetime": "2024-01-01/2024-02-01",
        "collections": ["s2"],
    },
    "stac_download": {"item_id": "x", "collection": "c", "bands": ["B04"]},
    "spatial_query": {
        "source": "s3://b/k.parquet",
        "geometry": {"type": "Point", "coordinates": [0, 0]},
    },
    "ndvi_profile": {
        "geometry": {"type": "Point", "coordinates": [0, 0]},
        "start_date": "2024-01-01",
        "end_date": "2024-02-01",
    },
    "vegetation_stress": {"geometry": {"type": "Point", "coordinates": [0, 0]}},
    "change_detect": {
        "bbox": [0.0, 0.0, 1.0, 1.0],
        "date_before": "2024-01-01",
        "date_after": "2024-06-01",
    },
    "discover_patterns": {"source": "auto", "tenant_id": "t", "space_id": "s"},
    "discover_patterns_cv": {
        "zone_ids": ["z1", "z2", "z3", "z4", "z5"],
        "source_template": "climate/holdout/unified/{window_label}/{zone_id}.parquet",
        "window_labels": ["2023_07"],
    },
    "era5_ingest": {"zone_id": "montpellier", "start_date": "2024-01-01", "end_date": "2024-01-10"},
    "climate_enrich_ndvi": {
        "zone_id": "montpellier",
        "start_date": "2024-01-01",
        "end_date": "2024-01-10",
    },
    "climate_percentiles": {"zone_id": "montpellier"},
    "forecast_query": {"zone_id": "montpellier"},  # zone-mode (XOR point) — defaults otherwise
    "pattern_alert": {"zone_id": "montpellier"},
}

# One contract-violating argument set per tool (caught by check_input, no I/O).
BAD_ARGS = {
    "stac_search": {"bbox": [0.0, 0.0], "datetime": "2024", "collections": ["s2"]},  # bbox arity
    "stac_download": {"item_id": "x", "collection": "c", "bands": []},  # empty bands
    "spatial_query": {
        "source": "",
        "geometry": {"type": "Point", "coordinates": [0, 0]},
    },  # empty source
    "ndvi_profile": {
        "geometry": {"type": "Point", "coordinates": [0, 0]},
        "start_date": "a",
        "end_date": "b",
        "cloud_cover_max": 200,  # out of [0, 100]
    },
    "vegetation_stress": {
        "geometry": {"type": "Point", "coordinates": [0, 0]},
        "ndvi_threshold": 5.0,
    },  # out of [-1,1]
    "change_detect": {"bbox": [0.0, 0.0, 1.0], "date_before": "a", "date_after": "b"},  # bbox arity
    # min_support range is validated gracefully by the handler, so the contract's
    # structural catch here is a type violation (non-numeric where float expected).
    "discover_patterns": {"source": "auto", "min_support": "high"},
    # O3 requires at least 5 unique zones for 5-fold cross-zone validation.
    "discover_patterns_cv": {"zone_ids": ["z1", "z1", "z2", "z3", "z4"]},
    # empty zone_id violates min_length=1 (caught by check_input before any fetch/DuckDB I/O)
    "era5_ingest": {"zone_id": "", "start_date": "2024-01-01", "end_date": "2024-01-10"},
    # tolerance_days out of [0, 31]
    "climate_enrich_ndvi": {
        "zone_id": "x",
        "start_date": "2024-01-01",
        "end_date": "2024-01-10",
        "tolerance_days": 99,
    },
    # classify_month is bounded to real calendar months.
    "climate_percentiles": {"zone_id": "x", "classify_month": 13},
    # forecast_days is bounded ge=1 (caught by check_input before any router/network I/O).
    "forecast_query": {"zone_id": "x", "forecast_days": 0},
    "pattern_alert": {"zone_id": "x", "forecast_horizon_days": 0},
}


class TestInputModels:
    @pytest.mark.parametrize("tool", sorted(GOOD_ARGS))
    def test_good_args_accepted(self, tool):
        model = C.TOOL_CONTRACTS[tool]["input"]
        model(**GOOD_ARGS[tool])  # must not raise

    @pytest.mark.parametrize("tool", sorted(BAD_ARGS))
    def test_bad_args_rejected(self, tool):
        model = C.TOOL_CONTRACTS[tool]["input"]
        with pytest.raises(ValidationError):
            model(**BAD_ARGS[tool])

    def test_unknown_field_forbidden(self):
        with pytest.raises(ValidationError):
            C.StacSearchInput(
                bbox=[0.0, 0.0, 1.0, 1.0], datetime="2024", collections=["s2"], typo_field=1
            )

    def test_interval_days_zero_allowed(self):
        # documented: 0 == "keep every observation"; must not be rejected
        C.NdviProfileInput(
            geometry={"type": "Point", "coordinates": [0, 0]},
            start_date="a",
            end_date="b",
            interval_days=0,
        )

    def test_vegetation_stress_lookback_is_bounded_far_past_the_archive(self):
        # #221: past ~739,884 days the handler's `now - timedelta(days=...)` leaves the calendar
        # (OverflowError "date value out of range"), so the contract caps the window at a hundred
        # years. Ten years would already refuse the 2015-2016 start of the Sentinel-2 archive.
        point = {"type": "Point", "coordinates": [0, 0]}
        for lookback_days in (3651, 36500):
            C.VegetationStressInput(geometry=point, lookback_days=lookback_days)
        for lookback_days in (36501, 1_000_000):
            with pytest.raises(ValidationError):
                C.VegetationStressInput(geometry=point, lookback_days=lookback_days)

    def test_bbox_domain_rejected(self):
        with pytest.raises(ValidationError):
            C.StacSearchInput(bbox=[10.0, 0.0, 5.0, 1.0], datetime="2024", collections=["s2"])
        with pytest.raises(ValidationError):
            C.ChangeDetectInput(
                bbox=[-181.0, 0.0, 1.0, 1.0],
                date_before="2024-01-01",
                date_after="2024-01-02",
            )

    def test_change_detect_dates_are_iso_and_ordered(self):
        with pytest.raises(ValidationError):
            C.ChangeDetectInput(
                bbox=[0.0, 0.0, 1.0, 1.0],
                date_before="2024/01/01",
                date_after="2024-01-02",
            )
        with pytest.raises(ValidationError):
            C.ChangeDetectInput(
                bbox=[0.0, 0.0, 1.0, 1.0],
                date_before="2024-01-02",
                date_after="2024-01-02",
            )

    def test_local_source_policy_is_published_in_input_schemas(self):
        spatial_source = C.SpatialQueryInput.model_json_schema()["properties"]["source"]
        alert_properties = C.PatternAlertInput.model_json_schema()["properties"]

        assert "ALLOW_LOCAL_PARQUET_SOURCE" in spatial_source["description"]
        assert "LOCAL_PARQUET_SOURCE_ROOT" in spatial_source["description"]
        assert "LOCAL_PARQUET_SOURCE_ROOT" in alert_properties["patterns_source"]["description"]
        assert "ALLOW_LOCAL_PARQUET_SOURCE" in alert_properties["allow_local_source"]["description"]


class TestGeometryValidation:
    """Shape-level GeoJSON geometry validation on the geometry-bearing inputs."""

    @pytest.mark.parametrize(
        "geom",
        [
            {},  # not a geometry
            {"type": "Banana", "coordinates": [0, 0]},  # unknown type
            {"type": "Polygon"},  # missing coordinates
            {"type": "Point", "coordinates": []},  # empty coordinates
            {"type": "Feature", "geometry": {"type": "Point", "coordinates": [0, 0]}},  # a Feature
            {"type": "GeometryCollection", "geometries": []},  # empty collection
            {
                "type": "GeometryCollection",
                "geometries": [{"coordinates": [0, 0]}],
            },  # child no type
            {
                "type": "GeometryCollection",
                "geometries": [{"type": "Point"}],
            },  # child has type but no coordinates
        ],
    )
    def test_rejects_malformed_geometry(self, geom):
        with pytest.raises(ValidationError):
            C.SpatialQueryInput(source="data.parquet", geometry=geom)

    @pytest.mark.parametrize(
        "geom",
        [
            {"type": "Point", "coordinates": [0, 0]},
            {"type": "Polygon", "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 0]]]},
            {
                "type": "GeometryCollection",
                "geometries": [{"type": "Point", "coordinates": [0, 0]}],
            },
        ],
    )
    def test_accepts_valid_geometry(self, geom):
        C.SpatialQueryInput(source="data.parquet", geometry=geom)
        C.NdviProfileInput(geometry=geom, start_date="2024-01-01", end_date="2024-02-01")
        C.VegetationStressInput(geometry=geom)

    @pytest.mark.parametrize(
        "geom",
        [
            {"type": "Point", "coordinates": [10**400, 0]},  # an int past the float range
            {
                "type": "Polygon",
                "coordinates": [[[0, 0], [1, 0], [1, -(10**400)], [0, 0]]],
            },  # deep in a ring
            {
                "type": "GeometryCollection",
                "geometries": [{"type": "Point", "coordinates": [0, 10**400]}],
            },  # in a collection member
            {"type": "Point", "coordinates": [float("inf"), 0]},
            {"type": "Point", "coordinates": [0, float("nan")]},
        ],
    )
    def test_rejects_a_coordinate_that_is_not_a_finite_float(self, geom):
        # #221: a JSON integer of 2**1024 or more reaches the handler as a Python int, and shapely
        # raises OverflowError on it, which Pydantic does not report as a validation error.
        with pytest.raises(ValidationError, match="finite numbers"):
            C.SpatialQueryInput(source="data.parquet", geometry=geom)
        with pytest.raises(ValidationError, match="finite numbers"):
            C.NdviProfileInput(geometry=geom, start_date="2024-01-01", end_date="2024-02-01")
        with pytest.raises(ValidationError, match="finite numbers"):
            C.VegetationStressInput(geometry=geom)

    def test_accepts_projected_coordinates_in_the_source_crs(self):
        # The finite-number rule is not a lon/lat range check, so Lambert-93 metres pass input
        # validation; spatial_query compares them with a WKB BLOB source in its own coordinates.
        C.SpatialQueryInput(
            source="data.parquet", geometry={"type": "Point", "coordinates": [700000, 6600000]}
        )

    @pytest.mark.parametrize(
        "geom",
        [
            {"type": "Point", "coordinates": [True, 0]},
            {"type": "Polygon", "coordinates": [[[0, 0], [1, "x"], [1, 1], [0, 0]]]},
        ],
    )
    def test_leaves_non_numeric_coordinate_members_to_shapely(self, geom):
        # The check stays structural for anything that is not a number, as before #221: a bool or
        # a string member passes the contract, as it did. (Shapely then reads a bool as 0 or 1
        # and fails on a string; neither is an overflow, so neither is changed here.)
        C.SpatialQueryInput(source="data.parquet", geometry=geom)
        C.NdviProfileInput(geometry=geom, start_date="2024-01-01", end_date="2024-02-01")
        C.VegetationStressInput(geometry=geom)

    def test_mesh_check_reports_an_int_past_the_float_range_as_a_value_error(self):
        # Pydantic turns only ValueError and AssertionError into a ValidationError; the
        # OverflowError that float() raised here escaped as a tool error instead (#221).
        with pytest.raises(ValueError, match="coordinate positions must contain finite numbers"):
            C._validate_mesh_geometry({"type": "Point", "coordinates": [10**400, 0]})


class TestHandlerWiring:
    def test_spatial_evidence_contract_is_published_and_wired(self):
        published = C.contracts_json_schema()["spatial_query"]
        assert "evidence_context" in published["input"]["properties"]
        assert "GeoAIEvidenceV1" in published["output"]["$defs"]
        assert "evidence" in published["output"]["anyOf"][0]["properties"]
        wired = C.output_schema_for("spatial_query")
        assert "evidence" in wired["properties"]

    def test_registered_tools_match_contract_registry(self):
        tools = _load_tools()
        assert set(tools) == set(C.TOOL_CONTRACTS), (
            "every registered MCP tool must have an input/output contract (and vice versa)"
        )

    def test_every_tool_wires_an_output_schema(self):
        # FastMCP 3.4.2 supports @mcp.tool(output_schema=...); each tool must wire one
        # (a single object type, per the MCP spec limitation) so the output contract is
        # discoverable through the MCP tools API, not just contracts_json_schema().
        cap = _load_capture()
        for name in C.TOOL_CONTRACTS:
            schema = cap.output_schemas.get(name)
            assert schema is not None, f"{name} registered without an output_schema"
            assert schema.get("type") == "object", f"{name} output_schema must be an object type"

    def test_wired_schema_strips_required_so_error_returns_pass(self):
        # FastMCP raises on a missing REQUIRED field, so the WIRED schema drops top-level
        # `required` — a handler's {"error": ...} return (which omits the success fields) then
        # passes runtime validation. The STRICT model, by contrast, still rejects it.
        for name in C.TOOL_CONTRACTS:
            schema = C.output_schema_for(name)
            assert "required" not in schema, name
            # R8: actually VALIDATE the error instance against the wired schema (not just
            # assert `required` is absent) — proves the {"error": ...} return really passes
            # runtime validation. Raises jsonschema ValidationError here if it does not.
            Draft202012Validator(schema).validate({"error": "boom"})
        # the strict published model is NOT lenient — it has real required fields
        with pytest.raises(ValidationError):
            C.NdviProfileOutput.model_validate({"error": "boom"})  # missing required `geometry`

    def test_real_fastmcp_accepts_the_wired_output_schemas(self):
        # registering on a REAL FastMCP must not raise — proves the wired output_schemas
        # are valid single-object MCP schemas (the _CaptureMCP stub can't catch a real
        # rejection at registration time).
        from fastmcp import FastMCP

        mcp = FastMCP("contract-test")
        register_stac_tools(mcp)
        register_spatial_tools(mcp)
        register_change_tools(mcp)
        register_pattern_tools(mcp)
        register_climate_tools(mcp)
        register_forecast_tools(mcp)
        register_pattern_alert_tools(mcp)

    @pytest.mark.parametrize("tool", sorted(BAD_ARGS))
    def test_handler_enforces_input_contract(self, tool):
        # the wired check_input runs FIRST, so a contract violation raises before
        # any network/DuckDB I/O — proving the contract is enforced at the boundary.
        handler = _load_tools()[tool]
        with pytest.raises(ValidationError):
            asyncio.run(handler(**BAD_ARGS[tool]))

    @pytest.fixture
    def no_imagery_io(self, monkeypatch):
        """Fail the test if an NDVI handler reaches its Sentinel-2 search, its first I/O."""

        async def _reached_io(*_args, **_kwargs):
            raise AssertionError("the handler reached imagery I/O before refusing its arguments")

        monkeypatch.setattr(_ndvi, "_collect_ndvi_observations", _reached_io)

    def test_vegetation_stress_refuses_an_out_of_range_lookback_before_any_io(self, no_imagery_io):
        # #221: the contract let 1,000,000 days through, and the handler then failed computing its
        # window start (OverflowError "date value out of range") instead of refusing.
        handler = _load_tools()["vegetation_stress"]
        with pytest.raises(ValidationError):
            asyncio.run(
                handler(geometry={"type": "Point", "coordinates": [0, 0]}, lookback_days=1_000_000)
            )

    @pytest.mark.parametrize(
        ("tool", "extra"),
        [
            ("ndvi_profile", {"start_date": "2024-01-01", "end_date": "2024-02-01"}),
            ("vegetation_stress", {}),
        ],
    )
    def test_ndvi_tools_refuse_a_coordinate_past_the_float_range_before_any_io(
        self, tool, extra, no_imagery_io
    ):
        # #221: the coordinate passed the contract and shapely raised OverflowError on it in
        # _validate_area, a tool error instead of the invalid-arguments refusal.
        handler = _load_tools()[tool]
        with pytest.raises(ValidationError):
            asyncio.run(handler(geometry={"type": "Point", "coordinates": [10**400, 0]}, **extra))


# (tool, param, settings attribute) — params whose default must track settings.DEFAULT_*.
# Both the handler-signature default (the runtime-binding source FastMCP fills omitted args from)
# AND the contract Field default now reference settings.DEFAULT_* directly, so the published JSON
# schema can no longer drift from runtime behaviour under a config override (Greptile #49 — T-Rex
# showed a literal Field(20) going stale when DEFAULT_CLOUD_COVER_MAX was overridden). These asserts
# guard against either side being re-hardcoded back to a literal.
_SETTINGS_BOUND_DEFAULTS = [
    ("stac_search", "cloud_cover_max", "DEFAULT_CLOUD_COVER_MAX"),
    ("ndvi_profile", "cloud_cover_max", "DEFAULT_CLOUD_COVER_MAX"),
    ("vegetation_stress", "ndvi_threshold", "DEFAULT_NDVI_STRESS_THRESHOLD"),
    ("vegetation_stress", "drop_threshold", "DEFAULT_NDVI_DROP_THRESHOLD"),
    ("change_detect", "min_change_area_ha", "DEFAULT_CHANGE_MIN_AREA_HA"),
    ("change_detect", "confidence_threshold", "DEFAULT_CHANGE_CONFIDENCE"),
    ("climate_enrich_ndvi", "cloud_cover_max", "DEFAULT_CLOUD_COVER_MAX"),
    # B5: the forecast horizon default tracks settings.FORECAST_MAX_DAYS (override-safe like the rest)
    ("forecast_query", "forecast_days", "FORECAST_MAX_DAYS"),
]


class TestDefaultsTrackSettings:
    """R6 — the runtime default for an omitted tool argument is bound from the *handler
    function signature* (what FastMCP fills from), NOT the Pydantic contract default (which is
    only the published JSON-Schema advertisement). Both must equal the single source of truth,
    ``settings.DEFAULT_*`` — a drift in either hands callers a stale default."""

    @pytest.mark.parametrize("tool, param, attr", _SETTINGS_BOUND_DEFAULTS)
    def test_handler_signature_default_tracks_settings(self, tool, param, attr):
        # The RUNTIME-binding source: FastMCP fills an omitted arg from the handler signature.
        handler = _load_tools()[tool]
        sig_default = inspect.signature(handler).parameters[param].default
        assert sig_default == getattr(settings, attr), (
            f"{tool}({param}=) runtime default must track settings.{attr}"
        )

    @pytest.mark.parametrize("tool, param, attr", _SETTINGS_BOUND_DEFAULTS)
    def test_contract_field_default_tracks_settings(self, tool, param, attr):
        # The PUBLISHED schema default must not advertise a stale value either.
        model = C.TOOL_CONTRACTS[tool]["input"]
        field_default = model.model_fields[param].default
        assert field_default == getattr(settings, attr), (
            f"{tool} contract {param} default must track settings.{attr}"
        )


class TestOutputModels:
    def test_feature_collection_outputs(self):
        payload = {
            "type": "FeatureCollection",
            "features": [{"type": "Feature", "id": "x", "geometry": {}, "properties": {}}],
            "metadata": {"total_results": 1, "elapsed_ms": 5},
        }
        for model in (C.StacSearchOutput, C.SpatialQueryOutput, C.ChangeDetectOutput):
            obj = model.model_validate(payload)
            assert obj.type == "FeatureCollection"

    def test_stac_download_output(self):
        C.StacDownloadOutput.model_validate(
            {
                "item_id": "x",
                "collection": "c",
                "datetime": None,
                "bands": {"B04": {"href": "https://...", "status": "url_ready"}},
                "metadata": {"bbox": [0, 0, 1, 1], "elapsed_ms": 3},
            }
        )

    def test_ndvi_profile_output(self):
        obj = C.NdviProfileOutput.model_validate(
            {
                "geometry": {"type": "Point", "coordinates": [0, 0]},
                "profile": [
                    {
                        "date": "2024-01-01",
                        "ndvi_mean": 0.6,
                        "ndvi_std": 0.1,
                        "cloud_cover": 4.0,
                        "item_id": "i",
                    }
                ],
                "statistics": {"mean": 0.6},
                "alerts": {"vegetation_stress_detected": False},
                "metadata": {"images_processed": 1},
            }
        )
        assert obj.profile[0].ndvi_mean == 0.6

    def test_ndvi_profile_entry_minimal(self):
        # only ndvi_mean is required; the rest are optional
        C.NdviProfileEntry.model_validate({"date": "2024-01-01", "ndvi_mean": 0.5})

    def test_ndvi_profile_entry_allows_null_date(self):
        # the producer emits date=None when the STAC item has no datetime
        obj = C.NdviProfileEntry.model_validate({"date": None, "ndvi_mean": 0.5})
        assert obj.date is None

    def test_vegetation_stress_output(self):
        C.VegetationStressOutput.model_validate(
            {
                "geometry": {"type": "Point"},
                "analysis": {"current_ndvi": 0.2, "baseline_ndvi": 0.5, "ndvi_change": -0.3},
                "alerts": [{"type": "low_ndvi", "severity": "warning"}],
                "metadata": {"recent_observations": 3},
            }
        )

    def test_discover_patterns_output(self):
        C.DiscoverPatternsOutput.model_validate(
            {
                "patterns": [{"gradual_items": [], "support": 0.8}],
                "statistics": {"n_patterns": 1, "runtime_seconds": 1.2},
                "metadata": {"algorithm": "graank"},
            }
        )

    def test_discover_patterns_cv_output(self):
        C.DiscoverPatternsCvOutput.model_validate(
            {
                "patterns": [
                    {
                        "pattern_id": 0,
                        "pattern": "temperature_2m+, ndvi_mean-",
                        "support": 0.82,
                        "confidence": None,
                        "zones_validated": ["z1", "z2", "z3", "z4", "z5"],
                        "n_folds": 5,
                        "stability": 1.0,
                    }
                ],
                "folds": [{"fold": 1, "validation_zones": ["z1"], "train_zones": ["z2"]}],
                "statistics": {"n_validated_patterns": 1},
                "metadata": {"algorithm": "graank"},
            }
        )

    def test_climate_percentiles_output(self):
        C.ClimatePercentilesOutput.model_validate(
            {
                "zone_id": "montpellier",
                "normals": [{"month": 1, "variable": "temperature_2m", "p50": 8.0}],
                "variables": ["temperature_2m"],
                "statistics": {"n_normals": 1},
                "metadata": {"window_label": "WMO_1991_2020"},
            }
        )

    def test_forecast_query_output(self):
        obj = C.ForecastQueryOutput.model_validate(
            {
                "zone_id": "montpellier",
                "forecast": [{"date": "2026-06-21", "temperature_2m": 26.6}],
                "statistics": {"temperature_2m": {"min": 26.6, "max": 26.6, "mean": 26.6, "n": 1}},
                "metadata": {"source": "openmeteo_physical", "horizon_days": 1},
            }
        )
        assert obj.forecast[0]["temperature_2m"] == 26.6
        # point-mode variant
        C.ForecastQueryOutput.model_validate(
            {"point": {"latitude": 43.6, "longitude": 3.5}, "forecast": []}
        )

    def test_pattern_alert_output(self):
        C.PatternAlertOutput.model_validate(
            {
                "zone_id": "montpellier",
                "alerts": [
                    {
                        "pattern_id": 1,
                        "pattern": "ndvi_mean+, (humidity-) +7.0 days",
                        "support": 0.65,
                        "stability": 0.8,
                        "provisional_score": 0.2,
                        "confidence": None,
                    }
                ],
                "statistics": {"n_alerts": 1},
                "metadata": {"confidence_status": "experimental_unratified"},
            }
        )

    def test_error_response(self):
        C.ErrorResponse.model_validate({"error": "boom"})
        C.ErrorResponse.model_validate({"error": "boom", "metadata": {"x": 1}})

    def test_outputs_allow_extra_keys(self):
        # additive metadata must never break a consumer (extra="allow")
        obj = C.StacSearchOutput.model_validate(
            {"type": "FeatureCollection", "features": [], "metadata": {}, "future_field": 123}
        )
        assert obj.model_dump().get("future_field") == 123


class TestNormalsReadBindingContract:
    @staticmethod
    def _success():
        return {
            "zone_id": "montpellier",
            "metadata": {
                "dataset": "science:normals:montpellier",
                "artifact_version": "sha256:" + "a" * 64,
                "artifact_size_bytes": 1024,
                "future_metadata": {"compatible": True},
            },
            "confidence": None,
        }

    def test_success_requires_typed_binding_and_preserves_additive_metadata(self):
        payload = self._success()
        parsed = C.ClimateNormalsOutput.model_validate(payload)
        assert isinstance(parsed.metadata, C.ClimateNormalsMetadata)
        assert parsed.metadata.model_dump()["future_metadata"] == {"compatible": True}
        schema = C.contracts_json_schema()["climate_normals"]["output"]
        success = schema["anyOf"][0]
        assert "metadata" in success["required"]
        metadata = schema["$defs"]["ClimateNormalsMetadata"]
        assert set(metadata["required"]) == {"dataset", "artifact_version", "artifact_size_bytes"}
        Draft202012Validator(schema).validate(payload)

    @pytest.mark.parametrize(
        "missing", ["metadata", "dataset", "artifact_version", "artifact_size_bytes"]
    )
    def test_success_refuses_a_missing_binding_field(self, missing):
        payload = self._success()
        if missing == "metadata":
            del payload[missing]
        else:
            del payload["metadata"][missing]
        with pytest.raises(ValidationError):
            C.ClimateNormalsOutput.model_validate(payload)
        # The union's error variant allows extra fields, so validate the success branch
        # itself when checking required success fields.
        published = C.contracts_json_schema()["climate_normals"]["output"]
        success = {**published["anyOf"][0], "$defs": published["$defs"]}
        assert not Draft202012Validator(success).is_valid(payload)

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("artifact_version", "a" * 64),
            ("artifact_version", "sha256:" + "A" * 64),
            ("artifact_version", "sha256:" + "a" * 63),
            ("artifact_version", "sha256:" + "a" * 64 + "\n"),
            ("artifact_size_bytes", 0),
            ("artifact_size_bytes", -1),
            ("artifact_size_bytes", True),
            ("artifact_size_bytes", "1024"),
            ("artifact_size_bytes", 1.5),
        ],
    )
    def test_success_refuses_an_invalid_binding(self, field, value):
        payload = self._success()
        payload["metadata"][field] = value
        with pytest.raises(ValidationError):
            C.ClimateNormalsOutput.model_validate(payload)
        assert not Draft202012Validator(C.output_schema_for("climate_normals")).is_valid(payload)

    def test_refusal_remains_valid_without_success_metadata(self):
        refusal = {"error": "stored normals unavailable", "confidence": None}
        Draft202012Validator(C.output_schema_for("climate_normals")).validate(refusal)
        Draft202012Validator(C.contracts_json_schema()["climate_normals"]["output"]).validate(
            refusal
        )
        with pytest.raises(ValidationError):
            C.ClimateNormalsOutput.model_validate(refusal)


class TestContractRegistry:
    def test_json_schema_exports_input_and_output_for_all_tools(self):
        schema = C.contracts_json_schema()
        assert set(schema) == set(C.TOOL_CONTRACTS)
        for spec in schema.values():
            Draft202012Validator.check_schema(spec["input"])
            Draft202012Validator.check_schema(spec["output"])
            assert "properties" in spec["input"]
            # output is published as anyOf(success, ErrorResponse) — both branches present
            branches = spec["output"]["anyOf"]
            assert len(branches) == 2
            assert all("properties" in b for b in branches)
            assert any("error" in b.get("properties", {}) for b in branches)

    def test_published_success_branch_is_strict(self):
        # the published success branch keeps real required fields (meaningful contract),
        # unlike the lenient wired schema
        schema = C.contracts_json_schema()
        for name in ("ndvi_profile", "stac_download", "discover_patterns", "stac_search"):
            success = schema[name]["output"]["anyOf"][0]
            assert success.get("required"), f"{name} published success branch should be strict"

    def test_nested_defs_are_hoisted_so_refs_resolve(self):
        # ndvi_profile's success model nests NdviProfileEntry via $ref; the $defs must be hoisted
        # to the per-tool root so the $ref resolves under anyOf (not orphaned inside the branch)
        out = C.contracts_json_schema()["ndvi_profile"]["output"]
        assert "$defs" in out and "NdviProfileEntry" in out["$defs"]
        success = out["anyOf"][0]
        assert "$defs" not in success  # hoisted out of the branch
        ref = success["properties"]["profile"]["items"]["$ref"]
        assert ref == "#/$defs/NdviProfileEntry"

    def test_output_schema_for_is_a_single_object(self):
        # MCP requires the wired output schema to be a single object type (no anyOf)
        for name in C.TOOL_CONTRACTS:
            wired = C.output_schema_for(name)
            Draft202012Validator.check_schema(wired)
            assert wired.get("type") == "object"
            assert "anyOf" not in wired


class TestToolCapabilities:
    def test_capability_registry_matches_contracts_and_registered_tools(self):
        names = set(C.TOOL_CONTRACTS)

        assert names == set(C.TOOL_CAPABILITIES)
        assert names == set(_load_tools())
        assert len(names) == 16

    def test_catalog_embeds_strict_schemas_and_derived_persistence(self):
        catalog = C.tool_capabilities_json()
        contracts = C.contracts_json_schema()

        assert catalog["schema_version"] == "geoai-tool-capabilities/v1"
        assert catalog["tool_count"] == 16
        assert catalog["metadata_only"] is True
        assert catalog["timeout_classes_enforced_by_geoai"] is False
        assert set(catalog["tools"]) == set(C.TOOL_CONTRACTS)

        for name, tool in catalog["tools"].items():
            assert tool["input_schema"] == contracts[name]["input"]
            assert tool["output_schema"] == contracts[name]["output"]
            Draft202012Validator.check_schema(tool["input_schema"])
            Draft202012Validator.check_schema(tool["output_schema"])

            write_output = C.TOOL_CONTRACTS[name]["input"].model_fields.get("write_output")
            assert tool["persistence"]["supported"] is (write_output is not None)
            assert tool["persistence"]["default"] is (
                bool(write_output.default) if write_output is not None else False
            )

    def test_persistence_metadata_matches_handler_signatures(self):
        catalog = C.tool_capabilities_json()["tools"]

        for name, handler in _load_tools().items():
            handler_parameter = inspect.signature(handler).parameters.get("write_output")
            contract_field = C.TOOL_CONTRACTS[name]["input"].model_fields.get("write_output")

            assert (handler_parameter is not None) is (contract_field is not None)
            if handler_parameter is not None and contract_field is not None:
                assert handler_parameter.default is contract_field.default
                assert catalog[name]["persistence"]["default"] is handler_parameter.default

    def test_connector_sensitive_profiles_are_explicit(self):
        catalog = C.tool_capabilities_json()["tools"]

        assert catalog["spatial_query"]["profile"] == "tenant_read"
        assert catalog["discover_patterns"]["profile"] == "tenant_derive"
        assert catalog["discover_patterns"]["persistence"] == {
            "supported": True,
            "default": True,
        }
        assert catalog["forecast_query"]["persistence"] == {
            "supported": True,
            "default": False,
        }
        assert catalog["pattern_alert"]["profile"] == "experimental"
        assert catalog["pattern_alert"]["stability_status"] == "experimental"
