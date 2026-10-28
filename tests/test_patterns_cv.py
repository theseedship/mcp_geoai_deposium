from __future__ import annotations

import pandas as pd
import pytest

from src.patterns.cross_validation import (
    canonical_pattern_key,
    cross_validate_mined_patterns,
    make_zone_folds,
)
from src.tools import patterns as patterns_tool
from src.tools.patterns import _load_cv_zone_frame, register_pattern_tools


def _pattern(a: str, b: str, support: float) -> dict:
    return {
        "gradual_items": [
            {"attribute": a, "direction": "+"},
            {"attribute": b, "direction": "-"},
        ],
        "support": support,
        "metrics": {"density": support},
    }


def _result(*patterns: dict) -> dict:
    return {
        "algorithm": "GRAANK",
        "algorithm_key": "graank",
        "patterns": list(patterns),
        "runtime_seconds": 0.0,
        "invalid_count": 0,
    }


class _CaptureMCP:
    def __init__(self):
        self.tools = {}

    def tool(self, *_args, **_kwargs):
        def deco(fn):
            self.tools[fn.__name__] = fn
            return fn

        return deco


def _cv_tool():
    m = _CaptureMCP()
    register_pattern_tools(m)
    return m.tools["discover_patterns_cv"]


def test_make_zone_folds_round_robin():
    folds = make_zone_folds(["z1", "z2", "z3", "z4", "z5", "z6"], 3)
    assert folds == [["z1", "z4"], ["z2", "z5"], ["z3", "z6"]]


def test_canonical_pattern_key_is_order_insensitive_for_graank():
    p1 = _pattern("temperature_2m", "ndvi_mean", 0.8)
    p2 = {
        "gradual_items": list(reversed(p1["gradual_items"])),
        "support": 0.8,
        "metrics": {},
    }
    assert canonical_pattern_key(p1, algorithm="graank") == canonical_pattern_key(
        p2, algorithm="graank"
    )


def test_cross_validate_mined_patterns_sets_o3_fields_without_confidence():
    zones = ["z1", "z2", "z3", "z4", "z5"]
    common = _pattern("temperature_2m", "ndvi_mean", 0.8)
    partial = _pattern("precipitation_sum", "ndvi_mean", 0.7)
    mined = {
        "z1": _result(common | {"support": 0.81}, partial | {"support": 0.71}),
        "z2": _result(common | {"support": 0.82}, partial | {"support": 0.72}),
        "z3": _result(common | {"support": 0.83}),
        "z4": _result(common | {"support": 0.84}),
        "z5": _result(common | {"support": 0.85}),
    }

    library, folds, stats = cross_validate_mined_patterns(
        mined,
        zone_ids=zones,
        algorithm="graank",
        min_support=0.5,
        n_folds=5,
        min_validated_zones=1,
    )

    assert [f["validation_zones"] for f in folds] == [["z1"], ["z2"], ["z3"], ["z4"], ["z5"]]
    assert stats["n_candidate_patterns"] == 2
    assert stats["n_validated_patterns"] == 2

    by_pattern = {record.pattern: record for record in library.patterns}
    common_record = by_pattern["temperature_2m+, ndvi_mean-"]
    assert common_record.confidence is None
    assert common_record.n_folds == 5
    assert common_record.stability == pytest.approx(1.0)
    assert common_record.zones_validated == zones

    partial_record = by_pattern["precipitation_sum+, ndvi_mean-"]
    assert partial_record.stability == pytest.approx(0.4)
    assert partial_record.zones_validated == ["z1", "z2"]


def test_cross_validate_rejects_missing_zone_result():
    with pytest.raises(ValueError, match="missing mined results"):
        cross_validate_mined_patterns(
            {"z1": _result()},
            zone_ids=["z1", "z2", "z3", "z4", "z5"],
            algorithm="graank",
            min_support=0.5,
            n_folds=5,
        )


async def test_discover_patterns_cv_tool_happy_path(monkeypatch):
    monkeypatch.setattr("src.tools.patterns.settings.S3_BUCKET", "deposium")
    zones = ["z1", "z2", "z3", "z4", "z5"]

    def _fake_load(source_uris, **_kwargs):
        return source_uris[0].rsplit("/", 1)[-1].removesuffix(".parquet")

    def _fake_mine(df, **_kwargs):
        support = 0.75 + (0.01 * int(df[-1]))
        return (
            _result(_pattern("temperature_2m", "ndvi_mean", support)),
            10,
            [
                "temperature_2m",
                "ndvi_mean",
            ],
        )

    monkeypatch.setattr("src.tools.patterns._load_cv_zone_frame", _fake_load)
    monkeypatch.setattr("src.tools.patterns._mine_cv_frame", _fake_mine)

    tool = _cv_tool()
    result = await tool(
        zone_ids=zones,
        source_template="climate/holdout/unified/2023_07/{zone_id}.parquet",
        min_support=0.5,
        n_folds=5,
        period_start="2023-07-01",
        period_end="2023-07-31",
        truncated_sources={"z5": ["2023_07"]},
    )

    assert "error" not in result, result
    assert result["statistics"]["n_validated_patterns"] == 1
    assert result["patterns"][0]["confidence"] is None
    assert result["patterns"][0]["n_folds"] == 5
    assert result["patterns"][0]["period_start"] == "2023-07-01"
    assert result["metadata"]["truncated_sources"] == {"z5": ["2023_07"]}
    assert result["output"] is None


async def test_discover_patterns_cv_default_template_uses_canonical_unified(monkeypatch):
    monkeypatch.setattr("src.tools.patterns.settings.S3_BUCKET", "deposium")
    captured_sources = []

    def _fake_load(source_uris, **_kwargs):
        captured_sources.append(source_uris)
        return source_uris[0].rsplit("/", 1)[-1].removesuffix(".parquet")

    def _fake_mine(df, **_kwargs):
        support = 0.75 + (0.01 * int(df[-1]))
        return (_result(_pattern("temperature_2m", "ndvi_mean", support)), 10, [])

    monkeypatch.setattr("src.tools.patterns._load_cv_zone_frame", _fake_load)
    monkeypatch.setattr("src.tools.patterns._mine_cv_frame", _fake_mine)

    tool = _cv_tool()
    result = await tool(zone_ids=["z1", "z2", "z3", "z4", "z5"])

    assert "error" not in result, result
    assert captured_sources[0] == ["s3://deposium/climate/unified/z1.parquet"]


async def test_discover_patterns_cv_rejects_cross_bucket_source(monkeypatch):
    monkeypatch.setattr("src.tools.patterns.settings.S3_BUCKET", "deposium")
    tool = _cv_tool()
    result = await tool(
        zone_ids=["z1", "z2", "z3", "z4", "z5"],
        sources={
            "z1": "s3://other/z1.parquet",
            "z2": "z2.parquet",
            "z3": "z3.parquet",
            "z4": "z4.parquet",
            "z5": "z5.parquet",
        },
    )
    assert "error" in result and "configured bucket" in result["error"]


async def test_discover_patterns_cv_rejects_unsafe_full_output_uri(monkeypatch):
    monkeypatch.setattr("src.tools.patterns.settings.S3_BUCKET", "deposium")
    tool = _cv_tool()
    result = await tool(
        zone_ids=["z1", "z2", "z3", "z4", "z5"],
        write_output=True,
        output_key="s3://deposium/climate/o3/bad';DROP.parquet",
    )
    assert "error" in result and "Invalid output_key" in result["error"]


async def test_discover_patterns_cv_rejects_template_without_zone_id(monkeypatch):
    monkeypatch.setattr("src.tools.patterns.settings.S3_BUCKET", "deposium")
    tool = _cv_tool()
    result = await tool(
        zone_ids=["z1", "z2", "z3", "z4", "z5"],
        source_template="climate/unified/shared.parquet",
    )
    assert "error" in result and "{zone_id}" in result["error"]


async def test_discover_patterns_cv_tgrad_projection_keeps_driver_columns(monkeypatch):
    monkeypatch.setattr("src.tools.patterns.settings.S3_BUCKET", "deposium")
    captured_columns = []

    def _fake_load(_source_uris, **kwargs):
        captured_columns.append(kwargs["columns"])
        return "zone-frame"

    def _fake_mine(_df, **_kwargs):
        return (
            _result(_pattern("temperature_2m", "ndvi_mean", 0.8)),
            10,
            [
                "date",
                "temperature_2m",
                "ndvi_mean",
            ],
        )

    monkeypatch.setattr("src.tools.patterns._load_cv_zone_frame", _fake_load)
    monkeypatch.setattr("src.tools.patterns._mine_cv_frame", _fake_mine)

    tool = _cv_tool()
    result = await tool(
        zone_ids=["z1", "z2", "z3", "z4", "z5"],
        algorithm="tgrad",
        columns=["temperature_2m"],
        order_by="date",
        target_column="ndvi_mean",
    )

    assert "error" not in result, result
    assert captured_columns
    assert all(cols == ["temperature_2m", "date", "ndvi_mean"] for cols in captured_columns)


async def test_discover_patterns_cv_tgrad_rejects_dropped_driver_columns(monkeypatch):
    monkeypatch.setattr("src.tools.patterns.settings.S3_BUCKET", "deposium")
    tool = _cv_tool()
    result = await tool(
        zone_ids=["z1", "z2", "z3", "z4", "z5"],
        algorithm="tgrad",
        order_by="date",
        target_column="ndvi_mean",
        drop_columns=["date"],
    )
    assert "error" in result and "drop_columns" in result["error"]


async def test_discover_patterns_cv_write_output_requires_output_key(monkeypatch):
    monkeypatch.setattr("src.tools.patterns.settings.S3_BUCKET", "deposium")
    tool = _cv_tool()
    result = await tool(zone_ids=["z1", "z2", "z3", "z4", "z5"], write_output=True)
    assert "error" in result and "output_key" in result["error"]


def test_load_cv_zone_frame_caps_concatenated_segments(monkeypatch):
    calls = []

    def _fake_load_dataframe(uri, **kwargs):
        calls.append((uri, kwargs["limit"]))
        return pd.DataFrame(
            {
                "date": pd.date_range("2024-01-01", periods=3),
                "value": [1, 2, 3],
            }
        )

    monkeypatch.setattr(patterns_tool, "load_dataframe", _fake_load_dataframe)

    frame = _load_cv_zone_frame(
        ["s3://deposium/a.parquet", "s3://deposium/b.parquet"],
        source_kind="parquet",
        columns=None,
        order_by="date",
        drop_columns=None,
        limit=4,
    )

    assert calls == [("s3://deposium/a.parquet", 4), ("s3://deposium/b.parquet", 4)]
    assert len(frame) == 4
