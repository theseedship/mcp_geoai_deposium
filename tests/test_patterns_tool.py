"""Tests for the discover_patterns MCP tool — hermetic (no S3, no so4gp).

Covers the prefix-agnostic source/output resolution, identifier validation, and
the error branches. The full load->mine->write chain is an integration concern
(needs RustFS + so4gp) validated at deploy; the so4gp call itself is covered by
test_so4gp_contract.py and the parser by test_patterns_adapter.py.
"""

import pytest

from src.config import settings
from src.tools.patterns import (
    _infer_kind,
    _validate_key,
    register_pattern_tools,
)

# --- pure helpers -----------------------------------------------------------


class TestInferKind:
    def test_from_extension(self):
        assert _infer_kind("x.parquet", None) == "parquet"
        assert _infer_kind("s3://b/k.geoparquet", None) == "geoparquet"
        assert _infer_kind("x.csv", None) == "csv"
        assert _infer_kind("x.json", None) == "json"

    def test_geojson_not_advertised(self):
        # .geojson is intentionally NOT mapped (FeatureCollection isn't flat) —
        # it falls through to the parquet default rather than being treated as
        # a tabular json source.
        assert _infer_kind("x.geojson", None) == "parquet"

    def test_default_is_parquet(self):
        assert _infer_kind("no-extension", None) == "parquet"

    def test_explicit_overrides(self):
        assert _infer_kind("x.csv", "parquet") == "parquet"

    def test_invalid_explicit_raises(self):
        with pytest.raises(ValueError, match="Invalid source_kind"):
            _infer_kind("x", "xlsx")


class TestValidateKey:
    def test_accepts_normal_key(self):
        k = "u-uuid/space-uuid/parquet/tabular_data.parquet"
        assert _validate_key(k, role="k") == k

    @pytest.mark.parametrize("bad", ["", "../etc/passwd", "a/../b", "a b", "a'b", 'a"b', "a;b"])
    def test_rejects_unsafe(self, bad):
        with pytest.raises(ValueError):
            _validate_key(bad, role="k")


# --- tool error branches (via a capture MCP) --------------------------------


class _CaptureMCP:
    def __init__(self):
        self.tools = {}

    def tool(self, *_args, **_kwargs):  # accepts @mcp.tool(output_schema=...)
        def deco(fn):
            self.tools[fn.__name__] = fn
            return fn

        return deco


def _discover_tool():
    m = _CaptureMCP()
    register_pattern_tools(m)
    return m.tools["discover_patterns"]


async def test_unsupported_algorithm():
    discover = _discover_tool()
    r = await discover(source="auto", algorithm="nope")
    assert "Unsupported algorithm" in r["error"]


async def test_bucket_not_configured(monkeypatch):
    monkeypatch.setattr(settings, "S3_BUCKET", "")
    discover = _discover_tool()
    r = await discover(source="auto", tenant_id="u", space_id="s")
    assert "S3_BUCKET" in r["error"]


async def test_auto_requires_ids(monkeypatch):
    monkeypatch.setattr(settings, "S3_BUCKET", "deposium")
    discover = _discover_tool()
    r = await discover(source="auto")
    assert "tenant_id and space_id" in r["error"]


async def test_invalid_source_key_rejected(monkeypatch):
    monkeypatch.setattr(settings, "S3_BUCKET", "deposium")
    discover = _discover_tool()
    r = await discover(source="../../etc/passwd")
    assert "error" in r and "source key" in r["error"]


async def test_cross_bucket_source_rejected(monkeypatch):
    # Security: an s3:// source outside the configured bucket must be refused.
    monkeypatch.setattr(settings, "S3_BUCKET", "deposium")
    discover = _discover_tool()
    r = await discover(source="s3://other-bucket/secret.parquet", write_output=False)
    assert "error" in r and "configured bucket" in r["error"]


async def test_deferred_algorithm_rejected(monkeypatch):
    # Only graank is wired in Palier 1; variants must be refused until their
    # output shape is contract-tested.
    monkeypatch.setattr(settings, "S3_BUCKET", "deposium")
    discover = _discover_tool()
    r = await discover(source="auto", tenant_id="u", space_id="s", algorithm="genetic_graank")
    assert "Unsupported algorithm" in r["error"]


async def test_graank_drops_numeric_order_by_and_reports_clamped_limit(monkeypatch):
    pd = pytest.importorskip("pandas")
    monkeypatch.setattr(settings, "S3_BUCKET", "deposium")
    monkeypatch.setattr(settings, "SPATIAL_QUERY_MAX_LIMIT", 3)

    captured = {}

    def _fake_load_dataframe(*_args, **kwargs):
        captured["limit"] = kwargs["limit"]
        return pd.DataFrame(
            {
                "t": [1, 2, 3],
                "precipitation": [0.1, 0.2, 0.3],
                "ndvi": [0.4, 0.5, 0.6],
            }
        )

    # _mine_in_tempdir now forwards (fn, df, **kwargs); the DataFrame is the 2nd positional.
    def _fake_mine(_fn, df, **_kwargs):
        captured["mined_columns"] = list(df.columns)
        return {
            "algorithm": "GRAANK",
            "algorithm_key": "graank",
            "patterns": [],
            "runtime_seconds": 0.0,
            "invalid_count": 0,
        }

    monkeypatch.setattr("src.tools.patterns.load_dataframe", _fake_load_dataframe)
    monkeypatch.setattr("src.tools.patterns._mine_in_tempdir", _fake_mine)

    discover = _discover_tool()
    r = await discover(
        source="tenant/space/parquet/tabular_data.parquet",
        order_by="t",
        limit=1_000_000,
        write_output=False,
    )

    assert "error" not in r, r
    assert captured["limit"] == 3
    assert captured["mined_columns"] == ["precipitation", "ndvi"]
    assert r["metadata"]["row_limit"] == 3
    assert r["metadata"]["columns_mined"] == ["precipitation", "ndvi"]


# --- output flattening (needs pandas; ships with the patterns extra) --------


def test_patterns_to_dataframe():
    pytest.importorskip("pandas")
    from src.tools.patterns import _patterns_to_dataframe

    result = {
        "patterns": [
            {
                "gradual_items": [
                    {"attribute": "precip", "direction": "+"},
                    {"attribute": "ndvi", "direction": "+"},
                ],
                "support": 0.82,
                "metrics": {"density": 0.82, "singularity_scr": 0.12},
            }
        ]
    }
    df = _patterns_to_dataframe(result)
    from src.tools.patterns import _PATTERN_COLUMNS

    assert list(df.columns) == _PATTERN_COLUMNS
    row = df.iloc[0]
    assert row["pattern"] == "precip+, ndvi+"
    assert row["support"] == pytest.approx(0.82)
    assert row["n_items"] == 2
    # A contemporaneous (GRAANK) pattern carries no temporal annotations.
    assert bool(row["is_temporal"]) is False
    assert row["target"] is None
    assert row["max_lag_days"] is None


def test_patterns_to_dataframe_empty_keeps_schema():
    # A legitimate 0-pattern result must still carry the typed columns so it
    # persists as a valid (empty) parquet instead of a column-less frame.
    pytest.importorskip("pandas")
    from src.tools.patterns import _PATTERN_COLUMNS, _patterns_to_dataframe

    df = _patterns_to_dataframe({"patterns": []})
    assert len(df) == 0
    assert list(df.columns) == _PATTERN_COLUMNS


def test_patterns_to_dataframe_raises_on_direction_less_item():
    # R4: the parquet renderer shares the strict ``render_gradual_item`` — a malformed item
    # with no 'direction' key fails loud (a '?' must never be silently written to the parquet).
    pytest.importorskip("pandas")
    from src.tools.patterns import _patterns_to_dataframe

    result = {
        "patterns": [
            {
                "gradual_items": [{"attribute": "x"}],  # no 'direction' key
                "support": 0.5,
                "metrics": {},
            }
        ]
    }
    with pytest.raises(KeyError):
        _patterns_to_dataframe(result)


# --- Palier 2: temporal (tgrad) branches ------------------------------------


async def test_tgrad_requires_target_column():
    # The temporal requirement is checked before any S3 access.
    discover = _discover_tool()
    r = await discover(source="auto", algorithm="tgrad", order_by="date")
    assert "error" in r and "target_column" in r["error"]


async def test_tgrad_requires_order_by():
    discover = _discover_tool()
    r = await discover(source="auto", algorithm="tgrad", target_column="ndvi")
    assert "error" in r and "order_by" in r["error"]


async def test_min_support_out_of_range():
    # Range checked before any S3 access; so4gp would otherwise silently return
    # an empty set for an out-of-range threshold.
    discover = _discover_tool()
    r = await discover(source="auto", min_support=1.5)
    assert "error" in r and "min_support" in r["error"]


async def test_tgrad_min_rep_out_of_range():
    discover = _discover_tool()
    r = await discover(
        source="auto", algorithm="tgrad", target_column="ndvi", order_by="date", min_rep=1.5
    )
    assert "error" in r and "min_rep" in r["error"]


async def test_tgrad_drop_columns_conflict():
    # Dropping the time/target column is a config error, caught up front with a
    # clear message rather than a downstream "column not found".
    discover = _discover_tool()
    r = await discover(
        source="auto",
        algorithm="tgrad",
        target_column="ndvi",
        order_by="date",
        drop_columns=["date"],
    )
    assert "error" in r and "drop_columns" in r["error"]


def test_patterns_to_dataframe_temporal():
    pytest.importorskip("pandas")
    from src.tools.patterns import _patterns_to_dataframe

    result = {
        "patterns": [
            {
                "is_temporal": True,
                "gradual_items": [
                    {
                        "attribute": "ndvi",
                        "direction": "+",
                        "role": "target",
                        "lag_days": 0.0,
                        "lag_raw": None,
                    },
                    {
                        "attribute": "precip",
                        "direction": "+",
                        "role": "lagged",
                        "lag_days": 7.0,
                        "lag_raw": "+7.0 days",
                    },
                    {
                        "attribute": "temp",
                        "direction": "-",
                        "role": "lagged",
                        "lag_days": 14.0,
                        "lag_raw": "+2.0 weeks",
                    },
                ],
                "support": 0.89,
                "metrics": {"density": 0.71},
            }
        ]
    }
    row = _patterns_to_dataframe(result).iloc[0]
    assert bool(row["is_temporal"]) is True
    assert row["target"] == "ndvi"
    assert row["max_lag_days"] == pytest.approx(14.0)
    assert row["pattern"] == "ndvi+, (precip+) +7.0 days, (temp-) +2.0 weeks"
    assert row["n_items"] == 3


class TestBuildTemporalFrame:
    def _df(self):
        pd = pytest.importorskip("pandas")
        return pd.DataFrame(
            {
                "date": ["2020-01-01", "2020-01-08", "2020-01-15"],
                "precip": [10, 20, 35],
                "ndvi": [0.3, 0.4, 0.55],
            }
        )

    def test_builds_frame_and_resolves_index(self):
        from src.patterns.parquet_bridge import build_temporal_frame

        frame, attr_cols, target_idx = build_temporal_frame(
            self._df(), time_column="date", target_column="ndvi", drop_columns=None
        )
        # Time column first (as string), then numeric attributes.
        assert list(frame.columns) == ["date", "precip", "ndvi"]
        pd = pytest.importorskip("pandas")
        assert pd.api.types.is_string_dtype(frame["date"])
        assert all(isinstance(value, str) for value in frame["date"].tolist())
        assert attr_cols == ["precip", "ndvi"]
        assert target_idx == 2

    def test_target_cannot_be_time_column(self):
        from src.patterns.parquet_bridge import build_temporal_frame

        with pytest.raises(ValueError, match="numeric attribute"):
            build_temporal_frame(
                self._df(), time_column="date", target_column="date", drop_columns=None
            )

    def test_missing_target_raises(self):
        from src.patterns.parquet_bridge import build_temporal_frame

        with pytest.raises(ValueError, match="not found"):
            build_temporal_frame(
                self._df(), time_column="date", target_column="nope", drop_columns=None
            )

    def test_needs_two_numeric_attributes(self):
        pd = pytest.importorskip("pandas")
        from src.patterns.parquet_bridge import build_temporal_frame

        one_attr = pd.DataFrame({"date": ["2020-01-01", "2020-01-08"], "ndvi": [0.3, 0.4]})
        with pytest.raises(ValueError, match="at least one other numeric"):
            build_temporal_frame(
                one_attr, time_column="date", target_column="ndvi", drop_columns=None
            )
