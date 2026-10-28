"""Unit tests for the DuckDB query builder — pure, no DB, runs in base 3.11."""

import pytest

from src.patterns.parquet_bridge import build_select_query


class TestBuildSelectQuery:
    def test_star_when_no_columns(self):
        sql = build_select_query("parquet")
        assert sql == "SELECT * FROM read_parquet($source)"

    def test_columns_are_double_quoted(self):
        sql = build_select_query("parquet", columns=["precip", "ndvi"])
        assert sql == 'SELECT "precip", "ndvi" FROM read_parquet($source)'

    def test_reserved_word_column_is_quoted(self):
        # 'order'/'end' are DuckDB reserved words — unquoted they error.
        sql = build_select_query("parquet", columns=["order", "end"])
        assert '"order"' in sql and '"end"' in sql

    def test_order_by_is_quoted(self):
        sql = build_select_query("parquet", order_by="date")
        assert sql.endswith('ORDER BY "date"')

    def test_reader_per_kind(self):
        assert "read_csv_auto" in build_select_query("csv")
        assert "read_json_auto" in build_select_query("json")
        assert "read_parquet" in build_select_query("geoparquet")

    def test_unknown_kind_raises(self):
        with pytest.raises(ValueError, match="Unknown source kind"):
            build_select_query("xlsx")

    def test_invalid_column_raises(self):
        with pytest.raises(ValueError, match="Invalid column identifier"):
            build_select_query("parquet", columns=["ok", "bad; DROP TABLE t"])

    def test_invalid_order_by_raises(self):
        with pytest.raises(ValueError, match="Invalid order_by identifier"):
            build_select_query("parquet", order_by="1; DROP TABLE t")

    def test_limit_clamped_and_appended(self):
        sql = build_select_query("parquet", limit=50)
        assert sql.endswith("LIMIT 50")
