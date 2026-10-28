import duckdb

print("GRAANK VALIDATED")
duckdb.sql(
    "SELECT count(*) AS rows, count(confidence) AS non_null_confidence FROM read_parquet('objects/climate/o3/holdout/q3_2023/graank_validated.parquet')"
).show()
duckdb.sql("""
           COPY (SELECT * FROM read_parquet('objects/climate/o3/holdout/q3_2023/graank_validated.parquet'))
           TO 'graank_validated.csv' (HEADER, DELIMITER ',')
           """)
# -- 30 rows, 0 non-null confidence values

print("T-GRAANK VALIDATED")
duckdb.sql(
    "SELECT count(*) AS rows, count(confidence) AS non_null_confidence FROM read_parquet('objects/climate/o3/holdout/q3_2023/tgrad_validated.parquet')"
).show()
duckdb.sql("""
           COPY (SELECT * FROM read_parquet('objects/climate/o3/holdout/q3_2023/tgrad_validated.parquet'))
           TO 'tgrad_validated.csv' (HEADER, DELIMITER ',')
           """)
# -- 149 rows, 0 non-null confidence values
