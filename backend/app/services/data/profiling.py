import json
import math
from pathlib import Path

import duckdb

from app.services.data.ingest import TABLE_NAME

SAMPLE_ROWS = 5


def _clean(value):
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, (int, float, str, bool)):
        return value
    return str(value)


def profile_table(db_path: Path) -> dict:
    """Column summary + sample rows, computed by DuckDB without loading the table into pandas."""
    with duckdb.connect(str(db_path), read_only=True) as con:
        row_count = con.execute(f"SELECT COUNT(*) FROM {TABLE_NAME}").fetchone()[0]
        summary = con.execute(f"SUMMARIZE {TABLE_NAME}").fetchdf()
        sample = con.execute(f"SELECT * FROM {TABLE_NAME} LIMIT {SAMPLE_ROWS}").fetchdf()

    columns = [
        {
            "name": row["column_name"],
            "type": row["column_type"],
            "null_pct": round(float(row["null_percentage"]), 2),
            "approx_unique": _clean(row["approx_unique"]),
            "min": _clean(row["min"]),
            "max": _clean(row["max"]),
            "mean": _clean(row["avg"]),
        }
        for _, row in summary.iterrows()
    ]
    sample_rows = json.loads(sample.to_json(orient="records", date_format="iso"))
    return {"row_count": row_count, "columns": columns, "sample": sample_rows}
