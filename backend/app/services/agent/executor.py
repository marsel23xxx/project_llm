import datetime as dt
import decimal
import math
import threading
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import duckdb

# Locked so a query cannot turn file access back on with SET.
SAFE_CONFIG = {
    "enable_external_access": False,
    "lock_configuration": True,
    "threads": 2,
    "memory_limit": "512MB",
}


class ExecutionError(RuntimeError):
    pass


@dataclass
class QueryResult:
    columns: list[str]
    types: list[str]
    rows: list[list]
    truncated: bool = False
    stdout: str = ""
    value: object = None
    extra: dict = field(default_factory=dict)

    def to_event(self) -> dict:
        return {
            "type": "table",
            "columns": self.columns,
            "types": self.types,
            "rows": self.rows,
            "truncated": self.truncated,
        }


def to_jsonable(value):
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return None if math.isnan(value) or math.isinf(value) else value
    if isinstance(value, decimal.Decimal):
        return float(value)
    if isinstance(value, (dt.date, dt.datetime, dt.time)):
        return value.isoformat()
    if isinstance(value, uuid.UUID):
        return str(value)
    return str(value)


def run_sql(db_path: Path, sql: str, timeout_seconds: float, max_rows: int) -> QueryResult:
    con = duckdb.connect(str(db_path), read_only=True, config=SAFE_CONFIG)
    timer = threading.Timer(timeout_seconds, con.interrupt)
    timer.start()
    try:
        cursor = con.execute(f"SELECT * FROM ({sql}) AS hasil LIMIT {max_rows + 1}")
        columns = [d[0] for d in cursor.description]
        types = [str(d[1]) for d in cursor.description]
        rows = cursor.fetchall()
    except duckdb.InterruptException as exc:
        raise ExecutionError(f"Query melebihi batas waktu {timeout_seconds:g} detik") from exc
    except duckdb.Error as exc:
        raise ExecutionError(str(exc)) from exc
    finally:
        timer.cancel()
        con.close()

    return QueryResult(
        columns=columns,
        types=types,
        rows=[[to_jsonable(v) for v in row] for row in rows[:max_rows]],
        truncated=len(rows) > max_rows,
    )
