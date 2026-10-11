"""Executed in a fresh, resource-limited subprocess: python -I runner.py <dataset.duckdb> <result.json>"""

import resource

CPU_SECONDS = 30
MEMORY_BYTES = 1536 * 1024 * 1024
MAX_FILE_BYTES = 10 * 1024 * 1024

# Applied before any other import so the agent's code can never run without them.
resource.setrlimit(resource.RLIMIT_CPU, (CPU_SECONDS, CPU_SECONDS))
resource.setrlimit(resource.RLIMIT_AS, (MEMORY_BYTES, MEMORY_BYTES))
resource.setrlimit(resource.RLIMIT_FSIZE, (MAX_FILE_BYTES, MAX_FILE_BYTES))

import contextlib  # noqa: E402
import io  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import traceback  # noqa: E402

import duckdb  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

MAX_ROWS = 1000
MAX_STDOUT = 8000


def _table(frame: pd.DataFrame) -> dict:
    frame = frame.reset_index() if not isinstance(frame.index, pd.RangeIndex) else frame
    frame.columns = [str(c) for c in frame.columns]
    head = frame.head(MAX_ROWS)
    split = json.loads(head.to_json(orient="split", date_format="iso", index=False))

    def kind(series: pd.Series) -> str:
        if pd.api.types.is_bool_dtype(series):
            return "BOOLEAN"
        if pd.api.types.is_numeric_dtype(series):
            return "DOUBLE"
        if pd.api.types.is_datetime64_any_dtype(series):
            return "TIMESTAMP"
        return "VARCHAR"

    return {
        "type": "table",
        "columns": split["columns"],
        "types": [kind(frame[c]) for c in frame.columns],
        "rows": split["data"],
        "truncated": len(frame) > MAX_ROWS,
    }


def serialize(value) -> dict | None:
    if value is None:
        return None
    if isinstance(value, pd.DataFrame):
        return _table(value)
    if isinstance(value, pd.Series):
        return _table(value.to_frame(name=value.name or "nilai"))
    if isinstance(value, np.generic):
        value = value.item()
    return {"type": "value", "value": json.loads(json.dumps(value, default=str))}


def main() -> None:
    db_path, out_path = sys.argv[1], sys.argv[2]
    code = sys.stdin.read()

    with duckdb.connect(db_path, read_only=True, config={"enable_external_access": False}) as con:
        df = con.execute("SELECT * FROM data").df()

    scope = {"df": df, "pd": pd, "np": np, "__name__": "__agent__"}
    stdout = io.StringIO()
    try:
        with contextlib.redirect_stdout(stdout):
            exec(compile(code, "<agent>", "exec"), scope)
        payload = {"ok": True, "stdout": stdout.getvalue()[-MAX_STDOUT:], "result": serialize(scope.get("result"))}
    except Exception:
        payload = {"ok": False, "stdout": stdout.getvalue()[-MAX_STDOUT:], "error": traceback.format_exc(limit=3)[-2000:]}

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, default=str)


if __name__ == "__main__":
    main()
