"""Pick a chart from the shape of a result table, so the frontend never has to guess."""

_NUMERIC_TYPES = ("INT", "DOUBLE", "FLOAT", "DECIMAL", "REAL", "NUMERIC")
_TEMPORAL_TYPES = ("DATE", "TIMESTAMP", "TIME")
MAX_BAR_CATEGORIES = 30
MAX_SERIES = 3
# Series share one y-axis, so only plot measures whose peaks are within this ratio of the first.
MAX_SCALE_RATIO = 10


def _is(type_name: str, kinds: tuple[str, ...]) -> bool:
    upper = type_name.upper()
    return any(k in upper for k in kinds)


def _peak(rows: list[list], index: int) -> float:
    values = [abs(r[index]) for r in rows if isinstance(r[index], (int, float))]
    return max(values, default=0.0)


def _comparable(columns: list[str], numeric: list[str], rows: list[list]) -> list[str]:
    base = _peak(rows, columns.index(numeric[0])) or 1.0
    picked = [numeric[0]]
    for col in numeric[1:]:
        peak = _peak(rows, columns.index(col))
        if peak and max(peak, base) / min(peak, base) <= MAX_SCALE_RATIO:
            picked.append(col)
    return picked[:MAX_SERIES]


def suggest_chart(columns: list[str], types: list[str], rows: list[list]) -> dict | None:
    row_count = len(rows)
    if row_count < 2 or len(columns) < 2:
        return None
    temporal = [c for c, t in zip(columns, types) if _is(t, _TEMPORAL_TYPES)]
    numeric = [c for c, t in zip(columns, types) if _is(t, _NUMERIC_TYPES) and not _is(t, _TEMPORAL_TYPES)]
    categorical = [c for c in columns if c not in temporal and c not in numeric]
    if not numeric:
        return None
    series = _comparable(columns, numeric, rows)
    if temporal:
        return {"kind": "line", "x": temporal[0], "y": series}
    if categorical and row_count <= MAX_BAR_CATEGORIES:
        return {"kind": "bar", "x": categorical[0], "y": series}
    return None
