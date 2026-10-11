"""Turn a forecasting question into an aggregated series, send it to the forecast service,
and build the result table, chart and a numbers-only answer (no LLM writes these figures)."""

import re
import statistics
from dataclasses import dataclass
from pathlib import Path

import duckdb
import httpx
import pandas as pd

from app.ml.text import normalize
from app.services.agent.executor import SAFE_CONFIG, QueryResult
from app.services.data.ingest import TABLE_NAME

HISTORY_ROWS = 36
MIN_PERIODS = 12

MODEL_NAMES = {
    "lstm": "LSTM (TensorFlow)",
    "seasonal_naive": "seasonal naive (nilai periode yang sama di siklus sebelumnya)",
    "naive": "naive (nilai terakhir)",
    "moving_average": "rata-rata bergerak",
}
NUMBER_WORDS = {
    "satu": 1, "dua": 2, "tiga": 3, "empat": 4, "lima": 5, "enam": 6, "tujuh": 7,
    "delapan": 8, "sembilan": 9, "sepuluh": 10, "sebelas": 11, "dua belas": 12,
}  # fmt: skip
VALUE_HINTS = ("total", "omzet", "penjualan", "pendapatan", "jumlah", "qty", "nilai", "laba", "revenue", "sales")
MONTHS = ["Jan", "Feb", "Mar", "Apr", "Mei", "Jun", "Jul", "Agu", "Sep", "Okt", "Nov", "Des"]
_NUMERIC = ("INT", "DOUBLE", "FLOAT", "DECIMAL", "REAL", "NUMERIC")
_TEMPORAL = ("DATE", "TIMESTAMP")


@dataclass(frozen=True)
class Frequency:
    unit: str  # date_trunc unit
    label: str
    season: int
    pandas: str
    default_horizon: int


FREQUENCIES = {
    "hari": Frequency("day", "hari", 7, "D", 7),
    "minggu": Frequency("week", "minggu", 52, "W-MON", 4),
    "bulan": Frequency("month", "bulan", 12, "MS", 3),
}
# Units the user may say -> (frequency, periods per unit)
UNIT_ALIASES = {
    "hari": ("hari", 1), "harian": ("hari", 1), "minggu": ("minggu", 1), "mingguan": ("minggu", 1),
    "pekan": ("minggu", 1), "bulan": ("bulan", 1), "bulanan": ("bulan", 1), "kuartal": ("bulan", 3),
    "triwulan": ("bulan", 3), "semester": ("bulan", 6), "tahun": ("bulan", 12), "tahunan": ("bulan", 12),
}  # fmt: skip


class ForecastUnavailable(ValueError):
    pass


def parse_horizon(question: str) -> tuple[str | None, int | None]:
    """Return (frequency key, number of periods) as far as the question states them."""
    text = normalize(question)
    units = "|".join(sorted(UNIT_ALIASES, key=len, reverse=True))
    words = "|".join(sorted(NUMBER_WORDS, key=len, reverse=True))
    match = re.search(rf"\b(\d+|{words})\s*({units})\b", text)
    if match:
        count = int(match.group(1)) if match.group(1).isdigit() else NUMBER_WORDS[match.group(1)]
        freq, factor = UNIT_ALIASES[match.group(2)]
        return freq, count * factor
    match = re.search(rf"\b({units})\s+(depan|berikutnya|mendatang|selanjutnya)\b", text)
    if match:
        freq, factor = UNIT_ALIASES[match.group(1)]
        return freq, factor
    if re.search(r"\bbesok\b", text):
        return "hari", 1
    match = re.search(rf"\b(?:per|tiap|setiap)\s*({units})\b|\b(harian|mingguan|bulanan)\b", text)
    if match:
        return UNIT_ALIASES[match.group(1) or match.group(2)][0], None
    return None, None


def _mentioned(column: str, text: str) -> bool:
    words = set(text.split())
    return all(part in words for part in column.split("_") if part)


def pick_columns(db_path: Path, question: str) -> tuple[str, str]:
    with duckdb.connect(str(db_path), read_only=True, config=SAFE_CONFIG) as con:
        columns = con.execute(
            "SELECT column_name, data_type FROM information_schema.columns WHERE table_name = ? ORDER BY ordinal_position",
            [TABLE_NAME],
        ).fetchall()
    dates = [c for c, t in columns if any(k in t.upper() for k in _TEMPORAL)]
    numbers = [c for c, t in columns if any(k in t.upper() for k in _NUMERIC)]
    if not dates:
        raise ForecastUnavailable("Prediksi butuh kolom tanggal, dan dataset ini tidak memilikinya.")
    if not numbers:
        raise ForecastUnavailable("Prediksi butuh kolom angka, dan dataset ini tidak memilikinya.")

    text = normalize(question)
    date_col = next((c for c in dates if _mentioned(c, text)), dates[0])
    value_col = (
        next((c for c in numbers if _mentioned(c, text)), None)
        or next((c for c in numbers if any(h in c for h in VALUE_HINTS)), None)
        or numbers[0]
    )
    return date_col, value_col


def load_series(db_path: Path, date_col: str, value_col: str, freq: Frequency, agg: str) -> tuple[pd.Series, bool]:
    with duckdb.connect(str(db_path), read_only=True, config=SAFE_CONFIG) as con:
        rows = con.execute(
            f'SELECT date_trunc(\'{freq.unit}\', "{date_col}") AS periode, {agg}("{value_col}"), count(*) '
            f'FROM {TABLE_NAME} WHERE "{date_col}" IS NOT NULL AND "{value_col}" IS NOT NULL GROUP BY 1 ORDER BY 1'
        ).fetchall()
    if not rows:
        return pd.Series(dtype=float), False

    # A final period with far fewer rows than usual is still in progress; forecasting from it
    # would read as a sudden drop.
    dropped_partial = False
    if len(rows) > 3 and rows[-1][2] < 0.5 * statistics.median(r[2] for r in rows[:-1]):
        rows, dropped_partial = rows[:-1], True

    series = pd.Series([float(r[1]) for r in rows], index=pd.to_datetime([r[0] for r in rows]))
    full_index = pd.date_range(series.index[0], series.index[-1], freq=freq.pandas)
    series = series.reindex(full_index)
    series = series.fillna(0.0) if agg == "sum" else series.interpolate().bfill()
    return series, dropped_partial


def choose_frequency(db_path: Path, date_col: str, value_col: str, asked: str | None, agg: str):
    """Use the unit the user asked for; otherwise the finest one with enough, mostly non-empty periods."""
    if asked:
        freq = FREQUENCIES[asked]
        return (freq, *load_series(db_path, date_col, value_col, freq, agg))
    fallback = None
    for freq in FREQUENCIES.values():
        series, dropped = load_series(db_path, date_col, value_col, freq, agg)
        fallback = (freq, series, dropped)
        # Sparse daily sales become mostly zeros once gaps are filled; a coarser period reads better.
        if MIN_PERIODS <= len(series) <= 400 and (series != 0).mean() >= 0.6:
            return fallback
    return fallback


def _date(ts: pd.Timestamp, freq: Frequency) -> str:
    month = f"{MONTHS[ts.month - 1]} {ts.year}"
    return month if freq.unit == "month" else f"{ts.day} {month}"


def _fmt(value: float) -> str:
    text = f"{value:,.0f}" if abs(value) >= 100 else f"{value:,.2f}".rstrip("0").rstrip(".")
    return text.replace(",", "_").replace(".", ",").replace("_", ".")


async def run(db_path: Path, question: str, forecast_url: str) -> tuple[QueryResult, dict, str, str]:
    """Return (table, chart spec, note, answer)."""
    date_col, value_col = pick_columns(db_path, question)
    asked_freq, asked_horizon = parse_horizon(question)
    agg = "avg" if re.search(r"rata-rata|rerata|average", normalize(question)) else "sum"

    freq, series, dropped_partial = choose_frequency(db_path, date_col, value_col, asked_freq, agg)
    if len(series) < MIN_PERIODS:
        raise ForecastUnavailable(
            f"Data baru mencakup {len(series)} {freq.label}, sedangkan prediksi butuh minimal {MIN_PERIODS}. "
            "Coba periode yang lebih pendek (misalnya per minggu) atau tambah riwayat datanya."
        )
    horizon = asked_horizon or freq.default_horizon

    try:
        async with httpx.AsyncClient(base_url=forecast_url, timeout=150.0) as client:
            response = await client.post(
                "/forecast", json={"values": series.tolist(), "horizon": horizon, "season": freq.season}
            )
    except httpx.HTTPError as exc:
        raise ForecastUnavailable("Layanan prediksi sedang tidak tersedia. Silakan coba lagi.") from exc
    if response.status_code == 422:
        raise ForecastUnavailable(response.json().get("detail", "Data tidak bisa diprediksi."))
    if response.status_code != 200:
        raise ForecastUnavailable("Layanan prediksi sedang tidak tersedia. Silakan coba lagi.")
    body = response.json()

    forecast = body["forecast"]
    future_index = pd.date_range(series.index[-1], periods=len(forecast) + 1, freq=freq.pandas)[1:]
    history = series.iloc[-HISTORY_ROWS:]
    rows = [[ts.isoformat(), round(float(v), 2), None, None, None] for ts, v in history.items()]
    rows[-1][2] = rows[-1][1]  # join the forecast line to the last actual point
    rows += [
        [ts.isoformat(), None, round(f, 2), round(lo, 2), round(up, 2)]
        for ts, f, lo, up in zip(future_index, forecast, body["lower"], body["upper"])
    ]
    table = QueryResult(
        columns=["periode", "aktual", "prediksi", "batas_bawah", "batas_atas"],
        types=["TIMESTAMP", "DOUBLE", "DOUBLE", "DOUBLE", "DOUBLE"],
        rows=rows,
    )
    chart = {"kind": "line", "x": "periode", "y": ["aktual", "prediksi"], "dashed": ["prediksi"]}

    scores = body["scores"]
    winner = body["model"]
    compared = ", ".join(f"{name} MAE {_fmt(s['mae'])}" for name, s in sorted(scores.items(), key=lambda kv: kv[1]["mae"]))
    note = (
        f"Deret: {agg.upper()}({value_col}) per {freq.label} dari kolom {date_col}, {body['points_used']} periode. "
        f"Backtest {body['backtest_points']} periode terakhir: {compared}."
    )
    if dropped_partial:
        note += " Periode terakhir belum lengkap sehingga tidak dipakai."
    for extra in body.get("notes", []):
        note += f" {extra}"

    total = sum(forecast)
    listed = "; ".join(f"{_date(ts, freq)}: {_fmt(f)}" for ts, f in list(zip(future_index, forecast))[:6])
    answer = (
        f"Perkiraan {value_col} untuk {len(forecast)} {freq.label} ke depan"
        + (f" berjumlah sekitar {_fmt(total)}" if agg == "sum" and len(forecast) > 1 else "")
        + f". {listed}{' ...' if len(forecast) > 6 else ''}.\n"
        f"Model terpilih: {MODEL_NAMES.get(winner, winner)}, dengan galat rata-rata (MAE) {_fmt(scores[winner]['mae'])} "
        f"dan sMAPE {_fmt(scores[winner]['smape'])}% pada backtest {body['backtest_points']} {freq.label} terakhir. "
        f"Rentang 80% di tabel adalah perkiraan kasar dari galat backtest."
    )
    return table, chart, note, answer
