import sys
from pathlib import Path

import httpx
import numpy as np
import pandas as pd
import pytest

from app.core.config import Settings
from app.ml.models import MLModels, Prediction
from app.services.agent import forecast as forecasting
from app.services.agent.graph import build_agent, stream_agent
from app.services.data.ingest import write_duckdb

# The forecast service is its own container; its pure-numpy logic is importable here (TensorFlow loads lazily).
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "forecast"))
from forecasting import ForecastError, run_forecast, seasonal_naive, smape  # noqa: E402


@pytest.fixture
def sales_db(tmp_path: Path) -> Path:
    days = pd.date_range("2024-01-01", "2026-09-10", freq="D")  # September 2026 is incomplete
    df = pd.DataFrame({"tanggal": days, "total": 100.0, "qty": 2, "kota": "Bandung"})
    df = df[~((df["tanggal"].dt.year == 2025) & (df["tanggal"].dt.month == 3))]  # a missing month
    path = tmp_path / "sales.duckdb"
    write_duckdb(df, path)
    return path


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("prediksi penjualan 3 bulan ke depan", ("bulan", 3)),
        ("perkiraan omzet bulan depan", ("bulan", 1)),
        ("ramalan qty enam minggu mendatang", ("minggu", 6)),
        ("prediksi penjualan tahun depan", ("bulan", 12)),
        ("forecast 2 kuartal ke depan", ("bulan", 6)),
        ("brp perkiraan penjualan 14 hr ke depan", ("hari", 14)),
        ("prediksi penjualan harian", ("hari", None)),
        ("prediksi penjualan", (None, None)),
    ],
)
def test_parse_horizon(question, expected):
    assert forecasting.parse_horizon(question) == expected


def test_pick_columns_prefers_named_then_hinted(sales_db):
    assert forecasting.pick_columns(sales_db, "prediksi qty bulan depan") == ("tanggal", "qty")
    assert forecasting.pick_columns(sales_db, "prediksi bulan depan") == ("tanggal", "total")


def test_series_fills_gaps_and_drops_partial_last_period(sales_db):
    series, dropped = forecasting.load_series(sales_db, "tanggal", "total", forecasting.FREQUENCIES["bulan"], "sum")
    assert dropped and series.index[-1] == pd.Timestamp("2026-08-01")
    assert series.loc["2025-03-01"] == 0.0  # the missing month is an explicit zero, not a skipped step
    assert series.loc["2025-01-01"] == 3100.0
    assert len(series) == 32


def test_baselines_and_backtest_pick_the_seasonal_pattern():
    assert seasonal_naive(np.array([1.0, 2, 3, 4, 5, 6]), 4, 3).tolist() == [4, 5, 6, 4]
    assert smape(np.array([0.0, 10]), np.array([0.0, 10])) == 0.0

    weekly_pattern = [10, 12, 15, 14, 20, 30, 25] * 8
    result = run_forecast(weekly_pattern, horizon=7, season=7, use_lstm=False)
    assert result["model"] == "seasonal_naive"
    assert result["forecast"] == [10, 12, 15, 14, 20, 30, 25]
    assert result["scores"]["seasonal_naive"]["mae"] == 0
    assert result["lower"] == result["forecast"]  # zero backtest error -> zero-width interval

    with pytest.raises(ForecastError):
        run_forecast([1, 2, 3], horizon=2, season=1, use_lstm=False)


def _fake_service(monkeypatch, handler):
    real = httpx.AsyncClient
    monkeypatch.setattr(
        forecasting.httpx, "AsyncClient", lambda **kw: real(transport=httpx.MockTransport(handler), base_url="http://forecast")
    )


class FixedIntent:
    name, threshold = "fake-intent", 0.6

    def predict(self, text):
        return Prediction("forecast", 0.99)


class NoLLM:
    async def complete(self, *args, **kwargs):
        raise AssertionError("prediksi tidak boleh memanggil LLM")


async def test_forecast_question_end_to_end_without_llm(sales_db, monkeypatch):
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        import json

        body = json.loads(request.content)
        seen.update(body)
        return httpx.Response(200, json=run_forecast(body["values"], body["horizon"], body["season"], use_lstm=False))

    _fake_service(monkeypatch, handler)
    agent = build_agent(NoLLM(), Settings(), MLModels(FixedIntent(), None))
    initial = {"question": "prediksi total 3 bulan ke depan", "context": "", "db_path": str(sales_db), "usage": []}
    events = [e async for e, _ in stream_agent(agent, initial)]

    assert [e["type"] for e in events] == ["intent", "table", "stdout", "chart", "answer"]
    assert seen["horizon"] == 3 and seen["season"] == 12 and len(seen["values"]) == 32
    table = events[1]
    future = [r for r in table["rows"] if r[1] is None]
    assert [r[0][:10] for r in future] == ["2026-09-01", "2026-10-01", "2026-11-01"]
    assert all(r[3] <= r[2] <= r[4] for r in future)
    assert events[3]["spec"]["dashed"] == ["prediksi"]
    assert "3 bulan ke depan" in events[4]["text"] and "Sep 2026" in events[4]["text"]


async def test_forecast_explains_when_history_is_too_short(tmp_path, monkeypatch):
    path = tmp_path / "short.duckdb"
    write_duckdb(pd.DataFrame({"tanggal": pd.date_range("2026-01-01", periods=5, freq="MS"), "total": [1.0, 2, 3, 4, 5]}), path)
    _fake_service(monkeypatch, lambda request: httpx.Response(500))
    with pytest.raises(forecasting.ForecastUnavailable, match="minimal 12"):
        await forecasting.run(path, "prediksi total 3 bulan ke depan", "http://forecast")
