"""Runs the full agent graph against a real DuckDB file with a scripted fake LLM."""

from pathlib import Path

import pandas as pd
import pytest

from app.core.config import Settings
from app.services.agent.charts import suggest_chart
from app.services.agent.graph import build_agent, extract_code, format_id, parse_intent, stream_agent
from app.services.data.ingest import write_duckdb
from app.services.llm.base import LLMResult


class FakeLLM:
    def __init__(self, replies: list[str]):
        self.replies = list(replies)
        self.prompts: list[str] = []

    async def complete(self, kind, system, messages, max_tokens=4096) -> LLMResult:
        self.prompts.append(messages[-1]["content"])
        return LLMResult(self.replies.pop(0), "fake", "fake-model", 10, 5)


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    df = pd.DataFrame(
        {
            "tanggal": pd.to_datetime(["2026-09-01", "2026-09-02", "2026-09-15"]),
            "produk": ["Kopi Susu", "Teh Manis", "Kopi Susu"],
            "qty": [3, 5, 4],
        }
    )
    path = tmp_path / "ds.duckdb"
    write_duckdb(df, path)
    return path


async def _run(llm: FakeLLM, db_path: Path, question: str) -> list[dict]:
    agent = build_agent(llm, Settings())
    initial = {"question": question, "context": "<skema>data</skema>", "db_path": str(db_path), "usage": []}
    return [event async for event, _ in stream_agent(agent, initial)]


async def test_sql_question_streams_code_table_chart_and_answer(db_path):
    llm = FakeLLM(
        [
            "sql",
            "```sql\nSELECT produk, SUM(qty) AS total_qty FROM data GROUP BY produk ORDER BY total_qty DESC\n```",
            "Kopi Susu paling laku dengan 7 cup.",
        ]
    )
    events = await _run(llm, db_path, "produk apa yang paling laku?")
    assert [e["type"] for e in events] == ["intent", "code", "table", "chart", "answer"]
    table = events[2]
    assert table["columns"] == ["produk", "total_qty"]
    assert table["rows"][0] == ["Kopi Susu", 7]
    assert events[3]["spec"] == {"kind": "bar", "x": "produk", "y": ["total_qty"]}
    assert events[4]["text"] == "Kopi Susu paling laku dengan 7 cup."


async def test_failed_sql_is_retried_with_the_error_message(db_path):
    llm = FakeLLM(
        [
            "sql",
            "```sql\nSELECT nama_salah FROM data\n```",
            "```sql\nSELECT COUNT(*) AS jumlah FROM data\n```",
            "Ada 3 transaksi.",
        ]
    )
    events = await _run(llm, db_path, "berapa transaksi?")
    assert [e["type"] for e in events] == ["intent", "code", "retry", "code", "table", "answer"]
    assert "nama_salah" in llm.prompts[2]  # the second attempt saw the failing query and its error


async def test_unsafe_sql_never_executes_and_gives_up_after_max_attempts(db_path):
    llm = FakeLLM(["sql"] + ["```sql\nDROP TABLE data\n```"] * 3)
    events = await _run(llm, db_path, "hapus semua data")
    assert [e["type"] for e in events].count("retry") == 3
    assert events[-1]["type"] == "answer" and "3 percobaan" in events[-1]["text"]


async def test_chat_intent_skips_code(db_path):
    llm = FakeLLM(["chat", "Halo! Tanyakan apa saja tentang datamu."])
    events = await _run(llm, db_path, "halo")
    assert [e["type"] for e in events] == ["intent", "answer"]


def test_helpers():
    assert extract_code("Berikut:\n```sql\nSELECT 1\n```") == "SELECT 1"
    assert extract_code("SELECT 2") == "SELECT 2"
    assert parse_intent("Python") == "python"
    assert parse_intent("tidak jelas") == "sql"
    assert format_id(40000.0) == "40.000"
    assert format_id(1234567.891) == "1.234.567,89"
    assert format_id("Kopi") == "Kopi"


def test_chart_picks_form_and_keeps_series_on_one_scale():
    trend = suggest_chart(["tanggal", "omzet"], ["TIMESTAMP", "DOUBLE"], [["2026-09-01", 10.0], ["2026-09-02", 12.0]])
    assert trend == {"kind": "line", "x": "tanggal", "y": ["omzet"]}
    mixed = suggest_chart(
        ["produk", "qty", "rupiah", "diskon"],
        ["VARCHAR", "BIGINT", "DOUBLE", "BIGINT"],
        [["Kopi", 5, 90000.0, 3], ["Teh", 5, 40000.0, 2]],
    )
    assert mixed == {"kind": "bar", "x": "produk", "y": ["qty", "diskon"]}
    assert suggest_chart(["produk"], ["VARCHAR"], [["a"], ["b"]]) is None
