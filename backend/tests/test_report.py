from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pytest

from app.api.v1.chat import EXPORT_HINT, split_export_request
from app.core.config import Settings
from app.services.agent import profile as profiling
from app.services.agent.graph import build_agent, stream_agent
from app.services.data.ingest import write_duckdb
from app.services.report import build_pdf


@pytest.mark.parametrize(
    ("question", "cleaned", "wants_file"),
    [
        ("oke tolong buat ringkasan mengenai list linen tersebut dengan pdf", "oke tolong buat ringkasan mengenai list linen tersebut", True),
        ("total penjualan per produk, export ke excel", "total penjualan per produk", True),
        ("download laporan penjualan bulan ini", "laporan penjualan bulan ini", True),
        ("produk apa yang paling laku?", "produk apa yang paling laku?", False),
        ("pdf", "pdf", True),
    ],
)
def test_split_export_request(question, cleaned, wants_file):
    assert split_export_request(question) == (cleaned, wants_file)
    assert "Unduh PDF" in EXPORT_HINT


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("Ringkas isi dataset ini", True),
        ("oke tolong buat ringkasan mengenai list linen tersebut", True),
        ("kolom apa saja yang ada", True),
        ("gambaran umum data ini dong", True),
        ("tolong rangkum penjualan per kategori", False),
        ("ringkasan data penjualan per bulan", False),
        ("produk apa yang paling laku", False),
    ],
)
def test_overview_questions_are_recognised(question, expected):
    assert profiling.is_overview_question(question) is expected


class NoLLM:
    async def complete(self, *args, **kwargs):
        raise AssertionError("ringkasan dataset tidak boleh memanggil LLM")


async def test_overview_is_answered_from_the_profile_without_llm(tmp_path: Path):
    df = pd.DataFrame(
        {
            "tanggal": pd.to_datetime(["2026-09-01", "2026-09-02", "2026-09-03"]),
            "nama_linen": ["Sprei", "Selimut", None],
            "jumlah": [1250, 830, 2100],
        }
    )
    path = tmp_path / "linen.duckdb"
    write_duckdb(df, path)
    agent = build_agent(NoLLM(), Settings())
    initial = {"question": "ringkas isi dataset ini", "context": "", "db_path": str(path), "usage": []}
    events = [e async for e, _ in stream_agent(agent, initial)]
    assert [e["type"] for e in events] == ["intent", "table", "answer"]
    assert events[0]["value"] == "profile"
    assert [row[:2] for row in events[1]["rows"]] == [["tanggal", "tanggal"], ["nama_linen", "teks"], ["jumlah", "angka bulat"]]
    answer = events[2]["text"]
    assert "3 baris dan 3 kolom" in answer and "2026-09-01 sampai 2026-09-03" in answer and "nama_linen (33.33%)" in answer


def _pdf_text(pdf: bytes) -> str:
    import io

    from pypdf import PdfReader

    return "\n".join(page.extract_text() for page in PdfReader(io.BytesIO(pdf)).pages)


def test_pdf_report_contains_answer_table_and_chart():
    steps = [
        {"type": "intent", "value": "forecast"},
        {
            "type": "table",
            "columns": ["periode", "aktual", "prediksi"],
            "types": ["TIMESTAMP", "DOUBLE", "DOUBLE"],
            "rows": [["2026-08-01T00:00:00", 120.0, None], ["2026-09-01T00:00:00", 135.5, 135.5], ["2026-10-01T00:00:00", None, 141.25]],
            "truncated": False,
        },
        {"type": "stdout", "value": "Backtest 6 periode terakhir: lstm MAE 9,21."},
        {"type": "chart", "spec": {"kind": "line", "x": "periode", "y": ["aktual", "prediksi"], "dashed": ["prediksi"]}},
        {"type": "answer", "text": "Perkiraan total 141,25."},
    ]
    pdf = build_pdf("prediksi <total> & qty bulan depan", "Perkiraan total 141,25.", steps, "penjualan.csv", datetime(2026, 10, 11, tzinfo=timezone.utc))
    assert pdf.startswith(b"%PDF") and len(pdf) > 2000
    text = _pdf_text(pdf)
    for expected in ("Laporan Analisis Data", "penjualan.csv", "Prediksi", "141,25", "2026-09-01", "Backtest 6 periode"):
        assert expected in text, expected


def test_pdf_report_handles_bar_chart_and_plain_chat():
    bar = [
        {"type": "intent", "value": "sql"},
        {"type": "code", "language": "sql", "value": "SELECT produk, SUM(qty) FROM data GROUP BY 1", "attempt": 1},
        {"type": "table", "columns": ["produk", "total"], "types": ["VARCHAR", "BIGINT"], "rows": [["Kopi", 7], ["Teh", 5]], "truncated": False},
        {"type": "chart", "spec": {"kind": "bar", "x": "produk", "y": ["total"]}},
    ]
    now = datetime(2026, 10, 11, tzinfo=timezone.utc)
    assert "SELECT produk" in _pdf_text(build_pdf("apa yang laku", "Kopi.", bar, "a.csv", now))
    assert build_pdf("halo", "Halo!", [{"type": "intent", "value": "chat"}], "a.csv", now).startswith(b"%PDF")


def test_pdf_report_renders_markdown_instead_of_printing_the_marks():
    answer = "## Saran\n\n1. **Perjelas tenggat** pelaporan (hal. 3).\n- Gunakan `formulir` baku & <cek> ulang."
    pdf = build_pdf("apa saranmu", answer, [{"type": "intent", "value": "doc_analysis"}], "sop.pdf", datetime(2026, 10, 12, tzinfo=timezone.utc))
    text = _pdf_text(pdf)
    assert "Perjelas tenggat" in text and "formulir" in text and "<cek>" in text and "Telaah dokumen" in text
    assert "**" not in text and "##" not in text and "`" not in text
