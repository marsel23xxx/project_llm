from pathlib import Path

import pytest
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.pdfgen import canvas
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate

from app.core.config import Settings
from app.services import documents
from app.services.agent.graph import build_agent, stream_agent
from app.services.llm.base import LLMResult

PAGES = [
    "Kebijakan Layanan Laundry Rumah Sakit. Dokumen ini mengatur penerimaan, pencucian, dan pengiriman linen "
    "untuk seluruh ruangan rawat inap. Berlaku mulai 1 September 2026.",
    "Jadwal pengambilan linen kotor dilakukan dua kali sehari, pukul 07.00 dan pukul 15.00. Petugas wajib "
    "mencatat jumlah linen per ruangan pada formulir serah terima sebelum linen dibawa ke unit laundry.",
    "Linen yang rusak atau hilang harus dilaporkan paling lambat 2 x 24 jam setelah serah terima. Penggantian "
    "linen rusak ditanggung unit laundry apabila kerusakan terjadi saat proses pencucian.",
    "Tarif pencucian adalah Rp 4.500 per kilogram untuk linen biasa dan Rp 7.000 per kilogram untuk linen "
    "infeksius. Tagihan dibuat setiap akhir bulan dan dibayar paling lambat tanggal 10 bulan berikutnya.",
]


@pytest.fixture
def policy_pdf(tmp_path: Path) -> Path:
    path = tmp_path / "kebijakan.pdf"
    style = getSampleStyleSheet()["Normal"]
    story = []
    for text in PAGES:
        story += [Paragraph(text, style), PageBreak()]
    SimpleDocTemplate(str(path), pagesize=A4).build(story)
    return path


def test_extract_and_chunk_keep_page_numbers(policy_pdf):
    pages = documents.extract_pages(policy_pdf)
    assert len(pages) == 4 and "Tarif pencucian" in pages[3]
    chunks = documents.chunk_pages(pages)
    assert [c.page for c in chunks] == [1, 2, 3, 4]
    assert [c.position for c in chunks] == [0, 1, 2, 3]


def test_long_page_is_split_with_overlap():
    words = [f"kata{i}" for i in range(400)]
    chunks = documents.chunk_pages([" ".join(words)])
    assert len(chunks) == 5 and all(c.page == 1 for c in chunks)
    assert chunks[0].text.split()[-documents.OVERLAP_WORDS :] == chunks[1].text.split()[: documents.OVERLAP_WORDS]
    assert chunks[-1].text.split()[-1] == "kata399"


@pytest.mark.parametrize(
    ("question", "page"),
    [
        ("berapa tarif pencucian linen infeksius?", 4),
        ("jam berapa linen kotor diambil", 2),
        ("kalau linen hilang lapornya kapan", 3),
        ("brp biaya cuci per kg", 4),  # slang and different wording than the document
    ],
)
def test_retriever_ranks_the_right_page_first(policy_pdf, question, page):
    chunks = documents.chunk_pages(documents.extract_pages(policy_pdf))
    hits = documents.Retriever(chunks).search(question)
    assert hits[-1][0].page == page  # strongest hit is last, next to the question in the prompt


def test_scanned_pdf_is_rejected_with_a_clear_message(tmp_path):
    path = tmp_path / "scan.pdf"
    page = canvas.Canvas(str(path), pagesize=A4)
    page.rect(100, 100, 300, 400, fill=1)  # a page with graphics but no text, like a scan
    page.save()
    with pytest.raises(documents.DocumentError, match="scan"):
        documents.extract_pages(path)


def test_unreadable_file_is_rejected(tmp_path):
    path = tmp_path / "rusak.pdf"
    path.write_bytes(b"ini bukan pdf")
    with pytest.raises(documents.DocumentError, match="tidak bisa dibaca"):
        documents.extract_pages(path)


class RecordingLLM:
    def __init__(self, reply: str):
        self.reply, self.requests, self.systems = reply, [], []

    async def complete(self, kind, system, messages, max_tokens=4096):
        self.systems.append(system)
        self.requests.append(messages[-1]["content"])
        return LLMResult(self.reply, "fake", "fake", 1, 1)


def _state(pdf: Path, question: str, **extra) -> dict:
    chunks = documents.chunk_pages(documents.extract_pages(pdf))
    return {
        "question": question,
        "context": "",
        "db_path": "",
        "usage": [],
        "doc_chunks": [{"page": c.page, "position": c.position, "text": c.text} for c in chunks],
        **extra,
    }


async def test_document_question_cites_retrieved_pages(policy_pdf):
    llm = RecordingLLM("Tarif linen infeksius Rp 7.000 per kilogram (hal. 4).")
    agent = build_agent(llm, Settings())
    events = [e async for e, _ in stream_agent(agent, _state(policy_pdf, "berapa tarif linen infeksius?"))]

    assert [e["type"] for e in events] == ["intent", "sources", "answer"]
    assert events[0]["value"] == "doc_qa"
    assert 4 in [s["page"] for s in events[1]["items"]]
    # The model was given the passage itself, marked with its page, and told to stay inside it.
    assert '<kutipan halaman="4">' in llm.requests[0] and "Rp 7.000" in llm.requests[0]
    assert "HANYA berdasarkan" in llm.systems[0]
    assert events[1]["items"][0]["page"] == 4  # sources are listed strongest first


async def test_short_follow_up_reuses_the_previous_question_for_retrieval(policy_pdf):
    llm = RecordingLLM("Dilaporkan paling lambat 2 x 24 jam (hal. 3).")
    agent = build_agent(llm, Settings())
    state = _state(policy_pdf, "jelaskan lagi", previous_question="kalau linen hilang lapornya kapan")
    events = [e async for e, _ in stream_agent(agent, state)]
    assert 3 in [s["page"] for s in events[1]["items"]]


async def test_short_document_is_summarised_in_full(policy_pdf):
    llm = RecordingLLM("Dokumen ini adalah kebijakan layanan laundry.")
    agent = build_agent(llm, Settings())
    events = [e async for e, _ in stream_agent(agent, _state(policy_pdf, "ringkas dokumen ini"))]
    assert [e["type"] for e in events] == ["intent", "stdout", "answer"]
    assert events[0]["value"] == "doc_summary"
    assert events[1]["value"] == "Ringkasan dibuat dari seluruh dokumen (4 halaman)."
    assert '<kutipan halaman="4">' in llm.requests[0]


def test_long_document_summary_samples_across_all_pages():
    chunks = documents.chunk_pages([" ".join(f"h{p}k{i}" for i in range(300)) for p in range(1, 41)])
    picked, sampled = documents.summary_excerpts(chunks)
    assert sampled and len(picked) == documents.SUMMARY_EXCERPTS
    assert picked[0].page == 1 and picked[-1].page == 40  # start and end of the document are both covered
    assert all(len(c.text.split()) <= documents.SUMMARY_EXCERPT_WORDS for c in picked)
    assert sum(len(c.text.split()) for c in picked) <= 900


async def test_question_with_no_matching_text_does_not_call_the_llm(policy_pdf):
    llm = RecordingLLM("tidak dipakai")
    agent = build_agent(llm, Settings())
    events = [e async for e, _ in stream_agent(agent, _state(policy_pdf, "zzzz qqqq xxxx wwww"))]
    assert events[-1]["text"].startswith("Saya tidak menemukan") and llm.requests == []
