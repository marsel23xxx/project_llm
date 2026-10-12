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
    extraction = documents.extract_pages(policy_pdf)
    pages = extraction.pages
    assert len(pages) == 4 and "Tarif pencucian" in pages[3] and extraction.ocr_pages == 0
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
    chunks = documents.chunk_pages(documents.extract_pages(policy_pdf).pages)
    hits = documents.Retriever(chunks).search(question)
    assert hits[-1][0].page == page  # strongest hit is last, next to the question in the prompt


def test_pdf_without_readable_text_is_rejected_with_a_clear_message(tmp_path):
    path = tmp_path / "scan.pdf"
    page = canvas.Canvas(str(path), pagesize=A4)
    page.rect(100, 100, 300, 400, fill=1)  # graphics but no text: nothing for the parser or OCR to read
    page.save()
    with pytest.raises(documents.DocumentError, match="scan"):
        documents.extract_pages(path)


def _text_image(path: Path, lines: list[str]) -> Path:
    """A clean 'photo' of printed text, drawn large enough for OCR."""
    from PIL import Image, ImageDraw, ImageFont

    image = Image.new("RGB", (1500, 140 + 90 * len(lines)), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=48)
    for i, line in enumerate(lines):
        draw.text((60, 60 + 90 * i), line, fill="black", font=font)
    image.save(path)
    return path


needs_ocr = pytest.mark.skipif(not documents.ocr.available(), reason="Tesseract tidak terpasang (diuji di container backend)")


@needs_ocr
def test_image_text_is_read_with_ocr(tmp_path):
    path = _text_image(tmp_path / "nota.png", ["NOTA PEMBAYARAN", "Total tagihan Rp 125.000", "Jatuh tempo 15 Oktober 2026"])
    extraction = documents.extract_image(path)
    text = extraction.pages[0]
    assert extraction.ocr_pages == 1
    assert "125.000" in text and "Oktober" in text and "PEMBAYARAN" in text.upper()


@needs_ocr
def test_scanned_pdf_pages_are_read_with_ocr(tmp_path):
    from reportlab.lib.utils import ImageReader

    image_path = _text_image(tmp_path / "hal.png", ["Surat Keterangan", "Nomor 042 tahun 2026", "Berlaku sampai Desember"])
    path = tmp_path / "scan.pdf"
    page = canvas.Canvas(str(path), pagesize=A4)
    page.drawImage(ImageReader(str(image_path)), 40, 500, width=500, height=140)  # a page that is only a picture
    page.save()
    extraction = documents.extract_pages(path)
    assert extraction.ocr_pages == 1 and "042" in extraction.pages[0] and "Desember" in extraction.pages[0]


def test_blank_image_is_rejected(tmp_path):
    from PIL import Image

    path = tmp_path / "kosong.png"
    Image.new("RGB", (800, 600), "white").save(path)
    with pytest.raises(documents.DocumentError):
        documents.extract_image(path)


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
    chunks = documents.chunk_pages(documents.extract_pages(pdf).pages)
    return {
        "question": question,
        "context": "",
        "db_path": "",
        "usage": [],
        "doc_chunks": [{"page": c.page, "position": c.position, "text": c.text, "source": c.source} for c in chunks],
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
    assert events[1]["value"] == "Ringkasan dibuat dari seluruh isi dokumen (4 halaman)."
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


class HostedLLM(RecordingLLM):
    has_strong_model = True


@pytest.mark.parametrize(
    ("question", "intent"),
    [
        ("apa saran kamu terhadap dokumen operasional ini??", "doc_analysis"),  # was misrouted to small talk
        ("apa kelemahan sop ini", "doc_analysis"),
        ("ringkas dokumen ini", "doc_summary"),
        ("berapa tarif linen infeksius?", "doc_qa"),
        ("halo, dokumen ini tentang apa", "doc_summary"),
        ("halo", "chat"),
        ("terima kasih banyak ya", "chat"),
    ],
)
async def test_document_session_routing(policy_pdf, question, intent):
    agent = build_agent(RecordingLLM("ok"), Settings())
    events = [e async for e, _ in stream_agent(agent, _state(policy_pdf, question))]
    assert events[0] == {"type": "intent", "value": intent}


def test_page_texts_removes_overlap_duplicates():
    words = [f"kata{i}" for i in range(400)]
    chunks = documents.chunk_pages([" ".join(words), "halaman dua pendek"])
    pages = documents.page_texts(chunks)
    assert pages[0] == ("", 1, " ".join(words)) and pages[1] == ("", 2, "halaman dua pendek")
    assert documents.word_count(chunks) == 403


async def test_hosted_model_reads_the_whole_document(policy_pdf):
    llm = HostedLLM("**Saran:** perjelas tenggat pelaporan (hal. 3).")
    agent = build_agent(llm, Settings())
    state = _state(policy_pdf, "apa saran kamu terhadap dokumen operasional ini??", file_note="File aktif")
    events = [e async for e, _ in stream_agent(agent, state)]

    assert [e["type"] for e in events] == ["intent", "stdout", "answer"]
    assert events[1]["value"] == "Jawaban dibuat dari seluruh isi dokumen (4 halaman)."
    request = llm.requests[0]
    assert all(f'<kutipan halaman="{n}">' in request for n in (1, 2, 3, 4))
    assert request.index("<kutipan") < request.index("Pertanyaan:")  # document first: cacheable prefix
    assert "Seluruh isi dokumen" in llm.systems[0]


async def test_hosted_model_falls_back_to_retrieval_for_very_long_documents(policy_pdf):
    llm = HostedLLM("Rp 7.000 (hal. 4).")
    agent = build_agent(llm, Settings(doc_full_context_words=50))
    events = [e async for e, _ in stream_agent(agent, _state(policy_pdf, "berapa tarif linen infeksius?"))]
    assert [e["type"] for e in events] == ["intent", "sources", "answer"]
    assert "Seluruh isi dokumen" not in llm.systems[0]


async def test_small_talk_in_a_document_session_knows_a_file_is_open(policy_pdf):
    llm = RecordingLLM("Halo! Silakan tanya tentang dokumennya.")
    agent = build_agent(llm, Settings())
    state = _state(policy_pdf, "halo", file_note='File aktif di sesi ini: dokumen PDF "kebijakan.pdf" (4 halaman).')
    [e async for e, _ in stream_agent(agent, state)]
    assert "kebijakan.pdf" in llm.systems[0]


def test_file_note_tells_the_assistant_what_is_attached():
    from types import SimpleNamespace as File

    from app.api.v1.chat import _file_note

    assert "Belum ada file" in _file_note([], None)
    pdf = File(original_filename="sop.pdf", row_count=5, kind="document")
    old = File(original_filename="lama.csv", row_count=10, kind="table")
    new = File(original_filename="baru.csv", row_count=20, kind="table")
    note = _file_note([pdf, old, new], new)
    assert '"sop.pdf" (5 halaman)' in note and "jangan meminta user mengunggah lagi" in note
    assert 'terakhir diunggah: "baru.csv"' in note


async def test_session_without_files_just_chats_with_history():
    llm = RecordingLLM("Tentu, silakan lampirkan filenya.")
    agent = build_agent(llm, Settings())
    state = {"question": "bisa bantu analisis data penjualan?", "context": "", "db_path": "", "usage": [],
             "history": "User: halo\nAsisten: Halo!", "file_note": "Belum ada file di sesi ini."}
    events = [e async for e, _ in stream_agent(agent, state)]
    assert [e["type"] for e in events] == ["intent", "answer"] and events[0]["value"] == "chat"
    assert "<riwayat>" in llm.requests[0] and "Belum ada file" in llm.systems[0]


async def test_two_documents_are_answered_together_and_cite_the_file(policy_pdf):
    first = documents.chunk_pages(documents.extract_pages(policy_pdf).pages, "kebijakan.pdf")
    second = documents.chunk_pages(["Nota pembayaran. Total tagihan bulan September adalah Rp 12.750.000."], "nota.png")
    merged = documents.renumber([first, second])
    assert [c.position for c in merged] == list(range(len(merged))) and merged[-1].source == "nota.png"

    llm = RecordingLLM("Rp 12.750.000 (nota.png, hal. 1).")
    agent = build_agent(llm, Settings())
    state = {"question": "berapa total tagihan september?", "context": "", "db_path": "", "usage": [],
             "doc_chunks": [{"page": c.page, "position": c.position, "text": c.text, "source": c.source} for c in merged]}
    events = [e async for e, _ in stream_agent(agent, state)]
    assert events[1]["items"][0] == {"file": "nota.png", "page": 1, "text": second[0].text}
    assert '<kutipan file="nota.png" halaman="1">' in llm.requests[0] and "tulis nama filenya" in llm.requests[0]


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("ringkas dokumen ini", "doc_summary"),          # names the document
        ("apa isi pasal 3 di kontrak", "doc_qa"),
        ("ringkas isi data ini", "profile"),              # about the table
    ],
)
async def test_session_with_table_and_document_routes_by_wording(policy_pdf, tmp_path, question, expected):
    import pandas as pd

    from app.services.data.ingest import write_duckdb

    db_path = tmp_path / "t.duckdb"
    write_duckdb(pd.DataFrame({"produk": ["Kopi", "Teh"], "qty": [3, 5]}), db_path)
    agent = build_agent(RecordingLLM("ok"), Settings())
    state = {**_state(policy_pdf, question), "db_path": str(db_path)}
    events = [e async for e, _ in stream_agent(agent, state)]
    assert events[0] == {"type": "intent", "value": expected}
