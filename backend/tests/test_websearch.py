import json
from datetime import date

import httpx

from app.core.config import Settings
from app.services import exports
from app.services.agent import sourcing
from app.services.agent.graph import build_agent, stream_agent
from app.services.llm.base import LLMResult
from app.services.websearch import SearchError, WebSearch, WebSource, search_from

LONG = "Kebakaran hutan dan lahan kembali meluas pada musim kemarau tahun ini di sejumlah provinsi. " * 3


def _transport(handler):
    return httpx.MockTransport(handler)


async def test_search_merges_queries_and_skips_duplicates_and_empty_pages():
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        seen.append((request.headers["authorization"], body["query"], body["max_results"]))
        if body["query"] == "gagal":
            return httpx.Response(500, text="boom")
        results = [
            {"title": f"Judul {body['query']}", "url": f"https://www.contoh.id/{body['query']}", "content": LONG, "raw_content": LONG * 40},
            {"title": "Sama", "url": "https://contoh.id/sama", "content": LONG},
            {"title": "Kosong", "url": "https://contoh.id/kosong", "content": "pendek"},
            {"title": "Aneh", "url": "javascript:alert(1)", "content": LONG},
        ]
        return httpx.Response(200, json={"results": results})

    search = WebSearch("tvly-rahasia", max_sources=8, transport=_transport(handler))
    sources = await search.search(["satu", "gagal", "dua"])

    assert all(auth == "Bearer tvly-rahasia" for auth, _, _ in seen) and len(seen) == 3
    # One failed query does not lose the others; first hits of each query come first.
    assert [s.url for s in sources] == ["https://www.contoh.id/satu", "https://www.contoh.id/dua", "https://contoh.id/sama"]
    assert sources[0].site == "contoh.id" and len(sources[0].text.split()) <= 600


async def test_search_reports_failure_when_every_query_fails():
    search = WebSearch("k", transport=_transport(lambda request: httpx.Response(401, text="invalid key")))
    try:
        await search.search(["a"])
    except SearchError as exc:
        assert "401" in str(exc)
    else:
        raise AssertionError("harus gagal")


def test_search_is_off_without_a_key_or_with_the_local_model():
    assert search_from(Settings(llm_mode="api", tavily_api_key="")) is None
    assert search_from(Settings(llm_mode="local", tavily_api_key="tvly-x")) is None
    assert search_from(Settings(llm_mode="api", tavily_api_key="tvly-x", web_search=False)) is None
    assert isinstance(search_from(Settings(llm_mode="api", tavily_api_key="tvly-x")), WebSearch)


def test_queries_are_read_one_per_line():
    assert sourcing.parse_queries('1. karhutla 2026 terbaru\n- "data luas karhutla KLHK"\n\nkarhutla 2026 terbaru\n', "x") == [
        "karhutla 2026 terbaru",
        "data luas karhutla KLHK",
    ]
    assert sourcing.parse_queries("", "buatkan makalah karhutla") == ["buatkan makalah karhutla"]


def test_numbers_are_matched_in_both_digit_groupings():
    text = "Luas terbakar 1.161.192 hektare, naik 12,5 persen; 2,345 hotspots and 38 provinsi."
    for value in (1161192, 12.5, 2345, 38):
        assert sourcing.number_in_text(value, text), value
    for value in (161, 12, 5, 1161, 2.4):  # parts of a longer number do not count
        assert not sourcing.number_in_text(value, text), value


def test_a_rounded_source_number_still_counts_as_the_source():
    # The chart of the first real test: drawn from a news table, rounded by the model, then wrongly called an estimate.
    text = "Kalbar: 38.310,89 ha; Riau: 16.249,24; Kalsel: 8.019,62; Kalteng: 7.634,46; Sumsel: 1.926,23; Jambi: ± 658,29. Naik 12,5 persen."
    for value in (38311, 16249, 8020, 7634, 1926, 658.29, 658.3, 13):
        assert sourcing.number_in_text(value, text), value
    for value in (38310, 38312, 8019, 12, 660):  # truncated or simply different: not the source's figure
        assert not sourcing.number_in_text(value, text), value
    chart = _chart("web", [38311, 16249])
    assert sourcing.verify_chart_sources(chart, text)[1] == 0


def _chart(source: str, values: list) -> str:
    return '```chart\n' + json.dumps({"type": "bar", "source": source, "labels": ["2022", "2023"], "values": values}) + '\n```'


def test_a_chart_only_keeps_its_source_when_its_numbers_are_in_the_source():
    text = "Luas karhutla 204.894 hektare pada 2022 dan 1.161.192 hektare pada 2023."
    kept, changed = sourcing.verify_chart_sources("Isi.\n\n" + _chart("web", [204894, 1161192]), text)
    assert changed == 0 and exports.parse_markdown(kept)[1].chart.origin == "web"

    made_up, changed = sourcing.verify_chart_sources("Isi.\n\n" + _chart("web", [204894, 900000]), text)
    assert changed == 1 and exports.parse_markdown(made_up)[1].chart.origin == "estimate"

    estimate, changed = sourcing.verify_chart_sources(_chart("perkiraan", [1, 2]), text)
    assert changed == 0 and "perkiraan" in estimate


SOURCES = [
    WebSource("Rekap karhutla 2023", "https://www.sipongi.contoh.id/rekap", "Luas karhutla 2023 tercatat 1.161.192 hektare, dari 204.894 hektare pada 2022."),
    WebSource("Tidak dipakai", "https://berita.contoh.id/a", "Isi lain yang tidak dirujuk sama sekali oleh penulis dokumen ini."),
    WebSource("Dampak asap", "https://kesehatan.contoh.id/ispa", "Kasus ISPA naik selama musim asap menurut dinas kesehatan setempat."),
]


def test_references_are_built_from_the_cited_sources_only():
    written = (
        "# Karhutla\n\nKasus ISPA naik [3]. Luas terbakar 1.161.192 hektare [1, 3]. Lihat juga [2023] dan [9].\n\n"
        + _chart("web", [204894, 1161192])
        + "\n\n## Daftar Pustaka\n\n1. Buku karangan model. Penerbit Fiktif, 2019.\n"
    )
    markdown, cited = sourcing.cite_sources(written, SOURCES, date(2026, 10, 13))

    assert [s.title for s in cited] == ["Dampak asap", "Rekap karhutla 2023"]  # in reading order, unused source left out
    assert "Kasus ISPA naik [1]. Luas terbakar 1.161.192 hektare [2, 1]." in markdown
    assert "[2023]" in markdown and "[9]" in markdown and "[204894, 1161192]" in markdown  # not citations: untouched
    assert "Fiktif" not in markdown and markdown.count("## Daftar Pustaka") == 1
    assert "1. Dampak asap. kesehatan.contoh.id. https://kesehatan.contoh.id/ispa (diakses 13 Oktober 2026)." in markdown
    assert "Tidak dipakai" not in markdown

    untouched, none = sourcing.cite_sources("# Judul\n\nTanpa rujukan.", SOURCES, date(2026, 10, 13))
    assert none == [] and "Daftar Pustaka" not in untouched


class SearchingWriter:
    has_strong_model = True

    def __init__(self, paper: str):
        self.paper, self.calls = paper, []

    async def complete(self, kind, system, messages, max_tokens=4096):
        self.calls.append((system, messages[-1]["content"]))
        if "menyiapkan pencarian web" in system:
            return LLMResult("karhutla 2026 terbaru\ndata luas karhutla KLHK\ndampak karhutla", "fake", "fake", 5, 20)
        return LLMResult(self.paper, "fake", "fake", 10, 500)


class FakeSearch:
    name = "fake"

    def __init__(self, sources=None, error: bool = False):
        self.sources, self.error, self.queries = sources or [], error, None

    async def search(self, queries):
        self.queries = queries
        if self.error:
            raise SearchError("mati")
        return self.sources


QUESTION = "buatkan saya makalah untuk masalah kebakaran yang sedang viral di kahutla dan berikan data dan grafik"


async def _run(llm, search, **extra):
    agent = build_agent(llm, Settings(), search=search)
    state = {"question": QUESTION, "context": "", "db_path": "", "usage": [], **extra}
    pairs = [pair async for pair in stream_agent(agent, state)]
    return [event for event, _ in pairs], pairs[-1][1]


async def test_a_paper_is_written_from_web_sources_and_says_so():
    paper = "# Karhutla 2023\n\nLuas terbakar 1.161.192 hektare [1]. Kasus ISPA naik [3].\n\n" + _chart("web", [204894, 1161192]) + "\n\n" + _chart("web", [7, 99])
    llm, search = SearchingWriter(paper), FakeSearch(SOURCES)
    events, usage = await _run(llm, search)

    assert events[0] == {"type": "intent", "value": "compose"}
    assert search.queries == ["karhutla 2026 terbaru", "data luas karhutla KLHK", "dampak karhutla"]
    system, request = llm.calls[1]
    assert '<sumber nomor="1" judul="Rekap karhutla 2023" situs="sipongi.contoh.id">' in request and "1.161.192 hektare" in request
    assert "kurung siku" in system and "tidak punya akses ke berita" not in system

    document, answer = events[1]["markdown"], events[2]["text"]
    charts = [b.chart.origin for b in exports.parse_markdown(document) if b.kind == "chart"]
    assert charts == ["web", "estimate"]  # the second chart's numbers are not in any source
    assert "## Daftar Pustaka" in document and "https://www.sipongi.contoh.id/rekap" in document and "berita.contoh.id" not in document
    assert "Ditulis dari 2 sumber web" in answer and "[Rekap karhutla 2023](https://www.sipongi.contoh.id/rekap)" in answer
    assert "1 grafik di dalamnya memakai angka perkiraan" in answer
    assert {"step": "search", "provider": "fake", "queries": 3, "results": 3} in usage
    exports.render_pdf(document), exports.render_docx(document)  # the cited document still exports


async def test_the_chat_says_so_when_the_search_fails_or_nothing_is_cited():
    events, _ = await _run(SearchingWriter("# Karhutla\n\nIsi umum."), FakeSearch(error=True))
    assert "pencarian web sedang gagal dihubungi" in events[2]["text"] and "Daftar Pustaka" not in events[1]["markdown"]

    events, _ = await _run(SearchingWriter("# Karhutla\n\nIsi umum."), FakeSearch([]))
    assert "tidak menemukan halaman yang relevan" in events[2]["text"]

    events, _ = await _run(SearchingWriter("# Karhutla\n\nIsi umum tanpa rujukan."), FakeSearch(SOURCES))
    assert "tidak ada yang dirujuk" in events[2]["text"] and "Daftar Pustaka" not in events[1]["markdown"]


async def test_attached_documents_are_enough_unless_current_data_is_asked_for():
    chunks = [{"page": 1, "position": 0, "text": "Luas karhutla 2023 tercatat 1,16 juta hektare.", "source": "laporan.pdf"}]
    search = FakeSearch(SOURCES)
    agent = build_agent(SearchingWriter("# Judul\n\nIsi."), Settings(), search=search)
    state = {"question": "buatkan makalah dari laporan ini", "context": "", "db_path": "", "usage": [], "doc_chunks": chunks}
    [e async for e in stream_agent(agent, state)]
    assert search.queries is None

    await _run(SearchingWriter("# Judul\n\nIsi."), search, doc_chunks=chunks)  # "yang sedang viral": search as well
    assert search.queries is not None
