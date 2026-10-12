"""The data-analyst agent as an explicit LangGraph state machine.

    route ─┬─ sql ──────► generate_sql ──► execute_sql ─┬─► summarize
           │                   ▲                         │
           │                   └──── error, retry ◄──────┤
           ├─ python ───► generate_python ─► run_python ──┤ (same retry loop)
           │                                             └─► fail (after max attempts)
           ├─ sentiment ► analyze_sentiment ─────────────────► summarize (or explain why not)
           ├─ forecast ─► run_forecast (LSTM vs baselines, answer built from the numbers)
           ├─ profile ──► describe_dataset (column profile, no LLM; chosen by a rule)
           ├─ compose ──► write_document (a full paper, proposal or letter; exported as DOCX/PDF)
           └─ chat ─────► answer_chat

A PDF session skips the table paths: route ─► answer_document (retrieve passages, answer with page citations).

`route` asks the trained intent classifier first and only calls the LLM when it is unsure.
"""

import asyncio
import json
import operator
import re
from collections.abc import AsyncIterator
from dataclasses import asdict
from pathlib import Path
from typing import Annotated, Literal, TypedDict

from langgraph.graph import END, START, StateGraph

from app.core.config import Settings
from app.ml.models import MLModels
from app.ml.text import normalize
from app.services.agent import prompts
from app.services.agent.charts import suggest_chart
from app.services.agent import forecast as forecasting
from app.services.agent import profile as profiling
from app.services import documents, exports
from app.services.agent.executor import ExecutionError, QueryResult, run_sql
from app.services.agent.guardrail import GuardrailError, validate_sql
from app.services.agent.sandbox_client import run_python
from app.services.agent.sentiment import SentimentUnavailable
from app.services.agent.sentiment import analyze as analyze_text_sentiment
from app.services.llm.base import LLMError, LLMResult
from app.services.llm.router import LLMRouter

Intent = Literal[
    "sql", "python", "sentiment", "forecast", "profile", "doc_qa", "doc_summary", "doc_analysis", "compose", "chat"
]
SUMMARY_ROWS = 20

DOC_SUMMARY_QUESTION = re.compile(
    r"\b(ringkas|ringkasan|rangkum|rangkuman|intisari|inti|garis besar|poin penting|poin utama|kesimpulan)\b"
    r"|\b(dokumen|file|pdf) (ini|nya)? ?(tentang|membahas|berisi|isinya) apa\b|\btentang apa\b"
)
DOC_ANALYSIS_QUESTION = re.compile(
    r"\b(saran|sarankan|rekomendasi|rekomendasikan|evaluasi|kritik|kelemahan|kekurangan|kelebihan|perbaikan|perbaiki"
    r"|review|tinjau|nilai|penilaian|pendapat|menurutmu|analisis|analisa|risiko|masukan)\b"
)
# A request to write something, as opposed to a question about the data.
_WRITE_VERB = r"(buat|buatkan|bikin|bikinkan|tulis|tuliskan|susun|susunkan|rancang|draf|draft|drafkan|ketik|ketikkan|kembangkan)"
_ALWAYS_PROSE = (
    r"(makalah|proposal|surat|artikel|esai|essay|paper|karya tulis|karya ilmiah|notulen|pidato|naskah|modul|panduan"
    r"|cv|resume|kontrak|cerpen|puisi|press release|siaran pers|memo|pengumuman|rpp|silabus|skripsi)"
)
# "laporan" and "dokumen" are prose only when no table is attached; with a table they usually mean an analysis.
_PROSE_WITHOUT_TABLE = r"(laporan|dokumen|ringkasan eksekutif|rencana|materi|tulisan|bab)"
_CHART_WORDS = re.compile(r"\b(grafik|chart|diagram|plot|tabel|hitung|jumlahkan|total)\b")


def is_writing_request(text: str, has_table: bool) -> bool:
    if _CHART_WORDS.search(text):
        return False
    nouns = _ALWAYS_PROSE if has_table else f"({_ALWAYS_PROSE}|{_PROSE_WITHOUT_TABLE})"
    return re.search(rf"\b{_WRITE_VERB}\b.*\b{nouns}\b", text) is not None


DOCUMENT_WORDING = re.compile(
    r"\b(dokumen|pdf|gambar|foto|scan|halaman|surat|kontrak|perjanjian|sop|makalah|artikel|laporan|teks|tulisan|pasal|bab|lampiran)\b"
)
# Whole-message matches only: "halo, apa isi dokumen ini" is a question, not a greeting.
SMALL_TALK = re.compile(
    r"(halo|hai|hi|hello|pagi|siang|sore|malam|selamat (pagi|siang|sore|malam)|terima kasih( banyak)?|oke|ok|sip|mantap|siap|tes|test)"
    r"( (ya|juga|sudah|siap|paham))?"
)
_CODE_BLOCK = re.compile(r"```(?:sql|python|py)?\s*\n?(.*?)```", re.DOTALL | re.IGNORECASE)


class AgentState(TypedDict, total=False):
    question: str
    context: str
    db_path: str
    intent: Intent
    code: str
    attempts: int
    error: str | None
    result: dict | None
    chart: dict | None
    samples: list[dict]
    # Document sessions (PDF): the stored chunks, and what was retrieved for this answer.
    doc_chunks: list[dict] | None
    history: str
    previous_question: str
    file_note: str
    sources: list[dict]
    document: dict
    answer: str
    usage: Annotated[list[dict], operator.add]


def strip_fence(text: str) -> str:
    """Models sometimes wrap a whole document in ```markdown ... ```; keep only the document."""
    stripped = text.strip()
    match = re.fullmatch(r"```(?:markdown|md)?\s*\n(.*)\n```", stripped, re.DOTALL)
    # Only an outer wrapper is removed: a document that merely ends with a chart block is left alone.
    if match and not match.group(1).lstrip().startswith("{"):
        return match.group(1).strip()
    return stripped


def extract_code(text: str) -> str:
    match = _CODE_BLOCK.search(text)
    return (match.group(1) if match else text).strip()


def parse_intent(text: str) -> Intent:
    words = re.findall(r"[a-z]+", text.lower())
    for intent in ("python", "sentiment", "forecast", "chat", "sql"):
        if intent in words[:3]:
            return intent
    return "sql"


def _usage(result: LLMResult, step: str) -> list[dict]:
    return [{"step": step, **{k: v for k, v in asdict(result).items() if k != "text"}}]


def format_id(value):
    """Pre-format numbers Indonesian-style so a small model copies them instead of re-formatting."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return value
    if isinstance(value, float) and not value.is_integer():
        text = f"{value:,.2f}"
    else:
        text = f"{int(value):,}"
    return text.replace(",", "_").replace(".", ",").replace("_", ".")


def _result_for_prompt(result: dict) -> str:
    if result.get("value") is not None:
        body = json.dumps(result["value"], ensure_ascii=False, default=str)[:4000]
    else:
        rows = [[format_id(v) for v in row] for row in result["rows"][:SUMMARY_ROWS]]
        body = json.dumps({"kolom": result["columns"], "baris": rows}, ensure_ascii=False, default=str)
        if result["truncated"] or len(result["rows"]) > SUMMARY_ROWS:
            body += f"\n(hanya {len(rows)} dari {len(result['rows'])}{'+' if result['truncated'] else ''} baris ditampilkan)"
    if result.get("stdout"):
        body += f"\nOutput print:\n{result['stdout'][-2000:]}"
    return body


def build_agent(llm: LLMRouter, settings: Settings, ml: MLModels | None = None):
    max_attempts = settings.agent_max_attempts
    intent_model = ml.intent if ml else None
    sentiment_model = ml.sentiment if ml else None

    async def route(state: AgentState) -> dict:
        text = normalize(state["question"])
        has_docs, has_table = bool(state.get("doc_chunks")), bool(state.get("db_path"))
        # No file yet, or a plain greeting: just talk. (The intent classifier is not used for this:
        # it was trained on table questions and labels "apa saranmu untuk dokumen ini" as small talk.)
        if SMALL_TALK.fullmatch(text):
            return {"intent": "chat", "attempts": 0, "usage": []}
        if is_writing_request(text, has_table):
            return {"intent": "compose", "attempts": 0, "usage": [{"step": "route", "provider": "rule", "model": "compose"}]}
        if not (has_docs or has_table):
            return {"intent": "chat", "attempts": 0, "usage": []}
        # With both a table and documents attached, wording decides which one the question is about.
        if has_docs and (not has_table or DOCUMENT_WORDING.search(text)):
            if DOC_SUMMARY_QUESTION.search(text):
                intent = "doc_summary"
            elif DOC_ANALYSIS_QUESTION.search(text):
                intent = "doc_analysis"
            else:
                intent = "doc_qa"
            return {"intent": intent, "attempts": 0, "usage": [{"step": "route", "provider": "rule", "model": "document"}]}
        if profiling.is_overview_question(state["question"]):
            return {"intent": "profile", "attempts": 0, "usage": [{"step": "route", "provider": "rule", "model": "overview"}]}
        kind = "simple"
        if intent_model:
            pred = intent_model.predict(state["question"])
            confident = pred.confidence >= (settings.intent_min_confidence or intent_model.threshold)
            # Only a stronger router is worth a call: the local 3B model routes worse than the
            # classifier (ml/reports/router_benchmark.json), so without Claude the classifier decides.
            if confident or not getattr(llm, "has_strong_model", False):
                usage = {"step": "route", "provider": "ml", "model": intent_model.name, "confidence": round(pred.confidence, 3)}
                return {"intent": pred.label, "attempts": 0, "usage": [usage]}
            kind = "complex"
        try:
            out = await llm.complete(kind, prompts.ROUTER_SYSTEM, [{"role": "user", "content": state["question"]}], 16)
        except LLMError:
            return {"intent": "sql", "attempts": 0, "usage": []}
        return {"intent": parse_intent(out.text), "attempts": 0, "usage": _usage(out, "route")}

    def _generator(system: str, step: str):
        async def generate(state: AgentState) -> dict:
            request = prompts.code_request(state["context"], state["question"], state.get("code"), state.get("error"))
            out = await llm.complete("complex", system, [{"role": "user", "content": request}], 1500)
            return {"code": extract_code(out.text), "attempts": state["attempts"] + 1, "usage": _usage(out, step)}

        return generate

    def _finish(result: QueryResult) -> dict:
        chart = suggest_chart(result.columns, result.types, result.rows) if result.columns else None
        payload = {**result.to_event(), "stdout": result.stdout, "value": result.value}
        return {"result": payload, "chart": chart, "error": None}

    async def execute_sql(state: AgentState) -> dict:
        try:
            sql = validate_sql(state["code"])
            result = await asyncio.to_thread(
                run_sql, Path(state["db_path"]), sql, settings.sql_timeout_seconds, settings.max_result_rows
            )
        except (GuardrailError, ExecutionError) as exc:
            return {"error": str(exc)}
        return _finish(result)

    async def execute_python(state: AgentState) -> dict:
        try:
            result = await run_python(settings.sandbox_url, Path(state["db_path"]), state["code"])
        except ExecutionError as exc:
            return {"error": str(exc)}
        return _finish(result)

    def after_execute(retry_node: str):
        def decide(state: AgentState) -> str:
            if not state.get("error"):
                return "summarize"
            return retry_node if state["attempts"] < max_attempts else "fail"

        return decide

    async def analyze_sentiment(state: AgentState) -> dict:
        if sentiment_model is None:
            return {"answer": "Model analisis sentimen belum tersedia di server ini."}
        try:
            result, meta = await asyncio.to_thread(
                analyze_text_sentiment, Path(state["db_path"]), state["question"], sentiment_model
            )
        except SentimentUnavailable as exc:
            return {"answer": str(exc)}
        note = (
            f"Kolom teks: {meta['text_column']}. Dianalisis {meta['analyzed']} dari {meta['total']} teks"
            f"{' (sampel acak)' if meta['sampled'] else ''} dengan model {meta['model']}."
        )
        if meta["group_column"]:
            note += f" Dikelompokkan per {meta['group_column']}."
        finished = _finish(result)
        finished["result"]["stdout"] = note
        if meta["group_column"]:
            # One measure per group reads cleaner than three category-coloured series for polarity.
            finished["chart"] = {"kind": "bar", "x": meta["group_column"], "y": ["persen_negatif"]}
        return {**finished, "samples": meta["samples"]}

    async def run_forecast(state: AgentState) -> dict:
        try:
            table, chart, note, answer = await forecasting.run(
                Path(state["db_path"]), state["question"], settings.forecast_url
            )
        except forecasting.ForecastUnavailable as exc:
            return {"answer": str(exc)}
        finished = _finish(table)
        finished["result"]["stdout"] = note
        return {**finished, "chart": chart, "answer": answer}

    async def answer_document(state: AgentState) -> dict:
        chunks = [documents.Chunk(**c) for c in state["doc_chunks"]]
        question, intent = state["question"], state["intent"]
        hosted = getattr(llm, "has_strong_model", False)
        file_names = list(dict.fromkeys(c.source for c in chunks if c.source))
        pages = len({(c.source, c.page) for c in chunks})
        described = f"{len(file_names)} file ({pages} halaman)" if len(file_names) > 1 else f"dokumen ({pages} halaman)"
        sources: list[dict] = []
        history = state.get("history", "")

        if hosted and documents.word_count(chunks) <= settings.doc_full_context_words:
            # A hosted model reads the whole document in a second or two, so any question (facts,
            # summary, critique) is answered from all of it rather than from a few retrieved passages.
            # The document comes first in the prompt so repeat questions hit the provider's prompt cache.
            passages = documents.page_texts(chunks)
            system = prompts.DOC_FULL_SYSTEM
            note = f"Jawaban dibuat dari seluruh isi {described}."
            max_tokens = 1500
        elif intent in ("doc_summary", "doc_analysis"):
            limit = documents.SUMMARY_EXCERPTS_HOSTED if hosted else documents.SUMMARY_EXCERPTS
            picked, sampled = documents.summary_excerpts(chunks, limit)
            passages = [(c.source, c.page, c.text) for c in picked]
            what = "Ringkasan" if intent == "doc_summary" else "Jawaban"
            if sampled:
                note = (
                    f"Isi {described} terlalu panjang untuk dibaca sekaligus; {what.lower()} dibuat dari "
                    f"{len(picked)} cuplikan yang tersebar merata."
                )
            else:
                note = f"{what} dibuat dari seluruh isi {described}."
            system = prompts.DOC_SUMMARY_SYSTEM if intent == "doc_summary" else prompts.DOC_ANALYSIS_SYSTEM
            # A CPU model writes about 7 tokens per second; a long answer would outlast the request timeout.
            max_tokens = 900 if hosted else 320
        else:
            # "jelaskan lebih detail" alone retrieves nothing useful; borrow the previous question's words.
            query = question if len(question.split()) >= 4 else f"{state.get('previous_question', '')} {question}"
            k = documents.TOP_K_HOSTED if hosted else documents.TOP_K
            hits = await asyncio.to_thread(lambda: documents.Retriever(chunks).search(query, k))
            if not hits:
                return {"answer": "Saya tidak menemukan bagian dokumen yang berkaitan dengan pertanyaan itu.", "sources": []}
            picked = [chunk for chunk, _ in hits]
            passages = [(c.source, c.page, c.text) for c in picked]
            # Shown strongest first, the reverse of the order the model reads them in.
            sources = [{"file": c.source, "page": c.page, "text": c.text[:400]} for c in reversed(picked)]
            system, note, max_tokens = prompts.DOC_QA_SYSTEM, "", 600

        request = prompts.document_request(question, passages, history)
        out = await llm.complete("complex", system, [{"role": "user", "content": request}], max_tokens)
        return {
            "answer": out.text.strip(),
            "sources": sources,
            "result": {"columns": [], "types": [], "rows": [], "truncated": False, "stdout": note, "value": None, "type": "table"},
            "usage": _usage(out, intent) + [{"step": "context", "passages": len(passages), "files": len(file_names) or 1}],
        }

    async def write_document(state: AgentState) -> dict:
        """Write a complete document (paper, proposal, letter, ...) rather than a chat-sized answer."""
        hosted = getattr(llm, "has_strong_model", False)
        passages: list[tuple[str, int, str]] = []
        if state.get("doc_chunks"):
            chunks = [documents.Chunk(**c) for c in state["doc_chunks"]]
            if hosted and documents.word_count(chunks) <= settings.doc_full_context_words:
                passages = documents.page_texts(chunks)
            else:
                limit = documents.SUMMARY_EXCERPTS_HOSTED if hosted else documents.SUMMARY_EXCERPTS
                passages = [(c.source, c.page, c.text) for c in documents.summary_excerpts(chunks, limit)[0]]

        system = prompts.COMPOSE_SYSTEM.format(
            length=prompts.COMPOSE_LENGTH_HOSTED if hosted else prompts.COMPOSE_LENGTH_LOCAL
        )
        request = prompts.document_request(state["question"], passages, state.get("history", ""))
        # A CPU model writes about 7 tokens per second, so it can only draft a short version in time.
        out = await llm.complete("complex", system, [{"role": "user", "content": request}], 6000 if hosted else 520)
        markdown = strip_fence(out.text)
        title, words = exports.document_title(markdown), exports.word_count(markdown)
        answer = f'Dokumen "{title}" selesai ditulis (sekitar {exports.format_count(words)} kata).'
        if not hosted:
            answer += " Ini versi ringkas dari model lokal; untuk dokumen panjang dan lengkap, aktifkan API LLM."
        answer += " Periksa kembali fakta, angka, dan rujukannya sebelum dipakai."
        return {
            "document": {"title": title, "markdown": markdown, "words": words},
            "answer": answer,
            "usage": _usage(out, "compose"),
        }

    async def describe_dataset(state: AgentState) -> dict:
        table, answer = await asyncio.to_thread(profiling.describe, Path(state["db_path"]))
        return {**_finish(table), "chart": None, "answer": answer}

    async def summarize(state: AgentState) -> dict:
        content = f"Pertanyaan: {state['question']}\n\n<hasil>\n{_result_for_prompt(state['result'])}\n</hasil>"
        try:
            out = await llm.complete("simple", prompts.SUMMARY_SYSTEM, [{"role": "user", "content": content}], 400)
        except LLMError:
            return {"answer": "Berikut hasil analisisnya.", "usage": []}
        return {"answer": out.text.strip(), "usage": _usage(out, "summarize")}

    async def answer_chat(state: AgentState) -> dict:
        system = prompts.CHAT_SYSTEM + (f"\n{state['file_note']}" if state.get("file_note") else "")
        content = state["question"]
        if state.get("history"):
            content = f"<riwayat>\n{state['history']}\n</riwayat>\n\n{content}"
        try:
            out = await llm.complete("simple", system, [{"role": "user", "content": content}], 600)
        except LLMError:
            return {"answer": "Halo! Silakan tanyakan sesuatu tentang dataset yang kamu unggah.", "usage": []}
        return {"answer": out.text.strip(), "usage": _usage(out, "chat")}

    async def fail(state: AgentState) -> dict:
        return {
            "answer": (
                f"Maaf, saya belum berhasil menjawab setelah {state['attempts']} percobaan. "
                "Coba ubah pertanyaannya agar lebih spesifik, misalnya sebutkan nama kolom yang dimaksud."
            )
        }

    graph = StateGraph(AgentState)
    graph.add_node("route", route)
    graph.add_node("generate_sql", _generator(prompts.SQL_SYSTEM, "generate_sql"))
    graph.add_node("execute_sql", execute_sql)
    graph.add_node("generate_python", _generator(prompts.PYTHON_SYSTEM, "generate_python"))
    graph.add_node("execute_python", execute_python)
    graph.add_node("summarize", summarize)
    graph.add_node("analyze_sentiment", analyze_sentiment)
    graph.add_node("run_forecast", run_forecast)
    graph.add_node("describe_dataset", describe_dataset)
    graph.add_node("answer_document", answer_document)
    graph.add_node("write_document", write_document)
    graph.add_node("answer_chat", answer_chat)
    graph.add_node("fail", fail)

    graph.add_edge(START, "route")
    graph.add_conditional_edges(
        "route",
        lambda s: s["intent"],
        {"sql": "generate_sql", "python": "generate_python", "sentiment": "analyze_sentiment", "forecast": "run_forecast", "profile": "describe_dataset", "chat": "answer_chat",
         "doc_qa": "answer_document", "doc_summary": "answer_document", "doc_analysis": "answer_document",
         "compose": "write_document"},
    )
    graph.add_conditional_edges(
        "analyze_sentiment", lambda s: END if s.get("answer") else "summarize", ["summarize", END]
    )
    graph.add_edge("generate_sql", "execute_sql")
    graph.add_conditional_edges("execute_sql", after_execute("generate_sql"), ["generate_sql", "summarize", "fail"])
    graph.add_edge("generate_python", "execute_python")
    graph.add_conditional_edges(
        "execute_python", after_execute("generate_python"), ["generate_python", "summarize", "fail"]
    )
    for terminal in ("summarize", "answer_chat", "fail", "run_forecast", "describe_dataset", "answer_document", "write_document"):
        graph.add_edge(terminal, END)
    return graph.compile()


def _events_for(node: str, update: dict) -> list[dict]:
    if node == "route":
        return [{"type": "intent", "value": update["intent"]}]
    if node in ("generate_sql", "generate_python"):
        language = "sql" if node == "generate_sql" else "python"
        return [{"type": "code", "language": language, "value": update["code"], "attempt": update["attempts"]}]
    if node in ("execute_sql", "execute_python", "analyze_sentiment", "run_forecast", "describe_dataset", "answer_document"):
        if update.get("error"):
            return [{"type": "retry", "error": update["error"]}]
        events = []
        result = update.get("result")
        if result:
            if result["columns"]:
                events.append({k: result[k] for k in ("type", "columns", "types", "rows", "truncated")})
            elif result.get("value") is not None:
                events.append({"type": "value", "value": result["value"]})
            if result.get("stdout"):
                events.append({"type": "stdout", "value": result["stdout"]})
            if update.get("chart"):
                events.append({"type": "chart", "spec": update["chart"]})
            if update.get("samples"):
                events.append({"type": "samples", "items": update["samples"]})
        if update.get("sources"):
            events.append({"type": "sources", "items": update["sources"]})
        # These nodes may also finish the turn themselves (a forecast, or "no text column").
        if update.get("answer"):
            events.append({"type": "answer", "text": update["answer"]})
        return events
    if node == "write_document":
        return [{"type": "document", **update["document"]}, {"type": "answer", "text": update["answer"]}]
    if node in ("summarize", "answer_chat", "fail"):
        return [{"type": "answer", "text": update["answer"]}]
    return []


async def stream_agent(agent, initial: AgentState) -> AsyncIterator[tuple[dict, list[dict]]]:
    """Yield (event, usage_so_far) pairs as each graph node finishes."""
    usage: list[dict] = []
    async for chunk in agent.astream(initial, stream_mode="updates"):
        for node, update in chunk.items():
            if not update:
                continue
            usage.extend(update.get("usage", []))
            for event in _events_for(node, update):
                yield event, usage
