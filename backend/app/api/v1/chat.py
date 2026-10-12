import asyncio
import json
import logging
import re
import time
import uuid
from datetime import date
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request, UploadFile, status
from fastapi.responses import Response, StreamingResponse
from sqlalchemy import select

from app.api.deps import CurrentUser, DbSession, RedisClient
from app.core import rate_limit
from app.core.config import get_settings
from app.db.models import ChatSession, Dataset, DocumentChunk, Message
from app.db.session import SessionLocal
from app.schemas import FileOut, MessageIn, SessionDetail, SessionOut
from app.services import documents, exports
from app.services.agent import prompts
from app.services.agent.executor import run_sql
from app.services.agent.graph import stream_agent
from app.services.llm.base import LLMError
from app.services.report import build_pdf
from app.services.uploads import remove_files, store_upload

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/chat", tags=["chat"])

SAMPLE_ROWS = 3
QUOTA_TTL_SECONDS = 2 * 86400
DEFAULT_TITLE = "Chat baru"
MAX_FILES_PER_SESSION = 10

# The agent itself cannot write files. Wording that asks for one ("... dalam bentuk pdf dan docx") is
# taken out of the question, and the reply then offers exactly those formats as downloads.
_FORMAT_WORDS = {
    "pdf": "pdf",
    "docx": "docx", "doc": "docx", "word": "docx",
    "xlsx": "xlsx", "xls": "xlsx", "excel": "xlsx", "csv": "xlsx", "spreadsheet": "xlsx",
}  # fmt: skip
_LEAD_IN = r"(?:(?:dengan|dalam|ke|jadi|menjadi|sebagai|format|bentuk|berbentuk|file|berupa|via|lewat|output|hasilnya|dan|atau|&|,)\s*)*"
_EXPORT_REQUEST = re.compile(
    rf"\b{_LEAD_IN}\.?({'|'.join(_FORMAT_WORDS)})\b|\b(?:unduh|download|export|ekspor|cetak|print)\b", re.IGNORECASE
)
EXPORT_HINT = "File bisa diunduh lewat tombol di bawah jawaban ini."


def split_export_request(question: str) -> tuple[str, list[str]]:
    """Return the question without the file-export wording, and the formats asked for (in order)."""
    matches = list(_EXPORT_REQUEST.finditer(question))
    formats: list[str] = []
    for match in matches:
        fmt = _FORMAT_WORDS.get((match.group(1) or "").lower())
        if fmt and fmt not in formats:
            formats.append(fmt)
    if matches and not formats:  # "unduh" or "cetak" with no format named means PDF
        formats = ["pdf"]
    cleaned = re.sub(r"\s{2,}", " ", _EXPORT_REQUEST.sub(" ", question)).strip(" ,.")
    return (cleaned if len(cleaned) >= 3 else question), formats


def offered_formats(requested: list[str], steps: list[dict]) -> list[str]:
    """Which download buttons a reply gets: none unless a file was asked for or a document was written."""
    kinds = {step.get("type") for step in steps}
    formats = list(requested)
    if "document" in kinds:  # a written document is a file by nature
        formats += [f for f in ("pdf", "docx") if f not in formats]
    return [f for f in formats if f != "xlsx" or "table" in kinds]  # a spreadsheet needs a result table


async def _get_session(db: DbSession, user: CurrentUser, session_id: uuid.UUID) -> ChatSession:
    session = await db.get(ChatSession, session_id)
    if session is None or session.user_id != user.id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Sesi chat tidak ditemukan")
    return session


def _file_note(files: list[Dataset], table: Dataset | None) -> str:
    """What the assistant should know about the session's files when it is just chatting."""
    if not files:
        return "Belum ada file di sesi ini. User bisa melampirkan CSV, Excel, PDF, atau gambar lewat tombol lampiran."
    listed = "; ".join(
        f'"{f.original_filename}" ({f.row_count} {"baris" if f.kind == "table" else "halaman"})' for f in files
    )
    note = f"File di sesi ini: {listed}. File sudah dilampirkan; jangan meminta user mengunggah lagi."
    if table and sum(f.kind == "table" for f in files) > 1:
        note += f' Tabel yang dianalisis adalah yang terakhir diunggah: "{table.original_filename}".'
    return note


def _schema_text(dataset: Dataset) -> str:
    lines = ["Tabel: data", f"Jumlah baris: {dataset.row_count}", "Kolom:"]
    for col in dataset.schema_json:
        detail = f"- {col['name']} ({col['type']})"
        if col.get("min") is not None:
            detail += f", rentang {col['min']} s.d. {col['max']}"
        if col.get("approx_unique") is not None:
            detail += f", ~{col['approx_unique']} nilai unik"
        lines.append(detail)
    return "\n".join(lines)


def _history_text(messages: list[Message]) -> str:
    lines = []
    for msg in messages:
        if msg.role == "user":
            lines.append(f"User: {msg.content}")
            continue
        code = next((s["value"] for s in reversed(msg.steps_json) if s.get("type") == "code"), None)
        lines.append(f"Asisten: {msg.content}" + (f"\nKode yang dipakai:\n{code}" if code else ""))
    return "\n".join(lines)


async def _save_reply(
    message_id: uuid.UUID, session_id: uuid.UUID, answer: str, steps: list, usage: list, latency_ms: int
) -> None:
    # The request-scoped DB session may already be closed once streaming starts.
    async with SessionLocal() as db:
        db.add(
            Message(
                id=message_id,
                session_id=session_id,
                role="assistant",
                content=answer,
                steps_json=steps,
                usage_json=usage,
                latency_ms=latency_ms,
            )
        )
        await db.commit()


def _file_out(dataset: Dataset) -> dict:
    return {
        "id": dataset.id,
        "name": dataset.original_filename,
        "kind": dataset.kind,
        "source": dataset.source or ("pdf" if dataset.kind == "document" else "csv"),
        "size": dataset.row_count,
        "ocr_pages": dataset.ocr_pages,
        "vision_pages": dataset.vision_pages,
        "expires_at": dataset.expires_at,
    }


async def _session_files(db: DbSession, session_ids: list[uuid.UUID]) -> dict[uuid.UUID, list[Dataset]]:
    grouped: dict[uuid.UUID, list[Dataset]] = {sid: [] for sid in session_ids}
    if session_ids:
        rows = await db.scalars(select(Dataset).where(Dataset.session_id.in_(session_ids)).order_by(Dataset.created_at))
        for dataset in rows:
            grouped[dataset.session_id].append(dataset)
    return grouped


def _session_out(session: ChatSession, files: list[Dataset]) -> dict:
    return {"id": session.id, "title": session.title, "created_at": session.created_at, "files": [_file_out(f) for f in files]}


def _sse(event: dict) -> str:
    return f"data: {json.dumps(event, ensure_ascii=False, default=str)}\n\n"


@router.post("/sessions", response_model=SessionOut, status_code=status.HTTP_201_CREATED)
async def create_session(user: CurrentUser, db: DbSession) -> dict:
    """Start an empty conversation. Files are attached afterwards, any number of times."""
    session = ChatSession(user_id=user.id, title=DEFAULT_TITLE)
    db.add(session)
    await db.commit()
    await db.refresh(session)
    return _session_out(session, [])


@router.get("/sessions", response_model=list[SessionOut])
async def list_sessions(user: CurrentUser, db: DbSession) -> list[dict]:
    sessions = list(
        await db.scalars(select(ChatSession).where(ChatSession.user_id == user.id).order_by(ChatSession.created_at.desc()))
    )
    files = await _session_files(db, [s.id for s in sessions])
    return [_session_out(s, files[s.id]) for s in sessions]


@router.get("/sessions/{session_id}", response_model=SessionDetail)
async def get_session(session_id: uuid.UUID, user: CurrentUser, db: DbSession) -> dict:
    session = await _get_session(db, user, session_id)
    messages = await db.scalars(
        select(Message).where(Message.session_id == session.id).order_by(Message.created_at)
    )
    files = await _session_files(db, [session.id])
    return {**_session_out(session, files[session.id]), "messages": list(messages)}


@router.delete("/sessions/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_session(session_id: uuid.UUID, user: CurrentUser, db: DbSession) -> None:
    """Delete a conversation with its messages and every file attached to it."""
    session = await _get_session(db, user, session_id)
    for dataset in (await _session_files(db, [session.id]))[session.id]:
        remove_files(dataset)
        await db.delete(dataset)
    await db.delete(session)
    await db.commit()


@router.post("/sessions/{session_id}/files", response_model=FileOut, status_code=status.HTTP_201_CREATED)
async def attach_file(session_id: uuid.UUID, file: UploadFile, user: CurrentUser, db: DbSession) -> dict:
    session = await _get_session(db, user, session_id)
    existing = (await _session_files(db, [session.id]))[session.id]
    if len(existing) >= MAX_FILES_PER_SESSION:
        raise HTTPException(
            status.HTTP_409_CONFLICT, f"Satu chat maksimal {MAX_FILES_PER_SESSION} file. Hapus file lama atau mulai chat baru."
        )
    dataset, _ = await store_upload(db, user.id, file, session.id)
    if session.title == DEFAULT_TITLE:
        session.title = dataset.original_filename[:120]
        await db.commit()
    return _file_out(dataset)


@router.post("/sessions/{session_id}/messages")
async def send_message(
    session_id: uuid.UUID, body: MessageIn, request: Request, user: CurrentUser, db: DbSession, redis: RedisClient
) -> StreamingResponse:
    settings = get_settings()
    session = await _get_session(db, user, session_id)
    files = (await _session_files(db, [session.id]))[session.id]
    docs = [f for f in files if f.kind == "document"]
    tables = [f for f in files if f.kind == "table" and Path(f.duckdb_path).exists()]
    # One table is queried at a time: the most recently attached.
    table = tables[-1] if tables else None

    quota_key = f"quota:{user.id}:{date.today().isoformat()}"
    if not await rate_limit.hit(redis, quota_key, settings.daily_question_quota, QUOTA_TTL_SECONDS):
        raise HTTPException(
            status.HTTP_429_TOO_MANY_REQUESTS, f"Kuota {settings.daily_question_quota} pertanyaan per hari sudah habis"
        )

    history = list(
        await db.scalars(
            select(Message)
            .where(Message.session_id == session.id)
            .order_by(Message.created_at.desc())
            .limit(settings.history_messages)
        )
    )[::-1]
    state: dict = {
        # Only the last exchange: a small model loses the thread with more, and context is tight.
        "history": _history_text(history[-2:]),
        "previous_question": next((m.content for m in reversed(history) if m.role == "user"), ""),
        "file_note": _file_note(files, table),
        "db_path": table.duckdb_path if table else "",
        "context": "",
    }
    if docs:
        rows = await db.scalars(
            select(DocumentChunk)
            .where(DocumentChunk.dataset_id.in_([d.id for d in docs]))
            .order_by(DocumentChunk.dataset_id, DocumentChunk.position)
        )
        names = {d.id: d.original_filename for d in docs}
        by_file: dict[uuid.UUID, list[documents.Chunk]] = {d.id: [] for d in docs}  # keeps upload order
        for c in rows:
            by_file[c.dataset_id].append(documents.Chunk(c.page, c.position, c.text, names[c.dataset_id]))
        merged = documents.renumber(list(by_file.values()))
        state["doc_chunks"] = [{"page": c.page, "position": c.position, "text": c.text, "source": c.source} for c in merged]
    if table:
        sample = await asyncio.to_thread(
            run_sql, Path(table.duckdb_path), "SELECT * FROM data", settings.sql_timeout_seconds, SAMPLE_ROWS
        )
        state["context"] = prompts.dataset_context(
            _schema_text(table),
            json.dumps({"kolom": sample.columns, "baris": sample.rows}, ensure_ascii=False, default=str),
            _history_text(history),
        )

    if session.title == DEFAULT_TITLE:
        session.title = body.content[:60]
    db.add(Message(session_id=session.id, role="user", content=body.content, steps_json=[], usage_json=[]))
    await db.commit()

    agent = request.app.state.agent
    question, requested_formats = split_export_request(body.content)
    initial = {"question": question, "usage": [], **state}
    message_id = uuid.uuid4()

    async def event_stream():
        started = time.monotonic()
        steps: list[dict] = []
        usage: list[dict] = []
        answer = ""
        completed = False
        try:
            try:
                async for event, usage in stream_agent(agent, initial):
                    if event["type"] == "answer":
                        formats = offered_formats(requested_formats, steps)
                        if formats:
                            # Sent before the answer so the buttons are known when the text arrives.
                            files_event = {"type": "files", "formats": formats}
                            steps.append(files_event)
                            yield _sse(files_event)
                            event = {**event, "text": f"{event['text']}\n\n{EXPORT_HINT}"}
                        answer = event["text"]
                    steps.append(event)
                    yield _sse(event)
            except LLMError as exc:
                logger.warning("LLM tidak tersedia untuk sesi %s: %s", session_id, exc)
                answer = "Model AI sedang tidak tersedia. Silakan coba beberapa saat lagi."
                error_event = {"type": "error", "message": answer}
                steps.append(error_event)
                yield _sse(error_event)
            except Exception:
                logger.exception("Agent gagal untuk sesi %s", session_id)
                answer = "Maaf, terjadi kesalahan saat memproses pertanyaan. Silakan coba lagi."
                error_event = {"type": "error", "message": answer}
                steps.append(error_event)
                yield _sse(error_event)
            completed = True
        finally:
            # Runs on client disconnect too, so the user's question never ends up without a reply.
            if not completed:
                answer = answer or "Permintaan terputus sebelum selesai."
                steps.append({"type": "error", "message": answer})
            latency_ms = int((time.monotonic() - started) * 1000)
            await asyncio.shield(_save_reply(message_id, session_id, answer, steps, usage, latency_ms))
        yield _sse({"type": "done", "message_id": str(message_id), "latency_ms": latency_ms, "usage": usage})

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


def _last(steps: list[dict], kind: str) -> dict | None:
    return next((step for step in reversed(steps) if step.get("type") == kind), None)


def build_export(fmt: str, question: str, message: Message, session_title: str) -> tuple[bytes, str]:
    """Render one stored reply as a file. Returns (content, file name without extension)."""
    steps = message.steps_json
    document, table = _last(steps, "document"), _last(steps, "table")

    if fmt == "xlsx":
        if table is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "Jawaban ini tidak memiliki tabel untuk diekspor")
        return exports.render_xlsx(table["columns"], table["rows"]), exports.slug(session_title)

    if document:  # a written document: the file is the document itself
        render = exports.render_pdf if fmt == "pdf" else exports.render_docx
        return render(document["markdown"]), exports.slug(document["title"])

    if fmt == "pdf":  # an analysis answer: question, answer, chart, table and method
        return build_pdf(question, message.content, steps, session_title, message.created_at), exports.slug(session_title)

    answer = message.content.replace(f"\n\n{EXPORT_HINT}", "")
    markdown = f"# {question}\n\n{answer}"
    if table:
        markdown += "\n\n## Data\n\n" + exports.table_as_markdown(table["columns"], table["rows"])
    return exports.render_docx(markdown), exports.slug(session_title)


@router.get("/messages/{message_id}/export")
async def export_message(message_id: uuid.UUID, user: CurrentUser, db: DbSession, format: str = "pdf") -> Response:
    if format not in exports.FORMATS:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Format tidak didukung. Pilih pdf, docx, atau xlsx.")
    message = await db.get(Message, message_id)
    session = await db.get(ChatSession, message.session_id) if message else None
    if message is None or message.role != "assistant" or session is None or session.user_id != user.id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Jawaban tidak ditemukan")

    question = await db.scalar(
        select(Message.content)
        .where(Message.session_id == session.id, Message.role == "user", Message.created_at <= message.created_at)
        .order_by(Message.created_at.desc())
        .limit(1)
    )
    content, name = await asyncio.to_thread(build_export, format, question or "-", message, session.title)
    return Response(
        content,
        media_type=exports.FORMATS[format][0],
        headers={"Content-Disposition": f'attachment; filename="{name}.{format}"'},
    )
