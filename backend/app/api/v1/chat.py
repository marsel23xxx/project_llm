import asyncio
import json
import logging
import re
import time
import uuid
from datetime import date
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request, status
from fastapi.responses import Response, StreamingResponse
from sqlalchemy import select

from app.api.deps import CurrentUser, DbSession, RedisClient
from app.core import rate_limit
from app.core.config import get_settings
from app.db.models import ChatSession, Dataset, DocumentChunk, Message
from app.db.session import SessionLocal
from app.schemas import MessageIn, SessionCreate, SessionDetail, SessionOut
from app.services.agent import prompts
from app.services.agent.executor import run_sql
from app.services.agent.graph import stream_agent
from app.services.llm.base import LLMError
from app.services.report import build_pdf

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/chat", tags=["chat"])

SAMPLE_ROWS = 3
QUOTA_TTL_SECONDS = 2 * 86400

# The agent cannot write files. A request such as "... dalam bentuk pdf" is answered normally and
# pointed at the report button, instead of letting the LLM try to generate a PDF in the sandbox.
_EXPORT_REQUEST = re.compile(
    r"\b(?:(?:dengan|dalam|ke|jadi|menjadi|sebagai|format|bentuk|file|berupa|via|lewat)\s+)*"
    r"(?:pdf|excel|xlsx|csv|word|docx|powerpoint|ppt)\b|\b(?:unduh|download|export|ekspor|cetak|print)\b",
    re.IGNORECASE,
)
EXPORT_HINT = "Untuk menyimpannya sebagai file, gunakan tombol Unduh PDF di bawah jawaban ini."


def split_export_request(question: str) -> tuple[str, bool]:
    """Return the question without the file-export wording, and whether such wording was present."""
    cleaned = re.sub(r"\s{2,}", " ", _EXPORT_REQUEST.sub(" ", question)).strip(" ,.")
    wants_file = cleaned != question.strip(" ,.")
    return (cleaned if len(cleaned) >= 3 else question), wants_file


async def _get_session(db: DbSession, user: CurrentUser, session_id: uuid.UUID) -> ChatSession:
    session = await db.get(ChatSession, session_id)
    if session is None or session.user_id != user.id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Sesi chat tidak ditemukan")
    return session


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


def _session_out(session: ChatSession, dataset: Dataset | None) -> dict:
    extra = {"kind": dataset.kind, "size": dataset.row_count} if dataset else {}
    return {**SessionOut.model_validate(session).model_dump(), **extra}


def _sse(event: dict) -> str:
    return f"data: {json.dumps(event, ensure_ascii=False, default=str)}\n\n"


@router.post("/sessions", response_model=SessionOut, status_code=status.HTTP_201_CREATED)
async def create_session(body: SessionCreate, user: CurrentUser, db: DbSession) -> ChatSession:
    dataset = await db.get(Dataset, body.dataset_id)
    if dataset is None or dataset.user_id != user.id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Dataset tidak ditemukan")
    session = ChatSession(user_id=user.id, dataset_id=dataset.id, title=dataset.original_filename[:120])
    db.add(session)
    await db.commit()
    await db.refresh(session)
    return _session_out(session, dataset)


@router.get("/sessions", response_model=list[SessionOut])
async def list_sessions(user: CurrentUser, db: DbSession) -> list[dict]:
    rows = await db.execute(
        select(ChatSession, Dataset)
        .join(Dataset, Dataset.id == ChatSession.dataset_id)
        .where(ChatSession.user_id == user.id)
        .order_by(ChatSession.created_at.desc())
    )
    return [_session_out(session, dataset) for session, dataset in rows]


@router.get("/sessions/{session_id}", response_model=SessionDetail)
async def get_session(session_id: uuid.UUID, user: CurrentUser, db: DbSession) -> dict:
    session = await _get_session(db, user, session_id)
    messages = await db.scalars(
        select(Message).where(Message.session_id == session.id).order_by(Message.created_at)
    )
    dataset = await db.get(Dataset, session.dataset_id)
    return {**_session_out(session, dataset), "messages": list(messages)}


@router.post("/sessions/{session_id}/messages")
async def send_message(
    session_id: uuid.UUID, body: MessageIn, request: Request, user: CurrentUser, db: DbSession, redis: RedisClient
) -> StreamingResponse:
    settings = get_settings()
    session = await _get_session(db, user, session_id)
    dataset = await db.get(Dataset, session.dataset_id)
    is_document = dataset is not None and dataset.kind == "document"
    if dataset is None or (not is_document and not Path(dataset.duckdb_path).exists()):
        raise HTTPException(status.HTTP_410_GONE, "Dataset sudah kedaluwarsa, silakan unggah ulang")

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
    extra: dict = {}
    if is_document:
        chunks = await db.scalars(
            select(DocumentChunk).where(DocumentChunk.dataset_id == dataset.id).order_by(DocumentChunk.position)
        )
        extra = {
            "doc_chunks": [{"page": c.page, "position": c.position, "text": c.text} for c in chunks],
            # Only the last exchange: a small model loses the thread with more, and context is tight.
            "history": _history_text(history[-2:]),
            "previous_question": next((m.content for m in reversed(history) if m.role == "user"), ""),
            "file_note": f"File aktif di sesi ini: dokumen PDF \"{dataset.original_filename}\" ({dataset.row_count} halaman).",
        }
        context = ""
    else:
        sample = await asyncio.to_thread(
            run_sql, Path(dataset.duckdb_path), "SELECT * FROM data", settings.sql_timeout_seconds, SAMPLE_ROWS
        )
        context = prompts.dataset_context(
            _schema_text(dataset),
            json.dumps({"kolom": sample.columns, "baris": sample.rows}, ensure_ascii=False, default=str),
            _history_text(history),
        )

    db.add(Message(session_id=session.id, role="user", content=body.content, steps_json=[], usage_json=[]))
    await db.commit()

    agent = request.app.state.agent
    question, wants_file = split_export_request(body.content)
    initial = {"question": question, "context": context, "db_path": dataset.duckdb_path, "usage": [], **extra}
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
                        if wants_file:
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


@router.get("/messages/{message_id}/pdf")
async def message_pdf(message_id: uuid.UUID, user: CurrentUser, db: DbSession) -> Response:
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
    pdf = await asyncio.to_thread(
        build_pdf, question or "-", message.content, message.steps_json, session.title, message.created_at
    )
    return Response(
        pdf,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="laporan-{str(message_id)[:8]}.pdf"'},
    )
