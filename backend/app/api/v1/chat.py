import asyncio
import json
import logging
import time
import uuid
from datetime import date
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request, status
from fastapi.responses import StreamingResponse
from sqlalchemy import select

from app.api.deps import CurrentUser, DbSession, RedisClient
from app.core import rate_limit
from app.core.config import get_settings
from app.db.models import ChatSession, Dataset, Message
from app.db.session import SessionLocal
from app.schemas import MessageIn, SessionCreate, SessionDetail, SessionOut
from app.services.agent import prompts
from app.services.agent.executor import run_sql
from app.services.agent.graph import stream_agent
from app.services.llm.base import LLMError

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/chat", tags=["chat"])

SAMPLE_ROWS = 3
QUOTA_TTL_SECONDS = 2 * 86400


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


async def _save_reply(session_id: uuid.UUID, answer: str, steps: list, usage: list, latency_ms: int) -> None:
    # The request-scoped DB session may already be closed once streaming starts.
    async with SessionLocal() as db:
        db.add(
            Message(
                session_id=session_id,
                role="assistant",
                content=answer,
                steps_json=steps,
                usage_json=usage,
                latency_ms=latency_ms,
            )
        )
        await db.commit()


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
    return session


@router.get("/sessions", response_model=list[SessionOut])
async def list_sessions(user: CurrentUser, db: DbSession) -> list[ChatSession]:
    rows = await db.scalars(
        select(ChatSession).where(ChatSession.user_id == user.id).order_by(ChatSession.created_at.desc())
    )
    return list(rows)


@router.get("/sessions/{session_id}", response_model=SessionDetail)
async def get_session(session_id: uuid.UUID, user: CurrentUser, db: DbSession) -> dict:
    session = await _get_session(db, user, session_id)
    messages = await db.scalars(
        select(Message).where(Message.session_id == session.id).order_by(Message.created_at)
    )
    return {**SessionOut.model_validate(session).model_dump(), "messages": list(messages)}


@router.post("/sessions/{session_id}/messages")
async def send_message(
    session_id: uuid.UUID, body: MessageIn, request: Request, user: CurrentUser, db: DbSession, redis: RedisClient
) -> StreamingResponse:
    settings = get_settings()
    session = await _get_session(db, user, session_id)
    dataset = await db.get(Dataset, session.dataset_id)
    if dataset is None or not Path(dataset.duckdb_path).exists():
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
    initial = {"question": body.content, "context": context, "db_path": dataset.duckdb_path, "usage": []}

    async def event_stream():
        started = time.monotonic()
        steps: list[dict] = []
        usage: list[dict] = []
        answer = ""
        completed = False
        try:
            try:
                async for event, usage in stream_agent(agent, initial):
                    steps.append(event)
                    if event["type"] == "answer":
                        answer = event["text"]
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
            await asyncio.shield(_save_reply(session_id, answer, steps, usage, latency_ms))
        yield _sse({"type": "done", "latency_ms": latency_ms, "usage": usage})

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
