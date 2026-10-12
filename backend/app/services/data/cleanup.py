import asyncio
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

from sqlalchemy import select

from app.core.config import get_settings
from app.db.models import ChatSession, Dataset
from app.db.session import SessionLocal

logger = logging.getLogger(__name__)

CLEANUP_INTERVAL_SECONDS = 3600


async def delete_expired_datasets() -> int:
    async with SessionLocal() as session:
        now = datetime.now(timezone.utc)
        expired = (await session.scalars(select(Dataset).where(Dataset.expires_at <= now))).all()
        for dataset in expired:
            if dataset.duckdb_path:  # documents keep their text in the database, not in a file
                Path(dataset.duckdb_path).unlink(missing_ok=True)
            await session.delete(dataset)
        await session.flush()

        # A conversation quotes the user's data, so it goes too once it is past the retention
        # window and none of its files remain.
        cutoff = now - timedelta(hours=get_settings().dataset_ttl_hours)
        has_files = select(Dataset.id).where(Dataset.session_id == ChatSession.id).exists()
        stale = await session.scalars(select(ChatSession).where(ChatSession.created_at <= cutoff, ~has_files))
        for chat in stale:
            await session.delete(chat)
        await session.commit()
        return len(expired)


async def cleanup_loop() -> None:
    while True:
        try:
            removed = await delete_expired_datasets()
            if removed:
                logger.info("Removed %d expired datasets", removed)
        except Exception:
            logger.exception("Dataset cleanup failed")
        await asyncio.sleep(CLEANUP_INTERVAL_SECONDS)
