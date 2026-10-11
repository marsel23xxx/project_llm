import asyncio
import logging
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import select

from app.db.models import Dataset
from app.db.session import SessionLocal

logger = logging.getLogger(__name__)

CLEANUP_INTERVAL_SECONDS = 3600


async def delete_expired_datasets() -> int:
    async with SessionLocal() as session:
        now = datetime.now(timezone.utc)
        expired = (await session.scalars(select(Dataset).where(Dataset.expires_at <= now))).all()
        for dataset in expired:
            Path(dataset.duckdb_path).unlink(missing_ok=True)
            await session.delete(dataset)
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
