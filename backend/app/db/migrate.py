"""Small, idempotent schema upgrades run at startup.

`create_all` creates missing tables but never alters existing ones, so every column added after the
first release is listed here. Each statement is safe to run again on an up-to-date database.
"""

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncConnection

STATEMENTS = [
    "ALTER TABLE datasets ADD COLUMN IF NOT EXISTS kind VARCHAR(16) NOT NULL DEFAULT 'table'",
    # Files now belong to a chat session (many per session) instead of a session pointing at one file.
    "ALTER TABLE datasets ADD COLUMN IF NOT EXISTS session_id UUID REFERENCES chat_sessions(id) ON DELETE CASCADE",
    "CREATE INDEX IF NOT EXISTS ix_datasets_session_id ON datasets (session_id)",
    "ALTER TABLE datasets ADD COLUMN IF NOT EXISTS source VARCHAR(16) NOT NULL DEFAULT ''",
    "ALTER TABLE datasets ADD COLUMN IF NOT EXISTS ocr_pages INTEGER NOT NULL DEFAULT 0",
    # Carry over sessions created under the old one-file model, then detach the old link so that
    # removing a file no longer deletes the whole conversation.
    """
    DO $$
    BEGIN
        IF EXISTS (
            SELECT 1 FROM information_schema.columns
            WHERE table_name = 'chat_sessions' AND column_name = 'dataset_id'
        ) THEN
            ALTER TABLE chat_sessions ALTER COLUMN dataset_id DROP NOT NULL;
            UPDATE datasets d SET session_id = s.id
            FROM chat_sessions s
            WHERE s.dataset_id = d.id AND d.session_id IS NULL;
            UPDATE chat_sessions SET dataset_id = NULL WHERE dataset_id IS NOT NULL;
        END IF;
    END $$
    """,
]


async def upgrade(conn: AsyncConnection) -> None:
    for statement in STATEMENTS:
        await conn.execute(text(statement))
