"""Turn an uploaded file into a stored dataset: a DuckDB table (CSV/XLSX) or document chunks (PDF/image)."""

import asyncio
import tempfile
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

from fastapi import HTTPException, UploadFile, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.db.models import Dataset, DocumentChunk
from app.services import documents
from app.services.data.ingest import SUPPORTED_SUFFIXES, IngestError, load_dataframe, write_duckdb
from app.services.data.profiling import profile_table

CHUNK_SIZE = 1024 * 1024
DOCUMENT_SUFFIXES = {".pdf"} | documents.IMAGE_SUFFIXES
ACCEPTED = "CSV, XLSX, PDF, atau gambar (PNG, JPG, WEBP)"


def _ingest_table(upload_path: Path, db_path: Path) -> tuple[int, dict]:
    df = load_dataframe(upload_path)
    row_count = write_duckdb(df, db_path)
    return row_count, profile_table(db_path)


async def store_upload(
    db: AsyncSession, user_id: uuid.UUID, file: UploadFile, session_id: uuid.UUID | None = None
) -> tuple[Dataset, dict]:
    """Validate, parse and persist one upload. Returns the dataset and its profile."""
    settings = get_settings()
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES | DOCUMENT_SUFFIXES:
        raise HTTPException(status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, f"Hanya file {ACCEPTED} yang didukung")

    max_bytes = settings.max_upload_mb * 1024 * 1024
    dataset_id = uuid.uuid4()
    expires_at = datetime.now(timezone.utc) + timedelta(hours=settings.dataset_ttl_hours)
    common = dict(id=dataset_id, user_id=user_id, session_id=session_id, original_filename=file.filename, expires_at=expires_at)

    with tempfile.TemporaryDirectory() as tmp:
        upload_path = Path(tmp) / f"upload{suffix}"
        size = 0
        with upload_path.open("wb") as out:
            while chunk := await file.read(CHUNK_SIZE):
                size += len(chunk)
                if size > max_bytes:
                    raise HTTPException(
                        status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, f"Ukuran file maksimal {settings.max_upload_mb} MB"
                    )
                out.write(chunk)

        if suffix in DOCUMENT_SUFFIXES:
            is_image = suffix in documents.IMAGE_SUFFIXES
            try:
                # Parsing and OCR are blocking; keep the event loop free for other users.
                extraction = await asyncio.to_thread(
                    documents.extract_image if is_image else documents.extract_pages, upload_path
                )
            except documents.DocumentError as exc:
                raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, str(exc)) from exc
            chunks = documents.chunk_pages(extraction.pages)
            dataset = Dataset(
                **common,
                kind="document",
                source="image" if is_image else "pdf",
                ocr_pages=extraction.ocr_pages,
                duckdb_path="",
                row_count=len(extraction.pages),
                schema_json=[],
            )
            db.add(dataset)
            db.add_all(DocumentChunk(dataset_id=dataset_id, page=c.page, position=c.position, text=c.text) for c in chunks)
            profile = {"row_count": len(extraction.pages), "columns": [], "sample": []}
        else:
            db_path = Path(settings.data_dir) / "duckdb" / str(user_id) / f"{dataset_id}.duckdb"
            try:
                row_count, profile = await asyncio.to_thread(_ingest_table, upload_path, db_path)
            except IngestError as exc:
                db_path.unlink(missing_ok=True)
                raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, str(exc)) from exc
            dataset = Dataset(
                **common,
                kind="table",
                source=suffix.lstrip("."),
                duckdb_path=str(db_path),
                row_count=row_count,
                schema_json=profile["columns"],
            )
            db.add(dataset)

    await db.commit()
    await db.refresh(dataset)
    return dataset, profile


def remove_files(dataset: Dataset) -> None:
    """Delete what a dataset keeps on disk (document text lives in the database and cascades)."""
    if dataset.duckdb_path:
        Path(dataset.duckdb_path).unlink(missing_ok=True)
