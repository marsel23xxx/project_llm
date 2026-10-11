import asyncio
import tempfile
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

from fastapi import APIRouter, HTTPException, UploadFile, status
from sqlalchemy import select

from app.api.deps import CurrentUser, DbSession
from app.core.config import get_settings
from app.db.models import Dataset
from app.schemas import DatasetOut, DatasetProfile, DatasetUploaded
from app.services.data.ingest import SUPPORTED_SUFFIXES, IngestError, load_dataframe, write_duckdb
from app.services.data.profiling import profile_table

router = APIRouter(prefix="/datasets", tags=["datasets"])

CHUNK_SIZE = 1024 * 1024


async def _get_owned(db: DbSession, user: CurrentUser, dataset_id: uuid.UUID) -> Dataset:
    dataset = await db.get(Dataset, dataset_id)
    if dataset is None or dataset.user_id != user.id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Dataset tidak ditemukan")
    return dataset


def _ingest(upload_path: Path, db_path: Path) -> tuple[int, dict]:
    df = load_dataframe(upload_path)
    row_count = write_duckdb(df, db_path)
    return row_count, profile_table(db_path)


@router.post("", response_model=DatasetUploaded, status_code=status.HTTP_201_CREATED)
async def upload_dataset(file: UploadFile, user: CurrentUser, db: DbSession) -> dict:
    settings = get_settings()
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise HTTPException(status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, "Hanya file CSV atau XLSX yang didukung")

    max_bytes = settings.max_upload_mb * 1024 * 1024
    dataset_id = uuid.uuid4()
    db_path = Path(settings.data_dir) / "duckdb" / str(user.id) / f"{dataset_id}.duckdb"

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
        try:
            # pandas/DuckDB are blocking; keep the event loop free for other users.
            row_count, profile = await asyncio.to_thread(_ingest, upload_path, db_path)
        except IngestError as exc:
            db_path.unlink(missing_ok=True)
            raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, str(exc)) from exc

    dataset = Dataset(
        id=dataset_id,
        user_id=user.id,
        original_filename=file.filename,
        duckdb_path=str(db_path),
        row_count=row_count,
        schema_json=profile["columns"],
        expires_at=datetime.now(timezone.utc) + timedelta(hours=settings.dataset_ttl_hours),
    )
    db.add(dataset)
    await db.commit()
    await db.refresh(dataset)
    return {**DatasetOut.model_validate(dataset).model_dump(), "profile": profile}


@router.get("", response_model=list[DatasetOut])
async def list_datasets(user: CurrentUser, db: DbSession) -> list[Dataset]:
    rows = await db.scalars(select(Dataset).where(Dataset.user_id == user.id).order_by(Dataset.created_at.desc()))
    return list(rows)


@router.get("/{dataset_id}/profile", response_model=DatasetProfile)
async def get_profile(dataset_id: uuid.UUID, user: CurrentUser, db: DbSession) -> dict:
    dataset = await _get_owned(db, user, dataset_id)
    return await asyncio.to_thread(profile_table, Path(dataset.duckdb_path))


@router.delete("/{dataset_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_dataset(dataset_id: uuid.UUID, user: CurrentUser, db: DbSession) -> None:
    dataset = await _get_owned(db, user, dataset_id)
    Path(dataset.duckdb_path).unlink(missing_ok=True)
    await db.delete(dataset)
    await db.commit()
