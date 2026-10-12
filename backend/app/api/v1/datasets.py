import asyncio
import uuid
from pathlib import Path

from fastapi import APIRouter, HTTPException, UploadFile, status
from sqlalchemy import select

from app.api.deps import CurrentUser, DbSession
from app.db.models import Dataset
from app.schemas import DatasetOut, DatasetProfile, DatasetUploaded
from app.services.data.profiling import profile_table
from app.services.uploads import remove_files, store_upload

router = APIRouter(prefix="/datasets", tags=["datasets"])


async def _get_owned(db: DbSession, user: CurrentUser, dataset_id: uuid.UUID) -> Dataset:
    dataset = await db.get(Dataset, dataset_id)
    if dataset is None or dataset.user_id != user.id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Dataset tidak ditemukan")
    return dataset


@router.post("", response_model=DatasetUploaded, status_code=status.HTTP_201_CREATED)
async def upload_dataset(file: UploadFile, user: CurrentUser, db: DbSession) -> dict:
    """Upload a file without a chat session. The app itself uploads through /chat/sessions/{id}/files."""
    dataset, profile = await store_upload(db, user.id, file)
    return {**DatasetOut.model_validate(dataset).model_dump(), "profile": profile}


@router.get("", response_model=list[DatasetOut])
async def list_datasets(user: CurrentUser, db: DbSession) -> list[Dataset]:
    rows = await db.scalars(select(Dataset).where(Dataset.user_id == user.id).order_by(Dataset.created_at.desc()))
    return list(rows)


@router.get("/{dataset_id}/profile", response_model=DatasetProfile)
async def get_profile(dataset_id: uuid.UUID, user: CurrentUser, db: DbSession) -> dict:
    dataset = await _get_owned(db, user, dataset_id)
    if dataset.kind != "table":
        return {"row_count": dataset.row_count, "columns": [], "sample": []}
    return await asyncio.to_thread(profile_table, Path(dataset.duckdb_path))


@router.delete("/{dataset_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_dataset(dataset_id: uuid.UUID, user: CurrentUser, db: DbSession) -> None:
    """Remove one file. The chat session it was attached to, and its messages, stay."""
    dataset = await _get_owned(db, user, dataset_id)
    remove_files(dataset)
    await db.delete(dataset)
    await db.commit()
