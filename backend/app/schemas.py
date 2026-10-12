import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict, EmailStr, Field


class RegisterRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8, max_length=128)


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class UserOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    email: EmailStr
    role: str
    created_at: datetime


class ColumnProfile(BaseModel):
    name: str
    type: str
    null_pct: float
    approx_unique: int | str | None
    min: str | int | float | None
    max: str | int | float | None
    mean: str | int | float | None


class DatasetProfile(BaseModel):
    row_count: int
    columns: list[ColumnProfile]
    sample: list[dict]


class DatasetOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    original_filename: str
    kind: str
    row_count: int
    created_at: datetime
    expires_at: datetime


class DatasetUploaded(DatasetOut):
    profile: DatasetProfile


class FileOut(BaseModel):
    """A file attached to a chat session."""

    id: uuid.UUID
    name: str
    kind: str  # "table" or "document"
    source: str  # "csv", "xlsx", "pdf" or "image"
    size: int  # rows for a table, pages for a document
    ocr_pages: int
    expires_at: datetime


class SessionOut(BaseModel):
    id: uuid.UUID
    title: str
    created_at: datetime
    files: list[FileOut] = []


class MessageIn(BaseModel):
    content: str = Field(min_length=1, max_length=2000)


class MessageOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    role: str
    content: str
    steps_json: list
    latency_ms: int | None
    created_at: datetime


class SessionDetail(SessionOut):
    messages: list[MessageOut]
