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


class SessionCreate(BaseModel):
    dataset_id: uuid.UUID


class SessionOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    dataset_id: uuid.UUID
    title: str
    kind: str = "table"
    created_at: datetime


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
