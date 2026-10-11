import uuid
from typing import Annotated

from fastapi import Depends, HTTPException, Request, status
from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.security import decode_token
from app.db.models import User
from app.db.session import get_db

ACCESS_COOKIE = "access_token"
REFRESH_COOKIE = "refresh_token"

DbSession = Annotated[AsyncSession, Depends(get_db)]


def get_redis(request: Request) -> Redis:
    return request.app.state.redis


RedisClient = Annotated[Redis, Depends(get_redis)]


async def get_current_user(request: Request, db: DbSession) -> User:
    token = request.cookies.get(ACCESS_COOKIE)
    auth_header = request.headers.get("Authorization", "")
    if not token and auth_header.startswith("Bearer "):
        token = auth_header.removeprefix("Bearer ")

    subject = decode_token(token, "access") if token else None
    if subject is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Silakan login terlebih dahulu")

    user = await db.get(User, uuid.UUID(subject))
    if user is None or not user.is_active:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Akun tidak aktif")
    return user


CurrentUser = Annotated[User, Depends(get_current_user)]
