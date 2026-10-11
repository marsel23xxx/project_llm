import uuid

from fastapi import APIRouter, HTTPException, Request, Response, status
from sqlalchemy import select

from app.api.deps import ACCESS_COOKIE, REFRESH_COOKIE, CurrentUser, DbSession, RedisClient
from app.core import rate_limit
from app.core.config import get_settings
from app.core.security import create_token, decode_token, hash_password, verify_password
from app.db.models import User
from app.schemas import LoginRequest, RegisterRequest, UserOut

router = APIRouter(prefix="/auth", tags=["auth"])

REFRESH_PATH = "/api/v1/auth"


def _set_auth_cookies(response: Response, user: User) -> None:
    settings = get_settings()
    common = {"httponly": True, "secure": settings.is_production, "samesite": "lax"}
    response.set_cookie(
        ACCESS_COOKIE,
        create_token(str(user.id), "access"),
        max_age=settings.access_token_minutes * 60,
        path="/",
        **common,
    )
    response.set_cookie(
        REFRESH_COOKIE,
        create_token(str(user.id), "refresh"),
        max_age=settings.refresh_token_days * 86400,
        path=REFRESH_PATH,
        **common,
    )


@router.post("/register", response_model=UserOut, status_code=status.HTTP_201_CREATED)
async def register(body: RegisterRequest, db: DbSession) -> User:
    email = body.email.lower()
    if await db.scalar(select(User).where(User.email == email)):
        raise HTTPException(status.HTTP_409_CONFLICT, "Email sudah terdaftar")
    user = User(email=email, password_hash=hash_password(body.password))
    db.add(user)
    await db.commit()
    await db.refresh(user)
    return user


@router.post("/login", response_model=UserOut)
async def login(body: LoginRequest, request: Request, response: Response, db: DbSession, redis: RedisClient) -> User:
    settings = get_settings()
    email = body.email.lower()
    client_ip = request.client.host if request.client else "unknown"
    limit_key = f"login:{client_ip}:{email}"

    if not await rate_limit.hit(redis, limit_key, settings.login_max_attempts, settings.login_window_seconds):
        raise HTTPException(status.HTTP_429_TOO_MANY_REQUESTS, "Terlalu banyak percobaan login, coba lagi nanti")

    user = await db.scalar(select(User).where(User.email == email))
    if user is None or not user.is_active or not verify_password(user.password_hash, body.password):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Email atau password salah")

    await rate_limit.reset(redis, limit_key)
    _set_auth_cookies(response, user)
    return user


@router.post("/refresh", response_model=UserOut)
async def refresh(request: Request, response: Response, db: DbSession) -> User:
    token = request.cookies.get(REFRESH_COOKIE)
    subject = decode_token(token, "refresh") if token else None
    user = await db.get(User, uuid.UUID(subject)) if subject else None
    if user is None or not user.is_active:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Sesi berakhir, silakan login ulang")
    _set_auth_cookies(response, user)
    return user


@router.post("/logout", status_code=status.HTTP_204_NO_CONTENT)
async def logout(response: Response) -> None:
    response.delete_cookie(ACCESS_COOKIE, path="/")
    response.delete_cookie(REFRESH_COOKIE, path=REFRESH_PATH)


@router.get("/me", response_model=UserOut)
async def me(user: CurrentUser) -> User:
    return user
