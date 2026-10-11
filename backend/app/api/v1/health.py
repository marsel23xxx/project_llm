from fastapi import APIRouter, Request
from sqlalchemy import text

from app.api.deps import DbSession, RedisClient

router = APIRouter(tags=["health"])


@router.get("/health")
async def health(request: Request, db: DbSession, redis: RedisClient) -> dict:
    checks = {}
    try:
        await db.execute(text("SELECT 1"))
        checks["database"] = "ok"
    except Exception:
        checks["database"] = "error"
    try:
        await redis.ping()
        checks["redis"] = "ok"
    except Exception:
        checks["redis"] = "error"

    llm = request.app.state.llm
    checks["llm_mode"] = llm.mode
    checks["claude_configured"] = llm.claude is not None
    checks["status"] = "ok" if checks["database"] == checks["redis"] == "ok" else "degraded"
    return checks
