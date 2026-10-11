import asyncio
import contextlib
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import APIRouter, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from redis.asyncio import Redis
from sqlalchemy import text

from app.api.v1 import auth, chat, datasets, health
from app.core.config import get_settings
from app.db.models import Base
from app.db.session import engine
from app.ml.models import load_models
from app.services.agent.graph import build_agent
from app.services.data.cleanup import cleanup_loop
from app.services.llm.router import LLMRouter

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    # Tables are created directly for now; switch to Alembic migrations before the schema settles.
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        # create_all never alters an existing table; columns added after the first release go here.
        await conn.execute(text("ALTER TABLE datasets ADD COLUMN IF NOT EXISTS kind VARCHAR(16) NOT NULL DEFAULT 'table'"))

    app.state.redis = Redis.from_url(settings.redis_url, decode_responses=True)
    app.state.llm = LLMRouter(settings)
    app.state.ml = await asyncio.to_thread(load_models, Path(settings.models_dir))
    app.state.agent = build_agent(app.state.llm, settings, app.state.ml)
    cleanup_task = asyncio.create_task(cleanup_loop())
    yield
    cleanup_task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await cleanup_task
    await app.state.llm.aclose()
    await app.state.redis.aclose()
    await engine.dispose()


settings = get_settings()
app = FastAPI(title=settings.app_name, lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[settings.frontend_origin],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

api = APIRouter(prefix="/api/v1")
api.include_router(health.router)
api.include_router(auth.router)
api.include_router(datasets.router)
api.include_router(chat.router)
app.include_router(api)
