"""Forecast service: receives an already-aggregated numeric series, never the user's raw data."""

import logging
import math
import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from forecasting import ForecastError, run_forecast

logger = logging.getLogger("forecast")
# Training is CPU-bound; one at a time keeps latency predictable on a small VPS.
_slot = threading.BoundedSemaphore(1)


def _warm_up() -> None:
    """Importing TensorFlow and tracing the first model takes about a minute; pay that at boot,
    not on a visitor's first forecast."""
    series = [100 + 10 * math.sin(i / 7 * 2 * math.pi) + i * 0.1 for i in range(120)]
    with _slot:
        try:
            run_forecast(series, horizon=7, season=7)
            logger.warning("Pemanasan TensorFlow selesai")
        except Exception:
            logger.exception("Pemanasan gagal")


@asynccontextmanager
async def lifespan(app: FastAPI):
    threading.Thread(target=_warm_up, daemon=True).start()
    yield


app = FastAPI(title="Forecast service", lifespan=lifespan)


class ForecastRequest(BaseModel):
    values: list[float] = Field(min_length=1, max_length=5000)
    horizon: int = Field(ge=1, le=120)
    season: int = Field(ge=1, le=366)
    use_lstm: bool = True


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/forecast")
def forecast(req: ForecastRequest) -> dict:
    if not _slot.acquire(timeout=120):
        raise HTTPException(503, "Layanan prediksi sedang sibuk, coba lagi")
    try:
        return run_forecast(req.values, req.horizon, req.season, req.use_lstm)
    except ForecastError as exc:
        raise HTTPException(422, str(exc)) from exc
    finally:
        _slot.release()
