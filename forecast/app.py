"""Forecast service: receives an already-aggregated numeric series, never the user's raw data."""

import threading

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from forecasting import ForecastError, run_forecast

app = FastAPI(title="Forecast service")
# Training is CPU-bound; one at a time keeps latency predictable on a small VPS.
_slot = threading.BoundedSemaphore(1)


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
    if not _slot.acquire(timeout=60):
        raise HTTPException(503, "Layanan prediksi sedang sibuk, coba lagi")
    try:
        return run_forecast(req.values, req.horizon, req.season, req.use_lstm)
    except ForecastError as exc:
        raise HTTPException(422, str(exc)) from exc
    finally:
        _slot.release()
