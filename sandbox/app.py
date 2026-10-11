"""Runs agent-written Python against one dataset, isolated from the backend.

Isolation layers:
- this container: no internet (internal Docker network), read-only filesystem, non-root,
  all capabilities dropped, memory/CPU/PID limits (docker-compose.yml)
- each run: a fresh subprocess (own session) that sets CPU-time, memory and file-size rlimits on
  itself before importing anything (runner.py), plus a wall-clock timeout here
- the dataset volume is mounted read-only and DuckDB opens it with external access disabled
"""

import json
import subprocess
import sys
import tempfile
import threading
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

RUNNER = Path(__file__).with_name("runner.py")
DATA_ROOT = Path("/data/duckdb")
WALL_TIMEOUT_SECONDS = 30
MAX_CONCURRENT_RUNS = 2

app = FastAPI(title="Python sandbox")
_slots = threading.BoundedSemaphore(MAX_CONCURRENT_RUNS)


class RunRequest(BaseModel):
    dataset_path: str
    code: str = Field(min_length=1, max_length=20_000)


def _dataset_path(raw: str) -> Path:
    path = Path(raw).resolve()
    if not path.is_relative_to(DATA_ROOT) or path.suffix != ".duckdb" or not path.is_file():
        raise HTTPException(400, "Dataset tidak valid")
    return path


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/run")
def run(req: RunRequest) -> dict:
    dataset = _dataset_path(req.dataset_path)
    if not _slots.acquire(timeout=10):
        raise HTTPException(503, "Sandbox sedang sibuk, coba lagi")
    try:
        with tempfile.TemporaryDirectory() as tmp:
            out_path = Path(tmp) / "result.json"
            try:
                proc = subprocess.run(
                    [sys.executable, "-I", str(RUNNER), str(dataset), str(out_path)],
                    input=req.code,
                    capture_output=True,
                    text=True,
                    timeout=WALL_TIMEOUT_SECONDS,
                    cwd=tmp,
                    env={"PATH": "/usr/local/bin:/usr/bin", "HOME": tmp, "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"},
                    start_new_session=True,
                )
            except subprocess.TimeoutExpired:
                return {"ok": False, "error": f"Kode melebihi batas waktu {WALL_TIMEOUT_SECONDS} detik"}

            if not out_path.exists():
                detail = (proc.stderr or "proses berhenti tanpa hasil")[-2000:]
                return {"ok": False, "error": f"Kode berhenti tidak normal (exit {proc.returncode}): {detail}"}
            return json.loads(out_path.read_text())
    finally:
        _slots.release()
