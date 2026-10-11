"""Shared helpers for training scripts."""

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import mlflow

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
BACKEND_MODELS = PROJECT / "backend" / "models"
REPORTS = ROOT / "reports"

# Pipelines pickle a reference to app.ml.text.normalize, so training imports it from the backend package.
sys.path.insert(0, str(PROJECT / "backend"))

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
mlflow.set_tracking_uri(f"sqlite:///{ROOT / 'mlflow.db'}")


def write_report(name: str, payload: dict) -> Path:
    REPORTS.mkdir(exist_ok=True)
    path = REPORTS / f"{name}.json"
    payload = {"generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), **payload}
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    return path
