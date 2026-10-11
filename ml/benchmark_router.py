"""Compare the trained intent classifier with LLM routing on the hand-written held-out set.

Runs inside the backend container so it reaches the local LLM through the exact serving code:

    docker compose cp ml/intent/heldout.csv backend:/tmp/heldout.csv
    docker compose exec -T backend python - < ml/benchmark_router.py
"""

import asyncio
import csv
import json
import time
from pathlib import Path

from app.core.config import get_settings
from app.ml.models import load_models
from app.services.agent import prompts
from app.services.agent.graph import parse_intent
from app.services.llm.router import LLMRouter

HELDOUT = Path("/tmp/heldout.csv")


async def main() -> None:
    rows = list(csv.DictReader(HELDOUT.open(encoding="utf-8")))
    settings = get_settings()
    models = load_models(Path(settings.models_dir))
    llm = LLMRouter(settings)

    report = {"questions": len(rows)}

    start = time.perf_counter()
    ml_preds = [models.intent.predict(r["text"]) for r in rows]
    ml_ms = (time.perf_counter() - start) * 1000 / len(rows)
    threshold = settings.intent_min_confidence or models.intent.threshold
    confident = [p for p in ml_preds if p.confidence >= threshold]
    report["ml"] = {
        "model": models.intent.name,
        "accuracy": round(sum(p.label == r["label"] for p, r in zip(ml_preds, rows)) / len(rows), 3),
        "ms_per_question": round(ml_ms, 2),
        "threshold": threshold,
        "confident_share": round(len(confident) / len(rows), 3),
    }

    llm_correct, llm_ms, hybrid_correct, llm_calls = 0, [], 0, 0
    for row, ml in zip(rows, ml_preds):
        t = time.perf_counter()
        out = await llm.complete("simple", prompts.ROUTER_SYSTEM, [{"role": "user", "content": row["text"]}], 16)
        llm_ms.append((time.perf_counter() - t) * 1000)
        llm_label = parse_intent(out.text)
        llm_correct += llm_label == row["label"]
        # Production behaviour: trust the classifier when confident, otherwise ask the LLM.
        if ml.confidence >= threshold:
            hybrid_correct += ml.label == row["label"]
        else:
            llm_calls += 1
            hybrid_correct += llm_label == row["label"]

    report["llm"] = {
        "model": settings.local_llm_model if llm.mode == "local" else settings.claude_model,
        "accuracy": round(llm_correct / len(rows), 3),
        "ms_per_question": round(sum(llm_ms) / len(llm_ms), 1),
    }
    report["ml_then_llm"] = {
        "accuracy": round(hybrid_correct / len(rows), 3),
        "llm_calls_saved": f"{len(rows) - llm_calls}/{len(rows)}",
    }
    await llm.aclose()
    print(json.dumps(report, indent=2))


asyncio.run(main())
