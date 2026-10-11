"""Evaluate document Q&A end to end on the sample PDF: retrieval, answer correctness, latency.

Runs inside the backend container so it uses the serving code and the configured LLM:

    docker compose cp samples/sop_layanan_linen.pdf backend:/tmp/doc.pdf
    docker compose cp ml/doc_eval/questions.json backend:/tmp/questions.json
    docker compose exec -T backend python - < ml/benchmark_documents.py > ml/reports/document_qa.json

An answer counts as correct when it contains every expected fragment. For the two questions the
document cannot answer, correct means the model said so instead of inventing an answer.
"""

import asyncio
import json
import sys
import time
from pathlib import Path

from app.core.config import get_settings
from app.services import documents
from app.services.agent import prompts
from app.services.llm.router import LLMRouter

REFUSAL = "tidak saya temukan"


async def main() -> None:
    settings = get_settings()
    llm = LLMRouter(settings)
    chunks = documents.chunk_pages(documents.extract_pages(Path("/tmp/doc.pdf")))
    retriever = documents.Retriever(chunks)
    questions = json.loads(Path("/tmp/questions.json").read_text())

    rows = []
    for item in questions:
        hits = [c for c, _ in retriever.search(item["q"])]
        request = prompts.document_request(item["q"], [(c.page, c.text) for c in hits])
        start = time.perf_counter()
        out = await llm.complete("complex", prompts.DOC_QA_SYSTEM, [{"role": "user", "content": request}], 600)
        seconds = time.perf_counter() - start
        answer = out.text.strip()
        refused = REFUSAL in answer.lower()
        answerable = item["page"] is not None
        correct = (all(m.lower() in answer.lower() for m in item["must"]) and not refused) if answerable else refused
        rows.append(
            {
                "question": item["q"],
                "answerable": answerable,
                "retrieved_pages": [c.page for c in hits],
                "page_found": (item["page"] in [c.page for c in hits]) if answerable else None,
                "top1_page_correct": (hits[-1].page == item["page"]) if answerable and hits else None,
                "correct": correct,
                "seconds": round(seconds, 1),
                "input_tokens": out.input_tokens,
                "answer": answer,
            }
        )
        print(f"{'OK ' if correct else 'SALAH'} {seconds:5.1f}s  {item['q']}  ->  {answer[:90]}", file=sys.stderr)

    answerable = [r for r in rows if r["answerable"]]
    unanswerable = [r for r in rows if not r["answerable"]]
    report = {
        "document": "samples/sop_layanan_linen.pdf (5 halaman, sintetis)",
        "model": settings.local_llm_model if llm.mode == "local" else settings.claude_model,
        "chunk_words": documents.CHUNK_WORDS,
        "top_k": documents.TOP_K,
        "chunks": len(chunks),
        "retrieval_page_in_top_k": f"{sum(r['page_found'] for r in answerable)}/{len(answerable)}",
        "retrieval_top1": f"{sum(r['top1_page_correct'] for r in answerable)}/{len(answerable)}",
        "answers_correct": f"{sum(r['correct'] for r in answerable)}/{len(answerable)}",
        "unanswerable_refused": f"{sum(r['correct'] for r in unanswerable)}/{len(unanswerable)}",
        "median_seconds": sorted(r["seconds"] for r in rows)[len(rows) // 2],
        "max_seconds": max(r["seconds"] for r in rows),
        "rows": rows,
    }
    await llm.aclose()
    print(json.dumps(report, indent=2, ensure_ascii=False))


asyncio.run(main())
