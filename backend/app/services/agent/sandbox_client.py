from pathlib import Path

import httpx

from app.services.agent.executor import ExecutionError, QueryResult


async def run_python(sandbox_url: str, db_path: Path, code: str, timeout_seconds: float = 45.0) -> QueryResult:
    try:
        async with httpx.AsyncClient(base_url=sandbox_url, timeout=timeout_seconds) as client:
            response = await client.post("/run", json={"dataset_path": str(db_path), "code": code})
            response.raise_for_status()
    except httpx.HTTPError as exc:
        raise ExecutionError(f"Sandbox Python tidak tersedia: {exc}") from exc

    body = response.json()
    if not body.get("ok"):
        raise ExecutionError(body.get("error", "Kode gagal dijalankan"))

    result = body.get("result") or {}
    stdout = body.get("stdout", "")
    if result.get("type") == "table":
        return QueryResult(
            columns=result["columns"],
            types=result["types"],
            rows=result["rows"],
            truncated=result.get("truncated", False),
            stdout=stdout,
        )
    return QueryResult(columns=[], types=[], rows=[], stdout=stdout, value=result.get("value"))
