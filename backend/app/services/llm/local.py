import httpx

from app.services.llm.base import LLMError, LLMResult


class LocalProvider:
    """llama.cpp server running on the VPS (its /v1/chat/completions endpoint)."""

    name = "local"

    def __init__(self, base_url: str, model: str, timeout: float = 170.0):
        self._model = model
        self._client = httpx.AsyncClient(base_url=base_url, timeout=timeout)

    async def complete(self, system: str, messages: list[dict], max_tokens: int = 2048) -> LLMResult:
        try:
            response = await self._client.post(
                "/v1/chat/completions",
                json={
                    "model": self._model,
                    "messages": [{"role": "system", "content": system}, *messages],
                    "max_tokens": max_tokens,
                    "temperature": 0.1,
                },
            )
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise LLMError(f"LLM lokal tidak bisa dihubungi: {exc}") from exc

        body = response.json()
        usage = body.get("usage", {})
        return LLMResult(
            text=body["choices"][0]["message"]["content"],
            provider=self.name,
            model=self._model,
            input_tokens=usage.get("prompt_tokens", 0),
            output_tokens=usage.get("completion_tokens", 0),
        )

    async def aclose(self) -> None:
        await self._client.aclose()
