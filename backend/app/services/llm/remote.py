import httpx

from app.services.llm.base import LLMError, LLMResult


class RemoteProvider:
    """Any hosted API that speaks the OpenAI chat-completions format (DeepSeek by default).

    Only what the agent already puts in a prompt leaves the server: column names, a few sample rows,
    or the retrieved document passages. Never the uploaded file itself.
    """

    name = "remote"

    def __init__(self, base_url: str, api_key: str, model: str, disable_thinking: bool = True, timeout: float = 90.0):
        self._model = model
        # Reasoning tokens are billed as output and add latency; SQL generation and short grounded
        # answers do not need them. DeepSeek enables thinking unless told otherwise.
        self._extra = {"thinking": {"type": "disabled"}} if disable_thinking else {}
        self._client = httpx.AsyncClient(
            base_url=base_url.rstrip("/"), timeout=timeout, headers={"Authorization": f"Bearer {api_key}"}
        )

    async def complete(self, system: str, messages: list[dict], max_tokens: int = 2048) -> LLMResult:
        payload = {
            "model": self._model,
            "messages": [{"role": "system", "content": system}, *messages],
            "max_tokens": max_tokens,
            "temperature": 0.1,
            **self._extra,
        }
        try:
            response = await self._client.post("/chat/completions", json=payload)
        except httpx.HTTPError as exc:
            raise LLMError(f"API LLM tidak bisa dihubungi: {exc}") from exc

        if response.status_code in (401, 403):
            raise LLMError("API key LLM ditolak")
        if response.status_code == 402:
            raise LLMError("Saldo API LLM habis")
        if response.status_code == 429:
            raise LLMError("API LLM sedang dibatasi (rate limit)")
        if response.status_code >= 400:
            raise LLMError(f"API LLM error {response.status_code}: {response.text[:200]}")

        body = response.json()
        text = (body["choices"][0]["message"].get("content") or "").strip()
        if not text:
            raise LLMError(f"API LLM tidak mengembalikan teks (finish_reason={body['choices'][0].get('finish_reason')})")
        usage = body.get("usage", {})
        return LLMResult(
            text=text,
            provider=self.name,
            model=body.get("model", self._model),
            input_tokens=usage.get("prompt_tokens", 0),
            output_tokens=usage.get("completion_tokens", 0),
        )

    async def aclose(self) -> None:
        await self._client.aclose()
