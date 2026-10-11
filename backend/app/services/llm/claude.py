import anthropic

from app.services.llm.base import LLMError, LLMResult

# Models that accept the server-side `fallbacks: "default"` parameter on the Claude API.
_FALLBACK_MODELS = {"claude-opus-5-5", "claude-opus-5", "claude-sonnet-5-5", "claude-fable-5-1"}
MIN_MAX_TOKENS = 8000


class ClaudeProvider:
    name = "claude"

    def __init__(self, api_key: str, model: str, effort: str | None):
        self._client = anthropic.AsyncAnthropic(api_key=api_key)
        self._model = model
        self._effort = effort

    async def complete(self, system: str, messages: list[dict], max_tokens: int = 16000) -> LLMResult:
        extra: dict = {}
        if self._effort:
            extra["output_config"] = {"effort": self._effort}
        if self._model in _FALLBACK_MODELS:
            # A policy decline is retried server-side on a suitable model instead of failing.
            extra["betas"] = ["server-side-fallback-2026-07-01"]
            extra["fallbacks"] = "default"

        try:
            response = await self._client.beta.messages.create(
                model=self._model,
                # Thinking counts toward max_tokens, so tiny caps meant for the local model
                # (e.g. 16 tokens for routing) would leave no room for the answer.
                # Only generated tokens are billed, so a higher ceiling costs nothing extra.
                max_tokens=max(max_tokens, MIN_MAX_TOKENS),
                system=system,
                messages=messages,
                # The system prompt and dataset schema repeat across turns; cache them.
                cache_control={"type": "ephemeral"},
                **extra,
            )
        except anthropic.RateLimitError as exc:
            raise LLMError("Claude API sedang dibatasi (rate limit)") from exc
        except anthropic.APIStatusError as exc:
            raise LLMError(f"Claude API error {exc.status_code}: {exc.message}") from exc
        except anthropic.APIConnectionError as exc:
            raise LLMError("Claude API tidak bisa dihubungi") from exc

        if response.stop_reason == "refusal":
            raise LLMError("Permintaan ditolak oleh model")
        text = "".join(block.text for block in response.content if block.type == "text")
        if not text.strip():
            raise LLMError(f"Claude tidak mengembalikan teks (stop_reason={response.stop_reason})")

        return LLMResult(
            text=text,
            provider=self.name,
            model=response.model,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
        )

    async def aclose(self) -> None:
        await self._client.close()
