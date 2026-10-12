import logging
from typing import Literal

from app.core.config import Settings
from app.services.llm.base import LLMError, LLMProvider, LLMResult
from app.services.llm.claude import ClaudeProvider
from app.services.llm.local import LocalProvider
from app.services.llm.remote import RemoteProvider

logger = logging.getLogger(__name__)

TaskKind = Literal["simple", "complex"]


class LLMRouter:
    """Pick a provider per task based on LLM_MODE, falling back to the other one on failure.

    "API" below is the hosted model that is configured: Claude if ANTHROPIC_API_KEY is set,
    otherwise the OpenAI-compatible provider (DeepSeek by default) if REMOTE_LLM_API_KEY is set.

    local  -> local only (no API cost)
    hybrid -> simple tasks: local then API; complex tasks: API then local
    api    -> API then local
    """

    def __init__(self, settings: Settings):
        self.mode = settings.llm_mode
        self.local = LocalProvider(settings.local_llm_url, settings.local_llm_model)
        self.claude: ClaudeProvider | None = None
        if settings.anthropic_api_key:
            self.claude = ClaudeProvider(settings.anthropic_api_key, settings.claude_model, settings.claude_effort)
        self.remote: RemoteProvider | None = None
        if settings.remote_llm_api_key:
            self.remote = RemoteProvider(
                settings.remote_llm_base_url,
                settings.remote_llm_api_key,
                settings.remote_llm_model,
                settings.remote_llm_disable_thinking,
            )
        self.hosted: LLMProvider | None = self.claude or self.remote
        self.hosted_model = (
            settings.claude_model if self.claude else settings.remote_llm_model if self.remote else None
        )
        if self.hosted is None and self.mode != "local":
            logger.warning("LLM_MODE=%s tetapi tidak ada API key; hanya LLM lokal yang dipakai", self.mode)

    @property
    def has_strong_model(self) -> bool:
        """True when a hosted model is configured and allowed, i.e. "complex" tasks reach it first."""
        return self.mode != "local" and self.hosted is not None

    def providers_for(self, kind: TaskKind) -> list[LLMProvider]:
        if self.mode == "local" or self.hosted is None:
            return [self.local]
        if self.mode == "hybrid" and kind == "simple":
            return [self.local, self.hosted]
        return [self.hosted, self.local]

    async def complete(self, kind: TaskKind, system: str, messages: list[dict], max_tokens: int = 4096) -> LLMResult:
        last_error: LLMError | None = None
        for provider in self.providers_for(kind):
            try:
                return await provider.complete(system, messages, max_tokens)
            except LLMError as exc:
                logger.warning("Provider %s gagal: %s", provider.name, exc)
                last_error = exc
        raise last_error or LLMError("Tidak ada provider LLM yang tersedia")

    async def aclose(self) -> None:
        await self.local.aclose()
        if self.claude:
            await self.claude.aclose()
        if self.remote:
            await self.remote.aclose()
