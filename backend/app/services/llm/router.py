import logging
from typing import Literal

from app.core.config import Settings
from app.services.llm.base import LLMError, LLMProvider, LLMResult
from app.services.llm.claude import ClaudeProvider
from app.services.llm.local import LocalProvider

logger = logging.getLogger(__name__)

TaskKind = Literal["simple", "complex"]


class LLMRouter:
    """Pick a provider per task based on LLM_MODE, falling back to the other one on failure.

    local  -> local only (no API cost)
    hybrid -> simple tasks: local then Claude; complex tasks: Claude then local
    api    -> Claude then local
    """

    def __init__(self, settings: Settings):
        self.mode = settings.llm_mode
        self.local = LocalProvider(settings.local_llm_url, settings.local_llm_model)
        self.claude: ClaudeProvider | None = None
        if settings.anthropic_api_key:
            self.claude = ClaudeProvider(settings.anthropic_api_key, settings.claude_model, settings.claude_effort)
        elif self.mode != "local":
            logger.warning("LLM_MODE=%s tetapi ANTHROPIC_API_KEY kosong; hanya LLM lokal yang dipakai", self.mode)

    @property
    def has_strong_model(self) -> bool:
        """True when Claude is configured and allowed, i.e. "complex" tasks reach it first."""
        return self.mode != "local" and self.claude is not None

    def providers_for(self, kind: TaskKind) -> list[LLMProvider]:
        if self.mode == "local" or self.claude is None:
            return [self.local]
        if self.mode == "hybrid" and kind == "simple":
            return [self.local, self.claude]
        return [self.claude, self.local]

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
