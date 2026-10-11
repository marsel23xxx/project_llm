from dataclasses import dataclass
from typing import Protocol


@dataclass
class LLMResult:
    text: str
    provider: str
    model: str
    input_tokens: int
    output_tokens: int


class LLMError(Exception):
    pass


class LLMProvider(Protocol):
    name: str

    async def complete(self, system: str, messages: list[dict], max_tokens: int) -> LLMResult: ...
