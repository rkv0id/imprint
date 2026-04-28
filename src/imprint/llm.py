"""LLM provider interface used by Imprint internals."""

from dataclasses import dataclass
from typing import Protocol


@dataclass
class LLMResponse:
    text: str
    input_tokens: int
    output_tokens: int


class LLMProvider(Protocol):
    async def complete(
        self,
        prompt: str,
        system: str | None = None,
        max_tokens: int = 1000,
        temperature: float = 0.0,
    ) -> LLMResponse: ...
