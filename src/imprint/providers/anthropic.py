"""Anthropic LLM provider - hits POST /v1/messages directly via httpx."""

import os
from typing import Any, cast

import httpx

from imprint.llm import LLMResponse

_API_URL = "https://api.anthropic.com/v1/messages"
_API_VERSION = "2023-06-01"
_DEFAULT_MODEL = "claude-haiku-4-5-20251001"


class AnthropicProvider:
    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str = _DEFAULT_MODEL,
        timeout: float = 30.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        resolved_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not resolved_key:
            raise ValueError(
                "Anthropic API key not provided. Pass api_key= or set ANTHROPIC_API_KEY."
            )
        self._api_key = resolved_key
        self.model = model
        self.timeout = timeout
        self._client = client

    async def complete(
        self,
        prompt: str,
        system: str | None = None,
        max_tokens: int = 1000,
        temperature: float = 0.0,
    ) -> LLMResponse:
        body: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
        }
        if system is not None:
            body["system"] = system

        headers = {
            "x-api-key": self._api_key,
            "anthropic-version": _API_VERSION,
            "content-type": "application/json",
        }

        if self._client is not None:
            response = await self._client.post(_API_URL, json=body, headers=headers)
        else:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(_API_URL, json=body, headers=headers)

        response.raise_for_status()
        return _parse_response(response.json())


def _parse_response(data: dict[str, Any]) -> LLMResponse:
    content_raw = data.get("content")
    if not isinstance(content_raw, list):
        raise RuntimeError(f"Unexpected Anthropic response shape: {data!r}")
    content = cast(list[dict[str, Any]], content_raw)

    text_parts: list[str] = [
        block["text"]
        for block in content
        if block.get("type") == "text" and isinstance(block.get("text"), str)
    ]
    if not text_parts:
        raise RuntimeError(f"No text blocks in Anthropic response: {data!r}")

    usage = cast(dict[str, Any], data.get("usage", {}))
    return LLMResponse(
        text="".join(text_parts),
        input_tokens=int(usage.get("input_tokens", 0)),
        output_tokens=int(usage.get("output_tokens", 0)),
    )
