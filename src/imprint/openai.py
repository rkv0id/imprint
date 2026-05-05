"""OpenAI embedder and token counter adapters.

Requires: pip install imprint-mem[openai]

OpenAIEmbedder calls the OpenAI embeddings API (async).
OpenAITokenCounter uses tiktoken locally -- no API call, counts tokens for
any OpenAI model without network round-trips.
"""

from __future__ import annotations

import asyncio
from typing import Any

_DIMS: dict[str, int] = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}


def _require_openai(adapter_name: str) -> Any:
    try:
        import openai  # type: ignore[import-untyped]

        return openai
    except ImportError as e:
        raise ImportError(
            f"openai is required for {adapter_name}; "
            "install it with: pip install imprint-mem[openai]"
        ) from e


def _require_tiktoken(adapter_name: str) -> Any:
    try:
        import tiktoken  # type: ignore[import-untyped]

        return tiktoken
    except ImportError as e:
        raise ImportError(
            f"tiktoken is required for {adapter_name}; "
            "install it with: pip install imprint-mem[openai]"
        ) from e


class OpenAIEmbedder:
    """Embedder backed by the OpenAI embeddings API.

    Uses AsyncOpenAI for native async calls. Supports native dimensions
    reduction (text-embedding-3-small and text-embedding-3-large only).

    api_key is optional; if omitted, the client reads OPENAI_API_KEY from
    the environment.

    Supported models and default dimensions:
      text-embedding-3-small  1536  (supports dimensions= reduction)
      text-embedding-3-large  3072  (supports dimensions= reduction)
      text-embedding-ada-002  1536  (fixed, no reduction)

    Requires: pip install imprint-mem[openai]
    """

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        dimensions: int | None = None,
        api_key: str | None = None,
    ) -> None:
        self._model = model
        self._api_key = api_key
        default_dim = _DIMS.get(model, 1536)
        self._dim = dimensions if dimensions is not None else default_dim
        self._client: Any = None

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        openai = _require_openai("OpenAIEmbedder")
        kwargs: dict[str, Any] = {}
        if self._api_key is not None:
            kwargs["api_key"] = self._api_key
        self._client = openai.AsyncOpenAI(**kwargs)
        return self._client

    @property
    def dim(self) -> int:
        return self._dim

    async def embed(self, text: str) -> list[float]:
        return (await self.embed_batch([text]))[0]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        client = self._get_client()
        kwargs: dict[str, Any] = {"input": texts, "model": self._model}
        # dimensions= is only supported by text-embedding-3-* models
        if self._model != "text-embedding-ada-002":
            kwargs["dimensions"] = self._dim
        response = await client.embeddings.create(**kwargs)
        return [item.embedding for item in response.data]


class OpenAITokenCounter:
    """Token counter using tiktoken locally.

    No API call -- tiktoken runs in-process using the model's encoding.
    The encoding data is downloaded once and cached by tiktoken.

    model should match the model whose context window you are targeting.
    Defaults to "gpt-4o" which uses the o200k_base encoding (same as
    gpt-4o-mini, o1, and most current OpenAI models).

    tiktoken.get_encoding runs synchronously. We wrap it in
    run_in_executor so it does not block the event loop on the first
    call (subsequent calls use tiktoken's internal LRU cache and are
    effectively instant).

    Requires: pip install imprint-mem[openai]
    """

    def __init__(self, model: str = "gpt-4o") -> None:
        self._model = model
        self._enc: Any = None

    def _get_encoding(self) -> Any:
        if self._enc is not None:
            return self._enc
        tiktoken = _require_tiktoken("OpenAITokenCounter")
        try:
            self._enc = tiktoken.encoding_for_model(self._model)
        except KeyError:
            self._enc = tiktoken.get_encoding("o200k_base")
        return self._enc

    def count(self, text: str) -> int:
        enc = self._get_encoding()
        return len(enc.encode(text))

    async def count_async(self, text: str) -> int:
        """Count tokens without blocking the event loop on first call."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.count, text)
