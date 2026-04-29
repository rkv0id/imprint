"""Voyage AI embedder adapter.

Requires: pip install imprint[voyage]
"""

from __future__ import annotations

from typing import Any


class VoyageEmbedder:
    """Embedder backed by the Voyage AI API.

    Uses AsyncClient for native async embed calls. The default model is
    voyage-3.5-lite at 1024 dimensions -- a good balance of quality and cost
    for memory retrieval. Pass a different model or dimension to override.

    The api_key argument is optional; if omitted, the client reads
    VOYAGE_API_KEY from the environment.

    Requires: pip install imprint[voyage]
    """

    def __init__(
        self,
        model: str = "voyage-3.5-lite",
        dim: int = 1024,
        api_key: str | None = None,
        input_type: str = "document",
    ) -> None:
        self._model = model
        self._dim = dim
        self._api_key = api_key
        self._input_type = input_type
        self._client: Any = None

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        try:
            import voyageai  # type: ignore[import-untyped]
        except ImportError as e:
            raise ImportError(
                "voyageai is required for VoyageEmbedder; "
                "install it with: pip install imprint[voyage]"
            ) from e
        kwargs: dict[str, Any] = {}
        if self._api_key is not None:
            kwargs["api_key"] = self._api_key
        self._client = voyageai.AsyncClient(**kwargs)  # type: ignore[union-attr]
        return self._client  # type: ignore[return-value]

    @property
    def dim(self) -> int:
        return self._dim

    async def embed(self, text: str) -> list[float]:
        return (await self.embed_batch([text]))[0]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        client = self._get_client()
        result = await client.embed(
            texts,
            model=self._model,
            input_type=self._input_type,
            output_dimension=self._dim,
        )
        return result.embeddings  # type: ignore[no-any-return]
