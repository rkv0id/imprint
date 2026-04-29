"""Voyage AI embedder and token counter adapters.

Requires: pip install imprint[voyage]

VoyageEmbedder calls the Voyage API for embeddings (async).
VoyageTokenCounter uses the Voyage tokenizer locally -- no API call,
but downloads the tokenizer from HuggingFace on first use (cached).
"""

from __future__ import annotations

from typing import Any


def _require_voyageai(adapter_name: str) -> Any:
    """Import voyageai or raise a clear error distinguishing missing package
    from missing transitive dependency (e.g. numpy)."""
    try:
        import voyageai  # type: ignore[import-untyped]

        return voyageai
    except ImportError as e:
        missing = getattr(e, "name", None)
        if missing == "voyageai" or missing is None:
            raise ImportError(
                f"voyageai is required for {adapter_name}; "
                "install it with: pip install imprint[voyage]"
            ) from e
        raise ImportError(
            f"{adapter_name} failed to import voyageai: missing transitive "
            f"dependency '{missing}'. Try: pip install imprint[voyage]"
        ) from e


class VoyageEmbedder:
    """Embedder backed by the Voyage AI API.

    Uses AsyncClient for native async embed calls. Default model is
    voyage-3.5-lite at 1024 dimensions -- lightweight, good quality,
    low cost for memory retrieval.

    api_key is optional; if omitted, the client reads VOYAGE_API_KEY
    from the environment.

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
        voyageai = _require_voyageai("VoyageEmbedder")
        kwargs: dict[str, Any] = {}
        if self._api_key is not None:
            kwargs["api_key"] = self._api_key
        self._client = voyageai.AsyncClient(**kwargs)
        return self._client

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


class VoyageTokenCounter:
    """Token counter using the Voyage tokenizer locally.

    No API call -- count_tokens runs the tokenizer in-process. The tokenizer
    is downloaded from HuggingFace on first use and cached locally (small
    file, not model weights).

    The model must match the embedding model in use so token counts reflect
    the same tokenizer. Defaults to voyage-3.5-lite.

    Requires: pip install imprint[voyage]
    """

    def __init__(
        self,
        model: str = "voyage-3.5-lite",
        api_key: str | None = None,
    ) -> None:
        self._model = model
        self._api_key = api_key
        self._client: Any = None

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        voyageai = _require_voyageai("VoyageTokenCounter")
        kwargs: dict[str, Any] = {}
        if self._api_key is not None:
            kwargs["api_key"] = self._api_key
        self._client = voyageai.Client(**kwargs)
        return self._client

    def count(self, text: str) -> int:
        return self._get_client().count_tokens([text], model=self._model)
