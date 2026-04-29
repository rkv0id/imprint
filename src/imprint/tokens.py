"""Exact token counting via the Anthropic API.

Requires: pip install imprint[anthropic-tokens]

The Anthropic count_tokens endpoint is free but adds a network round trip
per call. Use this when budget enforcement accuracy matters more than latency.
For most cases the default HeuristicTokenCounter (chars/4) is sufficient.
"""

from __future__ import annotations

from typing import Any


class AnthropicAPITokenCounter:
    """Exact token counter using Anthropic's count_tokens endpoint.

    Each call to count() makes one synchronous API request. The endpoint is
    free but rate-limited separately from message creation. Prefer
    HeuristicTokenCounter for high-frequency budget checks.

    The model argument controls which tokenizer is applied. Defaults to
    claude-haiku-4-5-20251001 (fast, cheap, same tokenizer family as other
    Claude models).

    Requires: pip install imprint[anthropic-tokens]
    """

    def __init__(
        self,
        model: str = "claude-haiku-4-5-20251001",
        api_key: str | None = None,
    ) -> None:
        self._model = model
        self._api_key = api_key
        self._client: Any = None

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        try:
            import anthropic  # type: ignore[import-untyped]
        except ImportError as e:
            missing = getattr(e, "name", None)
            if missing == "anthropic" or missing is None:
                raise ImportError(
                    "anthropic is required for AnthropicAPITokenCounter; "
                    "install it with: pip install imprint[anthropic-tokens]"
                ) from e
            raise ImportError(
                f"AnthropicAPITokenCounter failed to import anthropic: missing "
                f"transitive dependency '{missing}'. "
                "Try: pip install imprint[anthropic-tokens]"
            ) from e
        kwargs: dict[str, Any] = {}
        if self._api_key is not None:
            kwargs["api_key"] = self._api_key
        self._client = anthropic.Anthropic(**kwargs)
        return self._client

    def count(self, text: str) -> int:
        result = self._get_client().messages.count_tokens(
            model=self._model,
            messages=[{"role": "user", "content": text}],
        )
        return int(result.input_tokens)
