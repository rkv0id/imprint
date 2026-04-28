import json
import os
from typing import Any

import httpx
import pytest

from imprint.providers import AnthropicProvider


def _make_mock_client(
    *,
    response_body: dict[str, Any] | None = None,
    status_code: int = 200,
) -> tuple[httpx.AsyncClient, list[httpx.Request]]:
    captured: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        body: dict[str, Any] = response_body or {
            "id": "msg_test",
            "type": "message",
            "role": "assistant",
            "content": [{"type": "text", "text": "hello back"}],
            "model": "claude-haiku-4-5-20251001",
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": {"input_tokens": 11, "output_tokens": 22},
        }
        return httpx.Response(status_code, json=body)

    return httpx.AsyncClient(transport=httpx.MockTransport(handler)), captured


# ---------- construction ----------------------------------------------------


def test_missing_api_key_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
        AnthropicProvider()


async def test_explicit_api_key_overrides_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "from_env")
    client, captured = _make_mock_client()
    provider = AnthropicProvider(api_key="explicit", client=client)

    await provider.complete("hi")

    assert captured[0].headers["x-api-key"] == "explicit"


async def test_env_api_key_used_when_no_explicit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "from_env")
    client, captured = _make_mock_client()
    provider = AnthropicProvider(client=client)

    await provider.complete("hi")

    assert captured[0].headers["x-api-key"] == "from_env"


# ---------- request shape ---------------------------------------------------


async def test_request_includes_required_headers_and_body() -> None:
    client, captured = _make_mock_client()
    provider = AnthropicProvider(api_key="test_key", client=client)

    await provider.complete("hello", system="be terse", max_tokens=42, temperature=0.5)

    assert len(captured) == 1
    req = captured[0]
    assert str(req.url) == "https://api.anthropic.com/v1/messages"
    assert req.headers["x-api-key"] == "test_key"
    assert req.headers["anthropic-version"] == "2023-06-01"

    body = json.loads(req.content)
    assert body["model"] == "claude-haiku-4-5-20251001"
    assert body["max_tokens"] == 42
    assert body["temperature"] == 0.5
    assert body["system"] == "be terse"
    assert body["messages"] == [{"role": "user", "content": "hello"}]


async def test_system_omitted_when_not_provided() -> None:
    client, captured = _make_mock_client()
    provider = AnthropicProvider(api_key="k", client=client)

    await provider.complete("hi")

    body = json.loads(captured[0].content)
    assert "system" not in body


async def test_custom_model_is_used() -> None:
    client, captured = _make_mock_client()
    provider = AnthropicProvider(api_key="k", model="claude-opus-4-7", client=client)

    await provider.complete("hi")

    body = json.loads(captured[0].content)
    assert body["model"] == "claude-opus-4-7"


# ---------- response parsing ------------------------------------------------


async def test_response_parsed_into_llm_response() -> None:
    client, _ = _make_mock_client()
    provider = AnthropicProvider(api_key="k", client=client)

    response = await provider.complete("hi")

    assert response.text == "hello back"
    assert response.input_tokens == 11
    assert response.output_tokens == 22


async def test_concatenates_multiple_text_blocks() -> None:
    body: dict[str, Any] = {
        "content": [
            {"type": "text", "text": "first "},
            {"type": "text", "text": "second"},
        ],
        "usage": {"input_tokens": 1, "output_tokens": 2},
    }
    client, _ = _make_mock_client(response_body=body)
    provider = AnthropicProvider(api_key="k", client=client)

    response = await provider.complete("hi")
    assert response.text == "first second"


async def test_skips_non_text_blocks() -> None:
    body: dict[str, Any] = {
        "content": [
            {"type": "tool_use", "id": "t1", "name": "x", "input": {}},
            {"type": "text", "text": "the real text"},
        ],
        "usage": {"input_tokens": 1, "output_tokens": 2},
    }
    client, _ = _make_mock_client(response_body=body)
    provider = AnthropicProvider(api_key="k", client=client)

    response = await provider.complete("hi")
    assert response.text == "the real text"


async def test_response_with_no_text_blocks_raises() -> None:
    body: dict[str, Any] = {
        "content": [{"type": "tool_use", "id": "t1", "name": "x", "input": {}}],
        "usage": {"input_tokens": 1, "output_tokens": 2},
    }
    client, _ = _make_mock_client(response_body=body)
    provider = AnthropicProvider(api_key="k", client=client)

    with pytest.raises(RuntimeError, match="No text blocks"):
        await provider.complete("hi")


async def test_malformed_response_raises() -> None:
    client, _ = _make_mock_client(response_body={"error": "nope"})
    provider = AnthropicProvider(api_key="k", client=client)

    with pytest.raises(RuntimeError, match="Unexpected"):
        await provider.complete("hi")


# ---------- HTTP errors -----------------------------------------------------


async def test_http_error_raises() -> None:
    client, _ = _make_mock_client(
        response_body={"error": {"type": "invalid_request"}}, status_code=400
    )
    provider = AnthropicProvider(api_key="k", client=client)

    with pytest.raises(httpx.HTTPStatusError):
        await provider.complete("hi")


# ---------- live --------------------------------------------------------------


@pytest.mark.live
async def test_anthropic_live_smoke() -> None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        pytest.skip("ANTHROPIC_API_KEY not set")

    provider = AnthropicProvider()
    response = await provider.complete("Reply with the single word: pong", max_tokens=10)

    assert response.text.strip()
    assert response.input_tokens > 0
    assert response.output_tokens > 0
