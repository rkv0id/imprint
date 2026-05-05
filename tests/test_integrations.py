"""Tests for framework integration adapters.

Both handlers are tested without their respective framework packages
installed by mocking the import. When the frameworks are present the
handler construction, callback wiring, and observe() dispatch are tested
against real handler instances.
"""

import importlib.util
import sys
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from helpers import _make_imprint

# ---------------------------------------------------------------------------
# LangChain
# ---------------------------------------------------------------------------


async def test_langchain_handler_raises_without_dep() -> None:
    from imprint.integrations.langchain import ImprintCallbackHandler

    imprint, _, _, _, _, _, _ = _make_imprint(processing_mode="frugal")
    await imprint.connect()

    with (
        patch.dict(sys.modules, {"langchain_core": None}),
        pytest.raises(ImportError, match="langchain-core is required"),
    ):
        ImprintCallbackHandler(imprint=imprint, user_id="u")


async def test_langchain_handler_construction() -> None:
    if importlib.util.find_spec("langchain_core") is None:
        pytest.skip("langchain-core not installed (pip install imprint[langchain])")

    from imprint.integrations.langchain import ImprintCallbackHandler

    imprint, _, _, _, _, _, _ = _make_imprint(processing_mode="frugal")
    await imprint.connect()

    handler = ImprintCallbackHandler(imprint=imprint, user_id="u")
    assert handler._user_id == "u"
    assert handler._loop is None
    assert handler._context is None
    assert handler._last_generation == ""
    assert handler._chain_input == ""


async def test_langchain_on_chain_start_captures_input() -> None:
    if importlib.util.find_spec("langchain_core") is None:
        pytest.skip("langchain-core not installed (pip install imprint[langchain])")

    from imprint.integrations.langchain import ImprintCallbackHandler

    imprint, _, _, _, _, _, _ = _make_imprint(processing_mode="frugal")
    await imprint.connect()
    handler = ImprintCallbackHandler(imprint=imprint, user_id="u")

    handler.on_chain_start({}, {"input": "hello world"})
    assert handler._chain_input == "hello world"


async def test_langchain_on_chain_start_tries_fallback_keys() -> None:
    if importlib.util.find_spec("langchain_core") is None:
        pytest.skip("langchain-core not installed (pip install imprint[langchain])")

    from imprint.integrations.langchain import ImprintCallbackHandler

    imprint, _, _, _, _, _, _ = _make_imprint(processing_mode="frugal")
    await imprint.connect()
    handler = ImprintCallbackHandler(imprint=imprint, user_id="u")

    handler.on_chain_start({}, {"question": "what is the answer?"})
    assert handler._chain_input == "what is the answer?"


async def test_langchain_on_llm_end_captures_generation() -> None:
    if importlib.util.find_spec("langchain_core") is None:
        pytest.skip("langchain-core not installed (pip install imprint[langchain])")

    from imprint.integrations.langchain import ImprintCallbackHandler

    imprint, _, _, _, _, _, _ = _make_imprint(processing_mode="frugal")
    await imprint.connect()
    handler = ImprintCallbackHandler(imprint=imprint, user_id="u")

    gen = MagicMock()
    gen.text = "the LLM said this"
    response = MagicMock()
    response.generations = [[gen]]
    handler.on_llm_end(response)

    assert handler._last_generation == "the LLM said this"


async def test_langchain_on_agent_finish_fires_observe() -> None:
    if importlib.util.find_spec("langchain_core") is None:
        pytest.skip("langchain-core not installed (pip install imprint[langchain])")

    from imprint.integrations.langchain import ImprintCallbackHandler

    imprint, _, _, _, _, _, _ = _make_imprint(processing_mode="frugal")
    await imprint.connect()
    handler = ImprintCallbackHandler(imprint=imprint, user_id="u")

    observe_called = False
    original_observe = imprint.observe

    async def mock_observe(**kwargs: Any) -> None:
        nonlocal observe_called
        observe_called = True
        assert kwargs["user_id"] == "u"
        assert kwargs["agent_output"] == "the LLM said this"
        assert kwargs["user_response"] == "user input"

    imprint.observe = mock_observe  # type: ignore[method-assign]

    handler._chain_input = "user input"
    handler._last_generation = "the LLM said this"

    finish = MagicMock()
    finish.return_values = {"output": "final answer"}
    handler.on_agent_finish(finish)

    await handler.flush()
    assert observe_called

    imprint.observe = original_observe  # type: ignore[method-assign]


async def test_langchain_on_agent_finish_resets_state() -> None:
    if importlib.util.find_spec("langchain_core") is None:
        pytest.skip("langchain-core not installed (pip install imprint[langchain])")

    from imprint.integrations.langchain import ImprintCallbackHandler

    imprint, _, _, _, _, _, _ = _make_imprint(processing_mode="frugal")
    await imprint.connect()
    handler = ImprintCallbackHandler(imprint=imprint, user_id="u")

    handler._chain_input = "user input"
    handler._last_generation = "llm output"

    finish = MagicMock()
    finish.return_values = {"output": "done"}
    handler.on_agent_finish(finish)
    await handler.flush()

    assert handler._last_generation == ""
    assert handler._chain_input == ""


async def test_langchain_on_agent_finish_noop_when_no_input() -> None:
    if importlib.util.find_spec("langchain_core") is None:
        pytest.skip("langchain-core not installed (pip install imprint[langchain])")

    from imprint.integrations.langchain import ImprintCallbackHandler

    imprint, _, _, _, _, _, _ = _make_imprint(processing_mode="frugal")
    await imprint.connect()

    observe_called = False

    async def mock_observe(**kwargs: Any) -> None:
        nonlocal observe_called
        observe_called = True

    imprint.observe = mock_observe  # type: ignore[method-assign]
    handler = ImprintCallbackHandler(imprint=imprint, user_id="u")

    # No chain input, no generation, and no finish output -- truly no text at all.
    finish = MagicMock()
    finish.return_values = {"output": ""}
    handler.on_agent_finish(finish)
    await handler.flush()

    assert not observe_called


async def test_langchain_handler_accepts_loop() -> None:
    if importlib.util.find_spec("langchain_core") is None:
        pytest.skip("langchain-core not installed (pip install imprint[langchain])")

    from imprint.integrations.langchain import ImprintCallbackHandler

    imprint, _, _, _, _, _, _ = _make_imprint(processing_mode="frugal")
    await imprint.connect()

    loop = await imprint.open_loop(user_id="u")
    handler = ImprintCallbackHandler(imprint=imprint, user_id="u", loop=loop)
    assert handler._loop is loop


# ---------------------------------------------------------------------------
# LlamaIndex
# ---------------------------------------------------------------------------


async def test_llamaindex_handler_raises_without_dep() -> None:
    from imprint.integrations.llamaindex import ImprintEventHandler

    imprint, _, _, _, _, _, _ = _make_imprint(processing_mode="frugal")
    await imprint.connect()

    with (
        patch.dict(
            sys.modules,
            {
                "llama_index": None,
                "llama_index.core": None,
                "llama_index.core.instrumentation": None,
            },
        ),
        pytest.raises(ImportError, match="llama-index-core is required"),
    ):
        ImprintEventHandler(imprint=imprint, user_id="u")


async def test_llamaindex_handler_construction() -> None:
    if importlib.util.find_spec("llama_index") is None:
        pytest.skip("llama-index-core not installed (pip install imprint[llamaindex])")

    from imprint.integrations.llamaindex import ImprintEventHandler

    imprint, _, _, _, _, _, _ = _make_imprint(processing_mode="frugal")
    await imprint.connect()

    handler = ImprintEventHandler(imprint=imprint, user_id="u")
    assert handler._user_id == "u"
    assert handler._pending_query == ""
    assert ImprintEventHandler.class_name() == "ImprintEventHandler"


async def test_llamaindex_handle_query_start_captures_query() -> None:
    if importlib.util.find_spec("llama_index") is None:
        pytest.skip("llama-index-core not installed (pip install imprint[llamaindex])")

    from imprint.integrations.llamaindex import ImprintEventHandler

    imprint, _, _, _, _, _, _ = _make_imprint(processing_mode="frugal")
    await imprint.connect()
    handler = ImprintEventHandler(imprint=imprint, user_id="u")

    # The handler uses type(event).__name__ so we use a real class name.
    class QueryStartEvent:
        query = "what is the capital of France?"

    handler.handle(QueryStartEvent())
    assert handler._pending_query == "what is the capital of France?"


async def test_llamaindex_handle_query_end_fires_observe() -> None:
    if importlib.util.find_spec("llama_index") is None:
        pytest.skip("llama-index-core not installed (pip install imprint[llamaindex])")

    from imprint.integrations.llamaindex import ImprintEventHandler

    imprint, _, _, _, _, _, _ = _make_imprint(processing_mode="frugal")
    await imprint.connect()

    observe_called = False

    async def mock_observe(**kwargs: Any) -> None:
        nonlocal observe_called
        observe_called = True
        assert kwargs["user_id"] == "u"
        assert "Paris" in kwargs["agent_output"]
        assert "capital" in kwargs["user_response"]

    imprint.observe = mock_observe  # type: ignore[method-assign]
    handler = ImprintEventHandler(imprint=imprint, user_id="u")

    class QueryStartEvent:
        query = "what is the capital of France?"

    class _Response:
        response = "The capital of France is Paris."

    class QueryEndEvent:
        response = _Response()

    handler.handle(QueryStartEvent())
    handler.handle(QueryEndEvent())
    await handler.flush()

    assert observe_called


async def test_llamaindex_handler_accepts_loop() -> None:
    if importlib.util.find_spec("llama_index") is None:
        pytest.skip("llama-index-core not installed (pip install imprint[llamaindex])")

    from imprint.integrations.llamaindex import ImprintEventHandler

    imprint, _, _, _, _, _, _ = _make_imprint(processing_mode="frugal")
    await imprint.connect()

    loop = await imprint.open_loop(user_id="u")
    handler = ImprintEventHandler(imprint=imprint, user_id="u", loop=loop)
    assert handler._loop is loop
