"""Tests for the LLM tool interface."""

import os
from datetime import UTC, datetime
from typing import cast

import pytest
from helpers import _ConstantEmbedder, _InMemoryVectorStore, _make_imprint

from imprint import Imprint, SQLiteMemoryStore, make_pydantic_ai_tools
from imprint.tools import (
    _correct,
    _forget,
    _recall,
    _reinforce,
    _remember,
    _search,
    make_anthropic_tools,
)
from imprint.types import Memory, MemorySource, MemoryType


async def _insert(store: SQLiteMemoryStore, **kwargs: object) -> Memory:
    now = datetime.now(UTC)
    m = Memory(
        id=str(kwargs.get("id", "m1")),
        agent_id=str(kwargs.get("agent_id", "agent")),
        user_id=str(kwargs.get("user_id", "u")),
        type=MemoryType(str(kwargs.get("type", "rule"))),
        scope=str(kwargs.get("scope", "global")),
        content=str(kwargs.get("content", "rule content")),
        source=MemorySource.DETECTED,
        valid_from=now,
        created_at=now,
        updated_at=now,
    )
    await store.insert_memory(m)
    return m


async def test_remember_stores_memory() -> None:
    imprint, _, _, _, _, _, _ = _make_imprint(processing_mode="frugal")
    await imprint.connect()

    mid = await _remember(imprint, "u", "always use English")
    assert mid != ""

    memories = await imprint.list_memories("u")
    assert len(memories) == 1
    assert memories[0].content == "always use English"


async def test_recall_returns_empty_without_memories() -> None:
    imprint, _, _, _, _, _, _ = _make_imprint(processing_mode="frugal", compile_text="ok")
    await imprint.connect()

    text = await _recall(imprint, "u")
    assert text == ""


async def test_recall_returns_compiled_policy() -> None:
    imprint, _, _, _, _, _, _ = _make_imprint(processing_mode="frugal", compile_text="be concise")
    await imprint.connect()

    store = cast(SQLiteMemoryStore, imprint._store)
    await _insert(store)

    text = await _recall(imprint, "u")
    assert text == "be concise"


async def test_search_returns_empty_without_memories() -> None:
    imprint, _, _, _, _, _, _ = _make_imprint(processing_mode="frugal")
    await imprint.connect()

    results = await _search(imprint, "u", "python")
    assert results == []


async def test_search_returns_memories_as_dicts() -> None:
    imprint, _, _, _, _, _, _ = _make_imprint(processing_mode="frugal")
    await imprint.connect()

    store = cast(SQLiteMemoryStore, imprint._store)
    await _insert(store, id="m1", content="use type hints")

    results = await _search(imprint, "u", "python")
    assert len(results) == 1
    assert results[0]["id"] == "m1"
    assert results[0]["content"] == "use type hints"
    assert "type" in results[0]
    assert "scope" in results[0]


async def test_search_uses_vector_store_when_configured() -> None:
    vec_store = _InMemoryVectorStore()
    embedder = _ConstantEmbedder([1.0, 0.0, 0.0])

    imprint, _, _, _, _, _, _ = _make_imprint(processing_mode="frugal")
    imprint._vector_store = vec_store
    imprint._embedder = embedder
    await imprint.connect()

    store = cast(SQLiteMemoryStore, imprint._store)
    await _insert(store, id="m1", content="use type hints")
    await vec_store.upsert("m1", [1.0, 0.0, 0.0])

    results = await _search(imprint, "u", "type hints")
    assert any(r["id"] == "m1" for r in results)


async def test_forget_deactivates_memory() -> None:
    imprint, _, _, _, _, _, _ = _make_imprint(processing_mode="frugal")
    await imprint.connect()

    store = cast(SQLiteMemoryStore, imprint._store)
    await _insert(store, id="m1")

    result = await _forget(imprint, "u", "m1")
    assert result == "ok"

    memories = await imprint.list_memories("u")
    assert len(memories) == 0


async def test_forget_returns_not_found_for_missing_id() -> None:
    imprint, _, _, _, _, _, _ = _make_imprint(processing_mode="frugal")
    await imprint.connect()

    result = await _forget(imprint, "u", "nonexistent")
    assert result == "not_found"


async def test_correct_stores_memory_and_closes_loop() -> None:
    imprint, _, _, _, _, _, _ = _make_imprint(processing_mode="frugal", compile_text="ok")
    await imprint.connect()

    store = cast(SQLiteMemoryStore, imprint._store)
    await _insert(store, id="m1")

    await imprint.get_policy(user_id="u")
    assert "u" in imprint._open_loops

    mid = await _correct(imprint, "u", "never use bullet points")
    assert mid != ""
    assert "u" not in imprint._open_loops

    await imprint.drain()
    memories = await imprint.list_memories("u")
    assert any(m.content == "never use bullet points" for m in memories)


async def test_correct_no_loop_still_stores_memory() -> None:
    imprint, _, _, _, _, _, _ = _make_imprint(processing_mode="frugal")
    await imprint.connect()

    mid = await _correct(imprint, "u", "always be direct")
    assert mid != ""

    memories = await imprint.list_memories("u")
    assert len(memories) == 1


async def test_reinforce_closes_loop_with_positive_signal() -> None:
    from imprint import BanditAlphaTuner

    tuner = BanditAlphaTuner()
    imprint, _, _, _, _, _, _ = _make_imprint(processing_mode="frugal", compile_text="ok")
    imprint._alpha_tuner = tuner
    await imprint.connect()

    store = cast(SQLiteMemoryStore, imprint._store)
    await _insert(store, id="m1")

    await imprint.get_policy(user_id="u")
    initial = sum(tuner.get_state()["s"]) + sum(tuner.get_state()["f"])

    result = await _reinforce(imprint, "u")
    assert result == "ok"
    assert "u" not in imprint._open_loops

    await imprint.drain()
    final = sum(tuner.get_state()["s"]) + sum(tuner.get_state()["f"])
    assert final > initial


async def test_reinforce_no_loop_returns_no_loop() -> None:
    imprint, _, _, _, _, _, _ = _make_imprint(processing_mode="frugal")
    await imprint.connect()

    result = await _reinforce(imprint, "u")
    assert result == "no_loop"


async def test_make_pydantic_ai_tools_returns_six_tools() -> None:
    imprint, _, _, _, _, _, _ = _make_imprint(processing_mode="frugal", compile_text="ok")
    await imprint.connect()

    tools = make_pydantic_ai_tools(imprint, user_id="u")
    assert len(tools) == 6


async def test_make_pydantic_ai_tools_functions_work() -> None:
    imprint, _, _, _, _, _, _ = _make_imprint(processing_mode="frugal", compile_text="ok")
    await imprint.connect()

    tools = make_pydantic_ai_tools(imprint, user_id="u")
    tool_map = {t.name: t for t in tools}

    assert "remember" in tool_map
    assert "recall" in tool_map
    assert "search" in tool_map
    assert "forget" in tool_map
    assert "correct" in tool_map
    assert "reinforce" in tool_map


async def test_make_anthropic_tools_raises_without_dep() -> None:
    import sys
    from unittest.mock import patch

    imprint, _, _, _, _, _, _ = _make_imprint(processing_mode="frugal")
    await imprint.connect()

    with (
        patch.dict(sys.modules, {"anthropic": None}),
        pytest.raises(ImportError, match="anthropic is required"),
    ):
        make_anthropic_tools(imprint, user_id="u")


async def test_make_anthropic_tools_returns_six_defs_and_dispatch() -> None:
    pytest.importorskip("anthropic", reason="imprint-mem[anthropic] not installed")

    imprint, _, _, _, _, _, _ = _make_imprint(processing_mode="frugal")
    await imprint.connect()

    tool_defs, dispatch = make_anthropic_tools(imprint, user_id="u")
    assert len(tool_defs) == 6
    names = {t["name"] for t in tool_defs}
    assert names == {"remember", "recall", "search", "forget", "correct", "reinforce"}
    assert callable(dispatch)


async def test_anthropic_dispatch_unknown_tool_raises() -> None:
    pytest.importorskip("anthropic", reason="imprint-mem[anthropic] not installed")

    imprint, _, _, _, _, _, _ = _make_imprint(processing_mode="frugal")
    await imprint.connect()

    _, dispatch = make_anthropic_tools(imprint, user_id="u")
    with pytest.raises(ValueError, match="unknown imprint tool"):
        await dispatch("nonexistent", {})


async def test_imprint_list_memories_public_method() -> None:
    imprint, _, _, _, _, _, _ = _make_imprint(processing_mode="frugal")
    await imprint.connect()

    store = cast(SQLiteMemoryStore, imprint._store)
    await _insert(store, id="m1", scope="global")
    await _insert(store, id="m2", scope="global")

    memories = await imprint.list_memories("u")
    assert len(memories) == 2


async def test_imprint_deactivate_memory_public_method() -> None:
    imprint, _, _, _, _, _, _ = _make_imprint(processing_mode="frugal")
    await imprint.connect()

    store = cast(SQLiteMemoryStore, imprint._store)
    await _insert(store, id="m1")

    found = await imprint.deactivate_memory("u", "m1")
    assert found is True

    found_again = await imprint.deactivate_memory("u", "m1")
    assert found_again is False


async def test_imprint_close_loop_public_method() -> None:
    imprint, _, _, _, _, _, _ = _make_imprint(processing_mode="frugal", compile_text="ok")
    await imprint.connect()

    store = cast(SQLiteMemoryStore, imprint._store)
    await _insert(store, id="m1")

    await imprint.get_policy(user_id="u")
    assert "u" in imprint._open_loops

    closed = await imprint.close_loop("u", 0.5)
    assert closed is True
    assert "u" not in imprint._open_loops

    closed_again = await imprint.close_loop("u", 0.5)
    assert closed_again is False


@pytest.mark.live
async def test_tools_live_remember_and_recall() -> None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        pytest.skip("ANTHROPIC_API_KEY not set")

    imprint = Imprint(
        agent_id="tools_live",
        store=":memory:",
        processing_mode="frugal",
    )
    await imprint.connect()

    tools = make_pydantic_ai_tools(imprint, user_id="rami")
    tool_map = {t.name: t for t in tools}

    mid = await tool_map["remember"].function("always respond in English")
    assert mid != ""

    policy = await tool_map["recall"].function()
    assert isinstance(policy, str)

    await imprint.close()
