"""Tests for MCP tool handlers.

Tests call the handler functions in mcp/tools.py directly with a real
SQLite registry in frugal mode. No MCP transport machinery needed --
the handlers are plain async functions.

User identity in production is resolved by _MCPUserMiddleware and stored
in the _mcp_user_id ContextVar. Tests that call handlers directly must
set _mcp_user_id explicitly (done by the setup fixture). Tests for the
"no identity" error path deliberately skip the fixture to leave the
ContextVar at its default empty string.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from pathlib import Path

import pytest

from imprint_server.config import ServerConfig
from imprint_server.mcp.tools import (
    _mcp_user_id,
    handle_begin_session,
    handle_correct,
    handle_direct,
    handle_end_session,
    handle_get_policy,
    handle_observe,
    handle_recall,
    handle_reinforce,
)
from imprint_server.registry import AgentRegistry

AGENT = "mcp-test-agent"
USER = "mcp-test-user"


# -- Fixtures -----------------------------------------------------------------


@pytest.fixture()
async def setup(tmp_path: Path) -> AsyncGenerator[tuple[ServerConfig, AgentRegistry], None]:
    config = ServerConfig(
        store=f"sqlite:///{tmp_path / 'mcp_test.db'}",
        default_mode="frugal",
        auth_disabled=True,
        mcp_agent_id=AGENT,
        mcp_user_id=USER,
    )
    registry = AgentRegistry(config)
    await registry.startup()
    # Simulate what MCPUserMiddleware does on each request: set the user ContextVar.
    _mcp_user_id.set(USER)
    yield config, registry
    await registry.shutdown()


# -- Config validation --------------------------------------------------------


async def test_missing_mcp_agent_id_raises(tmp_path: Path) -> None:
    config = ServerConfig(
        store=f"sqlite:///{tmp_path / 'no_agent.db'}",
        mcp_agent_id="",
        mcp_user_id=USER,
    )
    registry = AgentRegistry(config)
    await registry.startup()
    _mcp_user_id.set(USER)
    try:
        with pytest.raises(ValueError, match="IMPRINT_MCP_AGENT_ID"):
            await handle_get_policy(config, registry)
    finally:
        await registry.shutdown()


async def test_unresolved_user_id_raises(tmp_path: Path) -> None:
    """When the ContextVar holds no user_id, handlers raise with a clear message."""
    config = ServerConfig(
        store=f"sqlite:///{tmp_path / 'no_user.db'}",
        mcp_agent_id=AGENT,
        mcp_user_id="",
    )
    registry = AgentRegistry(config)
    await registry.startup()
    # Deliberately do NOT set _mcp_user_id -- simulates missing middleware or
    # a key with no user_id when auth is enabled.
    _mcp_user_id.set("")
    try:
        with pytest.raises(ValueError, match="No user identity"):
            await handle_get_policy(config, registry)
    finally:
        await registry.shutdown()


async def test_user_id_resolved_from_context_var(tmp_path: Path) -> None:
    """Handlers use the ContextVar value, not config.mcp_user_id, to determine namespace."""
    other_user = "other-user"
    config = ServerConfig(
        store=f"sqlite:///{tmp_path / 'ctx_test.db'}",
        default_mode="frugal",
        mcp_agent_id=AGENT,
        mcp_user_id=USER,  # config says USER
    )
    registry = AgentRegistry(config)
    await registry.startup()
    try:
        # Store a direction under other_user via the ContextVar.
        _mcp_user_id.set(other_user)
        await handle_direct(config, registry, instruction="prefer short responses")

        # Recall under other_user should find the memory.
        result_other = await handle_recall(config, registry, query="prefer short responses")
        assert len(result_other["memories"]) == 1

        # Switch ContextVar to USER -- different namespace, no memories there.
        _mcp_user_id.set(USER)
        result_user = await handle_recall(config, registry, query="prefer short responses")
        assert len(result_user["memories"]) == 0
    finally:
        await registry.shutdown()


# -- begin_session ------------------------------------------------------------


async def test_begin_session_returns_session_id(
    setup: tuple[ServerConfig, AgentRegistry],
) -> None:
    config, registry = setup
    result = await handle_begin_session(config, registry)
    assert "session_id" in result
    assert result["session_id"].startswith("sess_")


async def test_begin_session_with_context(
    setup: tuple[ServerConfig, AgentRegistry],
) -> None:
    config, registry = setup
    result = await handle_begin_session(config, registry, context="code review")
    assert result["session_id"].startswith("sess_")


async def test_begin_session_ids_are_unique(
    setup: tuple[ServerConfig, AgentRegistry],
) -> None:
    config, registry = setup
    r1 = await handle_begin_session(config, registry)
    r2 = await handle_begin_session(config, registry)
    assert r1["session_id"] != r2["session_id"]


# -- get_policy ---------------------------------------------------------------


async def test_get_policy_returns_empty_with_no_memories(
    setup: tuple[ServerConfig, AgentRegistry],
) -> None:
    config, registry = setup
    result = await handle_get_policy(config, registry)
    assert "policy_text" in result
    assert result["memory_count"] == 0
    assert result["policy_text"] == ""


async def test_get_policy_with_session(
    setup: tuple[ServerConfig, AgentRegistry],
) -> None:
    config, registry = setup
    session = await handle_begin_session(config, registry, context="testing")
    result = await handle_get_policy(config, registry, session_id=session["session_id"])
    assert "policy_text" in result
    assert "memory_count" in result


async def test_get_policy_invalid_session_raises(
    setup: tuple[ServerConfig, AgentRegistry],
) -> None:
    config, registry = setup
    with pytest.raises(ValueError, match="not found"):
        await handle_get_policy(config, registry, session_id="sess_doesnotexist")


async def test_get_policy_logs_policy_event(
    setup: tuple[ServerConfig, AgentRegistry], tmp_path: Path
) -> None:
    config, registry = setup
    await handle_get_policy(config, registry, context="testing")
    import aiosqlite

    db_path = str(tmp_path / "mcp_test.db")
    async with (
        aiosqlite.connect(db_path) as conn,
        conn.execute("SELECT COUNT(*) FROM policy_events WHERE agent_id = ?", (AGENT,)) as cursor,
    ):
        row = await cursor.fetchone()
    assert row is not None and row[0] == 1


# -- observe ------------------------------------------------------------------


async def test_observe_returns_ok(
    setup: tuple[ServerConfig, AgentRegistry],
) -> None:
    config, registry = setup
    result = await handle_observe(
        config,
        registry,
        agent_output="Here is a bullet list.",
        user_response="Stop using bullet points.",
    )
    assert result["ok"] is True


async def test_observe_with_session(
    setup: tuple[ServerConfig, AgentRegistry],
) -> None:
    config, registry = setup
    session = await handle_begin_session(config, registry, context="coding")
    result = await handle_observe(
        config,
        registry,
        agent_output="summary",
        user_response="looks good",
        session_id=session["session_id"],
    )
    assert result["ok"] is True


async def test_observe_invalid_session_raises(
    setup: tuple[ServerConfig, AgentRegistry],
) -> None:
    config, registry = setup
    with pytest.raises(ValueError, match="not found"):
        await handle_observe(
            config,
            registry,
            agent_output="x",
            user_response="y",
            session_id="sess_doesnotexist",
        )


# -- recall -------------------------------------------------------------------


async def test_recall_empty_returns_empty_list(
    setup: tuple[ServerConfig, AgentRegistry],
) -> None:
    config, registry = setup
    result = await handle_recall(config, registry, query="anything")
    assert "memories" in result
    assert result["memories"] == []


async def test_recall_result_structure(
    setup: tuple[ServerConfig, AgentRegistry],
) -> None:
    config, registry = setup
    await handle_direct(
        config, registry, instruction="Always write in prose, never use bullet points."
    )
    result = await handle_recall(config, registry, query="bullet points")
    assert "memories" in result
    for m in result["memories"]:
        assert "id" in m
        assert "content" in m
        assert "type" in m
        assert "scope" in m
        assert "recall_count" in m


async def test_recall_respects_limit(
    setup: tuple[ServerConfig, AgentRegistry],
) -> None:
    config, registry = setup
    for i in range(5):
        await handle_direct(
            config, registry, instruction=f"Direction number {i}: always do something."
        )
    result = await handle_recall(config, registry, query="direction", limit=2)
    assert len(result["memories"]) <= 2


# -- direct -------------------------------------------------------------------


async def test_direct_returns_stored_count(
    setup: tuple[ServerConfig, AgentRegistry],
) -> None:
    config, registry = setup
    result = await handle_direct(config, registry, instruction="Always use conventional commits.")
    assert result["stored"] >= 0


async def test_direct_with_session(
    setup: tuple[ServerConfig, AgentRegistry],
) -> None:
    config, registry = setup
    session = await handle_begin_session(config, registry)
    result = await handle_direct(
        config,
        registry,
        instruction="Always write tests.",
        session_id=session["session_id"],
    )
    assert "stored" in result


async def test_direct_invalid_session_raises(
    setup: tuple[ServerConfig, AgentRegistry],
) -> None:
    config, registry = setup
    with pytest.raises(ValueError, match="not found"):
        await handle_direct(
            config,
            registry,
            instruction="some instruction",
            session_id="sess_doesnotexist",
        )


# -- end_session --------------------------------------------------------------


async def test_end_session_returns_ok(
    setup: tuple[ServerConfig, AgentRegistry],
) -> None:
    config, registry = setup
    session = await handle_begin_session(config, registry)
    result = await handle_end_session(config, registry, session_id=session["session_id"])
    assert result["ok"] is True


async def test_end_session_with_outcome(
    setup: tuple[ServerConfig, AgentRegistry],
) -> None:
    config, registry = setup
    session = await handle_begin_session(config, registry)
    result = await handle_end_session(
        config, registry, session_id=session["session_id"], outcome=0.9
    )
    assert result["ok"] is True


async def test_end_session_invalid_raises(
    setup: tuple[ServerConfig, AgentRegistry],
) -> None:
    config, registry = setup
    with pytest.raises(ValueError, match="not found"):
        await handle_end_session(config, registry, session_id="sess_doesnotexist")


async def test_end_session_already_closed_raises(
    setup: tuple[ServerConfig, AgentRegistry],
) -> None:
    config, registry = setup
    session = await handle_begin_session(config, registry)
    await handle_end_session(config, registry, session_id=session["session_id"])
    with pytest.raises(ValueError, match="already closed"):
        await handle_end_session(config, registry, session_id=session["session_id"])


# -- Full lifecycle -----------------------------------------------------------


async def test_full_mcp_lifecycle(
    setup: tuple[ServerConfig, AgentRegistry],
) -> None:
    """begin -> get_policy -> observe -> end without LLM calls (no memories)."""
    config, registry = setup

    session = await handle_begin_session(config, registry, context="code review")
    sid = session["session_id"]

    pol = await handle_get_policy(config, registry, session_id=sid, context="code review")
    assert pol["memory_count"] == 0

    obs = await handle_observe(
        config,
        registry,
        agent_output="Here is the summary.",
        user_response="Looks good.",
        session_id=sid,
    )
    assert obs["ok"] is True

    end = await handle_end_session(config, registry, session_id=sid)
    assert end["ok"] is True


# -- correct ------------------------------------------------------------------


async def test_correct_stores_memory(
    setup: tuple[ServerConfig, AgentRegistry],
) -> None:
    config, registry = setup
    result = await handle_correct(config, registry, content="Do not use passive voice.")
    assert result["ok"] is True
    assert result["memory_id"] is not None


async def test_correct_without_session_does_not_raise(
    setup: tuple[ServerConfig, AgentRegistry],
) -> None:
    config, registry = setup
    result = await handle_correct(
        config, registry, content="Prefer bullet points.", session_id=None
    )
    assert result["ok"] is True


async def test_correct_with_invalid_session_raises(
    setup: tuple[ServerConfig, AgentRegistry],
) -> None:
    config, registry = setup
    with pytest.raises(ValueError, match="not found"):
        await handle_correct(config, registry, content="Fix this.", session_id="sess_bad")


# -- reinforce ----------------------------------------------------------------


async def test_reinforce_without_session_returns_not_applied(
    setup: tuple[ServerConfig, AgentRegistry],
) -> None:
    config, registry = setup
    result = await handle_reinforce(config, registry, session_id=None)
    assert result["ok"] is True
    assert result["applied"] is False


async def test_reinforce_with_invalid_session_raises(
    setup: tuple[ServerConfig, AgentRegistry],
) -> None:
    config, registry = setup
    with pytest.raises(ValueError, match="not found"):
        await handle_reinforce(config, registry, session_id="sess_bad")


async def test_correct_then_reinforce_lifecycle(
    setup: tuple[ServerConfig, AgentRegistry],
) -> None:
    """correct closes session with -1.0; a second reinforce on the same session raises."""
    config, registry = setup

    session = await handle_begin_session(config, registry)
    sid = session["session_id"]

    result = await handle_correct(config, registry, content="Too verbose.", session_id=sid)
    assert result["ok"] is True

    with pytest.raises(ValueError, match="already closed"):
        await handle_reinforce(config, registry, session_id=sid)
