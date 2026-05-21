"""MCP tool handler implementations.

Each function is a pure async handler with typed inputs and a dict return.
They are called by the FastMCP tool decorators in server.py, but are
testable directly without any MCP transport machinery.

User identity is resolved per-request via _mcp_user_id, a ContextVar set
by the MCPUserMiddleware in server.py before each tool call is dispatched.
When auth is disabled, the middleware reads IMPRINT_MCP_USER_ID from config.
When auth is enabled, it reads user_id from the API key presented by the client.
Tests that call handlers directly must set _mcp_user_id explicitly.

Error handling: raise ValueError with a descriptive message. FastMCP
converts this to an MCP error response.
"""

from __future__ import annotations

from contextvars import ContextVar
from typing import TYPE_CHECKING, Any

from imprint_server._pool import get_pg_pool

if TYPE_CHECKING:
    from imprint_server.config import ServerConfig
    from imprint_server.registry import AgentRegistry

# Per-request user identity. Set by MCPUserMiddleware before tool dispatch.
# Default is empty string (unset). Handlers raise if they see an empty value.
_mcp_user_id: ContextVar[str] = ContextVar("mcp_user_id", default="")


def _require_mcp_ids(config: ServerConfig) -> tuple[str, str]:
    """Return (agent_id, user_id) or raise with a clear message if either is missing."""
    if not config.mcp_agent_id:
        raise ValueError(
            "IMPRINT_MCP_AGENT_ID is not set. "
            "Configure it in the server environment to use MCP tools."
        )
    user_id = _mcp_user_id.get()
    if not user_id:
        raise ValueError(
            "No user identity resolved for this MCP connection. "
            "When auth is disabled, set IMPRINT_MCP_USER_ID. "
            "When auth is enabled, use a key created with: imprint-server keys create --user <id>."
        )
    return config.mcp_agent_id, user_id


# -- begin_session ------------------------------------------------------------


async def handle_begin_session(
    config: ServerConfig,
    registry: AgentRegistry,
    context: str | None = None,
) -> dict[str, str]:
    """Open a new MemoryLoop session. Returns {session_id}."""
    agent_id, user_id = _require_mcp_ids(config)
    await registry.get(agent_id)  # ensure initialized

    if config.is_postgres:
        from imprint_server.stores.sessions import pg_create_session

        session_id = await pg_create_session(
            get_pg_pool(registry),
            agent_id=agent_id,
            user_id=user_id,
            context=context,
            ttl=config.session_ttl,
        )
    else:
        from imprint_server.stores.sessions import create_session

        session_id = await create_session(
            config, agent_id=agent_id, user_id=user_id, context=context
        )

    return {"session_id": session_id}


# -- get_policy ---------------------------------------------------------------


async def handle_get_policy(
    config: ServerConfig,
    registry: AgentRegistry,
    session_id: str | None = None,
    context: str | None = None,
    scopes: list[str] | None = None,
) -> dict[str, Any]:
    """Compile and return a behavioral policy. Returns {policy_text, memory_count}."""
    agent_id, user_id = _require_mcp_ids(config)
    imp = await registry.get(agent_id)

    if session_id is not None:
        from imprint import MemoryLoop

        from imprint_server.stores.policy_events import log_policy_event
        from imprint_server.stores.sessions import (
            get_session,
            pg_get_session,
            pg_update_session_policy,
            update_session_policy,
        )

        pool = get_pg_pool(registry) if config.is_postgres else None
        if pool is not None:
            session = await pg_get_session(pool, session_id)
        else:
            session = await get_session(config, session_id)

        if session is None or session.agent_id != agent_id or session.closed_at is not None:
            raise ValueError(f"Session {session_id!r} not found or already closed.")

        effective_context = context or session.context
        loop = MemoryLoop(user_id=user_id, session_id=session_id, imprint=imp)
        pol = await imp.get_policy(
            user_id=user_id,
            context=effective_context,
            scopes=scopes,
            loop=loop,
        )

        retrieved_ids = list(loop.retrieved_ids)
        alpha_used = loop.alpha_used

        if pool is not None:
            await pg_update_session_policy(
                pool,
                session_id,
                retrieved_ids=retrieved_ids,
                alpha_used=alpha_used,
                context=effective_context,
            )
        else:
            await update_session_policy(
                config,
                session_id,
                retrieved_ids=retrieved_ids,
                alpha_used=alpha_used,
                context=effective_context,
            )

        await log_policy_event(
            registry=registry,
            config=config,
            agent_id=agent_id,
            user_id=user_id,
            session_id=session_id,
            retrieved_memories=pol.memories,
            filtered_memories=pol.dropped_memories,
            alpha_used=alpha_used,
            context=effective_context,
        )
    else:
        from imprint_server.stores.policy_events import log_policy_event

        pol = await imp.get_policy(user_id=user_id, context=context, scopes=scopes)
        await log_policy_event(
            registry=registry,
            config=config,
            agent_id=agent_id,
            user_id=user_id,
            session_id=None,
            retrieved_memories=pol.memories,
            filtered_memories=pol.dropped_memories,
            alpha_used=0.0,
            context=context,
        )

    return {
        "policy_text": pol.text,
        "memory_count": len(pol.memories),
        "dropped_count": len(pol.dropped_memories),
        "compiled_at": pol.compiled_at.isoformat(),
    }


# -- observe ------------------------------------------------------------------


async def handle_observe(
    config: ServerConfig,
    registry: AgentRegistry,
    agent_output: str,
    user_response: str,
    session_id: str | None = None,
    scope: str | None = None,
) -> dict[str, bool]:
    """Record a turn. Returns {ok: true}."""
    agent_id, user_id = _require_mcp_ids(config)
    imp = await registry.get(agent_id)

    effective_context: str | None = None
    if session_id is not None:
        if config.is_postgres:
            from imprint_server.stores.sessions import pg_get_session

            session = await pg_get_session(get_pg_pool(registry), session_id)
        else:
            from imprint_server.stores.sessions import get_session

            session = await get_session(config, session_id)

        if session is None or session.agent_id != agent_id or session.closed_at is not None:
            raise ValueError(f"Session {session_id!r} not found or already closed.")
        effective_context = session.context

    async with await registry.get_op_lock(agent_id):
        await imp.observe(
            user_id=user_id,
            agent_output=agent_output,
            user_response=user_response,
            context=effective_context,
            scope=scope,
        )

    return {"ok": True}


# -- recall -------------------------------------------------------------------


async def handle_recall(
    config: ServerConfig,
    registry: AgentRegistry,
    query: str,
    scope: str | None = None,
    limit: int = 10,
) -> dict[str, list[dict[str, Any]]]:
    """Semantic search over memories. Returns {memories: [...]}."""
    agent_id, user_id = _require_mcp_ids(config)
    imp = await registry.get(agent_id)

    memories = await imp.search_memories(user_id, query, scope=scope)
    limited = memories[:limit]

    return {
        "memories": [
            {
                "id": m.id,
                "content": m.content,
                "type": m.type.value,
                "scope": m.scope,
                "recall_count": m.recall_count,
            }
            for m in limited
        ]
    }


# -- direct -------------------------------------------------------------------


async def handle_direct(
    config: ServerConfig,
    registry: AgentRegistry,
    instruction: str,
    session_id: str | None = None,
    scope: str | None = None,
) -> dict[str, int]:
    """Store an explicit behavioral direction. Returns {stored: N}."""
    agent_id, user_id = _require_mcp_ids(config)
    imp = await registry.get(agent_id)

    effective_context: str | None = None
    if session_id is not None:
        if config.is_postgres:
            from imprint_server.stores.sessions import pg_get_session

            session = await pg_get_session(get_pg_pool(registry), session_id)
        else:
            from imprint_server.stores.sessions import get_session

            session = await get_session(config, session_id)

        if session is None or session.agent_id != agent_id or session.closed_at is not None:
            raise ValueError(f"Session {session_id!r} not found or already closed.")
        effective_context = session.context

    async with await registry.get_op_lock(agent_id):
        stored = await imp.observe_directions(
            user_id=user_id,
            directions=[instruction],
            context=effective_context,
            scope=scope,
        )

    return {"stored": len(stored)}


# -- end_session --------------------------------------------------------------


async def handle_end_session(
    config: ServerConfig,
    registry: AgentRegistry,
    session_id: str,
    outcome: float | None = None,
    correction: str | None = None,
) -> dict[str, bool]:
    """Close a session and apply the learning signal. Returns {ok: true}."""
    agent_id, user_id = _require_mcp_ids(config)
    imp = await registry.get(agent_id)

    from imprint_server.stores.sessions import (
        close_session,
        get_session,
        pg_close_session,
        pg_get_session,
    )

    pg_pool_end = get_pg_pool(registry) if config.is_postgres else None
    if pg_pool_end is not None:
        session = await pg_get_session(pg_pool_end, session_id)
    else:
        session = await get_session(config, session_id)

    if session is None or session.agent_id != agent_id:
        raise ValueError(f"Session {session_id!r} not found.")
    if session.closed_at is not None:
        raise ValueError(f"Session {session_id!r} is already closed.")

    # Reconstruct MemoryLoop for finalize_loop().
    from imprint import MemoryLoop

    retrieved_memories = []
    if session.retrieved_ids:
        all_mems = await imp._store.list_memories(  # type: ignore[attr-defined]
            agent_id, user_id, active_only=False
        )
        id_set = set(session.retrieved_ids)
        retrieved_memories = [m for m in all_mems if m.id in id_set]

    loop = MemoryLoop(user_id=user_id, session_id=session_id, imprint=imp)
    loop.retrieved_ids = set(session.retrieved_ids)
    loop.retrieved_memories = retrieved_memories
    loop.alpha_used = session.alpha_used
    loop.context = session.context
    loop.closed = True

    if outcome is not None:
        loop.set_outcome(outcome, correction=correction)
    elif correction is not None:
        loop.correction = correction

    await imp.finalize_loop(loop)

    if pg_pool_end is not None:
        await pg_close_session(pg_pool_end, session_id, outcome=outcome, correction=correction)
    else:
        await close_session(config, session_id, outcome=outcome, correction=correction)

    return {"ok": True}


# -- correct ------------------------------------------------------------------


async def handle_correct(
    config: ServerConfig,
    registry: AgentRegistry,
    content: str,
    session_id: str | None = None,
) -> dict[str, Any]:
    """Store a user correction as a memory and apply a negative learning signal.

    Always stores the correction text. When session_id is provided, finalizes
    the session with outcome=-1.0 and uses the correction as attribution hint.
    Returns {ok, memory_id}.
    """
    from imprint.types import MemorySource

    agent_id, user_id = _require_mcp_ids(config)
    imp = await registry.get(agent_id)

    async with await registry.get_op_lock(agent_id):
        stored = await imp.observe_directions(
            user_id=user_id,
            directions=[content],
            source=MemorySource.DETECTED,
        )

    memory_id = stored[0].id if stored else None

    if session_id is not None:
        from imprint import MemoryLoop

        from imprint_server.stores.sessions import (
            close_session,
            get_session,
            pg_close_session,
            pg_get_session,
        )

        pool = get_pg_pool(registry) if config.is_postgres else None
        session = (
            await pg_get_session(pool, session_id)
            if pool is not None
            else await get_session(config, session_id)
        )
        if session is None or session.agent_id != agent_id:
            raise ValueError(f"Session {session_id!r} not found.")
        if session.closed_at is not None:
            raise ValueError(f"Session {session_id!r} is already closed.")

        all_mems = await imp._store.list_memories(  # type: ignore[attr-defined]
            agent_id, user_id, active_only=False
        )
        id_set = set(session.retrieved_ids)
        retrieved_memories = [m for m in all_mems if m.id in id_set]

        loop = MemoryLoop(user_id=user_id, session_id=session_id, imprint=imp)
        loop.retrieved_ids = id_set
        loop.retrieved_memories = retrieved_memories
        loop.alpha_used = session.alpha_used
        loop.context = session.context
        loop.closed = True
        loop.set_outcome(-1.0, correction=content)
        await imp.finalize_loop(loop)

        if pool is not None:
            await pg_close_session(pool, session_id, outcome=-1.0, correction=content)
        else:
            await close_session(config, session_id, outcome=-1.0, correction=content)

    return {"ok": True, "memory_id": memory_id}


# -- reinforce ----------------------------------------------------------------


async def handle_reinforce(
    config: ServerConfig,
    registry: AgentRegistry,
    session_id: str | None = None,
) -> dict[str, Any]:
    """Apply a positive learning signal for a session.

    Finalizes the session with outcome=0.8. No-op when session_id is None.
    Returns {ok, applied}.
    """
    if session_id is None:
        return {"ok": True, "applied": False}

    agent_id, user_id = _require_mcp_ids(config)
    imp = await registry.get(agent_id)

    from imprint import MemoryLoop

    from imprint_server.stores.sessions import (
        close_session,
        get_session,
        pg_close_session,
        pg_get_session,
    )

    pool = get_pg_pool(registry) if config.is_postgres else None
    session = (
        await pg_get_session(pool, session_id)
        if pool is not None
        else await get_session(config, session_id)
    )
    if session is None or session.agent_id != agent_id:
        raise ValueError(f"Session {session_id!r} not found.")
    if session.closed_at is not None:
        raise ValueError(f"Session {session_id!r} is already closed.")

    all_mems = await imp._store.list_memories(  # type: ignore[attr-defined]
        agent_id, user_id, active_only=False
    )
    id_set = set(session.retrieved_ids)
    retrieved_memories = [m for m in all_mems if m.id in id_set]

    loop = MemoryLoop(user_id=user_id, session_id=session_id, imprint=imp)
    loop.retrieved_ids = id_set
    loop.retrieved_memories = retrieved_memories
    loop.alpha_used = session.alpha_used
    loop.context = session.context
    loop.closed = True
    loop.set_outcome(0.8)
    await imp.finalize_loop(loop)

    if pool is not None:
        await pg_close_session(pool, session_id, outcome=0.8, correction=None)
    else:
        await close_session(config, session_id, outcome=0.8, correction=None)

    return {"ok": True, "applied": True}
