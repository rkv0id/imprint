"""FastMCP server for imprint-server.

Creates an MCP server with eight tools and returns it as an ASGI application
that can be mounted in the FastAPI application at /mcp.

Multi-user MCP: user identity is resolved per-connection from the Bearer
token's API key. The key must have a user_id set (created with
`imprint-server keys create --user <id>`). When auth is disabled, user
identity falls back to IMPRINT_MCP_USER_ID for local development.

The agent is still pre-scoped via IMPRINT_MCP_AGENT_ID. MCP clients do not
pass or manage agent identity.

Eight tools:
  imprint_begin_session   -- open a MemoryLoop session
  imprint_get_policy      -- compile and return a behavioral policy
  imprint_observe         -- record an agent-user exchange
  imprint_recall          -- semantic search over memories
  imprint_direct          -- store an explicit behavioral direction
  imprint_end_session     -- close a session and apply learning signal
  imprint_correct         -- store a correction and apply negative signal
  imprint_reinforce       -- apply a positive learning signal
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from mcp.server.fastmcp import FastMCP

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
from imprint_server.stores.api_keys import lookup_api_key

if TYPE_CHECKING:
    from imprint_server.config import ServerConfig
    from imprint_server.registry import AgentRegistry


# -- User identity middleware -------------------------------------------------


class _MCPUserMiddleware:
    """Resolve MCP user identity and set the per-request ContextVar.

    Wraps the FastMCP Starlette app. Runs on every HTTP request to /mcp/*
    (both the SSE connection and tool-call POSTs). Sets _mcp_user_id before
    the request reaches FastMCP so tool handlers can read it via .get().

    Auth disabled: reads IMPRINT_MCP_USER_ID from config.
    Auth enabled:  looks up the Bearer token key and reads key.user_id.
                   Master keys (no user_id) leave _mcp_user_id unset; the
                   tool handler raises with a clear error.
    """

    def __init__(
        self,
        app: object,
        config: ServerConfig,
        registry: AgentRegistry,
    ) -> None:
        self._app = app
        self._config = config
        self._registry = registry

    async def __call__(
        self,
        scope: dict[str, Any],
        receive: Any,
        send: Any,
    ) -> None:
        if scope["type"] == "http":
            await self._set_user_id(scope)
        await self._app(scope, receive, send)  # type: ignore[arg-type]

    async def _set_user_id(self, scope: dict[str, Any]) -> None:
        if self._config.auth_disabled:
            if self._config.mcp_user_id:
                _mcp_user_id.set(self._config.mcp_user_id)
            return
        headers: dict[bytes, bytes] = dict(scope.get("headers", []))
        auth = headers.get(b"authorization", b"").decode()
        if not auth.startswith("Bearer "):
            return
        raw_key = auth[len("Bearer ") :]
        row = await lookup_api_key(self._config, self._registry, raw_key)
        if row is not None and row.user_id:
            _mcp_user_id.set(row.user_id)


# -- MCP server factory -------------------------------------------------------


def create_mcp_server(config: ServerConfig, registry: AgentRegistry) -> FastMCP:
    """Create and configure the imprint FastMCP server.

    Tool handlers close over config and registry. The returned FastMCP
    instance is mounted as a Starlette sub-application in app.py.
    """
    mcp = FastMCP(
        "imprint",
        instructions=(
            "imprint gives you persistent memory across conversations. "
            "Call imprint_get_policy at the start of each conversation to load "
            "your behavioral instructions. Call imprint_observe after each turn "
            "to record what was said. Use imprint_begin_session and "
            "imprint_end_session to track multi-turn loops for learning feedback."
        ),
    )

    @mcp.tool()
    async def imprint_begin_session(context: str | None = None) -> dict[str, Any]:
        """Open a new memory session.

        Returns a session_id that can be passed to other tools. Sessions
        persist the retrieval state needed to apply a learning signal when
        the session ends. Use when you want the agent to learn from whether
        its memory-informed responses were helpful.

        Args:
            context: Optional context string describing the current task or
                     conversation topic. Helps scope memory retrieval.
        """
        return await handle_begin_session(config, registry, context=context)

    @mcp.tool()
    async def imprint_get_policy(
        session_id: str | None = None,
        context: str | None = None,
        scopes: list[str] | None = None,
    ) -> dict[str, Any]:
        """Compile and return a behavioral policy for the current user.

        Returns a policy_text string containing behavioral instructions
        derived from accumulated memories. Inject this into the agent's
        system prompt at the start of each conversation.

        Args:
            session_id: Optional session ID from imprint_begin_session.
                        When provided, retrieval state is tracked for learning.
            context:    Optional context string to scope retrieval.
            scopes:     Optional list of scope names to restrict retrieval to.
        """
        return await handle_get_policy(
            config, registry, session_id=session_id, context=context, scopes=scopes
        )

    @mcp.tool()
    async def imprint_observe(
        agent_output: str,
        user_response: str,
        session_id: str | None = None,
        scope: str | None = None,
    ) -> dict[str, Any]:
        """Record an agent-user exchange and extract memories from it.

        Call after each conversation turn. imprint analyzes the exchange for
        signals (preferences, corrections, decisions) and stores any discovered
        memories for future policy compilation.

        Args:
            agent_output:  The agent's most recent message or response.
            user_response: The user's reply or reaction.
            session_id:    Optional session ID to associate with the exchange.
            scope:         Optional scope name to categorize any extracted memory.
        """
        return await handle_observe(
            config,
            registry,
            agent_output=agent_output,
            user_response=user_response,
            session_id=session_id,
            scope=scope,
        )

    @mcp.tool()
    async def imprint_recall(
        query: str,
        scope: str | None = None,
        limit: int = 10,
    ) -> dict[str, Any]:
        """Search memories semantically. Returns structured memory objects.

        Faster and cheaper than imprint_get_policy for targeted lookups.
        Use when you want to check a specific fact without compiling a full policy.

        Args:
            query: Natural language search query.
            scope: Optional scope name to restrict the search to.
            limit: Maximum number of memories to return (default 10).
        """
        return await handle_recall(config, registry, query=query, scope=scope, limit=limit)

    @mcp.tool()
    async def imprint_direct(
        instruction: str,
        session_id: str | None = None,
        scope: str | None = None,
    ) -> dict[str, Any]:
        """Store an explicit behavioral instruction as a memory.

        Bypasses signal detection -- the instruction is stored directly
        without analyzing conversation context. Use for "always do X" or
        "never do Y" instructions that should be remembered permanently.

        Args:
            instruction: The behavioral instruction to store.
            session_id:  Optional session ID to associate with the memory.
            scope:       Optional scope name to categorize the instruction.
        """
        return await handle_direct(
            config, registry, instruction=instruction, session_id=session_id, scope=scope
        )

    @mcp.tool()
    async def imprint_end_session(
        session_id: str,
        outcome: float | None = None,
        correction: str | None = None,
    ) -> dict[str, Any]:
        """Close a session and apply a learning signal.

        Call at the end of a conversation to give imprint feedback on whether
        the session went well. This updates the memory decay model and retrieval
        weights based on the outcome.

        Args:
            session_id: Session ID from imprint_begin_session.
            outcome:    Optional score between 0.0 (bad) and 1.0 (good) indicating
                        how well the memory-informed responses performed.
            correction: Optional free-text description of what went wrong,
                        used to attribute which memories contributed to errors.
        """
        return await handle_end_session(
            config,
            registry,
            session_id=session_id,
            outcome=outcome,
            correction=correction,
        )

    @mcp.tool()
    async def imprint_correct(
        content: str,
        session_id: str | None = None,
    ) -> dict[str, Any]:
        """Signal that the user corrected the agent and store the correction.

        Stores the correction as a memory so the agent avoids the same mistake
        in future sessions. When session_id is provided, also finalizes the
        session with a negative learning signal so the memory retrieval weights
        are updated.

        Args:
            content:    The correction or feedback from the user (e.g. "Don't
                        use bullet points -- I prefer prose").
            session_id: Optional session ID from imprint_begin_session.
        """
        return await handle_correct(config, registry, content=content, session_id=session_id)

    @mcp.tool()
    async def imprint_reinforce(
        session_id: str | None = None,
    ) -> dict[str, Any]:
        """Signal that the session went well and reinforce the retrieved memories.

        Finalizes the session with a positive learning signal so the memories
        that were retrieved and used in this session get higher stability and
        retrieval weight. No-op when no session_id is provided.

        Args:
            session_id: Session ID from imprint_begin_session.
        """
        return await handle_reinforce(config, registry, session_id=session_id)

    return mcp


def create_mcp_starlette_app(config: ServerConfig, registry: AgentRegistry) -> _MCPUserMiddleware:
    """Create the MCP ASGI app for mounting at /mcp in FastAPI.

    Wraps the FastMCP Starlette app with _MCPUserMiddleware, which resolves
    user identity from the Bearer token on every request and sets the
    _mcp_user_id ContextVar so tool handlers can read it.
    """
    mcp_server = create_mcp_server(config, registry)
    return _MCPUserMiddleware(mcp_server.sse_app(), config, registry)
