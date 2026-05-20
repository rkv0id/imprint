"""FastMCP server for imprint-server.

Creates an MCP server with eight tools and returns it as a Starlette app
that can be mounted in the FastAPI application at /mcp.

The server is scoped to one agent and one user namespace via
IMPRINT_MCP_AGENT_ID and IMPRINT_MCP_USER_ID. MCP clients (Claude Code,
Cursor, Continue) connect to /mcp/sse and call tools without managing
agent or user identity -- the server is pre-scoped.

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
    handle_begin_session,
    handle_correct,
    handle_direct,
    handle_end_session,
    handle_get_policy,
    handle_observe,
    handle_recall,
    handle_reinforce,
)

if TYPE_CHECKING:
    from starlette.applications import Starlette

    from imprint_server.config import ServerConfig
    from imprint_server.registry import AgentRegistry


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


def create_mcp_starlette_app(config: ServerConfig, registry: AgentRegistry) -> Starlette:
    """Create the MCP Starlette app for mounting at /mcp in FastAPI.

    The returned app handles the SSE transport and tool dispatch.
    Mount with: app.mount("/mcp", create_mcp_starlette_app(config, registry))
    """
    mcp_server = create_mcp_server(config, registry)
    return mcp_server.sse_app()
