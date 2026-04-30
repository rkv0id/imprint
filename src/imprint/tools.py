"""LLM-callable tool interface for imprint.

Exposes six tools the LLM can call to interact with its memory layer:
remember, recall, search, forget, correct, reinforce.

Vendor adapters:
  make_pydantic_ai_tools(imprint, user_id) -- pydantic-ai Tool list (core)
  make_anthropic_tools(imprint, user_id)   -- Anthropic tool defs + dispatch
                                             (requires imprint-mem[anthropic])
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from imprint._core import Imprint


async def _remember(
    imprint: Imprint,
    user_id: str,
    content: str,
    scope: str | None = None,
) -> str:
    """Store something worth remembering about the current user.

    Returns the memory ID of the stored memory, or empty string on failure.
    """
    from imprint.types import MemorySource

    memories = await imprint.observe_directions(
        user_id=user_id,
        directions=[content],
        scope=scope,
        source=MemorySource.DETECTED,
    )
    return memories[0].id if memories else ""


async def _recall(
    imprint: Imprint,
    user_id: str,
    context: str | None = None,
    session_id: str | None = None,
) -> str:
    """Retrieve the compiled behavioral policy for the current user.

    Returns policy text, or empty string if no memories exist.
    """
    policy = await imprint.get_policy(
        user_id=user_id,
        context=context,
        session_id=session_id,
    )
    return policy.text


async def _search(
    imprint: Imprint,
    user_id: str,
    query: str,
    scope: str | None = None,
) -> list[dict[str, str]]:
    """Search memories by semantic query.

    Returns matching memories as dicts with id, type, scope, content fields.
    """
    memories = await imprint.search_memories(user_id, query, scope=scope)
    return [
        {
            "id": m.id,
            "type": m.type.value,
            "scope": m.scope,
            "content": m.content,
        }
        for m in memories
    ]


async def _forget(
    imprint: Imprint,
    user_id: str,
    memory_id: str,
) -> str:
    """Deactivate a specific memory by ID.

    Returns 'ok' on success, 'not_found' if the memory does not exist or
    is already inactive.
    """
    found = await imprint.deactivate_memory(user_id, memory_id)
    return "ok" if found else "not_found"


async def _correct(
    imprint: Imprint,
    user_id: str,
    content: str,
    session_id: str | None = None,
) -> str:
    """Signal that the user just corrected the agent and store the correction.

    Closes any open feedback loop with a negative signal. Returns the memory
    ID of the stored correction, or empty string on failure.
    """
    from imprint.types import MemorySource

    memories = await imprint.observe_directions(
        user_id=user_id,
        directions=[content],
        source=MemorySource.DETECTED,
    )
    await imprint.close_loop(user_id, -1.0, session_id=session_id)
    return memories[0].id if memories else ""


async def _reinforce(
    imprint: Imprint,
    user_id: str,
    session_id: str | None = None,
) -> str:
    """Signal that the interaction went well.

    Returns 'ok' if a loop was found and closed, 'no_loop' otherwise.
    """
    closed = await imprint.close_loop(user_id, 0.8, session_id=session_id)
    return "ok" if closed else "no_loop"


def make_pydantic_ai_tools(
    imprint: Imprint,
    *,
    user_id: str,
    session_id: str | None = None,
) -> list[Any]:
    """Return a list of pydantic-ai Tool objects for the six imprint tools.

    Usage:
        tools = make_pydantic_ai_tools(imprint, user_id="rami")
        agent = Agent(model="...", tools=tools)
    """
    from pydantic_ai import Tool

    async def remember(content: str, scope: str | None = None) -> str:
        """Store something worth remembering about the current user."""
        return await _remember(imprint, user_id, content, scope)

    async def recall(context: str | None = None) -> str:
        """Get the compiled behavioral policy for the current user."""
        return await _recall(imprint, user_id, context, session_id)

    async def search(query: str, scope: str | None = None) -> list[dict[str, str]]:
        """Search memories by semantic query."""
        return await _search(imprint, user_id, query, scope)

    async def forget(memory_id: str) -> str:
        """Deactivate a specific memory by ID."""
        return await _forget(imprint, user_id, memory_id)

    async def correct(content: str) -> str:
        """Signal a user correction and store it as a memory."""
        return await _correct(imprint, user_id, content, session_id)

    async def reinforce() -> str:
        """Signal that the interaction went well."""
        return await _reinforce(imprint, user_id, session_id)

    return [
        Tool(remember),
        Tool(recall),
        Tool(search),
        Tool(forget),
        Tool(correct),
        Tool(reinforce),
    ]


def make_anthropic_tools(
    imprint: Imprint,
    *,
    user_id: str,
    session_id: str | None = None,
) -> tuple[list[dict[str, Any]], Any]:
    """Return Anthropic tool definitions and a dispatch coroutine factory.

    Requires: pip install imprint-mem[anthropic]

    Usage:
        tool_defs, dispatch = make_anthropic_tools(imprint, user_id="rami")
        response = client.messages.create(tools=tool_defs, ...)
        for block in response.content:
            if block.type == "tool_use":
                result = await dispatch(block.name, block.input)
    """
    try:
        import anthropic as _anthropic

        _ = _anthropic  # imported for presence check only
    except ImportError as e:
        raise ImportError(
            "anthropic is required for make_anthropic_tools; "
            "install it with: pip install imprint-mem[anthropic]"
        ) from e

    tool_defs: list[dict[str, Any]] = [
        {
            "name": "remember",
            "description": "Store something worth remembering about the current user.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "content": {"type": "string", "description": "The memory content to store."},
                    "scope": {"type": "string", "description": "Optional scope tag."},
                },
                "required": ["content"],
            },
        },
        {
            "name": "recall",
            "description": "Get the compiled behavioral policy for the current user.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "context": {
                        "type": "string",
                        "description": "Optional context to focus the policy.",
                    },
                },
                "required": [],
            },
        },
        {
            "name": "search",
            "description": "Search memories by semantic query without compiling a full policy.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The search query."},
                    "scope": {"type": "string", "description": "Optional scope filter."},
                },
                "required": ["query"],
            },
        },
        {
            "name": "forget",
            "description": "Deactivate a specific memory by ID.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "memory_id": {"type": "string", "description": "ID of the memory to remove."},
                },
                "required": ["memory_id"],
            },
        },
        {
            "name": "correct",
            "description": "Signal that the user corrected the agent and store the correction.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "The correction content to store.",
                    },
                },
                "required": ["content"],
            },
        },
        {
            "name": "reinforce",
            "description": "Signal that the interaction went well.",
            "input_schema": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    ]

    async def dispatch(tool_name: str, tool_input: dict[str, Any]) -> Any:
        if tool_name == "remember":
            raw_scope = tool_input.get("scope")
            scope_str = str(raw_scope) if raw_scope else None
            return await _remember(imprint, user_id, str(tool_input["content"]), scope_str)
        if tool_name == "recall":
            ctx = tool_input.get("context")
            return await _recall(imprint, user_id, str(ctx) if ctx else None, session_id)
        if tool_name == "search":
            sc = tool_input.get("scope")
            return await _search(
                imprint, user_id, str(tool_input["query"]), str(sc) if sc else None
            )
        if tool_name == "forget":
            return await _forget(imprint, user_id, str(tool_input["memory_id"]))
        if tool_name == "correct":
            return await _correct(imprint, user_id, str(tool_input["content"]), session_id)
        if tool_name == "reinforce":
            return await _reinforce(imprint, user_id, session_id)
        raise ValueError(f"unknown imprint tool: {tool_name!r}")

    return tool_defs, dispatch
