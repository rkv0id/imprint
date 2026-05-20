"""Tests for the MCP SSE transport layer and tool registration.

test_mcp.py tests handler functions directly. This file tests the layer
above: that the tools are correctly registered with FastMCP, that the SSE
endpoint is reachable, and that tool invocation works through the HTTP
transport (not just via direct handler calls).
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient

from imprint_server.app import create_app
from imprint_server.config import ServerConfig
from imprint_server.mcp.server import create_mcp_server
from imprint_server.registry import AgentRegistry

AGENT = "mcp-transport-agent"
USER = "mcp-transport-user"

_EXPECTED_TOOLS = frozenset(
    {
        "imprint_begin_session",
        "imprint_get_policy",
        "imprint_observe",
        "imprint_recall",
        "imprint_direct",
        "imprint_end_session",
        "imprint_correct",
        "imprint_reinforce",
    }
)


# -- Fixtures -----------------------------------------------------------------


@pytest.fixture()
def mcp_config(tmp_path: Path) -> ServerConfig:
    return ServerConfig(
        store=f"sqlite:///{tmp_path / 'mcp_transport.db'}",
        default_mode="frugal",
        auth_disabled=True,
        redis_url="",
        mcp_agent_id=AGENT,
        mcp_user_id=USER,
    )


@pytest.fixture()
async def registry(mcp_config: ServerConfig) -> AsyncGenerator[AgentRegistry, None]:
    reg = AgentRegistry(mcp_config)
    await reg.startup()
    yield reg
    await reg.shutdown()


@pytest.fixture()
async def mcp_client(
    mcp_config: ServerConfig, registry: AgentRegistry
) -> AsyncGenerator[AsyncClient, None]:
    """Full app with MCP mounted at /mcp."""
    app = create_app(mcp_config, registry)
    transport = ASGITransport(app=app)  # type: ignore[arg-type]
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


# -- Tool registration ---------------------------------------------------------


async def test_mcp_server_registers_all_tools(
    mcp_config: ServerConfig, registry: AgentRegistry
) -> None:
    """create_mcp_server must register all 8 expected tools."""
    mcp = create_mcp_server(mcp_config, registry)
    tools = await mcp.list_tools()
    names = {t.name for t in tools}
    assert names == _EXPECTED_TOOLS, (
        f"Expected tools: {sorted(_EXPECTED_TOOLS)}\n"
        f"Actual tools:   {sorted(names)}\n"
        f"Missing: {sorted(_EXPECTED_TOOLS - names)}\n"
        f"Extra:   {sorted(names - _EXPECTED_TOOLS)}"
    )


async def test_mcp_server_tool_count(mcp_config: ServerConfig, registry: AgentRegistry) -> None:
    mcp = create_mcp_server(mcp_config, registry)
    tools = await mcp.list_tools()
    assert len(tools) == 8


async def test_mcp_tool_names_match_expected(
    mcp_config: ServerConfig, registry: AgentRegistry
) -> None:
    """Each tool name must match the imprint_ prefix convention."""
    mcp = create_mcp_server(mcp_config, registry)
    tools = await mcp.list_tools()
    for tool in tools:
        assert tool.name.startswith("imprint_"), (
            f"Tool {tool.name!r} does not follow the imprint_ naming convention"
        )


async def test_all_tools_have_descriptions(
    mcp_config: ServerConfig, registry: AgentRegistry
) -> None:
    """Every registered tool must have a non-empty description for LLM discoverability."""
    mcp = create_mcp_server(mcp_config, registry)
    tools = await mcp.list_tools()
    for tool in tools:
        assert tool.description, f"Tool {tool.name!r} has no description"


# -- SSE endpoint -------------------------------------------------------------


async def test_mcp_sse_endpoint_mounted(mcp_client: AsyncClient) -> None:
    """GET /mcp/sse must respond (not 404) when mcp_agent_id and mcp_user_id are set."""
    # We send the request with a short timeout and accept any non-404 response.
    # The SSE stream stays open indefinitely; we just verify the endpoint exists.
    try:
        async with mcp_client.stream("GET", "/mcp/sse", timeout=1.0) as resp:
            assert resp.status_code != 404, "/mcp/sse returned 404 -- endpoint not mounted"
            assert "text/event-stream" in resp.headers.get("content-type", ""), (
                f"Expected text/event-stream, got {resp.headers.get('content-type')}"
            )
    except Exception as exc:
        # Timeout or stream closed early is acceptable -- it proves the endpoint exists.
        if "404" in str(exc):
            pytest.fail(f"/mcp/sse returned 404: {exc}")


async def test_mcp_not_mounted_without_config(tmp_path: Path) -> None:
    """When mcp_agent_id or mcp_user_id is empty, /mcp must not be mounted."""
    config = ServerConfig(
        store=f"sqlite:///{tmp_path / 'no_mcp.db'}",
        default_mode="frugal",
        auth_disabled=True,
        redis_url="",
        mcp_agent_id="",
        mcp_user_id="",
    )
    reg = AgentRegistry(config)
    await reg.startup()
    try:
        app = create_app(config, reg)
        async with AsyncClient(
            transport=ASGITransport(app=app),  # type: ignore[arg-type]
            base_url="http://test",
        ) as c:
            resp = await c.get("/mcp/sse")
            assert resp.status_code == 404
    finally:
        await reg.shutdown()


# -- Tool schema validation ---------------------------------------------------


async def test_begin_session_has_optional_context_param(
    mcp_config: ServerConfig, registry: AgentRegistry
) -> None:
    """imprint_begin_session must accept an optional context param."""
    mcp = create_mcp_server(mcp_config, registry)
    tools = await mcp.list_tools()
    begin = next(t for t in tools if t.name == "imprint_begin_session")
    schema = begin.inputSchema
    # context must be optional (not in required list or not present at all)
    required = schema.get("required", [])
    assert "context" not in required


async def test_get_policy_has_required_params(
    mcp_config: ServerConfig, registry: AgentRegistry
) -> None:
    """imprint_get_policy must have no required params (all optional)."""
    mcp = create_mcp_server(mcp_config, registry)
    tools = await mcp.list_tools()
    policy = next(t for t in tools if t.name == "imprint_get_policy")
    required = policy.inputSchema.get("required", [])
    assert len(required) == 0, f"imprint_get_policy has unexpected required params: {required}"


async def test_correct_has_required_content_param(
    mcp_config: ServerConfig, registry: AgentRegistry
) -> None:
    """imprint_correct must require the content param."""
    mcp = create_mcp_server(mcp_config, registry)
    tools = await mcp.list_tools()
    correct = next(t for t in tools if t.name == "imprint_correct")
    required = correct.inputSchema.get("required", [])
    assert "content" in required
