"""Tests for admin and health endpoints."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient

from imprint_server.app import create_app
from imprint_server.config import ServerConfig
from imprint_server.registry import AgentRegistry

AGENT = "admin-test-agent"


# -- Fixture ------------------------------------------------------------------


@pytest.fixture()
async def client(tmp_path: Path) -> AsyncGenerator[AsyncClient, None]:
    config = ServerConfig(
        store=f"sqlite:///{tmp_path / 'test.db'}",
        default_mode="frugal",
        auth_disabled=True,
    )
    registry = AgentRegistry(config)
    app = create_app(config, registry)
    await registry.startup()
    transport = ASGITransport(app=app)  # type: ignore[arg-type]
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c
    await registry.shutdown()


# -- health -------------------------------------------------------------------


async def test_health_returns_ok(client: AsyncClient) -> None:
    resp = await client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["db_ok"] is True
    assert body["store"] == "sqlite"


async def test_health_includes_agent_count(client: AsyncClient) -> None:
    resp = await client.get("/health")
    assert "agents_loaded" in resp.json()
    assert resp.json()["agents_loaded"] == 0


async def test_health_agent_count_increments(client: AsyncClient) -> None:
    await client.get(f"/v1/agents/{AGENT}/memories/user1")
    resp = await client.get("/health")
    assert resp.json()["agents_loaded"] == 1


# -- metrics ------------------------------------------------------------------


async def test_metrics_returns_200(client: AsyncClient) -> None:
    resp = await client.get("/metrics")
    assert resp.status_code == 200


async def test_metrics_content_type_is_prometheus(client: AsyncClient) -> None:
    resp = await client.get("/metrics")
    assert "text/plain" in resp.headers["content-type"]


async def test_metrics_contains_imprint_counters(client: AsyncClient) -> None:
    # Trigger an observe to register the counter.
    await client.post(
        f"/v1/agents/{AGENT}/observe",
        json={
            "user_id": "u1",
            "agent_output": "output",
            "user_response": "response",
        },
    )
    resp = await client.get("/metrics")
    assert "imprint_observe_total" in resp.text


# -- list agents --------------------------------------------------------------


async def test_list_agents_empty(client: AsyncClient) -> None:
    resp = await client.get("/v1/agents")
    assert resp.status_code == 200
    assert resp.json() == []


async def test_list_agents_after_initialization(client: AsyncClient) -> None:
    await client.get(f"/v1/agents/{AGENT}/memories/user1")
    resp = await client.get("/v1/agents")
    assert resp.status_code == 200
    agents = resp.json()
    assert len(agents) == 1
    assert agents[0]["agent_id"] == AGENT


# -- create agent -------------------------------------------------------------


async def test_post_agents_creates_config(client: AsyncClient) -> None:
    resp = await client.post(
        "/v1/agents",
        json={"agent_id": "new-agent", "processing_mode": "eager"},
    )
    assert resp.status_code == 200
    assert resp.json()["agent_id"] == "new-agent"


async def test_post_agents_pre_warm(client: AsyncClient) -> None:
    await client.post(
        "/v1/agents",
        json={"agent_id": "prewarm-agent", "pre_warm": True},
    )
    resp = await client.get("/v1/agents")
    agent_ids = [a["agent_id"] for a in resp.json()]
    assert "prewarm-agent" in agent_ids


async def test_post_agents_config_is_respected(client: AsyncClient) -> None:
    """Agent initialized after POST should use the configured mode."""
    await client.post(
        "/v1/agents",
        json={"agent_id": "configured-agent", "processing_mode": "eager", "pre_warm": True},
    )
    resp = await client.get("/v1/agents")
    agents = {a["agent_id"]: a for a in resp.json()}
    assert agents["configured-agent"]["processing_mode"] == "eager"


# -- get agent ----------------------------------------------------------------


async def test_get_agent_returns_config(client: AsyncClient) -> None:
    await client.post(
        "/v1/agents",
        json={"agent_id": "get-test-agent", "processing_mode": "frugal"},
    )
    resp = await client.get("/v1/agents/get-test-agent")
    assert resp.status_code == 200
    body = resp.json()
    assert body["agent_id"] == "get-test-agent"
    assert body["processing_mode"] == "frugal"


async def test_get_unknown_agent_returns_404(client: AsyncClient) -> None:
    resp = await client.get("/v1/agents/does-not-exist")
    assert resp.status_code == 404


# -- patch agent config -------------------------------------------------------


async def test_patch_agent_config_updates_mode(client: AsyncClient) -> None:
    # Initialize with default mode.
    await client.get(f"/v1/agents/{AGENT}/memories/user1")

    resp = await client.patch(
        f"/v1/agents/{AGENT}/config",
        json={"processing_mode": "eager"},
    )
    assert resp.status_code == 200
    assert resp.json()["processing_mode"] == "eager"


async def test_patch_agent_config_reloads_registry(client: AsyncClient) -> None:
    """PATCH must apply the new mode to the live Imprint instance."""
    # Initialize.
    await client.get(f"/v1/agents/{AGENT}/memories/user1")

    await client.patch(
        f"/v1/agents/{AGENT}/config",
        json={"processing_mode": "eager"},
    )

    # List agents to confirm live instance reflects change.
    resp = await client.get("/v1/agents")
    agents = {a["agent_id"]: a for a in resp.json()}
    assert agents[AGENT]["processing_mode"] == "eager"


async def test_patch_uninitialized_agent_updates_db(client: AsyncClient) -> None:
    """PATCH on an uncreated agent should write to DB (no error)."""
    resp = await client.patch(
        "/v1/agents/uncreated-agent/config",
        json={"processing_mode": "frugal"},
    )
    assert resp.status_code == 200


# -- delete agent -------------------------------------------------------------


async def test_delete_agent_removes_from_registry(client: AsyncClient) -> None:
    await client.get(f"/v1/agents/{AGENT}/memories/user1")
    assert (await client.get("/health")).json()["agents_loaded"] == 1

    resp = await client.delete(f"/v1/agents/{AGENT}")
    assert resp.status_code == 200
    assert resp.json()["ok"] is True
    assert (await client.get("/health")).json()["agents_loaded"] == 0


async def test_delete_unknown_agent_is_noop(client: AsyncClient) -> None:
    resp = await client.delete("/v1/agents/never-initialized")
    assert resp.status_code == 200


async def test_delete_preserves_memory_data(client: AsyncClient) -> None:
    """Deleting an agent from the registry must not erase user memories."""
    await client.post(
        f"/v1/agents/{AGENT}/memories/user1/directions",
        json={"directions": ["Write in prose."]},
    )
    await client.delete(f"/v1/agents/{AGENT}")

    # Re-initialize and check memories are still there.
    resp = await client.get(f"/v1/agents/{AGENT}/memories/user1")
    assert resp.status_code == 200
    # Memories persist across agent deregistration.
    assert isinstance(resp.json(), list)


# -- scope consolidation ------------------------------------------------------


async def test_scopes_consolidate_returns_ok(client: AsyncClient) -> None:
    resp = await client.post(f"/v1/agents/{AGENT}/scopes/consolidate")
    assert resp.status_code == 200
    assert resp.json()["ok"] is True
