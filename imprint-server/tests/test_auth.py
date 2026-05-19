"""Tests for API key authentication middleware.

Tests both auth-disabled (default) and auth-enabled modes. Uses a real
SQLite store so key insertion and lookup hit the actual DB.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient

from imprint_server.app import create_app
from imprint_server.config import ServerConfig
from imprint_server.registry import AgentRegistry
from imprint_server.stores.api_keys import (
    generate_raw_key,
    hash_key,
    insert_key,
)

AGENT = "auth-test-agent"
USER = "auth-test-user"


# -- Fixtures -----------------------------------------------------------------


@pytest.fixture()
async def client_no_auth(tmp_path: Path) -> AsyncGenerator[AsyncClient, None]:
    """Auth disabled (default)."""
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


@pytest.fixture()
async def client_with_auth(
    tmp_path: Path,
) -> AsyncGenerator[tuple[AsyncClient, str, str], None]:
    """Auth enabled. Yields (client, master_key, agent_key)."""
    config = ServerConfig(
        store=f"sqlite:///{tmp_path / 'test.db'}",
        default_mode="frugal",
        auth_disabled=False,
    )
    registry = AgentRegistry(config)
    app = create_app(config, registry)
    await registry.startup()

    master_key = generate_raw_key()
    await insert_key(config, raw_key=master_key, label="master")

    agent_key = generate_raw_key()
    await insert_key(config, raw_key=agent_key, agent_id=AGENT, label="agent-scoped")

    transport = ASGITransport(app=app)  # type: ignore[arg-type]
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c, master_key, agent_key
    await registry.shutdown()


# -- Auth disabled (default behavior) ----------------------------------------


async def test_auth_disabled_allows_all_requests(client_no_auth: AsyncClient) -> None:
    resp = await client_no_auth.get(f"/v1/agents/{AGENT}/memories/{USER}")
    assert resp.status_code == 200


async def test_auth_disabled_health_passes(client_no_auth: AsyncClient) -> None:
    resp = await client_no_auth.get("/health")
    assert resp.status_code == 200


# -- Auth enabled: missing header ---------------------------------------------


async def test_missing_auth_header_returns_401(
    client_with_auth: tuple[AsyncClient, str, str],
) -> None:
    client, _, _ = client_with_auth
    resp = await client.get(f"/v1/agents/{AGENT}/memories/{USER}")
    assert resp.status_code == 401


async def test_wrong_scheme_returns_401(
    client_with_auth: tuple[AsyncClient, str, str],
) -> None:
    client, master_key, _ = client_with_auth
    resp = await client.get(
        f"/v1/agents/{AGENT}/memories/{USER}",
        headers={"Authorization": f"Token {master_key}"},
    )
    assert resp.status_code == 401


# -- Auth enabled: health/metrics bypass --------------------------------------


async def test_health_bypasses_auth(
    client_with_auth: tuple[AsyncClient, str, str],
) -> None:
    client, _, _ = client_with_auth
    resp = await client.get("/health")
    assert resp.status_code == 200


async def test_metrics_bypasses_auth(
    client_with_auth: tuple[AsyncClient, str, str],
) -> None:
    client, _, _ = client_with_auth
    resp = await client.get("/metrics")
    assert resp.status_code == 200


# -- Auth enabled: valid master key -------------------------------------------


async def test_master_key_authorizes_any_agent(
    client_with_auth: tuple[AsyncClient, str, str],
) -> None:
    client, master_key, _ = client_with_auth
    resp = await client.get(
        f"/v1/agents/{AGENT}/memories/{USER}",
        headers={"Authorization": f"Bearer {master_key}"},
    )
    assert resp.status_code == 200


async def test_master_key_authorizes_different_agents(
    client_with_auth: tuple[AsyncClient, str, str],
) -> None:
    client, master_key, _ = client_with_auth
    for agent in ["agent-a", "agent-b", "agent-c"]:
        resp = await client.get(
            f"/v1/agents/{agent}/memories/{USER}",
            headers={"Authorization": f"Bearer {master_key}"},
        )
        assert resp.status_code == 200, f"Failed for agent {agent}"


# -- Auth enabled: agent-scoped key -------------------------------------------


async def test_agent_key_authorizes_correct_agent(
    client_with_auth: tuple[AsyncClient, str, str],
) -> None:
    client, _, agent_key = client_with_auth
    resp = await client.get(
        f"/v1/agents/{AGENT}/memories/{USER}",
        headers={"Authorization": f"Bearer {agent_key}"},
    )
    assert resp.status_code == 200


async def test_agent_key_rejects_wrong_agent(
    client_with_auth: tuple[AsyncClient, str, str],
) -> None:
    client, _, agent_key = client_with_auth
    resp = await client.get(
        "/v1/agents/wrong-agent/memories/u",
        headers={"Authorization": f"Bearer {agent_key}"},
    )
    assert resp.status_code == 403


async def test_agent_key_rejects_unscoped_path(
    client_with_auth: tuple[AsyncClient, str, str],
) -> None:
    """An agent-scoped key cannot access /v1/agents (list) -- no agent_id in path."""
    client, _, agent_key = client_with_auth
    resp = await client.get(
        "/v1/agents",
        headers={"Authorization": f"Bearer {agent_key}"},
    )
    # None path agent_id vs scoped key -> 403
    assert resp.status_code == 403


# -- Auth enabled: invalid / expired keys -------------------------------------


async def test_unknown_key_returns_401(
    client_with_auth: tuple[AsyncClient, str, str],
) -> None:
    client, _, _ = client_with_auth
    resp = await client.get(
        f"/v1/agents/{AGENT}/memories/{USER}",
        headers={"Authorization": "Bearer sk-imp-" + "0" * 64},
    )
    assert resp.status_code == 401


async def test_revoked_key_returns_401(
    client_with_auth: tuple[AsyncClient, str, str],
    tmp_path: Path,
) -> None:
    client, _, _ = client_with_auth
    config = ServerConfig(
        store=f"sqlite:///{tmp_path / 'test.db'}",
        auth_disabled=False,
    )
    revoked_key = generate_raw_key()
    await insert_key(config, raw_key=revoked_key, label="revoked")

    from imprint_server.stores.api_keys import revoke_key

    await revoke_key(config, hash_key(revoked_key))

    resp = await client.get(
        f"/v1/agents/{AGENT}/memories/{USER}",
        headers={"Authorization": f"Bearer {revoked_key}"},
    )
    assert resp.status_code == 401


# -- Auto-generate master key -------------------------------------------------


async def test_auto_generate_inserts_key(tmp_path: Path) -> None:
    """On first auth-enabled startup with empty api_keys, a key is auto-generated."""
    from imprint_server.auth import maybe_generate_master_key
    from imprint_server.stores.api_keys import count_active_keys

    config = ServerConfig(
        store=f"sqlite:///{tmp_path / 'autogen.db'}",
        auth_disabled=False,
    )
    registry = AgentRegistry(config)
    await registry.startup()
    try:
        assert await count_active_keys(config) == 0
        await maybe_generate_master_key(config, registry)
        assert await count_active_keys(config) == 1
    finally:
        await registry.shutdown()


async def test_auto_generate_skipped_when_keys_exist(tmp_path: Path) -> None:
    """If keys already exist, auto-generate must not add another one."""
    from imprint_server.stores.api_keys import count_active_keys

    config = ServerConfig(
        store=f"sqlite:///{tmp_path / 'exist.db'}",
        auth_disabled=False,
    )
    registry = AgentRegistry(config)
    await registry.startup()

    existing_key = generate_raw_key()
    await insert_key(config, raw_key=existing_key, label="existing")
    count_before = await count_active_keys(config)

    from imprint_server.auth import maybe_generate_master_key

    await maybe_generate_master_key(config, registry)

    count_after = await count_active_keys(config)
    assert count_after == count_before
    await registry.shutdown()
