"""Postgres integration tests for imprint-server.

Marked @pytest.mark.postgres -- excluded from the default test run.
Requires a running Postgres instance with pgvector.

Run via:
  just server-integration-test        # starts Docker Postgres automatically
  just server-postgres-test           # against an already-running Postgres

Environment:
  IMPRINT_STORE=postgres://user:pass@host/db

Every test fixture truncates all server and library tables so tests are
fully isolated. The schema is created by registry.startup() via init_schema()
and init_server_schema().
"""

from __future__ import annotations

import os
from collections.abc import AsyncGenerator
from datetime import UTC, datetime

import pytest
from httpx import ASGITransport, AsyncClient

from imprint_server._pool import get_pg_pool
from imprint_server.app import create_app
from imprint_server.config import ServerConfig
from imprint_server.registry import AgentRegistry

POSTGRES_URL = os.environ.get("IMPRINT_STORE", "")

AGENT = "pg-test-agent"
USER = "pg-test-user"


# -- Fixtures -----------------------------------------------------------------


def _pg_config(auth_disabled: bool = True) -> ServerConfig:
    if not POSTGRES_URL or not POSTGRES_URL.startswith("postgres"):
        pytest.skip("IMPRINT_STORE is not a Postgres URL -- skipping postgres tests")
    return ServerConfig(
        store=POSTGRES_URL,
        default_mode="frugal",
        auth_disabled=auth_disabled,
    )


async def _truncate_all(registry: AgentRegistry) -> None:
    """Wipe all server and library tables. Called before each test."""
    pool = get_pg_pool(registry)
    # Server-specific tables
    await pool.execute("TRUNCATE sessions, jobs, api_keys, policy_events RESTART IDENTITY")
    # Library tables (CASCADE removes memory_signal_links, memory_events)
    await pool.execute(
        "TRUNCATE memories, signals, scopes, compiled_policies, agent_config"
        " RESTART IDENTITY CASCADE"
    )


@pytest.fixture()
async def pg_setup() -> AsyncGenerator[tuple[ServerConfig, AgentRegistry], None]:
    """Connected registry + truncated DB. Yields (config, registry)."""
    config = _pg_config()
    registry = AgentRegistry(config)
    await registry.startup()
    await _truncate_all(registry)
    yield config, registry
    await registry.shutdown()


@pytest.fixture()
async def pg_client(
    pg_setup: tuple[ServerConfig, AgentRegistry],
) -> AsyncGenerator[AsyncClient, None]:
    """Full FastAPI app backed by Postgres, accessed via httpx."""
    config, registry = pg_setup
    app = create_app(config, registry)
    transport = ASGITransport(app=app)  # type: ignore[arg-type]
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client


# -- Schema -------------------------------------------------------------------


@pytest.mark.postgres
async def test_init_server_schema_idempotent(
    pg_setup: tuple[ServerConfig, AgentRegistry],
) -> None:
    """Calling init_server_schema twice must not raise."""
    config, registry = pg_setup
    from imprint_server.db import init_server_schema

    await init_server_schema(config, registry.store)
    await init_server_schema(config, registry.store)


@pytest.mark.postgres
async def test_all_server_tables_exist(
    pg_setup: tuple[ServerConfig, AgentRegistry],
) -> None:
    _config, registry = pg_setup
    pool = get_pg_pool(registry)
    rows = await pool.fetch("SELECT tablename FROM pg_tables WHERE schemaname = 'public'")
    tables = {str(row["tablename"]) for row in rows}
    assert {"sessions", "jobs", "api_keys", "policy_events"} <= tables


# -- Sessions (Postgres paths) ------------------------------------------------


@pytest.mark.postgres
async def test_pg_create_and_get_session(
    pg_setup: tuple[ServerConfig, AgentRegistry],
) -> None:
    _config, registry = pg_setup
    from imprint_server.stores.sessions import pg_create_session, pg_get_session

    pool = get_pg_pool(registry)
    session_id = await pg_create_session(
        pool,
        agent_id=AGENT,
        user_id=USER,
        context="pg test",
        ttl=3600,
    )
    assert session_id.startswith("sess_")

    row = await pg_get_session(pool, session_id)
    assert row is not None
    assert row.agent_id == AGENT
    assert row.user_id == USER
    assert row.context == "pg test"
    assert row.closed_at is None


@pytest.mark.postgres
async def test_pg_get_session_returns_none_for_missing(
    pg_setup: tuple[ServerConfig, AgentRegistry],
) -> None:
    _, registry = pg_setup
    from imprint_server.stores.sessions import pg_get_session

    pool = get_pg_pool(registry)
    row = await pg_get_session(pool, "sess_doesnotexist")
    assert row is None


@pytest.mark.postgres
async def test_pg_update_session_policy(
    pg_setup: tuple[ServerConfig, AgentRegistry],
) -> None:
    _config, registry = pg_setup
    from imprint_server.stores.sessions import (
        pg_create_session,
        pg_get_session,
        pg_update_session_policy,
    )

    pool = get_pg_pool(registry)
    session_id = await pg_create_session(pool, agent_id=AGENT, user_id=USER, context=None, ttl=3600)
    await pg_update_session_policy(
        pool,
        session_id,
        retrieved_ids=["m1", "m2"],
        alpha_used=0.7,
        context="updated context",
    )
    row = await pg_get_session(pool, session_id)
    assert row is not None
    assert row.retrieved_ids == ["m1", "m2"]
    assert abs(row.alpha_used - 0.7) < 0.001
    assert row.context == "updated context"


@pytest.mark.postgres
async def test_pg_close_session(
    pg_setup: tuple[ServerConfig, AgentRegistry],
) -> None:
    _config, registry = pg_setup
    from imprint_server.stores.sessions import (
        pg_close_session,
        pg_create_session,
        pg_get_session,
    )

    pool = get_pg_pool(registry)
    session_id = await pg_create_session(pool, agent_id=AGENT, user_id=USER, context=None, ttl=3600)
    await pg_close_session(pool, session_id, outcome=0.9, correction=None)

    row = await pg_get_session(pool, session_id)
    assert row is not None
    assert row.closed_at is not None
    assert row.outcome is not None
    assert abs(row.outcome - 0.9) < 0.001


# -- API keys (Postgres paths) ------------------------------------------------


@pytest.mark.postgres
async def test_pg_insert_and_lookup_key(
    pg_setup: tuple[ServerConfig, AgentRegistry],
) -> None:
    _, registry = pg_setup
    from imprint_server.stores.api_keys import (
        ApiKeyRow,
        generate_raw_key,
        hash_key,
        pg_insert_with_pool,
        pg_lookup_with_pool,
    )

    pool = get_pg_pool(registry)
    raw = generate_raw_key()
    row = ApiKeyRow(
        key_hash=hash_key(raw),
        agent_id=None,
        label="test key",
        created_at=datetime.now(UTC),
        expires_at=None,
        active=True,
    )
    await pg_insert_with_pool(pool, row)

    found = await pg_lookup_with_pool(pool, hash_key(raw))
    assert found is not None
    assert found.label == "test key"
    assert found.agent_id is None
    assert found.active is True


@pytest.mark.postgres
async def test_pg_count_active_keys(
    pg_setup: tuple[ServerConfig, AgentRegistry],
) -> None:
    _, registry = pg_setup
    from imprint_server.stores.api_keys import (
        ApiKeyRow,
        generate_raw_key,
        hash_key,
        pg_count_with_pool,
        pg_insert_with_pool,
    )

    pool = get_pg_pool(registry)
    assert await pg_count_with_pool(pool) == 0

    for _ in range(3):
        raw = generate_raw_key()
        row = ApiKeyRow(
            key_hash=hash_key(raw),
            agent_id=None,
            label=None,
            created_at=datetime.now(UTC),
            expires_at=None,
            active=True,
        )
        await pg_insert_with_pool(pool, row)

    assert await pg_count_with_pool(pool) == 3


@pytest.mark.postgres
async def test_pg_revoke_key(
    pg_setup: tuple[ServerConfig, AgentRegistry],
) -> None:
    _, registry = pg_setup
    from imprint_server.stores.api_keys import (
        ApiKeyRow,
        generate_raw_key,
        hash_key,
        pg_insert_with_pool,
        pg_lookup_with_pool,
        pg_revoke_with_pool,
    )

    pool = get_pg_pool(registry)
    raw = generate_raw_key()
    kh = hash_key(raw)
    await pg_insert_with_pool(
        pool,
        ApiKeyRow(
            key_hash=kh,
            agent_id=None,
            label="to revoke",
            created_at=datetime.now(UTC),
            expires_at=None,
            active=True,
        ),
    )

    revoked = await pg_revoke_with_pool(pool, kh)
    assert revoked is True

    # Lookup returns None for inactive keys.
    found = await pg_lookup_with_pool(pool, kh)
    assert found is None


# -- Policy events (Postgres path) --------------------------------------------


@pytest.mark.postgres
async def test_pg_log_policy_event(
    pg_setup: tuple[ServerConfig, AgentRegistry],
) -> None:
    config, registry = pg_setup
    from imprint_server.stores.policy_events import log_policy_event

    await log_policy_event(
        registry=registry,
        config=config,
        agent_id=AGENT,
        user_id=USER,
        session_id=None,
        retrieved_memories=[],
        filtered_memories=[],
        alpha_used=0.5,
        context="test context",
    )

    pool = get_pg_pool(registry)
    rows = await pool.fetch("SELECT * FROM policy_events WHERE agent_id = $1", AGENT)
    assert len(rows) == 1
    assert str(rows[0]["user_id"]) == USER


# -- Full HTTP stack against Postgres -----------------------------------------


@pytest.mark.postgres
async def test_http_observe_and_policy(pg_client: AsyncClient) -> None:
    """observe() + get_policy() round-trip via HTTP against Postgres."""
    resp = await pg_client.post(
        f"/v1/agents/{AGENT}/observe",
        json={
            "user_id": USER,
            "agent_output": "Let me summarize with bullet points.",
            "user_response": "Please stop using bullet points.",
        },
    )
    assert resp.status_code == 200

    resp = await pg_client.post(
        f"/v1/agents/{AGENT}/policy",
        json={"user_id": USER, "context": "coding"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert "policy_text" in body
    assert "memory_count" in body


@pytest.mark.postgres
async def test_http_session_lifecycle(pg_client: AsyncClient) -> None:
    """Open -> policy -> close session lifecycle via HTTP against Postgres."""
    open_resp = await pg_client.post(
        f"/v1/agents/{AGENT}/sessions",
        json={"user_id": USER, "context": "integration test"},
    )
    assert open_resp.status_code == 200
    sid = open_resp.json()["session_id"]
    assert sid.startswith("sess_")

    pol_resp = await pg_client.post(
        f"/v1/agents/{AGENT}/sessions/{sid}/policy",
        json={"context": "integration test"},
    )
    assert pol_resp.status_code == 200

    close_resp = await pg_client.post(
        f"/v1/agents/{AGENT}/sessions/{sid}/close",
        json={"outcome": 0.8},
    )
    assert close_resp.status_code == 200
    assert close_resp.json()["ok"] is True


@pytest.mark.postgres
async def test_http_policy_event_logged_with_real_session_id(
    pg_client: AsyncClient,
    pg_setup: tuple[ServerConfig, AgentRegistry],
) -> None:
    """Policy calls within a session log the session_id to policy_events."""
    _config, registry = pg_setup
    open_resp = await pg_client.post(
        f"/v1/agents/{AGENT}/sessions",
        json={"user_id": USER},
    )
    sid = open_resp.json()["session_id"]

    await pg_client.post(
        f"/v1/agents/{AGENT}/sessions/{sid}/policy",
        json={"context": "test"},
    )

    pool = get_pg_pool(registry)
    rows = await pool.fetch("SELECT session_id FROM policy_events WHERE agent_id = $1", AGENT)
    assert len(rows) >= 1
    assert str(rows[0]["session_id"]) == sid


@pytest.mark.postgres
async def test_http_auth_with_postgres_key_store(
    pg_setup: tuple[ServerConfig, AgentRegistry],
) -> None:
    """Auth middleware reads keys from Postgres and enforces them correctly."""
    config_auth = ServerConfig(
        store=POSTGRES_URL,
        default_mode="frugal",
        auth_disabled=False,
    )
    registry_auth = AgentRegistry(config_auth)
    await registry_auth.startup()
    await _truncate_all(registry_auth)

    from imprint_server.stores.api_keys import (
        ApiKeyRow,
        generate_raw_key,
        hash_key,
        pg_insert_with_pool,
    )

    pool = get_pg_pool(registry_auth)
    raw = generate_raw_key()
    await pg_insert_with_pool(
        pool,
        ApiKeyRow(
            key_hash=hash_key(raw),
            agent_id=None,
            label="test",
            created_at=datetime.now(UTC),
            expires_at=None,
            active=True,
        ),
    )

    app = create_app(config_auth, registry_auth)
    transport = ASGITransport(app=app)  # type: ignore[arg-type]
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # No key -> 401
        r1 = await client.get(f"/v1/agents/{AGENT}/memories/{USER}")
        assert r1.status_code == 401

        # Valid key -> 200
        r2 = await client.get(
            f"/v1/agents/{AGENT}/memories/{USER}",
            headers={"Authorization": f"Bearer {raw}"},
        )
        assert r2.status_code == 200

    await registry_auth.shutdown()


@pytest.mark.postgres
async def test_http_admin_create_and_get_agent(pg_client: AsyncClient) -> None:
    """POST /v1/agents pre-configures an agent, GET /v1/agents/{id} reads it."""
    resp = await pg_client.post(
        "/v1/agents",
        json={"agent_id": "pg-admin-agent", "processing_mode": "eager", "pre_warm": True},
    )
    assert resp.status_code == 200

    get_resp = await pg_client.get("/v1/agents/pg-admin-agent")
    assert get_resp.status_code == 200
    assert get_resp.json()["processing_mode"] == "eager"


@pytest.mark.postgres
async def test_http_health_reports_postgres(pg_client: AsyncClient) -> None:
    """Health endpoint must report store=postgres and db_ok=true."""
    resp = await pg_client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["store"] == "postgres"
    assert body["db_ok"] is True
