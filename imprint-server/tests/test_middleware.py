"""Tests for RequestIDMiddleware, AccessLogMiddleware, and RateLimitMiddleware."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient

from imprint_server.app import create_app
from imprint_server.config import ServerConfig
from imprint_server.registry import AgentRegistry

AGENT = "mw-test-agent"
USER = "mw-test-user"


@pytest.fixture()
async def client(tmp_path: Path) -> AsyncGenerator[AsyncClient, None]:
    config = ServerConfig(
        store=f"sqlite:///{tmp_path / 'mw.db'}",
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


# -- X-Request-ID middleware --------------------------------------------------


async def test_request_id_generated_when_absent(client: AsyncClient) -> None:
    resp = await client.get("/health")
    assert "x-request-id" in resp.headers
    assert len(resp.headers["x-request-id"]) > 0


async def test_request_id_echoed_when_supplied(client: AsyncClient) -> None:
    resp = await client.get("/health", headers={"X-Request-ID": "test-req-abc"})
    assert resp.headers["x-request-id"] == "test-req-abc"


async def test_request_id_different_per_request(client: AsyncClient) -> None:
    r1 = await client.get("/health")
    r2 = await client.get("/health")
    assert r1.headers["x-request-id"] != r2.headers["x-request-id"]


# -- Health split -------------------------------------------------------------


async def test_health_live_always_200(client: AsyncClient) -> None:
    resp = await client.get("/health/live")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


async def test_health_ready_includes_redis_disabled(client: AsyncClient) -> None:
    resp = await client.get("/health/ready")
    assert resp.status_code == 200
    body = resp.json()
    assert body["redis"] == "disabled"
    assert body["db_ok"] is True


async def test_health_alias_matches_ready(client: AsyncClient) -> None:
    ready = (await client.get("/health/ready")).json()
    alias = (await client.get("/health")).json()
    assert alias["status"] == ready["status"]
    assert alias["db_ok"] == ready["db_ok"]


# -- Rate limiting (in-memory fallback, no Redis) -----------------------------


@pytest.fixture()
async def rate_limited_client(tmp_path: Path) -> AsyncGenerator[AsyncClient, None]:
    """In-memory rate limiter fixture. Pins redis_url='' to prevent env leakage."""
    config = ServerConfig(
        store=f"sqlite:///{tmp_path / 'rl.db'}",
        default_mode="frugal",
        auth_disabled=True,
        rate_limit_enabled=True,
        rate_limit_requests=3,
        rate_limit_window=60,
        redis_url="",
    )
    registry = AgentRegistry(config)
    app = create_app(config, registry)
    await registry.startup()
    transport = ASGITransport(app=app)  # type: ignore[arg-type]
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c
    await registry.shutdown()


async def test_rate_limit_allows_within_limit(rate_limited_client: AsyncClient) -> None:
    for _ in range(3):
        resp = await rate_limited_client.get("/health")
        assert resp.status_code == 200


async def test_rate_limit_blocks_when_exceeded(rate_limited_client: AsyncClient) -> None:
    for _ in range(3):
        await rate_limited_client.get("/health/ready")  # exempt -- doesn't count
    # health/* is exempt; hit a non-exempt path 3 times to fill the limit.
    for _ in range(3):
        await rate_limited_client.get(f"/v1/agents/{AGENT}/memories/{USER}")
    # 4th request should be rate limited.
    resp = await rate_limited_client.get(f"/v1/agents/{AGENT}/memories/{USER}")
    assert resp.status_code == 429


async def test_rate_limit_response_has_retry_after(rate_limited_client: AsyncClient) -> None:
    for _ in range(3):
        await rate_limited_client.get(f"/v1/agents/{AGENT}/memories/{USER}")
    resp = await rate_limited_client.get(f"/v1/agents/{AGENT}/memories/{USER}")
    assert resp.status_code == 429
    assert "retry-after" in resp.headers


async def test_health_endpoints_exempt_from_rate_limit(rate_limited_client: AsyncClient) -> None:
    """Health endpoints must never be rate limited."""
    for _ in range(10):
        resp = await rate_limited_client.get("/health/live")
        assert resp.status_code == 200


# -- Rate limiting (Redis-backed, requires IMPRINT_REDIS_URL) -----------------


@pytest.fixture()
async def redis_rate_limited_client(
    tmp_path: Path,
) -> AsyncGenerator[AsyncClient, None]:
    """Redis-backed rate limiter fixture. Skipped when Redis is not reachable.

    Uses redis://localhost:6379 -- start with `just redis-dev` first.
    Does not read IMPRINT_REDIS_URL from env (conftest.py clears it).
    """
    import socket

    redis_url = "redis://localhost:6379"

    # Check connectivity before creating the fixture to get a clean skip
    # rather than a connection error mid-setup.
    try:
        sock = socket.create_connection(("localhost", 6379), timeout=1)
        sock.close()
    except OSError:
        pytest.skip("Redis not reachable at localhost:6379 -- run `just redis-dev` first")

    config = ServerConfig(
        store=f"sqlite:///{tmp_path / 'rl_redis.db'}",
        default_mode="frugal",
        auth_disabled=True,
        rate_limit_enabled=True,
        rate_limit_requests=3,
        rate_limit_window=60,
        redis_url=redis_url,
    )
    registry = AgentRegistry(config)
    app = create_app(config, registry)
    await registry.startup()
    # Flush relevant keys so counts are isolated per test.
    if registry.redis is not None:
        await registry.redis.delete_pattern("imprint:rl:*")
    transport = ASGITransport(app=app)  # type: ignore[arg-type]
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c
    await registry.shutdown()


@pytest.mark.redis
async def test_redis_rate_limit_blocks_when_exceeded(
    redis_rate_limited_client: AsyncClient,
) -> None:
    for _ in range(3):
        await redis_rate_limited_client.get(f"/v1/agents/{AGENT}/memories/{USER}")
    resp = await redis_rate_limited_client.get(f"/v1/agents/{AGENT}/memories/{USER}")
    assert resp.status_code == 429


@pytest.mark.redis
async def test_redis_rate_limit_response_has_retry_after(
    redis_rate_limited_client: AsyncClient,
) -> None:
    for _ in range(3):
        await redis_rate_limited_client.get(f"/v1/agents/{AGENT}/memories/{USER}")
    resp = await redis_rate_limited_client.get(f"/v1/agents/{AGENT}/memories/{USER}")
    assert resp.status_code == 429
    assert "retry-after" in resp.headers


@pytest.mark.redis
async def test_redis_health_endpoints_exempt(
    redis_rate_limited_client: AsyncClient,
) -> None:
    """Health endpoints must never be rate limited even with Redis backing."""
    for _ in range(10):
        resp = await redis_rate_limited_client.get("/health/live")
        assert resp.status_code == 200
