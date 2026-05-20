"""Tests for paginated list_memories and list_events endpoints."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient

from imprint_server.app import create_app
from imprint_server.config import ServerConfig
from imprint_server.registry import AgentRegistry

AGENT = "page-test-agent"
USER = "page-test-user"


@pytest.fixture()
async def client(tmp_path: Path) -> AsyncGenerator[AsyncClient, None]:
    config = ServerConfig(
        store=f"sqlite:///{tmp_path / 'page.db'}",
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


async def _seed(client: AsyncClient, count: int) -> None:
    """Store count direction memories for USER."""
    for i in range(count):
        await client.post(
            f"/v1/agents/{AGENT}/memories/{USER}/directions",
            json={"directions": [f"Rule number {i}: always be explicit."]},
        )


# -- list_memories backward compat --------------------------------------------


async def test_list_memories_no_limit_returns_plain_list(client: AsyncClient) -> None:
    await _seed(client, 3)
    resp = await client.get(f"/v1/agents/{AGENT}/memories/{USER}")
    assert resp.status_code == 200
    assert isinstance(resp.json(), list)


# -- paginated memories -------------------------------------------------------


async def test_list_memories_with_limit_returns_envelope(client: AsyncClient) -> None:
    await _seed(client, 5)
    resp = await client.get(f"/v1/agents/{AGENT}/memories/{USER}", params={"limit": 3})
    assert resp.status_code == 200
    body = resp.json()
    assert "items" in body
    assert "next_cursor" in body
    assert len(body["items"]) == 3


async def test_list_memories_pagination_cursor_advances(client: AsyncClient) -> None:
    await _seed(client, 5)
    page1 = (await client.get(f"/v1/agents/{AGENT}/memories/{USER}", params={"limit": 2})).json()
    assert page1["next_cursor"] is not None

    page2 = (
        await client.get(
            f"/v1/agents/{AGENT}/memories/{USER}",
            params={"limit": 2, "cursor": page1["next_cursor"]},
        )
    ).json()
    assert len(page2["items"]) > 0
    # No items should appear on both pages.
    ids1 = {m["id"] for m in page1["items"]}
    ids2 = {m["id"] for m in page2["items"]}
    assert ids1.isdisjoint(ids2)


async def test_list_memories_last_page_has_null_cursor(client: AsyncClient) -> None:
    await _seed(client, 2)
    page = (await client.get(f"/v1/agents/{AGENT}/memories/{USER}", params={"limit": 10})).json()
    assert page["next_cursor"] is None


# -- list_events backward compat ----------------------------------------------


async def test_list_events_no_limit_returns_plain_list(client: AsyncClient) -> None:
    await _seed(client, 2)
    resp = await client.get(f"/v1/agents/{AGENT}/events/{USER}")
    assert resp.status_code == 200
    assert isinstance(resp.json(), list)


# -- paginated events ---------------------------------------------------------


async def test_list_events_with_limit_returns_envelope(client: AsyncClient) -> None:
    await _seed(client, 5)
    resp = await client.get(f"/v1/agents/{AGENT}/events/{USER}", params={"limit": 2})
    assert resp.status_code == 200
    body = resp.json()
    assert "items" in body
    assert "next_cursor" in body


async def test_list_events_last_page_has_null_cursor(client: AsyncClient) -> None:
    resp = await client.get(f"/v1/agents/{AGENT}/events/{USER}", params={"limit": 100})
    body = resp.json()
    assert body["next_cursor"] is None
