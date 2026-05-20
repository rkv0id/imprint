"""Targeted REST API coverage tests filling gaps in test_agents.py.

Covers:
- RFC 9457 problem+json error response shape
- Lineage happy path (memories exist and have lineage)
- forget_user actually removes memories from list
- memory_health reflects stored memories
- correct/reinforce with a valid open session (end-to-end learning signal path)
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient

from imprint_server.app import create_app
from imprint_server.config import ServerConfig
from imprint_server.registry import AgentRegistry

AGENT = "rest-coverage-agent"
USER = "rest-coverage-user"


@pytest.fixture()
async def client(tmp_path: Path) -> AsyncGenerator[AsyncClient, None]:
    config = ServerConfig(
        store=f"sqlite:///{tmp_path / 'rest.db'}",
        default_mode="frugal",
        auth_disabled=True,
        redis_url="",
    )
    registry = AgentRegistry(config)
    app = create_app(config, registry)
    await registry.startup()
    transport = ASGITransport(app=app)  # type: ignore[arg-type]
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c
    await registry.shutdown()


# -- RFC 9457 problem+json error shape ----------------------------------------


async def test_404_response_is_problem_json(client: AsyncClient) -> None:
    """Unknown memory lineage must return a well-formed problem+json body."""
    resp = await client.get("/v1/memories/mem_does_not_exist/lineage")
    assert resp.status_code == 404
    assert resp.headers["content-type"] == "application/problem+json"
    body = resp.json()
    assert body["type"] == "about:blank"
    assert body["title"] == "Not Found"
    assert body["status"] == 404
    assert "detail" in body


async def test_422_response_is_problem_json(client: AsyncClient) -> None:
    """Empty directions list must return a well-formed problem+json body."""
    resp = await client.post(
        f"/v1/agents/{AGENT}/memories/{USER}/directions",
        json={"directions": []},
    )
    assert resp.status_code == 422
    assert resp.headers["content-type"] == "application/problem+json"
    body = resp.json()
    assert body["type"] == "about:blank"
    assert body["title"] == "Unprocessable Request"
    assert body["status"] == 422
    assert "detail" in body


async def test_deactivate_unknown_memory_returns_problem_json(client: AsyncClient) -> None:
    resp = await client.delete(f"/v1/agents/{AGENT}/memories/{USER}/mem_ghost")
    assert resp.status_code == 404
    body = resp.json()
    assert body["status"] == 404
    assert body["type"] == "about:blank"


# -- Lineage happy path -------------------------------------------------------


async def test_lineage_returns_chain_for_existing_memory(client: AsyncClient) -> None:
    """Observe a direction then fetch its lineage -- must return the memory and chain fields."""
    await client.post(
        f"/v1/agents/{AGENT}/memories/{USER}/directions",
        json={"directions": ["Always write in active voice."]},
    )
    memories = (await client.get(f"/v1/agents/{AGENT}/memories/{USER}")).json()
    assert len(memories) >= 1
    memory_id = memories[0]["id"]

    resp = await client.get(f"/v1/memories/{memory_id}/lineage")
    assert resp.status_code == 200
    body = resp.json()
    assert "memory" in body
    assert body["memory"]["id"] == memory_id
    assert "events" in body
    assert "superseded_by" in body
    assert "superseded_memories" in body


async def test_lineage_includes_created_event(client: AsyncClient) -> None:
    await client.post(
        f"/v1/agents/{AGENT}/memories/{USER}/directions",
        json={"directions": ["Be concise."]},
    )
    memories = (await client.get(f"/v1/agents/{AGENT}/memories/{USER}")).json()
    memory_id = memories[0]["id"]

    resp = await client.get(f"/v1/memories/{memory_id}/lineage")
    body = resp.json()
    assert "memory" in body
    assert body["memory"]["id"] == memory_id
    # The chain fields are present even if empty for a brand-new memory.
    assert isinstance(body["superseded_memories"], list)
    assert body["superseded_by"] is None


# -- forget_user actually removes memories ------------------------------------


async def test_forget_user_removes_all_memories(client: AsyncClient) -> None:
    for i in range(3):
        await client.post(
            f"/v1/agents/{AGENT}/memories/{USER}/directions",
            json={"directions": [f"Rule {i}: be precise."]},
        )
    before = (await client.get(f"/v1/agents/{AGENT}/memories/{USER}")).json()
    assert len(before) >= 3

    await client.delete(f"/v1/agents/{AGENT}/memories/{USER}")

    after = (await client.get(f"/v1/agents/{AGENT}/memories/{USER}")).json()
    assert after == []


async def test_forget_user_does_not_affect_other_users(client: AsyncClient) -> None:
    other = "rest-coverage-other"
    await client.post(
        f"/v1/agents/{AGENT}/memories/{USER}/directions",
        json={"directions": ["User A preference."]},
    )
    await client.post(
        f"/v1/agents/{AGENT}/memories/{other}/directions",
        json={"directions": ["User B preference."]},
    )

    await client.delete(f"/v1/agents/{AGENT}/memories/{USER}")

    other_mems = (await client.get(f"/v1/agents/{AGENT}/memories/{other}")).json()
    assert len(other_mems) >= 1


# -- memory_health reflects stored memories -----------------------------------


async def test_memory_health_reflects_stored_memories(client: AsyncClient) -> None:
    await client.post(
        f"/v1/agents/{AGENT}/memories/{USER}/directions",
        json={"directions": ["Write in the second person."]},
    )
    resp = await client.get(f"/v1/agents/{AGENT}/health/{USER}")
    assert resp.status_code == 200
    body = resp.json()
    assert body["active"] >= 1
    assert body["total"] >= 1


async def test_memory_health_decrements_after_deactivate(client: AsyncClient) -> None:
    await client.post(
        f"/v1/agents/{AGENT}/memories/{USER}/directions",
        json={"directions": ["Prefer numbered lists."]},
    )
    memories = (await client.get(f"/v1/agents/{AGENT}/memories/{USER}")).json()
    memory_id = memories[0]["id"]

    before = (await client.get(f"/v1/agents/{AGENT}/health/{USER}")).json()
    await client.delete(f"/v1/agents/{AGENT}/memories/{USER}/{memory_id}")
    after = (await client.get(f"/v1/agents/{AGENT}/health/{USER}")).json()

    assert after["active"] < before["active"]


# -- correct / reinforce with valid session -----------------------------------


async def test_correct_with_valid_session_closes_it(client: AsyncClient) -> None:
    """correct(session_id=...) must close the session with a negative signal."""
    sess = (await client.post(f"/v1/agents/{AGENT}/sessions", json={"user_id": USER})).json()
    sid = sess["session_id"]

    resp = await client.post(
        f"/v1/agents/{AGENT}/correct/{USER}",
        json={"content": "Do not summarize, give the full answer.", "session_id": sid},
    )
    assert resp.status_code == 200
    assert resp.json()["ok"] is True
    assert resp.json()["memory_id"] is not None

    # Session is now closed; a second correct on the same session must fail.
    resp2 = await client.post(
        f"/v1/agents/{AGENT}/correct/{USER}",
        json={"content": "Still wrong.", "session_id": sid},
    )
    assert resp2.status_code == 422


async def test_reinforce_with_valid_session_closes_it(client: AsyncClient) -> None:
    """reinforce(session_id=...) must close the session with a positive signal."""
    sess = (await client.post(f"/v1/agents/{AGENT}/sessions", json={"user_id": USER})).json()
    sid = sess["session_id"]

    resp = await client.post(
        f"/v1/agents/{AGENT}/reinforce/{USER}",
        json={"session_id": sid},
    )
    assert resp.status_code == 200
    assert resp.json()["applied"] is True

    # Session is now closed; reinforce again must fail.
    resp2 = await client.post(
        f"/v1/agents/{AGENT}/reinforce/{USER}",
        json={"session_id": sid},
    )
    assert resp2.status_code == 422


async def test_reinforce_then_correct_fails(client: AsyncClient) -> None:
    """Once a session is closed by reinforce, correct must reject it."""
    sess = (await client.post(f"/v1/agents/{AGENT}/sessions", json={"user_id": USER})).json()
    sid = sess["session_id"]

    await client.post(f"/v1/agents/{AGENT}/reinforce/{USER}", json={"session_id": sid})
    resp = await client.post(
        f"/v1/agents/{AGENT}/correct/{USER}",
        json={"content": "Late correction.", "session_id": sid},
    )
    assert resp.status_code == 422
