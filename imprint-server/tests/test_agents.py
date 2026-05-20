"""Integration tests for the core agent operation endpoints.

Uses a real SQLite store in frugal mode (no LLM calls) and httpx.AsyncClient
over ASGITransport. Tests the full stack: routing -> handler -> registry ->
library -> SQLite. No mocking.

frugal mode means:
  - observe() uses heuristic signal detection only (no LLM).
  - Signals are detected only for explicit correction patterns.
  - observe_directions() skips LLM validation.
  - consolidate() skips LLM scope consolidation.
  - get_policy() only calls the compiler LLM if there are memories to compile.
    With no memories, returns empty policy without any LLM call.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient

from imprint_server.app import create_app
from imprint_server.config import ServerConfig
from imprint_server.registry import AgentRegistry

AGENT = "test-agent"
USER = "test-user"


# -- Fixture ------------------------------------------------------------------


@pytest.fixture()
async def client(tmp_path: Path) -> AsyncGenerator[AsyncClient, None]:
    """Real SQLite registry + imprint-server app, frugal mode, no LLM calls."""
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


# -- observe ------------------------------------------------------------------


async def test_observe_turn_returns_ok(client: AsyncClient) -> None:
    resp = await client.post(
        f"/v1/agents/{AGENT}/observe",
        json={
            "user_id": USER,
            "agent_output": "Here is a bullet list: - one - two",
            "user_response": "Stop using bullet points. Write in prose.",
        },
    )
    assert resp.status_code == 200
    assert resp.json()["ok"] is True


async def test_observe_directions_returns_ok(client: AsyncClient) -> None:
    resp = await client.post(
        f"/v1/agents/{AGENT}/observe",
        json={
            "user_id": USER,
            "directions": [
                "Always use conventional commits.",
                "No docstrings except on public API.",
            ],
        },
    )
    assert resp.status_code == 200
    assert resp.json()["ok"] is True


async def test_observe_missing_fields_returns_422(client: AsyncClient) -> None:
    resp = await client.post(
        f"/v1/agents/{AGENT}/observe",
        json={"user_id": USER},
    )
    assert resp.status_code == 422


async def test_observe_ambiguous_fields_returns_422(client: AsyncClient) -> None:
    """Providing both directions and agent_output is invalid."""
    resp = await client.post(
        f"/v1/agents/{AGENT}/observe",
        json={
            "user_id": USER,
            "agent_output": "some output",
            "user_response": "some response",
            "directions": ["some direction"],
        },
    )
    assert resp.status_code == 422


async def test_observe_missing_user_response_returns_422(client: AsyncClient) -> None:
    resp = await client.post(
        f"/v1/agents/{AGENT}/observe",
        json={"user_id": USER, "agent_output": "some output"},
    )
    assert resp.status_code == 422


# -- policy -------------------------------------------------------------------


async def test_policy_returns_empty_when_no_memories(client: AsyncClient) -> None:
    resp = await client.post(
        f"/v1/agents/{AGENT}/policy",
        json={"user_id": USER},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["policy_text"] == ""
    assert body["memory_count"] == 0
    assert body["dropped_count"] == 0
    assert "compiled_at" in body


async def test_policy_logs_policy_event(client: AsyncClient, tmp_path: Path) -> None:
    """Policy events should be inserted into the policy_events table."""
    await client.post(
        f"/v1/agents/{AGENT}/policy",
        json={"user_id": USER, "context": "coding"},
    )
    # Verify the row exists in the SQLite file.
    import aiosqlite

    db_path = str(tmp_path / "test.db")
    async with (
        aiosqlite.connect(db_path) as conn,
        conn.execute(
            "SELECT COUNT(*) FROM policy_events WHERE agent_id = ? AND user_id = ?",
            (AGENT, USER),
        ) as cursor,
    ):
        row = await cursor.fetchone()
    assert row is not None
    assert row[0] == 1


async def test_policy_context_hash_stored(client: AsyncClient, tmp_path: Path) -> None:
    await client.post(
        f"/v1/agents/{AGENT}/policy",
        json={"user_id": USER, "context": "my context"},
    )
    import hashlib

    import aiosqlite

    expected_hash = hashlib.sha256(b"my context").hexdigest()
    db_path = str(tmp_path / "test.db")
    async with (
        aiosqlite.connect(db_path) as conn,
        conn.execute(
            "SELECT context_hash FROM policy_events WHERE agent_id = ?",
            (AGENT,),
        ) as cursor,
    ):
        row = await cursor.fetchone()
    assert row is not None
    assert row[0] == expected_hash


async def test_policy_no_context_hash_is_null(client: AsyncClient, tmp_path: Path) -> None:
    await client.post(
        f"/v1/agents/{AGENT}/policy",
        json={"user_id": USER},
    )
    import aiosqlite

    db_path = str(tmp_path / "test.db")
    async with (
        aiosqlite.connect(db_path) as conn,
        conn.execute(
            "SELECT context_hash FROM policy_events WHERE agent_id = ?",
            (AGENT,),
        ) as cursor,
    ):
        row = await cursor.fetchone()
    assert row is not None
    assert row[0] is None


# -- memories -----------------------------------------------------------------


async def test_list_memories_empty(client: AsyncClient) -> None:
    resp = await client.get(f"/v1/agents/{AGENT}/memories/{USER}")
    assert resp.status_code == 200
    assert resp.json() == []


async def test_forget_user_returns_ok(client: AsyncClient) -> None:
    resp = await client.delete(f"/v1/agents/{AGENT}/memories/{USER}")
    assert resp.status_code == 200
    assert resp.json()["ok"] is True


async def test_consolidate_returns_pruned_count(client: AsyncClient) -> None:
    resp = await client.post(f"/v1/agents/{AGENT}/memories/{USER}/consolidate")
    assert resp.status_code == 200
    assert resp.json()["pruned"] == 0


async def test_observe_directions_endpoint(client: AsyncClient) -> None:
    resp = await client.post(
        f"/v1/agents/{AGENT}/memories/{USER}/directions",
        json={"directions": ["Use conventional commits.", "Write in prose."]},
    )
    assert resp.status_code == 200
    assert resp.json()["stored"] >= 0


async def test_observe_directions_empty_returns_422(client: AsyncClient) -> None:
    resp = await client.post(
        f"/v1/agents/{AGENT}/memories/{USER}/directions",
        json={"directions": []},
    )
    assert resp.status_code == 422


# -- events -------------------------------------------------------------------


async def test_list_events_empty(client: AsyncClient) -> None:
    resp = await client.get(f"/v1/agents/{AGENT}/events/{USER}")
    assert resp.status_code == 200
    assert resp.json() == []


# -- health -------------------------------------------------------------------


async def test_memory_health_zero_state(client: AsyncClient) -> None:
    resp = await client.get(f"/v1/agents/{AGENT}/health/{USER}")
    assert resp.status_code == 200
    body = resp.json()
    assert body["total"] == 0
    assert body["active"] == 0
    assert body["pinned"] == 0
    assert body["by_scope"] == {}
    assert body["by_type"] == {}


# -- lineage ------------------------------------------------------------------


async def test_lineage_unknown_memory_returns_404(client: AsyncClient) -> None:
    # Need to trigger agent initialization first.
    await client.get(f"/v1/agents/{AGENT}/memories/{USER}")
    resp = await client.get("/v1/memories/nonexistent_memory_id/lineage")
    assert resp.status_code == 404


async def test_lineage_no_agents_loaded_returns_404(client: AsyncClient) -> None:
    """Without any loaded agents, lineage cannot proceed."""
    resp = await client.get("/v1/memories/some_id/lineage")
    assert resp.status_code == 404


# -- multiple agents ----------------------------------------------------------


async def test_different_agents_are_independent(client: AsyncClient) -> None:
    """Two agent IDs must have isolated memory spaces."""
    resp_a = await client.get(f"/v1/agents/agent-a/memories/{USER}")
    resp_b = await client.get(f"/v1/agents/agent-b/memories/{USER}")
    assert resp_a.status_code == 200
    assert resp_b.status_code == 200
    assert resp_a.json() == []
    assert resp_b.json() == []


async def test_policy_called_multiple_times_accumulates_events(
    client: AsyncClient, tmp_path: Path
) -> None:
    for _ in range(3):
        await client.post(
            f"/v1/agents/{AGENT}/policy",
            json={"user_id": USER},
        )
    import aiosqlite

    db_path = str(tmp_path / "test.db")
    async with (
        aiosqlite.connect(db_path) as conn,
        conn.execute(
            "SELECT COUNT(*) FROM policy_events WHERE agent_id = ?",
            (AGENT,),
        ) as cursor,
    ):
        row = await cursor.fetchone()
    assert row is not None
    assert row[0] == 3


# -- search -------------------------------------------------------------------


async def test_search_memories_empty_returns_list(client: AsyncClient) -> None:
    resp = await client.get(
        f"/v1/agents/{AGENT}/memories/{USER}/search",
        params={"q": "prose style"},
    )
    assert resp.status_code == 200
    assert resp.json() == []


async def test_search_memories_returns_stored_memories(client: AsyncClient) -> None:
    """With no embedder configured, search falls back to list order."""
    await client.post(
        f"/v1/agents/{AGENT}/memories/{USER}/directions",
        json={"directions": ["Write in prose, not bullet points."]},
    )
    resp = await client.get(
        f"/v1/agents/{AGENT}/memories/{USER}/search",
        params={"q": "prose"},
    )
    assert resp.status_code == 200
    results = resp.json()
    assert len(results) >= 1
    assert all("id" in m and "content" in m for m in results)


async def test_search_memories_limit_respected(client: AsyncClient) -> None:
    for i in range(5):
        await client.post(
            f"/v1/agents/{AGENT}/memories/{USER}/directions",
            json={"directions": [f"Direction number {i}."]},
        )
    resp = await client.get(
        f"/v1/agents/{AGENT}/memories/{USER}/search",
        params={"q": "direction", "limit": 2},
    )
    assert resp.status_code == 200
    assert len(resp.json()) <= 2


async def test_search_memories_missing_q_returns_422(client: AsyncClient) -> None:
    resp = await client.get(f"/v1/agents/{AGENT}/memories/{USER}/search")
    assert resp.status_code == 422


# -- pin ----------------------------------------------------------------------


async def test_pin_memory_returns_ok(client: AsyncClient) -> None:
    await client.post(
        f"/v1/agents/{AGENT}/memories/{USER}/directions",
        json={"directions": ["Write in prose."]},
    )
    memories = (await client.get(f"/v1/agents/{AGENT}/memories/{USER}")).json()
    assert len(memories) >= 1
    memory_id = memories[0]["id"]

    resp = await client.post(f"/v1/agents/{AGENT}/memories/{memory_id}/pin")
    assert resp.status_code == 200
    assert resp.json()["ok"] is True


async def test_pin_memory_sets_pinned_flag(client: AsyncClient) -> None:
    await client.post(
        f"/v1/agents/{AGENT}/memories/{USER}/directions",
        json={"directions": ["Always be concise."]},
    )
    memory_id = (await client.get(f"/v1/agents/{AGENT}/memories/{USER}")).json()[0]["id"]

    await client.post(f"/v1/agents/{AGENT}/memories/{memory_id}/pin")

    memories = (await client.get(f"/v1/agents/{AGENT}/memories/{USER}")).json()
    pinned = next(m for m in memories if m["id"] == memory_id)
    assert pinned["pinned"] is True


# -- deactivate ---------------------------------------------------------------


async def test_deactivate_memory_returns_ok(client: AsyncClient) -> None:
    await client.post(
        f"/v1/agents/{AGENT}/memories/{USER}/directions",
        json={"directions": ["Prefer short sentences."]},
    )
    memory_id = (await client.get(f"/v1/agents/{AGENT}/memories/{USER}")).json()[0]["id"]

    resp = await client.delete(f"/v1/agents/{AGENT}/memories/{USER}/{memory_id}")
    assert resp.status_code == 200
    assert resp.json()["ok"] is True


async def test_deactivate_memory_removes_from_list(client: AsyncClient) -> None:
    await client.post(
        f"/v1/agents/{AGENT}/memories/{USER}/directions",
        json={"directions": ["Use active voice."]},
    )
    memory_id = (await client.get(f"/v1/agents/{AGENT}/memories/{USER}")).json()[0]["id"]

    await client.delete(f"/v1/agents/{AGENT}/memories/{USER}/{memory_id}")

    remaining = (await client.get(f"/v1/agents/{AGENT}/memories/{USER}")).json()
    assert all(m["id"] != memory_id for m in remaining)


async def test_deactivate_memory_unknown_returns_404(client: AsyncClient) -> None:
    resp = await client.delete(f"/v1/agents/{AGENT}/memories/{USER}/mem_does_not_exist")
    assert resp.status_code == 404


# -- correct ------------------------------------------------------------------


async def test_correct_returns_ok(client: AsyncClient) -> None:
    resp = await client.post(
        f"/v1/agents/{AGENT}/correct/{USER}",
        json={"content": "Do not summarize, give the full answer."},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is True
    assert body["memory_id"] is not None


async def test_correct_stores_memory(client: AsyncClient) -> None:
    content = "Always cite your sources."
    await client.post(
        f"/v1/agents/{AGENT}/correct/{USER}",
        json={"content": content},
    )
    memories = (await client.get(f"/v1/agents/{AGENT}/memories/{USER}")).json()
    assert any(content in m["content"] for m in memories)


async def test_correct_without_session_does_not_raise(client: AsyncClient) -> None:
    resp = await client.post(
        f"/v1/agents/{AGENT}/correct/{USER}",
        json={"content": "Be more specific.", "session_id": None},
    )
    assert resp.status_code == 200


# -- reinforce ----------------------------------------------------------------


async def test_reinforce_without_session_returns_not_applied(client: AsyncClient) -> None:
    resp = await client.post(
        f"/v1/agents/{AGENT}/reinforce/{USER}",
        json={},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is True
    assert body["applied"] is False


async def test_reinforce_with_invalid_session_returns_404(client: AsyncClient) -> None:
    resp = await client.post(
        f"/v1/agents/{AGENT}/reinforce/{USER}",
        json={"session_id": "sess_does_not_exist"},
    )
    assert resp.status_code == 404
