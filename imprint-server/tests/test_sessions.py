"""Integration tests for the session endpoints.

Uses real SQLite store in frugal mode. Tests the full open -> observe ->
policy -> close lifecycle, including the learning signal path on close.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient

from imprint_server.app import create_app
from imprint_server.config import ServerConfig
from imprint_server.registry import AgentRegistry

AGENT = "session-test-agent"
USER = "session-test-user"


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


# -- Open session -------------------------------------------------------------


async def test_open_session_returns_session_id(client: AsyncClient) -> None:
    resp = await client.post(
        f"/v1/agents/{AGENT}/sessions",
        json={"user_id": USER},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert "session_id" in body
    assert body["session_id"].startswith("sess_")


async def test_open_session_with_context(client: AsyncClient) -> None:
    resp = await client.post(
        f"/v1/agents/{AGENT}/sessions",
        json={"user_id": USER, "context": "code review"},
    )
    assert resp.status_code == 200
    assert resp.json()["session_id"].startswith("sess_")


async def test_multiple_sessions_have_distinct_ids(client: AsyncClient) -> None:
    r1 = await client.post(f"/v1/agents/{AGENT}/sessions", json={"user_id": USER})
    r2 = await client.post(f"/v1/agents/{AGENT}/sessions", json={"user_id": USER})
    assert r1.json()["session_id"] != r2.json()["session_id"]


# -- Observe within session ---------------------------------------------------


async def test_session_observe_turn_returns_ok(client: AsyncClient) -> None:
    open_resp = await client.post(f"/v1/agents/{AGENT}/sessions", json={"user_id": USER})
    sid = open_resp.json()["session_id"]

    resp = await client.post(
        f"/v1/agents/{AGENT}/sessions/{sid}/observe",
        json={
            "agent_output": "Here is a bullet list.",
            "user_response": "No, stop using bullet points.",
        },
    )
    assert resp.status_code == 200
    assert resp.json()["ok"] is True


async def test_session_observe_directions_returns_ok(client: AsyncClient) -> None:
    open_resp = await client.post(f"/v1/agents/{AGENT}/sessions", json={"user_id": USER})
    sid = open_resp.json()["session_id"]

    resp = await client.post(
        f"/v1/agents/{AGENT}/sessions/{sid}/observe",
        json={"directions": ["Always use conventional commits."]},
    )
    assert resp.status_code == 200


async def test_session_observe_unknown_session_returns_404(
    client: AsyncClient,
) -> None:
    resp = await client.post(
        f"/v1/agents/{AGENT}/sessions/sess_doesnotexist/observe",
        json={"agent_output": "x", "user_response": "y"},
    )
    assert resp.status_code == 404


async def test_session_observe_wrong_agent_returns_404(client: AsyncClient) -> None:
    open_resp = await client.post(f"/v1/agents/{AGENT}/sessions", json={"user_id": USER})
    sid = open_resp.json()["session_id"]

    resp = await client.post(
        f"/v1/agents/wrong-agent/sessions/{sid}/observe",
        json={"agent_output": "x", "user_response": "y"},
    )
    assert resp.status_code == 404


# -- Policy within session ----------------------------------------------------


async def test_session_policy_returns_policy(client: AsyncClient) -> None:
    open_resp = await client.post(f"/v1/agents/{AGENT}/sessions", json={"user_id": USER})
    sid = open_resp.json()["session_id"]

    resp = await client.post(
        f"/v1/agents/{AGENT}/sessions/{sid}/policy",
        json={},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert "policy_text" in body
    assert "memory_count" in body
    assert "compiled_at" in body
    assert "memory_ids" in body
    assert isinstance(body["memory_ids"], list)


async def test_session_policy_logs_policy_event(client: AsyncClient, tmp_path: Path) -> None:
    open_resp = await client.post(f"/v1/agents/{AGENT}/sessions", json={"user_id": USER})
    sid = open_resp.json()["session_id"]

    await client.post(
        f"/v1/agents/{AGENT}/sessions/{sid}/policy",
        json={"context": "testing"},
    )

    import aiosqlite

    db_path = str(tmp_path / "test.db")
    async with (
        aiosqlite.connect(db_path) as conn,
        conn.execute("SELECT session_id FROM policy_events WHERE agent_id = ?", (AGENT,)) as cursor,
    ):
        row = await cursor.fetchone()
    assert row is not None
    assert row[0] == sid


async def test_session_policy_updates_session_row(client: AsyncClient, tmp_path: Path) -> None:
    open_resp = await client.post(
        f"/v1/agents/{AGENT}/sessions",
        json={"user_id": USER, "context": "coding"},
    )
    sid = open_resp.json()["session_id"]

    await client.post(
        f"/v1/agents/{AGENT}/sessions/{sid}/policy",
        json={"context": "coding"},
    )

    import aiosqlite

    db_path = str(tmp_path / "test.db")
    async with (
        aiosqlite.connect(db_path) as conn,
        conn.execute("SELECT context FROM sessions WHERE id = ?", (sid,)) as cursor,
    ):
        row = await cursor.fetchone()
    assert row is not None
    assert row[0] == "coding"


# -- Close session ------------------------------------------------------------


async def test_close_session_returns_ok(client: AsyncClient) -> None:
    open_resp = await client.post(f"/v1/agents/{AGENT}/sessions", json={"user_id": USER})
    sid = open_resp.json()["session_id"]

    resp = await client.post(
        f"/v1/agents/{AGENT}/sessions/{sid}/close",
        json={},
    )
    assert resp.status_code == 200
    assert resp.json()["ok"] is True


async def test_close_session_with_outcome(client: AsyncClient) -> None:
    open_resp = await client.post(f"/v1/agents/{AGENT}/sessions", json={"user_id": USER})
    sid = open_resp.json()["session_id"]

    resp = await client.post(
        f"/v1/agents/{AGENT}/sessions/{sid}/close",
        json={"outcome": 0.8},
    )
    assert resp.status_code == 200


async def test_close_session_marks_closed_in_db(client: AsyncClient, tmp_path: Path) -> None:
    open_resp = await client.post(f"/v1/agents/{AGENT}/sessions", json={"user_id": USER})
    sid = open_resp.json()["session_id"]

    await client.post(
        f"/v1/agents/{AGENT}/sessions/{sid}/close",
        json={"outcome": 1.0},
    )

    import aiosqlite

    db_path = str(tmp_path / "test.db")
    async with (
        aiosqlite.connect(db_path) as conn,
        conn.execute("SELECT closed_at, outcome FROM sessions WHERE id = ?", (sid,)) as cursor,
    ):
        row = await cursor.fetchone()
    assert row is not None
    assert row[0] is not None  # closed_at set
    assert abs(float(row[1]) - 1.0) < 0.001


async def test_double_close_returns_422(client: AsyncClient) -> None:
    open_resp = await client.post(f"/v1/agents/{AGENT}/sessions", json={"user_id": USER})
    sid = open_resp.json()["session_id"]

    await client.post(f"/v1/agents/{AGENT}/sessions/{sid}/close", json={})
    resp2 = await client.post(f"/v1/agents/{AGENT}/sessions/{sid}/close", json={})
    assert resp2.status_code == 422


async def test_close_unknown_session_returns_404(client: AsyncClient) -> None:
    resp = await client.post(
        f"/v1/agents/{AGENT}/sessions/sess_doesnotexist/close",
        json={},
    )
    assert resp.status_code == 404


# -- Full lifecycle -----------------------------------------------------------


async def test_full_session_lifecycle(client: AsyncClient, tmp_path: Path) -> None:
    """Open -> observe (neutral turn) -> policy (empty, no LLM) -> close with outcome."""
    # Open
    open_resp = await client.post(
        f"/v1/agents/{AGENT}/sessions",
        json={"user_id": USER, "context": "code review"},
    )
    assert open_resp.status_code == 200
    sid = open_resp.json()["session_id"]

    # Observe a neutral turn -- "looks good" has no correction markers so the
    # frugal heuristic will not fire and no memory is stored.
    obs_resp = await client.post(
        f"/v1/agents/{AGENT}/sessions/{sid}/observe",
        json={
            "agent_output": "Here is the summary.",
            "user_response": "Looks good.",
        },
    )
    assert obs_resp.status_code == 200

    # Policy -- no memories stored, so get_policy() returns empty without any
    # LLM call (the compiler is only invoked when there are memories to compile).
    pol_resp = await client.post(
        f"/v1/agents/{AGENT}/sessions/{sid}/policy",
        json={"context": "code review"},
    )
    assert pol_resp.status_code == 200
    assert pol_resp.json()["memory_count"] == 0

    # Close without outcome -- finalize_loop() returns immediately when
    # outcome is None, so no LLM attribution calls are made.
    close_resp = await client.post(
        f"/v1/agents/{AGENT}/sessions/{sid}/close",
        json={},
    )
    assert close_resp.status_code == 200

    # Session is now closed in DB.
    import aiosqlite

    db_path = str(tmp_path / "test.db")
    async with (
        aiosqlite.connect(db_path) as conn,
        conn.execute("SELECT closed_at FROM sessions WHERE id = ?", (sid,)) as cursor,
    ):
        row = await cursor.fetchone()
    assert row is not None and row[0] is not None
