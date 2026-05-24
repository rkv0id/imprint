"""Tests for the memory diff endpoint.

Uses real SQLite store in frugal mode -- all changes happen synchronously
without LLM calls, giving deterministic before/after state to assert on.

All diff requests use httpx params= dict rather than URL string interpolation.
ISO timestamps contain '+00:00' which must be percent-encoded in query strings
-- params= handles this automatically.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient

from imprint_server.app import create_app
from imprint_server.config import ServerConfig
from imprint_server.registry import AgentRegistry

AGENT = "diff-test-agent"
USER = "diff-test-user"


@pytest.fixture()
async def client(tmp_path: Path) -> AsyncGenerator[AsyncClient, None]:
    config = ServerConfig(
        store=f"sqlite:///{tmp_path / 'diff_test.db'}",
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


def _now_minus(seconds: int) -> str:
    return (datetime.now(UTC) - timedelta(seconds=seconds)).isoformat()


def _now_plus(seconds: int) -> str:
    return (datetime.now(UTC) + timedelta(seconds=seconds)).isoformat()


# -- Basic structure ----------------------------------------------------------


async def test_diff_empty_window_returns_empty(client: AsyncClient) -> None:
    """When no changes happened, all categories are empty lists."""
    resp = await client.get(
        f"/v1/agents/{AGENT}/memories/{USER}/diff",
        params={"since": _now_minus(5)},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["added"] == []
    assert body["deactivated"] == []
    assert body["superseded"] == []
    assert body["summary"] == {"added": 0, "deactivated": 0, "superseded": 0}


async def test_diff_response_has_since_and_until(client: AsyncClient) -> None:
    resp = await client.get(
        f"/v1/agents/{AGENT}/memories/{USER}/diff",
        params={"since": _now_minus(5)},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert "since" in body
    assert "until" in body


async def test_diff_until_defaults_to_now(client: AsyncClient) -> None:
    resp = await client.get(
        f"/v1/agents/{AGENT}/memories/{USER}/diff",
        params={"since": _now_minus(5)},
    )
    assert resp.status_code == 200
    body = resp.json()
    until_dt = datetime.fromisoformat(body["until"])
    assert (datetime.now(UTC) - until_dt).total_seconds() < 5


# -- Added memories -----------------------------------------------------------


async def test_diff_shows_added_memories(client: AsyncClient) -> None:
    """Directions stored after since must appear in added."""
    since = _now_minus(2)

    await client.post(
        f"/v1/agents/{AGENT}/memories/{USER}/directions",
        json={"directions": ["always use prose", "never use bullet points"]},
    )

    resp = await client.get(
        f"/v1/agents/{AGENT}/memories/{USER}/diff",
        params={"since": since},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["summary"]["added"] >= 1
    contents = [m["content"] for m in body["added"]]
    assert any("prose" in c or "bullet" in c for c in contents)


async def test_diff_does_not_show_memories_before_since(client: AsyncClient) -> None:
    """Memories stored before since must not appear in added."""
    # Store the memory first.
    await client.post(
        f"/v1/agents/{AGENT}/memories/{USER}/directions",
        json={"directions": ["this was stored before the window"]},
    )

    # Query a window that started after the memory was stored.
    # Use since=now and until=now+10 so the window is valid but empty.
    resp = await client.get(
        f"/v1/agents/{AGENT}/memories/{USER}/diff",
        params={"since": _now_plus(0), "until": _now_plus(10)},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["added"] == []


# -- Deactivated memories -----------------------------------------------------


async def test_diff_shows_deactivated_memories(client: AsyncClient) -> None:
    """A memory deactivated after since with no replacement must appear in deactivated."""
    await client.post(
        f"/v1/agents/{AGENT}/memories/{USER}/directions",
        json={"directions": ["this memory will be deactivated"]},
    )
    list_resp = await client.get(f"/v1/agents/{AGENT}/memories/{USER}")
    memories = list_resp.json()
    assert len(memories) >= 1
    memory_id = memories[0]["id"]

    since = _now_minus(1)
    await client.delete(f"/v1/agents/{AGENT}/memories/{USER}/{memory_id}")

    resp = await client.get(
        f"/v1/agents/{AGENT}/memories/{USER}/diff",
        params={"since": since},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["summary"]["deactivated"] >= 1
    deactivated_ids = [m["id"] for m in body["deactivated"]]
    assert memory_id in deactivated_ids


# -- Summary correctness ------------------------------------------------------


async def test_diff_summary_matches_list_lengths(client: AsyncClient) -> None:
    """summary counts must equal len(added), len(deactivated), len(superseded)."""
    since = _now_minus(2)
    await client.post(
        f"/v1/agents/{AGENT}/memories/{USER}/directions",
        json={"directions": ["direction one", "direction two"]},
    )
    resp = await client.get(
        f"/v1/agents/{AGENT}/memories/{USER}/diff",
        params={"since": since},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["summary"]["added"] == len(body["added"])
    assert body["summary"]["deactivated"] == len(body["deactivated"])
    assert body["summary"]["superseded"] == len(body["superseded"])


# -- Validation ---------------------------------------------------------------


async def test_diff_missing_since_returns_422(client: AsyncClient) -> None:
    resp = await client.get(f"/v1/agents/{AGENT}/memories/{USER}/diff")
    assert resp.status_code == 422


async def test_diff_invalid_since_returns_422(client: AsyncClient) -> None:
    resp = await client.get(
        f"/v1/agents/{AGENT}/memories/{USER}/diff",
        params={"since": "not-a-date"},
    )
    assert resp.status_code == 422


async def test_diff_since_after_until_returns_422(client: AsyncClient) -> None:
    resp = await client.get(
        f"/v1/agents/{AGENT}/memories/{USER}/diff",
        params={"since": _now_plus(60), "until": _now_minus(60)},
    )
    assert resp.status_code == 422


async def test_diff_explicit_until(client: AsyncClient) -> None:
    resp = await client.get(
        f"/v1/agents/{AGENT}/memories/{USER}/diff",
        params={"since": _now_minus(10), "until": _now_plus(10)},
    )
    assert resp.status_code == 200


# -- OpenAPI ------------------------------------------------------------------


async def test_diff_operation_id_in_schema(client: AsyncClient) -> None:
    resp = await client.get("/openapi.json")
    schema = resp.json()
    op_ids = {
        data.get("operationId") for methods in schema["paths"].values() for data in methods.values()
    }
    assert "memory_diff" in op_ids
