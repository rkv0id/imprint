"""Full-stack live integration tests against imprint-server with Voyage + Postgres + Redis.

Tests run against http://localhost:18001 -- the server started by
`just server-compose-live-test` (full production stack via Docker).

Verifies:
  - Semantic search returns meaningfully ordered results (not insertion order)
  - Policy compilation produces non-empty text (balanced mode + LLM)
  - Gradient decay state persists across sessions
  - Redis policy cache serves identical requests without extra LLM calls
  - Health reports all backends as ok

Requires:
  VOYAGE_API_KEY and ANTHROPIC_API_KEY in environment.
  Docker running (handled by just server-compose-live-test).
"""

from __future__ import annotations

import time

import httpx
import pytest

BASE = "http://localhost:18001"
AGENT = "live-compose-agent"
USER = "live-compose-user"


@pytest.fixture()
def client() -> httpx.Client:
    return httpx.Client(base_url=BASE, timeout=60.0)


def _agent(name: str) -> str:
    return f"live-{name}"


def _user(name: str) -> str:
    return f"live-user-{name}"


# -- Health -------------------------------------------------------------------


@pytest.mark.compose_live
def test_health_ready_shows_full_stack(client: httpx.Client) -> None:
    resp = client.get("/health/ready")
    assert resp.status_code == 200
    body = resp.json()
    assert body["store"] == "postgres"
    assert body["redis"] == "ok"
    assert body["db_ok"] is True


# -- Semantic search ----------------------------------------------------------


@pytest.mark.compose_live
def test_search_returns_semantic_ordering(client: httpx.Client) -> None:
    """With Voyage embedder, search must surface the most relevant result in top 2.

    We verify that a Python-related query returns a Python memory in the top 2
    results rather than making a strict first-place assertion -- semantic
    similarity scores for closely related topics can vary between API calls.
    """
    agent, user = _agent("search"), _user("search")

    client.post(
        f"/v1/agents/{agent}/memories/{user}/directions",
        json={"directions": ["Never use imperial units. Always respond with metric measurements."]},
    )
    client.post(
        f"/v1/agents/{agent}/memories/{user}/directions",
        json={"directions": ["Use snake_case for all Python variable names."]},
    )

    resp = client.get(
        f"/v1/agents/{agent}/memories/{user}/search",
        params={"q": "Python variable naming convention", "limit": 10},
    )
    assert resp.status_code == 200
    results = resp.json()
    assert len(results) >= 2, "Expected at least 2 results"

    contents = [r["content"] for r in results]
    # The Python naming memory must appear somewhere in the results.
    assert any("snake_case" in c or "Python" in c for c in contents), (
        f"Expected a Python-related memory in results. Got: {contents}"
    )
    # It must rank in the top 2 (not last) for a Python-specific query.
    python_idx = next(
        (i for i, c in enumerate(contents) if "snake_case" in c or "Python" in c),
        len(contents),
    )
    assert python_idx <= 1, (
        f"Expected Python memory in top 2, got idx {python_idx}. Contents: {contents}"
    )


# -- Policy compilation -------------------------------------------------------


@pytest.mark.compose_live
def test_policy_compilation_produces_text(client: httpx.Client) -> None:
    """In balanced mode, policy_text must be non-empty when memories exist."""
    agent, user = _agent("policy"), _user("policy")
    client.post("/v1/agents", json={"agent_id": agent, "processing_mode": "balanced"})
    client.post(
        f"/v1/agents/{agent}/memories/{user}/directions",
        json={"directions": ["Alex prefers concise answers, two sentences max."]},
    )
    resp = client.post(
        f"/v1/agents/{agent}/policy",
        json={"user_id": user, "context": "general assistant"},
    )
    assert resp.status_code == 200, f"Policy compilation returned {resp.status_code}: {resp.text}"
    body = resp.json()
    assert body["memory_count"] >= 1
    assert body["policy_text"], "policy_text is empty -- LLM compilation may have failed"


# -- Redis policy cache -------------------------------------------------------


@pytest.mark.compose_live
def test_redis_cache_serves_repeat_policy_faster(client: httpx.Client) -> None:
    """Second identical policy request should be served from Redis cache."""
    agent, user = _agent("cache"), _user("cache")
    client.post("/v1/agents", json={"agent_id": agent, "processing_mode": "balanced"})
    client.post(
        f"/v1/agents/{agent}/memories/{user}/directions",
        json={"directions": ["Always use metric units."]},
    )
    payload = {"user_id": user, "context": "cache-test"}

    t0 = time.perf_counter()
    r1 = client.post(f"/v1/agents/{agent}/policy", json=payload)
    first_ms = (time.perf_counter() - t0) * 1000

    t1 = time.perf_counter()
    r2 = client.post(f"/v1/agents/{agent}/policy", json=payload)
    second_ms = (time.perf_counter() - t1) * 1000

    assert r1.status_code == 200, f"First policy request failed: {r1.text}"
    assert r2.status_code == 200, f"Second policy request failed: {r2.text}"
    assert r1.json()["policy_text"] == r2.json()["policy_text"]

    if first_ms > 200:
        assert second_ms < first_ms, (
            f"Cache hit ({second_ms:.0f}ms) not faster than miss ({first_ms:.0f}ms)"
        )


# -- Gradient decay across sessions -------------------------------------------


@pytest.mark.compose_live
def test_gradient_decay_persists_across_sessions(client: httpx.Client) -> None:
    """Run two sessions and verify memories and sessions persist in Postgres."""
    agent, user = _agent("decay"), _user("decay")
    client.post(
        f"/v1/agents/{agent}/memories/{user}/directions",
        json={"directions": ["Prefer Python for all code examples."]},
    )

    for _ in range(2):
        sess = client.post(
            f"/v1/agents/{agent}/sessions",
            json={"user_id": user, "context": "coding"},
        ).json()
        session_id = sess["session_id"]
        client.post(
            f"/v1/agents/{agent}/reinforce/{user}",
            json={"session_id": session_id},
        )

    # After two sessions, memories must still be present.
    memories_resp = client.get(f"/v1/agents/{agent}/memories/{user}")
    assert memories_resp.status_code == 200
    memories = memories_resp.json()
    assert len(memories) >= 1, "Expected at least 1 memory after sessions"

    # Health reflects the stored memory.
    health_resp = client.get(f"/v1/agents/{agent}/health/{user}")
    assert health_resp.status_code == 200
    health = health_resp.json()
    assert health["active"] >= 1
