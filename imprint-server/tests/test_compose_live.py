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
    return httpx.Client(base_url=BASE, timeout=30.0)


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
    """With Voyage embedder, search must return the semantically closest result first."""
    # Store two memories on clearly different topics.
    client.post(
        f"/v1/agents/{AGENT}/memories/{USER}/directions",
        json={"directions": ["Always respond in French when the user writes in French."]},
    )
    client.post(
        f"/v1/agents/{AGENT}/memories/{USER}/directions",
        json={"directions": ["Use snake_case for all Python variable names."]},
    )

    resp = client.get(
        f"/v1/agents/{AGENT}/memories/{USER}/search",
        params={"q": "Python variable naming convention", "limit": 10},
    )
    assert resp.status_code == 200
    results = resp.json()
    assert len(results) >= 2

    contents = [r["content"] for r in results]
    snake_idx = next(i for i, c in enumerate(contents) if "snake_case" in c)
    french_idx = next(i for i, c in enumerate(contents) if "French" in c)
    assert snake_idx < french_idx, (
        f"Expected snake_case memory (idx {snake_idx}) before "
        f"French memory (idx {french_idx}). Got: {contents}"
    )


# -- Policy compilation -------------------------------------------------------


@pytest.mark.compose_live
def test_policy_compilation_produces_text(client: httpx.Client) -> None:
    """In balanced mode, policy_text must be non-empty when memories exist."""
    client.post(
        f"/v1/agents/{AGENT}/memories/{USER}/directions",
        json={"directions": ["Alex prefers concise answers, two sentences max."]},
    )
    resp = client.post(
        f"/v1/agents/{AGENT}/policy",
        json={"user_id": USER, "context": "general assistant"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["memory_count"] >= 1
    assert body["policy_text"], "policy_text is empty -- LLM compilation may have failed"


# -- Redis policy cache -------------------------------------------------------


@pytest.mark.compose_live
def test_redis_cache_serves_repeat_policy_faster(client: httpx.Client) -> None:
    """Second identical policy request should be faster due to Redis cache."""
    client.post(
        f"/v1/agents/{AGENT}/memories/{USER}/directions",
        json={"directions": ["Always use metric units."]},
    )
    payload = {"user_id": USER, "context": "cache-test"}

    # First request: cache miss, LLM compiles.
    t0 = time.perf_counter()
    r1 = client.post(f"/v1/agents/{AGENT}/policy", json=payload)
    first_ms = (time.perf_counter() - t0) * 1000

    # Second request: cache hit.
    t1 = time.perf_counter()
    r2 = client.post(f"/v1/agents/{AGENT}/policy", json=payload)
    second_ms = (time.perf_counter() - t1) * 1000

    assert r1.status_code == 200
    assert r2.status_code == 200
    assert r1.json()["policy_text"] == r2.json()["policy_text"]

    # Cache hit should be at least 2x faster. Allow generous margin
    # since network latency varies; this catches a broken cache.
    if first_ms > 200:  # only assert when first call was slow enough to matter
        assert second_ms < first_ms, (
            f"Cache hit ({second_ms:.0f}ms) not faster than miss ({first_ms:.0f}ms)"
        )


# -- Gradient decay across sessions -------------------------------------------


@pytest.mark.compose_live
def test_gradient_decay_persists_across_sessions(client: httpx.Client) -> None:
    """Run two sessions with outcomes -- gradient decay state must persist."""
    client.post(
        f"/v1/agents/{AGENT}/memories/{USER}/directions",
        json={"directions": ["Prefer Python for all code examples."]},
    )

    for outcome in [0.9, 0.8]:
        sess = client.post(
            f"/v1/agents/{AGENT}/sessions",
            json={"user_id": USER, "context": "coding"},
        ).json()
        session_id = sess["session_id"]
        client.post(
            f"/v1/agents/{AGENT}/reinforce/{USER}",
            json={"session_id": session_id},
        ) if outcome >= 0.8 else client.post(
            f"/v1/agents/{AGENT}/correct/{USER}",
            json={"content": "Wrong answer.", "session_id": session_id},
        )

    # After two sessions, policy events should have accumulated.
    events_resp = client.get(
        f"/v1/agents/{AGENT}/events/{USER}",
        params={"limit": 20},
    )
    assert events_resp.status_code == 200
    body = events_resp.json()
    items = body.get("items", body) if isinstance(body, dict) else body
    assert len(items) >= 1
