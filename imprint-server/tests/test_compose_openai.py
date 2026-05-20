"""Live integration tests against imprint-server with OpenAI embedder + Postgres + Redis.

Tests run against http://localhost:18002 -- the server started by
`just server-compose-openai-test` (full stack with OpenAI embeddings).

Verifies that the Docker image ships the openai extra and the OpenAI code
path works end-to-end: embedding, pgvector storage, semantic search,
and LLM policy compilation all using OpenAI models.

Requires:
  OPENAI_API_KEY and ANTHROPIC_API_KEY in environment.
  Docker running (handled by just server-compose-openai-test).
"""

from __future__ import annotations

import httpx
import pytest

BASE = "http://localhost:18002"


@pytest.fixture()
def client() -> httpx.Client:
    return httpx.Client(base_url=BASE, timeout=60.0)


def _agent(name: str) -> str:
    return f"openai-{name}"


def _user(name: str) -> str:
    return f"openai-user-{name}"


@pytest.mark.compose_openai
def test_health_ready_shows_openai_stack(client: httpx.Client) -> None:
    """Server must report Postgres + Redis healthy with OpenAI embedder configured."""
    resp = client.get("/health/ready")
    assert resp.status_code == 200
    body = resp.json()
    assert body["store"] == "postgres"
    assert body["redis"] == "ok"
    assert body["db_ok"] is True


@pytest.mark.compose_openai
def test_openai_embedder_stores_vector(client: httpx.Client) -> None:
    """observe_directions must embed with OpenAI and upsert into pgvector."""
    agent, user = _agent("embed"), _user("embed")
    resp = client.post(
        f"/v1/agents/{agent}/memories/{user}/directions",
        json={"directions": ["Explain machine learning concepts with simple analogies."]},
    )
    assert resp.status_code == 200, f"observe_directions failed: {resp.text}"
    assert resp.json()["stored"] == 1

    memories = client.get(f"/v1/agents/{agent}/memories/{user}").json()
    assert len(memories) == 1


@pytest.mark.compose_openai
def test_openai_semantic_search_ordering(client: httpx.Client) -> None:
    """With OpenAI embedder, search must surface the semantically closest result first."""
    agent, user = _agent("search"), _user("search")

    client.post(
        f"/v1/agents/{agent}/memories/{user}/directions",
        json={"directions": ["Use snake_case for all Python variable names."]},
    )
    client.post(
        f"/v1/agents/{agent}/memories/{user}/directions",
        json={"directions": ["Never use imperial units. Always prefer metric."]},
    )

    resp = client.get(
        f"/v1/agents/{agent}/memories/{user}/search",
        params={"q": "Python naming conventions", "limit": 10},
    )
    assert resp.status_code == 200
    results = resp.json()
    assert len(results) >= 2

    contents = [r["content"] for r in results]
    python_idx = next(
        (i for i, c in enumerate(contents) if "snake_case" in c or "Python" in c),
        len(contents),
    )
    assert python_idx <= 1, (
        f"Expected Python memory in top 2, got idx {python_idx}. Contents: {contents}"
    )


@pytest.mark.compose_openai
def test_openai_policy_compilation(client: httpx.Client) -> None:
    """Balanced mode policy must compile with OpenAI embeddings + Anthropic LLM."""
    agent, user = _agent("policy"), _user("policy")
    client.post("/v1/agents", json={"agent_id": agent, "processing_mode": "balanced"})
    client.post(
        f"/v1/agents/{agent}/memories/{user}/directions",
        json={"directions": ["Sam prefers numbered lists over bullet points."]},
    )
    resp = client.post(
        f"/v1/agents/{agent}/policy",
        json={"user_id": user, "context": "formatting preferences"},
    )
    assert resp.status_code == 200, f"Policy compilation failed: {resp.text}"
    body = resp.json()
    assert body["memory_count"] >= 1
    assert body["policy_text"], "policy_text is empty -- LLM compilation may have failed"


@pytest.mark.compose_openai
def test_openai_vector_store_schema_initialized(client: httpx.Client) -> None:
    """memory_vectors table must exist with 1536-dim OpenAI vectors."""
    agent, user = _agent("schema"), _user("schema")
    store = client.post(
        f"/v1/agents/{agent}/memories/{user}/directions",
        json={"directions": ["Prefer async/await over callbacks in JavaScript."]},
    )
    assert store.status_code == 200

    resp = client.get(
        f"/v1/agents/{agent}/memories/{user}/search",
        params={"q": "JavaScript async programming", "limit": 5},
    )
    assert resp.status_code == 200
    results = resp.json()
    assert len(results) >= 1
    assert any("async" in r["content"].lower() or "JavaScript" in r["content"] for r in results)
