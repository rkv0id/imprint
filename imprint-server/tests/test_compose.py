"""Infrastructure integration tests against a real imprint-server stack.

Tests run against http://localhost:18000 -- the server started by
`just server-compose-test` (Postgres + Redis + imprint-server via Docker).

All tests use real HTTP (not ASGI transport). They exercise the full
server binary including: middleware stack, Postgres store, Redis rate
limiting and policy cache, health split, metrics endpoint, and the
REST API end-to-end.

Requires:
  docker compose -f imprint-server/docker-compose.test.yml up --wait
  (handled automatically by just server-compose-test)

No API keys required -- server runs in frugal mode.
"""

from __future__ import annotations

import httpx
import pytest

BASE = "http://localhost:18000"
AGENT = "compose-agent"
USER = "compose-user"


@pytest.fixture()
def client() -> httpx.Client:
    return httpx.Client(base_url=BASE, timeout=10.0)


# -- Health -------------------------------------------------------------------


@pytest.mark.compose
def test_health_live(client: httpx.Client) -> None:
    resp = client.get("/health/live")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


@pytest.mark.compose
def test_health_ready_shows_postgres_and_redis(client: httpx.Client) -> None:
    resp = client.get("/health/ready")
    assert resp.status_code == 200
    body = resp.json()
    assert body["store"] == "postgres"
    assert body["redis"] == "ok"
    assert body["db_ok"] is True


@pytest.mark.compose
def test_health_alias_works(client: httpx.Client) -> None:
    resp = client.get("/health")
    assert resp.status_code == 200
    assert "status" in resp.json()


@pytest.mark.compose
def test_request_id_header_present(client: httpx.Client) -> None:
    resp = client.get("/health/live")
    assert "x-request-id" in resp.headers


@pytest.mark.compose
def test_metrics_endpoint(client: httpx.Client) -> None:
    resp = client.get("/metrics")
    assert resp.status_code == 200
    assert "text/plain" in resp.headers["content-type"]
    # Prometheus format: at minimum the process metrics are present.
    assert "process_" in resp.text or "python_" in resp.text


# -- Agent CRUD ---------------------------------------------------------------


@pytest.mark.compose
def test_create_agent(client: httpx.Client) -> None:
    resp = client.post(
        "/v1/agents",
        json={"agent_id": AGENT, "processing_mode": "frugal"},
    )
    assert resp.status_code == 200
    assert resp.json()["agent_id"] == AGENT


@pytest.mark.compose
def test_get_agent_config(client: httpx.Client) -> None:
    client.post("/v1/agents", json={"agent_id": AGENT})
    resp = client.get(f"/v1/agents/{AGENT}")
    assert resp.status_code == 200
    body = resp.json()
    assert body["agent_id"] == AGENT
    assert "dynamic_scopes" in body


@pytest.mark.compose
def test_patch_agent_config(client: httpx.Client) -> None:
    client.post("/v1/agents", json={"agent_id": AGENT})
    resp = client.patch(
        f"/v1/agents/{AGENT}/config",
        json={"dynamic_scopes": True},
    )
    assert resp.status_code == 200
    assert resp.json()["dynamic_scopes"] is True


# -- Observe / policy / memories ----------------------------------------------


@pytest.mark.compose
def test_observe_returns_ok(client: httpx.Client) -> None:
    resp = client.post(
        f"/v1/agents/{AGENT}/observe",
        json={
            "user_id": USER,
            "agent_output": "Here is a bullet list.",
            "user_response": "No bullet points please.",
        },
    )
    assert resp.status_code == 200


@pytest.mark.compose
def test_directions_stored_and_listed(client: httpx.Client) -> None:
    client.post(
        f"/v1/agents/{AGENT}/memories/{USER}/directions",
        json={"directions": ["Always respond in plain prose."]},
    )
    resp = client.get(f"/v1/agents/{AGENT}/memories/{USER}")
    assert resp.status_code == 200
    memories = resp.json()
    assert len(memories) >= 1


@pytest.mark.compose
def test_policy_returns_ok(client: httpx.Client) -> None:
    resp = client.post(
        f"/v1/agents/{AGENT}/policy",
        json={"user_id": USER},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert "policy_text" in body
    assert "memory_count" in body


@pytest.mark.compose
def test_pagination_envelope(client: httpx.Client) -> None:
    for i in range(5):
        client.post(
            f"/v1/agents/{AGENT}/memories/{USER}/directions",
            json={"directions": [f"Preference {i}: be explicit."]},
        )
    resp = client.get(
        f"/v1/agents/{AGENT}/memories/{USER}",
        params={"limit": 2},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert "items" in body
    assert "next_cursor" in body
    assert len(body["items"]) == 2


@pytest.mark.compose
def test_search_endpoint(client: httpx.Client) -> None:
    client.post(
        f"/v1/agents/{AGENT}/memories/{USER}/directions",
        json={"directions": ["Always cite sources when mentioning facts."]},
    )
    resp = client.get(
        f"/v1/agents/{AGENT}/memories/{USER}/search",
        params={"q": "citations sources"},
    )
    assert resp.status_code == 200
    assert isinstance(resp.json(), list)


# -- Sessions -----------------------------------------------------------------


@pytest.mark.compose
def test_session_lifecycle(client: httpx.Client) -> None:
    # Open session.
    open_resp = client.post(
        f"/v1/agents/{AGENT}/sessions",
        json={"user_id": USER, "context": "compose-test"},
    )
    assert open_resp.status_code == 200
    session_id = open_resp.json()["session_id"]
    assert session_id.startswith("sess_")

    # Reinforce closes it.
    reinforce_resp = client.post(
        f"/v1/agents/{AGENT}/reinforce/{USER}",
        json={"session_id": session_id},
    )
    assert reinforce_resp.status_code == 200
    assert reinforce_resp.json()["applied"] is True

    # Second reinforce on closed session returns 422.
    second = client.post(
        f"/v1/agents/{AGENT}/reinforce/{USER}",
        json={"session_id": session_id},
    )
    assert second.status_code == 422


# -- Error shapes -------------------------------------------------------------


@pytest.mark.compose
def test_404_is_problem_json(client: httpx.Client) -> None:
    resp = client.get("/v1/memories/mem_does_not_exist/lineage")
    assert resp.status_code == 404
    assert resp.headers["content-type"] == "application/problem+json"
    body = resp.json()
    assert body["type"] == "about:blank"
    assert body["status"] == 404


# -- Rate limiting (Redis-backed) ---------------------------------------------


@pytest.mark.compose
def test_rate_limit_blocks_at_threshold(client: httpx.Client) -> None:
    """Server is configured with limit=20 per 60s. Hit 21 times, expect 429."""
    # Use a unique agent to avoid interference from other tests.
    rl_agent = "compose-rl-agent"
    rl_user = "compose-rl-user"
    path = f"/v1/agents/{rl_agent}/memories/{rl_user}"

    statuses = []
    for _ in range(25):
        resp = client.get(path)
        statuses.append(resp.status_code)

    assert 429 in statuses, (
        f"Expected at least one 429 in {statuses} after 25 requests "
        f"against a server with rate_limit_requests=20"
    )


@pytest.mark.compose
def test_rate_limit_includes_retry_after(client: httpx.Client) -> None:
    rl_agent = "compose-rl-agent-2"
    rl_user = "compose-rl-user-2"
    path = f"/v1/agents/{rl_agent}/memories/{rl_user}"

    for _ in range(25):
        resp = client.get(path)
        if resp.status_code == 429:
            assert "retry-after" in resp.headers
            return

    pytest.skip("Rate limit not reached in 25 requests -- limit may be higher than expected")


@pytest.mark.compose
def test_health_endpoints_not_rate_limited(client: httpx.Client) -> None:
    for _ in range(30):
        resp = client.get("/health/live")
        assert resp.status_code == 200
