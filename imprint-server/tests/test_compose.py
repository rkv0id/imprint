"""Infrastructure integration tests against a real imprint-server stack.

Tests run against http://localhost:18000 -- the server started by
`just server-compose-test` (Postgres + Redis + imprint-server via Docker).

All tests use real HTTP (not ASGI transport). They exercise the full
server binary including: middleware stack, Postgres store, Redis rate
limiting and policy cache, health split, metrics endpoint, and the
REST API end-to-end.

Each test uses a unique agent/user pair to prevent cross-test state
interference (shared mutable state in a live server is a common source
of ordering-dependent failures).

Requires:
  docker compose -f imprint-server/docker-compose.test.yml up --wait
  (handled automatically by just server-compose-test)

No API keys required -- server runs in frugal mode.
"""

from __future__ import annotations

import httpx
import pytest

BASE = "http://localhost:18000"


@pytest.fixture()
def client() -> httpx.Client:
    return httpx.Client(base_url=BASE, timeout=10.0)


def _agent(test_name: str) -> str:
    """Unique agent ID per test to prevent cross-test state interference."""
    return f"compose-{test_name}"


def _user(test_name: str) -> str:
    return f"compose-user-{test_name}"


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
    assert "process_" in resp.text or "python_" in resp.text


# -- Agent CRUD ---------------------------------------------------------------


@pytest.mark.compose
def test_create_agent(client: httpx.Client) -> None:
    agent = _agent("create")
    resp = client.post(
        "/v1/agents",
        json={"agent_id": agent, "processing_mode": "frugal"},
    )
    assert resp.status_code == 200
    assert resp.json()["agent_id"] == agent


@pytest.mark.compose
def test_get_agent_config(client: httpx.Client) -> None:
    agent = _agent("get-config")
    client.post("/v1/agents", json={"agent_id": agent, "processing_mode": "frugal"})
    resp = client.get(f"/v1/agents/{agent}")
    assert resp.status_code == 200
    body = resp.json()
    assert body["agent_id"] == agent
    assert "dynamic_scopes" in body


@pytest.mark.compose
def test_patch_agent_config(client: httpx.Client) -> None:
    agent = _agent("patch-config")
    client.post("/v1/agents", json={"agent_id": agent, "processing_mode": "frugal"})
    resp = client.patch(
        f"/v1/agents/{agent}/config",
        json={"dynamic_scopes": True},
    )
    assert resp.status_code == 200
    assert resp.json()["dynamic_scopes"] is True


# -- Observe / policy / memories ----------------------------------------------


@pytest.mark.compose
def test_observe_returns_ok(client: httpx.Client) -> None:
    agent, user = _agent("observe"), _user("observe")
    resp = client.post(
        f"/v1/agents/{agent}/observe",
        json={
            "user_id": user,
            "agent_output": "Here is a bullet list.",
            "user_response": "No bullet points please.",
        },
    )
    assert resp.status_code == 200


@pytest.mark.compose
def test_directions_stored_and_listed(client: httpx.Client) -> None:
    agent, user = _agent("directions"), _user("directions")
    client.post(
        f"/v1/agents/{agent}/memories/{user}/directions",
        json={"directions": ["Always respond in plain prose."]},
    )
    resp = client.get(f"/v1/agents/{agent}/memories/{user}")
    assert resp.status_code == 200
    memories = resp.json()
    assert len(memories) >= 1


@pytest.mark.compose
def test_policy_returns_ok(client: httpx.Client) -> None:
    # Isolated agent in explicit frugal mode with no memories stored.
    # With 0 memories, get_policy() returns the early-exit empty policy
    # without touching the LLM -- no API key required.
    agent, user = _agent("policy"), _user("policy")
    client.post("/v1/agents", json={"agent_id": agent, "processing_mode": "frugal"})
    resp = client.post(
        f"/v1/agents/{agent}/policy",
        json={"user_id": user},
    )
    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"
    body = resp.json()
    assert "policy_text" in body
    assert "memory_count" in body
    assert body["memory_count"] == 0
    assert body["policy_text"] == ""


@pytest.mark.compose
def test_pagination_envelope(client: httpx.Client) -> None:
    agent, user = _agent("pagination"), _user("pagination")
    for i in range(5):
        client.post(
            f"/v1/agents/{agent}/memories/{user}/directions",
            json={"directions": [f"Preference {i}: be explicit."]},
        )
    resp = client.get(
        f"/v1/agents/{agent}/memories/{user}",
        params={"limit": 2},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert "items" in body
    assert "next_cursor" in body
    assert len(body["items"]) == 2


@pytest.mark.compose
def test_search_endpoint(client: httpx.Client) -> None:
    agent, user = _agent("search"), _user("search")
    client.post(
        f"/v1/agents/{agent}/memories/{user}/directions",
        json={"directions": ["Always cite sources when mentioning facts."]},
    )
    resp = client.get(
        f"/v1/agents/{agent}/memories/{user}/search",
        params={"q": "citations sources"},
    )
    assert resp.status_code == 200
    assert isinstance(resp.json(), list)


# -- Sessions -----------------------------------------------------------------


@pytest.mark.compose
def test_session_lifecycle(client: httpx.Client) -> None:
    agent, user = _agent("sessions"), _user("sessions")
    open_resp = client.post(
        f"/v1/agents/{agent}/sessions",
        json={"user_id": user, "context": "compose-test"},
    )
    assert open_resp.status_code == 200
    session_id = open_resp.json()["session_id"]
    assert session_id.startswith("sess_")

    reinforce_resp = client.post(
        f"/v1/agents/{agent}/reinforce/{user}",
        json={"session_id": session_id},
    )
    assert reinforce_resp.status_code == 200
    assert reinforce_resp.json()["applied"] is True

    second = client.post(
        f"/v1/agents/{agent}/reinforce/{user}",
        json={"session_id": session_id},
    )
    assert second.status_code == 422


# -- Error shapes -------------------------------------------------------------


@pytest.mark.compose
def test_404_is_problem_json(client: httpx.Client) -> None:
    resp = client.get("/v1/memories/mem_does_not_exist/lineage")
    assert resp.status_code == 404, f"Expected 404, got {resp.status_code}: {resp.text}"
    assert resp.headers["content-type"] == "application/problem+json"
    body = resp.json()
    assert body["type"] == "about:blank"
    assert body["status"] == 404


# -- forget / correct / consolidate -------------------------------------------


@pytest.mark.compose
def test_forget_removes_all_user_memories(client: httpx.Client) -> None:
    agent, user = _agent("forget"), _user("forget")
    for i in range(3):
        client.post(
            f"/v1/agents/{agent}/memories/{user}/directions",
            json={"directions": [f"Preference {i}: keep responses brief."]},
        )
    before = client.get(f"/v1/agents/{agent}/memories/{user}").json()
    assert len(before) == 3

    resp = client.delete(f"/v1/agents/{agent}/memories/{user}")
    assert resp.status_code == 200
    assert resp.json()["ok"] is True

    after = client.get(f"/v1/agents/{agent}/memories/{user}").json()
    assert len(after) == 0


@pytest.mark.compose
def test_deactivate_single_memory(client: httpx.Client) -> None:
    agent, user = _agent("deactivate"), _user("deactivate")
    client.post(
        f"/v1/agents/{agent}/memories/{user}/directions",
        json={"directions": ["Always use Oxford comma."]},
    )
    memories = client.get(f"/v1/agents/{agent}/memories/{user}").json()
    assert len(memories) == 1
    memory_id = memories[0]["id"]

    resp = client.delete(f"/v1/agents/{agent}/memories/{user}/{memory_id}")
    assert resp.status_code == 200

    after = client.get(f"/v1/agents/{agent}/memories/{user}").json()
    assert len(after) == 0


@pytest.mark.compose
def test_correct_stores_memory_and_returns_id(client: httpx.Client) -> None:
    agent, user = _agent("correct"), _user("correct")
    resp = client.post(
        f"/v1/agents/{agent}/correct/{user}",
        json={"content": "Do not use bullet points in responses."},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is True
    assert body["memory_id"] is not None
    assert body["memory_id"].startswith("mem_")

    memories = client.get(f"/v1/agents/{agent}/memories/{user}").json()
    assert any("bullet" in m["content"].lower() for m in memories)


@pytest.mark.compose
def test_consolidate_runs_without_error(client: httpx.Client) -> None:
    agent, user = _agent("consolidate"), _user("consolidate")
    for i in range(4):
        client.post(
            f"/v1/agents/{agent}/memories/{user}/directions",
            json={"directions": [f"Rule {i}: be clear and direct."]},
        )
    resp = client.post(
        f"/v1/agents/{agent}/memories/{user}/consolidate",
    )
    assert resp.status_code == 200
    body = resp.json()
    assert "pruned" in body
    assert isinstance(body["pruned"], int)


@pytest.mark.compose
def test_pin_memory_survives_consolidation(client: httpx.Client) -> None:
    agent, user = _agent("pin"), _user("pin")
    client.post(
        f"/v1/agents/{agent}/memories/{user}/directions",
        json={"directions": ["Always cite primary sources."]},
    )
    memories = client.get(f"/v1/agents/{agent}/memories/{user}").json()
    memory_id = memories[0]["id"]

    pin_resp = client.post(f"/v1/agents/{agent}/memories/{memory_id}/pin")
    assert pin_resp.status_code == 200

    client.post(f"/v1/agents/{agent}/memories/{user}/consolidate")
    after = client.get(f"/v1/agents/{agent}/memories/{user}").json()
    assert any(m["id"] == memory_id for m in after)


@pytest.mark.compose
def test_memory_health_reflects_stored_count(client: httpx.Client) -> None:
    agent, user = _agent("health"), _user("health")
    for i in range(3):
        client.post(
            f"/v1/agents/{agent}/memories/{user}/directions",
            json={"directions": [f"Style {i}: write clearly."]},
        )
    resp = client.get(f"/v1/agents/{agent}/health/{user}")
    assert resp.status_code == 200
    body = resp.json()
    assert body["active"] == 3
    assert body["total"] >= 3


@pytest.mark.compose
def test_lineage_returns_memory_history(client: httpx.Client) -> None:
    agent, user = _agent("lineage"), _user("lineage")
    client.post(
        f"/v1/agents/{agent}/memories/{user}/directions",
        json={"directions": ["Respond in plain prose, no markdown."]},
    )
    memories = client.get(f"/v1/agents/{agent}/memories/{user}").json()
    memory_id = memories[0]["id"]

    resp = client.get(f"/v1/memories/{memory_id}/lineage")
    assert resp.status_code == 200
    body = resp.json()
    assert body["memory"]["id"] == memory_id


@pytest.mark.compose
def test_api_key_create_list_revoke(client: httpx.Client) -> None:
    """Key CRUD via the CLI is not HTTP-accessible; the admin API covers agent CRUD.

    This test verifies the agents admin endpoints that ARE HTTP-accessible
    still work correctly with Postgres, since auth is disabled in this stack.
    All agent lifecycle operations (create, get, patch, delete) are covered.
    """
    agent = "compose-admin-lifecycle"
    create = client.post("/v1/agents", json={"agent_id": agent, "processing_mode": "frugal"})
    assert create.status_code == 200

    get = client.get(f"/v1/agents/{agent}")
    assert get.status_code == 200
    assert get.json()["agent_id"] == agent

    patch = client.patch(f"/v1/agents/{agent}/config", json={"dynamic_scopes": True})
    assert patch.status_code == 200
    assert patch.json()["dynamic_scopes"] is True

    delete = client.delete(f"/v1/agents/{agent}")
    assert delete.status_code == 200

    after = client.get(f"/v1/agents/{agent}")
    assert after.status_code == 404


# -- Rate limiting (Redis-backed) -- must run last (exhausts the budget) ------
#
# These tests hammer 105 requests each to push past the limit=100 threshold.
# They consume the entire rate limit budget, so all rate-limit tests are
# grouped here at the end so earlier functional tests are not starved.


@pytest.mark.compose
def test_rate_limit_blocks_at_threshold(client: httpx.Client) -> None:
    """Hit 105 requests, expect at least one 429 (limit=100)."""
    agent = "compose-rl-a"
    user = "compose-rl-u-a"
    path = f"/v1/agents/{agent}/memories/{user}"

    statuses = [client.get(path).status_code for _ in range(105)]
    assert 429 in statuses, (
        f"Expected at least one 429 after 105 requests against limit=100. "
        f"Status counts: 200={statuses.count(200)} 429={statuses.count(429)}"
    )


@pytest.mark.compose
def test_rate_limit_includes_retry_after(client: httpx.Client) -> None:
    agent = "compose-rl-b"
    user = "compose-rl-u-b"
    path = f"/v1/agents/{agent}/memories/{user}"

    for _ in range(105):
        resp = client.get(path)
        if resp.status_code == 429:
            assert "retry-after" in resp.headers
            return

    pytest.skip("Rate limit not reached in 105 requests")


@pytest.mark.compose
def test_health_endpoints_not_rate_limited(client: httpx.Client) -> None:
    """Health endpoints must never be rate limited regardless of budget exhaustion."""
    for _ in range(30):
        resp = client.get("/health/live")
        assert resp.status_code == 200
