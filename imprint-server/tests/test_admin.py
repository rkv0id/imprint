"""Tests for admin and health endpoints."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient

from imprint_server.app import create_app
from imprint_server.config import ServerConfig
from imprint_server.registry import AgentRegistry

AGENT = "admin-test-agent"


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


# -- health -------------------------------------------------------------------


async def test_health_returns_ok(client: AsyncClient) -> None:
    resp = await client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["db_ok"] is True
    assert body["store"] == "sqlite"


async def test_health_includes_agent_count(client: AsyncClient) -> None:
    resp = await client.get("/health")
    assert "agents_loaded" in resp.json()
    assert resp.json()["agents_loaded"] == 0


async def test_health_agent_count_increments(client: AsyncClient) -> None:
    await client.get(f"/v1/agents/{AGENT}/memories/user1")
    resp = await client.get("/health")
    assert resp.json()["agents_loaded"] == 1


# -- metrics ------------------------------------------------------------------


async def test_metrics_returns_200(client: AsyncClient) -> None:
    resp = await client.get("/metrics")
    assert resp.status_code == 200


async def test_metrics_content_type_is_prometheus(client: AsyncClient) -> None:
    resp = await client.get("/metrics")
    assert "text/plain" in resp.headers["content-type"]


async def test_metrics_contains_imprint_counters(client: AsyncClient) -> None:
    # Trigger an observe to register the counters.
    await client.post(
        f"/v1/agents/{AGENT}/observe",
        json={
            "user_id": "u1",
            "agent_output": "output",
            "user_response": "response",
        },
    )
    resp = await client.get("/metrics")
    assert "imprint_observe_total" in resp.text
    assert "imprint_observe_latency_seconds" in resp.text
    assert "imprint_policy_cache_misses_total" in resp.text


# -- list agents --------------------------------------------------------------


async def test_list_agents_empty(client: AsyncClient) -> None:
    resp = await client.get("/v1/agents")
    assert resp.status_code == 200
    assert resp.json() == []


async def test_list_agents_after_initialization(client: AsyncClient) -> None:
    await client.get(f"/v1/agents/{AGENT}/memories/user1")
    resp = await client.get("/v1/agents")
    assert resp.status_code == 200
    agents = resp.json()
    assert len(agents) == 1
    assert agents[0]["agent_id"] == AGENT


# -- create agent -------------------------------------------------------------


async def test_post_agents_creates_config(client: AsyncClient) -> None:
    resp = await client.post(
        "/v1/agents",
        json={"agent_id": "new-agent", "processing_mode": "eager"},
    )
    assert resp.status_code == 200
    assert resp.json()["agent_id"] == "new-agent"


async def test_post_agents_pre_warm(client: AsyncClient) -> None:
    await client.post(
        "/v1/agents",
        json={"agent_id": "prewarm-agent", "pre_warm": True},
    )
    resp = await client.get("/v1/agents")
    agent_ids = [a["agent_id"] for a in resp.json()]
    assert "prewarm-agent" in agent_ids


async def test_post_agents_config_is_respected(client: AsyncClient) -> None:
    """Agent initialized after POST should use the configured mode."""
    await client.post(
        "/v1/agents",
        json={"agent_id": "configured-agent", "processing_mode": "eager", "pre_warm": True},
    )
    resp = await client.get("/v1/agents")
    agents = {a["agent_id"]: a for a in resp.json()}
    assert agents["configured-agent"]["processing_mode"] == "eager"


# -- get agent ----------------------------------------------------------------


async def test_get_agent_returns_config(client: AsyncClient) -> None:
    await client.post(
        "/v1/agents",
        json={"agent_id": "get-test-agent", "processing_mode": "frugal"},
    )
    resp = await client.get("/v1/agents/get-test-agent")
    assert resp.status_code == 200
    body = resp.json()
    assert body["agent_id"] == "get-test-agent"
    assert body["processing_mode"] == "frugal"


async def test_get_unknown_agent_returns_404(client: AsyncClient) -> None:
    resp = await client.get("/v1/agents/does-not-exist")
    assert resp.status_code == 404


# -- patch agent config -------------------------------------------------------


async def test_patch_agent_config_updates_mode(client: AsyncClient) -> None:
    # Initialize with default mode.
    await client.get(f"/v1/agents/{AGENT}/memories/user1")

    resp = await client.patch(
        f"/v1/agents/{AGENT}/config",
        json={"processing_mode": "eager"},
    )
    assert resp.status_code == 200
    assert resp.json()["processing_mode"] == "eager"


async def test_patch_agent_config_reloads_registry(client: AsyncClient) -> None:
    """PATCH must apply the new mode to the live Imprint instance."""
    # Initialize.
    await client.get(f"/v1/agents/{AGENT}/memories/user1")

    await client.patch(
        f"/v1/agents/{AGENT}/config",
        json={"processing_mode": "eager"},
    )

    # List agents to confirm live instance reflects change.
    resp = await client.get("/v1/agents")
    agents = {a["agent_id"]: a for a in resp.json()}
    assert agents[AGENT]["processing_mode"] == "eager"


async def test_patch_uninitialized_agent_updates_db(client: AsyncClient) -> None:
    """PATCH on an uncreated agent should write to DB (no error)."""
    resp = await client.patch(
        "/v1/agents/uncreated-agent/config",
        json={"processing_mode": "frugal"},
    )
    assert resp.status_code == 200


# -- delete agent -------------------------------------------------------------


async def test_delete_agent_removes_from_registry(client: AsyncClient) -> None:
    await client.get(f"/v1/agents/{AGENT}/memories/user1")
    assert (await client.get("/health")).json()["agents_loaded"] == 1

    resp = await client.delete(f"/v1/agents/{AGENT}")
    assert resp.status_code == 200
    assert resp.json()["ok"] is True
    assert (await client.get("/health")).json()["agents_loaded"] == 0


async def test_delete_unknown_agent_is_noop(client: AsyncClient) -> None:
    resp = await client.delete("/v1/agents/never-initialized")
    assert resp.status_code == 200


async def test_delete_preserves_memory_data(client: AsyncClient) -> None:
    """Deleting an agent from the registry must not erase user memories."""
    await client.post(
        f"/v1/agents/{AGENT}/memories/user1/directions",
        json={"directions": ["Write in prose."]},
    )
    await client.delete(f"/v1/agents/{AGENT}")

    # Re-initialize and check memories are still there.
    resp = await client.get(f"/v1/agents/{AGENT}/memories/user1")
    assert resp.status_code == 200
    # Memories persist across agent deregistration.
    assert isinstance(resp.json(), list)


# -- scope consolidation ------------------------------------------------------


async def test_scopes_consolidate_returns_ok(client: AsyncClient) -> None:
    resp = await client.post(f"/v1/agents/{AGENT}/scopes/consolidate")
    assert resp.status_code == 200
    assert resp.json()["ok"] is True


# -- dynamic_scopes -----------------------------------------------------------


async def test_create_agent_default_dynamic_scopes_false(client: AsyncClient) -> None:
    resp = await client.post(
        "/v1/agents",
        json={"agent_id": "ds-agent-default"},
    )
    assert resp.status_code == 200
    # GET config to verify dynamic_scopes stored correctly.
    get_resp = await client.get("/v1/agents/ds-agent-default")
    assert get_resp.status_code == 200
    assert get_resp.json()["dynamic_scopes"] is False


async def test_create_agent_dynamic_scopes_true(client: AsyncClient) -> None:
    resp = await client.post(
        "/v1/agents",
        json={"agent_id": "ds-agent-true", "dynamic_scopes": True},
    )
    assert resp.status_code == 200
    get_resp = await client.get("/v1/agents/ds-agent-true")
    assert get_resp.json()["dynamic_scopes"] is True


async def test_get_agent_includes_dynamic_scopes(client: AsyncClient) -> None:
    await client.post("/v1/agents", json={"agent_id": "ds-get-agent"})
    resp = await client.get("/v1/agents/ds-get-agent")
    assert resp.status_code == 200
    assert "dynamic_scopes" in resp.json()


async def test_patch_agent_config_flips_dynamic_scopes(client: AsyncClient) -> None:
    await client.post("/v1/agents", json={"agent_id": "ds-patch-agent"})
    resp = await client.patch(
        "/v1/agents/ds-patch-agent/config",
        json={"dynamic_scopes": True},
    )
    assert resp.status_code == 200
    assert resp.json()["dynamic_scopes"] is True


async def test_patch_dynamic_scopes_applies_to_live_instance(client: AsyncClient) -> None:
    """PATCH dynamic_scopes=True must be reflected in the live Imprint instance."""
    # Initialize agent first.
    await client.get(f"/v1/agents/{AGENT}/memories/user1")

    resp = await client.patch(
        f"/v1/agents/{AGENT}/config",
        json={"dynamic_scopes": True},
    )
    assert resp.json()["dynamic_scopes"] is True

    # Confirm list_agents reflects it too.
    list_resp = await client.get("/v1/agents")
    agents = {a["agent_id"]: a for a in list_resp.json()}
    assert agents[AGENT]["dynamic_scopes"] is True


async def test_patch_preserves_dynamic_scopes_when_not_provided(client: AsyncClient) -> None:
    """PATCH without dynamic_scopes field leaves existing value unchanged."""
    await client.post("/v1/agents", json={"agent_id": "ds-preserve", "dynamic_scopes": True})
    resp = await client.patch(
        "/v1/agents/ds-preserve/config",
        json={"processing_mode": "frugal"},
    )
    assert resp.json()["dynamic_scopes"] is True


# -- OpenAPI schema -----------------------------------------------------------


async def test_openapi_all_endpoints_have_operation_ids(client: AsyncClient) -> None:
    resp = await client.get("/openapi.json")
    assert resp.status_code == 200
    schema = resp.json()
    for path, methods in schema["paths"].items():
        for method, data in methods.items():
            assert "operationId" in data, f"{method.upper()} {path} missing operationId"
            op_id = data["operationId"]
            # Operation IDs must be clean identifiers, not auto-generated garbage.
            assert "__" not in op_id, (
                f"{method.upper()} {path} has auto-generated operationId: {op_id!r}"
            )


async def test_openapi_all_endpoints_have_tags(client: AsyncClient) -> None:
    resp = await client.get("/openapi.json")
    schema = resp.json()
    for path, methods in schema["paths"].items():
        for method, data in methods.items():
            assert data.get("tags"), f"{method.upper()} {path} has no tags"


async def test_openapi_tag_groups_are_defined(client: AsyncClient) -> None:
    resp = await client.get("/openapi.json")
    schema = resp.json()
    tag_names = {t["name"] for t in schema.get("tags", [])}
    assert "memory" in tag_names
    assert "sessions" in tag_names
    assert "agents" in tag_names
    assert "system" in tag_names


async def test_openapi_key_operation_ids_are_stable(client: AsyncClient) -> None:
    """Spot-check that critical operation IDs match expected values.

    SDK generators and client code reference these by name. Renames are
    breaking changes and must be caught early.
    """
    resp = await client.get("/openapi.json")
    schema = resp.json()
    op_ids: dict[str, str] = {}
    for path, methods in schema["paths"].items():
        for method, data in methods.items():
            op_ids[data.get("operationId", "")] = f"{method.upper()} {path}"

    expected = {
        "observe",
        "get_policy",
        "list_memories",
        "search_memories",
        "forget_user",
        "deactivate_memory",
        "pin_memory",
        "consolidate",
        "observe_directions",
        "correct",
        "reinforce",
        "list_events",
        "memory_health",
        "memory_lineage",
        "open_session",
        "session_observe",
        "session_policy",
        "close_session",
        "list_agents",
        "create_agent",
        "get_agent",
        "update_agent_config",
        "delete_agent",
        "consolidate_scopes",
        "health",
        "health_live",
        "health_ready",
        "metrics",
    }
    missing = expected - op_ids.keys()
    assert not missing, f"Operation IDs missing from schema: {sorted(missing)}"


async def test_openapi_response_models_have_examples(client: AsyncClient) -> None:
    """Key response schemas must have an example defined."""
    resp = await client.get("/openapi.json")
    schema = resp.json()
    components = schema.get("components", {}).get("schemas", {})
    models_requiring_examples = [
        "PolicyResponse",
        "ObserveResponse",
        "AgentConfigResponse",
        "OpenSessionResponse",
        "MemoryHealthResponse",
    ]
    for model_name in models_requiring_examples:
        assert model_name in components, f"{model_name} not in schema components"
        assert "example" in components[model_name], f"{model_name} has no example in schema"


# -- Admin dashboard ----------------------------------------------------------


async def test_admin_dashboard_returns_200(client: AsyncClient) -> None:
    resp = await client.get("/admin")
    assert resp.status_code == 200


async def test_admin_dashboard_content_type_is_html(client: AsyncClient) -> None:
    resp = await client.get("/admin")
    assert "text/html" in resp.headers["content-type"]


async def test_admin_dashboard_contains_imprint_branding(client: AsyncClient) -> None:
    resp = await client.get("/admin")
    assert "imprint" in resp.text
    assert "#0d9488" in resp.text  # teal brand color from the logo


async def test_admin_dashboard_not_in_openapi_schema(client: AsyncClient) -> None:
    """Dashboard is excluded from the OpenAPI spec (include_in_schema=False)."""
    schema = (await client.get("/openapi.json")).json()
    assert "/admin" not in schema["paths"]


async def test_keys_list_endpoint_returns_200(client: AsyncClient) -> None:
    resp = await client.get("/v1/keys")
    assert resp.status_code == 200
    assert isinstance(resp.json(), list)


async def test_keys_list_operation_id_in_schema(client: AsyncClient) -> None:
    schema = (await client.get("/openapi.json")).json()
    op_ids = {
        data.get("operationId") for methods in schema["paths"].values() for data in methods.values()
    }
    assert "list_keys" in op_ids


# -- Key create / revoke REST endpoints ---------------------------------------


async def test_create_key_returns_raw_key(client: AsyncClient) -> None:
    resp = await client.post("/v1/keys", json={"label": "test-key"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["raw_key"].startswith("sk-imp-")
    assert len(body["key_hash"]) == 16
    assert body["label"] == "test-key"
    assert body["agent_id"] is None
    assert body["user_id"] is None


async def test_create_key_scoped_to_agent(client: AsyncClient) -> None:
    resp = await client.post(
        "/v1/keys",
        json={"label": "scoped-key", "agent_id": "my-agent"},
    )
    assert resp.status_code == 200
    assert resp.json()["agent_id"] == "my-agent"


async def test_create_key_with_user(client: AsyncClient) -> None:
    resp = await client.post(
        "/v1/keys",
        json={"label": "user-key", "agent_id": "my-agent", "user_id": "alice"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["user_id"] == "alice"
    assert body["agent_id"] == "my-agent"


async def test_created_key_appears_in_list(client: AsyncClient) -> None:
    create_r = await client.post("/v1/keys", json={"label": "list-check-key"})
    assert create_r.status_code == 200
    key_hash = create_r.json()["key_hash"]

    list_r = await client.get("/v1/keys")
    assert list_r.status_code == 200
    hashes = [k["key_hash"] for k in list_r.json()]
    assert key_hash in hashes


async def test_revoke_key_marks_inactive(client: AsyncClient) -> None:
    create_r = await client.post("/v1/keys", json={"label": "to-revoke"})
    assert create_r.status_code == 200
    key_hash = create_r.json()["key_hash"]

    revoke_r = await client.delete(f"/v1/keys/{key_hash}")
    assert revoke_r.status_code == 200
    assert revoke_r.json()["revoked"] is True

    list_r = await client.get("/v1/keys")
    revoked = next((k for k in list_r.json() if k["key_hash"] == key_hash), None)
    assert revoked is not None
    assert revoked["active"] is False


async def test_revoke_nonexistent_key_returns_404(client: AsyncClient) -> None:
    resp = await client.delete("/v1/keys/doesnotexist1234")
    assert resp.status_code == 404


async def test_key_api_operation_ids_in_schema(client: AsyncClient) -> None:
    schema = (await client.get("/openapi.json")).json()
    op_ids = {
        data.get("operationId") for methods in schema["paths"].values() for data in methods.values()
    }
    assert "create_key" in op_ids
    assert "revoke_key" in op_ids
    assert "list_keys" in op_ids
