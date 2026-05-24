"""Agent administration endpoints for imprint-server.

These endpoints manage agent configuration and the registry lifecycle.
They do not touch user memory data -- data operations go through the
core agents endpoints.

GET    /v1/agents                                   list initialized agents
POST   /v1/agents                                   pre-configure an agent
GET    /v1/agents/{agent_id}                        get agent config
PATCH  /v1/agents/{agent_id}/config                update config + reload
DELETE /v1/agents/{agent_id}                        drain + deregister
POST   /v1/agents/{agent_id}/scopes/consolidate    run scope consolidation
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel

from imprint_server.api.agents import ConfigDep, RegistryDep
from imprint_server.errors import not_found

router = APIRouter()

_VALID_MODES = frozenset({"frugal", "balanced", "eager"})


# -- Request / response models ------------------------------------------------


class AgentConfigIn(BaseModel):
    processing_mode: str | None = None
    agent_description: str | None = None
    scopes: list[str] | None = None
    dynamic_scopes: bool | None = None


class CreateAgentRequest(BaseModel):
    agent_id: str
    processing_mode: str | None = None
    agent_description: str | None = None
    scopes: list[str] | None = None
    dynamic_scopes: bool = False
    pre_warm: bool = False


class CreateAgentResponse(BaseModel):
    model_config = {"json_schema_extra": {"example": {"agent_id": "my-agent", "created": True}}}

    agent_id: str
    created: bool = True


class AgentConfigResponse(BaseModel):
    model_config = {
        "json_schema_extra": {
            "example": {
                "agent_id": "my-agent",
                "processing_mode": "balanced",
                "agent_description": "Customer support agent",
                "scopes": ["global", "support"],
                "dynamic_scopes": False,
            }
        }
    }

    agent_id: str
    processing_mode: str | None
    agent_description: str | None
    scopes: list[str] | None
    dynamic_scopes: bool


class ScopeConsolidateResponse(BaseModel):
    model_config = {"json_schema_extra": {"example": {"ok": True}}}

    ok: bool = True


# -- List agents --------------------------------------------------------------


@router.get(
    "/agents",
    response_model=list[AgentConfigResponse],
    operation_id="list_agents",
    tags=["agents"],
    summary="List all initialized agents",
)
async def list_agents(registry: RegistryDep) -> list[dict[str, Any]]:
    """Return all agents currently initialized in the registry.

    Agents that have been configured but never received a request are not
    listed here -- they are initialized lazily on first request. Use
    GET /v1/agents/{agent_id} to check a specific agent's config regardless
    of whether it is initialized.
    """
    result: list[dict[str, object]] = []
    for agent_id in registry.agent_ids():
        imp = await registry.get(agent_id)
        from imprint_server.db import get_agent_dynamic_scopes

        dynamic_scopes = await get_agent_dynamic_scopes(
            registry.config,
            registry.store,
            agent_id,  # type: ignore[attr-defined]
        )
        result.append(
            {
                "agent_id": agent_id,
                "processing_mode": imp.processing_mode,
                "agent_description": imp.agent_description,
                "scopes": imp.scopes,
                "dynamic_scopes": dynamic_scopes,
            }
        )
    return result


# -- Create / pre-configure agent ---------------------------------------------


@router.post(
    "/agents",
    response_model=CreateAgentResponse,
    operation_id="create_agent",
    tags=["agents"],
    summary="Pre-configure an agent",
)
async def create_agent(
    body: CreateAgentRequest,
    registry: RegistryDep,
    config: ConfigDep,
) -> CreateAgentResponse:
    """Pre-configure an agent before its first request.

    Writes agent config to the DB so the first request picks up the
    desired processing_mode, scopes, and dynamic_scopes. If pre_warm=true,
    the agent is also initialized immediately.
    """
    from imprint_server.db import set_agent_dynamic_scopes

    mode = body.processing_mode or config.default_mode
    scopes = body.scopes or []

    await registry.store.put_agent_config(
        agent_id=body.agent_id,
        processing_mode=mode,
        agent_description=body.agent_description,
        scopes=scopes,
    )
    await set_agent_dynamic_scopes(config, registry.store, body.agent_id, body.dynamic_scopes)

    if body.pre_warm:
        await registry.get(body.agent_id)

    return CreateAgentResponse(agent_id=body.agent_id)


# -- Get agent config ---------------------------------------------------------


@router.get(
    "/agents/{agent_id}",
    response_model=AgentConfigResponse,
    operation_id="get_agent",
    tags=["agents"],
    summary="Get agent configuration",
)
async def get_agent(
    agent_id: str,
    registry: RegistryDep,
    config: ConfigDep,
) -> AgentConfigResponse:
    """Return the stored config for an agent.

    Reads from the DB -- works for agents that have been configured but
    not yet initialized.
    """
    from imprint_server.db import get_agent_dynamic_scopes

    stored = await registry.store.get_agent_config(agent_id)
    if stored is None:
        raise not_found(f"agent {agent_id!r} not found")
    dynamic_scopes = await get_agent_dynamic_scopes(config, registry.store, agent_id)
    return AgentConfigResponse(
        agent_id=agent_id,
        processing_mode=stored.processing_mode,
        agent_description=stored.agent_description,
        scopes=stored.scopes,
        dynamic_scopes=dynamic_scopes,
    )


# -- Patch agent config -------------------------------------------------------


@router.patch(
    "/agents/{agent_id}/config",
    response_model=AgentConfigResponse,
    operation_id="update_agent_config",
    tags=["agents"],
    summary="Update agent configuration",
)
async def patch_agent_config(
    agent_id: str,
    body: AgentConfigIn,
    registry: RegistryDep,
    config: ConfigDep,
) -> AgentConfigResponse:
    """Update agent configuration and apply to the live instance immediately.

    Updates the DB row then calls registry.reload_config() so the running
    Imprint instance picks up the new values without restarting. Fields
    not provided in the request body are left unchanged.
    """
    from imprint_server.db import get_agent_dynamic_scopes, set_agent_dynamic_scopes

    stored = await registry.store.get_agent_config(agent_id)
    current_dynamic = await get_agent_dynamic_scopes(config, registry.store, agent_id)

    # Merge: body fields override stored values; fall back to defaults.
    mode = body.processing_mode or (stored.processing_mode if stored else config.default_mode)
    description = body.agent_description or (stored.agent_description if stored else None)
    scopes = body.scopes if body.scopes is not None else (stored.scopes or [] if stored else [])
    dynamic_scopes = body.dynamic_scopes if body.dynamic_scopes is not None else current_dynamic

    await registry.store.put_agent_config(
        agent_id=agent_id,
        processing_mode=mode,
        agent_description=description,
        scopes=scopes,
    )
    await set_agent_dynamic_scopes(config, registry.store, agent_id, dynamic_scopes)

    # Apply to live instance if initialized.
    await registry.reload_config(agent_id)

    return AgentConfigResponse(
        agent_id=agent_id,
        processing_mode=mode,
        agent_description=description,
        scopes=scopes,
        dynamic_scopes=dynamic_scopes,
    )


# -- Delete agent -------------------------------------------------------------


class DeleteAgentResponse(BaseModel):
    model_config = {"json_schema_extra": {"example": {"ok": True}}}

    ok: bool = True


@router.delete(
    "/agents/{agent_id}",
    response_model=DeleteAgentResponse,
    operation_id="delete_agent",
    tags=["agents"],
    summary="Drain and remove an agent from the registry",
)
async def delete_agent(
    agent_id: str,
    registry: RegistryDep,
) -> DeleteAgentResponse:
    """Drain and remove an agent from the registry.

    Does not delete user memory data. The agent_config row in the DB is
    preserved so the agent can be re-initialized on the next request.
    To fully remove an agent, also call DELETE /v1/agents/{agent_id}/memories/{user_id}
    for each user namespace.
    """
    await registry.deregister(agent_id)
    return DeleteAgentResponse()


# -- Scope consolidation ------------------------------------------------------


@router.post(
    "/agents/{agent_id}/scopes/consolidate",
    response_model=ScopeConsolidateResponse,
    operation_id="consolidate_scopes",
    tags=["agents"],
    summary="Run scope vocabulary consolidation for an agent",
)
async def consolidate_scopes(
    agent_id: str,
    registry: RegistryDep,
) -> ScopeConsolidateResponse:
    """Run scope vocabulary consolidation for an agent.

    Merges overlapping scope names across all user namespaces. Equivalent
    to calling consolidate_scopes(user_id=None) on the library directly.
    Only meaningful in balanced or eager processing mode.
    """
    imp = await registry.get(agent_id)
    async with await registry.get_op_lock(agent_id):
        await imp.consolidate_scopes(None)  # type: ignore[attr-defined]
    return ScopeConsolidateResponse()


# -- API keys (read-only) -----------------------------------------------------


class ApiKeyResponse(BaseModel):
    model_config = {
        "json_schema_extra": {
            "example": {
                "key_hash": "a1b2c3d4e5f6a1b2",
                "label": "production-key",
                "agent_id": None,
                "user_id": None,
                "active": True,
                "created_at": "2025-04-01T12:00:00+00:00",
                "expires_at": None,
            }
        }
    }

    key_hash: str
    label: str | None
    agent_id: str | None
    user_id: str | None
    active: bool
    created_at: str
    expires_at: str | None


@router.get(
    "/keys",
    response_model=list[ApiKeyResponse],
    operation_id="list_keys",
    tags=["agents"],
    summary="List API keys (hashes only, read-only)",
)
async def list_keys(
    config: ConfigDep,
    registry: RegistryDep,
) -> list[ApiKeyResponse]:
    """Return all API key rows. Hashes only -- raw keys are never stored."""
    from imprint_server.stores.api_keys import list_keys as _list_keys
    from imprint_server.stores.api_keys import pg_list_with_pool

    if config.is_postgres:
        from imprint_server._pool import get_pg_pool

        rows = await pg_list_with_pool(get_pg_pool(registry))
    else:
        rows = await _list_keys(config)

    return [
        ApiKeyResponse(
            key_hash=row.key_hash[:16],
            label=row.label,
            agent_id=row.agent_id,
            user_id=row.user_id,
            active=row.active,
            created_at=row.created_at.isoformat(),
            expires_at=row.expires_at.isoformat() if row.expires_at else None,
        )
        for row in rows
    ]
