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


class CreateAgentRequest(BaseModel):
    agent_id: str
    processing_mode: str | None = None
    agent_description: str | None = None
    scopes: list[str] | None = None
    pre_warm: bool = False


class CreateAgentResponse(BaseModel):
    agent_id: str
    created: bool = True


class AgentConfigResponse(BaseModel):
    agent_id: str
    processing_mode: str | None
    agent_description: str | None
    scopes: list[str] | None


class ScopeConsolidateResponse(BaseModel):
    ok: bool = True


# -- List agents --------------------------------------------------------------


@router.get("/agents")
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
        result.append(
            {
                "agent_id": agent_id,
                "processing_mode": imp.processing_mode,
                "agent_description": imp.agent_description,
                "scopes": imp.scopes,
            }
        )
    return result


# -- Create / pre-configure agent ---------------------------------------------


@router.post("/agents", response_model=CreateAgentResponse)
async def create_agent(
    body: CreateAgentRequest,
    registry: RegistryDep,
    config: ConfigDep,
) -> CreateAgentResponse:
    """Pre-configure an agent before its first request.

    Writes agent config to the DB so the first request picks up the
    desired processing_mode and scopes. If pre_warm=true, the agent is
    also initialized immediately (loads the Imprint instance into the
    registry, connects to the store).
    """
    mode = body.processing_mode or config.default_mode
    scopes = body.scopes or []

    await registry.store.put_agent_config(
        agent_id=body.agent_id,
        processing_mode=mode,
        agent_description=body.agent_description,
        scopes=scopes,
    )

    if body.pre_warm:
        await registry.get(body.agent_id)

    return CreateAgentResponse(agent_id=body.agent_id)


# -- Get agent config ---------------------------------------------------------


@router.get("/agents/{agent_id}", response_model=AgentConfigResponse)
async def get_agent(
    agent_id: str,
    registry: RegistryDep,
) -> AgentConfigResponse:
    """Return the stored config for an agent.

    Reads from the DB -- works for agents that have been configured but
    not yet initialized.
    """
    stored = await registry.store.get_agent_config(agent_id)
    if stored is None:
        raise not_found(f"agent {agent_id!r} not found")
    return AgentConfigResponse(
        agent_id=agent_id,
        processing_mode=stored.processing_mode,
        agent_description=stored.agent_description,
        scopes=stored.scopes,
    )


# -- Patch agent config -------------------------------------------------------


@router.patch("/agents/{agent_id}/config", response_model=AgentConfigResponse)
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
    stored = await registry.store.get_agent_config(agent_id)

    # Merge: body fields override stored values; fall back to defaults.
    mode = body.processing_mode or (stored.processing_mode if stored else config.default_mode)
    description = body.agent_description or (stored.agent_description if stored else None)
    scopes = body.scopes if body.scopes is not None else (stored.scopes or [] if stored else [])

    await registry.store.put_agent_config(
        agent_id=agent_id,
        processing_mode=mode,
        agent_description=description,
        scopes=scopes,
    )

    # Apply to live instance if initialized.
    await registry.reload_config(agent_id)

    return AgentConfigResponse(
        agent_id=agent_id,
        processing_mode=mode,
        agent_description=description,
        scopes=scopes,
    )


# -- Delete agent -------------------------------------------------------------


@router.delete("/agents/{agent_id}")
async def delete_agent(
    agent_id: str,
    registry: RegistryDep,
) -> dict[str, bool]:
    """Drain and remove an agent from the registry.

    Does not delete user memory data. The agent_config row in the DB is
    preserved so the agent can be re-initialized on the next request.
    To fully remove an agent, also call DELETE /v1/agents/{agent_id}/memories/{user_id}
    for each user namespace.
    """
    await registry.deregister(agent_id)
    return {"ok": True}


# -- Scope consolidation ------------------------------------------------------


@router.post(
    "/agents/{agent_id}/scopes/consolidate",
    response_model=ScopeConsolidateResponse,
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
