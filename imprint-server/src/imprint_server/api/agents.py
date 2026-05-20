"""Core operation endpoints for imprint-server.

All endpoints that mutate agent state (observe, observe_directions, forget,
consolidate) hold the per-agent op_lock to prevent concurrent scope
consolidation races. Read-only endpoints (policy, list_memories, events,
health, lineage) do not hold the lock.
"""

from __future__ import annotations

import time
from typing import Annotated, Any

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel

from imprint_server.config import ServerConfig
from imprint_server.errors import bad_request, not_found
from imprint_server.metrics import (
    observe_duration,
    observe_total,
    policy_duration,
    policy_total,
)
from imprint_server.registry import AgentRegistry
from imprint_server.stores.policy_events import log_policy_event

router = APIRouter()


# -- Dependency helpers -------------------------------------------------------


def _registry(request: Request) -> AgentRegistry:
    reg: AgentRegistry = request.app.state.registry
    return reg


def _config(request: Request) -> ServerConfig:
    cfg: ServerConfig = request.app.state.config
    return cfg


RegistryDep = Annotated[AgentRegistry, Depends(_registry)]
ConfigDep = Annotated[ServerConfig, Depends(_config)]


# -- Request / response models ------------------------------------------------


class ObserveRequest(BaseModel):
    user_id: str
    agent_output: str | None = None
    user_response: str | None = None
    directions: list[str] | None = None
    context: str | None = None
    scope: str | None = None


class ObserveResponse(BaseModel):
    ok: bool = True


class PolicyRequest(BaseModel):
    user_id: str
    context: str | None = None
    existing_instructions: str | None = None
    max_input_tokens: int = 8000
    max_output_tokens: int = 3000
    scopes: list[str] | None = None
    session_id: str | None = None


class PolicyResponse(BaseModel):
    policy_text: str
    memory_count: int
    dropped_count: int
    compiled_at: str


class DirectionsRequest(BaseModel):
    directions: list[str]
    context: str | None = None
    scope: str | None = None


class DirectionsResponse(BaseModel):
    stored: int


class ConsolidateResponse(BaseModel):
    pruned: int


class DeleteResponse(BaseModel):
    ok: bool = True


# -- observe ------------------------------------------------------------------


@router.post("/agents/{agent_id}/observe", response_model=ObserveResponse)
async def observe(
    agent_id: str,
    body: ObserveRequest,
    registry: RegistryDep,
    config: ConfigDep,
) -> ObserveResponse:
    """Record one agent-user exchange or a set of explicit directions.

    Turn-by-turn: provide agent_output and user_response.
    Directions: provide directions (list of instruction strings).
    Exactly one mode must be used per call.
    """
    if body.directions is not None:
        if body.agent_output is not None or body.user_response is not None:
            raise bad_request(
                "Provide either directions or (agent_output + user_response), not both."
            )
    else:
        if body.agent_output is None or body.user_response is None:
            raise bad_request(
                "Provide both agent_output and user_response for turn-by-turn observation,"
                " or provide directions for direction observation."
            )

    imp = await registry.get(agent_id)
    t0 = time.perf_counter()

    try:
        async with await registry.get_op_lock(agent_id):
            if body.directions is not None:
                await imp.observe_directions(
                    user_id=body.user_id,
                    directions=body.directions,
                    context=body.context,
                    scope=body.scope,
                )
            else:
                assert body.agent_output is not None and body.user_response is not None
                await imp.observe(
                    user_id=body.user_id,
                    agent_output=body.agent_output,
                    user_response=body.user_response,
                    context=body.context,
                    scope=body.scope,
                )
    finally:
        observe_total.labels(agent_id=agent_id).inc()
        observe_duration.labels(agent_id=agent_id).observe(time.perf_counter() - t0)

    # Confusion-based early consolidation: check recent contradiction rate
    # and enqueue an immediate consolidation if above threshold.
    from imprint_server.workers.scheduler import check_confusion_and_enqueue

    await check_confusion_and_enqueue(config, registry, agent_id, body.user_id)

    return ObserveResponse()


# -- policy -------------------------------------------------------------------


@router.post("/agents/{agent_id}/policy", response_model=PolicyResponse)
async def policy(
    agent_id: str,
    body: PolicyRequest,
    registry: RegistryDep,
    config: ConfigDep,
) -> PolicyResponse:
    """Compile and return a behavioral policy for a user namespace.

    policy_events are logged on every call regardless of cache status.
    alpha_used is 0.0 for sessionless calls; session-based calls (step 4)
    will carry the real loop.alpha_used from the MemoryLoop.
    """
    imp = await registry.get(agent_id)
    t0 = time.perf_counter()

    try:
        pol = await imp.get_policy(
            user_id=body.user_id,
            context=body.context,
            existing_instructions=body.existing_instructions,
            max_input_tokens=body.max_input_tokens,
            max_output_tokens=body.max_output_tokens,
            scopes=body.scopes,
        )
    finally:
        policy_total.labels(agent_id=agent_id).inc()
        policy_duration.labels(agent_id=agent_id).observe(time.perf_counter() - t0)

    # Log counterfactual data for Phase 2 learning.
    await log_policy_event(
        registry=registry,
        config=config,
        agent_id=agent_id,
        user_id=body.user_id,
        session_id=body.session_id,
        retrieved_memories=pol.memories,
        filtered_memories=pol.dropped_memories,
        alpha_used=0.0,
        context=body.context,
    )

    return PolicyResponse(
        policy_text=pol.text,
        memory_count=len(pol.memories),
        dropped_count=len(pol.dropped_memories),
        compiled_at=pol.compiled_at.isoformat(),
    )


# -- memories -----------------------------------------------------------------


@router.get("/agents/{agent_id}/memories/{user_id}")
async def list_memories(
    agent_id: str,
    user_id: str,
    registry: RegistryDep,
    scopes: str | None = None,
) -> list[dict[str, Any]]:
    """Return active memories for a user namespace, optionally filtered by scope."""
    imp = await registry.get(agent_id)
    scope_list = [s.strip() for s in scopes.split(",")] if scopes else None
    memories = await imp.list_memories(user_id, scopes=scope_list)
    return [_memory_to_dict(m) for m in memories]


@router.get("/agents/{agent_id}/memories/{user_id}/search")
async def search_memories(
    agent_id: str,
    user_id: str,
    registry: RegistryDep,
    q: str,
    limit: int = 20,
) -> list[dict[str, Any]]:
    """Search memories by semantic similarity for a user namespace.

    Uses the configured embedder + vector store when available. Falls back to
    list order when no embedder is configured -- the library handles this
    gracefully without requiring an embedder to be present.
    """
    imp = await registry.get(agent_id)
    memories = await imp.search_memories(user_id, q)
    return [_memory_to_dict(m) for m in memories[:limit]]


@router.delete("/agents/{agent_id}/memories/{user_id}", response_model=DeleteResponse)
async def forget_user(
    agent_id: str,
    user_id: str,
    registry: RegistryDep,
) -> DeleteResponse:
    """Hard delete all memories, signals, and events for a user namespace.

    Irreversible. Does not touch the scope vocabulary.
    """
    imp = await registry.get(agent_id)
    async with await registry.get_op_lock(agent_id):
        await imp.forget(user_id)
    return DeleteResponse()


@router.post(
    "/agents/{agent_id}/memories/{user_id}/consolidate",
    response_model=ConsolidateResponse,
)
async def consolidate(
    agent_id: str,
    user_id: str,
    registry: RegistryDep,
    prune_threshold: float = 0.5,
) -> ConsolidateResponse:
    """Prune decayed memories and run scope consolidation for a user namespace."""
    imp = await registry.get(agent_id)
    async with await registry.get_op_lock(agent_id):
        pruned = await imp.consolidate(user_id, prune_threshold=prune_threshold)
    return ConsolidateResponse(pruned=pruned)


@router.post(
    "/agents/{agent_id}/memories/{user_id}/directions",
    response_model=DirectionsResponse,
)
async def observe_directions(
    agent_id: str,
    user_id: str,
    body: DirectionsRequest,
    registry: RegistryDep,
) -> DirectionsResponse:
    """Persist explicit behavioral directions as memories, bypassing signal detection."""
    if not body.directions:
        raise bad_request("directions list must not be empty")
    imp = await registry.get(agent_id)
    async with await registry.get_op_lock(agent_id):
        stored = await imp.observe_directions(
            user_id=user_id,
            directions=body.directions,
            context=body.context,
            scope=body.scope,
        )
    return DirectionsResponse(stored=len(stored))


# -- events & lineage ---------------------------------------------------------


@router.get("/agents/{agent_id}/events/{user_id}")
async def list_events(
    agent_id: str,
    user_id: str,
    registry: RegistryDep,
    memory_id: str | None = None,
    limit: int = 50,
) -> list[dict[str, Any]]:
    """Return logged memory events for a user namespace, newest first."""
    imp = await registry.get(agent_id)
    events = await imp.list_events(user_id, memory_id=memory_id, limit=limit)
    return [_event_to_dict(e) for e in events]


@router.get("/agents/{agent_id}/health/{user_id}")
async def memory_health(
    agent_id: str,
    user_id: str,
    registry: RegistryDep,
) -> dict[str, Any]:
    """Return aggregate memory health statistics for a user namespace."""
    imp = await registry.get(agent_id)
    health = await imp.memory_health(user_id)
    return {
        "total": health.total,
        "active": health.active,
        "pinned": health.pinned,
        "by_scope": health.by_scope,
        "by_type": health.by_type,
        "avg_recall_count": health.avg_recall_count,
        "oldest_active": (health.oldest_active.isoformat() if health.oldest_active else None),
        "newest_active": (health.newest_active.isoformat() if health.newest_active else None),
    }


@router.get("/memories/{memory_id}/lineage")
async def memory_lineage(
    memory_id: str,
    registry: RegistryDep,
) -> dict[str, Any]:
    """Return the full creation and mutation history of one memory.

    Note: requires at least one agent to be initialized. For multi-agent
    deployments, any loaded agent's store works since memories are keyed by ID.
    This endpoint requires the registry to have at least one initialized agent.
    """
    agent_ids = registry.agent_ids()
    if not agent_ids:
        raise not_found(f"memory {memory_id!r} not found (no agents loaded)")
    # Use the first loaded agent's Imprint instance -- all share the same store.
    imp = await registry.get(agent_ids[0])
    try:
        lineage = await imp.memory_lineage(memory_id)
    except KeyError:
        raise not_found(f"memory {memory_id!r} not found") from None
    return {
        "memory": _memory_to_dict(lineage.memory),
        "created_by_signal": (
            _signal_to_dict(lineage.created_by_signal) if lineage.created_by_signal else None
        ),
        "superseded_memories": [_memory_to_dict(m) for m in lineage.superseded_memories],
        "superseded_by": (
            _memory_to_dict(lineage.superseded_by) if lineage.superseded_by else None
        ),
        "events": [_event_to_dict(e) for e in lineage.events],
    }


# -- Serialization helpers ----------------------------------------------------


def _memory_to_dict(m: Any) -> dict[str, Any]:
    return {
        "id": m.id,
        "agent_id": m.agent_id,
        "user_id": m.user_id,
        "type": m.type.value,
        "scope": m.scope,
        "content": m.content,
        "source": m.source.value,
        "stability": m.stability,
        "recall_count": m.recall_count,
        "pinned": m.pinned,
        "active": m.active,
        "valid_from": m.valid_from.isoformat(),
        "valid_until": m.valid_until.isoformat() if m.valid_until else None,
        "created_at": m.created_at.isoformat(),
        "updated_at": m.updated_at.isoformat(),
    }


def _event_to_dict(e: Any) -> dict[str, Any]:
    return {
        "id": e.id,
        "memory_id": e.memory_id,
        "event_type": e.event_type,
        "metadata": e.metadata,
        "occurred_at": e.occurred_at.isoformat(),
    }


def _signal_to_dict(s: Any) -> dict[str, Any]:
    return {
        "id": s.id,
        "agent_id": s.agent_id,
        "user_id": s.user_id,
        "signal_type": s.signal_type.value,
        "content": s.content,
        "created_at": s.created_at.isoformat(),
    }
