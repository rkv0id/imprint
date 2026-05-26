"""Core operation endpoints for imprint-server.

All endpoints that mutate agent state (observe, observe_directions, forget,
consolidate) hold the per-agent op_lock to prevent concurrent scope
consolidation races. Read-only endpoints (policy, list_memories, events,
health, lineage) do not hold the lock.
"""

from __future__ import annotations

import base64
import logging
import time
import traceback
from typing import Annotated, Any

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel, Field

from imprint_server.config import ServerConfig
from imprint_server.errors import bad_request, not_found
from imprint_server.metrics import (
    consolidation_pruned,
    observe_errors,
    observe_latency,
    observe_total,
    policy_cache_hits,
    policy_cache_misses,
    policy_errors,
    policy_latency,
    policy_memories_dropped,
    policy_memories_retrieved,
    policy_total,
    redis_invalidations,
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
    model_config = {"json_schema_extra": {"example": {"ok": True}}}

    ok: bool = True


class BatchObserveRequest(BaseModel):
    model_config = {
        "json_schema_extra": {
            "example": {
                "items": [
                    {
                        "user_id": "user-42",
                        "agent_output": "Here is a summary.",
                        "user_response": "Too long, please be concise.",
                    },
                    {
                        "user_id": "user-42",
                        "directions": ["always use bullet points"],
                    },
                ]
            }
        }
    }

    items: list[ObserveRequest] = Field(min_length=1, max_length=100)


class BatchObserveItemResult(BaseModel):
    index: int
    ok: bool
    error: str | None = None


class BatchObserveResponse(BaseModel):
    model_config = {
        "json_schema_extra": {
            "example": {
                "processed": 2,
                "failed": 0,
                "results": [
                    {"index": 0, "ok": True, "error": None},
                    {"index": 1, "ok": True, "error": None},
                ],
            }
        }
    }

    processed: int
    failed: int
    results: list[BatchObserveItemResult]


class PolicyRequest(BaseModel):
    user_id: str
    context: str | None = None
    existing_instructions: str | None = None
    max_input_tokens: int = 8000
    max_output_tokens: int = 3000
    scopes: list[str] | None = None
    session_id: str | None = None


class PolicyResponse(BaseModel):
    model_config = {
        "json_schema_extra": {
            "example": {
                "policy_text": "Always respond in plain prose. Avoid bullet points.",
                "memory_count": 3,
                "dropped_count": 0,
                "compiled_at": "2025-04-01T12:00:00+00:00",
                "memory_ids": ["m_abc123", "m_def456", "m_ghi789"],
            }
        }
    }

    policy_text: str
    memory_count: int
    dropped_count: int
    compiled_at: str
    memory_ids: list[str] = []


class DirectionsRequest(BaseModel):
    directions: list[str]
    context: str | None = None
    scope: str | None = None


class DirectionsResponse(BaseModel):
    model_config = {"json_schema_extra": {"example": {"stored": 2}}}

    stored: int


class ConsolidateResponse(BaseModel):
    model_config = {"json_schema_extra": {"example": {"pruned": 3}}}

    pruned: int


class DeleteResponse(BaseModel):
    model_config = {"json_schema_extra": {"example": {"ok": True}}}

    ok: bool = True


class CorrectRequest(BaseModel):
    content: str
    session_id: str | None = None


class ReinforceRequest(BaseModel):
    session_id: str | None = None


class CorrectResponse(BaseModel):
    model_config = {"json_schema_extra": {"example": {"ok": True, "memory_id": "m_abc123"}}}

    ok: bool = True
    memory_id: str | None = None


class ReinforceResponse(BaseModel):
    model_config = {"json_schema_extra": {"example": {"ok": True, "applied": True}}}

    ok: bool = True
    applied: bool


class MemoryResponse(BaseModel):
    """Single memory object as returned by list and search endpoints."""

    model_config = {
        "json_schema_extra": {
            "example": {
                "id": "m_abc123",
                "agent_id": "my-agent",
                "user_id": "user-42",
                "type": "rule",
                "scope": "global",
                "content": "Always respond in plain prose.",
                "source": "detected",
                "stability": 0.85,
                "recall_count": 4,
                "pinned": False,
                "active": True,
                "valid_from": "2025-04-01T12:00:00+00:00",
                "valid_until": None,
                "created_at": "2025-04-01T12:00:00+00:00",
                "updated_at": "2025-04-01T12:00:00+00:00",
            }
        }
    }

    id: str
    agent_id: str
    user_id: str
    type: str
    scope: str
    content: str
    source: str
    stability: float
    recall_count: int
    pinned: bool
    active: bool
    valid_from: str
    valid_until: str | None
    created_at: str
    updated_at: str


class MemoryEventResponse(BaseModel):
    """Single memory event as returned by list_events."""

    model_config = {
        "json_schema_extra": {
            "example": {
                "id": "ev_xyz789",
                "memory_id": "m_abc123",
                "event_type": "recall",
                "detail": {"scope": "global"},
                "occurred_at": "2025-04-01T12:05:00+00:00",
            }
        }
    }

    id: str
    memory_id: str
    event_type: str
    detail: dict[str, object] | None
    occurred_at: str


class MemoryHealthResponse(BaseModel):
    """Aggregate memory statistics for a user namespace."""

    model_config = {
        "json_schema_extra": {
            "example": {
                "total": 12,
                "active": 9,
                "pinned": 1,
                "by_scope": {"global": 6, "coding": 3},
                "by_type": {"rule": 7, "preference": 2},
                "avg_recall_count": 3.2,
                "oldest_active": "2025-03-01T10:00:00+00:00",
                "newest_active": "2025-04-01T12:00:00+00:00",
            }
        }
    }

    total: int
    active: int
    pinned: int
    by_scope: dict[str, int]
    by_type: dict[str, int]
    avg_recall_count: float
    oldest_active: str | None
    newest_active: str | None


class SignalResponse(BaseModel):
    """Signal that triggered memory creation, as returned by lineage."""

    id: str
    agent_id: str
    user_id: str
    signal_type: str
    content: str
    created_at: str


class MemoryLineageResponse(BaseModel):
    """Full creation and mutation history of one memory."""

    model_config = {
        "json_schema_extra": {
            "example": {
                "memory": {
                    "id": "m_abc123",
                    "agent_id": "my-agent",
                    "user_id": "user-42",
                    "type": "rule",
                    "scope": "global",
                    "content": "Always respond in plain prose.",
                    "source": "detected",
                    "stability": 0.85,
                    "recall_count": 4,
                    "pinned": False,
                    "active": True,
                    "valid_from": "2025-04-01T12:00:00+00:00",
                    "valid_until": None,
                    "created_at": "2025-04-01T12:00:00+00:00",
                    "updated_at": "2025-04-01T12:00:00+00:00",
                },
                "created_by_signal": None,
                "superseded_memories": [],
                "superseded_by": None,
                "events": [],
            }
        }
    }

    memory: dict[str, Any]
    created_by_signal: dict[str, Any] | None
    superseded_memories: list[dict[str, Any]]
    superseded_by: dict[str, Any] | None
    events: list[dict[str, Any]]


class PaginatedResponse(BaseModel):
    """Paginated list envelope returned when limit is provided."""

    items: list[dict[str, Any]]
    next_cursor: str | None


# -- observe ------------------------------------------------------------------


@router.post(
    "/agents/{agent_id}/observe",
    response_model=ObserveResponse,
    operation_id="observe",
    tags=["memory"],
    summary="Record an agent-user exchange or explicit directions",
)
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
    mode = imp.processing_mode
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
    except Exception:
        observe_errors.labels(agent_id=agent_id, mode=mode).inc()
        raise
    finally:
        observe_total.labels(agent_id=agent_id, mode=mode).inc()
        observe_latency.labels(agent_id=agent_id, mode=mode).observe(time.perf_counter() - t0)

    # Confusion-based early consolidation: check recent contradiction rate
    # and enqueue an immediate consolidation if above threshold.
    from imprint_server.workers.scheduler import check_confusion_and_enqueue

    await check_confusion_and_enqueue(config, registry, agent_id, body.user_id)

    # Invalidate Redis policy cache for this user -- observe() changes memory state.
    await _redis_invalidate_policy(config, registry, agent_id, body.user_id)

    return ObserveResponse()


@router.post(
    "/agents/{agent_id}/observe/batch",
    response_model=BatchObserveResponse,
    operation_id="batch_observe",
    tags=["memory"],
    summary="Record multiple exchanges or directions in a single request",
)
async def batch_observe(
    agent_id: str,
    body: BatchObserveRequest,
    registry: RegistryDep,
    config: ConfigDep,
) -> BatchObserveResponse:
    """Record up to 100 agent-user exchanges or direction sets in one request.

    Items are processed sequentially under a single op_lock acquisition so
    the batch is serialized against concurrent writers. Each item is validated
    independently; a failure on one item does not abort the rest. Check
    failed > 0 in the response to detect partial failures.

    The Redis policy cache is invalidated once at the end (not per item) and
    the confusion check runs once across all affected user_ids.
    """
    imp = await registry.get(agent_id)
    mode = imp.processing_mode
    results: list[BatchObserveItemResult] = []
    affected_user_ids: set[str] = set()

    async with await registry.get_op_lock(agent_id):
        for i, item in enumerate(body.items):
            # Validate item fields before calling into the library.
            if item.directions is not None:
                if item.agent_output is not None or item.user_response is not None:
                    results.append(
                        BatchObserveItemResult(
                            index=i,
                            ok=False,
                            error=(
                                "Provide either directions or"
                                " (agent_output + user_response), not both."
                            ),
                        )
                    )
                    observe_errors.labels(agent_id=agent_id, mode=mode).inc()
                    continue
            else:
                if item.agent_output is None or item.user_response is None:
                    results.append(
                        BatchObserveItemResult(
                            index=i,
                            ok=False,
                            error=(
                                "Provide both agent_output and user_response,"
                                " or provide directions."
                            ),
                        )
                    )
                    observe_errors.labels(agent_id=agent_id, mode=mode).inc()
                    continue

            t0 = time.perf_counter()
            try:
                if item.directions is not None:
                    await imp.observe_directions(
                        user_id=item.user_id,
                        directions=item.directions,
                        context=item.context,
                        scope=item.scope,
                    )
                else:
                    assert item.agent_output is not None and item.user_response is not None
                    await imp.observe(
                        user_id=item.user_id,
                        agent_output=item.agent_output,
                        user_response=item.user_response,
                        context=item.context,
                        scope=item.scope,
                    )
                results.append(BatchObserveItemResult(index=i, ok=True))
                affected_user_ids.add(item.user_id)
            except Exception as exc:
                results.append(BatchObserveItemResult(index=i, ok=False, error=str(exc)))
                observe_errors.labels(agent_id=agent_id, mode=mode).inc()
            finally:
                observe_total.labels(agent_id=agent_id, mode=mode).inc()
                observe_latency.labels(agent_id=agent_id, mode=mode).observe(
                    time.perf_counter() - t0
                )

    from imprint_server.workers.scheduler import check_confusion_and_enqueue

    for user_id in affected_user_ids:
        await check_confusion_and_enqueue(config, registry, agent_id, user_id)
        await _redis_invalidate_policy(config, registry, agent_id, user_id)

    failed = sum(1 for r in results if not r.ok)
    return BatchObserveResponse(
        processed=len(results),
        failed=failed,
        results=results,
    )


# -- policy -------------------------------------------------------------------


@router.post(
    "/agents/{agent_id}/policy",
    response_model=PolicyResponse,
    operation_id="get_policy",
    tags=["memory"],
    summary="Compile a behavioral policy for a user namespace",
)
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

    Redis policy cache: when IMPRINT_REDIS_URL is configured, compiled
    policies are cached in Redis and served on subsequent identical requests
    without calling the LLM. Invalidated automatically on observe().
    """
    imp = await registry.get(agent_id)
    t0 = time.perf_counter()

    # Redis policy cache: check before compiling.
    params_hash = _policy_params_hash(
        body.context,
        body.scopes,
        body.existing_instructions,
        body.max_input_tokens,
        body.max_output_tokens,
    )
    cached_response = await _redis_get_policy(config, registry, agent_id, body.user_id, params_hash)
    if cached_response is not None:
        policy_total.labels(agent_id=agent_id).inc()
        policy_latency.labels(agent_id=agent_id, cached="true").observe(time.perf_counter() - t0)
        return cached_response

    try:
        pol = await imp.get_policy(
            user_id=body.user_id,
            context=body.context,
            existing_instructions=body.existing_instructions,
            max_input_tokens=body.max_input_tokens,
            max_output_tokens=body.max_output_tokens,
            scopes=body.scopes,
        )
    except Exception as _exc:
        _log = logging.getLogger("imprint_server.policy")
        _log.error(
            "get_policy failed for agent=%s user=%s -- %s: %s\n%s",
            agent_id,
            body.user_id,
            type(_exc).__name__,
            _exc,
            traceback.format_exc(),
        )
        policy_errors.labels(agent_id=agent_id).inc()
        raise
    finally:
        policy_total.labels(agent_id=agent_id).inc()
        policy_latency.labels(agent_id=agent_id, cached="false").observe(time.perf_counter() - t0)

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

    policy_memories_retrieved.labels(agent_id=agent_id).observe(len(pol.memories))
    policy_memories_dropped.labels(agent_id=agent_id).observe(len(pol.dropped_memories))

    response = PolicyResponse(
        policy_text=pol.text,
        memory_count=len(pol.memories),
        dropped_count=len(pol.dropped_memories),
        compiled_at=pol.compiled_at.isoformat(),
        memory_ids=[m.id for m in pol.memories],
    )

    # Store in Redis cache for subsequent identical requests.
    await _redis_put_policy(config, registry, agent_id, body.user_id, params_hash, response)

    return response


# -- memories -----------------------------------------------------------------


@router.get(
    "/agents/{agent_id}/memories/{user_id}",
    response_model=list[MemoryResponse] | PaginatedResponse,
    operation_id="list_memories",
    tags=["memory"],
    summary="List active memories for a user namespace",
)
async def list_memories(
    agent_id: str,
    user_id: str,
    registry: RegistryDep,
    scopes: str | None = None,
    limit: int | None = None,
    cursor: str | None = None,
) -> list[dict[str, Any]] | dict[str, Any]:
    """Return active memories for a user namespace.

    Without limit: returns a plain list (backward compatible).
    With limit: returns a paginated envelope {items, next_cursor}.
    cursor is an opaque string returned by a previous paged call.
    """
    imp = await registry.get(agent_id)
    scope_list = [s.strip() for s in scopes.split(",")] if scopes else None
    memories = await imp.list_memories(user_id, scopes=scope_list)
    dicts = [_memory_to_dict(m) for m in memories]
    if limit is None:
        return dicts
    return _paginate_asc(dicts, key="created_at", limit=limit, cursor=cursor)


@router.get(
    "/agents/{agent_id}/memories/{user_id}/search",
    response_model=list[MemoryResponse],
    operation_id="search_memories",
    tags=["memory"],
    summary="Search memories by semantic similarity",
)
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


@router.delete(
    "/agents/{agent_id}/memories/{user_id}",
    response_model=DeleteResponse,
    operation_id="forget_user",
    tags=["memory"],
    summary="Hard delete all memories for a user namespace",
)
async def forget_user(
    agent_id: str,
    user_id: str,
    registry: RegistryDep,
    config: ConfigDep,
) -> DeleteResponse:
    """Hard delete all memories, signals, and events for a user namespace.

    Irreversible. Does not touch the scope vocabulary.
    """
    imp = await registry.get(agent_id)
    async with await registry.get_op_lock(agent_id):
        await imp.forget(user_id)
    await _redis_invalidate_policy(config, registry, agent_id, user_id)
    return DeleteResponse()


@router.delete(
    "/agents/{agent_id}/memories/{user_id}/{memory_id}",
    response_model=DeleteResponse,
    operation_id="deactivate_memory",
    tags=["memory"],
    summary="Soft-deactivate a single memory",
)
async def deactivate_memory(
    agent_id: str,
    user_id: str,
    memory_id: str,
    registry: RegistryDep,
    config: ConfigDep,
) -> DeleteResponse:
    """Soft-deactivate a single memory.

    The memory is marked inactive and excluded from future policy compilations
    and retrieval. Unlike forget(), it is not hard-deleted and can still appear
    in lineage queries.
    """
    imp = await registry.get(agent_id)
    async with await registry.get_op_lock(agent_id):
        found = await imp.deactivate_memory(user_id, memory_id)
    if not found:
        raise not_found(f"memory {memory_id!r} not found or already inactive")
    await _redis_invalidate_policy(config, registry, agent_id, user_id)
    return DeleteResponse()


@router.post(
    "/agents/{agent_id}/memories/{memory_id}/pin",
    response_model=DeleteResponse,
    operation_id="pin_memory",
    tags=["memory"],
    summary="Pin a memory so it is never dropped by token budget truncation",
)
async def pin_memory(
    agent_id: str,
    memory_id: str,
    registry: RegistryDep,
) -> DeleteResponse:
    """Pin a memory so it is never dropped by token budget truncation.

    Pinned memories always appear in get_policy() output regardless of
    stability or token limits.
    """
    imp = await registry.get(agent_id)
    await imp.pin_memory(memory_id)
    return DeleteResponse()


@router.post(
    "/agents/{agent_id}/memories/{user_id}/consolidate",
    response_model=ConsolidateResponse,
    operation_id="consolidate",
    tags=["memory"],
    summary="Prune decayed memories and run scope consolidation",
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
    if pruned > 0:
        consolidation_pruned.labels(agent_id=agent_id).inc(pruned)
    return ConsolidateResponse(pruned=pruned)


@router.post(
    "/agents/{agent_id}/memories/{user_id}/directions",
    response_model=DirectionsResponse,
    operation_id="observe_directions",
    tags=["memory"],
    summary="Store explicit behavioral directions as memories",
)
async def observe_directions(
    agent_id: str,
    user_id: str,
    body: DirectionsRequest,
    registry: RegistryDep,
    config: ConfigDep,
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
    await _redis_invalidate_policy(config, registry, agent_id, user_id)
    return DirectionsResponse(stored=len(stored))


class MemoryDiffSummary(BaseModel):
    added: int
    deactivated: int
    superseded: int


class SupersededPairResponse(BaseModel):
    old: MemoryResponse
    new: MemoryResponse


class MemoryDiffResponse(BaseModel):
    """Temporal diff of a user's memory state between two points in time."""

    model_config = {
        "json_schema_extra": {
            "example": {
                "since": "2025-04-01T00:00:00+00:00",
                "until": "2025-04-15T12:00:00+00:00",
                "added": [],
                "deactivated": [],
                "superseded": [],
                "summary": {"added": 0, "deactivated": 0, "superseded": 0},
            }
        }
    }

    since: str
    until: str
    added: list[MemoryResponse]
    deactivated: list[MemoryResponse]
    superseded: list[SupersededPairResponse]
    summary: MemoryDiffSummary


# -- correct / reinforce ------------------------------------------------------


@router.post(
    "/agents/{agent_id}/correct/{user_id}",
    response_model=CorrectResponse,
    operation_id="correct",
    tags=["memory"],
    summary="Store a user correction and apply a negative learning signal",
)
async def correct(
    agent_id: str,
    user_id: str,
    body: CorrectRequest,
    registry: RegistryDep,
    config: ConfigDep,
) -> CorrectResponse:
    """Store a user correction as a memory and apply a negative learning signal.

    Always stores the correction text as a detected direction memory. When
    session_id is provided, finalizes the session with outcome=-1.0 and uses
    the correction text as the attribution hint for the decay and bandit models.
    """
    from imprint.types import MemorySource

    imp = await registry.get(agent_id)

    async with await registry.get_op_lock(agent_id):
        stored = await imp.observe_directions(
            user_id=user_id,
            directions=[body.content],
            source=MemorySource.DETECTED,
        )

    memory_id = stored[0].id if stored else None

    if body.session_id is not None:
        await _finalize_session_from_store(
            config=config,
            registry=registry,
            agent_id=agent_id,
            user_id=user_id,
            session_id=body.session_id,
            outcome=-1.0,
            correction=body.content,
        )

    await _redis_invalidate_policy(config, registry, agent_id, user_id)
    return CorrectResponse(memory_id=memory_id)


@router.post(
    "/agents/{agent_id}/reinforce/{user_id}",
    response_model=ReinforceResponse,
    operation_id="reinforce",
    tags=["memory"],
    summary="Apply a positive learning signal for a session",
)
async def reinforce(
    agent_id: str,
    user_id: str,
    body: ReinforceRequest,
    registry: RegistryDep,
    config: ConfigDep,
) -> ReinforceResponse:
    """Apply a positive learning signal for a session.

    Finalizes the session with outcome=0.8, updating the bandit alpha tuner
    and decay model. No-op when session_id is not provided.
    """
    if body.session_id is None:
        return ReinforceResponse(applied=False)

    await _finalize_session_from_store(
        config=config,
        registry=registry,
        agent_id=agent_id,
        user_id=user_id,
        session_id=body.session_id,
        outcome=0.8,
        correction=None,
    )
    return ReinforceResponse(applied=True)


async def _finalize_session_from_store(
    *,
    config: ConfigDep,
    registry: RegistryDep,
    agent_id: str,
    user_id: str,
    session_id: str,
    outcome: float,
    correction: str | None,
) -> None:
    """Reconstruct a MemoryLoop from session DB state and finalize it."""
    from imprint import MemoryLoop

    from imprint_server._pool import get_pg_pool
    from imprint_server.stores.sessions import (
        close_session,
        get_session,
        pg_close_session,
        pg_get_session,
    )

    pg_pool = get_pg_pool(registry) if config.is_postgres else None
    session = (
        await pg_get_session(pg_pool, session_id)
        if pg_pool is not None
        else await get_session(config, session_id)
    )

    if session is None or session.agent_id != agent_id:
        raise not_found(f"session {session_id!r} not found")
    if session.closed_at is not None:
        raise bad_request(f"session {session_id!r} is already closed")

    imp = await registry.get(agent_id)

    retrieved_memories = []
    if session.retrieved_ids:
        all_mems = await imp._store.list_memories(  # type: ignore[attr-defined]
            agent_id, user_id, active_only=False
        )
        id_set = set(session.retrieved_ids)
        retrieved_memories = [m for m in all_mems if m.id in id_set]

    loop = MemoryLoop(user_id=user_id, session_id=session_id, imprint=imp)
    loop.retrieved_ids = set(session.retrieved_ids)
    loop.retrieved_memories = retrieved_memories
    loop.alpha_used = session.alpha_used
    loop.context = session.context
    loop.closed = True
    loop.set_outcome(outcome, correction=correction)

    await imp.finalize_loop(loop)

    if pg_pool is not None:
        await pg_close_session(pg_pool, session_id, outcome=outcome, correction=correction)
    else:
        await close_session(config, session_id, outcome=outcome, correction=correction)


# -- events & lineage ---------------------------------------------------------


@router.get(
    "/agents/{agent_id}/events/{user_id}",
    response_model=list[MemoryEventResponse] | PaginatedResponse,
    operation_id="list_events",
    tags=["memory"],
    summary="List memory events for a user namespace",
)
async def list_events(
    agent_id: str,
    user_id: str,
    registry: RegistryDep,
    memory_id: str | None = None,
    limit: int | None = None,
    cursor: str | None = None,
) -> list[dict[str, Any]] | dict[str, Any]:
    """Return logged memory events for a user namespace, newest first.

    Without limit: returns a plain list (backward compatible), capped at 50.
    With limit: returns a paginated envelope {items, next_cursor}.
    Pass cursor from a previous response to get the next page.
    """
    imp = await registry.get(agent_id)
    effective_limit = limit if limit is not None else 50
    events = await imp.list_events(user_id, memory_id=memory_id, limit=effective_limit + 1)
    dicts = [_event_to_dict(e) for e in events]
    if limit is None and cursor is None:
        return dicts[:effective_limit]
    return _paginate_desc(dicts, key="occurred_at", limit=effective_limit, cursor=cursor)


@router.get(
    "/agents/{agent_id}/health/{user_id}",
    response_model=MemoryHealthResponse,
    operation_id="memory_health",
    tags=["memory"],
    summary="Aggregate memory health statistics for a user namespace",
)
async def memory_health(
    agent_id: str,
    user_id: str,
    registry: RegistryDep,
) -> MemoryHealthResponse:
    """Return aggregate memory health statistics for a user namespace."""
    imp = await registry.get(agent_id)
    health = await imp.memory_health(user_id)
    return MemoryHealthResponse(
        total=health.total,
        active=health.active,
        pinned=health.pinned,
        by_scope=health.by_scope,
        by_type=health.by_type,
        avg_recall_count=health.avg_recall_count,
        oldest_active=(health.oldest_active.isoformat() if health.oldest_active else None),
        newest_active=(health.newest_active.isoformat() if health.newest_active else None),
    )


@router.get(
    "/agents/{agent_id}/memories/{user_id}/diff",
    response_model=MemoryDiffResponse,
    operation_id="memory_diff",
    tags=["memory"],
    summary="Temporal diff of a user's memory state between two timestamps",
)
async def memory_diff(
    agent_id: str,
    user_id: str,
    registry: RegistryDep,
    since: str,
    until: str | None = None,
) -> MemoryDiffResponse:
    """Return what changed in a user's memory between since and until.

    since is required (ISO 8601 timestamp with timezone).
    until defaults to now when not provided.

    Three change categories are returned:

    added:       memories created in the window that are currently active
    deactivated: memories deactivated in the window with no replacement
                 (pruned by consolidation or forgotten explicitly)
    superseded:  (old, new) pairs where a memory was replaced by a correction
                 or update in the window

    Useful for: debugging behavior changes, auditing, syncing to external
    systems, and building change-feed integrations.
    """
    from datetime import UTC, datetime

    from pydantic import AwareDatetime

    def _parse_ts(value: str, field: str) -> AwareDatetime:
        try:
            dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=UTC)
            return dt  # type: ignore[return-value]
        except ValueError:
            raise bad_request(f"{field} is not a valid ISO 8601 timestamp: {value!r}") from None

    since_dt = _parse_ts(since, "since")
    until_dt = _parse_ts(until, "until") if until is not None else datetime.now(UTC)  # type: ignore[assignment]

    if since_dt >= until_dt:
        raise bad_request("since must be before until")

    imp = await registry.get(agent_id)
    diff = await imp.diff_memories(user_id, since_dt, until_dt)

    return MemoryDiffResponse(
        since=diff.since.isoformat(),
        until=diff.until.isoformat(),
        added=[MemoryResponse(**_memory_to_dict(m)) for m in diff.added],
        deactivated=[MemoryResponse(**_memory_to_dict(m)) for m in diff.deactivated],
        superseded=[
            SupersededPairResponse(
                old=MemoryResponse(**_memory_to_dict(p.old)),
                new=MemoryResponse(**_memory_to_dict(p.new)),
            )
            for p in diff.superseded
        ],
        summary=MemoryDiffSummary(**diff.summary),
    )


@router.get(
    "/memories/{memory_id}/lineage",
    response_model=MemoryLineageResponse,
    operation_id="memory_lineage",
    tags=["memory"],
    summary="Full creation and mutation history of one memory",
)
async def memory_lineage(
    memory_id: str,
    registry: RegistryDep,
) -> MemoryLineageResponse:
    """Return the full creation and mutation history of one memory.

    Note: requires at least one agent to be initialized. For multi-agent
    deployments, any loaded agent's store works since memories are keyed by ID.
    This endpoint requires the registry to have at least one initialized agent.
    """
    agent_ids = registry.agent_ids()
    if not agent_ids:
        raise not_found(f"memory {memory_id!r} not found (no agents loaded)")
    imp = await registry.get(agent_ids[0])
    try:
        lineage = await imp.memory_lineage(memory_id)
    except KeyError:
        raise not_found(f"memory {memory_id!r} not found") from None
    return MemoryLineageResponse(
        memory=_memory_to_dict(lineage.memory),
        created_by_signal=(
            _signal_to_dict(lineage.created_by_signal) if lineage.created_by_signal else None
        ),
        superseded_memories=[_memory_to_dict(m) for m in lineage.superseded_memories],
        superseded_by=(_memory_to_dict(lineage.superseded_by) if lineage.superseded_by else None),
        events=[_event_to_dict(e) for e in lineage.events],
    )


# -- Pagination helpers -------------------------------------------------------


def _encode_cursor(value: str) -> str:
    return base64.urlsafe_b64encode(value.encode()).decode()


def _decode_cursor(cursor: str) -> str | None:
    try:
        return base64.urlsafe_b64decode(cursor.encode()).decode()
    except Exception:
        return None


def _paginate_asc(
    items: list[dict[str, Any]],
    *,
    key: str,
    limit: int,
    cursor: str | None,
) -> dict[str, Any]:
    """Paginate an ASC-sorted list (e.g. memories by created_at).

    cursor represents the created_at of the last item seen. Next page
    contains items with that key value strictly greater than cursor.
    """
    if cursor is not None:
        threshold = _decode_cursor(cursor)
        if threshold is not None:
            items = [i for i in items if i.get(key, "") > threshold]
    page = items[:limit]
    has_more = len(page) == limit and len(items) > limit
    next_cursor = _encode_cursor(page[-1][key]) if has_more else None
    return {"items": page, "next_cursor": next_cursor}


def _paginate_desc(
    items: list[dict[str, Any]],
    *,
    key: str,
    limit: int,
    cursor: str | None,
) -> dict[str, Any]:
    """Paginate a DESC-sorted list (e.g. events by occurred_at).

    cursor represents the occurred_at of the last item seen. Next page
    contains items with that key value strictly less than cursor.
    """
    if cursor is not None:
        threshold = _decode_cursor(cursor)
        if threshold is not None:
            items = [i for i in items if i.get(key, "") < threshold]
    has_more = len(items) > limit
    page = items[:limit]
    next_cursor = _encode_cursor(page[-1][key]) if has_more and page else None
    return {"items": page, "next_cursor": next_cursor}


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
        "detail": e.detail,
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


# -- Redis policy cache -------------------------------------------------------


def _redis_policy_key(agent_id: str, user_id: str, params_hash: str) -> str:
    """Stable Redis key for a compiled policy."""
    return f"imprint:policy:{agent_id}:{user_id}:{params_hash}"


def _redis_policy_pattern(agent_id: str, user_id: str) -> str:
    """Glob pattern matching all cached policies for an agent-user pair."""
    return f"imprint:policy:{agent_id}:{user_id}:*"


def _policy_params_hash(
    context: str | None,
    scopes: list[str] | None,
    existing_instructions: str | None,
    max_input_tokens: int,
    max_output_tokens: int,
) -> str:
    """Hash of policy compilation parameters for cache keying."""
    import hashlib

    raw = "|".join(
        [
            context or "",
            ",".join(sorted(scopes)) if scopes else "",
            existing_instructions or "",
            str(max_input_tokens),
            str(max_output_tokens),
        ]
    )
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


async def _redis_get_policy(
    config: ConfigDep,
    registry: RegistryDep,
    agent_id: str,
    user_id: str,
    params_hash: str,
) -> PolicyResponse | None:
    """Return a cached PolicyResponse from Redis, or None on miss/disabled."""
    if not config.redis_enabled or registry.redis is None:
        return None
    import json

    key = _redis_policy_key(agent_id, user_id, params_hash)
    cached = await registry.redis.get(key)
    if cached is None:
        policy_cache_misses.labels(agent_id=agent_id).inc()
        return None
    try:
        data = json.loads(cached)
        policy_cache_hits.labels(agent_id=agent_id).inc()
        return PolicyResponse(**data)
    except Exception:
        return None


async def _redis_put_policy(
    config: ConfigDep,
    registry: RegistryDep,
    agent_id: str,
    user_id: str,
    params_hash: str,
    response: PolicyResponse,
) -> None:
    """Store a PolicyResponse in Redis with the configured TTL."""
    if not config.redis_enabled or registry.redis is None:
        return
    import json

    key = _redis_policy_key(agent_id, user_id, params_hash)
    await registry.redis.setex(key, config.cache_ttl, json.dumps(response.model_dump()))


async def _redis_invalidate_policy(
    config: ConfigDep,
    registry: RegistryDep,
    agent_id: str,
    user_id: str,
) -> None:
    """Delete all Redis policy cache entries for an agent-user pair."""
    if not config.redis_enabled or registry.redis is None:
        return
    pattern = _redis_policy_pattern(agent_id, user_id)
    await registry.redis.delete_pattern(pattern)
    redis_invalidations.labels(agent_id=agent_id).inc()
