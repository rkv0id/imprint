"""MemoryLoop-over-HTTP endpoints for imprint-server.

Sessions allow multi-turn interactions where the server tracks which memories
were retrieved and at what alpha weight, then applies a learning signal when
the session closes. This is the HTTP equivalent of the library's MemoryLoop
context manager.

Flow:
  1. POST /sessions              -> session_id
  2. POST /sessions/{id}/policy  -> policy_text (updates session row)
  3. POST /sessions/{id}/observe -> ok (optional, adds context to the exchange)
  4. POST /sessions/{id}/close   -> ok (calls finalize_loop with real data)

Steps 2 and 3 can be interleaved. Policy can be called multiple times within
a session (e.g., multi-turn conversation). Only the most recent policy call's
retrieved_ids are used for the learning signal on close.

The learning signal path on close:
  1. Read session row (retrieved_ids, alpha_used, context, user_id)
  2. Re-fetch Memory objects from store by their IDs
  3. Construct a MemoryLoop with the fetched data
  4. Call imprint.finalize_loop(loop)

Re-fetching reflects current memory state (stability may have changed since
the policy call) -- correct for decay updates and attribution.
"""

from __future__ import annotations

from fastapi import APIRouter
from imprint import MemoryLoop
from pydantic import BaseModel

from imprint_server._pool import get_pg_pool
from imprint_server.api.agents import ConfigDep, PolicyResponse, RegistryDep
from imprint_server.config import ServerConfig
from imprint_server.errors import bad_request, not_found
from imprint_server.metrics import session_total
from imprint_server.registry import AgentRegistry
from imprint_server.stores.policy_events import log_policy_event
from imprint_server.stores.sessions import (
    SessionRow,
    close_session,
    create_session,
    get_session,
    pg_close_session,
    pg_create_session,
    pg_get_session,
    pg_update_session_policy,
    update_session_policy,
)

router = APIRouter()


# -- Request / response models ------------------------------------------------


class OpenSessionRequest(BaseModel):
    user_id: str
    context: str | None = None


class OpenSessionResponse(BaseModel):
    model_config = {"json_schema_extra": {"example": {"session_id": "sess_abc123"}}}

    session_id: str


class SessionObserveRequest(BaseModel):
    agent_output: str | None = None
    user_response: str | None = None
    directions: list[str] | None = None
    context: str | None = None
    scope: str | None = None


class SessionObserveResponse(BaseModel):
    model_config = {"json_schema_extra": {"example": {"ok": True}}}

    ok: bool = True


class SessionPolicyRequest(BaseModel):
    context: str | None = None
    existing_instructions: str | None = None
    max_input_tokens: int = 8000
    max_output_tokens: int = 3000
    scopes: list[str] | None = None


class CloseSessionRequest(BaseModel):
    outcome: float | None = None
    correction: str | None = None


class CloseSessionResponse(BaseModel):
    model_config = {"json_schema_extra": {"example": {"ok": True}}}

    ok: bool = True


# -- Helpers ------------------------------------------------------------------


async def _load_session(
    config: ServerConfig, registry: AgentRegistry, session_id: str, agent_id: str
) -> SessionRow:
    """Load and validate a session row. Raises 404 if absent or wrong agent."""
    if config.is_postgres:
        from imprint_server._pool import get_pg_pool

        session = await pg_get_session(get_pg_pool(registry), session_id)
    else:
        session = await get_session(config, session_id)

    if session is None:
        raise not_found(f"session {session_id!r} not found")
    if session.agent_id != agent_id:
        raise not_found(f"session {session_id!r} not found")
    if session.closed_at is not None:
        raise bad_request(f"session {session_id!r} is already closed")
    return session


# -- Open session -------------------------------------------------------------


@router.post(
    "/agents/{agent_id}/sessions",
    response_model=OpenSessionResponse,
    operation_id="open_session",
    tags=["sessions"],
    summary="Open a new MemoryLoop session",
)
async def open_session(
    agent_id: str,
    body: OpenSessionRequest,
    registry: RegistryDep,
    config: ConfigDep,
) -> OpenSessionResponse:
    """Open a new MemoryLoop session for a user namespace."""
    # Ensure the agent is initialized.
    await registry.get(agent_id)
    session_total.labels(agent_id=agent_id).inc()

    if config.is_postgres:
        from imprint_server._pool import get_pg_pool

        session_id = await pg_create_session(
            get_pg_pool(registry),
            agent_id=agent_id,
            user_id=body.user_id,
            context=body.context,
            ttl=config.session_ttl,
        )
    else:
        session_id = await create_session(
            config,
            agent_id=agent_id,
            user_id=body.user_id,
            context=body.context,
        )

    return OpenSessionResponse(session_id=session_id)


# -- Observe within session ---------------------------------------------------


@router.post(
    "/agents/{agent_id}/sessions/{session_id}/observe",
    response_model=SessionObserveResponse,
    operation_id="session_observe",
    tags=["sessions"],
    summary="Record an observation within an open session",
)
async def session_observe(
    agent_id: str,
    session_id: str,
    body: SessionObserveRequest,
    registry: RegistryDep,
    config: ConfigDep,
) -> SessionObserveResponse:
    """Record an observation within an open session."""
    session = await _load_session(config, registry, session_id, agent_id)
    imp = await registry.get(agent_id)

    if body.directions is not None:
        if body.agent_output is not None or body.user_response is not None:
            raise bad_request(
                "Provide either directions or (agent_output + user_response), not both."
            )
        async with await registry.get_op_lock(agent_id):
            await imp.observe_directions(
                user_id=session.user_id,
                directions=body.directions,
                context=body.context or session.context,
                scope=body.scope,
            )
    else:
        if body.agent_output is None or body.user_response is None:
            raise bad_request("Provide both agent_output and user_response, or provide directions.")
        async with await registry.get_op_lock(agent_id):
            await imp.observe(
                user_id=session.user_id,
                agent_output=body.agent_output,
                user_response=body.user_response,
                context=body.context or session.context,
                scope=body.scope,
            )

    return SessionObserveResponse()


# -- Policy within session ----------------------------------------------------


@router.post(
    "/agents/{agent_id}/sessions/{session_id}/policy",
    response_model=PolicyResponse,
    operation_id="session_policy",
    tags=["sessions"],
    summary="Compile a policy within an open session",
)
async def session_policy(
    agent_id: str,
    session_id: str,
    body: SessionPolicyRequest,
    registry: RegistryDep,
    config: ConfigDep,
) -> PolicyResponse:
    """Compile a policy within an open session, tracking retrieved memories.

    Constructs a temporary MemoryLoop and passes it to get_policy() so the
    library records which memories were retrieved and at what alpha weight.
    The session row is updated with this data so close() can reconstruct the
    loop for finalize_loop().
    """
    session = await _load_session(config, registry, session_id, agent_id)
    imp = await registry.get(agent_id)
    pg_pool = get_pg_pool(registry) if config.is_postgres else None

    # Create a MemoryLoop without registering it in imp._active_loops.
    # The server manages loop lifecycle via the sessions table, not via WeakSet.
    loop = MemoryLoop(
        user_id=session.user_id,
        session_id=session_id,
        imprint=imp,
    )

    effective_context = body.context or session.context
    pol = await imp.get_policy(
        user_id=session.user_id,
        context=effective_context,
        existing_instructions=body.existing_instructions,
        max_input_tokens=body.max_input_tokens,
        max_output_tokens=body.max_output_tokens,
        scopes=body.scopes,
        loop=loop,
    )

    # Persist loop state to the session row.
    retrieved_ids = list(loop.retrieved_ids)
    alpha_used = loop.alpha_used

    if pg_pool is not None:
        await pg_update_session_policy(
            pg_pool,
            session_id,
            retrieved_ids=retrieved_ids,
            alpha_used=alpha_used,
            context=effective_context,
        )
    else:
        await update_session_policy(
            config,
            session_id,
            retrieved_ids=retrieved_ids,
            alpha_used=alpha_used,
            context=effective_context,
        )

    # Log counterfactual data with the real alpha_used.
    await log_policy_event(
        registry=registry,
        config=config,
        agent_id=agent_id,
        user_id=session.user_id,
        session_id=session_id,
        retrieved_memories=pol.memories,
        filtered_memories=pol.dropped_memories,
        alpha_used=alpha_used,
        context=effective_context,
    )

    return PolicyResponse(
        policy_text=pol.text,
        memory_count=len(pol.memories),
        dropped_count=len(pol.dropped_memories),
        compiled_at=pol.compiled_at.isoformat(),
        memory_ids=[m.id for m in pol.memories],
    )


# -- Close session ------------------------------------------------------------


@router.post(
    "/agents/{agent_id}/sessions/{session_id}/close",
    response_model=CloseSessionResponse,
    operation_id="close_session",
    tags=["sessions"],
    summary="Close a session and apply the learning signal",
)
async def close_session_endpoint(
    agent_id: str,
    session_id: str,
    body: CloseSessionRequest,
    registry: RegistryDep,
    config: ConfigDep,
) -> CloseSessionResponse:
    """Close a session and apply the learning signal via finalize_loop().

    Reconstructs the MemoryLoop from the session row and calls
    finalize_loop(), which updates the bandit alpha tuner and gradient decay
    model based on the outcome signal.
    """
    pg_pool = get_pg_pool(registry) if config.is_postgres else None
    session = await _load_session(config, registry, session_id, agent_id)
    imp = await registry.get(agent_id)

    # Re-fetch memories by their IDs to get current state.
    retrieved_memories = []
    if session.retrieved_ids:
        all_mems = await imp._store.list_memories(  # type: ignore[attr-defined]
            agent_id, session.user_id, active_only=False
        )
        id_set = set(session.retrieved_ids)
        retrieved_memories = [m for m in all_mems if m.id in id_set]

    # Reconstruct MemoryLoop with persisted state.
    loop = MemoryLoop(
        user_id=session.user_id,
        session_id=session_id,
        imprint=imp,
    )
    loop.retrieved_ids = set(session.retrieved_ids)
    loop.retrieved_memories = retrieved_memories
    loop.alpha_used = session.alpha_used
    loop.context = session.context
    loop.closed = True  # prevent double-close inside finalize_loop

    if body.outcome is not None:
        loop.set_outcome(body.outcome, correction=body.correction)
    elif body.correction is not None:
        loop.correction = body.correction

    # Apply learning signal.
    await imp.finalize_loop(loop)

    # Mark closed in DB.
    if pg_pool is not None:
        await pg_close_session(
            pg_pool,
            session_id,
            outcome=body.outcome,
            correction=body.correction,
        )
    else:
        await close_session(
            config,
            session_id,
            outcome=body.outcome,
            correction=body.correction,
        )

    return CloseSessionResponse()
