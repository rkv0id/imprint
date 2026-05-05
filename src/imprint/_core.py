"""Imprint facade: top-level SDK entry point."""

import asyncio
import contextvars
import hashlib
import os
import uuid
import weakref
from collections.abc import AsyncGenerator, Coroutine
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from datetime import timedelta as _timedelta
from typing import Literal, Self, cast

from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.models import Model

from imprint.budget import HeuristicTokenCounter, truncate_to_budget
from imprint.decay import FSRSStaticDecay
from imprint.detect import detect_signal_heuristic
from imprint.prompts import attribute as attribute_prompt
from imprint.prompts import consolidate as consolidate_prompt
from imprint.prompts import memory as memory_prompt
from imprint.prompts import policy as policy_prompt
from imprint.prompts import scope as scope_prompt
from imprint.prompts import signal as signal_prompt
from imprint.prompts import validate as validate_prompt
from imprint.protocols import (
    AlphaTuner,
    Compiler,
    DecayModel,
    Embedder,
    EventLogger,
    MemoryStore,
    TokenCounter,
    VectorStore,
)
from imprint.retrieval import BanditAlphaTuner, StaticAlphaTuner, rrf_fuse, sanitize_fts_query
from imprint.store import NullEventLogger, SQLiteEventLogger, SQLiteMemoryStore
from imprint.types import (
    Memory,
    MemoryEvent,
    MemoryHealth,
    MemoryLineage,
    MemorySource,
    MemoryType,
    Signal,
    SignalType,
)

ProcessingMode = Literal["frugal", "balanced", "eager"]

DEFAULT_MODEL = "anthropic:claude-haiku-4-5-20251001"

_VALID_PROCESSING_MODES: frozenset[str] = frozenset({"frugal", "balanced", "eager"})


def _parse_dt(value: str) -> datetime:
    """Parse an ISO datetime string as returned by SQLite/Turso stores."""
    dt = datetime.fromisoformat(value)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt


class _SignalDetection(BaseModel):
    """Structured output for the signal-detection agent."""

    signal_type: SignalType | None = None


class _DerivedMemory(BaseModel):
    """Structured output for the memory-derivation agent."""

    memory_type: MemoryType
    content: str
    scope: str = "global"


class _ConsolidationDecision(BaseModel):
    """One decision in a consolidation pass: what to do with one existing memory."""

    memory_id: str
    action: Literal["merge", "contradict", "distinct", "scope_override"]


class _ConsolidationOutput(BaseModel):
    """Structured output for the consolidation agent."""

    decisions: list[_ConsolidationDecision] = []


class _BatchConsolidationDecision(BaseModel):
    """One decision in a batch consolidation pass.

    candidate_index is the 0-based index of the new memory within the batch.
    Only merge, contradict, and scope_override decisions are returned; distinct
    pairs are omitted.
    """

    candidate_index: int
    memory_id: str
    action: Literal["merge", "contradict", "scope_override"]


class _BatchConsolidationOutput(BaseModel):
    """Structured output for batch consolidation of multiple candidates."""

    decisions: list[_BatchConsolidationDecision] = []


class LLMCompiler:
    """Concrete Compiler implementation that uses a pydantic-ai agent.

    Exposed so callers can instantiate it directly and customize the model
    or override the agent for testing. Custom compiler strategies (concat,
    template-based, no-compile passthrough) can implement the Compiler
    protocol without subclassing this class.
    """

    def __init__(self, model: str | Model) -> None:
        self._agent: Agent[None, str] = Agent(
            model,
            instructions=policy_prompt.SYSTEM,
            model_settings={"temperature": 0.0},
            defer_model_check=True,
        )

    @property
    def agent(self) -> "Agent[None, str]":
        """The underlying pydantic-ai agent. Useful for test model overrides."""
        return self._agent

    async def compile(
        self,
        *,
        memories: list[Memory],
        agent_description: str | None,
        context: str | None,
        existing_instructions: str | None,
        max_tokens: int,
    ) -> str:
        user_prompt = policy_prompt.build_user_prompt(
            memories=memories,
            existing_instructions=existing_instructions,
            context=context,
            agent_description=agent_description,
        )
        result = await self._agent.run(
            user_prompt,
            model_settings={"temperature": 0.0, "max_tokens": max_tokens},
        )
        return result.output


class _DirectionVerdict(BaseModel):
    """One verdict in an eager direction validation pass."""

    verdict: Literal["directive", "hedge", "contradiction", "non-directive"]


class _ScopeOutput(BaseModel):
    """Structured output for the scope inference agent."""

    relevant_scopes: list[str] = []


class _ValidationOutput(BaseModel):
    """Structured output for the direction validation agent."""

    verdicts: list[_DirectionVerdict] = []


class _AttributionOutput(BaseModel):
    """Indices (1-based) of memories that should have ranked higher."""

    relevant_indices: list[int] = []


class MemoryLoop:
    """Tracks one get_policy / outcome cycle for learning updates.

    Obtain via open_loop() or the loop() async context manager. Pass to
    get_policy(loop=loop) so the system knows which memories were retrieved.
    Close with loop.close(outcome=...) when the interaction is done.

    Typical usage:

      # Simple -- context manager handles open and close.
      async with imprint.loop(user_id="u") as loop:
          policy = await imprint.get_policy(user_id="u", loop=loop)
          loop.set_outcome(0.9)

      # Agentic -- explicit open and close.
      loop = await imprint.open_loop(user_id="u", session_id="s1")
      policy = await imprint.get_policy(user_id="u", loop=loop)
      # ... tool calls, parallel steps, supervisor hops ...
      await loop.close(outcome=0.7)

      # No learning signal.
      policy = await imprint.get_policy(user_id="u")
    """

    def __init__(
        self,
        *,
        user_id: str,
        session_id: str | None = None,
        timeout: int = 3600,
        imprint: "Imprint",
    ) -> None:
        self.user_id = user_id
        self.session_id = session_id
        self.opened_at: datetime = datetime.now(UTC)
        self.timeout = timeout
        self.retrieved_ids: set[str] = set()
        self.retrieved_memories: list[Memory] = []
        self.alpha_used: float = 0.3
        self.context: str | None = None
        self.correction: str | None = None
        self.outcome: float | None = None
        self.closed: bool = False
        self._imprint_ref: weakref.ref[Imprint] = weakref.ref(imprint)

    @property
    def expires_at(self) -> datetime:
        return self.opened_at + _timedelta(seconds=self.timeout)

    def set_outcome(self, outcome: float, *, correction: str | None = None) -> None:
        """Record interaction quality. outcome: -1.0 failure, 0.0 neutral, 1.0 success.

        correction: optional description of what went wrong; enables memory
        attribution when outcome < 0 and embedder or eager mode is configured.
        """
        self.outcome = max(-1.0, min(1.0, outcome))
        if correction is not None:
            self.correction = correction

    async def close(
        self,
        outcome: float | None = None,
        *,
        correction: str | None = None,
    ) -> None:
        """Finalize the loop and apply any learning signal. Idempotent."""
        if self.closed:
            return
        self.closed = True
        if outcome is not None:
            self.set_outcome(outcome, correction=correction)
        elif correction is not None:
            self.correction = correction
        imp = self._imprint_ref()
        if imp is None:
            return
        await imp.finalize_loop(self)


@dataclass(slots=True)
class Policy:
    text: str
    memories: list[Memory] = field(default_factory=list)  # type: ignore[assignment]
    dropped_memories: list[Memory] = field(default_factory=list)  # type: ignore[assignment]
    compiled_at: datetime = field(default_factory=lambda: datetime.now(UTC))


class Imprint:
    def __init__(
        self,
        *,
        agent_id: str,
        model: str | Model = DEFAULT_MODEL,
        store: str | MemoryStore = "sqlite:///~/.imprint/imprint.db",
        compiler: Compiler | None = None,
        event_logger: EventLogger | None = None,
        decay_model: DecayModel | None = None,
        token_counter: TokenCounter | None = None,
        vector_store: VectorStore | None = None,
        embedder: Embedder | None = None,
        alpha_tuner: AlphaTuner | None = None,
        agent_description: str | None = None,
        processing_mode: ProcessingMode | None = None,
        scopes: list[str] | None = None,
    ) -> None:
        self.agent_id = agent_id

        self._ctor_processing_mode = processing_mode
        self._ctor_agent_description = agent_description
        self._ctor_scopes = scopes

        self.processing_mode: ProcessingMode = (
            processing_mode if processing_mode is not None else "balanced"
        )
        self.agent_description: str | None = agent_description

        if scopes is not None:
            seen: set[str] = set()
            deduped: list[str] = []
            for s in scopes:
                if s == "global" or s in seen:
                    continue
                seen.add(s)
                deduped.append(s)
            self.scopes: list[str] = deduped
        else:
            self.scopes = []

        if isinstance(store, str):
            store_inst: MemoryStore
            if _is_turso_url(store):
                from imprint.turso import TursoMemoryStore

                url, token = _parse_turso_url(store)
                store_inst = TursoMemoryStore(url, auth_token=token)
            else:
                store_inst = SQLiteMemoryStore(_parse_store_url(store))
            self._store: MemoryStore = store_inst
            self._owns_store = True
        else:
            self._store = store
            self._owns_store = False

        self._event_logger: EventLogger | None = event_logger
        self._decay_model: DecayModel = (
            decay_model if decay_model is not None else FSRSStaticDecay()
        )
        self._token_counter: TokenCounter = (
            token_counter if token_counter is not None else HeuristicTokenCounter()
        )
        self._vector_store: VectorStore | None = vector_store
        self._embedder: Embedder | None = embedder
        self._alpha_tuner: AlphaTuner = (
            alpha_tuner if alpha_tuner is not None else StaticAlphaTuner(alpha=0.3)
        )
        self._pending_tasks: set[asyncio.Task[None]] = set()
        self._active_loops: weakref.WeakSet[MemoryLoop] = weakref.WeakSet()

        # Compiler: use injected compiler or build the default LLMCompiler.
        # _compile_agent is kept as a direct reference to LLMCompiler's agent so
        # test overrides (imprint._compile_agent.override(...)) still work when no
        # custom compiler is provided.
        if compiler is not None:
            self._compiler: Compiler = compiler
            # No _compile_agent when a custom compiler is injected. Tests that
            # access _compile_agent always use the default path.
        else:
            _default_compiler = LLMCompiler(model)
            self._compiler = _default_compiler
            self._compile_agent: Agent[None, str] = _default_compiler.agent

        self._detect_agent: Agent[None, _SignalDetection] = Agent(
            model,
            output_type=_SignalDetection,
            instructions=signal_prompt.SYSTEM,
            model_settings={"temperature": 0.0},
            defer_model_check=True,
        )
        self._derive_agent: Agent[None, _DerivedMemory] = Agent(
            model,
            output_type=_DerivedMemory,
            instructions=memory_prompt.SYSTEM,
            model_settings={"temperature": 0.0},
            defer_model_check=True,
        )
        self._consolidate_agent: Agent[None, _ConsolidationOutput] = Agent(
            model,
            output_type=_ConsolidationOutput,
            instructions=consolidate_prompt.SYSTEM,
            model_settings={"temperature": 0.0},
            defer_model_check=True,
        )
        self._batch_consolidate_agent: Agent[None, _BatchConsolidationOutput] = Agent(
            model,
            output_type=_BatchConsolidationOutput,
            instructions=consolidate_prompt.BATCH_SYSTEM,
            model_settings={"temperature": 0.0},
            defer_model_check=True,
        )
        self._validate_agent: Agent[None, _ValidationOutput] = Agent(
            model,
            output_type=_ValidationOutput,
            instructions=validate_prompt.SYSTEM,
            model_settings={"temperature": 0.0},
            defer_model_check=True,
        )
        self._attribute_agent: Agent[None, _AttributionOutput] = Agent(
            model,
            output_type=_AttributionOutput,
            instructions=attribute_prompt.SYSTEM,
            model_settings={"temperature": 0.0},
            defer_model_check=True,
        )
        self._scope_agent: Agent[None, _ScopeOutput] = Agent(
            model,
            output_type=_ScopeOutput,
            instructions=scope_prompt.SYSTEM,
            model_settings={"temperature": 0.0},
            defer_model_check=True,
        )

    async def connect(self) -> None:
        await self._store.connect()
        await self._store.init_schema()
        await self._sync_agent_config()
        if self._event_logger is None:
            if isinstance(self._store, SQLiteMemoryStore):
                self._event_logger = SQLiteEventLogger(self._store)
            else:
                self._event_logger = NullEventLogger()

    async def _sync_agent_config(self) -> None:
        stored = await self._store.get_agent_config(self.agent_id)

        if self._ctor_processing_mode is not None:
            self.processing_mode = self._ctor_processing_mode  # pyright: ignore[reportAttributeAccessIssue]
        elif stored is not None and stored.processing_mode in _VALID_PROCESSING_MODES:
            self.processing_mode = cast(ProcessingMode, stored.processing_mode)
        else:
            self.processing_mode = "balanced"

        if self._ctor_agent_description is not None:
            self.agent_description = self._ctor_agent_description
        elif stored is not None:
            self.agent_description = stored.agent_description

        if self._ctor_scopes is not None:
            pass
        elif stored is not None and stored.scopes is not None:
            seen: set[str] = set()
            deduped: list[str] = []
            for s in stored.scopes:
                if s == "global" or s in seen:
                    continue
                seen.add(s)
                deduped.append(s)
            self.scopes = deduped

        await self._store.put_agent_config(
            agent_id=self.agent_id,
            processing_mode=self.processing_mode,
            agent_description=self.agent_description,
            scopes=self.scopes,
        )
        if (
            isinstance(self._alpha_tuner, BanditAlphaTuner)
            and stored is not None
            and stored.alpha_tuner_state is not None
        ):
            import contextlib
            import json as _json

            with contextlib.suppress(Exception):
                self._alpha_tuner.set_state(_json.loads(stored.alpha_tuner_state))

        if stored is not None and stored.gradient_state is not None:
            _decay = self._decay_model
            if hasattr(_decay, "set_state"):
                import contextlib

                with contextlib.suppress(Exception):
                    _decay.set_state(stored.gradient_state)  # type: ignore[union-attr]

    async def close(self) -> None:
        await self.drain()
        if self._owns_store:
            await self._store.close()

    async def __aenter__(self) -> Self:
        await self.connect()
        return self

    async def __aexit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        await self.close()

    @classmethod
    def from_env(cls) -> "Imprint":
        """Construct an Imprint instance from environment variables.

        Required:
          IMPRINT_AGENT_ID      -- agent identifier

        Optional:
          IMPRINT_DATABASE_URL  -- store URL (default: sqlite:///~/.imprint/imprint.db)
          IMPRINT_MODEL         -- pydantic-ai model string (default: claude-haiku-4-5)
          IMPRINT_MODE          -- frugal | balanced | eager (default: balanced)
        """
        agent_id = os.environ["IMPRINT_AGENT_ID"]
        database_url = os.environ.get("IMPRINT_DATABASE_URL", "sqlite:///~/.imprint/imprint.db")
        model = os.environ.get("IMPRINT_MODEL", DEFAULT_MODEL)
        mode_str = os.environ.get("IMPRINT_MODE")
        mode: ProcessingMode | None = (
            cast(ProcessingMode, mode_str) if mode_str in _VALID_PROCESSING_MODES else None
        )
        return cls(
            agent_id=agent_id,
            store=database_url,
            model=model,
            processing_mode=mode,
        )

    async def list_memories(
        self,
        user_id: str,
        *,
        scopes: list[str] | None = None,
    ) -> list[Memory]:
        """Return the active memory list for a user, optionally filtered by scopes."""
        return await self._store.list_memories(self.agent_id, user_id, scopes=scopes)

    async def pin_memory(self, memory_id: str) -> None:
        """Pin a memory so it is never dropped by the token budget truncation.

        Pinned memories are always included in compiled policies regardless
        of memory count or token pressure. Use for memories that must always
        be present -- critical project conventions, hard constraints, etc.
        """
        await self._store.set_pinned(memory_id, True)
        # Pinning changes the memory's behavior in budget truncation but does
        # not change its content, so the cache key (based on memory IDs and
        # updated_at) will reflect the new updated_at after set_pinned writes it.

    async def deactivate_memory(self, user_id: str, memory_id: str) -> bool:
        """Deactivate a specific memory. Returns True if found and deactivated."""
        found = await self._store.deactivate_memory(memory_id)
        if found:
            await self._store.invalidate_cached_policies(self.agent_id, user_id)
        return found

    async def search_memories(
        self,
        user_id: str,
        query: str,
        *,
        scope: str | None = None,
    ) -> list[Memory]:
        """Search memories by semantic similarity. Falls back to list order without embedder."""
        scopes = [scope] if scope else None
        all_memories = await self._store.list_memories(self.agent_id, user_id, scopes=scopes)
        if not all_memories:
            return []
        if self._embedder is not None and self._vector_store is not None:
            try:
                embedding = await self._embedder.embed(query)
                hits = await self._vector_store.search(embedding, top_k=len(all_memories))
                hit_ids = {mid for mid, _ in hits}
                ordered = [m for mid, _ in hits for m in all_memories if m.id == mid]
                remaining = [m for m in all_memories if m.id not in hit_ids]
                return ordered + remaining
            except Exception:
                pass
        return all_memories

    async def list_events(
        self,
        user_id: str,
        *,
        memory_id: str | None = None,
        limit: int = 50,
    ) -> list[MemoryEvent]:
        """Return logged events for a user's memories, newest first.

        If memory_id is given, scoped to that memory only. Otherwise returns
        the most recent events across all of this user's memories.
        """
        rows = await self._store.list_events(
            self.agent_id,
            user_id,
            memory_id=memory_id,
            limit=limit,
        )
        return [
            MemoryEvent(
                memory_id=row["memory_id"],
                event_type=row["event_type"],
                detail=row.get("detail"),
                occurred_at=_parse_dt(row["occurred_at"]),
            )
            for row in rows
        ]

    async def memory_lineage(self, memory_id: str) -> MemoryLineage:
        """Return the full history of one memory.

        Includes the memory itself, the signal that created it, any memories
        it superseded, the memory that superseded it (if any), and all logged
        events.
        """
        target = await self._store.get_memory(memory_id)
        if target is None:
            raise KeyError(f"memory {memory_id!r} not found")

        successor, _ = await self._store.get_memory_with_supersession(memory_id)

        signal = await self._store.get_creating_signal(memory_id)
        superseded_memories = await self._store.get_superseded_memories(memory_id)

        events = await self.list_events(
            target.user_id or "",
            memory_id=memory_id,
            limit=200,
        )

        return MemoryLineage(
            memory=target,
            created_by_signal=signal,
            superseded_memories=superseded_memories,
            superseded_by=successor,
            events=events,
        )

    async def memory_health(self, user_id: str) -> MemoryHealth:
        """Return aggregate health statistics for a user's memory store."""
        all_memories = await self._store.list_memories(self.agent_id, user_id, active_only=False)
        if not all_memories:
            return MemoryHealth(
                total=0,
                active=0,
                by_scope={},
                by_type={},
                pinned=0,
                avg_recall_count=0.0,
            )

        active_memories = [m for m in all_memories if m.active]
        by_scope: dict[str, int] = {}
        by_type: dict[str, int] = {}
        for m in active_memories:
            by_scope[m.scope] = by_scope.get(m.scope, 0) + 1
            by_type[m.type.value] = by_type.get(m.type.value, 0) + 1

        pinned_count = sum(1 for m in active_memories if m.pinned)
        avg_recall = (
            sum(m.recall_count for m in active_memories) / len(active_memories)
            if active_memories
            else 0.0
        )

        valid_froms = [m.valid_from for m in active_memories]
        oldest = min(valid_froms) if valid_froms else None
        newest = max(valid_froms) if valid_froms else None

        return MemoryHealth(
            total=len(all_memories),
            active=len(active_memories),
            by_scope=by_scope,
            by_type=by_type,
            pinned=pinned_count,
            avg_recall_count=round(avg_recall, 3),
            oldest_active=oldest,
            newest_active=newest,
        )

    async def drain(self) -> None:
        """Await all pending background learning tasks.

        Learning updates (bandit, gradient decay, attribution) are scheduled
        as asyncio tasks so observe() returns immediately after memory
        persistence. Call drain() when you need to ensure all learning has
        completed -- primarily useful in tests.
        """
        if self._pending_tasks:
            await asyncio.gather(*self._pending_tasks, return_exceptions=True)

    async def open_loop(
        self,
        user_id: str,
        *,
        session_id: str | None = None,
        timeout: int = 3600,
    ) -> MemoryLoop:
        """Create and register an explicit feedback loop.

        Pass the returned MemoryLoop to get_policy(loop=loop) so retrieved
        memories and alpha are recorded. Call loop.close(outcome=...) when
        the interaction is done to apply the learning signal.
        """
        ml = MemoryLoop(
            user_id=user_id,
            session_id=session_id,
            timeout=timeout,
            imprint=self,
        )
        self._active_loops.add(ml)
        return ml

    @asynccontextmanager
    async def loop(
        self,
        user_id: str,
        *,
        session_id: str | None = None,
        timeout: int = 3600,
    ) -> AsyncGenerator[MemoryLoop, None]:
        """Async context manager that opens a MemoryLoop and closes it on exit.

        Closes with outcome=0.0 (neutral) if set_outcome() was never called.
        If set_outcome() was called, that outcome is used.

          async with imprint.loop(user_id="u") as loop:
              policy = await imprint.get_policy(user_id="u", loop=loop)
              loop.set_outcome(0.9)
        """
        ml = await self.open_loop(user_id=user_id, session_id=session_id, timeout=timeout)
        try:
            yield ml
        finally:
            if not ml.closed:
                if ml.outcome is None:
                    ml.outcome = 0.0
                await ml.close()

    def _sweep_expired_loops(self) -> None:
        """Finalize loops that have exceeded their timeout.

        Called lazily on every get_policy() and observe() call. Expired loops
        receive outcome=-0.15 (small penalty for abandoned retrieval). GC'd
        loops produce no signal.
        """
        now = datetime.now(UTC)
        for ml in list(self._active_loops):
            if ml.closed or ml.expires_at >= now:
                continue
            ml.closed = True
            ml.outcome = -0.15
            self._schedule_learning(self.finalize_loop(ml))

    async def finalize_loop(self, loop: MemoryLoop) -> None:
        """Apply learning signal for a closed loop. Called by loop.close() and expiry sweep."""
        self._active_loops.discard(loop)
        if loop.outcome is None:
            return
        now = datetime.now(UTC)
        outcome = loop.outcome
        attribution_text = loop.correction or loop.context
        if outcome < 0.0 and attribution_text is not None:
            if self._embedder is not None and self._vector_store is not None:
                await self._embedding_attribution(
                    loop=loop, correction=attribution_text, outcome=outcome, now=now
                )
                return
            if self.processing_mode == "eager":
                await self._llm_attribution(loop=loop, correction=attribution_text, now=now)
                return
        await self._apply_feedback(loop=loop, outcome=outcome, now=now)

    def _schedule_learning(self, coro: "Coroutine[object, object, None]") -> None:
        task: asyncio.Task[None] = asyncio.create_task(coro, context=contextvars.copy_context())
        self._pending_tasks.add(task)
        task.add_done_callback(self._pending_tasks.discard)

    async def observe(
        self,
        *,
        user_id: str,
        agent_output: str,
        user_response: str,
        context: str | None = None,
        scope: str | None = None,
    ) -> None:
        self._sweep_expired_loops()

        signal_type = await self._detect_signal(
            agent_output=agent_output, user_response=user_response
        )
        if signal_type is None:
            return

        derived = await self._derive_memory(
            agent_output=agent_output,
            user_response=user_response,
            signal_type=signal_type,
        )

        existing = await self._store.list_memories(self.agent_id, user_id)

        chosen_scope = scope if scope is not None else derived.scope

        now = datetime.now(UTC)
        memory = Memory(
            id=_new_id("mem"),
            agent_id=self.agent_id,
            user_id=user_id,
            type=derived.memory_type,
            scope=_resolve_scope(chosen_scope),
            content=derived.content,
            source=MemorySource.DETECTED,
            valid_from=now,
            created_at=now,
            updated_at=now,
        )
        signal = Signal(
            id=_new_id("sig"),
            agent_id=self.agent_id,
            user_id=user_id,
            signal_type=signal_type,
            content=user_response,
            context=context,
            created_at=now,
        )

        await self._store.invalidate_cached_policies(self.agent_id, user_id)

        await self._store.insert_signal(signal)
        await self._store.insert_memory(memory)
        await self._store.link_signal_to_memory(memory_id=memory.id, signal_id=signal.id)

        if self._embedder is not None and self._vector_store is not None:
            embedding = await self._embedder.embed(memory.content)
            await self._vector_store.upsert(memory.id, embedding)

        await self._consolidate_against_existing(
            candidate=memory,
            candidate_signal_type=signal_type,
            existing=existing,
        )

    async def observe_directions(
        self,
        *,
        user_id: str,
        directions: list[str],
        scope: str | None = None,
        context: str | None = None,
        source: MemorySource = MemorySource.USER_EDIT,
    ) -> list[Memory]:
        """Persist user-supplied directives directly as memories.

        All directions in one call are derived and stored first, then
        consolidated against pre-existing memories in a single LLM call
        (balanced/eager) or vector pass (frugal+vector). Detection is
        skipped -- directions are explicit instructions, not inferred signals.

        In eager mode a batched LLM validation pre-pass filters out hedges,
        contradictions, and non-directives before derivation runs.
        """
        if not directions:
            return []

        candidates = directions
        if self.processing_mode == "eager":
            candidates = await self._validate_directions(directions)
        if not candidates:
            return []

        # Snapshot existing memories and combined scope set once before any
        # new memories are inserted. Both are used during the batch.
        existing = await self._store.list_memories(self.agent_id, user_id)
        available_scopes = await self._combined_scopes()

        # Derive and store all new memories first.
        memories: list[Memory] = []
        for direction in candidates:
            memory = await self._derive_and_store_direction(
                user_id=user_id,
                direction=direction,
                scope=scope,
                context=context,
                source=source,
                available_scopes=available_scopes,
            )
            memories.append(memory)

        # Batch consolidation: one LLM call for all new memories against
        # the pre-existing snapshot.
        await self._consolidate_directions_batch(
            candidates=memories,
            existing=existing,
        )

        return memories

    async def _consolidate_directions_batch(
        self,
        *,
        candidates: list[Memory],
        existing: list[Memory],
    ) -> None:
        """Consolidate a batch of new direction memories against existing ones.

        frugal+vector: vector merge per candidate, no LLM.
        frugal (no vector): no-op.
        balanced/eager (no vector): one LLM call for the whole batch.
        balanced (vector): prefilter per candidate (top-10, threshold 0.5),
          take the union, one LLM call on the reduced set.
        eager (vector): same prefilter, one LLM call.
        """
        if not existing:
            return

        if self.processing_mode == "frugal":
            if self._embedder is not None and self._vector_store is not None:
                for candidate in candidates:
                    await self._consolidate_frugal_vector(candidate=candidate, existing=existing)
            return

        # balanced / eager: build the set of existing memories to check against.
        existing_to_check = existing
        if self._embedder is not None and self._vector_store is not None:
            seen_ids: set[str] = set()
            filtered: list[Memory] = []
            for candidate in candidates:
                hits = await self._prefilter_candidates(
                    candidate=candidate, existing=existing, top_k=10, threshold=0.5
                )
                for m in hits:
                    if m.id not in seen_ids:
                        seen_ids.add(m.id)
                        filtered.append(m)
            existing_to_check = filtered

        if not existing_to_check:
            return

        prompt = consolidate_prompt.build_batch_user_prompt(
            candidates=candidates,
            candidate_signal_type=SignalType.DIRECTION.value,
            existing=existing_to_check,
        )
        result = await self._batch_consolidate_agent.run(prompt)

        existing_by_id = {m.id: m for m in existing_to_check}
        deactivated: set[str] = set()
        now = datetime.now(UTC)

        for decision in result.output.decisions:
            if decision.candidate_index < 0 or decision.candidate_index >= len(candidates):
                continue
            if decision.memory_id not in existing_by_id:
                continue
            if decision.memory_id in deactivated:
                # already handled by an earlier candidate in this batch
                continue

            candidate = candidates[decision.candidate_index]
            existing_mem = existing_by_id[decision.memory_id]

            if decision.action == "merge":
                new_stability = self._decay_model.update_on_merge(existing_mem)
                await self._store.update_memory_stability(decision.memory_id, new_stability)
                await self._store.deactivate_memory(decision.memory_id, superseded_by=candidate.id)
                if self._vector_store is not None:
                    await self._vector_store.delete(decision.memory_id)
                if self._event_logger is not None:
                    await self._event_logger.log(
                        decision.memory_id,
                        "merge",
                        {"superseded_by": candidate.id},
                    )
                deactivated.add(decision.memory_id)
                self._schedule_learning(
                    self._update_alpha_tuner(candidate.user_id, decision.memory_id)
                )
            elif decision.action == "contradict":
                new_stability = self._decay_model.update_on_contradict(existing_mem)
                await self._store.update_memory_stability(decision.memory_id, new_stability)
                await self._store.deactivate_memory(
                    decision.memory_id,
                    superseded_by=candidate.id,
                    valid_until=now,
                )
                if self._vector_store is not None:
                    await self._vector_store.delete(decision.memory_id)
                await self._store.mark_signals_contradicted(decision.memory_id)
                if self._event_logger is not None:
                    await self._event_logger.log(
                        decision.memory_id,
                        "contradict",
                        {"superseded_by": candidate.id},
                    )
                deactivated.add(decision.memory_id)
                self._schedule_learning(
                    self._update_alpha_tuner(candidate.user_id, decision.memory_id)
                )
            elif decision.action == "scope_override":
                if self._event_logger is not None:
                    await self._event_logger.log(
                        decision.memory_id,
                        "scope_override",
                        {"overridden_by": candidate.id, "override_scope": candidate.scope},
                    )

    async def _derive_and_store_direction(
        self,
        *,
        user_id: str,
        direction: str,
        scope: str | None,
        context: str | None,
        source: MemorySource,
        available_scopes: list[str],
    ) -> Memory:
        """Derive and store one direction memory. Does not run consolidation.

        Consolidation for observe_directions() is batched separately by the
        caller after all directions are stored.
        """
        if self.processing_mode == "frugal":
            derived = _derive_memory_frugal(
                user_response=direction, signal_type=SignalType.DIRECTION
            )
        else:
            prompt = memory_prompt.build_user_prompt(
                agent_output="",
                user_response=direction,
                signal_type=SignalType.DIRECTION.value,
                available_scopes=available_scopes,
            )
            result = await self._derive_agent.run(prompt)
            derived = result.output

        chosen_scope = scope if scope is not None else derived.scope
        now = datetime.now(UTC)
        memory = Memory(
            id=_new_id("mem"),
            agent_id=self.agent_id,
            user_id=user_id,
            type=derived.memory_type,
            scope=_resolve_scope(chosen_scope),
            content=derived.content,
            source=source,
            valid_from=now,
            created_at=now,
            updated_at=now,
        )

        await self._store.invalidate_cached_policies(self.agent_id, user_id)
        await self._store.insert_memory(memory)

        if self._embedder is not None and self._vector_store is not None:
            embedding = await self._embedder.embed(memory.content)
            await self._vector_store.upsert(memory.id, embedding)

        return memory

    async def _apply_feedback(self, *, loop: MemoryLoop, outcome: float, now: datetime) -> None:
        bandit_reward = max(0.0, outcome)
        await self._alpha_tuner.update(loop.alpha_used, bandit_reward)

        _decay = self._decay_model
        if hasattr(_decay, "learn"):
            for m in loop.retrieved_memories:
                _decay.learn(m, now, outcome)  # type: ignore[union-attr,unknown-argument-type]
            if hasattr(_decay, "get_state"):
                state = _decay.get_state()  # type: ignore[union-attr]
                await self._store.put_gradient_state(self.agent_id, state)  # type: ignore[arg-type]

    async def _embedding_attribution(
        self,
        *,
        loop: MemoryLoop,
        correction: str,
        outcome: float,
        now: datetime,
    ) -> None:
        assert self._embedder is not None and self._vector_store is not None
        embedding = await self._embedder.embed(correction)
        hits = await self._vector_store.search(embedding, top_k=1)
        if not hits:
            await self._apply_feedback(loop=loop, outcome=outcome, now=now)
            return

        closest_id, _ = hits[0]
        _decay = self._decay_model
        for i, m in enumerate(loop.retrieved_memories):
            if m.id == closest_id:
                rank_reward = 1.0 - (i / max(len(loop.retrieved_memories), 1))
                await self._alpha_tuner.update(loop.alpha_used, rank_reward)
                if hasattr(_decay, "learn"):
                    _decay.learn(m, now, 1.0)  # type: ignore[union-attr,unknown-argument-type]
            else:
                if hasattr(_decay, "learn"):
                    _decay.learn(m, now, -0.2)  # type: ignore[union-attr,unknown-argument-type]

        if hasattr(_decay, "get_state"):
            state = _decay.get_state()  # type: ignore[union-attr]
            await self._store.put_gradient_state(self.agent_id, state)  # type: ignore[arg-type]

    async def _llm_attribution(
        self,
        *,
        loop: MemoryLoop,
        correction: str,
        now: datetime,
    ) -> None:
        if not loop.retrieved_memories:
            return
        prompt = attribute_prompt.build_user_prompt(
            correction=correction, memories=loop.retrieved_memories
        )
        result = await self._attribute_agent.run(prompt)
        attributed = {
            loop.retrieved_memories[i - 1].id
            for i in result.output.relevant_indices
            if 1 <= i <= len(loop.retrieved_memories)
        }
        _decay = self._decay_model
        for i, m in enumerate(loop.retrieved_memories):
            if m.id in attributed:
                rank_reward = 1.0 - (i / max(len(loop.retrieved_memories), 1))
                await self._alpha_tuner.update(loop.alpha_used, rank_reward)
                if hasattr(_decay, "learn"):
                    _decay.learn(m, now, 2.0)  # type: ignore[union-attr,unknown-argument-type]
            else:
                if hasattr(_decay, "learn"):
                    _decay.learn(m, now, -0.3)  # type: ignore[union-attr,unknown-argument-type]

        if hasattr(_decay, "get_state"):
            state = _decay.get_state()  # type: ignore[union-attr]
            await self._store.put_gradient_state(self.agent_id, state)  # type: ignore[arg-type]

    async def _validate_directions(self, directions: list[str]) -> list[str]:
        prompt = validate_prompt.build_user_prompt(directions=directions)
        result = await self._validate_agent.run(prompt)
        verdicts = result.output.verdicts
        passed: list[str] = []
        for i, direction in enumerate(directions):
            if i < len(verdicts) and verdicts[i].verdict == "directive":
                passed.append(direction)
        return passed

    async def get_policy(
        self,
        *,
        user_id: str,
        context: str | None = None,
        existing_instructions: str | None = None,
        max_input_tokens: int = 8000,
        max_output_tokens: int = 3000,
        on_budget_exceeded: Literal["truncate", "error"] = "truncate",
        scopes: list[str] | None = None,
        loop: MemoryLoop | None = None,
    ) -> Policy:
        self._sweep_expired_loops()

        effective_scopes: list[str] | None = scopes
        if scopes is None and context is not None:
            inferred = await self._infer_scopes(context)
            effective_scopes = inferred  # None means fetch-all

        all_memories = await self._store.list_memories(
            self.agent_id, user_id, scopes=effective_scopes
        )
        if not all_memories:
            return Policy(text="", memories=[], dropped_memories=[])

        now = datetime.now(UTC)

        alpha = 0.3
        if (
            self._embedder is not None
            and self._vector_store is not None
            and context is not None
            and self.processing_mode != "frugal"
        ):
            alpha = self._alpha_tuner.get_alpha(context)
            kept_memories = await self._hybrid_retrieve(
                candidates=all_memories,
                context=context,
                alpha=alpha,
            )
        else:
            kept_memories = all_memories
        kept, dropped = truncate_to_budget(
            memories=kept_memories,
            max_input_tokens=max_input_tokens,
            on_budget_exceeded=on_budget_exceeded,
            decay_model=self._decay_model,
            counter=self._token_counter,
            context=context,
            existing_instructions=existing_instructions,
            agent_description=self.agent_description,
            now=now,
        )

        if loop is not None:
            loop.retrieved_ids = {m.id for m in kept}
            loop.retrieved_memories = list(kept)
            loop.alpha_used = alpha
            loop.context = context

        cache_key = _policy_cache_key(
            agent_id=self.agent_id,
            user_id=user_id,
            memories=kept,
            context=context,
            existing_instructions=existing_instructions,
            max_output_tokens=max_output_tokens,
            scopes=effective_scopes,
        )

        cached = await self._store.get_cached_policy(cache_key)
        if cached is not None:
            cached_text, cached_at = cached
            await self._apply_recall(kept)
            return Policy(
                text=cached_text,
                memories=kept,
                dropped_memories=dropped,
                compiled_at=cached_at,
            )

        compiled_at = datetime.now(UTC)
        compiled_text = await self._compiler.compile(
            memories=kept,
            agent_description=self.agent_description,
            context=context,
            existing_instructions=existing_instructions,
            max_tokens=max_output_tokens,
        )
        await self._store.put_cached_policy(
            cache_key=cache_key,
            agent_id=self.agent_id,
            user_id=user_id,
            policy_text=compiled_text,
            compiled_at=compiled_at,
        )
        await self._apply_recall(kept)
        return Policy(
            text=compiled_text,
            memories=kept,
            dropped_memories=dropped,
            compiled_at=compiled_at,
        )

    async def _prefilter_candidates(
        self,
        *,
        candidate: Memory,
        existing: list[Memory],
        top_k: int,
        threshold: float,
    ) -> list[Memory]:
        assert self._embedder is not None and self._vector_store is not None
        embedding = await self._embedder.embed(candidate.content)
        hits = await self._vector_store.search(embedding, top_k=top_k)
        existing_by_id = {m.id: m for m in existing}
        result: list[Memory] = []
        for hit_id, distance in hits:
            if hit_id not in existing_by_id:
                continue
            similarity = 1.0 - distance
            if similarity >= threshold:
                result.append(existing_by_id[hit_id])
        return result

    async def _consolidate_frugal_vector(
        self,
        *,
        candidate: Memory,
        existing: list[Memory],
    ) -> None:
        assert self._embedder is not None and self._vector_store is not None
        embedding = await self._embedder.embed(candidate.content)
        hits = await self._vector_store.search(embedding, top_k=5)
        existing_by_id = {m.id: m for m in existing}
        for hit_id, distance in hits:
            if hit_id not in existing_by_id:
                continue
            similarity = 1.0 - distance
            if similarity >= 0.85:
                existing_mem = existing_by_id[hit_id]
                new_stability = self._decay_model.update_on_merge(existing_mem)
                await self._store.update_memory_stability(hit_id, new_stability)
                await self._store.deactivate_memory(hit_id, superseded_by=candidate.id)
                await self._vector_store.delete(hit_id)
                if self._event_logger is not None:
                    await self._event_logger.log(hit_id, "merge", {"superseded_by": candidate.id})

    async def _update_alpha_tuner(self, user_id: str | None, memory_id: str) -> None:
        if user_id is None:
            return
        updated = False
        for ml in list(self._active_loops):
            if ml.user_id != user_id or not ml.retrieved_ids:
                continue
            reward = 1.0 if memory_id in ml.retrieved_ids else 0.0
            await self._alpha_tuner.update(ml.alpha_used, reward)
            updated = True
        if updated and isinstance(self._alpha_tuner, BanditAlphaTuner):
            import json as _json

            await self._store.put_alpha_tuner_state(
                self.agent_id, _json.dumps(self._alpha_tuner.get_state())
            )

    async def _combined_scopes(self) -> list[str]:
        """Return the union of live DB scopes and constructor hint scopes.

        Live DB scopes (scopes with at least one active memory) come first.
        Constructor hint scopes not yet in the DB are appended after.
        Global is always excluded -- it is implicit, not a candidate.
        """
        live = await self._store.list_scopes(self.agent_id)
        seen = set(live)
        for s in self.scopes:
            if s not in seen:
                live.append(s)
                seen.add(s)
        return live

    async def _infer_scopes(self, context: str) -> list[str] | None:
        """Infer relevant scopes from context.

        Returns a list of inferred scopes (may be empty list only for an empty
        candidate set), or None when inference cannot narrow the scope and the
        caller should fall back to fetch-all.

        Frugal: cosine similarity between context embedding and each scope
        name embedding. Includes scopes with similarity >= 0.6, or the
        top scope if the max similarity is >= 0.4.

        Balanced: same embedding similarity, but falls through to an LLM
        call when the result is ambiguous (top score in 0.4-0.7 or top-vs-
        second gap < 0.2).

        Eager: LLM call directly.
        """
        candidate_scopes = await self._combined_scopes()
        if not candidate_scopes:
            return None

        if self.processing_mode == "eager":
            return await self._infer_scopes_llm(context, candidate_scopes)

        if self._embedder is None:
            return None

        try:
            all_vecs = await self._embedder.embed_batch([context, *candidate_scopes])
            ctx_vec = all_vecs[0]
            scope_vecs = all_vecs[1:]
        except Exception:
            return None

        scores = [_cosine(ctx_vec, sv) for sv in scope_vecs]
        paired = sorted(zip(scores, candidate_scopes, strict=True), reverse=True)

        if not paired:
            return None

        top_score = paired[0][0]
        second_score = paired[1][0] if len(paired) > 1 else 0.0
        gap = top_score - second_score

        if self.processing_mode == "balanced" and (
            top_score < 0.8 and (top_score >= 0.4 and gap < 0.2)
        ):
            result = await self._infer_scopes_llm(context, candidate_scopes)
            if result is not None:
                return result

        return [s for score, s in paired if score >= 0.6] or None

    async def _infer_scopes_llm(
        self, context: str, candidate_scopes: list[str]
    ) -> list[str] | None:
        try:
            prompt = scope_prompt.build_user_prompt(context=context, scope_names=candidate_scopes)
            result = await self._scope_agent.run(prompt)
            return [s for s in result.output.relevant_scopes if s in set(candidate_scopes)]
        except Exception:
            return None

    async def _hybrid_retrieve(
        self,
        *,
        candidates: list[Memory],
        context: str,
        alpha: float,
    ) -> list[Memory]:
        assert self._embedder is not None and self._vector_store is not None
        candidate_ids = {m.id for m in candidates}
        n = len(candidates)

        fts_query = sanitize_fts_query(context)
        fts_results = await self._store.search_fts(fts_query, candidate_ids, limit=n)
        sparse_ranks: dict[str, int] = {mid: i + 1 for i, (mid, _) in enumerate(fts_results)}

        embedding = await self._embedder.embed(context)
        dense_results = await self._vector_store.search(embedding, top_k=n)
        dense_results = [(mid, dist) for mid, dist in dense_results if mid in candidate_ids]
        dense_ranks: dict[str, int] = {mid: i + 1 for i, (mid, _) in enumerate(dense_results)}

        ranked_ids = rrf_fuse(
            candidates=[m.id for m in candidates],
            sparse_ranks=sparse_ranks,
            dense_ranks=dense_ranks,
            alpha=alpha,
        )
        by_id = {m.id: m for m in candidates}
        return [by_id[mid] for mid in ranked_ids]

    async def _apply_recall(self, memories: list[Memory]) -> None:
        for m in memories:
            await self._store.increment_recall_count(m.id)
            new_stability = self._decay_model.update_on_recall(m)
            if new_stability != m.stability:
                await self._store.update_memory_stability(m.id, new_stability)
            if self._event_logger is not None:
                await self._event_logger.log(m.id, "recall")

    async def _detect_signal(self, *, agent_output: str, user_response: str) -> SignalType | None:
        if self.processing_mode == "eager":
            return await self._detect_signal_llm(
                agent_output=agent_output, user_response=user_response
            )

        heuristic = detect_signal_heuristic(user_response)
        if self.processing_mode == "frugal":
            return heuristic
        if heuristic is not None:
            return heuristic
        return await self._detect_signal_llm(agent_output=agent_output, user_response=user_response)

    async def _detect_signal_llm(
        self, *, agent_output: str, user_response: str
    ) -> SignalType | None:
        prompt = signal_prompt.build_user_prompt(
            agent_output=agent_output, user_response=user_response
        )
        result = await self._detect_agent.run(prompt)
        return result.output.signal_type

    async def _derive_memory(
        self,
        *,
        agent_output: str,
        user_response: str,
        signal_type: SignalType,
    ) -> _DerivedMemory:
        if self.processing_mode == "frugal":
            return _derive_memory_frugal(user_response=user_response, signal_type=signal_type)

        prompt = memory_prompt.build_user_prompt(
            agent_output=agent_output,
            user_response=user_response,
            signal_type=signal_type.value,
            available_scopes=await self._combined_scopes(),
        )
        result = await self._derive_agent.run(prompt)
        return result.output

    async def _consolidate_against_existing(
        self,
        *,
        candidate: Memory,
        candidate_signal_type: SignalType,
        existing: list[Memory],
    ) -> None:
        if not existing:
            return

        if self.processing_mode == "frugal":
            if self._embedder is not None and self._vector_store is not None:
                await self._consolidate_frugal_vector(candidate=candidate, existing=existing)
            return

        candidates = existing
        if (
            self.processing_mode == "balanced"
            and self._embedder is not None
            and self._vector_store is not None
        ):
            candidates = await self._prefilter_candidates(
                candidate=candidate, existing=existing, top_k=10, threshold=0.5
            )
        if not candidates:
            return

        prompt = consolidate_prompt.build_user_prompt(
            candidate_type=candidate.type.value,
            candidate_content=candidate.content,
            candidate_scope=candidate.scope,
            candidate_signal_type=candidate_signal_type.value,
            existing=candidates,
        )
        result = await self._consolidate_agent.run(prompt)

        existing_by_id = {m.id: m for m in candidates}
        existing_ids = existing_by_id.keys()
        now = datetime.now(UTC)
        for decision in result.output.decisions:
            if decision.memory_id not in existing_ids:
                continue
            existing_mem = existing_by_id[decision.memory_id]
            if decision.action == "merge":
                new_stability = self._decay_model.update_on_merge(existing_mem)
                await self._store.update_memory_stability(decision.memory_id, new_stability)
                await self._store.deactivate_memory(decision.memory_id, superseded_by=candidate.id)
                if self._vector_store is not None:
                    await self._vector_store.delete(decision.memory_id)
                if self._event_logger is not None:
                    await self._event_logger.log(
                        decision.memory_id,
                        "merge",
                        {"superseded_by": candidate.id},
                    )
                self._schedule_learning(
                    self._update_alpha_tuner(candidate.user_id, decision.memory_id)
                )
            elif decision.action == "contradict":
                new_stability = self._decay_model.update_on_contradict(existing_mem)
                await self._store.update_memory_stability(decision.memory_id, new_stability)
                await self._store.deactivate_memory(
                    decision.memory_id,
                    superseded_by=candidate.id,
                    valid_until=now,
                )
                if self._vector_store is not None:
                    await self._vector_store.delete(decision.memory_id)
                await self._store.mark_signals_contradicted(decision.memory_id)
                if self._event_logger is not None:
                    await self._event_logger.log(
                        decision.memory_id,
                        "contradict",
                        {"superseded_by": candidate.id},
                    )
                self._schedule_learning(
                    self._update_alpha_tuner(candidate.user_id, decision.memory_id)
                )
            elif decision.action == "scope_override":
                # Both memories stay active. The candidate (named scope) takes
                # precedence over the existing (global) at compile time via the
                # most-specific-scope-wins rule in the policy prompt.
                if self._event_logger is not None:
                    await self._event_logger.log(
                        decision.memory_id,
                        "scope_override",
                        {"overridden_by": candidate.id, "override_scope": candidate.scope},
                    )
            elif decision.action == "distinct":
                if self._event_logger is not None:
                    await self._event_logger.log(decision.memory_id, "distinct")


def _derive_memory_frugal(*, user_response: str, signal_type: SignalType) -> _DerivedMemory:
    _TYPE_MAP: dict[SignalType, MemoryType] = {
        SignalType.CORRECTION: MemoryType.RULE,
        SignalType.DIRECTION: MemoryType.RULE,
        SignalType.PREFERENCE: MemoryType.RULE,
        SignalType.FACT: MemoryType.FACT,
        SignalType.REINFORCEMENT: MemoryType.CONTEXT,
    }
    memory_type = _TYPE_MAP[signal_type]
    content = " ".join(user_response.split())
    return _DerivedMemory(memory_type=memory_type, content=content, scope="global")


_TURSO_SCHEMES = ("libsql://", "ws://", "wss://", "https://", "http://", "turso://")


def _is_turso_url(url: str) -> bool:
    return any(url.startswith(s) for s in _TURSO_SCHEMES)


def _parse_turso_url(url: str) -> tuple[str, str | None]:
    """Parse a Turso store URL, extracting auth_token from query string if present.

    Returns (url_without_token, auth_token_or_None).
    Accepts turso:// as an alias for libsql://.
    """
    if url.startswith("turso://"):
        url = "libsql://" + url[len("turso://") :]
    if "?auth_token=" in url:
        base, token = url.split("?auth_token=", 1)
        return base, token
    return url, None


def _parse_store_url(url: str) -> str:
    """Parse a store URL into a SQLite path. Accepts:

    - `sqlite:///abs/path` -> /abs/path
    - `sqlite:///:memory:` -> :memory:
    - `:memory:` -> :memory:
    - bare absolute or relative path -> path (with ~ expansion)

    Rejects empty strings and non-sqlite URL schemes.
    """
    if not url:
        raise ValueError("store URL must be non-empty")
    if _is_turso_url(url):
        raise ValueError(
            f"Turso/libSQL URLs are handled automatically; pass the URL directly "
            f"as the store parameter: Imprint(store={url!r})"
        )
    if "://" in url and not url.startswith("sqlite://"):
        scheme = url.split("://", 1)[0]
        raise ValueError(f"unsupported store URL scheme: {scheme!r} (expected 'sqlite')")
    if url.startswith("sqlite:///"):
        path = url[len("sqlite:///") :]
        if path == ":memory:":
            return ":memory:"
        return os.path.expanduser(path)
    return os.path.expanduser(url)


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def _resolve_scope(requested: str | None) -> str:
    """Return the requested scope, or 'global' if absent or blank."""
    if not requested or not requested.strip():
        return "global"
    return requested.strip()


try:
    import numpy as _np  # type: ignore[import-untyped,import-not-found]

    def _cosine(a: list[float], b: list[float]) -> float:
        va = _np.array(a, dtype=_np.float32)  # type: ignore[reportUnknownMemberType]
        vb = _np.array(b, dtype=_np.float32)  # type: ignore[reportUnknownMemberType]
        denom = float(_np.linalg.norm(va) * _np.linalg.norm(vb))  # type: ignore[reportUnknownMemberType]
        return float(_np.dot(va, vb) / denom) if denom > 0.0 else 0.0  # type: ignore[reportUnknownMemberType]

except ImportError:

    def _cosine(a: list[float], b: list[float]) -> float:  # type: ignore[misc]
        dot = sum(x * y for x, y in zip(a, b, strict=False))
        mag_a = sum(x * x for x in a) ** 0.5
        mag_b = sum(x * x for x in b) ** 0.5
        if mag_a == 0.0 or mag_b == 0.0:
            return 0.0
        return dot / (mag_a * mag_b)


def _policy_cache_key(
    *,
    agent_id: str,
    user_id: str,
    memories: list[Memory],
    context: str | None,
    existing_instructions: str | None,
    max_output_tokens: int,
    scopes: list[str] | None,
) -> str:
    h = hashlib.sha256()
    h.update(b"agent\x00")
    h.update(agent_id.encode("utf-8"))
    h.update(b"\x00user\x00")
    h.update(user_id.encode("utf-8"))
    h.update(b"\x00mem\x00")
    for m in sorted(memories, key=lambda x: x.id):
        h.update(m.id.encode("utf-8"))
        h.update(b"|")
        h.update(m.updated_at.isoformat().encode("utf-8"))
        h.update(b"\x00")
    h.update(b"\x00ctx\x00")
    h.update((context or "").encode("utf-8"))
    h.update(b"\x00inst\x00")
    h.update((existing_instructions or "").encode("utf-8"))
    h.update(b"\x00max\x00")
    h.update(str(max_output_tokens).encode("utf-8"))
    h.update(b"\x00scopes\x00")
    if scopes is None:
        h.update(b"<none>")
    else:
        for s in sorted(scopes):
            h.update(s.encode("utf-8"))
            h.update(b"|")
    return h.hexdigest()
