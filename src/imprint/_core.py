"""Imprint facade: top-level SDK entry point."""

from __future__ import annotations

import asyncio
import contextvars
import os
import weakref
from collections.abc import AsyncGenerator, Coroutine
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from datetime import timedelta as _timedelta
from typing import Self, cast

from pydantic import AwareDatetime
from pydantic_ai import Agent
from pydantic_ai.models import Model

from imprint._feedback import _FeedbackMixin
from imprint._observe import _ObserveMixin
from imprint._policy import _PolicyMixin
from imprint._scope import _ScopeMixin
from imprint._utils import (
    _is_postgres_url,
    _parse_store_url,
)
from imprint.budget import HeuristicTokenCounter
from imprint.decay import FSRSStaticDecay
from imprint.prompts import attribute as attribute_prompt
from imprint.prompts import consolidate as consolidate_prompt
from imprint.prompts import memory as memory_prompt
from imprint.prompts import policy as policy_prompt
from imprint.prompts import scope as scope_prompt
from imprint.prompts import scope_consolidate as scope_consolidate_prompt
from imprint.prompts import signal as signal_prompt
from imprint.prompts import validate as validate_prompt
from imprint.prompts.attribute import _AttributionOutput
from imprint.prompts.consolidate import (
    _BatchConsolidationOutput,
    _ConsolidationOutput,
)
from imprint.prompts.memory import _DerivedMemory
from imprint.prompts.scope import _ScopeOutput
from imprint.prompts.scope_consolidate import _ScopeConsolidationOutput
from imprint.prompts.signal import _SignalDetection
from imprint.prompts.validate import _ValidationOutput
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
from imprint.retrieval import BanditAlphaTuner, StaticAlphaTuner
from imprint.stores.sqlite import NullEventLogger, SQLiteMemoryStore
from imprint.types import (
    Memory,
    MemoryDiff,
    MemoryEvent,
    MemoryHealth,
    MemoryLineage,
    ProcessingMode,
)

DEFAULT_MODEL = "anthropic:claude-haiku-4-5-20251001"

_VALID_PROCESSING_MODES: frozenset[str] = frozenset({"frugal", "balanced", "eager"})


# -- LLM compiler -------------------------------------------------------------


class LLMCompiler:
    """Default policy compiler: uses a pydantic-ai Agent to produce the policy text."""

    def __init__(self, model: str | Model) -> None:
        self._agent: Agent[None, str] = Agent(
            model,
            output_type=str,
            instructions=policy_prompt.SYSTEM,
            model_settings={"temperature": 0.0},
            defer_model_check=True,
        )

    @property
    def agent(self) -> Agent[None, str]:
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
            context=context,
            existing_instructions=existing_instructions,
            agent_description=agent_description,
        )
        result = await self._agent.run(
            user_prompt,
            model_settings={"temperature": 0.0, "max_tokens": max_tokens},
        )
        return result.output


# -- Feedback loop ------------------------------------------------------------


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
        imprint: Imprint,
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
        """Record interaction quality. outcome: -1.0 failure, 0.0 neutral, 1.0 success."""
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


# -- Policy -------------------------------------------------------------------


@dataclass(slots=True)
class Policy:
    text: str
    memories: list[Memory] = field(default_factory=list)  # type: ignore[assignment]
    dropped_memories: list[Memory] = field(default_factory=list)  # type: ignore[assignment]
    compiled_at: datetime = field(default_factory=lambda: datetime.now(UTC))


# -- Imprint ------------------------------------------------------------------


class Imprint(_ScopeMixin, _ObserveMixin, _PolicyMixin, _FeedbackMixin):
    """AI agent memory: compile-a-policy architecture.

    Observe agent-user exchanges to build a memory store, then compile those
    memories into a behavioral policy injected into the agent's system prompt.

    Usage:

      async with Imprint(agent_id="my-agent") as imp:
          await imp.observe(user_id="u1", agent_output="...", user_response="...")
          policy = await imp.get_policy(user_id="u1", context="...")
          # inject policy.text into your agent's system prompt
    """

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
        dynamic_scopes: bool = False,
        scope_consolidation_threshold: int = 5,
    ) -> None:
        self.agent_id = agent_id
        self._dynamic_scopes = dynamic_scopes
        self._scope_consolidation_threshold = scope_consolidation_threshold

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
            if _is_postgres_url(store):
                from imprint.stores.postgres import PostgresMemoryStore

                store_inst = PostgresMemoryStore(store)
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

        if compiler is not None:
            self._compiler: Compiler = compiler
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
            instructions=(
                memory_prompt.SYSTEM_DYNAMIC_SCOPES if dynamic_scopes else memory_prompt.SYSTEM
            ),
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
        self._scope_consolidate_agent: Agent[None, _ScopeConsolidationOutput] = Agent(
            model,
            output_type=_ScopeConsolidationOutput,
            instructions=scope_consolidate_prompt.SYSTEM,
            model_settings={"temperature": 0.0},
            defer_model_check=True,
        )

    # -- Lifecycle ------------------------------------------------------------

    async def connect(self) -> None:
        await self._store.connect()
        await self._store.init_schema()
        await self._sync_agent_config()
        if self._event_logger is None:
            if hasattr(self._store, "make_event_logger"):
                self._event_logger = self._store.make_event_logger()  # type: ignore[union-attr]
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

        if self._ctor_scopes is not None:
            await self._store.clear_scopes(self.agent_id)
            for scope in self.scopes:
                await self._store.insert_scope(self.agent_id, scope)
        else:
            for scope in self.scopes:
                await self._store.insert_scope(self.agent_id, scope)
        registered = await self._store.list_scopes(self.agent_id)
        seen_reg: set[str] = set(self.scopes)
        for s in registered:
            if s not in seen_reg and s != "global":
                self.scopes.append(s)
                seen_reg.add(s)

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
    def from_env(cls) -> Imprint:
        """Construct an Imprint instance from environment variables.

        Required:
          IMPRINT_AGENT_ID       -- agent identifier

        Optional:
          IMPRINT_STORE          -- store URL (default: sqlite:///~/.imprint/imprint.db)
          IMPRINT_MODEL          -- pydantic-ai model string (default: claude-haiku-4-5)
          IMPRINT_MODE           -- frugal | balanced | eager (default: balanced)
          IMPRINT_DYNAMIC_SCOPES -- true | 1 | yes to enable dynamic scope creation
        """
        agent_id = os.environ["IMPRINT_AGENT_ID"]
        store_url = os.environ.get("IMPRINT_STORE", "sqlite:///~/.imprint/imprint.db")
        model = os.environ.get("IMPRINT_MODEL", DEFAULT_MODEL)
        mode_str = os.environ.get("IMPRINT_MODE")
        mode: ProcessingMode | None = (
            cast(ProcessingMode, mode_str) if mode_str in _VALID_PROCESSING_MODES else None
        )
        dynamic = os.environ.get("IMPRINT_DYNAMIC_SCOPES", "").lower() in ("1", "true", "yes")
        return cls(
            agent_id=agent_id,
            store=store_url,
            model=model,
            processing_mode=mode,
            dynamic_scopes=dynamic,
        )

    # -- Public memory API ----------------------------------------------------

    async def list_memories(
        self,
        user_id: str,
        *,
        scopes: list[str] | None = None,
    ) -> list[Memory]:
        """Return the active memory list for a user, optionally filtered by scopes."""
        return await self._store.list_memories(self.agent_id, user_id, scopes=scopes)

    async def pin_memory(self, memory_id: str) -> None:
        """Pin a memory so it is never dropped by token budget truncation."""
        await self._store.set_pinned(memory_id, True)

    async def deactivate_memory(self, user_id: str, memory_id: str) -> bool:
        """Deactivate a specific memory. Returns True if found and deactivated."""
        found = await self._store.deactivate_memory(memory_id)
        if found:
            await self._store.invalidate_cached_policies(self.agent_id, user_id)
        return found

    async def forget(self, user_id: str) -> None:
        """Hard delete all memories, signals, and events for this user.

        Also removes embeddings from the vector store if one is configured.
        Does not touch the scope vocabulary. This is irreversible.
        """
        if self._vector_store is not None:
            all_memories = await self._store.list_memories(
                self.agent_id, user_id, active_only=False
            )
            if all_memories:
                await self._vector_store.delete_batch([m.id for m in all_memories])
        await self._store.delete_user_data(self.agent_id, user_id)
        await self._store.invalidate_cached_policies(self.agent_id, user_id)

    async def consolidate(self, user_id: str, *, prune_threshold: float = 0.5) -> int:
        """Force a consolidation pass: prune decayed memories then run scope consolidation.

        Returns the count of memories pruned.
        """
        now = datetime.now(UTC)
        all_memories = await self._store.list_memories(self.agent_id, user_id)
        pruned = 0

        for m in all_memories:
            if m.pinned:
                continue
            eff = self._decay_model.effective_stability(m, now)
            if eff < prune_threshold:
                deactivated = await self._store.deactivate_memory(m.id)
                if deactivated:
                    pruned += 1
                    if self._vector_store is not None:
                        await self._vector_store.delete(m.id)
                    if self._event_logger is not None:
                        await self._event_logger.log(
                            m.id,
                            "pruned",
                            {"effective_stability": round(eff, 4), "threshold": prune_threshold},
                        )

        if pruned > 0:
            await self._store.invalidate_cached_policies(self.agent_id, user_id)

        if self.processing_mode != "frugal":
            await self.consolidate_scopes(user_id)

        return pruned

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

    # -- Observability --------------------------------------------------------

    async def list_events(
        self,
        user_id: str,
        *,
        memory_id: str | None = None,
        limit: int = 50,
    ) -> list[MemoryEvent]:
        """Return logged events for a user's memories, newest first."""
        return await self._store.list_events(
            self.agent_id, user_id, memory_id=memory_id, limit=limit
        )

    async def memory_lineage(self, memory_id: str) -> MemoryLineage:
        """Return the full creation and mutation history of one memory."""
        target = await self._store.get_memory(memory_id)
        if target is None:
            raise KeyError(f"memory {memory_id!r} not found")
        successor, _ = await self._store.get_memory_with_supersession(memory_id)
        signal = await self._store.get_creating_signal(memory_id)
        superseded_memories = await self._store.get_superseded_memories(memory_id)
        events = await self.list_events(target.user_id or "", memory_id=memory_id, limit=200)
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
                total=0, active=0, by_scope={}, by_type={}, pinned=0, avg_recall_count=0.0
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
        return MemoryHealth(
            total=len(all_memories),
            active=len(active_memories),
            by_scope=by_scope,
            by_type=by_type,
            pinned=pinned_count,
            avg_recall_count=round(avg_recall, 3),
            oldest_active=min(valid_froms) if valid_froms else None,
            newest_active=max(valid_froms) if valid_froms else None,
        )

    # -- Observability --------------------------------------------------------

    async def diff_memories(
        self, user_id: str, since: AwareDatetime, until: AwareDatetime
    ) -> MemoryDiff:
        """Return the delta between memory state at since and until.

        Delegates to the underlying store so SQLite and Postgres implementations
        can use the most efficient query for each backend.
        """
        return await self._store.diff_memories(self.agent_id, user_id, since, until)

    @property
    def alpha_estimate(self) -> float:
        """Current alpha estimate from the retrieval tuner.

        Returns the deterministic expected alpha -- the mean of the best arm's
        Beta distribution for BanditAlphaTuner, or the fixed alpha for
        StaticAlphaTuner. This is suitable for dashboards and gauges; unlike
        get_alpha(), it does not sample from the distribution.

        alpha is the sparse weight in hybrid retrieval (dense = 1 - alpha).
        Higher values favor FTS; lower values favor vector similarity.
        """
        from imprint.retrieval import _ARMS, BanditAlphaTuner

        if not isinstance(self._alpha_tuner, BanditAlphaTuner):
            return self._alpha_tuner.get_alpha()
        state = self._alpha_tuner.get_state()
        succs = state["s"]
        fails = state["f"]
        # Mean of Beta(s+1, f+1) = (s+1)/(s+f+2). Find the arm with highest mean.
        means = [(s + 1) / (s + f + 2) for s, f in zip(succs, fails, strict=True)]
        return _ARMS[means.index(max(means))]

    # -- Loop management ------------------------------------------------------

    async def open_loop(
        self,
        user_id: str,
        *,
        session_id: str | None = None,
        timeout: int = 3600,
    ) -> MemoryLoop:
        """Create and register an explicit feedback loop."""
        ml = MemoryLoop(user_id=user_id, session_id=session_id, timeout=timeout, imprint=self)
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
        """Async context manager that opens a MemoryLoop and closes it on exit."""
        ml = await self.open_loop(user_id=user_id, session_id=session_id, timeout=timeout)
        try:
            yield ml
        finally:
            if not ml.closed:
                if ml.outcome is None:
                    ml.outcome = 0.0
                await ml.close()

    async def drain(self) -> None:
        """Await all pending background learning tasks."""
        if self._pending_tasks:
            await asyncio.gather(*self._pending_tasks, return_exceptions=True)

    # -- Infrastructure -------------------------------------------------------

    def _schedule_learning(self, coro: Coroutine[object, object, None]) -> None:
        task: asyncio.Task[None] = asyncio.create_task(coro, context=contextvars.copy_context())
        self._pending_tasks.add(task)
        task.add_done_callback(self._pending_tasks.discard)
