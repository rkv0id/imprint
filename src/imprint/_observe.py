"""Observe path mixin for Imprint.

_ObserveMixin provides:
  - observe / observe_directions (public)
  - _detect_signal / _detect_signal_llm
  - _derive_memory / _derive_and_store_direction / _validate_directions
  - _consolidate_against_existing / _consolidate_directions_batch
  - _consolidate_frugal_vector / _prefilter_candidates

Also exports the module-level helper _derive_memory_frugal.

_update_alpha_tuner lives in _feedback.py because it reads self._active_loops
(a loop-management attribute) and dispatches to the bandit's get_state().
"""

from __future__ import annotations

from collections.abc import Coroutine
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from imprint._detect import detect_signal_heuristic
from imprint._utils import _new_id, _resolve_scope
from imprint.prompts import consolidate as consolidate_prompt
from imprint.prompts import memory as memory_prompt
from imprint.prompts import signal as signal_prompt
from imprint.prompts import validate as validate_prompt
from imprint.prompts.consolidate import (
    _BatchConsolidationOutput,
    _ConsolidationOutput,
)
from imprint.prompts.memory import _DerivedMemory
from imprint.prompts.signal import _SignalDetection
from imprint.prompts.validate import _ValidationOutput
from imprint.types import Memory, MemorySource, MemoryType, Signal, SignalType

if TYPE_CHECKING:
    from pydantic_ai import Agent

    from imprint.protocols import (
        DecayModel,
        Embedder,
        EventLogger,
        MemoryStore,
        VectorStore,
    )
    from imprint.types import ProcessingMode


# -- Module-level helper ------------------------------------------------------


def _derive_memory_frugal(*, user_response: str, signal_type: SignalType) -> _DerivedMemory:
    """Deterministic (no-LLM) memory derivation used in frugal mode."""
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


# -- Mixin --------------------------------------------------------------------


class _ObserveMixin:
    """Observe path: signal detection, memory derivation, and consolidation.

    Attributes and cross-mixin method stubs declared under TYPE_CHECKING are
    provided by Imprint.__init__ or other mixins at runtime.
    """

    if TYPE_CHECKING:
        # Attributes provided by Imprint.__init__
        _store: MemoryStore
        _embedder: Embedder | None
        _vector_store: VectorStore | None
        _decay_model: DecayModel
        _event_logger: EventLogger | None
        agent_id: str
        processing_mode: ProcessingMode
        _dynamic_scopes: bool

        _detect_agent: Agent[None, _SignalDetection]
        _derive_agent: Agent[None, _DerivedMemory]
        _consolidate_agent: Agent[None, _ConsolidationOutput]
        _batch_consolidate_agent: Agent[None, _BatchConsolidationOutput]
        _validate_agent: Agent[None, _ValidationOutput]

        # Cross-mixin method stubs (provided by _ScopeMixin, _FeedbackMixin, or Imprint)
        async def _accept_scope(self, proposed: str, *, user_id: str | None = None) -> str: ...
        async def _register_scope(self, scope: str, *, user_id: str | None = None) -> None: ...
        async def _combined_scopes(self, user_id: str | None = None) -> list[str]: ...
        async def _maybe_trigger_scope_consolidation(self, user_id: str) -> None: ...
        async def _update_alpha_tuner(self, user_id: str | None, memory_id: str) -> None: ...
        def _schedule_learning(self, coro: Coroutine[object, object, None]) -> None: ...
        def _sweep_expired_loops(self) -> None: ...

    # -- Public methods -------------------------------------------------------

    async def observe(
        self,
        *,
        user_id: str,
        agent_output: str,
        user_response: str,
        context: str | None = None,
        scope: str | None = None,
    ) -> None:
        """Record one agent-user exchange and derive a memory from it if a signal is detected."""
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
            user_id=user_id,
        )

        existing = await self._store.list_memories(self.agent_id, user_id)

        chosen_scope = scope if scope is not None else derived.scope
        resolved_scope = _resolve_scope(chosen_scope)
        if resolved_scope != "global":
            await self._register_scope(resolved_scope, user_id=user_id)

        now = datetime.now(UTC)
        memory = Memory(
            id=_new_id("mem"),
            agent_id=self.agent_id,
            user_id=user_id,
            type=derived.memory_type,
            scope=resolved_scope,
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

        await self._store.insert_signal(signal)
        await self._store.insert_memory(memory)
        await self._store.link_signal_to_memory(memory_id=memory.id, signal_id=signal.id)
        await self._store.invalidate_cached_policies(self.agent_id, user_id)

        if self._embedder is not None and self._vector_store is not None:
            embedding = await self._embedder.embed(memory.content)
            await self._vector_store.upsert(memory.id, embedding)

        await self._consolidate_against_existing(
            candidate=memory,
            candidate_signal_type=signal_type,
            existing=existing,
        )

        if self._dynamic_scopes and self.processing_mode != "frugal":
            await self._maybe_trigger_scope_consolidation(user_id)

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

        existing = await self._store.list_memories(self.agent_id, user_id)
        available_scopes = await self._combined_scopes(user_id)

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

        await self._consolidate_directions_batch(
            candidates=memories,
            existing=existing,
        )

        if self._dynamic_scopes and self.processing_mode != "frugal":
            await self._maybe_trigger_scope_consolidation(user_id)

        return memories

    # -- Internal: signal detection -------------------------------------------

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

    # -- Internal: memory derivation ------------------------------------------

    async def _derive_memory(
        self,
        *,
        agent_output: str,
        user_response: str,
        signal_type: SignalType,
        user_id: str | None = None,
    ) -> _DerivedMemory:
        if self.processing_mode == "frugal":
            return _derive_memory_frugal(user_response=user_response, signal_type=signal_type)

        prompt = memory_prompt.build_user_prompt(
            agent_output=agent_output,
            user_response=user_response,
            signal_type=signal_type.value,
            available_scopes=await self._combined_scopes(user_id),
            dynamic_scopes=self._dynamic_scopes,
        )
        result = await self._derive_agent.run(prompt)
        derived = result.output
        if self._dynamic_scopes:
            accepted = await self._accept_scope(derived.scope, user_id=user_id)
            if accepted != derived.scope:
                derived = derived.model_copy(update={"scope": accepted})
        return derived

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
        """Derive and store one direction memory. Does not run consolidation."""
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
                dynamic_scopes=self._dynamic_scopes,
            )
            result = await self._derive_agent.run(prompt)
            derived = result.output
            if self._dynamic_scopes:
                accepted = await self._accept_scope(derived.scope, user_id=user_id)
                if accepted != derived.scope:
                    derived = derived.model_copy(update={"scope": accepted})

        chosen_scope = scope if scope is not None else derived.scope
        resolved_scope = _resolve_scope(chosen_scope)
        if resolved_scope != "global":
            await self._register_scope(resolved_scope, user_id=user_id)

        now = datetime.now(UTC)
        memory = Memory(
            id=_new_id("mem"),
            agent_id=self.agent_id,
            user_id=user_id,
            type=derived.memory_type,
            scope=resolved_scope,
            content=derived.content,
            source=source,
            valid_from=now,
            created_at=now,
            updated_at=now,
        )

        await self._store.insert_memory(memory)
        await self._store.invalidate_cached_policies(self.agent_id, user_id)

        if self._embedder is not None and self._vector_store is not None:
            embedding = await self._embedder.embed(memory.content)
            await self._vector_store.upsert(memory.id, embedding)

        return memory

    async def _validate_directions(self, directions: list[str]) -> list[str]:
        prompt = validate_prompt.build_user_prompt(directions=directions)
        result = await self._validate_agent.run(prompt)
        verdicts = result.output.verdicts
        passed: list[str] = []
        for i, direction in enumerate(directions):
            if i < len(verdicts) and verdicts[i].verdict == "directive":
                passed.append(direction)
        return passed

    # -- Internal: consolidation ----------------------------------------------

    async def _consolidate_directions_batch(
        self,
        *,
        candidates: list[Memory],
        existing: list[Memory],
    ) -> None:
        """Consolidate a batch of new direction memories against existing ones.

        frugal+vector: vector merge per candidate, no LLM.
        frugal (no vector): no-op.
        balanced/eager: one LLM call for the whole batch, optionally
          prefiltered by vector similarity.
        """
        if not existing:
            return

        if self.processing_mode == "frugal":
            if self._embedder is not None and self._vector_store is not None:
                for candidate in candidates:
                    await self._consolidate_frugal_vector(candidate=candidate, existing=existing)
            return

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
                        decision.memory_id, "merge", {"superseded_by": candidate.id}
                    )
                deactivated.add(decision.memory_id)
                self._schedule_learning(
                    self._update_alpha_tuner(candidate.user_id, decision.memory_id)
                )
            elif decision.action == "contradict":
                new_stability = self._decay_model.update_on_contradict(existing_mem)
                await self._store.update_memory_stability(decision.memory_id, new_stability)
                await self._store.deactivate_memory(
                    decision.memory_id, superseded_by=candidate.id, valid_until=now
                )
                if self._vector_store is not None:
                    await self._vector_store.delete(decision.memory_id)
                await self._store.mark_signals_contradicted(decision.memory_id)
                if self._event_logger is not None:
                    await self._event_logger.log(
                        decision.memory_id, "contradict", {"superseded_by": candidate.id}
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
                        decision.memory_id, "merge", {"superseded_by": candidate.id}
                    )
                self._schedule_learning(
                    self._update_alpha_tuner(candidate.user_id, decision.memory_id)
                )
            elif decision.action == "contradict":
                new_stability = self._decay_model.update_on_contradict(existing_mem)
                await self._store.update_memory_stability(decision.memory_id, new_stability)
                await self._store.deactivate_memory(
                    decision.memory_id, superseded_by=candidate.id, valid_until=now
                )
                if self._vector_store is not None:
                    await self._vector_store.delete(decision.memory_id)
                await self._store.mark_signals_contradicted(decision.memory_id)
                if self._event_logger is not None:
                    await self._event_logger.log(
                        decision.memory_id, "contradict", {"superseded_by": candidate.id}
                    )
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
            elif decision.action == "distinct":
                if self._event_logger is not None:
                    await self._event_logger.log(decision.memory_id, "distinct")

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
