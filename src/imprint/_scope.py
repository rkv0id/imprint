"""Scope management mixin for Imprint.

_ScopeMixin provides all scope-related methods:
  - consolidate_scopes / _apply_scope_consolidation (public + internal)
  - _maybe_trigger_scope_consolidation
  - _is_valid_scope / _find_canonical_scope / _accept_scope / _register_scope
  - _combined_scopes / _infer_scopes / _infer_scopes_llm

Intended only as a mixin component of Imprint. Do not instantiate directly.
"""

from __future__ import annotations

from collections.abc import Coroutine
from typing import TYPE_CHECKING, Any

from imprint._utils import _MAX_SCOPE_LEN, _cosine, _levenshtein
from imprint.prompts import scope as scope_prompt
from imprint.prompts import scope_consolidate as scope_consolidate_prompt
from imprint.prompts.scope import _ScopeOutput
from imprint.prompts.scope_consolidate import _ScopeConsolidationOutput

if TYPE_CHECKING:
    from pydantic_ai import Agent

    from imprint.protocols import Embedder, EventLogger, MemoryStore
    from imprint.types import ProcessingMode


class _ScopeMixin:
    """Scope vocabulary management.

    Attributes and cross-mixin methods declared under TYPE_CHECKING are
    provided by Imprint.__init__ or other mixins at runtime.
    """

    if TYPE_CHECKING:
        # Attributes provided by Imprint.__init__
        _store: MemoryStore
        _embedder: Embedder | None
        _event_logger: EventLogger | None
        agent_id: str
        scopes: list[str]
        processing_mode: ProcessingMode
        _dynamic_scopes: bool
        _scope_consolidation_threshold: int
        _scope_consolidate_agent: Agent[None, _ScopeConsolidationOutput]
        _scope_agent: Agent[None, _ScopeOutput]

        # Cross-mixin method stubs (provided by Imprint or _FeedbackMixin)
        def _schedule_learning(self, coro: Coroutine[object, object, None]) -> None: ...

    # -- Public method --------------------------------------------------------

    async def consolidate_scopes(self, user_id: str) -> None:
        """Consolidate the scope vocabulary: merge, rename, or split as needed.

        The LLM is shown all known scopes with their memory counts and a few
        sample memories from each. It decides which scopes to keep, rename,
        merge into each other, or split by reassigning individual memories.

        Call this manually at any time, or set dynamic_scopes=True and
        scope_consolidation_threshold to trigger it automatically in the
        background after observe() calls.

        No-op in frugal mode (frugal avoids LLM calls).
        No-op when fewer than two scopes exist.
        """
        if self.processing_mode == "frugal":
            return

        scopes = await self._store.list_scopes(self.agent_id)
        if len(scopes) < 2:
            return

        scope_summaries: list[dict[str, Any]] = []
        for scope_name in scopes:
            memories = await self._store.list_memories(self.agent_id, user_id, scopes=[scope_name])
            if not memories:
                continue
            scope_summaries.append(
                {
                    "name": scope_name,
                    "count": len(memories),
                    "memory_ids": [m.id for m in memories],
                    "samples": [m.content for m in memories[:3]],
                }
            )

        if len(scope_summaries) < 2:
            return

        prompt = scope_consolidate_prompt.build_user_prompt(scope_summaries)
        result = await self._scope_consolidate_agent.run(prompt)
        await self._apply_scope_consolidation(result.output, user_id)

    # -- Internal methods -----------------------------------------------------

    async def _apply_scope_consolidation(
        self, output: _ScopeConsolidationOutput, user_id: str
    ) -> None:
        """Apply the LLM's consolidation decisions to the store and self.scopes."""
        known = set(self.scopes)

        for action in output.actions:
            if action.kind == "keep":
                continue

            elif action.kind == "rename":
                if action.target is None or action.scope not in known:
                    continue
                new = action.target.strip().lower()
                if not new or new == "global" or len(new) > _MAX_SCOPE_LEN:
                    continue
                await self._store.rename_scope(self.agent_id, action.scope, new)
                await self._store.insert_scope(self.agent_id, new)
                if action.scope in self.scopes:
                    idx = self.scopes.index(action.scope)
                    self.scopes[idx] = new
                known.discard(action.scope)
                known.add(new)

            elif action.kind == "merge":
                if action.target is None:
                    continue
                if action.scope not in known or action.target not in known:
                    continue
                await self._store.merge_scopes(self.agent_id, action.scope, action.target)
                if action.scope in self.scopes:
                    self.scopes.remove(action.scope)
                known.discard(action.scope)

            elif action.kind == "split":
                if action.scope not in known:
                    continue
                for reassignment in action.reassignments:
                    new_scope = reassignment.new_scope.strip().lower()
                    if not new_scope or new_scope == "global":
                        continue
                    if len(new_scope) > _MAX_SCOPE_LEN:
                        continue
                    await self._store.update_memory_scope(reassignment.memory_id, new_scope)
                    if new_scope not in known:
                        await self._store.insert_scope(self.agent_id, new_scope)
                        self.scopes.append(new_scope)
                        known.add(new_scope)
                remaining = await self._store.list_memories(
                    self.agent_id, user_id, scopes=[action.scope]
                )
                if not remaining:
                    if action.scope in self.scopes:
                        self.scopes.remove(action.scope)
                    known.discard(action.scope)

        await self._store.invalidate_cached_policies(self.agent_id, user_id)

    async def _maybe_trigger_scope_consolidation(self, user_id: str) -> None:
        """Schedule scope consolidation if memory count crosses the threshold."""
        all_memories = await self._store.list_memories(self.agent_id, user_id)
        n = len(all_memories)
        if n > 0 and n % self._scope_consolidation_threshold == 0:
            self._schedule_learning(self.consolidate_scopes(user_id))

    async def _register_scope(self, scope: str, *, user_id: str | None = None) -> None:
        """Append scope to self.scopes and persist to the scopes table.

        If this is a genuinely new scope and dynamic_scopes is enabled,
        schedules a background consolidation pass immediately so semantically
        duplicate scopes get merged before more memories land in them.
        """
        if scope in self.scopes or scope == "global":
            return
        self.scopes.append(scope)
        await self._store.insert_scope(self.agent_id, scope)
        if (
            user_id is not None
            and self._dynamic_scopes
            and self.processing_mode != "frugal"
            and len(self.scopes) >= 2
        ):
            self._schedule_learning(self.consolidate_scopes(user_id))

    async def _accept_scope(self, proposed: str, *, user_id: str | None = None) -> str:
        """Validate and register a scope proposed by the derivation LLM.

        Returns the canonical scope name to use:
        - If proposed == "global" or is in the known list, return as-is.
        - If it is a near-duplicate of an existing scope, return the existing one.
        - If it passes format validation, register it and return it.
        - If it fails format validation, fall back to "global".
        """
        normalized = proposed.strip().lower()
        if normalized == "global" or not normalized:
            return "global"
        if normalized in self.scopes:
            return normalized
        canonical = self._find_canonical_scope(normalized)
        if canonical is not None:
            return canonical
        if not self._is_valid_scope(normalized):
            return "global"
        await self._register_scope(normalized, user_id=user_id)
        return normalized

    def _find_canonical_scope(self, proposed: str) -> str | None:
        """Return an existing scope if proposed is a near-duplicate, else None."""
        normalized = proposed.strip().lower()
        known = self.scopes
        if normalized in known:
            return normalized
        for existing in known:
            if _levenshtein(normalized, existing) <= 2:
                return existing
        return None

    @staticmethod
    def _is_valid_scope(name: str) -> bool:
        """Return True if name is usable as a dynamic scope."""
        return bool(name) and name != "global" and len(name) <= _MAX_SCOPE_LEN

    async def _combined_scopes(self, user_id: str | None = None) -> list[str]:
        """Return scopes for use as derivation/inference candidates.

        When user_id is provided (scope routing during observe/get_policy):
        start from scopes this user already has active memories in. This
        keeps other users' dynamic scope names invisible during derivation,
        avoiding cross-user scope name leakage.

        Constructor-declared scopes are always appended as base vocabulary.

        When user_id is None (vocabulary management in consolidate_scopes):
        use the full agent scope table.
        """
        if user_id is not None:
            live = await self._store.list_active_scopes_for_user(self.agent_id, user_id)
        else:
            live = await self._store.list_scopes(self.agent_id)
        seen = set(live)
        for s in self.scopes:
            if s not in seen:
                live.append(s)
                seen.add(s)
        return live

    async def _infer_scopes(self, context: str, user_id: str) -> list[str] | None:
        """Infer relevant scopes from context.

        Returns a list of inferred scopes or None (caller falls back to
        fetch-all). Candidates are user-specific active scopes plus
        constructor-declared base vocabulary.
        """
        candidate_scopes = await self._combined_scopes(user_id)
        if not candidate_scopes:
            return None

        if self.processing_mode == "eager":
            return await self._infer_scopes_llm(context, candidate_scopes)

        if self._embedder is None:
            if self.processing_mode == "balanced":
                return await self._infer_scopes_llm(context, candidate_scopes)
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
