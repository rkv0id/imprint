"""Retrieval mixin for Imprint.

_RetrievalMixin provides:
  - get_policy (public)
  - _hybrid_retrieve
  - _apply_recall
"""

from __future__ import annotations

from collections.abc import Coroutine
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Literal

from imprint._utils import _policy_cache_key
from imprint.budget import truncate_to_budget
from imprint.retrieval import rrf_fuse, sanitize_fts_query
from imprint.types import Memory

if TYPE_CHECKING:
    from imprint._core import MemoryLoop, Policy
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
    from imprint.types import ProcessingMode


class _RetrievalMixin:
    """Policy retrieval: get_policy, hybrid ranking, recall updates.

    Attributes and cross-mixin method stubs declared under TYPE_CHECKING are
    provided by Imprint.__init__ or other mixins at runtime.
    """

    if TYPE_CHECKING:
        # Attributes provided by Imprint.__init__
        _store: MemoryStore
        _embedder: Embedder | None
        _vector_store: VectorStore | None
        _decay_model: DecayModel
        _alpha_tuner: AlphaTuner
        _event_logger: EventLogger | None
        _token_counter: TokenCounter
        _compiler: Compiler
        agent_id: str
        agent_description: str | None
        processing_mode: ProcessingMode

        # Cross-mixin method stubs
        async def _infer_scopes(self, context: str, user_id: str) -> list[str] | None: ...
        def _sweep_expired_loops(self) -> None: ...
        def _schedule_learning(self, coro: Coroutine[object, object, None]) -> None: ...

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
        """Compile and return a behavioral policy for this user.

        Retrieves active memories, optionally filtered by inferred or explicit
        scopes. Applies hybrid ranking when an embedder and context are
        available. Compiles memories into an instruction string via the
        Compiler. Caches the result keyed on memory content + context.
        """
        from imprint._core import Policy

        self._sweep_expired_loops()

        effective_scopes: list[str] | None = scopes
        if scopes is None and context is not None:
            inferred = await self._infer_scopes(context, user_id)
            effective_scopes = inferred

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
        if not memories:
            return
        await self._store.increment_recall_count_batch([m.id for m in memories])
        for m in memories:
            new_stability = self._decay_model.update_on_recall(m)
            if new_stability != m.stability:
                await self._store.update_memory_stability(m.id, new_stability)
            if self._event_logger is not None:
                await self._event_logger.log(m.id, "recall")
