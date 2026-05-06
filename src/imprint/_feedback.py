"""Feedback and learning mixin for Imprint.

_FeedbackMixin provides:
  - finalize_loop (public)
  - _sweep_expired_loops
  - _apply_feedback
  - _embedding_attribution / _llm_attribution
  - _update_alpha_tuner
"""

from __future__ import annotations

import weakref
from collections.abc import Coroutine
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from imprint.prompts import attribute as attribute_prompt
from imprint.prompts.attribute import _AttributionOutput

if TYPE_CHECKING:
    from pydantic_ai import Agent

    from imprint._core import MemoryLoop
    from imprint.protocols import (
        AlphaTuner,
        DecayModel,
        Embedder,
        EventLogger,
        MemoryStore,
        VectorStore,
    )
    from imprint.types import ProcessingMode


class _FeedbackMixin:
    """Loop lifecycle and learning signal application.

    Attributes and cross-mixin method stubs declared under TYPE_CHECKING are
    provided by Imprint.__init__ at runtime.
    """

    if TYPE_CHECKING:
        # Attributes provided by Imprint.__init__
        _store: MemoryStore
        _embedder: Embedder | None
        _vector_store: VectorStore | None
        _decay_model: DecayModel
        _alpha_tuner: AlphaTuner
        _event_logger: EventLogger | None
        agent_id: str
        processing_mode: ProcessingMode
        _active_loops: weakref.WeakSet[MemoryLoop]
        _attribute_agent: Agent[None, _AttributionOutput]

        # Cross-mixin method stub (provided by Imprint base)
        def _schedule_learning(self, coro: Coroutine[object, object, None]) -> None: ...

    # -- Public method --------------------------------------------------------

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

    # -- Internal: loop sweep -------------------------------------------------

    def _sweep_expired_loops(self) -> None:
        """Finalize loops that have exceeded their timeout.

        Called lazily on every get_policy() and observe() call. Expired loops
        receive outcome=-0.15 (small penalty for abandoned retrieval).
        """
        now = datetime.now(UTC)
        for ml in list(self._active_loops):
            if ml.closed or ml.expires_at >= now:
                continue
            ml.closed = True
            ml.outcome = -0.15
            self._schedule_learning(self.finalize_loop(ml))

    # -- Internal: feedback application ---------------------------------------

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

        if loop.retrieved_memories and outcome != 0.0:
            for m in loop.retrieved_memories:
                current = await self._store.get_memory(m.id)
                if current is None or not current.active:
                    continue
                if outcome > 0.0:
                    new_s = min(current.stability * (1.0 + 0.1 * outcome), 100.0)
                else:
                    new_s = max(current.stability * (1.0 + 0.05 * outcome), 0.1)
                if abs(new_s - current.stability) > 0.001:
                    await self._store.update_memory_stability(m.id, new_s)

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
        if updated and hasattr(self._alpha_tuner, "get_state"):
            import json as _json

            await self._store.put_alpha_tuner_state(
                self.agent_id,
                _json.dumps(self._alpha_tuner.get_state()),  # type: ignore[union-attr]
            )
