"""Decay model implementations for Imprint.

FSRSStaticDecay is the default: fixed parameters, no learning. Stability
mutates on consolidation events (merge reinforces, contradict penalizes).
Effective stability is computed on read using elapsed time and recall count.
"""

import math
from datetime import datetime

from imprint.types import Memory


class FSRSStaticDecay:
    """FSRS-inspired decay with fixed parameters.

    Stability bounds: [0.1, 100.0].

    On merge: stability += 1.0 (implicit reinforcement).
    On contradict: stability *= 0.1 (prior contradiction lowers confidence).
    On recall: no stability change; recall_count is tracked by the store.

    Effective stability on read:
        s_eff = stability * (1 + 0.1 * recall_count) * exp(-days / (stability * 7))
    """

    _MIN_STABILITY: float = 0.1
    _MAX_STABILITY: float = 100.0

    def initial_stability(self, memory: Memory) -> float:
        return 5.0

    def update_on_merge(self, memory: Memory) -> float:
        return min(memory.stability + 1.0, self._MAX_STABILITY)

    def update_on_contradict(self, memory: Memory) -> float:
        return max(memory.stability * 0.1, self._MIN_STABILITY)

    def update_on_recall(self, memory: Memory) -> float:
        # Retrieval frequency is handled by the recall_count term in
        # effective_stability. Base stability only changes through meaningful
        # signals: merge, contradict, or outcome via finalize_loop.
        return memory.stability

    def effective_stability(self, memory: Memory, now: datetime) -> float:
        elapsed_days = (now - memory.created_at).total_seconds() / 86400.0
        decay = math.exp(-elapsed_days / max(memory.stability * 7.0, 0.001))
        recall_boost = 1.0 + 0.1 * memory.recall_count
        return memory.stability * recall_boost * decay
