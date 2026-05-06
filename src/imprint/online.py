"""Online decay model using River for incremental gradient updates.

Requires: pip install imprint-mem[online]
"""

from __future__ import annotations

import base64
import math
import pickle
from datetime import datetime
from typing import Any

from imprint.types import Memory, MemoryType

_TYPE_ENCODING: dict[MemoryType, float] = {
    MemoryType.CONTEXT: 0.0,
    MemoryType.FACT: 1.0,
    MemoryType.RULE: 2.0,
    MemoryType.DECISION: 3.0,
}

# Outcome quality multiplier bounds.
# outcome=1.0 -> multiplier up to _MULT_MAX (can exceed static)
# outcome=0.0 -> multiplier floors at _MULT_MIN (memory stays retrievable)
_MULT_MIN = 0.2
_MULT_MAX = 1.5


def _features(memory: Memory, now: datetime) -> dict[str, float]:
    elapsed_days = (now - memory.created_at).total_seconds() / 86400.0
    return {
        "stability": memory.stability,
        "recall_count": float(memory.recall_count),
        "elapsed_days": elapsed_days,
        "type": _TYPE_ENCODING.get(memory.type, 1.0),
    }


def _fsrs_base(memory: Memory, now: datetime) -> float:
    """Compute the same effective stability the static FSRS model would produce."""
    elapsed_days = (now - memory.created_at).total_seconds() / 86400.0
    recall_boost = 1.0 + 0.1 * memory.recall_count
    decay = math.exp(-elapsed_days / max(memory.stability * 7.0, 0.001))
    return memory.stability * recall_boost * decay


class FSRSGradientDecay:
    """FSRS-inspired decay with a learned quality multiplier.

    Uses a River Pipeline (StandardScaler | LinearRegression) trained on
    per-loop outcome signals. The model learns a quality multiplier in
    [_MULT_MIN, _MULT_MAX] that scales the standard FSRS effective stability.

    At outcome=1.0 the multiplier approaches _MULT_MAX (memory is fully
    useful, stability is boosted above the static formula). At outcome=0.0
    the multiplier floors at _MULT_MIN (memory contributes less but stays
    retrievable). As feedback accumulates the multiplier drifts to reflect
    actual agent-specific retrieval quality.

    Before sufficient training the model predicts ~0.0, so the multiplier
    is clamped at _MULT_MIN. A warmup pass (outcome=1.0 x N_WARMUP) at
    construction bootstraps the model to start near the static formula.

    Requires: pip install imprint-mem[online]
    """

    _N_WARMUP = 5  # synthetic positive samples to seed reasonable initial predictions

    def __init__(self, learning_rate: float = 0.01) -> None:
        self._lr = learning_rate
        self._model: Any = self._build_model()
        self._warmed_up = False

    def _build_model(self) -> Any:
        try:
            from river import linear_model, optim, preprocessing  # type: ignore[import-untyped]
        except ImportError as e:
            missing = getattr(e, "name", None)
            if missing in ("river", None):
                raise ImportError(
                    "river is required for FSRSGradientDecay; "
                    "install it with: pip install imprint-mem[online]"
                ) from e
            raise ImportError(
                f"FSRSGradientDecay failed to import river: missing transitive "
                f"dependency '{missing}'. Try: pip install imprint-mem[online]"
            ) from e
        return (  # type: ignore[no-any-return]
            preprocessing.StandardScaler()  # type: ignore[no-untyped-call]
            | linear_model.LinearRegression(  # type: ignore[no-untyped-call]
                optimizer=optim.SGD(self._lr)  # type: ignore[no-untyped-call]
            )
        )

    def _warmup(self, memory: Memory, now: datetime) -> None:
        """Seed the model with neutral (outcome=1.0) samples so initial predictions
        are in a useful range rather than the zero-weight default."""
        for _ in range(self._N_WARMUP):
            self._model.learn_one(_features(memory, now), 1.0)
        self._warmed_up = True

    def initial_stability(self, memory: Memory) -> float:
        return 5.0

    def update_on_merge(self, memory: Memory) -> float:
        return min(memory.stability + 1.0, 100.0)

    def update_on_contradict(self, memory: Memory) -> float:
        return max(memory.stability * 0.1, 0.1)

    def update_on_recall(self, memory: Memory) -> float:
        return memory.stability

    def effective_stability(self, memory: Memory, now: datetime) -> float:
        """FSRS base stability scaled by the learned quality multiplier."""
        if not self._warmed_up:
            self._warmup(memory, now)
        base = _fsrs_base(memory, now)
        raw = self._model.predict_one(_features(memory, now))
        multiplier = max(_MULT_MIN, min(_MULT_MAX, raw))
        return base * multiplier

    def learn(self, memory: Memory, now: datetime, outcome: float) -> None:
        """Train the model to predict the outcome quality multiplier.

        outcome is expected in [0.0, 1.0] (or slightly outside for attribution
        signals). The model directly learns to predict this value; effective_stability
        then uses it as a multiplier on the static FSRS formula.
        """
        if not self._warmed_up:
            self._warmup(memory, now)
        self._model.learn_one(_features(memory, now), outcome)

    def get_state(self) -> str:
        return base64.b64encode(pickle.dumps((self._model, self._warmed_up))).decode()

    def set_state(self, state_b64: str) -> None:
        import contextlib

        with contextlib.suppress(Exception):
            obj: Any = pickle.loads(base64.b64decode(state_b64))
            if isinstance(obj, tuple) and len(obj) == 2:  # type: ignore[misc]
                self._model, warmed = obj[0], obj[1]  # type: ignore[misc]
                self._warmed_up = bool(warmed)  # type: ignore[arg-type]
            else:
                # Legacy state (model only, no warmup flag).
                self._model = obj
                self._warmed_up = True
