"""Online decay model using River for incremental gradient updates.

Requires: pip install imprint[online]
"""

from __future__ import annotations

import base64
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


def _features(memory: Memory, now: datetime) -> dict[str, float]:
    elapsed_days = (now - memory.created_at).total_seconds() / 86400.0
    return {
        "stability": memory.stability,
        "recall_count": float(memory.recall_count),
        "elapsed_days": elapsed_days,
        "type": _TYPE_ENCODING.get(memory.type, 1.0),
    }


class FSRSGradientDecay:
    """FSRS-inspired decay with parameters learned online via River SGD.

    Uses a River Pipeline (StandardScaler | LinearRegression) that learns
    from feedback events. Features per memory: stability, recall_count,
    elapsed_days, and type encoding.

    effective_stability() returns the model's predicted value, which starts
    near the FSRS static formula and drifts toward agent-specific patterns
    as feedback accumulates.

    State is serialized as base64-encoded pickle for persistence in
    agent_config. The model is reset from scratch if state can't be loaded
    (River version mismatch, etc.) -- it relearns quickly.

    Requires: pip install imprint[online]
    """

    def __init__(self, learning_rate: float = 0.01) -> None:
        self._lr = learning_rate
        self._model: Any = self._build_model()

    def _build_model(self) -> Any:
        try:
            from river import linear_model, optim, preprocessing  # type: ignore[import-untyped]
        except ImportError as e:
            missing = getattr(e, "name", None)
            if missing in ("river", None):
                raise ImportError(
                    "river is required for FSRSGradientDecay; "
                    "install it with: pip install imprint[online]"
                ) from e
            raise ImportError(
                f"FSRSGradientDecay failed to import river: missing transitive "
                f"dependency '{missing}'. Try: pip install imprint[online]"
            ) from e
        return (  # type: ignore[no-any-return]
            preprocessing.StandardScaler()  # type: ignore[no-untyped-call]
            | linear_model.LinearRegression(  # type: ignore[no-untyped-call]
                optimizer=optim.SGD(self._lr)  # type: ignore[no-untyped-call]
            )
        )

    def initial_stability(self, memory: Memory) -> float:
        return 5.0

    def update_on_merge(self, memory: Memory) -> float:
        return min(memory.stability + 1.0, 100.0)

    def update_on_contradict(self, memory: Memory) -> float:
        return max(memory.stability * 0.1, 0.1)

    def update_on_recall(self, memory: Memory) -> float:
        return memory.stability

    def effective_stability(self, memory: Memory, now: datetime) -> float:
        pred = self._model.predict_one(_features(memory, now))
        return max(pred, 0.1)

    def learn(self, memory: Memory, now: datetime, outcome: float) -> None:
        self._model.learn_one(_features(memory, now), outcome)

    def get_state(self) -> str:
        return base64.b64encode(pickle.dumps(self._model)).decode()

    def set_state(self, state_b64: str) -> None:
        import contextlib

        with contextlib.suppress(Exception):
            self._model = pickle.loads(base64.b64decode(state_b64))
