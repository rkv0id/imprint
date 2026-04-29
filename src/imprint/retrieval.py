"""Retrieval utilities: alpha tuners and RRF fusion.

StaticAlphaTuner and BanditAlphaTuner implement the AlphaTuner protocol
for controlling the sparse/dense balance in hybrid retrieval.

BanditAlphaTuner uses Thompson sampling over a fixed set of alpha candidates.
State (Beta distribution parameters) is persisted to agent_config so the
bandit survives process restarts.
"""

from __future__ import annotations

import random
import re

_ARMS: list[float] = [0.1, 0.3, 0.5, 0.7, 0.9]
_RRF_K: int = 60
_FTS_SPECIAL = re.compile(r"[\"*()\^+\-]")


class StaticAlphaTuner:
    """Fixed alpha with no learning. Default adapter."""

    def __init__(self, alpha: float = 0.3) -> None:
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")
        self._alpha = alpha

    def get_alpha(self, query: str | None = None) -> float:
        return self._alpha

    async def update(self, alpha_used: float, reward: float) -> None:
        pass


class BanditAlphaTuner:
    """Thompson sampling bandit over five alpha candidates.

    Arms: [0.1, 0.3, 0.5, 0.7, 0.9] (sparse weight; dense = 1 - alpha).
    Each arm has a Beta(successes+1, failures+1) distribution. On get_alpha(),
    one sample is drawn from each arm and the highest wins (Thompson sampling).
    On update(), the arm closest to alpha_used receives a success or failure.

    State is two lists of floats (successes, failures) persisted to
    agent_config. get_state() and set_state() are used by Imprint for
    persistence -- they are not part of the AlphaTuner protocol.

    Reward signal: 1.0 when the retrieved set contained a memory that was
    subsequently merged or contradicted (retrieval found what consolidation
    confirmed was relevant). 0.0 otherwise.
    """

    def __init__(self) -> None:
        self._successes: list[float] = [1.0] * len(_ARMS)
        self._failures: list[float] = [1.0] * len(_ARMS)

    def get_alpha(self, query: str | None = None) -> float:
        samples = [
            random.betavariate(s + 1.0, f + 1.0)
            for s, f in zip(self._successes, self._failures, strict=True)
        ]
        return _ARMS[samples.index(max(samples))]

    async def update(self, alpha_used: float, reward: float) -> None:
        arm = min(range(len(_ARMS)), key=lambda i: abs(_ARMS[i] - alpha_used))
        if reward >= 0.5:
            self._successes[arm] += reward
        else:
            self._failures[arm] += 1.0 - reward

    def get_state(self) -> dict[str, list[float]]:
        return {"s": list(self._successes), "f": list(self._failures)}

    def set_state(self, state: dict[str, list[float]]) -> None:
        s = state.get("s", [1.0] * len(_ARMS))
        f = state.get("f", [1.0] * len(_ARMS))
        if len(s) == len(_ARMS) and len(f) == len(_ARMS):
            self._successes = s
            self._failures = f


def rrf_fuse(
    *,
    candidates: list[str],
    sparse_ranks: dict[str, int],
    dense_ranks: dict[str, int],
    alpha: float,
) -> list[str]:
    """Reciprocal Rank Fusion over sparse and dense rank lists.

    Returns candidates sorted by descending RRF score.

    score(id) = alpha * 1/(k + sparse_rank) + (1-alpha) * 1/(k + dense_rank)

    Candidates absent from a channel receive rank = len(candidates) + 1,
    which gives them a low but non-zero score from that channel.
    """
    n = len(candidates)
    fallback = n + 1

    def score(cid: str) -> float:
        sr = sparse_ranks.get(cid, fallback)
        dr = dense_ranks.get(cid, fallback)
        return alpha * (1.0 / (_RRF_K + sr)) + (1.0 - alpha) * (1.0 / (_RRF_K + dr))

    return sorted(candidates, key=score, reverse=True)


def sanitize_fts_query(text: str) -> str:
    """Strip FTS5 special characters from a natural language context string."""
    cleaned = _FTS_SPECIAL.sub(" ", text)
    return " ".join(cleaned.split())
