"""Adapter protocols for Imprint.

Each Protocol defines a swappable interface. Default implementations ship in
core; optional extras (imprint[vector], imprint[postgres], etc.) provide
alternatives.

Compiler, Detector, and Deriver are defined here for completeness but their
default implementations remain embedded in Imprint via pydantic-ai agents.
Extraction into named adapter classes is deferred.
"""

from datetime import datetime
from typing import Any, Protocol

from imprint.types import Memory, MemoryType, Signal, SignalType


class MemoryStore(Protocol):
    async def connect(self) -> None: ...

    async def close(self) -> None: ...

    async def init_schema(self) -> None: ...

    async def insert_memory(self, memory: Memory) -> None: ...

    async def insert_signal(self, signal: Signal) -> None: ...

    async def link_signal_to_memory(
        self,
        *,
        memory_id: str,
        signal_id: str,
        weight: float = 1.0,
    ) -> None: ...

    async def list_memories(
        self,
        agent_id: str,
        user_id: str | None,
        *,
        memory_type: MemoryType | None = None,
        scopes: list[str] | None = None,
        active_only: bool = True,
    ) -> list[Memory]: ...

    async def deactivate_memory(
        self,
        memory_id: str,
        *,
        superseded_by: str | None = None,
        valid_until: datetime | None = None,
    ) -> bool: ...

    async def mark_signals_contradicted(self, memory_id: str) -> None: ...

    async def get_cached_policy(self, cache_key: str) -> tuple[str, datetime] | None: ...

    async def put_cached_policy(
        self,
        *,
        cache_key: str,
        agent_id: str,
        user_id: str | None,
        policy_text: str,
        compiled_at: datetime,
    ) -> None: ...

    async def invalidate_cached_policies(self, agent_id: str, user_id: str | None) -> None: ...

    async def update_memory_stability(self, memory_id: str, stability: float) -> None: ...

    async def increment_recall_count(self, memory_id: str) -> None: ...

    async def search_fts(
        self,
        query: str,
        candidate_ids: set[str],
        limit: int = 200,
    ) -> list[tuple[str, float]]: ...

    async def get_agent_config(self, agent_id: str) -> Any: ...

    async def put_agent_config(
        self,
        *,
        agent_id: str,
        processing_mode: str,
        agent_description: str | None,
        scopes: list[str],
    ) -> None: ...

    async def put_alpha_tuner_state(self, agent_id: str, state: str) -> None: ...

    async def put_gradient_state(self, agent_id: str, state: str) -> None: ...


class EventLogger(Protocol):
    async def log(
        self,
        memory_id: str,
        event_type: str,
        metadata: dict[str, Any] | None = None,
    ) -> None: ...


class TokenCounter(Protocol):
    def count(self, text: str) -> int: ...


class DecayModel(Protocol):
    def initial_stability(self, memory: Memory) -> float: ...

    def update_on_merge(self, memory: Memory) -> float: ...

    def update_on_contradict(self, memory: Memory) -> float: ...

    def update_on_recall(self, memory: Memory) -> float: ...

    def effective_stability(self, memory: Memory, now: datetime) -> float: ...


class VectorStore(Protocol):
    async def upsert(self, id: str, embedding: list[float]) -> None: ...

    async def search(self, embedding: list[float], top_k: int) -> list[tuple[str, float]]: ...

    async def delete(self, id: str) -> None: ...


class Embedder(Protocol):
    async def embed(self, text: str) -> list[float]: ...

    async def embed_batch(self, texts: list[str]) -> list[list[float]]: ...

    @property
    def dim(self) -> int: ...


class AlphaTuner(Protocol):
    def get_alpha(self, query: str | None = None) -> float: ...

    async def update(self, alpha_used: float, reward: float) -> None: ...


class Compiler(Protocol):
    async def compile(
        self,
        *,
        memories: list[Memory],
        agent_description: str | None,
        context: str | None,
        existing_instructions: str | None,
        max_tokens: int,
    ) -> str: ...


class Detector(Protocol):
    async def detect(
        self,
        *,
        agent_output: str,
        user_response: str,
    ) -> SignalType | None: ...


class Deriver(Protocol):
    async def derive(
        self,
        *,
        agent_output: str,
        user_response: str,
        signal_type: SignalType,
        available_scopes: list[str],
    ) -> tuple[MemoryType, str, str]: ...
