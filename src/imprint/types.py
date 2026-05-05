"""Core data types for Imprint memories and signals."""

from enum import StrEnum

from pydantic import AwareDatetime, BaseModel, ConfigDict, model_validator


class BudgetExceededError(Exception):
    """Raised when memory content cannot be reduced to fit within max_input_tokens.

    This happens when even a single memory, combined with the fixed prompt
    overhead, exceeds the configured token budget.
    """


class MemoryType(StrEnum):
    FACT = "fact"
    RULE = "rule"
    DECISION = "decision"
    CONTEXT = "context"


class MemorySource(StrEnum):
    DETECTED = "detected"
    USER_EDIT = "user_edit"
    CONSOLIDATION = "consolidation"
    IMPORT = "import"


class SignalType(StrEnum):
    CORRECTION = "correction"
    DIRECTION = "direction"
    PREFERENCE = "preference"
    FACT = "fact"
    REINFORCEMENT = "reinforcement"


class Memory(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    agent_id: str
    user_id: str | None  # None => agent-level memory, shared across users
    type: MemoryType
    scope: str
    content: str
    source: MemorySource
    stability: float = 5.0
    recall_count: int = 0
    valid_from: AwareDatetime
    valid_until: AwareDatetime | None = None
    superseded_by: str | None = None
    pinned: bool = False
    active: bool = True
    created_at: AwareDatetime
    updated_at: AwareDatetime
    last_triggered: AwareDatetime | None = None

    @model_validator(mode="after")
    def _check_temporal_validity(self) -> "Memory":
        if self.valid_until is not None and self.valid_until <= self.valid_from:
            raise ValueError("valid_until must be after valid_from")
        return self


class Signal(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    agent_id: str
    user_id: str | None  # None => agent-level signal
    signal_type: SignalType
    content: str
    prediction_delta: str | None = None
    context: str | None = None
    memory_id: str | None = None
    contradicted: bool = False
    created_at: AwareDatetime


class MemoryEvent(BaseModel):
    """A single logged event for one memory."""

    model_config = ConfigDict(extra="forbid")

    memory_id: str
    event_type: str
    detail: dict[str, object] | None = None
    occurred_at: AwareDatetime


class MemoryLineage(BaseModel):
    """Full history of one memory: origin signal, supersession chain, events."""

    model_config = ConfigDict(extra="forbid")

    memory: Memory
    created_by_signal: Signal | None = None
    superseded_memories: list[Memory] = []
    superseded_by: Memory | None = None
    events: list[MemoryEvent] = []


class MemoryHealth(BaseModel):
    """Aggregate health statistics for a user's memory store."""

    model_config = ConfigDict(extra="forbid")

    total: int
    active: int
    by_scope: dict[str, int]
    by_type: dict[str, int]
    pinned: int
    avg_recall_count: float
    oldest_active: AwareDatetime | None = None
    newest_active: AwareDatetime | None = None
