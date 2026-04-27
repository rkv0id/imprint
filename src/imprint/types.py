"""Core data types for Imprint memories and signals."""

from enum import StrEnum

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, model_validator


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
    IMPLICIT = "implicit"


class ContextStat(BaseModel):
    """Per-context validation/contradiction counts on a memory."""

    model_config = ConfigDict(extra="forbid")

    validations: int = 0
    contradictions: int = 0


class Memory(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    agent_id: str
    user_id: str | None  # None => agent-level memory, shared across users
    type: MemoryType
    scope: str
    domain: str | None = None
    content: str
    applicability: str | None = None
    context_keys: list[str] = Field(default_factory=list)
    context_stats: dict[str, ContextStat] = Field(default_factory=dict)
    source: MemorySource
    stability: float = 5.0
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
        if self.superseded_by is not None and self.valid_until is None:
            raise ValueError("superseded_by requires valid_until to be set")
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
