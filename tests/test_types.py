from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from pydantic import ValidationError

from imprint import Memory, MemorySource, MemoryType


def _make_memory(**overrides: Any) -> Memory:
    now = datetime.now(UTC)
    fields: dict[str, Any] = {
        "id": "m_001",
        "agent_id": "agent_x",
        "user_id": "user_y",
        "type": MemoryType.RULE,
        "scope": "global",
        "content": "Be direct.",
        "source": MemorySource.DETECTED,
        "valid_from": now,
        "created_at": now,
        "updated_at": now,
    }
    fields.update(overrides)
    return Memory(**fields)


def test_naive_datetime_is_rejected() -> None:
    with pytest.raises(ValidationError):
        _make_memory(valid_from=datetime.now())  # noqa: DTZ005


def test_valid_until_must_be_after_valid_from() -> None:
    now = datetime.now(UTC)
    with pytest.raises(ValidationError, match="valid_until"):
        _make_memory(valid_from=now, valid_until=now - timedelta(seconds=1))


def test_supersedence_without_contradiction_is_allowed() -> None:
    """A MERGE consolidation sets superseded_by without valid_until.

    The memory wasn't contradicted, just absorbed into another. Both
    arrangements (with or without valid_until) are valid.
    """
    mem = _make_memory(superseded_by="m_002")
    assert mem.superseded_by == "m_002"
    assert mem.valid_until is None


def test_agent_level_memory_allows_null_user_id() -> None:
    mem = _make_memory(user_id=None)
    assert mem.user_id is None
