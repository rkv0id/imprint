"""Tests for item 8: observability API.

Covers list_events, memory_lineage, memory_health on Imprint, and the
underlying store methods list_events and get_memory_with_supersession.
"""

from datetime import UTC, datetime
from typing import cast

import pytest
from helpers import _make_imprint
from pydantic_ai.models.test import TestModel

from imprint import (
    Imprint,
    MemoryEvent,
    MemoryHealth,
    MemoryLineage,
    SQLiteMemoryStore,
)
from imprint.types import Memory, MemorySource, MemoryType


def _mem(
    store: SQLiteMemoryStore,
    *,
    memory_id: str = "m1",
    scope: str = "global",
    content: str = "always be concise",
    active: bool = True,
    superseded_by: str | None = None,
) -> Memory:
    now = datetime.now(UTC)
    return Memory(
        id=memory_id,
        agent_id="agent",
        user_id="u",
        type=MemoryType.RULE,
        scope=scope,
        content=content,
        source=MemorySource.DETECTED,
        valid_from=now,
        created_at=now,
        updated_at=now,
        active=active,
        superseded_by=superseded_by,
    )


async def test_list_events_empty_when_no_events() -> None:
    imprint, _, _, _, _, _, _ = _make_imprint()
    await imprint.connect()
    store = cast(SQLiteMemoryStore, imprint._store)
    await store.insert_memory(_mem(store))

    events = await imprint.list_events("u")
    assert events == []


async def test_list_events_returns_logged_events() -> None:
    imprint, _, _, _, _, _, _ = _make_imprint(processing_mode="frugal")
    await imprint.connect()

    await imprint.observe_directions(user_id="u", directions=["always be direct"])
    memories = await imprint.list_memories("u")
    assert len(memories) == 1

    # recall events are logged when get_policy retrieves memories
    await imprint.get_policy(user_id="u")

    events = await imprint.list_events("u")
    assert len(events) >= 1
    assert all(isinstance(e, MemoryEvent) for e in events)
    assert all(e.memory_id == memories[0].id for e in events)


async def test_list_events_scoped_to_memory_id() -> None:
    imprint, _, _, _, _, _, _ = _make_imprint(processing_mode="frugal")
    await imprint.connect()

    await imprint.observe_directions(user_id="u", directions=["direction one"])
    await imprint.observe_directions(user_id="u", directions=["direction two"])
    memories = await imprint.list_memories("u")
    assert len(memories) == 2

    m1_id = memories[0].id
    events = await imprint.list_events("u", memory_id=m1_id)
    assert all(e.memory_id == m1_id for e in events)


async def test_list_events_respects_limit() -> None:
    imprint, _, _, _, _, _, _ = _make_imprint(processing_mode="frugal")
    await imprint.connect()

    for i in range(5):
        await imprint.observe_directions(user_id="u", directions=[f"direction {i}"])

    events = await imprint.list_events("u", limit=3)
    assert len(events) <= 3


async def test_list_events_newest_first() -> None:
    imprint, _, _, _, _, _, _ = _make_imprint(processing_mode="frugal")
    await imprint.connect()

    await imprint.observe_directions(user_id="u", directions=["first", "second"])
    events = await imprint.list_events("u")

    if len(events) >= 2:
        assert events[0].occurred_at >= events[1].occurred_at


async def test_store_list_events_filters_by_memory_id() -> None:
    store = SQLiteMemoryStore(":memory:")
    await store.connect()
    await store.init_schema()

    now = datetime.now(UTC)
    m1 = Memory(
        id="m1",
        agent_id="a",
        user_id="u",
        type=MemoryType.RULE,
        scope="global",
        content="x",
        source=MemorySource.DETECTED,
        valid_from=now,
        created_at=now,
        updated_at=now,
    )
    m2 = Memory(
        id="m2",
        agent_id="a",
        user_id="u",
        type=MemoryType.RULE,
        scope="global",
        content="y",
        source=MemorySource.DETECTED,
        valid_from=now,
        created_at=now,
        updated_at=now,
    )
    await store.insert_memory(m1)
    await store.insert_memory(m2)

    from imprint.stores.sqlite import SQLiteEventLogger

    logger = SQLiteEventLogger(store)
    await logger.log("m1", "derive", {"scope": "global"})
    await logger.log("m2", "derive", {})

    events = await store.list_events("a", "u", memory_id="m1")
    assert len(events) == 1
    assert events[0].memory_id == "m1"
    await store.close()


async def test_store_get_memory_returns_none_for_unknown() -> None:
    store = SQLiteMemoryStore(":memory:")
    await store.connect()
    await store.init_schema()
    result = await store.get_memory("nonexistent")
    assert result is None
    await store.close()


async def test_store_get_memory_returns_memory() -> None:
    store = SQLiteMemoryStore(":memory:")
    await store.connect()
    await store.init_schema()

    now = datetime.now(UTC)
    m = Memory(
        id="m1",
        agent_id="a",
        user_id="u",
        type=MemoryType.RULE,
        scope="global",
        content="x",
        source=MemorySource.DETECTED,
        valid_from=now,
        created_at=now,
        updated_at=now,
    )
    await store.insert_memory(m)
    result = await store.get_memory("m1")
    assert result is not None
    assert result.id == "m1"
    await store.close()


async def test_store_get_memory_with_supersession() -> None:
    store = SQLiteMemoryStore(":memory:")
    await store.connect()
    await store.init_schema()

    now = datetime.now(UTC)
    m_old = Memory(
        id="m_old",
        agent_id="a",
        user_id="u",
        type=MemoryType.RULE,
        scope="global",
        content="old rule",
        source=MemorySource.DETECTED,
        valid_from=now,
        created_at=now,
        updated_at=now,
    )
    await store.insert_memory(m_old)

    m_new = Memory(
        id="m_new",
        agent_id="a",
        user_id="u",
        type=MemoryType.RULE,
        scope="global",
        content="new rule",
        source=MemorySource.DETECTED,
        valid_from=now,
        created_at=now,
        updated_at=now,
    )
    await store.insert_memory(m_new)
    await store.deactivate_memory("m_old", superseded_by="m_new")

    successor, predecessor = await store.get_memory_with_supersession("m_old")
    assert successor is not None and successor.id == "m_new"
    assert predecessor is None

    successor2, predecessor2 = await store.get_memory_with_supersession("m_new")
    assert successor2 is None
    assert predecessor2 is not None and predecessor2.id == "m_old"

    await store.close()


async def test_store_get_superseded_memories() -> None:
    store = SQLiteMemoryStore(":memory:")
    await store.connect()
    await store.init_schema()

    now = datetime.now(UTC)
    for i in range(2):
        m = Memory(
            id=f"m_old_{i}",
            agent_id="a",
            user_id="u",
            type=MemoryType.RULE,
            scope="global",
            content=f"old {i}",
            source=MemorySource.DETECTED,
            valid_from=now,
            created_at=now,
            updated_at=now,
        )
        await store.insert_memory(m)

    m_new = Memory(
        id="m_new",
        agent_id="a",
        user_id="u",
        type=MemoryType.RULE,
        scope="global",
        content="new",
        source=MemorySource.DETECTED,
        valid_from=now,
        created_at=now,
        updated_at=now,
    )
    await store.insert_memory(m_new)
    await store.deactivate_memory("m_old_0", superseded_by="m_new")
    await store.deactivate_memory("m_old_1", superseded_by="m_new")

    superseded = await store.get_superseded_memories("m_new")
    assert len(superseded) == 2
    assert {m.id for m in superseded} == {"m_old_0", "m_old_1"}
    await store.close()


async def test_memory_lineage_raises_for_unknown_id() -> None:
    imprint, _, _, _, _, _, _ = _make_imprint()
    await imprint.connect()

    with pytest.raises(KeyError, match="nonexistent"):
        await imprint.memory_lineage("nonexistent")


async def test_memory_lineage_basic() -> None:
    imprint, _, _, _, _, _, _ = _make_imprint(processing_mode="frugal")
    await imprint.connect()

    await imprint.observe_directions(user_id="u", directions=["always be direct"])
    memories = await imprint.list_memories("u")
    assert len(memories) == 1

    lineage = await imprint.memory_lineage(memories[0].id)
    assert isinstance(lineage, MemoryLineage)
    assert lineage.memory.id == memories[0].id
    assert lineage.superseded_by is None
    assert isinstance(lineage.events, list)


async def test_memory_lineage_tracks_supersession() -> None:
    imprint, _, _, _, _, _, _ = _make_imprint(processing_mode="frugal")
    await imprint.connect()

    await imprint.observe_directions(user_id="u", directions=["use formal English"])
    old_memories = await imprint.list_memories("u")
    old_id = old_memories[0].id

    imprint.processing_mode = "balanced"  # type: ignore[assignment]

    with (
        imprint._derive_agent.override(
            model=TestModel(
                custom_output_args={
                    "memory_type": "rule",
                    "content": "always use British English",
                    "scope": "global",
                }
            )
        ),
        imprint._batch_consolidate_agent.override(
            model=TestModel(
                custom_output_args={
                    "decisions": [{"candidate_index": 0, "memory_id": old_id, "action": "merge"}]
                }
            )
        ),
    ):
        await imprint.observe_directions(user_id="u", directions=["always use British English"])
        await imprint.drain()

    lineage = await imprint.memory_lineage(old_id)
    assert lineage.superseded_by is not None
    assert lineage.superseded_by.content == "always use British English"


async def test_memory_health_empty_user() -> None:
    imprint, _, _, _, _, _, _ = _make_imprint()
    await imprint.connect()

    health = await imprint.memory_health("u")
    assert isinstance(health, MemoryHealth)
    assert health.total == 0
    assert health.active == 0
    assert health.by_scope == {}
    assert health.by_type == {}
    assert health.pinned == 0
    assert health.avg_recall_count == 0.0
    assert health.oldest_active is None
    assert health.newest_active is None


async def test_memory_health_counts_active_and_total() -> None:
    imprint, _, _, _, _, _, _ = _make_imprint(processing_mode="frugal")
    await imprint.connect()

    await imprint.observe_directions(user_id="u", directions=["rule one", "rule two"])
    memories = await imprint.list_memories("u")
    old_id = memories[0].id
    await imprint.deactivate_memory("u", old_id)

    health = await imprint.memory_health("u")
    assert health.total == 2
    assert health.active == 1


async def test_memory_health_by_scope_and_type() -> None:
    imprint, _, _, _, _, _, _ = _make_imprint(processing_mode="frugal")
    await imprint.connect()

    await imprint.observe_directions(user_id="u", directions=["rule one"])

    health = await imprint.memory_health("u")
    assert health.active == 1
    assert "global" in health.by_scope
    assert health.by_scope["global"] == 1
    assert "rule" in health.by_type
    assert health.by_type["rule"] == 1


async def test_memory_health_pinned_count() -> None:
    imprint, _, _, _, _, _, _ = _make_imprint(processing_mode="frugal")
    await imprint.connect()

    await imprint.observe_directions(user_id="u", directions=["rule one", "rule two"])
    memories = await imprint.list_memories("u")
    await imprint.pin_memory(memories[0].id)

    health = await imprint.memory_health("u")
    assert health.pinned == 1


async def test_memory_health_oldest_newest() -> None:
    imprint, _, _, _, _, _, _ = _make_imprint(processing_mode="frugal")
    await imprint.connect()

    await imprint.observe_directions(user_id="u", directions=["rule one"])
    await imprint.observe_directions(user_id="u", directions=["rule two"])

    health = await imprint.memory_health("u")
    assert health.oldest_active is not None
    assert health.newest_active is not None
    assert health.newest_active >= health.oldest_active


# -- alpha_estimate -----------------------------------------------------------


async def test_alpha_estimate_static_returns_fixed_alpha() -> None:
    from imprint.retrieval import StaticAlphaTuner

    imprint = Imprint(
        agent_id="obs-agent",
        store=":memory:",
        alpha_tuner=StaticAlphaTuner(alpha=0.7),
    )
    await imprint.connect()
    assert imprint.alpha_estimate == 0.7


async def test_alpha_estimate_bandit_default_is_deterministic() -> None:
    """Uniform Beta priors -- all arms have equal mean. Result is stable across calls."""
    from imprint.retrieval import BanditAlphaTuner

    imprint = Imprint(
        agent_id="obs-agent-bandit",
        store=":memory:",
        alpha_tuner=BanditAlphaTuner(),
    )
    await imprint.connect()
    # Uniform priors -> all arm means equal -> first arm wins by index -> 0.1
    assert imprint.alpha_estimate == 0.1
    # Must be stable (not sampled)
    assert imprint.alpha_estimate == imprint.alpha_estimate


async def test_alpha_estimate_bandit_shifts_after_successes() -> None:
    """After accumulating successes on the high-alpha arm, estimate shifts upward."""
    from imprint.retrieval import BanditAlphaTuner

    tuner = BanditAlphaTuner()
    # Drive successes toward arm 4 (alpha=0.9)
    for _ in range(20):
        await tuner.update(alpha_used=0.9, reward=1.0)

    imprint = Imprint(
        agent_id="obs-agent-shifted",
        store=":memory:",
        alpha_tuner=tuner,
    )
    await imprint.connect()
    assert imprint.alpha_estimate == 0.9
