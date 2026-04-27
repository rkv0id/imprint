from datetime import UTC, datetime
from typing import Any

import pytest

from imprint import (
    ContextStat,
    Memory,
    MemorySource,
    MemoryType,
    Signal,
    SignalType,
    Store,
)


def _make_memory(**overrides: Any) -> Memory:
    now = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)
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


async def _opened_store() -> Store:
    store = Store(":memory:")
    await store.connect()
    await store.init_schema()
    return store


# ---------- schema -----------------------------------------------------------


async def test_init_schema_creates_expected_tables() -> None:
    store = await _opened_store()
    cursor = await store.conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
    )
    tables = {row["name"] for row in await cursor.fetchall()}
    assert {"memories", "signals", "memory_sources"} <= tables
    await store.close()


async def test_init_schema_is_idempotent() -> None:
    store = await _opened_store()
    await store.init_schema()
    await store.init_schema()
    await store.close()


async def test_foreign_keys_are_enforced() -> None:
    store = await _opened_store()
    with pytest.raises(Exception, match="FOREIGN KEY"):
        await store.conn.execute(
            "INSERT INTO memory_sources (memory_id, signal_id) VALUES (?, ?)",
            ("nonexistent_mem", "nonexistent_sig"),
        )
        await store.conn.commit()
    await store.close()


async def test_connect_is_idempotent() -> None:
    store = Store(":memory:")
    await store.connect()
    await store.connect()
    assert store.conn is not None
    await store.close()


async def test_conn_property_raises_when_not_connected() -> None:
    store = Store(":memory:")
    with pytest.raises(RuntimeError, match="not connected"):
        _ = store.conn


# ---------- round-trip -------------------------------------------------------


async def test_memory_round_trip_preserves_all_fields() -> None:
    store = await _opened_store()
    original = _make_memory(
        domain="coding",
        applicability="Coding contexts only.",
        context_keys=["python", "review"],
        context_stats={"python": ContextStat(validations=3, contradictions=1)},
        stability=7.5,
        pinned=True,
    )
    await store.insert_memory(original)

    [retrieved] = await store.list_memories("agent_x", "user_y")
    assert retrieved == original
    await store.close()


async def test_agent_level_memory_round_trips_with_null_user_id() -> None:
    store = await _opened_store()
    mem = _make_memory(id="m_agent_level", user_id=None)
    await store.insert_memory(mem)

    [retrieved] = await store.list_memories("agent_x", None)
    assert retrieved == mem
    assert retrieved.user_id is None
    await store.close()


async def test_optional_datetime_fields_round_trip() -> None:
    store = await _opened_store()
    valid_from = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)
    valid_until = datetime(2026, 6, 1, 12, 0, 0, tzinfo=UTC)
    last_triggered = datetime(2026, 3, 1, 12, 0, 0, tzinfo=UTC)

    replacement = _make_memory(id="m_replacement", valid_from=valid_until)
    await store.insert_memory(replacement)

    superseded = _make_memory(
        id="m_superseded",
        valid_from=valid_from,
        valid_until=valid_until,
        last_triggered=last_triggered,
        superseded_by="m_replacement",
        active=False,
    )
    await store.insert_memory(superseded)

    all_memories = await store.list_memories("agent_x", "user_y", active_only=False)
    by_id = {m.id: m for m in all_memories}
    assert by_id["m_superseded"].valid_until == valid_until
    assert by_id["m_superseded"].last_triggered == last_triggered
    assert by_id["m_superseded"].superseded_by == "m_replacement"
    await store.close()


# ---------- filtering --------------------------------------------------------


async def test_list_memories_filters_user_id_null_vs_value() -> None:
    store = await _opened_store()
    pair_mem = _make_memory(id="m_pair", user_id="user_y")
    agent_mem = _make_memory(id="m_agent", user_id=None)
    await store.insert_memory(pair_mem)
    await store.insert_memory(agent_mem)

    pair_results = await store.list_memories("agent_x", "user_y")
    assert [m.id for m in pair_results] == ["m_pair"]

    agent_results = await store.list_memories("agent_x", None)
    assert [m.id for m in agent_results] == ["m_agent"]
    await store.close()


async def test_list_memories_filters_by_type() -> None:
    store = await _opened_store()
    rule = _make_memory(id="r_1", type=MemoryType.RULE)
    fact = _make_memory(id="f_1", type=MemoryType.FACT)
    await store.insert_memory(rule)
    await store.insert_memory(fact)

    rules = await store.list_memories("agent_x", "user_y", memory_type=MemoryType.RULE)
    assert [m.id for m in rules] == ["r_1"]
    await store.close()


async def test_list_memories_excludes_inactive_by_default() -> None:
    store = await _opened_store()
    active = _make_memory(id="m_active", active=True)
    inactive = _make_memory(id="m_inactive", active=False)
    await store.insert_memory(active)
    await store.insert_memory(inactive)

    visible = await store.list_memories("agent_x", "user_y")
    assert [m.id for m in visible] == ["m_active"]

    all_ = await store.list_memories("agent_x", "user_y", active_only=False)
    assert {m.id for m in all_} == {"m_active", "m_inactive"}
    await store.close()


# ---------- signals & provenance --------------------------------------------


async def test_signal_insert_and_link_to_memory() -> None:
    store = await _opened_store()
    mem = _make_memory()
    await store.insert_memory(mem)

    sig = Signal(
        id="s_001",
        agent_id="agent_x",
        user_id="user_y",
        signal_type=SignalType.CORRECTION,
        content="Don't use bullet points",
        created_at=datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC),
    )
    await store.insert_signal(sig)
    await store.link_signal_to_memory(memory_id=mem.id, signal_id=sig.id, weight=0.8)

    cursor = await store.conn.execute("SELECT memory_id, signal_id, weight FROM memory_sources")
    rows = list(await cursor.fetchall())
    assert len(rows) == 1
    assert rows[0]["memory_id"] == mem.id
    assert rows[0]["signal_id"] == sig.id
    assert rows[0]["weight"] == 0.8
    await store.close()


async def test_link_rejects_dangling_memory_reference() -> None:
    store = await _opened_store()
    sig = Signal(
        id="s_001",
        agent_id="agent_x",
        user_id="user_y",
        signal_type=SignalType.CORRECTION,
        content="x",
        created_at=datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC),
    )
    await store.insert_signal(sig)
    with pytest.raises(Exception, match="FOREIGN KEY"):
        await store.link_signal_to_memory(memory_id="does_not_exist", signal_id=sig.id)
    await store.close()
