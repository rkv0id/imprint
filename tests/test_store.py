import pytest

from imprint import Store


async def _opened_store() -> Store:
    store = Store(":memory:")
    await store.connect()
    await store.init_schema()
    return store


async def test_init_schema_creates_expected_tables() -> None:
    store = await _opened_store()
    cursor = await store.conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
    )
    tables = {row[0] for row in await cursor.fetchall()}
    assert {"memories", "signals", "memory_sources"} <= tables
    await store.close()


async def test_init_schema_is_idempotent() -> None:
    store = await _opened_store()
    await store.init_schema()
    await store.init_schema()
    await store.close()


async def test_user_id_nullable_for_agent_level_memories() -> None:
    store = await _opened_store()
    await store.conn.execute(
        """
        INSERT INTO memories (
            id, agent_id, user_id, type, scope, content, source,
            valid_from, created_at, updated_at
        ) VALUES (?, ?, NULL, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "m_agent_level",
            "agent_x",
            "rule",
            "global",
            "Be direct.",
            "detected",
            "2026-01-01T00:00:00+00:00",
            "2026-01-01T00:00:00+00:00",
            "2026-01-01T00:00:00+00:00",
        ),
    )
    await store.conn.commit()

    cursor = await store.conn.execute("SELECT user_id FROM memories WHERE id = 'm_agent_level'")
    row = await cursor.fetchone()
    assert row is not None
    assert row[0] is None
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
