"""Live tests for PostgresMemoryStore and PostgresVectorStore.

Run with:
    IMPRINT_POSTGRES_URL=postgres://user:pass@localhost/imprint_test \\
        python -m pytest tests/test_postgres.py -m live -v

Requires a Postgres instance with a clean test database. The schema is
created fresh in each test via a temporary agent_id prefix.
"""

import os
from datetime import UTC, datetime
from typing import Any

import pytest

asyncpg = pytest.importorskip("asyncpg", reason="imprint-mem[postgres] not installed")

from imprint.postgres import PostgresMemoryStore, PostgresVectorStore  # noqa: E402
from imprint.types import Memory, MemorySource, MemoryType, Signal, SignalType  # noqa: E402

pytestmark = pytest.mark.live

POSTGRES_URL = os.environ.get("IMPRINT_POSTGRES_URL", "")


def _skip_if_no_url() -> None:
    if not POSTGRES_URL:
        pytest.skip("IMPRINT_POSTGRES_URL not set")


def _make_memory(agent_id: str, **overrides: Any) -> Memory:
    now = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)
    fields: dict[str, Any] = {
        "id": "m_001",
        "agent_id": agent_id,
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


def _make_signal(agent_id: str, **overrides: Any) -> Signal:
    now = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)
    fields: dict[str, Any] = {
        "id": "sig_001",
        "agent_id": agent_id,
        "user_id": "user_y",
        "signal_type": SignalType.CORRECTION,
        "content": "No bullet points.",
        "created_at": now,
    }
    fields.update(overrides)
    return Signal(**fields)


async def _fresh_store() -> PostgresMemoryStore:
    _skip_if_no_url()
    store = PostgresMemoryStore(POSTGRES_URL)
    await store.connect()
    await store.init_schema()
    return store


async def _insert(store: PostgresMemoryStore, agent_id: str, mem_id: str, **kw: Any) -> None:
    """Insert a memory with auto-filled timestamps."""
    now = datetime.now(UTC)
    await store.insert_memory(
        _make_memory(agent_id, id=mem_id, created_at=now, updated_at=now, valid_from=now, **kw)
    )


# -- Schema tests -------------------------------------------------------------


async def test_init_schema_is_idempotent() -> None:
    store = await _fresh_store()
    # Should not raise on second call.
    await store.init_schema()
    await store.close()


async def test_connect_is_idempotent() -> None:
    _skip_if_no_url()
    store = PostgresMemoryStore(POSTGRES_URL)
    await store.connect()
    await store.connect()  # second call is a no-op
    assert store._pool is not None
    await store.close()


async def test_pool_property_raises_when_not_connected() -> None:
    _skip_if_no_url()
    store = PostgresMemoryStore(POSTGRES_URL)
    with pytest.raises(RuntimeError, match="not connected"):
        _ = store.pool


# -- Memory round-trip --------------------------------------------------------


async def test_memory_round_trip_preserves_all_fields() -> None:
    store = await _fresh_store()
    agent_id = f"pg_test_{datetime.now(UTC).timestamp()}"
    now = datetime(2026, 1, 2, 10, 30, 0, tzinfo=UTC)
    mem = _make_memory(
        agent_id,
        id="m_rt_001",
        stability=7.5,
        recall_count=3,
        pinned=True,
        valid_until=datetime(2026, 6, 1, tzinfo=UTC),
        last_triggered=now,
        created_at=now,
        updated_at=now,
        valid_from=now,
    )
    await store.insert_memory(mem)
    result = await store.get_memory("m_rt_001")
    assert result is not None
    assert result.id == "m_rt_001"
    assert result.agent_id == agent_id
    assert result.stability == 7.5
    assert result.recall_count == 3
    assert result.pinned is True
    assert result.active is True
    assert result.valid_until is not None
    assert result.last_triggered is not None
    # Datetimes come back as timezone-aware from Postgres.
    assert result.valid_from.tzinfo is not None
    assert result.created_at.tzinfo is not None
    await store.close()


async def test_get_memory_returns_none_for_missing() -> None:
    store = await _fresh_store()
    result = await store.get_memory("nonexistent_id")
    assert result is None
    await store.close()


# -- list_memories ------------------------------------------------------------


async def test_list_memories_filters_by_scope() -> None:
    store = await _fresh_store()
    agent_id = f"pg_scope_{datetime.now(UTC).timestamp()}"
    for mem_id, scope in [("m_g", "global"), ("m_py", "python"), ("m_ts", "typescript")]:
        await _insert(store, agent_id, mem_id, scope=scope)
    all_mems = await store.list_memories(agent_id, "user_y")
    assert len(all_mems) == 3
    scoped = await store.list_memories(agent_id, "user_y", scopes=["python"])
    ids = {m.id for m in scoped}
    # global always included; python included; typescript excluded.
    assert "m_g" in ids
    assert "m_py" in ids
    assert "m_ts" not in ids
    await store.close()


async def test_list_memories_active_only_default() -> None:
    store = await _fresh_store()
    agent_id = f"pg_active_{datetime.now(UTC).timestamp()}"
    await _insert(store, agent_id, "m_act")
    await store.deactivate_memory("m_act")
    active = await store.list_memories(agent_id, "user_y", active_only=True)
    assert not any(m.id == "m_act" for m in active)
    all_mems = await store.list_memories(agent_id, "user_y", active_only=False)
    assert any(m.id == "m_act" for m in all_mems)
    await store.close()


# -- deactivate_memory --------------------------------------------------------


async def test_deactivate_memory_returns_true_when_found() -> None:
    store = await _fresh_store()
    agent_id = f"pg_deact_{datetime.now(UTC).timestamp()}"
    await _insert(store, agent_id, "m_d1")
    found = await store.deactivate_memory("m_d1")
    assert found is True
    mem = await store.get_memory("m_d1")
    assert mem is not None
    assert mem.active is False
    await store.close()


async def test_deactivate_memory_returns_false_when_not_found() -> None:
    store = await _fresh_store()
    found = await store.deactivate_memory("nonexistent_mem")
    assert found is False
    await store.close()


async def test_deactivate_memory_is_idempotent() -> None:
    store = await _fresh_store()
    agent_id = f"pg_idemp_{datetime.now(UTC).timestamp()}"
    await _insert(store, agent_id, "m_idem")
    assert await store.deactivate_memory("m_idem") is True
    assert await store.deactivate_memory("m_idem") is False
    await store.close()


# -- Signals and links --------------------------------------------------------


async def test_signal_round_trip() -> None:
    store = await _fresh_store()
    agent_id = f"pg_sig_{datetime.now(UTC).timestamp()}"
    now = datetime.now(UTC)
    mem = _make_memory(agent_id, id="m_sl1", created_at=now, updated_at=now, valid_from=now)
    sig = _make_signal(agent_id, id="s_sl1", created_at=now)
    await store.insert_memory(mem)
    await store.insert_signal(sig)
    await store.link_signal_to_memory(memory_id="m_sl1", signal_id="s_sl1", weight=0.9)
    result = await store.get_creating_signal("m_sl1")
    assert result is not None
    assert result.id == "s_sl1"
    assert result.signal_type == SignalType.CORRECTION
    await store.close()


async def test_mark_signals_contradicted() -> None:
    store = await _fresh_store()
    agent_id = f"pg_contra_{datetime.now(UTC).timestamp()}"
    now = datetime.now(UTC)
    mem = _make_memory(agent_id, id="m_c1", created_at=now, updated_at=now, valid_from=now)
    sig = _make_signal(agent_id, id="s_c1", created_at=now)
    await store.insert_memory(mem)
    await store.insert_signal(sig)
    await store.link_signal_to_memory(memory_id="m_c1", signal_id="s_c1")
    await store.mark_signals_contradicted("m_c1")
    # Verify via get_creating_signal -- the signal should now be contradicted.
    result = await store.get_creating_signal("m_c1")
    assert result is not None
    assert result.contradicted is True
    await store.close()


# -- Scopes -------------------------------------------------------------------


async def test_scope_lifecycle() -> None:
    store = await _fresh_store()
    agent_id = f"pg_scopes_{datetime.now(UTC).timestamp()}"
    await store.insert_scope(agent_id, "python")
    await store.insert_scope(agent_id, "typescript")
    scopes = await store.list_scopes(agent_id)
    assert "python" in scopes
    assert "typescript" in scopes
    # insert_scope is idempotent (ON CONFLICT DO NOTHING).
    await store.insert_scope(agent_id, "python")
    assert await store.list_scopes(agent_id) == scopes
    await store.close()


async def test_clear_scopes() -> None:
    store = await _fresh_store()
    agent_id = f"pg_clrscope_{datetime.now(UTC).timestamp()}"
    await store.insert_scope(agent_id, "python")
    await store.insert_scope(agent_id, "typescript")
    await store.clear_scopes(agent_id)
    assert await store.list_scopes(agent_id) == []
    await store.close()


async def test_rename_scope() -> None:
    store = await _fresh_store()
    agent_id = f"pg_renscope_{datetime.now(UTC).timestamp()}"
    now = datetime.now(UTC)
    await store.insert_scope(agent_id, "py")
    await store.insert_memory(
        _make_memory(
            agent_id, id="m_ren", scope="py", created_at=now, updated_at=now, valid_from=now
        )
    )
    await store.rename_scope(agent_id, "py", "python")
    scopes = await store.list_scopes(agent_id)
    assert "python" in scopes
    assert "py" not in scopes
    mem = await store.get_memory("m_ren")
    assert mem is not None
    assert mem.scope == "python"
    await store.close()


async def test_merge_scopes() -> None:
    store = await _fresh_store()
    agent_id = f"pg_merge_{datetime.now(UTC).timestamp()}"
    now = datetime.now(UTC)
    await store.insert_scope(agent_id, "py")
    await store.insert_scope(agent_id, "python")
    await store.insert_memory(
        _make_memory(
            agent_id, id="m_m1", scope="py", created_at=now, updated_at=now, valid_from=now
        )
    )
    await store.merge_scopes(agent_id, "py", "python")
    scopes = await store.list_scopes(agent_id)
    assert "py" not in scopes
    assert "python" in scopes
    mem = await store.get_memory("m_m1")
    assert mem is not None
    assert mem.scope == "python"
    await store.close()


# -- Policy cache -------------------------------------------------------------


async def test_policy_cache_round_trip() -> None:
    store = await _fresh_store()
    now = datetime.now(UTC)
    await store.put_cached_policy(
        cache_key="test_key_pg",
        agent_id="ag1",
        user_id="u1",
        policy_text="Be concise.",
        compiled_at=now,
    )
    result = await store.get_cached_policy("test_key_pg")
    assert result is not None
    text, compiled_at = result
    assert text == "Be concise."
    assert compiled_at.tzinfo is not None
    await store.close()


async def test_put_cached_policy_upserts() -> None:
    store = await _fresh_store()
    now = datetime.now(UTC)
    await store.put_cached_policy(
        cache_key="upsert_key_pg",
        agent_id="ag1",
        user_id="u1",
        policy_text="First.",
        compiled_at=now,
    )
    await store.put_cached_policy(
        cache_key="upsert_key_pg",
        agent_id="ag1",
        user_id="u1",
        policy_text="Second.",
        compiled_at=now,
    )
    result = await store.get_cached_policy("upsert_key_pg")
    assert result is not None
    assert result[0] == "Second."
    await store.close()


async def test_invalidate_cached_policies() -> None:
    store = await _fresh_store()
    now = datetime.now(UTC)
    agent_id = f"pg_inv_{datetime.now(UTC).timestamp()}"
    await store.put_cached_policy(
        cache_key=f"{agent_id}_k1",
        agent_id=agent_id,
        user_id="u1",
        policy_text="Cached.",
        compiled_at=now,
    )
    await store.invalidate_cached_policies(agent_id, "u1")
    assert await store.get_cached_policy(f"{agent_id}_k1") is None
    await store.close()


# -- Agent config -------------------------------------------------------------


async def test_agent_config_round_trip() -> None:
    store = await _fresh_store()
    agent_id = f"pg_cfg_{datetime.now(UTC).timestamp()}"
    await store.put_agent_config(
        agent_id=agent_id,
        processing_mode="balanced",
        agent_description="Test agent",
        scopes=["python", "typescript"],
    )
    cfg = await store.get_agent_config(agent_id)
    assert cfg is not None
    assert cfg.processing_mode == "balanced"
    assert cfg.agent_description == "Test agent"
    assert cfg.scopes == ["python", "typescript"]
    await store.close()


async def test_agent_config_upserts() -> None:
    store = await _fresh_store()
    agent_id = f"pg_cfgupsert_{datetime.now(UTC).timestamp()}"
    await store.put_agent_config(
        agent_id=agent_id,
        processing_mode="frugal",
        agent_description=None,
        scopes=[],
    )
    await store.put_agent_config(
        agent_id=agent_id,
        processing_mode="eager",
        agent_description="Updated",
        scopes=["billing"],
    )
    cfg = await store.get_agent_config(agent_id)
    assert cfg is not None
    assert cfg.processing_mode == "eager"
    assert cfg.agent_description == "Updated"
    await store.close()


# -- Stability and recall -----------------------------------------------------


async def test_update_memory_stability_does_not_touch_updated_at() -> None:
    """Stability updates must not bust the policy cache key (which hashes updated_at)."""
    store = await _fresh_store()
    agent_id = f"pg_stab_{datetime.now(UTC).timestamp()}"
    now = datetime.now(UTC)
    await store.insert_memory(
        _make_memory(
            agent_id, id="m_stab", created_at=now, updated_at=now, valid_from=now, stability=5.0
        )
    )
    original = await store.get_memory("m_stab")
    assert original is not None
    await store.update_memory_stability("m_stab", 8.0)
    after = await store.get_memory("m_stab")
    assert after is not None
    assert after.stability == 8.0
    # updated_at must be unchanged.
    assert after.updated_at == original.updated_at
    await store.close()


async def test_increment_recall_count() -> None:
    store = await _fresh_store()
    agent_id = f"pg_recall_{datetime.now(UTC).timestamp()}"
    now = datetime.now(UTC)
    await store.insert_memory(
        _make_memory(agent_id, id="m_rc", created_at=now, updated_at=now, valid_from=now)
    )
    await store.increment_recall_count("m_rc")
    await store.increment_recall_count("m_rc")
    mem = await store.get_memory("m_rc")
    assert mem is not None
    assert mem.recall_count == 2
    assert mem.last_triggered is not None
    await store.close()


# -- FTS (search_fts) ---------------------------------------------------------


async def test_search_fts_returns_matching_memories() -> None:
    store = await _fresh_store()
    agent_id = f"pg_fts_{datetime.now(UTC).timestamp()}"
    now = datetime.now(UTC)
    await store.insert_memory(
        _make_memory(
            agent_id,
            id="m_fts1",
            content="Always write prose paragraphs instead of bullet points",
            created_at=now,
            updated_at=now,
            valid_from=now,
        )
    )
    await store.insert_memory(
        _make_memory(
            agent_id,
            id="m_fts2",
            content="Use strict typing in Python code",
            created_at=now,
            updated_at=now,
            valid_from=now,
        )
    )
    results = await store.search_fts("bullet points", {"m_fts1", "m_fts2"}, limit=10)
    ids = [mid for mid, _ in results]
    assert "m_fts1" in ids
    # m_fts2 has no relevance to bullet -- may or may not appear.
    if "m_fts2" in ids:
        # If it appears, m_fts1 should rank higher.
        assert ids.index("m_fts1") < ids.index("m_fts2")
    await store.close()


async def test_search_fts_respects_candidate_ids_filter() -> None:
    store = await _fresh_store()
    agent_id = f"pg_ftsflt_{datetime.now(UTC).timestamp()}"
    now = datetime.now(UTC)
    await store.insert_memory(
        _make_memory(
            agent_id,
            id="m_fts_x",
            content="bullet points style",
            created_at=now,
            updated_at=now,
            valid_from=now,
        )
    )
    # Exclude m_fts_x from candidate_ids -- should not appear in results.
    results = await store.search_fts("bullet points", {"some_other_id"}, limit=10)
    ids = [mid for mid, _ in results]
    assert "m_fts_x" not in ids
    await store.close()


async def test_search_fts_empty_inputs() -> None:
    store = await _fresh_store()
    assert await store.search_fts("", {"m1"}) == []
    assert await store.search_fts("query", set()) == []
    await store.close()


# -- Events -------------------------------------------------------------------


async def test_event_logger_and_list_events() -> None:
    store = await _fresh_store()
    agent_id = f"pg_evt_{datetime.now(UTC).timestamp()}"
    now = datetime.now(UTC)
    await store.insert_memory(
        _make_memory(agent_id, id="m_ev1", created_at=now, updated_at=now, valid_from=now)
    )
    logger = store.make_event_logger()
    await logger.log("m_ev1", "merge", {"superseded_by": "m_ev2"})
    await logger.log("m_ev1", "distinct")
    events = await store.list_events(agent_id, "user_y")
    assert len(events) == 2
    types = {e["event_type"] for e in events}
    assert "merge" in types
    assert "distinct" in types
    await store.close()


# -- Supersession -------------------------------------------------------------


async def test_get_memory_with_supersession() -> None:
    store = await _fresh_store()
    agent_id = f"pg_super_{datetime.now(UTC).timestamp()}"
    now = datetime.now(UTC)
    await store.insert_memory(
        _make_memory(agent_id, id="m_old", created_at=now, updated_at=now, valid_from=now)
    )
    await store.insert_memory(
        _make_memory(agent_id, id="m_new", created_at=now, updated_at=now, valid_from=now)
    )
    await store.deactivate_memory("m_old", superseded_by="m_new")
    successor, _ = await store.get_memory_with_supersession("m_old")
    assert successor is not None
    assert successor.id == "m_new"
    await store.close()


async def test_get_superseded_memories() -> None:
    store = await _fresh_store()
    agent_id = f"pg_supby_{datetime.now(UTC).timestamp()}"
    now = datetime.now(UTC)
    await store.insert_memory(
        _make_memory(agent_id, id="m_base", created_at=now, updated_at=now, valid_from=now)
    )
    await store.insert_memory(
        _make_memory(agent_id, id="m_sup", created_at=now, updated_at=now, valid_from=now)
    )
    await store.deactivate_memory("m_base", superseded_by="m_sup")
    superseded = await store.get_superseded_memories("m_sup")
    assert any(m.id == "m_base" for m in superseded)
    await store.close()


# -- Vector store (pgvector) --------------------------------------------------


async def _pgvector_skip_if_unavailable(store: PostgresMemoryStore) -> None:
    """Skip the test if the pgvector extension is not available."""
    try:
        await store.pool.execute("CREATE EXTENSION IF NOT EXISTS vector")
    except Exception:
        pytest.skip("pgvector extension not available")


async def test_vector_store_upsert_and_search() -> None:
    store = await _fresh_store()
    # Check if pgvector is available before attempting.
    await _pgvector_skip_if_unavailable(store)
    dim = 4
    vs = PostgresVectorStore(store.pool, dim=dim)
    await vs.init_schema()
    agent_id = f"pg_vec_{datetime.now(UTC).timestamp()}"
    now = datetime.now(UTC)
    await store.insert_memory(
        _make_memory(agent_id, id="m_v1", created_at=now, updated_at=now, valid_from=now)
    )
    await store.insert_memory(
        _make_memory(agent_id, id="m_v2", created_at=now, updated_at=now, valid_from=now)
    )
    await vs.upsert("m_v1", [1.0, 0.0, 0.0, 0.0])
    await vs.upsert("m_v2", [0.0, 1.0, 0.0, 0.0])
    results = await vs.search([1.0, 0.0, 0.0, 0.0], top_k=2)
    assert len(results) >= 1
    ids = [mid for mid, _ in results]
    assert ids[0] == "m_v1"
    await store.close()


async def test_vector_store_delete() -> None:
    store = await _fresh_store()
    await _pgvector_skip_if_unavailable(store)
    vs = PostgresVectorStore(store.pool, dim=4)
    await vs.init_schema()
    agent_id = f"pg_vdel_{datetime.now(UTC).timestamp()}"
    now = datetime.now(UTC)
    await store.insert_memory(
        _make_memory(agent_id, id="m_vd", created_at=now, updated_at=now, valid_from=now)
    )
    await vs.upsert("m_vd", [1.0, 0.0, 0.0, 0.0])
    await vs.delete("m_vd")
    results = await vs.search([1.0, 0.0, 0.0, 0.0], top_k=10)
    ids = [mid for mid, _ in results]
    assert "m_vd" not in ids
    await store.close()


async def test_vector_store_upsert_replaces_existing() -> None:
    store = await _fresh_store()
    await _pgvector_skip_if_unavailable(store)
    vs = PostgresVectorStore(store.pool, dim=4)
    await vs.init_schema()
    agent_id = f"pg_vupsert_{datetime.now(UTC).timestamp()}"
    now = datetime.now(UTC)
    await store.insert_memory(
        _make_memory(agent_id, id="m_vu", created_at=now, updated_at=now, valid_from=now)
    )
    # Insert then update.
    await vs.upsert("m_vu", [1.0, 0.0, 0.0, 0.0])
    await vs.upsert("m_vu", [0.0, 1.0, 0.0, 0.0])
    # Should not raise and should still have exactly one entry.
    results = await vs.search([0.0, 1.0, 0.0, 0.0], top_k=1)
    assert len(results) == 1
    assert results[0][0] == "m_vu"
    await store.close()
