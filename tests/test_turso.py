"""Tests for TursoMemoryStore against a live sqld instance.

All tests require a running sqld server. Start one with:
    just turso-dev  -- starts sqld via Docker on :8080
    TURSO_DATABASE_URL=http://127.0.0.1:8080 just test-live

Tests are skipped when TURSO_DATABASE_URL is not set.

Requires: pip install imprint-mem[turso]
"""

import os
import uuid
from datetime import UTC, datetime

import pytest

httpx = pytest.importorskip("httpx", reason="imprint-mem[turso] not installed")

from imprint.turso import TursoMemoryStore  # noqa: E402
from imprint.types import Memory, MemorySource, MemoryType  # noqa: E402


def _sqld_url() -> str:
    url = os.environ.get("TURSO_DATABASE_URL", "")
    if not url:
        pytest.skip("TURSO_DATABASE_URL not set -- run just turso-dev first")
    return url


def _now() -> datetime:
    return datetime.now(UTC)


def _memory(
    *,
    mid: str | None = None,
    scope: str = "global",
    content: str = "always be concise",
    agent_id: str = "agent",
    user_id: str = "u",
) -> Memory:
    now = _now()
    return Memory(
        id=mid or f"m_{uuid.uuid4().hex[:8]}",
        agent_id=agent_id,
        user_id=user_id,
        type=MemoryType.RULE,
        scope=scope,
        content=content,
        source=MemorySource.DETECTED,
        valid_from=now,
        created_at=now,
        updated_at=now,
    )


@pytest.fixture
async def store() -> TursoMemoryStore:  # type: ignore[return]
    url = _sqld_url()
    auth_token = os.environ.get("TURSO_AUTH_TOKEN") or None
    s = TursoMemoryStore(url, auth_token=auth_token)
    await s.connect()
    await s.init_schema()
    yield s  # type: ignore[misc]
    await s.close()


@pytest.mark.live
async def test_connect_and_close() -> None:
    url = _sqld_url()
    s = TursoMemoryStore(url)
    await s.connect()
    assert s._client is not None
    await s.close()
    assert s._client is None


@pytest.mark.live
async def test_connect_twice_is_idempotent(store: TursoMemoryStore) -> None:
    first_client = store._client
    await store.connect()
    assert store._client is first_client


@pytest.mark.live
async def test_import_error_without_httpx() -> None:
    import sys
    from unittest.mock import patch

    s = TursoMemoryStore("http://127.0.0.1:8080")
    with (
        patch.dict(sys.modules, {"httpx": None}),
        pytest.raises(ImportError, match="httpx is required"),
    ):
        await s.connect()


@pytest.mark.live
async def test_insert_and_list_memory(store: TursoMemoryStore) -> None:
    m = _memory()
    await store.insert_memory(m)

    memories = await store.list_memories("agent", "u")
    assert any(x.id == m.id for x in memories)
    found = next(x for x in memories if x.id == m.id)
    assert found.content == "always be concise"
    assert found.scope == "global"


@pytest.mark.live
async def test_list_memories_scope_filter(store: TursoMemoryStore) -> None:
    m1 = _memory(scope="global")
    m2 = _memory(scope="work")
    m3 = _memory(scope="personal")
    for m in (m1, m2, m3):
        await store.insert_memory(m)

    work_only = await store.list_memories("agent", "u", scopes=["work"])
    ids = {x.id for x in work_only}
    assert m1.id in ids  # global always included
    assert m2.id in ids  # work in filter
    assert m3.id not in ids


@pytest.mark.live
async def test_deactivate_removes_from_list(store: TursoMemoryStore) -> None:
    m = _memory()
    await store.insert_memory(m)
    before = await store.list_memories("agent", "u")
    assert any(x.id == m.id for x in before)

    found = await store.deactivate_memory(m.id)
    assert found is True

    after = await store.list_memories("agent", "u")
    assert not any(x.id == m.id for x in after)


@pytest.mark.live
async def test_deactivate_removes_from_fts(store: TursoMemoryStore) -> None:
    m = _memory(content="always be concise")
    await store.insert_memory(m)

    hits_before = await store.search_fts("concise", {m.id})
    assert any(mid == m.id for mid, _ in hits_before)

    await store.deactivate_memory(m.id)

    hits_after = await store.search_fts("concise", {m.id})
    assert not any(mid == m.id for mid, _ in hits_after)


@pytest.mark.live
async def test_search_fts_returns_hits(store: TursoMemoryStore) -> None:
    m1 = _memory(content="prefer Python over Java")
    m2 = _memory(content="always use type hints")
    for m in (m1, m2):
        await store.insert_memory(m)

    hits = await store.search_fts("Python", {m1.id, m2.id})
    hit_ids = {mid for mid, _ in hits}
    assert m1.id in hit_ids
    assert m2.id not in hit_ids


@pytest.mark.live
async def test_search_fts_respects_candidate_filter(store: TursoMemoryStore) -> None:
    m1 = _memory(content="prefer Python")
    m2 = _memory(content="prefer Python as well")
    for m in (m1, m2):
        await store.insert_memory(m)

    hits = await store.search_fts("Python", {m1.id})
    hit_ids = {mid for mid, _ in hits}
    assert m1.id in hit_ids
    assert m2.id not in hit_ids


@pytest.mark.live
async def test_cached_policy_roundtrip(store: TursoMemoryStore) -> None:
    at = _now()
    await store.put_cached_policy(
        cache_key="ck1",
        agent_id="agent",
        user_id="u",
        policy_text="be concise",
        compiled_at=at,
    )
    result = await store.get_cached_policy("ck1")
    assert result is not None
    text, _ = result
    assert text == "be concise"


@pytest.mark.live
async def test_cached_policy_miss(store: TursoMemoryStore) -> None:
    result = await store.get_cached_policy("nonexistent_key_xyz")
    assert result is None


@pytest.mark.live
async def test_invalidate_cached_policies(store: TursoMemoryStore) -> None:
    await store.put_cached_policy(
        cache_key="ck_inv",
        agent_id="agent",
        user_id="u",
        policy_text="be concise",
        compiled_at=_now(),
    )
    assert await store.get_cached_policy("ck_inv") is not None
    await store.invalidate_cached_policies("agent", "u")
    assert await store.get_cached_policy("ck_inv") is None


@pytest.mark.live
async def test_agent_config_roundtrip(store: TursoMemoryStore) -> None:
    await store.put_agent_config(
        agent_id="agent",
        processing_mode="balanced",
        agent_description="A helpful assistant",
        scopes=["code", "billing"],
    )
    config = await store.get_agent_config("agent")
    assert config is not None
    assert config.processing_mode == "balanced"
    assert config.agent_description == "A helpful assistant"
    assert config.scopes == ["code", "billing"]


@pytest.mark.live
async def test_agent_config_missing(store: TursoMemoryStore) -> None:
    config = await store.get_agent_config("nonexistent_agent_xyz")
    assert config is None


@pytest.mark.live
async def test_put_alpha_tuner_state(store: TursoMemoryStore) -> None:
    await store.put_agent_config(
        agent_id="agent",
        processing_mode="frugal",
        agent_description=None,
        scopes=[],
    )
    await store.put_alpha_tuner_state("agent", '{"s":[1,1],"f":[1,1]}')
    config = await store.get_agent_config("agent")
    assert config is not None
    assert config.alpha_tuner_state == '{"s":[1,1],"f":[1,1]}'


@pytest.mark.live
async def test_update_memory_stability(store: TursoMemoryStore) -> None:
    m = _memory()
    await store.insert_memory(m)
    await store.update_memory_stability(m.id, 9.5)
    memories = await store.list_memories("agent", "u")
    found = next((x for x in memories if x.id == m.id), None)
    assert found is not None
    assert abs(found.stability - 9.5) < 0.01


@pytest.mark.live
async def test_increment_recall_count(store: TursoMemoryStore) -> None:
    m = _memory()
    await store.insert_memory(m)
    await store.increment_recall_count(m.id)
    await store.increment_recall_count(m.id)
    memories = await store.list_memories("agent", "u")
    found = next((x for x in memories if x.id == m.id), None)
    assert found is not None
    assert found.recall_count >= 2


@pytest.mark.live
async def test_turso_live_full_round_trip() -> None:
    """Full observe -> get_policy -> deactivate cycle against a real sqld server.

    Requires sqld or Docker running locally:
        docker run -p 8080:8080 ghcr.io/tursodatabase/libsql-server:latest

    Then run:
        TURSO_DATABASE_URL=http://127.0.0.1:8080 just test-live
    """
    from imprint import Imprint

    db_url = os.environ.get("TURSO_DATABASE_URL")
    auth_token = os.environ.get("TURSO_AUTH_TOKEN") or None
    if not db_url:
        pytest.skip("TURSO_DATABASE_URL not set")

    store = TursoMemoryStore(db_url, auth_token=auth_token)
    imprint = Imprint(
        agent_id="live_turso_test",
        store=store,
        processing_mode="frugal",
    )
    await imprint.connect()

    await imprint.observe_directions(
        user_id="live_u",
        directions=["always respond in English"],
    )
    policy = await imprint.get_policy(user_id="live_u")
    assert len(policy.memories) >= 1

    memories = await imprint.list_memories("live_u")
    assert len(memories) >= 1
    for m in memories:
        await imprint.deactivate_memory("live_u", m.id)

    await imprint.close()
