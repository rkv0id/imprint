"""Tests for TursoMemoryStore using a local file: URL.

These tests require libsql-client to be installed (imprint-mem[turso]).
They use file: URLs so no Turso cloud credentials are needed.

Live tests against a real Turso endpoint are gated by TURSO_DATABASE_URL
and TURSO_AUTH_TOKEN environment variables.
"""

import os
from datetime import UTC, datetime
from pathlib import Path

import pytest

libsql_client = pytest.importorskip("libsql_client", reason="imprint-mem[turso] not installed")

from imprint.turso import TursoMemoryStore  # noqa: E402
from imprint.types import Memory, MemorySource, MemoryType  # noqa: E402


def _now() -> datetime:
    return datetime.now(UTC)


def _memory(
    *,
    mid: str = "m1",
    scope: str = "global",
    content: str = "always be concise",
    agent_id: str = "agent",
    user_id: str = "u",
) -> Memory:
    now = _now()
    return Memory(
        id=mid,
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
async def store(tmp_path: Path):  # type: ignore[return]
    db_path = tmp_path / "test.db"
    s = TursoMemoryStore(f"file:{db_path}")
    await s.connect()
    await s.init_schema()
    yield s
    await s.close()


async def test_connect_and_close(tmp_path: Path) -> None:
    s = TursoMemoryStore(f"file:{tmp_path / 'test.db'}")
    await s.connect()
    assert s._client is not None
    await s.close()
    assert s._client is None


async def test_connect_twice_is_idempotent(store: TursoMemoryStore) -> None:
    first_client = store._client
    await store.connect()
    assert store._client is first_client


async def test_import_error_without_libsql(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import sys
    from unittest.mock import patch

    s = TursoMemoryStore(f"file:{tmp_path / 'test.db'}")
    with (
        patch.dict(sys.modules, {"libsql_client": None}),
        pytest.raises(ImportError, match="libsql-client is required"),
    ):
        await s.connect()


async def test_insert_and_list_memory(store: TursoMemoryStore) -> None:
    m = _memory()
    await store.insert_memory(m)

    memories = await store.list_memories("agent", "u")
    assert len(memories) == 1
    assert memories[0].id == "m1"
    assert memories[0].content == "always be concise"
    assert memories[0].scope == "global"


async def test_list_memories_scope_filter(store: TursoMemoryStore) -> None:
    await store.insert_memory(_memory(mid="m1", scope="global"))
    await store.insert_memory(_memory(mid="m2", scope="code"))
    await store.insert_memory(_memory(mid="m3", scope="billing"))

    code_only = await store.list_memories("agent", "u", scopes=["code"])
    ids = {m.id for m in code_only}
    assert "m2" in ids
    assert "m1" in ids  # global always included
    assert "m3" not in ids


async def test_deactivate_removes_from_list(store: TursoMemoryStore) -> None:
    await store.insert_memory(_memory())
    await store.deactivate_memory("m1")

    memories = await store.list_memories("agent", "u")
    assert len(memories) == 0


async def test_deactivate_removes_from_fts(store: TursoMemoryStore) -> None:
    await store.insert_memory(_memory(content="always be concise"))
    await store.deactivate_memory("m1")

    hits = await store.search_fts("concise", {"m1"})
    assert hits == []


async def test_search_fts_returns_hits(store: TursoMemoryStore) -> None:
    await store.insert_memory(_memory(mid="m1", content="prefer Python over Java"))
    await store.insert_memory(_memory(mid="m2", content="never use bullet points"))

    hits = await store.search_fts("Python", {"m1", "m2"})
    hit_ids = {h[0] for h in hits}
    assert "m1" in hit_ids
    assert "m2" not in hit_ids


async def test_search_fts_respects_candidate_filter(store: TursoMemoryStore) -> None:
    await store.insert_memory(_memory(mid="m1", content="prefer Python"))
    await store.insert_memory(_memory(mid="m2", content="prefer Python also"))

    # m2 not in candidate_ids -- should not appear
    hits = await store.search_fts("Python", {"m1"})
    hit_ids = {h[0] for h in hits}
    assert "m2" not in hit_ids


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
    text, ts = result
    assert text == "be concise"
    assert ts.isoformat() == at.isoformat()


async def test_cached_policy_miss(store: TursoMemoryStore) -> None:
    result = await store.get_cached_policy("nonexistent")
    assert result is None


async def test_invalidate_cached_policies(store: TursoMemoryStore) -> None:
    await store.put_cached_policy(
        cache_key="ck1",
        agent_id="agent",
        user_id="u",
        policy_text="be concise",
        compiled_at=_now(),
    )

    await store.invalidate_cached_policies("agent", "u")
    assert await store.get_cached_policy("ck1") is None


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


async def test_agent_config_missing(store: TursoMemoryStore) -> None:
    config = await store.get_agent_config("nonexistent")
    assert config is None


async def test_put_alpha_tuner_state(store: TursoMemoryStore) -> None:
    await store.put_agent_config(
        agent_id="agent",
        processing_mode="frugal",
        agent_description=None,
        scopes=[],
    )
    await store.put_alpha_tuner_state("agent", '{"s": [1, 1], "f": [1, 1]}')

    config = await store.get_agent_config("agent")
    assert config is not None
    assert config.alpha_tuner_state == '{"s": [1, 1], "f": [1, 1]}'


async def test_update_memory_stability(store: TursoMemoryStore) -> None:
    await store.insert_memory(_memory())
    await store.update_memory_stability("m1", 12.5)

    memories = await store.list_memories("agent", "u")
    assert memories[0].stability == 12.5


async def test_increment_recall_count(store: TursoMemoryStore) -> None:
    await store.insert_memory(_memory())
    await store.increment_recall_count("m1")
    await store.increment_recall_count("m1")

    memories = await store.list_memories("agent", "u")
    assert memories[0].recall_count == 2


async def test_turso_url_auto_detected_in_imprint() -> None:
    """TursoMemoryStore is instantiated automatically for turso:// URLs."""
    from imprint._core import _is_turso_url, _parse_turso_url

    assert _is_turso_url("libsql://db.turso.io")
    assert _is_turso_url("turso://db.turso.io")
    assert _is_turso_url("wss://localhost:8080")
    assert not _is_turso_url(":memory:")
    assert not _is_turso_url("sqlite:///path/to/db")

    url, token = _parse_turso_url("libsql://db.turso.io?auth_token=secret")
    assert url == "libsql://db.turso.io"
    assert token == "secret"

    url2, token2 = _parse_turso_url("turso://db.turso.io")
    assert url2 == "libsql://db.turso.io"
    assert token2 is None


@pytest.mark.live
async def test_turso_live_full_round_trip() -> None:
    """Full observe -> get_policy -> deactivate cycle against a real Turso/sqld server.

    Requires sqld or Docker running locally:
        docker run -p 8080:8080 ghcr.io/tursodatabase/libsql-server:latest
        OR
        sqld --http-listen-addr 127.0.0.1:8080 --db-path /tmp/imprint-test.sqld

    Then run:
        TURSO_DATABASE_URL=http://127.0.0.1:8080 just test-live
    """
    import sys

    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

    from imprint import Imprint

    db_url = os.environ.get("TURSO_DATABASE_URL")
    auth_token = os.environ.get("TURSO_AUTH_TOKEN")
    if not db_url:
        pytest.skip("TURSO_DATABASE_URL not set")

    store = TursoMemoryStore(db_url, auth_token=auth_token)
    await store.connect()
    await store.init_schema()

    # Use a unique run ID to avoid conflicts with stale data from prior runs
    import uuid as _uuid

    run_id = _uuid.uuid4().hex[:8]
    mid = f"live_{run_id}"
    agent_id = f"live_agent_{run_id}"
    user_id = f"live_u_{run_id}"

    # Insert a memory directly
    m = _memory(mid=mid, content="always respond in English", agent_id=agent_id, user_id=user_id)
    await store.insert_memory(m)

    # List it back
    memories = await store.list_memories(agent_id, user_id)
    assert any(mem.id == mid for mem in memories)

    # FTS search
    hits = await store.search_fts("English", {mid})
    assert any(h[0] == mid for h in hits)

    # Cache a policy
    at = _now()
    await store.put_cached_policy(
        cache_key=f"live_ck_{run_id}",
        agent_id=agent_id,
        user_id=user_id,
        policy_text="respond in English",
        compiled_at=at,
    )
    cached = await store.get_cached_policy(f"live_ck_{run_id}")
    assert cached is not None and cached[0] == "respond in English"

    # Invalidate cache
    await store.invalidate_cached_policies(agent_id, user_id)
    assert await store.get_cached_policy(f"live_ck_{run_id}") is None

    # Deactivate
    found = await store.deactivate_memory(mid)
    assert found is True
    memories_after = await store.list_memories(agent_id, user_id)
    assert not any(mem.id == mid for mem in memories_after)

    # Agent config roundtrip
    await store.put_agent_config(
        agent_id=agent_id,
        processing_mode="balanced",
        agent_description="test agent",
        scopes=["code", "billing"],
    )
    config = await store.get_agent_config(agent_id)
    assert config is not None
    assert config.processing_mode == "balanced"
    assert config.scopes == ["code", "billing"]

    # Full Imprint cycle via TursoMemoryStore
    imprint = Imprint(
        agent_id=agent_id,
        store=store,
        processing_mode="frugal",
    )
    await imprint.connect()

    await imprint.observe_directions(
        user_id=user_id,
        directions=["always be concise"],
    )

    policy = await imprint.get_policy(user_id=user_id)
    assert isinstance(policy.text, str)
    assert len(policy.memories) == 1

    await store.close()
