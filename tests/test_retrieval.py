from typing import cast

import pytest
from helpers import _ConstantEmbedder, _InMemoryVectorStore, _make_imprint

from imprint import SQLiteMemoryStore


async def test_static_alpha_tuner_returns_configured_alpha() -> None:
    from imprint import StaticAlphaTuner

    tuner = StaticAlphaTuner(alpha=0.4)
    assert tuner.get_alpha() == 0.4
    assert tuner.get_alpha("any context") == 0.4
    await tuner.update(0.4, 1.0)  # no-op, should not raise


async def test_static_alpha_tuner_rejects_invalid_alpha() -> None:
    from imprint import StaticAlphaTuner

    with pytest.raises(ValueError):
        StaticAlphaTuner(alpha=1.5)


async def test_bandit_alpha_tuner_returns_valid_arm() -> None:
    from imprint import BanditAlphaTuner

    tuner = BanditAlphaTuner()
    for _ in range(20):
        alpha = tuner.get_alpha()
        assert alpha in [0.1, 0.3, 0.5, 0.7, 0.9]


async def test_bandit_alpha_tuner_updates_on_reward() -> None:
    from imprint import BanditAlphaTuner

    tuner = BanditAlphaTuner()
    initial_state = tuner.get_state()
    total_initial = sum(initial_state["s"]) + sum(initial_state["f"])

    await tuner.update(0.3, 1.0)
    state_after = tuner.get_state()
    total_after = sum(state_after["s"]) + sum(state_after["f"])

    assert total_after > total_initial


async def test_bandit_alpha_tuner_state_roundtrip() -> None:
    from imprint import BanditAlphaTuner

    tuner = BanditAlphaTuner()
    await tuner.update(0.3, 1.0)
    await tuner.update(0.7, 0.0)
    state = tuner.get_state()

    tuner2 = BanditAlphaTuner()
    tuner2.set_state(state)
    assert tuner2.get_state() == state


async def test_rrf_fuse_ranks_by_combined_score() -> None:
    from imprint.retrieval import rrf_fuse

    candidates = ["m1", "m2", "m3"]

    # m1 is rank 1 in both channels -- should win
    ranked = rrf_fuse(
        candidates=candidates,
        sparse_ranks={"m1": 1, "m2": 2, "m3": 3},
        dense_ranks={"m1": 1, "m2": 3, "m3": 2},
        alpha=0.3,
    )
    assert ranked[0] == "m1"


async def test_rrf_fuse_handles_missing_from_one_channel() -> None:
    from imprint.retrieval import rrf_fuse

    candidates = ["m1", "m2", "m3"]

    # m2 only in dense, m1 only in sparse, m3 in neither
    ranked = rrf_fuse(
        candidates=candidates,
        sparse_ranks={"m1": 1},
        dense_ranks={"m2": 1},
        alpha=0.5,
    )
    # m3 (in neither) should rank last
    assert ranked[-1] == "m3"


async def test_sanitize_fts_query_strips_special_chars() -> None:
    from imprint.retrieval import sanitize_fts_query

    assert sanitize_fts_query('user "prefers" (tabs)') == "user prefers tabs"
    assert sanitize_fts_query("hello world") == "hello world"
    assert sanitize_fts_query("  extra   spaces  ") == "extra spaces"


async def test_fts5_search_returns_relevant_memory() -> None:
    from datetime import UTC, datetime

    from imprint.types import Memory, MemorySource, MemoryType

    imprint, _, _, _, _, _, _ = _make_imprint(processing_mode="balanced", compile_text="ok")
    await imprint.connect()
    store = cast(SQLiteMemoryStore, imprint._store)

    now = datetime.now(UTC)
    await store.insert_memory(
        Memory(
            id="m_code",
            agent_id="agent",
            user_id="u",
            type=MemoryType.RULE,
            scope="global",
            content="always use type hints in Python functions",
            source=MemorySource.DETECTED,
            valid_from=now,
            created_at=now,
            updated_at=now,
        )
    )
    await store.insert_memory(
        Memory(
            id="m_style",
            agent_id="agent",
            user_id="u",
            type=MemoryType.RULE,
            scope="global",
            content="prefer concise responses",
            source=MemorySource.DETECTED,
            valid_from=now,
            created_at=now,
            updated_at=now,
        )
    )

    results = await store.search_fts("Python type hints", {"m_code", "m_style"})

    result_ids = [r[0] for r in results]
    assert "m_code" in result_ids
    if "m_style" in result_ids:
        assert result_ids.index("m_code") < result_ids.index("m_style")


async def test_hybrid_retrieve_uses_context_for_ranking() -> None:
    from datetime import UTC, datetime

    from imprint.types import Memory, MemorySource, MemoryType

    code_vec = [1.0, 0.0, 0.0]
    style_vec = [0.0, 1.0, 0.0]
    context_vec = [0.9, 0.1, 0.0]  # similar to code_vec

    vec_store = _InMemoryVectorStore()
    embedder = _ConstantEmbedder(context_vec)

    imprint, _, _, _, _, _, _ = _make_imprint(processing_mode="balanced", compile_text="ok")
    imprint._vector_store = vec_store
    imprint._embedder = embedder
    await imprint.connect()

    store = cast(SQLiteMemoryStore, imprint._store)
    now = datetime.now(UTC)

    await store.insert_memory(
        Memory(
            id="m_code",
            agent_id="agent",
            user_id="u",
            type=MemoryType.RULE,
            scope="global",
            content="use type hints always",
            source=MemorySource.DETECTED,
            valid_from=now,
            created_at=now,
            updated_at=now,
        )
    )
    await store.insert_memory(
        Memory(
            id="m_style",
            agent_id="agent",
            user_id="u",
            type=MemoryType.RULE,
            scope="global",
            content="be concise",
            source=MemorySource.DETECTED,
            valid_from=now,
            created_at=now,
            updated_at=now,
        )
    )
    await vec_store.upsert("m_code", code_vec)
    await vec_store.upsert("m_style", style_vec)

    policy = await imprint.get_policy(user_id="u", context="writing a Python function")

    mem_ids = [m.id for m in policy.memories]
    assert "m_code" in mem_ids
    assert mem_ids.index("m_code") < mem_ids.index("m_style")


async def test_hybrid_retrieve_falls_back_without_context() -> None:
    from datetime import UTC, datetime

    from imprint.types import Memory, MemorySource, MemoryType

    vec_store = _InMemoryVectorStore()
    embedder = _ConstantEmbedder([1.0, 0.0, 0.0])

    imprint, _, _, _, _, _, _ = _make_imprint(processing_mode="balanced", compile_text="ok")
    imprint._vector_store = vec_store
    imprint._embedder = embedder
    await imprint.connect()

    store = cast(SQLiteMemoryStore, imprint._store)
    now = datetime.now(UTC)
    for i in range(3):
        await store.insert_memory(
            Memory(
                id=f"m_{i}",
                agent_id="agent",
                user_id="u",
                type=MemoryType.RULE,
                scope="global",
                content=f"rule {i}",
                source=MemorySource.DETECTED,
                valid_from=now,
                created_at=now,
                updated_at=now,
            )
        )

    # no context -- should fetch all 3 regardless of vector store
    policy = await imprint.get_policy(user_id="u")
    assert len(policy.memories) == 3


async def test_bandit_alpha_tuner_reward_signal_from_consolidation() -> None:
    from datetime import UTC, datetime

    from imprint import BanditAlphaTuner
    from imprint.types import Memory, MemorySource, MemoryType

    same_vec = [1.0, 0.0, 0.0]
    vec_store = _InMemoryVectorStore()
    embedder = _ConstantEmbedder(same_vec)
    tuner = BanditAlphaTuner()

    existing_id = "mem_existing"
    imprint, _, _, _, _, _, _ = _make_imprint(
        processing_mode="balanced",
        derived_content="new rule",
        consolidation_decisions=[{"memory_id": existing_id, "action": "merge"}],
        compile_text="policy",
    )
    imprint._vector_store = vec_store
    imprint._embedder = embedder
    imprint._alpha_tuner = tuner
    await imprint.connect()

    store = cast(SQLiteMemoryStore, imprint._store)
    now = datetime.now(UTC)
    existing = Memory(
        id=existing_id,
        agent_id="agent",
        user_id="u",
        type=MemoryType.RULE,
        scope="global",
        content="old rule",
        source=MemorySource.DETECTED,
        valid_from=now,
        created_at=now,
        updated_at=now,
    )
    await store.insert_memory(existing)
    await vec_store.upsert(existing_id, same_vec)

    # get_policy with context populates _last_retrieval
    await imprint.get_policy(user_id="u", context="some context")
    initial_state = tuner.get_state()

    # observe triggers consolidation which computes reward and updates tuner
    await imprint.observe(user_id="u", agent_output="x", user_response="always be concise")
    await imprint.drain()

    final_state = tuner.get_state()
    total_initial = sum(initial_state["s"]) + sum(initial_state["f"])
    total_final = sum(final_state["s"]) + sum(final_state["f"])
    assert total_final > total_initial


async def test_deactivate_memory_removes_from_fts() -> None:
    from datetime import UTC, datetime

    from imprint.types import Memory, MemorySource, MemoryType

    store = SQLiteMemoryStore(":memory:")
    await store.connect()
    await store.init_schema()

    now = datetime.now(UTC)
    await store.insert_memory(
        Memory(
            id="m1",
            agent_id="a",
            user_id="u",
            type=MemoryType.RULE,
            scope="global",
            content="always use type hints in Python",
            source=MemorySource.DETECTED,
            valid_from=now,
            created_at=now,
            updated_at=now,
        )
    )

    results_before = await store.search_fts("type hints Python", {"m1"})
    assert any(r[0] == "m1" for r in results_before)

    await store.deactivate_memory("m1")
    results_after = await store.search_fts("type hints Python", {"m1"})
    assert not any(r[0] == "m1" for r in results_after)
    await store.close()


async def test_search_fts_empty_candidate_set_returns_empty() -> None:
    from datetime import UTC, datetime

    from imprint.types import Memory, MemorySource, MemoryType

    store = SQLiteMemoryStore(":memory:")
    await store.connect()
    await store.init_schema()

    now = datetime.now(UTC)
    await store.insert_memory(
        Memory(
            id="m1",
            agent_id="a",
            user_id="u",
            type=MemoryType.RULE,
            scope="global",
            content="always be concise",
            source=MemorySource.DETECTED,
            valid_from=now,
            created_at=now,
            updated_at=now,
        )
    )

    results = await store.search_fts("concise", set())
    assert results == []
    await store.close()


async def test_search_fts_filters_to_candidate_set() -> None:
    from datetime import UTC, datetime

    from imprint.types import Memory, MemorySource, MemoryType

    store = SQLiteMemoryStore(":memory:")
    await store.connect()
    await store.init_schema()

    now = datetime.now(UTC)
    for mid, content in [("m1", "always use type hints"), ("m2", "always be concise")]:
        await store.insert_memory(
            Memory(
                id=mid,
                agent_id="a",
                user_id="u",
                type=MemoryType.RULE,
                scope="global",
                content=content,
                source=MemorySource.DETECTED,
                valid_from=now,
                created_at=now,
                updated_at=now,
            )
        )

    # only m1 in candidate set -- m2 should not appear even if FTS matches
    results = await store.search_fts("always", {"m1"})
    assert all(r[0] == "m1" for r in results)
    await store.close()


async def test_hybrid_retrieval_does_not_activate_in_frugal_mode() -> None:
    from datetime import UTC, datetime

    from imprint.types import Memory, MemorySource, MemoryType

    vec_store = _InMemoryVectorStore()
    embedder = _ConstantEmbedder([1.0, 0.0, 0.0])

    imprint, _, _, _, _, _, _ = _make_imprint(processing_mode="frugal", compile_text="ok")
    imprint._vector_store = vec_store
    imprint._embedder = embedder
    await imprint.connect()

    store = cast(SQLiteMemoryStore, imprint._store)
    now = datetime.now(UTC)
    await store.insert_memory(
        Memory(
            id="m1",
            agent_id="agent",
            user_id="u",
            type=MemoryType.RULE,
            scope="global",
            content="always be concise",
            source=MemorySource.DETECTED,
            valid_from=now,
            created_at=now,
            updated_at=now,
        )
    )
    await vec_store.upsert("m1", [1.0, 0.0, 0.0])

    # frugal + context + vector store -- should NOT trigger hybrid
    assert "u" not in imprint._last_retrieval
    policy = await imprint.get_policy(user_id="u", context="some context")
    # no retrieval recorded means hybrid path was not taken
    assert "u" not in imprint._last_retrieval
    assert len(policy.memories) == 1
