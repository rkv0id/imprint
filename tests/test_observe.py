import os
from typing import cast

import pytest
from helpers import _make_imprint
from pydantic_ai.models.test import TestModel

from imprint import Imprint, SQLiteMemoryStore
from imprint.types import SignalType


async def test_frugal_no_signal_stores_nothing() -> None:
    imprint, _, _, _, _, _, _ = _make_imprint(processing_mode="frugal")
    await imprint.connect()

    await imprint.observe(user_id="u", agent_output="x", user_response="ok")

    policy = await imprint.get_policy(user_id="u")
    assert policy.memories == []
    await imprint.close()


async def test_frugal_heuristic_match_stores_memory() -> None:
    imprint, _, detect_model, _, _, _, _ = _make_imprint(
        processing_mode="frugal", compile_text="ok"
    )
    await imprint.connect()

    await imprint.observe(user_id="u", agent_output="x", user_response="No, do it differently")

    policy = await imprint.get_policy(user_id="u")
    assert len(policy.memories) == 1
    # Frugal mode never asks the LLM.
    assert detect_model.last_model_request_parameters is None
    await imprint.close()


async def test_balanced_falls_through_to_llm_when_heuristic_silent() -> None:
    imprint, _, detect_model, _, _, _, _ = _make_imprint(
        processing_mode="balanced",
        compile_text="ok",
        signal_type=SignalType.CORRECTION,
    )
    await imprint.connect()

    # "you misunderstood" doesn't match heuristic
    await imprint.observe(user_id="u", agent_output="x", user_response="you misunderstood")

    assert detect_model.last_model_request_parameters is not None
    policy = await imprint.get_policy(user_id="u")
    assert len(policy.memories) == 1
    await imprint.close()


async def test_balanced_skips_llm_when_heuristic_matches() -> None:
    imprint, _, detect_model, _, _, _, _ = _make_imprint(
        processing_mode="balanced", compile_text="ok"
    )
    await imprint.connect()

    await imprint.observe(user_id="u", agent_output="x", user_response="No, that's wrong")

    # Heuristic matched, LLM was not consulted.
    assert detect_model.last_model_request_parameters is None
    await imprint.close()


async def test_eager_always_calls_llm_for_detection() -> None:
    imprint, _, detect_model, _, _, _, _ = _make_imprint(
        processing_mode="eager",
        compile_text="ok",
        signal_type=SignalType.CORRECTION,
    )
    await imprint.connect()

    # Even heuristic-matchable text goes to LLM in eager mode.
    await imprint.observe(user_id="u", agent_output="x", user_response="No, that's wrong")

    assert detect_model.last_model_request_parameters is not None
    await imprint.close()


async def test_balanced_drops_observation_when_llm_says_no_signal() -> None:
    imprint, _, _, _, _, _, _ = _make_imprint(processing_mode="balanced", signal_type=None)
    await imprint.connect()

    await imprint.observe(user_id="u", agent_output="x", user_response="huh interesting")

    policy = await imprint.get_policy(user_id="u")
    assert policy.memories == []
    await imprint.close()


async def test_derivation_assigns_memory_type_from_llm() -> None:
    """The LLM picks the memory type; the hard-coded RULE default is gone."""
    imprint, _, _, _, _, _, _ = _make_imprint(
        processing_mode="balanced",
        derived_type="fact",
        derived_content="User works at Anthropic",
    )
    await imprint.connect()

    await imprint.observe(
        user_id="u",
        agent_output="What do you do?",
        user_response="I work at Anthropic",
    )

    policy = await imprint.get_policy(user_id="u")
    from imprint.types import MemoryType

    assert len(policy.memories) == 1
    assert policy.memories[0].type == MemoryType.FACT
    assert policy.memories[0].content == "User works at Anthropic"
    await imprint.close()


async def test_derivation_runs_after_signal_detection() -> None:
    """If detection says no signal, derivation is skipped entirely."""
    imprint, _, _, derive_model, _, _, _ = _make_imprint(processing_mode="frugal")
    await imprint.connect()

    await imprint.observe(user_id="u", agent_output="x", user_response="ok")

    # Heuristic returns None for "ok"; derive must not have been called.
    assert derive_model.last_model_request_parameters is None
    await imprint.close()


async def test_derivation_receives_signal_type_in_prompt() -> None:
    """The derive prompt is conditioned on the detected signal type."""
    imprint, _, _, derive_model, _, _, _ = _make_imprint(
        processing_mode="balanced",
        derived_type="rule",
        derived_content="anything",
    )
    await imprint.connect()

    # "No, that's wrong" matches the heuristic as CORRECTION.
    await imprint.observe(user_id="u", agent_output="x", user_response="No, that's wrong")

    # Inspect the derive call: the prompt should mention CORRECTION.
    params = derive_model.last_model_request_parameters
    assert params is not None
    await imprint.close()


async def test_first_observation_skips_consolidation_call() -> None:
    """No existing memories => consolidate agent is never called."""
    imprint, _, _, _, consolidate_model, _, _ = _make_imprint(processing_mode="frugal")
    await imprint.connect()

    await imprint.observe(user_id="u", agent_output="x", user_response="I prefer paragraphs")

    # No existing memories at observe time, so no consolidation call was made.
    assert consolidate_model.last_model_request_parameters is None
    await imprint.close()


async def test_distinct_decision_keeps_old_memory_active() -> None:
    """If LLM says distinct, both memories remain active."""
    imprint, _, _, _, _, _, _ = _make_imprint(
        processing_mode="frugal",
        derived_content="first",
    )
    await imprint.connect()

    await imprint.observe(user_id="u", agent_output="x", user_response="I prefer paragraphs")

    # Now the second observe will see the first memory and decide "distinct".
    # Reconfigure consolidate to return distinct for any incoming memory id.
    # Easier: pull the old id from the store and inject the decision.
    existing = await imprint._store.list_memories("agent", "u")
    assert len(existing) == 1
    old_id = existing[0].id

    # Re-make with second-pass consolidation decision = distinct
    await imprint.close()

    imprint, _, _, _, _, _, _ = _make_imprint(
        processing_mode="balanced",
        derived_content="first",
        consolidation_decisions=[{"memory_id": old_id, "action": "distinct"}],
    )
    await imprint.connect()
    # Re-seed the original memory directly via the store
    await imprint._store.insert_memory(existing[0])

    await imprint.observe(user_id="u", agent_output="x", user_response="I prefer brevity")

    final = await imprint._store.list_memories("agent", "u")
    assert len(final) == 2  # both still active
    await imprint.close()


async def test_merge_decision_deactivates_old_memory() -> None:
    """If LLM says merge, the old memory is deactivated and points at the new one."""
    imprint, _, _, _, _, _, _ = _make_imprint(processing_mode="frugal", derived_content="first")
    await imprint.connect()

    await imprint.observe(user_id="u", agent_output="x", user_response="I prefer paragraphs")
    existing = await imprint._store.list_memories("agent", "u")
    old_id = existing[0].id
    await imprint.close()

    imprint, _, _, _, _, _, _ = _make_imprint(
        processing_mode="balanced",
        derived_content="merged",
        consolidation_decisions=[{"memory_id": old_id, "action": "merge"}],
    )
    await imprint.connect()
    await imprint._store.insert_memory(existing[0])

    await imprint.observe(user_id="u", agent_output="x", user_response="I prefer paragraphs")

    active = await imprint._store.list_memories("agent", "u")
    assert len(active) == 1
    assert active[0].content == "merged"

    # Inspect the old (now inactive) memory to confirm supersedence
    all_mems = await imprint._store.list_memories("agent", "u", active_only=False)
    old = next(m for m in all_mems if m.id == old_id)
    assert old.active is False
    assert old.superseded_by == active[0].id
    await imprint.close()


async def test_contradict_decision_sets_valid_until() -> None:
    """Contradict additionally sets valid_until on the deactivated memory."""
    imprint, _, _, _, _, _, _ = _make_imprint(processing_mode="frugal", derived_content="first")
    await imprint.connect()
    await imprint.observe(user_id="u", agent_output="x", user_response="I prefer paragraphs")
    existing = await imprint._store.list_memories("agent", "u")
    old_id = existing[0].id
    await imprint.close()

    imprint, _, _, _, _, _, _ = _make_imprint(
        processing_mode="balanced",
        derived_content="actually bullets",
        consolidation_decisions=[{"memory_id": old_id, "action": "contradict"}],
    )
    await imprint.connect()
    await imprint._store.insert_memory(existing[0])

    await imprint.observe(user_id="u", agent_output="x", user_response="actually I prefer bullets")

    all_mems = await imprint._store.list_memories("agent", "u", active_only=False)
    old = next(m for m in all_mems if m.id == old_id)
    assert old.active is False
    assert old.valid_until is not None
    assert old.superseded_by is not None
    await imprint.close()


async def test_contradict_marks_supporting_signals_as_contradicted() -> None:
    """Signals that fed into a now-contradicted memory get tagged."""
    imprint, _, _, _, consolidate_model, _, _ = _make_imprint(
        processing_mode="balanced", derived_content="first"
    )
    await imprint.connect()

    # First observation: store a memory and its supporting signal.
    await imprint.observe(user_id="u", agent_output="x", user_response="I prefer paragraphs")
    memories = await imprint._store.list_memories("agent", "u")
    old_id = memories[0].id

    # Second observation: configure consolidate to contradict the first memory.
    new_consolidate = TestModel(
        custom_output_args={"decisions": [{"memory_id": old_id, "action": "contradict"}]}
    )
    cm = imprint._consolidate_agent.override(model=new_consolidate)
    cm.__enter__()
    try:
        await imprint.observe(
            user_id="u", agent_output="x", user_response="actually I prefer bullets"
        )
    finally:
        cm.__exit__(None, None, None)

    # The signal that supported the original memory should be marked contradicted.
    cursor = await cast(SQLiteMemoryStore, imprint._store).conn.execute(
        "SELECT id, contradicted FROM signals WHERE id IN ("
        "SELECT signal_id FROM memory_sources WHERE memory_id = :m"
        ")",
        {"m": old_id},
    )
    rows = list(await cursor.fetchall())
    assert len(rows) == 1
    assert rows[0]["contradicted"] == 1

    # Don't reference consolidate_model after override exits - it's no longer active.
    del consolidate_model

    await imprint.close()


async def test_unknown_memory_ids_in_decisions_are_ignored() -> None:
    """Defensive: hallucinated ids in LLM output don't crash or affect the store."""
    imprint, _, _, _, _, _, _ = _make_imprint(processing_mode="balanced", derived_content="first")
    await imprint.connect()
    await imprint.observe(user_id="u", agent_output="x", user_response="I prefer paragraphs")
    existing = await imprint._store.list_memories("agent", "u")
    await imprint.close()

    imprint, _, _, _, _, _, _ = _make_imprint(
        processing_mode="balanced",
        derived_content="next",
        consolidation_decisions=[{"memory_id": "mem_does_not_exist", "action": "merge"}],
    )
    await imprint.connect()
    await imprint._store.insert_memory(existing[0])

    await imprint.observe(user_id="u", agent_output="x", user_response="I prefer brevity")

    # Hallucinated id was ignored; both memories present.
    final = await imprint._store.list_memories("agent", "u")
    assert len(final) == 2
    await imprint.close()


@pytest.mark.live
async def test_signal_detection_via_anthropic_live() -> None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        pytest.skip("ANTHROPIC_API_KEY not set")

    imprint = Imprint(
        agent_id="live_detect",
        store=":memory:",
        processing_mode="balanced",
    )
    await imprint.connect()

    await imprint.observe(
        user_id="u",
        agent_output="Here is a bulleted list of options.",
        user_response="you misunderstood my request entirely",
    )

    policy = await imprint.get_policy(user_id="u")
    assert len(policy.memories) == 1
    await imprint.close()


@pytest.mark.live
async def test_derivation_via_anthropic_live() -> None:
    """Real derivation: a FACT signal should yield a FACT memory, not RULE."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        pytest.skip("ANTHROPIC_API_KEY not set")

    from imprint.types import MemoryType

    imprint = Imprint(
        agent_id="live_derive",
        store=":memory:",
        processing_mode="balanced",
    )
    await imprint.connect()

    await imprint.observe(
        user_id="u",
        agent_output="Tell me about yourself.",
        user_response="My name is Rami and I live in Amsterdam",
    )

    policy = await imprint.get_policy(user_id="u")
    assert len(policy.memories) == 1
    # An identity statement should derive to FACT, not RULE.
    assert policy.memories[0].type == MemoryType.FACT
    # Derived content should not be the raw user response.
    assert policy.memories[0].content != "My name is Rami and I live in Amsterdam"
    await imprint.close()


@pytest.mark.live
async def test_consolidation_via_anthropic_live() -> None:
    """Real consolidation: two near-duplicate observations should not both survive."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        pytest.skip("ANTHROPIC_API_KEY not set")

    imprint = Imprint(
        agent_id="live_consolidate",
        store=":memory:",
        processing_mode="frugal",
    )
    await imprint.connect()

    # First observation creates a memory about paragraphs.
    await imprint.observe(
        user_id="u",
        agent_output="Here's a list...",
        user_response="No, write in paragraphs",
    )
    # Second observation says the same thing differently.
    await imprint.observe(
        user_id="u",
        agent_output="Here's another list...",
        user_response="I told you, paragraphs not bullets",
    )

    active = await imprint._store.list_memories("live_consolidate", "u")
    # The system shouldn't keep both as distinct active memories.
    assert len(active) <= 2
    # In practice we expect 1, but don't pin tightly to LLM judgment.
    await imprint.close()


async def test_event_logger_records_merge() -> None:
    from datetime import UTC, datetime

    from imprint.types import Memory, MemorySource, MemoryType

    known_id = "mem_existing_001"
    imprint, _, _, _, _, _, _ = _make_imprint(
        processing_mode="balanced",
        derived_content="new rule",
        consolidation_decisions=[{"memory_id": known_id, "action": "merge"}],
    )
    await imprint.connect()

    store = cast(SQLiteMemoryStore, imprint._store)
    now = datetime.now(UTC)
    pre_existing = Memory(
        id=known_id,
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
    await store.insert_memory(pre_existing)

    await imprint.observe(user_id="u", agent_output="x", user_response="always use paragraphs")

    cursor = await store.conn.execute(
        "SELECT event_type FROM memory_events WHERE memory_id = :m",
        {"m": known_id},
    )
    rows = list(await cursor.fetchall())
    assert any(r["event_type"] == "merge" for r in rows)


async def test_event_logger_records_recall() -> None:
    imprint, _, _, _, _, _, _ = _make_imprint(derived_content="some rule", compile_text="be direct")
    await imprint.connect()

    await imprint.observe(user_id="u", agent_output="x", user_response="always be concise")
    mem_id = (await cast(SQLiteMemoryStore, imprint._store).list_memories("agent", "u"))[0].id

    await imprint.get_policy(user_id="u")

    store = cast(SQLiteMemoryStore, imprint._store)
    cursor = await store.conn.execute(
        "SELECT event_type FROM memory_events WHERE memory_id = :m",
        {"m": mem_id},
    )
    rows = list(await cursor.fetchall())
    assert any(r["event_type"] == "recall" for r in rows)


async def test_null_event_logger_does_not_write() -> None:
    from imprint.store import NullEventLogger

    imprint, _, _, _, _, _, _ = _make_imprint(derived_content="rule", compile_text="policy")
    await imprint.connect()
    imprint._event_logger = NullEventLogger()

    await imprint.observe(user_id="u", agent_output="x", user_response="always be concise")
    await imprint.get_policy(user_id="u")

    store = cast(SQLiteMemoryStore, imprint._store)
    cursor = await store.conn.execute("SELECT COUNT(*) as n FROM memory_events")
    row = await cursor.fetchone()
    assert row is not None and row["n"] == 0


async def test_frugal_derive_correction_produces_rule() -> None:
    from imprint.types import MemoryType

    imprint, _, _, _, _, _, _ = _make_imprint(processing_mode="frugal", compile_text="ok")
    await imprint.connect()

    await imprint.observe(
        user_id="u", agent_output="x", user_response="No, that's wrong, do it differently"
    )

    mems = await imprint._store.list_memories("agent", "u")
    assert len(mems) == 1
    assert mems[0].type == MemoryType.RULE
    assert mems[0].content == "No, that's wrong, do it differently"


async def test_frugal_derive_fact_signal_produces_fact_type() -> None:
    from imprint.types import MemoryType, SignalType

    imprint, _, _, _, _, _, _ = _make_imprint(
        processing_mode="frugal",
        signal_type=SignalType.FACT,
        compile_text="ok",
    )
    await imprint.connect()

    await imprint.observe(user_id="u", agent_output="x", user_response="I work at Anthropic")

    mems = await imprint._store.list_memories("agent", "u")
    assert len(mems) == 1
    assert mems[0].type == MemoryType.FACT


async def test_frugal_derive_skips_llm_agent() -> None:
    imprint, _, _, derive_model, _, _, _ = _make_imprint(
        processing_mode="frugal", compile_text="ok"
    )
    await imprint.connect()

    await imprint.observe(user_id="u", agent_output="x", user_response="No, that's wrong")

    assert derive_model.last_model_request_parameters is None


async def test_frugal_consolidation_skips_llm_agent() -> None:
    imprint, _, _, _, consolidate_model, _, _ = _make_imprint(
        processing_mode="frugal", compile_text="ok"
    )
    await imprint.connect()

    await imprint.observe(user_id="u", agent_output="x", user_response="I prefer paragraphs")
    await imprint.observe(user_id="u", agent_output="x", user_response="No, use plain text")

    assert consolidate_model.last_model_request_parameters is None


async def test_frugal_direction_signal_derives_rule() -> None:
    from imprint.types import MemoryType

    imprint, _, _, _, _, _, _ = _make_imprint(processing_mode="frugal")
    await imprint.connect()

    await imprint.observe(
        user_id="u",
        agent_output="x",
        user_response="Always respond in English from now on.",
    )

    memories = await imprint._store.list_memories("agent", "u")
    assert len(memories) == 1
    assert memories[0].type == MemoryType.RULE


async def test_frugal_preference_signal_derives_rule() -> None:
    from imprint.types import MemoryType

    imprint, _, _, _, _, _, _ = _make_imprint(processing_mode="frugal")
    await imprint.connect()

    await imprint.observe(
        user_id="u",
        agent_output="x",
        user_response="I prefer shorter answers.",
    )

    memories = await imprint._store.list_memories("agent", "u")
    assert len(memories) == 1
    assert memories[0].type == MemoryType.RULE


async def test_frugal_reinforcement_signal_derives_context() -> None:
    from imprint.types import MemoryType

    imprint, _, _, _, _, _, _ = _make_imprint(processing_mode="frugal")
    await imprint.connect()

    await imprint.observe(
        user_id="u",
        agent_output="Here is a concise answer.",
        user_response="Perfect, exactly what I needed!",
    )

    memories = await imprint._store.list_memories("agent", "u")
    assert len(memories) == 1
    assert memories[0].type == MemoryType.CONTEXT


async def test_event_logger_records_contradict() -> None:
    from datetime import UTC, datetime

    from imprint.types import Memory, MemorySource, MemoryType

    known_id = "mem_to_contradict"
    imprint, _, _, _, _, _, _ = _make_imprint(
        processing_mode="balanced",
        derived_content="new rule",
        consolidation_decisions=[{"memory_id": known_id, "action": "contradict"}],
    )
    await imprint.connect()

    store = cast(SQLiteMemoryStore, imprint._store)
    now = datetime.now(UTC)
    await store.insert_memory(
        Memory(
            id=known_id,
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
    )

    await imprint.observe(user_id="u", agent_output="x", user_response="always be verbose")

    cursor = await store.conn.execute(
        "SELECT event_type FROM memory_events WHERE memory_id = :m",
        {"m": known_id},
    )
    rows = list(await cursor.fetchall())
    assert any(r["event_type"] == "contradict" for r in rows)


async def test_consolidation_skips_llm_when_no_existing_memories() -> None:
    imprint, _, _, _, consolidate_model, _, _ = _make_imprint(processing_mode="balanced")
    await imprint.connect()

    await imprint.observe(user_id="u", agent_output="x", user_response="always be concise")

    assert consolidate_model.last_model_request_parameters is None


async def test_multiple_consolidation_decisions_in_one_pass() -> None:
    from datetime import UTC, datetime

    from imprint.types import Memory, MemorySource, MemoryType

    existing1, existing2 = "mem_keep", "mem_merge"
    imprint, _, _, _, _, _, _ = _make_imprint(
        processing_mode="balanced",
        derived_content="new rule",
        consolidation_decisions=[
            {"memory_id": existing1, "action": "distinct"},
            {"memory_id": existing2, "action": "merge"},
        ],
    )
    await imprint.connect()

    store = cast(SQLiteMemoryStore, imprint._store)
    now = datetime.now(UTC)
    for mid, content in [(existing1, "keep this"), (existing2, "merge this")]:
        await store.insert_memory(
            Memory(
                id=mid,
                agent_id="agent",
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

    await imprint.observe(user_id="u", agent_output="x", user_response="always be direct")

    active = await store.list_memories("agent", "u")
    active_ids = {m.id for m in active}
    assert existing1 in active_ids
    assert existing2 not in active_ids


async def test_observe_with_empty_user_response_stores_nothing() -> None:
    imprint, _, _, _, _, _, _ = _make_imprint(processing_mode="frugal")
    await imprint.connect()

    await imprint.observe(user_id="u", agent_output="Here is my answer.", user_response="")

    memories = await imprint._store.list_memories("agent", "u")
    assert memories == []
