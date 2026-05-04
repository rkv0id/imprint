"""Tests for the explicit MemoryLoop feedback model.

All tests use open_loop() / loop() / get_policy(loop=loop) / loop.close().
The old implicit API (observe_feedback, close_loop, _open_loops) is gone.
"""

from datetime import UTC, datetime, timedelta
from typing import cast

import pytest
from helpers import _make_imprint

from imprint import BanditAlphaTuner, Imprint, MemoryLoop, SQLiteMemoryStore
from imprint.types import Memory, MemorySource, MemoryType


def _insert_memory(store: SQLiteMemoryStore, *, memory_id: str = "m1") -> Memory:
    now = datetime.now(UTC)
    return Memory(
        id=memory_id,
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


async def test_open_loop_returns_memory_loop() -> None:
    imprint, _, _, _, _, _, _ = _make_imprint()
    await imprint.connect()
    loop = await imprint.open_loop(user_id="u")
    assert isinstance(loop, MemoryLoop)
    assert loop.user_id == "u"
    assert not loop.closed


async def test_open_loop_registers_in_active_loops() -> None:
    imprint, _, _, _, _, _, _ = _make_imprint()
    await imprint.connect()
    loop = await imprint.open_loop(user_id="u")
    assert loop in imprint._active_loops


async def test_get_policy_records_memories_on_loop() -> None:
    imprint, _, _, _, _, _, _ = _make_imprint(compile_text="ok")
    await imprint.connect()
    store = cast(SQLiteMemoryStore, imprint._store)
    await store.insert_memory(_insert_memory(store))

    loop = await imprint.open_loop(user_id="u")
    await imprint.get_policy(user_id="u", loop=loop)

    assert "m1" in loop.retrieved_ids
    assert len(loop.retrieved_memories) == 1


async def test_get_policy_without_loop_works() -> None:
    imprint, _, _, _, _, _, _ = _make_imprint(compile_text="ok")
    await imprint.connect()
    store = cast(SQLiteMemoryStore, imprint._store)
    await store.insert_memory(_insert_memory(store))

    policy = await imprint.get_policy(user_id="u")
    assert policy.text == "ok"


async def test_loop_close_with_positive_outcome_updates_bandit() -> None:
    tuner = BanditAlphaTuner()
    imprint, _, _, _, _, _, _ = _make_imprint(compile_text="ok")
    imprint._alpha_tuner = tuner
    await imprint.connect()
    store = cast(SQLiteMemoryStore, imprint._store)
    await store.insert_memory(_insert_memory(store))

    initial = sum(tuner.get_state()["s"]) + sum(tuner.get_state()["f"])

    loop = await imprint.open_loop(user_id="u")
    await imprint.get_policy(user_id="u", loop=loop)
    await loop.close(outcome=1.0)

    final = sum(tuner.get_state()["s"]) + sum(tuner.get_state()["f"])
    assert final > initial


async def test_loop_close_with_negative_outcome_updates_bandit() -> None:
    tuner = BanditAlphaTuner()
    imprint, _, _, _, _, _, _ = _make_imprint(compile_text="ok")
    imprint._alpha_tuner = tuner
    await imprint.connect()
    store = cast(SQLiteMemoryStore, imprint._store)
    await store.insert_memory(_insert_memory(store))

    initial = sum(tuner.get_state()["s"]) + sum(tuner.get_state()["f"])

    loop = await imprint.open_loop(user_id="u")
    await imprint.get_policy(user_id="u", loop=loop)
    await loop.close(outcome=-1.0)

    final = sum(tuner.get_state()["s"]) + sum(tuner.get_state()["f"])
    assert final > initial


async def test_loop_close_is_idempotent() -> None:
    tuner = BanditAlphaTuner()
    imprint, _, _, _, _, _, _ = _make_imprint(compile_text="ok")
    imprint._alpha_tuner = tuner
    await imprint.connect()
    store = cast(SQLiteMemoryStore, imprint._store)
    await store.insert_memory(_insert_memory(store))

    loop = await imprint.open_loop(user_id="u")
    await imprint.get_policy(user_id="u", loop=loop)
    await loop.close(outcome=1.0)
    after_first = sum(tuner.get_state()["s"]) + sum(tuner.get_state()["f"])

    await loop.close(outcome=1.0)
    after_second = sum(tuner.get_state()["s"]) + sum(tuner.get_state()["f"])

    assert after_first == after_second


async def test_set_outcome_then_close() -> None:
    tuner = BanditAlphaTuner()
    imprint, _, _, _, _, _, _ = _make_imprint(compile_text="ok")
    imprint._alpha_tuner = tuner
    await imprint.connect()
    store = cast(SQLiteMemoryStore, imprint._store)
    await store.insert_memory(_insert_memory(store))

    initial = sum(tuner.get_state()["s"]) + sum(tuner.get_state()["f"])

    loop = await imprint.open_loop(user_id="u")
    await imprint.get_policy(user_id="u", loop=loop)
    loop.set_outcome(0.9)
    await loop.close()

    final = sum(tuner.get_state()["s"]) + sum(tuner.get_state()["f"])
    assert final > initial


async def test_context_manager_closes_with_neutral_when_no_outcome_set() -> None:
    tuner = BanditAlphaTuner()
    imprint, _, _, _, _, _, _ = _make_imprint(compile_text="ok")
    imprint._alpha_tuner = tuner
    await imprint.connect()
    store = cast(SQLiteMemoryStore, imprint._store)
    await store.insert_memory(_insert_memory(store))

    initial = sum(tuner.get_state()["s"]) + sum(tuner.get_state()["f"])

    async with imprint.loop(user_id="u") as loop:
        await imprint.get_policy(user_id="u", loop=loop)
        # no set_outcome -- should close with 0.0 (neutral, non-negative -> bandit updates)

    final = sum(tuner.get_state()["s"]) + sum(tuner.get_state()["f"])
    assert final > initial
    assert loop.outcome == 0.0


async def test_context_manager_uses_set_outcome() -> None:
    tuner = BanditAlphaTuner()
    imprint, _, _, _, _, _, _ = _make_imprint(compile_text="ok")
    imprint._alpha_tuner = tuner
    await imprint.connect()
    store = cast(SQLiteMemoryStore, imprint._store)
    await store.insert_memory(_insert_memory(store))

    async with imprint.loop(user_id="u") as loop:
        await imprint.get_policy(user_id="u", loop=loop)
        loop.set_outcome(0.9)

    assert loop.outcome == 0.9
    assert loop.closed


async def test_context_manager_closes_on_exception() -> None:
    imprint, _, _, _, _, _, _ = _make_imprint(compile_text="ok")
    await imprint.connect()

    captured: list[MemoryLoop] = []
    try:
        async with imprint.loop(user_id="u") as loop:
            captured.append(loop)
            raise ValueError("boom")
    except ValueError:
        pass

    assert captured[0].closed


async def test_two_loops_same_user_coexist() -> None:
    imprint, _, _, _, _, _, _ = _make_imprint(compile_text="ok")
    await imprint.connect()

    loop1 = await imprint.open_loop(user_id="u", session_id="s1")
    loop2 = await imprint.open_loop(user_id="u", session_id="s2")

    assert loop1 in imprint._active_loops
    assert loop2 in imprint._active_loops
    assert loop1 is not loop2


async def test_close_removes_loop_from_active_set() -> None:
    imprint, _, _, _, _, _, _ = _make_imprint()
    await imprint.connect()

    loop = await imprint.open_loop(user_id="u")
    assert loop in imprint._active_loops
    await loop.close(outcome=0.0)
    assert loop not in imprint._active_loops


async def test_expired_loop_finalized_with_penalty_on_next_sweep() -> None:
    tuner = BanditAlphaTuner()
    imprint, _, _, _, _, _, _ = _make_imprint(compile_text="ok")
    imprint._alpha_tuner = tuner
    await imprint.connect()
    store = cast(SQLiteMemoryStore, imprint._store)
    await store.insert_memory(_insert_memory(store))

    loop = await imprint.open_loop(user_id="u", timeout=3600)
    await imprint.get_policy(user_id="u", loop=loop)

    # backdate the loop so it appears expired
    loop.opened_at = datetime.now(UTC) - timedelta(seconds=7200)

    initial = sum(tuner.get_state()["s"]) + sum(tuner.get_state()["f"])

    # next call triggers lazy sweep
    await imprint.get_policy(user_id="u")
    await imprint.drain()

    final = sum(tuner.get_state()["s"]) + sum(tuner.get_state()["f"])
    assert loop.closed
    assert loop.outcome is not None and abs(loop.outcome - (-0.15)) < 1e-9
    assert final > initial


async def test_expired_loop_removed_from_active_set_after_sweep() -> None:
    imprint, _, _, _, _, _, _ = _make_imprint()
    await imprint.connect()

    loop = await imprint.open_loop(user_id="u", timeout=3600)
    loop.opened_at = datetime.now(UTC) - timedelta(seconds=7200)

    await imprint.observe(user_id="u", agent_output="x", user_response="ok")
    await imprint.drain()

    assert loop not in imprint._active_loops


async def test_loop_close_without_get_policy_is_no_op_for_learning() -> None:
    imprint, _, _, _, _, _, _ = _make_imprint()
    await imprint.connect()

    # closing a loop that never had get_policy called on it should not raise
    loop = await imprint.open_loop(user_id="u")
    await loop.close(outcome=0.9)
    assert loop.closed


async def test_full_explicit_loop_cycle() -> None:
    """Integration: open_loop -> get_policy -> close -> bandit updates."""
    tuner = BanditAlphaTuner()
    imprint, _, _, _, _, _, _ = _make_imprint(compile_text="ok")
    imprint._alpha_tuner = tuner
    await imprint.connect()
    store = cast(SQLiteMemoryStore, imprint._store)
    await store.insert_memory(_insert_memory(store))

    initial = sum(tuner.get_state()["s"]) + sum(tuner.get_state()["f"])

    loop = await imprint.open_loop(user_id="u")
    policy = await imprint.get_policy(user_id="u", loop=loop)
    assert policy.text == "ok"
    assert "m1" in loop.retrieved_ids

    await loop.close(outcome=0.8)

    final = sum(tuner.get_state()["s"]) + sum(tuner.get_state()["f"])
    assert final > initial
    assert loop.closed
    assert loop not in imprint._active_loops


@pytest.mark.live
async def test_full_loop_cycle_live() -> None:
    import os

    if not os.environ.get("ANTHROPIC_API_KEY"):
        pytest.skip("ANTHROPIC_API_KEY not set")

    imprint = Imprint(agent_id="live_loop_test", store=":memory:", processing_mode="balanced")
    await imprint.connect()

    await imprint.observe_directions(
        user_id="u",
        directions=["Always respond in English regardless of input language."],
    )

    async with imprint.loop(user_id="u") as loop:
        policy = await imprint.get_policy(user_id="u", loop=loop)
        assert len(policy.memories) >= 1
        loop.set_outcome(0.9)

    assert loop.closed
    await imprint.close()
