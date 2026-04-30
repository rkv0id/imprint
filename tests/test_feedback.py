from typing import cast

from helpers import _make_imprint

from imprint import SQLiteMemoryStore
from imprint.types import SignalType


async def test_get_policy_opens_feedback_loop() -> None:
    imprint, _, _, _, _, _, _ = _make_imprint(processing_mode="frugal", compile_text="ok")
    await imprint.connect()

    from datetime import UTC, datetime

    from imprint.types import Memory, MemorySource, MemoryType

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

    await imprint.get_policy(user_id="u")
    assert "u" in imprint._open_loops


async def test_observe_closes_feedback_loop_on_correction() -> None:

    imprint, _, _, _, _, _, _ = _make_imprint(
        processing_mode="frugal", signal_type=SignalType.CORRECTION, compile_text="ok"
    )
    await imprint.connect()

    from datetime import UTC, datetime

    from imprint.types import Memory, MemorySource, MemoryType

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

    await imprint.get_policy(user_id="u")
    assert "u" in imprint._open_loops

    await imprint.observe(user_id="u", agent_output="x", user_response="No, be verbose")
    assert "u" not in imprint._open_loops


async def test_observe_feedback_closes_loop_and_applies_outcome() -> None:
    imprint, _, _, _, _, _, _ = _make_imprint(
        processing_mode="frugal", compile_text="ok", derived_content="rule"
    )
    await imprint.connect()

    from datetime import UTC, datetime

    from imprint.types import Memory, MemorySource, MemoryType

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

    await imprint.get_policy(user_id="u")
    assert "u" in imprint._open_loops

    await imprint.observe_feedback(user_id="u", outcome=0.8)
    assert "u" not in imprint._open_loops
    await imprint.drain()


async def test_observe_feedback_no_op_without_open_loop() -> None:
    imprint, _, _, _, _, _, _ = _make_imprint(processing_mode="frugal")
    await imprint.connect()

    await imprint.observe_feedback(user_id="u", outcome=1.0)


async def test_session_id_creates_separate_loop() -> None:
    imprint, _, _, _, _, _, _ = _make_imprint(processing_mode="frugal", compile_text="ok")
    await imprint.connect()

    from datetime import UTC, datetime

    from imprint.types import Memory, MemorySource, MemoryType

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

    await imprint.get_policy(user_id="u", session_id="sess1")
    await imprint.get_policy(user_id="u", session_id="sess2")

    assert "u:sess1" in imprint._open_loops
    assert "u:sess2" in imprint._open_loops
    assert "u" not in imprint._open_loops


async def test_stale_loops_expire_lazily() -> None:
    from datetime import timedelta

    imprint, _, _, _, _, _, _ = _make_imprint(
        processing_mode="frugal", compile_text="ok", feedback_timeout=1
    )
    await imprint.connect()

    from datetime import UTC, datetime

    from imprint.types import Memory, MemorySource, MemoryType

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

    await imprint.get_policy(user_id="u")
    assert "u" in imprint._open_loops

    # Manually expire the loop by backdating it
    loop = imprint._open_loops["u"]
    imprint._open_loops["u"] = loop.__class__(
        user_id=loop.user_id,
        memory_ids_ordered=loop.memory_ids_ordered,
        memories=loop.memories,
        alpha_used=loop.alpha_used,
        context=loop.context,
        opened_at=loop.opened_at,
        expires_at=datetime.now(UTC) - timedelta(seconds=10),
    )

    # Any call triggers lazy expiry
    await imprint.observe(user_id="u", agent_output="x", user_response="ok")
    assert "u" not in imprint._open_loops


async def test_feedback_cycle_full_flow() -> None:
    """Integration: get_policy opens loop, observe correction closes it, bandit updates."""
    from imprint import BanditAlphaTuner

    tuner = BanditAlphaTuner()
    imprint, _, _, _, _, _, _ = _make_imprint(
        processing_mode="frugal",
        compile_text="ok",
        signal_type=SignalType.CORRECTION,
        derived_content="user prefers verbose responses",
    )
    imprint._alpha_tuner = tuner
    await imprint.connect()

    from datetime import UTC, datetime

    from imprint.types import Memory, MemorySource, MemoryType

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

    initial_bandit_total = sum(tuner.get_state()["s"]) + sum(tuner.get_state()["f"])

    # open
    await imprint.get_policy(user_id="u")
    assert "u" in imprint._open_loops

    # close with CORRECTION -- bandit should update
    await imprint.observe(
        user_id="u",
        agent_output="Here is a concise answer.",
        user_response="No, I want more detail please.",
    )
    assert "u" not in imprint._open_loops

    final_bandit_total = sum(tuner.get_state()["s"]) + sum(tuner.get_state()["f"])
    assert final_bandit_total > initial_bandit_total


async def test_feedback_cycle_reinforcement_updates_bandit() -> None:
    from imprint import BanditAlphaTuner

    tuner = BanditAlphaTuner()
    imprint, _, _, _, _, _, _ = _make_imprint(
        processing_mode="frugal",
        compile_text="ok",
        signal_type=SignalType.REINFORCEMENT,
    )
    imprint._alpha_tuner = tuner
    await imprint.connect()

    from datetime import UTC, datetime

    from imprint.types import Memory, MemorySource, MemoryType

    store = cast(SQLiteMemoryStore, imprint._store)
    now = datetime.now(UTC)
    await store.insert_memory(
        Memory(
            id="m1",
            agent_id="agent",
            user_id="u",
            type=MemoryType.RULE,
            scope="global",
            content="always be direct",
            source=MemorySource.DETECTED,
            valid_from=now,
            created_at=now,
            updated_at=now,
        )
    )

    initial_total = sum(tuner.get_state()["s"]) + sum(tuner.get_state()["f"])

    await imprint.get_policy(user_id="u")
    await imprint.observe(
        user_id="u",
        agent_output="Here is a direct answer.",
        user_response="Perfect, exactly what I needed.",
    )

    final_total = sum(tuner.get_state()["s"]) + sum(tuner.get_state()["f"])
    assert final_total > initial_total


async def test_feedback_cycle_neutral_signal_does_not_update_bandit() -> None:
    from imprint import BanditAlphaTuner

    tuner = BanditAlphaTuner()
    # balanced mode so the mock detect_model returning FACT is actually used
    imprint, _, _, _, _, _, _ = _make_imprint(
        processing_mode="balanced",
        compile_text="ok",
        signal_type=SignalType.FACT,
        derived_content="user is a software engineer",
    )
    imprint._alpha_tuner = tuner
    await imprint.connect()

    from datetime import UTC, datetime

    from imprint.types import Memory, MemorySource, MemoryType

    store = cast(SQLiteMemoryStore, imprint._store)
    now = datetime.now(UTC)
    await store.insert_memory(
        Memory(
            id="m1",
            agent_id="agent",
            user_id="u",
            type=MemoryType.RULE,
            scope="global",
            content="always be direct",
            source=MemorySource.DETECTED,
            valid_from=now,
            created_at=now,
            updated_at=now,
        )
    )

    initial_total = sum(tuner.get_state()["s"]) + sum(tuner.get_state()["f"])

    await imprint.get_policy(user_id="u")
    await imprint.observe(
        user_id="u",
        agent_output="What do you do?",
        user_response="I am a software engineer.",
    )
    assert "u" not in imprint._open_loops

    final_total = sum(tuner.get_state()["s"]) + sum(tuner.get_state()["f"])
    assert final_total == initial_total


async def test_no_loop_open_observe_does_not_affect_bandit() -> None:
    from imprint import BanditAlphaTuner

    tuner = BanditAlphaTuner()
    imprint, _, _, _, _, _, _ = _make_imprint(
        processing_mode="frugal",
        signal_type=SignalType.CORRECTION,
    )
    imprint._alpha_tuner = tuner
    await imprint.connect()

    initial_total = sum(tuner.get_state()["s"]) + sum(tuner.get_state()["f"])

    # observe without get_policy first -- no loop open, bandit unchanged
    await imprint.observe(
        user_id="u",
        agent_output="x",
        user_response="No that is wrong.",
    )

    final_total = sum(tuner.get_state()["s"]) + sum(tuner.get_state()["f"])
    assert final_total == initial_total


async def test_second_get_policy_replaces_open_loop() -> None:
    imprint, _, _, _, _, _, _ = _make_imprint(processing_mode="frugal", compile_text="ok")
    await imprint.connect()

    from datetime import UTC, datetime

    from imprint.types import Memory, MemorySource, MemoryType

    store = cast(SQLiteMemoryStore, imprint._store)
    now = datetime.now(UTC)
    await store.insert_memory(
        Memory(
            id="m1",
            agent_id="agent",
            user_id="u",
            type=MemoryType.RULE,
            scope="global",
            content="rule",
            source=MemorySource.DETECTED,
            valid_from=now,
            created_at=now,
            updated_at=now,
        )
    )

    await imprint.get_policy(user_id="u")
    first_loop = imprint._open_loops.get("u")
    assert first_loop is not None

    await imprint.get_policy(user_id="u")
    second_loop = imprint._open_loops.get("u")
    assert second_loop is not None
    assert second_loop is not first_loop


async def test_observe_with_no_signal_leaves_loop_open() -> None:
    imprint, _, _, _, _, _, _ = _make_imprint(
        processing_mode="frugal", compile_text="ok", signal_type=None
    )
    await imprint.connect()

    from datetime import UTC, datetime

    from imprint.types import Memory, MemorySource, MemoryType

    store = cast(SQLiteMemoryStore, imprint._store)
    now = datetime.now(UTC)
    await store.insert_memory(
        Memory(
            id="m1",
            agent_id="agent",
            user_id="u",
            type=MemoryType.RULE,
            scope="global",
            content="rule",
            source=MemorySource.DETECTED,
            valid_from=now,
            created_at=now,
            updated_at=now,
        )
    )

    await imprint.get_policy(user_id="u")
    assert "u" in imprint._open_loops

    # no heuristic pattern -> signal is None -> loop stays open
    await imprint.observe(user_id="u", agent_output="x", user_response="sure, thanks")
    assert "u" in imprint._open_loops


async def test_observe_feedback_with_session_id_targets_correct_loop() -> None:
    imprint, _, _, _, _, _, _ = _make_imprint(processing_mode="frugal", compile_text="ok")
    await imprint.connect()

    from datetime import UTC, datetime

    from imprint.types import Memory, MemorySource, MemoryType

    store = cast(SQLiteMemoryStore, imprint._store)
    now = datetime.now(UTC)
    await store.insert_memory(
        Memory(
            id="m1",
            agent_id="agent",
            user_id="u",
            type=MemoryType.RULE,
            scope="global",
            content="rule",
            source=MemorySource.DETECTED,
            valid_from=now,
            created_at=now,
            updated_at=now,
        )
    )

    await imprint.get_policy(user_id="u", session_id="s1")
    await imprint.get_policy(user_id="u", session_id="s2")
    assert "u:s1" in imprint._open_loops
    assert "u:s2" in imprint._open_loops

    await imprint.observe_feedback(user_id="u", outcome=1.0, session_id="s1")
    assert "u:s1" not in imprint._open_loops
    assert "u:s2" in imprint._open_loops
    await imprint.drain()


async def test_stale_loop_for_one_user_does_not_affect_another() -> None:
    from datetime import UTC, datetime, timedelta

    from imprint.types import Memory, MemorySource, MemoryType

    imprint, _, _, _, _, _, _ = _make_imprint(processing_mode="frugal", compile_text="ok")
    await imprint.connect()

    store = cast(SQLiteMemoryStore, imprint._store)
    now = datetime.now(UTC)
    for uid in ("u1", "u2"):
        await store.insert_memory(
            Memory(
                id=f"m_{uid}",
                agent_id="agent",
                user_id=uid,
                type=MemoryType.RULE,
                scope="global",
                content="rule",
                source=MemorySource.DETECTED,
                valid_from=now,
                created_at=now,
                updated_at=now,
            )
        )

    await imprint.get_policy(user_id="u1")
    await imprint.get_policy(user_id="u2")

    # expire u1's loop manually
    loop = imprint._open_loops["u1"]
    from imprint._core import _OpenLoop

    imprint._open_loops["u1"] = _OpenLoop(
        user_id=loop.user_id,
        memory_ids_ordered=loop.memory_ids_ordered,
        memories=loop.memories,
        alpha_used=loop.alpha_used,
        context=loop.context,
        opened_at=loop.opened_at,
        expires_at=datetime.now(UTC) - timedelta(seconds=1),
    )

    await imprint.observe_feedback(user_id="u1", outcome=0.5)
    assert "u1" not in imprint._open_loops
    assert "u2" in imprint._open_loops


async def test_observe_feedback_second_call_is_noop() -> None:
    from imprint import BanditAlphaTuner

    tuner = BanditAlphaTuner()
    imprint, _, _, _, _, _, _ = _make_imprint(processing_mode="frugal", compile_text="ok")
    imprint._alpha_tuner = tuner
    await imprint.connect()

    from datetime import UTC, datetime

    from imprint.types import Memory, MemorySource, MemoryType

    store = cast(SQLiteMemoryStore, imprint._store)
    now = datetime.now(UTC)
    await store.insert_memory(
        Memory(
            id="m1",
            agent_id="agent",
            user_id="u",
            type=MemoryType.RULE,
            scope="global",
            content="rule",
            source=MemorySource.DETECTED,
            valid_from=now,
            created_at=now,
            updated_at=now,
        )
    )

    await imprint.get_policy(user_id="u")
    await imprint.observe_feedback(user_id="u", outcome=1.0)
    await imprint.drain()
    after_first = sum(tuner.get_state()["s"]) + sum(tuner.get_state()["f"])

    # second call: no loop open, no-op
    await imprint.observe_feedback(user_id="u", outcome=1.0)
    after_second = sum(tuner.get_state()["s"]) + sum(tuner.get_state()["f"])
    assert after_first == after_second


async def test_observe_feedback_boundary_outcomes_update_bandit() -> None:
    from imprint import BanditAlphaTuner

    for outcome in (-1.0, 1.0):
        tuner = BanditAlphaTuner()
        imprint, _, _, _, _, _, _ = _make_imprint(processing_mode="frugal", compile_text="ok")
        imprint._alpha_tuner = tuner
        await imprint.connect()

        from datetime import UTC, datetime

        from imprint.types import Memory, MemorySource, MemoryType

        store = cast(SQLiteMemoryStore, imprint._store)
        now = datetime.now(UTC)
        await store.insert_memory(
            Memory(
                id="m1",
                agent_id="agent",
                user_id="u",
                type=MemoryType.RULE,
                scope="global",
                content="rule",
                source=MemorySource.DETECTED,
                valid_from=now,
                created_at=now,
                updated_at=now,
            )
        )

        initial = sum(tuner.get_state()["s"]) + sum(tuner.get_state()["f"])
        await imprint.get_policy(user_id="u")
        await imprint.observe_feedback(user_id="u", outcome=outcome)
        await imprint.drain()
        final = sum(tuner.get_state()["s"]) + sum(tuner.get_state()["f"])
        assert final > initial, f"outcome={outcome} should update bandit"
        await imprint.close()
