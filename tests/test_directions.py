import os
from typing import cast

import pytest
from helpers import _make_imprint

from imprint import Imprint, SQLiteMemoryStore


async def test_observe_directions_empty_list_returns_empty() -> None:
    imprint, _, _, _, _, _, _ = _make_imprint()
    await imprint.connect()
    result = await imprint.observe_directions(user_id="u", directions=[])
    assert result == []


async def test_observe_directions_frugal_stores_as_rule() -> None:
    from imprint.types import MemoryType

    imprint, _, _, _, _, _, _ = _make_imprint(processing_mode="frugal")
    await imprint.connect()

    memories = await imprint.observe_directions(
        user_id="u",
        directions=["always respond in English"],
    )

    assert len(memories) == 1
    assert memories[0].type == MemoryType.RULE
    assert memories[0].content == "always respond in English"


async def test_observe_directions_frugal_skips_llm() -> None:
    imprint, _, _, derive_model, _, validate_model, _ = _make_imprint(processing_mode="frugal")
    await imprint.connect()

    await imprint.observe_directions(user_id="u", directions=["always be concise"])

    assert derive_model.last_model_request_parameters is None
    assert validate_model.last_model_request_parameters is None


async def test_observe_directions_balanced_calls_derive_llm() -> None:
    imprint, _, _, derive_model, _, _, _ = _make_imprint(
        processing_mode="balanced",
        derived_content="always respond in English",
    )
    await imprint.connect()

    await imprint.observe_directions(user_id="u", directions=["always respond in English"])

    assert derive_model.last_model_request_parameters is not None


async def test_observe_directions_balanced_skips_validation() -> None:
    imprint, _, _, _, _, validate_model, _ = _make_imprint(processing_mode="balanced")
    await imprint.connect()

    await imprint.observe_directions(user_id="u", directions=["always be concise"])

    assert validate_model.last_model_request_parameters is None


async def test_observe_directions_eager_runs_validation_first() -> None:
    imprint, _, _, _, _, validate_model, _ = _make_imprint(
        processing_mode="eager",
        validation_verdicts=[{"verdict": "directive"}],
    )
    await imprint.connect()

    await imprint.observe_directions(user_id="u", directions=["always be concise"])

    assert validate_model.last_model_request_parameters is not None


async def test_observe_directions_eager_filters_non_directives() -> None:
    imprint, _, _, _, _, _, _ = _make_imprint(
        processing_mode="eager",
        validation_verdicts=[
            {"verdict": "directive"},
            {"verdict": "non-directive"},
            {"verdict": "hedge"},
        ],
    )
    await imprint.connect()

    memories = await imprint.observe_directions(
        user_id="u",
        directions=["always be concise", "I think maybe shorter?", "sometimes brief"],
    )

    # only the directive passes through
    assert len(memories) == 1

    stored = await imprint._store.list_memories("agent", "u")
    assert len(stored) == 1


async def test_observe_directions_multiple_frugal_stores_all() -> None:
    imprint, _, _, _, _, _, _ = _make_imprint(processing_mode="frugal")
    await imprint.connect()

    directions = ["always use English", "never use bullet points", "keep it short"]
    memories = await imprint.observe_directions(user_id="u", directions=directions)

    assert len(memories) == 3
    stored = await imprint._store.list_memories("agent", "u")
    assert len(stored) == 3


async def test_observe_directions_source_defaults_to_user_edit() -> None:
    from imprint.types import MemorySource

    imprint, _, _, _, _, _, _ = _make_imprint(processing_mode="frugal")
    await imprint.connect()

    memories = await imprint.observe_directions(user_id="u", directions=["always be direct"])

    assert memories[0].source == MemorySource.USER_EDIT


async def test_observe_directions_respects_scope_hint() -> None:
    imprint, _, _, _, _, _, _ = _make_imprint(processing_mode="frugal", scopes=["code"])
    await imprint.connect()

    memories = await imprint.observe_directions(
        user_id="u",
        directions=["always use type hints"],
        scope="code",
    )

    assert memories[0].scope == "code"


async def test_observe_directions_invalidates_cache() -> None:
    imprint, _, _, _, _, _, _ = _make_imprint(processing_mode="frugal", compile_text="policy")
    await imprint.connect()

    await imprint.observe_directions(user_id="u", directions=["always be concise"])
    await imprint.get_policy(user_id="u")

    await imprint.observe_directions(user_id="u", directions=["never use bullets"])

    store = cast(SQLiteMemoryStore, imprint._store)
    cursor = await store.conn.execute(
        "SELECT COUNT(*) as n FROM compiled_policies WHERE agent_id = 'agent' AND user_id = 'u'"
    )
    row = await cursor.fetchone()
    assert row is not None and row["n"] == 0


@pytest.mark.live
async def test_observe_directions_balanced_live() -> None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        pytest.skip("ANTHROPIC_API_KEY not set")

    imprint = Imprint(
        agent_id="live_directions",
        store=":memory:",
        processing_mode="balanced",
    )
    await imprint.connect()

    directions = [
        "Always respond in English regardless of the language used in the question.",
        "Never use bullet points or numbered lists in your responses.",
        "Keep all responses under 150 words unless explicitly asked for more detail.",
    ]

    memories = await imprint.observe_directions(user_id="u", directions=directions)

    assert len(memories) == 3
    from imprint.types import MemoryType

    for m in memories:
        assert m.type == MemoryType.RULE
        assert m.agent_id == "live_directions"
        assert m.user_id == "u"

    stored = await imprint._store.list_memories("live_directions", "u")
    assert len(stored) == 3

    await imprint.close()


@pytest.mark.live
async def test_observe_directions_eager_filters_live() -> None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        pytest.skip("ANTHROPIC_API_KEY not set")

    imprint = Imprint(
        agent_id="live_directions_eager",
        store=":memory:",
        processing_mode="eager",
    )
    await imprint.connect()

    directions = [
        "Always respond in English.",
        "I sometimes wonder if responses could be shorter maybe?",
        "What is the capital of France?",
    ]

    memories = await imprint.observe_directions(user_id="u", directions=directions)

    # only the clear directive should survive validation
    assert len(memories) <= 2
    stored = await imprint._store.list_memories("live_directions_eager", "u")
    assert len(stored) == len(memories)

    await imprint.close()


async def test_observe_directions_all_filtered_by_eager_returns_empty() -> None:
    imprint, _, _, _, _, _, _ = _make_imprint(
        processing_mode="eager",
        validation_verdicts=[
            {"verdict": "non-directive"},
            {"verdict": "hedge"},
        ],
    )
    await imprint.connect()

    memories = await imprint.observe_directions(
        user_id="u",
        directions=["maybe shorter?", "I wonder sometimes"],
    )

    assert memories == []
    stored = await imprint._store.list_memories("agent", "u")
    assert stored == []


async def test_observe_directions_fetches_existing_once() -> None:
    """Existing memories are fetched once before the batch, not once per direction."""
    from typing import Any

    imprint, _, _, _, _, _, _ = _make_imprint(processing_mode="frugal")
    await imprint.connect()

    fetch_count = 0
    original_list = imprint._store.list_memories

    async def counting_list(*args: Any, **kwargs: Any) -> Any:
        nonlocal fetch_count
        fetch_count += 1
        return await original_list(*args, **kwargs)

    imprint._store.list_memories = counting_list  # type: ignore[method-assign]
    await imprint.observe_directions(
        user_id="u",
        directions=["direction one", "direction two", "direction three"],
    )

    assert fetch_count == 1, f"Expected 1 existing-memory fetch, got {fetch_count}"


async def test_observe_directions_batch_skips_llm_with_no_existing() -> None:
    """Batch consolidation must not call the LLM when there are no existing memories."""
    from typing import Any

    from pydantic_ai.models.test import TestModel

    imprint, _, _, _, _, _, _ = _make_imprint(processing_mode="balanced")
    await imprint.connect()

    batch_fired = False

    async def fail_if_called(*args: Any, **kwargs: Any) -> Any:
        nonlocal batch_fired
        batch_fired = True
        raise AssertionError("batch consolidate LLM should not fire with no existing memories")

    imprint._batch_consolidate_agent.run = fail_if_called  # type: ignore[method-assign]

    with imprint._derive_agent.override(
        model=TestModel(
            custom_output_args={"memory_type": "rule", "content": "be direct", "scope": "global"}
        )
    ):
        await imprint.observe_directions(user_id="u", directions=["be direct", "be concise"])

    assert not batch_fired


async def test_observe_directions_batch_consolidation_merges_existing() -> None:
    """Batch consolidation deactivates existing memories that merge with new ones."""
    from pydantic_ai.models.test import TestModel

    imprint, _, _, _, _, _, _ = _make_imprint(processing_mode="frugal")
    await imprint.connect()

    await imprint.observe_directions(user_id="u", directions=["use formal English"])
    existing = await imprint._store.list_memories("agent", "u")
    assert len(existing) == 1
    existing_id = existing[0].id

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
                    "decisions": [
                        {"candidate_index": 0, "memory_id": existing_id, "action": "merge"}
                    ]
                }
            )
        ),
    ):
        await imprint.observe_directions(user_id="u", directions=["always use British English"])
        await imprint.drain()

    all_memories = await imprint._store.list_memories("agent", "u", active_only=False)
    deactivated = [m for m in all_memories if not m.active and m.id == existing_id]
    assert len(deactivated) == 1, "Existing memory should have been deactivated by merge"

    active = await imprint._store.list_memories("agent", "u")
    assert len(active) == 1
    assert active[0].content == "always use British English"


async def test_observe_directions_batch_skips_duplicate_deactivation() -> None:
    """Two candidates both targeting the same existing memory: only the first merge applies."""
    from pydantic_ai.models.test import TestModel

    imprint, _, _, _, _, _, _ = _make_imprint(processing_mode="frugal")
    await imprint.connect()

    await imprint.observe_directions(user_id="u", directions=["use formal English"])
    existing = await imprint._store.list_memories("agent", "u")
    existing_id = existing[0].id

    imprint.processing_mode = "balanced"  # type: ignore[assignment]

    with (
        imprint._derive_agent.override(
            model=TestModel(
                custom_output_args={
                    "memory_type": "rule",
                    "content": "British English",
                    "scope": "global",
                }
            )
        ),
        imprint._batch_consolidate_agent.override(
            model=TestModel(
                custom_output_args={
                    "decisions": [
                        {"candidate_index": 0, "memory_id": existing_id, "action": "merge"},
                        {"candidate_index": 1, "memory_id": existing_id, "action": "merge"},
                    ]
                }
            )
        ),
    ):
        await imprint.observe_directions(
            user_id="u", directions=["British English", "formal British English"]
        )
        await imprint.drain()

    all_memories = await imprint._store.list_memories("agent", "u", active_only=False)
    deactivated = [m for m in all_memories if not m.active and m.id == existing_id]
    assert len(deactivated) == 1
