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
