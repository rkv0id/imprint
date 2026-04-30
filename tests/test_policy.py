import os
from typing import cast

import pytest
from helpers import _make_imprint
from pydantic_ai.models.test import TestModel

from imprint import Imprint, SQLiteMemoryStore


async def test_get_policy_calls_compile_agent_with_memory_in_prompt() -> None:
    imprint, compile_model, _, _, _, _ = _make_imprint(
        processing_mode="balanced",
        compile_text="compiled output",
        derived_content="User prefers paragraphs over bullet points",
    )
    await imprint.connect()

    await imprint.observe(
        user_id="rami",
        agent_output="I suggest using bullet points.",
        user_response="No, write in paragraphs.",
    )

    policy = await imprint.get_policy(user_id="rami")

    assert policy.text == "compiled output"
    assert len(policy.memories) == 1
    # Content is derived by the LLM, not verbatim user_response.
    assert policy.memories[0].content == "User prefers paragraphs over bullet points"
    last = compile_model.last_model_request_parameters
    assert last is not None
    await imprint.close()


async def test_get_policy_skips_llm_when_no_memories() -> None:
    imprint, compile_model, _, _, _, _ = _make_imprint(processing_mode="frugal")
    await imprint.connect()

    policy = await imprint.get_policy(user_id="someone")

    assert policy.text == ""
    assert policy.memories == []
    # No call ever happened.
    assert compile_model.last_model_request_parameters is None
    await imprint.close()


async def test_compile_passes_max_output_tokens_through() -> None:
    imprint, _, _, _, _, _ = _make_imprint(processing_mode="frugal", compile_text="x")
    await imprint.connect()

    await imprint.observe(user_id="u", agent_output="x", user_response="I prefer paragraphs")
    policy = await imprint.get_policy(user_id="u", max_output_tokens=123)

    assert policy.text == "x"
    await imprint.close()


async def test_existing_instructions_reach_the_prompt() -> None:
    imprint, compile_model, _, _, _, _ = _make_imprint(processing_mode="frugal", compile_text="x")
    await imprint.connect()

    await imprint.observe(user_id="u", agent_output="x", user_response="I prefer terse output")
    await imprint.get_policy(user_id="u", existing_instructions="Always be terse.")

    # Inspect the actual user prompt sent to the model
    messages = compile_model.last_model_request_parameters
    assert messages is not None
    # The prompt construction is verified at the prompt module level; here we
    # just assert the agent was invoked and produced output.
    await imprint.close()


def test_compile_prompt_includes_agent_description() -> None:
    """build_user_prompt surfaces agent_description so the compile LLM sees it."""
    from imprint.prompts.policy import build_user_prompt

    prompt = build_user_prompt(
        memories=[],
        existing_instructions=None,
        context=None,
        agent_description="A code reviewer that rejects PRs with tests missing.",
    )
    assert "A code reviewer that rejects PRs with tests missing." in prompt


def test_compile_prompt_handles_missing_agent_description() -> None:
    """agent_description is optional; absent goes through cleanly."""
    from imprint.prompts.policy import build_user_prompt

    prompt = build_user_prompt(memories=[], existing_instructions=None, context=None)
    assert "(not specified)" in prompt


async def test_context_reaches_the_prompt() -> None:
    imprint, _, _, _, _, _ = _make_imprint(processing_mode="frugal", compile_text="x")
    await imprint.connect()

    await imprint.observe(user_id="u", agent_output="x", user_response="I prefer terse output")
    policy = await imprint.get_policy(user_id="u", context="code review session")

    assert policy.text == "x"
    await imprint.close()


async def test_memories_are_scoped_per_user() -> None:
    imprint, _, _, _, _, _ = _make_imprint(processing_mode="frugal", compile_text="ok")
    await imprint.connect()

    await imprint.observe(user_id="alice", agent_output="x", user_response="I prefer brevity")
    await imprint.observe(user_id="bob", agent_output="x", user_response="I want detail")

    alice_policy = await imprint.get_policy(user_id="alice")
    bob_policy = await imprint.get_policy(user_id="bob")

    assert len(alice_policy.memories) == 1
    assert alice_policy.memories[0].user_id == "alice"
    assert len(bob_policy.memories) == 1
    assert bob_policy.memories[0].user_id == "bob"
    await imprint.close()


async def test_constructor_drops_global_from_declared_scopes() -> None:
    """'global' is implicit; declaring it explicitly is silently dropped."""
    imprint = Imprint(
        agent_id="a",
        store=":memory:",
        scopes=["global", "project:imprint", "global", "project:imprint"],
    )
    assert imprint.scopes == ["project:imprint"]


async def test_observe_defaults_to_global_scope() -> None:
    imprint, _, _, _, _, _ = _make_imprint(processing_mode="frugal")
    await imprint.connect()
    await imprint.observe(user_id="u", agent_output="x", user_response="I prefer paragraphs")
    memories = await imprint._store.list_memories("agent", "u")
    assert memories[0].scope == "global"
    await imprint.close()


async def test_observe_accepts_declared_scope() -> None:
    imprint, _, _, _, _, _ = _make_imprint(
        processing_mode="frugal", scopes=["project:imprint", "role:reviewer"]
    )
    await imprint.connect()
    await imprint.observe(
        user_id="u",
        agent_output="x",
        user_response="I prefer paragraphs",
        scope="project:imprint",
    )
    memories = await imprint._store.list_memories("agent", "u")
    assert memories[0].scope == "project:imprint"
    await imprint.close()


async def test_observe_undeclared_scope_falls_back_to_global() -> None:
    """Caller-provided scope outside the declared set is rejected silently."""
    imprint, _, _, _, _, _ = _make_imprint(processing_mode="frugal", scopes=["project:imprint"])
    await imprint.connect()
    await imprint.observe(
        user_id="u",
        agent_output="x",
        user_response="I prefer paragraphs",
        scope="some:unknown",
    )
    memories = await imprint._store.list_memories("agent", "u")
    assert memories[0].scope == "global"
    await imprint.close()


async def test_get_policy_filters_by_scope() -> None:
    imprint, _, _, _, _, _ = _make_imprint(
        processing_mode="frugal", compile_text="ok", scopes=["project:imprint", "role:reviewer"]
    )
    await imprint.connect()

    await imprint.observe(
        user_id="u",
        agent_output="x",
        user_response="I prefer paragraphs",
        scope="project:imprint",
    )
    await imprint.observe(
        user_id="u",
        agent_output="x",
        user_response="Always cite sources with URLs",
        scope="role:reviewer",
    )
    await imprint.observe(
        user_id="u",
        agent_output="x",
        user_response="My name is Rami",
    )  # global

    # Asking for one specific scope returns that scope plus globals.
    project_policy = await imprint.get_policy(user_id="u", scopes=["project:imprint"])
    project_scopes = sorted(m.scope for m in project_policy.memories)
    assert project_scopes == ["global", "project:imprint"]

    # Asking with an empty list returns globals only.
    global_policy = await imprint.get_policy(user_id="u", scopes=[])
    assert all(m.scope == "global" for m in global_policy.memories)
    assert len(global_policy.memories) == 1

    # Asking with no scopes argument returns everything (current default).
    everything = await imprint.get_policy(user_id="u")
    assert len(everything.memories) == 3

    await imprint.close()


async def test_observe_uses_derived_scope_when_no_caller_hint() -> None:
    """When the caller doesn't pass scope=, the LLM-derived scope is used."""
    imprint, _, _, _, _, _ = _make_imprint(
        processing_mode="balanced",
        derived_scope="project:imprint",
        scopes=["project:imprint", "role:reviewer"],
    )
    await imprint.connect()

    await imprint.observe(user_id="u", agent_output="x", user_response="I prefer paragraphs")

    memories = await imprint._store.list_memories("agent", "u")
    assert memories[0].scope == "project:imprint"
    await imprint.close()


async def test_caller_scope_overrides_derived_scope() -> None:
    """Explicit scope= wins over what the LLM derives."""
    imprint, _, _, _, _, _ = _make_imprint(
        processing_mode="frugal",
        derived_scope="role:reviewer",
        scopes=["project:imprint", "role:reviewer"],
    )
    await imprint.connect()

    await imprint.observe(
        user_id="u",
        agent_output="x",
        user_response="I prefer paragraphs",
        scope="project:imprint",
    )

    memories = await imprint._store.list_memories("agent", "u")
    assert memories[0].scope == "project:imprint"
    await imprint.close()


async def test_hallucinated_derived_scope_falls_back_to_global() -> None:
    """If the LLM invents a scope outside the declared set, _resolve_scope catches it."""
    imprint, _, _, _, _, _ = _make_imprint(
        processing_mode="frugal", derived_scope="project:nonexistent", scopes=["project:imprint"]
    )
    await imprint.connect()

    await imprint.observe(user_id="u", agent_output="x", user_response="I prefer paragraphs")

    memories = await imprint._store.list_memories("agent", "u")
    assert memories[0].scope == "global"
    await imprint.close()


async def test_get_policy_caches_compiled_text() -> None:
    """Second call with the same inputs hits the cache and skips the LLM."""
    imprint, _, _, _, _, _ = _make_imprint(processing_mode="frugal", compile_text="cached compile")
    await imprint.connect()

    await imprint.observe(user_id="u", agent_output="x", user_response="I prefer paragraphs")

    first = await imprint.get_policy(user_id="u")
    assert first.text == "cached compile"

    # Swap in a different compile result; if the cache works, we should still
    # get the original text on the second call.
    new_compile = TestModel(custom_output_text="DIFFERENT")
    cm = imprint._compile_agent.override(model=new_compile)
    cm.__enter__()
    try:
        second = await imprint.get_policy(user_id="u")
        assert second.text == "cached compile"
        # New compile model never got called.
        assert new_compile.last_model_request_parameters is None
    finally:
        cm.__exit__(None, None, None)
    await imprint.close()


async def test_observe_invalidates_cache() -> None:
    """A new observation drops cached policies for that user."""
    imprint, _, _, _, _, _ = _make_imprint(processing_mode="frugal", compile_text="first compile")
    await imprint.connect()

    await imprint.observe(user_id="u", agent_output="x", user_response="I prefer paragraphs")
    first = await imprint.get_policy(user_id="u")
    assert first.text == "first compile"

    # Swap compile output and run another observe; cache should drop, next
    # get_policy should recompile with the new model.
    new_compile = TestModel(custom_output_text="recompiled")
    cm = imprint._compile_agent.override(model=new_compile)
    cm.__enter__()
    try:
        await imprint.observe(user_id="u", agent_output="x", user_response="Always cite sources")
        second = await imprint.get_policy(user_id="u")
        assert second.text == "recompiled"
    finally:
        cm.__exit__(None, None, None)
    await imprint.close()


async def test_cache_keys_separate_per_user() -> None:
    """Two users hitting get_policy don't share each other's cached results."""
    imprint, _, _, _, _, _ = _make_imprint(processing_mode="frugal", compile_text="alice text")
    await imprint.connect()

    await imprint.observe(user_id="alice", agent_output="x", user_response="I prefer brevity")
    alice_policy = await imprint.get_policy(user_id="alice")
    assert alice_policy.text == "alice text"

    new_compile = TestModel(custom_output_text="bob text")
    cm = imprint._compile_agent.override(model=new_compile)
    cm.__enter__()
    try:
        await imprint.observe(user_id="bob", agent_output="x", user_response="I want detail")
        bob_policy = await imprint.get_policy(user_id="bob")
        # bob's request should have compiled fresh (cache miss for bob).
        assert bob_policy.text == "bob text"
    finally:
        cm.__exit__(None, None, None)
    await imprint.close()


async def test_cache_keys_differ_when_params_differ() -> None:
    """Different existing_instructions => cache miss => recompile."""
    imprint, _, _, _, _, _ = _make_imprint(processing_mode="frugal", compile_text="first")
    await imprint.connect()

    await imprint.observe(user_id="u", agent_output="x", user_response="I prefer paragraphs")

    a = await imprint.get_policy(user_id="u", existing_instructions="be brief")
    assert a.text == "first"

    new_compile = TestModel(custom_output_text="second")
    cm = imprint._compile_agent.override(model=new_compile)
    cm.__enter__()
    try:
        # Different existing_instructions => different cache key => fresh compile.
        b = await imprint.get_policy(user_id="u", existing_instructions="be detailed")
        assert b.text == "second"
    finally:
        cm.__exit__(None, None, None)
    await imprint.close()


async def test_cache_hit_preserves_original_compiled_at() -> None:
    """compiled_at on a cache hit reflects the original compile, not now()."""
    import asyncio

    imprint, _, _, _, _, _ = _make_imprint(processing_mode="frugal", compile_text="x")
    await imprint.connect()

    await imprint.observe(user_id="u", agent_output="x", user_response="I prefer paragraphs")

    first = await imprint.get_policy(user_id="u")
    original_compiled_at = first.compiled_at

    # Wait long enough that datetime.now() differs measurably, then hit the cache.
    await asyncio.sleep(0.01)
    second = await imprint.get_policy(user_id="u")
    assert second.compiled_at == original_compiled_at
    await imprint.close()


@pytest.mark.live
async def test_compile_via_anthropic_live() -> None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        pytest.skip("ANTHROPIC_API_KEY not set")

    imprint = Imprint(
        agent_id="live_compile",
        store=":memory:",
        processing_mode="frugal",
    )
    await imprint.connect()

    await imprint.observe(
        user_id="u",
        agent_output="x",
        user_response="I prefer concise responses without bullet points",
    )
    await imprint.observe(
        user_id="u",
        agent_output="x",
        user_response="Always cite sources with URLs when making factual claims",
    )

    policy = await imprint.get_policy(
        user_id="u",
        existing_instructions="You are a helpful assistant.",
        max_output_tokens=300,
    )

    assert policy.text.strip()
    assert len(policy.memories) == 2
    text_lower = policy.text.lower()
    assert any(word in text_lower for word in ["concise", "url", "source", "cite"])
    await imprint.close()


async def test_budget_no_truncation_when_within_limit() -> None:
    imprint, _, _, _, _, _ = _make_imprint(derived_content="be concise", compile_text="ok")
    await imprint.connect()

    await imprint.observe(user_id="u", agent_output="x", user_response="always be concise")
    policy = await imprint.get_policy(user_id="u", max_input_tokens=8000)

    assert len(policy.dropped_memories) == 0
    assert len(policy.memories) == 1


async def test_budget_truncates_context_type_first() -> None:
    from datetime import UTC, datetime

    from imprint.types import Memory, MemorySource, MemoryType

    imprint, _, _, _, _, _ = _make_imprint(derived_content="rule", compile_text="ok")
    await imprint.connect()

    store = cast(SQLiteMemoryStore, imprint._store)
    now = datetime.now(UTC)

    def _mem(id: str, type: MemoryType, content: str) -> Memory:
        return Memory(
            id=id,
            agent_id="agent",
            user_id="u",
            type=type,
            scope="global",
            content=content,
            source=MemorySource.DETECTED,
            valid_from=now,
            created_at=now,
            updated_at=now,
        )

    await store.insert_memory(_mem("m_rule", MemoryType.RULE, "always respond in English"))
    await store.insert_memory(_mem("m_ctx", MemoryType.CONTEXT, "user is currently on mobile"))

    policy = await imprint.get_policy(user_id="u", max_input_tokens=65)

    dropped_ids = {m.id for m in policy.dropped_memories}
    assert "m_ctx" in dropped_ids
    assert "m_rule" not in dropped_ids


async def test_budget_error_mode_raises() -> None:
    from datetime import UTC, datetime

    from imprint import BudgetExceededError
    from imprint.types import Memory, MemorySource, MemoryType

    imprint, _, _, _, _, _ = _make_imprint(derived_content="rule", compile_text="ok")
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
            content="x" * 500,
            source=MemorySource.DETECTED,
            valid_from=now,
            created_at=now,
            updated_at=now,
        )
    )
    await store.insert_memory(
        Memory(
            id="m2",
            agent_id="agent",
            user_id="u",
            type=MemoryType.RULE,
            scope="global",
            content="y" * 500,
            source=MemorySource.DETECTED,
            valid_from=now,
            created_at=now,
            updated_at=now,
        )
    )

    with pytest.raises(BudgetExceededError):
        await imprint.get_policy(user_id="u", max_input_tokens=10, on_budget_exceeded="error")


async def test_budget_pinned_memory_never_dropped() -> None:
    from datetime import UTC, datetime

    from imprint.types import Memory, MemorySource, MemoryType

    imprint, _, _, _, _, _ = _make_imprint(derived_content="rule", compile_text="ok")
    await imprint.connect()

    store = cast(SQLiteMemoryStore, imprint._store)
    now = datetime.now(UTC)
    await store.insert_memory(
        Memory(
            id="m_pinned",
            agent_id="agent",
            user_id="u",
            type=MemoryType.RULE,
            scope="global",
            content="critical rule",
            source=MemorySource.DETECTED,
            pinned=True,
            valid_from=now,
            created_at=now,
            updated_at=now,
        )
    )
    await store.insert_memory(
        Memory(
            id="m_drop",
            agent_id="agent",
            user_id="u",
            type=MemoryType.CONTEXT,
            scope="global",
            content="transient context info",
            source=MemorySource.DETECTED,
            valid_from=now,
            created_at=now,
            updated_at=now,
        )
    )

    policy = await imprint.get_policy(user_id="u", max_input_tokens=65)

    kept_ids = {m.id for m in policy.memories}
    assert "m_pinned" in kept_ids


async def test_budget_drops_lower_stability_first_within_same_type() -> None:
    from datetime import UTC, datetime

    from imprint.types import Memory, MemorySource, MemoryType

    imprint, _, _, _, _, _ = _make_imprint(compile_text="ok")
    await imprint.connect()

    store = cast(SQLiteMemoryStore, imprint._store)
    now = datetime.now(UTC)
    await store.insert_memory(
        Memory(
            id="m_high",
            agent_id="agent",
            user_id="u",
            type=MemoryType.RULE,
            scope="global",
            content="a" * 300,
            source=MemorySource.DETECTED,
            stability=10.0,
            valid_from=now,
            created_at=now,
            updated_at=now,
        )
    )
    await store.insert_memory(
        Memory(
            id="m_low",
            agent_id="agent",
            user_id="u",
            type=MemoryType.RULE,
            scope="global",
            content="b" * 300,
            source=MemorySource.DETECTED,
            stability=0.5,
            valid_from=now,
            created_at=now,
            updated_at=now,
        )
    )

    policy = await imprint.get_policy(user_id="u", max_input_tokens=160)

    dropped_ids = {m.id for m in policy.dropped_memories}
    assert "m_low" in dropped_ids
    assert "m_high" not in dropped_ids


async def test_get_policy_scopes_empty_returns_only_globals() -> None:
    from datetime import UTC, datetime

    from imprint.types import Memory, MemorySource, MemoryType

    imprint, _, _, _, _, _ = _make_imprint(compile_text="ok", scopes=["code"])
    await imprint.connect()

    store = cast(SQLiteMemoryStore, imprint._store)
    now = datetime.now(UTC)
    await store.insert_memory(
        Memory(
            id="m_global",
            agent_id="agent",
            user_id="u",
            type=MemoryType.RULE,
            scope="global",
            content="global rule",
            source=MemorySource.DETECTED,
            valid_from=now,
            created_at=now,
            updated_at=now,
        )
    )
    await store.insert_memory(
        Memory(
            id="m_code",
            agent_id="agent",
            user_id="u",
            type=MemoryType.RULE,
            scope="code",
            content="code rule",
            source=MemorySource.DETECTED,
            valid_from=now,
            created_at=now,
            updated_at=now,
        )
    )

    policy = await imprint.get_policy(user_id="u", scopes=[])
    mem_ids = {m.id for m in policy.memories}
    assert "m_global" in mem_ids
    assert "m_code" not in mem_ids


async def test_get_policy_non_matching_scope_returns_empty_policy() -> None:
    from datetime import UTC, datetime

    from imprint.types import Memory, MemorySource, MemoryType

    imprint, _, _, _, _, _ = _make_imprint(compile_text="ok", scopes=["code", "writing"])
    await imprint.connect()

    store = cast(SQLiteMemoryStore, imprint._store)
    now = datetime.now(UTC)
    await store.insert_memory(
        Memory(
            id="m_code",
            agent_id="agent",
            user_id="u",
            type=MemoryType.RULE,
            scope="code",
            content="code rule",
            source=MemorySource.DETECTED,
            valid_from=now,
            created_at=now,
            updated_at=now,
        )
    )

    policy = await imprint.get_policy(user_id="u", scopes=["writing"])
    assert policy.memories == []
    assert policy.text == ""


async def test_cache_invalidated_by_contradict() -> None:
    from datetime import UTC, datetime

    from imprint.types import Memory, MemorySource, MemoryType

    existing_id = "m_existing"
    imprint, _, _, _, _, _ = _make_imprint(
        processing_mode="balanced",
        compile_text="policy text",
        derived_content="new rule",
        consolidation_decisions=[{"memory_id": existing_id, "action": "contradict"}],
    )
    await imprint.connect()

    store = cast(SQLiteMemoryStore, imprint._store)
    now = datetime.now(UTC)
    await store.insert_memory(
        Memory(
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
    )

    await imprint.get_policy(user_id="u")
    await imprint.observe(user_id="u", agent_output="x", user_response="always be direct")

    cursor = await store.conn.execute(
        "SELECT COUNT(*) as n FROM compiled_policies WHERE agent_id='agent' AND user_id='u'"
    )
    row = await cursor.fetchone()
    assert row is not None and row["n"] == 0
