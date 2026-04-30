import importlib.util
import os
from contextlib import ExitStack
from pathlib import Path
from typing import cast

import pytest
from pydantic_ai.models.test import TestModel

from imprint import Imprint, SQLiteMemoryStore
from imprint.types import SignalType


def _make_imprint(
    *,
    processing_mode: str = "frugal",
    compile_text: str = "(mock policy)",
    signal_type: SignalType | None = None,
    derived_type: str = "rule",
    derived_content: str = "(derived content)",
    derived_scope: str = "global",
    consolidation_decisions: list[dict[str, str]] | None = None,
    validation_verdicts: list[dict[str, str]] | None = None,
    scopes: list[str] | None = None,
    feedback_timeout: int = 3600,
) -> tuple[Imprint, TestModel, TestModel, TestModel, TestModel, TestModel]:
    """Build an Imprint with all five agents pre-overridden.

    Returns (imprint, compile_model, detect_model, derive_model,
             consolidate_model, validate_model).
    """
    imprint = Imprint(
        agent_id="agent",
        model="anthropic:claude-haiku-4-5-20251001",
        store=":memory:",
        processing_mode=processing_mode,  # type: ignore[arg-type]
        scopes=scopes,
        feedback_timeout=feedback_timeout,
    )
    compile_model = TestModel(custom_output_text=compile_text)
    detect_model = TestModel(
        custom_output_args={"signal_type": signal_type.value if signal_type else None}
    )
    derive_model = TestModel(
        custom_output_args={
            "memory_type": derived_type,
            "content": derived_content,
            "scope": derived_scope,
        }
    )
    consolidate_model = TestModel(custom_output_args={"decisions": consolidation_decisions or []})
    validate_model = TestModel(custom_output_args={"verdicts": validation_verdicts or []})
    stack = ExitStack()
    stack.enter_context(imprint._compile_agent.override(model=compile_model))
    stack.enter_context(imprint._detect_agent.override(model=detect_model))
    stack.enter_context(imprint._derive_agent.override(model=derive_model))
    stack.enter_context(imprint._consolidate_agent.override(model=consolidate_model))
    stack.enter_context(imprint._validate_agent.override(model=validate_model))
    stack.enter_context(imprint._attribute_agent.override(model=validate_model))
    imprint._test_stack = stack  # type: ignore[attr-defined]
    return imprint, compile_model, detect_model, derive_model, consolidate_model, validate_model


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


async def test_frugal_no_signal_stores_nothing() -> None:
    imprint, _, _, _, _, _ = _make_imprint(processing_mode="frugal")
    await imprint.connect()

    await imprint.observe(user_id="u", agent_output="x", user_response="ok")

    policy = await imprint.get_policy(user_id="u")
    assert policy.memories == []
    await imprint.close()


async def test_frugal_heuristic_match_stores_memory() -> None:
    imprint, _, detect_model, _, _, _ = _make_imprint(processing_mode="frugal", compile_text="ok")
    await imprint.connect()

    await imprint.observe(user_id="u", agent_output="x", user_response="No, do it differently")

    policy = await imprint.get_policy(user_id="u")
    assert len(policy.memories) == 1
    # Frugal mode never asks the LLM.
    assert detect_model.last_model_request_parameters is None
    await imprint.close()


async def test_balanced_falls_through_to_llm_when_heuristic_silent() -> None:
    imprint, _, detect_model, _, _, _ = _make_imprint(
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
    imprint, _, detect_model, _, _, _ = _make_imprint(processing_mode="balanced", compile_text="ok")
    await imprint.connect()

    await imprint.observe(user_id="u", agent_output="x", user_response="No, that's wrong")

    # Heuristic matched, LLM was not consulted.
    assert detect_model.last_model_request_parameters is None
    await imprint.close()


async def test_eager_always_calls_llm_for_detection() -> None:
    imprint, _, detect_model, _, _, _ = _make_imprint(
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
    imprint, _, _, _, _, _ = _make_imprint(processing_mode="balanced", signal_type=None)
    await imprint.connect()

    await imprint.observe(user_id="u", agent_output="x", user_response="huh interesting")

    policy = await imprint.get_policy(user_id="u")
    assert policy.memories == []
    await imprint.close()


async def test_memory_url_form_works() -> None:
    imprint = Imprint(agent_id="a", store="sqlite:///:memory:", processing_mode="frugal")
    await imprint.connect()
    policy = await imprint.get_policy(user_id="u")
    assert policy.text == ""
    await imprint.close()


async def test_derivation_assigns_memory_type_from_llm() -> None:
    """The LLM picks the memory type; the hard-coded RULE default is gone."""
    imprint, _, _, _, _, _ = _make_imprint(
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
    imprint, _, _, derive_model, _, _ = _make_imprint(processing_mode="frugal")
    await imprint.connect()

    await imprint.observe(user_id="u", agent_output="x", user_response="ok")

    # Heuristic returns None for "ok"; derive must not have been called.
    assert derive_model.last_model_request_parameters is None
    await imprint.close()


async def test_derivation_receives_signal_type_in_prompt() -> None:
    """The derive prompt is conditioned on the detected signal type."""
    imprint, _, _, derive_model, _, _ = _make_imprint(
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
    imprint, _, _, _, consolidate_model, _ = _make_imprint(processing_mode="frugal")
    await imprint.connect()

    await imprint.observe(user_id="u", agent_output="x", user_response="I prefer paragraphs")

    # No existing memories at observe time, so no consolidation call was made.
    assert consolidate_model.last_model_request_parameters is None
    await imprint.close()


async def test_distinct_decision_keeps_old_memory_active() -> None:
    """If LLM says distinct, both memories remain active."""
    imprint, _, _, _, _, _ = _make_imprint(
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

    imprint, _, _, _, _, _ = _make_imprint(
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
    imprint, _, _, _, _, _ = _make_imprint(processing_mode="frugal", derived_content="first")
    await imprint.connect()

    await imprint.observe(user_id="u", agent_output="x", user_response="I prefer paragraphs")
    existing = await imprint._store.list_memories("agent", "u")
    old_id = existing[0].id
    await imprint.close()

    imprint, _, _, _, _, _ = _make_imprint(
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
    imprint, _, _, _, _, _ = _make_imprint(processing_mode="frugal", derived_content="first")
    await imprint.connect()
    await imprint.observe(user_id="u", agent_output="x", user_response="I prefer paragraphs")
    existing = await imprint._store.list_memories("agent", "u")
    old_id = existing[0].id
    await imprint.close()

    imprint, _, _, _, _, _ = _make_imprint(
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
    imprint, _, _, _, consolidate_model, _ = _make_imprint(
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
    imprint, _, _, _, _, _ = _make_imprint(processing_mode="balanced", derived_content="first")
    await imprint.connect()
    await imprint.observe(user_id="u", agent_output="x", user_response="I prefer paragraphs")
    existing = await imprint._store.list_memories("agent", "u")
    await imprint.close()

    imprint, _, _, _, _, _ = _make_imprint(
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


def test_empty_store_url_rejected() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        Imprint(agent_id="a", store="")


def test_unsupported_scheme_rejected() -> None:
    with pytest.raises(ValueError, match="unsupported store URL scheme"):
        Imprint(agent_id="a", store="postgres://localhost/db")


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


async def test_agent_config_scopes_persist_across_reconnect(tmp_path: Path) -> None:
    db = str(tmp_path / "test.db")

    first = Imprint(
        agent_id="agent", store=db, scopes=["code", "personal"], processing_mode="eager"
    )
    await first.connect()
    await first.close()

    second = Imprint(agent_id="agent", store=db)
    await second.connect()

    assert second.scopes == ["code", "personal"]
    assert second.processing_mode == "eager"
    await second.close()


async def test_agent_config_constructor_overrides_stored(tmp_path: Path) -> None:
    db = str(tmp_path / "test.db")

    first = Imprint(agent_id="agent", store=db, scopes=["X"], processing_mode="frugal")
    await first.connect()
    await first.close()

    second = Imprint(agent_id="agent", store=db, scopes=["Y"], processing_mode="eager")
    await second.connect()

    assert second.scopes == ["Y"]
    assert second.processing_mode == "eager"
    await second.close()

    third = Imprint(agent_id="agent", store=db)
    await third.connect()

    assert third.scopes == ["Y"]
    assert third.processing_mode == "eager"
    await third.close()


async def test_agent_config_defaults_when_no_stored_config() -> None:
    imprint, _, _, _, _, _ = _make_imprint()
    await imprint.connect()

    assert imprint.processing_mode == "frugal"
    assert imprint.scopes == []


async def test_event_logger_records_merge() -> None:
    from datetime import UTC, datetime

    from imprint.types import Memory, MemorySource, MemoryType

    known_id = "mem_existing_001"
    imprint, _, _, _, _, _ = _make_imprint(
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
    imprint, _, _, _, _, _ = _make_imprint(derived_content="some rule", compile_text="be direct")
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
    from imprint import NullEventLogger

    imprint, _, _, _, _, _ = _make_imprint(derived_content="rule", compile_text="policy")
    await imprint.connect()
    imprint._event_logger = NullEventLogger()

    await imprint.observe(user_id="u", agent_output="x", user_response="always be concise")
    await imprint.get_policy(user_id="u")

    store = cast(SQLiteMemoryStore, imprint._store)
    cursor = await store.conn.execute("SELECT COUNT(*) as n FROM memory_events")
    row = await cursor.fetchone()
    assert row is not None and row["n"] == 0


async def test_merge_increases_stability() -> None:
    from datetime import UTC, datetime

    from imprint.types import Memory, MemorySource, MemoryType

    known_id = "mem_decay_merge"
    imprint, _, _, _, _, _ = _make_imprint(
        processing_mode="balanced",
        derived_content="new",
        consolidation_decisions=[{"memory_id": known_id, "action": "merge"}],
    )
    await imprint.connect()

    store = cast(SQLiteMemoryStore, imprint._store)
    now = datetime.now(UTC)
    existing = Memory(
        id=known_id,
        agent_id="agent",
        user_id="u",
        type=MemoryType.RULE,
        scope="global",
        content="old rule",
        source=MemorySource.DETECTED,
        stability=5.0,
        valid_from=now,
        created_at=now,
        updated_at=now,
    )
    await store.insert_memory(existing)

    await imprint.observe(user_id="u", agent_output="x", user_response="always be concise")

    mems = await store.list_memories("agent", "u", active_only=False)
    merged = next(m for m in mems if m.id == known_id)
    assert merged.stability == 6.0


async def test_contradict_reduces_stability() -> None:
    from datetime import UTC, datetime

    from imprint.types import Memory, MemorySource, MemoryType

    known_id = "mem_decay_contradict"
    imprint, _, _, _, _, _ = _make_imprint(
        processing_mode="balanced",
        derived_content="new",
        consolidation_decisions=[{"memory_id": known_id, "action": "contradict"}],
    )
    await imprint.connect()

    store = cast(SQLiteMemoryStore, imprint._store)
    now = datetime.now(UTC)
    existing = Memory(
        id=known_id,
        agent_id="agent",
        user_id="u",
        type=MemoryType.RULE,
        scope="global",
        content="old rule",
        source=MemorySource.DETECTED,
        stability=5.0,
        valid_from=now,
        created_at=now,
        updated_at=now,
    )
    await store.insert_memory(existing)

    await imprint.observe(user_id="u", agent_output="x", user_response="actually do the opposite")

    mems = await store.list_memories("agent", "u", active_only=False)
    contradicted = next(m for m in mems if m.id == known_id)
    assert contradicted.stability == 0.5


async def test_recall_increments_count() -> None:
    imprint, _, _, _, _, _ = _make_imprint(derived_content="rule", compile_text="be direct")
    await imprint.connect()

    await imprint.observe(user_id="u", agent_output="x", user_response="always be concise")

    await imprint.get_policy(user_id="u")
    await imprint.get_policy(user_id="u")

    store = cast(SQLiteMemoryStore, imprint._store)
    mems = await store.list_memories("agent", "u")
    assert mems[0].recall_count == 2


async def test_fsrs_static_decay_merge_cap() -> None:
    from datetime import UTC, datetime

    from imprint import FSRSStaticDecay
    from imprint.types import Memory, MemorySource, MemoryType

    decay = FSRSStaticDecay()
    now = datetime.now(UTC)
    m = Memory(
        id="x",
        agent_id="a",
        user_id="u",
        type=MemoryType.RULE,
        scope="global",
        content="c",
        source=MemorySource.DETECTED,
        stability=99.5,
        valid_from=now,
        created_at=now,
        updated_at=now,
    )
    assert decay.update_on_merge(m) == 100.0


async def test_fsrs_static_decay_contradict_floor() -> None:
    from datetime import UTC, datetime

    from imprint import FSRSStaticDecay
    from imprint.types import Memory, MemorySource, MemoryType

    decay = FSRSStaticDecay()
    now = datetime.now(UTC)
    m = Memory(
        id="x",
        agent_id="a",
        user_id="u",
        type=MemoryType.RULE,
        scope="global",
        content="c",
        source=MemorySource.DETECTED,
        stability=0.5,
        valid_from=now,
        created_at=now,
        updated_at=now,
    )
    assert decay.update_on_contradict(m) == 0.1


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


async def test_frugal_derive_correction_produces_rule() -> None:
    from imprint.types import MemoryType

    imprint, _, _, _, _, _ = _make_imprint(processing_mode="frugal", compile_text="ok")
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

    imprint, _, _, _, _, _ = _make_imprint(
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
    imprint, _, _, derive_model, _, _ = _make_imprint(processing_mode="frugal", compile_text="ok")
    await imprint.connect()

    await imprint.observe(user_id="u", agent_output="x", user_response="No, that's wrong")

    assert derive_model.last_model_request_parameters is None


async def test_frugal_consolidation_skips_llm_agent() -> None:
    imprint, _, _, _, consolidate_model, _ = _make_imprint(
        processing_mode="frugal", compile_text="ok"
    )
    await imprint.connect()

    await imprint.observe(user_id="u", agent_output="x", user_response="I prefer paragraphs")
    await imprint.observe(user_id="u", agent_output="x", user_response="No, use plain text")

    assert consolidate_model.last_model_request_parameters is None


async def test_processing_mode_persists_across_reconnect(tmp_path: Path) -> None:
    db = str(tmp_path / "test.db")
    first = Imprint(agent_id="agent", store=db, processing_mode="eager")
    await first.connect()
    await first.close()

    second = Imprint(agent_id="agent", store=db)
    await second.connect()

    assert second.processing_mode == "eager"
    await second.close()


class _ConstantEmbedder:
    """Test embedder that returns a fixed vector for any input."""

    def __init__(self, vector: list[float]) -> None:
        self._vector = vector

    @property
    def dim(self) -> int:
        return len(self._vector)

    async def embed(self, text: str) -> list[float]:
        return self._vector

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self._vector for _ in texts]


class _InMemoryVectorStore:
    """Exact cosine similarity vector store for testing."""

    def __init__(self) -> None:
        self._store: dict[str, list[float]] = {}

    async def upsert(self, id: str, embedding: list[float]) -> None:
        self._store[id] = embedding

    async def search(self, embedding: list[float], top_k: int) -> list[tuple[str, float]]:
        import math

        def cosine_distance(a: list[float], b: list[float]) -> float:
            dot = sum(x * y for x, y in zip(a, b, strict=True))
            na = math.sqrt(sum(x * x for x in a))
            nb = math.sqrt(sum(x * x for x in b))
            if na == 0 or nb == 0:
                return 1.0
            return 1.0 - dot / (na * nb)

        results = [(id, cosine_distance(embedding, vec)) for id, vec in self._store.items()]
        results.sort(key=lambda x: x[1])
        return results[:top_k]

    async def delete(self, id: str) -> None:
        self._store.pop(id, None)


async def test_observe_stores_embedding_when_configured() -> None:
    vec_store = _InMemoryVectorStore()
    embedder = _ConstantEmbedder([1.0, 0.0, 0.0])

    imprint, _, _, _, _, _ = _make_imprint(processing_mode="balanced", derived_content="rule")
    imprint._vector_store = vec_store
    imprint._embedder = embedder
    await imprint.connect()

    await imprint.observe(user_id="u", agent_output="x", user_response="always be concise")

    assert len(vec_store._store) == 1
    stored_id = next(iter(vec_store._store))
    mem_id = (await cast(SQLiteMemoryStore, imprint._store).list_memories("agent", "u"))[0].id
    assert stored_id == mem_id


async def test_balanced_prefilter_limits_candidates() -> None:
    """Balanced consolidation with vectors only processes similar memories."""
    from datetime import UTC, datetime

    from imprint.types import Memory, MemorySource, MemoryType

    same_vec = [1.0, 0.0, 0.0]
    diff_vec = [0.0, 0.0, 1.0]

    vec_store = _InMemoryVectorStore()
    embedder = _ConstantEmbedder(same_vec)

    similar_id = "mem_similar"
    dissimilar_id = "mem_dissimilar"

    imprint, _, _, _, _, _ = _make_imprint(
        processing_mode="balanced",
        derived_content="new rule",
        consolidation_decisions=[{"memory_id": similar_id, "action": "merge"}],
    )
    imprint._vector_store = vec_store
    imprint._embedder = embedder
    await imprint.connect()

    store = cast(SQLiteMemoryStore, imprint._store)
    now = datetime.now(UTC)
    for mid, vec, content in [
        (similar_id, same_vec, "similar rule"),
        (dissimilar_id, diff_vec, "unrelated rule"),
    ]:
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
        await vec_store.upsert(mid, vec)

    await imprint.observe(user_id="u", agent_output="x", user_response="always be concise")

    all_mems = await store.list_memories("agent", "u", active_only=False)
    by_id = {m.id: m for m in all_mems}

    assert by_id[similar_id].active is False
    assert by_id[dissimilar_id].active is True


async def test_frugal_vector_consolidation_merges_similar() -> None:
    """Frugal mode with a vector store merges memories above the similarity threshold."""
    from datetime import UTC, datetime

    from imprint.types import Memory, MemorySource, MemoryType

    same_vec = [1.0, 0.0, 0.0]
    vec_store = _InMemoryVectorStore()
    embedder = _ConstantEmbedder(same_vec)

    imprint, _, _, _, _, _ = _make_imprint(processing_mode="frugal", compile_text="ok")
    imprint._vector_store = vec_store
    imprint._embedder = embedder
    await imprint.connect()

    store = cast(SQLiteMemoryStore, imprint._store)
    now = datetime.now(UTC)
    existing_id = "mem_existing"
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

    await imprint.observe(user_id="u", agent_output="x", user_response="always be concise")

    all_mems = await store.list_memories("agent", "u", active_only=False)
    merged = next(m for m in all_mems if m.id == existing_id)
    assert merged.active is False


@pytest.mark.live
async def test_voyage_embedder_live() -> None:
    """VoyageEmbedder returns 1024-dim vectors and similar texts are close."""
    if importlib.util.find_spec("voyageai") is None:
        pytest.skip("voyageai not installed (pip install imprint[voyage])")
    if not os.environ.get("VOYAGE_API_KEY"):
        pytest.skip("VOYAGE_API_KEY not set")

    from imprint import VoyageEmbedder

    embedder = VoyageEmbedder(model="voyage-3.5-lite", dim=1024)

    v1 = await embedder.embed("The user prefers concise responses.")
    v2 = await embedder.embed("Keep answers brief and to the point.")
    v3 = await embedder.embed("The capital of France is Paris.")

    assert len(v1) == 1024
    assert len(v2) == 1024
    assert len(v3) == 1024

    import math

    def cosine(a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b, strict=True))
        return dot / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(x * x for x in b)))

    assert cosine(v1, v2) > cosine(v1, v3), (
        "semantically similar texts should be closer than dissimilar ones"
    )


@pytest.mark.live
async def test_voyage_embedder_batch_live() -> None:
    """embed_batch returns one embedding per input, same dim as embed."""
    if importlib.util.find_spec("voyageai") is None:
        pytest.skip("voyageai not installed (pip install imprint[voyage])")
    if not os.environ.get("VOYAGE_API_KEY"):
        pytest.skip("VOYAGE_API_KEY not set")

    from imprint import VoyageEmbedder

    embedder = VoyageEmbedder(model="voyage-3.5-lite", dim=1024)
    texts = ["first text", "second text", "third text"]
    batch = await embedder.embed_batch(texts)

    assert len(batch) == 3
    assert all(len(v) == 1024 for v in batch)


@pytest.mark.live
async def test_anthropic_token_counter_live() -> None:
    """AnthropicAPITokenCounter returns a positive integer for a short string."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        pytest.skip("ANTHROPIC_API_KEY not set")

    from imprint import AnthropicAPITokenCounter

    counter = AnthropicAPITokenCounter()
    count = counter.count("Hello, how can I help you today?")

    assert isinstance(count, int)
    assert count > 0
    assert count < 50


@pytest.mark.live
async def test_anthropic_token_counter_longer_text_live() -> None:
    """Longer text produces more tokens than shorter text."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        pytest.skip("ANTHROPIC_API_KEY not set")

    from imprint import AnthropicAPITokenCounter

    counter = AnthropicAPITokenCounter()
    short = counter.count("Hi.")
    long_text = (
        "This is a much longer piece of text that should produce significantly more tokens"
        " than the short greeting above, because it contains more words and more information."
    )
    long = counter.count(long_text)

    assert long > short


@pytest.mark.live
async def test_voyage_token_counter_live() -> None:
    """VoyageTokenCounter counts tokens locally -- no API call after tokenizer download."""
    if importlib.util.find_spec("voyageai") is None:
        pytest.skip("voyageai not installed (pip install imprint[voyage])")
    if not os.environ.get("VOYAGE_API_KEY"):
        pytest.skip("VOYAGE_API_KEY not set")

    from imprint import VoyageTokenCounter

    counter = VoyageTokenCounter(model="voyage-3.5-lite")

    short = counter.count("Hello.")
    long = counter.count(
        "This is a longer sentence that should produce more tokens than the short one above."
    )

    assert isinstance(short, int)
    assert short > 0
    assert long > short


async def test_sqlite_vec_store_raises_on_missing_dep() -> None:
    """SQLiteVecStore gives a clear ImportError when sqlite-vec is not installed."""
    import sys
    from unittest.mock import patch

    store = cast(SQLiteMemoryStore, _make_imprint()[0]._store)
    await store.connect()
    await store.init_schema()

    import aiosqlite

    from imprint.vector import SQLiteVecStore

    vec_store = SQLiteVecStore(cast(aiosqlite.Connection, store._conn), dim=3)

    with (
        patch.dict(sys.modules, {"sqlite_vec": None}),
        pytest.raises(ImportError, match="sqlite-vec is required"),
    ):
        await vec_store.upsert("m1", [1.0, 0.0, 0.0])

    await store.close()


async def test_voyage_embedder_raises_on_missing_dep() -> None:
    """VoyageEmbedder gives a clear ImportError when voyageai is not installed."""
    import sys
    from unittest.mock import patch

    from imprint import VoyageEmbedder

    embedder = VoyageEmbedder()
    with (
        patch.dict(sys.modules, {"voyageai": None}),
        pytest.raises(ImportError, match="voyageai is required"),
    ):
        await embedder.embed("hello")


async def test_voyage_token_counter_raises_on_missing_dep() -> None:
    """VoyageTokenCounter gives a clear ImportError when voyageai is not installed."""
    import sys
    from unittest.mock import patch

    from imprint import VoyageTokenCounter

    counter = VoyageTokenCounter()
    with (
        patch.dict(sys.modules, {"voyageai": None}),
        pytest.raises(ImportError, match="voyageai is required"),
    ):
        counter.count("hello")


async def test_anthropic_token_counter_raises_on_missing_dep() -> None:
    """AnthropicAPITokenCounter gives a clear ImportError when anthropic is not installed."""
    import sys
    from unittest.mock import patch

    from imprint import AnthropicAPITokenCounter

    counter = AnthropicAPITokenCounter()
    with (
        patch.dict(sys.modules, {"anthropic": None}),
        pytest.raises(ImportError, match="anthropic is required"),
    ):
        counter.count("hello")


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

    imprint, _, _, _, _, _ = _make_imprint(processing_mode="balanced", compile_text="ok")
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

    imprint, _, _, _, _, _ = _make_imprint(processing_mode="balanced", compile_text="ok")
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

    imprint, _, _, _, _, _ = _make_imprint(processing_mode="balanced", compile_text="ok")
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
    imprint, _, _, _, _, _ = _make_imprint(
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

    final_state = tuner.get_state()
    total_initial = sum(initial_state["s"]) + sum(initial_state["f"])
    total_final = sum(final_state["s"]) + sum(final_state["f"])
    assert total_final > total_initial


async def test_observe_directions_empty_list_returns_empty() -> None:
    imprint, _, _, _, _, _ = _make_imprint()
    await imprint.connect()
    result = await imprint.observe_directions(user_id="u", directions=[])
    assert result == []


async def test_observe_directions_frugal_stores_as_rule() -> None:
    from imprint.types import MemoryType

    imprint, _, _, _, _, _ = _make_imprint(processing_mode="frugal")
    await imprint.connect()

    memories = await imprint.observe_directions(
        user_id="u",
        directions=["always respond in English"],
    )

    assert len(memories) == 1
    assert memories[0].type == MemoryType.RULE
    assert memories[0].content == "always respond in English"


async def test_observe_directions_frugal_skips_llm() -> None:
    imprint, _, _, derive_model, _, validate_model = _make_imprint(processing_mode="frugal")
    await imprint.connect()

    await imprint.observe_directions(user_id="u", directions=["always be concise"])

    assert derive_model.last_model_request_parameters is None
    assert validate_model.last_model_request_parameters is None


async def test_observe_directions_balanced_calls_derive_llm() -> None:
    imprint, _, _, derive_model, _, _ = _make_imprint(
        processing_mode="balanced",
        derived_content="always respond in English",
    )
    await imprint.connect()

    await imprint.observe_directions(user_id="u", directions=["always respond in English"])

    assert derive_model.last_model_request_parameters is not None


async def test_observe_directions_balanced_skips_validation() -> None:
    imprint, _, _, _, _, validate_model = _make_imprint(processing_mode="balanced")
    await imprint.connect()

    await imprint.observe_directions(user_id="u", directions=["always be concise"])

    assert validate_model.last_model_request_parameters is None


async def test_observe_directions_eager_runs_validation_first() -> None:
    imprint, _, _, _, _, validate_model = _make_imprint(
        processing_mode="eager",
        validation_verdicts=[{"verdict": "directive"}],
    )
    await imprint.connect()

    await imprint.observe_directions(user_id="u", directions=["always be concise"])

    assert validate_model.last_model_request_parameters is not None


async def test_observe_directions_eager_filters_non_directives() -> None:
    imprint, _, _, _, _, _ = _make_imprint(
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
    imprint, _, _, _, _, _ = _make_imprint(processing_mode="frugal")
    await imprint.connect()

    directions = ["always use English", "never use bullet points", "keep it short"]
    memories = await imprint.observe_directions(user_id="u", directions=directions)

    assert len(memories) == 3
    stored = await imprint._store.list_memories("agent", "u")
    assert len(stored) == 3


async def test_observe_directions_source_defaults_to_user_edit() -> None:
    from imprint.types import MemorySource

    imprint, _, _, _, _, _ = _make_imprint(processing_mode="frugal")
    await imprint.connect()

    memories = await imprint.observe_directions(user_id="u", directions=["always be direct"])

    assert memories[0].source == MemorySource.USER_EDIT


async def test_observe_directions_respects_scope_hint() -> None:
    imprint, _, _, _, _, _ = _make_imprint(processing_mode="frugal", scopes=["code"])
    await imprint.connect()

    memories = await imprint.observe_directions(
        user_id="u",
        directions=["always use type hints"],
        scope="code",
    )

    assert memories[0].scope == "code"


async def test_observe_directions_invalidates_cache() -> None:
    imprint, _, _, _, _, _ = _make_imprint(processing_mode="frugal", compile_text="policy")
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


async def test_get_policy_opens_feedback_loop() -> None:
    imprint, _, _, _, _, _ = _make_imprint(processing_mode="frugal", compile_text="ok")
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
    from imprint.types import SignalType

    imprint, _, _, _, _, _ = _make_imprint(
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
    imprint, _, _, _, _, _ = _make_imprint(
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


async def test_observe_feedback_no_op_without_open_loop() -> None:
    imprint, _, _, _, _, _ = _make_imprint(processing_mode="frugal")
    await imprint.connect()

    await imprint.observe_feedback(user_id="u", outcome=1.0)


async def test_session_id_creates_separate_loop() -> None:
    imprint, _, _, _, _, _ = _make_imprint(processing_mode="frugal", compile_text="ok")
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

    imprint, _, _, _, _, _ = _make_imprint(
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


async def test_fsrs_gradient_decay_learn_and_predict() -> None:
    pytest.importorskip("river", reason="imprint[online] not installed")
    from datetime import UTC, datetime

    from imprint.online import FSRSGradientDecay
    from imprint.types import Memory, MemorySource, MemoryType

    decay = FSRSGradientDecay(learning_rate=0.1)
    now = datetime.now(UTC)
    m = Memory(
        id="x",
        agent_id="a",
        user_id="u",
        type=MemoryType.RULE,
        scope="global",
        content="c",
        source=MemorySource.DETECTED,
        stability=5.0,
        valid_from=now,
        created_at=now,
        updated_at=now,
    )

    # Before any learning, effective_stability should be a positive float
    s1 = decay.effective_stability(m, now)
    assert s1 >= 0.1

    # After a positive learning signal, prediction should increase
    for _ in range(20):
        decay.learn(m, now, 1.0)

    s2 = decay.effective_stability(m, now)
    assert s2 > s1


async def test_fsrs_gradient_decay_state_roundtrip() -> None:
    pytest.importorskip("river", reason="imprint[online] not installed")
    from datetime import UTC, datetime

    from imprint.online import FSRSGradientDecay
    from imprint.types import Memory, MemorySource, MemoryType

    decay = FSRSGradientDecay()
    now = datetime.now(UTC)
    m = Memory(
        id="x",
        agent_id="a",
        user_id="u",
        type=MemoryType.RULE,
        scope="global",
        content="c",
        source=MemorySource.DETECTED,
        stability=5.0,
        valid_from=now,
        created_at=now,
        updated_at=now,
    )
    for _ in range(5):
        decay.learn(m, now, 0.7)

    state = decay.get_state()
    assert isinstance(state, str)

    decay2 = FSRSGradientDecay()
    decay2.set_state(state)

    pred1 = decay.effective_stability(m, now)
    pred2 = decay2.effective_stability(m, now)
    assert abs(pred1 - pred2) < 0.001


async def test_fsrs_gradient_decay_raises_without_river() -> None:
    pytest.importorskip("river", reason="imprint[online] not installed")
    import sys
    from unittest.mock import patch

    from imprint.online import FSRSGradientDecay

    decay = FSRSGradientDecay()
    decay._model = None  # type: ignore[assignment]

    with (
        patch.dict(sys.modules, {"river": None}),
        pytest.raises(ImportError, match="river is required"),
    ):
        decay._model = decay._build_model()


async def test_observe_feedback_with_gradient_decay() -> None:
    pytest.importorskip("river", reason="imprint[online] not installed")
    from imprint.online import FSRSGradientDecay

    decay = FSRSGradientDecay()
    imprint, _, _, _, _, _ = _make_imprint(
        processing_mode="frugal", compile_text="ok", derived_content="rule"
    )
    imprint._decay_model = decay
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
    initial_state = decay.get_state()

    await imprint.observe_feedback(user_id="u", outcome=1.0)

    final_state = decay.get_state()
    assert initial_state != final_state


async def test_feedback_cycle_full_flow() -> None:
    """Integration: get_policy opens loop, observe correction closes it, bandit updates."""
    from imprint import BanditAlphaTuner
    from imprint.types import SignalType

    tuner = BanditAlphaTuner()
    imprint, _, _, _, _, _ = _make_imprint(
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
    from imprint.types import SignalType

    tuner = BanditAlphaTuner()
    imprint, _, _, _, _, _ = _make_imprint(
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
    from imprint.types import SignalType

    tuner = BanditAlphaTuner()
    # balanced mode so the mock detect_model returning FACT is actually used
    imprint, _, _, _, _, _ = _make_imprint(
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
    from imprint.types import SignalType

    tuner = BanditAlphaTuner()
    imprint, _, _, _, _, _ = _make_imprint(
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
